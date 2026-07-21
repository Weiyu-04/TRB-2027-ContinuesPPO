"""
TRB 环境 · Phase 3 连续投影盾层（Node A，2026-06-17(b)）
=========================================================
把 Phase 2 的连续投影盾 `ContinuousColregsProjection.safe_action`（usv_projection，档位A 经验性）接进
`USVEnv`（连续动作物理）→ **SAC 可直接训练的连续投影环境**。SAC 输出期望连续动作 u_desired，本层每决策步把
u_desired 投影成【总是可执行】的安全动作 u_safe（投影优先、P=∅ 落兜底）再喂 env.step。

四方对比同口径（D34/D37-B）：Continuous-safe = 本类 + SAC；Discrete-safe = ShieldedUSVEnv + MaskablePPO；
Base/RR = UnshieldedUSVEnv。本类保留 `_ego_vs/_obs_vs`（与 ShieldedUSVEnv 同语义）供 ViolationCounter 同口径计违规。

集成层必做项（03 D38 + Phase 2 复核挂起 D32/D33，全部在此落地）：
  ① **env.reset() → proj.reset()**（防 EmergencyController 跨 episode 静默错动作 = D13 头号风险 / D33 B-EC-CROSS-EPISODE）；
  ② **proj box == 动力学 box**（构造即对齐 self.env.p；safe_action→project_qp 内部再 assert 兜底，复审 B-EMERGENCY-BOX）；
  ③ **每决策步只调一次 safe_action**（别和 project/project_qp 混调 → 状态机 _prev_rho stale，D32/D33）；
  ④ **u_safe 经 env._map_action 做最终 box 限幅**（任何 source 的越界值无条件截进 box；emergency=EC 输出【原则上】可越 box——
     实测 tracking_controller 内部已 clip、常不触发，限幅是 belt-and-suspenders 兜底）+ 端到端 ∈box（u_applied 可核）；
  ⑤ **info 带 source/rho/give_way_dir/emergency_mode** → 评估按 source 归类违规/紧急步%（同 D34 对比表口径、Node C 用）。

时序（与 ShieldedUSVEnv 等价喂状态机）：safe_action 在 env.step **之前**用当前态(ego@t, obs@t) 推状态机一步；
  首 step 用 state 0、之后 1,2,…（连续序列，persistent 谓词正确）。s_obs=None（窗外）→ 短路不投影、不推状态机（同 ShieldedUSVEnv）。

⚠️ 当前 = **档位A 经验性**（一步前瞻 + d_safe 大裕度、非 provable）；provable = 档位B = Phase 4 不变集（D5/D28）。
⚠️ 本层 = **Node A 集成骨架（纯接线 + 冒烟，不训练）**；SAC 训练 setup（同款 VecNormalize / 丢 r_colregs / SAC 自带探索别搬
   离散 ent_coef）= Node B、四方对比 harness = Node C、多前瞻 τ + action aliasing 惩罚 = Node D（03 D38）。
⚠️ SAC 连续训练**不用 action_masks**（那是 MaskablePPO/离散）；连续 eval 路径（SAC.predict→连续动作→env.step）= Node C，
   现有 evaluate.run_episode 是离散专用（int 动作+mask），本类仅暴露 Node C 所需（_ego_vs/_obs_vs/info[source,rho]）。
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from .usv_colregs import RHO_NAMES, RHO_NO_CONFLICT, VesselState
from .usv_env import (A_NORMAL_ACCEL_MAX, A_NORMAL_OMEGA_MAX, C_REACH, V_LOW, USVEnv,
                      assert_single_obstacle)
from .usv_projection import ContinuousColregsProjection

_N_RHO_STATES = len(RHO_NAMES)   # 🆕 腿1（`03` L150/L152）：态势感知 ρ one-hot 维数 = ρ 状态数（6·NO_CONFLICT/STAND_ON/HEAD_ON/CROSSING/OVERTAKE/EMERGENCY）·单一真相源 RHO_NAMES 防漂


class ContinuousProjectionEnv(gym.Env):
    """连续投影盾环境（Node A）：USVEnv(连续) + ContinuousColregsProjection.safe_action → SAC 可训练。

    用法（Phase 3）：
        env = ContinuousProjectionEnv(scenario, planning_problem)
        obs, info = env.reset(seed=0)
        obs, r, term, trunc, info = env.step([a, ω])   # SAC 期望动作 → 投影 → 安全动作执行
        # info["source"] ∈ {projection, emergency, relaxed, collision_min, degenerate, no_obstacle}
    """

    metadata = {"render_modes": []}

    def __init__(self, scenario, planning_problem, *, clip_velocity: bool = True,
                 shield: bool = True,   # 🆕 P0(L146)：SE-RL 盾开关。True(默认)=有盾=逐位等价现状 bit-identical；False=【连续无盾臂】(施 RL 原动作·口径对齐 UnshieldedUSVEnv:rho 恒 NO_CONFLICT/无状态机/无紧急)·解耦"盾诱发崩塌"混淆 + why-RL C 臂。
                 augment_rho: bool = False,   # 🆕 腿1(L150/L152)：态势感知观测增广。False(默认)=obs 27维=bit-identical；True=obs 尾拼 ρ one-hot(6)+give_way_dir(1)=34维（让策略看见此刻态势→治抖/治违规·非治崩[15-35%·L150]）。仅连续臂·内层 USVEnv 27维一字不动=离散臂忠实。无盾臂(shield=False)self._rho 恒 NO_CONFLICT=常数零信息→run_step4e fail-fast 挡（离散臂 obs 27维·四方须独立 TAG 绘图层合并）。
                 colregs_weight: float = 0.0,
                 gamma: float = 0.99, well_shaping_weight: float = 0.0, shaping_radius: float = 500.0,
                 xtrack_weight: float = 0.0, xtrack_radius: float = 80.0,   # 对症 横向进带势（`03` L88·显式具名在 **proj_kwargs 前·防误路由）
                 park_weight: float = 0.0, park_radius: float = 400.0, park_v_target: float = 4.0,   # 想法B 终端保速势（`03` L109·连续臂专属·PBRS 透传 RewardFunction）
                 c_step: float = 0.0,   # 修法C 每步生存成本（`03` L123·连续臂专属·非PBRS 真改最优·透传 RewardFunction·默认 0.0=逐位等价）
                 c_dwell: float = 0.0, w_dwell: float = 90.0, h_dwell: float = 0.52, dwell_radius: float = 250.0, b_dwell: float = 0.0,   # r_dwell 入库赤字滞留成本（`03` L161/L162·连续臂专属·非PBRS·默认关=逐位等价·透传 USVEnv→RewardFunction）
                 c_reach: float = C_REACH, dock_radius: float = 0.0, v_dock: float = V_LOW,   # 第二条腿修法（`03` L172·连续臂专属·显式具名在 **proj_kwargs 前·防误路由·默认关=逐位等价·透传 USVEnv→RewardFunction）：c_reach 重标 r_goal / dock_radius+v_dock 泊位精修门降速度地板
                 arrival_heading_slack: float = 0.0,   # 🆕 B1(`03` L153)：到达门朝向容差课程 slack（透传内层 USVEnv→term_checker）。默认 0.0=真门=bit-identical；>0=训练放宽终端朝向门（退火·评估恒 0=诚实红线）。
                 goal_ignore_orientation: bool = False,   # 🆕 L185(user 2026-07-13)：训练目标去朝向硬门→1_goal 只判位置到达区域（透传内层 USVEnv→term_checker）。默认 False=严格真门=bit-identical；True=位置-only（治崩种子被朝向门逼出的绕圈·两阶段stage-1）。
                 start_frac: float = 1.0, start_v=None,   # 🆕 逆向起点课程（方案C-B·`03` L181·Florensa 2017）：透传内层 USVEnv.set_start_frac。默认 (1.0,None)=真起点=bit-identical；<1=训练时 ego 生到更靠泊位门（f→0贴门·教终端捕获）·评估恒 1.0=真起点诚实红线。start_v=课程重生速度(None=真init速度)。
                 alias_weight: float = 0.0,   # 动作混叠惩罚 w（Markgraf 2026 式20·`03` L97·默认 0.0=关=与现状逐位等价 bit-identical）
                 rate_weight: float = 0.0,    # action-rate 平滑惩罚 w（治 bang-bang 抖动·`03` L98·默认 0.0=关=逐位等价）
                 rate_dock: float | None = None,   # 🆕 第二条腿 rank1（`03` L173）：泊位精修门控【治抖 r_rate】。默认 None=off=bit-identical；设值(如 0/0.1)=船进泊位区(‖ego−goal‖≤dock_radius)时把 rate_weight 降到 rate_dock=放行入库急打舵对齐窄朝向门（治"接近后朝向捕获失败/高速绕圈"·L171 点名却在 L172 漏做的那半）·复用 dock_radius 不新增区半径旋钮·连续臂专属。
                 # ⚠️ alias_weight/rate_weight 现可被【惩罚退火】在训练中改写（`03` L103·MultiScenarioEnv.set_penalty_weight→setattr 本实例·施加逻辑 step() 不变·每步读 self.* 当前值）。
                 # ⚠️ 子类化口径：alias/rate 二者都是【实例属性·step() 每步读 self.alias_weight/self.rate_weight】→退火 setattr 即时生效；本类构造逻辑/施加公式一字未因退火改动（保 bit-identical）。
                 **proj_kwargs):
        """
        colregs_weight : 透传 USVEnv→RewardFunction。⚠️ **默认 0.0**（=本类唯一正确用途 Continuous-safe：合规靠投影约束(档位A 经验性·非档位B provable 硬保证)、
                         非软奖励，D37-B；与离散 RR/Safe 口径差须写作声明）。**默认设 0.0 使误差走安全侧**（D39/L44 footgun 修复，
                         2026-06-17b：原默认 1.0 与该类唯一用途冲突=四方钱图静默污染风险；离散侧靠 make_base/rule maker 显式强制
                         0/1、连续侧此前裸吃默认=系统性非对称）。Node B SAC 入口仍应硬编 0.0 + 断言防回归。
        proj_kwargs    : 透传 ContinuousColregsProjection（omega_turn/eps_omega/eps_a/statechart）。
        """
        super().__init__()
        self.env = USVEnv(scenario, planning_problem, continuous=True,
                          clip_velocity=clip_velocity, colregs_weight=colregs_weight,
                          gamma=gamma, well_shaping_weight=well_shaping_weight, shaping_radius=shaping_radius,
                          xtrack_weight=xtrack_weight, xtrack_radius=xtrack_radius,
                          park_weight=park_weight, park_radius=park_radius, park_v_target=park_v_target,   # 想法B 终端保速势透传（`03` L109）
                          c_step=c_step,   # 修法C 每步生存成本透传（`03` L123·连续臂专属）
                          c_dwell=c_dwell, w_dwell=w_dwell, h_dwell=h_dwell, dwell_radius=dwell_radius, b_dwell=b_dwell,   # r_dwell 入库赤字滞留成本透传（`03` L161/L162·连续臂专属·非PBRS）
                          c_reach=c_reach, dock_radius=dock_radius, v_dock=v_dock,   # 第二条腿修法透传（`03` L172·连续臂专属·默认关 bit-identical）
                          arrival_heading_slack=arrival_heading_slack,   # 🆕 B1：到达门朝向容差 slack 透传内层 USVEnv→term_checker（`03` L153）
                          goal_ignore_orientation=goal_ignore_orientation)   # 🆕 L185：去朝向硬门透传内层 USVEnv→term_checker（默认 False=严格真门=bit-identical）
        self.env.set_start_frac(start_frac, start_v)   # 🆕 逆向起点课程（`03` L181）：默认 (1.0,None)=no-op=bit-identical；<1=ego 生到更靠门。透传内层 USVEnv。
        # ② proj box 对齐动力学 box（safe_action→project_qp 内部再 assert 兜底）
        # proj_kwargs 透传 goal_cone_half/goal_v_floor（统一态势盾·ρ0 朝目标锥安全集·`03` 方案①）；默认 None=关=逐位等价现状。
        self.proj = ContinuousColregsProjection(self.env.p.a_max, self.env.p.w_max, **proj_kwargs)
        # 注入目标点（ρ0 朝目标锥安全集用）：锥关（goal_cone_half=None）时 goal 不被读取、无副作用。
        if hasattr(self.env, "goal_center"):
            self.proj.set_goal(self.env.goal_center)
        self.shield = bool(shield)   # 🆕 P0：False=连续无盾臂（step 施 RL 原动作·不投影/不锥/不推状态机）
        # 动作混叠惩罚（Markgraf 2026 式20 h=w‖u−uφ‖²·`03` L97·治"策略↔盾分歧塑偏接近段航迹"=反事实证 42% 残余主因）
        self.alias_weight = float(alias_weight)
        if not (np.isfinite(self.alias_weight) and self.alias_weight >= 0.0):   # 复审 MINOR：nan/inf<0 均 False 会漏过→r_alias 变 nan 静默毒化奖励·须 isfinite 双守卫
            raise ValueError(f"alias_weight 必须是有限非负数，得 {self.alias_weight}")
        # action-rate 平滑惩罚（治 bang-bang 满舵交替=策略原始动作逐步猛变·`03` L98·罚 ‖Δu_desired‖²·标准控制平滑技术·非 Markgraf 式20[那是空间分歧·此为时间抖动]）
        self.rate_weight = float(rate_weight)
        if not (np.isfinite(self.rate_weight) and self.rate_weight >= 0.0):
            raise ValueError(f"rate_weight 必须是有限非负数，得 {self.rate_weight}")
        # 🆕 第二条腿 rank1（`03` L173）：泊位精修门控 r_rate。复用 dock_radius 作区半径（须存本类·内层 USVEnv 已收 dock_radius 用于 r_velocity 门·此处独立用于 r_rate 门）。
        self._dock_radius = float(dock_radius)                  # 泊位精修区半径（与 r_velocity 门共用同一区·rank1 门控 r_rate 用）
        self._rate_dock = None if rate_dock is None else float(rate_dock)   # 区内治抖罚降到的值（None=off=bit-identical；0=区内完全免治抖罚·>0 小地板=保部分平滑防门口颤振）
        if self._rate_dock is not None:
            if not (np.isfinite(self._rate_dock) and self._rate_dock >= 0.0):
                raise ValueError(f"rate_dock 须有限非负（区内治抖罚权重），得 {self._rate_dock}")
            if self._dock_radius <= 0.0:                        # 复用 dock_radius 作区半径→rate_dock 无区可门=silent no-op·fail-fast（同 v_dock 需 dock_radius>0 逻辑）
                raise ValueError(f"rate_dock={self._rate_dock} 需 dock_radius>0（泊位区半径·复用 r_velocity 门同区）·得 dock_radius={self._dock_radius}")
        self._prev_u_desired: np.ndarray | None = None         # 上一步策略原始动作（reset 清·首步无罚·仅 rate_weight>0 时追踪=保 bit-identical）
        # RL 动作箱 = Krasowski【正常操作】±0.048/±0.018（非内层 USVEnv 的物理满程 ±a_max/±w_max）。
        # 根因修（03 L63 Fix②）：SAC 在【正常操作】范围探索（=离散臂动作权限同款），消停船墙〔满程 ±0.24
        #   是离散网格 5×→随机探索 v→0 撞吸收壁早死〕+ 修四方公平 confound（03 L1a-4）。盾/EC/动力学仍用物理 ±a_max：
        #   u_safe 经 env._map_action clip 到物理 box（正常操作 ⊂ 物理 → 对 RL 动作恒 no-op）；EC/proj 用物理满程不受影响。
        self.action_space = gym.spaces.Box(
            low=np.array([-A_NORMAL_ACCEL_MAX, -A_NORMAL_OMEGA_MAX], dtype=np.float32),
            high=np.array([A_NORMAL_ACCEL_MAX, A_NORMAL_OMEGA_MAX], dtype=np.float32),
            dtype=np.float32)
        # 动作混叠惩罚的归一化尺度 = RL 动作箱（防尺度陷阱：原始 ‖u−uφ‖²~1e-2 会淹没 Markgraf w∈{0.1..2}·`03` L97；
        # 归一化后 w∈{0.1..2} 才有意义·类比 r_colregs 归一化口径）。
        # ⚠️ 尺度界（复审 math MINOR 修正）：u_safe 由 project_qp 在【物理箱 ±a_max/±w_max】上解（COLREGs give-way 可要求大幅右转/减速、越 RL 正常操作箱），
        #   故 (u−u_safe)/u_box 每分量可 >2、‖·‖² 理论上界 ≈ Σ(2·物理/RL)² = (2·0.24/0.048)²+(2·0.03/0.018)² ≈ 111（w=2 最坏 r_alias≈−222）；
        #   但实测分布集中在 ‖·‖²<20（典型 give-way 步 r_alias mean≈−0.05/max≈−1.5·远未触 VecNorm clip）。标定 w 须按【实测 r_alias 直方图】非[0,8]假设。
        self._u_box = np.array([A_NORMAL_ACCEL_MAX, A_NORMAL_OMEGA_MAX], dtype=float)
        # 🆕 腿1（`03` L150/L152）：态势感知观测增广。默认关 → = 内层 Box(27) 逐位等价 bit-identical；开 → 27 ‖ ρ one-hot(6) ‖ give_way_dir(1) = Box(34)。
        #   ⚠️ observation_space 必须在此按 augment_rho 定死（VecNormalize/SB3 policy 网都据它派形状·L151 抓的最高危坑：若留 27 而 obs 发 34→VecNorm broadcast/policy Linear 首步响崩·非静默）。
        self._augment_rho = bool(augment_rho)
        _base_space = self.env.observation_space             # Box(27,)（内层 USVEnv 一字不动=离散臂忠实）
        if self._augment_rho:
            _lo = np.concatenate([_base_space.low,  np.zeros(_N_RHO_STATES), np.array([-1.0])]).astype(_base_space.dtype)
            _hi = np.concatenate([_base_space.high, np.ones(_N_RHO_STATES),  np.array([ 1.0])]).astype(_base_space.dtype)
            self.observation_space = gym.spaces.Box(low=_lo, high=_hi, dtype=_base_space.dtype)   # Box(34,)
        else:
            self.observation_space = _base_space             # Box(27,) = bit-identical
        self._ego_length = float(self.env.p.l)               # 本船 SR108 = 175
        self._obstacles = self.env.obstacles                 # 单他船（assumption 2）
        assert_single_obstacle(self._obstacles, type(self).__name__)   # D40#4/L49#2：多他船 fail-fast（盾只护 obstacles[0]）
        self._obs_length = (float(self._obstacles[0].obstacle_shape.length)
                            if self._obstacles else self._ego_length)   # 真实他船长（同 ShieldedUSVEnv 决定 4）
        self._rho = RHO_NO_CONFLICT
        self._source: str | None = None

    def set_arrival_slack(self, slack: float):
        """🆕 B1（`03` L153）：转发到达门朝向容差 slack 到内层 USVEnv→term_checker（转发链中间层·
        MultiScenarioEnv.set_arrival_slack 经 VecEnv.env_method 调本方法·同 set_penalty_weight 退火范式，但目标深一层）。
        slack=0（默认/评估）→ 真门 bit-identical；slack>0（仅训练）→ 放宽终端朝向门。本层【纯委托、不持 slack 状态】（真相在内层 term_checker）。"""
        self.env.set_arrival_slack(slack)

    def set_start_frac(self, frac: float, v=None):
        """🆕 逆向起点课程（`03` L181·Florensa 2017）：转发起点系数到内层 USVEnv（转发链中间层·MultiScenarioEnv.set_start_frac 经 env_method 调本方法）。
        frac=1.0（默认/评估）→ 真起点 bit-identical；frac<1（仅训练）→ ego 生到更靠门。本层【纯委托、不持状态】（真相在内层 USVEnv.start_frac）。"""
        self.env.set_start_frac(frac, v)

    # ---- 🆕 腿1：态势感知观测增广（`03` L150/L152）----
    def _augment(self, obs, rho, give_way_dir):
        """把 shield 层态势 ρ + 让路方向拼进 obs 尾部（仅 augment_rho=True）。
        augment_rho=False（默认）→ 原样返回内层 27 维 = bit-identical。开 → obs27 ‖ one_hot(ρ,6) ‖ give_way_dir = 34 维。
        🔑 索引映射（下游/A关守卫按此·L151 真静默变体=错位 → 守卫须断言此映射非仅 shape）：
           obs[27+k] = 1{ρ==k}（k = RHO_* 枚举值 0-5；NO_CONFLICT=0 → 置 bit0 = **非全零**·reset 首帧同此，防"零填≠NO_CONFLICT"静默错）；
           obs[33]   = give_way_dir ∈ {left:−1, none:0, right:+1}·映射从【Python None】取 0（proj 产出 'left'/'right'/None·从不吐 'none' 字符串→用 .get(...,0) 防 KeyError）。
        dtype 对齐 obs（=observation_space.dtype·同 VecNormalize 归一化口径）。ρ@t 语义：调用方传 self._rho=盾这步实际据以动作的 pre-step ρ（1 步滞后·同 obs 内 last_action 口径·不重推状态机）。
        """
        if not self._augment_rho:
            return obs
        onehot = np.zeros(_N_RHO_STATES, dtype=obs.dtype)
        onehot[int(rho)] = 1                                  # ρ∈[0,5]（状态机产出）·越界→IndexError 响错（非静默）
        gw = np.array([{"left": -1.0, "right": 1.0}.get(give_way_dir, 0.0)], dtype=obs.dtype)
        out = np.concatenate([obs, onehot, gw])
        assert out.shape == self.observation_space.shape, (   # 运行期兜底：产出 obs 必与声明空间同形（防增广逻辑与 space 声明漂移·L151 VecNorm 形状坑）
            f"态势增广 obs shape {out.shape} ≠ observation_space {self.observation_space.shape}")
        return out

    # ---- gymnasium 契约 ----
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs, info = self.env.reset(seed=seed)
        self.proj.reset()                                    # ① 集成层必做：清状态机+EC+ρ边沿（防 EC 跨 episode 静默错动作 D33）
        self._rho = RHO_NO_CONFLICT
        self._source = None
        self._prev_u_desired = None                          # action-rate：清上一步动作（防跨 episode 注伪抖动罚·首步无罚·`03` L98）
        return self._augment(obs, self._rho, None), {**info, "rho": self._rho, "rho_acting": self._rho, "source": self._source}   # 🆕 腿1：reset 首帧填 NO_CONFLICT one-hot(置bit0·非全零)+give_way=0（augment 关时 _augment 原样返回=bit-identical）

    def render(self):
        return None

    def step(self, action):
        """SAC 期望连续动作 u_desired → 投影成安全动作 u_safe → env.step（_map_action 最终 box 限幅）。"""
        u_desired = np.asarray(action, dtype=float)
        if u_desired.shape != (2,):
            raise ValueError(f"连续动作应为 [a, ω]，得到 shape={u_desired.shape}")
        if not np.all(np.isfinite(u_desired)):
            raise ValueError(f"动作含非有限值（NaN/inf）：{u_desired}")

        s_obs = self._obs_vs()
        if not self.shield:                                  # 🆕 P0 连续无盾（SE-RL 盾 off）：施 RL 原动作·口径对齐 UnshieldedUSVEnv（rho 恒 NO_CONFLICT·无状态机·无紧急·保 _ego_vs/_obs_vs 供 ViolationCounter 测裸违规=连续版 Base 锚点）。
            u_safe = u_desired
            self._rho = RHO_NO_CONFLICT
            self._source = "unshielded"
            give_way_dir = emergency_mode = None
        elif s_obs is None:                                  # 他船窗外/无 → 无规则适用、无碰撞投影（同 ShieldedUSVEnv 短路、不推状态机）
            # 统一态势盾：无冲突空旷水域也约束"朝目标锥"防游荡（硬约束·**不推状态机/不碰碰撞**·`03` 方案①）。
            #   锥关（goal_cone_half=None 或 goal=None）→ goal_cone_action 返回 None → u_safe=u_desired = 逐位等价现状 bit-identical。
            u_cone = self.proj.goal_cone_action(self._ego_vs(), u_desired, self.env.dt)
            u_safe = u_cone if u_cone is not None else u_desired
            self._rho = RHO_NO_CONFLICT
            self._source = "goal_cone" if u_cone is not None else "no_obstacle"
            give_way_dir = emergency_mode = None
        else:                                                # ③ 每决策步只调一次 safe_action（含 ② box-match assert + 推状态机一步）
            res = self.proj.safe_action(self._ego_vs(), s_obs, u_desired, self.env.dt, self.env.p)
            u_safe = res.u_safe
            self._rho = res.rho
            self._source = res.source
            give_way_dir = res.give_way_dir
            emergency_mode = res.emergency_mode

        # ④ 连续 _map_action 做最终 box 限幅（emergency 越 box 被截）；emergency_used 传 reward 使紧急惩罚 C_EMERGENCY 口径==离散 Discrete-safe（L43-续①修四方混杂）
        obs, r, term, trunc, info = self.env.step(u_safe, emergency_used=(self._source == "emergency"))
        out_info = {
            **info, "rho": self._rho, "rho_acting": self._rho, "source": self._source,
            # rho_acting=ρ@t（连续臂 info["rho"] 本就 pre-step）→ 与离散臂 rho_acting 同口径供 evaluate 紧急步%（D40 #1）
            "give_way_dir": give_way_dir, "emergency_mode": emergency_mode,
            "u_desired": u_desired, "u_applied": self.env.last_action.copy(),   # ⑤ + 端到端 ∈box 可核（u_applied=env 限幅后实施值）
            # ⚠️ B7 护栏（03 L63·Fix① 框架裁决）：u_applied 在 fallback/emergency 步可达【物理满程 ±a_max=0.24/±w_max=0.03】，
            #   越过 RL 动作箱 ±0.048/±0.018（Fix②）。【禁止】把 u_applied 直接经 SB3 policy.scale_action 写回 replay buffer
            #   做"shield-aware 信用分配"——scale([0.24,0.03])=[5.0,1.667] 越 [-1,1]=脏数据（实测 emergency/collision_min 步越界）。
            #   且本项目=SE-RL（盾即环境·Markgraf 2026）：buffer 存采样 u_raw 本就正确、非 bug。
        }
        # 动作混叠惩罚（Markgraf 2026 式20 h=w‖u−uφ‖²·`03` L97·反事实证 aliasing/盾塑偏=42% 残余主因 → 此前"待 A/B 证再上"的条件已满足）。
        #   r_alias = −w·‖(u_desired − u_safe)/u_box‖²·【仅投影步 source=="projection"】（u 被投到 COLREGs-安全集时罚分歧；
        #   no_obstacle 步 u_safe==u_desired 本就 0、emergency/fallback 步 u_safe=EC 输出大尺度=非"混叠"故【排除】防尺度爆炸）。
        #   ⚠️ alias_weight=0（默认）→ 整块跳过 → r/out_info 与现状【逐位等价 bit-identical】（不改 r、不加 "r_alias" 键）。
        if self.alias_weight > 0.0 and self._source == "projection":
            _d = (u_desired - u_safe) / self._u_box
            r_alias = -self.alias_weight * float(np.dot(_d, _d))
            r = r + r_alias
            out_info["r_alias"] = r_alias                       # additive·可消融·仅 alias_weight>0 时出现（保 bit-identical）
        # action-rate 平滑惩罚（治 bang-bang 满舵交替·`03` L98）：r_rate = −w·‖(u_desired − u_desired_prev)/u_box‖²·【全步·罚策略原始动作逐步猛变】。
        #   罚【时间维】策略自身输出抖动（与 alias 的【空间维】策略↔盾分歧正交·两靶独立）；用 u_desired（策略原始·不罚盾介入跳变）。
        #   首步 _prev=None 无罚（reset 清）。⚠️ rate_weight=0（默认）→ 整块跳过（不追踪 _prev·不改 r·不加键）→ 逐位等价 bit-identical。
        # ⚠️ 尺度（严格复审 L99 锐化·原注过保守已纠）：u_desired 在 SB3 rollout 已裁进 RL 箱（PPO np.clip / SAC tanh-squash）
        #   → ‖Δu_desired/u_box‖²∈[0,8]（满舵反转 +box→−box 恰得 8 = bang-bang 最大罚·靶打得准）。**每步原始 r_rate 满舵 −8w**。
        #   🔑 **量级（二次复审 L100 锐化·纠原两处 over-claim："唯一区分信号" + "ret_std~100"）**：
        #      在【两条都到达的轨迹间】r_goal telescoping 近似抵消 → r_rate 在该子集是干净区分项；**但残余失败是【停带外·未到达】=端点不同**·
        #      r_goal Σ 不抵消、主导仍是 r_goal+sparse C_GOAL(+50) → r_rate 【非唯一】区分信号（措辞不可写"唯一"·现有奖励本就偏好到达）。
        #      ⚠️ episode Σ r_rate(w∈{0.5,1.0}·典型抖动)≈−85~−340（满 bang-bang 更负）与【本就脆弱的】到达激励(reach−stall≈+150:C_GOAL+50/避C_STOPPED~+40/r_goal终端~+60)同量级
        #      → 真风险=【太强→策略少打舵欠驱动停带外】(伪 null)·非"被淹没"(太弱)；判读须三向消歧、null 不可单看到达率（见下 🔬）。
        #   clip_reward=10 作用于【归一化后】reward（÷√ret_var）：ret_std 未实测（telescoping 抬高·按机制估 ~300-460·非"~100"）·但 VecNorm 对所有 reward 分量除【同一】
        #      ret_std=不改相对量级·且 ret_std 越大越不触 clip → "远不触 clip"结论仍成立（w=1 安全）。判读须对账训练中实测 ret_rms.var + 归一化后 r_rate 直方图。
        #   🔬 判读三向消歧（让 null 可干净归因）：① r_rate 直方图(raw+归一后)既非可忽略也非饱和触 clip；② jerk(CAT7 ‖Δû‖)/终端横向偏 随 w 单调改善=机制命中；
        #      ③ 安全锚点(碰撞/违规/紧急步%/兜底步%)漂=过保守欠驱动(伪 null·非想法不成立)。三者齐看才能区分"想法不成立 vs 罚太强 vs 罚太弱"。
        #   w 上界真受限于【SAC 探索压塌张力】（per-step 罚也税采样方差·须监控 α；PPO 此张力弱）·非 clip。建议扫 {0,0.5,1.0}·判读核 r_rate 直方图（`03` L98/L99）。
        if self.rate_weight > 0.0:
            if self._prev_u_desired is not None:
                _rw = self.rate_weight
                # 🆕 rank1（`03` L173）：泊位精修区内(‖ego−goal‖≤dock_radius)把治抖罚权重降到 rate_dock=放行入库急打舵对齐窄朝向门。
                #   默认 self._rate_dock=None → 整段跳过 → _rw=self.rate_weight → 与现状【逐位等价 bit-identical】。⚠️ _prev_u_desired 追踪【仍只 gate 在 rate_weight>0】(不动)→区内区外追踪一致·不断链。
                if self._rate_dock is not None and self._dock_radius > 0.0:
                    _dg = self._ego_vs().position - self.env.goal_center
                    if float(_dg[0] * _dg[0] + _dg[1] * _dg[1]) <= self._dock_radius * self._dock_radius:   # 比平方避 sqrt（热路径）
                        _rw = self._rate_dock
                _dr = (u_desired - self._prev_u_desired) / self._u_box
                r_rate = -_rw * float(np.dot(_dr, _dr))
                r = r + r_rate
                out_info["r_rate"] = r_rate                     # additive·可消融·仅 rate_weight>0 且非首步时出现（保 bit-identical·区内值=−rate_dock·‖·‖²）
            self._prev_u_desired = u_desired.copy()             # 更新（仅 rate_weight>0 路径·off 时永不写=off 路径零状态变化·门控不改此追踪条件）
        return self._augment(obs, self._rho, give_way_dir), r, term, trunc, out_info   # 🆕 腿1：post-step obs 尾拼 ρ@t one-hot + give_way_dir（augment 关时 _augment 原样返回=bit-identical）

    # ---- ViolationCounter 喂数（与 ShieldedUSVEnv._ego_vs/_obs_vs 同语义、四方同口径）----
    def _ego_vs(self) -> VesselState:
        e = self.env.ego
        return VesselState(position=np.array([e[0], e[1]], dtype=float),
                           orientation=float(e[2]), velocity=float(e[3]), length=self._ego_length)

    def _obs_vs(self) -> VesselState | None:
        """单他船当前时刻 VesselState；预测窗外（None）→ None（无冲突）。"""
        if not self._obstacles:
            return None
        ob = self._obstacles[0]
        t = self.env.step_idx
        s = ob.initial_state if t == 0 else ob.prediction.trajectory.state_at_time_step(t)
        if s is None:
            return None
        return VesselState(position=np.asarray(s.position, dtype=float),
                           orientation=float(s.orientation), velocity=float(s.velocity),
                           length=self._obs_length)
