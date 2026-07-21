"""
TRB 环境 · gym 接线模块（step2 收尾）
====================================
把 4 件基础设施（usv_dynamics / usv_observation / usv_reward / usv_termination）+ CommonOcean 场景
拼成一个能跑 episode 的环境。这是 Phase 1 step2 的最后一块。

每步循环（忠实论文 §VI Fig.1）：动作 → 动力学前进 → 取他船当前占据 → 终止判定 + 奖励 + 新观测。

⚠️ **所有设计决定 + 依据 + deferred 项都在此列明**（不随意简化任何参数；待 agent 严格复审）：

1. **gymnasium.Env 正式化（step4b，2026-06-11）**：step2 时 gym/gymnasium 未装、建纯类遵契约（reset→(obs,info)
   / step→5 元组）；step4b 装 gymnasium 0.29.1（sb3 2.x 兼容）后 subclass `gym.Env` + 真 spaces
   （observation_space=Box(OBS_DIM,) 无界 float64、不归一化 D6；action_space=连续 Box[a,ω]float32 / 离散
   Discrete(50)）+ reset(seed) 播种 np_random。契约与返回值不变（向后兼容 step2 全部冒烟）。mask + a_em 调度
   留 step4c shield 层（SafeActionScheduler）。

2. **动作空间 = 连续 [a,ω] + 离散 50（49 regular 网格 + a_em 槽位；2026-06-10 step3 组件3/4 接线完成）**：
   - 连续：a∈[−a_max,a_max]=[−0.24,0.24]，ω∈[−ω_max,ω_max]=[−0.03,0.03]（我们的方法 + smoke 用）。
   - 离散 49 = A_a(7) × A_ω(7)，**取值精确**（A_ω 用修正值，论文 −0.06 是 typo，见《文献核实笔记》视觉复核①）。
   - **a_em = 下标 IDX_EMERGENCY(49)**：状态相关非网格点，env 不持状态机 → step(49,
     emergency_action=(a,ω)) 由调度器（usv_colregs.SafeActionScheduler/EmergencyController 按
     Alg.1 算）传值，不传 → ValueError。论文 50 动作齐（Table III 复现接口就绪）。

3. **v_max=9.5 强制 = step 里 clip_velocity（默认 True）**：论文 §VI Table I 把 R2 安全航速记为"本船动力学
   天然保证"。但 yp RHS 不限 v（动力学件忠实保留），且 a_max 在 170 步内不足以自然约束 v → 必有 clip。
   故本环境默认在 step 里把 v 截到 [0, v_max]（dynamics 的 clip_velocity 选项）。⚠️ **可配置 + 待 step4
   用 Table III 核对**（她到底软约束还是硬截 v 未公开，见《02》挂起项）。**非随意定**——基于论文原句 + 数学。

4. **terminated vs truncated（gymnasium 语义）**：1_time（到 k_max）= truncated（时间截断）；
   1_area/stopped/collision/goal = terminated（真终止）。标准 gymnasium 拆分。

5. **emergency_used = 本步是否走 a_em 槽位**（step2 时恒 False；2026-06-10 组件3/4 接线后由
   _map_action 返回：离散 49 → True，其余 → False，传给奖励的 c_emergency 分量）。

6. **reset 时 last_action=[0,0]**：reset 尚未施加动作，控制输入未定义 → 观测的 a_ego/ω_ego 取 0；
   首个 step 后由真实动作填充。(initial_state 带 acceleration 字段，step4 若需更忠实可改用它。)

7. **他船 occupancy 窗外（None）过滤**：他船轨迹 time_step=1..170，t=0 取 initial_state、t>170 为 None。
   episode 内 step≤k_max=170 占据均有效；本环境对 None 占据**过滤掉**（终止件对 None fail-fast，故必须在此滤）。
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .usv_dynamics import DECISION_DT, PAPER_V_MAX, make_vessel_params, step as dyn_step, wrap_to_pi
from .usv_observation import OBS_DIM, ObservationBuilder
from .usv_reward import RewardFunction, C_REACH, V_LOW
from .usv_termination import TerminationChecker

# 动作网格（已锁，见《文献核实笔记》视觉复核①；A_ω 修正值，论文 −0.06 是 typo）
A_ACC: tuple = (-0.048, -0.032, -0.016, 0.0, 0.016, 0.032, 0.048)
A_OMEGA: tuple = (-0.018, -0.012, -0.006, 0.0, 0.006, 0.012, 0.018)
# 离散 regular 网格 49 = 7×7 + 紧急槽位 1 = 共 50（论文 49+a_em；step3 组件3/4 已接线）
DISCRETE_ACTIONS: tuple = tuple((a, w) for a in A_ACC for w in A_OMEGA)
N_DISCRETE: int = len(DISCRETE_ACTIONS)   # 49（regular 网格数）
IDX_EMERGENCY: int = N_DISCRETE           # 49 = a_em 槽位下标（状态相关非网格点，值由调度器算）
N_ACTIONS_TOTAL: int = N_DISCRETE + 1     # 50

# RL 智能体【正常操作】动作范围 = A_a/A_ω 满格 ±0.048/±0.018（程序化抽自网格·非硬编·常量改自动跟上）。
# 依据：Krasowski《文献核实笔记》:94/96 —— RL 智能体 normal operation 缩到 A_a/A_ω；
#       【满程 [±a_max,±w_max] 只给紧急控制器】（a_max=0.24/w_max=0.03 是物理/紧急范围、非 RL 正常控制权限）。
# 连续臂 ContinuousProjectionEnv 的 SAC 动作箱用此范围（根因修 03 L63 Fix②：原用满程 ±0.24/±0.03
#   = 离散臂 5×/1.67× 过权限 = 停船墙根 + 四方公平 confound〔03 L1a-4〕；缩到正常操作=离散臂同款权限·更忠实·消停船墙。
#   盾/紧急控制器/动力学/_map_action 仍用物理 ±a_max（proj box 不变=B-EMERGENCY-BOX 契约不破））。
A_NORMAL_ACCEL_MAX: float = float(max(A_ACC))    # 0.048（=A_a 上界）
A_NORMAL_OMEGA_MAX: float = float(max(A_OMEGA))  # 0.018（=A_ω 上界）


def assert_single_obstacle(obstacles, cls_name: str) -> None:
    """盾/投影族 env 的【单他船 assumption 2】显式契约（D40 #4 / L49 #2，2026-06-17d）。

    盾/投影只护 `obstacles[0]`（ShieldedUSVEnv/UnshieldedUSVEnv/ContinuousProjectionEnv 的 `_obs_vs` 恒取第一艘），
    但 USVEnv 碰撞检测（`_obstacles_at`）遍历【全部】他船 → 多他船场景下盾会静默只护一艘、对 obstacles[1+] 的危险
    动作被标 safe 放行（与"盾保无碰撞"卖点不符的安全漏洞，且无报警）。当前场景池 HandcraftedTwoVesselEncounters
    全单船（ego+1）= 不触发；Phase 5 多船须【先扩盾护全部他船】再放开此契约。把潜伏假设变成显式 fail-fast 契约。
    """
    if len(obstacles) > 1:
        raise NotImplementedError(
            f"{cls_name}: 检测到 {len(obstacles)} 艘他船，但投影盾只护 obstacles[0]（单他船 assumption 2）。"
            "多他船 = Phase 5（须先扩盾护全部他船 + 多船算例），当前勿载入多船场景（D40 #4 / 03 L49 #2）。")


class USVEnv(gym.Env):
    """Krasowski 海事避碰环境（连续动作）。正式 gymnasium.Env（step4b 正式化，见 docstring 决定 1）。

    用法：
        env = USVEnv(scenario, planning_problem)          # 从 CommonOcean 读出的对象
        obs, info = env.reset(seed=0)
        obs, reward, terminated, truncated, info = env.step([a, w])   # 连续；离散则 step(动作下标)

    ⚠️ 离散 IDX_EMERGENCY(49) 槽位的 a_em 由上层 shield（SafeActionScheduler，step4c）经 step(49,
    emergency_action=…) 提供 → 裸 USVEnv 不可直接 action_space.sample() 用 49（会 ValueError，契约见决定 2）。
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario,
        planning_problem,
        *,
        continuous: bool = True,
        clip_velocity: bool = True,
        v_max: float = PAPER_V_MAX,
        colregs_weight: float = 1.0,
        gamma: float = 0.99,
        well_shaping_weight: float = 0.0,
        shaping_radius: float = 500.0,
        xtrack_weight: float = 0.0,
        xtrack_radius: float = 80.0,
        park_weight: float = 0.0,
        park_radius: float = 400.0,
        park_v_target: float = 4.0,
        c_step: float = 0.0,
        c_dwell: float = 0.0, w_dwell: float = 90.0, h_dwell: float = 0.52, dwell_radius: float = 250.0, b_dwell: float = 0.0,   # r_dwell 入库赤字滞留成本（`03` L161/L162·连续臂专属·非PBRS·默认关=逐位等价·透传 RewardFunction）
        c_reach: float = C_REACH, dock_radius: float = 0.0, v_dock: float = V_LOW,   # 第二条腿修法（`03` L172·连续臂专属·默认关=逐位等价·透传 RewardFunction）：c_reach=重标 r_goal 治 corr≈0 / dock_radius+v_dock=泊位精修门降速度地板治过路惩罚泄漏
        arrival_heading_slack: float = 0.0,
        goal_ignore_orientation: bool = False,   # 🆕 L185（user 2026-07-13）：训练目标去朝向硬门→1_goal 只判位置到达区域（透传 TerminationChecker·默认 False=严格真门=bit-identical）
    ):
        """
        scenario         : CommonOcean Scenario（含 dynamic_obstacles）
        planning_problem : CommonOcean PlanningProblem（initial_state + goal）
        continuous       : True=动作 [a,ω] 连续；False=离散下标 → 49 网格
        clip_velocity    : step 里是否把 v 截到 [0,v_max]（默认 True，v_max 强制，见 docstring 决定 3）
        v_max            : 最大速度（论文 9.5）
        colregs_weight   : r_colregs 总权重（Base/RR 开关，4d-②）：1.0=RR/Safe（默认、现状）/ 0.0=Base（关 r_colregs）。透传 RewardFunction。
        gamma / well_shaping_weight / shaping_radius : 修法A 进门势 PBRS（`03` L81 接线·透传 RewardFunction）。
            well_shaping_weight 默认 0.0=关=与现状逐位等价 bit-identical；gamma 须与 trainer 同源（破 Ng 风险·run_step4e _GAMMA 单源）；
            goal_orientation 此处内部自带（line 129 从场景算·不需调用方传）。
        """
        self.scenario = scenario
        self.continuous = bool(continuous)
        self.clip_velocity = bool(clip_velocity)
        self.p = make_vessel_params(v_max)
        self.dt = DECISION_DT
        self.obstacles = list(scenario.dynamic_obstacles)

        self.init_state = planning_problem.initial_state
        goal = planning_problem.goal
        g0 = goal.state_list[0]
        self.k_max = int(g0.time_step.end)                          # 170
        goal_center = np.asarray(g0.position.center, dtype=float)
        self.goal_center = goal_center               # 存目标点 [x, y]（供连续投影盾 ρ0 朝目标锥安全集 set_goal 用）
        goal_orientation = (float(g0.orientation.start), float(g0.orientation.end))
        init_pos = np.asarray(self.init_state.position, dtype=float)

        # 4 件
        self.obs_builder = ObservationBuilder(goal_center, goal_orientation, init_pos, self.k_max)
        self.reward_fn = RewardFunction(goal_center, init_pos, colregs_weight=colregs_weight,
                                        gamma=gamma, goal_orientation=goal_orientation,
                                        well_shaping_weight=well_shaping_weight, shaping_radius=shaping_radius,
                                        xtrack_weight=xtrack_weight, xtrack_radius=xtrack_radius,   # 对症 横向进带势透传（`03` L88）
                                        park_weight=park_weight, park_radius=park_radius, park_v_target=park_v_target,   # 想法B 终端保速势透传（`03` L109）
                                        c_step=c_step,   # 修法C 每步生存成本透传（`03` L123·连续臂专属）
                                        c_dwell=c_dwell, w_dwell=w_dwell, h_dwell=h_dwell, dwell_radius=dwell_radius, b_dwell=b_dwell,   # r_dwell 入库赤字滞留成本透传（`03` L161/L162·连续臂专属·非PBRS）
                                        c_reach=c_reach, dock_radius=dock_radius, v_dock=v_dock)   # 第二条腿修法透传（`03` L172·连续臂专属·默认关 bit-identical）
        self.term_checker = TerminationChecker(goal, self.k_max, goal_ignore_orientation=goal_ignore_orientation)   # 🆕 L185：去朝向硬门透传（默认 False=严格真门=bit-identical）
        # 🆕 B1（`03` L153）：到达门朝向容差课程 slack。训练放宽终端朝向门→崩种子探索中撞中 +50 学会终端·退火回真门；
        #   评估 env【绝不】传 slack>0 / 不挂退火 callback → 恒 0 = 真门（诚实红线）。默认 0.0 → term_checker slack=0
        #   = _goal_widened None = bit-identical。>0 时由 run_step4e 分段退火经 set_arrival_slack 注入（MultiScenarioEnv 双写透传）。
        self.arrival_heading_slack = float(arrival_heading_slack)
        if self.arrival_heading_slack > 0.0:
            self.term_checker.set_arrival_slack(self.arrival_heading_slack)

        # gymnasium 正式空间（step4b）：观测原始物理量不归一化（D6）→ 无界 Box，归一化交训练层（VecNormalize, step4c/Phase3）
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float64)
        if self.continuous:
            self.action_space = spaces.Box(
                low=np.array([-self.p.a_max, -self.p.w_max], dtype=np.float32),
                high=np.array([self.p.a_max, self.p.w_max], dtype=np.float32),
                dtype=np.float32)                              # [a, ω]∈[±a_max, ±w_max]（连续方法 / Phase 3 SAC）
        else:
            self.action_space = spaces.Discrete(N_ACTIONS_TOTAL)   # 49 regular 网格 + a_em 槽 = 50（基线 MaskablePPO）

        self.step_idx = 0
        self.ego = None
        self.last_action = np.array([0.0, 0.0])
        # 🆕 逆向起点课程（方案C-B·`03` L181·攻崩种子高速绕圈坏盆地）：把 ego 重生到 init→goal 参考线上更靠泊位门处。
        #   start_frac=1.0（默认/评估）→ ego 生在真 init_state = **bit-identical**；<1 → goal_center+f·(init−goal)=更靠门（f→0 贴门口）。
        #   init_position（参考线锚点·构造时进 reward/obs）【永不变】→ 只挪 ego 落点、d_lat≈0、obs/reward 口径一致、不重建。
        #   start_v=None → 用真 init 速度（bit-identical）；设值 → 课程重生速度（须覆盖真实到门速度分布含高速·否则学假终端·`03` L181 综合弱点②）。
        self.start_frac = 1.0
        self.start_v = None

    def set_start_frac(self, frac: float, v=None):
        """🆕 逆向起点课程 setter（`03` L181·经 ContinuousProjectionEnv→MultiScenarioEnv 双写·同 set_arrival_slack 退火范式）。
        frac=1.0（默认/评估·退火关时【从不被调用】）→ 真起点 = bit-identical；frac<1（仅训练）→ ego 生到更靠门。
        v=None → 真 init 速度；v=float → 课程重生速度。**eval env 从不调本方法 → 恒真起点 = 诚实红线（同 arrival_slack）**。"""
        self.start_frac = float(frac)
        self.start_v = None if v is None else float(v)

    def set_arrival_slack(self, slack: float):
        """🆕 B1（`03` L153）：到达门朝向容差课程 setter（经 ContinuousProjectionEnv→MultiScenarioEnv 双写、
        VecEnv.env_method 跨进程调用·同 set_penalty_weight 退火范式，但目标在更内层 term_checker）。
        slack=0（默认/评估）→ 真门 = bit-identical；slack>0（仅训练）→ 放宽朝向门（term_checker deepcopy 加宽副本·不碰真 goal）。
        退火【关闭】时本方法【从不被调用】→ term_checker 恒 slack=0 = 字节级不变。"""
        self.arrival_heading_slack = float(slack)
        self.term_checker.set_arrival_slack(slack)

    def reset(self, *, seed=None, options=None):
        """重置到场景初始态，返回 (obs, info)。seed 透传 gymnasium.Env 播种 np_random（step4b）。"""
        super().reset(seed=seed)
        self.step_idx = 0
        # 🆕 逆向起点课程（`03` L181）：frac>=1.0（默认）走【与原代码逐字相同】的分支 = bit-identical；<1 才沿 init→goal 线挪向门。
        if self.start_frac >= 1.0:
            _px, _py = float(self.init_state.position[0]), float(self.init_state.position[1])
            _v = float(self.init_state.velocity)
        else:
            _ip = np.asarray(self.init_state.position, dtype=float)
            _pos = self.goal_center + self.start_frac * (_ip - self.goal_center)   # 沿参考线·f→0 贴门（init_position 不动=obs/reward 口径一致）
            _px, _py = float(_pos[0]), float(_pos[1])
            _v = float(self.init_state.velocity) if self.start_v is None else self.start_v
        self.ego = np.array([
            _px,
            _py,
            wrap_to_pi(self.init_state.orientation),
            _v,
        ])
        self.last_action = np.array([0.0, 0.0])          # 见 docstring 决定 6
        self.obs_builder.reset()
        self.reward_fn.reset(self.ego)

        # 注：reset 这帧 build 会给 ObservationBuilder 记下 t=0 各他船距离 → 首个 step 的 ḋ 是真实
        # 距离变化率 d(t1)−d(t0)（非"首次出现=0"）。物理正确 + 确定性一致（独立复核 NIT）。
        states, _ = self._obstacles_at(0)
        obs = self.obs_builder.build(
            self.ego, self.last_action,
            [(oid, pos) for oid, pos, _ in states], step=0, term_flags=None,
        )
        return obs, {"step": 0}

    def render(self):
        """render_mode=None → 返回 None（gymnasium 推荐，免基类 NotImplementedError；本 env 无内建可视化，绘图用 commonocean）。"""
        return None

    def step(self, action, emergency_action=None, emergency_used=False):
        """前进一步，返回 (obs, reward, terminated, truncated, info)。

        emergency_action：离散动作 IDX_EMERGENCY(49) 的实际控制 (a,ω)=a_em——由调度器
        （usv_colregs.SafeActionScheduler / EmergencyController）按当前状态算好传入；
        选 49 不传 → ValueError（a_em 状态相关，env 不持状态机）。其余动作忽略此参数。
        emergency_used：**连续动作路径**下由调用方（连续投影盾 ContinuousProjectionEnv）传入"本步是否调用了
        紧急控制器"（source=='emergency'）→ 进 reward 的 c_emergency(−0.5)，与离散 idx49 口径一致（2026-06-17b
        D39/L43-续 修四方紧急惩罚口径不一致）。**离散路径由 _map_action 按 idx49 判定、默认 False 不受此参数影响=向后兼容**。
        """
        if self.ego is None:
            raise RuntimeError("step() 前必须先 reset()")
        a, w, em_used = self._map_action(action, emergency_action)
        em_used = em_used or bool(emergency_used)    # 离散 idx49→True 已定；连续路径由调用方传（默认 False 不改离散 = 向后兼容位级恒等）

        # 1) 动力学前进（v_max 由 clip_velocity 强制，见决定 3）
        self.ego = dyn_step(self.ego, (a, w), self.dt, self.p, clip_velocity=self.clip_velocity)
        self.step_idx += 1

        # 2) 取他船当前时刻 state（pos/vel）+ 占据（碰撞）；None 过滤（决定 7）
        states, footprints = self._obstacles_at(self.step_idx)

        # 3) 终止判定
        done, flags = self.term_checker.check(self.ego, self.step_idx, footprints)

        # 4) 奖励（em_used = 本步是否走 a_em 槽位〔离散 idx49〕或连续投影盾标 source=='emergency'）
        reward, parts = self.reward_fn.step(
            self.ego, states, term_flags=flags, emergency_used=em_used,
        )

        # 5) 新观测
        obs = self.obs_builder.build(
            self.ego, np.array([a, w]),
            [(oid, pos) for oid, pos, _ in states], step=self.step_idx, term_flags=flags,
        )
        self.last_action = np.array([a, w])

        terminated = flags["area"] or flags["stopped"] or flags["collision"] or flags["goal"]
        truncated = flags["time"]
        info = {"flags": flags, "reward_parts": parts, "step": self.step_idx}
        return obs, reward, terminated, truncated, info

    def _map_action(self, action, emergency_action=None):
        """动作 → (a, ω, emergency_used)。连续=直接取 [a,ω]；离散=下标查 49 网格或 49=a_em 槽位。"""
        if self.continuous:
            act = np.asarray(action, dtype=float)
            if act.shape != (2,):
                raise ValueError(f"连续动作应为 [a, ω]，得到 shape={act.shape}")
            if not np.all(np.isfinite(act)):
                raise ValueError(f"动作含非有限值（NaN/inf）：{act}")
            # 截到执行器物理上限：dynamics 内部也会截，这里截使观测 a_ego/ω_ego 反映"实际施加"值
            # 而非"指令值"（否则越界动作下 obs 报指令、船却只动到上限 → obs 与现实不一致，独立复核 MINOR）
            a = float(np.clip(act[0], -self.p.a_max, self.p.a_max))
            w = float(np.clip(act[1], -self.p.w_max, self.p.w_max))
            return a, w, False
        idx = int(action)
        if idx != action:   # 拒非整数下标（如 24.9 会被 int() 静默截成 24，独立复核 MINOR）
            raise ValueError(f"离散动作下标应为整数，得到 {action}")
        if not (0 <= idx < N_ACTIONS_TOTAL):
            raise ValueError(f"离散动作下标应 ∈ [0,{N_ACTIONS_TOTAL})，得到 {idx}")
        if idx == IDX_EMERGENCY:                       # 49 = a_em 槽位：值须由调度器算好传入
            if emergency_action is None:
                raise ValueError("动作 49=a_em 需要 emergency_action=(a,ω)（由调度器/紧急控制器提供）")
            em = np.asarray(emergency_action, dtype=float)
            if em.shape != (2,) or not np.all(np.isfinite(em)):
                raise ValueError(f"emergency_action 应为有限 (a,ω)，得到 {emergency_action!r}")
            a = float(np.clip(em[0], -self.p.a_max, self.p.a_max))
            w = float(np.clip(em[1], -self.p.w_max, self.p.w_max))
            return a, w, True
        a, w = DISCRETE_ACTIONS[idx]
        return a, w, False

    def _obstacles_at(self, t: int):
        """取 t 时刻各他船：states=[(id, pos, vel_vec)]、footprints=[shapely 占据]。窗外(None) 过滤。"""
        states = []
        footprints = []
        for ob in self.obstacles:
            s = ob.initial_state if t == 0 else ob.prediction.trajectory.state_at_time_step(t)
            occ = ob.occupancy_at_time(t)
            if s is None or occ is None:                  # 预测窗外 → 过滤（决定 7）
                continue
            pos = np.asarray(s.position, dtype=float)
            spd, ori = float(s.velocity), float(s.orientation)
            vel_vec = spd * np.array([np.cos(ori), np.sin(ori)])   # 速度向量（reward v_y 用）
            states.append((ob.obstacle_id, pos, vel_vec))
            footprints.append(occ.shape.shapely_object)
        return states, footprints
