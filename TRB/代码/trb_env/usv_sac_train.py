"""
TRB · Phase 3 Node B —— Continuous-safe SAC 训练 setup（2026-06-17(c)）
======================================================================
把 Node A 的连续投影盾 env（`ContinuousProjectionEnv`）接成 **SAC 可训练的 Continuous-safe 臂**。
**独立 SAC 入口**——刻意【不复用】train.py 的 MaskablePPO `train_multiscene`：那是离散专用（硬编 MaskablePPO +
ent_coef=0.01 + 默认 colregs_weight=1.0），直接拿来跑连续臂会 ① MaskablePPO 遇连续 Box 构造即崩 ② 默认
colregs_weight=1.0 复活 r_colregs（堵 D40 train_multiscene 连续臂 footgun，03 L44）。

四方对比同口径（D34/D37-B/D38）：
  · Base/RR        = UnshieldedUSVEnv + MaskablePPO（colregs_weight 0/1）
  · Discrete-safe  = ShieldedUSVEnv   + MaskablePPO（As(ρ) mask）
  · Continuous-safe= ContinuousProjectionEnv + **SAC**（本模块）
四方共用：同 env 物理 / 同场景池 / 同 evaluate+ViolationCounter（Node C）/ **同款 obs 归一化**（VecNormalize，
单一真相源 = train.VECNORM_KWARGS）。唯一有意差异 = 盾/投影 on-off + 算法（MaskablePPO vs SAC，D2）+ r_colregs。

【本模块的 Node B 决定（fact-based）】：
  · **算法 = SAC**（stable_baselines3；连续动作 + 自带最大熵探索 ent_coef='auto' 自动调温）。D38：**别搬离散
    ent_coef=0.01**（那是 PPO 解停船塌缩的；SAC 不靠 masking 清零熵、自带熵正则）→ 本模块**不传 ent_coef**，
    并断言模型 ent_coef=='auto'（防误覆盖）。
  · **colregs_weight 默认 0.0(可经 STEP4E_COLREGS_WEIGHT 覆盖·A/B 复活 r_colregs)**（丢 r_colregs、合规靠投影约束(档位A 经验性·非档位B provable 硬保证)，D37-B；写作声明口径差）+ 运行时 probe 断言。
  · **网络 = MLP [64,64]**（忠实 Krasowski §VII，单一真相源 = train.POLICY_NET_ARCH）。
  · **gamma 同进 SAC 与 VecNormalize**（L21 MINOR②：两处必一致，否则 reward 归一化的 return 估计与 critic 折扣不符）。

【可调、留 Tier-3 训练时定（做成参数、默认有据、不阻塞 Node B 建成）】：
  · `norm_reward`：SAC 是 off-policy（replay buffer 存的是插入时归一化的 reward、统计会漂移）→ norm_reward 与 SAC
    有已知张力。**但钱图指标（到达/碰撞/违规/紧急步%/Ep长）全从原始轨迹经 ViolationCounter 算、不依赖 reward
    归一化** → 默认依 D38 保 True（四方平价），训练不稳时可 A/B 关掉（钱图指标不受影响、仅声明训练口径差）。
  · `n_envs`：SAC off-policy 不像 PPO 靠并行 rollout 提采样效率 → 默认偏小（1）；Tier-3 按吞吐实测调。

⚠️ **真训练 = `train_continuous_safe`（model.learn(3M)）= Tier 3 烧算力**，**待 user 拍板** + 与 T-ITS 错峰。
   本模块只给配好的 make/train 入口；冒烟仅 learn 极少步验端到端集成（SAC 动作 → 投影盾 → env.step）。
⚠️ **Node B/C 接线前必堵 D40 P0**（03 L44/D40）：① #1 紧急步% 两臂时序口径统一（评估侧，Node C）；
   ② 机制性「四方同 ego 轨迹对拍」强制回归测试（Node C 评估接线时立）。本模块 `assert_continuous_safe_caliber`
   是训练侧的口径自检（net/算法/归一化/colregs_weight/ent_coef），评估侧对拍留 Node C。
"""
from __future__ import annotations

import math

import numpy as np
import stable_baselines3 as _sb3
import torch as th
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ContinuousCritic
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.sac.policies import SACPolicy
from torch import nn
from torch.nn import functional as F

from .train import ENT_COEF, POLICY_NET_ARCH, VECNORM_KWARGS   # 单一真相源：四方同网络 + 同 obs 归一化口径 + 同探索 ent_coef
from .usv_dynamics import wrap_to_pi   # P1 朴素基线航向误差用（角度 wrap 到 [-π,π]）
from .usv_continuous_shield import ContinuousProjectionEnv
from .usv_env import A_NORMAL_ACCEL_MAX, A_NORMAL_OMEGA_MAX, C_REACH, V_LOW   # RL 正常操作动作箱（L63 Fix②）+ 第二条腿修法默认（`03` L172）
from .usv_scenarios import load_scenario_pool, make_vec_env

# --- Continuous-safe 不可变口径（硬编、防 footgun）---
CONTINUOUS_SAFE_COLREGS_WEIGHT = 0.0   # 丢 r_colregs（合规靠投影约束(档位A 经验性·非档位B provable 硬保证)，D37-B）；硬编防 D40 #2/train_multiscene footgun
assert CONTINUOUS_SAFE_COLREGS_WEIGHT == 0.0, "Continuous-safe 必须丢 r_colregs（colregs_weight=0.0，D37-B）"

TOTAL_TIMESTEPS = 3_000_000            # 每 seed 3M（忠实 Krasowski §VII；Tier-3 训练用）

_SB3_TRAIN_REF_VERSION = "2.3.2"       # StableSAC.train() 逐字复制自此版本·sb3 升级须重核 train() 并同步本类


class PursuitNaivePolicy:
    """P1 公平朴素基线（why-RL 消融 B 臂·L146·user 2026-07-04 定"追目标+冲突减速"）：
    【非学习】手写控制律 = ① 追目标：朝目标位置转向（世界方位角误差 → 0，比例控制·饱和到动作箱）
                        ② 冲突减速：他船进近场 → 降速让盾有余量做合规避碰（避碰=盾干的·不是朴素干的）。
    定位：盾(shield=True)保证每步合规+无碰撞·朴素只提供"去目标+适度减速"意图 → RL 须赢在【盾内的速度调度/让路时机/平滑】(非避碰)才叫真增量·否则 why-RL 空心。
    几何来源：用 env 原始状态（bind_env 注入 → env._ego_vs()/env.env.goal_center/env._obs_vs()）算【到目标位置方位角】。
      这是 RL 那 27 维 obs 的【同一信息、更直接】(obs 由同一原始状态派生·β_goal 是航向-vs-目标朝向区间差非方位角→从 obs 解方位脆弱)；
      手写基线用干净几何(如 GPS 方位)是标准做法·且是更强/更公平的基线（RL 若仍赢=更有说服力）。
    接口：predict(obs, deterministic=) → (action=[a,ω], None)（sb3 兼容·run_episode_continuous 会先 bind_env·不改 eval 度量口径）。"""
    def __init__(self, *, dt, v_max, a_box, w_box, d_slow=3000.0, v_slow=6.0, k_head=1.0):
        self.dt = float(dt); self.v_max = float(v_max)
        self.a_box = float(a_box); self.w_box = float(w_box)
        self.d_slow = float(d_slow); self.v_slow = float(v_slow); self.k_head = float(k_head)
        self._env = None

    def bind_env(self, env):
        """run_episode_continuous 每 episode reset 后注入当前 env（供读原始几何）。"""
        self._env = env

    def predict(self, obs, deterministic=True, **kwargs):
        env = self._env
        if env is None:
            raise RuntimeError("PursuitNaivePolicy 未 bind_env（run_episode_continuous 须先 model.bind_env(env)）")
        ego = env._ego_vs()                                     # VesselState: position[x,y]/orientation/velocity
        goal = np.asarray(env.env.goal_center, dtype=float)     # 目标区中心 [x,y]
        px, py = float(ego.position[0]), float(ego.position[1])
        # ① 追目标：到目标位置的世界方位角 − 本船航向 = 航向误差；朝目标转（ω>0=port/增θ）·比例·饱和
        bearing = math.atan2(goal[1] - py, goal[0] - px)
        head_err = wrap_to_pi(bearing - float(ego.orientation))
        omega = float(np.clip(self.k_head * head_err / self.dt, -self.w_box, self.w_box))
        # ② 冲突减速：他船进近场 → 降到 v_slow 让盾有余量；否则开阔水域全速。a=(v_target−v_ego)/dt 饱和
        s_obs = env._obs_vs()
        d_min = (float(np.linalg.norm(np.asarray(s_obs.position, float) - np.array([px, py]))) if s_obs is not None else float("inf"))
        v_target = self.v_slow if d_min < self.d_slow else self.v_max
        accel = float(np.clip((v_target - float(ego.velocity)) / self.dt, -self.a_box, self.a_box))
        return np.array([accel, omega], dtype=np.float32), None


# ---------------- 学习率退火（LR anneal·`03` L88·治连续臂晚期漂移/方差）----------------
class LRAnnealSchedule:
    """学习率线性退火 start→end（前 anneal_steps 累积训练步退完·之后恒 end）。SB3 model.lr_schedule 替换体。

    🔑 **有意忽略 SB3 传入的 progress_remaining**：本项目分段训练（NSEG 段·每段独立 `model.learn(reset_num_timesteps=False)`）
    下，SB3 的 `_current_progress_remaining = 1 − num_timesteps/_total_timesteps`，而 `_total_timesteps` 在续段时
    `+= num_timesteps`（base_class.py:284/416 实读坐实）→ progress 在【每段内】从 (1−c/(c+1)) 降到 0 = 锯齿、全局不单调；
    直接用会每段末把 lr 退到 end。改读自身 `num_timesteps`（由 LRAnnealSyncCallback 每步同步成 model 的【累积】num_timesteps）
    → 对分段鲁棒、与既有 ent 退火（_EntAnneal 同样用累积 num_timesteps）一个口径。

    模块级普通类（非 closure/lambda）→ SB3 `model.save()` 经 cloudpickle（save_util.py·lr_schedule 不在 _excluded_save_params）
    可序列化 + 重载（eval 不训练→该 schedule 不被调用·无害）。仅含 float/int 状态、无 model 反向引用→无循环引用、不撑大存档。
    """

    def __init__(self, start, end, anneal_steps):
        self.start = float(start)
        self.end = float(end)
        self.anneal_steps = max(1.0, float(anneal_steps))
        self.num_timesteps = 0                    # 由 LRAnnealSyncCallback 每步写成 model.num_timesteps（累积）

    def __call__(self, progress_remaining):       # SB3 接口签名传入·有意忽略（见类 docstring）
        frac = min(1.0, max(0.0, self.num_timesteps / self.anneal_steps))
        return self.start + frac * (self.end - self.start)


class LRAnnealSyncCallback(BaseCallback):
    """每个环境步把 model.num_timesteps（累积·跨段持久）同步进 LRAnnealSchedule.num_timesteps，
    使 SB3 train() 内 `_update_learning_rate` → `lr_schedule(progress)` 拿到正确的【全局】训练进度。

    纯写一个 int·不读/不改任何训练张量·不 advance RNG（同 _EntAnneal/_CurveLogger 的只读纪律 → 零扰动·
    退火关闭时本回调【不被安装】= 训练字节级不变）。_on_step 在 collect_rollouts 内每步触发、早于 train() →
    train() 调 lr_schedule 时 num_timesteps 已是当前累积值（PPO/SAC 均如此·StableSAC.train() 亦调 _update_learning_rate）。
    """

    def __init__(self, sched):
        super().__init__()
        self._sched = sched

    def _on_step(self):
        self._sched.num_timesteps = self.model.num_timesteps
        return True


# ---------------- 惩罚权重退火（penalty anneal·`03` L103·治"惩罚从第0步压脆弱种子=先学躲再没机会学到达"）----------------
class PenaltyAnnealSchedule:
    """惩罚权重退火 hold-then-ramp（通用·支持 alias_weight/rate_weight）：
      phase1  t < ramp_start_steps          → start（=0·让策略先学会到达/起飞·复刻无罚起飞条件）
      phase2  ramp_start ≤ t < +anneal_steps → 线性 start→end
      phase3  t ≥ ramp_start+anneal_steps    → 恒 end
    与 LRAnnealSchedule 同款【读自身累积 num_timesteps·有意忽略 SB3 锯齿 progress】（分段训练 NSEG·reset_num_timesteps=False
    下 SB3 progress 段内锯齿、全局不单调；改读累积 num_timesteps 对分段鲁棒·见 LRAnnealSchedule docstring）。
    num_timesteps 由 PenaltyAnnealSyncCallback 每步写成 model 累积步。纯 float/int 状态、无 model 反向引用
    （不进 model.save·只活在 callback 里·无 pickle 顾虑）。ramp_start_steps 是 LR 退火没有的新参数=延迟起 ramp（hold 段）。"""

    def __init__(self, start, end, ramp_start_steps, anneal_steps):
        self.start = float(start)
        self.end = float(end)
        self.ramp_start_steps = max(0.0, float(ramp_start_steps))
        self.anneal_steps = max(1.0, float(anneal_steps))
        self.num_timesteps = 0                    # 由 PenaltyAnnealSyncCallback 每步写成 model.num_timesteps（累积）

    def value(self):
        frac = min(1.0, max(0.0, (self.num_timesteps - self.ramp_start_steps) / self.anneal_steps))
        return self.start + frac * (self.end - self.start)


class PenaltyAnnealSyncCallback(BaseCallback):
    """每环境步：① 把 model.num_timesteps（累积·跨段持久）同步进各 PenaltyAnnealSchedule；② 把当前 value() 经
    venv.env_method('set_penalty_weight', name, w) 推到【所有子 env】（SubprocVecEnv 跨进程经 pipe·VecNormalize wrapper 透传 env_method）。
    持 venv 引用（装时传入训练 venv·不靠 self.model.get_env() 避 VecNormalize 包装层歧义；eval env 另建·不被触碰）+ scheds 字典 {name→schedule}。
    纯写 python float 属性·不读/不改训练张量·不 advance RNG（同 LRAnnealSyncCallback 只读纪律 → 零扰动·关闭时本回调【不安装】= 字节级不变）。
    _on_step 在 collect_rollouts 内每步触发、早于本步 env.step 之后 → 更新在【下一步】env.step 生效（一步延迟·同 LR 退火·量级无害）。
    perf：仅在 value 变化时才 env_method（hold 段 w 恒 start=0 → 跳过冗余 IPC；ramp 段每步变→每步推）。"""

    def __init__(self, scheds, venv):
        super().__init__()
        self._scheds = dict(scheds)               # {'alias_weight'|'rate_weight': PenaltyAnnealSchedule}
        self._venv = venv
        self._last = {name: None for name in self._scheds}

    def _on_step(self):
        t = self.model.num_timesteps
        for name, sched in self._scheds.items():
            sched.num_timesteps = t
            w = sched.value()
            if self._last[name] is None or w != self._last[name]:   # 仅变化时推（hold 段恒 0→跳过·省跨进程 IPC）
                self._venv.env_method("set_penalty_weight", name, w)
                self._last[name] = w
        return True


class ArrivalSlackAnnealSchedule:
    """🆕 B1（`03` L153）：到达门朝向容差 slack 退火 start→0（线性·**量化 n_levels 档**）。
      phase1  t < anneal_steps  → 线性 start→0（量化到 n_levels 档）
      phase2  t ≥ anneal_steps  → 恒 0（真门·训练后段用真门让策略学会真精度=收敛到真到达门·再评估）
    与 PenaltyAnnealSchedule 同款【读自身累积 num_timesteps·忽略 SB3 分段锯齿 progress】（NSEG 分段鲁棒）。
    ⚠️【关键设计·与 penalty 不同】：set_arrival_slack 每次改都 deepcopy goal（贵）→ 必须【量化】value() 到 n_levels 档，
       使 ArrivalSlackAnnealSyncCallback 的"仅变化时推"守卫把总 deepcopy/IPC 次数压到 ~n_levels（非每步·penalty 每步推靠 setattr 便宜）。
    num_timesteps 由 callback 每步写。纯 float/int 状态、无 model 反向引用（不进 model.save·只活在 callback·无 pickle 顾虑）。"""

    def __init__(self, start, anneal_steps, n_levels=20):
        self.start = float(start)                 # slack_start（rad·>0）
        if not (np.isfinite(self.start) and self.start > 0.0):   # M1 防御（对抗审）:对齐 set_arrival_slack 守卫·防 nan/inf/≤0 start 在 value() round() 崩（run_step4e 上游已校验·此为一致性兜底）
            raise ValueError(f"ArrivalSlackAnnealSchedule start 须有限且 >0，得 {self.start}")
        self.anneal_steps = max(1.0, float(anneal_steps))
        self.n_levels = max(1, int(n_levels))
        self.num_timesteps = 0                     # 由 ArrivalSlackAnnealSyncCallback 每步写成 model 累积步

    def value(self):
        frac = min(1.0, max(0.0, self.num_timesteps / self.anneal_steps))   # 0→1 over anneal_steps
        raw = self.start * (1.0 - frac)           # 线性 start→0
        if raw <= 0.0:
            return 0.0
        step = self.start / self.n_levels          # 量化步长
        q = round(raw / step) * step               # 量化到 n_levels 档（∈[0, start]）
        return max(0.0, min(self.start, q))


class ArrivalSlackAnnealSyncCallback(BaseCallback):
    """🆕 B1（`03` L153）：每环境步把 model.num_timesteps 同步进 ArrivalSlackAnnealSchedule，并在【量化 slack
    变化时】经 venv.env_method('set_arrival_slack', v) 推到所有子 env（穿透到内层 term_checker·MultiScenarioEnv 双写）。
    与 PenaltyAnnealSyncCallback 同款【只读纪律】：不改训练张量/不 advance RNG（零扰动·**关闭时本回调不安装=字节级不变**）。
    【关键差异】set_arrival_slack 每次 deepcopy goal（贵）→ 靠 schedule 量化 + "仅变化时推"把总推送压到 ~n_levels 次（非每步）。
    持【训练 venv】引用（eval env 另建·从不调 set_arrival_slack·恒真门=诚实红线·不被本回调触碰）。
    _on_step 在 env.step 之后触发 → 更新在【下一步】env.step 生效（一步延迟·同 penalty/LR 退火·量级无害）。"""

    def __init__(self, sched, venv):
        super().__init__()
        self._sched = sched
        self._venv = venv
        self._last = None

    def _on_step(self):
        self._sched.num_timesteps = self.model.num_timesteps
        v = self._sched.value()
        if self._last is None or v != self._last:      # 仅量化档位变化时推（总 ~n_levels 次·省 deepcopy/跨进程 IPC）
            self._venv.env_method("set_arrival_slack", v)
            self._last = v
        return True


class StableSAC(SAC):
    """SAC + 两种稳化手段（critic/actor 梯度裁剪 + target-Q 值裁剪）—— 修连续臂 critic Q 高估发散（03 L65/L66/L67）。

    诊断（03 L65·主窗口亲核 + 3 维 8-agent 对抗）：sb3 2.3.2 SAC.train()【无任何梯度裁剪/Q 裁剪】→ 策略学进
    高价值区后 critic 自举把 Q 无界放大（致命三件套：函数逼近 × bootstrap × off-policy）。100k 实跑：
    ep_rew_mean 从 -7000 学到 -347（臂确实能学）后 critic_loss→2.9e5、Q→±12000(高估 2400×)、α 被拖着
    暴涨 0.015→2.12、ep_rew_mean 崩回 -6000。**已排除**：非 reward-norm 漂移（亲核 ret_rms 只漂 2.85×、
    归一化奖励恒 ~0.05 小）、非 α 触发（critic 先炸·α 是被拖的放大器）。

    本类提供【两个独立稳化旋钮】，可分别/组合用（默认配置见 make_continuous_safe_model）：
      · **`max_grad_norm`（梯度裁剪·★L65）**：critic/actor optimizer.step **之前**各加 `clip_grad_norm_`。
        ⚠️⚠️【03 L66/L67 实证·勿误信】**梯度裁剪在 Adam 优化器下对 critic Q 高估发散基本无效·lr 才是真旋钮**
        ——Adam 二阶矩归一化使更新≈lr·sign(g)、裁剪只施加【与梯度尺度无关的常数缩放】、对无界 Q 增长【零自适应
        阻尼】（本地 Adam 测试 clip/no-clip 步长比对梯度尺度恒定·warm 多层网络下 ~0.3-0.55·非"不缩步长"）。
        故 `clip_grad_norm_` 在本类里**真正的留存价值 = `error_if_nonfinite=True` 的 NaN/inf 梯度 fail-fast**
        （别静默传染烧算力出 nan 模型）、**不是**有效的发散稳化手段。`max_grad_norm=None` → 退化 vanilla SAC
        （供 A/B 基线复现发散；⚠️此分支无 NaN fail-fast=耦合在裁剪块内·B4·仅供短程复现）。
      · **`target_q_clip`（target-Q 值裁剪·★L67·对症修法）**：在 no_grad 块内把 bootstrap target_q 钳到
        [−target_q_clip, +target_q_clip]。这是【正确作用面】——直接 cap L66 教训②点名的"实际控制量"(回归 label
        无界增长)、与优化器无关（玩具自举回归坐实：vanilla 发散到 287、grad-clip 只降到 62 不 cap、target-Q
        clamp 把 Q 钉在天花板内）。`target_q_clip=None`（默认）→ 不裁剪（保 L65 行为·A/B 对照）。

    train() 其余【逐字复制 sb3 2.3.2 SAC.train()】，仅插裁剪行（梯度裁剪标 ★L65、target-Q 裁剪标 ★L67）；
    __init__ 版本断言防 sb3 升级后用 stale 复制（静默丢新逻辑=最高审核标准不可接受）。
    """

    def __init__(self, *args, max_grad_norm: float | None = 1.0,
                 target_q_clip: float | None = None, **kwargs):
        if _sb3.__version__ != _SB3_TRAIN_REF_VERSION:
            raise RuntimeError(
                f"StableSAC.train() 逐字复制自 sb3 {_SB3_TRAIN_REF_VERSION}，当前装的是 {_sb3.__version__}。"
                "升级 sb3 后须重新核对 SAC.train() 源码并同步更新本类（否则 stale 复制=静默丢新逻辑）。")
        super().__init__(*args, **kwargs)
        self.max_grad_norm = None if max_grad_norm is None else float(max_grad_norm)
        self.target_q_clip = None if target_q_clip is None else float(target_q_clip)   # ★L67 对症旋钮（默认关）
        if self.target_q_clip is not None and not (np.isfinite(self.target_q_clip) and self.target_q_clip > 0.0):
            # 防御做在前（红队 B5）：nan<=0 为 False 会漏过→target_q 静默被 nan 污染（no_grad 内·vanilla 分支 NaN fail-fast 不触发=出 nan 模型烧算力）
            raise ValueError(f"target_q_clip 须为【有限正数】（Q 裁剪天花板·防 nan/inf 静默污染），得 {self.target_q_clip}")

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # ⚠️【逐字复制 sb3 2.3.2 SAC.train()】，仅在 critic/actor optimizer.step 前各加 1 行裁剪（# ★L65）。
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            if self.use_sde:
                self.actor.reset_noise()

            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values
                if self.target_q_clip is not None:                                       # ★L67 target-Q 值裁剪（对症·cap bootstrap Q 高估发散·L66 教训②"实际控制量"）
                    target_q_values = th.clamp(target_q_values, -self.target_q_clip, self.target_q_clip)   # ★L67

            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)
            critic_losses.append(critic_loss.item())  # type: ignore[union-attr]

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            if self.max_grad_norm is not None:                                          # ★L65 critic 梯度裁剪
                th.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm,
                                            error_if_nonfinite=True)                     # ★L65 NaN/inf 梯度 fail-fast(B5g·别静默传染烧算力出 nan 模型)
            self.critic.optimizer.step()

            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            if self.max_grad_norm is not None:                                          # ★L65 actor 梯度裁剪
                th.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm,
                                            error_if_nonfinite=True)                     # ★L65 NaN/inf 梯度 fail-fast(B5g)
            self.actor.optimizer.step()

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))


# ---- LayerNorm critic = 连续臂 SAC critic Q 高估发散的【根因】修法（L67-续2）----
# 文献：**BRO（Nauman 2024·arXiv 2405.16158）**——critic 每 dense 层后插 LayerNorm 是 off-policy 值发散/高估
# 【最有效的单一手段】：归一化激活压住特征无界放大（=Q 高估自举的根），且【压发散却不杀学习】（区别于降 lr/狠 target-Q
# clamp 的"稳但不学"张力）。〔CrossQ(Bhatt ICLR 2024) 是相关【BatchNorm + 去 target 网络】路线·非 LayerNorm·勿混引〕
# sb3 2.3.2 的 create_mlp 不支持归一化层 → 自定义 critic 重建 q_networks；**不碰 SAC.train()**（梯度裁剪/target-Q
# 裁剪/difflib 忠实性 CI 全不受影响·正交叠加）。红队 15k 步实测：vanilla Q 爬 4.5→48.9 仍加速、LayerNorm 升 15.5 后回落 12.7。
class LayerNormContinuousCritic(ContinuousCritic):
    """ContinuousCritic + 每隐层 LayerNorm（根因压 Q 高估发散·L67-续2）。

    q_net = [Linear, LayerNorm, activation, …(net_arch 各隐层), Linear(→1)]（末层不归一·BRO 标准）。
    forward 继承父类（仅遍历 self.q_networks·重建后逐式工作）；critic.optimizer 在 make_critic【之后】才建 → 见新参数。
    """

    def __init__(self, observation_space, action_space, net_arch, features_extractor, features_dim,
                 activation_fn=nn.ReLU, normalize_images=True, n_critics=2, share_features_extractor=True):
        super().__init__(observation_space, action_space, net_arch, features_extractor, features_dim,
                         activation_fn, normalize_images, n_critics, share_features_extractor)
        in_dim = features_dim + get_action_dim(self.action_space)
        self.q_networks = []
        for idx in range(n_critics):
            layers, last = [], in_dim
            for h in net_arch:
                layers += [nn.Linear(last, h), nn.LayerNorm(h), activation_fn()]
                last = h
            layers.append(nn.Linear(last, 1))                    # 末层 →1·不归一（CrossQ/BRO）
            qnet = nn.Sequential(*layers)
            setattr(self, f"qf{idx}", qnet)                      # 覆盖 super 建的同名 module（nn.Module.__setattr__ 重注册到 _modules）
            self.q_networks.append(qnet)


class LayerNormSACPolicy(SACPolicy):
    """SACPolicy，critic 换 LayerNormContinuousCritic（actor 不变·只稳 critic 这一侧·L67-续2）。"""

    def make_critic(self, features_extractor=None):
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return LayerNormContinuousCritic(**critic_kwargs).to(self.device)


def _critic_has_layernorm(model) -> bool:
    """运行时坐实 critic 真含 LayerNorm（caliber/守护·防自定义 policy 接错或回退默认 MLP）。
    用 all()【每个 q_network 都含 LayerNorm】（红队②：any() 弱守护漏"部分 critic 无 LayerNorm"的畸形态）。"""
    qnets = getattr(model.critic, "q_networks", [])
    return len(qnets) > 0 and all(any(isinstance(m, nn.LayerNorm) for m in q.modules()) for q in qnets)


def _make_adamw_critic_optimizer(critic, lr, weight_decay):
    """给 critic 建 AdamW 优化器（完整 BRO·L67-续7 闭 LN-5 权重驱动残余发散通道）。
    **解耦权重衰减只施 Linear 权重(ndim≥2)·不施 LayerNorm affine/bias(ndim≤1)**（标准做法·衰减 norm 尺度会害 LayerNorm）。"""
    decay, no_decay = [], []
    for p in critic.parameters():
        if not p.requires_grad:
            continue
        (decay if p.ndim >= 2 else no_decay).append(p)    # Linear 权重→衰减；bias/LayerNorm affine(1维)→不衰减
    groups = [{"params": decay, "weight_decay": float(weight_decay)},
              {"params": no_decay, "weight_decay": 0.0}]
    return th.optim.AdamW(groups, lr=float(lr))


def load_sac_for_eval(path, device: str = "cpu"):
    """鲁棒载 SAC checkpoint 供【eval 重放】：重建 policy + 只灌 policy 权重·**跳过优化器 state**（L67-续8·二审 BRO-3）。
    避开 BRO（critic AdamW 2 param_group·L67-续7）vs sb3 SAC.load 默认重建 Adam(1 group) 的
    'different number of parameter groups' 崩溃——eval 只用 policy.predict·不碰优化器→丢优化器 state 对 eval【逐位无影响】
    （三档[完整BRO/纯LN/vanilla]实测 predict max|Δ|=0）。✅ 替代 replay_eval 里裸 SAC.load（对 wd>0 checkpoint 鲁棒）。"""
    from stable_baselines3.common.save_util import load_from_zip_file
    data, params, _pyt = load_from_zip_file(path, device=device)
    model = SAC(policy=data["policy_class"], env=None, device=device, _init_setup_model=False)
    model.__dict__.update(data)
    model._setup_model()
    model.set_parameters({"policy": params["policy"]}, exact_match=False, device=device)   # 只灌 policy·跳优化器
    return model


def apply_warmstart(model, venv, warmstart_ckpt: str, device: str = "cpu"):
    """🆕 热启动（`03` L190·探索侧治崩·JSRL 2204.02372 / AWAC 2006.09359 式·user 2026-07-16 拍）：
    把新建 model 的 policy 权重灌成【源 ckpt 的 policy】+ 把源 VecNormalize 的 obs_rms(+ret_rms) 复制进 venv
    → 新策略在【源同一归一化坐标系】里起步 = step0 行为 ≈ 源策略（把崩种子放进好盆地·治"从没探索到好行为"）。

    设计红线（基础设施最高标准·CLAUDE §2）：
      · **只灌 policy·跳优化器**（fresh Adam·同 load_sac_for_eval:436 范式·热启动标准做法）；log_std 重置回 maker in-box 初值（不灌源 σ·保 F1）。
      · **必须同时复制源 vecnorm obs_rms**：策略吃归一化 obs·只灌 policy 不灌 obs_rms=新 venv 从 mean0/std1 起步=喂错分布=热启动前若干步策略行为错乱=白热启动（[[verify-full-plumbing-chain-end-to-end]] 命门）。
      · **结构守卫（对抗审 wy3rlm90p HIGH-1）**：源 policy state_dict 键集须 == 目标（防误指 SAC 存档=键集不相交→set_parameters(exact_match=False) 静默不灌=白热启动无报错）。
      · **obs 维守卫**：源 obs 维须 == 目标 venv obs 维（augment_rho→27/34）·不匹配 fail-fast。
      · **⚠️ 不查语义配置**：shield/goal_cone/colregs_weight 等【不改 obs 维】的配置本函数【不校验】（改 obs 维的 augment_rho 才被守卫抓）→**源须与本 run 同 shield/colregs 配置=调用方/烧前预检责任**（对抗审 W3.5 MEDIUM·别误配串门）。
      · **venv 保持 training=True**（stats 随训练续更·源 count 大→起步稳在源 stats）；不冻结。
    ⚠️ 调用方保证 warmstart_ckpt=None/'' 时【整块不调】= bit-identical（见 make_continuous_safe_ppo_model ⑤）。
    """
    import os as _os, copy as _copy, pickle as _pickle
    from stable_baselines3.common.save_util import load_from_zip_file
    zip_path, vn_path = warmstart_ckpt + ".zip", warmstart_ckpt + "_vecnorm.pkl"
    if not (_os.path.exists(zip_path) and _os.path.exists(vn_path)):
        raise FileNotFoundError(f"🔒 热启动源 ckpt 缺 .zip 或 _vecnorm.pkl: {warmstart_ckpt}（须两者同在）")
    # ① 灌 policy 权重（只 policy·跳优化器）
    _data, _params, _pyt = load_from_zip_file(zip_path, device=device)
    if "policy" not in _params:
        raise ValueError(f"🔒 热启动源 ckpt 无 'policy' 参数键（非标准 SB3 policy ckpt?）: {zip_path}")
    # 🔴 结构守卫（对抗审 wy3rlm90p HIGH-1·实测坐实）：set_parameters(exact_match=False) 底层 load_state_dict(strict=False)→
    #   源与目标 policy 键集【不相交】时 torch 静默不灌一个权重、不 raise（典型=误指 SAC 存档·SAC 是默认臂产同名 .zip+_vecnorm.pkl+也有 'policy' 键）
    #   → 从随机初始化训练却自以为热启动=白热启动无报错。故显式校键集相等，不等即 fail-fast。
    _src_keys = set(_params["policy"].keys())
    _tgt_keys = set(model.policy.state_dict().keys())
    if _src_keys != _tgt_keys:
        _cls = getattr(_data, "get", lambda *a: None)("policy_class") if isinstance(_data, dict) else None
        raise ValueError(f"🔒 热启动源 policy 结构与目标不匹配（键集不等·set_parameters 会静默不灌=白热启动）: "
                         f"目标独有键 {sorted(_tgt_keys - _src_keys)[:4]} 源独有键 {sorted(_src_keys - _tgt_keys)[:4]}"
                         f"（源 policy_class={_cls}·常见=误指 SAC/其它臂存档·热启动源须同款连续 PPO ckpt）")
    # 🔴 F1 守卫交互（端到端 smoke 抓·`03` L190）：源 ckpt 的【已训 log_std】(σ~0.062/0.030) 可能略超动作箱半宽(±0.048/±0.018)
    #   → caliber assert(F1·构造期 σ 须在箱内)会 fire。设计决策=**热启动只灌【均值策略】(=好盆地所在)·log_std 重置回 maker 的 in-box 初值**
    #   （标准迁移做法·给崩种子受控的 fresh 探索·非继承源收敛 σ·保 F1 守卫不被削弱·均值策略=真正要传的好行为）。
    _saved_log_std = model.policy.log_std.data.clone() if hasattr(model.policy, "log_std") else None
    model.set_parameters({"policy": _params["policy"]}, exact_match=False, device=device)
    if _saved_log_std is not None and hasattr(model.policy, "log_std"):
        with th.no_grad():
            model.policy.log_std.data.copy_(_saved_log_std)          # 恢复 maker 的 in-box log_std_init（灌均值·不灌源探索 σ）
    # ② 复制源 VecNormalize obs_rms(+ret_rms) 进 venv（直接 pickle 载·不 wrap 活 venv 防干扰）
    if not isinstance(venv, VecNormalize):
        raise AssertionError("🔒 热启动须 venv 为 VecNormalize（obs 归一化坐标系复制的载体）")
    with open(vn_path, "rb") as _f:
        _src_vn = _pickle.load(_f)                            # SB3 VecNormalize.save 后 pickle·venv 未随存→直接取 stats
    _src_shape = tuple(np.asarray(_src_vn.obs_rms.mean).shape)
    _tgt_shape = tuple(np.asarray(venv.obs_rms.mean).shape)
    if _src_shape != _tgt_shape:                              # obs 维守卫（augment_rho 27/34 不匹配 → fail-fast；⚠️shield/colregs 不改 obs 维=此处抓不到=预检责任）
        raise ValueError(f"🔒 热启动 vecnorm obs 维不匹配: 源 {_src_shape} ≠ 目标 {_tgt_shape}（源须与本 run 同 augment_rho 配置·shield/colregs 语义配置须调用方保证）")
    venv.obs_rms = _copy.deepcopy(_src_vn.obs_rms)            # 复制源 obs 归一化 stats（策略在源坐标系起步）
    if getattr(venv, "norm_reward", False) and getattr(_src_vn, "ret_rms", None) is not None:
        venv.ret_rms = _copy.deepcopy(_src_vn.ret_rms)       # 复制源 return 归一化 stats（value/advantage 尺度一致）
    return model, venv


def _probe_colregs_weight(scenario_pool=None, paths=None,
                          colregs_weight: float = CONTINUOUS_SAFE_COLREGS_WEIGHT) -> float:
    """构造一个 probe ContinuousProjectionEnv 读回其 reward_fn.colregs_weight（运行时坐实【意图 colregs_weight 是否真落地】·silent-no-op 防御，防未来误改/漏接线）。"""
    if scenario_pool:
        sc, pp = scenario_pool[0]
    elif paths:
        sc, pp = load_scenario_pool([paths[0]])[0]
    else:
        raise ValueError("需 scenario_pool 或 paths")
    probe = ContinuousProjectionEnv(sc, pp, colregs_weight=colregs_weight)
    return float(probe.env.reward_fn.colregs_weight)


def make_continuous_safe_model(scenario_pool=None, *, paths=None, seed: int = 0, n_envs: int = 1,
                               use_vecnorm: bool = True, gamma: float = 0.99,
                               norm_reward: bool = True, subproc: bool = False,
                               learning_starts: int = 5000,
                               max_grad_norm: float | None = 1.0, learning_rate: float = 1e-4,
                               target_q_clip: float | None = None,
                               use_critic_layernorm: bool = False, n_critics: int = 2, tau: float = 0.005,
                               critic_weight_decay: float = 0.0,
                               well_shaping_weight: float = 0.0, shaping_radius: float = 500.0,
                               xtrack_weight: float = 0.0, xtrack_radius: float = 80.0,   # 对症 横向进带势（`03` L88·显式具名在 **sac_kwargs 前）
                               park_weight: float = 0.0, park_radius: float = 400.0, park_v_target: float = 4.0,   # 想法B 终端保速势（`03` L109·连续臂专属）
                               c_step: float = 0.0,   # 修法C 每步生存成本（`03` L123·连续臂专属·非PBRS·默认 0=关=逐位等价）
                               c_dwell: float = 0.0, w_dwell: float = 90.0, h_dwell: float = 0.52, dwell_radius: float = 250.0, b_dwell: float = 0.0,   # r_dwell 入库赤字滞留成本（`03` L161/L162·连续臂专属·非PBRS·默认关=逐位等价·透传 ContinuousProjectionEnv）
                               alias_weight: float = 0.0,   # 动作混叠惩罚 w（Markgraf 式20·`03` L97·默认 0=关=逐位等价）
                               rate_weight: float = 0.0,    # action-rate 平滑惩罚 w（治 bang-bang·`03` L98·默认 0=关=逐位等价）
                               rate_dock: float | None = None,   # 🆕 第二条腿 rank1（`03` L173）：泊位精修门控治抖·默认 None=off=bit-identical·透传 ContinuousProjectionEnv
                               colregs_weight: float = CONTINUOUS_SAFE_COLREGS_WEIGHT,   # r_colregs 权重·默认0.0=丢r_colregs现状·A/B=1.0复活Meyer式26
                               shield: bool = True,   # 🆕 P0(L146)：SE-RL 盾开关·默认 True=有盾=bit-identical·False=连续无盾臂（透传 ContinuousProjectionEnv）
                               goal_cone_half: float | None = None, goal_v_floor: float = 2.0,   # 🆕 ρ0 朝目标锥（PhaseC·`03` L145/L147）·默认 None=关=bit-identical（透传 ContinuousProjectionEnv→proj·仅锥开时用 v_floor）
                               augment_rho: bool = False,   # 🆕 腿1(L150/L152)：态势感知观测增广（透传 ContinuousProjectionEnv·默认 False=27维=bit-identical·True=34维=ρ one-hot+give_way_dir）
                               arrival_heading_slack: float = 0.0,   # 🆕 B1(`03` L153)：到达门朝向容差课程 slack 起始值（rad·透传 ContinuousProjectionEnv→USVEnv→term_checker）·默认 0.0=真门=bit-identical·>0=训练放宽（退火·评估恒0）
                               start_frac: float = 1.0, start_v=None,   # 🆕 逆向起点课程（方案C-B·`03` L181·Florensa 2017）：起点系数（透传 ContinuousProjectionEnv→USVEnv.set_start_frac）·默认 (1.0,None)=真起点=bit-identical·<1=训练时 ego 生更靠门（评估恒1）·start_v=课程重生速度
                               goal_ignore_orientation: bool = False,   # 🆕 L185(user 2026-07-13)：训练目标去朝向硬门→1_goal 只判位置到达区域（透传 ContinuousProjectionEnv→USVEnv→term_checker）·默认 False=严格真门=bit-identical·True=位置-only（两阶段stage-1·治崩种子绕圈）
                               c_reach: float = C_REACH, dock_radius: float = 0.0, v_dock: float = V_LOW,   # 🆕 第二条腿修法（`03` L172·连续臂专属·默认关 bit-identical·透传 ContinuousProjectionEnv→USVEnv→RewardFunction）
                               **sac_kwargs):
    """构造 Continuous-safe SAC 模型（ContinuousProjectionEnv + SAC + 同款 VecNormalize）。返回 `(model, venv)`。

    scenario_pool : 预加载场景池 [(sc,pp),…]（DummyVecEnv）；或传 paths（xml 路径）。
    n_envs        : 并行 env 数（SAC off-policy 默认 1；Tier-3 按吞吐调）。
    norm_reward   : VecNormalize 是否归一化 reward（默认 True 依 D38；off-policy 张力见模块 docstring）。
    learning_starts : SAC 开学前纯随机 warmup 步数（L63 Fix③：SAC 默认 100<1 局→buffer 全早死负样本→critic 早钉退化盆；
                      默认 ↑5000≈30 局·与 Fix② 动作箱配套；可经 param/sac_kwargs 覆盖供 A/B）。
    max_grad_norm : critic/actor 梯度裁剪范数（L65·StableSAC）。⚠️【L66/L67 实证】梯度裁剪在 Adam 下对 Q 高估
                    发散【基本无效·lr 才是真旋钮】——默认 1.0 留存价值=`error_if_nonfinite` NaN fail-fast、非有效稳化；
                    **None → 退化 vanilla SAC**（A/B 基线复现发散·此分支无 NaN fail-fast）。
    learning_rate : SAC 学习率（L65：默认 3e-4→**1e-4**·稳化自举·可调供 A/B）。
    target_q_clip : ⭐**target-Q 值裁剪天花板**（L67·对症修法）。None（默认）→ 不裁剪（保 L65 行为）；设【有限正数】→ 把
                    bootstrap target_q 钳到 [−v,+v]=直接 cap Q 高估发散的【实际控制量】（对作用面·与优化器无关·
                    L66 教训②）。⚠️**值按【VecNormalize 归一化后】reward 尺度定**（合法 |Q| ~5-15·随 ret_std 变·发散到
                    ~12000；**非 raw 尺度 ~6000**——填 raw 值=永不咬=静默 no-op·红队 B③footgun）→ 取 ~50-1000 兜爆炸不
                    伤合法值·供 A/B 标定（STEP4E_TARGET_Q_CLIP）。⚠️太低（如 ≲15）会咬合法 Q→可能"稳但学不动"（如 L66 gclr 臂）。
    use_critic_layernorm : ⭐**critic 每隐层 LayerNorm**（L67-续2·**根因**修 Q 高估发散·BRO Nauman 2024）。False（默认）→ 原
                    MLP critic（A/B 基线）；True → LayerNormSACPolicy（归一化激活压特征无界放大=Q 高估的根·压发散却不杀学习）。
                    **推荐配置 = True + learning_rate 回 3e-4**（LayerNorm 让高 lr 既稳又能学·解 L66"稳vs学"张力）。STEP4E_CRITIC_LAYERNORM。
    n_critics     : critic 数（REDQ-lite·默认 2=sb3）。↑（如 5）→ clipped-min 压乐观高估·叠加 LayerNorm。STEP4E_N_CRITICS。
    tau           : 目标网络 polyak 平滑系数（默认 0.005=sb3）。↓→ 慢化移动目标·稳自举。STEP4E_TAU。
    critic_weight_decay : ⭐**critic AdamW 解耦权重衰减**（L67-续7·**完整 BRO** = LayerNorm + weight decay·闭深核 LN-5 指出的
                    "权重驱动残余发散通道"：LN 使特征有界但末层 Q 头/affine 权重仍可绕过归一化增长）。>0 → critic 优化器换 AdamW、
                    衰减只施 Linear 权重(不碰 LN affine/bias)；actor 不动。默认 0（=纯 LayerNorm·A/B 基线）。荐 ~1e-4~1e-2·STEP4E_CRITIC_WD。
    sac_kwargs    : 覆盖 SAC 默认（buffer_size/batch_size/…，标定可调）。**禁经此偷传 ent_coef**（见下断言）。

    强制口径（防 footgun）：colregs_weight 默认 0.0(可经 STEP4E_COLREGS_WEIGHT 覆盖·A/B 复活 r_colregs) + probe 断言；ent_coef 不传 → SAC 'auto' + 断言；
    net_arch [64,64] + gamma 同进 SAC 与 VecNormalize（单一真相源）；use_critic_layernorm=True 时接线守护 critic 真含 LayerNorm。
    """
    for _forbid in ("ent_coef", "target_entropy"):   # 自带最大熵机制不接受外覆（含目标熵 target_entropy，红队 LOW）
        if _forbid in sac_kwargs:
            raise ValueError(f"Continuous-safe SAC 不接受显式 {_forbid}（D38：SAC 自带最大熵 'auto' + 目标熵 -dim(a)，"
                             "别搬离散 ent_coef=0.01）。如确需手调熵温，请改本模块并记 03 决策。")
    if not (0.0 < tau <= 1.0):                        # 防御做在前（红队③：sb3 不校验 tau·docstring 既宣传为 knob 就该守）
        raise ValueError(f"tau 须 ∈(0,1]（目标网络 polyak 平滑系数），得 {tau}")
    if n_critics < 1:
        raise ValueError(f"n_critics 须 ≥1，得 {n_critics}")
    # ① 运行时坐实 colregs_weight 意图真落地（D37-B / D40 #2 footgun 防御）；显式比【意图 colregs_weight】（默认0.0=现状），抓 silent no-op
    cw = _probe_colregs_weight(scenario_pool=scenario_pool, paths=paths, colregs_weight=colregs_weight)
    if cw != colregs_weight:
        raise RuntimeError(f"Continuous-safe colregs_weight 接线未落地: probe 读回 {cw}≠意图 {colregs_weight}(silent no-op)")

    # ② VecEnv：ContinuousProjectionEnv（连续投影盾），env_kwargs colregs_weight=意图值（默认0.0·可A/B覆盖）+ 修法A 进门势透传（`03` L81·gamma 同源）
    venv = make_vec_env(scenario_pool=scenario_pool, paths=paths, n_envs=n_envs,
                        env_cls=ContinuousProjectionEnv,
                        env_kwargs=dict(colregs_weight=colregs_weight, shield=shield,   # 🆕 P0：SE-RL 盾开关透传（默认 True=bit-identical）
                                        goal_cone_half=goal_cone_half, goal_v_floor=goal_v_floor,   # 🆕 ρ0 朝目标锥透传（PhaseC·`03` L147·默认 None=关=bit-identical）
                                        augment_rho=augment_rho,   # 🆕 腿1(L150/L152)：态势感知观测增广透传（默认 False=bit-identical·venv obs_space 据此自动 27/34→VecNorm obs_rms 同形→SB3 policy 网自动 sizing）
                                        arrival_heading_slack=arrival_heading_slack,   # 🆕 B1(`03` L153)：到达门朝向容差 slack 起始值透传（默认 0.0=真门=bit-identical·退火经 env_method set_arrival_slack 覆盖）
                                        start_frac=start_frac, start_v=start_v,   # 🆕 逆向起点课程（`03` L181）：起点系数透传（默认 (1.0,None)=真起点=bit-identical·退火经 env_method set_start_frac 覆盖·评估 fac 恒不传=真起点）
                                        goal_ignore_orientation=goal_ignore_orientation,   # 🆕 L185：去朝向硬门透传（默认 False=严格真门=bit-identical·构造期常量·MultiScenarioEnv reset 重建经 env_kwargs 透传继承）
                                        gamma=gamma, well_shaping_weight=well_shaping_weight,
                                        shaping_radius=shaping_radius,
                                        xtrack_weight=xtrack_weight, xtrack_radius=xtrack_radius,
                                        park_weight=park_weight, park_radius=park_radius, park_v_target=park_v_target,   # 想法B 终端保速势（`03` L109）
                                        c_step=c_step,   # 修法C 每步生存成本（`03` L123·连续臂专属·非PBRS）
                                        c_dwell=c_dwell, w_dwell=w_dwell, h_dwell=h_dwell, dwell_radius=dwell_radius, b_dwell=b_dwell,   # r_dwell 入库赤字滞留成本（`03` L161/L162·连续臂专属·非PBRS）
                                        c_reach=c_reach, dock_radius=dock_radius, v_dock=v_dock,   # 🆕 第二条腿修法透传（`03` L172·连续臂专属·默认关 bit-identical）
                                        alias_weight=alias_weight, rate_weight=rate_weight, rate_dock=rate_dock),   # 动作混叠(Markgraf 式20·L97) + action-rate 平滑(L98) + rank1 泊位门控治抖(L173)
                        subproc=subproc, seed=seed)
    # ③ 同款 VecNormalize（obs+clip 单一真相源 train.VECNORM_KWARGS；norm_reward 可调；gamma 同进）
    if use_vecnorm:
        vn = dict(VECNORM_KWARGS)
        vn["norm_reward"] = norm_reward                # 仅 reward-norm 可调；obs-norm/clip 与离散逐式一致（四方平价）
        venv = VecNormalize(venv, gamma=gamma, **vn)

    # ④ StableSAC（=SAC+梯度裁剪·L65；不传 ent_coef → 'auto'；net [64,64]；gamma 与 VecNormalize 一致）
    cfg = dict(policy_kwargs=dict(net_arch=POLICY_NET_ARCH, n_critics=n_critics), seed=seed, gamma=gamma,
               learning_starts=learning_starts, learning_rate=learning_rate, tau=tau, verbose=0)
    #   L63 Fix③ learning_starts ↑5000(~30 局 warmup)；L65 Fix learning_rate ↓1e-4 + 下方 max_grad_norm 稳化自举·可经 param/sac_kwargs 覆盖供 A/B
    #   L67-续2：n_critics(REDQ-lite 压高估)/tau(目标平滑)·use_critic_layernorm(根因·LayerNorm critic)·均 A/B knob
    cfg.update(sac_kwargs)
    policy = LayerNormSACPolicy if use_critic_layernorm else "MlpPolicy"
    model = StableSAC(policy, venv, max_grad_norm=max_grad_norm, target_q_clip=target_q_clip, **cfg)
    if use_critic_layernorm and not _critic_has_layernorm(model):     # 接线守护：自定义 policy 接错/回退默认 MLP 即 fail-fast（别静默丢 LayerNorm 烧算力）
        raise RuntimeError("use_critic_layernorm=True 但 critic 无 LayerNorm（LayerNormSACPolicy 接线失败）")
    if critic_weight_decay > 0.0:                                     # L67-续7：完整 BRO = LayerNorm + critic AdamW 解耦权重衰减（闭 LN-5 权重驱动残余发散通道）
        model.critic.optimizer = _make_adamw_critic_optimizer(model.critic, learning_rate, critic_weight_decay)
    if str(model.ent_coef) != "auto":               # 断言自带最大熵未被覆盖（D38）
        raise RuntimeError(f"SAC ent_coef={model.ent_coef!r}≠'auto'（自带最大熵被覆盖、违反 D38）")
    # ⑤ 过门强制：构造时即跑口径自检（net_arch/gamma/colregs/VecNorm 全守）——不依赖 Node C/Tier-3 调用方"记得调"
    #    （双 agent MEDIUM：policy_kwargs 经 sac_kwargs→cfg.update 可静默覆盖 net_arch，唯一拦截原是事后 caliber、易被跳过）
    if use_vecnorm:                                  # caliber 要求 VecNormalize；use_vecnorm=False(仅测试变异用)时跳过自调
        assert_continuous_safe_caliber(model, venv, colregs_weight=colregs_weight)
    return model, venv


def assert_continuous_safe_caliber(model, venv, colregs_weight: float = CONTINUOUS_SAFE_COLREGS_WEIGHT) -> None:
    """Continuous-safe 训练侧口径自检（net/算法/归一化/colregs_weight/ent_coef/gamma）。不符即 raise。

    （评估侧「四方同 ego 轨迹对拍」= Node C 接评估时立，D40 035 机制性建议；本函数是训练侧那一半。）
    """
    if not isinstance(model, SAC):
        raise AssertionError(f"Continuous-safe 算法须 SAC，得 {type(model).__name__}")
    if not isinstance(model, StableSAC) or not hasattr(model, "max_grad_norm"):   # L65 Fix：须含梯度裁剪的 StableSAC（防回归 vanilla SAC=Q 高估发散）
        raise AssertionError(f"Continuous-safe 须 StableSAC（含 critic/actor 梯度裁剪·L65），得 {type(model).__name__}")
    if str(model.ent_coef) != "auto":
        raise AssertionError(f"SAC ent_coef 须 'auto'（自带最大熵，D38），得 {model.ent_coef!r}")
    arch = model.policy_kwargs.get("net_arch")
    if list(arch) != list(POLICY_NET_ARCH):
        raise AssertionError(f"net_arch 须 {POLICY_NET_ARCH}（忠实 Krasowski），得 {arch}")
    if isinstance(model.policy, LayerNormSACPolicy) != _critic_has_layernorm(model):   # L67-续2：policy 类型与 critic LayerNorm 必一致（防接错/静默回退默认 MLP）
        raise AssertionError(f"critic LayerNorm 一致性破：policy={type(model.policy).__name__} 但 critic_has_layernorm={_critic_has_layernorm(model)}")
    if not isinstance(venv, VecNormalize):
        raise AssertionError("Continuous-safe 须 VecNormalize（同款 obs 归一化，四方平价）")
    if not venv.norm_obs:
        raise AssertionError("VecNormalize.norm_obs 须 True（obs 归一化是停船配方命门，D22）")
    if float(venv.clip_obs) != float(VECNORM_KWARGS["clip_obs"]):   # obs 归一化口径完整性（含 clip，红队 LOW）
        raise AssertionError(f"clip_obs 须 {VECNORM_KWARGS['clip_obs']}（obs 归一化口径，四方平价），得 {venv.clip_obs}")
    if abs(float(venv.gamma) - float(model.gamma)) > 1e-12:
        raise AssertionError(f"gamma 须同进 VecNormalize({venv.gamma}) 与 SAC({model.gamma})（L21 MINOR②）")
    # colregs_weight：get_attr('env') 对 Dummy/Subproc 都生效（MultiScenarioEnv.env property→USVEnv；subproc 实测可读）。
    # 真 env 的 0.0 由 make 时 _probe 把关，此处是补全的口径自检（含 subproc，堵 B-MEDIUM 内省盲区）。
    for e in venv.venv.get_attr("env"):
        cw = float(e.reward_fn.colregs_weight)
        if cw != colregs_weight:
            raise AssertionError(f"colregs_weight 接线未落地: env 读回 {cw}≠意图 {colregs_weight}")
        if abs(float(e.reward_fn.gamma) - float(model.gamma)) > 1e-12:   # 修法A PBRS 第三处 gamma 同源（破则 Ng 不变性失效·`03` L81 BLOCKER 风险）
            raise AssertionError(f"reward_fn.gamma({e.reward_fn.gamma}) 须 == model.gamma({model.gamma})（修法A PBRS 须与 trainer/VecNorm 同源 _GAMMA）")
    # 动作箱口径（根因修 03 L63 Fix②）：连续臂 RL 动作箱须 = Krasowski 正常操作 ±0.048/±0.018，
    # 非满程 ±a_max=0.24/±w_max=0.03（满程是停船墙根+四方公平 confound；满程只给紧急控制器）。
    hi = np.asarray(venv.action_space.high, dtype=float)
    lo = np.asarray(venv.action_space.low, dtype=float)
    exp_hi = np.array([A_NORMAL_ACCEL_MAX, A_NORMAL_OMEGA_MAX])
    if not (np.allclose(hi, exp_hi) and np.allclose(lo, -exp_hi)):
        raise AssertionError(
            f"连续臂动作箱须 ±[{A_NORMAL_ACCEL_MAX}, {A_NORMAL_OMEGA_MAX}]（Krasowski 正常操作·L63 Fix②·"
            f"非满程 ±a_max），得 high={hi} low={lo}")


def train_continuous_safe(scenario_pool=None, *, paths=None, seed: int = 0,
                          total_timesteps: int = TOTAL_TIMESTEPS, **make_kwargs):
    """⚠️ **Tier 3（待 user 拍板，烧算力 + 错峰 T-ITS）**：实跑 Continuous-safe SAC 训练。返回 `(model, venv)`。

    Node B/C **不在过门时调用此函数**（3M 是真训练，非脚手架验证）。Tier-3 拍板后由 Node C 四方 harness 调用。
    eval 必须用训练后 venv 的 obs 归一化（同离散 make_obs_transform 思路，Node C 接）。
    """
    model, venv = make_continuous_safe_model(scenario_pool=scenario_pool, paths=paths, seed=seed, **make_kwargs)
    model.learn(total_timesteps=total_timesteps)
    return model, venv


# ====================================================================================================
# 节点2 / L67-续3：连续 PPO 入口 = SAC 岔路的【并行 hedge】（on-policy·无 off-policy critic 发散）
# ====================================================================================================
# 动机（L66④B / L67 推荐②）：SAC 连续臂遇 critic Q 高估【off-policy 自举发散】；on-policy PPO 根本无 replay buffer
# 自举→无【此】发散机制；且与离散三方同族（sb3 PPO·离散用 sb3-contrib MaskablePPO）→ 四方全 PPO【撤掉 SAC-vs-PPO
# 算法混淆变量】、"连续投影 vs 离散掩码"因果主张更干净（=真卖点）。**此论据独立于 SAC 是否修好。**
# ⚠️ 边界（红队/二审 D6-F6·勿 over-claim）：① PPO 能否学好【连续+昂贵投影盾】= 不同 regime·未实测（on-policy 采样
#    效率/fps 须实测·~29fps 盾下 3M wall-clock 待测）；② **action aliasing 未解**——盾投影使多 u_raw→同 u_safe·PPO 的
#    log_prob/ratio 在 u_raw 上算、advantage 来自 u_safe→pre-image 内 advantage 平→对"被投影掉方向"无区分梯度。
#    on-policy 避开 off-policy 发散【≠】避开 aliasing（不同机制·L64 已区分 aliasing≠到达失败）→ A/B 须观察·必要时上
#    Markgraf 式(20) 惩罚 −w·‖u_desired−u_safe‖²。
# 🔒 pre-fix（二审 D6-F5）：PPO 作【opt-in 算法·默认仍 SAC】——不改现有四方 metadata/caliber 的 SAC 专属硬编（=避免假
#    论文声明）；A/B 用诊断 TAG。待 A/B 定主臂后再做"全 PPO"metadata 重写（届时一并改 run_metadata/run_step4e 叙述）。

def make_continuous_safe_ppo_model(scenario_pool=None, *, paths=None, seed: int = 0, n_envs: int = 8,
                                   use_vecnorm: bool = True, gamma: float = 0.99,
                                   norm_reward: bool = True, subproc: bool = True,
                                   ent_coef: float = ENT_COEF, log_std_init: float | None = None,
                                   well_shaping_weight: float = 0.0, shaping_radius: float = 500.0,
                                   xtrack_weight: float = 0.0, xtrack_radius: float = 80.0,
                                   park_weight: float = 0.0, park_radius: float = 400.0, park_v_target: float = 4.0,   # 想法B 终端保速势（`03` L109·连续臂专属）
                                   c_step: float = 0.0,   # 修法C 每步生存成本（`03` L123·连续臂专属·非PBRS·默认 0=关=逐位等价）
                                   c_dwell: float = 0.0, w_dwell: float = 90.0, h_dwell: float = 0.52, dwell_radius: float = 250.0, b_dwell: float = 0.0,   # r_dwell 入库赤字滞留成本（`03` L161/L162·连续臂专属·非PBRS·默认关=逐位等价·透传 ContinuousProjectionEnv）
                                   alias_weight: float = 0.0,   # 动作混叠惩罚 w（Markgraf 式20·`03` L97·默认 0=关=逐位等价）
                                   rate_weight: float = 0.0,    # action-rate 平滑惩罚 w（治 bang-bang·`03` L98·默认 0=关=逐位等价）
                               rate_dock: float | None = None,   # 🆕 第二条腿 rank1（`03` L173）：泊位精修门控治抖·默认 None=off=bit-identical·透传 ContinuousProjectionEnv
                                   colregs_weight: float = CONTINUOUS_SAFE_COLREGS_WEIGHT,   # r_colregs 权重·默认0.0=丢r_colregs现状·A/B=1.0复活Meyer式26
                                   shield: bool = True,   # 🆕 P0(L146)：SE-RL 盾开关·默认 True=有盾=bit-identical·False=连续无盾臂（透传 ContinuousProjectionEnv）
                                   goal_cone_half: float | None = None, goal_v_floor: float = 2.0,   # 🆕 ρ0 朝目标锥（PhaseC·`03` L145/L147）·默认 None=关=bit-identical（透传 ContinuousProjectionEnv→proj·仅锥开时用 v_floor）
                                   augment_rho: bool = False,   # 🆕 腿1(L150/L152)：态势感知观测增广（透传 ContinuousProjectionEnv·默认 False=27维=bit-identical·True=34维=ρ one-hot+give_way_dir）
                                   arrival_heading_slack: float = 0.0,   # 🆕 B1(`03` L153)：到达门朝向容差课程 slack 起始值（rad·透传 ContinuousProjectionEnv→USVEnv→term_checker）·默认 0.0=真门=bit-identical·>0=训练放宽（退火·评估恒0）
                               start_frac: float = 1.0, start_v=None,   # 🆕 逆向起点课程（方案C-B·`03` L181·Florensa 2017）：起点系数（透传 ContinuousProjectionEnv→USVEnv.set_start_frac）·默认 (1.0,None)=真起点=bit-identical·<1=训练时 ego 生更靠门（评估恒1）·start_v=课程重生速度
                                   goal_ignore_orientation: bool = False,   # 🆕 L185(user 2026-07-13)：训练目标去朝向硬门→1_goal 只判位置到达区域（透传 ContinuousProjectionEnv→USVEnv→term_checker）·默认 False=严格真门=bit-identical·True=位置-only（两阶段stage-1·治崩种子绕圈）
                                   c_reach: float = C_REACH, dock_radius: float = 0.0, v_dock: float = V_LOW,   # 🆕 第二条腿修法（`03` L172·连续臂专属·默认关 bit-identical·透传 ContinuousProjectionEnv→USVEnv→RewardFunction）
                                   warmstart_ckpt: str | None = None,   # 🆕 L190（user 2026-07-16）：热启动源 ckpt 路径（不含扩展名·须 <base>.zip+<base>_vecnorm.pkl）·默认 None=不热启动=bit-identical·探索侧治崩（JSRL/AWAC 式·灌源 policy+源 vecnorm stats）
                                   **ppo_kwargs):   # 对症 横向进带势（`03` L88·显式具名在 **ppo_kwargs 前）
    """构造 Continuous-safe 【PPO】模型（ContinuousProjectionEnv + sb3 PPO）。返回 `(model, venv)`。

    同口径（与 SAC 臂 + 离散三方平价）：同 ContinuousProjectionEnv 物理/投影盾 + 同 VecNormalize（train.VECNORM_KWARGS）
    + net_arch [64,64] + colregs_weight 默认 0.0(可经 STEP4E_COLREGS_WEIGHT 覆盖·A/B 复活 r_colregs)（丢 r_colregs·档位A 经验性）+ 动作箱 ±0.048（Krasowski 正常操作·L63 Fix②）。
    **ent_coef 默认 = 离散臂 ENT_COEF(0.01)**（四方全 PPO 同 ent_coef【数值】·唯一有意差异 = 连续 vs 离散动作 = 真卖点）。
      ⚠️【红队·勿 over-claim】"同数值"≠"同探索【效力】"：离散是类别熵（有界 ≤ln49）、连续是高斯熵（无界·含 σ 自由参数）→
      同系数对两者尺度/方向不同·标定后连续 PPO 可能需各自调 ent_coef（实测 0.01 偏弱·σ 几乎不动 ~1.0 远超动作箱）。
    n_envs : on-policy PPO 靠并行 rollout 提采样效率（默认 8·同离散；SubprocVecEnv；run_step4e 用 N_ENVS_PPO 分派）。
    ppo_kwargs : 覆盖 PPO 默认（n_steps/batch_size/gae_lambda/clip_range/n_epochs/learning_rate…·标定可调）。
    """
    # ① 运行时坐实 colregs_weight 意图真落地（同 SAC 臂·D37-B footgun 防御）·抓 silent no-op
    cw = _probe_colregs_weight(scenario_pool=scenario_pool, paths=paths, colregs_weight=colregs_weight)
    if cw != colregs_weight:
        raise RuntimeError(f"Continuous-safe colregs_weight 接线未落地: probe 读回 {cw}≠意图 {colregs_weight}(silent no-op)")
    # ② VecEnv：ContinuousProjectionEnv（连续投影盾·env_kwargs colregs_weight=意图值(默认0.0·可A/B覆盖) + 修法A 进门势透传·`03` L81·gamma 同源）
    venv = make_vec_env(scenario_pool=scenario_pool, paths=paths, n_envs=n_envs,
                        env_cls=ContinuousProjectionEnv,
                        env_kwargs=dict(colregs_weight=colregs_weight, shield=shield,   # 🆕 P0：SE-RL 盾开关透传（默认 True=bit-identical）
                                        goal_cone_half=goal_cone_half, goal_v_floor=goal_v_floor,   # 🆕 ρ0 朝目标锥透传（PhaseC·`03` L147·默认 None=关=bit-identical）
                                        augment_rho=augment_rho,   # 🆕 腿1(L150/L152)：态势感知观测增广透传（默认 False=bit-identical·venv obs_space 据此自动 27/34→VecNorm obs_rms 同形→SB3 policy 网自动 sizing）
                                        arrival_heading_slack=arrival_heading_slack,   # 🆕 B1(`03` L153)：到达门朝向容差 slack 起始值透传（默认 0.0=真门=bit-identical·退火经 env_method set_arrival_slack 覆盖）
                                        start_frac=start_frac, start_v=start_v,   # 🆕 逆向起点课程（`03` L181）：起点系数透传（默认 (1.0,None)=真起点=bit-identical·退火经 env_method set_start_frac 覆盖·评估 fac 恒不传=真起点）
                                        goal_ignore_orientation=goal_ignore_orientation,   # 🆕 L185：去朝向硬门透传（默认 False=严格真门=bit-identical·构造期常量·MultiScenarioEnv reset 重建经 env_kwargs 透传继承）
                                        gamma=gamma, well_shaping_weight=well_shaping_weight,
                                        shaping_radius=shaping_radius,
                                        xtrack_weight=xtrack_weight, xtrack_radius=xtrack_radius,
                                        park_weight=park_weight, park_radius=park_radius, park_v_target=park_v_target,   # 想法B 终端保速势（`03` L109）
                                        c_step=c_step,   # 修法C 每步生存成本（`03` L123·连续臂专属·非PBRS）
                                        c_dwell=c_dwell, w_dwell=w_dwell, h_dwell=h_dwell, dwell_radius=dwell_radius, b_dwell=b_dwell,   # r_dwell 入库赤字滞留成本（`03` L161/L162·连续臂专属·非PBRS）
                                        c_reach=c_reach, dock_radius=dock_radius, v_dock=v_dock,   # 🆕 第二条腿修法透传（`03` L172·连续臂专属·默认关 bit-identical）
                                        alias_weight=alias_weight, rate_weight=rate_weight, rate_dock=rate_dock),   # 动作混叠(Markgraf 式20·L97) + action-rate 平滑(L98) + rank1 泊位门控治抖(L173)
                        subproc=subproc, seed=seed)
    # ③ 同款 VecNormalize（obs+clip 单一真相源·norm_reward 可调·gamma 同进）
    if use_vecnorm:
        vn = dict(VECNORM_KWARGS)
        vn["norm_reward"] = norm_reward
        venv = VecNormalize(venv, gamma=gamma, **vn)
    # ④ sb3 PPO（连续 Box·net [64,64]·ent_coef 同离散·gamma 与 VecNormalize 一致）
    # 🔴 F1（深核 MAJOR）：sb3 PPO 用【不 squash 的高斯】·log_std_init=0→σ=1.0·而动作箱仅 ±0.048/±0.018→采样几乎全被 clip
    #    到边界=探索退化成 bang-bang + log_prob 与 clip 后动作错配=学不动。SAC 没此病（tanh squash+缩放）。修=初始 σ 落箱内。
    if log_std_init is None:
        log_std_init = float(np.log(A_NORMAL_OMEGA_MAX / 2.0))   # σ≈0.009·落进两轴箱内(±0.048/±0.018)；log_std 仍 per-dim 可学
    cfg = dict(policy_kwargs=dict(net_arch=POLICY_NET_ARCH, log_std_init=log_std_init), seed=seed, gamma=gamma,
               ent_coef=ent_coef, verbose=0)
    cfg.update(ppo_kwargs)
    model = PPO("MlpPolicy", venv, **cfg)
    # 🆕⑤ 热启动（`03` L190·默认 None=整块不调=bit-identical）：灌源 policy + 复制源 vecnorm stats（须先有 venv=VecNormalize）
    if warmstart_ckpt:
        if not use_vecnorm:
            raise ValueError("🔒 热启动须 use_vecnorm=True（须复制源 obs_rms 归一化坐标系·否则策略喂错分布）")
        model, venv = apply_warmstart(model, venv, warmstart_ckpt, device=str(model.device))
    # ⑥ 过门强制：构造时即跑 PPO 口径自检（热启动后仍须满足=算法/net/gamma/colregs 不被热启动改）
    if use_vecnorm:
        assert_continuous_safe_ppo_caliber(model, venv, colregs_weight=colregs_weight)
    return model, venv


def assert_continuous_safe_ppo_caliber(model, venv, colregs_weight: float = CONTINUOUS_SAFE_COLREGS_WEIGHT) -> None:
    """Continuous-safe-PPO 训练侧口径自检（算法/net/归一化/colregs_weight/gamma/动作箱）。不符即 raise。

    与 SAC caliber 平价（去掉 SAC 专属 ent_coef=='auto'/StableSAC·换 PPO 专属：算法须 sb3 PPO·非 MaskablePPO[那是离散]）。
    """
    if not isinstance(model, PPO):
        raise AssertionError(f"Continuous-safe-PPO 算法须 sb3 PPO，得 {type(model).__name__}")
    try:                                                        # 排除 sb3-contrib MaskablePPO（离散臂专用·连续臂不该用）
        from sb3_contrib import MaskablePPO as _MaskablePPO
        if isinstance(model, _MaskablePPO):
            raise AssertionError("连续臂须 sb3 PPO（非 MaskablePPO=离散 action masking 专用）")
    except ImportError:
        pass
    arch = model.policy_kwargs.get("net_arch")
    if list(arch) != list(POLICY_NET_ARCH):
        raise AssertionError(f"net_arch 须 {POLICY_NET_ARCH}（忠实 Krasowski），得 {arch}")
    if not isinstance(venv, VecNormalize):
        raise AssertionError("Continuous-safe-PPO 须 VecNormalize（同款 obs 归一化，四方平价）")
    if not venv.norm_obs:
        raise AssertionError("VecNormalize.norm_obs 须 True（obs 归一化是停船配方命门，D22）")
    if float(venv.clip_obs) != float(VECNORM_KWARGS["clip_obs"]):
        raise AssertionError(f"clip_obs 须 {VECNORM_KWARGS['clip_obs']}（obs 归一化口径，四方平价），得 {venv.clip_obs}")
    if abs(float(venv.gamma) - float(model.gamma)) > 1e-12:
        raise AssertionError(f"gamma 须同进 VecNormalize({venv.gamma}) 与 PPO({model.gamma})（L21 MINOR②）")
    for e in venv.venv.get_attr("env"):                         # colregs_weight 意图真落地（默认0.0=丢 r_colregs·D37-B·可A/B覆盖）
        cw = float(e.reward_fn.colregs_weight)
        if cw != colregs_weight:
            raise AssertionError(f"colregs_weight 接线未落地: env 读回 {cw}≠意图 {colregs_weight}")
        if abs(float(e.reward_fn.gamma) - float(model.gamma)) > 1e-12:   # 修法A PBRS 第三处 gamma 同源（破则 Ng 失效·`03` L81）
            raise AssertionError(f"reward_fn.gamma({e.reward_fn.gamma}) 须 == model.gamma({model.gamma})（修法A PBRS 须与 trainer/VecNorm 同源 _GAMMA）")
    hi = np.asarray(venv.action_space.high, dtype=float)        # 动作箱 = Krasowski 正常操作 ±0.048/±0.018（L63 Fix②·四方公平）
    lo = np.asarray(venv.action_space.low, dtype=float)
    exp_hi = np.array([A_NORMAL_ACCEL_MAX, A_NORMAL_OMEGA_MAX])
    if not (np.allclose(hi, exp_hi) and np.allclose(lo, -exp_hi)):
        raise AssertionError(f"连续臂动作箱须 ±[{A_NORMAL_ACCEL_MAX}, {A_NORMAL_OMEGA_MAX}]（L63 Fix②），得 high={hi} low={lo}")
    # F1（深核 MAJOR）：初始高斯 σ 必须落进动作箱内（每轴 σ≤箱半宽），否则采样几乎全 clip 到边界=探索退化 bang-bang。
    # 构造时 log_std==log_std_init（未训练）→ 此处校验初始探索尺度合理（σ=1.0 vs 箱 0.048 是 footgun）。
    init_std = np.exp(np.asarray(model.policy.log_std.detach().cpu().numpy(), dtype=float))
    if np.any(init_std > exp_hi):
        raise AssertionError(f"PPO 初始高斯 σ={init_std} 超动作箱半宽 ±{exp_hi}（log_std_init 太大→采样全 clip 到边界=探索退化·F1）")
