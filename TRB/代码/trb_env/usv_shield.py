"""
TRB 环境 · SE-RL 安全盾层（step4c）
====================================
把 step3 的 `SafeActionScheduler`（COLREGs 状态机 + 机动合成）与 step4b 的 `USVEnv`（gymnasium.Env 物理）
拼成 **sb3-contrib MaskablePPO 可直接训练的 masked 离散环境**。每决策步算 As(ρ) → 50 维 action mask；
紧急槽 49 的状态相关 a_em 由调度器算好、step 内部自动传入（agent 只选下标、不需知道 a_em 数值）。

fact-based（2024 §V-C Theorem 2，p9 实抽）：
  "Action masking then restricts the RL agent to this set of verified rule-compliant actions."
  As(ρ0)=A_regular（全49）/ As(ρ5)=a_em / As(ρ1-4)=Alg.3 输出 —— 与 SafeActionScheduler 三档一致。
  论文用 PPO + action masking [8]（基于 stable-baselines3 [66]）；step4 基线复现用 sb3-contrib MaskablePPO。

设计决定（step4c）：
  1. **单 (ego, obstacle) pair**（论文 assumption 2；多船 deferred Phase 3/5）。
  2. **49 槽不变量（step4b 复核 Agent A footgun + D15 带入）**：mask 保证 ρ≠ρ5 → idx49=False；
     ρ5 → 仅 idx49=True 且 _a_em 已备好。选 49 而 _a_em=None → RuntimeError（不该发生，mask 已屏蔽）。
  3. **As=∅（give-way 无 verified 机动）→ a_keep 兜底**：保持航向（最小机动、还不紧急时安全），
     等 is_emergency 升级 ρ5 兜底。**论文 §V-B 末明示 "If G is an empty set, the ego vessel is a
     stand-on vessel and the only selectable action is akeep"**（机动合成空 → 退化 stand-on → a_keep），
     本层忠实落地（Agent A 复核坐实，主窗口读原文核验）。Phase 2-3 卖点 = 这层离散 masking 换连续投影；
     step5 做"两船夹击 / 他船违规"的完整 P=∅ 兜底（碰撞风险最小化 + 放松 COLREGs 方向）。
  4. **他船 length = 真实 obstacle_shape.length**（不硬编 175）——解 02 挂起（全面审计 Agent 6：colregs
     的 r_m/外接圆/dobs,safety 全靠他船 length，硬编 175 会偏小、在危险侧）。obs_width 暂用 scheduler
     默认保守 w=l（02 挂起，Phase 3 标定）。
  5. action_masks() = MaskablePPO 接口；mask 时序 = 当前状态（reset/step 后即算），agent 在 step 前读。
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from .usv_colregs import (A_KEEP, RHO_EMERGENCY, RHO_NO_CONFLICT, SafeActionScheduler,
                          VesselState)
from .usv_env import DISCRETE_ACTIONS, IDX_EMERGENCY, N_ACTIONS_TOTAL, USVEnv, assert_single_obstacle


def _key(aw) -> tuple:
    """(a,ω) → round 后的 dict key（As 值同源 49 网格、round 6 位防浮点尾巴）。"""
    return (round(float(aw[0]), 6), round(float(aw[1]), 6))


# (a,ω) → 49 网格下标（精确映射；As ⊂ 网格，已实测 A_KEEP/ATR/ATL/AACC 全在网格上）
ACTION_TO_IDX: dict = {_key(aw): i for i, aw in enumerate(DISCRETE_ACTIONS)}
_AKEEP_IDX: int = ACTION_TO_IDX[_key(A_KEEP)]   # a_keep=(0,0) 的下标（As=∅ 兜底用）


class ShieldedUSVEnv(gym.Env):
    """SE-RL 安全盾：USVEnv（离散物理）+ SafeActionScheduler → MaskablePPO 可训练 masked 环境。

    用法（step4c 训练脚手架）：
        env = ShieldedUSVEnv(scenario, planning_problem)
        # MaskablePPO 自动调 env.action_masks() 获取合法动作
        obs, info = env.reset(seed=0)
        mask = env.action_masks()                 # 50 bool；当前合法动作
        obs, r, term, trunc, info = env.step(action)   # 49 槽 a_em 自动传入
    """

    metadata = {"render_modes": []}

    def __init__(self, scenario, planning_problem, *, clip_velocity: bool = True,
                 colregs_weight: float = 1.0,
                 gamma: float = 0.99, well_shaping_weight: float = 0.0, shaping_radius: float = 500.0,
                 xtrack_weight: float = 0.0, xtrack_radius: float = 80.0,   # 对症 横向进带势（`03` L88·显式具名在 **scheduler_kwargs 前）
                 **scheduler_kwargs):
        super().__init__()
        # 🔴 防御 fail-fast（`03` L111①a 二审抓·补防御缺口）：park_* 是【连续臂专属】终端保速势(Φ_park·PBRS·`03` L109)
        #   配方·离散臂【不接入=不替对手开挂】。误传会被下方 **scheduler_kwargs→SafeActionScheduler **alg3_kwargs 静默吞掉
        #   (well_park 永远=0·不污染奖励)→看似无害但静默无效=footgun；此处显式硬拒(对齐 UnshieldedUSVEnv 的 TypeError 行为)。
        #   c_step 同理（修法C 每步生存成本·非PBRS·`03` L123·连续臂专属·离散复现臂忠实 Krasowski 不接）→ 一并硬拒。
        _PARK_KEYS = {"park_weight", "park_radius", "park_v_target", "c_step",
                      "c_dwell", "w_dwell", "h_dwell", "dwell_radius", "b_dwell"}   # r_dwell 入库赤字滞留成本键（`03` L161/L162·连续臂专属·离散硬拒=不替对手开挂）
        _bad_park = _PARK_KEYS & set(scheduler_kwargs)
        if _bad_park:
            raise TypeError(
                f"ShieldedUSVEnv 不接受参数 {sorted(_bad_park)}：park_*（Φ_park 终端保速势·`03` L109/L111）与 c_step"
                "（修法C 每步生存成本·`03` L123）与 r_dwell（入库赤字滞留成本·`03` L161/L162）均=【连续臂专属】训练配方，离散复现臂不接入（忠实 Krasowski·不开挂）；"
                "误传会被 **scheduler_kwargs 静默吞掉故此处 fail-fast。")
        self.env = USVEnv(scenario, planning_problem, continuous=False,
                          clip_velocity=clip_velocity, colregs_weight=colregs_weight,
                          gamma=gamma, well_shaping_weight=well_shaping_weight, shaping_radius=shaping_radius,
                          xtrack_weight=xtrack_weight, xtrack_radius=xtrack_radius)
        self.scheduler = SafeActionScheduler(vessel_params=self.env.p, **scheduler_kwargs)
        self.action_space = self.env.action_space            # Discrete(50)
        self.observation_space = self.env.observation_space  # Box(27,)
        self._ego_length = float(self.env.p.l)               # 本船 SR108 = 175
        self._obstacles = self.env.obstacles                 # 单他船（assumption 2）
        assert_single_obstacle(self._obstacles, type(self).__name__)   # D40#4/L49#2：多他船 fail-fast（盾只护 obstacles[0]）
        self._obs_length = (float(self._obstacles[0].obstacle_shape.length)
                            if self._obstacles else self._ego_length)   # 真实他船长（决定 4）
        self._mask: np.ndarray | None = None
        self._a_em: tuple | None = None
        self._rho: int = RHO_NO_CONFLICT

    # ---- gymnasium 契约 ----
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs, info = self.env.reset(seed=seed)
        self.scheduler.reset()
        self._compute_shield()
        return obs, {**info, "rho": self._rho, "rho_acting": self._rho, "action_mask": self._mask.copy()}

    def step(self, action):
        if int(action) != action:                            # 拒非整数下标（不被 int() 绕过 USVEnv 守护，Agent B MINOR）
            raise ValueError(f"离散动作下标应为整数，得到 {action!r}")
        rho_acting = self._rho                                # ρ@t：本步动作所处态势（gated 当前 mask）；评估紧急步%口径用此（D40 #1，pre-step 与连续臂/C_EMERGENCY 同源）
        a = int(action)
        if a == IDX_EMERGENCY:
            if self._a_em is None:                           # 49 槽不变量违反（mask 应已屏蔽，决定 2）
                raise RuntimeError("选了紧急槽 49 但当前 _a_em=None（非 ρ5）——mask 未屏蔽 49，"
                                   "违反 49 槽不变量（step4b D15）")
            obs, r, term, trunc, info = self.env.step(IDX_EMERGENCY, emergency_action=self._a_em)
        else:
            obs, r, term, trunc, info = self.env.step(a)
        self._compute_shield()                               # 推到 ρ@(t+1)：info["rho"] 与 action_mask 配对（供 next 决策）
        return obs, r, term, trunc, {**info, "rho": self._rho, "rho_acting": rho_acting,
                                     "action_mask": self._mask.copy()}

    def action_masks(self) -> np.ndarray:
        """MaskablePPO 接口：当前状态合法动作的 50 维 bool（True=可选）。"""
        if self._mask is None:                               # 先 reset 守护（Agent B OBS，报错友好）
            raise RuntimeError("action_masks() 前必须先 reset()")
        return self._mask.copy()

    # ---- shield 内部 ----
    def _ego_vs(self) -> VesselState:
        e = self.env.ego
        return VesselState(position=np.array([e[0], e[1]], dtype=float),
                           orientation=float(e[2]), velocity=float(e[3]),
                           length=self._ego_length)

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

    def _compute_shield(self) -> None:
        """推状态机一步、算当前 (ρ, As) → mask + a_em。"""
        s_obs = self._obs_vs()
        if s_obs is None:                                    # 无他船 → 无规则适用，全 49（非紧急）
            self._rho = RHO_NO_CONFLICT
            self._mask = self._regular_mask()
            self._a_em = None
            return
        rho, a_s = self.scheduler.step(self._ego_vs(), s_obs)
        self._rho = rho
        self._a_em = (_to_pair(next(iter(a_s))) if rho == RHO_EMERGENCY and a_s else None)
        self._mask = self._as_to_mask(a_s, rho)

    @staticmethod
    def _regular_mask() -> np.ndarray:
        m = np.zeros(N_ACTIONS_TOTAL, dtype=bool)
        m[:len(DISCRETE_ACTIONS)] = True                     # 前 49 True，紧急槽 49 False
        return m

    def _as_to_mask(self, a_s, rho: int) -> np.ndarray:
        m = np.zeros(N_ACTIONS_TOTAL, dtype=bool)
        if rho == RHO_EMERGENCY:
            m[IDX_EMERGENCY] = True                          # 仅紧急槽（49 不变量）
            return m
        for aw in a_s:                                       # ρ0-4：As ⊂ 49 网格 → 置对应下标
            idx = ACTION_TO_IDX.get(_key(aw))
            if idx is not None:
                m[idx] = True
        if not m.any():                                      # As=∅ give-way 无解 → a_keep 兜底（决定 3）
            m[_AKEEP_IDX] = True
        return m


def _to_pair(aw) -> tuple:
    return (float(aw[0]), float(aw[1]))


class UnshieldedUSVEnv(gym.Env):
    """无盾对照环境（Base / Rule-reward 基线，step4d-②）= USVEnv + action_masks 恒全 49。

    与 `ShieldedUSVEnv` 配对，供四方对比（钱图）同口径评估：
      Discrete-safe = ShieldedUSVEnv（有 As(ρ) mask）；**Base / Rule-reward = 本类（无盾）**；
      Continuous-safe = SAC + 连续投影（Phase 3）。

    fact-based（Krasowski 2024 §VII Table III，笔记④⑤）：
      Base / Rule-reward 的 Verify=✗ → **无安全验证机制 → 无紧急控制器 → 无 a_em**，agent 只在 49 个
      A_regular 网格动作上自由探索（Table III Base/RR 的"紧急步"列 = "–"）。故：
      · `action_masks()` 恒全 49（idx0-48 True、**紧急槽 idx49 永 False**，无 As(ρ) 约束）；
      · 无状态机 → `info["rho"]` 恒 RHO_NO_CONFLICT（评估管线紧急步% = 0，对应 Table III "–"）；
      · **保留 `_ego_vs/_obs_vs`**：`ViolationCounter` 据此测【裸策略违规】——这是 Table III
        **Base 违规 2.65 / RR 2.24 锚点的来源**（与有盾 Safe=0 对照）。
    Base vs Rule-reward 的差异**仅在 reward**（论文 §VII p12：RR 含 r_colregs〔soft-constrained
      rule-reward〕、Base = 式(10) 关 r_colregs），**与本 env 的动力学/动作/mask 无关** → 二者共用本类。
      **r_colregs 开关 = 构造参数 `colregs_weight`**（已接线，4d-②）：Base 传 `colregs_weight=0.0`、RR 传 `1.0`（默认）；
      透传 `USVEnv`→`RewardFunction`；train.py 的 `make_base_model`/`make_rule_reward_model` 即此。

    与 `ShieldedUSVEnv` 接口一致（evaluate.run_episode duck-typed 复用、零改动）：reset/step/action_masks/
      _ego_vs/_obs_vs/.env.dt。`ShieldedUSVEnv` 字节不动（本类仅新增）。
    """

    metadata = {"render_modes": []}

    def __init__(self, scenario, planning_problem, *, clip_velocity: bool = True,
                 colregs_weight: float = 1.0,
                 gamma: float = 0.99, well_shaping_weight: float = 0.0, shaping_radius: float = 500.0,
                 xtrack_weight: float = 0.0, xtrack_radius: float = 80.0):   # 对症 横向进带势（`03` L88）
        super().__init__()
        self.env = USVEnv(scenario, planning_problem, continuous=False,
                          clip_velocity=clip_velocity, colregs_weight=colregs_weight,
                          gamma=gamma, well_shaping_weight=well_shaping_weight, shaping_radius=shaping_radius,
                          xtrack_weight=xtrack_weight, xtrack_radius=xtrack_radius)
        self.action_space = self.env.action_space            # Discrete(50)（idx49 永被 mask）
        self.observation_space = self.env.observation_space  # Box(27,)
        self._ego_length = float(self.env.p.l)
        self._obstacles = self.env.obstacles
        assert_single_obstacle(self._obstacles, type(self).__name__)   # D40#4/L49#2：多他船 fail-fast（盾只护 obstacles[0]）
        self._obs_length = (float(self._obstacles[0].obstacle_shape.length)
                            if self._obstacles else self._ego_length)
        self._mask = np.zeros(N_ACTIONS_TOTAL, dtype=bool)
        self._mask[:len(DISCRETE_ACTIONS)] = True            # 全 49 True、紧急槽 49 False（无盾不变量）

    # ---- gymnasium 契约（与 ShieldedUSVEnv 一致；info 带 rho=NO_CONFLICT + mask）----
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs, info = self.env.reset(seed=seed)
        return obs, {**info, "rho": RHO_NO_CONFLICT, "rho_acting": RHO_NO_CONFLICT, "action_mask": self._mask.copy()}

    def step(self, action):
        if int(action) != action:                            # 拒非整数下标（不被 int() 绕过 USVEnv 守护）
            raise ValueError(f"离散动作下标应为整数，得到 {action!r}")
        a = int(action)
        if a == IDX_EMERGENCY:                               # 无盾不提供紧急槽（mask 已屏蔽 idx49）
            raise RuntimeError("无盾 env 不提供紧急槽 49（mask 恒屏蔽 idx49）——Base/RR 无安全验证 / 紧急控制器")
        obs, r, term, trunc, info = self.env.step(a)
        return obs, r, term, trunc, {**info, "rho": RHO_NO_CONFLICT, "rho_acting": RHO_NO_CONFLICT,
                                     "action_mask": self._mask.copy()}

    def action_masks(self) -> np.ndarray:
        """MaskablePPO 接口：恒全 49（无 As(ρ) 约束）。全 True mask ⟹ 等价无 masking 的 PPO over 49 动作。"""
        return self._mask.copy()

    # ---- ViolationCounter 喂数（与 ShieldedUSVEnv._ego_vs/_obs_vs 同语义）----
    def _ego_vs(self) -> VesselState:
        e = self.env.ego
        return VesselState(position=np.array([e[0], e[1]], dtype=float),
                           orientation=float(e[2]), velocity=float(e[3]), length=self._ego_length)

    def _obs_vs(self) -> VesselState | None:
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
