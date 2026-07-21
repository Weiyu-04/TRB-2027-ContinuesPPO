"""
TRB 环境 · 本船动力学模块
==========================
本船动力学 = Krasowski 偏航受限点质量模型 Ω_yc（论文 Krasowski & Althoff 2024 式(1)）。

⭐ 核心不自己写、不转写、不猜：直接调用 **Krasowski 官方包** `vesselmodels.vessel_dynamics_yp`
   （作者 Hanna Krasowski / TUM CPS，随 commonocean-io 发布、有许可）。
   **其 RHS 已逐行核对 = 论文式(1)（两个独立 agent 复核确认）**：
       f = [cos(θ)·v,  sin(θ)·v,  ω,  a]
   状态 x=[px, py, θ, v]，输入 u=[a, ω]；函数内部已做 a/ω 限幅（见 input_constraints.py）。
   ⚠️ "一致"严格只指 RHS 公式；整条 ego 轨迹还取决于积分器，而她的确切积分器未公开 →
      本模块按 commonroad/commonocean 约定用 odeint（实测与高精度积分误差亚微米：纯转向~1e-8m / 耦合加速+转向~3.5e-7m，见下）。

本模块只做 Krasowski 那个 RHS 函数“没做、但环境需要”的事：
  1. **积分**：用 scipy.integrate.odeint 在一个决策步 [0, Δt] 上积分连续 RHS。
     —— 约定来自姊妹包 vehiclemodels（同 TUM-CPS）的 `odeint(func, x0, t, args=(u, p))` 用法。
        本模块用 [0, Δt] 两点输出：对常值 (a,ω) 下精确可积的 yp RHS，两点与细网格等价
        （LSODA 在输出点间自适应内部步长决定精度；实测两者及与闭式真值误差亚微米：纯转向~1e-8m、耦合加速+转向~3.5e-7m，审计核）。
  2. **步后处理**：θ wrap 到 [-π, π]（便于观测/比较，不影响动力学，因 RHS 只用 cos/sin）。

施工参数来源（全部 fact-based，对照《文献核实笔记》① + Table II）：
  - 船型 = `parameters_vessel_1`（SR108 集装箱船）：l=175m / a_max=0.24 / w_max=0.03 —— 与 Table II 一致 ✓。
  - ⚠️ **v_max**：官方包默认 16.8 m/s，但论文 §VII 明确 “reduce the maximum velocity ... to 9.5 m/s”
       → 本模块把 p.v_max 覆盖为 **9.5**（供下游 safe_speed 谓词 / 观测 / 奖励用）。
  - ⚠️ **v_max 不在动力学里硬卡**：`vessel_dynamics_yp` 的 RHS 只限 a/ω、**不限 v**。
       因此本模块默认 **不 clip v**（= 忠实她的官方代码）。v≤v_max 由 env 层的 safe_speed(R2)/速度奖励管。
       —— 待 Phase 1 step4 复现基线时，与她实际行为核对此点（已在《03》/《02》标记）。

决策步长 Δt = 10 s（场景文件实测 dt=10.0 ✓，论文 ∆t=10s）。
"""
import numpy as np
from scipy.integrate import odeint
from vesselmodels.vessel_dynamics_yp import vessel_dynamics_yp
from vesselmodels.parameters_vessel_1 import parameters_vessel_1

# 论文 §VII 把 vessel_1 默认 v_max(16.8) 降到 9.5
PAPER_V_MAX: float = 9.5
# 决策步长（论文 ∆t / 场景 dt 实测均为 10s）
DECISION_DT: float = 10.0


def make_vessel_params(v_max: float = PAPER_V_MAX):
    """集装箱船参数（Krasowski 官方 `parameters_vessel_1`），v_max 覆盖为论文值 9.5。"""
    p = parameters_vessel_1()
    p.v_max = v_max
    return p


def wrap_to_pi(angle: float) -> float:
    """角度 wrap 到 [-π, π]。"""
    return (float(angle) + np.pi) % (2.0 * np.pi) - np.pi


def _rhs(x, t, u, p):
    """odeint 适配层：把 Krasowski 的 vessel_dynamics_yp(x, u, p) 适配成 odeint 的 func(y, t, *args)。

    注意：vessel_dynamics_yp 每次被调用都会对常值 u 做一次 a/ω 限幅（幂等），
    所以 odeint 自适应多次评估 RHS 时限幅一致，无副作用。
    """
    return vessel_dynamics_yp(x, u, p)


def step(state, action, dt: float, p, clip_velocity: bool = False):
    """本船前进一个决策步（忠实 Krasowski 偏航受限模型 + 标准 odeint 积分）。

    参数
    ----
    state : array-like [px, py, θ, v]   位置(m) / 艏向(rad) / 沿艏速度(m/s)
    action: array-like [a, ω]           沿艏向加速度(m/s²) / 转艏率(rad/s)；
                                        a/ω 的限幅由 Krasowski 函数内部完成（±a_max / ±w_max）。
    dt    : float                       决策步长(s)
    p     : VesselParameters            船参数（用 make_vessel_params()）
    clip_velocity : bool                是否步后把 v clip 到 [0, p.v_max]。
                                        默认 False = 忠实 vessel_dynamics_yp（其 RHS 不限速）。
                                        设 True 仅用于实验对照（会引入“位置按未限速 v 积分、v 却被截断”的不一致）。

    返回
    ----
    next_state : np.ndarray [px, py, θ, v]
    """
    state = np.asarray(state, dtype=float).copy()
    action = np.asarray(action, dtype=float)
    if state.shape != (4,):
        raise ValueError(f"state 应为 4 维 [px,py,θ,v]，得到 shape={state.shape}")
    if action.shape != (2,):
        raise ValueError(f"action 应为 2 维 [a,ω]，得到 shape={action.shape}")

    # 在 [0, dt] 上积分常值控制的连续 RHS（commonroad/commonocean 约定）
    traj = odeint(_rhs, state, [0.0, dt], args=(action, p))
    nxt = np.asarray(traj[-1], dtype=float)

    nxt[2] = wrap_to_pi(nxt[2])  # θ wrap（RHS 只用 cos/sin，wrap 不改动力学）
    if clip_velocity:
        nxt[3] = float(np.clip(nxt[3], 0.0, p.v_max))
    return nxt
