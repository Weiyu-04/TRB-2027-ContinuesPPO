"""
TRB 环境 · 连续 COLREGs 投影层（Phase 2 主线，蓝图 §12 方法核心）
================================================================
把连续动作 u=[a, ω] 投影进"COLREGs 合规方向 ∩ 物理限幅 ∩（Node 2 起）无碰撞"安全集。
卖点 = 把 Krasowski 2024 的**离散动作屏蔽**换成**连续投影**（同状态机、同环境、同动力学）。

⚠️ 节点式构建（每节点过 CLAUDE §2 全流程门后才进下一）：
  Node 1（本文件当前）= U_box ∩ U_colregs（合规方向，复用状态机 ρ）的【精确解析投影】。
     —— 本节约束全是【轴对齐】：U_colregs 只约束 ω（让路/保向），stand-on 另约束 a（保速），U_box 是箱
        → 在 box 内逐轴区间投影即 Euclidean 投影精确解，**无需 QP 求解器**（微秒级、无 cvxpy 每步开销）。
  Node 2（待建）= U_collision-free（§12.3.3 分离超平面 + 标量一阶展开 + d_safe）—— 引入【耦合 a,ω】约束
     → 届时换 QP（cvxpy/OSQP 已装），并对本节可分离情形做等价核验。
  Node 3 = 最小闭环（对遇 + 非 SAC 简单策略）→ 通过门 2（整段零违规/零碰撞/能到达）。
  Node 4 = P=∅ / ρ5 紧急兜底（§12.4）。

复用 Phase-1 已验证件（usv_colregs：状态机/常量/让路方向），**只新增本文件、不碰共享 env**。

⚠️ 安全边界（Node 1 现状，勿误用）：
  · 无碰撞约束 Node 2 才有 → **Node 1 的输出不是无碰撞安全动作**，只验"COLREGs 方向投影"逻辑本身。
  · ρ5（紧急）本节**不做合规约束、不保证安全** → 仅 box 投影 + needs_fallback=True 显式标记，交 Node 4。
  · 调用契约：project() 内部推进状态机一步，**每决策步必须且只能调一次**（同 SafeActionScheduler.step）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import usv_dynamics
from . import uterm_terminal as _uterm   # U_term SOUND 终端核心（Prop4 v2 backup-maneuver·纯·已本机全测·任务A 2026-07-25）
from .usv_colregs import (
    ColregsStatechart,
    EmergencyController,       # Node 4 紧急兜底（Alg.1，复用 Phase-1 组件3 D13）
    get_turning_act,
    DELTA_LARGE_TURN,
    DELTA_NO_TURN,
    T_M,                       # ⚠️ 段机动时间 = 40s（Table II）；非 T_MANEUVER=70（反应窗），勿混（复审 catch）
    RHO_NO_CONFLICT,
    RHO_STAND_ON,
    RHO_HEAD_ON,
    RHO_CROSSING,
    RHO_OVERTAKE,
    RHO_EMERGENCY,
    # ── Node 2（collision-free）原料：全部复用 Phase-1 已验证件 ──
    predict_state_cv,          # 他船恒速预测（§12.3.1，规则态势保向保速）
    _vessel_circumradius,      # 船体外接圆半径（usv_colregs:388）
    DOBS_SAFETY_FACTOR,        # dobs,safety = 2·l_obs（Table II，usv_colregs:631）
    EGO_WIDTH,                 # 本船宽 25.4m（SR108 vessel_1）
    DT,                        # 决策步长 10s（gap#1 迫近不可避证书 v_bnd 用·`03` L165）
    # ── N1 档位B* 递归可行性终端检查（存在性级）原料·复用已 2-agent 复核件、不重写 ──
    VesselState,               # 构造落点 s' 状态（位置+朝向+速度）
    encounter_action_verification,  # As(ρ') 非空 ⟺ 落点存在合规脱身机动（Alg.3，:855）
)

# ── 合规约束阈值（全部 fact-based 自 Phase-1 验证常量；⚠️ 用 T_M=40 段机动时间）──────────────────
# 让路"明显转向"率下限：ω 持续 T_M 秒累积转角 ≥ Δ_large_turn(20°) → ω_turn = Δ_large_turn / T_M。
#   = COLREGs 阈值本尊（蓝图 §12.2"Δ_large_turn → 转艏率下限"）；离散基线 ATR 用 ω∈{0.012,0.018}
#   = 该阈值的离散网格最近可表示值（更保守）。默认取 COLREGs 阈值、可配置切换对齐离散（Node 3 评审门定）。
DEFAULT_OMEGA_TURN: float = DELTA_LARGE_TURN / T_M      # 20°/40s ≈ 0.008727 rad/s
# 直航"保向"窄带：ω 持续 T_M 秒累积转角 ≤ Δ_no_turn(10°) → ε_ω = Δ_no_turn / T_M。
DEFAULT_EPS_OMEGA: float = DELTA_NO_TURN / T_M          # 10°/40s ≈ 0.004363 rad/s
# ⚠️ 口径差异（对抗复核 minor，挂 Node 3 标定门）：本窄带只保证【单 T_M=40s 窗内】累积转角 ≤ Δ_no_turn(10°)；
#   而离散 R_G6 stand-on 违规谓词是【整段 keep spell 跨每决策步累积】(usv_colregs ViolationCounter)→ 带边缘 ω
#   持续多步可累积超 10°、触发离散违规而本逐步带允许之。属蓝图 §12.2 已背书的 ε 近似；精确口径（逐步带 vs
#   整段窗）留 Node 3 闭环用违规计数实测标定（含是否改"累积式"约束或缩小 ε_ω）。
# 直航"保速"窄带：§VII-A 违规口径只测航向、无 Krasowski 直接速度常量 → 取离散 A_ACC 一个网格步(0.016)
#   = "距精确保持(a=0)一个离散步内"，工程整定、可配置（Node 3 评审门标定）。
DEFAULT_EPS_A: float = 0.016


@dataclass
class ProjectionResult:
    """project() 的返回（含投影动作 + 诊断/兜底标记）。"""
    u_safe: np.ndarray          # 投影后动作 [a, ω]（float64，∈ U_box）。⚠️ needs_fallback=True 时这是【合规占位、非无碰撞安全】(复审 B2)：调用方须改用 Node 4 兜底动作、绝不可直接执行 u_safe。
    rho: int                    # 当前相遇态势 ρ（0-5，RHO_*）
    corrected: bool             # 投影点是否 ≠ u_desired（是否真做了修正）
    needs_fallback: bool        # True = ρ5 紧急 / 安全集空 / 退化 / P=∅ → 交 Node 4 兜底；**调用方必须先查此标志再用 u_safe**（Node 3 集成层须加断言，复审 B2）
    give_way_dir: str | None     # 让路方向 'right'/'left'/None（日志/调试）


@dataclass
class SafeActionResult:
    """safe_action()（Node 2c 投影 + Node 4 兜底）的返回 = 总是给出一个【可直接执行】的安全动作。"""
    u_safe: np.ndarray          # 最终安全动作 [a, ω]（float64）——恒为可执行动作（兜底已落地）。projection/relaxed/collision_min ∈ U_box；emergency = Alg.1 EC 输出（env _map_action 做最终 box 限幅，同离散基线）
    rho: int                    # 当前相遇态势 ρ（0-5）
    give_way_dir: str | None    # 让路方向（日志）
    source: str                 # 来源：'projection'(Node2c QP)/'emergency'(ρ5 Alg.1)/'relaxed'(P=∅放松COLREGs保无碰撞)/'collision_min'(无可行无碰撞动作→碰撞风险最小化)/'degenerate'(圆心重合=已碰)
    emergency_mode: str | None  # source='emergency' 时 Alg.1 模式 'ahead'/'stern'/'base'，否则 None（观测/日志）


class ContinuousColregsProjection:
    """连续 COLREGs 投影盾（Node 1：U_box ∩ U_colregs 解析投影）。

    复用 ColregsStatechart 判 ρ（**与离散基线同一状态机** → 四方对比同口径）。
    用法：
        proj = ContinuousColregsProjection(a_max, w_max)
        proj.reset()                                   # 每 episode 开始
        res = proj.project(s_ego, s_obs, u_desired)    # 每决策步一次
        u = res.u_safe
    """

    def __init__(
        self,
        a_max: float,
        w_max: float,
        *,
        omega_turn: float = DEFAULT_OMEGA_TURN,
        eps_omega: float = DEFAULT_EPS_OMEGA,
        eps_a: float = DEFAULT_EPS_A,
        statechart: ColregsStatechart | None = None,
        recursive_feasibility: bool = False,
        terminal_mode: str = "discrete",
        terminal_dt_sim: float = 0.5,
        # ── 统一态势盾·ρ0 朝目标锥安全集（方案①，2026-07-04）──
        # goal_cone_half=None（默认）→ ρ0 分支完全不生效 = 逐位等价现状 bit-identical。
        # goal_cone_half=Φ(rad) → ρ0（无冲突空旷水域）也约束动作"朝目标锥"防游荡（硬约束）。
        goal_cone_half: float | None = None,
        goal_v_floor: float = 2.0,
        v_max: float = 9.5,
    ):
        if not (a_max > 0.0 and w_max > 0.0):
            raise ValueError(f"a_max/w_max 必须 > 0，得到 {a_max}/{w_max}")
        if not (0.0 < omega_turn <= w_max):
            raise ValueError(f"omega_turn 须 ∈ (0, w_max={w_max}]，得到 {omega_turn}（否则让路约束不可行）")
        if not (0.0 <= eps_omega <= w_max):
            raise ValueError(f"eps_omega 须 ∈ [0, w_max={w_max}]，得到 {eps_omega}")
        if not (0.0 <= eps_a <= a_max):
            raise ValueError(f"eps_a 须 ∈ [0, a_max={a_max}]，得到 {eps_a}")
        self.a_max = float(a_max)
        self.w_max = float(w_max)
        self.omega_turn = float(omega_turn)
        self.eps_omega = float(eps_omega)
        self.eps_a = float(eps_a)
        # ── ρ0 朝目标锥安全集参数（None=关=逐位等价现状）──
        if goal_cone_half is not None:
            gch = float(goal_cone_half)
            if not (0.0 < gch <= np.pi):      # 锥半角 ∈ (0, π]（Φ=0 退化空锥、>π 无意义）
                raise ValueError(f"goal_cone_half 须 ∈ (0, π]（rad），得到 {gch}")
            self.goal_cone_half: float | None = gch
        else:
            self.goal_cone_half = None
        self.goal_v_floor = float(goal_v_floor)          # ρ0 锥保正速下界（默认 2.0）
        if not (0.0 <= self.goal_v_floor <= float(v_max)):
            raise ValueError(f"goal_v_floor 须 ∈ [0, v_max={v_max}]，得到 {self.goal_v_floor}")
        self.v_max = float(v_max)                        # 保速上界（动力学 v_max=9.5，锥 a 区间用）
        self.goal: np.ndarray | None = None              # 目标点 [x, y]（set_goal 注入；None=无目标=锥不生效）
        # ── N1 档位B*（默认关 recursive_feasibility=False → project_qp 逐位等价现状 bit-identical）──
        self.recursive_feasibility = bool(recursive_feasibility)   # 开=project_qp 加"落点存在合规脱身机动"终端检查
        # terminal_mode（仅 recursive_feasibility=True 时生效·默认 'discrete'=现行为 back-compat）：
        #   'discrete'=旧 encounter_action_verification(dt_sim Euler·~3m 漂移)；
        #   'certv2'  =block1-SOUND cert_v2 backup-maneuver(uterm_terminal·配 provably·2026-07-25 任务A)。
        if terminal_mode not in ("discrete", "certv2"):
            raise ValueError(f"terminal_mode 须 ∈ {{'discrete','certv2'}}，得到 {terminal_mode!r}")
        self.terminal_mode = str(terminal_mode)
        self.terminal_dt_sim = float(terminal_dt_sim)              # 终端 maneuver_verified 积分步（运行时 0.5 快·证明性重放用 0.1）
        if not (self.terminal_dt_sim > 0.0):
            raise ValueError(f"terminal_dt_sim 须 > 0，得到 {terminal_dt_sim}")
        # 🔴 对抗审：certv2 积分器的 10s 边界钳要求 terminal_dt_sim 整除 10s（否则钳错拍=轨迹不忠实）。
        #   在此 raise（构造期硬失败）·非只靠 _integrate_maneuver_official 里的 assert（-O 会剥离）。
        if abs(round(_uterm.DECISION_DT / self.terminal_dt_sim) * self.terminal_dt_sim - _uterm.DECISION_DT) >= 1e-9:
            raise ValueError(f"terminal_dt_sim 须整除 {_uterm.DECISION_DT}s（10s 边界钳不错拍），得到 {terminal_dt_sim}")
        self._sc = statechart if statechart is not None else ColregsStatechart()
        # Node 4 兜底：紧急控制器(Alg.1)懒创建（首次 safe_action 用其 vessel_params/dt）+ ρ5 进入边沿 reset 用的 prev_rho
        self._ec: EmergencyController | None = None
        self._prev_rho: int = RHO_NO_CONFLICT

    def reset(self) -> None:
        """episode 边界：清状态机 + 紧急控制器 + ρ 边沿追踪。"""
        self._sc.reset()
        if self._ec is not None:
            self._ec.reset()
        self._prev_rho = RHO_NO_CONFLICT

    @property
    def rho(self) -> int:
        return self._sc.rho

    def set_goal(self, goal) -> None:
        """注入目标点 [x, y]（供 ρ0 朝目标锥安全集用）。goal=None → 清目标（锥不生效）。

        由集成层（ContinuousProjectionEnv 构造）从 env.goal_center 注入。锥关（goal_cone_half=None）
        时本值不被读取、无副作用。
        """
        self.goal = None if goal is None else np.asarray(goal, dtype=float)

    def _goal_cone_interval(self, s_ego, dt):
        """ρ0 朝目标锥安全集：返回 (a区间, ω区间)=((a_lo,a_hi),(w_lo,w_hi))（全轴对齐 box）。

        动力学闭式（主窗口亲验逐位）：θ_next=θ+ω·dt，v_next=v+a·dt。符号：ω>0=左转(port,增θ,CCW)。
        e = wrap_to_pi(θ_goal − θ_ego)，θ_goal=atan2(goal_y−ego_y, goal_x−ego_x)，Φ=self.goal_cone_half。
        w_max=self.w_max，a_max=self.a_max（RL 箱=传入盾的 box）。

        ω 两区制：
          · 锥内 |e|≤Φ：允许朝目标附近微调 → w_lo=max((e−Φ)/dt,−w_max)，w_hi=min((e+Φ)/dt,+w_max)。
          · 锥外 |e|>Φ：强制朝目标（航向误差单调减）。⚠️ 用 reach 下界（非"入锥"上界）：
              reach=(|e|−Φ)/dt=最小转到锥边界所需 |ω|；hi_mag=min(w_max,(|e|+Φ)/dt)；
              lo_mag=min(reach,w_max,hi_mag)。e>0→[lo_mag,hi_mag]；e<0→[−hi_mag,−lo_mag]。
            （naive"入锥上界"e=40°→lo>hi 空集；用 reach 下界 → 全角域 -179°..179° 恒非空。）
        a 区间（保正速）：a_lo=max(−a_max,(v_floor−v)/dt)，a_hi=min(+a_max,(v_max−v)/dt)。
        """
        goal = self.goal
        ego_x, ego_y = float(s_ego.position[0]), float(s_ego.position[1])
        theta_goal = float(np.arctan2(goal[1] - ego_y, goal[0] - ego_x))
        e = usv_dynamics.wrap_to_pi(theta_goal - float(s_ego.orientation))   # ∈ [−π, π]
        phi = float(self.goal_cone_half)
        w_max = self.w_max
        dt = float(dt)

        if abs(e) <= phi:                                          # 锥内：朝目标锥两侧微调
            w_lo = max((e - phi) / dt, -w_max)
            w_hi = min((e + phi) / dt, +w_max)
        else:                                                      # 锥外：强制朝目标（reach 下界，恒非空）
            reach = (abs(e) - phi) / dt
            hi_mag = min(w_max, (abs(e) + phi) / dt)
            lo_mag = min(reach, w_max, hi_mag)                     # ⚠️ lo_mag ≤ hi_mag 恒成立（reach≤(|e|+Φ)/dt 且 ≤w_max 已夹）
            if e > 0.0:                                            # 需增 θ（左转，ω>0）
                w_lo, w_hi = lo_mag, hi_mag
            else:                                                  # 需减 θ（右转，ω<0）
                w_lo, w_hi = -hi_mag, -lo_mag

        v = float(s_ego.velocity)
        a_max = self.a_max
        a_lo = max(-a_max, (self.goal_v_floor - v) / dt)          # 保 v_next ≥ v_floor（>0=强制离停船）
        a_hi = min(+a_max, (self.v_max - v) / dt)                 # 保 v_next ≤ v_max（顶速禁加）
        # ⚠️ 防御性可达裕度夹（2026-07-04 impl 自查）：单步动作权限有限——v 很低时 (v_floor−v)/dt 可 > a_max
        #   （默认 v_floor=2.0/dt=10/RL 箱 a_max=0.048 → v=0 时 a_lo=0.2 > a_hi=0.048=空区间）。若不夹 → colregs_interval
        #   空区间守卫会在 ρ0 误触 needs_fallback=True（空旷水域坠 Node 4 兜底），且 goal_cone_action 的 np.clip(lo>hi) 语义诡异。
        #   夹 a_lo≤a_hi = "一步够不到 floor 就尽力加速"（a_lo→a_hi=a_max=满舵加速离停船），保区间恒非空 + 意图忠实。
        a_lo = min(a_lo, a_hi)
        return (a_lo, a_hi), (w_lo, w_hi)

    def goal_cone_action(self, s_ego, u_desired, dt):
        """公开锥投影：把 u_desired 逐轴 clip 进 ρ0 朝目标锥区间，返回安全 [a, ω]。

        锥关（goal_cone_half=None 或 goal=None）→ 返回 None（=不生效，caller 保持原样 = bit-identical）。
        锥开 → 返回 clip 后 [a, ω]（**不推状态机、不碰碰撞约束**，供 s_obs=None 短路用）。
        """
        if self.goal_cone_half is None or self.goal is None:
            return None
        u = np.asarray(u_desired, dtype=float)
        if u.shape != (2,):
            raise ValueError(f"u_desired 须为 2 维 [a, ω]，得到 shape={u.shape}")
        if not np.all(np.isfinite(u)):
            raise ValueError(f"u_desired 含非有限值（NaN/inf）：{u}")
        (a_lo, a_hi), (w_lo, w_hi) = self._goal_cone_interval(s_ego, dt)
        a_safe = float(np.clip(u[0], a_lo, a_hi))
        w_safe = float(np.clip(u[1], w_lo, w_hi))
        return np.array([a_safe, w_safe], dtype=float)

    def colregs_interval(self, rho: int, s_ego, s_obs, dt=None):
        """给定 ρ + 当前态势，返回 (a 区间, ω 区间, give_way_dir, needs_fallback)。

        本节 U_colregs 全轴对齐：U_box ∩ U_colregs = [a_lo,a_hi]×[w_lo,w_hi]（可分离）。
        约定 ω<0 = 右转(starboard)，ω>0 = 左转(port)。

        ⚠️ dt 参数（统一态势盾·2026-07-04）：仅 ρ0 朝目标锥安全集需要（区间用 dt 换算转艏率/加速度）。
           dt=None（默认，如 project() legacy 调用）→ ρ0 保持全箱 pass = 逐位等价现状 bit-identical。
           dt 传入 + goal_cone_half/goal 均已配 → ρ0 走朝目标锥区间（防空旷水域游荡）。
        """
        a_lo, a_hi = -self.a_max, self.a_max
        w_lo, w_hi = -self.w_max, self.w_max
        give_way_dir: str | None = None
        needs_fallback = False

        if rho == RHO_NO_CONFLICT:
            # ρ0：默认无额外约束（pass=全箱）；统一态势盾开启时约束"朝目标锥"防游荡（硬约束）。
            if self.goal_cone_half is not None and self.goal is not None and dt is not None:
                (a_lo, a_hi), (w_lo, w_hi) = self._goal_cone_interval(s_ego, dt)
        elif rho == RHO_STAND_ON:
            w_lo, w_hi = -self.eps_omega, self.eps_omega          # |ω| ≤ ε_ω（保向）
            a_lo, a_hi = -self.eps_a, self.eps_a                  # |a| ≤ ε_a（保速）
        elif rho in (RHO_HEAD_ON, RHO_CROSSING):
            w_hi = -self.omega_turn                               # ω ≤ −ω_turn（COLREGs Rule14/15：右转）
            give_way_dir = "right"
        elif rho == RHO_OVERTAKE:
            turn_set = get_turning_act(s_ego, s_obs)              # 单一真相源（与 Alg.3/状态机同方向逻辑）
            left = float(next(iter(turn_set))[1]) > 0.0           # ATL(ω>0)=左 / ATR(ω<0)=右
            if left:
                w_lo = self.omega_turn                            # ω ≥ +ω_turn
                give_way_dir = "left"
            else:
                w_hi = -self.omega_turn                           # ω ≤ −ω_turn
                give_way_dir = "right"
        elif rho == RHO_EMERGENCY:
            needs_fallback = True                                 # ρ5：Node 1 不处理 → 交 Node 4（仅 box 投影占位）
        else:
            raise RuntimeError(f"未知 ρ={rho}（应 ∈ 0..5）")

        # 区间非空守卫（防异常参数/状态致空集；Node 1 默认参数下不触发，为 Node 2/鲁棒性预置）
        if a_lo > a_hi or w_lo > w_hi:
            needs_fallback = True
            a_lo, a_hi, w_lo, w_hi = -self.a_max, self.a_max, -self.w_max, self.w_max

        return (a_lo, a_hi), (w_lo, w_hi), give_way_dir, needs_fallback

    def project(self, s_ego, s_obs, u_desired) -> ProjectionResult:
        """推状态机一步 → 把 u_desired 投影进 U_box ∩ U_colregs（逐轴区间裁剪=精确投影）。"""
        u = np.asarray(u_desired, dtype=float)
        if u.shape != (2,):
            raise ValueError(f"u_desired 须为 2 维 [a, ω]，得到 shape={u.shape}")
        if not np.all(np.isfinite(u)):
            raise ValueError(f"u_desired 含非有限值（NaN/inf）：{u}")

        rho = int(self._sc.step(s_ego, s_obs))
        (a_lo, a_hi), (w_lo, w_hi), give_way_dir, needs_fallback = self.colregs_interval(rho, s_ego, s_obs)

        a_safe = float(np.clip(u[0], a_lo, a_hi))
        w_safe = float(np.clip(u[1], w_lo, w_hi))
        u_safe = np.array([a_safe, w_safe], dtype=float)
        # corrected = 是否真改动了动作：np.clip 对 in-range 值返回【逐位原值】→ 精确比较即可，无需容差
        #   （避免 <1e-12 的极小修正被误报 False、日志把"盾介入率"误读为"从未介入"，对抗复核 minor 已修）。
        corrected = bool(not np.array_equal(u_safe, u))
        return ProjectionResult(
            u_safe=u_safe,
            rho=rho,
            corrected=corrected,
            needs_fallback=needs_fallback,
            give_way_dir=give_way_dir,
        )

    def project_qp(self, s_ego, s_obs, u_desired, dt, vessel_params, taus=None, obs_width=None):
        """Node 2c：把 u_desired 投影进 U_box ∩ U_colregs ∩ U_collision-free（QP，蓝图 §4 路线1=直接在 H-多胞形投影）。

        与 project()(Node 1) 同契约：内部推状态机一步、**每决策步只调一次**（别和 project() 混调）。
        流程：判 ρ → 取 U_box∩U_colregs 区间(轴对齐) → 以"合规标称动作"u_box 为线性化基点造无碰撞线性约束(每 τ 一条)
              → QP 解最近可行点。**P=∅（合规∩无碰撞冲突 / 退化 / ρ5 紧急）→ needs_fallback=True 交 Node 4，绝不静默放行。**
        ⚠️ taus 默认 [dt]（单步=档位A 最小闭环，蓝图 §6）；多 τ 须传【各 τ 把 ego 与 obs 都预测到 τ】的语义——
           当前 collision_free_constraint 是单步(ego@dt vs obs@τ)、tau≠dt 会 raise，故多前瞻须等 Node 2c 多 τ 几何落地(03 L34)。
        ⚠️ 计算优化（D31）：QP 用 cvxpy(OSQP)=正确性优先 + 干净 infeasible 状态；Node 3 吞吐基准时换直接 OSQP 缓存 setup + 解析 Jacobian。
        """
        u = np.asarray(u_desired, dtype=float)
        if u.shape != (2,):
            raise ValueError(f"u_desired 须为 2 维 [a, ω]，得到 shape={u.shape}")
        if not np.all(np.isfinite(u)):
            raise ValueError(f"u_desired 含非有限值（NaN/inf）：{u}")
        if not (float(dt) > 0.0):
            raise ValueError(f"dt 须 > 0，得到 {dt}（复审 B3）")
        if (abs(self.a_max - float(vessel_params.a_max)) > 1e-12
                or abs(self.w_max - float(vessel_params.w_max)) > 1e-12):   # ⚠️ 复审 B-EMERGENCY-BOX/A7：盾 box 必须 == 动力学 box
            raise ValueError(
                f"projection box(±{self.a_max},±{self.w_max}) 须 == vessel_params box(±{vessel_params.a_max},±{vessel_params.w_max})"
                f"；否则 emergency 越 box / 无碰撞约束 box 失真（复审 B-EMERGENCY-BOX/A7）"
            )

        rho = int(self._sc.step(s_ego, s_obs))
        # ⚠️ 传 dt → ρ0 走朝目标锥安全集（若 goal_cone_half/goal 已配；否则 colregs_interval 内部 pass=全箱 bit-identical）。
        (a_lo, a_hi), (w_lo, w_hi), give_way_dir, needs_fallback = self.colregs_interval(rho, s_ego, s_obs, dt=dt)
        # 合规标称动作 = u_desired 在 U_box∩U_colregs 上的精确投影（=Node 1 逐轴 clip、∈box 满足 CODE-2）；既是兜底占位、也是线性化基点。
        u_box = np.array([float(np.clip(u[0], a_lo, a_hi)), float(np.clip(u[1], w_lo, w_hi))], dtype=float)

        def _fallback():
            return ProjectionResult(
                u_safe=u_box, rho=rho,
                corrected=bool(not np.array_equal(u_box, u)),
                needs_fallback=True, give_way_dir=give_way_dir,
            )

        if needs_fallback:                              # ρ5 紧急 或 U_box∩U_colregs=∅ → 交 Node 4，不做无碰撞 QP
            return _fallback()

        taus = [dt] if taus is None else list(taus)
        if not taus:        # ⚠️ 复审 B1(MAJOR)：taus=[] 会跳过所有无碰撞约束 = 静默放行碰撞动作 → 必须 raise（默认单步传 None）
            raise ValueError("taus 不可为空（空=跳过无碰撞约束=静默放行不安全动作，复审 B1）；默认单步请传 None")
        rows = []
        for tau in taus:
            g, h, _dist, _dsafe = collision_free_constraint(s_ego, s_obs, u_box, tau, dt, vessel_params)
            if g is None:                               # 退化/圆心重合 = P=∅ 候选 → 交 Node 4，绝不静默放行
                return _fallback()
            rows.append((g, h))

        u_qp, feasible = _solve_box_halfplane_qp(u, (a_lo, a_hi), (w_lo, w_hi), rows)
        if not feasible:                                # U_box∩U_colregs∩U_collision-free = ∅ (P=∅) → 交 Node 4
            return _fallback()

        u_safe = np.asarray(u_qp, dtype=float)
        # ── N1 档位B*：递归可行性终端检查（默认关=recursive_feasibility False 时整块跳过=逐位等价）。
        #    落点无合规脱身机动 → 退兜底（走与 P=∅ 同一 _fallback 出口，交 safe_action 的 relaxed/emergency，绝不静默放行）──
        if self.recursive_feasibility and not self._terminal_ok(s_ego, s_obs, u_safe, rho, dt, vessel_params, obs_width):
            return _fallback()
        return ProjectionResult(
            u_safe=u_safe, rho=rho,
            corrected=bool(not np.allclose(u_safe, u, rtol=0.0, atol=1e-6)),  # QP 数值容差(OSQP 解非逐位)，非 Node1 的精确 array_equal
            needs_fallback=False, give_way_dir=give_way_dir,
        )

    def safe_action(self, s_ego, s_obs, u_desired, dt, vessel_params, taus=None, obs_width=None):
        """Node 2c + Node 4 总入口：把 u_desired 变成【总是可直接执行】的安全动作（投影优先、P=∅ 落兜底）。

        与 project()/project_qp() 同契约：内部经 project_qp 推状态机**一步**、每决策步只调一次（别和 project/project_qp 混调）。
        档位（蓝图 §4 + Krasowski Alg.1）：
          1) Node 2c QP 可行 → 'projection'（合规 ∩ 无碰撞投影动作）。
          2) needs_fallback 且 ρ5 紧急 → 'emergency'：复用 Phase-1 EmergencyController(Alg.1)，ρ5 进入边沿 reset、驻留续跑（同 SafeActionScheduler）。
          3) needs_fallback 且 ρ0-4（P=∅，合规与无碰撞冲突）→ 蓝图 §4 兜底【放松 COLREGs、保无碰撞】：
             3a) box ∩ 无碰撞 可行 → 'relaxed'（最近无碰撞动作、去 COLREGs 方向约束）；
             3b) 仍不可行（无一步动作能保 d_safe）→ 'collision_min'：软 QP 最小化最坏碰撞侵入 = 碰撞风险最小化 best-effort。
          4) 圆心重合/已碰（collision_free 返回 None 的 dist<1e-9 退化）→ 'degenerate'：无分离方向、返回 box 占位（≈已碰，Phase 5/多船边界）。
        ⚠️ 'relaxed'/'collision_min' 是 COLREGs **违规**动作（紧急冲突下 COLREGs 本无明确规范，蓝图 §4/§5 诚实标注）；
           违规/紧急步计数应按 source 归类（Node 3 评估接线）。⚠️ EC dt 取首次调用值（episode 内 dt 须一致）。
        ⚠️ **caller 必须在 episode 边界调 reset()**（清 EC mode + prev_rho）：否则新 episode 直接起于 ρ5 会漏判进入边沿、
           EC 沿用上一事件 mode = 静默错误动作（D13 头号风险，无运行时守卫、同离散 SafeActionScheduler；Node 3/env 须 env.reset()→proj.reset()）。
        ⚠️ **projection box 须 == vessel_params box**（否则 emergency 越 box / 约束失真；project_qp 已 assert，复审 B-EMERGENCY-BOX）。
        """
        res = self.project_qp(s_ego, s_obs, u_desired, dt, vessel_params, taus=taus, obs_width=obs_width)  # 推状态机一步 + Node2c（含 taus/dt 校验）
        rho = res.rho
        prev = self._prev_rho
        self._prev_rho = rho

        if not res.needs_fallback:
            return SafeActionResult(res.u_safe, rho, res.give_way_dir, source="projection", emergency_mode=None)

        if rho == RHO_EMERGENCY:                                    # (2) ρ5 紧急 → Alg.1 紧急控制器
            if self._ec is None:
                self._ec = EmergencyController(vessel_params=vessel_params, dt=float(dt))
            if prev != RHO_EMERGENCY:                               # 进入边沿 = 新紧急事件（同 SafeActionScheduler）
                self._ec.reset()
            a_em = np.asarray(self._ec.step(s_ego, s_obs), dtype=float)
            return SafeActionResult(a_em, rho, res.give_way_dir, source="emergency", emergency_mode=self._ec.mode)

        # (3) ρ0-4 P=∅：放松 COLREGs、保无碰撞（蓝图 §4）。重建无碰撞约束（基点用 box-clip，∈box 满足 CODE-2）。
        u = np.asarray(u_desired, dtype=float)
        u_box_full = np.array([float(np.clip(u[0], -self.a_max, self.a_max)),
                               float(np.clip(u[1], -self.w_max, self.w_max))], dtype=float)
        box, wbox = (-self.a_max, self.a_max), (-self.w_max, self.w_max)
        rows = []
        for tau in ([dt] if taus is None else list(taus)):
            g, h, _d, _ds = collision_free_constraint(s_ego, s_obs, u_box_full, tau, dt, vessel_params)
            if g is None:                                           # (4) 圆心重合/已碰 → 无分离方向
                return SafeActionResult(u_box_full, rho, res.give_way_dir, source="degenerate", emergency_mode=None)
            rows.append((g, h))
        u_relax, feasible = _solve_box_halfplane_qp(u, box, wbox, rows)   # 3a) box∩无碰撞（无 COLREGs）
        if feasible:
            return SafeActionResult(np.asarray(u_relax, float), rho, res.give_way_dir, source="relaxed", emergency_mode=None)
        u_min = _solve_collision_min_qp(u, box, wbox, rows)              # 3b) 软 QP 碰撞风险最小化 best-effort
        return SafeActionResult(np.asarray(u_min, float), rho, res.give_way_dir, source="collision_min", emergency_mode=None)

    def _terminal_feasible(self, s_ego, s_obs, u_applied, current_rho, dt, vessel_params) -> bool:
        """N1 档位B*（存在性级递归可行性）终端检查：施加 u_applied 后的落点 s' 是否【存在】一条合规脱身机动。

        返回 True=落点可行（放行 u_applied）；False=落点无脱身机动 → caller 退兜底（同 P=∅ 出口）。
        ⚠️ **存在性级**（user 2026-07-02 定）：测"落点是否存在合规脱身"（As(s')≠∅），非"u_applied 本身是否脱身"。
           实际驾驶的是 RL+单步投影的 receding-horizon 轨迹、非被验证脱身序列（执行期 deferred，usv_colregs:759/832）
           → 保证 = "每步落点仍存在合规脱身机动（区制内递归可行）"，**非**"沿脱身序列全程无碰"（后者=轨迹级，不可 claim）。
        ⚠️ 落点态势 ρ' 用【当前 ρ 播种的临时状态机】step(s') 严格复现路径依赖（Requirement 2：give-way 维持到 ¬cp）；
           **绝不用 fresh ρ0 克隆**——会把持续 give-way 误判为 ρ0→用错 As→归纳链裂（2026-07-02 亲验：分岔状态 fresh 判 ρ0、
           当前ρ播种判 crossing）。step() 输出仅依赖 (self.rho, s_l, s_m)、无其它可变态 → 当前ρ播种确定性复现真实 ρ'。
        ρ' 分档：ρ0→True（U_box 全箱非空·平凡）/ ρ1→True（akeep∈As）/ ρ2,3,4→encounter_action_verification 非空 /
           ρ5→True（紧急控制器 Alg.1 恒给动作·**但经验兜底非 provably·诚实 limitation D13：ρ5 不在可证明零碰撞内**）。
        """
        # 落点全状态（位置+朝向+速度）：复用已验证 dynamics.step（忠实 eq(1)，不手推积分）
        nxt = usv_dynamics.step(_ego_state_vec(s_ego), np.asarray(u_applied, dtype=float), dt, vessel_params)
        s_ego_n = VesselState(position=np.asarray(nxt[:2], dtype=float).copy(),
                              orientation=float(nxt[2]), velocity=float(nxt[3]), length=s_ego.length)
        s_obs_n = predict_state_cv(s_obs, dt)          # 他船恒速预测（规则态势保向保速，同 collision_free_constraint）
        # ρ' 用当前 ρ 播种临时状态机严格复现（无副作用·不碰 self._sc）
        tmp = ColregsStatechart(t_horizon=self._sc.t_horizon, t_pred=self._sc.t_pred,
                                dt=self._sc.dt, t_react=self._sc.t_react)
        tmp.rho = int(current_rho)
        rho_n = int(tmp.step(s_ego_n, s_obs_n))
        if rho_n in (RHO_NO_CONFLICT, RHO_STAND_ON, RHO_EMERGENCY):
            return True                                # ρ0 全箱 / ρ1 akeep∈As / ρ5 紧急兜底可用（经验·诚实 limitation）
        psi = {RHO_HEAD_ON: "head_on", RHO_CROSSING: "crossing", RHO_OVERTAKE: "overtake"}[rho_n]
        a_s = encounter_action_verification(s_ego_n, s_obs_n, psi,
                                            dt_sim=self.terminal_dt_sim, vessel_params=vessel_params)
        return len(a_s) > 0

    def _terminal_ok(self, s_ego, s_obs, u_applied, current_rho, dt, vessel_params, obs_width=None) -> bool:
        """终端检查 dispatch（recursive_feasibility=True 时 project_qp 调用）。terminal_mode 选判据。"""
        if self.terminal_mode == "certv2":
            return self._terminal_feasible_certv2(s_ego, s_obs, u_applied, current_rho, dt, vessel_params, obs_width)
        return self._terminal_feasible(s_ego, s_obs, u_applied, current_rho, dt, vessel_params)

    def _terminal_feasible_certv2(self, s_ego, s_obs, u_applied, current_rho, dt, vessel_params, obs_width=None) -> bool:
        """N1 档位B* · cert_v2(block1 SOUND)版终端检查：落点 s' 是否 ∈A（∃ certified 恒速直行尾脱离·让路态要求合规首步）。
        比 _terminal_feasible（离散 encounter_action_verification·dt_sim Euler·~3m 漂移）严格 → 配 provably（任务A 2026-07-25；
        soundness 核心 uterm_terminal 已本机全测：0 假放行 + first_unsafe_t==block3.clearance_profile 逐点相等）。
        ⚠️ **待服务器闭环冒烟**（本机无 vesselmodels 跑不了官方 step）。
        ⚠️ obs_width=None → 保守 w=length（sound·悲观·会多退兜底）；env 传【真宽】才 recover 高率（见设计文档 §3 OPEN①）。
        ⚠️ s'/ρ' 计算与 _terminal_feasible 同（当前 ρ 播种·防持续 give-way 误判 ρ0·归纳链不裂）。"""
        nxt = usv_dynamics.step(_ego_state_vec(s_ego), np.asarray(u_applied, dtype=float), dt, vessel_params)
        s_ego_n = VesselState(position=np.asarray(nxt[:2], dtype=float).copy(),
                              orientation=float(nxt[2]), velocity=float(nxt[3]), length=s_ego.length)
        s_obs_n = predict_state_cv(s_obs, dt)
        tmp = ColregsStatechart(t_horizon=self._sc.t_horizon, t_pred=self._sc.t_pred,
                                dt=self._sc.dt, t_react=self._sc.t_react)
        tmp.rho = int(current_rho)
        rho_n = int(tmp.step(s_ego_n, s_obs_n))
        if rho_n in (RHO_NO_CONFLICT, RHO_STAND_ON, RHO_EMERGENCY):
            return True                                # ρ0 全箱 / ρ1 保向 / ρ5 紧急兜底（同 _terminal_feasible·经验·诚实 limitation D13）
        # 让路态 ρ2/3/4：要求【合规首步】cert_v2 certified 脱离存在（A∩U_colregs 非空）
        ego_vec = [float(s_ego_n.position[0]), float(s_ego_n.position[1]), float(s_ego_n.orientation), float(s_ego_n.velocity)]
        obs_vec = [float(s_obs_n.position[0]), float(s_obs_n.position[1]), float(s_obs_n.orientation), float(s_obs_n.velocity)]
        olen = float(s_obs_n.length)
        owid = float(obs_width) if (obs_width is not None and float(obs_width) > 0.0) else olen   # None→保守 w=length（sound over-approx）
        sign = -1 if rho_n in (RHO_HEAD_ON, RHO_CROSSING) else 0   # head_on/crossing→starboard(Rule14/15)·overtake→任意向(Rule13 两侧皆可=合规)
        # 🔴 对抗审 Finding D：uterm 的 Lipschitz 界写死 SR108 常量(A_MAX/W_MAX/V_MAX/L_SHIP)·须与本盾 vessel_params 一致·
        #   否则 a_max>0.24 时 A_MAX·hh 速度 overshoot 项欠估 L → 假放行(unsound)。非 SR108 → fail-fast·别静默假 certify。
        if not (abs(vessel_params.a_max - _uterm.A_MAX) < 1e-9 and abs(vessel_params.w_max - _uterm.W_MAX) < 1e-9
                and abs(float(self.v_max) - _uterm.V_MAX) < 1e-9 and abs(float(s_ego.length) - _uterm.L_SHIP) < 1e-9):
            raise ValueError(f"terminal_mode='certv2' 仅支持 SR108 常量(a_max={_uterm.A_MAX}/w_max={_uterm.W_MAX}/"
                             f"v_max={_uterm.V_MAX}/l={_uterm.L_SHIP})·得 a_max={vessel_params.a_max}/w_max={vessel_params.w_max}/"
                             f"v_max={self.v_max}/l={s_ego.length}（uterm Lipschitz 界写死这些·不一致会假放行）")
        integ = lambda e, segs, T, h: self._integrate_maneuver_official(e, segs, T, h, vessel_params)
        # 🔴 对抗审 Finding C：H=120(=uterm 默认+机动族 max 转 120s+ fuzz 验证的 regime)·非 statechart t_horizon(=420·未验+过保守)。
        #   引理1 凸性(过 CPA→永久增)使 120 足够充分证永久清·无需 420。
        in_A, _ = _uterm.successor_in_A(ego_vec, obs_vec, olen, owid, integ,
                                        H=120.0, h=self.terminal_dt_sim, require_omega_sign=sign)
        return bool(in_A)

    def _integrate_maneuver_official(self, ego_vec, segments, T, h, vessel_params):
        """分段常控积分（官方 usv_dynamics.step·10s 边界钳 v=执行口径·忠实 block3.integrate_maneuver_official）
        → (ts[N], traj[N,4], omega_seg[N-1])·供 uterm.cert_v2。h 须整除 10s（钳不错拍）。"""
        assert abs(round(_uterm.DECISION_DT / h) * h - _uterm.DECISION_DT) < 1e-9, \
            f"h={h} 须整除 {_uterm.DECISION_DT}s（否则 10s 边界钳错拍·轨迹不忠实）"
        n = int(round(T / h))
        x = np.asarray(ego_vec, dtype=float).copy()
        ts = [0.0]; out = [x.copy()]; oseg = []
        for i in range(n):
            a, w = _seg_at_maneuver(segments, i * h)
            oseg.append(abs(w))
            x = usv_dynamics.step(x, (a, w), h, vessel_params, clip_velocity=False)   # 窗内不截（忠实执行）
            t = (i + 1) * h
            if abs(t / _uterm.DECISION_DT - round(t / _uterm.DECISION_DT)) < 1e-9:     # 10s 边界钳 v（=usv_env clip_velocity=True）
                x[3] = float(np.clip(x[3], 0.0, self.v_max))
            ts.append(t); out.append(x.copy())
        return np.array(ts), np.array(out), np.array(oseg)


def _seg_at_maneuver(segments, t):
    """常控分段 → t 时刻 (a, ω)。dur=None → 吃到末尾（忠实 block3._seg_at / uterm）。"""
    acc = 0.0
    for a, w, dur in segments:
        if dur is None:
            return a, w
        if t < acc + dur - 1e-9:
            return a, w
        acc += dur
    return segments[-1][0], segments[-1][1]


# ============================================================================
# Node 2a：collision-free 约束的"原料"——他船预测占据圆 + 本船一步可达位置 + 位置 Jacobian
#   （全部 fact-based 自 Phase-1；分离超平面 + 标量一阶线性约束在 Node 2b/2c，本节只产原料）
#   ⚠️ 占据用【圆盘外接】保守 over-approximate（分离超平面数学最干净）；Jacobian 用【中心差分作用于
#      已验证 dynamics.step】（复用官方 vessel_dynamics_yp，避免手推 cos/sin 积分解析式出错）。
# ============================================================================

def _ego_state_vec(s_ego):
    """VesselState → dynamics.step 的 [px, py, θ, v]。"""
    return np.array([s_ego.position[0], s_ego.position[1], s_ego.orientation, s_ego.velocity], dtype=float)


def obstacle_occupancy_disk(s_obs, tau):
    """O_obs(τ)：他船 τ 秒后预测占据圆（恒速预测 + 船体外接圆 + dobs,safety 膨胀；规则态势保向保速）。

    返回 (center[2], radius)；radius = circumradius(l_obs) + dobs,safety(=2·l_obs)。
    ⚠️ **保守口径（对抗复核 minor，下游 Node 2c/4 勿误读）**：此圆盘相对【裸船体】保守（外接圆 + 350m 大裕度、
       物理无碰），但**非 maneuver_verified 的 box-Minkowski keep-out 的处处上界**（角方向 473.7m < 它的 618.7m）
       → 不可当作与 As(ρ) 离散验证集等价的安全集。
    ⚠️ **固定半径 = 规则态势专用**（保向保速 → 无过程不确定性增长，故半径不随 τ 增长）；紧急态势的 τ-增长全可达集
       （reach_radius_pm）留 Node 4。
    """
    if tau < 0.0:
        raise ValueError(f"tau 须 ≥ 0，得到 {tau}")
    pred = predict_state_cv(s_obs, float(tau))
    radius = _vessel_circumradius(s_obs.length) + DOBS_SAFETY_FACTOR * s_obs.length
    return np.asarray(pred.position, dtype=float).copy(), float(radius)


def ego_circumradius(s_ego):
    """本船外接圆半径（用已知本船宽 EGO_WIDTH=25.4，非保守 w=l 上界）。"""
    return float(_vessel_circumradius(s_ego.length, EGO_WIDTH))


def ego_next_position(s_ego, u, dt, vessel_params):
    """本船施加常值 u=[a,ω] 前向一个决策步 dt 后的位置 p_ego(s,u)（复用已验证 dynamics.step）。"""
    u = np.asarray(u, dtype=float)
    if u.shape != (2,) or not np.all(np.isfinite(u)):
        raise ValueError(f"u 须为 2 维有限 [a,ω]，得到 {u!r}")
    return np.asarray(usv_dynamics.step(_ego_state_vec(s_ego), u, dt, vessel_params)[:2], dtype=float)


def position_jacobian(s_ego, u0, dt, vessel_params, h: float = 1e-5):
    """J_p = ∂p_ego/∂u |_(s,u0)（2×2），中心差分作用于已验证 dynamics.step（避免解析推导出错）。

    列 0 = ∂p/∂a，列 1 = ∂p/∂ω。h 默认 1e-5（动作量级 a~0.24/ω~0.03 下中心差分精度足够）。
    """
    u0 = np.asarray(u0, dtype=float)
    if u0.shape != (2,) or not np.all(np.isfinite(u0)):
        raise ValueError(f"u0 须为 2 维有限 [a,ω]，得到 {u0!r}")
    if not (h > 0.0):
        raise ValueError(f"h 须 > 0，得到 {h}")
    J = np.zeros((2, 2), dtype=float)
    for k in range(2):
        du = np.zeros(2, dtype=float)
        du[k] = h
        p_plus = ego_next_position(s_ego, u0 + du, dt, vessel_params)
        p_minus = ego_next_position(s_ego, u0 - du, dt, vessel_params)
        J[:, k] = (p_plus - p_minus) / (2.0 * h)
    return J


# ============================================================================
# Node 2b：分离超平面 + 标量一阶展开 → 对 u 的一条线性无碰撞约束（§12.3.3）
#   构造（两圆盘）：分离方向 n=(p_nom−p_obs)/‖·‖；要求本船下一步沿 n 距他船圆心 ≥ d_safe；
#     p_ego(u) ≈ p_nom + J(u−u_nom) 一阶展开 → nJ·u ≥ d_safe−dist+nJ·u_nom → g·u ≤ h（g=−nJ, h=−rhs）。
#   d_safe = R_ego + R_obs（R_obs 已含 dobs,safety=2·l_obs）= 圆心最小安全间距。
#
# ⚠️⚠️ 安全设计点（"别埋祸患"，挂 Node 2c/Node 4）：d_safe(~562m) ≫ 单步动作权限(~12m) → 一步前瞻投影
#   只能纠"刚进安全区(dist≈d_safe)"的他船；他船已深入(dist≪d_safe)时本约束在 box 内【不可行(P=∅)】=
#   蓝图 §12.5 递归可行性问题（一步前瞻≠可证明安全）。本函数【只造约束、不判可行性】；
#   P=∅ 由 Node 2c 的 QP 检测 → 落 Node 4 紧急兜底，**绝不静默放行**。多前瞻时刻 τ + 档位B 不变集是正解。
#   ⚠️ 复审 CODE-1 校准（2026-06-15c → 2026-06-16 量化更正）：并非"所有 binding 都 box-不可行"——正横(abeam)近边界几何下
#     约束可 box-可行、QP 解带【区制相关】一阶线性化裕量：实测最坏 ρ0 全箱 ~6.8m / ρ3 钳ω ~2.3m / ρ1 双钳 ~0.1m
#     （旧注"≤~2.8m"只对 ρ3 成立、对 ρ0 低报；可复现界见 test_usv_projection.py ⑲，遵 DATA-1 落盘）。
#     无论哪档，被 d_safe 相对裸船体碰撞 ~343m 净空吸收、物理无碰（80000 算例 0 裸碰）= 档位A 经验性、非 P=∅ 静默放行。
# ============================================================================

_NJ_DEGEN_TOL = 1e-4   # ‖nJ‖ 退化阈值（≫FD噪声~3e-7、≪最小真nJ~0.87）；方向退化+binding→交兜底（对抗复核 MAJOR）


def collision_free_constraint(s_ego, s_obs, u_nom, tau, dt, vessel_params):
    """把"本船下一步占据 ∩ 他船 τ 预测占据 = ∅（含 d_safe 裕度）"线性化为对 u 的一条约束 g·u ≤ h。

    返回 (g[2], h, dist, d_safe)：g·[a,ω] ≤ h 即合规无碰撞（一阶近似）；
       binding ⟺ 标称动作 u_nom 违反它（g·u_nom > h ⟺ dist < d_safe）。
    返回 (None, None, dist, d_safe)：① 圆心重合/已碰（dist<1e-9，无分离方向）；或 ② 方向退化 nJ≈0 且 binding
       （P=∅，本船该方向无控制权限）→ 两类均**显式交 Node 4 兜底，不返回伪装可解约束**（对抗复核 MAJOR 已修）。
    ⚠️ **tau 必须 == dt（单步语义，已加运行时守卫，复审 CODE-3）**：tau≠dt 时比较 ego@dt vs obs@τ 不同时刻位置、
       几何无意义、会静默把危险判成安全。**多前瞻 τ ≠"直接循环本函数改 tau"**（那样 ego 仍只推进 dt = 错）——
       须由 Node 2c 设计为"每 τ 把 ego 与 obs 都预测到 τ"，见 03 L34/D30 挂起。
    """
    u_nom = np.asarray(u_nom, dtype=float)
    if u_nom.shape != (2,) or not np.all(np.isfinite(u_nom)):
        raise ValueError(f"u_nom 须为 2 维有限 [a,ω]，得到 {u_nom!r}")
    # ⚠️ 复审守卫 CODE-3（tau==dt 单步语义）：tau≠dt 比较 ego@dt vs obs@τ 不同时刻、几何无意义、会静默把危险判成安全。
    if abs(float(tau) - float(dt)) > 1e-9:
        raise ValueError(f"tau({tau}) 必须 == dt({dt})（单步语义）；多前瞻 τ 由 Node 2c 设计、须同步预测 ego（03 L34/D30）")
    # ⚠️ 复审守卫 CODE-2（u_nom ∈ U_box）：底层 vessel_dynamics_yp 内部硬限幅 u → 越界 u_nom 会让 Jacobian 该轴归零、
    #   rhs 用未限幅 u_nom = 静默劣化约束。契约：caller（Node 2c）须先把策略动作 clip 进 box 再作线性化基点。
    a_max, w_max = float(vessel_params.a_max), float(vessel_params.w_max)
    if abs(u_nom[0]) > a_max + 1e-12 or abs(u_nom[1]) > w_max + 1e-12:
        raise ValueError(f"u_nom={u_nom} 越界 U_box([±{a_max},±{w_max}])；caller 须先 clip（复审 CODE-2）")
    p_obs, R_obs = obstacle_occupancy_disk(s_obs, tau)        # R_obs 已含 dobs,safety
    R_ego = ego_circumradius(s_ego)
    d_safe = R_ego + R_obs
    p_nom = ego_next_position(s_ego, u_nom, dt, vessel_params)
    diff = p_nom - p_obs
    dist = float(np.linalg.norm(diff))
    if dist < 1e-9:
        return None, None, dist, float(d_safe)               # 无分离方向 → 兜底
    n = diff / dist                                          # 分离方向（他船→本船标称位）
    J = position_jacobian(s_ego, u_nom, dt, vessel_params)
    nJ = n @ J                                               # 1×2：∂(nᵀp_ego)/∂u
    rhs = float(d_safe - dist + nJ @ u_nom)                  # nJ·u ≥ rhs
    g = -np.asarray(nJ, dtype=float)                         # −nJ·u ≤ −rhs
    h = -rhs
    # ⚠️⚠️ 方向退化守卫（对抗复核 MAJOR，2026-06-15）：nJ≈0（本船静止/低速 → ∂p/∂ω=0、J 降秩；
    #   且他船在 Jacobian 左零空间方位，如 v=0、本船朝东、他船正北）→ 约束退化为 0·u≤h。若此时 binding
    #   （h<0 ⟺ dist<d_safe）则【对所有 u 不可行 = P=∅】→ **必须显式返回 None 交 Node 4 兜底**，
    #   绝不返回 0·u≤负数 这种"伪装可解"约束（否则 2c 若剔近零行/软化不可行 → 静默放行 = 以为避让实则没动）。
    #   阈值 1e-4 ≫ FD 噪声(~3e-7)、≪ 最小真 nJ(~0.87)；(1e-4, ~0.87) 灰区的可行性交 QP 判（Node 3 可标定）。
    if float(np.linalg.norm(nJ)) < _NJ_DEGEN_TOL and h < 0.0:
        return None, None, dist, float(d_safe)
    return g, float(h), dist, float(d_safe)


# ============================================================================
# Node 2c QP 求解器：min ½‖x−u_des‖² s.t. x∈[a]×[w] ∧ g·x≤h（路线1=H-多胞形直接投影，蓝图 §4）
#   ⚠️ 安全核心：infeasible(P=∅) 必须可靠返回 (None, False) → caller 落 Node 4 兜底，绝不回伪解。
#   ⚠️ 计算优化（D31）：cvxpy(OSQP)=正确性优先 + 干净 infeasible 状态；Node 3 吞吐基准换直接 OSQP 缓存 setup。
#   eps 收紧到 1e-9（2 变量 QP 极小、紧容差廉价）使非 binding 解 ≈u（否则 OSQP 默认 1e-3 对 ω~0.03 量级太粗）。
# ============================================================================

def _solve_box_halfplane_qp(u_des, a_int, w_int, rows):
    """返回 (x[2], True) 若可行（已投影最近点）；(None, False) 若 P=∅/不可解（交兜底）。

    ⚡ 吞吐优化（D31，2026-06-16c）：直接 OSQP（避免 cvxpy 每次重编译，~11× 提速 1879→176μs；
    cvxpy QP 实测占 safe_action ~44% = 主线最大瓶颈、非 Jacobian）。min ½‖x−u_des‖² s.t. box + 半平面 g·x≤h
    → P=I, q=−u_des, A=[I; g 行], l=[box_lo; −∞], u=[box_hi; h]。**与旧 cvxpy(OSQP) 实现等价**：
    2000 fuzz 最近点差 1.1e-8、P=∅(可行性)检测 100% 一致（cvxpy 参考保留于 _solve_box_halfplane_qp_cvxpy、
    test ⑳ 永久守护等价；OSQP eps/max_iter 同旧值 1e-9/20000）。**安全语义不变**：infeasible→(None,False)→Node 4 兜底。"""
    import osqp
    from scipy import sparse
    u_des = np.asarray(u_des, dtype=float)
    A_rows = [[1.0, 0.0], [0.0, 1.0]]                      # box 两轴
    lo = [float(a_int[0]), float(w_int[0])]
    hi = [float(a_int[1]), float(w_int[1])]
    for g, h in rows:                                       # 半平面 g·x ≤ h（下界 −∞）
        A_rows.append([float(g[0]), float(g[1])])
        lo.append(-np.inf)
        hi.append(float(h))
    P = sparse.eye(2, format="csc")                         # ½xᵀPx + qᵀx = ½‖x−u_des‖² − const
    q = -u_des
    A = sparse.csc_matrix(np.asarray(A_rows, dtype=float))
    m = osqp.OSQP()
    try:                                                    # setup 一并纳入 try：退化/空 box(l>u)/非有限输入 → 同 cvxpy 优雅兜底(None,False)（复审 A/B 一致 MINOR）
        m.setup(P=P, q=q, A=A, l=np.asarray(lo, dtype=float), u=np.asarray(hi, dtype=float),
                eps_abs=1e-9, eps_rel=1e-9, max_iter=20000, verbose=False)
        res = m.solve()
    except Exception:
        return None, False
    st = res.info.status
    if st in ("solved", "solved inaccurate") and res.x is not None and np.all(np.isfinite(res.x)):
        return np.asarray(res.x, dtype=float), True
    return None, False          # primal infeasible / unbounded / error → P=∅，交 Node 4


def _solve_box_halfplane_qp_cvxpy(u_des, a_int, w_int, rows):
    """旧 cvxpy(OSQP) 实现，仅 test ⑳ 等价参考用（生产已换直接 OSQP，见 _solve_box_halfplane_qp）。"""
    import cvxpy as cp
    u_des = np.asarray(u_des, dtype=float)
    x = cp.Variable(2)
    cons = [x[0] >= a_int[0], x[0] <= a_int[1], x[1] >= w_int[0], x[1] <= w_int[1]]
    for g, h in rows:
        cons.append(float(g[0]) * x[0] + float(g[1]) * x[1] <= float(h))
    prob = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(x - u_des)), cons)
    try:
        prob.solve(solver=cp.OSQP, eps_abs=1e-9, eps_rel=1e-9, max_iter=20000, verbose=False)
    except cp.error.SolverError:
        return None, False
    if prob.status in ("optimal", "optimal_inaccurate") and x.value is not None:
        return np.asarray(x.value, dtype=float), True
    return None, False          # infeasible / unbounded / error → P=∅，交 Node 4


def _solve_collision_min_qp(u_des, a_int, w_int, rows):
    """碰撞风险最小化 best-effort（蓝图 §4，box∩无碰撞 不可行时）：min t s.t. x∈box ∧ g·x−h≤t ∀rows。
    t=最坏碰撞约束侵入量；最小化它 = box 内最大化离他船净空。加微正则 ε‖x−u_des‖² 取唯一解、贴近期望。
    box 非空 → 恒可行、恒返回动作（绝不返 None；返回前再 clip 进 box 防 OSQP 边界数值）。返回 x[2]。"""
    import cvxpy as cp
    u_des = np.asarray(u_des, dtype=float)
    x = cp.Variable(2)
    t = cp.Variable()
    cons = [x[0] >= a_int[0], x[0] <= a_int[1], x[1] >= w_int[0], x[1] <= w_int[1]]
    for g, h in rows:
        cons.append(float(g[0]) * x[0] + float(g[1]) * x[1] - float(h) <= t)
    prob = cp.Problem(cp.Minimize(t + 1e-6 * cp.sum_squares(x - u_des)), cons)
    prob.solve(solver=cp.OSQP, eps_abs=1e-9, eps_rel=1e-9, max_iter=20000, verbose=False)
    val = u_des if x.value is None else np.asarray(x.value, dtype=float)   # 极端兜底（不该发生）：退回 box-clip 期望
    return np.clip(val, [a_int[0], w_int[0]], [a_int[1], w_int[1]])


# ============================================================================
# gap#1 迫近不可避证书（imminent-unavoidability certificate·`03` L165·正式命题3）
#   `Paper/可证明层_正式命题_0707.md` 命题3。**纯分析工具·不在 QP 控制路径**（本节不被 safe_action/
#   _terminal_feasible 调用 → 盾行为逐位不变 bit-identical）。替换旧 unsound turn-only（L138/L139·只查
#   转向逃逸·漏加减速→"不可避"侧 45% 假阳不 sound）。本证书 = SOUND 充分条件（对抗审 0假阳/~2.9万场景·
#   主窗口硬化 v_bnd=v_max+a_max·dt=11.9 + L_lat 转90°后直行包络=全 t sound·`03` L165）。
# ── 判据：单恒速他船·∃t*∈[0,T]: 本船可达中心过近似 R_box(t*) ⊆ 他船体 O(t*)⊕disk(r_insc) ⟹ 迫近不可避。
#   soundness：① R_box 过近似真可达中心（‖dv⃗/dt‖≤a_eff·双积分→纵 ½a_eff t²/横 L_lat 转向限）②内切盘
#   r_insc=½船宽⊆船体（任朝向）③单时刻全可达落必撞区⟹每条轨迹此刻撞（正式命题3 三步证明）。
# 🚫 诚实作用域：迫近·单CV障碍·per-pair·**非**完备/早预警/多障碍/机动他船·no-fire≠可避（单侧充分）。
# ============================================================================

def _dist_point_to_rect(p, center, theta, length, width) -> float:
    """点 p 到旋转矩形（中心 center·艏向 theta·长 length[沿艏向]·宽 width[法向]）的欧氏距离（内部=0）。"""
    d = np.asarray(p, dtype=float) - np.asarray(center, dtype=float)
    c, s = np.cos(theta), np.sin(theta)
    lx = d[0] * c + d[1] * s                     # 局部：沿长（艏向）轴
    ly = -d[0] * s + d[1] * c                    # 局部：沿宽（法向）轴
    dx = max(0.0, abs(lx) - 0.5 * length)
    dy = max(0.0, abs(ly) - 0.5 * width)
    return float(np.hypot(dx, dy))


def _reach_params(vessel_params, dt_step: float | None = None):
    """gap#1 证书**硬化**可达参数（`03` L165·可测·防 un-harden 回归）：
      v_bnd = v_max + a_max·dt_step（步内速度上界·对抗审抓 v 步内不 clip 冲 11.9）；
      a_eff = √(a_max² + (v_bnd·w_max)²)（‖dv⃗/dt‖=√(a²+(vω)²) 的一致上界）。返回 (v_bnd, a_eff)。"""
    vp = vessel_params
    a_max, w_max, v_max = float(vp.a_max), float(vp.w_max), float(vp.v_max)
    dt = float(dt_step) if dt_step is not None else DT
    v_bnd = v_max + a_max * dt
    return v_bnd, float(np.hypot(a_max, v_bnd * w_max))


def _lateral_reach_bound(t: float, v_bnd: float, w_max: float) -> float:
    """本船横向可达上界 L_lat(t)（正式命题3·`03` L165/L167·转向限·全 t sound）：
      ω_max·t ≤ π/2 → (v_bnd/ω_max)(1−cos ω_max·t)（|sin Δθ|≤sin ω_max·s 内紧界）；
      ω_max·t > π/2 → (v_bnd/ω_max) + v_bnd·(t − (π/2)/ω_max)（转 90° 后全速横行包络·|sin|≤1）。
    **抽为模块级=两分支可单测**（防删 >π/2 包络的静默 unsound 回归·L167 finding②·此分支真实 fire 从不触及[t*≤~11s]
    但删掉会使 t>52.4s 处 L_lat 低估→过近似失效→理论 unsound）。"""
    half = np.pi / 2.0
    wt = w_max * t
    if wt <= half:
        return (v_bnd / w_max) * (1.0 - np.cos(wt))
    return (v_bnd / w_max) + v_bnd * (t - half / w_max)


def imminent_unavoidable_certificate(ego, obs, vessel_params, *, obs_width: float | None = None,
                                     t_horizon: float = 120.0, n_grid: int = 241,
                                     dt_step: float | None = None):
    """gap#1 SOUND 迫近不可避证书（正式命题3·`03` L165）。返回 (unavoidable: bool, t_star: float | None)。

    True  = 该 (本船, 他船恒速) 对下碰撞【迫近不可避】（任一可行控制序列都在 t_star 撞上他船体）。
    False = 未判定（**≠ 可避**·本判据单侧充分非必要·no-fire 不代表能避）。

    ⚠️ **纯分析函数·绝不改盾控制**（不被 safe_action 调用）。SOUND 充分条件（对抗审 0 假阳/~2.9万场景·L165）。
    假设（正式命题3·**必挂论文**）：单他船 · 他船恒速 CV(stand-on) · per-pair · 内切盘 r_insc=½船宽⊆船体 ·
      迫近（非早预警）· 单时刻 containment（充分非必要）。机动他船 / 多障碍 / 完备性 均**不覆盖**。
    参数：
      ego / obs   : VesselState（position/orientation/velocity/length）。obs 取恒速外推（predict_state_cv）。
      obs_width   : 他船宽。**soundness 前置条件（调用方保证）：obs_width ≤ 真实他船宽（under-approx）**——
                    传大于真实宽 → O 过近似 → 可能假阳（unsound·对抗审实证 obs_width=60 对真宽 25.4 会假阳）。
                    None → 默认 EGO_WIDTH=25.4（基准两船皆 SR108·精确）；他船更窄时调用方**须显式传真实宽**。
      dt_step     : 决策步长（默认 DT=10s；v_bnd=v_max+a_max·dt_step=步内速度上界，含 env clip 后超调）。
    """
    if t_horizon < 0.0:                                    # 防御 fail-fast（同 obstacle_occupancy_disk tau<0）
        raise ValueError(f"t_horizon 须 ≥ 0，得 {t_horizon}")
    if int(n_grid) < 1:                                    # n_grid<1 → 空 linspace → 静默不判=footgun·硬拒
        raise ValueError(f"n_grid 须 ≥ 1，得 {n_grid}")
    w_max = float(vessel_params.w_max)
    v_bnd, a_eff = _reach_params(vessel_params, dt_step)   # 硬化可达参数（v_bnd=11.9/a_eff·`03` L165·可测·防 un-harden 回归）

    r_insc = 0.5 * EGO_WIDTH                               # 本船内切盘半径=½船宽=12.7m（⊆船体·任朝向）
    L_obs = float(obs.length)
    W_obs = float(obs_width) if obs_width is not None else EGO_WIDTH
    p_e = np.asarray(ego.position, dtype=float)
    th_e = float(ego.orientation)
    v_e = float(ego.velocity)
    ve_vec = v_e * np.array([np.cos(th_e), np.sin(th_e)])
    h = np.array([np.cos(th_e), np.sin(th_e)])            # 艏向单位
    hp = np.array([-h[1], h[0]])                          # 法向单位

    for t in np.linspace(0.0, float(t_horizon), int(n_grid)):
        obs_t = predict_state_cv(obs, float(t))           # 他船恒速外推（复用已验证件）
        c_e = p_e + ve_vec * t                            # 本船恒速外推中心
        b_lon = 0.5 * a_eff * t * t
        b_lat = _lateral_reach_bound(t, v_bnd, w_max)     # L_lat（模块级·两分支可单测·L167 finding②）
        corners = (c_e + b_lon * h + b_lat * hp, c_e + b_lon * h - b_lat * hp,
                   c_e - b_lon * h - b_lat * hp, c_e - b_lon * h + b_lat * hp)
        # R_box(凸矩形) ⊆ O(t)⊕disk(r_insc)（凸）⟺ 4 角均到他船体矩形距离 ≤ r_insc（精确·无 shapely 近似）
        if all(_dist_point_to_rect(cn, obs_t.position, float(obs_t.orientation), L_obs, W_obs) <= r_insc
               for cn in corners):
            return True, float(t)
    return False, None
