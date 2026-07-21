"""
TRB 环境 · COLREGs 状态机 + 安全验证模块（step3）
=================================================
实现 Krasowski & Althoff 2024（锚点论文 2402.08502v2）的「安全盾」核心：
相遇态势分类 + 状态机 Γ（§IV）+ 安全验证 / 机动合成（§V）+ 违规率口径（§VII-A）。

⚠️ **fact-based 来源（全部亲自读，不靠转述）**：
  - 2024 论文 §IV（规范 + 状态机 Γ Fig.3）+ §V（机动合成 Alg.1-3）+ **Appendix-A Table IV**（全部谓词权威定义）。
  - Krasowski 2021 官方代码 `commonocean-rules`（GPL v3，L2：**只读参照、不 vendor、不抄**，引用其论文）——
    提供谓词「结构」+ 速度障碍几何参照；但**扇区边界以 2024 Table IV 为准**（见下「关键 fact」）。
  - 阈值三方交叉确认：2024 Table II/IV = 2021 `traffic_rules_ship.yaml` = `参考资料/文献核实笔记.md` ②。

⚠️ **关键 fact（建模时 catch，2026-06-09）**：clone 的 2021 代码 `position_predicates_ship.py` 的扇区函数
   （如 `in_behind_sector` 手算 = 艉向 ±45°）**已演进 / 不同于** 2024 论文 Table IV
   （behind = in_sector(112.5°, 247.5°) = 艉向 ±67.5°）。→ 本模块扇区**按 2024 Table IV 权威实现**，
   **不照搬 2021 代码扇区函数**。2021 代码仅作谓词「组合结构」+ 速度障碍几何参照。

⚠️ **2024 对 collision_possible 的两处再参数化（Appendix-A）**：
   ① 碰撞锥半径 `r_m = 3·l_m`（2024 论文文本权威值；论文称 [15] 用 l_m，但 clone 的 2021 代码
      `utils_ship.construct_velocity_obstacle` 实际用 l_l+l_m——本模块遵 2024 论文 3·l_m）
      → 检测「不保持至少 2 倍他船长安全距离」即算有碰撞可能；
   ② 不只查当前速度，查速度集 `V_l = {λ·unit_v(s_l) | λ ∈ [v−v_ε, v+v_ε]}`，v_ε = 1 m/s。

本模块分三小件**逐件建**（CLAUDE §2：每件 fact-based → 冒烟 → ≥2 独立 agent 复核 → 过了进下一件）：
  - **(a.1) 几何原语**（本文件当前内容）：扇区谓词 / 朝向谓词 / 速度谓词 / collision_possible（速度障碍）。
  - (a.2) 态势分类：head_on/crossing/overtake/keep + is_emergency（集合预测）+ is_emergency_resolved + persistent_X。← 下一件
  - (a.3) 状态机 Γ：ρ0-ρ5 转移逻辑。

角度约定：本 env 状态 θ ∈ [−π, π]（`usv_dynamics.wrap_to_pi`）。扇区用相对方位 β（**右舷 starboard 正 / 左舷 port 负**，
  与 `usv_observation` D6 一致），β = wrap(θ_ego − atan2(Δy, Δx))，实现 Table IV 扇区区间，避开论文 in_sector 抽取的符号歧义。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely.affinity import translate as _shp_translate
from shapely.geometry import LineString, Point, Polygon

from .usv_dynamics import make_vessel_params as _make_vessel_params

# ============================================================================
# 锁定阈值（fact-based，三方确认：2024 Table II/IV + 2021 yaml + 文献核实笔记②）
# ============================================================================
_DEG = np.pi / 180.0

# --- COLREGs 角度阈值（论文 Table II / Table IV / yaml traffic_rules_param）---
DELTA_HEAD_ON = 5.0 * _DEG       # Δhead-on：前扇区半角 + 对遇航向带
DELTA_OVERTAKE = 67.5 * _DEG     # 追越航向判定（max_orientation_diff_overtake）
DELTA_NO_TURN = 10.0 * _DEG      # 直航保向容许带（max_orientation_diff_no_change）
DELTA_LARGE_TURN = 20.0 * _DEG   # 让路明显转向阈值（sig_orientation_diff）
SECTOR_SIDE = 112.5 * _DEG       # 左/右扇区外边界（Table IV）
SECTOR_BEHIND_LO = 112.5 * _DEG  # 后扇区下界（Table IV）
SECTOR_BEHIND_HI = 247.5 * _DEG  # 后扇区上界（Table IV）

# --- 时间参数（论文 Table II，秒）---
T_HORIZON = 420.0   # collision_possible 检查时域（t_horizon_collision_possible）
T_PRED = 180.0      # 集合预测时域（is_emergency，a.2 用）
T_REACT = 60.0      # 反应时间
T_MANEUVER = 70.0   # 机动时间
DT = 10.0           # 决策步长（2021 yaml dt:10 / 论文 ∆t；Table II 无此项）

# --- collision_possible 几何（2024 Appendix-A 再参数化）---
R_M_FACTOR = 3.0    # r_m = 3·l_m（2024 论文文本权威；2021 代码实际用 l_l+l_m，本处遵 2024）
V_EPS = 1.0         # 速度集 ±v_ε (m/s)


@dataclass
class VesselState:
    """一艘船在某时刻的状态（适配本 env 的 [px, py, θ, v] 表示 + 船长）。

    position    : np.ndarray [x, y] (m)
    orientation : θ (rad)，艏向；本模块对任意输入内部 wrap，不要求预先归一化
    velocity    : v (m/s)，沿艏向标量速度（≥ 0）
    length      : 船长 (m)
    """
    position: np.ndarray
    orientation: float
    velocity: float
    length: float

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=float)
        if self.position.shape != (2,):
            raise ValueError(f"position 应为 2 维 [x, y]，得到 shape={self.position.shape}")
        if not np.all(np.isfinite(self.position)):
            raise ValueError(f"position 含非有限值（NaN/inf）：{self.position}")
        for nm, val in (("orientation", self.orientation), ("velocity", self.velocity),
                        ("length", self.length)):
            if not np.isfinite(val):
                raise ValueError(f"{nm} 非有限值（NaN/inf）：{val}")
        if self.length <= 0.0:
            raise ValueError(f"length 必须 > 0，得到 {self.length}")
        if self.velocity < 0.0:
            raise ValueError(f"velocity 必须 ≥ 0（沿艏向标量速度），得到 {self.velocity}")


# ============================================================================
# 角度 / 向量 原语
# ============================================================================
def wrap_to_pi(angle: float) -> float:
    """角度 wrap 到 [−π, π]（与 usv_dynamics 一致）。"""
    return (float(angle) + np.pi) % (2.0 * np.pi) - np.pi


def mod_2pi(angle: float) -> float:
    """角度 mod 到 [0, 2π)（Table IV 的 orientation 谓词用 mod(·, 2π)）。"""
    return float(angle) % (2.0 * np.pi)


def velocity_vector(s: VesselState) -> np.ndarray:
    """速度向量 v·[cos θ, sin θ]（= 2021 utils velocity_vector）。"""
    return s.velocity * np.array([np.cos(s.orientation), np.sin(s.orientation)])


def relative_bearing(s_l: VesselState, s_m: VesselState) -> float:
    """他船 m 相对本船 l 的方位角 β ∈ [−π, π]，**右舷(starboard)正 / 左舷(port)负**。

    β = wrap(θ_l − atan2(Δy, Δx))，与 usv_observation D6 约定一致。
    实现 Table IV 扇区：right = in_sector(+, +) 为正、left = in_sector(−, −) 为负，正负与此 β 对齐。
    """
    dp = s_m.position - s_l.position
    los = np.arctan2(dp[1], dp[0])               # 本船→他船 的视线角（绝对）
    return wrap_to_pi(s_l.orientation - los)     # 右舷正


# ============================================================================
# (a.1-i) 扇区谓词（2024 Table IV，权威；用 β 实现其区间，非 2021 代码扇区函数）
# ============================================================================
# Table IV：front=in_sector(−Δhead, Δhead) / left=in_sector(−112.5°,−Δhead)
#           / right=in_sector(Δhead, 112.5°) / behind=in_sector(112.5°, 247.5°)
# 用 β（右舷正）实现：4 扇区两两不重叠、覆盖整圆（区间取半开，边界归一侧）。
def in_front_sector(s_l: VesselState, s_m: VesselState) -> bool:
    """他船在本船前扇区（|β| ≤ Δhead-on，即 ±5°）。"""
    return abs(relative_bearing(s_l, s_m)) <= DELTA_HEAD_ON


def in_right_sector(s_l: VesselState, s_m: VesselState) -> bool:
    """他船在本船右(starboard)扇区（β ∈ (Δhead-on, 112.5°]）。"""
    b = relative_bearing(s_l, s_m)
    return DELTA_HEAD_ON < b <= SECTOR_SIDE


def in_left_sector(s_l: VesselState, s_m: VesselState) -> bool:
    """他船在本船左(port)扇区（β ∈ [−112.5°, −Δhead-on)）。"""
    b = relative_bearing(s_l, s_m)
    return -SECTOR_SIDE <= b < -DELTA_HEAD_ON


def in_behind_sector(s_l: VesselState, s_m: VesselState) -> bool:
    """他船在本船后扇区（|β| > 112.5°，即 Table IV [112.5°, 247.5°] = 艉向 ±67.5°）。"""
    return abs(relative_bearing(s_l, s_m)) > SECTOR_SIDE


# ============================================================================
# (a.1-ii) 朝向谓词（2024 Table IV）
# ============================================================================
def orientation_delta(s_l: VesselState, s_m: VesselState, max_orient_diff: float,
                      offset: float = 0.0) -> bool:
    """Table IV：mod(θ_m − θ_l + offset, 2π) ∈ [Δorient, 2π − Δorient]。

    = 两船航向差（加偏置 offset）**超过** max_orient_diff（即不在 ±max_orient_diff 带内）。
    head_on 用 offset=π（反平行带）；overtake 用 offset=0（同向带）。
    """
    diff = mod_2pi(s_m.orientation - s_l.orientation + offset)
    return max_orient_diff <= diff <= (2.0 * np.pi - max_orient_diff)


def orientation_towards_left(s_l: VesselState, s_m: VesselState,
                             head_on_angle: float = DELTA_HEAD_ON) -> bool:
    """Table IV：mod(θ_m − θ_l, 2π) ∈ [Δhead-on, π − Δhead-on]（他船朝向偏左）。"""
    diff = mod_2pi(s_m.orientation - s_l.orientation)
    return head_on_angle <= diff <= (np.pi - head_on_angle)


def orientation_towards_right(s_l: VesselState, s_m: VesselState,
                              head_on_angle: float = DELTA_HEAD_ON) -> bool:
    """Table IV：mod(θ_m − θ_l, 2π) ∈ [−π + Δhead-on, −Δhead-on]（他船朝向偏右）。

    [−π+Δ, −Δ] 在 [0, 2π) 下 = [π + Δhead-on, 2π − Δhead-on]。
    （论文 Table IV "Detects" 栏把本谓词误写成 "toward right"，公式区间为准——偏右。）
    """
    diff = mod_2pi(s_m.orientation - s_l.orientation)
    return (np.pi + head_on_angle) <= diff <= (2.0 * np.pi - head_on_angle)


# ============================================================================
# (a.1-iii) 速度谓词（2024 Table IV）
# ============================================================================
def drives_faster(s_l: VesselState, s_m: VesselState) -> bool:
    """Table IV：proj_v(s_l) > proj_v(s_m)（本船 l 比他船 m 快）。"""
    return s_l.velocity > s_m.velocity


def safe_speed(s_l: VesselState, v_max: float) -> bool:
    """Table IV：0 ≤ proj_v(s_l) ≤ v_max。

    注：R2 安全航速「由本船动力学天然保证」、**不进规则库**（论文 §IV-A），状态机不用它做规则；
    此谓词仅备查，v_max 强制方式见 usv_env clip_velocity（step4 裁）。
    """
    return 0.0 <= s_l.velocity <= v_max


# ============================================================================
# (a.1-iv) collision_possible（速度障碍 / 碰撞锥 CC'，2024 Appendix-A 再参数化）
# ============================================================================
def _circle_intersections(p1: np.ndarray, p2: np.ndarray, r1: float, r2: float,
                          d: float):
    """两圆交点（移植 2021 utils_ship.intersection_between_two_circles）。

    p1/p2 圆心，r1/r2 半径，d 圆心距。不相交返回 None（调用方已保证相交）。
    """
    if d <= 0.0 or d < abs(r1 - r2) or d > (r1 + r2):
        return None
    a = (r1 ** 2 - r2 ** 2 + d ** 2) / (2.0 * d)
    h_sq = r1 ** 2 - a ** 2
    if h_sq < 0.0:
        return None
    h = np.sqrt(h_sq)
    ex = (p2 - p1) / d                      # 圆心连线单位向量
    base = p1 + a * ex
    perp = np.array([-ex[1], ex[0]])        # 垂直单位向量
    return base + h * perp, base - h * perp


def _collision_cone(p_l: np.ndarray, p_m: np.ndarray, r_m: float) -> Polygon:
    """碰撞锥 CC'（顶点 p_l，经膨胀半径 r_m 的他船圆的两切点）。

    移植 2021 utils_ship.construct_velocity_obstacle 的切点构造（他船圆半径用 2024 r_m=3·l_m）：
    切点 = 「以 p_m 为心、r_m 为半径的圆」∩「以 p_l-p_m 中点为心、d/2 为半径的 Thales 圆」。
    调用方保证 d > r_m（见 collision_possible 的 too-close 守卫），故两圆必相交。
    """
    d = float(np.linalg.norm(p_l - p_m))
    mid = 0.5 * (p_l + p_m)
    pts = _circle_intersections(p_m, mid, r_m, d / 2.0, d / 2.0)
    if pts is None:                         # 守卫已保证相交；兜底 fail-fast
        raise RuntimeError(f"碰撞锥切点构造失败：d={d:.3f}, r_m={r_m:.3f}（应满足 d>r_m）")
    t0, t1 = pts
    return Polygon([p_l, t0, p_m, t1])


def collision_possible(s_l: VesselState, s_m: VesselState,
                       t_horizon: float = T_HORIZON) -> bool:
    """两船是否处于碰撞航向（2024 Table IV + Appendix-A 再参数化）。

    Table IV：collision_possible = (V_l ∈ CC') ∧ (‖v_l − v_m‖ ≥ ‖p_l − p_m‖ / t_horizon)。
      - CC'：碰撞锥，他船膨胀半径 r_m = R_M_FACTOR·l_m = 3·l_m（Appendix-A 再参数化①）。
      - V_l：速度集 {λ·unit_v(s_l) | λ ∈ [v−v_ε, v+v_ε]}（Appendix-A 再参数化②，v_ε=1）；
             λ 钳到 ≥0（物理 v≥0）。判 V_l 与 CC' **相交**（任一速度落锥内即算有碰撞可能，保守安全侧）。
             ⚠️ 「V_l ∈ CC'」论文记号是成员还是相交略有歧义，取保守（相交）解读（2 独立 agent 复核认可）。
    ⚠️ **已知保守偏差（2 独立 agent 复核 2026-06-09）**：接近条件用全心距 d（= Table IV 字面 + 2021 代码一致），
       故对「慢速尾追」存在相对速度带 ≈ [d·(1−r_m/d)/t_h, d/t_h) 内会在 t < t_horizon 实际相撞却判 safe 的检测盲区。
       collision_possible 是**态势分类**谓词（yaml 注 "situation detection"），**非最终安全保证**；硬安全由 a.2
       is_emergency（集合预测 t_pred=180s）兜底——**a.2 落地后须回归确认此盲区被集合预测覆盖**。
    """
    if t_horizon <= 0.0:
        raise ValueError(f"t_horizon 必须 > 0，得到 {t_horizon}")
    p_l, p_m = s_l.position, s_m.position
    d = float(np.linalg.norm(p_l - p_m))
    r_m = R_M_FACTOR * s_m.length

    # too-close 守卫：本船已在他船膨胀圆内 → 必有碰撞可能（同时保证下方切点构造合法 d>r_m）
    if d <= r_m:
        return True

    # 碰撞锥（顶点 p_l），按速度障碍平移 v_m（= 2021 VO.translate_rotate(v_other)）
    v_m_vec = velocity_vector(s_m)
    cone = _collision_cone(p_l, p_m, r_m)
    cone_vo = _shp_translate(cone, xoff=float(v_m_vec[0]), yoff=float(v_m_vec[1]))

    # 速度集 V_l：以 p_l 为锚的线段（λ ∈ [max(0,v−v_ε), v+v_ε]，沿本船艏向）
    unit_l = np.array([np.cos(s_l.orientation), np.sin(s_l.orientation)])
    lam_lo = max(0.0, s_l.velocity - V_EPS)
    lam_hi = s_l.velocity + V_EPS
    p_lo = p_l + lam_lo * unit_l
    p_hi = p_l + lam_hi * unit_l
    if np.allclose(p_lo, p_hi):
        v_in_cone = cone_vo.intersects(Point(p_lo))
    else:
        v_in_cone = cone_vo.intersects(LineString([p_lo, p_hi]))

    # 接近条件：相对速度足够快，能在 t_horizon 内闭合当前距离
    v_l_vec = velocity_vector(s_l)
    closing = float(np.linalg.norm(v_l_vec - v_m_vec)) >= (d / t_horizon)

    return bool(v_in_cone and closing)


# ============================================================================
# (a.2-i) 给路态势分类（2024 Table IV：用 a.1 原语组合 = 状态机 Γ 的 ρ1-ρ4 谓词）
# ============================================================================
# ⚠️ **Lemma 1 修正（本窗口实证发现，2026-06-09）**：论文 Lemma 1 称四者「至多一个真」，但**在字面
#   Table IV 不严格成立**——论文 Appendix-B 证明：case IV 第一支**显式误写** overtake 扇区为本船视角
#   in_behind_sector(s_l,s_m)（Table IV 实为他船视角 in_behind_sector(s_m,s_l)，物理上 overtake=本船在他船后、
#   他船视角才对）；case II/III 符号虽对，但其「=⊥」靠一个**对跨参考系无效**的几何论证（误引 case I「同船不能
#   在两扇区」，overtake 扇区在他船系、crossing/keep 在本船系）；两参考系解耦 → 实测固有重叠 {crossing∧overtake,
#   keep∧overtake}（随机采样 ~0.1-0.25%，seed 相关；冒烟 seed=1/N=5000 实测 7 例；独立 agent 200k 金标准 0
#   mismatch，谓词忠实 Table IV）。**本模块忠实 Table IV 不动谓词**。
#   同视角/被航向排除的对仍**严格互斥**（head_on 与任意、crossing∧keep = 0，冒烟验证扇区/朝向正确）。
#   重叠处理（a.3 状态机，2 agent 复核收紧措辞 2026-06-10）：crossing∧overtake（皆 give-way）由**确定性 tie-break
#   （head_on>overtake>crossing）解**；keep∧overtake **即时支实际落 keep**（persistent 需 ¬X(now)，R5>R6 未在即时支强制）——
#   assumption 5 + persistent 于 onset 前一步先进 ρ4 + is_emergency 兜底 → 连续轨迹不暴露，仅"裸生成进冲突区"才现（Phase 3 核对，02 挂起）。
# ⚠️ overtake / keep 的**参数对调**（Table IV，易错）：
#   overtake 用 in_behind_sector(s_m, s_l) = 本船 s_l 在他船 s_m 的后扇区（本船从后追）；
#   keep 第二支用 overtake(s_m, s_l) = 他船 s_m 在追越本船 s_l（本船被追越 → 直航）。
def head_on(s_l: VesselState, s_m: VesselState, t_horizon: float = T_HORIZON) -> bool:
    """对遇 give-way（ρ2，Table IV）：collision_possible ∧ 前扇区 ∧ 航向近反平行（¬orientation_delta offset π）。"""
    return bool(collision_possible(s_l, s_m, t_horizon)
                and in_front_sector(s_l, s_m)
                and not orientation_delta(s_l, s_m, DELTA_HEAD_ON, offset=np.pi))


def crossing(s_l: VesselState, s_m: VesselState, t_horizon: float = T_HORIZON) -> bool:
    """交叉 give-way（ρ3，Table IV）：collision_possible ∧ 右扇区 ∧ 他船朝向偏左。"""
    return bool(collision_possible(s_l, s_m, t_horizon)
                and in_right_sector(s_l, s_m)
                and orientation_towards_left(s_l, s_m, DELTA_HEAD_ON))


def overtake(s_l: VesselState, s_m: VesselState, t_horizon: float = T_HORIZON) -> bool:
    """追越 give-way（ρ4，Table IV）：collision_possible ∧ **本船在他船后扇区**(in_behind_sector(s_m,s_l))
       ∧ 本船更快(drives_faster) ∧ 航向同向带(¬orientation_delta(67.5°,0))。"""
    return bool(collision_possible(s_l, s_m, t_horizon)
                and in_behind_sector(s_m, s_l)         # 参数对调：本船 s_l 在他船 s_m 后扇区
                and drives_faster(s_l, s_m)
                and not orientation_delta(s_l, s_m, DELTA_OVERTAKE, offset=0.0))


def keep(s_l: VesselState, s_m: VesselState, t_horizon: float = T_HORIZON) -> bool:
    """直航 stand-on（ρ1，Table IV）：
       [collision_possible ∧ 左扇区 ∧ 他船朝向偏右] ∨ **他船在追越本船**(overtake(s_m,s_l))。"""
    crossing_standon = bool(collision_possible(s_l, s_m, t_horizon)
                            and in_left_sector(s_l, s_m)
                            and orientation_towards_right(s_l, s_m, DELTA_HEAD_ON))
    return crossing_standon or overtake(s_m, s_l, t_horizon)   # 参数对调：他船 s_m 追越本船 s_l


# ============================================================================
# (a.2-ii) 紧急规则谓词（2024 §IV-B 规则 R1）+ persistent_X（§IV-A 状态机转移触发）
# ============================================================================
# ⚠️ fact-based（亲读 2024 PDF §III-b 式(2)(3)(4) + §IV-A/B + Fig.3 + Table II，非转述）：
#   is_emergency ⟺ ∃t∈[t0,t0+t_pred]: O_pm(他船,点质量 Ω_pm,开环) ∩ O_traj(本船,Ω_yc,恒速 u_keep) ≠ ∅。
#   论文明文：velocity obstacle（= a.1 collision_possible）「不足以」检测 R1 的 imminent risk → 用集合预测。
#   论文用 set-based reachability（CORA）算可达集；本环境无 CORA/commonroad-reach（侦察确认）→ 档位 A 解析重建。
# ⚠️ **档位 A（user 拍板 2026-06-09，承诺不降质量）**：点质量 Ω_pm 的 v/a 约束各向同性（圆盘），
#   其位置可达集**精确就是圆盘**（非粗近似；数值 16000 子步验证经验可达半径 → ½·a_pm,max·t²，t=180 差 0.05m）。
#   zonotope 反而是拿多边形逼近此圆盘 → 档位 A 不弱于档位 B。
# ⚠️ **过近似裕度（防漏报，吸取 L4；2 独立 agent 复核措辞收紧 2026-06-09）**：连续 Ω_pm 的位置可达半径
#   **精确** = ½·a·t²（解析 + 收敛数值双验证）。落地半径 r_reach(t)=½·a_pm,max·t·(t+Δt_reach)
#   = 连续真值 + ½·a·t·Δt_reach，**正裕度严格 over 连续 Ω_pm**（真实他船连续运动 ⊂ 连续 Ω_pm by
#   reachset-conformance ⊂ 本圆盘 → 不漏报）。t=180 裕度 0.56%（紧、不过度保守）。
#   **与离散步长无关**：is_emergency 不积分点质量、仅解析调用 reach（Δt_reach 只是额外正裕度，非"包住某离散"）。
# ⚠️ deferred（交 agent 复核 + step4 核对）：① 他船宽度不在 state → 占据外接圆用保守上界 w=l；
#   ② 本船 shape 用 SR108 175×25.4（同 usv_termination）；③ resolved 第三支符号按物理语义取 ≥（论文疑 typo）。

# --- 紧急规则参数（2024 Table II，页 11 实抽）---
V_PM_MAX = 10.0            # 他船点质量模型最大速度 (m/s)
A_PM_MAX = 0.045           # 他船点质量模型最大加速度 (m/s²)
D_RESOLVED_FACTOR = 2.0    # d_resolved = 2·l_ego
DT_REACH = 1.0             # 可达圆盘过近似裕度子步 (s)；越小越紧、仍严格 over（数值验证）
RESOLVED_BEHIND_HALF = np.pi / 2.0   # is_emergency_resolved 后扇区半角 ±90°（论文 in_sector(3π/2,π/2)）
EGO_WIDTH = 25.4           # 本船 SR108 宽 (m)，vessel_1 实测，同 usv_termination
_MOVING_AWAY_TOL = 1e-9    # moving_away 点积容差：吸收 cos(π/2)≈6e-17 浮点误差，使「恰好垂直航向」按论文 ≤0 判为远离


def reach_radius_pm(t: float, dt_reach: float = DT_REACH) -> float:
    """他船点质量 Ω_pm 在未来 t 秒的位置可达圆盘半径（相对恒速外推中心），对**连续** Ω_pm 严格过近似。

    连续可达集位置半径**精确** = ½·a_pm,max·t²（各向同性圆盘；解析 + 收敛数值双验证）。
    本式 = ½·a_pm,max·t·(t+Δt_reach) = 连续真值 + ½·a_pm,max·t·Δt_reach，**正裕度严格 over 连续 Ω_pm**
    （防漏报根基；与离散步长无关——is_emergency 不积分点质量、仅解析调用本式）。Δt_reach 越小越紧。
    速度饱和上界 v_pm,max·t（切换点 t*=2·v_pm,max/a_pm,max=444s，t_pred=180s 内加速主导，min 取小者）。
    """
    if t < 0.0:
        raise ValueError(f"t 必须 ≥ 0，得到 {t}")
    r_acc = 0.5 * A_PM_MAX * t * (t + dt_reach)
    r_vsat = V_PM_MAX * (t + dt_reach)
    return min(r_acc, r_vsat)


def predict_state_cv(s: VesselState, t: float) -> VesselState:
    """恒速恒向外推 state 到未来 t 秒（u_keep=[0,0] 下式(1) 精确解 = 直线匀速）。"""
    v_vec = velocity_vector(s)
    return VesselState(position=s.position + v_vec * t,
                       orientation=s.orientation, velocity=s.velocity, length=s.length)


def _vessel_circumradius(length: float, width: float | None = None) -> float:
    """船体外接圆半径（保守占据）。width=None → 用保守上界 w=length（r=½√2·l，绝不漏船角）。

    ⚠️ 他船宽度不在 VesselState；step4 拿到场景 shape 可精化为 ½·√(l²+w²)（agent 复核 deferred）。
    """
    if width is None:
        width = length
    return 0.5 * float(np.hypot(length, width))


def _ego_rect(center: np.ndarray, theta: float, length: float, width: float) -> Polygon:
    """本船旋转矩形占据（中心 center、艏向 theta、长 length、宽 width）。"""
    hl, hw = 0.5 * length, 0.5 * width
    c, s = np.cos(theta), np.sin(theta)
    corners = [(+hl, +hw), (+hl, -hw), (-hl, -hw), (-hl, +hw)]
    pts = [(center[0] + x * c - y * s, center[1] + x * s + y * c) for x, y in corners]
    return Polygon(pts)


def _point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """点 p 到线段 [a,b] 的欧氏距离（含退化 a==b → 点距）。"""
    ab = b - a
    denom = float(ab @ ab)
    if denom <= 1e-18:                                   # 退化为点
        return float(np.hypot(p[0] - a[0], p[1] - a[1]))
    t = min(1.0, max(0.0, float((p - a) @ ab) / denom))
    proj = a + t * ab
    return float(np.hypot(p[0] - proj[0], p[1] - proj[1]))


def _segments_intersect(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> bool:
    """2D 线段 [p1,p2] 与 [p3,p4] 是否真相交（共线重叠返 False、由端点距=0 兜住）。"""
    def _cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    d1, d2 = _cross(p3, p4, p1), _cross(p3, p4, p2)
    d3, d4 = _cross(p1, p2, p3), _cross(p1, p2, p4)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def _segment_segment_distance(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> float:
    """2D 线段 [p1,p2] 与 [p3,p4] 最近距离（相交→0，否则 4 端点-线段距取小）。纯 numpy 闭式、O(1)。"""
    if _segments_intersect(p1, p2, p3, p4):
        return 0.0
    return min(_point_segment_distance(p1, p3, p4), _point_segment_distance(p2, p3, p4),
              _point_segment_distance(p3, p1, p2), _point_segment_distance(p4, p1, p2))


def is_emergency(s_l: VesselState, s_m: VesselState, t_pred: float = T_PRED,
                 dt_check: float = DT, dt_reach: float = DT_REACH,
                 obs_width: float | None = None, ego_width: float = EGO_WIDTH) -> bool:
    """2024 §IV-B 规则 R1：本船恒速(u_keep) 下他船可达占据与本船占据是否相交（集合预测）。

    他船 O_pm（开环点质量）：圆盘(恒速中心 p_m+v_m·t, 半径 reach_radius_pm(t)+他船外接圆)。
    本船 O_traj（闭环 u_keep）：旋转矩形(恒速恒向 p_l+v_l·t, 艏向 θ_l, 本船 SR108 shape)。
    时间区间 [t,t+dt_check] 保守覆盖：他船取圆心扫掠 capsule + 区间末半径；本船取首末矩形凸包。
    ∃ 采样区间相交 → True。采样 t=0,dt_check,...,t_pred。
    """
    if t_pred < 0.0:
        raise ValueError(f"t_pred 必须 ≥ 0，得到 {t_pred}")
    if dt_check <= 0.0:
        raise ValueError(f"dt_check 必须 > 0，得到 {dt_check}")
    r_obs = _vessel_circumradius(s_m.length, obs_width)
    v_l = velocity_vector(s_l)
    v_m = velocity_vector(s_m)
    # ── O(1) 远场保守早退（P1/D42·L40 等价提速、逐位不变行为）─────────────────────────────
    # 完整循环的他船占据 ⊆ capsule(他船中心线段[p_m,p_m+v_m·t_pred], 半径 reach_radius_pm(t_pred)+r_obs)，
    # 本船占据 ⊆ capsule(本船中心线段, 本船外接圆)；两 capsule 不交 ⟺ 两中心线段距离 > 半径和 → 完整循环必无交。
    # 故"线段距离 > 半径和"⟹ is_emergency 必 False，可 O(1) 短路（绝不漏检真紧急=严格保守；近场不满足→落回下方原循环）。
    # reach 用 t_pred+DT 裕度（D40#7 防离散化漏报）；早退 True⟹完整循环 False 由海量 fuzz 0 反例坐实（等价验证）。
    ego_circ = _vessel_circumradius(s_l.length, ego_width)
    _far_R = reach_radius_pm(t_pred + DT, dt_reach) + r_obs + ego_circ
    if _segment_segment_distance(s_m.position, s_m.position + v_m * t_pred,
                                 s_l.position, s_l.position + v_l * t_pred) > _far_R:
        return False
    # ── 近场：落回原 shapely 逐区间精确判定（一字未改）──────────────────────────────────────
    n = int(np.floor(t_pred / dt_check + 1e-9))
    for k in range(n + 1):
        t0 = k * dt_check
        t1 = min(t0 + dt_check, t_pred)
        # 他船区间占据：圆心 t0→t1 扫掠线段 buffer(区间末半径 + 外接圆) = capsule（保守覆盖区间）
        c0 = s_m.position + v_m * t0
        c1 = s_m.position + v_m * t1
        radius = reach_radius_pm(t1, dt_reach) + r_obs
        if np.allclose(c0, c1):
            obs_occ = Point(float(c0[0]), float(c0[1])).buffer(radius)
        else:
            obs_occ = LineString([(float(c0[0]), float(c0[1])),
                                  (float(c1[0]), float(c1[1]))]).buffer(radius)
        # 本船区间占据：首末矩形凸包（恒速恒向直线扫掠，凸 → 凸包覆盖中间）
        e0 = _ego_rect(s_l.position + v_l * t0, s_l.orientation, s_l.length, ego_width)
        e1 = _ego_rect(s_l.position + v_l * t1, s_l.orientation, s_l.length, ego_width)
        ego_occ = e0.union(e1).convex_hull
        if obs_occ.intersects(ego_occ):
            return True
    return False


def is_emergency_resolved(s_l: VesselState, s_m: VesselState,
                          d_resolved: float | None = None) -> bool:
    """2024 §IV-B：他船在本船后(±90°) ∧ 远离(朝向点积≤0) ∧ 距离≥d_resolved → 紧急解除。

    ⚠️ 论文第三支字面 '‖p_m−p_l‖₂ ≤ d_resolved' 与注释 'distance is large enough' 矛盾 = typo，
       按物理语义取 **≥**（同 r_goal typo 套路，诚实标注，写作须声明）。
    后扇区用 ±90°（论文 in_sector(3π/2,π/2)），**≠ a.1 in_behind_sector 的 ±67.5°**（两者本就不同语义）。
    d_resolved 默认 2·l_ego（Table II）。
    """
    if d_resolved is None:
        d_resolved = D_RESOLVED_FACTOR * s_l.length
    behind = abs(relative_bearing(s_l, s_m)) > RESOLVED_BEHIND_HALF
    u_l = np.array([np.cos(s_l.orientation), np.sin(s_l.orientation)])
    u_m = np.array([np.cos(s_m.orientation), np.sin(s_m.orientation)])
    moving_away = float(u_m @ u_l) <= _MOVING_AWAY_TOL   # 论文 ≤0；+tol 吸收垂直航向浮点误差（faithful + robust）
    far = float(np.linalg.norm(s_m.position - s_l.position)) >= d_resolved
    return bool(behind and moving_away and far)


def _persistent(pred_fn, s_l: VesselState, s_m: VesselState, dt: float = DT,
                t_react: float = T_REACT, t_horizon: float = T_HORIZON) -> bool:
    """2024 §IV-A：persistent_X = ¬X(now) ∧ G[dt,t_react](X 在恒速预测下持续为真)。

    {give_way}∈{crossing,head_on,overtake}（**不含 keep**）。两船恒速外推，[dt,t_react] 每采样点 X 真。
    语义：现在尚非 X，但未来反应窗内持续 X → 触发 ρ0→ρ2/3/4（提前进入应对）。
    """
    if pred_fn(s_l, s_m, t_horizon):           # ¬X(now)：现在已是 X → 非 persistent
        return False
    t = dt
    while t <= t_react + 1e-9:
        if not pred_fn(predict_state_cv(s_l, t), predict_state_cv(s_m, t), t_horizon):
            return False
        t += dt
    return True


def persistent_crossing(s_l: VesselState, s_m: VesselState, **kw) -> bool:
    return _persistent(crossing, s_l, s_m, **kw)


def persistent_head_on(s_l: VesselState, s_m: VesselState, **kw) -> bool:
    return _persistent(head_on, s_l, s_m, **kw)


def persistent_overtake(s_l: VesselState, s_m: VesselState, **kw) -> bool:
    return _persistent(overtake, s_l, s_m, **kw)


# ============================================================================
# (a.3) 状态机 Γ（2024 §IV-C + Fig.3 + Theorem 1：ρ0-ρ5 合规规则调度）
# ============================================================================
# fact-based（亲读 2024 §IV-C p6 + Fig.3 + Table I + COLREGS Requirement 1/2）：
#   6 状态：ρ0 无冲突 / ρ1 stand-on(keep,R6) / ρ2 head-on(R4) / ρ3 crossing(R3) / ρ4 overtake(R5) / ρ5 emergency(R1)。
#   优先级（Requirement 1）：R1 > R3-R5(give-way 等优先级) > R6(keep)。
#   维持（Requirement 2）：一旦进 give-way ρ2/3/4，维持同一机动直到 ¬collision_possible ∨ is_emergency。
#   转移（Fig.3 + Theorem 1）：
#     · 任意状态 is_emergency → ρ5（R1 覆盖一切）；ρ5 仅 is_emergency_resolved 退出（→ 同步重新评估）。
#     · ρ0：persistent_head_on/crossing/overtake → ρ2/3/4；keep → ρ1。
#     · ρ1(keep)：¬keep ∧ collision_possible → 即时分类进 ρ2/3/4；¬collision_possible → ρ0。
#     · ρ2/3/4：¬collision_possible → ρ0。
# ⚠️ stateful：维护当前 ρ，转移依赖当前状态（statechart 有记忆）；reset() 复位 ρ0。
# ⚠️ 工程决定（论文未逐一明确，诚实标注 + 交 agent 复核）：
#   ① ρ0→give-way 用 persistent_X（Fig.3 明确，避免瞬时误判频繁机动）；ρ1→give-way 用即时 X
#      （已在 collision_possible 冲突中且 ¬keep，无需再等持续性；论文正文 "R3-R5 apply" 即时语义）。
#   ② **tie-break 解 a.2-i 固有重叠**（crossing∧overtake 同真，皆 give-way）：确定性优先序 **head_on > overtake > crossing**
#      （head_on 正面最危险优先；overtake 按 COLREGS Rule 13 "追越判定凌驾交叉 Rule 15" 优先于 crossing）。固定序保证确定性、可复现。
#      ⚠️ **keep∧overtake 重叠不在此 tie-break 内**（2 agent 复核收紧）：即时支落 keep（persistent_overtake 需 ¬overtake(now)，
#      R5>R6 未在即时支强制）；assumption 5 + persistent onset 前一步先进 ρ4 + is_emergency 兜底 → 连续轨迹不暴露、安全。
#   ③ ρ5 经 is_emergency_resolved 退出后回 ρ0 并**同步重新评估**（Fig.3 emergency_resolved 回 normal）。
#   ④ 单 (ego, obstacle) pair（论文 assumption 2 双船）；多他船取最高优先级 pair，**deferred Phase 3/5**。
# ⚠️ assumption 5（论文）：初始无规则适用 → 从 ρ0 起；persistent 在"由无到有且持续"时触发，突变态势靠
#    is_emergency 兜底（give-way 用 persistent 可能延迟一步进，但真正危险 is_emergency 即时捕获，分层合理）。
# ⚠️ **cp 空隙（2 agent 复核，论文级缺陷传导）**：collision_possible 真但 keep/head_on/crossing/overtake 全假的几何空隙存在
#    （~12.6%，四谓词在扇区上叠航向带所致；论文 Prop1/Lemma1 的"cp 真⟹转移到某 ρi"对字面 Table IV 不完备）。落空隙时
#    ρ0 维持 ρ0 / ρ1 维持 ρ1（保守），imminent 由 is_emergency 兜底。非状态机偷工（a.2 谓词层论文缺陷传导）；写作引 Thm1 完备性须加此限定。
# ⚠️ 论文 typo（记录，不影响实现，印证 L6）：Theorem 1 case(III) 两处自引用 "case (III)" 应为 "case (II)"。

RHO_NO_CONFLICT = 0   # ρ0
RHO_STAND_ON = 1      # ρ1 keep (R6)
RHO_HEAD_ON = 2       # ρ2 (R4)
RHO_CROSSING = 3      # ρ3 (R3)
RHO_OVERTAKE = 4      # ρ4 (R5)
RHO_EMERGENCY = 5     # ρ5 (R1)

RHO_NAMES = {0: "no_conflict", 1: "stand_on", 2: "head_on", 3: "crossing", 4: "overtake", 5: "emergency"}


class ColregsStatechart:
    """COLREGs 合规状态机 Γ（2024 §IV-C Fig.3）。

    stateful：持当前状态 ρ，每步 step(s_l, s_m) 按谓词转移并返回新 ρ。
    单 (ego=s_l, obstacle=s_m) pair（论文双船 assumption 2）。
    """

    def __init__(self, t_horizon: float = T_HORIZON, t_pred: float = T_PRED,
                 dt: float = DT, t_react: float = T_REACT):
        self.t_horizon = t_horizon
        self.t_pred = t_pred
        self.dt = dt
        self.t_react = t_react
        self.rho = RHO_NO_CONFLICT

    def reset(self) -> None:
        """复位到 ρ0（assumption 5：每个交通情形初始无规则适用）。"""
        self.rho = RHO_NO_CONFLICT

    def _giveway_persistent(self, s_l: VesselState, s_m: VesselState):
        """ρ0 转入：持续性确认的 give-way 分类（tie-break 序 head_on > overtake > crossing）。"""
        kw = dict(dt=self.dt, t_react=self.t_react, t_horizon=self.t_horizon)
        if persistent_head_on(s_l, s_m, **kw):
            return RHO_HEAD_ON
        if persistent_overtake(s_l, s_m, **kw):
            return RHO_OVERTAKE
        if persistent_crossing(s_l, s_m, **kw):
            return RHO_CROSSING
        return None

    def _giveway_instant(self, s_l: VesselState, s_m: VesselState):
        """ρ1 转入：即时 give-way 分类（tie-break 序 head_on > overtake > crossing）。"""
        if head_on(s_l, s_m, self.t_horizon):
            return RHO_HEAD_ON
        if overtake(s_l, s_m, self.t_horizon):
            return RHO_OVERTAKE
        if crossing(s_l, s_m, self.t_horizon):
            return RHO_CROSSING
        return None

    def step(self, s_l: VesselState, s_m: VesselState) -> int:
        """推进一步：依当前 ρ + 两船 state 转移，返回新 ρ（同时更新 self.rho）。"""
        # R1 最高优先级：is_emergency 覆盖一切 → ρ5
        if is_emergency(s_l, s_m, t_pred=self.t_pred):
            self.rho = RHO_EMERGENCY
            return self.rho
        # ρ5：is_emergency 已 False；仅 is_emergency_resolved 才退出，否则维持 ρ5
        if self.rho == RHO_EMERGENCY:
            if not is_emergency_resolved(s_l, s_m):
                return self.rho
            self.rho = RHO_NO_CONFLICT   # 解除 → 回 ρ0，下面同步重新评估

        cp = collision_possible(s_l, s_m, self.t_horizon)

        # give-way（ρ2/3/4）：Requirement 2，维持同一机动直到 ¬collision_possible
        if self.rho in (RHO_HEAD_ON, RHO_CROSSING, RHO_OVERTAKE):
            if not cp:
                self.rho = RHO_NO_CONFLICT
            return self.rho

        # stand-on（ρ1 keep）
        if self.rho == RHO_STAND_ON:
            if not cp:
                self.rho = RHO_NO_CONFLICT
            elif not keep(s_l, s_m, self.t_horizon):
                gw = self._giveway_instant(s_l, s_m)   # 更高优先级 give-way 现身（即时）
                self.rho = gw if gw is not None else RHO_STAND_ON
            return self.rho

        # ρ0（含刚 resolved 落下）：重新分类。give-way 用 persistent；keep 即时；否则维持 ρ0
        gw = self._giveway_persistent(s_l, s_m)
        if gw is not None:
            self.rho = gw
        elif keep(s_l, s_m, self.t_horizon):
            self.rho = RHO_STAND_ON
        else:
            self.rho = RHO_NO_CONFLICT
        return self.rho

    def current_rule(self) -> str:
        """当前状态对应的规则名（供安全验证 (b) 件按状态选合规动作集）。"""
        return RHO_NAMES[self.rho]


# ============================================================================
# (b) 安全验证 / 机动合成（2024 §V：Alg.1 紧急 / Alg.2 BFS / Alg.3 会遇动作验证 + As(ρ)）
# ============================================================================
# fact-based（亲读 2024 §V p7-10 + Table II p11）。**本段先落地核心地基谓词 maneuver_verified（式8）**；
# 其余（Alg.1 紧急 3 模式 ahead/stern/base + tracking_controller[Appendix-C] / Alg.2 build_st BFS /
# Alg.3 encounter action verification / As(ρ) 三档）fact 已抽、设计见 `02`「下一步」，下一窗口续实现。
# Table II 新参数：Δahead=45° / Δstern=20° / a_stern=0.2·a_max / dobs,safety=2·l_obs / d_min,ahead=3·l_obs
#   / Δlarge_turn=20°（a.1 已有） / tm=40s / tmax,m=200s（=t_react+2·t_maneuver=60+2·70）。
# ⚠️ As(ρ)（§V-C + Theorem 2）：ρ0→Aregular(全 49 常规动作) / ρ5→a_em(Alg.1) / ρ1-4→Alg.3 输出。
#    论文用**离散 49 动作 + action masking**；本项目卖点 = 把 As(ρ) 改用**连续投影**（Phase 2-3）。

DELTA_AHEAD = 45.0 * _DEG          # 紧急 ahead 模式航向带（Table II）
DELTA_STERN = 20.0 * _DEG          # 紧急 stern 模式扇区偏移（Table II）
A_STERN_FACTOR = 0.2               # a_stern = 0.2·a_max（Table II）
DOBS_SAFETY_FACTOR = 2.0           # dobs,safety = 2·l_obs（机动验证他船膨胀，Table II）
DMIN_AHEAD_FACTOR = 3.0            # d_min,ahead = 3·l_obs（Table II）
T_M = 40.0                         # 机动段时间（Table II）
T_MAX_M = 200.0                    # 机动时域 = t_react + 2·t_maneuver（Table II）
# --- runaway 兜底守卫（2026-06-10 组件1 对抗复核 Agent B MINOR；论文实际值远在其下）---
_MAX_SUBSTEPS = 1_000_000          # maneuver_verified 单段子步数上界（论文 t_m/dt_sim≈80；防 absurd t_m 挂起）
_MAX_BFS_LEVELS = 1000             # build_st BFS 深度上界 = t_max_m/t_m（论文=5；防病态比值 runaway/ULP停滞）


def maneuver_verified(s_ego: VesselState, s_obs: VesselState, control_seq,
                      t_m: float = T_M, dt_sim: float = 0.5,
                      ego_width: float = EGO_WIDTH, obs_width: float | None = None,
                      t_horizon: float = T_HORIZON, vessel_params=None) -> bool:
    """2024 式(8)：本船沿机动 control_seq 前向仿真，验证 rule-compliant：
       ① 机动末 ¬collision_possible；② 全程本船占据 ∩ 他船占据(恒速 ukeep + dobs,safety 膨胀 Vobs+) = ∅。

    control_seq : [[a,ω], ...]，每段常值 hold t_m 秒（= Alg.2 的 a2u 序列；§V-B "tm 内 hold 常值前向仿真式(1)"）。
    他船 ukeep 恒速恒向（COLREGs 假设守规方保持航向航速）；Vobs+ = shape 膨胀 dobs,safety=2·l_obs。
    ⚠️ deferred / 歧义（交 agent 复核 + step4）：
      · 他船宽度不在 state → obs_width=None 用 l_obs（同 a.2-ii）；膨胀取每边 +dobs,safety（length/width 各 +2·dobs,safety，
        保守上界；论文 "enlarged ... for width and length" 未给精确边距）。
      · 本船用**带约束细步积分**（v 饱和[0,v_max]+位置用饱和后v）忠实 eq(1) 且位置-速度一致；dt_sim=0.5s 积分+占据步
        （实测漂移 1-3m）。**不用 step(clip_velocity)**——那有 D9 位置不一致（漂移随步长增大：dt=2→42m、**真实决策步 dt=10→215m**、dt=40→821m，靠大膨胀掩盖不算正确，2026-06-10 复核 + 审计量化）。
      · head-on 时他船恒速预测是保守的（论文 §V-B：他船本也该右转避让 → 本船需转更多才能解，恒速假设安全侧）。
    """
    if not control_seq:
        raise ValueError("control_seq 不能为空")
    if t_m <= 0.0 or dt_sim <= 0.0:
        raise ValueError(f"t_m/dt_sim 必须 > 0，得到 t_m={t_m}, dt_sim={dt_sim}")
    if vessel_params is None:
        vessel_params = _make_vessel_params()

    inflate = DOBS_SAFETY_FACTOR * s_obs.length          # dobs,safety = 2·l_obs
    obs_l = s_obs.length + 2.0 * inflate                  # 每边 +dobs,safety（保守上界）
    obs_w = (obs_width if obs_width is not None else s_obs.length) + 2.0 * inflate
    n_per_seg = max(1, int(round(t_m / dt_sim)))
    if n_per_seg > _MAX_SUBSTEPS:                         # absurd t_m/dt_sim → 单段子步爆炸/挂起（组件1 复核 Agent B 同源）
        raise ValueError(f"t_m/dt_sim={t_m / dt_sim:.0f} 子步数超上界 {_MAX_SUBSTEPS}（论文≈80）；防 runaway")

    # ② t=0 初始占据也要查（式8 ∀t∈[t0,t0+tend] **含 t0**；否则起步即重叠会被漏判 → 2 agent 复核 MAJOR-B 修复）
    ego_poly0 = _ego_rect(np.asarray(s_ego.position, dtype=float), float(s_ego.orientation),
                          s_ego.length, ego_width)
    obs_poly0 = _ego_rect(np.asarray(s_obs.position, dtype=float), float(s_obs.orientation), obs_l, obs_w)
    if ego_poly0.intersects(obs_poly0):
        return False

    # 本船前向仿真：**带约束细步积分**（v 饱和 [0,v_max] + 位置用饱和后 v）——忠实 eq(1)、位置-速度一致、不倒退/不超速。
    # ⚠️ **替代 step(clip_velocity=True)**（2026-06-10 正确性复核）：后者"位置按未截 v 积分、v 却被截"= D9 不一致
    #   （漂移随步长增大：dt=2→42m、真实决策步 dt=10→**215m**、dt=40→821m，靠 dobs,safety 大膨胀掩盖不算正确；减速时占据还会倒退）。本约束积分实测漂移 1-3m、无系统偏移。
    # a/ω 限幅 clip ±a_max/±w_max（= yp acceleration_constraints/yaw_constraints；已实测 yp 为纯 eq(1)：f=[cosθ·v,sinθ·v,ω,a]）。
    v_max, a_max, w_max = float(vessel_params.v_max), float(vessel_params.a_max), float(vessel_params.w_max)
    px, py = float(s_ego.position[0]), float(s_ego.position[1])
    th, v = float(s_ego.orientation), float(s_ego.velocity)
    t_elapsed = 0.0
    for action in control_seq:                           # 分段常值控制，跨段累积
        a = float(np.clip(action[0], -a_max, a_max))
        w = float(np.clip(action[1], -w_max, w_max))
        for _ in range(n_per_seg):
            v = min(max(v + a * dt_sim, 0.0), v_max)      # v 饱和 [0,v_max]（解 MAJOR-A：不 <0 崩溃、不超速）
            th = wrap_to_pi(th + w * dt_sim)
            px += v * np.cos(th) * dt_sim                 # 位置用饱和后 v（一致，无 D9 漂移）
            py += v * np.sin(th) * dt_sim
            t_elapsed += dt_sim
            ego_poly = _ego_rect(np.array([px, py]), th, s_ego.length, ego_width)
            s_obs_t = predict_state_cv(s_obs, t_elapsed)
            obs_poly = _ego_rect(s_obs_t.position, s_obs_t.orientation, obs_l, obs_w)
            if ego_poly.intersects(obs_poly):            # ② 全程占据不相交（t>0 各积分步）
                return False

    s_ego_end = VesselState(position=np.array([px, py]), orientation=th, velocity=v, length=s_ego.length)
    s_obs_end = predict_state_cv(s_obs, t_elapsed)
    return not collision_possible(s_ego_end, s_obs_end, t_horizon)   # ① 机动末 ¬collision_possible


# ============================================================================
# (b-2) Alg.2 build_st（BFS 带规则合规剪枝，2024 §V-B p9）+ a2u
# ============================================================================
# fact-based（亲读 2024 §V-B p8-9 伪代码 + p11 Table II，2026-06-10）：
#   a2u：动作 → 控制序列，每段常值 hold t_m 秒前向仿真式(1)（= maneuver_verified 的 control_seq 语义）。
#   build_st：从候选转向动作 ac 起，BFS 生长机动序列到 tmax,m，调 maneuver_verified（式8，✅ 已实现）验证。
#   生长规则（论文剪枝，line 15-27）：对树中每序列 a'——
#     · last(a')=ac（仍在转向段）→ 扩 [a',ac]（继续转） + 每个 [a',aacc]（aacc∈Aacc，转毕加速/保速）；
#     · last(a')≠ac（已切到某 aacc）→ 只扩 [a',last(a')]（**保持同一加速度、不切换** → 论文"does not switch
#       between different accelerations"）。
#   停：某深度 ≥1 序列 verified（line 7 退出）→ 返回该深度 verified 序列集；或 tend 到 tmax,m 仍无 → 返回 ∅（line 13）。
#   深度上界 = tmax,m/t_m = 200/40 = 5 段（tend ∈ n·t_m, n∈{1..5}）。复杂度 O(n·Nc·Nacc)。
# ⚠️ **返回语义（诚实标注，对论文伪代码的 1 处明确化）**：论文 Output 写"verified part of search tree G"、
#   而伪代码 line 28 `G←Gtemp` 含当前深度全部兄弟节点（含未验证）。本实现返回**该深度真正 verified 的序列子集**
#   （= Output 字面"verified part"），**非空 ⟺ ac 能在 tmax,m 内合成出合规机动**（Alg.3 只查非空，语义等价）。
#   运行时按树 parent-child 续选子节点（论文 §V-B 末）= 执行期，deferred Phase 3。
# ⚠️ ac 须"t_m 内转够 Δlarge_turn"由调用方（Alg.3 生成 Atr/Atl）保证；build_st 不强制、按传入 ac 处理。


def a2u(action_seq):
    """2024 §V-B：动作序列 → 控制序列（每段常值 hold t_m，喂 maneuver_verified）。

    action_seq : 可迭代的动作序列，每个动作 = (a, ω)。返回 [[a,ω], ...]（maneuver_verified 的 control_seq）。
    """
    return [[float(a[0]), float(a[1])] for a in action_seq]


def build_st(s_ego: VesselState, s_obs: VesselState, ac, a_acc,
             t_m: float = T_M, t_max_m: float = T_MAX_M, **mv_kwargs):
    """2024 Alg.2 build_st（BFS + 规则合规剪枝）：从候选转向动作 ac 合成 verified 机动序列。

    返回该深度 **verified 的机动序列集合**（每序列 = 动作 tuple，如 ((a,ω),(a,ω),...)）；
    **非空 ⟺ ac 能在 t_max_m 内合成出合规机动**（∅ = 无解）。详见上方段注「返回语义」。

    s_ego/s_obs : 当前本/他船 state。 ac : 候选转向动作 (a,ω)。 a_acc : 加速/保速动作集 Aacc。
    t_m : 机动段时间(40s)。 t_max_m : 机动时域上界(200s=t_react+2·t_maneuver)。
    **mv_kwargs : 透传 maneuver_verified（ego_width/obs_width/dt_sim/t_horizon/vessel_params）。
    """
    if t_m <= 0.0 or t_max_m <= 0.0:
        raise ValueError(f"t_m/t_max_m 必须 > 0，得到 t_m={t_m}, t_max_m={t_max_m}")
    if t_max_m / t_m > _MAX_BFS_LEVELS:                   # 病态比值 → BFS 深度 runaway/ULP 停滞（组件1 复核 Agent B MINOR-1）
        raise ValueError(f"t_max_m/t_m={t_max_m / t_m:.1f} 超 BFS 深度上界 {_MAX_BFS_LEVELS}（论文=5）；防 runaway")
    ac = (float(ac[0]), float(ac[1]))
    a_acc = [(float(a[0]), float(a[1])) for a in a_acc]
    if not all(np.isfinite(x) for x in (*ac, *(v for a in a_acc for v in a))):   # NaN/inf 否则穿透 shapely 抛 GEOSException（Agent B MINOR-2）
        raise ValueError(f"ac/a_acc 含非有限值（NaN/inf）：ac={ac}, a_acc={a_acc}")

    def _verified(node):
        return maneuver_verified(s_ego, s_obs, a2u(node), t_m=t_m, **mv_kwargs)

    # 深度 1（Alg.2 line 1-4）：单段候选 ac
    g = {(ac,)}
    verified = {node for node in g if _verified(node)}
    if verified:
        return verified

    # 生长（Alg.2 line 7-29）：tend 每轮 +t_m 到 t_max_m，某深度有 verified 即停
    tend = t_m
    while tend < t_max_m:                                  # line 10：tend<tmax,m 才继续生长
        tend += t_m
        g_temp = set()
        for node in g:                                    # line 15-27
            if node[-1] == ac:                            # last(a')=ac → 续转 ac + 每个 aacc
                g_temp.add(node + (ac,))
                for aacc in a_acc:
                    g_temp.add(node + (aacc,))
            else:                                         # last(a')≠ac → 只续同一 aacc（不切换加速度）
                g_temp.add(node + (node[-1],))
        g = g_temp                                        # line 28
        verified = {node for node in g if _verified(node)}
        if verified:                                      # line 7：当前深度有 verified → 停
            return verified
    return set()                                          # line 13：到 tmax,m 仍无 → ∅


# ============================================================================
# (b-3) Alg.3 会遇动作验证（2024 §V-B p9）+ get_turning_act + 候选/加速动作集 → As(ρ1-4)
# ============================================================================
# fact-based（亲读 2024 §V-B p8-9 + Fig.6 + Table II p11，2026-06-10 全 fact-derive 验证）：
#   论文离散动作网格 = 49（A_a×A_ω，Table II；= usv_env DISCRETE_ACTIONS，冒烟测试断言一致防漂移）。
#   会遇机动从「候选转向动作」起、调 build_st(Alg.2)→maneuver_verified(式8) 验证。As(ρ1-4) = 通过验证的候选集。
# ⚠️ **动作集成员（论文给 Fig.6"每方向 2 候选 ad,1/ad,2" + Δlarge_turn 要求，本处 fact-derive，交 agent 复核）**：
#   · 候选转向 ac 须 tm 内转够 Δlarge_turn=20° → |ω|·tm≥0.349 → |ω|≥0.00873 → A_ω 中资格者 {0.012,0.018}
#     （0.006×40s=13.8°<20° 不够，实测核）。右转=ω<0（starboard，实测 ω=-0.012 hold 40s→θ=-27.5°）。
#     → **ATR=((0,-0.012),(0,-0.018))（右转候选，a=0 纯转）/ ATL=((0,0.012),(0,0.018))（左转候选）**，各 2 个 = Fig.6 ad,1/ad,2。
#   · **AACC=((0,0),(0.016,0),(0.032,0),(0.048,0))**（§V-B "keep the speed or accelerate"→a≥0、ω=0；含 keep akeep=(0,0)）。
# ⚠️ As 累积所有 verified 候选（非 Alg.3 伪代码字面 `As←a` 覆盖；正文 "all actions that lead to verified maneuvers"，笔记⑧）。
# ⚠️ 论文用离散 As + action masking；**本项目卖点 = As(ρ) 改连续投影**（Phase 2-3）；本件先落论文忠实离散版（服务 step4 基线 + 真值）。
# ⚠️ 运行时按 build_st 返回的搜索树 G 续选子节点（论文 §V-B 末）= 执行期，deferred Phase 3；本件只算 As（非空判定足够）。

# --- 离散动作网格（2024 Table II；= usv_env，冒烟断言一致）---
A_A = (-0.048, -0.032, -0.016, 0.0, 0.016, 0.032, 0.048)        # A_a 常规加速度集 (m/s²)
A_OMEGA_GRID = (-0.018, -0.012, -0.006, 0.0, 0.006, 0.012, 0.018)  # A_ω 常规转艏率集 (rad/s)，0.006 修正值
A_KEEP = (0.0, 0.0)                                              # akeep：保持航向航速（stand-on ρ1 唯一动作）
ATR = ((0.0, -0.012), (0.0, -0.018))                            # 右转候选（ω<0 且 |ω|·tm≥Δlarge_turn），a=0
ATL = ((0.0, 0.012), (0.0, 0.018))                              # 左转候选（ω>0）
AACC = ((0.0, 0.0), (0.016, 0.0), (0.032, 0.0), (0.048, 0.0))   # 保速/加速集（a≥0, ω=0）


def get_turning_act(s_ego: VesselState, s_obs: VesselState, a_tr=ATR, a_tl=ATL):
    """2024 §V-B：overtake 会遇的转向方向选择。

    论文文字："turning direction is to the left if the orientation of the obstacle vessel is more to the
    right than the orientation of the ego vessel, and otherwise turning direction is to the right."
    → 他船朝向比本船**更靠右**（顺时针，wrap(θ_obs−θ_ego)<0）→ 返回左转集 a_tl；否则右转集 a_tr。
    （COLREGS Rule 13 未规定 overtake 转向侧，此为论文工程选择；实测符号核对见 fact-derive。）
    """
    rel = wrap_to_pi(s_obs.orientation - s_ego.orientation)
    return a_tl if rel < 0.0 else a_tr


def encounter_action_verification(s_ego: VesselState, s_obs: VesselState, psi_e: str,
                                  a_keep=A_KEEP, a_tr=ATR, a_tl=ATL, a_acc=AACC, **mv_kwargs):
    """2024 Alg.3 会遇动作验证 → 安全动作集 As(ρ1-4)。

    psi_e : 会遇态势 ∈ {"keep"(ρ1 stand-on), "head_on"(ρ2), "crossing"(ρ3), "overtake"(ρ4)}。
    返回 **As = 通过 build_st(式8) 验证的候选动作集**（set[(a,ω)]）：
      · keep   → {akeep}（stand-on，唯一动作 = 保持航向航速）；
      · head_on/crossing → Atr 中能合成 verified 机动的（give-way 永远右转）；
      · overtake → get_turning_act 选定方向集中能合成的。
    **空集 ⟺ give-way 无解**（应已升级 is_emergency / 交投影兜底 Phase 2-5）。运行时树 G deferred Phase 3。
    **mv_kwargs 透传 build_st→maneuver_verified（ego_width/obs_width/dt_sim/t_horizon/vessel_params/t_m/t_max_m）。
    ⚠️ **As 的"verified"精度限于 dt_sim 离散化（组件2 复核 Agent B MINOR-1，2026-06-10）**：maneuver_verified 用
       dt_sim=0.5 Euler 积分（漂移 1-3m，靠 dobs,safety=2·l 大膨胀吸收）；对 build_st 能合成的**最长最急转**
       （ω=0.018×5段=206°持续转，发生率~0.018% 仅 overtake 长转支）可能在 dt=0.5 判 verified、dt≤0.1 判否
       （擦 dobs,safety 安全膨胀带 ~2m）。**物理 collision-free 稳固**（裸障碍净空~350m，零碰撞）；仅"严格守满
       膨胀带"有 ~3m 离散裕度。**影响 "provably" 措辞精度（D5）→ 见 02 挂起：Phase 2/4 定收紧 dt_sim 或限定声明。**
    """
    if psi_e == "keep":
        a_keep_t = tuple(float(x) for x in a_keep)              # MINOR-2：akeep 形状/有限性守卫
        if len(a_keep_t) != 2 or not all(np.isfinite(x) for x in a_keep_t):
            raise ValueError(f"a_keep 须为 2 维有限 (a,ω)，得到 {a_keep!r}")
        return {a_keep_t}                                      # ρ1 stand-on：As = {akeep}
    if psi_e in ("head_on", "crossing"):
        a_temp = a_tr                                          # head_on/crossing：give-way 永远右转（Atr）
    elif psi_e == "overtake":
        a_temp = get_turning_act(s_ego, s_obs, a_tr, a_tl)     # overtake：按他船朝向选转向侧
    else:
        raise ValueError(f"psi_e 须 ∈ {{keep,head_on,crossing,overtake}}，得到 {psi_e!r}")

    a_s = set()
    for a in a_temp:                                          # 对每候选转向，build_st 非空 ⟺ 可合成 verified 机动
        if build_st(s_ego, s_obs, a, a_acc, **mv_kwargs):
            a_s.add(tuple(float(x) for x in a))               # 累积（非覆盖，笔记⑧）
    return a_s


# ============================================================================
# (b-4) Alg.1 紧急控制器（2024 §V-A 式6/7 + Alg.1 p8 + Appendix-C p17）→ As(ρ5) = {a_em}
# ============================================================================
# fact-based（亲读 2024 §V-A p7-8 正文 + Alg.1 伪代码 + Appendix-C 方程 + Fig.4/5 视觉渲染，2026-06-10）：
#   3 模式：ahead(式6，转90°；行驶距>d_min,ahead 未解 → 切 base) / stern(式7，加速序列 u_acc) /
#   base(兜底，绕他船艉)。Alg.1 = 有状态控制器：进 ρ5 首步定 mode + 记初始快照（line 1），每决策步
#   输出 u=[a,ω]=a_em（line 15）；唯一 mode 切换 = ahead→base（line 3-4）。
#   目标点：ahead 用**初始快照** (s_ego,0, s_obs,0) = 固定点（line 9）；base 用**当前** s_obs = 动目标（line 12）。
# ⚠️ 6 个论文未给闭式/有矛盾的工程决定（fact-derive + 诚实标注；2026-06-10 组件3 过门：2 对抗 agent
#    + 主窗口矢量复核裁决后 WD1/WD2 修正、WD6 补门控——初版像素目测读图两处读错，矢量抽取为准，见 03 L13）：
#   WD1 stern 扇区：式(7) 字面 in_sector(3π/2+Δstern, π/2+Δstern) 按 Table IV 的 [lo,hi] 区间语义
#       = 跨船头的前方扇区，与正文 "almost astern" 矛盾（typo 实锤，同 A_ω −0.06 先例）。
#       **Fig.4(c) 矢量实测**（pymupdf get_drawings，agent A + 主窗口独立复核一致）：灰区边界射线
#       β=71.6°/290.8°、Δstern 弧从右正横**向前**扫 20°、图例他船 β≈78-80° 在区内 → 图意 =
#       **正艉对称 ±(π/2+Δstern)**，即 β∈[π/2−Δstern, 3π/2+Δstern]=[70°,290°]。按图意实现。
#       （初版误读 [110°,290°] 为图意子集、保守不漏判但排除了论文自身示例场景，已改。）
#   WD2 Fig.5 转向判据：论文只给 4-case 图。**Fig.5 矢量实测**（同上双重复核）：四 case 朝向各不同
#       （θ¹=−60°/θ²=−40°/θ³=−155°/θ⁴=−132°，本船朝北），同位置仅差 20° 朝向即反号 → 转向**不是**
#       朝向差的函数（初版"与 §V-B get_turning_act 同构"判据被图证伪：case1/4 反号，恰是仅有的两个
#       ahead-eligible case）。图意判据 = **航迹侧**：本船在他船航迹线左舷（relative_bearing(s_obs,
#       s_ego)<0）→ 右转，右舷 → 左转（4/4 全中，case 余量 −10.6°/+9.4°/−20.0°/+3.0°）；物理 =
#       转离来船航迹、向其艉后半平面。β≈0（恰在航迹线上，含正对头）→ 确定性右转（Rule 14 精神）。
#   WD3 get_target_ahead 距离：Fig.4(b) 绿叉无公式（矢量实测 1.44·l_obs @ 方位~70°，示意性）→ 取
#       d_min,ahead=3·l_obs（Table II 唯一 ahead 距离参数；与 Alg.1 行驶距切换阈值同尺度自洽），可配置。
#   WD4 get_target_base 尾后距离：Fig.4(a) 绿叉在他船艉后轴线上（矢量实测距艉 0.65·l_obs，示意性）
#       无公式 → 中心距 = 0.5·l_obs + d_obs,safety=2·l_obs（Table II 挂钩），**并加 ego 尺度下限
#       2.5·l_ego**（B 复核 MAJOR-2：纯随 l_obs 缩放在 l_obs<0.8·l_ego 时目标点 < d_resolved=2·l_ego
#       → 完美跟踪也永不 resolved、贴身绕圈；下限使任意 l_obs 下目标 ≥ 2.5·l_ego > d_resolved；
#       论文域 l_obs≥l_ego 不触发=零行为变化），可配置，Phase 5 实测调。
#   WD5 tracking p_des：Appendix-C "either p_target or generated ... approximately maintains v_desired"
#       后者无公式 → p_des=p_target + v>v_desired 时 a 钳 ≤0（实现 approximately maintains，简单可核）。
#   WD6 tracking 奇点 + 后半平面加固：ω 分母 = −sin(2φ)（φ=航向到目标方位差）在 φ∈{0,±π/2,π} 为 0；
#       且 Vω=sin²φ 以 φ=π（背对目标）为另一极小点 → 论文公式在 |φ|>π/2 域会收敛到**背对**（实测
#       φ=3π/4 时公式给出加大偏差方向）。处理：|φ|<π/2 忠实用公式（该域内收敛 φ→0 正确；B 复核
#       20 万样本对拍逐点一致、收敛性全扫无背对/无 chattering）；|φ|≥π/2 → 满舵 ±ω_max 朝目标侧
#       （sinφ 符号；φ≈π 退确定性右转）。a 通道：Vω>ΔVω gate（33.2°<|φ|<146.8° 时 a=0；|φ|≥146.8°
#       Vω 复又 ≤ΔVω，公式给 a<0 减速=物理合理）；v≈0 起步守卫带 **cphi>0 门控**（B 复核 MINOR-3：
#       无门控时 v=0 背对目标会满加速远离）。输出恒 clip ±a_max/±ω_max。

V_DESIRED = 6.0         # 紧急跟踪期望速度 m/s（Appendix-C）
LAMBDA_1 = 4.0          # 转向 Lyapunov 增益 λ1（Appendix-C）
LAMBDA_2 = 0.04         # 加速 Lyapunov 增益 λ2（Appendix-C）
DELTA_V_OMEGA = 0.3     # Vω 阈值：超此只转向不加速（Appendix-C）
_SING_EPS = 1e-9        # 分母奇点判据
_TARGET_REACHED_EPS = 1e-6  # 已到目标点判据（m）


def turning_direction(s_ego: VesselState, s_obs: VesselState) -> int:
    """Fig.5 转向方向：+1 = 右转（ω<0）/ −1 = 左转（ω>0）。

    判据（WD2，矢量复核修正版）= **航迹侧**：β_me = relative_bearing(s_obs, s_ego)（本船在他船系
    的方位，右舷正）。β_me<0（本船在他船航迹线左舷）→ 右转 +1；β_me>0 → 左转 −1；β_me≈0
    （恰在航迹线上，含正对头）→ 确定性右转（Rule 14 精神）。= 转离来船航迹、向其艉后半平面；
    Fig.5 四 case 矢量几何 4/4 全中（初版朝向差判据 2/4 错，被图证伪——见段注 WD2 / 03 L13）。
    """
    beta_me = relative_bearing(s_obs, s_ego)
    if abs(beta_me) < 1e-9:
        return 1                                # 恰在他船航迹线上（含正对头）→ 右转
    return 1 if beta_me < 0.0 else -1


def ahead_emergency(s_ego: VesselState, s_obs: VesselState,
                    delta_ahead: float = DELTA_AHEAD) -> bool:
    """2024 式(6)（除 in(ρ5) 外两条；ρ5 由状态机/调用方保证——本谓词是 mode 选择）。

    = 他船在本船前扇区 ±Δahead **∧** 本船朝向与他船**反向**朝向差 ≤ Δahead（近对头）。
    ¬orientation_delta(·,Δahead,π) = 差(含 π 偏置)不超过 Δahead（orientation_delta 语义=「超过」）。
    """
    near_reversed = not orientation_delta(s_ego, s_obs, delta_ahead, offset=np.pi)
    in_ahead = abs(relative_bearing(s_ego, s_obs)) <= delta_ahead
    return bool(near_reversed and in_ahead)


def _in_stern_sector(s_ego: VesselState, s_obs: VesselState,
                     delta_stern: float = DELTA_STERN) -> bool:
    """式(7) 扇区（WD1，矢量复核修正版）：正艉对称 ±(π/2+Δstern)，即 |wrap(β−π)| ≤ π/2+Δstern
    （⟺ β mod 2π ∈ [π/2−Δstern, 3π/2+Δstern] = [70°, 290°]）。

    ⚠️ 论文字面 in_sector(3π/2+Δstern, π/2+Δstern) 按 Table IV 区间语义=跨船头前方扇区，与
    "almost astern" 矛盾（typo）；Fig.4(c) 矢量实测：边界射线 β=71.6°/290.8°、Δstern 弧从右正横
    **向前**扫、图例他船 β≈78° 在区内 → 图意=本式（2 复核 agent + 主窗口矢量抽取一致裁决，03 L13）。
    """
    b = relative_bearing(s_ego, s_obs)
    return abs(wrap_to_pi(b - np.pi)) <= (0.5 * np.pi + delta_stern)


def _pos_under_stern_acc(s_ego: VesselState, t: float, a_stern: float,
                         t_react: float, v_max: float) -> np.ndarray:
    """stern 模式 u_acc 下本船位置解析（直线运动 ω=0）：t≤t_react 匀加速（v 饱和 v_max）、之后匀速。

    弧长 s(t) 分段精确积分（无数值漂移）；v0>v_max 输入按 v_max 截（环境层 clip 后不应出现）。
    """
    if not (t >= 0.0):                                        # not(≥) 拒 NaN（B 复核 MINOR-1/2 风格）
        raise ValueError(f"t 必须 ≥ 0 且有限，得到 {t}")
    if not (a_stern >= 0.0 and np.isfinite(a_stern)):         # a<0 破坏弧长单调性→凸包覆盖前提失效
        raise ValueError(f"a_stern 必须 ≥ 0 且有限，得到 {a_stern}")
    if not (t_react >= 0.0):
        raise ValueError(f"t_react 必须 ≥ 0 且有限，得到 {t_react}")
    v0 = min(max(float(s_ego.velocity), 0.0), v_max)
    t_acc = min(t, t_react)                                   # 加速段时长（未饱和前提）
    t_sat = (v_max - v0) / a_stern if a_stern > 0.0 else np.inf  # 达 v_max 时刻
    if t_sat >= t_acc:                                        # 加速段内未饱和
        s_len = v0 * t_acc + 0.5 * a_stern * t_acc ** 2
        v_end = v0 + a_stern * t_acc
    else:                                                     # 加速段内饱和：先加速到 v_max 再匀速
        s_len = v0 * t_sat + 0.5 * a_stern * t_sat ** 2 + v_max * (t_acc - t_sat)
        v_end = v_max
    if t > t_react:                                           # 反应期后匀速 [0,0]
        s_len += v_end * (t - t_react)
    u = np.array([np.cos(s_ego.orientation), np.sin(s_ego.orientation)])
    return np.asarray(s_ego.position, dtype=float) + u * s_len


def is_emergency_under_acc(s_ego: VesselState, s_obs: VesselState,
                           a_stern: float | None = None, t_react: float = T_REACT,
                           t_pred: float = T_PRED, dt_check: float = DT,
                           dt_reach: float = DT_REACH, obs_width: float | None = None,
                           ego_width: float = EGO_WIDTH, vessel_params=None) -> bool:
    """式(7) 第三条的 is_emergency(s_ego, s_obs, ..., u_acc(t)) 变体：本船沿 u_acc 的**轨迹占据**
    （非 u_keep 恒速）∩ 他船点质量可达占据。骨架同 is_emergency（a.2-ii），仅本船中心轨迹替换。

    u_acc = [a_stern, 0] ∀t≤t_react，然后 [0,0]（论文式7 下方原文）。直线运动 → 区间占据用首末
    矩形凸包仍保守（位置一维单调、朝向不变）。
    ⚠️ 硬假设（B 复核 MINOR-4）：v_ego ≤ v_max（本函数对 v0 截到 [0,v_max]，is_emergency 不截）——
    由 env 层 clip_velocity=True 保证；若 env 关 clip 且 v 超 9.5，两函数在 a_stern=0 时不再等价。
    本函数无扇区前提也健全（False ⟺ u_acc 轨迹避开全部点质量可达占据，与他船方位无关）；
    stern 扇区条件在 stern_emergency 里、属论文模式选择而非本函数健全性条件。
    ⚠️ **认证 vs 执行精度（D 复核 MINOR-1，2026-06-10，触及 "provably"/D5）**：本函数（= stern_emergency
       "加速确定解除" 的认证）用 _pos_under_stern_acc 的**步内 v 饱和解析模型**；而闭环**执行**走
       env dyn_step(clip_velocity=True)（步末才截 v=D9 不一致）。当 v0>v_max−a_stern·t_react≈6.62 仍
       加速时两者分歧：t_pred 末执行比认证**多走** ~12.8m（v0=9.3 实测，主窗口坐实）。方向安全
       （执行多走=背离后方他船，零碰撞），仅"provably resolves"精度受限（同组件2 dt_sim 一簇）。
       **Phase 4/D5 二选一**：① is_emergency_under_acc 也改约束细步积分对齐执行（消 ~13m）；
       ② 写作把 stern "provably resolves" 限定为"对解析 u_acc 模型"。与 v_max clip 裁决（02 挂起）一并定。
    """
    if t_pred < 0.0:
        raise ValueError(f"t_pred 必须 ≥ 0，得到 {t_pred}")
    if dt_check <= 0.0:
        raise ValueError(f"dt_check 必须 > 0，得到 {dt_check}")
    if vessel_params is None:
        vessel_params = _make_vessel_params()
    if a_stern is None:
        a_stern = A_STERN_FACTOR * float(vessel_params.a_max)
    v_max = float(vessel_params.v_max)
    r_obs = _vessel_circumradius(s_obs.length, obs_width)
    v_m = velocity_vector(s_obs)
    n = int(np.floor(t_pred / dt_check + 1e-9))
    for k in range(n + 1):
        t0 = k * dt_check
        t1 = min(t0 + dt_check, t_pred)
        c0 = s_obs.position + v_m * t0
        c1 = s_obs.position + v_m * t1
        radius = reach_radius_pm(t1, dt_reach) + r_obs
        if np.allclose(c0, c1):
            obs_occ = Point(float(c0[0]), float(c0[1])).buffer(radius)
        else:
            obs_occ = LineString([(float(c0[0]), float(c0[1])),
                                  (float(c1[0]), float(c1[1]))]).buffer(radius)
        p0 = _pos_under_stern_acc(s_ego, t0, a_stern, t_react, v_max)
        p1 = _pos_under_stern_acc(s_ego, t1, a_stern, t_react, v_max)
        e0 = _ego_rect(p0, s_ego.orientation, s_ego.length, ego_width)
        e1 = _ego_rect(p1, s_ego.orientation, s_ego.length, ego_width)
        if e0.union(e1).convex_hull.intersects(obs_occ):
            return True
    return False


def stern_emergency(s_ego: VesselState, s_obs: VesselState,
                    delta_stern: float = DELTA_STERN, **iea_kwargs) -> bool:
    """2024 式(7)（除 in(ρ5) 外两条）：他船在后方扇区（WD1）∧ 沿 u_acc 加速**确定能解除**紧急。

    **iea_kwargs 透传 is_emergency_under_acc（a_stern/t_react/t_pred/dt_check/...）。
    """
    return bool(_in_stern_sector(s_ego, s_obs, delta_stern)
                and not is_emergency_under_acc(s_ego, s_obs, **iea_kwargs))


def get_target_ahead(s_ego0: VesselState, s_obs0: VesselState,
                     d_ahead: float | None = None) -> np.ndarray:
    """ahead 模式目标点（Alg.1 line 9，基于**初始快照**=固定点）：从初始位置向转向侧 90° 方向
    d_ahead 处（"we require the ego vessel to turn 90°"，方向按 Fig.5/WD2）。

    d_ahead 默认 d_min,ahead = 3·l_obs（WD3，论文无公式）。右转(+1) → 目标方位 θ_ego,0 − 90°。
    签名注记（A 复核 MINOR-1）：论文 line 9 为 get_target_ahead(s_ego, s_ego,0, s_obs,0)（多一个
    当前态）；目标按"转 90°"语义只依赖初始快照 → 本实现收窄签名为 (s_ego0, s_obs0)，闭式化已披露。
    """
    if d_ahead is None:
        d_ahead = DMIN_AHEAD_FACTOR * s_obs0.length
    if d_ahead <= 0.0:
        raise ValueError(f"d_ahead 必须 > 0，得到 {d_ahead}")
    direction = turning_direction(s_ego0, s_obs0)
    ang = float(s_ego0.orientation) - direction * 0.5 * np.pi
    return np.asarray(s_ego0.position, dtype=float) + d_ahead * np.array([np.cos(ang), np.sin(ang)])


def get_target_base(s_ego: VesselState, s_obs: VesselState,
                    d_behind: float | None = None) -> np.ndarray:
    """base 模式目标点（Alg.1 line 12，基于**当前** s_obs=动目标）：他船艉后、他船轴线上
    （Fig.4(a) 绿叉；"steers the ego vessel behind the stern of the obstacle vessel"）。

    中心距艉 = 0.5·l_obs + d_behind，d_behind 默认 d_obs,safety = 2·l_obs（WD4，论文无公式）；
    默认路径下中心距并取 **ego 尺度下限 2.5·l_ego**（B 复核 MAJOR-2：纯随 l_obs 缩放在小他船时
    目标点 < d_resolved=2·l_ego → 结构性永不 resolved + 贴身绕圈；论文域 l_obs≥l_ego 下限不触发
    = 零行为变化；显式传 d_behind 时不加下限、调用方自担）。s_ego 用于下限的 l_ego（论文签名
    get_target_base(s_ego, s_obs) 本就含 s_ego）。
    """
    u = np.array([np.cos(s_obs.orientation), np.sin(s_obs.orientation)])
    if d_behind is None:
        dist_center = max(0.5 * s_obs.length + DOBS_SAFETY_FACTOR * s_obs.length,
                          2.5 * s_ego.length)
        return np.asarray(s_obs.position, dtype=float) - u * dist_center
    if d_behind <= 0.0:
        raise ValueError(f"d_behind 必须 > 0，得到 {d_behind}")
    return np.asarray(s_obs.position, dtype=float) - u * (0.5 * s_obs.length + d_behind)


def tracking_controller(s_ego: VesselState, p_target, v_desired: float = V_DESIRED,
                        lam1: float = LAMBDA_1, lam2: float = LAMBDA_2,
                        dv_omega: float = DELTA_V_OMEGA, vessel_params=None) -> tuple:
    """Appendix-C 李雅普诺夫位置跟踪：(s_ego, p_target) → (a, ω)，恒 clip ±a_max/±ω_max。

    Vω = 1−([cosθ,sinθ]·w_des)² = sin²φ；Va = ½‖p_des−p‖²；w_des = (p_des−p)/‖·‖。
    ω = −λ1·Vω / [−2·([−sinθ,cosθ]·w_des)·([cosθ,sinθ]·w_des)]（分母 = −sin 2φ）；
    a = −λ2·Va / [−(p_des−p)·v·[cosθ,sinθ]ᵀ]；Vω>ΔVω → a=0（论文）。
    守卫（WD5/WD6，诚实标注）：p_des=p_target；v>v_desired → a 钳 ≤0；|φ|≥π/2 → 满舵朝目标侧
    （论文公式该域收敛到背对，见段注）；v≈0 → a=a_max 起步；已到目标 → (0,0)。
    """
    if vessel_params is None:
        vessel_params = _make_vessel_params()
    a_max, w_max = float(vessel_params.a_max), float(vessel_params.w_max)
    p = np.asarray(s_ego.position, dtype=float)
    d = np.asarray(p_target, dtype=float) - p
    if not np.all(np.isfinite(d)):
        raise ValueError(f"p_target/position 含非有限值：target={p_target!r}, pos={s_ego.position!r}")
    dist = float(np.linalg.norm(d))
    if dist < _TARGET_REACHED_EPS:
        return (0.0, 0.0)                                     # 已到目标（w_des 0/0 守卫）
    w_des = d / dist
    th = float(s_ego.orientation)
    cphi = float(np.cos(th) * w_des[0] + np.sin(th) * w_des[1])    # [cosθ,sinθ]·w_des = cos φ
    sphi = float(-np.sin(th) * w_des[0] + np.cos(th) * w_des[1])   # [−sinθ,cosθ]·w_des = sin φ
    v_omega = 1.0 - cphi ** 2                                      # = sin²φ

    # --- ω 通道 ---
    if cphi <= 0.0:                                           # |φ|≥π/2：论文公式错误收敛域（WD6）→ 满舵朝目标侧
        if abs(sphi) < _SING_EPS:
            omega = -w_max                                    # φ≈π 正背对 → 确定性右转掉头
        else:
            omega = float(np.copysign(w_max, sphi))           # 目标在左(sphi>0)→左满舵 ω>0
    else:
        denom_w = -2.0 * sphi * cphi                          # = −sin 2φ
        if abs(denom_w) < _SING_EPS:
            # 分母≈0 且 cphi>0 的两种几何（用 |sphi| 区分，0.5 安全分离）：
            #   φ≈0（sphi≈0）→ 已对准 ω=0；φ≈±π/2 但 cphi 带 +ε 浮点尾巴（|sphi|≈1）→ 满舵朝目标侧。
            #   （后者实测：get_target_ahead 正横目标经 cos(−π/2)≈6e-17 使 cphi=+ε 落本分支，若一律
            #    给 0 则正横不转向 = 真 bug，冒烟 N 段锁定。）
            omega = 0.0 if abs(sphi) < 0.5 else float(np.copysign(w_max, sphi))
        else:
            omega = float(np.clip(-lam1 * v_omega / denom_w, -w_max, w_max))

    # --- a 通道 ---
    if v_omega > dv_omega:
        a = 0.0                                               # Vω gate：33.2°<|φ|<146.8° 只转向（论文）
    else:
        v = float(s_ego.velocity)
        if v < _SING_EPS:
            # 静止起步守卫（a 分母 0）：仅大致朝向目标（cphi>0）才给油；背对（|φ|≥146.8° 过 gate
            # 后 Vω≤ΔVω 复又激活本通道）时 a=0 原地转向，防满加速远离目标（B 复核 MINOR-3）。
            a = a_max if cphi > 0.0 else 0.0
        else:
            denom_a = -float(dist * v * cphi)                 # −(p_des−p)·v_vec
            if abs(denom_a) < _SING_EPS:
                a = 0.0
            else:
                va = 0.5 * dist ** 2
                a = float(np.clip(-lam2 * va / denom_a, -a_max, a_max))
        if float(s_ego.velocity) > v_desired:
            a = min(a, 0.0)                                   # 超 v_desired 不再加速（WD5）
    return (a, float(np.clip(omega, -w_max, w_max)))


class EmergencyController:
    """2024 Alg.1 emergency_maneuver 的有状态实现：每决策步 step(s_ego, s_obs) → (a, ω) = a_em。

    生命周期 = 一次 ρ5 紧急事件：状态机进 ρ5 后每步调 step；首步评估 mode（ahead/stern 谓词互斥
    ——前扇区 ±45° 与后方扇区不相交；均假 → base）并记初始快照（Alg.1 line 1）；ahead 行驶距
    > d_min,ahead 单向切 base（line 3-4）；ρ5 退出（is_emergency_resolved，状态机管）后调 reset()。
    ⚠️ 跨事件复用**必须** reset（B 复核 MINOR-5）：忘 reset 则 mode/计时沿用上一事件 = 静默错误
    动作（无运行时守卫，契约同"调用方保证 in(ρ5)"）。
    a_em 即 env 动作空间第 50 动作（状态相关非网格点；env 接线在组件4/调度器）。
    """

    def __init__(self, vessel_params=None, dt: float = DT):
        self._vp = vessel_params if vessel_params is not None else _make_vessel_params()
        if not (dt > 0.0):                                    # not(>) 拒 NaN（B 复核 MINOR-2）
            raise ValueError(f"dt 必须 > 0 且有限，得到 {dt}")
        self._dt = float(dt)
        self.reset()

    def reset(self) -> None:
        """清空事件状态（mode/初始快照/计时）；每次新 ρ5 事件前调用。"""
        self._mode: str | None = None
        self._p0: np.ndarray | None = None
        self._target_ahead: np.ndarray | None = None
        self._d_switch: float = 0.0
        self._t: float = 0.0

    @property
    def mode(self) -> str | None:
        """当前模式 ∈ {None(未激活), 'ahead', 'stern', 'base'}。"""
        return self._mode

    def step(self, s_ego: VesselState, s_obs: VesselState) -> tuple:
        """输出本决策步 a_em = (a, ω)（Alg.1 line 6-15）。调用方保证当前确在 ρ5（in(ρ5)）。"""
        if self._mode is None:                                # 进入 ρ5 首步：定 mode + 快照（line 1）
            if ahead_emergency(s_ego, s_obs):
                self._mode = "ahead"
                self._target_ahead = get_target_ahead(s_ego, s_obs)   # 固定目标（初始快照，line 9）
            elif stern_emergency(s_ego, s_obs, vessel_params=self._vp):
                self._mode = "stern"
            else:
                self._mode = "base"
            self._p0 = np.asarray(s_ego.position, dtype=float).copy()
            self._d_switch = DMIN_AHEAD_FACTOR * s_obs.length         # d_min,ahead = 3·l_obs
            self._t = 0.0
        if (self._mode == "ahead"
                and float(np.linalg.norm(np.asarray(s_ego.position, dtype=float) - self._p0))
                > self._d_switch):
            self._mode = "base"                               # ahead 行驶距超限未解 → 切 base（line 3-4）
        if self._mode == "stern":
            a_stern = A_STERN_FACTOR * float(self._vp.a_max)
            # u_acc 的 ZOH 落地用 **<**（A-MAJOR-3/B-MAJOR-1 一致发现）：`<=` 会让第 7 步（_t=60）
            # 仍加速、实际执行 70s ≠ 认证模型 _pos_under_stern_acc 的 60s（t_pred 末偏差 55m，蚕食
            # provably 裕度）。`<` 下执行=[0,60) 加速 = 模型逐点相等（t=60 单点 ZOH 无测度）。
            u = (a_stern, 0.0) if self._t < T_REACT else (0.0, 0.0)
        elif self._mode == "ahead":
            u = tracking_controller(s_ego, self._target_ahead, vessel_params=self._vp)
        else:                                                 # base：动目标（当前 s_obs，line 12）
            u = tracking_controller(s_ego, get_target_base(s_ego, s_obs), vessel_params=self._vp)
        self._t += self._dt
        return u


# ============================================================================
# (b-5) As(ρ) 调度器（2024 §V-C + Theorem 2，p10）：状态机 ρ → 安全动作集 As
# ============================================================================
# fact-based（笔记⑧ As(ρ) 三档）：ρ0 → A_regular（全 49 常规动作，无规则适用）/ ρ1-4 → Alg.3
# 输出（encounter_action_verification，(b-3)）/ ρ5 → {a_em}（EmergencyController 输出，(b-4)）。
# 论文用离散 As + action masking（Theorem 2）；本项目卖点 = 这层换连续投影（Phase 2-3），
# 本件先落论文忠实离散版（服务 step4 基线复现 + Phase 2 投影真值）。
# ⚠️ 单 (ego,obstacle) pair（同状态机 (a.3) 决定④；多他船 deferred Phase 3/5）。

A_REGULAR = tuple((a, w) for a in A_A for w in A_OMEGA_GRID)   # 49 = 7×7（= env DISCRETE_ACTIONS，冒烟断言防漂移）


class SafeActionScheduler:
    """As(ρ) 调度：step(s_ego, s_obs) → (ρ, As)。持 ColregsStatechart + EmergencyController。

    As 语义（Theorem 2）：ρ0 → A_regular 全集；ρ1 → {a_keep}；ρ2-4 → Alg.3 verified 候选集
    （可能 ∅ = give-way 无解，上层走紧急/投影兜底）；ρ5 → {a_em}（状态相关、非网格点）。
    **ρ5 进入边沿自动 reset EmergencyController**（一次紧急事件一个 Alg.1 生命周期；ρ5 连续
    驻留不 reset、退出后再进 = 新事件再 reset）。返回 As = set[(a,ω)]。
    **alg3_kwargs 透传 encounter_action_verification（t_m/t_max_m/dt_sim/obs_width/...）。
    """

    def __init__(self, vessel_params=None, dt: float = DT, **alg3_kwargs):
        self._vp = vessel_params if vessel_params is not None else _make_vessel_params()
        self._statechart = ColregsStatechart()
        self._ec = EmergencyController(vessel_params=self._vp, dt=dt)
        self._alg3_kwargs = dict(alg3_kwargs)
        self._prev_rho = RHO_NO_CONFLICT

    def reset(self) -> None:
        """episode 边界：清状态机 + 紧急控制器。"""
        self._statechart.reset()
        self._ec.reset()
        self._prev_rho = RHO_NO_CONFLICT

    @property
    def rho(self) -> int:
        return self._statechart.rho

    @property
    def emergency_mode(self) -> str | None:
        """当前紧急事件的 Alg.1 模式（None=非 ρ5 / 'ahead'/'stern'/'base'），观测/日志用。"""
        return self._ec.mode

    def step(self, s_ego: VesselState, s_obs: VesselState):
        """推状态机一步并给出 (ρ, As(ρ))。"""
        rho = self._statechart.step(s_ego, s_obs)
        if rho == RHO_EMERGENCY:
            if self._prev_rho != RHO_EMERGENCY:
                self._ec.reset()                              # 进入边沿 = 新紧急事件
            a_s = {tuple(float(x) for x in self._ec.step(s_ego, s_obs))}
        elif rho == RHO_NO_CONFLICT:
            a_s = set(A_REGULAR)                              # 无规则适用 → 全 49
        else:                                                 # ρ1-4 → Alg.3（keep 含在内）
            psi_e = {RHO_STAND_ON: "keep", RHO_HEAD_ON: "head_on",
                     RHO_CROSSING: "crossing", RHO_OVERTAKE: "overtake"}[rho]
            a_s = encounter_action_verification(s_ego, s_obs, psi_e, **self._alg3_kwargs)
        self._prev_rho = rho
        return rho, a_s


# ============================================================================
# (c) 违规率计数（2024 §VII-A p12：rule violations per episode）
# ============================================================================
# fact-based（亲读 2024 §VII-A 逐字 + commonocean-rules R_G3-6 MTL 映射，笔记 ③ 深化）：
#   "we count: • every time step of violating the stand-on vessel position;
#              • every crossing/overtaking/head-on encounter for which no proper maneuver is taken."
# 权威映射 commonocean-rules R_G3-6（Krasowski 2021，GPL v3 只读参照 D10，自重写不 vendor、引用其论文）：
#   R_G6 stand-on (time-step): G(keep → (no_turning U ¬keep))；no_turning = keep onset 起累积航向 < Δno_turn=10°。
#   R_G3-5 give-way (encounter): maneuver_crossing/head_on = 累积 ≥Δlarge_turn=20° ∧ turning_to_starboard（右转）；
#     maneuver_overtake = 累积 ≥20°（不要求 starboard）。相遇 onset→解除全程未达 maneuver → +1。
# ⚠️ 单步航向变化用 wrap_to_pi(θ[t-1]−θ[t])（最短路径，正=右转/starboard）= commonocean signed_modulo 对
#    单步小 Δ（|Δ|≤ω_max·dt=0.3rad ≪ π）数值等价、且 wrap 鲁棒。turning_to_starboard ≡ wrap_to_pi(cur−onset)<0。
# ⚠️ 待定（step4 校准）：2024 未明示是否照搬 2021 精确 MTL 时间窗（F[0,12]/F[6,18]/G[1,6] 步）；本实现用
#    纯态势谓词 onset→解除全程累积近似 "per encounter no proper maneuver"（去精确步窗）。违规计数用**原始态势
#    谓词**（非状态机 ρ）——R_G3-6 直接基于谓词；评估"裸策略违规多少"（基线 Base/RR），Safe agent 有盾→0。
#    单 (ego,obstacle) pair（多船 deferred Phase 3/5）。
# ⚠️ **方向判据 >180° 净朝向边界（新窗口接手复核裁定 2026-06-11，03 L16）**：give-way 的 starboard 判据
#    wrap_to_pi(theta - onset)<0 用**净朝向**（onset 前一步 vs 当前），单次相遇本船单调净转 >180° 时把净左转
#    误读为 starboard → 该相遇欠计违规。**全角域对拍官方 commonocean turning_to_starboard：除 ±180° 整点
#    （测度零浮点边界）外 0 mismatch** → 官方口径固有行为（Table III 由官方 monitor 算，须对齐），**非 bug、不改**；
#    改路径语义(cum>0)反偏离官方。触发域窄（≥11 步单向转 >180° 且相遇全程不解除），Handcrafted 场景几乎不触发。

DELTA_NO_TURN = 10.0 * _DEG        # max_orientation_diff_no_change：stand-on 不转向阈值（yaml）


_VIOL_CUM_TOL = 1e-9    # 累积航向阈值比较容差（D3-4/L38）：cum=Σwrap_to_pi(prev−θ) 是浮点累加，恰达阈值时
#   会比解析 threshold 略少（短累积 ~7 ULP=4e-16 rad；长累积最坏实测 ~数百 ULP≈2e-14 rad@N≈100-200）。盾给路
#   投影恰输出 ω=−ω_turn → 净恰好 20°、stand-on 带边沿恰好 10° = 最易命中边界 → 无容差 `>=` 在此翻转
#   （give-way 虚增违规 / stand-on 漏计违规）。1e-9 rad 容差使「数学上恰达阈值」被正确判达标：① 比最坏浮点
#   累积误差(2e-14)大 ~5 万× = 稳吸收末位；② =5.7e-8°、远小于任何真实欠机动角度 = 绝不吃掉真违规（对抗复核
#   30000+200万 序列实测 0 非边界翻转）。与同文件 _MOVING_AWAY_TOL / persistent 窗 1e-9 风格一致、不改语义。


class ViolationCounter:
    """2024 §VII-A 违规计数（per episode），忠实 commonocean-rules R_G3-6 MTL。

    逐 episode 有状态：每决策步调 step(s_ego, s_obs)；episode 末调 finalize() 结算未解除的 give-way 相遇。
    - stand-on (R_G6, time-step 级)：keep 期间累积航向 ≥ Δno_turn → 每步 +1。
    - give-way (R_G3-5, encounter 级)：crossing/head_on/overtake 相遇 onset→解除全程未达 maneuver → 每相遇 +1。
    standon_violations + giveway_violations = 该 episode 违规总次数（对齐 Table III "Rules violated"）。
    """

    _GW_SITUATIONS = ("crossing", "head_on", "overtake")

    def __init__(self, delta_large_turn: float = DELTA_LARGE_TURN,
                 delta_no_turn: float = DELTA_NO_TURN, t_horizon: float = T_HORIZON):
        self.delta_large_turn = float(delta_large_turn)
        self.delta_no_turn = float(delta_no_turn)
        self.t_horizon = float(t_horizon)
        self.reset()

    def reset(self) -> None:
        """episode 边界：清累积/相遇追踪 + 违规计数归零。"""
        self._prev_theta: float | None = None
        self._keep_active = False
        self._keep_cum = 0.0
        self._gw: dict = {}                  # situ -> {cum, onset_theta, maneuver_done}
        self.standon_violations = 0
        self.giveway_violations = 0

    @property
    def total(self) -> int:
        return self.standon_violations + self.giveway_violations

    def _situation(self, name: str, s_ego: VesselState, s_obs: VesselState) -> bool:
        if name == "crossing":
            return crossing(s_ego, s_obs, self.t_horizon)
        if name == "head_on":
            return head_on(s_ego, s_obs, self.t_horizon)
        return overtake(s_ego, s_obs, self.t_horizon)

    def step(self, s_ego: VesselState, s_obs: VesselState) -> dict:
        """推进一步，返回本步新增违规 + 累计。"""
        theta = float(s_ego.orientation)
        dtheta = 0.0 if self._prev_theta is None else wrap_to_pi(self._prev_theta - theta)   # 正=右转

        # --- stand-on (R_G6, time-step)：keep 期间累积航向 ≥ Δno_turn → 违规 ---
        new_standon = 0
        if keep(s_ego, s_obs, self.t_horizon):
            # onset 重置 cum 与下方 ¬keep 分支 reset 互为**双重保险**（主窗口实测任一单删无行为差=死代码，
            # 两个都删才跨 keep spell 累加误报；留双重防御，P12 守护至少一个，2026-06-10）
            self._keep_cum = dtheta if not self._keep_active else self._keep_cum + dtheta
            self._keep_active = True
            if abs(self._keep_cum) >= self.delta_no_turn - _VIOL_CUM_TOL:   # ¬no_turning → 违规（容差防恰达阈值漏计，D3-4）
                new_standon = 1
        else:
            self._keep_active = False
            self._keep_cum = 0.0
        self.standon_violations += new_standon

        # --- give-way (R_G3-5, encounter)：相遇 onset→解除全程未达 maneuver → 违规 ---
        new_giveway = 0
        for name in self._GW_SITUATIONS:
            x = self._situation(name, s_ego, s_obs)
            st = self._gw.get(name)
            if x:
                if st is None:                                   # onset（单步首次为真；G[1,6] 持续进入简化为单步，见段注待定）
                    st = {"cum": dtheta,
                          "onset_theta": self._prev_theta if self._prev_theta is not None else theta,
                          "maneuver_done": False}
                    self._gw[name] = st
                else:
                    st["cum"] += dtheta
                if not st["maneuver_done"] and abs(st["cum"]) >= self.delta_large_turn - _VIOL_CUM_TOL:   # 容差防恰达 20° 虚增违规（D3-4）
                    if name == "overtake":
                        st["maneuver_done"] = True               # overtake 不要求 starboard（可左可右）
                    elif wrap_to_pi(theta - st["onset_theta"]) < 0.0:   # crossing/head_on 须右转 starboard（净朝向语义=忠实官方；>180°边界见 (c) 段注/L16）
                        st["maneuver_done"] = True
            elif st is not None:                                 # 相遇解除（offset）
                if not st["maneuver_done"]:
                    new_giveway += 1
                self._gw[name] = None
        self.giveway_violations += new_giveway

        self._prev_theta = theta
        return {"standon": new_standon, "giveway": new_giveway,
                "standon_total": self.standon_violations, "giveway_total": self.giveway_violations,
                "total": self.total}

    def finalize(self) -> int:
        """episode 末结算仍 active（未解除）的 give-way 相遇：未达 maneuver → 违规 +1。返回新增数。

        ⚠️ 不可重入（B2 复核 MINOR-3）：将槽置 None；若不 reset 直接续 step 且相遇仍真，会重新 onset
        致重复计数。合约 = finalize 仅 episode 末调、之后必 reset()。
        """
        new = 0
        for name in self._GW_SITUATIONS:
            st = self._gw.get(name)
            if st is not None and not st["maneuver_done"]:
                new += 1
            self._gw[name] = None
        self.giveway_violations += new
        return new
