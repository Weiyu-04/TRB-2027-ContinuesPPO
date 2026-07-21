"""
TRB 环境 · 27 维观测模块
=========================
忠实复现 Krasowski & Althoff 2024 论文 §VI-A 的 27 维观测空间。
（论文 = 参考资料/2402.08502v2.pdf；事实抽取见 参考资料/文献核实笔记.md ⑦ + 本文 docstring 标注）

观测布局（27 维，顺序固定，下游 env / 训练必须按此索引）：
  [0:4]   ego（本船）   : v_ego, θ_ego, a_ego, ω_ego
  [4:9]   goal（目标）   : d_goal, remaining_steps, β_goal, d_long, d_lat
  [9]     bool goal     : 1{ min(|d_lat|,|d_long|) > d_hull }
  [10:22] traffic（他船）: 4 扇区 × {d_j, β_j, ḋ_j}，扇区顺序 = front, left, right, behind
  [22:27] termination   : 1_time, 1_area, 1_stopped, 1_collision, 1_goal
  合计 4 + 5 + 1 + 12 + 5 = 27 ✓（论文原文 "The observation space has 27 dimensions"）

⚠️ 论文未写死、本模块按 fact-based 推荐处理的 4 点（与 v_max 同样诚实标注，留 step4 复现核对）：
  ① 空扇区（无船检测 / 超出感知距离）默认值 = (d=d_sense, β=0, ḋ=0)
     —— "远到无威胁" 语义；论文只说 "detected if distance ≤ d_sense"，未规定无检测时填什么。
  ② 4 扇区在向量里的排序 = front, left, right, behind（与论文正文列举序一致；
     论文 Fig.7 把扇区编号成 1=front/2=left/3=behind/4=right，与正文列举序不同——这里按正文序，
     内部一致即可，扇区身份由 SECTOR_NAMES 显式记录）。
  ③ β_goal（朝向区间差）：θ_ego 落在目标朝向区间内 → 0；否则 = 到最近边界的带符号最短角差。
  ④ ḋ_j（距离变化率）：按 "每艘他船 id" 追踪上一步真实距离，ḋ = d_now − d_prev（首次出现 = 0），
     再归入该船当前所在扇区（单他船场景天然无歧义；多船同扇区时取最近者，见 build()）。

⚠️ 数值约定（同样留训练/复现核对）：
  - 观测输出 **原始物理量**（米 / 弧度 / 步数 / m·s⁻¹），不做归一化；归一化交给 Phase 3 训练包装层。
  - 角度（θ_ego / β_goal / β_j）单位 = 弧度，wrap 到 [-π, π]。
  - 相对方位 β 符号 = **右舷（starboard）为正、左舷（port）为负**（对齐 Table IV 扇区定义：
    left[-112.5°,-5°] 负 / right[5°,112.5°] 正）。即 β = wrap(θ_ego − atan2(Δy, Δx))。
    （已几何验证：本船朝 +x、他船在右舷 −y → β=+90° 落 right；他船在左舷 +y → β=−90° 落 left。）
"""
from __future__ import annotations

import numpy as np

from .usv_dynamics import wrap_to_pi

OBS_DIM: int = 27

# 论文 Table II 观测相关参数（fact-based，PAGE 11）
DEFAULT_D_SENSE: float = 8000.0   # 感知距离 d_sense（m）：他船距离 > 此值不检测
DEFAULT_D_HULL: float = 2000.0    # d_hull（m）：bool goal 偏离判定阈值
N_SECTORS: int = 4                # J = 4

# 4 扇区相对方位边界（Table IV，PAGE 18；右舷为正、左舷为负）。顺序见待定点②。
# ⚠️ 采用论文正文列举序 front/left/right/behind；非 Fig.7 编号序（1/2/3/4 = front/left/behind/right）。
SECTOR_NAMES: tuple[str, ...] = ("front", "left", "right", "behind")
_FRONT_HALF_DEG: float = 5.0      # front = |β| ≤ 5°（也 = ∆head-on 带）
_SIDE_MAX_DEG: float = 112.5      # left/right 外边界；behind = |β| > 112.5°


def _classify_sector(beta_rad: float) -> int:
    """按相对方位 β（弧度，右舷正/左舷负）分扇区，返回 SECTOR_NAMES 的下标。

    front  : |β| ≤ 5°
    right  : 5° < β ≤ 112.5°          （starboard，正）
    left   : -112.5° ≤ β < -5°        （port，负）
    behind : 其余（|β| > 112.5°）
    边界互斥无缝覆盖 [-180°,180°]。
    """
    b = np.degrees(beta_rad)
    if abs(b) <= _FRONT_HALF_DEG:
        return SECTOR_NAMES.index("front")
    if _FRONT_HALF_DEG < b <= _SIDE_MAX_DEG:
        return SECTOR_NAMES.index("right")
    if -_SIDE_MAX_DEG <= b < -_FRONT_HALF_DEG:
        return SECTOR_NAMES.index("left")
    return SECTOR_NAMES.index("behind")


def _angle_to_interval(theta: float, lo: float, hi: float) -> float:
    """θ 到角度区间 [lo, hi]（沿 CCW 从 lo 到 hi 的弧）的带符号最短角差。

    - θ 落在区间内 → 0。
    - 否则 → 到最近边界的带符号最短角差（wrap 到 [-π, π]），取 |·| 较小的一边。
    约定区间宽度 < 2π（目标朝向区间很小，如 T-0 = [-0.17, 0.17]）。
    """
    width = (hi - lo) % (2.0 * np.pi)          # CCW 弧宽
    pos = (theta - lo) % (2.0 * np.pi)         # θ 相对 lo 的 CCW 位置
    if pos <= width:                            # 区间内
        return 0.0
    d_hi = wrap_to_pi(theta - hi)
    d_lo = wrap_to_pi(theta - lo)
    return d_hi if abs(d_hi) < abs(d_lo) else d_lo


class ObservationBuilder:
    """构造 27 维观测。静态持有 目标 / 初始位置 / 参数；内部追踪每船上一步距离（算 ḋ）。

    用法：
        ob = ObservationBuilder(goal_center, goal_orientation, init_position, k_max)
        ob.reset()                              # 每个 episode 开始
        o = ob.build(ego_state, last_action, obstacles, step, term_flags)
    """

    def __init__(
        self,
        goal_center,
        goal_orientation: tuple[float, float],
        init_position,
        k_max: int,
        *,
        d_sense: float = DEFAULT_D_SENSE,
        d_hull: float = DEFAULT_D_HULL,
    ):
        """
        goal_center      : [x, y] 目标区中心（CommonOcean goal.state_list[0].position.center）
        goal_orientation : (θ_lo, θ_hi) 目标朝向区间（goal.state_list[0].orientation 的 start/end，弧度）
        init_position    : [x, y] 本船初始位置（pp.initial_state.position）—— 定义 init→goal 参考线
        k_max            : int 最大时间步（goal.state_list[0].time_step.end，T-0 实测 = 170）
        d_sense / d_hull : 感知距离 / 偏离阈值（默认 = Table II）
        """
        self.goal_center = np.asarray(goal_center, dtype=float)
        self.goal_lo, self.goal_hi = float(goal_orientation[0]), float(goal_orientation[1])
        self.init_position = np.asarray(init_position, dtype=float)
        self.k_max = int(k_max)
        self.d_sense = float(d_sense)
        self.d_hull = float(d_hull)

        if self.goal_center.shape != (2,) or self.init_position.shape != (2,):
            raise ValueError("goal_center / init_position 必须是 2 维 [x, y]")

        # init→goal 参考线的纵向单位向量 e_long 与横向单位向量 e_lat（e_long 逆时针转 90°）
        line = self.goal_center - self.init_position
        norm = float(np.linalg.norm(line))
        if norm < 1e-9:
            raise ValueError("初始位置与目标中心重合，无法定义 init→goal 参考线")
        self.e_long = line / norm                          # 纵向（朝目标）
        self.e_lat = np.array([-self.e_long[1], self.e_long[0]])  # 横向（参考线左侧为正）

        self._prev_dist: dict = {}                         # obstacle_id -> 上一步真实距离（算 ḋ）

    def reset(self) -> None:
        """清空每船距离历史（每个 episode 开始时调用）。"""
        self._prev_dist = {}

    def build(self, ego_state, last_action, obstacles, step: int, term_flags=None) -> np.ndarray:
        """构造一帧 27 维观测。

        ego_state   : [px, py, θ, v]（来自 dynamics.step）
        last_action : [a, ω]（上一步施加的动作；reset 时由 env 传 [0, 0]）
        obstacles   : list[(obstacle_id, [x, y])]  当前时刻各他船 id 与位置
        step        : int 当前时间步（用于 remaining_steps）
        term_flags  : dict 或 None，键 time/area/stopped/collision/goal（bool；缺省全 False）

        返回：np.ndarray shape (27,) float64，原始物理量。
        """
        ego = np.asarray(ego_state, dtype=float)
        act = np.asarray(last_action, dtype=float)
        if ego.shape != (4,):
            raise ValueError(f"ego_state 应为 4 维 [px,py,θ,v]，得到 {ego.shape}")
        if act.shape != (2,):
            raise ValueError(f"last_action 应为 2 维 [a,ω]，得到 {act.shape}")
        # fail-fast：非有限值（NaN/inf）若静默灌进观测会无声毒化训练梯度（独立复核 MAJOR）
        if not np.all(np.isfinite(ego)):
            raise ValueError(f"ego_state 含非有限值（NaN/inf）：{ego}")
        if not np.all(np.isfinite(act)):
            raise ValueError(f"last_action 含非有限值（NaN/inf）：{act}")

        p_ego = ego[:2]
        theta_ego = wrap_to_pi(ego[2])
        v_ego = ego[3]
        a_ego, w_ego = act[0], act[1]

        obs = np.zeros(OBS_DIM, dtype=float)

        # ---- [0:4] ego ----
        obs[0] = v_ego
        obs[1] = theta_ego
        obs[2] = a_ego
        obs[3] = w_ego

        # ---- [4:9] goal ----
        obs[4] = float(np.linalg.norm(p_ego - self.goal_center))          # d_goal
        obs[5] = float(max(0, self.k_max - int(step)))                    # remaining_steps
        obs[6] = _angle_to_interval(theta_ego, self.goal_lo, self.goal_hi)  # β_goal
        rel = p_ego - self.init_position
        d_long = float(np.dot(rel, self.e_long))
        d_lat = float(np.dot(rel, self.e_lat))
        obs[7] = d_long
        obs[8] = d_lat

        # ---- [9] bool goal（论文原文：min(|d_lat|, |d_long|) > d_hull）----
        obs[9] = 1.0 if min(abs(d_lat), abs(d_long)) > self.d_hull else 0.0

        # ---- [10:22] traffic（4 扇区 × {d, β, ḋ}）----
        # 每扇区默认 (d=d_sense, β=0, ḋ=0)（待定点①）；写入时取最近他船（待定点④）。
        sector_d = [self.d_sense] * N_SECTORS
        sector_b = [0.0] * N_SECTORS
        sector_dd = [0.0] * N_SECTORS
        sector_best = [np.inf] * N_SECTORS    # 当前各扇区已记录的最近距离（多船去歧义）

        for obs_id, pos in obstacles:
            p_obs = np.asarray(pos, dtype=float)
            # 校验放在更新 _prev_dist 之前：否则一个 NaN 他船会污染历史，下一帧 ḋ=NaN（独立复核 MAJOR）
            if p_obs.shape != (2,):
                raise ValueError(f"他船 {obs_id} 的 position 应为 2 维 [x,y]，得到 {p_obs.shape}")
            if not np.all(np.isfinite(p_obs)):
                raise ValueError(f"他船 {obs_id} 的 position 含非有限值（NaN/inf）：{p_obs}")
            delta = p_obs - p_ego
            d = float(np.linalg.norm(delta))
            # ḋ 用每船真实上一步距离（即便上一步超感知距离，仿真有真值），首次出现 = 0
            d_prev = self._prev_dist.get(obs_id, d)
            dd = d - d_prev
            self._prev_dist[obs_id] = d
            # 仅当在感知距离内才写入观测（论文："detected if distance ≤ d_sense"）
            if d > self.d_sense:
                continue
            beta = wrap_to_pi(theta_ego - np.arctan2(delta[1], delta[0]))  # 右舷正/左舷负
            s = _classify_sector(beta)
            if d < sector_best[s]:            # 同扇区多船：保留最近
                sector_best[s] = d
                sector_d[s] = d
                sector_b[s] = beta
                sector_dd[s] = dd

        for s in range(N_SECTORS):
            obs[10 + 3 * s + 0] = sector_d[s]
            obs[10 + 3 * s + 1] = sector_b[s]
            obs[10 + 3 * s + 2] = sector_dd[s]

        # ---- [22:27] termination ----
        tf = term_flags or {}
        # 拼错键（如 "collison"）会静默吞掉终止信号，碰撞尤危险 → 未知键直接报错（独立复核 MINOR）
        _unknown = set(tf.keys()) - {"time", "area", "stopped", "collision", "goal"}
        if _unknown:
            raise ValueError(f"term_flags 含未知键 {_unknown}（合法键：time/area/stopped/collision/goal）")
        obs[22] = 1.0 if tf.get("time", False) else 0.0
        obs[23] = 1.0 if tf.get("area", False) else 0.0
        obs[24] = 1.0 if tf.get("stopped", False) else 0.0
        obs[25] = 1.0 if tf.get("collision", False) else 0.0
        obs[26] = 1.0 if tf.get("goal", False) else 0.0

        return obs
