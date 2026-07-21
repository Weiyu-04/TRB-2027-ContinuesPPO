"""
TRB 环境 · 终止模块
===================
忠实复现 Krasowski & Althoff 2024 §VI-A 的 5 个终止条件（bool）。一局在任一条件成立时结束。
（论文 = 参考资料/2402.08502v2.pdf 第10-11页；§VI-A 原句"terminate the episode if one of these observations is present"）

5 个终止条件（同时也是观测向量 [22:27] 的 5 个 bool + 喂奖励件 term_flags）：
  1_time      : 到最大步 k_max（=170，goal.time_step.end）
  1_area      : 本船出航行区
  1_stopped   : 本船速度归零（论文"velocity is zero"）
  1_collision : 本船与他船占据相交
  1_goal      : 本船到达目标区

⭐ 实跑探明的事实（fact-gathering by running，非猜）：
  - **1_goal 复用 CommonOcean 官方 `GoalRegion.is_reached(state)`**（L3 省法）：实测它查
    位置∈目标多边形 **且** 朝向∈区间 **且** time_step∈区间（T-0 goal 三项都查）。需 State 带
    position/orientation/time_step（缺则 ValueError），velocity 多给无妨。
  - **1_collision 无官方检测器**（commonocean/commonroad 无 collision 模块，commonroad_dc 未装）
    → 用 shapely 形状相交（Rectangle.shapely_object.intersects，实测重叠 True/远离 False）。
    本船占据 = Rectangle(l=175, w=25.4)（vessel_1 参数实测）按 (pos, θ) 摆放。
  - 他船时刻占据由 env 用 `dynamic_obstacle.occupancy_at_time(step).shape.shapely_object` 取，传进来。

⚠️ 论文未明确 / 公开数据没有，按推荐处理（同 v_max 挂起逻辑）：
  - **1_area 边界**：场景 `shallows`/`waterways` 空、`Location` 无几何边界（实测）→ 航行区边界**不在
    公开数据里**（藏在 Krasowski 未公开 env）。本模块做成**可配置包围盒 `nav_area_box`**：给了才查、
    默认 None = 1_area 永不触发（不假装有边界）。step4 复现若拿到她的边界再填（可能是场景大包围盒）。
  - **1_stopped 阈值**：论文"velocity is zero"，确切 ε 未给 → 默认 `stopped_eps=1e-3` 可配置。

⚠️ 多条件同时成立（如 step=k_max 同时到达目标）：本模块**返回全部 5 个 bool**（不替 env 选优先级），
  done = 任一为真。终止奖励优先级（goal vs time 谁算数）是 env 接线/奖励层的事，那里再定。
"""
from __future__ import annotations

import copy

import numpy as np
from commonroad.scenario.state import CustomState
from shapely.affinity import rotate as _rotate, translate as _translate
from shapely.geometry import box as _box

from .usv_dynamics import wrap_to_pi

# 本船尺寸（vessel_1 实测：l=175 长 / w=25.4 宽），碰撞矩形用
EGO_LENGTH: float = 175.0
EGO_WIDTH: float = 25.4
DEFAULT_STOPPED_EPS: float = 1e-3


def _ego_footprint(px: float, py: float, theta: float, length: float, width: float):
    """本船占据（shapely 多边形）：原点处轴对齐矩形 → 旋转 θ → 平移到 (px, py)。"""
    b = _box(-length / 2.0, -width / 2.0, length / 2.0, width / 2.0)
    b = _rotate(b, theta, origin=(0.0, 0.0), use_radians=True)
    return _translate(b, px, py)


class TerminationChecker:
    """判定 5 个终止条件。静态持有 goal / k_max / 本船尺寸 / 参数。

    用法：
        tc = TerminationChecker(goal, k_max)
        done, flags = tc.check(ego_state, step, obstacle_footprints)
    """

    def __init__(
        self,
        goal,
        k_max: int,
        *,
        ego_length: float = EGO_LENGTH,
        ego_width: float = EGO_WIDTH,
        stopped_eps: float = DEFAULT_STOPPED_EPS,
        nav_area_box=None,
        goal_ignore_orientation: bool = False,
    ):
        """
        goal         : CommonOcean `GoalRegion`（用其官方 is_reached 判 1_goal）
        k_max        : int 最大时间步（=goal.time_step.end，T-0 实测 170）
        ego_length/width : 本船碰撞矩形尺寸（默认 vessel_1 实测 175 / 25.4）
        stopped_eps  : 1_stopped 阈值，|v| ≤ eps 算停（默认 1e-3）
        nav_area_box : 1_area 航行区包围盒 (xmin, ymin, xmax, ymax)；None = 不查（公开数据无边界，见 docstring）
        """
        self.goal = goal
        self.k_max = int(k_max)
        self.ego_length = float(ego_length)
        self.ego_width = float(ego_width)
        self.stopped_eps = float(stopped_eps)
        if nav_area_box is not None:
            box = tuple(float(v) for v in nav_area_box)
            # 含 NaN 会让 px>nan 恒 False → 1_area 静默失效；显式拒（独立复核 MINOR）
            if len(box) != 4 or not all(np.isfinite(box)) or box[0] >= box[2] or box[1] >= box[3]:
                raise ValueError(f"nav_area_box 应为 (xmin,ymin,xmax,ymax) 有限值且 min<max，得 {nav_area_box}")
            self.nav_area_box = box
        else:
            self.nav_area_box = None
        # 🆕 B1（`03` L153）：到达门朝向容差课程 slack。训练期放宽朝向门让崩种子扎中 +50、学会终端行为，退火回真门；
        #   评估 env【绝不】调 set_arrival_slack → 恒 0 = 真门（诚实红线：报的到达率永远在真 ±9.74° 门上）。默认 0 = bit-identical。
        self.arrival_heading_slack = 0.0
        self._goal_widened = None
        # 🆕 L185（user 2026-07-13）：训练目标【去朝向硬门】——1_goal 只判"位置到达目标区域"（忠实原文字面 "reached the goal area"·治崩种子被朝向门逼出的高速绕圈）。
        #   True → 深拷 goal 删各 state 的 orientation 属性（只剩 position+time_step）→ is_reached 只判位置（实测 L185：门内任意朝向→True·门外/超时→False·原 goal 深拷隔离不受影响）。
        #   False（默认/严格）→ 官方真门(位置+朝向±9.74°) = bit-identical。评估层仍同时记 in_box_aligned_steps=位置+朝向严版 → 两指标都可报。
        self.goal_ignore_orientation = bool(goal_ignore_orientation)
        self._goal_posonly = None
        if self.goal_ignore_orientation:
            gp = copy.deepcopy(self.goal)
            for gs in gp.state_list:
                try:
                    delattr(gs, "orientation")        # 只留 position+time_step → is_reached 只判位置
                except (AttributeError, KeyError):
                    pass                              # 该 goal 本无朝向约束→已是位置-only
            self._goal_posonly = gp

    def set_arrival_slack(self, slack: float):
        """🆕 B1：设训练期到达门朝向松弛量 slack（弧度·把 goal 朝向区间两侧各加宽 slack）。
        slack=0 → 用官方 goal 真门 = bit-identical；slack>0 → 用 deepcopy 的加宽副本（不碰真 goal）。
        ⚠️ 评估 env 绝不调此 → 恒 0 = 真门（`03` L153 诚实红线·committed 测试守 eval 恒 0）。"""
        slack = float(slack)
        if not (np.isfinite(slack) and slack >= 0.0):
            raise ValueError(f"arrival_heading_slack 必须有限非负，得 {slack}")
        self.arrival_heading_slack = slack
        if slack <= 0.0:
            self._goal_widened = None                         # 真门（bit-identical）
        else:
            # 🔒 宽度守卫（`03` L153 独立复审·D3 抓的防御缺口）：commonocean `AngleInterval` 在【区间全宽 ≥ π】时
            #   `is_reached` 内部 `assert interval_diff >= 0` 崩溃（实测真门±0.17：slack≥80.3°→全宽≥π 即 AssertionError）。
            #   加宽后全宽 = 真门全宽 + 2·slack。退火意图上限 45°(0.785rad·全宽~99°) 远低于此，但【防 STEP4E_ARR_SLACK_START
            #   误配过大】→ 在 setter fail-fast（清晰报错），而非训练中途抛 cryptic AngleInterval 断言。评估恒 slack=0 不受影响。
            #   ⚠️ 仅防"全宽≥π 崩溃"；真门朝向近 ±π 的【环绕语义】另需核（本 HOCR 集全朝东±0.17=无环绕·大集上 B1 前须核 goal 朝向）。
            o0 = self.goal.state_list[0].orientation
            orig_width = float(o0.end) - float(o0.start)
            _MAX_WIDTH = np.pi - 0.05                          # < π（崩溃硬界）留 ~2.9° margin
            if orig_width + 2.0 * slack >= _MAX_WIDTH:
                max_slack = max(0.0, (_MAX_WIDTH - orig_width) / 2.0)
                raise ValueError(
                    f"arrival_heading_slack={slack:.3f}rad({np.degrees(slack):.1f}°) 过大：加宽后朝向门全宽 "
                    f"{orig_width + 2.0 * slack:.3f}rad ≥ {_MAX_WIDTH:.3f}rad 会触发 commonocean AngleInterval 断言崩溃。"
                    f"本 goal 真门全宽 {orig_width:.3f}rad → 最大允许 slack {max_slack:.3f}rad({np.degrees(max_slack):.1f}°)"
                    f"（B1 退火意图上限应 ≤45°；如需更宽须先核 commonocean 环绕语义）")
            gw = copy.deepcopy(self.goal)                     # 独立副本·不碰真 goal（评估恒用真 goal）
            o = gw.state_list[0].orientation
            o.start = float(o.start) - slack                  # 朝向区间两侧各加宽 slack = 放宽朝向门（位置/时间门不动）
            o.end = float(o.end) + slack
            self._goal_widened = gw

    def check(self, ego_state, step: int, obstacle_footprints=()):
        """判定终止。

        ego_state          : [px, py, θ, v]
        step               : int 当前时间步
        obstacle_footprints: 可迭代的 shapely 几何（各他船当前时刻占据；env 用 occupancy_at_time 取）

        返回：(done: bool, flags: dict{time,area,stopped,collision,goal})
        """
        ego = np.asarray(ego_state, dtype=float)
        if ego.shape != (4,):
            raise ValueError(f"ego_state 应为 4 维 [px,py,θ,v]，得到 {ego.shape}")
        if not np.all(np.isfinite(ego)):
            raise ValueError(f"ego_state 含非有限值（NaN/inf）：{ego}")

        px, py = float(ego[0]), float(ego[1])
        theta = wrap_to_pi(ego[2])
        v = float(ego[3])

        # 1_time
        f_time = int(step) >= self.k_max

        # 1_stopped（论文"velocity is zero"）
        f_stopped = abs(v) <= self.stopped_eps

        # 1_area（仅当给了航行区包围盒；公开数据无边界，默认 None→永不触发）
        if self.nav_area_box is not None:
            xmin, ymin, xmax, ymax = self.nav_area_box
            f_area = px < xmin or px > xmax or py < ymin or py > ymax
        else:
            f_area = False

        # 1_collision（本船占据 ∩ 任一他船占据；无官方检测器→shapely 相交）
        f_collision = False
        if obstacle_footprints:
            ego_fp = _ego_footprint(px, py, theta, self.ego_length, self.ego_width)
            for ofp in obstacle_footprints:
                # None 会让 shapely.intersects 静默返 False → 漏判碰撞（安全方向危险）；env 须先过滤（独立复核 MINOR）
                if ofp is None:
                    raise ValueError("obstacle_footprints 含 None（env 须先过滤无效占据，否则静默漏判碰撞）")
                if ego_fp.intersects(ofp):
                    f_collision = True
                    break

        # 1_goal（复用官方 is_reached；需 position/orientation/time_step，velocity 多给无妨）
        state = CustomState(
            position=np.array([px, py], dtype=float),
            orientation=theta,
            velocity=v,
            time_step=int(step),
        )
        # 🆕 L185：去朝向硬门（仅训练/位置-only配置）优先 → 用删朝向的 goal 副本（只判位置）；
        #   否则 B1 slack>0（仅训练放宽）→ 加宽朝向门副本；否则（默认/严格评估）→ 官方真门 = bit-identical
        if self.goal_ignore_orientation and self._goal_posonly is not None:
            _goal_for_check = self._goal_posonly
        elif self.arrival_heading_slack > 0.0 and self._goal_widened is not None:
            _goal_for_check = self._goal_widened
        else:
            _goal_for_check = self.goal
        f_goal = bool(_goal_for_check.is_reached(state))

        flags = {
            "time": f_time,
            "area": f_area,
            "stopped": f_stopped,
            "collision": f_collision,
            "goal": f_goal,
        }
        done = any(flags.values())
        return done, flags
