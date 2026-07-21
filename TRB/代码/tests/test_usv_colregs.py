"""
COLREGs 模块 (a.1 几何原语) 冒烟测试 —— 全部断言用手算几何 / Table IV 定义，fact-based。
跑：/opt/miniconda3/envs/trb/bin/python 代码/tests/test_usv_colregs.py

约定：β = wrap(θ_ego − atan2(Δy,Δx))，右舷(starboard)正 / 左舷(port)负（= usv_observation D6 + 2024 Table IV）。
扇区（Table IV，权威）：front |β|≤5° / right (5°,112.5°] / left [−112.5°,−5°) / behind |β|>112.5°(=艉向±67.5°)。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from trb_env.usv_colregs import (  # noqa: E402
    A_PM_MAX, DELTA_HEAD_ON, DELTA_LARGE_TURN, R_M_FACTOR, T_HORIZON, T_M, V_PM_MAX, ColregsStatechart,
    RHO_CROSSING, RHO_EMERGENCY, RHO_HEAD_ON, RHO_NO_CONFLICT, RHO_OVERTAKE,
    RHO_STAND_ON, VesselState, _collision_cone, a2u, build_st, collision_possible, crossing,
    drives_faster, head_on, in_behind_sector, in_front_sector, in_left_sector,
    in_right_sector, is_emergency, is_emergency_resolved, keep, maneuver_verified,
    mod_2pi, orientation_delta, orientation_towards_left, orientation_towards_right,
    overtake, persistent_crossing, persistent_head_on, persistent_overtake,
    reach_radius_pm, relative_bearing, safe_speed,
    A_A, A_OMEGA_GRID, ATR, ATL, AACC, get_turning_act, encounter_action_verification,
    DT, EmergencyController, _in_stern_sector, _pos_under_stern_acc, ahead_emergency,
    get_target_ahead, get_target_base, is_emergency_under_acc, stern_emergency,
    tracking_controller, turning_direction,
)
from trb_env.usv_dynamics import make_vessel_params, step as dyn_step  # noqa: E402

_fail = 0
_n = 0
DEG = np.pi / 180.0
L = 175.0  # 本/他船长（集装箱船，Table II）


def chk(tag, got, exp):
    global _fail, _n
    _n += 1
    ok = (got == exp)
    print(f"[{'PASS' if ok else 'FAIL'}] {tag}: got={got} exp={exp}")
    if not ok:
        _fail += 1


def ego(theta_deg=0.0, v=5.0):
    return VesselState(position=np.array([0.0, 0.0]), orientation=theta_deg * DEG,
                       velocity=v, length=L)


def other(x, y, theta_deg=0.0, v=5.0):
    return VesselState(position=np.array([float(x), float(y)]), orientation=theta_deg * DEG,
                       velocity=v, length=L)


print("===== A) 扇区谓词（Table IV，β 右舷正）=====")
e = ego(theta_deg=0.0)  # 本船朝东(+x)
# 正前方(+x) → front
chk("① 正前方 → front", in_front_sector(e, other(2000, 0)), True)
chk("①b 正前方 非 right/left/behind",
    (in_right_sector(e, other(2000, 0)) or in_left_sector(e, other(2000, 0))
     or in_behind_sector(e, other(2000, 0))), False)
# 正右舷(南,−y，朝东时右舷=南) → β=+90° → right
chk("② 右舷(南) β≈+90 → right", in_right_sector(e, other(0, -2000)), True)
chk("②b 右舷 β 值≈+90°", round(relative_bearing(e, other(0, -2000)) / DEG, 1), 90.0)
# 正左舷(北,+y) → β=−90° → left
chk("③ 左舷(北) β≈−90 → left", in_left_sector(e, other(0, 2000)), True)
# 正后方(−x) → |β|=180 → behind
chk("④ 正后方 → behind", in_behind_sector(e, other(-2000, 0)), True)
# ⭐ 判别用例：β=120°（艉向 60°）—— 2024 Table IV = behind（|β|>112.5）；2021 代码(±45°) 会判 right
o120 = other(np.cos(-120 * DEG) * 2000, np.sin(-120 * DEG) * 2000)  # los=−120° → β=+120°
chk("⑤ β=+120° → behind (2024 ±67.5，非 2021 ±45)", in_behind_sector(e, o120), True)
chk("⑤b β=+120° 非 right", in_right_sector(e, o120), False)
# 判别边界另一侧：β=100°（<112.5）→ right（非 behind）
o100 = other(np.cos(-100 * DEG) * 2000, np.sin(-100 * DEG) * 2000)  # β=+100°
chk("⑥ β=+100° → right（非 behind）", in_right_sector(e, o100), True)
chk("⑥b β=+100° 非 behind", in_behind_sector(e, o100), False)
# 扇区互斥 + 全覆盖：随机 50 方位，恰好落 1 个扇区
np.random.seed(0)
multi = 0
for _ in range(50):
    ang = np.random.uniform(-np.pi, np.pi)
    om = other(np.cos(ang) * 1500, np.sin(ang) * 1500, theta_deg=37.0)
    cnt = sum([in_front_sector(e, om), in_right_sector(e, om),
               in_left_sector(e, om), in_behind_sector(e, om)])
    if cnt != 1:
        multi += 1
chk("⑦ 50 随机方位扇区互斥且全覆盖(恰 1 个)", multi, 0)

print("\n===== B) 朝向谓词（Table IV，mod 2π 区间）=====")
# head_on 用 ¬orientation_delta(Δhead, offset=π)：反平行 → delta False
chk("⑧ 反平行(θ_m=180) delta(offset π)=False", orientation_delta(ego(0), other(1, 0, 180), DELTA_HEAD_ON, np.pi), False)
chk("⑨ 同向(θ_m=0) delta(offset π)=True", orientation_delta(ego(0), other(1, 0, 0), DELTA_HEAD_ON, np.pi), True)
# overtake 用 ¬orientation_delta(67.5, 0)：同向 → delta False
chk("⑩ 同向(θ_m=0) delta(offset 0)=False", orientation_delta(ego(0), other(1, 0, 0), 67.5 * DEG, 0.0), False)
# towards_left：θ_m−θ_l=+90 → True；towards_right：−90 → True
chk("⑪ θ_m=+90 towards_left=True", orientation_towards_left(ego(0), other(1, 0, 90)), True)
chk("⑫ θ_m=+90 towards_right=False", orientation_towards_right(ego(0), other(1, 0, 90)), False)
chk("⑬ θ_m=−90 towards_right=True", orientation_towards_right(ego(0), other(1, 0, -90)), True)
chk("⑭ θ_m=−90 towards_left=False", orientation_towards_left(ego(0), other(1, 0, -90)), False)
chk("⑮ mod_2pi(−90°)=270°", round(mod_2pi(-90 * DEG) / DEG, 1), 270.0)

print("\n===== C) 速度谓词 =====")
chk("⑯ drives_faster(5>3)=True", drives_faster(ego(0, 5), other(1, 0, 0, 3)), True)
chk("⑰ drives_faster(3>5)=False", drives_faster(ego(0, 3), other(1, 0, 0, 5)), False)
chk("⑱ safe_speed(5≤9.5)=True", safe_speed(ego(0, 5), 9.5), True)
chk("⑲ safe_speed(10≤9.5)=False", safe_speed(ego(0, 10), 9.5), False)

print("\n===== D) collision_possible（速度障碍 + r_m=3·l_m + v_ε）=====")
# 对遇接近：本船东行 v5，他船在 +x 2000 处西行 v5 → 相对速度沿 +x 指向他船 → True
chk("⑳ 对遇接近 → True", collision_possible(ego(0, 5), other(2000, 0, 180, 5), T_HORIZON), True)
# 同向远离：他船同速同向在前 → 相对速度 0 → 不接近 → False
chk("㉑ 同向不接近 → False", collision_possible(ego(0, 5), other(2000, 0, 0, 5), T_HORIZON), False)
# too-close：d=100 < r_m=525 → True（守卫）
chk("㉒ too-close(d=100<r_m=525) → True", collision_possible(ego(0, 5), other(100, 0, 180, 5), T_HORIZON), True)
chk("㉒b r_m=3·l_m=525", R_M_FACTOR * L, 525.0)
# 接近但艏向不指向他船：本船东行，他船正北静止 → 速度不落锥内 → False
chk("㉓ 东行 vs 正北静止 → False(锥外)", collision_possible(ego(0, 10), other(0, 1000, -90, 0), T_HORIZON), False)
# 接近条件失败：他船很远，相对速度不足以 t_horizon 内闭合 → False
far = other(np.cos(0) * 50000, 0, 180, 0.05)  # d=50000，相对速度小
chk("㉔ 太远不闭合 → False", collision_possible(ego(0, 0.05), far, T_HORIZON), False)

print("\n===== E) fail-fast =====")
try:
    VesselState(position=np.array([0.0, np.nan]), orientation=0.0, velocity=5.0, length=L)
    chk("㉕ position NaN → 报错", "未报错", "ValueError")
except ValueError:
    chk("㉕ position NaN → 报错", "ValueError", "ValueError")
try:
    VesselState(position=np.array([0.0, 0.0]), orientation=0.0, velocity=5.0, length=-1.0)
    chk("㉖ length≤0 → 报错", "未报错", "ValueError")
except ValueError:
    chk("㉖ length≤0 → 报错", "ValueError", "ValueError")
try:
    VesselState(position=np.array([0.0, 0.0]), orientation=0.0, velocity=-5.0, length=L)
    chk("㉗ velocity<0 → 报错", "未报错", "ValueError")
except ValueError:
    chk("㉗ velocity<0 → 报错", "ValueError", "ValueError")


print("\n===== F) 扇区边界 / 切点 / 已知盲区（agent 复核补强）=====")
def at_beta(beta_deg, d=2000.0, theta_m_deg=0.0, v=5.0):
    """在本船(朝东)的相对方位 β=beta_deg 处放他船（β 右舷正 → los = −β）。"""
    los = -beta_deg * DEG
    return other(np.cos(los) * d, np.sin(los) * d, theta_m_deg, v)

# 扇区精确边界跨界（front±5° / 侧扇区到 112.5°）
chk("㉘ β=4° → front（边界内）", in_front_sector(e, at_beta(4)), True)
chk("㉙ β=6° → right（出 front）", in_right_sector(e, at_beta(6)), True)
chk("㉚ β=111° → right（未到后扇区）", in_right_sector(e, at_beta(111)), True)
chk("㉛ β=114° → behind（过 112.5°）", in_behind_sector(e, at_beta(114)), True)
# 朝向 towards_left / towards_right 全圆互斥
both = sum(1 for k in range(72)
           if orientation_towards_left(ego(0), other(1, 0, k * 5.0))
           and orientation_towards_right(ego(0), other(1, 0, k * 5.0)))
chk("㉜ towards_left/right 全圆互斥（0 重叠）", both, 0)
# 碰撞锥切点到他船心距 = r_m = 3·l_m = 525
cone = _collision_cone(np.array([0.0, 0.0]), np.array([2000.0, 0.0]), R_M_FACTOR * L)
coords = list(cone.exterior.coords)  # [p_l, t0, p_m, t1, p_l]
chk("㉝ 切点 t0 到他船心 = r_m=525",
    round(float(np.linalg.norm(np.array(coords[1]) - np.array([2000.0, 0.0]))), 3), 525.0)
chk("㉞ 切点 t1 到他船心 = r_m=525",
    round(float(np.linalg.norm(np.array(coords[3]) - np.array([2000.0, 0.0]))), 3), 525.0)
# 已知保守盲区（agent 复核 🟡-A，继承自论文 Table IV 全心距 d）：慢速尾追 t<420s 会撞却判 safe
#   他船正后方 −3000 同向 v=12 追 ego v=5：相对速度 7 < d/t_h=7.14 → False（盲区，由 a.2 is_emergency 兜底）
chk("㉟ 慢速尾追已知盲区 → False（态势分类保守；a.2 is_emergency 兜底）",
    collision_possible(ego(0, 5), other(-3000, 0, 0, 12), T_HORIZON), False)
# t_horizon ≤ 0 守卫
try:
    collision_possible(ego(0, 5), other(2000, 0, 180, 5), 0.0)
    chk("㊱ t_horizon=0 → 报错", "未报错", "ValueError")
except ValueError:
    chk("㊱ t_horizon=0 → 报错", "ValueError", "ValueError")


print("\n===== G) (a.2-i) 态势分类 head_on/crossing/overtake/keep + Lemma 1 互斥 =====")
def classify(s_l, s_m):
    return {"head_on": head_on(s_l, s_m), "crossing": crossing(s_l, s_m),
            "overtake": overtake(s_l, s_m), "keep": keep(s_l, s_m)}

OFF = {"head_on": False, "crossing": False, "overtake": False, "keep": False}
def only(name):
    d = dict(OFF)
    d[name] = True
    return d

# 对遇：ego 东行 v5，他船正前 2000 西行 v5（反平行汇聚）
chk("㊲ 对遇 → 仅 head_on", classify(ego(0, 5), other(2000, 0, 180, 5)), only("head_on"))
# 交叉 ego give-way：他船右前方北行，汇聚
chk("㊳ 交叉 give-way → 仅 crossing", classify(ego(0, 5), other(700, -700, 90, 5)), only("crossing"))
# 追越 ego 追：他船正前慢速同向
chk("㊴ 追越 → 仅 overtake", classify(ego(0, 8), other(1000, 0, 0, 3)), only("overtake"))
# 直航 ego 被追越：他船正后快速同向
chk("㊵ 被追越 → 仅 keep", classify(ego(0, 3), other(-1000, 0, 0, 8)), only("keep"))
# 直航 ego 是交叉 stand-on：他船左前方南行，汇聚
chk("㊶ 交叉 stand-on → 仅 keep", classify(ego(0, 5), other(700, 700, -90, 5)), only("keep"))
# ρ0 无冲突：他船很远不接近 → 全 False
chk("㊷ 无冲突 → 全 False (ρ0)", classify(ego(0, 5), other(50000, 50000, 0, 0.01)), OFF)
# Lemma 1 修正（本窗口发现）：论文「四者至多一个真」在字面 Table IV 不严格成立——overtake 用他船视角
#   in_behind_sector(s_m,s_l)，与 crossing/keep 的本船视角扇区在近边界+航向偏斜区会解耦 → 固有重叠
#   {crossing∧overtake, keep∧overtake}（交 a.3 状态机优先级解）。但**同视角/被航向排除的对必须严格互斥**
#   （验证扇区/朝向谓词正确）：head_on 与任何同真 / crossing∧keep 必须 = 0。
np.random.seed(1)
forbidden, allowed = 0, 0
for _ in range(5000):
    sl = ego(np.random.uniform(-180, 180), np.random.uniform(0.0, 9.5))
    sm = other(np.random.uniform(-3000, 3000), np.random.uniform(-3000, 3000),
               np.random.uniform(-180, 180), np.random.uniform(0.0, 9.5))
    h, c, o, k = head_on(sl, sm), crossing(sl, sm), overtake(sl, sm), keep(sl, sm)
    if (h and (c or o or k)) or (c and k):       # 禁止：head_on 任意 / crossing∧keep（同/近视角必互斥）
        forbidden += 1
    if (c and o) or (k and o):                   # 允许：cross-frame overtake 固有重叠（a.3 优先级解）
        allowed += 1
chk("㊸ Lemma 1 禁止重叠(head_on 任意 / crossing∧keep)=0（扇区+朝向正确）", forbidden, 0)
print(f"   (信息) 5000 场景 cross-frame overtake 固有重叠 = {allowed}（忠实 Table IV，a.3 状态机优先级解，非 bug）")

print("===== E) is_emergency（集合预测占据相交，2024 §IV-B R1）=====")
# 迎面：他船正前方 1000m 相向，都速 8，~62s 相撞（t_pred=180 内）→ 紧急
chk("E1 迎面相向必撞 → 紧急", is_emergency(ego(0, 8), other(1000, 0, 180, 8)), True)
# 交叉接近：他船右前方(南偏前)朝北(+y)穿越，~112s 交汇 → 紧急
chk("E2 交叉接近交汇 → 紧急", is_emergency(ego(0, 8), other(900, -900, 90, 8)), True)
# 侧方 6000m 平行同向同速 → 永不接近 → 非紧急
chk("E3 侧方平行远离 → 非紧急", is_emergency(ego(0, 5), other(0, 6000, 0, 5)), False)
# 远处侧方静止他船（v=0，仍按点质量可达膨胀）→ 本船直线不经过 → 非紧急
chk("E4 远处侧方静止 → 非紧急", is_emergency(ego(0, 5), other(3000, 8000, 0, 0)), False)
# fail-fast
try:
    is_emergency(ego(), other(100, 0), t_pred=-1.0); _r = False
except ValueError:
    _r = True
chk("E5 t_pred<0 fail-fast", _r, True)
try:
    is_emergency(ego(), other(100, 0), dt_check=0.0); _r = False
except ValueError:
    _r = True
chk("E6 dt_check≤0 fail-fast", _r, True)

print("===== E') is_emergency O(1) 远场保守早退（P1/D42·等价提速、绝不漏检真紧急）=====")
import trb_env.usv_colregs as _cl
import numpy as _np
# 保守界 = reach_radius_pm(t_pred+DT) + 他船外接圆 + 本船外接圆
def _far_R(sl, sm):
    return (_cl.reach_radius_pm(_cl.T_PRED + _cl.DT, _cl.DT_REACH)
            + _cl._vessel_circumradius(sm.length, None) + _cl._vessel_circumradius(sl.length, _cl.EGO_WIDTH))
def _ctr_dist(sl, sm):
    vm, vl = _cl.velocity_vector(sm), _cl.velocity_vector(sl)
    return _cl._segment_segment_distance(sm.position, sm.position + vm * _cl.T_PRED,
                                         sl.position, sl.position + vl * _cl.T_PRED)
# E7 远场静止他船 → 中心线段距离 > 保守界(早退短路) ∧ is_emergency False
_fl, _fm = ego(0, 5), other(4000, 4000, 0, 0)
chk("E7 远场 → 线段距离 > 保守界(早退短路)", _ctr_dist(_fl, _fm) > _far_R(_fl, _fm), True)
chk("E7b 远场 → is_emergency False", is_emergency(_fl, _fm), False)
# E8 近场真迎面紧急 → 距离 ≤ 界(早退【不】短路、落回精确循环) ∧ is_emergency True
_nl, _nm = ego(0, 8), other(1000, 0, 180, 8)
chk("E8 近场真紧急 → 距离 ≤ 保守界(不短路、落回 shapely 精确)", _ctr_dist(_nl, _nm) <= _far_R(_nl, _nm), True)
chk("E8b 近场真紧急 → is_emergency True", is_emergency(_nl, _nm), True)
# E9 ⭐等价守护(load-bearing)：关早退(monkeypatch 距离=-1=永不短路=原 shapely 循环) vs 开早退，3000 fuzz 逐位相同。
#    破坏保守界(如界调小漏 ego_circ)→近场真紧急被误短路→开/关结果分叉→本项翻 FAIL。
_orig_segd = _cl._segment_segment_distance
def _ref_emer(sl, sm):
    _cl._segment_segment_distance = lambda *a, **k: -1.0   # 恒 -1 → 永不 > 界 → 跑完整原循环=参照
    try:
        return is_emergency(sl, sm)
    finally:
        _cl._segment_segment_distance = _orig_segd
_rng = _np.random.default_rng(7)
_mis = _ntrue = 0
for _i in range(3000):
    _sl = VesselState(position=_np.array([0.0, 0.0]), orientation=_rng.uniform(-_np.pi, _np.pi),
                      velocity=_rng.uniform(0, 10), length=175.0)
    if _i % 2 == 0:                                         # 一半近场（确保测到真紧急不被误短路）
        _dd, _aa = _rng.uniform(50, 1200), _rng.uniform(-_np.pi, _np.pi)
        _p = _np.array([_dd * _np.cos(_aa), _dd * _np.sin(_aa)])
    else:
        _p = _rng.uniform(-5000, 5000, size=2)
    _sm = VesselState(position=_p, orientation=_rng.uniform(-_np.pi, _np.pi),
                      velocity=_rng.uniform(0, 10), length=_rng.uniform(100, 280))
    if is_emergency(_sl, _sm) != _ref_emer(_sl, _sm):
        _mis += 1
    if _ref_emer(_sl, _sm):
        _ntrue += 1
chk("E9 早退等价守护：3000 fuzz 开/关早退逐位相同(含真紧急·破坏保守界则翻FAIL)", _mis == 0 and _ntrue > 100, True)
# E10 线段距离 helper 边界：退化点(静止船段=点)→点距 ∧ 真相交→0
chk("E10 退化点-线段距离", abs(_cl._point_segment_distance(_np.array([0., 3.]), _np.array([0., 0.]), _np.array([0., 0.])) - 3.0) < 1e-9, True)
chk("E10b 相交线段距离=0", _cl._segment_segment_distance(_np.array([-1., 0.]), _np.array([1., 0.]), _np.array([0., -1.]), _np.array([0., 1.])) == 0.0, True)

print("===== F) is_emergency_resolved（後扇区±90 ∧ 远离(朝向点积≤0) ∧ 距离≥d_resolved）=====")
# d_resolved 默认 = 2·l_ego = 350m
chk("F1 后方400m朝反向远离 → 解除", is_emergency_resolved(ego(0, 5), other(-400, 0, 180, 5)), True)
chk("F2 后方但仅300m(<350) → 未解除", is_emergency_resolved(ego(0, 5), other(-300, 0, 180, 5)), False)
chk("F3 前方400m → 未解除(非后扇区)", is_emergency_resolved(ego(0, 5), other(400, 0, 180, 5)), False)
chk("F4 后方但他船同向(点积>0) → 未解除", is_emergency_resolved(ego(0, 5), other(-400, 0, 0, 5)), False)
# 边界：恰好 d_resolved=350 → ≥ 取真（验证 typo 修正方向：物理语义"距离够大"=≥ 而非论文字面 ≤）
chk("F5 恰好350m → 解除(≥，typo 修正方向)", is_emergency_resolved(ego(0, 5), other(-350, 0, 180, 5)), True)
# 朝向恰好垂直(点积=0) → 远离支取真(≤0)
chk("F6 后方且朝向垂直(点积=0) → 解除", is_emergency_resolved(ego(0, 5), other(-400, 0, 90, 5)), True)

print("===== G) persistent_X（¬X(now) ∧ G[Δt,t_react] 恒速预测 X 持续；不含 keep）=====")
# 平行远离：现在/未来均无冲突 → 三个 persistent 全 False
chk("G1 平行远离 → persistent_crossing False", persistent_crossing(ego(0, 5), other(0, 6000, 0, 5)), False)
chk("G2 平行远离 → persistent_head_on False", persistent_head_on(ego(0, 5), other(0, 6000, 0, 5)), False)
chk("G3 平行远离 → persistent_overtake False", persistent_overtake(ego(0, 5), other(0, 6000, 0, 5)), False)
# 迎面 7650m：现在尚非 head_on(太远 collision_possible False)，但 t=10..60 恒速接近持续 head_on
chk("G4 同场景现在尚非 head_on(¬X now)", head_on(ego(0, 9), other(7650, 0, 180, 9)), False)
chk("G5 迎面将持续 head_on → persistent True", persistent_head_on(ego(0, 9), other(7650, 0, 180, 9)), True)
# 现在已 head_on(近距迎面) → persistent_head_on False（¬X(now) 不满足）
chk("G6 现在已 head_on → persistent False(¬X now)", persistent_head_on(ego(0, 8), other(1500, 0, 180, 8)), False)

print("===== H) 可达圆盘 over-approximate 连续 Ω_pm（质量命门 = 不漏报根因；2 agent 复核收紧措辞）=====")
# 真正的不漏报保证：reach 圆盘严格包住【连续】点质量可达集（真实他船连续运动 ⊂ 连续 Ω_pm ⊂ 圆盘），
# 与离散步长无关。H1a 确认连续可达半径精确 = ½·a·t²；H1b 解析确认 reach 对它有正裕度 ½·a·t·Δt_reach。
def _cont_reach_emp(v0mag, t, h=0.1, n_dir=48):
    """细离散 bang-bang 方向扫（h→0 逼近连续可达集最大相对半径）。"""
    c = np.array([v0mag, 0.0]) * t
    steps = int(round(t / h))
    mx = -1e18
    for kk in range(n_dir):
        e = np.array([np.cos(2 * np.pi * kk / n_dir), np.sin(2 * np.pi * kk / n_dir)])
        p = np.zeros(2); v = np.array([v0mag, 0.0])
        for _ in range(steps):
            v = v + A_PM_MAX * e * h
            nv = np.linalg.norm(v)
            if nv > V_PM_MAX:
                v = v * (V_PM_MAX / nv)
            p = p + v * h
        mx = max(mx, float(np.linalg.norm(p - c)))
    return mx
# H1a：连续可达半径经验值（细离散 h=0.1）≈ 解析连续真值 ½·a·t²
_t = 180.0
_emp = _cont_reach_emp(5.0, _t)
_cont = 0.5 * A_PM_MAX * _t * _t
chk("H1a 连续可达半径经验 ≈ ½·a·t² (误差<1m)", abs(_emp - _cont) < 1.0, True)
# H1b：reach 圆盘对连续真值 ½·a·t² 严格正裕度（= ½·a·t·Δt_reach > 0，解析确定）→ 不漏报根基
_margins = [reach_radius_pm(t) - 0.5 * A_PM_MAX * t * t for t in (30.0, 60.0, 120.0, 180.0)]
chk("H1b reach 严格 over 连续真值(全 t 正裕度)", all(m > 0.0 for m in _margins), True)
print(f"   (信息) reach−连续真值 裕度 @t=180 = {_margins[-1]:.3f}m = ½·a·t·Δt_reach（>0 严格不漏报，与离散步长无关）")
# 回归：is_emergency 集合预测覆盖 a.1 collision_possible 慢速尾追盲区——
#   圆盘不漏连续可达点(H1b) → t_pred 内任何真实可达碰撞必被捕获；t∈(t_pred,t_horizon] 的远期碰撞随时间
#   推进变 imminent 时被后续步捕获（receding-horizon，忠实论文 t_pred=180<t_horizon=420 的设计）。
print("   (信息) 回归：is_emergency 用集合预测(不依赖 collision_possible)，H1b 证圆盘不漏连续可达点 → 覆盖 a.1 盲区(imminent 侧)")

print("===== I) 状态机 Γ（2024 §IV-C Fig.3：ρ0-ρ5 转移 + 优先级 + tie-break）=====")
def _vs(x, y, th_deg, v, l=L):
    return VesselState(position=np.array([float(x), float(y)]), orientation=th_deg * DEG, velocity=v, length=l)
def sim_visit(ego_v, ego_th, ox0, oy0, oth_th, oth_v, t_max=700, dt=10.0):
    """两船恒速接近，逐步推进状态机，返回访问过的状态集合（测 ρ0→give-way 经 persistent 自然进入）。"""
    sc = ColregsStatechart()
    ev = ego_v * np.array([np.cos(ego_th * DEG), np.sin(ego_th * DEG)])
    ov = oth_v * np.array([np.cos(oth_th * DEG), np.sin(oth_th * DEG)])
    visited = set()
    for k in range(int(t_max / dt) + 1):
        t = k * dt
        el = VesselState(position=ev * t, orientation=ego_th * DEG, velocity=ego_v, length=L)
        om = VesselState(position=np.array([ox0, oy0]) + ov * t, orientation=oth_th * DEG, velocity=oth_v, length=L)
        visited.add(sc.step(el, om))
    return visited

sc = ColregsStatechart()
chk("I1 初始 → ρ0", sc.rho, RHO_NO_CONFLICT)
chk("I2 平行远离 → 维持 ρ0", sc.step(_vs(0, 0, 0, 5), _vs(0, 6000, 0, 5)), RHO_NO_CONFLICT)
chk("I3 近距迎面 → ρ5 紧急(R1 最高)", sc.step(_vs(0, 0, 0, 8), _vs(300, 0, 180, 8)), RHO_EMERGENCY)
chk("I4 仍紧急 → 维持 ρ5", sc.step(_vs(0, 0, 0, 8), _vs(250, 0, 180, 8)), RHO_EMERGENCY)
chk("I5 他船后方远离(resolved) → 退出 ρ5 回 ρ0", sc.step(_vs(0, 0, 0, 8), _vs(-400, 0, 180, 8)), RHO_NO_CONFLICT)
# R1 优先级覆盖：处于 give-way 时 is_emergency 真 → 强制 ρ5
sc.rho = RHO_HEAD_ON
chk("I6 give-way 中 is_emergency → 强制 ρ5(R1>R3-R5)", sc.step(_vs(0, 0, 0, 8), _vs(300, 0, 180, 8)), RHO_EMERGENCY)
# ρ0 → ρ1 keep（即时，被追越场景；距离落在 cp(≤420s) 内但非紧急(180s 内不撞)区间 → 非 ρ5）
# ⚠️ 设计含义：is_emergency(R1) 在 t_pred=180s 内会撞时抢占一切；故 ρ1-ρ4 仅出现于「cp(420s) 但非 imminent(180s)」区间。
sc2 = ColregsStatechart()
chk("I7 ρ0 被追越(keep，非紧急区间) → ρ1 stand-on", sc2.step(_vs(0, 0, 0, 3), _vs(-2000, 0, 0, 8)), RHO_STAND_ON)
# ρ1 → ρ0（¬collision_possible）
sc2.rho = RHO_STAND_ON
chk("I8 ρ1 脱离(¬cp) → ρ0", sc2.step(_vs(0, 0, 0, 5), _vs(0, 9000, 0, 5)), RHO_NO_CONFLICT)
# ρ1 → give-way（¬keep ∧ cp，即时分类，更高优先级 give-way 现身；距离非紧急区间）
sc2.rho = RHO_STAND_ON
chk("I9 ρ1 中迎面现身(¬keep∧cp，非紧急区间) → ρ2", sc2.step(_vs(0, 0, 0, 9), _vs(5000, 0, 180, 9)), RHO_HEAD_ON)
# give-way 维持（Requirement 2）+ 脱离
sc2.rho = RHO_HEAD_ON
chk("I10 give-way 接近中 → 维持 ρ2", sc2.step(_vs(0, 0, 0, 9), _vs(5000, 0, 180, 9)), RHO_HEAD_ON)
chk("I11 give-way 脱离(¬cp) → ρ0", sc2.step(_vs(0, 0, 0, 9), _vs(9500, 0, 180, 9)), RHO_NO_CONFLICT)
# ρ0 → give-way 经 persistent 自然进入（恒速接近模拟，访问集合含目标状态）
chk("I12 迎面接近(persistent) → 访问 ρ2", RHO_HEAD_ON in sim_visit(9, 0, 12000, 0, 180, 9), True)
chk("I13 交叉接近(persistent) → 访问 ρ3", RHO_CROSSING in sim_visit(5, 0, 2500, -2500, 90, 5), True)
chk("I14 追越接近(persistent) → 访问 ρ4", RHO_OVERTAKE in sim_visit(8, 0, 3000, 0, 0, 3), True)
# tie-break：crossing∧overtake 固有重叠 → 确定性判 overtake（head_on>overtake>crossing）
#   ⚠️ 2026-06-10 全面审计修：原 I16 用 seed(1) 随机搜重叠样本，Agent 4 实测偶发 flaky（搜到的样本可能贴碰撞锥切线，
#   shapely 极罕见抖动）→ 改用 margin-robust 钉死样本（±12m/12° 扰动稳定），消除非确定性（03 L12）。
_tb = ColregsStatechart()
_ov_sl, _ov_sm = ego(26, 7.6), other(1038, 145, 36, 2.1)   # 钉死的 crossing∧overtake 即时重叠样本
chk("I15 钉死样本确为 crossing∧overtake∧¬head_on 即时重叠",
    crossing(_ov_sl, _ov_sm) and overtake(_ov_sl, _ov_sm) and not head_on(_ov_sl, _ov_sm), True)
chk("I16 tie-break 即时重叠 → 判 overtake(确定性，head_on>overtake>crossing)",
    _tb._giveway_instant(_ov_sl, _ov_sm), RHO_OVERTAKE)
# reset
_tb.rho = RHO_EMERGENCY
_tb.reset()
chk("I17 reset() → ρ0", _tb.rho, RHO_NO_CONFLICT)

print("===== J) maneuver_verified（2024 式8：机动 rule-compliant 验证；(b) 地基谓词）=====")
KEEP5 = [[0.0, 0.0]] * 5            # 保持航向航速 5 段×40s=200s
# J1 他船远侧平行 + 本船保持 → 全程不撞 + 末 ¬cp → verified
chk("J1 远侧平行+保持 → verified", maneuver_verified(_vs(0, 0, 0, 5), _vs(0, 8000, 0, 5), KEEP5), True)
# J2 迎面对撞 + 保持直行 → 占据相交 → 不 verified
chk("J2 迎面对撞+保持直行 → 不verified", maneuver_verified(_vs(0, 0, 0, 8), _vs(1500, 0, 180, 8), KEEP5), False)
# J3 交叉 give-way 局面 + 本船不让(直行) → 不 verified（汇聚相撞）
chk("J3 交叉+不让直行 → 不verified", maneuver_verified(_vs(0, 0, 0, 5), _vs(700, -700, 90, 5), KEEP5), False)
# J4 dobs,safety 膨胀生效：他船基形(175)在 y=300 侧旁本不撞本船直行(y≈0)，膨胀 +dobs,safety/边 → 覆盖 → 不verified
chk("J4 dobs,safety 膨胀生效(基形不撞/膨胀撞) → 不verified",
    maneuver_verified(_vs(0, 0, 0, 5), _vs(600, 300, 0, 0.0), KEEP5), False)
# J5/J6 fail-fast
try:
    maneuver_verified(_vs(0, 0, 0, 5), _vs(100, 0, 180, 5), []); _r = False
except ValueError:
    _r = True
chk("J5 空 control_seq fail-fast", _r, True)
try:
    maneuver_verified(_vs(0, 0, 0, 5), _vs(100, 0, 180, 5), KEEP5, t_m=0.0); _r = False
except ValueError:
    _r = True
chk("J6 t_m≤0 fail-fast", _r, True)
# J7 多段控制 state 跨段累积（3 段加速，速度逐段增长）+ 远他船 → 链式仿真不崩 + verified
chk("J7 多段加速链式仿真(远他船) → verified", maneuver_verified(_vs(0, 0, 0, 5), _vs(0, 8000, 0, 5), [[0.3, 0.0]] * 3), True)
# J8 减速机动不崩溃（MAJOR-A 回归）：大减速使 v→0，clip_velocity 应 floor 不抛、远他船 → verified
chk("J8 大减速机动不崩溃(MAJOR-A) → verified", maneuver_verified(_vs(0, 0, 0, 9), _vs(0, 9000, 0, 5), [[-0.24, 0.0]] * 5), True)
# J9 t=0 初始占据相交（清晰重叠案）：本船起步即深陷他船 dobs,safety 膨胀块内 → t=0 检出 → 不verified。
#   ⚠️ 诚实标注（2026-06-10 主窗口复核）：此案为**持续重叠**（本船穿行膨胀块，t>0 采样也会抓到），
#   **删掉 t=0 检查仍返回 False、不构成回归守护**；MAJOR-B 真隔离回归见 J11。
chk("J9 t=0 初始占据相交检出(清晰重叠) → 不verified", maneuver_verified(_vs(0, 0, 0, 5), _vs(200, 0, 90, 0.0), [[0.0, 0.0]] * 2), False)
# J10 约束积分末位准确（防回退到 clip_velocity 漂移方案）：本船加速到约束真值末位 x≈1858 处放小膨胀他船 →
#   方法C 末位准 → 末态检出碰撞 False；旧 clip 方案会偏前 ~42m 漏判（True）。锁定正确性复核成果。
chk("J10 约束积分末位准(防 clip 漂移回退) → 检出碰撞", maneuver_verified(_vs(0, 0, 0, 5), _vs(1858, 0, 0, 0.0), [[0.24, 0.0]] * 5, obs_width=10.0), False)
# J11 t=0-ONLY 占据相交（**真隔离 MAJOR-B**，补 J9 不能隔离之缺；2026-06-10 主窗口复核加）：
#   本船朝 -x(v=9.5)、他船 (520,0) 朝 +x(v=9.5) 双向分离 → t=0 重叠 5m（他船左缘 82.5 < 本船前缘 87.5）、
#   t=0.5 间隙 4.5m。**仅 t=0 检查能判 False**；删 t=0 检查则 t>0 采样全 miss、末态远离 → 误返 True。
#   故此用例**删修复即 FAIL**（数值已核：真=False / 无t0=True / t=0相交=True / t=0.5相交=False）。
chk("J11 t=0-only 占据相交(真隔离 MAJOR-B) → 不verified",
    maneuver_verified(_vs(0, 0, 180, 9.5), _vs(520, 0, 0, 9.5), [[0.0, 0.0]] * 1), False)

print("\n===== K) Alg.2 build_st + a2u（2024 §V-B：BFS 机动合成，地基 = maneuver_verified）=====")
# K1/K2 a2u：动作序列 → control_seq（每段 hold t_m）
chk("K1 a2u 单动作 → control_seq", a2u([(0.0, -0.012)]), [[0.0, -0.012]])
chk("K2 a2u 多段(tuple) → list", a2u(((0.0, -0.012), (0.016, 0.0))), [[0.0, -0.012], [0.016, 0.0]])
AACC_K = [(0.0, 0.0), (0.016, 0.0), (0.048, 0.0)]   # 保速 + 加速集 Aacc
# K3 远他船 + 右转候选 → 深度1 即 verified（非空 ∧ 最短节点长 1）
_r = build_st(_vs(0, 0, 0, 5), _vs(0, 8000, 0, 5), (0.0, -0.012), AACC_K)
chk("K3 远他船/右转 → 深度1 verified", bool(_r) and min(len(n) for n in _r) == 1, True)
# K4 ego 起步即陷他船膨胀块(t=0 重叠) → 任何机动失败 → ∅
chk("K4 t=0重叠 → build_st ∅", len(build_st(_vs(0, 0, 0, 5), _vs(200, 0, 90, 0.0), (0.0, -0.012), AACC_K)), 0)
# K5 迎面 3000m + 强转(0.018→41°/40s) → 深度1 解开
_r = build_st(_vs(0, 0, 0, 8), _vs(3000, 0, 180, 8), (0.0, -0.018), AACC_K)
chk("K5 迎面3000/强转 → 深度1 verified", bool(_r) and min(len(n) for n in _r) == 1, True)
# K6 ⭐ 迎面 2000m + 中转(0.012→27°/40s)：单段40s 不够、需 BFS 生长到 depth-2 才解开（旁证 K6b）
_r = build_st(_vs(0, 0, 0, 8), _vs(2000, 0, 180, 8), (0.0, -0.012), AACC_K)
chk("K6 迎面2000/中转 → BFS 生长到深度2", bool(_r) and min(len(n) for n in _r) == 2, True)
chk("K6b 旁证：depth-1 单段40s 确实失败(故须生长)", maneuver_verified(_vs(0, 0, 0, 8), _vs(2000, 0, 180, 8), [[0.0, -0.012]]), False)
# K7 弱转(13.7°<Δlarge_turn) + 空 Aacc + 近迎面 → 无解 ∅
chk("K7 弱转/空Aacc/近迎面 → ∅", len(build_st(_vs(0, 0, 0, 8), _vs(1500, 0, 180, 8), (0.0, -0.006), [])), 0)
# K8 fail-fast
try:
    build_st(_vs(0, 0, 0, 5), _vs(0, 8000, 0, 5), (0.0, -0.012), AACC_K, t_m=0.0); _r = False
except ValueError:
    _r = True
chk("K8 t_m≤0 fail-fast", _r, True)
# K9 返回结构 = set[tuple[(a,ω) tuple, ...]]
_r = build_st(_vs(0, 0, 0, 5), _vs(0, 8000, 0, 5), (0.0, -0.012), AACC_K)
_ok = isinstance(_r, set) and all(isinstance(n, tuple) and all(isinstance(a, tuple) and len(a) == 2 for a in n) for n in _r)
chk("K9 返回 set[tuple[(a,ω)...]] 结构", _ok, True)
# K10/K11/K12 输入守卫（2026-06-10 组件1 对抗复核 Agent B 2 MINOR + 同源 mv 子步缺口；修复回归）
try:
    build_st(_vs(0, 0, 0, 5), _vs(0, 8000, 0, 5), (0.0, -0.012), [(0.0, 0.0)], t_m=1e-2, t_max_m=1e4); _r = False
except ValueError:
    _r = True
chk("K10 build_st 病态 t_max_m/t_m 比值 → fail-fast(防runaway)", _r, True)
try:
    build_st(_vs(0, 0, 0, 8), _vs(2000, 0, 180, 8), (float("nan"), -0.012), [(0.0, 0.0)]); _r = False
except ValueError:
    _r = True
chk("K11 build_st ac 含 NaN → 干净 fail-fast(非GEOSException)", _r, True)
try:
    maneuver_verified(_vs(0, 0, 0, 5), _vs(0, 8000, 0, 5), [[0.0, 0.0]], t_m=1e9); _r = False
except ValueError:
    _r = True
chk("K12 maneuver_verified absurd t_m 子步爆炸 → fail-fast", _r, True)
# K13 旁证守卫是 backstop 非紧约束：合法高比值(ratio=100) 仍正常
chk("K13 合法高比值 ratio=100 不误伤", bool(build_st(_vs(0, 0, 0, 5), _vs(0, 8000, 0, 5), (0.0, -0.012), [(0.0, 0.0)], t_m=2.0, t_max_m=200.0)), True)

print("\n===== L) 全面审计补漏：give-way 判别支 + 状态机退出门 + persistent tie-break（FG 假守护修复，2026-06-10）=====")
# 背景：全面审计变异测试发现一簇 J9 式假守护——给路谓词每个"判别支"删掉、状态机 ρ5 退出门删掉、
#   persistent tie-break 调换，旧测试仍全绿（不守护）。下列 L1-L6 每条均为 margin-robust 见证（±15m/15° 扰动稳定，
#   非随机搜避 I16 脆弱），且经变异验证"删被守护逻辑即翻 FAIL"（详见 03 L12）。
# L1-L4：give-way 谓词"判别支拒真"——cp∧其余支=True 但判别支=False → 谓词须 False（删该支即误判 True）。
chk("L1 head_on 需 in_front：右扇区+近反平行+cp → False", head_on(_vs(0, 0, 72, 8.2), _vs(-300, 114, -110, 3.3)), False)
chk("L2 crossing 需 tow_left：右扇区+他船非左转+cp → False", crossing(_vs(0, 0, 115, 2.7), _vs(196, 470, -31, 1.6)), False)
chk("L3 overtake 需 drives_faster：后扇区+同向+本船更慢(v_ε内cp真) → False", overtake(_vs(0, 0, -174, 5.8), _vs(-59, -211, -157, 6.0)), False)
chk("L4 keep 需 in_left：右扇区+他船朝右+cp+他船未追越 → False", keep(_vs(0, 0, 1.5, 5.9), _vs(228, -347, -96, 3.4)), False)
# L5：状态机 ρ5 退出门——is_emergency=F ∧ is_emergency_resolved=F 时须维持 ρ5（删退出门会误退 ρ0=谎报脱离紧急）。
_scL = ColregsStatechart(); _scL.rho = RHO_EMERGENCY
chk("L5 ρ5 中 ¬emergency∧¬resolved → 维持 ρ5(退出门守护)", _scL.step(_vs(0, 0, 0, 5), _vs(5000, 200, 0, 5)), RHO_EMERGENCY)
# L6：persistent 路径 tie-break——persistent_crossing∧persistent_overtake 双真 → 判 overtake(ρ4)（调换顺序误判 crossing；I16 只守 instant 路径）。
chk("L6 persistent tie-break 双真 → overtake(ρ4)",
    ColregsStatechart()._giveway_persistent(_vs(0, 0, -164, 8.4), _vs(-2428, -99, -150, 2.8)), RHO_OVERTAKE)

print("\n===== M) 组件2 Alg.3 会遇动作验证 + get_turning_act + 动作集（2024 §V-B → As(ρ1-4)）=====")
# M1 跨模块网格一致（防漂移，全面审计精神）：colregs 动作网格 == env DISCRETE_ACTIONS，且 ATR/ATL/AACC ⊂ 49 网格
from trb_env.usv_env import A_ACC as _ENV_AA, A_OMEGA as _ENV_AW, DISCRETE_ACTIONS as _ENV_GRID  # noqa: E402
chk("M1 colregs 动作网格 == env(防漂移) + 候选⊂49网格",
    A_A == tuple(_ENV_AA) and A_OMEGA_GRID == tuple(_ENV_AW) and all(a in _ENV_GRID for a in ATR + ATL + AACC), True)
# M2 候选转向 ω 资格：|ω|·tm ≥ Δlarge_turn（ATR/ATL 都够，0.006 不够）
chk("M2 候选转向 |ω|·tm≥Δlarge_turn(资格)",
    all(abs(w) * T_M >= DELTA_LARGE_TURN - 1e-9 for _, w in ATR + ATL) and 0.006 * T_M < DELTA_LARGE_TURN, True)
# M3 get_turning_act：他船朝向更右(rel<0)→左转Atl / 更左(rel>0)→右转Atr
chk("M3 get_turning_act 他船更右 → Atl", get_turning_act(_vs(0, 0, 0, 5), _vs(1000, 0, -30, 3)), ATL)
chk("M3b get_turning_act 他船更左 → Atr", get_turning_act(_vs(0, 0, 0, 5), _vs(1000, 0, 30, 3)), ATR)
# M4 keep(ρ1 stand-on) → As={akeep}
chk("M4 keep → As={akeep}", encounter_action_verification(_vs(0, 0, 0, 5), _vs(-2000, 0, 0, 8), "keep"), {(0.0, 0.0)})
# M5 head_on give-way → 右转候选(Atr 都验证通过)
chk("M5 head_on → As={右转候选}", encounter_action_verification(_vs(0, 0, 0, 8), _vs(3000, 0, 180, 8), "head_on"), {(0.0, -0.012), (0.0, -0.018)})
# M6 crossing give-way → 右转
chk("M6 crossing → As={右转候选}", encounter_action_verification(_vs(0, 0, 0, 6), _vs(1600, -1300, 90, 6), "crossing"), {(0.0, -0.012), (0.0, -0.018)})
# M7 overtake(同向后追) → get_turning_act 选右转、As 非空且全右转(ω<0)
_aov = encounter_action_verification(_vs(0, 0, 0, 8), _vs(1000, 0, 0, 3), "overtake")
chk("M7 overtake → As 非空且全右转(ω<0)", len(_aov) > 0 and all(w < 0 for _, w in _aov), True)
# M8 无解(ego 起步陷他船膨胀块 t=0 重叠) → As=∅（give-way 无安全动作，交 emergency/投影兜底）
chk("M8 无解 → As=∅", encounter_action_verification(_vs(0, 0, 0, 5), _vs(200, 0, 90, 0.0), "head_on"), set())
# M9 非法 psi_e → fail-fast
try:
    encounter_action_verification(_vs(0, 0, 0, 5), _vs(0, 8000, 0, 5), "bogus"); _r = False
except ValueError:
    _r = True
chk("M9 非法 psi_e → fail-fast", _r, True)
# M10 give-way As 全是右转(ω<0)（head_on/crossing 永远右转，COLREGS Rule 14/15）
_ahd = encounter_action_verification(_vs(0, 0, 0, 8), _vs(3000, 0, 180, 8), "head_on")
chk("M10 head_on As 全右转(ω<0)", len(_ahd) > 0 and all(w < 0 for _, w in _ahd), True)
# M11 a_keep 形状/有限性守卫（组件2 复核 Agent B MINOR-2；畸形 akeep 不静默产畸形动作）
try:
    encounter_action_verification(_vs(0, 0, 0, 5), _vs(-2000, 0, 0, 8), "keep", a_keep=(0.0,)); _r = False
except ValueError:
    _r = True
chk("M11 a_keep 畸形 → fail-fast", _r, True)

print("\n===== N) 组件3 Alg.1 紧急控制器（2024 §V-A 式6/7 + Alg.1 + Appendix-C → As(ρ5)=a_em）=====")
# N1 turning_direction = Fig.5 四 case（**矢量实测几何**，pymupdf get_drawings，A 复核+主窗口一致：
#    本船朝北；obs_L 相对位置 (−50.5,+58.9) 载 case1 θ=−60°/case2 θ=−40°、obs_R (+67.3,+67.3) 载
#    case3 θ=−155°/case4 θ=−132°；caption: 1,3 右转 / 2,4 左转。同位置仅差 20° 朝向即反号 →
#    判据=航迹侧（β_me<0 本船在他船左舷→右转），非朝向差）
_e9 = ego(theta_deg=90.0, v=5)
chk("N1 Fig.5 case1(obs_L,θ=−60°) → 右转+1", turning_direction(_e9, other(-50.5, 58.9, -60)), 1)
chk("N1b Fig.5 case2(obs_L,θ=−40°) → 左转−1", turning_direction(_e9, other(-50.5, 58.9, -40)), -1)
chk("N1c Fig.5 case3(obs_R,θ=−155°) → 右转+1", turning_direction(_e9, other(67.3, 67.3, -155)), 1)
chk("N1d Fig.5 case4(obs_R,θ=−132°) → 左转−1", turning_direction(_e9, other(67.3, 67.3, -132)), -1)
# N1e 正对头（β_me=0 恰在航迹线上）→ 确定性右转（Rule 14）；侧偏反平行 ego 在 obs 左舷 → 右转
chk("N1e 正对头 β_me=0 → 右转", turning_direction(ego(0, 5), other(3000, 0, 180)), 1)
chk("N1f 侧偏反平行(obs 航迹左舷侧) → 右转", turning_direction(ego(0, 5), other(3000, 500, 180)), 1)
# N2 ahead_emergency（式6 两条：前扇区 ±45° ∧ 近反平行 ≤45°）
chk("N2 正前反向 → ahead", ahead_emergency(ego(0, 8), other(3000, 0, 180, 8)), True)
chk("N2b 正前同向（orientation 不反）→ ¬ahead", ahead_emergency(ego(0, 8), other(3000, 0, 0, 3)), False)
chk("N2c 侧后反向（出前扇区）→ ¬ahead", ahead_emergency(ego(0, 8), other(-3000, 0, 180, 8)), False)
chk("N2d 前方 β=46°（擦出扇区）→ ¬ahead",
    ahead_emergency(ego(0, 8), other(3000 * np.cos(46 * DEG), -3000 * np.sin(46 * DEG), 180, 8)), False)
# N3 stern 扇区（WD1 矢量修正版：正艉±110° ⟺ β∈[70°,290°]；±0.5° 余量避浮点端点）
for _bd, _exp in ((180.0, True), (0.0, False), (70.5, True), (69.5, False), (289.5, True), (290.5, False)):
    _los = -_bd * DEG
    chk(f"N3 stern 扇区 β={_bd}° → {_exp}",
        _in_stern_sector(ego(0, 5), other(3000 * np.cos(_los), 3000 * np.sin(_los), 0)), _exp)
# N4 _pos_under_stern_acc 解析手算（v0=5,a=0.048,t_react=60,v_max=9.5）
chk("N4 stern 位置 t=10（加速段）", round(float(_pos_under_stern_acc(ego(0, 5), 10, 0.048, 60, 9.5)[0]), 2), 52.40)
chk("N4b stern 位置 t=70（跨反应期）", round(float(_pos_under_stern_acc(ego(0, 5), 70, 0.048, 60, 9.5)[0]), 2), 465.20)
chk("N4c stern 位置 v0=9 饱和 t=20", round(float(_pos_under_stern_acc(ego(0, 9), 20, 0.048, 60, 9.5)[0]), 2), 187.40)
# N5 stern 链路（实测标定：dist=1500 恒速撞+加速解 → stern；1300 加速也不解 → ¬stern）
chk("N5 dist=1500 后方快船：恒速紧急 ∧ 加速能解 → stern_emergency",
    (is_emergency(ego(0, 4), other(-1500, 0, 0, 9)), is_emergency_under_acc(ego(0, 4), other(-1500, 0, 0, 9)),
     stern_emergency(ego(0, 4), other(-1500, 0, 0, 9))), (True, False, True))
chk("N5b dist=1300 加速也不解 → ¬stern（应走 base）",
    (is_emergency_under_acc(ego(0, 4), other(-1300, 0, 0, 9)), stern_emergency(ego(0, 4), other(-1300, 0, 0, 9))),
    (True, False))
# N6 get_target_ahead 手算：ego(0,0,朝东) vs 反向他船 → 反平行右转 → 目标 = 右舷 90°（正南）距 3·l_obs=525
_ta = get_target_ahead(ego(0, 8), other(2600, 0, 180, 8))
chk("N6 target_ahead = 右舷 525m", (round(float(_ta[0]), 6), round(float(_ta[1]), 1)), (0.0, -525.0))
# N7 get_target_base 手算：obs(1000,0,朝东,l=175) → 艉后轴线上，中心距 = 0.5·175+2·175=437.5 → (562.5,0)
_tb = get_target_base(ego(0, 5), other(1000, 0, 0, 5))
chk("N7 target_base = 艉后 437.5m", (round(float(_tb[0]), 1), round(float(_tb[1]), 6)), (562.5, 0.0))
# N7b 小他船 ego 尺度下限（B MAJOR-2）：l_obs=32 → 2.5·32=80 < 2.5·l_ego=437.5 → 取 437.5（>d_resolved=350）
_tbs = get_target_base(ego(0, 5), VesselState(position=np.array([1000.0, 0.0]), orientation=0.0,
                                              velocity=5.0, length=32.0))
chk("N7b 小他船(l=32) → 下限 2.5·l_ego=437.5", (round(float(_tbs[0]), 1), round(float(_tbs[1]), 6)), (562.5, 0.0))
# N8 tracking_controller（Appendix-C + WD5/WD6 守卫）
chk("N8 目标正前 → 直行加速", tracking_controller(ego(0, 5), [1000, 0]), (0.24, 0.0))
chk("N8b 目标右前 30° → 右转", tracking_controller(ego(0, 5), [866, -500])[1] < 0, True)
chk("N8c 目标正横右(cphi=0) → 右满舵+不加速", tracking_controller(ego(0, 5), [0, -500]), (0.0, -0.03))
chk("N8d 正横 cphi=+ε 浮点尾巴仍满舵（ω守卫修复回归）",
    (tracking_controller(ego(0, 5), [1e-13, -525]), tracking_controller(ego(0, 5), [-1e-13, -525])),
    ((0.0, -0.03), (0.0, -0.03)))
chk("N8e 目标正后 → 减速+确定右转掉头", tracking_controller(ego(0, 5), [-1000, 0]), (-0.24, -0.03))
chk("N8f 超 v_desired(v=8>6) 朝目标 → a 钳 ≤0", tracking_controller(ego(0, 8), [1000, 0])[0] <= 0.0, True)
chk("N8g v=0 对准 → 满加速起步", tracking_controller(ego(0, 0), [1000, 0]), (0.24, 0.0))
chk("N8h 已到目标 → (0,0)", tracking_controller(ego(0, 5), [0, 0]), (0.0, 0.0))
chk("N8j v=0 背对目标 → 只原地转不给油（起步门控，B MINOR-3）",
    tracking_controller(ego(0, 0), [-1000, 0]), (0.0, -0.03))
try:
    tracking_controller(ego(0, 5), [np.nan, 0]); _r = False
except ValueError:
    _r = True
chk("N8i NaN 目标 → fail-fast", _r, True)
# N9 EmergencyController（Alg.1 状态机：mode 选择 / u_acc 时序 / ahead→base 切换 / reset）
_ec = EmergencyController()
chk("N9 未激活 mode=None", _ec.mode, None)
_u0 = _ec.step(ego(0, 8), other(2600, 0, 180, 8))
chk("N9b ahead 场景首步 → mode=ahead 且立即右满舵（修复回归）", (_ec.mode, _u0), ("ahead", (0.0, -0.03)))
_ec.step(VesselState(position=np.array([600.0, 0.0]), orientation=0.0, velocity=8.0, length=L),
         other(2600, 0, 180, 8))
chk("N9c ahead 行驶距 600>525 → 切 base（Alg.1 line3-4）", _ec.mode, "base")
_ec.reset()
chk("N9d reset → None", _ec.mode, None)
_ec2 = EmergencyController()
_us = [_ec2.step(ego(0, 4), other(-1500, 0, 0, 9)) for _ in range(8)]
chk("N9e stern u_acc 时序 = 认证模型：t∈[0,60)=6步 [0.048,0]、第7步起 coast（off-by-one 修复回归）",
    (_ec2.mode, all(u == (0.048, 0.0) for u in _us[:6]), _us[6], _us[7]),
    ("stern", True, (0.0, 0.0), (0.0, 0.0)))
_ec3 = EmergencyController()
_ec3.step(ego(0, 4), other(-1300, 0, 0, 9))
chk("N9f 加速不解(1300) → mode=base 兜底", _ec3.mode, "base")
# N10 端到端闭环：ahead 对遇 2600m，EmergencyController+官方动力学 ≤40 步解除紧急
_vp = make_vessel_params()
_ecl = EmergencyController()
_st = np.array([0.0, 0.0, 0.0, 8.0])
_ob = other(2600, 0, 180, 8)
_resolved_at = -1
for _k in range(40):
    _se = VesselState(position=_st[:2].copy(), orientation=float(_st[2]), velocity=float(_st[3]), length=L)
    if _k > 0 and not is_emergency(_se, _ob):
        _resolved_at = _k
        break
    _a, _w = _ecl.step(_se, _ob)
    _st = dyn_step(_st, [_a, _w], DT, _vp, clip_velocity=True)
    _ob = VesselState(position=_ob.position + np.array([np.cos(_ob.orientation), np.sin(_ob.orientation)]) * _ob.velocity * DT,
                      orientation=_ob.orientation, velocity=_ob.velocity, length=_ob.length)
chk("N10 闭环 ahead 紧急 ≤40 步解除", 0 < _resolved_at <= 40, True)
# N11 fail-fast
try:
    EmergencyController(dt=0.0); _r = False
except ValueError:
    _r = True
chk("N11 dt≤0 → fail-fast", _r, True)
try:
    _pos_under_stern_acc(ego(0, 5), -1.0, 0.048, 60, 9.5); _r = False
except ValueError:
    _r = True
chk("N11b t<0 → fail-fast", _r, True)
try:
    get_target_ahead(ego(0, 8), other(2600, 0, 180, 8), d_ahead=0.0); _r = False
except ValueError:
    _r = True
chk("N11c d_ahead≤0 → fail-fast", _r, True)
try:
    get_target_base(ego(0, 5), other(1000, 0, 0, 5), d_behind=-1.0); _r = False
except ValueError:
    _r = True
chk("N11d d_behind≤0 → fail-fast", _r, True)
try:
    EmergencyController(dt=float("nan")); _r = False
except ValueError:
    _r = True
chk("N11e dt=NaN 穿不过守卫 → fail-fast（B MINOR-2）", _r, True)
try:
    _pos_under_stern_acc(ego(0, 5), 10.0, -0.1, 60, 9.5); _r = False
except ValueError:
    _r = True
chk("N11f a_stern<0（破坏弧长单调）→ fail-fast（B MINOR-1）", _r, True)
try:
    is_emergency_under_acc(ego(0, 4), other(-1500, 0, 0, 9), a_stern=float("nan")); _r = False
except ValueError:
    _r = True
chk("N11g a_stern=NaN → fail-fast 而非裸 GEOSException（B MINOR-1）", _r, True)

print("\n===== O) 组件4 SafeActionScheduler（2024 §V-C Theorem 2：ρ → As 调度）=====")
from trb_env.usv_colregs import A_REGULAR, SafeActionScheduler  # noqa: E402

# O1 跨模块 0 漂移：A_REGULAR == env 49 网格
from trb_env.usv_env import DISCRETE_ACTIONS as _ENV_GRID2, IDX_EMERGENCY as _IDX_EM, N_ACTIONS_TOTAL as _N_TOT  # noqa: E402
chk("O1 A_REGULAR == env DISCRETE_ACTIONS（49，防漂移）+ env 槽位常量",
    (A_REGULAR == _ENV_GRID2, len(A_REGULAR), _IDX_EM, _N_TOT), (True, 49, 49, 50))
# O2 无冲突 → (ρ0, A_regular 全集)
_sch = SafeActionScheduler()
_r, _as = _sch.step(ego(0, 5), other(0, 8000, 0, 5))
chk("O2 远离平行 → (ρ0, 全 49)", (_r, _as == set(A_REGULAR)), (RHO_NO_CONFLICT, True))
# O3 紧急（近距对遇 R1 立即真）→ (ρ5, {a_em})，a_em 有限 2 维
_sch3 = SafeActionScheduler()
_r3, _as3 = _sch3.step(ego(0, 8), other(2600, 0, 180, 8))
_aem = next(iter(_as3))
chk("O3 近距对遇 → (ρ5, 单元素 a_em 有限)",
    (_r3, len(_as3), len(_aem) == 2 and all(np.isfinite(x) for x in _aem)), (RHO_EMERGENCY, 1, True))
chk("O3b ρ5 内 emergency_mode 激活", _sch3.emergency_mode in ("ahead", "stern", "base"), True)
# O4 ρ5 **再入**边沿 reset：事件1 stern → 退出回 ρ0 → 事件2 ahead，mode 必须重评（守护"再入重评"；
#   ⚠️ C 复核 NIT-1：O4 只守护再入（恒 False=不重评则卡 stern 被本用例抓）；"驻留不 reset EC"
#   由 O9 守护（恒 True=每步 reset 能存活 O1-O8 却损坏 stern 刹车，C MINOR-1 已主窗口坐实）
_sch4 = SafeActionScheduler()
_sch4.step(ego(0, 4), other(-1500, 0, 0, 9))                       # 事件1：后方快船 → stern
_m1 = _sch4.emergency_mode
_sch4.step(ego(0, 5), other(-8000, 0, 180, 5))                     # 他船正后方驶离 → resolved → ρ0
_back = _sch4.rho
_sch4.step(ego(0, 8), other(2600, 0, 180, 8))                      # 事件2：正前对遇 → ahead
chk("O4 ρ5 边沿 reset：事件1 stern → ρ0 → 事件2 ahead（mode 重评）",
    (_m1, _back, _sch4.emergency_mode), ("stern", RHO_NO_CONFLICT, "ahead"))
# O5 对遇接近全程（I12 同款轨迹）：访问 ρ2 且 ρ2 步 As ⊆ ATR 非空；全程 As 是 set 无异常
_sch5 = SafeActionScheduler()
_ev = 9 * np.array([1.0, 0.0]); _ov = 9 * np.array([np.cos(180 * DEG), np.sin(180 * DEG)])
_seen = {}; _ok_types = True
for _k in range(36):                                                # 0..350s
    _t = _k * 10.0
    _el = VesselState(position=_ev * _t, orientation=0.0, velocity=9, length=L)
    _om = VesselState(position=np.array([12000.0, 0.0]) + _ov * _t, orientation=np.pi, velocity=9, length=L)
    _rr, _aa = _sch5.step(_el, _om)
    _ok_types = _ok_types and isinstance(_aa, set)
    _seen.setdefault(_rr, _aa)
chk("O5 对遇接近：访问 ρ2 且该步 As⊆ATR 非空 + 全程 set",
    (RHO_HEAD_ON in _seen, _seen.get(RHO_HEAD_ON, set()) != set()
     and set(_seen.get(RHO_HEAD_ON, set())) <= set(ATR), _ok_types), (True, True, True))
# O6 R1 抢占：临撞场景直接 ρ5 给 a_em（give-way ∅ 不可达——D12 推论）
_sch6 = SafeActionScheduler()
_r6, _as6 = _sch6.step(ego(0, 5), other(200, 0, 90, 0.0))
chk("O6 临撞 → R1 抢占 ρ5（非 give-way ∅）", (_r6, len(_as6)), (RHO_EMERGENCY, 1))
# O7 keep 即时支 → (ρ1, {akeep})：他船左舷交叉来船（本船 stand-on；实测 keep✓ cp✓ ¬is_em）
#   注：正后方同线追越场景会被 R1 可达圆盘抢占进 ρ5（D12 推论：ρ1-4 只存在于 cp 真但非 imminent 窗口）
_sch7 = SafeActionScheduler()
_r7, _as7 = _sch7.step(ego(0, 5), other(1800, 1800, -90, 5))
chk("O7 左舷交叉来船 → (ρ1 stand-on, {akeep})", (_r7, _as7), (RHO_STAND_ON, {(0.0, 0.0)}))
# O8 reset → ρ0 + mode 清
_sch3.reset()
chk("O8 reset → (ρ0, mode None)", (_sch3.rho, _sch3.emergency_mode), (RHO_NO_CONFLICT, None))
# O9 ρ5 **驻留**不 reset EC：连续 stern 驻留 9 步（90s>t_react=60），EC 计时 _t 连续递增 + u_acc 时序
#   （前 6 步加速 a_stern=0.048、第 7 步起刹车 0.0）。守护"驻留不 reset"——C 复核 MINOR-1：变异
#   `_prev_rho != RHO_EMERGENCY`→恒 True（每步 reset EC）存活 O1-O8 却使 _t 永卡 10、stern 永不刹车
#   （主窗口 /tmp 变异坐实）。本用例 + 变异验证锁定（L11/L12 范式：删守护逻辑→本用例翻 FAIL）。
_sch9 = SafeActionScheduler()
_ts9 = []; _us9 = []
for _k in range(9):
    _el9 = VesselState(position=np.array([40.0 * _k, 0.0]), orientation=0.0, velocity=4.0, length=L)
    _r9, _a9 = _sch9.step(_el9, other(-1500 + 90 * _k, 0, 0, 9))
    _ts9.append(round(_sch9._ec._t, 1)); _us9.append(round(next(iter(_a9))[0], 3))
chk("O9 ρ5 stern 驻留：mode 持稳 + _t 连续递增[10..90] + 前6步加速/第7步起刹车（守护驻留不 reset）",
    (_sch9.emergency_mode, _ts9, _us9[:6] == [0.048] * 6, _us9[6:] == [0.0] * 3),
    ("stern", [10.0 * (i + 1) for i in range(9)], True, True))

print("\n===== P) 组件(c) ViolationCounter（2024 §VII-A 违规计数，R_G3-6 MTL）=====")
from trb_env.usv_colregs import ViolationCounter, DELTA_NO_TURN  # noqa: E402

_OBS_KEEP = other(1800, 1800, -90, 5)      # 本船 stand-on 几何（keep True / crossing,head_on False）
_OBS_CROSS = other(1700, -1700, 90, 6)     # 本船 give-way crossing 几何（crossing True / keep,head_on False）
_OBS_OT = other(1000, 0, 0, 3)             # 本船 give-way overtake 几何（本船快追前船）
# P1 stand-on 保持不转 → 0 违规（R_G6 no_turning 满足）
_vc = ViolationCounter()
for _ in range(6):
    _vc.step(ego(0, 5), _OBS_KEEP)
chk("P1 stand-on 保持不转 → 0 违规", (_vc.standon_violations, _vc.giveway_violations), (0, 0))
# P2 stand-on 转向累积≥Δno_turn=10° → standon 违规（R_G6 违反，time-step 级）
_vc = ViolationCounter()
for _k in range(5):
    _vc.step(ego(-_k * 4, 5), _OBS_KEEP)   # 右转 4°/步，累积过 10°
chk("P2 stand-on 转向累积≥10° → standon 违规 ≥1（time-step 级）", _vc.standon_violations >= 1, True)
# P3 crossing give-way 右转累积≥Δlarge_turn=20° ∧ starboard → maneuver_done → 0 违规
_vc = ViolationCounter()
for _k in range(8):
    _vc.step(ego(-_k * 4, 6), _OBS_CROSS)
chk("P3 crossing 右转≥20°达标 → 0 giveway 违规", _vc.giveway_violations, 0)
# P4 crossing 不转 → 相遇解除时 giveway +1（R_G3 违反，encounter 级）
_vc = ViolationCounter()
for _ in range(6):
    _vc.step(ego(0, 6), _OBS_CROSS)
_vc.step(ego(0, 6), other(0, 9000, 90, 6))   # 他船驶远 → crossing 解除
chk("P4 crossing 不转→解除 → giveway +1（encounter 级）", _vc.giveway_violations, 1)
# P5 crossing 左转（累积≥20° 但非 starboard）→ maneuver_done=False → giveway +1（方向错）
_vc = ViolationCounter()
for _k in range(8):
    _vc.step(ego(_k * 4, 6), _OBS_CROSS)     # 左转
_vc.step(ego(28, 6), other(0, 9000, 90, 6))
chk("P5 crossing 左转(非 starboard)→ giveway +1（give-way 须右转）", _vc.giveway_violations, 1)
# P6 overtake 左转≥20° → maneuver_done（不要求 starboard）→ 0 违规
_vc = ViolationCounter()
for _k in range(8):
    _vc.step(ego(_k * 4, 8), _OBS_OT)
chk("P6 overtake 左转≥20° → 0 违规（overtake 不要求 starboard）", _vc.giveway_violations, 0)
# P7 finalize 结算未解除（active）give-way 相遇 → +1
_vc = ViolationCounter()
for _ in range(4):
    _vc.step(ego(0, 6), _OBS_CROSS)
_fin = _vc.finalize()
chk("P7 finalize 结算 active give-way 相遇 → +1", (_fin, _vc.giveway_violations), (1, 1))
# P8 reset → 清零
_vc.reset()
chk("P8 reset → 清零", (_vc.total, _vc._keep_active, _vc._gw), (0, False, {}))
# P9 真实轨迹端到端（head_on，本船 dyn_step 右转避让）：计数有限 + 无异常（head_on 几何敏感，须真实移动轨迹）
_vp = make_vessel_params()
_vc = ViolationCounter()
_st = np.array([0.0, 0.0, 0.0, 8.0])
_ob = other(3000, 0, 180, 8)
for _k in range(30):
    _e = VesselState(position=_st[:2].copy(), orientation=float(_st[2]), velocity=float(_st[3]), length=L)
    _vc.step(_e, _ob)
    _a, _w = (0.0, -0.03) if _k < 6 else (0.0, 0.0)   # 前段右转避让
    _st = dyn_step(_st, [_a, _w], DT, _vp, clip_velocity=True)
    _ob = VesselState(position=_ob.position + np.array([np.cos(_ob.orientation), np.sin(_ob.orientation)]) * _ob.velocity * DT,
                      orientation=_ob.orientation, velocity=_ob.velocity, length=L)
_vc.finalize()
chk("P9 真实轨迹 head_on 右转避让：计数有限 int ≥0 + 无异常（集成 smoke）",
    isinstance(_vc.total, int) and _vc.total >= 0, True)
# P10 Δno_turn 常量 = 10°（yaml max_orientation_diff_no_change）
chk("P10 Δno_turn = 10°（max_orientation_diff_no_change）", abs(DELTA_NO_TURN - 10.0 * DEG) < 1e-12, True)

# --- P11-14：注入态势序列隔离计数逻辑，守护 2 复核 agent + 主窗口发现的 P 段缺口（L9/L11/L12）---
import trb_env.usv_colregs as _ucol  # noqa: E402


def _vc_inject(situ_seqs, theta_deg_seq, finalize=True):
    """monkeypatch 态势谓词为按步序列 + 喂航向序列，跑 ViolationCounter（隔离计数逻辑，用后恢复）。"""
    _orig = {n: getattr(_ucol, n) for n in ("crossing", "head_on", "overtake", "keep")}
    _i = [0]
    try:
        for _n in ("crossing", "head_on", "overtake", "keep"):
            _seq = situ_seqs.get(_n, [False] * len(theta_deg_seq))
            setattr(_ucol, _n, (lambda s: (lambda e, o, t=None: s[_i[0]]))(_seq))
        _vci = ViolationCounter()
        for _k, _th in enumerate(theta_deg_seq):
            _i[0] = _k
            _vci.step(VesselState(position=np.array([0.0, 0.0]), orientation=_th * DEG, velocity=6.0, length=L), None)
        if finalize:
            _vci.finalize()
        return _vci
    finally:
        for _n, _f in _orig.items():
            setattr(_ucol, _n, _f)


# P11 telescoping（B2-MINOR-1 实质 + 主窗口）：give-way cum 是**带符号净航向变化**非路径长——overtake
#   右摆左摆净≈0 → 未达 maneuver → 违规；变异 cum+=abs(dtheta)（路径长 60°≥20°）会误判 done→0 漏计。
_vc = _vc_inject({"overtake": [True] * 5 + [False]}, [0, -15, 0, -15, 0, 0])
chk("P11 telescoping：overtake 右摆左摆净≈0 → giveway +1（守护 cum 带符号累积非路径长）",
    _vc.giveway_violations, 1)
# P12 keep 闪烁（B2-MINOR-2 纠误诊后）：每 keep spell 转<10° → standon=0；守护 keep cum 跨 spell 重置
#   （主窗口实测 onset+¬keep 双重重置冗余，单删=死代码，MUT-both 两个都删才跨 spell 累加误报→本用例翻）。
_vc = _vc_inject({"keep": [True, True, False, True, True, False, True, True]},
                 [0, -4, -4, -8, -12, -12, -16, -20])
chk("P12 keep 闪烁每 spell<10° → standon=0（守护跨 spell 重置，纠 B2 单删误诊）", _vc.standon_violations, 0)
# P13 首步即 keep + 转向（A2-MINOR-3）：首步 _prev_theta=None → dtheta=0 不计（匹配 commonocean
#   time_step-1 not in → no_turning True），累积从第 2 步起 → standon=2（非 3）。
_vc = _vc_inject({"keep": [True, True, True]}, [0, -15, -30], finalize=False)
chk("P13 首步即 keep+转向：首步豁免(dtheta=0) → standon=2（累积从第2步起）", _vc.standon_violations, 2)
# P14 give-way cum 符号约定（B2-MINOR-1）：单调右转 → 内部 cum>0（锁符号约定，防未来重构用 cum 符号判方向；
#   当前 cum 符号是功能性死代码——方向由独立 wrap(cur−onset)<0 判，故锁约定不锁行为）。
_vc = _vc_inject({"crossing": [True, True, True]}, [0, -8, -16], finalize=False)
chk("P14 give-way 单调右转 → 内部 cum>0（符号约定白盒断言，B2-MINOR-1 死代码防御）",
    _vc._gw.get("crossing", {}).get("cum", 0.0) > 0.0, True)

# P15/P16 D3-4（深度审核 L38）：累积航向 cum=Σwrap_to_pi(prev−θ) 恰达阈值时浮点比解析少 ~1 ULP（4×5°=
#   19.9999..°<20°，差 3.9e-16 rad）。盾给路投影恰输出 ω=−ω_turn（净恰 20°）、stand-on 带边沿恰 10° = 最易命中。
#   _VIOL_CUM_TOL=1e-9 容差使「数学上恰达阈值」被正确判达标（give-way 不虚增违规 / stand-on 不漏计违规）。
#   ⭐ 用【真实 crossing()/keep() 几何 + 精确 rad 朝向】(非 _vc_inject monkeypatch，整数度不复现 shortfall)，
#   并【内置变异守护】：临时把 _VIOL_CUM_TOL 置 0（=删容差）→ 同序列翻成 bug 行为，证容差 load-bearing。
_CROSS_OBS = VesselState(position=np.array([1500.0, -1500.0]), orientation=90.0 * DEG, velocity=5.0, length=120.0)
_OFF_OBS = VesselState(position=np.array([-3000.0, 3000.0]), orientation=90.0 * DEG, velocity=5.0, length=120.0)
_KEEP_OBS = VesselState(position=np.array([1500.0, 1500.0]), orientation=-90.0 * DEG, velocity=5.0, length=120.0)


def _run_giveway_net20(tol):
    """give-way 给路恰净 20°（5 步右转 + 1 步 offset），返回 giveway_violations（tol 可控以做变异守护）。"""
    _old = _ucol._VIOL_CUM_TOL
    _ucol._VIOL_CUM_TOL = tol
    try:
        _s = DELTA_LARGE_TURN / 4.0          # 5°/步（rad）→ 4 增量恰净 20°
        _vci = ViolationCounter()
        for _k in range(5):                  # k=0..4：crossing 持续真、cum 在 k=4 恰达 20°
            _vci.step(VesselState(position=np.array([0.0, 0.0]), orientation=float(-_s * _k), velocity=5.0, length=L), _CROSS_OBS)
        _vci.step(VesselState(position=np.array([0.0, 0.0]), orientation=float(-_s * 4), velocity=5.0, length=L), _OFF_OBS)  # offset
        return _vci.giveway_violations
    finally:
        _ucol._VIOL_CUM_TOL = _old


def _run_standon_net10(tol):
    """stand-on 直航船恰净 10°（5 步带边沿转），返回 standon_violations。"""
    _old = _ucol._VIOL_CUM_TOL
    _ucol._VIOL_CUM_TOL = tol
    try:
        _s = DELTA_NO_TURN / 4.0             # 2.5°/步（rad）→ 4 增量恰净 10°
        _vci = ViolationCounter()
        for _k in range(5):
            _vci.step(VesselState(position=np.array([0.0, 0.0]), orientation=float(-_s * _k), velocity=5.0, length=L), _KEEP_OBS)
        return _vci.standon_violations
    finally:
        _ucol._VIOL_CUM_TOL = _old


def _run_giveway_net(net_deg, tol):
    """give-way 给路净 net_deg 度（5 步右转 + 1 步 offset），返回 giveway_violations。
    用于【上界容差守护】：真欠转 net<20° 应计违规、容差过大会误吃掉它。net 离阈值远（0.5°）→不依赖 ULP 精度。"""
    _old = _ucol._VIOL_CUM_TOL
    _ucol._VIOL_CUM_TOL = tol
    try:
        _s = (net_deg * DEG) / 4.0
        _vci = ViolationCounter()
        for _k in range(5):
            _vci.step(VesselState(position=np.array([0.0, 0.0]), orientation=float(-_s * _k), velocity=5.0, length=L), _CROSS_OBS)
        _vci.step(VesselState(position=np.array([0.0, 0.0]), orientation=float(-_s * 4), velocity=5.0, length=L), _OFF_OBS)
        return _vci.giveway_violations
    finally:
        _ucol._VIOL_CUM_TOL = _old


def _run_standon_net(net_deg, tol):
    """stand-on 直航船净 net_deg 度（5 步带边沿转），返回 standon_violations。
    用于【上界容差守护】：真守向 net<10° 不应计违规、容差过大会虚增。"""
    _old = _ucol._VIOL_CUM_TOL
    _ucol._VIOL_CUM_TOL = tol
    try:
        _s = (net_deg * DEG) / 4.0
        _vci = ViolationCounter()
        for _k in range(5):
            _vci.step(VesselState(position=np.array([0.0, 0.0]), orientation=float(-_s * _k), velocity=5.0, length=L), _KEEP_OBS)
        return _vci.standon_violations
    finally:
        _ucol._VIOL_CUM_TOL = _old


chk("P15 D3-4 give-way 恰净20°：容差→不虚增违规（giveway_violations=0）", _run_giveway_net20(_ucol._VIOL_CUM_TOL), 0)
chk("P15b D3-4 变异守护：删容差(tol=0)→恰20°被1ULP误判未达标→虚增违规(=1)", _run_giveway_net20(0.0), 1)
chk("P16 D3-4 stand-on 恰净10°：容差→正确计违规（standon_violations=1，不漏计）", _run_standon_net10(_ucol._VIOL_CUM_TOL), 1)
chk("P16b D3-4 变异守护：删容差(tol=0)→恰10°被1ULP漏计→漏报违规(=0)", _run_standon_net10(0.0), 0)
# P15c/P16c D3-4（交接审核 2026-06-17b 补：双边夹逼，堵"容差过大"覆盖缺口）：P15/P15b/P16/P16b 只守容差过小(=0)；
#   若误把 _VIOL_CUM_TOL 调大(>~0.5°)会静默吃真违规/虚增违规、而旧测试全绿。下面用【真欠转 19.5°】【真守向 9.5°】
#   在真实容差下断言正确计数(=上界守护，源容差调大则 P15c/P16c 翻 FAIL) + 过大容差(1.0°)变异坐实 load-bearing。
chk("P15c D3-4 上界守护 give-way 真欠转19.5°：真实容差下正确计违规(=1)", _run_giveway_net(19.5, _ucol._VIOL_CUM_TOL), 1)
chk("P15d D3-4 变异守护：容差过大(1.0°)→真欠转19.5°被误判达标→漏计违规(=0)", _run_giveway_net(19.5, 1.0 * DEG), 0)
chk("P16c D3-4 上界守护 stand-on 真守向9.5°：真实容差下不计违规(=0)", _run_standon_net(9.5, _ucol._VIOL_CUM_TOL), 0)
chk("P16d D3-4 变异守护：容差过大(1.0°)→真守向9.5°被虚增违规(=1)", _run_standon_net(9.5, 1.0 * DEG), 1)

print()
if _fail == 0:
    print(f"✅ 全部 PASS ({_n}/{_n})")
else:
    print(f"❌ {_fail}/{_n} FAIL")
    sys.exit(1)
