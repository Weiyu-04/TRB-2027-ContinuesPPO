"""
Node 1 冒烟测试：连续 COLREGs 投影层（U_box ∩ U_colregs 解析逐轴投影）。
跑：/opt/miniconda3/envs/trb/bin/python -B 代码/tests/test_usv_projection.py
（项目脚本风格：模块级断言 + ok() 计数 + sys.exit；非 pytest。）

策略：桩状态机固定 ρ，解耦"投影逻辑"测试 vs 已验证的"状态机几何分类"（test_usv_colregs 已覆盖）。
覆盖：常量回归（守护复审 catch = T_M=40 非 T_MANEUVER=70）/ 各 ρ 约束 / 方向单一真相源 / box / 守卫。
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from trb_env.usv_colregs import (
    VesselState, get_turning_act,
    DELTA_LARGE_TURN, DELTA_NO_TURN, T_M,
    RHO_NO_CONFLICT, RHO_STAND_ON, RHO_HEAD_ON, RHO_CROSSING, RHO_OVERTAKE, RHO_EMERGENCY,
    predict_state_cv, _vessel_circumradius, DOBS_SAFETY_FACTOR,
)
from trb_env.usv_dynamics import make_vessel_params
from trb_env.usv_projection import (
    ContinuousColregsProjection,
    DEFAULT_OMEGA_TURN, DEFAULT_EPS_OMEGA, DEFAULT_EPS_A,
    obstacle_occupancy_disk, ego_next_position, position_jacobian, ego_circumradius,
    collision_free_constraint,
)

A_MAX, W_MAX = 0.24, 0.03
_fail = 0
_total = 0


def ok(name, cond):
    global _fail, _total
    _total += 1
    if not cond:
        _fail += 1
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")


def approx(a, b, tol=1e-9):
    return abs(float(a) - float(b)) <= tol


def raises(fn):
    try:
        fn(); return False
    except (ValueError, RuntimeError):
        return True


class _StubChart:
    """固定 ρ 的桩状态机（解耦投影逻辑 vs 已验证的状态机几何分类）。"""
    def __init__(self, rho): self._rho = rho; self.rho = rho
    def step(self, s_ego, s_obs): return self._rho
    def reset(self): self.rho = self._rho


def proj(rho):
    return ContinuousColregsProjection(A_MAX, W_MAX, statechart=_StubChart(rho))


def vs(orient=0.0, pos=(0.0, 0.0), v=5.0, length=175.0):
    return VesselState(position=np.array(pos, dtype=float), orientation=orient, velocity=v, length=length)


EGO, OBS = vs(), vs(pos=(1000.0, 0.0))

# ① 常量回归（守护复审 catch：必须用 T_M=40，不是 T_MANEUVER=70）
ok("① T_M == 40", T_M == 40.0)
ok("① ω_turn = Δlarge/40 ≈0.008727", approx(DEFAULT_OMEGA_TURN, np.radians(20.0) / 40.0))
ok("① ω_turn ≠ 误用 T_MANEUVER=70 的值(0.004986)", abs(DEFAULT_OMEGA_TURN - np.radians(20.0) / 70.0) > 1e-4)
ok("① ε_ω = Δno/40 ≈0.004363", approx(DEFAULT_EPS_OMEGA, np.radians(10.0) / 40.0))
ok("① ε_a = 0.016", DEFAULT_EPS_A == 0.016)
ok("① 阈值均 ∈ (0, w_max]", 0.0 < DEFAULT_OMEGA_TURN <= W_MAX and 0.0 < DEFAULT_EPS_OMEGA <= W_MAX)

# ② ρ0 无约束：原样穿过
r = proj(RHO_NO_CONFLICT).project(EGO, OBS, np.array([0.2, 0.025]))
ok("② ρ0 原样穿过", np.allclose(r.u_safe, [0.2, 0.025]) and not r.corrected and not r.needs_fallback)

# ③ ρ1 stand-on：|ω|≤ε_ω ∧ |a|≤ε_a
r = proj(RHO_STAND_ON).project(EGO, OBS, np.array([0.2, 0.025]))
ok("③ ρ1 a→+ε_a", approx(r.u_safe[0], DEFAULT_EPS_A))
ok("③ ρ1 ω→+ε_ω", approx(r.u_safe[1], DEFAULT_EPS_OMEGA))
r = proj(RHO_STAND_ON).project(EGO, OBS, np.array([0.01, 0.001]))
ok("③ ρ1 带内不改", np.allclose(r.u_safe, [0.01, 0.001]) and not r.corrected)

# ④ ρ2/ρ3 head_on/crossing：ω≤−ω_turn（右转），a 自由
for rho, nm in [(RHO_HEAD_ON, "head_on"), (RHO_CROSSING, "crossing")]:
    r = proj(rho).project(EGO, OBS, np.array([0.1, 0.02]))     # 想左转+a=0.1
    ok(f"④ ρ({nm}) ω→−ω_turn", approx(r.u_safe[1], -DEFAULT_OMEGA_TURN))
    ok(f"④ ρ({nm}) a 自由不改 + dir=right", approx(r.u_safe[0], 0.1) and r.give_way_dir == "right" and r.corrected)
    r = proj(rho).project(EGO, OBS, np.array([0.0, -0.02]))    # 已右转 ∈ box
    ok(f"④ ρ({nm}) 合规穿过", np.allclose(r.u_safe, [0.0, -0.02]) and not r.corrected)

# ⑤ ρ4 overtake：方向由 get_turning_act 定（单一真相源）
ego, obs = vs(orient=0.0), vs(orient=-0.5, pos=(1000.0, 0.0))  # 他船更右(rel<0)→ATL(左)
ok("⑤ 他船更右→ATL(左)", float(next(iter(get_turning_act(ego, obs)))[1]) > 0.0)
r = ContinuousColregsProjection(A_MAX, W_MAX, statechart=_StubChart(RHO_OVERTAKE)).project(ego, obs, np.array([0.0, -0.02]))
ok("⑤ overtake 左转 ω→+ω_turn + dir=left", approx(r.u_safe[1], DEFAULT_OMEGA_TURN) and r.give_way_dir == "left")
ego2, obs2 = vs(orient=0.0), vs(orient=0.5, pos=(1000.0, 0.0))  # 他船更左(rel>0)→ATR(右)
ok("⑤ 他船更左→ATR(右)", float(next(iter(get_turning_act(ego2, obs2)))[1]) < 0.0)
r = ContinuousColregsProjection(A_MAX, W_MAX, statechart=_StubChart(RHO_OVERTAKE)).project(ego2, obs2, np.array([0.0, 0.02]))
ok("⑤ overtake 右转 ω→−ω_turn + dir=right", approx(r.u_safe[1], -DEFAULT_OMEGA_TURN) and r.give_way_dir == "right")

# ⑥ ρ5 emergency：needs_fallback + 仅 box 投影占位（不静默）
r = proj(RHO_EMERGENCY).project(EGO, OBS, np.array([0.5, 0.05]))  # 超 box
ok("⑥ ρ5 needs_fallback + box 裁剪", r.needs_fallback and approx(r.u_safe[0], A_MAX) and approx(r.u_safe[1], W_MAX))

# ⑦ u_safe 恒 ∈ box（逐轴 clip = 精确投影）
allbox = True
for rho in (RHO_NO_CONFLICT, RHO_STAND_ON, RHO_HEAD_ON, RHO_CROSSING, RHO_OVERTAKE, RHO_EMERGENCY):
    for u in (np.array([1.0, 1.0]), np.array([-1.0, -1.0]), np.array([0.0, 0.0])):
        rr = proj(rho).project(EGO, OBS, u)
        if not (-A_MAX - 1e-12 <= rr.u_safe[0] <= A_MAX + 1e-12 and -W_MAX - 1e-12 <= rr.u_safe[1] <= W_MAX + 1e-12):
            allbox = False
ok("⑦ u_safe 恒 ∈ box", allbox)

# ⑧ 输入/参数守卫
p0 = proj(RHO_NO_CONFLICT)
ok("⑧ u 维度错→raise", raises(lambda: p0.project(EGO, OBS, np.array([0.1]))))
ok("⑧ u 非有限→raise", raises(lambda: p0.project(EGO, OBS, np.array([np.nan, 0.0]))))
ok("⑧ a_max≤0→raise", raises(lambda: ContinuousColregsProjection(-0.1, 0.03)))
ok("⑧ ω_turn>w_max→raise(不可行)", raises(lambda: ContinuousColregsProjection(0.24, 0.03, omega_turn=0.05)))
ok("⑧ eps_a>a_max→raise", raises(lambda: ContinuousColregsProjection(0.24, 0.03, eps_a=0.5)))

# ⑨ 真实状态机集成（补对抗复核 minor：原 27 项全用桩、真 ColregsStatechart + reset 路径零覆盖）
real = ContinuousColregsProjection(A_MAX, W_MAX)          # 默认 = 真实 ColregsStatechart
far = vs(orient=np.pi, pos=(8000.0, 8000.0), v=3.0)       # 远距(11314m>d_sense)+背离 → 无冲突 ρ0
real.reset()
r = real.project(EGO, far, np.array([0.1, 0.02]))
ok("⑨ 真实状态机 ρ0(远距无冲突)→原样穿过", r.rho == RHO_NO_CONFLICT and np.allclose(r.u_safe, [0.1, 0.02]) and not r.corrected)
real.reset()
ok("⑨ reset 后真实状态机回 ρ0", real.rho == RHO_NO_CONFLICT)

# ⑩ corrected 精确性守护（修复对抗复核 minor：旧 atol=1e-12 会把 <1e-12 修正误报 False）
r = proj(RHO_HEAD_ON).project(EGO, OBS, np.array([0.0, -DEFAULT_OMEGA_TURN + 5e-13]))  # 仅超界 5e-13(<旧容差)
ok("⑩ 极小修正(5e-13)也标 corrected=True", r.corrected and approx(r.u_safe[1], -DEFAULT_OMEGA_TURN))
r = proj(RHO_NO_CONFLICT).project(EGO, OBS, np.array([0.1, 0.02]))                      # 未改动 → corrected=False
ok("⑩ 未改动→corrected=False", not r.corrected)

# ⑪ Node 2a 原料：他船预测占据圆 / 本船一步可达位置 / 位置 Jacobian（含已知答案校验）
VP = make_vessel_params()
DT2 = 10.0
obs_mv = vs(orient=0.0, pos=(1000.0, 0.0), v=3.0)        # 东行 v=3
c, R = obstacle_occupancy_disk(obs_mv, 10.0)
ok("⑪ O_obs center = 恒速预测 (1030,0)", np.allclose(c, [1030.0, 0.0]))
ok("⑪ O_obs radius = circumradius+2·l_obs", approx(R, _vessel_circumradius(175.0) + DOBS_SAFETY_FACTOR * 175.0, tol=1e-6))
ok("⑪ ego_circumradius = circ(175,25.4)", approx(ego_circumradius(EGO), _vessel_circumradius(175.0, 25.4)))
p = ego_next_position(EGO, [0.0, 0.0], DT2, VP)          # 直行 v=5 dt=10 → +50m 东
ok("⑪ ego_next 直行→(50,0)", np.allclose(p, [50.0, 0.0], atol=1e-3))
J = position_jacobian(EGO, [0.0, 0.0], DT2, VP)          # 已知答案 [[50,0],[0,250]]
ok("⑪ Jacobian 已知答案 [[50,0],[0,250]]", np.allclose(J, [[50.0, 0.0], [0.0, 250.0]], atol=1e-2))
J2 = position_jacobian(EGO, [0.0, 0.0], DT2, VP, h=1e-4)
ok("⑪ Jacobian FD h-稳定(1e-5 vs 1e-4)", np.allclose(J, J2, atol=1e-3))
ok("⑪ tau<0→raise", raises(lambda: obstacle_occupancy_disk(OBS, -1.0)))
ok("⑪ u 维度错→raise", raises(lambda: ego_next_position(EGO, [0.0], DT2, VP)))
ok("⑪ h≤0→raise", raises(lambda: position_jacobian(EGO, [0.0, 0.0], DT2, VP, h=0.0)))

# ⑫ Node 2b：分离超平面 + 标量一阶 → 线性无碰撞约束 g·u ≤ h
egoH = vs(orient=0.0, pos=(0.0, 0.0), v=5.0)             # 本船东行 v=5
d_safe_exp = ego_circumradius(egoH) + (_vessel_circumradius(175.0) + DOBS_SAFETY_FACTOR * 175.0)
far = vs(orient=np.pi, pos=(5000.0, 0.0), v=3.0)        # 远距迎面
g, h, dist, dsafe = collision_free_constraint(egoH, far, [0.0, 0.0], 10.0, 10.0, VP)
ok("⑫ d_safe = R_ego+R_obs（标源）", approx(dsafe, d_safe_exp, tol=1e-6))
ok("⑫ 远距→不 binding（u_nom 满足 g·u_nom≤h, h>0）", float(g @ np.array([0.0, 0.0])) <= h and h > 0.0)
appr = vs(orient=np.pi, pos=(600.0, 0.0), v=5.0)        # 迎面接近
g2, h2, dist2, dsafe2 = collision_free_constraint(egoH, appr, [0.0, 0.0], 10.0, 10.0, VP)
ok("⑫ 迎面→binding（u_nom 违反 g·u_nom>h）", float(g2 @ np.array([0.0, 0.0])) > h2)
ok("⑫ h = dist−d_safe（u_nom=0 线性化精确）", approx(h2, dist2 - dsafe2, tol=1e-6))
ok("⑫ 迎面约束 ∂(g·u)/∂a>0（减速=对的方向）", float(g2[0]) > 0.0)
deg = vs(orient=np.pi, pos=(80.0, 0.0), v=3.0)          # 预测10s→[50,0]=本船标称位
gd, hd, distd, _ = collision_free_constraint(egoH, deg, [0.0, 0.0], 10.0, 10.0, VP)
ok("⑫ 退化(圆心重合)→(None,None) 交兜底", gd is None and hd is None and distd < 1e-9)

# ⑬ 对抗复核补漏：方向退化守卫(MAJOR) / Jacobian 非轴对齐防转置 / 安全裕度常量锁
ego_static = vs(orient=0.0, pos=(0.0, 0.0), v=0.0)      # 本船静止朝东
obs_north = vs(orient=0.0, pos=(0.0, 400.0), v=0.0)     # 他船正北 400m(<d_safe) 静止 → 无动作能拉开南北距
gN, hN, distN, dsN = collision_free_constraint(ego_static, obs_north, [0.0, 0.0], 10.0, 10.0, VP)
ok("⑬ 方向退化(v=0,他船正北)+binding → 交兜底(None)", gN is None and hN is None and distN < dsN)
obs_far_n = vs(orient=0.0, pos=(0.0, 5000.0), v=0.0)    # 同退化方向但远(>d_safe) → 非兜底(平凡非binding)
gF, hF, distF, dsF = collision_free_constraint(ego_static, obs_far_n, [0.0, 0.0], 10.0, 10.0, VP)
ok("⑬ 方向退化但远(not-binding) → 非兜底", gF is not None and distF >= dsF)
c45 = float(np.cos(np.pi / 4.0))                        # sin(π/4)=cos(π/4)
J45 = position_jacobian(vs(orient=np.pi / 4.0, v=5.0), [0.0, 0.0], 10.0, VP)
ok("⑬ Jacobian orient=π/4 = [[50c,-250c],[50c,250c]]（防转置）",
   np.allclose(J45, [[50 * c45, -250 * c45], [50 * c45, 250 * c45]], atol=1e-2))
ok("⑬ J(π/4) ≠ 其转置（该测试真能抓转置 bug）", not np.allclose(J45, J45.T, atol=1.0))
ok("⑬ DOBS_SAFETY_FACTOR == 2.0 字面锁（钉死 dobs,safety=2·l_obs）", DOBS_SAFETY_FACTOR == 2.0)

# ⑭ 复审守卫（CODE-2 u_nom 越界 / CODE-3 tau≠dt，2026-06-15c）——堵 Node 2c 跨节点祸患
ok("⑭ CODE-3 tau≠dt → raise", raises(lambda: collision_free_constraint(EGO, OBS, [0.0, 0.0], 5.0, 10.0, VP)))
ok("⑭ CODE-3 tau==dt 不 raise", not raises(lambda: collision_free_constraint(EGO, OBS, [0.0, 0.0], 10.0, 10.0, VP)))
ok("⑭ CODE-2 u_nom a 越界(0.3) → raise", raises(lambda: collision_free_constraint(EGO, OBS, [0.3, 0.0], 10.0, 10.0, VP)))
ok("⑭ CODE-2 u_nom ω 越界(0.05) → raise", raises(lambda: collision_free_constraint(EGO, OBS, [0.0, 0.05], 10.0, 10.0, VP)))
ok("⑭ CODE-2 u_nom 边界内(0.24,0.03) 不 raise", not raises(lambda: collision_free_constraint(EGO, OBS, [0.24, 0.03], 10.0, 10.0, VP)))

# ⑮ Node 2c：QP 集成（U_box∩U_colregs∩U_collision-free + P=∅→兜底 + 对 Node 1 等价；几何经实跑标定）
egoE = vs(orient=0.0, pos=(0.0, 0.0), v=5.0)              # 本船东行 v=5

# ⑮a 远距(非 binding)：QP == Node 1 逐轴 clip（等价核验）+ 非兜底
far_qp = vs(orient=np.pi, pos=(6000.0, 0.0), v=3.0)
n1 = proj(RHO_CROSSING).project(egoE, far_qp, np.array([0.1, 0.02]))
qp = proj(RHO_CROSSING).project_qp(egoE, far_qp, np.array([0.1, 0.02]), DT2, VP)
ok("⑮a 远距非binding: QP==Node1 clip + 非兜底", np.allclose(qp.u_safe, n1.u_safe, atol=1e-5) and not qp.needs_fallback)

# ⑮b 迎面@650m(feasible-binding)：非兜底 + corrected + 解∈box∩colregs + 满足无碰撞约束 + 真被 collision 移动
b = vs(orient=np.pi, pos=(650.0, 0.0), v=5.0)
rb = proj(RHO_CROSSING).project_qp(egoE, b, np.array([0.2, 0.0]), DT2, VP)
u_box_b = np.array([float(np.clip(0.2, -A_MAX, A_MAX)), float(np.clip(0.0, -W_MAX, -DEFAULT_OMEGA_TURN))])  # =project_qp 内部 u_box (Node1 clip)
gb, hb, _, _ = collision_free_constraint(egoE, b, u_box_b, DT2, DT2, VP)
ok("⑮b binding: 非兜底 + corrected", (not rb.needs_fallback) and rb.corrected)
ok("⑮b binding: 解 ∈ box∩colregs(ω≤−ω_turn, |a|≤a_max)",
   rb.u_safe[1] <= -DEFAULT_OMEGA_TURN + 1e-6 and -A_MAX - 1e-6 <= rb.u_safe[0] <= A_MAX + 1e-6)
ok("⑮b binding: 解满足无碰撞约束 g·u≤h + 真被 collision 移动(≠纯colregs clip)",
   float(gb @ rb.u_safe) <= hb + 1e-4 and not np.allclose(rb.u_safe, u_box_b, atol=1e-3))

# ⑮c 深入@500m(P=∅)：合规∩无碰撞冲突 → needs_fallback（绝不静默放行）
rc = proj(RHO_CROSSING).project_qp(egoE, vs(orient=np.pi, pos=(500.0, 0.0), v=5.0), np.array([0.2, 0.0]), DT2, VP)
ok("⑮c 深入 P=∅ → needs_fallback(交 Node4)", rc.needs_fallback)

# ⑮d ρ5 紧急 → needs_fallback（不做无碰撞 QP）
rd = proj(RHO_EMERGENCY).project_qp(egoE, vs(orient=np.pi, pos=(650.0, 0.0), v=5.0), np.array([0.1, 0.0]), DT2, VP)
ok("⑮d ρ5 → needs_fallback", rd.needs_fallback)

# ⑮e 退化(圆心重合 dist≈0) → needs_fallback
re = proj(RHO_NO_CONFLICT).project_qp(egoE, vs(orient=np.pi, pos=(80.0, 0.0), v=3.0), np.array([0.0, 0.0]), DT2, VP)
ok("⑮e 退化(圆心重合) → needs_fallback", re.needs_fallback)

# ⑮f ρ0 无冲突 + 远距：原样穿过（非兜底、非 corrected）
rf = proj(RHO_NO_CONFLICT).project_qp(egoE, far_qp, np.array([0.1, 0.01]), DT2, VP)
ok("⑮f ρ0 远距 QP 原样穿过", (not rf.needs_fallback) and (not rf.corrected) and np.allclose(rf.u_safe, [0.1, 0.01], atol=1e-5))

# ⑯ Node 2c 对抗复核守卫（B1 taus=[] 静默放行 / B3 dt≤0，2026-06-15d）
ok("⑯ B1 taus=[] → raise(防静默跳过无碰撞约束)",
   raises(lambda: proj(RHO_CROSSING).project_qp(egoE, vs(orient=np.pi, pos=(300.0, 0.0), v=5.0), np.array([0.2, 0.0]), DT2, VP, taus=[])))
ok("⑯ B1 深入@300m 默认(taus=None) 仍 needs_fallback(对照 B1 守卫前会静默放行)",
   proj(RHO_CROSSING).project_qp(egoE, vs(orient=np.pi, pos=(300.0, 0.0), v=5.0), np.array([0.2, 0.0]), DT2, VP).needs_fallback)
ok("⑯ B3 dt=0 → raise", raises(lambda: proj(RHO_CROSSING).project_qp(egoE, far_qp, np.array([0.1, 0.0]), 0.0, VP)))
ok("⑯ B3 dt<0 → raise", raises(lambda: proj(RHO_CROSSING).project_qp(egoE, far_qp, np.array([0.1, 0.0]), -1.0, VP)))

# ⑰ Node 4：safe_action 兜底编排（projection / emergency / relaxed / collision_min / degenerate + EC 生命周期；几何经实跑标定）
def in_box(u):
    return -A_MAX - 1e-6 <= u[0] <= A_MAX + 1e-6 and -W_MAX - 1e-6 <= u[1] <= W_MAX + 1e-6

# ⑰a 远距 → projection（Node 2c 投影动作）
ra = proj(RHO_CROSSING).safe_action(egoE, far_qp, np.array([0.1, 0.02]), DT2, VP)
ok("⑰a 远距 → source=projection + ∈box", ra.source == "projection" and in_box(ra.u_safe))

# ⑰b 侧方@285°/580m crossing → relaxed（放松 COLREGs 保无碰撞：非合规右转 + 满足无碰撞约束）
obs_relax = vs(orient=np.pi, pos=(580 * np.cos(np.radians(285)), 580 * np.sin(np.radians(285))), v=5.0)
rb = proj(RHO_CROSSING).safe_action(egoE, obs_relax, np.array([0.2, 0.0]), DT2, VP)
gR, hR, _, _ = collision_free_constraint(egoE, obs_relax, np.array([0.2, 0.0]), DT2, DT2, VP)  # u_box_full=clip([0.2,0])=[0.2,0]
ok("⑰b relaxed: source=relaxed + ω 非合规(>−ω_turn) + 满足无碰撞 g·u≤h + ∈box",
   rb.source == "relaxed" and rb.u_safe[1] > -DEFAULT_OMEGA_TURN and float(gR @ rb.u_safe) <= hR + 1e-4 and in_box(rb.u_safe))

# ⑰c 深入@600m crossing(P=∅ 且 relax 也不可行) → collision_min（碰撞风险最小化 best-effort）
rc = proj(RHO_CROSSING).safe_action(egoE, vs(orient=np.pi, pos=(600.0, 0.0), v=5.0), np.array([0.2, 0.0]), DT2, VP)
ok("⑰c 深入 → source=collision_min + ∈box + 有限", rc.source == "collision_min" and in_box(rc.u_safe) and np.all(np.isfinite(rc.u_safe)))

# ⑰d ρ5 紧急 → emergency（复用 Alg.1 EmergencyController，mode 立、动作有限）
rd = proj(RHO_EMERGENCY).safe_action(egoE, vs(orient=np.pi, pos=(400.0, 0.0), v=8.0), np.array([0.1, 0.0]), DT2, VP)
ok("⑰d ρ5 → source=emergency + mode∈{ahead,stern,base} + 有限",
   rd.source == "emergency" and rd.emergency_mode in ("ahead", "stern", "base") and np.all(np.isfinite(rd.u_safe)))

# ⑰e 圆心重合 → degenerate
re_ = proj(RHO_NO_CONFLICT).safe_action(egoE, vs(orient=np.pi, pos=(80.0, 0.0), v=3.0), np.array([0.0, 0.0]), DT2, VP)
ok("⑰e 圆心重合 → source=degenerate", re_.source == "degenerate")

# ⑰f EC 生命周期: ρ0→ρ5 进入边沿(创建+reset、mode 立)→ρ5 驻留(不 reset、mode 续)；reset() 后再进=新事件
scL = _StubChart(RHO_NO_CONFLICT)
shL = ContinuousColregsProjection(A_MAX, W_MAX, statechart=scL)
r0 = shL.safe_action(egoE, far_qp, np.array([0.1, 0.0]), DT2, VP)
scL._rho = RHO_EMERGENCY
r1 = shL.safe_action(egoE, vs(orient=np.pi, pos=(400.0, 0.0), v=8.0), np.array([0.1, 0.0]), DT2, VP)  # 进入
r2 = shL.safe_action(egoE, vs(orient=np.pi, pos=(380.0, 0.0), v=8.0), np.array([0.1, 0.0]), DT2, VP)  # 驻留
ok("⑰f 生命周期: ρ0 projection→ρ5 进入 emergency(mode 立)→驻留 emergency(mode 续)",
   r0.source == "projection" and r1.source == "emergency" and r1.emergency_mode is not None and r2.source == "emergency")
shL.reset(); scL._rho = RHO_EMERGENCY
r3 = shL.safe_action(egoE, vs(orient=np.pi, pos=(400.0, 0.0), v=8.0), np.array([0.1, 0.0]), DT2, VP)
ok("⑰f reset 后再进 ρ5 = 新紧急事件(mode 重定)", r3.source == "emergency" and r3.emergency_mode is not None)

# ⑰g safe_action 恒给可执行动作（无 needs_fallback 泄漏；QP 源 ∈box）
ok("⑰g 四源 u_safe 全有限 + QP源∈box", all(np.all(np.isfinite(x.u_safe)) for x in (ra, rb, rc, rd, re_)) and in_box(rb.u_safe) and in_box(rc.u_safe))

# ⑱ Node 4 对抗复核守卫（B-EMERGENCY-BOX/A7：projection box ≠ vessel_params box → raise；防 emergency 越 box）
_mismatch = ContinuousColregsProjection(0.10, 0.01, statechart=_StubChart(RHO_CROSSING))  # proj box ≠ VP(0.24,0.03)
ok("⑱ proj box ≠ vp box → project_qp raise", raises(lambda: _mismatch.project_qp(egoE, far_qp, np.array([0.0, 0.0]), DT2, VP)))
ok("⑱ proj box ≠ vp box → safe_action raise(经 project_qp)", raises(lambda: _mismatch.safe_action(egoE, far_qp, np.array([0.0, 0.0]), DT2, VP)))
ok("⑱ proj box == vp box → 不 raise", not raises(lambda: proj(RHO_CROSSING).safe_action(egoE, far_qp, np.array([0.1, 0.0]), DT2, VP)))

# ⑲ 最坏一步线性化侵入【可复现界】（复审 2026-06-16 + Agent 校正：固化 line385 经验界 + 守护"物理无碰靠 d_safe 大裕度"）
#   projection 源 = QP 把无碰撞约束钉边界 g·u=h（线性 dist=d_safe）；真实 dist 因一阶误差略 < d_safe → 侵入=d_safe−真实dist。
#   侵入【区制相关】：ρ0 全箱摆幅最大→最坏；ρ3 钳 ω→中；ρ1 双钳→近 0（旧注"≤~2.8m"只对 ρ3、对 ρ0 低报）。
#   ⚠️ 最坏是【采样下估、随种子/样本量变】（此种子 n=600 得 ρ0≈4.7m；跨种子 6 万样本 ρ0≈6.8m，D32）→ 断言用 <8m【界】、非精确最坏。
#   ⚠️ 裸船体净空【随 l_obs 缩放】（净空主由 d_safe 内含 2·l_obs 撑）：l_obs=175→~345m / l_obs=120→~239m，均 ≫0 = 物理无碰（Agent 校正：旧"~343m/>300m"只对大他船、不可普适）。
def _worst_intrusion(rho, n=600, seed=20260616, obs_length=175.0):
    rng = np.random.default_rng(seed)
    pj = proj(rho)
    worst, min_bare = 0.0, 1e18
    for _ in range(n):
        ego = vs(orient=rng.uniform(-np.pi, np.pi), v=rng.uniform(0.5, 9.5))
        d = rng.uniform(450.0, 600.0); a = rng.uniform(-np.pi, np.pi)
        obs = vs(orient=rng.uniform(-np.pi, np.pi), pos=(d * np.cos(a), d * np.sin(a)),
                 v=rng.uniform(0.5, 9.5), length=obs_length)
        u_des = np.array([rng.uniform(-A_MAX, A_MAX), rng.uniform(-W_MAX, W_MAX)])
        try:
            r = pj.project_qp(ego, obs, u_des, 10.0, VP)
        except Exception:
            continue
        if r.needs_fallback:
            continue
        p_next = ego_next_position(ego, r.u_safe, 10.0, VP)
        p_obs, R_obs = obstacle_occupancy_disk(obs, 10.0)
        R_ego = ego_circumradius(ego)
        true_dist = float(np.linalg.norm(p_next - p_obs))
        worst = max(worst, (R_ego + R_obs) - true_dist)
        min_bare = min(min_bare, true_dist - (R_ego + _vessel_circumradius(obs.length)))
    return worst, min_bare

_w0, _b0 = _worst_intrusion(RHO_NO_CONFLICT)            # l_obs=175（与本测试障碍同口径）
_w3, _b3 = _worst_intrusion(RHO_CROSSING)
_w1, _b1 = _worst_intrusion(RHO_STAND_ON)
_, _b0_120 = _worst_intrusion(RHO_NO_CONFLICT, obs_length=120.0)   # 小他船净空（Agent 校正：净空随 l_obs 缩放）
print(f"   [线性化侵入,n=600采样下估] ρ0={_w0:.2f}m / ρ3={_w3:.2f}m / ρ1={_w1:.2f}m；裸净空 l175={_b0:.0f}m l120={_b0_120:.0f}m")
ok("⑲ 最坏一步侵入 < 8m 界（区制相关；采样下估、跨种子真最坏≈6.8m<8，D32）", max(_w0, _w3, _w1) < 8.0)
ok("⑲ 物理无碰：裸船体净空 > 150m（l175→~345m / l120→~239m，随 2·l_obs 缩放、均≫0=档位A 安全核心）",
   min(_b0, _b3, _b1) > 150.0 and _b0_120 > 150.0)
ok("⑲ 侵入区制序 ρ0(全箱) > ρ1(双钳)（动作自由度↑→线性化误差↑，文档界须按区制报）", _w0 > _w1)

# ============================================================================
# ⑳ 吞吐优化等价守护（D31，2026-06-16c）：生产 _solve_box_halfplane_qp（直接 OSQP）
#    必须与旧 cvxpy(OSQP) 参考实现在【最近点 + P=∅(可行性)检测】上等价。fuzz 随机 box+半平面。
#    变异守护：若 OSQP 版与 cvxpy 在任一实例可行性判定不一致 / 最近点差 >1e-5 → 翻 FAIL。
# ============================================================================
from trb_env.usv_projection import _solve_box_halfplane_qp, _solve_box_halfplane_qp_cvxpy  # noqa: E402

_rng_eq = np.random.default_rng(20260616)
_mism_feas = 0
_mism_sol = 0
_maxd = 0.0
_n_feas = 0
_n_infeas = 0
for _ in range(800):
    _ud = _rng_eq.uniform(-0.3, 0.3, 2)
    _ai = (-A_MAX, A_MAX)
    _wi = (-W_MAX, W_MAX)
    _rows = []
    for _ in range(int(_rng_eq.integers(0, 3))):
        _rows.append((_rng_eq.uniform(-60, 60, 2), float(_rng_eq.uniform(-200, 200))))
    _xo, _fo = _solve_box_halfplane_qp(_ud, _ai, _wi, _rows)          # 生产：直接 OSQP
    _xc, _fc = _solve_box_halfplane_qp_cvxpy(_ud, _ai, _wi, _rows)    # 参考：cvxpy
    if _fo != _fc:
        _mism_feas += 1
        continue
    if _fo:
        _n_feas += 1
        _d = float(np.linalg.norm(np.asarray(_xo) - np.asarray(_xc)))
        _maxd = max(_maxd, _d)
        if _d > 1e-5:
            _mism_sol += 1
    else:
        _n_infeas += 1
print(f"   [⑳ OSQP↔cvxpy 等价] 可行 {_n_feas}/不可行 {_n_infeas}；可行性不一致 {_mism_feas}；最近点最大差 {_maxd:.2e}")
ok("⑳ 直接 OSQP 与 cvxpy 参考：P=∅(可行性)检测 100% 一致（安全语义不变）", _mism_feas == 0)
ok("⑳ 直接 OSQP 与 cvxpy 参考：最近点等价（差 <1e-5，800 fuzz）", _mism_sol == 0 and _maxd < 1e-5)
ok("⑳ fuzz 真覆盖可行∧不可行两侧（非真空）", _n_feas > 50 and _n_infeas > 50)

# ============================================================================
# ㉑ 危险方向 false-feasible=0【真实 g 量级 + scipy(HiGHS) 地面真值】（交接审核 2026-06-17b 补）：
#    ⑳ 的 g∈[-60,60] 窄于真实（实测 ‖g‖ 中位~128/可达~550）→ L40/L41 自认其"可行性100%一致"仅窄域成立、
#    真实域危险方向=0 此前只由一次性审核脚本坐实、未固化进测试。本块用真实量级 g（g0∈[-50,50]/g1∈[-550,550]）
#    造贴角/空集算例 + scipy linprog 可行性地面真值。安全核心断言：OSQP 报"可行"返回的点必真满足全部约束
#    （box + g·u≤h，违约≤1e-5）= 绝不把碰撞动作当可行放行；偶发 false-infeasible(更保守→落兜底)是安全侧。
from scipy.optimize import linprog  # noqa: E402

_rng_dg = np.random.default_rng(20260617)
_n_dfeas = _n_danger = _n_ff = _n_safe_inf = _n_both_inf = 0
for _ in range(1500):
    _ud = _rng_dg.uniform(-0.3, 0.3, 2)
    _ai = (-A_MAX, A_MAX); _wi = (-W_MAX, W_MAX)
    _rows = []
    for _ in range(int(_rng_dg.integers(1, 3))):                 # 1-2 条真实量级半平面
        _g = np.array([_rng_dg.uniform(-50.0, 50.0), _rng_dg.uniform(-550.0, 550.0)])
        _corner = _g[0] * _rng_dg.choice([_ai[0], _ai[1]]) + _g[1] * _rng_dg.choice([_wi[0], _wi[1]])
        _rows.append((_g, float(_corner + _rng_dg.uniform(-30.0, 5.0))))   # h 取 box 角附近→贴角/空集
    _xo, _fo = _solve_box_halfplane_qp(_ud, _ai, _wi, _rows)
    _Aub = np.array([_r[0] for _r in _rows]); _bub = np.array([_r[1] for _r in _rows])
    _lp = linprog(c=[0.0, 0.0], A_ub=_Aub, b_ub=_bub, bounds=[(_ai[0], _ai[1]), (_wi[0], _wi[1])], method="highs")
    _gt = (_lp.status == 0)
    if _fo:
        _n_dfeas += 1
        _x = np.asarray(_xo)
        _v = max(_ai[0] - _x[0], _x[0] - _ai[1], _wi[0] - _x[1], _x[1] - _wi[1])
        for _g, _h in _rows:
            _v = max(_v, float(_g @ _x) - _h)
        if _v > 1e-5:
            _n_danger += 1                       # 危险：OSQP 报可行但返回点真违约
        if not _gt:
            _n_ff += 1                           # OSQP 可行 ∧ GT 空集（应全部非危险=边界级）
    else:
        if _gt:
            _n_safe_inf += 1                     # 安全侧：OSQP 更保守→落兜底
        else:
            _n_both_inf += 1
print(f"   [㉑ 危险方向·真实g量级] OSQP可行 {_n_dfeas}/danger {_n_danger}/可行∧GT空集 {_n_ff}；安全侧false-infeas {_n_safe_inf}；双方判空 {_n_both_inf}")
ok("㉑ 危险方向 false-feasible=0：OSQP'可行'点真满足全部约束(违约≤1e-5，真实g量级1500算例+scipy地面真值)", _n_danger == 0)
ok("㉑ 非真空：真实量级 fuzz 同时触达可行(>50)与不可行(>50)两侧", _n_dfeas > 50 and (_n_both_inf + _n_safe_inf) > 50)

print("\n" + (f"✅ 全部 PASS（{_total} 项）" if _fail == 0 else f"❌ {_fail}/{_total} 项 FAIL"))
sys.exit(1 if _fail else 0)
