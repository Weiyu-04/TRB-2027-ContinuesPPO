#!/usr/bin/env python3
"""gap#1 迫近不可避证书 committed 测试（`03` L165·正式命题3·`Paper/可证明层_正式命题_0707.md`）。

验：① fires on imminent head-on + 每个 fire 经轻量逃逸搜索确认真不可避（soundness 守卫·0 假阳）
    ② 不 fire on 明显可避（大偏移/远场）·no-fire≠可避（单侧充分·诚实作用域）
    ③ 硬化界正确（a_eff 用 v_bnd=v_max+a_max·dt=11.9 非 v_max；L_lat 转90°包络 π/2 连续+超之线性）
    ④ t=0 已撞→fire t*=0  ⑤ 所有 fire 的 t* < 验证限 (π/2)/w_max=52.4s（regime 有效·t_crit≈21.6）
    ⑥ 点-矩形距离函数正确  ⑦ 纯分析·不改盾（projection 回归另测·此处查函数不在 safe_action 源码）
运行：/opt/miniconda3/envs/trb/bin/python -B 代码/tests/test_imminent_unavoidable.py
"""
import os, sys, math, inspect
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from trb_env.usv_projection import imminent_unavoidable_certificate as CERT, _dist_point_to_rect
import trb_env.usv_projection as PROJ
from trb_env.usv_colregs import VesselState, _ego_rect
from trb_env import usv_dynamics

P = usv_dynamics.make_vessel_params(9.5)
L, WID = 175.0, 25.4
N_FAIL = 0


def chk(cond, msg):
    global N_FAIL
    print(("  ✅ " if cond else "  ❌ ") + msg)
    if not cond:
        N_FAIL += 1


def VS(p, th, v):
    return VesselState(position=np.array(p, float), orientation=th, velocity=v, length=L)


def _hit(pe, the, pm, thm):
    return _ego_rect(np.array(pe), the, L, WID).intersects(_ego_rect(np.array(pm), thm, L, WID))


def _escapes(pe, the, ve, pm, thm, vm, T=180.0):
    """轻量 ground-truth：某可行控制序列可避开船体碰撞？(5x5 常量 + 单切换·subdt=2·精确船体)。"""
    steps = int(T / 10.0); sub = 2.0
    ag = np.linspace(-P.a_max, P.a_max, 5); wg = np.linspace(-P.w_max, P.w_max, 5)
    ctr = [(a, w) for a in ag for w in wg]
    seqs = [[c] * steps for c in ctr]
    for c1 in ctr[::3]:
        for c2 in ctr[::3]:
            seqs.append([c1] * (steps // 2) + [c2] * (steps - steps // 2))
    for seq in seqs:
        st = np.array([pe[0], pe[1], the, ve], float)
        col = _hit(st[:2], st[2], np.array(pm), thm)   # t=0 guard
        for k, (a, w) in enumerate(seq):
            if col:
                break
            for j in range(int(10.0 / sub)):
                st = usv_dynamics.step(st, np.array([a, w]), sub, P); st[3] = min(max(st[3], 0.0), 9.5)
                tn = k * 10.0 + (j + 1) * sub
                if _hit(st[:2], st[2], np.array(pm) + np.array(vm) * tn, thm):
                    col = True; break
        if not col:
            return True
    return False


print("① fires on imminent head-on + soundness 守卫（每 fire 真不可避·0 假阳）")
imminent = [("d0=180 off=0 vc=18", [-180, 0], 9., [0, 0], math.pi, [-9, 0]),
            ("d0=200 off=0 vc=16", [-200, 0], 8., [0, 0], math.pi, [-8, 0]),
            ("d0=180 off=15 vc=18", [-180, 15], 9., [0, 0], math.pi, [-9, 0]),
            ("d0=240 off=0 vc=16", [-240, 0], 8., [0, 0], math.pi, [-8, 0])]
n_fire = 0; max_tstar = 0.0; fp = 0
for tag, pe, ve, pm, thm, vm in imminent:
    u, tw = CERT(VS(pe, 0.0, ve), VS(pm, thm, math.hypot(*vm)), P)
    if u:
        n_fire += 1; max_tstar = max(max_tstar, tw)
        if _escapes(pe, 0.0, ve, pm, thm, vm):    # fire 但有逃逸 = FALSE POSITIVE
            fp += 1; print(f"    FALSE POSITIVE: {tag}")
chk(n_fire >= 3, f"迫近对开触发 ≥3/4（实 {n_fire}·非空）")
chk(fp == 0, f"触发的全部真不可避（0 假阳·soundness 守卫）")

print("② 不 fire on 明显可避 + no-fire≠可避（诚实作用域）")
avoidable = [("大偏移 off=200", [-300, 200], 8., [0, 0], math.pi, [-8, 0]),
             ("crossing", [-400, -400], 7., [0, 0], math.pi / 2, [0, 7]),
             ("远场 d0=800", [-800, 0], 8., [0, 0], math.pi, [-8, 0])]
for tag, pe, ve, pm, thm, vm in avoidable:
    u, tw = CERT(VS(pe, 0.0, ve), VS(pm, thm, math.hypot(*vm)), P)
    chk(u is False and tw is None, f"{tag} → 不触发（False,None·no-fire≠断言可避）")

print("③ 硬化界正确（v_bnd / a_eff / L_lat 包络）")
v_bnd = P.v_max + P.a_max * 10.0
chk(abs(v_bnd - 11.9) < 1e-9, f"v_bnd = v_max+a_max·dt = {v_bnd}（=11.9·非 v_max=9.5）")
a_eff = math.hypot(P.a_max, v_bnd * P.w_max)
chk(abs(a_eff - 0.43016) < 1e-3, f"a_eff = √(a_max²+(v_bnd·w_max)²) = {a_eff:.5f}（硬化后·非 0.3726）")
# L_lat 包络：π/2 处 (1-cos) 分支与线性分支连续
whalf = (math.pi / 2) / P.w_max
lcos = (v_bnd / P.w_max) * (1 - math.cos(P.w_max * whalf))
llin = (v_bnd / P.w_max) + v_bnd * (whalf - (math.pi / 2) / P.w_max)
chk(abs(lcos - llin) < 1e-6, f"L_lat 在 w_max·t=π/2 处两分支连续（{lcos:.2f}≈{llin:.2f}）")

print("④ t=0 已撞 → 立即 fire t*=0")
u0, tw0 = CERT(VS([0, 0], 0.0, 8.), VS([50, 0], math.pi, 8.), P)   # 中心距50<船长→已重叠
chk(u0 and tw0 == 0.0, f"t=0 船体重叠 → 触发 t*=0（实 {u0},{tw0}）")

print("⑤ 所有 fire 的 t* < 验证限 (π/2)/w_max=52.4s（regime 有效·t_crit≈21.6）")
chk(max_tstar < (math.pi / 2) / P.w_max, f"max t* = {max_tstar:.1f}s < 52.4s（触发困在 L_lat 紧区·主窗口亲核 t_crit=21.6）")

print("⑥ 点-矩形距离函数正确")
chk(abs(_dist_point_to_rect([0, 0], [0, 0], 0.0, 100, 20) - 0.0) < 1e-9, "中心点→0")
chk(abs(_dist_point_to_rect([60, 0], [0, 0], 0.0, 100, 20) - 10.0) < 1e-9, "沿长轴外 10m（半长50·点60）→10")
chk(abs(_dist_point_to_rect([0, 20], [0, 0], 0.0, 100, 20) - 10.0) < 1e-9, "沿宽轴外 10m（半宽10·点20）→10")
chk(abs(_dist_point_to_rect([0, 20], [0, 0], math.pi / 2, 100, 20) - 0.0) < 1e-9, "旋转90°后 [0,20] 落进长轴→0")

print("⑦ 纯分析·不改盾（证书不被 safe_action 控制路径调用）")
try:
    from trb_env.usv_continuous_shield import ContinuousColregsProjection
    _src = inspect.getsource(ContinuousColregsProjection)
    chk("imminent_unavoidable_certificate" not in _src, "证书不出现在 ContinuousColregsProjection 源码（控制路径干净）")
except Exception:
    # 退而查 safe_action 相关源码不引证书
    chk("imminent_unavoidable_certificate" not in inspect.getsource(PROJ.ContinuousColregsProjection.safe_action)
        if hasattr(PROJ, "ContinuousColregsProjection") else True, "safe_action 不引证书")

print("⑧ 硬化 pin（防 un-harden 回归·对抗审 ISSUE-E）+ 近边界 FP-guard（ISSUE-D）+ 防御守卫（ISSUE-B/C）")
from trb_env.usv_projection import _reach_params
vb, ae = _reach_params(P, 10.0)
chk(abs(vb - 11.9) < 1e-9 and abs(ae - 0.43016) < 1e-3,
    f"_reach_params 硬化 pin：v_bnd={vb}(=11.9·非 v_max 9.5) a_eff={ae:.5f}(=0.4302·非软 0.3726)——证书内用之·防 un-harden")
# 近边界可避对开：off=40（off=15 触发·off=40 不触发=紧贴 fire 边界）·直行即让开 40m 横向→真可避。
#   若回归膨胀 R_box/内切盘/obs_width（over-firing·unsound 方向）误触发此局，"不触发" 断言失败。
pe, the, ve, pm, thm, vm = [-200, 40], 0.0, 9.0, [0, 0], math.pi, [-9, 0]
u_av, _ = CERT(VS(pe, the, ve), VS(pm, thm, 9.0), P)
chk(u_av is False, f"近边界可避对开(d0=200 off=40) → 不触发（={u_av}·防 over-firing 回归·FP-guard）")
chk(_escapes(pe, the, ve, pm, thm, vm), "  且该局确有逃逸（真可避·直行让开 40m 横向·FP-guard 有效）")
try:
    CERT(VS([0, 0], 0.0, 8.0), VS([500, 0], math.pi, 8.0), P, t_horizon=-1.0); chk(False, "t_horizon<0 应 ValueError")
except ValueError:
    chk(True, "t_horizon<0 → ValueError（守卫）")
try:
    CERT(VS([0, 0], 0.0, 8.0), VS([500, 0], math.pi, 8.0), P, n_grid=0); chk(False, "n_grid=0 应 ValueError")
except ValueError:
    chk(True, "n_grid=0 → ValueError（守卫·防静默不判 footgun）")

print("⑨ L_lat 横向可达界守卫（`03` L167 finding② · 补 L166 committed 测试 2 空洞守卫）")
#   复审 L167(Agent-3 变异审)抓：原 test 对 (iii) 删 >π/2 包络 / (ii-b) 缩 L_lat（unsound over-firing 方向）
#   都 catch 不住（真实 fire 全 t*≤11s·>π/2 支从不被走·横向非 binding）。抽 _lateral_reach_bound 为模块级后直测两分支+全幅值。
from trb_env.usv_projection import _lateral_reach_bound
vb2 = P.v_max + P.a_max * 10.0                             # 11.9（硬化 v_bnd）
wm2 = P.w_max
tq = (math.pi / 2) / wm2                                   # 52.36s：两分支切点
# (a) 两分支在 ω_max·t=π/2 连续
chk(abs(_lateral_reach_bound(tq - 1e-6, vb2, wm2) - _lateral_reach_bound(tq + 1e-6, vb2, wm2)) < 1e-3,
    "L_lat 两分支在 ω_max·t=π/2 连续")
# (b) t=120s（ω_max·t=3.6>π/2·regime 外）：必用【线性包络】·非 (1−cos) 延拓（后者低估→R_box 不再过近似→unsound）
lin120 = (vb2 / wm2) + vb2 * (120.0 - (math.pi / 2) / wm2)
cos120 = (vb2 / wm2) * (1.0 - math.cos(wm2 * 120.0))       # 删包络后错误值（低估）
got120 = _lateral_reach_bound(120.0, vb2, wm2)
chk(abs(got120 - lin120) < 1e-3, f"t=120s 用线性包络 L_lat={got120:.1f}（=预期 {lin120:.1f}·防删 >π/2 包络回归·iii）")
chk(got120 > cos120 + 100.0, f"线性包络 {got120:.1f} ＞ (1−cos)延拓 {cos120:.1f}（包络真放大=sound 方向）")
# (c) 全幅值·单调不减（防 L_lat ×k 缩小的 over-firing 回归·ii-b：缩小任一点幅值必破此断言）
tsq = [10.0, 30.0, 52.36, 60.0, 90.0, 120.0]
vq = [_lateral_reach_bound(t, vb2, wm2) for t in tsq]
exp = [((vb2 / wm2) * (1 - math.cos(wm2 * t)) if wm2 * t <= math.pi / 2
        else (vb2 / wm2) + vb2 * (t - (math.pi / 2) / wm2)) for t in tsq]
chk(all(abs(vq[i] - exp[i]) < 1e-6 for i in range(len(tsq))), f"L_lat 全 t 幅值=硬化公式（防缩小 over-firing·ii-b·{[round(v) for v in vq]}）")
chk(all(vq[i + 1] >= vq[i] - 1e-9 for i in range(len(vq) - 1)), "L_lat 单调不减（有效上界）")
# (d) 证书真调用 _lateral_reach_bound（防绕过模块函数内联错 L_lat）
chk("_lateral_reach_bound" in inspect.getsource(CERT), "证书源码引用 _lateral_reach_bound（守卫真用之·非内联旁路）")

print("\n" + ("=" * 50))
print("✅ 全部通过" if N_FAIL == 0 else f"❌ {N_FAIL} 项失败")
sys.exit(1 if N_FAIL else 0)
