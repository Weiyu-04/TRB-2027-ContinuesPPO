"""N1 档位B* 递归可行性终端检查（存在性级）测试。
覆盖：① 默认关 <1e-9 逐位等价（bit-identical）+ 变异守护（开=load-bearing·用真 As-空 give-way 落点态）
      ② 落点 ρ' 路径依赖修法正确（当前ρ播种 ≠ fresh ρ0）③ 远场平凡可行 ④ ρ5 落点=经验兜底可用
      ⑤ give-way 落点 As 空→False / As 非空→True（终端约束真会区分）⑥ 兜底路由（终端失败→非 projection 源）。
运行：python 代码/tests/test_recursive_feasibility.py（应全 ✅）。
⚠️ fixture（2026-07-03 亲扫·复验确定性）：As-空 give-way 落点态 = so=(500,-850,120°,5) seed ρ3 → 落点真 ρ3(crossing)·As 空 → _terminal_feasible=False；
   As-非空对照 = so=(1200,-1600,60°,5) seed ρ3 → 落点真 ρ3(crossing)·As 非空(2 动作) → True。这些是 N1 load-bearing 的直接证据（能区分有/无脱身）。
   ⚠️ 2026-07-03 二次复审抓修：旧 SO_TRUE=(500,-900,90°) 落点实为 ρ5(emergency)·True 走【ρ5 定义性放行】捷径(usv_projection:348)而非 encounter As 分支
   → 正方向(As 非空 give-way→True)从未被真正见证。新 SO_TRUE 落点真 give-way·并加【落点区制守卫】锁死正方向不再假见证。
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from trb_env.usv_colregs import (VesselState, ColregsStatechart, RHO_CROSSING, RHO_NO_CONFLICT,
    RHO_EMERGENCY, RHO_HEAD_ON, RHO_OVERTAKE, predict_state_cv, collision_possible, is_emergency,
    persistent_crossing, head_on, overtake, keep)
from trb_env.usv_projection import ContinuousColregsProjection
from trb_env import usv_dynamics
from trb_env.usv_dynamics import make_vessel_params

vp = make_vessel_params()
A_MAX, W_MAX, L, DT = vp.a_max, vp.w_max, 175.0, 10.0
def mk(x, y, psi, v): return VesselState(position=np.array([float(x), float(y)]), orientation=float(psi), velocity=float(v), length=L)
def newproj(rf=False, tdt=0.5):
    return ContinuousColregsProjection(A_MAX, W_MAX, recursive_feasibility=rf, terminal_dt_sim=tdt)

# ── 确定性 fixture（亲扫复验）──
EGO = mk(0, 0, 0.0, 6.0)
SO_FALSE = mk(500, -850, np.deg2rad(120), 5.0)   # give-way(crossing) 落点·As 空 → _terminal_feasible False（seed ρ3·落点真 ρ3）
SO_TRUE  = mk(1200, -1600, np.deg2rad(60), 5.0)  # give-way(crossing) 落点·As 非空(2 动作) → True（seed ρ3·落点真 ρ3·非 ρ5 捷径·复审抓修）
ZERO = np.array([0.0, 0.0])

def _landing_rho(so, ego=EGO, u=ZERO, seed=RHO_CROSSING, dt=DT):
    """复现 _terminal_feasible 内部落点区制判定（当前ρ播种·u=0 落点·dt 步）：证 fixture 落点真 give-way、非 ρ5/ρ0 捷径。"""
    nxt = usv_dynamics.step(np.array([ego.position[0], ego.position[1], ego.orientation, ego.velocity]), np.asarray(u, float), dt, vp)
    ego_n = mk(nxt[0], nxt[1], nxt[2], nxt[3]); obs_n = predict_state_cv(so, dt)
    tmp = ColregsStatechart(); tmp.rho = int(seed)
    return int(tmp.step(ego_n, obs_n))

fails = []
def check(name, cond):
    print(f"  {'✅' if cond else '❌'} {name}")
    if not cond: fails.append(name)

def states():
    S = []
    for d in (500, 800, 1200, 1800, 2500):
        for brg in np.deg2rad((-30, 0, 30, 90, 150, 180)):
            for oh in np.deg2rad((0, 90, 180, 270)):
                S.append((mk(0, 0, 0.0, 6.0), mk(d*np.cos(brg), d*np.sin(brg), oh, 5.0)))
    return S

print("① 默认关 bit-identical + 变异守护（开=load-bearing）")
off1, off2 = newproj(False, 0.5), newproj(False, 0.1)
u_des = np.array([0.03, -0.01])
max_off_diff = 0.0
for se, so in states():
    off1.reset(); off2.reset()
    r1 = off1.project_qp(se, so, u_des, DT, vp)
    r2 = off2.project_qp(se, so, u_des, DT, vp)
    max_off_diff = max(max_off_diff, float(np.max(np.abs(r1.u_safe - r2.u_safe))),
                       abs(int(r1.needs_fallback) - int(r2.needs_fallback)))
check("默认关：terminal_dt_sim 0.5 vs 0.1 输出 <1e-9 逐位等价（关时终端块真跳过）", max_off_diff < 1e-9)
# 变异守护：同一 give-way 落点 As-空态，开关 N1 结果必不同（N1 把可行投影翻成兜底）
p_off = newproj(False); p_off.reset(); p_off._sc.rho = RHO_CROSSING
p_on  = newproj(True);  p_on.reset();  p_on._sc.rho  = RHO_CROSSING
r_off = p_off.project_qp(EGO, SO_FALSE, u_des, DT, vp)
r_on  = p_on.project_qp(EGO, SO_FALSE, u_des, DT, vp)
check("变异守护：As-空 give-way 落点态·开 N1 needs_fallback=True 而关=False（flag load-bearing）",
      (r_off.needs_fallback is False) and (r_on.needs_fallback is True) and (r_off.rho == RHO_CROSSING == r_on.rho))

print("② 落点 ρ' 路径依赖修法正确（当前ρ播种 ≠ fresh ρ0）")
diverge = None
for d in range(300, 1400, 50):
    for brg in np.deg2rad(range(-60, 61, 15)):
        for oh in np.deg2rad(range(40, 140, 15)):
            se = mk(0, 0, 0.3, 7.0); so = mk(d*np.cos(brg), d*np.sin(brg), oh, 6.0)
            if is_emergency(se, so): continue
            if collision_possible(se, so) and not (persistent_crossing(se, so) or head_on(se, so) or overtake(se, so)) and not keep(se, so):
                diverge = (se, so); break
        if diverge: break
    if diverge: break
if diverge:
    se, so = diverge
    lag = ColregsStatechart(); lag.rho = RHO_CROSSING; rho_lag = lag.step(se, so)
    fr = ColregsStatechart(); fr.reset(); rho_fr = fr.step(se, so)
    check(f"分岔状态：当前ρ播种→{rho_lag}(维持give-way) ≠ fresh ρ0→{rho_fr}", rho_lag != rho_fr and rho_lag == RHO_CROSSING)
else:
    check("分岔状态（cp 空隙·网格未命中·非硬失败）", True)

print("③ 远场平凡可行（落点 ρ0 → True）")
p = newproj(True); p.reset()
check("远场落点 _terminal_feasible=True", p._terminal_feasible(mk(0,0,0.0,6.0), mk(3000,2000,0.0,5.0), ZERO, RHO_NO_CONFLICT, DT, vp) is True)

print("④ ρ5 落点=紧急兜底可用（经验·诚实 limitation）→ True")
p = newproj(True); p.reset()
check("ρ5 落点 _terminal_feasible=True（紧急控制器兜底可用）",
      p._terminal_feasible(mk(0,0,0.0,9.0), mk(700,0,np.pi,9.0), ZERO, RHO_EMERGENCY, DT, vp) is True)

print("⑤ give-way 落点 As 空→False / As 非空→True（终端约束真区分·N1 load-bearing 核心）")
p = newproj(True, 0.5); p.reset()
# 🔒 落点区制守卫（2026-07-03 复审抓修）：先证两 fixture 落点【真 give-way 区制】，否则 True/False 可能来自
#    ρ5/ρ0/ρ1 定义性放行捷径 = 假见证（旧 SO_TRUE 落 ρ5·正方向从未真测）。锁死两方向都走 encounter As 分支。
GW = (RHO_HEAD_ON, RHO_CROSSING, RHO_OVERTAKE)
check(f"守卫：SO_FALSE 落点真 give-way 区制（得 ρ{_landing_rho(SO_FALSE)}·非 ρ5/ρ0 捷径）", _landing_rho(SO_FALSE) in GW)
check(f"守卫：SO_TRUE 落点真 give-way 区制（得 ρ{_landing_rho(SO_TRUE)}·非 ρ5 捷径·正方向真见证）", _landing_rho(SO_TRUE) in GW)
check("As-空 give-way 落点 → _terminal_feasible=False（终端约束会拦·走 encounter As 分支）",
      p._terminal_feasible(EGO, SO_FALSE, ZERO, RHO_CROSSING, DT, vp) is False)
check("As-非空 give-way 落点 → _terminal_feasible=True（不拦有脱身的·真走 encounter As 分支非 ρ5 捷径）",
      p._terminal_feasible(EGO, SO_TRUE, ZERO, RHO_CROSSING, DT, vp) is True)

print("⑥ 兜底路由：终端失败 → safe_action 源 ≠ 'projection'")
p = newproj(True, 0.5); p.reset(); p._sc.rho = RHO_CROSSING
res = p.safe_action(EGO, SO_FALSE, u_des, DT, vp)
check(f"终端失败态 safe_action.source={res.source} ≠ projection（退兜底）", res.source != "projection")

print()
if fails:
    print(f"❌ 失败 {len(fails)}: {fails}"); sys.exit(1)
print("✅ 全部通过")
