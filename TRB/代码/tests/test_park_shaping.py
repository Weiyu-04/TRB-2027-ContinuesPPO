"""
想法B 终端保速势 Φ_park 冒烟测试（`03` L109·2026-06-28）——治"终端横向修正时过早减速停带外"的 PBRS 终端保速势。
跑：/opt/miniconda3/envs/trb/bin/python -B 代码/tests/test_park_shaping.py
覆盖：① Φ_park=well_park·prox_radial(‖r‖;R_park)·speed_frac(v;V_target) 曲线值（含边界/封顶）
   ② park_weight=0 → reward 与无 park【逐位等价 bit-identical】（不加 shape_park 键）
   ③ park_weight>0 机制：近目标【减速被罚·保速给奖】（直击"近目标 r_goal 衰减→减速停短"失败）
   ④ 远目标(d>R_park) → prox=0 → 无作用 ⑤ PBRS 三势加性（shape==Φ_A+Φ_xtrack+Φ_park 各单独·线性）
   ⑥ 守卫（park_weight<0 / park_radius<=0·park_v_target<=0 → ValueError） ⑦ 到达 terminal Φ(s')=0（PBRS 终止）。
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from trb_env.usv_reward import RewardFunction

_fail = 0; _total = 0
def ok(name, cond):
    global _fail, _total
    _total += 1
    if not cond: _fail += 1
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")

GC = np.array([5000.0, 478.0]); IP = np.array([1000.0, 600.0]); GO = (-0.17, 0.17)
def mk(pw=0.0, pr=400.0, vt=4.0, wb=0.0, wx=0.0):
    return RewardFunction(GC, IP, goal_orientation=(GO if (wb > 0 or wx > 0) else None),
                          gamma=0.99, park_weight=pw, park_radius=pr, park_v_target=vt,
                          well_shaping_weight=wb, xtrack_weight=wx)

print("===== A) Φ_park 曲线（prox_radial × speed_frac·封顶 V_target）=====")
rf = mk(pw=50.0, pr=400.0, vt=4.0)
# 直接验 _phi_park：ego=[px,py,θ,v]
def phi(px, py, v): return rf._phi_park(np.array([px, py, 0.0, v]))
ok("A1 远目标(d>R_park) → Φ=0", abs(phi(GC[0]-500, GC[1], 5.0) - 0.0) < 1e-12)           # d=500>400
ok("A2 目标中心+满速 → Φ=well_park·1·1=50", abs(phi(GC[0], GC[1], 4.0) - 50.0) < 1e-9)     # prox=1,speed_frac=1
ok("A3 速度封顶 V_target（v=8>4 仍 speed_frac=1）", abs(phi(GC[0], GC[1], 8.0) - 50.0) < 1e-9)
ok("A4 半速 → Φ=50·1·0.5=25", abs(phi(GC[0], GC[1], 2.0) - 25.0) < 1e-9)                   # speed_frac=0.5
ok("A5 v=0 → speed_frac=0 → Φ=0（停船无势=正是要避免的态）", abs(phi(GC[0], GC[1], 0.0) - 0.0) < 1e-12)
# prox 线性：d=200 → prox=1−200/400=0.5
ok("A6 d=200·满速 → Φ=50·0.5·1=25", abs(phi(GC[0]-200, GC[1], 4.0) - 25.0) < 1e-9)

print("\n===== B) park=0 → reward 与无 park【逐位等价 bit-identical】=====")
rng = np.random.default_rng(0)
seq = [np.array([4700.0 + 30*i, 520.0 - 5*i, 0.02*i, 5.0 - 0.5*i]) for i in range(8)]
def run(pw):
    e = mk(pw=pw); e.reset(seq[0]); out = []
    for s in seq:
        r, parts = e.step(s, [])
        out.append((r, parts.get("shape_park", "NONE")))
    return out
base = run(0.0); pk = run(50.0)
# 参照 = 【完全不传 park 参数】纯默认构造（park 不存在的代码路径代理）→ 与 park=0 应逐位等价
_ref_rf = RewardFunction(GC, IP, goal_orientation=None, gamma=0.99)
_ref_rf.reset(seq[0]); ref = []
for s in seq:
    _r, _p = _ref_rf.step(s, []); ref.append((_r, _p.get("shape_park", "NONE")))
ok("B1 park=0 与【完全不传 park 参数】reward 序列逐位等价(bit-identical) + 两者均无 shape_park 键",
   all(abs(a[0] - b[0]) < 1e-12 for a, b in zip(base, ref))   # 🔴 修：对照纯默认 ref(非 base 自比·`03` L111①b)
   and all(a[1] == "NONE" for a in base) and all(b[1] == "NONE" for b in ref))
ok("B1b park=50 → 每步有 shape_park 键（feature 真生效·非静默无效·用上 pk 验非对称）",
   all(b[1] != "NONE" for b in pk))

print("\n===== C) park>0 机制：近目标【减速被罚·保速给奖】=====")
# 单调减速序列（v 5→0）·近目标 → shape_park 在减速步应为负（PBRS γΦ'−Φ·Φ随v降而降→F<0）
e = mk(pw=50.0); decel = [np.array([4800.0, 500.0, 0.0, v]) for v in (5.0, 4.0, 3.0, 2.0, 1.0, 0.2)]
e.reset(decel[0]); sp = []
for s in decel:
    _r, parts = e.step(s, []); sp.append(parts.get("shape_park"))
ok("C1 近目标【持续减速】→ shape_park 各步为负（保速被破=罚·直击减速停短）",
   all(x is not None and x < 0 for x in sp[1:]))   # 首步是 Φ(s0) telescoping 边界·看后续减速步
# 保速序列（v 恒 4）·近目标且 prox 上升（越来越近）→ shape_park 应为正（Φ 升）
e2 = mk(pw=50.0); keep = [np.array([4700.0 + 40*i, 500.0, 0.0, 4.0]) for i in range(5)]
e2.reset(keep[0]); sp2 = []
for s in keep:
    _r, parts = e2.step(s, []); sp2.append(parts.get("shape_park"))
ok("C2 近目标【保满速+逼近】→ shape_park 为正（保速给奖·prox 升 Φ 升）", all(x is not None and x > 0 for x in sp2[1:]))

print("\n===== D) 远目标 prox=0 → 无作用（不碰避碰段）=====")
e3 = mk(pw=50.0); far = [np.array([1000.0 + 50*i, 600.0, 0.0, 5.0]) for i in range(4)]  # d~4000>R_park
e3.reset(far[0]); fr = []
for s in far:
    _r, parts = e3.step(s, []); fr.append(parts.get("shape_park"))
ok("D1 远目标(d≫R_park) → shape_park 恒 0（远场不给梯度·同 Φ_A/Φ_xtrack 哲学）",
   all(abs(x) < 1e-12 for x in fr))

print("\n===== E) PBRS 三势加性（shape == Φ_A + Φ_xtrack + Φ_park 各单独·线性）=====")
def total_shape(pw, wb, wx):
    e = mk(pw=pw, wb=wb, wx=wx); e.reset(seq[0]); s = []
    for st in seq:
        _r, parts = e.step(st, []); s.append(parts.get("shape", 0.0))
    return np.array(s)
s_all = total_shape(50.0, 200.0, 200.0)
s_b = total_shape(0.0, 200.0, 0.0); s_x = total_shape(0.0, 0.0, 200.0); s_p = total_shape(50.0, 0.0, 0.0)
ok("E1 combined shape == well_B单独 + well_X单独 + park单独（PBRS 线性·<1e-9）",
   np.allclose(s_all, s_b + s_x + s_p, atol=1e-9))

print("\n===== F) 守卫 + 到达 terminal Φ(s')=0 =====")
def raises(fn):
    try: fn(); return False
    except ValueError: return True
ok("F1 park_weight<0 → ValueError", raises(lambda: mk(pw=-1.0)))
ok("F2 park_weight>0 但 park_radius<=0 → ValueError", raises(lambda: mk(pw=50.0, pr=0.0)))
ok("F3 park_weight>0 但 park_v_target<=0 → ValueError", raises(lambda: mk(pw=50.0, vt=0.0)))
# 到达(goal=True) → Φ(s')=0 → shape_park = γ·0 − Φ(s) = −Φ(s)（PBRS 终止·不 bootstrap 假势）
e4 = mk(pw=50.0); e4.reset(np.array([4900.0, 480.0, 0.0, 4.0]))
_r, parts = e4.step(np.array([4950.0, 478.0, 0.0, 4.0]), [], term_flags={"goal": True})
# Φ(s)=50·prox·1（近目标·满速）>0 → shape_park 应 = −Φ(prev)（含 γΦ'=0）
ok("F4 到达 terminal → Φ(s')=0（PBRS 终止·shape_park=−Φ(prev)<0·不注假 bootstrap 势）",
   parts.get("shape_park") is not None and parts["shape_park"] < 0)

print()
if _fail == 0:
    print(f"✅ 全部 PASS（{_total} 项）")
else:
    print(f"❌ {_fail}/{_total} 项 FAIL")
sys.exit(1 if _fail else 0)
