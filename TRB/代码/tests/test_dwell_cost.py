#!/usr/bin/env python3
"""非PBRS 入库赤字滞留成本 r_dwell 冒烟测试（`03` L161/L162·连续臂专属·治 corr≈0 终端入库病）。

验（对齐 test_step_cost.py 范式 + L161 实现前须验清单）：
  ① c_dwell=0 与 baseline 逐位等价 bit-identical（<1e-12·不加键）
  ② c_dwell>0 近场每步 r_sparse −= c_dwell·g(s)·g 与独立手算逐位一致（数学正确）·其它项不动（正交）
  ③ 🔴 θ_c 门控改点（唯一既有逻辑改点·L161 铁证②）：dwell-only(well_B=well_X=0·c_dwell>0)→θ_c 正确算出·不崩；缺 goal_orientation → 报错；shaping_radius=0 不牵连 dwell-only
  ④ 边界/饱和：远场(‖r‖>R_DWELL)→dwell=0（逐位）；|e_cross|>W & |dθ|>H → g 饱和=1 → 成本=−c_dwell（farm 免疫上界）
  ⑤ farm 免疫：全轨迹 dwell 贡献恒 ≤ 0（无正的每步项可刷）
  ⑥ B_DWELL 终端锚：真入库 +B_DWELL（仅 goal 步·嵌 c_dwell>0 块）；b_dwell>0 但 c_dwell=0 → 报错（防 silent no-op）
  ⑦ 离散盾 ShieldedUSVEnv / UnshieldedUSVEnv 硬拒 dwell 键 + config_conflict 区分 dwell on/off + ContinuousProjectionEnv 透传
运行：/opt/miniconda3/envs/trb/bin/python -B 代码/tests/test_dwell_cost.py
"""
import os, sys, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from trb_env.usv_reward import RewardFunction
from trb_env.usv_dynamics import wrap_to_pi

GOAL = [5000.0, 1100.0]
INIT = [1000.0, 600.0]
ORI = (-0.17, 0.17)          # → θ_c = 0.0
THETA_C = 0.0
N_FAIL = 0


def chk(cond, msg):
    global N_FAIL
    print(("  ✅ " if cond else "  ❌ ") + msg)
    if not cond:
        N_FAIL += 1


def _mk(c_dwell=0.0, well_b=0.0, w=90.0, h=0.52, r=250.0, b=0.0, ori=ORI, shaping_radius=500.0):
    return RewardFunction(GOAL, INIT, goal_orientation=ori, well_shaping_weight=well_b,
                          shaping_radius=shaping_radius, gamma=0.99,
                          c_dwell=c_dwell, w_dwell=w, h_dwell=h, dwell_radius=r, b_dwell=b)


def _g_manual(px, py, psi, w=90.0, h=0.52):
    """独立手算 g(s)（不碰 RewardFunction 内部）：g=0.5·min(|e_cross|/W,1)+0.5·min(|dθ|/H,1)。"""
    rx, ry = GOAL[0] - px, GOAL[1] - py
    e_cross = rx * (-math.sin(THETA_C)) + ry * math.cos(THETA_C)
    dtheta = abs(wrap_to_pi(psi - THETA_C))
    return 0.5 * min(abs(e_cross) / w, 1.0) + 0.5 * min(dtheta / h, 1.0)


TRAJ = [
    (np.array([1000.0, 600.0, 0.2, 6.7]), {}, False),
    (np.array([1500.0, 700.0, 0.1, 7.0]), {}, False),
    (np.array([3000.0, 900.0, 0.05, 8.0]), {}, True),
    (np.array([4800.0, 1090.0, 0.0, 3.0]), {"goal": True}, False),   # 近场 + 进门
    (np.array([4000.0, 1000.0, 0.0, 2.0]), {"time": True}, False),   # 远场 + 超时
]


def run_traj(rf):
    rf.reset(TRAJ[0][0])
    out = []
    for ego, tf, emer in TRAJ:
        total, parts = rf.step(ego, [], tf, emer)
        out.append((total, dict(parts)))
    return out


print("① c_dwell=0 与 baseline(默认) 逐位等价 bit-identical")
base = run_traj(_mk(0.0))
base_default = run_traj(RewardFunction(GOAL, INIT, goal_orientation=ORI, shaping_radius=500.0, gamma=0.99))
ok = all(abs(a[0] - b[0]) < 1e-12 and a[1].keys() == b[1].keys() and
         all(abs(a[1][k] - b[1][k]) < 1e-12 for k in a[1]) for a, b in zip(base, base_default))
chk(ok, "c_dwell=0.0 与 不传 dwell(默认) 全步 total/parts 逐位相同(<1e-12)")
chk(all(len(p[1]) == 5 for p in base), "c_dwell=0 不加键(parts 恒 5 键·无 dwell 泄漏)")

print("② c_dwell=0.5 近场每步 −c_dwell·g(s)·g 与独立手算逐位一致·其它项不动")
d5 = run_traj(_mk(0.5))
# 单点精确手算校验（θ_c=0·近场·横向+朝向都非零非饱和）
rf1 = _mk(0.5); rf1.reset(np.array([4950.0, 1140.0, 0.3, 3.0]))
_, p1 = rf1.step(np.array([4950.0, 1140.0, 0.3, 3.0]), [], {}, False)
rfb = _mk(0.0); rfb.reset(np.array([4950.0, 1140.0, 0.3, 3.0]))
_, pb = rfb.step(np.array([4950.0, 1140.0, 0.3, 3.0]), [], {}, False)
g_exp = _g_manual(4950.0, 1140.0, 0.3)
cost_exp = -0.5 * g_exp
chk(abs((p1["sparse"] - pb["sparse"]) - cost_exp) < 1e-9,
    f"单点 dwell 成本 = −c·g 与手算逐位一致（g={g_exp:.6f}·cost={cost_exp:.6f}·实测Δ={p1['sparse']-pb['sparse']:.6f}）")
ok_other = all(all(abs(c[1][k] - b[1][k]) < 1e-12 for k in ("colregs", "goal", "velocity", "deviate"))
               for c, b in zip(d5, base))
chk(ok_other, "其它项 colregs/goal/velocity/deviate 一字不动（dwell 只进 sparse·正交）")
# 正交叠加 well_B（dwell 仅依位置/朝向·与 PBRS 无关）
b_wb = run_traj(_mk(0.0, well_b=200.0)); d_wb = run_traj(_mk(0.5, well_b=200.0))
ok_orth = all(abs((d[1]["sparse"] - b[1]["sparse"]) - (dd[1]["sparse"] - bb[1]["sparse"])) < 1e-12
              for d, b, dd, bb in zip(d5, base, d_wb, b_wb))
chk(ok_orth, "well_B=200 下 dwell 成本与无 well_B 时逐位相同（与 PBRS shaping 正交）")

print("③ 🔴 θ_c 门控改点（唯一既有逻辑改点）")
rf_dw = _mk(0.5)
chk(rf_dw.theta_c is not None and abs(rf_dw.theta_c - 0.0) < 1e-12,
    "dwell-only(well_B=well_X=0·c_dwell>0) → θ_c 正确算出(=0.0·非 None)")
try:
    RewardFunction(GOAL, INIT, c_dwell=0.5); chk(False, "缺 goal_orientation 应 ValueError")
except ValueError as e:
    chk("c_dwell" in str(e), "c_dwell>0 缺 goal_orientation → ValueError（含 c_dwell 提示）")
try:
    _mk(0.5, shaping_radius=0.0)   # dwell 不用 shaping_radius → 不该牵连报错
    chk(True, "dwell-only + shaping_radius=0 → 不报错（门控重构正确·shaping_radius 仅 well_B/X 用）")
except ValueError:
    chk(False, "dwell-only + shaping_radius=0 误报错（门控重构错·牵连了 dwell）")

print("④ 边界/饱和")
# 远场（‖r‖>R_DWELL=250）→ dwell=0 逐位
far = np.array([1000.0, 600.0, 0.2, 6.7])
rf_f = _mk(0.5); rf_f.reset(far); _, pf = rf_f.step(far, [], {}, False)
rf_fb = _mk(0.0); rf_fb.reset(far); _, pfb = rf_fb.step(far, [], {}, False)
chk(pf["sparse"] == pfb["sparse"], "远场(‖r‖>250) dwell 成本 = 0（逐位·带外恒 0）")
# 双饱和：|e_cross|>90 且 |dθ|>0.52 → g=1 → 成本=−c_dwell 精确
sat = np.array([5000.0, 980.0, 1.5, 3.0])   # r=[0,120]→|e_cross|=120>90；|dθ|=1.5>0.52
rf_s = _mk(0.5); rf_s.reset(sat); _, ps = rf_s.step(sat, [], {}, False)
rf_sb = _mk(0.0); rf_sb.reset(sat); _, psb = rf_sb.step(sat, [], {}, False)
chk(abs((ps["sparse"] - psb["sparse"]) - (-0.5)) < 1e-12,
    "双饱和(|e_cross|>W & |dθ|>H) → g=1 → 成本=−c_dwell=−0.5（上界·farm 免疫）")

print("⑤ farm 免疫：全轨迹 dwell 贡献恒 ≤ 0（无正的每步项可刷）")
deltas = [d[1]["sparse"] - b[1]["sparse"] for d, b in zip(d5, base)]
chk(all(x <= 1e-12 for x in deltas), f"全步 dwell 贡献 ≤ 0（无正项·deltas={[round(x,4) for x in deltas]}）")

print("⑥ B_DWELL 终端锚")
d5b = run_traj(_mk(0.5, b=50.0))
# 仅 goal 步(索引3)相对 c_dwell-only 多 +50·其它步一字不动
diffs = [db[1]["sparse"] - d[1]["sparse"] for db, d in zip(d5b, d5)]
chk(abs(diffs[3] - 50.0) < 1e-9, f"goal 步 sparse 相对无 B_DWELL 多 +50（真入库锚·实测 {diffs[3]:.4f}）")
chk(all(abs(x) < 1e-12 for i, x in enumerate(diffs) if i != 3), "非 goal 步 B_DWELL 不施加（仅真入库）")
try:
    _mk(0.0, b=50.0); chk(False, "b_dwell>0 但 c_dwell=0 应 ValueError")
except ValueError:
    chk(True, "b_dwell>0 且 c_dwell=0 → ValueError（防 silent no-op）")

print("⑦ 离散硬拒 + config_conflict + 透传")
from trb_env.usv_shield import ShieldedUSVEnv, UnshieldedUSVEnv
for k in ("c_dwell", "w_dwell", "dwell_radius"):
    try:
        ShieldedUSVEnv(None, None, **{k: 0.5}); chk(False, f"ShieldedUSVEnv 应 TypeError 拒 {k}")
    except TypeError as e:
        chk(k in str(e), f"ShieldedUSVEnv({k}=0.5) → TypeError（连续臂专属·离散硬拒）")
    except Exception as e:
        chk(False, f"ShieldedUSVEnv 拒 {k} 但异常类型非 TypeError: {type(e).__name__}")
try:
    UnshieldedUSVEnv(None, None, c_dwell=0.5); chk(False, "UnshieldedUSVEnv 应 TypeError")
except TypeError:
    chk(True, "UnshieldedUSVEnv(c_dwell=0.5) → TypeError（自然拒·无 **kwargs）")
from run_step4e import config_conflict
rec0 = [{"steps": 5000000, "n_total": 200, "c_dwell": 0.0}]
chk(len(config_conflict(rec0, 5000000, 200, c_dwell=0.5)) > 0,
    "已存 c_dwell=0 记录 + 当前 c_dwell=0.5 → 冲突（不混进同一 jsonl）")
chk(len(config_conflict(rec0, 5000000, 200, c_dwell=0.0)) == 0,
    "c_dwell=0 vs c_dwell=0(缺字段归一化) → 无冲突（旧记录兼容）")
# 旧记录完全无 dwell 字段（真 pre-dwell jsonl）→ dwell off 无冲突
rec_old = [{"steps": 5000000, "n_total": 200}]
chk(len(config_conflict(rec_old, 5000000, 200)) == 0, "完全无 dwell 字段的旧记录 + dwell off → 无冲突（真 back-compat）")
import inspect
from trb_env.usv_continuous_shield import ContinuousProjectionEnv
sig = inspect.signature(ContinuousProjectionEnv.__init__).parameters
chk(all(k in sig for k in ("c_dwell", "w_dwell", "h_dwell", "dwell_radius", "b_dwell")),
    "ContinuousProjectionEnv.__init__ 有全部 5 个 dwell 形参")
src = inspect.getsource(ContinuousProjectionEnv.__init__)
chk("c_dwell=c_dwell" in src, "ContinuousProjectionEnv 把 c_dwell 透传给内层 USVEnv")

print("\n" + ("=" * 50))
print("✅ 全部通过" if N_FAIL == 0 else f"❌ {N_FAIL} 项失败")
sys.exit(1 if N_FAIL else 0)
