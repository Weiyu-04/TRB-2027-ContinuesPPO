"""
奖励模块冒烟测试 —— 断言全部手算（论文系数 / 闭式几何），fact-based。
跑：/opt/miniconda3/envs/trb/bin/python -B 代码/tests/test_usv_reward.py

clean 合成几何：goal_center=[1000,0], init=[0,0] → e_lat=[0,1]，可手算精确值。
每个分量单独隔离测（其余分量置 0）：不动→r_goal=0；v∈[2.5,8]→r_velocity=0；线上→r_deviate=0；
无他船→r_colregs=0；无终止/紧急→r_sparse=0。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from trb_env.usv_reward import (  # noqa: E402
    RewardFunction,
    _meyer_sector,
    ALPHA_X,
)

_fail = 0


def check(name, got, exp, tol=1e-4):
    global _fail
    got = float(got)
    exp = float(exp)
    ok = abs(got - exp) <= tol
    if not ok:
        _fail += 1
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: got={got:.6g} exp={exp:.6g}")


def check_raises(name, fn, exc=ValueError):
    global _fail
    try:
        fn()
    except exc:
        print(f"[PASS] {name}: 正确抛 {exc.__name__}")
        return
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] {name}: 抛了 {type(e).__name__} 非 {exc.__name__}")
        _fail += 1
        return
    print(f"[FAIL] {name}: 期望抛 {exc.__name__} 但没抛")
    _fail += 1


GOAL = [1000.0, 0.0]
INIT = [0.0, 0.0]
rf = RewardFunction(GOAL, INIT)   # 默认 v_pm_max=10, d_sense=8000, colregs_vel_scale=10

print("===== Meyer 3 扇区分类（φ 右舷正/左舷负）=====")
check("① _meyer_sector(0°)=starboard", _meyer_sector(0.0) == "starboard", True)
check("② _meyer_sector(90°)=starboard", _meyer_sector(np.radians(90)) == "starboard", True)
check("③ _meyer_sector(-90°)=port", _meyer_sector(np.radians(-90)) == "port", True)
check("④ _meyer_sector(120°)=stern", _meyer_sector(np.radians(120)) == "stern", True)
check("⑤ _meyer_sector(112.5°)=stern(边界)", _meyer_sector(np.radians(112.5)) == "stern", True)
check("⑥ _meyer_sector(-112.5°)=port(边界)", _meyer_sector(np.radians(-112.5)) == "port", True)

print("\n===== r_sparse（隔离：不动+v=5+线上+无他船）=====")
rf.reset([100, 0, 0, 5])
r, p = rf.step([100, 0, 0, 5], [], term_flags={"goal": True})
check("⑦ r_sparse goal → +50", p["sparse"], 50.0)
check("⑦ 其余分量为 0", p["goal"] + p["velocity"] + p["deviate"] + p["colregs"], 0.0)
rf.reset([100, 0, 0, 5])
r, p = rf.step([100, 0, 0, 5], [], term_flags={"collision": True})
check("⑧ r_sparse collision → -50", p["sparse"], -50.0)
rf.reset([100, 0, 0, 5])
r, p = rf.step([100, 0, 0, 5], [], term_flags={"time": True}, emergency_used=True)
check("⑨ r_sparse time+emergency → -25.5", p["sparse"], -25.5)

print("\n===== r_goal（typo 修正：前进→正）=====")
rf.reset([0, 0, 0, 5])                       # prev d=1000
r, p = rf.step([100, 0, 0, 5], [])           # now d=900 → 前进 100
check("⑩ 前进 100 → r_goal=+150", p["goal"], 150.0)
check("⑩ 其余分量 0（隔离）", p["sparse"] + p["velocity"] + p["deviate"] + p["colregs"], 0.0)
rf.reset([100, 0, 0, 5])                      # prev d=900
r, p = rf.step([0, 0, 0, 5], [])             # now d=1000 → 后退
check("⑪ 后退 → r_goal=-150", p["goal"], -150.0)

print("\n===== r_velocity =====")
rf.reset([100, 0, 0, 10]); _, p = rf.step([100, 0, 0, 10], [])
check("⑫ v=10>8 → -4", p["velocity"], -4.0)
rf.reset([100, 0, 0, 1]); _, p = rf.step([100, 0, 0, 1], [])
check("⑬ v=1<2.5 → -3", p["velocity"], -3.0)
rf.reset([100, 0, 0, 5]); _, p = rf.step([100, 0, 0, 5], [])
check("⑭ v=5 ∈ 区间 → 0", p["velocity"], 0.0)

print("\n===== r_deviate = c_deviate·min(|d_lat|, d_hull) =====")
rf.reset([100, 30, 0, 5]); _, p = rf.step([100, 30, 0, 5], [])
check("⑮ d_lat=30 → -0.03", p["deviate"], -0.03)
rf.reset([100, 3000, 0, 5]); _, p = rf.step([100, 3000, 0, 5], [])
check("⑯ d_lat=3000 → 截到 d_hull=2000 → -2.0", p["deviate"], -2.0)

print("\n===== r_colregs（Meyer 式26，v_y 归一化）=====")
# 正前方 approaching：obs[500,0] vel[-5,0]（向西=驶向本船）。手算：
#   d=500, v_y=+5, v_y_eff=0.5, φ=0→starboard, ζ_x=0.007 ζ_v=0.004, weight=0.5
#   exponent=(0.004·0.5−0.007)·500=−2.5 → r=−0.5·75·exp(−2.5)
rf.reset([0, 0, 0, 5]); _, p = rf.step([0, 0, 0, 5], [(1, [500, 0], [-5, 0])])
check("⑰ 正前 approaching d=500 → 手算", p["colregs"], -0.5 * ALPHA_X * np.exp(-2.5))
check("⑰ 其余分量 0（隔离）", p["sparse"] + p["goal"] + p["velocity"] + p["deviate"], 0.0)

# 🔴 关键：同场景 d=2000（原始式会 −6.7e12）→ 归一化后必须有界
rf.reset([0, 0, 0, 5]); _, p = rf.step([0, 0, 0, 5], [(1, [2000, 0], [-5, 0])])
exp_far = -0.5 * ALPHA_X * np.exp((0.004 * 0.5 - 0.007) * 2000)
check("⑱ d=2000 approaching → 有界(非爆炸)", p["colregs"], exp_far)
print(f"      （= {p['colregs']:.4g}，远非 −6.7e12，BLOCKER 已用归一化解决）")
if abs(p["colregs"]) >= 1.0:
    print("[FAIL] ⑱ r_colregs 在 d=2000 仍 |·|≥1，疑似未有界"); _fail += 1

# receding：obs[500,0] vel[+5,0]（向东=驶离）→ 惩罚≈0
rf.reset([0, 0, 0, 5]); _, p = rf.step([0, 0, 0, 5], [(1, [500, 0], [5, 0])])
check("⑲ 正前 receding → 惩罚≈0", p["colregs"], -0.5 * ALPHA_X * np.exp((0.05 * -0.5 - 0.007) * 500))
print(f"      （= {p['colregs']:.3e} ≈ 0，驶离不罚）")

# 超 d_sense 不计
rf.reset([0, 0, 0, 5]); _, p = rf.step([0, 0, 0, 5], [(1, [9000, 0], [-5, 0])])
check("⑳ 超 d_sense(8000) → r_colregs=0", p["colregs"], 0.0)

print("\n===== 不变量 + reset + fail-fast =====")
# total == 5 分量之和
rf.reset([0, 0, 0, 9])
r, p = rf.step([100, 40, 0, 9], [(1, [600, 0], [-3, 0])], term_flags={"goal": True}, emergency_used=True)
check("㉑ total == 分量之和", r, sum(p.values()))
assert all(np.isfinite(v) for v in p.values()), "分量含 NaN"
print("[PASS] ㉑ 分量全有限")

# reset 后第一步 prev=curr → r_goal=0
rf.reset([300, 0, 0, 5]); _, p = rf.step([300, 0, 0, 5], [])
check("㉒ reset 后不动 → r_goal=0", p["goal"], 0.0)

check_raises("㉓ step 前未 reset → RuntimeError",
             lambda: RewardFunction(GOAL, INIT).step([0, 0, 0, 5], []), RuntimeError)
check_raises("㉔ ego 含 NaN → 报错",
             lambda: (rf.reset([0, 0, 0, 5]), rf.step([0, 0, np.nan, 5], []))[1])
check_raises("㉕ 他船 velocity 含 NaN → 报错",
             lambda: (rf.reset([0, 0, 0, 5]), rf.step([0, 0, 0, 5], [(1, [500, 0], [np.nan, 0])]))[1])
check_raises("㉖ term_flags 拼错键 → 报错",
             lambda: (rf.reset([0, 0, 0, 5]), rf.step([0, 0, 0, 5], [], term_flags={"collison": True}))[1])
check_raises("㉗ init==goal → 报错", lambda: RewardFunction([0, 0], [0, 0]))

print("\n===== r_colregs 全域有界（独立复核 MAJOR 修复验证）=====")
# 高速他船（远超 v_pm_max=10）曾让 r_colregs 爆到 −7e12/−inf；指数钳 ≤0 后必有界
rf.reset([0, 0, 0, 5]); _, p = rf.step([0, 0, 0, 5], [(1, [2000, 0], [-50, 0])])  # v_obs=50≫10
check("㉘ 高速 v_obs=50,d=2000 → |r_colregs|≤α_x(曾 −7e12)", abs(p["colregs"]) <= ALPHA_X, True)
print(f"      （= {p['colregs']:.4g}）")
# 全域扫描 d×v_obs×扇区：全部有限 且 |·|≤α_x（单他船）
bad = 0
for d in [1, 500, 2000, 5000, 7999]:
    for vmag in [-300, -50, -10, 0, 10, 50, 300]:
        for ang in [0, 90, -90, 150, 180]:          # 覆盖 starboard/port/stern
            rad = np.radians(ang)
            obs_pos = [d * np.cos(rad), d * np.sin(rad)]
            rf.reset([0, 0, 0, 5])
            _, pp = rf.step([0, 0, 0, 5], [(1, obs_pos, [vmag, 0])])
            if not np.isfinite(pp["colregs"]) or abs(pp["colregs"]) > ALPHA_X + 1e-6:
                bad += 1
ok = bad == 0
if not ok:
    _fail += 1
print(f"[{'PASS' if ok else 'FAIL'}] ㉙ 全域 d×v_obs×扇区(175 组) r_colregs 有限且 |·|≤α_x（越界 {bad} 个）")
# reset 有限性（与 step 对称）
check_raises("㉚ reset 含 NaN → 报错", lambda: rf.reset([0, 0, np.nan, 5]))

print("\n===== colregs_weight 开关（Base/RR 区分，4d-②）=====")
# 同 (ego, 正前 approaching 他船) 下，weight=1(RR/默认) vs 0(Base) vs 0.5：只缩放 r_colregs、不动其余 4 分量
_OBS = [(1, [500, 0], [-5, 0])]; _EGO = [0, 0, 0, 5]
_RR_C = -0.5 * ALPHA_X * np.exp(-2.5)                                  # ⑰ 手算值（weight=1）
rf1 = RewardFunction(GOAL, INIT); rf1.reset(_EGO); t1, p1 = rf1.step(_EGO, _OBS)                       # RR（默认 1.0）
rf0 = RewardFunction(GOAL, INIT, colregs_weight=0.0); rf0.reset(_EGO); t0, p0 = rf0.step(_EGO, _OBS)   # Base
rfh = RewardFunction(GOAL, INIT, colregs_weight=0.5); rfh.reset(_EGO); th, ph = rfh.step(_EGO, _OBS)   # 半权
check("㉛ weight=1.0(默认/RR)：r_colregs 不变（=⑰ 手算）", p1["colregs"], _RR_C)
check("㉜ weight=0.0(Base)：r_colregs==0（论文 §VII p11：Base 无 r_colregs）", p0["colregs"], 0.0)
check("㉝ weight=0.5：r_colregs 线性缩放 0.5×", ph["colregs"], 0.5 * _RR_C)
check("㉞ Base total == RR total − r_colregs（仅关 r_colregs）", t0, t1 - p1["colregs"])
# ㉟ 用 4 分量均非零的 ego（前进+偏离参考线+越速 v=9+终止旗）→ 真区分"乘0"vs"未动其余分量"
#    （补 Agent 2：旧 ㉟ 的 ego 恰使 4 分量全 0 → 平凡守护）。同时断言 4 分量和非零（防 vacuous）。
_E2a, _E2b, _TF = [0, 0, 0, 9], [100, 30, 0, 9], {"goal": True}
r1c = RewardFunction(GOAL, INIT); r1c.reset(_E2a); _, q1 = r1c.step(_E2b, _OBS, term_flags=_TF)
r0c = RewardFunction(GOAL, INIT, colregs_weight=0.0); r0c.reset(_E2a); _, q0 = r0c.step(_E2b, _OBS, term_flags=_TF)
_s0 = q0["sparse"] + q0["goal"] + q0["velocity"] + q0["deviate"]
_s1 = q1["sparse"] + q1["goal"] + q1["velocity"] + q1["deviate"]
check("㉟ 其余 4 分量 weight 无关（4 分量均非零、真区分乘0 vs 未动）",
      float(abs(_s0 - _s1) < 1e-9 and abs(_s1) > 1e-6), 1.0)
check_raises("㊱ colregs_weight<0 → ValueError", lambda: RewardFunction(GOAL, INIT, colregs_weight=-1.0))

# ── 修法A 进门势 PBRS 永久回归（CleanR1 Q6 固化主窗口 7 项冒烟 + Q4f theta_c 跨±π·`03` L80-续9/续10）──
_ORI = (-0.17, 0.17); _WB = 200.0; _GAM = 0.99
def _rf_shape(ori=_ORI, **kw):
    return RewardFunction(GOAL, INIT, well_shaping_weight=_WB, goal_orientation=ori, gamma=_GAM, shaping_radius=500.0, **kw)
# (1) well_B=0 → bit-identical 现状（无 shape 键 + total=5分量和）
_r0 = RewardFunction(GOAL, INIT); _r0.reset([1000, 600, 0.0, 5.0]); _t0, _p0 = _r0.step([1010, 600, 0.0, 5.0], [])
check("㊲ well_B=0：parts 无 'shape' 键（bit-identical）", float("shape" not in _p0), 1.0)
check("㊲a well_B=0：total==5分量和", float(abs(_t0 - sum(_p0[k] for k in ("sparse", "colregs", "goal", "velocity", "deviate"))) < 1e-12), 1.0)
# (2) Ng telescoping（非终止）：Σγ^t·r_shape == γ^T·Φ_T − Φ_0 （<1e-9·策略不变）
_TRAJ = [[600, 0, 0.0, 5.0], [650, 0, 0.1, 5.0], [750, 0, 0.0, 5.0], [900, 0, 0.05, 5.0], [990, 0, 0.0, 5.0]]  # 趋近 GOAL=[1000,0]·全在 R_near=500 内（prox>0）
_r = _rf_shape(); _r.reset(_TRAJ[0]); _phi0 = _r._phi(_TRAJ[0]); _ds = 0.0
for _i in range(1, len(_TRAJ)):
    _, _pp = _r.step(_TRAJ[_i], []); _ds += (_GAM ** (_i - 1)) * _pp["shape"]
_phiT = _r._phi(_TRAJ[-1]); _Tn = len(_TRAJ) - 1
check("㊲b Ng telescoping：Σγ^t·r_shape==γ^T·Φ_T−Φ_0（<1e-9）", float(abs(_ds - ((_GAM ** _Tn) * _phiT - _phi0)) < 1e-9), 1.0)
# (3) terminated(goal) → Φ'=0：r_shape=−Φ_prev
_r2 = _rf_shape(); _r2.reset(_TRAJ[0])
for _i in range(1, len(_TRAJ) - 1):
    _r2.step(_TRAJ[_i], [])
_pp2 = _r2._prev_phi; _, _pg = _r2.step(_TRAJ[-1], [], term_flags={"goal": True})
check("㊲c terminated(goal)：r_shape=−Φ_prev", float(abs(_pg["shape"] - (-_pp2)) < 1e-9), 1.0)
# (4) truncated(time) → Φ' bootstrap 真实·≠−Φ_prev
_r3 = _rf_shape(); _r3.reset(_TRAJ[0])
for _i in range(1, len(_TRAJ) - 1):
    _r3.step(_TRAJ[_i], [])
_pp3 = _r3._prev_phi; _pc3 = _r3._phi(_TRAJ[-1]); _, _pt = _r3.step(_TRAJ[-1], [], term_flags={"time": True})
check("㊲d truncated(time)：Φ' bootstrap 真实·r_shape=γΦ_cur−Φ_prev≠−Φ_prev",
      float(abs(_pt["shape"] - (_GAM * _pc3 - _pp3)) < 1e-9 and abs(_pt["shape"] + _pp3) > 1e-6), 1.0)
# (5) align 方向 + prox 远场=0
_r4 = _rf_shape()
check("㊲e align：对齐 Φ>反向 Φ≥0（cos 平滑）", float(_r4._phi([990, 0, 0.0, 5.0]) > _r4._phi([990, 0, 3.1416, 5.0]) >= 0), 1.0)
check("㊲f prox：远场(d>R_near)Φ=0（不干扰避碰）", float(_r4._phi([0, 0, 0.0, 5.0]) == 0.0), 1.0)
# (6) Q4f theta_c 跨±π = 有向弧心（非算术中点 0）
_rc = _rf_shape(ori=(3.0, -3.0))
check("㊲g theta_c 跨±π：有向弧心≈±π（非算术中点 0·CleanR1 Q4f）", float(abs(abs(_rc.theta_c) - np.pi) < 0.01), 1.0)
# (7) raise 路径
check_raises("㊲h well_B>0 缺 goal_orientation → ValueError", lambda: RewardFunction(GOAL, INIT, well_shaping_weight=200.0))
check_raises("㊲i well_B>0 且 shaping_radius<=0 → ValueError",
             lambda: RewardFunction(GOAL, INIT, well_shaping_weight=200.0, goal_orientation=_ORI, shaping_radius=0.0))

# ── 对症 横向进带势 Φ_xtrack PBRS 永久回归（`03` L88·镜像修法A 范式·θ_c=0→n_perp=(0,1)→e_cross=−pos_y）──
_WX = 200.0; _RLAT = 80.0
def _rf_x(**kw):
    return RewardFunction(GOAL, INIT, well_shaping_weight=0.0, xtrack_weight=_WX, xtrack_radius=_RLAT,
                          goal_orientation=_ORI, gamma=_GAM, shaping_radius=500.0, **kw)
# (1) well_X=0 → bit-identical（well_B=200/well_X=0 == well_B=200 单独·parts 无 'shape_xtrack' 键）
_rb = _rf_shape(); _rb.reset([960, 34, 0.0, 3.0]); _tb, _pb = _rb.step([965, 28, 0.05, 3.2], [])
_rx = _rf_shape(xtrack_weight=0.0, xtrack_radius=80.0); _rx.reset([960, 34, 0.0, 3.0]); _tx, _px = _rx.step([965, 28, 0.05, 3.2], [])
check("㊳ well_X=0：reward 逐位==well_B单独 + parts 无 'shape_xtrack' 键（bit-identical）",
      float(abs(_tb - _tx) < 1e-15 and "shape_xtrack" not in _px), 1.0)
# (2) Φ_xtrack telescoping（terminated→−Φ_x(s0)·<1e-9·策略不变）
_TX = [[900, 40, 0.0, 3.0], [940, 20, 0.0, 3.0], [980, 5, 0.0, 3.0], [1000, 0, 0.0, 3.0]]   # 趋近 GOAL=[1000,0]·R_near 内
_r = _rf_x(); _r.reset(_TX[0]); _px0 = _r._phi_xtrack(_TX[0]); _acc = 0.0
for _i in range(1, len(_TX)):
    _, _pp = _r.step(_TX[_i], [], term_flags={"goal": (_i == len(_TX) - 1)}); _acc += (_GAM ** (_i - 1)) * _pp["shape"]
check("㊳a Φ_xtrack telescoping：Σγ^t·shape==−Φ_x(s0)（terminated·<1e-9·策略不变）", float(abs(_acc - (-_px0)) < 1e-9), 1.0)
# (3) e_cross 方向：往中心线 shape_xtrack>0·往外<0（横向进带梯度·对症 94% 位置门 miss）
_re = _rf_x(); _re.reset([1000, 40, 0.0, 3.0]); _, _pin = _re.step([1000, 30, 0.0, 3.0], [])
_re2 = _rf_x(); _re2.reset([1000, 40, 0.0, 3.0]); _, _pout = _re2.step([1000, 50, 0.0, 3.0], [])
check("㊳b e_cross 方向：往中心线 shape_xtrack>0·往外<0（对症横向梯度）",
      float(_pin["shape_xtrack"] > 0 and _pout["shape_xtrack"] < 0), 1.0)
# (4) prox_lat 线性：|e_cross| 越大 Φ_x 越小·|e_cross|>=R_lat 归 0（带外恒定拉回·复审 L88 选线性非高斯）
_rl = _rf_x()
check("㊳c prox_lat 线性：Φ_x(|e|=40)>Φ_x(|e|=70)>0 且 Φ_x(|e|>=R_lat=80)=0（带外恒定拉回）",
      float(_rl._phi_xtrack([1000, 40, 0.0, 3.0]) > _rl._phi_xtrack([1000, 70, 0.0, 3.0]) > 0
            and _rl._phi_xtrack([1000, 80, 0.0, 3.0]) == 0.0), 1.0)
# (5) well_X=0 → _phi_xtrack 恒 0（短路·无副作用）+ raise 路径
check("㊳d well_X=0：_phi_xtrack 恒 0（短路）", float(RewardFunction(GOAL, INIT)._phi_xtrack([1000, 40, 0.0, 3.0]) == 0.0), 1.0)
check_raises("㊳e well_X>0 缺 goal_orientation → ValueError", lambda: RewardFunction(GOAL, INIT, xtrack_weight=200.0))
check_raises("㊳f well_X>0 且 xtrack_radius<=0 → ValueError",
             lambda: RewardFunction(GOAL, INIT, xtrack_weight=200.0, goal_orientation=_ORI, xtrack_radius=0.0))
# (6) e_cross 走 θ_c 法向·独立于 self.e_lat(init→goal 横向)·未污染（复审 L88 BLOCKER：不复用 e_lat）
check("㊳g e_cross 内联算(θ_c 法向)·不复用/不污染 self.e_lat(init→goal 横向)",
      float(hasattr(_rl, "e_lat") and not hasattr(_rl, "e_cross")), 1.0)
# (7) 两势加性（PBRS 线性叠加·复审 L88 实算 diff=0）：combined shape == well_B单独 + well_X单独·shape_xtrack==well_X单独（可消融剥离）
_STEP_A = [1000, 40, 0.0, 3.0]; _STEP_B = [1000, 30, 0.0, 3.0]
_ra = RewardFunction(GOAL, INIT, well_shaping_weight=_WB, goal_orientation=_ORI, gamma=_GAM, shaping_radius=500.0)   # well_B 单独
_ra.reset(_STEP_A); _, _pa = _ra.step(_STEP_B, [])
_rxo = _rf_x(); _rxo.reset(_STEP_A); _, _pxo = _rxo.step(_STEP_B, [])                                               # well_X 单独
_rab = RewardFunction(GOAL, INIT, well_shaping_weight=_WB, xtrack_weight=_WX, xtrack_radius=_RLAT,                  # 两势叠加
                      goal_orientation=_ORI, gamma=_GAM, shaping_radius=500.0)
_rab.reset(_STEP_A); _, _pab = _rab.step(_STEP_B, [])
check("㊳h 两势加性：combined shape==well_B单独+well_X单独(PBRS 线性·<1e-9) + shape_xtrack==well_X单独(可剥离消融)",
      float(abs(_pab["shape"] - (_pa["shape"] + _pxo["shape"])) < 1e-9
            and abs(_pab["shape_xtrack"] - _pxo["shape"]) < 1e-9
            and _pxo["shape"] > 1e-6), 1.0)

print("\n" + ("✅ 全部 PASS" if _fail == 0 else f"❌ {_fail} 项 FAIL"))
sys.exit(1 if _fail else 0)
