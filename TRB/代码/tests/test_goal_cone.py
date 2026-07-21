#!/usr/bin/env python3
"""统一态势盾·ρ0 朝目标锥安全集 冒烟测试（方案①·2026-07-04·连续投影盾专属）。

验：
  T1 全角域非空     : e∈{−179..179°}·锥开(Φ=60°)·各速度·_goal_cone_interval 恒 w_lo≤w_hi ∧ a_lo≤a_hi。
  T2 锥外单调减     : |e|>Φ 时·端点 ω 施加后 err_next=|wrap(θ_goal−(θ+ω·dt))| 严格 < |e|（几个 e）。
  T3 a 保速         : v=0→a_lo>0（强制加速离停船）·v=v_max→a_hi≤0（顶速禁加）。
  T4 默认关 bit-identical: goal_cone_half=None → colregs_interval ρ0 返回全箱(=改前)·goal_cone_action 返回 None·
                      端到端一步 u_safe==u_desired(s_obs=None 时)逐位。
  T5 短路不推状态机 : s_obs=None 锥开时·goal_cone_action 后 proj 的 _sc.rho 不被推进(与锥关同)·_prev_rho 不污染。
运行：cd 代码 && /opt/miniconda3/envs/trb/bin/python tests/test_goal_cone.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from trb_env.usv_colregs import VesselState, wrap_to_pi, RHO_NO_CONFLICT
from trb_env.usv_projection import ContinuousColregsProjection

A_MAX, W_MAX = 0.048, 0.018      # RL 正常操作箱（Krasowski，见 usv_env A_NORMAL_*）
DT = 10.0                         # 决策步长
V_MAX = 9.5
PHI = np.deg2rad(60.0)            # 锥半角 60°
V_FLOOR = 2.0
N_FAIL = 0


def chk(cond, msg):
    global N_FAIL
    print(("  [OK] " if cond else "  [XX] ") + msg)
    if not cond:
        N_FAIL += 1


def mk_proj(cone=True):
    kw = dict(omega_turn=min(0.008727, W_MAX))
    if cone:
        kw.update(goal_cone_half=PHI, goal_v_floor=V_FLOOR, v_max=V_MAX)
    return ContinuousColregsProjection(A_MAX, W_MAX, **kw)


def ego_with_e(e, v=5.0):
    """构造使 wrap(θ_goal−θ)=e 的 ego（goal 固定在 +x 轴 → θ_goal=0 → θ=−e）。"""
    return VesselState(position=np.array([0.0, 0.0]), orientation=float(-e), velocity=float(v), length=175.0)


GOAL = np.array([1000.0, 0.0])   # θ_goal=atan2(0, 1000)=0


print("T1 全角域非空（Φ=60°·各速度·w_lo≤w_hi ∧ a_lo≤a_hi 恒成立）")
proj = mk_proj(cone=True)
proj.set_goal(GOAL)
empties = 0
min_wgap = 1e9
for edeg in range(-179, 180):
    e = np.deg2rad(edeg)
    for v in (0.0, 0.3, 2.0, 5.0, 9.5):
        (a_lo, a_hi), (w_lo, w_hi) = proj._goal_cone_interval(ego_with_e(e, v), DT)
        if w_lo > w_hi + 1e-15 or a_lo > a_hi + 1e-15:
            empties += 1
        min_wgap = min(min_wgap, w_hi - w_lo)
chk(empties == 0, f"359°×5 速度 = 1795 样本·空区间数 = {empties}（须 0；min ω-gap={min_wgap:.3e}）")

print("T2 锥外单调减（|e|>Φ·端点 ω 施加后航向误差严格 <|e|）")
# ⚠️ 必须 |e|>Φ 才是锥外（forced-turn 区制）。spec 例 e=40/90/150° 用 Φ=30° 锥（则均满足 |e|>Φ=30°）。
#   （锥内 |e|≤Φ 无强制单调性：区间允许在锥内两侧微调、含增大误差·合设计意图。）
proj_t2 = mk_proj(cone=True)
proj_t2.__init__(A_MAX, W_MAX, omega_turn=min(0.008727, W_MAX),
                 goal_cone_half=np.deg2rad(30.0), goal_v_floor=V_FLOOR, v_max=V_MAX)
proj_t2.set_goal(GOAL)
PHI_T2 = np.deg2rad(30.0)
mono_fail = 0
for edeg in (-150, -90, -40, 40, 90, 150):     # 全在锥外(|e|>Φ=30°)
    e = np.deg2rad(edeg)
    assert abs(e) > PHI_T2, f"测试自洽：e={edeg}° 须 >Φ=30°"
    theta = float(-e)                           # θ_goal=0 → θ=−e
    (a_lo, a_hi), (w_lo, w_hi) = proj_t2._goal_cone_interval(ego_with_e(e), DT)
    for w_end in (w_lo, w_hi):
        theta_next = wrap_to_pi(theta + w_end * DT)
        err_next = abs(wrap_to_pi(0.0 - theta_next))
        if not (err_next < abs(e) - 1e-12):
            mono_fail += 1
            print(f"    e={edeg}° w={w_end:.5f} err_next={np.rad2deg(err_next):.2f}° !< |e|={abs(edeg)}°")
chk(mono_fail == 0, "6 个锥外 e(|e|>30°)·两端点 ω 施加后 err_next 严格 <|e|（朝目标·无越界回摆）")

print("T3 a 保速（v=0→a_lo>0 强制加速·v=v_max→a_hi≤0 顶速禁加）")
(a_lo0, a_hi0), _ = proj._goal_cone_interval(ego_with_e(0.0, v=0.0), DT)
(a_loM, a_hiM), _ = proj._goal_cone_interval(ego_with_e(0.0, v=V_MAX), DT)
chk(a_lo0 > 0.0, f"v=0 → a_lo={a_lo0:.5f} > 0（强制离停船·且 ≤a_hi={a_hi0:.5f} 非空）")
chk(a_lo0 <= a_hi0 + 1e-15, f"v=0 → a_lo≤a_hi（单步够不到 floor 时夹到满舵加速·非空区间）")
chk(a_hiM <= 0.0 + 1e-15, f"v=v_max → a_hi={a_hiM:.5f} ≤ 0（顶速禁加）")

print("T4 默认关（goal_cone_half=None）bit-identical")
proj_off = mk_proj(cone=False)
proj_off.set_goal(GOAL)      # 即使注入 goal，锥关也不生效
# (a) colregs_interval ρ0 返回全箱（=改前 pass）
ego = ego_with_e(np.deg2rad(120.0), v=3.0)     # 锥外角度·若锥生效会被约束
(a_lo, a_hi), (w_lo, w_hi), gwd, nfb = proj_off.colregs_interval(RHO_NO_CONFLICT, ego, None, dt=DT)
full_box = (abs(a_lo + A_MAX) < 1e-15 and abs(a_hi - A_MAX) < 1e-15
            and abs(w_lo + W_MAX) < 1e-15 and abs(w_hi - W_MAX) < 1e-15)
chk(full_box and not nfb and gwd is None,
    "锥关·colregs_interval(ρ0, dt=DT) 返回全箱 [±a_max]×[±w_max]·no fallback（=改前 pass 逐位）")
# (b) goal_cone_action 返回 None
chk(proj_off.goal_cone_action(ego, np.array([0.02, 0.005]), DT) is None,
    "锥关·goal_cone_action 返回 None（caller 保持 u_desired）")
# (b') 锥开但 goal=None 也返回 None
proj_nogoal = mk_proj(cone=True)   # 未 set_goal → goal=None
chk(proj_nogoal.goal_cone_action(ego, np.array([0.02, 0.005]), DT) is None,
    "锥开但 goal=None·goal_cone_action 返回 None（无目标=锥不生效）")
# (c) 锥开·colregs_interval 也须在 dt=None 时保持全箱（project() legacy 路径 bit-identical）
proj_on = mk_proj(cone=True)
proj_on.set_goal(GOAL)
(a_lo2, a_hi2), (w_lo2, w_hi2), _, nfb2 = proj_on.colregs_interval(RHO_NO_CONFLICT, ego, None, dt=None)
full_box2 = (abs(a_lo2 + A_MAX) < 1e-15 and abs(a_hi2 - A_MAX) < 1e-15
             and abs(w_lo2 + W_MAX) < 1e-15 and abs(w_hi2 - W_MAX) < 1e-15)
chk(full_box2 and not nfb2, "锥开但 dt=None（legacy project 路径）·colregs_interval ρ0 仍全箱（bit-identical 守护）")

print("T4' 端到端 s_obs=None·锥关时 u_safe==u_desired 逐位")
# 用真实场景 env 端到端验（锥关默认 → ContinuousProjectionEnv 无 goal_cone_half kwarg = None）。
# 场景夹具同 test_usv_continuous_shield.py（/tmp/trb_T0.xml·缺则联网下载·离线则整块 SKIP）。
_T0 = "/tmp/trb_T0.xml"
if not os.path.exists(_T0):
    try:
        import urllib.request
        _url = ("https://gitlab.lrz.de/tum-cps/commonocean-scenarios/-/raw/main/scenarios/"
                "HandcraftedTwoVesselEncounters_01_24/ZAM_AAA-1_20240121_T-0.xml")
        urllib.request.urlretrieve(_url, _T0)
    except Exception as _e:
        print(f"    [SKIP 端到端块：/tmp/trb_T0.xml 不在且离线下载失败（{_e}）]")
try:
    from commonocean.common.file_reader import CommonOceanFileReader
    from trb_env.usv_continuous_shield import ContinuousProjectionEnv
    sc, _pp = CommonOceanFileReader(_T0).open()
    pp = list(_pp.planning_problem_dict.values())[0]
    env_off = ContinuousProjectionEnv(sc, pp)     # 默认锥关
    env_off.reset(seed=0)
    # 直接测短路：把 obstacle 拿掉 → s_obs=None 分支
    env_off._obstacles = []
    u_des = np.array([0.03, -0.007])
    obs, r, term, trunc, info = env_off.step(u_des)
    ua = info["u_applied"]
    bit_id = np.array_equal(np.asarray(ua, float), np.clip(u_des, [-A_MAX, -W_MAX], [A_MAX, W_MAX]).astype(float))
    chk(info["source"] == "no_obstacle", f"锥关·s_obs=None → source='no_obstacle'（得 {info['source']!r}）")
    chk(bit_id, "锥关·s_obs=None → u_applied==clip(u_desired)（锥不介入·逐位）")
    _scenario_ok = True
except Exception as ex:
    print(f"    [跳过场景端到端 T4'/T5：{type(ex).__name__}: {ex}]")
    _scenario_ok = False

print("T5 短路不推状态机（s_obs=None 锥开·goal_cone_action 后 _sc.rho / _prev_rho 不污染）")
proj_t5 = mk_proj(cone=True)
proj_t5.set_goal(GOAL)
proj_t5.reset()
rho_before = proj_t5._sc.rho
prev_before = proj_t5._prev_rho
ua = proj_t5.goal_cone_action(ego_with_e(np.deg2rad(100.0), v=3.0), np.array([0.02, 0.005]), DT)
chk(ua is not None, "锥开·goal_cone_action 返回动作（非 None）")
chk(proj_t5._sc.rho == rho_before, f"goal_cone_action 后 _sc.rho 未推进（{rho_before}→{proj_t5._sc.rho}·须不变）")
chk(proj_t5._prev_rho == prev_before, "goal_cone_action 后 _prev_rho 未污染（不碰兜底边沿追踪）")
# 端到端：s_obs=None 锥开时 env._source='goal_cone' 但状态机 rho 不推进
if _scenario_ok:
    try:
        env_on = ContinuousProjectionEnv(sc, pp, goal_cone_half=PHI, goal_v_floor=V_FLOOR)
        env_on.reset(seed=0)
        env_on._obstacles = []                 # 强制 s_obs=None 短路
        sc_rho_before = env_on.proj._sc.rho
        prev_before2 = env_on.proj._prev_rho
        obs, r, term, trunc, info = env_on.step(np.array([0.03, -0.007]))
        chk(info["source"] == "goal_cone", f"锥开·s_obs=None → source='goal_cone'（得 {info['source']!r}）")
        chk(env_on.proj._sc.rho == sc_rho_before, "锥开短路·env.proj._sc.rho 未推进（不推状态机）")
        chk(env_on.proj._prev_rho == prev_before2, "锥开短路·env.proj._prev_rho 未污染")
    except Exception as ex:
        print(f"    [跳过锥开端到端 T5：{type(ex).__name__}: {ex}]")

print("\n" + ("=" * 50))
print("[全部通过]" if N_FAIL == 0 else f"[{N_FAIL} 项失败]")
sys.exit(1 if N_FAIL else 0)
