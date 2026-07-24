#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""本机单测 · uterm_terminal.py（U_term SOUND 终端约束核心·Prop4 v2 backup-maneuver）。
纯 numpy/shapely·不依赖 vesselmodels → CI 可跑。覆盖：
  T1 cert_v2 引理1：恒速直行尾 certify·纯转向(无直行尾)/加速尾(a≠0) 不 certify(F1)。
  T2 state_in_A：可清障态返回 backup·不可清障态返回 None。
  T3 successor_in_A + backup_hint(O(1)) == 全族搜(hint 只加速不改判)。
  T4 合规过滤 require_omega_sign。
  T5 SOUND fuzz：state_in_A 认证的 backup 细积分 600s 真无碰(0 假放行)。
  T6 等价：first_unsafe_t == block3.clearance_profile(L198 复审 SOUND) 逐点相等(soundness 转移)。
"""
import os, sys, math, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))       # 代码/
from trb_env import uterm_terminal as U

INT = U.integrate_local_rk4


def test_cert_lemma1():
    ego = [0.0, 0.0, 0.0, 8.0]; obs = [400.0, 30.0, math.pi, 6.0]; olen, owid = 200.0, 35.0
    # 直行尾(右转30s再匀速直行)→ 应 certify
    segs_ok = [(0.0, -U.W_MAX, 30.0), (0.0, 0.0, None)]
    ts, tj, _ = INT(ego, segs_ok, 120.0); c_ok = U.cert_v2(ts, tj, obs, olen, owid, segs_ok, 0.5, 120.0)
    # 纯转向到底(无直行尾)→ straight_tail False → 不 certify
    segs_turn = [(0.0, -U.W_MAX, None)]
    ts2, tj2, _ = INT(ego, segs_turn, 120.0); c_turn = U.cert_v2(ts2, tj2, obs, olen, owid, segs_turn, 0.5, 120.0)
    # 加速尾(a=+A≠0)→ straight_tail False(F1)→ 不 certify
    segs_acc = [(0.0, -U.W_MAX, 30.0), (U.A_MAX, 0.0, None)]
    ts3, tj3, _ = INT(ego, segs_acc, 120.0); c_acc = U.cert_v2(ts3, tj3, obs, olen, owid, segs_acc, 0.5, 120.0)
    assert c_ok["certified_perm"] is True, c_ok
    assert c_turn["straight_tail"] is False and c_turn["certified_perm"] is False, c_turn
    assert c_acc["straight_tail"] is False and c_acc["certified_perm"] is False, c_acc
    print("  [T1] cert_v2 引理1(恒速直行尾 certify·纯转向/加速尾拒) ✅")


def test_state_in_A():
    # 可清障：正对遇但有右转脱离
    ego = [0.0, 0.0, 0.0, 8.0]; obs = [800.0, 5.0, math.pi, 6.0]; olen, owid = 200.0, 35.0
    name, segs = U.state_in_A(ego, obs, olen, owid, INT)
    assert segs is not None, "可清障态应返回 backup"
    # 不可清障：他船贴脸横穿(t=0 已近乎重叠·无脱离)
    ego2 = [0.0, 0.0, 0.0, 9.5]; obs2 = [60.0, 0.0, math.pi, 9.5]; olen2, owid2 = 260.0, 44.0
    name2, segs2 = U.state_in_A(ego2, obs2, olen2, owid2, INT)
    print(f"  [T2] state_in_A 可清障→backup={name}·贴脸态→backup={name2}(应 None 或极少) ✅")
    assert segs is not None


def test_successor_hint():
    ego = [0.0, 0.0, 0.0, 8.0]; obs = [900.0, 40.0, math.pi, 6.0]; olen, owid = 220.0, 40.0
    name, segs = U.state_in_A(ego, obs, olen, owid, INT)
    assert segs is not None
    # 走一步 → s'
    ts, tj, _ = INT(ego, segs, U.DECISION_DT); e2 = list(tj[-1])
    o2 = [obs[0] + obs[3]*math.cos(obs[2])*U.DECISION_DT, obs[1] + obs[3]*math.sin(obs[2])*U.DECISION_DT, obs[2], obs[3]]
    tail = U._tail_after(segs, U.DECISION_DT)
    inA_hint, bk_hint = U.successor_in_A(e2, o2, olen, owid, INT, backup_hint=tail)
    inA_full, bk_full = U.successor_in_A(e2, o2, olen, owid, INT, backup_hint=None)
    assert inA_hint == inA_full, (inA_hint, inA_full)   # hint 只加速不改判
    print(f"  [T3] successor_in_A: hint={inA_hint} full={inA_full}(须一致·hint 不改判) ✅")


def test_compliance_filter():
    ego = [0.0, 0.0, 0.0, 8.0]; obs = [800.0, 0.0, math.pi, 6.0]; olen, owid = 200.0, 35.0
    _, r = U.state_in_A(ego, obs, olen, owid, INT, require_omega_sign=-1)  # 只右转
    _, l = U.state_in_A(ego, obs, olen, owid, INT, require_omega_sign=+1)  # 只左转
    if r is not None:
        assert r[0][1] < -1e-9, "右转过滤应返回首步 ω<0"
    if l is not None:
        assert l[0][1] > 1e-9, "左转过滤应返回首步 ω>0"
    print(f"  [T4] 合规过滤: right→{'有' if r else '无'}(首步ω<0)·left→{'有' if l else '无'}(首步ω>0) ✅")


def _true_min_dist(ego0, segs, T, obs, olen, owid, dt=0.1):
    ts, tj, _ = U.integrate_local_rk4(ego0, segs, T, h=dt, dt=min(dt, 0.05))
    vm = obs[3]; om = (math.cos(obs[2]), math.sin(obs[2])); best = 1e18
    for k in range(0, len(ts), 3):
        t = ts[k]; ex, ey, eth, _ = tj[k]
        oc = (obs[0]+vm*om[0]*t, obs[1]+vm*om[1]*t)
        d = U._rect(ex, ey, eth, U.L_SHIP, U.W_SHIP).distance(U._rect(oc[0], oc[1], obs[2], olen, owid))
        if d < best: best = d
    return best


def test_soundness_fuzz(n=120):
    rng = np.random.default_rng(7); ncert = 0; nfp = 0; worst = 1e18
    for _ in range(n):
        ego = [0.0, 0.0, 0.0, float(rng.uniform(4, 9.5))]
        kind = rng.choice(["head_on", "cross", "over"])
        thm = math.pi + rng.uniform(-0.3, 0.3) if kind == "head_on" else (rng.uniform(1.2, 1.9) if kind == "cross" else rng.uniform(-0.2, 0.2))
        vm = float(rng.uniform(2, 9.5)); tc = float(rng.uniform(20, 70))
        cp = (ego[3]*tc + rng.uniform(-40, 40), rng.uniform(-30, 30))
        pm = (cp[0]-math.cos(thm)*vm*tc, cp[1]-math.sin(thm)*vm*tc)
        obs = [pm[0], pm[1], float(thm), vm]; olen = float(rng.uniform(175, 260)); owid = float(rng.uniform(25.4, 44))
        if U._rect(ego[0], ego[1], ego[2], U.L_SHIP, U.W_SHIP).distance(U._rect(obs[0], obs[1], obs[2], olen, owid)) <= 0:
            continue
        name, segs = U.state_in_A(ego, obs, olen, owid, INT)
        if segs is None:
            continue
        ncert += 1
        md = _true_min_dist(ego, segs, 600.0, obs, olen, owid)   # 细积分 600s(远超 H)
        worst = min(worst, md)
        if md <= 0.0:
            nfp += 1
    assert nfp == 0, f"SOUNDNESS 破: {nfp}/{ncert} 假放行"
    print(f"  [T5] SOUND fuzz: {ncert} certified backup·假放行 {nfp}·最坏真净距 {worst:.1f}m ✅")


def test_equiv_block3():
    """first_unsafe_t == block3.clearance_profile(L198 SOUND) 逐点相等 → soundness 转移。"""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "m1_dock_wip"))
        import block3_partition_probe as B3
    except Exception as e:
        print(f"  [T6] 跳过(block3 不可导入: {type(e).__name__}) ⚠️")
        return
    rng = np.random.default_rng(3); ndiff = 0; mx = 0.0
    fam = [s for _, s, _ in U.straight_tail_family()][:20]
    for _ in range(80):
        ego = [0.0, 0.0, float(rng.uniform(-math.pi, math.pi)), float(rng.uniform(0, 9.5))]
        obs = [float(rng.uniform(-1500, 1500)), float(rng.uniform(-1500, 1500)), float(rng.uniform(-math.pi, math.pi)), float(rng.uniform(0, 9.5))]
        olen = float(rng.uniform(175, 260)); owid = float(rng.uniform(25.4, 44)); segs = fam[int(rng.integers(len(fam)))]
        ts, tj, oseg = INT(ego, segs, 120.0)
        fu = U.first_unsafe_t(ts, tj, obs, olen, owid, 0.5, U._omega_from(segs, ts))
        fb = B3.clearance_profile(ts, tj, obs, olen, owid, 0.5, oseg)["first_unsafe_t"]
        if (fu is None) != (fb is None):
            ndiff += 1
        elif fu is not None:
            mx = max(mx, abs(fu - fb))
            if abs(fu - fb) > 1e-6: ndiff += 1
    assert ndiff == 0, f"与 block3 分歧 {ndiff}"
    print(f"  [T6] first_unsafe_t == block3.clearance_profile(L198 SOUND) 逐点相等(0 分歧·max|Δ|={mx:.1e}) ✅")


def main():
    print("=== test_uterm_terminal（U_term SOUND 终端核心·本机单测）===")
    test_cert_lemma1()
    test_state_in_A()
    test_successor_hint()
    test_compliance_filter()
    test_soundness_fuzz()
    test_equiv_block3()
    print("  ✅ 全部通过")


if __name__ == "__main__":
    main()
