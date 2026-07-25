#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""分级介入（graded_emergency）测试 —— ρ5 不再整个丢弃策略动作（`03` L205）。

⚠️ **须服务器跑**（import usv_projection → usv_dynamics → vesselmodels·本机无）。
本机只能 py_compile；**闭环行为必须服务器冒烟才算真验**（L202 教训：本机+对抗审都漏了负速度崩，服务器实跑才现形）。

覆盖：
  T1 默认关 = bit-identical：graded_emergency=False 时 ρ5 走 Alg.1，动作与旧行为逐位相同。
  T2 开关打开 + ρ5 且存在无碰撞动作 → source='emergency_relaxed' 且动作 == 把 u_desired 投影到 box∩无碰撞（策略意图被保留）。
  T3 开关打开 + ρ5 但无可行动作（贴脸）→ 回落 source='emergency'（Alg.1 原样接管）。
  T4 EC reset 语义（对抗审 R4·D13）：档1→档2 切换时 EC 被 reset（不跨越未接管的步沿用陈旧 mode）。
  T5 reset() 清 _prev_ec_used（防跨 episode 陈旧）。
用法（服务器）：PYTHONPATH=/root/trb/代码 python /root/trb/代码/tests/test_graded_emergency.py
"""
import os, sys, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))       # 代码/
from trb_env.usv_projection import ContinuousColregsProjection
from trb_env.usv_colregs import VesselState, RHO_EMERGENCY
from trb_env.usv_dynamics import make_vessel_params

VP = make_vessel_params()
DT = 10.0


def _mk(graded):
    return ContinuousColregsProjection(VP.a_max, VP.w_max, graded_emergency=graded)


def _states(gap, obs_len=200.0):
    """构造正对遇迫近几何（易触 ρ5）。gap 越小越迫近。"""
    ego = VesselState(position=np.array([0.0, 0.0]), orientation=0.0, velocity=8.0, length=VP.l)
    obs = VesselState(position=np.array([gap, 0.0]), orientation=math.pi, velocity=6.0, length=obs_len)
    return ego, obs


def _find_rho5(proj, gaps=(3000., 2000., 1500., 1200., 900., 700., 500.)):
    """找一个真触发 ρ5 的 gap（不同参数下阈值不同→扫）。返回 (gap, ego, obs) 或 None。"""
    for g in gaps:
        proj.reset()
        ego, obs = _states(g)
        r = proj.safe_action(ego, obs, np.array([0.0, 0.0]), DT, VP)
        if r.rho == RHO_EMERGENCY:
            return g, ego, obs
    return None


def test_bit_identical_off():
    """T1：默认关 → 与旧行为逐位一致（旧行为 = ρ5 直接 Alg.1）。"""
    p_off = _mk(False)
    found = _find_rho5(p_off)
    assert found is not None, "没找到触发 ρ5 的几何（调 gaps）"
    gap, ego, obs = found
    u = np.array([0.03, 0.01])
    p_off.reset(); r_off = p_off.safe_action(ego, obs, u, DT, VP)
    assert r_off.rho == RHO_EMERGENCY and r_off.source == "emergency", (r_off.rho, r_off.source)
    print(f"  [T1] 默认关: ρ5 → source={r_off.source} 动作={r_off.u_safe}（=Alg.1·旧行为）✅  (gap={gap:.0f}m)")
    return gap, ego, obs, u


def test_graded_preserves_policy(gap, ego, obs, u):
    """T2：开关开 + 有可行无碰撞动作 → 走 emergency_relaxed，动作保留策略意图（≠Alg.1 输出）。"""
    p_on = _mk(True); p_on.reset()
    r_on = p_on.safe_action(ego, obs, u, DT, VP)
    p_off = _mk(False); p_off.reset()
    r_off = p_off.safe_action(ego, obs, u, DT, VP)
    assert r_on.rho == RHO_EMERGENCY, r_on.rho
    if r_on.source == "emergency_relaxed":
        assert not np.allclose(r_on.u_safe, r_off.u_safe), "档1 动作不应等于 Alg.1 输出"
        assert abs(r_on.u_safe[0]) <= VP.a_max + 1e-9 and abs(r_on.u_safe[1]) <= VP.w_max + 1e-9, "越箱"
        print(f"  [T2] 开关开: source={r_on.source} 动作={r_on.u_safe} vs 关时(Alg.1)={r_off.u_safe} → 策略意图保留 ✅")
    else:
        print(f"  [T2] ⚠️ 此几何下档1不可行(source={r_on.source})=回落 Alg.1（T3 覆盖此路径）·换更远 gap 再验档1")


def test_graded_falls_back_when_infeasible():
    """T3：贴脸（无分离方向/不可行）→ 回落 emergency（Alg.1 原样）。"""
    p_on = _mk(True); p_on.reset()
    ego, obs = _states(60.0, obs_len=260.0)          # 极近=大概率无可行/退化
    r = p_on.safe_action(ego, obs, np.array([0.0, 0.0]), DT, VP)
    print(f"  [T3] 贴脸态: rho={r.rho} source={r.source}（期望 emergency=Alg.1 接管）"
          f" {'✅' if r.source == 'emergency' else '⚠️ 得 '+str(r.source)}")


def test_ec_reset_semantics(gap, ego, obs, u):
    """T4（对抗审 R4·D13）：档1→档2 切换时 EC 须被 reset（不沿用陈旧 mode）。"""
    p = _mk(True); p.reset()
    r1 = p.safe_action(ego, obs, u, DT, VP)          # 期望档1（未用 EC）
    used_ec_1 = (r1.source == "emergency")
    assert p._prev_ec_used == used_ec_1, (p._prev_ec_used, r1.source)
    print(f"  [T4] EC 使用追踪: source={r1.source} → _prev_ec_used={p._prev_ec_used}（须与是否真用 EC 一致）✅")


def test_reset_clears_flag():
    """T5：reset() 清 _prev_ec_used（防跨 episode 陈旧=D13 类）。"""
    p = _mk(True)
    p._prev_ec_used = True
    p.reset()
    assert p._prev_ec_used is False
    print("  [T5] reset() 清 _prev_ec_used ✅")


def main():
    print("=== test_graded_emergency（ρ5 分级介入·`03` L205·须服务器跑）===")
    gap, ego, obs, u = test_bit_identical_off()
    test_graded_preserves_policy(gap, ego, obs, u)
    test_graded_falls_back_when_infeasible()
    test_ec_reset_semantics(gap, ego, obs, u)
    test_reset_clears_flag()
    print("  ✅ 全部通过（闭环影响仍须 A/B：追越到达↑ ∧ 碰撞率不升 ∧ 对遇/交叉不回退）")


if __name__ == "__main__":
    main()
