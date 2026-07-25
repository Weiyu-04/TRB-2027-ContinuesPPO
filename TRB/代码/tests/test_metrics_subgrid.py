#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""本机单测 · metrics_subgrid（次网格细调率 + 按态势拆转艏）。纯 numpy·不依赖 vesselmodels → CI/本机可跑。
覆盖：T1 离散臂(格点动作)次网格率恒0(by construction·诚实口径的实证) · T2 连续细调被正确计入 ·
      T3 箱外(紧急/兜底)步被排除 · T4 按态势拆 |Δω| 分组正确 · T5 网格常量守卫 assert_grid_matches ·
      T6 退化输入(空/单步/畸形)不崩。
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))       # 代码/
from trb_env import metrics_subgrid as M


def test_discrete_subgrid_zero():
    """离散臂只能走格点(0/±0.006/±0.012...)→ 0<|Δ|<格步 不可能 → 次网格率恒 0(=诚实口径:非同轴比高低)。"""
    A, W = 0.016, 0.006
    seq = [(0.0, 0.0), (A, W), (A, 0.0), (-A, -W), (0.0, 2 * W), (2 * A, W)]
    r = M.subgrid_and_rho_split(seq)
    assert r["subgrid_accel_frac"] == 0.0, r
    assert r["subgrid_yaw_frac"] == 0.0, r
    print(f"  [T1] 离散格点动作 次网格率 accel={r['subgrid_accel_frac']} yaw={r['subgrid_yaw_frac']}(须0·by construction) ✅")


def test_continuous_subgrid_counted():
    """连续做【比格步细】的调整 → 应被计入(离散做不到的分辨率)。"""
    seq = [(0.0, 0.0), (0.004, 0.001), (0.007, 0.0025), (0.010, 0.004)]   # 每步 Δa≈0.003-0.004<0.016·Δω≈0.0015<0.006
    r = M.subgrid_and_rho_split(seq)
    assert r["subgrid_accel_frac"] == 1.0 and r["subgrid_yaw_frac"] == 1.0, r
    assert r["n_inbox_pairs"] == 3, r
    print(f"  [T2] 连续次网格细调 全计入 accel={r['subgrid_accel_frac']} yaw={r['subgrid_yaw_frac']} n={r['n_inbox_pairs']} ✅")


def test_outofbox_excluded():
    """箱外步(紧急/兜底·物理满程 0.24/0.03)须被排除(同 _control_quality 口径·防污染平滑度)。"""
    seq = [(0.001, 0.001), (0.004, 0.002), (0.24, 0.03), (0.005, 0.003)]  # 第3步箱外
    r = M.subgrid_and_rho_split(seq)
    # 合法相邻对只有 (0,1)：(1,2)/(2,3) 都含箱外步
    assert r["n_inbox_pairs"] == 1, r
    print(f"  [T3] 箱外步排除: n_inbox_pairs={r['n_inbox_pairs']}(须1·(1,2)(2,3)含满程步被剔) ✅")


def test_rho_split():
    """|Δω| 按态势分组：让路态(ρ∈2,3,4)大转艏 vs 非让路小转艏 → giveway 均值应显著更大。"""
    seq = [(0.0, 0.000), (0.0, 0.001), (0.0, 0.002),     # 非让路：小转艏变化 0.001
           (0.0, 0.014), (0.0, 0.002)]                    # 让路：大转艏变化 0.012 / 0.012
    rhos = [0, 0, 0, 3, 3]                                # 变化归属 ρ[k+1]：对(2,3)→ρ3=让路·对(3,4)→ρ3=让路
    r = M.subgrid_and_rho_split(seq, rhos)
    assert r["yaw_incr_giveway"] > r["yaw_incr_other"], r
    assert r["n_pairs_giveway"] == 2 and r["n_pairs_other"] == 2, r
    print(f"  [T4] 按态势拆: 让路|Δω|={r['yaw_incr_giveway']} > 非让路={r['yaw_incr_other']}"
          f"(n={r['n_pairs_giveway']}/{r['n_pairs_other']}) ✅")


def test_grid_guard():
    """网格常量守卫：与真相源一致→True；漂移→raise(防静默口径错)。"""
    A_ACC = (-0.048, -0.032, -0.016, 0.0, 0.016, 0.032, 0.048)
    A_OMEGA = (-0.018, -0.012, -0.006, 0.0, 0.006, 0.012, 0.018)
    assert M.assert_grid_matches(A_ACC, A_OMEGA) is True
    bad = tuple(x * 2 for x in A_OMEGA)     # 漂移的转艏网格
    try:
        M.assert_grid_matches(A_ACC, bad)
        raise AssertionError("网格漂移未被守卫拦下")
    except ValueError:
        pass
    print("  [T5] 网格守卫: 一致→True·漂移→raise ✅")


def test_degenerate():
    for bad in (None, [], [(0.0, 0.0)], [(1, 2, 3), (4, 5, 6)]):
        r = M.subgrid_and_rho_split(bad)
        assert r["subgrid_accel_frac"] is None, (bad, r)
    r = M.subgrid_and_rho_split([(0.001, 0.001), (0.002, 0.002)], rhos=[0])   # rhos 长度不匹配→只跳过②
    assert r["subgrid_accel_frac"] is not None and r["yaw_incr_giveway"] is None, r
    print("  [T6] 退化输入(空/单步/畸形/rhos不匹配) 不崩·优雅降级 ✅")


def main():
    print("=== test_metrics_subgrid（次网格细调率 + 按态势拆转艏·本机单测）===")
    test_discrete_subgrid_zero()
    test_continuous_subgrid_counted()
    test_outofbox_excluded()
    test_rho_split()
    test_grid_guard()
    test_degenerate()
    print("  ✅ 全部通过")


if __name__ == "__main__":
    main()
