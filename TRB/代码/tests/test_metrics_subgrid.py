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

_USV_ENV_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "trb_env", "usv_env.py")


def _grid_from_truth_source():
    """从 `usv_env.py` 的**源码文本**里读 A_ACC / A_OMEGA —— 不 import 它。

    为什么不 import：`usv_env` 顶层 import vesselmodels，本机没这个包 ⟹ 一 import 全套测试就跑不了，
    而本文件的价值恰恰在"纯 numpy·本机可跑"（`03` L224 前的 bug 就是靠本机跑才抓到的）。
    为什么不直接抄常量：抄 = 镜像，两处一起写错就测不出来（这正是 `metrics_subgrid` 需要
    `assert_grid_matches` 的原因）。读源码文本 = 既守住真相源、又不引依赖。
    """
    import ast
    with open(_USV_ENV_SRC, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    got = {}
    for node in tree.body:                                    # 只看模块级赋值（带类型标注的 AnnAssign 也算）
        tgt = (node.target if isinstance(node, ast.AnnAssign)
               else node.targets[0] if isinstance(node, ast.Assign) and len(node.targets) == 1 else None)
        if isinstance(tgt, ast.Name) and tgt.id in ("A_ACC", "A_OMEGA") and node.value is not None:
            got[tgt.id] = tuple(ast.literal_eval(node.value))
    missing = {"A_ACC", "A_OMEGA"} - set(got)
    if missing:                                               # 常量被改名/挪走 → 硬失败，不静默跳过
        raise AssertionError(f"在真相源 {_USV_ENV_SRC} 里找不到 {sorted(missing)} 的模块级赋值 → "
                             "常量被改名或挪位置了，本测试的真相源链断了，先修这里。")
    return got["A_ACC"], got["A_OMEGA"]


def test_discrete_subgrid_zero():
    """离散臂只能走格点 → 0<|Δ|<格步 不可能 → 次网格率恒 0(=诚实口径:非同轴比高低)。

    🔴 用【真网格全枚举】而不是手挑几步（`03` L224 教训）：老版本手挑了 6 步 `(0,±W,±2W...)`，
    **恰好绕开了唯一会翻车的那两对** `±0.012 ↔ ±0.018`（浮点差 0.005999999999999998 < 0.006），
    于是"离散臂恒 0"这条断言绿了整整一批实验，而线上真数是 5.4%。
    全枚举 = 断言钉在【真值域的每一对】上，不给"样本恰好避开缺陷"留空间。
    """
    A_ACC, A_OMEGA = _grid_from_truth_source()
    M.assert_grid_matches(A_ACC, A_OMEGA)
    worst_a = worst_w = None
    for a1 in A_ACC:                                          # 7×7 全部有序格点对 = 2401 组两步序列
        for w1 in A_OMEGA:
            for a2 in A_ACC:
                for w2 in A_OMEGA:
                    r = M.subgrid_and_rho_split([(a1, w1), (a2, w2)])
                    if r["subgrid_accel_frac"] != 0.0:
                        worst_a = (a1, a2, abs(a1 - a2))
                    if r["subgrid_yaw_frac"] != 0.0:
                        worst_w = (w1, w2, abs(w1 - w2))
    assert worst_a is None, f"油门：格点对 {worst_a} 被误判成次网格（|Δ|={worst_a[2]!r} vs 格步 {M.A_GRID_STEP!r}）"
    assert worst_w is None, f"转向：格点对 {worst_w} 被误判成次网格（|Δ|={worst_w[2]!r} vs 格步 {M.W_GRID_STEP!r}）"
    n = len(A_ACC) ** 2 * len(A_OMEGA) ** 2
    print(f"  [T1] 真网格全枚举 {n} 组格点对 → 次网格率全 0(by construction·含 ±0.012↔±0.018 浮点雷) ✅")


def test_full_grid_step_not_counted_as_subgrid():
    """回归钉死 `03` L224 那个 bug 本身：**整整挪一格【永远】不算次网格**（每一维、每个方向）。"""
    A_ACC, A_OMEGA = _grid_from_truth_source()
    for grid, idx, name, step in ((A_ACC, 0, "油门", M.A_GRID_STEP), (A_OMEGA, 1, "转向", M.W_GRID_STEP)):
        g = sorted(float(x) for x in grid)
        for lo, hi in zip(g, g[1:]):                          # 相邻格点 = 恰好一格·两个方向都测
            for x, y in ((lo, hi), (hi, lo)):
                seq = [(x, 0.0), (y, 0.0)] if idx == 0 else [(0.0, x), (0.0, y)]
                r = M.subgrid_and_rho_split(seq)
                key = "subgrid_accel_frac" if idx == 0 else "subgrid_yaw_frac"
                assert r[key] == 0.0, (
                    f"{name}：{x!r}→{y!r} 是【整一格】，却被判成次网格。"
                    f"|Δ|={abs(x - y)!r}，格步={step!r} —— 这正是 L224 的 bug。")
    # 反证：把容差去掉就会翻（证明这条测试不是空的）
    raw = [(0.0, -0.018), (0.0, -0.012)]
    d = abs(-0.018 - -0.012)
    assert d < M.W_GRID_STEP, f"前提变了：{d!r} 不再 < {M.W_GRID_STEP!r}，这条回归失去意义"
    assert M.subgrid_and_rho_split(raw)["subgrid_yaw_frac"] == 0.0, "容差没生效"
    print(f"  [T1b] 整格移动不算次网格（含 |Δ|={d!r} < 格步 {M.W_GRID_STEP!r} 这个浮点雷·反证已过）✅")


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
    test_full_grid_step_not_counted_as_subgrid()
    test_continuous_subgrid_counted()
    test_outofbox_excluded()
    test_rho_split()
    test_grid_guard()
    test_degenerate()
    print("  ✅ 全部通过")


if __name__ == "__main__":
    main()
