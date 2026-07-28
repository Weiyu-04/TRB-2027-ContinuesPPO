# -*- coding: utf-8 -*-
"""L232 独立复审脚本（2026-07-28 新窗口 · 对应 `03` L232-G 清单前 4 条）。

设计原则：**不读 L232 的任何结论，只读 `结果/结果0728-全臂重评/g[0-7].json` 的原始数**，
从零重算头条表 / 消融 / 碰撞统计，并把每个数字与 L232 白纸黑字写的值逐个对照。

跑法（零依赖，只用标准库）：
    python3 -B 代码/tests/review_l232.py

`03` L232-G 第 4 条（两个开关的逐位等价）另见同目录 `review_l232_gw_diff.py`
（那条需要 numpy/shapely/vesselmodels，因为要真跑状态机差分）。
"""
import glob
import json
import math
import os
import re
import statistics as st
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.abspath(os.path.join(HERE, "..", "..", "结果"))
D0728 = os.path.join(RES, "结果0728-全臂重评")
D0727 = os.path.join(RES, "结果0727-38臂同趟重评")

CRASH_ARR = 50.0   # 崩塌判据：到达 < 此 = 崩。取自项目自己的 `代码/bgate_judge.py:16`，不另立标准。
SEC = "strict"     # L232 铁律：表里只用 strict 563，训练期 40 场景里程碑数一律不入表。

ARMS = [
    ("Base 离散·无盾",            lambda ck: ck.startswith("Base_")),
    ("Rule-reward 离散·软奖励",   lambda ck: ck.startswith("Rule-reward_")),
    ("Discrete-safe（对标论文）",  lambda ck: ck.startswith("Discrete-safe_")),
    ("金标 连续·从零·旧配方",       lambda ck: "_L1rateON_ppo_" in ck),
    ("主线 连续·热启动·旧配方",     lambda ck: "wsHOCRppo" in ck),
    ("大集探针（已判输）",          lambda ck: "wsBIGppo" in ck),
    ("A 从零·只 Beta",           lambda ck: "A231betaPpo" in ck),
    ("B 从零·只改状态机",          lambda ck: "B231gwsymPpo" in ck),
    ("C 从零·两个都上",           lambda ck: "C231bothPpo" in ck),
]

# L232 §A 白纸黑字写的值（复审即"我算的 vs 它写的"逐格对照）
CLAIMED = {
    # 臂:                        (n, 健康, 到达, 碰撞, 违规, 让路, 直航, 转艏,     油门,   紧急)
    "Base 离散·无盾":            (5, "5/5", 98.08, 1.88, 2.318, 0.767, 1.551, 0.01378, 0.0130, 0.00),
    "Rule-reward 离散·软奖励":   (5, "5/5", 98.26, 1.67, 2.301, 0.759, 1.541, 0.01328, 0.0123, 0.00),
    "Discrete-safe（对标论文）":  (5, "3/5", 48.63, 0.07, 1.369, 0.512, 0.857, 0.01315, 0.0239, 6.83),
    "金标 连续·从零·旧配方":       (10, "8/10", 58.61, 0.09, 1.369, 0.567, 0.803, 0.01472, 0.0089, 4.22),
    "主线 连续·热启动·旧配方":     (10, "10/10", 90.91, 0.05, 1.450, 0.563, 0.887, 0.01471, 0.0036, 3.93),
    "大集探针（已判输）":          (3, "3/3", 82.24, 0.06, 1.018, 0.388, 0.629, 0.01558, 0.0037, 2.25),
    "A 从零·只 Beta":            (5, "4/5", 73.61, 0.14, 0.825, 0.352, 0.473, 0.00403, 0.0073, 3.69),
    "B 从零·只改状态机":          (3, "3/3", 65.07, 0.06, 0.967, 0.294, 0.674, 0.01547, 0.0050, 4.50),
    "C 从零·两个都上":            (10, "10/10", 92.11, 0.14, 0.710, 0.253, 0.457, 0.00256, 0.0043, 3.59),
}


def load_pass(d, prefix):
    """读一趟重评的全部分组。**只吃 `<前缀><数字>.json`** —— `g*_traj.json` 混进来会污染统计
    （L232-G 自述第一次就踩了这个坑）。"""
    files = sorted(f for f in glob.glob(os.path.join(d, prefix + "*.json"))
                   if re.fullmatch(prefix + r"\d+\.json", os.path.basename(f)))
    groups = {os.path.basename(f): json.load(open(f, encoding="utf-8")) for f in files}
    rows, dups = {}, []
    for g in groups.values():
        for ck, v in g["结果"].items():
            if ck == "_锚点汇总":
                continue
            if ck in rows:
                dups.append(ck)
            rows[ck] = v
    return files, groups, rows, dups


def arm_of(ck):
    for name, pred in ARMS:
        if pred(ck):
            return name
    return None


def fisher_two_sided(a, b, c, d):
    """2×2 Fisher 精确检验（两侧 = 所有概率 ≤ 观测表概率的表求和）。用 Fraction 精确算，不依赖 scipy。"""
    n, r1, r2, c1 = a + b + c + d, a + b, c + d, a + c
    p = lambda x: (Fraction(math.comb(r1, x)) * math.comb(r2, c1 - x)) / math.comb(n, c1)
    p_obs = p(a)
    lo, hi = max(0, c1 - r2), min(r1, c1)
    guard = Fraction(1000000001, 1000000000)   # 吸收精确有理数比较的边界抖动
    return float(sum(p(x) for x in range(lo, hi + 1) if p(x) <= p_obs * guard))


def main():
    fail = []

    # ── 第 1 条 · 头条表 ──────────────────────────────────────────────────
    print("=" * 100)
    print("【第 1 条】头条表：从 g*.json 原始数重算 + 分母/锚点自查")
    files, groups, rows, dups = load_pass(D0728, "g")
    print(f"  参与文件（已排除 *_traj.json）: {[os.path.basename(f) for f in files]}")
    ref = groups[os.path.basename(files[0])]["strict键"]
    same = all(g["strict键"] == ref for g in groups.values())
    print(f"  strict 键：8 组逐位（含顺序）相同={same} · 长度={len(ref)} · 无重复={len(set(ref)) == len(ref)}")
    if not (same and len(ref) == 563):
        fail.append("strict 键列表不一致或 ≠563")
    _, g27, _, _ = load_pass(D0727, "p")
    cross = all(g["strict键"] == ref for g in g27.values())
    print(f"  与 0727 那趟 strict 键逐位相同={cross}（同分母才谈得上跨趟比）")
    print(f"  checkpoint 总数={len(rows)} · 组间重复={dups or '无'}")
    anc = sum(1 for v in rows.values() if v.get("anchor", {}).get("通过") is True)
    print(f"  锚点自检通过={anc}/{len(rows)}")
    if anc != len(rows) or len(rows) != 56 or dups:
        fail.append("臂数/锚点/重复异常")

    by_arm = {}
    for ck, v in rows.items():
        a = arm_of(ck)
        if a is None:
            fail.append(f"无法归臂：{ck}")
            continue
        by_arm.setdefault(a, []).append((ck, v))

    print(f"\n  {'臂':<26}{'n':>3}{'健康':>7}{'到达%':>9}{'碰撞%':>8}{'违规/局':>9}{'让路':>8}"
          f"{'直航':>8}{'转艏Δ':>10}{'油门Δ':>9}{'紧急%':>8}   对照 L232-§A")
    detail = {}
    for name, _ in ARMS:
        lst = sorted(by_arm[name], key=lambda t: t[1]["seed"])
        m = lambda f: st.mean(f(v) for _, v in lst)
        cq = lambda k: m(lambda v: v[SEC]["控制质量"][k])
        got = (len(lst),
               f"{sum(1 for _, v in lst if v[SEC]['到达率%'] >= CRASH_ARR)}/{len(lst)}",
               m(lambda v: v[SEC]["到达率%"]), m(lambda v: v[SEC]["碰撞率%"]),
               m(lambda v: v[SEC]["违规次数/局"]), cq("giveway_violations"),
               cq("standon_violations"), cq("yaw_incr_mean"), cq("accel_incr_mean"),
               m(lambda v: v[SEC]["紧急步%"]))
        detail[name] = lst
        want = CLAIMED[name]
        def _off(i):
            """逐格比：字符串格（n/健康）要求完全相同，数值格按 L232 写的小数位数定容差。"""
            if isinstance(want[i], str):
                return got[i] != want[i]
            return abs(got[i] - want[i]) > 0.6 * 10 ** -_dec(want[i])

        bad = [i for i in range(10) if _off(i)]
        mark = "✅" if not bad else "❌ 第 " + "/".join(str(i) for i in bad) + " 列对不上"
        print(f"  {name:<26}{got[0]:>3}{got[1]:>7}{got[2]:>9.2f}{got[3]:>8.2f}{got[4]:>9.3f}"
              f"{got[5]:>8.3f}{got[6]:>8.3f}{got[7]:>10.5f}{got[8]:>9.4f}{got[9]:>8.2f}   {mark}")
        if bad:
            fail.append(f"{name} 表格对不上：{[(i, got[i], want[i]) for i in bad]}")
            for ck, v in lst:
                print(f"      s{v['seed']} {ck:<44} 到达={v[SEC]['到达率%']:7.2f}")

    # ── 第 2 条 · 消融归因（同种子配对，不比跨种子集的均值）────────────────
    print("\n" + "=" * 100)
    print("【第 2 条】消融归因 §D：改用**同种子配对**（原表拿 3 颗种子的均值比 10 颗种子的均值 = 跨条件比绝对值）")
    idx = {n: {v["seed"]: v for _, v in detail[n]} for n, _ in ARMS}
    G, A, B, C = (idx["金标 连续·从零·旧配方"], idx["A 从零·只 Beta"],
                  idx["B 从零·只改状态机"], idx["C 从零·两个都上"])
    f = lambda D, s, k: (D[s][SEC]["控制质量"][k] if k in ("yaw_incr_mean", "giveway_violations",
                                                          "standon_violations", "accel_incr_mean")
                         else D[s][SEC][k])
    common3 = [0, 1, 2]
    print(f"  {'':<12}{'到达%':>9}{'转艏Δ':>10}{'让路违规':>10}")
    for nm, D in [("金标", G), ("A 只Beta", A), ("B 只状态机", B), ("C 两个都上", C)]:
        print(f"  {nm:<12}{st.mean(f(D, s, '到达率%') for s in common3):>9.2f}"
              f"{st.mean(f(D, s, 'yaw_incr_mean') for s in common3):>10.5f}"
              f"{st.mean(f(D, s, 'giveway_violations') for s in common3):>10.3f}   (种子 0/1/2)")
    print("\n  ① B 臂转艏Δ 真的『纹丝不动』吗（同种子配对）：")
    for s in common3:
        a, b = f(G, s, "yaw_incr_mean"), f(B, s, "yaw_incr_mean")
        print(f"     s{s}: 金标 {a:.5f} → B {b:.5f}  ({(b / a - 1) * 100:+.1f}%)")
    print(f"     金标 10 颗种子转艏Δ 的 SD = {st.stdev(f(G, s, 'yaw_incr_mean') for s in range(10)):.5f}")
    print("\n  ② B 的让路违规 0.294 有没有被自己那颗崩种子污染：")
    for s in common3:
        print(f"     B s{s}: 到达 {f(B, s, '到达率%'):6.2f}%  让路 {f(B, s, 'giveway_violations'):.3f}")
    hB = [s for s in common3 if f(B, s, "到达率%") >= CRASH_ARR]
    print(f"     只取健康种子 {hB}: 让路 {st.mean(f(B, s, 'giveway_violations') for s in hB):.3f}"
          f"  · 三颗全算 {st.mean(f(B, s, 'giveway_violations') for s in common3):.3f}")
    print("\n  ③ 同种子配对下 B 相对金标的到达率（『只改状态机也让种子全活』这句站不站得住）：")
    for s in common3:
        print(f"     s{s}: 金标 {f(G, s, '到达率%'):6.2f} → B {f(B, s, '到达率%'):6.2f}"
              f"  ({f(B, s, '到达率%') - f(G, s, '到达率%'):+6.2f})")

    # ── 第 3 条 · 碰撞率 ────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("【第 3 条】碰撞率 §C：自己数局数 + 自己跑 Fisher（Fraction 精确算，不用 scipy）")
    cnt = {}
    for name, _ in ARMS:
        c = n = 0
        per = []
        for _, v in sorted(detail[name], key=lambda t: t[1]["seed"]):
            raw = v[SEC]["碰撞率%"] * v[SEC]["n"] / 100.0
            k = round(raw)
            assert abs(raw - k) < 1e-6, (name, raw)
            c += k
            n += v[SEC]["n"]
            per.append(f"s{v['seed']}:{k}")
        cnt[name] = (c, n)
        print(f"  {name:<26}{c:>3}/{n:<5} = {100 * c / n:6.3f}%   [{' '.join(per)}]")
    Cc, Cn = cnt["C 从零·两个都上"]
    for other in ["主线 连续·热启动·旧配方", "Discrete-safe（对标论文）", "Base 离散·无盾"]:
        oc, on = cnt[other]
        print(f"  Fisher 两侧 p（C vs {other}）= {fisher_two_sided(Cc, Cn - Cc, oc, on - oc):.4f}")
    bc, bn = cnt["Base 离散·无盾"]
    dc, dn = cnt["Discrete-safe（对标论文）"]
    print(f"  无盾 ÷ C = {(bc / bn) / (Cc / Cn):.2f}×（同时换了『加盾』和『离散→连续』两件事）")
    print(f"  单变量干净版 无盾离散 ÷ 带盾离散 = {(bc / bn) / (dc / dn):.2f}×（只差有没有盾）")

    # ── 跨趟抖动（L232-G 三条口径提醒里的第一条）────────────────────────
    print("\n" + "=" * 100)
    print("【口径核】L232-G 说跨机器连续臂抖动 ±0.5pt —— 用 0727/0728 两趟共有的 38 条臂实测")
    _, _, r27, _ = load_pass(D0727, "p")
    disc = lambda ck: ck.startswith(("Base_", "Rule-reward_", "Discrete-safe_"))
    ds = [(ck, r27[ck][SEC]["到达率%"], rows[ck][SEC]["到达率%"]) for ck in sorted(set(r27) & set(rows))]
    cont = [t for t in ds if not disc(t[0])]
    disc_ = [t for t in ds if disc(t[0])]
    print(f"  连续臂 n={len(cont)}: 最大|Δ|={max(abs(b - a) for _, a, b in cont):.2f}pt"
          f" · 中位|Δ|={st.median([abs(b - a) for _, a, b in cont]):.2f}pt"
          f" · 超 0.5pt 的 {sum(1 for _, a, b in cont if abs(b - a) > 0.5)}/{len(cont)}")
    print(f"  离散臂 n={len(disc_)}: 全部逐位相同={all(a == b for _, a, b in disc_)}")
    print(f"  最大的那条：" + max(((abs(b - a), ck, a, b) for ck, a, b in cont))[1]
          + f"  {max(((abs(b - a), ck, a, b) for ck, a, b in cont))[2]:.2f} → "
          + f"{max(((abs(b - a), ck, a, b) for ck, a, b in cont))[3]:.2f}")

    print("\n" + "=" * 100)
    print("复审结论：" + ("✅ 全部对得上" if not fail else "❌ 发现 %d 处对不上：" % len(fail)))
    for x in fail:
        print("   ·", x)
    return 1 if fail else 0


def _dec(x):
    """按 L232 表里那个数的小数位数定容差（它写 0.01378 就按 5 位比）。"""
    s = repr(float(x))
    return len(s.split(".")[1]) if "." in s else 0


if __name__ == "__main__":
    raise SystemExit(main())
