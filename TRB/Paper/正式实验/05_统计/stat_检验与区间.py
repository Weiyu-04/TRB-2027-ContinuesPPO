# -*- coding: utf-8 -*-
"""正式实验统计（`Paper/正式实验/05_统计/`）—— 全部自算，**不调 scipy**（服务器依赖已够杂）。

三样（沿用 `CLAUDE.md` §6 从 USV 项目迁来的评估方法论）：
  ① **同种子配对符号检验** —— 配对才对，跨种子集比均值是 `03` L234-C 那族错
  ② **自助法置信区间**（bootstrap over seeds）—— 报均值必带区间，n=3/5/10 时尤其
  ③ **Fisher 精确检验**（碰撞这种个位数事件）—— 用精确有理数算，不近似

⚠️ 池化 Fisher 把每一局当独立、忽略同种子内相关性；但该偏差**方向 = 让 p 变小**
   ⟹ 用来支持「统计上区分不开」这类结论是**保守**的、安全（`03` L234-F）。反向结论则不可用。

跑法：  python3 -B Paper/正式实验/05_统计/stat_检验与区间.py <重评目录>
"""
import math
import os
import random
import statistics as st
import sys
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _common as C   # noqa: E402

B_BOOT = 20000
RNG = random.Random(20260729)     # 固定种子 ⟹ 报出来的区间可复现


def sign_test(a, b):
    """两侧符号检验（配对）。返回 (变好数, 有效对数, p)。持平的对按惯例剔除。"""
    d = [y - x for x, y in zip(a, b) if y != x]
    n, k = len(d), sum(1 for x in d if x > 0)
    if n == 0:
        return 0, 0, 1.0
    tail = min(k, n - k)
    p = 2 * sum(math.comb(n, i) for i in range(tail + 1)) / 2 ** n
    return k, n, min(1.0, p)


def boot_ci(xs, alpha=0.05, stat=st.mean):
    """按**种子**重采样的自助法百分位区间（种子才是独立单位，不是 episode）。"""
    if len(xs) < 2:
        return (float("nan"), float("nan"))
    vals = sorted(stat([xs[RNG.randrange(len(xs))] for _ in xs]) for _ in range(B_BOOT))
    lo = vals[int(alpha / 2 * B_BOOT)]
    hi = vals[min(B_BOOT - 1, int((1 - alpha / 2) * B_BOOT))]
    return lo, hi


def fisher(a, b, c, d):
    """2×2 两侧 Fisher 精确检验（精确有理数·不近似）。"""
    n, r1, r2, c1 = a + b + c + d, a + b, c + d, a + c
    p = lambda x: (Fraction(math.comb(r1, x)) * math.comb(r2, c1 - x)) / math.comb(n, c1)
    p_obs, guard = p(a), Fraction(1000000001, 1000000000)
    lo, hi = max(0, c1 - r2), min(r1, c1)
    return float(sum(p(x) for x in range(lo, hi + 1) if p(x) <= p_obs * guard))


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    d = os.path.abspath(sys.argv[1])
    _exp = os.environ.get("EXPECT_STRICT")
    _exp = int(_exp) if _exp else (C.FORMAL["n_strict"] if "正式" in os.path.basename(d) else None)
    rows, n_strict, n_grp = C.load_pass(d, expect_strict=_exp)
    ba = C.by_arm(rows)
    # 🔴 预算从这趟数据自己读（`03` L243）；无条件套 FORMAL 常量会把错口径印进论文
    print(f"# 统计 · {os.path.basename(d)}\n\n> "
          + C.budget_note(n_strict, ckpt_policy=os.environ.get("CKPT_POLICY", "存档口径见正文"), rows=rows))
    print(f"> 自助法 B={B_BOOT}、按**种子**重采样（种子才是独立单位）、固定随机种子 ⟹ 区间可复现。\n")

    print("## ① 到达率：均值 + 95% 自助法区间（按种子重采样）\n")
    print("| 臂 | n | 到达均值% | 95% 区间 | 逐种子 SD | 练成/崩/欠训 |")
    print("|---|---|---|---|---|---|")
    for name, _, in_head in C.ARM_SPECS:
        if name not in ba:
            continue
        m = C.metrics(ba[name])
        lo, hi = boot_ci(m["逐种子到达"])
        # 🆕 `03` L243-改③：结局三分（练成/崩/欠训），别再用单一 <50 混成一个数
        _c = f"{m['练成']}/{m['崩']}/{m['欠训']}" + (f" ⚠️{m['过渡带']}" if m['过渡带'] else "")
        print(f"| {name} | {m['n']} | {m['到达']:.2f} | [{lo:.2f}, {hi:.2f}] | {m['到达SD']:.2f} | {_c} |")

    print("\n## ② 同种子配对符号检验（**只在两条臂有共同种子时做**）\n")
    PAIRS = [("金标（连续·从零·旧配方）", "C（从零·两个都上·主线候选）"),
             ("金标（连续·从零·旧配方）", "A（从零·只 Beta）"),
             ("金标（连续·从零·旧配方）", "B（从零·只改状态机）"),
             ("C（从零·两个都上·主线候选）", "大集探针（新配方 D）")]
    print("| 对比（前 → 后） | 共同种子 | 变好 | p（两侧） | 前均值 | 后均值 | 中位差 |")
    print("|---|---|---|---|---|---|---|")
    for x, y in PAIRS:
        if x not in ba or y not in ba:
            print(f"| {x} → {y} | — | — | — | — | — | *（缺臂，跳过）* |")
            continue
        ax, ay = C.per_seed(ba[x], "到达率%"), C.per_seed(ba[y], "到达率%")
        common = sorted(set(ax) & set(ay))
        if not common:
            print(f"| {x} → {y} | 0 | — | — | — | — | *（无共同种子）* |")
            continue
        va, vb = [ax[s] for s in common], [ay[s] for s in common]
        k, n, p = sign_test(va, vb)
        med = st.median([b - a for a, b in zip(va, vb)])
        print(f"| {x} → {y} | {len(common)} 颗 {common} | {k}/{n} | {p:.5f} | "
              f"{st.mean(va):.2f} | {st.mean(vb):.2f} | {med:+.2f} |")

    print("\n## ③ 碰撞：局数 + Fisher 精确检验\n")
    print("| 臂 | 碰撞局/总局 | 率% |")
    print("|---|---|---|")
    cnt = {}
    for name, _, _ in C.ARM_SPECS:
        if name not in ba:
            continue
        m = C.metrics(ba[name])
        cnt[name] = (m["碰撞局"], m["总局"])
        print(f"| {name} | {m['碰撞局']}/{m['总局']} | {m['碰撞率']:.3f} |")
    base = "C（从零·两个都上·主线候选）"
    if base in cnt:
        print(f"\n**以 {base} 为基准的两两 Fisher（两侧）**\n")
        print("| vs | p | 说明 |")
        print("|---|---|---|")
        ca, cn = cnt[base]
        for other, (oa, on) in cnt.items():
            if other == base:
                continue
            p = fisher(ca, cn - ca, oa, on - oa)
            note = "统计上区分不开" if p > 0.05 else "有差异"
            print(f"| {other} | {p:.4f} | {note} |")
    if "Base（离散·无盾）" in cnt and "Discrete-safe（对标论文）" in cnt:
        (b, bn), (s, sn) = cnt["Base（离散·无盾）"], cnt["Discrete-safe（对标论文）"]
        print(f"\n**盾的价值 · 单变量干净版**（同为离散·只差有没有盾·`03` L234-E⑤）："
              f"无盾 {100*b/bn:.3f}% → 带盾 {100*s/sn:.3f}% = **压 {(b/bn)/(s/sn):.1f}×**。")
        if base in cnt:
            ca, cn = cnt[base]
            print(f"（跨动作空间那版 {100*b/bn:.3f}% → {100*ca/cn:.3f}% = {(b/bn)/(ca/cn):.1f}×，"
                  "同时换了『加盾』与『离散→连续』两件事 ⟹ 只能说『我们这一套比无盾基线安全 N 倍』。）")


if __name__ == "__main__":
    main()
