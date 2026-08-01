# -*- coding: utf-8 -*-
"""头条表（表 2）的 LaTeX 行 —— **只读重评产物，一个数字都不手抄**。

用法：python3 结果/结果0801-出图草图/make_headline_table.py [正式-最佳|正式-末段]
"""
import json, math, os, re, sys
import statistics as st
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
B = os.path.join(ROOT, "Paper", "正式实验", "02_重评产物")
PAT = re.compile(r"F240([A-Za-z0-9]+?)Ppo[Ss](\d+)")
CRASH = 50.0
#: 行序 = 梯子序（与表 1 一致，`03` L243-续49 B.1）；名字逐字与表 1 相同
ROWS = [("base", "Discrete baseline"), ("rr", "Discrete, soft reward"),
        ("disc", "Discrete, masking"), ("uns", "Continuous, no shield"),
        ("ush", "Continuous, shield"), ("ab0", "Ablation reference"),
        ("abB", "Ablation, bounded only"), ("abG", "Ablation, symmetric only"),
        ("ours", "Ours")]
#: (列名, 取值函数, 小数位, 越小越好?)  —— None = 不参与加粗
COLS = [("arr", lambda r: r["strict"]["到达率%"], 1, False),
        ("col", lambda r: r["strict"]["碰撞率%"], 2, True),
        ("vio", lambda r: r["strict"]["违规次数/局"], 2, True),
        ("yaw", lambda r: r["strict"]["控制质量"]["yaw_incr_mean"], 4, None),
        ("acc", lambda r: r["strict"]["控制质量"]["accel_incr_mean"], 4, None),
        ("emg", lambda r: r["strict"]["紧急步%"], 2, None)]


def load(pas):
    d = json.load(open(os.path.join(B, pas, "all.json"), encoding="utf-8"))
    out = {}
    for k, v in d.items():
        m = PAT.search(k)
        if m:
            out.setdefault(m.group(1), {})[int(m.group(2))] = v
    return out


def boot(v, n=10000, seed=0):
    """种子层面自助法 95% 区间（与统计一节声明的口径一致）。seed 固定 ⟹ 可复现。"""
    r = np.random.default_rng(seed)
    b = np.median(r.choice(np.asarray(v, float), (n, len(v)), replace=True), axis=1)
    return np.percentile(b, 2.5), np.percentile(b, 97.5)


def main():
    pas = sys.argv[1] if len(sys.argv) > 1 else "正式-最佳"
    D = load(pas)
    vals = {}
    for tag, _ in ROWS:
        r = D[tag]
        vals[tag] = {c: st.median([f(x) for x in r.values()]) for c, f, _, _ in COLS}
        arr = [x["strict"]["到达率%"] for x in r.values()]
        vals[tag]["_ci"] = boot(arr)
        vals[tag]["_n"] = len(arr)
        vals[tag]["_conv"] = sum(1 for a in arr if a >= CRASH)
    best = {}
    for c, _, nd, small in COLS:
        if small is None:
            continue
        col = [vals[t][c] for t, _ in ROWS]
        best[c] = min(col) if small else max(col)
    print(f"% ==== 自动生成，勿手改：make_headline_table.py {pas} ====")
    for tag, name in ROWS:
        v = vals[tag]
        cells = [str(v["_n"]),
                 ("\\textbf{%d/8}" % v["_conv"]) if v["_conv"] == 8 else ("%d/8" % v["_conv"])]
        for c, _, nd, small in COLS:
            x = v[c]
            s = f"%.{nd}f" % x
            if c == "arr":
                lo, hi = v["_ci"]
                s = f"{s} [{lo:.0f},\\,{hi:.0f}]"
            if small is not None and abs(x - best[c]) < 10 ** (-nd - 1):
                s = "\\textbf{%s}" % s
            cells.append(s)
        print(f"  {name:<26}& " + " & ".join(cells) + r" \\")


if __name__ == "__main__":
    main()
