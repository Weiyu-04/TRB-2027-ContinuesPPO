# -*- coding: utf-8 -*-
"""图 2（安全—效率权衡）与图 3（两项改进的贡献拆解）—— **草图版**。

user 2026-08-01：「为什么没有类似 Fig2_ablation 的图呢？我感觉全是曲线图啊。」
——正是因为这两格一直是占位框。它们原本排队等重评，但**训练期 `trend` 里已经有
到达率 / 碰撞率 / 违规次数每局**，所以现在就能画，等重评出来再换官方口径的数。

🔴 口径（图注必须写）：**官方测试集 600**、验证集挑出的最佳存档 —— 与主表逐字同源。
   （2026-08-01 later-5 之前画的是验证集 100，主表换官方数之后必须一起换，否则图文对不上。）

🔴 外部几何基线（表 6 那三条）不在本图里：它们不训练、没有训练期产物，
   必须等重评。图注按“待补”处理，不能拿别的数顶上。

跑法：<venv>/bin/python 结果/结果0801-出图草图/make_fig23_draft.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_style as PS                                  # noqa: E402
import runs_data as R                                     # noqa: E402
import matplotlib.pyplot as plt                           # noqa: E402
import matplotlib.ticker as mt                            # noqa: E402


def fig2(D):
    """安全—效率权衡（1×2）。

    左：到达率 × 每局违规（对数纵轴）；右：逐种子碰撞率。
    🔴 右格**不画中位点了事**——带盾的五条臂 40 个 run 里 39 个碰撞率恰为零，
       用中位数会把「每一颗种子都是零」这件事压成一个点。改成一颗种子一个点。
    """
    fig, (a, b) = plt.subplots(1, 2, figsize=(PS.COL2, PS.COL2 * 0.35),
                               gridspec_kw={"width_ratios": [1.0, 0.85]})

    # ── (a) 到达率 × 每局违规 ──────────────────────────────────────────────
    for tag, (lab, col, space, shield) in R.ARMS.items():
        xs = list(R.final_re(D, tag, "到达率%").values())
        ys = list(R.final_re(D, tag, "违规次数/局").values())
        if not xs:
            continue
        mx, my = np.median(xs), np.median(ys)
        xlo, xhi = R.boot_ci(xs)
        ylo, yhi = R.boot_ci(ys)
        a.errorbar(mx, my, xerr=[[mx - xlo], [xhi - mx]], yerr=[[my - ylo], [yhi - my]],
                   fmt="none", ecolor=col, elinewidth=0.7, capsize=1.3, capthick=0.7,
                   alpha=0.8, zorder=2)
        a.scatter(mx, my, marker=("o" if space == "cont" else "^"), s=30,
                  facecolor=(col if shield else "white"), edgecolor=col,
                  linewidths=1.0, zorder=3)
    #: 标签相对点的偏移（点数）与对齐 —— 9 个点手工排一次，比自动避让稳
    #: 🔴 标签文字**必须与表 1/表 2 的行名逐字相同**（`03` L243-续51）。只留偏移量与对齐方式。
    LAB = {"ours": (R.ARMS["ours"][0], 8, -8, "left"),
           "disc": (R.ARMS["disc"][0], 8, 1, "left"),
           "base": (R.ARMS["base"][0], 0, 9, "center"),
           "rr":   (R.ARMS["rr"][0], -8, -7, "right"),
           "uns":  (R.ARMS["uns"][0], 0, 10, "center"),
           "ush":  (R.ARMS["ush"][0], 0, 15, "center"),
           "ab0":  (R.ARMS["ab0"][0], 0, 9, "center"),
           "abB":  (R.ARMS["abB"][0], -6, -10, "right"),
           "abG":  (R.ARMS["abG"][0], 0, -11, "center")}
    for tag, (txt, dx, dy, ha) in LAB.items():
        xs = list(R.final_re(D, tag, "到达率%").values())
        ys = list(R.final_re(D, tag, "违规次数/局").values())
        if xs:
            a.annotate(txt, (np.median(xs), np.median(ys)), xytext=(dx, dy),
                       textcoords="offset points", fontsize=6.3,
                       color=R.ARMS[tag][1], ha=ha, va="center", zorder=5)
    a.set_yscale("log")
    a.set_ylim(0.34, 6.4)
    a.set_xlim(69, 103)
    a.yaxis.set_major_locator(mt.FixedLocator([0.4, 0.6, 1, 2, 4]))
    a.yaxis.set_major_formatter(mt.FixedFormatter(["0.4", "0.6", "1", "2", "4"]))
    a.yaxis.set_minor_locator(mt.NullLocator())
    a.set_xlabel("Arrival rate (%)")
    a.set_ylabel("COLREGs violations per episode")
    a.set_title("(a) Safety\u2013efficiency trade-off")

    # ── (b) 逐种子碰撞率 ──────────────────────────────────────────────────
    tags = list(R.ARMS)
    rng = np.random.default_rng(7)                 # 抖动写死，重跑图不变样
    for i, tag in enumerate(tags):
        lab, col, space, shield = R.ARMS[tag]
        y = len(tags) - 1 - i
        v = list(R.final_re(D, tag, "碰撞率%").values())
        if not v:
            continue
        b.scatter(v, y + rng.uniform(-0.16, 0.16, len(v)), s=13, color=col,
                  linewidths=0, alpha=0.8, zorder=3)
        b.plot([np.median(v)] * 2, [y - 0.30, y + 0.30], color=col, linewidth=1.6, zorder=4)
    b.set_yticks(range(len(tags)))
    b.set_yticklabels([R.ARMS[t][0]
                       for t in tags][::-1], fontsize=6.6)
    b.set_ylim(-0.7, len(tags) - 0.3)
    b.set_xlim(-0.35, 6.6)
    b.set_xlabel("Collision rate (%),  one dot per seed")
    b.set_title("(b) Collisions, every seed shown")
    b.grid(axis="y", visible=False)

    fig.tight_layout(w_pad=1.6)
    PS.save(fig, "Fig2_tradeoff", R.OUT_DIRS)
    plt.close(fig)


def fig3(D):
    """两项改进的贡献拆解：$2\\times2$ 消融，三格用三种图型。

    🔴 user 2026-08-01 的三条要求，逐条落实：
       ① **去掉同种子配对连线**——四条配置之间连线把柱面糊住了，遮挡大于信息。
          配对关系改由「显著性记号 + 同向种子数」承载，那才是配对检验真正的结论。
       ② **(c) 到达率不再用柱+区间**。该指标有崩掉的种子（到达率 0），分布是双峰的，
          用一根对称的自助法区间去概括它，误差棒必然拉到底——那不是画得丑，是图型选错。
          改成**箱线 + 逐种子点**，双峰一眼可见。
       ③ **柱顶只标相对变化，不标显著性记号**（user 2026-08-01 决定）。
          检验结果放正文里说，图上不挂星号。
    """
    ORDER = ["ab0", "abB", "abG", "ours"]
    #: 🔴 与表 1 / 表 2 的行名逐字相同，只在逗号处换行（`03` L243-续51）
    XTICK = [R.ARMS[t][0].replace(", ", ",\n") for t in ORDER]

    fig, AX = plt.subplots(1, 3, figsize=(PS.COL2, PS.COL2 * 0.30))
    rng = np.random.default_rng(3)
    PANELS = [("yaw", "(a) Yaw increment", "Yaw increment (norm.)", "bar"),
              ("违规次数/局", "(b) COLREGs violations", "Violations per episode", "bar"),
              ("到达率%", "(c) Arrival rate", "Arrival rate (%)", "box")]

    for ax, (key, title, ylab, kind) in zip(AX, PANELS):
        vals = [R.final_re(D, t, key) for t in ORDER]
        seeds = sorted(set.intersection(*[set(v) for v in vals]))
        cols = [R.ARMS[t][1] for t in ORDER]
        series = [[v[sd] for sd in seeds] for v in vals]
        jit = rng.uniform(-0.12, 0.12, len(seeds))

        if kind == "bar":
            med = [float(np.median(y)) for y in series]
            top = []
            for i, y in enumerate(series):
                ax.bar(i, med[i], width=0.60, color=cols[i], alpha=0.28,
                       edgecolor=cols[i], linewidth=0.9, zorder=1)
                ax.scatter(i + jit, y, s=9, facecolor="white", edgecolor=cols[i],
                           linewidths=0.7, zorder=3)
                lo, hi = R.boot_ci(y)
                top.append(max(hi, max(y)))
                ax.errorbar(i, med[i], yerr=[[med[i] - lo], [hi - med[i]]], fmt="none",
                            ecolor=PS.PALETTE["neutral_black"], elinewidth=0.9,
                            capsize=2.2, capthick=0.9, zorder=4)
            for i in (1, 2, 3):
                pct = (med[i] / med[0] - 1) * 100
                ax.annotate("%+.0f%%" % pct, (i, top[i]), xytext=(0, 4),
                            textcoords="offset points", ha="center", va="bottom",
                            fontsize=6.4, color=PS.PALETTE["neutral_black"])
            ax.set_ylim(0, max(top) * 1.22)
        else:
            bp = ax.boxplot(series, positions=range(4), widths=0.52, showfliers=False,
                            medianprops=dict(color=PS.PALETTE["neutral_black"], linewidth=1.4),
                            whiskerprops=dict(linewidth=0.8),
                            capprops=dict(linewidth=0.8), patch_artist=True)
            for patch, c in zip(bp["boxes"], cols):
                patch.set_facecolor(c); patch.set_alpha(0.26)
                patch.set_edgecolor(c); patch.set_linewidth(0.9)
            for i, y in enumerate(series):
                ax.scatter(i + jit, y, s=9, facecolor="white", edgecolor=cols[i],
                           linewidths=0.7, zorder=3)
            #: 崩掉的种子数直接标出来——这正是箱体被拉长的原因，不写清楚读者会以为是噪声
            for i, y in enumerate(series):
                nc = sum(1 for v in y if v < R.CRASH_ARR)
                if nc:
                    ax.annotate("%d collapsed" % nc, (i, min(y)), xytext=(0, -11),
                                textcoords="offset points", ha="center", va="top",
                                fontsize=5.6, color=PS.PALETTE["red_strong"])
            ax.set_ylim(-16, 112)
        ax.set_xticks(range(4))
        ax.set_xticklabels(XTICK, fontsize=6.4)
        ax.set_xlim(-0.62, 3.62)
        ax.set_title(title)
        ax.set_ylabel(ylab)
        ax.grid(axis="x", visible=False)
    fig.tight_layout(w_pad=1.8)
    PS.save(fig, "Fig3_ablation", R.OUT_DIRS)
    plt.close(fig)


def main():
    PS.apply()
    #: 🔴 2026-08-01 later-5：改画**官方测试集 600**（与主表、§5.3–§5.7 同口径）。
    #   此前画验证集 100，主表换成官方数之后就会「图与分析对不上」。
    D = R.load_reeval("正式-最佳")
    print("载入重评产物：%d 条臂 · 各 %s 颗种子"
          % (len(D), sorted({len(v) for v in D.values()})))
    fig2(D)
    fig3(D)
    print("\n⚠️ 口径提醒：本两图取自**官方测试集 600**（验证集挑出的最佳存档），与主表同源；"
          "违规为合计值（直航 + 让路）。外部几何基线未入图（不训练、无种子）。")


if __name__ == "__main__":
    main()
