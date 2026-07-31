# -*- coding: utf-8 -*-
"""图 2（安全—效率权衡）与图 3（两项改进的贡献拆解）—— **草图版**。

user 2026-08-01：「为什么没有类似 Fig2_ablation 的图呢？我感觉全是曲线图啊。」
——正是因为这两格一直是占位框。它们原本排队等重评，但**训练期 `trend` 里已经有
到达率 / 碰撞率 / 违规次数每局**，所以现在就能画，等重评出来再换官方口径的数。

🔴 口径（图注必须写）：验证集 100 场景，**不是**主表的官方测试集 600；
   违规为**合计**（直航 + 让路），训练期不回传拆分值。

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
    fig, (a, b) = plt.subplots(1, 2, figsize=(PS.COL2, PS.COL2 * 0.40),
                               gridspec_kw={"width_ratios": [1.0, 0.85]})

    # ── (a) 到达率 × 每局违规 ──────────────────────────────────────────────
    for tag, (lab, col, space, shield) in R.ARMS.items():
        xs = list(R.final(D, tag, "到达率%").values())
        ys = list(R.final(D, tag, "违规次数/局").values())
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
    LAB = {"ours": ("Ours", 8, -8, "left"), "disc": ("Discrete-safe", 8, 1, "left"),
           "base": ("Base", 0, 9, "center"), "rr": ("Rule-reward", -8, -7, "right"),
           "uns": ("Cont., no shield", 0, 10, "center"),
           "ush": ("Cont. + shield", 0, 15, "center"),
           "ab0": ("abl. neither", 0, 9, "center"),
           "abB": ("abl. bounded only", -6, -10, "right"),
           "abG": ("abl. sym. entry only", 0, -11, "center")}
    for tag, (txt, dx, dy, ha) in LAB.items():
        xs = list(R.final(D, tag, "到达率%").values())
        ys = list(R.final(D, tag, "违规次数/局").values())
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
    a.annotate("better", (0.845, 0.155), xycoords="axes fraction", fontsize=6.6,
               style="italic", color=PS.PALETTE["neutral_mid"], ha="center")
    a.annotate("", xy=(0.955, 0.045), xytext=(0.815, 0.135), xycoords="axes fraction",
               arrowprops=dict(arrowstyle="->", lw=0.8, color=PS.PALETTE["neutral_mid"]))
    a.annotate("circle: continuous   triangle: discrete   filled: shielded   open: unshielded",
               (0.5, -0.235), xycoords="axes fraction", ha="center",
               fontsize=6.2, color=PS.PALETTE["neutral_mid"])

    # ── (b) 逐种子碰撞率 ──────────────────────────────────────────────────
    tags = list(R.ARMS)
    rng = np.random.default_rng(7)                 # 抖动写死，重跑图不变样
    for i, tag in enumerate(tags):
        lab, col, space, shield = R.ARMS[tag]
        y = len(tags) - 1 - i
        v = list(R.final(D, tag, "碰撞率%").values())
        if not v:
            continue
        b.scatter(v, y + rng.uniform(-0.16, 0.16, len(v)), s=13, color=col,
                  linewidths=0, alpha=0.8, zorder=3)
        b.plot([np.median(v)] * 2, [y - 0.30, y + 0.30], color=col, linewidth=1.6, zorder=4)
    b.set_yticks(range(len(tags)))
    b.set_yticklabels([R.ARMS[t][0].replace("Ablation: ", "abl. ").split(" (")[0]
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
    """两项改进的贡献拆解：2×2 消融，三个指标，同种子配对连线。"""
    ORDER = ["ab0", "abB", "abG", "ours"]
    XTICK = ["neither", "bounded\nonly", "sym. entry\nonly", "both\n(ours)"]
    PANELS = [("yaw", "(a) Yaw increment", "Yaw increment (norm.)"),
              ("违规次数/局", "(b) COLREGs violations", "Violations per episode"),
              ("到达率%", "(c) Arrival rate", "Arrival rate (%)")]

    fig, AX = plt.subplots(1, 3, figsize=(PS.COL2, PS.COL2 * 0.30))
    for ax, (key, title, ylab) in zip(AX, PANELS):
        vals = [(R.yaw_incr(D, t) if key == "yaw" else R.final(D, t, key)) for t in ORDER]
        seeds = sorted(set.intersection(*[set(v) for v in vals]))
        # 同种子配对连线：一颗种子一条细灰线，四个配置串起来
        for sd in seeds:
            ax.plot(range(4), [v[sd] for v in vals], color=PS.PALETTE["neutral_light"],
                    linewidth=0.6, zorder=1)
        for i, t in enumerate(ORDER):
            col = R.ARMS[t][1]
            y = [vals[i][sd] for sd in seeds]
            ax.scatter([i] * len(y), y, s=11, color=col, linewidths=0, alpha=0.75, zorder=2)
            m = np.median(y)
            lo, hi = R.boot_ci(y)
            ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor=col,
                        elinewidth=0.9, capsize=2.0, capthick=0.9, zorder=3)
            ax.plot([i - 0.22, i + 0.22], [m, m], color=col, linewidth=1.8, zorder=4)
        ax.set_xticks(range(4))
        ax.set_xticklabels(XTICK, fontsize=6.4)
        ax.set_xlim(-0.5, 3.5)
        ax.set_title(title)
        ax.set_ylabel(ylab)
        if key == "yaw":
            ax.annotate("n=%d seeds, paired" % len(seeds), (0.97, 0.95),
                        xycoords="axes fraction", ha="right", va="top",
                        fontsize=6.2, color=PS.PALETTE["neutral_mid"])
    AX[1].set_ylim(bottom=0)
    fig.tight_layout(w_pad=1.8)
    PS.save(fig, "Fig3_ablation", R.OUT_DIRS)
    plt.close(fig)


def main():
    PS.apply()
    D = R.load()
    R.survey(D)
    fig2(D)
    fig3(D)
    print("\n⚠️ 口径提醒：本两图取自**验证集 100 场景**，非主表的官方测试集 600；"
          "违规为合计值（直航+让路），拆分须等重评。外部几何基线未入图（无训练期产物）。")


if __name__ == "__main__":
    main()
