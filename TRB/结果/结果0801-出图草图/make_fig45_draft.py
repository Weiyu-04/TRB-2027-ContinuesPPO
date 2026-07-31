# -*- coding: utf-8 -*-
"""图 4（训练可靠性与样本效率）与图 5（盾行为演化）—— **草图版**。

🔴 为什么放在 `结果/` 而不是 `Paper/正式实验/04_图/`：
   后者的 `*.py` 进 `BASE_PAPER` 树指纹，而基准常量写在 `代码/tests/preflight_formal.sh`（`代码/` 冻结中）。
   训练还有 3 个 run 在跑 ⟹ 现在往 `Paper/正式实验/` 加 .py 会让出表机预检变红。
   **训练全部收工后，连同 `04_图/make_figures.py` 的臂名册一起正式迁**（见 `04 §八`）。
   图的**输出**直接写 `Paper/正式实验/04_图/`（PDF/PNG/SVG 不参与指纹）。

🔴 数据口径：全部取自训练期产物 `Paper/正式实验/01_训练产物/**/*.progress.json`
   · `trend`  = 每段在**验证集 100 场景**上的评估 ⟹ **不是**论文报数用的官方测试集 600，**不能进主表**
   · `curves` = 每次 rollout 的训练期遥测（盾归口 / 打满舵率 / 盾改写量 / 态势分布 / PPO 内部量）
   两者都与重评无关，所以在重评之前就能画。

🎨 画法（user 2026-08-01 指定，对齐 Fig1_sample_efficiency 那一版的观感）：
   **不画分位数色带**。8 颗种子里有崩的，[25,75] 带会糊成一大片，既难看又不说明问题。
   改为**每颗种子一条细半透明线 + 一条粗中位线**，分散度由真实轨迹本身呈现。
   另加浅色横向网格、行末内联标签（省掉图例框）、面板标题居上。

跑法（用隔离 venv，别污染为基线钉住的 numpy）：
    <venv>/bin/python 结果/结果0801-出图草图/make_fig45_draft.py
"""
import json
import os
import sys
import glob
import re
import collections

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # …/TRB
sys.path.insert(0, os.path.join(ROOT, "Paper", "正式实验", "04_图"))
import nature_style as N                                # noqa: E402
import matplotlib.pyplot as plt                         # noqa: E402

SRC = os.path.join(ROOT, "Paper", "正式实验", "01_训练产物")
OUT = os.path.join(ROOT, "Paper", "正式实验", "04_图")
#: 论文直接引用 `01_论文稿/figs/`（与图 1 同一处，转 Word 时路径简单）⟹ 出图后同步一份过去
OUT_PAPER = os.path.join(ROOT, "Paper", "01_论文稿", "figs")
PAPER_NAME = {"Fig4_training_reliability": "fig4_training_reliability",
              "Fig5_shield_behaviour": "fig5_shield_behaviour"}
FULL_SEG = 20                                           # 满额段数

#: 正式 9 条臂：TAG → (英文标签, 颜色)。🔴 `nature_style.ARM_COLOR` 仍是探索期那套，
#  故本草图自带一份；正式迁移时应把这份并进 `nature_style.py`（`02` C1 待办）。
#  与旧表的两处出入，迁移时按本表为准：base 由 neutral_light 改 neutral_dark、
#  rr 由 neutral_light 改 neutral_mid——图 4(a) 里 base 与 rr 的曲线几乎重合，
#  两条都用最浅的灰会看不见、也分不开。
ARMS = collections.OrderedDict([
    ("ours", ("Ours (continuous + shield)",      N.PALETTE["blue_main"])),
    ("disc", ("Discrete-safe (benchmark)",       N.PALETTE["red_strong"])),
    ("base", ("Base (discrete, no shield)",      N.PALETTE["neutral_dark"])),
    ("rr",   ("Rule-reward (discrete)",          N.PALETTE["neutral_mid"])),
    ("uns",  ("Continuous, no shield",           N.PALETTE["gold"])),
    ("ush",  ("Continuous + shield",             N.PALETTE["teal"])),
    ("ab0",  ("Ablation: neither",               N.PALETTE["neutral_black"])),
    ("abB",  ("Ablation: Beta only",             N.PALETTE["green_3"])),
    ("abG",  ("Ablation: symmetric entry only",  N.PALETTE["violet"])),
])
CRASH_ARR = 50.0        # 与 `代码/bgate_judge.py:16` / `_common.py:44` 同一判据，禁止另定


def load():
    """{tag: {seed: {"trend": [...], "curves": [...], "n": 段数}}}

    🔴 同一个 (臂, 种子) 在树里可能出现多份：
       · `**/segments/*@sNN.progress.json` 是逐段存档，只含前 NN+1 段（rr s0 有 20 份）
       · 跨机目录之间也可能撞到同名产物
       递归 glob 的返回次序不保证，直接按 seed 覆盖会让**某一份残档随机盖掉整份**
       （2026-08-01 实测：rr s0 明明有满额 20 段，却被 @s13 那份盖成 14 段）。
       故按「段数多者胜、段数相同则遥测多者胜」归并，与文件出现次序无关。
    """
    out = collections.defaultdict(dict)
    for p in glob.glob(os.path.join(SRC, "**", "*.progress.json"), recursive=True):
        d = json.load(open(p, encoding="utf-8"))
        m = re.search(r"F240([a-zA-Z0-9]+?)Ppo", os.path.basename(p))
        if not m or m.group(1) not in ARMS:
            continue
        rec = {"trend": d.get("trend") or [], "curves": d.get("curves") or []}
        rec["n"] = len(rec["trend"])
        old = out[m.group(1)].get(d["seed"])
        if old is None or (rec["n"], len(rec["curves"])) > (old["n"], len(old["curves"])):
            out[m.group(1)][d["seed"]] = rec
    return out


def _curve(runs, key, nbin=40):
    """把若干 run 的逐-rollout 序列按 step 分箱求中位数 → (step, 中位数)。

    分箱而不是原样画：620 次 rollout × 8 颗种子太密，且各 run 的 rollout 步点不对齐。
    """
    xs, ys = [], []
    for r in runs:
        for c in r["curves"]:
            v = c.get(key)
            if isinstance(v, (int, float)):
                xs.append(c["step"]); ys.append(float(v))
    if not xs:
        return np.array([]), np.array([])
    xs, ys = np.asarray(xs), np.asarray(ys)
    edges = np.linspace(xs.min(), xs.max(), nbin + 1)
    idx = np.clip(np.digitize(xs, edges) - 1, 0, nbin - 1)
    bx = np.array([xs[idx == i].mean() if (idx == i).any() else np.nan for i in range(nbin)])
    by = np.array([np.median(ys[idx == i]) if (idx == i).any() else np.nan for i in range(nbin)])
    ok = ~np.isnan(bx)
    return bx[ok], by[ok]


def _src_share(runs, group, nbin=40):
    """`roll_source` 里某一组来源占该 rollout 全部来源计数的比例（按 step 分箱取中位数）。"""
    FB = {"relaxed", "collision_min", "degenerate", "emergency_relaxed"}
    pick = {"projection": {"projection"}, "emergency": {"emergency"}, "fallback": FB}[group]
    xs, ys = [], []
    for r in runs:
        for c in r["curves"]:
            s = c.get("roll_source")
            if not isinstance(s, dict):
                continue
            tot = sum(v for v in s.values() if isinstance(v, (int, float)))
            if tot <= 0:
                continue
            xs.append(c["step"]); ys.append(100.0 * sum(s.get(k, 0) for k in pick) / tot)
    if not xs:
        return np.array([]), np.array([])
    xs, ys = np.asarray(xs), np.asarray(ys)
    edges = np.linspace(xs.min(), xs.max(), nbin + 1)
    idx = np.clip(np.digitize(xs, edges) - 1, 0, nbin - 1)
    bx = np.array([xs[idx == i].mean() if (idx == i).any() else np.nan for i in range(nbin)])
    by = np.array([np.median(ys[idx == i]) if (idx == i).any() else np.nan for i in range(nbin)])
    ok = ~np.isnan(bx)
    return bx[ok], by[ok]


def grid(ax, axis="y"):
    """浅色网格，压在数据下面。"""
    ax.grid(axis=axis, color=N.PALETTE["neutral_light"], linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)


def symlog_y(ax, linthresh, ticks, top):
    """对数纵轴（含 0）。刻度必须写死，不能交给自动定位器。

    🔴 `symlog` 的自动定位器在下界取到 0 时会一路向零枚举十进位，刻度多到把坐标轴
       撑爆——2026-08-01 实测图 5 因此导出成 3787×96981 像素（97000 像素高）。
       写死刻度并关掉次刻度即可。
    """
    import matplotlib.ticker as mt
    ax.set_yscale("symlog", linthresh=linthresh)
    ax.set_ylim(0, top)
    ax.yaxis.set_major_locator(mt.FixedLocator(ticks))
    ax.yaxis.set_major_formatter(mt.FixedFormatter(
        [("%g" % t) for t in ticks]))
    ax.yaxis.set_minor_locator(mt.NullLocator())


def seed_lines(ax, runs, key, color, x_scale=1e6):
    """每颗种子一条细线 + 一条粗中位线。返回中位线末端坐标，供内联标签定位。"""
    series = []
    for r in runs:
        xs = [t["step"] / x_scale for t in r["trend"]]
        ys = [t[key] for t in r["trend"]]
        ax.plot(xs, ys, color=color, linewidth=0.5, alpha=0.22, zorder=1)
        series.append((xs, ys))
    L = max(len(y) for _, y in series)
    mx, my = [], []
    for i in range(L):
        v = [y[i] for _, y in series if i < len(y)]
        if v:
            mx.append(next(x[i] for x, y in series if i < len(y)))
            my.append(float(np.median(v)))
    ax.plot(mx, my, color=color, linewidth=1.6, zorder=3, solid_capstyle="round")
    return mx[-1], my[-1]



class LabelStack:
    """行末内联标签：先收集，最后按 y 排序并强制最小间距，避免互相压住。

    🔴 间距在**坐标轴归一化坐标**里算，不在数据坐标里算——图 5 有两个面板用对数纵轴，
       数据坐标下的「等距」在图上并不等距。转成 0–1 的轴内比例后两种轴一视同仁。
       所以调用前必须先 `set_ylim` / `set_yscale`，否则换算依据的是临时量程。

    🔴 换算前必须先 `get_xlim()/get_ylim()` 把**挂起的自动缩放**落定。matplotlib 的
       `ax.plot()` 只登记「待缩放」，真正的量程要到绘制时才算；而 `ax.transData` 是
       普通属性，读它\\emph{不会}触发落定，`get_?lim()` 才会。少了这一步，没有显式
       `set_ylim` 的面板会拿默认量程 (0,1) 去换算，标签被丢到轴内比例 98 的位置，
       `bbox_inches="tight"` 于是把整张图撑成 97000 像素高（2026-08-01 实测）。
    """

    def __init__(self, ax, gap_frac=0.052, size=5.6, floor=0.022):
        self.ax, self.items, self.gap_frac, self.size = ax, [], gap_frac, size
        self.floor = floor                          # 最低那条别贴在横轴线上

    def add(self, x, y, text, color):
        self.items.append([x, y, text, color])

    def draw(self):
        if not self.items:
            return
        self.ax.get_xlim(); self.ax.get_ylim()      # 落定挂起的自动缩放，别删
        to_frac = self.ax.transAxes.inverted().transform
        pts = []
        for x, y, text, color in self.items:
            fx, fy = to_frac(self.ax.transData.transform((x, y)))
            pts.append([fx, max(fy, self.floor), text, color])
        pts.sort(key=lambda t: t[1])
        for i in range(1, len(pts)):
            if pts[i][1] - pts[i - 1][1] < self.gap_frac:
                pts[i][1] = pts[i - 1][1] + self.gap_frac
        for fx, fy, text, color in pts:
            self.ax.annotate(text, (fx, fy), xycoords="axes fraction",
                             xytext=(4, 0), textcoords="offset points",
                             color=color, fontsize=self.size, fontweight="bold",
                             va="center", zorder=5, annotation_clip=False)


def _trend_band(runs):
    """逐段验证集到达率：中位数、[25,75] 分位带，以及每段实际参与的种子数。

    🔴 不按最短 run 截断（那会让未跑满的臂看起来「提前停止学习」）。改为每一段用
    **当时有数据的全部种子**统计，并返回逐段的 n，调用方在 n 下降处改画虚线并标注。
    """
    if not runs:
        return (np.array([]),) * 5
    L = max(len(r["trend"]) for r in runs)
    step, med, lo, hi, ns = [], [], [], [], []
    for i in range(L):
        v = [r["trend"][i]["到达率%"] for r in runs if i < len(r["trend"])]
        if not v:
            continue
        step.append(next(r["trend"][i]["step"] for r in runs if i < len(r["trend"])))
        med.append(np.median(v)); lo.append(np.percentile(v, 25)); hi.append(np.percentile(v, 75))
        ns.append(len(v))
    return (np.array(step, float), np.array(med), np.array(lo), np.array(hi), np.array(ns))


def save_pub(fig, name):
    for ext in ("pdf", "png", "svg"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight",
                    dpi=(600 if ext == "png" else None))
    if name in PAPER_NAME:                      # 同步论文用的那份，避免两处不一致
        import shutil
        shutil.copy2(os.path.join(OUT, f"{name}.pdf"),
                     os.path.join(OUT_PAPER, f"{PAPER_NAME[name]}.pdf"))
    print(f"  [出图] {name}.pdf / .png / .svg" + ("  → 已同步 01_论文稿/figs/" if name in PAPER_NAME else ""))


# ══════════════════════════════════════════════════════════════════════════════
def fig4(D):
    """训练可靠性（2x2）。细线=逐种子，粗线=中位数，不画分位色带。"""
    fig, AX = plt.subplots(2, 2, figsize=(N.COL2, N.COL2 * 0.60))
    (a, b), (c, d) = AX

    a.set_title("(a) Validation arrival rate", pad=4)
    a.set_ylim(-2, 122); a.set_xlim(0, 16.5)
    LSa = LabelStack(a, gap_frac=0.055)
    for tag in ("base", "rr", "disc", "ours"):
        runs = list(D[tag].values())
        if not runs:
            continue
        lab, col = ARMS[tag]
        x, y = seed_lines(a, runs, "\u5230\u8fbe\u7387%", col)
        LSa.add(x, y, lab.split(" (")[0], col)
    LSa.draw()
    a.set_xlabel("Training steps (millions)"); a.set_ylabel("Arrival rate (%)")
    a.set_xticks([0, 2, 4, 6, 8, 10]); a.set_yticks([0, 20, 40, 60, 80, 100]); grid(a)

    g, h = "ab0", "ours"
    common = sorted(set(D[g]) & set(D[h]))
    b.set_title("(b) Paired by seed (n=%d)" % len(common), pad=4)
    for sd in common:
        y0 = D[g][sd]["trend"][-1]["\u5230\u8fbe\u7387%"]
        y1 = D[h][sd]["trend"][-1]["\u5230\u8fbe\u7387%"]
        b.plot([0, 1], [y0, y1], color=N.PALETTE["neutral_light"], linewidth=0.8, zorder=1)
        b.scatter([0, 1], [y0, y1], s=14, zorder=2, color=[ARMS[g][1], ARMS[h][1]], linewidths=0)
    b.set_xticks([0, 1]); b.set_xticklabels(["Ablation:\nneither", "Ours"])
    b.set_ylabel("Arrival rate (%)"); b.set_xlim(-0.45, 1.45); b.set_ylim(-2, 104); grid(b)

    c.set_title("(c) Seeds reaching the 50% arrival criterion", pad=4)
    c.set_ylim(-0.3, 8.6); c.set_xlim(0, 19.5)
    LSc = LabelStack(c, gap_frac=0.075)
    for tag in ("ours", "disc", "abB", "abG", "ab0"):
        runs = list(D[tag].values())
        if not runs:
            continue
        lab, col = ARMS[tag]
        L = max(len(r["trend"]) for r in runs)
        xs = [next(r["trend"][i]["step"] for r in runs if i < len(r["trend"])) / 1e6 for i in range(L)]
        ys = [sum(1 for r in runs if i < len(r["trend"])
                  and r["trend"][i]["\u5230\u8fbe\u7387%"] >= CRASH_ARR) for i in range(L)]
        c.plot(xs, ys, color=col, linewidth=1.3, drawstyle="steps-post")
        LSc.add(xs[-1], ys[-1], lab.replace("Ablation: ", "abl. ").split(" (")[0], col)
    LSc.draw()
    c.set_xlabel("Training steps (millions)"); c.set_ylabel("Seeds (of 8)")
    c.set_xticks([0, 2, 4, 6, 8, 10]); grid(c)

    d.set_title("(d) Value-function explained variance", pad=4)
    d.set_ylim(0.86, 1.02); d.set_xlim(0, 19.5)
    LSd = LabelStack(d)
    for tag in ("uns", "ours", "ush", "ab0"):
        x, y = _curve(list(D[tag].values()), "explained_variance")
        if not len(x):
            continue
        lab, col = ARMS[tag]
        d.plot(x / 1e6, y, color=col, linewidth=1.3)
        LSd.add(x[-1] / 1e6, y[-1], lab.replace("Ablation: ", "abl. ").split(" (")[0], col)
    LSd.draw()
    d.set_xlabel("Training steps (millions)"); d.set_ylabel("Explained variance")
    d.set_xticks([0, 2, 4, 6, 8, 10]); grid(d)

    fig.tight_layout(w_pad=2.6, h_pad=1.8)
    save_pub(fig, "Fig4_training_reliability")
    plt.close(fig)


def _rho_share(runs, k, nbin=40):
    """`roll_rho` 里第 k 类态势占该 rollout 全部态势计数的比例（按 step 分箱取中位数）。"""
    xs, ys = [], []
    for r in runs:
        for cc in r["curves"]:
            rh = cc.get("roll_rho")
            if not isinstance(rh, dict):
                continue
            tot = sum(v for v in rh.values() if isinstance(v, (int, float)))
            if tot > 0:
                xs.append(cc["step"]); ys.append(100.0 * rh.get(k, 0) / tot)
    if not xs:
        return np.array([]), np.array([])
    xs, ys = np.asarray(xs), np.asarray(ys)
    e = np.linspace(xs.min(), xs.max(), nbin + 1)
    i = np.clip(np.digitize(xs, e) - 1, 0, nbin - 1)
    bx = np.array([xs[i == j].mean() if (i == j).any() else np.nan for j in range(nbin)])
    by = np.array([np.median(ys[i == j]) if (i == j).any() else np.nan for j in range(nbin)])
    ok = ~np.isnan(bx)
    return bx[ok], by[ok]


def fig5(D):
    """盾的行为随训练演化（2×2）。与图 4 同一套画法：无图例框、行末内联标签、浅色网格。"""
    fig, AX = plt.subplots(2, 2, figsize=(N.COL2, N.COL2 * 0.64))
    (a, b), (c, d) = AX
    XMAX = 10.4                                   # 训练到 10.16M 步，留一点余量

    # ── (a) 每一步控制由哪一支产生（本文方法）──────────────────────────────
    a.set_title("(a) Which branch produced the control", pad=4)
    symlog_y(a, 1.0, [0, 1, 10, 100], 150); a.set_xlim(0, XMAX * 1.62)
    LSa = LabelStack(a, gap_frac=0.075)
    for grp, col, lab in (("projection", N.PALETTE["blue_main"], "Projection (QP)"),
                          ("emergency", N.PALETTE["red_strong"], "Emergency"),
                          ("fallback", N.PALETTE["violet"], "Fallback")):
        x, y = _src_share(list(D["ours"].values()), grp)
        if not len(x):
            continue
        a.plot(x / 1e6, y, color=col, linewidth=1.3)
        LSa.add(x[-1] / 1e6, y[-1], lab, col)
    LSa.draw()
    a.set_xlabel("Training steps (millions)"); a.set_ylabel("Share of control steps (%)")
    a.set_xticks([0, 2, 4, 6, 8, 10]); grid(a)

    # ── (b) 动作打满率：有界 Beta vs 无界高斯 vs 离散网格，转艏轴与加速度轴分开 ──
    #    六条线全挂内联标签会糊，改成「颜色区分配置、线型区分轴」，只在转艏线末标配置名，
    #    线型含义写进面板标题（放角落会被黄色那条压住）。
    b.set_title("(b) Action saturation  (solid: yaw, dashed: accel.)", pad=4)
    b.set_xlim(0, XMAX * 1.52)
    LSb = LabelStack(b, gap_frac=0.075)
    for tag in ("ours", "uns", "disc"):
        lab, col = ARMS[tag]
        for key, ls in (("roll_yaw_sat_frac", "-"), ("roll_acc_sat_frac", (0, (2.2, 1.4)))):
            x, y = _curve(list(D[tag].values()), key)
            if not len(x):
                continue
            b.plot(x / 1e6, 100 * y, color=col, linestyle=ls, linewidth=1.1)
            if key == "roll_yaw_sat_frac":
                LSb.add(x[-1] / 1e6, 100 * y[-1], lab.split(" (")[0], col)
    LSb.draw()
    b.set_xlabel("Training steps (millions)"); b.set_ylabel("Action saturation (%)")
    b.set_xticks([0, 2, 4, 6, 8, 10]); grid(b)

    # ── (c) 投影修正量（按动作箱半宽归一化）────────────────────────────────
    c.set_title("(c) Projection correction magnitude", pad=4)
    c.set_xlim(0, XMAX * 1.52)
    LSc = LabelStack(c, gap_frac=0.062)
    for tag in ("ours", "ush", "abB", "abG"):
        x, y = _curve(list(D[tag].values()), "roll_shield_corr_norm_mean")
        if not len(x):
            continue
        lab, col = ARMS[tag]
        c.plot(x / 1e6, y, color=col, linewidth=1.3)
        LSc.add(x[-1] / 1e6, y[-1], lab.replace("Ablation: ", "abl. ").split(" (")[0], col)
    LSc.draw()
    c.set_xlabel("Training steps (millions)")
    c.set_ylabel("Correction (norm.)")
    c.set_xticks([0, 2, 4, 6, 8, 10]); grid(c)

    # ── (d) 会遇态势占比（本文方法）────────────────────────────────────────
    d.set_title("(d) Encounter-situation profile", pad=4)
    symlog_y(d, 0.1, [0, 0.1, 1, 10, 100], 200); d.set_xlim(0, XMAX * 1.42)
    RHO = [("0", "$\\rho_0$ none", N.PALETTE["neutral_light"]),
           ("1", "$\\rho_1$ give-way", N.PALETTE["blue_main"]),
           ("2", "$\\rho_2$ give-way", N.PALETTE["blue_secondary"]),
           ("3", "$\\rho_3$ give-way", N.PALETTE["teal"]),
           ("4", "$\\rho_4$ stand-on", N.PALETTE["green_3"]),
           ("5", "$\\rho_5$ emergency", N.PALETTE["red_strong"])]
    LSd = LabelStack(d, gap_frac=0.078)
    runs = list(D["ours"].values())
    for k, lab, col in RHO:
        bx, by = _rho_share(runs, k)
        if not len(bx):
            continue
        d.plot(bx / 1e6, by, color=col, linewidth=1.2)
        LSd.add(bx[-1] / 1e6, by[-1], lab, col)
    LSd.draw()
    d.set_xlabel("Training steps (millions)"); d.set_ylabel("Share of steps (%)")
    d.set_xticks([0, 2, 4, 6, 8, 10]); grid(d)

    fig.tight_layout(w_pad=2.6, h_pad=1.8)
    save_pub(fig, "Fig5_shield_behaviour")
    plt.close(fig)


def main():
    N.apply_publication_style()
    D = load()
    print("载入：")
    for tag in ARMS:
        runs = D.get(tag, {})
        part = [f"s{s}({r['n']}/20)" for s, r in sorted(runs.items()) if r["n"] < FULL_SEG]
        print(f"  {tag:5s} {len(runs)} 颗种子" + (f"  ⚠️ 未满额：{', '.join(part)}" if part else ""))
    fig4(D)
    fig5(D)
    print("\n⚠️ 草图口径提醒：曲线取自**验证集 100 场景**与训练期遥测，"
          "**不是**官方测试集 600 的报数口径；未满额的 run 按其已有段数原样画。")


if __name__ == "__main__":
    main()
