# -*- coding: utf-8 -*-
"""图 4（训练可靠性与样本效率，2×3）与图 5（盾行为演化，2×2）—— **草图版**。

🔴 为什么放在 `结果/` 而不是 `Paper/正式实验/04_图/`：
   后者的 `*.py` 进 `BASE_PAPER` 树指纹，而基准常量写在 `代码/tests/preflight_formal.sh`（`代码/` 冻结中）。
   训练还有 2 个 run 在跑 ⟹ 现在往 `Paper/正式实验/` 加 .py 会让出表机预检变红。
   **训练全部收工后，连同 `04_图/make_figures.py` 的臂名册一起正式迁**（见 `04 §八`）。
   图的**输出**直接写 `Paper/正式实验/04_图/` 与 `Paper/01_论文稿/figs/`（PDF/PNG/SVG 不参与指纹）。

🎨 画法（user 2026-08-01 指定）：衬线字（Times 观感）+ 四边封框 + 紧凑网格，见 `paper_style.py`。
   **不画分位数色带**——8 颗种子里有崩的，[25,75] 带会糊成一大片，既难看又不说明问题；
   改为每颗种子一条细线 + 一条粗中位线，分散度由真实轨迹本身呈现。
   图例框全撤，换行末内联标签（`LabelStack` 自动避让）。

跑法：<venv>/bin/python 结果/结果0801-出图草图/make_fig45_draft.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_style as PS                                  # noqa: E402
import runs_data as R                                     # noqa: E402
import matplotlib.pyplot as plt                           # noqa: E402
import matplotlib.ticker as mt                            # noqa: E402

XMAX = 10.4                 # 训练到 10.16M 步，画到这里留一点余量


# ══════════════════════════════════════════════════════════════════════════════
#  画法工具
# ══════════════════════════════════════════════════════════════════════════════
class LabelStack:
    """行末内联标签：先收集，最后按 y 排序并强制最小间距，避免互相压住。

    🔴 间距在**坐标轴归一化坐标**里算，不在数据坐标里算——本文件有对数纵轴的面板，
       数据坐标下的「等距」在图上并不等距。转成 0–1 的轴内比例后两种轴一视同仁。

    🔴 换算前必须先 `get_xlim()/get_ylim()` 把**挂起的自动缩放**落定。matplotlib 的
       `ax.plot()` 只登记「待缩放」，真正的量程要到绘制时才算；而 `ax.transData` 是
       普通属性，读它不会触发落定，`get_?lim()` 才会。少了这一步，没有显式 `set_ylim`
       的面板会拿默认量程 (0,1) 去换算，标签被丢到轴内比例 98 的位置，
       `bbox_inches="tight"` 于是把整张图撑成 97000 像素高（2026-08-01 实测）。
    """

    def __init__(self, ax, gap_frac=0.058, size=5.8, floor=0.022):
        self.ax, self.items, self.gap_frac, self.size = ax, [], gap_frac, size
        self.floor = floor                       # 最低那条别贴在横轴线上

    def add(self, x, y, text, color):
        self.items.append([x, y, text, color])

    def draw(self):
        if not self.items:
            return
        self.ax.get_xlim(); self.ax.get_ylim()   # 落定挂起的自动缩放，别删
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
                             xytext=(3, 0), textcoords="offset points",
                             color=color, fontsize=self.size,
                             va="center", zorder=5, annotation_clip=False)


def symlog_y(ax, linthresh, ticks, top):
    """对数纵轴（含 0）。刻度必须写死，不能交给自动定位器。

    🔴 `symlog` 的自动定位器在下界取到 0 时会一路向零枚举十进位，刻度多到把坐标轴撑爆。
    """
    ax.set_yscale("symlog", linthresh=linthresh)
    ax.set_ylim(0, top)
    ax.yaxis.set_major_locator(mt.FixedLocator(ticks))
    ax.yaxis.set_major_formatter(mt.FixedFormatter([("%g" % t) for t in ticks]))
    ax.yaxis.set_minor_locator(mt.NullLocator())


def band(ax, runs, key, color, x_scale=1e6, lw=1.5):
    """中位线 + 四分位带（user 2026-08-01 要求恢复置信区间）。

    🔴 带子与逐种子细线**不同时画**——两者表达的是同一件事（分散度），叠在一起只会糊。
    🔴 每一段用当时**有数据的全部种子**统计，不按最短 run 截断；末段种子数下降的情形
       由图注披露（离散安全基线还有两颗在跑）。
    返回第一个评估点的横坐标，供调用方把横轴左端对齐到它。
    """
    L = max(len(r["trend"]) for r in runs)
    xs, lo, md, hi = [], [], [], []
    for i in range(L):
        v = [r["trend"][i][key] for r in runs if i < len(r["trend"])]
        if not v:
            continue
        xs.append(next(r["trend"][i]["step"] for r in runs if i < len(r["trend"])) / x_scale)
        lo.append(np.percentile(v, 25)); md.append(np.median(v)); hi.append(np.percentile(v, 75))
    ax.fill_between(xs, lo, hi, color=color, alpha=0.16, linewidth=0, zorder=1)
    ax.plot(xs, md, color=color, linewidth=lw, zorder=3)
    return xs[0]


def seed_lines(ax, runs, key, color, x_scale=1e6):
    """每颗种子一条细线 + 一条粗中位线。返回中位线末端坐标，供内联标签定位。

    每一段用**当时有数据的全部种子**统计，不按最短 run 截断（那会让未跑满的臂看起来
    「提前停止学习」）。种子数在末段下降的情形由图注披露。
    """
    series = []
    for r in runs:
        xs = [t["step"] / x_scale for t in r["trend"]]
        ys = [t[key] for t in r["trend"]]
        ax.plot(xs, ys, color=color, linewidth=0.45, alpha=0.20, zorder=1)
        series.append((xs, ys))
    L = max(len(y) for _, y in series)
    mx, my = [], []
    for i in range(L):
        v = [y[i] for _, y in series if i < len(y)]
        if v:
            mx.append(next(x[i] for x, y in series if i < len(y)))
            my.append(float(np.median(v)))
    ax.plot(mx, my, color=color, linewidth=1.5, zorder=3)
    return mx[-1], my[-1]


def _bin(xs, ys, nbin=40):
    """按 step 分箱取中位数。620 次 rollout × 8 颗种子太密，且各 run 步点不对齐。"""
    if not len(xs):
        return np.array([]), np.array([])
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    e = np.linspace(xs.min(), xs.max(), nbin + 1)
    i = np.clip(np.digitize(xs, e) - 1, 0, nbin - 1)
    bx = np.array([xs[i == j].mean() if (i == j).any() else np.nan for j in range(nbin)])
    by = np.array([np.median(ys[i == j]) if (i == j).any() else np.nan for j in range(nbin)])
    ok = ~np.isnan(bx)
    return bx[ok], by[ok]


def curve(runs, key, nbin=40):
    xs, ys = [], []
    for r in runs:
        for c in r["curves"]:
            v = c.get(key)
            if isinstance(v, (int, float)):
                xs.append(c["step"]); ys.append(float(v))
    return _bin(xs, ys, nbin)


def src_share(runs, group, nbin=40):
    """`roll_source` 里某一组来源占该 rollout 全部来源计数的比例。"""
    FB = {"relaxed", "collision_min", "degenerate", "emergency_relaxed"}
    pick = {"projection": {"projection"}, "emergency": {"emergency"}, "fallback": FB}[group]
    xs, ys = [], []
    for r in runs:
        for c in r["curves"]:
            s = c.get("roll_source")
            if not isinstance(s, dict):
                continue
            tot = sum(v for v in s.values() if isinstance(v, (int, float)))
            if tot > 0:
                xs.append(c["step"]); ys.append(100.0 * sum(s.get(k, 0) for k in pick) / tot)
    return _bin(xs, ys, nbin)


def rho_share(runs, k, nbin=40):
    """`roll_rho` 里第 k 类态势占该 rollout 全部态势计数的比例。"""
    xs, ys = [], []
    for r in runs:
        for c in r["curves"]:
            rh = c.get("roll_rho")
            if not isinstance(rh, dict):
                continue
            tot = sum(v for v in rh.values() if isinstance(v, (int, float)))
            if tot > 0:
                xs.append(c["step"]); ys.append(100.0 * rh.get(k, 0) / tot)
    return _bin(xs, ys, nbin)


def short(tag):
    return R.ARMS[tag][0].replace("Ablation: ", "abl. ").split(" (")[0]


def xaxis(ax):
    ax.set_xlabel("Training steps (millions)")
    ax.set_xticks([0, 2, 4, 6, 8, 10])


# ══════════════════════════════════════════════════════════════════════════════
def fig4(D):
    """训练可靠性与样本效率（2×3）。

    🔴 三处按 user 2026-08-01 的意见重做：
       ① **用图例，不再自己在行末标名字**。行末标签要占横向空间，我原来把 xlim 拉到
          数据末端的 1.6 倍去腾位置，结果右侧空出一大片白——为了标签牺牲画幅，本末倒置。
          改成整图共用一个图例，横轴收紧到数据末端。
       ② **消融配置不进本图**。这里回答的是「安全盾要不要付学习速度的代价」，
          属于主对照的问题；消融另有图 3，混进来只是把线画多。
       ③ 六格用同一组配置 ⟹ 一个图例管全图，不必每格重复。
       ④ **中位线 + 四分位带**（user 2026-08-01 要求恢复置信区间）；带子与逐种子细线
          不同时画，两者说的是同一件事。
       ⑤ **横轴左端对齐到第一个评估点（0.5M 步）**。此前从 0 起画，而最早的评估在 0.5M，
          曲线与纵轴之间空一段，看起来像从原点直接跳到 40%——我们并没有 0 步的评估数据，
          补一个 (0,0) 等于编点。
    """
    MAIN = ("base", "rr", "disc", "ours")
    #: 🔴 带子只统计**收敛种子**（末段到达率 ≥ 判据），发散种子会把四分位带拉到 0，
    #   那是「训练不稳」这件事，该由 (d) 格去讲，不该糊在每一条曲线上。
    #   收敛数直接写进图例（如 Discrete-safe 6/8），一眼可查，不额外堆文字。
    #   这是论文统计一节**事先声明**的第二种报数口径，不是事后挑数据。
    OK = {t: [sd for sd, r in D[t].items()
              if r["trend"] and r["trend"][-1]["到达率%"] >= R.CRASH_ARR] for t in MAIN}
    fig, AX = plt.subplots(2, 3, figsize=(PS.COL2, PS.COL2 * 0.52))
    (a, b, c), (d, e, f) = AX

    #: 🔴 第一个评估点在 0.5M 步，**没有 0 步的评估**。横轴若从 0 起画，
    #   曲线与纵轴之间会空一段，读起来像「从原点直接跳到 40%」。
    #   补一个 (0,0) 等于编数据；正确做法是**横轴左端对齐到第一个评估点**。
    X0 = min(r["trend"][0]["step"] for t in MAIN for r in D[t].values()) / 1e6

    def trend_panel(ax, key, title, ylab, ylim):
        for tag in MAIN:
            runs = [D[tag][sd] for sd in OK[tag]]
            if runs:
                band(ax, runs, key, R.ARMS[tag][1])
        ax.set_title(title); ax.set_ylim(*ylim); ax.set_xlim(X0, XMAX)
        ax.set_ylabel(ylab); xaxis(ax)

    trend_panel(a, "到达率%", "(a) Arrival rate", "Arrival rate (%)", (-3, 103))
    trend_panel(b, "碰撞率%", "(b) Collision rate", "Collision rate (%)", (-0.4, 7.5))
    trend_panel(c, "违规次数/局", "(c) COLREGs violations", "Violations per episode", (-0.15, 4.6))
    trend_panel(e, "紧急步%", "(e) Emergency-control steps", "Emergency steps (%)", (-0.4, 10.5))

    # (d) 达到收敛判据的种子数
    for tag in MAIN:
        runs = list(D[tag].values())
        if not runs:
            continue
        L = max(len(r["trend"]) for r in runs)
        xs = [next(r["trend"][i]["step"] for r in runs if i < len(r["trend"])) / 1e6
              for i in range(L)]
        ys = [sum(1 for r in runs if i < len(r["trend"])
                  and r["trend"][i]["到达率%"] >= R.CRASH_ARR) for i in range(L)]
        d.plot(xs, ys, color=R.ARMS[tag][1], linewidth=1.4, drawstyle="steps-post")
    d.set_title("(d) Seeds reaching the 50% criterion")
    d.set_ylim(-0.35, 8.5); d.set_xlim(X0, XMAX)
    d.set_ylabel("Seeds (of 8)"); d.set_yticks([0, 2, 4, 6, 8]); xaxis(d)

    # (f) 值函数可解释方差
    for tag in MAIN:
        x, y = curve([D[tag][sd] for sd in OK[tag]], "explained_variance")
        if len(x):
            f.plot(x / 1e6, y, color=R.ARMS[tag][1], linewidth=1.3)
    f.set_title("(f) Value-function explained variance")
    f.set_xlim(0, XMAX); f.set_ylabel("Explained variance"); xaxis(f)
    #: (f) 取自逐 rollout 遥测，第一个点就在训练最开头，故横轴仍从 0 起

    # ── 整图共用一个图例，放在顶部 ────────────────────────────────────────
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=R.ARMS[t][1], linewidth=1.8,
                      label="%s  %d/%d" % (short(t), len(OK[t]), len(D[t])))
               for t in MAIN]
    fig.legend(handles=handles, loc="upper center", ncol=len(MAIN),
               bbox_to_anchor=(0.5, 1.005), frameon=False,
               handlelength=1.8, columnspacing=1.8, fontsize=7)
    fig.tight_layout(w_pad=2.0, h_pad=1.3, rect=(0, 0, 1, 0.955))
    PS.save(fig, "Fig4_training_reliability", R.OUT_DIRS)
    plt.close(fig)


def fig5(D):
    """安全盾的行为随训练演化 —— 四格用四种图型（user 2026-08-01：组合可视化、信息表达最大化）。

    (a) **对数折线 + 图例**：每一步控制的归口。曾试过堆叠面积，但三支的量级差了三个数量级
        （$93.6\\%$ / $6.2\\%$ / $0.2\\%$），面积图里后两支被压成两条看不清的边，
        反而不如对数轴上三条线看得清楚。
    (b) **折线**：动作打满率本来就是时间序列，折线最合适，保留。
    (c) **雨云图**（半边密度 + 逐点散布）：投影修正量是一个分布，不是一条曲线；
        画成分布能同时看出中心与形状，比四条抖动的折线信息量大。
    (d) **极坐标玫瑰图**：会遇态势本来就按方位扇区判定，角度放该态势的\emph{定义方位}、
        半径放实测占比，既贴海事语义又比对数折线好看。
        🔴 角度是**该态势的定义扇区**，不是实测方位分布——图注必须写清楚，别让人误读成
           「我们测了他船方位」。
    """
    #: 紧凑型（user 2026-08-01）：整图收到双栏宽的 0.78，2×2 每格接近正方形。
    #  原来是宽扁格子——横向留白多、纵向被挤，四张排开显得又大又空。
    #: 版式取舍（user 2026-08-01 两轮反馈）：
    #  2×2 满宽 → 「四张排列感觉好大」；1×4 一行 → 「实在是太窄了」。
    #  折中：**2×2，整图收到双栏宽的 0.82**，每格接近正方（约 75×62 mm），
    #  既不占满一页宽，也不把极坐标与雨云挤成条状。
    fig = plt.figure(figsize=(PS.COL2 * 0.82, PS.COL2 * 0.70))
    gs = fig.add_gridspec(2, 2)
    a = fig.add_subplot(gs[0, 0]); b = fig.add_subplot(gs[0, 1])
    c = fig.add_subplot(gs[1, 0]); d = fig.add_subplot(gs[1, 1], projection="polar")

    # ── (a) 控制归口：对数折线 + 图例 ────────────────────────────────────
    runs = list(D["ours"].values())
    GRP = [("projection", PS.PALETTE["blue_main"], "Projection (QP)"),
           ("emergency", PS.PALETTE["red_strong"], "Emergency"),
           ("fallback", PS.PALETTE["violet"], "Fallback")]
    symlog_y(a, 0.1, [0, 0.1, 1, 10, 100], 200)
    for g, col, lab in GRP:
        x, y = src_share(runs, g)
        if len(x):
            a.plot(x / 1e6, y, color=col, linewidth=1.4, label=lab)
    a.set_xlim(0, XMAX)
    a.legend(loc="lower left", fontsize=5.6, borderpad=0.3)
    a.set_title("(a) Which branch produced the control")
    a.set_ylabel("Share of control steps (%)"); xaxis(a)

    # ── (b) 动作打满率：折线 ──────────────────────────────────────────────
    b.set_title("(b) Action saturation")
    b.set_xlim(0, XMAX)
    for tag in ("ours", "uns", "disc"):
        col = R.ARMS[tag][1]
        for key, ls in (("roll_yaw_sat_frac", "-"), ("roll_acc_sat_frac", (0, (2.2, 1.4)))):
            x, y = curve(list(D[tag].values()), key)
            if len(x):
                b.plot(x / 1e6, 100 * y, color=col, linestyle=ls, linewidth=1.1,
                       label=(short(tag) if key == "roll_yaw_sat_frac" else None))
    b.legend(loc="upper left", fontsize=5.6, borderpad=0.3)
    b.set_ylabel("Action saturation (%)"); xaxis(b)

    # ── (c) 投影修正量：雨云图（半边密度 + 散点）──────────────────────────
    c.set_title("(c) Projection correction magnitude")
    ARMS_C = ["ours", "ush", "abB", "abG"]
    rng = np.random.default_rng(11)
    for i, tag in enumerate(ARMS_C):
        col = R.ARMS[tag][1]
        v = np.array([cc["roll_shield_corr_norm_mean"] for r in D[tag].values()
                      for cc in r["curves"][-60:]
                      if isinstance(cc.get("roll_shield_corr_norm_mean"), (int, float))])
        if not len(v):
            continue
        # 半边密度（高斯核，手写，免 scipy 依赖）
        grid = np.linspace(v.min(), v.max(), 120)
        h = 1.06 * v.std() * len(v) ** (-0.2)
        dens = np.exp(-0.5 * ((grid[:, None] - v[None, :]) / h) ** 2).sum(1)
        dens = dens / dens.max() * 0.34
        c.fill_betweenx(grid, i, i + dens, color=col, alpha=0.42, linewidth=0)
        c.plot(i + dens, grid, color=col, linewidth=0.8)
        sub = rng.choice(v, size=min(120, len(v)), replace=False)
        c.scatter(i - rng.uniform(0.06, 0.30, len(sub)), sub, s=2.0,
                  color=col, alpha=0.45, linewidths=0)
        c.plot([i - 0.34, i + 0.02], [np.median(v)] * 2,
               color=PS.PALETTE["neutral_black"], linewidth=1.2, zorder=5)
    c.set_xticks(range(len(ARMS_C)))
    c.set_xticklabels(["Ours", "Cont.\n+shield", "abl.\nbounded", "abl.\nsym.entry"], fontsize=5.4)
    c.set_xlim(-0.5, len(ARMS_C) - 0.4)
    c.set_ylabel("Correction (norm.)")
    c.grid(axis="x", visible=False)

    # ── (d) 会遇态势：极坐标玫瑰图 ────────────────────────────────────────
    #   角度 = 该态势的**定义方位扇区**（公约的舷灯分界），半径 = 实测占比（对数刻度）
    ROSE = [("2", "$\\rho_2$ head-on",   0.0,  45.0, PS.PALETTE["blue_secondary"]),
            ("3", "$\\rho_3$ crossing",  67.5,  90.0, PS.PALETTE["teal"]),
            ("4", "$\\rho_4$ overtaken", 180.0, 90.0, PS.PALETTE["green_3"]),
            ("1", "$\\rho_1$ stand-on", -67.5,  90.0, PS.PALETTE["blue_main"])]
    runs = list(D["ours"].values())
    tot = collections_counter(runs)
    T = sum(tot.values())
    d.set_theta_zero_location("N"); d.set_theta_direction(-1)
    rmin = 1e-3
    RMAX = np.log10(100 / rmin)
    #: 四个扇区几乎铺满整圈，圈外没有空位放标签 ⟹ 统一收成轴外的色标块
    LEG = []
    for k, lab, ang, width, col in ROSE:
        share = 100.0 * tot.get(k, 0) / T
        rr = np.log10(max(share, rmin) / rmin)
        d.bar(np.radians(ang), rr, width=np.radians(width), color=col, alpha=0.62,
              edgecolor=col, linewidth=0.8, zorder=3)
        LEG.append((lab, share, col))
    #: 紧急态由迫近判据触发、不属于任何方位扇区 ⟹ 画成一圈虚线环，不占扇形
    em = 100.0 * tot.get("5", 0) / T
    th = np.linspace(0, 2 * np.pi, 200)
    d.plot(th, np.full_like(th, np.log10(em / rmin)),
           color=PS.PALETTE["red_strong"], linewidth=1.1, linestyle=(0, (3, 2)), zorder=5)
    #: 图例（不自己标注）。紧急那条用虚线样式，与图上的虚线环对应
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    hs = [Patch(facecolor=c, alpha=0.62, edgecolor=c, label="%s  %.2f%%" % (l, v))
          for l, v, c in LEG]
    hs.append(Line2D([], [], color=PS.PALETTE["red_strong"], linestyle=(0, (3, 2)),
                     linewidth=1.1,
                     label="$\\rho_5$ emergency (any bearing)  %.2f%%" % em))
    d.legend(handles=hs, loc="upper center", bbox_to_anchor=(0.5, -0.13),
             frameon=False, fontsize=5.2, handlelength=1.1, handletextpad=0.35,
             labelspacing=0.28, ncol=1)
    d.set_rlim(0, RMAX * 1.12)
    d.set_rticks([np.log10(x / rmin) for x in (0.01, 0.1, 1, 10, 100)])
    d.set_yticklabels(["", "0.1", "1", "10", "100%"], fontsize=4.8)
    d.set_rlabel_position(123)
    d.set_xticks(np.radians([0, 90, 180, 270]))
    d.set_xticklabels(["ahead", "stbd", "astern", "port"], fontsize=5.8)
    d.tick_params(pad=1.0)
    d.set_title("(d) Encounter situations", pad=8)
    d.grid(color="#DDDDDD", linewidth=0.5)

    fig.tight_layout(w_pad=1.8, h_pad=1.4)
    PS.save(fig, "Fig5_shield_behaviour", R.OUT_DIRS)
    plt.close(fig)


def collections_counter(runs):
    """整段训练里各会遇态势的累计步数。"""
    import collections
    tot = collections.Counter()
    for r in runs:
        for c in r["curves"]:
            rh = c.get("roll_rho")
            if isinstance(rh, dict):
                for k, v in rh.items():
                    if isinstance(v, (int, float)):
                        tot[k] += v
    return tot


def main():
    PS.apply()
    D = R.load()
    R.survey(D)
    fig4(D)
    fig5(D)
    print("\n⚠️ 口径提醒：曲线取自**验证集 100 场景**与训练期遥测，"
          "**不是**官方测试集 600 的报数口径；未满额的 run 按其已有段数原样画。")


if __name__ == "__main__":
    main()
