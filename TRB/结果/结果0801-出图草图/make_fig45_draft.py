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


def _bin_q(xs, ys, nbin):
    """按 step 分箱，每箱给出 25/50/75 分位。dense_band 用。"""
    if not len(xs):
        return (np.array([]),) * 4
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    e = np.linspace(xs.min(), xs.max(), nbin + 1)
    i = np.clip(np.digitize(xs, e) - 1, 0, nbin - 1)
    bx, lo, md, hi = [], [], [], []
    for j in range(nbin):
        m = i == j
        if not m.any():
            continue
        bx.append(xs[m].mean())
        q = np.percentile(ys[m], [25, 50, 75])
        lo.append(q[0]); md.append(q[1]); hi.append(q[2])
    return (np.array(bx), np.array(lo), np.array(md), np.array(hi))


def dense_band(ax, runs, get, color, nbin=310, lw=0.9):
    """🔴 **每次 rollout 一个点**的密集曲线（每条 run 620 个点），中位线 + 四分位带。

    user 2026-08-01 later-4：「现在的线段看起来比较有清晰的棱角，如果是比较密集的步下
    线段样子应该不是这样的」—— 说的就是原来那几格用的是 `trend`（每条 run **只有 20 个点**，
    因为验证集评估一段只跑一次），点少所以是折线。
    训练遥测 `curves` 是每次 rollout 落一条（620 点 × 8 颗种子 = 4960 个样本），
    分 80 箱就足够平滑，且**一个点都没有编**。
    """
    xs, ys = [], []
    for r in runs:
        for c in r["curves"]:
            v = get(c)
            if v is not None:
                xs.append(c["step"]); ys.append(v)
    bx, lo, md, hi = _bin_q(xs, ys, nbin)
    if not len(bx):
        return
    ax.fill_between(bx / 1e6, lo, hi, color=color, alpha=0.15, linewidth=0, zorder=1)
    ax.plot(bx / 1e6, md, color=color, linewidth=lw, zorder=3)


def g_num(key):
    def f(c):
        v = c.get(key)
        return float(v) if isinstance(v, (int, float)) else None
    return f


def g_flag(flag):
    """训练 rollout 里该结局占本次 rollout 全部回合的比例（%）。

    `roll_ep_flags` = {'goal': 到达, 'collision': 碰撞, 'time': 超时, 'stopped': 停住}，
    `roll_eps` = 本次 rollout 结束的回合数。**这是训练场景上的、带探索噪声的口径**，
    与验证集 100 场景（确定性策略）不是一回事 —— 图注必须写明。
    """
    def f(c):
        fl, n = c.get("roll_ep_flags"), c.get("roll_eps")
        if not isinstance(fl, dict) or not isinstance(n, (int, float)) or n <= 0:
            return None
        return 100.0 * float(fl.get(flag, 0)) / float(n)
    return f


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
    return R.ARMS[tag][0]   # 🔴 与表 1/表 2 的行名逐字相同，别在这里另起缩写


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
    #   收敛数直接写进图例（如 Discrete, masking 6/8），一眼可查，不额外堆文字。
    #   这是论文统计一节**事先声明**的第二种报数口径，不是事后挑数据。
    OK = {t: [sd for sd, r in D[t].items()
              if r["trend"] and r["trend"][-1]["到达率%"] >= R.CRASH_ARR] for t in MAIN}
    #: 🔴 高宽比 0.52→0.45（2026-08-01 later-3）：论文里按 `width=\linewidth` 收，
    #  高度 = 165 mm × 高宽比 ⟹ 0.52 时 86 mm，比图 2（66）图 3（56）高出一截。
    #  格子仍是 2×3，内容一格没减，只把每格压扁一点。
    fig, AX = plt.subplots(2, 3, figsize=(PS.COL2, PS.COL2 * 0.45))
    (a, b, c), (d, e, f) = AX

    def dense_panel(ax, get, title, ylab, ylim=None, inset=None):
        """密集格：每次 rollout 一个点（620 点/run × 8 种子），310 箱中位 + 四分位带。

        `inset` = (左, 下, 宽, 高, x 上限, y 下限, y 上限)：在本格**空白处**放一个局部放大图。
        放大的都是**横轴最左端那一段**——主图横轴跨 10M 步，最有信息量的爬升/瞬变全挤在
        最左边一小条里，放大它才有意义（user 2026-08-01 later-5）。
        """
        for tag in MAIN:
            runs = [D[tag][sd] for sd in OK[tag]]
            if runs:
                dense_band(ax, runs, get, R.ARMS[tag][1])
        ax.set_title(title)
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_xlim(0, XMAX); ax.set_ylabel(ylab); xaxis(ax)
        if inset:
            L, B_, W, H, xhi, ylo, yhi = inset
            ins = ax.inset_axes([L, B_, W, H])
            for tag in MAIN:
                runs = [D[tag][sd] for sd in OK[tag]]
                if runs:
                    dense_band(ins, runs, get, R.ARMS[tag][1], nbin=120, lw=0.8)
            ins.set_xlim(0, xhi); ins.set_ylim(ylo, yhi)
            ins.tick_params(labelsize=5.6, length=1.6, pad=1.0)
            ins.set_xticks([0, xhi / 2, xhi])
            for sp in ins.spines.values():
                sp.set_linewidth(0.6)
            ins.grid(color="#E8E8E8", linewidth=0.4)
            #: 主图上用一个细框标出被放大的区间，并连到插图（matplotlib 自带，不是我画的说明线）
            ax.indicate_inset_zoom(ins, edgecolor="#888888", linewidth=0.6, alpha=0.9)

    #: 🔴 (a) 各配置的奖励函数**不一样**（本方法多了连续动作专属整形、规则奖励臂多了合规项）
    #   ⟹ 纵向高低**不可跨配置比较**，这一格看的是「收没收敛」，不是「谁的奖励高」。图注写死。
    #: (a) 曲线 0.5M 步就冲到平台，右下角一大片空白 ⟹ 插图放右下、放大最初 1M 步的爬升段
    dense_panel(a, g_num("ep_rew_mean"), "(a) Mean episode return", "Return",
                inset=(0.42, 0.13, 0.55, 0.45, 1.0, -7000, 6200))
    dense_panel(b, g_flag("goal"), "(b) Arrival rate (training)", "Arrival rate (%)", (-3, 103))
    #: (c) 初始瞬变冲到 6%，之后全程贴地 ⟹ 上方一大片空白，插图放右上、放大最初 1M 步
    dense_panel(c, g_flag("collision"), "(c) Collision rate (training)", "Collision rate (%)", (-0.4, 12),
                inset=(0.40, 0.46, 0.57, 0.48, 1.0, -0.3, 8.0))
    dense_panel(f, g_num("explained_variance"), "(f) Value-function explained variance",
                "Explained variance", (0.90, 1.005))

    #: (d) 违规次数只有验证集口径 —— 训练遥测里没有逐 rollout 的违规计数（离线判分器只在评估侧跑）。
    #   ⟹ 这一格**只有 20 个点**，形状本来就是折线，与密集格并列时靠格标题里的 (validation) 区分。
    for tag in MAIN:
        runs = [D[tag][sd] for sd in OK[tag]]
        if runs:
            band(d, runs, "违规次数/局", R.ARMS[tag][1])
    d.set_title("(d) COLREGs violations (validation)")
    d.set_ylim(-0.15, 4.6); d.set_xlim(0, XMAX)
    d.set_ylabel("Violations per episode"); xaxis(d)

    # (e) 达到收敛判据的种子数（阶梯图，本来就该是阶梯）
    for tag in MAIN:
        runs = list(D[tag].values())
        if not runs:
            continue
        L = max(len(r["trend"]) for r in runs)
        xs = [next(r["trend"][i]["step"] for r in runs if i < len(r["trend"])) / 1e6
              for i in range(L)]
        ys = [sum(1 for r in runs if i < len(r["trend"])
                  and r["trend"][i]["到达率%"] >= R.CRASH_ARR) for i in range(L)]
        e.plot(xs, ys, color=R.ARMS[tag][1], linewidth=1.4, drawstyle="steps-post")
    e.set_title("(e) Seeds reaching the 50% criterion")
    e.set_ylim(-0.35, 8.5); e.set_xlim(0, XMAX)
    e.set_ylabel("Seeds (of 8)"); e.set_yticks([0, 2, 4, 6, 8]); xaxis(e)

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
    #: 🔴 2026-08-01 later-3（user：「图 4 实在是太大了，跟前面的尺寸相差太多」）——
    #  病根不在这里的 figsize，而在**论文里是 `width=\linewidth` 收的**：
    #  以前 figsize 宽 = COL2×0.82 = 150 mm，正文行宽 165 mm ⟹ LaTeX 把它**放大 1.12 倍**，
    #  高度从 128 mm 涨到 144 mm、字号也跟着涨 12%，于是这一张比图 2/3 高出一倍还多。
    #  ⟹ 宽度必须**就是 COL2**（与图 2/3/4 同宽，缩放系数一致、字号才一致），
    #     "别太大" 要靠**压高宽比**来实现，不是靠缩宽度。
    #  高宽比 0.56 ⟹ 论文里 165×92 mm，与图 2（165×66）图 3（165×56）同一量级。
    fig = plt.figure(figsize=(PS.COL2, PS.COL2 * 0.56))
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
    #: 与表的行名逐字相同，只在空格处换行以适应格宽
    c.set_xticklabels([R.ARMS[t][0].replace(", ", ",\n") for t in ARMS_C], fontsize=5.0)
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
    #: 🔴 图例移到极坐标**右侧**（原来在下方）：整图压扁之后纵向最紧、横向反而有余，
    #  legend 放下面会把玫瑰图挤成一条；放右边正好吃掉极坐标（正方）留出的横向空档。
    d.legend(handles=hs, loc="center left", bbox_to_anchor=(1.14, 0.5),
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
