# -*- coding: utf-8 -*-
"""安全控制集的几何 —— **从真实会遇步算出来，不是示意图**。

为什么要这张图（2026-08-01 later-11，user：「每个可视化都得有作用」）：
全文最核心的那句话是「让路条款在控制量平面上恰好是一个半平面 ⟹ 安全集是凸的 ⟹ 可以投影」。
在此之前它只以文字、公式和图 1 右下角一个**手画示意**的形式出现，**没有任何一张从真实数据画出来的证据**。

本图直接把那句话画出来：取轨迹产物里的两个**真实决策步**，用**生产代码**
（`代码/trb_env/usv_projection.py` 的 `colregs_interval` 与 `collision_free_constraint`）
算出该步的三个约束集与它们的交集，再把该步**实际执行的命令**标上去。

  (a) 让路步（ρ3 交叉）：投影恰好落在合规半平面的**边界** ω = −ω_turn 上
      —— 实测 |ω − (−ω_turn)| = 0（小数点后六位精确）。这正是「取到最小合规转艏率」，
      也是离散网格做不到的那件事（7×7 网格要过转 37.5%，见 §6.2）。
  (b) 直航步（ρ1）：合规集退化成原点附近的一个小区间（保向保速），形状与 (a) 完全不同。

数据来源：`Paper/正式实验/02_重评产物/正式-轨迹全集/`（每步存了双船完整状态 + ρ + 出命令的分支）。
执行命令由相邻两步状态反推（零阶保持 ⟹ ω=Δψ/Δt、a=Δv/Δt，精确）。

用法：python3 结果/结果0801-出图草图/make_fig_safeset.py
"""
import glob
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "代码"))

import paper_style as PS  # noqa: E402
import runs_data as R  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

from trb_env.usv_projection import (ContinuousColregsProjection,          # noqa: E402
                                    collision_free_constraint, VesselState)
from trb_env.usv_dynamics import make_vessel_params, DECISION_DT          # noqa: E402

TRAJ = os.path.join(ROOT, "Paper", "正式实验", "02_重评产物", "正式-轨迹全集")
W_TURN = 0.008726646259971648
A_MAX, W_MAX = 0.24, 0.03           # 执行器量程（§3.1）
A_POL, W_POL = 0.048, 0.018         # 策略操作箱（§3.2）
#: 🔴 他船船长**必须取场景文件里的真值**，不能取库的中值。
#   2026-08-01 later-11 踩过：取 250 m 猜值重建，同一步算出「安全集为空」，
#   而产物明明记着 source=projection（投影成功）—— 猜的常量会画出一张假图。
#   取真值后，三个抽检步与产物记录**逐个一致**（u_safe 均满足重建出的无碰撞约束）。
SCEN = os.path.join(ROOT, "scenarios")


def target_length(ep):
    """从场景文件读他船船长真值（第一个 <length> 是水域范围，取其后 100--400 m 的那个）。"""
    import re
    txt = open(os.path.join(SCEN, f"T-{ep}.xml"), encoding="utf-8", errors="ignore").read()
    return next(x for x in (float(v) for v in re.findall(r"<length>([^<]*)</length>", txt))
                if 100.0 < x < 400.0)


def load_steps():
    """把 Ours 的所有轨迹步摊平，附上由相邻步反推的执行命令。"""
    out = []
    for f in sorted(glob.glob(os.path.join(TRAJ, "g*_traj.json"))):
        d = json.load(open(f, encoding="utf-8"))
        for arm, eps in d.items():
            if "ours" not in arm.lower():
                continue
            for ep, steps in eps.items():
                for i, s in enumerate(steps[:-1]):
                    nx = steps[i + 1]
                    dpsi = (nx["ego_psi"] - s["ego_psi"] + math.pi) % (2 * math.pi) - math.pi
                    # 🔴 轨迹记录里**没有他船速度**，但可由相邻两步位置精确反推
                    #    （他船按恒速外推、零阶保持 ⟹ v = |Δp| / Δt）。
                    #    2026-08-01 later-11 踩过：默认一个 5 m/s 去重建无碰撞约束，
                    #    算出来整个合规带都不可行，与产物记的 source=projection 直接矛盾。
                    ov = math.hypot(nx["obs_x"] - s["obs_x"],
                                    nx["obs_y"] - s["obs_y"]) / DECISION_DT
                    out.append(dict(s, ep=ep, idx=i, file=os.path.basename(f),
                                    w=dpsi / DECISION_DT, obs_v=ov,
                                    a=(nx["ego_v"] - s["ego_v"]) / DECISION_DT))
    return out


def geometry(s):
    """给一个真实步，用**生产代码**算出 U_box∩U_colregs 的区间与 U_cf 的那条线性约束。"""
    e = VesselState(position=np.array([s["ego_x"], s["ego_y"]]),
                    orientation=s["ego_psi"], velocity=s["ego_v"], length=175.0)
    o = VesselState(position=np.array([s["obs_x"], s["obs_y"]]),
                    orientation=s["obs_psi"], velocity=float(s["obs_v"]), length=target_length(s["ep"]))
    proj = ContinuousColregsProjection(A_MAX, W_MAX)
    (a_lo, a_hi), (w_lo, w_hi), gw, need_fb = proj.colregs_interval(int(s["rho"]), e, o,
                                                                    dt=DECISION_DT)
    u_nom = np.array([float(np.clip(s["a"], a_lo, a_hi)),
                      float(np.clip(s["w"], w_lo, w_hi))])
    g, h, dist, d_safe = collision_free_constraint(e, o, u_nom, DECISION_DT, DECISION_DT,
                                                   make_vessel_params(9.5))
    return (a_lo, a_hi), (w_lo, w_hi), g, h, dist, d_safe


def polygon(a_lo, a_hi, w_lo, w_hi, g, h, n=400):
    """区间盒 ∩ 半平面 g·u ≤ h 的顶点（凸多边形）。用网格采样后取凸包，够画图用。"""
    from scipy.spatial import ConvexHull
    aa = np.linspace(a_lo, a_hi, n)
    ww = np.linspace(w_lo, w_hi, n)
    A, W = np.meshgrid(aa, ww)
    ok = np.ones_like(A, dtype=bool)
    if g is not None:
        ok &= (g[0] * A + g[1] * W) <= h + 1e-15
    pts = np.column_stack([A[ok], W[ok]])
    if len(pts) < 3:
        return None
    return pts[ConvexHull(pts).vertices]


def panel(ax, s, title, show_ylabel=True, wlim=None):
    (a_lo, a_hi), (w_lo, w_hi), g, h, dist, d_safe = geometry(s)
    P = PS.PALETTE
    # 执行器量程（外框）与策略操作箱（内框）
    ax.add_patch(plt.Rectangle((-A_MAX, -W_MAX), 2 * A_MAX, 2 * W_MAX, fill=False,
                               edgecolor=P["neutral_black"], linewidth=0.9,
                               linestyle=(0, (4, 2)), zorder=2))
    ax.add_patch(plt.Rectangle((-A_POL, -W_POL), 2 * A_POL, 2 * W_POL, fill=False,
                               edgecolor=P["neutral_black"], linewidth=0.7,
                               linestyle=(0, (1.5, 1.5)), alpha=0.75, zorder=2))
    # 合规半平面 / 区间的边界
    if int(s["rho"]) in (2, 3):
        ax.axhline(-W_TURN, color=P["blue_main"], linewidth=1.1, zorder=3)
        ax.fill_between([-A_MAX, A_MAX], -W_MAX, -W_TURN, color=P["blue_main"],
                        alpha=0.10, zorder=1)
    else:
        for y in (w_lo, w_hi):
            ax.axhline(y, color=P["blue_main"], linewidth=1.1, zorder=3)
        ax.fill_between([-A_MAX, A_MAX], w_lo, w_hi, color=P["blue_main"], alpha=0.10, zorder=1)
    # 单步无碰撞约束
    if g is not None and abs(g[1]) > 1e-12:
        xs = np.array([-A_MAX, A_MAX])
        ax.plot(xs, (h - g[0] * xs) / g[1], color=P["red_strong"], linewidth=1.1, zorder=3)
    # 安全控制集
    V = polygon(a_lo, a_hi, w_lo, w_hi, g, h)
    if V is not None:
        ax.add_patch(Polygon(V, closed=True, facecolor=P["green_3"], alpha=0.38,
                             edgecolor=P["green_3"], linewidth=1.0, zorder=4))
    # 该步实际执行的命令
    ax.scatter([s["a"]], [s["w"]], marker="o", s=26, facecolor="white",
               edgecolor=P["neutral_black"], linewidths=1.1, zorder=6)
    ax.annotate(r"$u_{\mathrm{safe}}$", (s["a"], s["w"]), xytext=(6, -7),
                textcoords="offset points", fontsize=6.0, zorder=6)
    # 面板内直接标注（不放图例：直接标更省版面，也免得图例压住内容）
    P2 = PS.PALETTE
    ax.annotate("actuator limits", (A_MAX, -W_MAX), xytext=(-2, 3),
                textcoords="offset points", fontsize=5.2, ha="right", va="bottom",
                color=P2["neutral_black"], zorder=7)
    ax.annotate("policy box", (A_POL, W_POL), xytext=(2, 1), textcoords="offset points",
                fontsize=5.2, ha="left", va="bottom", color=P2["neutral_black"],
                alpha=0.85, zorder=7)
    if int(s["rho"]) in (2, 3):
        ax.annotate(r"$\omega \leq -\omega_{\mathrm{turn}}$  (compliant)",
                    (-A_MAX, -W_TURN), xytext=(3, -8), textcoords="offset points",
                    fontsize=5.2, ha="left", va="top", color=P2["blue_main"], zorder=7)
    else:
        ax.annotate(r"$|\omega| \leq \varepsilon_\omega$  (hold course)",
                    (-A_MAX, w_hi), xytext=(3, 3), textcoords="offset points",
                    fontsize=5.2, ha="left", va="bottom", color=P2["blue_main"], zorder=7)
    if g is not None and abs(g[1]) > 1e-12:
        xr = A_MAX * 0.55
        ax.annotate("one-step collision-free", (xr, (h - g[0] * xr) / g[1]),
                    xytext=(0, 4), textcoords="offset points", fontsize=5.2,
                    ha="center", va="bottom", color=P2["red_strong"], zorder=7)
    if V is not None:
        cx, cy = V[:, 0].mean(), V[:, 1].mean()
        ax.annotate(r"$U_{\mathrm{safe}}$", (cx, cy), xytext=(0, -7),
                    textcoords="offset points", fontsize=6.4, ha="center",
                    va="center", color=P2["neutral_black"], zorder=7)
    ax.set_xlim(-A_MAX * 1.06, A_MAX * 1.06)
    #: 直航带只有 ±ε_ω = ±0.0044，比让路格窄七倍 ⟹ (b) 单独放大纵轴，否则安全集是一条看不见的缝。
    #  图注里必须写明两格纵轴刻度不同（图和图注是两处真相）。
    ax.set_ylim(*(wlim if wlim else (-W_MAX * 1.10, W_MAX * 1.10)))
    ax.set_xlabel(r"Acceleration $a$ (m/s$^2$)")
    if show_ylabel:
        ax.set_ylabel(r"Turn rate $\omega$ (rad/s)")
    ax.set_title(title)
    ax.grid(color="#EEEEEE", linewidth=0.5)
    return dist, d_safe


def main():
    steps = load_steps()

    def cf_crosses_box(s):
        """无碰撞约束线是否穿过执行器量程箱 —— 不穿过就画不出来，三集合的故事就缺一块。"""
        try:
            (alo, ahi), (wlo, whi), g, h, dist, dsafe = geometry(s)
        except Exception:
            return False
        if g is None:
            return False
        v = [g[0] * a + g[1] * w - h for a in (-A_MAX, A_MAX) for w in (-W_MAX, W_MAX)]
        if not (min(v) < 0 < max(v)):
            return False
        V = polygon(alo, ahi, wlo, whi, g, h)
        return V is not None and len(V) >= 5      # 顶点 ≥5 = 约束真的切掉了一个角

    def near(s):
        return math.hypot(s["obs_x"] - s["ego_x"], s["obs_y"] - s["ego_y"])

    # (a) 让路步：① 投影恰好落在合规边界 ω = −ω_turn 上；② 无碰撞约束线穿过动作箱
    gw = [s for s in steps if int(s["rho"]) in (2, 3) and s["source"] == "projection"
          and near(s) < 1600]
    gw.sort(key=lambda s: abs(s["w"] + W_TURN))
    A = next(s for s in gw if cf_crosses_box(s))
    # (b) 直航步：合规集退化成保向保速的窄带；同样要求无碰撞约束线可见
    so = [s for s in steps if int(s["rho"]) == 1 and s["source"] == "projection"
          and near(s) < 1600]
    so.sort(key=near)
    B = next((s for s in so if cf_crosses_box(s)), so[0])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(PS.COL2, PS.COL2 * 0.34))
    d1, ds1 = panel(ax1, A, "(a) Give-way step (crossing)")
    d2, ds2 = panel(ax2, B, "(b) Stand-on step", show_ylabel=False,
                    wlim=(-0.009, 0.009))
    fig.tight_layout(w_pad=1.6)
    PS.save(fig, "Fig8_safe_set", R.OUT_DIRS)
    plt.close(fig)

    print(f"\n(a) 让路步 {A['file']} ep{A['ep']} step{A['idx']}: "
          f"ρ={A['rho']} · 中心距 {d1:.0f} m · d_safe {ds1:.0f} m")
    print(f"    执行命令 a={A['a']:+.5f}  ω={A['w']:+.6f}   "
          f"(−ω_turn = {-W_TURN:+.6f}, 差 {abs(A['w']+W_TURN):.2e})")
    print(f"(b) 直航步 {B['file']} ep{B['ep']} step{B['idx']}: "
          f"ρ={B['rho']} · 中心距 {d2:.0f} m · d_safe {ds2:.0f} m")
    print(f"    执行命令 a={B['a']:+.5f}  ω={B['w']:+.6f}")


if __name__ == "__main__":
    main()
