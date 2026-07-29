# -*- coding: utf-8 -*-
"""图 1（方法总览）—— 两格示意图：(a) 系统框图 · (b) 动作平面几何。

═══ 五点契约（nature-figure 规范要求先写、再写代码）═══
① **结论**：盾把"COLREGs 合规"变成动作空间里的一个**硬半平面约束**；并且**哪块可证明、哪块是经验**
   一眼可分。连续动作能精确取到合规转向的下确界，而 7 点网格取不到、必须多转。
② **证据链**：(a) 全部模块名与归档取自 `代码/trb_env/`（状态机 / 投影 / 兜底链）；
   (b) 全部数值取自代码常量，非示意：动作箱 ±0.048 / ±0.018（`usv_env.A_NORMAL_*_MAX`）·
   ω_turn = Δ_large_turn/T_M = 20°/40 s = 0.008727 rad/s（`usv_colregs.DELTA_LARGE_TURN`/`T_M`）·
   离散网格 A_ACC/A_OMEGA 七档（`usv_env.A_ACC`/`A_OMEGA`）。
   ⟹ 网格上最小的合规右转档是 −0.012（因为 −0.006 不满足 |ω|·T_M ≥ 20°），
      比连续的下确界 −0.008727 **多转 37.5%**。这个数由本脚本直接算出、不手写。
③ **形制**：双栏宽 183 mm × 高 78 mm，1×2 面板；(a) 无坐标轴的框图，(b) 带坐标轴的动作平面。
④ **后端**：Python/matplotlib（规范要求后端排他，**不得用 TikZ 或其他语言**）。
⑤ **导出**：.svg（主·可编辑）+ .pdf（矢量投稿）+ .tiff（600 dpi）+ .png（预览）。

═══ 为什么这个脚本放在 `Paper/01_论文稿/figs/` 而不是 `Paper/正式实验/04_图/` ═══
`Paper/正式实验/` 整棵树的 *.py 进 `BASE_PAPER` 树指纹（`代码/tests/preflight_formal.sh`）。
本图是**手作示意图、不读任何实验产物**，本来就不属于"由重评产物生成的数据图"那一类
⟹ 放在稿件旁边，**既符合职责划分，也不会动到训练期不该动的指纹**。

用法：  python3 -B Paper/01_论文稿/figs/make_fig1_overview.py [输出目录]
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle   # noqa: E402

# 复用规范样式层（配色 / 字号 / 可编辑文字 / 尺寸），不另起一套
_HERE = os.path.dirname(os.path.abspath(__file__))
_STYLE_DIR = os.path.normpath(os.path.join(_HERE, "..", "..", "正式实验", "04_图"))
sys.path.insert(0, _STYLE_DIR)
from nature_style import COL2, PALETTE, apply_publication_style, panel_label   # noqa: E402

# ── 强制项：可编辑文字（规范要求每个绘图脚本自己也写一份，不只依赖共享层）──
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42

# ══ 代码常量（与 `代码/trb_env/` 逐字一致；改了这里 = 改了论文数字，必须回代码核）══
A_MAX_RL, W_MAX_RL = 0.048, 0.018          # usv_env.A_NORMAL_ACCEL_MAX / A_NORMAL_OMEGA_MAX
A_ACC = (-0.048, -0.032, -0.016, 0.0, 0.016, 0.032, 0.048)      # usv_env.A_ACC
A_OMEGA = (-0.018, -0.012, -0.006, 0.0, 0.006, 0.012, 0.018)    # usv_env.A_OMEGA
DELTA_LARGE_TURN_DEG, T_M = 20.0, 40.0     # usv_colregs.DELTA_LARGE_TURN / T_M
OMEGA_TURN = (DELTA_LARGE_TURN_DEG * 3.141592653589793 / 180.0) / T_M   # ≈ 0.008727 rad/s

#: 网格上最小的**合规**右转档（|ω|·T_M ≥ Δ_large_turn）—— 由常量算出，不手写
GRID_MIN_COMPLIANT = min((g for g in A_OMEGA if g < 0 and abs(g) >= OMEGA_TURN - 1e-12),
                         key=abs)
EXTRA_TURN_PCT = (abs(GRID_MIN_COMPLIANT) / OMEGA_TURN - 1.0) * 100.0

# ── 配色语义（本图只用一个中性族 + 一个信号族 + 一个强调族）────────────────────
C_PROVABLE = PALETTE["blue_main"]        # 可证明块
C_EMPIRICAL = PALETTE["neutral_mid"]     # 经验块
C_ACCENT = PALETTE["red_strong"]         # 强调（策略指令 / 违规侧）
C_FEASIBLE = PALETTE["green_3"]          # 可行集
C_INK = PALETTE["neutral_black"]

FS = 7.0        # 规范区间 5.2–8 pt


def _box(ax, x, y, w, h, text, provable: bool):
    """一个模块框。**实线深色 = 可证明；虚线灰色 = 经验** —— 诚实性画进图里。"""
    ax.add_patch(Rectangle(
        (x, y), w, h, facecolor="white", zorder=2,
        edgecolor=C_PROVABLE if provable else C_EMPIRICAL,
        linewidth=1.1 if provable else 0.8,
        linestyle="-" if provable else (0, (2.4, 1.6))))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=FS - 0.5, color=C_INK, zorder=3, linespacing=1.25)


def _arrow(ax, p, q, color=C_INK, style="-"):
    ax.add_patch(FancyArrowPatch(
        p, q, arrowstyle="-|>", mutation_scale=6.5, linewidth=0.8,
        color=color, linestyle=style, shrinkA=1.0, shrinkB=1.0, zorder=1))


def panel_a(ax):
    """(a) 系统框图。"""
    ax.set_xlim(0, 100); ax.set_ylim(0, 62); ax.axis("off")

    # 主链（自左向右）
    _box(ax, 1, 40, 17, 12, "Observation", provable=False)
    _box(ax, 22, 40, 20, 12, "Continuous policy\n(PPO, bounded Beta)", provable=False)
    _box(ax, 46, 40, 22, 12, "COLREGs statechart\nregime $\\rho$ + give-way side", provable=True)
    _box(ax, 72, 40, 26, 12, "Safe action set\nbox $\\cap$ compliant half-plane\n$\\cap$ single-step no-collision",
         provable=True)
    _box(ax, 60, 20, 24, 11, "QP projection\n$u_{\\mathrm{safe}}=\\Pi_{P}(u)$", provable=True)
    _box(ax, 22, 20, 20, 11, "Environment\n$\\Delta t = 10$ s", provable=False)

    _arrow(ax, (18, 46), (22, 46))
    _arrow(ax, (42, 46), (46, 46))
    _arrow(ax, (68, 46), (72, 46))
    _arrow(ax, (85, 40), (76, 31))
    _arrow(ax, (60, 25.5), (42, 25.5))
    _arrow(ax, (24, 31), (24, 40))          # 环境回到观测（SE-RL：策略对加了盾之后的转移学习）
    ax.text(19.0, 35.5, "SE-RL loop", fontsize=FS - 1.2, color=C_EMPIRICAL,
            rotation=90, ha="center", va="center")

    # 兜底链（经验层）
    _box(ax, 60, 3, 24, 11, "Fallback chain\nemergency $\\to$ relaxed\n$\\to$ collision-min", provable=False)
    _arrow(ax, (72, 20), (72, 14), color=C_EMPIRICAL, style=(0, (2.4, 1.6)))
    ax.text(85.5, 17, "$P=\\varnothing$", fontsize=FS - 1.0, color=C_EMPIRICAL, ha="left", va="center")

    # 图例：可证明 vs 经验（规范偏好直接标注，但这条语义必须有显式图例）
    ax.add_patch(Rectangle((2, 10), 7, 4.2, facecolor="white", edgecolor=C_PROVABLE, linewidth=1.1))
    ax.text(10.5, 12.1, "provable", fontsize=FS - 1.0, va="center", color=C_INK)
    ax.add_patch(Rectangle((2, 3.5), 7, 4.2, facecolor="white", edgecolor=C_EMPIRICAL,
                           linewidth=0.8, linestyle=(0, (2.4, 1.6))))
    ax.text(10.5, 5.6, "empirical", fontsize=FS - 1.0, va="center", color=C_INK)


def panel_b(ax):
    """(b) 动作平面 $(a,\\omega)$ 上的几何。"""
    pad_a, pad_w = A_MAX_RL * 1.45, W_MAX_RL * 1.55
    ax.set_xlim(-pad_a, pad_a); ax.set_ylim(-pad_w, pad_w)
    ax.set_xlabel("Acceleration $a$  (m s$^{-2}$)", fontsize=FS)
    ax.set_ylabel("Yaw rate $\\omega$  (rad s$^{-1}$)", fontsize=FS)
    ax.set_xticks([-0.048, -0.024, 0, 0.024, 0.048])
    ax.set_yticks([-0.018, -0.009, 0, 0.009, 0.018])
    ax.axhline(0, color=PALETTE["neutral_light"], linewidth=0.5, zorder=0)
    ax.axvline(0, color=PALETTE["neutral_light"], linewidth=0.5, zorder=0)

    # RL 动作箱（= 离散网格的张成范围 ⟹ 两种动作空间权限相同、只差分辨率）
    ax.add_patch(Rectangle((-A_MAX_RL, -W_MAX_RL), 2 * A_MAX_RL, 2 * W_MAX_RL,
                           facecolor="none", edgecolor=C_INK, linewidth=0.9, zorder=3))
    ax.text(0, W_MAX_RL * 1.10, "RL action box  =  span of the $7\\times7$ grid",
            fontsize=FS - 1.0, ha="center", va="bottom", color=C_INK)

    # 离散 7×7 网格
    ax.plot([a for a in A_ACC for _ in A_OMEGA], [w for _ in A_ACC for w in A_OMEGA],
            marker="o", markersize=1.9, markerfacecolor="white",
            markeredgecolor=PALETTE["neutral_mid"], markeredgewidth=0.5,
            linestyle="none", zorder=4)

    # COLREGs 合规半平面：让路右转 ω ≤ −ω_turn
    ax.axhline(-OMEGA_TURN, color=C_PROVABLE, linewidth=1.1, zorder=5)
    ax.fill_between([-pad_a, pad_a], -pad_w, -OMEGA_TURN,
                    facecolor=C_PROVABLE, alpha=0.055, zorder=1)
    ax.text(-pad_a * 0.97, -OMEGA_TURN - W_MAX_RL * 0.055,
            "$\\omega \\leq -\\omega_{\\mathrm{turn}}$   (COLREGs give-way, starboard)",
            fontsize=FS - 1.0, color=C_PROVABLE, ha="left", va="top")

    # 单步无碰的线性约束（示意一条：斜率取负、把箱右上角切掉）
    x0, x1 = -pad_a, pad_a
    slope, icpt = -0.20, 0.0125
    ax.plot([x0, x1], [slope * x0 + icpt, slope * x1 + icpt],
            color=C_PROVABLE, linewidth=0.9, linestyle=(0, (4, 1.6)), zorder=5)
    ax.text(pad_a * 0.98, slope * pad_a * 0.98 + icpt + W_MAX_RL * 0.05,
            "single-step\nno-collision", fontsize=FS - 1.2, color=C_PROVABLE,
            ha="right", va="bottom", linespacing=1.15)

    # 可行集 P（箱 ∩ 半平面 ∩ 无碰）
    ax.fill_between([-A_MAX_RL, A_MAX_RL], -W_MAX_RL, -OMEGA_TURN,
                    facecolor=C_FEASIBLE, alpha=0.42, zorder=2)
    ax.text(0.0, (-W_MAX_RL - OMEGA_TURN) / 2, "feasible set $P$",
            fontsize=FS - 0.5, ha="center", va="center", color=C_INK, zorder=6)

    # 策略指令在集合外 → 投影落点
    u_des = (0.030, 0.0125)
    u_safe = (0.030, -OMEGA_TURN)
    ax.plot(*u_des, marker="o", markersize=3.4, color=C_ACCENT, zorder=7)
    ax.plot(*u_safe, marker="o", markersize=3.4, markerfacecolor="white",
            markeredgecolor=C_ACCENT, markeredgewidth=1.0, zorder=7)
    _arrow(ax, u_des, u_safe, color=C_ACCENT)
    ax.text(u_des[0] + 0.003, u_des[1], "$u_{\\mathrm{desired}}$", fontsize=FS - 0.5,
            color=C_ACCENT, ha="left", va="center")
    ax.text(u_safe[0] + 0.003, u_safe[1] - W_MAX_RL * 0.02, "$u_{\\mathrm{safe}}$",
            fontsize=FS - 0.5, color=C_ACCENT, ha="left", va="top")

    # 🔴 本图的核心信息：连续取到下确界，网格取不到
    ax.plot([-A_MAX_RL * 0.93], [-OMEGA_TURN], marker="_", markersize=0, zorder=7)
    ax.annotate(
        f"continuous reaches the infimum\n$-\\omega_{{\\mathrm{{turn}}}} = -{OMEGA_TURN:.5f}$;\n"
        f"nearest compliant grid point is\n${GRID_MIN_COMPLIANT:.3f}$  "
        f"($+{EXTRA_TURN_PCT:.1f}\\%$ turn rate)",
        xy=(-A_MAX_RL * 0.62, GRID_MIN_COMPLIANT),
        xytext=(-A_MAX_RL * 1.34, -W_MAX_RL * 0.60),
        fontsize=FS - 1.2, color=C_INK, ha="left", va="center", linespacing=1.3,
        arrowprops=dict(arrowstyle="-|>", mutation_scale=5.5, linewidth=0.7, color=C_INK,
                        shrinkA=1.0, shrinkB=2.0))


def main(outdir=None):
    outdir = outdir or _HERE
    os.makedirs(outdir, exist_ok=True)
    apply_publication_style(font_size=FS)
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 78 / 25.4),
                             gridspec_kw=dict(width_ratios=[1.06, 1.0], wspace=0.28))
    panel_a(axes[0]); panel_label(axes[0], "a", x=-0.02, y=1.00, fontsize=8)
    panel_b(axes[1]); panel_label(axes[1], "b", x=-0.17, y=1.00, fontsize=8)
    fig.subplots_adjust(left=0.045, right=0.985, top=0.95, bottom=0.115)

    base = os.path.join(outdir, "Fig1_overview")
    for ext, kw in (("svg", {}), ("pdf", {}), ("png", dict(dpi=300)),
                    ("tiff", dict(dpi=600, pil_kwargs={"compression": "tiff_lzw"}))):
        fig.savefig(f"{base}.{ext}", **kw)
    plt.close(fig)
    print(f"✅ 图 1 已出 → {base}.{{svg,pdf,png,tiff}}")
    print(f"   ω_turn        = {OMEGA_TURN:.6f} rad/s  (= {DELTA_LARGE_TURN_DEG}°/{T_M}s)")
    print(f"   网格最小合规档 = {GRID_MIN_COMPLIANT:.3f} rad/s  ⟹ 比下确界多转 {EXTRA_TURN_PCT:.1f}%")
    print("   🔴 下一步 = 人工看渲染图（静态预检查不出排版重叠）")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
