# -*- coding: utf-8 -*-
"""出版级绘图样式层（遵循 nature-figure 技能的 Python 后端约定）。

🔴 三条**不可协商**的规则（放在任何 figure 创建之前）：
    font.family / font.sans-serif / svg.fonttype='none'
  —— `svg.fonttype='none'` 让文字在 SVG 里保持 <text> 节点（可选中、可搜索、可在
     Illustrator/Inkscape 里重排），matplotlib 默认的 'path' 会把每个字形变成贝塞尔轮廓。

🔴 **图内文字一律英文**：论文投 TRB，图本来就该是英文；且规范要求 Arial/Helvetica 字体族，
   中文标签既不合规、又会在没装中文字体的机器上出豆腐块。中文只留在代码注释与 README 里。

**输出**：由各绘图脚本自己的 `save_pub()` 负责（规范要求导出语句静态可见于绘图脚本）。
**尺寸**：单栏 ≈ 89 mm，双栏 ≈ 183 mm（本模块给出常量，别再各处硬编码英寸）。
"""
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

# ── 尺寸（毫米 → 英寸）────────────────────────────────────────────────────────
MM = 1.0 / 25.4
COL1 = 89 * MM          # 单栏
COL2 = 183 * MM         # 双栏

# ── 调色板（取自规范的 PALETTE；一张图只用「一个中性族 + 一个信号族 + 一个强调族」）──
PALETTE = {
    "blue_main": "#0F4D92", "blue_secondary": "#3775BA",
    "green_3": "#8BCF8B", "red_strong": "#B64342",
    "neutral_light": "#CFCECE", "neutral_mid": "#767676",
    "neutral_dark": "#4D4D4D", "neutral_black": "#272727",
    "teal": "#42949E", "violet": "#9A4D8E", "gold": "#FFD700",
}
#: 各条臂的固定配色 —— **全项目每张图必须一致**，读者才不用重新认颜色
ARM_COLOR = {
    "Base (discrete, no shield)": PALETTE["neutral_light"],
    "Rule-reward (discrete, soft reward)": PALETTE["neutral_mid"],
    "Discrete-safe (benchmark)": PALETTE["red_strong"],
    "Gold standard (continuous, old recipe)": PALETTE["neutral_dark"],
    "Warm-start (continuous, old recipe)": PALETTE["teal"],
    "Ours (continuous + projection shield)": PALETTE["blue_main"],
    "Beta only": PALETTE["green_3"],
    "Statechart only": PALETTE["violet"],
    "Large-set probe": PALETTE["gold"],
}


def apply_publication_style(font_size=7, axes_linewidth=0.8):
    """一次性设好 Nature 系风格。**必须在建任何 figure 之前调**。"""
    # ── 强制项：可编辑 SVG 文字 ──────────────────────────────────────────────
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    # ── 版式 ────────────────────────────────────────────────────────────────
    plt.rcParams.update({
        "pdf.fonttype": 42,                 # PDF 里也保持可编辑 TrueType
        "font.size": font_size,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": axes_linewidth,
        "axes.labelsize": font_size,
        "axes.titlesize": font_size,
        "xtick.labelsize": font_size - 0.5,
        "ytick.labelsize": font_size - 0.5,
        "xtick.major.width": axes_linewidth,
        "ytick.major.width": axes_linewidth,
        "legend.frameon": False,
        "legend.fontsize": font_size - 0.5,
        "lines.linewidth": 1.0,
        "figure.dpi": 150,
    })


def panel_label(ax, label, x=-0.14, y=1.04, fontsize=8):
    """Nature 式面板编号：小写、粗体、贴左上角。"""
    ax.text(x, y, label, transform=ax.transAxes, fontsize=fontsize,
            fontweight="bold", ha="left", va="bottom")
