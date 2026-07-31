# -*- coding: utf-8 -*-
"""投稿版出图风格 —— 衬线（Times 观感）+ 四边封框 + 紧凑网格。

user 2026-08-01 指定：「字体都用新罗马」「紧凑信息丰富的感觉」，并给了一张 2×3 的样例。
与 `Paper/正式实验/04_图/nature_style.py`（无衬线、去上/右边框）是**两套**，本轮全部图走本套。
训练收工正式迁移时，把本文件并进 `nature_style.py` 或整体替换（`02` C1 待办）。

🔴 字体：容器里没有 Times New Roman 本体，用 **Liberation Serif**——它与 Times 度量完全一致
   （字宽、行高逐字符相同），排版结果与 Times 无差别；数学符号走 STIX（Times 系数学字体）。
   出 PDF/SVG 时字形按 TrueType 内嵌（`fonttype 42`），转 Word 后仍可编辑。
"""
import matplotlib
import matplotlib.pyplot as plt

MM = 1.0 / 25.4
COL1 = 89 * MM              # 单栏
COL2 = 183 * MM             # 双栏

#: 与 `nature_style.PALETTE` 同源，只补了本轮要用的两档灰
PALETTE = {
    "blue_main": "#0F4D92", "blue_secondary": "#3775BA",
    "green_3": "#5FA85F", "red_strong": "#B64342",
    "neutral_light": "#C8C8C8", "neutral_mid": "#767676",
    "neutral_dark": "#4D4D4D", "neutral_black": "#272727",
    "teal": "#42949E", "violet": "#9A4D8E", "gold": "#D9A400",
}

SERIF = ["Liberation Serif", "STIXGeneral", "Times New Roman", "DejaVu Serif"]


def apply(font_size=8.0):
    """一次性设好风格。**必须在建任何 figure 之前调。**"""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": SERIF,
        "mathtext.fontset": "stix",
        "font.size": font_size,
        "axes.titlesize": font_size + 1.0,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size - 0.6,
        "ytick.labelsize": font_size - 0.6,
        "legend.fontsize": font_size - 0.8,
        # 四边封框（样例就是这个观感），线宽略细以免抢数据
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#000000",
        "axes.spines.top": True, "axes.spines.right": True,
        "axes.titlepad": 4.0,
        "axes.labelpad": 2.0,
        # 刻度朝外、短
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 2.6, "ytick.major.size": 2.6,
        "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "xtick.minor.visible": False, "ytick.minor.visible": False,
        # 极淡的网格，压在数据下面
        "axes.grid": True, "axes.axisbelow": True,
        "grid.color": "#DDDDDD", "grid.linewidth": 0.5, "grid.alpha": 1.0,
        "lines.linewidth": 1.5, "lines.solid_capstyle": "round",
        "legend.frameon": False, "legend.handlelength": 1.6,
        "legend.handletextpad": 0.5, "legend.labelspacing": 0.32,
        "legend.borderaxespad": 0.3,
        "figure.dpi": 300, "savefig.dpi": 600,
        # 字形按 TrueType 内嵌 ⟹ PDF/SVG 里的文字仍可编辑
        "pdf.fonttype": 42, "svg.fonttype": "none", "ps.fonttype": 42,
    })


def panel(ax, title=None):
    """面板通用收尾：标题居上、网格在下。"""
    if title:
        ax.set_title(title)
    ax.set_axisbelow(True)
    return ax


def save(fig, name, outdirs, also_png=True):
    """存 PDF（投稿用）+ PNG（预览用）+ SVG（转 Word 可编辑），并同步到多个目录。

    🔴 第一个目录（`04_图/`）用原名 `FigN_xxx`；其余目录（`01_论文稿/figs/`）**一律小写**，
       因为 .tex 里写的是 `figs/fig2_tradeoff.pdf`。
       2026-08-01 踩过：新图存成 `Fig4_...pdf`，而 figs/ 里还留着上一版的 `fig4_...pdf`，
       .tex 引的是小写那个 ⟹ **论文里嵌的一直是旧图，编译还不报错**
       （kpathsea 对大小写有回退查找，fig2/fig3 反而找对了，更难发现）。
       统一小写并在此处删掉同名异壳的旧文件，杜绝再出现两份。
    """
    import os
    import shutil
    exts = ("pdf", "svg") + (("png",) if also_png else ())
    first = outdirs[0]
    os.makedirs(first, exist_ok=True)
    for ext in exts:
        fig.savefig(os.path.join(first, f"{name}.{ext}"), bbox_inches="tight",
                    dpi=(600 if ext == "png" else None))
    low = name.lower()
    for d in outdirs[1:]:
        os.makedirs(d, exist_ok=True)
        for stale in (f"{name}.pdf",):                 # 同名异壳的旧文件，清掉
            p = os.path.join(d, stale)
            if os.path.exists(p) and stale != f"{low}.pdf":
                os.remove(p)
        shutil.copy2(os.path.join(first, f"{name}.pdf"), os.path.join(d, f"{low}.pdf"))
    print(f"  [出图] {name}  → {first.split('/')[-1]}/  +  {len(outdirs)-1} 处（小写 {low}.pdf）")
