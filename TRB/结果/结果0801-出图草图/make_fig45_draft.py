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
ARMS = collections.OrderedDict([
    ("ours", ("Ours (continuous + shield)",      N.PALETTE["blue_main"])),
    ("disc", ("Discrete-safe (benchmark)",       N.PALETTE["red_strong"])),
    ("base", ("Base (discrete, no shield)",      N.PALETTE["neutral_dark"])),
    ("rr",   ("Rule-reward (discrete)",          N.PALETTE["neutral_light"])),
    ("uns",  ("Continuous, no shield",           N.PALETTE["gold"])),
    ("ush",  ("Continuous + shield",             N.PALETTE["teal"])),
    ("ab0",  ("Ablation: neither",               N.PALETTE["neutral_black"])),
    ("abB",  ("Ablation: Beta only",             N.PALETTE["green_3"])),
    ("abG",  ("Ablation: symmetric entry only",  N.PALETTE["violet"])),
])
CRASH_ARR = 50.0        # 与 `代码/bgate_judge.py:16` / `_common.py:44` 同一判据，禁止另定


def load():
    """{tag: {seed: {"trend": [...], "curves": [...], "n": 段数}}}"""
    out = collections.defaultdict(dict)
    for p in glob.glob(os.path.join(SRC, "**", "*.progress.json"), recursive=True):
        d = json.load(open(p, encoding="utf-8"))
        m = re.search(r"F240([a-zA-Z0-9]+?)Ppo", os.path.basename(p))
        if not m or m.group(1) not in ARMS:
            continue
        tr = d.get("trend") or []
        out[m.group(1)][d["seed"]] = {"trend": tr, "curves": d.get("curves") or [], "n": len(tr)}
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
    """训练可靠性与样本效率（2×2）。"""
    fig, AX = plt.subplots(2, 2, figsize=(N.COL2, N.COL2 * 0.62))
    (a, b), (c, d) = AX

    # (a) 学习曲线 —— 四条主对照臂
    for tag in ("ours", "disc", "base", "rr"):
        runs = list(D[tag].values())
        x, m, lo, hi, ns = _trend_band(runs)
        if not len(x):
            continue
        lab, col = ARMS[tag]
        k = int((ns == ns.max()).sum())            # 全部种子都在的最后一段
        a.plot(x[:k] / 1e6, m[:k], color=col, label=f"{lab} (n={ns.max()})")
        if k < len(x):                              # 后段样本变少 → 虚线并标 n
            a.plot(x[k - 1:] / 1e6, m[k - 1:], color=col, linestyle="--", linewidth=0.9)
            a.annotate(f"n={ns[-1]}", (x[-1] / 1e6, m[-1]), color=col, fontsize=5,
                       xytext=(3, 0), textcoords="offset points", va="center")
        a.fill_between(x / 1e6, lo, hi, color=col, alpha=0.16, linewidth=0)
    a.set_xlabel("Training steps (millions)"); a.set_ylabel("Validation arrival rate (%)")
    a.legend(loc="lower right", fontsize=5.6); a.set_ylim(0, 100)
    a.text(0.02, 0.97, "dashed: fewer seeds still running (n shown)", transform=a.transAxes,
           fontsize=5.2, va="top", color=N.PALETTE["neutral_mid"], style="italic")
    N.panel_label(a, "a")

    # (b) 同种子配对：⑦ 消融·都不改 → ① 本文方法
    g, h = "ab0", "ours"
    common = sorted(set(D[g]) & set(D[h]))
    for i, s in enumerate(common):
        y0 = D[g][s]["trend"][-1]["到达率%"]; y1 = D[h][s]["trend"][-1]["到达率%"]
        b.plot([0, 1], [y0, y1], color=N.PALETTE["neutral_mid"], linewidth=0.7, zorder=1)
        b.scatter([0, 1], [y0, y1], s=12, zorder=2,
                  color=[ARMS[g][1], ARMS[h][1]])
    b.set_xticks([0, 1]); b.set_xticklabels(["Ablation:\nneither", "Ours"])
    b.set_ylabel("Validation arrival rate (%)"); b.set_xlim(-0.35, 1.35); b.set_ylim(0, 100)
    b.set_title(f"paired by seed (n={len(common)})", pad=3)
    N.panel_label(b, "b")

    # (c) 达到「练成」判据（验证集到达 ≥50）的种子数随训练
    for tag in ("ours", "disc", "ab0", "abB", "abG"):
        runs = list(D[tag].values())
        if not runs:
            continue
        L = max(len(r["trend"]) for r in runs)
        x = np.array([next(r["trend"][i]["step"] for r in runs if i < len(r["trend"]))
                      for i in range(L)], dtype=float)
        y = [sum(1 for r in runs if i < len(r["trend"]) and r["trend"][i]["到达率%"] >= CRASH_ARR)
             for i in range(L)]
        lab, col = ARMS[tag]
        c.plot(x / 1e6, y, color=col, label=lab, drawstyle="steps-post")
    c.set_xlabel("Training steps (millions)")
    c.set_ylabel(f"Seeds with arrival $\\geq${CRASH_ARR:.0f}%")
    c.set_ylim(-0.3, 8.4); c.legend(loc="upper left", fontsize=5.6)
    N.panel_label(c, "c")

    # (d) 值函数健康度
    for tag in ("ours", "ush", "uns", "ab0"):
        x, y = _curve(list(D[tag].values()), "explained_variance")
        if not len(x):
            continue
        lab, col = ARMS[tag]
        d.plot(x / 1e6, y, color=col, label=lab)
    d.set_xlabel("Training steps (millions)"); d.set_ylabel("Explained variance of value fn.")
    d.set_ylim(0, 1.02); d.legend(loc="lower right")
    N.panel_label(d, "d")

    fig.tight_layout(w_pad=2.0, h_pad=1.6)
    save_pub(fig, "Fig4_training_reliability")
    plt.close(fig)


def fig5(D):
    """盾的行为随训练演化（2×2）。"""
    fig, AX = plt.subplots(2, 2, figsize=(N.COL2, N.COL2 * 0.62))
    (a, b), (c, d) = AX

    # (a) 盾归口占比（本文方法）
    for grp, col, lab in (("projection", N.PALETTE["blue_main"], "Projection (QP)"),
                          ("emergency", N.PALETTE["red_strong"], "Emergency controller"),
                          ("fallback", N.PALETTE["violet"], "Fallback (relaxed / collision-min)")):
        x, y = _src_share(list(D["ours"].values()), grp)
        if len(x):
            a.plot(x / 1e6, y, color=col, label=lab)
    a.set_xlabel("Training steps (millions)"); a.set_ylabel("Share of control steps (%)")
    a.set_yscale("symlog", linthresh=1.0)
    a.legend(loc="lower left", bbox_to_anchor=(0.0, 0.06), fontsize=5.6)
    a.set_title("Ours: which branch produced the control", pad=3)
    N.panel_label(a, "a")

    # (b) 动作打满率两轴：有界 Beta vs 无界高斯 vs 离散网格
    for tag, ls in (("ours", "-"), ("uns", "--"), ("disc", ":")):
        lab, col = ARMS[tag]
        for key, mark in (("roll_yaw_sat_frac", None), ("roll_acc_sat_frac", "o")):
            x, y = _curve(list(D[tag].values()), key)
            if not len(x):
                continue
            b.plot(x / 1e6, 100 * y, color=col, linestyle=ls, marker=mark,
                   markersize=1.6, markevery=6, linewidth=0.9,
                   label=f"{lab} · {'yaw' if 'yaw' in key else 'accel.'}")
    b.set_xlabel("Training steps (millions)"); b.set_ylabel("Action saturation (%)")
    b.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=5.4, ncol=1)
    N.panel_label(b, "b")

    # (c) 盾改写量（按箱半宽归一化）
    for tag in ("ours", "ush", "abB", "abG"):
        x, y = _curve(list(D[tag].values()), "roll_shield_corr_norm_mean")
        if not len(x):
            continue
        lab, col = ARMS[tag]
        c.plot(x / 1e6, y, color=col, label=lab)
    c.set_xlabel("Training steps (millions)")
    c.set_ylabel("Projection correction (norm.)")
    c.legend(loc="upper right", fontsize=5.6)
    N.panel_label(c, "c")

    # (d) 会遇态势占比（本文方法）
    RHO = {"0": ("$\\rho_0$ no conflict", N.PALETTE["neutral_light"]),
           "1": ("$\\rho_1$ give-way", N.PALETTE["blue_main"]),
           "2": ("$\\rho_2$ give-way", N.PALETTE["blue_secondary"]),
           "3": ("$\\rho_3$ give-way", N.PALETTE["teal"]),
           "4": ("$\\rho_4$ stand-on", N.PALETTE["green_3"]),
           "5": ("$\\rho_5$ emergency", N.PALETTE["red_strong"])}
    runs = list(D["ours"].values())
    for k, (lab, col) in RHO.items():
        xs, ys = [], []
        for r in runs:
            for cc in r["curves"]:
                rh = cc.get("roll_rho")
                if isinstance(rh, dict):
                    tot = sum(v for v in rh.values() if isinstance(v, (int, float)))
                    if tot > 0:
                        xs.append(cc["step"]); ys.append(100.0 * rh.get(k, 0) / tot)
        if not xs:
            continue
        xs, ys = np.asarray(xs), np.asarray(ys)
        e = np.linspace(xs.min(), xs.max(), 41)
        i = np.clip(np.digitize(xs, e) - 1, 0, 39)
        bx = np.array([xs[i == j].mean() if (i == j).any() else np.nan for j in range(40)])
        by = np.array([np.median(ys[i == j]) if (i == j).any() else np.nan for j in range(40)])
        ok = ~np.isnan(bx)
        d.plot(bx[ok] / 1e6, by[ok], color=col, label=lab)
    d.set_xlabel("Training steps (millions)"); d.set_ylabel("Share of steps (%)")
    d.set_yscale("symlog", linthresh=0.5)
    d.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=5.4, ncol=1)
    d.set_title("Ours: encounter-situation profile", pad=3)
    N.panel_label(d, "d")

    fig.tight_layout(w_pad=2.0, h_pad=1.6)
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
