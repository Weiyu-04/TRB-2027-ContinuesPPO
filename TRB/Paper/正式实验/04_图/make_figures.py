# -*- coding: utf-8 -*-
"""正式实验出图 —— 遵循 nature-figure 技能（Python 后端）的五点契约。

═══ 图的契约（按规范要求，先写清楚再画）═══════════════════════════════════════
【Fig.1 样本效率 · hero】
  核心结论（一句话）：在**同一 5.08M 步训练预算**下，新配方把学习速度提高约一个数量级；
    旧配方到期时**仍未收敛** ⟹ 这是**样本效率**声明，不是"收敛后更好"。
  证据链：a 学习曲线（hero·旧 vs 新）→ 速度差；b 同种子配对哑铃图 → 逐颗都变好、非均值假象；
          c 到期仍在上升的种子数 → 坐实"预算点非收敛点"（`03` L236-A）。
  形制：quantitative grid（一个 hero + 两个从属证据面板）。

【Fig.2 消融 · 同种子配对】
  核心结论：两把钥匙**各治各的**——有界动作分布治转艏抖动，状态机修法治让路违规，缺一不可。
  证据链：a 转艏增量（hero）；b 让路违规；c 到达率。三格都只用**四条臂共有的种子**。
  形制：quantitative grid。

【Fig.3 安全-效率权衡（钱图）】
  核心结论：带盾的连续方案位于 Pareto 前沿——违规与碰撞都低，且到达率不输无盾基线太多。
  形制：asymmetric mixed-modality（散点 + 直接标注，不用图例）。

【Fig.4 轨迹】
  核心结论：投影盾产出的是平顺的右转让路机动，离散基线是满舵摆动。
  形制：image-plate 式小多图（同一批场景、同一趟评估）。
═══════════════════════════════════════════════════════════════════════════════

统计口径（规范要求写进图里，不是"标题清洁工作"）：
  · n = 种子数（**种子才是独立单位**，不是 episode）
  · 中心 = 均值；误差棒 = **按种子重采样的 95% 自助法区间**（B=20000·固定随机种子 ⟹ 可复现）
  · 每张图脚注标死："5.08M-step budget · last checkpoint · official test set, strict 563 · single machine, single pass"

跑法：
    python3 -B Paper/正式实验/04_图/make_figures.py <重评目录> [输出目录]
"""
import glob
import json
import os
import random
import re
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import nature_style as N   # noqa: E402
import _common as C        # noqa: E402

import matplotlib as mpl   # noqa: E402

mpl.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

# ═══ 强制项（nature-figure 规范：**每个绘图脚本顶部都要有**，不可只放在共享层）═══
#   svg.fonttype='none' ⟹ SVG 里文字保持 <text> 节点（可选中/可搜索/可在 Illustrator 重排）；
#   matplotlib 默认 'path' 会把每个字形变成贝塞尔轮廓，事后一个字都改不了。
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})
N.apply_publication_style()          # 其余版式（字号/线宽/去上右边框/无图例框）

# ── 最终成图宽度：**写死**（规范要求最终尺寸静态可读、投稿前就定死，不许出图时才算）──
fig_width_mm = 183          # 双栏
fig_width_mm_single = 89    # 单栏
RASTER_DPI = 600
# 各图英寸宽 = 上面的毫米数 ÷ 25.4：7.20 in = 183 mm（双栏）· 4.72 in = 120 mm · 3.50 in = 89 mm（单栏）


def save_pub(fig, path_no_ext):
    """SVG 为主（可编辑）+ PDF（矢量投稿）+ TIFF（投稿栅格）+ PNG（预览/汇报）。"""
    fig.savefig(path_no_ext + ".svg", bbox_inches="tight")
    fig.savefig(path_no_ext + ".pdf", bbox_inches="tight")
    fig.savefig(path_no_ext + ".tiff", dpi=RASTER_DPI, bbox_inches="tight")
    fig.savefig(path_no_ext + ".png", dpi=RASTER_DPI, bbox_inches="tight")
    plt.close(fig)

RES = os.path.abspath(os.path.join(HERE, "..", "..", "..", "结果"))
TRAIN_DIRS = [os.path.join(RES, x) for x in
              ("结果0702-地基第1版-12:18", "结果0710-22:00-10种子最优方案",
               "结果0727-大集指标提升", "结果0728-beta测试训练", "结果0728-正式大集smoke")]
RNG = random.Random(20260729)
B_BOOT = 20000
FOOT = ("5.08M-step budget · last checkpoint · official test set (strict 563) · single machine, single pass")

#: 中文臂名 → 图上用的英文名（图内一律英文；中文只留在代码与 README）
EN = {
    "Base（离散·无盾）": "Base (discrete, no shield)",
    "Rule-reward（离散·软奖励）": "Rule-reward (discrete, soft reward)",
    "Discrete-safe（对标论文）": "Discrete-safe (benchmark)",
    "金标（连续·从零·旧配方）": "Gold standard (continuous, old recipe)",
    "主线（连续·热启动·旧配方）": "Warm-start (continuous, old recipe)",
    "C（从零·两个都上·主线候选）": "Ours (continuous + projection shield)",
    "A（从零·只 Beta）": "Beta only",
    "B（从零·只改状态机）": "Statechart only",
    "大集探针（新配方 D）": "Large-set probe",
}


def boot_ci(xs, alpha=0.05):
    """按**种子**重采样的百分位自助法区间。种子少于 2 颗就不画误差棒（不编区间）。"""
    if len(xs) < 2:
        return 0.0, 0.0
    vals = sorted(st.mean([xs[RNG.randrange(len(xs))] for _ in xs]) for _ in range(B_BOOT))
    m = st.mean(xs)
    return m - vals[int(alpha / 2 * B_BOOT)], vals[min(B_BOOT - 1, int((1 - alpha / 2) * B_BOOT))] - m


def trend(tag, seed):
    for d in TRAIN_DIRS:
        for pat in (f"step4e_partial_{tag}_s{seed}.jsonl", f"step4e_partial_{tag}S{seed}.jsonl"):
            p = os.path.join(d, pat)
            if os.path.exists(p):
                with open(p, encoding="utf-8") as f:
                    return json.loads(f.read().strip().split("\n")[-1])["trend"]
    return None


def fig1_sample_efficiency(ba, out):
    """hero = 学习曲线；从属 = 配对哑铃 + 到期仍在上升的种子数。"""
    fig = plt.figure(figsize=(7.20, 2.55))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.0, 1.15, 0.8], wspace=0.42)
    ax_a, ax_b, ax_c = (fig.add_subplot(gs[0, i]) for i in range(3))

    # ── a：学习曲线（hero）───────────────────────────────────────────────────
    series = [("L1rateON_ppo", range(10), "Old recipe", PALETTE_OLD := N.PALETTE["neutral_mid"]),
              ("C231bothPpo", range(10), "Ours", N.PALETTE["blue_main"])]
    for tag, seeds, lab, col in series:
        ts = [t for t in (trend(tag, s) for s in seeds) if t]
        if not ts:
            continue
        for t in ts:
            ax_a.plot([x["step"] / 1e6 for x in t], [x["到达率%"] for x in t],
                      color=col, alpha=0.22, lw=0.5)
        n_seg = min(len(t) for t in ts)
        xs = [ts[0][i]["step"] / 1e6 for i in range(n_seg)]
        med = [st.median([t[i]["到达率%"] for t in ts]) for i in range(n_seg)]
        ax_a.plot(xs, med, color=col, lw=1.8)
        ax_a.text(xs[-1] + 0.06, med[-1], f"{lab}\n(n={len(ts)})", color=col,
                  fontsize=6, va="center", ha="left", fontweight="bold")   # 直接标注，不用图例
    ax_a.set_xlim(0.3, 6.6)
    ax_a.set_ylim(-3, 105)
    ax_a.set_xlabel("Training steps (millions)")
    ax_a.set_ylabel("Arrival rate (%)\ntrain-time milestone, 40 scenarios")
    ax_a.set_xticks([1, 2, 3, 4, 5])
    ax_a.grid(axis="y", alpha=0.2, lw=0.4)
    N.panel_label(ax_a, "a", x=-0.13)

    # ── b：同种子配对哑铃（金标 → 我们）─────────────────────────────────────
    g, c = "金标（连续·从零·旧配方）", "C（从零·两个都上·主线候选）"
    if g in ba and c in ba:
        ag, ac = C.per_seed(ba[g], "到达率%"), C.per_seed(ba[c], "到达率%")
        common = sorted(set(ag) & set(ac))
        order = sorted(common, key=lambda s: ag[s])
        for i, s in enumerate(order):
            ax_b.plot([ag[s], ac[s]], [i, i], color=N.PALETTE["neutral_light"], lw=0.9, zorder=1)
            ax_b.scatter([ag[s]], [i], s=9, color=N.PALETTE["neutral_mid"], zorder=2)
            ax_b.scatter([ac[s]], [i], s=9, color=N.PALETTE["blue_main"], zorder=2)
        ax_b.set_yticks(range(len(order)))
        ax_b.set_yticklabels([f"s{s}" for s in order])
        ax_b.set_xlabel("Arrival rate (%), strict 563")
        n_up = sum(1 for s in common if ac[s] > ag[s])
        ax_b.set_title(f"paired by seed: {n_up}/{len(common)} improved\ntwo-sided sign test p = {2/2**len(common):.2e}",
                       fontsize=6, pad=3)
        ax_b.grid(axis="x", alpha=0.2, lw=0.4)
    N.panel_label(ax_b, "b", x=-0.30)

    # ── c：到期仍在明显上升的种子数（"预算点非收敛点"的直接证据）────────────
    bars = []
    for tag, seeds, lab in [("L1rateON_ppo", range(10), "Old\nrecipe"), ("C231bothPpo", range(10), "Ours")]:
        up = tot = 0
        for s in seeds:
            t = trend(tag, s)
            if not t:
                continue
            a = [x["到达率%"] for x in t]
            if st.mean(a[-3:]) < 20:          # 崩掉的种子另论，不参与"还能不能练出来"
                continue
            tot += 1
            up += (st.mean(a[-3:]) - st.mean(a[4:7])) > 9.0    # 2×SE(120 局)
        bars.append((lab, up, tot))
    xs = range(len(bars))
    ax_c.bar(xs, [b[1] / b[2] * 100 for b in bars], width=0.55,
             color=[N.PALETTE["neutral_mid"], N.PALETTE["blue_main"]])
    for i, (lab, up, tot) in enumerate(bars):
        ax_c.text(i, up / tot * 100 + 2, f"{up}/{tot}", ha="center", fontsize=6)
    ax_c.set_xticks(list(xs))
    ax_c.set_xticklabels([b[0] for b in bars])
    ax_c.set_ylim(0, 118)
    ax_c.set_ylabel("Seeds still improving\nat the 5.08M cutoff (%)")
    ax_c.grid(axis="y", alpha=0.2, lw=0.4)
    N.panel_label(ax_c, "c", x=-0.42)

    fig.text(0.008, -0.10, "Fig. 1 | Sample efficiency under an identical training budget. "
             "a, Learning curves (thin: individual seeds; thick: median). b, Same-seed pairing on the official "
             "test set. c, Fraction of non-collapsed seeds whose last three segments still improve by more than "
             "two standard errors — the old recipe has not converged at the cutoff, so the comparison is a "
             "sample-efficiency claim, not a converged-performance claim. " + FOOT,
             fontsize=5.4, va="top", ha="left", wrap=True)
    save_pub(fig, os.path.join(out, "Fig1_sample_efficiency"))
    print("  ✅ Fig.1 sample efficiency")


def fig2_ablation(ba, out):
    names = ["金标（连续·从零·旧配方）", "A（从零·只 Beta）", "B（从零·只改状态机）", "C（从零·两个都上·主线候选）"]
    short = ["neither", "bounded\naction dist.", "statechart\nfix", "both\n(ours)"]
    cols = [N.PALETTE["neutral_mid"], N.PALETTE["green_3"], N.PALETTE["violet"], N.PALETTE["blue_main"]]
    if not all(n in ba for n in names):
        print("  ⚠️ Fig.2 跳过（缺臂）")
        return
    common = sorted(set.intersection(*[set(C.per_seed(ba[n], "到达率%")) for n in names]))
    panels = [("yaw_incr_mean", "Yaw increment per step\n(lower = smoother)", "{:.4f}"),
              ("giveway_violations", "Give-way violations\nper episode", "{:.3f}"),
              ("到达率%", "Arrival rate (%)", "{:.1f}")]
    fig, axes = plt.subplots(1, 3, figsize=(7.20, 2.15))
    for k, (ax, (key, lab, fmt)) in enumerate(zip(axes, panels)):
        vals, errs = [], []
        for n in names:
            per = C.per_seed(ba[n], key)
            xs = [per[s] for s in common]
            vals.append(st.mean(xs))
            errs.append(boot_ci(xs))
        ax.bar(range(4), vals, width=0.6, color=cols,
               yerr=[[e[0] for e in errs], [e[1] for e in errs]],
               error_kw=dict(elinewidth=0.7, capthick=0.7, capsize=2, ecolor=N.PALETTE["neutral_dark"]))
        for i, v in enumerate(vals):
            ax.text(i, v + errs[i][1] + max(vals) * 0.03, fmt.format(v), ha="center", fontsize=5.6)
        ax.set_xticks(range(4))
        ax.set_xticklabels(short, fontsize=5.8)
        ax.set_ylabel(lab)
        ax.set_ylim(0, max(v + e[1] for v, e in zip(vals, errs)) * 1.22)
        ax.grid(axis="y", alpha=0.2, lw=0.4)
        N.panel_label(ax, "abc"[k], x=-0.22)
    fig.tight_layout()
    fig.text(0.008, -0.08,
             f"Fig. 2 | Ablation of the two mechanisms, paired on the {len(common)} seeds shared by all four arms "
             f"(seeds {common}). Bars, mean over seeds; error bars, 95% bootstrap CI resampled over seeds "
             f"(B={B_BOOT:,}). a, Only the bounded action distribution removes rudder chatter — the statechart fix "
             "alone makes it slightly worse. b, The statechart fix is what drives give-way violations down. "
             "c, Neither alone reaches the arrival rate of the combination. " + FOOT,
             fontsize=5.4, va="top", ha="left", wrap=True)
    save_pub(fig, os.path.join(out, "Fig2_ablation"))
    print(f"  ✅ Fig.2 ablation (paired seeds {common})")


def fig3_tradeoff(ba, out):
    """钱图：违规 vs 到达，点面积 ∝ 碰撞率。**直接标注不用图例** ⟹ 标签必须逐点定偏移防重叠。"""
    # 短名（图上不塞括号说明·长名进图注）+ 逐点标签偏移（视觉 QA 实测定的，防重叠/防出血）
    SHORT = {"Base (discrete, no shield)": "Base", "Rule-reward (discrete, soft reward)": "Rule-reward",
             "Discrete-safe (benchmark)": "Discrete-safe", "Gold standard (continuous, old recipe)": "Gold std.",
             "Warm-start (continuous, old recipe)": "Warm-start",
             "Ours (continuous + projection shield)": "Ours", "Beta only": "Beta only",
             "Statechart only": "Statechart only", "Large-set probe": "Large-set probe"}
    OFF = {"Base": (0, -11), "Rule-reward": (0, 9), "Discrete-safe": (5, 4), "Gold std.": (5, -9),
           "Warm-start": (-6, -11), "Ours": (-4, 7), "Beta only": (5, 4),
           "Statechart only": (5, -9), "Large-set probe": (5, 4)}
    fig, ax = plt.subplots(figsize=(4.72, 2.80))
    pts = []
    for zh, en in EN.items():
        if zh in ba:
            m = C.metrics(ba[zh])
            pts.append((SHORT[en], en, m["到达"], m["违规"], m["碰撞率"], m["n"]))
    for sh, en, arr, vio, col_rate, n in pts:
        c = N.ARM_COLOR.get(en, N.PALETTE["neutral_mid"])
        ax.scatter([arr], [vio], s=14 + col_rate * 110, color=c,
                   edgecolor="white", linewidth=0.5, zorder=3)
        dx, dy = OFF.get(sh, (5, 4))
        ax.annotate(f"{sh} (n={n})", (arr, vio), textcoords="offset points", xytext=(dx, dy),
                    fontsize=5.2, color=c, fontweight="bold",
                    ha="right" if dx < 0 else ("center" if dx == 0 else "left"))
    xs = [p[2] for p in pts]
    ys = [p[3] for p in pts]
    ax.set_xlim(min(xs) - 14, max(xs) + 10)          # 留白给标签，别让字出血
    ax.set_ylim(max(ys) + 0.30, min(ys) - 0.22)      # y 轴反向（越低越好）⟹ 上界在前
    ax.set_xlabel("Arrival rate (%)   →  better")
    ax.set_ylabel("COLREGs violations per episode\n←  better")
    ax.grid(alpha=0.2, lw=0.4)
    ax.text(0.015, 0.03, "marker area ∝ collision rate", transform=ax.transAxes,
            fontsize=5.2, color=N.PALETTE["neutral_dark"], va="bottom")
    fig.text(0.008, -0.13, "Fig. 3 | Safety-efficiency trade-off on the official test set. "
             "Each marker is one arm (mean over its seeds); marker area is proportional to the pooled collision "
             "rate. Up-and-left is better. Arms: Base and Rule-reward are discrete without a shield; "
             "Discrete-safe is the discrete action-masking benchmark; Gold standard and Warm-start are the "
             "previous continuous recipe; Beta only and Statechart only are single-mechanism ablations. "
             + FOOT, fontsize=5.4, va="top", ha="left", wrap=True)
    save_pub(fig, os.path.join(out, "Fig3_tradeoff"))
    print("  ✅ Fig.3 trade-off")


def fig4_traj(d, out):
    trajs = {}
    for f in sorted(glob.glob(os.path.join(d, "g*_traj.json"))):
        if re.fullmatch(r"g\d+_traj\.json", os.path.basename(f)):
            trajs.update(json.load(open(f, encoding="utf-8")))
    if not trajs:
        print("  ⚠️ Fig.4 跳过（没有 g*_traj.json）")
        return
    want = [("Discrete-safe (benchmark)", "discStdW0"), ("Gold standard (continuous, old recipe)", "L1rateON_ppo"),
            ("Ours (continuous + projection shield)", "C231bothPpo")]
    pick = {lab: sorted([k for k in trajs if pat in k])[0]
            for lab, pat in want if any(pat in k for k in trajs)}
    keys = sorted({k for v in trajs.values() for k in v})[:4]
    fig, axes = plt.subplots(1, len(keys), figsize=(7.20, 2.05))
    axes = list(axes) if len(keys) > 1 else [axes]
    for i, (ax, sk) in enumerate(zip(axes, keys)):
        for lab, ck in pick.items():
            tr = trajs[ck].get(sk)
            if not tr:
                continue
            col = N.ARM_COLOR[lab]
            ax.plot([p["ego_x"] / 1000 for p in tr], [p["ego_y"] / 1000 for p in tr], color=col, lw=1.0)
            ax.scatter([tr[0]["ego_x"] / 1000], [tr[0]["ego_y"] / 1000], s=7, color=col, zorder=3)
        tr = trajs[next(iter(pick.values()))].get(sk)
        if tr:
            ax.plot([p["obs_x"] / 1000 for p in tr], [p["obs_y"] / 1000 for p in tr],
                    color=N.PALETTE["neutral_black"], ls=(0, (3, 2)), lw=0.7)
        ax.set_title(f"scenario {sk}", fontsize=6, pad=2)
        ax.set_aspect("equal", "datalim")
        ax.grid(alpha=0.18, lw=0.4)
        ax.set_xlabel("x (km)")
        if i == 0:
            ax.set_ylabel("y (km)")
        N.panel_label(ax, "abcd"[i], x=-0.20)
    # 直接标注代替图例（放第一格）
    for j, (lab, _) in enumerate(pick.items()):
        axes[0].text(0.03, 0.97 - j * 0.09, lab.split(" (")[0], transform=axes[0].transAxes,
                     fontsize=5.2, color=N.ARM_COLOR[lab], fontweight="bold", va="top")
    axes[0].text(0.03, 0.97 - len(pick) * 0.09, "target ship", transform=axes[0].transAxes,
                 fontsize=5.2, color=N.PALETTE["neutral_black"], va="top")
    fig.tight_layout()
    fig.text(0.008, -0.10, "Fig. 4 | Own-ship trajectories on four representative encounters (solid) with the "
             "target ship (dashed), all taken from the same evaluation pass. One seed per arm; markers denote "
             "the start point. " + FOOT, fontsize=5.4, va="top", ha="left", wrap=True)
    save_pub(fig, os.path.join(out, "Fig4_trajectories"))
    print(f"  ✅ Fig.4 trajectories (scenarios {keys})")


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    d = os.path.abspath(sys.argv[1])
    out = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else HERE
    os.makedirs(out, exist_ok=True)
    rows, n_strict, n_grp = C.load_pass(d)     # fail-closed
    ba = C.by_arm(rows)
    print(f"[figures] {os.path.basename(d)} · {len(rows)} arms · strict {n_strict} · {n_grp} groups")
    fig1_sample_efficiency(ba, out)
    fig2_ablation(ba, out)
    fig3_tradeoff(ba, out)
    fig4_traj(d, out)
    print(f"[written] {out}  (svg 主 + pdf + png)")


if __name__ == "__main__":
    main()
