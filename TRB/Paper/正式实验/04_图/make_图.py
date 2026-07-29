# -*- coding: utf-8 -*-
"""正式实验出图（`Paper/正式实验/04_图/`）—— 三张图，全部只读原始产物。

  图1 学习曲线（C 10 种子 vs 金标 10 种子）—— ⚠️ 用的是**训练期里程碑数**，
       **必须在图里标死分母与"不进报数表"**（`03` L232 铁律：训练期数与 strict 563 永不混表）。
  图2 三臂消融（转艏Δ / 让路违规 / 到达）—— **同种子配对**画（`03` L234-C：跨种子集比均值是错的）。
  图3 多算法轨迹对比（同一场景四条臂叠一起，按盾判的态势上色）。

跑法：
    python3 -B Paper/正式实验/04_图/make_图.py <重评目录> [输出目录]
"""
import json
import os
import statistics as st
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
from matplotlib import font_manager   # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _common as C   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.abspath(os.path.join(HERE, "..", "..", "..", "结果"))
TRAIN_DIRS = [os.path.join(RES, x) for x in
              ("结果0702-地基第1版-12:18", "结果0710-22:00-10种子最优方案",
               "结果0727-大集指标提升", "结果0728-beta测试训练", "结果0728-正式大集smoke")]

# 中文字体：服务器/容器多半没有 ⟹ 缺就自动退英文标签，别让图挂掉或出豆腐块
_HAS_CJK = any("Noto Sans CJK" in f.name or "WenQuanYi" in f.name or "SimHei" in f.name
               for f in font_manager.fontManager.ttflist)
if _HAS_CJK:
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
L = (lambda zh, en: zh) if _HAS_CJK else (lambda zh, en: en)


def trend(tag, seed):
    for d in TRAIN_DIRS:
        for pat in (f"step4e_partial_{tag}_s{seed}.jsonl", f"step4e_partial_{tag}S{seed}.jsonl"):
            p = os.path.join(d, pat)
            if os.path.exists(p):
                with open(p, encoding="utf-8") as f:
                    return json.loads(f.read().strip().split("\n")[-1])["trend"]
    return None


def fig1_learning(out):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for k, (tag, seeds, title, col) in enumerate([
            ("L1rateON_ppo", range(10), L("旧配方（金标）", "Old recipe"), "#8c8c8c"),
            ("C231bothPpo", range(10), L("新配方 C（两个开关都上）", "New recipe C"), "#1f6feb")]):
        got = 0
        for s in seeds:
            t = trend(tag, s)
            if not t:
                continue
            got += 1
            xs = [x["step"] / 1e6 for x in t]
            ax[k].plot(xs, [x["到达率%"] for x in t], color=col, alpha=0.35, lw=1.2)
        med = []
        for i in range(10):
            v = [trend(tag, s)[i]["到达率%"] for s in seeds if trend(tag, s)]
            med.append(st.median(v))
        ax[k].plot([(i + 1) * 0.5 for i in range(10)], med, color=col, lw=2.8,
                   label=L(f"中位数（{got} 颗种子）", f"median (n={got})"))
        ax[k].set_title(title)
        ax[k].set_xlabel(L("训练步数（百万）", "steps (M)"))
        ax[k].grid(alpha=.25)
        ax[k].legend(loc="upper left", fontsize=9)
    ax[0].set_ylabel(L("到达率 %（训练期·40 场景）", "arrival % (train-time, 40 scen.)"))
    fig.suptitle(L("图1 学习曲线（训练期 40 场景里程碑数 —— 只看形状，不进任何报数表）",
                   "Fig.1 Learning curves — train-time milestone (40 scen.), NOT the reported metric"),
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "图1_学习曲线.pdf"))
    fig.savefig(os.path.join(out, "图1_学习曲线.png"), dpi=160)
    plt.close(fig)
    print("  ✅ 图1 学习曲线")


def fig2_ablation(ba, out):
    """同种子配对：四条臂共有的种子 0/1/2。"""
    names = ["金标（连续·从零·旧配方）", "A（从零·只 Beta）", "B（从零·只改状态机）", "C（从零·两个都上·主线候选）"]
    short = [L("都不改", "neither"), L("只 Beta", "Beta only"),
             L("只改状态机", "statechart only"), L("两个都上", "both")]
    if not all(n in ba for n in names):
        print("  ⚠️ 图2 跳过（缺臂）")
        return
    common = sorted(set.intersection(*[set(C.per_seed(ba[n], "到达率%")) for n in names]))
    metrics = [("yaw_incr_mean", L("转艏增量（越小越平顺）", "yaw incr. (lower=smoother)"), "{:.5f}"),
               ("giveway_violations", L("让路违规 / 局", "give-way viol./ep"), "{:.3f}"),
               ("到达率%", L("到达率 %", "arrival %"), "{:.1f}")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (key, lab, fmt) in zip(axes, metrics):
        vals = [st.mean(C.per_seed(ba[n], key)[s] for s in common) for n in names]
        bars = ax.bar(short, vals, color=["#8c8c8c", "#f0a202", "#5fa8d3", "#1f6feb"])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v), ha="center", va="bottom", fontsize=8)
        ax.set_title(lab, fontsize=10)
        ax.grid(axis="y", alpha=.25)
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle(L(f"图2 三臂消融（同种子配对 · 共同种子 {common} · strict 563）",
                   f"Fig.2 Ablation (paired seeds {common}, strict 563)"), fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "图2_三臂消融.pdf"))
    fig.savefig(os.path.join(out, "图2_三臂消融.png"), dpi=160)
    plt.close(fig)
    print(f"  ✅ 图2 三臂消融（配对种子 {common}）")


def fig3_traj(d, out):
    import glob
    import re
    trajs = {}
    for f in sorted(glob.glob(os.path.join(d, "g*_traj.json"))):
        if not re.fullmatch(r"g\d+_traj\.json", os.path.basename(f)):
            continue
        trajs.update(json.load(open(f, encoding="utf-8")))
    if not trajs:
        print("  ⚠️ 图3 跳过（没有 g*_traj.json）")
        return
    want = [("Discrete-safe", "discStdW0", "#d1495b"), ("金标", "L1rateON_ppo", "#8c8c8c"),
            ("C", "C231bothPpo", "#1f6feb")]
    pick = {}
    for lab, pat, col in want:
        hit = [k for k in trajs if pat in k]
        if hit:
            pick[lab] = (sorted(hit)[0], col)
    keys = sorted({k for v in trajs.values() for k in v})[:4]
    fig, axes = plt.subplots(1, len(keys), figsize=(4.0 * len(keys), 4.0))
    axes = axes if len(keys) > 1 else [axes]
    for ax, sk in zip(axes, keys):
        for lab, (ck, col) in pick.items():
            tr = trajs[ck].get(sk)
            if not tr:
                continue
            ax.plot([p["ego_x"] for p in tr], [p["ego_y"] for p in tr], color=col, lw=1.8, label=lab)
            ax.scatter([tr[0]["ego_x"]], [tr[0]["ego_y"]], color=col, s=18, marker="o")
        any_ck = next(iter(pick.values()))[0]
        tr = trajs[any_ck].get(sk)
        if tr:
            ax.plot([p["obs_x"] for p in tr], [p["obs_y"] for p in tr], color="#444", ls="--", lw=1.2,
                    label=L("他船", "target ship"))
        ax.set_title(L(f"场景 {sk}", f"scenario {sk}"), fontsize=10)
        ax.set_aspect("equal", "datalim")
        ax.grid(alpha=.25)
        ax.tick_params(labelsize=7)
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle(L("图3 多算法轨迹对比（同一场景·同一趟评估）",
                   "Fig.3 Trajectory comparison (same scenarios, same eval pass)"), fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "图3_轨迹对比.pdf"))
    fig.savefig(os.path.join(out, "图3_轨迹对比.png"), dpi=160)
    plt.close(fig)
    print(f"  ✅ 图3 轨迹对比（场景 {keys}）")


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    d = os.path.abspath(sys.argv[1])
    out = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else HERE
    os.makedirs(out, exist_ok=True)
    rows, n_strict, n_grp = C.load_pass(d)
    ba = C.by_arm(rows)
    print(f"[出图] 重评目录 {os.path.basename(d)} · {len(rows)} 条臂 · strict {n_strict}"
          + ("" if _HAS_CJK else "  ⚠️ 无中文字体，标签自动退英文"))
    fig1_learning(out)
    fig2_ablation(ba, out)
    fig3_traj(d, out)
    print(f"[已写入] {out}")


if __name__ == "__main__":
    main()
