# -*- coding: utf-8 -*-
"""Discussion 用的小实验：把「枚举换成投影」这句话量化。

user 2026-08-01：「discussion 我认为不只是大白话去说文字吧，是不是要做实验的？」
——对。Discussion 的核心主张是「离散动作要\\emph{枚举}才能屏蔽，而连续动作\\emph{投影}即可」，
这句话此前只有论证没有测量。本脚本给它两条可测的支撑：

  A. **分辨率**（精确算术，无随机性）：$K\\times K$ 网格能取到的最小合规转艏率比下确界
     $\\omega_{\\mathrm{turn}}$ 高多少。这是纯几何事实，与训练无关。
  B. **代价**（本机实测）：投影一次二次规划的耗时与网格无关；枚举屏蔽的耗时随 $K^2$ 增长。

🔴 **实测结论与直觉相反，正文必须照实写**（2026-08-01）：
   · A 的超出比例**不随 K 单调下降**。$5\times5$ 只超 3.1%，$7\times7$ 反而超 37.5%——
     取决于网格步长能否整除下确界，加密网格不保证更接近。所以不能写「网格越细越接近」。
   · B 里**投影比枚举贵得多**：本机投影约 0.8 ms/步，而 $K$ 取到 121（14641 个动作）时
     枚举屏蔽仍只要 83 μs。⟹ **绝不能把投影写成"更省算力"**，它的理由是能力不是速度
     （连续动作集上枚举根本无从谈起，且投影能精确取到下确界）。

🔴 只读冻结代码，不改。`代码/trb_env/usv_projection.py` 因依赖 `vesselmodels` 无法整包导入，
   故按源码位置**摘出** `_solve_box_halfplane_qp` 单独执行——摘的是原函数正文，逐字未改。

🔴 B 的绝对耗时**依赖本机**，不能当作论文里"部署时的单步耗时"。论文报数的那个耗时
   必须来自重评产物（评估侧统一机器）。本脚本只用来说明**随 K 的增长关系**。

跑法：<venv>/bin/python 结果/结果0801-出图草图/exp_grid_vs_projection.py
"""
import math
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_style as PS                                  # noqa: E402
import runs_data as R                                     # noqa: E402
import matplotlib.pyplot as plt                           # noqa: E402

SRC = os.path.join(R.ROOT, "代码", "trb_env", "usv_projection.py")

# ── 常量：全部取自冻结代码，不另定 ─────────────────────────────────────────
#   `代码/trb_env/usv_env.py:53-54` 的 7×7 网格 = A_a × A_ω，满格 ±0.048 / ±0.018
A_MAX, W_MAX = 0.048, 0.018
#   「足够明显」的量化：机动时段 40 s 内航向改变不少于 20°（`usv_colregs.py:698` 的 Δlarge_turn/t_m）
T_MANEUVER, DTHETA = 40.0, math.radians(20.0)
W_TURN = DTHETA / T_MANEUVER                              # 合规转艏率下确界 rad/s


def load_qp():
    """从冻结源码里摘出 `_solve_box_halfplane_qp` 单独执行（整包导入要 vesselmodels）。"""
    s = open(SRC, encoding="utf-8").read()
    i = s.index("def _solve_box_halfplane_qp(")
    j = s.index("\ndef ", i + 1)
    ns = {"np": np}
    exec(compile(s[i:j], SRC, "exec"), ns)
    return ns["_solve_box_halfplane_qp"]


def grid_excess(K):
    """$K\\times K$ 网格（对称、含 0、满格 ±W_MAX）能取到的最小合规 $|\\omega|$ 及其超出比例。

    合规要求 $|\\omega|\\ge\\omega_{\\mathrm{turn}}$，网格只能取到格点，故必然取到不小于它的最近格点。
    """
    step = W_MAX / ((K - 1) / 2)
    n = math.ceil(W_TURN / step - 1e-12)                  # 最近的、不小于下确界的格点序号
    w = n * step
    if w > W_MAX + 1e-12:                                 # 网格太粗，满格都够不到合规下界
        return None, None
    return w, (w / W_TURN - 1.0) * 100.0


def main():
    PS.apply()
    qp = load_qp()
    rng = np.random.default_rng(0)

    print(f"合规转艏率下确界 ω_turn = {W_TURN:.6f} rad/s"
          f"（{math.degrees(DTHETA):.0f}° / {T_MANEUVER:.0f} s）\n")

    KS = [3, 5, 7, 9, 11, 15, 21, 31, 51, 81, 121]
    print("A. 分辨率（精确算术）")
    rows = []
    for K in KS:
        w, ex = grid_excess(K)
        rows.append((K, w, ex))
        print(f"   {K:3d}×{K:<3d} 最小合规 |ω| = "
              + (f"{w:.6f}  超出下确界 {ex:5.1f}%" if w else "满格都够不到合规下界"))

    # ── B. 单步代价 ────────────────────────────────────────────────────────
    #   投影：一次 box + 半平面的二次规划，规模与 K 无关
    #   枚举：把 K² 个格点逐个验合规半平面与动作箱，再取策略打分最高的可行者
    N = 400
    des = rng.uniform([-A_MAX, -W_MAX], [A_MAX, W_MAX], size=(N, 2))
    row = [(np.array([0.0, 1.0]), -W_TURN)]               # ω ≤ −ω_turn（右转合规半平面）
    for _ in range(30):                                   # 预热（首次 setup 含编译开销）
        qp(des[0], (-A_MAX, A_MAX), (-W_MAX, W_MAX), row)
    t0 = time.perf_counter()
    for u in des:
        qp(u, (-A_MAX, A_MAX), (-W_MAX, W_MAX), row)
    t_qp = (time.perf_counter() - t0) / N * 1e6           # μs/步

    print(f"\nB. 单步代价（本机，仅用于看随 K 的增长关系，不作论文报数）")
    print(f"   投影（二次规划，与 K 无关）  {t_qp:8.1f} μs/步")
    t_enum = {}
    for K in KS:
        aa, ww = np.meshgrid(np.linspace(-A_MAX, A_MAX, K), np.linspace(-W_MAX, W_MAX, K))
        grid = np.stack([aa.ravel(), ww.ravel()], 1)
        logits = rng.normal(size=(N, K * K))
        t0 = time.perf_counter()
        for i in range(N):
            ok = grid[:, 1] <= -W_TURN + 1e-12            # 合规半平面
            ok &= (np.abs(grid[:, 0]) <= A_MAX) & (np.abs(grid[:, 1]) <= W_MAX)
            if ok.any():
                np.argmax(np.where(ok, logits[i], -np.inf))
        t_enum[K] = (time.perf_counter() - t0) / N * 1e6
        print(f"   枚举屏蔽 K={K:3d}（{K*K:6d} 个动作） {t_enum[K]:8.1f} μs/步")

    # ── 出图：一张 1×2，左=分辨率、右=代价 ────────────────────────────────
    fig, (a, b) = plt.subplots(1, 2, figsize=(PS.COL2 * 0.80, PS.COL2 * 0.30))
    ok = [(K, ex) for K, w, ex in rows if w]
    a.bar([str(K) for K, _ in ok], [ex for _, ex in ok], width=0.66,
          color=PS.PALETTE["blue_main"], alpha=0.32,
          edgecolor=PS.PALETTE["blue_main"], linewidth=0.9)
    for i, (K, ex) in enumerate(ok):
        if K in (7, 21, 121):
            a.annotate(f"{ex:.1f}%", (i, ex), xytext=(0, 3), textcoords="offset points",
                       ha="center", fontsize=6.2)
    a.axhline(0, color=PS.PALETTE["red_strong"], linewidth=1.2, zorder=4)
    a.annotate("projection", (0.985, 0.045), xycoords="axes fraction", ha="right",
               fontsize=6.4, color=PS.PALETTE["red_strong"])
    a.set_xlabel("Discrete grid size $K$ (per axis)")
    a.set_ylabel("Excess over $\\omega_{\\mathrm{turn}}$ (%)")
    a.set_title("(a) Compliance-turn granularity")
    a.grid(axis="x", visible=False)

    b.plot(KS, [t_enum[K] for K in KS], marker="o", markersize=2.6,
           color=PS.PALETTE["neutral_black"], linewidth=1.2)
    b.axhline(t_qp, color=PS.PALETTE["blue_main"], linewidth=1.4)
    b.annotate("projection", (0.03, t_qp), xycoords=("axes fraction", "data"),
               xytext=(0, 4), textcoords="offset points", fontsize=6.4,
               color=PS.PALETTE["blue_main"])
    b.annotate("enumerate $+$ mask", (KS[-1], t_enum[KS[-1]]), xytext=(-4, -9),
               textcoords="offset points", ha="right", fontsize=6.4,
               color=PS.PALETTE["neutral_black"])
    b.set_xscale("log"); b.set_yscale("log")
    b.set_xlabel("Discrete grid size $K$ (per axis)")
    b.set_ylabel("Time per decision step ($\\mu$s)")
    b.set_title("(b) Per-step cost")
    fig.tight_layout(w_pad=1.8)
    PS.save(fig, "Fig7_grid_vs_projection", R.OUT_DIRS)
    plt.close(fig)

    # ── 给正文用的结论行 ───────────────────────────────────────────────────
    k7 = dict((K, ex) for K, w, ex in rows if w)[7]
    fine = next(K for K, w, ex in rows if w and ex < 5.0)
    print(f"\n结论：7×7 网格超出下确界 {k7:.1f}%；要压到 5% 以内需 K≥{fine}"
          f"（{fine*fine} 个动作，是 49 的 {fine*fine/49:.0f} 倍），"
          f"此时枚举屏蔽耗时约为投影的 {t_enum[fine]/t_qp:.2f} 倍。")


if __name__ == "__main__":
    main()
