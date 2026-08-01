# -*- coding: utf-8 -*-
"""图 6：多算法轨迹对比（`04 §1.5 ⑥` user 明令「可视化必须含多算法轨迹对比 + 各指标对比」）。

🔴 **场景怎么挑 —— 只按几何，绝不看成绩**（`代码/tests/pick_traj_keys.py` 立的规矩）：
   看了各臂跑得好不好再决定画哪个场景 = 变相 cherry-pick，审稿人问一句就说不清。
   本脚本的规则**事先声明、确定性、与任何算法无关**：
     ① 用冻结代码 `代码/make_official_manifest.py` 的 `classify_all` 给全库 2000 个场景分类
        （它自带与 `03` L111 记录的硬比对，对不上会中止）；
     ② 取官方**测试集 600** 里属于该会遇类型的键，取**常速最近会遇距离最小**的那一个；
     ③ 种子固定 **s0**（轨迹专趟只采了 s0/s1/s2，事先声明的那三颗）。
   ⟹ 换台机器重跑，挑出来的场景一模一样。图注必须写明场景号与种子号。

   🔴 **为什么不是"取正中间那一个"**（第一版就是这么挑的，作废）：实测那样挑出来的
   T-920 / T-992 **全程 ρ=0**，双方根本没走近到状态机会判态势 —— 画出来三条线几乎重合，
   一点信息都没有。改判据用的仍是**纯几何量**：双方都按第 0 步速度直行时的最近距离
   （只吃两船初始位姿 + 他船 0→1 位移，与任何算法无关），取该类型里最小的那个。

   🔴 **实测：这两个场景里本船是【直航船】（ρ=1），不是让路船。** 全库让路触发率本就只有
   ~13-16%（`CLAUDE.md` §0）。直航义务是"保向保速"，而**直航违规正是本文违规量的大头**
   （Ours 每局 0.53 里让路只占 0.17）⟹ 画直航段比画让路段更贴主线，不是退而求其次。

🔴 **轨迹专趟用的是【末段存档】**（`run_reeval_all.sh` 的 traj 分支直接拼臂名、不读最佳存档清单，
   `03` L243-续48 E）⟹ 与主表（验证集最佳存档）**不同源**，图注必须写明，别让人默认一致。

跑法：<venv>/bin/python 结果/结果0801-出图草图/make_fig6_traj.py
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_style as PS                                  # noqa: E402
import runs_data as R                                     # noqa: E402
import matplotlib.pyplot as plt                           # noqa: E402

TRAJ_DIR = os.path.join(R.ROOT, "Paper", "正式实验", "02_重评产物", "正式-轨迹全集")
SEED = 0                                    # 事先声明的三颗种子里的第一颗
#: 画哪几条臂（标签一律取 `runs_data.ARMS`，不在这里另起名字）
SHOW = ["base", "disc", "ours"]
DT = 10.0                                   # 决策周期（秒）


def cv_cpa(st):
    """双方按第 0 步速度直行时的最近距离（米）。**纯几何**：只吃两船初始位姿与他船 0→1 位移。"""
    if len(st) < 2:
        return None
    p = np.array([st[0]["ego_x"], st[0]["ego_y"]])
    q = np.array([st[0]["obs_x"], st[0]["obs_y"]])
    vq = (np.array([st[1]["obs_x"], st[1]["obs_y"]]) - q) / DT
    vp = st[0]["ego_v"] * np.array([np.cos(st[0]["ego_psi"]), np.sin(st[0]["ego_psi"])])
    dr, dv = q - p, vq - vp
    n = float(dv @ dv)
    t = 0.0 if n < 1e-12 else max(0.0, -float(dr @ dv) / n)
    return float(np.linalg.norm(dr + dv * t))


def scenario_of_type():
    """→ {'head-on': tid, 'crossing': tid}，按上面声明的规则确定性地挑。"""
    code = os.path.join(R.ROOT, "代码")
    sys.path.insert(0, code)
    from make_official_manifest import official_split, classify_all
    sdir = os.environ.get("STEP4E_SDIR") or os.path.join(R.ROOT, "scenarios")
    _train, test = official_split()
    kind = classify_all(sdir)
    #: 常速最近会遇距离要逐场景算 ⟹ 借任意一条臂的记录取第 0/1 步（那两步与算法无关）
    ref = None
    for f in sorted(glob.glob(os.path.join(TRAJ_DIR, "g*_traj.json"))):
        d = json.load(open(f, encoding="utf-8"))
        for name, per in d.items():
            if name.endswith(f"_s{SEED}_F240basePpoS{SEED}"):
                ref = per
                break
        del d
        if ref:
            break
    if not ref:
        raise SystemExit("🔒 轨迹产物里找不到用来算几何量的那条臂")
    dist = {int(k): cv_cpa(v) for k, v in ref.items() if cv_cpa(v) is not None}
    out = {}
    for ty in ("head-on", "crossing"):
        ids = [t for t in dist if kind.get(t) == ty and t in set(test)]
        ids.sort(key=lambda t: dist[t])
        out[ty] = ids[0]
        print(f"  {ty:<9} 测试集里 {len(ids)} 个 → 常速最近会遇距离最小的是 T-{out[ty]}"
              f"（{dist[out[ty]]:.0f} m）")
    return out


def load_tracks(tids):
    """从四个 g*_traj.json 里只取需要的 (臂, 场景)，逐个文件读完就释放。"""
    want = {}
    for tag in SHOW:
        want[tag] = None
    got = {tag: {} for tag in SHOW}
    for f in sorted(glob.glob(os.path.join(TRAJ_DIR, "g*_traj.json"))):
        d = json.load(open(f, encoding="utf-8"))
        for name, per_sc in d.items():
            for tag in SHOW:
                pat = f"_s{SEED}_F240{tag}PpoS{SEED}"
                if name.endswith(pat):
                    for tid in tids:
                        k = str(tid)
                        if k in per_sc:
                            got[tag][tid] = per_sc[k]
        del d
    miss = [(t, i) for t in SHOW for i in tids if i not in got[t]]
    if miss:
        raise SystemExit(f"🔒 轨迹产物里缺这些 (臂, 场景)：{miss}")
    return got


def draw_track(ax, steps, color, lw=1.2, ls="-"):
    x = np.array([s["ego_x"] for s in steps]) / 1000.0
    y = np.array([s["ego_y"] for s in steps]) / 1000.0
    ax.plot(x, y, color=color, linewidth=lw, linestyle=ls, zorder=3)
    ax.scatter(x[-1], y[-1], s=9, color=color, zorder=4, linewidths=0)
    return x, y


def main():
    PS.apply()
    print("挑场景（只按几何，与成绩无关）：")
    pick = scenario_of_type()
    tids = [pick["head-on"], pick["crossing"]]
    print("载入轨迹…")
    T = load_tracks(tids)

    #: 🔴 2026-08-01 later-11（user）：加一行**拉长的动力学格**。
    #  理由：全文此前**没有任何一张图画「执行的命令随时间怎么变」**——表 3 只给转艏增量一个
    #  聚合数字，看不出过程。而这两个会遇里本船都是**直航船**，直航义务正是"保向保速"，
    #  所以转艏率与航速的时间曲线，直接把「Ours 稳住、两条离散臂来回摆」画出来，
    #  正是直航违规计数在数的那件事（本文违规量的大头）。
    #  版面：上排三格（轨迹×2 + 距离），下排两格拉通（转艏率、航速）。
    #: 🔴 2026-08-01 later-13（user）：改成**两列 × 三行**——左列对遇、右列交叉，
    #  每个案例的动力学**对齐放在它自己的轨迹图正下方**，两列可以横向对读。
    #  行 1 轨迹（比原来窄，给下面两行让位）· 行 2 与他船的距离 · 行 3 转艏率。
    #  同一行的两格共用纵轴范围，左右可比。
    fig = plt.figure(figsize=(PS.COL2, PS.COL2 * 0.62))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.25, 0.62, 0.62], hspace=0.55, wspace=0.22)
    AXT = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    AXR = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    AXW = [fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])]
    EPS_W = 0.004363323129985824
    TY = ["head-on", "crossing"]

    for col, ty in enumerate(TY):
        tid = pick[ty]
        ax = AXT[col]
        ref = T[SHOW[0]][tid]
        ox = np.array([s2["obs_x"] for s2 in ref]) / 1000.0
        oy = np.array([s2["obs_y"] for s2 in ref]) / 1000.0
        ax.plot(ox, oy, color=PS.PALETTE["neutral_mid"], linewidth=1.0,
                linestyle=(0, (3, 2)), zorder=2)
        ax.scatter(ox[0], oy[0], s=13, marker="s", facecolor="white",
                   edgecolor=PS.PALETTE["neutral_mid"], linewidths=0.8, zorder=5)
        for tag in SHOW:
            draw_track(ax, T[tag][tid], R.ARMS[tag][1])
        ax.scatter(ref[0]["ego_x"] / 1000.0, ref[0]["ego_y"] / 1000.0, s=15, marker="o",
                   facecolor="white", edgecolor=PS.PALETTE["neutral_black"],
                   linewidths=0.8, zorder=6)
        ax.set_title(f"({'ab'[col]}) {ty.capitalize()}, T-{tid}")
        ax.set_xlabel("East (km)"); ax.set_aspect("equal", adjustable="datalim")
        if col == 0:
            ax.set_ylabel("North (km)")
            from matplotlib.lines import Line2D
            ax.legend(handles=[Line2D([], [], color=R.ARMS[t2][1], linewidth=1.2,
                                      label=R.ARMS[t2][0]) for t2 in SHOW],
                      loc="upper left", frameon=False, fontsize=5.0,
                      handlelength=1.3, labelspacing=0.2, borderpad=0.12)

        # 行 2：与他船的距离
        axr = AXR[col]
        for tag in SHOW:
            st = T[tag][tid]
            dd = np.hypot(np.array([q["ego_x"] - q["obs_x"] for q in st]),
                          np.array([q["ego_y"] - q["obs_y"] for q in st])) / 1000.0
            axr.plot(np.arange(len(dd)) * DT / 60.0, dd, color=R.ARMS[tag][1], linewidth=1.0)
            j = int(np.argmin(dd))
            axr.scatter(j * DT / 60.0, dd[j], s=10, color=R.ARMS[tag][1], zorder=5, linewidths=0)
        axr.axhline(cv_cpa(T[SHOW[0]][tid]) / 1000.0, color=PS.PALETTE["neutral_mid"],
                    linewidth=0.8, linestyle=(0, (3, 2)), zorder=1)
        axr.set_title(f"({'cd'[col]}) Range to target vessel")
        axr.set_ylim(bottom=0); axr.tick_params(labelbottom=False)
        if col == 0:
            axr.set_ylabel("Range (km)")

        # 行 3：转艏率
        #: 🔴 2026-08-01 later-15（user：「(e)(f) 淡化对比算法的线，突出我们这条，
        #   也能看出我们平滑一点」）：三条同样粗细时挤成一团，谁都读不出来。
        #   两条离散臂降到 0.7 pt + 30% 不透明度压在底层，Ours 加粗到 1.5 pt 画在最上面。
        #   🔴 **先量了再画**：一步一步的转艏率变化量 |Δω| 的均值
        #     对遇 T-1848  基线 0.01253 · 掩码 0.01175 · Ours 0.00790
        #     交叉 T-76    基线 0.01309 · 掩码 0.01102 · Ours 0.00602
        #   ⟹ 「Ours 更平滑」两个场景都成立，突出它不是靠画法造出来的观感。
        #   （注意：|ω| 的**均值**不是一律更低——对遇里 Ours 0.00791 反而高于基线 0.00704。
        #     所以正文与图注只准说"逐步变化量"，不准说"转得更少"。）
        axw = AXW[col]
        axw.axhspan(-EPS_W, EPS_W, color=PS.PALETTE["neutral_mid"], alpha=0.16, zorder=1)
        for tag in SHOW:
            psi = np.array([q["ego_psi"] for q in T[tag][tid]])
            w = ((np.diff(psi) + np.pi) % (2 * np.pi) - np.pi) / DT
            hero = tag == "ours"
            axw.plot(np.arange(len(w)) * DT / 60.0, w, color=R.ARMS[tag][1],
                     linewidth=(1.5 if hero else 0.7),
                     alpha=(1.0 if hero else 0.30),
                     zorder=(4 if hero else 2))
        axw.set_ylim(-0.023, 0.023)
        axw.set_title(f"({'ef'[col]}) Turn rate")
        axw.set_xlabel("Time (min)")
        if col == 0:
            axw.set_ylabel(r"$\omega$ (rad/s)")
            axw.annotate(r"$|\omega|\leq\varepsilon_\omega$", (0.99, EPS_W),
                         xycoords=("axes fraction", "data"), xytext=(0, 2),
                         textcoords="offset points", fontsize=5.0, ha="right", va="bottom",
                         color=PS.PALETTE["neutral_black"])

    fig.tight_layout(w_pad=1.6, h_pad=0.6)
    PS.save(fig, "Fig6_trajectories", R.OUT_DIRS)
    plt.close(fig)
    print("\n🔴 图注必须写明：场景号 T-%d / T-%d · 种子 s%d · **末段存档**（轨迹专趟不读最佳存档清单）。"
          % (pick["head-on"], pick["crossing"], SEED))


if __name__ == "__main__":
    main()
