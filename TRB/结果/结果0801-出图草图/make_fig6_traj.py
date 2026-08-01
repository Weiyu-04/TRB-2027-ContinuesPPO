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
    fig = plt.figure(figsize=(PS.COL2, PS.COL2 * 0.56))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 0.82])
    a = fig.add_subplot(gs[0, 0:2])
    b = fig.add_subplot(gs[0, 2:4])
    c = fig.add_subplot(gs[0, 4:6])
    dax = fig.add_subplot(gs[1, 0:3])
    eax = fig.add_subplot(gs[1, 3:6])

    for ax, ty in ((a, "head-on"), (b, "crossing")):
        tid = pick[ty]
        #: 他船轨迹只画一次（四条臂看到的是同一条他船）
        ref = T[SHOW[0]][tid]
        ox = np.array([s["obs_x"] for s in ref]) / 1000.0
        oy = np.array([s["obs_y"] for s in ref]) / 1000.0
        ax.plot(ox, oy, color=PS.PALETTE["neutral_mid"], linewidth=1.0,
                linestyle=(0, (3, 2)), zorder=2)
        ax.scatter(ox[0], oy[0], s=14, marker="s", facecolor="white",
                   edgecolor=PS.PALETTE["neutral_mid"], linewidths=0.8, zorder=5)
        for tag in SHOW:
            draw_track(ax, T[tag][tid], R.ARMS[tag][1])
        x0 = ref[0]["ego_x"] / 1000.0
        y0 = ref[0]["ego_y"] / 1000.0
        ax.scatter(x0, y0, s=16, marker="o", facecolor="white",
                   edgecolor=PS.PALETTE["neutral_black"], linewidths=0.8, zorder=6)
        ax.set_title(f"({'a' if ty=='head-on' else 'b'}) {ty.capitalize()}, T-{tid}")
        ax.set_xlabel("East (km)")
        ax.set_ylabel("North (km)")
        ax.set_aspect("equal", adjustable="datalim")

    # ── (c) 与他船的距离随时间 ────────────────────────────────────────────
    #: 🔴 第一版这一格画的是「相对初始航向的累计转艏」，**作废**：那个量被"驶向目标"的
    #   常规转向占满，与合规无关，属于图注说一套、图画另一套。
    #   合规的判据是**义务期内**的航向改变，而义务期由离线判分器在评估侧判定、
    #   轨迹产物里只有带盾臂记了 ρ（无盾臂 source=None、ρ 恒 0）⟹ 三条臂没有可比的公共窗口。
    #   ⟹ 换成**与他船的距离**：三条臂共用同一条他船、同一把尺，可直接叠比，
    #      且正是这个会遇真正关心的量。虚线是常速直行下的最近距离（本图的挑选判据本身）。
    tid = pick["crossing"]
    for tag in SHOW:
        st = T[tag][tid]
        d = np.hypot(np.array([q["ego_x"] - q["obs_x"] for q in st]),
                     np.array([q["ego_y"] - q["obs_y"] for q in st])) / 1000.0
        t = np.arange(len(d)) * DT / 60.0
        c.plot(t, d, color=R.ARMS[tag][1], linewidth=1.2, label=R.ARMS[tag][0])
        j = int(np.argmin(d))
        c.scatter(t[j], d[j], s=12, color=R.ARMS[tag][1], zorder=5, linewidths=0)
    c.axhline(cv_cpa(T[SHOW[0]][tid]) / 1000.0, color=PS.PALETTE["neutral_mid"],
              linewidth=0.8, linestyle=(0, (3, 2)), zorder=1)
    c.set_title(f"(c) Range to target vessel, T-{tid}")
    c.set_xlabel("Time (min)")
    c.set_ylabel("Range (km)")
    c.set_ylim(bottom=0)

    # ── (d)(e) 执行命令随时间：转艏率与航速 ────────────────────────────────
    #: 两个量都由轨迹产物**精确反推**（命令零阶保持 ⟹ ω=Δψ/Δt、v 直接存了）。
    #  🔴 (d) 里那条淡带是**盾**的直航容差 |ω| ≤ ε_ω；离线判分器判违规用的是
    #     **义务期内累计航向改变**，是另一个判据（§5.2）——图注必须写明，别让人读混。
    EPS_W = 0.004363323129985824
    dax.axhspan(-EPS_W, EPS_W, color=PS.PALETTE["neutral_mid"], alpha=0.16, zorder=1)
    for tag in SHOW:
        st = T[tag][tid]
        psi = np.array([q["ego_psi"] for q in st])
        dpsi = (np.diff(psi) + np.pi) % (2 * np.pi) - np.pi
        w = dpsi / DT
        t = np.arange(len(w)) * DT / 60.0
        dax.plot(t, w, color=R.ARMS[tag][1], linewidth=1.0, zorder=3)
        v = np.array([q["ego_v"] for q in st])
        eax.plot(np.arange(len(v)) * DT / 60.0, v, color=R.ARMS[tag][1], linewidth=1.0,
                 zorder=3)
    dax.set_title(f"(d) Turn rate, T-{tid}")
    dax.set_xlabel("Time (min)"); dax.set_ylabel(r"$\omega$ (rad/s)")
    dax.annotate(r"$|\omega|\leq\varepsilon_\omega$ (shield, stand-on)", (0.985, EPS_W),
                 xycoords=("axes fraction", "data"), xytext=(0, 3),
                 textcoords="offset points", fontsize=5.0, ha="right", va="bottom",
                 color=PS.PALETTE["neutral_black"])
    eax.set_title(f"(e) Speed, T-{tid}")
    eax.set_xlabel("Time (min)"); eax.set_ylabel("Speed (m/s)")
    #: 图例整图只留一份（在 (c) 里），(a)(b)(d)(e) 共用同一套配色
    c.legend(loc="best", fontsize=5.6, borderpad=0.3)

    fig.tight_layout(w_pad=1.6)
    PS.save(fig, "Fig6_trajectories", R.OUT_DIRS)
    plt.close(fig)
    print("\n🔴 图注必须写明：场景号 T-%d / T-%d · 种子 s%d · **末段存档**（轨迹专趟不读最佳存档清单）。"
          % (pick["head-on"], pick["crossing"], SEED))


if __name__ == "__main__":
    main()
