# -*- coding: utf-8 -*-
"""T1 判据：**这个步数预算够不够**（`03` L243-改② · user 2026-07-29 拍板 · 零依赖）。

═══ 为什么要它 ═══════════════════════════════════════════════════════════════
`03` L238/L241 只写了「两个信号一起看，任一还在动就算没收敛」，**"每局时长"那一路
根本没有数值门槛** ⟹ 复审实测：被当作**已收敛**的小集 C s3 是 −57.3 秒、被判**还在爬**
的大集 D s1 是 −12.3 秒 —— **同一句话能读出相反结论**。那不是判据，是事后解释。
本脚本把判据变成一条命令：喂进去 → 直接吐「够 / 不够 + 哪几颗还在爬」，不靠人眼。

═══ 判据（**先定死·起跑前就写在这里·别到时候看着数字找理由**）═══════════════
对每一颗种子，取训练期验证曲线（`trend`，跑在官方 1400 里切出的 100 个验证场景上），
比 **最后 3 段** 与 **再往前 3 段**（B=预算段数 ⟹ [B-2..B] vs [B-5..B-3]；
n=10 时正好还原 `03` L236/L241 用的「末 3 段 vs 第 5-7 段」）：

  ① 到达率（**滞后**指标）：涨幅 > **+6 个百分点** ⟹ 还在明显上升
     🔴 **不是 9 点**。9 点是按 **40** 个验证场景推的（120 局·二项 SE≈4.5pt·2SE≈9）；
        本方案用 **100** 个验证场景（3 段 = 300 局·SE≈2.89pt·2SE≈5.8）⟹ 门槛是 **6**。
        照抄 9 点会把「还在爬」误判成「走平」—— 正是 `03` L238-B 刚吃过的那个亏的镜像。
     ⚠️ 口径诚实：这个 2SE 是**单个均值**的；两个均值之差的 SE 是 √2 倍 ⟹ 6 点约等于
        1.4 个"差的 SE"，偏松 = 偏向判"还在爬" = 偏向多买步数 = 安全方向。**别把它说成 95% 置信。**
  ② 每局时长（**先行**指标）：**相对**降幅 > **5%** ⟹ 还在动。
     🔴 必须用相对量。绝对门槛在「刚爬出 1700 秒打转」和「已在 600 秒巡航」两种状态下
        差两个量级（实测 −602.8 秒 vs −12.3 秒），一个绝对数不可能同时判对两边。
     🔴 **只对末段到达率 < 50% 的种子生效**。这个信号当初是为一件具体的事发明的（`03` L238-B）：
        **到达率看着走平、其实策略刚从打转吸引子里爬出来**（D 臂 s3：到达 +3.0 看着平，
        每局时长却掉了 40%）。而一颗到达率已经 95% 的种子，每局时长再慢慢降几个百分点
        只是「路走得更利索」，不是「还没练完」—— 拿它判没收敛会**永远判不完**。
        本窗口反向验证坐实：不加这条限定，小集 C 会被判成 6/10 还在爬，与 `03` L236 的
        3/10 对不上；加上之后**逐颗还原 L236 与 L241 的原判**（见文末「反向验证」）。

  ③ 另外单独拎出**崩掉**的种子（末段每局时长 ≥ 1360 秒 = 0.8 × 回合上限 1700 秒）：
     它们会"走平"在天花板上 ⟹ 机械地算进"已收敛"是错的。**崩既不支持也不反对加步数**，
     单独报，别混进分母（同 `Paper/正式实验/_common.py` 的三分类）。

  **两个信号任一还在动 ⟹ 这颗种子没收敛。**

═══ 反向验证（本脚本必须先能还原项目自己的旧判决，才配拿来判新数据）═══════════
  · 大集 D 臂（100 个验证场景 ⟹ 门槛 6）→ **3/3 还在爬**，与 `03` L241-B 逐颗一致
    （s1 +22.0 / s4 +20.0 靠到达率信号；s3 到达 +3.0 看着平，靠 −40.5% 的时长信号抓出来）
  · 小集 C 臂（40 个验证场景 ⟹ `N_VAL=40` ⟹ 门槛 9）→ **3/10 还在爬（s6/s8/s9）**，与 `03` L236-A 一致
    复跑：`N_VAL=40 python3 -B 代码/tests/judge_t1_budget.py 结果 C231bothPpo 10`
  🔴 **改判据前先跑这两条**：动了门槛/窗口而这两条不再还原，就是把判据调歪了。

═══ 决策 ═════════════════════════════════════════════════════════════════════
  「还在爬」的种子 ≤ **2** 颗 ⟹ **这个预算够** ⟹ 按此 BUDGET_SEG 报数，其余臂可用同段数起跑
  否则 ⟹ 预算不够 ⟹ 让主线继续跑到更大的段数，其余臂也用更大的段数

═══ 用法 ═════════════════════════════════════════════════════════════════════
    python3 -B 代码/tests/judge_t1_budget.py <结果目录> [臂特征串] [预算段数]
    python3 -B 代码/tests/judge_t1_budget.py 结果 F240oursPpo 20     # 第 20 段处判
    python3 -B 代码/tests/judge_t1_budget.py 结果 F240oursPpo 30     # 第 30 段处再判一次
退出码：0 = 预算够 · 2 = 不够 · 1 = 数据有问题（段数不足/找不到 run）
"""
import glob
import json
import os
import sys

import math

#: 训练期验证场景数。正式实验 = 100（官方 1400 切出来的那 100 个）；
#  反向验证旧的小集臂时要设 `N_VAL=40`（它们的曲线跑在 40 个场景上，门槛因此是 9 不是 6）。
N_VAL = int(os.environ.get("N_VAL", "100"))
#: 到达率涨幅门 = 2 × 三段合并的二项标准误，向上取整到整数百分点。
#  N_VAL=100 ⟹ 2×sqrt(0.25/300)=5.77 ⟹ **6**；N_VAL=40 ⟹ 2×sqrt(0.25/120)=9.13 ⟹ **9**（= `03` L236 用的那个）
ARR_GATE = math.ceil(2.0 * math.sqrt(0.25 / (3 * N_VAL)) * 100.0)
EP_REL_GATE = 5.0           # 每局时长相对降幅门（%）
EP_SIGNAL_ARR_MAX = 50.0    # 每局时长信号**只在末段到达率低于此**时生效（见模块 docstring ②）
EP_CAP_S = 170 * 10.0       # 回合时长天花板 = k_max × dt（`trb_env/usv_env.py:143`）
SPIN_LINE_S = 0.8 * EP_CAP_S
MAX_CLIMBING = 2            # 允许几颗还在爬


def load_trends(root, needle):
    """{run 名: trend}。只吃主 sidecar，跳过 `segments/` 里的分段副本（它们的 trend 是残缺的）。"""
    out = {}
    for p in glob.glob(os.path.join(root, "**", "checkpoints", "*.progress.json"), recursive=True):
        if os.sep + "segments" + os.sep in p or needle not in os.path.basename(p):
            continue
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        if d.get("trend"):
            out[os.path.basename(p)[:-len(".progress.json")]] = d["trend"]
    return out


def judge_one(trend, budget):
    """→ (标签, Δ到达pt, Δ时长%, 末段到达, 末段每局秒)。budget = 在第几段处判。"""
    tr = trend[:budget]
    # 🔴 L243-续8（C 线 R1）：原来只要 `len(tr) >= 6` 就判，**与 budget 完全脱钩** ⟹
    #   一条只跑到第 7 段就崩掉的 run，在"第 20 段处判"时会被当成**已经跑满 20 段**来判，
    #   而它最后 6 段其实是第 2~7 段。判据本身没错，错在拿错了段。⟹ 必须真的跑到 budget 段。
    if len(trend) < budget:
        return "段数不足", None, None, None, None
    if len(tr) < 6:
        return "段数不足", None, None, None, None
    arr = [x["到达率%"] for x in tr]
    ep = [x["Ep长s"] for x in tr]
    avg = lambda v, a, b: sum(v[a:b]) / (b - a)          # 0 起的半开区间
    n = len(tr)
    d_arr = avg(arr, n - 3, n) - avg(arr, n - 6, n - 3)   # 最后 3 段 − 再往前 3 段
    late, early = avg(ep, n - 3, n), avg(ep, n - 6, n - 3)
    d_ep = (late / early - 1.0) * 100.0 if early else 0.0
    if ep[-1] >= SPIN_LINE_S:
        return "崩(打转)", d_arr, d_ep, arr[-1], ep[-1]
    # 时长信号只在到达率还低的时候生效（专治「到达看着平、其实刚爬出打转」·见 docstring ②）
    ep_climbing = (arr[-1] < EP_SIGNAL_ARR_MAX) and (d_ep < -EP_REL_GATE)
    climbing = (d_arr > ARR_GATE) or ep_climbing
    return ("还在爬" if climbing else "已收敛"), d_arr, d_ep, arr[-1], ep[-1]


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "结果"
    needle = sys.argv[2] if len(sys.argv) > 2 else "F240oursPpo"
    budget = int(sys.argv[3]) if len(sys.argv) > 3 else 20

    print(f"【T1 步数判据】目录={root} · 臂={needle} · 在第 {budget} 段处判")
    _se = 2.0 * (0.25 / (3 * N_VAL)) ** 0.5 * 100.0
    print(f"  门槛①：到达率涨幅 > {ARR_GATE:.0f} 点（{N_VAL} 个验证场景 · 3 段 = {3*N_VAL} 局 ⟹ 2SE={_se:.2f}）")
    print(f"  门槛②：每局时长相对降幅 > {EP_REL_GATE:.0f}%，**且末段到达率 < {EP_SIGNAL_ARR_MAX:.0f}%** 才算数")
    print(f"         （专治「到达看着走平、其实刚爬出打转」；到达已高时时长仍慢降 = 路走得更利索，不是没练完）")
    print(f"  崩线：末段每局时长 ≥ {SPIN_LINE_S:.0f} 秒（= 0.8 × 回合上限 {EP_CAP_S:.0f} 秒）\n")

    T = load_trends(root, needle)
    if not T:
        print(f"❌ 在 {root} 下找不到含【{needle}】的 run（还没跑？还是特征串写错了？）")
        return 1

    tally, short = {}, 0
    print(f"  {'run':<44}{'段数':>5}{'末段到达%':>10}{'末段每局秒':>12}{'Δ到达pt':>10}{'Δ时长%':>9}   判")
    for name in sorted(T):
        lab, da, de, a, e = judge_one(T[name], budget)
        if lab == "段数不足":
            short += 1
            print(f"  {name[:43]:<44}{len(T[name]):>5}{'':>10}{'':>12}{'':>10}{'':>9}   ⚠️ 段数不足（需 ≥6 段）")
            continue
        tally[lab] = tally.get(lab, 0) + 1
        mark = {"已收敛": "✅", "还在爬": "⬆️", "崩(打转)": "💥"}[lab]
        print(f"  {name[:43]:<44}{min(len(T[name]), budget):>5}{a:>10.1f}{e:>12.0f}"
              f"{da:>+10.1f}{de:>+9.1f}   {mark} {lab}")

    n_ok = tally.get("已收敛", 0)
    n_up = tally.get("还在爬", 0)
    n_spin = tally.get("崩(打转)", 0)
    print(f"\n  汇总：已收敛 {n_ok} · 还在爬 {n_up} · 崩 {n_spin}"
          + (f" · 段数不足 {short}" if short else ""))
    if short:
        print(f"  ❌ 有 {short} 个 run **没跑满 {budget} 段** ⟹ 要么还在跑（等跑到了再判），"
              f"要么中途崩了（那几条不能进同预算比较）。先用 check_formal_integrity.py 分清是哪一种。")
        return 1
    print(f"  ⚠️ 崩掉的 {n_spin} 颗**不进分母**：它们会走平在天花板上，机械算成『已收敛』是错的；"
          "加步数也救不了打转（`03` L235/L238）。" if n_spin else "")

    # 🔴 L243-续8（C 线 R2）：原来只看"还在爬"几颗，**崩了几颗完全不影响结论**。
    #   极端情形：10 颗里 8 颗崩、2 颗收敛、0 颗在爬 ⟹ 照样打印"✅ 这个预算够"。
    #   但 8/10 崩的臂根本不是"预算够不够"的问题，是**这条配方本身不成立**，
    #   拿它的段数去定其余 8 条臂的预算是把错误放大 9 倍。⟹ 崩过半直接判否。
    n_all = n_ok + n_up + n_spin
    if n_all and n_spin * 2 >= n_all:
        print(f"\n  ❌ **崩掉 {n_spin}/{n_all} 颗（过半）—— 这不是预算问题，是这条配方本身有问题。**")
        print(f"     加步数救不了打转（`03` L235/L238）。先看是不是配方/种子的事，别拿它定其余臂的预算。")
        return 1
    if n_ok == 0 and n_all:
        print(f"\n  ❌ **一颗都没收敛**（{n_all} 颗里 {n_up} 颗还在爬、{n_spin} 颗崩）⟹ 这个预算显然不够。")
        return 1
    if n_up <= MAX_CLIMBING:
        print(f"\n  ✅ **这个预算够**（还在爬 {n_up} ≤ {MAX_CLIMBING}；已收敛 {n_ok} · 崩 {n_spin}）")
        print(f"     ⟹ 报数用 BUDGET_SEG={budget}（= {budget*507904:,} 步）；其余 8 条臂可用 NSEG={budget} 起跑。")
        print(f"     ⟹ 主线这 {n_ok+n_up+n_spin} 个 run 可在第 {budget} 段停（已完成的段都在 segments/ 里，一步没浪费）。")
        return 0
    print(f"\n  ❌ **这个预算不够**（还在爬 {n_up} > {MAX_CLIMBING}）")
    print(f"     ⟹ 主线继续跑满 30 段，其余 8 条臂也用 NSEG=30。到 30 段再跑一次本脚本。")
    print("     🚫 判读口径（`03` L241-B 定死）：只能写「该预算下尚未收敛」，"
          "**绝不能写「官方大集效果更差 / 大数据集不好」**——那是把预算问题说成数据问题。")
    return 2


if __name__ == "__main__":
    sys.exit(main())
