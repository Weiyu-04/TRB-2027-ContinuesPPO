# -*- coding: utf-8 -*-
"""正式实验**完整性体检**（`03` L243-续2 · 零依赖）—— **中途就能跑，别等两天后才发现白烧**。

═══ 为什么要它 ═══════════════════════════════════════════════════════════════
user 2026-07-29：「数据采集只有一次机会，就是这次正式去跑」+ 两天硬期限。
本项目**已经三次**栽在同一族坑上：指标接进去了但下游没取（`03` L203）、
分母写死（L240）、口径常量套错趟（L243-续）—— 共同点是**不报错、跑完才发现**。

⟹ 这个脚本把"跑完才发现"提前到"跑了一列就发现"：
   **第一颗种子的 9 条臂一跑完（约 6~8 小时）就跑一次**，任何系统性问题当场暴露，
   还剩一天多可以补救；等到最后再查，就只剩重跑一条路，而时间不够。

═══ 查什么（每一条都对应一个"会静默毁掉这次采集"的失败模式）═════════════════
  ① 该有的 run 在不在      —— 漏了哪条臂 / 哪颗种子
  ② 预算是否一致          —— total_steps / n_seg 有一个不同 = 不是同预算比较 = 不能同表
  ③ 各 run 跑到第几段      —— 报数只能取**所有 run 都达到**的段数（= BUDGET_SEG 上限）
  ④ 分段副本齐不齐        —— "验证集挑最佳存档"整个口径全靠它，而它**至今没在真实训练里跑起来过**
  ⑤ 单变量对子干不干净    —— 逐对比 config_sig，只许差计划内的那一个键
  ⑥ 训练集是不是官方 1300 —— 配错 = 分母不是 600 = 整套口径塌
  ⑦ A 类量采到没有        —— `roll_*` 跑完补不回来；连续臂还要 `roll_n_act > 0`
                             （光看 roll_steps 拦不住：采集包在 try/except 里，
                              动作那截抛异常时 roll_steps 照样非零、动作类量全空）
  ⑧ 存档不重名            —— 重名 = 组间重复计数

═══ 用法 ═════════════════════════════════════════════════════════════════════
    python3 -B 代码/tests/check_formal_integrity.py [结果目录] [期望种子数]
    SEEDS="0 1 2 3" python3 -B 代码/tests/check_formal_integrity.py 结果    # 只查这几颗
退出码：0 = 全过 · 1 = 有硬伤（别继续，先修）· 2 = 只有提醒
"""
import glob
import json
import os
import re
import sys

SEG_STEPS = 507904

#: 9 条正式臂：臂名 → (party, TAG)。与 `代码/run_formal_2027.sh` 的 arm_cfg 一一对应。
ARMS = {
    "ours": ("Continuous-safe", "_F240oursPpoS"), "disc": ("Discrete-safe", "_F240discPpoS"),
    "base": ("Base", "_F240basePpoS"),            "rr":   ("Rule-reward", "_F240rrPpoS"),
    "uns":  ("Continuous-safe", "_F240unsPpoS"),  "ush":  ("Continuous-safe", "_F240ushPpoS"),
    "ab0":  ("Continuous-safe", "_F240ab0PpoS"),  "abB":  ("Continuous-safe", "_F240abBPpoS"),
    "abG":  ("Continuous-safe", "_F240abGPpoS"),
}
CONT = {a for a, (p, _) in ARMS.items() if p == "Continuous-safe"}

#: 单变量对子：(左, 右, 只许差的 config_sig 键集合)。空集 = 必须逐键全同。
PAIRS = [
    ("uns", "ush", set()),                  # 盾不进 config_sig ⟹ 这两条的 config_sig 应当**完全一样**
    ("ab0", "abB", {"act_dist"}),
    ("ab0", "abG", {"gw_entry"}),
    ("abB", "ours", {"gw_entry"}),
    ("abG", "ours", {"act_dist"}),
    ("ush", "ab0", {"well_shaping_weight", "xtrack_weight", "park_weight", "rate_weight",
                    "park_radius", "park_v_target"}),
]
#: 逐 run 天然会不同、不参与比对的键
IGNORE = {"seed", "total_steps", "n_seg"}

N_HARD = N_WARN = 0


def hard(m):
    global N_HARD
    N_HARD += 1
    print(f"  ❌ {m}")


def warn(m):
    global N_WARN
    N_WARN += 1
    print(f"  ⚠️ {m}")


def ok(m):
    print(f"  ✅ {m}")


def load(root):
    """{(臂, 种子): (sidecar dict, 存档 base 路径)}"""
    out = {}
    for p in glob.glob(os.path.join(root, "**", "checkpoints", "*.progress.json"), recursive=True):
        if os.sep + "segments" + os.sep in p:
            continue
        b = os.path.basename(p)[:-len(".progress.json")]
        for arm, (party, tag) in ARMS.items():
            m = re.fullmatch(re.escape(party) + r"_s(\d+)" + re.escape(tag) + r"(\d+)", b)
            if m and m.group(1) == m.group(2):
                try:
                    out[(arm, int(m.group(1)))] = (json.load(open(p, encoding="utf-8")),
                                                   p[:-len(".progress.json")])
                except Exception as e:
                    hard(f"{b} 的 progress.json 读不了：{e}")
    return out


def main():
    root = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "结果")
    seeds_env = os.environ.get("SEEDS")
    print(f"【正式实验完整性体检】{root}")

    runs = load(root)
    if not runs:
        print(f"  ❌ 一个正式臂的存档都没找到（TAG 前缀 _F240*）—— 还没起跑？目录给错了？")
        return 1
    seeds = ([int(x) for x in seeds_env.split()] if seeds_env
             else sorted({s for _, s in runs}))
    print(f"  查 {len(ARMS)} 条臂 × {len(seeds)} 颗种子 {seeds} = {len(ARMS)*len(seeds)} 个 run\n")

    # ① 该有的在不在 ───────────────────────────────────────────────────────────
    print("① 该有的 run 在不在")
    miss = [(a, s) for a in ARMS for s in seeds if (a, s) not in runs]
    if miss:
        # 分清"整颗种子还没轮到"（正常）与"某条臂缺了"（异常）
        untouched = [s for s in seeds if all((a, s) not in runs for a in ARMS)]
        partial = [(a, s) for a, s in miss if s not in untouched]
        if untouched:
            print(f"  · 还没轮到的种子（正常·跑到就有）：{untouched}")
        if partial:
            hard(f"已开工的种子里缺 {len(partial)} 个 run（**这才是问题**）：{partial[:8]}")
        else:
            ok(f"已开工的种子里 9 条臂齐全（现有 {len(runs)} 个 run）")
    else:
        ok(f"{len(runs)} 个 run 全在")

    # ② 预算一致 ──────────────────────────────────────────────────────────────
    print("\n② 预算是否一致（不一致 = 不是同预算比较 = 不能同表）")
    tot = {r["total_steps"] for r, _ in runs.values()}
    nsg = {r["n_seg"] for r, _ in runs.values()}
    (ok if len(tot) == 1 else hard)(f"total_steps = {sorted(tot)}")
    (ok if len(nsg) == 1 else hard)(f"n_seg = {sorted(nsg)}")
    if len(tot) == 1 and len(nsg) == 1:
        t, n = tot.pop(), nsg.pop()
        seg = max(1, t // n)
        ok(f"每段名义 {seg:,} 步 ⟹ SB3 按 rollout 取整后应为 {SEG_STEPS:,}"
           if seg == 500000 else f"⚠️ 每段名义 {seg:,} 步 ≠ 500,000 ⟹ **不在既有网格上**")
        if seg != 500000:
            hard(f"每段步数 {seg:,} 不是 500,000 ⟹ 与既有臂不同网格、学习曲线叠不上")

    # ③ 跑到第几段 ────────────────────────────────────────────────────────────
    print("\n③ 各 run 跑到第几段（报数只能取**所有 run 都达到**的段数）")
    done = {k: r["seg_done"] + 1 for k, (r, _) in runs.items()}       # seg_done 是 0 起的下标
    by_arm = {}
    for (a, s), d in done.items():
        by_arm.setdefault(a, []).append(d)
    for a in ARMS:
        if a in by_arm:
            v = sorted(by_arm[a])
            print(f"  {a:<5} n={len(v):<3} 段数 {v[0]}~{v[-1]}")
    lo = min(done.values())
    ok(f"**BUDGET_SEG 上限 = {lo}**（= {lo*SEG_STEPS:,} 步）—— 报数时全部臂都用这个数")
    if lo * SEG_STEPS < 5_000_000:
        warn(f"目前最小段数才 {lo}（{lo*SEG_STEPS:,} 步），还早")

    # ④ 分段副本 ──────────────────────────────────────────────────────────────
    print("\n④ 分段副本齐不齐（『验证集挑最佳存档』全靠它·这功能从没在真实训练里跑过）")
    bad_seg = []
    for (a, s), (r, base) in sorted(runs.items()):
        d = r["seg_done"] + 1
        segdir = os.path.join(os.path.dirname(base), "segments")
        nm = os.path.basename(base)
        nz = len(glob.glob(os.path.join(segdir, f"{nm}@s*.zip")))
        npkl = len(glob.glob(os.path.join(segdir, f"{nm}@s*_vecnorm.pkl")))
        npj = len(glob.glob(os.path.join(segdir, f"{nm}@s*.progress.json")))
        if not (nz == npkl == npj == d):
            bad_seg.append(f"{a}/s{s}: 已完成 {d} 段，但副本 zip={nz} pkl={npkl} json={npj}")
    if bad_seg:
        hard(f"{len(bad_seg)} 个 run 的分段副本不全 ⟹ 挑不了最佳存档：")
        for x in bad_seg[:6]:
            print(f"       · {x}")
    else:
        ok(f"{len(runs)} 个 run 的分段副本三件套逐段齐全")

    # ⑤ 单变量对子 ────────────────────────────────────────────────────────────
    print("\n⑤ 单变量对子干不干净（逐对比 config_sig，只许差计划内那一个键）")
    for x, y, allowed in PAIRS:
        s0 = next((s for s in seeds if (x, s) in runs and (y, s) in runs), None)
        if s0 is None:
            print(f"  · {x} ↔ {y}：还没有同种子的两条都跑完，跳过")
            continue
        cx = runs[(x, s0)][0].get("config_sig") or {}
        cy = runs[(y, s0)][0].get("config_sig") or {}
        diff = {k for k in set(cx) | set(cy) if k not in IGNORE and cx.get(k) != cy.get(k)}
        extra = diff - allowed
        if extra:
            hard(f"{x} ↔ {y}（s{s0}）出现计划外差异 {sorted(extra)}："
                 + "; ".join(f"{k}: {cx.get(k)!r} vs {cy.get(k)!r}" for k in sorted(extra)))
        else:
            ok(f"{x} ↔ {y}（s{s0}）差异仅 {sorted(diff) or '（完全相同）'}")

    # ⑥ 训练集 ────────────────────────────────────────────────────────────────
    print("\n⑥ 训练集是不是官方 1300（配错 = 分母不是 600 = 整套口径塌）")
    ds = {(r.get("config_sig") or {}).get("dataset") for r, _ in runs.values()}
    (ok if ds == {"manifest_official_1300.json"} else hard)(f"dataset = {sorted(x for x in ds if x)}")

    # ⑦ A 类量 ────────────────────────────────────────────────────────────────
    print("\n⑦ A 类量采到没有（跑完补不回来）")
    no_curves, no_roll, no_act = [], [], []
    for (a, s), (r, _) in sorted(runs.items()):
        cur = r.get("curves") or []
        if not cur:
            no_curves.append(f"{a}/s{s}")
            continue
        if max((int(c.get("roll_steps") or 0) for c in cur), default=0) <= 0:
            no_roll.append(f"{a}/s{s}")
        if a in CONT and max((int(c.get("roll_n_act") or 0) for c in cur), default=0) <= 0:
            no_act.append(f"{a}/s{s}")
    if no_curves:
        hard(f"{len(no_curves)} 个 run 一条 curves 都没有（STEP4E_LOG_CURVES 没开？）：{no_curves[:6]}")
    if no_roll:
        hard(f"{len(no_roll)} 个 run 的 roll_steps 全 0（rollout 采集没生效）：{no_roll[:6]}")
    if no_act:
        hard(f"{len(no_act)} 个**连续臂** roll_n_act 全 0 ⟹ 打满舵率/盾改写量这些"
             f"卖点级 A 类量【采不到】（采集被 try/except 静默吞了）：{no_act[:6]}")
    if not (no_curves or no_roll or no_act):
        ok(f"{len(runs)} 个 run 的 curves + roll_* 都在；连续臂 roll_n_act > 0")

    # ⑧ 不重名 ────────────────────────────────────────────────────────────────
    print("\n⑧ 存档不重名")
    names = [os.path.basename(b) for _, b in runs.values()]
    (ok if len(names) == len(set(names)) else hard)(f"{len(names)} 个存档名 · 去重后 {len(set(names))} 个")

    print("\n" + "=" * 90)
    if N_HARD:
        print(f"❌ **{N_HARD} 处硬伤**（+{N_WARN} 处提醒）—— 先停下来修，别让它继续跑满两天。")
        return 1
    print(f"✅ 全过（{N_WARN} 处提醒）。BUDGET_SEG 上限 = {lo} 段 = {lo*SEG_STEPS:,} 步。")
    return 2 if N_WARN else 0


if __name__ == "__main__":
    sys.exit(main())
