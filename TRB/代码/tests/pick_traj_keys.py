# -*- coding: utf-8 -*-
"""挑**轨迹对比图**要采的场景（`03` L243-续3 · user 2026-07-29 要求"多算法轨迹放一张图"）。

═══ 为什么要挑、以及为什么必须这样挑 ═══════════════════════════════════════════
论文要一张"多个算法在同一场景上的轨迹叠在一起"的图。轨迹是**评估期**产物，
由 `REEVAL_TRAJ_KEYS` 指定采哪几个场景；原来只采 4 个。

4 个太少 —— 你事先不知道哪几个画出来有代表性（对遇/交叉各要有、盾真出手的要有…），
挑不出来就得**重跑一趟评估**。而一条轨迹只有约 11 KB：
**20 个场景 × 108 条臂 ≈ 24 MB，基本免费。**⟹ 多采，事后再从里面选。

🔴 **必须按【几何】分层挑，绝不能按【结果】挑**：
   看了各臂的成绩再决定画哪个场景 = 变相 cherry-pick，审稿人问一句就说不清。
   ⟹ 只用**会遇类型**（场景自身的几何属性，与任何算法无关）分层，
      每类用项目现成的 `stride_pick` **确定性等距**取样（不用随机数 ⟹ 换机器重跑完全一致）。

═══ 跑法 ═════════════════════════════════════════════════════════════════════
    python3 -B 代码/tests/pick_traj_keys.py            # 打印 REEVAL_TRAJ_KEYS 该设成什么
    python3 -B 代码/tests/pick_traj_keys.py --verify "1,100,..."   # 核对某串是否 = 本规则的输出
产物是**确定性**的 ⟹ 直接把输出硬写进 `run_reeval_all.sh`，跑评估时不再依赖场景 XML 在不在。
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_CODE)
sys.path.insert(0, _CODE)

N_PER_TYPE = 10                 # 每类取几个（对遇 / 交叉）
#: 旧口径已经采过的 4 个 —— **强制保留**，这样新老两趟的轨迹图可以直接对照
LEGACY = [1, 100, 1006, 1016]


def main():
    from make_official_manifest import official_split, classify_all, stride_pick

    sdir = os.environ.get("STEP4E_SDIR") or os.path.join(_ROOT, "scenarios")
    if not os.path.isdir(sdir):
        raise SystemExit(f"🔒 找不到场景目录：{sdir}（设 STEP4E_SDIR=）")

    _, test = official_split()                       # 官方测试 600（与训练/验证零交集）
    test_set = set(test)
    cls = classify_all(sdir)                         # 全库 2000 分类（自带与 `03` L111 的硬比对）

    by = {}
    for t in sorted(test_set):
        by.setdefault(cls.get(t, "no-obstacle"), []).append(t)
    print(f"\n官方测试 600 的会遇类型构成：{ {k: len(v) for k, v in sorted(by.items())} }")

    picked = set(LEGACY)
    for typ in ("head-on", "crossing"):
        ids = by.get(typ, [])
        if not ids:
            print(f"  ⚠️ {typ} 在测试集里为空，跳过")
            continue
        sel = stride_pick(ids, N_PER_TYPE)
        picked |= set(sel)
        print(f"  {typ:<10} 池内 {len(ids):>3} 个 → 等距取 {len(sel)}：{sel}")

    bad = sorted(x for x in LEGACY if x not in test_set)
    if bad:
        print(f"  🔴 旧的 4 个键里有不在测试 600 内的：{bad} —— 会让 reeval 直接报错，得改")

    keys = sorted(picked)
    print(f"\n合计 {len(keys)} 个场景（含强制保留的旧 4 个 {LEGACY}）")
    print(f"逐个类型：{ {t: sum(1 for k in keys if cls.get(k) == t) for t in ('head-on', 'crossing')} }")
    s = ",".join(str(k) for k in keys)
    print(f"\nREEVAL_TRAJ_KEYS=\"{s}\"")

    if len(sys.argv) > 2 and sys.argv[1] == "--verify":
        got = ",".join(x.strip() for x in sys.argv[2].replace(",", " ").split() if x.strip())
        if got == s:
            print("✅ 与传入的串逐字一致")
            return 0
        print(f"❌ 与传入的串不一致\n   传入：{got}\n   应为：{s}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
