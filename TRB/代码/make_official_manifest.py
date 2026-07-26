#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成【官方 1400/600 划分】的自描述场景清单（`03` L223）——给"小集训起点 + 官方全集续训"那一趟用。

═══ 为什么非要造这个清单（`03` L207-B 说过"别再造 manifest"，这里是有正当理由的例外）═══
不设 `STEP4E_MANIFEST` 时，`run_step4e` 默认模式**本来就是**官方 1400/600 划分（L207-B 亲读码坐实）。
但那样跑出来的 checkpoint，其 `config_sig.dataset` 会记成 **`"strided"`**，而
`代码/tests/reeval_official.py` 有一道 **fail-closed 闸**专门拦它：
`"strided"` 分不清是 strided-200（训 140·可能与官方 600 有交集）还是 strided-2000（训 = 官方 1400）
⟹ 泄漏剔不干净 ⟹ **跑完了评不了**（除非硬开 `REEVAL_ALLOW_UNKNOWN_DATASET`，那等于放弃泄漏剔除）。

⟹ 用一个**自描述的清单**把训练集写明白，评估端就能精确还原、正常剔泄漏。**一行训练代码都不用改。**

═══ 顺带解决第二个问题：训练期的评估不许看报数的那 600 个 ═══
`run_step4e` 用清单里的 `test` 当**训练期里程碑评估集**。若直接用官方 600 当它，就等于训练全程盯着报数集看。
本脚本从**官方训练 1400** 里切出 `--val`（默认 100）个当验证集 ⟹
  **训练 1300 / 验证 100 / 官方测试 600 全程不碰。**

═══ 分母不会乱（重要）═══
本清单的训练键与验证键**全部来自官方 1400**，与官方测试 600 **交集为 0**（划分定义决定）。
⟹ 用 `reeval_official.py` 评这条臂时，它自己不贡献任何泄漏；
   而**与既有四臂放在同一趟评**时，泄漏取所有被评 checkpoint 的**并集** ⟹ 仍然是 **strict 563**、同分母可比。
   🔴 **别单独评这条臂再把数字塞进主表**（单独评会得到 600，与其它臂 563 不同分母）。

用法：
    python 代码/make_official_manifest.py                       # 本机跑·写 balanced_pool/manifest_official_1300.json
    python 代码/make_official_manifest.py --val 100 --out <路径>
    python 代码/make_official_manifest.py --check <路径>         # 只校验已有清单，不重写
"""
from __future__ import annotations
import argparse
import json
import os
import random
import sys
import types
import xml.etree.ElementTree as ET

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

N_TOTAL, TEST_FRAC, SPLIT_SEED, POOL = 2000, 0.30, 0, 2000
EXPECT_TRAIN, EXPECT_TEST = 1400, 600
EXPECT_CROSSING, EXPECT_HEADON = 1291, 709          # `03` L111 三法交叉坐实的全库构成（本脚本会重算并硬比对）


def official_split():
    """官方 1400/600 —— 与 `run_step4e.make_split` 同逻辑（`_pool_eff` = None，因 POOL 不 > n_total）。"""
    ids = list(range(N_TOTAL))                       # pool_size 不 > n_total ⟹ 不 striding
    random.Random(SPLIT_SEED).shuffle(ids)
    n_test = int(round(N_TOTAL * TEST_FRAC))
    return sorted(ids[n_test:]), sorted(ids[:n_test])


def _stub_commonocean():
    """`classify_scenarios` 模块级 import 了 commonocean，但我们只用它里面的**纯几何** `classify`。

    桩掉那个 import 即可在本机跑。**不是绕过校验**：`classify` 本身不碰 commonocean 对象，
    它只吃 numpy 数组；而且下面会用全库分类结果与 `03` L111 的记录**硬比对**，对不上就中止。
    """
    if "commonocean" in sys.modules:
        return
    for name, attrs in (("commonocean", {}), ("commonocean.common", {}),
                        ("commonocean.common.file_reader", {"CommonOceanFileReader": object})):
        m = types.ModuleType(name)
        m.__path__ = []
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[name] = m


def _geom(sdir, tid):
    """从场景 XML 直读分类所需的几何量（不经 commonocean·本机可跑）。无他船 → None。"""
    import numpy as np
    r = ET.parse(os.path.join(sdir, f"T-{tid}.xml")).getroot()
    pp = r.find("planningProblem")

    def _scalar(node):
        ex = node.find("exact")
        return float(((ex if ex is not None else node).text or "0").strip())

    p = pp.find("initialState/position/point")
    ego_p = np.array([float(p.find("x").text), float(p.find("y").text)])
    ego_psi = _scalar(pp.find("initialState/orientation"))
    ego_v = _scalar(pp.find("initialState/velocity"))
    g = pp.find("goalState/position/rectangle/center")
    gc = np.array([float(g.find("x").text), float(g.find("y").text)])
    ob = r.find("dynamicObstacle")
    if ob is None:
        return None
    op = ob.find("initialState/position/point")
    obs_p = np.array([float(op.find("x").text), float(op.find("y").text)])
    return (ego_p, ego_psi, ego_v, gc, obs_p,
            _scalar(ob.find("initialState/orientation")), _scalar(ob.find("initialState/velocity")))


def classify_all(sdir):
    """全库 2000 分类 → {tid: 'head-on'|'crossing'|…}，并与 `03` L111 的记录**硬比对**（对不上中止）。"""
    _stub_commonocean()
    from classify_scenarios import classify
    from collections import Counter
    out = {}
    for tid in range(N_TOTAL):
        g = _geom(sdir, tid)
        out[tid] = "no-obstacle" if g is None else classify(*g)[0]
    cnt = Counter(out.values())
    if cnt.get("crossing") != EXPECT_CROSSING or cnt.get("head-on") != EXPECT_HEADON:
        raise SystemExit(f"🔒 全库分类与 `03` L111 记录不符：实得 {dict(cnt)}，"
                         f"应为 crossing={EXPECT_CROSSING} / head-on={EXPECT_HEADON} → 分类判据或场景库变了，中止。")
    print(f"  ✅ 全库分类与 `03` L111 记录逐个对上：{dict(cnt)}")
    return out


def stride_pick(ids, n):
    """从有序 id 列表里**确定性**等距取 n 个（不用随机数 ⟹ 换机器/重跑完全一致）。"""
    ids = list(ids)
    if n <= 0 or n >= len(ids):
        return list(ids) if n >= len(ids) else []
    step = len(ids) / float(n)
    picked, seen = [], set()
    for i in range(n):
        j = min(len(ids) - 1, int(i * step))
        while j in seen:                              # 极端情况下防重复取同一个
            j += 1
        seen.add(j)
        picked.append(ids[j])
    return sorted(picked)


def build(sdir, n_val):
    train, test = official_split()
    if (len(train), len(test)) != (EXPECT_TRAIN, EXPECT_TEST):
        raise SystemExit(f"🔒 官方划分应为 {EXPECT_TRAIN}/{EXPECT_TEST}，实得 {len(train)}/{len(test)}，中止。")
    types_of = classify_all(sdir)

    by = {"head-on": [], "crossing": []}
    other = []
    for t in train:
        by.get(types_of[t], other).append(t)
    if other:
        raise SystemExit(f"🔒 官方训练集里出现非对遇/交叉的场景 {len(other)} 个（{other[:5]}…）→ "
                         "与 `03` L111「官方 2000 只有对遇+交叉」矛盾，中止。")

    man = {"head_on": {"train": [], "test": []},
           "crossing": {"train": [], "test": []},
           "overtaking": {"train": [], "test": []}}       # 官方库无追越（L111 坐实）→ 空
    # 验证集按类型**等比例**抽（不是整体抽后再分，免得某一类被抽空）
    for zh, key in (("head-on", "head_on"), ("crossing", "crossing")):
        ids = sorted(by[zh])
        k = round(n_val * len(ids) / float(len(train)))
        val = stride_pick(ids, k)
        vs = set(val)
        man[key]["test"] = val
        man[key]["train"] = [i for i in ids if i not in vs]
    man["_note"] = ("官方 1400/600 划分的训练侧再切验证集：train+test 全部来自官方训练 1400，"
                    "与官方测试 600 交集为 0。由 代码/make_official_manifest.py 生成（`03` L223）。")
    return man, set(train), set(test)


def check(man, train_set, test_set, n_val):
    tr = set(man["head_on"]["train"]) | set(man["crossing"]["train"])
    va = set(man["head_on"]["test"]) | set(man["crossing"]["test"])
    ok = True

    def chk(name, cond, extra=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'✅' if cond else '🔴'}] {name} {extra}")

    chk("训练 + 验证 = 官方训练 1400（无遗漏无重复）", tr | va == train_set and not (tr & va),
        f"训练 {len(tr)} + 验证 {len(va)} = {len(tr | va)}")
    chk(f"验证集大小 ≈ 请求的 {n_val}", abs(len(va) - n_val) <= 1, f"实得 {len(va)}")
    chk("训练集 ∩ 官方测试 600 = 0（本清单自己不贡献任何泄漏）", not (tr & test_set), f"交集 {len(tr & test_set)}")
    chk("验证集 ∩ 官方测试 600 = 0（训练期评估不碰报数集）", not (va & test_set), f"交集 {len(va & test_set)}")
    chk("追越为空（官方库无追越·`03` L111）", not man["overtaking"]["train"] and not man["overtaking"]["test"])
    chk("两类都非空（验证集没被某一类抽空）",
        man["head_on"]["test"] and man["crossing"]["test"],
        f"对遇验证 {len(man['head_on']['test'])} · 交叉验证 {len(man['crossing']['test'])}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sdir", default=os.path.join(_HERE, "..", "scenarios"), help="T-*.xml 所在目录")
    ap.add_argument("--val", type=int, default=100, help="从官方训练 1400 里切多少个当验证集")
    ap.add_argument("--out", default=os.path.join(_HERE, "..", "balanced_pool", "manifest_official_1300.json"))
    ap.add_argument("--check", default="", help="只校验已有清单，不重写")
    a = ap.parse_args()

    if a.check:
        with open(a.check, encoding="utf-8") as fh:
            man = json.load(fh)
        train_set, test_set = (set(x) for x in official_split())
        n_val = len(set(man["head_on"]["test"]) | set(man["crossing"]["test"]))
        print(f"【校验】{a.check}")
        return 0 if check(man, train_set, test_set, n_val) else 1

    sdir = os.path.abspath(a.sdir)
    print(f"【生成】官方划分清单 · 场景目录 {sdir} · 验证集 {a.val} 个")
    man, train_set, test_set = build(sdir, a.val)
    print("【校验】")
    if not check(man, train_set, test_set, a.val):
        raise SystemExit("🔒 校验没全过 → 不写文件（宁可不生成，也不生成一个错的清单）。")
    out = os.path.abspath(a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(man, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, out)
    print(f"  → {out}")
    print(f"  训练 {len(man['head_on']['train']) + len(man['crossing']['train'])}"
          f"（对遇 {len(man['head_on']['train'])} / 交叉 {len(man['crossing']['train'])}）"
          f" · 验证 {len(man['head_on']['test']) + len(man['crossing']['test'])}"
          f"（对遇 {len(man['head_on']['test'])} / 交叉 {len(man['crossing']['test'])}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
