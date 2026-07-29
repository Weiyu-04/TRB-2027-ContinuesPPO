# -*- coding: utf-8 -*-
"""正式实验表/图/统计的公共读数层（零依赖·只用标准库）。

**唯一入口**：所有表、图、统计都必须经这里读原始产物，**不许任何脚本自己手抄数字**
（`Paper/正式实验/README.md` §1 铁律）。

**fail-closed 设计**：`load_pass()` 在下列任一情况**直接抛错、不返回半成品**——
  · 8 组的 strict 键不是逐位相同 / 长度 ≠ 期望值 ⟹ 分母不一致，拼出来的表是错的
  · 有 checkpoint 重名 ⟹ 组间重复计数
  · 有锚点自检没过 ⟹ 评估可能配错（`03` L192 那个洞）
宁可不出表，也不出一张看不出错的错表。
"""
import glob
import json
import os
import re
import statistics as st

CRASH_ARR = 50.0     # 崩塌线：到达 < 此 = 崩。取自 `代码/bgate_judge.py:16`，全项目统一，**禁止逐臂另定**（`03` L234-B）
SEC = "strict"       # 报数一律 strict（官方测试 600 剔泄漏）；训练期里程碑数永不进表（`03` L232 铁律）

#: 臂定义 = (显示名, checkpoint 名里的特征串, 是否进头条表)
ARM_SPECS = [
    ("Base（离散·无盾）",            lambda ck: ck.startswith("Base_"),            True),
    ("Rule-reward（离散·软奖励）",    lambda ck: ck.startswith("Rule-reward_"),     True),
    ("Discrete-safe（对标论文）",     lambda ck: ck.startswith("Discrete-safe_") and "dsSeg" not in ck, True),
    ("Discrete-safe（重训·带分段）",  lambda ck: "dsSeg" in ck,                     False),
    ("金标（连续·从零·旧配方）",       lambda ck: "_L1rateON_ppo_" in ck,            True),
    ("主线（连续·热启动·旧配方）",     lambda ck: "wsHOCRppo" in ck,                 True),
    ("大集探针（旧配方·已判输）",      lambda ck: "wsBIGppo" in ck,                  False),
    ("大集探针（新配方 D）",          lambda ck: "D232bigCppo" in ck,               True),
    ("A（从零·只 Beta）",            lambda ck: "A231betaPpo" in ck,               False),
    ("B（从零·只改状态机）",          lambda ck: "B231gwsymPpo" in ck,              False),
    ("C（从零·两个都上·主线候选）",    lambda ck: "C231bothPpo" in ck,               True),
]


def arm_of(ck):
    for name, pred, _ in ARM_SPECS:
        if pred(ck):
            return name
    return None


def load_pass(d, prefix="g", expect_strict=563, strict_n=True):
    """读一趟同趟重评 → {checkpoint: 结果}。**只吃 `<前缀><数字>.json`**（`g*_traj.json` 混进来会污染统计）。"""
    files = sorted(f for f in glob.glob(os.path.join(d, prefix + "*.json"))
                   if re.fullmatch(prefix + r"\d+\.json", os.path.basename(f)))
    if not files:
        raise SystemExit(f"🔒 {d} 里没找到 {prefix}<数字>.json")
    groups = {os.path.basename(f): json.load(open(f, encoding="utf-8")) for f in files}
    ref = groups[os.path.basename(files[0])][f"{SEC}键"]
    for n, g in groups.items():
        if g[f"{SEC}键"] != ref:
            raise SystemExit(f"🔒 {n} 的 {SEC} 键与第一组**不逐位相同** ⟹ 分母不一致，这张表不能出。")
    if strict_n and len(ref) != expect_strict:
        raise SystemExit(f"🔒 {SEC} 场景数 = {len(ref)}，期望 {expect_strict} ⟹ 口径不对（单独评过？跨趟拼过？），这张表不能出。")
    rows = {}
    for g in groups.values():
        for ck, v in g["结果"].items():
            if ck == "_锚点汇总":
                continue
            if ck in rows:
                raise SystemExit(f"🔒 checkpoint 重名：{ck} ⟹ 组间重复计数，这张表不能出。")
            rows[ck] = v
    bad = [ck for ck, v in rows.items() if v.get("anchor", {}).get("通过") is not True]
    if bad:
        raise SystemExit(f"🔒 锚点自检未通过 {len(bad)} 条：{bad[:5]} ⟹ 评估可能配错，这张表不能出。")
    return rows, len(ref), len(files)


def by_arm(rows):
    """{臂显示名: [(ck, 结果), ...]}（按种子排序）。未归到臂的 checkpoint 直接抛错，不静默丢。"""
    out, orphan = {}, []
    for ck, v in rows.items():
        a = arm_of(ck)
        if a is None:
            orphan.append(ck)
        else:
            out.setdefault(a, []).append((ck, v))
    if orphan:
        raise SystemExit(f"🔒 有 {len(orphan)} 条 checkpoint 归不到任何臂：{orphan[:5]}"
                         " ⟹ 请先在 `_common.ARM_SPECS` 里登记，别让它静默漏掉。")
    for a in out:
        out[a].sort(key=lambda t: t[1]["seed"])
    return out


def metrics(entries, healthy_only=False):
    """一条臂的聚合。`healthy_only=True` ⟹ 只算练成的种子（`03` L234-E④：逐局平均的指标必须两版都报）。"""
    lst = [(ck, v) for ck, v in entries if not healthy_only or v[SEC]["到达率%"] >= CRASH_ARR]
    if not lst:
        return None
    cq = lambda k: st.mean(v[SEC]["控制质量"][k] for _, v in lst)
    m = lambda k: st.mean(v[SEC][k] for _, v in lst)
    arr = [v[SEC]["到达率%"] for _, v in lst]
    n_ep = sum(v[SEC]["n"] for _, v in lst)
    n_col = sum(round(v[SEC]["碰撞率%"] * v[SEC]["n"] / 100.0) for _, v in lst)
    return dict(
        n=len(lst), 健康=sum(1 for x in arr if x >= CRASH_ARR), 种子=[v["seed"] for _, v in lst],
        到达=m("到达率%"), 到达SD=(st.stdev(arr) if len(arr) > 1 else 0.0), 逐种子到达=arr,
        碰撞局=n_col, 总局=n_ep, 碰撞率=100.0 * n_col / n_ep,
        违规=m("违规次数/局"), 让路=cq("giveway_violations"), 直航=cq("standon_violations"),
        转艏=cq("yaw_incr_mean"), 油门=cq("accel_incr_mean"), 紧急=m("紧急步%"),
    )


def per_seed(entries, key):
    """{seed: 值}，key ∈ 顶层指标名或 控制质量 里的名字。给同种子配对用。"""
    out = {}
    for _, v in entries:
        s = v[SEC]
        out[v["seed"]] = s[key] if key in s else s["控制质量"][key]
    return out


BUDGET_NOTE = ("口径：**5.08M 步训练预算 · 末段存档 · 官方测试 600 剔泄漏后 strict 563 · 同机同趟评**。"
               "『5.08M 是预算点、不是收敛点』见 `03` L236-A。")
