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
    # ── 正式实验 9 条臂（`03` L240 定稿·TAG 前缀 F240）────────────────────────────
    #    TAG 里必须含 ppo/diag/probe/ab 之一，否则 run_step4e 的 PPO 隔离闸会拦（:405-410）
    ("【正式】Ours（Beta+对称让路+盾）",  lambda ck: "F240oursPpo" in ck,               True),
    ("【正式】Discrete-safe 对标",      lambda ck: "F240discPpo" in ck,               True),
    ("【正式】Base 离散无盾",           lambda ck: "F240basePpo" in ck,               True),
    ("【正式】Rule-reward 离散软奖励",   lambda ck: "F240rrPpo" in ck,                 True),
    ("【正式】U-无盾（连续·极简）",       lambda ck: "F240unsPpo" in ck,                True),
    ("【正式】U-有盾（与无盾逐字同配方）",  lambda ck: "F240ushPpo" in ck,                True),
    ("【正式】消融·都不改",             lambda ck: "F240ab0Ppo" in ck,                False),
    ("【正式】消融·只 Beta",           lambda ck: "F240abBPpo" in ck,                False),
    ("【正式】消融·只改状态机",          lambda ck: "F240abGPpo" in ck,                False),
]


def arm_of(ck):
    for name, pred, _ in ARM_SPECS:
        if pred(ck):
            return name
    return None


def load_pass(d, prefix="g", expect_strict=None):
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
    # 🔴 分母**不再写死**（`03` L240）：以往写死 563 是因为所有臂都在小集上训、与官方测试 600 撞了
    #   23 训练 + 14 验证 = 37 个场景。正式实验全部改在官方 1300 上训，与测试 600 **零交集**
    #   ⟹ 分母是 **600**。写死一个数就会在换口径时要么让脚本罢工、要么把错数字印进论文。
    #   ⟹ 调用方**必须显式**说自己期望多少（`expect_strict=`），不传就只查组间一致并把实测值打出来。
    if expect_strict is not None and len(ref) != expect_strict:
        raise SystemExit(f"🔒 {SEC} 场景数 = {len(ref)}，调用方期望 {expect_strict} ⟹ 口径不对"
                         "（同趟里混进了别的训练集的臂？单独评过？跨趟拼过？）这张表不能出。")
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


def budget_note(n_strict, steps=None, ckpt_policy=None):
    """口径脚注**按实测拼**，不许写死（`03` L240）。

    🔴 教训：原来这里是一句字面量 "5.08M 步 … strict 563"。它**不会报错**，
    换了口径之后会一声不响地把**错的预算和错的分母**印进论文的表和图。
    三处同款字面量（本文件 + 出图脚本两处）在起跑前审计里被一起抓出来。
    """
    lead = f"{steps}" if steps else "训练预算见正文"
    pol = ckpt_policy or "存档口径见正文"
    return (f"口径：**{lead} · {pol} · 官方测试集 strict {n_strict} · 同机同趟评**。"
            "『步数上限是预算点、不是收敛点』见 `03` L236-A。")


#: 正式实验的口径常量（定稿后写死在这里，各脚本一律从这里取，别各处再抄一遍）
FORMAL = {"steps": "10.16M 步预算（10,158,080 = 20 段 × 507,904）",
          "ckpt": "验证集最佳存档", "ckpt_alt": "末段存档", "n_strict": 600, "n_seeds": 10}
