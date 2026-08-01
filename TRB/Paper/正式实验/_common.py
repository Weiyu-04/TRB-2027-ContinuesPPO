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

SEC = "strict"       # 报数一律 strict（官方测试 600 剔泄漏）；训练期里程碑数永不进表（`03` L232 铁律）

# ══════════════════════════════════════════════════════════════════════════════════════════
# 种子结局三分类（`03` L243-改③ · user 2026-07-29 拍板）
#
# 🔴 **为什么必须三分、不能沿用单一 `到达 <50 = 崩`**：
#   L235 / L238 / L241 **三次**证明"到达低 ≠ 崩" —— B 臂 s2（26.82%）与 D 臂 s3（9.77%）都不是掉进坏盆地，
#   而是**起飞晚、预算到期时还在爬**。而主表有「练成的种子数」这一列，论文要拿它讲**训练可靠性**；
#   把"掉进打转吸引子"和"预算不够"混成同一个数，那条 finding 就站不住（审稿人看曲线就会指出来）。
#
# 判据（**不带旋钮·从机制推出来的，不是调出来的**）：
#   打转 ≡ 跑满回合上限却从没到过目标 ⟹ 每局时长顶到天花板。
#   天花板 = k_max × dt = 170 步 × 10 秒 = 1700 秒（`代码/trb_env/usv_env.py:143` g0.time_step.end）。
#   取 0.8 × 天花板 = **1360 秒** 为界。
#
# 实证（本窗口在 `结果0729-59臂同趟重评` 上重算 · 用的就是报数表自己的量 `strict.控制质量.ep_len_s`）：
#   练成（到达 ≥50，n=52）  每局秒 [ 466,  642]
#   欠训（到达 <50，n=2）   每局秒 [ 675,  814]   ← B 臂 s2 / D 臂 s3，与 L235/L241 的诊断一致 ✅
#   崩  （到达 <50，n=5）   每局秒 [1609, 1691]   ← 金标 s5/s6、A 臂 s6、对标 s0/s1
#   ⟹ 814 与 1609 之间**空了近 800 秒**，1360 这条线落在正中间，不敏感。
#
# 🔴 **诚实边界（论文这一列必须一起写）**：全库更早那批 run 里约 12% 的低到达 run 落在
#   1100~1600 秒的**过渡带**，那一带的标签本身就是灰的 ⟹ 报表时必须**同时报过渡带条数**，
#   不能只甩一个"练成 N/12"。`classify_counts()` 会把它一并算出来。
# ══════════════════════════════════════════════════════════════════════════════════════════
CRASH_ARR = 50.0        # 到达率门：< 此 = 没练成（`代码/bgate_judge.py:16`，全项目统一，**禁止逐臂另定**·`03` L234-B）
EP_CAP_S = 170 * 10.0   # 回合时长天花板 = k_max × dt（单一真相源：`trb_env/usv_env.py:143` + 每步 10 秒）
SPIN_LINE_S = 0.8 * EP_CAP_S            # = 1360 秒：到达 <50 且每局时长 ≥ 此 ⟹ 崩（打转吸引子）
GREY_BAND_S = (1100.0, 1600.0)          # 过渡带：标签本身是灰的 ⟹ 必须报条数，别装作没有

OUTCOME_TRAINED, OUTCOME_SPIN, OUTCOME_UNDER = "练成", "崩", "欠训"


def outcome_of(v):
    """一颗种子的结局：练成 / 崩（打转吸引子）/ 欠训（还在爬·没起飞）。参数 v = 该 checkpoint 的重评结果。"""
    s = v[SEC]
    if s["到达率%"] >= CRASH_ARR:
        return OUTCOME_TRAINED
    return OUTCOME_SPIN if s["控制质量"]["ep_len_s"] >= SPIN_LINE_S else OUTCOME_UNDER


def classify_counts(entries):
    """一条臂的结局计数 + 过渡带条数（论文那一列要连过渡带一起报）。"""
    out = {OUTCOME_TRAINED: 0, OUTCOME_SPIN: 0, OUTCOME_UNDER: 0, "过渡带": 0, "n": len(entries)}
    for _, v in entries:
        out[outcome_of(v)] += 1
        if GREY_BAND_S[0] <= v[SEC]["控制质量"]["ep_len_s"] <= GREY_BAND_S[1]:
            out["过渡带"] += 1
    return out


def outcome_note():
    """给表/图脚注用的判据说明（**判据必须与数字同处出现**，不许只甩一个 N/12）。"""
    return (f"种子结局判据：练成 = 到达率 ≥ {CRASH_ARR:.0f}%；崩（打转吸引子）= 到达率 < {CRASH_ARR:.0f}% "
            f"且每局时长 ≥ {SPIN_LINE_S:.0f} 秒（= 0.8 × 回合上限 {EP_CAP_S:.0f} 秒）；欠训 = 其余。"
            f"过渡带（{GREY_BAND_S[0]:.0f}~{GREY_BAND_S[1]:.0f} 秒）内标签存在歧义，条数一并列出。")

#: 臂定义 = (显示名, checkpoint 名里的特征串, 是否进头条表)
ARM_SPECS = [
    # 🔴 L243-续8（D 线 R3·实跑 arm_of() 坐实）：`arm_of` 是**第一个命中就返回**，而正式臂的存档名
    #   恰好也是 `Base_s3_F240basePpoS3` / `Rule-reward_s0_F240rrPpoS0` / `Discrete-safe_s0_F240discPpoS0`
    #   ⟹ 下面三条**历史**判据会先命中，把正式臂贴上探索期的标签，两代实验混进同一行。
    #   ⟹ 三条都加 `"F240" not in ck` 把正式臂排除掉（正式臂由下面 F240* 那组认领）。
    ("Base（离散·无盾）",            lambda ck: ck.startswith("Base_") and "F240" not in ck,        True),
    ("Rule-reward（离散·软奖励）",    lambda ck: ck.startswith("Rule-reward_") and "F240" not in ck, True),
    ("Discrete-safe（对标论文）",     lambda ck: ck.startswith("Discrete-safe_") and "dsSeg" not in ck and "F240" not in ck, True),
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
    # 🔴 2026-08-01（`04 §八` 早就记过，一直没迁）：这三条原来是 `False` ⟹ `make_头条表.py`
    #   开头就 `if not in_head: continue` ⟹ **主表少输出 ⑦⑧⑨ 三行**，而论文主表是 9 行 × 8 列 = 72 个待填数。
    #   少行**不会报错**，正是本项目反复栽的「静默」那一族。合成正式臂数据实测：改前 6 行、改后 9 行。
    #   （不在数据里的臂本来就会被 `name not in ba` 跳过，所以置 True 不会把探索期的臂混进来。）
    ("【正式】消融·都不改",             lambda ck: "F240ab0Ppo" in ck,                True),
    ("【正式】消融·只 Beta",           lambda ck: "F240abBPpo" in ck,                True),
    ("【正式】消融·只改状态机",          lambda ck: "F240abGPpo" in ck,                True),
]


def arm_of(ck):
    for name, pred, _ in ARM_SPECS:
        if pred(ck):
            return name
    return None


def load_pass(d, prefix="g", expect_strict=None, expect_rows=None):
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
    # 🔴 L243-续8（D 线 R5）：`v.get("anchor", {})` 在 `anchor` 显式为 **None** 时返回的是 None、
    #   不是 {}，`.get` 会 AttributeError 把整张表打掉；而 `anchor` 缺失/为 None 恰恰是
    #   "锚点检查被跳过"（sidecar 与 ckpt 不同步时 reeval 会打印一行就跳过）的正常表现。
    #   ⟹ 用 `(v.get("anchor") or {})`，并且**把"跳过"也算未通过** —— 跳过 = 没验过 = 不能进表。
    bad = [ck for ck, v in rows.items() if (v.get("anchor") or {}).get("通过") is not True]
    if bad:
        raise SystemExit(f"🔒 锚点自检未通过/被跳过 {len(bad)} 条：{bad[:5]} ⟹ 评估可能配错，这张表不能出。"
                         "（被跳过 = 那条 run 的 sidecar 与存档不同步，等于没验，同样不能出表。）")
    # 🔴 L243-续8：再核一遍"这趟是不是完整的" —— 组文件在、键也一致，但某一组里的臂**跑挂了几条**时，
    #   上面每一道闸都过得去。调用方给了 expect_rows 就硬核。
    if expect_rows is not None and len(rows) != expect_rows:
        raise SystemExit(f"🔒 这趟只有 {len(rows)} 条 checkpoint、期望 {expect_rows} ⟹ 有臂跑挂了，"
                         "先看重评日志里的 ❌，别拿缺臂的表出数。")
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
    cls = classify_counts(lst)          # 🆕 `03` L243-改③：三分类 + 过渡带，与「健康」一起带出
    return dict(
        n=len(lst), 健康=sum(1 for x in arr if x >= CRASH_ARR), 种子=[v["seed"] for _, v in lst],
        练成=cls[OUTCOME_TRAINED], 崩=cls[OUTCOME_SPIN], 欠训=cls[OUTCOME_UNDER], 过渡带=cls["过渡带"],
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


def steps_from_rows(rows):
    """从**这趟数据自己**读训练预算（每条臂的 `num_timesteps`），不靠任何常量。

    🔴 `03` L243 复审又抓到同一族坑的下一层：`budget_note` 虽然不写死数字了，
       但调用方一律传 `FORMAL['steps']` 常量 ⟹ 拿它去印**历史那趟 5.08M 的表**，
       表头会一声不响写成"15.24M 步预算 · 验证集最佳存档"。**数字不是手抄的，但口径是抄错的**，
       后果一样（L240 那条教训的原话：不报错，会把错数字印进论文）。
       ⟹ 预算改成从数据推；步数不一致时**明写"混合"**，绝不挑一个印上去。
    """
    ts = sorted({int(v.get("num_timesteps") or 0) for v in rows.values() if v.get("num_timesteps")})
    if not ts:
        return "训练预算未记录（存档 sidecar 缺 num_timesteps）"
    if len(ts) == 1:
        n = ts[0]
        return f"{n:,} 步预算" + (f"（{n // SEG_STEPS} 段 × {SEG_STEPS:,}）" if n % SEG_STEPS == 0 else "")
    return (f"🔴 **步数不一致**：{ts[0]:,} ~ {ts[-1]:,}（{len(ts)} 种）"
            " ⟹ 这不是同预算比较，别当一张表报")


def budget_note(n_strict, steps=None, ckpt_policy=None, rows=None):
    """口径脚注**按实测拼**，不许写死（`03` L240 + L243）。

    `rows` 给了就从数据推预算（优先），否则退回调用方给的 `steps`。
    """
    lead = steps_from_rows(rows) if rows else (f"{steps}" if steps else "训练预算见正文")
    pol = ckpt_policy or "存档口径见正文"
    return (f"口径：**{lead} · {pol} · 官方测试集 strict {n_strict} · 同机同趟评**。"
            "『步数上限是预算点、不是收敛点』见 `03` L236-A。")


#: 正式实验的口径常量（user 2026-07-29 拍板：3 台机 · 12 颗种子 · NSEG=30 起跑 · 报数时往回截）
#  🔴 `budget_seg` = **报数用的段数**，跑的时候是 30 段；T1 判据说 20 段就够 ⟹ 把它改成 20 并重出全部表。
#     同一张表里所有臂必须用同一个 budget_seg（`03` L242-A）。报数写**实际步数**，别写名义的 "10M"。
SEG_STEPS = 507904                                  # 每段实际步数（SB3 按 rollout 2048×8=16384 取整后的值）
#  🔴 `03` L243-续2：两天硬期限把 NSEG=30 那个决定推翻了 —— 30 段在 48 小时里只跑得出 3~6 颗种子，
#     20 段能跑 6~12 颗。**丢种子是致命的（n=3 时符号检验最小 p=0.25，到不了 0.05）；
#     丢步数只是论文里一句「该预算下尚未收敛」**（写法 `03` L241-B 已定死）⟹ 回到 20 段。
FORMAL = {"n_seg_run": 20,                          # 起跑段数（10,158,080 步）
          "budget_seg": 20,                         # 报数段数（体检脚本给出实际可用的最小段数）
          "ckpt": "验证集最佳存档", "ckpt_alt": "末段存档",
          "n_strict": 600, "n_seeds": 12, "n_machines": 3, "lanes_per_machine": 10}
FORMAL["steps"] = (f"{FORMAL['budget_seg'] * SEG_STEPS:,} 步预算"
                   f"（{FORMAL['budget_seg']} 段 × {SEG_STEPS:,}）")
