# -*- coding: utf-8 -*-
"""训练期产物的读取与聚合 —— 图 2/3/4/5 共用。

🔴 数据口径（每张用到本模块的图都必须在图注里写清楚）：
   · `trend`  = 每段在**验证集 100 场景**上的评估，20 个点。含到达率 / 碰撞率 /
                违规次数每局 / 紧急步% / 每局时长。**不是**论文主表用的官方测试集 600。
   · `curves` = 每次 rollout 的训练期遥测（盾归口 / 打满舵率 / 转艏增量 / 态势分布 / PPO 内部量）。
   两者都与重评无关，所以在重评之前就能画。

🔴 `违规次数/局` 是**总违规**（直航 + 让路），不是让路违规。
   `代码/tests/test_usv_evaluate.py:381` 钉死了 `violations == standon + giveway`，
   而训练期只回传合计值。**要拆开必须等重评**，图注与正文一律写“每局 COLREGs 违规（合计）”。
"""
import json
import os
import glob
import re
import collections

import numpy as np

import paper_style as PS

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))                 # …/TRB
SRC = os.path.join(ROOT, "Paper", "正式实验", "01_训练产物")
OUT_DIRS = [os.path.join(ROOT, "Paper", "正式实验", "04_图"),
            os.path.join(ROOT, "Paper", "01_论文稿", "figs")]
FULL_SEG = 20
CRASH_ARR = 50.0     # 与 `代码/bgate_judge.py:16` / `_common.py:44` 同一判据，禁止另定

#: TAG → (英文标签, 颜色, 动作空间, 是否带盾)。**全项目每张图必须一致**。
#  🔴 2026-08-01 later-5：这里的标签**必须与论文表 1 / 表 2 的行名逐字相同**（user 明令
#     「用词不统一……不要有类似的情况出现了」）。改这里 = 四张图的图例与轴标签一起改。
#  🔴 与 `nature_style.ARM_COLOR` 的出入：base 由 neutral_light 改 neutral_dark、
#     rr 由 neutral_light 改 neutral_mid——两条曲线在图 4(a) 里几乎重合，都用最浅的灰
#     会看不见也分不开。正式迁移时按本表。
ARMS = collections.OrderedDict([
    ("ours", ("Ours",                     PS.PALETTE["blue_main"],      "cont", True)),
    ("disc", ("Discrete, masking",        PS.PALETTE["red_strong"],     "disc", True)),
    ("base", ("Discrete baseline",        PS.PALETTE["neutral_dark"],   "disc", False)),
    ("rr",   ("Discrete, soft reward",    PS.PALETTE["neutral_mid"],    "disc", False)),
    ("uns",  ("Continuous, no shield",          PS.PALETTE["gold"],           "cont", False)),
    ("ush",  ("Continuous, shield",       PS.PALETTE["teal"],           "cont", True)),
    ("ab0",  ("Ablation reference",       PS.PALETTE["neutral_black"],  "cont", True)),
    ("abB",  ("Ablation, bounded only",   PS.PALETTE["green_3"],        "cont", True)),
    ("abG",  ("Ablation, symmetric only", PS.PALETTE["violet"],         "cont", True)),
])

#: `trend` 里的指标键 → 英文轴名
TREND = {"到达率%": "Arrival rate (%)", "碰撞率%": "Collision rate (%)",
         "违规次数/局": "COLREGs violations per episode",
         "紧急步%": "Emergency-control steps (%)", "Ep长s": "Episode length (s)"}


def load():
    """{tag: {seed: {"trend": [...], "curves": [...], "n": 段数}}}

    🔴 同一个 (臂, 种子) 在树里可能出现多份：`**/segments/*@sNN.progress.json` 是逐段存档，
       只含前 NN+1 段；跨机目录之间也可能撞到同名产物。递归 glob 的返回次序不保证，
       直接按 seed 覆盖会让**某一份残档随机盖掉整份**（2026-08-01 实测：rr s0 明明有满额
       20 段，却被 @s13 那份盖成 14 段）。故按「段数多者胜、段数相同则遥测多者胜」归并。
    """
    out = collections.defaultdict(dict)
    for p in glob.glob(os.path.join(SRC, "**", "*.progress.json"), recursive=True):
        d = json.load(open(p, encoding="utf-8"))
        m = re.search(r"F240([a-zA-Z0-9]+?)Ppo", os.path.basename(p))
        if not m or m.group(1) not in ARMS:
            continue
        rec = {"trend": d.get("trend") or [], "curves": d.get("curves") or []}
        rec["n"] = len(rec["trend"])
        old = out[m.group(1)].get(d["seed"])
        if old is None or (rec["n"], len(rec["curves"])) > (old["n"], len(old["curves"])):
            out[m.group(1)][d["seed"]] = rec
    return out


def final(D, tag, key):
    """某条臂在**各自最后一段**上的逐种子取值 → {seed: 值}。

    未跑满的 run 取它自己的最后一段（不截断到最短），调用方须自行披露段数差异。
    """
    return {sd: r["trend"][-1][key] for sd, r in sorted(D[tag].items()) if r["trend"]}


def yaw_incr(D, tag, last_k=20):
    """转艏增量：每颗种子取其末 `last_k` 次 rollout 的中位数 → {seed: 值}。

    单次 rollout 的值抖得厉害（16384 步一采），取末段中位数作为「训练末期水平」。
    """
    out = {}
    for sd, r in sorted(D[tag].items()):
        v = [c["roll_yaw_incr_mean"] for c in r["curves"][-last_k:]
             if isinstance(c.get("roll_yaw_incr_mean"), (int, float))]
        if v:
            out[sd] = float(np.median(v))
    return out


def boot_ci(vals, n_boot=10000, seed=0, stat=np.median):
    """按种子重采样的自助法 95% 区间。种子写死 ⟹ 同样输入必得同样区间。"""
    v = np.asarray([x for x in vals if x is not None], float)
    if len(v) < 2:
        return (float(v[0]), float(v[0])) if len(v) else (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    bs = stat(rng.choice(v, size=(n_boot, len(v)), replace=True), axis=1)
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


# ══════════════════════════════════════════════════════════════════════════════
#  🔴 重评产物（官方测试集 600）—— 2026-08-01 later-5 起，图 2 / 图 3 一律走这里
#
#  为什么必须换：主表与 §5.3–§5.7 的数字已经全部换成官方测试集 600 了。
#  图若还画验证集 100，就会出现「图和分析对不上」（user 明令：不要牛头不对马嘴）。
#  训练曲线（图 4）不在此列 —— 它画的是训练过程，本来就该用训练期遥测。
# ══════════════════════════════════════════════════════════════════════════════
REEVAL = os.path.join(ROOT, "Paper", "正式实验", "02_重评产物")
_RE_PAT = re.compile(r"F240([A-Za-z0-9]+?)Ppo[Ss](\d+)")


def load_reeval(pas="正式-最佳"):
    """→ {tag: {seed: strict 记录}}。strict = 官方测试集 600 且零泄漏的那一档。"""
    path = os.path.join(REEVAL, pas, "all.json")
    if not os.path.exists(path):
        raise SystemExit(f"🔒 找不到重评产物 {path} —— 图 2/3 必须画官方测试集，不许拿验证集顶上")
    raw = json.load(open(path, encoding="utf-8"))
    out = {}
    for k, v in raw.items():
        m = _RE_PAT.search(k)
        if m:
            out.setdefault(m.group(1), {})[int(m.group(2))] = v["strict"]
    return out


#: 图上用的键 → 重评记录里的取法（控制质量是嵌一层的）
_RE_GET = {
    "到达率%": lambda r: r["到达率%"],
    "碰撞率%": lambda r: r["碰撞率%"],
    "违规次数/局": lambda r: r["违规次数/局"],
    "紧急步%": lambda r: r["紧急步%"],
    "yaw": lambda r: r["控制质量"]["yaw_incr_mean"],
    "acc": lambda r: r["控制质量"]["accel_incr_mean"],
    "giveway": lambda r: r["控制质量"]["giveway_violations"],
}


def final_re(RD, tag, key):
    """重评口径的逐种子取值 → {seed: 值}。签名与 `final()` 对齐，出图代码改动最小。"""
    g = _RE_GET[key]
    return {sd: g(r) for sd, r in sorted(RD.get(tag, {}).items())}


def survey(D):
    """把各臂的种子数与未满额情况打出来，出图前肉眼过一遍。"""
    print("载入：")
    short = []
    for tag in ARMS:
        runs = D.get(tag, {})
        part = [f"s{s}({r['n']}/{FULL_SEG})" for s, r in sorted(runs.items()) if r["n"] < FULL_SEG]
        print(f"  {tag:5s} {len(runs)} 颗种子" + (f"  ⚠️ 未满额：{', '.join(part)}" if part else ""))
        short += [(tag, s, r["n"]) for s, r in sorted(runs.items()) if r["n"] < FULL_SEG]
    return short


def sign_test(a, b):
    """配对符号检验（双侧，精确二项）。返回 (下降的对数, 有效对数, p)。

    🔴 $n=8$ 时双侧最小可达 $p=2/2^8=0.0078$；7/8 同向即 $p=0.0703$。
       报 p 的同时必须报同向对数，否则读者会把 0.07 读成“不显著”而非“样本量到顶”。
    """
    import math
    d = [x - y for x, y in zip(a, b) if x != y]
    n = len(d)
    if n == 0:
        return 0, 0, 1.0
    k = sum(1 for x in d if x > 0)
    p = sum(math.comb(n, i) for i in range(n + 1)
            if abs(i - n / 2) >= abs(k - n / 2)) / 2 ** n
    return n - k, n, min(1.0, p)


def wilcoxon(a, b):
    """配对 Wilcoxon 符号秩检验（双侧，精确枚举）。返回 (下降的对数, 有效对数, p)。

    🔴 **主检验用它、不用符号检验**：符号检验只看方向、丢掉幅度，$n=8$ 时 7/8 同向
       就顶到 $p=0.07$，会把「转艏增量降 39%、7 颗种子同向」这种明显效应标成 n.s.，
       图上说的与数据说的正好相反。符号秩用上幅度，同样数据给到 $p=0.016$。
       $n\le10$ 全枚举 $2^n$ 即精确 $p$，无需大样本近似。
    """
    import itertools
    d = [x - y for x, y in zip(a, b) if x != y]
    n = len(d)
    if n == 0:
        return 0, 0, 1.0
    order = sorted(range(n), key=lambda i: abs(d[i]))
    rank = [0.0] * n
    i = 0
    while i < n:                                   # 绝对值并列取平均秩
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        for k in range(i, j + 1):
            rank[order[k]] = (i + j) / 2 + 1
        i = j + 1
    tot = sum(rank)
    W = sum(rank[i] for i in range(n) if d[i] > 0)
    obs = abs(W - tot / 2)
    cnt = sum(1 for m in itertools.product([0, 1], repeat=n)
              if abs(sum(rank[i] for i in range(n) if m[i]) - tot / 2) >= obs - 1e-9)
    return n - sum(1 for x in d if x > 0), n, cnt / 2 ** n


def stars(p):
    """显著性记号。"""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
