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
#  🔴 与 `nature_style.ARM_COLOR` 的出入：base 由 neutral_light 改 neutral_dark、
#     rr 由 neutral_light 改 neutral_mid——两条曲线在图 4(a) 里几乎重合，都用最浅的灰
#     会看不见也分不开。正式迁移时按本表。
ARMS = collections.OrderedDict([
    ("ours", ("Ours (continuous + shield)",     PS.PALETTE["blue_main"],      "cont", True)),
    ("disc", ("Discrete-safe (benchmark)",      PS.PALETTE["red_strong"],     "disc", True)),
    ("base", ("Base (discrete, no shield)",     PS.PALETTE["neutral_dark"],   "disc", False)),
    ("rr",   ("Rule-reward (discrete)",         PS.PALETTE["neutral_mid"],    "disc", False)),
    ("uns",  ("Continuous, no shield",          PS.PALETTE["gold"],           "cont", False)),
    ("ush",  ("Continuous + shield",            PS.PALETTE["teal"],           "cont", True)),
    ("ab0",  ("Ablation: neither",              PS.PALETTE["neutral_black"],  "cont", True)),
    ("abB",  ("Ablation: bounded only",         PS.PALETTE["green_3"],        "cont", True)),
    ("abG",  ("Ablation: symmetric entry only", PS.PALETTE["violet"],         "cont", True)),
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
