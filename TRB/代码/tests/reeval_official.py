#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E5 / E6 官方测试集【泛化测】——eval-only·零训练·不烧训练卡（`03` L207-E ·`02` later-3「下一步①」）。

═══ 这个脚本回答什么 ═══
  我们复现的两条臂（连续 Continuous-safe / 离散 Discrete-safe）都是在 **94 个训练场景**（`manifest_hocr_200`）
  上训的；Krasowski & Althoff 2024 Table III 的 Discrete-safe **86.3%** 是在 **官方 1400 训 / 600 测** 上跑的。
  E3（`03` L206）实测我们的离散臂只有 56-57% ⟹ **是"94 训太小"还是"我们复现本身有问题"？**
  本脚本 = **不重训**，直接把【现有 checkpoint】搬到【官方 held-out 测试集】上评：
    · 仍 ~80%（连续）/ ~86%（离散） ⟹ 94 训配方泛化没问题 ⟹ 直接用官方测试集报数 ⟹ **省掉整趟大集烧卡**。
    · 明显掉（<60%） ⟹ 训练集确实太小 ⟹ **才**值得烧大集（10M / 热启动之间挑·另需 user 拍板）。

═══ 🔴 泄漏必须先剔（本脚本的核心·别用裸 600）═══
  `manifest_hocr_200` 的 94 个训练 T-id 里 **23 个落在官方 600 测试集内（24.5%）** ⟹ 直接用 600 = 自我污染。
  本脚本给【两个】口径（同一趟 eval·零额外成本）：
    · **577 = 600 − 23 训练泄漏**  ← 主口径（`03` L207-E3 定的）
    · **563 = 577 − 14 验证泄漏**  ← 更严口径：manifest 的 40 个"测试"场景在训练期被【每段反复看】+
      **源种子是看着它挑的**（`03` L192 定性=「验证集」非干净测试集）⟹ 563 才是"真·一眼没见过"。
  两个都报 → 论文按需选口径、且差多少一目了然（若两者接近 = 泄漏影响小 = 结论更硬）。

═══ 三层防误判（都是防"跑出来的数是错的却看不出来"）═══
  ① **锚点复现检查**（默认开·`REEVAL_ANCHOR=1`）：先在 **checkpoint 自己记录的测试集**上重评一次，
     与它 `.progress.json` 里 `trend[-1]["到达率%"]` 对比。**对不上 = eval 环境配置与训练时不一致**
     （连续臂的 shield/cone/augment_rho/去朝向门这些 knob **不进 config_sig**、错了会【静默】改数字）→ 默认直接中止。
     这一步只花 40 个场景，是本脚本最重要的防御（`03` L192「goal_ignore_orientation 能静默混两套判据」那条洞）。
  ② **数据集一致性检查**：逐 checkpoint 读 `config_sig["dataset"]`，泄漏集 = 所有被评 checkpoint 的训练 id
     **并集**（不同臂若用了不同 manifest，clean 集自动收紧 + 大声打印）→ 保证「每条臂都真没见过」且【同一分母】可比。
  ③ **下载缺额守卫**：clean 集实际拿到 <95% 直接中止（防静默缩水污染结论）。

═══ 用法（**在 ~/trb 根目录下跑**·脚本随 代码/ 同步上服务器·不以 test_ 开头故不进回归套件）═══
  0) 纯逻辑自检（**本机就能跑·无需 commonocean/vesselmodels**·验 split/泄漏/计数）：
       REEVAL_SELFTEST=1 python 代码/tests/reeval_official.py
  1) 先列 checkpoint（零风险·只读 sidecar·几秒）：
       cd ~/trb && STEP4E_SDIR=$HOME/trb/scenarios STEP4E_CODE_DIR=代码 REEVAL_LIST=1 python 代码/tests/reeval_official.py
  2) 冒烟（~2-5 分钟·30 场景·单 ckpt·验 harness/路径/池）：
       cd ~/trb && STEP4E_SDIR=$HOME/trb/scenarios STEP4E_CODE_DIR=代码 REEVAL_SMOKE=1 \
         REEVAL_CKPTS=<连续臂ckpt名> python 代码/tests/reeval_official.py
  3) 全量（E5+E6·577 场景 × 每个 ckpt）：
       cd ~/trb && STEP4E_SDIR=$HOME/trb/scenarios STEP4E_CODE_DIR=代码 \
         REEVAL_CKPTS=<连续ckpt>,<离散ckpt> REEVAL_OUT=reeval_official.json \
         python 代码/tests/reeval_official.py

【环境变量】
  STEP4E_CODE_DIR   代码目录（服务器设 `代码`·让 import run_step4e/trb_env 找得到；默认 '.'）
  STEP4E_SDIR       T-*.xml 场景缓存目录（缺的会自动下·已缓存跳过）
  REEVAL_CKPTS      要评的 checkpoint base 名（逗号分隔·不带 .zip）；空 = 在 CKDIRS 里自动发现全部
  REEVAL_CKDIRS     checkpoint 目录（冒号分隔·**支持通配符**如 `/root/trb/*/checkpoints`）
  REEVAL_FORCE_KIND / REEVAL_FORCE_DATASET  老 ckpt 无 .progress.json 时的显式降级（此时【做不了锚点检查】）
  REEVAL_ANCHOR     1(默认)=做锚点复现检查 / 0=跳过（**别随便关**）
  REEVAL_ANCHOR_TOL 锚点容差（百分点·默认 0.0）。⚠️ 实际生效容差 = max(该值, 一局的分量 100/N)：
                    配置错会差【很多】，差一局是浮点噪声——一局内只警告不中止，超过就中止。
  REEVAL_ANCHOR_SOFT 1 = 锚点对不上只【警告】不中止（默认 0 = 中止）
  REEVAL_MANIFEST_DIRS  找 manifest/OT 文件的目录（冒号分隔·默认 balanced_pool 的几个常见位置）
  REEVAL_STRICT_ONLY 1 = 只在 563 严口径上评（省 14 局·默认 0 = 评 577 并【同时】报 563 子集）
  REEVAL_N          clean 集上限（默认 0=全部）；调试用
  REEVAL_SMOKE      1 → REEVAL_N=30 且只评第一个 ckpt
  REEVAL_OUT        结果 json（默认 reeval_official.json·写 cwd）
  REEVAL_ENVCFG_ACK 1 = 承认「连续臂 eval env knob 非金标默认」（默认非默认即中止）
  REEVAL_ALLOW_UNKNOWN_DATASET 1 = 允许评「数据集无法从 sidecar 还原」的 checkpoint（默认中止·因泄漏剔不干净）

⚠️ 本脚本【不训练、不写 checkpoint、不改任何模型字节】。到达率不进梯度 → 重评纯测量。
"""
from __future__ import annotations
import glob
import json
import math
import os
import random
import sys

# ---------- 开关 ----------
_CODE = os.environ.get("STEP4E_CODE_DIR", ".")
SELFTEST = os.environ.get("REEVAL_SELFTEST", "0") == "1"
LIST_ONLY = os.environ.get("REEVAL_LIST", "0") == "1"
SMOKE = os.environ.get("REEVAL_SMOKE", "0") == "1"
ANCHOR = os.environ.get("REEVAL_ANCHOR", "1") == "1"
ANCHOR_TOL = float(os.environ.get("REEVAL_ANCHOR_TOL", "0.0"))
ANCHOR_SOFT = os.environ.get("REEVAL_ANCHOR_SOFT", "0") == "1"
STRICT_ONLY = os.environ.get("REEVAL_STRICT_ONLY", "0") == "1"
CLEAN_N = int(os.environ.get("REEVAL_N", "0"))
OUT = os.environ.get("REEVAL_OUT", "reeval_official.json")
ENVCFG_ACK = os.environ.get("REEVAL_ENVCFG_ACK", "0") == "1"
if SMOKE:
    CLEAN_N = 30

_CKDIRS_RAW = os.environ.get(
    "REEVAL_CKDIRS",
    "结果/checkpoints:checkpoints:*/checkpoints:结果/*/checkpoints").split(":")   # server 结果目录直挂 ~/trb 下 → `*/checkpoints`
CKDIRS = []
for _d in _CKDIRS_RAW:                                     # 支持通配符（服务器上结果目录名带日期·别逐个硬写）
    CKDIRS.extend(sorted(glob.glob(_d)) if "*" in _d else [_d])
MANIFEST_DIRS = os.environ.get(
    "REEVAL_MANIFEST_DIRS",
    "balanced_pool:../balanced_pool:TRB/balanced_pool:" + os.path.join(_CODE, "..", "balanced_pool")).split(":")

# 官方口径常量（= `run_step4e.py` 默认模式·`03` L207-B 亲读码坐实：不设 STEP4E_MANIFEST 时就是这套）
OFF_N_TOTAL, OFF_TEST_FRAC, OFF_SPLIT_SEED, OFF_POOL = 2000, 0.30, 0, 2000
EXPECT_OFFICIAL_TEST = 600          # 2000×0.30
EXPECT_CLEAN_HOCR = 577             # 600 − 23（manifest_hocr_200 训练泄漏）
EXPECT_STRICT_HOCR = 563            # 577 − 14（manifest_hocr_200 验证泄漏）


# ---------------- 纯逻辑层（无重依赖·本机可自检） ----------------
def make_split_mirror(n_total, test_frac, split_seed=0, pool_size=None):
    """`run_step4e.make_split` 的【逐行镜像】——**只为让本机（无 vesselmodels）能自检**。

    🔴 真跑时【一定】会与 `R.make_split` 的返回值硬比对（见 `official_test_ids`），不一致直接中止 →
       镜像漂移不可能悄悄污染结果。别在这里"顺手改进"，它的唯一职责是复刻。
    """
    if not (0 < test_frac < 1):
        raise ValueError(f"test_frac 须 ∈ (0,1)，得到 {test_frac}")
    if pool_size and pool_size > n_total:
        ids = [i * pool_size // n_total for i in range(n_total)]
    else:
        ids = list(range(n_total))
    random.Random(split_seed).shuffle(ids)
    n_test = int(round(n_total * test_frac))
    return sorted(ids[n_test:]), sorted(ids[:n_test])


def official_split(R=None):
    """官方 1400/600 划分。`_pool_eff = POOL if POOL > n_total else None`（run_step4e:1575）→ 2000 不 > 2000 → None。"""
    pool_eff = OFF_POOL if OFF_POOL > OFF_N_TOTAL else None
    tr, te = make_split_mirror(OFF_N_TOTAL, OFF_TEST_FRAC, OFF_SPLIT_SEED, pool_size=pool_eff)
    if R is not None:                                          # 🔴 真身校验：镜像必须与官方实现逐元素相同
        rtr, rte = R.make_split(OFF_N_TOTAL, OFF_TEST_FRAC, OFF_SPLIT_SEED, pool_size=pool_eff)
        if (tr, te) != (rtr, rte):
            raise SystemExit("🔒 make_split 镜像与 run_step4e.make_split 不一致 → 本脚本的官方划分不可信，中止。"
                             f"（镜像 train/test={len(tr)}/{len(te)} vs 官方={len(rtr)}/{len(rte)}）")
        tr, te = rtr, rte                                      # 一律用官方实现的返回值（镜像只作校验）
    if len(te) != EXPECT_OFFICIAL_TEST:
        raise SystemExit(f"🔒 官方测试集应为 {EXPECT_OFFICIAL_TEST} 个，实得 {len(te)} → 常量/实现变了，中止。")
    return tr, te


def find_manifest(basename):
    """按 basename 在 MANIFEST_DIRS 里找 manifest 文件 → 绝对路径；找不到返回 None。"""
    if not basename or basename == "strided":
        return None
    for d in MANIFEST_DIRS:
        p = os.path.join(d, os.path.basename(basename))
        if os.path.exists(p):
            return os.path.abspath(p)
    return None


def manifest_tids(manifest_path):
    """读 manifest → (训练 T-id 集合, 测试 T-id 集合)。**只含 head_on/crossing 的 T-id**——
    追越是本项目自造的 OT-*.xml、**不在官方 2000 池里**（`03` L207-A），与官方测试集无交集、不参与泄漏计算。"""
    with open(manifest_path, encoding="utf-8") as fh:
        man = json.load(fh)
    tr = {int(x) for x in man["head_on"]["train"]} | {int(x) for x in man["crossing"]["train"]}
    te = {int(x) for x in man["head_on"]["test"]} | {int(x) for x in man["crossing"]["test"]}
    return tr, te


def selftest():
    """纯逻辑自检（本机可跑）：官方划分 + 泄漏剔除 + 计数断言。"""
    print("【SELFTEST】官方划分 + 泄漏剔除（纯逻辑·不碰模型/场景文件）", flush=True)
    tr, te = official_split(None)
    assert len(tr) == 1400 and len(te) == 600, (len(tr), len(te))
    assert not (set(tr) & set(te)), "官方 train/test 重叠"
    print(f"  ✅ 官方划分 1400/600（测试前 5：{te[:5]}）")
    mp = find_manifest("manifest_hocr_200.json")
    if mp is None:
        print(f"  ⚠️ 没找到 manifest_hocr_200.json（找过 {MANIFEST_DIRS}）→ 跳过泄漏断言")
        return
    mtr, mte = manifest_tids(mp)
    assert len(mtr) == 94, f"manifest_hocr_200 训练应 94 个 T-id，实得 {len(mtr)}"
    assert len(mte) == 40, f"manifest_hocr_200 测试应 40 个 T-id，实得 {len(mte)}"
    teS = set(te)
    leak_tr, leak_te = teS & mtr, teS & mte
    clean, strict = sorted(teS - mtr), sorted(teS - mtr - mte)
    assert len(leak_tr) == 23, f"训练泄漏应 23，实得 {len(leak_tr)}"
    assert len(leak_te) == 14, f"验证泄漏应 14，实得 {len(leak_te)}"
    assert len(clean) == EXPECT_CLEAN_HOCR, len(clean)
    assert len(strict) == EXPECT_STRICT_HOCR, len(strict)
    assert not (set(clean) & mtr) and not (set(strict) & (mtr | mte)), "剔除后仍有泄漏"
    print(f"  ✅ 泄漏：训练 23/94（24.5%）· 验证 14/40 → clean 577 / strict 563")
    print("【SELFTEST】全部通过 ✅", flush=True)


# ---------------- checkpoint 发现 / sidecar ----------------
FORCE_KIND = os.environ.get("REEVAL_FORCE_KIND", "").strip()          # 老 ckpt 无 sidecar 时的显式降级（会大声打印）
FORCE_DATASET = os.environ.get("REEVAL_FORCE_DATASET", "").strip()


def read_sidecar(base):
    """读 `<base>.progress.json` → dict（缺/坏 → 用 REEVAL_FORCE_* 拼一个最小 sidecar，否则 None）。"""
    pj = base + ".progress.json"
    if not os.path.exists(pj):
        if FORCE_KIND and FORCE_DATASET:                              # 显式降级：kind/dataset 人工给·但【没有 trend】=锚点检查做不了
            return {"party": "(forced)", "kind": FORCE_KIND, "colregs_weight": 0.0, "seed": None,
                    "num_timesteps": None, "trend": [],
                    "config_sig": {"kind": FORCE_KIND, "dataset": FORCE_DATASET}, "_forced": True}
        return None
    try:
        with open(pj, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as e:                                     # noqa: BLE001
        print(f"  ⚠️ sidecar 读失败 {pj}: {e}", flush=True)
        return None


def discover_ckpts():
    """在 CKDIRS 里找所有 `*.zip` 且有同名 `_vecnorm.pkl` 的 checkpoint → [base, ...]（去重·保序）。"""
    out, seen = [], set()
    for d in CKDIRS:
        for z in sorted(glob.glob(os.path.join(d, "*.zip"))):
            base = z[:-4]
            if not os.path.exists(base + "_vecnorm.pkl"):
                continue
            key = os.path.basename(base)
            if key in seen:                                    # 同名优先取 CKDIRS 靠前的目录
                continue
            seen.add(key)
            out.append(base)
    return out


def resolve_ckpts():
    """REEVAL_CKPTS 指定 → 在 CKDIRS 里按名解析；否则自动发现全部。"""
    spec = [s.strip() for s in os.environ.get("REEVAL_CKPTS", "").split(",") if s.strip()]
    if not spec:
        return discover_ckpts()
    out = []
    for name in spec:
        if os.path.exists(name + ".zip"):                      # 允许直接给路径
            out.append(name)
            continue
        hit = None
        for d in CKDIRS:
            cand = os.path.join(d, os.path.basename(name))
            if os.path.exists(cand + ".zip"):
                hit = cand
                break
        if hit is None:
            raise SystemExit(f"🔒 找不到 checkpoint `{name}`（在 {CKDIRS} 里都没有 .zip）→ 先跑 REEVAL_LIST=1 看有哪些。")
        if not os.path.exists(hit + "_vecnorm.pkl"):
            raise SystemExit(f"🔒 `{hit}` 缺 _vecnorm.pkl → 观测归一化重建不了、评出来的数是错的，中止。")
        out.append(hit)
    _dup = [n for n in {os.path.basename(x) for x in out} if [os.path.basename(y) for y in out].count(n) > 1]
    if _dup:                                                   # 结果按 basename 存 → 重名会互相覆盖、静默丢一条
        raise SystemExit(f"🔒 REEVAL_CKPTS 里有重名 checkpoint {_dup} → 结果会互相覆盖，中止（去重后重跑）。")
    return out


def describe(base, sc):
    """一行 checkpoint 摘要（给 REEVAL_LIST=1 / 主流程打印用）。"""
    if sc is None:
        return f"{os.path.basename(base):<52} （无 sidecar·无法判臂/数据集）"
    cfg = sc.get("config_sig") or {}
    trend = sc.get("trend") or []
    last = (trend[-1] or {}) if trend else {}
    return (f"{os.path.basename(base):<52} 臂={str(sc.get('party')):<16} kind={str(sc.get('kind')):<11} "
            f"seed={sc.get('seed')} 步={sc.get('num_timesteps')} 数据集={cfg.get('dataset')} "
            f"末段到达={last.get('到达率%')}")


# ---------------- 主流程 ----------------
def main():
    if SELFTEST:
        selftest()
        return

    sys.path.insert(0, _CODE)
    ckpts = resolve_ckpts()
    if not ckpts:
        raise SystemExit(f"🔒 没发现任何 checkpoint（找过 {CKDIRS}）→ 确认服务器上 checkpoint 的位置，或用 REEVAL_CKDIRS 指定。")
    sidecars = {b: read_sidecar(b) for b in ckpts}

    print("=" * 100)
    print(f"发现 {len(ckpts)} 个 checkpoint：")
    for b in ckpts:
        print("  " + describe(b, sidecars[b]))
    print("=" * 100, flush=True)
    if LIST_ONLY:
        print("REEVAL_LIST=1 → 只列不评，退出。", flush=True)
        return
    if SMOKE and len(ckpts) > 1:
        ckpts = ckpts[:1]
        print(f"【SMOKE】只评第一个：{os.path.basename(ckpts[0])}", flush=True)

    # ---- 重依赖（本机无 vesselmodels 会在这里失败·属预期） ----
    import numpy as np
    import run_step4e as R
    from trb_env.usv_scenarios import load_scenario_pool

    # ---- 连续臂 eval env 配置守卫（这些 knob 不进 config_sig·错了会【静默】改数字） ----
    _golden = {"STEP4E_CONTINUOUS_SHIELD": R._CONTINUOUS_SHIELD is True,
               "STEP4E_GOAL_CONE_HALF": R._GOAL_CONE_HALF_DEG is None,
               "STEP4E_AUGMENT_RHO": R._AUGMENT_RHO is False,
               "STEP4E_GOAL_IGNORE_ORIENT": R._GOAL_IGNORE_ORIENT is False}
    print(f"[env] 连续臂 eval env：shield={R._CONTINUOUS_SHIELD} cone={R._GOAL_CONE_HALF_DEG} "
          f"v_floor={R._GOAL_V_FLOOR} augment_rho={R._AUGMENT_RHO} 去朝向门={R._GOAL_IGNORE_ORIENT}", flush=True)
    _bad = [k for k, ok in _golden.items() if not ok]
    if _bad and not ENVCFG_ACK:
        raise SystemExit(f"🔒 连续臂 eval env knob 非【金标默认】：{_bad} → 与训练时不一致会静默改数字（`03` L192）。"
                         "确认是故意的就加 REEVAL_ENVCFG_ACK=1；否则清掉这些 STEP4E_* 变量重跑。")

    # ---- 官方划分 + 泄漏并集 ----
    _, official_test = official_split(R)
    leak_train, leak_val, ds_seen = set(), set(), {}
    for b in ckpts:
        sc = sidecars[b]
        ds = ((sc or {}).get("config_sig") or {}).get("dataset")
        ds_seen[os.path.basename(b)] = ds
        if ds in (None, "strided"):
            # sidecar 的 config_sig 【不记】n_total → "strided" 到底是 strided-200(训 140·可能与官方 600 有交集)
            # 还是 strided-2000(训 = 官方 1400·则官方 600 天然干净) 无法还原 ⟹ 泄漏剔不了 ⟹ 数不可信 ⟹ fail-closed。
            msg = (f"🔒 {os.path.basename(b)} 的数据集={ds!r}（非 manifest）→ 从 sidecar 【还原不出它的训练集】，"
                   "泄漏就剔不干净，官方测试集上的数字不可信。\n"
                   "   要么改评 manifest 训的 checkpoint；要么确认它训练集与官方 600 无交集后加 "
                   "REEVAL_ALLOW_UNKNOWN_DATASET=1（**并在论文口径里如实写**）。")
            if os.environ.get("REEVAL_ALLOW_UNKNOWN_DATASET", "0") == "1":
                print("  ⚠️ " + msg, flush=True)
                continue
            raise SystemExit(msg)
        mp = find_manifest(ds)
        if mp is None:
            raise SystemExit(f"🔒 {os.path.basename(b)} 记的数据集是 `{ds}`，但在 {MANIFEST_DIRS} 里找不到该 manifest → "
                             "泄漏剔不干净就等于自我污染，中止（用 REEVAL_MANIFEST_DIRS 指到 balanced_pool）。")
        t_tr, t_te = manifest_tids(mp)
        leak_train |= t_tr
        leak_val |= t_te
        print(f"  [数据集] {os.path.basename(b)} ← {os.path.basename(mp)}：训练 {len(t_tr)} T-id / 验证 {len(t_te)} T-id", flush=True)
    if len({v for v in ds_seen.values() if v}) > 1:
        print(f"  ⚠️⚠️ 被评的 checkpoint 用了【不同数据集】{ds_seen} → clean 集取并集剔除（更小但对所有臂同分母·可比）", flush=True)

    teS = set(official_test)
    clean_ids = sorted(teS - leak_train)
    strict_ids = sorted(teS - leak_train - leak_val)
    print(f"\n[口径] 官方测试 {len(teS)} − 训练泄漏 {len(teS & leak_train)} = **clean {len(clean_ids)}**"
          f" ； 再 − 验证泄漏 {len(teS & leak_val)} = **strict {len(strict_ids)}**", flush=True)
    eval_ids = strict_ids if STRICT_ONLY else clean_ids
    if CLEAN_N > 0:
        eval_ids = eval_ids[:CLEAN_N]
        print(f"  （REEVAL_N/SMOKE → 只取前 {len(eval_ids)} 个·**不是正式数**）", flush=True)

    # ---- 下载 + 载入 clean 池（保 id ↔ 池序对齐） ----
    R._download(eval_ids)
    kept = [(i, f"{R._SDIR}/T-{i}.xml") for i in eval_ids
            if os.path.exists(f"{R._SDIR}/T-{i}.xml") and os.path.getsize(f"{R._SDIR}/T-{i}.xml") > 1000]
    if len(kept) < 0.95 * len(eval_ids):
        raise SystemExit(f"🔒 clean 集只拿到 {len(kept)}/{len(eval_ids)}（<95%）→ 分母静默缩水会污染结论，中止（查网络/STEP4E_SDIR）。")
    pool_ids = [i for i, _ in kept]
    pool = load_scenario_pool([p for _, p in kept])
    print(f"[池] clean 池 N={len(pool)}（请求 {len(eval_ids)}）·二项 SE@p=0.8 ≈ "
          f"{100 * math.sqrt(0.8 * 0.2 / max(len(pool), 1)):.2f}pt", flush=True)

    # ---- 会遇类型（复用 classify_scenarios 的判据·不另写规则） ----
    types = {}
    try:
        from classify_scenarios import classify as _classify
        for idx, (sc_obj, pp_obj) in enumerate(pool):
            init = pp_obj.initial_state
            ego_p = np.asarray(init.position, dtype=float)
            ego_psi = float(init.orientation)
            ego_v = float(getattr(init, "velocity", 5.0))
            try:
                gc = np.asarray(getattr(pp_obj.goal.state_list[0].position, "center", None), dtype=float)
            except Exception:                                  # noqa: BLE001
                gc = None
            obs = sc_obj.dynamic_obstacles
            if not obs:
                types[idx] = "no-obstacle"
                continue
            o0 = obs[0].initial_state
            t, _b, _d = _classify(ego_p, ego_psi, ego_v, gc,
                                  np.asarray(o0.position, dtype=float),
                                  float(o0.orientation), float(getattr(o0, "velocity", 5.0)))
            types[idx] = t
    except Exception as e:                                     # noqa: BLE001 —— 分型失败不该拖垮主指标
        print(f"  ⚠️ 会遇类型分类失败（{e}）→ 只报总体、不报分型", flush=True)
        types = {}
    if types:
        from collections import Counter
        print(f"[分型] clean 池会遇类型：{dict(Counter(types.values()))}", flush=True)

    strictS = set(strict_ids)

    def agg_of(per, idx_filter=None):
        """从逐局明细算聚合（口径与 evaluate/evaluate_continuous 完全一致：都是 per 上取均值）。
        idx_filter=None → 全部；否则只取 scenario_idx ∈ 该集合的局（用于 577 里切出 563 子集）。"""
        rows = [p for p in per if idx_filter is None or p.get("scenario_idx") in idx_filter]
        n = len(rows)
        if n == 0:
            return None
        arr = 100.0 * sum(bool(p["reached"]) for p in rows) / n
        out = {"n": n, "到达率%": arr,
               "二项SE": 100 * math.sqrt(max(arr / 100 * (1 - arr / 100), 0.0) / n),
               "碰撞率%": 100.0 * sum(bool(p["collided"]) for p in rows) / n,
               "违规次数/局": sum(p.get("violations", 0) for p in rows) / n,
               "紧急步%": sum(p.get("emergency_pct", 0.0) for p in rows) / n}
        # 位置-only 上界代理（**仅连续臂有**·`03` L186 caveat：进过框≠到达·含"满速绕圈扫框"平凡解
        # → 它是【宽松上界】，绝不当主指标，只用来看"我们的严格朝向门吃掉了多少个点"）。
        ib = [p.get("in_box_steps") for p in rows if p.get("in_box_steps") is not None]
        if ib:
            out["位置进框%(宽松上界)"] = 100.0 * sum(1 for v in ib if v > 0) / len(ib)
        return out

    def per_type(per):
        """按会遇类型分组的到达率（scenario_idx → 池序 → 类型）。"""
        if not types:
            return {}
        groups = {}
        for p in per:
            groups.setdefault(types.get(p.get("scenario_idx"), "unknown"), []).append(p)
        return {t: {"n": len(v), "到达率%": 100.0 * sum(bool(x["reached"]) for x in v) / len(v),
                    "紧急步%": sum(x.get("emergency_pct", 0.0) for x in v) / len(v)}
                for t, v in sorted(groups.items())}

    # ---- 逐 checkpoint：锚点复现检查 → clean 集正式评 ----
    results = {}
    for b in ckpts:
        sc = sidecars[b]
        name = os.path.basename(b)
        if sc is None:
            raise SystemExit(f"🔒 {name} 没有 .progress.json → 判不出臂类型(kind)/训练数据集/锚点值，中止（sidecar 是 provenance 命门）。\n"
                             "   老 ckpt 确无 sidecar 时可显式降级：REEVAL_FORCE_KIND=continuous|shielded|unshielded "
                             "+ REEVAL_FORCE_DATASET=manifest_hocr_200.json（**此时锚点检查做不了·数字的可信度自行承担**）。")
        if sc.get("_forced"):
            print(f"  ⚠️⚠️ {name} 无 sidecar → 用人工指定 kind={sc['kind']} dataset={FORCE_DATASET}；"
                  "**锚点复现检查无法进行** = eval 配置是否与训练一致【未经验证】。", flush=True)
        kind, weight = sc.get("kind"), sc.get("colregs_weight", 0.0)
        if kind not in ("continuous", "shielded", "unshielded"):
            raise SystemExit(f"🔒 {name} 的 kind={kind!r} 不认识（须 ∈ continuous/shielded/unshielded），中止。")
        algo = ((sc.get("config_sig") or {}).get("continuous_algo")) if kind == "continuous" else None
        print("\n" + "─" * 100)
        print(f"▶ {name}  臂={sc.get('party')} kind={kind} seed={sc.get('seed')} 步={sc.get('num_timesteps')}", flush=True)

        # ① 锚点复现：在它【自己记录的测试集】上重评 → 必须复现 trend[-1]
        anchor = None
        if ANCHOR:
            ds = ((sc.get("config_sig") or {}).get("dataset"))
            mp = find_manifest(ds)
            trend = sc.get("trend") or []
            want = (trend[-1] or {}).get("到达率%") if trend else None
            if mp is None or want is None:
                print(f"  ⚠️ 锚点检查跳过（数据集={ds!r} manifest={'找到' if mp else '没找到'}·trend={'有' if trend else '无'}）"
                      f" → **本 ckpt 的 eval 配置未经验证**，数字请谨慎采信。", flush=True)
            else:
                _tr_p, _te_p, _info = R.load_manifest_split(mp, os.path.dirname(mp))
                a_pool = load_scenario_pool(_te_p)
                a_agg, _a_per = R.replay_eval(b, kind, weight, a_pool, continuous_algo=algo, return_per=True)
                got = a_agg["到达率%"]
                d = abs(got - want)
                one_ep = 100.0 / max(len(a_pool), 1)           # 一局翻转的分量：配置错会差【很多】，一局是浮点噪声
                tol_eff = max(ANCHOR_TOL, one_ep)
                anchor = {"n": len(a_pool), "记录值": want, "重评值": got, "差": d,
                          "容差": tol_eff, "差几局": round(d / one_ep, 2), "通过": d <= tol_eff}
                tag = ("✅ 逐位复现" if d == 0.0 else
                       f"🟡 差 {d / one_ep:.0f} 局（容差内·浮点噪声）" if d <= tol_eff else "❌ 对不上")
                print(f"  [锚点] 自记测试集 N={len(a_pool)}：记录 {want:.2f}% vs 重评 {got:.2f}%（差 {d:.2f}pt）{tag}", flush=True)
                if d > tol_eff:
                    msg = (f"🔒 锚点复现失败（{name}）：差 {d:.2f}pt = {d / one_ep:.1f} 局 > 容差 {tol_eff:.2f}pt。\n"
                           "   含义：本次 eval 的环境配置与训练时【不一致】（连续臂 shield/cone/augment_rho/去朝向门这些"
                           "不进 config_sig 的 knob 最可能），此时官方测试集上的数字【不可信】。\n"
                           "   先查这些 STEP4E_* 变量；确认容差合理再用 REEVAL_ANCHOR_TOL / REEVAL_ANCHOR_SOFT=1 放行。")
                    if ANCHOR_SOFT:
                        print("  ⚠️ " + msg, flush=True)
                    else:
                        raise SystemExit(msg)

        # ② clean 集正式评
        agg, per = R.replay_eval(b, kind, weight, pool, continuous_algo=algo, return_per=True)
        pid = {i: pool_ids[i] for i in range(len(pool_ids))}    # 池序 → T-id
        strict_idx = {i for i, t in pid.items() if t in strictS}
        a_clean = agg_of(per)
        a_strict = agg_of(per, strict_idx)
        results[name] = {"kind": kind, "party": sc.get("party"), "seed": sc.get("seed"),
                         "num_timesteps": sc.get("num_timesteps"),
                         "dataset": ((sc.get("config_sig") or {}).get("dataset")),
                         "anchor": anchor, "clean": a_clean, "strict": a_strict,
                         "clean_per_type": per_type(per)}
        _c, _s = a_clean, a_strict
        print(f"  [clean  N={_c['n']:>3}] 到达 {_c['到达率%']:5.2f}% ±{_c['二项SE']:.2f}  "
              f"碰撞 {_c['碰撞率%']:.2f}%  违规/局 {_c['违规次数/局']:.2f}  紧急步 {_c['紧急步%']:.1f}%"
              + (f"  位置进框(宽松) {_c['位置进框%(宽松上界)']:.1f}%" if "位置进框%(宽松上界)" in _c else ""), flush=True)
        if _s:
            print(f"  [strict N={_s['n']:>3}] 到达 {_s['到达率%']:5.2f}% ±{_s['二项SE']:.2f}  "
                  f"碰撞 {_s['碰撞率%']:.2f}%  违规/局 {_s['违规次数/局']:.2f}  紧急步 {_s['紧急步%']:.1f}%", flush=True)
        for t, m in (results[name]["clean_per_type"] or {}).items():
            print(f"     · {t:<24} n={m['n']:>3}  到达 {m['到达率%']:5.1f}%  紧急步 {m['紧急步%']:.1f}%", flush=True)

    # ---- 落盘 ----
    payload = {"口径": {"官方测试": len(teS), "clean": len(clean_ids), "strict": len(strict_ids),
                        "实评": len(pool), "训练泄漏": len(teS & leak_train), "验证泄漏": len(teS & leak_val),
                        "smoke": SMOKE, "REEVAL_N": CLEAN_N},
               "官方划分": {"n_total": OFF_N_TOTAL, "test_frac": OFF_TEST_FRAC, "split_seed": OFF_SPLIT_SEED,
                            "pool": OFF_POOL},
               "会遇类型计数": ({} if not types else {t: sum(1 for v in types.values() if v == t) for t in set(types.values())}),
               "实评T_id": pool_ids, "strict_T_id": strict_ids, "clean_T_id": clean_ids,   # 口径可审计（论文写口径直接引这里）
               "结果": results}
    if os.path.dirname(OUT):
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)
    print(f"\n[reeval_official] 完成 → {OUT}", flush=True)
    if SMOKE or CLEAN_N:
        print("⚠️ 这是【冒烟/截断】跑，不是正式数——正式跑请清掉 REEVAL_SMOKE / REEVAL_N。", flush=True)


if __name__ == "__main__":
    main()
