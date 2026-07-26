#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E5 / E6 泛化测——eval-only·零训练·不烧训练卡（`03` L207-E / L208 / L209 ·`02` later-4「下一步①」）。

═══ 这个脚本回答什么 ═══
  我们的两条臂（连续 Continuous-safe / 离散 Discrete-safe）都是在 **94 个训练场景**（`manifest_hocr_200`）上训的；
  Krasowski & Althoff 2024 Table III 的 Discrete-safe **86.3%** 是在 **官方 1400 训 / 600 测** 上跑的。
  本脚本 = **不重训**，把【现有 checkpoint】搬到【没训练过的场景】上评 ⟹ 判"我们的数低是【训练集太小】还是
  【复现本身有问题】"。**连续臂已由 `03` L209 答完（大集 360 真没见过 = 71.9%）；现在缺的是【离散臂 E6】。**

═══ 🔴 泄漏必剔（本脚本的核心）═══
  同一趟 eval 同时给三个口径（切子集·零额外成本）：
    · **全部**  = 池里所有场景（含被看过的·用来量"看过 vs 没看过"差多少）
    · **clean** = 全部 − 该 checkpoint 的【训练】场景
    · **strict**= clean − 该 checkpoint 的【验证】场景（训练期每段反复评过·源种子是看着它挑的 ⟹ 不算干净·`03` L192）
  官方池实测：600 − 23 训练泄漏 = **577**；再 − 14 验证泄漏 = **563**（`03` L208 亲算）。

═══ 两种场景池（`REEVAL_POOL`）═══
  · `official`（默认）= 官方 1400/600 划分的那 600 个（= `run_step4e` 不设 manifest 时的默认口径·**官方 2000 无追越**）
  · `manifest:manifest.json` = 均衡大集的 600 个测试场景（对遇200+交叉200+**追越**200）
      ← **E6 要用这个**：`03` L209 的连续臂数就是在它上面出的，离散臂用同一个池才能直接对比。

═══ 三层防误判 ═══
  ① **锚点复现检查**（默认开）：先在 checkpoint **自己记录的测试集**上重评，与 `.progress.json` 的
     `trend[-1]["到达率%"]` 对比，差超过一局就**中止**——连续臂的 shield/cone/augment_rho/去朝向门这些 knob
     **不进 config_sig**、配错了会【静默】改数字（`03` L192 那条洞）。
  ② **数据集不可还原**（`dataset=strided`/缺）→ 中止（泄漏剔不干净）。
  ③ **下载缺额 <95%** → 中止（防分母静默缩水）。

═══ 用法（**在 ~/trb 根目录下跑**·随 代码/ 同步上服务器·不以 test_ 开头故不进回归套件）═══
  0) 纯逻辑自检（**本机就能跑**·无需 commonocean/vesselmodels）：
       REEVAL_SELFTEST=1 python 代码/tests/reeval_official.py
  1) 先列 checkpoint（零风险·只读 sidecar·几秒）：  REEVAL_LIST=1 …
  2) 冒烟（30 场景·单 ckpt）：                      REEVAL_SMOKE=1 REEVAL_CKPTS=… …
  3) 全量：                                         REEVAL_CKPTS=… REEVAL_OUT=… …
  完整逐字命令（含 screen）见 `04_运行手册.md §2` 最上面那节。

【环境变量】
  STEP4E_CODE_DIR   代码目录（服务器设 `代码`；默认 '.'）
  STEP4E_SDIR       T-*.xml 场景缓存目录（缺的自动下·已缓存跳过）
  REEVAL_POOL       `official`（默认）/ `manifest:<manifest 文件名>`
  REEVAL_CKPTS      要评的 checkpoint base 名（逗号分隔·不带 .zip）；空 = 在 CKDIRS 里自动发现
  REEVAL_CKDIRS     checkpoint 目录（冒号分隔·**支持通配符**如 `/root/trb/*/checkpoints`）
  REEVAL_SEEDS      只评名字里含这些种子标记的 ckpt（可选·逗号分隔·如 `s0,s1`）
  REEVAL_ANCHOR     1(默认)=做锚点复现检查 / 0=跳过（**别随便关**）
  REEVAL_ANCHOR_TOL 锚点容差（百分点·默认 0.0）。实际生效 = max(该值, 一局的分量 100/N)：
                    配置错会差【很多】，差一局是浮点噪声——一局内只警告，超过就中止。
  REEVAL_ANCHOR_SOFT 1 = 锚点对不上只警告不中止（默认 0）
  REEVAL_MANIFEST_DIRS  找 manifest / OT 文件的目录（冒号分隔）
  REEVAL_N          池上限（默认 0=全部）；调试用    REEVAL_SMOKE 1 → N=30 且只评第一个 ckpt
  REEVAL_OUT        结果 json（默认 reeval_official.json·写 cwd）
  REEVAL_ENVCFG_ACK 1 = 承认「连续臂 eval env knob 非金标默认」（默认非默认即中止）
  REEVAL_ALLOW_UNKNOWN_DATASET 1 = 允许评「训练集无法从 sidecar 还原」的 ckpt（默认中止）
  REEVAL_FORCE_KIND / REEVAL_FORCE_DATASET  老 ckpt 无 .progress.json 时的显式降级（此时【做不了锚点检查】）

⚠️ 本脚本【不训练、不写 checkpoint、不改任何模型字节】。到达率不进梯度 → 重评纯测量。
"""
from __future__ import annotations
import glob
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict

# 🔴 脚本版本号：**每次改动必手动 +1**。服务器同步后用 `grep SCRIPT_REV <文件>` 一眼验是不是最新
#    （靠"我几点同步的"判断不可靠——已踩过）。跑起来时也会打印，log 里永久留痕。
SCRIPT_REV = "r10-2026-07-26"  # r10: REEVAL_YAW_SLEW —— 舵速率限制（第二个平滑旋钮·有物理依据·`03` L221）
#                                r9: REEVAL_YAW_LOWPASS —— 转向低通滤波权衡曲线（治抖·纯评估期·`03` L218·user 拍）
#                                r8: REEVAL_TRAJ_KEYS —— 采几个场景的逐步轨迹（多算法轨迹对比图·`03` L215-G·user 拍）
#                                r7: 会遇类型分型提成 `classify_pool`（外部基线 runner 共用·分型判据只准有一处）
#                                r6: 违规拆分(让路/直航)进聚合——r5 把它排除了·总数输了却分不出病灶
#                                ⚠️ r7 是**纯提取**（同一段代码搬进函数·调用点等价）· r8 **不设 TRAJ_KEYS 时一行都不动**
#                                   ⟹ 两者数字都与 r6 逐位相同

# ---------- 开关 ----------
_CODE = os.environ.get("STEP4E_CODE_DIR", ".")
SELFTEST = os.environ.get("REEVAL_SELFTEST", "0") == "1"
LIST_ONLY = os.environ.get("REEVAL_LIST", "0") == "1"
SMOKE = os.environ.get("REEVAL_SMOKE", "0") == "1"
ANCHOR = os.environ.get("REEVAL_ANCHOR", "1") == "1"
ANCHOR_TOL = float(os.environ.get("REEVAL_ANCHOR_TOL", "0.0"))
ANCHOR_SOFT = os.environ.get("REEVAL_ANCHOR_SOFT", "0") == "1"
POOL_SPEC = os.environ.get("REEVAL_POOL", "official").strip()
CLEAN_N = int(os.environ.get("REEVAL_N", "0"))
OUT = os.environ.get("REEVAL_OUT", "reeval_official.json")
ENVCFG_ACK = os.environ.get("REEVAL_ENVCFG_ACK", "0") == "1"
FORCE_KIND = os.environ.get("REEVAL_FORCE_KIND", "").strip()
FORCE_DATASET = os.environ.get("REEVAL_FORCE_DATASET", "").strip()
# 🆕 r8（`03` L215-G）：要采逐步轨迹的**池键**（T-id·空格或逗号分隔）→ 多算法轨迹对比图用。
#    不设 = 一行记录块都不执行 = 与 r6/r7 逐位相同。轨迹另落一个文件，**不塞进主 json**（免把它撑大）。
TRAJ_KEYS = [s.strip() for s in os.environ.get("REEVAL_TRAJ_KEYS", "").replace(",", " ").split() if s.strip()]
TRAJ_OUT = os.environ.get("REEVAL_TRAJ_OUT", "").strip()
# 🆕 r9（`03` L218）：转向低通滤波【权衡曲线】。给一串 α（逗号/空格分隔），每个 α 在同一池上评一遍
#    ⟹ 一趟就出"转艏平顺度 vs 到达率"的曲线。**α=1.0 = 不滤波 = 对照组**（建议总把 1.0 写进去，同趟拿对照）。
#    不设 = 完全不进这条路 = 与 r6/r7/r8 逐位相同。**纯评估期·零训练算力·安全关键文件一行不改。**
YAW_LOWPASS = [float(x) for x in os.environ.get("REEVAL_YAW_LOWPASS", "").replace(",", " ").split() if x.strip()]
# 🆕 r10（`03` L221）：**舵速率限制**——每步转向指令的变化不许超过「动作箱 × 该系数」。
#    给一串系数（1.0 = 允许从满左直接翻到满右 = 不限 = 现状；0.5 = 每步最多动半个箱）。
#    比低通更有物理依据：**真船的舵机本来就有最大转舵速率**，而本仿真允许 ω 一步从 +0.018 翻到 −0.018
#    （相当于舵机瞬间从满左打到满右）⟹ 加这个限制不是为了刷指标，是在补一条本来缺失的物理约束。
YAW_SLEW = [float(x) for x in os.environ.get("REEVAL_YAW_SLEW", "").replace(",", " ").split() if x.strip()]
if SMOKE:
    CLEAN_N = 30

_CKDIRS_RAW = os.environ.get(
    "REEVAL_CKDIRS",
    "结果/checkpoints:checkpoints:*/checkpoints:结果/*/checkpoints").split(":")   # server 结果目录直挂 ~/trb 下
CKDIRS = []
for _d in _CKDIRS_RAW:                                     # 支持通配符（服务器结果目录名带日期·别逐个硬写）
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

    🔴 真跑时【一定】会与 `R.make_split` 硬比对（见 `official_split`），不一致直接中止 → 镜像漂移不可能悄悄污染结果。
    别在这里"顺手改进"，它的唯一职责是复刻。
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
        tr, te = rtr, rte
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


def key_of_path(p):
    """场景文件 → 池内唯一键：`T-123.xml` → int 123（官方池）· `OT-7.xml` → 'OT-7.xml'（本项目自造追越）。"""
    b = os.path.basename(p)
    if b.startswith("T-") and b.endswith(".xml"):
        try:
            return int(b[2:-4])
        except ValueError:
            return b
    return b


def manifest_keys(manifest_path):
    """读 manifest → (训练键集合, 测试键集合, 键→类型)。键 = T-id(int) 或 OT 文件名(str)。

    ⚠️ 追越那 667 个 `OT-*.xml` 是**本项目自造**的、**不在官方 2000 池里**（`03` L207-A）⟹ 与官方测试集天然无交集；
    但若某 checkpoint 是【训练时带追越】的（manifest_200 / manifest.json），它的 OT 训练文件同样要算进泄漏集。
    """
    with open(manifest_path, encoding="utf-8") as fh:
        man = json.load(fh)
    tr, te, typ = set(), set(), {}
    for zh, en in (("对遇", "head_on"), ("交叉", "crossing"), ("追越", "overtaking")):
        for split, bag in (("train", tr), ("test", te)):
            for x in man[en][split]:
                k = int(x) if en != "overtaking" else os.path.basename(str(x))
                bag.add(k)
                typ[k] = zh
    return tr, te, typ


def manifest_tids(manifest_path):
    """（兼容旧接口·测试在用）→ (训练 T-id 集合, 测试 T-id 集合)，只含 head_on/crossing 的整数 T-id。"""
    tr, te, _ = manifest_keys(manifest_path)
    return {k for k in tr if isinstance(k, int)}, {k for k in te if isinstance(k, int)}


def selftest():
    """纯逻辑自检（本机可跑）：官方划分 + 泄漏剔除 + 计数断言。"""
    print(f"【SELFTEST】{SCRIPT_REV} · 官方划分 + 泄漏剔除（纯逻辑·不碰模型/场景文件）", flush=True)
    tr, te = official_split(None)
    assert len(tr) == 1400 and len(te) == 600, (len(tr), len(te))
    assert not (set(tr) & set(te)), "官方 train/test 重叠"
    print(f"  ✅ 官方划分 1400/600（测试前 5：{te[:5]}）")
    assert key_of_path("/x/T-42.xml") == 42 and key_of_path("/x/OT-7.xml") == "OT-7.xml"
    mp = find_manifest("manifest_hocr_200.json")
    if mp is None:
        print(f"  ⚠️ 没找到 manifest_hocr_200.json（找过 {MANIFEST_DIRS}）→ 跳过泄漏断言")
        return
    mtr, mte = manifest_tids(mp)
    assert len(mtr) == 94, f"manifest_hocr_200 训练应 94 个 T-id，实得 {len(mtr)}"
    assert len(mte) == 40, f"manifest_hocr_200 测试应 40 个 T-id，实得 {len(mte)}"
    teS = set(te)
    assert len(teS & mtr) == 23, f"训练泄漏应 23，实得 {len(teS & mtr)}"
    assert len(teS & mte) == 14, f"验证泄漏应 14，实得 {len(teS & mte)}"
    assert len(teS - mtr) == EXPECT_CLEAN_HOCR and len(teS - mtr - mte) == EXPECT_STRICT_HOCR
    print("  ✅ 泄漏：训练 23/94（24.5%）· 验证 14/40 → clean 577 / strict 563")
    big = find_manifest("manifest.json")                       # `03` L209 用的均衡大集（E6 要用同一个池）
    if big:
        btr, bte, _ = manifest_keys(big)
        assert len(bte) == 600, f"大集测试应 600，实得 {len(bte)}"
        assert not (mtr & bte), "金标 94 训练竟落在大集测试里（应为 0）"
        print(f"  ✅ 大集 manifest.json：测试 600（金标 94 训练 ∩ 它 = 0 · 金标 40 验证 ∩ 它 = {len(mte & bte)}）")
    print("【SELFTEST】全部通过 ✅", flush=True)


# ---------------- checkpoint 发现 / sidecar ----------------
def read_sidecar(base):
    """读 `<base>.progress.json` → dict（缺/坏 → 用 REEVAL_FORCE_* 拼最小 sidecar，否则 None）。"""
    pj = base + ".progress.json"
    if not os.path.exists(pj):
        if FORCE_KIND and FORCE_DATASET:                       # 显式降级：kind/dataset 人工给·但【没有 trend】=锚点做不了
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


def _sidecar_in_sync(base, sc):
    """progress.json 写入时记了当时 .zip 的 mtime+size（`ckpt_fingerprint`）→ 与现在的 zip 比对。

    对不上 = 模型文件比进度记录【新】（run 被杀在"存 zip"与"写 progress.json 提交点"之间·或事后覆盖）
    ⟹ `trend[-1]` 描述的不是这个模型 ⟹ 锚点比对无意义。缺指纹（老 ckpt）→ 视为同步（无从判断·不冤枉）。
    """
    fp = (sc or {}).get("ckpt_fingerprint")
    if not fp or "zip_size" not in fp:
        return True
    z = base + ".zip"
    if not os.path.exists(z):
        return True
    stt = os.stat(z)
    if int(stt.st_size) != int(fp["zip_size"]):
        return False
    return abs(float(stt.st_mtime) - float(fp.get("zip_mtime", stt.st_mtime))) <= 1.0   # 1s 容差（拷贝/文件系统精度）


def discover_ckpts():
    """在 CKDIRS 里找所有 `*.zip` 且有同名 `_vecnorm.pkl` 的 checkpoint → [base, ...]（按 basename 去重·保序）。"""
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
    """REEVAL_CKPTS 指定 → 在 CKDIRS 里按名解析；否则自动发现全部。REEVAL_SEEDS 可再按名字过滤。"""
    spec = [s.strip() for s in os.environ.get("REEVAL_CKPTS", "").split(",") if s.strip()]
    if spec:
        out = []
        for name in spec:
            if os.path.exists(name + ".zip"):                  # 允许直接给路径
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
    else:
        out = discover_ckpts()
    seeds = [s.strip() for s in os.environ.get("REEVAL_SEEDS", "").split(",") if s.strip()]
    if seeds:
        out = [b for b in out if any(f"_{s}_" in os.path.basename(b) or os.path.basename(b).endswith(f"_{s}")
                                     for s in seeds)]
    names = [os.path.basename(x) for x in out]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:                                                    # 结果按 basename 存 → 重名会互相覆盖、静默丢一条
        raise SystemExit(f"🔒 有重名 checkpoint {dup} → 结果会互相覆盖，中止（去重后重跑）。")
    return out


def describe(base, sc):
    """一行 checkpoint 摘要。"""
    if sc is None:
        return f"{os.path.basename(base):<52} （无 sidecar·无法判臂/数据集）"
    cfg = sc.get("config_sig") or {}
    trend = sc.get("trend") or []
    last = (trend[-1] or {}) if trend else {}
    return (f"{os.path.basename(base):<52} 臂={str(sc.get('party')):<16} kind={str(sc.get('kind')):<11} "
            f"seed={sc.get('seed')} 步={sc.get('num_timesteps')} 数据集={cfg.get('dataset')} "
            f"末段到达={last.get('到达率%')}")


# ---------------- 聚合 ----------------
def _emergency_pct(p):
    return float(p.get("emergency_pct") or 0.0)


# 逐局里【不参与"求均值"】的键：身份/标志/嵌套结构 + 已在头条行单独报的。其余**数值键一律自动聚合**
# —— 🔴 用"自动发现"而不是白名单，是因为本项目已经两次栽在"新指标接进 evaluate 了、但下游忘了取"
#    （`03` L203 接指标那次的 commit 原话："否则四方头条拿不到=跑完要返工"；本脚本 2026-07-26 又犯一次）。
#    自动发现 ⟹ 以后 evaluate 再加指标，这里【自动就带上】，不会再漏。
_AGG_SKIP = {"scenario_idx", "reached", "collided", "traj", "term_flags", "end_state", "goal_geom",
             "ep_src", "scenario_type", "scenario_file", "violations", "emergency_pct"}
# 🔴 `giveway_violations` / `standon_violations` **必须留在聚合里**（2026-07-26 修）：
#   头条行只给"违规/局"总数 ⟹ 一旦总数输了，**分不出是"让路方向违规"还是"直航保向违规"**，
#   而这两者的含义天差地别——我们的盾**只约束让路步的方向**，**直航保向根本不在盾的作用域**
#   （CLAUDE.md §0 红线口径）。缺这个拆分 ⟹ 诊断不下去、论文也没法把作用域讲清楚。


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _mean_extras(rows):
    """把逐局里所有【数值型】指标自动求均值（跳 None·空则不出该键）——平滑度/次网格/按态势拆转艏/进近诊断全在里面。"""
    keys = set()
    for p in rows:
        keys |= {k for k, v in p.items() if k not in _AGG_SKIP and _is_num(v)}
    out = {}
    for k in sorted(keys):
        vals = [p[k] for p in rows if _is_num(p.get(k))]
        if vals:
            out[k] = round(sum(vals) / len(vals), 6)
    return out


def agg_of(per, idx_filter=None):
    """从逐局明细算聚合（口径与 evaluate/evaluate_continuous 一致：per 上取均值）。"""
    rows = [p for p in per if idx_filter is None or p.get("scenario_idx") in idx_filter]
    n = len(rows)
    if n == 0:
        return None
    arr = 100.0 * sum(bool(p["reached"]) for p in rows) / n
    out = {"n": n, "到达率%": arr,
           "二项SE": 100 * math.sqrt(max(arr / 100 * (1 - arr / 100), 0.0) / n),
           "碰撞率%": 100.0 * sum(bool(p["collided"]) for p in rows) / n,
           "违规次数/局": sum(p.get("violations", 0) for p in rows) / n,
           "紧急步%": sum(_emergency_pct(p) for p in rows) / n}
    # 位置-only 上界代理（**仅连续臂有**·`03` L186 caveat：进过框≠到达、含"满速绕圈扫框"平凡解
    # ⟹ **绝不当主指标**，只用来看"严格朝向门吃掉了多少个点"）。
    ib = [p.get("in_box_steps") for p in rows if p.get("in_box_steps") is not None]
    if ib:
        out["位置进框%(宽松上界)"] = 100.0 * sum(1 for v in ib if v > 0) / len(ib)
    # 🔴 全套指标（平滑度 jerk/转艏增量/油门增量 + **次网格细调率** + **按态势拆转艏** + 路径长/进近诊断）
    #    `03` L203 的两个卖点指标就在这里：subgrid_*_frac（用掉多少连续分辨率·离散臂 by construction 恒 0）
    #    与 yaw_incr_giveway/other（转艏活动是否集中在让路步 = 合规代价而非控制毛病）。
    out["控制质量"] = _mean_extras(rows)
    # 路径长度/平滑度须【分到达与否】看（游荡局天然更长·`evaluate._control_quality` docstring 红队 MEDIUM L72）
    arrived_rows = [p for p in rows if p.get("reached")]
    if arrived_rows:
        out["控制质量_仅到达局"] = _mean_extras(arrived_rows)
    return out


class YawLowPassPolicy:
    """转向指令**低通滤波**包装器（`03` L218·纯评估期·零训练算力）。

    ═══ 治什么 ═══
    我们唯一输给离散臂的指标是**转艏增量**（0.0150 vs 0.0122 = 输 1.23×）。机理已查实（`03` L203）：
    船转极慢（ω_max=1.7°/s），想让路或对准目标就得几乎打满舵 ⟹ 每步平均动掉 83.6% 的动作箱。

    ═══ 怎么做（一行公式）═══
        ω_filt[k] = (1−α)·ω_filt[k−1] + α·ω_raw[k]     · α ∈ (0, 1]
    α=1 ⟹ `ω_filt ≡ ω_raw` ⟹ **与不加滤波【逐位相同】**（回归测试钉死这条）。α 越小越平滑、跟随越慢。

    ═══ 🔴 为什么安全保证一字不动 ═══
    本包装器只改 **u_desired**（策略想做的动作），**盾仍在最后**：
        观测 → 策略 → **本滤波器** → 状态机 + zonotope 安全集 + 二次规划投影 → 执行
    ⟹ 盾的输入变了、盾本身没变 ⟹ 「每让路步方向合规 + 远场单步无碰」这些命题**按构造仍然成立**
       （它们是对**任意**输入动作的性质，不依赖输入怎么来）。
    ⟹ 且 `usv_continuous_shield.py` / `usv_projection.py` **一行都没改**（`03` L218 刻意的架构选择）。
    ⚠️ **但"保证不变"≠"数字不变"**：滤波会改变轨迹 ⟹ 到达率/碰撞率都可能动。
       **判读硬红线：碰撞率不许上升。** 若上升 ⟹ 说明滤波把船拖进了盾也救不回的几何 ⟹ 该 α 直接判死。

    ═══ 逐局状态必须清 ═══
    滤波器有记忆（ω_filt[k−1]）⟹ **跨局不清就会把上一局的舵角带进下一局**。
    `evaluate.run_episode_continuous` 每局 reset 后会调 `bind_env` ⟹ 用它当局边界清状态
    （`03` L215-A 的教训：这个钩子名字叫 `bind_env` 不叫 `bind`，写错了官方评估器根本不会调）。
    """

    def __init__(self, model, alpha):
        a = float(alpha)
        if not (0.0 < a <= 1.0):
            raise ValueError(f"低通系数 α 须 ∈ (0,1]，得到 {a}（α=1 表示不滤波）")
        import numpy                                       # 本模块刻意不在顶层 import numpy（纯逻辑自检不需要重依赖）
        self._np = numpy
        self.model, self.alpha = model, a
        self._prev = None
        self.n_steps = 0

    def bind_env(self, env):
        """每局开头清滤波器记忆（否则上一局的舵角会漏进这一局）。"""
        self._prev = None
        inner = getattr(self.model, "bind_env", None)      # 内层若也有这个钩子（如朴素基线）→ 一并转发
        if callable(inner):
            inner(env)
        return env

    def predict(self, obs, deterministic=True, **kw):
        u, state = self.model.predict(obs, deterministic=deterministic, **kw)
        arr = self._np.asarray(u, dtype=float).copy()
        if arr.shape[-1] != 2:                            # 只认 (a, ω)；形状不对宁可原样放行也不猜
            return u, state
        w_raw = float(arr[..., 1])
        prev = self._prev
        w = w_raw if prev is None else (1.0 - self.alpha) * prev + self.alpha * w_raw
        # 凸组合守卫：α∈(0,1] ⟹ w 必落在 prev 与 w_raw 之间 ⟹ 两个箱内值之间也在箱内 ⟹ 不需额外夹取。
        # 断言在这里，是为了将来谁改公式（比如改成带增益的滤波）会当场炸，而不是静默越箱。
        if prev is not None:
            assert min(prev, w_raw) - 1e-12 <= w <= max(prev, w_raw) + 1e-12, (prev, w_raw, w, self.alpha)
        self._prev = w
        arr[..., 1] = w
        self.n_steps += 1
        return arr, state

    def __getattr__(self, name):                          # 其余属性透传给内层模型（policy/observation_space…）
        return getattr(self.model, name)


class YawSlewLimitPolicy:
    """转向指令**速率限制**包装器（`03` L221·纯评估期·安全关键文件一行不改）。

    ═══ 一行公式 ═══
        ω_out[k] = clip(ω_raw[k], ω_out[k−1] ± Δmax)   ·   Δmax = frac × W_BOX
    `frac=1.0` ⟹ 允许一步从满左翻到满右（= 现状）⟹ **与不加限制逐位相同**（回归钉死）。

    ═══ 🔴 为什么这个比低通更站得住 ═══
    **真船的舵机有最大转舵速率**（几度每秒），不可能一步从满左打到满右。而本仿真的动作空间
    只限 |ω| ≤ ω_max、**不限 |Δω|** ⟹ 策略学出的「满左 / 满右 交替」在真实舵机上物理上做不到。
    ⟹ 加这条限制 = **补一条本来缺失的执行器物理约束**，不是为了把某个指标做好看。
    ⚠️ **但仍必须诚实**：本基准（Krasowski）的动作空间确实允许瞬时翻转，我们**只对自己这条臂加限制**
       ⟹ 这是**自我设限**（只可能让我们其它指标变差或不变），写作时要写明"我们的控制器额外施加了舵速率限制"。
    ⚠️ 且：`|Δω|` 正是「转艏增量」这个指标本身 ⟹ **这个旋钮 by construction 会改善那个指标**
       ⟹ **绝不能写成"我们在转艏上赢了离散臂"**，只能写成"存在一条可调的平顺度—到达率权衡曲线"
       （而离散臂**结构上没有这个旋钮**：它的动作是固定格点，滤过就不再是合法离散动作了）。

    ═══ 逐局状态必须清 ═══
    同低通：靠 `bind_env` 当局边界（`03` L215-A 的教训——钩子名字是 `bind_env`）。
    """

    def __init__(self, model, frac, w_box=0.018):
        f = float(frac)
        if not (0.0 < f <= 1.0):
            raise ValueError(f"速率限制系数须 ∈ (0,1]，得到 {f}（1.0 表示不限）")
        import numpy
        self._np = numpy
        self.model, self.frac = model, f
        self.dmax = f * 2.0 * float(w_box)      # 箱宽 = 2·W_BOX（从 −W_BOX 到 +W_BOX）
        self._prev = None
        self.n_steps = self.n_clipped = 0

    def bind_env(self, env):
        self._prev = None
        inner = getattr(self.model, "bind_env", None)
        if callable(inner):
            inner(env)
        return env

    def predict(self, obs, deterministic=True, **kw):
        u, state = self.model.predict(obs, deterministic=deterministic, **kw)
        arr = self._np.asarray(u, dtype=float).copy()
        if arr.shape[-1] != 2:
            return u, state
        w_raw = float(arr[..., 1])
        if self._prev is None:
            w = w_raw                            # 首步无历史 ⟹ 不限（否则等于凭空规定初始舵角）
        else:
            lo, hi = self._prev - self.dmax, self._prev + self.dmax
            w = min(max(w_raw, lo), hi)
            if w != w_raw:
                self.n_clipped += 1
        # 限速只会把值往【上一步的值】拉近 ⟹ 必落在 [min(prev,raw), max(prev,raw)] ⟹ 不可能越箱
        if self._prev is not None:
            assert min(self._prev, w_raw) - 1e-12 <= w <= max(self._prev, w_raw) + 1e-12, (self._prev, w_raw, w)
        self._prev = w
        arr[..., 1] = w
        self.n_steps += 1
        return arr, state

    def __getattr__(self, name):
        return getattr(self.model, name)


def classify_pool(pool, keys, type_of=None, np=None):
    """池 → `{池序 i: 会遇类型}`。**外部基线与我们四臂必须共用本函数**（`03` L215-D）。

    · `type_of` 非空（manifest 模式）→ 直接用 manifest 的标注（比几何分类可靠）。
    · 否则（官方池）→ 用 `classify_scenarios.classify` 的判据，**不另写规则**（复制一份 = 口径漂移的起点：
      我们的臂和外部基线若各自分型，"对遇/交叉"这两列就不可比了）。
    · 分类失败 → 返回 `{}`（只报总体、不报分型）——**分型不该拖垮主指标**。
    """
    types = {}
    if type_of:
        return {i: type_of.get(k, "unknown") for i, k in enumerate(keys)}
    if np is None:
        import numpy as np                                     # noqa: PLC0415 —— 本机自检不需 numpy 时不 import
    try:
        from classify_scenarios import classify as _classify
        for i, (sc_obj, pp_obj) in enumerate(pool):
            init = pp_obj.initial_state
            try:
                gc = np.asarray(getattr(pp_obj.goal.state_list[0].position, "center", None), dtype=float)
            except Exception:                                  # noqa: BLE001
                gc = None
            obs = sc_obj.dynamic_obstacles
            if not obs:
                types[i] = "no-obstacle"
                continue
            o0 = obs[0].initial_state
            types[i] = _classify(np.asarray(init.position, dtype=float), float(init.orientation),
                                 float(getattr(init, "velocity", 5.0)), gc,
                                 np.asarray(o0.position, dtype=float), float(o0.orientation),
                                 float(getattr(o0, "velocity", 5.0)))[0]
    except Exception as e:                                     # noqa: BLE001 —— 分型失败不该拖垮主指标
        print(f"  ⚠️ 会遇类型分类失败（{e}）→ 只报总体、不报分型", flush=True)
        return {}
    return types


def fmt(label, m):
    if not m:
        return f"  {label:<26} （空）"
    s = (f"  {label:<26} n={m['n']:>4}  到达 {m['到达率%']:5.2f}%±{m['二项SE']:.2f}  碰撞 {m['碰撞率%']:.2f}%  "
         f"违规/局 {m['违规次数/局']:5.2f}  紧急步 {m['紧急步%']:5.1f}%")
    if "位置进框%(宽松上界)" in m:
        s += f"  位置进框(宽松) {m['位置进框%(宽松上界)']:.1f}%"
    return s


def fmt_ctrl(m, label="控制质量_仅到达局"):
    """卖点指标行：平滑度 + **次网格细调率** + **按态势拆转艏**（`03` L203）。"""
    c = (m or {}).get(label) or (m or {}).get("控制质量") or {}
    if not c:
        return None
    def g(k, d=3):
        v = c.get(k)
        return "—" if v is None else f"{v:.{d}f}"
    return ("      平滑: jerk " + g("ctrl_jerk_norm_mean") + " 转艏Δ " + g("yaw_incr_mean", 4)
            + " 油门Δ " + g("accel_incr_mean", 4) + " 路径 " + g("path_len_m", 0) + "m"
            + " ｜ 次网格细调率: 转艏 " + g("subgrid_yaw_frac") + " 油门 " + g("subgrid_accel_frac")
            + " ｜ 转艏|Δω|: 让路步 " + g("yaw_incr_giveway", 4) + " 其他步 " + g("yaw_incr_other", 4))


# ---------------- 主流程 ----------------
def main():
    if SELFTEST:
        selftest()
        return

    sys.path.insert(0, _CODE)
    ckpts = resolve_ckpts()
    if not ckpts:
        raise SystemExit(f"🔒 没发现任何 checkpoint（找过 {CKDIRS}）→ 确认位置，或用 REEVAL_CKDIRS 指定。")
    sidecars = {b: read_sidecar(b) for b in ckpts}

    print("=" * 104)
    print(f"[reeval_official {SCRIPT_REV}] 发现 {len(ckpts)} 个 checkpoint：")
    for b in ckpts:
        print("  " + describe(b, sidecars[b]))
    print("=" * 104, flush=True)
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
    golden = {"STEP4E_CONTINUOUS_SHIELD": R._CONTINUOUS_SHIELD is True,
              "STEP4E_GOAL_CONE_HALF": R._GOAL_CONE_HALF_DEG is None,
              "STEP4E_AUGMENT_RHO": R._AUGMENT_RHO is False,
              "STEP4E_GOAL_IGNORE_ORIENT": R._GOAL_IGNORE_ORIENT is False}
    print(f"[env] 连续臂 eval env：shield={R._CONTINUOUS_SHIELD} cone={R._GOAL_CONE_HALF_DEG} "
          f"v_floor={R._GOAL_V_FLOOR} augment_rho={R._AUGMENT_RHO} 去朝向门={R._GOAL_IGNORE_ORIENT}", flush=True)
    bad = [k for k, ok in golden.items() if not ok]
    if bad and not ENVCFG_ACK:
        raise SystemExit(f"🔒 连续臂 eval env knob 非【金标默认】：{bad} → 与训练时不一致会静默改数字（`03` L192）。"
                         "确认是故意的就加 REEVAL_ENVCFG_ACK=1；否则清掉这些 STEP4E_* 变量重跑。")

    # ---- 每个 checkpoint 的训练/验证键（=泄漏集·取并集 → 所有臂同分母可比） ----
    leak_train, leak_val, ds_seen = set(), set(), {}
    for b in ckpts:
        sc = sidecars[b]
        ds = ((sc or {}).get("config_sig") or {}).get("dataset")
        ds_seen[os.path.basename(b)] = ds
        if ds in (None, "strided"):
            # config_sig 【不记】n_total ⟹ "strided" 分不清是 strided-200（训 140·可能与官方 600 有交集）还是
            # strided-2000（训 = 官方 1400）⟹ 泄漏剔不了 ⟹ 数不可信 ⟹ fail-closed。
            msg = (f"🔒 {os.path.basename(b)} 的数据集={ds!r}（非 manifest）→ 从 sidecar 【还原不出它的训练集】，"
                   "泄漏就剔不干净，数字不可信。\n"
                   "   要么改评 manifest 训的 checkpoint；要么确认它训练集与本池无交集后加 "
                   "REEVAL_ALLOW_UNKNOWN_DATASET=1（**并在论文口径里如实写**）。")
            if os.environ.get("REEVAL_ALLOW_UNKNOWN_DATASET", "0") == "1":
                print("  ⚠️ " + msg, flush=True)
                continue
            raise SystemExit(msg)
        mp = find_manifest(ds)
        if mp is None:
            raise SystemExit(f"🔒 {os.path.basename(b)} 记的数据集是 `{ds}`，但在 {MANIFEST_DIRS} 里找不到该 manifest → "
                             "泄漏剔不干净就等于自我污染，中止（用 REEVAL_MANIFEST_DIRS 指到 balanced_pool）。")
        t_tr, t_te, _ = manifest_keys(mp)
        leak_train |= t_tr
        leak_val |= t_te
        print(f"  [数据集] {os.path.basename(b)} ← {os.path.basename(mp)}：训练 {len(t_tr)} / 验证 {len(t_te)}", flush=True)
    if len({v for v in ds_seen.values() if v}) > 1:
        print(f"  ⚠️⚠️ 被评的 checkpoint 用了【不同数据集】{ds_seen} → 泄漏取并集剔除（clean 更小但所有臂同分母·可比）", flush=True)

    # ---- 建池（两种模式·统一用「键」记账·不靠位置对齐） ----
    type_of = {}
    if POOL_SPEC == "official":
        _, official_test = official_split(R)
        want_keys = list(official_test)
        if CLEAN_N > 0:
            want_keys = want_keys[:CLEAN_N]
        R._download(want_keys)
        paths = [f"{R._SDIR}/T-{i}.xml" for i in want_keys]
        paths = [p for p in paths if os.path.exists(p) and os.path.getsize(p) > 1000]
        pool_desc = f"官方 1400/600 划分的测试 600（官方 2000 无追越）"
    elif POOL_SPEC.startswith("manifest:"):
        mname = POOL_SPEC.split(":", 1)[1].strip()
        mp = find_manifest(mname)
        if mp is None:
            raise SystemExit(f"🔒 REEVAL_POOL 指的 manifest `{mname}` 在 {MANIFEST_DIRS} 里找不到，中止。")
        _tr_p, te_p, _info = R.load_manifest_split(mp, os.path.dirname(mp))
        paths = te_p[:CLEAN_N] if CLEAN_N > 0 else te_p
        _, _, type_of = manifest_keys(mp)                      # 类型直接用 manifest 标注（比几何分类可靠）
        want_keys = [key_of_path(p) for p in paths]
        pool_desc = f"{os.path.basename(mp)} 的测试集"
    else:
        raise SystemExit(f"🔒 REEVAL_POOL={POOL_SPEC!r} 不认识（须是 `official` 或 `manifest:<文件名>`）。")

    if len(paths) < 0.95 * len(want_keys):
        raise SystemExit(f"🔒 场景只拿到 {len(paths)}/{len(want_keys)}（<95%）→ 分母静默缩水会污染结论，中止（查网络/STEP4E_SDIR）。")
    keys = [key_of_path(p) for p in paths]                     # 池序 i ↔ keys[i]（由文件名反解·不靠位置假设）
    pool = load_scenario_pool(paths)
    print(f"\n[池] {pool_desc}：N={len(pool)}（请求 {len(want_keys)}）·"
          f"二项 SE@p=0.8 ≈ {100 * math.sqrt(0.8 * 0.2 / max(len(pool), 1)):.2f}pt", flush=True)

    clean_idx = {i for i, k in enumerate(keys) if k not in leak_train}
    strict_idx = {i for i in clean_idx if keys[i] not in leak_val}
    seen_idx = set(range(len(keys))) - strict_idx              # 训练过 或 训练期反复评过
    print(f"[口径] 全部 {len(keys)} − 训练泄漏 {len(keys) - len(clean_idx)} = **clean {len(clean_idx)}**"
          f" ； 再 − 验证泄漏 {len(clean_idx) - len(strict_idx)} = **strict {len(strict_idx)}**", flush=True)

    # ---- 会遇类型：manifest 模式用标注；官方模式用几何分类（复用 classify_scenarios 判据·不另写规则） ----
    #      🔴 已提成 `classify_pool` 供外部基线 runner 共用 —— 分型判据必须**只有一处**（`03` L215-D）。
    types = classify_pool(pool, keys, type_of, np)
    if types:
        print(f"[分型] {dict(Counter(types.values()))}", flush=True)

    # ---- 🆕 r8：要采轨迹的场景（多算法轨迹对比图·`03` L215-G）。不设 REEVAL_TRAJ_KEYS ⟹ traj_idxs=None ⟹ 逐位不变 ----
    traj_idxs, trajs = None, {}
    if TRAJ_KEYS:
        kmap = {str(k): i for i, k in enumerate(keys)}
        miss = [k for k in TRAJ_KEYS if k not in kmap]
        if miss:
            # fail-closed：跑完几千局才发现"想画的那几个场景不在池里"= 白跑一趟
            raise SystemExit(f"🔒 REEVAL_TRAJ_KEYS 里这些键不在本池内：{miss}（池里共 {len(keys)} 个）→ "
                             "画不出图还静默跑完，中止（先看 json 的 `strict键` 挑）。")
        traj_idxs = {kmap[k] for k in TRAJ_KEYS}
        print(f"[轨迹] 记 {sorted(TRAJ_KEYS)} 这几个场景的逐步轨迹（`04 §1.5` 第⑥条·离散臂与连续臂同格式可叠图）",
              flush=True)

    # ---- 逐 checkpoint：锚点复现检查 → 正式评 ----
    results = {}
    _ALPHA_OF = {}                                              # out_name → 该档用的低通 α（进 json 供画权衡曲线）

    def _dump(final=False):
        """**每评完一个 checkpoint 就落盘一次**（本项目两次被中途打断咬过：欠费 / SSH SIGHUP·`03` L193/L192-C）
        → 半途被杀也留得住已评出的臂，不用从头再来。"""
        payload = {"池": {"spec": POOL_SPEC, "说明": pool_desc, "N": len(pool), "请求": len(want_keys),
                          "clean": len(clean_idx), "strict": len(strict_idx),
                          "训练泄漏": len(keys) - len(clean_idx), "验证泄漏": len(clean_idx) - len(strict_idx),
                          "smoke": SMOKE, "REEVAL_N": CLEAN_N},
                   "官方划分": {"n_total": OFF_N_TOTAL, "test_frac": OFF_TEST_FRAC,
                                "split_seed": OFF_SPLIT_SEED, "pool": OFF_POOL},
                   "会遇类型计数": (dict(Counter(types.values())) if types else {}),
                   "池键": keys, "clean键": sorted((keys[i] for i in clean_idx), key=str),
                   "strict键": sorted((keys[i] for i in strict_idx), key=str),
                   "已完成": len(results), "待评": len(ckpts) - len(results), "全部完成": final,
                   "结果": results}
        if os.path.dirname(OUT):
            os.makedirs(os.path.dirname(OUT), exist_ok=True)
        tmp = OUT + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:          # 原子写：半写被杀不留损坏 json
            json.dump(payload, fh, ensure_ascii=False, indent=1)
        os.replace(tmp, OUT)
        if trajs:                                             # 🆕 r8：轨迹单独一个文件（不塞主 json·免撑大）·同样原子写
            tp = TRAJ_OUT or ((OUT[:-5] if OUT.endswith(".json") else OUT) + "_traj.json")
            with open(tp + ".tmp", "w", encoding="utf-8") as fh:
                json.dump(trajs, fh, ensure_ascii=False)
            os.replace(tp + ".tmp", tp)

    def _run_one(base, kind, weight, algo, out_name, sc, anchor, wrap):
        """评一个（checkpoint × 转向滤波档）→ 落进 results[out_name] 并刷盘。

        抽成函数是为了让 r9 的 α 循环复用同一段口径（三口径切子集 / 分型 / 卖点指标 / 落盘）——
        **同一段代码跑所有档**，免得"对照组和滤波组走了两条略不同的路"这种最难查的错。
        """
        agg, per = R.replay_eval(base, kind, weight, pool, continuous_algo=algo, return_per=True,
                                 traj_idxs=traj_idxs, policy_wrap=wrap)
        if traj_idxs:
            trajs[out_name] = {str(keys[p["scenario_idx"]]): p.get("traj")
                               for p in per if p.get("scenario_idx") in traj_idxs and p.get("traj")}
        by_type, by_type_clean = {}, {}
        if types:
            g = defaultdict(set)
            for i, t in types.items():
                g[t].add(i)
            by_type = {t: agg_of(per, ix) for t, ix in sorted(g.items())}
            by_type_clean = {t: agg_of(per, ix & clean_idx) for t, ix in sorted(g.items())}
        results[out_name] = {"kind": kind, "party": sc.get("party"), "seed": sc.get("seed"),
                             "num_timesteps": sc.get("num_timesteps"),
                             "dataset": (sc.get("config_sig") or {}).get("dataset"),
                             "anchor": anchor,
                             "平滑档": (None if wrap is None else _ALPHA_OF.get(out_name)),
                             "全部": agg_of(per), "clean": agg_of(per, clean_idx),
                             "strict": agg_of(per, strict_idx), "看过的": agg_of(per, seen_idx),
                             "分型_全部": by_type, "分型_clean": by_type_clean}
        r = results[out_name]
        print(fmt("全部", r["全部"]))
        print(fmt("clean（没训练过）", r["clean"]))
        print(fmt("strict（真一眼没见过）", r["strict"]))
        _cl = fmt_ctrl(r["strict"])                            # 卖点指标（平滑/次网格/按态势拆转艏）·`03` L203
        if _cl:
            print(_cl)
        if r["看过的"]:
            print(fmt("对照：训练/验证见过的", r["看过的"]))
        for t, m in (r["分型_clean"] or r["分型_全部"] or {}).items():
            print(fmt(f"  · {t}", m))
        _dump()                                                # 🔴 每评完一档就落盘（中途被杀也留得住）

    for b in ckpts:
        sc, name = sidecars[b], os.path.basename(b)
        if sc is None:
            raise SystemExit(f"🔒 {name} 没有 .progress.json → 判不出臂类型(kind)/训练数据集/锚点值，中止（sidecar 是 provenance 命门）。\n"
                             "   老 ckpt 确无 sidecar 时可显式降级：REEVAL_FORCE_KIND=continuous|shielded|unshielded "
                             "+ REEVAL_FORCE_DATASET=manifest_hocr_200.json（**此时锚点检查做不了·可信度自行承担**）。")
        if sc.get("_forced"):
            print(f"  ⚠️⚠️ {name} 无 sidecar → 用人工指定 kind={sc['kind']} dataset={FORCE_DATASET}；"
                  "**锚点复现检查无法进行** = eval 配置是否与训练一致【未经验证】。", flush=True)
        kind, weight = sc.get("kind"), sc.get("colregs_weight", 0.0)
        if kind not in ("continuous", "shielded", "unshielded"):
            raise SystemExit(f"🔒 {name} 的 kind={kind!r} 不认识（须 ∈ continuous/shielded/unshielded），中止。")
        algo = ((sc.get("config_sig") or {}).get("continuous_algo")) if kind == "continuous" else None
        print("\n" + "─" * 104)
        print(f"▶ {name}  臂={sc.get('party')} kind={kind} seed={sc.get('seed')} 步={sc.get('num_timesteps')}", flush=True)

        anchor = None
        if ANCHOR:
            ds = (sc.get("config_sig") or {}).get("dataset")
            mp = find_manifest(ds)
            trend = sc.get("trend") or []
            want = (trend[-1] or {}).get("到达率%") if trend else None
            if mp is None or want is None:
                print(f"  ⚠️ 锚点检查跳过（数据集={ds!r}·manifest={'找到' if mp else '没找到'}·trend={'有' if trend else '无'}）"
                      " → **本 ckpt 的 eval 配置未经验证**，数字请谨慎采信。", flush=True)
            elif not _sidecar_in_sync(b, sc):
                # progress.json 里记着写它那一刻的 .zip 指纹（mtime+size）。对不上 ⟹ 模型文件比进度记录【新】
                #   （典型成因：run 被杀在 "存 zip" 与 "写 progress.json 提交点" 之间·或事后续跑覆盖了 zip）
                #   ⟹ trend[-1] 描述的不是这个模型 ⟹ **锚点比对本身无效**，不是配置错 ⟹ 只警告不中止。
                print(f"  ⚠️ 锚点检查跳过：sidecar 与 ckpt **不同步**（progress.json 记的 zip 指纹 ≠ 现在的 zip）"
                      " → `trend[-1]` 描述的不是这个模型，比了也没意义。**本 ckpt 的 eval 配置未经锚点验证。**", flush=True)
                anchor = {"skipped": "sidecar_out_of_sync"}
            else:
                _tp, a_paths, _i = R.load_manifest_split(mp, os.path.dirname(mp))
                a_pool = load_scenario_pool(a_paths)
                a_agg, _ap = R.replay_eval(b, kind, weight, a_pool, continuous_algo=algo, return_per=True)
                got, one_ep = a_agg["到达率%"], 100.0 / max(len(a_pool), 1)
                d = abs(got - want)
                # 🔴 容差取【本项目实测的重放噪声】而非"零容忍"：`03` L192-C 实测同 ckpt/同 40 场景重放
                #   最大偏移 ±7.5pp（3/40 局判定翻转·浮点平台差非 bug）⟹ 逐种子零容忍会被噪声误杀
                #   （2026-07-26 实证：热启动 s1 差 2 局即被误判中止，而同批金标 6/6 逐位复现=配置本就没错）。
                #   真正的配置错会【所有种子系统性偏移】⟹ 靠下面的「锚点汇总」抓，不靠单种子。
                tol = max(ANCHOR_TOL, 4.0 * one_ep)            # 4 局 = 实测最大噪声 3 局 + 1 局余量
                anchor = {"n": len(a_pool), "记录值": want, "重评值": got, "差": d, "有符号差": got - want,
                          "容差": tol, "差几局": round(d / one_ep, 2), "通过": d <= tol}
                tag = ("✅ 逐位复现" if d == 0.0 else
                       f"🟡 差 {d / one_ep:.0f} 局（实测重放噪声内·`03` L192-C）" if d <= tol else "❌ 对不上")
                print(f"  [锚点] 自记测试集 N={len(a_pool)}：记录 {want:.2f}% vs 重评 {got:.2f}%（差 {d:.2f}pt）{tag}", flush=True)
                if d > tol:
                    msg = (f"🔒 锚点复现失败（{name}）：差 {d:.2f}pt = {d / one_ep:.1f} 局 > 容差 {tol:.2f}pt"
                           f"（= {tol / one_ep:.0f} 局·已按 `03` L192-C 实测重放噪声放宽）。\n"
                           "   含义：超出浮点噪声可解释的范围 → eval 环境配置可能与训练时【不一致】"
                           "（连续臂 shield/cone/augment_rho/去朝向门这些不进 config_sig 的 knob 最可能）。\n"
                           "   先查这些 STEP4E_* 变量；确认无误再用 REEVAL_ANCHOR_TOL / REEVAL_ANCHOR_SOFT=1 放行。")
                    if ANCHOR_SOFT:
                        print("  ⚠️ " + msg, flush=True)
                    else:
                        raise SystemExit(msg)

        # 🔴 traj_idxs 只给【正式池】这一趟；上面的**锚点**那趟用的是另一个池（自记 40 场景）⟹ 下标含义不同、
        #    绝不能把同一组下标传给它（传了就会记错场景的轨迹）。
        # 🆕 r9/r10：两个平滑旋钮共用同一段口径。两个都不设 ⟹ 只跑一趟、`policy_wrap=None`
        #    ⟹ 与 r8 逐位相同。设了就每档各评一趟（键名带 @lp / @sl 后缀）。
        _specs = []
        for _a in YAW_LOWPASS:
            _specs.append(("lp", _a, (lambda m, v=_a: YawLowPassPolicy(m, v))))
        for _f in YAW_SLEW:
            _specs.append(("sl", _f, (lambda m, v=_f: YawSlewLimitPolicy(m, v))))
        if not _specs:
            _specs = [(None, None, None)]
        for _kindtag, _val, _mk in _specs:
            if _kindtag is None:
                _wrap, _suffix = None, ""
            elif kind != "continuous":
                print(f"  ⚠️ {name} 是离散臂 → 跳过平滑档（网格下标做连续量平滑无意义）", flush=True)
                continue
            else:
                _wrap = _mk
                _suffix = f"@{_kindtag}{_val:g}"
                _ALPHA_OF[name + _suffix] = {"mode": _kindtag, "value": _val}
                _label = "转向低通 α" if _kindtag == "lp" else "舵速率限制 系数"
                print(f"  ── {_label}={_val:g}" + ("（=不施加·对照组）" if _val >= 1.0 else ""), flush=True)
            _run_one(b, kind, weight, algo, name + _suffix, sc, anchor, _wrap)

    # ---- 🔴 锚点【汇总】：真正的配置错会让所有种子【系统性同向偏移】；单种子的 1-3 局翻转只是浮点噪声 ----
    _anc = [v["anchor"] for v in results.values() if v.get("anchor") and "有符号差" in v["anchor"]]
    if _anc:
        sg = [a["有符号差"] for a in _anc]
        mean_sg = sum(sg) / len(sg)
        worst = max(abs(x) for x in sg)
        pos, neg = sum(1 for x in sg if x > 0), sum(1 for x in sg if x < 0)
        print("\n" + "─" * 104)
        print(f"■ 锚点汇总（{len(sg)} 个 ckpt）：平均有符号差 {mean_sg:+.2f}pt · 最大绝对差 {worst:.2f}pt · "
              f"{sum(1 for x in sg if x == 0)} 个逐位复现 / {pos} 偏高 / {neg} 偏低")
        print("  判读：平均差 ≈ 0 且正负混杂 ⟹ 只是浮点重放噪声，**eval 配置正确**；"
              "若平均差明显偏离 0 且方向一致 ⟹ 配置很可能真的错了，别信本次数字。", flush=True)
        payload_extra = {"n": len(sg), "平均有符号差": mean_sg, "最大绝对差": worst, "逐位复现数": sum(1 for x in sg if x == 0)}
        results["_锚点汇总"] = payload_extra

    _dump(final=True)
    print(f"\n[reeval_official] 完成 → {OUT}", flush=True)
    if trajs:
        tp = TRAJ_OUT or ((OUT[:-5] if OUT.endswith(".json") else OUT) + "_traj.json")
        print(f"[轨迹] {len(trajs)} 个 checkpoint × {len(TRAJ_KEYS)} 个场景 → {tp}", flush=True)
    if SMOKE or CLEAN_N:
        print("⚠️ 这是【冒烟/截断】跑，不是正式数——正式跑请清掉 REEVAL_SMOKE / REEVAL_N。", flush=True)


if __name__ == "__main__":
    main()
