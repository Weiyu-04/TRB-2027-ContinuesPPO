#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""外部对比基线【官方测试集闭环跑】—— `usv_baseline_runner.py` 缺的"接池子 + 落盘"那一半（`03` L213-E①/L215）。

═══ 这个文件负责什么 ═══
`usv_baseline_runner.py` 已经把 **纯 VO / 纯 CBF / 纯 PD** 包装成 `.predict()`（可走官方评估器）；
本文件补上剩下的三件事，使外部基线与我们四臂**落在同一张表里**：
  ① **池子**：复用 `代码/tests/reeval_official.py` 的**官方 1400/600 划分** + **两层泄漏剔除**
     （600 → clean 577 → **strict 563**）+ **同一套 `agg_of` 聚合** + **同一套会遇类型分型 `classify_pool`**。
     ⟹ 分母、指标口径、分型判据**都不是重写的，是复用的**（重写一份就是口径漂移的起点·`03` L215-D）。
  ② **闭环**：`ContinuousProjectionEnv(shield=False)` + 官方 `evaluate_continuous`
     ⟹ 到达/碰撞/违规/平滑度/次网格细调率**与我们四臂逐字同源**。
  ③ **落盘**：与 `reeval_official.py` **同款 json**（同样的「池 / 官方划分 / 结果」结构）⟹ 下游出表脚本不用改。

═══ 🔴 反稻草人（`03` L129 对 CBF · L213-D 对 VO · 本文件强制执行）═══
  · **参数必须扫、报基线自己最好的配置**：VO 的 (tau, margin) · CBF 的 (a1,a2,margin)。
  · **`BASELINE_BOX` 的 rl / full 两档都跑、都报**（rl=同我方操作权限 · full=给基线全物理量程=更慷慨）。
  · **调参池与报数池【不重叠】**：调参用**官方训练 1400 里的样本**，报数用 **strict 563**
    ⟹ 既满足"取基线最好配置"，又不是在报数的那批场景上调参（免"你替基线在测试集上调参"的反向指责）。
    `BASELINE_TUNE_SRC=test` 可改成在测试子样本上调（对基线**更慷慨**）——**用了就必须在论文里写明**。
  · **framing 恒为 Pareto 前沿 / 各有所长**，不是"我们样样赢" ⟹ 本文件**不做单一标量排序**，
    而是报 (碰撞↓, 违规/局↓, 到达↑) 三目标的**非支配前沿** + 三个单轴极值，全部配置的明细也一并落盘。
  · **没有静默截断**：最终跑的配置数受 `BASELINE_FINAL_MAX` 限制时，**被丢掉的配置会打印出来**。

═══ ⚠️ 诚实边界（判读/写作时必带）═══
  · **纯基线不会学习 ⟹ 到达率天然吃亏**（`03` L213-F）⟹ **到达率不是这条线的看点**
    （本项目早已把到达率移出卖点）；看点 = **碰撞 / COLREGs 违规 / 不可行率 / 平滑度**。
  · **`yaw_incr_giveway / yaw_incr_other`（按态势拆转艏）对外部基线缺失**：`shield=False` 时 env 不推状态机
    ⟹ 拿不到 ρ（`03` L213-E 已记）。其余指标全有。补法=给 env 加"只算 ρ 不投影"开关（additive·默认关）
    = **可选、不 gate 主线**。
  · **`full` 档确实可执行**（已核实非幻觉）：连续动作在 `usv_env._map_action` 只截到**物理** ±0.24/±0.03，
    不是 RL 箱 ±0.048/±0.018。
  · 本文件**从未在服务器真跑过**（本机无 vesselmodels/commonocean）⟹ 按 `03` L202 教训，
    **服务器冒烟通过前不算真验**。

用法：
  本机自检（纯逻辑·不需 vesselmodels·秒级）：
      python 代码/m1_dock_wip/run_baselines_official.py --selftest
  服务器冒烟（30 场景 · 不扫参）：
      BASELINE_SMOKE=1 ... python 代码/m1_dock_wip/run_baselines_official.py --run
  服务器全量：
      BASELINE_METHODS=vo,cbf,pd BASELINE_BOX=rl,full ... python 代码/m1_dock_wip/run_baselines_official.py --run
  （也可经 `usv_baseline_runner.py --run` 进来，它会 import 本文件的 main。）
"""
import json
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.dirname(_HERE)
_TESTS = os.path.join(_CODE, "tests")
for _p in (_HERE, _CODE, _TESTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# 🔴 `reeval_official` 在 **import 时**就读环境变量（MANIFEST_DIRS / CKDIRS 等）⟹ 必须在 import 【之前】
#    把本文件的 `BASELINE_MANIFEST_DIRS` 翻译过去，否则 `find_manifest` 找不到 balanced_pool、泄漏剔不掉。
if os.environ.get("BASELINE_MANIFEST_DIRS"):
    os.environ.setdefault("REEVAL_MANIFEST_DIRS", os.environ["BASELINE_MANIFEST_DIRS"])

import reeval_official as RO                                   # noqa: E402 —— 必须在上面那行之后
from usv_baseline_runner import BOXES, GeometricPolicy, SCRIPT_REV as RUNNER_REV, sweep_grid   # noqa: E402

SCRIPT_REV = "rb1-2026-07-26"      # 同 reeval_official 的做法：服务器同步后 `grep SCRIPT_REV` 一眼验版本

# ---------- 开关 ----------
SELFTEST = "--selftest" in sys.argv or os.environ.get("BASELINE_SELFTEST", "0") == "1"
SMOKE = os.environ.get("BASELINE_SMOKE", "0") == "1"
METHODS = [m.strip() for m in os.environ.get("BASELINE_METHODS", "vo,cbf,pd").split(",") if m.strip()]
BOX_NAMES = [b.strip() for b in os.environ.get("BASELINE_BOX", "rl,full").split(",") if b.strip()]
VARIANTS = [v.strip() for v in os.environ.get("BASELINE_VARIANTS", "colregs").split(",") if v.strip()]
TUNE_N = int(os.environ.get("BASELINE_TUNE_N", "100"))
TUNE_SRC = os.environ.get("BASELINE_TUNE_SRC", "train").strip()      # train（默认·与报数池不重叠）| test
FINAL_MAX = int(os.environ.get("BASELINE_FINAL_MAX", "3"))
FINAL_N = int(os.environ.get("BASELINE_N", "0"))                     # >0 = 截断报数池（冒烟用·非正式数）
LEAK_MANIFESTS = [s.strip() for s in os.environ.get(
    "BASELINE_LEAK_MANIFESTS", "manifest_hocr_200.json").replace(":", ",").split(",") if s.strip()]
KEYS_REF = os.environ.get("BASELINE_KEYS_REF", "").strip()           # 主线 json → strict 键集合硬比对
OUT = os.environ.get("BASELINE_OUT", "baselines_official.json")
TRAJ_KEYS = [s.strip() for s in os.environ.get("BASELINE_TRAJ_KEYS", "").replace(",", " ").split() if s.strip()]
ENVCFG_ACK = os.environ.get("BASELINE_ENVCFG_ACK", "0") == "1"
LEAK_ACK = os.environ.get("BASELINE_LEAK_ACK", "0") == "1"
NOMINAL = os.environ.get("BASELINE_NOMINAL", "pd").strip()           # pd=路B（头条）| rl=路A（消融·需 ckpt）
RL_CKPT = os.environ.get("BASELINE_RL_CKPT", "").strip()
if SMOKE:
    FINAL_N = FINAL_N or 30
    TUNE_N = min(TUNE_N, 10)
    os.environ.setdefault("BASELINE_SWEEP", "off")

# 三目标（键, +1=越大越好 / −1=越小越好）。**刻意不做标量化**——framing 是 Pareto 前沿（`03` L129/L213-D）。
OBJECTIVES = (("碰撞率%", -1), ("违规次数/局", -1), ("到达率%", +1))


# ═════════════════════════ 纯逻辑层（本机可自检·不碰重依赖） ═════════════════════════
def config_tag(method, box, variant, params):
    """配置 → 稳定可读的 json 键。参数按名排序 ⟹ 同一配置永远同一个键（跨机跨次可比对）。"""
    ps = ",".join(f"{k}={params[k]:g}" if isinstance(params[k], (int, float)) else f"{k}={params[k]}"
                  for k in sorted(params))
    return f"{method}|{box}|{variant}" + (f"|{ps}" if ps else "")


def build_configs(methods, boxes, variants, grid):
    """(方法 × 箱 × 变体 × 参数网格) → 配置列表。缺网格的方法 fail-closed（免静默跑一个"默认参数"充数）。"""
    out = []
    for m in methods:
        if m not in grid:
            raise SystemExit(f"🔒 方法 `{m}` 在 sweep_grid() 里没有参数网格 → 会静默跑一个没扫过的默认点，中止。"
                             f"（可用：{sorted(grid)}）")
        for b in boxes:
            if b not in BOXES:
                raise SystemExit(f"🔒 未知动作箱 `{b}`（须 ∈ {sorted(BOXES)}）。")
            for v in variants:
                for p in grid[m]:
                    out.append({"method": m, "box": b, "variant": v, "params": dict(p),
                                "tag": config_tag(m, b, v, p)})
    return out


def _better(val_a, val_b, sign):
    return (val_a > val_b) if sign > 0 else (val_a < val_b)


def dominates(a, b, objectives=OBJECTIVES):
    """a 是否 Pareto 支配 b：所有目标上不差，且至少一个目标上更好。缺指标（None）视为"最差"。"""
    def g(row, k, sign):
        v = row.get(k)
        return (-math.inf if sign > 0 else math.inf) if v is None else float(v)
    not_worse = all(not _better(g(b, k, s), g(a, k, s), s) for k, s in objectives)
    strictly = any(_better(g(a, k, s), g(b, k, s), s) for k, s in objectives)
    return not_worse and strictly


def pareto_front(rows, objectives=OBJECTIVES):
    """非支配集（保持输入顺序·不去重相同指标的配置：它们互不支配、都留着让论文看清前沿）。"""
    return [r for r in rows if not any(dominates(o, r, objectives) for o in rows if o is not r)]


def select_configs(rows, max_n, objectives=OBJECTIVES):
    """挑最终要在报数池上跑的配置 → (选中, 被丢掉)。

    规则（**明写出来免事后找补**）：
      ① 先取三个**单轴极值**（碰撞最低 / 违规最低 / 到达最高）—— 保证"基线在每一根轴上自己最好的成绩"都被报出来。
      ② 再按 Pareto 前沿补齐（前沿内按到达率降序）。
      ③ 超过 `max_n` 的**打印出来**（`03` 纪律：不许静默截断）。
    `rows` 每项 = {"tag":..., 指标键...}；缺指标的行不会被选成极值（`dominates` 里按最差处理）。
    """
    if not rows:
        return [], []
    picked, seen = [], set()

    def take(r):
        if r is not None and r["tag"] not in seen:
            seen.add(r["tag"])
            picked.append(r)

    for k, sign in objectives:
        have = [r for r in rows if r.get(k) is not None]
        if have:
            take(max(have, key=lambda r: float(r[k])) if sign > 0 else min(have, key=lambda r: float(r[k])))
    for r in sorted(pareto_front(rows, objectives), key=lambda r: -(r.get("到达率%") or -math.inf)):
        take(r)
    dropped = [r for r in picked[max_n:]]
    return picked[:max_n], dropped


def stride_sample(ids, n):
    """从有序 id 列表里**确定性**取 n 个（等距抽样·不用随机数 ⟹ 换机器/重跑完全一致）。"""
    ids = list(ids)
    if n <= 0 or n >= len(ids):
        return ids
    step = len(ids) / float(n)
    return [ids[min(len(ids) - 1, int(i * step))] for i in range(n)]


def leak_keys(manifests):
    """泄漏集 = 所有给定 manifest 的（训练键 ∪ 验证键）。返回 (训练泄漏集, 验证泄漏集, 明细)。

    外部基线**自己没有训练集**（几何控制器不训练）⟹ 它本身无泄漏；但要与我们四臂**同分母**可比，
    就必须剔掉**我们的臂**看过的那些场景 ⟹ 泄漏集取自我方 manifest（默认 `manifest_hocr_200.json`）。
    """
    tr, va, detail = set(), set(), {}
    for name in manifests:
        mp = RO.find_manifest(name)
        if mp is None:
            raise SystemExit(f"🔒 找不到 manifest `{name}`（找过 {RO.MANIFEST_DIRS}）→ 泄漏剔不干净就等于自我污染，"
                             "中止（用 BASELINE_MANIFEST_DIRS/REEVAL_MANIFEST_DIRS 指到 balanced_pool）。")
        t, v, _ = RO.manifest_keys(mp)
        tr |= t
        va |= v
        detail[os.path.basename(mp)] = {"训练": len(t), "验证": len(v)}
    return tr, va, detail


# ═════════════════════════ 闭环层（需 vesselmodels/commonocean·服务器） ═════════════════════════
def _build_pool(R, load_scenario_pool, want_keys, what):
    """下载 + 过滤 + 载入场景池 → (pool, keys)。缺额 <95% 中止（防分母静默缩水·同 reeval_official 第③闸）。"""
    R._download(want_keys)
    paths = [f"{R._SDIR}/T-{i}.xml" for i in want_keys]
    paths = [p for p in paths if os.path.exists(p) and os.path.getsize(p) > 1000]
    if len(paths) < 0.95 * len(want_keys):
        raise SystemExit(f"🔒 {what}：场景只拿到 {len(paths)}/{len(want_keys)}（<95%）→ 分母静默缩水会污染结论，"
                         "中止（查网络 / STEP4E_SDIR）。")
    return load_scenario_pool(paths), [RO.key_of_path(p) for p in paths]


def _make_policy(cfg, rl_model=None):
    return GeometricPolicy(cfg["method"], box=cfg["box"], variant=cfg["variant"],
                           params=cfg["params"], rl_model=rl_model)


def eval_config(cfg, pool, env_factory, evaluate_continuous, *, rl_model=None, obs_tf=None, traj_idxs=None):
    """跑一个配置 → (per, 逐局不可行率已注入)。

    🔴 **逐局对齐硬闸**：策略每步调一次 `predict` ⟹ 策略记的本局步数必须**恰好等于** evaluate 记的
       `per[i]["steps"]`。对不上 = 绑定/切局出了错（比如 `bind_env` 没被调、或 evaluate 换了调用次序）
       ⟹ 不可行率就会错位到别的局上 ⟹ **fail-closed，不猜**。
    """
    pol = _make_policy(cfg, rl_model=rl_model)
    _agg, per = evaluate_continuous(env_factory, pol, pool, obs_transform=obs_tf, traj_idxs=traj_idxs)
    eps = pol.finalize()
    if len(eps) != len(per):
        raise SystemExit(f"🔒 {cfg['tag']}：策略记了 {len(eps)} 局、评估器给了 {len(per)} 局 → 逐局计数对不上，"
                         "不可行率会错位，中止（检查 GeometricPolicy.bind_env 是否被 evaluate 调到）。")
    for i, (row, e) in enumerate(zip(per, eps)):
        if int(e["steps"]) != int(row.get("steps", -1)):
            raise SystemExit(f"🔒 {cfg['tag']} 第 {i} 局：策略步数 {e['steps']} ≠ 评估步数 {row.get('steps')} → "
                             "逐局对齐断了，中止（别在错位的数上判读）。")
        st = max(int(e["steps"]), 1)
        row["baseline_infeasible_pct"] = round(100.0 * e["infeasible"] / st, 6)   # 数值键 ⟹ agg_of 自动聚合
    return per


def main():
    if SELFTEST:
        return selftest()

    if NOMINAL not in ("pd", "rl"):
        raise SystemExit(f"🔒 BASELINE_NOMINAL={NOMINAL!r} 不认识（须是 pd=路B 头条 / rl=路A 消融）。")
    if NOMINAL == "rl" and not RL_CKPT:
        raise SystemExit("🔒 BASELINE_NOMINAL=rl（路 A 消融：标称=我们训的策略）必须给 BASELINE_RL_CKPT=<ckpt 去掉 .zip 的路径>。")

    sys.path.insert(0, _CODE)
    import numpy as np
    import run_step4e as R
    from trb_env.evaluate import evaluate_continuous
    from trb_env.usv_continuous_shield import ContinuousProjectionEnv
    from trb_env.usv_scenarios import load_scenario_pool

    print("=" * 104)
    print(f"[run_baselines_official {SCRIPT_REV}] runner={RUNNER_REV} · reeval={RO.SCRIPT_REV}")
    print(f"  方法={METHODS} · 箱={BOX_NAMES} · 变体={VARIANTS} · 标称={NOMINAL}"
          + (f"（ckpt={os.path.basename(RL_CKPT)}）" if NOMINAL == "rl" else ""))

    # ---- env 配置守卫：终端到达门必须与我们四臂一致，否则"到达率"这一列根本不可比 ----
    #      （`goal_cone_half`/`goal_v_floor` 是**投影**参数·shield=False 时不生效；`goal_ignore_orientation`
    #        是**终端门**参数·会直接改到达判定 ⟹ 必须锁死成金标默认。）
    golden = {"STEP4E_GOAL_IGNORE_ORIENT": R._GOAL_IGNORE_ORIENT is False,
              "STEP4E_AUGMENT_RHO": R._AUGMENT_RHO is False}
    print(f"[env] 外部基线 env：shield=False（固定）· 去朝向门={R._GOAL_IGNORE_ORIENT} · "
          f"augment_rho={R._AUGMENT_RHO} · clip_velocity=True（同 replay_eval 默认）", flush=True)
    bad = [k for k, ok in golden.items() if not ok]
    if bad and not ENVCFG_ACK:
        raise SystemExit(f"🔒 终端到达门/观测 knob 非【金标默认】：{bad} → 与我们四臂不同口径，到达率不可比"
                         "（`03` L192 那类静默改数字）。确认是故意的加 BASELINE_ENVCFG_ACK=1，否则清掉这些 STEP4E_* 重跑。")

    # ---- 官方划分（镜像 vs 真身硬比对在 RO.official_split 里做·不一致直接中止） ----
    off_train, off_test = RO.official_split(R)
    lk_tr, lk_va, lk_detail = leak_keys(LEAK_MANIFESTS)
    print(f"[泄漏集] {lk_detail} → 训练键 {len(lk_tr)} · 验证键 {len(lk_va)}", flush=True)

    # ---- 报数池 = 官方测试 600（截断只用于冒烟） ----
    want = off_test[:FINAL_N] if FINAL_N > 0 else list(off_test)
    pool, keys = _build_pool(R, load_scenario_pool, want, "报数池")
    clean_idx = {i for i, k in enumerate(keys) if k not in lk_tr}
    strict_idx = {i for i in clean_idx if keys[i] not in lk_va}
    seen_idx = set(range(len(keys))) - strict_idx
    print(f"[口径] 全部 {len(keys)} − 训练泄漏 {len(keys) - len(clean_idx)} = **clean {len(clean_idx)}**"
          f" ； 再 − 验证泄漏 {len(clean_idx) - len(strict_idx)} = **strict {len(strict_idx)}**", flush=True)
    if FINAL_N == 0 and not LEAK_ACK:
        # 全量跑时钉死这两个数：它们是主线表的分母，对不上就说明划分/manifest 变了 ⟹ 表不可比。
        if len(clean_idx) != RO.EXPECT_CLEAN_HOCR or len(strict_idx) != RO.EXPECT_STRICT_HOCR:
            raise SystemExit(f"🔒 分母对不上主线表：clean={len(clean_idx)}（应 {RO.EXPECT_CLEAN_HOCR}）· "
                             f"strict={len(strict_idx)}（应 {RO.EXPECT_STRICT_HOCR}）→ 与四臂【不同分母】就不能同表比，"
                             "中止（换了泄漏 manifest 才该放行：BASELINE_LEAK_ACK=1，**并在论文写清口径**）。")

    # ---- 同分母【硬比对】：与主线 json 的 strict 键集合逐个对（这是"同一张表"最强的保证） ----
    if KEYS_REF:
        with open(KEYS_REF, encoding="utf-8") as fh:
            ref = json.load(fh)
        ref_strict = {str(x) for x in (ref.get("strict键") or [])}
        ours = {str(keys[i]) for i in strict_idx}
        if not ref_strict:
            raise SystemExit(f"🔒 BASELINE_KEYS_REF={KEYS_REF} 里没有 `strict键` → 比不了，中止。")
        if ref_strict != ours:
            raise SystemExit(f"🔒 strict 键集合与主线 json 不一致（主线 {len(ref_strict)} · 本次 {len(ours)} · "
                             f"只在主线 {len(ref_strict - ours)} · 只在本次 {len(ours - ref_strict)}）→ "
                             "不是同一批场景就不能同表比，中止。")
        print(f"  ✅ 同分母硬比对通过：strict 键集合与 {os.path.basename(KEYS_REF)} **逐个相同**（{len(ours)} 个）", flush=True)
    else:
        print("  ⚠️ 没给 BASELINE_KEYS_REF → 分母只靠 577/563 计数断言，**没有与主线 json 逐键比对**"
              "（强烈建议指过去：BASELINE_KEYS_REF=结果/结果0725-大数据集测试/reeval_official_ws.json）", flush=True)

    types = RO.classify_pool(pool, keys, None, np)              # 与我们四臂**同一个**分型函数
    if types:
        from collections import Counter
        print(f"[分型] {dict(Counter(types.values()))}", flush=True)

    traj_idxs = None
    if TRAJ_KEYS:
        kmap = {str(k): i for i, k in enumerate(keys)}
        miss = [k for k in TRAJ_KEYS if k not in kmap]
        if miss:
            raise SystemExit(f"🔒 BASELINE_TRAJ_KEYS 里这些键不在报数池内：{miss} → 画不出图还静默跑完，中止。")
        traj_idxs = {kmap[k] for k in TRAJ_KEYS}
        print(f"[轨迹] 记 {sorted(TRAJ_KEYS)} 这几个场景的逐步轨迹（多算法轨迹对比图用·`04 §1.5` 第⑥条）", flush=True)

    # ---- 路 A（消融）：标称 = 我们训的策略。重建 VecNormalize 的方式与 `replay_eval` **逐字相同** ----
    rl_model, obs_tf = None, None
    if NOMINAL == "rl":
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        from trb_env.train import make_obs_transform
        pkl = RL_CKPT + "_vecnorm.pkl"
        if not os.path.exists(RL_CKPT + ".zip") or not os.path.exists(pkl):
            raise SystemExit(f"🔒 路 A 需要 `{RL_CKPT}.zip` + `_vecnorm.pkl` 两个都在（缺 vecnorm ⟹ 观测归一化重建不了、"
                             "策略看错分布、数是错的），中止。")
        _sc0, _pp0 = pool[0]
        _bv = DummyVecEnv([lambda: ContinuousProjectionEnv(_sc0, _pp0, shield=False)])
        _vn = VecNormalize.load(pkl, _bv)
        _vn.training = False
        obs_tf = make_obs_transform(_vn)                       # evaluate 层做归一化 ⟹ 策略侧 obs_tf 留 None
        rl_model = PPO.load(RL_CKPT + ".zip", device="cpu")
        print(f"  路 A：标称 = {os.path.basename(RL_CKPT)}（**confounded·只当消融/附录·不进头条**·`03` L213-F）", flush=True)

    def env_factory(sc, pp):
        # 与 `replay_eval` 的连续臂 env **只差 shield=False**；终端门/限速口径逐字相同 ⟹ 同一把尺。
        return ContinuousProjectionEnv(sc, pp, shield=False,
                                       augment_rho=R._AUGMENT_RHO,
                                       goal_ignore_orientation=R._GOAL_IGNORE_ORIENT)

    grid = sweep_grid()
    configs = build_configs(METHODS, BOX_NAMES, VARIANTS, grid)
    print(f"\n[配置] 共 {len(configs)} 个（扫参={'off' if os.environ.get('BASELINE_SWEEP') == 'off' else 'on'}）", flush=True)

    results = {"_meta": {"script_rev": SCRIPT_REV, "runner_rev": RUNNER_REV, "reeval_rev": RO.SCRIPT_REV,
                         "nominal": NOMINAL, "rl_ckpt": (os.path.basename(RL_CKPT) if NOMINAL == "rl" else None),
                         "tune_src": TUNE_SRC, "tune_n": TUNE_N, "final_max": FINAL_MAX,
                         "leak_manifests": lk_detail, "boxes": BOX_NAMES, "variants": VARIANTS,
                         "objectives": [k for k, _ in OBJECTIVES]}}
    tune_rows, trajs = {}, {}

    def _dump(final=False):
        """每跑完一个配置就落盘（原子写）——本项目被欠费/SIGHUP 咬过两次（`03` L192-C/L193）。"""
        payload = {"池": {"spec": "official", "说明": "官方 1400/600 划分的测试 600（官方 2000 无追越）",
                          "N": len(pool), "请求": len(want), "clean": len(clean_idx), "strict": len(strict_idx),
                          "训练泄漏": len(keys) - len(clean_idx), "验证泄漏": len(clean_idx) - len(strict_idx),
                          "smoke": SMOKE, "BASELINE_N": FINAL_N},
                   "官方划分": {"n_total": RO.OFF_N_TOTAL, "test_frac": RO.OFF_TEST_FRAC,
                                "split_seed": RO.OFF_SPLIT_SEED, "pool": RO.OFF_POOL},
                   "会遇类型计数": ({} if not types else dict(__import__("collections").Counter(types.values()))),
                   "池键": keys, "clean键": sorted((keys[i] for i in clean_idx), key=str),
                   "strict键": sorted((keys[i] for i in strict_idx), key=str),
                   "调参结果": tune_rows, "全部完成": final, "结果": results}
        if os.path.dirname(OUT):
            os.makedirs(os.path.dirname(OUT), exist_ok=True)
        tmp = OUT + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=1)
        os.replace(tmp, OUT)

    # ═══ 第 1 阶段：扫参（在**与报数池不重叠**的调参池上）→ 每个 (方法,箱,变体) 选出最终要跑的配置 ═══
    groups = {}
    for c in configs:
        groups.setdefault((c["method"], c["box"], c["variant"]), []).append(c)
    finals = []
    if all(len(v) == 1 for v in groups.values()):
        print("[调参] 每组只有一个配置（扫参关或网格退化）→ 跳过调参阶段，直接跑报数池。", flush=True)
        finals = configs
    else:
        if TUNE_SRC == "train":
            tune_keys = stride_sample(off_train, TUNE_N)
            tune_desc = f"官方训练 1400 等距抽 {len(tune_keys)}（**与报数的 563 完全不重叠**）"
        elif TUNE_SRC == "test":
            tune_keys = stride_sample([keys[i] for i in sorted(strict_idx)], TUNE_N)
            tune_desc = (f"strict 报数池等距抽 {len(tune_keys)}（**对基线更慷慨：在报数场景上调参**"
                         "⟹ 论文必须写明）")
        else:
            raise SystemExit(f"🔒 BASELINE_TUNE_SRC={TUNE_SRC!r} 不认识（须是 train / test）。")
        tune_pool, _tk = _build_pool(R, load_scenario_pool, tune_keys, "调参池")
        print(f"\n[调参池] {tune_desc} → N={len(tune_pool)}", flush=True)
        for c in configs:
            per = eval_config(c, tune_pool, env_factory, evaluate_continuous,
                              rl_model=rl_model, obs_tf=obs_tf)
            m = RO.agg_of(per)
            tune_rows[c["tag"]] = m
            print(RO.fmt(c["tag"], m), flush=True)
            _dump()
        for gk, gcs in sorted(groups.items()):
            rows = [{"tag": c["tag"], **{k: (tune_rows[c["tag"]] or {}).get(k) for k, _ in OBJECTIVES}}
                    for c in gcs]
            keep, dropped = select_configs(rows, FINAL_MAX)
            keep_tags = {r["tag"] for r in keep}
            finals.extend([c for c in gcs if c["tag"] in keep_tags])
            print(f"  · {gk} 选中 {sorted(keep_tags)}", flush=True)
            if dropped:      # 🔴 不许静默截断
                print(f"    ⚠️ 因 BASELINE_FINAL_MAX={FINAL_MAX} 丢掉了同样在前沿上的："
                      f"{[r['tag'] for r in dropped]}（想全跑就调大 FINAL_MAX）", flush=True)

    # ═══ 第 2 阶段：报数（官方测试池·三口径 + 分型 + 卖点指标） ═══
    print("\n" + "=" * 104)
    print(f"■ 报数阶段：{len(finals)} 个配置 × N={len(pool)}", flush=True)
    for c in finals:
        print("\n" + "─" * 104)
        print(f"▶ {c['tag']}", flush=True)
        per = eval_config(c, pool, env_factory, evaluate_continuous,
                          rl_model=rl_model, obs_tf=obs_tf, traj_idxs=traj_idxs)
        by_type, by_type_clean = {}, {}
        if types:
            from collections import defaultdict
            g = defaultdict(set)
            for i, t in types.items():
                g[t].add(i)
            by_type = {t: RO.agg_of(per, ix) for t, ix in sorted(g.items())}
            by_type_clean = {t: RO.agg_of(per, ix & clean_idx) for t, ix in sorted(g.items())}
        results[c["tag"]] = {"method": c["method"], "box": c["box"], "variant": c["variant"],
                             "params": c["params"], "nominal": NOMINAL,
                             "调参指标": tune_rows.get(c["tag"]),
                             "全部": RO.agg_of(per), "clean": RO.agg_of(per, clean_idx),
                             "strict": RO.agg_of(per, strict_idx), "看过的": RO.agg_of(per, seen_idx),
                             "分型_全部": by_type, "分型_clean": by_type_clean}
        r = results[c["tag"]]
        print(RO.fmt("全部", r["全部"]))
        print(RO.fmt("clean（我们没训练过）", r["clean"]))
        print(RO.fmt("strict（与四臂同分母）", r["strict"]))
        _cl = RO.fmt_ctrl(r["strict"])
        if _cl:
            print(_cl)
        _inf = ((r["strict"] or {}).get("控制质量") or {}).get("baseline_infeasible_pct")
        if _inf is not None:
            print(f"      不可行步%（VO 锥外无候选 / CBF QP 无解）: {_inf:.2f}%")
        for t, m in (by_type_clean or by_type or {}).items():
            print(RO.fmt(f"  · {t}", m))
        if traj_idxs:
            trajs[c["tag"]] = {str(keys[p["scenario_idx"]]): p.get("traj")
                               for p in per if p.get("scenario_idx") in traj_idxs and p.get("traj")}
        _dump()

    # ---- Pareto 前沿汇总（报数池·strict 口径）：framing 恒为"各有所长"，不是"我们样样赢" ----
    rows = [{"tag": t, **{k: (v["strict"] or {}).get(k) for k, _ in OBJECTIVES}}
            for t, v in results.items() if not t.startswith("_")]
    front = [r["tag"] for r in pareto_front(rows)]
    results["_pareto"] = {"目标": [k for k, _ in OBJECTIVES], "前沿": front,
                          "说明": "在 strict 563 上的非支配配置；论文报前沿、不报单一'最好'（`03` L129/L213-D）"}
    print("\n" + "─" * 104)
    print(f"■ Pareto 前沿（strict·目标 {[k for k, _ in OBJECTIVES]}）：{front}", flush=True)
    _dump(final=True)

    if trajs:
        tp = (OUT[:-5] if OUT.endswith(".json") else OUT) + "_traj.json"
        with open(tp, "w", encoding="utf-8") as fh:
            json.dump(trajs, fh, ensure_ascii=False)
        print(f"[轨迹] → {tp}", flush=True)
    print(f"\n[run_baselines_official] 完成 → {OUT}", flush=True)
    if SMOKE or FINAL_N:
        print("⚠️ 这是【冒烟/截断】跑，不是正式数——正式跑请清掉 BASELINE_SMOKE / BASELINE_N。", flush=True)
    return 0


# ═════════════════════════ 自检（纯逻辑·本机可跑） ═════════════════════════
def selftest():
    ok = True

    def chk(name, cond, extra=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'✅' if cond else '🔴'}] {name} {extra}")

    print(f"【SELFTEST】{SCRIPT_REV} · 外部基线官方池闭环（纯逻辑·不碰模型/场景/vesselmodels）")

    # S1 官方划分 + 两层泄漏 → 必须给出 577 / 563（与主线表同分母的根）
    tr, te = RO.official_split(None)
    lk_tr, lk_va, detail = leak_keys(["manifest_hocr_200.json"])
    teS = set(te)
    clean, strict = teS - lk_tr, teS - lk_tr - lk_va
    chk("S1 官方划分 1400/600", len(tr) == 1400 and len(te) == 600)
    chk("S2 两层泄漏 → clean 577 / strict 563",
        len(clean) == RO.EXPECT_CLEAN_HOCR and len(strict) == RO.EXPECT_STRICT_HOCR,
        f"clean={len(clean)} strict={len(strict)} · {detail}")

    # S3 调参池（官方训练）与报数池（strict）必须【零重叠】——反稻草人的关键前提
    tune = set(stride_sample(tr, 100))
    chk("S3 调参池(训练抽样) ∩ 报数池(strict) = 0", not (tune & strict), f"|tune|={len(tune)}")

    # S4 配置网格：方法缺网格要 fail-closed（免静默跑没扫过的默认点）
    grid = sweep_grid()
    cfgs = build_configs(["vo", "cbf"], ["rl", "full"], ["colregs"], grid)
    n_expect = (len(grid["vo"]) + len(grid["cbf"])) * 2
    chk("S4 配置数 = (VO网格+CBF网格)×两个箱", len(cfgs) == n_expect, f"{len(cfgs)}（期望 {n_expect}）")
    chk("S4b 配置键唯一（键会当 json 主键·重了就静默互相覆盖）",
        len({c['tag'] for c in cfgs}) == len(cfgs))
    try:
        build_configs(["nosuch"], ["rl"], ["colregs"], grid)
        chk("S4c 未知方法 fail-closed", False)
    except SystemExit:
        chk("S4c 未知方法 fail-closed", True)
    try:
        build_configs(["vo"], ["nosuchbox"], ["colregs"], grid)
        chk("S4d 未知动作箱 fail-closed", False)
    except SystemExit:
        chk("S4d 未知动作箱 fail-closed", True)

    # S5 Pareto/选择逻辑：手造一组带明确支配关系的行
    A = {"tag": "A", "碰撞率%": 0.0, "违规次数/局": 1.0, "到达率%": 50.0}    # 三轴都最好 → 支配 B
    B = {"tag": "B", "碰撞率%": 1.0, "违规次数/局": 2.0, "到达率%": 40.0}
    C = {"tag": "C", "碰撞率%": 5.0, "违规次数/局": 0.5, "到达率%": 80.0}    # 违规/到达更好、碰撞更差 → 与 A 互不支配
    chk("S5 支配判定（A 支配 B · A 与 C 互不支配）",
        dominates(A, B) and not dominates(B, A) and not dominates(A, C) and not dominates(C, A))
    chk("S5b Pareto 前沿 = {A, C}", {r["tag"] for r in pareto_front([A, B, C])} == {"A", "C"})
    keep, dropped = select_configs([A, B, C], 2)
    chk("S5c 选择：单轴极值优先 → 取到 A 与 C（B 被支配·不入选）",
        {r["tag"] for r in keep} == {"A", "C"} and not dropped, f"keep={[r['tag'] for r in keep]}")
    keep1, drop1 = select_configs([A, B, C], 1)
    chk("S5d 超额被丢掉的配置会【报出来】（不许静默截断）",
        len(keep1) == 1 and len(drop1) >= 1, f"keep={[r['tag'] for r in keep1]} dropped={[r['tag'] for r in drop1]}")
    # 缺指标的行（某配置崩了/没跑出数）**不许**被当成"最好"选进报数阶段——否则会拿一个空行去代表基线
    chk("S5e 缺指标的行不会被选中（既非极值也不在前沿）",
        {r["tag"] for r in select_configs([A, {"tag": "N"}], 2)[0]} == {"A"})

    # S6 等距抽样确定性 + 边界
    chk("S6 等距抽样确定性/边界",
        stride_sample(list(range(10)), 3) == stride_sample(list(range(10)), 3)
        and len(stride_sample(list(range(10)), 3)) == 3
        and stride_sample([1, 2], 5) == [1, 2] and stride_sample([1, 2], 0) == [1, 2])

    # S7 配置键格式稳定（参数按名排序 ⟹ 顺序无关）
    chk("S7 配置键与参数书写顺序无关",
        config_tag("vo", "rl", "colregs", {"tau": 60, "margin": 0})
        == config_tag("vo", "rl", "colregs", {"margin": 0, "tau": 60}),
        config_tag("vo", "rl", "colregs", {"tau": 60, "margin": 0}))

    # S8 分型函数是【复用】reeval_official 的那一个（不是本文件重写的）——判据只准有一处（`03` L215-D）
    chk("S8 分型走 reeval_official.classify_pool", callable(getattr(RO, "classify_pool", None)))
    chk("S8b manifest 标注模式按池序映射",
        RO.classify_pool([None, None], [1, 2], {1: "对遇", 2: "交叉"}) == {0: "对遇", 1: "交叉"})
    chk("S8c 标注里没有的键 → unknown（不静默丢局）",
        RO.classify_pool([None], [999], {1: "对遇"}) == {0: "unknown"})

    print("  " + ("✅ selftest 通过（口径/分母/选择逻辑均对·**闭环仍须服务器冒烟才算真验**·`03` L202）"
                  if ok else "🔴 有洞"))
    return 0 if ok else 1


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "--selftest"
    if mode == "--run":
        sys.exit(main())
    sys.exit(selftest())
