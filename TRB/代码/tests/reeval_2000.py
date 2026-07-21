#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""修法A（well_B=200/well_X=0）在【2000 场景 held-out】上的去噪重评（eval-only·无训练·无 GPU·`03` L94）。
目的：把弱种子的 60 场景到达率（±15-30pt eval 噪声）去噪到真实分布——看 58-92 的 spread 有多少是测量噪声、
多少是真·策略差异（决定残余②终端精度有多大、要不要再上靶向攻法）。

机制（按已跑通的 reeval_lastmile.py 范式）：
  · PPO.load 载连续臂 checkpoint（**非 load_sac_for_eval·后者对 PPO 崩**·02 残余④/run_step4e:469 地雷）
  · VecNormalize.load 重建 obs_transform（eval 用·training=False）
  · evaluate_continuous(traj_idxs=None) → 每局总记 reached/term_flags/end_state/goal_geom（不存重 traj）
  · held-out = 全 2000 池 − 140 训练场景（各种子同一 split·split_seed=0）→ 干净泛化·去噪

判读：逐种子到达率（大 N·二项 CI 窄）+ Mean/IQM + 跨种子自助 CI + 失败分解（刹停%/超时%/|e_cross|·查残余②是否仍"停带外"）。

用法（**在 ~/trb 根目录下跑**·STEP4E_CODE_DIR=代码·脚本本体放 代码/tests/·随代码同步上服务器）：
  ⚠️ 必从 ~/trb 根跑（不是从 代码/）——CKDIRS/OUT 是相对 cwd 的 结果/ 路径·从根才对。
  环境变量：
    STEP4E_CODE_DIR  代码目录（**服务器设 代码**·让 import run_step4e/trb_env 找得到；默认 '.'）
    REEVAL_CKDIRS    checkpoint 目录（冒号分隔·默认 结果/checkpoints 优先[服务器扁平]·再退本地分批布局）
    REEVAL_SEEDS     种子（默认 0,1,2,3,4）
    REEVAL_ARMS      臂（默认 wx0,wx200=两臂都评·去噪 A/B；到达率不进梯度·重评纯测量·`03` L94⑧-续）
    REEVAL_HELDOUT_N held-out 上限（默认 0=全部 1860）；SMOKE 用小数验 harness
    REEVAL_SMOKE     =1 → HELDOUT_N=30 + SEEDS 取第一颗（两臂·~2min 验 harness/路径/池可用）
    REEVAL_OUT       结果 json（默认 结果/reeval_2000_xtrack_ab.json·相对 cwd=~/trb）
  smoke： cd ~/trb && STEP4E_SDIR=$HOME/trb/scenarios STEP4E_CODE_DIR=代码 REEVAL_SMOKE=1 python 代码/tests/reeval_2000.py
  全量： cd ~/trb && STEP4E_SDIR=$HOME/trb/scenarios STEP4E_CODE_DIR=代码 python 代码/tests/reeval_2000.py
"""
import sys, os, json, math, statistics as st
import warnings; warnings.filterwarnings("ignore")

_CODE = os.environ.get("STEP4E_CODE_DIR", ".")
sys.path.insert(0, _CODE)
os.environ.setdefault("STEP4E_SDIR", os.path.expanduser("~/trb/scenarios"))

import numpy as np
import run_step4e as R
from trb_env.usv_scenarios import load_scenario_pool
from trb_env.usv_continuous_shield import ContinuousProjectionEnv
from trb_env.evaluate import evaluate_continuous
from trb_env.train import make_obs_transform
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# ---------- 配置 ----------
POOL = int(os.environ.get("STEP4E_POOL", "2000"))
N_TOTAL = int(os.environ.get("STEP4E_NTOTAL", "200"))
TEST_FRAC = float(os.environ.get("STEP4E_TEST_FRAC", "0.3"))
SPLIT_SEED = int(os.environ.get("STEP4E_SPLIT_SEED", "0"))
SMOKE = os.environ.get("REEVAL_SMOKE", "0") == "1"
SEEDS = [int(s) for s in os.environ.get("REEVAL_SEEDS", "0,1,2,3,4").split(",")]
ARMS = os.environ.get("REEVAL_ARMS", "wx0,wx200").split(",")   # ARMS[0]=对照臂 / ARMS[1]=处理臂·两臂都评=2000去噪 A/B（到达率不进梯度·重评纯测量·`03` L94）
# checkpoint 名模板（{seed}/{arm} 占位）——切换两套 A/B：
#   well_X A/B(0626): Continuous-safe_s{seed}_diagXT_wb200_{arm}_s{seed}  ARMS=wx0,wx200
#   头条 well_B A/B(0625): Continuous-safe_s{seed}_diagAB{arm}_s{seed}     ARMS=wb0,wb200
CKPT_TMPL = os.environ.get("REEVAL_CKPT_TMPL", "Continuous-safe_s{seed}_diagXT_wb200_{arm}_s{seed}")
ARM_LABELS = os.environ.get("REEVAL_ARM_LABELS", "")   # 可选·逗号分隔·与 ARMS 对应的中文名（仅显示用）
HELDOUT_N = int(os.environ.get("REEVAL_HELDOUT_N", "0"))
OUT = os.environ.get("REEVAL_OUT", "reeval_2000_xtrack_ab.json")   # 默认写 cwd（=~/trb 根·server 无 结果/ 这层）
CKDIRS = os.environ.get("REEVAL_CKDIRS",   # server 布局=~/trb/结果0626-*/checkpoints（无 结果/ 父层）优先·再退本地嵌套 结果/结果0626-* 与扁平 结果/checkpoints
    "结果0626-巩固修复失败1次/checkpoints:结果0626-第2次修复/checkpoints:"
    "结果/checkpoints:结果/结果0626-巩固修复失败1次/checkpoints:结果/结果0626-第2次修复/checkpoints").split(":")
if SMOKE:
    HELDOUT_N = 30; SEEDS = SEEDS[:1]
    print("【SMOKE】HELDOUT_N=30·单种子·两臂·验 harness/路径/池可用", flush=True)
R_LAT = 80.0

# ---------- held-out 场景 = 全池 − 训练 140 ----------
train_ids, test_ids = R.make_split(N_TOTAL, TEST_FRAC, SPLIT_SEED, pool_size=POOL)
heldout_ids = sorted(set(range(POOL)) - set(train_ids))      # 1860（含 60 test + 1800 未采样）·各种子同一 split
if HELDOUT_N > 0:
    heldout_ids = heldout_ids[:HELDOUT_N]
print(f"[reeval2000] 训练 {len(train_ids)} 留出请求 {len(heldout_ids)}（全池 {POOL}·split_seed={SPLIT_SEED}）", flush=True)
ho_paths, fails = R._download(heldout_ids)
print(f"[reeval2000] held-out 实际下载 {len(ho_paths)}/{len(heldout_ids)}（缺 {len(fails)}）", flush=True)
if len(ho_paths) < 0.5 * len(heldout_ids):
    print("⚠️⚠️ 下载到的 held-out 远少于请求——服务器场景池可能不全 2000·去噪受限·请确认 STEP4E_SDIR/池", flush=True)
pool = load_scenario_pool(ho_paths)
N = len(pool)
print(f"[reeval2000] held-out pool N={N}（二项 SE@p=0.77 ≈ {100*math.sqrt(0.77*0.23/max(N,1)):.2f}pt vs 60场景 ≈ 5.4pt）\n", flush=True)
if os.environ.get("REEVAL_DOWNLOAD_ONLY") == "1":   # 仅 prime 场景缓存后退出（并行多 screen 前先跑一次·避免同时抢下同批场景·`03` L94）
    print(f"[reeval2000] DOWNLOAD_ONLY=1 → {N} 场景已缓存·退出（供并行 prime）", flush=True)
    sys.exit(0)

def ecross(e):
    gg = e.get("goal_geom"); es = e.get("end_state")
    if not gg or not es: return None
    cx, cy = gg["center"]; ori = gg["rect_orientation"]
    dx, dy = es["px"]-cx, es["py"]-cy
    return abs(dx*(-math.sin(ori)) + dy*math.cos(ori))

def find_ckpt(seed, arm):
    name = CKPT_TMPL.format(seed=seed, arm=arm)
    for d in CKDIRS:
        base = os.path.join(d, name)
        if os.path.exists(base + ".zip"):
            return base
    return None

def iqm(x):
    xs = np.sort(np.asarray(x, float)); k = len(xs); lo = int(math.floor(0.2*k))
    return float(xs[lo:k-lo].mean()) if k-2*lo > 0 else float(xs.mean())

def eval_model(base):
    """加载冻结模型 + VecNorm → 确定性 eval（纯测量·不训练·到达率不进梯度）→ 指标 dict。"""
    model = PPO.load(base + ".zip", device="cpu")
    _bv = DummyVecEnv([lambda: ContinuousProjectionEnv(*pool[0])])
    _vn = VecNormalize.load(base + "_vecnorm.pkl", _bv); _vn.training = False
    tf = make_obs_transform(_vn)
    agg, per = evaluate_continuous(lambda sc, pp: ContinuousProjectionEnv(sc, pp), model, pool,
                                   obs_transform=tf, traj_idxs=None)   # 不存重 traj·但 term_flags/end_state/goal_geom 总记
    _bv.close()
    reached = sum(1 for e in per if e["reached"])
    fails_e = [e for e in per if not e["reached"]]
    timeout = sum(1 for e in fails_e if (e.get("term_flags") or {}).get("time"))
    stopped = sum(1 for e in fails_e if (e.get("term_flags") or {}).get("stopped"))
    coll = sum(1 for e in per if e["collided"])
    viol = st.mean([e.get("violations", 0) for e in per])
    ec = [v for v in (ecross(e) for e in fails_e) if v is not None]
    ec_arr = np.array(ec) if ec else np.array([])
    near = int((ec_arr <= R_LAT).sum()) if len(ec_arr) else 0
    far = int((ec_arr > R_LAT).sum()) if len(ec_arr) else 0
    arrival = 100.0*reached/N
    se = 100*math.sqrt(arrival/100*(1-arrival/100)/N)
    return dict(N=N, arrival=arrival, se=se, reached=reached, nfail=len(fails_e),
                timeout=timeout, stopped=stopped, coll=100.0*coll/N, viol=viol,
                ec_med=(float(np.median(ec_arr)) if len(ec_arr) else None), near=near, far=far)

# ---------- 逐(种子,臂)重评 ----------
_lbls = [x for x in ARM_LABELS.split(",") if x] if ARM_LABELS else []
ARM_NAME = {a: (_lbls[i] if i < len(_lbls) else a) for i, a in enumerate(ARMS)}   # 显示名（REEVAL_ARM_LABELS 给·否则用臂串）
rows = {}   # (seed, arm) -> metrics
for s in SEEDS:
    for arm in ARMS:
        base = find_ckpt(s, arm)
        if base is None:
            print(f"s{s} {arm}: ❌ 未找到 checkpoint（在 {CKDIRS}）·跳过", flush=True); continue
        m = eval_model(base)
        rows[(s, arm)] = m
        print(f"s{s} {arm:>5}: 到达 {m['arrival']:5.2f}% ±{m['se']:.2f}(二项) | 失败 {m['nfail']:4d} 超时 {m['timeout']} 刹停 {m['stopped']} | "
              f"|e_cross|中位 {m['ec_med'] if m['ec_med'] is None else round(m['ec_med'],1)} 近带(≤80) {m['near']} 远 {m['far']} | "
              f"碰撞 {m['coll']:.2f}% 违规/局 {m['viol']:.2f}", flush=True)

# ---------- 聚合：逐臂分布 + 去噪 A/B ----------
rng = np.random.default_rng(0)
def boot_ci(vals, fn, B=10000):
    a = np.asarray(vals, float)
    s = np.array([fn(a[rng.integers(0, len(a), len(a))]) for _ in range(B)])
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))

for arm in ARMS:
    ss = sorted(s for (s, a) in rows if a == arm)
    if len(ss) < 2:
        continue
    arr = np.array([rows[(s, arm)]["arrival"] for s in ss])
    print(f"\n=== {ARM_NAME.get(arm, arm)} 在 {N} 场景 held-out 的真实分布（{len(ss)} 种子: {ss}）===")
    print(f"  逐种子: {[round(rows[(s,arm)]['arrival'],1) for s in ss]}")
    print(f"  Mean {arr.mean():.1f}  IQM {iqm(arr):.1f}  std(ddof1) {arr.std(ddof=1):.1f}  min {arr.min():.1f}")
    cm = boot_ci(arr, np.mean); ci = boot_ci(arr, iqm)
    print(f"  跨种子自助CI: Mean [{cm[0]:.1f},{cm[1]:.1f}]  IQM [{ci[0]:.1f},{ci[1]:.1f}]")

# 去噪 A/B（对照臂 ARMS[0] vs 处理臂 ARMS[1]·两臂都有的种子配对·主窗口对照 60 场景旧值判读）
if len(ARMS) >= 2:
    ctrl, treat = ARMS[0], ARMS[1]
    paired = sorted(s for s in {s for (s, a) in rows} if (s, ctrl) in rows and (s, treat) in rows)
    if len(paired) >= 2:
        a0 = np.array([rows[(s, ctrl)]["arrival"] for s in paired])
        a2 = np.array([rows[(s, treat)]["arrival"] for s in paired])
        d = a2 - a0
        print(f"\n=== 去噪 A/B：{ARM_NAME.get(treat,treat)}(处理) vs {ARM_NAME.get(ctrl,ctrl)}(对照)（{N} 场景·配对种子 {paired}）===")
        print(f"  {ctrl:>6}(对照): {[round(x,1) for x in a0]}  Mean {a0.mean():.1f} IQM {iqm(a0):.1f} std {a0.std(ddof=1):.1f}")
        print(f"  {treat:>6}(处理): {[round(x,1) for x in a2]}  Mean {a2.mean():.1f} IQM {iqm(a2):.1f} std {a2.std(ddof=1):.1f}")
        print(f"  逐种子Δ: {[round(x,1) for x in d]}  Mean diff {d.mean():+.1f}  IQM diff {iqm(a2)-iqm(a0):+.1f}  ({int((d>0).sum())}升/{int((d<0).sum())}降)")
        try:
            from scipy import stats as _sp
            print(f"  sign binom p={_sp.binomtest(int((d>0).sum()), len(d)).pvalue:.3f}  Wilcoxon p={_sp.wilcoxon(a2,a0).pvalue:.3f}  配对t p={_sp.ttest_rel(a2,a0).pvalue:.3f}")
        except Exception as _e:
            print(f"  (scipy 检验跳过: {_e})")
        print(f"  leave-one-out mean-diff:", {f"去s{paired[i]}": round((np.delete(a2,i)-np.delete(a0,i)).mean(),1) for i in range(len(paired))})

if not rows:
    print(f"\n❌ 未评到任何模型（checkpoint 全未找到）。请确认 server 上有 checkpoint"
          f"（名模板 `{CKPT_TMPL}`.zip+_vecnorm.pkl·在 REEVAL_CKDIRS={CKDIRS} 之一）。", flush=True)
os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)   # OUT 父目录不存在则建（server 上 结果/ 可能未建·防 FileNotFoundError）
json.dump({"N": N, "arms": ARMS, "rows": {f"s{s}_{a}": m for (s, a), m in rows.items()},
           "heldout_requested": len(heldout_ids), "heldout_downloaded": len(ho_paths)},
          open(OUT, "w"), ensure_ascii=False, indent=1)
print(f"\n[reeval2000] 完成 → {OUT}", flush=True)
