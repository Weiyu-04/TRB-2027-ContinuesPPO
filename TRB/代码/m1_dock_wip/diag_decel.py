#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""诊断：策略到底有没有学会「①遇他船减速避让 ②接近目标减速」？（user 2026-07-18 问）

测法（纯 eval·只读 ckpt·真 40 测试集）：
  ① 遇他船减速 = 速度 vs COLREGs 冲突态 rho（0=无冲突 · 1-4=各相遇态势 · 5=紧急）
     —— 若 rho 升高时速度显著下降 = 学会了「见船减速」。
  ② 近门减速 = 速度 vs 离目标距离分箱 —— 若距离缩小时速度单调下降 = 学会了「进港减速」。
对照 金标(从零) vs 热启动，看热启动有没有把这两件事做得更好/更差。

用法：STEP4E_BALANCED_DIR=<...> STEP4E_SDIR=<...> python diag_decel.py
"""
import os, sys, math, statistics, numpy as np
from collections import defaultdict
ROOT = '/Users/weiyutang/Desktop/TRB'
sys.path.insert(0, ROOT + '/代码')
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from trb_env.train import make_obs_transform
from trb_env.usv_continuous_shield import ContinuousProjectionEnv
from trb_env.usv_scenarios import load_scenario_pool
from run_step4e import load_manifest_split

SEEDS = [int(x) for x in os.environ.get('SEEDS', '0 1 2 3 4 5 6 7 8 9').split()]
ARMS = {
    '金标(从零5M)': (f"{ROOT}/结果/结果0710-22:00-10种子最优方案/checkpoints",
                     "Continuous-safe_s{s}_L1rateON_ppo_s{s}"),
    '热启动':        (f"{ROOT}/结果/结果0717-22:04-热启动-没测试完-3M/checkpoints",
                     "Continuous-safe_s{s}_wsHOCRppo_s{s}"),
}
_t, tp, _ = load_manifest_split(f"{ROOT}/balanced_pool/manifest_hocr_200.json", f"{ROOT}/balanced_pool")
pool = load_scenario_pool(tp)
mk = lambda sc, pp: ContinuousProjectionEnv(sc, pp, shield=True)

GOAL_BINS = [(0, 100), (100, 200), (200, 400), (400, 800), (800, 1600), (1600, 1e9)]


def nearest_obst(env, ego):
    """本船到最近他船中心的距离（用 state_at_time·障碍是 DynamicObstacle）。"""
    d = []
    for o in (env._obstacles or []):
        try:
            st = o.state_at_time(env.env.time_step)
            if st is not None:
                d.append(float(np.hypot(ego.position[0] - st.position[0], ego.position[1] - st.position[1])))
        except Exception:
            pass
    return min(d) if d else float('nan')


def run_arm(ck_dir, tmpl, label):
    by_rho = defaultdict(list)       # rho -> [速度]
    by_goal = defaultdict(list)      # 距离分箱 -> [速度]
    v_first_conflict, v_before = [], []   # 冲突【首次出现】前后的速度（配对）
    _bv = DummyVecEnv([lambda: mk(pool[0][0], pool[0][1])])
    for s in SEEDS:
        ck = os.path.join(ck_dir, tmpl.format(s=s))
        if not os.path.exists(ck + '.zip'):
            continue
        vn = VecNormalize.load(ck + '_vecnorm.pkl', _bv); vn.training = False
        tf = make_obs_transform(vn); model = PPO.load(ck + '.zip', device='cpu')
        for sc, pp in pool:
            env = mk(sc, pp); obs, _ = env.reset(seed=0)
            prev_v, seen_conflict = None, False
            for _ in range(200):
                a, _x = model.predict(tf(obs), deterministic=True)
                ego = env._ego_vs(); rho = int(env._rho)
                v = float(getattr(ego, 'velocity', 0.0))
                dg = float(np.hypot(ego.position[0] - env.env.goal_center[0],
                                    ego.position[1] - env.env.goal_center[1]))
                by_rho[rho].append(v)
                for lo, hi in GOAL_BINS:
                    if lo <= dg < hi:
                        by_goal[(lo, hi)].append(v); break
                if rho != 0 and not seen_conflict and prev_v is not None:
                    seen_conflict = True; v_before.append(prev_v); v_first_conflict.append(v)
                prev_v = v
                obs, _r, te, tr, info = env.step(np.asarray(a, dtype=float))
                fl = info.get('flags', {})
                if fl.get('goal') or fl.get('collision') or te or tr:
                    break
    print(f"\n{'='*66}\n【{label}】 {len(SEEDS)} 个种子 × {len(pool)} 场景")
    print("\n① 遇他船会不会减速？（按 COLREGs 冲突态分组的平均速度）")
    names = {0: '无冲突', 1: '对遇', 2: '交叉-让路', 3: '交叉-直航', 4: '追越', 5: '紧急'}
    for r in sorted(by_rho):
        v = by_rho[r]
        if len(v) < 20: continue
        print(f"    rho={r} {names.get(r,'?'):>8}  平均速度 {statistics.mean(v):5.2f} m/s  (n={len(v):>6} 步)")
    print("\n② 接近目标会不会减速？（按离目标距离分箱的平均速度）")
    for k in GOAL_BINS:
        v = by_goal[k]
        if not v: continue
        hi = '∞' if k[1] > 1e8 else str(k[1])
        print(f"    离目标 {k[0]:>4}-{hi:<5}m  平均速度 {statistics.mean(v):5.2f} m/s  (n={len(v):>6} 步)")
    if v_before:
        d = [b - a for b, a in zip(v_before, v_first_conflict)]
        print(f"\n③ 冲突【刚出现】那一步的配对变化：平均 {statistics.mean(d):+.3f} m/s "
              f"(正=减速·n={len(d)} 局)")
        try:
            from scipy.stats import wilcoxon
            nz = [x for x in d if x != 0]
            if len(nz) > 5:
                print(f"    Wilcoxon p={wilcoxon(nz).pvalue:.2e}  → {'显著减速' if statistics.mean(d)>0 else '显著加速'}")
        except Exception:
            pass
    return by_rho, by_goal


for label, (cd, tm) in ARMS.items():
    run_arm(cd, tm, label)
