#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""崩根跨种子诊断（阶段1②·独立复审）。

问题：崩种子 s5/s6 失败的【同场景】，健康 8 种子(0,1,2,3,4,7,8,9)解到 ~80% 吗？
若是 → 同奖励+同场景+同门·只种子号变 → 崩=训练不稳(种子掷骰子)·非奖励/非场景。

做法：全 10 种子 × 40 测试集·纯 RL 单独臂(无停车)·忠实镜像 run_episode_continuous·
     记录 per-(seed,scenario) reached → 交叉表。
纯 eval·不训练·不碰配方·逐字对齐金标 env(shield=True/cone=None/vfloor=2.0/augment=False/严格门)。
"""
import os, sys, math, numpy as np, json
sys.path.insert(0, '/Users/weiyutang/Desktop/TRB/代码')
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from trb_env.train import make_obs_transform
from trb_env.usv_continuous_shield import ContinuousProjectionEnv
from trb_env.usv_scenarios import load_scenario_pool
from run_step4e import load_manifest_split

CKPT_DIR  = os.environ["CKPT_DIR"]
CKPT_TMPL = os.environ.get("CKPT_TMPL", "Continuous-safe_s{s}_L1rateON_ppo_s{s}")
SEEDS     = [int(x) for x in os.environ.get("SEEDS", "0 1 2 3 4 5 6 7 8 9").split()]
MANIFEST  = os.environ["STEP4E_MANIFEST"]
_bdir     = os.environ.get("STEP4E_BALANCED_DIR") or os.path.dirname(MANIFEST)
_tr, test_paths, _i = load_manifest_split(MANIFEST, _bdir)
pool = load_scenario_pool(test_paths)
N = len(pool)
print(f"跨种子诊断 | 测试集 n={N} | 种子={SEEDS} | 严格门(位置+朝向) RL单独臂(无停车)")

def mk_env(sc, pp):
    return ContinuousProjectionEnv(sc, pp, shield=True, goal_cone_half=None, goal_v_floor=2.0, augment_rho=False)

def run_ep(model, tf, sc, pp):
    env = mk_env(sc, pp)
    obs, info = env.reset(seed=0)
    reached = False
    for _ in range(200):
        act, _ = model.predict(tf(obs), deterministic=True)
        obs, _r, term, trunc, info = env.step(np.asarray(act, dtype=float))
        if info.get('flags', {}).get('goal', False): reached = True; break
        if term or trunc: break
    return reached

# reach[s] = boolean array length N
reach = {}
for s in SEEDS:
    ck = os.path.join(CKPT_DIR, CKPT_TMPL.format(s=s))
    _bv = DummyVecEnv([lambda: mk_env(pool[0][0], pool[0][1])])
    _vn = VecNormalize.load(ck + '_vecnorm.pkl', _bv); _vn.training = False
    tf = make_obs_transform(_vn)
    model = PPO.load(ck + '.zip', device='cpu')
    arr = np.array([run_ep(model, tf, sc, pp) for sc, pp in pool], dtype=int)
    reach[s] = arr
    print(f"  s{s}: 到达 {arr.sum()}/{N} = {100*arr.mean():.1f}%", flush=True)

HEALTHY = [s for s in SEEDS if s not in (5, 6)]
crashed = [s for s in SEEDS if s in (5, 6)]
print(f"\n健康种子={HEALTHY} | 崩种子={crashed}")

# 崩种子 s5/s6 各自失败的场景集
for cs in crashed:
    fail_idx = np.where(reach[cs] == 0)[0]
    if len(fail_idx) == 0:
        print(f"s{cs} 无失败场景"); continue
    # 健康种子在这些场景的平均到达率
    healthy_on_fail = np.array([reach[h][fail_idx].mean() for h in HEALTHY])
    print(f"\ns{cs} 失败 {len(fail_idx)}/{N} 个场景 → 健康种子在【这些场景】的到达率:")
    for h, r in zip(HEALTHY, healthy_on_fail):
        print(f"    s{h}: {100*r:.1f}%")
    print(f"  → 健康种子均值 = {100*healthy_on_fail.mean():.1f}% (中位 {100*np.median(healthy_on_fail):.1f}%)")

# s5 且 s6 都失败的场景（两崩种子共同失败）
both_fail = np.where((reach[5] == 0) & (reach[6] == 0))[0] if (5 in SEEDS and 6 in SEEDS) else []
if len(both_fail):
    healthy_on_both = np.array([reach[h][both_fail].mean() for h in HEALTHY])
    print(f"\ns5&s6 共同失败 {len(both_fail)}/{N} 场景 → 健康8种子均值到达 = {100*healthy_on_both.mean():.1f}%")
    # 每个共同失败场景·有几个健康种子解开
    solved_by = np.array([reach[h][both_fail] for h in HEALTHY]).sum(axis=0)
    print(f"  这些场景里·平均每个被 {solved_by.mean():.1f}/{len(HEALTHY)} 个健康种子解开")
    print(f"  完全没有健康种子解开的场景 = {(solved_by==0).sum()}/{len(both_fail)} 个(=真难场景)")

# 落盘
OUT = os.environ.get("OUT", "")
if OUT:
    np.savez(OUT, **{f"s{s}": reach[s] for s in SEEDS})
    print(f"\n✅ per-scenario reach 落盘 {OUT}")
