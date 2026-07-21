#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""量化"几何早接管管多少活"(阶段1·诚实叙事·独立复审 user 追问"~22%开挂?")。

对 900m 早接管·每局记:总步数·接管步数·接管步占比·首次接管时离目标距离。
分崩种子(s5/s6)vs 健康(s0/s2/s7)看几何控制器在最终"到达"里承担多少。
纯 eval·逐字对齐金标 env。
"""
import os, sys, math, numpy as np
sys.path.insert(0, '/Users/weiyutang/Desktop/TRB/代码')
sys.path.insert(0, '/Users/weiyutang/Desktop/TRB/代码/m1_dock_wip')
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from trb_env.train import make_obs_transform
from trb_env.usv_continuous_shield import ContinuousProjectionEnv
from trb_env.usv_colregs import RHO_NO_CONFLICT
from trb_env.usv_scenarios import load_scenario_pool
from trb_env.usv_env import A_NORMAL_OMEGA_MAX, A_NORMAL_ACCEL_MAX
import dock_controller_v4 as DC
from run_step4e import load_manifest_split

CKPT_DIR  = os.environ["CKPT_DIR"]
CKPT_TMPL = "Continuous-safe_s{s}_L1rateON_ppo_s{s}"
SEEDS     = [int(x) for x in os.environ.get("SEEDS", "0 2 5 6 7").split()]
TAKEOVER_R = float(os.environ.get("TAKEOVER_R", "900"))
MANIFEST  = os.environ["STEP4E_MANIFEST"]
_bdir     = os.environ.get("STEP4E_BALANCED_DIR") or os.path.dirname(MANIFEST)
_tr, test_paths, _i = load_manifest_split(MANIFEST, _bdir)
pool = load_scenario_pool(test_paths)
print(f"接管占比量化 | n={len(pool)} | 种子={SEEDS} | 接管半径={TAKEOVER_R}")

def mk_env(sc, pp):
    return ContinuousProjectionEnv(sc, pp, shield=True, goal_cone_half=None, goal_v_floor=2.0, augment_rho=False)

def run_ep(model, tf, sc, pp):
    env = mk_env(sc, pp)
    obs, info = env.reset(seed=0)
    reached=False; steps=0; took=0; dist_first=None; path=0.0
    prev=None
    for _ in range(200):
        act, _ = model.predict(tf(obs), deterministic=True)
        ego = env._ego_vs(); goal = env.env.goal_center; rho = env._rho
        dist = float(np.hypot(ego.position[0]-goal[0], ego.position[1]-goal[1]))
        if dist <= TAKEOVER_R and rho == RHO_NO_CONFLICT:
            st=[ego.position[0],ego.position[1],ego.orientation,float(getattr(ego,'velocity',0.0))]
            _u = DC.dock_controller(st,(goal[0],goal[1]),wmax=A_NORMAL_OMEGA_MAX)
            _u = np.array([float(np.clip(_u[0],-A_NORMAL_ACCEL_MAX,A_NORMAL_ACCEL_MAX)), _u[1]])
            act=_u; took+=1
            if dist_first is None: dist_first=dist
        obs,_r,term,trunc,info = env.step(np.asarray(act,dtype=float))
        p = env._ego_vs().position
        if prev is not None: path += float(np.hypot(p[0]-prev[0],p[1]-prev[1]))
        prev=p; steps+=1
        if info.get('flags',{}).get('goal',False): reached=True; break
        if term or trunc: break
    return dict(reached=reached, steps=steps, took=took, dist_first=dist_first, path=path)

print(f"\n{'seed':>4} | {'到达局':>6} | 到达局里: 接管步占比 中位/均值 | 首次接管距目标 中位 | 接管路径占比 中位")
for s in SEEDS:
    ck=os.path.join(CKPT_DIR,CKPT_TMPL.format(s=s))
    _bv=DummyVecEnv([lambda: mk_env(pool[0][0],pool[0][1])]); _vn=VecNormalize.load(ck+'_vecnorm.pkl',_bv); _vn.training=False
    tf=make_obs_transform(_vn); model=PPO.load(ck+'.zip',device='cpu')
    reached_eps=[]
    for sc,pp in pool:
        r=run_ep(model,tf,sc,pp)
        if r['reached']: reached_eps.append(r)
    if not reached_eps: print(f"{s:>4} | 0 到达"); continue
    frac=[e['took']/e['steps'] for e in reached_eps]
    dfirst=[e['dist_first'] for e in reached_eps if e['dist_first'] is not None]
    # 接管期间走的路 占全程比例·近似 = took步/总步(用步数占比;若匀速则≈路径占比)
    print(f"{s:>4} | {len(reached_eps):>4}/{len(pool)} | 步占比 中位{100*np.median(frac):.0f}%/均值{100*np.mean(frac):.0f}% | "
          f"接管距目标中位{np.median(dfirst):.0f}m | (接管步数中位{np.median([e['took'] for e in reached_eps]):.0f}/总步中位{np.median([e['steps'] for e in reached_eps]):.0f})")
