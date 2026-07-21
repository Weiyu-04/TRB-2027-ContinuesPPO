#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""两靶重叠度：每个失败 episode 同看 关盾(盾塑偏靶) vs 抹平(抖动靶) 能否救→交叉统计。
重叠大=两修法部分冗余(治一个够)·不重叠=互补(合更好)。⚠️ eval 时现有策略·训练时近似。"""
import sys, os
sys.path.insert(0, '.')
import warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("STEP4E_SDIR", "/tmp/trb_scenarios_pool")
import numpy as np
import run_step4e as R
from trb_env.usv_scenarios import load_scenario_pool
from trb_env.usv_continuous_shield import ContinuousProjectionEnv
from trb_env.train import make_obs_transform
from trb_env.usv_env import A_NORMAL_ACCEL_MAX, A_NORMAL_OMEGA_MAX
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

_, test_ids = R.make_split(200,0.3,0,pool_size=2000)
paths,_=R._download(test_ids); pool=load_scenario_pool(paths); N=len(pool)
UBOX=np.array([A_NORMAL_ACCEL_MAX,A_NORMAL_OMEGA_MAX])
CKDIRS=["../结果/结果0625-奖励改造第2次/checkpoints","../结果0625-奖励改造第2次/checkpoints"]
def find_ck(s):
    for d in CKDIRS:
        b=f"{d}/Continuous-safe_s{s}_diagABwb200_s{s}"
        if os.path.exists(b+".zip"): return b
    return None
def rollout(env, model, tf, mode):
    obs,info=env.reset(seed=0); reached=collided=False; u_prev=None
    for t in range(10000):
        a,_=model.predict(tf(obs),deterministic=True); a=np.asarray(a,dtype=float)
        if mode=="smooth" and u_prev is not None: a=0.7*u_prev+0.3*a   # ema0.7(最灵)
        u_prev=a.copy()
        if mode=="shieldoff":
            obs,r,term,trunc,info=env.env.step(a)
        else:
            obs,r,term,trunc,info=env.step(a)
        reached=reached or bool(info["flags"]["goal"]); collided=collided or bool(info["flags"]["collision"])
        if term or trunc: break
    return reached, collided

both=onlyS=onlyW=neither=0; tot=0
for s in range(5):
    base=find_ck(s)
    if not base: continue
    model=PPO.load(base+".zip",device="cpu")
    _bv=DummyVecEnv([lambda:ContinuousProjectionEnv(*pool[0])]); _vn=VecNormalize.load(base+"_vecnorm.pkl",_bv);_vn.training=False
    tf=make_obs_transform(_vn)
    for i in range(N):
        r_on,_=rollout(ContinuousProjectionEnv(*pool[i]),model,tf,"normal")
        if r_on: continue
        tot+=1
        rS,_=rollout(ContinuousProjectionEnv(*pool[i]),model,tf,"shieldoff")   # 关盾救?
        rW,cW=rollout(ContinuousProjectionEnv(*pool[i]),model,tf,"smooth")      # 抹平救?(洁净才算)
        rW=rW and not cW
        if rS and rW: both+=1
        elif rS: onlyS+=1
        elif rW: onlyW+=1
        else: neither+=1
    _bv.close()

print("="*72)
print(f"【两靶重叠度】修法A 失败 {tot} 个 episode")
print("="*72)
print(f"  关盾∧抹平 都能救(重叠)  : {both}/{tot} ({100*both/max(tot,1):.0f}%)")
print(f"  仅关盾能救(盾塑偏专属)  : {onlyS}/{tot} ({100*onlyS/max(tot,1):.0f}%)")
print(f"  仅抹平能救(抖动专属)    : {onlyW}/{tot} ({100*onlyW/max(tot,1):.0f}%)")
print(f"  都救不了(其它残余)      : {neither}/{tot} ({100*neither/max(tot,1):.0f}%)")
union=both+onlyS+onlyW
print(f"\n  并集(至少一个能救)      : {union}/{tot} ({100*union/max(tot,1):.0f}%)")
print(f"  → 合并两修法理论天花板 = 救回这 {100*union/max(tot,1):.0f}% → 修法A 失败率{100*tot/(5*N):.0f}%·理论可降到 {100*(tot-union)/(5*N):.0f}%")
print("\n判读：重叠大(both 高)=两修法治同批·部分冗余(合并≈单个)；仅关盾/仅抹平 都高=互补(合并>单个·并集是天花板)。")
print("⚠️ eval 时现有策略·训练时近似(抹平≈训平滑策略·关盾≈训低混叠)·定性指示非定量保证。")
