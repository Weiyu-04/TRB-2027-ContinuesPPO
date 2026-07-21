#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""锁死"抖动是不是元凶"的决定性反事实（镜像关盾实验·03 L98 前置）。
把修法A 失败 episode 的【策略动作人为抹平】(限速 rate-limit/EMA)重放·看是否因此到达。
  抹平后到达↑ → 抖动是元凶 → 治抖(rate 惩罚)有效·才值得烧。
  抹平后不变 → 抖动非元凶(奖励/学习问题)→ 治抖白烧·换药。
⚠️ 这是 eval 时抹平【现有 bang-bang 策略】·近似"训练出平滑策略会怎样"(非完全等价·但抹平向【策略均值】=测均值方向对不对)。"""
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

def rollout(env, model, tf, smooth=None):
    """smooth=None 原始; smooth=('rate',f) 每步动作变化限 ±f·u_box; smooth=('ema',a) EMA。"""
    obs,info=env.reset(seed=0); reached=collided=False; u_prev=None
    for t in range(10000):
        a,_=model.predict(tf(obs),deterministic=True); a=np.asarray(a,dtype=float)
        if smooth is not None and u_prev is not None:
            if smooth[0]=='rate':
                dmax=smooth[1]*UBOX; a=u_prev+np.clip(a-u_prev,-dmax,dmax)
            elif smooth[0]=='ema':
                a=smooth[1]*u_prev+(1-smooth[1])*a
        u_prev=a.copy()
        obs,r,term,trunc,info=env.step(a)
        reached=reached or bool(info["flags"]["goal"]); collided=collided or bool(info["flags"]["collision"])
        if term or trunc: break
    return reached, collided

SMOOTHS=[("rate",0.5),("rate",0.25),("ema",0.7),("ema",0.85)]
print(f"修法A 失败 episode 抹平动作重放（限速/EMA 多档）")
agg={sm:{"arr":0,"clean":0,"coll":0} for sm in SMOOTHS}; tot_fail=0
perseed={}
for s in range(5):
    base=find_ck(s)
    if not base: continue
    model=PPO.load(base+".zip",device="cpu")
    _bv=DummyVecEnv([lambda:ContinuousProjectionEnv(*pool[0])]); _vn=VecNormalize.load(base+"_vecnorm.pkl",_bv);_vn.training=False
    tf=make_obs_transform(_vn)
    fO=0; sd={sm:0 for sm in SMOOTHS}
    for i in range(N):
        r_on,_=rollout(ContinuousProjectionEnv(*pool[i]),model,tf,None)
        if r_on: continue
        fO+=1
        for sm in SMOOTHS:
            r_sm,c_sm=rollout(ContinuousProjectionEnv(*pool[i]),model,tf,sm)
            if r_sm:
                agg[sm]["arr"]+=1; sd[sm]+=1
                if not c_sm: agg[sm]["clean"]+=1
            if c_sm: agg[sm]["coll"]+=1
    tot_fail+=fO; perseed[s]=(fO,sd)
    print(f"s{s}: 失败 {fO} → 抹平后到达 " + " | ".join(f"{sm[0]}{sm[1]}:{sd[sm]}" for sm in SMOOTHS), flush=True)
    _bv.close()

print("\n"+"="*78)
print(f"【锁死结果】修法A 失败 {tot_fail} 个·抹平动作后到达情况")
print("="*78)
for sm in SMOOTHS:
    a=agg[sm]
    print(f"  {sm[0]}({sm[1]}): 到达 {a['arr']}/{tot_fail} ({100*a['arr']/max(tot_fail,1):.0f}%) | 洁净 {a['clean']} | 抹平致碰撞 {a['coll']}")
print("\n判读：抹平后到达% 显著>0 → 抖动是元凶(治抖有效)；≈0 → 抖动非元凶(奖励/学习问题·治抖白烧)。")
print("对照锚点：关盾反事实=42% 失败变到达(L97)。抹平若也显著→抖动是真靶。")
