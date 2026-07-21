#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""量化 bang-bang：失败 vs 成功 的终端控制质量（饱和率/符号反转率）。
判别 bang-bang→失败 是否因果：若成功终端更平滑、失败才 bang-bang → 因果成立。"""
import sys, os, math
sys.path.insert(0, '.')
import warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("STEP4E_SDIR", "/tmp/trb_scenarios_pool")
import numpy as np
import run_step4e as R
from trb_env.usv_scenarios import load_scenario_pool
from trb_env.usv_continuous_shield import ContinuousProjectionEnv
from trb_env.usv_env import A_NORMAL_ACCEL_MAX, A_NORMAL_OMEGA_MAX
from trb_env.train import make_obs_transform
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

print(f"动作箱: a∈±{A_NORMAL_ACCEL_MAX} ω∈±{A_NORMAL_OMEGA_MAX}")
_, test_ids = R.make_split(200, 0.3, 0, pool_size=2000)
test_paths, _ = R._download(test_ids)
pool = load_scenario_pool(test_paths); N=len(pool)
CKDIRS = ["../结果/结果0625-奖励改造第2次/checkpoints", "../结果0625-奖励改造第2次/checkpoints"]
def find_ck(s):
    for d in CKDIRS:
        b=f"{d}/Continuous-safe_s{s}_diagABwb200_s{s}"
        if os.path.exists(b+".zip"): return b
    return None

def metrics(acts):
    """acts: list[[a,ω]] → 饱和率(|a|>0.9max) + 符号反转率(连续步符号翻转)。"""
    if len(acts)<2: return None
    A=np.array(acts)
    sat_a = float((np.abs(A[:,0])>0.9*A_NORMAL_ACCEL_MAX).mean())
    sat_w = float((np.abs(A[:,1])>0.9*A_NORMAL_OMEGA_MAX).mean())
    # 符号反转率：相邻步动作符号翻转的比例
    sa=np.sign(A[:,0]); sw=np.sign(A[:,1])
    rev_a=float((sa[1:]*sa[:-1]<0).mean()); rev_w=float((sw[1:]*sw[:-1]<0).mean())
    return sat_a,sat_w,rev_a,rev_w

# 累积 失败 vs 成功 的 终端(末10) 与 全程 控制质量
buckets={("fail","term"):[], ("fail","full"):[], ("succ","term"):[], ("succ","full"):[]}
for s in range(5):
    base=find_ck(s)
    if not base: continue
    model=PPO.load(base+".zip", device="cpu")
    _bv=DummyVecEnv([lambda: ContinuousProjectionEnv(*pool[0])])
    _vn=VecNormalize.load(base+"_vecnorm.pkl",_bv); _vn.training=False
    tf=make_obs_transform(_vn)
    for i in range(N):
        env=ContinuousProjectionEnv(*pool[i]); obs,_=env.reset(seed=0)
        acts=[]; reached=False
        for t in range(10000):
            a,_=model.predict(tf(obs),deterministic=True)
            obs,r,term,trunc,info=env.step(np.asarray(a,dtype=float))
            acts.append([float(info["u_desired"][0]),float(info["u_desired"][1])])  # 策略原始动作
            if bool(info["flags"]["goal"]): reached=True
            if term or trunc: break
        cat="succ" if reached else "fail"
        mf=metrics(acts); mt=metrics(acts[-10:])
        if mf: buckets[(cat,"full")].append(mf)
        if mt: buckets[(cat,"term")].append(mt)
    _bv.close()

print("\n"+"="*92)
print("【bang-bang 量化】策略原始动作 u_desired 的 饱和率 / 符号反转率（越高=越 bang-bang）")
print("="*92)
print(f"{'分组':<16}{'n':>5}{'饱和率a':>9}{'饱和率ω':>9}{'符号反转a':>11}{'符号反转ω':>11}")
for cat in ["succ","fail"]:
    for span in ["full","term"]:
        b=buckets[(cat,span)]
        if not b: continue
        B=np.array(b)
        lbl = ("成功" if cat=="succ" else "失败")+("·全程" if span=="full" else "·终端末10")
        print(f"{lbl:<16}{len(b):>5}{B[:,0].mean():>9.2f}{B[:,1].mean():>9.2f}{B[:,2].mean():>11.2f}{B[:,3].mean():>11.2f}")
print("\n解读：饱和率≈1 + 符号反转率≈0.5 = 满程交替 = bang-bang。对比 成功·终端 vs 失败·终端。")
