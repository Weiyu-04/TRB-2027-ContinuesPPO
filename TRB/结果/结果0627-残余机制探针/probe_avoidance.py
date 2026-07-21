#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""挖病灶最后一层：失败的终端横向偏移是不是 COLREGs 避让造成的？
追踪 横向偏移(t) vs rho(t)/盾介入(t)/give_way_dir。判别：
  · 若横向偏移在遭遇阶段(rho≠0/盾介入)被推大、遭遇解除后合不回来 → 病灶=避让后横向航迹恢复失败(可对症+连招牌)
  · 若横向偏移从一开始就在、与遭遇无关 → 策略本身不对齐(更接近 RL 内禀/控制质量)
"""
import sys, os, math
sys.path.insert(0, '.')
import warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("STEP4E_SDIR", "/tmp/trb_scenarios_pool")
import numpy as np
import run_step4e as R
from trb_env.usv_scenarios import load_scenario_pool
from trb_env.usv_continuous_shield import ContinuousProjectionEnv
from trb_env.usv_colregs import RHO_NO_CONFLICT
from trb_env.evaluate import _goal_xy
from trb_env.train import make_obs_transform
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

print(f"RHO_NO_CONFLICT={RHO_NO_CONFLICT}")
_, test_ids = R.make_split(200, 0.3, 0, pool_size=2000)
test_paths,_=R._download(test_ids)
pool=load_scenario_pool(test_paths); N=len(pool)
CKDIRS=["../结果/结果0625-奖励改造第2次/checkpoints","../结果0625-奖励改造第2次/checkpoints"]
def find_ck(s):
    for d in CKDIRS:
        b=f"{d}/Continuous-safe_s{s}_diagABwb200_s{s}"
        if os.path.exists(b+".zip"): return b
    return None

def run_ep(env, model, tf):
    obs,_=env.reset(seed=0)
    gc=_goal_xy(env)
    rec=[]; reached=False
    for t in range(10000):
        a,_=model.predict(tf(obs),deterministic=True)
        obs,r,term,trunc,info=env.step(np.asarray(a,dtype=float))
        ev=env._ego_vs()
        ct = (ev.position[1]-gc[1]) if gc is not None else None   # rect_orient=0 → cross-track=dy
        rho=info.get("rho"); src=info.get("source")
        corr=float(np.linalg.norm(np.array(info["u_applied"])-np.array(info["u_desired"])))
        rec.append({"t":t,"ct":(None if ct is None else float(ct)),"v":float(ev.velocity),
                    "rho":rho,"src":src,"corr":corr,"gwd":info.get("give_way_dir")})
        if bool(info["flags"]["goal"]): reached=True
        if term or trunc: break
    return reached, rec, gc

# 遭遇阶段 = rho≠no_conflict 的步；判断 横向偏移 在遭遇阶段 vs 之后
agg={"fail":{"n":0,"ct_start":[],"ct_maxenc":[],"ct_encend":[],"ct_term":[],"grew_in_enc":0,"recovered_after":0,"no_encounter":0,"enc_displaced_unrecovered":0},
     "succ":{"n":0,"ct_start":[],"ct_maxenc":[],"ct_encend":[],"ct_term":[],"grew_in_enc":0,"recovered_after":0,"no_encounter":0,"enc_displaced_unrecovered":0}}
samples=[]
for s in range(5):
    base=find_ck(s)
    if not base: continue
    model=PPO.load(base+".zip",device="cpu")
    _bv=DummyVecEnv([lambda:ContinuousProjectionEnv(*pool[0])])
    _vn=VecNormalize.load(base+"_vecnorm.pkl",_bv);_vn.training=False
    tf=make_obs_transform(_vn)
    for i in range(N):
        env=ContinuousProjectionEnv(*pool[i])
        reached,rec,gc=run_ep(env,model,tf)
        if gc is None: continue
        cat="succ" if reached else "fail"
        cts=[r["ct"] for r in rec]
        enc_idx=[k for k,r in enumerate(rec) if (r["rho"]!=RHO_NO_CONFLICT) or (r["corr"]>1e-6)]  # 遭遇/盾介入步
        a=agg[cat]; a["n"]+=1
        ct_start=cts[0]; ct_term=cts[-1]
        a["ct_start"].append(abs(ct_start)); a["ct_term"].append(abs(ct_term))
        if not enc_idx:
            a["no_encounter"]+=1
            a["ct_maxenc"].append(0.0); a["ct_encend"].append(abs(ct_start))
        else:
            enc_cts=[abs(cts[k]) for k in enc_idx]
            ct_maxenc=max(enc_cts); enc_end=enc_idx[-1]; ct_encend=abs(cts[enc_end])
            a["ct_maxenc"].append(ct_maxenc); a["ct_encend"].append(ct_encend)
            # 遭遇阶段横向被推大？(遭遇内 max > 遭遇前)
            pre = abs(cts[max(0,enc_idx[0]-1)])
            if ct_maxenc > pre + 10: a["grew_in_enc"]+=1
            # 遭遇后是否合回来？(终端 < 遭遇结束时·收窄>10m)
            if abs(ct_term) < ct_encend - 10: a["recovered_after"]+=1
            # 病灶签名：遭遇推大 + 之后没合回(终端仍偏大)
            if ct_maxenc > pre+10 and abs(ct_term) > 30: a["enc_displaced_unrecovered"]+=1
        if cat=="fail" and len(samples)<6:
            samples.append((s,i,[round(c,0) if c is not None else None for c in cts],
                            [r["rho"] for r in rec], [round(r["corr"],3) for r in rec]))
    _bv.close()

def summ(a,lbl):
    n=max(a["n"],1)
    print(f"\n【{lbl}】n={a['n']}")
    print(f"  |横向偏|: 起点中位 {np.median(a['ct_start']):.0f}m → 遭遇内max中位 {np.median(a['ct_maxenc']):.0f}m → 遭遇结束中位 {np.median(a['ct_encend']):.0f}m → 终端中位 {np.median(a['ct_term']):.0f}m")
    print(f"  无遭遇(全程rho=0且盾零介入): {a['no_encounter']}/{a['n']} ({100*a['no_encounter']/n:.0f}%)")
    print(f"  遭遇内横向被推大(>10m): {a['grew_in_enc']}/{a['n']} ({100*a['grew_in_enc']/n:.0f}%)")
    print(f"  遭遇后合回来(收窄>10m): {a['recovered_after']}/{a['n']} ({100*a['recovered_after']/n:.0f}%)")
    print(f"  🎯病灶签名[遭遇推大+终端仍偏>30m未合回]: {a['enc_displaced_unrecovered']}/{a['n']} ({100*a['enc_displaced_unrecovered']/n:.0f}%)")

print("\n"+"="*92)
print("【避让因果定论】横向偏移 vs 遭遇阶段")
print("="*92)
summ(agg["fail"],"失败"); summ(agg["succ"],"成功")

print("\n"+"="*92)
print("【失败样本·横向偏移(t) 轨迹 + rho】(看横向偏在哪个阶段长出来)")
print("="*92)
for s,i,cts,rhos,corrs in samples:
    enc=[k for k in range(len(rhos)) if rhos[k]!=RHO_NO_CONFLICT or corrs[k]>1e-6]
    encr = f"步{enc[0]}-{enc[-1]}" if enc else "无"
    print(f"\n— s{s} 场景{i}: 遭遇/盾介入={encr} (共{len(cts)}步)")
    # 抽稀打印横向偏移轨迹(每5步)
    show=[(k,cts[k]) for k in range(0,len(cts),max(1,len(cts)//16))]
    print("  横向偏(每~1/16步):", " ".join(f"t{k}:{c:+.0f}" for k,c in show))
