#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""早接管 900m 平滑度核(补上窗口"平滑度待办·没做"·复审 D2 open 缺口)。

问题:900m 几何早接管替 RL 做终端·会不会伤【油门平滑 5.4×】这个不吃半径的结构性卖点?
测:连续 base(纯RL) vs 连续 dock(900m接管) vs 离散 dock·用 evaluate.py 同一个 _control_quality
   (last_action=盾+限幅后执行控制·仅正常箱内步·忠实论文口径)。到达局池化·健康(s0-4) vs 崩(s5,s6)。
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
from trb_env.usv_scenarios import load_scenario_pool, ShieldedUSVEnv
from trb_env.usv_env import A_NORMAL_OMEGA_MAX, A_NORMAL_ACCEL_MAX, A_ACC, A_OMEGA
from trb_env.evaluate import _control_quality
import dock_controller_v4 as DC
from run_step4e import load_manifest_split

KIND = os.environ.get('KIND', 'continuous').lower()
CKPT_DIR  = os.environ["CKPT_DIR"]
CKPT_TMPL = os.environ["CKPT_TMPL"]
SEEDS     = [int(x) for x in os.environ.get("SEEDS", "0 1 2 3 4 5 6").split()]
TAKEOVER_R= float(os.environ.get("TAKEOVER_R", "900"))
MANIFEST  = os.environ["STEP4E_MANIFEST"]
_bdir     = os.environ.get("STEP4E_BALANCED_DIR") or os.path.dirname(MANIFEST)
_tr, test_paths, _i = load_manifest_split(MANIFEST, _bdir)
pool = load_scenario_pool(test_paths)
_A_OMEGA_N = len(A_OMEGA)
def _quantize_discrete(a, w):
    ia = int(np.argmin([abs(float(a)-x) for x in A_ACC])); iw = int(np.argmin([abs(float(w)-x) for x in A_OMEGA]))
    return ia*_A_OMEGA_N + iw

def mk_env(sc, pp):
    if KIND == 'discrete': return ShieldedUSVEnv(sc, pp, colregs_weight=1.0)
    return ContinuousProjectionEnv(sc, pp, shield=True, goal_cone_half=None, goal_v_floor=2.0, augment_rho=False)

def run_ep(model, tf, sc, pp, use_dock):
    env = mk_env(sc, pp); obs, info = env.reset(seed=0)
    reached=False; applied=[]; positions=[]
    for _ in range(200):
        a_obs = tf(obs)
        if KIND=='discrete': act,_=model.predict(a_obs, action_masks=env.action_masks(), deterministic=True)
        else: act,_=model.predict(a_obs, deterministic=True)
        if use_dock:
            ego=env._ego_vs(); goal=env.env.goal_center; rho=env._rho
            dist=float(np.hypot(ego.position[0]-goal[0], ego.position[1]-goal[1]))
            if dist<=TAKEOVER_R and rho==RHO_NO_CONFLICT:
                st=[ego.position[0],ego.position[1],ego.orientation,float(getattr(ego,'velocity',0.0))]
                _u=DC.dock_controller(st,(goal[0],goal[1]),wmax=A_NORMAL_OMEGA_MAX)
                _u=np.array([float(np.clip(_u[0],-A_NORMAL_ACCEL_MAX,A_NORMAL_ACCEL_MAX)), _u[1]])
                act=_quantize_discrete(_u[0],_u[1]) if KIND=='discrete' else _u
        if KIND=='discrete': obs,_r,term,trunc,info=env.step(int(act))
        else: obs,_r,term,trunc,info=env.step(np.asarray(act,dtype=float))
        la=getattr(env.env,'last_action',None)
        if la is not None: applied.append(np.asarray(la,dtype=float))
        positions.append(env._ego_vs().position.copy())
        if info.get('flags',{}).get('goal',False): reached=True; break
        if term or trunc: break
    return reached, applied, positions

def pooled(seeds, use_dock):
    """池化到达局的 accel_incr/yaw_incr/jerk (原单位·忠实口径)。"""
    accel=[]; yaw=[]; jerk=[]
    for s in seeds:
        ck=os.path.join(CKPT_DIR,CKPT_TMPL.format(s=s))
        _bv=DummyVecEnv([lambda: mk_env(pool[0][0],pool[0][1])]); _vn=VecNormalize.load(ck+'_vecnorm.pkl',_bv); _vn.training=False
        tf=make_obs_transform(_vn)
        if KIND=='discrete':
            from sb3_contrib import MaskablePPO; model=MaskablePPO.load(ck+'.zip',device='cpu')
        else: model=PPO.load(ck+'.zip',device='cpu')
        for sc,pp in pool:
            reached,applied,positions=run_ep(model,tf,sc,pp,use_dock)
            if not reached: continue
            cq=_control_quality(applied, positions)
            if cq['accel_incr_mean'] is not None: accel.append(cq['accel_incr_mean'])
            if cq['yaw_incr_mean'] is not None: yaw.append(cq['yaw_incr_mean'])
            if cq['ctrl_jerk_norm_mean'] is not None: jerk.append(cq['ctrl_jerk_norm_mean'])
    m=lambda v: (round(float(np.mean(v)),5) if v else None, len(v))
    return dict(accel=m(accel), yaw=m(yaw), jerk=m(jerk))

# ── 崩种子名单：必须【按臂各自实测】·不能硬编码（`03` L192 B2 抓出的真 bug）────────────────
#   旧版写死 HEALTHY = s not in (5,6)——那是【连续臂】的崩模式，被原样套到离散臂上；
#   而离散臂真崩种子是 discStdW0 的 s0/s1、b1disc 的 s2/s3 → 离散"健康池"实为 3健康+2崩，
#   崩种子 RL 段更抖 → 同时把 连续÷离散 比值 和 "接管反略好" 两个结论都做大。
#   实测：崩配平后 dock 比值 5.18× → 4.63×/4.15×，且 base→dock 由"上升"翻转为"下降"。
#   现改为：从各臂自己的纯 RL(base) 到达率实测判崩（<CRASH_PCT 视为崩），或用 STEP4E_CRASH_SEEDS 显式覆盖。
CRASH_PCT = float(os.environ.get('SMOOTH_CRASH_PCT', '10'))   # 纯RL到达率 < 此值 = 崩
_env_crash = os.environ.get('SMOOTH_CRASH_SEEDS')             # 逗号分隔显式覆盖（如 "0,1"）

def _detect_crash_seeds():
    """按【本臂本 ckpt】实测纯 RL 到达率判崩·而非硬编码种子号。"""
    if _env_crash is not None:
        return sorted({int(x) for x in _env_crash.split(',') if x.strip() != ''})
    crash = []
    for s in SEEDS:
        ck = os.path.join(CKPT_DIR, CKPT_TMPL.format(s=s))
        _bv = DummyVecEnv([lambda: mk_env(pool[0][0], pool[0][1])])
        _vn = VecNormalize.load(ck + '_vecnorm.pkl', _bv); _vn.training = False
        tf = make_obs_transform(_vn)
        if KIND == 'discrete':
            from sb3_contrib import MaskablePPO; model = MaskablePPO.load(ck + '.zip', device='cpu')
        else:
            model = PPO.load(ck + '.zip', device='cpu')
        n_ok = sum(1 for sc, pp in pool if run_ep(model, tf, sc, pp, False)[0])
        pct = 100.0 * n_ok / len(pool)
        if pct < CRASH_PCT:
            crash.append(s)
        print(f"    [崩判定] s{s}: 纯RL 到达 {pct:.1f}%  {'← 崩' if pct < CRASH_PCT else ''}")
    return crash

print(f"平滑度核 | 臂={KIND} | 半径={TAKEOVER_R} | 崩判定阈值={CRASH_PCT}% (可用 SMOOTH_CRASH_SEEDS 覆盖)")
CRASH = _detect_crash_seeds()
HEALTHY = [s for s in SEEDS if s not in CRASH]
print(f"平滑度核 | 臂={KIND} | 半径={TAKEOVER_R} | 健康{HEALTHY} 崩{CRASH} | 指标=accel_incr/yaw_incr/jerk(到达局池化·原单位·仅箱内步)")
print(f"  ⚠️ 跨臂比较（连续÷离散）必须【两臂各自崩配平后】再比·否则比值被崩种子污染（`03` L192 B2）")
for tag, seeds in [("健康", HEALTHY), ("崩", CRASH)]:
    if not seeds: continue
    b=pooled(seeds, False); d=pooled(seeds, True)
    print(f"\n[{tag}种子]")
    print(f"  base(纯RL)   accel_incr={b['accel']} yaw_incr={b['yaw']} jerk={b['jerk']}")
    print(f"  dock(900m接管) accel_incr={d['accel']} yaw_incr={d['yaw']} jerk={d['jerk']}")
