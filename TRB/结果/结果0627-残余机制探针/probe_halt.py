#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""机制探针：修法A 失败 episode 的刹停点【盾挡 vs 策略弱】定论。
本地 eval-only·不训练·不烧 GPU。逐步记 u_desired(策略原始)/u_applied(盾后)/source/rho/横向缺口。
判据：
  · 刹停点 source=no_obstacle 或 rho 无冲突 + 盾零修正 → 盾不活动 → 停是【策略自选】
  · 刹停点 u_desired 想朝带推(a>0 或 ω 朝带) 但 u_applied 被改 → 【盾挡】
  · 刹停点 u_desired 本身就 a≤0/不朝带 → 【策略弱·没学会推最后一程】
"""
import sys, os, json, math
sys.path.insert(0, '.')
import warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("STEP4E_SDIR", "/tmp/trb_scenarios_pool")
import numpy as np
import run_step4e as R
from trb_env.usv_scenarios import load_scenario_pool
from trb_env.usv_continuous_shield import ContinuousProjectionEnv
from trb_env.train import make_obs_transform
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

_, test_ids = R.make_split(200, 0.3, 0, pool_size=2000)
test_paths, fails = R._download(test_ids)
print(f"[probe] test 场景 {len(test_paths)}/{len(test_ids)} (缺{len(fails)})", flush=True)
pool = load_scenario_pool(test_paths)
N = len(pool)
CKDIRS = ["../结果/结果0625-奖励改造第2次/checkpoints", "../结果0625-奖励改造第2次/checkpoints"]

def find_ck(s):
    for d in CKDIRS:
        b = f"{d}/Continuous-safe_s{s}_diagABwb200_s{s}"
        if os.path.exists(b+".zip"): return b
    return None

def band_geom(env):
    """从 env 取目标矩形几何：center, 朝向, 半宽(横向), 半长(纵向)。"""
    pp = env.env.planning_problem
    gs = pp.goal.state_list[0]
    pos = gs.position
    # commonocean GoalRegion: position 可能是 Rectangle/ShapeGroup
    try:
        verts = pos.shapes[0].vertices if hasattr(pos, "shapes") else pos.vertices
    except Exception:
        verts = None
    return verts

def cross_track(px, py, gg_center, ori):
    dx, dy = px-gg_center[0], py-gg_center[1]
    return dx*(-math.sin(ori)) + dy*math.cos(ori)   # 带符号横向偏移（+/− 哪侧）

SEEDS = [int(x) for x in os.environ.get("PROBE_SEEDS", "0,1,2,3,4").split(",")]
LASTK = 8
agg = {"n_fail":0, "term_no_obstacle":0, "term_shield_active":0,
       "shield_corrected_last10":0, "policy_decel_at_halt":0,
       "policy_wants_push_blocked":0}
detail_samples = []

for s in SEEDS:
    base = find_ck(s)
    if base is None:
        print(f"s{s}: ❌ 无 checkpoint·跳过"); continue
    model = PPO.load(base+".zip", device="cpu")
    _bv = DummyVecEnv([lambda: ContinuousProjectionEnv(*pool[0])])
    _vn = VecNormalize.load(base+"_vecnorm.pkl", _bv); _vn.training=False
    tf = make_obs_transform(_vn)
    nfail_s = 0; samples_s = []
    for i in range(N):
        env = ContinuousProjectionEnv(*pool[i])
        obs, info = env.reset(seed=0)
        # 目标几何
        gg_center = None; ori = 0.0; halfw = 30.0
        try:
            from trb_env.evaluate import _goal_xy
            gg_center = _goal_xy(env)
        except Exception:
            pass
        steps_rec = []
        reached = False; term=trunc=False
        for t in range(10000):
            a_obs = tf(obs)
            action,_ = model.predict(a_obs, deterministic=True)
            obs, r, term, trunc, info = env.step(np.asarray(action,dtype=float))
            ev = env._ego_vs()
            steps_rec.append({
                "t": t, "px": float(ev.position[0]), "py": float(ev.position[1]),
                "psi": float(ev.orientation), "v": float(ev.velocity),
                "ud": [float(x) for x in info["u_desired"]],
                "ua": [float(x) for x in info["u_applied"]],
                "src": info.get("source"), "rho": info.get("rho"),
            })
            if bool(info["flags"]["goal"]): reached=True
            if term or trunc: break
        if reached:
            env.env.close() if hasattr(env.env,"close") else None
            continue
        nfail_s += 1; agg["n_fail"] += 1
        last = steps_rec[-LASTK:]
        term_step = steps_rec[-1]
        # 横向缺口
        ct = None
        if gg_center is not None:
            ct = cross_track(term_step["px"], term_step["py"], gg_center, ori)
        # 盾活动度（最后 10 步）
        last10 = steps_rec[-10:]
        shield_active_steps = sum(1 for r_ in last10 if r_["src"] not in (None,"no_obstacle"))
        corrected = sum(1 for r_ in last10 if np.linalg.norm(np.array(r_["ua"])-np.array(r_["ud"]))>1e-6)
        term_no_obs = term_step["src"] in (None,"no_obstacle")
        # 策略在刹停点想干嘛（原始 u_desired）
        ud_a = term_step["ud"][0]; ud_w = term_step["ud"][1]
        ua_a = term_step["ua"][0]
        policy_decel = ud_a <= 1e-4                     # 策略原始加速度≤0=自己想减速/停
        # 策略想推但被盾改（加速度上）：u_desired 想加速但 u_applied 被砍
        push_blocked = (ud_a > 1e-3) and (ua_a < ud_a - 1e-3)
        if term_no_obs: agg["term_no_obstacle"] += 1
        else: agg["term_shield_active"] += 1
        if corrected>0: agg["shield_corrected_last10"] += 1
        if policy_decel: agg["policy_decel_at_halt"] += 1
        if push_blocked: agg["policy_wants_push_blocked"] += 1
        if len(samples_s) < 3:   # 每种子留 3 个详细样本
            samples_s.append({"scen": i, "ct": ct, "v_term": term_step["v"],
                              "term_src": term_step["src"], "term_rho": term_step["rho"],
                              "shield_active_last10": shield_active_steps, "corrected_last10": corrected,
                              "last": last})
        env.env.close() if hasattr(env.env,"close") else None
    print(f"s{s}: 失败 {nfail_s}/{N}", flush=True)
    detail_samples.append((s, samples_s))
    _bv.close()

print("\n"+"="*90)
print("【聚合·盾 vs 策略 定论】", agg["n_fail"], "个失败 episode")
print("="*90)
nf = max(agg["n_fail"],1)
print(f"  刹停点 source=no_obstacle(盾不活动): {agg['term_no_obstacle']}/{nf} ({100*agg['term_no_obstacle']/nf:.0f}%)")
print(f"  刹停点 盾仍活动:                     {agg['term_shield_active']}/{nf} ({100*agg['term_shield_active']/nf:.0f}%)")
print(f"  最后10步盾有修正动作(‖ua-ud‖>1e-6):  {agg['shield_corrected_last10']}/{nf} ({100*agg['shield_corrected_last10']/nf:.0f}%)")
print(f"  刹停点策略原始就减速(ud_a≤0):         {agg['policy_decel_at_halt']}/{nf} ({100*agg['policy_decel_at_halt']/nf:.0f}%)  ← 策略自选停")
print(f"  刹停点策略想加速但被盾砍(push_blocked):{agg['policy_wants_push_blocked']}/{nf} ({100*agg['policy_wants_push_blocked']/nf:.0f}%)  ← 盾挡")

print("\n"+"="*90)
print("【详细样本·每种子前 3 个失败的最后 8 步】(t/位置/v/ψ | ud=策略原始[a,ω] / ua=盾后[a,ω] / src / rho)")
print("="*90)
for s, samples in detail_samples:
    for sm in samples:
        ctv = "None" if sm["ct"] is None else f"{sm['ct']:+.1f}"
        print(f"\n— s{s} 场景{sm['scen']}: 横向偏{ctv}m v_term={sm['v_term']:.3f} 末src={sm['term_src']} 末rho={sm['term_rho']} 盾活动(末10){sm['shield_active_last10']} 盾修正(末10){sm['corrected_last10']}")
        for r_ in sm["last"]:
            corr = np.linalg.norm(np.array(r_["ua"])-np.array(r_["ud"]))
            print(f"    t{r_['t']:>3} ({r_['px']:.0f},{r_['py']:.0f}) v={r_['v']:.3f} ψ={r_['psi']:+.3f} | ud[{r_['ud'][0]:+.4f},{r_['ud'][1]:+.4f}] ua[{r_['ua'][0]:+.4f},{r_['ua'][1]:+.4f}] |Δ|={corr:.4f} {r_['src']}/{r_['rho']}")
