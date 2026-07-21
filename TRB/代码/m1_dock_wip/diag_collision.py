#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""定位并拆解「热启动 + 小半径停车接管」引入的那 1 次碰撞（`03` L192 H 待查项）。

问：接管每步仍走 env.step→safe_action（盾在），且只在 rho==NO_CONFLICT 时接管，为何还会撞？
测：s5 @150m 逐场景找出碰撞局 → 逐步 dump（rho / 盾分支 source / 是否接管 / 与他船距离 / 速度）。
纯 eval·只读 ckpt。
"""
import os, sys, math, numpy as np
ROOT = '/Users/weiyutang/Desktop/TRB'
sys.path.insert(0, ROOT + '/代码')
sys.path.insert(0, ROOT + '/代码/m1_dock_wip')
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from trb_env.train import make_obs_transform
from trb_env.usv_continuous_shield import ContinuousProjectionEnv
from trb_env.usv_colregs import RHO_NO_CONFLICT
from trb_env.usv_scenarios import load_scenario_pool
from trb_env.usv_env import A_NORMAL_OMEGA_MAX, A_NORMAL_ACCEL_MAX
import dock_controller_v4 as DC
from run_step4e import load_manifest_split

TAKEOVER_R = float(os.environ.get('TAKEOVER_R', '150'))
SEED = int(os.environ.get('SEED', '5'))
CKPT = f"{ROOT}/结果/结果0717-22:04-热启动-没测试完-3M/checkpoints/Continuous-safe_s{SEED}_wsHOCRppo_s{SEED}"

_train, test_paths, _ = load_manifest_split(f"{ROOT}/balanced_pool/manifest_hocr_200.json", f"{ROOT}/balanced_pool")
pool = load_scenario_pool(test_paths)
mk = lambda sc, pp: ContinuousProjectionEnv(sc, pp, shield=True)
_bv = DummyVecEnv([lambda: mk(pool[0][0], pool[0][1])])
_vn = VecNormalize.load(CKPT + '_vecnorm.pkl', _bv); _vn.training = False
tf = make_obs_transform(_vn)
model = PPO.load(CKPT + '.zip', device='cpu')
DOCK_WMAX, DOCK_AMAX = A_NORMAL_OMEGA_MAX, A_NORMAL_ACCEL_MAX


def obst_min_dist(env):
    """本船到最近他船的距离（用 env 自己的障碍列表·失败返回 nan）。"""
    try:
        ego = env._ego_vs(); p = np.asarray(ego.position, dtype=float)
        d = []
        for o in (env._obstacles or []):
            q = np.asarray(getattr(o, 'position', None), dtype=float)
            if q is not None and q.size >= 2:
                d.append(float(np.hypot(p[0] - q[0], p[1] - q[1])))
        return min(d) if d else float('nan')
    except Exception:
        return float('nan')


def run(sc, pp, use_dock, trace=False):
    env = mk(sc, pp)
    obs, info = env.reset(seed=0)
    log = []
    for t in range(200):
        a_obs = tf(obs)
        act, _ = model.predict(a_obs, deterministic=True)
        ego = env._ego_vs(); goal = env.env.goal_center; rho = env._rho
        dist = float(np.hypot(ego.position[0] - goal[0], ego.position[1] - goal[1]))
        did = False
        if use_dock and dist <= TAKEOVER_R and rho == RHO_NO_CONFLICT:
            st = [ego.position[0], ego.position[1], ego.orientation, float(getattr(ego, 'velocity', 0.0))]
            u = DC.dock_controller(st, (goal[0], goal[1]), wmax=DOCK_WMAX)
            act = np.array([float(np.clip(u[0], -DOCK_AMAX, DOCK_AMAX)), u[1]])
            did = True
        if trace:
            log.append(dict(t=t, rho=int(rho), dock=did, d_goal=round(dist, 1),
                            d_obst=round(obst_min_dist(env), 1),
                            v=round(float(getattr(ego, 'velocity', 0.0)), 2),
                            a=round(float(act[0]), 4), w=round(float(act[1]), 4)))
        obs, _r, term, trunc, info = env.step(np.asarray(act, dtype=float))
        fl = info.get('flags', {})
        if trace and log:
            log[-1]['src'] = info.get('source', '?'); log[-1]['em'] = info.get('emergency_mode')
            log[-1]['flags'] = [k for k, v in fl.items() if v]
        if fl.get('collision'): return 'collision', log
        if fl.get('goal'): return 'goal', log
        if term or trunc: return 'end', log
    return 'end', log


print(f"=== s{SEED} @ {TAKEOVER_R}m · 逐场景找碰撞局 ===")
hits = []
for i, (sc, pp) in enumerate(pool):
    r_dock, _ = run(sc, pp, True)
    if r_dock == 'collision':
        r_rl, _ = run(sc, pp, False)
        name = os.path.basename(getattr(sc, 'benchmark_id', f'idx{i}'))
        hits.append((i, name, r_rl))
        print(f"  🔴 场景 idx={i} ({name}): +停车=碰撞  ·  纯RL={r_rl}")
if not hits:
    print("  未复现碰撞")
    sys.exit(0)

i, name, r_rl = hits[0]
print(f"\n=== 逐步拆解 idx={i} ({name}) · 加停车 ===")
_, log = run(pool[i][0], pool[i][1], True, trace=True)
print(f"{'步':>3} {'rho':>4} {'接管':>4} {'离目标':>8} {'离他船':>8} {'速度':>6} {'a':>8} {'w':>8} {'盾分支':>12} 标志")
for e in ([x for x in log if x['dock']][:6] + [None] + log[-14:]):
    if e is None: print('   ...(中间略)...'); continue
    print(f"{e['t']:>3} {e['rho']:>4} {'✓' if e['dock'] else '':>4} {e['d_goal']:>8.1f} {e['d_obst']:>8.1f} "
          f"{e['v']:>6.2f} {e['a']:>8.4f} {e['w']:>8.4f} {str(e.get('src','?')):>12} {e.get('flags',[])}")

print(f"\n=== 同一局 · 纯 RL（不接管）对照 ===")
_, log2 = run(pool[i][0], pool[i][1], False, trace=True)
print(f"{'步':>3} {'rho':>4} {'离目标':>8} {'离他船':>8} {'速度':>6} {'盾分支':>12} 标志")
for e in log2[-18:]:
    print(f"{e['t']:>3} {e['rho']:>4} {e['d_goal']:>8.1f} {e['d_obst']:>8.1f} {e['v']:>6.2f} {str(e.get('src','?')):>12} {e.get('flags',[])}")
