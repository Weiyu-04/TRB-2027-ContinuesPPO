#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""终端捕获探针的判据 eval（方案C-B·`03` L182·对抗审 HIGH-2 修：eval 恒真起点看不到近门捕获→建单独近门诊断 eval）。

判据：把 checkpoint 在【近门起点 start_frac=PROBE_FRAC】评估捕获率，减去【空策略 null 基线】，重点看【硬子集=init 朝向落目标门外(需转向对齐)】。
- GO：崩种子 s5/s6 的 (trained−null) 捕获 delta 追平健康对照 → PPO 生在门口能学会捕获 → 崩是远场探索/坏盆地 → 退火课程可攻。
- NO-GO：崩种子生在门口仍不捕获(仍绕圈) → 终端捕获本身坏 → 课程救不了 → 转终端控制器 M1。
⚠️此=诊断 eval(设 start_frac 于诊断 env)·非 Table III·不碰真起点诚实红线。null 基线+硬子集 防平凡捕获污染(对抗审 HIGH-1)。

用法: PYTHONPATH=代码 STEP4E_MANIFEST=... CKPT_DIR=... CKPT_TMPL='...s{s}...' SEEDS='5 6 2' PROBE_FRAC=0.2 START_V=6 python 代码/m1_dock_wip/probe_capture_eval.py
"""
import os, sys, glob, math, numpy as np
sys.path.insert(0, '/Users/weiyutang/Desktop/TRB/代码'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from trb_env.train import make_obs_transform
from trb_env.usv_continuous_shield import ContinuousProjectionEnv
from trb_env.usv_scenarios import load_scenario_pool
def wrap(a): return (a + math.pi) % (2 * math.pi) - math.pi

PROBE_FRAC = float(os.environ.get('PROBE_FRAC', '0.2'))
START_V    = os.environ.get('START_V', '6'); START_V = None if START_V.lower() in ('off','none','') else float(START_V)
CKPT_DIR   = os.environ['CKPT_DIR']
CKPT_TMPL  = os.environ.get('CKPT_TMPL', 'Continuous-safe_s{s}_probeRC_ppo_s{s}')
SEEDS      = [int(x) for x in os.environ.get('SEEDS', '5 6 2').split()]
MANIFEST   = os.environ['STEP4E_MANIFEST']
from run_step4e import load_manifest_split
_bdir = os.environ.get('STEP4E_BALANCED_DIR') or os.path.dirname(MANIFEST)
_tr, test_paths, _i = load_manifest_split(MANIFEST, _bdir)
pool = load_scenario_pool(test_paths)
print(f"终端捕获探针诊断 | 近门起点 frac={PROBE_FRAC} start_v={START_V} | 测试集 n={len(pool)} | 种子={SEEDS}")

def mk_env(sc, pp):
    e = ContinuousProjectionEnv(sc, pp, shield=True, goal_cone_half=None, goal_v_floor=2.0)
    e.set_start_frac(PROBE_FRAC, START_V)   # 诊断:近门起点
    return e

def is_hard(sc, pp):
    """硬子集：init 朝向落目标门外(±门朝向容差)=需转向对齐·空策略难白拿。"""
    e = mk_env(sc, pp); e.reset(seed=0); inner = e.env
    try:
        lo = inner.goal.orientation.start; hi = inner.goal.orientation.end
    except Exception:
        lo, hi = -0.17, 0.17
    th = float(inner.init_state.orientation)
    return not (wrap(th - lo) >= -1e-9 and wrap(th - hi) <= 1e-9) if hi >= lo else True

def run_pol(sc, pp, model, tf):
    """返回 dict:reached + 失败机制诊断(区分"接近段绕圈"[从没到门]vs"终端对齐失败"[到门对不进]vs"停短")。
    方法论关键(L182 user 强调):不止测"是否到达"·须测"为什么"·防"800m接近段又绕圈"被误判成"终端捕获坏"污染 go/no-go。"""
    env = mk_env(sc, pp); obs, info = env.reset(seed=0)
    reached = False; term = 'other'; mind = 1e9; hd = 0.0; path = 0.0; prev = None; steps = 0
    gg = env.env.goal_center
    try: theta_g = 0.5 * (env.env.goal.orientation.start + env.env.goal.orientation.end)
    except Exception: theta_g = 0.0
    for _ in range(200):
        act, _ = model.predict(tf(obs), deterministic=True)
        obs, _r, t, tr, info = env.step(np.asarray(act, dtype=float)); steps += 1
        ego = env._ego_vs(); p = np.array(ego.position)
        if prev is not None: path += float(np.hypot(*(p - prev)))
        prev = p
        d = float(np.hypot(p[0] - gg[0], p[1] - gg[1]))
        if d < mind: mind = d; hd = math.degrees(abs(wrap(ego.orientation - theta_g)))
        fl = info.get('flags', {})
        if fl.get('goal', False): reached = True; term = 'goal'; break
        if t or tr:
            term = next((k for k in ('collision', 'stopped', 'area', 'time') if fl.get(k)), 'other'); break
    ego = env._ego_vs(); endv = float(getattr(ego, 'velocity', 0.0))
    # 失败机制(离门0.2起点约800m·approach绕圈=从没到门near·terminal=到门near但绕/对不进)
    cls = 'reach'
    if not reached:
        if term == 'stopped' or (endv < 0.5 and mind < 200): cls = '停短'
        elif mind > 350: cls = '接近段绕圈(从没到门)'
        elif mind <= 350 and path > 3500: cls = '终端绕圈/对不进门'
        else: cls = '其他'
    return dict(reached=reached, term=term, mind=mind, hd=hd, path=path, endv=endv, cls=cls)

def run_null(sc, pp):
    env = mk_env(sc, pp); obs, info = env.reset(seed=0)
    for _ in range(200):
        obs, _r, term, trunc, info = env.step(np.array([0.0, 0.0]))
        if info.get('flags', {}).get('goal', False): return True
        if term or trunc: return False
    return False

hard = [is_hard(sc, pp) for sc, pp in pool]
n = len(pool); nh = sum(hard)
# null 基线(与种子无关·算一次)
null_all = sum(run_null(sc, pp) for sc, pp in pool)
null_hard = sum(run_null(sc, pp) for (sc, pp), h in zip(pool, hard) if h)
print(f"\n空策略 null 基线: 全 {null_all}/{n}={100*null_all/n:.0f}% | 硬子集 {null_hard}/{nh}={100*null_hard/max(nh,1):.0f}% (硬子集应≈0=无平凡捕获)")
print(f"\n{'seed':>4} | {'trained全':>9} {'trained硬':>9} | {'d硬':>6} | 失败机制(硬子集·区分绕圈段vs终端·防归因污染)")
for s in SEEDS:
    ck = os.path.join(CKPT_DIR, CKPT_TMPL.format(s=s))
    if not (os.path.exists(ck + '.zip') and os.path.exists(ck + '_vecnorm.pkl')):
        print(f"{s:>4} | ❌ 缺 {ck}"); continue
    _bv = DummyVecEnv([lambda: mk_env(pool[0][0], pool[0][1])])
    _vn = VecNormalize.load(ck + '_vecnorm.pkl', _bv); _vn.training = False
    tf = make_obs_transform(_vn); model = PPO.load(ck + '.zip', device='cpu')
    res = [run_pol(sc, pp, model, tf) for sc, pp in pool]
    tr_all = sum(r['reached'] for r in res)
    tr_hard = sum(r['reached'] for r, h in zip(res, hard) if h)
    d_all = 100*(tr_all - null_all)/n; d_hard = 100*(tr_hard - null_hard)/max(nh, 1)
    # 硬子集失败机制分布(方法论:知道"为什么没到"·防"接近段绕圈"被误当"终端坏")
    fmodes = {}
    for r, h in zip(res, hard):
        if h and not r['reached']: fmodes[r['cls']] = fmodes.get(r['cls'], 0) + 1
    fm = ' '.join(f"{k}:{v}" for k, v in sorted(fmodes.items())) or '(硬子集全捕获)'
    print(f"{s:>4} | {tr_all:>4}/{n}={100*tr_all/n:>3.0f}% {tr_hard:>4}/{nh}={100*tr_hard/max(nh,1):>3.0f}% | {d_hard:>+5.0f}% | {fm}")
print("\n判读(方法论·不止看delta·看失败机制干净归因):")
print("  🔴前置守卫(防假GO/假NO-GO·L184 对抗审 MEDIUM·SEEDS 必含健康对照如 s2):先看【健康对照】硬子集【绝对】捕获率——")
print("     健康对照未过绝对地板(trained硬 <~50%)→ 1M 欠训 / 近门奖励教不会【任何】种子(setup 病)=【inconclusive·非GO非NO-GO】·先查 奖励是否奖捕获/start_v=6是否太快/步数是否够·别据此下 GO 或 NO-GO。")
print("     健康对照过地板(trained硬 ≥~50%)后 → 它才是有效标尺·再判下面 GO/NO-GO。")
print("  GO=崩种子 s5/s6 的 d硬 追平【已过地板的】健康对照 → 近门能学会捕获 → 崩是远场问题 → 退火课程可攻。")
print("  NO-GO(终端真坏)=健康对照【过地板】但 崩种子硬子集失败多为【终端绕圈/对不进门】(到了门口对不进) → 终端捕获本身坏 → 转 M1。")
print("  ⚠️歧义(须再判)=崩种子失败多为【接近段绕圈(从没到门)】 → 800m接近段又绕圈 → frac=0.2可能太远·或approach即坏 → 试更近frac(0.1)或看健康对照是否也如此。")
