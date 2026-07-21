#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test1（方案C 设计 wf w045x4hyy·对抗综合推荐·零算力 go/no-go）：
纯几何基准控制器【全程】过原盾跑测试集·不载 checkpoint·不用策略。

【判据·对症】崩=终端"高速绕圈超时"(2/10 种子·8/10 能解同 40 场景=强示【训练动力学/坏盆地】非场景硬度)。
- 若纯基准(全程朝目标+近门减速蠕行·RL箱权限)过盾在【崩种子失败的那些场景】到达率高
  → 坐实"目标可由 朝目标+减速 轨迹达成"=崩是训练问题·残差/课程有好地基 → GO(可投产验证)。
- 若纯基准也崩这些场景 → 问题是场景硬度/避碰几何·非训练 → 残差/课程都救不了 → 【立即中止别烧5M】。

场景源: STEP4E_MANIFEST 设→真40测试集; 否则 SCN_GLOB(本机sanity)。基准 RL箱权限(公平·同 dock harness)。
用法(本机): PYTHONPATH=代码 python 代码/m1_dock_wip/test1_pure_baseline.py
用法(服务器): 同 closed_loop_dock.py 的 STEP4E_MANIFEST 模式 + OUT_DIR。
"""
import os, sys, glob, math, numpy as np
sys.path.insert(0, '/Users/weiyutang/Desktop/TRB/代码')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trb_env.usv_continuous_shield import ContinuousProjectionEnv
from trb_env.usv_scenarios import load_scenario_pool
from trb_env.usv_env import A_NORMAL_OMEGA_MAX, A_NORMAL_ACCEL_MAX
import dock_controller_v4 as DC

MANIFEST = os.environ.get('STEP4E_MANIFEST', '')
OUT_DIR  = os.environ.get('OUT_DIR', '')
V_RUN    = float(os.environ.get('BASE_VRUN', '2.6'))          # 远场巡航速度(默认2.6·太慢会超时·可调)
FULL_PHYS = os.environ.get('DOCK_FULL_PHYSICAL', '0') == '1'
WMAX = DC.W_MAX if FULL_PHYS else A_NORMAL_OMEGA_MAX
AMAX = None if FULL_PHYS else A_NORMAL_ACCEL_MAX

if MANIFEST:
    from run_step4e import load_manifest_split
    _bdir = os.environ.get('STEP4E_BALANCED_DIR') or os.path.dirname(os.path.abspath(MANIFEST))
    _tr, test_paths, _i = load_manifest_split(MANIFEST, _bdir)
    scn = test_paths; SRC = f"manifest 测试集(n={len(scn)})"
else:
    scn = sorted(glob.glob(os.environ.get('SCN_GLOB', '/private/tmp/trb_scenarios_pool/T-*.xml')))[:40]
    SRC = f"本机 glob sanity(n={len(scn)}·非标准集)"
pool = load_scenario_pool(scn)
print(f"Test1 纯基准全程过盾 | {SRC} | 基准权限={'满物理' if FULL_PHYS else 'RL箱(公平)'} | v_run={V_RUN}")

def mk_env(sc, pp):
    return ContinuousProjectionEnv(sc, pp, shield=True, goal_cone_half=None, goal_v_floor=2.0, augment_rho=False)

def wrap(a): return (a + math.pi) % (2 * math.pi) - math.pi

def run_ep(sc, pp):
    env = mk_env(sc, pp); obs, info = env.reset(seed=0)
    reached = False; term_flag = 'other'
    mind = 1e9; hd_at_min = 0.0; path = 0.0; prev = None; src = {}; emg = 0; steps = 0
    for _ in range(200):
        ego = env._ego_vs(); gg = env.env.goal_center
        try:
            theta_g = 0.5 * (env.env.goal.orientation.start + env.env.goal.orientation.end)
        except Exception:
            theta_g = 0.0
        st = [ego.position[0], ego.position[1], ego.orientation, float(getattr(ego, 'velocity', 0.0))]
        act = DC.dock_controller(st, (gg[0], gg[1]), theta_g=theta_g, wmax=WMAX, v_run=V_RUN)
        if AMAX is not None:
            act = np.array([float(np.clip(act[0], -AMAX, AMAX)), act[1]])
        obs, _r, term, trunc, info = env.step(np.asarray(act, dtype=float)); steps += 1
        _s = info.get('source', '?'); src[_s] = src.get(_s, 0) + 1
        if info.get('rho_acting') == 5 or _s == 'emergency': emg += 1
        ego = env._ego_vs(); p = np.array(ego.position)
        if prev is not None: path += float(np.hypot(*(p - prev)))
        prev = p
        d = float(np.hypot(p[0] - gg[0], p[1] - gg[1]))
        if d < mind: mind = d; hd_at_min = math.degrees(abs(wrap(ego.orientation - theta_g)))
        fl = info.get('flags', {})
        if fl.get('goal', False): reached = True; term_flag = 'goal'; break
        if term or trunc:
            term_flag = next((k for k in ('collision', 'stopped', 'area', 'time') if fl.get(k)), 'other'); break
    ego = env._ego_vs(); gg = env.env.goal_center
    endv = float(getattr(ego, 'velocity', 0.0)); enddist = float(np.hypot(ego.position[0] - gg[0], ego.position[1] - gg[1]))
    # 失败机制分类: 近门绕圈(捕获病·同崩种子) vs 避让改航(避碰病·基准不会避)
    cls = 'reach'
    if not reached:
        if term_flag == 'collision': cls = 'collision(盾紧急兜底)'
        elif mind < 200 and path > 5000: cls = 'capture近门绕圈'
        elif emg > 10 or mind > 500: cls = 'avoid避让改航/远离'
        else: cls = '其他'
    return dict(reached=reached, term=term_flag, mind=mind, hd=hd_at_min, path=path, endv=endv,
                enddist=enddist, emg=round(100 * emg / max(steps, 1), 1), cls=cls,
                src={k: v for k, v in src.items() if v})

reach = 0; terms = {}; clss = {}; rows = []
print(f"\n{'scn':>3} {'到达':>4} {'term':>9} {'最近门':>6} {'朝向°':>6} {'路径':>6} {'末v':>5} {'紧急%':>5}  机制分类")
for i, (sc, pp) in enumerate(pool):
    r = run_ep(sc, pp); reach += r['reached']
    terms[r['term']] = terms.get(r['term'], 0) + 1
    clss[r['cls']] = clss.get(r['cls'], 0) + 1
    rows.append({'scn': i, **r})
    print(f"{i:>3} {'✅' if r['reached'] else '❌':>3} {r['term']:>9} {r['mind']:>6.0f} {r['hd']:>6.0f} {r['path']:>6.0f} {r['endv']:>5.1f} {r['emg']:>5.1f}  {r['cls']}")
print(f"\n=== 纯基准到达率: {reach}/{len(pool)} = {100*reach/len(pool):.1f}% | term 分布: {terms} ===")
print(f"=== 失败机制分类: {clss} ===")
print("判读: 失败若多为【avoid避让改航】=基准不会避让(那是RL的活)·终端捕获在到达的场景已证可行→学习方案有地基;")
print("      失败若多为【capture近门绕圈】=连控制器都对不进窄门=终端捕获本身难→对学习方案是坏信号。")
if OUT_DIR:
    import json as _json
    os.makedirs(OUT_DIR, exist_ok=True)
    _p = os.path.join(OUT_DIR, 'test1_pure_baseline' + ('_40set' if MANIFEST else '_sanity') + '.json')
    _json.dump(dict(source=SRC, v_run=V_RUN, authority=('full' if FULL_PHYS else 'rl_box'),
                    n=len(pool), reached=reach, reach_pct=round(100*reach/len(pool), 1),
                    terms=terms, fail_class=clss, per_scenario=rows),
               open(_p, 'w'), ensure_ascii=False, indent=1)
    print(f"✅ 落盘: {_p}")
