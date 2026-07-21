#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""模块1(终端入库控制器)零算力开环可行性验证。
问:一个几何控制器(用满物理 a∈±0.24/ω∈±0.03·非RL的±0.018枷锁),从崩种子近门状态出发,
   能不能把船头拧进 ±9.74° 窄门+位置进 400×60 框(170步超时内)?
行→模块可行(几何能做 RL 结构上做不到的紧急拧头);不行→终端问题更难,重议。"""
import json, math, numpy as np
from trb_env.usv_dynamics import make_vessel_params, step, DECISION_DT

p = make_vessel_params()
A_MAX, W_MAX = p.a_max, p.w_max         # 0.24 / 0.03（物理满程·控制器可用·RL 被卡 0.018）
print(f'物理箱: a_max={A_MAX} w_max={W_MAX}(=17.19°/步·RL 只有 0.018=10.31°/步)')

def wrap(a): return (a + math.pi) % (2*math.pi) - math.pi

def load_fp(path):
    for l in open(path).read().splitlines():
        d = json.loads(l)
        if d.get('final_per'): return d['final_per']

# ---- 几何终端控制器（Stanley 横向对齐 + 速度调度让转弯半径够小）----
def dock_controller(state, goal, theta_g=0.0):
    """state=[px,py,θ,v]  goal=(gx,gy)  theta_g=目标朝向(0=+x)。返回 [a, ω] 已裁物理箱。"""
    px, py, th, v = state
    gx, gy = goal
    # 横向偏差(相对'进框中线'y=gy·朝向 θ_g=0 的进近线)
    e_cross = gy - py
    heading_err = wrap(theta_g - th)                 # 想让 θ→0
    # Stanley: 期望航向修正 = 航向误差 + atan(k·横偏/速度)
    k_e = 0.6
    desired_corr = heading_err + math.atan2(k_e * e_cross, max(v, 1.0))
    desired_corr = wrap(desired_corr)
    omega = float(np.clip(desired_corr / DECISION_DT, -W_MAX, W_MAX))   # 一步内尽量拧到期望
    # 速度调度:朝向没对齐/横偏大/临近 x 时→减速(转弯半径∝v·慢才拧得进窄门);对齐了→保适中速进框
    dist_x = gx - px
    align = abs(heading_err) < math.radians(20) and abs(e_cross) < 40
    v_target = 3.0 if align else 1.0                 # 未对齐狠减速(让 17.19°/步够拧);对齐后 3m/s 稳进框
    if dist_x < 250 and not align: v_target = 0.8    # 快到 x 还没对齐→更慢,别冲过
    a = float(np.clip(0.5 * (v_target - v), -A_MAX, A_MAX))
    return np.array([a, omega])

def in_gate(state, gg):
    px, py, th, v = state
    vx = gg['vertices']
    xs = [q[0] for q in vx]; ys = [q[1] for q in vx]
    in_box = (min(xs) <= px <= max(xs)) and (min(ys) <= py <= max(ys))
    aligned = gg['orient_lo'] <= wrap(th) <= gg['orient_hi']
    return in_box and aligned

def rollout(state0, goal, gg, budget):
    st = np.array(state0, dtype=float)
    for k in range(budget):
        if in_gate(st, gg): return True, k, st
        a = dock_controller(st, goal)
        st = step(st, a, DECISION_DT, p)
    return in_gate(st, gg), budget, st

# ---- 从崩种子近门状态取 takeover 点(<=350m·崩种子实际到过的近门态)测捕获 ----
BASE = '结果0710-22:00-10种子最优方案'
def crash_states(seed):
    fp = load_fp(f'{BASE}/step4e_partial_L1rateON_ppo_s{seed}.jsonl')
    out = []
    for e in fp:
        tj = e.get('traj')
        if not tj: continue
        gx, gy = e['goal_geom']['center']; gg = e['goal_geom']
        for t in tj:
            d = math.hypot(gx - t['ego_x'], gy - t['ego_y'])
            if d <= 350:                    # takeover 区
                st0 = [t['ego_x'], t['ego_y'], t['ego_psi'], t['ego_v']]
                budget = max(5, 170 - t['step'])   # 剩余时间预算
                out.append((st0, (gx, gy), gg, budget, e['reached'], round(d), round(math.degrees(abs(wrap(0 - t['ego_psi']))))))
                break                       # 每局取第一次进 350m 的态
    return out

print('\n=== 崩种子近门 takeover 态 → 几何控制器能否捕获窄门 ===')
for seed, tag in [(5, 's5[崩·高速]'), (6, 's6[崩·低速迷路]')]:
    states = crash_states(seed)
    cap = 0
    print(f'\n基线 {tag}: {len(states)} 个近门态(仅有traj的3局)')
    for i, (st0, goal, gg, budget, orig_reached, d0, hd0) in enumerate(states):
        ok, k, stf = rollout(st0, goal, gg, budget)
        cap += ok
        df = math.hypot(goal[0]-stf[0], goal[1]-stf[1])
        print(f'  态{i}: 起 dist={d0}m 船头偏={hd0}° v={st0[3]:.1f} 预算{budget}步 | RL原结局={"到达" if orig_reached else "崩"} '
              f'→ 控制器: {"✅捕获@"+str(k)+"步" if ok else "❌未捕获"}(末 dist={df:.0f}m 船头偏={math.degrees(abs(wrap(stf[2]))):.0f}°)')
    print(f'  → 控制器捕获率 {cap}/{len(states)}')

# ---- 构造 worst-case:高速冲、船头背对(崩塌最狠态)----
print('\n=== 构造 worst-case(高速+船头大偏·崩塌最狠) → 控制器 ===')
gg_demo = None
fp = load_fp(f'{BASE}/step4e_partial_L1rateON_ppo_s5.jsonl')
for e in fp:
    if e.get('traj'): gg_demo = e['goal_geom']; goal_demo = tuple(e['goal_geom']['center']); break
for (d0, hd_deg, v0, desc) in [(300, 90, 8.0, '300m/偏90°/8m/s'),
                                (200, 150, 9.0, '200m/偏150°/9m/s(几乎背对)'),
                                (150, 60, 6.0, '150m/偏60°/6m/s'),
                                (100, 120, 4.0, '100m/偏120°/4m/s')]:
    gx, gy = goal_demo
    # 放在目标正西 d0 处、船头偏 hd_deg
    st0 = [gx - d0, gy, math.radians(hd_deg), v0]
    budget = 120
    ok, k, stf = rollout(st0, goal_demo, gg_demo, budget)
    df = math.hypot(gx-stf[0], gy-stf[1])
    print(f'  {desc}: → {"✅捕获@"+str(k)+"步" if ok else "❌未捕获"}(末 dist={df:.0f}m 船头偏={math.degrees(abs(wrap(stf[2]))):.0f}° v={stf[3]:.1f})')
