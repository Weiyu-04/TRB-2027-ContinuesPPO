#!/usr/bin/env python3
"""独立复现 block1 反向假认证 bug（声称①）。
我自己实现，不 import 探针。核心：证明 signed-v 的 Lipschitz 常数不是 |d'(t)| 的上界 → 假认证；
|v|+a_max·h 的修正版是合法上界 → 正确判 clears=False。同时手推验证 Lipschitz 性质本身。"""
import math
import numpy as np
from shapely.geometry import Polygon

A_MAX, W_MAX, V_MAX = 0.24, 0.03, 9.5
L_SHIP, W_SHIP = 175.0, 25.4
R_CIRC = 0.5*math.hypot(L_SHIP, W_SHIP)   # 88.4
DT10 = 10.0

def rect(cx, cy, th, l, w):
    hl, hw = 0.5*l, 0.5*w; c, s = math.cos(th), math.sin(th)
    loc = [(hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw)]
    return Polygon([(cx+x*c-y*s, cy+x*s+y*c) for x, y in loc])

def integrate_ego(ego0, a, w, T, dt, clip10=True):
    """生产口径积分：常控 (a,w)，步内 v 不地板（允许冲负/overshoot），仅在 10s 边界 clip 到 [0,9.5]。
    解析积分常控 RHS：v'=a, θ'=w, x'=v cosθ, y'=v sinθ。用小步 dt 前向欧拉足够（dt=0.01）。"""
    x = np.array(ego0, float); ts = [0.0]; traj = [x.copy()]
    n = int(round(T/dt))
    for i in range(n):
        v, th = x[3], x[2]
        x = x + dt*np.array([v*math.cos(th), v*math.sin(th), w, a])
        t = (i+1)*dt
        if clip10 and abs(t/DT10 - round(t/DT10)) < 1e-9:
            x[3] = float(np.clip(x[3], 0.0, V_MAX))
        ts.append(t); traj.append(x.copy())
    return np.array(ts), np.array(traj)

def dist_profile(ts, traj, obs0, olen, owid):
    vm = obs0[3]; om = (math.cos(obs0[2]), math.sin(obs0[2]))
    ds = []
    for k in range(len(ts)):
        t = ts[k]; ex, ey, eth, ev = traj[k]
        oc = (obs0[0]+vm*om[0]*t, obs0[1]+vm*om[1]*t)
        ds.append(rect(ex, ey, eth, L_SHIP, W_SHIP).distance(rect(oc[0], oc[1], obs0[2], olen, owid)))
    return np.array(ds)

def lb_from_L(ts, traj, ds, obs0, h, mode):
    """在步长 h 的子采样上算区间下界 (d_k+d_{k+1}-L*h)/2。mode='fixed' 或 'signed'。"""
    dt0 = ts[1]-ts[0]; stride = max(1, int(round(h/dt0)))
    idx = np.arange(0, len(ts), stride)
    if idx[-1] != len(ts)-1: idx = np.append(idx, len(ts)-1)
    hh = stride*dt0
    vv = traj[idx, 3]; vm = obs0[3]
    dsub = ds[idx]
    if mode == 'fixed':
        v_term = np.maximum(np.abs(vv[:-1]), np.abs(vv[1:])) + A_MAX*hh   # 修：|v|+a_max*h
        w_term = W_MAX*R_CIRC
    elif mode == 'signed':
        v_term = np.maximum(vv[:-1], vv[1:])                              # bug：带符号 v，无 a_max*h
        w_term = W_MAX*R_CIRC
    L = v_term + w_term + abs(vm)
    ilb = (dsub[:-1]+dsub[1:] - L*hh)/2.0
    return ilb, L, idx, hh

# ── 文档记录的反向假认证复例（selftest line 631）─────────────────────────
ego0 = [0.0, 0.0, -2.654, 0.324]; a = -A_MAX; w = -W_MAX
obs0 = [-113.66, 82.4, -1.35, 0.45]; h = 0.5

# 1) 生产口径积分 + 细网格真距
ts, traj = integrate_ego(ego0, a, w, 60.0, dt=0.01, clip10=True)
ds = dist_profile(ts, traj, obs0, L_SHIP, W_SHIP)
true_min = float(ds.min()); t_at = float(ts[int(ds.argmin())])
print(f"=== 反向假认证复例（a={a}, w={w}, ego0={ego0}, obs0={obs0}）===")
print(f"步内 v 轨迹范围: [{traj[:,3].min():.3f}, {traj[:,3].max():.3f}]  (负值=反向冲负 → bug 触发条件)")
print(f"真实船体最小距离(0.01s细网格) = {true_min:.4f}m @ t={t_at:.2f}s   ({'撞' if true_min<=0 else '未撞'})")

# 2) 修正版 L → 应判 clears=False
ilb_fix, Lfix, idx, hh = lb_from_L(ts, traj, ds, obs0, h, 'fixed')
clears_fix = bool(np.all(ilb_fix > 0))
print(f"\n[修正版 L=|v|+a_max*h+|ω|R+|v_m|] min区间下界 = {ilb_fix.min():+.4f}m → clears={clears_fix}  "
      f"({'✅ 正确=不认证' if not clears_fix else '🔴 仍假认证'})")

# 3) signed-v L → 复现假认证
ilb_sgn, Lsgn, _, _ = lb_from_L(ts, traj, ds, obs0, h, 'signed')
clears_sgn = bool(np.all(ilb_sgn > 0))
print(f"[旧 signed-v L] min区间下界 = {ilb_sgn.min():+.4f}m → clears={clears_sgn}  "
      f"({'🔴 假认证=认证清障但真撞' if clears_sgn and true_min<=0 else 'ok'})")

# 4) 数学核心：Lipschitz 上界性质。经验 |Δd/Δt| 与两个 L 比。合法 L 必 ≥ 全程 |d'|。
dd = np.abs(np.diff(ds))/np.diff(ts)
emp_max = float(dd.max())
print(f"\n=== Lipschitz 上界核验（合法常数须 ≥ 经验 max|d'(t)|）===")
print(f"经验 max|Δd/Δt|(细网格)   = {emp_max:.4f} m/s")
print(f"修正版 L (逐段最大)        = {Lfix.max():.4f} m/s  → {'✅ 是上界' if Lfix.max()>=emp_max-1e-6 else '🔴 非上界'}")
print(f"signed-v L (逐段最大)      = {Lsgn.max():.4f} m/s  → {'🔴 非上界=不合法' if Lsgn.max()<emp_max-1e-6 else '是上界'}")

# 5) 理论真上界（反向时点速|v|+转向|ω|R+他船|v_m|）供对照
vmax_pt = np.abs(traj[:,3]).max() + W_MAX*R_CIRC + abs(obs0[3])
print(f"理论点速上界 |v|max+|ω|R+|v_m| = {vmax_pt:.4f} m/s (真 Lipschitz 常数量级)")
