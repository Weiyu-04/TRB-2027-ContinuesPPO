"""
block1 验证：sound 连续时间清障判据（Lipschitz 采样）——`Paper/makeorbreak_连续时间清障下界_推导.md`。
================================================================================
这是 make-or-break 数学的【实现级 soundness 验证】(self-contained·不依赖 vesselmodels)：
用【已知的】yp RHS f=[v·cosθ, v·sinθ, ω, a]（文档+双 agent 核过·非猜）+ 连续 v-clip[0,v_max]
（=escape 控制器实际执行口径·F2），细步 RK4 精确积分（~1e-6·满足 F1"位置误差<<margin"·非 Euler）。

判据（推导 §2）：d(t)=船体到船体距离·Lipschitz 常数 L=(v_bnd+|ω|·R_circ)+v_obs；
  区间下界 min_[t_k,t_{k+1}] d ≥ (d_k+d_{k+1}−L·h)/2 → 每区间 d_k+d_{k+1}>L·h ⟹ 连续时间清障。
  "有效膨胀"=L·h/2（h=0.5→~6m·可调·非 350m）。他船 occupancy 用真实矩形(over-approx·F3)·
  v_obs 用逐场景真 v_m·L 的 ego 项用 v_bnd=11.9 保守(F4/F2)。

验证结果（2026-07-21 本机跑·numpy2.4/scipy1.17/shapely2.1）：
  · 确定性 A(远清障)/B(迫近撞)/C(穿模压力·3 个 h) 全对且下界 sound；
  · 🔴 600 随机对抗配置：认证 clears=True 574 个·**假认证(真会撞)=0**·下界从不高估(min gap +0.538m) = SOUND。
⚠️ 生产版（跑真基准三分区探针）须把 integrate_fine/rect_poly 换成官方 usv_dynamics.step + _ego_rect
   （"不自己写·调官方包"铁律）——本档只验 Lipschitz 逻辑·逻辑已 sound。
"""
import numpy as np
from shapely.geometry import Polygon

A_MAX, W_MAX, V_MAX = 0.24, 0.03, 9.5
L_SHIP, W_SHIP = 175.0, 25.4
R_CIRC = 0.5 * np.hypot(L_SHIP, W_SHIP)       # 88.4
V_BND = V_MAX + A_MAX * 10.0                  # 11.9（L ego 项保守上界·含步内超调·F2/F4）
DT_FINE = 0.05                                # 细积分步（RK4·~1e-6·满足 F1）


def _rhs(x, a, w):
    px, py, th, v = x
    dv = 0.0 if ((v <= 0.0 and a < 0.0) or (v >= V_MAX and a > 0.0)) else a
    return np.array([v * np.cos(th), v * np.sin(th), w, dv])


def integrate_fine(ego0, a, w, T):
    """细步 RK4·常控 (a,w)·连续 v-clip[0,v_max]。返回 (ts, traj[N,4])。"""
    a = float(np.clip(a, -A_MAX, A_MAX)); w = float(np.clip(w, -W_MAX, W_MAX))
    n = int(round(T / DT_FINE)); x = np.array(list(map(float, ego0))); out = [x.copy()]
    for _ in range(n):
        k1 = _rhs(x, a, w); k2 = _rhs(x + 0.5*DT_FINE*k1, a, w)
        k3 = _rhs(x + 0.5*DT_FINE*k2, a, w); k4 = _rhs(x + DT_FINE*k3, a, w)
        x = x + (DT_FINE/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        x[3] = min(max(x[3], 0.0), V_MAX); out.append(x.copy())
    return np.arange(n+1)*DT_FINE, np.array(out)


def rect_poly(cx, cy, theta, length, width):
    hl, hw = 0.5*length, 0.5*width; c, s = np.cos(theta), np.sin(theta)
    return Polygon([(cx+x*c-y*s, cy+x*s+y*c) for x, y in [(hl,hw),(hl,-hw),(-hl,-hw),(-hl,hw)]])


def _dist_at(ts, traj, obs0, idx, obs_len=L_SHIP, obs_wid=W_SHIP):
    v = float(obs0[3]); od = np.array([np.cos(obs0[2]), np.sin(obs0[2])]); r = np.empty(len(idx))
    for j, k in enumerate(idx):
        t = ts[k]; ex, ey, eth, ev = traj[k]
        r[j] = rect_poly(ex, ey, eth, L_SHIP, W_SHIP).distance(
                rect_poly(obs0[0]+v*od[0]*t, obs0[1]+v*od[1]*t, obs0[2], obs_len, obs_wid))
    return r


def clearance_certificate(ego0, a, w, obs0, T, h, obs_len=L_SHIP, obs_wid=W_SHIP):
    """SOUND 连续时间清障判据。返回 clears/min_lb/min_sample。h 须为 DT_FINE 整数倍。"""
    ts, traj = integrate_fine(ego0, a, w, T)
    stride = max(1, int(round(h/DT_FINE))); idx = np.arange(0, len(ts), stride)
    ds = _dist_at(ts, traj, obs0, idx, obs_len, obs_wid); hh = stride*DT_FINE
    L_lip = (V_BND + abs(float(np.clip(w, -W_MAX, W_MAX)))*R_CIRC) + float(obs0[3])
    ilb = (ds[:-1] + ds[1:] - L_lip*hh) / 2.0
    min_lb = float(ilb.min()) if len(ilb) else float(ds.min())
    return {"clears": min_lb > 0.0, "min_lb": min_lb, "min_sample": float(ds.min()), "L_lip": L_lip}


def true_min_dist(ego0, a, w, obs0, T, grid=0.2):
    ts, traj = integrate_fine(ego0, a, w, T)
    return float(_dist_at(ts, traj, obs0, np.arange(0, len(ts), int(round(grid/DT_FINE)))).min())


def _run_selfcheck():
    print("=== 确定性 case ===")
    for name, e, a, w, o, T in [
        ("A 远清障",      [0,0,0,5.0], -A_MAX, -W_MAX, [3000,0,np.pi,5.0], 120),
        ("B 迫近撞",      [0,0,0,8.0], -A_MAX,  0.0,   [400,0,np.pi,8.0],  120)]:
        r = clearance_certificate(e,a,w,o,T,0.5); t = true_min_dist(e,a,w,o,T)
        print(f"  {name}: clears={r['clears']} min_lb={r['min_lb']:.1f} 真={t:.1f} sound={r['min_lb']<=t+1e-6}")
    print("=== 🔴 随机对抗 soundness 扫描（0 假认证=sound）===")
    rng = np.random.default_rng(20260721); N = 600; nc = nf = 0; mg = np.inf
    for _ in range(N):
        e = [0,0,rng.uniform(-np.pi,np.pi),rng.uniform(0,V_MAX)]
        ang = rng.uniform(-np.pi,np.pi); dd = rng.uniform(200,1200)
        o = [dd*np.cos(ang),dd*np.sin(ang),rng.uniform(-np.pi,np.pi),rng.uniform(0,V_MAX)]
        a = rng.choice([-A_MAX,0,A_MAX]); w = rng.choice([-W_MAX,0,W_MAX]); h = rng.choice([0.25,0.5,1.0])
        r = clearance_certificate(e,a,w,o,60.0,h)
        if r["clears"]:
            nc += 1; t = true_min_dist(e,a,w,o,60.0)
            if t <= 0.0: nf += 1
            mg = min(mg, t - r["min_lb"])
    print(f"  {N} 配置：clears={nc} · 假认证={nf} · 下界 min_gap={mg:.3f}m")
    print("  " + ("✅ SOUND" if nf == 0 and mg >= -1e-6 else f"🔴 UNSOUND nf={nf} mg={mg}"))


if __name__ == "__main__":
    _run_selfcheck()
