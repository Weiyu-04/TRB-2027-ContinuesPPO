#!/usr/bin/env python3
"""B1 首测 · CBF-QP 海事避碰滤波在【合成对撞几何】上闭环跑·实测10s ZOH下裸船体碰撞率(别假设0)。
HOCBF: h=‖p_rel‖²−d_safe²(相对度2)·ψ̇1+α2ψ1≥0 → 对u=[a,ω]线性 → 解析投影min‖u−u_nom‖² s.t. g·u≤b, u∈box。
闭环: u_nom=直奔前方目标(=想穿过去·keep-course会撞→逼CBF避)·每步QP·走10s(步内细积分dt=0.1·10s钳v)·障碍CV。
测: 全程细网格裸船体最小距(inter-sample)·撞=矩形重叠。报碰撞率+QP不可行率。别claim 0。"""
import sys, math, json
import numpy as np
sys.path.insert(0, "/tmp/claude-0/-home-user-TRB-2027-ContinuesPPO/c66f9aab-d514-56eb-b3c3-8a5123b55141/scratchpad")
import reclassify as R
A_MAX, W_MAX, V_MAX = R.A_MAX, R.W_MAX, R.V_MAX
L_SHIP, W_SHIP = R.L_SHIP, R.W_SHIP

def circum(l, w): return 0.5*math.hypot(l, w)

def qp_project(u_nom, g, b, box):
    """min‖u−u_nom‖² s.t. g·u≤b, u∈box(2D). 返回(u, feasible)。"""
    a_lo, a_hi, w_lo, w_hi = box
    un = np.clip(u_nom, [a_lo, w_lo], [a_hi, w_hi])
    if float(g @ un) <= b + 1e-9:
        return un, True
    # 约束 active：在 g·u=b 直线∩box 段上投影 u_nom
    gn = float(g @ g)
    if gn < 1e-18:
        return un, (b >= 0)
    u0 = b * g / gn                      # 线上离原点最近点
    d = np.array([-g[1], g[0]]) / math.sqrt(gn)   # 沿线方向
    # box 各边与线求 t 范围
    ts = []
    for i, (lo, hi) in enumerate([(a_lo, a_hi), (w_lo, w_hi)]):
        if abs(d[i]) > 1e-12:
            ts.append(((lo - u0[i]) / d[i], (hi - u0[i]) / d[i]))
    if not ts:
        u = u0
        feas = (a_lo <= u[0] <= a_hi and w_lo <= u[1] <= w_hi)
        return u, feas
    tmin = max(min(t) for t in ts); tmax = min(max(t) for t in ts)
    if tmin > tmax + 1e-9:
        return un, False   # box∩halfplane 空 = QP 不可行
    tstar = np.clip(float((u_nom - u0) @ d), tmin, tmax)
    u = u0 + tstar * d
    return np.clip(u, [a_lo, w_lo], [a_hi, w_hi]), True

def hocbf_constraint(ego, obs, d_safe, a1, a2):
    px, py, th, v = ego; ox, oy, oth, ov = obs
    p_rel = np.array([px - ox, py - oy])
    v_ego = v * np.array([math.cos(th), math.sin(th)])
    v_obs = ov * np.array([math.cos(oth), math.sin(oth)])
    v_rel = v_ego - v_obs
    h = float(p_rel @ p_rel) - d_safe**2
    hd = 2.0 * float(p_rel @ v_rel)
    hb_dir = np.array([math.cos(th), math.sin(th)]); hb_lat = np.array([-math.sin(th), math.cos(th)])
    A_coef = 2.0 * float(p_rel @ hb_dir)
    W_coef = 2.0 * float(p_rel @ hb_lat) * v
    const = 2.0 * float(v_rel @ v_rel) + (a1 + a2) * hd + a1 * a2 * h
    g = np.array([-A_coef, -W_coef]); b = const
    return g, b

def step_ego(ego, u, T=10.0, dt=0.1):
    a, w = float(np.clip(u[0], -A_MAX, A_MAX)), float(np.clip(u[1], -W_MAX, W_MAX))
    x = np.array(ego, float); traj = [x.copy()]
    n = int(round(T/dt))
    for i in range(n):
        v, th = x[3], x[2]
        x = x + dt*np.array([v*math.cos(th), v*math.sin(th), w, a])
        traj.append(x.copy())
    x[3] = float(np.clip(x[3], 0.0, V_MAX))   # 10s 钳
    traj[-1] = x
    return x, np.array(traj)

def run_episode(ego0, obs0, olen, owid, a1, a2, n_steps=24):
    d_safe = circum(L_SHIP, W_SHIP) + circum(olen, owid)   # CBF 安全半径=两外接圆和
    ego = list(ego0); obs = list(obs0)
    goal = np.array([ego0[0] + 6000*math.cos(ego0[2]), ego0[1] + 6000*math.sin(ego0[2])])
    min_d = 1e18; infeas = 0
    for k in range(n_steps):
        # u_nom = 朝目标(直奔·转向按方位·加速保速)
        brg = math.atan2(goal[1]-ego[1], goal[0]-ego[0]) - ego[2]
        brg = (brg + math.pi) % (2*math.pi) - math.pi
        w_nom = float(np.clip(brg/10.0, -W_MAX, W_MAX))
        a_nom = float(np.clip((V_MAX - ego[3])/10.0, -A_MAX, A_MAX))
        g, b = hocbf_constraint(ego, obs, d_safe, a1, a2)
        u, feas = qp_project(np.array([a_nom, w_nom]), g, b, (-A_MAX, A_MAX, -W_MAX, W_MAX))
        if not feas:
            infeas += 1
            u = np.array([-A_MAX, W_MAX if brg > 0 else -W_MAX])   # fallback: 满减速+转离(无保证)
        ego, etraj = step_ego(ego, u)
        # 障碍 CV 同步 + 细网格裸船体最小距
        for j in range(len(etraj)):
            t = j*0.1
            oc = (obs[0]+obs[3]*math.cos(obs[2])*t, obs[1]+obs[3]*math.sin(obs[2])*t)
            dd = R.rect(etraj[j][0], etraj[j][1], etraj[j][2], L_SHIP, W_SHIP).distance(
                 R.rect(oc[0], oc[1], obs[2], olen, owid))
            if dd < min_d: min_d = dd
        obs = [obs[0]+obs[3]*math.cos(obs[2])*10.0, obs[1]+obs[3]*math.sin(obs[2])*10.0, obs[2], obs[3]]
        if min_d <= 0: break
    return min_d, infeas

def main():
    NSAMP = int(sys.argv[1]) if len(sys.argv) > 1 else 80
    recs = [json.loads(l) for l in open("/home/user/TRB-2027-ContinuesPPO/TRB/结果/结果-block3-0722/block3_synthetic_states.jsonl")]
    # 剔 t=0 退化态(已撞)
    clean = [r for r in recs if R.rect(r['ego'][0],r['ego'][1],r['ego'][2],L_SHIP,W_SHIP).distance(
             R.rect(r['obs'][0],r['obs'][1],r['obs'][2],r['obs_len'],r['obs_wid'])) > 0][:NSAMP]
    print(f"B1 CBF-QP 闭环实测 · 合成对撞几何(剔退化) n={len(clean)} · 10s ZOH · 裸船体碰撞")
    for (a1, a2) in [(0.3, 0.3), (0.5, 0.5), (1.0, 1.0)]:
        ncol = 0; ninf = 0; mind_list = []
        for r in clean:
            md, inf = run_episode(r['ego'], r['obs'], r['obs_len'], r['obs_wid'], a1, a2)
            if md <= 0: ncol += 1
            ninf += (inf > 0)
            mind_list.append(md)
        print(f"  α1=α2={a1}: 碰撞 {ncol}/{len(clean)} ({100*ncol/len(clean):.1f}%) · QP不可行发生局 {ninf} · 最小距中位 {np.median(mind_list):.0f}m")
    print("  判读：若碰撞>0 → 经典 CBF-QP 在10s ZOH下【非0碰撞】=实测证据(别假设平手)·且我们可对照差异化。")
    print("        (注:本测=单CV合成几何·nominal直奔·非真RL nominal·非真benchmark·真值须服务器)")

if __name__ == "__main__":
    main()
