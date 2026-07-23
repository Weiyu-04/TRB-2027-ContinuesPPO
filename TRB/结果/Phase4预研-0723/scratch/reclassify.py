#!/usr/bin/env python3
"""独立重建 block3 分类流水线复核声称②（未决 21-27%）。
- 动力学：常控 (a,ω) 解析可积 → 前向欧拉 dt=0.01（与官方 odeint 误差~1e-8）+ 10s 边界钳 v∈[0,9.5]。
- gap#1：直接复制 usv_projection.imminent_unavoidable_certificate 的 numpy 逻辑（无需 vesselmodels）。
- 清障判据：复制修正版 Lipschitz。机动族：窗口版 vs 我加浓版。
关键问题：加浓机动族后 undecided 是否大幅下降？下降 → 未决=机动族弱的产物(不可靠上界)；不下降 → 接近场景天花板。"""
import json, math, sys, time
import numpy as np
from shapely.geometry import Polygon

A_MAX, W_MAX, V_MAX = 0.24, 0.03, 9.5
L_SHIP, W_SHIP = 175.0, 25.4
EGO_WIDTH = 25.4
R_CIRC = 0.5*math.hypot(L_SHIP, W_SHIP)
DT10 = 10.0

def rect(cx, cy, th, l, w):
    hl, hw = 0.5*l, 0.5*w; c, s = math.cos(th), math.sin(th)
    loc = [(hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw)]
    return Polygon([(cx+x*c-y*s, cy+x*s+y*c) for x, y in loc])

# ── gap#1 证书（复制 usv_projection，纯 numpy）──────────────────────────────
def _dist_point_to_rect(p, center, theta, length, width):
    d = np.asarray(p, float) - np.asarray(center, float)
    c, s = math.cos(theta), math.sin(theta)
    lx = d[0]*c + d[1]*s; ly = -d[0]*s + d[1]*c
    return math.hypot(max(0.0, abs(lx)-0.5*length), max(0.0, abs(ly)-0.5*width))

def _lateral_reach_bound(t, v_bnd, w_max):
    half = math.pi/2; wt = w_max*t
    if wt <= half: return (v_bnd/w_max)*(1-math.cos(wt))
    return (v_bnd/w_max) + v_bnd*(t - half/w_max)

def gap1_unavoidable(ego, obs, obs_len, obs_wid, t_horizon=120.0, n_grid=241):
    a_max, w_max, v_max = A_MAX, W_MAX, V_MAX
    v_bnd = v_max + a_max*DT10
    a_eff = math.hypot(a_max, v_bnd*w_max)
    r_insc = 0.5*EGO_WIDTH
    p_e = np.array(ego[:2], float); th_e = ego[2]; v_e = ego[3]
    ve_vec = v_e*np.array([math.cos(th_e), math.sin(th_e)])
    h = np.array([math.cos(th_e), math.sin(th_e)]); hp = np.array([-h[1], h[0]])
    ox, oy, oth, ov = obs; ovec = ov*np.array([math.cos(oth), math.sin(oth)])
    for t in np.linspace(0.0, t_horizon, n_grid):
        oc = np.array([ox, oy]) + ovec*t
        c_e = p_e + ve_vec*t
        b_lon = 0.5*a_eff*t*t; b_lat = _lateral_reach_bound(t, v_bnd, w_max)
        corners = (c_e+b_lon*h+b_lat*hp, c_e+b_lon*h-b_lat*hp, c_e-b_lon*h-b_lat*hp, c_e-b_lon*h+b_lat*hp)
        if all(_dist_point_to_rect(cn, oc, oth, obs_len, obs_wid) <= r_insc for cn in corners):
            return True, float(t)
    return False, None

# ── 机动积分（常控 + 10s 钳，忠实官方 step 口径）────────────────────────────
def integ(ego0, segments, T, h=0.5):
    def seg_at(t):
        acc=0.0
        for a,w,dur in segments:
            if dur is None: return a,w
            if t < acc+dur-1e-9: return a,w
            acc+=dur
        return segments[-1][0], segments[-1][1]
    dt=0.1; nsub=int(round(h/dt))
    def rhs(x,a,w):
        v,th=x[3],x[2]
        return np.array([v*math.cos(th), v*math.sin(th), w, a])
    x=np.array(ego0,float); ts=[0.0]; traj=[x.copy()]; oseg=[]
    n=int(round(T/h))
    for i in range(n):
        a,w=seg_at(i*h); a=float(np.clip(a,-A_MAX,A_MAX)); w=float(np.clip(w,-W_MAX,W_MAX)); oseg.append(abs(w))
        for _ in range(nsub):
            k1=rhs(x,a,w); k2=rhs(x+0.5*dt*k1,a,w); k3=rhs(x+0.5*dt*k2,a,w); k4=rhs(x+dt*k3,a,w)
            x=x+(dt/6.0)*(k1+2*k2+2*k3+k4)
        t=(i+1)*h
        if abs(t/DT10-round(t/DT10))<1e-9: x[3]=float(np.clip(x[3],0.0,V_MAX))
        ts.append(t); traj.append(x.copy())
    return np.array(ts), np.array(traj), np.array(oseg)

def first_unsafe_t(ts, traj, obs, obs_len, obs_wid, h, omega_seg):
    vm=obs[3]; om=(math.cos(obs[2]), math.sin(obs[2]))
    N=len(ts); ds=np.empty(N)
    for k in range(N):
        t=ts[k]; ex,ey,eth,ev=traj[k]
        oc=(obs[0]+vm*om[0]*t, obs[1]+vm*om[1]*t)
        ds[k]=rect(ex,ey,eth,L_SHIP,W_SHIP).distance(rect(oc[0],oc[1],obs[2],obs_len,obs_wid))
    hh=ts[1]-ts[0]
    vv=traj[:,3]
    v_seg=np.maximum(np.abs(vv[:-1]),np.abs(vv[1:]))+A_MAX*hh
    ws=np.abs(omega_seg)
    w_term = ws*R_CIRC if len(ws)==N-1 else W_MAX*R_CIRC
    L=v_seg+w_term+abs(vm)
    ilb=(ds[:-1]+ds[1:]-L*hh)/2.0
    unsafe=np.where(ilb<=0.0)[0]
    return float(ts[1:][unsafe[0]]) if len(unsafe) else None

def family_window():
    A,W=A_MAX,W_MAX; fam=[]
    for a in (-A,0.0,A):
        for w in (-W,0.0,W):
            if a==0 and w==0: continue
            fam.append((f"a{a:+.2f}_w{w:+.3f}", [(a,w,None)]))
    for w in (-W,W):
        for t1 in (20.0,40.0,60.0):
            fam.append((f"turn{w:+.3f}_{int(t1)}", [(0.0,w,t1),(0.0,0.0,None)]))
            fam.append((f"acc_turn{w:+.3f}_{int(t1)}", [(A,w,t1),(A,0.0,None)]))
    for w in (-W,W):
        fam.append((f"dec30_w{w:+.3f}", [(-A,0.0,30.0),(0.0,w,None)]))
    return fam

def family_enriched():
    """加浓：更多转向时长、加/减速×转、两段/三段更丰富。测 undecided 是否 family 敏感。"""
    A,W=A_MAX,W_MAX; fam=family_window()
    for w in (-W,W):
        for t1 in (5,10,15,25,30,35,45,50,70,80,90):
            fam.append((f"turn{w:+.3f}_{t1}", [(0.0,w,float(t1)),(0.0,0.0,None)]))
            fam.append((f"acc{w:+.3f}_{t1}", [(A,w,float(t1)),(A,0.0,None)]))
            fam.append((f"dec{w:+.3f}_{t1}", [(-A,w,float(t1)),(0.0,0.0,None)]))
    # 减速一段再转（更长）+ 三段：减速→转→回正
    for w in (-W,W):
        for td in (10,20,40,60):
            fam.append((f"dec{td}_w{w:+.3f}", [(-A,0.0,float(td)),(0.0,w,None)]))
        for t1 in (20,40):
            for t2 in (20,40):
                fam.append((f"tri{w:+.3f}_{t1}_{t2}", [(0.0,w,float(t1)),(0.0,-w,float(t2)),(0.0,0.0,None)]))
    return fam

def classify(ego, obs, obs_len, obs_wid, fam, T=120.0, h=0.5):
    un, tstar = gap1_unavoidable(ego, obs, obs_len, obs_wid, t_horizon=T)
    if un: return "unavoid", {}
    clear_times={}
    for name,segs in fam:
        ts,traj,oseg=integ(ego,segs,T,h)
        clear_times[name]=first_unsafe_t(ts,traj,obs,obs_len,obs_wid,h,oseg)
    return None, clear_times

def partition(un, clear_times, Tc):
    if un: return "unavoid"
    for fut in clear_times.values():
        if fut is None or fut>Tc: return "avoid"
    return "undec"

# ── 跑合成真对撞态 ─────────────────────────────────────────────────────────
if __name__=="__main__":
  recs=[json.loads(l) for l in open("/home/user/TRB-2027-ContinuesPPO/TRB/结果/结果-block3-0722/block3_synthetic_states.jsonl")]
  NSAMP=int(sys.argv[1]) if len(sys.argv)>1 else 120
  recs=recs[:NSAMP]
  print(f"重建分类 · 合成真对撞态 n={len(recs)} · 窗口族 vs 加浓族 · T∈{{40,120}}")
  print(f"  窗口族 |{len(family_window())}| 机动 · 加浓族 |{len(family_enriched())}| 机动")

  for fam_name, fam in [("窗口族", family_window()), ("加浓族", family_enriched())]:
    t0=time.time()
    un_list=[]; ct_list=[]
    for r in recs:
        un, ct = classify(r["ego"], r["obs"], r["obs_len"], r["obs_wid"], fam, T=120.0)
        un_list.append(un); ct_list.append(ct)
    for Tc in (40.0, 120.0):
        c={"unavoid":0,"avoid":0,"undec":0}
        for un,ct in zip(un_list,ct_list):
            c[partition(bool(un), ct, Tc)]+=1
        N=len(recs)
        print(f"  [{fam_name}] T={Tc:5.0f}: unavoid {100*c['unavoid']/N:5.1f}% · avoid {100*c['avoid']/N:5.1f}% · undec {100*c['undec']/N:5.1f}%")
    print(f"    ({fam_name} 用时 {time.time()-t0:.0f}s)")
