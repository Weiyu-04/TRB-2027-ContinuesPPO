"""Independent numerical verification of B1 hocbf_constraint derivation and qp_project geometry."""
import sys, math
import numpy as np
sys.path.insert(0,"/home/user/TRB-2027-ContinuesPPO/TRB/代码/m1_dock_wip")
import b1_cbf_baseline as B1

# ---- 1. HOCBF derivation: verify hddot = 2 vrel.vrel + A_coef*a + W_coef*omega via finite diff of h ----
def h_of(state, obs):
    p=np.array([state[0]-obs[0], state[1]-obs[1]]); return float(p@p)  # d_safe cancels in derivatives
def dyn(state, u, dt):
    a,w=u; x=np.array(state,float)
    x=x+dt*np.array([x[3]*math.cos(x[2]), x[3]*math.sin(x[2]), w, a]); return x
rng=np.random.default_rng(3); bad=0; worst=0.0
for _ in range(2000):
    ego=[rng.uniform(-500,500),rng.uniform(-500,500),rng.uniform(-math.pi,math.pi),rng.uniform(0.5,9.5)]
    obs=[rng.uniform(-500,500),rng.uniform(-500,500),rng.uniform(-math.pi,math.pi),rng.uniform(0,9.5)]
    u=[rng.uniform(-0.24,0.24), rng.uniform(-0.03,0.03)]
    ovec=obs[3]*np.array([math.cos(obs[2]),math.sin(obs[2])])
    # analytic hddot from code coefficients
    px,py,th,v=ego; p_rel=np.array([px-obs[0],py-obs[1]])
    v_ego=v*np.array([math.cos(th),math.sin(th)]); v_obs=ovec; v_rel=v_ego-v_obs
    A_coef=2.0*float(p_rel@np.array([math.cos(th),math.sin(th)]))
    W_coef=2.0*float(p_rel@np.array([-math.sin(th),math.cos(th)]))*v
    hddot_analytic = 2.0*float(v_rel@v_rel) + A_coef*u[0] + W_coef*u[1]
    # finite-diff hddot: h(t) with ego under (a,w), obs under CV
    dt=1e-3
    def hh(t):
        # integrate ego with constant u for small t (RK-ish exact-ish via fine steps), obs CV
        st=np.array(ego,float); n=max(1,int(t/1e-4)); ddt=t/n
        for _ in range(n):
            st=st+ddt*np.array([st[3]*math.cos(st[2]), st[3]*math.sin(st[2]), u[1], u[0]])
        ob=np.array([obs[0]+ovec[0]*t, obs[1]+ovec[1]*t])
        p=np.array([st[0]-ob[0], st[1]-ob[1]]); return float(p@p)
    h0=hh(0.0); hp=hh(dt); hm_=h_of(ego,obs)  # h0 should == hm_
    hddot_fd=(hh(dt)-2*hh(0.0)+hh(-dt))/(dt*dt) if False else (hh(2*dt)-2*hh(dt)+hh(0.0))/(dt*dt)
    err=abs(hddot_fd-hddot_analytic); worst=max(worst,err)
    if err>1e-2*max(1.0,abs(hddot_analytic)): 
        bad+=1
        if bad<=3: print(f"  HOCBF mismatch: analytic={hddot_analytic:.4f} fd={hddot_fd:.4f}")
print(f"[HOCBF derivation] {2000-bad}/2000 match · worst rel-ish err {worst:.2e} → {'CORRECT' if bad==0 else str(bad)+' MISMATCH'}")

# ---- 2. qp_project: verify against brute-force projection onto {g.u<=b} ∩ box ----
def brute(u_nom,g,b,box):
    a_lo,a_hi,w_lo,w_hi=box; best=None; bd=1e18
    for a in np.linspace(a_lo,a_hi,401):
        for w in np.linspace(w_lo,w_hi,401):
            if g[0]*a+g[1]*w<=b+1e-9:
                d=(a-u_nom[0])**2+(w-u_nom[1])**2
                if d<bd: bd=d; best=(a,w)
    return best
box=(-0.24,0.24,-0.03,0.03); rng=np.random.default_rng(11); nbad=0; wgap=0.0; ninf=0
for _ in range(400):
    un=[rng.uniform(-0.3,0.3),rng.uniform(-0.04,0.04)]
    g=np.array([rng.uniform(-2,2),rng.uniform(-2,2)]); b=rng.uniform(-0.1,0.1)
    u,feas=B1.qp_project(un,g,b,box)
    bf=brute(un,g,b,box)
    if bf is None:
        if feas: nbad+=1
        else: ninf+=1
        continue
    if not feas: nbad+=1; continue
    # check constraint satisfied
    if g@u>b+1e-6: nbad+=1; continue
    gap=math.hypot(u[0]-bf[0],u[1]-bf[1])  # coarse brute grid ~ 0.0012 spacing in a, 0.00015 in w
    wgap=max(wgap,gap)
print(f"[qp_project] mismatches {nbad}/400 · infeasible-detected {ninf} · worst dist-to-brute-optimum {wgap:.4f} (brute grid ~1e-3) → {'OK' if nbad==0 and wgap<0.01 else 'CHECK'}")
