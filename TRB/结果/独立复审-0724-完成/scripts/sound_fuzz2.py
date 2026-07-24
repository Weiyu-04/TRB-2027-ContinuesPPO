"""Efficient soundness fuzz: for each state, take the FIRST certified backup (what the gate uses as m*),
verify TRUE hull distance stays >0 over a long horizon (600s, dt=0.1). A collision = FALSE POSITIVE."""
import json, math, random
import numpy as np
import reclassify as R
from phase4_gate1_corrected import cert_v2, straight_tail_family

FAM = straight_tail_family()

def first_cert(e, o, l, w):
    for name, segs in FAM:
        if cert_v2(e, o, l, w, segs, H=120.0)["certified_perm"]:
            return name, segs
    return None, None

def fine_integ(ego0, segments, T, dt=0.1):
    def seg_at(t):
        acc = 0.0
        for a, w, dur in segments:
            if dur is None:
                return a, w
            if t < acc + dur - 1e-9:
                return a, w
            acc += dur
        return segments[-1][0], segments[-1][1]
    def rhs(x, a, w):
        v, th = x[3], x[2]
        return np.array([v*math.cos(th), v*math.sin(th), w, a])
    x = np.array(ego0, float); traj = [x.copy()]; ts = [0.0]; n = int(round(T/dt))
    for i in range(n):
        a, w = seg_at(i*dt); a = float(np.clip(a, -R.A_MAX, R.A_MAX)); w = float(np.clip(w, -R.W_MAX, R.W_MAX))
        k1 = rhs(x, a, w); k2 = rhs(x+0.5*dt*k1, a, w); k3 = rhs(x+0.5*dt*k2, a, w); k4 = rhs(x+dt*k3, a, w)
        x = x + (dt/6.0)*(k1+2*k2+2*k3+k4); tt = (i+1)*dt
        if abs(tt/R.DT10-round(tt/R.DT10)) < 1e-9:
            x[3] = float(np.clip(x[3], 0.0, R.V_MAX))
        traj.append(x.copy()); ts.append(tt)
    return np.array(ts), np.array(traj)

def true_min(ts, traj, obs, olen, owid, stride=3):
    vm = obs[3]; om = (math.cos(obs[2]), math.sin(obs[2])); best = 1e18; targ = None
    for k in range(0, len(ts), stride):
        t = ts[k]; ex, ey, eth, _ = traj[k]
        oc = (obs[0]+vm*om[0]*t, obs[1]+vm*om[1]*t)
        d = R.rect(ex, ey, eth, R.L_SHIP, R.W_SHIP).distance(R.rect(oc[0], oc[1], obs[2], olen, owid))
        if d < best:
            best = d; targ = t
    return best, targ

gw = [json.loads(l) for l in open("/home/user/TRB-2027-ContinuesPPO/TRB/结果/结果-A3-0723/a3_giveway_states_BIG.jsonl") if json.loads(l)["rho"] in (2, 3, 4)]
syn = [json.loads(l) for l in open("/home/user/TRB-2027-ContinuesPPO/TRB/结果/结果-block3-0722/block3_synthetic_states.jsonl")]
random.seed(2)
pool = [r for r in gw if r["rho"] in (2, 4)] + random.sample([r for r in gw if r["rho"] == 3], 150) + random.sample(syn, 150)
print(f"pool={len(pool)} (all head-on+overtake give-way, 150 crossing, 150 synthetic) · horizon 600s dt=0.1", flush=True)
LONG = 600.0; ncert = 0; nfp = 0; worst = 1e18; ex = []
for i, r in enumerate(pool):
    e, o, l, w = r["ego"], r["obs"], r["obs_len"], r["obs_wid"]
    if R.rect(e[0], e[1], e[2], R.L_SHIP, R.W_SHIP).distance(R.rect(o[0], o[1], o[2], l, w)) <= 0:
        continue
    name, segs = first_cert(e, o, l, w)
    if segs is None:
        continue
    ncert += 1
    ts, traj = fine_integ(e, segs, LONG, dt=0.1)
    md, tm = true_min(ts, traj, o, l, w, stride=3)
    if md < worst:
        worst = md
    if md <= 0.0:
        nfp += 1
        if len(ex) < 8:
            ex.append((name, round(md, 2), round(tm, 1), round(l), round(w, 1), r.get("rho")))
    if (i+1) % 100 == 0:
        print(f"  ...{i+1}/{len(pool)} ncert={ncert} nfp={nfp} worst={worst:.1f}", flush=True)
print(f"\n=== cert_v2 permanent-clearance soundness (fine {LONG}s) ===")
print(f" certified backups checked: {ncert}")
print(f" FALSE POSITIVES (certified but true collision over [0,{LONG}]): {nfp}")
print(f" worst true min hull-dist among certified: {worst:.2f} m")
for e_ in ex:
    print("   FP:", e_)
print(" VERDICT:", "SOUND (0 false positives)" if nfp == 0 else f"UNSOUND ({nfp} false positives)")
