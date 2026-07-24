"""Confirm the fine-starboard RESCUES are genuinely collision-free (not a cert_v2 bug on these shapes):
for crossing states that FAIL base-right but PASS fine-starboard, integrate the fine-starboard certified
maneuver finely to 600s and confirm true min hull dist > 0. Also report the genuine residual states."""
import json, math, random
import numpy as np
import reclassify as R
from phase4_gate1_corrected import cert_v2
A, W = R.A_MAX, R.W_MAX

def base_right():
    f = []
    for t1 in (10., 20., 30., 40., 60., 80.):
        f.append([(0.0, -W, t1), (0.0, 0.0, None)]); f.append([(-A, -W, t1), (0.0, 0.0, None)])
    return f
def fine_starboard():
    f = list(base_right())
    for t1 in (5., 15., 25., 35., 45., 50., 55., 65., 70., 75., 90., 100., 110., 120.):
        f.append([(0.0, -W, t1), (0.0, 0.0, None)]); f.append([(-A, -W, t1), (0.0, 0.0, None)])
    for t1 in (20., 40., 60.):
        f.append([(A, -W, t1), (-A, 0.0, 20.0), (0.0, 0.0, None)])
    return f
def first_cert(e, o, l, w, fam):
    for segs in fam:
        if cert_v2(e, o, l, w, segs, H=120.0)["certified_perm"]:
            return segs
    return None
def fine_integ(ego0, segments, T, dt=0.1):
    def seg_at(t):
        acc = 0.0
        for a, w, dur in segments:
            if dur is None: return a, w
            if t < acc + dur - 1e-9: return a, w
            acc += dur
        return segments[-1][0], segments[-1][1]
    def rhs(x, a, w):
        v, th = x[3], x[2]; return np.array([v*math.cos(th), v*math.sin(th), w, a])
    x = np.array(ego0, float); traj = [x.copy()]; ts = [0.0]; n = int(round(T/dt))
    for i in range(n):
        a, w = seg_at(i*dt); a = float(np.clip(a, -A, A)); w = float(np.clip(w, -W, W))
        k1 = rhs(x, a, w); k2 = rhs(x+0.5*dt*k1, a, w); k3 = rhs(x+0.5*dt*k2, a, w); k4 = rhs(x+dt*k3, a, w)
        x = x + (dt/6.0)*(k1+2*k2+2*k3+k4); tt = (i+1)*dt
        if abs(tt/R.DT10-round(tt/R.DT10)) < 1e-9: x[3] = float(np.clip(x[3], 0.0, R.V_MAX))
        traj.append(x.copy()); ts.append(tt)
    return np.array(ts), np.array(traj)
def true_min(ts, traj, obs, olen, owid, stride=2):
    vm = obs[3]; om = (math.cos(obs[2]), math.sin(obs[2])); best = 1e18
    for k in range(0, len(ts), stride):
        t = ts[k]; ex, ey, eth, _ = traj[k]
        oc = (obs[0]+vm*om[0]*t, obs[1]+vm*om[1]*t)
        d = R.rect(ex, ey, eth, R.L_SHIP, R.W_SHIP).distance(R.rect(oc[0], oc[1], obs[2], olen, owid))
        if d < best: best = d
    return best

BR = base_right(); FS = fine_starboard()
gw = [json.loads(l) for l in open("/home/user/TRB-2027-ContinuesPPO/TRB/结果/结果-A3-0723/a3_giveway_states_BIG.jsonl") if json.loads(l)["rho"] == 3]
random.seed(5); samp = random.sample(gw, 200)
print(f"checking fine-starboard rescues for collision-freeness (fine 600s integ)", flush=True)
n_rescue = 0; n_bad = 0; worst = 1e18; n_genuine_resid = 0
for r in samp:
    e, o, l, w = r["ego"], r["obs"], r["obs_len"], r["obs_wid"]
    if R.rect(e[0], e[1], e[2], R.L_SHIP, R.W_SHIP).distance(R.rect(o[0], o[1], o[2], l, w)) <= 0: continue
    if first_cert(e, o, l, w, BR) is not None: continue  # base already compliant, skip
    segs = first_cert(e, o, l, w, FS)
    if segs is None:
        n_genuine_resid += 1  # even fine-starboard cannot clear compliantly
        continue
    n_rescue += 1
    ts, traj = fine_integ(e, segs, 600.0, dt=0.1)
    md = true_min(ts, traj, o, l, w, stride=2)
    if md < worst: worst = md
    if md <= 0.0:
        n_bad += 1
        if n_bad <= 5: print(f"  BAD RESCUE (cert bug!): md={md:.2f} segs={segs}")
print(f"\n=== fine-starboard rescue soundness ===")
print(f" rescued states (base fail -> fine-starboard cert): {n_rescue}")
print(f" of those, FALSE (true collision over 600s = cert bug): {n_bad}")
print(f" worst true min hull-dist among rescues: {worst:.2f} m")
print(f" genuine compliant residual (even fine-starboard fails): {n_genuine_resid}/200 = {100*n_genuine_resid/200:.1f}%")
print(" VERDICT:", "rescues SOUND (real compliant escapes) => 46% residual was a family artifact" if n_bad==0 else f"{n_bad} bad rescues (cert issue)")
