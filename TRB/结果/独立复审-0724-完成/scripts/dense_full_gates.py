"""Full gate table (A-membership + gate1 1a/1b/1c + gate2 compliant-backup) with the DENSE maneuver family,
over ALL 860 give-way states. Produces the CORRECTED, consistent numbers for the paper.
Reports gate2 under BOTH denominators (full-population = gw_gates conv, and in-A-only = a3-probe conv)."""
import json, math, time
import numpy as np
import reclassify as R
from phase4_gate1_corrected import cert_v2, tail_after
A, W = R.A_MAX, R.W_MAX
NM = {2: "head-on", 3: "crossing", 4: "overtake"}

# Dense family: finer turn durations both directions + accel-turn-then-decel tail (all end constant-velocity straight)
def dense_family(signs=(-1, 1)):
    f = []
    durs = (5., 10., 15., 20., 25., 30., 35., 40., 45., 50., 55., 60., 65., 70., 75., 80., 90., 100., 110., 120.)
    for s in signs:
        for t1 in durs:
            f.append((f"turn{s}_{int(t1)}", [(0.0, s*W, t1), (0.0, 0.0, None)]))
            f.append((f"dec{s}_{int(t1)}", [(-A, s*W, t1), (0.0, 0.0, None)]))
        for t1 in (20., 40., 60.):
            f.append((f"acc{s}_{int(t1)}", [(A, s*W, t1), (-A, 0.0, 20.0), (0.0, 0.0, None)]))
    for a in (-A, 0.0, A):
        f.append((f"straight{a:+.2f}", [(a, 0.0, None)]))
    return f

FAM_ALL = dense_family((-1, 1))
FAM_R = dense_family((-1,))   # starboard only
FAM_L = dense_family((1,))    # port only

def first_cert(e, o, l, w, fam):
    for name, segs in fam:
        if cert_v2(e, o, l, w, segs, H=120.0)["certified_perm"]:
            return name, segs
    return None, None

def range_rate(e, o):
    p = np.array([e[0]-o[0], e[1]-o[1]]); n = np.hypot(*p)
    if n < 1e-9: return 0.0
    v = e[3]*np.array([math.cos(e[2]), math.sin(e[2])]) - o[3]*np.array([math.cos(o[2]), math.sin(o[2])])
    return float(p @ v / n)

recs = [json.loads(l) for l in open("/home/user/TRB-2027-ContinuesPPO/TRB/结果/结果-A3-0723/a3_giveway_states_BIG.jsonl")]
gw = [r for r in recs if r["rho"] in (2, 3, 4)]
print(f"DENSE full gate table · give-way n={len(gw)} · family |{len(FAM_ALL)}|", flush=True)
st = {}; t0 = time.time()
for i, r in enumerate(gw):
    e, o, l, w, rho, g = r["ego"], r["obs"], r["obs_len"], r["obs_wid"], r["rho"], r.get("give_way_dir")
    d = st.setdefault(rho, dict(n=0, inA=0, g1a=0, g1b=0, g1c=0, cb=0, cb_full=0, cb_inA=0))
    d["n"] += 1
    name, segs = first_cert(e, o, l, w, FAM_ALL)
    sign = -1 if g == "right" else (1 if g == "left" else 0)
    compliant_fam = FAM_R if sign < 0 else (FAM_L if sign > 0 else None)
    has_compliant = (compliant_fam is not None) and (first_cert(e, o, l, w, compliant_fam)[1] is not None)
    if segs is None:
        # not in A: gate2 full-pop denominator counts it as fail; in-A denominator excludes it
        if sign != 0:
            d["cb_full"] += 1  # denominator only (cb not incremented)
        continue
    d["inA"] += 1
    if sign != 0:
        d["cb_full"] += 1; d["cb_inA"] += 1
        d["cb"] += int(has_compliant)
    # gate1: step 10s along backup, re-certify tail
    tsm, tj, _ = R.integ(e, segs, R.DT10, h=0.5); e2 = list(tj[-1])
    o2 = [o[0]+o[3]*math.cos(o[2])*R.DT10, o[1]+o[3]*math.sin(o[2])*R.DT10, o[2], o[3]]
    tl = tail_after(segs, R.DT10)
    c1a = cert_v2(e2, o2, l, w, tl, H=120.0); c1b = cert_v2(e2, o2, l, w, tl, H=110.0)
    tsT, tjT, _ = R.integ(e2, tl, 120.0, h=0.5); tH = tsT[-1]
    oT = [o2[0]+o2[3]*math.cos(o2[2])*tH, o2[1]+o2[3]*math.sin(o2[2])*tH, o2[2], o2[3]]
    d["g1a"] += int(c1a["certified_perm"]); d["g1b"] += int(c1b["certified_perm"]); d["g1c"] += int(range_rate(list(tjT[-1]), oT) > 0)
    if (i+1) % 100 == 0: print(f"  ...{i+1}/{len(gw)} ({time.time()-t0:.0f}s)", flush=True)

print("\n  situation   n   A-mem%  门1(1a/1b/1c)%   gate2(full-pop%)  gate2(in-A%)", flush=True)
for rho in sorted(st):
    d = st[rho]; nA = max(1, d["inA"])
    g2f = 100*d["cb"]/max(1, d["cb_full"]); g2a = 100*d["cb"]/max(1, d["cb_inA"])
    print(f"  {NM[rho]:9s} {d['n']:4d}  {100*d['inA']/d['n']:5.1f}  {100*d['g1a']/nA:.1f}/{100*d['g1b']/nA:.1f}/{100*d['g1c']/nA:.1f}   {g2f:5.1f} ({d['cb']}/{d['cb_full']})  {g2a:5.1f} ({d['cb']}/{d['cb_inA']})", flush=True)
print(f"\n[timing] {time.time()-t0:.0f}s · DENSE family DONE", flush=True)
