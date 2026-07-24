"""Definitive full gate table with the 10s-ALIGNED (admissible/executable) family, using the production
uterm_terminal module directly. All 860 give-way states. A-membership + gate1(1a/1b/1c) + gate2(both denoms)."""
import json, math, time, sys
import numpy as np
sys.path.insert(0, "/home/user/TRB-2027-ContinuesPPO/TRB/代码/trb_env")
import uterm_terminal as U
INT = U.integrate_local_rk4
NM = {2: "head-on", 3: "crossing", 4: "overtake"}

def first_cert(e, o, l, w, sign):
    for name, segs, w0 in U.straight_tail_family():
        if sign < 0 and not (w0 < -1e-9): continue
        if sign > 0 and not (w0 > 1e-9): continue
        ts, tj, _ = INT(e, segs, 120.0)
        if U.cert_v2(ts, tj, o, l, w, segs, 0.5, 120.0)["certified_perm"]:
            return segs
    return None

def range_rate(e, o):
    p = np.array([e[0]-o[0], e[1]-o[1]]); n = np.hypot(*p)
    if n < 1e-9: return 0.0
    v = e[3]*np.array([math.cos(e[2]), math.sin(e[2])]) - o[3]*np.array([math.cos(o[2]), math.sin(o[2])])
    return float(p @ v / n)

recs = [json.loads(x) for x in open("/home/user/TRB-2027-ContinuesPPO/TRB/结果/结果-A3-0723/a3_giveway_states_BIG.jsonl")]
gw = [r for r in recs if r["rho"] in (2, 3, 4)]
print(f"ALIGNED family full gate · give-way n={len(gw)} · family|{len(U.straight_tail_family())}|", flush=True)
st = {}; t0 = time.time()
for i, r in enumerate(gw):
    e, o, l, w, rho, g = r["ego"], r["obs"], r["obs_len"], r["obs_wid"], r["rho"], r.get("give_way_dir")
    d = st.setdefault(rho, dict(n=0, inA=0, g1a=0, g1b=0, g1c=0, cb=0, cb_full=0, cb_inA=0)); d["n"] += 1
    segs = first_cert(e, o, l, w, 0)                       # A-membership (any dir)
    sign = -1 if g == "right" else (1 if g == "left" else 0)
    has_comp = (sign != 0) and (first_cert(e, o, l, w, sign) is not None)
    if segs is None:
        if sign != 0: d["cb_full"] += 1
        continue
    d["inA"] += 1
    if sign != 0:
        d["cb_full"] += 1; d["cb_inA"] += 1; d["cb"] += int(has_comp)
    ts, tj, _ = INT(e, segs, U.DECISION_DT); e2 = list(tj[-1])
    o2 = [o[0]+o[3]*math.cos(o[2])*U.DECISION_DT, o[1]+o[3]*math.sin(o[2])*U.DECISION_DT, o[2], o[3]]
    tl = U._tail_after(segs, U.DECISION_DT)
    ts1a, tj1a, _ = INT(e2, tl, 120.0); c1a = U.cert_v2(ts1a, tj1a, o2, l, w, tl, 0.5, 120.0)
    ts1b, tj1b, _ = INT(e2, tl, 110.0); c1b = U.cert_v2(ts1b, tj1b, o2, l, w, tl, 0.5, 110.0)
    tH = ts1a[-1]; oT = [o2[0]+o2[3]*math.cos(o2[2])*tH, o2[1]+o2[3]*math.sin(o2[2])*tH, o2[2], o2[3]]
    d["g1a"] += int(c1a["certified_perm"]); d["g1b"] += int(c1b["certified_perm"]); d["g1c"] += int(range_rate(list(tj1a[-1]), oT) > 0)
    if (i+1) % 100 == 0: print(f"  ...{i+1}/{len(gw)} ({time.time()-t0:.0f}s)", flush=True)
print("\n  situation   n   A-mem%  门1(1a/1b/1c)%   gate2(full%)   gate2(in-A%)", flush=True)
for rho in sorted(st):
    d = st[rho]; nA = max(1, d["inA"])
    g2f = 100*d["cb"]/max(1, d["cb_full"]); g2a = 100*d["cb"]/max(1, d["cb_inA"])
    print(f"  {NM[rho]:9s} {d['n']:4d}  {100*d['inA']/d['n']:5.1f}  {100*d['g1a']/nA:.1f}/{100*d['g1b']/nA:.1f}/{100*d['g1c']/nA:.1f}   {g2f:5.1f}({d['cb']}/{d['cb_full']})  {g2a:5.1f}({d['cb']}/{d['cb_inA']})", flush=True)
print(f"\n[timing] {time.time()-t0:.0f}s · ALIGNED family DONE", flush=True)
