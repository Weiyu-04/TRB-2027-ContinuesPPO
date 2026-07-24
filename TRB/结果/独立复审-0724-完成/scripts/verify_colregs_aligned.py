"""A∩U_colregs forward-invariance with the 10s-ALIGNED (admissible) family.
With aligned family the shortest compliant turn is 10s, so CASE2 (turn completes in one step, t1==Δ) CAN occur
(unlike the fine family). Report CASE1/CASE2 split + whether s' re-admits a fresh compliant escape (robustness)."""
import json, math, random, sys
import numpy as np
sys.path.insert(0, "/home/user/TRB-2027-ContinuesPPO/TRB/代码/trb_env")
import uterm_terminal as U
A, W = U.A_MAX, U.W_MAX
INT = U.integrate_local_rk4

def compliant_family(sign):   # 10s-aligned compliant (turn/dec), shortest-first
    f = []
    for t1 in (10., 20., 30., 40., 50., 60., 70., 80., 90., 100., 110., 120.):
        f.append((t1, [(0.0, sign*W, t1), (0.0, 0.0, None)]))
        f.append((t1, [(-A, sign*W, t1), (0.0, 0.0, None)]))
    return f

def first_compliant(e, o, l, w, sign):
    for t1, segs in compliant_family(sign):
        ts, tj, _ = INT(e, segs, 120.0)
        if U.cert_v2(ts, tj, o, l, w, segs, 0.5, 120.0)["certified_perm"]:
            return t1, segs
    return None, None

recs = [json.loads(x) for x in open("/home/user/TRB-2027-ContinuesPPO/TRB/结果/结果-A3-0723/a3_giveway_states_BIG.jsonl")]
gw = [r for r in recs if r["rho"] in (2, 3, 4)]
random.seed(9)
pool = [r for r in gw if r["rho"] in (2, 4)] + random.sample([r for r in gw if r["rho"] == 3], 150)
print(f"pool={len(pool)} · ALIGNED compliant family", flush=True)
n_have = 0; n_c1 = 0; n_c2 = 0; c1_tail_ok = 0; c1_tail_comp = 0; readmit = 0
for r in pool:
    e, o, l, w, g = r["ego"], r["obs"], r["obs_len"], r["obs_wid"], r.get("give_way_dir")
    sign = -1 if g == "right" else (1 if g == "left" else 0)
    if sign == 0: continue
    if U._rect(e[0], e[1], e[2], U.L_SHIP, U.W_SHIP).distance(U._rect(o[0], o[1], o[2], l, w)) <= 0: continue
    t1, segs = first_compliant(e, o, l, w, sign)
    if segs is None: continue
    n_have += 1
    ts, tj, _ = INT(e, segs, U.DECISION_DT); e2 = list(tj[-1])
    o2 = [o[0]+o[3]*math.cos(o[2])*U.DECISION_DT, o[1]+o[3]*math.sin(o[2])*U.DECISION_DT, o[2], o[3]]
    tail = U._tail_after(segs, U.DECISION_DT)
    tfo = tail[0][1]
    tail_comp = (tfo < -1e-9) if sign < 0 else (tfo > 1e-9)
    tsa, tja, _ = INT(e2, tail, 120.0)
    tail_cert = U.cert_v2(tsa, tja, o2, l, w, tail, 0.5, 120.0)["certified_perm"]
    _, fresh = first_compliant(e2, o2, l, w, sign)
    if fresh is not None: readmit += 1
    if t1 > U.DECISION_DT + 1e-9:
        n_c1 += 1; c1_tail_ok += int(tail_cert); c1_tail_comp += int(tail_comp)
    else:   # t1 == 10 (turn completes in one step) => CASE2
        n_c2 += 1
print(f"\n=== A∩U_colregs (ALIGNED·states w/ compliant backup: {n_have}) ===", flush=True)
print(f" CASE1 (t1>10s): {n_c1}  · tail-certifies {c1_tail_ok}/{max(1,n_c1)} · tail-compliant {c1_tail_comp}/{max(1,n_c1)}", flush=True)
print(f" CASE2 (t1==10s, completes in one step): {n_c2}", flush=True)
print(f" s' re-admits a fresh compliant certified escape: {readmit}/{n_have} = {100*readmit/max(1,n_have):.1f}%", flush=True)
print(" => invariance holds if (CASE1 tail-ok/compliant ~100%) AND (re-admit ~100% covers CASE2). DONE", flush=True)
