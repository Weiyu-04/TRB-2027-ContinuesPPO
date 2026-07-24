"""Definitive compliant-backup (gate2) rate for ALL give-way states, base family vs finer STRICT-compliant family.
head-on/crossing: compliant = first-step starboard (omega<0). overtake: compliant = give_way_dir (mixed).
Locks the headline correction: is the residual a family artifact across ALL situations?"""
import json, sys, math, time
import numpy as np
sys.path.insert(0, "/tmp/claude-0/-home-user-TRB-2027-ContinuesPPO/ca55e258-190d-50df-92f3-79eb85fefe66/scratchpad")
import reclassify as R
from phase4_gate1_corrected import cert_v2
A, W = R.A_MAX, R.W_MAX
NM = {2: "head-on", 3: "crossing", 4: "overtake"}

def fam_base(sign):  # gate's own 12 per direction
    f = []
    for t1 in (10., 20., 30., 40., 60., 80.):
        f.append([(0.0, sign*W, t1), (0.0, 0.0, None)]); f.append([(-A, sign*W, t1), (0.0, 0.0, None)])
    return f
def fam_fine(sign):  # finer STRICT-compliant (first-step omega=sign*W)
    f = fam_base(sign)
    for t1 in (5., 15., 25., 35., 45., 50., 55., 65., 70., 75., 90., 100., 110., 120.):
        f.append([(0.0, sign*W, t1), (0.0, 0.0, None)]); f.append([(-A, sign*W, t1), (0.0, 0.0, None)])
    for t1 in (20., 40., 60.):
        f.append([(A, sign*W, t1), (-A, 0.0, 20.0), (0.0, 0.0, None)])
    return f
def hit(e, o, l, w, fam):
    for segs in fam:
        if cert_v2(e, o, l, w, segs, H=120.0)["certified_perm"]:
            return True
    return False
def sign_of(gw):
    return -1 if gw == "right" else (1 if gw == "left" else 0)

recs = [json.loads(l) for l in open("/home/user/TRB-2027-ContinuesPPO/TRB/结果/结果-A3-0723/a3_giveway_states_BIG.jsonl")]
gw = [r for r in recs if r["rho"] in (2, 3, 4)]
print(f"definitive gate2 over ALL give-way n={len(gw)}", flush=True)
BF = {-1: fam_base(-1), 1: fam_base(1)}
FF = {-1: fam_fine(-1), 1: fam_fine(1)}
st = {}; t0 = time.time()
for i, r in enumerate(gw):
    e, o, l, w = r["ego"], r["obs"], r["obs_len"], r["obs_wid"]
    s = sign_of(r.get("give_way_dir"))
    if s == 0:
        continue
    d = st.setdefault(r["rho"], dict(n=0, base=0, fine=0)); d["n"] += 1
    if hit(e, o, l, w, BF[s]): d["base"] += 1; d["fine"] += 1; continue
    if hit(e, o, l, w, FF[s]): d["fine"] += 1
    if (i+1) % 100 == 0: print(f"  ...{i+1}/{len(gw)} ({time.time()-t0:.0f}s)", flush=True)
print("\n  situation   n    gate2-base%    gate2-fine%(strict-compliant)", flush=True)
for rho in sorted(st):
    d = st[rho]
    print(f"  {NM[rho]:9s} {d['n']:4d}   {100*d['base']/d['n']:5.1f}   ->   {100*d['fine']/d['n']:5.1f}", flush=True)
print(f"\n[timing] {time.time()-t0:.0f}s", flush=True)
print("INTERPRETATION: if fine% >> base% for all situations, the 'residual' is a family artifact across the board.", flush=True)
