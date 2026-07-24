"""Split the crossing gate2 rescue by compliance strictness:
 - base (gate's own 12): turn-W + dec-W, first-step omega<0 (strict starboard).
 - fine-starboard: base + finer/longer turn-W & dec-W durations, ALL first-step omega<0 (STRICT compliant).
 - +decstr: also decel-straight-then-starboard, first-step omega=0 (NOT strict-compliant; slow-then-turn).
Tells us: is the 46% residual an artifact of (i) too-few starboard-turn TIMINGS (strict-compliant fix),
 or (ii) needing slow-then-starboard (compliance-definition question), or (iii) genuinely infeasible."""
import json, math, random, time
import numpy as np
import reclassify as R
from phase4_gate1_corrected import cert_v2
A, W = R.A_MAX, R.W_MAX

def base_right():
    f = []
    for t1 in (10., 20., 30., 40., 60., 80.):
        f.append([(0.0, -W, t1), (0.0, 0.0, None)]); f.append([(-A, -W, t1), (0.0, 0.0, None)])
    return f
def fine_starboard():  # ALL first-step omega=-W<0 => STRICT compliant
    f = list(base_right())
    for t1 in (5., 15., 25., 35., 45., 50., 55., 65., 70., 75., 90., 100., 110., 120.):
        f.append([(0.0, -W, t1), (0.0, 0.0, None)]); f.append([(-A, -W, t1), (0.0, 0.0, None)])
    # also accel-while-starboard then decel-straight tail (first step omega<0, tail a=0)
    for t1 in (20., 40., 60.):
        f.append([(A, -W, t1), (-A, 0.0, 20.0), (0.0, 0.0, None)])
    return f
def decstr_right():  # first-step omega=0 (slow straight) then starboard: NOT strict-compliant
    f = []
    for td in (10., 20., 30., 40., 60.):
        for t1 in (20., 40., 60., 80.):
            f.append([(-A, 0.0, td), (0.0, -W, t1), (0.0, 0.0, None)])
    return f
def full_family():
    f = []
    for ww in (-W, W):
        for t1 in (10., 20., 30., 40., 60., 80.):
            f.append([(0.0, ww, t1), (0.0, 0.0, None)]); f.append([(-A, ww, t1), (0.0, 0.0, None)])
    for a in (-A, 0.0, A):
        f.append([(a, 0.0, None)])
    return f

def hit(e, o, l, w, fam):
    for segs in fam:
        if cert_v2(e, o, l, w, segs, H=120.0)["certified_perm"]:
            return True
    return False

FULL = full_family(); BR = base_right(); FS = fine_starboard(); DS = decstr_right()
gw = [json.loads(l) for l in open("/home/user/TRB-2027-ContinuesPPO/TRB/结果/结果-A3-0723/a3_giveway_states_BIG.jsonl") if json.loads(l)["rho"] == 3]
random.seed(5); samp = random.sample(gw, 200)
print(f"crossing n={len(gw)} sample={len(samp)} · base|{len(BR)}| fine-starboard|{len(FS)}| +decstr|{len(DS)}|", flush=True)
N = 0; nA = 0; b = 0; fs = 0; fs_or_ds = 0; t0 = time.time()
for i, r in enumerate(samp):
    e, o, l, w = r["ego"], r["obs"], r["obs_len"], r["obs_wid"]
    if R.rect(e[0], e[1], e[2], R.L_SHIP, R.W_SHIP).distance(R.rect(o[0], o[1], o[2], l, w)) <= 0:
        continue
    N += 1
    inA = hit(e, o, l, w, FULL)
    if inA: nA += 1
    if hit(e, o, l, w, BR): b += 1; fs += 1; fs_or_ds += 1; continue
    if hit(e, o, l, w, FS): fs += 1; fs_or_ds += 1; continue
    if hit(e, o, l, w, DS): fs_or_ds += 1
    if (i+1) % 50 == 0: print(f"  ...{i+1} ({time.time()-t0:.0f}s)", flush=True)
print(f"\n=== crossing compliant-backup coverage (n={N}) ===")
print(f" A-membership (any dir):            {nA}/{N} = {100*nA/N:.1f}%")
print(f" base starboard (gate's 12, first-step omega<0):        {b}/{N} = {100*b/N:.1f}%   [reported 54%]")
print(f" + finer STRICT-starboard timings (still first-step omega<0): {fs}/{N} = {100*fs/N:.1f}%")
print(f" + slow-straight-then-starboard (first-step omega=0, permissive): {fs_or_ds}/{N} = {100*fs_or_ds/N:.1f}%")
print(f"\n INTERPRETATION:")
print(f"  strict-compliant residual (A-member but NO strict starboard escape) ~ {100*(nA-fs)/N:.1f}% of all")
print(f"  genuinely no compliant escape even permissive ~ {100*(nA-fs_or_ds)/N:.1f}% of A-members-ish")
