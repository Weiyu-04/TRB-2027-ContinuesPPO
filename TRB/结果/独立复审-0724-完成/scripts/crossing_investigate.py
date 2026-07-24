"""Independent investigation of crossing gate2 ~46% residual: genuine A∩U_colregs=empty vs family/cert artifact.
For sampled crossing give-way states: classify A-membership, right/left/straight backup, keep-course conflict,
and for gate2-failing A-members test an ENRICHED right-turn family (finer t1, decel-then-right, dec+right).
If enriched right closes many failures -> residual overstated (family artifact). If not -> genuine tension."""
import json, math, random, time
import numpy as np
import reclassify as R
from phase4_gate1_corrected import cert_v2

A, W = R.A_MAX, R.W_MAX

def base_right():
    fam = []
    for t1 in (10., 20., 30., 40., 60., 80.):
        fam.append(("turn-W_%d" % t1, [(0.0, -W, t1), (0.0, 0.0, None)]))
        fam.append(("dec-W_%d" % t1, [(-A, -W, t1), (0.0, 0.0, None)]))
    return fam

def enriched_right():
    fam = list(base_right())
    # finer + longer turn durations
    for t1 in (5., 15., 25., 35., 45., 50., 70., 90., 100., 120.):
        fam.append(("turn-W_%d" % t1, [(0.0, -W, t1), (0.0, 0.0, None)]))
        fam.append(("dec-W_%d" % t1, [(-A, -W, t1), (0.0, 0.0, None)]))
    # decelerate straight first, then hard right (buy time then turn)
    for td in (10., 20., 30., 40., 60.):
        for t1 in (20., 40., 60., 80.):
            fam.append(("decstr%d_turnR%d" % (td, t1), [(-A, 0.0, td), (0.0, -W, t1), (0.0, 0.0, None)]))
    # accelerate+right then straight (already in family as acc but tail a!=0 never certifies; skip)
    return fam

def any_backup(e, o, l, w, fam):
    for name, segs in fam:
        if cert_v2(e, o, l, w, segs, H=120.0)["certified_perm"]:
            return name
    return None

# full any-direction family (for A-membership) = right + left + straight
def full_family():
    fam = []
    for ww in (-W, W):
        for t1 in (10., 20., 30., 40., 60., 80.):
            fam.append(("turn%.3f_%d" % (ww, t1), [(0.0, ww, t1), (0.0, 0.0, None)]))
            fam.append(("dec%.3f_%d" % (ww, t1), [(-A, ww, t1), (0.0, 0.0, None)]))
    for a in (-A, 0.0, A):
        fam.append(("straight%.2f" % a, [(a, 0.0, None)]))
    return fam

def left_family():
    fam = []
    for t1 in (10., 20., 30., 40., 60., 80.):
        fam.append(("turn+W_%d" % t1, [(0.0, W, t1), (0.0, 0.0, None)]))
        fam.append(("dec+W_%d" % t1, [(-A, W, t1), (0.0, 0.0, None)]))
    return fam

FULL = full_family(); BR = base_right(); ER = enriched_right(); LF = left_family()
gw = [json.loads(l) for l in open("/home/user/TRB-2027-ContinuesPPO/TRB/结果/结果-A3-0723/a3_giveway_states_BIG.jsonl") if json.loads(l)["rho"] == 3]
random.seed(5)
samp = random.sample(gw, 200)
print(f"crossing give-way total={len(gw)} · sampled {len(samp)} · base-right |{len(BR)}| enriched-right |{len(ER)}|", flush=True)

nA = 0; n_rightbase = 0; n_conflict = 0
# buckets among gate2(base-right) failures
fail_notA = 0          # not in A at all
fail_A_only_left = 0   # A-member, base-right fails, left works
fail_A_neither = 0     # A-member, base-right fails, left also fails (only straight?)
enriched_rescued = 0   # base-right failed A-members that enriched-right rescues
enriched_still_fail = 0
t0 = time.time()
for i, r in enumerate(samp):
    e, o, l, w = r["ego"], r["obs"], r["obs_len"], r["obs_wid"]
    if R.rect(e[0], e[1], e[2], R.L_SHIP, R.W_SHIP).distance(R.rect(o[0], o[1], o[2], l, w)) <= 0:
        continue
    kc = None
    # keep-course conflict?
    inA = any_backup(e, o, l, w, FULL) is not None
    rb = any_backup(e, o, l, w, BR) is not None
    if inA: nA += 1
    if rb: n_rightbase += 1
    if not rb:  # gate2 (base) fails
        if not inA:
            fail_notA += 1
        else:
            lb = any_backup(e, o, l, w, LF) is not None
            if lb: fail_A_only_left += 1
            else: fail_A_neither += 1
            # try enriched right
            er = any_backup(e, o, l, w, ER) is not None
            if er: enriched_rescued += 1
            else: enriched_still_fail += 1
    if (i+1) % 50 == 0:
        print(f"  ...{i+1}/{len(samp)} ({time.time()-t0:.0f}s) A={nA} rightbase={n_rightbase}", flush=True)

N = len(samp)
print(f"\n=== crossing gate2 residual anatomy (n={N}) ===")
print(f" A-membership (any-dir backup): {nA}/{N} = {100*nA/N:.1f}%")
print(f" base-right compliant backup (gate2): {n_rightbase}/{N} = {100*n_rightbase/N:.1f}%   [reported ~54%]")
print(f" gate2 FAILURES broken down:")
print(f"   - not in A at all (unavoidable/undecided, falls to fallback regardless): {fail_notA}")
print(f"   - A-member, no right, but LEFT works (genuine COLREGs∩safety tension): {fail_A_only_left}")
print(f"   - A-member, neither right nor left (only straight/decel clears): {fail_A_neither}")
print(f" ENRICHED right-turn family on base-right-failing A-members:")
print(f"   - rescued by enriched right (=> base residual OVERSTATED, family artifact): {enriched_rescued}")
print(f"   - still fail with enriched right (=> genuine right-direction infeasible): {enriched_still_fail}")
if n_rightbase < N:
    print(f"\n INTERPRETATION: of {N-n_rightbase} gate2 failures, {fail_notA} are not-in-A (fallback anyway),")
    print(f"   {fail_A_only_left+fail_A_neither} are A-members without a base-right escape; enriched right rescues {enriched_rescued} of them.")
