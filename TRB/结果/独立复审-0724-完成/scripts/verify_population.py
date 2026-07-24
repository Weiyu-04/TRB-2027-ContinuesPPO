"""Independently verify (self-run, don't trust the agent):
 (1) '0/860 give-way states are genuine collision courses' -> keep_course_min_dist for ALL 860.
 (2) head-on 95% gate2 seed-concentration -> per-seed head-on count + gate2 pass."""
import json, sys, math
import numpy as np
sys.path.insert(0, "/home/user/TRB-2027-ContinuesPPO/TRB/代码/m1_dock_wip")
import block3_partition_probe as B3
sys.path.insert(0, "/tmp/claude-0/-home-user-TRB-2027-ContinuesPPO/ca55e258-190d-50df-92f3-79eb85fefe66/scratchpad")
from phase4_gate1_corrected import cert_v2, straight_tail_family

recs = [json.loads(l) for l in open("/home/user/TRB-2027-ContinuesPPO/TRB/结果/结果-A3-0723/a3_giveway_states_BIG.jsonl")]
gw = [r for r in recs if r["rho"] in (2, 3, 4)]
NM = {2: "head-on", 3: "crossing", 4: "overtake"}

# (1) keep-course genuine conflict over all 860
print("=== (1) keep_course_min_dist over ALL 860 give-way states (kc<=0 = genuine collision course) ===")
by = {2: [], 3: [], 4: []}
for r in gw:
    kc = B3.keep_course_min_dist(r["ego"], r["obs"], r["obs_len"], r["obs_wid"], T=120.0)
    by[r["rho"]].append(kc)
tot_genuine = 0
for rho in (2, 3, 4):
    arr = np.array(by[rho]); ng = int((arr <= 0).sum()); tot_genuine += ng
    print(f"  {NM[rho]:9s} n={len(arr):4d}  genuine(kc<=0)={ng:3d}  min={arr.min():.0f}m  median={np.median(arr):.0f}m  <200m={(arr<200).sum()}")
print(f"  TOTAL genuine collision courses: {tot_genuine}/{len(gw)}")

# (2) head-on per-seed + gate2 pass
print("\n=== (2) head-on (rho=2) per-seed + compliant-backup(gate2) pass ===")
ho = [r for r in gw if r["rho"] == 2]
def compliant_right_exists(e, o, l, w):
    for name, segs in straight_tail_family():
        if segs[0][1] < -1e-9:  # first-step starboard (right = compliant for head-on)
            if cert_v2(e, o, l, w, segs, H=120.0)["certified_perm"]:
                return True
    return False
from collections import defaultdict
seed_tot = defaultdict(int); seed_pass = defaultdict(int)
for r in ho:
    ok = compliant_right_exists(r["ego"], r["obs"], r["obs_len"], r["obs_wid"])
    seed_tot[r["seed"]] += 1; seed_pass[r["seed"]] += int(ok)
tot = sum(seed_tot.values()); passed = sum(seed_pass.values())
print(f"  head-on total n={tot} · gate2 pass={passed} ({100*passed/tot:.1f}%)")
for s in sorted(seed_tot):
    print(f"   seed {s}: n={seed_tot[s]:2d} pass={seed_pass[s]:2d}")
print(f"  -> is 95% driven by one seed? largest seed = {max(seed_tot.values())}/{tot} states")
