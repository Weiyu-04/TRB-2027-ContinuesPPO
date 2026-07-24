"""Numerically validate the A∩U_colregs forward-invariance CASE 1 proof (design doc §2):
 For in-A give-way states with a COMPLIANT backup m=(turn-compliant t1, straight) where t1>Δ=10s:
   apply u0 (first 10s), get s'; the tail m'=(turn-compliant t1-Δ, straight) should:
     (a) still certify s' (cert_v2 permanent) [Prop4 tail argument], AND
     (b) have first step still compliant (same give-way sign) [case1 claim]
   => s' in A∩U_colregs. Count success rate. Also report how often t1>Δ (case1 applies) vs t1<=Δ (case2).
Also: for case1 states, confirm s' actually re-admits a compliant certified escape (belt-and-suspenders)."""
import json, math, random
import numpy as np
import reclassify as R
from phase4_gate1_corrected import cert_v2, tail_after
A, W = R.A_MAX, R.W_MAX

def compliant_family(sign):
    """turn-compliant then straight, at many t1; first step omega=sign*W (compliant)."""
    f = []
    for t1 in (5.,10.,15.,20.,25.,30.,35.,40.,45.,50.,55.,60.,65.,70.,75.,80.,90.,100.,110.,120.):
        f.append((t1, [(0.0, sign*W, t1), (0.0, 0.0, None)]))
        f.append((t1, [(-A, sign*W, t1), (0.0, 0.0, None)]))
    return f

def first_compliant_backup(e,o,l,w,sign):
    for t1,segs in compliant_family(sign):
        if cert_v2(e,o,l,w,segs,H=120.0)["certified_perm"]:
            return t1,segs
    return None,None

recs=[json.loads(x) for x in open("/home/user/TRB-2027-ContinuesPPO/TRB/结果/结果-A3-0723/a3_giveway_states_BIG.jsonl")]
gw=[r for r in recs if r["rho"] in (2,3,4)]
random.seed(9)
# sample crossing (the bulk) + all head-on/overtake
pool=[r for r in gw if r["rho"] in (2,4)] + random.sample([r for r in gw if r["rho"]==3], 150)
print(f"pool={len(pool)} give-way states with a compliant backup", flush=True)
n_have=0; n_case1=0; n_case2=0
c1_tail_certifies=0; c1_tail_compliant=0; c1_sp_readmits=0
c2_readmits=0
for r in pool:
    e,o,l,w,g=r["ego"],r["obs"],r["obs_len"],r["obs_wid"],r.get("give_way_dir")
    sign=-1 if g=="right" else (1 if g=="left" else 0)
    if sign==0: continue
    if R.rect(e[0],e[1],e[2],R.L_SHIP,R.W_SHIP).distance(R.rect(o[0],o[1],o[2],l,w))<=0: continue
    t1,segs=first_compliant_backup(e,o,l,w,sign)
    if segs is None: continue
    n_have+=1
    # apply u0 = first 10s of the compliant backup
    tsm,tj,_=R.integ(e,segs,R.DT10,h=0.5); e2=list(tj[-1])
    o2=[o[0]+o[3]*math.cos(o[2])*R.DT10, o[1]+o[3]*math.sin(o[2])*R.DT10, o[2], o[3]]
    tail=tail_after(segs,R.DT10)
    tail_first_omega = tail[0][1]
    tail_compliant = (tail_first_omega < -1e-9) if sign<0 else (tail_first_omega > 1e-9)
    tail_cert = cert_v2(e2,o2,l,w,tail,H=120.0)["certified_perm"]
    # does s' re-admit ANY compliant certified escape (fresh search)?
    _,fresh = first_compliant_backup(e2,o2,l,w,sign)
    readmits = fresh is not None
    if t1 > R.DT10 + 1e-9:   # case1: turn extends beyond current step
        n_case1+=1
        c1_tail_certifies += int(tail_cert)
        c1_tail_compliant += int(tail_compliant)
        c1_sp_readmits += int(readmits)
    else:                     # case2: turn completes within the step
        n_case2+=1
        c2_readmits += int(readmits)
print(f"\n=== A∩U_colregs forward-invariance validation (states with compliant backup: {n_have}) ===")
print(f" CASE1 (compliant backup turn t1>Δ=10s): {n_case1}")
if n_case1:
    print(f"   tail still certifies s' (Prop4 tail arg):        {c1_tail_certifies}/{n_case1} = {100*c1_tail_certifies/n_case1:.1f}%")
    print(f"   tail first-step still compliant (case1 claim):    {c1_tail_compliant}/{n_case1} = {100*c1_tail_compliant/n_case1:.1f}%")
    print(f"   => s' re-admits a compliant certified escape:     {c1_sp_readmits}/{n_case1} = {100*c1_sp_readmits/n_case1:.1f}%")
print(f" CASE2 (turn completes within step, t1<=Δ): {n_case2}")
if n_case2:
    print(f"   s' STILL re-admits a compliant escape (empirical): {c2_readmits}/{n_case2} = {100*c2_readmits/n_case2:.1f}%")
print(f"\n VERDICT: case1 proof holds numerically if tail-certifies & tail-compliant ~100%; case2 residual empirically closed if re-admit ~100%.")
