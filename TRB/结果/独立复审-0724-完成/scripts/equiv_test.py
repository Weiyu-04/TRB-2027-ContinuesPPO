"""Verify reclassify.first_unsafe_t (produced the results) == block3.clearance_profile (L198-verified SOUND)."""
import sys, math, random
import numpy as np
sys.path.insert(0,"/home/user/TRB-2027-ContinuesPPO/TRB/代码/m1_dock_wip")
import reclassify as R
import block3_partition_probe as B3
print("block3 _HAVE_OFFICIAL:", B3._HAVE_OFFICIAL, "(clearance_profile is pure numpy/shapely regardless)")
# Compare constants
print("consts: A_MAX", R.A_MAX==B3.A_MAX, "W_MAX", R.W_MAX==B3.W_MAX, "V_MAX", R.V_MAX==B3.V_MAX,
      "L_SHIP", R.L_SHIP==B3.L_SHIP, "W_SHIP", R.W_SHIP==B3.W_SHIP, "R_CIRC", abs(R.R_CIRC-B3.R_CIRC)<1e-9, "DT10", R.DT10==B3.DECISION_DT)
random.seed(7); rng=np.random.default_rng(7)
ndiff=0; ntest=0; maxdiff=0.0
fam=[[(a,w,None)] for a in (-R.A_MAX,0,R.A_MAX) for w in (-R.W_MAX,0,R.W_MAX)] + \
    [[(0.0,w,t1),(0.0,0.0,None)] for w in (-R.W_MAX,R.W_MAX) for t1 in (20.,40.,60.)]
for _ in range(200):
    ego=[0.0,0.0,rng.uniform(-math.pi,math.pi),rng.uniform(0,9.5)]
    obs=[rng.uniform(-1500,1500),rng.uniform(-1500,1500),rng.uniform(-math.pi,math.pi),rng.uniform(0,9.5)]
    olen=rng.uniform(175,260); owid=rng.uniform(25.4,44)
    segs=random.choice(fam)
    ts,traj,oseg=R.integ(ego,segs,120.0,h=0.5)
    fut_R=R.first_unsafe_t(ts,traj,obs,olen,owid,0.5,oseg)
    prof=B3.clearance_profile(ts,traj,obs,olen,owid,0.5,oseg)
    fut_B=prof["first_unsafe_t"]
    ntest+=1
    # compare (both None, or both float ~equal)
    if (fut_R is None) != (fut_B is None):
        ndiff+=1
        if ndiff<=5: print(f"  DIFF None-mismatch: R={fut_R} B={fut_B} ego={ego} obs={obs}")
    elif fut_R is not None:
        d=abs(fut_R-fut_B); maxdiff=max(maxdiff,d)
        if d>1e-6:
            ndiff+=1
            if ndiff<=5: print(f"  DIFF t: R={fut_R} B={fut_B}")
print(f"\n tested {ntest} (traj,obs) pairs · disagreements: {ndiff} · max |first_unsafe_t diff|: {maxdiff:.2e}")
print(" => reclassify.first_unsafe_t", "== block3.clearance_profile (EQUIVALENT, soundness transfers)" if ndiff==0 else "DIVERGES (soundness does NOT auto-transfer)")
