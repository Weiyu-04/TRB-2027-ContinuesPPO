"""Adversarial Finding A: certifying escapes must be ADMISSIBLE = executable by the 10s-decision-step
controller = piecewise-constant with breakpoints at MULTIPLES of 10s (turn durations 10,20,...,120; NOT 5,15,25).
Re-run gate2 (compliant certified escape exists) with a 10s-ALIGNED family vs the fine family, to see if the
corrected headline (~97-100%) survives the admissibility constraint. Also verify witnesses are step-decomposable."""
import json, math, random, sys
import numpy as np
sys.path.insert(0, "/home/user/TRB-2027-ContinuesPPO/TRB/代码/trb_env")
import uterm_terminal as U   # pure module (numpy/shapely/math only)
A, W = U.A_MAX, U.W_MAX
INT = U.integrate_local_rk4

def fam_fine(sign):   # OLD (5,15,25...): first-step admissible only if t1>=10, but sub-steps switch mid-decision-step
    durs = (5.,10.,15.,20.,25.,30.,35.,40.,45.,50.,55.,60.,65.,70.,75.,80.,90.,100.,110.,120.)
    f=[]
    for t1 in durs:
        f.append([(0.0,sign*W,t1),(0.0,0.0,None)]); f.append([(-A,sign*W,t1),(0.0,0.0,None)])
    for t1 in (20.,40.,60.):
        f.append([(A,sign*W,t1),(-A,0.0,20.0),(0.0,0.0,None)])
    return f

def fam_aligned(sign):   # 10s-ALIGNED: every segment boundary at a decision-step boundary => admissible
    durs = (10.,20.,30.,40.,50.,60.,70.,80.,90.,100.,110.,120.)
    f=[]
    for t1 in durs:
        f.append([(0.0,sign*W,t1),(0.0,0.0,None)]); f.append([(-A,sign*W,t1),(0.0,0.0,None)])
    for t1 in (20.,40.,60.):
        f.append([(A,sign*W,t1),(-A,0.0,20.0),(0.0,0.0,None)])
    return f

def has_cert(e,o,l,w,fam):
    for segs in fam:
        if U.cert_v2(*_prep(e,segs,o,l,w))["certified_perm"]:
            return segs
    return None
def _prep(e,segs,o,l,w):
    ts,tj,_=INT(e,segs,120.0); return ts,tj,o,l,w,segs,0.5,120.0

recs=[json.loads(x) for x in open("/home/user/TRB-2027-ContinuesPPO/TRB/结果/结果-A3-0723/a3_giveway_states_BIG.jsonl")]
gw=[r for r in recs if r["rho"] in (2,3,4)]
random.seed(5)
NM={2:"head-on",3:"crossing",4:"overtake"}
pool=[r for r in gw if r["rho"] in (2,4)] + random.sample([r for r in gw if r["rho"]==3],150)
print(f"pool={len(pool)} (head-on63+overtake30+crossing150) · fine|{len(fam_fine(-1))}| aligned|{len(fam_aligned(-1))}|",flush=True)
st={}
for r in pool:
    e,o,l,w,rho,g=r["ego"],r["obs"],r["obs_len"],r["obs_wid"],r["rho"],r.get("give_way_dir")
    if U._rect(e[0],e[1],e[2],U.L_SHIP,U.W_SHIP).distance(U._rect(o[0],o[1],o[2],l,w))<=0: continue
    sign=-1 if rho in (2,3) else 0
    d=st.setdefault(rho,dict(n=0,fine=0,aligned=0)); d["n"]+=1
    # any-direction for overtake (sign=0): use both dirs
    if sign==0:
        ff=fam_fine(-1)+fam_fine(1); fa=fam_aligned(-1)+fam_aligned(1)
    else:
        ff=fam_fine(sign); fa=fam_aligned(sign)
    if has_cert(e,o,l,w,ff): d["fine"]+=1
    if has_cert(e,o,l,w,fa): d["aligned"]+=1
print("\n  situation   n   gate2-fine%   gate2-ALIGNED%(admissible)",flush=True)
for rho in sorted(st):
    d=st[rho]
    print(f"  {NM[rho]:9s} {d['n']:4d}   {100*d['fine']/d['n']:5.1f}   ->   {100*d['aligned']/d['n']:5.1f}",flush=True)
print("\nVERDICT: if aligned% stays ~97-100%, the corrected headline survives admissibility (family artifact fix holds on executable escapes).",flush=True)
