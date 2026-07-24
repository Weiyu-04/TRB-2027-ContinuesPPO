import json, math, sys, time
import numpy as np
import reclassify as R
from phase4_gate1_corrected import cert_v2, straight_tail_family, tail_after
DT10 = R.DT10; NM = {2:"head-on",3:"crossing",4:"overtake"}

def cs(gw): return -1 if gw=="right" else (1 if gw=="left" else 0)
def fb(e,o,l,w,s=0):
    for nm,sg in straight_tail_family():
        w0=sg[0][1]
        if s<0 and not (w0<-1e-9): continue
        if s>0 and not (w0>1e-9): continue
        if cert_v2(e,o,l,w,sg)["certified_perm"]: return sg
    return None
def rr(e,o):
    p=np.array([e[0]-o[0],e[1]-o[1]]); n=np.hypot(*p)
    if n<1e-9: return 0.0
    v=e[3]*np.array([math.cos(e[2]),math.sin(e[2])])-o[3]*np.array([math.cos(o[2]),math.sin(o[2])])
    return float(p@v/n)

recs=[json.loads(l) for l in open("/home/user/TRB-2027-ContinuesPPO/TRB/结果/结果-A3-0723/a3_giveway_states_BIG.jsonl")]
gw=[r for r in recs if r["rho"] in (2,3,4)]
print(f"[INDEP REPRO] give-way n={len(gw)} (head-on {sum(1 for r in gw if r['rho']==2)}/crossing {sum(1 for r in gw if r['rho']==3)}/overtake {sum(1 for r in gw if r['rho']==4)})",flush=True)
st={}; t0=time.time()
# store per-state details for overtake 1c investigation
ovt_detail=[]
for i,r in enumerate(gw):
    e,o,l,w,rho,g=r["ego"],r["obs"],r["obs_len"],r["obs_wid"],r["rho"],r.get("give_way_dir")
    d=st.setdefault(rho,dict(n=0,inA=0,g1a=0,g1b=0,g1c=0,cb=0,cbt=0)); d["n"]+=1
    sg=fb(e,o,l,w,0)
    if sg is None:
        if cs(g)!=0: d["cbt"]+=1
        continue
    d["inA"]+=1
    tsm,tj,_=R.integ(e,sg,DT10,h=0.5); e2=list(tj[-1])
    o2=[o[0]+o[3]*math.cos(o[2])*DT10,o[1]+o[3]*math.sin(o[2])*DT10,o[2],o[3]]
    tl=tail_after(sg,DT10); c1a=cert_v2(e2,o2,l,w,tl,H=120.0); c1b=cert_v2(e2,o2,l,w,tl,H=110.0)
    tsT,tjT,_=R.integ(e2,tl,120.0,h=0.5); tH=tsT[-1]
    oT=[o2[0]+o2[3]*math.cos(o2[2])*tH,o2[1]+o2[3]*math.sin(o2[2])*tH,o2[2],o2[3]]
    g1c_val=rr(list(tjT[-1]),oT)
    d["g1a"]+=int(c1a["certified_perm"]); d["g1b"]+=int(c1b["certified_perm"]); d["g1c"]+=int(g1c_val>0)
    if rho==4:
        ovt_detail.append(dict(idx=i,g1c_rr=g1c_val,g1a=c1a["certified_perm"],ds_end=c1a["ds_end"],
                               olen=l,owid=w,seed=r.get("seed")))
    if cs(g)!=0:
        d["cbt"]+=1; d["cb"]+=int(fb(e,o,l,w,cs(g)) is not None)
    if (i+1)%100==0: print(f"  ...{i+1}/{len(gw)}  ({time.time()-t0:.0f}s)",flush=True)

print("\n  态势        n  A成员%  门1(1a/1b/1c)%  门2合规backup%",flush=True)
for rho in sorted(st):
    d=st[rho]; nA=max(1,d["inA"])
    cb=f"{100*d['cb']/max(1,d['cbt']):.1f} ({d['cb']}/{d['cbt']})" if d['cbt'] else "n/a"
    print(f"  {NM[rho]:9s} {d['n']:4d}  {100*d['inA']/d['n']:5.1f}  {100*d['g1a']/nA:.1f}/{100*d['g1b']/nA:.1f}/{100*d['g1c']/nA:.1f}    {cb}",flush=True)
print(f"\n[timing] {time.time()-t0:.0f}s total",flush=True)
# dump overtake detail
import pickle
pickle.dump(ovt_detail, open("ovt_detail.pkl","wb"))
print("\n=== OVERTAKE 1c investigation (range-rate at tail end) ===")
for od in ovt_detail:
    flag = "  <-- 1c FAIL" if od["g1c_rr"]<=0 else ""
    print(f"  idx={od['idx']} seed={od['seed']} rr={od['g1c_rr']:+.3f} 1a={od['g1a']} ds_end={od['ds_end']:.1f} olen={od['olen']:.0f} owid={od['owid']:.1f}{flag}")
