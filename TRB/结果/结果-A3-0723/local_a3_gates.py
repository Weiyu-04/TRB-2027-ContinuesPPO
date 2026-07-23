#!/usr/bin/env python3
"""本机·对 A3 真态跑门逻辑初判（RK4 重建·和官方 odeint 误差~1e-8·官方 --gates 出权威值）。
门1(前向不变·三机制正交)+门2(让路合规 backup)+A-成员率·按 ρ 分层+per-encounter。修正版理论(引理1恒速尾/F1/F2)。"""
import json, math, sys
import numpy as np
sys.path.insert(0, "/tmp/claude-0/-home-user-TRB-2027-ContinuesPPO/c66f9aab-d514-56eb-b3c3-8a5123b55141/scratchpad")
import reclassify as R
from phase4_gate1_corrected import cert_v2, straight_tail_family, tail_after
DT10 = R.DT10; RHO_NAME = {1:"stand-on",2:"head-on",3:"crossing",4:"overtake",5:"emergency"}

def compliant_sign(gw): return -1 if gw=="right" else (1 if gw=="left" else 0)

def find_backup(ego, obs, olen, owid, sign=0):
    for name, segs in straight_tail_family():
        w0 = segs[0][1]
        if sign < 0 and not (w0 < -1e-9): continue
        if sign > 0 and not (w0 > 1e-9): continue
        if cert_v2(ego, obs, olen, owid, segs)["certified_perm"]:
            return name, segs
    return None, None

def range_rate(ego, obs):
    prel = np.array([ego[0]-obs[0], ego[1]-obs[1]]); n = np.hypot(*prel)
    if n < 1e-9: return 0.0
    vr = ego[3]*np.array([math.cos(ego[2]),math.sin(ego[2])]) - obs[3]*np.array([math.cos(obs[2]),math.sin(obs[2])])
    return float(prel@vr/n)

recs = [json.loads(l) for l in open("/home/user/TRB-2027-ContinuesPPO/TRB/结果/结果-A3-0723/a3_giveway_states.jsonl")]
print(f"A3 真态 n={len(recs)} · 本机 RK4 初判(官方--gates权威)")
by = {}
for r in recs:
    ego,obs,olen,owid,rho,gw = r["ego"],r["obs"],r["obs_len"],r["obs_wid"],r["rho"],r.get("give_way_dir")
    d = by.setdefault(rho, dict(n=0,inA=0,g1a=0,g1b=0,g1c=0,cb=0,cbtot=0))
    d["n"]+=1
    name,segs = find_backup(ego,obs,olen,owid,0)
    if segs is None:
        # 门2 仍记(无任何backup=合规也无)
        if compliant_sign(gw)!=0: d["cbtot"]+=1
        continue
    d["inA"]+=1
    # 门1 三机制正交
    tsm,trajm,_ = R.integ(ego,segs,DT10,h=0.5)
    ego2=list(trajm[-1]); obs2=[obs[0]+obs[3]*math.cos(obs[2])*DT10,obs[1]+obs[3]*math.sin(obs[2])*DT10,obs[2],obs[3]]
    tail=tail_after(segs,DT10)
    c1a=cert_v2(ego2,obs2,olen,owid,tail,H=120.0)
    c1b=cert_v2(ego2,obs2,olen,owid,tail,H=110.0)
    tsT,trajT,_=R.integ(ego2,tail,120.0,h=0.5); tH=tsT[-1]
    obsT=[obs2[0]+obs2[3]*math.cos(obs2[2])*tH,obs2[1]+obs2[3]*math.sin(obs2[2])*tH,obs2[2],obs2[3]]
    d["g1a"]+=int(c1a["certified_perm"]); d["g1b"]+=int(c1b["certified_perm"]); d["g1c"]+=int(range_rate(list(trajT[-1]),obsT)>0)
    # 门2 合规 backup
    sign=compliant_sign(gw)
    if sign!=0:
        d["cbtot"]+=1
        _,cs=find_backup(ego,obs,olen,owid,sign); d["cb"]+=int(cs is not None)

print(f"\n  {'ρ':>10} | {'n':>4} {'A-成员%':>7} | 门1: {'1a%':>6} {'1b%':>6} {'1c%':>6} | 门2合规backup%")
for rho in sorted(by):
    d=by[rho]; nA=max(1,d["inA"])
    cb=f"{100*d['cb']/max(1,d['cbtot']):.1f}({d['cb']}/{d['cbtot']})" if d["cbtot"] else "n/a"
    print(f"  {RHO_NAME.get(rho,rho):>10} | {d['n']:>4} {100*d['inA']/d['n']:>6.1f} | {100*d['g1a']/nA:>5.1f} {100*d['g1b']/nA:>5.1f} {100*d['g1c']/nA:>5.1f} | {cb}")
# 让路态汇总
gw_rec=[r for r in recs if r["rho"] in (2,3,4)]
print(f"\n  让路态(ρ2/3/4)总 n={len(gw_rec)} (head-on {sum(1 for r in gw_rec if r['rho']==2)}/crossing {sum(1 for r in gw_rec if r['rho']==3)}/overtake {sum(1 for r in gw_rec if r['rho']==4)})")
