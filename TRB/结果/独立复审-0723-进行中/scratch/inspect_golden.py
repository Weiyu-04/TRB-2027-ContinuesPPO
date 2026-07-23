#!/usr/bin/env python3
"""挖出金标 ρ5 里 kc<=50 的态，核实是否真对撞（非退化：t=0 是否已重叠/障碍是否在后方远离）。"""
import json, math
import numpy as np
from shapely.geometry import Polygon
L_SHIP, W_SHIP = 175.0, 25.4
def rect(cx,cy,th,l,w):
    hl,hw=0.5*l,0.5*w; c,s=math.cos(th),math.sin(th)
    loc=[(hl,hw),(hl,-hw),(-hl,-hw),(-hl,hw)]
    return Polygon([(cx+x*c-y*s, cy+x*s+y*c) for x,y in loc])
def kc_profile(ego,obs,olen,owid,T=120,dt=0.5):
    ex,ey,eth,ev=ego; ox,oy,oth,ov=obs
    ce,se=math.cos(eth),math.sin(eth); co,so=math.cos(oth),math.sin(oth)
    ds=[]
    for k in range(int(T/dt)+1):
        t=k*dt
        d=rect(ex+ev*ce*t,ey+ev*se*t,eth,L_SHIP,W_SHIP).distance(rect(ox+ov*co*t,oy+ov*so*t,oth,olen,owid))
        ds.append((t,d))
    return ds
recs=[json.loads(l) for l in open("/home/user/TRB-2027-ContinuesPPO/TRB/结果/结果-block3-0722/block3_rho5_states.jsonl")]
rows=[]
for r in recs:
    ds=kc_profile(r["ego"],r["obs"],r["obs_len"],r["obs_wid"])
    kc=min(d for _,d in ds); tmin=min(ds,key=lambda x:x[1])[0]
    d0=ds[0][1]
    rows.append((kc,tmin,d0,r))
rows.sort(key=lambda x:x[0])
print("金标 ρ5 里 keep-course 净空最小的 8 个态：")
for kc,tmin,d0,r in rows[:8]:
    ego,obs=r["ego"],r["obs"]
    cdist=math.hypot(ego[0]-obs[0],ego[1]-obs[1])
    # 障碍相对 ego 的方位（前方/后方）
    brg=math.atan2(obs[1]-ego[1],obs[0]-ego[0])-ego[2]
    brg=(brg+math.pi)%(2*math.pi)-math.pi
    print(f"  kc={kc:6.1f}m @t={tmin:5.1f}s | t=0船体距={d0:6.1f}m 中心距={cdist:6.0f}m "
          f"相对方位={math.degrees(brg):+.0f}° | ego v={ego[3]:.2f} obs v={obs[3]:.2f} "
          f"obs(l={r['obs_len']:.0f},w={r['obs_wid']:.0f}) seed={r['seed']} scn_idx={r['scn_idx']} step={r['step']}")
