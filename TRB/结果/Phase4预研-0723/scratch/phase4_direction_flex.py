#!/usr/bin/env python3
"""Phase4 可行性预研 · 探针2 = 方向弹性（COLREGs∩安全 张力的前哨）。
对每个可清障态，看清障机动的转向方向分布：右转(ω<0)/左转(ω>0)/直行(ω=0) 各有没有清障选项。
- 两方向都有清障选项的态多 → 不管 COLREGs 要哪边、合规清障机动都在 → 张力小、Phase4 合规不变集可行。
- 只单边能清的态多 → 若 COLREGs 要另一边 → 合规∩安全交空 → 张力真实(=§12.5)。
另按 synthetic kind 分组(head_on/cross_star 要右转·cross_port 通常 stand-on·overtake 较松)看是否结构性偏一边。"""
import sys, math, json
sys.path.insert(0, "/tmp/claude-0/-home-user-TRB-2027-ContinuesPPO/c66f9aab-d514-56eb-b3c3-8a5123b55141/scratchpad")
import reclassify as R

def first_omega(segs):
    return segs[0][1]   # 第一段的 ω = 初始转向方向

NSAMP = int(sys.argv[1]) if len(sys.argv) > 1 else 120
Tc = 40.0
recs = [json.loads(l) for l in open("/home/user/TRB-2027-ContinuesPPO/TRB/结果/结果-block3-0722/block3_synthetic_states.jsonl")][:NSAMP]
fam = R.family_window(); fam_d = dict(fam)
om_of = {name: first_omega(segs) for name, segs in fam}

print(f"Phase4 方向弹性探针 · 合成真对撞 n={len(recs)} · Tc={Tc}")
both = right_only = left_only = straight_only = neither = 0
bykind = {}
for r in recs:
    un, ct = R.classify(r["ego"], r["obs"], r["obs_len"], r["obs_wid"], fam, T=120.0)
    if un or R.partition(bool(un), ct, Tc) != "avoid":
        continue
    has_r = has_l = has_s = False
    for name, fut in ct.items():
        if fut is None or fut > Tc:   # 该机动清障
            w = om_of[name]
            if w < -1e-9: has_r = True
            elif w > 1e-9: has_l = True
            else: has_s = True
    k = r.get("kind", "?")
    d = bykind.setdefault(k, {"n":0,"both":0,"r":0,"l":0,"s":0})
    d["n"] += 1
    if has_r and has_l: both += 1; d["both"] += 1
    elif has_r and not has_l: right_only += 1; d["r"] += 1
    elif has_l and not has_r: left_only += 1; d["l"] += 1
    elif has_s: straight_only += 1; d["s"] += 1
    else: neither += 1

navoid = both+right_only+left_only+straight_only+neither
print(f"\n可清障态 n={navoid}:")
print(f"  左右都能清(方向弹性充分)     : {both} ({100*both/max(1,navoid):.1f}%)")
print(f"  只右转能清(ω<0)             : {right_only} ({100*right_only/max(1,navoid):.1f}%)")
print(f"  只左转能清(ω>0)             : {left_only} ({100*left_only/max(1,navoid):.1f}%)")
print(f"  只直行/加减速能清(ω=0)       : {straight_only}")
print(f"\n按 kind 分组(n·both·右only·左only·直only):")
for k,d in sorted(bykind.items()):
    print(f"  {k:12s}: n={d['n']:3d}  both={d['both']:3d}  右={d['r']:3d}  左={d['l']:3d}  直={d['s']:3d}")
print("\n判读：both% 高 → 合规清障机动大概率存在(不管COLREGs要哪边) → Phase4合规不变集可行；")
print("      若某 kind 结构性'只能单边清'且与该 kind 的COLREGs要求方向相反 → 该 kind 上合规∩安全交空=真张力。")
