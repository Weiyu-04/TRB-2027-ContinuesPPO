#!/usr/bin/env python3
"""Phase4 · 修正版理论落地验证（本机·合成态）：
1. clearance_cert_v2 = block1 清障证书 + 【闭合率交付】(引理1前提)：
   d>0于[0,H](Lipschitz) ∧ 尾段直行(ω=0) ∧ H处船体距在增(过CPA·直行尾g凸→增at H⟹永久增) ⟹ certified 永久清障。
2. certify_sequence: 对具体控制序列跑 v2 证书(不问族成员=修Prop4尾闭合洞)。
3. backup-maneuver U_term 判定 + 【非循环门1重测】(修75/75循环)：
   对 s∈A(v2·直行尾脱离序列)·取m*首步u0走一受盾步得s'·三查隔离:
     (1a) 同一尾巴 m*_tail 用 v2 证书 certify s' ?
     (1b) 收缩视界 H-Δ ?
     (1c) 引理1闭合(v2证书直接输出的闭合率判据) ?
   报逐机制通过率。GO≥99%。"""
import sys, math, json
import numpy as np
sys.path.insert(0, "/tmp/claude-0/-home-user-TRB-2027-ContinuesPPO/c66f9aab-d514-56eb-b3c3-8a5123b55141/scratchpad")
import reclassify as R

A_MAX, W_MAX, V_MAX, DT10, R_CIRC = R.A_MAX, R.W_MAX, R.V_MAX, R.DT10, R.R_CIRC

def _dist_profile(ts, traj, obs, olen, owid):
    vm = obs[3]; om = (math.cos(obs[2]), math.sin(obs[2]))
    ds = np.empty(len(ts))
    for k in range(len(ts)):
        t = ts[k]; ex, ey, eth, ev = traj[k]
        oc = (obs[0]+vm*om[0]*t, obs[1]+vm*om[1]*t)
        ds[k] = R.rect(ex, ey, eth, R.L_SHIP, R.W_SHIP).distance(R.rect(oc[0], oc[1], obs[2], olen, owid))
    return ds

def cert_v2(ego, obs, olen, owid, segs, H=120.0, h=0.5):
    ts, traj, oseg = R.integ(ego, segs, H, h=h)
    fut = R.first_unsafe_t(ts, traj, obs, olen, owid, h, oseg)   # Lipschitz 下界首次不安全
    clears_H = (fut is None) or (fut > H)
    # 尾段直行？segs 最后一段 ω=0
    straight_tail = abs(segs[-1][1]) < 1e-9
    # 过CPA？直行尾上 g 在 H 处增（用采样距离末两点·直行尾 g 凸→增at H⟹永久增）
    ds = _dist_profile(ts, traj, obs, olen, owid)
    # 只在直行尾区间判增：找尾段起始 t
    t_tail = sum(d for a,w,d in segs[:-1] if d is not None)
    tail_idx = np.where(ts >= t_tail - 1e-9)[0]
    if len(tail_idx) >= 2:
        past_cpa = ds[tail_idx[-1]] > ds[tail_idx[-1]-1] + 1e-9   # H处严格在增
        # 且直行尾内最小点非末点(CPA已在尾内出现)——更严：尾内argmin不在末端
        tail_ds = ds[tail_idx]
        past_cpa = past_cpa and (int(np.argmin(tail_ds)) < len(tail_ds)-1)
    else:
        past_cpa = False
    certified_perm = bool(clears_H and straight_tail and past_cpa)
    return dict(certified_perm=certified_perm, clears_H=bool(clears_H),
                straight_tail=bool(straight_tail), past_cpa=bool(past_cpa),
                first_unsafe_t=fut, ds_end=float(ds[-1]), ts=ts, traj=traj, oseg=oseg)

def straight_tail_family():
    """只保留【带直行尾】的脱离序列(turn t1 then straight·accel_turn then straight)+纯加减速直行。"""
    A, W = A_MAX, W_MAX; fam = []
    for w in (-W, W):
        for t1 in (10.,20.,30.,40.,60.,80.):
            fam.append((f"turn{w:+.3f}_{int(t1)}s", [(0.0, w, t1), (0.0, 0.0, None)]))
            fam.append((f"acc{w:+.3f}_{int(t1)}s", [(A, w, t1), (A, 0.0, None)]))
            fam.append((f"dec{w:+.3f}_{t1:.0f}s", [(-A, w, t1), (0.0, 0.0, None)]))
    for a in (-A, 0.0, A):   # 纯直行(不转)
        fam.append((f"straight_a{a:+.2f}", [(a, 0.0, None)]))
    return fam

def tail_after(segs, dt):
    """机动 segs 走 dt 秒后的【尾巴序列】(把首段裁掉 dt)。"""
    out = []; used = 0.0; started = False
    for a, w, dur in segs:
        if dur is None:
            out.append((a, w, None)); started = True; continue
        if used + dur <= dt + 1e-9:
            used += dur; continue
        # 这段跨过 dt
        rem = dur - (dt - used) if used < dt else dur
        if not started and used < dt:
            out.append((a, w, dur - (dt - used))); started = True; used = dt
        else:
            out.append((a, w, dur))
    if not out:
        out = [(segs[-1][0], segs[-1][1], None)]
    return out

def main():
    NSAMP = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    recs = [json.loads(l) for l in open("/home/user/TRB-2027-ContinuesPPO/TRB/结果/结果-block3-0722/block3_synthetic_states.jsonl")][:NSAMP]
    fam = straight_tail_family()
    print(f"Phase4 修正版门1非循环重测 · 合成真对撞 n={len(recs)} · 直行尾脱离族 |{len(fam)}|")
    # 剔 t=0 已撞退化态(复审发现)
    clean = []
    for r in recs:
        d0 = R.rect(r['ego'][0],r['ego'][1],r['ego'][2],R.L_SHIP,R.W_SHIP).distance(
             R.rect(r['obs'][0],r['obs'][1],r['obs'][2],r['obs_len'],r['obs_wid']))
        if d0 > 0: clean.append(r)
    print(f"  剔 t=0 退化态后 clean n={len(clean)}")

    nA = 0; g1a=g1b=g1c=0; no_backup=0
    for r in clean:
        ego, obs, olen, owid = r['ego'], r['obs'], r['obs_len'], r['obs_wid']
        # 找一个 certified 永久清障的直行尾脱离序列 m*
        mstar = None
        for name, segs in fam:
            c = cert_v2(ego, obs, olen, owid, segs)
            if c['certified_perm']:
                mstar = (name, segs); break
        if mstar is None:
            no_backup += 1; continue
        nA += 1
        name, segs = mstar
        # 走一受盾步 u0 = m* 前 Δ=10s
        ts, traj, oseg = R.integ(ego, segs, DT10, h=0.5)
        ego2 = list(traj[-1]); obs2 = [obs[0]+obs[3]*math.cos(obs[2])*DT10, obs[1]+obs[3]*math.sin(obs[2])*DT10, obs[2], obs[3]]
        tail = tail_after(segs, DT10)
        # (1a) 同一尾巴 certify s' (全视界 H)
        c1a = cert_v2(ego2, obs2, olen, owid, tail, H=120.0)
        # (1b) 收缩视界 H-Δ=110
        c1b = cert_v2(ego2, obs2, olen, owid, tail, H=110.0)
        # (1c) 引理1闭合(v2 已交付 past_cpa+straight_tail)
        if c1a['certified_perm']: g1a += 1
        if c1b['clears_H']: g1b += 1
        if c1a['straight_tail'] and c1a['past_cpa']: g1c += 1
    print(f"\n  可清障(∃certified直行尾脱离m*) A-成员 = {nA}/{len(clean)} ({100*nA/max(1,len(clean)):.1f}%) · 无backup={no_backup}")
    if nA:
        print(f"  门1 逐机制(非循环·s∈A→后继s'):")
        print(f"    (1a) 同一尾巴 v2-certify s' 永久清 : {g1a}/{nA} ({100*g1a/nA:.1f}%)")
        print(f"    (1b) 收缩视界 H-Δ clears           : {g1b}/{nA} ({100*g1b/nA:.1f}%)")
        print(f"    (1c) 引理1闭合(直行尾+过CPA)交付   : {g1c}/{nA} ({100*g1c/nA:.1f}%)")
        print(f"  判读：三机制≥99% → 修正版命题4机制(非循环)在合成态坐实·A真前向不变；<99%查因。")
        print(f"        (注:这次是【隔离机制】测·非全量重分类·修了75/75的循环)")

if __name__ == "__main__":
    main()
