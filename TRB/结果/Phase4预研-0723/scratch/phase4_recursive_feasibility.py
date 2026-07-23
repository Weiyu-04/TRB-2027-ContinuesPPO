#!/usr/bin/env python3
"""Phase 4 可行性预研 · 探针1 = 递归可行性（可清障集 A 是否前向不变）。
候选不变集 A = {∃ sound 机动全程清障 = avoidable}。
测试：从每个 avoidable 态，取一个清障机动，走【官方口径】一个决策步(10s)到后继态(障碍同步 CV 前进10s)，
      对后继态【重新全量分类】——它还 avoidable 吗？
若 ~100% 后继仍 avoidable → A 近似前向不变 → 递归可行 → Phase 4 有底(可写终端约束)。
若显著掉出(变 unavoid/undec) → 朴素可清障集不是不变集 → 需更强构造(funnel/收缩视界)。
复用 reclassify.py 的忠实重建动力学+gap#1+清障判据。纯几何·不烧卡·无 vesselmodels。"""
import sys, math, json, time
import numpy as np
sys.path.insert(0, "/tmp/claude-0/-home-user-TRB-2027-ContinuesPPO/c66f9aab-d514-56eb-b3c3-8a5123b55141/scratchpad")
import reclassify as R

DT10 = 10.0
def advance_obs_cv(obs, t):
    ox, oy, oth, ov = obs
    return [ox+ov*math.cos(oth)*t, oy+ov*math.sin(oth)*t, oth, ov]

def first_step_successor(ego, segs, h=0.5):
    """走官方口径一个决策步(10s)：用该机动的控制积分 10s，取末态(含10s边界钳v)。"""
    ts, traj, oseg = R.integ(ego, segs, DT10, h=h)
    return list(traj[-1])   # [px,py,θ,v] after 10s

def main():
    NSAMP = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    Tcls = 120.0
    recs = [json.loads(l) for l in open("/home/user/TRB-2027-ContinuesPPO/TRB/结果/结果-block3-0722/block3_synthetic_states.jsonl")][:NSAMP]
    fam = R.family_window()
    fam_d = dict(fam)
    print(f"Phase4 递归可行性探针 · 合成真对撞 n={len(recs)} · 分类视界 T={Tcls}")

    t0 = time.time()
    # 先分类每个态
    cur = []
    for r in recs:
        un, ct = R.classify(r["ego"], r["obs"], r["obs_len"], r["obs_wid"], fam, T=Tcls)
        cur.append((r, un, ct))

    for Tc in (40.0, 60.0):
        n_avoid = 0
        succ = {"avoid": 0, "unavoid": 0, "undec": 0, "cleared_past": 0}
        examples_bad = []
        for r, un, ct in cur:
            if R.partition(bool(un), ct, Tc) != "avoid":
                continue
            n_avoid += 1
            # 选一个在 Tc 内清障的机动(first_unsafe None 或 > Tc)
            chosen = None
            for name, fut in ct.items():
                if fut is None or fut > Tc:
                    chosen = name; break
            segs = fam_d[chosen]
            ego2 = first_step_successor(r["ego"], segs)
            obs2 = advance_obs_cv(r["obs"], DT10)
            # 后继态：若障碍已过顶(相对 ego 在后方且远离)→ 视为永久安全
            # 判后继是否仍 avoidable：重新全量分类
            un2, ct2 = R.classify(ego2, obs2, r["obs_len"], r["obs_wid"], fam, T=Tcls)
            # keep-course 后继净空(判是否已脱离冲突)
            kc2 = R.rect  # placeholder
            part2 = R.partition(bool(un2), ct2, Tc)
            # 是否"已清过顶"：后继 keep-course 全程>0 且距离在增大
            d_now = math.hypot(r["ego"][0]-r["obs"][0], r["ego"][1]-r["obs"][1])
            d_next = math.hypot(ego2[0]-obs2[0], ego2[1]-obs2[1])
            succ[part2] += 1
            if part2 != "avoid" and len(examples_bad) < 5:
                examples_bad.append((part2, chosen, round(d_now), round(d_next)))
        inv = 100*succ["avoid"]/max(1, n_avoid)
        print(f"\n  [Tc={Tc:.0f}s] avoidable 态 n={n_avoid} → 后继再分类:")
        print(f"     仍 avoidable(不变式保持): {succ['avoid']} ({inv:.1f}%)")
        print(f"     变 undecided:            {succ['undec']}")
        print(f"     变 unavoidable(最坏·真掉出): {succ['unavoid']}")
        if examples_bad:
            print(f"     掉出例(part,机动,d_now→d_next): {examples_bad}")
    print(f"\n  用时 {time.time()-t0:.0f}s")
    print("  判读：仍avoidable≈100% → 可清障集≈前向不变 → 递归可行 → Phase4 终端约束有底；")
    print("        若大量变 unavoid → 走一步就进死局 → 朴素可清障集非不变集 → 需收缩视界/funnel。")

if __name__ == "__main__":
    main()
