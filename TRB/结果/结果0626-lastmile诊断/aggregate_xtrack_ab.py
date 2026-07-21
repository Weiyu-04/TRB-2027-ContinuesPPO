#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""正式 well_X A/B 5 种子聚合判读（`03` L90·机制小验命中后·定 Φ_xtrack 多种子稳不稳）。
glob 全 diagXT_wb200_wx{0,200}_s*.jsonl（复用 s1/s2 机制小验 + 新 s0/s3/s4）→ 配对 A/B：
  · 头条=到达率：逐种子 + Mean/IQM + 配对差 + 自助CI(mean&IQM) + sign/Wilcoxon/配对t（多检验并列·不挑度量·防 L89 退化伪显著）
  · 机制=带内刹停率↓  · 安全锚点=碰撞/违规不漂(必查)  · 失败率(塌种子数)
诚实地板（继承）：报全 5 种子·绝不挑种子；Φ_xtrack 有效性未证前不 over-claim。
用法：python aggregate_xtrack_ab.py [结果根目录,默认 结果]
"""
import sys, os, glob, math, json
import numpy as np
import statistics as st
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
from analyze_xtrack import faithful, ecross, R_LAT
try:
    from scipy import stats as _sps
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def metrics(path):
    o = json.loads(open(path).read().strip().split("\n")[-1])
    fp = o["final_per"]; n = len(fp)
    reached = sum(1 for e in fp if e["reached"])
    posmiss = inband = timeout = coll = 0
    viol = [e.get("violations", 0) for e in fp]
    ec_reach = []
    for e in fp:
        if e["collided"]: coll += 1
        if e["reached"]: continue
        tf = e.get("term_flags") or {}; es = e.get("end_state"); gg = e.get("goal_geom")
        if tf.get("time"): timeout += 1
        if es and gg:
            ip, io, it = faithful(gg, es["px"], es["py"], es["psi"], es["time_step"])
            if not ip:
                posmiss += 1
                if tf.get("stopped"): inband += 1
                ec = ecross(gg, es["px"], es["py"])
                if ec <= R_LAT: ec_reach.append(ec)
    return {"n": n, "arrival": 100.0*reached/n, "posmiss": 100.0*posmiss/n,
            "inband": 100.0*inband/n, "timeout": timeout, "coll": 100.0*coll/n,
            "viol": st.mean(viol), "ec_reach_med": (st.median(ec_reach) if ec_reach else None),
            "well_B": o.get("well_shaping_weight"), "well_X": o.get("xtrack_weight"),
            "xt_field": o.get("xtrack_weight"), "steps": o.get("steps")}


def iqm(x):  # 20% 截尾均值
    xs = np.sort(np.asarray(x, float)); k = len(xs); lo = int(math.floor(0.2*k))
    mid = xs[lo:k-lo] if k-2*lo > 0 else xs
    return float(mid.mean())


def boot_ci(diffs, fn, B=10000):
    rng = np.random.default_rng(0); d = np.asarray(diffs, float); m = len(d)
    s = np.array([fn(d[rng.integers(0, m, m)]) for _ in range(B)])
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "结果"
    runs = {}
    for f in glob.glob(os.path.join(root, "**", "step4e_partial_diagXT_wb200_wx*_s*.jsonl"), recursive=True):
        b = os.path.basename(f)
        import re
        mt = re.search(r"_wx(\d+)_s(\d+)\.jsonl$", b)
        if not mt: continue
        wx, s = int(mt.group(1)), int(mt.group(2))
        runs[(s, wx)] = metrics(f)
    seeds = sorted({s for (s, _) in runs if (s, 0) in runs and (s, 200) in runs})
    miss = sorted({s for (s, _) in runs} - set(seeds))
    if not seeds:
        print("未找到成对(wx0&wx200)的 wb200 种子"); return
    print(f"配对种子: {seeds}" + (f"  | 不成对/缺对照: {miss}" if miss else ""))
    if len(seeds) < 5:
        print(f"⚠️ 仅 {len(seeds)} 颗成对（正式 A/B 需 5 颗 s0-4）→ 下方为【部分】·待 s0/3/4 回传补全\n")

    print(f"{'seed':>4} | {'到达 wx0→wx200(Δ)':>22} | {'带内刹停 0→200(Δ)':>22} | {'碰撞':>10} | {'违规/局 0→200':>14}")
    a0 = []; a2 = []; ib0 = []; ib2 = []; c0 = []; c2 = []; v0 = []; v2 = []
    for s in seeds:
        m0, m2 = runs[(s, 0)], runs[(s, 200)]
        a0.append(m0["arrival"]); a2.append(m2["arrival"])
        ib0.append(m0["inband"]); ib2.append(m2["inband"])
        c0.append(m0["coll"]); c2.append(m2["coll"]); v0.append(m0["viol"]); v2.append(m2["viol"])
        print(f"{s:>4} | {m0['arrival']:6.1f}→{m2['arrival']:6.1f}({m2['arrival']-m0['arrival']:+5.1f}) | "
              f"{m0['inband']:6.1f}→{m2['inband']:6.1f}({m2['inband']-m0['inband']:+5.1f}) | "
              f"{m0['coll']:.1f}→{m2['coll']:.1f}% | {m0['viol']:.2f}→{m2['viol']:.2f}")
        if m2["xt_field"] is None:
            print(f"      ⚠️ s{s} wx200 jsonl xtrack_weight=None（旧码未传/缺陷未修就跑·靠文件名认 well_X·数据仍可用）")

    diffs = [a2[i]-a0[i] for i in range(len(seeds))]
    print(f"\n=== 头条·到达率（绝不挑种子·报全 {len(seeds)} 颗）===")
    print(f"  wx0   : {[round(x,1) for x in a0]}  Mean {np.mean(a0):.1f}  IQM {iqm(a0):.1f}  std {np.std(a0,ddof=1):.1f}")
    print(f"  wx200 : {[round(x,1) for x in a2]}  Mean {np.mean(a2):.1f}  IQM {iqm(a2):.1f}  std {np.std(a2,ddof=1):.1f}")
    print(f"  配对差 : {[round(x,1) for x in diffs]}  Mean diff {np.mean(diffs):+.1f}  IQM diff {iqm(a2)-iqm(a0):+.1f}")
    if len(seeds) >= 2:
        mlo, mhi = boot_ci(diffs, np.mean)
        print(f"  自助CI(mean diff): [{mlo:+.1f}, {mhi:+.1f}]  {'跨零=不显著' if mlo<=0<=mhi else '不跨零'}")
        npos = sum(1 for d in diffs if d > 0)
        print(f"  sign: {npos}/{len(seeds)} 升" + (f"  binom p={_sps.binomtest(npos,len(seeds)).pvalue:.3f}" if _HAS_SCIPY else ""))
        if _HAS_SCIPY and len(seeds) >= 5:
            try: print(f"  Wilcoxon p={_sps.wilcoxon(a2,a0).pvalue:.3f}  配对t p={_sps.ttest_rel(a2,a0).pvalue:.3f}")
            except Exception as e: print(f"  (Wilcoxon/t skip: {e})")
        if all(d > 0 for d in diffs):
            print("  ⚠️ 全差为正→均值/IQM 的百分位 bootstrap 数学上恒不跨零=退化·别当显著性证据(L89)·看 sign/Wilcoxon")
    fail = sum(1 for x in a2 if x < 20)
    print(f"\n=== 机制·安全·失败率 ===")
    print(f"  带内刹停率 Mean: wx0 {np.mean(ib0):.1f}% → wx200 {np.mean(ib2):.1f}%  (Δ {np.mean(ib2)-np.mean(ib0):+.1f}pt·机制命中应↓)")
    print(f"  碰撞 Mean: wx0 {np.mean(c0):.2f}% → wx200 {np.mean(c2):.2f}%  | 违规/局 Mean: {np.mean(v0):.2f} → {np.mean(v2):.2f}  (安全锚点·应不漂↑)")
    print(f"  塌缩(wx200 到达<20%): {fail}/{len(seeds)} 颗")
    print(f"\n判据(04): well_X=200 IQM 显著>0 + 带内刹停↓ + 碰撞/违规不漂 = 多种子确认→升 2000 钱图；")
    print(f"          不显著/锚点漂→退诚实地板(关 well_X 发修法A·盾可证明方向合规+IQM/CI 不挑种子)。")
    print(f"诚实：n={len(seeds)}·sign/Wilcoxon/配对t/自助CI 多检验并列·绝不挑度量/种子。")


if __name__ == "__main__":
    main()
