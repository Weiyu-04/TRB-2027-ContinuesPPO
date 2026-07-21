#!/usr/bin/env python3
"""B 关配对判读（治疗臂 vs 基线·同种子·`03` L150/L153 go/kill 判据）。
用法: python bgate_judge.py <治疗tag> <基线tag> [结果dir]
  例(验证机制): python bgate_judge.py coneP45vf2HOCRppo noconeHOCRppo 结果/结果0705-重新改进-小测-11:36
  B关实用:     python bgate_judge.py augRhoHOCRppo   noconeHOCRppo "结果/结果0706-腿B-1:33"
机制已验：cone-vs-nocone 应复现锥净伤到达 中位80→12.5（`03` L153）。
判据(L150·不挑种子·IQM+失败率)：
  ✅ 早信号=崩种子 EpLen 从~1700 钉死松动(min EpLen 显著低于基线) + 健康种子到达不掉 → 扩5种子/跑满
  ⚠️ EpLen 仍钉~1700(治标不治崩) → 换杠杆治崩；❌ 噪声内 → 弃
"""
import json, sys, glob, statistics as st

TREAT, BASE = sys.argv[1], sys.argv[2]
DIR = sys.argv[3] if len(sys.argv) > 3 else "结果"
SEEDS = list(range(5))
CRASH_ARR = 50.0          # 到达<此=崩(塌种子)
PIN_EPLEN = 1400.0        # EpLen>此≈钉在超时(健康降到~550)

def load(tag, s):
    fs = glob.glob(f"{DIR}/step4e_partial_{tag}_s{s}.jsonl")
    if not fs: return None
    rec = [json.loads(l) for l in open(fs[0]) if l.strip()][0]
    fp = rec.get("final_per") or []
    jerk_all = [e["ctrl_jerk_norm_mean"] for e in fp if e.get("ctrl_jerk_norm_mean") is not None]
    jerk_arr = [e["ctrl_jerk_norm_mean"] for e in fp if e.get("reached") and e.get("ctrl_jerk_norm_mean") is not None]
    return dict(arr=rec["final"]["到达率%"], viol=rec["final"]["违规次数/局"], coll=rec["final"]["碰撞率%"],
                eplen_trend=[t["Ep长s"] for t in rec["trend"]], arr_trend=[t["到达率%"] for t in rec["trend"]],
                jerk=st.mean(jerk_all) if jerk_all else None, jerk_arr=st.mean(jerk_arr) if jerk_arr else None)

def iqm(xs):
    xs = sorted(xs)
    if len(xs) < 4: return st.mean(xs)
    k = len(xs) // 4
    return st.mean(xs[k:len(xs)-k])

B = {s: load(BASE, s) for s in SEEDS}
T = {s: load(TREAT, s) for s in SEEDS}
if any(v is None for v in T.values()):
    print(f"⚠️ 治疗 {TREAT} 缺种子: {[s for s in SEEDS if T[s] is None]}（数据未回传全？）")

print(f"\n===== B 关配对判读: 治疗[{TREAT}] vs 基线[{BASE}] =====")
print(f"{'seed':>4} | {'基线到达':>8} {'治疗到达':>8} {'Δ到达':>7} | {'基线minEp':>9} {'治疗minEp':>9} | {'基线jerk':>8} {'治疗jerk':>8} | 分类")
crash_signal, healthy_hurt = [], []
for s in SEEDS:
    b, t = B[s], T[s]
    if b is None or t is None:
        print(f"{s:>4} | {'—':>8} {'—':>8}"); continue
    b_min, t_min = min(b["eplen_trend"]), min(t["eplen_trend"])
    cls = "崩" if b["arr"] < CRASH_ARR else "健康"; tag = ""
    if cls == "崩":
        loosened = (t_min < PIN_EPLEN) and (b_min - t_min > 200); lifted = t["arr"] - b["arr"] > 10
        tag = "松动✅" if (loosened or lifted) else "仍钉死"; crash_signal.append(loosened or lifted)
    else:
        if b["arr"] - t["arr"] > 15: healthy_hurt.append(s); tag = "🔴到达掉"
    jb = f"{b['jerk']:.2f}" if b['jerk'] else "—"; jt = f"{t['jerk']:.2f}" if t['jerk'] else "—"
    print(f"{s:>4} | {b['arr']:>8.1f} {t['arr']:>8.1f} {t['arr']-b['arr']:>+7.1f} | {b_min:>9.0f} {t_min:>9.0f} | {jb:>8} {jt:>8} | {cls} {tag}")

bl = [B[s]['arr'] for s in SEEDS if B[s]]; tl = [T[s]['arr'] for s in SEEDS if T[s]]
if bl and tl and len(tl) == len(SEEDS):
    print(f"\n聚合: 到达 中位 {st.median(bl):.1f}→{st.median(tl):.1f} | IQM {iqm(bl):.1f}→{iqm(tl):.1f} | 崩率 {sum(a<CRASH_ARR for a in bl)}/5→{sum(a<CRASH_ARR for a in tl)}/5")
    bj = [B[s]['jerk'] for s in SEEDS if B[s] and B[s]['jerk']]; tj = [T[s]['jerk'] for s in SEEDS if T[s] and T[s]['jerk']]
    bv = [B[s]['viol'] for s in SEEDS if B[s]]; tv = [T[s]['viol'] for s in SEEDS if T[s]]
    if bj and tj: print(f"      治抖 jerk 均 {st.mean(bj):.2f}→{st.mean(tj):.2f} | 治违规 {st.mean(bv):.2f}→{st.mean(tv):.2f}/局")
    print("\n----- 建议裁决（L150·仅供参考·主窗口须亲核·绝不挑种子）-----")
    if crash_signal and any(crash_signal):
        print(f"  ✅ 崩种子早信号：{sum(crash_signal)}/{len(crash_signal)} 松动/离地", "+ 健康不掉 → 扩5种子/跑满" if not healthy_hurt else f"· 但健康 {healthy_hurt} 到达掉 → 谨慎")
    elif crash_signal:
        print("  ⚠️ 崩种子 EpLen 仍钉~1700=治标不治崩 → 换杠杆")
    if tj and bj and st.mean(tj) < st.mean(bj): print(f"  ✅ 治抖有效（jerk {st.mean(bj):.2f}→{st.mean(tj):.2f}）")
    print("  🔴 报 IQM+崩率·符号未知须多种子坐实·绝不挑种子")
