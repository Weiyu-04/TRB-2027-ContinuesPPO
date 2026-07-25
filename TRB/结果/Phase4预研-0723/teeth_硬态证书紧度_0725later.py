"""硬态 teeth 探针 v2(正式化·拆"证书保守 vs 真避不了")。
造真对撞硬让路态(block3.gen_synthetic_conflicts·纯几何本机)·跑 uterm cert:
  in-A       = cert_v2 认证有【合规向】certified 永久清脱离(sound·保守)。
  not-in-A 再拆:
    cert保守 = cert 拒·但同族某机动【真】细积分 200s min>0(=有真能避·只是 sound 证书没认)。
    族避不了 = cert 拒·且同 57 族无一真避(near-unavoidable·更富族可能仍避·故不写"绝对不可避")。
⚠️ 边界:合成几何(非策略闭环);give-way 向按 kind 几何近似(head_on/crossing→starboard·overtake→任意)·非状态机;
   "族避不了"限于本 57 对齐族(可执行·admissible)·非绝对不可避。=定性证据(约束有牙齿+A内近似紧度)·非精确不可避率。"""
import sys, math
import numpy as np
from collections import defaultdict
sys.path.insert(0, "/home/user/TRB-2027-ContinuesPPO/TRB/代码/trb_env")
sys.path.insert(0, "/home/user/TRB-2027-ContinuesPPO/TRB/代码/m1_dock_wip")
import uterm_terminal as U
import block3_partition_probe as B3
INT = U.integrate_local_rk4


def kind_sign(kind):
    return 0 if kind == "overtake" else -1


def cert_escape(ego, obs, olen, owid, sign):
    _, segs = U.state_in_A(ego, obs, olen, owid, INT, require_omega_sign=sign)
    return segs is not None


def true_min_dist(ego, segs, obs, olen, owid, T=200.0):
    ts, tj, _ = INT(ego, segs, T, h=0.5, dt=0.1)
    vm = obs[3]; om = (math.cos(obs[2]), math.sin(obs[2])); best = 1e18
    for k in range(0, len(ts), 2):
        t = ts[k]; ex, ey, eth, _ = tj[k]
        oc = (obs[0] + vm*om[0]*t, obs[1] + vm*om[1]*t)
        d = U._rect(ex, ey, eth, U.L_SHIP, U.W_SHIP).distance(U._rect(oc[0], oc[1], obs[2], olen, owid))
        if d < best: best = d
        if best <= 0: return 0.0
    return best


def family_true_avoids(ego, obs, olen, owid):
    """本 57 对齐族里·任一机动【真】细积分 200s min>0 ⟹ 该态有真能避的可执行机动。"""
    for _, segs, _ in U.straight_tail_family():
        if true_min_dist(ego, segs, obs, olen, owid) > 0.0:
            return True
    return False


def run(N, seed):
    rng = np.random.default_rng(seed)
    recs = B3.gen_synthetic_conflicts(N, rng)
    st = defaultdict(lambda: dict(n=0, inA=0, comp=0, cons=0, unav=0))
    for i, r in enumerate(recs):
        ego, obs, olen, owid, kind = r["ego"], r["obs"], r["obs_len"], r["obs_wid"], r["kind"]
        sign = kind_sign(kind)
        d = st[kind]; d["n"] += 1
        inA = cert_escape(ego, obs, olen, owid, 0)
        d["inA"] += int(inA)
        d["comp"] += int(cert_escape(ego, obs, olen, owid, sign))
        if not inA:                                   # 拒 → 拆 保守 vs 族避不了
            if family_true_avoids(ego, obs, olen, owid):
                d["cons"] += 1
            else:
                d["unav"] += 1
        if (i+1) % 40 == 0:
            print(f"  ...{i+1}/{len(recs)}", flush=True)
    return recs, st


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    recs, st = run(N, 20260725)
    print(f"\n合成真对撞硬让路态 n={len(recs)}")
    print(f"{'kind':11s} {'n':>4s} {'in-A%':>6s} {'合规in-A%':>9s} {'拒→证书保守%':>12s} {'拒→族避不了%':>12s}")
    tot = dict(n=0, inA=0, comp=0, cons=0, unav=0)
    for kind in sorted(st):
        d = st[kind]; n = d["n"]
        for k in tot: tot[k] += d[k]
        print(f"{kind:11s} {n:4d} {100*d['inA']/n:6.1f} {100*d['comp']/n:9.1f} {100*d['cons']/n:12.1f} {100*d['unav']/n:12.1f}")
    n = tot["n"]
    print(f"{'TOTAL':11s} {n:4d} {100*tot['inA']/n:6.1f} {100*tot['comp']/n:9.1f} {100*tot['cons']/n:12.1f} {100*tot['unav']/n:12.1f}")
    print(f"\n解读: in-A={100*tot['inA']/n:.1f}%(盾放行·有sound合规退路) · 拒绝共{100*(n-tot['inA'])/n:.1f}%")
    print(f"  其中【证书保守】{100*tot['cons']/n:.1f}%(有真能避的可执行机动·sound证书太严没认=A内近似可再放宽) ·")
    print(f"       【族避不了】{100*tot['unav']/n:.1f}%(本可执行族真避不了·近不可避·落兜底=诚实在A外)。")
    print("  ⟹ 约束有牙齿(拒>0·非空转);拒绝里 unav 部分=真该退兜底·cons 部分=A 内近似保守量(未来可细族/松证书回收)。")

# ══════════════ 结果(N=120·合成真对撞硬让路态·2026-07-25 later) ══════════════
#  kind        n   in-A%  合规in-A%  拒→证书保守%  拒→族避不了%
#  cross_port  20   65.0    60.0        0.0          35.0
#  cross_star  30   80.0    76.7        0.0          20.0
#  head_on     33   81.8    81.8        0.0          18.2
#  overtake    37   27.0    27.0        2.7          70.3
#  TOTAL      120   61.7    60.0        0.8          37.5
#
#  结论:① 约束【有牙齿】=硬态拒 38.3%(easy 基准 0%·坐实 no-op 是数据太温和非空转)。
#       ② 证书【紧·非过保守】=拒绝里仅 0.8% 是"证书保守"(有真能避机动却没认)·37.5% 是
#          "本可执行族真避不了"(近不可避·该退兜底) → A 是【紧内近似】·盾几乎只拒真该拒的。
#       ③ overtake 70% 族避不了(追尾几何本就近·或族对追越偏弱)·值得单看。
#  边界: 合成几何(非策略闭环)·give-way 向按 kind 几何近似(非状态机)·"族避不了"限本 57 对齐
#        可执行族(更富族可能再避回一点)=定性证据(有牙齿+A 紧度)·非精确不可避率。
