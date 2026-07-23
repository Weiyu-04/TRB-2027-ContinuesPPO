#!/usr/bin/env python3
"""独立复算 keep-course 冲突分桶（复审声称③ + 真冲突罕见）。
我自己从零实现直线外推 + 旋转矩形最近距离，再与探针原函数交叉验证。
不引入 vesselmodels。ego=SR108 175x25.4；obs 用数据里的真长真宽。"""
import json, math, sys, os
import numpy as np
from shapely.geometry import Polygon

L_SHIP, W_SHIP = 175.0, 25.4

def my_rect(cx, cy, theta, length, width):
    """我自己的旋转矩形（独立实现，corner 顺序与探针无关）。"""
    hl, hw = 0.5*length, 0.5*width
    c, s = math.cos(theta), math.sin(theta)
    # 顺序：前右, 后右, 后左, 前左（任意闭合顺序都给同一多边形）
    local = [(hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw)]
    pts = [(cx + x*c - y*s, cy + x*s + y*c) for x, y in local]
    return Polygon(pts)

def my_keep_course_min_dist(ego, obs, olen, owid, T=120.0, dt=0.5):
    """我独立实现：ego 恒速恒向、obs 恒速 CV，全程 [0,T] 步长 dt 的船体最小距离。"""
    ex, ey, eth, ev = ego
    ox, oy, oth, ov = obs
    ce, se = math.cos(eth), math.sin(eth)
    co, so = math.cos(oth), math.sin(oth)
    best = float('inf')
    n = int(round(T/dt))
    for k in range(n+1):
        t = k*dt
        egc_x, egc_y = ex + ev*ce*t, ey + ev*se*t
        obc_x, obc_y = ox + ov*co*t, oy + ov*so*t
        d = my_rect(egc_x, egc_y, eth, L_SHIP, W_SHIP).distance(my_rect(obc_x, obc_y, oth, olen, owid))
        if d < best:
            best = d
        if best <= 0.0:
            return 0.0
    return float(best)

# 探针原函数（交叉验证用）
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "..", "..", "TRB", "代码", "m1_dock_wip"))
sys.path.insert(0, "/home/user/TRB-2027-ContinuesPPO/TRB/代码/m1_dock_wip")
import block3_partition_probe as P   # _HAVE_OFFICIAL 会是 False，keep_course_min_dist 仍可用（纯几何 fallback）
print(f"[env] probe _HAVE_OFFICIAL={P._HAVE_OFFICIAL} (期望 False=本机无 vesselmodels)")

def analyze(path, label):
    recs = [json.loads(l) for l in open(path)]
    genuine=nearmiss=farsafe=0
    dists=[]; kcs=[]
    mismatch=0; max_abs_diff=0.0
    for r in recs:
        ego, obs = r["ego"], r["obs"]
        olen, owid = r["obs_len"], r["obs_wid"]
        kc_mine = my_keep_course_min_dist(ego, obs, olen, owid)
        kc_probe = P.keep_course_min_dist(ego, obs, olen, owid)   # 交叉验证
        diff = abs(kc_mine - kc_probe)
        if diff > max_abs_diff: max_abs_diff = diff
        # 两者可能一个提前 return 0.0、一个给正数 → 只在都>0 时比数值
        if not ((kc_mine<=0 and kc_probe<=0) or diff < 1e-6):
            mismatch += 1
        kc = kc_mine
        kcs.append(kc)
        dists.append(math.hypot(ego[0]-obs[0], ego[1]-obs[1]))
        if kc <= 0.0: genuine += 1
        elif kc <= 50.0: nearmiss += 1
        else: farsafe += 1
    N = len(recs)
    print(f"\n===== {label}  (n={N}, 文件={os.path.basename(path)}) =====")
    print(f"  真对撞 kc<=0    : {genuine:5d}  ({100*genuine/N:5.2f}%)")
    print(f"  擦边   0<kc<=50 : {nearmiss:5d}  ({100*nearmiss/N:5.2f}%)")
    print(f"  假紧急 kc>50    : {farsafe:5d}  ({100*farsafe/N:5.2f}%)")
    print(f"  ego-obs 中心距 : 中位 {np.median(dists):.1f}m  min {np.min(dists):.1f}  max {np.max(dists):.1f}")
    # 最近的几个（kc 最小）态供核对 L196 的“最近10态净空57-210m”
    order = np.argsort(kcs)[:10]
    near_kc = [round(kcs[i],1) for i in order]
    print(f"  kc 最小的10态净空: {near_kc}")
    print(f"  [交叉验证] 与探针原函数: mismatch={mismatch}, max|Δ|={max_abs_diff:.2e}m")
    return dict(N=N, genuine=genuine, nearmiss=nearmiss, farsafe=farsafe,
                med_dist=float(np.median(dists)))

base = "/home/user/TRB-2027-ContinuesPPO/TRB/结果/结果-block3-0722"
r_adv = analyze(f"{base}/block3_adv_states.jsonl", "对抗基线 ρ5 (adv)")
r_gold = analyze(f"{base}/block3_rho5_states.jsonl", "金标策略 ρ5 (golden)")
r_syn = analyze(f"{base}/block3_synthetic_states.jsonl", "合成真对撞 (synthetic)")

print("\n" + "="*70)
print("对账窗口声称：")
print(f"  L197 声称 adv: 真对撞15(1.1%) / 假紧急1330(97.6%) / 1363总")
print(f"    我复算 adv: 真对撞{r_adv['genuine']} / 假紧急{r_adv['farsafe']} / {r_adv['N']}总")
print(f"  L196 声称 golden: 真对撞0 / 中位距~2605m / 423总")
print(f"    我复算 golden: 真对撞{r_gold['genuine']} / 中位距{r_gold['med_dist']:.0f}m / {r_gold['N']}总")
print(f"  合成声称: 全部真对撞(构造保证)")
print(f"    我复算 syn: 真对撞{r_syn['genuine']}/{r_syn['N']}")
