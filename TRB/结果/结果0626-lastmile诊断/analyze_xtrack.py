#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Φ_xtrack 机制小验判读脚本（`03` L89·二审搭·等数据回传即用）。
读 diagXT 训练 run 的 jsonl(final_per 带 term_flags/end_state/goal_geom·当前码已仪表化)，
用【已二审坐实的 faithful 法】(Polygon+AngleInterval·== 官方 is_reached·mism=0) 把失败分解，
对 well_X=200 vs 0 比【真打靶量·非到达率】：位置门 miss% / 带内刹停率 / |e_cross| 分簇(可达≤R_lat / 远偏)。

⚠️ well_X 靠文件名 TAG 认（连续臂 jsonl xtrack_weight 漏记缺陷·L89⑧·未修前必如此）。
用法：python analyze_xtrack.py <目录>        # 自动扫 *_diagXT_wb{WB}_wx{WX}_s{S}.jsonl
      python analyze_xtrack.py <单文件> WB WX S  # 显式（fixture/非标准名）
"""
import sys, os, json, math, re, glob
import statistics as st
from shapely.geometry import Point, Polygon

R_LAT = 80.0   # Φ_xtrack 横向势半径（命令默认·决定"可达簇"边界）

def faithful(gg, px, py, psi, ts):
    """== 官方 GoalRegion.is_reached（二审 L89 用官方 oracle+6万 fuzz 验等价·mism=0）。返回 (in_pos,in_ori,in_t)。"""
    if not gg or not gg.get("vertices"):
        return None
    d = math.atan2(math.sin(psi - gg["orient_lo"]), math.cos(psi - gg["orient_lo"]))
    w = math.atan2(math.sin(gg["orient_hi"] - gg["orient_lo"]), math.cos(gg["orient_hi"] - gg["orient_lo"]))
    in_pos = Polygon(gg["vertices"]).intersects(Point(px, py))
    in_ori = (-1e-9 <= d <= w + 1e-9)
    in_t = gg["time_lo"] <= ts <= gg["time_hi"]
    return in_pos, in_ori, in_t

def ecross(gg, px, py):
    """到目标带中心线的横向(cross-track)垂距 = (end-center)·n_perp·n_perp⊥长轴(rect_orientation)。θ=0 时 = |py-cy|。"""
    cx, cy = gg["center"]
    th = gg.get("rect_orientation", 0.0)
    return abs((px - cx) * (-math.sin(th)) + (py - cy) * math.cos(th))

def analyze_one(path):
    o = json.loads(open(path).read().strip().split("\n")[-1])
    fp = o["final_per"]; n = len(fp)
    reached = sum(1 for e in fp if e["reached"])
    pos_miss = ori_miss = time_miss = inband_stop = stopped = timeout = collided = 0
    ec_reach = []; ec_far = []     # 位置门外失败的 |e_cross|·按 R_lat 分簇
    skipped = 0
    for e in fp:
        if e["reached"]:
            continue
        tf = e.get("term_flags") or {}
        es = e.get("end_state"); gg = e.get("goal_geom")
        if tf.get("stopped"): stopped += 1
        if tf.get("time"): timeout += 1
        if tf.get("collision"): collided += 1
        if not (es and gg):
            skipped += 1; continue
        fr = faithful(gg, es["px"], es["py"], es["psi"], es["time_step"])
        if fr is None:
            skipped += 1; continue
        ip, io, it = fr
        if not ip:
            pos_miss += 1
            ec = ecross(gg, es["px"], es["py"])
            (ec_reach if ec <= R_LAT else ec_far).append(ec)
            if tf.get("stopped"): inband_stop += 1   # 带内刹停=刹停且位置门外(last-mile 模式)
        if not io: ori_miss += 1
        if not it: time_miss += 1
    # reward-clip 代理（jsonl 不可精测·只给 ep_rew 量级供人工判 VecNorm ±10 是否吃满）
    cv = o.get("curves") or []
    eprew = [c.get("ep_rew_mean") for c in cv if c.get("ep_rew_mean") is not None]
    return {
        "n": n, "reached": reached, "arrival_pct": 100.0 * reached / n,
        "pos_miss": pos_miss, "pos_miss_pct": 100.0 * pos_miss / n,
        "inband_stop": inband_stop, "inband_stop_pct": 100.0 * inband_stop / n,
        "ori_miss": ori_miss, "time_miss": time_miss, "timeout": timeout, "collided": collided,
        "ec_reach": ec_reach, "ec_far": ec_far,
        "ec_reach_med": (st.median(ec_reach) if ec_reach else None),
        "ec_all_med": (st.median(ec_reach + ec_far) if (ec_reach + ec_far) else None),
        "eprew_last": (eprew[-1] if eprew else None),
        "xtrack_weight_field": o.get("xtrack_weight"),  # 缺陷验：连续臂应为 None
        "skipped": skipped,
    }

def fmt(m):
    er = f"{m['ec_reach_med']:.1f}" if m['ec_reach_med'] is not None else "—"
    return (f"到达{m['arrival_pct']:5.1f}% | 位置门miss {m['pos_miss']:2d}={m['pos_miss_pct']:5.1f}% | "
            f"带内刹停 {m['inband_stop']:2d}={m['inband_stop_pct']:5.1f}% | "
            f"|e_cross|可达簇中位 {er}m(n={len(m['ec_reach'])}) 远偏n={len(m['ec_far'])} | "
            f"超时{m['timeout']} 朝向miss{m['ori_miss']} | epRew末{m['eprew_last']}")

def main():
    args = sys.argv[1:]
    runs = {}   # (wb,wx,s) -> metrics
    if len(args) == 4 and os.path.isfile(args[0]):
        p, wb, wx, s = args[0], int(args[1]), int(args[2]), int(args[3])
        runs[(wb, wx, s)] = analyze_one(p)
        print(f"[fixture] {os.path.basename(p)} (wb={wb} wx={wx} s={s}): {fmt(runs[(wb,wx,s)])}")
        m = runs[(wb, wx, s)]
        print(f"  xtrack_weight 字段={m['xtrack_weight_field']}（连续臂应为 None=L89⑧ 缺陷·靠 TAG 认 well_X）· skipped={m['skipped']}")
        return
    d = args[0] if args else "."
    files = sorted(glob.glob(os.path.join(d, "*_diagXT_wb*_wx*_s*.jsonl")))
    if not files:
        print(f"未在 {d} 找到 *_diagXT_wb*_wx*_s*.jsonl"); return
    pat = re.compile(r"_diagXT_wb(\d+)_wx(\d+)_s(\d+)\.jsonl$")
    for f in files:
        mt = pat.search(f)
        if not mt: continue
        wb, wx, s = int(mt.group(1)), int(mt.group(2)), int(mt.group(3))
        runs[(wb, wx, s)] = analyze_one(f)
    seeds = sorted({s for (_, _, s) in runs}); wbs = sorted({wb for (wb, _, _) in runs})
    for wb in wbs:
        stage = "阶段A(well_B=0·well_X 单独)" if wb == 0 else f"阶段B(well_B={wb}·叠修法A)"
        print(f"\n========== {stage} ==========")
        for s in seeds:
            m0 = runs.get((wb, 0, s)); m2 = runs.get((wb, 200, s))
            if not (m0 and m2):
                if m0 or m2: print(f"  s{s}: 缺对照臂(只有 {'wx0' if m0 else 'wx200'})");
                continue
            print(f"  s{s} wx=0  : {fmt(m0)}")
            print(f"  s{s} wx=200: {fmt(m2)}")
            d_pos = m2['pos_miss_pct'] - m0['pos_miss_pct']
            d_stop = m2['inband_stop_pct'] - m0['inband_stop_pct']
            d_ec = ((m2['ec_reach_med'] - m0['ec_reach_med']) if (m2['ec_reach_med'] and m0['ec_reach_med']) else None)
            d_to = m2['timeout'] - m0['timeout']
            ec_str = f"{d_ec:+.1f}m" if d_ec is not None else "—"
            hit = (d_pos < 0 and (d_ec is None or d_ec < 0))
            print(f"  → Δ(wx200−wx0): 位置门miss {d_pos:+.1f}pt | 带内刹停 {d_stop:+.1f}pt | "
                  f"可达簇|e_cross| {ec_str} | 超时 {d_to:+d} | {'✅ 命中迹象' if hit else '⚠️ 未降/反升'}")
    print("\n判读：命中=位置门miss%↓ + 可达簇|e_cross|↓（拉进带·中位→<30）；远偏簇不动属预期(Φ_xtrack 够不到)。")
    print("超时率↑/可达簇|e_cross| 不动 → 疑 reward clip(VecNorm ±10 对 Φ~190)截断·或 R_lat/well_X 需调。")

if __name__ == "__main__":
    main()
