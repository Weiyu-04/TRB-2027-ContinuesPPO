#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生产版 block3 三分区探针 —— 方向 A 的 go/no-go（`Paper/方向A_可行性研判_紧急态可证明控制.md` §8-② +
`Paper/makeorbreak_连续时间清障下界_推导.md`；接手复审 2026-07-22·经 2 轮对抗审改硬）。

【它回答什么】把基准里策略真实进入的紧急态 ρ5，逐个判进三分区：
  · unavoidable = gap#1（迫近不可避 sound 证书·`usv_projection.imminent_unavoidable_certificate`）触发；
  · avoidable   = 机动族里 ∃ 一个成员，用【连续时间 sound 清障判据】（去 350m 膨胀·Lipschitz 采样）
                  在【相遇视界】内全程净空>0；
  · undecided   = 两者都不 = 方向 A 的命门。
未决区小 → 方向 A 有底；大 → 机动族太保守 / 视界артефакт / 方向 A 退回 demonstration。

【铁律：不自己写动力学】机动积分一律走官方 `usv_dynamics.step`（Krasowski 官方 RHS + odeint）；
ego 占据一律走官方 `usv_colregs._ego_rect`；不可避走官方 `imminent_unavoidable_certificate`。
本档只提供「清障判据的 Lipschitz 逻辑」+「采样/判分流程」，动力学与几何全调库。

════════════════════════════════════════════════════════════════════════════════════════════
【2 轮对抗审（2026-07-22）抓的坑·全在代码里焊死】——新窗口改这里前先读全，别倒退：
────────────────────────────────────────────────────────────────────────────────────────────
🔴 CRITICAL（判据 agent 造出真假认证·本窗口亲复现：认证+0.139m 而真距 0.0m 撞）：
  官方 RHS `dv/dt=a` 步内【不限速】(usv_dynamics clip_velocity 只在 10s 边界截)→ 满减速机动(默认族成员)
  步内 v【冲负】到 −2.4；旧代码 L 用【带符号】v 的 max → 反向转弯角速 |v|+|ω|R≈5 被 L 只记 0.45 → L 非上界 → 假认证。
  ✅ 修：L 的 ego 速度项用 **|v| + a_max·h**（|v|=真点速；+a_max·h 兜住子区间内单调增长 + 10s 边界钳前的 overshoot·
        因边界采样点是【钳后】值·会漏掉钳前峰）。转向项用【逐子区间真 |ω|】（本档积分每步 ω 恒定·精确又紧）。
🔴 HIGH（收集/分类 agent·视界偏差·会把"未决"虚高→假 NO-GO）：
  gap#1 只在 t*≤~11s 触发·ρ5 铺到 ~733m(~77s)·但清障若要求整 120s 净空>0·而 ω_max=0.03 开环转 120s 兜 317m 大圈
  绕回撞"已开走 1140m 的幽灵 CV 他船"→ 本来可避判未决。✅ 修：(a) 报【T 视界扫描】{20,30,40,60,80,120}s 让偏差显式
  (相遇视界≈t_cpa·obstacle 过顶后无碰)·(b) 加【转到清了就回正】机动(不兜圈)。绝不把单个 T=120 数当结论。
🟠 HIGH（per-step 双重计数·收集 agent）：一次相遇连续几步高度相关·项目口径=encounter 级 → 记 step_idx·报【按相遇】。
🟠 MEDIUM：h 必须整除 10s（否则 10s 钳错拍·轨迹不忠实执行）→ 入口 assert；他船宽须【精确真值】
  (gap#1 要 ≤真=under·清障要 ≥真=over·只有精确相等两边都 sound)→ 缺属性 fail-fast·分类 assert==SR108。
🟡 selftest 双档：原档(v 地板)【测不到】反向/overshoot 生产口径 → 新增【生产口径 selftest】(反向+10s钳)·
  确认修后 L 把上面那个 +0.139-vs-0 反向撞例判成 clears=False。
════════════════════════════════════════════════════════════════════════════════════════════

【F1-F5（推导 §4.5）落死处】
  F1 官方 odeint 常控 ~1e-8·子步细采 h≤0.5s；F2 见上 CRITICAL；F3 清障传【真他船长宽】over-approx；
  F4 L 他船项用逐场景真 |v_m|·取 abs()；F5 本档只判「裸不碰 d>0」·L·h/2≠keep-out。

【两阶段·纯 eval 不烧训练卡】
  --collect ：服务器·load 策略 rollout 真基准 → 收每步 ρ5 的 (ego,obs)+step_idx → 落 jsonl（需 vesselmodels/env）。
  --classify：对 ρ5 态逐个判三分区 + T 视界扫描 + 按相遇聚合 + 分距离/机动/gap#1 细分（需 vesselmodels 跑官方 step）。
  --selftest：本机·判据逻辑双档验证（不依赖 vesselmodels·CI 可跑）。
  --audit   ：本机·gap#1 说不可避的态·断言无机动能清障（soundness 交叉校验；需 vesselmodels）。
服务器跑前须 user 拍板 + 逐字预检 + screen 后台（04 运行手册）。
"""
import os
import sys
import json
import math
import numpy as np
from shapely.geometry import Polygon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 代码/ 进 path

# ── 官方常量（对齐 usv_dynamics / Table II）───────────────────────────────────────────────
A_MAX, W_MAX, V_MAX = 0.24, 0.03, 9.5
DECISION_DT = 10.0
L_SHIP, W_SHIP = 175.0, 25.4
R_CIRC = 0.5 * float(np.hypot(L_SHIP, W_SHIP))   # 88.4

# ── 官方件（生产路径必用；本机 --selftest 下 vesselmodels 缺 → 只验判据逻辑）─────────────────
_HAVE_OFFICIAL = True
try:
    from trb_env import usv_dynamics as _dyn
    from trb_env.usv_colregs import _ego_rect, VesselState, RHO_EMERGENCY
    from trb_env.usv_projection import imminent_unavoidable_certificate
except Exception as _e:
    _HAVE_OFFICIAL = False
    _IMPORT_ERR = _e

    def _ego_rect(center, theta, length, width):                 # 与官方逐字等价（已核 usv_colregs.py:398）
        hl, hw = 0.5 * length, 0.5 * width
        c, s = math.cos(theta), math.sin(theta)
        pts = [(center[0] + x * c - y * s, center[1] + x * s + y * c)
               for x, y in [(+hl, +hw), (+hl, -hw), (-hl, -hw), (-hl, +hw)]]
        return Polygon(pts)


# ════════════════════════════════════════════════════════════════════════════════════════
# 核心 1 · 连续时间 sound 清障判据（Lipschitz 采样·|v|+边界·逐子区间真 ω·可本机单测）
#   输入：已积分好的 ego 轨迹 + 逐子区间控制 ω + 他船 CV。返回 min_lb 和【首次不安全时刻】
#   （→ 让 --classify 一次积分、对多个 T 视界截断判 clears，Attack 5 视界扫描）。
# ════════════════════════════════════════════════════════════════════════════════════════
def clearance_profile(ts, traj, obs0, obs_len, obs_wid, h, omega_seg=None):
    """连续时间清障剖面。ts(N,) traj(N,4)[px,py,θ,v]·obs0(4,)·omega_seg=逐子区间|ω|上界(len N_sub)或 None→W_MAX。
    返回 dict(min_lb, first_unsafe_t, min_sample, sub_t, sub_ilb)。sub_t[k]=子区间右端时刻。"""
    ts = np.asarray(ts, float)
    traj = np.asarray(traj, float)
    if len(ts) < 2:
        d = _pair_dist(traj[0], obs0, ts[0], obs_len, obs_wid)
        return dict(min_lb=d, first_unsafe_t=(None if d > 0 else 0.0), min_sample=d, sub_t=[], sub_ilb=[])
    dt0 = ts[1] - ts[0]
    stride = max(1, int(round(h / dt0)))
    # 🟡 边界/尾部覆盖 assert（判据 agent Finding 6）：h 须为采样步整数倍·且末点被覆盖
    assert abs(stride * dt0 - h) < 1e-9, f"h={h} 非采样步 {dt0} 整数倍（会静默漏采）"
    idx = np.arange(0, len(ts), stride)
    if idx[-1] != len(ts) - 1:
        idx = np.append(idx, len(ts) - 1)                       # 强制覆盖末点（防尾段不检）
    hh = stride * dt0

    v_m = float(obs0[3])
    om = np.array([math.cos(obs0[2]), math.sin(obs0[2])])
    ds = np.empty(len(idx)); vv = np.empty(len(idx)); tt = np.empty(len(idx))
    for j, k in enumerate(idx):
        t = ts[k]; ex, ey, eth, ev = traj[k]
        oc = (obs0[0] + v_m * om[0] * t, obs0[1] + v_m * om[1] * t)
        ds[j] = _ego_rect((ex, ey), eth, L_SHIP, W_SHIP).distance(_ego_rect(oc, obs0[2], obs_len, obs_wid))
        vv[j] = ev; tt[j] = t

    # 🔴 L 的 ego 速度项：|v|（真点速·反向也对）+ a_max·hh（兜子区间内单调增长 + 10s 钳前 overshoot·CRITICAL 修）
    v_seg = np.maximum(np.abs(vv[:-1]), np.abs(vv[1:])) + A_MAX * hh
    if omega_seg is None:
        w_term = W_MAX * R_CIRC                                  # 未知 ω → 保守全局上界（sound·略松）
    else:
        ws = np.abs(np.asarray(omega_seg, float))
        if len(ws) != len(idx) - 1:                             # 采 stride≠1（selftest 细积分）→ 退回全局上界（sound）
            w_term = W_MAX * R_CIRC
        else:
            w_term = ws * R_CIRC
    L_seg = v_seg + w_term + abs(v_m)                           # F4 他船项 abs(v_m)
    ilb = (ds[:-1] + ds[1:] - L_seg * hh) / 2.0
    sub_t = tt[1:]                                              # 子区间右端时刻（首次不安全定位用）
    unsafe = np.where(ilb <= 0.0)[0]
    first_unsafe_t = float(sub_t[unsafe[0]]) if len(unsafe) else None
    return dict(min_lb=float(ilb.min()), first_unsafe_t=first_unsafe_t,
                min_sample=float(ds.min()), sub_t=sub_t, sub_ilb=ilb)


def _pair_dist(ego, obs0, t, obs_len, obs_wid):
    v_m = float(obs0[3]); om = np.array([math.cos(obs0[2]), math.sin(obs0[2])])
    oc = (obs0[0] + v_m * om[0] * t, obs0[1] + v_m * om[1] * t)
    return _ego_rect((ego[0], ego[1]), ego[2], L_SHIP, W_SHIP).distance(_ego_rect(oc, obs0[2], obs_len, obs_wid))


def clearance_lower_bound(ts, traj, obs0, obs_len, obs_wid, h, omega_seg=None, horizon=None):
    """便捷：clears over [0, horizon]（horizon=None→全程）。selftest 用。"""
    prof = clearance_profile(ts, traj, obs0, obs_len, obs_wid, h, omega_seg)
    fut = prof["first_unsafe_t"]
    clears = (fut is None) or (horizon is not None and fut > horizon)
    return dict(clears=bool(clears), min_lb=prof["min_lb"], min_sample=prof["min_sample"],
                first_unsafe_t=fut)


def keep_course_min_dist(ego, obs, obs_len, obs_wid, T=120.0, dt=0.5):
    """keep-course（ego 恒速恒向 vs 他船 CV）全程船体最小距离·纯几何（两 CV 矩形·无积分·无 vesselmodels）。
    <=0 = 真对撞航向（=方向 A 该解的真冲突）；>0 = 不做机动也安全（L196 揭示金标 ρ5 全落这类=假紧急）。"""
    ve = float(ego[3]); eh = (math.cos(ego[2]), math.sin(ego[2]))
    vm = float(obs[3]); oh = (math.cos(obs[2]), math.sin(obs[2]))
    n = int(round(T / dt)); best = 1e18
    for k in range(n + 1):
        t = k * dt
        ec = (ego[0] + ve * eh[0] * t, ego[1] + ve * eh[1] * t)
        oc = (obs[0] + vm * oh[0] * t, obs[1] + vm * oh[1] * t)
        d = _ego_rect(ec, ego[2], L_SHIP, W_SHIP).distance(_ego_rect(oc, obs[2], obs_len, obs_wid))
        if d < best:
            best = d
        if best <= 0.0:
            return 0.0
    return float(best)


def gen_synthetic_conflicts(n_target, rng, dims=None):
    """生成【真对撞】(ego,obs) 硬态：ego 原点朝 +x·速度 v_e；他船按碰撞点回溯放（head-on/crossing/overtake）。
    只保留 keep-course 真撞（min<=0）→ 保证是真冲突（方向 A 该解的硬态·非 L196 那种假紧急）。dims=(len,wid) None→采基准范围。"""
    recs = []
    tries = 0
    while len(recs) < n_target and tries < n_target * 80:
        tries += 1
        v_e = float(rng.uniform(4.0, V_MAX))
        kind = str(rng.choice(["head_on", "cross_port", "cross_star", "overtake"]))
        v_m = float(rng.uniform(2.0, V_MAX))
        if kind == "head_on":      th_m = math.pi + float(rng.uniform(-0.3, 0.3))
        elif kind == "cross_port": th_m = -math.pi / 2 + float(rng.uniform(-0.4, 0.4))
        elif kind == "cross_star": th_m = math.pi / 2 + float(rng.uniform(-0.4, 0.4))
        else:                      th_m = 0.0 + float(rng.uniform(-0.2, 0.2))   # overtake：同向·ego 追慢船
        om = (math.cos(th_m), math.sin(th_m))
        t_c = float(rng.uniform(15.0, 70.0))                    # 碰撞时刻
        jt = float(rng.uniform(-40.0, 40.0))                    # 碰撞点沿 ego 航向抖动
        cp = (v_e * t_c + jt, 0.0)                              # 碰撞点（ego 前方附近）
        p_m = (cp[0] - om[0] * v_m * t_c, cp[1] - om[1] * v_m * t_c)   # 他船回溯初位
        olen, owid = dims if dims else (float(rng.uniform(175.0, 260.0)), float(rng.uniform(25.4, 44.0)))
        ego = [0.0, 0.0, 0.0, v_e]; obs = [p_m[0], p_m[1], th_m, v_m]
        if keep_course_min_dist(ego, obs, olen, owid) <= 0.0:   # 真对撞才留
            recs.append(dict(kind=kind, ego=ego, obs=obs, obs_len=olen, obs_wid=owid))
    return recs


# ════════════════════════════════════════════════════════════════════════════════════════
# 核心 2 · 机动族（8 bang-bang·L139/L141 + 转到清就回正[Attack 5·不兜圈] + 可选两段）
# ════════════════════════════════════════════════════════════════════════════════════════
def maneuver_family(mode="full"):
    """返回 [(name, segments)]，segment=(a, ω, dur秒)·dur=None→吃到 T。
    mode: 'corners'=8 bang-bang；'full'=+转到清就回正(满舵 t1 后 ω=0 保安全脱离·不兜317m大圈)；'two'=full+两段减/加速。"""
    A, W = A_MAX, W_MAX
    fam = []
    for a in (-A, 0.0, +A):                                     # 8 bang-bang 角
        for w in (-W, 0.0, +W):
            if a == 0.0 and w == 0.0:
                continue
            fam.append((f"a{a:+.2f}_w{w:+.3f}", [(a, w, None)]))
    if mode in ("full", "two"):
        for w in (-W, +W):                                     # 转 t1 秒再回正（治 Attack 5 开环兜圈）
            for t1 in (20.0, 40.0, 60.0):
                fam.append((f"turn{w:+.3f}_{int(t1)}s_then_straight", [(0.0, w, t1), (0.0, 0.0, None)]))
                fam.append((f"accel_turn{w:+.3f}_{int(t1)}s_then_straight", [(+A, w, t1), (+A, 0.0, None)]))
    if mode == "two":
        for w in (-W, +W):
            fam.append((f"decel30_then_w{w:+.3f}", [(-A, 0.0, 30.0), (0.0, w, None)]))
    return fam


def _seg_at(segments, t):
    acc = 0.0
    for a, w, dur in segments:
        if dur is None:
            return (a, w)
        if t < acc + dur - 1e-9:
            return (a, w)
        acc += dur
    return (segments[-1][0], segments[-1][1])


# ════════════════════════════════════════════════════════════════════════════════════════
# 核心 3 · 机动积分（🔴 官方 usv_dynamics.step·10s 边界限速=执行口径·F1/F2·返回逐步 ω）
# ════════════════════════════════════════════════════════════════════════════════════════
def integrate_maneuver_official(ego0, segments, T, h=0.5, p=None):
    """官方积分。返回 (ts, traj[N,4], omega_seg[N-1])。仅生产（需 vesselmodels）。"""
    assert _HAVE_OFFICIAL, "integrate_maneuver_official 需官方 vesselmodels（本机→用 --selftest）"
    assert abs(round(DECISION_DT / h) * h - DECISION_DT) < 1e-9, \
        f"h={h} 必须整除 {DECISION_DT}s（否则 10s 边界钳错拍·轨迹不忠实·判据 agent Finding 3）"
    if p is None:
        p = _dyn.make_vessel_params(V_MAX)
    n = int(round(T / h))
    x = np.asarray(ego0, float).copy()
    ts = [0.0]; out = [x.copy()]; oseg = []
    for i in range(n):
        a, w = _seg_at(segments, i * h)
        oseg.append(w)
        x = _dyn.step(x, (a, w), h, p, clip_velocity=False)     # 窗内不截（忠实执行）
        t = (i + 1) * h
        if abs(t / DECISION_DT - round(t / DECISION_DT)) < 1e-9:  # 10s 边界钳 v（=usv_env clip_velocity=True）
            x[3] = float(np.clip(x[3], 0.0, V_MAX))
        ts.append(t); out.append(x.copy())
    return np.array(ts), np.array(out), np.array(oseg)


# ════════════════════════════════════════════════════════════════════════════════════════
# 分类：单个 ρ5 态 → 富结果（gap#1 + 各机动首次不安全时刻）→ 报告按 T 视界推分区
# ════════════════════════════════════════════════════════════════════════════════════════
def classify_state(ego, obs, obs_len, obs_wid, T=120.0, h=0.5, fam_mode="full", p=None):
    """返回 dict(gap1_unavoid, gap1_tstar, clear_times{name:first_unsafe_t或None=全程清}, dist)。"""
    # 🔴 他船占据 = 基准真实 shape（env 碰撞用 occupancy_at_time().shape·usv_termination:21·非假设的 SR108）。
    #   实测本基准他船宽 ~44.35 ≠ 项目一直假设的 SR108 25.4！传【该场景真宽/真长】给 gap#1（≤真=under 边界）
    #   与 clearance（≥真=over 边界）→ 精确真值使两个方向边界重合、两边都 sound。仅 fail-fast 挡垃圾值。
    assert obs_wid > 0.0 and obs_len > 0.0, f"他船尺寸非法 obs_len={obs_len} obs_wid={obs_wid}"
    dist = float(np.hypot(ego[0] - obs[0], ego[1] - obs[1]))
    gap1_unavoid, tstar = False, None
    if _HAVE_OFFICIAL:
        vp = p if p is not None else _dyn.make_vessel_params(V_MAX)
        ego_vs = VesselState(position=np.array(ego[:2], float), orientation=float(ego[2]),
                             velocity=float(ego[3]), length=L_SHIP)
        obs_vs = VesselState(position=np.array(obs[:2], float), orientation=float(obs[2]),
                             velocity=float(obs[3]), length=obs_len)
        gap1_unavoid, tstar = imminent_unavoidable_certificate(ego_vs, obs_vs, vp, obs_width=obs_wid, t_horizon=T)
    clear_times = {}
    if not gap1_unavoid:
        for name, segs in maneuver_family(fam_mode):
            ts, traj, oseg = integrate_maneuver_official(ego, segs, T, h, p)
            prof = clearance_profile(ts, traj, obs, obs_len, obs_wid, h, oseg)
            clear_times[name] = prof["first_unsafe_t"]          # None=全程清·数字=首次可能撞时刻
    return dict(gap1_unavoid=bool(gap1_unavoid), gap1_tstar=tstar, clear_times=clear_times, dist=dist)


def partition_at(res, T_cap):
    """给定分类富结果 + 视界 T_cap → {unavoidable/avoidable/undecided}。"""
    if res["gap1_unavoid"]:
        return "unavoidable"
    for name, fut in res["clear_times"].items():
        if fut is None or fut > T_cap:                          # 在 T_cap 内全程清
            return "avoidable"
    return "undecided"


# ════════════════════════════════════════════════════════════════════════════════════════
# 阶段 A · 收集（服务器·rollout 真基准·收 ρ5 态 + step_idx）
# ════════════════════════════════════════════════════════════════════════════════════════
def phase_collect():
    """SRC 分流：golden(金标策略·L196=假紧急) / adversarial(纯几何基线过盾·真硬态) / synthetic(合成真对撞·本机可跑)。"""
    src = os.environ.get("SRC", "golden").lower()
    print(f"[collect] SRC={src}", flush=True)
    if src == "synthetic":
        _collect_synthetic()
    elif src == "adversarial":
        _collect_adversarial()
    elif src == "golden":
        _collect_golden()
    else:
        raise SystemExit(f"未知 SRC={src}（golden/adversarial/synthetic）")


def _collect_synthetic():
    """合成真对撞硬态（纯几何·不需 vesselmodels·本机可跑）→ jsonl。每态独立 scn_idx（=独立相遇）。"""
    N = int(os.environ.get("SYNTH_N", "400"))
    seed = int(os.environ.get("SYNTH_SEED", "20260722"))
    OUT = os.environ.get("OUT_JSONL", "block3_synthetic_states.jsonl")
    _d = os.environ.get("SYNTH_DIMS", "")                       # "len,wid" 固定尺寸；空=采基准范围
    dims = tuple(float(x) for x in _d.split(",")) if _d else None
    rng = np.random.default_rng(seed)
    recs = gen_synthetic_conflicts(N, rng, dims)
    with open(OUT, "w") as fo:
        for i, r in enumerate(recs):
            fo.write(json.dumps(dict(seed=-1, scn="synthetic", scn_idx=i, step=0,
                     ego=r["ego"], obs=r["obs"], obs_len=r["obs_len"], obs_wid=r["obs_wid"],
                     source="synthetic", kind=r["kind"])) + "\n")
    kinds = {}
    for r in recs:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    print(f"[collect synthetic] {len(recs)} 真对撞态（keep-course 全真撞）→ {OUT} · 类型 {kinds}", flush=True)


def _collect_adversarial():
    """对抗基线（纯几何 dock_controller_v4 全程过盾·test1_pure_baseline 同款）·收 ρ5 真硬态。需 env。"""
    assert _HAVE_OFFICIAL, f"--collect adversarial 需 env（{_IMPORT_ERR}）"
    from trb_env.usv_continuous_shield import ContinuousProjectionEnv
    from trb_env.usv_scenarios import load_scenario_pool
    from trb_env.usv_env import A_NORMAL_OMEGA_MAX, A_NORMAL_ACCEL_MAX
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import dock_controller_v4 as DC
    from run_step4e import load_manifest_split
    MANIFEST = os.environ["STEP4E_MANIFEST"]
    OUT = os.environ.get("OUT_JSONL", "block3_adversarial_states.jsonl")
    V_RUN = float(os.environ.get("BASE_VRUN", "2.6"))
    bdir = os.environ.get("STEP4E_BALANCED_DIR") or os.path.dirname(os.path.abspath(MANIFEST))
    _tr, test_paths, _i = load_manifest_split(MANIFEST, bdir)
    pool = load_scenario_pool(test_paths)
    WMAX, AMAX = A_NORMAL_OMEGA_MAX, A_NORMAL_ACCEL_MAX
    print(f"[collect adversarial] 纯几何基线过盾·{len(pool)} 场景·v_run={V_RUN} → {OUT}", flush=True)

    def _mk(sc, pp):
        return ContinuousProjectionEnv(sc, pp, shield=True, goal_cone_half=None, goal_v_floor=2.0, augment_rho=False)
    n_rho5 = 0
    with open(OUT, "w") as fo:
        for si, (sc, pp) in enumerate(pool):
            env = _mk(sc, pp); obs, info = env.reset(seed=0)
            for step_i in range(200):
                ego = env._ego_vs(); gg = env.env.goal_center
                try:
                    theta_g = 0.5 * (env.env.goal.orientation.start + env.env.goal.orientation.end)
                except Exception:
                    theta_g = 0.0
                st = [ego.position[0], ego.position[1], ego.orientation, float(getattr(ego, "velocity", 0.0))]
                act = DC.dock_controller(st, (gg[0], gg[1]), theta_g=theta_g, wmax=WMAX, v_run=V_RUN)
                act = np.array([float(np.clip(act[0], -AMAX, AMAX)), act[1]])
                ev, ov = env._ego_vs(), env._obs_vs()
                ob = env._obstacles[0] if env._obstacles else None
                owid = float(ob.obstacle_shape.width) if ob is not None else W_SHIP
                olen = float(env._obs_length)
                obs, _r, term, trunc, info = env.step(np.asarray(act, float))
                if info.get("rho") == RHO_EMERGENCY and ov is not None:
                    fo.write(json.dumps(dict(seed=-1, scn=os.path.basename(str(sc)), scn_idx=si, step=step_i,
                        ego=[float(ev.position[0]), float(ev.position[1]), float(ev.orientation), float(ev.velocity)],
                        obs=[float(ov.position[0]), float(ov.position[1]), float(ov.orientation), float(ov.velocity)],
                        obs_len=olen, obs_wid=owid, source=info.get("source"))) + "\n")
                    n_rho5 += 1
            fo.flush()
    print(f"[collect adversarial] done · ρ5 态={n_rho5} → {OUT}", flush=True)


def _collect_golden():
    assert _HAVE_OFFICIAL, f"--collect 需 vesselmodels/env（本机缺：{_IMPORT_ERR}）→ 服务器跑"
    import glob as _glob
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from trb_env.train import make_obs_transform
    from trb_env.usv_continuous_shield import ContinuousProjectionEnv
    from trb_env.usv_scenarios import load_scenario_pool

    CKPT_DIR = os.environ["CKPT_DIR"]
    CKPT_TMPL = os.environ.get("CKPT_TMPL", "Continuous-safe_s{s}_L1rateON_ppo_s{s}")
    SEEDS = [int(x) for x in os.environ.get("SEEDS", "0 1 2 3 4 5 6 7 8 9").split()]
    MANIFEST = os.environ.get("STEP4E_MANIFEST", "")
    OUT = os.environ.get("OUT_JSONL", "block3_rho5_states.jsonl")

    if MANIFEST:
        from run_step4e import load_manifest_split
        bdir = os.environ.get("STEP4E_BALANCED_DIR") or os.path.dirname(os.path.abspath(MANIFEST))
        _tr, test_paths, _info = load_manifest_split(MANIFEST, bdir)
        scn = test_paths; src = f"manifest 测试集(n={len(scn)})"
    else:
        scn = sorted(_glob.glob(os.environ.get("SCN_GLOB", "/tmp/trb_scenarios_pool/T-*.xml")))
        src = f"⚠️glob sanity(n={len(scn)}·非标准集)"
    pool = load_scenario_pool(scn)
    print(f"[collect] 场景源={src} 种子={SEEDS} → {OUT}", flush=True)
    print("  ⚠️ 群体=【本策略确定性部署】遇到的 ρ5（偏易·欠采迫近硬态）·非 worst-case·判读须披露（Attack 2b）", flush=True)

    n_rho5 = 0
    with open(OUT, "w") as fo:
        for s in SEEDS:
            ck = os.path.join(CKPT_DIR, CKPT_TMPL.format(s=s))
            if not (os.path.exists(ck + ".zip") and os.path.exists(ck + "_vecnorm.pkl")):
                print(f"  s{s}: 缺 checkpoint → 跳过", flush=True); continue
            # 🔴 env 配置须【逐字】= 金标训练/eval 配置（金标 run_config：shield=True/goal_cone=None/
            #    goal_v_floor=2.0/augment_rho=False·已核 run_metadata_L1rateON_ppo_s0.json）·= 已验证 harness
            #    closed_loop_dock.py 同款（它复现金标逐种子到达率）→ 收集的 ρ5 态才忠实、vecnorm 维才对得上。
            def _mk(sc, pp):
                return ContinuousProjectionEnv(sc, pp, shield=True, goal_cone_half=None,
                                               goal_v_floor=2.0, augment_rho=False)
            bv = DummyVecEnv([lambda: _mk(pool[0][0], pool[0][1])])
            vn = VecNormalize.load(ck + "_vecnorm.pkl", bv); vn.training = False
            if int(np.asarray(vn.obs_rms.mean).shape[0]) != int(bv.observation_space.shape[0]):
                raise SystemExit(f"s{s}: vecnorm 维≠env 维（盾/augment 配置不匹配 checkpoint）")
            tf = make_obs_transform(vn)
            model = PPO.load(ck + ".zip", device="cpu")
            for si, (sc, pp) in enumerate(pool):
                env = _mk(sc, pp)
                obs, info = env.reset(seed=0)
                for step_i in range(200):
                    act, _ = model.predict(tf(obs), deterministic=True)
                    ev, ov = env._ego_vs(), env._obs_vs()       # pre-step（=盾这步据以分类的态）
                    ob = env._obstacles[0] if env._obstacles else None
                    if ob is not None and not hasattr(ob.obstacle_shape, "width"):
                        raise SystemExit(f"他船 shape 无 width 属性 → 无法保证 obs_wid 精确（宽度双用·必须真值）")
                    owid = float(ob.obstacle_shape.width) if ob is not None else W_SHIP
                    obs, _r, term, trunc, info = env.step(np.asarray(act, float))
                    if info.get("rho") == RHO_EMERGENCY and ov is not None:
                        fo.write(json.dumps(dict(
                            seed=s, scn=os.path.basename(str(sc)), scn_idx=si, step=step_i,
                            ego=[float(ev.position[0]), float(ev.position[1]), float(ev.orientation), float(ev.velocity)],
                            obs=[float(ov.position[0]), float(ov.position[1]), float(ov.orientation), float(ov.velocity)],
                            obs_len=float(env._obs_length), obs_wid=owid, source=info.get("source"))) + "\n")
                        n_rho5 += 1
                    if term or trunc:
                        break
            fo.flush()                                          # 逐种子落盘（断线不丢·Finding 6 LOW）
            print(f"  s{s}: 累计 ρ5 态 {n_rho5}", flush=True)
    print(f"[collect] done · ρ5 态总数={n_rho5} → {OUT}", flush=True)


# ════════════════════════════════════════════════════════════════════════════════════════
# 阶段 B · 分类 + 报告（T 视界扫描 + 按相遇聚合 + 分距离/机动/gap#1）
# ════════════════════════════════════════════════════════════════════════════════════════
def phase_classify():
    assert _HAVE_OFFICIAL, f"--classify 需 vesselmodels 跑官方 step（本机缺：{_IMPORT_ERR}）→ 服务器跑"
    INP = os.environ.get("IN_JSONL", "block3_rho5_states.jsonl")
    Tmax = float(os.environ.get("PROBE_T", "120"))
    h = float(os.environ.get("PROBE_H", "0.5"))
    fam_mode = os.environ.get("PROBE_FAM", "full")              # corners/full/two
    T_SWEEP = [float(x) for x in os.environ.get("PROBE_TSWEEP", "20 30 40 60 80 120").split()]
    recs = [json.loads(l) for l in open(INP)]
    print(f"[classify] {len(recs)} 个 ρ5 态 · Tmax={Tmax} h={h} 机动族={fam_mode} · 视界扫描={T_SWEEP}", flush=True)
    _ws = sorted(set(round(float(r["obs_wid"]), 2) for r in recs))
    _ls = sorted(set(round(float(r["obs_len"]), 2) for r in recs))
    print(f"  他船真实 shape：宽∈{_ws[:8]}{'…' if len(_ws) > 8 else ''}  长∈{_ls[:8]}{'…' if len(_ls) > 8 else ''}", flush=True)
    if any(abs(w - W_SHIP) > 1e-6 for w in _ws):
        print(f"  ⚠️ 他船宽 ≠ SR108 {W_SHIP}m → 项目'两船皆 SR108'假设不成立·gap#1 默认 obs_width=25.4 是 under-approx(sound 但保守)·本探针用真宽(两边 sound)", flush=True)
    p = _dyn.make_vessel_params(V_MAX)
    results = []
    for i, r in enumerate(recs):
        res = classify_state(r["ego"], r["obs"], r["obs_len"], r["obs_wid"], T=Tmax, h=h, fam_mode=fam_mode, p=p)
        res["_rec"] = r
        res["kc"] = keep_course_min_dist(r["ego"], r["obs"], r["obs_len"], r["obs_wid"])   # 不做机动最小净空
        results.append(res)
        if (i + 1) % 100 == 0:
            print(f"  ...{i+1}/{len(recs)}", flush=True)
    N = max(1, len(results))

    # ── 0) 🔴 冲突严重度分桶（L196 命门：多数 ρ5 是"假紧急"·keep-course 不做机动已安全→隔离真冲突）──
    genuine = [r for r in results if r["kc"] <= 0.0]           # 真对撞航向 = 方向 A 该解的硬态
    nearmiss = [r for r in results if 0.0 < r["kc"] <= 50.0]
    farsafe = [r for r in results if r["kc"] > 50.0]
    print(f"\n===== 冲突严重度（keep-course=不做机动的全程最小净空）=====")
    print(f"  真对撞(kc≤0)={len(genuine)}({100*len(genuine)/N:.1f}%) · 擦边(0-50m)={len(nearmiss)} · 假紧急(>50m 不做也安全)={len(farsafe)}({100*len(farsafe)/N:.1f}%)")
    if not genuine:
        print("  ⚠️🔴 无真对撞态 → 本群体【测不了方向 A】（=L196 金标同款）·须换硬源 SRC=adversarial/synthetic")

    # ── 1) 🔴 T 视界扫描（Attack 5）· 全 population + 【真对撞子集】(真答案落在这里) ──────────
    for label, subset in [("全 population", results), ("真对撞子集(kc≤0)", genuine), ("真对撞+擦边(kc≤50)", genuine + nearmiss)]:
        if not subset:
            continue
        m = len(subset)
        print(f"\n----- T 视界扫描 · {label}（n={m}）-----")
        print(f"  {'T(s)':>5} | {'unavoid%':>9} {'avoid%':>8} {'undec%':>8}")
        for Tc in T_SWEEP:
            c = {"unavoidable": 0, "avoidable": 0, "undecided": 0}
            for res in subset:
                c[partition_at(res, Tc)] += 1
            print(f"  {Tc:>5.0f} | {100*c['unavoidable']/m:>8.1f} {100*c['avoidable']/m:>7.1f} {100*c['undecided']/m:>7.1f}")
    print("  ⓘ 【真对撞子集】的 undec/unavoid 才是方向 A 命门数；undec 随 T 缩到相遇视界(≈40-60s)仍大=真命门·即塌=视界артефакт。")

    # ── 2) 按【相遇】聚合（Attack 2a·项目口径 encounter 级）─────────────────────────────
    #   相遇 = 同 (seed,scn_idx) 内 step 连续的一段 ρ5。相遇"未决"= 该段任一步未决。
    enc = {}
    for res in results:
        r = res["_rec"]; key = (r["seed"], r["scn_idx"])
        enc.setdefault(key, []).append((r["step"], res))
    print("\n===== 按相遇聚合（视界=40s 为例·相遇有≥1 未决步则记未决）=====")
    for Tc in (40.0, 80.0):
        n_enc = 0; undec_enc = 0; unavoid_enc = 0
        for key, steps in enc.items():
            steps.sort()
            # 切成连续 run（step 相邻）
            runs = []; cur = [steps[0]]
            for a, b in zip(steps, steps[1:]):
                if b[0] == a[0] + 1: cur.append(b)
                else: runs.append(cur); cur = [b]
            runs.append(cur)
            for run in runs:
                n_enc += 1
                parts = [partition_at(rs, Tc) for _, rs in run]
                if any(pt == "undecided" for pt in parts): undec_enc += 1
                elif all(pt == "unavoidable" for pt in parts): unavoid_enc += 1
        print(f"  T={Tc:.0f}s: 相遇总数={n_enc} · 有未决步的相遇={undec_enc}({100*undec_enc/max(1,n_enc):.1f}%) · 全程不可避={unavoid_enc}")

    # ── 3) 分距离 + gap#1 触发率（视界=40s）────────────────────────────────────────────
    bins = [(0, 400), (400, 780), (780, 1e18)]
    bt = {b: {"unavoidable": 0, "avoidable": 0, "undecided": 0} for b in bins}
    gap1_fire = sum(1 for res in results if res["gap1_unavoid"])
    Tc = 40.0
    for res in results:
        for b in bins:
            if b[0] <= res["dist"] < b[1]:
                bt[b][partition_at(res, Tc)] += 1
    print(f"\n===== 分距离（视界=40s·未决是否只集中近距）· gap#1 触发率={100*gap1_fire/N:.2f}% of ρ5 =====")
    print("  （复审：gap#1 若近空 → A-ii 有界严重度打空集·须砍/hedge）")
    for b in bins:
        tot = sum(bt[b].values())
        if tot:
            hi = "∞" if b[1] > 1e17 else f"{b[1]:.0f}"
            print(f"  [{b[0]:>4}-{hi:>4}]m n={tot:>5}: unavoid {100*bt[b]['unavoidable']/tot:4.1f}% / "
                  f"avoid {100*bt[b]['avoidable']/tot:4.1f}% / undec {100*bt[b]['undecided']/tot:4.1f}%")

    # ── 4) 机动族覆盖（哪些机动最常首个清障·视界=40s）────────────────────────────────
    by_clear = {}
    for res in results:
        if res["gap1_unavoid"]: continue
        for name, fut in res["clear_times"].items():
            if fut is None or fut > Tc:
                by_clear[name] = by_clear.get(name, 0) + 1
    print("\n===== 机动族覆盖（各机动在 40s 视界内清障的态数·诊断族够不够）=====")
    for k, v in sorted(by_clear.items(), key=lambda x: -x[1])[:12]:
        print(f"  {k:>28}: {v}")
    print("\n🔴 go/no-go 判读（别照抄单数）：")
    print("   · 🎯 真答案 = 【真对撞子集(kc≤0)】的分区：unavoid+avoid 高·undec 小 → 方向 A 有底(能可证明处理硬态)；")
    print("     undec 大(随 T 缩到相遇视界仍大) → 机动族覆盖不到 → 方向 A 弱 / 退 demonstration。")
    print("   · 若真对撞子集=0(如 SRC=golden L196)：本群体测不了方向 A·换 SRC=adversarial/synthetic。")
    print("   · gap#1 触发率近空 → 砍 A-ii 有界严重度。按相遇 undec% = encounter 级口径。")
    print("   · 边界：单障碍 CV·真 go 还须核 block2 执行接管(L192-I OOD 反噬)+多障碍不合成。")


# ════════════════════════════════════════════════════════════════════════════════════════
# --audit：gap#1 说不可避的态·断言无机动能清障（soundness 交叉校验·Attack 3）
# ════════════════════════════════════════════════════════════════════════════════════════
def phase_audit():
    assert _HAVE_OFFICIAL, f"--audit 需 vesselmodels（本机缺：{_IMPORT_ERR}）"
    INP = os.environ.get("IN_JSONL", "block3_rho5_states.jsonl")
    h = float(os.environ.get("PROBE_H", "0.5")); Tmax = float(os.environ.get("PROBE_T", "120"))
    p = _dyn.make_vessel_params(V_MAX)
    recs = [json.loads(l) for l in open(INP)]
    n_unavoid = n_bad = 0
    for r in recs:
        res = classify_state(r["ego"], r["obs"], r["obs_len"], r["obs_wid"], T=Tmax, h=h, fam_mode="two", p=p)
        if res["gap1_unavoid"]:
            n_unavoid += 1
            # gap#1 说不可避 → 任何机动都不该【全程】清障（否则两 sound 证书自相矛盾=有 bug）
            ts0 = None
            for name, segs in maneuver_family("two"):
                ts, traj, oseg = integrate_maneuver_official(r["ego"], segs, Tmax, h, p)
                prof = clearance_profile(ts, traj, r["obs"], r["obs_len"], r["obs_wid"], h, oseg)
                if prof["first_unsafe_t"] is None:
                    n_bad += 1
                    print(f"  🔴 矛盾：gap#1 不可避 但 {name} 全程清障 → soundness bug（ego={r['ego']} obs={r['obs']}）")
                    break
    print(f"[audit] gap#1 不可避态={n_unavoid} · 矛盾(机动却清障)={n_bad} → {'✅ 一致' if n_bad==0 else '🔴 有 soundness bug'}")


# ════════════════════════════════════════════════════════════════════════════════════════
# --selftest：判据逻辑双档验证（本机·不依赖 vesselmodels）
# ════════════════════════════════════════════════════════════════════════════════════════
def _rk4(ego0, a, w, T, dt=0.05, floor_v=True, clip10=False):
    """自包含 RK4·常控。floor_v=True→步内 v 地板 0（原档）；clip10=True→只 10s 边界钳(=生产口径·允许反向/overshoot)。"""
    a = float(np.clip(a, -A_MAX, A_MAX)); w = float(np.clip(w, -W_MAX, W_MAX))
    def rhs(x):
        v = x[3]
        dv = 0.0 if (floor_v and ((v <= 0.0 and a < 0.0) or (v >= V_MAX and a > 0.0))) else a
        return np.array([v * math.cos(x[2]), v * math.sin(x[2]), w, dv])
    n = int(round(T / dt)); x = np.asarray(ego0, float).copy(); out = [x.copy()]
    for i in range(n):
        k1 = rhs(x); k2 = rhs(x + 0.5*dt*k1); k3 = rhs(x + 0.5*dt*k2); k4 = rhs(x + dt*k3)
        x = x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        t = (i+1)*dt
        if floor_v:
            x[3] = min(max(x[3], 0.0), V_MAX)
        elif clip10 and abs(t/DECISION_DT - round(t/DECISION_DT)) < 1e-9:
            x[3] = float(np.clip(x[3], 0.0, V_MAX))
        out.append(x.copy())
    return np.arange(n+1)*dt, np.array(out)


def _scan(floor_v, clip10, label, seed):
    rng = np.random.default_rng(seed); N = 600; nc = nf = 0; mg = np.inf; worst = None
    for _ in range(N):
        e = [0, 0, rng.uniform(-np.pi, np.pi), rng.uniform(0, V_MAX)]
        ang = rng.uniform(-np.pi, np.pi); dd = rng.uniform(200, 1200)
        o = [dd*np.cos(ang), dd*np.sin(ang), rng.uniform(-np.pi, np.pi), rng.uniform(0, V_MAX)]
        a = rng.choice([-A_MAX, 0, A_MAX]); w = rng.choice([-W_MAX, 0, W_MAX]); h = rng.choice([0.25, 0.5, 1.0])
        ts, traj = _rk4(e, a, w, 60.0, floor_v=floor_v, clip10=clip10)   # 0.05s 网格（L≈24→格间≤1.2m·足够抓穿模）
        res = clearance_lower_bound(ts, traj, o, L_SHIP, W_SHIP, h)   # selftest 用全局 W_MAX 项（stride≠1）
        if res["clears"]:
            nc += 1
            v = float(o[3]); om = np.array([math.cos(o[2]), math.sin(o[2])])
            tmin = min(_ego_rect(traj[k][:2], traj[k][2], L_SHIP, W_SHIP).distance(   # 真值=0.05s 全轨迹（反向撞落此尺度）
                       _ego_rect((o[0]+v*om[0]*ts[k], o[1]+v*om[1]*ts[k]), o[2], L_SHIP, W_SHIP))
                       for k in range(len(ts)))
            if tmin <= 0.0:
                nf += 1
            if tmin - res["min_lb"] < mg:
                mg = tmin - res["min_lb"]; worst = (e, a, w, tmin, res["min_lb"])
    ok = (nf == 0 and mg >= -1e-6)
    print(f"  [{label}] clears={nc} 假认证={nf} min(true-lb)={mg:.4f}m → {'✅ SOUND' if ok else '🔴 UNSOUND'}")
    return ok


def phase_selftest():
    print("=== --selftest：判据逻辑双档（RK4 自包含·不依赖 vesselmodels·真值 0.05s 网格·L≈24→格间≤1.2m）===")
    ok1 = _scan(floor_v=True, clip10=False, label="档A·v地板[0,9.5]（原档）", seed=20260721)
    print("  档B=【生产口径】：步内不地板·只 10s 边界钳 → 允许【反向 v<0 + overshoot v>9.5】(=官方执行·CRITICAL 修覆盖)")
    ok2 = _scan(floor_v=False, clip10=True, label="档B·生产口径(反向+overshoot)", seed=42)
    # 定向复现 2 轮审那个反向假认证例·确认修后判 clears=False
    e = [0, 0, -2.654, 0.324]; a = -0.24; w = -0.030; o = [-113.66, 82.4, -1.35, 0.45]; h = 0.5
    ts, traj = _rk4(e, a, w, 60.0, floor_v=False, clip10=True)
    r = clearance_lower_bound(ts, traj, o, L_SHIP, W_SHIP, h)
    v = float(o[3]); om = np.array([math.cos(o[2]), math.sin(o[2])])
    tsf, trajf = _rk4(e, a, w, 60.0, dt=0.01, floor_v=False, clip10=True)
    tmin = min(_ego_rect(trajf[k][:2], trajf[k][2], L_SHIP, W_SHIP).distance(
               _ego_rect((o[0]+v*om[0]*tsf[k], o[1]+v*om[1]*tsf[k]), o[2], L_SHIP, W_SHIP)) for k in range(len(tsf)))
    fixed = not (r["clears"] and tmin <= 0.0)
    print(f"  [反向假认证复例] clears={r['clears']} min_lb={r['min_lb']:.4f} 真距={tmin:.4f}m → "
          f"{'✅ 修后不再假认证' if fixed else '🔴 仍假认证'}")
    ok = ok1 and ok2 and fixed
    print("  " + ("✅ 判据逻辑 SOUND（两档 + 反向复例均过·生产用官方 step 喂同一判据）" if ok else "🔴 判据仍有洞"))
    return 0 if ok else 1


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "--selftest"
    if mode == "--collect":
        phase_collect()
    elif mode == "--classify":
        phase_classify()
    elif mode == "--audit":
        phase_audit()
    elif mode == "--selftest":
        sys.exit(phase_selftest())
    else:
        print(__doc__)
        print("用法: python block3_partition_probe.py [--collect | --classify | --audit | --selftest]")
        sys.exit(2)
