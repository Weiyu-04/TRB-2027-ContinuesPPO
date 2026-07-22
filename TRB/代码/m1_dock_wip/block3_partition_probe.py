#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生产版 block3 三分区探针 —— 方向 A 的 go/no-go（`Paper/方向A_可行性研判_紧急态可证明控制.md` §8-② +
`Paper/makeorbreak_连续时间清障下界_推导.md`；接手复审 2026-07-22）。

【它回答什么】把基准里策略真实进入的紧急态 ρ5，逐个判进三分区：
  · unavoidable = gap#1（迫近不可避 sound 证书·`usv_projection.imminent_unavoidable_certificate`）触发；
  · avoidable   = 机动族里 ∃ 一个成员，用【连续时间 sound 清障判据】（去 350m 膨胀·Lipschitz 采样）全程净空>0；
  · undecided   = 两者都不（既证不了不可避、也证不了可避）= 方向 A 的命门。
未决区小 → 方向 A 有底；大 → 机动族太保守 / 方向 A 退回 demonstration。

【铁律：不自己写动力学】机动积分一律走官方 `usv_dynamics.step`（Krasowski 官方 RHS + odeint）；
ego 占据一律走官方 `usv_colregs._ego_rect`；不可避走官方 `imminent_unavoidable_certificate`。
本档只提供「清障判据的 Lipschitz 逻辑」+「采样/判分流程」，动力学与几何全调库。

【复审锁定的 F1-F5（每条都在代码里落死，别再踩）】
  F1 采样精度：官方 step=odeint，常控 ~1e-8m；子步细采（h≤0.5s）→ 采样值精确、不吃 margin。绝不用 Euler。
  F2 速度口径（🔴 2026-07-22 复审生产迁移 agent 造出真假认证的坑）：官方 env 执行 = 每 10s 边界把 v 截回 v_max
      （usv_env clip_velocity=True）→ 步内 v 可冲到 v_bnd=11.9。**机动必须按这个 10s 边界口径积分**（本档
      integrate_maneuver 就是这么做）；且 **每子区间的 L 用该区间实测最高速**（不是全局 11.9），sound 且紧、
      且就算有人改了限速节奏也不会静默失效。绝不用 clip_velocity=False 裸积分（那会 v→23.9 → L 偏小 → 假认证）。
  F3 他船占据 over-approx：清障方向要 over-approx 真实他船 → 传【真实他船长宽】（基准 SR108=25.4×175·精确）。
  F4 他船速度：L 的他船项用【该场景真 v_m】（CV·已知），取 abs()（复审数学 agent：原 selfcheck 用裸值·负速会 unsound）。
  F5 两种 margin 别混：L·h/2=采样 soundness 裕度（本档）；350m/keep-out=安全设计裕度（命题1）。本档只判「裸不碰 d>0」。

【两阶段·纯 eval 不烧训练卡】
  --collect  ：服务器·load 策略 rollout 真基准 → 收集每步 ρ5 的 (ego,obs) → 落 jsonl（需 vesselmodels/env）。
  --classify ：对收集的 ρ5 态逐个判三分区 → 报分区占比 + 分距离/机动/gap#1 触发率细分（需 vesselmodels 跑官方 step）。
  --selftest ：本机·用【已验证的 RK4 自包含积分】喂同一套清障判据逻辑，复现 selfcheck 的 600 对抗 0 假认证
               → 证明「判据逻辑」正确（不依赖 vesselmodels·CI 可跑）。生产路径与它共用同一个 clearance_lower_bound。

用法见文件尾 __main__ 顶部与 04 运行手册（服务器跑前须 user 拍板 + 逐字预检 + screen 后台）。
"""
import os
import sys
import json
import math
import numpy as np
from shapely.geometry import Polygon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 代码/ 进 path

# ── 官方常量（对齐 usv_dynamics / Table II·别在此另立真值）───────────────────────────────
A_MAX, W_MAX, V_MAX = 0.24, 0.03, 9.5
DECISION_DT = 10.0                 # 决策步长（官方 usv_dynamics.DECISION_DT）
L_SHIP, W_SHIP = 175.0, 25.4       # 本船 SR108（basein 他船同型）
R_CIRC = 0.5 * float(np.hypot(L_SHIP, W_SHIP))   # 88.4·ego 外接圆（Lipschitz ego 旋转项）

# ── 官方件（生产路径必用；本机 selftest 下 vesselmodels 缺 → 用自包含 RK4 只验判据逻辑）──────
_HAVE_OFFICIAL = True
try:
    from trb_env import usv_dynamics as _dyn                      # 官方 step（odeint + Krasowski RHS）
    from trb_env.usv_colregs import _ego_rect, is_emergency, VesselState, RHO_EMERGENCY
    from trb_env.usv_projection import imminent_unavoidable_certificate
except Exception as _e:                                           # 本机无 vesselmodels → 只能跑 --selftest
    _HAVE_OFFICIAL = False
    _IMPORT_ERR = _e

    def _ego_rect(center, theta, length, width):                 # selftest 用·与官方逐字等价（已核 usv_colregs.py:398）
        hl, hw = 0.5 * length, 0.5 * width
        c, s = math.cos(theta), math.sin(theta)
        pts = [(center[0] + x * c - y * s, center[1] + x * s + y * c)
               for x, y in [(+hl, +hw), (+hl, -hw), (-hl, -hw), (-hl, +hw)]]
        return Polygon(pts)


# ════════════════════════════════════════════════════════════════════════════════════════
# 核心 1 · 连续时间 sound 清障判据（Lipschitz 采样·per-subinterval 局部 L·可本机单测）
#   输入：一条【已积分好】的 ego 轨迹（时刻 ts + 状态 traj[N,4]）+ 他船 CV 初值 + 尺寸。
#   与「怎么积分的」解耦 → 生产喂官方 step 轨迹、selftest 喂 RK4 轨迹，判据逻辑同一份。
# ════════════════════════════════════════════════════════════════════════════════════════
def clearance_lower_bound(ts, traj, obs0, obs_len, obs_wid, h):
    """连续时间清障下界（推导 §2·per-subinterval 局部 L·F2/F4 落死）。

    ts   : (N,) 采样时刻（须含 h 的整数倍网格；h=子区间步长）。
    traj : (N,4) ego 状态 [px,py,θ,v]，**须为按执行口径（10s 边界限速）积分出的真轨迹**（F2）。
    obs0 : (4,) 他船初值 [px,py,θ,v]（CV 外推·恒向）。
    返回 {clears, min_lb, min_sample}。clears=（连续时间最小净空严格下界>0）。
    """
    ts = np.asarray(ts, float)
    traj = np.asarray(traj, float)
    stride = max(1, int(round(h / (ts[1] - ts[0])))) if len(ts) > 1 else 1
    idx = np.arange(0, len(ts), stride)
    hh = stride * (ts[1] - ts[0]) if len(ts) > 1 else h

    v_m = float(obs0[3])
    om = np.array([math.cos(obs0[2]), math.sin(obs0[2])])
    # 采样点精确船体距离（凸多边形·shapely·非近似·非膨胀）
    ds = np.empty(len(idx))
    vv = np.empty(len(idx))
    for j, k in enumerate(idx):
        t = ts[k]
        ex, ey, eth, ev = traj[k]
        oc = (obs0[0] + v_m * om[0] * t, obs0[1] + v_m * om[1] * t)
        ds[j] = _ego_rect((ex, ey), eth, L_SHIP, W_SHIP).distance(
                _ego_rect(oc, obs0[2], obs_len, obs_wid))
        vv[j] = ev
    if len(idx) < 2:
        return {"clears": bool(ds.min() > 0.0), "min_lb": float(ds.min()), "min_sample": float(ds.min())}

    # per-subinterval 局部 L_k：ego 项用【该子区间两端实测最高速】（F2·常控内 v 单调 → 端点取 max 即区间 max），
    #   他船项用 |v_m|（F4·abs 防负速 unsound·复审数学 agent 抓的 selfcheck 隐患·此处落死）。
    v_seg = np.maximum(vv[:-1], vv[1:])                          # (M,) 各子区间 ego 速上界
    # 转向项：本机动恒 ω → 用 |ω|·R_circ（ego 旋转点最大线速）。ω 未知时的通用上界=W_MAX·R_circ；
    #   这里由调用方传入的轨迹已隐含固定 ω，用全局 W_MAX·R_circ 保守上界（sound·略松·可选传真 ω 收紧）。
    L_seg = v_seg + W_MAX * R_CIRC + abs(v_m)
    ilb = (ds[:-1] + ds[1:] - L_seg * hh) / 2.0
    return {"clears": bool(ilb.min() > 0.0), "min_lb": float(ilb.min()), "min_sample": float(ds.min())}


# ════════════════════════════════════════════════════════════════════════════════════════
# 核心 2 · 机动族（初版=控制箱 8 bang-bang·L139/L141；可选两段序列·复审 flag 迫近段单段转不过）
# ════════════════════════════════════════════════════════════════════════════════════════
def maneuver_family(two_segment=False):
    """返回机动列表。每个 = (name, segments)，segment=(a, ω, dur秒)。恒开环。
    单段 8 个 = {−a,0,+a}×{−ω,0,+ω} − (0,0)（保向漂移无意义·去掉）。
    two_segment=True 追加「先减速/加速 t1 再满舵转」等复合（L141：逃逸强几何依赖·无干净规律·须多试）。"""
    A, W = A_MAX, W_MAX
    fam = []
    for a in (-A, 0.0, +A):
        for w in (-W, 0.0, +W):
            if a == 0.0 and w == 0.0:
                continue
            fam.append((f"a{a:+.2f}_w{w:+.3f}", [(a, w, None)]))   # dur=None → 用全程 T
    if two_segment:
        for w in (-W, +W):                                        # 先满减速拖时间→再满舵转（争转向角）
            fam.append((f"decel30_then_w{w:+.3f}", [(-A, 0.0, 30.0), (0.0, w, None)]))
        for w in (-W, +W):                                        # 先满加速拉开半径→再满舵转
            fam.append((f"accel30_then_w{w:+.3f}", [(+A, 0.0, 30.0), (0.0, w, None)]))
    return fam


# ════════════════════════════════════════════════════════════════════════════════════════
# 核心 3 · 机动积分（🔴 生产=官方 usv_dynamics.step·按执行口径 10s 边界限速·F1/F2）
#   在每个 10s 决策窗内以细步 h 子积分（官方 step·clip_velocity=False 不在窗内截），
#   只在 10s 边界把 v 截回 [0,v_max]（=usv_env 真实执行）。→ 得执行口径的真轨迹 + 细采样。
# ════════════════════════════════════════════════════════════════════════════════════════
def integrate_maneuver_official(ego0, segments, T, h=0.5, p=None):
    """官方动力学积分（F1/F2 口径）。返回 (ts, traj[N,4])。仅生产（需 vesselmodels）。"""
    assert _HAVE_OFFICIAL, "integrate_maneuver_official 需官方 vesselmodels（本机缺→用 --selftest 验逻辑）"
    if p is None:
        p = _dyn.make_vessel_params(V_MAX)
    n = int(round(T / h))
    x = np.asarray(ego0, float).copy()
    ts = [0.0]
    out = [x.copy()]
    for i in range(n):
        seg = _seg_at(segments, i * h, T)                        # 该时刻生效的 (a,ω)
        x = _dyn.step(x, (seg[0], seg[1]), h, p, clip_velocity=False)   # 窗内不截（忠实执行）
        t = (i + 1) * h
        if abs(t / DECISION_DT - round(t / DECISION_DT)) < 1e-9:  # 到 10s 边界 → 截 v（=usv_env clip_velocity=True）
            x[3] = float(np.clip(x[3], 0.0, V_MAX))
        ts.append(t)
        out.append(x.copy())
    return np.array(ts), np.array(out)


def _seg_at(segments, t, T):
    """两段/多段 open-loop：返回 t 时刻生效的 (a,ω)。dur=None → 吃到 T。"""
    acc = 0.0
    for a, w, dur in segments:
        d = (T - acc) if dur is None else dur
        if t < acc + d + 1e-9:
            return (a, w)
        acc += d
    return (segments[-1][0], segments[-1][1])


# ════════════════════════════════════════════════════════════════════════════════════════
# 分类：单个 ρ5 态 → {unavoidable / avoidable / undecided}
# ════════════════════════════════════════════════════════════════════════════════════════
def classify_state(ego, obs, obs_len, obs_wid, T=120.0, h=0.5, two_segment=False, p=None):
    """ego/obs = (px,py,θ,v)。返回 dict(part, gap1_tstar, cleared_by, min_lb_best, dist)。"""
    dist = float(np.hypot(ego[0] - obs[0], ego[1] - obs[1]))
    # ① gap#1 不可避（官方证书·F3/A7 传真他船宽 obs_wid≤真宽=SR108 精确）
    if _HAVE_OFFICIAL:
        vp = p if p is not None else _dyn.make_vessel_params(V_MAX)
        ego_vs = VesselState(position=np.array(ego[:2], float), orientation=float(ego[2]),
                             velocity=float(ego[3]), length=L_SHIP)
        obs_vs = VesselState(position=np.array(obs[:2], float), orientation=float(obs[2]),
                             velocity=float(obs[3]), length=obs_len)
        unavoid, tstar = imminent_unavoidable_certificate(ego_vs, obs_vs, vp, obs_width=obs_wid,
                                                          t_horizon=T)
        if unavoid:
            return dict(part="unavoidable", gap1_tstar=tstar, cleared_by=None, min_lb_best=None, dist=dist)
    else:
        unavoid, tstar = False, None                            # selftest 不判 gap#1（无官方）

    # ② 机动族 ∃ 清障 = 可避
    best_lb = -np.inf
    cleared_by = None
    for name, segs in maneuver_family(two_segment):
        ts, traj = integrate_maneuver_official(ego, segs, T, h, p)
        res = clearance_lower_bound(ts, traj, obs, obs_len, obs_wid, h)
        if res["min_lb"] > best_lb:
            best_lb = res["min_lb"]
        if res["clears"]:
            cleared_by = name
            return dict(part="avoidable", gap1_tstar=tstar, cleared_by=name,
                        min_lb_best=res["min_lb"], dist=dist)
    # ③ 都不 = 未决
    return dict(part="undecided", gap1_tstar=tstar, cleared_by=None, min_lb_best=float(best_lb), dist=dist)


# ════════════════════════════════════════════════════════════════════════════════════════
# 阶段 A · 收集（服务器·rollout 真基准·收 ρ5 态）——镜像 closed_loop_dock.py 的经验管线
# ════════════════════════════════════════════════════════════════════════════════════════
def phase_collect():
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

    if MANIFEST:                                                 # 真 40 测试集（同 run_step4e 拆分·数字可比）
        from run_step4e import load_manifest_split
        bdir = os.environ.get("STEP4E_BALANCED_DIR") or os.path.dirname(os.path.abspath(MANIFEST))
        _tr, test_paths, _info = load_manifest_split(MANIFEST, bdir)
        scn = test_paths
        src = f"manifest 测试集(n={len(scn)})"
    else:
        scn = sorted(_glob.glob(os.environ.get("SCN_GLOB", "/tmp/trb_scenarios_pool/T-*.xml")))
        src = f"⚠️glob sanity(n={len(scn)}·非标准集)"
    pool = load_scenario_pool(scn)
    print(f"[collect] 场景源={src} 种子={SEEDS} → {OUT}", flush=True)

    n_rho5 = 0
    with open(OUT, "w") as fo:
        for s in SEEDS:
            ck = os.path.join(CKPT_DIR, CKPT_TMPL.format(s=s))
            if not (os.path.exists(ck + ".zip") and os.path.exists(ck + "_vecnorm.pkl")):
                print(f"  s{s}: 缺 checkpoint → 跳过", flush=True)
                continue
            bv = DummyVecEnv([lambda: ContinuousProjectionEnv(pool[0][0], pool[0][1], shield=True)])
            vn = VecNormalize.load(ck + "_vecnorm.pkl", bv)
            vn.training = False
            if int(np.asarray(vn.obs_rms.mean).shape[0]) != int(bv.observation_space.shape[0]):
                raise SystemExit(f"s{s}: vecnorm 维≠env 维（盾/augment 配置不匹配 checkpoint）")
            tf = make_obs_transform(vn)
            model = PPO.load(ck + ".zip", device="cpu")
            for sc, pp in pool:
                env = ContinuousProjectionEnv(sc, pp, shield=True)
                obs, info = env.reset(seed=0)
                for _ in range(200):
                    act, _ = model.predict(tf(obs), deterministic=True)
                    # 🔴 时序：盾的 _rho 在 env.step 里才更新 → step 前读是【上一步的 stale 值】。
                    #    正解=先抓【本步 pre-step 状态】(=盾这步据以分类的 ego/obs)，再 step，
                    #    再用 info["rho"]（=盾这步实际动作的 ρ）判是否 ρ5，记 pre-step 状态。
                    ev, ov = env._ego_vs(), env._obs_vs()
                    ob = env._obstacles[0] if env._obstacles else None
                    owid = float(getattr(ob.obstacle_shape, "width", W_SHIP)) if ob is not None else W_SHIP
                    olen = float(env._obs_length)
                    obs, _r, term, trunc, info = env.step(np.asarray(act, float))
                    if info.get("rho") == RHO_EMERGENCY and ov is not None:
                        rec = dict(seed=s, scn=os.path.basename(str(sc)),
                                   ego=[float(ev.position[0]), float(ev.position[1]),
                                        float(ev.orientation), float(ev.velocity)],
                                   obs=[float(ov.position[0]), float(ov.position[1]),
                                        float(ov.orientation), float(ov.velocity)],
                                   obs_len=olen, obs_wid=owid, source=info.get("source"))
                        fo.write(json.dumps(rec) + "\n")
                        n_rho5 += 1
                    if term or trunc:
                        break
            print(f"  s{s}: 累计 ρ5 态 {n_rho5}", flush=True)
    print(f"[collect] done · ρ5 态总数={n_rho5} → {OUT}", flush=True)


# ════════════════════════════════════════════════════════════════════════════════════════
# 阶段 B · 分类 + 报告
# ════════════════════════════════════════════════════════════════════════════════════════
def phase_classify():
    assert _HAVE_OFFICIAL, f"--classify 需 vesselmodels 跑官方 step（本机缺：{_IMPORT_ERR}）→ 服务器跑"
    INP = os.environ.get("IN_JSONL", "block3_rho5_states.jsonl")
    T = float(os.environ.get("PROBE_T", "120"))
    h = float(os.environ.get("PROBE_H", "0.5"))
    two = os.environ.get("PROBE_TWOSEG", "0") == "1"
    recs = [json.loads(l) for l in open(INP)]
    print(f"[classify] {len(recs)} 个 ρ5 态 · T={T} h={h} two_segment={two}", flush=True)
    p = _dyn.make_vessel_params(V_MAX)
    counts = {"unavoidable": 0, "avoidable": 0, "undecided": 0}
    by_clearer = {}
    gap1_fire = 0
    undecided_dists = []
    # 距离分箱：暴露「近距子区间 vs 宽 gap#1-silent 带」（复审 flag：>15% 是近距数别当全体数）
    bins = [(0, 400), (400, 780), (780, 1e9)]
    bin_tab = {b: {"unavoidable": 0, "avoidable": 0, "undecided": 0} for b in bins}
    for i, r in enumerate(recs):
        res = classify_state(r["ego"], r["obs"], r["obs_len"], r["obs_wid"], T=T, h=h, two_segment=two, p=p)
        counts[res["part"]] += 1
        if res["gap1_tstar"] is not None:
            gap1_fire += 1
        if res["cleared_by"]:
            by_clearer[res["cleared_by"]] = by_clearer.get(res["cleared_by"], 0) + 1
        if res["part"] == "undecided":
            undecided_dists.append(res["dist"])
        for b in bins:
            if b[0] <= res["dist"] < b[1]:
                bin_tab[b][res["part"]] += 1
        if (i + 1) % 200 == 0:
            print(f"  ...{i+1}/{len(recs)}", flush=True)
    N = max(1, len(recs))
    print("\n===== block3 三分区（全 ρ5 population）=====")
    for k in ("unavoidable", "avoidable", "undecided"):
        print(f"  {k:>12}: {counts[k]:>5}  = {100.0*counts[k]/N:5.1f}%")
    print(f"  gap#1 触发率 = {100.0*gap1_fire/N:.2f}% of ρ5   （复审：A-ii 有界严重度打在这个集上·近空则须砍/hedge）")
    print("\n----- 按 ego-他船距离分箱（未决是否集中在近距）-----")
    for b in bins:
        tot = sum(bin_tab[b].values())
        if tot:
            print(f"  [{b[0]:>4}-{b[1] if b[1]<1e9 else '∞':>4}]m n={tot:>5}: "
                  f"unavoid {100*bin_tab[b]['unavoidable']/tot:4.1f}% / "
                  f"avoid {100*bin_tab[b]['avoidable']/tot:4.1f}% / "
                  f"undec {100*bin_tab[b]['undecided']/tot:4.1f}%")
    if undecided_dists:
        print(f"\n  未决态距离：中位 {np.median(undecided_dists):.0f}m  范围 [{min(undecided_dists):.0f},{max(undecided_dists):.0f}]m")
    print("\n----- 可避态由哪个机动清障（族覆盖诊断）-----")
    for k, v in sorted(by_clearer.items(), key=lambda x: -x[1]):
        print(f"  {k:>22}: {v}")
    print("\n🔴 go/no-go：未决区小(且不集中在多障碍/迫近末端) → 方向A有底；大 → 机动族扩/退 demonstration。")
    print("   复审提醒：真命门可能不是 block3 而是 block2 执行接管(L192-I OOD 反噬) + 多障碍不合成——本探针只量单障碍 CV。")


# ════════════════════════════════════════════════════════════════════════════════════════
# --selftest：本机用【已验证 RK4 自包含积分】喂同一 clearance_lower_bound → 复现 600 对抗 0 假认证
#   证明「判据逻辑」正确（不依赖 vesselmodels）。生产 integrate_maneuver_official 与它共用判据。
# ════════════════════════════════════════════════════════════════════════════════════════
def _rk4_selfcontained(ego0, a, w, T, dt=0.05):
    """自包含 RK4（=clearance_certificate_selfcheck 同款·已核 RHS=官方 f=[v cosθ,v sinθ,ω,a]·每 dt 连续截 v）。"""
    a = float(np.clip(a, -A_MAX, A_MAX)); w = float(np.clip(w, -W_MAX, W_MAX))
    def rhs(x):
        v = x[3]
        dv = 0.0 if ((v <= 0.0 and a < 0.0) or (v >= V_MAX and a > 0.0)) else a
        return np.array([v * math.cos(x[2]), v * math.sin(x[2]), w, dv])
    n = int(round(T / dt)); x = np.asarray(ego0, float).copy(); out = [x.copy()]
    for _ in range(n):
        k1 = rhs(x); k2 = rhs(x + 0.5*dt*k1); k3 = rhs(x + 0.5*dt*k2); k4 = rhs(x + dt*k3)
        x = x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        x[3] = min(max(x[3], 0.0), V_MAX); out.append(x.copy())
    return np.arange(n+1)*dt, np.array(out)


def phase_selftest():
    print("=== --selftest：判据逻辑本机验证（RK4 自包含·不依赖官方 vesselmodels）===")
    rng = np.random.default_rng(20260721)
    N = 600; nc = nf = 0; mg = np.inf
    for _ in range(N):
        e = [0, 0, rng.uniform(-np.pi, np.pi), rng.uniform(0, V_MAX)]
        ang = rng.uniform(-np.pi, np.pi); dd = rng.uniform(200, 1200)
        o = [dd*np.cos(ang), dd*np.sin(ang), rng.uniform(-np.pi, np.pi), rng.uniform(0, V_MAX)]
        a = rng.choice([-A_MAX, 0, A_MAX]); w = rng.choice([-W_MAX, 0, W_MAX]); h = rng.choice([0.25, 0.5, 1.0])
        ts, traj = _rk4_selfcontained(e, a, w, 60.0)
        res = clearance_lower_bound(ts, traj, o, L_SHIP, W_SHIP, h)
        if res["clears"]:
            nc += 1
            # 真值（细网格）：0.2s stride 采样最小体距
            v = float(o[3]); om = np.array([math.cos(o[2]), math.sin(o[2])])
            tmin = min(_ego_rect(traj[k][:2], traj[k][2], L_SHIP, W_SHIP).distance(
                       _ego_rect((o[0]+v*om[0]*ts[k], o[1]+v*om[1]*ts[k]), o[2], L_SHIP, W_SHIP))
                       for k in range(0, len(ts), 4))
            if tmin <= 0.0:
                nf += 1
            mg = min(mg, tmin - res["min_lb"])
    print(f"  600 配置：clears={nc} · 假认证={nf} · 下界 min_gap={mg:.3f}m")
    ok = (nf == 0 and mg >= -1e-6)
    print("  " + ("✅ 判据逻辑 SOUND（生产用官方 step 喂同一判据）" if ok else f"🔴 UNSOUND nf={nf} mg={mg}"))
    # per-subinterval 局部 L 比全局 11.9 更紧的额外自证：L 用实测速 → 满减速机动 margin 明显更小
    ts, traj = _rk4_selfcontained([0, 0, 0, V_MAX], -A_MAX, 0.0, 60.0)
    print(f"  [紧度自证] 满减速机动末速={traj[-1,3]:.2f}（v→0）→ per-subinterval L 尾段显著<全局，margin 更紧。")
    return 0 if ok else 1


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "--selftest"
    if mode == "--collect":
        phase_collect()
    elif mode == "--classify":
        phase_classify()
    elif mode == "--selftest":
        sys.exit(phase_selftest())
    else:
        print(__doc__)
        print("用法: python block3_partition_probe.py [--collect | --classify | --selftest]")
        sys.exit(2)
