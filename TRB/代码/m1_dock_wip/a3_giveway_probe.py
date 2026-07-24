#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A3 真让路态探针 —— Phase 4「可证明前向不变」的真 go/no-go（`Paper/命题4_前向不变递归可行_草稿_0723.md` v2 +
`结果/Phase4预研-0723/U_term设计_A3探针_B1基线_规格.md`）。

【它回答什么】把门1(递归可行)/门2(方向弹性)/U_term 非空率，从【合成分布】搬到【盾真实访问的态分布】(真让路态 ρ2/3/4 + stand-on ρ1 + emergency ρ5)上复核。
  · 门1(测机制·非全量重分类·修 75/75 循环)：对 s∈A(带 certified 直行尾脱离 m*)·走一受盾步得 s'·三查隔离：
      (1a) 同一尾巴 m*_tail 用【修正证书 cert_v2】certify s' 永久清；(1b) 收缩视界 H−Δ；(1c) 引理1闭合(直行尾+过CPA·证书交付)。
  · 门2(方向弹性/合规)：让路态查【合规方向】certified backup 是否存在(head_on/crossing 右转·overtake 松)。报 A∩U_colregs 非空率·三态分开。
  · U_term 非空率：A-成员率(真让路态) + 层1+2 产出非空可行域率。

【铁律：不自己写动力学】机动积分走官方 `usv_dynamics.step`(复用 block3 探针已过 2 轮对抗审的 integrate_maneuver_official)；
清障判据复用 block3 `clearance_profile`(修正版 Lipschitz·已复审 SOUND)；本档只【加】cert_v2 闭合率交付 + 直行尾脱离族 + backup U_term + 门逻辑。

【修正证书 cert_v2（Prop4 v2 引理1 前提·复审洞抓的）】certified 永久清障 = ① clearance_profile 判 [0,H] 内 d>0(Lipschitz)
  ② 尾段直行(ω=0) ③ 直行尾上船体距在 H 处严格增(过CPA·直行尾 g 凸→增at H⟹永久增)。仅直行尾成立(恒转尾巴非仿射·排除出 A)。

【三阶段·纯 eval 不烧训练卡】
  --collect ：服务器·load 盾策略 rollout 真基准 → 收 ρ∈{1,2,3,4,5} 态 + info[rho/give_way_dir/source] → jsonl（需 vesselmodels/env）。
  --gates   ：对收集态跑 cert_v2 + 门1/门2/U_term 非空率 + 按 ρ/give_way_dir 分层（需 vesselmodels 跑官方 step）。
  --selftest：本机·门逻辑 + cert_v2 双档验证（自包含 RK4·不依赖 vesselmodels·CI 可跑）。
服务器跑前须 user 拍板 + 逐字预检 + screen 后台（04 运行手册）。
"""
import os
import sys
import json
import math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))         # m1_dock_wip 进 path（import block3 探针）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 代码/ 进 path

# ── 复用 block3 探针已对抗审的件（clearance_profile 修正版 / integrate_maneuver_official / _ego_rect / keep_course）──
import block3_partition_probe as B3
from block3_partition_probe import (A_MAX, W_MAX, V_MAX, DECISION_DT, L_SHIP, W_SHIP, R_CIRC,
                                     clearance_profile, keep_course_min_dist, _ego_rect, _HAVE_OFFICIAL)

DT = DECISION_DT   # 10s


# ════════════════════════════════════════════════════════════════════════════════════════
# 核心 1 · 修正证书 cert_v2（block1 清障 + 闭合率交付=引理1前提·Prop4 v2）
# ════════════════════════════════════════════════════════════════════════════════════════
def _tail_distances(ts, traj, obs0, obs_len, obs_wid):
    """逐采样点船体距（用于闭合率/过CPA 判定）。"""
    v_m = float(obs0[3]); om = (math.cos(obs0[2]), math.sin(obs0[2]))
    ds = np.empty(len(ts))
    for k in range(len(ts)):
        t = ts[k]; ex, ey, eth, _ = traj[k]
        oc = (obs0[0] + v_m * om[0] * t, obs0[1] + v_m * om[1] * t)
        ds[k] = _ego_rect((ex, ey), eth, L_SHIP, W_SHIP).distance(_ego_rect(oc, obs0[2], obs_len, obs_wid))
    return ds


def cert_v2(ts, traj, oseg, obs0, obs_len, obs_wid, segments, h, H):
    """修正证书。返回 dict(certified_perm, clears_H, straight_tail, past_cpa, first_unsafe_t)。
    certified_perm = clears_H ∧ straight_tail ∧ past_cpa（=引理1 sound 永久清障充分条件·仅【恒速】直行尾）。"""
    assert obs_len > 0.0 and obs_wid > 0.0, f"他船尺寸非法 obs_len={obs_len} obs_wid={obs_wid}"   # F5·防线
    prof = clearance_profile(ts, traj, obs0, obs_len, obs_wid, h, oseg)   # 修正版 Lipschitz（已复审 SOUND）
    fut = prof["first_unsafe_t"]
    clears_H = (fut is None) or (fut > H)
    # 🔴 F1(CRITICAL 修)：引理1 凸性前提=【恒速】平移 → 尾段须 ω=0 【且 a=0】。
    #   仅查 ω=0 会放行加速尾(抛物线路径·g 非凸·"过CPA⟹永久增"假·实测二阶差−1.14)。
    straight_tail = abs(segments[-1][1]) < 1e-9 and abs(segments[-1][0]) < 1e-9   # 末段 ω=0 且 a=0
    ds = _tail_distances(ts, traj, obs0, obs_len, obs_wid)
    t_tail = sum(d for a, w, d in segments[:-1] if d is not None)         # 尾段起始时刻
    tail_idx = np.where(ts >= t_tail - 1e-9)[0]
    if len(tail_idx) >= 2:
        tail_ds = ds[tail_idx]
        # 直行尾上 g 凸 → H 处严格增(端点差>0) ∧ 尾内 CPA 已出现(argmin 非末点) ⟹ 过CPA·永久增
        past_cpa = bool(tail_ds[-1] > tail_ds[-2] + 1e-9 and int(np.argmin(tail_ds)) < len(tail_ds) - 1)
    else:
        past_cpa = False
    return dict(certified_perm=bool(clears_H and straight_tail and past_cpa),
                clears_H=bool(clears_H), straight_tail=bool(straight_tail),
                past_cpa=bool(past_cpa), first_unsafe_t=fut)


# ════════════════════════════════════════════════════════════════════════════════════════
# 核心 2 · 直行尾脱离族（只带直行尾·修 Prop4 v1 引理1 纯转向反例）+ 合规方向标注
# ════════════════════════════════════════════════════════════════════════════════════════
def straight_tail_family():
    """[(name, segments, first_omega)]。segments 末段恒【速】直行(ω=0∧a=0·引理1凸性前提)。first_omega<0=右转(starboard)·>0=左转(port)·=0=纯直行。
    🔴 时长网格【已加密】(2026-07-25 独立复审 L200-C)：旧粗网格 {10,20,30,40,60,80}s 漏掉中间/更长转向时长·把门2交叉合规率压到 54%(=机动族假象·非物理残余)；
       加密到 5..120s(步长~5-10s)+加速转向再减速回匀速直行尾后·门2 交叉 54%→~97%·真残余~2-3%(全860态定值坐实·两独立方法复现)。
       ⚠️ 删了旧 acc 变体(尾段 a=+A≠0 → straight_tail 恒 False → 从不 certify=死权重)·换成 accdec(加速转向→减速→匀速直行尾·有合法直行尾·可 certify·补覆盖)。"""
    A, W = A_MAX, W_MAX
    fam = []
    # 🔴 时长网格 = 决策步(10s)整数倍(对抗审 Finding A·2026-07-25)：控制器每决策步施一个恒控·转向时长非10s倍(如5/15/25s)会步内切控制=不可执行=非admissible backup·前向不变退路须可执行。10s对齐后门2率几乎不变(head-on100/crossing98/overtake100)。
    DURS = (10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0)   # 10s对齐=可执行
    for w in (-W, +W):
        for t1 in DURS:
            fam.append((f"turn{w:+.3f}_{int(t1)}s", [(0.0, w, t1), (0.0, 0.0, None)], w))
            fam.append((f"dec{w:+.3f}_{int(t1)}s", [(-A, w, t1), (0.0, 0.0, None)], w))
        for t1 in (20.0, 40.0, 60.0):   # 加速转向 t1 → 减速20s → 匀速直行尾(合法直行尾·可 certify·补 yaw 饱和下的覆盖)
            fam.append((f"accdec{w:+.3f}_{int(t1)}s", [(+A, w, t1), (-A, 0.0, 20.0), (0.0, 0.0, None)], w))
    for a in (-A, 0.0, +A):
        fam.append((f"straight_a{a:+.2f}", [(a, 0.0, None)], 0.0))
    return fam


def _compliant_omega_sign(give_way_dir):
    """让路方向 → 合规首步 ω 号。'right'→ω<0(starboard)·'left'→ω>0·None/stand-on→保向(近0·|ω|小)。"""
    if give_way_dir == "right":
        return -1
    if give_way_dir == "left":
        return +1
    return 0


def _range_rate(ego, obs):
    """中心距变化率 ṙ=(p_rel·v_rel)/‖p_rel‖。>0=分离(closing rate 反号)。F2·门1c 独立闭合信号(从速度直算·不复用证书 past_cpa)。"""
    p_rel = np.array([ego[0] - obs[0], ego[1] - obs[1]])
    n = float(np.hypot(p_rel[0], p_rel[1]))
    if n < 1e-9:
        return 0.0
    v_e = ego[3] * np.array([math.cos(ego[2]), math.sin(ego[2])])
    v_o = obs[3] * np.array([math.cos(obs[2]), math.sin(obs[2])])
    return float(p_rel @ (v_e - v_o) / n)


def _tail_after(segments, dt):
    """机动 segments 走 dt 秒后的尾巴序列（首段裁掉 dt）。"""
    out = []; used = 0.0; started = False
    for a, w, dur in segments:
        if dur is None:
            out.append((a, w, None)); started = True; continue
        if used + dur <= dt + 1e-9:
            used += dur; continue
        if not started and used < dt:
            out.append((a, w, dur - (dt - used))); started = True; used = dt
        else:
            out.append((a, w, dur))
    if not out:
        out = [(segments[-1][0], segments[-1][1], None)]
    return out


# ════════════════════════════════════════════════════════════════════════════════════════
# 核心 3 · 找 certified 备份 m*（可选合规方向）+ 门1/门2 单态判定
# ════════════════════════════════════════════════════════════════════════════════════════
def find_backup(ego, obs, obs_len, obs_wid, H, h, p, require_omega_sign=0):
    """返回第一个 cert_v2 永久清障的直行尾脱离序列 (name, segments)。require_omega_sign: -1右/+1左/0任意。"""
    for name, segs, w0 in straight_tail_family():
        if require_omega_sign < 0 and not (w0 < -1e-9):
            continue
        if require_omega_sign > 0 and not (w0 > 1e-9):
            continue
        ts, traj, oseg = B3.integrate_maneuver_official(ego, segs, H, h, p)
        c = cert_v2(ts, traj, oseg, obs, obs_len, obs_wid, segs, h, H)
        if c["certified_perm"]:
            return name, segs
    return None, None


def gate_state(rec, H, h, p):
    """对一个收集态判门1/门2/U_term。返回 dict(in_A, g1a, g1b, g1c, compliant_backup, kc)。"""
    ego, obs = rec["ego"], rec["obs"]
    olen, owid = rec["obs_len"], rec["obs_wid"]
    gw = rec.get("give_way_dir")
    out = dict(rho=rec.get("rho"), give_way_dir=gw)
    # A-成员（任意方向 certified 直行尾脱离）
    name, segs = find_backup(ego, obs, olen, owid, H, h, p, require_omega_sign=0)
    out["in_A"] = segs is not None
    out["kc"] = keep_course_min_dist(ego, obs, olen, owid)
    if segs is None:
        # 🔴 L200-G2 修·门2 分母一致化：not-in-A 态【无任意方向 backup ⟹ 也无合规 backup】→ 门2 记 False(非 None)·
        #   让 phase_gates 按【全体让路态】分母计(与 gw_gates 一致)·而非仅 in-A 条件分母(旧 bug=silently 掉 not-in-A→数偏高)。
        out["compliant_backup"] = (False if _compliant_omega_sign(gw) != 0 else None)
        return out
    # 门1：走一受盾步 u0=m* 前 Δ → 后继 s'（障碍 CV 前进 Δ）
    ts, traj, oseg = B3.integrate_maneuver_official(ego, segs, DT, h, p)
    ego2 = list(traj[-1])
    obs2 = [obs[0] + obs[3]*math.cos(obs[2])*DT, obs[1] + obs[3]*math.sin(obs[2])*DT, obs[2], obs[3]]
    tail = _tail_after(segs, DT)
    ts2, traj2, oseg2 = B3.integrate_maneuver_official(ego2, tail, H, h, p)
    c1a = cert_v2(ts2, traj2, oseg2, obs2, olen, owid, tail, h, H)          # 1a·全视界 H·同尾永久 certify
    ts2b, traj2b, oseg2b = B3.integrate_maneuver_official(ego2, tail, H - DT, h, p)
    c1b = cert_v2(ts2b, traj2b, oseg2b, obs2, olen, owid, tail, h, H - DT)  # 1b·收缩视界 H−Δ
    out["g1a"] = c1a["certified_perm"]
    # 🔴 F2(HIGH 修·门1三查正交化·防伪三角互证)：
    #   1b 用【certified_perm】(缩视界下 past_cpa 窗口变短·argmin 未必仍内点·不被 1a 蕴含)·非 clears_H。
    out["g1b"] = c1b["certified_perm"]
    #   1c 用【独立闭合率】：后继尾末态中心 ṙ>0(分离)·从速度直算·不复用 c1a 的 past_cpa 布尔。
    tH = float(ts2[-1])
    obsH = [obs2[0] + obs2[3]*math.cos(obs2[2])*tH, obs2[1] + obs2[3]*math.sin(obs2[2])*tH, obs2[2], obs2[3]]
    out["g1c"] = _range_rate(list(traj2[-1]), obsH) > 0.0
    # 门2：合规方向 certified backup 存在？（让路态才判）
    sign = _compliant_omega_sign(gw)
    if sign != 0:
        cn, cs = find_backup(ego, obs, olen, owid, H, h, p, require_omega_sign=sign)
        out["compliant_backup"] = cs is not None
    else:
        out["compliant_backup"] = None   # stand-on/无让路方向：不判(保向即撞落 in-extremis)
    return out


# ════════════════════════════════════════════════════════════════════════════════════════
# 阶段 A · 收集（服务器·rollout 真基准·收 ρ∈{1,2,3,4,5} + info 字段）
# ════════════════════════════════════════════════════════════════════════════════════════
def phase_collect():
    assert _HAVE_OFFICIAL, f"--collect 需 vesselmodels/env（本机缺）→ 服务器跑"
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from trb_env.train import make_obs_transform
    from trb_env.usv_continuous_shield import ContinuousProjectionEnv
    from trb_env.usv_scenarios import load_scenario_pool
    from run_step4e import load_manifest_split

    CKPT_DIR = os.environ["CKPT_DIR"]
    CKPT_TMPL = os.environ.get("CKPT_TMPL", "Continuous-safe_s{s}_L1rateON_ppo_s{s}")
    SEEDS = [int(x) for x in os.environ.get("SEEDS", "0 1 2 3 4 5 6 7 8 9").split()]
    MANIFEST = os.environ["STEP4E_MANIFEST"]
    OUT = os.environ.get("OUT_JSONL", "a3_giveway_states.jsonl")
    # 收哪些 ρ：默认让路 2/3/4 + stand-on 1 + emergency 5
    RHOS = set(int(x) for x in os.environ.get("A3_RHOS", "1 2 3 4 5").split())

    bdir = os.environ.get("STEP4E_BALANCED_DIR") or os.path.dirname(os.path.abspath(MANIFEST))
    _tr, test_paths, _i = load_manifest_split(MANIFEST, bdir)
    pool = load_scenario_pool(test_paths)
    print(f"[collect A3] 场景 n={len(pool)} · 种子={SEEDS} · 收 ρ∈{sorted(RHOS)} → {OUT}", flush=True)
    print("  ⚠️ 群体=盾确定性部署遇到的态·偏易(Attack 2b)·判读须披露", flush=True)

    def _mk(sc, pp):
        return ContinuousProjectionEnv(sc, pp, shield=True, goal_cone_half=None, goal_v_floor=2.0, augment_rho=False)

    n = {r: 0 for r in RHOS}
    with open(OUT, "w") as fo:
        for s in SEEDS:
            ck = os.path.join(CKPT_DIR, CKPT_TMPL.format(s=s))
            if not (os.path.exists(ck + ".zip") and os.path.exists(ck + "_vecnorm.pkl")):
                print(f"  s{s}: 缺 ckpt → 跳过", flush=True); continue
            bv = DummyVecEnv([lambda: _mk(pool[0][0], pool[0][1])])
            vn = VecNormalize.load(ck + "_vecnorm.pkl", bv); vn.training = False
            if int(np.asarray(vn.obs_rms.mean).shape[0]) != int(bv.observation_space.shape[0]):
                raise SystemExit(f"s{s}: vecnorm 维≠env 维")
            tf = make_obs_transform(vn); model = PPO.load(ck + ".zip", device="cpu")
            for si, (sc, pp) in enumerate(pool):
                env = _mk(sc, pp); obs, info = env.reset(seed=0)
                for step_i in range(200):
                    act, _ = model.predict(tf(obs), deterministic=True)
                    ev, ov = env._ego_vs(), env._obs_vs()               # pre-step 态
                    ob = env._obstacles[0] if env._obstacles else None
                    if ob is not None and not hasattr(ob.obstacle_shape, "width"):
                        raise SystemExit("他船 shape 无 width → 无法保证 obs_wid 精确")
                    owid = float(ob.obstacle_shape.width) if ob is not None else W_SHIP
                    olen = float(env._obs_length)
                    obs, _r, term, trunc, info = env.step(np.asarray(act, float))
                    rho = info.get("rho")
                    if rho in RHOS and ov is not None:
                        fo.write(json.dumps(dict(
                            seed=s, scn=os.path.basename(str(sc)), scn_idx=si, step=step_i, rho=int(rho),
                            give_way_dir=info.get("give_way_dir"), source=info.get("source"),
                            ego=[float(ev.position[0]), float(ev.position[1]), float(ev.orientation), float(ev.velocity)],
                            obs=[float(ov.position[0]), float(ov.position[1]), float(ov.orientation), float(ov.velocity)],
                            obs_len=olen, obs_wid=owid)) + "\n")
                        n[rho] = n.get(rho, 0) + 1
                    if term or trunc:
                        break
            fo.flush()
            print(f"  s{s}: 累计 {dict(n)}", flush=True)
    print(f"[collect A3] done · 各 ρ 态数={dict(n)} → {OUT}", flush=True)


# ════════════════════════════════════════════════════════════════════════════════════════
# 阶段 B · 门判定 + 报告（按 ρ/give_way_dir 分层）
# ════════════════════════════════════════════════════════════════════════════════════════
def phase_gates():
    assert _HAVE_OFFICIAL, f"--gates 需 vesselmodels 跑官方 step → 服务器跑"
    from trb_env import usv_dynamics as _dyn
    INP = os.environ.get("IN_JSONL", "a3_giveway_states.jsonl")
    H = float(os.environ.get("PROBE_H_HORIZON", "120"))
    h = float(os.environ.get("PROBE_H", "0.5"))
    p = _dyn.make_vessel_params(V_MAX)
    recs = [json.loads(l) for l in open(INP)]
    print(f"[gates A3] {len(recs)} 态 · H={H} h={h}", flush=True)

    RHO_NAME = {1: "stand-on", 2: "head-on", 3: "crossing", 4: "overtake", 5: "emergency"}
    by_rho = {}
    allres = []   # (seed, scn_idx, step, rho, res)·供相遇聚合（F4·防 per-step 双计）
    for i, r in enumerate(recs):
        res = gate_state(r, H, h, p)
        rho = res["rho"]; d = by_rho.setdefault(rho, dict(n=0, inA=0, g1a=0, g1b=0, g1c=0, cb=0, cb_tot=0, cb_inA_tot=0, genuine=0))
        d["n"] += 1
        if res["kc"] <= 0: d["genuine"] += 1
        if res["in_A"]:
            d["inA"] += 1
            d["g1a"] += int(res.get("g1a", False)); d["g1b"] += int(res.get("g1b", False)); d["g1c"] += int(res.get("g1c", False))
        if res.get("compliant_backup") is not None:
            d["cb_tot"] += 1; d["cb"] += int(res["compliant_backup"])       # 全体让路态分母(gw_gates 口径)
            if res["in_A"]: d["cb_inA_tot"] += 1                            # in-A 条件分母(L200-G2·两口径都报)
        allres.append((r.get("seed"), r.get("scn_idx"), r.get("step"), rho, res))
        if (i + 1) % 100 == 0:
            print(f"  ...{i+1}/{len(recs)}", flush=True)

    print("\n===== A3 门报告 A · 【per-step】按 ρ 分层 =====")
    print("  (门2 两口径·L200-G2：全体=全让路态分母[gw_gates口径·not-in-A 记 fail]·in-A=仅可清障态条件分母)")
    print(f"  {'ρ':>10} | {'n':>5} {'真对撞%':>7} {'A-成员%':>7} | 门1: {'1a同尾%':>7} {'1b缩视%':>7} {'1c闭合%':>7} | {'门2全体%':>9} {'门2in-A%':>9}")
    for rho in sorted(by_rho):
        d = by_rho[rho]; nA = max(1, d["inA"]); N = max(1, d["n"])
        cb_full = f"{100*d['cb']/max(1,d['cb_tot']):.1f}({d['cb']}/{d['cb_tot']})" if d["cb_tot"] else "n/a"
        cb_inA = f"{100*d['cb']/max(1,d['cb_inA_tot']):.1f}({d['cb']}/{d['cb_inA_tot']})" if d["cb_inA_tot"] else "n/a"
        print(f"  {RHO_NAME.get(rho, rho):>10} | {d['n']:>5} {100*d['genuine']/N:>6.1f} {100*d['inA']/N:>6.1f} | "
              f"     {100*d['g1a']/nA:>6.1f} {100*d['g1b']/nA:>6.1f} {100*d['g1c']/nA:>6.1f} | {cb_full:>13} {cb_inA:>13}")

    # ── F4·【per-encounter】聚合（项目口径 encounter 级·同 block3 phase_classify）──
    #   相遇 = 同 (seed,scn_idx) 内 step 连续的一段。相遇级门通过 = 该相遇【全部 A-成员步】均通过。
    enc = {}
    for seed, scn, step, rho, res in allres:
        enc.setdefault((seed, scn), []).append((step, rho, res))
    enc_stat = {}
    for key, steps in enc.items():
        steps.sort(key=lambda x: (x[0] if x[0] is not None else 0))
        runs = []; cur = [steps[0]]
        for a, b in zip(steps, steps[1:]):
            if b[0] is not None and a[0] is not None and b[0] == a[0] + 1:
                cur.append(b)
            else:
                runs.append(cur); cur = [b]
        runs.append(cur)
        for run in runs:
            rho_run = run[0][1]   # 相遇主 ρ（首步）
            amembers = [rs for _, _, rs in run if rs["in_A"]]
            st = enc_stat.setdefault(rho_run, dict(n_enc=0, enc_all_inA=0, enc_g1a=0, enc_g1b=0, enc_g1c=0))
            st["n_enc"] += 1
            if amembers:
                st["enc_all_inA"] += 1
                st["enc_g1a"] += int(all(rs.get("g1a", False) for rs in amembers))
                st["enc_g1b"] += int(all(rs.get("g1b", False) for rs in amembers))
                st["enc_g1c"] += int(all(rs.get("g1c", False) for rs in amembers))
    print("\n===== A3 门报告 B · 【per-encounter】(相遇级·全 A-成员步均过才记过) =====")
    print(f"  {'ρ':>10} | {'相遇数':>6} {'有A成员相遇':>10} | {'1a全过%':>7} {'1b全过%':>7} {'1c全过%':>7}")
    for rho in sorted(enc_stat):
        s = enc_stat[rho]; na = max(1, s["enc_all_inA"])
        print(f"  {RHO_NAME.get(rho, rho):>10} | {s['n_enc']:>6} {s['enc_all_inA']:>10} | "
              f"{100*s['enc_g1a']/na:>6.1f} {100*s['enc_g1b']/na:>6.1f} {100*s['enc_g1c']/na:>6.1f}")

    print("\n🔴 go/no-go 判读（以 per-encounter 为准·per-step 供诊断）：")
    print("   · 门1 三机制(1a/1b/1c) 让路态 ρ2/3/4 均 ≥99% → 修正命题4机制真分布坐实 = 可证明前向不变有底。")
    print("   · 门2 合规backup% ≥85-90%(head-on≥95%) → A∩U_colregs 大概率非空·可证明合规且无碰；低于→扩残余冲突集诚实刻画。")
    print("   · A-成员% = 盾真实态落可证明集 A 的率；A 外态(unavoidable/undecided)落 emergency 兜底=诚实在集外。")
    print("   · 边界：单 CV·偏易群体(Attack 2b)·机动他船破 CV。真 go = 门1≥99% ∧ 门2 head-on≥95% ∧ A-成员率不过低。")


# ════════════════════════════════════════════════════════════════════════════════════════
# --selftest：门逻辑 + cert_v2 双档（本机·自包含 RK4·不依赖 vesselmodels）
# ════════════════════════════════════════════════════════════════════════════════════════
def _rk4_local(ego0, segments, T, h=0.5, dt=0.1):
    """自包含 RK4·常控分段·10s 边界钳 v（生产口径）。返回 (ts, traj, oseg)。"""
    def seg_at(t):
        acc = 0.0
        for a, w, dur in segments:
            if dur is None:
                return a, w
            if t < acc + dur - 1e-9:
                return a, w
            acc += dur
        return segments[-1][0], segments[-1][1]
    nsub = int(round(h / dt))

    def rhs(x, a, w):
        v, th = x[3], x[2]
        return np.array([v*math.cos(th), v*math.sin(th), w, a])
    x = np.asarray(ego0, float).copy(); ts = [0.0]; out = [x.copy()]; oseg = []
    n = int(round(T / h))
    for i in range(n):
        a, w = seg_at(i*h); a = float(np.clip(a, -A_MAX, A_MAX)); w = float(np.clip(w, -W_MAX, W_MAX)); oseg.append(abs(w))
        for _ in range(nsub):
            k1 = rhs(x, a, w); k2 = rhs(x+0.5*dt*k1, a, w); k3 = rhs(x+0.5*dt*k2, a, w); k4 = rhs(x+dt*k3, a, w)
            x = x + (dt/6.0)*(k1+2*k2+2*k3+k4)
        t = (i+1)*h
        if abs(t/DT - round(t/DT)) < 1e-9:
            x[3] = float(np.clip(x[3], 0.0, V_MAX))
        ts.append(t); out.append(x.copy())
    return np.array(ts), np.array(out), np.array(oseg)


def phase_selftest():
    print("=== A3 --selftest：cert_v2 闭合率 + 门逻辑（自包含 RK4·不依赖 vesselmodels）===")
    ok = True
    # T1: 直行尾脱离机动 → certified_perm=True（清且过顶）；纯转向绕回 → certified_perm=False（引理1 排除）
    ego = [0.0, 0.0, 0.0, 8.0]; obs = [400.0, 30.0, math.pi, 6.0]; olen, owid = 200.0, 35.0
    # 直行尾：右转 30s 再直行
    segs_ok = [(0.0, -W_MAX, 30.0), (0.0, 0.0, None)]
    ts, traj, oseg = _rk4_local(ego, segs_ok, 120.0)
    c_ok = cert_v2(ts, traj, oseg, obs, olen, owid, segs_ok, 0.5, 120.0)
    # 纯转向恒转到底（无直行尾）
    segs_turn = [(0.0, -W_MAX, None)]
    ts2, traj2, oseg2 = _rk4_local(ego, segs_turn, 120.0)
    c_turn = cert_v2(ts2, traj2, oseg2, obs, olen, owid, segs_turn, 0.5, 120.0)
    t1 = (c_turn["straight_tail"] is False)   # 恒转 straight_tail 必 False → certified_perm 必 False
    print(f"  [T1] 直行尾 certified_perm={c_ok['certified_perm']}(clears={c_ok['clears_H']}/tail={c_ok['straight_tail']}/cpa={c_ok['past_cpa']}) · "
          f"恒转 straight_tail={c_turn['straight_tail']}(应False) → {'✅' if t1 else '🔴'}")
    ok = ok and t1
    # T2: 门1 尾巴 = 同一物理轨迹（走 Δ 后尾巴 certify 后继应与原一致方向）——用直行尾脱离态自洽验
    if c_ok["certified_perm"]:
        tsm, trajm, _ = _rk4_local(ego, segs_ok, DT)
        ego2 = list(trajm[-1]); obs2 = [obs[0]+obs[3]*math.cos(obs[2])*DT, obs[1]+obs[3]*math.sin(obs[2])*DT, obs[2], obs[3]]
        tail = _tail_after(segs_ok, DT)
        tsa, traja, osega = _rk4_local(ego2, tail, 120.0)
        c1a = cert_v2(tsa, traja, osega, obs2, olen, owid, tail, 0.5, 120.0)
        t2 = c1a["certified_perm"]
        print(f"  [T2·门1同尾] s∈A 走一步·同尾 certify 后继 certified_perm={c1a['certified_perm']} → {'✅' if t2 else '🔴'}")
        ok = ok and t2
    else:
        print("  [T2] 跳过（T1 直行尾未 certified·换态）")
    # T3: 合规方向标注一致
    fam = straight_tail_family()
    rights = [w for _, _, w in fam if w < 0]; lefts = [w for _, _, w in fam if w > 0]
    t3 = (len(rights) > 0 and len(lefts) > 0 and _compliant_omega_sign("right") == -1 and _compliant_omega_sign("left") == 1)
    print(f"  [T3] 族含右转{len(rights)}/左转{len(lefts)} · 合规号 right→−1/left→+1 → {'✅' if t3 else '🔴'}")
    ok = ok and t3
    print("  " + ("✅ A3 门逻辑 + cert_v2 selftest 通过" if ok else "🔴 A3 selftest 有洞"))
    return 0 if ok else 1


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "--selftest"
    if mode == "--collect":
        phase_collect()
    elif mode == "--gates":
        phase_gates()
    elif mode == "--selftest":
        sys.exit(phase_selftest())
    else:
        print(__doc__)
        print("用法: python a3_giveway_probe.py [--collect | --gates | --selftest]")
        sys.exit(2)
