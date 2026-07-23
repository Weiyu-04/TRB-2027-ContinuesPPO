#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""B1 · CBF-QP 海事避碰外部基线（`结果/Phase4预研-0723/U_term设计_A3探针_B1基线_规格.md`）。
兑现 user "别假设去测"：实测经典 CBF-QP 在 10s ZOH 下是不是真 0 碰撞·差异化对照本文 Prop4 前向不变盾。

【实现】标准海事 CBF-QP 动作滤波：min‖u−u_nom‖² s.t. HOCBF 约束 + u∈U_box。h=‖p_rel‖²−d_safe²(相对度2·HOCBF)。
  QP 极小(2变量+1线性约束+箱)→解析投影(不依赖 cvxpy/osqp)。
  两档消融(规格要求)：
    · 'plain'   = 纯各向同性距离 HOCBF（**正对遇退化**：p_rel 沿艏向时转向横向系数=0→只会减速不会转→10s步大船躲不掉·实测非0碰撞）。
    · 'colregs' = HOCBF + COLREGs-aware 标称(让路态标称偏右转 starboard)→修退化。
  ⚠️ **诚实**：真论文须引一篇【已发表】COLREGs-CBF 公式(D28: KAIST 2504.19247 / Patil 2603.02484·非对称/旋转 barrier)并复验；
     本档是【代表性实现】·用于"实测 CBF 非自动 0 碰撞"+差异化·**绝不 claim 数量碾压**(朴素退化=打稻草人·禁)。

【测碰撞·别假设】CBF 前向不变只在连续时间+精确模型+安全集真控制不变下成立·本设定破三条(10s ZOH/QP不可行fallback/yaw饱和非线性)
  → 测【裸船体】碰撞(非 d_safe)·【步内细积分】复检最小距·报 QP 不可行率·**报碰撞率带口径·绝不 claim 0**。

【阶段】--synth(本机·合成对撞几何消融·不需 vesselmodels) / --run(服务器·真基准·u_nom=RL 策略动作·同 harness 对照·需 env) / --selftest(本机)。
服务器跑前 user 拍板 + 逐字预检 + screen。
"""
import os
import sys
import json
import math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import block3_partition_probe as B3
from block3_partition_probe import A_MAX, W_MAX, V_MAX, DECISION_DT, L_SHIP, W_SHIP, _ego_rect, _HAVE_OFFICIAL

DT = DECISION_DT


def _circum(l, w):
    return 0.5 * math.hypot(l, w)


# ── QP：min‖u−u_nom‖² s.t. g·u≤b, u∈box(2D·解析)──────────────────────────────────────
def qp_project(u_nom, g, b, box):
    a_lo, a_hi, w_lo, w_hi = box
    un = np.clip(np.asarray(u_nom, float), [a_lo, w_lo], [a_hi, w_hi])
    g = np.asarray(g, float)
    if float(g @ un) <= b + 1e-9:
        return un, True
    gn = float(g @ g)
    if gn < 1e-18:
        return un, (b >= 0.0)
    u0 = b * g / gn
    d = np.array([-g[1], g[0]]) / math.sqrt(gn)
    ts = []
    for i, (lo, hi) in enumerate([(a_lo, a_hi), (w_lo, w_hi)]):
        if abs(d[i]) > 1e-12:
            ts.append(((lo - u0[i]) / d[i], (hi - u0[i]) / d[i]))
    if not ts:
        u = u0
        return u, (a_lo <= u[0] <= a_hi and w_lo <= u[1] <= w_hi)
    tmin = max(min(t) for t in ts); tmax = min(max(t) for t in ts)
    if tmin > tmax + 1e-9:
        return un, False   # box∩halfplane 空 = QP 不可行
    tstar = float(np.clip((np.asarray(u_nom, float) - u0) @ d, tmin, tmax))
    u = u0 + tstar * d
    return np.clip(u, [a_lo, w_lo], [a_hi, w_hi]), True


def hocbf_constraint(ego, obs, d_safe, a1, a2):
    """h=‖p_rel‖²−d_safe²·HOCBF ḧ+(a1+a2)ḣ+a1a2 h≥0 → g·u≤b（对 u=[a,ω] 线性）。"""
    px, py, th, v = ego; ox, oy, oth, ov = obs
    p_rel = np.array([px - ox, py - oy])
    v_ego = v * np.array([math.cos(th), math.sin(th)])
    v_obs = ov * np.array([math.cos(oth), math.sin(oth)])
    v_rel = v_ego - v_obs
    h = float(p_rel @ p_rel) - d_safe**2
    hd = 2.0 * float(p_rel @ v_rel)
    A_coef = 2.0 * float(p_rel @ np.array([math.cos(th), math.sin(th)]))
    W_coef = 2.0 * float(p_rel @ np.array([-math.sin(th), math.cos(th)])) * v
    const = 2.0 * float(v_rel @ v_rel) + (a1 + a2) * hd + a1 * a2 * h
    return np.array([-A_coef, -W_coef]), const


def relbearing(ego, obs):
    """他船相对本船艏向的方位 β（右舷 starboard 负·左舷 port 正·与 usv_colregs 反号约定，本档内部自洽即可）。"""
    los = math.atan2(obs[1] - ego[1], obs[0] - ego[0])
    b = los - ego[2]
    return (b + math.pi) % (2 * math.pi) - math.pi


def colregs_nominal(ego, obs, goal, variant):
    """标称控制。plain=直奔目标；colregs=让路态(他船前方扇区+闭合+近场)偏右转 starboard(ω<0)修正对遇退化。
    ⚠️ 代表性实现·真论文引已发表 COLREGs-CBF·非本 nominal 偏置。"""
    brg = math.atan2(goal[1] - ego[1], goal[0] - ego[0]) - ego[2]
    brg = (brg + math.pi) % (2 * math.pi) - math.pi
    w_nom = float(np.clip(brg / 10.0, -W_MAX, W_MAX))
    a_nom = float(np.clip((V_MAX - ego[3]) / 10.0, -A_MAX, A_MAX))
    if variant == "colregs":
        beta = relbearing(ego, obs)                       # 他船方位
        rng = math.hypot(ego[0] - obs[0], ego[1] - obs[1])
        p_rel = np.array([obs[0] - ego[0], obs[1] - ego[1]])
        v_rel = ego[3]*np.array([math.cos(ego[2]), math.sin(ego[2])]) - obs[3]*np.array([math.cos(obs[2]), math.sin(obs[2])])
        closing = float(p_rel @ v_rel) > 0.0              # 接近中
        # 让路态：他船在前方±60°扇区 ∧ 闭合 ∧ 近场 → 偏右转(ω<0=starboard)
        if abs(beta) < math.radians(60) and closing and rng < 2500.0:
            w_nom = -W_MAX                                 # 满右转 nominal（CBF 仍保碰撞安全）
    return np.array([a_nom, w_nom])


def step_ego(ego, u, T=DT, dt=0.1):
    a, w = float(np.clip(u[0], -A_MAX, A_MAX)), float(np.clip(u[1], -W_MAX, W_MAX))
    x = np.array(ego, float); traj = [x.copy()]
    for _ in range(int(round(T / dt))):
        v, th = x[3], x[2]
        x = x + dt * np.array([v*math.cos(th), v*math.sin(th), w, a])
        traj.append(x.copy())
    x[3] = float(np.clip(x[3], 0.0, V_MAX))
    traj[-1] = x
    return x, np.array(traj)


def run_episode(ego0, obs0, olen, owid, a1, a2, variant, n_steps=24):
    d_safe = _circum(L_SHIP, W_SHIP) + _circum(olen, owid)
    ego = list(ego0); obs = list(obs0)
    goal = np.array([ego0[0] + 6000*math.cos(ego0[2]), ego0[1] + 6000*math.sin(ego0[2])])
    min_d = 1e18; infeas = 0
    for k in range(n_steps):
        u_nom = colregs_nominal(ego, obs, goal, variant)
        g, b = hocbf_constraint(ego, obs, d_safe, a1, a2)
        u, feas = qp_project(u_nom, g, b, (-A_MAX, A_MAX, -W_MAX, W_MAX))
        if not feas:
            infeas += 1
            beta = relbearing(ego, obs)
            u = np.array([-A_MAX, -W_MAX if beta > 0 else W_MAX])   # fallback: 满减速+转离(无保证)
        ego, etraj = step_ego(ego, u)
        for j in range(len(etraj)):
            t = j * 0.1
            oc = (obs[0]+obs[3]*math.cos(obs[2])*t, obs[1]+obs[3]*math.sin(obs[2])*t)
            dd = _ego_rect((etraj[j][0], etraj[j][1]), etraj[j][2], L_SHIP, W_SHIP).distance(
                 _ego_rect(oc, obs[2], olen, owid))
            if dd < min_d:
                min_d = dd
        obs = [obs[0]+obs[3]*math.cos(obs[2])*DT, obs[1]+obs[3]*math.sin(obs[2])*DT, obs[2], obs[3]]
        if min_d <= 0:
            break
    return min_d, infeas


def phase_synth():
    """本机·合成对撞几何消融(plain vs colregs)·不需 vesselmodels。"""
    NSAMP = int(os.environ.get("SYNTH_N", "120"))
    SYN = os.environ.get("SYN_JSONL", "/home/user/TRB-2027-ContinuesPPO/TRB/结果/结果-block3-0722/block3_synthetic_states.jsonl")
    recs = [json.loads(l) for l in open(SYN)]
    clean = [r for r in recs if _ego_rect((r['ego'][0], r['ego'][1]), r['ego'][2], L_SHIP, W_SHIP).distance(
             _ego_rect((r['obs'][0], r['obs'][1]), r['obs'][2], r['obs_len'], r['obs_wid'])) > 0][:NSAMP]
    print(f"B1 CBF-QP 合成消融 · 对撞几何(剔退化) n={len(clean)} · 10s ZOH · 裸船体碰撞", flush=True)
    for variant in ("plain", "colregs"):
        for (a1, a2) in [(0.3, 0.3), (0.5, 0.5)]:
            ncol = ninf = 0; mds = []
            for r in clean:
                md, inf = run_episode(r['ego'], r['obs'], r['obs_len'], r['obs_wid'], a1, a2, variant)
                ncol += (md <= 0); ninf += (inf > 0); mds.append(md)
            print(f"  [{variant:7s}] α={a1}: 碰撞 {ncol}/{len(clean)} ({100*ncol/len(clean):.1f}%) · QP不可行发生局 {ninf} · 最小距中位 {np.median(mds):.0f}m", flush=True)
    print("  判读(诚实)：plain 高碰撞=退化朴素CBF(打稻草人·别吹)；colregs 降多少=偏置修退化幅度。", flush=True)
    print("    → 真结论=CBF 非自动 0 碰撞(10s ZOH+大船)·数量对比须服务器真基准+引已发表公式。", flush=True)


def phase_run():
    """服务器·真基准·u_nom=盾策略动作·CBF-QP 滤波·同 harness 对照本文盾。需 vesselmodels/env。"""
    assert _HAVE_OFFICIAL, "--run 需 vesselmodels/env → 服务器跑"
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from trb_env.train import make_obs_transform
    from trb_env.usv_continuous_shield import ContinuousProjectionEnv
    from trb_env.usv_scenarios import load_scenario_pool
    from run_step4e import load_manifest_split
    CKPT_DIR = os.environ["CKPT_DIR"]; CKPT_TMPL = os.environ.get("CKPT_TMPL", "Continuous-safe_s{s}_L1rateON_ppo_s{s}")
    SEEDS = [int(x) for x in os.environ.get("SEEDS", "0 1 2 3 4 5 6 7 8 9").split()]
    MANIFEST = os.environ["STEP4E_MANIFEST"]; VARIANT = os.environ.get("CBF_VARIANT", "colregs")
    A1 = float(os.environ.get("CBF_A1", "0.3")); A2 = float(os.environ.get("CBF_A2", "0.3"))
    OUT = os.environ.get("OUT_JSONL", "b1_cbf_run.jsonl")
    bdir = os.environ.get("STEP4E_BALANCED_DIR") or os.path.dirname(os.path.abspath(MANIFEST))
    _tr, test_paths, _i = load_manifest_split(MANIFEST, bdir); pool = load_scenario_pool(test_paths)
    print(f"[run B1] CBF-QP variant={VARIANT} α=({A1},{A2}) · 场景 n={len(pool)} 种子={SEEDS} → {OUT}", flush=True)
    print("  ⚠️ u_nom=盾策略确定性动作(同口径对照)·裸船体碰撞(步内细积分)·报 QP 不可行率·绝不 claim 0", flush=True)

    def _mk(sc, pp):
        return ContinuousProjectionEnv(sc, pp, shield=True, goal_cone_half=None, goal_v_floor=2.0, augment_rho=False)
    # ⚠️ 关键契约(须服务器验)：本 --run 用 env 提供 ego/obs 真态·但【用 CBF-QP 的 u 替换盾的投影】跑闭环——
    #   须确认 ContinuousProjectionEnv 能接受外部原始动作并【绕过其自身投影】(shield=False 分支或 raw step)·否则是"盾上叠CBF"非公平对照。
    #   → 预检项：核 env 是否有 raw/no-shield step 接口喂 CBF 动作。此处留 NotImplemented 待接线+对抗审。
    raise SystemExit("[run B1] 占位：须先接 env raw-action 接口(绕过盾投影)喂 CBF-QP 动作·见上契约·接线后对抗审再跑")


def phase_selftest():
    print("=== B1 --selftest：CBF-QP 滤波 + 正对遇退化复现（本机·不依赖 vesselmodels）===")
    ok = True
    # T1: 清晰 head-on·plain HOCBF 应退化(只减速不转·撞或擦)·colregs 应转开
    ego = [0, 0, 0, 9.5]; obs = [2000, 0, math.pi, 6.0]; olen, owid = 200, 35
    mdp, _ = run_episode(ego, obs, olen, owid, 0.3, 0.3, "plain")
    mdc, _ = run_episode(ego, obs, olen, owid, 0.3, 0.3, "colregs")
    t1 = mdc > mdp   # colregs 最小距应比 plain 大(转开了)
    print(f"  [T1] head-on 最小距: plain={mdp:.0f}m colregs={mdc:.0f}m → colregs 应更大(修退化) {'✅' if t1 else '🔴'}")
    ok = ok and t1
    # T2: QP 投影正确性(约束满足)
    g = np.array([1.0, 2.0]); b = -1.0
    u, feas = qp_project(np.array([0.0, 0.0]), g, b, (-A_MAX, A_MAX, -W_MAX, W_MAX))
    t2 = (not feas) or (float(g @ u) <= b + 1e-6)   # 可行则须满足约束
    print(f"  [T2] QP 投影 u={u} feas={feas} g·u={float(g@u):.4f}≤b={b}? → {'✅' if t2 else '🔴'}")
    ok = ok and t2
    print("  " + ("✅ B1 selftest 通过" if ok else "🔴 B1 selftest 有洞"))
    return 0 if ok else 1


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "--selftest"
    if mode == "--synth":
        phase_synth()
    elif mode == "--run":
        phase_run()
    elif mode == "--selftest":
        sys.exit(phase_selftest())
    else:
        print(__doc__)
        print("用法: python b1_cbf_baseline.py [--synth | --run | --selftest]")
        sys.exit(2)
