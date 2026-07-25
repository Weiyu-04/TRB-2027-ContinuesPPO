#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""盾 cert_v2 终端约束【闭环 eval】（任务A·2026-07-25·纯 eval 不烧训练卡）。
把 U_term backup-maneuver 终端约束(terminal_mode='certv2')真跑进盾·对照默认(off)·量它对闭环的影响。

【它回答什么】前向不变终端门接进盾后：① 会不会破坏现有 0 碰撞(regression) ② 介入(退兜底)率多高(certv2 更严→更多 fallback?)
  ③ 到达率代价 ④ bit-identical 冒烟(off 配置与现状一致)。=盾从"经验一步前瞻"升级"可证明前向不变"的闭环代价体检。

【三档 env 配置】(同一金标策略·同场景池·同种子·只换盾终端模式)：
  · off       = recursive_feasibility=False（现状·bit-identical 基线）
  · discrete  = recursive_feasibility=True, terminal_mode='discrete'（旧 encounter_action_verification 终端·L192 机件）
  · certv2    = recursive_feasibility=True, terminal_mode='certv2'（block1-SOUND backup-maneuver·本任务·真宽 obs_width 自动 plumb）

【口径】碰撞=env 决策步 shapely 相交(term·真录制障碍)；source 分布=projection/emergency/relaxed/collision_min(介入率)；到达=flags.goal。
【阶段】--run(服务器·需 vesselmodels/env) / --selftest(本机·查 uterm 族 10s 对齐 + import)。
服务器跑前 user 拍板 + 逐字预检 + screen（04 运行手册）。
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from block3_partition_probe import _HAVE_OFFICIAL


def phase_run():
    assert _HAVE_OFFICIAL, "--run 需 vesselmodels/env → 服务器跑"
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from trb_env.train import make_obs_transform
    from trb_env.usv_continuous_shield import ContinuousProjectionEnv
    from trb_env.usv_scenarios import load_scenario_pool
    from trb_env import usv_projection as _proj   # 诊断计数 reset/读
    from trb_env.usv_colregs import ViolationCounter          # COLREGs 违规(§VII-A 官方口径·复用 evaluate·非重造)
    from trb_env.evaluate import _control_quality             # 平滑/jerk/转艏增量/油门增量/路径长(复用 evaluate CAT7·非重造)
    from trb_env import metrics_subgrid as _msg                # 次网格细调率 + 按态势拆转艏(本机已单测 6 项)
    from trb_env.usv_env import A_ACC as _A_ACC, A_OMEGA as _A_OMEGA
    _msg.assert_grid_matches(_A_ACC, _A_OMEGA)                # 🔴 守卫:镜像网格常量==usv_env 真相源(防静默漂移·同 uterm 范式)
    from run_step4e import load_manifest_split

    CKPT_DIR = os.environ["CKPT_DIR"]
    CKPT_TMPL = os.environ.get("CKPT_TMPL", "Continuous-safe_s{s}_L1rateON_ppo_s{s}")
    SEEDS = [int(x) for x in os.environ.get("SEEDS", "0 1 2 3 4 5 6 7 8 9").split()]
    MANIFEST = os.environ["STEP4E_MANIFEST"]
    MODES = os.environ.get("SHIELD_MODES", "off certv2").split()   # 默认 off vs certv2（加 discrete 三档全跑）
    OUT = os.environ.get("OUT_JSONL", "shield_certv2_eval.jsonl")
    TRAJ_OUT = os.environ.get("TRAJ_JSONL", OUT[:-6] + "_traj.jsonl" if OUT.endswith(".jsonl") else OUT + "_traj.jsonl")
    TRAJ_IDXS = set(int(x) for x in os.environ.get("TRAJ_IDXS", "0 1 2 3 4 5").split())   # 存这几个场景逐步轨迹(仅首种子·多算法轨迹对比画图用)
    bdir = os.environ.get("STEP4E_BALANCED_DIR") or os.path.dirname(os.path.abspath(MANIFEST))
    _tr, test_paths, _i = load_manifest_split(MANIFEST, bdir)
    pool = load_scenario_pool(test_paths)
    print(f"[shield certv2 eval] 场景 n={len(pool)} 种子={SEEDS} 档={MODES} → {OUT}", flush=True)
    print("  ⚠️ 纯 eval·金标策略确定性·同池同种子·只换盾终端模式。off=bit-identical 基线。", flush=True)

    def _mk(sc, pp, mode):
        # 🔴 env 用 **proj_kwargs 变参透传 ContinuousColregsProjection → recursive_feasibility/terminal_mode 须【直接铺开】作 kwarg
        #   (别包成 proj_kwargs={...}·那会把字面 key 'proj_kwargs' 塞进投影构造=TypeError)。基座 kwargs 照抄 A3 collect(已验)。
        kw = dict(shield=True, goal_cone_half=None, goal_v_floor=2.0, augment_rho=False)
        if mode == "discrete":
            kw.update(recursive_feasibility=True, terminal_mode="discrete")   # 旧离散终端(L192 机件)
        elif mode == "certv2":
            kw.update(recursive_feasibility=True, terminal_mode="certv2")     # block1-SOUND backup-maneuver(本任务)
        elif mode != "off":                                                   # off=recursive_feasibility 默认 False=现状 bit-identical
            raise SystemExit(f"未知 mode={mode}")
        return ContinuousProjectionEnv(sc, pp, **kw)

    fo = open(OUT, "w"); ftraj = open(TRAJ_OUT, "w")
    _MK = ("ctrl_jerk_norm_mean", "yaw_incr_mean", "accel_incr_mean", "path_len_m")   # 平滑度/路径套件(evaluate CAT7)
    _SG = ("subgrid_accel_frac", "subgrid_yaw_frac", "yaw_incr_giveway", "yaw_incr_other")   # 次网格率+按态势拆转艏
    for mode in MODES:
        _proj.reset_certv2_stats()                     # 诊断计数清零(每档)
        n_ep = n_col = n_arr = 0
        src = {}
        agg = {"violations": [], **{k: [] for k in _MK}, **{k: [] for k in _SG}}   # 全指标聚合(违规+平滑+次网格/态势拆)
        for s in SEEDS:
            ck = os.path.join(CKPT_DIR, CKPT_TMPL.format(s=s))
            if not (os.path.exists(ck + ".zip") and os.path.exists(ck + "_vecnorm.pkl")):
                print(f"  [{mode}] s{s}: 缺 ckpt → 跳过", flush=True); continue
            bv = DummyVecEnv([lambda: _mk(pool[0][0], pool[0][1], mode)])
            vn = VecNormalize.load(ck + "_vecnorm.pkl", bv); vn.training = False
            if int(np.asarray(vn.obs_rms.mean).shape[0]) != int(bv.observation_space.shape[0]):
                raise SystemExit(f"s{s}: vecnorm 维≠env 维")
            tf = make_obs_transform(vn); model = PPO.load(ck + ".zip", device="cpu")
            for si, (sc, pp) in enumerate(pool):
                env = _mk(sc, pp, mode); obs, info = env.reset(seed=0)
                n_ep += 1; collided = arrived = False
                vc = ViolationCounter()                                    # COLREGs 违规(喂真态 pre-step·同 evaluate·忠实官方 monitor)
                applied = []; rhos = []; positions = [np.asarray(env._ego_vs().position, float)]   # 控制质量素材(执行控制+ρ+位置序列)
                save_traj = (si in TRAJ_IDXS and s == SEEDS[0]); traj = [] if save_traj else None
                for step_i in range(200):
                    s_obs = env._obs_vs()                                  # pre-step 真态(违规口径+轨迹·同 evaluate)
                    if s_obs is not None:
                        vc.step(env._ego_vs(), s_obs)
                    if save_traj:
                        _e = env._ego_vs()
                        traj.append(dict(step=step_i, ex=float(_e.position[0]), ey=float(_e.position[1]), eth=float(_e.orientation),
                                         ox=(float(s_obs.position[0]) if s_obs is not None else None),
                                         oy=(float(s_obs.position[1]) if s_obs is not None else None),
                                         oth=(float(s_obs.orientation) if s_obs is not None else None), rho=int(env._rho)))
                    act, _ = model.predict(tf(obs), deterministic=True)
                    obs, _r, term, trunc, info = env.step(np.asarray(act, float))
                    _la = getattr(env.env, "last_action", None)           # 执行控制(=平滑度素材·同 evaluate CAT7·仅正常操作步入 jerk)
                    if _la is not None:
                        applied.append(np.asarray(_la, float)); rhos.append(int(env._rho))   # ρ 同步记(按态势拆转艏用·pre-step ρ 同 obs 口径)
                    positions.append(np.asarray(env._ego_vs().position, float))
                    so = info.get("source")
                    if so is not None:
                        src[so] = src.get(so, 0) + 1
                    flags = info.get("flags", {})
                    if flags.get("collision"):
                        collided = True
                    if flags.get("goal"):
                        arrived = True
                    if term or trunc:
                        break
                s_obs_final = env._obs_vs()                               # 补喂终止态 @k(忠实官方离线 monitor·同 evaluate)
                if s_obs_final is not None:
                    vc.step(env._ego_vs(), s_obs_final)
                vc.finalize()
                viol = int(vc.standon_violations + vc.giveway_violations)
                cq = _control_quality(applied, positions) or {}
                sg = _msg.subgrid_and_rho_split(applied, rhos)             # 次网格细调率 + 按态势拆 |Δω|
                n_col += int(collided); n_arr += int(arrived)
                agg["violations"].append(viol)
                for k in _MK:
                    if cq.get(k) is not None:
                        agg[k].append(cq[k])
                for k in _SG:
                    if sg.get(k) is not None:
                        agg[k].append(sg[k])
                fo.write(json.dumps(dict(mode=mode, seed=s, scn_idx=si, collided=collided, arrived=arrived, violations=viol,
                                         giveway_violations=int(vc.giveway_violations), standon_violations=int(vc.standon_violations),
                                         **{k: cq.get(k) for k in
                                            ("ctrl_jerk_norm_mean", "yaw_incr_mean", "accel_incr_mean", "ctrl_effort_norm_mean", "path_len_m")},
                                         **{k: sg.get(k) for k in
                                            ("subgrid_accel_frac", "subgrid_yaw_frac", "n_inbox_pairs",
                                             "yaw_incr_giveway", "yaw_incr_other", "n_pairs_giveway", "n_pairs_other")})) + "\n")
                if save_traj:
                    ftraj.write(json.dumps(dict(mode=mode, seed=s, scn_idx=si, collided=collided, arrived=arrived, traj=traj)) + "\n")
            fo.flush(); ftraj.flush()
            print(f"  [{mode}] s{s}: 累计 ep={n_ep} 碰撞={n_col} 到达={n_arr}", flush=True)
        tot_src = max(1, sum(src.values()))
        interv = 100 * (tot_src - src.get("projection", 0)) / tot_src   # 非 projection = 退兜底 = 介入
        st = dict(_proj._CERTV2_STATS)                 # 诊断：终端检查 calls/giveway/rejects
        _mean = lambda xs: (sum(xs) / len(xs)) if xs else float("nan")
        print(f"[{mode}] done · ep={n_ep} · 碰撞率={100*n_col/max(1,n_ep):.2f}%({n_col}) · 到达率={100*n_arr/max(1,n_ep):.2f}% · "
              f"介入率={interv:.2f}% · 违规/局={_mean(agg['violations']):.3f} · jerk平滑={_mean(agg['ctrl_jerk_norm_mean']):.4f} · "
              f"转艏增量={_mean(agg['yaw_incr_mean']):.4f} · 油门增量={_mean(agg['accel_incr_mean']):.4f} · 路径长={_mean(agg['path_len_m']):.0f}m · "
              f"source={src} · 终端检查[calls={st['calls']} giveway={st['giveway']} rejects={st['rejects']}]", flush=True)
        print(f"   └ 次网格细调率(离散恒0·=用掉的连续分辨率): 油门={100*_mean(agg['subgrid_accel_frac']):.1f}% 转艏={100*_mean(agg['subgrid_yaw_frac']):.1f}%"
              f" ｜ 转艏增量按态势: 让路={_mean(agg['yaw_incr_giveway']):.4f} vs 非让路={_mean(agg['yaw_incr_other']):.4f}"
              f"(让路更大 ⟹ 转艏活动=合规让路机动的代价)", flush=True)
    fo.close(); ftraj.close()
    print(f"[shield certv2 eval] done → {OUT} · 轨迹 → {TRAJ_OUT}", flush=True)
    print("  判读：off 碰撞率应=现状基线(~0·regression 冒烟)；certv2 碰撞率≤off(终端门更严·不该更差)；"
          "介入率 certv2≥off(更严→更多 fallback·代价);到达率代价看差多少。", flush=True)


def phase_selftest():
    print("=== shield_certv2_eval --selftest（本机·查 uterm 族 10s 对齐 + import·不跑闭环）===")
    ok = True
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "trb_env"))
    import uterm_terminal as U
    fam = U.straight_tail_family()
    # 每个机动的每段 dur 须是 10s 整数倍（可执行=对抗审 Finding A）
    bad = []
    for name, segs, w0 in fam:
        for a, w, dur in segs:
            if dur is not None and abs(round(dur / U.DECISION_DT) * U.DECISION_DT - dur) > 1e-9:
                bad.append((name, dur))
    t1 = (len(bad) == 0)
    print(f"  [T1] 机动族 {len(fam)} 条·非10s对齐段 {len(bad)}(应0=可执行) → {'✅' if t1 else '🔴 '+str(bad[:3])}")
    ok = ok and t1
    t2 = (len(fam) == 57)   # 12 durs ×2 dir ×2(turn/dec) + 3 accdec×2 + 3 straight
    print(f"  [T2] 族大小 {len(fam)}(应57=可执行族) → {'✅' if t2 else '🔴'}")
    ok = ok and t2
    print("  " + ("✅ selftest 通过（族可执行·闭环 eval 逻辑须服务器冒烟）" if ok else "🔴 有洞"))
    return 0 if ok else 1


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "--selftest"
    if mode == "--run":
        phase_run()
    elif mode == "--selftest":
        sys.exit(phase_selftest())
    else:
        print(__doc__)
        print("用法: python shield_certv2_eval.py [--run | --selftest]")
        sys.exit(2)
