"""
TRB · 训练 run 静态元数据采集（Node L · L1a，2026-06-17d）
============================================================
**目的**（user 要求"第一次跑就把所有信息+论文叙事参数全记齐、不重复跑"）：在 C3 训练 run 开始时，
把【所有静态参数】程序化内省成一份 JSON 落盘——船舶动力学参数 / COLREGs 参数 / 投影盾参数 / reward 系数 /
训练配置 / 动作网格 / 论文叙事关键量。**内省各模块的真实常量**（非硬编）→ 常量变了元数据自动跟上、永不 stale。

用法（run_step4e C3 起跑时调一次）：
    from trb_env.run_metadata import build_static_metadata, write_run_metadata
    write_run_metadata("结果/run_metadata_4way.json", run_config={...})  # run_config = 本次 run 的 seeds/场景/tag…

论文写作：直接引这份 JSON 的数值（动力学/COLREGs/reward 全 fact-based 对 Krasowski 2024 Table II + Meyer 2020），
叙事关键量（ω_turn 连续 vs 离散精度、动作网格、档位A 经验性边界）也在 paper_narrative 段，不用回头翻代码/重跑。
"""
from __future__ import annotations

import inspect
import json
import math
import os


def _rad_deg(x: float) -> dict:
    """角度常量同时给 rad + deg（论文可读）。"""
    return {"rad": float(x), "deg": round(math.degrees(float(x)), 4)}


def _lib_versions() -> dict:
    """CAT6：关键库版本（论文复现性·importlib.metadata 实读非硬编·升级自动跟上）。"""
    import importlib.metadata as _md
    import platform as _pf
    out = {"python": _pf.python_version()}
    for pkg in ("stable-baselines3", "sb3-contrib", "torch", "gymnasium",
                "numpy", "shapely", "osqp", "commonocean-io"):
        try:
            out[pkg] = _md.version(pkg)
        except Exception:
            out[pkg] = None
    return out


def _num_or_repr(v):
    """sb3 默认值 → JSON 友好（bool/数值原样、其余 repr，如 learning_rate schedule / train_freq）。"""
    if isinstance(v, bool) or isinstance(v, (int, float)):
        return v
    return repr(v)


def _sb3_hyperparameters() -> dict:
    """CAT6：sb3 算法【真实超参】——maker 未覆盖项 = sb3 类 __init__ 默认，inspect.signature 内省（非硬编·升级 sb3 自动跟上·同 L1a-1/L1a-2 范式）。
    maker 显式覆盖：gamma(0.99)/net_arch[64,64]/ent_coef(离散=run_config.ent_start/end/frac·连续 auto)/seed/device/norm_reward(VecNormalize)；其余取类默认（记录于此）。"""
    out = {"overridden_by_maker": ["gamma", "net_arch", "ent_coef(discrete=见 run_config.ent_start/end/frac·continuous=auto)", "seed", "device", "norm_reward"],
           "note": "values below = sb3 class __init__ defaults for params the makers do NOT override; learning_rate may be a schedule (repr); 论文报告超参直接引此 + training_config 的 maker 覆盖项"}
    try:
        from stable_baselines3 import SAC
        _sig = inspect.signature(SAC.__init__).parameters
        out["SAC"] = {k: _num_or_repr(_sig[k].default) for k in
                      ("learning_rate", "buffer_size", "learning_starts", "batch_size",
                       "tau", "train_freq", "gradient_steps", "target_update_interval") if k in _sig}
    except Exception as _e:
        out["SAC"] = f"introspect failed: {_e!r}"
    try:
        from sb3_contrib import MaskablePPO
        _sig = inspect.signature(MaskablePPO.__init__).parameters
        out["MaskablePPO"] = {k: _num_or_repr(_sig[k].default) for k in
                              ("learning_rate", "n_steps", "batch_size", "n_epochs",
                               "gae_lambda", "clip_range", "vf_coef", "max_grad_norm") if k in _sig}
    except Exception as _e:
        out["MaskablePPO"] = f"introspect failed: {_e!r}"
    return out


def _krasowski_reference() -> dict:
    """Phase-1 复现对照锚（Krasowski & Althoff 2024 Table III【文献值】·非本项目计算·盾别归属见 03 L34）。
    ⚠️ 文献值【无法从代码派生】、只能引用 → 此处硬记 + 引文 + 诚实 caveat（区别于 ω_turn/gamma 那类可派生项）。"""
    return {
        "source": "Krasowski & Althoff 2024, Table III (PDF p11-13; §VII); shield-attribution per 03 L34",
        "colregs_violations_per_episode": {
            "Base_unshielded": 2.65, "Rule-reward_unshielded": 2.24, "shielded_Safe": 0.0},
        "arrival_rate_pct_approx": 86,
        "honesty_note": "literature anchors for Phase-1 reproduction comparison; OUR shielded violations (~0.98) do NOT beat her shielded Safe=0 (honesty red line); verify exact per-cell vs PDF Table III before final paper",
    }


def build_static_metadata() -> dict:
    """内省 trb_env 各模块的静态参数 → 嵌套 dict（不烧算力、可随时重建）。"""
    from . import usv_colregs as cl
    from . import usv_reward as rw
    from . import usv_env as ev
    from . import usv_dynamics as dyn
    from . import usv_projection as pj
    from . import train as tr

    p = dyn.make_vessel_params()
    # 船舶动力学参数（Krasowski 官方 parameters_vessel_1 = SR108 集装箱船；内省全部 public 数值属性）
    vessel = {}
    for name in dir(p):
        if name.startswith("_"):
            continue
        try:
            val = getattr(p, name)
        except Exception:
            continue
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            vessel[name] = float(val)

    ship_dynamics = {
        "model": "vessel_dynamics_yp (Krasowski & Althoff 2024 eq.(1); yaw-constrained point mass Ω_yc)",
        "rhs": "f = [cos(theta)*v, sin(theta)*v, omega, a]  (state x=[px,py,theta,v], input u=[a,omega])",
        "integrator": "scipy.integrate.odeint over [0,dt] (constant-u; ego trajectory convention, her exact integrator unpublished)",
        "vessel_class": "parameters_vessel_1 (SR108 container ship)",
        "length_m": float(p.l),
        "width_m": float(cl.EGO_WIDTH),
        "a_max_m_s2": float(p.a_max),
        "omega_max_rad_s": float(p.w_max),
        "v_max_m_s": float(p.v_max),
        "v_max_note": "paper §VII reduced default 16.8 -> 9.5 m/s; RHS does not clip v (env layer manages via safe_speed)",
        "decision_dt_s": float(dyn.DECISION_DT),
        "vessel_params_full": vessel,
    }

    colregs_params = {
        "sector_angles": {
            "delta_head_on": _rad_deg(cl.DELTA_HEAD_ON),
            "delta_overtake": _rad_deg(cl.DELTA_OVERTAKE),
            "delta_no_turn": _rad_deg(cl.DELTA_NO_TURN),
            "delta_large_turn": _rad_deg(cl.DELTA_LARGE_TURN),
            "sector_side": _rad_deg(cl.SECTOR_SIDE),
            "sector_behind_lo": _rad_deg(cl.SECTOR_BEHIND_LO),
            "sector_behind_hi": _rad_deg(cl.SECTOR_BEHIND_HI),
        },
        "time_horizons_s": {
            "t_horizon_collision_possible": float(cl.T_HORIZON),
            "t_pred_emergency": float(cl.T_PRED),
            "t_react": float(cl.T_REACT),
            "t_maneuver": float(cl.T_MANEUVER),
            "dt_decision": float(cl.DT),
        },
        "collision_possible": {
            "r_m_factor": float(cl.R_M_FACTOR), "v_eps_m_s": float(cl.V_EPS),
        },
        "emergency_reachset": {
            "v_pm_max_m_s": float(cl.V_PM_MAX), "a_pm_max_m_s2": float(cl.A_PM_MAX),
            "dt_reach_s": float(cl.DT_REACH), "d_resolved_factor": float(cl.D_RESOLVED_FACTOR),
            "reach_radius_pm": "min(0.5*a_pm_max*t*(t+dt_reach), v_pm_max*(t+dt_reach))  (strict over-approx of continuous Ω_pm)",
        },
        "emergency_controller": {
            "delta_ahead": _rad_deg(cl.DELTA_AHEAD), "delta_stern": _rad_deg(cl.DELTA_STERN),
            "a_stern_factor": float(cl.A_STERN_FACTOR), "dobs_safety_factor": float(cl.DOBS_SAFETY_FACTOR),
            "dmin_ahead_factor": float(cl.DMIN_AHEAD_FACTOR), "t_m_s": float(cl.T_M), "t_max_m_s": float(cl.T_MAX_M),
            "v_desired_m_s": float(cl.V_DESIRED), "delta_v_omega": float(cl.DELTA_V_OMEGA),
        },
        "rho_states": {"NO_CONFLICT": cl.RHO_NO_CONFLICT, "STAND_ON": cl.RHO_STAND_ON,
                       "HEAD_ON": cl.RHO_HEAD_ON, "CROSSING": cl.RHO_CROSSING,
                       "OVERTAKE": cl.RHO_OVERTAKE, "EMERGENCY": cl.RHO_EMERGENCY},
        "ego_width_m": float(cl.EGO_WIDTH),
    }

    proj = pj.ContinuousColregsProjection(p.a_max, p.w_max)   # 默认参数 probe
    projection_params = {
        "tier": "A (empirical, one-step lookahead + large d_safe margin; NOT provable — provable = tier B = Phase 4 invariant set)",
        "omega_turn_rad_s": float(proj.omega_turn),
        "omega_turn_note": "give-way turn rate = DELTA_LARGE_TURN(20deg)/T_M(40s) ~= 0.008727 rad/s (exact infimum)",
        "eps_omega_rad_s": float(proj.eps_omega),
        "eps_a_m_s2": float(proj.eps_a),
        "nj_degen_tol": float(pj._NJ_DEGEN_TOL),
        "d_safe": "R_ego + R_obs (R_obs includes dobs_safety = 2*l_obs); separating-hyperplane + scalar first-order linearization",
        "fallback_chain": ["projection", "emergency", "relaxed", "collision_min", "degenerate"],
        "fallback_note": "P=empty detected by QP -> falls to emergency/collision_min/relaxed; never silently passes unsafe action",
    }

    reward_coefficients = {
        "source": "Krasowski 2024 Table II (PDF p11) + Meyer 2020 Table 4 (r_colregs)",
        "c_time": float(rw.C_TIME), "c_area": float(rw.C_AREA), "c_goal": float(rw.C_GOAL),
        "c_stopped": float(rw.C_STOPPED), "c_collision": float(rw.C_COLLISION),
        "c_emergency": float(rw.C_EMERGENCY), "c_reach": float(rw.C_REACH),
        "c_v": float(rw.C_V), "c_deviate": float(rw.C_DEVIATE),
        "v_low_m_s": float(rw.V_LOW), "v_high_m_s": float(rw.V_HIGH), "d_hull_m": float(rw.D_HULL),
        "r_colregs_meyer": {
            "alpha_x": float(rw.ALPHA_X), "gamma_theta_dyn": float(rw.GAMMA_THETA_DYN),
            "zeta_x": dict(rw.ZETA_X), "zeta_v": {k: list(v) if isinstance(v, (list, tuple)) else v
                                                  for k, v in rw.ZETA_V.items()},
            "default_d_sense_m": float(rw.DEFAULT_D_SENSE),
            "deviation_note": "v_y normalized by v_pm_max + exp clamp (BLOCKER fix) — MUST declare this deviation from Meyer literal in paper",
        },
    }

    # gamma 内省自 trainer 默认参数（非硬编 → 改默认元数据自动跟上；四方同 gamma，由 usv_sac_train 口径自检强制，L1a-2）
    _gamma = float(inspect.signature(tr.train_multiscene).parameters["gamma"].default)
    training_config = {
        "policy_net_arch": list(tr.POLICY_NET_ARCH),
        "total_timesteps_per_seed": int(tr.TOTAL_TIMESTEPS),
        "n_seeds_paper": int(tr.N_SEEDS),
        # ⚠️ 这是 train.py 静态产品默认·仅参考；本 run 实际离散 ent 配方（常量或退火·env 驱动）见 run_config.ent_start/ent_end/ent_frac（`03` L59·勿照抄此值进论文）。
        "ent_coef_discrete_traindefault": float(tr.ENT_COEF),
        "ent_coef_discrete_actual_in": "run_config.ent_start/ent_end/ent_frac (本 run 真实配方)",
        "ent_coef_continuous": "auto (SAC max-entropy)",
        "vecnorm_kwargs": dict(tr.VECNORM_KWARGS),
        "gamma": _gamma,
        "algorithms": {"Base": "MaskablePPO (unshielded, colregs_weight=0)",
                       "Rule-reward": "MaskablePPO (unshielded, colregs_weight=1)",
                       "Discrete-safe": "MaskablePPO + As(rho) action mask (shielded)",
                       "Continuous-safe": "SAC + continuous projection shield (this work)"},
    }

    action_space = {
        "continuous": {
            # RL 正常操作动作箱 = Krasowski A_a/A_ω 范围（=离散网格 span·Fix② 03 L63）；满程 ±a_max/±w_max 只给盾/紧急控制器
            "a_range": [-float(ev.A_NORMAL_ACCEL_MAX), float(ev.A_NORMAL_ACCEL_MAX)],
            "omega_range": [-float(ev.A_NORMAL_OMEGA_MAX), float(ev.A_NORMAL_OMEGA_MAX)],
            "physical_emergency_range": {"a": [-float(p.a_max), float(p.a_max)],
                                         "omega": [-float(p.w_max), float(p.w_max)]},
            "physical_range_note": "physical ±a_max/±w_max used ONLY by shield/emergency-controller/_map_action box-clip, NOT RL normal-operation authority (Fix② 03 L63)",
        },
        "discrete_grid": {"a_values": list(ev.A_ACC), "omega_values": list(ev.A_OMEGA),
                          "n_regular": int(ev.N_DISCRETE), "idx_emergency": int(ev.IDX_EMERGENCY)},
    }

    # 离散网格最小合规让路转艏率：A_ω 中满足 |ω|·T_M ≥ Δlarge_turn(20°) 的最小正值（usv_colregs:826 fact-derive，非硬编 → 网格/T_M/阈值改则自动跟上，L1a-1）
    _omega_turn_discrete = float(min(g for g in ev.A_OMEGA if g > 0 and g * cl.T_M >= cl.DELTA_LARGE_TURN - 1e-9))
    paper_narrative = {
        "contribution": "continuous action ∩ provably(tier-B/Phase4) COLREGs-compliant ∩ maritime collision avoidance via projection shield",
        "omega_turn_continuous_rad_s": float(proj.omega_turn),
        "omega_turn_discrete_grid_rad_s": _omega_turn_discrete,
        "omega_turn_note": "continuous exact infimum (omega_turn_continuous_rad_s) vs discrete grid min-compliant rate (omega_turn_discrete_grid_rad_s) = continuous fine-control advantage; anchor to these fields, declare in writing",
        "action_range_caveat": "RL action box (continuous arm) = Krasowski normal-operation range ±0.048/±0.018, EQUAL to the discrete grid's span (Fix② 03 L63). The earlier RANGE-difference confound (review L1a-4) is now ELIMINATED: continuous and discrete arms have MATCHED action authority; the only remaining action-space difference is RESOLUTION (continuous interval vs 7-point grid). Physical ±a_max/±w_max is reserved for the shield/emergency-controller (NOT RL authority).",
        "action_aliasing_metric_caveat": "||u_applied - u_desired|| uses u_applied = env action AFTER box clip, so it = projection correction + box clipping (NOT pure projection). Anchor the zero-correction-fraction selling point to keep / trained-SAC policy (~0.96 on T-0 keep, NOT the 0.48 test stub); disclose box-clip inclusion (review L1b-3/L1b-1)",
        "four_party_intended_differences_only": ["shield/projection on-off", "algorithm (MaskablePPO vs SAC)", "r_colregs (RR/Safe have, Base/Continuous drop)"],
        "honesty_red_lines": [
            "tier-A here is empirical (one-step lookahead + d_safe), NOT provable — provable is tier-B Phase 4",
            "COLREGs-violation count must NOT be claimed to beat Krasowski (ours shielded 0.98 > her shielded Safe 0)",
            "reproduced discrete baselines are OUR re-implementation; seed fragility not attributed to Krasowski",
        ],
        "key_metrics_table3": ["arrival rate (%)", "collision rate (%)", "COLREGs violations per episode",
                               "emergency-control steps (%)", "episode length (s)"],
        "continuous_arm_extra_diagnostics": ["fallback steps (%) [P=empty]", "projection correction magnitude ||u_safe - u_desired|| [action aliasing]"],
    }

    return {
        "schema_version": 1,
        "ship_dynamics": ship_dynamics,
        "colregs_params": colregs_params,
        "projection_params": projection_params,
        "reward_coefficients": reward_coefficients,
        "training_config": training_config,
        "action_space": action_space,
        "paper_narrative": paper_narrative,
        "library_versions": _lib_versions(),                  # CAT6 L1a补：复现性（python/sb3/torch/...）
        "sb3_hyperparameters": _sb3_hyperparameters(),        # CAT6 L1a补：sb3 真实超参内省（maker 未覆盖项=类默认）
        "krasowski_table3_reference": _krasowski_reference(),  # L1a补：Phase-1 复现对照锚（文献值+诚实 caveat）
    }


def ev_or(mod, name, default):
    """安全取模块常量（缺则默认；防内省 KeyError）。"""
    return getattr(mod, name, default)


def write_run_metadata(path: str, run_config: dict | None = None) -> dict:
    """组装 静态元数据 + 本次 run 配置 → 写 JSON（UTF-8、缩进），返回完整 dict。

    run_config（run_step4e 传入）：tag / parties / seeds / n_total / total_steps / pool_size /
        split_seed / test_frac / n_envs / n_train / n_test / created_at（时间戳，由调用方传，本模块不取系统时间）。
    """
    meta = build_static_metadata()
    meta["run_config"] = dict(run_config or {})
    # 原子写（.<pid>.tmp + os.replace，同 run_step4e.write_atomic 范式）：launcher 把四方拆单方子进程
    # 并发重写【同名 run_metadata{tag}.json】（L52 MINOR）→ 非原子的 open(w) 有截断读窗口、写中途崩溃留损坏文件；
    # 原子落盘使并发读者只见【旧或新的完整文件】、崩溃只留 .tmp 孤儿不污染真文件。
    tmp = f"{path}.{os.getpid()}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2, sort_keys=False)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return meta
