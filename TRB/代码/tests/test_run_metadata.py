"""
Node L · L1a 静态元数据模块冒烟测试（2026-06-17d）。
跑：/opt/miniconda3/envs/trb/bin/python -B 代码/tests/test_run_metadata.py
（自动计数、非 pytest；纯内省常量、无需场景夹具、离线可跑。）

覆盖：① 各段齐全 + JSON 可序列化 + write/reload；② 关键参数值正确（船舶动力学/COLREGs/reward/训练配置/动作网格/论文叙事）；
③ **live 内省守护**（元数据值 == 源模块常量，非硬编 → 常量改了元数据自动跟上、test 仍守住一致性）。
"""
import sys, os, json, tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from trb_env.run_metadata import build_static_metadata, write_run_metadata
import trb_env.usv_colregs as cl
import trb_env.usv_reward as rw
import trb_env.usv_env as ev
import trb_env.usv_dynamics as dyn
import trb_env.usv_projection as pj
import trb_env.train as tr
import trb_env.usv_sac_train as sac
import inspect

_fail = 0
_n = 0


def ok(name, cond):
    global _fail, _n
    _n += 1
    if not cond:
        _fail += 1
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")


m = build_static_metadata()

print("===== A) 各段齐全 + 可序列化 =====")
for seg in ("ship_dynamics", "colregs_params", "projection_params", "reward_coefficients",
            "training_config", "action_space", "paper_narrative",
            "library_versions", "sb3_hyperparameters", "krasowski_table3_reference"):  # L1a补 3 新段
    ok(f"A 段 {seg} 存在且非空", isinstance(m.get(seg), dict) and len(m[seg]) > 0)
_s = json.dumps(m, ensure_ascii=False)
ok("A JSON 可序列化", len(_s) > 2000)

print("\n===== B) live 内省守护（值 == 源模块常量，非硬编）=====")
_p = dyn.make_vessel_params()
ok("B 船长 == p.l", m["ship_dynamics"]["length_m"] == float(_p.l))
ok("B a_max == p.a_max", m["ship_dynamics"]["a_max_m_s2"] == float(_p.a_max))
ok("B omega_max == p.w_max", m["ship_dynamics"]["omega_max_rad_s"] == float(_p.w_max))
ok("B v_max == p.v_max", m["ship_dynamics"]["v_max_m_s"] == float(_p.v_max))
ok("B 船宽 == cl.EGO_WIDTH", m["ship_dynamics"]["width_m"] == float(cl.EGO_WIDTH))
ok("B t_pred == cl.T_PRED", m["colregs_params"]["time_horizons_s"]["t_pred_emergency"] == float(cl.T_PRED))
ok("B v_pm_max == cl.V_PM_MAX", m["colregs_params"]["emergency_reachset"]["v_pm_max_m_s"] == float(cl.V_PM_MAX))
ok("B delta_large_turn rad == cl.DELTA_LARGE_TURN", m["colregs_params"]["sector_angles"]["delta_large_turn"]["rad"] == float(cl.DELTA_LARGE_TURN))
ok("B C_GOAL == rw.C_GOAL", m["reward_coefficients"]["c_goal"] == float(rw.C_GOAL))
ok("B C_EMERGENCY == rw.C_EMERGENCY", m["reward_coefficients"]["c_emergency"] == float(rw.C_EMERGENCY))
ok("B C_COLLISION == rw.C_COLLISION", m["reward_coefficients"]["c_collision"] == float(rw.C_COLLISION))
ok("B net_arch == tr.POLICY_NET_ARCH", m["training_config"]["policy_net_arch"] == list(tr.POLICY_NET_ARCH))
ok("B total_steps == tr.TOTAL_TIMESTEPS", m["training_config"]["total_timesteps_per_seed"] == int(tr.TOTAL_TIMESTEPS))
ok("B vecnorm clip_obs == tr.VECNORM_KWARGS", m["training_config"]["vecnorm_kwargs"]["clip_obs"] == tr.VECNORM_KWARGS["clip_obs"])
ok("B gamma == train_multiscene 默认(inspect 内省非硬编·改默认必跟随，L1a-2)",
   m["training_config"]["gamma"] == float(inspect.signature(tr.train_multiscene).parameters["gamma"].default))
ok("B gamma 四方一致(离散 train_multiscene 默认 == 连续 make_continuous_safe_model 默认)",
   inspect.signature(tr.train_multiscene).parameters["gamma"].default
   == inspect.signature(sac.make_continuous_safe_model).parameters["gamma"].default)
# B-ent：ent_coef_discrete 已改名为 *_traindefault（防误导照抄·实际配方在 run_config）·无裸误导键（`03` L59）
ok("B ent_coef 标注非误导：有 *_traindefault + 指向 run_config·无裸 ent_coef_discrete 键（L59 防论文照抄 0.01）",
   "ent_coef_discrete_traindefault" in m["training_config"]
   and "ent_coef_discrete" not in m["training_config"]
   and "run_config.ent_start" in m["training_config"]["ent_coef_discrete_actual_in"])
_proj = pj.ContinuousColregsProjection(_p.a_max, _p.w_max)
ok("B omega_turn == proj 默认", abs(m["projection_params"]["omega_turn_rad_s"] - float(_proj.omega_turn)) < 1e-12)
ok("B eps_a == proj 默认", m["projection_params"]["eps_a_m_s2"] == float(_proj.eps_a))
ok("B 动作网格 a_values == ev.A_ACC", m["action_space"]["discrete_grid"]["a_values"] == list(ev.A_ACC))
ok("B idx_emergency == ev.IDX_EMERGENCY", m["action_space"]["discrete_grid"]["idx_emergency"] == int(ev.IDX_EMERGENCY))
ok("B 连续 RL 动作范围 == Krasowski 正常操作 ±A_NORMAL_*（Fix② 03 L63·=离散 span·非物理满程 ±a_max；改回满程则 FAIL）",
   m["action_space"]["continuous"]["a_range"] == [-float(ev.A_NORMAL_ACCEL_MAX), float(ev.A_NORMAL_ACCEL_MAX)]
   and m["action_space"]["continuous"]["omega_range"] == [-float(ev.A_NORMAL_OMEGA_MAX), float(ev.A_NORMAL_OMEGA_MAX)])

print("\n===== C) 论文叙事 + 诚实红线 + 值正确性 =====")
_omega_turn_disc_exp = min(g for g in ev.A_OMEGA if g > 0 and g * cl.T_M >= cl.DELTA_LARGE_TURN - 1e-9)
ok("C ω_turn 连续==proj infimum + 离散==源派生(A_ω/T_M/Δlarge_turn·非字面0.012·改网格元数据必跟随)，L1a-1",
   abs(m["paper_narrative"]["omega_turn_continuous_rad_s"] - 0.008726646) < 1e-6
   and m["paper_narrative"]["omega_turn_discrete_grid_rad_s"] == _omega_turn_disc_exp
   and _omega_turn_disc_exp == 0.012)
ok("C action_range_caveat 反映 RANGE confound【已消除】(Fix② 后连续 RL 箱==离散 span·四方动作权限对齐·非声称 confound 存在)",
   "ELIMINATED" in m["paper_narrative"].get("action_range_caveat", "")
   and "MATCHED" in m["paper_narrative"].get("action_range_caveat", ""))
ok("C action_aliasing_metric_caveat 在(L1b-3 度量含 box 限幅·锚 keep 0.96 非 stub 0.48)", "box clip" in m["paper_narrative"].get("action_aliasing_metric_caveat", ""))
ok("C 诚实红线 3 条（违规不可称胜/档位A非provable/复现framing）", len(m["paper_narrative"]["honesty_red_lines"]) == 3)
ok("C Table III 5 指标列齐", len(m["paper_narrative"]["key_metrics_table3"]) == 5)
ok("C 四方有意差异 3 条", len(m["paper_narrative"]["four_party_intended_differences_only"]) == 3)
ok("C r_colregs 归一化偏离声明", "deviation" in m["reward_coefficients"]["r_colregs_meyer"]["deviation_note"].lower())
ok("C 投影标 档位A 经验性非 provable", "tier" in m["projection_params"] and "NOT provable" in m["projection_params"]["tier"])
ok("C vessel_params_full 内省非空", len(m["ship_dynamics"]["vessel_params_full"]) > 0)

print("\n===== C2) L1a补：库版本 + sb3 超参内省 + Krasowski 参考（live 守护）=====")
import importlib.metadata as _md
from trb_env.run_metadata import _num_or_repr as _nor
from stable_baselines3 import SAC as _SAC
from sb3_contrib import MaskablePPO as _MPPO
ok("C2 library_versions sb3 == importlib.metadata 实读(非硬编·升级跟随)",
   m["library_versions"]["stable-baselines3"] == _md.version("stable-baselines3"))
ok("C2 library_versions python/torch 非空", bool(m["library_versions"].get("python")) and bool(m["library_versions"].get("torch")))
ok("C2 sb3_hyperparameters SAC learning_starts == SAC.__init__ 默认(inspect 内省·改 sb3 跟随，非硬编)",
   m["sb3_hyperparameters"]["SAC"]["learning_starts"] == _nor(inspect.signature(_SAC.__init__).parameters["learning_starts"].default))
ok("C2 sb3_hyperparameters MaskablePPO n_steps == MaskablePPO.__init__ 默认(inspect 内省)",
   m["sb3_hyperparameters"]["MaskablePPO"]["n_steps"] == _nor(inspect.signature(_MPPO.__init__).parameters["n_steps"].default))
ok("C2 sb3_hyperparameters 标注 maker 覆盖项(gamma 在内)", any("gamma" in x for x in m["sb3_hyperparameters"]["overridden_by_maker"]))
ok("C2 Krasowski 参考有违规锚 Base 2.65 + 诚实 caveat(不可称胜)",
   m["krasowski_table3_reference"]["colregs_violations_per_episode"]["Base_unshielded"] == 2.65
   and "do NOT beat" in m["krasowski_table3_reference"]["honesty_note"])

print("\n===== D) write_run_metadata 落盘 + reload =====")
_tf = tempfile.NamedTemporaryFile(suffix=".json", delete=False).name
try:
    write_run_metadata(_tf, run_config={"tag": "_test", "seeds": [0, 1, 2], "n_total": 200, "pool_size": 2000})
    _loaded = json.load(open(_tf, encoding="utf-8"))
    ok("D reload run_config 正确", _loaded["run_config"]["n_total"] == 200 and _loaded["run_config"]["seeds"] == [0, 1, 2])
    ok("D reload 静态段保真", _loaded["ship_dynamics"]["length_m"] == 175.0 and _loaded["schema_version"] == 1)
    # D-ent：run_config 透传本 run 实际 ent/clip/norm/log_curves 配方（L59·真相源·passthrough 不丢）
    write_run_metadata(_tf, run_config={"tag": "_t2", "ent_start": 0.03, "ent_end": 0.005, "ent_frac": 0.6,
                                        "ent_schedule_discrete": "anneal", "clip_reward": 10.0, "norm_reward": True,
                                        "log_curves": True})
    _l2 = json.load(open(_tf, encoding="utf-8"))["run_config"]
    ok("D run_config 透传实际 ent/clip/norm/log_curves（L59·本 run 真相源·passthrough 保真）",
       _l2["ent_start"] == 0.03 and _l2["ent_end"] == 0.005 and _l2["ent_frac"] == 0.6
       and _l2["ent_schedule_discrete"] == "anneal" and _l2["clip_reward"] == 10.0
       and _l2["norm_reward"] is True and _l2["log_curves"] is True)
finally:
    os.path.exists(_tf) and os.unlink(_tf)

print()
if _fail == 0:
    print(f"✅ 全部 PASS ({_n}/{_n})")
else:
    print(f"❌ {_fail}/{_n} FAIL")
    sys.exit(1)
