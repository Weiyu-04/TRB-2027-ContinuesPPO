"""
env 接线集成冒烟测试 —— 用真实 T-0 场景实跑完整 episode，验证 4 件拼起来能跑。
跑：/opt/miniconda3/envs/trb/bin/python 代码/tests/test_usv_env.py

需要 /tmp/trb_T0.xml（不在则自动下载；下载失败则 SKIP，env 测试需真实场景）。
"""
import os
import sys
import urllib.request

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from trb_env.usv_env import (USVEnv, N_DISCRETE, DISCRETE_ACTIONS,  # noqa: E402
                             IDX_EMERGENCY, N_ACTIONS_TOTAL)
from trb_env.usv_observation import OBS_DIM  # noqa: E402

_fail = 0


def check(name, cond):
    global _fail
    ok = bool(cond)
    if not ok:
        _fail += 1
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")


def check_raises(name, fn, exc=Exception):
    global _fail
    try:
        fn()
    except exc:
        print(f"[PASS] {name}: 正确抛 {exc.__name__ if exc is not Exception else '异常'}")
        return
    print(f"[FAIL] {name}: 期望抛异常但没抛")
    _fail += 1


T0 = "/tmp/trb_T0.xml"
if not os.path.exists(T0):
    try:
        urllib.request.urlretrieve(
            "https://gitlab.lrz.de/tum-cps/commonocean-scenarios/-/raw/main/"
            "scenarios/HandcraftedTwoVesselEncounters_01_24/ZAM_AAA-1_20240121_T-0.xml", T0)
    except Exception as e:  # noqa: BLE001
        print(f"[SKIP] 无法获取 T-0 场景（{type(e).__name__}）→ env 集成测试需真实场景，跳过")
        sys.exit(0)

from commonocean.common.file_reader import CommonOceanFileReader  # noqa: E402

sc, pps = CommonOceanFileReader(T0).open()
pp = list(pps.planning_problem_dict.values())[0]


def fresh_env(**kw):
    sc2, pps2 = CommonOceanFileReader(T0).open()
    return USVEnv(sc2, list(pps2.planning_problem_dict.values())[0], **kw)


print("===== A) reset =====")
env = fresh_env()
obs, info = env.reset()
check("① reset obs 维度 27", obs.shape == (OBS_DIM,))
check("② reset obs 全有限", np.all(np.isfinite(obs)))
check("③ reset info 有 step=0", info.get("step") == 0)

print("\n===== B) 单步 step（连续）=====")
obs, reward, terminated, truncated, info = env.step([0.02, 0.0])
check("④ step obs 维度 27 全有限", obs.shape == (OBS_DIM,) and np.all(np.isfinite(obs)))
check("⑤ reward 有限", np.isfinite(reward))
check("⑥ terminated/truncated 是 bool", isinstance(terminated, bool) and isinstance(truncated, bool))
check("⑦ info 含 flags/reward_parts/step", {"flags", "reward_parts", "step"} <= set(info.keys()))
check("⑧ reward == 分量之和", abs(reward - sum(info["reward_parts"].values())) < 1e-9)

print("\n===== C) 完整 episode 实跑（≤k_max=170 步内结束，全程有限）=====")
env = fresh_env()
env.reset()
ended_step, end_flags, all_finite = None, None, True
end_term, end_trunc = None, None
for k in range(1, 171):
    obs, reward, terminated, truncated, info = env.step([0.0, 0.0])   # 滑行
    if not (np.all(np.isfinite(obs)) and np.isfinite(reward)):
        all_finite = False
    if terminated or truncated:
        ended_step, end_flags = k, info["flags"]
        end_term, end_trunc = terminated, truncated
        break
check("⑨ episode 在 ≤170 步内结束", ended_step is not None and ended_step <= 170)
check("⑩ 全程 obs/reward 有限", all_finite)
check("⑪ 结束时至少一个终止 flag 为真", end_flags is not None and any(end_flags.values()))
print(f"      （结束于 step {ended_step}，flags={ {k: v for k, v in (end_flags or {}).items() if v} }）")
# FG9（全面审计补漏）：超时局 → gymnasium 语义 truncated=True ∧ terminated=False（Phase3 SB3 bootstrapping 依赖此拆分）。
#   旧测试⑥只查 isinstance bool、⑪只查"任一flag真"，删 terminated/truncated 拆分逻辑不翻 FAIL（假守护，03 L12）。
check("⑪b 超时局 → truncated=True ∧ terminated=False（gymnasium 语义，FG9）",
      bool(end_flags and end_flags.get("time")) and end_trunc is True and end_term is False)

print("\n===== D) v_max=9.5 强制（clip_velocity 默认 True）=====")
env = fresh_env()
env.reset()
vmax_ok = True
for _ in range(30):
    obs, *_ = env.step([0.24, 0.0])     # 一直最大加速
    if env.ego[3] > 9.5 + 1e-9:
        vmax_ok = False
check("⑫ 一直加速 v 仍 ≤ 9.5", vmax_ok and abs(env.ego[3] - 9.5) < 1e-6)
# 关掉 clip → v 应能超 9.5（验证开关真生效，非永远限速）
env2 = fresh_env(clip_velocity=False)
env2.reset()
for _ in range(30):
    env2.step([0.24, 0.0])
check("⑬ 关 clip_velocity → v 能超 9.5（开关真生效）", env2.ego[3] > 9.5)

print("\n===== E) 离散动作模式 =====")
envd = fresh_env(continuous=False)
envd.reset()
check("⑭ 动作空间 n=50（49 regular + a_em 槽位）",
      envd.action_space.n == N_ACTIONS_TOTAL == 50 and N_DISCRETE == IDX_EMERGENCY == 49)
obs, reward, *_ = envd.step(24)         # 下标 24（中间动作 = (0,0)）
check("⑮ 离散 step 跑通 obs 有限", obs.shape == (OBS_DIM,) and np.all(np.isfinite(obs)))
check("⑯ 下标 24 = (0,0)（网格中点）", DISCRETE_ACTIONS[24] == (0.0, 0.0))
check_raises("⑰ 槽位 49 不传 emergency_action → 报错", lambda: envd.step(49), ValueError)
check_raises("⑰b 离散下标越界(50) → 报错", lambda: envd.step(50), ValueError)
# ⑰c a_em 槽位接线：49 + emergency_action 跑通、emergency_used 进 reward 通路、值反映实际施加
obs_em, r_em, *_rest = envd.step(IDX_EMERGENCY, emergency_action=(0.048, 0.0))
check("⑰c 槽位 49 + emergency_action 跑通且 obs a_ego=0.048",
      np.all(np.isfinite(obs_em)) and abs(obs_em[2] - 0.048) < 1e-9)
check_raises("⑰d emergency_action 含 NaN → 报错",
             lambda: envd.step(IDX_EMERGENCY, emergency_action=(np.nan, 0.0)), ValueError)

print("\n===== F) 确定性复现（同动作 → 同轨迹）=====")
e1, e2 = fresh_env(), fresh_env()
e1.reset(); e2.reset()
acts = [[0.01, 0.005], [-0.02, -0.01], [0.0, 0.012], [0.03, 0.0], [0.0, 0.0]]
det_ok = True
for a in acts:
    o1, r1, *_ = e1.step(a)
    o2, r2, *_ = e2.step(a)
    if not (np.allclose(e1.ego, e2.ego, atol=1e-12) and np.array_equal(o1, o2) and r1 == r2):
        det_ok = False
check("⑱ 两 env 同动作 → ego+obs+reward 逐步逐字节一致", det_ok)

print("\n===== G) 碰撞集成（env 取的他船占据能触发碰撞）=====")
env = fresh_env()
env.reset()
states, footprints = env._obstacles_at(1)            # step1 他船 state + 占据
check("⑲ env 取到他船占据", len(footprints) >= 1)
# 把 ego 摆到他船位置 → 终止件应判 collision（验证 env→终止件的占据流对）
ob_pos = states[0][1]
done, flags = env.term_checker.check([ob_pos[0], ob_pos[1], 0.0, 5.0], 1, footprints)
check("⑳ ego 摆到他船处 → collision True", flags["collision"])

print("\n===== H) fail-fast =====")
check_raises("㉑ step 前未 reset → 报错", lambda: fresh_env().step([0, 0]), RuntimeError)
env = fresh_env(); env.reset()
check_raises("㉒ 连续动作含 NaN → 报错", lambda: env.step([np.nan, 0]), ValueError)
check_raises("㉓ 连续动作维度错 → 报错", lambda: env.step([0, 0, 0]), ValueError)

print("\n===== I) 复核加固（MINOR 修复验证）=====")
# MINOR-1：越界连续动作 → obs 的 a_ego/ω_ego 反映"实际施加"(截断后)值，非指令值
env = fresh_env(); env.reset()
obs, *_ = env.step([100.0, 100.0])     # 远超 a_max=0.24 / w_max=0.03
check("㉔ 越界动作 → obs a_ego=0.24(a_max)/ω_ego=0.03(w_max)（反映实际施加）",
      abs(obs[2] - 0.24) < 1e-9 and abs(obs[3] - 0.03) < 1e-9)
# MINOR-2：离散非整数下标拒绝（不静默截断）
envd2 = fresh_env(continuous=False); envd2.reset()
check_raises("㉕ 离散非整数下标(24.9) → 报错", lambda: envd2.step(24.9), ValueError)

print("\n===== J) 全面审计补漏：env→reward 他船速度通路（FG8）=====")
# env 把他船速度向量 vel_vec=spd·[cos,sin] 喂 reward 算 r_colregs(v_y=径向速度)。旧测试无一钉过此通路数值
#   → 归零 vel_vec（FG8 变异）不破坏任何测试(假守护，03 L12)。T-0 episode step54 他船最逼近 → colregs=−2.04
#   显著依赖他船径向速度；若 vel_vec 归零则 v_y=0、exp 项→~0、colregs→~0（断言<−1 会翻 FAIL，真守护）。
envj = fresh_env(); envj.reset()
_cl54 = None
for _k in range(1, 55):
    _, _, _, _, _infoj = envj.step([0.0, 0.0])
    if _k == 54:
        _cl54 = _infoj["reward_parts"]["colregs"]
check("㉖ step54 r_colregs 显著非零(<−1，依赖他船速度通路，FG8)", _cl54 is not None and _cl54 < -1.0)

print("\n===== K) gymnasium.Env 正式化（step4b）=====")
import gymnasium as gym
from gymnasium import spaces as gspaces
envk = fresh_env(); envk2 = fresh_env(continuous=False)
check("㉗ USVEnv 是 gymnasium.Env 实例", isinstance(envk, gym.Env))
check("㉘ observation_space = Box(27,) float64 无界",
      isinstance(envk.observation_space, gspaces.Box) and envk.observation_space.shape == (OBS_DIM,)
      and envk.observation_space.dtype == np.float64)
check("㉙ 连续 action_space = Box(2,) 边界=±a_max/±w_max",
      isinstance(envk.action_space, gspaces.Box) and envk.action_space.shape == (2,)
      and np.allclose(envk.action_space.high, [0.24, 0.03]) and np.allclose(envk.action_space.low, [-0.24, -0.03]))
check("㉚ 离散 action_space = Discrete(50)",
      isinstance(envk2.action_space, gspaces.Discrete) and envk2.action_space.n == N_ACTIONS_TOTAL)
obsk, infok = envk.reset(seed=0)
check("㉛ reset(seed=0) 返回 obs ∈ observation_space（contains）", envk.observation_space.contains(obsk))
check("㉜ reset(seed) 播种 np_random（gymnasium 契约）", envk.np_random is not None)
# 确定性环境：obs 由固定场景决定、不依赖 seed → 同 seed reset 逐字节一致（验证 seed 透传不破坏确定性）
oa, _ = fresh_env().reset(seed=42); ob, _ = fresh_env().reset(seed=42)
check("㉝ 同 seed reset → obs 逐字节一致（确定性环境）", np.array_equal(oa, ob))
# action_space.sample() 落在 spaces 内（连续 + 离散）
check("㉞ 连续 action_space.sample() ∈ Box", envk.action_space.contains(envk.action_space.sample()))
check("㉟ 离散 action_space.sample() ∈ [0,50)", 0 <= int(envk2.action_space.sample()) < N_ACTIONS_TOTAL)
check("㊱ render() 返回 None 不抛（render_mode=None gymnasium 推荐）", envk.render() is None)

# ㊲㊳ emergency_used 连续路径参数（2026-06-17b D39/L43-续①：四方紧急惩罚口径对齐——连续投影盾经此传 source=='emergency'）
_e_def = fresh_env(continuous=True); _e_def.reset(seed=0)
_, _, _, _, _i_def = _e_def.step(np.array([0.05, 0.02]))                      # 默认 emergency_used=False（向后兼容）
_e_em = fresh_env(continuous=True); _e_em.reset(seed=0)
_, _, _, _, _i_em = _e_em.step(np.array([0.05, 0.02]), emergency_used=True)   # 显式 True
check("㊲ 连续 step(emergency_used=True) → reward sparse 含 C_EMERGENCY、与默认差恰 −0.5",
      abs((_i_em["reward_parts"]["sparse"] - _i_def["reward_parts"]["sparse"]) - (-0.5)) < 1e-9)
check("㊳ 连续 step 默认(不传 emergency_used) → sparse 不含 −0.5（向后兼容位级、离散 idx49 口径不受影响）",
      abs(_i_def["reward_parts"]["sparse"]) < 1e-9)

print("\n" + ("✅ 全部 PASS" if _fail == 0 else f"❌ {_fail} 项 FAIL"))
sys.exit(1 if _fail else 0)
