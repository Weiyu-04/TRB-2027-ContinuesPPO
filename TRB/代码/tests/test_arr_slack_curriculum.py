"""
B1 到达门朝向容差课程（`03` L153）committed 测试
=================================================
锁定 B1（arrival_heading_slack 课程）的承重不变式，防回归/防误接线破诚实红线。
覆盖：
  A. TerminationChecker：slack=0 bit-identical / 加宽机制 / deepcopy 隔离真 goal / 宽度 clamp fail-fast / 退火回真门
  B. env 四层转发链（USVEnv → ContinuousProjectionEnv）：默认 bit-identical + 穿透 + 真 goal 不污染
  C. MultiScenarioEnv 双写：退火值扛得住 _inner 每 episode 重建（不静默失效）
  D. ArrivalSlackAnnealSchedule/Callback：量化退火 start→0（推送次数 ~n_levels 而非每步）
  E. 🔴🔴 eval 恒真门（诚实红线）：不传 arrival_heading_slack 的 env（=fac/eval 路径）恒用真 ±9.74° 门

跑法：/opt/miniconda3/envs/trb/bin/python -B 代码/tests/test_arr_slack_curriculum.py
（需 commonocean → trb conda env；T0 场景 download-or-skip，同 test_usv_env.py）
"""
import os
import sys
import urllib.request

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from commonroad.common.util import Interval, AngleInterval  # noqa: E402
from commonroad.geometry.shape import Rectangle  # noqa: E402
from commonroad.planning.goal import GoalRegion  # noqa: E402
from commonroad.scenario.state import CustomState  # noqa: E402

from trb_env.usv_termination import TerminationChecker  # noqa: E402
from trb_env.usv_sac_train import (  # noqa: E402
    ArrivalSlackAnnealSchedule, ArrivalSlackAnnealSyncCallback)

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
    print(f"[FAIL] {name}: 期望抛 {exc.__name__ if exc is not Exception else '异常'} 但没抛")
    _fail += 1


# 合成 goal（T-0 同构：朝东 ±0.17rad=±9.74° 真门）
GC = np.array([5000.0, 0.0])


def make_tc(lo=-0.17, hi=0.17):
    goal = GoalRegion([CustomState(
        position=Rectangle(length=400.0, width=60.0, center=GC, orientation=0.0),
        orientation=AngleInterval(lo, hi),
        time_step=Interval(0, 170),
    )])
    return TerminationChecker(goal, 170), goal


def reached(tc, theta):
    return tc.check([GC[0], GC[1], theta, 5.0], 10)[1]["goal"]


# ===================== A) TerminationChecker slack =====================
print("===== A) TerminationChecker 到达门朝向容差 slack =====")
tc, goal = make_tc()
# A1 默认 slack=0 = 真门 bit-identical
check("A1 默认 slack=0 → _goal_widened None（真门·bit-identical）", tc._goal_widened is None)
check("A1 slack=0 → 0.30rad 拒 / 0.0 收（官方真门）", (not reached(tc, 0.30)) and reached(tc, 0.0))
# A2 加宽机制：slack>0 放宽朝向门
tc.set_arrival_slack(0.30)
check("A2 slack=0.30 → 0.30rad 收（放宽生效）", reached(tc, 0.30))
check("A2 slack=0.30 → 0.50rad 仍拒（未过度放宽·±0.47 门外）", not reached(tc, 0.50))
# A3 🔴 deepcopy 隔离：真 goal 绝不被污染
o = goal.state_list[0].orientation
check("A3 真 goal orientation 未被污染（仍 ±0.17）",
      abs(float(o.start) + 0.17) < 1e-9 and abs(float(o.end) - 0.17) < 1e-9)
check("A3 真 goal 与加宽副本是不同对象（deepcopy 真隔离）",
      o is not tc._goal_widened.state_list[0].orientation)
# A4 退火回 0 → 真门恢复
tc.set_arrival_slack(0.0)
check("A4 退火回 slack=0 → _goal_widened None + 0.30rad 复拒（真门恢复）",
      tc._goal_widened is None and not reached(tc, 0.30))
# A5 宽度 clamp：加宽全宽 ≥ π 会崩 commonocean AngleInterval → setter fail-fast（非 cryptic assert）
check_raises("A5 slack=85°（全宽≥π）→ clamp 抛 ValueError（非 AssertionError）",
             lambda: tc.set_arrival_slack(np.radians(85)), ValueError)
tc.set_arrival_slack(0.0)  # 复位（clamp 抛异常不改状态·稳妥复位）
check("A5 45°（退火意图上限）不触 clamp（全宽~99°<π）",
      reached(make_tc()[0], 0.0) is not None)
mtc = make_tc()[0]
mtc.set_arrival_slack(np.radians(45))
check("A5 slack=45° → check 不崩且放宽（0.60rad 收）", reached(mtc, 0.60))
# A6 clamp 按每个 goal 真宽算：窄门允许更大 slack
ntc, _ = make_tc(-0.05, 0.05)
ntc.set_arrival_slack(np.radians(85))  # 窄门全宽 0.10+2*1.484=3.07<π → 不该崩
check("A6 窄门 ±0.05 + slack=85°（全宽<π）→ 不抛（clamp 按真宽算）",
      ntc._goal_widened is not None)
# A7 非法 slack 守卫
check_raises("A7 slack=-0.1（负）→ ValueError", lambda: make_tc()[0].set_arrival_slack(-0.1), ValueError)
check_raises("A7 slack=nan → ValueError", lambda: make_tc()[0].set_arrival_slack(float("nan")), ValueError)


# ===================== D) 退火 schedule + callback =====================
print("\n===== D) ArrivalSlackAnnealSchedule / Callback（量化退火）=====")
START = np.radians(45)
ANNEAL = 100_000
s = ArrivalSlackAnnealSchedule(START, ANNEAL, n_levels=20)
s.num_timesteps = 0
check("D1 t=0 → value≈start(45°)", abs(s.value() - START) < 1e-9)
s.num_timesteps = ANNEAL
check("D2 t=anneal_steps → 0（真门后段）", s.value() == 0.0)
s.num_timesteps = int(ANNEAL * 2)
check("D2 t>anneal → 恒 0（真门后段收敛真精度）", s.value() == 0.0)
# 单调 + 量化（推送次数 ~n_levels 而非每步）
vals = []
for t in range(0, ANNEAL + 1, max(1, ANNEAL // 500)):
    s.num_timesteps = t
    vals.append(s.value())
distinct = sorted(set(round(v, 9) for v in vals))
check("D3 单调非增", all(vals[i] >= vals[i + 1] - 1e-12 for i in range(len(vals) - 1)))
check(f"D3 量化档数 ~n_levels（实测 {len(distinct)}·≤22）", len(distinct) <= 22)
check("D3 所有值 ∈[0, start]（不越界·clamp 上游安全）", all(-1e-12 <= v <= START + 1e-12 for v in vals))


# 无场景则 A/D 段已覆盖核心逻辑；B/C/E 段需真场景（env 构造）
T0 = "/tmp/trb_T0.xml"
if not os.path.exists(T0):
    try:
        urllib.request.urlretrieve(
            "https://gitlab.lrz.de/tum-cps/commonocean-scenarios/-/raw/main/"
            "scenarios/HandcraftedTwoVesselEncounters_01_24/ZAM_AAA-1_20240121_T-0.xml", T0)
    except Exception as e:  # noqa: BLE001
        print(f"\n[SKIP] 无法获取 T-0 场景（{type(e).__name__}）→ B/C/E 段（env 集成 + eval 恒真门）跳过")
        print(f"\n{'='*40}\n{'✅ 全部 PASS（A/D 段·env 段 SKIP）' if _fail == 0 else f'❌ {_fail} 项 FAIL'}")
        sys.exit(1 if _fail else 0)

from commonocean.common.file_reader import CommonOceanFileReader  # noqa: E402
from trb_env.usv_env import USVEnv  # noqa: E402
from trb_env.usv_continuous_shield import ContinuousProjectionEnv  # noqa: E402
from trb_env.usv_scenarios import MultiScenarioEnv  # noqa: E402

_sc, _pps = CommonOceanFileReader(T0).open()
_pp = list(_pps.planning_problem_dict.values())[0]


def _tc_of(env):
    for path in (lambda e: e.term_checker, lambda e: e.env.term_checker,
                 lambda e: e._inner.env.term_checker):
        try:
            return path(env)
        except Exception:
            continue
    return None


# ===================== B) env 转发链 =====================
print("\n===== B) env 四层转发链（USVEnv → ContinuousProjectionEnv）=====")
e_def = USVEnv(_sc, _pp, continuous=True)
check("B1 USVEnv 默认（不传 slack）→ 内层 term_checker 真门（bit-identical）",
      e_def.term_checker._goal_widened is None)
e_def.set_arrival_slack(0.30)
check("B2 USVEnv.set_arrival_slack(0.30) → 转发到 term_checker 加宽",
      e_def.term_checker._goal_widened is not None)
cpe_def = ContinuousProjectionEnv(_sc, _pp)
check("B3 ContinuousProjectionEnv 默认 → 内层真门 + obs Box(27)（bit-identical）",
      _tc_of(cpe_def)._goal_widened is None and cpe_def.observation_space.shape == (27,))
cpe = ContinuousProjectionEnv(_sc, _pp, arrival_heading_slack=0.30)
check("B4 ContinuousProjectionEnv(arrival_heading_slack=0.30) → 穿两层到 term_checker 加宽",
      _tc_of(cpe)._goal_widened is not None)
cpe.set_arrival_slack(0.0)
check("B4 ContinuousProjectionEnv.set_arrival_slack(0) → 真门恢复", _tc_of(cpe)._goal_widened is None)
_o = _tc_of(cpe).goal.state_list[0].orientation
check("B5 真 goal 未污染（窄门·评估安全命门）", (float(_o.end) - float(_o.start)) < 0.4)


# ===================== C) MultiScenarioEnv 双写扛重建 =====================
print("\n===== C) MultiScenarioEnv 双写（退火扛 _inner 每 episode 重建）=====")
m = MultiScenarioEnv([(_sc, _pp)], env_cls=ContinuousProjectionEnv)
check("C1 默认 env_kwargs 无 arrival_heading_slack 键（eval 路径=真门）",
      "arrival_heading_slack" not in m.env_kwargs)
m.set_arrival_slack(0.30)
check("C2 set_arrival_slack(0.30) → env_kwargs 记 + 当前 _inner 穿透加宽",
      m.env_kwargs.get("arrival_heading_slack") == 0.30 and _tc_of(m._inner)._goal_widened is not None)
m.reset(seed=0)
check("C3 🔴 reset 重建 _inner 后 slack 仍在（env_kwargs 继承·退火不静默失效）",
      _tc_of(m._inner)._goal_widened is not None)
m.reset(seed=1)
_om = _tc_of(m._inner).goal.state_list[0].orientation
check("C4 多次重建后真 goal 仍未污染", (float(_om.end) - float(_om.start)) < 0.4)
m.set_arrival_slack(0.0)
m.reset(seed=2)
check("C5 退火回 0 + 重建 → 真门（_goal_widened None）", _tc_of(m._inner)._goal_widened is None)
check_raises("C6 set_arrival_slack(nan) → ValueError（挡 IPC 毒化）",
             lambda: m.set_arrival_slack(float("nan")), ValueError)


# ===================== E) 🔴🔴 eval 恒真门（诚实红线）=====================
print("\n===== E) 🔴🔴 eval 恒真门（诚实红线·训练放宽/评估不放水）=====")
# eval 路径 = 不传 arrival_heading_slack 构造的 ContinuousProjectionEnv（run_step4e fac / replay_eval 就这么建）。
# 即便训练 env 被放宽，eval env 必须恒用真 ±9.74° 门。此测锁死：默认构造 = 真门 + 真门外朝向被拒。
eval_env = ContinuousProjectionEnv(_sc, _pp)  # = fac(sc,pp) 的构造（无 slack）
etc = _tc_of(eval_env)
check("E1 eval 路径（无 slack 构造）→ term_checker.arrival_heading_slack == 0",
      etc.arrival_heading_slack == 0.0)
check("E1 eval 路径 → _goal_widened is None（用官方真 goal）", etc._goal_widened is None)
# E2 真门外的朝向（0.30rad=17.2° > 9.74°）在 eval env 必须被拒（不放水）
# 用 eval_env 的真 goal 中心构造刚好在框内、朝向超真门的 state
_gc_real = np.array(etc.goal.state_list[0].position.center, dtype=float)
in_box_wrong_heading = [_gc_real[0], _gc_real[1], 0.30, 5.0]   # 位置∈框·朝向 0.30rad 超 ±0.17 真门
in_box_right_heading = [_gc_real[0], _gc_real[1], 0.0, 5.0]    # 位置∈框·朝向 0（正东·真门内）
check("E2 eval env：框内但朝向 0.30rad（超真门）→ 到达=False（不放水）",
      etc.check(in_box_wrong_heading, 10)[1]["goal"] is False)
check("E2 eval env：框内朝向 0（真门内）→ 到达=True（真门正常收）",
      etc.check(in_box_right_heading, 10)[1]["goal"] is True)
# E3 变异自证：若有人误把 slack 接进 eval env（构造传 slack>0），此断言会翻——守卫在此
bad_eval = ContinuousProjectionEnv(_sc, _pp, arrival_heading_slack=0.30)  # 模拟【错误】的 eval 接线
check("E3 变异自证：eval env 若误传 slack=0.30 → 框内 0.30rad 会被误收（=红线被破的信号）",
      _tc_of(bad_eval).check(in_box_wrong_heading, 10)[1]["goal"] is True)
print("     ↑ E3 说明：正确的 eval 路径【绝不】给 ContinuousProjectionEnv 传 arrival_heading_slack（fac/replay_eval 已守）。")

print(f"\n{'='*40}")
print('✅ 全部 PASS' if _fail == 0 else f'❌ {_fail} 项 FAIL')
sys.exit(1 if _fail else 0)
