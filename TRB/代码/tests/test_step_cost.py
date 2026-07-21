#!/usr/bin/env python3
"""修法C 每步生存成本 c_step 冒烟测试（`03` L123·非PBRS·连续臂专属）。

验：① c_step=0 与 baseline 逐位等价 bit-identical ② c_step>0 每步精确 −c_step 进 r_sparse·其它项不动
    ③ 负 c_step 拒 ④ 离散盾 ShieldedUSVEnv 硬拒 c_step ⑤ Base/RR UnshieldedUSVEnv 自然拒 ⑥ config_conflict 区分 c_step
    ⑦ 连续盾 ContinuousProjectionEnv 透传 c_step 到 RewardFunction。
运行：python 代码/tests/test_step_cost.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from trb_env.usv_reward import RewardFunction

GOAL = [5000.0, 1100.0]
INIT = [1000.0, 600.0]
ORI = (-0.17, 0.17)
N_FAIL = 0


def chk(cond, msg):
    global N_FAIL
    print(("  ✅ " if cond else "  ❌ ") + msg)
    if not cond:
        N_FAIL += 1


def _mk(c_step, well_b=0.0):
    return RewardFunction(GOAL, INIT, goal_orientation=ORI, well_shaping_weight=well_b,
                          shaping_radius=500.0, gamma=0.99, c_step=c_step)


# 一条逐步轨迹（ego=[x,y,psi,v]·从远到近·含各 term_flags）
TRAJ = [
    (np.array([1000.0, 600.0, 0.2, 6.7]), {}, False),
    (np.array([1500.0, 700.0, 0.1, 7.0]), {}, False),
    (np.array([3000.0, 900.0, 0.05, 8.0]), {}, True),    # emergency_used=True 这步
    (np.array([4800.0, 1090.0, 0.0, 3.0]), {"goal": True}, False),   # 进门
    (np.array([4000.0, 1000.0, 0.0, 2.0]), {"time": True}, False),   # 超时
]


def run_traj(rf):
    rf.reset(TRAJ[0][0])
    out = []
    for ego, tf, emer in TRAJ:
        total, parts = rf.step(ego, [], tf, emer)
        out.append((total, dict(parts)))
    return out


print("① c_step=0 与 baseline(默认) 逐位等价 bit-identical")
base = run_traj(_mk(0.0))
base_default = run_traj(RewardFunction(GOAL, INIT, goal_orientation=ORI, shaping_radius=500.0, gamma=0.99))
ok = all(abs(a[0] - b[0]) < 1e-12 and a[1].keys() == b[1].keys() and
         all(abs(a[1][k] - b[1][k]) < 1e-12 for k in a[1]) for a, b in zip(base, base_default))
chk(ok, "c_step=0.0 与 不传 c_step(默认) 全步 total/parts 逐位相同(<1e-12)")
chk(all("live" not in p[1] for p in base), "c_step=0 不引入 'live' 键(parts 结构不变)")

print("② c_step=0.5 每步精确 −0.5 进 r_sparse·其它项不动")
c5 = run_traj(_mk(0.5))
ok_sparse = all(abs((c[1]["sparse"]) - (b[1]["sparse"] - 0.5)) < 1e-12 for c, b in zip(c5, base))
ok_total = all(abs(c[0] - (b[0] - 0.5)) < 1e-12 for c, b in zip(c5, base))
ok_other = all(all(abs(c[1][k] - b[1][k]) < 1e-12 for k in ("colregs", "goal", "velocity", "deviate"))
               for c, b in zip(c5, base))
chk(ok_sparse, "每步 parts['sparse'] = baseline_sparse − 0.5（精确）")
chk(ok_total, "每步 total = baseline_total − 0.5（精确·每步无条件·不漏不重）")
chk(ok_other, "其它项 colregs/goal/velocity/deviate 一字不动（c_step 正交）")

print("③ c_step 与 well_B(PBRS) 正交·叠加不互扰")
b_wb = run_traj(_mk(0.0, well_b=200.0))
c_wb = run_traj(_mk(0.5, well_b=200.0))
ok_orth = all(abs(c[0] - (b[0] - 0.5)) < 1e-12 for c, b in zip(c_wb, b_wb))
chk(ok_orth, "well_B=200 下 c_step=0.5 仍精确每步 −0.5（与 PBRS shaping 正交）")

print("④ 负 c_step 拒")
try:
    _mk(-0.5); chk(False, "负 c_step 应 ValueError")
except ValueError:
    chk(True, "c_step=-0.5 → ValueError")

print("⑤ 离散盾 ShieldedUSVEnv 硬拒 c_step（连续臂专属·忠实 Krasowski）")
from trb_env.usv_shield import ShieldedUSVEnv, UnshieldedUSVEnv
try:
    ShieldedUSVEnv(None, None, c_step=0.5); chk(False, "ShieldedUSVEnv 应 TypeError 拒 c_step")
except TypeError as e:
    chk("c_step" in str(e), "ShieldedUSVEnv(c_step=0.5) → TypeError（硬拒·含 c_step 提示）")
except Exception as e:
    chk(False, f"ShieldedUSVEnv 拒 c_step 但异常类型非 TypeError: {type(e).__name__}")

print("⑥ Base/RR UnshieldedUSVEnv 自然拒 c_step（无 **kwargs）")
try:
    UnshieldedUSVEnv(None, None, c_step=0.5); chk(False, "UnshieldedUSVEnv 应 TypeError")
except TypeError:
    chk(True, "UnshieldedUSVEnv(c_step=0.5) → TypeError（自然拒·无 **kwargs）")

print("⑦ config_conflict 区分 c_step on/off（防续训/钱图混配置）")
from run_step4e import config_conflict
rec0 = [{"steps": 5000000, "n_total": 200, "c_step": 0.0}]
conf = config_conflict(rec0, 5000000, 200, c_step=0.5)
chk(len(conf) > 0, "已存 c_step=0 记录 + 当前 c_step=0.5 → 冲突（不混进同一 jsonl）")
conf_same = config_conflict(rec0, 5000000, 200, c_step=0.0)
chk(len(conf_same) == 0, "c_step=0 vs c_step=0(缺字段归一化) → 无冲突（旧记录兼容）")

print("⑧ 连续盾 ContinuousProjectionEnv 透传 c_step 到内层 RewardFunction")
from trb_env.usv_continuous_shield import ContinuousProjectionEnv
import inspect
sig = inspect.signature(ContinuousProjectionEnv.__init__).parameters
chk("c_step" in sig, "ContinuousProjectionEnv.__init__ 有 c_step 形参")
src = inspect.getsource(ContinuousProjectionEnv.__init__)
chk("c_step=c_step" in src, "ContinuousProjectionEnv 把 c_step 透传给内层 USVEnv")

print("\n" + ("=" * 50))
print("✅ 全部通过" if N_FAIL == 0 else f"❌ {N_FAIL} 项失败")
sys.exit(1 if N_FAIL else 0)
