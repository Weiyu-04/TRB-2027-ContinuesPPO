#!/usr/bin/env python3
"""STEP4E_COLREGS_WEIGHT / colregs_weight 旋钮冒烟测试（连续臂 SAC + PPO）。

背景：连续臂此前把 r_colregs 权重 colregs_weight 硬编 0.0（丢 r_colregs·合规靠投影约束·D37-B），
并用 5 处守卫强制 ==0.0（probe/caliber）。本改把硬编换成【意图参数】colregs_weight（默认 0.0=保 bit-identical·
A/B 传 1.0 复活 Meyer 式26），守卫从"必须==0.0"改成"必须==意图值"（抓 silent no-op 漏接线）。

验：
  T1 bit-identical: 不传 colregs_weight(默认0.0) → SAC/PPO 构造成功·probe/caliber 不 raise·env.reward_fn.colregs_weight==0.0。
  T2 真进 reward:   传 colregs_weight=1.0 → SAC/PPO 构造成功(probe/caliber 认 1.0 不 raise)·venv 内 env.reward_fn.colregs_weight==1.0。
  T3 silent-no-op 捕获: 手造 colregs_weight=0.0 的 venv 但传意图 1.0 给 caliber → 必 raise AssertionError（"接线未落地"被抓）。

运行：cd 代码 && /opt/miniconda3/envs/trb/bin/python tests/test_colregs_weight_knob.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from trb_env.usv_sac_train import (
    make_continuous_safe_model, make_continuous_safe_ppo_model,
    assert_continuous_safe_caliber, assert_continuous_safe_ppo_caliber,
)

N_FAIL = 0


def chk(cond, msg):
    global N_FAIL
    print(("  ✅ " if cond else "  ❌ ") + msg)
    if not cond:
        N_FAIL += 1


# --- 场景 paths（取本地池前 2 个·避免联网）；池不在则跳整测（非代码回归·如上游 harness）---
_POOL_DIR = "/private/tmp/trb_scenarios_pool"
_PATHS = []
if os.path.isdir(_POOL_DIR):
    _PATHS = sorted(
        os.path.join(_POOL_DIR, f) for f in os.listdir(_POOL_DIR) if f.endswith(".xml")
    )[:2]

if not _PATHS:
    print(f"[SKIP] 本地场景池 {_POOL_DIR} 不存在或无 xml → 跳过冒烟（非代码回归·需真场景构 env）")
    sys.exit(0)

# 小 buffer/学习步只为验集成（不真训练）；subproc=False → DummyVecEnv 便于 get_attr('env') 在进程内读
_SK = dict(buffer_size=2000, learning_starts=10, batch_size=32, subproc=False)


def _read_env_cw(venv):
    """从连续臂 venv 读回每个子 env 的 reward_fn.colregs_weight（DummyVecEnv/Subproc 都用 get_attr('env')·同 caliber 口径）。"""
    return [float(e.reward_fn.colregs_weight) for e in venv.venv.get_attr("env")]


# ==================================================================================
print("T1 bit-identical: 不传 colregs_weight(默认0.0) → 构造成功·probe/caliber 不 raise·env cw==0.0")
# --- SAC ---
try:
    m1s, v1s = make_continuous_safe_model(paths=_PATHS, n_envs=1, seed=0, **_SK)
    chk(True, "T1-SAC make_continuous_safe_model(默认 colregs_weight) 构造成功（probe/caliber 未 raise）")
    cws = _read_env_cw(v1s)
    chk(all(c == 0.0 for c in cws), f"T1-SAC env.reward_fn.colregs_weight 全 0.0（bit-identical 现状），得 {cws}")
    v1s.close()
except Exception as e:
    chk(False, f"T1-SAC 默认构造不该 raise，但抛 {type(e).__name__}: {e}")
# --- PPO ---
try:
    m1p, v1p = make_continuous_safe_ppo_model(paths=_PATHS, n_envs=1, seed=0, subproc=False)
    chk(True, "T1-PPO make_continuous_safe_ppo_model(默认 colregs_weight) 构造成功（probe/caliber 未 raise）")
    cwp = _read_env_cw(v1p)
    chk(all(c == 0.0 for c in cwp), f"T1-PPO env.reward_fn.colregs_weight 全 0.0（bit-identical 现状），得 {cwp}")
    v1p.close()
except Exception as e:
    chk(False, f"T1-PPO 默认构造不该 raise，但抛 {type(e).__name__}: {e}")


# ==================================================================================
print("T2 真进 reward: 传 colregs_weight=1.0 → 构造成功(probe/caliber 认 1.0)·env cw==1.0")
# --- SAC ---
try:
    m2s, v2s = make_continuous_safe_model(paths=_PATHS, n_envs=1, seed=0, colregs_weight=1.0, **_SK)
    chk(True, "T2-SAC make_continuous_safe_model(colregs_weight=1.0) 构造成功（probe/caliber 认 1.0·未 raise）")
    cws2 = _read_env_cw(v2s)
    chk(all(c == 1.0 for c in cws2), f"T2-SAC env.reward_fn.colregs_weight 全 1.0（r_colregs 真复活进 reward），得 {cws2}")
    v2s.close()
except Exception as e:
    chk(False, f"T2-SAC colregs_weight=1.0 构造不该 raise，但抛 {type(e).__name__}: {e}")
# --- PPO ---
try:
    m2p, v2p = make_continuous_safe_ppo_model(paths=_PATHS, n_envs=1, seed=0, subproc=False, colregs_weight=1.0)
    chk(True, "T2-PPO make_continuous_safe_ppo_model(colregs_weight=1.0) 构造成功（probe/caliber 认 1.0·未 raise）")
    cwp2 = _read_env_cw(v2p)
    chk(all(c == 1.0 for c in cwp2), f"T2-PPO env.reward_fn.colregs_weight 全 1.0（r_colregs 真复活进 reward），得 {cwp2}")
    v2p.close()
except Exception as e:
    chk(False, f"T2-PPO colregs_weight=1.0 构造不该 raise，但抛 {type(e).__name__}: {e}")


# ==================================================================================
print("T3 silent-no-op 捕获: 手造 cw=0.0 的 venv 但传意图 1.0 给 caliber → 必 raise AssertionError")
# 造一个 colregs_weight=0.0 的合法 venv（=env_kwargs 未落地 1.0 的模拟），然后拿意图 1.0 去校验：
# caliber 读回 env cw=0.0 ≠ 意图 1.0 → 应 raise（证"接线未落地/silent no-op"被抓）。
# --- SAC caliber ---
m3s, v3s = make_continuous_safe_model(paths=_PATHS, n_envs=1, seed=0, **_SK)   # 真 env cw=0.0
try:
    assert_continuous_safe_caliber(m3s, v3s, colregs_weight=1.0)   # 意图 1.0 但 env 是 0.0 → 须 raise
    chk(False, "T3-SAC assert_continuous_safe_caliber(意图 1.0 vs env 0.0) 应 raise AssertionError（未 raise=silent no-op 漏网）")
except AssertionError as e:
    chk("接线未落地" in str(e) or "colregs_weight" in str(e),
        f"T3-SAC caliber 抓到 silent no-op（env 0.0 ≠ 意图 1.0）→ AssertionError: {e}")
except Exception as e:
    chk(False, f"T3-SAC 应 AssertionError 但抛 {type(e).__name__}: {e}")
# 正对照：意图 0.0 vs env 0.0 → 不该 raise（守卫不误伤 bit-identical 现状）
try:
    assert_continuous_safe_caliber(m3s, v3s, colregs_weight=0.0)
    chk(True, "T3-SAC 正对照：caliber(意图 0.0 vs env 0.0) 不 raise（守卫不误伤现状）")
except Exception as e:
    chk(False, f"T3-SAC 正对照不该 raise，但抛 {type(e).__name__}: {e}")
v3s.close()
# --- PPO caliber ---
m3p, v3p = make_continuous_safe_ppo_model(paths=_PATHS, n_envs=1, seed=0, subproc=False)   # 真 env cw=0.0
try:
    assert_continuous_safe_ppo_caliber(m3p, v3p, colregs_weight=1.0)
    chk(False, "T3-PPO assert_continuous_safe_ppo_caliber(意图 1.0 vs env 0.0) 应 raise AssertionError（未 raise=silent no-op 漏网）")
except AssertionError as e:
    chk("接线未落地" in str(e) or "colregs_weight" in str(e),
        f"T3-PPO caliber 抓到 silent no-op（env 0.0 ≠ 意图 1.0）→ AssertionError: {e}")
except Exception as e:
    chk(False, f"T3-PPO 应 AssertionError 但抛 {type(e).__name__}: {e}")
try:
    assert_continuous_safe_ppo_caliber(m3p, v3p, colregs_weight=0.0)
    chk(True, "T3-PPO 正对照：caliber(意图 0.0 vs env 0.0) 不 raise（守卫不误伤现状）")
except Exception as e:
    chk(False, f"T3-PPO 正对照不该 raise，但抛 {type(e).__name__}: {e}")
v3p.close()


print("\n" + ("=" * 50))
print("✅ 全部通过" if N_FAIL == 0 else f"❌ {N_FAIL} 项失败")
sys.exit(1 if N_FAIL else 0)
