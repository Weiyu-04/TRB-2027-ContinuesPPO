"""多场景训练 env 冒烟测试（step4d-③a）。
跑：/opt/miniconda3/envs/trb/bin/python -B 代码/tests/test_usv_scenarios.py
需场景子集（T-0/1/2/500/1000 于 /tmp/trb_scenarios，缺则下载）；离线 SKIP。
"""
import sys
import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

_fail = 0
def check(name, ok):
    global _fail
    if not ok:
        _fail += 1
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")

_SDIR = "/tmp/trb_scenarios"
_IDS = [0, 1, 2, 500, 1000]
_BASE = ("https://gitlab.lrz.de/tum-cps/commonocean-scenarios/-/raw/main/scenarios/"
         "HandcraftedTwoVesselEncounters_01_24/ZAM_AAA-1_20240121_T-{}.xml")
try:
    import urllib.request
    os.makedirs(_SDIR, exist_ok=True)
    _paths = []
    for _n in _IDS:
        _dst = f"{_SDIR}/T-{_n}.xml"
        if not os.path.exists(_dst):
            urllib.request.urlretrieve(_BASE.format(_n), _dst)
        _paths.append(_dst)
    from trb_env.usv_scenarios import load_scenario_pool, MultiScenarioEnv, make_vec_env
    from trb_env.usv_shield import UnshieldedUSVEnv, ShieldedUSVEnv
    from trb_env.evaluate import run_episode
    _pool = load_scenario_pool(_paths)
except Exception as e:                                     # noqa: BLE001
    print(f"[SKIP] 需场景子集 + 联网/sb3（{type(e).__name__}: {e}）")
    sys.exit(0)


def _keep(o, m):                                           # keep-heading（ρ5 退第一合法）
    return 24 if m[24] else int(np.where(m)[0][0])


print("===== A) load_scenario_pool 预加载进内存 =====")
check("① 预加载池 = 5 个", len(_pool) == 5)
check("② 元素是 (scenario, planning_problem)", all(hasattr(sc, "dynamic_obstacles") for sc, _ in _pool))

print("===== B) MultiScenarioEnv reset 抽场景 + 委托 =====")
_me = MultiScenarioEnv(_pool)
_o1, _i1 = _me.reset(seed=7)
_o2, _i2 = _me.reset(seed=7)
check("③ 同 seed → 同 scenario_idx + 同首 obs（确定性）",
      _i1["scenario_idx"] == _i2["scenario_idx"] and np.array_equal(_o1, _o2))
_idxs = {MultiScenarioEnv(_pool).reset(seed=_s)[1]["scenario_idx"] for _s in range(40)}
check(f"④ 多 seed 覆盖池中多个场景（抽到 {len(_idxs)} 个不同 idx，非恒定一个）", len(_idxs) >= 3)
_me.reset(seed=3)
check("⑤ 委托 action_masks(50)/_ego_vs/_obs_vs/.env.dt 通",
      len(_me.action_masks()) == 50 and _me._ego_vs() is not None and _me.env.dt == 10.0)
check("⑥ MultiScenarioEnv._ego_vs == 内层选中场景 ego（委托正确）",
      np.allclose(_me._ego_vs().position, _me._inner._ego_vs().position))

print("===== C) run_episode 跑多场景 env =====")
_res = run_episode(MultiScenarioEnv(_pool), _keep)
check("⑦ run_episode(MultiScenarioEnv Shielded) 跑通 + 字段自洽",
      _res["steps"] >= 1 and _res["ep_len_s"] == _res["steps"] * 10.0
      and 0 <= _res["emergency_pct"] <= 100 and _res["violations"] >= 0)
_meu = MultiScenarioEnv(_pool, env_cls=UnshieldedUSVEnv, env_kwargs={"colregs_weight": 0.0})
_meu.reset(seed=0)
check("⑧ MultiScenarioEnv(Unshielded, colregs_weight=0) → 内层 reward_fn==0 + mask 全49（Base 多场景）",
      _meu._inner.env.reward_fn.colregs_weight == 0.0
      and _meu.action_masks()[:49].all() and not _meu.action_masks()[49])

print("===== D) make_vec_env（DummyVecEnv 并行采样）=====")
_vec = make_vec_env(_pool, n_envs=3)
_obs = _vec.reset()
check("⑨ DummyVecEnv n_envs=3：reset → obs shape (3,27)", _obs.shape == (3, 27))
_obs2, _rews, _dones, _infos = _vec.step(np.array([24, 24, 24]))
check("⑩ DummyVecEnv step → 3 env 并行推进（obs (3,27) + 3 reward）",
      _obs2.shape == (3, 27) and len(_rews) == 3)
_vec.close()

print("===== E) 委托 ground-truth 等价（堵 Agent 2 M-C/M-F 逃逸）=====")
# ⑪ reset 内层真重建到【抽中场景】（非恒 pool[0]）：内层 ego 初速 == 抽中场景 init_v，且覆盖 idx≠0（堵 M-C）
_me2 = MultiScenarioEnv(_pool)
_ok_mc = True
_saw_nonzero = False
for _s in range(25):
    _me2.reset(seed=_s)
    _picked_v = float(_pool[_me2._idx][1].initial_state.velocity)
    _inner_v = float(_me2._inner.env.init_state.velocity)
    if abs(_inner_v - _picked_v) > 1e-9:
        _ok_mc = False
        break
    if _me2._idx != 0:
        _saw_nonzero = True
check("⑪ reset 内层重建到抽中场景（init_v 匹配抽中 idx、含 idx≠0、堵 M-C）", _ok_mc and _saw_nonzero)

# ⑫ 委托 bit-exact：MultiScenarioEnv[单场景] vs 直接 ShieldedUSVEnv 同场景逐步对拍 obs/reward/done（堵 M-F）
_sc0 = _pool[0]
_msg = MultiScenarioEnv([_sc0]); _om0, _ = _msg.reset(seed=0)
_sd = ShieldedUSVEnv(_sc0[0], _sc0[1]); _os0, _ = _sd.reset(seed=0)
_lock = np.array_equal(_om0, _os0)
for _ in range(30):
    _m = _msg.action_masks()
    _a = 24 if _m[24] else int(np.where(_m)[0][0])
    _om, _rm, _tm, _trm, _ = _msg.step(_a)
    _osx, _rs, _ts, _trs, _ = _sd.step(_a)
    if not (np.array_equal(_om, _osx) and _rm == _rs and _tm == _ts and _trm == _trs):
        _lock = False
        break
    if _tm or _trm:
        break
check("⑫ 委托 bit-exact：Multi[单场景] vs Shielded 同场景逐步 obs/reward/done 一致（堵 M-F）", _lock)

print("\n" + ("✅ 全部 PASS" if _fail == 0 else f"❌ {_fail} 项 FAIL"))
sys.exit(1 if _fail else 0)
