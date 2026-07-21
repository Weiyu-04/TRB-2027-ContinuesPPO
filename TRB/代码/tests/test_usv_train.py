"""train.py maker 接线测试（step4d-②）：Base/RR/Discrete-safe maker 的 colregs_weight 映射锁定。
跑：/opt/miniconda3/envs/trb/bin/python -B 代码/tests/test_usv_train.py

补 Agent 2 缺口：make_base_model 若误写 colregs_weight=1.0（=RR）会静默污染钱图 Base 列
（Base 应 r_colregs=0），需测试守护此 load-bearing 映射。离线无 T-0 则 SKIP。
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

_fail = 0
def check(name, ok):
    global _fail
    if not ok:
        _fail += 1
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")

_T0 = "/tmp/trb_T0.xml"
try:
    if not os.path.exists(_T0):
        import urllib.request
        urllib.request.urlretrieve(
            "https://gitlab.lrz.de/tum-cps/commonocean-scenarios/-/raw/main/scenarios/"
            "HandcraftedTwoVesselEncounters_01_24/ZAM_AAA-1_20240121_T-0.xml", _T0)
    import numpy as np
    from stable_baselines3.common.vec_env import VecNormalize
    from commonocean.common.file_reader import CommonOceanFileReader
    from trb_env.train import (make_base_model, make_rule_reward_model, make_discrete_safe_model,
                               train_multiscene, make_obs_transform, ENT_COEF)
    from trb_env.usv_scenarios import make_vec_env
    from trb_env.usv_shield import ShieldedUSVEnv
    _sc, _pp = CommonOceanFileReader(_T0).open()
    _ppx = list(_pp.planning_problem_dict.values())[0]
except Exception as e:                                     # noqa: BLE001
    print(f"[SKIP] 需 /tmp/trb_T0.xml + sb3（离线/缺包 {type(e).__name__}）")
    sys.exit(0)


def _weight(model):
    """从 MaskablePPO 鲁棒解包到底层 env 的 reward_fn.colregs_weight（穿 VecEnv/Monitor 包装层）。"""
    venv = model.get_env()
    e = venv.envs[0] if hasattr(venv, "envs") else venv
    for _ in range(8):
        if hasattr(e, "reward_fn"):                        # USVEnv 持 reward_fn
            return e.reward_fn.colregs_weight
        e = getattr(e, "env", None)
        if e is None:
            break
    raise RuntimeError("解包未找到 reward_fn")


print("===== train.py maker colregs_weight 映射（4d-②）=====")
check("① make_base_model → colregs_weight==0.0（Base 关 r_colregs，论文 §VII p12）",
      _weight(make_base_model(_sc, _ppx)) == 0.0)
check("② make_rule_reward_model → colregs_weight==1.0（RR 含 r_colregs）",
      _weight(make_rule_reward_model(_sc, _ppx)) == 1.0)
check("③ make_discrete_safe_model → colregs_weight==1.0（Safe 含 r_colregs = 式(10)+安全验证）",
      _weight(make_discrete_safe_model(_sc, _ppx)) == 1.0)

print("\n===== 停船修复配方：VecNormalize + ent_coef（D22/L19）=====")
# ④ make_obs_transform 与 VecNormalize.normalize_obs 逐式一致（命门：eval 归一化须 == 训练、否则策略看错分布→Table III 失真）
_venv = make_vec_env(paths=[_T0], n_envs=1, env_cls=ShieldedUSVEnv, subproc=False)
_venv = VecNormalize(_venv, norm_obs=True, norm_reward=True, clip_obs=10.0)
_venv.reset()
_venv.obs_rms.update(np.outer(1.0 + np.arange(40), np.ones(27)) * 50.0)   # 确定性填充统计（不依赖 step/mask）
_tf = make_obs_transform(_venv)
_test_obs = np.linspace(-10000.0, 10000.0, 27)   # 跨 clip 范围（部分越 ±10 归一化界）→ 同时守护 mean/var/eps/clip
check("④ make_obs_transform == VecNormalize.normalize_obs（逐元素，eval 归一化忠实训练）",
      np.allclose(_tf(_test_obs), _venv.normalize_obs(_test_obs.astype(np.float32)), atol=1e-5)
      and _tf(_test_obs).shape == (27,))
# ⑤ 非 VecNormalize → None（基线/无归一化路径，evaluate 不变换）
check("⑤ make_obs_transform(非VecNormalize) == None（基线不归一化）",
      make_obs_transform(_venv.venv) is None)
_venv.close()
# ⑥ train_multiscene 构造 = VecNormalize 包装 + ent_coef 配方（极少步冒烟、DummyVecEnv 避 spawn 重导入）
_m6, _vn6 = train_multiscene([_T0], env_cls=ShieldedUSVEnv, seed=0, total_timesteps=64,
                             n_envs=1, subproc=False, n_steps=64, batch_size=64)
check("⑥ train_multiscene → VecNormalize 包装 + ent_coef==0.01（停船配方接线）",
      isinstance(_vn6, VecNormalize) and abs(float(_m6.ent_coef) - ENT_COEF) < 1e-12)
_vn6.close()

print("\n" + ("✅ 全部 PASS" if _fail == 0 else f"❌ {_fail} 项 FAIL"))
sys.exit(1 if _fail else 0)
