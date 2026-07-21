"""
TRB · Discrete-safe 基线训练脚手架（step4c）
=============================================
忠实 Krasowski 2024 §VII（p11 实抽）：model-free RL = **PPO（基于 stable-baselines3）+ action masking [8]**；
agent 网络 = **MLP 2 层 × 64 神经元**；**10 random seeds × 3 million env steps / seed**。
我们用 **sb3-contrib MaskablePPO**（= PPO + action masking），mask = ShieldedUSVEnv 的 As(ρ)（step4c）。
其余 PPO 超参（lr / n_steps / batch_size / gamma / gae_lambda…）论文未明示 → **sb3 默认**（step4 标定时按 Table III 调）。

⚠️ **实跑训练 = step4e（Tier 3：烧 GPU + 与 T-ITS 错峰，待 user 拍板）**。本模块只给配置好的
   make_model / train 入口；step4c **不实跑大训练**（仅冒烟 F 段 learn 极少步验证端到端集成）。
⚠️ **多场景（step4d/e）**：论文在 ~2000 个随机 `HandcraftedTwoVesselEncounters` 上训练（每 episode 抽一场景）。
   当前 `ShieldedUSVEnv` = 单场景；多场景 env（reset 随机选场景 + VecEnv 并行）与评估管线一并在 step4d/e 建。
⚠️ **四方对比同口径（钱图）**：Base / Rule-reward / Discrete-safe / Continuous-safe 用同 env、同评估。
   Discrete-safe = make_discrete_safe_model（有盾 mask）；Base/RR = make_base_model / make_rule_reward_model
   （UnshieldedUSVEnv + colregs_weight 0/1，step4d-②）；
   Continuous-safe = SAC + 投影（Phase 3，算法换连续，D2 同环境对比）。
"""
from __future__ import annotations

import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import VecNormalize

from .usv_scenarios import make_vec_env
from .usv_shield import ShieldedUSVEnv, UnshieldedUSVEnv

# --- 论文 §VII 明确超参（fact-based，p11）---
POLICY_NET_ARCH = [64, 64]       # MLP 2 层 × 64 神经元
TOTAL_TIMESTEPS = 3_000_000      # 每 seed 3M environment steps
N_SEEDS = 10                     # 10 random seeds

# --- 停船 setup 修复配方（2026-06-11 实验验证，`03` D22/L19；**不动 Krasowski 忠实 reward 系数**）---
ENT_COEF = 0.01                  # 探索熵奖励（sb3 默认 0.0 → 探索塌缩到停船；`03` L17/L18）
VECNORM_KWARGS = dict(norm_obs=True, norm_reward=True, clip_obs=10.0)   # 观测+奖励归一化（命门：解 r_goal 大尺度致停船塌缩；实验坐实 d_final 1766→496m）


def make_discrete_safe_model(scenario, planning_problem, *, seed: int = 0,
                             env_kwargs: dict | None = None, **ppo_kwargs) -> MaskablePPO:
    """构造 Discrete-safe 基线 MaskablePPO（ShieldedUSVEnv 的 As(ρ) mask + MLP 2×64）。

    seed       : 训练种子（论文 10 seed 之一）。
    env_kwargs : 透传 ShieldedUSVEnv（scheduler_kwargs / clip_velocity…）。
    ppo_kwargs : 覆盖 PPO 默认（论文未明示项 = sb3 默认；step4 标定可调 lr/n_steps/…）。
    """
    env = ShieldedUSVEnv(scenario, planning_problem, **(env_kwargs or {}))
    cfg = dict(policy_kwargs=dict(net_arch=POLICY_NET_ARCH), seed=seed, verbose=0)
    cfg.update(ppo_kwargs)
    return MaskablePPO("MlpPolicy", env, **cfg)


def make_unshielded_model(scenario, planning_problem, *, seed: int = 0,
                          colregs_weight: float = 1.0, env_kwargs: dict | None = None,
                          **ppo_kwargs) -> MaskablePPO:
    """无盾基线 MaskablePPO（`UnshieldedUSVEnv`，mask 恒全 49 = 等价无 masking 的 PPO over 49 动作，step4d-②）。

    colregs_weight : 1.0=Rule-reward（含 r_colregs 软约束）/ 0.0=Base（关 r_colregs，论文 §VII p11）。
    env_kwargs     : 透传 UnshieldedUSVEnv（clip_velocity…）。ppo_kwargs : 覆盖 PPO 默认（同 Discrete-safe）。
    """
    env = UnshieldedUSVEnv(scenario, planning_problem, colregs_weight=colregs_weight, **(env_kwargs or {}))
    cfg = dict(policy_kwargs=dict(net_arch=POLICY_NET_ARCH), seed=seed, verbose=0)
    cfg.update(ppo_kwargs)
    return MaskablePPO("MlpPolicy", env, **cfg)


def make_base_model(scenario, planning_problem, *, seed: int = 0, **kw) -> MaskablePPO:
    """Base 基线（无盾 + **r_colregs=0**；论文 §VII p11 = r_sparse+r_goal+r_velocity+r_deviate）。"""
    return make_unshielded_model(scenario, planning_problem, seed=seed, colregs_weight=0.0, **kw)


def make_rule_reward_model(scenario, planning_problem, *, seed: int = 0, **kw) -> MaskablePPO:
    """Rule-reward 基线（无盾 + **r_colregs 开**；论文式(10) 全量、无安全验证盾）。"""
    return make_unshielded_model(scenario, planning_problem, seed=seed, colregs_weight=1.0, **kw)


def train_discrete_safe(scenario, planning_problem, *, seed: int = 0,
                        total_timesteps: int = TOTAL_TIMESTEPS, **ppo_kwargs) -> MaskablePPO:
    """⚠️ **step4e Tier 3（待 user 拍板，烧 GPU + 错峰 T-ITS）**：实跑训练。

    step4c **不调用**此函数（单场景 + 3M steps 是真训练，非脚手架验证）。Tier 3 拍板后由 step4e 在
    多场景 VecEnv 上跑 10 seed × 3M，产 Discrete-safe 基线对齐 Table III。
    """
    model = make_discrete_safe_model(scenario, planning_problem, seed=seed, **ppo_kwargs)
    model.learn(total_timesteps=total_timesteps)
    return model


def train_multiscene(paths, *, env_cls=ShieldedUSVEnv, colregs_weight: float = 1.0, seed: int = 0,
                     total_timesteps: int = TOTAL_TIMESTEPS, n_envs: int = 8,
                     ent_coef: float = ENT_COEF, use_vecnorm: bool = True, gamma: float = 0.99,
                     subproc: bool = True, **ppo_kwargs):
    """⚠️ **step4e Tier 3（待 user 拍板，烧 CPU + 错峰 T-ITS）**：多场景训练入口（VecEnv → VecNormalize → MaskablePPO）。

    **停船 setup 修复配方（2026-06-11 实验验证，`03` D22/L19）**：VecNormalize(obs+reward) 解 r_goal 大尺度致的
    停船塌缩 + ent_coef 探索；**不动 Krasowski 忠实 reward 系数**（telescoping 证 reward 偏好到达、停船是优化 setup）。
    四方对比：Discrete-safe=ShieldedUSVEnv / Base=UnshieldedUSVEnv+colregs_weight=0 / RR=UnshieldedUSVEnv+colregs_weight=1。
    返回 `(model, venv)`；venv 含 VecNormalize 统计 → **eval 必须用 `make_obs_transform(venv)`** 取同款 obs 归一化喂
    `evaluate.evaluate(..., obs_transform=)`，否则策略看错分布、Table III 失真。
    """
    venv = make_vec_env(paths=paths, n_envs=n_envs, env_cls=env_cls,
                        env_kwargs=dict(colregs_weight=colregs_weight), subproc=subproc, seed=seed)
    if use_vecnorm:
        venv = VecNormalize(venv, gamma=gamma, **VECNORM_KWARGS)
    # gamma 同进 MaskablePPO 与 VecNormalize（L21 MINOR②/D40 A-MINOR：原只进 VecNormalize，非默认 gamma 会两处分叉；
    # 默认 0.99 两处一致=位级不变、向后兼容）。ppo_kwargs 显式 gamma 仍可覆盖（cfg.update 在后）。
    cfg = dict(policy_kwargs=dict(net_arch=POLICY_NET_ARCH), seed=seed, ent_coef=ent_coef,
               gamma=gamma, verbose=0)
    cfg.update(ppo_kwargs)
    model = MaskablePPO("MlpPolicy", venv, **cfg)
    model.learn(total_timesteps=total_timesteps)
    return model, venv


def make_obs_transform(venv):
    """从训练好的 VecNormalize 取 obs 归一化变换（eval 用，与 `VecNormalize.normalize_obs` 逐式一致、快照冻结统计）。

    非 VecNormalize（use_vecnorm=False / 无归一化基线）→ 返回 None（evaluate 不归一化）。
    传给 `evaluate.evaluate(..., obs_transform=make_obs_transform(venv))`；原始状态仍喂 ViolationCounter（违规口径不变）。
    """
    if not isinstance(venv, VecNormalize):
        return None
    mean = np.asarray(venv.obs_rms.mean, dtype=np.float64).copy()   # 快照冻结（训练时 venv.training 会更新统计）
    var = np.asarray(venv.obs_rms.var, dtype=np.float64).copy()
    eps, clip = float(venv.epsilon), float(venv.clip_obs)
    return lambda obs: np.clip((np.asarray(obs, dtype=np.float64) - mean) / np.sqrt(var + eps),
                               -clip, clip).astype(np.float32)
