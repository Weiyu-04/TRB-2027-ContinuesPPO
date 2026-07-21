import ast

import numpy as np
import torch.nn as nn
from stable_baselines3.common.noise import NormalActionNoise

from serl_sprl.sb3_contrib.algorithm_configs import (
    A2CDiffProjHyperparameters,
    A2CHyperparameters,
    PPODiffProjHyperparameters,
    PPOHyperparameters,
    SACDiffProjHyperparameters,
    SACHyperparameters,
    TD3DiffProjHyperparameters,
    TD3Hyperparameters,
)


def parse_seeds(seeds_str):
    """Parse seeds argument from string to list of integers."""
    try:
        # Try to evaluate as a literal (e.g., "[1,2,3]" or "7")
        result = ast.literal_eval(seeds_str)

        # If it's a single integer, wrap it in a list
        if isinstance(result, int):
            return [result]
        elif isinstance(result, list):
            return result
        else:
            raise ValueError(f"Invalid seeds format: {seeds_str}")

    except (ValueError, SyntaxError):
        # If that fails, try splitting by comma (e.g., "1,2,3")
        try:
            return [int(s.strip()) for s in seeds_str.split(",")]
        except ValueError:
            raise ValueError(f"Invalid seeds format: {seeds_str}. Use '[1,2,3]', '1,2,3', or '7'")


def get_hyperparams_td3():
    return TD3Hyperparameters(
        learning_rate=0.004903576673457052,
        buffer_size=10000,
        batch_size=16,
        gamma=0.97,
        action_noise=NormalActionNoise(mean=np.zeros(1), sigma=0.9589708077963088 * np.ones(1)),
    )


def get_hyperparams_sac():
    return SACHyperparameters(
        learning_rate=0.004903576673457052,
        buffer_size=10000,
        batch_size=16,
        gamma=0.97,
    )


def get_seeds_td3():
    # the seeds are legacy code from previous experiments
    return [1, 3, 4, 5, 6, 8, 10]


def get_seeds_sac():
    return [1, 2, 3, 4, 5, 6, 7]


def get_seeds_ppo():
    return [1, 2, 3, 4, 5, 6, 7]


def get_seeds_a2c():
    # the seeds are legacy code from previous experiments
    return [1, 2, 3, 4, 6, 8, 10]


def get_num_total_timesteps():
    return 10000


def get_policy_kwargs_td3():
    return {"net_arch": [32, 32], "activation_fn": nn.ReLU}


def get_policy_kwargs_sac():
    return {"net_arch": [32, 32], "activation_fn": nn.ReLU}


def get_policy_kwargs_a2c():
    return {
        "net_arch": [32, 32],
        "activation_fn": nn.ReLU,
        "log_std_init": -2.4045965676917342,
        "squash_output": True,
        "squash_mean": True,
    }


def get_policy_kwargs_ppo():
    return {
        "net_arch": [32, 32],
        "activation_fn": nn.ReLU,
        "log_std_init": -1,
        "squash_output": True,
        "squash_mean": True,
    }


def get_hyperparams_a2c():
    return A2CHyperparameters(
        learning_rate=0.0027687411546956824,
        ent_coef=0.0,
        max_grad_norm=0.5,
        n_steps=8,
        gae_lambda=0.9,
        vf_coef=0.5,
        gamma=0.99,
    )


def get_hyperparams_ppo():
    return PPOHyperparameters(
        batch_size=8,
        learning_rate=1e-3,
        ent_coef=0.0,
        n_steps=8,
        gae_lambda=0.9,
        gamma=0.99,
        clip_range=0.1,
        n_epochs=5,
    )


def get_hyperparams_td3_diff_proj(use_penalty_critic: bool, use_per_sample_loss: bool, penalty_factor: float):
    return TD3DiffProjHyperparameters(
        learning_rate=0.004903576673457052,
        buffer_size=10000,
        batch_size=16,
        gamma=0.97,
        action_noise=NormalActionNoise(mean=np.zeros(1), sigma=0.9589708077963088 * np.ones(1)),
        use_penalty_critic=use_penalty_critic,
        use_per_sample_loss=use_per_sample_loss,
        penalty_factor=penalty_factor,
    )


def get_hyperparams_sac_diff_proj(use_penalty_critic: bool, use_per_sample_loss: bool, penalty_factor: float):
    return SACDiffProjHyperparameters(
        learning_rate=0.004903576673457052,
        buffer_size=10000,
        batch_size=16,
        gamma=0.97,
        use_penalty_critic=use_penalty_critic,
        use_per_sample_loss=use_per_sample_loss,
        penalty_factor=penalty_factor,
    )


def get_hyperparams_a2c_diff_proj(use_penalty_critic: bool, use_per_sample_loss: bool, penalty_factor: float):
    return A2CDiffProjHyperparameters(
        learning_rate=0.0027687411546956824,
        ent_coef=0.0,
        max_grad_norm=0.5,
        n_steps=8,
        gae_lambda=0.9,
        vf_coef=0.5,
        gamma=0.99,
        use_penalty_critic=use_penalty_critic,
        use_per_sample_loss=use_per_sample_loss,
        penalty_factor=penalty_factor,
    )


def get_hyperparams_ppo_diff_proj(use_penalty_critic: bool, use_per_sample_loss: bool, penalty_factor: float):
    return PPODiffProjHyperparameters(
        batch_size=8,
        learning_rate=1e-3,
        ent_coef=0.0,
        n_steps=8,
        gae_lambda=0.9,
        gamma=0.99,
        clip_range=0.1,
        n_epochs=5,
        use_penalty_critic=use_penalty_critic,
        use_per_sample_loss=use_per_sample_loss,
        penalty_factor=penalty_factor,
    )


def sample_ppo_params(trial):
    """Sampler for PPO hyperparams.

    :param trial: (optuna.trial)
    :return: (dict)
    """
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    step_factor = 4
    n_steps = step_factor * batch_size
    gamma = 0.98  # trial.suggest_categorical('gamma', [0.95, 0.98, 0.999])
    learning_rate = trial.suggest_float("lr", 1e-6, 0.01, log=True)
    ent_coef = 0.0
    # ent_coef = trial.suggest_float('ent_coef', 1e-7, 1e-4, log=True)
    clip_range = 0.1  # trial.suggest_categorical('clip_range', [0.01, 0.05, 0.1, 0.2, 0.3, 0.4])
    log_std_init = trial.suggest_float("log_std_init", -4, -1)
    gae_lambda = 0.9  # trial.suggest_categorical('lambda', [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
    network_size = 32  # trial.suggest_categorical('network_size', [16, 32, 64, 128, 256, 512])
    # activation = trial.suggest_categorical('activation_fn', ["tanh", "relu"])
    # activation_fn = nn.ReLU if activation == "relu" else nn.Tanh
    activation_fn = nn.ReLU
    policy_kwargs = {
        "net_arch": [network_size, network_size],
        "activation_fn": activation_fn,
        "squash_output": True,
        "log_std_init": log_std_init,
    }
    use_sde = True

    return {
        "batch_size": batch_size,
        "n_steps": n_steps,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "clip_range": clip_range,
        "gae_lambda": gae_lambda,
        "use_sde": use_sde,
        "policy_kwargs": policy_kwargs,
    }
