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
from serl_sprl.sb3_contrib.noise import LinearNormalActionNoise


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
    total_training_timesteps = get_num_total_timesteps()
    noise_steps = total_training_timesteps - 0.1 * total_training_timesteps
    return TD3Hyperparameters(
        learning_rate=1e-5,
        buffer_size=10000,
        batch_size=256,
        gamma=0.99,
        action_noise=LinearNormalActionNoise(
            mean=np.zeros((2,)), sigma=0.32280725356879164 * np.ones((2,)), max_steps=noise_steps
        ),
    )


def get_hyperparams_sac():
    return SACHyperparameters(
        learning_rate=0.0002609068824650364,
        buffer_size=10000,
        batch_size=16,
        gamma=0.98,
    )


def get_seeds_td3():
    return [1, 2, 3, 4, 5, 6, 7]


def get_seeds_sac():
    return [1, 2, 3, 4, 5, 6, 7]


def get_seeds_a2c():
    return [1, 2, 3, 4, 5, 6, 7]


def get_seeds_ppo():
    return [1, 2, 3, 4, 5, 6, 7]


def get_num_total_timesteps():
    return 50000


def get_policy_kwargs_td3():
    return {"net_arch": [128, 128], "activation_fn": nn.ReLU}


def get_policy_kwargs_sac():
    return {"net_arch": [128, 128], "activation_fn": nn.ReLU}


def get_policy_kwargs_a2c():
    return {
        "net_arch": [128, 128],
        "activation_fn": nn.ReLU,
        "log_std_init": -2,
        "squash_output": False,
        "squash_mean": True,
    }


def get_policy_kwargs_ppo():
    return {
        "net_arch": [128, 128],
        "activation_fn": nn.ReLU,
        "log_std_init": -2,
        "squash_output": False,
        "squash_mean": True,
    }


def get_hyperparams_a2c():
    return A2CHyperparameters(
        learning_rate=1e-5,
        ent_coef=0.0,
        max_grad_norm=0.5,
        n_steps=32,
        gae_lambda=0.9,
        vf_coef=0.4,
        gamma=0.9,
    )


def get_hyperparams_ppo():
    return PPOHyperparameters(
        batch_size=32,
        learning_rate=1e-5,
        ent_coef=0.0,
        n_steps=3 * 32,
        gae_lambda=0.9,
        gamma=0.9,
        clip_range=0.1,
        n_epochs=5,
    )


def get_hyperparams_td3_diff_proj(use_penalty_critic: bool, use_per_sample_loss: bool, penalty_factor: float):
    return TD3DiffProjHyperparameters(
        learning_rate=0.0002609068824650364,
        buffer_size=10000,
        batch_size=48,
        gamma=0.98,
        action_noise=NormalActionNoise(mean=np.zeros((2,)), sigma=0.07358482238558972 * np.ones((2,))),
        use_penalty_critic=use_penalty_critic,
        use_per_sample_loss=use_per_sample_loss,
        penalty_factor=penalty_factor,
    )


def get_hyperparams_sac_diff_proj(use_penalty_critic: bool, use_per_sample_loss: bool, penalty_factor: float):
    return SACDiffProjHyperparameters(
        learning_rate=0.0002609068824650364,
        buffer_size=10000,
        batch_size=16,
        gamma=0.98,
        use_penalty_critic=use_penalty_critic,
        use_per_sample_loss=use_per_sample_loss,
        penalty_factor=penalty_factor,
    )


def get_hyperparams_a2c_diff_proj(use_penalty_critic: bool, use_per_sample_loss: bool, penalty_factor: float):
    return A2CDiffProjHyperparameters(
        learning_rate=1e-5,
        ent_coef=0.0,
        max_grad_norm=0.5,
        n_steps=32,
        gae_lambda=0.9,
        vf_coef=0.4,
        gamma=0.9,
        use_penalty_critic=use_penalty_critic,
        use_per_sample_loss=use_per_sample_loss,
        penalty_factor=penalty_factor,
    )


def get_hyperparams_ppo_diff_proj(use_penalty_critic: bool, use_per_sample_loss: bool, penalty_factor: float):
    return PPODiffProjHyperparameters(
        batch_size=32,
        learning_rate=1e-5,
        ent_coef=0.0,
        n_steps=3 * 32,
        gae_lambda=0.9,
        gamma=0.9,
        clip_range=0.1,
        n_epochs=5,
        use_penalty_critic=use_penalty_critic,
        use_per_sample_loss=use_per_sample_loss,
        penalty_factor=penalty_factor,
    )


# hyperparameter sets for tuning
def sample_td3_params(trial):
    """Sampler for TD3 hyperparams.

    :param trial: (optuna.trial)
    :return: (dict)
    """
    gamma = 0.99
    learning_rate = trial.suggest_float("lr", 1e-6, 0.001, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    buffer_size = int(1e5)
    train_freq = 1
    gradient_steps = train_freq
    noise_type = "linear-normal"
    noise_std = trial.suggest_float("noise_std", 0, 1)
    network_size = 128
    activation_fn = nn.ReLU
    policy_kwargs = {"net_arch": [network_size, network_size], "activation_fn": activation_fn}

    hyperparams = {
        "gamma": gamma,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "buffer_size": buffer_size,
        "train_freq": train_freq,
        "gradient_steps": gradient_steps,
        "policy_kwargs": policy_kwargs,
    }

    if noise_type == "normal":
        hyperparams["action_noise"] = NormalActionNoise(
            mean=np.zeros(trial.n_actions), sigma=noise_std * np.ones(trial.n_actions)
        )

    elif noise_type == "linear-normal":
        steps_noise = trial.n_timesteps - 0.05 * trial.n_timesteps  # Last episodes should run without noise
        hyperparams["action_noise"] = LinearNormalActionNoise(
            mean=np.zeros(trial.n_actions),
            sigma=noise_std * np.ones(trial.n_actions),
            max_steps=steps_noise,
        )

    return hyperparams
