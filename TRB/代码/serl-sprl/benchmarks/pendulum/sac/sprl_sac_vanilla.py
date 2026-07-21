import sys
from pathlib import Path

# Add project root to Python path (must be before other imports)
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import gymnasium as gym

from benchmarks.pendulum import common
from serl_sprl.benchmarking.base import Experiment
from serl_sprl.benchmarking.environments import EnvCreatorFactory
from serl_sprl.envs.pendulum.pendulum import PendulumEnvConfig, PendulumProjConfig
from serl_sprl.sb3_contrib.algorithm_configs import SACDiffProjConfig


def main():
    env_config = PendulumEnvConfig(randomize_env=True)
    proj_config = PendulumProjConfig()

    if not (env_config.id in gym.envs.registry.keys()):
        from gymnasium.envs.registration import register

        register(id=env_config.id, entry_point="serl_sprl.envs.pendulum:SimplePendulumEnv")

    algorithm_config = SACDiffProjConfig(
        hyperparams=common.get_hyperparams_sac_diff_proj(
            use_penalty_critic=False, use_per_sample_loss=False, penalty_factor=0.0
        ),
        total_timesteps=common.get_num_total_timesteps(),
        policy_kwargs=common.get_policy_kwargs_sac(),
    )
    path = "SPRL/Pendulum/SAC/Vanilla"
    seeds = common.get_seeds_sac()
    env_factory = EnvCreatorFactory(approach="sprl", improvement_strategy="none", env_id=env_config.id)
    experiment = Experiment(
        env_factory=env_factory,
        env_config=env_config,
        algorithm_config=algorithm_config,
        proj_config=proj_config,
        seeds=seeds,
        tag=path,
    )
    experiment.run_training(tags=["SPRL", "Pendulum", "SAC", "Vanilla"])
    experiment.run_evaluation(seeds=seeds)


if __name__ == "__main__":
    main()
