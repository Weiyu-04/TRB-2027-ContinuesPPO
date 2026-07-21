import argparse
import sys
from pathlib import Path

# Add project root to Python path (must be before other imports)
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import gymnasium as gym
import numpy as np

from benchmarks.seeker import common
from serl_sprl.benchmarking.base import Experiment
from serl_sprl.benchmarking.environments import EnvCreatorFactory
from serl_sprl.envs.seeker.seeker_point_mass import SeekerEnvConfig, SeekerProjConfig
from serl_sprl.sb3_contrib.algorithm_configs import PPODiffProjConfig


def main(penalty_factor: float, seeds: list):
    env_config = SeekerEnvConfig(randomize_env=True, dtype=np.float64)
    proj_config = SeekerProjConfig()

    if not (env_config.id in gym.envs.registry.keys()):
        from gymnasium.envs.registration import register

        register(id=env_config.id, entry_point="serl_sprl.envs.seeker:SeekerEnvPointMass")

    ppo_hyperparams = common.get_hyperparams_ppo_diff_proj(
        use_penalty_critic=False, use_per_sample_loss=True, penalty_factor=penalty_factor
    )
    ppo_hyperparams.use_sde = False
    algorithm_config = PPODiffProjConfig(
        hyperparams=ppo_hyperparams,
        total_timesteps=common.get_num_total_timesteps(),
        policy_kwargs=common.get_policy_kwargs_ppo(),
    )
    path = f"SPRL/Seeker/PPO/PSL/{penalty_factor}"
    env_factory = EnvCreatorFactory(approach="sprl", improvement_strategy="psl", env_id=env_config.id)
    experiment = Experiment(
        env_factory=env_factory,
        env_config=env_config,
        algorithm_config=algorithm_config,
        proj_config=proj_config,
        seeds=seeds,
        tag=path,
    )
    experiment.run_training(tags=["SPRL", "Seeker", "PPO", "PSL"])
    experiment.run_evaluation(seeds=seeds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--penalty_factor", type=float, default=None, required=False)
    parser.add_argument("--seeds", type=common.parse_seeds, default=common.get_seeds_ppo(), required=False)
    args, _ = parser.parse_known_args()
    args = vars(args)
    if args["penalty_factor"] is not None:
        main(args["penalty_factor"], args["seeds"])
    else:
        penalty_factors = [0.1]
        for penalty_factor in penalty_factors:
            main(penalty_factor, args["seeds"])
