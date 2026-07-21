import argparse
import sys
from pathlib import Path

# Add project root to Python path (must be before other imports)
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import gymnasium as gym

from benchmarks.seeker import common
from serl_sprl.benchmarking.base import Experiment
from serl_sprl.benchmarking.environments import EnvCreatorFactory
from serl_sprl.envs.seeker.seeker_point_mass import SeekerEnvConfig, SeekerProjConfig
from serl_sprl.sb3_contrib.algorithm_configs import PPOConfig


def main(penalty_factor: float, seeds: list):
    env_config = SeekerEnvConfig(randomize_env=True)
    proj_config = SeekerProjConfig()
    proj_config.penalty_factor = penalty_factor
    if not (env_config.id in gym.envs.registry.keys()):
        from gymnasium.envs.registration import register

        register(id=env_config.id, entry_point="serl_sprl.envs.seeker:SeekerEnvPointMass")

    hyperparams_ppo = common.get_hyperparams_ppo()
    hyperparams_ppo.use_sde = False
    algorithm_config = PPOConfig(
        hyperparams=hyperparams_ppo,
        total_timesteps=common.get_num_total_timesteps(),
        policy_kwargs=common.get_policy_kwargs_ppo(),
    )
    path = f"SERL/Seeker/PPO/Penalty/{penalty_factor}"
    seeds = common.get_seeds_ppo()
    env_factory = EnvCreatorFactory(approach="serl", improvement_strategy="penalty", env_id=env_config.id)
    experiment = Experiment(
        env_factory=env_factory,
        env_config=env_config,
        algorithm_config=algorithm_config,
        proj_config=proj_config,
        seeds=seeds,
        tag=path,
    )
    experiment.run_training(tags=["SERL", "Seeker", "PPO", "Penalty"])
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
