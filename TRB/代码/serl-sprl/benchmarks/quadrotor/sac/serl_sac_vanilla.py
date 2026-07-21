import sys
from pathlib import Path

# Add project root to Python path (must be before other imports)
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import gymnasium as gym

from benchmarks.quadrotor import common
from serl_sprl.benchmarking.base import Experiment
from serl_sprl.benchmarking.environments import EnvCreatorFactory
from serl_sprl.envs.quadrotor.quadrotor_coupled_dynamics import Quad2dCoupledEnvConfig, Quad2dProjConfig
from serl_sprl.sb3_contrib.algorithm_configs import SACConfig


def main():
    env_config = Quad2dCoupledEnvConfig(randomize_env=True)
    proj_config = Quad2dProjConfig()
    if not (env_config.id in gym.envs.registry.keys()):
        from gymnasium.envs.registration import register

        register(id=env_config.id, entry_point="serl_sprl.envs.quadrotor:Quad2dCoupledEnv")

    algorithm_config = SACConfig(
        hyperparams=common.get_hyperparams_sac(),
        total_timesteps=common.get_num_total_timesteps_sac(),
        policy_kwargs=common.get_policy_kwargs_sac(),
    )
    path = "SERL/Quadrotor/SAC/Vanilla"
    seeds = common.get_seeds_sac()
    env_factory = EnvCreatorFactory(approach="serl", improvement_strategy="none", env_id=env_config.id)
    experiment = Experiment(
        env_factory=env_factory,
        env_config=env_config,
        algorithm_config=algorithm_config,
        proj_config=proj_config,
        seeds=seeds,
        tag=path,
    )
    experiment.run_training(tags=["SERL", "Quadrotor", "SAC", "Vanilla"])
    experiment.run_evaluation(seeds=seeds)


if __name__ == "__main__":
    main()
