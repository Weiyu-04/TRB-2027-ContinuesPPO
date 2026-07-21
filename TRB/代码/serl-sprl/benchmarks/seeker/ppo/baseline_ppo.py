import sys
from pathlib import Path

# Add project root to Python path (must be before other imports)
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from benchmarks.seeker import common
import gymnasium as gym

from serl_sprl.benchmarking.base import Experiment
from serl_sprl.benchmarking.environments import EnvCreatorFactory
from serl_sprl.envs.seeker.seeker_point_mass import SeekerEnvConfig, SeekerProjConfig
from serl_sprl.sb3_contrib.algorithm_configs import PPOConfig


def main():
    env_config = SeekerEnvConfig(randomize_env=True)
    proj_config = SeekerProjConfig()
    if not (env_config.id in gym.envs.registry.keys()):
        from gymnasium.envs.registration import register

        register(id=env_config.id, entry_point="serl_sprl.envs.seeker:SeekerEnvPointMass")

    algorithm_config = PPOConfig(
        hyperparams=common.get_hyperparams_ppo(),
        total_timesteps=common.get_num_total_timesteps(),
        policy_kwargs=common.get_policy_kwargs_ppo(),
    )
    path = "Baseline/Seeker/PPO/Vanilla"
    env_factory = EnvCreatorFactory(approach="baseline", improvement_strategy="none", env_id=env_config.id)
    experiment = Experiment(
        env_factory=env_factory,
        env_config=env_config,
        algorithm_config=algorithm_config,
        proj_config=proj_config,
        seeds=common.get_seeds_a2c(),
        tag=path,
    )
    best_trial, study = experiment.run_hp_tuning(
        sampler_fn=common.sample_ppo_params, n_trials=100, tags=["PPO", "Baseline_HP_Tuning"]
    )
    print("Best trial:", best_trial)

    print("Value: ", best_trial.value)

    print("Params: ")
    for key, value in best_trial.params.items():
        print("    {}: {}".format(key, value))


if __name__ == "__main__":
    main()
