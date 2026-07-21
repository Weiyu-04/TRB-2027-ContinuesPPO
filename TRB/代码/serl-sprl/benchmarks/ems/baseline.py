import common
import gymnasium as gym

from serl_sprl.benchmarking.base import Experiment
from serl_sprl.benchmarking.environments import EnvCreatorFactory
from serl_sprl.envs.energy_management_system.ems import EMSEnvConfig, EMSProjConfig
from serl_sprl.sb3_contrib.algorithm_configs import TD3Config


def main():
    env_config = EMSEnvConfig(randomize_env=True)
    proj_config = EMSProjConfig()
    if not (env_config.id in gym.envs.registry.keys()):
        from gymnasium.envs.registration import register

        register(id=env_config.id, entry_point="serl_sprl.envs.energy_management_system:EnergyManagementSystemEnv")

    algorithm_config = TD3Config(
        hyperparams=common.get_hyperparams_td3(),
        total_timesteps=common.get_num_total_timesteps(),
        policy_kwargs=common.get_policy_kwargs_td3(),
    )
    path = "Baseline/EMS/TD3/Vanilla"
    env_factory = EnvCreatorFactory(approach="baseline", improvement_strategy="none", env_id=env_config.id)
    experiment = Experiment(
        env_factory=env_factory,
        env_config=env_config,
        algorithm_config=algorithm_config,
        proj_config=proj_config,
        seeds=common.get_seeds_td3(),
        tag=path,
    )
    best_trial, study = experiment.run_hp_tuning(
        sampler_fn=common.sample_td3_params, n_trials=100, tags=["TD3", "Baseline_HP_Tuning"]
    )
    print("Best trial:", best_trial)

    print("Value: ", best_trial.value)

    print("Params: ")
    for key, value in best_trial.params.items():
        print("    {}: {}".format(key, value))


if __name__ == "__main__":
    main()
