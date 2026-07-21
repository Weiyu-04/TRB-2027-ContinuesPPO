import os
import pickle

import pandas as pd
from stable_baselines3.common.utils import configure_logger

import wandb
from serl_sprl.benchmarking.environments import EnvCreatorFactory
from serl_sprl.envs.configs import BaseEnvConfig, BaseModel, BaseProjectionConfig
from serl_sprl.sb3_contrib.callbacks import DeploySafetyCallback, TrainSafetyCallback


class Experiment:
    def __init__(
        self,
        env_factory: EnvCreatorFactory,
        env_config: BaseEnvConfig,
        algorithm_config: BaseModel,
        seeds: list,
        tag: str,
        proj_config: BaseProjectionConfig = None,
        device: str = "cpu",
        n_eval_episodes: int = 10,
    ):
        self.custom_config = {
            **proj_config.model_dump(exclude=["safe_control_fn"]),
            **algorithm_config.model_dump(),
            **env_config.model_dump(exclude="safe_region"),
        }
        proj_config = proj_config.model_dump() if proj_config is not None else {}
        algorithm_config = algorithm_config.model_dump()
        proj_config["dtype"] = env_config.dtype  # Ensure dtype consistency
        self.algorithm_config = algorithm_config
        self.env_config = env_config
        self.proj_config = proj_config
        self.n_eval_episodes = n_eval_episodes

        self.env_factory = env_factory
        self.env_creator = self.env_factory.get_env_creator(wrapper_kwargs=proj_config)

        # Use OUTPUT_ROOT instead of os.getcwd()
        output_root = os.environ.get("OUTPUT_ROOT", os.getcwd())
        # Construct paths relative to OUTPUT_ROOT
        self.model_dir = os.path.join(output_root, "models", tag + "/")
        self.tb_log_dir = os.path.join(output_root, "tensorboard", tag)
        self.results_dir = os.path.join(output_root, "results", tag)
        self.seeds = seeds

        self.seeds = seeds
        self.n_runs = 7
        self.device = device
        os.makedirs(self.model_dir, exist_ok=True)

        # Save config as pickle file
        with open(self.model_dir + "config.pkl", "wb") as f:
            pickle.dump(self.custom_config, f)

    def run_training(self, tags: list = None):
        for run, seed in enumerate(self.seeds):
            # initialize environment
            env = self.env_creator.create_env(env_config=self.env_config, num_envs=1, device=self.device)
            # set random seed for safe region if available (for backwards compatibility with old code)
            try:
                env.envs[0].get_wrapper_attr("safe_region").rng = int("1707" + str(seed))
            except AttributeError:
                # we are in the seeker environment
                pass
            # Initialize wandb
            wandb.init(project="serl_sprl", sync_tensorboard=True, tags=tags, name="_".join(tags))
            wandb.run.name = f"{wandb.run.name}_seed_{seed}"
            # If SPRL, we need to pass the safeguard to the policy
            if getattr(self.env_factory, "approach") == "sprl":
                self.algorithm_config["policy_kwargs"]["safeguard"] = env.envs[0].get_wrapper_attr("safeguard")
            # Set up model
            model = self.algorithm_config.get("algorithm")(
                seed=seed,
                env=env,
                tensorboard_log=self.tb_log_dir,
                policy=self.algorithm_config.get("policy"),
                policy_kwargs=self.algorithm_config.get("policy_kwargs", {}),
                device=self.device,
                **self.algorithm_config.get("hyperparams", {}),
            )

            custom_config = wandb.Artifact("custom_config", type="config")
            custom_config.add_file(self.model_dir + "config.pkl")
            wandb.run.log_artifact(custom_config)

            # Train model
            model.learn(
                total_timesteps=self.algorithm_config.get("total_timesteps", 100000),
                callback=TrainSafetyCallback(model_save_path=self.model_dir + str(seed)),
                log_interval=1,
                progress_bar=True,
            )
            wandb.finish()

    def run_evaluation(self, seeds: list, deployment_approaches: list = ["safe", "unsafe"]):
        callback = DeploySafetyCallback()
        original_approach = self.env_factory.approach
        n_existing_models = len(
            [name for name in os.listdir(self.model_dir) if os.path.isdir(os.path.join(self.model_dir, name))]
        )
        if n_existing_models < len(seeds):
            raise ValueError(
                f"Not enough trained models found in {self.model_dir}. \
                Expected at least {len(seeds)}, but found {n_existing_models}."
            )

        for deployment_approach in deployment_approaches:
            if deployment_approach == "unsafe":
                self.env_factory.set_approach("baseline")
            env = self.env_factory.get_env_creator(wrapper_kwargs=self.proj_config).create_env(
                self.env_config, num_envs=1
            )
            for i in range(1, self.n_runs + 1):
                # set random seed for safe region if available (for backwards compatibility with old code)
                try:
                    env.envs[0].get_wrapper_attr("safe_region").rng = int("2022" + str(i))
                except AttributeError:
                    # we are in the seeker environment
                    pass
                # Check if deployment results for this config exist
                if os.path.exists(self.results_dir + f"/{deployment_approach}/_{i}"):
                    if len(os.listdir(self.results_dir + f"/{deployment_approach}/_{i}")) != 0:

                        raise ValueError(
                            f"Results for {deployment_approach} deployment and run {i} already exist. \
                            Please remove them before running evaluation again."
                        )

                model_path = self.model_dir + str(seeds[i - 1]) + "/model"
                model = self.algorithm_config.get("algorithm").load(path=model_path, env=env)
                if original_approach == "sprl":
                    if deployment_approach == "unsafe":
                        # No longer needed, but doesn't hurt either
                        model.policy.deactivate_projection()
                    else:
                        model.policy.activate_projection()

                # set up logger
                logger = configure_logger(
                    tb_log_name="", tensorboard_log=self.results_dir + f"/{deployment_approach}/", verbose=1
                )
                model.set_logger(logger)

                callback.init_callback(model=model)
                callback.on_rollout_start()

                # evaluate policy over n_eval_ep rollouts
                for j in range(self.n_eval_episodes):
                    dones = False
                    env.seed(seed=self.n_runs + j)
                    obs = env.reset()
                    callback.update_locals(locals())
                    while not dones:
                        action, _ = model.predict(observation=obs, deterministic=True)
                        obs, reward, dones, infos = env.step(action)
                        if hasattr(model.policy, "last_projected") and deployment_approach == "safe":
                            policy_projected = getattr(model.policy, "last_projected")
                            infos[0]["projection"]["policy_projected"] = policy_projected

                        # Give access to local variables
                        callback.update_locals(locals())
                        if callback.on_step() is False:
                            return
                    callback.on_rollout_end()

                env.close()
            self.compute_statistics(deployment_approach)

    def run_hp_tuning(
        self,
        sampler_fn,
        n_trials: int = 100,
        n_timesteps: int = 50000,
        n_eval_episodes: int = 5,
        eval_freq: int = 5000,
        study_name: str = None,
        storage: str = None,
        pruner_type: str = "median",
        tags: list = None,
    ):
        """
        Run hyperparameter tuning using Optuna.

        Args:
            sampler_fn: Callable that takes a trial and returns hyperparameters dict
            n_trials: Number of optimization trials to run
            n_timesteps: Number of timesteps for each trial
            n_eval_episodes: Number of episodes for evaluation during training
            eval_freq: Frequency of evaluation during training
            study_name: Name of the Optuna study
            storage: Storage backend for the study (e.g., sqlite database)
            pruner_type: Type of pruner ("median", "successive_halving", or "hyperband")
            tags: Tags for wandb logging

        Returns:
            Best trial and study object
        """
        import optuna
        from optuna.integration import WeightsAndBiasesCallback
        from stable_baselines3.common.evaluation import evaluate_policy

        # Set up pruner
        if pruner_type == "median":
            pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=eval_freq)
        elif pruner_type == "successive_halving":
            pruner = optuna.pruners.SuccessiveHalvingPruner()
        elif pruner_type == "hyperband":
            pruner = optuna.pruners.HyperbandPruner()
        else:
            raise ValueError(f"Unknown pruner type: {pruner_type}")

        # Set up sampler (TPE for non-random sampling)
        sampler = optuna.samplers.TPESampler(seed=42)

        # Create study
        if study_name is None:
            study_name = f"hp_tuning_{self.__class__.__name__}"

        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
        )

        def objective(trial):
            """Objective function for hyperparameter optimization."""

            # Sample hyperparameters using the provided sampler function
            trial.n_actions = self.env_config.action_space.shape[0] if hasattr(self.env_config, "action_space") else 2
            trial.n_timesteps = n_timesteps
            hyperparams = sampler_fn(trial)

            # Create environment for this trial in baseline mode (no projection)
            original_approach = self.env_factory.approach
            self.env_factory.set_approach("baseline")
            baseline_env_creator = self.env_factory.get_env_creator(wrapper_kwargs=self.proj_config)
            env = baseline_env_creator.create_env(env_config=self.env_config, num_envs=1, device=self.device)

            # Set random seed for reproducibility
            seed = trial.number + 42
            try:
                env.envs[0].get_wrapper_attr("safe_region").rng = int("1707" + str(seed))
            except AttributeError:
                pass

            # Initialize wandb for this trial
            if tags is None:
                trial_tags = ["hp_tuning", f"trial_{trial.number}"]
            else:
                trial_tags = tags + ["hp_tuning", f"trial_{trial.number}"]

            wandb.init(
                project="serl_sprl_hp_tuning",
                sync_tensorboard=True,
                tags=trial_tags,
                name=f"trial_{trial.number}",
                reinit=True,
            )

            # Log hyperparameters to wandb
            wandb.config.update(hyperparams)
            wandb.config.update({"trial_number": trial.number})

            try:
                # Update algorithm config with sampled hyperparameters
                updated_algorithm_config = self.algorithm_config.copy()
                for key, value in hyperparams.items():
                    updated_algorithm_config[key] = value

                # No safeguard needed for HP tuning as we're always in baseline mode

                # Create model with sampled hyperparameters
                model = updated_algorithm_config.get("algorithm")(
                    seed=seed,
                    env=env,
                    tensorboard_log=None,  # Disable TB logging for HP tuning
                    policy=updated_algorithm_config.get("policy"),
                    policy_kwargs=updated_algorithm_config.get("policy_kwargs", {}),
                    device=self.device,
                    **{k: v for k, v in hyperparams.items() if k not in ["algorithm", "policy", "policy_kwargs"]},
                )

                # Create evaluation environment (also in baseline mode)
                eval_env = baseline_env_creator.create_env(env_config=self.env_config, num_envs=1, device=self.device)

                # Custom callback for pruning
                from stable_baselines3.common.callbacks import BaseCallback

                class TrialEvalCallback(BaseCallback):
                    """Custom callback for Optuna trial evaluation and pruning."""

                    def __init__(self, eval_env, trial, n_eval_episodes, eval_freq, verbose=0):
                        super().__init__(verbose)
                        self.eval_env = eval_env
                        self.trial = trial
                        self.n_eval_episodes = n_eval_episodes
                        self.eval_freq = eval_freq
                        self.eval_count = 0

                    def _on_step(self) -> bool:
                        if self.eval_freq > 0 and self.num_timesteps % self.eval_freq == 0:
                            # Evaluate the model
                            from stable_baselines3.common.evaluation import evaluate_policy

                            mean_reward, std_reward = evaluate_policy(
                                self.model,
                                self.eval_env,
                                n_eval_episodes=self.n_eval_episodes,
                                deterministic=True,
                                render=False,
                            )

                            # Report to Optuna
                            self.trial.report(mean_reward, self.num_timesteps)

                            # Log to wandb
                            import wandb

                            wandb.log(
                                {
                                    "eval/mean_reward": mean_reward,
                                    "eval/std_reward": std_reward,
                                    "eval/timesteps": self.num_timesteps,
                                }
                            )

                            # Check if trial should be pruned
                            if self.trial.should_prune():
                                if self.verbose > 0:
                                    print(f"Trial {self.trial.number} pruned at timestep {self.num_timesteps}")
                                raise optuna.TrialPruned()

                        return True

                # Create the custom callback
                trial_callback = TrialEvalCallback(
                    eval_env=eval_env, trial=trial, n_eval_episodes=n_eval_episodes, eval_freq=eval_freq, verbose=0
                )

                # Train the model
                model.learn(
                    total_timesteps=n_timesteps,
                    callback=trial_callback,
                    log_interval=None,  # Disable logging
                    progress_bar=False,
                )

                # Final evaluation
                mean_reward, std_reward = evaluate_policy(
                    model,
                    eval_env,
                    n_eval_episodes=n_eval_episodes * 2,  # More episodes for final eval
                    deterministic=True,
                    render=False,
                )

                # Log final results to wandb
                wandb.log(
                    {"final_mean_reward": mean_reward, "final_std_reward": std_reward, "trial_number": trial.number}
                )

                # Clean up
                env.close()
                eval_env.close()

                # Restore original approach
                self.env_factory.set_approach(original_approach)

                return mean_reward

            except optuna.TrialPruned:
                # Log pruned trial
                wandb.log({"trial_pruned": True, "trial_number": trial.number})
                # Restore original approach before re-raising
                self.env_factory.set_approach(original_approach)
                raise

            except Exception as e:
                # Log failed trial
                wandb.log({"trial_failed": True, "error": str(e), "trial_number": trial.number})
                # Restore original approach before re-raising
                self.env_factory.set_approach(original_approach)
                raise

            finally:
                wandb.finish()

        # Set up WandB callback for Optuna
        wandbc = WeightsAndBiasesCallback(
            metric_name="value",
            wandb_kwargs={"project": "serl_sprl_hp_tuning_summary", "name": f"{study_name}_summary"},
        )

        # Run optimization
        print(f"Starting hyperparameter tuning with {n_trials} trials...")
        study.optimize(objective, n_trials=n_trials, callbacks=[wandbc])

        # Print results
        print("Study statistics: ")
        print(f"Number of finished trials:{len(study.trials)}")
        print(f"Number of pruned trials:{len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
        print(
            f"Number of complete trials:{len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}"
        )

        print("Best trial:")
        trial = study.best_trial
        print(f"  Value: {trial.value}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

        # Save study results
        os.makedirs(self.results_dir, exist_ok=True)
        study_results_path = os.path.join(self.results_dir, f"hp_tuning_study_{study_name}.pkl")
        with open(study_results_path, "wb") as f:
            pickle.dump(study, f)

        return trial, study

    def compute_statistics(self, deployment_approach):
        # compute mean, std over results
        results_df = pd.DataFrame(
            columns=[
                "mean_interventions",
                "std_interventions",
                "mean_failsafe_interventions",
                "std_failsafe_interventions",
                "mean_safety_violations",
                "std_safety_violations",
                "mean_return",
                "std_return",
                "mean_return_without_pun",
                "std_return_without_pun",
            ]
        )
        for i in range(1, self.n_runs + 1):
            results_path = self.results_dir + "/" + deployment_approach + f"/_{i}/deploy_logs.csv"
            if i == 1:
                results = pd.read_csv(results_path, delimiter=";")
            else:
                results = pd.concat([results, pd.read_csv(results_path, delimiter=";")], ignore_index=True, axis=0)
        for col in results.columns:
            results_df["mean_" + col] = [results[col].mean()]
            results_df["std_" + col] = [results[col].std()]

        results_df.to_csv(self.results_dir + "/total_results_" + deployment_approach + ".csv")
