import csv
import os
import shutil
from collections import deque

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from wandb.integration.sb3 import WandbCallback


class TrainSafetyCallback(WandbCallback):
    """
    Tensorboard/W&B metrics for evaluating safety

    mean_interventions: mean number of interventions over the last 100 rollouts
    mean_fail_safe_action: mean number of times the failsafe controller had to intervene over the last 100 rollouts
    mean_safety_violations: mean number of times the system state was outside the state constraints (NOT the same as
    the safe region, which is the RCI set) over the last 100 rollouts
    """

    def __init__(
        self,
        verbose: int = 0,
        model_save_path: str = None,
        model_save_freq: int = 0,
        gradient_save_freq: int = 0,
    ):
        super(TrainSafetyCallback, self).__init__(verbose, model_save_path, model_save_freq, gradient_save_freq)
        self.total_rollout_policy_interventions = deque(maxlen=100)
        self.total_rollout_interventions = deque(maxlen=100)
        self.total_rollout_fail_safe_action = deque(maxlen=100)
        self.total_rollout_safety_violations = deque(maxlen=100)
        self.total_rollout_return_without_pun = deque(maxlen=100)
        self.total_rollout_pun = deque(maxlen=100)
        self._reset()

    def _reset(self):
        self.rollout_policy_interventions = 0.0
        self.rollout_interventions = 0.0
        self.rollout_fail_safe_action = 0.0
        self.rollout_safety_violations = 0.0
        self.total_env_reward = 0.0
        self.rollout_punishment = 0.0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if "projection" in infos[0]:
            wrapper_infos = [info["projection"] for info in infos]
            if "policy_projected" in wrapper_infos[0]:
                self.rollout_policy_interventions += np.sum(
                    [wrapper_info["policy_projected"] for wrapper_info in wrapper_infos]
                )
            self.rollout_interventions += np.sum([wrapper_info["last_projected"] for wrapper_info in wrapper_infos])
            self.rollout_fail_safe_action += np.sum([wrapper_info["infeasible"] for wrapper_info in wrapper_infos])
            self.rollout_safety_violations += np.sum([info["collision"] for info in infos])
            if wrapper_infos[0]["pun_reward"] is not None:
                self.rollout_punishment += np.mean([wrapper_info["pun_reward"] for wrapper_info in wrapper_infos])
        if "baseline" in infos[0]:
            wrapper_infos = [info["baseline"] for info in infos]
            self.rollout_safety_violations += np.sum([info["collision"] for info in infos])

        # General information
        self.total_env_reward += np.mean(
            [wrapper_info["env_reward"] for wrapper_info in wrapper_infos]
        )  # reward without penalty

        if np.any(self.locals.get("dones")) or any([info["TimeLimit.truncated"] for info in infos]):
            self.total_rollout_policy_interventions.append(self.rollout_policy_interventions)
            self.total_rollout_interventions.append(self.rollout_interventions)
            self.total_rollout_fail_safe_action.append(self.rollout_fail_safe_action)
            self.total_rollout_safety_violations.append(self.rollout_safety_violations)
            self.total_rollout_return_without_pun.append(self.total_env_reward)
            self.total_rollout_pun.append(self.rollout_punishment)
            self._reset()

        return True

    def _on_rollout_end(self) -> None:
        self.logger.record("safety/ep_mean_policy_interventions", np.mean(self.total_rollout_policy_interventions))
        self.logger.record("safety/ep_mean_interventions", np.mean(self.total_rollout_interventions))
        self.logger.record("safety/ep_mean_failsafe_interventions", np.mean(self.total_rollout_fail_safe_action))
        self.logger.record("safety/ep_mean_violations", np.mean(self.total_rollout_safety_violations))
        self.logger.record("safety/return_without_pun", np.mean(self.total_rollout_return_without_pun))
        self.logger.record("safety/ep_mean_punishment", np.mean(self.total_rollout_pun))
        # self.logger.dump(step=self.n_calls)


class DeploySafetyCallback(BaseCallback):
    """
    Tensorboard/W&B metrics for evaluating safety

    mean_interventions: mean number of interventions over the last 100 rollouts
    mean_fail_safe_action: mean number of times the failsafe controller had to intervene over the last 100 rollouts
    mean_safety_violations: mean number of times the system state was outside the state constraints (NOT the same as
    the safe region, which is the RCI set) over the last 100 rollouts
    """

    def __init__(self):
        super().__init__()
        self._reset()

    def _reset(self):
        self.total_rollout_policy_interventions = deque(maxlen=100)
        self.total_rollout_interventions = deque(maxlen=100)
        self.total_rollout_fail_safe_action = deque(maxlen=100)
        self.total_rollout_safety_violations = deque(maxlen=100)
        self.total_rollout_return_without_pun = deque(maxlen=100)
        self.total_rollout_return = deque(maxlen=100)
        self.total_rollout_pun = deque(maxlen=100)
        self.rollout_policy_interventions = 0.0
        self.rollout_interventions = 0.0
        self.rollout_fail_safe_action = 0.0
        self.rollout_safety_violations = 0.0
        self.rollout_return_without_pun = 0.0
        self.rollout_punishment = 0.0
        self.rollout_return = 0.0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")[0]
        if "projection" in infos:
            wrapper_info = infos["projection"]
            if "policy_projected" in wrapper_info:
                self.rollout_policy_interventions += np.sum(wrapper_info["policy_projected"])
            self.rollout_interventions += np.sum(wrapper_info["last_projected"])
            self.rollout_fail_safe_action += np.sum(wrapper_info["infeasible"])
            self.rollout_safety_violations += np.sum(infos["collision"])
            if wrapper_info["pun_reward"] is not None:
                self.rollout_punishment += wrapper_info["pun_reward"]
        if "baseline" in infos:
            wrapper_info = infos["baseline"]
            self.rollout_safety_violations += np.sum(infos["collision"])

        # General information
        self.rollout_return_without_pun += wrapper_info["env_reward"]  # reward without penalty
        self.rollout_return = self.rollout_return_without_pun + self.rollout_punishment

        if self.locals.get("dones")[0] or infos["TimeLimit.truncated"]:
            self.total_rollout_policy_interventions.append(self.rollout_policy_interventions)
            self.total_rollout_interventions.append(self.rollout_interventions)
            self.total_rollout_fail_safe_action.append(self.rollout_fail_safe_action)
            self.total_rollout_safety_violations.append(self.rollout_safety_violations)
            self.total_rollout_return_without_pun.append(self.rollout_return_without_pun)
            self.total_rollout_pun.append(self.rollout_punishment)
            self.total_rollout_return.append(self.rollout_return)

        return True

    def _on_rollout_end(self) -> None:
        self.logger.record("safety/ep_mean_policy_interventions", np.mean(self.total_rollout_policy_interventions))
        self.logger.record("safety/ep_mean_interventions", np.mean(self.total_rollout_interventions))
        self.logger.record("safety/ep_mean_failsafe_interventions", np.mean(self.total_rollout_fail_safe_action))
        self.logger.record("safety/ep_mean_violations", np.mean(self.total_rollout_safety_violations))
        self.logger.record("safety/return_without_pun", np.mean(self.total_rollout_return_without_pun))
        self.logger.record("safety/ep_mean_punishment", np.mean(self.total_rollout_pun))
        self.logger.record("rollout/return", np.mean(self.total_rollout_return))
        self.logger.dump(step=self.n_calls)
        data = [
            np.mean(self.total_rollout_policy_interventions),
            np.mean(self.total_rollout_interventions),
            np.mean(self.total_rollout_fail_safe_action),
            np.mean(self.total_rollout_safety_violations),
            np.mean(self.total_rollout_return),
            np.mean(self.total_rollout_return_without_pun),
        ]
        if os.path.exists(self.csv_log_dir):
            with open(self.csv_log_dir, "a") as csvfile:
                csv_writer = csv.writer(csvfile, delimiter=";")
                csv_writer.writerow(data)
        else:
            with open(self.csv_log_dir, "w") as csvfile:
                csv_writer = csv.writer(csvfile, delimiter=";")
                csv_writer.writerow(
                    [
                        "policy_interventions",
                        "interventions",
                        "failsafe_interventions",
                        "safety_violations",
                        "return",
                        "return_without_pun",
                    ]
                )
                csv_writer.writerow(data)

        self._reset()

    def _on_rollout_start(self) -> None:
        self.csv_log_dir = self.logger.get_dir() + "/deploy_logs.csv"
        # we want to generate fresh results whenever we run this script
        if os.path.exists(os.getcwd() + f"/{self.csv_log_dir}") and os.path.isdir(os.getcwd() + f"/{self.csv_log_dir}"):
            shutil.rmtree(os.getcwd() + f"/{self.csv_log_dir}")
