from copy import deepcopy
from typing import Callable, Literal

import gymnasium as gym
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from serl_sprl.envs.configs import BaseEnvConfig
from serl_sprl.projection.base import ProjectionFactory
from serl_sprl.projection.projection_wrappers import ActionProjectionWrapper, InformerWrapper
from serl_sprl.sets import Zonotope


def transform_action_space_fn(action: np.ndarray, action_limits: list):
    """Convert action from [-1, 1] to [u_min, u_max]."""
    # [-1,1] -> [u_min, u_max]
    return np.clip(
        ((action + 1) / 2) * (action_limits[1] - action_limits[0]) + action_limits[0],
        action_limits[0],
        action_limits[1],
    )


def inv_transform_action_space_fn(action: np.ndarray, action_limits: list):
    """Convert action from [u_min, u_max] to [-1, 1]."""
    # [a_min, a_max] -> [-1,1]
    return np.clip(((action - action_limits[0]) / (action_limits[1] - action_limits[0])) * 2 - 1, -1, 1)


class BaseCreator:
    def __init__(self, env_id: str, wrapper_kwargs: dict = None):
        self.env_id = env_id
        self.wrapper_kwargs = wrapper_kwargs

    def create_env(self, env_config: BaseEnvConfig, num_envs: int = 1, device: str = "cpu") -> gym.Env:
        def make_env(rank: int) -> Callable[[], gym.Env]:
            def _init() -> gym.Env:
                env = gym.make(self.env_id, **env_config.model_dump(exclude=["id", "max_rollout_steps", "noise_bound"]))
                env = gym.wrappers.TimeLimit(env, env_config.max_rollout_steps)
                # transformation into zonotopes
                u_eq = (env.unwrapped.action_space.high + env.unwrapped.action_space.low) / 2
                allowable_input_set_factors = np.array(
                    (env.unwrapped.action_space.high - env.unwrapped.action_space.low) / 2
                ).T
                u_space_zono = Zonotope(G=np.eye(u_eq.shape[0]) * allowable_input_set_factors, c=u_eq.reshape((-1, 1)))
                if hasattr(env_config, "noise_bound"):
                    noise_set_zonotope = Zonotope(
                        G=env_config.noise_bound * np.eye(u_eq.shape[0]).T, c=np.zeros(u_eq.shape).reshape((-1, 1))
                    )
                else:
                    noise_set_zonotope = None

                current_wrapper_kwargs = self.wrapper_kwargs.copy() if self.wrapper_kwargs is not None else {}
                current_wrapper_kwargs["noise_set_zonotope"] = noise_set_zonotope
                current_wrapper_kwargs["u_space_zono"] = u_space_zono

                # wrap environment if needed
                env = self._wrap_env(env, current_wrapper_kwargs, env_config, device)
                # add monitor wrapper for logging
                env = Monitor(env, None)
                return env

            return _init

        return DummyVecEnv([make_env(i) for i in range(num_envs)])

    def _wrap_env(self, env: gym.Env, wrapper_kwargs: dict, env_config: BaseEnvConfig, device: str) -> gym.Env:
        raise NotImplementedError


class BaselineCreator(BaseCreator):
    def _wrap_env(
        self, env: gym.Env, wrapper_kwargs: dict = None, env_config: BaseEnvConfig = None, device: str = "cpu"
    ) -> gym.Env:
        # Create baseline environment without safeguarding
        alter_action_space = (
            gym.spaces.Box(
                low=-1, high=1, shape=env.action_space.shape, dtype=wrapper_kwargs.get("dtype", env.action_space.dtype)
            )
            if wrapper_kwargs and wrapper_kwargs.get("scale_actions")
            else None
        )
        env = InformerWrapper(
            env=env,
            alter_action_space=alter_action_space,
            transform_action_space_fn=(
                transform_action_space_fn if wrapper_kwargs and wrapper_kwargs.get("scale_actions") else None
            ),
            inv_transform_action_space_fn=(
                inv_transform_action_space_fn if wrapper_kwargs and wrapper_kwargs.get("scale_actions") else None
            ),
        )
        return env


class SafeEnvCreator(BaseCreator):
    def _wrap_env(
        self, env: gym.Env, wrapper_kwargs: dict = None, env_config: BaseEnvConfig = None, device: str = "cpu"
    ) -> gym.Env:
        # Create SERL environment with penalty improvement strategy
        alter_action_space = (
            gym.spaces.Box(
                low=-1, high=1, shape=env.action_space.shape, dtype=wrapper_kwargs.get("dtype", env.action_space.dtype)
            )
            if wrapper_kwargs and wrapper_kwargs.get("scale_actions")
            else None
        )
        env = ActionProjectionWrapper(
            env,
            safeguard_factory=ProjectionFactory(projection_config=wrapper_kwargs, env_config=env_config, device=device),
            admissible_input_set=wrapper_kwargs.get("u_space_zono"),
            safe_control_fn=wrapper_kwargs.get("safe_control_fn", None),
            alter_action_space=alter_action_space,
            penalty_factor=wrapper_kwargs.get("penalty_factor", 0.0),
            transform_action_space_fn=(
                transform_action_space_fn if wrapper_kwargs and wrapper_kwargs.get("scale_actions") else None
            ),
            inv_transform_action_space_fn=(
                inv_transform_action_space_fn if wrapper_kwargs and wrapper_kwargs.get("scale_actions") else None
            ),
        )
        # ToDo: This is ugly
        env.set_safeguard(env.safeguard_factory.get_safeguard(env))
        return env


class EnvCreatorFactory:
    def __init__(
        self,
        approach: Literal["baseline", "serl", "sprl"],
        improvement_strategy: Literal["penalty", "psl", "penalty_critic", "none"],
        env_id: str,
    ):
        self.approach = approach
        self.improvement_strategy = improvement_strategy
        self.env_id = env_id

    def get_env_creator(self, wrapper_kwargs: dict = None) -> BaseCreator:
        env_wrapper_kwargs = deepcopy(wrapper_kwargs) if wrapper_kwargs is not None else {}
        if self.approach == "baseline":
            return BaselineCreator(self.env_id, wrapper_kwargs=env_wrapper_kwargs)
        elif self.approach == "serl":
            if self.improvement_strategy == "none":
                # Penalty factor is 0.0 in this case
                env_wrapper_kwargs["penalty_factor"] = 0.0
                return SafeEnvCreator(self.env_id, wrapper_kwargs=env_wrapper_kwargs)
            elif self.improvement_strategy == "penalty":
                return SafeEnvCreator(self.env_id, wrapper_kwargs=env_wrapper_kwargs)
            else:
                raise ValueError(
                    f"Unknown improvement strategy {self.improvement_strategy} for approach {self.approach}"
                )
        elif self.approach == "sprl":
            # We still wrap the environment in a safety wrapper to account for numerical errors.
            # However, the projection in the environment should be unnecessary
            # as the policy already performs safeguarding.
            if self.improvement_strategy not in ["none", "penalty_critic", "psl"]:
                raise ValueError(
                    f"Unknown improvement strategy {self.improvement_strategy} for approach {self.approach}"
                )
            # Penalty factor is always 0.0 in SPRL as it is added via the policy loss
            env_wrapper_kwargs["penalty_factor"] = 0.0
            return SafeEnvCreator(self.env_id, wrapper_kwargs=env_wrapper_kwargs)

    def set_approach(self, approach: Literal["baseline", "serl", "sprl"]):
        # Required for deployment when approach changes between safe and unsafe
        self.approach = approach
