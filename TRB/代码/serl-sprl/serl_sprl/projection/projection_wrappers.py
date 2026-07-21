import warnings

import cvxpy as cp
import gymnasium as gym
import numpy as np

from serl_sprl.projection.base import BaseProjectionSafeguard, ProjectionFactory


def fetch_fn(env, fn):
    if fn is not None:
        if isinstance(fn, str):
            fn = getattr(env, fn)
        if not callable(fn):
            raise ValueError(f"Attribute {fn} is not callable")
    return fn


class InformerWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        alter_action_space=None,
        transform_action_space_fn=None,
        inv_transform_action_space_fn=None,
    ):
        super().__init__(env)

        self._transform_action_space_fn = fetch_fn(self.env, transform_action_space_fn)
        self._inv_transform_action_space_fn = fetch_fn(self.env, inv_transform_action_space_fn)

        if not hasattr(self.env, "action_space"):
            warnings.warn("Environment has no attribute ``action_space``")

        if alter_action_space is not None:
            self.action_space = alter_action_space
            if transform_action_space_fn is None:
                warnings.warn("Set ``alter_action_space`` but no ``transform_action_space_fn``")

    def step(self, action):
        # Optional action transformation
        if self._transform_action_space_fn is not None:
            action = self._transform_action_space_fn(
                action, [self.unwrapped.action_space.low, self.unwrapped.action_space.high]
            )

        obs, reward, done, truncated, info = self.env.step(action)
        info["baseline"] = {"policy_action": action, "env_reward": reward}
        self.unwrapped.last_action = self._inv_transform_action_space_fn(
            action, [self.unwrapped.action_space.low, self.unwrapped.action_space.high]
        )

        return obs, reward, done, truncated, info


class ActionProjectionWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        admissible_input_set,
        safeguard_factory: ProjectionFactory,
        penalty_factor: float = 0.0,
        safe_control_fn=None,
        alter_action_space=None,
        transform_action_space_fn=None,
        inv_transform_action_space_fn=None,
    ):
        super().__init__(env)
        self._penalty_factor = penalty_factor
        self.safeguard_factory = safeguard_factory
        self.safeguard = None
        self._safe_control_fn = fetch_fn(self.env, safe_control_fn)
        self._input_set = admissible_input_set

        if alter_action_space is not None:
            self.action_space = alter_action_space
            if transform_action_space_fn is None or inv_transform_action_space_fn is None:
                warnings.warn(
                    "Set ``alter_action_space`` but no ``transform_action_space_fn``"
                    " or  ``inv_transform_action_space_fn``"
                )
            else:
                self._transform_action_space_fn = fetch_fn(self.env, transform_action_space_fn)
                self._inv_transform_action_space_fn = fetch_fn(self.env, inv_transform_action_space_fn)

        self._infeasible_step = False
        self.last_projected = False

    def set_safeguard(self, safeguard: BaseProjectionSafeguard):
        self.safeguard = safeguard

    def punishment_fn(self, unsafe_action, safe_action):
        return (
            -np.linalg.norm(unsafe_action - safe_action, 2) ** 2 * self._penalty_factor
        )  # ToDo: This is a mistake, should not be squared. If we rerun experiments, change.

    def step(self, action):
        """Steps through the environment with the projection of the action`.
        Args:
            action: action to step through the environment (before being projected)
        Returns:
            (observation, reward, done, info)
        """
        assert not np.any(np.isnan(action)), "NaN in action"
        self._infeasible_step = False
        # Optional action transformation
        if self._transform_action_space_fn is not None:
            action = self._transform_action_space_fn(
                action, [self.unwrapped.action_space.low, self.unwrapped.action_space.high]
            )
        # Check if action is safe
        obs = self.env.unwrapped.get_obs()
        action_is_safe = self.safeguard.is_action_safe(obs, action)

        # ToDo: In the seeker, we used to differentiate between SERL and SPRL here,
        # meaning that actions would only get projected in SERL.
        # Do we want to keep this distinction?
        if not action_is_safe:
            state = getattr(self.unwrapped, "state")
            # Project the action
            try:
                safe_action = self.safeguard.project_action(state, action)
            except cp.SolverError as e:
                print("Solver Error: {}".format(e))
                safe_action = self._safe_control_fn(self.env, self.env.get_wrapper_attr("safe_region"))
                self._infeasible_step = True

            if safe_action is None:
                print("Infeasible projection problem encountered. Using backup safe controller.")
                safe_action = self._safe_control_fn(self.env, self.env.get_wrapper_attr("safe_region"))
                self._infeasible_step = True

            obs, reward, done, truncated, info = self.env.step(safe_action)
            info["projection"] = {"env_reward": reward, "infeasible": self._infeasible_step}
            info["projection"]["safe_action"] = safe_action

            # Optional reward punishment
            punishment = self.punishment_fn(action, safe_action)
            info["projection"]["pun_reward"] = punishment
            reward += punishment
            self.last_projected = True
            info["projection"]["action_projected"] = True
            self.unwrapped.last_action = self._inv_transform_action_space_fn(
                safe_action, [self.unwrapped.action_space.low, self.unwrapped.action_space.high]
            )
        else:
            # action is safe
            obs, reward, done, truncated, info = self.env.step(action)
            info["projection"] = {"env_reward": reward, "pun_reward": 0.0, "infeasible": self._infeasible_step}
            info["projection"]["action_projected"] = False
            self.last_projected = False
            self.unwrapped.last_action = self._inv_transform_action_space_fn(
                action, [self.unwrapped.action_space.low, self.unwrapped.action_space.high]
            )

        info["projection"]["policy_action"] = action
        info["projection"]["last_projected"] = self.last_projected

        return obs, reward, done, truncated, info
