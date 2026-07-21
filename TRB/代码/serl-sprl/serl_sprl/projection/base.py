from typing import Optional

import cvxpy as cp
import gymnasium as gym
import numpy as np
import torch as th
from continuoussets.convexsets import Zonotope

from serl_sprl.envs.configs import BaseEnvConfig
from serl_sprl.projection.projection_helpers import (
    create_problem,
    create_problem_safe_action_set,
    create_proj_layer_scaled,
)
from serl_sprl.sets import zonotope_contains


class BaseProjectionSafeguard:
    def __init__(self, projection_config: dict, env: gym.Env, device: str = "cpu"):
        self.projection_config = projection_config
        self.env = env
        self.device = device
        self.projection_layer_zono = create_proj_layer_scaled(
            self.get_projection_config(), env.get_wrapper_attr("multi_step_safeguarding")
        )

    def is_action_safe(self, state: np.ndarray, action: np.ndarray) -> bool:
        raise NotImplementedError

    def project_action(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def project_policy_action(
        self, state: th.Tensor, mean_actions: th.Tensor, safe_action_set: Optional[list[Zonotope]] = None
    ) -> tuple[th.Tensor, Optional[list[Zonotope]]]:
        raise NotImplementedError

    def get_projection_config(self) -> dict:
        raise NotImplementedError


class SafeStateSetProjectionSafeguard(BaseProjectionSafeguard):
    def __init__(self, projection_config: dict, env: gym.Env, device: str = "cpu"):
        from serl_sprl.envs.safe_region import ControlInvariantSetZonotope
        from serl_sprl.sets import Zonotope

        self._safe_control_fn = env.get_wrapper_attr("_safe_control_fn")
        self._noise_set = projection_config.get("noise_set_zonotope").map(env.get_wrapper_attr("E_d"))
        super().__init__(projection_config, env, device=device)
        # As the solver is parametrized, we create the problem structures and get the path of the changeable variables
        # so we can update them in the future and them resolve the problem faster
        if isinstance(env.get_wrapper_attr("safe_region"), Zonotope):
            G_omega = env.get_wrapper_attr("safe_region").G
            c_omega = env.get_wrapper_attr("safe_region").c.flatten()
        elif isinstance(env.get_wrapper_attr("safe_region"), ControlInvariantSetZonotope):
            G_omega = env.get_wrapper_attr("safe_region").RCI_zonotope.G
            c_omega = env.get_wrapper_attr("safe_region").RCI_zonotope.c.flatten()
        else:
            raise ValueError("safe_region must be of type Zonotope or ControlInvariantSetZonotope")
        self.prob, self.u_rl_p, self.x_k_p, self.u = create_problem(
            c_u=env.get_wrapper_attr("_input_set").c.flatten(),
            G_u=env.get_wrapper_attr("_input_set").G,
            c_w=self._noise_set.c.flatten(),
            G_w_hat=self._noise_set.G,
            G_omega=G_omega,
            c_omega=c_omega,
            A_hat=env.get_wrapper_attr("A_d"),
            B_hat=env.get_wrapper_attr("B_d"),
            u_eq=env.get_wrapper_attr("u_eq"),
            x_eq=env.get_wrapper_attr("x_eq"),
            multi_step=env.get_wrapper_attr("multi_step_safeguarding"),
        )

    def is_action_safe(self, state: np.ndarray, action: np.ndarray) -> bool:
        potential_next_state = self.env.unwrapped.dynamics_fn(action)
        return self.env.get_wrapper_attr("safe_region").contains(potential_next_state)

    def project_action(self, current_state: np.ndarray, unsafe_action: np.ndarray) -> np.ndarray:
        self.u_rl_p.value = unsafe_action
        self.x_k_p.value = current_state
        # Run the solver with the new values
        self.prob.solve(verbose=False, solver=cp.GUROBI, TimeLimit=1.0)
        safe_action = self.u.value
        return safe_action

    def project_policy_action(
        self, state: th.Tensor, mean_actions: th.Tensor, safe_action_set: Optional[list[Zonotope]] = None
    ) -> th.Tensor:
        # Project the action using the projection layer
        return self.projection_layer_zono(mean_actions, state)[0], safe_action_set

    def get_projection_config(self) -> dict:
        from serl_sprl.envs.safe_region import ControlInvariantSetZonotope
        from serl_sprl.sets import Zonotope

        config = dict()
        config["G_u"] = self.env.get_wrapper_attr("_input_set").G
        config["c_u"] = self.env.get_wrapper_attr("_input_set").c.flatten()
        config["c_w"] = self._noise_set.c.flatten()
        config["G_w_hat"] = self._noise_set.G
        if isinstance(self.env.get_wrapper_attr("safe_region"), Zonotope):
            config["G_omega"] = self.env.get_wrapper_attr("safe_region").G
            config["c_omega"] = self.env.get_wrapper_attr("safe_region").c.flatten()
        elif isinstance(self.env.get_wrapper_attr("safe_region"), ControlInvariantSetZonotope):
            config["G_omega"] = self.env.get_wrapper_attr("safe_region").RCI_zonotope.G
            config["c_omega"] = self.env.get_wrapper_attr("safe_region").RCI_zonotope.c.flatten()
        else:
            raise ValueError("safe_region must be of type Zonotope or ControlInvariantSetZonotope")
        config["A_hat"] = getattr(self.env.unwrapped, "A_d")
        config["B_hat"] = getattr(self.env.unwrapped, "B_d")
        config["u_eq"] = getattr(self.env.unwrapped, "u_eq")
        config["x_eq"] = getattr(self.env.unwrapped, "x_eq")
        config["u_low"] = getattr(
            getattr(self.env.unwrapped, "action_space"), "low"
        )  # ToDo: check if this is scaled/unscaled
        config["u_high"] = getattr(getattr(self.env.unwrapped, "action_space"), "high")
        return config


class SafeActionSetProjectionSafeguard(BaseProjectionSafeguard):
    def __init__(self, projection_config: dict, env: gym.Env, device: str = "cpu"):
        # As the solver is parametrized, we create the problem structures and get the path of the changeable variables
        # so we can update them in the future and them resolve the problem faster
        self.safe_set_calculator = projection_config["safe_set_calculator"]
        generator_matrix = np.eye(env.action_space.shape[0], dtype=env.get_wrapper_attr("_dtype"))
        center = np.zeros(env.action_space.shape[0], dtype=env.get_wrapper_attr("_dtype"))
        self.dummy_safe_action_set = Zonotope(G=generator_matrix, c=center)
        super().__init__(projection_config, env, device=device)
        self.prob, self.u_rl_p, self.c_safe, self.G_safe, self.u = create_problem_safe_action_set(
            self.dummy_safe_action_set.c, self.dummy_safe_action_set.G
        )
        self.current_safe_action_set = self.dummy_safe_action_set

    def is_action_safe(self, state: np.ndarray, action: np.ndarray) -> bool:
        info = self.env.unwrapped.get_info()
        safe_action_set = self.safe_set_calculator.compute_input_set(state, info)
        self.current_safe_action_set = safe_action_set
        return zonotope_contains(safe_action_set, action)

    def project_action(self, current_state: np.ndarray, unsafe_action: np.ndarray) -> np.ndarray:
        # Change the values of the parametrized variables inside the solver
        self.u_rl_p.value = unsafe_action
        self.c_safe.value = self.current_safe_action_set.c.flatten()
        self.G_safe.value = self.current_safe_action_set.G
        # Run the solver with the new values
        self.prob.solve(verbose=False, solver=cp.GUROBI)
        safe_action = self.u.value
        return safe_action

    def project_policy_action(
        self, obs: th.Tensor, mean_actions: th.Tensor, safe_action_set: Optional[list[Zonotope]] = None
    ) -> th.Tensor:
        # Project the action using the projection layer
        info = self.env.unwrapped.get_info()

        if safe_action_set is None:
            # During "collect_rollouts" we compute the safe action set here
            safe_action_set = self.safe_set_calculator.compute_input_set(obs.detach().cpu().numpy().flatten(), info)
            self.current_safe_action_set = safe_action_set
            c_safe_tensor = th.tensor(safe_action_set.c).flatten()
            G_safe_tensor = th.tensor(safe_action_set.G)

        else:
            c_safe_list = [th.tensor(sas.c).flatten() for sas in safe_action_set]
            G_safe_list = [th.tensor(sas.G) for sas in safe_action_set]
            c_safe_tensor = th.stack(c_safe_list)
            G_safe_tensor = th.stack(G_safe_list)

        safe_mean_actions = self.projection_layer_zono(
            mean_actions, c_safe_tensor.to(self.device), G_safe_tensor.to(self.device)
        )[0]

        if not isinstance(safe_action_set, list):
            safe_action_set = [safe_action_set]
        return safe_mean_actions, safe_action_set

    def get_projection_config(self) -> dict:
        config = dict()
        config["G_omega"] = self.dummy_safe_action_set.G
        config["c_omega"] = self.dummy_safe_action_set.c.flatten()
        config["u_low"] = getattr(
            getattr(self.env.unwrapped, "action_space"), "low"
        )  # ToDo: check if this is scaled/unscaled
        config["u_high"] = getattr(getattr(self.env.unwrapped, "action_space"), "high")
        return config


class ProjectionFactory:
    def __init__(self, projection_config: dict, env_config: BaseEnvConfig, device: str = "cpu"):
        if not hasattr(env_config, "safe_region") and projection_config["safe_set_calculator"] is None:
            raise ValueError("Either `safe_region` or `safe_set_calculator` must be provided.")
        self.projection_config = projection_config
        self.env_config = env_config
        self.device = device

    def get_safeguard(self, env: gym.Env) -> BaseProjectionSafeguard:
        if self.projection_config["safe_set_calculator"] is not None:
            return SafeActionSetProjectionSafeguard(self.projection_config, env=env, device=self.device)
        else:
            return SafeStateSetProjectionSafeguard(self.projection_config, env=env, device=self.device)
