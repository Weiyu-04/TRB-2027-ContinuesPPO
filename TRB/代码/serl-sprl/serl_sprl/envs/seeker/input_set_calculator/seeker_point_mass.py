import math
from typing import Any

import cvxpy as cp
import numpy as np
import pygame
from continuoussets import Interval, Zonotope

from serl_sprl.envs.seeker.geometry_utils import circle_contains_point
from serl_sprl.envs.seeker.input_set_calculator.definitions import Set, ZonoOptimizationMode
from serl_sprl.envs.seeker.input_set_calculator.seeker_advanced import SeekerSafeInputSetCalculator, _interval_vertices


class SeekerSafeInputSetCalculatorPointMass(SeekerSafeInputSetCalculator):

    def __init__(
        self,
        template_set: Set,
        optimization_mode: ZonoOptimizationMode,
        delta_t: float,
        noise: Interval = None,
        dim: int = 2,
        debug: bool = False,
        conservative_safeguarding: bool = False,
    ):
        super().__init__(template_set, optimization_mode, noise, dim, debug)
        self.dt = delta_t
        self.conservative_safeguarding = conservative_safeguarding

    def _get_attributes_from_obs(self, observation: np.ndarray) -> tuple[np.ndarray, ...]:
        agent_position = observation[: self.dim]
        agent_velocity = observation[self.dim : 2 * self.dim]
        goal_position = observation[2 * self.dim : 3 * self.dim]

        obstacle_data = observation[3 * self.dim :]
        num_obstacles = len(obstacle_data) // (self.dim + 1)

        obstacles_data = obstacle_data.reshape(num_obstacles, self.dim + 1)

        obstacle_position = obstacles_data[:, :-1]
        obstacle_radius = obstacles_data[:, -1]

        return agent_position, agent_velocity, goal_position, obstacle_position, obstacle_radius

    def _acceleration_constraint(
        self, agent_position: np.array, agent_velocity: np.array, obstacle_position: np.array, obstacle_radius, a_max
    ):
        if self.conservative_safeguarding:
            a_max *= 0.9999
            dist_obs = np.linalg.norm(obstacle_position - agent_position) - obstacle_radius
            dir_obs = obstacle_position - agent_position
            dir_obs /= np.linalg.norm(dir_obs)
            vel = agent_velocity @ dir_obs

            acc = -a_max - vel + math.sqrt((a_max - self.max_noise) * (a_max - self.max_noise + 2 * dist_obs))

        else:
            dir_obs = obstacle_position - agent_position
            dir_obs /= np.linalg.norm(dir_obs)
            vel = agent_velocity @ dir_obs

            a_max *= np.linalg.norm(dir_obs, 1)
            a_max *= 0.9999  # numerical stability

            dist_obs = np.linalg.norm(obstacle_position - agent_position) - obstacle_radius

            acc = -a_max - vel + math.sqrt((a_max - self.max_noise) * (a_max - self.max_noise + 2 * dist_obs))

            acc /= self.dt

        return acc, dir_obs

    def _compute_as_zonotope(self, observation: np.ndarray, info: dict[str, Any]) -> Zonotope:
        # From SeekerEnvAdvanced _get_obs
        agent_position, agent_velocity, _, obstacle_position, obstacle_radius = self._get_attributes_from_obs(
            observation
        )

        boundary_size = info["boundary_size"]
        action_range = info["action_range"]

        tU = self.template_set

        n_U = self.dim
        nG_U = self.template_set.G.shape[1]

        c_U = cp.Variable(n_U)
        G_U = cp.Variable((n_U, nG_U))
        # G_U = cp.Variable((nG_U, n_U))

        def support_function(d, G, c):
            return d @ c + cp.sum([cp.abs(d @ G[:, i]) for i in range(G.shape[1])])

        p = cp.Variable((nG_U, 1), nonneg=True)

        constraints = [
            # scaling factors for template input set
            G_U - tU.G.T @ cp.diag(p) == 0,
            # U_\phi \in U_F
            c_U + cp.sum(cp.abs(G_U), axis=1) <= action_range,
            -c_U + cp.sum(cp.abs(G_U), axis=1) <= action_range,
        ]

        # boundary constraints
        for i in range(self.dim):
            boundary_pos = agent_position.copy()
            boundary_pos[i] = boundary_size
            b, a = self._acceleration_constraint(agent_position, agent_velocity, boundary_pos, 1e-5, action_range)
            constraints += [a @ c_U + cp.sum(cp.abs(a @ G_U)) <= b]

            boundary_pos[i] = -boundary_size
            b, a = self._acceleration_constraint(agent_position, agent_velocity, boundary_pos, 1e-5, action_range)
            constraints += [a @ c_U + cp.sum(cp.abs(a @ G_U)) <= b]

        for i in range(len(obstacle_position)):
            b, a = self._acceleration_constraint(
                agent_position, agent_velocity, obstacle_position[i], obstacle_radius[i], action_range
            )
            constraints += [a @ c_U + cp.sum(cp.abs(a @ G_U)) <= b]

        if self.optimization_mode is ZonoOptimizationMode.VOL_MAX:
            objective = cp.Maximize(cp.geo_mean(p))
        else:
            raise NotImplementedError(f"Mode {self.optimization_mode} not implemented.")

        problem = cp.Problem(objective, constraints)
        problem.solve(cp.GUROBI)

        if self.debug:
            self._debug_plot_halfspace(G_U, c_U, obstacle_position, agent_position, obstacle_radius, boundary_size)

        if problem.status not in [cp.OPTIMAL]:
            raise ValueError(f"Safe input set cannot be calculated; Problem.status: {problem.status}")

        result = Zonotope(G=G_U.value.T.astype(np.float32), c=c_U.value.astype(np.float32))
        self._current_safe_input_set = result

        return result

    def _compute_as_interval(self, observation: np.ndarray, info: dict[str, Any]) -> Interval:
        agent_position, agent_velocity, _, obstacle_position, obstacle_radius = self._get_attributes_from_obs(
            observation
        )

        boundary_size = info["boundary_size"]
        action_range = info["action_range"]

        dim = len(agent_position)
        lb_var = cp.Variable(dim)
        ub_var = cp.Variable(dim)

        constraints = [
            # template constraints
            lb_var >= self.template_set.lb,
            ub_var <= self.template_set.ub,
            # action range constraints
            cp.abs(lb_var) <= action_range,
            cp.abs(ub_var) <= action_range,
        ]

        vertices = _interval_vertices(lb_var, ub_var, len(agent_position))

        # boundary constraints
        for i in range(self.dim):
            boundary_pos = agent_position.copy()
            boundary_pos[i] = boundary_size
            b, a = self._acceleration_constraint(agent_position, agent_velocity, boundary_pos, 0.0, action_range)
            constraints += [vertices @ a <= b]

            boundary_pos[i] = -boundary_size
            b, a = self._acceleration_constraint(agent_position, agent_velocity, boundary_pos, 0.0, action_range)
            constraints += [vertices @ a <= b]

        # obstacle constraints
        for i in range(len(obstacle_position)):
            b, a = self._acceleration_constraint(
                agent_position, agent_velocity, obstacle_position[i], obstacle_radius[i], action_range
            )
            constraints += [vertices @ a <= b]

        objective = cp.Maximize(cp.sum(cp.log(ub_var - lb_var)))

        problem = cp.Problem(objective, constraints)
        problem.solve(cp.CLARABEL)

        if problem.status not in [cp.OPTIMAL]:
            raise ValueError(f"Safe input set cannot be calculated; Problem.status: {problem.status}")

        interval = Interval(lb=lb_var.value, ub=ub_var.value)
        self._current_safe_input_set = interval
        return interval

    def compute_input(self, observation: np.ndarray, info: dict[str, Any], action: np.ndarray | None) -> np.ndarray:
        agent_position, agent_velocity, _, obstacle_position, obstacle_radius = self._get_attributes_from_obs(
            observation
        )

        size: float = info["boundary_size"]
        action_range = info["action_range"]

        def action_safe(test_action):
            next_pos = agent_position + agent_velocity + test_action
            safe = np.all(np.abs(next_pos) < size) and np.all(np.abs(action) < action_range)
            for i in range(len(obstacle_radius)):
                if circle_contains_point(obstacle_position[i], obstacle_radius[i].item(), next_pos):
                    safe = False
                    break
            return safe

        if action_safe(action):
            return action

        for _ in range(100):
            action = np.random.uniform(-action_range / 10, action_range / 10, self.dim)
            if action_safe(action):
                return action

        safe_input = -agent_velocity.copy()
        return np.clip(safe_input, -action_range, action_range)

    def render_safe_input_set_pygame(
        self,
        observation: np.ndarray,
        surface: pygame.Surface,
        action: np.ndarray,
        x: int,
        y: int,
        offset: int,
        screen_dim: int,
        grid_size: int,
    ):
        """Hook for rendering the safe input set in the SeekerEnv."""

        super().render_safe_input_set_pygame(observation, surface, action, x, y, offset, screen_dim, grid_size)

        agent_position, agent_velocity, _, _, _ = self._get_attributes_from_obs(observation)
        stretch_factor = screen_dim / (2 * grid_size)

        # draw velocity line
        start = agent_position
        end = start + agent_velocity
        pygame.draw.line(
            surface,
            (64, 224, 208),
            (
                int(((start[x] + grid_size) * stretch_factor) + offset * screen_dim),
                int((start[y] + grid_size) * stretch_factor),
            ),
            (
                int(((end[x] + grid_size) * stretch_factor) + offset * screen_dim),
                int((end[y] + grid_size) * stretch_factor),
            ),
            1,
        )
