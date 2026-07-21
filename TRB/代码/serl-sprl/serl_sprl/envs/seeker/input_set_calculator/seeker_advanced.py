import itertools
from copy import deepcopy
from typing import Any

import cvxpy as cp
import numpy as np
import pygame
from continuoussets import Interval
from continuoussets.convexsets import Zonotope

from serl_sprl.envs.seeker.geometry_utils import circle_contains_point, order_vertices_clockwise
from serl_sprl.envs.seeker.input_set_calculator.base import RelevantInputSetCalculator, SafeInputCalculator
from serl_sprl.envs.seeker.input_set_calculator.definitions import Set, ZonoOptimizationMode


def _interval_vertices(lb_var, ub_var, dim):
    vertices = []
    for binary_selection in itertools.product([0, 1], repeat=dim):
        vertex = cp.hstack(
            [(1 - binary_selection[i]) * lb_var[i] + binary_selection[i] * ub_var[i] for i in range(dim)]
        )
        vertices.append(vertex)

    vertices_matrix = cp.vstack(vertices)
    return vertices_matrix


class SeekerSafeInputSetCalculator(RelevantInputSetCalculator, SafeInputCalculator):
    _current_safe_input_set: Set = None

    def __init__(
        self,
        template_set: Set,
        optimization_mode: ZonoOptimizationMode | None,
        noise: float | None = None,
        dim: int = 2,
        debug: bool = False,
    ):
        self.template_set = template_set
        self.optimization_mode = optimization_mode
        self.debug = debug
        self.dim = dim
        self._current_safe_input_set = template_set
        self.max_noise_single_dim = 0 if noise is None else abs(noise)
        self.noise_set = (
            None
            if noise is None
            else Interval(lb=-np.ones(dim) * self.max_noise_single_dim, ub=np.ones(dim) * self.max_noise_single_dim)
        )
        self.max_noise = 0 if noise is None else np.linalg.norm(self.noise_set.ub)

    def _get_attributes_from_obs(self, observation: np.ndarray) -> tuple[np.ndarray, ...]:
        agent_position = observation[: self.dim]
        goal_position = observation[self.dim : 2 * self.dim]

        obstacle_data = observation[2 * self.dim :]
        num_obstacles = len(obstacle_data) // (self.dim + 1)

        obstacles_data = obstacle_data.reshape(num_obstacles, self.dim + 1)

        obstacle_position = obstacles_data[:, :-1]
        obstacle_radius = obstacles_data[:, -1]

        return agent_position, goal_position, obstacle_position, obstacle_radius

    def _halfspace_constraint(self, agent_position: np.array, obstacle_position: np.array, obstacle_radius):
        a = obstacle_position - agent_position
        a /= np.linalg.norm(a)
        b = np.linalg.norm(obstacle_position - agent_position)
        b -= obstacle_radius
        b -= self.max_noise
        return b, a

    def reset(self):
        self._current_safe_input_set = self.template_set.copy()

    def _obstacle_constraint_zonotope(self, obstacle_position, obstacle_radius, agent_position, c_1, G_U):
        # X_1 \cap Obstacle = \{ \}
        # i.e., X_1 \in \{ x | a_T x \leq b \}
        c3 = []
        for i in range(len(obstacle_position)):
            a = obstacle_position[i] - agent_position
            a /= np.linalg.norm(a)
            b = np.dot(a, obstacle_position[i] - a * obstacle_radius[i])
            c3.append(a @ c_1 + cp.sum(cp.abs(a @ G_U)) <= (b - self.max_noise))
        return c3

    def _debug_plot_halfspace(self, G_U, c_U, obstacle_position, agent_position, obstacle_radius, boundary_size):
        # Plot zonotope
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is missing. Please install with `poetry install --with tests`.")

        input_set = Zonotope(G=G_U.value.T, c=c_U.value)
        state_set = deepcopy(input_set)
        state_set.c += agent_position

        fig, ax = plt.subplots()
        vertices = state_set.vertices()

        a = obstacle_position - agent_position
        a /= np.linalg.norm(a)
        b = np.dot(a, obstacle_position - a * obstacle_radius)

        ax.plot(vertices[0, :], vertices[1, :], "k", label="state set")

        point = obstacle_position - obstacle_radius * a
        ax.plot(point[0], point[1], "ro", label="Point")

        # Plot the halfspace \{ x | a_T x \leq b \}
        x = np.linspace(-boundary_size, boundary_size, 100)
        y = (b - a[0] * x) / a[1]
        mask = (y >= -boundary_size) & (y <= boundary_size)
        x = x[mask]
        y = y[mask]
        ax.plot(x, y, "r", label="Halfspace")

        # Plot a
        ax.quiver(point[0], point[1], a[0], a[1], color="g", label="a")

        # Plot the obstacle
        circle = plt.Circle(
            obstacle_position.tolist(),
            obstacle_radius.item(),
            color="r",
            fill=False,
            label="Obstacle",
        )
        ax.add_artist(circle)

        # Plot the agent
        ax.plot(agent_position[0], agent_position[1], "bo", label="Agent")

        ax.set_aspect("equal")
        plt.legend()

        plt.show()

    def _compute_as_zonotope(self, observation: np.ndarray, info: dict[str, Any]) -> Zonotope:
        # From SeekerEnv _get_obs
        agent_position, _, obstacle_position, obstacle_radius = self._get_attributes_from_obs(observation)
        boundary_size = info["boundary_size"]
        action_range = info["action_range"]

        tU = self.template_set

        n_U = self.dim
        nG_U = self.template_set.number_generators()

        c_U = cp.Variable(n_U)
        G_U = cp.Variable((n_U, nG_U))
        # G_U = cp.Variable((nG_U, n_U))

        # scaling factors for template input set
        p = cp.Variable((nG_U, 1), nonneg=True)
        c0 = G_U - tU.G.T @ cp.diag(p) == 0

        def support_function(d, G, c):
            return d @ c + cp.sum([cp.abs(d @ G[:, i]) for i in range(G.shape[1])])

        # U_\phi \in U_F
        c11 = c_U + cp.sum(cp.abs(G_U), axis=1) <= action_range
        c12 = -c_U + cp.sum(cp.abs(G_U), axis=1) <= action_range

        # X_1 \in X_s
        c_1 = agent_position + c_U
        c21 = c_1 + cp.sum(cp.abs(G_U), axis=1) <= (boundary_size - self.max_noise_single_dim)
        c22 = -c_1 + cp.sum(cp.abs(G_U), axis=1) <= (boundary_size - self.max_noise_single_dim)

        # X_1 \cap Obstacle = \{ \}
        # i.e., X_1 \in \{ x | a_T x \leq b \}
        c3 = self._obstacle_constraint_zonotope(obstacle_position, obstacle_radius, agent_position, c_1, G_U)

        constraints = [c0, c11, c12, c21, c22] + c3

        if self.optimization_mode is ZonoOptimizationMode.VOL_MAX:
            objective = cp.Maximize(cp.geo_mean(p))
        else:
            raise NotImplementedError(f"Mode {self.optimization_mode} not implemented.")

        problem = cp.Problem(objective, constraints)
        problem.solve(cp.CLARABEL)

        if self.debug:
            self._debug_plot_halfspace(G_U, c_U, obstacle_position, agent_position, obstacle_radius, boundary_size)

        if problem.status not in [cp.OPTIMAL]:
            raise ValueError(f"Safe input set cannot be calculated; Problem.status: {problem.status}")

        result = Zonotope(G=G_U.value.T.astype(np.float32), c=c_U.value.astype(np.float32))
        self._current_safe_input_set = result

        return result

    def _compute_as_interval(self, observation: np.ndarray, info: dict[str, Any]) -> Interval:

        agent_position, _, obstacle_position, obstacle_radius = self._get_attributes_from_obs(observation)
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
            # boundary constraints
            cp.abs(agent_position + lb_var) <= boundary_size - self.max_noise_single_dim,
            cp.abs(agent_position + ub_var) <= boundary_size - self.max_noise_single_dim,
        ]

        # obstacle constraints
        vertices = _interval_vertices(lb_var, ub_var, len(agent_position))
        for i in range(len(obstacle_position)):
            b, a = self._halfspace_constraint(agent_position, obstacle_position[i], obstacle_radius[i])
            constraints += [vertices @ a <= b]

        objective = cp.Maximize(cp.sum(cp.log(ub_var - lb_var)))

        problem = cp.Problem(objective, constraints)
        problem.solve(cp.CLARABEL)

        if problem.status not in [cp.OPTIMAL]:
            raise ValueError(f"Safe input set cannot be calculated; Problem.status: {problem.status}")

        interval = Interval(lb=lb_var.value, ub=ub_var.value)
        self._current_safe_input_set = interval
        return interval

    def compute_input_set(self, observation: np.ndarray, info: dict[str, Any]) -> Set:
        if isinstance(self.template_set, Zonotope):
            return self._compute_as_zonotope(observation, info)

        if isinstance(self.template_set, Interval):
            return self._compute_as_interval(observation, info)

    def compute_input(self, observation: np.ndarray, info: dict[str, Any], action: np.ndarray | None) -> np.ndarray:
        agent_position, _, obstacle_position, obstacle_radius = self._get_attributes_from_obs(observation)
        size: float = info["boundary_size"]
        action_range = info["action_range"]

        def action_safe(test_action):
            next_pos = agent_position + test_action
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

        return np.zeros(self.dim)

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
        assert self._current_safe_input_set is not None

        agent_position = self._get_attributes_from_obs(observation)[0]
        action_pos = agent_position + action

        stretch_factor = screen_dim / (2 * grid_size)
        pygame.draw.circle(
            surface,
            (128, 0, 128),
            (
                int(((action_pos[x] + grid_size) * stretch_factor) + offset * screen_dim),
                int((action_pos[y] + grid_size) * stretch_factor),
            ),
            int(0.02 * grid_size * stretch_factor),
        )

        safe_input_set = self._current_safe_input_set.copy()
        safe_vertices = safe_input_set.vertices() + agent_position
        safe_input_set_polygon = np.array(
            [
                (
                    int(((point[x] + grid_size) * stretch_factor) + offset * screen_dim),
                    int((point[y] + grid_size) * stretch_factor),
                )
                for point in safe_vertices
            ]
        )
        safe_input_set_polygon = order_vertices_clockwise(safe_input_set_polygon)
        pygame.draw.polygon(surface, (0, 0, 0), safe_input_set_polygon, 1)
