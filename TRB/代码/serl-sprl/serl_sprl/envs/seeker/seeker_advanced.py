import math
from typing import Any, Callable

import gymnasium as gym
import numpy as np
import pygame
from continuoussets import Interval
from gymnasium.core import ObsType, RenderFrame

from serl_sprl.envs.seeker.geometry_utils import any_circle_overlap, circle_contains_point, order_vertices_clockwise


class SeekerEnvAdvanced(gym.Env):
    screen = None
    screen_dim = 512
    clock = None
    _size = 10
    _random_retries = 1000

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10,
    }

    def __init__(
        self,
        randomize_env: bool = True,
        render_mode: str = None,
        num_obstacles: int = 3,
        render_hook: Callable[[np.ndarray, pygame.Surface, np.ndarray, int, int, int, int, int], None] = None,
        noise: float | None = None,
        dim: int = 2,
        agent_range: float = 1,
        render_axis=None,
        collision_reward: float = 0,
        goal_reward: float = 0.0,
        done_on_collision: bool = False,
        done_on_goal: bool = True,
        dtype: type = np.float32,
        dt: float = 1.0,
    ):
        if render_axis is None:
            render_axis = set()
            render_axis.add(frozenset({0, 1}))
        self.dim = dim
        self._dtype = dtype
        self.dt = dt
        self._num_obstacles = num_obstacles
        self.randomize = randomize_env
        self.remaining_retries = self._random_retries
        self.render_mode = render_mode
        self.noise_set = (
            None
            if noise is None or noise == 0.0
            else Interval(lb=-np.ones(dim) * abs(noise), ub=np.ones(dim) * abs(noise))
        )
        self.agent_range = agent_range
        self.render_hook = render_hook
        self.render_axis = render_axis

        obs_shape = 2 * dim + num_obstacles * (dim + 1)
        self.observation_space = gym.spaces.Box(low=-self._size, high=self._size, shape=(obs_shape,), dtype=self._dtype)
        self.action_space = gym.spaces.Box(
            low=-self.agent_range, high=self.agent_range, shape=(dim,), dtype=self._dtype
        )

        # rewards and dones
        self._collision_reward = collision_reward
        self._goal_reward = goal_reward
        self._done_on_collision = done_on_collision
        self._done_on_goal = done_on_goal
        self._collision = False

        # Observations
        self._agent_position = np.zeros(dim, dtype=self._dtype)
        self._agent_radius = 0.02 * self._size
        self._obstacle_position = np.zeros((num_obstacles, dim), dtype=self._dtype)
        self._obstacle_radius = np.zeros(num_obstacles, dtype=self._dtype)
        self._goal_position = np.zeros(dim, dtype=self._dtype)
        self._goal_radius = 0.05 * self._size
        self._collision_tolerance = 1e-5

        self._action = np.zeros(dim, dtype=self._dtype)
        self._last_action = None
        self._input_change = None
        self._intended_action = np.zeros(dim, dtype=self._dtype)

    def _get_obs(self) -> np.ndarray:
        obstacles_data = np.column_stack((self._obstacle_position, self._obstacle_radius))
        return np.concatenate([self._agent_position, self._goal_position, obstacles_data.flatten()]).astype(self._dtype)

    def _get_info(self):
        return {
            "distance": np.linalg.norm(self._goal_position - self._agent_position),
            "boundary_size": self._size,
            "action_range": self.agent_range,
            "collision": self._collision,
            "input_change": self._input_change,
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self._collision = False
        self.remaining_retries = self._random_retries

        if self.randomize:
            self._random_initialization()
        else:
            self._deterministic_initialization()

        if self.render_mode == "human":
            self.render()

        self.state = self._get_obs()

        return self._get_obs(), self._get_info()

    def _deterministic_initialization(self):
        self._agent_position = (
            np.zeros(self.dim, dtype=self._dtype) - 6 + np.random.uniform(-0.1, 0.1, self.dim).astype(self._dtype)
        )
        self._action = np.zeros(self.dim, dtype=self._dtype)
        self._intended_action = np.zeros(self.dim, dtype=self._dtype)
        self._goal_position = np.zeros(self.dim, dtype=self._dtype) + 6

        if self._num_obstacles == 1:
            self._obstacle_position = np.zeros((1, self.dim), dtype=self._dtype)
            self._obstacle_radius = np.array([4], dtype=self._dtype)
        else:
            self._num_obstacles = 3
            self._obstacle_position = np.zeros((self._num_obstacles, self.dim), dtype=self._dtype)
            self._obstacle_position[1][0] -= 7
            self._obstacle_position[1][1] += 4
            self._obstacle_position[2][0] += 4
            self._obstacle_position[2][1] -= 7

            self._obstacle_radius = np.array([4, 2, 2], dtype=self._dtype)

    def _random_init_single(self):
        self._agent_position = np.random.uniform(-self._size + 1, self._size - 1, (self.dim,))
        self._action = np.zeros(self.dim, dtype=self._dtype)
        self._intended_action = np.zeros(self.dim, dtype=self._dtype)

        # Sample the goal position from the other half of the box, where the agent is not
        self._goal_position = np.random.uniform(-self._size + 1, self._size - 1, (self.dim,))
        while np.linalg.norm(self._agent_position - self._goal_position) < self._size / 2:
            self._goal_position = np.random.uniform(-self._size + 1, self._size - 1, (self.dim,))

        # Get a random point on the line between agent and goal
        obstacle_pos = self._agent_position + np.random.uniform(0.4, 0.6) * (self._goal_position - self._agent_position)

        min_radius = 1
        max_radius = min(
            np.linalg.norm(self._agent_position - self._goal_position) / 3,
            np.max(np.concatenate((obstacle_pos - self._size, obstacle_pos + self._size))),
        )

        radius = np.random.uniform(min_radius, max_radius)

        if circle_contains_point(obstacle_pos, radius, self._agent_position) or circle_contains_point(
            obstacle_pos, radius, self._goal_position
        ):
            print("Repeating random initialization!")
            self._random_initialization()

        # Construct the obstacle
        self._obstacle_position = np.array([obstacle_pos])
        self._obstacle_radius = np.array([radius])

    def _random_init_multiple(self):
        # use single implementation to ensure there is an obstacle between agent and goal
        self._random_init_single()

        # add remaining obstacles at random, ensuring that none overlap to form a barrier for the agent
        entity_positions = [self._agent_position, self._goal_position, self._obstacle_position[0]]
        entity_radii = [self._agent_radius, self._goal_radius, self._obstacle_radius[0]]

        max_radius = self._size / math.sqrt(self._size)
        min_radius = self._agent_radius
        for _ in range(self._num_obstacles - 1):
            # Create a random obstacles parameters
            temp_obstacle_pos = np.random.uniform(-self._size, self._size, (self.dim,))
            temp_obstacle_radius = np.random.uniform(min_radius, max_radius)
            # Check for overlaps (safety distance to make sure that the path to the goal is not blocked)
            while any_circle_overlap(
                temp_obstacle_pos,
                temp_obstacle_radius,
                entity_positions,
                entity_radii,
                safety_distance=2 * self._agent_radius + 1,
            ):
                self.remaining_retries -= 1
                temp_obstacle_pos = np.random.uniform(-self._size, self._size, (self.dim,))
                temp_obstacle_radius = np.random.uniform(min_radius, max_radius)
                if self.remaining_retries <= 0:
                    break

            if self.remaining_retries <= 0:
                gym.logger.warn(
                    "You have run out of retries to find valid obstacle positions.\n"
                    f"random_retries = {self._random_retries}; num_obstacles = {self._num_obstacles}; "
                    f"obstacles inserted = {len(self._obstacle_position)}"
                )
                break

            entity_positions += [temp_obstacle_pos]
            entity_radii += [temp_obstacle_radius]

            self._obstacle_position = np.concatenate((self._obstacle_position, [temp_obstacle_pos]))
            self._obstacle_radius = np.concatenate((self._obstacle_radius, [temp_obstacle_radius]))

        self.remaining_retries = self._random_retries
        self._num_obstacles = len(self._obstacle_position)

    def _random_initialization(self):
        if self._num_obstacles == 1:
            self._random_init_single()
        else:
            self._random_init_multiple()

    def _agent_update(self):
        self._agent_position = self._agent_position + self._action

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._collision = False
        self._action = action.copy()
        self._input_change = (self._action - self._last_action) / self.dt if self._last_action is not None else None
        self._last_action = action.copy()
        self._intended_action = action.copy()
        self._agent_position.copy()
        if self.noise_set is not None:
            self._action += np.random.uniform(self.noise_set.lb, self.noise_set.ub)

        if self.render_mode == "human":
            self.render()

        self._agent_update()

        dist_to_goal = np.linalg.norm(self._goal_position - self._agent_position)
        # time penalty
        time_penalty = -1

        reward = time_penalty + np.exp(-dist_to_goal / self._size)
        terminated = False
        truncated = False

        if self._check_collision():
            self._collision = True
            reward = self._collision_reward
            terminated = self._done_on_collision

        if self._goal_reached():
            reward = self._goal_reward
            terminated = self._done_on_goal

        self.state = self._get_obs()

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _goal_reached(self) -> bool:
        return np.linalg.norm(self._goal_position - self._agent_position) < (self._goal_radius + self._agent_radius)

    def _check_collision(self) -> bool:
        if np.any(np.abs(self._agent_position) - self._collision_tolerance > self._size):
            return True

        for i in range(self._num_obstacles):
            if circle_contains_point(self._obstacle_position[i], self._obstacle_radius[i].item(), self._agent_position):
                return True

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        return self._render_frame()

    def _draw_agent(self, surface, x, y, offset):
        stretch_factor = self.screen_dim / (2 * self._size)
        pygame.draw.circle(
            surface,
            (0, 0, 255),
            (
                int(((self._agent_position[x] + self._size) * stretch_factor) + offset * self.screen_dim),
                int((self._agent_position[y] + self._size) * stretch_factor),
            ),
            int(self._agent_radius * stretch_factor),
        )

    def _draw_obstacle(self, surface, x, y, offset):
        stretch_factor = self.screen_dim / (2 * self._size)
        for i in range(self._num_obstacles):
            pygame.draw.circle(
                surface,
                (255, 0, 0),
                (
                    int(((self._obstacle_position[i][x] + self._size) * stretch_factor) + offset * self.screen_dim),
                    int((self._obstacle_position[i][y] + self._size) * stretch_factor),
                ),
                int(self._obstacle_radius[i] * stretch_factor),
            )

    def _draw_goal(self, surface, x, y, offset):
        stretch_factor = self.screen_dim / (2 * self._size)
        pygame.draw.circle(
            surface,
            (0, 255, 0),
            (
                int(((self._goal_position[x] + self._size) * stretch_factor) + offset * self.screen_dim),
                int((self._goal_position[y] + self._size) * stretch_factor),
            ),
            int(self._goal_radius * stretch_factor),
        )

    def _draw_noise(self, surface, x, y, offset):
        intended_action_pos = self._agent_position + self._intended_action
        stretch_factor = self.screen_dim / (2 * self._size)
        pygame.draw.circle(
            surface,
            (64, 224, 208),  # turquoise
            (
                int(((intended_action_pos[x] + self._size) * stretch_factor) + offset * self.screen_dim),
                int((intended_action_pos[y] + self._size) * stretch_factor),
            ),
            int(self._agent_radius * stretch_factor),
        )
        # Draw applied noise interval
        noise_vertices = self.noise_set.vertices() + intended_action_pos
        noise_polygon = np.array(
            [
                (
                    int(((point[x] + self._size) * stretch_factor) + offset * self.screen_dim),
                    int((point[y] + self._size) * stretch_factor),
                )
                for point in noise_vertices
            ]
        )
        noise_polygon = order_vertices_clockwise(noise_polygon)
        pygame.draw.polygon(surface, (192, 192, 192), noise_polygon, 1)

    def _draw_label(self, surface, x, y, offset):
        font = pygame.font.Font(None, 30)
        text_surface = font.render(f"x: dim {x}   y: dim {y}", True, (0, 0, 0))
        surface.blit(text_surface, (int(offset * self.screen_dim), int(0)))

    def _render_frame(self):
        num_displays = len(self.render_axis)
        if self.screen is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.screen_dim * num_displays, self.screen_dim))
            pygame.display.set_caption("Seeker")

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        surface = pygame.Surface((self.screen_dim * num_displays, self.screen_dim))
        surface.fill((255, 255, 255))

        offset = 0
        for view in self.render_axis:
            ax_iter = iter(view)
            x, y = next(ax_iter), next(ax_iter)

            self._draw_obstacle(surface, x, y, offset)

            # This allows to insert any rendering from outside the class (e.g. rendering a safe action set)
            if self.render_hook is not None:
                self.render_hook(self._get_obs(), surface, self._action, x, y, offset, self.screen_dim, self._size)

            if self.noise_set is not None:
                # Draw intended agent position before noise application
                self._draw_noise(surface, x, y, offset)

            self._draw_agent(surface, x, y, offset)
            self._draw_goal(surface, x, y, offset)
            self._draw_label(surface, x, y, offset)
            offset += 1

        line_color = (0, 0, 0)
        line_width = 1

        for i in range(1, num_displays):
            line_x = i * self.screen_dim  # Position of the dividing line
            pygame.draw.line(surface, line_color, (line_x, 0), (line_x, self.screen_dim), line_width)

        self.screen.blit(surface, surface.get_rect())
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.update()

        else:  # rgb_array
            return np.transpose(np.array(pygame.surfarray.pixels3d(surface)), axes=(1, 0, 2))

    def close(self):
        pygame.display.quit()
        pygame.quit()

    def get_size(self):
        return self._size

    def get_obs(self):
        return self._get_obs()

    def get_info(self):
        return self._get_info()
