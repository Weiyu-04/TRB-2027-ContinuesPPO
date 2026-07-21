from typing import Callable

import gymnasium as gym
import numpy as np
import pygame
from continuoussets import Interval, Zonotope
from stable_baselines3.common.vec_env import DummyVecEnv

from serl_sprl.envs.configs import BaseEnvConfig, BaseProjectionConfig
from serl_sprl.envs.seeker.input_set_calculator.base import RelevantInputSetCalculator
from serl_sprl.envs.seeker.input_set_calculator.definitions import ZonoOptimizationMode
from serl_sprl.envs.seeker.input_set_calculator.seeker_point_mass import SeekerSafeInputSetCalculatorPointMass
from serl_sprl.envs.seeker.seeker_advanced import SeekerEnvAdvanced


def safe_control_fn(env):
    print("Fail Safe Action")
    agent_pos = env.get_attr("_agent_position") if isinstance(env, DummyVecEnv) else env.unwrapped._agent_position
    obstacle_pos = (
        env.get_attr("_obstacle_position") if isinstance(env, DummyVecEnv) else env.unwrapped._obstacle_position
    )
    obstacle_radius = (
        env.get_attr("_obstacle_radius") if isinstance(env, DummyVecEnv) else env.unwrapped._obstacle_radius
    )
    env_size = env.get_attr("size") if isinstance(env, DummyVecEnv) else env.unwrapped.size

    action = agent_pos - obstacle_pos
    action = action / (np.linalg.norm(action) * 5)

    # Check if action would lead out of bounds
    if ((agent_pos + action) >= env_size).any():
        action = -action

    # Check if action now would collide with the obstacle
    if np.linalg.norm(agent_pos + action) <= obstacle_radius:
        # 90 degrees
        action = np.array([action[1], -action[0]])

        # Check if action would lead out of bounds
        if ((agent_pos + action) >= env_size).any():
            action = -action

    return action


class SeekerEnvConfig(BaseEnvConfig):
    randomize_env: bool
    max_rollout_steps: int = 100
    id: str = "serl_sprl/seeker"
    collision_reward: float = 0.0
    goal_reward: float = 0.0
    done_on_collision: bool = False
    done_on_goal: bool = True
    agent_range: float = 1.0
    num_obstacles: int = 3
    # Dynamics
    dt: float = 1.0


class SeekerProjConfig(BaseProjectionConfig):
    safe_set_calculator: RelevantInputSetCalculator = SeekerSafeInputSetCalculatorPointMass(
        template_set=Zonotope(
            G=np.eye(2, dtype=np.float32),
            c=np.zeros(2, dtype=np.float32),
        ),
        optimization_mode=ZonoOptimizationMode.VOL_MAX,
        noise=0.1,
        delta_t=1.0,
        conservative_safeguarding=False,  # legacy code, not used anymore
    )
    safe_region: Callable = safe_control_fn


class SeekerEnvPointMass(SeekerEnvAdvanced):
    def __init__(
        self,
        randomize_env: bool = True,
        render_mode: str = None,
        render_hook: Callable[[np.ndarray, pygame.Surface, np.ndarray, int, int, int, int, int], None] = None,
        num_obstacles: int = 3,
        noise: Interval = None,
        dim: int = 2,
        agent_range: float = 1,
        render_axis=None,
        collision_reward: float = 0,
        goal_reward: float = 100.0,
        done_on_collision: bool = False,
        done_on_goal: bool = True,
        dtype: type = np.float32,
        dt: float = 1.0,
        multi_step_safeguarding: bool = False,
    ):
        super().__init__(
            randomize_env,
            render_mode,
            num_obstacles,
            render_hook,
            noise,
            dim,
            agent_range,
            render_axis,
            collision_reward,
            goal_reward,
            done_on_collision,
            done_on_goal,
            dtype,
            dt,
        )

        obs_shape = 3 * dim + num_obstacles * (dim + 1)
        self.observation_space = gym.spaces.Box(low=-self._size, high=self._size, shape=(obs_shape,), dtype=self._dtype)
        self.state = np.zeros(obs_shape, dtype=self._dtype)

        self._agent_velocity = np.zeros(dim, dtype=self._dtype)
        self._intended = self._agent_velocity.copy()
        self.multi_step_safeguarding = multi_step_safeguarding

    def _get_obs(self) -> np.ndarray:
        obstacles_data = np.column_stack((self._obstacle_position, self._obstacle_radius))
        return np.concatenate(
            [self._agent_position, self._agent_velocity, self._goal_position, obstacles_data.flatten()]
        ).astype(self._dtype)

    def _deterministic_initialization(self):
        super()._deterministic_initialization()
        self._agent_velocity = np.zeros(self.dim, dtype=self._dtype)
        self._intended = np.zeros(self.dim, dtype=self._dtype)

    def _random_initialization(self):
        super()._random_initialization()
        self._agent_velocity = np.zeros(self.dim, dtype=self._dtype)
        self._intended = np.zeros(self.dim, dtype=self._dtype)

    def _agent_update(self):
        self._agent_velocity += self._action * self.dt * self.agent_range
        self._agent_position += self._agent_velocity * self.dt
