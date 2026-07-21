import math
from typing import Any, Callable, Dict, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
from scipy.linalg import expm
from stable_baselines3.common.vec_env import DummyVecEnv

from serl_sprl.envs.configs import BaseEnvConfig, BaseProjectionConfig
from serl_sprl.envs.safe_region import ControlInvariantSetZonotope
from serl_sprl.sets import Zonotope


def safe_control_fn(env, safe_region):
    # Sample a safe action from the failsafe controller
    state = env.envs[0].get_wrapper_attr("state") if isinstance(env, DummyVecEnv) else env.get_wrapper_attr("state")
    u_uncoupled = safe_region.sample_ctrl(state)
    return u_uncoupled


class Quad2dCoupledEnvConfig(BaseEnvConfig):
    id: str = "serl_sprl/quad2d_coupled"
    max_rollout_steps: int = 200
    safe_region: ControlInvariantSetZonotope = ControlInvariantSetZonotope(
        S_ctrl_csv="S_ctrl_LongQuadrotor.csv",
        S_RCI_csv="S_RCI_LongQuadrotor.csv",
        x_goal=[0, 1, 0, 0, 0, 0],
        x_lim_low=[
            -1.7,
            0.3,
            -0.8,
            -1,
            -0.261799387799149407829446545292739756405353546142578125,
            -1.5707963267948965579989817342720925807952880859375,
        ],
        x_lim_high=[
            1.7,
            2.0,
            0.8,
            1.0,
            0.261799387799149407829446545292739756405353546142578125,
            1.5707963267948965579989817342720925807952880859375,
        ],
    )
    randomize_env: bool
    dt: float = 0.05
    goal_distance: float = 0.01
    done_on_collision: bool = False
    done_on_goal_reached: bool = False
    collision_reward: float = -1.0
    goal_reward: float = 1.0
    step_reward: float = -1.0
    rew_state_weight: float = 1.0
    rew_act_weight: float = 0.01
    reward_shaping: bool = True
    noise_bound: Optional[float] = 0.08
    w: Optional[np.ndarray] = np.array([0.08, 0.08])
    # Dynamics
    dt: float = 0.05
    K: float = 0.89 / 1.4
    d0: float = 70
    d1: float = 17
    n0: float = 55
    gravity: float = 9.81


class Quad2dProjConfig(BaseProjectionConfig):
    safe_control_fn: Callable = safe_control_fn


class Quad2dCoupledEnv(gym.Env):
    """A longitudinal quadrotor environment representing [1].

    [1] I. M. Mitchell et al. "Invariant, Viability and Discriminating
        Kernel Under-Approximation via Zonotope Scaling", 2019,
        Proceedings of the 22nd ACM International Conference on Hybrid
        Systems: Computation and Control, pp. 268-269

    Goal:
        Hovering at an equilibrium point.

    Characteristics:
        - linear approximation of dynamics at hover point
        - bounded disturbances
        - possibly random initial state
        - variable control frequency

    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        dt: float = 0.05,
        K: float = 0.89 / 1.4,
        d0: float = 70,
        d1: float = 17,
        n0: float = 55,
        gravity: float = 9.81,
        w: np.ndarray = np.array([0.0, 0.0]),
        input_limits: np.ndarray = np.array(
            [[6.834830643179076, 6.834830643179076], [8.596630030978226, 8.596630030978226]]
        ),
        goal_distance: float = 0.01,
        done_on_collision: bool = False,
        done_on_goal_reached: bool = False,
        state_constraints: Optional[np.ndarray] = np.array(
            [[-1.7, 0.3, -0.8, -1, -np.pi / 12, -np.pi / 2], [1.7, 2.0, 0.8, 1.0, np.pi / 12, np.pi / 2]]
        ),
        randomize_env: bool = True,
        start_state: np.ndarray = np.array([0, 1, 0, 0, 0, 0], dtype=np.float32),
        collision_reward: float = -1.0,
        goal_reward: float = 1.0,
        step_reward: float = -1.0,
        rew_state_weight: float = 1.0,
        rew_act_weight: float = 0.01,
        reward_shaping: bool = True,
        size: float = 3,
        render_mode: Optional[str] = None,
        seed: int = 42,
        safe_region: Optional[Union[Zonotope, np.ndarray]] = None,
        dtype: type = np.float32,
        multi_step_safeguarding: bool = False,
    ):
        """Create the reach environment.

        State space:
        x = [x, z, dx, dz, theta, dtheta]

        Continuous dynamics:
        f[0] = x[2]
        f[1] = x[3]
        f[2] = g * x[4] + w[0]
        f[3] = -g + (u[0] + u[1]) * K + w[1]
        f[4] = x[5]
        f[5] = -d0 * x[4] - d1 * x[5] + n0 * (-u[0] + u[1])

        Discrete dynamics:
        x_{k+1} = x_{k} + f(x_{k}) * dt

        Args:
            dt (float): Time step of the discrete dynamics.
            K (float): Thrust coefficient.
            d0 (float): Angular gain coefficient (theta).
            d1 (float): Angular gain coefficient (dtheta).
            n0 (float): Torque input coefficient.
            w (ndarray): disturbance bound in x z - shape (2,).
            input_limits (np.ndarray): Input limits in the form [[min_u1, min_u2], [max_u1, max_u2]].
            goal_distance (float): Distance to goal to consider it reached.
            done_on_collision (bool): If true, the episode is finished if the quadrotor collides with the constraint
                set.
            done_on_goal_reached (bool): If true, the episode is finished if the goal is reached.
            state_constraints (Optional, np.ndarray): State constraints of the form
                [[min_x, min_z,...], [max_x, max_z,...]].
            randomize_env (bool): If true, the environment is randomized on reset.
            start_state (np.ndarray): Start state of the quadrotor.
            collision_reward (float): Reward for exceeding the state constraints.
            goal_reward (float): Reward for being inside the goal region.
            step_reward (float): Reward for each step.
            rew_state_weight (float): Weight of the state reward.
            rew_act_weight (float): Weight of the action reward.
            reward_shaping (bool): If true, the reward is shaped by the distance to the goal.
            size (float): Size of the environment. Defualts to 3.
            render_mode (Optional, str): Optional render mode. Can be "human" or "rgb_array".
            seed (int): Random seed.
            safe_region (Optional, Union[Zonotope, np.ndarray]): Safe region (RCI set) of the env.
                Needed to sample safe initial states from.
            dt
        """
        self._dtype = dtype
        self.dt = dt
        self.K = K
        self.d0 = d0
        self.d1 = d1
        self.n0 = n0
        self.w = w
        self.g = gravity
        self.n_dim = 6
        self.action_dim = 2
        self.A = np.array(
            [
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, self.g, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, -d0, -d1],
            ]
        )
        self.B = np.array([[0, 0], [0, 0], [0, 0], [K, K], [0, 0], [-n0, n0]])
        self.E = np.array([[0, 0], [0, 0], [1, 0], [0, 1], [0, 0], [0, 0]])
        self.x_eq = np.array([0, 1, 0, 0, 0, 0])
        self.u_eq = np.array([self.g / (2 * K), self.g / (2 * K)])
        self.w_eq = np.array([0, 0])
        self.A_d = expm(self.A * dt)
        temp = np.eye(self.A.shape[0]) * dt
        interval = temp
        i = 1
        while not np.all(np.abs(temp) < 2.2204e-16):
            temp = (np.linalg.matrix_power(self.A, i) / math.factorial(i + 1)) * (
                dt ** (i + 1)
            )  # temp * dt/(i+1) * self.A #
            interval = interval + temp
            i += 1
        self.B_d = np.matmul(interval, self.B)
        self.E_d = np.matmul(interval, self.E)
        # For reward shaping only:
        self.u_range = input_limits
        self.goal_distance = goal_distance
        self.done_on_collision = done_on_collision
        self.done_on_goal_reached = done_on_goal_reached
        self.state_constraints = np.array(state_constraints, dtype=self._dtype)
        self.randomize_env = randomize_env
        self.start_state = np.array(start_state, dtype=self._dtype)
        self.collision_reward = collision_reward
        self.goal_reward = goal_reward
        self.step_reward = step_reward
        self.rew_state_weight = rew_state_weight
        self.rew_act_weight = rew_act_weight
        self.reward_shaping = reward_shaping
        self.render_mode = render_mode
        self.rnd_seed = seed
        self.np_random = np.random.RandomState(self.rnd_seed)
        if self.render_mode is not None:
            self.window_size = 512  # The size of the PyGame window
        self.size = size
        self.observation_space = spaces.Box(
            low=-self.size, high=self.size, shape=(self.n_dim,), dtype=self._dtype, seed=self.rnd_seed
        )
        self.action_space = spaces.Box(
            low=input_limits[0], high=input_limits[1], shape=(self.action_dim,), dtype=self._dtype, seed=self.rnd_seed
        )
        self.state = np.zeros((self.n_dim,), dtype=self._dtype)
        self.goal = start_state
        # For reward shaping
        self.max_distance = np.linalg.norm(state_constraints[1] - state_constraints[0])
        self._collision = False
        self.safe_region = safe_region
        self.multi_step_safeguarding = multi_step_safeguarding
        if randomize_env and safe_region is None:
            raise ValueError("If randomize_env is True, safe_region must be provided to sample from.")
        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is
        used to ensure that the environment is rendered at the correct
        framerate in human-mode. They will remain `None` until human-mode
        is used for the first time.
        """
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self._goal_reached_counter = 0
        self._collision_counter = 0
        # for saving last actions:
        self._last_action = None
        self._input_change = None

    def get_obs(self) -> np.ndarray:
        """Get the observation, which is the agent location."""
        return self.state

    def _get_info(self) -> Dict:
        """Get the info dictionary."""
        return {
            "distance": np.linalg.norm(self.state - self.goal),
            "collision": self._collision,
            "n_goal_reached": self._goal_reached_counter,
            "n_collision": self._collision_counter,
            "input_change": self._input_change,
        }

    def _get_goal_reached(self) -> bool:
        """Get the information if the goal was reached.

        Returns:
            True if goal was reached, false otherwise.
        """
        return np.linalg.norm(self.goal - self.state) < self.goal_distance

    def _get_done(self) -> bool:
        """Return if the episode is finished.

        Returns:
            True if done flag should be set, false otherwise.
        """
        if self.done_on_collision and self._collision:
            return True
        if self.done_on_goal_reached and self._get_goal_reached():
            return True
        return False

    def _get_reward(self, act) -> float:
        """Compute the step reward."""
        reward = self.step_reward
        distance = np.linalg.norm(self.goal - self.state)
        goal_reached = distance < self.goal_distance
        if self._collision:
            reward += self.collision_reward
        if self.reward_shaping:
            dist = np.sum(self.rew_state_weight * distance)
            act_cost = np.mean((act - self.u_range[0]) / (self.u_range[1] - self.u_range[0]))
            dist += np.sum(self.rew_act_weight * act_cost)
            reward += np.exp(-dist)
        else:
            reward += goal_reached * self.goal_reward
        return reward

    def reset(
        self, seed: Union[int, None] = None, options: Union[Dict[str, Any], None] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment.
        """
        super().reset(seed=seed)
        self._collision = False
        # Choose the agent's and goal location.
        if self.randomize_env:
            # Sample safe initial state from RCI set.
            self.state = self.safe_region.sample()
        else:
            self.state = self.start_state

        observation = self.get_obs()

        if self.render_mode == "human":
            self._render_frame()
        self._goal_reached_counter = 0
        self._collision_counter = 0
        info = self._get_info()
        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Run one timestep of the environment's dynamics.

        When end of episode is reached, you are responsible for
        calling `reset()` to reset this environment's state.

        Accepts an action and returns a tuple
            (observation, reward, done, info).

        Args:
            action (np.ndarray): an action provided by the agent

        Returns:
            observation (np.ndarray): observation: agent state.
            reward (float): amount of reward returned after previous action.
            done (bool): whether the episode has terminated (failed or goal reached).
            truncated (bool): whether the maximum number of steps (time limit) was reached.
            info (dict): contains auxiliary diagnostic information.
        """
        assert not np.any(np.isnan(action)), "NaN in action"

        self._collision = False
        self._input_change = (action - self._last_action) / self.dt if self._last_action is not None else None
        self._last_action = action.copy()
        # TODO: Switch to bounded Gaussian noise?
        w = self.np_random.random(2)
        w *= self.w
        # do the collision check with maximum disturbance self.w
        if self.collision_check_fn(action, self.w):
            self._collision = True
            self._collision_counter += 1
        # Discrete Dynamics
        self.state = self.dynamics_fn(action, w)
        observation = self.get_obs()
        goal_reached = self._get_goal_reached()
        self._goal_reached_counter += int(goal_reached)
        info = self._get_info()
        reward = self._get_reward(action)
        done = self._get_done()
        truncated = done

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, done, truncated, info

    def dynamics_fn(self, u: np.ndarray, w: np.ndarray = None) -> np.ndarray:
        r"""Calculate the next state of the environment based on the given action and current state.

        State space:
        x = [x, z, dx, dz, theta, dtheta]

        Discrete dynamics:
        x_{t+1} = A_d (x_{t} - x*) + B_d (u_{t} - u*) + E_d (w_{t} -w*)

        where
            A_d = e^{A dt}
            B_d = e^{A dt} \int_0^dt e^{-A d\tau} d\tau B
            E_d = e^{A dt} \int_0^dt e^{-A d\tau} d\tau E

        Based on continuous dynamics:
        \dot{x}[0] = x[2]
        \dot{x}[1] = x[3]
        \dot{x}[2] = g * x[4] + w[0]
        \dot{x}[3] = -g + (u[0] + u[1]) * K + w[1]
        \dot{x}[4] = x[5]
        \dot{x}[5] = -d0 * x[4] - d1 * x[5] + n0 * (-u[0] + u[1])

        With linear taylor expansion around the equilibrium point *
        \dot{x} = A (x - x*) + B (u - u*) + E (w - w*)

        Args:
            u (np.ndarray): the action to execute.
            state (np.ndarray): the current state of the environment.
            w (np.ndarray): the noise to add to the state.
        """
        state = self.state
        w = self.w if w is None else w
        next_state = (
            self.x_eq + self.A_d @ (state - self.x_eq) + self.B_d @ (u - self.u_eq) + self.E_d @ (w - self.w_eq)
        )

        return next_state

    def collision_check_fn(
        self,
        action: np.ndarray,
        w: np.ndarray = np.array([0.0, 0.0]),
    ) -> bool:
        """Return true if the given action would collide.

        Args:
            u (np.ndarray): the action to execute.
            w (np.ndarray): the noise to add to the state.
        Returns:
            bool: True if the action would collide, False otherwise.
        """
        next_state = self.dynamics_fn(action, w)
        if self.safe_region is not None:
            collision = not self.safe_region.contains(next_state)
        else:
            collision = np.any(next_state <= self.state_constraints[0]) or np.any(
                next_state >= self.state_constraints[1]
            )
        return collision

    def render(self):
        """Render a frame."""
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        """Render a frame of the current state using rgb_array method."""
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        agent_width = 30
        agent_height = 2
        target = 1 / (2 * self.size) * np.array([self.goal[0] + self.size, self.goal[1] + self.size])
        agent = 1 / (2 * self.size) * np.array([self.state[0] + self.size, self.state[1] + self.size])
        # First we draw the target
        points = []
        radius = math.sqrt((agent_height / 2) ** 2 + (agent_width / 2) ** 2)
        angle = math.atan2(agent_height / 2, agent_width / 2)
        angles = [angle, -angle + math.pi, angle + math.pi, -angle]
        rot_radians = self.goal[4]
        target_pos = self.window_size * np.array(target) - np.array([agent_width / 2, agent_height / 2])
        for angle in angles:
            y_offset = -1 * radius * math.sin(angle + rot_radians)
            x_offset = radius * math.cos(angle + rot_radians)
            points.append((target_pos[0] + x_offset, target_pos[1] + y_offset))
        pygame.draw.polygon(canvas, (120, 120, 120), points)
        # Now we draw the agent
        points = []
        radius = math.sqrt((agent_height / 2) ** 2 + (agent_width / 2) ** 2)
        angle = math.atan2(agent_height / 2, agent_width / 2)
        angles = [angle, -angle + math.pi, angle + math.pi, -angle]
        rot_radians = self.state[4]
        agent_pos = self.window_size * np.array(agent) - np.array([agent_width / 2, agent_height / 2])
        for angle in angles:
            y_offset = -1 * radius * math.sin(angle + rot_radians)
            x_offset = radius * math.cos(angle + rot_radians)
            points.append((agent_pos[0] + x_offset, agent_pos[1] + y_offset))
        pygame.draw.polygon(canvas, (255, 0, 0) if self._collision else (0, 255, 0), points)

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the
            # visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the
            # predefined framerate.
            # The following line will automatically add a delay to keep
            # the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        """Close the render window."""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
