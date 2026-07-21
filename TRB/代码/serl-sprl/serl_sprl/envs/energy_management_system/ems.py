from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from serl_sprl.envs.configs import BaseEnvConfig, BaseProjectionConfig
from serl_sprl.sets import Zonotope


def safe_control_fn(env, safe_region):
    return None  # No failsafe controller implemented for this environment


class EMSEnvConfig(BaseEnvConfig):
    randomize_env: bool
    max_rollout_steps: int = 24  # one day with hourly steps
    id: str = "serl_sprl/ems"
    max_charge_power: float = 2.0
    min_heat_power: float = 0.0
    max_heat_power: float = 2.5
    battery_capacity: float = 10.0
    forecast_length: int = 4
    dtype: type = np.float32
    safe_region: Zonotope = Zonotope.from_interval(np.array([[0.0, 18.0, 10.0], [10.0, 24.0, 100.0]]).transpose())
    # Dynamics
    dt: float = 1.0
    noise_bound: float = 0.13  # = max abs( outdoor temperature ) / TAU
    multi_step_safeguarding: bool = True


class EMSProjConfig(BaseProjectionConfig):
    safe_control_fn: Callable = safe_control_fn


class EnergyManagementSystemEnv(gym.Env):
    """
    The environment simulates a household with a battery and a heat pump.

    ## Action Set
    | Num | Action                | Min  | Max |
    |-----|-----------------------|------|-----|
    | 0   | Battery Charge Power  | -{max_charge_power} | {max_charge_power} |
    | 1   | Heat Pump Power       | 0 | {max_heat_power} |

    ## State Set
    | Num | State              | Min  | Max   |
    |-----|--------------------|------|-------|
    | 0   | State of Charge    | 0.0  | 10.0  |
    | 1   | Indoor Temperature | 18.0 | 24.0  |
    | 2   | Return Temperature | 10.0 | 100.0 |

    ## Observation Set
    | Num  | Observation                  | Min  | Max   |
    |------|------------------------------|------|-------|
    | 0    | State of Charge              | 0.0  | 10.0  |
    | 1    | Indoor Temperature           | 18.0 | 24.0  |
    | 2    | Return Temperature           | 10.0 | 100.0 |
    | 3-7  | Load Forecast                | 0.0  | 1.0   |
    | 8-12 | PV Forecast                  | 0.0  | 1.0   |
    | 13-17| Outside Temperature Forecast | -10.0| 40.0  |
    | 18-22| Buying Price Forecast        | 0.0 | 1.0   |
    """

    H_FH: float = 1.1
    H_OUT: float = 0.26
    TAU: float = 240
    C_W_FH: float = 1.1625
    # optional: can use temperature-dependent COP to make more complex, see CommonPower.
    # If you do that, also adjust observation space to include forecasts of COP.
    COP: float = 3.01
    C_0: float = (H_FH + H_OUT) / (H_OUT * TAU)
    C_1: float = H_FH / (H_OUT * TAU)
    C_2: float = H_FH / C_W_FH
    C_4: float = COP / C_W_FH

    START_TIME = 7800  # corresponds to November 21st, 2016, 00:00

    BUYING_DATA = [
        0.2,
        0.2,
        0.2,
        0.2,
        0.2,
        0.2,
        0.4,
        0.4,
        0.4,
        0.4,
        0.4,
        0.2,
        0.2,
        0.2,
        0.2,
        0.2,
        0.2,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.2,
        0.2,
    ]

    def __init__(
        self,
        dt: float = 1.0,
        dtype: type = np.float32,
        max_charge_power: float = 2.0,
        min_heat_power: float = 0.0,
        max_heat_power: float = 5.0,
        battery_capacity: float = 10.0,
        forecast_length: int = 4,
        seed: int = 42,
        safe_region: Optional[Zonotope] = None,
        noise_bound: float = None,
        randomize_env: bool = False,
        multi_step_safeguarding: bool = True,
    ):
        self.dt = dt
        self.current_step = 0
        self.rnd_seed = seed
        self._dtype = dtype
        self.randomize_env = randomize_env
        self.start_state = np.array([5.0, 21.0, 25.0])
        self.forecast_length = forecast_length

        self.max_charge_power = max_charge_power
        self.max_heat_power = max_heat_power
        self.min_heat_power = min_heat_power
        self.battery_capacity = battery_capacity
        self.dt = dt
        self.asset_path = Path(__file__).parent / "assets"
        self.load_data = pd.read_csv(self.asset_path / "load.csv")["p"].to_numpy()
        self.pv_data = pd.read_csv(self.asset_path / "pv_power.csv")["p"].to_numpy()
        self.temp_data = pd.read_csv(self.asset_path / "outdoor_temperature.csv", sep=";")["outside_temp"].to_numpy()
        self.buying_data = np.array(self.BUYING_DATA).repeat(366)  # data from 2016 which is a leap year
        self.selling_price = 0.08
        self.cost_coefficient_hp = 1.0  # weight for comfort cost in reward function

        # discretized dynamics
        self.A_d = np.array(
            [
                [1, 0, 0],
                [0, 1 - self.C_0 * self.dt, self.C_1 * self.dt],
                [0, self.C_2 * self.dt, 1 - self.C_2 * self.dt],
            ]
        )
        self.B_d = np.array([[self.dt, 0], [0, 0], [0, self.C_4 * self.dt]])
        self.x_eq = np.array([0, 0, 0])
        self.u_eq = np.array([0, 0])
        # We consider a two-dimensional disturbance corresponding to the outdoor temperature
        # that only affects the indoor temperature
        self.E_d = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
        self.noise_bound = (
            noise_bound if noise_bound is not None else np.round(np.max(np.abs(self.temp_data)) / self.TAU, 2)
        )  # outdoor temperature / TAU

        n_obs = self.forecast_length + 1
        self.obs_low = np.array(
            [[0.0, 18.0, 10.0] + [0.0] * n_obs + [0.0] * n_obs + [-10.0] * n_obs + [0.0] * n_obs], dtype=self._dtype
        )
        self.obs_high = np.array(
            [[10.0, 24.0, 100.0] + [1.0] * n_obs + [1.0] * n_obs + [40.0] * n_obs + [1.0] * n_obs], dtype=self._dtype
        )
        self.observation_space = spaces.Box(
            low=self.obs_low.flatten(), high=self.obs_high.flatten(), dtype=self._dtype, seed=self.rnd_seed
        )
        self.action_space = spaces.Box(
            low=np.array([-self.max_charge_power, self.min_heat_power]),
            high=np.array([self.max_charge_power, self.max_heat_power]),
            dtype=self._dtype,
        )

        self.state = np.zeros(self.A_d.shape[0], dtype=self._dtype)
        self.target_temp = 21.0
        self._last_action = None

        self.safe_region = safe_region
        self.initial_region_low = np.array([4.0, 20.5, 24.5])
        self.initial_region_high = np.array([6.0, 21.5, 25.5])
        self.multi_step_safeguarding = (
            multi_step_safeguarding  # whether to use multi-step ahead prediction in projection
        )
        self._collision = False

    def reset(
        self, seed: Union[int, None] = None, options: Union[Dict[str, Any], None] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self._last_action = None
        self._collision = False
        self.current_step = 0
        if self.randomize_env:
            state = np.random.uniform(self.initial_region_low, self.initial_region_high)
            self.state = state
        else:
            self.state = self.start_state
        obs_info = self._get_info()
        return self.get_obs(), obs_info

    def _get_info(self) -> dict:
        return {"collision": self._collision, "last_action": self._last_action}

    def get_obs(self) -> np.ndarray:
        time_index = self.START_TIME + self.current_step
        # ToDo: Validate!
        load_forecast = self.load_data[time_index : time_index + self.forecast_length + 1].tolist()
        pv_forecast = self.pv_data[time_index : time_index + self.forecast_length + 1].tolist()
        temp_forecast = self.temp_data[time_index : time_index + self.forecast_length + 1].tolist()
        price_forecast = self.buying_data[time_index : time_index + self.forecast_length + 1].tolist()
        obs = np.array(self.state.tolist() + load_forecast + pv_forecast + temp_forecast + price_forecast)
        return obs

    def dynamics_fn(self, action: np.ndarray) -> np.ndarray:
        noise = self.temp_data[self.START_TIME + self.current_step] / self.TAU
        noise_array = np.array([noise, 0.0])  # second disturbance component is zero
        if noise > self.noise_bound:
            raise ValueError("Noise exceeds noise bound!")
        state = deepcopy(self.state)
        next_state = (
            self.A_d @ (state - self.x_eq) + self.B_d @ (action - self.u_eq) + self.E_d @ noise_array + self.x_eq
        )
        return next_state

    def collision_check_fn(self):
        if self.safe_region is not None:
            collision = not self.safe_region.contains(self.state)
        else:
            collision = np.any(self.state <= self.state_constraints[0]) or np.any(
                self.state >= self.state_constraints[1]
            )
        return collision

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self._last_action = action
        self.state = self.dynamics_fn(action)
        if self.collision_check_fn():
            self._collision = True
        reward = self._get_reward(action)
        done = False
        truncated = False
        self.current_step += 1
        return self.get_obs(), reward, done, truncated, self._get_info()

    def _get_reward(self, action: np.ndarray) -> float:
        time_index = self.START_TIME + self.current_step
        buying_price = self.buying_data[time_index]
        load = self.load_data[time_index]
        pv = self.pv_data[time_index]
        battery_power = action[0]
        heat_power = action[1]
        net_load = load - pv + battery_power + heat_power
        electricity_cost = np.where(
            net_load >= 0, net_load * self.dt * buying_price, net_load * self.dt * self.selling_price
        )
        comfort_cost = (self.state[1] - self.target_temp) ** 2  # quadratic penalty for deviation from target temp
        cost = electricity_cost + self.cost_coefficient_hp * comfort_cost  # weight comfort cost
        return -cost  # negative cost as reward
