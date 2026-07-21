import os
from abc import abstractmethod

import numpy as np
import scipy as sc

from serl_sprl.sets import Zonotope


class SafeRegion:
    """
    Base class for safe regions.
    """

    def __init__(self, x_goal: np.ndarray, x_lim_low: np.ndarray, x_lim_high: np.ndarray, seed=None):
        """
        Args:
        x_goal (np.ndarray): Goal state.
        x_lim_low (np.ndarray): Lower bound of the state space.
        x_lim_high (np.ndarray): Upper bound of the state space.
        seed (int, optional): Random seed. Defaults to None.
        """
        self.x_goal = x_goal
        self.x_lim_low = x_lim_low
        self.x_lim_high = x_lim_high
        self._seed = seed
        self._rng = np.random.default_rng(self._seed)

    @property
    def rng(self):
        return self._rng

    @rng.setter
    def rng(self, seed):
        self._seed = seed
        self._rng = np.random.default_rng(self._seed)

    @classmethod
    def compute_safe_region(cls):
        """
        :return: safe region
        """
        raise NotImplementedError

    def sample(self):
        """Sample a point from the control invariant set."""
        sample = None
        while sample is None:
            sample = self._rng.uniform(self.x_lim_low, self.x_lim_high)
            try:
                if sample not in self:
                    sample = None
            # We need this because the __contains__ method of the zonotope class
            # uses a solver that sometimes fails randomly.
            except ValueError:
                print("Contains solver failed, retrying...")
                sample = None
        return sample

    def sample_ctrl(self, x):
        """Sample a control input from the control invariant set.

        Args:
            x (np.ndarray): State of the system.
        Returns:
            np.ndarray: Control input.
        """
        u_ctrl = None
        b_eq_ctrl = x - self.x_goal
        sol = sc.optimize.linprog(
            self.f_ctrl, A_ub=self.A_ctrl, b_ub=self.b_ctrl, A_eq=self.A_eq_ctrl, b_eq=b_eq_ctrl, bounds=self.x_bounds
        )
        if sol.status == 0 and sol.fun <= 1 + 1e-6:
            x_para = sol.x[1:]
            u_ctrl = self.G_center + self.G_ctrl @ x_para
        elif sol.status == 0 or sol.status == 4:
            x_para = sol.x[1:]
            u_ctrl = self.G_center + self.G_ctrl @ x_para
            print("Solver status is {}".format(sol.status))
            print("Check state {} and control {}".format(x, u_ctrl))
        else:
            raise ValueError("Error in fail-safe planner, no solution found for state {}".format(x))
        return u_ctrl

    @abstractmethod
    def __contains__(self, state):
        """
        Args:
            state: state
        Returns: True iff state is inside safe region
        """
        raise NotImplementedError


class ControlInvariantSetZonotope(SafeRegion):
    """Control invariant set based on Zonotopes."""

    def __init__(
        self,
        S_ctrl_csv: str = None,
        S_RCI_csv: str = None,
        x_goal: np.ndarray = None,
        x_lim_low: np.ndarray = None,
        x_lim_high: np.ndarray = None,
        seed=None,
    ):
        """
        Initialize the control invariant set.
        Args:
            S_ctrl_csv (str): Path to the CSV file containing the S_ctrl.
            S_RCI_csv (str): Path to the CSV file containing the S_RCI.
            x_goal (np.ndarray): Goal state.
            x_lim_low (np.ndarray): Lower bound of the state space.
            x_lim_high (np.ndarray): Upper bound of the state space.
            seed (int, optional): Random seed. Defaults to None.
        """
        root = os.path.dirname(os.path.abspath(__file__)) + "/matlab/"

        # Init base class
        super(ControlInvariantSetZonotope, self).__init__(
            x_goal=x_goal, x_lim_low=x_lim_low, x_lim_high=x_lim_high, seed=seed
        )

        zonotope_S_ctrl = np.genfromtxt(root + S_ctrl_csv, delimiter=",")
        # 1 dimensional input:
        if len(zonotope_S_ctrl.shape) == 1:
            zonotope_S_ctrl = zonotope_S_ctrl.reshape((1, -1))
        self.G_ctrl = zonotope_S_ctrl[:, 1:]
        self.G_center = zonotope_S_ctrl[:, 0]
        zonotope_S_RCI = np.genfromtxt(root + S_RCI_csv, delimiter=",")
        self.G_S = zonotope_S_RCI[:, 1:]
        self.c_S = zonotope_S_RCI[:, 0]
        self.RCI_zonotope = Zonotope(G=self.G_S, c=self.c_S.reshape((len(zonotope_S_RCI[:, 0]), 1)))
        self.A_ctrl = np.c_[
            -np.ones(2 * self.G_S.shape[1]), np.vstack((np.eye(self.G_S.shape[1]), -np.eye(self.G_S.shape[1])))
        ]
        self.b_ctrl = np.zeros(2 * self.G_S.shape[1])
        self.A_eq_ctrl = np.c_[np.zeros(self.G_S.shape[0]), self.G_S]
        self.f_ctrl = np.zeros(self.G_S.shape[1] + 1)
        self.f_ctrl[0] = 1
        self.x_bounds = (-1, 1)

    def __contains__(self, state):
        """
        Check if state lies within safe region.
        Args:
            state (np.ndarray): State of the system.
        Returns:
            True iff state is inside safe region.
        """
        return self.RCI_zonotope.contains_point(np.array(state.reshape(-1, 1)))

    def contains(self, state):
        return state in self
