import warnings
from collections.abc import Generator
from typing import Any, Dict, List, NamedTuple, Optional, Union

import numpy as np
import psutil
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import BaseBuffer
from stable_baselines3.common.vec_env import VecNormalize


class ReplayBufferSamplesSafe(NamedTuple):
    observations: th.Tensor
    safe_actions: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor


class RolloutBufferSamplesSafe(NamedTuple):
    observations: th.Tensor
    states: th.Tensor
    safe_actions: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_values_penalty: th.Tensor
    old_log_prob: th.Tensor
    old_log_prob_original: th.Tensor
    advantages: th.Tensor
    advantages_penalty: th.Tensor
    returns: th.Tensor
    returns_penalty: th.Tensor
    safe_action_set_centers: th.Tensor
    safe_action_set_generators: th.Tensor


class ReplayBufferSamplesLoss(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    unsafe_actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    proj_losses: th.Tensor
    states: th.Tensor
    next_states: th.Tensor
    safe_action_set_centers: th.Tensor
    safe_action_set_generators: th.Tensor


class RolloutBufferSafe(BaseBuffer):
    """
    Rollout buffer used in on-policy algorithms like A2C/PPO.
    It corresponds to ``buffer_size`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.

    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    observations: np.ndarray
    states: np.ndarray
    safe_actions: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    punishments: np.ndarray
    advantages: np.ndarray
    advantages_penalty: np.ndarray
    returns: np.ndarray
    returns_penalty: np.ndarray
    episode_starts: np.ndarray
    log_probs: np.ndarray
    log_probs_original: np.ndarray
    values: np.ndarray
    values_penalty: np.ndarray
    safe_action_set_centers: np.ndarray
    safe_action_set_generators: np.ndarray

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        n_states: int,
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
        dtype: np.dtype = np.float32,  # Add dtype parameter
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self._dtype = dtype  # Store the dtype
        self.generator_ready = False
        self.n_states = n_states
        self.reset()

    def reset(self) -> None:
        # Use self._dtype instead of hardcoded np.float32
        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=self._dtype)
        self.states = np.zeros((self.buffer_size, self.n_envs, self.n_states), dtype=self._dtype)
        self.safe_actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=self._dtype)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=self._dtype)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=self._dtype)
        self.punishments = np.zeros((self.buffer_size, self.n_envs), dtype=self._dtype)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=self._dtype)
        self.returns_penalty = np.zeros((self.buffer_size, self.n_envs), dtype=self._dtype)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=self._dtype)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=self._dtype)
        self.values_penalty = np.zeros((self.buffer_size, self.n_envs), dtype=self._dtype)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=self._dtype)
        self.log_probs_original = np.zeros((self.buffer_size, self.n_envs), dtype=self._dtype)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=self._dtype)
        self.advantages_penalty = np.zeros((self.buffer_size, self.n_envs), dtype=self._dtype)
        # Storage for safe action sets (Zonotope centers and generators)
        self.safe_action_set_centers = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=self._dtype)
        self.safe_action_set_generators = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim, self.action_dim), dtype=self._dtype
        )
        self.generator_ready = False
        super().reset()

    def compute_returns_and_advantage(
        self, last_values: th.Tensor, last_penalty_values: th.Tensor, dones: np.ndarray
    ) -> None:
        """
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain Monte-Carlo advantage estimate (A(s) = R - V(S))
        where R is the sum of discounted reward with value bootstrap
        (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization.

        The TD(lambda) estimator has also two special cases:
        - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
        - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

        For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.

        :param last_values: state value estimation for the last step (one for each env)
        :param last_penalty_values: penalty value estimation for the last step (one for each env)
        :param dones: if the last step was a terminal step (one bool for each env).
        """
        # Convert to numpy with proper dtype
        last_values = last_values.clone().cpu().numpy().astype(self._dtype).flatten()  # type: ignore[assignment]
        last_penalty_values = last_penalty_values.clone().cpu().numpy().astype(self._dtype).flatten()

        last_gae_lam = 0
        last_gae_lam_penalty = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones.astype(self._dtype)
                next_values = last_values
                next_penalty_values = last_penalty_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
                next_penalty_values = self.values_penalty[step + 1]

            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            delta_penalty = (
                self.punishments[step]
                + self.gamma * next_penalty_values * next_non_terminal
                - self.values_penalty[step]
            )
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            last_gae_lam_penalty = (
                delta_penalty + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam_penalty
            )
            self.advantages[step] = last_gae_lam
            self.advantages_penalty[step] = last_gae_lam_penalty
        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        self.returns = self.advantages + self.values
        self.returns_penalty = self.advantages_penalty + self.values_penalty

    def add(
        self,
        obs: np.ndarray,
        state: np.ndarray,
        safe_action: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        punishment: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        value_penalty: th.Tensor,
        log_prob: th.Tensor,
        log_prob_original: th.Tensor,
        safe_action_set_center: Optional[np.ndarray] = None,
        safe_action_set_generator: Optional[np.ndarray] = None,
    ) -> None:
        """
        :param obs: Observation
        :param state: State
        :param safe_action: safeguarded action
        :param action: Action
        :param reward: Reward
        :param punishment: Punishment
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param value_penalty: estimated penalty value of the current state
        :param log_prob: log probability of the action
            following the current policy.
        :param log_prob_original: original log probability of the action
            before any modifications or projections.
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)
        if len(log_prob_original.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob_original = log_prob_original.reshape(-1, 1)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))
        safe_action = safe_action.reshape((self.n_envs, self.action_dim))

        # Store with explicit dtype conversion
        self.observations[self.pos] = np.array(obs, dtype=self._dtype)
        self.states[self.pos] = np.array(state, dtype=self._dtype)
        self.safe_actions[self.pos] = np.array(safe_action, dtype=self._dtype)
        self.actions[self.pos] = np.array(action, dtype=self._dtype)
        self.rewards[self.pos] = np.array(reward, dtype=self._dtype)
        self.punishments[self.pos] = np.array(punishment, dtype=self._dtype)
        self.episode_starts[self.pos] = np.array(episode_start, dtype=self._dtype)

        # Convert tensor values to numpy with proper dtype
        self.values[self.pos] = value.clone().cpu().numpy().astype(self._dtype).flatten()
        self.values_penalty[self.pos] = value_penalty.clone().cpu().numpy().astype(self._dtype).flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy().astype(self._dtype)
        self.log_probs_original[self.pos] = log_prob_original.clone().cpu().numpy().astype(self._dtype)

        # Store safe action set if provided
        if safe_action_set_center is not None:
            self.safe_action_set_centers[self.pos] = safe_action_set_center
        if safe_action_set_generator is not None:
            self.safe_action_set_generators[self.pos] = safe_action_set_generator

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(self, batch_size: Optional[int] = None) -> Generator[RolloutBufferSamplesSafe, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            _tensor_names = [
                "observations",
                "states",
                "safe_actions",
                "actions",
                "values",
                "values_penalty",
                "log_probs",
                "log_probs_original",
                "advantages",
                "advantages_penalty",
                "returns",
                "returns_penalty",
                "safe_action_set_centers",
                "safe_action_set_generators",
            ]

            # Swap and flatten standard tensors
            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> RolloutBufferSamplesSafe:
        data = (
            self.observations[batch_inds],
            self.states[batch_inds],
            self.safe_actions[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.values_penalty[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.log_probs_original[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.advantages_penalty[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            self.returns_penalty[batch_inds].flatten(),
            self.safe_action_set_centers[batch_inds],
            self.safe_action_set_generators[batch_inds],
        )

        # Convert all data to torch tensors with proper dtype
        torch_data = []
        for array in data:
            tensor = self.to_torch(array)
            # Convert to the current PyTorch default dtype
            if tensor.dtype != th.get_default_dtype():
                tensor = tensor.to(dtype=th.get_default_dtype())
            torch_data.append(tensor)

        return RolloutBufferSamplesSafe(*torch_data)


class ReplayBufferSafe(BaseBuffer):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
        Cannot be used in combination with handle_timeout_termination.
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    observations: np.ndarray
    next_observations: np.ndarray
    safe_actions: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    timeouts: np.ndarray
    safe_action_set_centers: np.ndarray
    safe_action_set_generators: np.ndarray
    _dtype: np.float32

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        # Adjust buffer size
        self.buffer_size = max(buffer_size // n_envs, 1)

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        # there is a bug if both optimize_memory_usage and handle_timeout_termination are true
        # see https://github.com/DLR-RM/stable-baselines3/issues/934
        if optimize_memory_usage and handle_timeout_termination:
            raise ValueError(
                "ReplayBuffer does not support optimize_memory_usage = True "
                "and handle_timeout_termination = True simultaneously."
            )
        self.optimize_memory_usage = optimize_memory_usage

        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=observation_space.dtype)

        if not optimize_memory_usage:
            # When optimizing memory, `observations` contains also the next observation
            self.next_observations = np.zeros(
                (self.buffer_size, self.n_envs, *self.obs_shape), dtype=observation_space.dtype
            )

        self.actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=self._maybe_cast_dtype(action_space.dtype)
        )

        self.safe_actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=self._maybe_cast_dtype(action_space.dtype)
        )

        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        if psutil is not None:
            total_memory_usage: float = (
                self.observations.nbytes
                + self.safe_actions.nbytes
                + self.actions.nbytes
                + self.rewards.nbytes
                + self.dones.nbytes
            )

            if not optimize_memory_usage:
                total_memory_usage += self.next_observations.nbytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        safe_action: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs, dtype=self._dtype)

        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs, dtype=self._dtype)
        else:
            self.next_observations[self.pos] = np.array(next_obs, dtype=self._dtype)

        self.safe_actions[self.pos] = np.array(safe_action, dtype=self._dtype)
        self.actions[self.pos] = np.array(action, dtype=self._dtype)
        self.rewards[self.pos] = np.array(reward, dtype=self._dtype)
        self.dones[self.pos] = np.array(done, dtype=self._dtype)

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array(
                [info.get("TimeLimit.truncated", False) for info in infos], dtype=self._dtype
            )

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamplesLoss:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        if not self.optimize_memory_usage:
            return super().sample(batch_size=batch_size, env=env)
        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> ReplayBufferSamplesSafe:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.safe_actions[batch_inds, env_indices, :],
            self.actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
        )
        return ReplayBufferSamplesSafe(*tuple(map(self.to_torch, data)))

    @staticmethod
    def _maybe_cast_dtype(dtype: np.typing.DTypeLike) -> np.typing.DTypeLike:
        """
        Cast `np.float32` action datatype to `np.float32`,
        keep the others dtype unchanged.
        See GH#1572 for more information.

        :param dtype: The original action space dtype
        :return: ``np.float32`` if the dtype was float32,
            the original dtype otherwise.
        """
        if dtype == np.float32:
            return np.float32
        return dtype


class ReplayBufferProjLoss(BaseBuffer):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
        Cannot be used in combination with handle_timeout_termination.
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    observations: np.ndarray
    next_observations: np.ndarray
    actions: np.ndarray
    unsafe_actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    timeouts: np.ndarray
    proj_losses: Union[np.ndarray, th.Tensor]
    states: np.ndarray
    next_states: np.ndarray

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        state_dim: int,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        gamma: float = 0.99,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        self._dtype = np.float32
        self.gamma = gamma
        self.pos_within_eps = 0
        self.pos_last_termination = 0
        self.state_dim = state_dim

        # Adjust buffer size
        self.buffer_size = max(buffer_size // n_envs, 1)

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        # there is a bug if both optimize_memory_usage and handle_timeout_termination are true
        # see https://github.com/DLR-RM/stable-baselines3/issues/934
        if optimize_memory_usage and handle_timeout_termination:
            raise ValueError(
                "ReplayBuffer does not support optimize_memory_usage = True "
                "and handle_timeout_termination = True simultaneously."
            )
        self.optimize_memory_usage = optimize_memory_usage

        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=self._dtype)
        self.states = np.zeros((self.buffer_size, self.n_envs, self.state_dim), dtype=self._dtype)

        if not optimize_memory_usage:
            # When optimizing memory, `observations` contains also the next observation
            self.next_observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=self._dtype)
            self.next_states = np.zeros((self.buffer_size, self.n_envs, self.state_dim), dtype=self._dtype)

        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=self._dtype)

        self.unsafe_actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=self._dtype)

        self.proj_losses = np.zeros((self.buffer_size, self.n_envs), dtype=self._dtype)

        # Storage for safe action sets (Zonotope centers and generators)
        self.safe_action_set_centers = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=self._dtype)
        self.safe_action_set_generators = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim, self.action_dim), dtype=self._dtype
        )

        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        if psutil is not None:
            total_memory_usage: float = (
                self.observations.nbytes
                + self.states.nbytes
                + self.actions.nbytes
                + self.unsafe_actions.nbytes
                + self.rewards.nbytes
                + self.dones.nbytes
                + self.proj_losses.nbytes
            )

            if not optimize_memory_usage:
                total_memory_usage += self.next_observations.nbytes + self.next_states.nbytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.pos = 0
        self.pos_within_eps = 0
        self.full = False

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        state: np.ndarray,
        next_state: np.ndarray,
        action: np.ndarray,
        unsafe_action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        proj_loss: Union[np.ndarray, th.Tensor],
        infos: List[Dict[str, Any]],
        safe_action_set_center: Optional[np.ndarray] = None,
        safe_action_set_generator: Optional[np.ndarray] = None,
    ) -> None:
        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape states to handle multi-env case
        state = state.reshape((self.n_envs, self.state_dim))
        next_state = next_state.reshape((self.n_envs, self.state_dim))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))
        unsafe_action = unsafe_action.reshape((self.n_envs, self.action_dim))

        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs, dtype=self._dtype)
        self.states[self.pos] = np.array(state, dtype=self._dtype)

        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs, dtype=self._dtype)
            self.states[(self.pos + 1) % self.buffer_size] = np.array(next_state, dtype=self._dtype)
        else:
            self.next_observations[self.pos] = np.array(next_obs, dtype=self._dtype)
            self.next_states[self.pos] = np.array(next_state, dtype=self._dtype)

        self.proj_losses[self.pos] = proj_loss
        self.actions[self.pos] = np.array(action, dtype=self._dtype)
        self.unsafe_actions[self.pos] = np.array(unsafe_action, dtype=self._dtype)
        self.rewards[self.pos] = np.array(reward, dtype=self._dtype)
        self.dones[self.pos] = np.array(done, dtype=self._dtype)

        # Store safe action set if provided
        if safe_action_set_center is not None:
            self.safe_action_set_centers[self.pos] = safe_action_set_center
        if safe_action_set_generator is not None:
            self.safe_action_set_generators[self.pos] = safe_action_set_generator

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array(
                [info.get("TimeLimit.truncated", False) for info in infos], dtype=self._dtype
            )

        if self.timeouts[self.pos].any() or self.dones[self.pos].any():
            self.pos_within_eps = 0
            self.pos_last_termination = self.pos

        self.pos += 1
        self.pos_within_eps += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0
            self.pos_within_eps = 0

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamplesSafe:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        if not self.optimize_memory_usage:
            return super().sample(batch_size=batch_size, env=env)
        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> ReplayBufferSamplesLoss:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
            next_states = self.states[(batch_inds + 1) % self.buffer_size, env_indices, :]
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)
            next_states = self.next_states[batch_inds, env_indices, :]

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            self.unsafe_actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
            self.proj_losses[batch_inds, env_indices],
            self.states[batch_inds, env_indices, :],
            next_states,
            self.safe_action_set_centers[batch_inds, env_indices, :],
            self.safe_action_set_generators[batch_inds, env_indices, :, :],
        )
        return ReplayBufferSamplesLoss(*tuple(map(self.to_torch, data)))

    @staticmethod
    def _maybe_cast_dtype(dtype: np.typing.DTypeLike) -> np.typing.DTypeLike:
        """
        Cast `np.float32` action datatype to `np.float32`,
        keep the others dtype unchanged.
        See GH#1572 for more information.

        :param dtype: The original action space dtype
        :return: ``np.float32`` if the dtype was float32,
            the original dtype otherwise.
        """
        if dtype == np.float32:
            return np.float32
        return dtype
