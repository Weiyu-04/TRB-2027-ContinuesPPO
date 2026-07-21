import collections
import warnings
from copy import deepcopy
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch as th
from continuoussets.convexsets import Zonotope
from gymnasium import spaces
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor, NatureCNN
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from torch import nn

from serl_sprl.projection.base import BaseProjectionSafeguard
from serl_sprl.sb3_contrib.common import MlpExtractor
from serl_sprl.sb3_contrib.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)
from serl_sprl.sb3_contrib.helpers import compute_gaussian_boundary_mass_torch


class ActorCriticPolicy(BasePolicy):
    """
    Policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param share_features_extractor: If True, the features extractor is shared between the policy and value networks.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param project_mean: whether or not to project the mean of the policy
    :param use_squared_loss: whether to use the squared distance loss (instead of the non-squared)
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        squash_mean: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        use_per_sample_loss: bool = False,
        use_penalty_critic: bool = False,
        safeguard: Optional[BaseProjectionSafeguard] = None,
        penalty_factor: float = 0.0,
    ):
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == th.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5

        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=squash_output,
            normalize_images=normalize_images,
        )

        if isinstance(net_arch, list) and len(net_arch) > 0 and isinstance(net_arch[0], dict):
            warnings.warn(
                (
                    "As shared layers in the mlp_extractor are removed since SB3 v1.8.0, "
                    "you should now pass directly a dictionary and not a list "
                    "(net_arch=dict(pi=..., vf=...) instead of net_arch=[dict(pi=..., vf=...)])"
                ),
            )
            net_arch = net_arch[0]

        # safeguarding
        self._penalty_factor = penalty_factor
        self.last_projected = False
        self.last_punishments = []
        self.last_unsafe_actions = None
        self.use_per_sample_loss = use_per_sample_loss
        self.use_penalty_critic = use_penalty_critic
        if use_per_sample_loss and safeguard is None:
            raise ValueError("If `use_per_sample_loss` is True, a `safeguard` must be provided.")
        self.safeguard = safeguard
        # Default network architecture, from stable-baselines
        if net_arch is None:
            if features_extractor_class == NatureCNN:
                net_arch = []
            else:
                net_arch = dict(pi=[64, 64], vf=[64, 64])

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.ortho_init = ortho_init

        self.share_features_extractor = share_features_extractor
        self.features_extractor = self.make_features_extractor()
        self.features_dim = self.features_extractor.features_dim
        if self.share_features_extractor:
            self.pi_features_extractor = self.features_extractor
            self.vf_features_extractor = self.features_extractor
            self.penalty_vf_features_extractor = self.features_extractor
        else:
            self.pi_features_extractor = self.features_extractor
            self.vf_features_extractor = self.make_features_extractor()
            self.penalty_vf_features_extractor = self.make_features_extractor()

        self.log_std_init = log_std_init
        dist_kwargs = None

        assert not (
            squash_output and not use_sde
        ), "squash_output=True is only available when using gSDE (use_sde=True)"
        # Keyword arguments for gSDE distribution
        if use_sde:
            dist_kwargs = {
                "full_std": full_std,
                "squash_output": squash_output,
                "squash_mean": squash_mean,
                "use_expln": use_expln,
                "learn_features": False,
            }

        self.use_sde = use_sde
        self.dist_kwargs = dist_kwargs

        # Action distribution
        self.action_dist = make_proba_distribution(action_space, use_sde=use_sde, dist_kwargs=dist_kwargs)

        self._build(lr_schedule)

    def deactivate_projection(self) -> None:
        self.project_mean = False

    def activate_projection(self) -> None:
        self.project_mean = True

    def set_env(self, env) -> None:
        self.env = env

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        default_none_kwargs = self.dist_kwargs or collections.defaultdict(lambda: None)

        data.update(
            dict(
                net_arch=self.net_arch,
                activation_fn=self.activation_fn,
                use_sde=self.use_sde,
                log_std_init=self.log_std_init,
                squash_output=default_none_kwargs["squash_output"],
                squash_mean=default_none_kwargs.get("squash_mean", False),
                full_std=default_none_kwargs["full_std"],
                use_expln=default_none_kwargs["use_expln"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                ortho_init=self.ortho_init,
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )
        return data

    def reset_noise(self, n_envs: int = 1) -> None:
        """
        Sample new weights for the exploration matrix.

        :param n_envs:
        """
        assert isinstance(
            self.action_dist, StateDependentNoiseDistribution
        ), "reset_noise() is only available when using gSDE"
        self.action_dist.sample_weights(self.log_std, batch_size=n_envs)

    def _build_mlp_extractor(self) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        # Note: If net_arch is None and some features extractor is used,
        #       net_arch here is an empty list and mlp_extractor does not
        #       really contain any layers (acts like an identity module).
        self.mlp_extractor = MlpExtractor(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self._build_mlp_extractor()

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        if isinstance(self.action_dist, DiagGaussianDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, latent_sde_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(
            self.action_dist, (CategoricalDistribution, MultiCategoricalDistribution, BernoulliDistribution)
        ):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        # Always create penalty_value_net to maintain consistent RNG state
        # Even if not used, this ensures reproducible initialization
        self.penalty_value_net = nn.Linear(self.mlp_extractor.latent_dim_penalty, 1)
        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
                self.penalty_value_net: 1,
            }
            if not self.share_features_extractor:
                # Note(antonin): this is to keep SB3 results
                # consistent, see GH#1148
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)
                module_gains[self.penalty_vf_features_extractor] = np.sqrt(2)

            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def forward(
        self, obs: th.Tensor, state: th.Tensor, deterministic: bool = False
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        policy_projected = [False for _ in range(len(obs))]
        punishments = [0.0 for _ in range(len(obs))]
        if self.share_features_extractor:
            latent_pi, latent_vf, latent_vf_penalty = self.mlp_extractor(features)
        else:
            pi_features, vf_features, penalty_vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
            latent_vf_penalty = self.mlp_extractor.forward_penalty_critic(penalty_vf_features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        penalty_values = self.penalty_value_net(latent_vf_penalty)
        distribution, _ = self._get_action_dist_from_latent(latent_pi, state)

        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        orig_log_prob = log_prob.clone()
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        # project sampled actions
        actions_array = actions.detach().cpu().numpy()
        self.last_unsafe_actions = deepcopy(actions_array)
        for i in range(len(obs)):
            safe_actions, safe_action_set = self.safeguard.project_policy_action(state[i].to(actions.dtype), actions[i])
            if len(safe_actions.shape) == 2:
                safe_actions = safe_actions.squeeze(0)
            safe_actions = safe_actions.detach().cpu().numpy()
            action_distance = np.linalg.norm(safe_actions - actions_array[i], 2)
            punishment = -(action_distance**2) * self._penalty_factor
            if self.use_penalty_critic:
                punishments[i] = punishment
            if action_distance > 1e-4:
                normal = (actions_array[i] - safe_actions) / action_distance
                policy_projected[i] = True
                probability_mass = compute_gaussian_boundary_mass_torch(
                    mu=distribution.distribution.mean,
                    Sigma=distribution.distribution.stddev**2,
                    boundary_points=th.Tensor(safe_actions.reshape(1, -1)),
                    normals=th.Tensor(normal),
                    squashed=self.squash_output,
                )
                # This seems to be numerically unstable - to be investigated
                # intersection_points = compute_intersection_points_2d(
                #     th.Tensor(safe_actions.reshape(1,safe_actions.shape[0])),
                #     th.Tensor(normal.reshape(1,normal.shape[0])),
                #     limit=5.0
                #     )
                # probability_mass = independent_gaussian_ray_integral(
                #     distribution.distribution.mean,
                #     distribution.distribution.stddev**2,
                #     th.Tensor(safe_actions.reshape(1,safe_actions.shape[0])),
                #     intersection_points
                #     )

                log_prob[i] = th.log(probability_mass)
            actions[i, :] = th.Tensor(safe_actions)
        self.last_punishments = punishments
        self.last_projected = policy_projected
        return actions, values, penalty_values, log_prob, orig_log_prob, safe_action_set

    def extract_features(  # type: ignore[override]
        self, obs: PyTorchObs, features_extractor: Optional[BaseFeaturesExtractor] = None
    ) -> Union[th.Tensor, Tuple[th.Tensor, th.Tensor]]:
        """
        Preprocess the observation if needed and extract features.

        :param obs: Observation
        :param features_extractor: The features extractor to use. If None, then ``self.features_extractor`` is used.
        :return: The extracted features. If features extractor is not shared, returns a tuple with the
            features for the actor and the features for the critic.
        """
        if self.share_features_extractor:
            return super().extract_features(
                obs, self.features_extractor if features_extractor is None else features_extractor
            )
        else:
            if features_extractor is not None:
                warnings.warn(
                    "Provided features_extractor will be ignored because the features extractor is not shared.",
                    UserWarning,
                )

            pi_features = super().extract_features(obs, self.pi_features_extractor)
            vf_features = super().extract_features(obs, self.vf_features_extractor)
            penalty_vf_features = super().extract_features(obs, self.penalty_vf_features_extractor)
            return pi_features, vf_features, penalty_vf_features

    def _get_action_dist_from_latent(
        self, latent_pi: th.Tensor, state: th.Tensor, safe_action_set: Optional[list[Zonotope]] = None
    ) -> Tuple[Distribution, th.Tensor]:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        mean_actions = self.action_net(latent_pi)

        projection_loss = th.zeros((len(mean_actions),))
        distance = th.nn.PairwiseDistance(p=2)  # compute pairwise norm between mean of distribution and safe action

        if self.use_per_sample_loss:
            # We only project here to obtain the projection loss.
            # However, we continue to use the unprojected mean for the action distribution.
            safe_mean_actions, _ = self.safeguard.project_policy_action(
                state.to(mean_actions.dtype), mean_actions, safe_action_set=safe_action_set
            )
            projection_loss = th.square(distance(mean_actions, safe_mean_actions))

        if th.isnan(self.log_std).any():
            raise ValueError("NaN detected in log_std of action distribution")

        # if hasattr(self, 'log_std'):
        #     # Clamp to reasonable range: log(-2) gives std≈0.135, log(2) gives std≈7.39
        #     self.log_std.data = th.clamp(self.log_std.data, min=-2.0, max=2.0)

        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std), projection_loss
        elif isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            return self.action_dist.proba_distribution(action_logits=mean_actions), projection_loss
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            # Here mean_actions are the flattened logits
            return self.action_dist.proba_distribution(action_logits=mean_actions), projection_loss
        elif isinstance(self.action_dist, BernoulliDistribution):
            # Here mean_actions are the logits (before rounding to get the binary actions)
            return self.action_dist.proba_distribution(action_logits=mean_actions), projection_loss
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std, latent_pi), projection_loss
        else:
            raise ValueError("Invalid action distribution")

    def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        return self.get_distribution(observation).get_actions(deterministic=deterministic)

    def evaluate_actions(
        self,
        obs: PyTorchObs,
        states: th.Tensor,
        actions: th.Tensor,
        safe_actions: Optional[th.Tensor] = None,
        safe_action_set: Optional[list[Zonotope]] = None,
    ) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor], Optional[th.Tensor]]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions, entropy of the action distribution,
            mean of the action distribution.
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf, latent_vf_penalty = self.mlp_extractor(features)
        else:
            pi_features, vf_features, penalty_vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
            latent_vf_penalty = self.mlp_extractor.forward_penalty_critic(penalty_vf_features)
        distribution, projection_loss = self._get_action_dist_from_latent(
            latent_pi, states, safe_action_set=safe_action_set
        )

        log_prob = distribution.log_prob(actions)
        # If projected, compute cumulative mass at boundary point
        projected_mask = th.norm(safe_actions - actions, p=2, dim=1) >= 1e-4
        action_distance = th.norm(actions - safe_actions, p=2, dim=1, keepdim=True)
        outward_normals = (actions - safe_actions) / action_distance
        outward_normals = th.where(action_distance < 1e-6, th.ones_like(outward_normals), outward_normals)
        boundary_mass = compute_gaussian_boundary_mass_torch(
            mu=distribution.distribution.mean,
            Sigma=distribution.distribution.stddev**2,
            boundary_points=safe_actions,
            normals=outward_normals,
            squashed=self.squash_output,
        )
        # This seems to be numerically unstable - to be investigated
        # intersection_points = compute_intersection_points_2d(safe_actions, outward_normals, limit=5.0)
        # boundary_mass = independent_gaussian_ray_integral(
        #     distribution.distribution.mean,
        #     distribution.distribution.stddev**2,
        #     safe_actions,
        #     intersection_points
        # )
        hybrid_log_prob = th.where(projected_mask, th.log(boundary_mass), log_prob)

        values = self.value_net(latent_vf)
        if self.use_penalty_critic:
            penalty_values = self.penalty_value_net(latent_vf_penalty)
        else:
            penalty_values = th.zeros_like(values)
        entropy = distribution.entropy()
        return values, penalty_values, hybrid_log_prob, log_prob, entropy, projection_loss

    def get_distribution(self, obs: PyTorchObs) -> Distribution:
        """
        Get the current policy distribution given the observations.

        :param obs:
        :return: the action distribution.
        """
        features = super().extract_features(obs, self.pi_features_extractor)
        latent_pi = self.mlp_extractor.forward_actor(features)
        distribution, proj_loss = self._get_action_dist_from_latent(latent_pi, obs)
        policy_projected = list(proj_loss.detach().cpu().numpy() > 1e-5)
        self.last_projected = policy_projected
        return distribution

    def predict_values(self, obs: PyTorchObs) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation
        :return: the estimated values.
        """
        features = super().extract_features(obs, self.vf_features_extractor)
        latent_vf = self.mlp_extractor.forward_critic(features)
        return self.value_net(latent_vf)

    def predict_penalty_values(self, obs: PyTorchObs) -> th.Tensor:
        """
        Get the estimated penalty values according to the current policy given the observations.

        :param obs: Observation
        :return: the estimated penalty values.
        """
        if not self.use_penalty_critic:
            return None
        else:
            features = super().extract_features(obs, self.penalty_vf_features_extractor)
            latent_vf_penalty = self.mlp_extractor.forward_penalty_critic(features)
            penalty_value = self.penalty_value_net(latent_vf_penalty)
            return penalty_value
