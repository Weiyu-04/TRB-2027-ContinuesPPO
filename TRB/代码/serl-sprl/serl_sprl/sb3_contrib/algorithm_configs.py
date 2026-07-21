import stable_baselines3.common.preprocessing as preprocessing
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.sac import SAC
from stable_baselines3.td3 import TD3

from serl_sprl.envs.configs import BaseModel
from serl_sprl.sb3_contrib.algorithms.a2c.a2c import A2C
from serl_sprl.sb3_contrib.algorithms.ppo.ppo import PPO
from serl_sprl.sb3_contrib.helpers import patch_after_imports, preprocess_obs_float64

# Apply the patch
preprocessing.preprocess_obs = preprocess_obs_float64

from serl_sprl.sb3_contrib.algorithms.a2c.a2cdiffproj import A2CDiffProj
from serl_sprl.sb3_contrib.algorithms.ppo.ppodiffproj import PPODiffProj
from serl_sprl.sb3_contrib.algorithms.sac.sacdiffproj import SACDiffProj
from serl_sprl.sb3_contrib.algorithms.td3.td3diffproj import TD3DiffProj

patch_after_imports()


class SACHyperparameters(BaseModel):
    learning_rate: float
    batch_size: int
    gamma: float
    tau: float = 0.005
    buffer_size: int = 100000
    train_freq: int = 1
    gradient_steps: int = 1


class SACDiffProjHyperparameters(SACHyperparameters):
    use_penalty_critic: bool
    use_per_sample_loss: bool
    penalty_factor: float = 0.0


class TD3Hyperparameters(BaseModel):
    learning_rate: float
    action_noise: ActionNoise
    batch_size: int
    gamma: float
    buffer_size: int = 100000
    train_freq: int = 1
    gradient_steps: int = 1


class TD3DiffProjHyperparameters(TD3Hyperparameters):
    use_penalty_critic: bool
    use_per_sample_loss: bool
    penalty_factor: float = 0.0


class A2CHyperparameters(BaseModel):
    learning_rate: float
    n_steps: int
    ent_coef: float
    vf_coef: float
    gamma: float
    gae_lambda: float
    max_grad_norm: float
    use_rms_prop: bool = True
    normalize_advantage: bool = False
    use_sde: bool = True


class A2CDiffProjHyperparameters(A2CHyperparameters):
    use_penalty_critic: bool
    use_per_sample_loss: bool
    penalty_factor: float = 0.0


class A2CConfig(BaseModel):
    total_timesteps: int
    policy_kwargs: dict
    hyperparams: A2CHyperparameters
    algorithm: BaseAlgorithm = A2C
    policy: str = "MlpPolicy"


class A2CDiffProjConfig(A2CConfig):
    algorithm: BaseAlgorithm = A2CDiffProj
    hyperparams: A2CDiffProjHyperparameters


class PPOHyperparameters(BaseModel):
    learning_rate: float
    n_steps: int
    ent_coef: float
    gamma: float
    gae_lambda: float
    batch_size: int
    clip_range: float
    n_epochs: int = 10
    normalize_advantage: bool = False
    use_sde: bool = True


class PPODiffProjHyperparameters(PPOHyperparameters):
    use_penalty_critic: bool
    use_per_sample_loss: bool
    penalty_factor: float = 0.0


class PPOConfig(BaseModel):
    total_timesteps: int
    policy_kwargs: dict
    hyperparams: PPOHyperparameters
    algorithm: BaseAlgorithm = PPO
    policy: str = "MlpPolicy"


class PPODiffProjConfig(PPOConfig):
    algorithm: BaseAlgorithm = PPODiffProj
    hyperparams: PPODiffProjHyperparameters


class TD3Config(BaseModel):
    total_timesteps: int
    policy_kwargs: dict
    policy: str = "MlpPolicy"
    hyperparams: TD3Hyperparameters
    algorithm: BaseAlgorithm = TD3


class TD3DiffProjConfig(TD3Config):
    algorithm: BaseAlgorithm = TD3DiffProj
    hyperparams: TD3DiffProjHyperparameters


class SACConfig(BaseModel):
    total_timesteps: int
    policy_kwargs: dict
    policy: str = "MlpPolicy"
    hyperparams: SACHyperparameters
    algorithm: BaseAlgorithm = SAC


class SACDiffProjConfig(SACConfig):
    algorithm: BaseAlgorithm = SACDiffProj
    hyperparams: SACDiffProjHyperparameters
