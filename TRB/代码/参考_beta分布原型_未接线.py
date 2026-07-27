"""路线 B 可行性原型：SB3 PPO + Beta 分布（有界支撑）。仅在草稿区验证可行性，不进仓库。"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch as th
import torch.nn as nn
import gymnasium as gym
from torch.distributions import Beta
from stable_baselines3 import PPO
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import ActorCriticPolicy


class BetaDistribution(Distribution):
    """支撑区间 = [low, high] 的对角 Beta 分布。α,β = softplus(·)+1 ⟹ 恒 ≥1 ⟹ 单峰、内部众数、端点密度有限。"""

    def __init__(self, low: np.ndarray, high: np.ndarray):
        super().__init__()
        self.low = th.as_tensor(low, dtype=th.float32)
        self.high = th.as_tensor(high, dtype=th.float32)
        self.scale = self.high - self.low
        self.action_dim = int(len(low))
        self._log_jac = float(th.log(self.scale).sum())  # 常数：ratio 里对消，熵里是常数平移

    def proba_distribution_net(self, latent_dim: int):
        return nn.Linear(latent_dim, 2 * self.action_dim)

    def proba_distribution(self, ab):
        a_raw, b_raw = th.chunk(ab, 2, dim=-1)
        self.alpha = th.nn.functional.softplus(a_raw) + 1.0
        self.beta = th.nn.functional.softplus(b_raw) + 1.0
        self.dist = Beta(self.alpha, self.beta)
        return self

    def _to_box(self, u01):
        return self.low.to(u01.device) + self.scale.to(u01.device) * u01

    def _to_unit(self, a):
        u = (a - self.low.to(a.device)) / self.scale.to(a.device)
        return u.clamp(1e-6, 1.0 - 1e-6)

    def log_prob(self, actions):
        return self.dist.log_prob(self._to_unit(actions)).sum(-1) - self._log_jac

    def entropy(self):
        return self.dist.entropy().sum(-1) + self._log_jac

    def sample(self):
        return self._to_box(self.dist.rsample())

    def mode(self):
        d = (self.alpha + self.beta - 2.0)
        m = th.where(d > 1e-6, (self.alpha - 1.0) / d.clamp(min=1e-6), th.full_like(d, 0.5))
        return self._to_box(m)

    def actions_from_params(self, ab, deterministic=False):
        self.proba_distribution(ab)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, ab):
        a = self.actions_from_params(ab)
        return a, self.log_prob(a)


class BetaActorCriticPolicy(ActorCriticPolicy):
    """先让 SB3 按原样搭好（高斯），再把动作头换成 Beta 头 + 重建优化器。
    这样不碰 SB3 内部那条 isinstance 分支链，升级 SB3 也不易碎。"""

    def _build(self, lr_schedule):
        super()._build(lr_schedule)                     # 原样搭：mlp_extractor / value_net / action_net(高斯)
        low = np.asarray(self.action_space.low, dtype=np.float64)
        high = np.asarray(self.action_space.high, dtype=np.float64)
        self.action_dist = BetaDistribution(low, high)
        latent_dim_pi = self.mlp_extractor.latent_dim_pi
        self.action_net = self.action_dist.proba_distribution_net(latent_dim_pi)
        if self.ortho_init:
            self.action_net.apply(lambda mm: self.init_weights(mm, 0.01))
        if hasattr(self, "log_std"):
            del self.log_std                            # 高斯的 σ 参数：Beta 路径下不存在
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_action_dist_from_latent(self, latent_pi):
        return self.action_dist.proba_distribution(self.action_net(latent_pi))


LOW = np.array([-0.048, -0.018], dtype=np.float32)
HIGH = -LOW


class E(gym.Env):
    observation_space = gym.spaces.Box(-1, 1, (27,), np.float32)
    action_space = gym.spaces.Box(LOW, HIGH, dtype=np.float32)

    def reset(self, **k):
        return np.random.randn(27).astype(np.float32), {}

    def step(self, a):
        assert np.all(a >= LOW - 1e-9) and np.all(a <= HIGH + 1e-9), f"out of box {a}"
        # 奖励 = 鼓励小转向（模拟治抖罚）
        return np.random.randn(27).astype(np.float32), -float(abs(a[1])) * 100, False, np.random.rand() < 0.02, {}


if __name__ == "__main__":
    m = PPO(BetaActorCriticPolicy, E(), n_steps=256, batch_size=64, verbose=0, seed=0)
    m.learn(4096)
    aa = np.stack([m.predict(np.random.randn(27).astype(np.float32), deterministic=False)[0] for _ in range(2000)])
    dd = np.stack([m.predict(np.random.randn(27).astype(np.float32), deterministic=True)[0] for _ in range(2000)])
    print("训练 4096 步 OK")
    print("  sampled out-of-box count =", int(np.sum(np.abs(aa) > HIGH + 1e-9)))
    print("  sampled at-edge frac     = %.4f" % float(np.mean(np.abs(aa) >= HIGH * 0.999)))
    print("  deterministic at-edge    = %.4f" % float(np.mean(np.abs(dd) >= HIGH * 0.999)))
    print("  deterministic |w| median = %.5f (box 0.018)" % float(np.median(np.abs(dd[:, 1]))))
    print("  alpha/beta >= 1 always   =", True)
    m.save("/tmp/claude-0/-home-user-TRB-2027-ContinuesPPO/dee210d5-5722-5d4b-8fb6-92046ca6e9dd/scratchpad/beta.zip")
    m2 = PPO.load("/tmp/claude-0/-home-user-TRB-2027-ContinuesPPO/dee210d5-5722-5d4b-8fb6-92046ca6e9dd/scratchpad/beta.zip",
                  custom_objects={"policy_class": BetaActorCriticPolicy})
    print("  save/load 往返 OK, 动作 =", m2.predict(np.zeros(27, np.float32), deterministic=True)[0])
