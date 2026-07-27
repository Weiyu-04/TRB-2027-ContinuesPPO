"""路线 B 可行性原型：SB3 PPO + **Beta 分布**（有界支撑）——`03` L229-E。

🔴 **本文件【未接进主流程】**，是留给下个窗口的起点，不被任何生产代码 import。
   直接跑自检：`python 代码/参考_beta分布原型_未接线.py`（约 1 分钟，纯合成 env，不碰场景/存档）。

为什么要它：现在连续臂用的是**无界对角高斯 + 硬裁剪**，σ_ω 训练中涨到 2.18× 半箱 ⟹ 73% 的步打满舵、
0% 能输出居中舵；且裁剪之外是**梯度平高原**，治抖奖励结构上无着力点（`03` L229-C/D）。
Beta 的支撑天生就是一个闭区间 ⟹ 数学上不可能越箱，且能表达细微转向。

🔴 **不可省的设计**：α,β 用 `softplus(·)+1` **强制 ≥1**。不强制则 Beta 可退化成 U 形（两端高中间低）
   = **数学形式的 bang-bang，比现状更糟**。≥1 ⟹ 恒单峰、众数在区间内部、端点密度有限。

🔴 **接线时必须一起做的三件事**（漏一件就会出现"用 A 分布训、用 B 分布评"而聚合数字上完全看不出来）：
   ① 新开关 `STEP4E_ACT_DIST`（gauss|beta|sde·默认 gauss = 与现状逐位一致）进 `config_sig`；
   ② 同步进 `run_step4e.py` 的 `_SEMANTIC_KEYS` **和** `_cur_sig_probe`（后者现在漏了 ctrl_* 两键 = 闸空转·`03` L229-F）；
   ③ `tests/reeval_official.py` 要 import 到本策略类所在模块，否则 `PPO.load` 解析不出类。

⚠️ 换分布 ⟹ **旧存档灌不进去**（高斯 log_std 形状 (2,)；gSDE (64,2)；Beta 无此参数）⟹ 只能从零训。
"""
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
    import tempfile, os as _os
    _TMP = _os.path.join(tempfile.mkdtemp(), "beta_proto.zip")
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
    m.save(_TMP)
    m2 = PPO.load(_TMP)
    print("  save/load 往返 OK, 动作 =", m2.predict(np.zeros(27, np.float32), deterministic=True)[0])
