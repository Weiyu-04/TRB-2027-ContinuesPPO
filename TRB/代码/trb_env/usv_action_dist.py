"""连续臂**有界动作分布**（Beta）——治 bang-bang 的根因修法（`03` L229-E / L230-§4）。

═══ 为什么要它（根因链·全部实测·`03` L229-C/D + L230-§1）═══
现在连续臂用 sb3 PPO 默认的 **无界对角高斯 + 硬裁剪**：
  ① 10 个存档直读 `log_std`：σ_a = 1.438× 半箱、**σ_ω = 2.182× 半箱**；
  ② 真实观测下策略的**确定性均值本身**就有 **76~84% 的步落在动作箱外**（中位 2.0~3.2× 半箱·L230-§1）
     ⟹ 就算把探索方差压到 0，裁剪照样把它按死在满舵上 ⟹ **压 σ / 退火熵系数这类便宜修法全部无效**；
  ③ 轨迹反推：**73.5% 的步贴箱边**、恰居中 0.1%、满舵左右一步翻面 23.3%（贡献 |Δω| 的 53.0%）；
  ④ SB3 `on_policy_algorithm.py` **L193** 把动作裁剪后送 env、**L226** 把**未裁剪**动作存进 buffer
     ⟹ 均值一旦出箱，再往外挪奖励一分不变 = **梯度平高原**，治抖罚项（`rate_weight`，一直开在 1.0）无着力点；
  ⑤ 高斯熵 = Σ(logσ)+常数、**无上界**，而 `ent_coef` 恒 0.01 ⟹ σ 越大越多样本被裁到箱角、`r_rate` 边际代价趋 0，
     熵奖励却仍随 logσ 线性增长 ⟹ **σ 有一条"白拿"的上行通道**。这是 σ 从 0.009 涨到 0.039 的驱动力。

Beta 同时堵住 ③④⑤：支撑**天生**就是闭区间 [low, high] ⟹ 动作**不可能**越箱 ⟹ 裁剪层退化成恒等 ⟹
`r_rate` 对策略参数**处处有梯度**（平高原消失）；且 Beta 的熵**有上界**（α=β=1 的均匀分布）⟹ 熵系数顶不出去。
实测佐证（L230-§4e）：ent_coef 取 0 / 0.01 / 0.05 三档，α,β 都停在 1.6~2.0、采样贴边率**全 0.000%**。

═══ 🔴 α,β 强制 ≥1 不可省（两条理由·第二条是命门·L230-§4d）═══
① **形状**：α 或 β < 1 时 Beta 是 **U 形**（两端密度发散、中间低）= 数学形式的 bang-bang。
   实测（三种奖励 × 加/不加 ≥1·各 40960 步）：不加时 **97.7~100% 的状态** α<1 或 β<1、**28.8~99.9% 是完整 U 形**；
   加了之后全为 0.00%。成因还包括初始化——`ortho_init` gain 0.01 ⟹ 网络初始输出≈0 ⟹ `softplus(0)=0.693<1`
   ⟹ **一开局就是 U 形**，训练也拉不回来。
② 🔴 **众数（= 确定性动作）**：α 或 β < 1 时 Beta 的众数在**两个端点**，区间内部那个驻点是**密度极小值**。
   而 `mode()` 在 α+β−2 ≤ 0 时只能兜底返回区间正中 ⟹ **确定性策略会输出"最不可能的动作"**。
   **我们所有报数都用 `deterministic=True` ⟹ 这会静默污染每一个数字、而聚合指标上完全看不出来。**

⚠️ **别把 ≥1 理解成"它会阻止满舵"**（L230-§4d 实测反例）：α=8.19 / β=1.002 时众数 = (α−1)/(α+β−2) = 0.9997
   ≈ 贴着箱边、确定性 |ω| 达 95.6% 箱。≥1 只保证**单峰 + 端点密度有限**，不保证不打满舵——
   真正治病的是"支撑全在箱内 ⟹ 罚项处处有梯度"和"熵有上界"这两条。

═══ 换分布的硬约束 ═══
**旧存档灌不进去**（高斯 `log_std` 形状 (2,)；Beta 无此参数）⟹ 只能**从零训**，热启动源必须同为 Beta
（`run_step4e.py` 的 `_SEMANTIC_KEYS` 已含 `act_dist`，配错会 fail-fast）。

═══ 评估端必须能 import 到本模块 ═══
`PPO.load` 反序列化 policy 类靠**模块可导入**。`run_step4e.replay_eval` 在 `PPO.load` 之前显式
`import trb_env.usv_action_dist`，`tests/reeval_official.py` 经 `replay_eval` 间接吃到 ⟹ 不必各处手动 import。
"""
from __future__ import annotations

import numpy as np
import torch as th
import torch.nn as nn
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import ActorCriticPolicy
from torch.distributions import Beta as _TorchBeta

#: `mode()` 的兜底阈值：α+β−2 ≤ 此值时无良定义内点众数（softplus+1 下不可达·仅防数值退化）
_MODE_EPS = 1e-6
#: `_to_unit` 的夹紧量：Beta 在 0/1 处 log_prob 可能 ±inf（α 或 β = 1 时端点密度有限但仍需防 0/1 精确落点）
_UNIT_EPS = 1e-6


class BetaDistribution(Distribution):
    """支撑 = [low, high] 的**对角 Beta** 分布（每个动作维独立）。

    参数化：网络输出 2·d 个 logit → `α = softplus(·)+1`、`β = softplus(·)+1` ⟹ **恒 > 1**
    ⟹ 单峰、端点密度有限、众数严格落在开区间内（理由见模块 docstring）。

    与 SB3 `DiagGaussianDistribution` 的接口对齐（`proba_distribution_net` / `proba_distribution` /
    `log_prob` / `entropy` / `sample` / `mode` / `actions_from_params` / `log_prob_from_params`），
    因此可直接替换 `ActorCriticPolicy.action_dist` 而不动 SB3 其余代码路径。

    坐标变换：Y = low + (high−low)·X，X ~ Beta(α,β) on [0,1]。
      · log p_Y(y) = log p_X(x) − Σ log(high−low)
      · h(Y)       = h(X)       + Σ log(high−low)
    常数 `Σ log(high−low)` 在 PPO 的 ratio 里对消，只对熵是常数平移（不改梯度）。
    """

    def __init__(self, low, high):
        super().__init__()
        low = np.asarray(low, dtype=np.float64)
        high = np.asarray(high, dtype=np.float64)
        if low.shape != high.shape or low.ndim != 1:
            raise ValueError(f"low/high 须同形一维，得 low{low.shape} high{high.shape}")
        if not np.all(high > low):
            raise ValueError(f"须 high > low 逐维成立，得 low={low} high={high}")
        self.low = th.as_tensor(low, dtype=th.float32)
        self.high = th.as_tensor(high, dtype=th.float32)
        self.scale = self.high - self.low
        self.action_dim = int(low.shape[0])
        self._log_jac = float(th.log(self.scale).sum())
        self.alpha = self.beta = self.dist = None

    # ---- SB3 Distribution 接口 ----
    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        """动作头：latent → 2·d 个 logit（前 d 个给 α、后 d 个给 β）。"""
        return nn.Linear(latent_dim, 2 * self.action_dim)

    def proba_distribution(self, action_logits: th.Tensor) -> "BetaDistribution":
        a_raw, b_raw = th.chunk(action_logits, 2, dim=-1)
        self.alpha = th.nn.functional.softplus(a_raw) + 1.0      # 🔴 +1 不可省（见模块 docstring）
        self.beta = th.nn.functional.softplus(b_raw) + 1.0
        self.dist = _TorchBeta(self.alpha, self.beta)
        return self

    def _to_box(self, u01: th.Tensor) -> th.Tensor:
        return self.low.to(u01.device) + self.scale.to(u01.device) * u01

    def _to_unit(self, actions: th.Tensor) -> th.Tensor:
        u = (actions - self.low.to(actions.device)) / self.scale.to(actions.device)
        return u.clamp(_UNIT_EPS, 1.0 - _UNIT_EPS)

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        return self.dist.log_prob(self._to_unit(actions)).sum(-1) - self._log_jac

    def entropy(self) -> th.Tensor:
        return self.dist.entropy().sum(-1) + self._log_jac

    def sample(self) -> th.Tensor:
        return self._to_box(self.dist.rsample())

    def mode(self) -> th.Tensor:
        """众数 = (α−1)/(α+β−2)（α,β>1 时良定义且严格在开区间内）。

        α+β−2 ≤ `_MODE_EPS` 在 softplus+1 参数化下**不可达**（α,β 恒 >1 ⟹ 和 >2）；
        兜底返回区间正中仅为数值退化保险，**不是** α,β<1 情形的正确众数
        （那时众数在两端、正中是密度极小值——正因如此才必须强制 ≥1，见模块 docstring）。
        """
        d = self.alpha + self.beta - 2.0
        m = th.where(d > _MODE_EPS, (self.alpha - 1.0) / d.clamp(min=_MODE_EPS), th.full_like(d, 0.5))
        return self._to_box(m.clamp(0.0, 1.0))

    def actions_from_params(self, action_logits: th.Tensor, deterministic: bool = False) -> th.Tensor:
        self.proba_distribution(action_logits)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, action_logits: th.Tensor):
        actions = self.actions_from_params(action_logits)
        return actions, self.log_prob(actions)


class BetaActorCriticPolicy(ActorCriticPolicy):
    """`ActorCriticPolicy` + Beta 动作头。

    **实现策略 = 先让 SB3 按原样搭好（高斯），再换动作头**：
      · 不碰 SB3 `policies.py` L607 那条 `isinstance(self.action_dist, ...)` 分支链（升级 SB3 不易碎）；
      · 换完 `action_net` 后**必须重建 optimizer**——`super()._build()` 建的 optimizer 绑的是**旧**动作头
        的参数对象，不重建则新头**永远不更新**、旧头还占着优化器状态（静默学不动）。
    删掉高斯的 `log_std`：Beta 路径下不存在该参数；留着会进 state_dict ⟹ 与源/目标键集比对（热启动
    结构守卫）和 `save/load` 语义都出现幽灵参数。
    """

    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)                       # 原样搭：mlp_extractor / value_net / action_net(高斯)
        self.action_dist = BetaDistribution(np.asarray(self.action_space.low, dtype=np.float64),
                                            np.asarray(self.action_space.high, dtype=np.float64))
        self.action_net = self.action_dist.proba_distribution_net(self.mlp_extractor.latent_dim_pi)
        if self.ortho_init:
            self.action_net.apply(lambda m: self.init_weights(m, 0.01))
        if hasattr(self, "log_std"):
            del self.log_std                              # 高斯专属 σ 参数：Beta 无此物
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> BetaDistribution:
        return self.action_dist.proba_distribution(self.action_net(latent_pi))


#: `STEP4E_ACT_DIST` 取值 → PPO policy 实参。'gauss' 走 SB3 原生 "MlpPolicy"（=现状 bit-identical）
ACT_DIST_CHOICES = ("gauss", "beta")


def policy_for(act_dist: str):
    """`act_dist` → 传给 `PPO(...)` 的 policy 实参。未知取值 fail-fast（不静默回落高斯）。"""
    a = (act_dist or "gauss").strip().lower()
    if a == "gauss":
        return "MlpPolicy"
    if a == "beta":
        return BetaActorCriticPolicy
    raise ValueError(f"act_dist 须 ∈ {ACT_DIST_CHOICES}，得 {act_dist!r}")
