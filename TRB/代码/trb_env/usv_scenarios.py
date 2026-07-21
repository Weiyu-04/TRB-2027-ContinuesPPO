"""
TRB 环境 · 多场景训练支持（step4d-③a）
========================================
论文 §VII：在 ~2000 个 `HandcraftedTwoVesselEncounters` 上训练，**每 episode 抽一个场景**。
本模块提供：
  · `load_scenario_pool(paths)` —— 预加载场景池进内存（D17③：训练前一次性加载、**不每 episode 重读 xml**）。
  · `MultiScenarioEnv`         —— 训练用：每 episode reset 随机抽一场景、重建内层 Shielded/Unshielded env、委托其接口。
  · `make_vec_env`             —— DummyVecEnv（默认、单进程）/ SubprocVecEnv（多进程并行采样，D17④）。

⚠️ **评估**不用 MultiScenarioEnv：`evaluate.evaluate(env_factory, policy, pool)` 本就迭代场景列表（确定性逐场景），
   多场景评估直接 `evaluate(lambda sc,pp: ShieldedUSVEnv(sc,pp), policy, pool)`。MultiScenarioEnv 专供训练随机抽样。
⚠️ **全量 2000 场景**留 step4e 训练前批下载（README §2）；本模块用小子集测机制即可。
⚠️ **SubprocVecEnv 跨进程**：传 **xml 路径**、在子进程内各自 load（避免 spawn 时把大场景对象〔~39KB/场景〕×池×n_envs
   经管道传给每个 worker、各 worker 独立 load 更省内存；**CommonOcean 对象本身可 pickle〔实测 39KB round-trip〕、非不可 pickle**）。
   DummyVecEnv（单进程）直接用预加载 pool。⚠️ 调用方须在 `if __name__=='__main__'` 守护下起 SubprocVecEnv（spawn 约束）。
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from .usv_shield import ShieldedUSVEnv


def load_scenario_pool(paths) -> list:
    """从 xml 路径列表预加载场景池进内存：返回 [(scenario, planning_problem), ...]（D17③）。"""
    from commonocean.common.file_reader import CommonOceanFileReader
    pool = []
    for p in paths:
        sc, pps = CommonOceanFileReader(p).open()
        ppx = list(pps.planning_problem_dict.values())[0]
        pool.append((sc, ppx))
    if not pool:
        raise ValueError("场景池为空（paths 无有效场景）")
    return pool


class MultiScenarioEnv(gym.Env):
    """多场景训练 env（step4d-③a）：预加载池 + 每 episode reset 随机抽一场景 → 重建内层 env → 委托。

    内层每 episode 重建（Shielded/Unshielded）= 状态机/scheduler 自然 fresh（本就该每 episode 重置）。
    spaces 场景无关（同 vessel params / obs 27 / action Discrete(50)）→ 从首个场景的内层 env 定。
    委托 `evaluate.run_episode` 所需契约：reset / step / action_masks / _ego_vs / _obs_vs / `.env.dt`。
    ⚠️ 随机抽样靠 np_random 流推进：训练用 VecEnv auto-reset（无 seed）每 episode 真随机抽场景；
       **固定 seed 每次 reset 会钉死同一场景**（确定性=可复现，别在训练循环里每 episode 传固定 seed）。
    """

    metadata = {"render_modes": []}

    def __init__(self, scenario_pool, *, env_cls=ShieldedUSVEnv, env_kwargs: dict | None = None):
        super().__init__()
        self.pool = list(scenario_pool)
        if not self.pool:
            raise ValueError("scenario_pool 不能为空")
        self.env_cls = env_cls
        self.env_kwargs = dict(env_kwargs or {})
        self._inner = env_cls(self.pool[0][0], self.pool[0][1], **self.env_kwargs)   # 定 spaces
        self.action_space = self._inner.action_space
        self.observation_space = self._inner.observation_space
        self._idx = 0

    # ---- gymnasium 契约（委托内层）----
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._idx = int(self.np_random.integers(len(self.pool)))     # 随机抽一场景（np_random 由 seed 播种）
        sc, pp = self.pool[self._idx]
        self._inner = self.env_cls(sc, pp, **self.env_kwargs)        # 重建内层（fresh 状态机）
        obs, info = self._inner.reset(seed=seed)
        return obs, {**info, "scenario_idx": self._idx}

    def step(self, action):
        return self._inner.step(action)

    def set_penalty_weight(self, name, value):
        """惩罚退火专用 setter（`03` L103·镜像 LR 退火·经 VecEnv.env_method 跨进程调用）：把时变惩罚权重
        （'alias_weight'/'rate_weight'）【双写】——① self.env_kwargs[name]（下次 reset 在 :67 重建 _inner 时透传继承）
        ② 当前 self._inner 的同名属性（当前 episode 立即生效）。
        ⚠️ 必须双写：_inner 每 episode reset 重建（:67 self._inner=env_cls(...,**self.env_kwargs)）→ 单写 _inner 必被
        下次 reset 抹掉=退火静默失效；单写 env_kwargs 则当前 episode 不变。也【不能】用 VecEnv.set_attr（只设本 wrapper
        顶层属性、够不到 _inner）。校验同 ContinuousProjectionEnv 构造守卫（isfinite≥0·shield:74/78）→ 挡 nan/inf 经
        env_method 序列化进子进程毒化奖励。退火【关闭】时本方法【从不被调用】→ env_kwargs/_inner 权重恒为构造常量（默认 0.0）= 字节级不变。
        """
        if name not in ("alias_weight", "rate_weight"):
            raise ValueError(f"set_penalty_weight 仅支持 'alias_weight'/'rate_weight'，得 {name!r}")
        v = float(value)
        if not (np.isfinite(v) and v >= 0.0):
            raise ValueError(f"{name} 必须是有限非负数，得 {v}")
        self.env_kwargs[name] = v                       # 下次 reset 重建 _inner 时透传继承
        if hasattr(self._inner, name):
            setattr(self._inner, name, v)               # 当前 episode 立即生效

    def set_arrival_slack(self, slack):
        """🆕 B1（`03` L153）：到达门朝向容差课程 setter（镜像 set_penalty_weight 双写·经 VecEnv.env_method 跨进程调用）。
        ⚠️ 与 penalty 退火【关键不同】：目标（term_checker.arrival_heading_slack）在【更深一层】
        （_inner=ContinuousProjectionEnv → USVEnv → term_checker）→ 立即写用【方法调用穿透】
        `self._inner.set_arrival_slack(v)`（**非 setattr**·setattr 只会在 _inner 顶层挂个死属性、到不了内层 term_checker）。双写：
        ① self.env_kwargs['arrival_heading_slack']（下次 reset 重建 _inner 时透传继承）② 当前 _inner 穿透立即生效。
        必须双写：_inner 每 episode reset 重建 → 单写 _inner 必被下次 reset 抹掉=退火静默失效；单写 env_kwargs 则当前 episode 不变。
        校验 isfinite≥0（同 set_penalty_weight·挡 nan/inf 经 env_method 序列化进子进程毒化到达判定；宽度 clamp 在内层 term_checker）。
        退火【关闭】时本方法【从不被调用】→ env_kwargs 无此键、_inner 恒 slack=0 = 字节级不变（仅连续臂·**eval 恒不调=真门诚实红线**）。"""
        v = float(slack)
        if not (np.isfinite(v) and v >= 0.0):
            raise ValueError(f"arrival_heading_slack 必须是有限非负数，得 {v}")
        self.env_kwargs["arrival_heading_slack"] = v    # ① 下次 reset 重建 _inner 时透传继承
        if hasattr(self._inner, "set_arrival_slack"):
            self._inner.set_arrival_slack(v)            # ② 当前 episode 立即生效（方法穿透到内层 term_checker·非 setattr）

    def set_start_frac(self, frac, v=None):
        """🆕 逆向起点课程 setter（`03` L181·Florensa 2017·镜像 set_arrival_slack 双写·经 VecEnv.env_method 跨进程调用·目标在内层 USVEnv.start_frac）。
        双写：① self.env_kwargs['start_frac']（下次 reset:67 重建 _inner 透传继承）② 当前 _inner 穿透立即生效（方法调用非 setattr）。
        frac=1.0（默认/评估·退火关时【从不被调用】）→ 真起点 = 字节级不变；frac<1（仅训练）→ ego 生更靠门。**eval 恒不调=真起点诚实红线（同 arrival_slack）**。
        校验 0<frac≤1（frac=0 会把 ego 生在 goal 中心=退化·挡）。start_v 可选（None=不改内层默认真init速度）。"""
        f = float(frac)
        if not (np.isfinite(f) and 0.0 < f <= 1.0):
            raise ValueError(f"start_frac 必须 0<frac≤1（得 {f}·1=真起点/→0贴门/0退化）")
        self.env_kwargs["start_frac"] = f               # ① 下次 reset 重建 _inner 时透传继承
        if v is not None:
            self.env_kwargs["start_v"] = float(v)
        # 🔴修(L182 对抗审 LOW·双写对称)：v 省略时用 env_kwargs 现存 start_v 做立即写·保"当前 episode"与"下次 reset 重建"一致（退火 callback 只改 frac 不动 v 时不误清）。
        _v_eff = float(v) if v is not None else self.env_kwargs.get("start_v")
        if hasattr(self._inner, "set_start_frac"):
            self._inner.set_start_frac(f, _v_eff)       # ② 当前 episode 立即生效（方法穿透到内层 USVEnv·非 setattr）

    def render(self):
        return None

    # ---- 委托：MaskablePPO mask + ViolationCounter 喂数 + run_episode 的 .env.dt ----
    def action_masks(self) -> np.ndarray:
        return self._inner.action_masks()

    def _ego_vs(self):
        return self._inner._ego_vs()

    def _obs_vs(self):
        return self._inner._obs_vs()

    @property
    def env(self):
        """run_episode 用 `env.env.dt`；委托内层的底层 USVEnv。"""
        return self._inner.env


def make_vec_env(scenario_pool=None, *, n_envs: int = 4, env_cls=ShieldedUSVEnv,
                 env_kwargs: dict | None = None, subproc: bool = False, seed: int = 0,
                 paths=None):
    """构造 n_envs 个 `MultiScenarioEnv` 的 VecEnv（采样并行，D17④）。

    DummyVecEnv（默认，单进程）：用预加载 `scenario_pool`。
    SubprocVecEnv（subproc=True，多进程）：传 `paths`（xml 路径列表）、**在各子进程内 load_scenario_pool**
        （避免把大场景对象经 spawn 管道传给每 worker、各自 load 更省内存；CommonOcean 对象本身可 pickle、非不可）。
        ⚠️ 调用方须在 `if __name__=='__main__'` 守护下（macOS spawn 约束）。
    """
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    if subproc:
        if paths is None:
            raise ValueError("subproc=True 需传 paths（子进程内各自加载，避免跨进程 pickle 场景对象）")

        def _mk_sub(rank):
            def _f():
                pool = load_scenario_pool(paths)
                e = MultiScenarioEnv(pool, env_cls=env_cls, env_kwargs=env_kwargs)
                e.reset(seed=seed + rank)
                return e
            return _f
        return SubprocVecEnv([_mk_sub(i) for i in range(n_envs)])

    if scenario_pool is None:
        if paths is None:
            raise ValueError("DummyVecEnv 需传 scenario_pool 或 paths")
        scenario_pool = load_scenario_pool(paths)

    def _mk(rank):
        def _f():
            e = MultiScenarioEnv(scenario_pool, env_cls=env_cls, env_kwargs=env_kwargs)
            e.reset(seed=seed + rank)
            return e
        return _f
    return DummyVecEnv([_mk(i) for i in range(n_envs)])
