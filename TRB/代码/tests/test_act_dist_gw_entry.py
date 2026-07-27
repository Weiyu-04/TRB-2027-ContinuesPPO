#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""L230 两个新开关的回归套件：`STEP4E_ACT_DIST`（动作分布档）+ `STEP4E_GW_ENTRY`（状态机让路入口档）。

跑法（本机即可·不需要 GPU·约 1-2 分钟）：
    STEP4E_SDIR=<TRB>/scenarios python -B 代码/tests/test_act_dist_gw_entry.py

覆盖四类：
  A 逐位等价（**最要紧**）：开关取默认值时，状态机转移序列 / PPO 策略类 / env_kwargs 构造 **与改动前完全一致**。
  B 机制正确：Beta 零越箱、α,β 恒 >1、众数在开区间内、save/load 往返一致、optimizer 真重建、log_std 真删；
             symmetric 档在"开局即让路态势"上真的能进 give-way，而 paper 档进不去。
  C 防静默：非法取值 fail-fast；训练/评估同档（sidecar 回读）；热启动跨档被拦。
  D 一致性：前瞻用的临时状态机与在役状态机同档（否则终端可行性判据与真实转移不一致）。
"""
import os
import sys
import json
import tempfile
import warnings

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))            # 代码/
import numpy as np                                     # noqa: E402
import torch as th                                     # noqa: E402

from trb_env.usv_colregs import (VesselState, ColregsStatechart, SafeActionScheduler,   # noqa: E402
                                 crossing, is_emergency, persistent_crossing,
                                 DT, T_HORIZON, RHO_CROSSING, RHO_HEAD_ON, RHO_OVERTAKE,
                                 GW_ENTRY_CHOICES)
from trb_env.usv_action_dist import (BetaDistribution, BetaActorCriticPolicy,   # noqa: E402
                                     ACT_DIST_CHOICES, policy_for)

_L = 175.0
_GW = (RHO_HEAD_ON, RHO_CROSSING, RHO_OVERTAKE)
_PASS = _FAIL = 0


def ok(cond, msg):
    global _PASS, _FAIL
    if cond:
        _PASS += 1
        print(f"[PASS] {msg}")
    else:
        _FAIL += 1
        print(f"[FAIL] {msg}")


def _adv(s):
    return VesselState(s.position + s.velocity * DT * np.array([np.cos(s.orientation), np.sin(s.orientation)]),
                       s.orientation, s.velocity, _L)


def _find_starts(n=6, seed=0):
    """随机搜"第 0 步 crossing() 已为真、且未到紧急"的初始状态（= 场景开局就处于让路态势）。"""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(400000):
        d = rng.uniform(800, 4000)
        b = np.deg2rad(rng.uniform(6, 110))
        e = VesselState(np.array([0.0, 0.0]), 0.0, rng.uniform(4, 9.5), _L)
        o = VesselState(d * np.array([np.cos(-b), np.sin(-b)]), rng.uniform(-np.pi, np.pi), rng.uniform(3, 9), _L)
        if crossing(e, o, T_HORIZON) and not is_emergency(e, o):
            out.append((e, o))
            if len(out) >= n:
                break
    return out


# ════════════════════════════ A. 逐位等价 ════════════════════════════
def test_bit_identical():
    print("\n── A 逐位等价（开关取默认 = 改动前行为）──")
    ok(ColregsStatechart().gw_entry == "paper", "① ColregsStatechart 默认档 = 'paper'")
    ok(SafeActionScheduler()._statechart.gw_entry == "paper", "② SafeActionScheduler 默认档 = 'paper'")
    ok(policy_for("gauss") == "MlpPolicy", "③ act_dist='gauss' → SB3 原生 'MlpPolicy'（=改动前那一支）")

    # 默认档下，状态机在随机轨迹上的转移序列必须与"根本没有这个分支"完全一致。
    # 等价参照 = paper 档的 _giveway_persistent 结果（symmetric 分支被 `and self.gw_entry == "symmetric"` 短路）。
    rng = np.random.default_rng(7)
    same = True
    for _ in range(40):
        e = VesselState(np.array([0.0, 0.0]), rng.uniform(-np.pi, np.pi), rng.uniform(3, 9.5), _L)
        o = VesselState(rng.uniform(-3000, 3000, 2), rng.uniform(-np.pi, np.pi), rng.uniform(3, 9), _L)
        a, b = ColregsStatechart(), ColregsStatechart(gw_entry="paper")
        ee, oo = e, o
        for _t in range(30):
            if a.step(ee, oo) != b.step(ee, oo):
                same = False
                break
            ee, oo = _adv(ee), _adv(oo)
        if not same:
            break
    ok(same, "④ 默认档与显式 'paper' 档在 40×30 步随机轨迹上转移序列逐步相同")

    # symmetric 只加不减：它认出的让路态是 paper 的超集（persistent 中了两边一样，没中才多看一眼）
    starts = _find_starts(4, seed=3)
    superset = True
    for e, o in starts:
        p, q = ColregsStatechart(gw_entry="paper"), ColregsStatechart(gw_entry="symmetric")
        ee, oo = e, o
        for _t in range(60):
            rp, rq = p.step(ee, oo), q.step(ee, oo)
            if rp in _GW and rq not in _GW:
                superset = False
            ee, oo = _adv(ee), _adv(oo)
    ok(superset, "⑤ 'symmetric' 认出的让路步是 'paper' 的超集（只加不减·不会把已认出的判丢）")


# ════════════════════════════ B. 机制正确 ════════════════════════════
def test_beta_mechanism():
    print("\n── B1 Beta 分布数学与工程 ──")
    low = np.array([-0.048, -0.018])
    high = -low
    d = BetaDistribution(low, high)
    net = d.proba_distribution_net(16)
    logits = th.randn(4096, 4) * 3.0                      # 故意给大 logit，逼极端 α/β
    d.proba_distribution(logits)
    ok(bool((d.alpha > 1.0).all() and (d.beta > 1.0).all()), "① α,β 恒 > 1（softplus+1·哪怕 logit 很大很小）")

    a = d.sample().numpy()
    n_out = int(np.sum((a > high + 1e-9) | (a < low - 1e-9)))
    ok(n_out == 0, f"② 采样零越箱（4096×2 个样本·越箱 {n_out} 个）")

    m = d.mode().numpy()
    ok(bool(np.all(m > low) and np.all(m < high)), "③ 众数严格落在开区间内（不是端点·也不是兜底的箱正中）")

    # 众数确实是密度最大点（拿一维小网格暴力核）
    d1 = BetaDistribution(np.array([-1.0]), np.array([1.0]))
    d1.proba_distribution(th.tensor([[0.8, 0.2]]))
    grid = th.linspace(-0.999, 0.999, 4001).unsqueeze(-1)
    lp = d1.log_prob(grid).numpy()
    ok(abs(float(grid[int(np.argmax(lp))]) - float(d1.mode()[0, 0])) < 5e-3,
       "④ mode() 与网格暴力搜出的密度最大点一致（说明 mode 是真众数、不是反众数）")

    # log_prob 的坐标变换：∫ p(y) dy 应 ≈ 1
    dy = float(grid[1] - grid[0])
    ok(abs(float(np.exp(lp).sum() * dy) - 1.0) < 1e-2, "⑤ log_prob 归一（∫p(y)dy≈1·雅可比项没写反）")

    # 熵的坐标变换：h(Y) = h(X) + log(scale)
    from torch.distributions import Beta as TB
    hx = TB(d1.alpha, d1.beta).entropy().sum(-1)
    ok(bool(th.allclose(d1.entropy(), hx + float(np.log(2.0)), atol=1e-5)), "⑥ entropy 坐标变换正确（+log scale）")

    ok(policy_for("beta") is BetaActorCriticPolicy, "⑦ act_dist='beta' → BetaActorCriticPolicy")
    ok(policy_for(" Beta ") is BetaActorCriticPolicy, "⑧ 大小写/空白被规整（' Beta ' → beta）")
    _bad = 0
    for bad in ("gaussian", "normal", "tanh", "off"):
        try:
            policy_for(bad)
        except ValueError:
            _bad += 1
    ok(_bad == 4, f"⑨ 非法 act_dist 一律 fail-fast（不静默回落高斯）·4/4 得 {_bad}")


def test_beta_policy_build():
    print("\n── B2 Beta 策略：optimizer 重建 / log_std 真删 / save-load ──")
    import gymnasium as gym
    from stable_baselines3 import PPO

    LOW = np.array([-0.048, -0.018], dtype=np.float32)

    class E(gym.Env):
        observation_space = gym.spaces.Box(-1, 1, (27,), np.float32)
        action_space = gym.spaces.Box(LOW, -LOW, dtype=np.float32)

        def reset(self, **k):
            return np.zeros(27, np.float32), {}

        def step(self, a):
            assert np.all(a >= LOW - 1e-9) and np.all(a <= -LOW + 1e-9), f"越箱 {a}"
            return np.zeros(27, np.float32), 0.0, False, True, {}

    m = PPO(BetaActorCriticPolicy, E(), n_steps=64, batch_size=32, seed=0, verbose=0)
    p = m.policy
    ok(not hasattr(p, "log_std"), "① 高斯的 log_std 已删（Beta 无此参数·留着=幽灵参数进 state_dict）")
    ok("log_std" not in p.state_dict(), "② state_dict 里也没有 log_std")
    opt_ids = {id(q) for g in p.optimizer.param_groups for q in g["params"]}
    new_ids = {id(q) for q in p.action_net.parameters()}
    ok(new_ids <= opt_ids, "③ 新动作头的参数【在】optimizer 里（不重建则它永远不更新=静默学不动）")
    ok(len(opt_ids) == len(list(p.parameters())), "④ optimizer 参数数 == policy 参数数（无遗漏/无幽灵）")

    m.learn(128)                                            # 真训几步：走通 rollout + 反传
    ok(True, "⑤ 真训练 128 步跑通（rollout/log_prob/entropy/反传全链路）")

    tmp = os.path.join(tempfile.mkdtemp(), "b.zip")
    m.save(tmp)
    m2 = PPO.load(tmp, device="cpu")
    o = np.random.RandomState(0).randn(256, 27).astype(np.float32)
    a1 = m.predict(o, deterministic=True)[0]
    a2 = m2.predict(o, deterministic=True)[0]
    ok(np.array_equal(a1, a2), "⑥ save/load 往返后确定性动作逐位一致")
    ok(type(m2.policy.action_dist).__name__ == "BetaDistribution", "⑦ 裸 PPO.load 能解析出 BetaDistribution（模块可导入）")
    ok(int(np.sum(np.abs(a1) > -LOW + 1e-9)) == 0, "⑧ 确定性动作零越箱")


def test_gw_entry_mechanism():
    print("\n── B3 让路入口档：开局即让路态势能不能进 ──")
    starts = _find_starts(6, seed=0)
    ok(len(starts) == 6, f"① 搜到 6 组『开局即 crossing 让路态势且未到紧急』的初始状态（得 {len(starts)}）")
    ok(all(not persistent_crossing(e, o) for e, o in starts),
       "② 这些状态上 persistent_crossing 恒 False（因为它要求 ¬X(now)·而 X(now) 已真）")

    n_paper = n_sym = 0
    for e, o in starts:
        for tag, sc in (("paper", ColregsStatechart(gw_entry="paper")),
                        ("symmetric", ColregsStatechart(gw_entry="symmetric"))):
            ee, oo = e, o
            hit = 0
            for _t in range(80):
                if sc.step(ee, oo) in _GW:
                    hit += 1
                ee, oo = _adv(ee), _adv(oo)
            if tag == "paper":
                n_paper += (hit > 0)
            else:
                n_sym += (hit > 0)
    ok(n_paper == 0, f"③ 'paper' 档：6/6 全程进不去让路态（进得去的有 {n_paper} 组·应为 0）")
    ok(n_sym == 6, f"④ 'symmetric' 档：6/6 全部进得去让路态（进得去的有 {n_sym} 组·应为 6）")

    # 态势由无到有时，两档都该进（symmetric 不改这条路径）
    both = 0
    for e, o in starts:
        back = lambda s, n: VesselState(s.position - s.velocity * DT * n * np.array(
            [np.cos(s.orientation), np.sin(s.orientation)]), s.orientation, s.velocity, _L)
        hits = []
        for g in ("paper", "symmetric"):
            sc = ColregsStatechart(gw_entry=g)
            ee, oo = back(e, 40), back(o, 40)
            h = 0
            for _t in range(120):
                if sc.step(ee, oo) in _GW:
                    h += 1
                ee, oo = _adv(ee), _adv(oo)
            hits.append(h)
        both += (hits[0] > 0 and hits[1] > 0)
    ok(both == 6, f"⑤ 态势【由无到有】时两档都能进（6/6·得 {both}）—— symmetric 不动这条既有路径")


# ════════════════════════════ C. 防静默 ════════════════════════════
def test_fail_fast():
    print("\n── C 防静默：非法取值 / 跨档热启动 / 训评同档 ──")
    for bad in ("Paper2", "sym", "", "off"):
        try:
            ColregsStatechart(gw_entry=bad)
            r = False
        except ValueError:
            r = True
        ok(r or bad == "", f"① ColregsStatechart(gw_entry={bad!r}) fail-fast")
        if bad == "":
            ok(ColregsStatechart(gw_entry="").gw_entry == "paper", "①b 空串规整成默认 'paper'（不是崩）")
    ok(set(GW_ENTRY_CHOICES) == {"paper", "symmetric"}, "② GW_ENTRY_CHOICES 就这两档")
    ok(set(ACT_DIST_CHOICES) == {"gauss", "beta"}, "③ ACT_DIST_CHOICES 就这两档")

    # run_step4e 的 sidecar 回读：训练什么就部署什么
    import importlib
    os.environ.pop("STEP4E_GW_ENTRY_FORCE", None)
    os.environ["STEP4E_GW_ENTRY"] = "paper"
    r4 = importlib.import_module("run_step4e")
    importlib.reload(r4)
    d = tempfile.mkdtemp()
    base = os.path.join(d, "arm")
    json.dump({"config_sig": {"gw_entry": "symmetric"}}, open(base + ".progress.json", "w"))
    ok(r4._read_gw_entry(base) == "symmetric",
       "④ 评估端从 sidecar 回读到 'symmetric'（当前环境变量是 'paper' 也不影响）= 训练什么就部署什么")
    base2 = os.path.join(d, "old")
    json.dump({"config_sig": {"kind": "continuous"}}, open(base2 + ".progress.json", "w"))
    ok(r4._read_gw_entry(base2) is None, "⑤ 老存档（sidecar 无该键）→ None → 调用方回落 'paper'（它训练时的真实档）")
    ok(r4._read_gw_entry(os.path.join(d, "nope")) is None, "⑥ 没有 sidecar → None（不崩）")

    # 热启动那道闸：白名单 / 本 run 探针 / config_sig **三处**都要有这几个键，缺任一处闸就空转（L229-F 的形状）
    _src = open(os.path.join(os.path.dirname(_HERE), "run_step4e.py"), encoding="utf-8").read()
    _whitelist = _src[_src.index("_SEMANTIC_KEYS = ("):_src.index("影响【策略语义/环境动力学】的键")]
    _probe = _src[_src.index("_cur_sig_probe = dict("):_src.index("_mism = [")]
    _i = _src.index('_config_sig = {"kind": "continuous"')     # 🔴 必须从连续臂那份起找（离散臂另有一份·直接 index 会命中前面那个 → 切片反向 → 假 FAIL）
    _sig = _src[_i:_src.index('"lr_anneal_end": _LR_ANNEAL_END', _i)]
    for k in ("act_dist", "gw_entry", "ctrl_slew_frac", "ctrl_lowpass_alpha"):
        ok(k in _whitelist and k in _probe and k in _sig,
           f"⑦ 热启动闸三处齐全: {k}（白名单 {k in _whitelist} / 本run探针 {k in _probe} / config_sig {k in _sig}）")


# ════════════════════════════ D. 前瞻一致性 ════════════════════════════
def test_projection_lookahead_same_grade():
    print("\n── D 前瞻用的临时状态机与在役状态机同档 ──")
    from trb_env.usv_projection import ContinuousColregsProjection
    for g in ("paper", "symmetric"):
        pr = ContinuousColregsProjection(0.24, 0.03, gw_entry=g)
        ok(pr._sc.gw_entry == g, f"① gw_entry={g!r} 传到了投影盾自建的状态机")
    src = open(os.path.join(os.path.dirname(_HERE), "trb_env", "usv_projection.py"), encoding="utf-8").read()
    ok(src.count("gw_entry=self._sc.gw_entry") == 2,
       "② 两处终端可行性前瞻的临时状态机都显式继承在役档（否则前瞻与真实转移不一致）")
    # 显式传入 statechart 时以那个实例为准（单一真相源·不被 gw_entry 形参覆盖）
    sc = ColregsStatechart(gw_entry="symmetric")
    pr = ContinuousColregsProjection(0.24, 0.03, statechart=sc, gw_entry="paper")
    ok(pr._sc is sc and pr._sc.gw_entry == "symmetric", "③ 显式传 statechart 时以该实例自己的档为准（不被形参悄悄改掉）")


if __name__ == "__main__":
    test_bit_identical()
    test_beta_mechanism()
    test_beta_policy_build()
    test_gw_entry_mechanism()
    test_fail_fast()
    test_projection_lookahead_same_grade()
    print(f"\n{'✅ 全部 PASS' if _FAIL == 0 else '❌ 有 FAIL'}：{_PASS} passed / {_FAIL} failed")
    sys.exit(1 if _FAIL else 0)
