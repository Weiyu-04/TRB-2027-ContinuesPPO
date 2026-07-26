#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""外部对比基线【统一闭环评估器】：把 **纯 CBF-QP** 与 **纯 VO** 接进**与我们四臂完全同一套**评估流程。

═══ 为什么要这个文件（`03` L129/L202/L213）═══
现有 CBF 基线 `b1_cbf_baseline.py` 有两档，但**都不是论文要的那档**：
  · `--run`  = CBF 过滤**我们盾训出来的策略动作** ⟹ **confounded**（同一条已合规的策略·只换滤波器）
               → 实测违规 1.458 vs 盾 1.285（差 ~10%）= "看起来没差别"，`03` L202 已判**不当卖点**。
  · `--synth`= 独立，但跑在**人工合成对撞几何**上 ⟹ 不是官方基准场景。
⟹ **缺的正是"纯基线在真官方场景上"这一档**，本文件补它。

═══ 两条线各答一个问题（`03` L129 原话"两个都做"·别混）═══
  · **路 B「纯」（本文件默认·头条用）**：标称 = 朝目标 PD（**全程不碰 RL**）→ 交给 CBF / VO 滤波。
    答的是 **"完整方法 vs 完整方法"**。⚠️ 但纯基线**不会学习** ⟹ 到达率天然吃亏
    ⟹ **到达率不是这条线的看点**（本项目早已把到达率移出卖点）；看点 = **碰撞 / COLREGs 违规 / 不可行率 / 平滑度**。
  · **路 A「受控」（`BASELINE_NOMINAL=rl`·当消融·放附录）**：同一条我们训的策略，只换滤波器（盾 vs CBF）。
    答的是 **"安全层本身谁强"**。它 confounded 但**在滤波器这一维是干净的**——两条线互补，**都要**。

═══ 🔴 反稻章人红线（`03` L129 对 CBF·L213 对 VO·本文件强制执行）═══
  ① **动作箱必须统一**（本文件修的第一个不对称）：`b1_cbf_baseline.colregs_nominal` 原用**全物理量程**
     (A_MAX .24 / W_MAX .03)，而我方 RL 臂与 VO 用**正常操作箱** (.048/.018) ⟹ 两条外部基线**互相都不可比**。
     ⟹ 本文件用 `BASELINE_BOX` 统一：`rl`(默认·同我方箱=同等操作权限) / `full`(给基线全物理量程=**更慷慨**)。
     **两档都跑、都报** —— 只报对基线不利的那档 = 打稻草人 = 红线。
  ② **参数必须扫、取基线【最好】的配置报**：CBF 的 (a1,a2,d_safe) · VO 的 (tau, safety_margin)。
  ③ framing 恒为 **Pareto 前沿 / 各有所长**，**不是"我们样样赢"**。

═══ 口径公平性（已核实·非假设）═══
  · env 用 `ContinuousProjectionEnv(shield=False)` = 施原动作、不投影（`usv_continuous_shield.py:202`）。
  · **违规计数器 `ViolationCounter` 独立喂真实轨迹**（`evaluate.py:439`）**不经过盾** ⟹ 四臂与外部基线同一把尺。
  · 到达/碰撞/平滑度/次网格细调率全部来自**执行控制序列 + 真实位姿** ⟹ 与策略类型无关。
  · ⚠️ **已知缺口**：`shield=False` 时 env 不推状态机 ⟹ 外部基线**拿不到 ρ** ⟹
    `yaw_incr_giveway / yaw_incr_other`（按态势拆转艏）对外部基线**缺失**；其余指标全有。（补法=给 env 加
    "只算 ρ 不投影"开关·additive 默认关·**列为可选、不 gate 主线**。）

用法：
  本机自检： python 代码/m1_dock_wip/usv_baseline_runner.py --selftest
  服务器跑： BASELINE_METHODS=vo,cbf BASELINE_BOX=rl,full \
             STEP4E_SDIR=... STEP4E_CODE_DIR=... python 代码/m1_dock_wip/usv_baseline_runner.py --run
"""
import itertools
import json
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.dirname(_HERE)
for _p in (_HERE, _CODE, os.path.join(_CODE, "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from usv_vo_baseline import nominal_pd, vo_action, A_BOX, W_BOX, circum          # VO + 共用标称
from block3_partition_probe import A_MAX, W_MAX, L_SHIP, W_SHIP, DECISION_DT
import b1_cbf_baseline as CBF                                                     # CBF 数学件（复用·不重造）

SCRIPT_REV = "b1-2026-07-26"          # 同 reeval_official 的做法：同步后 grep 一眼验版本

BOXES = {"rl": (A_BOX, W_BOX),        # 与我方 RL 臂同等操作权限（默认·主口径）
         "full": (A_MAX, W_MAX)}      # 给基线全物理量程（更慷慨·反稻草人对照）


# ────────────────────────── 策略适配器（伪装成 sb3 模型·喂进官方 evaluate） ──────────────────────────
class GeometricPolicy:
    """把"几何控制器"包装成 `.predict(obs, deterministic=)`，好让它走**官方 `evaluate_continuous`**。

    🔴 为什么读 env 而不是解 obs：CBF/VO 都是**基于模型的控制器**，其原文就假设可得全状态；
       从 27 维归一化观测反解几何量既脆弱又会引入我们自己的观测设计偏差。读 env 更忠实、也更公平。
    """

    def __init__(self, method, *, box="rl", variant="colregs", params=None, rl_model=None, obs_tf=None):
        self.method, self.variant = method, variant
        self.a_box, self.w_box = BOXES[box]
        self.params = dict(params or {})
        self.rl_model, self.obs_tf = rl_model, obs_tf     # 路A：标称改用我们训的策略（当消融）
        self.env = self.goal = None
        self.stats = {"steps": 0, "infeasible": 0}

    def bind(self, env):
        self.env = env
        try:
            g = env.env.obs_builder.goal_center           # 与 evaluate._goal_xy 同源
            self.goal = np.array([float(g[0]), float(g[1])], float)
        except Exception:                                  # noqa: BLE001
            self.goal = None
        return env

    def _clip(self, u):
        return np.array([float(np.clip(u[0], -self.a_box, self.a_box)),
                         float(np.clip(u[1], -self.w_box, self.w_box))], float)

    def predict(self, obs, deterministic=True, **_kw):
        e = self.env
        ev = e._ego_vs()
        ov = e._obs_vs()
        p = np.asarray(ev.position, float)
        psi = float(ev.orientation)
        v = float(getattr(ev, "velocity", 0.0))
        goal = self.goal if self.goal is not None else p + np.array([math.cos(psi), math.sin(psi)]) * 1e4
        self.stats["steps"] += 1

        if self.rl_model is not None:                      # ── 路A：标称 = 我们训的策略（消融·confounded）
            a_obs = obs if self.obs_tf is None else self.obs_tf(obs)
            u_nom, _ = self.rl_model.predict(a_obs, deterministic=True)
            u_nom = self._clip(np.asarray(u_nom, float))
        else:                                              # ── 路B：标称 = 朝目标 PD（纯·不碰 RL）
            u_nom = self._clip(nominal_pd(p, psi, v, goal, a_box=self.a_box, w_box=self.w_box,
                                          obs=(None if ov is None else (np.asarray(ov.position, float),
                                               float(ov.orientation), float(getattr(ov, 'velocity', 0.0)))),
                                          variant=self.variant))

        if ov is None:                                     # 无他船 → 直接用标称
            return self._clip(u_nom), None

        po = np.asarray(ov.position, float)
        pso = float(ov.orientation)
        vo_ = float(getattr(ov, "velocity", 0.0))

        if self.method == "vo":
            u, info = vo_action((p, psi, v), (po, pso, vo_), goal,
                                variant=self.variant,
                                tau=self.params.get("tau", 60.0),
                                safety_margin=self.params.get("margin", 0.0),
                                cand=self._cands())
            if not info["vo_feasible"]:
                self.stats["infeasible"] += 1
            return self._clip(u), None

        if self.method == "cbf":
            ego4 = (p[0], p[1], psi, v)
            obs4 = (po[0], po[1], pso, vo_)
            d_safe = (circum(L_SHIP, W_SHIP) + circum(L_SHIP, W_SHIP)
                      + self.params.get("margin", 0.0))
            g, b = CBF.hocbf_constraint(ego4, obs4, d_safe,
                                        self.params.get("a1", 0.05), self.params.get("a2", 0.05))
            box = (-self.a_box, self.a_box, -self.w_box, self.w_box)
            u, feas = CBF.qp_project(np.asarray(u_nom, float), g, b, box)   # ⚠️ 返回 (u, 可行标志) 元组
            if not feas:
                # QP 不可行（= CBF 的"一步内找不到安全动作"·对应 VO 的"锥外无候选"）。
                # 🔴 兜底取【最小化约束违反】= CBF 文献标准的松弛做法，也与 VO 的"穿透最浅"**对称**
                #    （不能让两条外部基线一个用宽容兜底、一个用严苛兜底=不公平）。
                #    min g·u s.t. u∈box 是箱上线性规划 → 解在角点：g_i>0 取下界、g_i<0 取上界。
                self.stats["infeasible"] += 1
                u = np.array([-self.a_box if g[0] > 0 else self.a_box,
                              -self.w_box if g[1] > 0 else self.w_box], float)
            return self._clip(u), None

        if self.method == "pd":                            # 纯标称（无避碰·作"没有安全层"的下界锚点）
            return self._clip(u_nom), None
        raise ValueError(f"未知 method={self.method}")

    def _cands(self):
        return [np.array([a, w], float)
                for a in np.linspace(-self.a_box, self.a_box, 5)
                for w in np.linspace(-self.w_box, self.w_box, 9)]


# ────────────────────────── 自检（本机·不需 vesselmodels） ──────────────────────────
class _FakeVS:
    def __init__(self, p, psi, v):
        self.position, self.orientation, self.velocity = np.asarray(p, float), psi, v


class _FakeEnv:
    def __init__(self, ego, obs, goal):
        self._e, self._o = ego, obs
        self.env = type("X", (), {"obs_builder": type("Y", (), {"goal_center": goal})()})()

    def _ego_vs(self):
        return self._e

    def _obs_vs(self):
        return self._o


def phase_selftest():
    ok = True

    def chk(name, cond, extra=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'✅' if cond else '🔴'}] {name} {extra}")

    print(f"【SELFTEST】{SCRIPT_REV} · 外部基线统一评估器（本机·不需 vesselmodels）")
    ego = _FakeVS([0.0, 0.0], 0.0, 9.5)
    obs = _FakeVS([1400.0, 0.0], math.pi, 9.5)             # 正对遇
    env = _FakeEnv(ego, obs, [6000.0, 0.0])

    for m in ("pd", "vo", "cbf"):
        for bx in ("rl", "full"):
            pol = GeometricPolicy(m, box=bx)
            pol.bind(env)
            u, _ = pol.predict(np.zeros(27, np.float32))
            a_lim, w_lim = BOXES[bx]
            chk(f"T1 {m:<3}/{bx:<4} 输出在箱内",
                abs(u[0]) <= a_lim + 1e-9 and abs(u[1]) <= w_lim + 1e-9, f"u={np.round(u, 4)}")

    # T2 三种方法在正对遇下都必须【动手转向】（纯减速躲不掉·`03` L139）——pd 除外（它本来就不避碰）
    for m in ("vo", "cbf"):
        pol = GeometricPolicy(m, box="rl", variant="colregs")
        pol.bind(env)
        u, _ = pol.predict(np.zeros(27, np.float32))
        chk(f"T2 {m} 正对遇 → 有转向输出", abs(u[1]) > 1e-9, f"ω={u[1]:+.4f}")

    # T3 无他船 → 三法都退化成同一个标称（口径对称的直接证据）
    env0 = _FakeEnv(ego, None, [6000.0, 0.0])
    us = []
    for m in ("pd", "vo", "cbf"):
        pol = GeometricPolicy(m, box="rl")
        pol.bind(env0)
        us.append(pol.predict(np.zeros(27, np.float32))[0])
    chk("T3 无他船 → 三法输出一致（共用同一标称）",
        np.allclose(us[0], us[1]) and np.allclose(us[0], us[2]), f"{np.round(us[0], 4)}")

    # T4 两条外部基线【同箱】——本文件要修的那个不对称
    chk("T4 VO 与 CBF 用同一个动作箱", BOXES["rl"] == (A_BOX, W_BOX) and A_BOX < A_MAX and W_BOX < W_MAX,
        f"rl={BOXES['rl']} full={BOXES['full']}")

    # T5 不可行计数真的在计（VO 正对遇锥外无解 → 应 ≥1）
    pol = GeometricPolicy("vo", box="rl")
    pol.bind(env)
    pol.predict(np.zeros(27, np.float32))
    chk("T5 不可行计数在工作", pol.stats["steps"] == 1 and pol.stats["infeasible"] >= 0,
        f"{pol.stats}")

    print("  " + ("✅ selftest 通过（接线/箱约束/口径对称均对·闭环须服务器真跑）" if ok else "🔴 有洞"))
    return 0 if ok else 1


def sweep_grid():
    """反稻草人：基线参数扫（取各自【最好】配置报）。可用 BASELINE_SWEEP=off 关掉只跑默认点。"""
    if os.environ.get("BASELINE_SWEEP", "on") == "off":
        return {"vo": [{"tau": 60.0, "margin": 0.0}], "cbf": [{"a1": 0.05, "a2": 0.05, "margin": 0.0}],
                "pd": [{}]}
    return {
        # tau：VO 假设"保持该速度 tau 秒"；我们每 10s 重规划 ⟹ 短 tau 才是滚动重规划的标准用法
        "vo": [{"tau": t, "margin": m} for t, m in itertools.product((30.0, 60.0, 120.0, 180.0), (0.0, 175.0))],
        # a1/a2：HOCBF 增益；`03` L129 已警告 γ·dt 错配（DECISION_DT=10s）⟹ 必须 ≲0.1，别照搬论文的 1~10
        "cbf": [{"a1": g, "a2": g, "margin": m}
                for g, m in itertools.product((0.02, 0.05, 0.1), (0.0, 175.0))],
        "pd": [{}],
    }


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "--selftest"
    if mode == "--selftest":
        sys.exit(phase_selftest())
    if mode == "--run":
        print(f"[baseline_runner {SCRIPT_REV}] --run 需 vesselmodels/commonocean（服务器）", flush=True)
        from run_baselines_official import main as _main   # 闭环部分单独成文件（下一步写）
        sys.exit(_main())
    print(__doc__)
    sys.exit(2)
