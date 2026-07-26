#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""外部基线 #2 · **速度障碍（Velocity Obstacle, VO）** 反应式避碰（`03` L129 选型定案·奠基 Fiorini-Shiller IJRR 1998）。

【为什么选它】`03` L129：VO 是离 CBF / 离我们**都最远的第三范式**（反应式几何·无优化无学习）⟹ 互补性最强；
  海事有真部署背景（Kuwata IEEE JOE 2014·NASA JPL 真 USV）。**不选 MPC 当第二**（对抗验证实测 MPC 两处致命坑·稻草人风险 HIGH）。

【本实现的形态（诚实口径·写作必带）】
  · **非反向 VO（不用 RVO/ORCA 的互惠假设）**：本基准的他船是**录制轨迹、不反应** ⟹ 互惠项没有物理依据，
    用 plain VO 才忠实。（RVO/ORCA 的互惠是"双方都在避让"的场景假设。）
  · **10s 零阶保持 + 动作箱受限** ⟹ 一步内可达的速度集很小 ⟹ 实现为
    **"在一步可达的 (a, ω) 候选里挑离标称最近、且不落进速度障碍锥的那个"**；全落锥内则挑**穿透最浅**的（兜底）。
    这是 VO 在"低机动性大船 + 长决策步"下的**忠实落地**，不是简化偷懒——连续时间 VO 的"任选锥外速度"
    在这里物理上做不到（一步转不过来）。**须在论文如实写这条适配。**
  · **COLREGs 变体**（`variant='colregs'`）：让路态下**对右转候选加偏好权重**（不是硬约束）——
    与 CBF 基线的 `colregs_nominal` 同一思路，保证两条外部基线口径对称、不厚此薄彼。
  · ⚠️ **这是【代表性实现】·不是某篇已发表 VO-COLREGs 公式的复现**。写作时要么引已发表公式并复验，
    要么明写"representative implementation"。**绝不 claim "我们比 VO 强 X%"当卖点**（打稻草人=红线）。

【🔴🔴 反稻草人红线（闭环跑之前必须先做·否则结论作废）】
  自检 T4 已暴露：正对遇 1400m、10s 保持下，**VO 一个"锥外候选"都没有**（n_free=0）→ 直接跑就会
  得到"VO 到处不可行/撞"的难看数字。但那里面**掺着参数没调好的成分**，不全是方法本身的缺陷：
  · **视界 tau 是敏感参数**：VO 假设"保持这个速度 tau 秒"，而我们的船每 10s 重规划一次 ⟹ tau 取 180s
    （=我方 `is_emergency` 的可达集视界）对 VO 是**偏悲观**的用法；滚动重规划下取更短的 tau 是标准做法。
  · **安全裕度 `safety_margin` 同理**：我方盾把他船当 440m 圆盘（含 Krasowski Table II 的 350m 裕度），
    但那是**我们**的口径；强加给 VO 就是替它绑手。
  ⟹ **闭环评估必须【扫 tau × safety_margin】并报 VO 的【最好配置】**（`03` L129 对 CBF 的 γ·dt 已立同款要求：
     "不修=假稻草人结论"）。**只报单点、且那个点恰好让 VO 难看 = 打稻草人 = 项目红线。**
  ⟹ 论文写法恒为 **Pareto 前沿 / 各有所长**，不是"我们样样赢"（`03` L129 定的 framing）。

【纯 numpy·不 import vesselmodels】→ 本机可 selftest（吸取"能本机验的就本机验"教训）。
用法：python 代码/m1_dock_wip/usv_vo_baseline.py --selftest
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from block3_partition_probe import A_MAX, W_MAX, V_MAX, DECISION_DT, L_SHIP, W_SHIP   # 单一真相源

DT = DECISION_DT
# RL 动作箱（与四臂同口径·`03` L203 metrics_subgrid 同源）：外部基线也只准用这个箱，否则不公平
A_BOX, W_BOX = 0.048, 0.018


def circum(l, w):
    """外接圆半径（与盾/CBF 基线同口径）。"""
    return 0.5 * math.hypot(l, w)


# ────────────────────────── 标称控制器（目标导引 PD·两条外部基线共用） ──────────────────────────
def is_giveway_like(p, psi, obs, *, sector_deg=60.0, rng_max=2500.0):
    """粗略让路态判据（他船在前方扇区 ∧ 正在接近 ∧ 近场）——与 `b1_cbf_baseline.colregs_nominal` 同款条件。

    ⚠️ 这是**外部基线自带的简易判据**，不是我们的 `ColregsStatechart`。故意如此：外部基线是独立方法，
       不该借用我们的状态机（借了就不"独立"了）。两条外部基线用**同一个**判据 ⟹ 彼此对称。
    """
    if obs is None:
        return False
    p_o, psi_o, v_o = np.asarray(obs[0], float), float(obs[1]), float(obs[2])
    p_rel = p_o - np.asarray(p, float)
    rng = float(np.linalg.norm(p_rel))
    beta = (math.atan2(p_rel[1], p_rel[0]) - psi + math.pi) % (2 * math.pi) - math.pi
    return abs(beta) < math.radians(sector_deg) and rng < rng_max


def nominal_pd(p, psi, v, goal, *, v_des=None, k_psi=0.45, k_v=0.35,
               obs=None, variant="plain", a_box=None, w_box=None):
    """朝目标的 PD 标称控制 u=(a, ω)，输出**夹进动作箱**。**两条外部基线共用** ⟹ 标称一致 = 公平。

    · 航向：ω = clip(k_psi · wrap(θ_goal − ψ) / DT, ±w_box) —— 除 DT 是因为 ω 是角**速度**、一步作用 DT 秒。
    · 速度：接近目标时降速（避免冲过头——本项目"临门一脚"老毛病），远场巡航 v_des。
    · `variant='colregs'` + 让路态 → **标称偏满右转（starboard）**。
      🔴 这条必须在**共用标称**里，不能只给某一条基线：
         `b1_cbf_baseline.colregs_nominal` 原本就是靠这个偏置**修 CBF 正对遇退化**
         （正对遇时 HOCBF 的 ω 系数恒等于 0 ⟹ CBF 自己只会减速、不会转·`03` L200-F / L205-补2）。
         我第一版把标称统一成纯 PD，**等于把 CBF 的这个修复删掉了**，被自检 T2 抓出。
    **它不做任何避碰**——避碰交给外面的 VO / CBF 滤波器。
    """
    a_box = A_BOX if a_box is None else a_box
    w_box = W_BOX if w_box is None else w_box
    d = np.asarray(goal, float) - np.asarray(p, float)
    dist = float(np.linalg.norm(d))
    if v_des is None:
        v_des = min(V_MAX, max(1.5, 0.06 * dist))          # 近目标自动降速（600m→~9m/s·100m→1.5m/s）
    psi_goal = math.atan2(d[1], d[0]) if dist > 1e-9 else psi
    e_psi = (psi_goal - psi + math.pi) % (2 * math.pi) - math.pi
    omega = float(np.clip(k_psi * e_psi / DT, -w_box, w_box))
    accel = float(np.clip(k_v * (v_des - v) / DT, -a_box, a_box))
    if variant == "colregs" and is_giveway_like(p, psi, obs):
        omega = -w_box                                      # 满右转标称（starboard·同 colregs_nominal 口径）
    return np.array([accel, omega], float)


# ────────────────────────── 速度障碍核心 ──────────────────────────
def in_velocity_obstacle(p_rel, v_rel, R, tau):
    """是否落在速度障碍锥内（视界 tau 秒）。判据 = ‖p_rel + v_rel·t‖ ≤ R 在 t ∈ (0, tau] 上有解。

    🔴 **符号约定（两个都必须是"他船相对本船"·搞反会把正对撞判成安全）**：
        `p_rel = p_obs − p_ego`  ·  `v_rel = v_obs − v_ego`
    自检 T1a 就是钉这条的（正对撞必须判"在锥内"）——**首版把 v_rel 写成 v_ego − v_obs，
    结果正对遇被判成不会撞、VO 完全不动手，被 T1/T4 当场抓出**。
    返回 (是否在锥内, 最早碰撞时刻 or inf, 视界内最近距离)。
    """
    p = np.asarray(p_rel, float)
    v = np.asarray(v_rel, float)
    a = float(v @ v)
    b = 2.0 * float(p @ v)
    c = float(p @ p) - R * R
    if c <= 0.0:                                            # 已经重叠
        return True, 0.0, math.sqrt(max(float(p @ p), 0.0))
    if a <= 1e-12:                                          # 相对静止
        return False, math.inf, math.sqrt(float(p @ p))
    disc = b * b - 4 * a * c
    t_star = max(0.0, min(tau, -b / (2 * a)))               # 视界内的最近点时刻
    d_min = float(np.linalg.norm(p + v * t_star))
    if disc <= 0.0:
        return False, math.inf, d_min
    sq = math.sqrt(disc)
    t1 = (-b - sq) / (2 * a)
    t2 = (-b + sq) / (2 * a)
    if t2 <= 0.0 or t1 > tau:                               # 碰撞时刻在过去、或在视界之外
        return False, math.inf, d_min
    return True, max(t1, 0.0), d_min


def _step_state(p, psi, v, u, T=DT, sub=10):
    """用与盾/CBF 基线同款的子步积分推一步（纯运动学·不 import 官方动力学=本机可跑）。"""
    p = np.asarray(p, float).copy()
    dt = T / sub
    for _ in range(sub):
        v = float(np.clip(v + u[0] * dt, 0.0, V_MAX))
        psi = psi + u[1] * dt
        p = p + np.array([math.cos(psi), math.sin(psi)]) * v * dt
    return p, psi, v


def _candidates(n_a=5, n_w=9):
    """一步可达的动作候选网格（在 RL 动作箱内·两条外部基线同箱=公平）。"""
    return [np.array([a, w], float)
            for a in np.linspace(-A_BOX, A_BOX, n_a)
            for w in np.linspace(-W_BOX, W_BOX, n_w)]


def vo_action(ego, obs, goal, *, variant="colregs", tau=180.0, safety_margin=0.0,
              starboard_bonus=0.35, cand=None):
    """VO 反应式避碰的一步动作。

    ego / obs = (p, psi, v)；goal = 目标点。返回 (u, info)。
    · 标称 = `nominal_pd`（朝目标）。
    · 对每个一步可达候选：推一步得到**新速度矢量**，判它是否落进速度障碍锥（视界 tau）。
    · 锥外候选里挑"离标称最近"（COLREGs 变体：让路态给右转候选打折=偏好）；**全在锥内则挑穿透最浅的**。
    · `tau=180s` 与本项目 `is_emergency` 的可达集视界同源（Krasowski Table II）⟹ 与我们同一时间尺度。
    """
    p_e, psi_e, v_e = np.asarray(ego[0], float), float(ego[1]), float(ego[2])
    u_nom = nominal_pd(p_e, psi_e, v_e, goal, obs=obs, variant=variant)
    info = {"vo_feasible": True, "n_free": 0, "fallback": False}
    if obs is None:
        return u_nom, info
    p_o, psi_o, v_o = np.asarray(obs[0], float), float(obs[1]), float(obs[2])
    vel_o = np.array([math.cos(psi_o), math.sin(psi_o)]) * v_o
    R = circum(L_SHIP, W_SHIP) + circum(L_SHIP, W_SHIP) + safety_margin   # 两船外接圆和（+可选裕度）
    p_rel = p_o - p_e

    # COLREGs 偏好方向：他船在右舷前方（让路态）→ 偏好右转（ω<0 = starboard·与项目符号约定一致）
    prefer_stbd = False
    if variant == "colregs":
        beta = (math.atan2(p_rel[1], p_rel[0]) - psi_e + math.pi) % (2 * math.pi) - math.pi
        prefer_stbd = (-math.pi / 2 <= beta <= math.pi * 5 / 8)       # 右舷~正前的宽扇区=让路侧

    best, best_cost, deepest, deep_cost = None, math.inf, None, math.inf
    for u in (cand if cand is not None else _candidates()):
        p1, psi1, v1 = _step_state(p_e, psi_e, v_e, u)
        vel_e1 = np.array([math.cos(psi1), math.sin(psi1)]) * v1
        hit, _t, d_min = in_velocity_obstacle(p_rel, vel_o - vel_e1, R, tau)   # 🔴 他船相对本船（见函数 docstring）
        dev = float(np.linalg.norm((u - u_nom) / np.array([A_BOX, W_BOX])))   # 归一化偏离标称
        if prefer_stbd and u[1] < 0:
            dev *= (1.0 - starboard_bonus)                                     # 右转候选打折=偏好
        if not hit:
            info["n_free"] += 1
            if dev < best_cost:
                best, best_cost = u, dev
        else:
            pen = (R - d_min) + 0.05 * dev                                     # 穿透深度（越小越好）
            if pen < deep_cost:
                deepest, deep_cost = u, pen
    if best is not None:
        return best, info
    info.update(vo_feasible=False, fallback=True)          # 锥外无解 = VO 的"不可行"（对应 CBF 的 QP 无解）
    return (deepest if deepest is not None else u_nom), info


# ────────────────────────── 自检（本机·纯 numpy） ──────────────────────────
def phase_selftest():
    ok = True

    def chk(name, cond, extra=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'✅' if cond else '🔴'}] {name} {extra}")

    print("【SELFTEST】VO 基线（纯 numpy·不需 vesselmodels）")

    # T1 VO 判据：迎头对撞必落锥内；侧向远离必在锥外
    hit, t, _d = in_velocity_obstacle(np.array([1000.0, 0.0]), np.array([-10.0, 0.0]), 200.0, 180.0)
    chk("T1a 正对撞 → 在锥内", hit and 0 < t < 180, f"(t={t:.1f}s)")
    hit2, _t2, _d2 = in_velocity_obstacle(np.array([1000.0, 0.0]), np.array([0.0, 10.0]), 200.0, 180.0)
    chk("T1b 纯横向掠过 → 不在锥内", not hit2)
    hit3, _t3, _d3 = in_velocity_obstacle(np.array([1000.0, 0.0]), np.array([10.0, 0.0]), 200.0, 180.0)
    chk("T1c 相对远离 → 不在锥内", not hit3)

    # T2 标称控制器：输出必在动作箱内，且朝目标转
    u = nominal_pd(np.array([0.0, 0.0]), 0.0, 5.0, np.array([1000.0, 1000.0]))
    chk("T2a 标称在动作箱内", abs(u[0]) <= A_BOX + 1e-9 and abs(u[1]) <= W_BOX + 1e-9, f"u={u}")
    chk("T2b 目标在左前 → 左转(ω>0)", u[1] > 0)
    u2 = nominal_pd(np.array([0.0, 0.0]), 0.0, 5.0, np.array([1000.0, -1000.0]))
    chk("T2c 目标在右前 → 右转(ω<0)", u2[1] < 0)

    # T3 无他船 → 退化成标称
    u3, i3 = vo_action((np.array([0.0, 0.0]), 0.0, 5.0), None, np.array([2000.0, 0.0]))
    chk("T3 无他船 → 等于标称", np.allclose(u3, nominal_pd(np.array([0., 0.]), 0.0, 5.0, np.array([2000., 0.]))) and i3["vo_feasible"])

    # T4 正对遇：VO 必须选出【非零转向】（纯减速躲不掉——`03` L139 已证 head-on 减速反有害）
    ego = (np.array([0.0, 0.0]), 0.0, 9.5)
    obs = (np.array([1400.0, 0.0]), math.pi, 9.5)
    u4, i4 = vo_action(ego, obs, np.array([6000.0, 0.0]), variant="colregs")
    chk("T4a 正对遇 → 输出转向而非只减速", abs(u4[1]) > 1e-6, f"u={u4} 锥外候选 {i4['n_free']}")
    chk("T4b COLREGs 变体 → 右转(负ω·starboard)", u4[1] < 0, f"ω={u4[1]:+.4f}")

    # T5 动作箱红线：任何输出都不许越箱（越箱=对外部基线放水=不公平）
    rng = np.random.default_rng(0)
    bad = 0
    for _ in range(200):
        e = (rng.uniform(-3000, 3000, 2), rng.uniform(-math.pi, math.pi), rng.uniform(0, V_MAX))
        o = (rng.uniform(-3000, 3000, 2), rng.uniform(-math.pi, math.pi), rng.uniform(0, V_MAX))
        uu, _ = vo_action(e, o, rng.uniform(-6000, 6000, 2))
        if abs(uu[0]) > A_BOX + 1e-9 or abs(uu[1]) > W_BOX + 1e-9:
            bad += 1
    chk("T5 200 随机态 → 输出全在动作箱内", bad == 0, f"越箱 {bad}")

    # T6 不可行兜底：他船贴脸包围时必须仍返回一个合法动作（不许崩、不许 None）
    u6, i6 = vo_action((np.array([0.0, 0.0]), 0.0, 9.5), (np.array([150.0, 0.0]), math.pi, 9.5),
                       np.array([3000.0, 0.0]))
    chk("T6 贴脸态 → 仍给出合法动作 + 标记不可行",
        u6 is not None and abs(u6[0]) <= A_BOX + 1e-9 and (not i6["vo_feasible"] or i6["n_free"] >= 0),
        f"fallback={i6['fallback']}")

    print("  " + ("✅ selftest 通过（几何/箱约束/兜底均对·闭环行为须服务器真跑）" if ok else "🔴 有洞"))
    return 0 if ok else 1


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "--selftest"
    if mode == "--selftest":
        sys.exit(phase_selftest())
    print(__doc__)
    print("用法: python usv_vo_baseline.py --selftest   （闭环评估走 usv_baseline_runner.py）")
    sys.exit(2)
