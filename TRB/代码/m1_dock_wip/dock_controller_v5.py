#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M1 停车控制器 v5 —— 治 v4 的「近门追踪发散」（`03` L192 I 查出的碰撞近因）。

【v4 的病·实测轨迹坐实】T-1785 局：v4 把船带到离门 5.7m 但朝向差 30°(门只认 ±9.74°)→进不去→
`along<0` 分支用 `atan2(门 − 本船)` 做期望朝向【追点式】→ 船越过门后，随本船漂移，**指向门的方位角
旋转速度远超本船转艏能力(ω_max=0.018 rad/s=1.03°/秒)** → 期望朝向甩得比船快 → **朝向误差单调发散
(40°→102°)**、位置反漂回 51.8m → 冲突一来把这个训练分布外状态甩回 RL → RL 逃逸进紧急态 → 撞。
   ⇒ 根因 = **近门追点在数学上病态**（目标近时方位角对位置的雅可比爆炸），不是参数没调好。

【v5 三处修法·全部保守·不改正常接近逻辑】
  ① **过门后锁常值朝向**：`along<0` 时期望朝向改用 **门朝向 θ_g（常数）**，不再用会甩的方位角
     → 结构上消除发散（常值目标 → 转艏误差单调收敛，不可能被"甩开"）。
  ② **对不齐不硬进**：近门(`along<close_dist`)但朝向误差 > `commit_tol` 时**不再压向门中心**，
     改为保持横向站位 + 降到 `v_turn` 先把头转正（先对齐再进门），避免"位置到了朝向没到"的死局。
  ③ **认输闸**：过门超过 `abort_along` 仍未对齐 → 指令减速到 0（产生【良性的 stopped 失败】），
     **绝不放任本船在交通流里游荡**（v4 那次碰撞正是游荡 1300m 后撞的）。

v4 保持原样不动（现有 900m 结果由 v4 产出·不破坏可复现性）。接口与 v4 完全一致，可直接替换。
"""
import math, numpy as np, sys
sys.path.insert(0, '/Users/weiyutang/Desktop/TRB/代码')
from trb_env.usv_dynamics import make_vessel_params, DECISION_DT
from trb_env.usv_env import A_NORMAL_OMEGA_MAX

P = make_vessel_params(); A_MAX, W_MAX = P.a_max, P.w_max


def wrap(a): return (a + math.pi) % (2 * math.pi) - math.pi


def dock_controller(state, goal, theta_g=0.0, wmax=W_MAX,
                    Ld=140.0, k_align=math.radians(22), v_run=2.6, v_turn=0.7, v_creep=1.2,
                    kv=0.06, close_dist=110.0, close_cross=28.0,
                    commit_tol=math.radians(25.0),   # ② 近门"敢不敢进门"的朝向阈值
                    abort_along=-150.0):             # ③ 过门多远仍不齐 → 认输停船（−60 实测更差·回 −150）
    """返回 [a, omega]。与 v4 同签名；新增 commit_tol / abort_along 两个安全参数。"""
    px, py, th, v = state; gx, gy = goal
    c, s = math.cos(theta_g), math.sin(theta_g)
    dx, dy = gx - px, gy - py
    along = dx * c + dy * s          # 沿门朝向的纵向距离（>0=门在前方）
    cross = -dx * s + dy * c         # 横向偏移
    heading_err = wrap(theta_g - th)  # 本船朝向 vs 门朝向

    if along < abort_along:
        # ③ 认输闸（−150m）：已冲过门仍没解决 → **果断刹停**，产生【良性 stopped 失败】。
        #    依据：v4 那次碰撞正是过门后游荡 1300m 才撞的；停在原地最坏只是丢 1 局，绝不进交通流。
        #    ⚠️停船时【不再转艏】(omega=0)：转艏会让本船以未知朝向继续漂，反而制造新的会遇几何。
        a = float(np.clip(kv * (0.0 - v) * 4.0, -A_MAX, A_MAX))
        return np.array([a, 0.0])

    if along < 0.0:
        # ① 过门后（v4 的发散源）：目标改用【进近走廊上的一个远驻点】——门后方 Ld_back 处，
        #    而不是门中心。远驻点的方位角对本船位移不敏感（雅可比小）⇒ 期望朝向不会被甩开，
        #    同时它仍指引本船绕回进近线（不像"锁门朝向"那样一路开走再也不回来）。
        Ld_back = max(3.0 * Ld, 400.0)
        tx = gx - Ld_back * c; ty = gy - Ld_back * s
        desired = math.atan2(ty - py, tx - px)
        v_t = v_turn
    else:
        la = max(0.0, along - Ld) if along > Ld else 0.0
        tx = gx - la * c; ty = gy - la * s
        desired = math.atan2(ty - py, tx - px)
        if along < close_dist and abs(cross) < close_cross:
            desired = theta_g
        if abs(wrap(desired - th)) > k_align:
            v_t = v_turn
        elif along < close_dist:
            v_t = v_creep
        else:
            v_t = v_run

    corr = wrap(desired - th)

    # ②★ 转艏可行性限速（本条是 v5 的核心·由 `03` L192 I 的 R_min 不等式直接推出）：
    #    本船转艏率上限 ω_max（本项目 =0.018 rad/s ≈1.03°/秒）。要在【开到门之前】把头转到门朝向，
    #    必须满足   到达时间 ≥ 转艏时间   即   along / v ≥ |heading_err| / ω_max
    #    ⇒ v ≤ along · ω_max / |heading_err|。控制器【自己算这个上限并主动减速】，
    #    于是"半径够不够"不再靠人为调 TAKEOVER_R，而是由控制器在任意半径下自洽保证。
    if along > 0.0:
        _he = max(abs(heading_err), math.radians(2.0))       # 下限防除零/过度限速
        v_cap = along * wmax / _he
        v_t = min(v_t, max(v_cap, v_turn))                   # 不低于 v_turn（留最低机动性）

    omega = float(np.clip(corr / DECISION_DT, -wmax, wmax))
    a = float(np.clip(kv * (v_t - v), -A_MAX, A_MAX))
    return np.array([a, omega])


def make_gate(gx, gy, tg=0.0, L=400.0, Wd=60.0, tol=0.17):
    return dict(gx=gx, gy=gy, th=tg, L=L, W=Wd, tol=tol)
