#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""转向平滑算子（舵速率限制 / 低通）—— **训练期与评估期共用的唯一定义**（`03` L228）。

═══ 为什么要单独一个文件 ═══
这个算子有**两个使用场景**，而且两边必须**逐字一致**，否则整件事失去意义：
  · **评估期**（`代码/tests/reeval_official.py` 的 `YawSlewLimitPolicy` / `YawLowPassPolicy`·`03` L218/L221）
    —— 拿已训好的策略跑一遍，看平顺度—到达率的取舍曲线；
  · **训练期**（本文件被 `usv_continuous_shield.ContinuousProjectionEnv` 用·`03` L228）
    —— 把限制放进训练循环，让策略**在知道自己指令会被限速的前提下学**，自己学会补偿。

**两边不一致 = 训练时学的和部署时执行的不是同一个东西**（"train what you deploy" 被破坏），
那正是评估期方案的弱点所在；如果训练期又另写一份公式，这个实验从根上就废了。
⟹ **公式只准写在这里一处**。评估期那两个包装器有契约测试盯着它们与本文件逐位一致。

═══ 三条不可协商的性质（回归钉死） ═══
1. **不施加档逐位等价**：`frac=1.0`（限速）/ `alpha=1.0`（低通）⟹ 输出 **==** 输入（不是"约等于"）。
   这是对照组的干净性，也是"默认关 = bit-identical"的基础。
2. **永不越箱**：输出恒在 ±w_box 内。限速是"往参考值拉近"、低通是凸组合 ⟹ 两者天然满足；
   但**参考值本身必须先夹回箱内**——盾/紧急控制器会施加物理满程 ±0.03，直接拿它当参考会把
   窗口整段推到 RL 箱外（`03` L222-#5 那个自引入 bug 的原始形态）。
3. **只动转向那一维**：油门不碰（抖动是转向的问题·`03` L220-B）。
"""
import numpy as np


def clamp_reference(w_applied, w_box):
    """参考舵位 = 【盾之后真正施加】的 ω 在**我们自己权限**内的投影。

    为什么不是"自己上一步的指令"：盾会改写约 7% 的步，一改写内部状态就与舵实际位置脱节（`03` L222-#4）。
    为什么必须夹回箱内：紧急控制器走物理满程 ±0.03，超出 RL 正常操作箱 ±0.018；
    直接拿它当参考会让窗口整段落在箱外 ⟹ 我们的指令越箱 = 偷用超出对照口径的操作权限（`03` L222-#5）。
    """
    w = float(w_applied)
    b = float(w_box)
    return b if w > b else (-b if w < -b else w)


def apply_slew_limit(w_raw, w_ref, frac, w_box):
    """舵速率限制：`ω_out = clip(ω_raw, ω_ref ± frac·箱宽)`，箱宽 = 2·w_box。

    `frac=1.0` ⟹ 一步足以从满左翻到满右 ⟹ **与不限速逐位相同**。
    `w_ref=None`（首步/拿不到参考）⟹ 不限。
    物理依据：真船舵机有最大转舵速率，而本仿真的动作空间只限 |ω|、不限 |Δω|（`03` L221-B）。
    """
    f = float(frac)
    if not (0.0 < f <= 1.0):
        raise ValueError(f"速率限制系数须 ∈ (0,1]，得到 {f}（1.0 表示不限）")
    if w_ref is None:
        return float(w_raw)
    d = f * 2.0 * float(w_box)
    lo, hi = float(w_ref) - d, float(w_ref) + d
    w = float(w_raw)
    return lo if w < lo else (hi if w > hi else w)


def apply_lowpass(w_raw, w_ref, alpha):
    """转向低通：`ω_out = (1−α)·ω_ref + α·ω_raw`，α ∈ (0,1]。α=1 ⟹ **逐位等于不滤波**。"""
    a = float(alpha)
    if not (0.0 < a <= 1.0):
        raise ValueError(f"低通系数 α 须 ∈ (0,1]，得到 {a}（α=1 表示不滤波）")
    if w_ref is None:
        return float(w_raw)
    return (1.0 - a) * float(w_ref) + a * float(w_raw)


def smooth_yaw(u_desired, w_applied_prev, w_box, *, slew_frac=None, lowpass_alpha=None):
    """对 (a, ω) 的**转向那一维**施加平滑；两个旋钮都不给 ⟹ **原样返回**（逐位不变）。

    `w_applied_prev`：上一步**盾之后真正施加**的 ω（None = 首步/拿不到 ⟹ 不施加）。
    两个旋钮同时给会 fail-fast —— 同时施加两种平滑，出来的曲线归因不清（是低通的功劳还是限速的？）。
    """
    if slew_frac is None and lowpass_alpha is None:
        return u_desired
    if slew_frac is not None and lowpass_alpha is not None:
        raise ValueError("舵速率限制与低通只能开一个（同开则平顺度改善归因不清）。")
    u = np.asarray(u_desired, dtype=float).copy()
    ref = None if w_applied_prev is None else clamp_reference(w_applied_prev, w_box)
    w_raw = float(u[1])
    w = (apply_slew_limit(w_raw, ref, slew_frac, w_box) if slew_frac is not None
         else apply_lowpass(w_raw, ref, lowpass_alpha))
    # 🔴 不可协商的不变量：输出永远在 RL 动作箱内（四臂公平比较的口径底线）
    assert abs(w) <= float(w_box) + 1e-9, f"平滑输出 {w} 越出 RL 箱 ±{w_box}"
    u[1] = w
    return u
