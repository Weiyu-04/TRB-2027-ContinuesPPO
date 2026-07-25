#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""次网格细调率 + 按态势拆转艏 —— 连续臂"结构性优势/合规代价"的量化（2026-07-25 later·任务:指标提升）。

【为什么要这两个指标】平滑度家底(`03` 3600)：油门赢 5.4×(铁)·**转艏输 18%·jerk 略输**。
  读机理(本窗口算)：连续油门每步均动 0.0037 = 离散最小格(0.016)的 **23%** = 在做离散【物理上做不到】的细调；
  连续转艏每步均动 0.0156 = 动作箱(0.018)的 **87%** = 几乎打满舵在抖(因船转极慢 ω_max=1.7°/s·想让路/对准就得满舵)。
  ⟹ ① 该报的不是"我们更平滑"(转艏输)·而是**"我们能做离散做不到的细调"**(次网格细调率·本模块①)；
     ② 转艏活动若集中在【让路步】·则那是**可证明合规的代价**·非控制毛病(按态势拆·本模块②)。

【① 次网格细调率 subgrid_*_frac】= 相邻两步执行控制之差满足 `0 < |Δ| < 离散最小非零格步` 的比例。
  ⚠️⚠️ **诚实口径(写作必带·防被当 tautology 打)**：离散臂该指标【恒 0·by construction】(它的 Δ 只能是 0 或格步整数倍)
     → 本指标**不是"两臂同轴比高低"**·而是量化**"我们实际用掉了多少连续分辨率"**(离散无此自由度)。
     与"油门赢 5.4×"是同一结构性事实的两种表述·后者才是可比的两臂对拼数。

【② 按态势拆转艏 yaw_incr_giveway / yaw_incr_other】= |Δω| 均值按该步态势 ρ 分组。
  ρ∈{2,3,4}(head_on/crossing/overtake)=让路态。让路步转艏更大 ⟹ 转艏活动=合规让路机动的代价(可写)。
  变化 |u[k+1]−u[k]| 归属 **ρ[k+1]**(=该新动作被选择时所处态势)。

【口径与 evaluate._control_quality 对齐】只用【正常操作步】(|a|≤A_BOX ∧ |ω|≤W_BOX)·jerk 同款"相邻两步都在箱内"才算；
  排除紧急/兜底步(那些是共享紧急控制器的物理满程输出·非 RL 策略·混入污染且各臂不对称·红队 MAJOR L72)。

【纯模块】只 numpy·**不 import vesselmodels** → 本机可单测(吸取"负速度崩本机没验出来"教训:能本机验的就本机验)。
  常量与 `usv_env` 单一真相源【须一致】·调用方(有 vesselmodels 时)应 assert 相等(见 assert_grid_matches)。
"""
import numpy as np

# 离散 49=7×7 网格（源真相 usv_env.A_ACC/A_OMEGA·此处镜像·调用方须 assert_grid_matches 校验）
A_BOX = 0.048          # RL 正常操作油门箱 = max(A_ACC)
W_BOX = 0.018          # RL 正常操作转艏箱 = max(A_OMEGA)
A_GRID_STEP = 0.016    # 离散油门最小非零格步（A_ACC 相邻间距）
W_GRID_STEP = 0.006    # 离散转艏最小非零格步（A_OMEGA 相邻间距）
GIVEWAY_RHOS = (2, 3, 4)   # RHO_HEAD_ON / RHO_CROSSING / RHO_OVERTAKE


def assert_grid_matches(a_acc, a_omega):
    """调用方(有 vesselmodels)校验本模块镜像常量 == usv_env 真相源·防静默漂移（同 uterm 的 DECISION_DT 守卫范式）。"""
    a = sorted(float(x) for x in a_acc); w = sorted(float(x) for x in a_omega)
    a_step = min(a[i + 1] - a[i] for i in range(len(a) - 1))
    w_step = min(w[i + 1] - w[i] for i in range(len(w) - 1))
    for got, exp, name in ((max(a), A_BOX, "A_BOX"), (max(w), W_BOX, "W_BOX"),
                           (a_step, A_GRID_STEP, "A_GRID_STEP"), (w_step, W_GRID_STEP, "W_GRID_STEP")):
        if abs(got - exp) > 1e-9:
            raise ValueError(f"metrics_subgrid.{name}={exp} ≠ usv_env 真相源 {got}（网格漂移→次网格率口径错）")
    return True


def subgrid_and_rho_split(applied, rhos=None):
    """applied: [(a,ω)...] 逐步执行控制（=env.last_action·post-shield 真施加值）；rhos: 同长 ρ 列表（可选·给②）。
    返回 dict（数据不足→对应键 None）：
      subgrid_accel_frac / subgrid_yaw_frac : 0<|Δ|<格步 的比例（=用掉连续分辨率的比例·离散恒 0 by construction）
      n_inbox_pairs                          : 参与统计的相邻箱内对数（样本量·判读须看）
      yaw_incr_giveway / yaw_incr_other      : |Δω| 均值按让路/非让路态分组（②·合规代价证据）
      n_pairs_giveway / n_pairs_other        : 各组对数
    """
    out = {k: None for k in ("subgrid_accel_frac", "subgrid_yaw_frac", "n_inbox_pairs",
                             "yaw_incr_giveway", "yaw_incr_other", "n_pairs_giveway", "n_pairs_other")}
    if applied is None or len(applied) < 2:
        return out
    U = np.asarray(applied, dtype=float)
    if U.ndim != 2 or U.shape[1] != 2:
        return out
    inbox = (np.abs(U[:, 0]) <= A_BOX + 1e-12) & (np.abs(U[:, 1]) <= W_BOX + 1e-12)
    adj = inbox[:-1] & inbox[1:]                       # 相邻两步都在箱内（同 _control_quality 的 jerk 口径）
    if not adj.any():
        return out
    dU = np.abs(np.diff(U, axis=0))[adj]               # |Δa|,|Δω|（仅箱内相邻对）
    n = int(dU.shape[0]); out["n_inbox_pairs"] = n
    # ① 次网格细调率：0 < |Δ| < 离散最小非零格步（严格小于=离散做不到的细度；|Δ|=0 排除因离散也能"重复同动作"）
    out["subgrid_accel_frac"] = round(float(np.mean((dU[:, 0] > 0.0) & (dU[:, 0] < A_GRID_STEP))), 6)
    out["subgrid_yaw_frac"] = round(float(np.mean((dU[:, 1] > 0.0) & (dU[:, 1] < W_GRID_STEP))), 6)
    # ② 按态势拆 |Δω|：变化归属 ρ[k+1]（新动作被选时的态势）
    if rhos is not None and len(rhos) == len(U):
        r_next = np.asarray(rhos, dtype=int)[1:][adj]
        gw = np.isin(r_next, GIVEWAY_RHOS)
        if gw.any():
            out["yaw_incr_giveway"] = round(float(dU[gw, 1].mean()), 6)
        if (~gw).any():
            out["yaw_incr_other"] = round(float(dU[~gw, 1].mean()), 6)
        out["n_pairs_giveway"] = int(gw.sum()); out["n_pairs_other"] = int((~gw).sum())
    return out
