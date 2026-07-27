#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""次网格细调率 + 按态势拆转艏 —— 连续臂"结构性优势/合规代价"的量化（2026-07-25 later·任务:指标提升）。

【为什么要这两个指标】平滑度家底(`03` 3600)：油门赢 5.4×(铁)·**转艏输 18%·jerk 略输**。
  读机理(本窗口算)：连续油门每步均动 0.0037 = 离散最小格(0.016)的 **23%** = 在做离散【物理上做不到】的细调；
  连续转艏每步均动 0.0156 = 动作箱(0.018)的 **87%** = 几乎打满舵在抖(因船转极慢 ω_max=1.7°/s·想让路/对准就得满舵)。
  ⟹ ① 该报的不是"我们更平滑"(转艏输)·而是**"我们能做离散做不到的细调"**(次网格细调率·本模块①)；
     ② 转艏活动若集中在【让路步】·则那是**可证明合规的代价**·非控制毛病(按态势拆·本模块②)。

【① 次网格细调率 subgrid_*_frac】= 相邻两步执行控制之差满足 `0 < |Δ| < 离散最小非零格步` 的比例。
  ⚠️⚠️ **诚实口径(写作必带·防被当 tautology 打)**：离散臂该指标在**纯网格步之间恒 0**(它的 Δ 只能是 0 或格步整数倍)
     → 本指标**不是"两臂同轴比高低"**·而是量化**"我们实际用掉了多少连续分辨率"**(离散无此自由度)。
     与"油门赢 5.4×"是同一结构性事实的两种表述·后者才是可比的两臂对拼数。
  🔴 **"恒 0"不是无条件的(2026-07-27 独立复审 L226-L 收紧措辞·原文写的是"恒 0·by construction")**：
     离散臂除 49 个网格动作外还有**紧急槽 idx49**，其值由 Alg.1 紧急控制器现算(`usv_colregs.EmergencyController`)、
     是**连续量**，只被 `usv_env._map_action` 截到**物理** ±0.24/±0.03、不落在网格上。
     本模块靠"两步都在 RL 箱内"这条**代理**判据排除紧急步；但紧急动作若恰好落进 RL 箱(实测窗口：
     跟踪器 ω=2·tan(φ)，需 |φ|≲0.52°；ahead/base 模式下可达)，该步就会被计入 ⟹ 其 Δ 非格点倍数
     ⟹ **离散臂该指标可以非 0**。已构造反例验证(单步箱内紧急动作 → subgrid_yaw_frac=0.667)。
     ⟹ **写作红线**：只能写"离散动作方法在其网格动作之间无法做出次格步细调"，
        **绝不写裸的"离散臂该指标 by construction 恒为 0"**；若实测非 0，先查是不是箱内紧急步，别当回归。
     ⟹ 真正的根治 = 把代理判据换成真判据(`source`/`rho_acting` 排除紧急与兜底步·`03` L216-D 建议 2)，
        但那改的是**指标定义 = 影响论文 claim**，须 user 拍板，本次复审不擅自改。

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

# 🔴🔴 判据容差（2026-07-26 修·`03` L224）——**没有它，本模块对离散臂给出的数就是错的**：
#   `abs(-0.018 - -0.012) = 0.005999999999999998 < 0.006` —— 二进制浮点表示不出 0.006，
#   于是"整整挪一格"被 `< W_GRID_STEP` 判成"比一格还细"。实测后果：离散臂 subgrid_yaw_frac
#   本该恒 0，实得 2.5%-12.1%（5 个 ckpt·均值 5.4%），而 subgrid_accel_frac 恰好正常（=0.0），
#   **因为油门格点两两之差最小恰好 = 0.016 精确可表示、转艏的 ±0.012↔±0.018 那两对不可表示**。
#   缝宽只有 1.7e-18 ⟹ 连续臂几乎不受影响（它的 |Δ| 连续分布·落进这条缝的概率≈0），
#   但离散臂的 |Δ| **全部堆在格点倍数上**，正好踩中 ⟹ 只污染离散臂 = 最难看出来的那种错。
#   取 1e-9：比浮点残差(~1e-18)大九个量级、比任何有物理意义的指令差(~1e-5 rad/s)小四个量级。
#   两侧都用它 ⟹ 上界不再把"整格"误判成"次格"，下界不再把 1e-15 级的数值噪声当成"精细操作"。
_GRID_TOL = 1e-9

# 🔴🔴 箱内判据容差（2026-07-27 修·独立复审 L227）——**没有它，本模块对【连续臂】给出的数就是错的**：
#   连续策略的动作是 **float32**，而 A_BOX 是 float64 字面量 0.048。策略在油门上饱和时输出正好等于箱边，
#   但 `float(np.float32(0.048)) = 0.04800000041723251` ⟹ **比 0.048 大 4.17e-10**。
#   原判据容差只有 **1e-12** ⟹ 这些"恰在箱边"的步被判成【越箱】而整批丢掉。
#   **实测后果（热启动 s0·8 个真实场景 413 步）**：**93.9% 的步被误判越箱**，且**全部 source=projection**
#   （即常规投影步、根本不是紧急/兜底步）。逐局"箱内相邻对"因此从 ~47 掉到 **3.06**（离散臂是 48.1）。
#   ⟹ ① 卖点①「次网格细调率」对连续臂**只在剩下 6% 的步上算**，而那 6% 恰恰是"油门没打满"的步
#      ⟹ 系统性**高估**（细调本来就只可能发生在没饱和的步上）；
#      ② 卖点②「按态势拆转艏」对连续臂**必然为空**——让路步本身只占 0.44%，再乘 6% ⟹ 期望不到 1 对
#      ⟹ 这就是 `03` L216-D 挂了很久那笔债的**真正机理**（此前排除过四个原因、都不是它）。
#   **只咬连续臂**：离散臂的施加值来自 `DISCRETE_ACTIONS` 的 float64 字面量，精确等于 0.048 ⟹ 判据成立。
#   （**恰是 `03` L224-A 那个"只坏离散臂"浮点 bug 的镜像**：同一个模块、同一类根因、咬的是另一条臂。）
#   取 1e-6：**与 `evaluate._control_quality` 的 `tol` 逐字相同**（本模块 docstring 一直声称"口径与它对齐"，
#   而此前两处一个 1e-12 一个 1e-6 = 又一处"两边各写一份"的漂移）；比 float32 残差(~4e-10)大三个量级、
#   比紧急控制器的物理满程(0.24/0.03)小五个量级 ⟹ 该排除的照样排除。
_BOX_TOL = 1e-6


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
    inbox = (np.abs(U[:, 0]) <= A_BOX + _BOX_TOL) & (np.abs(U[:, 1]) <= W_BOX + _BOX_TOL)
    adj = inbox[:-1] & inbox[1:]                       # 相邻两步都在箱内（同 _control_quality 的 jerk 口径）
    if not adj.any():
        # 🔴 2026-07-27（独立复审 L226-K）：这条早退路径原先把**全部 7 个键**一起丢成 None，
        #   包括刚为了"把『没采到』和『采到但样本量 0』分开"而加的三个样本量键（`03` L216-D-续）
        #   ⟹ 在最需要分辨的那种局（整局没有任何"相邻两步都在箱内"的对，例如让路步与紧急步大量重叠）
        #   歧义原封不动。这里显式落 `n_inbox_pairs = 0`：
        #     · 键缺失  ⟹ 本函数没跑到这一步（输入无效/步数不足）
        #     · 键 = 0  ⟹ 跑到了，但确实一个箱内相邻对都没有
        #   （`n_pairs_giveway/other` 仍留 None——②块本来就没进，语义是"没算"，不是"算出来是 0"。）
        out["n_inbox_pairs"] = 0
        return out
    dU = np.abs(np.diff(U, axis=0))[adj]               # |Δa|,|Δω|（仅箱内相邻对）
    n = int(dU.shape[0]); out["n_inbox_pairs"] = n
    # ① 次网格细调率：_GRID_TOL < |Δ| < 格步−_GRID_TOL（=离散做不到的细度；|Δ|≈0 排除因离散也能"重复同动作"）
    #    🔴 两侧容差【非可选】：见 _GRID_TOL 注释——少了上界容差，"整整挪一格"会被判成"比一格还细"，
    #    离散臂本该恒 0 的指标实测变成 5.4%（且只坏转向那一维），聚合数字上完全看不出来。
    out["subgrid_accel_frac"] = round(float(np.mean(
        (dU[:, 0] > _GRID_TOL) & (dU[:, 0] < A_GRID_STEP - _GRID_TOL))), 6)
    out["subgrid_yaw_frac"] = round(float(np.mean(
        (dU[:, 1] > _GRID_TOL) & (dU[:, 1] < W_GRID_STEP - _GRID_TOL))), 6)
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
