"""
TRB 环境 · 奖励模块
===================
忠实复现 Krasowski & Althoff 2024 §VI-B 式(10)：r = r_sparse + r_colregs + r_goal + r_velocity + r_deviate。
（论文 = 参考资料/2402.08502v2.pdf 第11页；r_colregs 来自 [41]=Meyer 2020，见《文献核实笔记》三）

5 个分量（系数全部 fact-based 来自 Krasowski Table II，PDF 第11页）：
  r_sparse  : c_time·1_time + c_area·1_area + c_goal·1_goal + c_stopped·1_stopped
              + c_collision·1_collision + c_emergency·1_emergency（除 c_goal 外全负）
  r_goal    : 进/退目标的稠密奖励（见下「⚠️ r_goal 符号」）
  r_velocity: 速度偏离 [v_low, v_high] 的惩罚
  r_deviate : c_deviate·min(|d_lat|, d_hull)（偏离 init→goal 参考线）
  r_colregs : Meyer 2020 式(26) 动态障碍 COLREG 惩罚（见下「⚠️ r_colregs」）

⚠️ **r_goal 符号（偏离论文字面，已判定，见《文献核实笔记》三 + `03` 待补）**：
  论文字面 r_goal = c_reach·(‖p_t−goal‖ − ‖p_{t-1}−goal‖)，配 c_reach=+1.5 → 前进(距离减小)得负奖励，
  与正文"reinforced in goal reaching"矛盾，~90% 是论文 typo。本模块用**物理正确式**：
      r_goal = c_reach·(‖p_{t-1}−goal‖ − ‖p_t−goal‖)   （前进→正奖励）

⚠️ **r_colregs（Meyer 2020 式26 + Table 4 参数；3 处事实 + 1 个 step4 待标定）**：
  形式：r_colregs = −[1/(1+exp(γ_θ,dyn·|φ|))]·α_x·exp((ζ_v(φ,v_y)·v_ŷ − ζ_x(φ))·d)
  - φ   = 他船相对方位（右舷正/左舷负，= 观测模块的 β），用于 weight + Meyer 3 扇区(st.b./port/stern)分类。
  - v_y = 他船径向速度，**>0 = 驶向本船(approaching)**（采物理正确符号，gym-auv 一致；偏 Krasowski 字面，见笔记三）。
  - ζ_x(φ) 3 扇区 / ζ_v(φ,v_y) 3 扇区×v_y 符号 = Meyer Table 4（≠观测的 4 扇区，别混）。
  - 🔴 **尺度归一化（BLOCKER 的处理，见笔记三）**：Meyer 原始参数为 ~150m 量程调，直接代进 Krasowski
    的 8000m 会让惩罚爆炸（指数随距离增长）。本模块默认把 **v_y 用 v_pm,max 归一化**（v_ŷ=v_y/v_pm,max，
    gym-auv 对传感器速度的标准做法）+ **指数钳到 ≤0 双保险**（物理上惩罚只该随距离衰减，钳后恒有界
    [−weight·α_x, 0]，即便他船速超尺度也不爆炸/不 inf）。**精确尺度 step4 用 Table III 标定**
    （`colregs_vel_scale` 可配置）。d 保持原始米（远场自然≈0）。
    ⚠️ **论文写作必须显式声明此归一化偏离**（归一化 v_y 等价缩放了有效 ζ_v，已非 Meyer 字面值）——
    不能在论文里写"uses Meyer's values"而不提归一化，否则成 paper-level over-claim（独立复核 Agent1 硬要求）。

⚠️ 接口约定：本模块计算 5 分量需要的几何量自洽（d_lat / φ / d / v_y），与观测模块用同一符号约定；
  归一化（喂给网络的）不在这里做——这里出**原始奖励标量**。
"""
from __future__ import annotations

import numpy as np

from .usv_dynamics import wrap_to_pi

# ── Krasowski Table II 系数（PDF 第11页，fact-based）─────────────────────────
C_TIME: float = -25.0
C_AREA: float = -5.0
C_GOAL: float = 50.0
C_STOPPED: float = -40.0
C_COLLISION: float = -50.0
C_EMERGENCY: float = -0.5
C_REACH: float = 1.5
C_V: float = -2.0
C_DEVIATE: float = -0.001
V_LOW: float = 2.5
V_HIGH: float = 8.0
D_HULL: float = 2000.0

# ── Meyer 2020 Table 4（r_colregs，PDF 第16页；见《文献核实笔记》三）──────────
ALPHA_X: float = 75.0
GAMMA_THETA_DYN: float = 1.0
# ζ_x(φ) 按 Meyer 3 扇区
ZETA_X: dict = {"starboard": 0.007, "port": 0.009, "stern": 0.01}
# ζ_v(φ, v_y) 按 3 扇区 × v_y 符号：(v_y>=0 即 approaching, v_y<0 即 receding)
ZETA_V: dict = {
    "starboard": (0.004, 0.05),
    "port": (0.007, 0.005),
    "stern": (0.007, 0.005),
}
DEFAULT_D_SENSE: float = 8000.0       # 只对感知距离内他船算 r_colregs
DEFAULT_V_PM_MAX: float = 10.0        # 他船点质量模型最大速度（Table II）→ v_y 归一化尺度

# Meyer 3 扇区边界（式25；右舷正/左舷负，与观测同号但只 3 区，≠观测 4 区）
_MEYER_SIDE_DEG: float = 112.5

_COMPONENT_KEYS = ("sparse", "colregs", "goal", "velocity", "deviate")
_TERM_KEYS = {"time", "area", "stopped", "collision", "goal"}


def _meyer_sector(phi_rad: float) -> str:
    """按相对方位 φ（右舷正/左舷负）分 Meyer 3 扇区。

    starboard: 0° ≤ φ < 112.5°   port: -112.5° ≤ φ < 0°   stern: |φ| ≥ 112.5°
    """
    d = np.degrees(phi_rad)
    if 0.0 <= d < _MEYER_SIDE_DEG:
        return "starboard"
    if -_MEYER_SIDE_DEG <= d < 0.0:
        return "port"
    return "stern"


class RewardFunction:
    """计算 Krasowski 式(10) 的逐步奖励。静态持有目标/初始位置/参数；内部追踪上一步位置（算 r_goal）。

    用法：
        rf = RewardFunction(goal_center, init_position)
        rf.reset(ego_state)                                  # 每个 episode 开始
        r, parts = rf.step(ego_state, obstacles, term_flags, emergency_used)
    """

    def __init__(
        self,
        goal_center,
        init_position,
        *,
        v_pm_max: float = DEFAULT_V_PM_MAX,
        d_sense: float = DEFAULT_D_SENSE,
        colregs_vel_scale: float | None = None,
        colregs_weight: float = 1.0,
        gamma: float = 0.99,
        goal_orientation: tuple | None = None,
        well_shaping_weight: float = 0.0,
        shaping_radius: float = 500.0,
        xtrack_weight: float = 0.0,
        xtrack_radius: float = 80.0,
        park_weight: float = 0.0,
        park_radius: float = 400.0,
        park_v_target: float = 4.0,
        c_step: float = 0.0,
        c_dwell: float = 0.0,
        w_dwell: float = 90.0,
        h_dwell: float = 0.52,
        dwell_radius: float = 250.0,
        b_dwell: float = 0.0,
        c_reach: float = C_REACH,
        dock_radius: float = 0.0,
        v_dock: float = V_LOW,
    ):
        """
        goal_center       : [x, y] 目标区中心
        init_position     : [x, y] 本船初始位置（定义 init→goal 参考线，算 d_lat）
        v_pm_max          : 他船最大速度（Table II=10），默认 v_y 归一化尺度
        d_sense           : 感知距离（Table II=8000），超出不计 r_colregs
        colregs_vel_scale : r_colregs 中 v_y 的归一化尺度（默认 = v_pm_max）。
                            ⚠️ step4 用 Table III 标定的旋钮（见 docstring「r_colregs」）。
        colregs_weight    : r_colregs 总权重开关（Base/RR 区分，4d-②）。1.0=Rule-reward（默认、现状、忠实式10）；
                            0.0=Base（关 r_colregs；论文 §VII p11：Base=r_sparse+r_goal+r_velocity+r_deviate）。
        ── 修法A 进门势函数（PBRS·`03` L80-续3/续8·治"到达率与主导奖励解耦/弱进门梯度"根因）──
        gamma             : 折扣因子（PBRS F=γΦ(s')−Φ(s) 的 γ·须与 trainer 同源 0.99·Ng 1999 不变性要求）。
        goal_orientation  : (start, end) 目标朝向区间（rad）→ θ_c=区间中心（算 align）。well_shaping_weight>0 时必给。
        well_shaping_weight: well_B·进门势强度（默认 0.0=关=与现状逐位等价 bit-identical）。A/B 用 200=4×C_GOAL。
        shaping_radius    : 近场势作用半径 R_near（默认 500m·仅目标 ~500m 内给梯度·远离冲突区 ~3700m·不干扰避碰·R3 C2）。
        ── 对症修法 横向进带势 Φ_xtrack（PBRS·`03` L88·治"终端横向 cross-track 进不了窄带·刹停在带外"·与修法A 并列独立项）──
        xtrack_weight     : well_X·横向进带势强度（默认 0.0=关=与现状逐位等价 bit-identical）。Φ_xtrack=well_X·prox_radial(‖r‖;R_near)·prox_lat(e_cross;R_lat)。
                            e_cross=经 θ_c 法向的横向(cross-track)偏差=到"过目标中心、沿 θ_c 方向中心线"的垂距（独立量·非 self.e_lat 的 init→goal 横向）；
                            prox_lat=max(0,1−|e_cross|/R_lat) 线性核（同 well_B 的 prox 形式·带外恒定拉回梯度·Müller&Kudenko2025/复审 L88 选线性非高斯）。well_X>0 时必给 goal_orientation。
        xtrack_radius     : 横向势半径 R_lat（默认 80m·覆盖实测失败 |e_cross|=32-44m + 60m 窄带半宽 30m + 余量·`03` L88 重评钉死）。
        """
        self.goal_center = np.asarray(goal_center, dtype=float)
        self.init_position = np.asarray(init_position, dtype=float)
        if self.goal_center.shape != (2,) or self.init_position.shape != (2,):
            raise ValueError("goal_center / init_position 必须是 2 维 [x, y]")
        self.v_pm_max = float(v_pm_max)
        self.d_sense = float(d_sense)
        self.colregs_vel_scale = float(colregs_vel_scale) if colregs_vel_scale is not None else float(v_pm_max)
        if self.colregs_vel_scale <= 0:
            raise ValueError("colregs_vel_scale 必须 > 0")
        self.colregs_weight = float(colregs_weight)        # Base/RR 开关：1=RR(默认) / 0=Base（4d-②）
        if self.colregs_weight < 0:
            raise ValueError("colregs_weight 必须 ≥ 0")

        # 修法A 进门势函数参数（PBRS）+ 对症修法 横向进带势参数（PBRS·`03` L88）
        self.gamma = float(gamma)
        self.well_B = float(well_shaping_weight)
        self.shaping_radius = float(shaping_radius)
        self.well_X = float(xtrack_weight)                 # 对症：横向进带势强度（0=关=逐字节不变）
        self.xtrack_radius = float(xtrack_radius)          # 对症：横向势半径 R_lat
        # ── 想法B 终端保速势 Φ_park（PBRS·`03` L109·治"终端横向修正时过早减速到 v≈0 刹停带外·成功局保 ~3.5 速度"·连续臂专属）──
        #   数据指靶（L109）：失败局 x 进带、y 出带 ~13-48m 时减速到 1.6 卡死；成功局保 ~3.5 速度有机动力把横偏修进带。
        #   到达判据 is_reached【不要求停下】→ 保速不碍到达（移动中位置/朝向进区即到达）→ Φ_park 近目标奖励保速、补 r_goal 近门衰减。
        #   Φ_park=well_park·prox_radial(‖r‖;R_park)·speed_frac(v;V_target)·PBRS 策略不变（学习梯度·不改最优）。park_weight=0=关=逐位等价。
        self.well_park = float(park_weight)                # 想法B：终端保速势强度（0=关=逐字节不变）
        self.park_radius = float(park_radius)              # 想法B：近目标作用半径 R_park
        self.park_v_target = float(park_v_target)          # 想法B：目标机动速度（speed_frac 封顶·成功局 ~3.5-4.5）
        # ── 修法C 全局每步生存成本 c_step（非PBRS·真改最优·`03` L123·治"游荡局部最优·门口零梯度平台"·连续臂专属）──
        #   每步无条件 r_sparse −= c_step（像 C_TIME 但每步、非超时 flag 门控）。当前奖励里"门口/游荡区每步净≈0"=零梯度平台
        #   →策略拿不到"进门 vs 不进门"局部信号；−c 把平台倾斜成稠密负斜坡→唯一比超时(−25)好的回合出口是到达(+50)
        #   →等价稠密"朝进门"压力·补稀疏 +50 探索缺口。非PBRS：Σγ^t(−c) 依赖回合长度 T（罚长游荡）→真改最优（不像
        #   well_B/X/park 那三个 PBRS 因 telescoping 守恒只缩 std 消不掉局部最优）。c_step=0=关=与现状逐位等价 bit-identical。
        self.c_step = float(c_step)                        # 修法C：每步生存成本（0=关=逐字节不变·连续臂专属·离散盾硬拒）
        # ── 非PBRS 入库赤字滞留成本 r_dwell（非PBRS·真改最优·`03` L161/L162·治"corr≈0 奖励-到达脱钩·到门口没停进去"·连续臂专属）──
        #   近场每步按"离真入库差多少"(横向 e_cross + 朝向 dθ)扣 −c_dwell·g(s)∈[−c_dwell,0]·真入库 is_reached→回合止→成本停。
        #   结构上无正的每步项可刷=farm 免疫（与被淘汰方案 A2"盒内 +250/步可刷"相反）；Σγ^t(−c·g) 依赖滞留时长→真改最优（非PBRS·同 c_step 通道·入 r_sparse）。
        #   c_dwell=0=关=与现状逐位等价 bit-identical（整块 step() 跳过·不减不加键）。b_dwell=终端入库锚（默认 0·仅纯 dwell 现"回避近场躲成本"病象时开·补正锚）。
        self.c_dwell = float(c_dwell)                      # r_dwell：入库赤字滞留成本系数（0=关=逐字节不变·连续臂专属·离散盾硬拒）
        self.w_dwell = float(w_dwell)                      # r_dwell：横向 cross-track 赤字尺度 W_DWELL（默认 90m）
        self.h_dwell = float(h_dwell)                      # r_dwell：朝向赤字尺度 H_DWELL（默认 0.52rad≈30°）
        self.dwell_radius = float(dwell_radius)            # r_dwell：近场作用半径 R_DWELL（默认 250m·带外 r_dwell=0）
        self.b_dwell = float(b_dwell)                      # r_dwell：终端入库锚 B_DWELL（默认 0·真入库 +B_DWELL·补"回避近场躲成本"病）
        # ── Rung1 重标 r_goal 系数 c_reach（非PBRS·真改最优·`03` L172/设计 wf·治"corr≈0 奖励-到达脱钩：+50 占近常数回报 ~0.83% 被淹没"）──
        #   r_goal=c_reach·(d_prev−d_now)·telescoping Σ=c_reach·(d起−d终)·对【既有】r_goal 均匀标量缩放·【不新增势/驻点/吸引子】（well_X/park/dwell 崩健康种子教训的反面）。
        #   降 c_reach(1.5→0.2)压回报量级→VecNormalize 除数(√ret_var)变小→终端 +50/−40/−25 在归一化 reward 里存活为更大更可学信号（+50 归一化占比 ~0.83%→~6%）。
        #   ⚠️起点距【跨局近常数】(n=62 亲验 ego 起点恒(1000,600)/goal 恒 x=5000/d_init 变异 0.34%)→回报≈c_reach·d起 近常数·+50 被淹没=corr≈0 真因（非"d_start 方差"·`03` L172 (a) 自我订正）。
        #   默认=C_REACH(模块常量·非硬编 1.5·防常量漂移)=与现状逐位等价 bit-identical。连续臂专属（离散 maker 不传=忠实 Krasowski typo-fix 版）。
        self.c_reach = float(c_reach)                      # Rung1：r_goal 进度奖励系数（默认 C_REACH=1.5=现状·降救脱钩·连续臂专属）
        # ── Rung2 泊位精修门 dock_radius/v_dock（非PBRS·真改最优·`03` L172/设计 wf·治"过路态速度地板泄漏进入库精修段：入库须减速却每步吃 −2·(2.5−v)"）──
        #   泊位区(‖p−goal_center‖≤dock_radius)内把 r_velocity 低速地板下限从 V_LOW【降到 v_dock·非清零】→入库减速免罚（消泄漏）；v<v_dock 仍被罚 + C_STOPPED 终端不动=天然【防"停门口"新坑】。
        #   dock_radius=0(默认)→整块无效=bit-identical；v_dock=V_LOW(默认)→即便 dock_radius>0 也 no-op=bit-identical（双默认保险）。连续臂专属。
        self.dock_radius = float(dock_radius)              # Rung2：泊位精修门半径 R_dock（默认 0=关·区内减免速度地板·连续臂专属）
        self.v_dock = float(v_dock)                        # Rung2：区内残余速度地板下限（默认 V_LOW=不变·降到~1.0 让必要减速免罚·非零保 C_STOPPED 防停门口）
        if self.well_B < 0:
            raise ValueError("well_shaping_weight 必须 ≥ 0")
        if self.well_X < 0:
            raise ValueError("xtrack_weight 必须 ≥ 0")
        if self.well_X > 0.0 and self.xtrack_radius <= 0:
            raise ValueError("xtrack_radius 必须 > 0（well_X>0 时）")
        if self.well_park < 0:
            raise ValueError("park_weight 必须 ≥ 0")
        if self.well_park > 0.0 and (self.park_radius <= 0 or self.park_v_target <= 0):
            raise ValueError("park_radius / park_v_target 必须 > 0（park_weight>0 时）")
        if self.c_step < 0:
            raise ValueError("c_step 必须 ≥ 0（每步生存成本·非负）")
        if self.c_dwell < 0:
            raise ValueError("c_dwell 必须 ≥ 0（入库赤字滞留成本系数·非负）")
        if self.c_dwell > 0.0 and (self.w_dwell <= 0 or self.h_dwell <= 0 or self.dwell_radius <= 0):
            raise ValueError("w_dwell / h_dwell / dwell_radius 必须 > 0（c_dwell>0 时）")
        if self.b_dwell < 0:
            raise ValueError("b_dwell 必须 ≥ 0（终端入库锚·非负）")
        if self.b_dwell > 0.0 and self.c_dwell <= 0.0:
            raise ValueError("b_dwell>0 需 c_dwell>0（终端锚仅配合 dwell 成本用·防 silent no-op）")
        if self.c_reach < 0:
            raise ValueError("c_reach 必须 ≥ 0（r_goal 进度奖励系数·非负）")
        if self.dock_radius < 0:
            raise ValueError("dock_radius 必须 ≥ 0（泊位精修门半径·非负·0=关）")
        # v_dock 仅 dock_radius>0 时生效：须 0.48<v_dock≤V_LOW。下界 0.48=离零的保守正地板(防 v_dock→0 复活"停船墙")；
        # ⚠️【订正·`03` L176 对抗复审 CONFIRMED】旧注释"0.48=单步 max 减速→离停船墙≥一整步"物理推导错 5×：真实单步 max 减速=a_max·dt=0.24·10=2.4 m/s(非0.48)·故"离停船墙≥1步"不成立(v_dock=1.0 实际仅差~0.4步就触 stopped)·0.48 仅作离零正裕度·真正防"停门口"靠 v_dock>0 + C_STOPPED 终端罚(非"一步缓冲")。注:阶段1实测 v_dock=1.0 确让健康种子减速停死(`03` L176)→v_dock 已列为败招·当前不用。
        #   上界 ≤V_LOW=只准【降低】地板(减免)、禁止抬高地板(否则区内反罚更多·与治病相反)。
        if self.dock_radius > 0.0 and not (0.48 < self.v_dock <= V_LOW):
            raise ValueError(f"dock_radius>0 时 v_dock 须 0.48<v_dock≤{V_LOW}（得 {self.v_dock}·防停船墙复活/防抬高地板反罚）")
        # ⚠️ L161 门控扩（唯一既有逻辑改点）：dwell 也需 θ_c（e_cross 走 θ_c 法向 + dθ=θ−θ_c）→ 加 `or self.c_dwell>0`。
        if self.well_B > 0.0 or self.well_X > 0.0 or self.c_dwell > 0.0:   # 两势任一开或 dwell 开都需 θ_c（align/e_cross/dθ 都用·复审 L88·L161 门控扩）
            if goal_orientation is None:
                raise ValueError("well_shaping_weight>0 或 xtrack_weight>0 或 c_dwell>0 需 goal_orientation（算 θ_c）")
            if (self.well_B > 0.0 or self.well_X > 0.0) and self.shaping_radius <= 0:   # shaping_radius 仅 well_B/well_X 的 prox 用·dwell 用 dwell_radius（不牵连）
                raise ValueError("shaping_radius 必须 > 0")
            # θ_c = 朝向区间【有向弧心】（wrap 到 (−π,π]）→ align=(1+cos(θ−θ_c))/2 的目标朝向。
            # ⚠️ 用 lo+0.5·((hi−lo) mod 2π)【非】算术中点 0.5·(lo+hi)：后者对跨 ±π 的有向角区间错（偏 ~π·奖励背离方向）。
            # 与已上线 observation 模块 usv_observation.py:78 的有向弧 width=(hi−lo)%2π 同口径（CleanR1 Q4f 抓·`03` L80-续10）。
            _lo, _hi = float(goal_orientation[0]), float(goal_orientation[1])
            self.theta_c = wrap_to_pi(_lo + 0.5 * ((_hi - _lo) % (2.0 * np.pi)))
        else:
            self.theta_c = None

        # init→goal 参考线横向单位向量（算 d_lat，与观测模块同约定）
        line = self.goal_center - self.init_position
        norm = float(np.linalg.norm(line))
        if norm < 1e-9:
            raise ValueError("初始位置与目标中心重合，无法定义 init→goal 参考线")
        e_long = line / norm
        self.e_lat = np.array([-e_long[1], e_long[0]])

        self._prev_pos: np.ndarray | None = None
        self._prev_phi: float = 0.0                        # 修法A：上一步 Φ_A(s)（reset 设·well_B=0 恒 0）
        self._prev_phi_x: float = 0.0                      # 对症：上一步 Φ_xtrack(s)（reset 设·well_X=0 恒 0·`03` L88）
        self._prev_phi_p: float = 0.0                      # 想法B：上一步 Φ_park(s)（reset 设·well_park=0 恒 0·`03` L109）

    def _phi(self, ego) -> float:
        """修法A 进门势 Φ(s)=well_B·prox(d_to_center)·align_cos(θ)（纯状态函数→Ng PBRS 策略不变·`03` L80-续3/续8）。
        prox=max(0,1−d/R_near)：仅目标近场给梯度（远场=0·不碰避碰段·R3 C2）；
        align=(1+cos(wrap(θ−θ_c)))/2：cos 平滑·几乎处处有"拧向进门航向 θ_c"梯度（治原 align 硬门零梯度·L80 MAJOR-1）。
        well_B=0 → 恒返 0（→ shaping 全 0 → 与现状逐位等价）。"""
        if self.well_B <= 0.0:
            return 0.0
        d = float(np.linalg.norm(np.asarray(ego[:2], dtype=float) - self.goal_center))
        prox = max(0.0, 1.0 - d / self.shaping_radius)
        if prox <= 0.0:
            return 0.0
        align = 0.5 * (1.0 + np.cos(wrap_to_pi(float(ego[2]) - self.theta_c)))
        return self.well_B * prox * float(align)

    def _phi_xtrack(self, ego) -> float:
        """对症 横向进带势 Φ_xtrack(s)=well_X·prox_radial(‖r‖)·prox_lat(e_cross)（纯状态函数→Ng PBRS 策略不变·`03` L88）。
        prox_radial=max(0,1−‖r‖/R_near)：复用 R_near=shaping_radius 近场门（远场=0·不碰避碰段·同 Φ_A 哲学）；
        e_cross=r·n_perp（r=goal_center−pos·n_perp=(−sinθ_c,cosθ_c)）=到"过目标中心沿 θ_c 中心线"的横向(cross-track)垂距；
        prox_lat=max(0,1−|e_cross|/R_lat)：线性核（带外恒定拉回梯度·复审 L88 实测高斯近中心线会反号·选线性·R_lat=xtrack_radius）。
        well_X=0 → 恒返 0（→ Φ_xtrack 全 0 → 与现状逐位等价）。【独立于 self.e_lat(init→goal 横向·r_deviate 用)·n_perp 走 θ_c·复审 L88 BLOCKER 区分】。"""
        if self.well_X <= 0.0:
            return 0.0
        r = self.goal_center - np.asarray(ego[:2], dtype=float)
        d = float(np.linalg.norm(r))
        prox = max(0.0, 1.0 - d / self.shaping_radius)
        if prox <= 0.0:
            return 0.0
        e_cross = float(r[0] * (-np.sin(self.theta_c)) + r[1] * np.cos(self.theta_c))   # θ_c 法向横向偏差
        prox_lat = max(0.0, 1.0 - abs(e_cross) / self.xtrack_radius)
        if prox_lat <= 0.0:
            return 0.0
        return self.well_X * prox * prox_lat

    def _phi_park(self, ego) -> float:
        """想法B 终端保速势 Φ_park(s)=well_park·prox_radial(‖r‖;R_park)·speed_frac(v;V_target)（纯状态函数→Ng PBRS 策略不变·`03` L109）。
        prox_radial=max(0,1−‖r‖/R_park)：仅目标近场给梯度（远场=0·不碰避碰段·同 Φ_A/Φ_xtrack 哲学·R_park 独立默认 400m）；
        speed_frac=min(1,v/V_target)：近目标保住机动速度的势·封顶 V_target（成功局 ~3.5-4.5·不催暴速/防 overshoot）。
        机制：近目标 r_goal 进展奖励衰减→船减速停短（v→0 stopped 带外·L109 数据指靶）；Φ_park 给"近目标保速"梯度·
        到达 is_reached 不要求停→保速移动中位置/朝向进区即到达·不碍到达。well_park=0 → 恒返 0（→ Φ_park 全 0 → 逐位等价）。
        ⚠️ 纯状态势（位置+速度·非动作）→ PBRS 策略不变（与 well_B/well_X 同 Ng1999 框架·telescoping 守恒）·不改最优只导学习。"""
        if self.well_park <= 0.0:
            return 0.0
        d = float(np.linalg.norm(np.asarray(ego[:2], dtype=float) - self.goal_center))
        prox = max(0.0, 1.0 - d / self.park_radius)
        if prox <= 0.0:
            return 0.0
        v = float(ego[3])
        speed_frac = min(1.0, max(0.0, v / self.park_v_target))   # 封顶 V_target·非负（v 物理≥0·防御 clamp）
        return self.well_park * prox * speed_frac

    def reset(self, ego_state) -> None:
        """记录初始位置（r_goal 用），每个 episode 开始时调用。"""
        ego = np.asarray(ego_state, dtype=float)
        if ego.shape != (4,):
            raise ValueError(f"ego_state 应为 4 维 [px,py,θ,v]，得到 {ego.shape}")
        if not np.all(np.isfinite(ego)):  # 与 step 对称的 fail-fast（独立复核 MINOR）
            raise ValueError(f"ego_state 含非有限值（NaN/inf）：{ego}")
        self._prev_pos = ego[:2].copy()
        self._prev_phi = self._phi(ego)                    # 修法A：存初始 Φ_A(s0)（well_B=0 时=0=无副作用·`03` L80-续3 A-1）
        self._prev_phi_x = self._phi_xtrack(ego)           # 对症：存初始 Φ_xtrack(s0)（well_X=0 时=0=无副作用·`03` L88）
        self._prev_phi_p = self._phi_park(ego)             # 想法B：存初始 Φ_park(s0)（well_park=0 时=0=无副作用·`03` L109）

    def step(self, ego_state, obstacles, term_flags=None, emergency_used: bool = False):
        """计算一步奖励。

        ego_state     : [px, py, θ, v]
        obstacles     : list[(id, [x,y] 位置, [vx,vy] 速度)]  当前各他船（r_colregs 需速度算 v_y）
        term_flags    : dict 或 None，键 time/area/stopped/collision/goal（bool；缺省全 False）
        emergency_used: bool，本步是否调用了紧急控制器（→ c_emergency）

        返回：(reward_total: float, components: dict{sparse,colregs,goal,velocity,deviate})
        """
        ego = np.asarray(ego_state, dtype=float)
        if ego.shape != (4,):
            raise ValueError(f"ego_state 应为 4 维 [px,py,θ,v]，得到 {ego.shape}")
        if not np.all(np.isfinite(ego)):
            raise ValueError(f"ego_state 含非有限值（NaN/inf）：{ego}")
        if self._prev_pos is None:
            raise RuntimeError("step() 前必须先 reset(ego_state)")

        p_ego = ego[:2]
        theta_ego = wrap_to_pi(ego[2])
        v_ego = ego[3]

        tf = term_flags or {}
        _unknown = set(tf.keys()) - _TERM_KEYS
        if _unknown:
            raise ValueError(f"term_flags 含未知键 {_unknown}（合法键：{sorted(_TERM_KEYS)}）")

        # ── r_sparse ──
        r_sparse = (
            C_TIME * bool(tf.get("time", False))
            + C_AREA * bool(tf.get("area", False))
            + C_GOAL * bool(tf.get("goal", False))
            + C_STOPPED * bool(tf.get("stopped", False))
            + C_COLLISION * bool(tf.get("collision", False))
            + C_EMERGENCY * bool(emergency_used)
        )
        # ── 修法C 全局每步生存成本（非PBRS·真改最优·`03` L123·连续臂专属）──
        #   c_step=0 → 整块跳过 → r_sparse/total/parts 与现状【逐位等价 bit-identical】（不减、不加键）。
        #   c_step>0 → 每步无条件 r_sparse −= c_step（落进 parts["sparse"]·与三个 PBRS shaping 项严格正交·走真 reward 通道非 γΦ'−Φ）。
        if self.c_step > 0.0:
            r_sparse = r_sparse - self.c_step
        # ── 非PBRS 入库赤字滞留成本 r_dwell（非PBRS·真改最优·`03` L161/L162·连续臂专属·治 corr≈0 终端入库病）──
        #   c_dwell=0 → 整块跳过 → r_sparse/total/parts 与现状【逐位等价 bit-identical】（不减、不加键）。
        #   c_dwell>0 → 近场(‖r‖≤R_DWELL)每步 r_sparse −= c_dwell·g(s)，g=0.5·min(|e_cross|/W,1)+0.5·min(|dθ|/H,1)∈[0,1]（横向+朝向赤字各半）·
        #   e_cross 口径复用 _phi_xtrack:236 逐字（r=goal−pos·θ_c 法向）·dθ=wrap(θ−θ_c)·真入库→回合止→成本停（无正项=farm 免疫·入 r_sparse 非 γΦ'−Φ）。
        if self.c_dwell > 0.0:
            _r_dw = self.goal_center - p_ego                                    # 复用 _phi_xtrack:231 口径（r=goal_center−pos）
            _d_dw = float(np.linalg.norm(_r_dw))
            if _d_dw <= self.dwell_radius:
                _e_cross = float(_r_dw[0] * (-np.sin(self.theta_c)) + _r_dw[1] * np.cos(self.theta_c))   # θ_c 法向横向偏差·复用 _phi_xtrack:236 逐字
                _dtheta = abs(float(wrap_to_pi(theta_ego - self.theta_c)))       # 朝向赤字 |θ−θ_c|（同 _phi align 的 wrap 口径）
                _g = 0.5 * min(abs(_e_cross) / self.w_dwell, 1.0) + 0.5 * min(_dtheta / self.h_dwell, 1.0)
                r_sparse = r_sparse - self.c_dwell * _g                          # 非PBRS 赤字滞留成本（∈[−c_dwell,0]·落进 parts["sparse"]·与 3 PBRS 势正交）
            if self.b_dwell > 0.0 and bool(tf.get("goal", False)):
                r_sparse = r_sparse + self.b_dwell                              # 终端入库锚（真入库 +B_DWELL·补"回避近场躲成本"病·默认 0=无·嵌 c_dwell>0 块内）

        # ── r_goal（typo 修正版：前进→正·Rung1 系数 self.c_reach·默认=C_REACH=1.5 逐位等价）──
        d_now = float(np.linalg.norm(p_ego - self.goal_center))
        d_prev = float(np.linalg.norm(self._prev_pos - self.goal_center))
        r_goal = self.c_reach * (d_prev - d_now)

        # ── r_velocity（Rung2 泊位精修门：区内(d_now≤dock_radius)把低速地板下限 V_LOW→v_dock·降非清零·复用上方已算 d_now 零额外距离计算）──
        v_low_eff = self.v_dock if (self.dock_radius > 0.0 and d_now <= self.dock_radius) else V_LOW   # dock_radius=0 或 v_dock=V_LOW → 恒=V_LOW=bit-identical
        if v_ego > V_HIGH:
            r_velocity = C_V * (v_ego - V_HIGH)
        elif v_ego < v_low_eff:
            r_velocity = C_V * (v_low_eff - v_ego)
        else:
            r_velocity = 0.0

        # ── r_deviate ──
        d_lat = float(np.dot(p_ego - self.init_position, self.e_lat))
        r_deviate = C_DEVIATE * min(abs(d_lat), D_HULL)

        # ── r_colregs（Meyer 式26，v_y 归一化避免爆炸；对感知距离内每艘他船求和）──
        r_colregs = 0.0
        for obs_id, pos, vel in obstacles:
            p_obs = np.asarray(pos, dtype=float)
            v_obs = np.asarray(vel, dtype=float)
            if p_obs.shape != (2,) or v_obs.shape != (2,):
                raise ValueError(f"他船 {obs_id} 的 position/velocity 须为 2 维，得 {p_obs.shape}/{v_obs.shape}")
            if not (np.all(np.isfinite(p_obs)) and np.all(np.isfinite(v_obs))):
                raise ValueError(f"他船 {obs_id} 的 position/velocity 含非有限值")
            delta = p_obs - p_ego
            d = float(np.linalg.norm(delta))
            if d > self.d_sense or d < 1e-9:
                continue
            r_hat = delta / d                              # ego→obs 单位向量
            v_y = -float(np.dot(v_obs, r_hat))             # >0 = 驶向本船(approaching)
            v_y_eff = v_y / self.colregs_vel_scale         # 归一化（防爆炸，见 docstring）
            phi = wrap_to_pi(theta_ego - np.arctan2(delta[1], delta[0]))  # 右舷正/左舷负
            sector = _meyer_sector(phi)
            zeta_x = ZETA_X[sector]
            zeta_v = ZETA_V[sector][0] if v_y >= 0 else ZETA_V[sector][1]
            weight = 1.0 / (1.0 + np.exp(GAMMA_THETA_DYN * abs(phi)))
            # 指数钳到 ≤0：惩罚物理上只该随距离衰减、绝不该随距离增长 → 钳后恒有界 [−weight·α_x, 0]，
            # 杜绝 v_y 超尺度（如他船速 > 每扇区阈值）时的指数爆炸/静默 inf（独立复核 MAJOR）。
            # 正常区（|v_y| ≤ v_pm_max、scale=v_pm_max）指数本就 ≤0、不受影响。
            exponent = min((zeta_v * v_y_eff - zeta_x) * d, 0.0)
            r_colregs += -weight * ALPHA_X * np.exp(exponent)

        self._prev_pos = p_ego.copy()
        r_colregs *= self.colregs_weight                   # Base/RR 开关（4d-②）：0=Base(无 r_colregs) / 1=RR(默认、现状)
        total = r_sparse + r_colregs + r_goal + r_velocity + r_deviate
        parts = {
            "sparse": r_sparse,
            "colregs": r_colregs,
            "goal": r_goal,
            "velocity": r_velocity,
            "deviate": r_deviate,
        }
        # ── r_shape：进门势 PBRS（F=γΦ(s')−Φ(s)·Ng 1999 策略不变）= 修法A Φ_A(well_B) + 对症 Φ_xtrack(well_X) + 想法B Φ_park(well_park)·`03` L80-续3/续8/L88/L109 ──
        # well_B=0 且 well_X=0 且 well_park=0 → 整块跳过 → total/parts 与现状【逐位等价 bit-identical】（不加 r_shape、不加 "shape" 键）。
        # 三纯状态势之和仍严格 PBRS（线性·Ng 不变性·复审 L88 实算 telescoping 守恒 + 加性逐位）。
        if self.well_B > 0.0 or self.well_X > 0.0 or self.well_park > 0.0:
            # 真终止(area/stopped/collision/goal) → Φ(s')=0；时间截断(time·truncated) → bootstrap=真实 Φ(s')
            # （gymnasium/SB3 对 truncated 仍 bootstrap V≠0·D 复审 time-truncation：写反会在近门超时局注伪罚、破不变性、恰打在塌种子最高频路径）。
            terminated = bool(tf.get("area") or tf.get("stopped") or tf.get("collision") or tf.get("goal"))
            # 修法A 项 Φ_A（well_B<=0 时 _phi 恒 0 → r_shape 此项=0·不扰）
            phi_cur = self._phi(ego)
            phi_next = 0.0 if terminated else phi_cur
            r_shape = self.gamma * phi_next - self._prev_phi
            self._prev_phi = phi_cur                        # 下一步的 Φ_A(s)（真实值·不受 terminal 置 0 影响）
            # 对症项 Φ_xtrack（well_X>0 才计；well_X=0 时整块跳过 → r_shape/parts 与【仅修法A】逐位一致·`03` L88）
            if self.well_X > 0.0:
                phi_x_cur = self._phi_xtrack(ego)
                phi_x_next = 0.0 if terminated else phi_x_cur
                r_shape_x = self.gamma * phi_x_next - self._prev_phi_x
                self._prev_phi_x = phi_x_cur                # 下一步的 Φ_xtrack(s)（真实值·不受 terminal 置 0 影响）
                r_shape += r_shape_x
                parts["shape_xtrack"] = r_shape_x           # 单列·可消融（shape_A = shape − shape_xtrack）
            # 想法B项 Φ_park（well_park>0 才计；well_park=0 时整块跳过 → r_shape/parts 与【不含park】逐位一致·`03` L109）
            if self.well_park > 0.0:
                phi_p_cur = self._phi_park(ego)
                phi_p_next = 0.0 if terminated else phi_p_cur
                r_shape_p = self.gamma * phi_p_next - self._prev_phi_p
                self._prev_phi_p = phi_p_cur                # 下一步的 Φ_park(s)（真实值·不受 terminal 置 0 影响）
                r_shape += r_shape_p
                parts["shape_park"] = r_shape_p             # 单列·可消融（shape_A = shape − shape_xtrack − shape_park）
            total += r_shape
            parts["shape"] = r_shape                        # 总 shaping（Φ_A+Φ_xtrack）·单列·可消融·可从报告回报剥离保 Fig.8
        return float(total), parts
