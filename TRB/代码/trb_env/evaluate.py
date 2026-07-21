"""
TRB 评估管线（step4d）—— 跑 episode → Table III 口径指标
========================================================
忠实 Krasowski 2024 §VII-A（评估指标，2 对抗 agent + 主窗口逐列核 = 口径全对，03 D18）：
  到达率(%场景) / Ep.长(steps×dt 秒) / 碰撞率(%场景) / 违规次数(/局) / 紧急步%(=#ρ5步/总步)。
违规用 `ViolationCounter`（§VII-A 原始态势谓词口径，笔记③；忠实官方离线 monitor、**不豁免 is_emergency**）。

⚠️ **有盾保证 0 碰撞，但不机械保证 0 违规**（口径三方核，03 D18）：ViolationCounter 用裸态势谓词——任意策略
   （含 keep-heading）在 is_emergency 紧急转向期间若裸 `keep` 仍真，会被记 stand-on 违规（主窗口实测
   keep-heading on T-0 = 1 standon、0 giveway，确定性）。Table III Safe=0 靠【训练后 Safe agent（盾把 ρ1 约束成
   no-turn）+ §VII-C-e satisfiability 参数整定（encounter 检测早于 emergency）】，非计数豁免（官方 R_G6 无 emergency
   守卫，agent+主窗口核官方代码坐实）。→ **step4e 实跑 Safe agent 验证复现 0；若 >0 源自 emergency-while-keep
   则修盾/整定，不改 ViolationCounter（忠实官方 > 直觉）**。
⚠️ 口径待 step4 校准（02 挂起）：违规 MTL 时间窗简化（精确步窗 / 谓词敏感性）；聚合用 macro 平均（各局%再平均，
   step4 与论文核对语义）。碰撞率="每 episode 是否碰撞"%场景（agent 坐实 = Table III 口径、已对）。
⚠️ 当前 run_episode 要 `ShieldedUSVEnv`（用 _ego_vs/_obs_vs 喂 ViolationCounter）；Base/RR 无盾对照随 4d-② 扩。
"""
from __future__ import annotations

import numpy as np

import numpy as np

from .usv_colregs import RHO_EMERGENCY, ViolationCounter, _vessel_circumradius
from .usv_dynamics import make_vessel_params as _mk_vp
from .usv_env import A_NORMAL_ACCEL_MAX as _A_RANGE, A_NORMAL_OMEGA_MAX as _W_RANGE  # RL 动作箱半宽(±0.048/±0.018)=控制质量归一化尺度(CAT7)

# 本船外接圆半径（CPA 安全裕度用·disk 模型与 colregs r_m/d_safe 同口径·EGO_WIDTH=25.4）
_EGO_CIRC = _vessel_circumradius(float(_mk_vp().l), 25.4)


def _cpa_step(ego_vs, obs_vs):
    """单步两船【中心距】+ 保守【圆盘间隙】(中心距 − 双船外接圆)。disk 模型(无需他船宽·与 colregs 安全机制同口径·保守)。"""
    d_center = float(np.hypot(ego_vs.position[0] - obs_vs.position[0], ego_vs.position[1] - obs_vs.position[1]))
    clearance = d_center - _EGO_CIRC - _vessel_circumradius(obs_vs.length, None)
    return d_center, clearance


def _rho_hist():
    return {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}


def _goal_xy(env):
    """Node L CAT5 例图：目标区中心 [x,y]（从 USVEnv.obs_builder.goal_center 读·纯只读·绘 goal 标记用）。
    取不到（属性缺/异常）→ None（不崩·图按无 goal 处理）。仅 record_traj=True 时调=默认路径不触。"""
    try:
        g = env.env.obs_builder.goal_center
        return [float(g[0]), float(g[1])]
    except Exception:
        return None


def _goal_geom(env):
    """目标区几何（`03` L88·last-mile 诊断·additive·纯只读）：中心 + 朝向门区间 + 时间门区间 + 矩形参数/顶点。
    供后处理把 last-mile 失败【分解】成 位置(横向 cross-track 进不了窄 y 带) vs 朝向门 vs 时间门——定对症修法该打哪个靶。
    读 env.env.term_checker.goal.state_list[0]（GoalRegion·is_reached 的同源对象）·取不到→None（不崩·诊断缺失按无几何处理）。"""
    try:
        g0 = env.env.term_checker.goal.state_list[0]
        pos = g0.position
        geom = {
            "center": [float(pos.center[0]), float(pos.center[1])],
            "orient_lo": float(g0.orientation.start), "orient_hi": float(g0.orientation.end),
            "time_lo": int(g0.time_step.start), "time_hi": int(g0.time_step.end),
        }
        for a in ("length", "width", "orientation"):          # Rectangle 参数（若该 Shape 提供）
            if hasattr(pos, a):
                try:
                    geom["rect_" + a] = float(getattr(pos, a))
                except Exception:
                    pass
        if hasattr(pos, "vertices") and pos.vertices is not None:   # 顶点=任意朝向/形状都能做 in-rect/cross-track 判定（最稳）
            try:
                geom["vertices"] = [[float(v[0]), float(v[1])] for v in pos.vertices]
            except Exception:
                pass
        return geom
    except Exception:
        return None


def _terminal_diag(env, flags, steps):
    """last-mile 终端诊断（`03` L88·additive·不入钱图 5 列·纯只读）：
      · term_flags = 终止步 5 旗（time/area/stopped/collision/goal）→ 【直证 f_stopped】替排除法推断（final_per 原不存终止旗）；
      · end_state  = 终端 post-step 态 {px,py,psi,v,time_step} → 解 traj off-by-one（traj 记 pre-step·真正触发终止那步未入 traj）+ 给终端几何；
      · goal_geom  = 目标区几何（_goal_geom）→ 供分解 cross-track/朝向/时间门。
    取不到→各键 None（不崩）。"""
    return {
        "term_flags": ({k: bool(v) for k, v in flags.items()} if flags else None),
        "end_state": _terminal_state(env, steps),
        "goal_geom": _goal_geom(env),
    }


def _terminal_state(env, steps):
    """终端 post-step 本船态（纯只读 env._ego_vs()·统一接口·四方/连续皆适用）·取不到→None。"""
    try:
        e = env._ego_vs()
        return {"px": float(e.position[0]), "py": float(e.position[1]),
                "psi": float(e.orientation), "v": float(getattr(e, "velocity", float("nan"))),
                "time_step": int(steps)}
    except Exception:
        return None


def _traj_pose(ego_vs, obs_vs):
    """Node L CAT5 示例轨迹单步位姿：ego 位置(x,y)+朝向 + 他船位置(x,y)+朝向（他船预测窗外→None）。
    给路/迎面/追越例图用（绘 ego 航迹 + 他船航迹 + 按 ρ 着色）。纯只读取 VesselState、无副作用。"""
    return {
        "ego_x": float(ego_vs.position[0]), "ego_y": float(ego_vs.position[1]),
        "ego_psi": float(ego_vs.orientation),
        "ego_v": float(getattr(ego_vs, "velocity", float("nan"))),   # last-mile 诊断（`03` L88·additive）：逐步速度→看终端减速剖面
        "obs_x": (None if obs_vs is None else float(obs_vs.position[0])),
        "obs_y": (None if obs_vs is None else float(obs_vs.position[1])),
        "obs_psi": (None if obs_vs is None else float(obs_vs.orientation)),
    }

# 连续投影盾 source 六元 universe（usv_continuous_shield/usv_projection）的评估归口：
#   · emergency                      → 紧急步%（=调用 Alg.1 紧急控制器/ρ5；经 rho_acting==ρ5 计、与离散臂同口径）
#   · relaxed/collision_min/degenerate → 兜底步%（P=∅ 兜底：放松 COLREGs / 碰撞风险最小化 / 退化）= 下面这个集合
#   · projection / no_obstacle       → 两不计（常规合规投影 / 他船窗外无规则适用；既非紧急也非兜底，与离散 _obs_vs=None 短路同口径）
#   （reset 时 source=None 不入计数——计数只在 step 循环内。Q2/D40#1：兜底步现 evaluate 既不计紧急%也不计违规，须显式单列。）
_FALLBACK_SOURCES = frozenset({"relaxed", "collision_min", "degenerate"})

# ── Node L CAT7 控制质量（连续臂细控优势量化·additive·【绝不入钱图 5 列】）──
# 执行控制 u=(a,ω) 取 env.env.last_action（四臂同口径：离散=所选网格点 (a,ω)，连续=投影+限幅后 u_safe，
# 均经基类 USVEnv.step 设 last_action=实施值）。归一化按 RL 动作箱半宽(±0.048/±0.018)使 a/ω 可比。
_CTRL_KEYS = ("ctrl_jerk_norm_mean", "accel_incr_mean", "yaw_incr_mean", "ctrl_effort_norm_mean", "path_len_m")


def _control_quality(applied, positions):
    """从逐步执行控制 u=(a,ω) + 本船位置序列算控制质量（离散/连续可比·量化【学到的策略】平滑度）。数据不足→None。
    🔴 jerk/effort/分量【仅用正常操作步】= 执行控制在 RL 动作箱内（|a|≤0.048 且 |ω|≤0.018）的步——
       排除紧急/兜底步（红队 MAJOR L72：那些步 last_action 是【共享紧急控制器】物理满程 ±0.24/±0.03·归一化后 |û|>1·
       非 RL 策略输出·混入会污染平滑度且各臂紧急步数不同=不对称）。jerk 仅在【相邻两步都在箱内】的对上算。
    · ctrl_jerk_norm_mean : 归一化控制增量 ‖Δû‖ 均值（û=(a/A,ω/W)）=平滑度（低=更平滑）。⚠️非 tautology但有量化地板：
       离散网格最小非零增量归一化=步距/箱（accel 1/3·yaw 1/3）→ 连续仅在用【低于该粒度】的微调时才真更平滑（写作据实·勿断言"必更平滑"）。
    · accel_incr_mean / yaw_incr_mean : |Δa|/|Δω| 原单位分量（透明·非归一化）
    · ctrl_effort_norm_mean : 归一化控制幅值 ‖û‖ 均值=控制努力（正常操作步）
    · path_len_m : 轨迹总长 Σ‖Δpos‖（m）=路径效率（全步·⚠️聚合须分到达/未到达·游荡局更长，红队 MEDIUM L72）"""
    out = {k: None for k in _CTRL_KEYS}
    scale = np.array([_A_RANGE, _W_RANGE], dtype=float)
    if applied:
        U = np.asarray(applied, dtype=float)
        tol = 1e-6
        inbox = (np.abs(U[:, 0]) <= _A_RANGE + tol) & (np.abs(U[:, 1]) <= _W_RANGE + tol)  # 正常操作步（排紧急/兜底物理满程）
        Un = U / scale
        if inbox.any():
            out["ctrl_effort_norm_mean"] = round(float(np.linalg.norm(Un[inbox], axis=1).mean()), 6)
        if len(U) >= 2:
            adj = inbox[:-1] & inbox[1:]                       # 相邻两步【都在箱内】=正常操作连续对（jerk 才有意义）
            if adj.any():
                dUn = np.diff(Un, axis=0)[adj]
                dU = np.diff(U, axis=0)[adj]
                out["ctrl_jerk_norm_mean"] = round(float(np.linalg.norm(dUn, axis=1).mean()), 6)
                out["accel_incr_mean"] = round(float(np.abs(dU[:, 0]).mean()), 6)
                out["yaw_incr_mean"] = round(float(np.abs(dU[:, 1]).mean()), 6)
    if positions is not None and len(positions) >= 2:
        P = np.asarray(positions, dtype=float)
        out["path_len_m"] = round(float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()), 3)
    return out


def _agg_ctrl(per):
    """聚合控制质量（各局均值·跳 None·additive）。某指标全 None→该键 None（不编）。"""
    out = {}
    for k in _CTRL_KEYS:
        vals = [p[k] for p in per if p.get(k) is not None]
        out[k] = (round(sum(vals) / len(vals), 6) if vals else None)
    return out


# ── Step-0 进近诊断（additive·纯只读几何后处理·【绝不入钱图 5 列】）────────────────────
# 补"每局只存 3 条示例轨迹（record_traj·idx 0/1/2）"的盲区：从逐步 ego(x,y)+朝向 + goal_geom，
# 给【每一局】算 4 个标量，把失败局机制分开——
#   · 接近失败(游荡)          : min_goal_dist_m 大（从未接近目标）
#   · 位置进不了(横向 cross-track): in_box_steps==0 但 min_goal_dist_m 小（到门口进不了窄 y 带）
#   · 门口捕获失败-朝向        : in_box_steps>0 且 in_box_aligned_steps==0（进过框·朝向没对上）
#   · 门内已对齐但未达         : in_box_aligned_steps>0 且未 reached（卡时间门/未在门内定住）
# 与官方 is_reached 三分量【同源】(vertices 位置 + AngleInterval 朝向·忠实 test_usv_evaluate ㊳d)·仅位置/朝向、不看时间。
# + 速度维度（健康信号·治抖/入库任务核心·user 2026-07-09"各指标都得健康别遗漏"）：
#   · speed_at_min_ms : 离目标最近那步的速度 → 低=稳减速入库(健康) / 高=猛冲过头(病·L168 "fly-by" 就靠它判)
#   · max_speed_ms    : 全程最高速 → 猛冲的上界
#   · speed_reversals : 全程加/减速【来回切换】次数(|Δv|>0.1m/s 才计·滤噪) → 高=速度摆动/来回不稳(病)·低=单调减速入库(健康)。
#     ⚠️【不可离线导出】=全程速度序列出局即弃(只留这几个标量)·须训练时当场记·GAP 审计最高优先([[monitor-all-metrics-not-single]])。
_APPROACH_KEYS = ("min_goal_dist_m", "heading_err_at_min_deg", "in_box_steps", "in_box_aligned_steps",
                  "speed_at_min_ms", "max_speed_ms", "speed_reversals")


def _ang_diff(a, b):
    """最短角差 wrap 到 [-π,π]（禁 naive 减法·±π 会错）。"""
    return float(np.arctan2(np.sin(a - b), np.cos(a - b)))


def _ang_in_gate(psi, lo, hi):
    """psi 是否在朝向门 [lo,hi]（AngleInterval 同款·忠实官方 is_reached / test ㊳d 的 _ang_in·窗宽<π）。"""
    d = _ang_diff(psi, lo)
    w = _ang_diff(hi, lo)
    return bool(-1e-9 <= d <= w + 1e-9)


def _in_rect(px, py, verts):
    """点是否在凸多边形（目标矩形·闭合顶点环 verts[-1]==verts[0]）内（含边界）。叉积同号法·无 shapely 依赖·winding 无关。"""
    sign = 0
    for i in range(len(verts) - 1):
        x1, y1 = verts[i]; x2, y2 = verts[i + 1]
        cr = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        if cr > 1e-9:
            s = 1
        elif cr < -1e-9:
            s = -1
        else:
            continue                                          # 边上→不否决（含边界）
        if sign == 0:
            sign = s
        elif s != sign:
            return False
    return True


def _approach_diag(positions, headings, geom, speeds=None):
    """从逐步 ego 位置/朝向(+可选速度) + goal_geom 算进近诊断标量（见上方注释）。
    goal_geom 缺(None)/缺 center → 各键 None（不崩·与 _control_quality/_terminal_diag 同款优雅降级）。
    朝向门缺(orient_lo/hi 任一 None)→ heading_err/in_box_aligned 记 None（不编）·in_box_steps 仍算（纯位置）。
    speeds 缺(None)/长度不符 → speed_at_min/max_speed 记 None（未知≠0）。
    ⚠️ 假设 orient_lo≤orient_hi 且门宽<π（忠实官方 AngleInterval / is_reached·基准 ±0.17≪π）。
    整体 try/except→各键 None（与 _terminal_state/_goal_geom 同款容错·诊断绝不 abort 一局成功 eval）。"""
    out = {k: None for k in _APPROACH_KEYS}
    try:
        if not geom or geom.get("center") is None or positions is None or len(positions) == 0:
            return out
        cx, cy = geom["center"]
        P = np.asarray(positions, dtype=float)
        d = np.hypot(P[:, 0] - cx, P[:, 1] - cy)
        imin = int(np.argmin(d))
        out["min_goal_dist_m"] = round(float(d[imin]), 3)
        lo, hi = geom.get("orient_lo"), geom.get("orient_hi")
        _H_ok = headings is not None and len(headings) == len(positions)
        if _H_ok and lo is not None and hi is not None:
            out["heading_err_at_min_deg"] = round(abs(np.degrees(_ang_diff(headings[imin], (lo + hi) / 2.0))), 2)
        if speeds is not None and len(speeds) == len(positions):   # 速度维度：最近点速度(猛冲vs减速) + 全程峰值 + 摆动次数
            _S = np.asarray(speeds, dtype=float)
            if not np.any(np.isnan(_S)):
                out["speed_at_min_ms"] = round(float(_S[imin]), 3)
                out["max_speed_ms"] = round(float(_S.max()), 3)
                _dv = np.diff(_S)
                _sig = _dv[np.abs(_dv) > 0.1]                       # 只算 |Δv|>0.1m/s 的实质变化(滤噪)
                out["speed_reversals"] = (int(np.sum(np.sign(_sig[1:]) != np.sign(_sig[:-1]))) if len(_sig) >= 2 else 0)
        verts = geom.get("vertices")
        if verts is not None and len(verts) >= 4:
            can_align = _H_ok and lo is not None and hi is not None   # 无朝向/无门→aligned 记 None（未知 ≠ 0）
            nbox = naligned = 0
            for i, (px, py) in enumerate(P):
                if _in_rect(px, py, verts):
                    nbox += 1
                    if can_align and _ang_in_gate(headings[i], lo, hi):
                        naligned += 1
            out["in_box_steps"] = int(nbox)
            out["in_box_aligned_steps"] = (int(naligned) if can_align else None)
        return out
    except Exception:
        return {k: None for k in _APPROACH_KEYS}


def run_episode(env, policy, *, seed=None, deterministic=True, max_steps=10_000, obs_transform=None, record_traj=False):
    """跑一个 episode，返回单局 Table III 口径指标。

    record_traj : Node L CAT5 示例轨迹。**默认 False = 逐位不变**（不执行任何记录块）；True 时返回 dict 多
                  `traj` 键（逐步 ego/他船位姿 + ρ）供例图（给路/迎面/追越）。仅给少数代表场景开（evaluate 的 traj_idxs）。

    env    : ShieldedUSVEnv（有 _ego_vs / _obs_vs / action_masks / step_idx）
    policy : 有 .predict(obs, action_masks=, deterministic=) → (action,_) 的 sb3 模型；
             或 callable(obs, mask) -> int（基线策略）。
    obs_transform : 可选 obs→obs 变换。**VecNormalize 训练的模型必须传 `train.make_obs_transform(venv)`** 做
                    obs 归一化，否则策略在原始物理尺度上看错分布 → 动作错、Table III 失真。原始状态仍喂 VC（违规口径不变）。
    违规计数喂 @0..@k 完整轨迹（含终止后最终状态 @k，忠实官方离线 monitor；见循环末注释）。
    """
    obs, info = env.reset(seed=seed)
    vc = ViolationCounter()
    reached = collided = False
    emergency_steps = steps = 0
    rho_hist = _rho_hist()                                    # Node L CAT2：ρ 态势分布（encounter profile + 紧急 profile）
    cpa_center = cpa_clear = float("inf")                     # Node L CAT3：最近接近距离（安全裕度证据）
    cpa_step = -1
    traj = [] if record_traj else None                        # Node L CAT5：示例轨迹（仅 record_traj=True 累积）
    applied = []                                              # CAT7：逐步执行控制 u=(a,ω)（env.env.last_action·全局记=控制质量）
    positions = [np.asarray(env._ego_vs().position, dtype=float)]  # CAT7：本船位置序列（起点+逐步）→ 路径长
    flags = None                                              # last-mile 诊断（`03` L88）：循环后持终止步 5 旗（直证 f_stopped）
    while steps < max_steps:
        s_obs = env._obs_vs()                                 # 违规计数走真实轨迹（§VII-A 原始谓词）
        if s_obs is not None:                                 # 他船预测窗外→None→无态势可计（与终止 @k 同款守护，跨场景鲁棒；active 相遇靠 finalize）
            ego_vs = env._ego_vs()
            vc.step(ego_vs, s_obs)
            _dc, _cl = _cpa_step(ego_vs, s_obs)               # CAT3：disk 模型中心距 + 保守圆盘间隙
            if _dc < cpa_center:
                cpa_center, cpa_clear, cpa_step = _dc, _cl, steps
        mask = env.action_masks()
        a_obs = obs if obs_transform is None else obs_transform(obs)   # VecNormalize obs 归一化（评估须与训练同款）
        if hasattr(policy, "predict"):
            action, _ = policy.predict(a_obs, action_masks=mask, deterministic=deterministic)
            action = int(action)
        else:
            action = int(policy(a_obs, mask))
        _pose = _traj_pose(env._ego_vs(), s_obs) if record_traj else None   # CAT5：pre-step 位姿（动作所处状态）
        obs, _r, term, trunc, info = env.step(action)
        steps += 1
        _la = getattr(env.env, "last_action", None)            # CAT7：本步执行控制（离散=所选网格 (a,ω)）；env 无 last_action→优雅降级 None
        if _la is not None:
            applied.append(np.asarray(_la, dtype=float))
        positions.append(np.asarray(env._ego_vs().position, dtype=float))  # CAT7：post-step 位置
        # 紧急步%口径 = rho_acting（ρ@t，本步动作所处态势）——pre-step，与连续臂/C_EMERGENCY reward 同源（D40 #1）；
        # info["rho"] 离散侧是 post-step ρ@(t+1)（配 next mask），用它计会跨臂差 1 步污染钱图；rho_acting 缺则回退 rho。
        _ract = info.get("rho_acting", info.get("rho"))
        if _ract in rho_hist:
            rho_hist[_ract] += 1
        if _ract == RHO_EMERGENCY:
            emergency_steps += 1
        if record_traj:
            traj.append({**_pose, "step": steps - 1, "rho": _ract})   # CAT5：位姿 + 本步 ρ@t
        flags = info["flags"]
        collided = collided or bool(flags["collision"])
        reached = reached or bool(flags["goal"])
        if term or trunc:
            break
    # 终止后最终状态 @k：上面循环只把 @0..@(k-1) 喂给 ViolationCounter（每步动作之前），@k 漏掉。
    # 官方离线 monitor 对完整记录轨迹逐状态求值（含终止状态）→ 补喂 @k 才忠实（2026-06-11 审核 MAJOR 修复）。
    # 窗外他船（_obs_vs=None）→ @k 无态势可计、finalize 兜底未解除相遇，故 None 时安全跳过（ViolationCounter.step 不收 None）。
    s_obs_final = env._obs_vs()
    if s_obs_final is not None:
        ego_vs = env._ego_vs()
        vc.step(ego_vs, s_obs_final)
        _dc, _cl = _cpa_step(ego_vs, s_obs_final)
        if _dc < cpa_center:
            cpa_center, cpa_clear, cpa_step = _dc, _cl, steps
    vc.finalize()
    out = {
        "reached": reached,
        "collided": collided,
        "violations": vc.standon_violations + vc.giveway_violations,
        "emergency_pct": 100.0 * emergency_steps / max(steps, 1),
        "ep_len_s": steps * env.env.dt,
        "steps": steps,
        # ── Node L CAT2/3 诊断（additive·不入钱图 5 列·供单局明细落盘 + 统计 + by-encounter 分解）──
        "standon_violations": vc.standon_violations,          # CAT2 违规分解（stand-on R_G6）
        "giveway_violations": vc.giveway_violations,          # CAT2 违规分解（give-way R_G3-5）
        "rho_hist": rho_hist,                                 # CAT2 ρ 态势分布（步数 per ρ0-5）→ encounter-type
        "cpa_center_m": (None if cpa_center == float("inf") else round(cpa_center, 3)),   # CAT3 最近中心距
        "cpa_clearance_m": (None if cpa_clear == float("inf") else round(cpa_clear, 3)),  # CAT3 保守圆盘间隙(中心距−双圆)
        "cpa_step": cpa_step,
    }
    out.update(_control_quality(applied, positions))          # CAT7 控制质量（additive·5 列钱图不受影响）
    out.update(_terminal_diag(env, flags, steps))             # last-mile 诊断（`03` L88·additive·term_flags/end_state/goal_geom·钱图 5 列不受影响）
    if record_traj:                                           # Node L CAT5：仅请求时带 traj/goal 键（默认不带→返回 dict 逐位不变）
        out["traj"] = traj
        out["goal"] = _goal_xy(env)                           # 目标区中心（例图绘 goal 标记·取不到→None）
    return out


def evaluate(env_factory, policy, scenarios, *, seed=0, deterministic=True, obs_transform=None, traj_idxs=None):
    """对一组场景各跑一 episode，聚合成 Table III 口径。

    env_factory(scenario, planning_problem) -> ShieldedUSVEnv
    scenarios = [(scenario, planning_problem), ...]
    obs_transform : 透传 run_episode（VecNormalize 模型须传 `train.make_obs_transform(venv)`）。
    traj_idxs : Node L CAT5 — 要记示例轨迹的场景索引集合（默认 None=都不记=钱图路径逐位不变）。
    返回 (聚合 dict, 每局 list)。
    """
    _tset = set(traj_idxs) if traj_idxs is not None else None
    per = [{**run_episode(env_factory(sc, pp), policy, seed=seed, deterministic=deterministic,
                          obs_transform=obs_transform, record_traj=(_tset is not None and i in _tset)),
            "scenario_idx": i}   # CAT2：单局明细可追溯到场景
           for i, (sc, pp) in enumerate(scenarios)]
    n = max(len(per), 1)
    agg = {
        "n": len(per),
        "到达率%": 100.0 * sum(p["reached"] for p in per) / n,
        "碰撞率%": 100.0 * sum(p["collided"] for p in per) / n,
        "违规次数/局": sum(p["violations"] for p in per) / n,
        "紧急步%": sum(p["emergency_pct"] for p in per) / n,
        "Ep长s": sum(p["ep_len_s"] for p in per) / n,
    }
    agg.update(_agg_ctrl(per))                                # CAT7 控制质量聚合（additive·钱图 5 列不变）
    return agg, per


def run_episode_continuous(env, model, *, seed=None, deterministic=True, max_steps=10_000, obs_transform=None, record_traj=False):
    """连续臂（Continuous-safe = SAC + ContinuousProjectionEnv）单局评估，返回 Table III 口径指标（Node C C1）。

    record_traj : Node L CAT5 示例轨迹（**默认 False = 逐位不变**）；True 时返回多 `traj` 键（逐步 ego/他船位姿 + ρ + source）。

    env   : ContinuousProjectionEnv（_ego_vs / _obs_vs / .env.dt / info["rho_acting","source"]）。
    model : SAC（`.predict(obs, deterministic=) → (action,_)`，**连续动作、不调 action_masks**——离散 run_episode 吃 mask、
            连续 env 无 action_masks() 会 AttributeError，故连续臂【必须】走本函数，禁喂离散 run_episode。
    obs_transform : VecNormalize 训练的模型须传 `train.make_obs_transform(venv)`（同离散，否则策略看错分布）。

    与离散 run_episode 同口径（四方平价）：① 违规走真实轨迹 @0..@k 喂 ViolationCounter（pre-step 状态）；
      ② **紧急步% = `rho_acting`==ρ5（pre-step ρ@t，与离散臂 rho_acting/C_EMERGENCY 同源，D40 #1）**——
         注：连续臂 rho_acting==ρ5 ⟺ source=='emergency'（实测 T-0 65/65 一致），但**统一以 rho_acting 为跨臂口径键**
         （离散臂无 source、只有 rho_acting）；故紧急%不按 source、按 rho_acting，与离散臂逐字节同口径；
      ③ reached/collided = flags OR。**额外** `fallback_pct` = P=∅ 兜底步（source∈relaxed/collision_min/degenerate）占比
      （连续臂独有诊断列，Q2/D40#1：这些步既非紧急也非常规、单列防被 ρ5 口径漏计；projection/no_obstacle 两不计）。
    """
    obs, info = env.reset(seed=seed)
    if hasattr(model, "bind_env"):        # P1 朴素基线（PursuitNaivePolicy）需读 env 原始几何算方位角；SB3 模型无此方法→跳过（RL 路径 bit-identical）
        model.bind_env(env)
    vc = ViolationCounter()
    reached = collided = False
    emergency_steps = fallback_steps = steps = 0
    rho_hist = _rho_hist()                                    # CAT2 ρ 态势分布
    cpa_center = cpa_clear = float("inf"); cpa_step = -1      # CAT3 最近接近距离
    src_counts = {"projection": 0, "emergency": 0, "relaxed": 0, "collision_min": 0,
                  "degenerate": 0, "no_obstacle": 0}         # CAT4 兜底链逐档分解（别合并）
    em_modes = {"ahead": 0, "stern": 0, "base": 0}           # CAT4 紧急控制器模式分布
    corrections = []                                         # CAT4 投影修正量 ‖u_applied−u_desired‖（盾介入步）
    traj = [] if record_traj else None                       # CAT5：示例轨迹（仅 record_traj=True 累积）
    applied = []                                              # CAT7：逐步执行控制 u=(a,ω)（=投影+限幅后 u_safe·env.env.last_action）
    positions = [np.asarray(env._ego_vs().position, dtype=float)]  # CAT7：本船位置序列（起点+逐步）→ 路径长
    headings = [float(env._ego_vs().orientation)]             # Step-0 进近诊断：本船朝向序列（与 positions 同索引·起点+逐步）→ heading_err/in_box_aligned
    speeds = [float(getattr(env._ego_vs(), "velocity", float("nan")))]   # Step-0 速度维度：本船速度序列（与 positions 同索引）→ speed_at_min/max_speed（猛冲vs减速）
    flags = None                                              # last-mile 诊断（`03` L88）：循环后持终止步 5 旗（直证 f_stopped）
    while steps < max_steps:
        s_obs = env._obs_vs()                                 # 违规走真实轨迹（§VII-A 原始谓词，与离散同口径）
        if s_obs is not None:
            ego_vs = env._ego_vs()
            vc.step(ego_vs, s_obs)
            _dc, _cl = _cpa_step(ego_vs, s_obs)               # CAT3
            if _dc < cpa_center:
                cpa_center, cpa_clear, cpa_step = _dc, _cl, steps
        a_obs = obs if obs_transform is None else obs_transform(obs)
        action, _ = model.predict(a_obs, deterministic=deterministic)   # 连续动作，不传 action_masks（SAC 无 mask）
        _pose = _traj_pose(env._ego_vs(), s_obs) if record_traj else None   # CAT5：pre-step 位姿
        obs, _r, term, trunc, info = env.step(np.asarray(action, dtype=float))
        steps += 1
        _la = getattr(env.env, "last_action", None)            # CAT7：本步执行控制（连续=投影+限幅后 u_safe）；env 无 last_action→优雅降级 None
        if _la is not None:
            applied.append(np.asarray(_la, dtype=float))
        positions.append(np.asarray(env._ego_vs().position, dtype=float))  # CAT7：post-step 位置
        headings.append(float(env._ego_vs().orientation))     # Step-0：post-step 朝向（与 positions 同索引对齐）
        speeds.append(float(getattr(env._ego_vs(), "velocity", float("nan"))))   # Step-0：post-step 速度（同索引）
        _ract = info.get("rho_acting", info.get("rho"))
        if _ract in rho_hist:
            rho_hist[_ract] += 1
        if _ract == RHO_EMERGENCY:                                     # 紧急步%：pre-step ρ@t（同离散臂 D40#1）
            emergency_steps += 1
        _src = info.get("source")
        if record_traj:
            traj.append({**_pose, "step": steps - 1, "rho": _ract, "source": _src})   # CAT5：位姿 + ρ@t + source（连续臂多记盾归口）
        if _src in src_counts:
            src_counts[_src] += 1
        if _src in _FALLBACK_SOURCES:                                  # P=∅ 兜底步（连续臂诊断、Q2）
            fallback_steps += 1
        _emode = info.get("emergency_mode")
        if _emode in em_modes:
            em_modes[_emode] += 1
        # CAT4 投影修正量 ‖u_applied − u_desired‖（=action aliasing，论文头号诊断）：仅有规则适用步(非 no_obstacle/非 unshielded[P0 无盾臂恒 0·排除防摊薄·L146])
        if _src is not None and _src not in ("no_obstacle", "unshielded") and "u_desired" in info and "u_applied" in info:
            ud = np.asarray(info["u_desired"], dtype=float); ua = np.asarray(info["u_applied"], dtype=float)
            corrections.append(float(np.linalg.norm(ua - ud)))
        flags = info["flags"]
        collided = collided or bool(flags["collision"])
        reached = reached or bool(flags["goal"])
        if term or trunc:
            break
    # 终止后 @k：与离散 run_episode 同款补喂（忠实官方离线 monitor；窗外 None 安全跳过）
    s_obs_final = env._obs_vs()
    if s_obs_final is not None:
        ego_vs = env._ego_vs()
        vc.step(ego_vs, s_obs_final)
        _dc, _cl = _cpa_step(ego_vs, s_obs_final)
        if _dc < cpa_center:
            cpa_center, cpa_clear, cpa_step = _dc, _cl, steps
    vc.finalize()
    _corr = np.asarray(corrections, dtype=float) if corrections else None
    out = {
        "reached": reached,
        "collided": collided,
        "violations": vc.standon_violations + vc.giveway_violations,
        "emergency_pct": 100.0 * emergency_steps / max(steps, 1),
        "fallback_pct": 100.0 * fallback_steps / max(steps, 1),
        "ep_len_s": steps * env.env.dt,
        "steps": steps,
        # ── Node L CAT2/3/4 诊断（additive·不入钱图列）──
        "standon_violations": vc.standon_violations,
        "giveway_violations": vc.giveway_violations,
        "rho_hist": rho_hist,
        "cpa_center_m": (None if cpa_center == float("inf") else round(cpa_center, 3)),
        "cpa_clearance_m": (None if cpa_clear == float("inf") else round(cpa_clear, 3)),
        "cpa_step": cpa_step,
        "source_counts": src_counts,                          # CAT4 六档 source 分解（projection/emergency/relaxed/collision_min/degenerate/no_obstacle）
        "emergency_modes": em_modes,                          # CAT4 紧急模式 ahead/stern/base
        # CAT4 投影修正量统计（action aliasing：盾介入最小、策略学到接近合规的核心证据）
        "proj_correction_mean": (None if _corr is None else round(float(_corr.mean()), 6)),
        "proj_correction_max": (None if _corr is None else round(float(_corr.max()), 6)),
        "proj_correction_p95": (None if _corr is None else round(float(np.percentile(_corr, 95)), 6)),
        "proj_correction_zero_frac": (None if _corr is None else round(float((_corr < 1e-9).mean()), 4)),  # 盾零介入步占比
        "n_shield_steps": (0 if _corr is None else int(_corr.size)),
    }
    out.update(_control_quality(applied, positions))          # CAT7 控制质量（additive·5 列钱图不受影响）
    out.update(_terminal_diag(env, flags, steps))             # last-mile 诊断（`03` L88·additive·term_flags/end_state/goal_geom·钱图 5 列不受影响）
    out.update(_approach_diag(positions, headings, out.get("goal_geom"), speeds=speeds))  # Step-0 进近诊断（additive·复用已算 goal_geom·失败局机制分类+速度·钱图不受影响）
    if record_traj:                                           # Node L CAT5：仅请求时带 traj/goal 键（默认不带→返回 dict 逐位不变）
        out["traj"] = traj
        out["goal"] = _goal_xy(env)                           # 目标区中心（例图绘 goal 标记·取不到→None）
    return out


def evaluate_continuous(env_factory, model, scenarios, *, seed=0, deterministic=True, obs_transform=None, traj_idxs=None):
    """连续臂（Continuous-safe）多场景聚合，Table III 口径 + 兜底步%（Node C C1）。

    env_factory(scenario, planning_problem) -> ContinuousProjectionEnv；其余同 evaluate（四方同口径聚合）。
    traj_idxs : Node L CAT5 — 要记示例轨迹的场景索引集合（默认 None=都不记=钱图路径逐位不变）。
    """
    _tset = set(traj_idxs) if traj_idxs is not None else None
    per = [{**run_episode_continuous(env_factory(sc, pp), model, seed=seed, deterministic=deterministic,
                                    obs_transform=obs_transform, record_traj=(_tset is not None and i in _tset)),
            "scenario_idx": i}   # CAT2 可追溯
           for i, (sc, pp) in enumerate(scenarios)]
    n = max(len(per), 1)
    agg = {
        "n": len(per),
        "到达率%": 100.0 * sum(p["reached"] for p in per) / n,
        "碰撞率%": 100.0 * sum(p["collided"] for p in per) / n,
        "违规次数/局": sum(p["violations"] for p in per) / n,
        "紧急步%": sum(p["emergency_pct"] for p in per) / n,
        "兜底步%": sum(p["fallback_pct"] for p in per) / n,
        "Ep长s": sum(p["ep_len_s"] for p in per) / n,
    }
    agg.update(_agg_ctrl(per))                                # CAT7 控制质量聚合（additive·钱图 5 列不变）
    return agg, per
