"""Node 3：连续投影盾的【最小闭环】验证 harness（Phase 2 / 通过门 2）。

蓝图第二层 §6「最小可证明目标」：单他船 + 给路态势 + ω≤−ω_turn + τ=Δt 单步 +
路线1 QP 投影 → 验"非 SAC 简单策略输出被投影后整段轨迹零违规/零碰撞/能到达"
= 方案核心假设成立（此时经验性零碰撞 = 档位A 措辞）。

**只新增文件、不碰共享 env**。复用已验证件：dyn_step(动力学) / predict_state_cv(他船 CV) /
ColregsStatechart+safe_action(连续投影盾，usv_projection) / ViolationCounter(违规计数) /
_ego_footprint(裸船体 shapely 碰撞，usv_termination)。

集成必做项（03 D32/D33/L36）落地：① 每 episode 起调 proj.reset()；② 每决策步只调一次
safe_action；③ u_exec 恒 ∈box（safe_action 在 box-match 下保证）+ 断言；④ 评估按 source 归类。

⚠️ 档位A 经验性：零碰撞靠 d_safe 大裕度，非可证明（provable 须档位B/Phase4）。
"""
import numpy as np

from .usv_dynamics import make_vessel_params, step as dyn_step
from .usv_colregs import VesselState, predict_state_cv, ViolationCounter
from .usv_projection import ContinuousColregsProjection
from .usv_termination import _ego_footprint

DT = 10.0
EGO_LENGTH, EGO_WIDTH = 175.0, 25.4     # SR108（与离散基线、投影盾同口径）


def make_head_on_scenario(
    goal_x: float = 4000.0,
    obs_x0: float = 3000.0,
    v_ego0: float = 5.0,
    v_obs: float = 5.0,
    obs_length: float = 175.0,
    obs_width: float = 25.4,
):
    """构造单他船【对遇(head-on)】场景：本船朝东(+x)驶向正东目标；他船在正前方、朝西(−x)迎面恒速逼近。

    几何 = 同一航道(y≈0) 反向接近 → 无避让必撞（负对照）。他船起点 obs_x0 ≫ d_safe(~562m)、起始 ρ0 无冲突。
    ⚠️ 实测（复审坐实，D35）：本几何下 statechart **直接 ρ0→ρ5 emergency、ρ2 head-on give-way 不触发**——
       is_emergency(180s 集合预测) 早于 persistent_head_on 持续窗触发 = 紧急主导固有性质。故本场景验"loop 闭合
       （到达+零碰撞）经紧急兜底路径"；【给路投影】这条核心卖点路径由 make_crossing_giveway_scenario 验。
    返回 dict：ego0[px,py,θ,v] / obs0(VesselState) / goal(xy) / params。
    """
    ego0 = np.array([0.0, 0.0, 0.0, float(v_ego0)], dtype=float)        # 朝东 θ=0
    obs0 = VesselState(position=np.array([float(obs_x0), 0.0]),
                       orientation=np.pi, velocity=float(v_obs),         # 朝西 θ=π（迎面）
                       length=float(obs_length))
    return {
        "ego0": ego0,
        "obs0": obs0,
        "obs_width": float(obs_width),
        "goal": np.array([float(goal_x), 0.0]),
        "params": make_vessel_params(),
    }


def make_crossing_giveway_scenario(
    goal_x: float = 5000.0,
    obs_pos=(2500.0, -1700.0),
    obs_heading_deg: float = 90.0,
    v_ego0: float = 5.0,
    v_obs: float = 5.0,
    obs_length: float = 120.0,
    obs_width: float = 43.0,
    target_v: float = 3.0,
):
    """构造单他船【交叉(crossing)给路·干净】场景：本船朝东驶向目标；他船从右舷(南，−y)向北穿越本船航线。

    ⚠️【参数全是搜索选定的合成值、非 fact-based 从论文导出】(2026-06-16 主窗口 576 场景细搜)。
    令真实 statechart 进 ρ3 crossing give-way → safe_action 返回 source=projection 的合规右转(ω≤−ω_turn) =
    验【给路投影】核心卖点路径（通过门2 准则③）；整段【零紧急/零违规/零碰撞/到达】= 字面通过门2 达成。
    ⚠️【实测张力，复审 Agent 校正 D35 over-claim】：干净给路episode的给路【只 engage 1 步】+ 他船远擦肩(min_gap~1.2km)
       ——本框架里【持续给路】必触发紧急兜底(emergency)、进而 L20 满足性残余违规；故干净例必是【温和冲突】。
       字面"整段零违规"对给路相遇【可达】(此即反例、推翻旧 D35"达不到")，但代价=给路 engage 短暂。
    target_v=3：本船趋速（搜索定，慢一点让给路在紧急前化解）。返回 dict 含 target_v（run_min_loop 用）。
    """
    ego0 = np.array([0.0, 0.0, 0.0, float(v_ego0)], dtype=float)
    obs0 = VesselState(position=np.array([float(obs_pos[0]), float(obs_pos[1])]),
                       orientation=np.radians(float(obs_heading_deg)), velocity=float(v_obs),
                       length=float(obs_length))
    return {
        "ego0": ego0,
        "obs0": obs0,
        "obs_width": float(obs_width),
        "goal": np.array([float(goal_x), 0.0]),
        "params": make_vessel_params(),
        "target_v": float(target_v),
    }


def simple_policy(ego, goal_xy, p, target_v=5.0, k_head=0.5, k_acc=0.3):
    """非 SAC 比例控制：转艏对准目标 + 趋目标速度 → u_desired=[a,ω]（已 clip 进 box）。

    无盾时该策略直驶目标 = 朝东穿过迎面他船 = 必撞（证场景是真冲突）。
    """
    dx, dy = goal_xy[0] - ego[0], goal_xy[1] - ego[1]
    desired_head = np.arctan2(dy, dx)
    head_err = (desired_head - ego[2] + np.pi) % (2 * np.pi) - np.pi   # wrap 到 [−π,π]
    omega = float(np.clip(k_head * head_err, -p.w_max, p.w_max))
    acc = float(np.clip(k_acc * (target_v - ego[3]), -p.a_max, p.a_max))
    return np.array([acc, omega], dtype=float)


def _collided(ego, obs_vs, obs_width):
    """真实裸船体碰撞 = 两船 shapely 旋转矩形相交（非膨胀盘；忠实 usv_termination 口径）。"""
    f_ego = _ego_footprint(ego[0], ego[1], ego[2], EGO_LENGTH, EGO_WIDTH)
    f_obs = _ego_footprint(obs_vs.position[0], obs_vs.position[1], obs_vs.orientation,
                           obs_vs.length, obs_width)
    return bool(f_ego.intersects(f_obs))


def _reached(ego, goal_xy, tol=200.0):
    """到达 = 本船位置进目标容差圈（最小闭环用位置容差；真 GoalRegion[位置∧朝向±10°∧时间] 留 Phase 3 接 env）。"""
    return bool(np.linalg.norm(ego[:2] - goal_xy) <= tol)


def run_min_loop(scenario, use_shield=True, max_steps=200, target_v=5.0, dt=DT, proj=None, verbose=False):
    """跑一局最小闭环。use_shield=True 走连续投影盾 safe_action；False=负对照直驶。

    proj : 可选，传入【复用的】ContinuousColregsProjection（测试 proj.reset 跨 episode 契约用）；
           None=每 episode 新建。无论哪种 **episode 起都调 proj.reset()**（集成必做①、D33 B-EC-CROSS-EPISODE）。
    返回 dict：reached / collided / violations / source_counts / gw_proj_* / steps / 轨迹/最近距离。
    """
    ego = scenario["ego0"].copy()
    obs0 = scenario["obs0"]
    obs_width = scenario["obs_width"]
    goal = scenario["goal"]
    p = scenario["params"]
    target_v = scenario.get("target_v", target_v)   # 场景可带推荐趋速（给路干净例需慢一点，见 make_crossing_giveway_scenario）

    if use_shield:
        if proj is None:
            proj = ContinuousColregsProjection(p.a_max, p.w_max)    # 默认真 ColregsStatechart
        proj.reset()                                            # 集成必做①：episode 起 reset（含复用 proj，清 EC/_prev_rho）

    vc = ViolationCounter()
    source_counts = {}
    rho_counts = {}
    gw_proj_steps = 0           # 给路态势(ρ2/3/4) 经 projection 输出的步数
    gw_proj_compliant = True    # 这些步是否全合规右转/左转（按 give_way_dir，|ω|≥ω_turn）
    gw_proj_omegas = []         # 给路投影步的 ω（供测试独立断言方向+幅度，堵弱守护）
    reached = collided = False
    min_gap = np.inf
    traj = []
    from .usv_colregs import RHO_HEAD_ON, RHO_CROSSING, RHO_OVERTAKE
    _GIVEWAY_RHOS = (RHO_HEAD_ON, RHO_CROSSING, RHO_OVERTAKE)

    for k in range(max_steps):
        # 他船当前时刻 = CV 从 t0 外推 k·dt（恒速恒向，规则态势假设）
        obs_now = predict_state_cv(obs0, k * dt)
        ego_vs = VesselState(position=ego[:2].copy(), orientation=float(ego[2]),
                             velocity=float(ego[3]), length=EGO_LENGTH)

        # 违规计数走真实轨迹原始谓词（§VII-A 口径，同 evaluate.run_episode）
        vc.step(ego_vs, obs_now)

        # 记录最近裸船体间距（诊断）
        gap = float(np.linalg.norm(ego[:2] - obs_now.position)
                    - 0.5 * np.hypot(EGO_LENGTH, EGO_WIDTH)
                    - 0.5 * np.hypot(obs_now.length, obs_width))
        min_gap = min(min_gap, gap)

        # 碰撞 / 到达判定（在动作前判，捕捉当前态）
        if _collided(ego, obs_now, obs_width):
            collided = True
            break
        if _reached(ego, goal):
            reached = True
            break

        u_des = simple_policy(ego, goal, p, target_v=target_v)
        if use_shield:
            res = proj.safe_action(ego_vs, obs_now, u_des, dt, p)   # 必做②：每步只调一次
            u_exec = np.asarray(res.u_safe, dtype=float)
            source = res.source
            rho = int(res.rho)
            rho_counts[rho] = rho_counts.get(rho, 0) + 1
            # 必做③：u_exec 恒 ∈box（box-match 下 safe_action 保证）
            assert abs(u_exec[0]) <= p.a_max + 1e-9 and abs(u_exec[1]) <= p.w_max + 1e-9, \
                f"u_exec={u_exec} 越 box（违反 safe_action ∈box 契约）"
            # 准则③：给路态势经 projection 输出 → 须合规（give_way_dir 方向 + |ω|≥ω_turn）
            if rho in _GIVEWAY_RHOS and source == "projection":
                gw_proj_steps += 1
                om = float(u_exec[1])
                gw_proj_omegas.append(om)
                ok = (om <= -proj.omega_turn + 1e-9) if res.give_way_dir == "right" \
                    else (om >= proj.omega_turn - 1e-9) if res.give_way_dir == "left" else False
                if not ok:
                    gw_proj_compliant = False
        else:
            u_exec = u_des
            source = "none"
        source_counts[source] = source_counts.get(source, 0) + 1
        traj.append((float(ego[0]), float(ego[1]), float(ego[2]), source))

        ego = dyn_step(ego, u_exec, dt, p, clip_velocity=True)

    # M-2 修复（2026-06-16 复审）：碰撞/到达在循环顶部、动作【前】判 → 循环跑满（未提前 break）时，
    #   最后一次 dyn_step 后的终态从未被检（实测 max_steps=29 漏检、max_steps=30 才检出）。补检终态：
    #   当前两门场景都提前 reached break、不触发此路；防未来缩短 horizon/换更快场景致末步碰撞静默漏报。
    #   变异守护见 test ⑤（删本块 → max_steps=29 漏检翻 FAIL）。
    if not collided and not reached:
        obs_final = predict_state_cv(obs0, max_steps * dt)
        gap_final = float(np.linalg.norm(ego[:2] - obs_final.position)
                          - 0.5 * np.hypot(EGO_LENGTH, EGO_WIDTH)
                          - 0.5 * np.hypot(obs_final.length, obs_width))
        min_gap = min(min_gap, gap_final)
        if _collided(ego, obs_final, obs_width):
            collided = True
        elif _reached(ego, goal):
            reached = True

    vc.finalize()
    return {
        "reached": reached,
        "collided": collided,
        "violations": vc.standon_violations + vc.giveway_violations,
        "standon_v": vc.standon_violations,
        "giveway_v": vc.giveway_violations,
        "source_counts": source_counts,
        "rho_counts": rho_counts,
        "gw_proj_steps": gw_proj_steps,             # 给路态势经 projection 输出的步数
        "gw_proj_compliant": gw_proj_compliant,     # 准则③：那些步是否全合规
        "gw_proj_omegas": gw_proj_omegas,           # 给路投影步 ω（测试独立断言方向+幅度）
        "steps": k + 1,
        "min_gap": min_gap,
        "traj": traj,
    }
