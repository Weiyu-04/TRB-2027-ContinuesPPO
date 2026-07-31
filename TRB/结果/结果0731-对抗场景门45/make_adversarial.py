#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""对抗 / 让路密集场景源（通过门 4 + 门 5 · 纯评估 · 不碰 代码/）。

【为什么需要】官方 2000 场景里，本船保向保速时几乎从不真撞：实测 60 例中只有 1 例
最近中心距 <250 m，且发生在 t≈395 s（远超判据视界 120 s），最近距中位数 1293 m。
⟹ 安全动作集 P 实际上从不为空 ⟹ 兜底链（放松 / 碰撞风险最小化 / 退化 / 紧急控制器）
从未被真正触发，门 5 无法通过；certv2 终端约束也因让路落点稀少而 ≡ off，门 4 显不出价值。

【本文件做什么】以官方场景为模板，只重造他船轨迹，使「本船保向保速」与「他船恒速」
在指定时刻 T_c 真正相交（几何闭式解），从而构造出**真冲突**的让路场景。

【闭式解】把本船放在原点、艏向 +x：
    碰撞点     P   = (v_e·T_c, 0)
    他船起点   p_m = R·(cos β, sin β)          β = 他船相对本船的方位角
    他船需在 T_c 内走完 |P − p_m| ⟹ v_m = |P − p_m| / T_c
    给定 β、T_c、v_m 反解 R：
        R = T_c · [ v_e·cos β + sqrt(v_m² − v_e²·sin²β) ]      （要求 v_m ≥ v_e·|sin β|）

【态势不靠猜】生成后一律用 代码/trb_env/usv_colregs.py 里的**真实谓词**
（head_on / crossing / overtake / keep / is_emergency）复判，只保留真被判为让路态的场景。
"""
import os
import sys
import copy
import numpy as np

_TRB = os.environ.get("TRB_ROOT", "/home/user/TRB-2027-ContinuesPPO/TRB")
sys.path.insert(0, os.path.join(_TRB, "代码"))

from trb_env.usv_scenarios import load_scenario_pool          # noqa: E402
from trb_env import usv_colregs as CO                          # noqa: E402
from commonroad.scenario.state import CustomState              # noqa: E402
# 🔴 Trajectory 与 Prediction 都必须用 commonocean 的（同名类，但 setter 断言的是 commonocean 版）
from commonocean.scenario.trajectory import Trajectory         # noqa: E402
from commonocean.prediction.prediction import TrajectoryPrediction  # noqa: E402


def solve_R(v_e, v_m, beta, T_c):
    """反解他船初始距离 R，使双方恒速在 T_c 时刻同点。无解返回 None。"""
    disc = v_m ** 2 - (v_e * np.sin(beta)) ** 2
    if disc < 0:
        return None
    R = T_c * (v_e * np.cos(beta) + np.sqrt(disc))
    return R if R > 0 else None


def build_obstacle_traj(p_m0, theta_m, v_m, n_steps, dt, shape):
    """按恒速直行造他船轨迹（与官方格式一致：CustomState 列表，time_step 从 1 起）。"""
    states = []
    d = np.array([np.cos(theta_m), np.sin(theta_m)])
    for k in range(1, n_steps + 1):
        pos = p_m0 + v_m * dt * k * d
        states.append(CustomState(position=pos, orientation=float(theta_m),
                                  velocity=float(v_m), time_step=k))
    traj = Trajectory(1, states)
    return TrajectoryPrediction(traj, shape)


def make_conflict_scenario(template, beta_deg, T_c, v_m, keep_goal=True):
    """以 template=(scenario, planning_problem) 为模板，重造他船使之与本船真冲突。

    返回 (scenario, planning_problem) 或 None（几何无解）。
    """
    sc_t, pp = template
    ini = pp.initial_state
    p_e = np.array(ini.position, float)
    th_e = float(ini.orientation)
    v_e = float(ini.velocity)
    if v_e <= 0.1:
        return None

    beta = np.deg2rad(beta_deg)
    R = solve_R(v_e, v_m, beta, T_c)
    if R is None:
        return None

    # 本船坐标系 → 世界坐标系
    c, s = np.cos(th_e), np.sin(th_e)
    Rot = np.array([[c, -s], [s, c]])
    p_m0 = p_e + Rot @ (R * np.array([np.cos(beta), np.sin(beta)]))
    P_coll = p_e + v_e * T_c * np.array([c, s])
    dirv = P_coll - p_m0
    theta_m = float(np.arctan2(dirv[1], dirv[0]))

    sc = copy.deepcopy(sc_t)
    ob = sc.dynamic_obstacles[0]
    dt = float(sc.dt)
    n_steps = len(ob.prediction.trajectory.state_list)

    ob.initial_state = CustomState(position=p_m0, orientation=theta_m,
                                   velocity=float(v_m), time_step=0)
    ob.prediction = build_obstacle_traj(p_m0, theta_m, v_m, n_steps, dt,
                                        ob.obstacle_shape)
    return sc, pp


def classify(sc, pp):
    """用代码里的真实谓词判 t=0 的态势（不猜约定）。返回 dict。"""
    ob = sc.dynamic_obstacles[0]
    ini = pp.initial_state
    s_e = CO.VesselState(position=np.array(ini.position, float),
                         orientation=float(ini.orientation),
                         velocity=float(ini.velocity),
                         length=175.0)
    st = ob.initial_state
    shp = ob.obstacle_shape
    s_m = CO.VesselState(position=np.array(st.position, float),
                         orientation=float(st.orientation),
                         velocity=float(st.velocity),
                         length=float(shp.length))
    return {
        "collision_possible": bool(CO.collision_possible(s_e, s_m)),
        "head_on": bool(CO.head_on(s_e, s_m)),
        "crossing": bool(CO.crossing(s_e, s_m)),
        "overtake": bool(CO.overtake(s_e, s_m)),
        "keep": bool(CO.keep(s_e, s_m)),
        "is_emergency": bool(CO.is_emergency(s_e, s_m)),
    }


def min_center_distance(sc, pp, horizon=400.0, step=5.0):
    """本船保向保速 vs 他船恒速的最近中心距（用于确认『真会撞』）。"""
    ob = sc.dynamic_obstacles[0]
    ini = pp.initial_state
    p_e = np.array(ini.position, float); th = float(ini.orientation); v = float(ini.velocity)
    st = ob.initial_state
    p_m = np.array(st.position, float); thm = float(st.orientation); vm = float(st.velocity)
    ts = np.arange(0, horizon, step)
    de = p_e[None, :] + np.outer(ts, v * np.array([np.cos(th), np.sin(th)]))
    dm = p_m[None, :] + np.outer(ts, vm * np.array([np.cos(thm), np.sin(thm)]))
    d = np.linalg.norm(de - dm, axis=1)
    return float(d.min()), float(ts[int(d.argmin())])


def generate(template_paths, betas=None, T_cs=None, v_ms=None, want=None):
    """扫参生成对抗场景，只保留真被判为让路态（或紧急态）的。"""
    betas = betas if betas is not None else [-100, -75, -50, -30, -15, 0, 15, 30, 50, 75, 100]
    T_cs = T_cs if T_cs is not None else [100.0, 150.0, 200.0, 250.0]
    v_ms = v_ms if v_ms is not None else [3.0, 5.0, 7.0]
    templates = load_scenario_pool(template_paths)
    out = []
    for ti, tpl in enumerate(templates):
        for b in betas:
            for T in T_cs:
                for vm in v_ms:
                    r = make_conflict_scenario(tpl, b, T, vm)
                    if r is None:
                        continue
                    sc, pp = r
                    cls = classify(sc, pp)
                    if not (cls["head_on"] or cls["crossing"] or cls["overtake"]
                            or cls["keep"] or cls["is_emergency"]):
                        continue
                    dmin, tmin = min_center_distance(sc, pp)
                    out.append({
                        "scenario": sc, "pp": pp,
                        "meta": dict(template=ti, beta=b, T_c=T, v_m=vm,
                                     dmin=dmin, t_at_min=tmin, **cls),
                    })
                    if want and len(out) >= want:
                        return out
    return out


if __name__ == "__main__":
    import glob
    paths = sorted(glob.glob(os.path.join(_TRB, "scenarios", "T-*.xml")),
                   key=lambda p: int(p.split("T-")[1].split(".")[0]))[:3]
    got = generate(paths, want=40)
    print(f"生成 {len(got)} 个对抗场景\n")
    from collections import Counter
    cnt = Counter()
    for g in got:
        m = g["meta"]
        for k in ("head_on", "crossing", "overtake", "keep", "is_emergency"):
            if m[k]:
                cnt[k] += 1
    print("态势分布:", dict(cnt))
    dmins = [g["meta"]["dmin"] for g in got]
    print(f"最近中心距: 中位 {np.median(dmins):.0f} m · 最小 {min(dmins):.0f} m "
          f"· <250m(真撞) 占 {sum(1 for d in dmins if d < 250)}/{len(dmins)}")
    print("\n样例:")
    for g in got[:6]:
        m = g["meta"]
        tags = [k for k in ("head_on", "crossing", "overtake", "keep", "is_emergency") if m[k]]
        print(f"  β={m['beta']:+4}° T_c={m['T_c']:5.0f}s v_m={m['v_m']:.1f} "
              f"→ 最近距 {m['dmin']:6.1f}m@{m['t_at_min']:.0f}s  {tags}")
