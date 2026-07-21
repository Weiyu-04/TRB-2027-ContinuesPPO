# -*- coding: utf-8 -*-
"""场景会遇类型分类（`03` L110·查实训练/测试场景是否覆盖 head-on / crossing / overtaking）。

⚠️ 背景（实验有效性硬伤·user 2026-06-28 判"致命·必先补救"）：
   本地 94 个 HandcraftedTwoVesselEncounters 场景（跨 T-0..T-1990 抽样）经分类 = 对遇 + 交叉、**0 追越(overtaking)**。
   多种分类口径（初始态势几何 / 策略轨迹 rho>0 会遇时刻）都一致：**追越 = 0**（head-on/crossing 计数随口径变·overtaking 恒 0）。
   → 强烈怀疑该场景源【不含追越】→ 训练/测试覆盖不全。**本脚本在【服务器】对训练真实用的池分类·确认 200/2000 覆盖·先补救再推进。**

分类口径：用【初始态势几何】= 本船初始艏向 vs 他船初始相对方位 β + 航向差 Δψ（这两船是按"相遇"摆的·初始几何即设计的会遇类型）。
   ⚠️ 近似：会遇类型严格应在 collision_possible 的会遇时刻判·但初始两船远→正规 COLREGs 谓词在初始/CPA 都多报 no-conflict（误导）·
   且本船真实是避让路径非直线（nominal CPA 失真）。∴ 用初始几何作【设计类型】代理（实测稳定·overtaking 恒 0）。
   **最权威的逐场景类型 = 跑策略找 rho>0 会遇时刻分类（见 scratchpad/encounter_figs.py 法·需 checkpoint）·可作交叉验证。**

跑（服务器·几分钟·零算力·只读场景）：
   STEP4E_SDIR=$HOME/trb/scenarios python 代码/classify_scenarios.py --mode strided200   # 训练真用的 strided-200（i*2000//200）
   STEP4E_SDIR=$HOME/trb/scenarios python 代码/classify_scenarios.py --mode all           # 全部已下载场景
输出：每类计数 + 各类样例 T-index（供 STEP4E_TRAJ_IDXS 录轨迹）+ 【缺某类】告警。
"""
import sys, os, glob, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from collections import Counter, defaultdict
from commonocean.common.file_reader import CommonOceanFileReader

def wrap(a): return (a + np.pi) % (2 * np.pi) - np.pi

def read_scenario(path):
    """→ (ego_p, ego_psi, ego_v, goal_center, obs_p, obs_psi, obs_v)。无他船→obs_*=None。"""
    sc, pps = CommonOceanFileReader(path).open()
    pp = list(pps.planning_problem_dict.values())[0]
    init = pp.initial_state
    ego_p = np.asarray(init.position, dtype=float); ego_psi = float(init.orientation)
    ego_v = float(getattr(init, "velocity", 5.0))
    gc = None
    try: gc = np.asarray(getattr(pp.goal.state_list[0].position, "center", None), dtype=float)
    except Exception: gc = None
    obs = sc.dynamic_obstacles
    if not obs: return ego_p, ego_psi, ego_v, gc, None, None, None
    os0 = obs[0].initial_state
    return (ego_p, ego_psi, ego_v, gc, np.asarray(os0.position, dtype=float),
            float(os0.orientation), float(getattr(os0, "velocity", 5.0)))

def classify(ego_p, ego_psi, ego_v, gc, obs_p, obs_psi, obs_v):
    """初始态势几何分类。本船参考航向=朝目标（无目标用初始艏向）。返回 (类型, β°, Δψ°)。"""
    course = ego_psi
    if gc is not None and np.linalg.norm(gc - ego_p) > 1e-6:
        d = gc - ego_p; course = float(np.arctan2(d[1], d[0]))
    dxy = obs_p - ego_p
    beta = abs(np.degrees(wrap(np.arctan2(dxy[1], dxy[0]) - course)))   # 他船在本船参考航向的方位（0=正前）
    dpsi = abs(np.degrees(wrap(obs_psi - course)))                      # 航向差（0=同向·180=对向）
    if beta <= 45 and dpsi >= 135: t = "head-on"
    elif beta <= 45 and dpsi <= 45: t = "overtaking" if ego_v > obs_v else "same-dir(ego-slower)"
    elif beta >= 112.5 and dpsi <= 45: t = "being-overtaken(stand-on)"
    elif beta <= 112.5: t = "crossing"
    else: t = "other"
    return t, beta, dpsi

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="strided200", choices=["strided200", "all"])
    ap.add_argument("--ids", default=None)
    ap.add_argument("--sdir", default=os.environ.get("STEP4E_SDIR", "/tmp/trb_scenarios_pool"))
    args = ap.parse_args()
    if args.ids: ids = [int(x) for x in args.ids.split(",")]
    elif args.mode == "strided200": ids = [i * 2000 // 200 for i in range(200)]
    else: ids = None
    if ids is not None:
        paths = [f"{args.sdir}/T-{i}.xml" for i in ids]; paths = [p for p in paths if os.path.exists(p)]
    else:
        paths = sorted(glob.glob(f"{args.sdir}/*.xml"))
    print(f"场景目录: {args.sdir} | mode={args.mode} | 找到 {len(paths)} 个场景")
    if not paths:
        print("⚠️ 没找到场景！检查 STEP4E_SDIR 路径 / 是否已下载。"); return
    dist = Counter(); ex = defaultdict(list)
    for p in paths:
        ti = int(p.split("T-")[-1].split(".")[0])
        try:
            ego_p, ego_psi, ego_v, gc, obs_p, obs_psi, obs_v = read_scenario(p)
        except Exception as e:
            dist[f"load-err"] += 1; continue
        if obs_p is None: dist["no_obstacle"] += 1; continue
        t, b, dp = classify(ego_p, ego_psi, ego_v, gc, obs_p, obs_psi, obs_v)
        dist[t] += 1; ex[t].append(ti)
    print("\n=== 会遇类型分布（初始态势几何）===")
    for k, v in dist.most_common(): print(f"  {k:28}: {v}")
    print("\n=== 各类型样例 T-index（前 10·供 STEP4E_TRAJ_IDXS 录轨迹）===")
    for k in ["head-on", "crossing", "overtaking", "being-overtaken(stand-on)"]:
        idxs = sorted(ex.get(k, [])); print(f"  {k:30}: {idxs[:10]}" + ("" if idxs else "  ⚠️⚠️ 缺！=未覆盖"))
    print("\n=== 🔴 覆盖判定（三大 give-way 会遇类型）===")
    missing = [k for k in ["head-on", "crossing", "overtaking"] if dist.get(k, 0) == 0]
    if missing:
        print(f"  ❌❌ 缺失: {missing} → 训练/测试【未覆盖】这些会遇类型 → 实验有效性硬伤 → 必须补场景再推进！")
        print(f"     （overtaking 缺 = 本场景源不含追越·须从 CommonOcean 别的集找/造追越场景补进训练+测试）")
    else:
        print(f"  ✅ 三类全覆盖（head-on/crossing/overtaking 各 ≥1）→ 覆盖完整、可放心推进。")
    print("\n注：本分类=初始态势几何代理（实测 overtaking 恒 0 跨口径稳健）；最权威逐场景类型可跑策略在 rho>0 会遇时刻交叉验证。")

if __name__ == "__main__":
    main()
