"""
观测模块冒烟测试 —— 断言全部用手算闭式几何 / 论文定义，fact-based。
跑：/opt/miniconda3/envs/trb/bin/python 代码/tests/test_observation.py

设计：
  A) clean 合成几何（init=[0,0], goal_center=[1000,0] → e_long=[1,0]/e_lat=[0,1]，可手算精确值）
     逐项校验 27 维每一块。
  B) T-0 真实场景端到端集成 sanity（shape 27 / 无 NaN / d_goal 对 numpy 真值）。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from trb_env.usv_observation import (  # noqa: E402
    OBS_DIM,
    SECTOR_NAMES,
    ObservationBuilder,
)
from trb_env.usv_dynamics import wrap_to_pi  # noqa: E402

_fail = 0


def check(name, got, exp, tol=1e-9):
    global _fail
    got = np.asarray(got, dtype=float)
    exp = np.asarray(exp, dtype=float)
    ok = got.shape == exp.shape and np.all(np.abs(got - exp) <= tol)
    if not ok:
        _fail += 1
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: got={np.round(got, 5)} exp={np.round(exp, 5)}")


# 扇区在向量里的下标（front,left,right,behind）；每扇区占 obs[10+3s : 13+3s]
SEC = {n: SECTOR_NAMES.index(n) for n in SECTOR_NAMES}


def sector_slice(name):
    s = SEC[name]
    return slice(10 + 3 * s, 13 + 3 * s)


# ============ A) clean 合成几何 ============
INIT = [0.0, 0.0]
GOAL = [1000.0, 0.0]            # e_long=[1,0], e_lat=[0,1]
GORI = (-0.1, 0.1)             # 目标朝向区间
KMAX = 100

ob = ObservationBuilder(GOAL, GORI, INIT, KMAX)
ob.reset()

print("===== 维度 / 健康 =====")
o = ob.build([100, 0, 0.05, 5], [0.02, -0.01], [], step=10)
check("0) obs 维度 = 27", [o.shape[0]], [OBS_DIM])
assert not np.any(np.isnan(o)), "obs 含 NaN"
print(f"[PASS] obs 无 NaN, dtype={o.dtype}")

print("\n===== [0:4] ego =====")
check("① ego [v,θ,a,ω]", o[0:4], [5, 0.05, 0.02, -0.01])

print("\n===== [4:9] goal =====")
# ego 在 [100,0]：d_goal=900；step10/kmax100→remaining=90；θ=0.05∈[-0.1,0.1]→β_goal=0；d_long=100,d_lat=0
check("② d_goal", [o[4]], [900.0])
check("③ remaining_steps", [o[5]], [90.0])
check("④ β_goal（朝向区间内→0）", [o[6]], [0.0])
check("⑤ d_long / d_lat（线上）", [o[7], o[8]], [100.0, 0.0])

# β_goal 区间外：θ=0.3 → 到上边界 0.1 距 0.2；θ=-0.3 → 到下边界 -0.1 距 -0.2
o2 = ob.build([0, 0, 0.3, 5], [0, 0], [], step=0)
check("⑥ β_goal（区间外 +）", [o2[6]], [0.2])
o3 = ob.build([0, 0, -0.3, 5], [0, 0], [], step=0)
check("⑦ β_goal（区间外 -）", [o3[6]], [-0.2])

# d_lat 偏离：ego[100,30]→d_long=100,d_lat=30（参考线左侧正）
o4 = ob.build([100, 30, 0, 5], [0, 0], [], step=0)
check("⑧ d_lat（偏离 +30）", [o4[7], o4[8]], [100.0, 30.0])

print("\n===== [9] bool goal =====")
# min(|d_lat|,|d_long|)>2000 才 True
oA = ob.build([100, 0, 0, 5], [0, 0], [], step=0)        # min(0,100)=0<2000→0
check("⑨ bool goal（线上→0）", [oA[9]], [0.0])
oB = ob.build([3000, 0, 0, 5], [0, 0], [], step=0)       # min(0,3000)=0<2000→0
check("⑩ bool goal（远但在线上→0）", [oB[9]], [0.0])
oC = ob.build([3000, 3000, 0, 5], [0, 0], [], step=0)    # min(3000,3000)>2000→1
check("⑪ bool goal（双向远→1）", [oC[9]], [1.0])

print("\n===== [10:22] traffic（扇区 + β 符号）=====")
ob.reset()
# ego 朝 +x；他船正前 [500,0]→front, d=500, β=0
of = ob.build([0, 0, 0, 5], [0, 0], [(1, [500, 0])], step=0)
check("⑫ front 扇区(d,β,ḋ)", of[sector_slice("front")], [500, 0, 0])
check("⑫ 其余扇区默认(d_sense)", [of[sector_slice("left")][0], of[sector_slice("right")][0],
                                of[sector_slice("behind")][0]], [8000, 8000, 8000])

ob.reset()  # 右舷 -y → β=+90°→right
orr = ob.build([0, 0, 0, 5], [0, 0], [(1, [0, -500])], step=0)
check("⑬ 右舷他船→right 扇区, β=+90°", orr[sector_slice("right")], [500, np.pi / 2, 0])

ob.reset()  # 左舷 +y → β=-90°→left
ol = ob.build([0, 0, 0, 5], [0, 0], [(1, [0, 500])], step=0)
check("⑭ 左舷他船→left 扇区, β=-90°", ol[sector_slice("left")], [500, -np.pi / 2, 0])

ob.reset()  # 正后 [-500,0]→behind, β=±π
obh = ob.build([0, 0, 0, 5], [0, 0], [(1, [-500, 0])], step=0)
check("⑮ 正后他船→behind 扇区", [obh[sector_slice("behind")][0], abs(obh[sector_slice("behind")][1])],
      [500, np.pi])

print("\n===== 感知距离门限 + ḋ =====")
ob.reset()  # 超 d_sense(8000)→不检测，全默认
oo = ob.build([0, 0, 0, 5], [0, 0], [(1, [9000, 0])], step=0)
check("⑯ 超感知距离→front 默认 8000", oo[sector_slice("front")], [8000, 0, 0])

ob.reset()  # ḋ：靠近→负，远离→正，首次=0
ob.build([0, 0, 0, 5], [0, 0], [(7, [500, 0])], step=0)            # d=500, ḋ=0
oapp = ob.build([0, 0, 0, 5], [0, 0], [(7, [400, 0])], step=1)     # d=400, ḋ=-100
check("⑰ ḋ 靠近→-100", [oapp[sector_slice("front")][2]], [-100.0])
orec = ob.build([0, 0, 0, 5], [0, 0], [(7, [600, 0])], step=2)     # d=600, ḋ=+200
check("⑱ ḋ 远离→+200", [orec[sector_slice("front")][2]], [200.0])

# reset 清历史 → ḋ 重新从 0
ob.reset()
ofresh = ob.build([0, 0, 0, 5], [0, 0], [(7, [400, 0])], step=0)
check("⑲ reset 后 ḋ 归零", [ofresh[sector_slice("front")][2]], [0.0])

# 同扇区多船取最近
ob.reset()
omul = ob.build([0, 0, 0, 5], [0, 0], [(1, [800, 0]), (2, [300, 0])], step=0)
check("⑳ 同扇区多船取最近(300)", [omul[sector_slice("front")][0]], [300.0])

# FG7（全面审计补漏）：ḋ 用每船真实历史"即便上一步超感知距离"（docstring 承诺）——他船 in→out→back，
#   ḋ 须对真上一步距离。旧测试 ⑰⑱ 只测连续在界，把 _prev_dist 更新移到 `d>d_sense continue` 之后不翻 FAIL（假守护，03 L12）。
ob.reset()
ob.build([0, 0, 0, 5], [0, 0], [(7, [500, 0])], step=0)            # 在界 d=500
ob.build([0, 0, 0, 5], [0, 0], [(7, [9000, 0])], step=1)          # 出界 d=9000(>d_sense，不写obs但记历史)
oback = ob.build([0, 0, 0, 5], [0, 0], [(7, [400, 0])], step=2)   # 回界 d=400 → ḋ=400-9000=-8600(真历史)
check("⑳b ḋ 跨感知边界用真历史(in→out→back → -8600，非过期-100)", [oback[sector_slice("front")][2]], [-8600.0])

print("\n===== [22:27] termination =====")
ob.reset()
ot = ob.build([0, 0, 0, 5], [0, 0], [],
              step=0, term_flags={"time": False, "area": True, "stopped": False,
                                  "collision": True, "goal": False})
check("㉑ termination [time,area,stopped,collision,goal]", ot[22:27], [0, 1, 0, 1, 0])

# ============ B) T-0 真实场景集成 sanity ============
print("\n===== B) T-0 真实场景集成 =====")
T0 = "/tmp/trb_T0.xml"
if os.path.exists(T0):
    from commonocean.common.file_reader import CommonOceanFileReader

    sc, pps = CommonOceanFileReader(T0).open()
    pp = list(pps.planning_problem_dict.values())[0]
    g = pp.goal.state_list[0]
    gc = g.position.center
    gori = (float(g.orientation.start), float(g.orientation.end))
    kmax = int(g.time_step.end)
    init = pp.initial_state

    rob = ObservationBuilder(gc, gori, init.position, kmax)
    rob.reset()
    obst = sc.dynamic_obstacles[0]
    ego0 = [init.position[0], init.position[1], float(init.orientation), float(init.velocity)]
    oT = rob.build(ego0, [0, 0], [(obst.obstacle_id, obst.initial_state.position)], step=0)
    check("㉒ T-0 obs 维度", [oT.shape[0]], [OBS_DIM])
    assert not np.any(np.isnan(oT)), "T-0 obs 含 NaN"
    # d_goal 对 numpy 真值
    d_goal_true = float(np.linalg.norm(np.asarray(init.position) - np.asarray(gc)))
    check("㉓ T-0 d_goal 对 numpy 真值", [oT[4]], [d_goal_true], tol=1e-6)
    # 他船距 5759.5m < d_sense → 应被检测（某扇区 d < 8000）
    detected = any(oT[10 + 3 * s] < 8000.0 for s in range(4))
    print(f"[{'PASS' if detected else 'FAIL'}] ㉔ T-0 他船在感知距离内被检测: {detected}")
    if not detected:
        _fail += 1
    print(f"      remaining_steps={oT[5]} (kmax={kmax}) / d_goal={oT[4]:.1f}")
else:
    print("[SKIP] /tmp/trb_T0.xml 不存在（仅缺集成 sanity，A 段已覆盖逻辑）")

# ============ C) 覆盖盲区 + 防御性加固（独立复核新增）============
print("\n===== C) 覆盖盲区 + fail-fast 守卫 =====")


def check_raises(name, fn):
    global _fail
    try:
        fn()
    except ValueError as e:
        print(f"[PASS] {name}: 正确抛 ValueError（{str(e)[:36]}…）")
        return
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] {name}: 抛了非 ValueError = {type(e).__name__}")
        _fail += 1
        return
    print(f"[FAIL] {name}: 期望抛 ValueError 但没抛")
    _fail += 1


# bool goal min 严格性：仅单向超阈值 → min 仍 < 阈值 → False（验证是 min 不是 max）
ob.reset()
oM = ob.build([3000, 100, 0, 5], [0, 0], [], step=0)   # d_long=3000>2000, d_lat=100<2000 → min=100→0
check("㉕ bool goal（仅纵向超→0，验 min 非 max）", [oM[9]], [0.0])

# behind β 带符号（不只 abs）：后方右舷 [-300,-300] → behind, β=+3π/4
ob.reset()
oBR = ob.build([0, 0, 0, 5], [0, 0], [(1, [-300, -300])], step=0)
check("㉖ behind 右后 β=+3π/4, d=300√2", oBR[sector_slice("behind")],
      [300 * np.sqrt(2), 3 * np.pi / 4, 0])

# 多船同帧分到 4 个不同扇区
ob.reset()
oMS = ob.build([0, 0, 0, 5], [0, 0],
               [(1, [500, 0]), (2, [0, -500]), (3, [0, 500]), (4, [-500, 0])], step=0)
check("㉗ 多扇区同帧 front/right/left d", [oMS[sector_slice("front")][0],
      oMS[sector_slice("right")][0], oMS[sector_slice("left")][0]], [500, 500, 500])
check("㉗ 多扇区同帧 right/left β 符号", [oMS[sector_slice("right")][1],
      oMS[sector_slice("left")][1]], [np.pi / 2, -np.pi / 2])

# β_goal 需先 wrap：ego 朝向 6.22 rad（≈-0.063 wrap 后）落区间 [-0.1,0.1] → 0（T-0 真实情形）
ob.reset()
oW = ob.build([0, 0, 6.22, 5], [0, 0], [], step=0)
check("㉘ β_goal 先 wrap 再判区间（6.22→内→0）", [oW[6]], [0.0])

# fail-fast 守卫
check_raises("㉙ ego 含 NaN → 报错", lambda: ob.build([0, 0, np.nan, 5], [0, 0], [], step=0))
check_raises("㉚ 他船 position 标量 → 报错", lambda: ob.build([0, 0, 0, 5], [0, 0], [(1, 5)], step=0))
check_raises("㉛ term_flags 拼错键 → 报错",
             lambda: ob.build([0, 0, 0, 5], [0, 0], [], step=0, term_flags={"collison": True}))

# 关键 MAJOR 修复：NaN 他船报错时不污染 _prev_dist，下一帧 ḋ 仍正确（非 NaN）
ob.reset()
ob.build([0, 0, 0, 5], [0, 0], [(7, [500, 0])], step=0)               # d=500 入历史
check_raises("㉜ NaN 他船当帧报错", lambda: ob.build([0, 0, 0, 5], [0, 0], [(7, [np.nan, 0])], step=1))
oRec = ob.build([0, 0, 0, 5], [0, 0], [(7, [400, 0])], step=2)        # ḋ 应=400-500=-100（未被 NaN 污染）
check("㉝ NaN 报错后 _prev_dist 未污染（ḋ=-100 非 NaN）", [oRec[sector_slice("front")][2]], [-100.0])

print("\n" + ("✅ 全部 PASS" if _fail == 0 else f"❌ {_fail} 项 FAIL"))
sys.exit(1 if _fail else 0)
