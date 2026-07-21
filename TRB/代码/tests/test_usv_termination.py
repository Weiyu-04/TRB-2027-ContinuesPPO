"""
终止模块冒烟测试 —— 断言全部手算 / 论文定义，fact-based，自包含（合成 goal + shapely 占据，不依赖网络）。
跑：/opt/miniconda3/envs/trb/bin/python 代码/tests/test_usv_termination.py

合成 goal（= T-0 同构，已实跑验证 is_reached 行为一致）：中心[5000,0]，朝向[-0.17,0.17]，time[0,170]。
本船占据：l=175 / w=25.4，原点朝 +x 时跨 x∈[-87.5,87.5] / y∈[-12.7,12.7]。
"""
import os
import sys

import numpy as np
from shapely.geometry import box

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from trb_env.usv_termination import TerminationChecker  # noqa: E402

from commonroad.common.util import Interval, AngleInterval  # noqa: E402
from commonroad.geometry.shape import Rectangle  # noqa: E402
from commonroad.planning.goal import GoalRegion  # noqa: E402
from commonroad.scenario.state import CustomState  # noqa: E402

_fail = 0


def check(name, got, exp):
    global _fail
    ok = bool(got) == bool(exp)
    if not ok:
        _fail += 1
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: got={bool(got)} exp={bool(exp)}")


def check_raises(name, fn, exc=ValueError):
    global _fail
    try:
        fn()
    except exc:
        print(f"[PASS] {name}: 正确抛 {exc.__name__}")
        return
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] {name}: 抛了 {type(e).__name__} 非 {exc.__name__}")
        _fail += 1
        return
    print(f"[FAIL] {name}: 期望抛 {exc.__name__} 但没抛")
    _fail += 1


# 合成 goal（T-0 同构）
GC = np.array([5000.0, 0.0])
goal = GoalRegion([CustomState(
    position=Rectangle(length=400.0, width=60.0, center=GC, orientation=0.0),
    orientation=AngleInterval(-0.17, 0.17),
    time_step=Interval(0, 170),
)])
KMAX = 170
tc = TerminationChecker(goal, KMAX)

print("===== 1_time（≥k_max=170）=====")
check("① step=100 → time False", tc.check([0, 0, 0, 5], 100)[1]["time"], False)
check("② step=170 → time True", tc.check([0, 0, 0, 5], 170)[1]["time"], True)
check("③ step=200 → time True", tc.check([0, 0, 0, 5], 200)[1]["time"], True)

print("\n===== 1_stopped（|v|≤eps=1e-3）=====")
check("④ v=5 → stopped False", tc.check([0, 0, 0, 5], 10)[1]["stopped"], False)
check("⑤ v=0 → stopped True", tc.check([0, 0, 0, 0], 10)[1]["stopped"], True)
check("⑥ v=1e-4 → stopped True", tc.check([0, 0, 0, 1e-4], 10)[1]["stopped"], True)
check("⑦ v=0.5 → stopped False", tc.check([0, 0, 0, 0.5], 10)[1]["stopped"], False)

print("\n===== 1_collision（shapely 形状相交）=====")
# 本船 [0,0,0]：跨 x[-87.5,87.5] y[-12.7,12.7]
check("⑧ 重叠他船 → collision True", tc.check([0, 0, 0, 5], 10, [box(50, -5, 150, 5)])[1]["collision"], True)
check("⑨ 远离他船 → collision False", tc.check([0, 0, 0, 5], 10, [box(500, -5, 600, 5)])[1]["collision"], False)
check("⑩ 无他船 → collision False", tc.check([0, 0, 0, 5], 10, [])[1]["collision"], False)
# 旋转本船 θ=π/2：跨 x[-12.7,12.7] y[-87.5,87.5]
check("⑪ 旋转后 +y 向他船 → collision True",
      tc.check([0, 0, np.pi / 2, 5], 10, [box(-5, 50, 5, 80)])[1]["collision"], True)
check("⑫ 旋转后 +x 向他船(超宽) → collision False",
      tc.check([0, 0, np.pi / 2, 5], 10, [box(50, -5, 80, 5)])[1]["collision"], False)

print("\n===== 1_goal（官方 is_reached：位置+朝向+时间三查）=====")
check("⑬ 中心+朝向0+t10 → goal True", tc.check([5000, 0, 0, 5], 10)[1]["goal"], True)
check("⑭ 远离中心 → goal False", tc.check([0, 0, 0, 5], 10)[1]["goal"], False)
check("⑮ 中心+朝向2.0(超区间) → goal False", tc.check([5000, 0, 2.0, 5], 10)[1]["goal"], False)
check("⑯ 中心+t999(超时间区间) → goal False", tc.check([5000, 0, 0, 5], 999)[1]["goal"], False)

print("\n===== 1_area（可配置包围盒；默认 None 不触发）=====")
tc_box = TerminationChecker(goal, KMAX, nav_area_box=(0, 0, 1000, 1000))
check("⑰ 盒内(500,500) → area False", tc_box.check([500, 500, 0, 5], 10)[1]["area"], False)
check("⑱ 盒外(2000,500) → area True", tc_box.check([2000, 500, 0, 5], 10)[1]["area"], True)
check("⑲ 盒外(-5,500) → area True", tc_box.check([-5, 500, 0, 5], 10)[1]["area"], True)
check("⑳ 默认无盒 → area 永 False", tc.check([99999, 99999, 0, 5], 10)[1]["area"], False)

print("\n===== done = 任一为真 + 多条件同时 + fail-fast =====")
done, flags = tc.check([0, 0, 0, 5], 10, [])           # 全 False
check("㉑ 全条件不满足 → done False", done, False)
done, flags = tc.check([0, 0, 0, 5], 10, [box(50, -5, 150, 5)])  # 仅 collision
check("㉒ 撞了 → done True", done, True)
# 多条件同时：到达目标 + step=170（goal True + time True）
done, flags = tc.check([5000, 0, 0, 5], 170)
check("㉓ 到达+到最大步 → goal+time 都 True", flags["goal"] and flags["time"], True)
check("㉓ done True", done, True)
# flags 键齐全
check("㉔ flags 5 键齐全", set(flags.keys()) == {"time", "area", "stopped", "collision", "goal"}, True)

check_raises("㉕ ego 含 NaN → 报错", lambda: tc.check([0, 0, np.nan, 5], 10))
check_raises("㉖ ego 维度错 → 报错", lambda: tc.check([0, 0, 5], 10))
check_raises("㉗ nav_area_box min≥max → 报错", lambda: TerminationChecker(goal, KMAX, nav_area_box=(0, 0, 0, 100)))

print("\n===== 加固 + 边界（独立复核新增）=====")
check_raises("㉘ nav_area_box 含 NaN → 报错",
             lambda: TerminationChecker(goal, KMAX, nav_area_box=(0, 0, float("nan"), 100)))
check_raises("㉙ obstacle_footprints 含 None → 报错", lambda: tc.check([0, 0, 0, 5], 10, [None]))
check("㉚ v=1e-3(=eps) → stopped True", tc.check([0, 0, 0, 1e-3], 10)[1]["stopped"], True)
check("㉛ v=1.001e-3(>eps) → stopped False", tc.check([0, 0, 0, 1.001e-3], 10)[1]["stopped"], False)
check("㉜ 恰在边界(1000,500) → area False(闭含)", tc_box.check([1000, 500, 0, 5], 10)[1]["area"], False)
# 本船[0,0,0]右上角=(87.5,12.7)；他船左下角恰贴该点 → intersects 含接触 → True
check("㉝ 顶点接触 → collision True", tc.check([0, 0, 0, 5], 10, [box(87.5, 12.7, 200, 100)])[1]["collision"], True)

print("\n" + ("✅ 全部 PASS" if _fail == 0 else f"❌ {_fail} 项 FAIL"))
sys.exit(1 if _fail else 0)
