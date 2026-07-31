#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Q3 复核脚本：把 ρ4（追越）的四个合取项在全部 2000 个官方场景上逐项拆开计数。

用法（在 TRB/ 下）：python3 Paper/05_外部复审/Q3_overtake_decompose.py
结论（2026-07-31 实测）：① Be(s_m,s_e) 命中 0 ⟹ 卡在第一个合取项，
即基准库的初始会遇几何中不存在"本船位于他船后方"的构型，非后续合取项过严。
⚠️ 本脚本只看初始时刻几何，未逐步扫描全轨迹。
"""
import glob
import os
import sys

import numpy as np

_TRB = os.environ.get("TRB_ROOT", os.getcwd())
sys.path.insert(0, os.path.join(_TRB, "代码"))

from trb_env.usv_scenarios import load_scenario_pool   # noqa: E402
from trb_env import usv_colregs as CO                  # noqa: E402

paths = sorted(glob.glob(os.path.join(_TRB, "scenarios", "T-*.xml")),
               key=lambda p: int(p.split("T-")[1].split(".")[0]))
n = dict(total=0, cp=0, be=0, be_sd=0, be_sd_fast=0, rho4=0)
for pth in paths:
    try:
        sc, pp = load_scenario_pool([pth])[0]
        ob = sc.dynamic_obstacles[0]
        ini, st = pp.initial_state, ob.initial_state
        se = CO.VesselState(np.array(ini.position, float), float(ini.orientation),
                            float(ini.velocity), 175.0)
        sm = CO.VesselState(np.array(st.position, float), float(st.orientation),
                            float(st.velocity), float(ob.obstacle_shape.length))
    except Exception:
        continue
    n["total"] += 1
    cp = CO.collision_possible(se, sm)
    n["cp"] += cp
    if not CO.in_behind_sector(sm, se):        # ① 本船在他船后扇区
        continue
    n["be"] += 1
    if CO.orientation_delta(se, sm, CO.DELTA_OVERTAKE, offset=0.0):   # ② 航向近同向
        continue
    n["be_sd"] += 1
    if not CO.drives_faster(se, sm):           # ③ 本船更快
        continue
    n["be_sd_fast"] += 1
    if cp:                                     # ④ 碰撞可能
        n["rho4"] += 1

print(f"有效场景 {n['total']} · collision_possible {n['cp']}")
print(f"① Be(s_m,s_e)                {n['be']}")
print(f"② ①∧Sd(67.5°)                {n['be_sd']}")
print(f"③ ②∧v_e>v_m                  {n['be_sd_fast']}")
print(f"④ ③∧collision_possible = ρ4  {n['rho4']}")
