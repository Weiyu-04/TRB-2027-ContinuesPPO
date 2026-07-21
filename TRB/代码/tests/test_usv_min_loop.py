"""Node 3 冒烟测试：连续投影盾【最小闭环】通过门2（Phase 2）。

跑：/opt/miniconda3/envs/trb/bin/python -B 代码/tests/test_usv_min_loop.py
（项目脚本风格：模块级断言 + ok() 自动计数 + sys.exit；非 pytest。）

⭐ 通过门2（蓝图第二层 §6 字面判据，2026-06-16 复审校正 D35 over-claim 后回归字面）=
   简单策略输出经投影后【整段零违规 ∧ 零碰撞 ∧ 能到达】+ 给路投影正确（核心卖点）。
   = ② 干净 crossing 给路场景达成（零违规/零碰撞/零紧急/到达 + 给路投影合规右转）。

⚠️ 实测张力（576 场景搜索 + Agent 复审坐实，D35 校正）：
- 字面"整段零违规"对给路相遇【可达】（②即反例，推翻旧 D35"达不到"的 over-claim）；
- 但代价 = 干净给路episode的给路【只 engage 1 步】+ 他船远擦肩——本框架【持续给路】必触发紧急兜底→L20 残余违规；
- ③ head-on 热冲突 = 紧急主导（is_emergency 180s 集合预测早触发）→ loop 仍闭合(零碰撞+到达) 但带 L20 残余违规（emergency-while-keep、与离散 Safe 同源、待 step4e satisfiability 整定、**非投影失败**）。

集成必做项（D32/D33/L36）：episode 起 proj.reset()；每决策步只调一次 safe_action；u_exec 恒 ∈box（run 内断言）。
⚠️ 档位A 经验性零碰撞（靠 d_safe 大裕度）、非 provable（provable=档位B/Phase4）。
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from trb_env.usv_min_loop import (
    make_head_on_scenario, make_crossing_giveway_scenario, run_min_loop,
)
from trb_env.usv_colregs import (
    RHO_CROSSING, RHO_HEAD_ON, RHO_EMERGENCY, EmergencyController, VesselState as _VS,
)
from trb_env.usv_projection import ContinuousColregsProjection, DEFAULT_OMEGA_TURN

_fail = 0
_total = 0


def ok(name, cond):
    global _fail, _total
    _total += 1
    if not cond:
        _fail += 1
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")


# ============================================================================
# ① 负对照：无盾 head-on 直驶 → 真撞（证场景是真冲突、门有意义）+ 策略 sanity
# ============================================================================
ho = make_head_on_scenario(obs_x0=3000.0, goal_x=4000.0, v_ego0=5.0, v_obs=5.0)
r_unshield = run_min_loop(ho, use_shield=False, max_steps=200)
ok("① 负对照：无盾 head-on 真撞（场景是真冲突）", r_unshield["collided"] and not r_unshield["reached"])
r_far = run_min_loop(make_head_on_scenario(obs_x0=1e9, goal_x=4000.0), use_shield=False, max_steps=200)
ok("① 策略 sane：无障碍能到达（非平凡）", r_far["reached"] and not r_far["collided"])

# ============================================================================
# ② ⭐ 通过门2：干净 crossing 给路场景 = 整段零违规 ∧ 零碰撞 ∧ 到达 + 给路投影正确（核心卖点）
# ============================================================================
cg = make_crossing_giveway_scenario()
r_cg = run_min_loop(cg, use_shield=True, max_steps=220)

ok("② 零碰撞（裸船体不相交全程，经验性档位A）", not r_cg["collided"])
ok("② min_gap>0（裸船体净空）", r_cg["min_gap"] > 0)
ok("② 能到达目标", r_cg["reached"])
ok("② ⭐整段零违规（字面通过门2、推翻旧 D35'达不到'）", r_cg["violations"] == 0)
ok("② 零紧急（干净给路、不靠兜底）", r_cg["source_counts"].get("emergency", 0) == 0)
# 给路投影正确（核心卖点路径被验到）
ok("② 给路投影被触发（ρ2/3 经 projection 输出 >0 步）", r_cg["gw_proj_steps"] > 0)
ok("② ⭐给路投影方向正确（右转 ω≤−ω_turn = COLREGs 强制最小转向）= 连续投影进合规集【机制】成立", r_cg["gw_proj_compliant"])
_gw_om = r_cg["gw_proj_omegas"]
ok("② 给路投影 ω 独立核：全部 ≤ −ω_turn（不靠 harness 标志、堵弱守护）",
   len(_gw_om) > 0 and all(om <= -DEFAULT_OMEGA_TURN + 1e-9 for om in _gw_om))
ok("② crossing 给路态势确实出现（ρ3 ∈ rho_counts）", r_cg["rho_counts"].get(RHO_CROSSING, 0) > 0)
print(f"   [②干净给路诊断] 到达={r_cg['reached']} 碰撞={r_cg['collided']} 违规={r_cg['violations']} "
      f"给路投影{r_cg['gw_proj_steps']}步 ω={[round(o,4) for o in _gw_om]} 紧急{r_cg['source_counts'].get('emergency',0)} "
      f"min_gap={r_cg['min_gap']:.0f}m → ⚠️给路仅{r_cg['gw_proj_steps']}步=温和冲突(框架张力,D35)")

# ============================================================================
# ③ 诚实表征：head-on 热冲突 = 紧急主导（is_emergency 早触发）→ loop 仍闭合、带 L20 残余违规
#    （非通过门2 准则、非投影失败；是框架固有性质 + ViolationCounter 路径守护）
# ============================================================================
r_ho = run_min_loop(ho, use_shield=True, max_steps=200)
ok("③ head-on 有盾：loop 闭合（到达 + 零碰撞）", r_ho["reached"] and not r_ho["collided"])
ok("③ head-on 紧急主导（is_emergency 早于 persistent give-way 触发、D35）",
   r_ho["source_counts"].get("emergency", 0) > 0)
# 残余违规 = L20 满足性 gap（实测含 stand-on + give-way 两类、非单一 emergency-while-keep；与离散 Safe 同源）。
#   同时【守护 ViolationCounter 路径】：删 vc.step → head-on 违规 0 → 本断言翻 FAIL（变异坐实）。
ok("③ head-on 带 L20 残余违规（stand-on+give-way 混合；守护 ViolationCounter 路径，删 vc.step→翻 FAIL）",
   r_ho["violations"] > 0)
print(f"   [③head-on诊断] 到达={r_ho['reached']} 碰撞={r_ho['collided']} 违规={r_ho['violations']}(L20残余) "
      f"紧急{r_ho['source_counts'].get('emergency',0)}步 ρ={r_ho['rho_counts']} → 持续给路不可清洁(框架张力)")

# ============================================================================
# ④ 集成必做项 + 确定性 + proj.reset 跨 episode 守护
# ============================================================================
ok("④ 必做③ u_exec 恒 ∈box（run 内断言未抛 = 全程满足）", True)
r_cg2 = run_min_loop(make_crossing_giveway_scenario(), use_shield=True, max_steps=220)
ok("④ 确定性可复现（两跑逐字段一致）",
   r_cg2["reached"] == r_cg["reached"] and r_cg2["collided"] == r_cg["collided"]
   and r_cg2["violations"] == r_cg["violations"] and r_cg2["source_counts"] == r_cg["source_counts"])

# proj.reset 跨 episode 守护（堵复审 M-3，D33 B-EC-CROSS-EPISODE）：
#   safe_action 的 ρ5 进入边沿(prev≠ρ5)自带 EC.reset()，唯一漏网 = episode2 首步就是 ρ5（prev=ρ5 泄漏→无进入边沿）。
#   故守护场景须【起步即 ρ5】：近距 head-on obs_x0=2000。弄脏 proj（_prev_rho=ρ5 + EC 留某 mode）→ run 起 reset 应清。
#   删 reset：泄漏使首步 EC 沿用错 mode（ω 全 0 vs −0.03、source 全异）→ 翻 FAIL = 真守护（主窗口变异坐实）。
_ho_emerg = make_head_on_scenario(obs_x0=2000.0, goal_x=4000.0)
_VPe = _ho_emerg["params"]
_r_fresh_e = run_min_loop(_ho_emerg, use_shield=True, max_steps=60)
_dirty = ContinuousColregsProjection(_VPe.a_max, _VPe.w_max)
_dirty._prev_rho = RHO_EMERGENCY
_dirty._ec = EmergencyController(vessel_params=_VPe, dt=10.0)
_dirty._ec.step(_VS(position=np.array([0.0, 0.0]), orientation=0.0, velocity=2.0, length=175.0),
                _VS(position=np.array([-300.0, 50.0]), orientation=0.0, velocity=2.0, length=175.0))
_r_dirty = run_min_loop(_ho_emerg, use_shield=True, max_steps=60, proj=_dirty)
ok("④ proj.reset 守护泄漏（弄脏 proj 经 run 起 reset 后 == 全新；删 reset→ρ5 起步泄漏→翻 FAIL）",
   _r_dirty["source_counts"] == _r_fresh_e["source_counts"]
   and _r_dirty["violations"] == _r_fresh_e["violations"]
   and _r_dirty["gw_proj_steps"] == _r_fresh_e["gw_proj_steps"])

# ============================================================================
# ⑤ M-2 守护：循环跑满时末步碰撞不漏检（复审 M-2，2026-06-16）
#    碰撞/到达在循环顶部、动作【前】判。无盾 head-on 碰撞在顶部 k=29 首检；max_steps=29 → 最后一次
#    dyn_step 后的终态(对应 k=29)永不被顶部检 → 修复前 collided=False(漏检)；usv_min_loop 末态检测块
#    补检终态 → True。删该块 → 本断言翻 FAIL（变异坐实，对照 max_steps=30 在循环内即检出）。
# ============================================================================
_ho_m2 = make_head_on_scenario(obs_x0=3000.0, goal_x=4000.0, v_ego0=5.0, v_obs=5.0)
_r_m2 = run_min_loop(_ho_m2, use_shield=False, max_steps=29)
ok("⑤ 末步碰撞不漏检（max_steps=29 跑满、终态碰撞被末态检测捕捉；删末态块→翻 FAIL）",
   _r_m2["collided"])

print("\n" + (f"✅ 全部 PASS（{_total} 项）" if _fail == 0 else f"❌ {_fail}/{_total} 项 FAIL"))
sys.exit(1 if _fail else 0)
