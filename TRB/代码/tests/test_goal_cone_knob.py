#!/usr/bin/env python3
"""STEP4E_GOAL_CONE_HALF / goal_cone_half 旋钮冒烟测试（ρ0 朝目标锥·统一态势盾方案①·PhaseC·L147）。

背景：ρ0 朝目标锥旋钮此前在 env（usv_projection）实现但【未接到 run_step4e 环境变量】。本测守护接线：
  STEP4E_GOAL_CONE_HALF=Φ(度)→内部转弧度传盾·默认 off=None=bit-identical·仅连续臂·进 config_conflict·无盾臂 fail-fast。

验：
  T1 maker 签名+透传+默认 bit-identical：两 maker 接受 goal_cone_half/goal_v_floor；make(默认)→ proj.goal_cone_half==None（现状）；
     make(goal_cone_half=rad(45))→ 全==rad(45)（透传落地）。
  T2 config_conflict 锥混配检测：锥 on vs off → 冲突；同锥 → 无冲突；旧记录无字段 → 不误冲突；异 v_floor → 冲突。
  T3 import-time 守卫（子进程）：默认 off=None·Φ=45→DEG45/RAD·'0'=off·负/>180/'foo'/v_floor<0/v_floor>v_max→fail-fast·无盾+锥→fail-fast。
注：两套默认【独立·互不兜底】——maker 参数默认 goal_cone_half=None（T1 守）vs run_step4e env 默认 "off"（T3 守）·各自被覆盖（变异审 F2）。

运行：cd 代码 && /opt/miniconda3/envs/trb/bin/python tests/test_goal_cone_knob.py
"""
import os
import sys
import math
import inspect
import subprocess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import run_step4e as S
from trb_env.usv_sac_train import make_continuous_safe_model, make_continuous_safe_ppo_model

N_FAIL = 0
_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def chk(cond, msg):
    global N_FAIL
    print(("  ✅ " if cond else "  ❌ ") + msg)
    if not cond:
        N_FAIL += 1


# ---------------- T2 config_conflict（纯函数·无需场景）----------------
print("T2 config_conflict 锥混配检测")
_base = {"steps": 100, "n_total": 200}   # 其余字段缺→.get 默认=与 cur 默认一致（隔离锥）
chk(S.config_conflict([{**_base, "goal_cone_half_deg": None}], 100, 200) == set(),
    "T2 锥都 off·cur off → 无冲突")
chk(bool(S.config_conflict([{**_base, "goal_cone_half_deg": 45.0}], 100, 200)),
    "T2 记录锥45·cur off(默认) → 冲突（防锥 on/off 混进同一 jsonl）")
chk(S.config_conflict([{**_base, "goal_cone_half_deg": 45.0}], 100, 200, goal_cone_half_deg=45.0) == set(),
    "T2 记录锥45·cur45 → 无冲突（同锥正常续跑）")
chk(S.config_conflict([_base], 100, 200) == set(),
    "T2 旧记录无 goal_cone_half_deg 字段·cur off → 不误冲突（向后兼容·离散记录同）")
chk(bool(S.config_conflict([{**_base, "goal_cone_half_deg": 45.0, "goal_v_floor": 2.0}],
                           100, 200, goal_cone_half_deg=45.0, goal_v_floor=3.0)),
    "T2 同锥45·异 goal_v_floor(2 vs 3) → 冲突（防 v_floor 混表）")

# ---------------- T3 import-time 守卫（子进程·env-var 行为）----------------
print("T3 import-time 守卫（子进程）")


def _imp(env):
    e = dict(os.environ)
    e.update(env)
    return subprocess.run(
        [sys.executable, "-B", "-c",
         "import sys;sys.path.insert(0,'.');import run_step4e as S;"
         "print('DEG', S._GOAL_CONE_HALF_DEG, 'RAD', S._GOAL_CONE_HALF_RAD, 'VF', S._GOAL_V_FLOOR)"],
        cwd=_ROOT, env=e, capture_output=True, text=True)


r = _imp({})
chk(r.returncode == 0 and "DEG None" in r.stdout, "T3 默认 off → DEG None（bit-identical）")
r = _imp({"STEP4E_GOAL_CONE_HALF": "45"})
_rad_ok = False
if r.returncode == 0 and "DEG 45.0" in r.stdout:
    try:
        _rad = float(r.stdout.split("RAD")[1].split("VF")[0].strip())   # 数值比·非仅子串（变异审 F1：deg→rad 只此一条守·须 pin 精确值）
        _rad_ok = abs(_rad - math.radians(45)) < 1e-9
    except Exception:
        _rad_ok = False
chk(r.returncode == 0 and "DEG 45.0" in r.stdout and _rad_ok, "T3 Φ=45 → DEG45 & RAD==radians(45) 数值验（度→弧度·非仅子串·变异审 F1）")
r = _imp({"STEP4E_GOAL_CONE_HALF": "0"})
chk(r.returncode == 0 and "DEG None" in r.stdout, "T3 Φ='0' → off(None)")
for bad, lbl in [("-5", "负"), ("181", ">180"), ("foo", "非数")]:
    r = _imp({"STEP4E_GOAL_CONE_HALF": bad})
    chk(r.returncode != 0 and "GOAL_CONE_HALF" in (r.stdout + r.stderr), f"T3 Φ={bad}({lbl}) → fail-fast")
r = _imp({"STEP4E_GOAL_CONE_HALF": "45", "STEP4E_GOAL_V_FLOOR": "-1"})
chk(r.returncode != 0 and "GOAL_V_FLOOR" in (r.stdout + r.stderr), "T3 v_floor<0 → fail-fast")
r = _imp({"STEP4E_GOAL_CONE_HALF": "45", "STEP4E_GOAL_V_FLOOR": "10"})
chk(r.returncode != 0 and "GOAL_V_FLOOR" in (r.stdout + r.stderr), "T3 v_floor>v_max(10>9.5) → fail-fast（上界对称·复审⑤·否则子进程盾层晚爆 EOFError）")
r = _imp({"STEP4E_GOAL_CONE_HALF": "45", "STEP4E_CONTINUOUS_SHIELD": "0"})
chk(r.returncode != 0 and "无盾" in (r.stdout + r.stderr), "T3 无盾+锥 → fail-fast（锥是盾内机制·无盾臂 step 短路不施）")

# ---------------- T1 maker 签名 + 透传 + 默认 bit-identical ----------------
print("T1 maker 签名 + 透传")
for f in (make_continuous_safe_model, make_continuous_safe_ppo_model):
    ps = inspect.signature(f).parameters
    chk("goal_cone_half" in ps and "goal_v_floor" in ps,
        f"T1 {f.__name__} 签名含 goal_cone_half/goal_v_floor")

_POOL = "/private/tmp/trb_scenarios_pool"
_PATHS = (sorted(os.path.join(_POOL, f) for f in os.listdir(_POOL) if f.endswith(".xml"))[:2]
          if os.path.isdir(_POOL) else [])
if not _PATHS:
    print(f"[SKIP] 场景池 {_POOL} 不存在 → 跳 T1 maker 透传【实构】断言。⚠️ 缺池时【maker→env_kwargs 透传 / 默认 bit-identical 未被验证】"
          f"（变异审 R3-R1：签名检查抓不住 silent no-op·绿≠完整通过）→ 有池环境（trb 服务器/本机）必跑。")
else:
    _SK = dict(buffer_size=2000, learning_starts=10, batch_size=32, subproc=False)

    def _cones(venv):
        # MultiScenarioEnv 把 ContinuousProjectionEnv 存 self._inner（每 reset 用 self.env_kwargs 重建·锥随之透传继承）→ 读 _inner.proj.goal_cone_half
        return [inner.proj.goal_cone_half for inner in venv.venv.get_attr("_inner")]

    for name, mk, kw in [("SAC", make_continuous_safe_model, _SK),
                         ("PPO", make_continuous_safe_ppo_model, dict(subproc=False))]:
        m, v = mk(paths=_PATHS, n_envs=1, seed=0, **kw)                       # 默认 → None（bit-identical）
        chk(all(c is None for c in _cones(v)), f"T1-{name} 默认 → proj.goal_cone_half 全 None（bit-identical 现状）")
        v.close()
        m, v = mk(paths=_PATHS, n_envs=1, seed=0, goal_cone_half=math.radians(45), goal_v_floor=3.0, **kw)
        cs = _cones(v)
        chk(all(c is not None and abs(c - math.radians(45)) < 1e-9 for c in cs),
            f"T1-{name} goal_cone_half=rad(45) 透传落地到 proj（maker→env_kwargs→ContinuousColregsProjection），得 {cs}")
        v.close()

print("\n" + ("=" * 50))
print("✅ 全部通过" if N_FAIL == 0 else f"❌ {N_FAIL} 项失败")
sys.exit(1 if N_FAIL else 0)
