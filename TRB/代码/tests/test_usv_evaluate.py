"""
评估管线冒烟测试（step4d-①）—— run_episode / evaluate 的 Table III 口径簿记逻辑。
跑：/opt/miniconda3/envs/trb/bin/python -B 代码/tests/test_usv_evaluate.py

设计（L11/L12 mutation-sensitive，不写 J9 式假守护）：
  用【脚本化 stub env】回放预定 (ego, obs, flags, rho, term/trunc) 序列，把 run_episode 的簿记逻辑
  与重物理 env 解耦、确定性单元测：
    · violations = 独立 ViolationCounter 重放同序列（oracle，非复用 run_episode 内部）→ 穿线正确
    · emergency_pct = 100·#(rho==EMERGENCY)/steps · ep_len_s = steps·dt · reached/collided = OR flags
    · evaluate 聚合算术 + per 列表保真
  非零违规由【已确认 crossing 的几何】驱动（ego 静止不机动 → 相遇不解除 → finalize +1），
  使"删 finalize / 漏 vc.step / 只数 standon"等变异必翻 FAIL（次窗口变异验证坐实）。
  ⚠️ 有盾 ShieldedUSVEnv 的端到端 violations 必为 0（mask 强制机动）→ 测不动违规路径；故此处用 stub
     脚本非零违规。无盾 Base/RR 真实轨迹违规留 4d-② 端到端补。
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from trb_env.evaluate import (run_episode, evaluate, run_episode_continuous, evaluate_continuous,
                              _FALLBACK_SOURCES)
from trb_env.usv_colregs import (ViolationCounter, VesselState, crossing, head_on, overtake,
                                 keep, T_HORIZON, DELTA_NO_TURN, RHO_EMERGENCY, RHO_CROSSING,
                                 RHO_NO_CONFLICT)
from trb_env.usv_env import N_ACTIONS_TOTAL

_fail = 0
def check(name, ok):
    global _fail
    if not ok: _fail += 1
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")

DT = 10.0


# ---- 脚本化 stub env：回放 (ego, obs, flags, rho, term, trunc) 序列 ----
class _Inner:
    def __init__(self, dt): self.dt = dt

class _StubEnv:
    """run_episode 的最小契约：reset / _ego_vs / _obs_vs / action_masks / step / .env.dt。"""
    def __init__(self, script, dt=DT):
        self.script = script; self.env = _Inner(dt); self.i = 0
    def reset(self, *, seed=None, options=None):
        self.i = 0; return np.zeros(27), {}
    def _ego_vs(self): return self.script[self.i]["ego"]
    def _obs_vs(self): return self.script[self.i]["obs"]
    def action_masks(self): return np.ones(N_ACTIONS_TOTAL, dtype=bool)
    def step(self, action):
        s = self.script[self.i]; self.i += 1
        info = {"flags": s["flags"], "rho": s["rho"]}
        # 可选 info 字段透传（缺则不带·向后兼容）：rho_acting(③/⑬ 缺则回退 rho) / source(连续归口) /
        # emergency_mode·u_desired·u_applied(连续臂 CAT4 诊断 load-bearing 守护用，㉛a-d)。
        for _k in ("rho_acting", "source", "emergency_mode", "u_desired", "u_applied"):
            if _k in s:
                info[_k] = s[_k]
        return (np.zeros(27), 0.0, s.get("term", False), s.get("trunc", False), info)

def _flags(goal=False, collision=False):
    return {"goal": goal, "collision": collision, "area": False, "stopped": False, "time": False}

def _policy(obs, mask):       # stub 忽略 action，返回任意合法下标
    return 0

def _term_script(state):
    """单步终止 stub script：决策态 state（带 term/trunc）+ 终止后状态 @k（去 term/trunc 标志）。
    run_episode 修复后会补喂终止后 @k（镜像真实 env：step 后 env 停在合法的 @k）→ stub 须提供该状态。"""
    post = {k: v for k, v in state.items() if k not in ("term", "trunc")}
    return [state, post]


# ---- 几何 1：已确认 crossing（ego 静止参考态 + 他船右舷向北穿越 ego 航路）----
EGO = VesselState(position=np.array([0.0, 0.0]), orientation=0.0, velocity=6.0, length=175.0)
def _he(t):
    return VesselState(position=np.array([2000.0, -2000.0 + 6.0 * t]),
                       orientation=np.pi / 2, velocity=6.0, length=175.0)
_cross_ts = [t for t in range(0, 400, 10) if crossing(EGO, _he(t), T_HORIZON)]
assert len(_cross_ts) >= 6, f"crossing 窗口不足（得 {len(_cross_ts)}）——调几何"
_WIN = _cross_ts[:6]   # 6 态：@0..@4 = 5 个决策步、@5 = 终止后状态 @k（run_episode 修复后补喂）
assert all(crossing(EGO, _he(t), T_HORIZON) and not head_on(EGO, _he(t), T_HORIZON)
           and not overtake(EGO, _he(t), T_HORIZON) and not keep(EGO, _he(t), T_HORIZON)
           for t in _WIN), "窗口内有非 crossing 态势叠加——期望计数不干净，调几何"

# ---- 几何 2：远处静止他船（无任何态势 → 0 违规，用于 reached/collided/聚合）----
HE_FAR = VesselState(position=np.array([1e5, 1e5]), orientation=0.0, velocity=0.0, length=175.0)
assert not any(f(EGO, HE_FAR, T_HORIZON) for f in (crossing, head_on, overtake, keep)), \
    "远船仍触发态势——调距离"


def _oracle(states):
    """独立 ViolationCounter 重放（非复用 run_episode 内部）= violations 的金标准。"""
    vc = ViolationCounter()
    for ego, obs in states:
        vc.step(ego, obs)
    vc.finalize()
    return vc.standon_violations + vc.giveway_violations


print("===== A) run_episode 簿记逻辑（stub 脚本回放）=====")
# 主脚本：5 步 crossing（ego 静止不机动）→ finalize 记 give-way 违规；rho 中 2 步 EMERGENCY → 40%
_RHO = [RHO_CROSSING, RHO_CROSSING, RHO_EMERGENCY, RHO_EMERGENCY, RHO_CROSSING, RHO_CROSSING]
# trunc 在 index 4（第 5 个决策步产生终止状态 @5）；index 5 = 终止后 @k（run_episode 修复后补喂、不再被 step）
_script = [{"ego": EGO, "obs": _he(t), "flags": _flags(), "rho": r, "trunc": (k == 4)}
           for k, (t, r) in enumerate(zip(_WIN, _RHO))]
_exp_v = _oracle([(s["ego"], s["obs"]) for s in _script])
res = run_episode(_StubEnv(_script), _policy)

check("① steps==5（回放到 trunc 即停）", res["steps"] == 5)
check("② ep_len_s==steps·dt==50", res["ep_len_s"] == 5 * DT)
check("③ emergency_pct==100·2/5==40", abs(res["emergency_pct"] - 40.0) < 1e-9)
check(f"④ violations==oracle 独立重放(={_exp_v}) 且非零（穿线正确，mutation 守护）",
      res["violations"] == _exp_v and _exp_v >= 1)
check("⑤ reached/collided==False（flags 全 False）",
      (not res["reached"]) and (not res["collided"]))

# reached / collided True 路径 + term 即停
_r6 = run_episode(_StubEnv(_term_script({"ego": EGO, "obs": HE_FAR, "flags": _flags(goal=True),
                                         "rho": RHO_NO_CONFLICT, "term": True})), _policy)
check("⑥ goal flag → reached==True、term 即停 steps==1",
      _r6["reached"] and (not _r6["collided"]) and _r6["steps"] == 1)
_r7 = run_episode(_StubEnv(_term_script({"ego": EGO, "obs": HE_FAR, "flags": _flags(collision=True),
                                         "rho": RHO_NO_CONFLICT, "term": True})), _policy)
check("⑦ collision flag → collided==True", bool(_r7["collided"]))

# max_steps 截断（脚本无 term/trunc → 跑满 max_steps）
_long = [{"ego": EGO, "obs": HE_FAR, "flags": _flags(), "rho": RHO_NO_CONFLICT} for _ in range(20)]
check("⑧ max_steps 截断 → steps==3", run_episode(_StubEnv(_long), _policy, max_steps=3)["steps"] == 3)

# ㉓ @k 守护（2026-06-11 审核 MAJOR 修复的变异守护，L11/L12）：stand-on(keep) 终止步转向 →
#   仅当 run_episode 补喂终止后状态 @k 才计 standon；漏 @k 则 @0..@4 全保向 cum=0 → standon=0（变异翻 FAIL）。
_KEEP_OBS = VesselState(position=np.array([-1500., -1500.]), orientation=0.987, velocity=6.0, length=175.0)
def _keep_ego(th):
    return VesselState(position=np.array([0., 0.]), orientation=float(th), velocity=6.0, length=175.0)
assert keep(_keep_ego(-np.pi), _KEEP_OBS, T_HORIZON), "keep 几何失效——调参"
_KTH = [-np.pi] * 5 + [-np.pi + 1.2 * DELTA_NO_TURN]   # @0..@4 保向(cum=0)、@5(=@k) 转 1.2·Δno_turn(>10°)
_kscript = [{"ego": _keep_ego(_KTH[k]), "obs": _KEEP_OBS, "flags": _flags(),
             "rho": RHO_NO_CONFLICT, "trunc": (k == 4)} for k in range(6)]
_k_oracle = _oracle([(s["ego"], s["obs"]) for s in _kscript])    # 完整轨迹 @0..@5 金标准
_rk = run_episode(_StubEnv(_kscript), _policy)
check(f"㉓ @k 守护：keep 终止步转向 → violations=={_k_oracle}（补喂 @k；漏 @k 则 standon=0≠{_k_oracle}）",
      _rk["violations"] == _k_oracle and _k_oracle == 1)

# ㉔ None 守护（跨场景鲁棒，Agent A @k 复核发现）：他船预测窗外 _obs_vs()=None 时，循环内(line38)与终止 @k 都不崩、跳过该步。
#   stub 中段 + @k 插 obs=None（模拟他船出预测窗）→ run_episode 不崩（漏守护则 ViolationCounter.step(ego,None) AttributeError）。
_none_script = [
    {"ego": EGO, "obs": _he(_WIN[0]), "flags": _flags(), "rho": RHO_CROSSING},
    {"ego": EGO, "obs": None, "flags": _flags(), "rho": RHO_NO_CONFLICT},                        # 他船窗外（line-38 守护）
    {"ego": EGO, "obs": _he(_WIN[1]), "flags": _flags(), "rho": RHO_CROSSING, "trunc": True},
    {"ego": EGO, "obs": None, "flags": _flags(), "rho": RHO_NO_CONFLICT},                        # @k 也窗外（@k 守护）
]
try:
    _rn = run_episode(_StubEnv(_none_script), _policy); _none_ok = (_rn["steps"] == 3)
except Exception:
    _none_ok = False
check("㉔ None 守护：他船窗外 _obs_vs=None 时循环内+@k 都不崩、steps==3（跨场景鲁棒）", _none_ok)

print("\n===== B) evaluate 聚合算术 + per 保真 =====")
# 两场景：A 远船到达 0 违规 / B 主脚本未到达 + _exp_v 违规 + 40% 紧急
_scA = _term_script({"ego": EGO, "obs": HE_FAR, "flags": _flags(goal=True),
                     "rho": RHO_NO_CONFLICT, "term": True})
scenarios = [(_scA, None), (_script, None)]
agg, per = evaluate(lambda sc, pp: _StubEnv(sc), _policy, scenarios)
check("⑨ n==2", agg["n"] == 2)
check("⑩ 到达率%==50（A 到 / B 未到）", abs(agg["到达率%"] - 50.0) < 1e-9)
check("⑪ 碰撞率%==0", agg["碰撞率%"] == 0.0)
check(f"⑫ 违规次数/局==(0+{_exp_v})/2", abs(agg["违规次数/局"] - _exp_v / 2) < 1e-9)
check("⑬ 紧急步%==(0+40)/2==20", abs(agg["紧急步%"] - 20.0) < 1e-9)
check("⑭ Ep长s==(10+50)/2==30", abs(agg["Ep长s"] - (1 * DT + 5 * DT) / 2) < 1e-9)
check("⑮ per 列表逐局保真（len==2、第二局 violations==oracle）",
      len(per) == 2 and per[1]["violations"] == _exp_v)
# n≠2 守护 evaluate 除真实局数（非硬编 2）+ 空场景防除零（复核 Agent 2 MUT-H 缺口）
_agg1, _ = evaluate(lambda sc, pp: _StubEnv(sc), _policy, [(_scA, None)])
check("⑯ evaluate n==1：到达率%==100（除真实局数非硬编 2 → 杀 MUT-H）",
      _agg1["n"] == 1 and abs(_agg1["到达率%"] - 100.0) < 1e-9)
_agg0, _per0 = evaluate(lambda sc, pp: _StubEnv(sc), _policy, [])
check("⑰ evaluate n==0（空场景）：不崩、各率 0、per==[]",
      _agg0["n"] == 0 and _agg0["到达率%"] == 0.0 and _per0 == [])

print("\n===== C) policy.predict 分支 + 真实 ShieldedUSVEnv 端到端 =====")
# policy 有 .predict（sb3 模型路径）→ 走 predict 分支 + action_masks 传参 + int(np.int64) 转换
class _PredictPolicy:
    def __init__(self): self.got_mask = None
    def predict(self, obs, action_masks=None, deterministic=True):
        self.got_mask = action_masks
        return np.int64(0), None
_pol = _PredictPolicy()
_rp = run_episode(_StubEnv(_term_script({"ego": EGO, "obs": HE_FAR, "flags": _flags(goal=True),
                                         "rho": RHO_NO_CONFLICT, "term": True})), _pol)
check("⑱ policy.predict 分支被选用 + action_masks 正确传入 + 50 维",
      _rp["reached"] and _pol.got_mask is not None and len(_pol.got_mask) == N_ACTIONS_TOTAL)

# ㉕ obs_transform 钩子（VecNormalize eval 接线，D22/L19）：run_episode 把【变换后】obs 喂 policy.predict（非原始）
class _ObsRecPolicy:
    def __init__(self): self.seen = []
    def predict(self, obs, action_masks=None, deterministic=True):
        self.seen.append(np.array(obs, dtype=float)); return np.int64(0), None
_orp = _ObsRecPolicy()
run_episode(_StubEnv(_term_script({"ego": EGO, "obs": HE_FAR, "flags": _flags(goal=True),
                                   "rho": RHO_NO_CONFLICT, "term": True})), _orp,
            obs_transform=lambda o: o * 0.0 + 7.0)               # 变换：把 obs 恒置 7
check("㉕ obs_transform 钩子：变换后 obs（恒7）喂 predict（漏接则喂原始 0）",
      len(_orp.seen) >= 1 and np.allclose(_orp.seen[0], 7.0))

# 真实 ShieldedUSVEnv(T-0) 端到端：覆盖 standon 违规路径（杀 MUT-J 漏 standon）+ 集成自洽。离线 SKIP。
_T0 = "/tmp/trb_T0.xml"; _skip_C = False
try:
    if not os.path.exists(_T0):
        import urllib.request
        urllib.request.urlretrieve(
            "https://gitlab.lrz.de/tum-cps/commonocean-scenarios/-/raw/main/scenarios/"
            "HandcraftedTwoVesselEncounters_01_24/ZAM_AAA-1_20240121_T-0.xml", _T0)
    from commonocean.common.file_reader import CommonOceanFileReader
    from trb_env.usv_shield import ShieldedUSVEnv, _AKEEP_IDX
    _sc, _pp = CommonOceanFileReader(_T0).open()
    _ppx = list(_pp.planning_problem_dict.values())[0]
except Exception as e:
    _skip_C = True
    print(f"[SKIP] C 段真实 env 需 /tmp/trb_T0.xml（离线下载失败 {type(e).__name__}）；纯 stub A/B 仍验逻辑")

if not _skip_C:
    def _keep_heading(obs, mask):                        # 保持航向；ρ5 处退第一合法（紧急槽）
        return _AKEEP_IDX if mask[_AKEEP_IDX] else int(np.where(mask)[0][0])
    _env = ShieldedUSVEnv(_sc, _ppx); _o, _inf = _env.reset(seed=0)   # 独立重放真实轨迹算 oracle（含 @k）
    _vcC = ViolationCounter()
    while True:
        _vcC.step(_env._ego_vs(), _env._obs_vs())
        _o, _r, _t, _tr, _inf = _env.step(_keep_heading(_o, _env.action_masks()))
        if _t or _tr: break
    _sofC = _env._obs_vs()                                   # 终止后 @k（与 run_episode 同口径）
    if _sofC is not None:
        _vcC.step(_env._ego_vs(), _sofC)
    _vcC.finalize()
    _so, _gw = _vcC.standon_violations, _vcC.giveway_violations
    _resC = run_episode(ShieldedUSVEnv(_sc, _ppx), _keep_heading)     # 独立第二次跑
    check("⑲ 真实 T-0：run_episode.violations == 独立重放 oracle（standon+giveway）",
          _resC["violations"] == _so + _gw)
    check(f"⑳ T-0 含 stand-on 违规 standon={_so}≥1（覆盖 standon 路径 → 杀 MUT-J 漏 standon）",
          _so >= 1)
    check(f"㉑ T-0 violations==1（standon={_so},giveway={_gw}；紧急转向撞裸 keep 谓词 = 03 口径实例）",
          _resC["violations"] == 1 and _so == 1 and _gw == 0)
    check("㉒ T-0 确定性+自洽：steps=170 / em%≈4.118 / collided=False / reached=False / ep_len=1700",
          _resC["steps"] == 170 and abs(_resC["emergency_pct"] - 100 * 7 / 170) < 1e-6
          and (not _resC["collided"]) and (not _resC["reached"]) and _resC["ep_len_s"] == 1700.0)

print("\n===== D) Node C C1：紧急步%两臂口径统一(rho_acting/pre-step) + 连续臂评估 + 对拍 =====")

class _ConstModel:
    """连续策略 stub：.predict(obs,deterministic)→(常值动作,None)；记录 seen 供 obs_transform 核。"""
    def __init__(self, a=(0.05, 0.02)): self.a = np.array(a, float); self.seen = []
    def predict(self, obs, deterministic=True):
        self.seen.append(np.asarray(obs)); return self.a, None

# ㉖ run_episode 紧急步%数 rho_acting(pre-step ρ@t)，非 rho(post-step ρ@(t+1))：rho_acting=[EMER,EMER]/rho=[CROSS,CROSS]
#    → 数 rho_acting 得 100%、若仍数 rho 得 0%（D40 #1 pre-step 口径变异守护；HE_FAR 无态势=无违规噪声）
_distinct = [{"ego": EGO, "obs": HE_FAR, "flags": _flags(), "rho": RHO_CROSSING, "rho_acting": RHO_EMERGENCY},
             {"ego": EGO, "obs": HE_FAR, "flags": _flags(), "rho": RHO_CROSSING, "rho_acting": RHO_EMERGENCY, "trunc": True},
             {"ego": EGO, "obs": HE_FAR, "flags": _flags(), "rho": RHO_NO_CONFLICT, "rho_acting": RHO_NO_CONFLICT}]
check("㉖ run_episode 数 rho_acting(pre-step)≠rho(post-step) → em%==100（D40#1 口径生效；旧 post-step 会得 0）",
      abs(run_episode(_StubEnv(_distinct), _policy)["emergency_pct"] - 100.0) < 1e-9)

# ㉗ rho_acting 缺省回退 rho（stub ③/⑬ 无 rho_acting 仍按 rho 计 = 向后兼容、不破既有口径）
_fallback_rho = [{"ego": EGO, "obs": HE_FAR, "flags": _flags(), "rho": RHO_EMERGENCY},
                 {"ego": EGO, "obs": HE_FAR, "flags": _flags(), "rho": RHO_NO_CONFLICT, "trunc": True},
                 {"ego": EGO, "obs": HE_FAR, "flags": _flags(), "rho": RHO_NO_CONFLICT}]
check("㉗ 无 rho_acting 时回退 info[rho]（向后兼容，stub ③/⑬ 口径不破）：em%==50",
      abs(run_episode(_StubEnv(_fallback_rho), _policy)["emergency_pct"] - 50.0) < 1e-9)

# ㉘ run_episode_continuous 簿记：紧急步%=rho_acting==ρ5、fallback_pct=source∈{relaxed,collision_min,degenerate}
#    ⭐【非对称】1 紧急 + 2 兜底(collision_min) → em%=1/3≠fallback%=2/3：钉死 fallback 谓词（若误把 `in _FALLBACK_SOURCES`
#    改成 `=="emergency"`，fallback% 会变 1/3 → 本断言翻 FAIL；对称 1+1 则蒙混过关，故必非对称）。
_cscript = [{"ego": EGO, "obs": HE_FAR, "flags": _flags(), "rho": RHO_NO_CONFLICT, "rho_acting": RHO_EMERGENCY, "source": "emergency"},
            {"ego": EGO, "obs": HE_FAR, "flags": _flags(), "rho": RHO_NO_CONFLICT, "rho_acting": RHO_NO_CONFLICT, "source": "collision_min"},
            {"ego": EGO, "obs": HE_FAR, "flags": _flags(goal=True), "rho": RHO_NO_CONFLICT, "rho_acting": RHO_NO_CONFLICT, "source": "collision_min", "term": True},
            {"ego": EGO, "obs": HE_FAR, "flags": _flags(goal=True), "rho": RHO_NO_CONFLICT, "rho_acting": RHO_NO_CONFLICT, "source": "projection"}]
_cb = run_episode_continuous(_StubEnv(_cscript), _ConstModel())
check("㉘ run_episode_continuous：em%==1/3、fallback_pct==2/3(2×collision_min·非对称钉死谓词)、reached、3 决策步",
      _cb["steps"] == 3 and abs(_cb["emergency_pct"] - 100/3) < 1e-9
      and abs(_cb["fallback_pct"] - 200/3) < 1e-9 and _cb["reached"])

# ㉙ ⭐对拍：离散 run_episode vs 连续 run_episode_continuous 喂【同轨迹】→ violations/em%/reached/collided 逐项相等
#    （D40 035 机制性"四方同 ego 轨迹对拍"——caliber 单源、跨臂不分叉；含 crossing give-way 违规 + 紧急步）
_copra = [{"ego": EGO, "obs": _he(_WIN[0]), "flags": _flags(), "rho": RHO_CROSSING, "rho_acting": RHO_CROSSING, "source": "projection"},
          {"ego": EGO, "obs": _he(_WIN[1]), "flags": _flags(), "rho": RHO_CROSSING, "rho_acting": RHO_EMERGENCY, "source": "emergency"},
          {"ego": EGO, "obs": _he(_WIN[2]), "flags": _flags(), "rho": RHO_CROSSING, "rho_acting": RHO_EMERGENCY, "source": "emergency", "trunc": True},
          {"ego": EGO, "obs": _he(_WIN[3]), "flags": _flags(), "rho": RHO_NO_CONFLICT, "rho_acting": RHO_NO_CONFLICT, "source": "projection"}]
_d = run_episode(_StubEnv([dict(s) for s in _copra]), _policy)
_c = run_episode_continuous(_StubEnv([dict(s) for s in _copra]), _ConstModel())
check("㉙ ⭐对拍：离散vs连续同轨迹 → violations/em%/reached/collided 逐项相等（跨臂 caliber 单源不分叉）",
      _d["violations"] == _c["violations"] and abs(_d["emergency_pct"] - _c["emergency_pct"]) < 1e-9
      and _d["reached"] == _c["reached"] and _d["collided"] == _c["collided"])
check("㉙b 对拍非平凡：该轨迹确含违规且 em%>0（否则对拍是空相等）",
      _d["violations"] >= 1 and _d["emergency_pct"] > 0.0)

# ㉚ _FALLBACK_SOURCES 不含 emergency/projection/no_obstacle（口径定义正确：兜底≠紧急≠常规）
check("㉚ _FALLBACK_SOURCES=={relaxed,collision_min,degenerate}（兜底 source 归口定义）",
      _FALLBACK_SOURCES == frozenset({"relaxed", "collision_min", "degenerate"}))

# ㉛ evaluate_continuous 聚合：含「兜底步%」列 + n 正确
_aggc, _perc = evaluate_continuous(lambda sc, pp: _StubEnv([dict(s) for s in _cscript]),
                                   _ConstModel(), [(None, None), (None, None)])
check("㉛ evaluate_continuous 聚合：n==2 + 含「兜底步%」列（连续臂诊断）",
      _aggc["n"] == 2 and "兜底步%" in _aggc and "紧急步%" in _aggc)

# ── Node L L1b CAT3/4 诊断【load-bearing 数值守护】（审核补：原 ㉝b/c/e/f 仅一致性/范围校验、对诊断算错不设防，review L1b-F1/F2/F3）──
from trb_env.evaluate import _EGO_CIRC
from trb_env.usv_colregs import _vessel_circumradius

# ㉛a 投影修正量 ‖u_applied−u_desired‖（action aliasing 头号卖点）命中解析值：漏减 u_desired（norm(ua) 替 norm(ua−ud)）必翻 FAIL
_ud0, _ua0 = (0.05, 0.02), (0.04, 0.01)     # 修正=‖[0.01,0.01]‖≈0.0141421356
_corr_script = [
    {"ego": EGO, "obs": HE_FAR, "flags": _flags(), "rho": RHO_NO_CONFLICT, "rho_acting": RHO_NO_CONFLICT,
     "source": "projection", "u_desired": _ud0, "u_applied": _ua0},
    {"ego": EGO, "obs": HE_FAR, "flags": _flags(goal=True), "rho": RHO_NO_CONFLICT, "rho_acting": RHO_NO_CONFLICT,
     "source": "projection", "u_desired": _ud0, "u_applied": _ud0, "term": True},   # 修正=0（u_applied==u_desired）
    {"ego": EGO, "obs": HE_FAR, "flags": _flags(goal=True), "rho": RHO_NO_CONFLICT, "rho_acting": RHO_NO_CONFLICT, "source": "projection"},  # @k 不计
]
_c0 = float(np.linalg.norm(np.array(_ua0) - np.array(_ud0))); _c1 = 0.0
_rc = run_episode_continuous(_StubEnv(_corr_script), _ConstModel())
check("㉛a 投影修正量 load-bearing：mean/max/zero_frac/n_shield 命中解析值（漏减 u_desired 必翻 FAIL）",
      _rc["proj_correction_mean"] == round((_c0 + _c1) / 2, 6)
      and _rc["proj_correction_max"] == round(max(_c0, _c1), 6)
      and _rc["proj_correction_zero_frac"] == 0.5 and _rc["n_shield_steps"] == 2)

# ㉛b CPA 安全裕度命中解析值：clearance 公式错(多扣半径)/ min→max 取反 / cpa_step 记错 任一必翻 FAIL（原 ㉝b 仅验 clearance≤center）
def _obs_at(d):
    return VesselState(position=np.array([float(d), 0.0]), orientation=0.0, velocity=6.0, length=175.0)
_cpa_script = [
    {"ego": EGO, "obs": _obs_at(1000.0), "flags": _flags(), "rho": RHO_NO_CONFLICT, "rho_acting": RHO_NO_CONFLICT, "source": "projection"},
    {"ego": EGO, "obs": _obs_at(500.0),  "flags": _flags(), "rho": RHO_NO_CONFLICT, "rho_acting": RHO_NO_CONFLICT, "source": "projection"},  # 最近·步idx=1
    {"ego": EGO, "obs": _obs_at(800.0),  "flags": _flags(goal=True), "rho": RHO_NO_CONFLICT, "rho_acting": RHO_NO_CONFLICT, "source": "projection", "term": True},
    {"ego": EGO, "obs": _obs_at(900.0),  "flags": _flags(goal=True), "rho": RHO_NO_CONFLICT, "rho_acting": RHO_NO_CONFLICT, "source": "projection"},  # @k·d=900>500
]
_rcpa = run_episode_continuous(_StubEnv(_cpa_script), _ConstModel())
_exp_clear = round(500.0 - _EGO_CIRC - _vessel_circumradius(175.0, None), 3)
check("㉛b CPA load-bearing：cpa_center==500/clearance==解析(单扣双圆)/step==1（公式错·min→max·step错 必翻 FAIL）",
      _rcpa["cpa_center_m"] == 500.0 and _rcpa["cpa_clearance_m"] == _exp_clear and _rcpa["cpa_step"] == 1)

# ㉛c/㉛d source 六档 + emergency_modes 三档【逐桶】命中：桶间错路(relaxed→degenerate / ahead→stern)必翻 FAIL（原 ㉝c/㉝f 仅验和）
_bucket_script = [
    {"ego": EGO, "obs": HE_FAR, "flags": _flags(), "rho": RHO_NO_CONFLICT, "rho_acting": RHO_NO_CONFLICT, "source": "relaxed"},
    {"ego": EGO, "obs": HE_FAR, "flags": _flags(), "rho": RHO_NO_CONFLICT, "rho_acting": RHO_NO_CONFLICT, "source": "collision_min"},
    {"ego": EGO, "obs": HE_FAR, "flags": _flags(), "rho": RHO_NO_CONFLICT, "rho_acting": RHO_NO_CONFLICT, "source": "degenerate"},
    {"ego": EGO, "obs": HE_FAR, "flags": _flags(), "rho": RHO_NO_CONFLICT, "rho_acting": RHO_NO_CONFLICT, "source": "projection"},
    {"ego": EGO, "obs": HE_FAR, "flags": _flags(), "rho": RHO_NO_CONFLICT, "rho_acting": RHO_EMERGENCY, "source": "emergency", "emergency_mode": "ahead"},
    {"ego": EGO, "obs": HE_FAR, "flags": _flags(goal=True), "rho": RHO_NO_CONFLICT, "rho_acting": RHO_EMERGENCY, "source": "emergency", "emergency_mode": "stern", "term": True},
    {"ego": EGO, "obs": HE_FAR, "flags": _flags(goal=True), "rho": RHO_NO_CONFLICT, "rho_acting": RHO_NO_CONFLICT, "source": "projection"},  # @k 不计 source
]
_rb = run_episode_continuous(_StubEnv(_bucket_script), _ConstModel())
_sc6, _em3 = _rb["source_counts"], _rb["emergency_modes"]
check("㉛c source 六档 per-bucket load-bearing：relaxed/collision_min/degenerate/projection/emergency/no_obstacle 逐档命中（桶间错路必翻 FAIL）",
      _sc6 == {"relaxed": 1, "collision_min": 1, "degenerate": 1, "projection": 1, "emergency": 2, "no_obstacle": 0})
check("㉛d emergency_modes per-bucket load-bearing：ahead==1/stern==1/base==0（ahead→stern 错路必翻 FAIL）",
      _em3 == {"ahead": 1, "stern": 1, "base": 0})

if not _skip_C:
    # ㉜ 真实 T-0 连续臂：run_episode_continuous 跑 ContinuousProjectionEnv（不调 action_masks=连续 env 无此方法→若误调即 AttributeError）
    from trb_env.usv_continuous_shield import ContinuousProjectionEnv
    _resCC = run_episode_continuous(ContinuousProjectionEnv(_sc, _ppx), _ConstModel(a=(0.05, 0.02)))
    check("㉜ 真实 T-0 连续臂跑通（不调 action_masks）+ 指标自洽（em%/fallback∈[0,100]、ep_len=steps·dt）",
          _resCC["steps"] >= 1 and 0.0 <= _resCC["emergency_pct"] <= 100.0
          and 0.0 <= _resCC["fallback_pct"] <= 100.0
          and abs(_resCC["ep_len_s"] - _resCC["steps"] * 10.0) < 1e-9)

    # ── Node L CAT2/3/4 逐 episode 诊断字段（additive·不破钱图 5 列）──
    check("㉝ CAT2 违规分解：violations == standon_violations + giveway_violations（连续臂·钱图 violations 不变）",
          _resCC["violations"] == _resCC["standon_violations"] + _resCC["giveway_violations"])
    check("㉝a CAT2 rho_hist 6 档 ρ0-5 步数和 == steps（态势 profile 完整无漏）",
          sum(_resCC["rho_hist"].values()) == _resCC["steps"] and set(_resCC["rho_hist"]) == {0, 1, 2, 3, 4, 5})
    check("㉝b CAT3 CPA：cpa_center_m 非 None 且 cpa_clearance_m ≤ cpa_center_m（圆盘间隙=中心距−双圆<中心距）",
          _resCC["cpa_center_m"] is not None and _resCC["cpa_clearance_m"] <= _resCC["cpa_center_m"])
    check("㉝c CAT4 source_counts 六档和 == steps（兜底链逐档分解·无合并漏计）",
          sum(_resCC["source_counts"].values()) == _resCC["steps"]
          and set(_resCC["source_counts"]) == {"projection", "emergency", "relaxed", "collision_min", "degenerate", "no_obstacle"})
    check("㉝d CAT4 紧急步数自洽：source_counts['emergency'] == rho_hist[5]（emergency⟺ρ5）",
          _resCC["source_counts"]["emergency"] == _resCC["rho_hist"][5])
    check("㉝e CAT4 投影修正量（action aliasing）：mean/max/zero_frac 非 None、zero_frac∈[0,1]、max≥mean≥0",
          _resCC["proj_correction_mean"] is not None and _resCC["proj_correction_max"] is not None
          and 0.0 <= _resCC["proj_correction_zero_frac"] <= 1.0
          and _resCC["proj_correction_max"] >= _resCC["proj_correction_mean"] >= 0.0)
    check("㉝f CAT4 紧急模式分布 ahead/stern/base 和 == source_counts['emergency']（每紧急步恰一模式）",
          sum(_resCC["emergency_modes"].values()) == _resCC["source_counts"]["emergency"])
    # Step-0 进近诊断（连续臂·真实 T-0·additive）：4 键存在 + 自洽 + 与 is_reached 一致
    check("㉝i Step-0 进近诊断 7 键存在且自洽：真env全非None / min≥0 / in_box_aligned≤in_box_steps / reached⟹in_box_aligned≥1 / 进框⟹min≤半长+裕度 / speed_at_min≤max_speed / reversals≥0（同源 is_reached·丢 headings/speeds.append 即翻 FAIL）",
          all(k in _resCC for k in ("min_goal_dist_m", "heading_err_at_min_deg", "in_box_steps", "in_box_aligned_steps", "speed_at_min_ms", "max_speed_ms", "speed_reversals"))
          and _resCC["min_goal_dist_m"] is not None and _resCC["min_goal_dist_m"] >= 0.0
          and _resCC["in_box_steps"] is not None and _resCC["in_box_aligned_steps"] is not None   # 真 env(goal_geom 全 + headings 齐)→非 None（丢 append/长度错=None→FAIL）
          and _resCC["in_box_aligned_steps"] <= _resCC["in_box_steps"]
          and (not _resCC["reached"] or _resCC["in_box_aligned_steps"] >= 1)
          and (_resCC["in_box_steps"] == 0 or _resCC["min_goal_dist_m"] <= 205.0)   # 进过框⟹最近≤半长+裕度
          and _resCC["speed_at_min_ms"] is not None and _resCC["max_speed_ms"] is not None   # 真 env 速度齐→非 None（丢 speeds.append=None→FAIL）
          and _resCC["speed_at_min_ms"] <= _resCC["max_speed_ms"] + 1e-9 and _resCC["max_speed_ms"] >= 0.0   # 最近点速度≤全程峰值
          and _resCC["speed_reversals"] is not None and _resCC["speed_reversals"] >= 0)  # 摆动次数非负整数
    # 离散臂也带 CAT2/3（无 CAT4 投影诊断）
    _resCD = run_episode(ShieldedUSVEnv(_sc, _ppx), _keep_heading)
    check("㉝g 离散臂也带 CAT2/3：violations==standon+giveway、rho_hist 和==steps、cpa_center_m 非 None",
          _resCD["violations"] == _resCD["standon_violations"] + _resCD["giveway_violations"]
          and sum(_resCD["rho_hist"].values()) == _resCD["steps"] and _resCD["cpa_center_m"] is not None)
    # CAT2 evaluate 单局明细带 scenario_idx（可追溯）
    _aggC, _perC = evaluate_continuous(lambda s, p: ContinuousProjectionEnv(s, p),
                                       _ConstModel(a=(0.05, 0.02)), [(_sc, _ppx), (_sc, _ppx)])
    check("㉝h CAT2 evaluate_continuous 单局明细带 scenario_idx（可追溯到场景）+ 含诊断字段",
          [p["scenario_idx"] for p in _perC] == [0, 1] and all("proj_correction_mean" in p for p in _perC))

    # ── Node L CAT5 示例轨迹（additive·默认 record_traj=False 钱图路径逐位不变）──
    _r5off = run_episode(ShieldedUSVEnv(_sc, _ppx), _keep_heading, seed=0, record_traj=False)
    _r5on = run_episode(ShieldedUSVEnv(_sc, _ppx), _keep_heading, seed=0, record_traj=True)
    check("㉞ CAT5 record_traj：False→无 traj 键（默认 dict 形状不变）/ True→traj 列表 len==steps、每步含 ego/他船位姿+step+rho、step 单调",
          "traj" not in _r5off and isinstance(_r5on.get("traj"), list)
          and len(_r5on["traj"]) == _r5on["steps"] and len(_r5on["traj"]) > 0
          and all({"ego_x", "ego_y", "ego_psi", "obs_x", "obs_y", "obs_psi", "step", "rho"} <= set(e) for e in _r5on["traj"])
          and [e["step"] for e in _r5on["traj"]] == list(range(_r5on["steps"])))
    check("㉞a CAT5 additive【load-bearing】：record_traj True vs False 除 traj/goal 外所有字段逐位相等（CAT5 记录误改钱图/CAT 字段即翻 FAIL）",
          set(_r5on) - set(_r5off) == {"traj", "goal"} and all(_r5off[k] == _r5on[k] for k in _r5off))
    _r5c = run_episode_continuous(ContinuousProjectionEnv(_sc, _ppx), _ConstModel(a=(0.05, 0.02)), seed=0, record_traj=True)
    check("㉞b CAT5 连续臂 traj 多记 source（盾归口）、离散臂 traj 无 source",
          len(_r5c["traj"]) > 0 and "source" in _r5c["traj"][0] and "source" not in _r5on["traj"][0])
    _scn3 = [(_sc, _ppx), (_sc, _ppx), (_sc, _ppx)]
    _agg5n, _per5n = evaluate(lambda s, p: ShieldedUSVEnv(s, p), _keep_heading, _scn3, traj_idxs=None)
    _agg5s, _per5s = evaluate(lambda s, p: ShieldedUSVEnv(s, p), _keep_heading, _scn3, traj_idxs={1})
    check("㉞c CAT5 evaluate traj_idxs：仅选中场景带 traj + agg 钱图 5 列 None vs {1} 逐位相等（additive 不污染聚合）",
          [i for i, p in enumerate(_per5s) if "traj" in p] == [1] and all("traj" not in p for p in _per5n)
          and all(_agg5n[k] == _agg5s[k] for k in ("到达率%", "碰撞率%", "违规次数/局", "紧急步%", "Ep长s")))
    check("㉞d CAT5 goal：record_traj True→out['goal']=[x,y] 目标区中心(例图绘 goal★·离散+连续臂均记)/ False→无 goal 键(additive)",
          "goal" not in _r5off
          and isinstance(_r5on.get("goal"), list) and len(_r5on["goal"]) == 2
          and isinstance(_r5c.get("goal"), list) and len(_r5c["goal"]) == 2)

# ---------------- CAT7 控制质量（连续臂细控优势量化·additive·不入钱图5列·L72）----------------
from trb_env.evaluate import _control_quality as _cq, _agg_ctrl as _ac, _CTRL_KEYS as _CK
import math as _math
_g = _cq([[0.048, 0.0], [0.0, 0.018], [0.048, 0.0]], [[0., 0.], [3., 4.], [3., 4.]])
check("㉟ CAT7 _control_quality 数学（jerk=√2 / accel=0.048 / yaw=0.018 / effort=1 / path=5·手算逐项·容差容 round(,6)）",
      abs(_g["ctrl_jerk_norm_mean"] - _math.sqrt(2)) < 1e-6 and abs(_g["accel_incr_mean"] - 0.048) < 1e-6
      and abs(_g["yaw_incr_mean"] - 0.018) < 1e-6 and abs(_g["ctrl_effort_norm_mean"] - 1.0) < 1e-6
      and abs(_g["path_len_m"] - 5.0) < 1e-6)
_smooth = _cq([[0.01, 0.005]] * 5, None)["ctrl_jerk_norm_mean"]                      # 恒定动作→增量0→jerk0
_jumpy = _cq([[0.048, 0.018], [-0.048, -0.018]] * 3, None)["ctrl_jerk_norm_mean"]    # 满幅来回→jerk≫0
check("㉟a CAT7 jerk 区分平滑(恒定→0) vs 抖动(满幅来回→≫1)（load-bearing·去归一化/错公式翻 FAIL）",
      _smooth is not None and _smooth < 1e-9 and _jumpy is not None and _jumpy > 1.0)
check("㉟b CAT7 边界：空→全 None / 单步→jerk None·effort 有值·path None（数据不足不编）",
      all(_cq([], [[0, 0]])[k] is None for k in _CK)
      and _cq([[0.01, 0.01]], [[0, 0]])["ctrl_jerk_norm_mean"] is None
      and _cq([[0.01, 0.01]], [[0, 0]])["ctrl_effort_norm_mean"] is not None)
check("㉟c CAT7 _agg_ctrl 跳 None（混入无指标局→只均有效局·全 None→该键 None）",
      _ac([{**{k: None for k in _CK}, "ctrl_jerk_norm_mean": 0.2},
           {k: None for k in _CK}])["ctrl_jerk_norm_mean"] == 0.2
      and _ac([{k: None for k in _CK}])["path_len_m"] is None)
_mixed = _cq([[0.01, 0.005], [0.01, 0.005], [0.24, 0.03], [0.01, 0.005], [0.01, 0.005]], None)  # 中间一步=紧急满程(越 RL 箱)
check("㉟d CAT7 排除紧急/兜底步（执行控制越 RL 箱 ±0.24/±0.03 不计入 jerk/effort·红队 MAJOR L72·去过滤翻 FAIL）",
      _mixed["ctrl_jerk_norm_mean"] is not None and _mixed["ctrl_jerk_norm_mean"] < 1e-9          # 恒定正常步间增量0·紧急步被排
      and _mixed["ctrl_effort_norm_mean"] is not None and _mixed["ctrl_effort_norm_mean"] < 1.0)  # effort 不被满程 ‖û‖≈5.3 污染

# ---------------- Step-0 进近诊断 _approach_diag（失败局机制分类·additive·纯几何后处理）----------------
from trb_env.evaluate import _approach_diag as _ad, _in_rect as _ir, _ang_in_gate as _aig, _APPROACH_KEYS as _AK
import math as _m
# 合成目标框：中心(0,0)·长 400(x·±200)·宽 60(y·±30)·朝向门 ±0.17rad·闭合顶点环
_GG = {"center": [0.0, 0.0], "orient_lo": -0.17, "orient_hi": 0.17,
       "rect_length": 400.0, "rect_width": 60.0, "rect_orientation": 0.0,
       "vertices": [[-200.0, -30.0], [-200.0, 30.0], [200.0, 30.0], [200.0, -30.0], [-200.0, -30.0]]}
# A) 干净进近·朝向对齐进框：min=0 / 朝向差=0 / 2 步在框·2 步对齐
_aA = _ad([[-1000., 0.], [-100., 0.], [0., 0.]], [0., 0., 0.], _GG)
check("㊱ Step-0 A 干净进框对齐：min_goal_dist=0 / heading_err=0 / in_box=2 / in_box_aligned=2",
      abs(_aA["min_goal_dist_m"] - 0.0) < 1e-9 and abs(_aA["heading_err_at_min_deg"] - 0.0) < 1e-9
      and _aA["in_box_steps"] == 2 and _aA["in_box_aligned_steps"] == 2)
# B) 进框但朝向错=门口捕获失败（heading-capture）：进框 1 步·对齐 0·朝向差≈86°（load-bearing 分类）
_aB = _ad([[-1000., 0.], [10., 0.]], [0., 1.5], _GG)
check("㊱a Step-0 B 门口捕获失败：in_box=1 但 in_box_aligned=0 + heading_err≈85.94°（进框·朝向没对上）",
      _aB["in_box_steps"] == 1 and _aB["in_box_aligned_steps"] == 0
      and abs(_aB["heading_err_at_min_deg"] - _m.degrees(1.5)) < 0.01 and abs(_aB["min_goal_dist_m"] - 10.0) < 1e-9)
# C) 接近但进不了框=横向 cross-track（停框外窄 y 带外）：in_box=0·min 小
_aC = _ad([[-1000., 0.], [0., 50.]], [0., 0.], _GG)
check("㊱b Step-0 C 横向进不了框：in_box=0 + in_box_aligned=0 + min_goal_dist=50（y=50 出 ±30 窄带）",
      _aC["in_box_steps"] == 0 and _aC["in_box_aligned_steps"] == 0 and abs(_aC["min_goal_dist_m"] - 50.0) < 1e-9)
# D) 游荡从未接近：min 大·in_box=0
_aD = _ad([[5000., 5000.], [6000., 6000.]], [0., 0.], _GG)
check("㊱c Step-0 D 游荡从未接近：min_goal_dist≈7071 + in_box=0（与门口捕获/横向区分）",
      abs(_aD["min_goal_dist_m"] - round(_m.hypot(5000., 5000.), 3)) < 1e-3 and _aD["in_box_steps"] == 0)
# E) 角度 wrap 安全（load-bearing）：朝向 6.15rad ≡ -0.133rad ∈门 → 对齐；naive lo<=psi<=hi 会误判 6.15>0.17=不对齐
_aE = _ad([[0., 0.]], [6.15], _GG)
check("㊱d Step-0 E 朝向门 wrap 安全：ψ=6.15rad(≡−0.133) 判【对齐】(naive 区间夹比翻 FAIL)",
      _aE["in_box_steps"] == 1 and _aE["in_box_aligned_steps"] == 1
      and _aig(6.15, -0.17, 0.17) is True and _aig(1.5, -0.17, 0.17) is False)
# E2) 速度维度：稳减速入库 [9.5,5,2]→speed_at_min=2/max=9.5/摆动0（单调）；猛冲/摆动 [2,8,2,8]→摆动2
_aS = _ad([[-1000., 0.], [10., 0.], [0., 0.]], [0., 0., 0.], _GG, speeds=[9.5, 5.0, 2.0])
check("㊱d1 Step-0 速度-稳减速：speed_at_min=最近点(idx2)速度2.0 / max=9.5 / speed_reversals=0（单调减速=健康入库）",
      abs(_aS["speed_at_min_ms"] - 2.0) < 1e-9 and abs(_aS["max_speed_ms"] - 9.5) < 1e-9 and _aS["speed_reversals"] == 0)
_aSw = _ad([[-1000., 0.], [-100., 0.], [10., 0.], [0., 0.]], [0.]*4, _GG, speeds=[2.0, 8.0, 2.0, 8.0])
check("㊱d1b Step-0 速度-摆动：来回加减速 [2,8,2,8] → speed_reversals=2（速度摆动=不健康·|Δv|>0.1 才计·load-bearing）",
      _aSw["speed_reversals"] == 2)
_aSf = _ad([[-1000., 0.], [10., 0.]], [0., 0.], _GG)                     # 不传 speeds → 速度键 None
_aSn = _ad([[-1000., 0.], [10., 0.]], [0., 0.], _GG, speeds=[9.5, float("nan")])   # 速度含 NaN → 全 None(不编 nan)
check("㊱d2 Step-0 速度降级：不传/长度不符/含NaN → speed_at_min/max/reversals 全 None（未知≠0·向后兼容·NaN 不编）",
      all(_aSf[k] is None for k in ("speed_at_min_ms", "max_speed_ms", "speed_reversals"))
      and _ad([[0.,0.],[1.,0.]], [0.,0.], _GG, speeds=[1.0])["max_speed_ms"] is None            # 长度不符
      and all(_aSn[k] is None for k in ("speed_at_min_ms", "max_speed_ms", "speed_reversals")))  # NaN → 全 None
# F) 优雅降级：geom=None 全 None / 缺 vertices→in_box 键 None 但 min 仍算 / headings 长度不符→heading_err+aligned 记 None（未知≠0）
_aF1 = _ad([[0., 0.]], [0.], None)
_aF2 = _ad([[0., 0.], [500., 0.]], [0., 0.], {"center": [0.0, 0.0]})            # 缺 vertices/朝向门
_aF3 = _ad([[0., 0.], [10., 0.]], [0.], _GG)                                    # headings 少一个（长度不符）
check("㊱e Step-0 F 优雅降级：geom=None→全 None(含速度键) / 缺 vertices→in_box 键 None·min 仍算 / headings 长度不符→heading_err+aligned=None（未知≠0）",
      all(_aF1[k] is None for k in _AK)
      and _aF2["in_box_steps"] is None and _aF2["in_box_aligned_steps"] is None and abs(_aF2["min_goal_dist_m"] - 0.0) < 1e-9
      and _aF3["heading_err_at_min_deg"] is None and _aF3["in_box_aligned_steps"] is None and _aF3["in_box_steps"] == 2)
# G) _in_rect 边界（含边界·winding 无关）：中心/角点 in·框外 out
check("㊱f Step-0 _in_rect 凸多边形含边界：中心 in / 角点 in / 边外 out",
      _ir(0., 0., _GG["vertices"]) and _ir(200., 30., _GG["vertices"]) and not _ir(201., 0., _GG["vertices"]))

print("\n===== E) last-mile 终端诊断仪表化（`03` L88·term_flags/end_state/goal_geom/ego_v）=====")
# ㊳ stub：stopped 终止 → term_flags 直存终止步真旗(替排除法) + term_flags['goal']==reached + end_state 取 post-step 态
_STOP_FL = {"goal": False, "collision": False, "area": False, "stopped": True, "time": False}
_stop_post_ego = VesselState(position=np.array([100.0, 0.0]), orientation=0.3, velocity=0.0, length=175.0)  # post-step v→0
_stop_script = [{"ego": EGO, "obs": HE_FAR, "flags": _STOP_FL, "rho": RHO_NO_CONFLICT, "term": True},
                {"ego": _stop_post_ego, "obs": HE_FAR, "flags": _flags(), "rho": RHO_NO_CONFLICT}]
_rstop = run_episode(_StubEnv(_stop_script), _policy)
check("㊳ term_flags 直存终止步 5 旗：stopped 局→term_flags['stopped']=True + ['goal']==reached(False)（变异：误取 _flags() 全 False 则翻 FAIL）",
      _rstop["term_flags"] is not None and _rstop["term_flags"]["stopped"] is True
      and _rstop["term_flags"]["goal"] == _rstop["reached"] and _rstop["reached"] is False)
check("㊳a end_state=post-step 终端态(解 traj off-by-one)：v 来自终止后 _ego_vs()==0.0 + time_step==steps",
      _rstop["end_state"] is not None and abs(_rstop["end_state"]["v"] - 0.0) < 1e-12
      and _rstop["end_state"]["time_step"] == _rstop["steps"])
# ㊳b 超时 stub → term_flags['time']=True（与 stopped 区分·直证终止类型）
_TIME_FL = {"goal": False, "collision": False, "area": False, "stopped": False, "time": True}
_rtime = run_episode(_StubEnv([{"ego": EGO, "obs": HE_FAR, "flags": _TIME_FL, "rho": RHO_NO_CONFLICT, "trunc": True},
                               {"ego": EGO, "obs": HE_FAR, "flags": _flags(), "rho": RHO_NO_CONFLICT}]), _policy)
check("㊳b 超时局 → term_flags['time']=True（终止类型可分辨）", _rtime["term_flags"]["time"] is True)
# ㊳c 0 迭代边界(max_steps=0)：term_flags=None(无步可报旗)·end_state 仍非 None(post-reset 态)·不崩
_r0 = run_episode(_StubEnv(_stop_script), _policy, max_steps=0)
check("㊳c max_steps=0(0 迭代)：term_flags is None + end_state 非 None(post-reset)",
      _r0["term_flags"] is None and _r0["end_state"] is not None)
# ㊳d/e T-0 真 env：goal_geom faithful 分解（Polygon顶点 + AngleInterval 角差·复审 MEDIUM 钉死后处理法）复现官方 is_reached
if not _skip_C:
    import math as _math
    from shapely.geometry import Point as _Pt, Polygon as _Poly
    from commonroad.scenario.state import CustomState as _CS
    _envF = ShieldedUSVEnv(_sc, _ppx)
    _rF = run_episode(_envF, _keep_heading)
    _gg, _es = _rF["goal_geom"], _rF["end_state"]
    _goalF = _envF.env.term_checker.goal

    def _ang_in(psi, lo, hi):                     # AngleInterval 同款角差(禁 naive lo<=psi<=hi·±π wrap 会错)
        d = _math.atan2(_math.sin(psi - lo), _math.cos(psi - lo))
        w = _math.atan2(_math.sin(hi - lo), _math.cos(hi - lo))
        return -1e-9 <= d <= w + 1e-9

    def _faithful_reached(px, py, psi, ts):       # 用 goal_geom 复现 is_reached 三分量(位置 Polygon + 朝向角差 + 时间)
        return bool(_Poly(_gg["vertices"]).intersects(_Pt(px, py))
                    and _ang_in(psi, _gg["orient_lo"], _gg["orient_hi"])
                    and _gg["time_lo"] <= ts <= _gg["time_hi"])

    def _official(px, py, psi, ts):               # 官方 oracle
        return bool(_goalF.is_reached(_CS(position=np.array([px, py]), orientation=psi, velocity=3.0, time_step=int(ts))))
    _cx, _cy = _gg["center"]; _thc = (_gg["orient_lo"] + _gg["orient_hi"]) / 2.0
    _in_ok = _faithful_reached(_cx, _cy, _thc, 10) == _official(_cx, _cy, _thc, 10) == True       # 门内态：两法皆 True
    _out_ok = _faithful_reached(_cx, _cy + 1e5, _thc, 10) == _official(_cx, _cy + 1e5, _thc, 10) == False  # 门外态：两法皆 False
    check("㊳d goal_geom faithful 分解(Polygon+AngleInterval) == 官方 is_reached：门内态 True / 门外态 False（钉死后处理法·杀 naive 区间夹比）",
          _in_ok and _out_ok)
    check("㊳e T-0 终端：faithful(end_state) == term_flags['goal'](=False·超时未达) + term_flags['time']=True + goal_geom 含闭合顶点环",
          _faithful_reached(_es["px"], _es["py"], _es["psi"], _es["time_step"]) == _rF["term_flags"]["goal"] is False
          and _rF["term_flags"]["time"] is True
          and isinstance(_gg.get("vertices"), list) and _gg["vertices"][0] == _gg["vertices"][-1])

print("\n" + ("✅ 全部 PASS" if _fail == 0 else f"❌ {_fail} 项 FAIL"))
sys.exit(1 if _fail else 0)
