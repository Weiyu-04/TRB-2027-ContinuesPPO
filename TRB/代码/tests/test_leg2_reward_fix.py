#!/usr/bin/env python3
"""第二条腿修法 committed 测试（`03` L172·连续臂专属·治崩塌"corr≈0 脱钩 + 过路惩罚泄漏入库精修段"）。

修法 = 两旋钮（默认全关 = 与现状逐位等价 bit-identical）：
  · c_reach   ：重标 r_goal 系数（Rung1·默认 C_REACH=1.5·降到 0.2 压近常数回报量级→抬 +50 归一化占比→治 corr≈0）
  · dock_radius+v_dock：泊位精修门（Rung2·默认关·泊位区内把速度地板 V_LOW→v_dock·降非清零→治入库减速被罚泄漏·防停门口）

验（对齐 test_dwell_cost.py 范式）：
  T1 c_reach 默认(=C_REACH) vs 显式 1.5 → r_goal 逐位等价(<1e-12)·parts 不加键
  T2 dock 默认(dock_radius=0 / v_dock=V_LOW) → r_velocity 逐位等价（v×距离 扫描）
  T3 端到端：ContinuousProjectionEnv 默认 vs 显式默认 → 逐步 reward 逐位等价（有场景时·否则 inspect 兜底）
  T4 c_reach=0.2 机制：r_goal == 0.2·(d_prev−d_now) 精确 == 默认×(0.2/1.5)
  T5 泊位门机制：区内 v=v_dock→r_velocity=0 / v<v_dock→按 v_dock 罚 / 区外→按 V_LOW 罚（不变）
  T6 防停门口：区内 v<v_dock → 仍被罚（地板降非清零·<0）
  T7 忠实红线：泊位门只在 reward 层·不污染 termination（stopped→C_STOPPED 不受门影响）·gate 不加 parts 键
  T8 fail-fast：v_dock≤0.48 / v_dock>V_LOW / c_reach<0 / dock_radius<0 → ValueError
  T9 离散忠实：RewardFunction 默认 c_reach==C_REACH / dock_radius==0（离散 maker 不传=用默认=忠实 Krasowski）；run_step4e 离散 env_kwargs 不含 c_reach
  T10 config_conflict：缺 c_reach/dock_radius/v_dock 旧记录+当前默认→无冲突；旧记录+当前 c_reach=0.2→硬冲突（防脏钱图）
  P  plumbing：USVEnv/ContinuousProjectionEnv 有 3 形参且透传；run_step4e 有 _C_REACH/_DOCK_R/_V_DOCK 旋钮
运行：/opt/miniconda3/envs/trb/bin/python -B 代码/tests/test_leg2_reward_fix.py
"""
import os, sys, glob, inspect
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from trb_env.usv_reward import RewardFunction, C_REACH, V_LOW, V_HIGH, C_V, C_STOPPED

GOAL = [5000.0, 300.0]
INIT = [1000.0, 600.0]
ORI = (-0.17, 0.17)
N_FAIL = 0


def chk(cond, msg):
    global N_FAIL
    print(("  ✅ " if cond else "  ❌ ") + msg)
    if not cond:
        N_FAIL += 1


def _mk(**kw):
    return RewardFunction(GOAL, INIT, goal_orientation=ORI, gamma=0.99, **kw)


def _step_at(rf, px, py, psi, v, prev=None):
    """在给定态 step 一次，返回 (total, parts)。prev=上一步位置（None→用 INIT reset）。"""
    rf.reset(np.array([INIT[0], INIT[1], 0.0, 5.0]))
    if prev is not None:
        rf._prev_pos = np.asarray(prev, dtype=float)
    return rf.step(np.array([px, py, psi, v]), [], {}, False)


print("===== T1 c_reach 默认 == 显式 1.5 逐位等价（bit-identical）=====")
rf_def = _mk(); rf_15 = _mk(c_reach=1.5)
_, p0 = _step_at(rf_def, 1100.0, 600.0, 0.0, 5.0, prev=INIT)
_, p1 = _step_at(rf_15, 1100.0, 600.0, 0.0, 5.0, prev=INIT)
chk(abs(p0["goal"] - p1["goal"]) < 1e-12, f"默认 c_reach r_goal == 显式1.5（差 {abs(p0['goal']-p1['goal']):.2e}<1e-12·变异:改默认硬编≠C_REACH 则翻）")
chk(set(p0.keys()) == {"sparse", "colregs", "goal", "velocity", "deviate"}, f"默认 parts 键无新增（{sorted(p0.keys())}）")
chk(rf_def.c_reach == C_REACH, f"RewardFunction 默认 self.c_reach == C_REACH({C_REACH})=模块常量非硬编（防漂移）")

print("===== T2 dock 默认(dock_radius=0/v_dock=V_LOW) → r_velocity 逐位等价 =====")
rf_dockdef = _mk(dock_radius=0.0, v_dock=V_LOW)
bad = 0
for d, (gx) in [(50.0, GOAL[0]-50), (3000.0, INIT[0])]:      # 近门 & 远门
    for v in [0.0, 1.0, 2.4, 2.5, 5.0, 8.0, 9.0]:
        px = gx; py = GOAL[1] if d < 100 else 600.0
        _, pa = _step_at(rf_def, px, py, 0.0, v, prev=[px-1, py])
        _, pb = _step_at(rf_dockdef, px, py, 0.0, v, prev=[px-1, py])
        if abs(pa["velocity"] - pb["velocity"]) >= 1e-12:
            bad += 1
chk(bad == 0, f"dock 默认 r_velocity 与 baseline 逐位等价（{bad} 处不符·须 0）")

print("===== T4 c_reach=0.2 机制（重标精确）=====")
rf02 = _mk(c_reach=0.2)
_, p02 = _step_at(rf02, 1100.0, 600.0, 0.0, 5.0, prev=INIT)
d_prev = float(np.hypot(INIT[0]-GOAL[0], INIT[1]-GOAL[1]))
d_now = float(np.hypot(1100.0-GOAL[0], 600.0-GOAL[1]))
chk(abs(p02["goal"] - 0.2*(d_prev-d_now)) < 1e-9, f"r_goal == 0.2·(d_prev−d_now)（得 {p02['goal']:.4f} 期 {0.2*(d_prev-d_now):.4f}）")
chk(abs(p02["goal"] - p0["goal"]*0.2/1.5) < 1e-9, "r_goal == 默认×(0.2/1.5)（均匀缩放·变异:漏乘 self.c_reach 则翻）")

print("===== T5 泊位门机制（区内降地板/区外不变）=====")
# goal=(5000,300)·放 ego 在 d≈80 (4920,300)=区内；d≈1000=区外
rf_g = _mk(dock_radius=350.0, v_dock=1.0)
_, g_in_vdock = _step_at(rf_g, 4920.0, 300.0, 0.0, 1.0, prev=[4919, 300])   # v=v_dock=1.0→免罚
_, g_in_low = _step_at(rf_g, 4920.0, 300.0, 0.0, 0.5, prev=[4919, 300])     # v=0.5<v_dock→按 v_dock 罚
_, g_out = _step_at(rf_g, 4000.0, 300.0, 0.0, 1.0, prev=[3999, 300])        # d=1000>350 区外→按 V_LOW 罚
chk(abs(g_in_vdock["velocity"] - 0.0) < 1e-9, f"区内 v=v_dock(1.0)→r_velocity=0 免罚（得 {g_in_vdock['velocity']:.4f}）")
chk(abs(g_in_low["velocity"] - C_V*(1.0-0.5)) < 1e-9, f"区内 v=0.5<v_dock→r_velocity=C_V·(1.0−0.5)={C_V*0.5}（得 {g_in_low['velocity']:.4f}）")
chk(abs(g_out["velocity"] - C_V*(V_LOW-1.0)) < 1e-9, f"区外 v=1.0→r_velocity=C_V·(V_LOW−1.0)={C_V*(V_LOW-1.0)}（不变·得 {g_out['velocity']:.4f}）")

print("===== T6 防停门口（区内 v<v_dock 仍被罚·地板降非清零）=====")
_, g_stop = _step_at(rf_g, 4920.0, 300.0, 0.0, 0.3, prev=[4919, 300])
chk(g_stop["velocity"] < 0.0 and abs(g_stop["velocity"] - C_V*(1.0-0.3)) < 1e-9,
    f"区内 v=0.3(<v_dock)→r_velocity=C_V·(1.0−0.3)={C_V*0.7:.2f}<0 仍罚（得 {g_stop['velocity']:.4f}·变异:清零地板则=0 翻）")

print("===== T7 忠实红线（门只在 reward·不污染 termination·gate 不加键）=====")
# stopped flag → C_STOPPED 仍施加（门不碰 r_sparse 的 stopped 项）
_, g_termkeys = _step_at(rf_g, 4920.0, 300.0, 0.0, 1.0, prev=[4919, 300])
chk(set(g_termkeys.keys()) == {"sparse", "colregs", "goal", "velocity", "deviate"}, f"泊位门开·parts 键无新增（{sorted(g_termkeys.keys())}）")
rf_g.reset(np.array([INIT[0], INIT[1], 0.0, 5.0])); rf_g._prev_pos = np.array([4919.0, 300.0])
_, g_sp = rf_g.step(np.array([4920.0, 300.0, 0.0, 0.0]), [], {"stopped": True}, False)
chk(abs(g_sp["sparse"] - C_STOPPED) < 1e-9 or C_STOPPED in (g_sp["sparse"],) or abs((g_sp["sparse"]) - C_STOPPED) < 1e-6,
    f"stopped→C_STOPPED({C_STOPPED}) 仍进 r_sparse（门不碰终端罚·得 sparse={g_sp['sparse']:.2f}）")

print("===== T8 fail-fast =====")
for bad_kw, label in [(dict(dock_radius=350.0, v_dock=0.48), "v_dock=0.48(≤下界)"),
                      (dict(dock_radius=350.0, v_dock=0.4), "v_dock=0.4(<下界)"),
                      (dict(dock_radius=350.0, v_dock=2.6), "v_dock>V_LOW"),
                      (dict(c_reach=-0.1), "c_reach<0"),
                      (dict(dock_radius=-1.0), "dock_radius<0")]:
    try:
        _mk(**bad_kw); chk(False, f"{label} 未 raise")
    except ValueError:
        chk(True, f"{label} → ValueError")
# 边界：dock_radius>0 且 v_dock=V_LOW（上界含）→ 合法（no-op）
try:
    _mk(dock_radius=350.0, v_dock=V_LOW); chk(True, f"v_dock=V_LOW({V_LOW}) 合法(上界含·=no-op)")
except ValueError:
    chk(False, "v_dock=V_LOW 误报错")

print("===== T9 离散忠实（默认=Krasowski typo-fix·c_reach 不泄漏离散）=====")
chk(_mk().c_reach == C_REACH and _mk().dock_radius == 0.0 and _mk().v_dock == V_LOW,
    "RewardFunction 默认 c_reach=C_REACH/dock_radius=0/v_dock=V_LOW（离散不传=用默认=忠实）")
_rs = open(os.path.join(os.path.dirname(__file__), "..", "run_step4e.py")).read()
# c_reach=_C_REACH 只该出现在【连续】处：2 个连续 maker(PPO/SAC) + 1 个 config_conflict 守卫调用
#   + 🆕 1 个【热启动源配置校验 probe】(`03` L190 D3·train_eval_one_continuous 内·连续PPO臂专属·读源 sidecar config_sig 比对本 run 语义配方) = 恰 4 处（离散 maker 不传=忠实 Krasowski）。
# 变异：若误把 c_reach=_C_REACH 加进离散 MaskablePPO env 构造→计数>4→翻 FAIL（拦"泄漏进离散破忠实"）。
#   ⚠️ 本条是【粗计数】防线；**精确防线 = 下面的 _disc_seg 段扫描**（专查离散函数段内不含 c_reach·加新连续用法不误触）。
chk(_rs.count("c_reach=_C_REACH") == 4,
    f"run_step4e 'c_reach=_C_REACH' 恰 4 处(2 连续 maker+1 守卫+1 热启动源配置校验 probe·离散不传·得 {_rs.count('c_reach=_C_REACH')})")
# 离散训练路径 make_vec_env(train_paths, env_cls=...) 不注入 c_reach（离散 env_cls=ShieldedUSVEnv 走 USVEnv 默认 c_reach=C_REACH=1.5）
_disc_seg = _rs[_rs.find("def train_eval_one("):_rs.find("def train_eval_one_continuous(")]
chk("c_reach" not in _disc_seg,
    "离散 train_eval_one 段不含 c_reach（离散走 USVEnv 默认=忠实 Krasowski typo-fix）")

print("===== T10 config_conflict 归一化 =====")
import run_step4e as S
_base = {"steps": 5_000_000, "n_total": 200, "pool_size": 40, "n_seg": 10}
chk(S.config_conflict([dict(_base)], 5_000_000, 200, pool_size=40, n_seg=10) == set(),
    "缺 c_reach/dock_radius/v_dock 旧记录 + 当前默认 → 无冲突（归一化 1.5/0.0/2.5·续写兼容）")
chk(bool(S.config_conflict([dict(_base)], 5_000_000, 200, pool_size=40, n_seg=10, c_reach=0.2)),
    "旧记录 + 当前 c_reach=0.2 → 硬冲突（防脏钱图混写）")
chk(bool(S.config_conflict([dict(_base)], 5_000_000, 200, pool_size=40, n_seg=10, dock_radius=350.0, v_dock=1.0)),
    "旧记录 + 当前开泊位门 → 硬冲突（防脏钱图混写）")

print("===== P plumbing（形参 + 透传 + 旋钮）=====")
from trb_env.usv_env import USVEnv
from trb_env.usv_continuous_shield import ContinuousProjectionEnv
_ue_sig = inspect.signature(USVEnv.__init__).parameters
chk(all(k in _ue_sig for k in ("c_reach", "dock_radius", "v_dock")), "USVEnv.__init__ 有 c_reach/dock_radius/v_dock 形参")
chk("c_reach=c_reach" in inspect.getsource(USVEnv.__init__), "USVEnv 把 c_reach 透传 RewardFunction")
_cp_sig = inspect.signature(ContinuousProjectionEnv.__init__).parameters
chk(all(k in _cp_sig for k in ("c_reach", "dock_radius", "v_dock")), "ContinuousProjectionEnv.__init__ 有 3 形参")
chk("c_reach=c_reach" in inspect.getsource(ContinuousProjectionEnv.__init__), "ContinuousProjectionEnv 把 c_reach 透传内层 USVEnv")
chk(all(k in _rs for k in ("STEP4E_C_REACH", "STEP4E_DOCK_R", "STEP4E_V_DOCK")), "run_step4e 有 3 个 env 旋钮")
chk("c_reach=_C_REACH" in _rs, "run_step4e 连续 maker 透传 _C_REACH")

print("===== M maker 层端到端（防断链·复审 wvbzr5av3 抓的 HIGH：run_step4e 传 c_reach 给 maker 但 maker 无形参→**kwargs→SAC/PPO 崩）=====")
# 【机器无关·硬守】两个连续 maker 必须有 c_reach/dock_radius/v_dock 形参 + env_kwargs 透传（否则 run_step4e 传参→**sac/ppo_kwargs→SAC/PPO.__init__ TypeError·连默认都崩）
from trb_env.usv_sac_train import make_continuous_safe_model, make_continuous_safe_ppo_model
for _mk_fn, _nm in [(make_continuous_safe_model, "SAC"), (make_continuous_safe_ppo_model, "PPO")]:
    _sig = inspect.signature(_mk_fn).parameters
    chk(all(k in _sig for k in ("c_reach", "dock_radius", "v_dock")),
        f"make_continuous_safe_{'model' if _nm=='SAC' else 'ppo_model'}({_nm}) 有 c_reach/dock_radius/v_dock 形参（缺则 run_step4e 传参落 **kwargs→SAC/PPO 崩）")
    _src = inspect.getsource(_mk_fn)
    chk("c_reach=c_reach" in _src and "dock_radius=dock_radius" in _src and "v_dock=v_dock" in _src,
        f"{_nm} maker env_kwargs 透传 c_reach/dock_radius/v_dock（否则 knob 到不了 RewardFunction=静默空转）")
# 【本地有场景则真端到端·无则显式标注·绝不静默吞】：真构造 maker→挖内层 RewardFunction 验 knob 落地
_scn = (sorted(glob.glob("/private/tmp/trb_scenarios_pool/T-*.xml"))
        or sorted(glob.glob("/private/tmp/trb_speedcheck/T-*.xml"))
        or sorted(glob.glob("/private/tmp/trb_scenarios/T-*.xml")))[:2]
if _scn:
    from trb_env.usv_scenarios import load_scenario_pool
    _pool = load_scenario_pool(_scn)

    def _probe_rf(venv):
        inner = venv.envs[0] if hasattr(venv, "envs") else venv
        while not hasattr(inner, "reward_fn") and hasattr(inner, "env"):
            inner = inner.env
        return inner.reward_fn

    _code_err = None
    try:
        _m, _v = make_continuous_safe_model(scenario_pool=_pool, seed=0)   # 默认(c_reach=1.5)
        _rf = _probe_rf(_v)
        chk(_rf.c_reach == C_REACH and _rf.dock_radius == 0.0 and _rf.v_dock == V_LOW,
            f"SAC maker 默认端到端：RewardFunction c_reach={_rf.c_reach}/dock={_rf.dock_radius}/v_dock={_rf.v_dock}(应 1.5/0/2.5)")
        _v.close()
        _m2, _v2 = make_continuous_safe_model(scenario_pool=_pool, seed=0, c_reach=0.2, dock_radius=350.0, v_dock=1.0)
        _rf2 = _probe_rf(_v2)
        chk(_rf2.c_reach == 0.2 and _rf2.dock_radius == 350.0 and _rf2.v_dock == 1.0,
            f"SAC maker c_reach=0.2+门端到端：knob 落地 RewardFunction(c_reach={_rf2.c_reach}/dock={_rf2.dock_radius}/v_dock={_rf2.v_dock})")
        _v2.close()
        _m3, _v3 = make_continuous_safe_ppo_model(scenario_pool=_pool, seed=0, n_envs=1, subproc=False, c_reach=0.2)
        _rf3 = _probe_rf(_v3)
        chk(_rf3.c_reach == 0.2, f"PPO maker c_reach=0.2 端到端：knob 落地 RewardFunction(c_reach={_rf3.c_reach})")
        _v3.close()
    except (TypeError, AttributeError) as _e:   # 🔴 代码错(断链/形参缺)必须 FAIL·不吞（复审教训：Node L 用 except Exception 把此类 TypeError 吞成 SKIP=假绿）
        chk(False, f"maker 端到端崩于代码错(断链?)：{type(_e).__name__}: {_e}")
        _code_err = _e
else:
    print("  ⚠️[显式标注·非静默] 本机无 /private/tmp 场景→跳过 maker 真构造(inspect 形参/透传守卫仍已硬守上方)·服务器/有场景时补跑")

print("===== N rank1 泊位门控治抖 r_rate（`03` L173·你的『停车别罚急打舵』想法·默认 off bit-identical）=====")
# config_conflict（无需场景）：rate_dock 变化→硬冲突（防脏钱图）
chk(bool(S.config_conflict([dict(_base)], 5_000_000, 200, pool_size=40, n_seg=10, rate_dock=0.0)),
    "config_conflict：旧记录(无 rate_dock) + 当前 rate_dock=0.0 → 硬冲突（防脏钱图混写）")
chk(S.config_conflict([dict(_base, rate_dock=None)], 5_000_000, 200, pool_size=40, n_seg=10) == set(),
    "config_conflict：rate_dock=None 旧记录 + 当前默认 None → 无冲突（续写兼容）")
# 【机器无关】plumbing：盾/maker 有 rate_dock 形参 + 透传
from trb_env.usv_continuous_shield import ContinuousProjectionEnv as _CPE
chk("rate_dock" in inspect.signature(_CPE.__init__).parameters, "ContinuousProjectionEnv.__init__ 有 rate_dock 形参")
for _mk_fn, _nm in [(make_continuous_safe_model, "SAC"), (make_continuous_safe_ppo_model, "PPO")]:
    _s = inspect.signature(_mk_fn).parameters
    _src2 = inspect.getsource(_mk_fn)
    chk("rate_dock" in _s and "rate_dock=rate_dock" in _src2,
        f"{_nm} maker 有 rate_dock 形参且 env_kwargs 透传（缺则 run_step4e 传参落 **kwargs→崩/知识到不了盾）")
chk(_rs.count("rate_dock=_RATE_DOCK") >= 3, f"run_step4e 'rate_dock=_RATE_DOCK' ≥3 处(2 maker+守卫·得 {_rs.count('rate_dock=_RATE_DOCK')})")
# 【本地有场景则真验机制】默认 bit-identical + 区内免罚 + 区外不变 + fail-fast + maker 传参
if _scn:
    import numpy as _np
    _sc0, _pp0 = load_scenario_pool(_scn[:1])[0]

    def _rr_seq(rate_weight, rate_dock, dock_radius, n=6, seed=0):
        _e = _CPE(_sc0, _pp0, rate_weight=rate_weight, rate_dock=rate_dock, dock_radius=dock_radius)
        _e.reset(seed=seed); _rng = _np.random.RandomState(seed)
        _out = []
        for _ in range(n):
            _a = _rng.uniform(-0.04, 0.04, 2)
            _, _, _t, _tr, _i = _e.step(_a); _out.append(_i.get("r_rate"))
            if _t or _tr: break
        return _out
    _b = _rr_seq(1.0, None, 0.0)
    _ng = _rr_seq(1.0, None, 350.0)   # dock 设了但 rate_dock=None → 仍 off
    chk(max(abs((x or 0) - (y or 0)) for x, y in zip(_b, _ng)) < 1e-12,
        "N1 默认 rate_dock=None → r_rate 逐位等价 bit-identical（变异：门控漏 None 守卫则翻）")
    _all = _rr_seq(1.0, 0.0, 10000.0)   # 泊位区覆盖全程 + rate_dock=0
    chk(all((x is None or abs(x) < 1e-12) for x in _all),
        "N2 区覆盖全程+rate_dock=0 → r_rate 全免（=0·区内治抖罚真放开·变异：不用 _rw 则仍满罚 翻）")
    # N2b 正向缩放（`03` L176 对抗复审补·抓 N2/N1 都抓不到的变异）：区内 rate_dock=0.5 → r_rate 恰=0.5×满罚。
    #   rate_weight 只入奖励不改施加动作/状态→同 seed 下 Δu_desired 与 _b(满罚基线)逐步相同→r_rate 精确按 _rw 缩放。
    #   抓 `_rw=self._rate_dock` 写死 0.0（N2 用 rate_dock=0 抓不到）+ 门控空转 `_rw=self.rate_weight`（得满罚≠半罚 翻）。
    _half = _rr_seq(1.0, 0.5, 10000.0)   # 泊位区覆盖全程 + rate_dock=0.5
    _pairs = [(h, b) for h, b in zip(_half, _b) if h is not None and b is not None and abs(b) > 1e-12]
    chk(len(_pairs) > 0 and all(abs(h - 0.5 * b) < 1e-12 for h, b in _pairs),
        f"N2b 区内 rate_dock=0.5 → r_rate 恰=0.5×满罚（{len(_pairs)}步核·变异 _rw 写死0 或门控空转 皆翻）")
    _far = _rr_seq(1.0, 0.0, 350.0)   # 区小·起点远(区外)
    chk(max(abs((x or 0) - (y or 0)) for x, y in zip(_b, _far)) < 1e-12,
        "N3 区小(350)起点远(区外) → r_rate=满罚不变（区外不误门控）")
    # fail-fast
    for _bad, _lab in [(dict(rate_weight=1.0, rate_dock=0.0, dock_radius=0.0), "rate_dock 设但 dock_radius=0(无区)"),
                       (dict(rate_weight=1.0, rate_dock=-0.1, dock_radius=350.0), "rate_dock<0")]:
        try:
            _CPE(_sc0, _pp0, **_bad); chk(False, f"N4 {_lab} 未 raise")
        except ValueError:
            chk(True, f"N4 {_lab} → ValueError")
    # maker 传参(防断链·env_kwargs 带 rate_dock)
    _m, _v = make_continuous_safe_model(scenario_pool=[(_sc0, _pp0)], seed=0, rate_weight=1.0, rate_dock=0.0, dock_radius=350.0)
    chk(_v.get_attr("env_kwargs")[0].get("rate_dock") == 0.0, "N5 SAC maker env_kwargs 带 rate_dock=0.0（防断链·端到端到盾）")
    _v.close()
else:
    print("  ⚠️[显式标注·非静默] 本机无场景→跳过 N1-N5 机制/maker 真验(plumbing 形参守卫已硬守上方)")

print("\n" + ("=" * 50))
print("✅ 全部通过" if N_FAIL == 0 else f"❌ {N_FAIL} 项失败")
sys.exit(1 if N_FAIL else 0)
