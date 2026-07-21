"""
Phase 3 Node A 连续投影盾环境冒烟测试（2026-06-17(b)）——集成层接线正确性 + D38 五条必做项 + 变异守护。
跑：/opt/miniconda3/envs/trb/bin/python -B 代码/tests/test_usv_continuous_shield.py
（自动计数、非 pytest；需 /tmp/trb_T0.xml 夹具，缺则联网下载、离线则该块 SKIP。）

覆盖：① env.reset→proj.reset（防 EC 跨 episode 静默错动作，变异坐实）② proj box==动力学 box（mismatch→raise）
③ 每决策步只调一次 safe_action（状态机步数==env 步数，变异坐实）④ u_applied 经 env 最终 box 限幅（含越界 desired
+ emergency）⑤ info 带 source/rho（T-0 覆盖 projection+emergency 多档）+ 给路投影合规右转 + 确定性 + _ego/_obs_vs。
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from trb_env.usv_colregs import RHO_NO_CONFLICT, RHO_STAND_ON, RHO_EMERGENCY, VesselState
from trb_env.usv_continuous_shield import ContinuousProjectionEnv
from trb_env.usv_projection import ContinuousColregsProjection

_fail = 0
_total = 0


def ok(name, cond):
    global _fail, _total
    _total += 1
    if not cond:
        _fail += 1
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")


_T0 = "/tmp/trb_T0.xml"
_HAVE_T0 = os.path.exists(_T0)
if not _HAVE_T0:
    try:
        import urllib.request
        url = ("https://gitlab.lrz.de/tum-cps/commonocean-scenarios/-/raw/main/scenarios/"
               "HandcraftedTwoVesselEncounters_01_24/ZAM_AAA-1_20240121_T-0.xml")
        urllib.request.urlretrieve(url, _T0)
        _HAVE_T0 = True
    except Exception as e:
        print(f"[SKIP] /tmp/trb_T0.xml 不在且离线下载失败（{e}）→ 跳过端到端块（非代码回归）")

A_MAX, W_MAX = 0.24, 0.03

if _HAVE_T0:
    from commonocean.common.file_reader import CommonOceanFileReader
    _sc, _pp = CommonOceanFileReader(_T0).open()
    _ppx = list(_pp.planning_problem_dict.values())[0]

    def fresh(**kw):
        return ContinuousProjectionEnv(_sc, _ppx, **kw)

    def run_ep(env, u=(0.05, 0.02), max_steps=220, collect=False):
        recs = []
        for _ in range(max_steps):
            _o, _r, te, tr, info = env.step(np.array(u, dtype=float))
            if collect:
                recs.append({"source": info["source"], "rho": info["rho"],
                             "u_applied": np.asarray(info["u_applied"], float),
                             "emergency_mode": info["emergency_mode"]})
            if te or tr:
                break
        return recs

    print("===== A) 集成基础（spaces / reset info / box 对齐）=====")
    env = fresh()
    ok("① action_space=连续 Box([a,ω]) (2,)", env.action_space.shape == (2,) and env.observation_space.shape == (27,))
    ok("① RL 动作箱=Krasowski 正常操作 ±0.048/±0.018（L63 Fix②·非满程 ±a_max=停船墙根；改回满程则 FAIL）",
       np.allclose(env.action_space.high, [0.048, 0.018]) and np.allclose(env.action_space.low, [-0.048, -0.018]))
    ok("① 内层 USVEnv 仍物理满程 ±a_max/±w_max（盾/EC/动力学不变·正常操作⊂物理→_map_action 对 RL 动作 no-op）",
       np.allclose(env.env.action_space.high, [A_MAX, W_MAX]) and np.allclose(env.env.action_space.low, [-A_MAX, -W_MAX]))
    ok("① proj box == 动力学 box（②集成必做：构造即对齐·满程物理 box·B-EMERGENCY-BOX 契约不破）",
       env.proj.a_max == env.env.p.a_max and env.proj.w_max == env.env.p.w_max)
    ok("① colregs_weight 默认=0.0（footgun 修复 L44：=Continuous-safe 唯一用途、误差走安全侧；改回 1.0 则 FAIL）",
       env.env.reward_fn.colregs_weight == 0.0)
    obs, info = env.reset(seed=0)
    ok("① reset → obs∈Box + info 带 rho/source", env.observation_space.contains(obs)
       and info["rho"] == RHO_NO_CONFLICT and info["source"] is None)
    ok("① 他船 length = 真实 obstacle_shape.length（≈236.4，非硬编175）", abs(env._obs_length - 236.4) < 1.0)

    print("\n===== F) ⑤ source 归类 + T-0 端到端覆盖多档 =====")
    env = fresh(); env.reset(seed=0)
    recs = run_ep(env, collect=True)
    srcs = {r["source"] for r in recs}
    ok("⑤ T-0 episode 覆盖 projection 档", "projection" in srcs)
    ok("⑤ T-0 episode 覆盖 emergency 档（head-on→ρ5）", "emergency" in srcs)
    ok("⑤ 每步 info 带合法 source（∈ 已知 6 档）", all(
        r["source"] in {"projection", "emergency", "relaxed", "collision_min", "degenerate", "no_obstacle"} for r in recs))
    ok("⑤ emergency 步 rho==RHO_EMERGENCY 且 emergency_mode∈{ahead,stern,base}", all(
        (r["rho"] == RHO_EMERGENCY and r["emergency_mode"] in {"ahead", "stern", "base"})
        for r in recs if r["source"] == "emergency"))

    print("\n===== F2) 紧急惩罚 C_EMERGENCY 口径对齐（L43-续①修：Continuous-safe emergency 步 == Discrete-safe）=====")
    # Node A 把 source=='emergency' 经 emergency_used 传给 reward → 紧急步 sparse 含 −0.5（==离散 idx49 口径）。
    # load-bearing：若 Node A 回退为不传 emergency_used，则 emergency 步 sparse=0.0 → 本断言 FAIL。
    env = fresh(); env.reset(seed=0); _em_sparse = []
    for _ in range(220):
        _o, _r, te, tr, info = env.step(np.array([0.05, 0.02]))
        if info["source"] == "emergency":
            _em_sparse.append(round(info["reward_parts"]["sparse"], 3))
        if te or tr:
            break
    ok("F2 emergency 步 reward sparse 含 C_EMERGENCY(−0.5)（四方紧急惩罚口径==Discrete-safe，L43-续①修）",
       len(_em_sparse) >= 1 and any(abs(s - (-0.5)) < 1e-9 or abs(s - (-40.5)) < 1e-9 for s in _em_sparse))

    print("\n===== E) ④ u_applied 经 env 最终 box 限幅（端到端 ∈box）=====")
    in_box = all((-A_MAX - 1e-9 <= r["u_applied"][0] <= A_MAX + 1e-9
                  and -W_MAX - 1e-9 <= r["u_applied"][1] <= W_MAX + 1e-9) for r in recs)
    ok("④ T-0 全程 u_applied ∈ box", in_box)
    # 越界 desired → 应用值被 env 截进 box
    env = fresh(); env.reset(seed=0)
    _o, _r, _te, _tr, info = env.step(np.array([99.0, 99.0]))   # 远超 box 的期望动作
    ua = np.asarray(info["u_applied"], float)
    ok("④ 越界 desired(99,99) → u_applied 被截进 box", -A_MAX - 1e-9 <= ua[0] <= A_MAX + 1e-9 and -W_MAX - 1e-9 <= ua[1] <= W_MAX + 1e-9)

    print("\n===== B) ① env.reset() → proj.reset()（防 EC 跨 episode 静默错动作，变异坐实）=====")
    # 跑到进入 emergency（EC 懒创建、_prev_rho=ρ5），再 reset，proj 应被清回 NO_CONFLICT
    env = fresh(); env.reset(seed=0); run_ep(env)
    entered_em = env.proj._ec is not None
    env.reset(seed=0)
    ok("① 正常 env.reset() 后 proj._prev_rho 清回 NO_CONFLICT（EC 也 reset）",
       entered_em and env.proj._prev_rho == RHO_NO_CONFLICT and env.proj._ec.mode is None)

    # ⚠️ 本变异守护要求 episode 终态 _prev_rho≠ρ0（默认动作(0.05,0.02)下终态=ρ5、成立）；下方断言左支
    #    `stale != RHO_NO_CONFLICT` 已防夹具退化致假阳性（若终态恰 ρ0 则此 ok 直接 FAIL 报警、非静默通过）。
    class _NoResetEnv(ContinuousProjectionEnv):
        """变异：reset 故意【不】调 proj.reset() → ρ 边沿追踪 + EC mode 跨 episode 泄漏。"""
        def reset(self, *, seed=None, options=None):
            super(ContinuousProjectionEnv, self).reset(seed=seed)
            obs, info = self.env.reset(seed=seed)
            self._rho = RHO_NO_CONFLICT; self._source = None        # 不调 self.proj.reset()
            return obs, {**info, "rho": self._rho, "source": self._source}

    menv = _NoResetEnv(_sc, _ppx); menv.reset(seed=0); run_ep(menv)
    stale_prev = menv.proj._prev_rho
    menv.reset(seed=0)
    ok("①(变异守护) 跳过 proj.reset() → 跨 episode proj._prev_rho 泄漏(≠NO_CONFLICT)、坐实 reset 真守护",
       stale_prev != RHO_NO_CONFLICT and menv.proj._prev_rho == stale_prev)

    print("\n===== D) ③ 每决策步只调一次 safe_action（状态机步数==有他船的 env 步数，变异坐实）=====")
    env = fresh(); env.reset(seed=0)
    _calls = {"n": 0}
    _orig_step = env.proj._sc.step
    def _counting_step(se, so):
        _calls["n"] += 1
        return _orig_step(se, so)
    env.proj._sc.step = _counting_step
    n_obs_steps = 0
    for _ in range(220):
        had_obs = env._obs_vs() is not None
        _o, _r, te, tr, _info = env.step(np.array([0.05, 0.02]))
        if had_obs:
            n_obs_steps += 1
        if te or tr:
            break
    ok("③ 状态机 step 调用数 == 有他船的 env 步数（每步恰一次、未混调双推）", _calls["n"] == n_obs_steps and n_obs_steps > 0)

    print("\n===== C) ② proj box != 动力学 box → safe_action 内部 assert raise（B-EMERGENCY-BOX 守护）=====")
    env = fresh(); env.reset(seed=0)
    env.proj = ContinuousColregsProjection(A_MAX * 0.5, W_MAX)    # 故意制造 box 不匹配
    raised = False
    try:
        for _ in range(220):
            _o, _r, te, tr, _i = env.step(np.array([0.05, 0.02]))
            if te or tr:
                break
    except ValueError:
        raised = True
    ok("② box 不匹配(0.5×a_max) → 进入投影时 raise ValueError（盾 box 必须==动力学 box）", raised)

    print("\n===== G) 集成核心：env 执行【投影后】u_safe、非 agent 原始 desired =====")
    # Node A 的职责 = 把 safe_action 输出真接进 env.step（非透传 agent 原值）。给路(ρ2/3)合规右转(ω≤−ω_turn)
    # 由 projection ④(ρ2/3 ω→−ω_turn) + min_loop(干净 crossing give-way) 覆盖；T-0 head-on 紧急主导、无 ρ2/3
    # give-way projection 步（D35/L38 框架性质，上面 rho 分布坐实=ρ0 透传+ρ1 stand-on+ρ5 emergency）→ 本块测集成属性。
    env = fresh(); env.reset(seed=0); recs = run_ep(env, collect=True)
    _desired = np.array([0.05, 0.02])
    _corr = [r for r in recs if np.linalg.norm(r["u_applied"] - _desired) > 1e-6]
    ok("G 盾真介入：存在 u_applied≠原始 desired 的步（投影/兜底动作真接进 env.step、非透传 agent 原值）", len(_corr) >= 1)
    _em = [r for r in recs if r["source"] == "emergency"]
    ok("G emergency 步 u_applied=EC 输出、与原始 desired 明显不同（safe_action 输出真被执行）",
       len(_em) >= 1 and any(np.linalg.norm(r["u_applied"] - _desired) > 1e-3 for r in _em))
    _so = [r for r in recs if r["source"] == "projection" and r["rho"] == RHO_STAND_ON]
    ok("G ρ1 stand-on 投影步 u_applied 夹进保向窄带(|ω|≤ε_ω)（保速保向约束经 env 落地）",
       len(_so) == 0 or all(abs(r["u_applied"][1]) <= env.proj.eps_omega + 1e-9 for r in _so))

    print("\n===== H) 确定性 + I) eval-compat =====")
    def traj(seed):
        e = fresh(); e.reset(seed=seed); out = []
        for _ in range(30):
            _o, _r, te, tr, inf = e.step(np.array([0.05, 0.02]))
            out.append((inf["source"], inf["rho"], tuple(np.round(inf["u_applied"], 9))))
            if te or tr:
                break
        return out
    ok("H 同 seed 两跑逐字节一致（确定性）", traj(0) == traj(0))
    env = fresh(); env.reset(seed=0)
    ok("I _ego_vs() → VesselState(含 length)", isinstance(env._ego_vs(), VesselState) and env._ego_vs().length == 175.0)
    ok("I _obs_vs() → VesselState 或 None（窗外）", env._obs_vs() is None or isinstance(env._obs_vs(), VesselState))

    print("\n===== J) 多他船 runtime guard（D40#4/L49#2：盾只护 obstacles[0]，多船须 fail-fast）=====")
    import copy as _copy
    from trb_env.usv_env import assert_single_obstacle
    # 纯函数逻辑（load-bearing）：len≤1 放行、len>1 raise NotImplementedError
    _pf_pass = True
    try:
        assert_single_obstacle([], "X"); assert_single_obstacle([object()], "X")   # 无船/单船不 raise
    except Exception:
        _pf_pass = False
    ok("J assert_single_obstacle [] / [a] 不 raise（单船/无船放行）", _pf_pass)
    _pf_raised = False
    try:
        assert_single_obstacle([object(), object()], "X")
    except NotImplementedError:
        _pf_raised = True
    ok("J assert_single_obstacle [a,b] → NotImplementedError（>1 他船 fail-fast；改 >1 为 >2 则 FAIL）", _pf_raised)
    # 集成：构造双他船场景 → ContinuousProjectionEnv.__init__ 必 raise（坐实守护已接线、删 __init__ 调用则此项 FAIL）
    _sc2 = _copy.deepcopy(_sc)
    _ob1 = _copy.deepcopy(_sc2.dynamic_obstacles[0])
    _ob1._obstacle_id = _sc2.generate_object_id()
    _sc2.add_objects(_ob1)
    _int_raised = False
    try:
        ContinuousProjectionEnv(_sc2, _ppx)
    except NotImplementedError:
        _int_raised = True
    ok("J 双他船场景 → ContinuousProjectionEnv 构造 raise NotImplementedError（守护已接线）",
       _int_raised and len(_sc2.dynamic_obstacles) == 2)

    print("\n===== K) 动作混叠惩罚（Markgraf 2026 式20 h=w‖u−uφ‖²·`03` L97）=====")
    from trb_env.usv_env import A_NORMAL_ACCEL_MAX as _ANA, A_NORMAL_OMEGA_MAX as _ANO
    _ubox = np.array([_ANA, _ANO], dtype=float)   # ⚠️ 用 float64 常量（=impl 的 self._u_box 同源）·非 action_space.high(float32→量化差~1e-7 破 <1e-9 比对·memory bit-identical 陷阱）
    _rng = np.random.default_rng(0)
    _acts = [_rng.uniform(-_ubox, _ubox) for _ in range(200)]            # 固定动作序列（两 run 同序列·覆盖多 source）
    def _run_alias(w):
        e = fresh(alias_weight=w); e.reset(seed=0); out = []
        for a in _acts:
            _o, _r, te, tr, info = e.step(a)
            out.append((_r, info.get("r_alias", "ABSENT"), info["source"],
                        np.asarray(info["u_desired"], float), np.asarray(info["u_applied"], float),
                        e.env.ego.copy()))
            if te or tr:
                break
        return out
    _r0 = _run_alias(0.0); _rw = _run_alias(1.5)
    ok("K w=0 默认无 r_alias 键（逐位等价 bit-identical）", all(s[1] == "ABSENT" for s in _r0))
    ok("K w=1.5 与 w=0 状态轨迹逐位相同（惩罚只动 reward·不改动力学/动作）",
       len(_r0) == len(_rw) and all(float(np.max(np.abs(a[5] - b[5]))) < 1e-12 for a, b in zip(_r0, _rw)))
    _nproj = _okf = _okr = _nonproj_key = 0
    for a, b in zip(_r0, _rw):
        if b[2] == "projection":                                          # 投影步 u_applied==u_safe（投影在 box 内）
            _nproj += 1
            _d = (b[3] - b[4]) / _ubox; _exp = -1.5 * float(np.dot(_d, _d))
            if b[1] != "ABSENT" and abs(b[1] - _exp) < 1e-9: _okf += 1
            if abs(b[0] - (a[0] + (b[1] if b[1] != "ABSENT" else 0.0))) < 1e-9: _okr += 1
        elif b[1] != "ABSENT":
            _nonproj_key += 1
    ok(f"K 投影步({_nproj})公式 r_alias=−w‖(u_des−u_safe)/u_box‖² 全对（改归一化/符号则 FAIL）", _nproj > 0 and _okf == _nproj)
    ok("K 投影步 r 恰好被减去 r_alias（additive·钱图列可剥离）", _nproj > 0 and _okr == _nproj)
    ok("K 非投影步（no_obstacle/emergency/fallback）无 r_alias 键（只投影步激活·防尺度爆炸）", _nonproj_key == 0)
    _wneg = False
    try:
        fresh(alias_weight=-1.0)
    except ValueError:
        _wneg = True
    ok("K alias_weight<0 → ValueError（fail-fast）", _wneg)
    _wnan = 0
    for _bad in (float("nan"), float("inf")):
        try:
            fresh(alias_weight=_bad)
        except ValueError:
            _wnan += 1
    ok("K alias_weight=nan/inf → ValueError（fail-fast·复审 MINOR：nan<0/inf<0 均 False 会漏过）", _wnan == 2)
    # 对抗动作子用例（复审 NIT）：随机 RL 箱内动作的投影修正量≈0(‖d‖²~1e-17)·对 scale/符号 bug 判别力弱；
    #   故用恒定满舵 give-way-违逆动作强迫盾真修正(‖d‖²≫0)·在【真修正步】手算核公式锁住归一化与符号。
    _adv = [np.array([_ubox[0], _ubox[1]], dtype=float)] * 60   # 满舵加速+左转（常违逆 give-way 右转→盾真投影）
    _ea = fresh(alias_weight=1.5); _ea.reset(seed=0); _amax = (-1.0, None)
    for a in _adv:
        _o, _r, te, tr, info = _ea.step(a)
        if info["source"] == "projection":
            _d = (np.asarray(info["u_desired"], float) - np.asarray(info["u_applied"], float)) / _ubox
            _n2 = float(np.dot(_d, _d))
            if _n2 > _amax[0]:
                _amax = (_n2, (info.get("r_alias", "ABSENT"), -1.5 * _n2))
        if te or tr:
            break
    _has_real = _amax[0] > 0.01   # 真有实质修正步（非 ≈0 regime）
    _adv_ok = _has_real and _amax[1][0] != "ABSENT" and abs(_amax[1][0] - _amax[1][1]) < 1e-9
    ok(f"K 对抗动作真修正步(‖d‖²={_amax[0]:.3f}>0.01)上公式 r_alias=−w‖d‖² 手算吻合（锁归一化/符号·random 测不到）", _adv_ok)

    print("\n===== K2) action-rate 平滑惩罚（治 bang-bang·`03` L98·r_rate=−w‖Δu_desired/u_box‖²）=====")
    _acts2 = [_rng.uniform(-_ubox, _ubox) for _ in range(120)]   # 变化动作=consecutive 差≠0=真测 r_rate
    def _run_rate(w):
        e = fresh(rate_weight=w); e.reset(seed=0); out = []
        for a in _acts2:
            _o, _r, te, tr, info = e.step(a)
            out.append((_r, info.get("r_rate", "ABSENT"), np.asarray(info["u_desired"], float), e.env.ego.copy()))
            if te or tr:
                break
        return out
    _q0 = _run_rate(0.0); _qw = _run_rate(1.5)
    ok("K2 w=0 默认无 r_rate 键（逐位等价 bit-identical）", all(s[1] == "ABSENT" for s in _q0))
    ok("K2 w=1.5 与 w=0 状态轨迹逐位相同（惩罚只动 reward·不改动力学）",
       len(_q0) == len(_qw) and all(float(np.max(np.abs(a[3] - b[3]))) < 1e-12 for a, b in zip(_q0, _qw)))
    ok("K2 首步无 r_rate 键（无上一步动作·reset 清）", _qw[0][1] == "ABSENT")
    _rf = _rr = 0; _nck = 0
    for i in range(1, len(_qw)):
        _du = (_qw[i][2] - _qw[i-1][2]) / _ubox; _exp = -1.5 * float(np.dot(_du, _du)); _nck += 1
        if _qw[i][1] != "ABSENT" and abs(_qw[i][1] - _exp) < 1e-9: _rf += 1
        if abs(_qw[i][0] - (_q0[i][0] + (_qw[i][1] if _qw[i][1] != "ABSENT" else 0.0))) < 1e-9: _rr += 1
    ok(f"K2 公式 r_rate=−w‖(u_t−u_{{t-1}})/u_box‖² 全对（{_nck} 步·改归一化/符号则 FAIL）", _nck > 0 and _rf == _nck)
    ok("K2 r 恰好被减去 r_rate（additive·可消融）", _nck > 0 and _rr == _nck)
    # reset 清 _prev：跑 5 步 → reset → 新 episode 首步必无罚（不沿用上 episode 末动作=不跨 episode 注伪抖动罚）
    _er = fresh(rate_weight=1.5); _er.reset(seed=0)
    for a in _acts2[:5]: _er.step(a)
    _er.reset(seed=0)
    _o, _r, _te, _tr, _i2 = _er.step(_acts2[0])
    ok("K2 reset 后新 episode 首步无 r_rate 键（不跨 episode 注伪抖动罚）", _i2.get("r_rate", "ABSENT") == "ABSENT")
    _rnan = 0
    for _bad in (-1.0, float("nan"), float("inf")):
        try:
            fresh(rate_weight=_bad)
        except ValueError:
            _rnan += 1
    ok("K2 rate_weight<0/nan/inf → ValueError（fail-fast）", _rnan == 3)
    # alias + rate 共存正交（同开两惩罚·r 同时含 r_alias[投影步] + r_rate[非首步]·互不干扰）
    _eb = fresh(alias_weight=1.0, rate_weight=1.0); _eb.reset(seed=0)
    _both_keys = set()
    for a in _acts2[:40]:
        _o, _r, te, tr, info = _eb.step(a)
        if "r_alias" in info: _both_keys.add("alias")
        if "r_rate" in info: _both_keys.add("rate")
        if te or tr: break
    ok("K2 alias+rate 同开可共存（两键独立出现·正交两靶）", _both_keys == {"alias", "rate"})

    # ---------------- P0(L147 复审补测) SE-RL 盾开关 shield flag（硬门:守护默认有盾臂 bit-identical + 无盾臂语义·此前 0 committed 用例）----------------
    print("===== P0(L147) SE-RL 盾开关 shield flag =====")
    # P0-a flag 默认值锁定（改默认→连续臂静默变无盾→钱图静默错=最贵 failure）
    ok("P0-a 默认 shield=True（改默认则连续臂静默变无盾·钱图静默错）", fresh().shield is True)
    ok("P0-a shield=False 可显式关（连续无盾臂）", fresh(shield=False).shield is False)
    # P0-b 默认(缺省 shield) ≡ 显式 shield=True 逐位等价（flag no-op·锁"默认有盾臂"不被未来重构 step() 静默改坏）
    _p0_acts = [(0.03, 0.01), (-0.02, 0.015), (0.048, -0.018), (0.0, 0.0), (0.01, -0.005)]
    _ed = fresh(); _et = fresh(shield=True)
    _ed.reset(seed=0); _et.reset(seed=0)
    _mdr = _mdu = _mdo = 0.0; _sync = True
    for _k in range(140):
        _a = np.array(_p0_acts[_k % len(_p0_acts)], float)
        _o1, _r1, _t1, _tr1, _i1 = _ed.step(_a)
        _o2, _r2, _t2, _tr2, _i2 = _et.step(_a)
        _mdr = max(_mdr, abs(_r1 - _r2))
        _mdu = max(_mdu, float(np.max(np.abs(np.asarray(_i1["u_applied"], float) - np.asarray(_i2["u_applied"], float)))))
        _mdo = max(_mdo, float(np.max(np.abs(_o1 - _o2))))
        if (_t1 or _tr1) != (_t2 or _tr2): _sync = False
        if _t1 or _tr1:
            break
    ok(f"P0-b 默认(缺省)≡shield=True 逐位等价 Δr={_mdr:.1e}/Δu={_mdu:.1e}/Δobs={_mdo:.1e}（<1e-12·锁默认有盾臂）",
       _mdr < 1e-12 and _mdu < 1e-12 and _mdo < 1e-12 and _sync)
    # P0-c 有盾真做避碰/合规工作（source 含 projection/emergency 等主动盾档·非全 no_obstacle 空跑）→防"盾分支被改坏→静默全 unshielded"
    _active = {"projection", "emergency", "relaxed", "collision_min", "degenerate"}
    _env_on = fresh(shield=True); _env_on.reset(seed=0)            # run_ep 不自带 reset（env.ego 未初始化会崩）
    _recs_on = run_ep(_env_on, u=(0.05, 0.02), max_steps=220, collect=True)
    _srcs_on = {r["source"] for r in _recs_on}
    ok(f"P0-c shield=True 盾真介入（source 含主动档 {_srcs_on & _active}·且无 'unshielded'）",
       bool(_srcs_on & _active) and "unshielded" not in _srcs_on)
    # P0-d 无盾臂语义（对齐 UnshieldedUSVEnv 语义·非数值）：每步 rho==NO_CONFLICT / source=='unshielded' / 不推状态机 / 无紧急
    _eo = fresh(shield=False); _eo.reset(seed=0)
    _ok_off = True; _srcs_off = set()
    for _k in range(220):
        _o, _r, _te, _tr, _info = _eo.step(np.array([0.05, 0.02], float))
        _srcs_off.add(_info["source"])
        if (_info["rho"] != RHO_NO_CONFLICT or _info["source"] != "unshielded"
                or _info["give_way_dir"] is not None or _info["emergency_mode"] is not None):
            _ok_off = False
        if _te or _tr:
            break
    ok(f"P0-d shield=False 每步 rho==NO_CONFLICT & source=='unshielded' & 无 give_way/emergency（sources={_srcs_off}）",
       _ok_off and _srcs_off == {"unshielded"})
    # P0-e 无盾臂 u_applied==clip(u_desired 到物理箱 ±a_max/±w_max)（施 RL 原动作·仅物理限幅·不投影）
    _eo2 = fresh(shield=False); _eo2.reset(seed=0)
    _ain = np.array([0.04, 0.015], float)                          # 箱内 → clip 恒等 → u_applied==u_desired
    _o, _r, _te, _tr, _iin = _eo2.step(_ain)
    ok(f"P0-e shield=False 箱内动作 u_applied==u_desired（{np.asarray(_iin['u_applied'], float)} vs {_ain}）",
       np.allclose(np.asarray(_iin["u_applied"], float), _ain, atol=1e-6))
    _eo3 = fresh(shield=False); _eo3.reset(seed=0)
    _aout = np.array([0.5, -0.5], float)                           # 远越物理满程 → clip 到 ±A_MAX/±W_MAX
    _o, _r, _te, _tr, _iout = _eo3.step(_aout)
    ok(f"P0-e shield=False 越物理箱动作 u_applied==clip(±{A_MAX}/±{W_MAX})（{np.asarray(_iout['u_applied'], float)}）",
       np.allclose(np.asarray(_iout["u_applied"], float), [A_MAX, -W_MAX], atol=1e-6))

print()
if _fail == 0:
    print(f"✅ 全部 PASS（{_total} 项）")
else:
    print(f"❌ {_fail}/{_total} 项 FAIL")
sys.exit(1 if _fail else 0)
