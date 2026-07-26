#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`run_baselines_official.py`（外部基线 · 官方测试集闭环）的结构冒烟测试。

**本机可跑**（无需 commonocean / vesselmodels / sb3）。与 `test_reeval_official.py` 的桩法同源，但**关键差别**：

  🔴 **`trb_env.evaluate` 用【真的】，不是桩** —— 只把 env / 场景载入 / 官方动力学换成桩。
     理由（`03` L215-A 的事故）：本项目上一版基线 runner 的 11 项本机自检全过，
     但 `GeometricPolicy` 只有 `bind()`、而官方评估器找的钩子叫 **`bind_env`**
     ⟹ env 永不绑定 ⟹ **真闭环第一步就 AttributeError**。
     纯桩测抓不到这种"契约名字对不上"，**跑真评估器才抓得到** ⟹ 本文件坚持用真的。
     （桩掉的动力学被做成"一被调到就 raise"，避免悄悄用假物理糊过去。）

覆盖：
  ① **钩子契约**：从 `trb_env/evaluate.py` 源码里**读出**它 hasattr 的那个名字，断言策略类真有它（改名两边都会炸）
  ② **闭环真跑**：真 `evaluate_continuous` + 桩 env → 落盘 json 结构 / 三口径 / 卖点指标 / Pareto 汇总
  ③ **逐局对齐闸**：策略步数 ≠ 评估步数 → fail-closed（不可行率会错位到别的局）
  ④ **分母闸**：clean/strict 对不上 577/563 → fail-closed（与四臂不同分母就不能同表比）
  ⑤ **同分母硬比对闸**：strict 键集合与主线 json 不一致 → fail-closed
  ⑥ **终端门 knob 闸**：去朝向门被打开 → fail-closed（到达率不可比）
  ⑦ **轨迹键闸**：要画轨迹的场景不在池里 → fail-closed（免"跑完才发现画不出图"）
  ⑧ 纯逻辑自检（Pareto / 选择 / 抽样 / 泄漏计数）全过

⚠️ 桩测只保「接线与口径」，**不保真跑**——真跑仍须服务器冒烟（`03` L202 教训）。

跑： python 代码/tests/test_run_baselines_official.py
"""
import importlib
import json
import math
import os
import re
import sys
import tempfile
import types
import unittest

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.dirname(_HERE)                       # …/代码
_TRB = os.path.dirname(_CODE)                        # …/TRB
for _p in (_HERE, _CODE, os.path.join(_CODE, "m1_dock_wip")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_BALANCED = os.path.join(_TRB, "balanced_pool")


# ════════════════════════ 重依赖桩（动力学"被调到就炸"·不许静默用假物理） ════════════════════════
def _mod(name, **attrs):
    m = types.ModuleType(name)
    m.__path__ = []
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


class _VesselParams:
    def __init__(self):
        self.l, self.w, self.a_max, self.w_max, self.v_max = 175.0, 25.4, 0.24, 0.03, 9.5


def _install_base_stubs():
    """`vesselmodels` / `commonroad` 最小桩——只为让 `trb_env.evaluate` 能 import 成功。"""
    def _boom(*_a, **_k):
        raise AssertionError("桩：本测试不该调到官方动力学（若调到 = 桩 env 漏了、请修测试而不是放行）")

    _mod("vesselmodels")
    _mod("vesselmodels.vessel_dynamics_yp", vessel_dynamics_yp=_boom)
    _mod("vesselmodels.parameters_vessel_1", parameters_vessel_1=_VesselParams)

    class CustomState:                                   # usv_termination 只当数据容器用
        def __init__(self, **kw):
            self.__dict__.update(kw)

    _mod("commonroad")
    _mod("commonroad.scenario")
    _mod("commonroad.scenario.state", CustomState=CustomState)


# ════════════════════════ 桩 env：实现 run_episode_continuous 要的那套契约 ════════════════════════
class _Inner:
    """站位内层 USVEnv：只提供 `dt` / `last_action` / `obs_builder.goal_center`（评估器与策略真会读的那几个）。"""

    def __init__(self, goal):
        self.dt = 10.0
        self.last_action = np.zeros(2, float)
        self.goal_center = np.asarray(goal, float)
        self.obs_builder = types.SimpleNamespace(goal_center=self.goal_center)


class _FakeScenario:
    """站位 commonocean Scenario：`classify_pool` 只读 `.dynamic_obstacles[0].initial_state`。"""

    def __init__(self, tid):
        self.tid = int(tid)
        o = types.SimpleNamespace(position=np.array([2000.0, 0.0]), orientation=math.pi, velocity=9.5)
        self.dynamic_obstacles = [types.SimpleNamespace(initial_state=o)]


class _FakePlanningProblem:
    """站位 PlanningProblem：`classify_pool` 读 `.initial_state` 与 `.goal.state_list[0].position.center`。

    ego 起点的 x 故意取 T-id ⟹ 桩 `classify` 能按它给出两种会遇类型 ⟹ 分型聚合真被测到。
    """

    def __init__(self, tid):
        self.initial_state = types.SimpleNamespace(position=np.array([float(tid), 0.0]),
                                                  orientation=0.0, velocity=9.5)
        self.goal = types.SimpleNamespace(
            state_list=[types.SimpleNamespace(position=types.SimpleNamespace(center=np.array([900.0, 0.0])))])


def _make_fake_env_cls():
    from trb_env.usv_colregs import VesselState                  # 真 dataclass（ViolationCounter 要吃它）

    class FakeContinuousEnv:
        """桩 `ContinuousProjectionEnv(shield=False)`：纯运动学 + 正对遇他船。

        标定原则（不然测不出东西）：
          · **必须真会动、且动作不同 → 轨迹不同** ⟹ 平滑度/次网格/路径长这些指标才有区分力。
          · **到达率必须落在 (0, 100) 之间**：目标只有 300m（全速一步 95m），到达门 60-140m 随 T-id 变
            ⟹ 有的场景够、有的差一点 ⟹ 聚合不是 0% 也不是 100%（那两种都测不出聚合逻辑）。
          · 碰撞判据取 200m（船长 175m 量级），让"不避碰的纯 PD"确实会撞出一些 ⟹ 碰撞聚合也被测到。
          · **`plain` 变体下三个方法才会分化**：`colregs` 变体在让路态会把 ω 打满右舵（三法都饱和 = 同数），
            所以"三法给出不同结果"这条断言必须用 `plain` 跑（见 `test_arrival_differs_between_methods`）。
        """
        MAX_STEPS = 12

        def __init__(self, sc, pp, *, shield=True, **kw):
            assert shield is False, "外部基线必须 shield=False（施原动作·不投影）"
            self.sid = int(getattr(sc, "tid", sc or 0))
            self.kw = kw
            self.goal = np.array([300.0, 0.0], float)
            self.env = _Inner(self.goal)
            self.reset()                                   # 让 `_ego_vs()` 在 reset 之前被调也不炸

        def reset(self, *, seed=None, options=None):
            self.p = np.array([0.0, 0.0], float)
            self.psi, self.v = 0.0, 9.5
            self.po = np.array([700.0 + 40.0 * (self.sid % 7), 0.0], float)    # 他船正前方（正对遇）
            self.pso, self.vo = math.pi, 9.5
            self.t = 0
            return np.zeros(27, np.float32), {}

        def _vs(self, p, psi, v):
            return VesselState(position=np.asarray(p, float), orientation=float(psi),
                               velocity=float(v), length=175.0)

        def _ego_vs(self):
            return self._vs(self.p, self.psi, self.v)

        def _obs_vs(self):
            return self._vs(self.po, self.pso, self.vo)

        def step(self, action):
            u = np.asarray(action, float)
            a = float(np.clip(u[0], -0.24, 0.24))
            w = float(np.clip(u[1], -0.03, 0.03))
            dt = self.env.dt
            self.v = float(np.clip(self.v + a * dt, 0.0, 9.5))
            self.psi += w * dt
            self.p = self.p + np.array([math.cos(self.psi), math.sin(self.psi)]) * self.v * dt
            self.po = self.po + np.array([math.cos(self.pso), math.sin(self.pso)]) * self.vo * dt
            self.env.last_action = np.array([a, w], float)
            self.t += 1
            d = float(np.linalg.norm(self.goal - self.p))
            goal = d < 60.0 + 20.0 * (self.sid % 5)           # 让不同场景难度不同 ⟹ 到达率落在中间
            hit = float(np.linalg.norm(self.po - self.p)) < 200.0     # 船长 175m 量级
            flags = {"time": self.t >= self.MAX_STEPS, "area": False, "stopped": False,
                     "collision": hit, "goal": goal}
            info = {"flags": flags, "rho": 0, "rho_acting": 0, "source": "unshielded",
                    "u_desired": u.copy(), "u_applied": self.env.last_action.copy()}
            done = goal or hit or self.t >= self.MAX_STEPS
            return np.zeros(27, np.float32), 0.0, bool(done), False, info

    return FakeContinuousEnv


def _install_run_stubs(tmp, *, n_pool_files, goal_ignore_orient=False, augment_rho=False):
    """桩 `run_step4e` / `trb_env.usv_scenarios` / `trb_env.usv_continuous_shield` / `classify_scenarios`。"""
    RO = importlib.import_module("reeval_official")

    sdir = os.path.join(tmp, "scen")
    os.makedirs(sdir, exist_ok=True)
    for i in range(n_pool_files):
        with open(os.path.join(sdir, f"T-{i}.xml"), "w", encoding="utf-8") as fh:
            fh.write("x" * 1100)                       # >1000 字节，才会被 _build_pool 认成"下到了"

    R = types.ModuleType("run_step4e")
    R.make_split = lambda n, f, s=0, pool_size=None: RO.make_split_mirror(n, f, s, pool_size)
    R._SDIR = sdir
    R._download = lambda ids, workers=16: ([], [])
    R._CONTINUOUS_SHIELD, R._GOAL_CONE_HALF_DEG, R._GOAL_CONE_HALF_RAD = True, None, None
    R._GOAL_V_FLOOR = 2.0
    R._AUGMENT_RHO, R._GOAL_IGNORE_ORIENT = augment_rho, goal_ignore_orient
    sys.modules["run_step4e"] = R

    # 场景池：把 T-id 当 scenario 传下去（桩 env 用它造不同难度）
    scen = types.ModuleType("trb_env.usv_scenarios")
    scen.load_scenario_pool = lambda paths: [(_FakeScenario(RO.key_of_path(p)),
                                              _FakePlanningProblem(RO.key_of_path(p))) for p in paths]
    sys.modules["trb_env.usv_scenarios"] = scen

    sh = types.ModuleType("trb_env.usv_continuous_shield")
    sh.ContinuousProjectionEnv = _make_fake_env_cls()
    sys.modules["trb_env.usv_continuous_shield"] = sh

    cs = types.ModuleType("classify_scenarios")
    cs.classify = lambda *a, **k: (("crossing" if int(a[0][0]) % 2 == 0 else "head_on"), 0.0, 0.0)
    sys.modules["classify_scenarios"] = cs
    return R


def _run_main(env, *, tmp, n_pool_files=2000, **stub_kw):
    """在给定环境变量下 reload 并跑 `run_baselines_official.main()` → 返回落盘 payload。"""
    _install_base_stubs()
    old = {k: os.environ.get(k) for k in env}
    os.environ.update({k: str(v) for k, v in env.items()})
    try:
        import trb_env.evaluate                        # 🔴 真的（先 import 好，免得被下面的桩挡住）
        assert trb_env.evaluate.evaluate_continuous
        for m in ("run_step4e", "trb_env.usv_scenarios", "trb_env.usv_continuous_shield",
                  "classify_scenarios"):
            sys.modules.pop(m, None)
        _install_run_stubs(tmp, n_pool_files=n_pool_files, **stub_kw)
        mod = importlib.reload(importlib.import_module("run_baselines_official"))
        mod.main()
        out = env.get("BASELINE_OUT")
        if out and os.path.exists(out):
            with open(out, encoding="utf-8") as fh:
                return json.load(fh)
        return None
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


_BASE_ENV = {"BASELINE_MANIFEST_DIRS": _BALANCED, "REEVAL_MANIFEST_DIRS": _BALANCED,
             "BASELINE_METHODS": "vo,cbf,pd", "BASELINE_BOX": "rl", "BASELINE_SWEEP": "off",
             "BASELINE_SMOKE": "0", "BASELINE_TUNE_N": "6", "BASELINE_N": "0",
             "BASELINE_KEYS_REF": "", "BASELINE_TRAJ_KEYS": "", "BASELINE_ENVCFG_ACK": "0",
             "BASELINE_LEAK_ACK": "0", "BASELINE_NOMINAL": "pd", "BASELINE_TUNE_SRC": "train"}


# ════════════════════════════════════ 测试 ════════════════════════════════════
class TestHookContract(unittest.TestCase):
    """① 钩子契约：名字从**真源码**里读出来，两边任一改名都会炸（这就是 L215-A 那个洞的回归钉子）。"""

    def test_bind_hook_name_matches_policy(self):
        with open(os.path.join(_CODE, "trb_env", "evaluate.py"), encoding="utf-8") as fh:
            src = fh.read()
        names = set(re.findall(r'hasattr\(\s*model\s*,\s*[\'"]([A-Za-z_]\w*)[\'"]\s*\)', src))
        self.assertTrue(names, "在 evaluate.py 里没找到 hasattr(model, '...') 钩子——契约变了，测试须同步更新")
        _install_base_stubs()
        from usv_baseline_runner import GeometricPolicy
        for n in names:
            self.assertTrue(hasattr(GeometricPolicy, n),
                            f"官方评估器会调 model.{n}(...)，但 GeometricPolicy 没有 ⟹ 真闭环会崩")

    def test_predict_signature_accepts_deterministic(self):
        _install_base_stubs()
        from usv_baseline_runner import GeometricPolicy
        import inspect
        sig = inspect.signature(GeometricPolicy.predict)
        self.assertIn("deterministic", sig.parameters)


class TestClosedLoop(unittest.TestCase):
    """② 闭环真跑（真 evaluate_continuous + 桩 env）。"""

    def test_full_pool_three_scopes_and_headline_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "bl.json")
            pay = _run_main({**_BASE_ENV, "BASELINE_OUT": out, "STEP4E_SDIR": os.path.join(tmp, "scen")},
                            tmp=tmp)
            self.assertIsNotNone(pay)
            # 分母 = 与四臂同一套（600 → 577 → 563）
            self.assertEqual(pay["池"]["N"], 600)
            self.assertEqual(pay["池"]["clean"], 577)
            self.assertEqual(pay["池"]["strict"], 563)
            self.assertEqual(len(pay["strict键"]), 563)
            # 三个方法都出了数，且三口径都在
            tags = [t for t in pay["结果"] if not t.startswith("_")]
            self.assertEqual(len(tags), 3, tags)
            for t in tags:
                r = pay["结果"][t]
                for scope in ("全部", "clean", "strict"):
                    self.assertIsNotNone(r[scope], (t, scope))
                self.assertEqual(r["strict"]["n"], 563)
                self.assertTrue(0.0 <= r["strict"]["到达率%"] <= 100.0)
                # 🔴 卖点/平滑指标必须自动带上（`04 §1.5` 第①②条·本项目已两次栽在"下游忘了取"）
                ctrl = r["strict"]["控制质量"]
                for k in ("ctrl_jerk_norm_mean", "yaw_incr_mean", "accel_incr_mean", "path_len_m",
                          "subgrid_yaw_frac", "subgrid_accel_frac"):
                    self.assertIn(k, ctrl, (t, k))
                # 不可行率（这条线的看点之一）必须逐局注入并被聚合
                self.assertIn("baseline_infeasible_pct", ctrl, t)
                # 🔴 样本量键必须在（`03` L216-D 的事故）：缺了它们，指标为 None 时键直接消失
                #    ⟹「没采到」和「采到了但样本量是 0」分不开，能静默整整一批实验。
                for k in ("n_inbox_pairs", "n_pairs_giveway", "n_pairs_other"):
                    self.assertIn(k, ctrl, f"{t} 缺样本量键 {k} —— 指标为空时就分不清是没采到还是样本量为 0")
            # 分型（用的是 reeval_official.classify_pool·与四臂同判据）
            self.assertTrue(pay["会遇类型计数"])
            self.assertTrue(pay["结果"][tags[0]]["分型_clean"])
            # Pareto 汇总（framing = 各有所长·不是"我们样样赢"）
            self.assertIn("_pareto", pay["结果"])
            self.assertTrue(pay["结果"]["_pareto"]["前沿"])
            self.assertTrue(pay["全部完成"])

    def test_arrival_differs_between_methods(self):
        """三个方法必须给出【不同】结果——否则说明桩 env 没区分力、上面那些断言都是空转。

        用 `plain` 变体：`colregs` 变体在让路态会把 ω 打满右舵（三法都饱和成同一个动作 = 同数），
        那是**真实性质**（三条基线共用同一个标称），不是 bug；要看它们各自的避碰逻辑就得关掉这个偏置。
        """
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "bl.json")
            pay = _run_main({**_BASE_ENV, "BASELINE_OUT": out, "BASELINE_N": "60",
                             "BASELINE_VARIANTS": "plain", "BASELINE_LEAK_ACK": "1",
                             "STEP4E_SDIR": os.path.join(tmp, "scen")}, tmp=tmp)
            res = {t: pay["结果"][t]["全部"] for t in pay["结果"] if not t.startswith("_")}
            sig = {t: (round(m["到达率%"], 6), round(m["碰撞率%"], 6),
                       round((m["控制质量"] or {}).get("path_len_m", 0.0), 3)) for t, m in res.items()}
            self.assertGreater(len(set(sig.values())), 1,
                               f"三个方法给出完全相同的画像 → 桩 env 没区分力：{sig}")

    def test_two_boxes_both_reported(self):
        """反稻草人：rl / full 两档都要出现在结果里。"""
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "bl.json")
            pay = _run_main({**_BASE_ENV, "BASELINE_OUT": out, "BASELINE_BOX": "rl,full",
                             "BASELINE_METHODS": "vo", "BASELINE_N": "40", "BASELINE_LEAK_ACK": "1",
                             "STEP4E_SDIR": os.path.join(tmp, "scen")}, tmp=tmp, n_pool_files=2000)
            boxes = {pay["结果"][t]["box"] for t in pay["结果"] if not t.startswith("_")}
            self.assertEqual(boxes, {"rl", "full"})

    def test_sweep_selects_and_reports_dropped(self):
        """扫参 → 调参池与报数池不重叠 → 选出的配置 ≤ FINAL_MAX，且调参明细落盘。"""
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "bl.json")
            pay = _run_main({**_BASE_ENV, "BASELINE_OUT": out, "BASELINE_SWEEP": "on",
                             "BASELINE_METHODS": "vo", "BASELINE_BOX": "rl", "BASELINE_TUNE_N": "5",
                             "BASELINE_FINAL_MAX": "2", "BASELINE_N": "40", "BASELINE_LEAK_ACK": "1",
                             "STEP4E_SDIR": os.path.join(tmp, "scen")}, tmp=tmp, n_pool_files=2000)
            self.assertTrue(pay["调参结果"], "扫参跑了却没落调参明细")
            self.assertEqual(len(pay["调参结果"]), 8, "VO 网格应是 4 个 tau × 2 个 margin")
            finals = [t for t in pay["结果"] if not t.startswith("_")]
            self.assertLessEqual(len(finals), 2, finals)
            for t in finals:
                self.assertIsNotNone(pay["结果"][t]["调参指标"])


class TestFailClosed(unittest.TestCase):
    """③④⑤⑥⑦ 五道闸：该炸的地方必须炸（防"数是错的却看不出来"）。"""

    def test_episode_misalignment_aborts(self):
        _install_base_stubs()
        mod = importlib.reload(importlib.import_module("run_baselines_official"))
        cfg = {"method": "pd", "box": "rl", "variant": "colregs", "params": {}, "tag": "pd|rl|colregs"}

        def _fake_eval_cont(_factory, pol, pool, **_kw):
            pol.bind_env(_make_fake_env_cls()(0, None, shield=False))
            pol.predict(np.zeros(27, np.float32))
            per = [{"scenario_idx": 0, "reached": True, "collided": False, "steps": 99}]   # 步数故意对不上
            return {}, per

        with self.assertRaises(SystemExit) as cm:
            mod.eval_config(cfg, [(0, None)], lambda *a: None, _fake_eval_cont)
        self.assertIn("逐局对齐", str(cm.exception) + str(cm.exception.code))

    def test_episode_count_mismatch_aborts(self):
        _install_base_stubs()
        mod = importlib.reload(importlib.import_module("run_baselines_official"))
        cfg = {"method": "pd", "box": "rl", "variant": "colregs", "params": {}, "tag": "pd|rl|colregs"}

        def _fake_eval_cont(_factory, pol, pool, **_kw):
            return {}, [{"scenario_idx": 0, "reached": True, "collided": False, "steps": 1}]   # 一局没 bind

        with self.assertRaises(SystemExit) as cm:
            mod.eval_config(cfg, [(0, None)], lambda *a: None, _fake_eval_cont)
        self.assertIn("局", str(cm.exception.code))

    def test_wrong_denominator_aborts(self):
        """换了泄漏 manifest → clean/strict 不再是 577/563 → 必须中止（不同分母不能同表比）。"""
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit) as cm:
                _run_main({**_BASE_ENV, "BASELINE_OUT": os.path.join(tmp, "bl.json"),
                           "BASELINE_LEAK_MANIFESTS": "manifest_bigtr_hocr.json",
                           "STEP4E_SDIR": os.path.join(tmp, "scen")}, tmp=tmp)
            self.assertIn("分母", str(cm.exception.code))

    def test_keys_ref_mismatch_aborts(self):
        with tempfile.TemporaryDirectory() as tmp:
            ref = os.path.join(tmp, "ref.json")
            with open(ref, "w", encoding="utf-8") as fh:
                json.dump({"strict键": [1, 2, 3]}, fh)
            with self.assertRaises(SystemExit) as cm:
                _run_main({**_BASE_ENV, "BASELINE_OUT": os.path.join(tmp, "bl.json"),
                           "BASELINE_KEYS_REF": ref, "STEP4E_SDIR": os.path.join(tmp, "scen")}, tmp=tmp)
            self.assertIn("strict 键集合", str(cm.exception.code))

    def test_keys_ref_match_passes(self):
        """同分母硬比对：把真 strict 键写进 ref → 必须放行（闸不能只会炸、也要会过）。"""
        _install_base_stubs()
        RO = importlib.import_module("reeval_official")
        _tr, te = RO.official_split(None)
        mp = RO.find_manifest("manifest_hocr_200.json")
        mtr, mte, _ = RO.manifest_keys(mp)
        strict = sorted(set(te) - mtr - mte, key=str)
        with tempfile.TemporaryDirectory() as tmp:
            ref = os.path.join(tmp, "ref.json")
            with open(ref, "w", encoding="utf-8") as fh:
                json.dump({"strict键": strict}, fh)
            out = os.path.join(tmp, "bl.json")
            pay = _run_main({**_BASE_ENV, "BASELINE_OUT": out, "BASELINE_METHODS": "pd",
                             "BASELINE_KEYS_REF": ref, "STEP4E_SDIR": os.path.join(tmp, "scen")}, tmp=tmp)
            self.assertEqual(pay["池"]["strict"], 563)

    def test_goal_gate_knob_aborts(self):
        """去朝向门被打开 = 到达判据与四臂不同 ⟹ 到达率不可比 ⟹ 中止（除非显式 ACK）。"""
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit) as cm:
                _run_main({**_BASE_ENV, "BASELINE_OUT": os.path.join(tmp, "bl.json"),
                           "STEP4E_SDIR": os.path.join(tmp, "scen")}, tmp=tmp, goal_ignore_orient=True)
            self.assertIn("金标默认", str(cm.exception.code))

    def test_bad_traj_key_aborts(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit) as cm:
                _run_main({**_BASE_ENV, "BASELINE_OUT": os.path.join(tmp, "bl.json"),
                           "BASELINE_TRAJ_KEYS": "99999", "STEP4E_SDIR": os.path.join(tmp, "scen")}, tmp=tmp)
            self.assertIn("BASELINE_TRAJ_KEYS", str(cm.exception.code))

    def test_traj_written(self):
        """⑥ 轨迹真落盘（多算法轨迹对比图的输入·`04 §1.5` 第⑥条）。"""
        _install_base_stubs()
        RO = importlib.import_module("reeval_official")
        _tr, te = RO.official_split(None)
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "bl.json")
            pay = _run_main({**_BASE_ENV, "BASELINE_OUT": out, "BASELINE_METHODS": "vo",
                             "BASELINE_TRAJ_KEYS": f"{te[0]} {te[1]}",
                             "STEP4E_SDIR": os.path.join(tmp, "scen")}, tmp=tmp)
            self.assertEqual(pay["池"]["strict"], 563)
            tp = out[:-5] + "_traj.json"
            self.assertTrue(os.path.exists(tp), "开了 BASELINE_TRAJ_KEYS 却没落轨迹文件")
            with open(tp, encoding="utf-8") as fh:
                tj = json.load(fh)
            one = next(iter(tj.values()))
            self.assertEqual(set(one), {str(te[0]), str(te[1])})
            self.assertTrue(one[str(te[0])], "轨迹是空的")
            self.assertIn("ego_x", one[str(te[0])][0])

    def test_rl_nominal_requires_ckpt(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit) as cm:
                _run_main({**_BASE_ENV, "BASELINE_OUT": os.path.join(tmp, "bl.json"),
                           "BASELINE_NOMINAL": "rl", "STEP4E_SDIR": os.path.join(tmp, "scen")}, tmp=tmp)
            self.assertIn("BASELINE_RL_CKPT", str(cm.exception.code))


class TestPureLogic(unittest.TestCase):
    """⑧ 纯逻辑（Pareto / 选择 / 抽样 / 泄漏计数）——由脚本自带 selftest 一次跑完。"""

    def test_selftest_all_green(self):
        _install_base_stubs()
        os.environ["BASELINE_MANIFEST_DIRS"] = _BALANCED
        os.environ["REEVAL_MANIFEST_DIRS"] = _BALANCED
        mod = importlib.reload(importlib.import_module("run_baselines_official"))
        self.assertEqual(mod.selftest(), 0)

    def test_no_silent_truncation(self):
        _install_base_stubs()
        mod = importlib.reload(importlib.import_module("run_baselines_official"))
        rows = [{"tag": f"c{i}", "碰撞率%": float(i), "违规次数/局": float(10 - i), "到达率%": float(i)}
                for i in range(5)]                       # 互不支配（碰撞越大、违规越小）
        keep, dropped = mod.select_configs(rows, 2)
        self.assertEqual(len(keep), 2)
        self.assertTrue(dropped, "超出 FINAL_MAX 的配置必须被报出来，不许静默丢掉")


if __name__ == "__main__":
    unittest.main(verbosity=2)
