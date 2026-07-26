#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`reeval_official.py`（E5/E6 官方测试集泛化测）的结构冒烟测试。

**本机可跑**（无需 commonocean / vesselmodels / sb3）：用桩（stub）替掉 `run_step4e` / `trb_env.usv_scenarios`
/ `classify_scenarios` 三个重依赖，只测【本脚本自己的逻辑】：
  ① 官方 1400/600 划分 + 镜像与官方实现一致性校验
  ② 泄漏剔除：600 −23训练泄漏= 577 · 再 −14验证泄漏= 563（本项目亲算的数·钉死防回归）
  ③ 池序 ↔ T-id 对齐、strict 子集从同一趟 eval 里正确切出
  ④ 锚点复现检查：逐位复现放行 / 差一局警告放行 / 差很多【中止】
  ⑤ 数据集不可还原（strided/None）→ fail-closed
⚠️ 桩测只保「接线与口径」，**不保真跑**——真跑仍须服务器冒烟（`03` L202 教训：本机+对抗审漏过负速度崩）。

跑： python 代码/tests/test_reeval_official.py    或   pytest 代码/tests/test_reeval_official.py
"""
import importlib
import json
import os
import re
import sys
import tempfile
import types
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.dirname(_HERE)                       # …/代码
_TRB = os.path.dirname(_CODE)                        # …/TRB
sys.path.insert(0, _HERE)
sys.path.insert(0, _CODE)

_BALANCED = os.path.join(_TRB, "balanced_pool")
_SCEN = os.path.join(_TRB, "scenarios")


class _FakeInnerModel:
    """给 `policy_wrap` 套的极简内层模型：`.predict()` 返回固定 (a, ω)。"""

    def __init__(self, w=0.018):
        self.w = float(w)

    def predict(self, obs, deterministic=True, **kw):
        import numpy as _np
        return _np.array([0.0, self.w], dtype=float), None


# ---------------- 桩 ----------------
def _install_stubs(*, replay_arrival=None, anchor_arrival=None, n_anchor=40, augment_rho=False):
    """把三个重依赖换成桩。replay_arrival→到达局数；anchor_arrival→锚点重评到达率%；augment_rho=非金标 knob。"""
    mirror = importlib.import_module("reeval_official").make_split_mirror

    R = types.ModuleType("run_step4e")
    R.make_split = lambda n, f, s=0, pool_size=None: mirror(n, f, s, pool_size)   # 与镜像一致=校验应通过
    R._SDIR = _SCEN
    R._download = lambda ids, workers=16: ([f"{_SCEN}/T-{i}.xml" for i in ids], [])
    R._CONTINUOUS_SHIELD, R._GOAL_CONE_HALF_DEG = True, None
    R._GOAL_V_FLOOR, R._AUGMENT_RHO, R._GOAL_IGNORE_ORIENT = 2.0, augment_rho, False

    def _fake_per(n, n_reached):
        return [{"scenario_idx": i, "reached": i < n_reached, "collided": False,
                 "violations": 0.0, "emergency_pct": 3.0, "in_box_steps": (5 if i < n_reached else 0),
                 # `03` L203 的两个卖点指标 + 平滑度套件（evaluate.py 逐局真有这些键·下游必须带上）
                 "ctrl_jerk_norm_mean": 0.9, "yaw_incr_mean": 0.015, "accel_incr_mean": 0.0036,
                 "path_len_m": 4000.0, "subgrid_yaw_frac": 0.63, "subgrid_accel_frac": 0.44,
                 "yaw_incr_giveway": 0.017, "yaw_incr_other": 0.012}
                for i in range(n)]

    # ⚠️ 桩的签名必须跟真 `replay_eval` 一起改（r8 加了 traj_idxs）——否则脚本一传新参数、桩就 TypeError，
    #    而那是"测试自己坏了"、不是脚本坏了，很容易误判。`_traj_seen` 记录传进来的下标供断言。
    _traj_seen = []
    _wrap_seen = []

    def _replay(base, kind, weight, pool, *, continuous_algo=None, return_per=False, traj_idxs=None,
                policy_wrap=None):
        n = len(pool)
        if n == n_anchor and anchor_arrival is not None:                       # 锚点池
            k = round(anchor_arrival / 100.0 * n)
            per = _fake_per(n, k)
        else:
            k = replay_arrival if replay_arrival is not None else n // 2
            per = _fake_per(n, k)
        _traj_seen.append(None if traj_idxs is None else sorted(traj_idxs))
        _wrap_seen.append(policy_wrap)
        if policy_wrap is not None:      # 真 replay_eval 会把它套在载入的模型上；桩这里只验它能被正常调用
            policy_wrap(_FakeInnerModel())
        if traj_idxs:                                       # 模拟 evaluate 的行为：只给请求的那几局加 traj 键
            for p in per:
                if p["scenario_idx"] in traj_idxs:
                    p["traj"] = [{"ego_x": 0.0, "ego_y": 0.0, "ego_psi": 0.0, "step": 0, "rho": 0}]
        agg = {"n": n, "到达率%": 100.0 * sum(p["reached"] for p in per) / n}
        return (agg, per) if return_per else agg

    R.replay_eval = _replay
    R._traj_seen = _traj_seen
    R._wrap_seen = _wrap_seen

    def _lms(mp, bdir=None):
        """桩 load_manifest_split：manifest 池模式返回该 manifest 的真测试路径；锚点模式返回 n_anchor 个。"""
        with open(mp, encoding="utf-8") as fh:
            man = json.load(fh)
        if os.environ.get("_STUB_LMS_REAL") == "1":
            te = ([f"{_SCEN}/T-{int(x)}.xml" for x in man["head_on"]["test"]]
                  + [f"{_SCEN}/T-{int(x)}.xml" for x in man["crossing"]["test"]]
                  + [os.path.join(_BALANCED, os.path.basename(str(x))) for x in man["overtaking"]["test"]])
            return [], te, {"n_test": len(te)}
        return [], [f"{_SCEN}/T-{i}.xml" for i in range(n_anchor)], {"n_test": n_anchor}

    R.load_manifest_split = _lms
    sys.modules["run_step4e"] = R

    scen = types.ModuleType("trb_env.usv_scenarios")
    scen.load_scenario_pool = lambda paths: [(None, None) for _ in paths]
    pkg = types.ModuleType("trb_env")
    pkg.usv_scenarios = scen
    sys.modules["trb_env"] = pkg
    sys.modules["trb_env.usv_scenarios"] = scen

    cs = types.ModuleType("classify_scenarios")
    cs.classify = lambda *a, **k: ("crossing", 0.0, 0.0)
    sys.modules["classify_scenarios"] = cs
    return R


def _make_ckpt(dirpath, name, *, kind="continuous", party="Continuous-safe", seed=0,
               dataset="manifest_hocr_200.json", last_arrival=82.5):
    """造一个假 checkpoint（空 .zip/_vecnorm.pkl + 真 .progress.json）。"""
    os.makedirs(dirpath, exist_ok=True)
    base = os.path.join(dirpath, name)
    open(base + ".zip", "wb").close()
    open(base + "_vecnorm.pkl", "wb").close()
    rec = {"party": party, "kind": kind, "colregs_weight": 0.0, "seed": seed,
           "num_timesteps": 5_000_000, "trend": [{"step": 5_000_000, "到达率%": last_arrival}],
           "config_sig": {"kind": kind, "dataset": dataset, "continuous_algo": "ppo"}}
    with open(base + ".progress.json", "w", encoding="utf-8") as fh:
        json.dump(rec, fh, ensure_ascii=False)
    return base


def _run(env, **stub_kw):
    """在给定环境变量下 reload 并跑 reeval_official.main()；返回落盘 payload（若有）。"""
    old = {k: os.environ.get(k) for k in list(env)}
    os.environ.update({k: str(v) for k, v in env.items()})
    try:
        for m in ("run_step4e", "trb_env", "trb_env.usv_scenarios", "classify_scenarios"):
            sys.modules.pop(m, None)
        mod = importlib.reload(importlib.import_module("reeval_official"))
        _install_stubs(**stub_kw)
        mod.main()
        out = env.get("REEVAL_OUT")
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


class TestPureLogic(unittest.TestCase):
    def setUp(self):
        os.environ["REEVAL_MANIFEST_DIRS"] = _BALANCED
        self.mod = importlib.reload(importlib.import_module("reeval_official"))

    def test_official_split_1400_600(self):
        tr, te = self.mod.official_split(None)
        self.assertEqual((len(tr), len(te)), (1400, 600))
        self.assertFalse(set(tr) & set(te))

    def test_leak_counts_577_563(self):
        """本项目亲算的泄漏数——钉死，任何一方（划分/ manifest）改动都会在这里炸。"""
        _, te = self.mod.official_split(None)
        mp = self.mod.find_manifest("manifest_hocr_200.json")
        self.assertIsNotNone(mp, "找不到 balanced_pool/manifest_hocr_200.json")
        mtr, mte = self.mod.manifest_tids(mp)
        self.assertEqual((len(mtr), len(mte)), (94, 40))
        teS = set(te)
        self.assertEqual(len(teS & mtr), 23)
        self.assertEqual(len(teS & mte), 14)
        self.assertEqual(len(teS - mtr), 577)
        self.assertEqual(len(teS - mtr - mte), 563)

    def test_mirror_must_match_official(self):
        """镜像与官方 make_split 不一致 → 必须中止（防镜像漂移悄悄污染口径）。"""
        R = types.SimpleNamespace(make_split=lambda n, f, s=0, pool_size=None: ([0], [1]))
        with self.assertRaises(SystemExit):
            self.mod.official_split(R)

    def test_selftest_runs(self):
        self.mod.selftest()


class TestEndToEndStubbed(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="reeval_ck_")
        self.out = os.path.join(self.tmp, "out.json")
        self.base_env = {"REEVAL_MANIFEST_DIRS": _BALANCED, "REEVAL_CKDIRS": self.tmp,
                         "STEP4E_SDIR": _SCEN, "REEVAL_OUT": self.out}

    def test_official_pool_600_577_563_and_alignment(self):
        _make_ckpt(self.tmp, "Continuous-safe_s0_gold")
        p = _run({**self.base_env}, replay_arrival=400, anchor_arrival=82.5)
        self.assertEqual(p["池"]["N"], 600)                            # 全部都评（含被看过的·用来量污染幅度）
        self.assertEqual(p["池"]["clean"], 577)
        self.assertEqual(p["池"]["strict"], 563)
        self.assertEqual(p["池"]["训练泄漏"], 23)
        self.assertEqual(p["池"]["验证泄漏"], 14)
        r = p["结果"]["Continuous-safe_s0_gold"]
        self.assertEqual((r["全部"]["n"], r["clean"]["n"], r["strict"]["n"]), (600, 577, 563))
        self.assertEqual(r["看过的"]["n"], 37)                          # 600−563
        self.assertTrue(r["anchor"]["通过"])
        self.assertEqual(len(p["clean键"]), 577)
        self.assertFalse(set(p["clean键"]) & set(self.mod_train_ids()))  # clean 集与训练集零交集
        self.assertFalse(set(p["strict键"]) & set(self.mod_val_ids()))   # strict 集与验证集零交集

    def mod_train_ids(self):
        m = importlib.import_module("reeval_official")
        return m.manifest_tids(m.find_manifest("manifest_hocr_200.json"))[0]

    def mod_val_ids(self):
        m = importlib.import_module("reeval_official")
        return m.manifest_tids(m.find_manifest("manifest_hocr_200.json"))[1]

    def test_selling_point_metrics_survive_to_output(self):
        """🔴 卖点指标（次网格细调率 + 按态势拆转艏 + 平滑度套件）必须落进输出——
        本项目已两次栽在"指标接进 evaluate 了、下游忘了取"（`03` L203 那次 / 本脚本 2026-07-26）。"""
        _make_ckpt(self.tmp, "Continuous-safe_s0_gold")
        p = _run({**self.base_env, "REEVAL_ANCHOR": "0"}, replay_arrival=400)
        c = p["结果"]["Continuous-safe_s0_gold"]["strict"]["控制质量"]
        for k in ("subgrid_yaw_frac", "subgrid_accel_frac", "yaw_incr_giveway", "yaw_incr_other",
                  "ctrl_jerk_norm_mean", "yaw_incr_mean", "accel_incr_mean", "path_len_m"):
            self.assertIn(k, c, f"卖点/平滑指标 {k} 被下游丢了")
        self.assertAlmostEqual(c["subgrid_yaw_frac"], 0.63, places=6)
        self.assertIn("控制质量_仅到达局", p["结果"]["Continuous-safe_s0_gold"]["strict"])

    def test_manifest_pool_mode_600_with_overtaking(self):
        """E6 用的池：均衡大集测试 600（对遇200+交叉200+追越200）·金标训练零泄漏、验证 40 个在里面。"""
        _make_ckpt(self.tmp, "Discrete-safe_s0_gold", kind="shielded", party="Discrete-safe")
        os.environ["_STUB_LMS_REAL"] = "1"
        try:
            p = _run({**self.base_env, "REEVAL_POOL": "manifest:manifest.json", "REEVAL_ANCHOR": "0"},
                     replay_arrival=300)
        finally:
            os.environ.pop("_STUB_LMS_REAL", None)
        self.assertEqual(p["池"]["N"], 600)
        self.assertEqual(p["池"]["训练泄漏"], 0)                        # 94 训练全在大集训练侧
        self.assertEqual(p["池"]["验证泄漏"], 40)                       # 小集 test ⊂ 大集 test
        self.assertEqual(p["会遇类型计数"], {"对遇": 200, "交叉": 200, "追越": 200})
        r = p["结果"]["Discrete-safe_s0_gold"]
        self.assertEqual((r["全部"]["n"], r["clean"]["n"], r["strict"]["n"]), (600, 600, 560))
        self.assertEqual({t: m["n"] for t, m in r["分型_全部"].items()},
                         {"对遇": 200, "交叉": 200, "追越": 200})
        # 🔴 分型必须**加起来等于同名那一档的分母**（`03` L224）——这条不变量本可以早点抓到那个错：
        #    汇报里「按会遇态势拆开」那张表用的是 clean(577) 的数，配文却写 600 的类型个数，
        #    而全文其它数字都是 strict(563) ⟹ 一份汇报里混了三个分母，聚合数字上完全看不出来。
        for lvl, bt in (("全部", "分型_全部"), ("clean", "分型_clean"), ("strict", "分型_strict")):
            self.assertTrue(r[bt], f"{bt} 是空的 —— 分型没算出来，下游只能去借别档的数填表")
            self.assertEqual(sum(m["n"] for m in r[bt].values()), r[lvl]["n"],
                             f"{bt} 各类型局数之和 ≠ {lvl} 的分母 —— 拆开的表和总表对不上")

    def test_anchor_one_episode_off_passes_with_warning(self):
        """差一局 = 浮点噪声 → 放行（否则会因单局翻转白白中止一趟 eval）。"""
        _make_ckpt(self.tmp, "Continuous-safe_s0_gold", last_arrival=82.5)
        p = _run({**self.base_env}, replay_arrival=400, anchor_arrival=85.0)   # 40 局里差 1 局 = 2.5pt
        self.assertTrue(p["结果"]["Continuous-safe_s0_gold"]["anchor"]["通过"])

    def test_anchor_big_mismatch_aborts(self):
        """差很多 = eval 环境配置与训练时不一致 → 必须中止（本脚本最重要的防御）。"""
        _make_ckpt(self.tmp, "Continuous-safe_s0_gold", last_arrival=82.5)
        with self.assertRaises(SystemExit):
            _run({**self.base_env}, replay_arrival=400, anchor_arrival=40.0)

    def test_sidecar_out_of_sync_skips_anchor_not_abort(self):
        """progress.json 记的 zip 指纹对不上（run 被杀在存 zip 与写提交点之间）→ 跳过锚点、**不中止**。
        实证背景：2026-07-26 热启动那趟被这条误杀过（当时是零容忍 + 无指纹检查）。"""
        base = _make_ckpt(self.tmp, "Continuous-safe_s0_gold")
        with open(base + ".progress.json", encoding="utf-8") as fh:
            rec = json.load(fh)
        rec["ckpt_fingerprint"] = {"zip_mtime": 1.0, "zip_size": 999999}   # 与真 zip(0 字节) 对不上
        with open(base + ".progress.json", "w", encoding="utf-8") as fh:
            json.dump(rec, fh, ensure_ascii=False)
        p = _run({**self.base_env}, replay_arrival=400, anchor_arrival=40.0)   # 锚点差很多也不该中止
        r = p["结果"]["Continuous-safe_s0_gold"]
        self.assertEqual(r["anchor"], {"skipped": "sidecar_out_of_sync"})
        self.assertEqual(r["strict"]["n"], 563)                                 # 主评照常完成

    def test_anchor_tolerance_matches_measured_replay_noise(self):
        """容差须 ≥4 局（`03` L192-C 实测重放噪声最大 3/40 局）——2 局被误杀过一次，别退回去。"""
        _make_ckpt(self.tmp, "Continuous-safe_s0_gold", last_arrival=90.0)
        p = _run({**self.base_env}, replay_arrival=400, anchor_arrival=97.5)    # 40 局差 3 局 = 7.5pt
        a = p["结果"]["Continuous-safe_s0_gold"]["anchor"]
        self.assertTrue(a["通过"], a)
        self.assertGreaterEqual(a["容差"], 10.0 - 1e-9)                          # 4 局 × 2.5pt

    def test_unknown_dataset_fails_closed(self):
        _make_ckpt(self.tmp, "Continuous-safe_s0_strided", dataset="strided")
        with self.assertRaises(SystemExit):
            _run({**self.base_env}, replay_arrival=400, anchor_arrival=82.5)
        p = _run({**self.base_env, "REEVAL_ALLOW_UNKNOWN_DATASET": "1", "REEVAL_ANCHOR": "0"},
                 replay_arrival=300)
        self.assertEqual(p["池"]["clean"], 600)                       # 剔不了泄漏 → 裸 600（已显式承认）
        self.assertEqual(p["池"]["训练泄漏"], 0)

    def test_two_arms_share_same_denominator(self):
        """E5(连续)+E6(离散) 同跑：同一 clean 集、同一分母 → 可比。"""
        _make_ckpt(self.tmp, "Continuous-safe_s0_gold", kind="continuous", party="Continuous-safe")
        _make_ckpt(self.tmp, "Discrete-safe_s0_gold", kind="shielded", party="Discrete-safe")
        p = _run({**self.base_env, "REEVAL_ANCHOR": "0"}, replay_arrival=400)
        ns = {k: v["clean"]["n"] for k, v in p["结果"].items()}
        self.assertEqual(set(ns.values()), {577})
        self.assertEqual(len(p["结果"]), 2)
        self.assertEqual((p["已完成"], p["待评"], p["全部完成"]), (2, 0, True))   # 增量落盘的进度字段

    def test_incremental_dump_survives_interruption(self):
        """每评完一个 checkpoint 就落盘 → 中途被杀也留得住已评出的臂（欠费/SIGHUP 咬过两次）。"""
        _make_ckpt(self.tmp, "A_s0_x", kind="continuous", party="Continuous-safe")
        _make_ckpt(self.tmp, "B_s0_x", kind="continuous", party="Continuous-safe")
        boom = {"n": 0}
        orig = globals()["_install_stubs"]                      # 先绑住真身·否则下面覆盖 globals 会自我递归

        def _stubs(**kw):
            R = orig(**kw)
            real_replay = R.replay_eval

            def _explode(*a, **k):                              # 第 2 个 ckpt 评到一半"被杀"
                boom["n"] += 1
                if boom["n"] > 1:
                    raise KeyboardInterrupt("模拟被杀")
                return real_replay(*a, **k)
            R.replay_eval = _explode
            return R
        old = orig
        try:
            globals()["_install_stubs"] = _stubs
            with self.assertRaises(KeyboardInterrupt):
                _run({**self.base_env, "REEVAL_ANCHOR": "0"}, replay_arrival=400)
        finally:
            globals()["_install_stubs"] = old
        with open(self.out, encoding="utf-8") as fh:            # 被杀了·但第 1 个臂的结果已落盘
            p = json.load(fh)
        self.assertEqual((p["已完成"], p["待评"], p["全部完成"]), (1, 1, False))
        self.assertEqual(len(p["结果"]), 1)

    def test_envcfg_guard_blocks_non_golden(self):
        """连续臂 eval env knob 非金标默认（这些 knob 不进 config_sig·会【静默】改数字）→ 必须中止。"""
        _make_ckpt(self.tmp, "Continuous-safe_s0_gold")
        with self.assertRaises(SystemExit):
            _run({**self.base_env}, replay_arrival=400, anchor_arrival=82.5, augment_rho=True)
        p = _run({**self.base_env, "REEVAL_ENVCFG_ACK": "1"},         # 显式承认 → 放行
                 replay_arrival=400, anchor_arrival=82.5, augment_rho=True)
        self.assertEqual(p["结果"]["Continuous-safe_s0_gold"]["clean"]["n"], 577)


class TestTrajCollection(unittest.TestCase):
    """r8 新增（`03` L215-G·user 2026-07-26 拍板走"给 replay_eval 加默认关掉的轨迹开关"这条路）。"""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp = os.path.join(self.tmpdir.name, "checkpoints")
        self.out = os.path.join(self.tmpdir.name, "out.json")
        self.base_env = {"REEVAL_CKDIRS": self.tmp, "REEVAL_MANIFEST_DIRS": _BALANCED,
                         "REEVAL_OUT": self.out, "STEP4E_SDIR": _SCEN, "REEVAL_ANCHOR": "0",
                         "REEVAL_TRAJ_KEYS": "", "REEVAL_TRAJ_OUT": ""}
        _make_ckpt(self.tmp, "Continuous-safe_s0_gold")

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_default_off_is_bit_identical(self):
        """不设 REEVAL_TRAJ_KEYS ⟹ traj_idxs 必须是 None（record_traj 一行都不执行）+ 不产轨迹文件。"""
        p = _run({**self.base_env}, replay_arrival=400)
        import run_step4e as R
        self.assertTrue(all(x is None for x in R._traj_seen), R._traj_seen)
        self.assertFalse(os.path.exists(self.out[:-5] + "_traj.json"))
        self.assertEqual(p["结果"]["Continuous-safe_s0_gold"]["strict"]["n"], 563)

    def test_traj_written_and_keyed_by_pool_key(self):
        """给两个池键 → 轨迹另落一个文件、按 checkpoint 名 → 池键（T-id）组织，且主 json 不被撑大。"""
        mod = importlib.import_module("reeval_official")
        _tr, te = mod.official_split(None)
        k1, k2 = te[0], te[5]
        p = _run({**self.base_env, "REEVAL_TRAJ_KEYS": f"{k1} {k2}"}, replay_arrival=400)
        tp = self.out[:-5] + "_traj.json"
        self.assertTrue(os.path.exists(tp), "开了 REEVAL_TRAJ_KEYS 却没落轨迹文件")
        with open(tp, encoding="utf-8") as fh:
            tj = json.load(fh)
        self.assertEqual(set(tj), {"Continuous-safe_s0_gold"})
        self.assertEqual(set(tj["Continuous-safe_s0_gold"]), {str(k1), str(k2)})
        self.assertIn("ego_x", tj["Continuous-safe_s0_gold"][str(k1)][0])
        self.assertNotIn("traj", json.dumps(p["结果"], ensure_ascii=False))   # 主 json 不带轨迹
        self.assertEqual(p["结果"]["Continuous-safe_s0_gold"]["strict"]["n"], 563)

    def test_bad_traj_key_aborts(self):
        with self.assertRaises(SystemExit) as cm:
            _run({**self.base_env, "REEVAL_TRAJ_KEYS": "999999"}, replay_arrival=400)
        self.assertIn("REEVAL_TRAJ_KEYS", str(cm.exception.code))

    def test_anchor_pass_never_gets_traj_idxs(self):
        """🔴 锚点那趟用的是【另一个池】（自记 40 场景）⟹ 下标含义不同 ⟹ 绝不能把同一组下标传给它。"""
        mod = importlib.import_module("reeval_official")
        _tr, te = mod.official_split(None)
        _run({**self.base_env, "REEVAL_ANCHOR": "1", "REEVAL_TRAJ_KEYS": str(te[0])},
             replay_arrival=400, anchor_arrival=82.5)
        import run_step4e as R
        self.assertEqual(len(R._traj_seen), 2, "应是两趟：锚点 + 正式池")
        self.assertIsNone(R._traj_seen[0], "锚点那趟收到了 traj_idxs → 会记错场景的轨迹")
        self.assertIsNotNone(R._traj_seen[1], "正式池那趟没收到 traj_idxs")

    def test_replay_eval_really_has_the_param(self):
        """契约：真 `run_step4e.replay_eval` 必须真的有 traj_idxs 这个参数（桩不算）。

        本机 import 不了 run_step4e（缺 vesselmodels）⟹ 直接查源码签名，两边任一改动都会在这里炸。
        """
        with open(os.path.join(_CODE, "run_step4e.py"), encoding="utf-8") as fh:
            src = fh.read()
        m = re.search(r"^def replay_eval\(([^)]*)\)", src, re.M)
        self.assertIsNotNone(m, "找不到 replay_eval 定义——签名变了，测试须同步")
        self.assertIn("traj_idxs", m.group(1))
        # 两个调用点（连续臂 evaluate_continuous / 离散臂 evaluate）都必须把它透传下去
        self.assertEqual(len(re.findall(r"traj_idxs=traj_idxs", src)), 2,
                         "replay_eval 里应恰好有 2 处 traj_idxs=traj_idxs（连续臂 + 离散臂）")


class TestYawLowPass(unittest.TestCase):
    """r9 新增（`03` L218）：转向低通滤波（纯评估期治抖·安全关键文件一行不改）。"""

    def setUp(self):
        os.environ["REEVAL_MANIFEST_DIRS"] = _BALANCED
        self.mod = importlib.reload(importlib.import_module("reeval_official"))
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp = os.path.join(self.tmpdir.name, "checkpoints")
        self.out = os.path.join(self.tmpdir.name, "out.json")
        self.base_env = {"REEVAL_CKDIRS": self.tmp, "REEVAL_MANIFEST_DIRS": _BALANCED,
                         "REEVAL_OUT": self.out, "STEP4E_SDIR": _SCEN, "REEVAL_ANCHOR": "0",
                         "REEVAL_TRAJ_KEYS": "", "REEVAL_YAW_LOWPASS": ""}

    def tearDown(self):
        self.tmpdir.cleanup()

    # ── 滤波器本体（纯逻辑·不需要 env） ──
    def test_alpha_one_is_bit_identical(self):
        """🔴 最关键一条：α=1 必须【逐位】等于不滤波——否则"对照组"本身就带偏差，整条曲线没意义。"""
        import numpy as np
        raw = [0.018, -0.012, 0.006, -0.018, 0.0]
        inner = _FakeInnerModel()
        pol = self.mod.YawLowPassPolicy(inner, 1.0)
        got = []
        for w in raw:
            inner.w = w
            u, _ = pol.predict(np.zeros(27, np.float32))
            got.append(float(u[1]))
        self.assertEqual(got, raw, "α=1 竟然改了动作 → 对照组不干净")

    def test_filter_is_convex_combination(self):
        import numpy as np
        inner = _FakeInnerModel(0.018)
        pol = self.mod.YawLowPassPolicy(inner, 0.5)
        u1, _ = pol.predict(np.zeros(27, np.float32))
        self.assertAlmostEqual(float(u1[1]), 0.018)      # 第一步无历史 → 原样
        inner.w = -0.018
        u2, _ = pol.predict(np.zeros(27, np.float32))
        self.assertAlmostEqual(float(u2[1]), 0.5 * 0.018 + 0.5 * (-0.018))   # = 0.0
        inner.w = -0.018
        u3, _ = pol.predict(np.zeros(27, np.float32))
        self.assertAlmostEqual(float(u3[1]), 0.5 * 0.0 + 0.5 * (-0.018))

    def test_stays_in_action_box(self):
        """凸组合 ⟹ 永不越箱（越箱会被 env 夹取、悄悄改变语义）。"""
        import numpy as np
        W_BOX = 0.018
        inner = _FakeInnerModel()
        pol = self.mod.YawLowPassPolicy(inner, 0.3)
        for w in (0.018, -0.018) * 20:
            inner.w = w
            u, _ = pol.predict(np.zeros(27, np.float32))
            self.assertLessEqual(abs(float(u[1])), W_BOX + 1e-12)

    def test_state_reset_per_episode(self):
        """🔴 逐局必须清记忆：不清就把上一局的舵角带进下一局（跨局污染·且聚合上看不出来）。"""
        import numpy as np
        inner = _FakeInnerModel(0.018)
        pol = self.mod.YawLowPassPolicy(inner, 0.5)
        pol.predict(np.zeros(27, np.float32))            # 建立历史 0.018
        pol.bind_env(object())                            # 新的一局
        inner.w = -0.018
        u, _ = pol.predict(np.zeros(27, np.float32))
        self.assertAlmostEqual(float(u[1]), -0.018, msg="bind_env 没清掉上一局的滤波记忆")

    def test_bad_alpha_rejected(self):
        for bad in (0.0, -0.1, 1.5):
            with self.assertRaises(ValueError):
                self.mod.YawLowPassPolicy(_FakeInnerModel(), bad)

    # ── 接线（走 main·桩 replay_eval 记录收到了什么） ──
    def test_default_off_passes_no_wrap(self):
        _make_ckpt(self.tmp, "Continuous-safe_s0_gold")
        p = _run({**self.base_env}, replay_arrival=400)
        import run_step4e as R
        self.assertTrue(all(w is None for w in R._wrap_seen), "不设 REEVAL_YAW_LOWPASS 却传了 policy_wrap")
        self.assertEqual(list(p["结果"]), ["Continuous-safe_s0_gold"], "键名不该带 @lp 后缀")

    def test_sweep_produces_one_entry_per_alpha(self):
        _make_ckpt(self.tmp, "Continuous-safe_s0_gold")
        p = _run({**self.base_env, "REEVAL_YAW_LOWPASS": "1.0,0.5,0.3"}, replay_arrival=400)
        self.assertEqual(sorted(p["结果"]),
                         sorted(["Continuous-safe_s0_gold@lp1", "Continuous-safe_s0_gold@lp0.5",
                                 "Continuous-safe_s0_gold@lp0.3"]))
        self.assertEqual(p["结果"]["Continuous-safe_s0_gold@lp0.5"]["平滑档"],
                         {"mode": "lp", "value": 0.5})
        for k in p["结果"]:
            self.assertEqual(p["结果"][k]["strict"]["n"], 563, k)

    def test_anchor_pass_never_filtered(self):
        """🔴 锚点是拿来比训练记录的 ⟹ 必须【不滤波】，否则一定对不上、还会被判成配置错。"""
        _make_ckpt(self.tmp, "Continuous-safe_s0_gold")
        _run({**self.base_env, "REEVAL_ANCHOR": "1", "REEVAL_YAW_LOWPASS": "0.5"},
             replay_arrival=400, anchor_arrival=82.5)
        import run_step4e as R
        self.assertIsNone(R._wrap_seen[0], "锚点那趟带了滤波 → 必然对不上训练记录、被误判成配置错")
        self.assertIsNotNone(R._wrap_seen[1], "正式池那趟没带滤波")

    def test_discrete_arm_skipped(self):
        """离散臂动作是网格下标 ⟹ 跳过滤波档（而不是静默产出一个"以为滤过"的数）。"""
        _make_ckpt(self.tmp, "Discrete-safe_s0", kind="shielded", party="Discrete-safe")
        p = _run({**self.base_env, "REEVAL_YAW_LOWPASS": "0.5"}, replay_arrival=400)
        self.assertEqual(p["结果"], {}, "离散臂不该产出滤波档")

    def test_replay_eval_really_has_policy_wrap(self):
        """契约：真 `run_step4e.replay_eval` 必须真的有 policy_wrap，且只在连续臂分支施加。"""
        with open(os.path.join(_CODE, "run_step4e.py"), encoding="utf-8") as fh:
            src = fh.read()
        m = re.search(r"^def replay_eval\(([^)]*)\)", src, re.M | re.S)
        self.assertIsNotNone(m)
        self.assertIn("policy_wrap", m.group(1))
        self.assertIn("model = policy_wrap(model)", src)
        # 离散分支必须 fail-fast（不静默忽略）
        self.assertIn("policy_wrap 只支持连续臂", src)


class TestYawSlewLimit(unittest.TestCase):
    """r10 新增（`03` L221）：舵速率限制——第二个平滑旋钮，比低通更有物理依据。"""

    def setUp(self):
        os.environ["REEVAL_MANIFEST_DIRS"] = _BALANCED
        self.mod = importlib.reload(importlib.import_module("reeval_official"))
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp = os.path.join(self.tmpdir.name, "checkpoints")
        self.out = os.path.join(self.tmpdir.name, "out.json")
        self.base_env = {"REEVAL_CKDIRS": self.tmp, "REEVAL_MANIFEST_DIRS": _BALANCED,
                         "REEVAL_OUT": self.out, "STEP4E_SDIR": _SCEN, "REEVAL_ANCHOR": "0",
                         "REEVAL_TRAJ_KEYS": "", "REEVAL_YAW_LOWPASS": "", "REEVAL_YAW_SLEW": ""}

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_frac_one_is_bit_identical(self):
        """🔴 系数=1 必须【逐位】等于不限速（对照组必须干净）。箱宽=2×0.018 ⟹ 一步足以从满左翻满右。"""
        import numpy as np
        raw = [0.018, -0.018, 0.018, -0.006, 0.012]
        inner = _FakeInnerModel(); pol = self.mod.YawSlewLimitPolicy(inner, 1.0)
        got = []
        for w in raw:
            inner.w = w
            u, _ = pol.predict(np.zeros(27, np.float32)); got.append(float(u[1]))
        # 🔴 不许 round：原先用 round(...,12) 会把 1e-18 级的削值掩盖掉（自审发现·`03` L222）
        self.assertEqual(got, raw, "系数=1 竟然改了动作（逐位比对，未取整）")
        self.assertEqual(pol.n_clipped, 0)

    def test_half_box_limits_reversal(self):
        """系数 0.5 ⟹ 每步最多动半个箱（0.018）⟹ 满左到满右要两步。"""
        import numpy as np
        inner = _FakeInnerModel(0.018); pol = self.mod.YawSlewLimitPolicy(inner, 0.5)
        u1, _ = pol.predict(np.zeros(27, np.float32))
        self.assertAlmostEqual(float(u1[1]), 0.018)          # 首步不限
        inner.w = -0.018
        u2, _ = pol.predict(np.zeros(27, np.float32))
        self.assertAlmostEqual(float(u2[1]), 0.018 - 0.018)  # = 0.0（被限速拉住）
        u3, _ = pol.predict(np.zeros(27, np.float32))
        self.assertAlmostEqual(float(u3[1]), -0.018)         # 第二步才到满右
        self.assertEqual(pol.n_clipped, 1)

    def test_never_exceeds_box(self):
        import numpy as np
        inner = _FakeInnerModel(); pol = self.mod.YawSlewLimitPolicy(inner, 0.3)
        for w in (0.018, -0.018) * 25:
            inner.w = w
            u, _ = pol.predict(np.zeros(27, np.float32))
            self.assertLessEqual(abs(float(u[1])), 0.018 + 1e-12)

    def test_state_reset_per_episode(self):
        import numpy as np
        inner = _FakeInnerModel(0.018); pol = self.mod.YawSlewLimitPolicy(inner, 0.25)
        pol.predict(np.zeros(27, np.float32))
        pol.bind_env(object())
        inner.w = -0.018
        u, _ = pol.predict(np.zeros(27, np.float32))
        self.assertAlmostEqual(float(u[1]), -0.018, msg="bind_env 没清掉上一局的舵角")

    def test_bad_frac_rejected(self):
        for bad in (0.0, -0.2, 1.4):
            with self.assertRaises(ValueError):
                self.mod.YawSlewLimitPolicy(_FakeInnerModel(), bad)

    def test_two_knobs_in_one_run(self):
        """两个旋钮可以同一趟一起扫 ⟹ 一次拿两条权衡曲线（键名分别带 @lp / @sl）。"""
        _make_ckpt(self.tmp, "Continuous-safe_s0_gold")
        p = _run({**self.base_env, "REEVAL_YAW_LOWPASS": "1.0,0.5",
                  "REEVAL_YAW_SLEW": "1.0,0.5"}, replay_arrival=400)
        self.assertEqual(sorted(p["结果"]), sorted([
            "Continuous-safe_s0_gold@lp1", "Continuous-safe_s0_gold@lp0.5",
            "Continuous-safe_s0_gold@sl1", "Continuous-safe_s0_gold@sl0.5"]))
        self.assertEqual(p["结果"]["Continuous-safe_s0_gold@sl0.5"]["平滑档"],
                         {"mode": "sl", "value": 0.5})
        for k in p["结果"]:
            self.assertEqual(p["结果"][k]["strict"]["n"], 563, k)

    def test_discrete_arm_skipped(self):
        _make_ckpt(self.tmp, "Discrete-safe_s0", kind="shielded", party="Discrete-safe")
        p = _run({**self.base_env, "REEVAL_YAW_SLEW": "0.5"}, replay_arrival=400)
        self.assertEqual(p["结果"], {})


class TestSmoothRefIsAppliedAction(unittest.TestCase):
    """r11 自审修（`03` L222）：参考舵位必须取【盾之后真正施加】的 ω，不是我们发出的指令。"""

    class _Env:
        """最小站位 env：`env.env.last_action` = 盾之后真正施加的 (a, ω)。"""
        def __init__(self, applied_w=0.0):
            self.env = types.SimpleNamespace(last_action=[0.0, float(applied_w)])
        def set_applied(self, w):
            self.env.last_action = [0.0, float(w)]

    def setUp(self):
        os.environ["REEVAL_MANIFEST_DIRS"] = _BALANCED
        self.mod = importlib.reload(importlib.import_module("reeval_official"))

    def test_slew_limits_against_applied_not_commanded(self):
        """盾把动作改写成 −0.012 后，限速必须相对【实际施加的 −0.012】，而不是相对我们上次的指令。

        这里刻意用**箱内**的施加值，好把"参考取施加值"这件事与"越箱要夹回"分开测
        （越箱那条由 `test_output_never_leaves_rl_box_even_after_shield_override` 专门盯）。
        """
        import numpy as np
        env = self._Env(applied_w=-0.012)         # 盾把上一步改写成了 −0.012（箱内）
        inner = _FakeInnerModel(0.018)            # 策略这一步想要满左 +0.018
        pol = self.mod.YawSlewLimitPolicy(inner, 0.25)   # Δmax = 0.25×0.036 = 0.009
        pol.bind_env(env)
        u, _ = pol.predict(np.zeros(27, np.float32))
        self.assertAlmostEqual(float(u[1]), -0.012 + 0.009, places=12,
                               msg="限速没有以【实际施加值】为参考")
        self.assertGreaterEqual(pol.n_ref_from_env, 1, "根本没读到 env 里的施加值")
        self.assertEqual(pol.n_ref_clamped, 0, "箱内的施加值不该被夹")

        # 反证：若参考错用了"我们自己上一步的输出"，第一步没有历史 ⟹ 会原样放行 +0.018
        env2 = self._Env(applied_w=-0.012)
        pol2 = self.mod.YawSlewLimitPolicy(_FakeInnerModel(0.018), 0.25)
        pol2.bind_env(env2)
        self.assertNotAlmostEqual(float(pol2.predict(np.zeros(27, np.float32))[0][1]), 0.018,
                                  msg="看起来仍在用指令值当参考（首步原样放行了）")

    def test_lowpass_uses_applied_reference(self):
        import numpy as np
        env = self._Env(applied_w=0.0)
        inner = _FakeInnerModel(0.018)
        pol = self.mod.YawLowPassPolicy(inner, 0.5)
        pol.bind_env(env)
        u, _ = pol.predict(np.zeros(27, np.float32))
        # 首步 env 的 last_action 是 [0,0]（reset 时舵居中）⟹ 相对 0 滤波 = 物理正确
        self.assertAlmostEqual(float(u[1]), 0.5 * 0.0 + 0.5 * 0.018, places=12)

    def test_noop_settings_still_bit_identical_with_env(self):
        """🔴 关键：绑了 env、参考舵位改了之后，『不施加』那两档仍必须逐位不变。"""
        import numpy as np
        for mk in (lambda m: self.mod.YawLowPassPolicy(m, 1.0),
                   lambda m: self.mod.YawSlewLimitPolicy(m, 1.0)):
            env = self._Env(0.0)
            inner = _FakeInnerModel(); pol = mk(inner); pol.bind_env(env)
            for w in (0.018, -0.018, 0.006, -0.018, 0.018, 0.0):
                inner.w = w
                u, _ = pol.predict(np.zeros(27, np.float32))
                env.set_applied(float(u[1]))       # 模拟盾透传：施加值 = 指令值
                self.assertEqual(float(u[1]), w, f"{pol.__class__.__name__} 不施加档竟然改了动作")

    def test_falls_back_when_env_absent(self):
        """未绑 env（或 env 没有那个属性）→ 回退成『首步不限』，不炸。"""
        import numpy as np
        inner = _FakeInnerModel(0.018)
        pol = self.mod.YawSlewLimitPolicy(inner, 0.25)
        u, _ = pol.predict(np.zeros(27, np.float32))
        self.assertAlmostEqual(float(u[1]), 0.018)
        self.assertEqual(pol.n_ref_from_env, 0)
        pol.bind_env(object())                     # 有 env 但没有 .env.last_action
        inner.w = -0.018
        u2, _ = pol.predict(np.zeros(27, np.float32))
        self.assertAlmostEqual(float(u2[1]), -0.018)
        self.assertEqual(pol.n_ref_from_env, 0)

    def test_output_never_leaves_rl_box_even_after_shield_override(self):
        """🔴🔴 不可协商的不变量（自审第二轮抓出的真 bug·`03` L222-B）：

        紧急控制器/盾走**物理满程 ±0.03**，超出 RL 箱 ±0.018。若拿它当参考而不夹回箱内，
        窗口整段可能落在箱外 ⟹ 我们的指令越箱 ⟹ **偷用了超出四臂对照口径的操作权限**。
        实测（修前）：低通 α=0.5 输出 0.024、限速系数 0.25 输出 0.021，都越箱。
        """
        import numpy as np
        W = 0.018
        for mk in [lambda m: self.mod.YawLowPassPolicy(m, a) for a in (0.7, 0.5, 0.3)] + \
                  [lambda m: self.mod.YawSlewLimitPolicy(m, f) for f in (0.5, 0.33, 0.25)]:
            for applied in (0.03, -0.03, 0.025, -0.025):       # 盾/紧急控制器可能施加的越箱值
                for want in (0.018, -0.018, 0.0):
                    env = self._Env(applied)
                    pol = mk(_FakeInnerModel(want)); pol.bind_env(env)
                    w = float(pol.predict(np.zeros(27, np.float32))[0][1])
                    self.assertLessEqual(abs(w), W + 1e-12,
                                         f"{pol.__class__.__name__} 在施加值 {applied} 下输出 {w} 越出 RL 箱")
                    self.assertGreaterEqual(pol.n_ref_clamped, 1, "越箱的参考没有被夹回")

    def test_getattr_guard_no_recursion(self):
        """🔴 半构造对象访问属性必须给 AttributeError，不能无限递归（自审实测过原实现会 RecursionError）。"""
        for cls in (self.mod.YawLowPassPolicy, self.mod.YawSlewLimitPolicy):
            obj = cls.__new__(cls)
            with self.assertRaises(AttributeError):
                obj.anything
            with self.assertRaises(AttributeError):
                obj.model

    def test_w_box_matches_truth_or_aborts(self):
        """箱常量必须与真相源一致；显式传一个错的箱 → 本机导不到真相源时允许，导得到时必须中止。"""
        w, from_truth = self.mod._resolve_w_box(None)
        self.assertAlmostEqual(w, 0.018)
        if from_truth:
            with self.assertRaises(SystemExit):
                self.mod._resolve_w_box(0.05)


class TestAnchorSummaryDedup(unittest.TestCase):
    """r11 自审修（`03` L222）：扫平滑档时，锚点汇总必须按 checkpoint 去重。"""

    def setUp(self):
        os.environ["REEVAL_MANIFEST_DIRS"] = _BALANCED
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp = os.path.join(self.tmpdir.name, "checkpoints")
        self.out = os.path.join(self.tmpdir.name, "out.json")
        self.base_env = {"REEVAL_CKDIRS": self.tmp, "REEVAL_MANIFEST_DIRS": _BALANCED,
                         "REEVAL_OUT": self.out, "STEP4E_SDIR": _SCEN, "REEVAL_ANCHOR": "1",
                         "REEVAL_TRAJ_KEYS": "", "REEVAL_YAW_LOWPASS": "", "REEVAL_YAW_SLEW": ""}

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_one_ckpt_four_levels_counts_once(self):
        _make_ckpt(self.tmp, "Continuous-safe_s0_gold")
        p = _run({**self.base_env, "REEVAL_YAW_LOWPASS": "1.0,0.5,0.3", "REEVAL_YAW_SLEW": "0.5"},
                 replay_arrival=400, anchor_arrival=82.5)
        self.assertEqual(len([k for k in p["结果"] if not k.startswith("_")]), 4, "应产出 4 档")
        self.assertEqual(p["结果"]["_锚点汇总"]["n"], 1,
                         "锚点只做过 1 次，汇总却计了多次 ⟹ 群体判据的样本量被放大（L212-C 那道闸被削弱）")

    def test_two_ckpts_counts_twice(self):
        _make_ckpt(self.tmp, "Continuous-safe_s0_gold")
        _make_ckpt(self.tmp, "Continuous-safe_s1_gold", seed=1)
        p = _run({**self.base_env, "REEVAL_YAW_LOWPASS": "1.0,0.5"},
                 replay_arrival=400, anchor_arrival=82.5)
        self.assertEqual(p["结果"]["_锚点汇总"]["n"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
