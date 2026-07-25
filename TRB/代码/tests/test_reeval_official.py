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
                 "violations": 0.0, "emergency_pct": 3.0, "in_box_steps": (5 if i < n_reached else 0)}
                for i in range(n)]

    def _replay(base, kind, weight, pool, *, continuous_algo=None, return_per=False):
        n = len(pool)
        if n == n_anchor and anchor_arrival is not None:                       # 锚点池
            k = round(anchor_arrival / 100.0 * n)
            per = _fake_per(n, k)
        else:
            k = replay_arrival if replay_arrival is not None else n // 2
            per = _fake_per(n, k)
        agg = {"n": n, "到达率%": 100.0 * sum(p["reached"] for p in per) / n}
        return (agg, per) if return_per else agg

    R.replay_eval = _replay

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
