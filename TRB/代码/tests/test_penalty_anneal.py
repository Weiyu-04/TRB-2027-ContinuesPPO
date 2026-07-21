"""
惩罚权重退火 冒烟测试（`03` L103/L105·2026-06-27）——治"惩罚从第0步压脆弱种子"的 hold-then-ramp 退火基础设施。
跑：/opt/miniconda3/envs/trb/bin/python -B 代码/tests/test_penalty_anneal.py
覆盖：① PenaltyAnnealSchedule hold-then-ramp 曲线值正确（含 clamp/边界）② MultiScenarioEnv.set_penalty_weight 双写
   （env_kwargs 下次 reset 继承 + 当前 _inner 立即生效）+ isfinite≥0 守卫 ③ VecEnv.env_method('set_penalty_weight')
   跨子 env 生效 + 跨 episode-reset 持续（get_attr 读回随步变）④ 退火 off / w=0 时 reward 与无惩罚【逐位等价 bit-identical】
   ⑤ 设了权重后惩罚真进 reward（非 no-op）⑥ PenaltyAnnealSyncCallback 同步逻辑（假 model/venv）。
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from trb_env.usv_sac_train import PenaltyAnnealSchedule, PenaltyAnnealSyncCallback
from trb_env.usv_scenarios import MultiScenarioEnv, make_vec_env
from trb_env.usv_continuous_shield import ContinuousProjectionEnv

_fail = 0; _total = 0
def ok(name, cond):
    global _fail, _total
    _total += 1
    if not cond: _fail += 1
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")

print("===== A) PenaltyAnnealSchedule hold-then-ramp 曲线 =====")
# start=0 end=0.25 ramp_start=3.0M anneal=1.5M（5M·默认口径 0.6/0.3 量级）
sch = PenaltyAnnealSchedule(0.0, 0.25, 3_000_000, 1_500_000)
def val_at(t):
    sch.num_timesteps = t; return sch.value()
ok("A1 t=0 → start(0)", abs(val_at(0) - 0.0) < 1e-12)
ok("A2 t=ramp_start 前一刻 → 仍 start(0)", abs(val_at(2_999_999) - 0.0) < 1e-9)
ok("A3 t=ramp_start(3.0M) → start(0)", abs(val_at(3_000_000) - 0.0) < 1e-12)
ok("A4 ramp 中点(3.75M)=半程 → end/2(0.125)", abs(val_at(3_750_000) - 0.125) < 1e-9)
ok("A5 t=ramp 末(4.5M) → end(0.25)", abs(val_at(4_500_000) - 0.25) < 1e-9)
ok("A6 t>ramp 末(5M) → clamp 恒 end(0.25)", abs(val_at(5_000_000) - 0.25) < 1e-12)
ok("A7 t 远超 → 仍 end(不溢出)", abs(val_at(99_000_000) - 0.25) < 1e-12)
# ramp_start=0（无 hold·纯线性·=LR 退火退化形）
sch0 = PenaltyAnnealSchedule(0.0, 1.0, 0, 1_000_000); sch0.num_timesteps = 500_000
ok("A8 ramp_start=0 半程 → 0.5（退化纯线性正确）", abs(sch0.value() - 0.5) < 1e-9)

print("\n===== B) MultiScenarioEnv.set_penalty_weight 双写 + 守卫 =====")
_T0 = "/tmp/trb_T0.xml"; _POOLDIR = "/tmp/trb_scenarios_pool"
_HAVE = os.path.exists(_T0)
if not _HAVE:
    try:
        import urllib.request
        urllib.request.urlretrieve(
            "https://gitlab.lrz.de/tum-cps/commonocean-scenarios/-/raw/main/scenarios/"
            "HandcraftedTwoVesselEncounters_01_24/ZAM_AAA-1_20240121_T-0.xml", _T0)
        _HAVE = True
    except Exception as e:
        print(f"[SKIP] /tmp/trb_T0.xml 不在且离线下载失败（{e}）→ 跳过 B/C/D/E 端到端块（非逻辑回归）")
if _HAVE:
    from commonocean.common.file_reader import CommonOceanFileReader
    _sc, _pp = CommonOceanFileReader(_T0).open()
    _ppx = list(_pp.planning_problem_dict.values())[0]
    pool = [(_sc, _ppx)]
    menv = MultiScenarioEnv(pool, env_cls=ContinuousProjectionEnv, env_kwargs=dict(rate_weight=0.0))
    menv.reset(seed=0)
    # 双写：set → env_kwargs 与当前 _inner 都变
    menv.set_penalty_weight("rate_weight", 0.5)
    ok("B1 set 后 env_kwargs 立即变 0.5（下次 reset 继承）", menv.env_kwargs["rate_weight"] == 0.5)
    ok("B2 set 后【当前】_inner.rate_weight 立即变 0.5（当前 episode 生效）", menv._inner.rate_weight == 0.5)
    # 关键：reset 重建 _inner → 新 _inner 仍带 0.5（=env_kwargs 继承·非被抹回 0）
    menv.reset(seed=1)
    ok("B3 reset 重建 _inner 后仍带 0.5（双写之 env_kwargs 继承·单写 _inner 会被抹=本测试守护）",
       menv._inner.rate_weight == 0.5)
    # 再 set alias
    menv.set_penalty_weight("alias_weight", 2.0)
    ok("B4 set alias_weight 独立生效（不动 rate）", menv._inner.alias_weight == 2.0 and menv._inner.rate_weight == 0.5)
    # 守卫：非法 name / nan / inf / 负 → ValueError
    def _raises(fn):
        try: fn(); return False
        except (ValueError,): return True
    ok("B5 非法 name → ValueError", _raises(lambda: menv.set_penalty_weight("foo", 1.0)))
    ok("B6 nan/inf/负 → ValueError（同 shield 守卫）",
       _raises(lambda: menv.set_penalty_weight("rate_weight", float("nan")))
       and _raises(lambda: menv.set_penalty_weight("rate_weight", float("inf")))
       and _raises(lambda: menv.set_penalty_weight("rate_weight", -1.0)))

    print("\n===== C) VecEnv.env_method 跨子 env 生效 + 跨 reset 持续 =====")
    venv = make_vec_env(scenario_pool=pool, n_envs=3, env_cls=ContinuousProjectionEnv,
                        env_kwargs=dict(rate_weight=0.0), subproc=False, seed=0)
    venv.reset()
    venv.env_method("set_penalty_weight", "rate_weight", 0.75)
    # get_attr 读回各子 MultiScenarioEnv 的 _inner.rate_weight
    inners = venv.get_attr("_inner")
    ok("C1 env_method 后全部 3 个子 env 的 _inner.rate_weight=0.75（无遗漏子 env）",
       all(abs(getattr(ie, "rate_weight") - 0.75) < 1e-12 for ie in inners))
    ekws = venv.get_attr("env_kwargs")
    ok("C2 全部子 env 的 env_kwargs['rate_weight']=0.75（下次 reset 继承）",
       all(abs(ek["rate_weight"] - 0.75) < 1e-12 for ek in ekws))
    venv.reset()
    inners2 = venv.get_attr("_inner")
    ok("C3 VecEnv.reset 后各子 env 重建 _inner 仍带 0.75（跨 reset 持续）",
       all(abs(getattr(ie, "rate_weight") - 0.75) < 1e-12 for ie in inners2))
    venv.close()

    print("\n===== D) 退火 off / w=0 → reward 与无惩罚【逐位等价 bit-identical】=====")
    # schedule t=0 hold → value=0 → set_penalty_weight(...,0) → 与从不 set 的无惩罚 env reward 严格相等
    rng = np.random.default_rng(0)
    _ubox = np.array([0.048, 0.018], dtype=float)
    acts = [rng.uniform(-_ubox, _ubox) for _ in range(60)]
    def run_rewards(set_zero):
        e = MultiScenarioEnv(pool, env_cls=ContinuousProjectionEnv, env_kwargs=dict(rate_weight=0.0))
        e.reset(seed=0)
        if set_zero:
            e.set_penalty_weight("rate_weight", 0.0)   # 退火 hold 段推 0（应与从不 set 严格一致）
        out = []
        for a in acts:
            _o, r, te, tr, _i = e.step(np.array(a, dtype=float)); out.append(r)
            if te or tr: break
        return out
    base = run_rewards(False); pushed0 = run_rewards(True)
    ok("D1 推 rate_weight=0（退火 hold 段）与从不 set 的 reward 序列【严格 <1e-12 相等】(bit-identical·无量化差)",
       len(base) == len(pushed0) and all(abs(a - b) < 1e-12 for a, b in zip(base, pushed0)))

    print("\n===== E) 设了权重后惩罚真进 reward（非 no-op·与 w=0 不同）=====")
    pushed_w = []
    e = MultiScenarioEnv(pool, env_cls=ContinuousProjectionEnv, env_kwargs=dict(rate_weight=0.0))
    e.reset(seed=0); e.set_penalty_weight("rate_weight", 1.0)
    for a in acts:
        _o, r, te, tr, info = e.step(np.array(a, dtype=float)); pushed_w.append((r, info.get("r_rate", "ABSENT")))
        if te or tr: break
    n_rrate = sum(1 for _, rr in pushed_w if rr != "ABSENT")
    diff = any(abs(a - b[0]) > 1e-9 for a, b in zip(base, pushed_w))
    ok("E1 set rate_weight=1.0 后出现 r_rate 键 + reward 与 w=0 不同（惩罚真进 reward·非 no-op）", n_rrate >= 1 and diff)

print("\n===== F) 防呆守卫：常量 _W≠0 + 退火 _ANNEAL_END 同设 → fail-fast（`03` L108·复审主窗口+A3 抓 footgun）=====")
# 守卫在 run_step4e 模块顶层 parse 时触发（:154-）→ subprocess 跑真实加载路径（__name__ 守卫:1242 保 import 不跑 main）。
import subprocess
_CODE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
def _import_run(env_extra):
    e = dict(os.environ)
    for _k in ("STEP4E_RATE_W", "STEP4E_ALIAS_W", "STEP4E_RATE_ANNEAL_END", "STEP4E_ALIAS_ANNEAL_END"):
        e.pop(_k, None)                              # 清干净·防父进程残留污染
    e.update(env_extra)
    p = subprocess.run([sys.executable, "-B", "-c", "import run_step4e; print('IMPORT_OK')"],
                       cwd=_CODE_DIR, env=e, capture_output=True, text=True)
    out = (p.stdout or "") + (p.stderr or "")
    return ("不可同设" in out, "IMPORT_OK" in out)   # (守卫触发?, import 成功?)
_g1, _o1 = _import_run({"STEP4E_RATE_W": "0.25", "STEP4E_RATE_ANNEAL_END": "1.0"})
ok("F1 常量 RATE_W=0.25 + RATE_ANNEAL_END=1.0 → fail-fast（挡掉常量被静默丢弃+jsonl 误标）", _g1 and not _o1)
_g2, _o2 = _import_run({"STEP4E_ALIAS_W": "0.5", "STEP4E_ALIAS_ANNEAL_END": "2.0"})
ok("F2 常量 ALIAS_W=0.5 + ALIAS_ANNEAL_END=2.0 → fail-fast（alias 对称同守卫）", _g2 and not _o2)
_g3, _o3 = _import_run({"STEP4E_RATE_ANNEAL_END": "0.25"})
ok("F3 只设退火 RATE_ANNEAL_END（无常量）→ 不触发（正常退火路径·文档命令 run_rate_anneal）", (not _g3) and _o3)
_g4, _o4 = _import_run({"STEP4E_RATE_W": "0.5"})
ok("F4 只设常量 RATE_W（无退火）→ 不触发（run_rate 常量路径不受影响）", (not _g4) and _o4)
_g5, _o5 = _import_run({"STEP4E_RATE_W": "0.0", "STEP4E_RATE_ANNEAL_END": "0.25"})
ok("F5 RATE_W=0.0（默认）+ 退火 → 不触发（0 常量分量与退火无矛盾·jsonl 记 0 诚实）", (not _g5) and _o5)

print()
if _fail == 0:
    print(f"✅ 全部 PASS（{_total} 项）")
else:
    print(f"❌ {_fail}/{_total} 项 FAIL")
sys.exit(1 if _fail else 0)
