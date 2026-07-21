#!/usr/bin/env python3
"""STEP4E_AUGMENT_RHO / augment_rho 旋钮测试（腿1 态势感知观测增广·`03` L150/L152）。

背景：把 shield 层态势 ρ（one-hot 6）+ give_way_dir（1）拼进连续臂 obs 尾部（27→34）·让策略看见此刻态势
（治抖/治违规·非治崩[15-35%·L150]）。默认关=27维=bit-identical·仅连续臂·内层 USVEnv 27维一字不动=离散臂忠实。

A0-A6 门（对齐蓝图·全零算力·烧卡前必绿）：
  A0 默认关 bit-identical：observation_space==Box(27)·_augment(off)=pass-through 恒等（连 rho/gwd 有值也不改）。
  A1 形状（开）：observation_space==Box(34)·dtype 对齐·reset 与 step 都 34维（挡 reset 坑）·reset 首帧尾部=NO_CONFLICT one-hot(置bit0·非全零)+gw0。
  A2 ρ 穿对（索引映射·L151 真静默变体守卫）：逐步 argmax(obs[27:33])==info['rho_acting'] 且 obs[33]==map(give_way_dir)·真 one-hot(sum1·{0,1})。
  A3 VecNorm 形状（L151 最高危坑）：maker(augment_rho=True) → venv.obs_rms.mean.shape==(34,)·默认→(27,)·SB3 policy 网据 34 自动 sizing（建模不崩即证）。
  A4 四方隔离：config_conflict augment on/off 混写→冲突·同→无·旧记录/离散无字段→不误冲突；无盾+augment→import 期 fail-fast（子进程）。
  A5 续训跨边界：config_conflict 拦 augment-on 续 augment-off TAG（=resume 跨 augment 边界 obs 维度不兼容的实际守护）。
  A6 give_way_dir 映射：从 Python None 取 0（不 KeyError）·'left'→−1·'right'→+1。

运行：cd 代码 && /opt/miniconda3/envs/trb/bin/python -B tests/test_augment_rho_knob.py
"""
import os
import sys
import inspect
import subprocess

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import run_step4e as S
from trb_env.usv_continuous_shield import ContinuousProjectionEnv, _N_RHO_STATES
from trb_env.usv_colregs import RHO_NO_CONFLICT
from trb_env.usv_sac_train import make_continuous_safe_model, make_continuous_safe_ppo_model

N_FAIL = 0
_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
_BASE, _NRHO = 27, _N_RHO_STATES          # 内层 obs 维·ρ one-hot 维（6）
_AUG = _BASE + _NRHO + 1                   # = 34
GW_MAP = {"left": -1.0, "none": 0.0, "right": 1.0}


def chk(cond, msg):
    global N_FAIL
    print(("  ✅ " if cond else "  ❌ ") + msg)
    if not cond:
        N_FAIL += 1


# ================= 纯逻辑（无需场景池）=================
print("A6 give_way_dir 映射（从 None 取 0·不 KeyError）")
_m = {"left": -1.0, "right": 1.0}
chk(_m.get(None, 0.0) == 0.0, "A6 None → 0.0（.get 默认·代码从不吐 'none' 字符串）")
chk(_m.get("left", 0.0) == -1.0 and _m.get("right", 0.0) == 1.0, "A6 'left'→−1 / 'right'→+1")

print("A4/A5 config_conflict augment_rho 混配检测（纯函数）")
_b = {"steps": 100, "n_total": 200}       # 缺字段→.get 默认=cur 默认（隔离 augment）
chk(S.config_conflict([{**_b, "augment_rho": False}], 100, 200) == set(),
    "A4 augment 都 off·cur off → 无冲突")
chk(bool(S.config_conflict([{**_b, "augment_rho": True}], 100, 200)),
    "A4 记录 augment=True·cur off(默认) → 冲突（防 27/34维混进同一 jsonl）")
chk(S.config_conflict([{**_b, "augment_rho": True}], 100, 200, augment_rho=True) == set(),
    "A4/A5 记录 augment=True·cur=True → 无冲突（同增广正常续跑）")
chk(S.config_conflict([_b], 100, 200) == set(),
    "A4 旧记录/离散无 augment_rho 字段·cur off → 不误冲突（向后兼容·归 False=关隐含）")
chk(bool(S.config_conflict([{**_b, "augment_rho": False}], 100, 200, augment_rho=True)),
    "A5 记录 augment=False·cur=True → 冲突（拦 resume 跨 augment 边界·27/34维不兼容）")

print("A4 import-time fail-fast（子进程·env-var 行为）")


def _imp(env):
    e = dict(os.environ); e.update(env)
    return subprocess.run(
        [sys.executable, "-B", "-c",
         "import sys;sys.path.insert(0,'.');import run_step4e as S;print('AUG', S._AUGMENT_RHO)"],
        cwd=_ROOT, env=e, capture_output=True, text=True)


r = _imp({})
chk(r.returncode == 0 and "AUG False" in r.stdout, "A4 默认（未设）→ _AUGMENT_RHO False（bit-identical）")
r = _imp({"STEP4E_AUGMENT_RHO": "1"})
chk(r.returncode == 0 and "AUG True" in r.stdout, "A4 STEP4E_AUGMENT_RHO=1（盾默认开）→ True·不误拦")
r = _imp({"STEP4E_AUGMENT_RHO": "0"})
chk(r.returncode == 0 and "AUG False" in r.stdout, "A4 STEP4E_AUGMENT_RHO=0 → False")
r = _imp({"STEP4E_AUGMENT_RHO": "1", "STEP4E_CONTINUOUS_SHIELD": "0"})
chk(r.returncode != 0 and ("AUGMENT_RHO" in (r.stdout + r.stderr) or "无盾" in (r.stdout + r.stderr)),
    "A4 无盾(SHIELD=0)+augment → fail-fast（ρ 恒 NO_CONFLICT=常数零信息·且破 obs 维度平价）")

print("A0 maker 签名含 augment_rho + 默认 bit-identical（_augment pass-through）")
for f in (make_continuous_safe_model, make_continuous_safe_ppo_model):
    chk("augment_rho" in inspect.signature(f).parameters, f"A0 {f.__name__} 签名含 augment_rho")

# ================= 需场景池 =================
_POOL = "/private/tmp/trb_scenarios_pool"
_PATHS = (sorted(os.path.join(_POOL, f) for f in os.listdir(_POOL) if f.endswith(".xml"))
          if os.path.isdir(_POOL) else [])
if not _PATHS:
    print(f"[SKIP] 场景池 {_POOL} 不存在 → 跳 A0(env)/A1/A2/A3【实构】断言。⚠️ 缺池时【增广形状/ρ穿对/VecNorm 形状/bit-identical 未被实测】"
          f"（签名+纯函数抓不住运行期 silent no-op·绿≠完整通过）→ 有池环境（trb 服务器/本机）必跑。")
else:
    from commonocean.common.file_reader import CommonOceanFileReader

    def _load(p):
        sc, pp = CommonOceanFileReader(p).open()
        return sc, list(pp.planning_problem_dict.values())[0]

    sc0, pp0 = _load(_PATHS[0])

    # ---- A0 默认关 bit-identical ----
    print("A0 默认关 bit-identical（observation_space + _augment 恒等）")
    e_off = ContinuousProjectionEnv(sc0, pp0)                     # 默认 augment_rho=False
    chk(e_off.observation_space.shape == (_BASE,), f"A0 默认 observation_space={e_off.observation_space.shape}==(27,)")
    _raw = np.arange(_BASE, dtype=e_off.observation_space.dtype)
    _pt = e_off._augment(_raw, 3, "left")                         # 即便 rho=3/gwd='left'·关时也【恒等 pass-through】
    chk(_pt is _raw, "A0 _augment(off) 恒等 pass-through（返回同一对象·连 rho/gwd 有值也不改=bit-identical 命门）")
    o_off, _ = e_off.reset(seed=0)
    chk(o_off.shape == (_BASE,), "A0 默认 reset obs 27维")

    # ---- A1 形状（开）+ reset 首帧 one-hot ----
    print("A1 形状（开）+ reset 首帧 NO_CONFLICT one-hot")
    e_on = ContinuousProjectionEnv(sc0, pp0, augment_rho=True)
    chk(e_on.observation_space.shape == (_AUG,) == (34,), f"A1 augment observation_space={e_on.observation_space.shape}==(34,)")
    chk(_N_RHO_STATES == 6, f"A1 _N_RHO_STATES={_N_RHO_STATES}==6（NO_CONFLICT..EMERGENCY）")
    o_on, _ = e_on.reset(seed=0)
    chk(o_on.shape == (_AUG,), "A1 reset obs 34维")
    chk(o_on.dtype == e_on.observation_space.dtype == o_off.dtype, f"A1 dtype 对齐 obs={o_on.dtype} space={e_on.observation_space.dtype}")
    chk(np.array_equal(o_off, o_on[:_BASE]), "A1 增广不改内层 27维（离散臂忠实）")
    _tail = o_on[_BASE:]
    _exp = np.zeros(_NRHO + 1); _exp[RHO_NO_CONFLICT] = 1.0       # bit0=1·其余0·gw(末位)=0
    chk(np.array_equal(_tail, _exp), f"A1 reset 首帧尾部={_tail.tolist()}=NO_CONFLICT one-hot(置bit0·非全零)+gw0（reset 拼接钩子·L151 坑）")
    # step 也 34 维（挡"只在 step 拼、reset 忘拼 or 反之"的维度不一致）
    o_s, _, _, _, _ = e_on.step(e_on.action_space.sample())
    chk(o_s.shape == (_AUG,), "A1 step obs 也 34维（reset 与 step 同维·挡 reset 坑）")

    # ---- A6b 代码侧 give_way_dir map 三键直验（AC2 MEDIUM：'left' 在池中罕见[benchmark-lacks-overtaking]·A2 只跑到 right/none·A6 字面字典对代码 map 空洞）----
    print("A6b 代码侧 give_way_dir map 三键直验（不靠场景覆盖）")
    _z = np.zeros(_BASE, dtype=e_on.observation_space.dtype)
    chk(e_on._augment(_z, RHO_NO_CONFLICT, "left")[_BASE + _NRHO] == -1.0, "A6b 代码 map 'left'→−1（直验·池无左舷让路也守）")
    chk(e_on._augment(_z, RHO_NO_CONFLICT, "right")[_BASE + _NRHO] == 1.0, "A6b 代码 map 'right'→+1")
    chk(e_on._augment(_z, RHO_NO_CONFLICT, None)[_BASE + _NRHO] == 0.0, "A6b 代码 map None→0（不 KeyError）")

    # ---- A2 ρ 穿对（索引映射不变式·多场景×多步）----
    print("A2 ρ 穿对不变式（argmax(obs[27:33])==rho_acting·obs[33]==map(gwd)·真 one-hot）")
    _nstep = _nrho = _ngw = 0; _seen_r = set(); _seen_g = set(); _ok = True
    for p in _PATHS[:60]:
        sc, pp = _load(p)
        env = ContinuousProjectionEnv(sc, pp, augment_rho=True)
        obs, info = env.reset(seed=0)
        for _ in range(25):
            obs, _, term, trunc, info = env.step(env.action_space.sample())
            _nstep += 1
            oh = obs[_BASE:_BASE + _NRHO]
            if int(np.argmax(oh)) != int(info["rho_acting"]): _ok = False
            if not (oh.sum() == 1.0 and set(np.unique(oh)) <= {0.0, 1.0}): _ok = False
            gwd = info["give_way_dir"]; key = gwd if gwd is not None else "none"
            if obs[_BASE + _NRHO] != GW_MAP[key]: _ok = False
            _seen_r.add(int(info["rho_acting"])); _seen_g.add(key)
            if int(info["rho_acting"]) != RHO_NO_CONFLICT: _nrho += 1
            if gwd is not None: _ngw += 1
            if term or trunc: break
    chk(_ok, f"A2 {_nstep} 步 ρ穿对不变式全过（见 ρ={sorted(_seen_r)} gw={sorted(_seen_g)}·ρ≠0 步={_nrho}·gw≠None 步={_ngw}）")
    chk(_nrho > 0, f"A2 载荷非空：真见过 ρ≠0（{_nrho} 步·否则 one-hot 恒 bit0=空洞验）")

    # ---- A3 VecNorm 形状（L151 最高危坑）+ SB3 policy 34-sizing ----
    print("A3 VecNorm obs_rms 形状 + SB3 policy 34-sizing（maker 实构）")
    _SK = dict(buffer_size=2000, learning_starts=10, batch_size=32, subproc=False)
    for name, mk, kw in [("SAC", make_continuous_safe_model, _SK),
                         ("PPO", make_continuous_safe_ppo_model, dict(subproc=False))]:
        m, v = mk(paths=_PATHS[:2], n_envs=1, seed=0, augment_rho=True, **kw)   # 建模不崩=SB3 据 34 sizing 成功
        chk(tuple(v.obs_rms.mean.shape) == (_AUG,), f"A3-{name} augment=True → venv.obs_rms.mean.shape={tuple(v.obs_rms.mean.shape)}==(34,)（VecNorm 按 34 归一化·非静默 27）")
        chk(tuple(v.observation_space.shape) == (_AUG,), f"A3-{name} venv.observation_space=={tuple(v.observation_space.shape)}==(34,)")
        v.close()
        m, v = mk(paths=_PATHS[:2], n_envs=1, seed=0, **kw)                     # 默认 → 27（bit-identical）
        chk(tuple(v.obs_rms.mean.shape) == (_BASE,), f"A3-{name} 默认 → venv.obs_rms.mean.shape==(27,)（bit-identical）")
        v.close()

    # ---- A3b replay_eval 路径 34维对齐（AC2 HIGH gap·mutation#8 守卫：maker 路径外·A3 抓不到 line-666 _bv）----
    print("A3b replay_eval vecnorm-load 34维对齐（守 run_step4e:666 _bv·AC2 HIGH）")
    import tempfile
    import shutil as _shutil
    _td = tempfile.mkdtemp(); _cbase = os.path.join(_td, "aug_ckpt")
    _m, _v = make_continuous_safe_ppo_model(paths=_PATHS[:2], n_envs=1, seed=0, augment_rho=True, subproc=False)
    _m.save(_cbase); _v.save(_cbase + "_vecnorm.pkl"); _v.close()               # 存 augment-on 微 ckpt（34维 vecnorm）
    _tp = [_load(p) for p in _PATHS[:2]]
    _sa = S._AUGMENT_RHO; S._AUGMENT_RHO = True                                 # replay_eval 读 module 级 _AUGMENT_RHO
    try:
        _agg = S.replay_eval(_cbase, "continuous", 0.0, _tp, continuous_algo="ppo")
        chk(isinstance(_agg, dict),
            "A3b replay_eval(augment-on) 不崩返回 agg（line-666 丢 augment→_bv 27维→VecNormalize.load 'spaces must have the same shape' 硬崩·钱图 replay 全废）")
    except Exception as _ex:
        chk(False, f"A3b replay_eval(augment-on) 崩：{type(_ex).__name__}: {_ex}（line-666 _bv 修复丢失？）")
    finally:
        S._AUGMENT_RHO = _sa; _shutil.rmtree(_td, ignore_errors=True)

print("\n" + ("=" * 50))
print("✅ 全部通过" if N_FAIL == 0 else f"❌ {N_FAIL} 项失败")
sys.exit(1 if N_FAIL else 0)
