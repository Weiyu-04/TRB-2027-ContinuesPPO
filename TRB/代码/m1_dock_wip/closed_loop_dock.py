#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""闭环停车模块评估 harness（`03` L179 升级·服务器40测试集·全10种子·公平权限）。

【机制/评估测·非生产集成】停车动作走 env.step(经 safe_action·近门无冲突时盾恒等·保盾跨步状态)。
RL单独臂循环忠实镜像 evaluate.run_episode_continuous(reset/obs_transform/predict deterministic/env.step/reached=flags['goal'])
→ 有 manifest 时 RL单独臂应【复现训练 eval 的逐种子到达率】(同 40 测试集+同循环)·dock 臂=同管线+近门无冲突注入=干净 delta。

场景源(二选一):
  · STEP4E_MANIFEST 设 → load_manifest_split 取【真 40 测试集】(复用 run_step4e·下载 T-id+追越 OT·须服务器)——真数。
  · 否则 SCN_GLOB(默认本机13场景) → 任意 glob[:N]——【本机 sanity·非标准集·数字不可比 eval】。
种子: SEEDS(默认 "0 1 2 3 4 5 6 7 8 9")·各 load checkpoints/<CKPT_TMPL>。
dock 权限: 默认【限 RL 箱】(a±A_NORMAL_ACCEL_MAX/ω±A_NORMAL_OMEGA_MAX=公平·同 RL/离散权限·L179 confound 修)；
           DOCK_FULL_PHYSICAL=1 → 满物理(a±0.24/ω±0.03·仅诊断·会引入 5×加速/1.67×转向 confound)。
接管: dist≤TAKEOVER_R(350) 且 env._rho==RHO_NO_CONFLICT → dock 覆盖 u_desired(仍走盾)。

用法(本机sanity): PYTHONPATH=代码 CKPT_DIR=<dir> SEEDS="2 6" python 代码/m1_dock_wip/closed_loop_dock.py
用法(服务器真数): PYTHONPATH=~/trb/代码 STEP4E_MANIFEST=~/trb/balanced_pool/manifest_hocr_200.json \
    STEP4E_BALANCED_DIR=~/trb/balanced_pool STEP4E_SDIR=~/trb/scenarios \
    CKPT_DIR=~/trb/结果/结果0710-22:00-10种子最优方案/checkpoints /root/miniconda3/bin/python 代码/m1_dock_wip/closed_loop_dock.py
"""
import os, sys, glob, math, statistics, numpy as np
sys.path.insert(0, '/Users/weiyutang/Desktop/TRB/代码')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from trb_env.train import make_obs_transform
from trb_env.usv_continuous_shield import ContinuousProjectionEnv
from trb_env.usv_colregs import RHO_NO_CONFLICT
from trb_env.usv_scenarios import load_scenario_pool
from trb_env.usv_env import A_NORMAL_OMEGA_MAX, A_NORMAL_ACCEL_MAX, A_ACC, A_OMEGA
# 🆕 DOCK_VER 选控制器版本：v4=原版(现有900m结果由它产出) · v5=治近门追踪发散(`03` L192 I/J)
_DV = os.environ.get('DOCK_VER', 'v4').lower()
if _DV == 'v5':
    import dock_controller_v5 as DC
else:
    import dock_controller_v4 as DC

def wrap(a): return (a + math.pi) % (2 * math.pi) - math.pi

# 🆕 `03` L186（user 2026-07-14）：KIND 开关支持【离散臂+停车】(四方到达率公平对比·补缺的一格)。
#   默认 'continuous' = 逐字不变(连续臂已产出真数据·bit-identical)；'discrete' = 离散臂(MaskablePPO+ShieldedUSVEnv)。
#   ⚠️离散设计=停车 dock 出连续[a,ω]→【量化到最近离散动作】(保离散臂全程离散·不给连续逃生舱)。dock 权限本=离散范围
#   (±0.048/±0.018=A_ACC/A_OMEGA 上界) → 量化=7级网格取整·扰动小=离散能达的最好停车·忠实"离散动作"约束。
KIND = os.environ.get('KIND', 'continuous').lower()
_A_OMEGA_N = len(A_OMEGA)                                     # =7（index=ia*7+iw·忠实 DISCRETE_ACTIONS=[(a,w) for a in A_ACC for w in A_OMEGA]）
def _quantize_discrete(a, w):
    """dock 连续 [a,ω] → 最近离散动作下标（DISCRETE_ACTIONS 顺序 = a 外层 w 内层 → idx = ia*7 + iw）。"""
    ia = int(np.argmin([abs(float(a) - x) for x in A_ACC]))
    iw = int(np.argmin([abs(float(w) - x) for x in A_OMEGA]))
    return ia * _A_OMEGA_N + iw

# ── 配置 ──
CKPT_DIR   = os.environ.get('CKPT_DIR', '/Users/weiyutang/Desktop/TRB/结果/结果0710-22:00-10种子最优方案/checkpoints')
CKPT_TMPL  = os.environ.get('CKPT_TMPL', 'Continuous-safe_s{s}_L1rateON_ppo_s{s}')  # {s}=种子号
SEEDS      = [int(x) for x in os.environ.get('SEEDS', '0 1 2 3 4 5 6 7 8 9').split()]
TAKEOVER_R = float(os.environ.get('TAKEOVER_R', '350'))
FULL_PHYS  = os.environ.get('DOCK_FULL_PHYSICAL', '0') == '1'
MANIFEST   = os.environ.get('STEP4E_MANIFEST', '')
# env 配置【须匹配 checkpoint 训练配置】(默认=L1rateON 金标 run·复用别的 run 须 env 覆盖·`03` L179 harness审 H1-MEDIUM 防静默错数)
ENV_SHIELD    = os.environ.get('ENV_SHIELD', '1') == '1'
ENV_GOAL_CONE = os.environ.get('ENV_GOAL_CONE', '')            # 空=None(锥关)
ENV_VFLOOR    = float(os.environ.get('ENV_VFLOOR', '2.0'))
ENV_AUGMENT   = os.environ.get('ENV_AUGMENT', '0') == '1'
OUT_DIR       = os.environ.get('OUT_DIR', '')                 # 设则把结构化结果(.json+.txt 表)落盘到此目录(断线/screen挂不丢·好解析)

# ── 场景源 ──
if MANIFEST:                                                 # 真数：复用 run_step4e 已久经考验的 manifest 拆分
    from run_step4e import load_manifest_split
    _bdir = os.environ.get('STEP4E_BALANCED_DIR') or os.path.dirname(os.path.abspath(MANIFEST))
    _train, test_paths, _info = load_manifest_split(MANIFEST, _bdir)
    scn = test_paths                                         # 【真 40 测试集】=训练 eval 同集
    SRC = f"manifest 测试集(n={len(scn)}·train={len(_train)})"
else:
    SCN = os.environ.get('SCN_GLOB', '/private/tmp/trb_scenarios_pool/T-*.xml')
    scn = sorted(glob.glob(SCN))[:int(os.environ.get('SCN_N', '40'))]
    SRC = f"本机 glob sanity(n={len(scn)}·⚠️非标准集·不可比 eval)"

pool = load_scenario_pool(scn)
DOCK_WMAX = DC.W_MAX if FULL_PHYS else A_NORMAL_OMEGA_MAX
DOCK_AMAX = None       if FULL_PHYS else A_NORMAL_ACCEL_MAX  # None=不额外限(env clip 物理±0.24)
print(f"臂={KIND} | 场景源={SRC} | 种子={SEEDS} | 接管半径={TAKEOVER_R} | dock权限={'满物理±0.24/±0.03' if FULL_PHYS else 'RL箱±0.048/±0.018(公平)'}"
      + ("｜离散 dock 量化到最近网格动作" if KIND == 'discrete' else ""))

def mk_env(sc, pp):
    if KIND == 'discrete':                                    # 🆕 L186：离散臂 = ShieldedUSVEnv(As(ρ) masking·Discrete-safe 配置 colregs_weight=1.0·run_step4e:299)
        from trb_env.usv_scenarios import ShieldedUSVEnv     # eval colregs_weight 只影响 reward 不影响到达/obs → 与训练同构
        return ShieldedUSVEnv(sc, pp, colregs_weight=1.0)
    _gc = float(ENV_GOAL_CONE) if ENV_GOAL_CONE else None
    return ContinuousProjectionEnv(sc, pp, shield=ENV_SHIELD, goal_cone_half=_gc, goal_v_floor=ENV_VFLOOR, augment_rho=ENV_AUGMENT)

def run_ep(env_f, model, tf, sc, pp, use_dock):
    """单局：忠实镜像 run_episode_continuous + 可选近门无冲突 dock 接管。返回诊断。"""
    env = env_f(sc, pp)
    obs, info = env.reset(seed=0)
    reached = False; took_over = 0
    for _ in range(200):                                     # 200=backstop；env 在时限(~170)先 truncate
        a_obs = tf(obs)
        if KIND == 'discrete':                               # 🆕 L186：离散 MaskablePPO 须喂 action_masks（忠实 evaluate.run_episode:296-299）
            act, _ = model.predict(a_obs, action_masks=env.action_masks(), deterministic=True)
        else:
            act, _ = model.predict(a_obs, deterministic=True)
        if use_dock:
            ego = env._ego_vs(); goal = env.env.goal_center; rho = env._rho
            dist = float(np.hypot(ego.position[0] - goal[0], ego.position[1] - goal[1]))
            if dist <= TAKEOVER_R and rho == RHO_NO_CONFLICT:
                st = [ego.position[0], ego.position[1], ego.orientation, float(getattr(ego, 'velocity', 0.0))]
                _u = DC.dock_controller(st, (goal[0], goal[1]), wmax=DOCK_WMAX)
                if DOCK_AMAX is not None:                     # 公平：加速度也限 RL 箱(L179·dock 原硬编 ±0.24=5×)
                    _u = np.array([float(np.clip(_u[0], -DOCK_AMAX, DOCK_AMAX)), _u[1]])
                # 🆕 L186：连续臂 dock 直接施连续[a,ω]；离散臂 dock 量化到最近离散动作(保离散全程离散·rho==NO_CONFLICT→regular mask 全通·量化下标必合法)
                act = _quantize_discrete(_u[0], _u[1]) if KIND == 'discrete' else _u
                took_over += 1
        if KIND == 'discrete':                               # 离散 step 吃整数下标（ShieldedUSVEnv.step:103 拒非整数）
            obs, _r, term, trunc, info = env.step(int(act))
        else:
            obs, _r, term, trunc, info = env.step(np.asarray(act, dtype=float))
        fl = info.get('flags', {})
        if fl.get('goal', False): reached = True; break
        if term or trunc: break
    # 失败模式(RL单独臂用)：停短(stopped/末速≈0) vs 绕圈(time/末速>1)
    ego = env._ego_vs(); goal = env.env.goal_center
    endv = float(getattr(ego, 'velocity', 0.0))
    enddist = float(np.hypot(ego.position[0] - goal[0], ego.position[1] - goal[1]))
    fl = info.get('flags', {})
    term_flag = next((k for k in ('collision', 'goal', 'stopped', 'area', 'time') if fl.get(k)), 'other')
    if reached: mode = 'reach'
    elif term_flag == 'stopped' or (endv < 0.5 and enddist < 200): mode = 'stopshort'
    elif term_flag == 'time': mode = 'circle'
    else: mode = 'other'
    return dict(reached=reached, took_over=took_over, mode=mode, term=term_flag, endv=endv, enddist=enddist)

def iqm(vals):
    if not vals: return float('nan')
    s = sorted(vals); n = len(s); lo = int(n * 0.25)
    mid = s[lo:n - lo] if n - lo > lo else s
    return sum(mid) / len(mid)

# ── 逐种子跑 ──
n = len(pool)
rows = {}
seed_modes = {}                                              # 逐种子失败模式(落盘用)
mode_tab = {}                                                # 失败模式 × 是否救活
print(f"\n{'seed':>4} | {'RL单独':>7} {'+停车':>7} {'Δ':>5} | 失败模式(RL单独)")
for s in SEEDS:
    ck = os.path.join(CKPT_DIR, CKPT_TMPL.format(s=s))
    if not (os.path.exists(ck + '.zip') and os.path.exists(ck + '_vecnorm.pkl')):   # 🔴修 harness审 H2-MEDIUM:查全 .zip+.pkl(缺 pkl 原会跑一半崩丢后续种子)
        print(f"{s:>4} | ❌ 缺 checkpoint(.zip 或 _vecnorm.pkl) {ck} → 跳过"); continue
    _bv = DummyVecEnv([lambda: mk_env(pool[0][0], pool[0][1])])
    _vn = VecNormalize.load(ck + '_vecnorm.pkl', _bv); _vn.training = False
    _env_dim = int(_bv.observation_space.shape[0]); _vn_dim = int(np.asarray(_vn.obs_rms.mean).shape[0])   # 🔴修 harness审 H1-MEDIUM:断言 vecnorm 维=env 维
    if _vn_dim != _env_dim:                                   # augment_rho 配置不匹配 checkpoint → 静默错数/形状崩 → fail-fast
        raise SystemExit(f"🔒 s{s}: vecnorm obs 维={_vn_dim} ≠ env obs 维={_env_dim}(ENV_AUGMENT/盾配置不匹配 checkpoint 训练配置)→中止防静默错数(augment run 须设 ENV_AUGMENT=1)")
    tf = make_obs_transform(_vn)
    if KIND == 'discrete':                                    # 🆕 L186：离散臂 = MaskablePPO（连续臂 = plain PPO·不变）
        from sb3_contrib import MaskablePPO
        model = MaskablePPO.load(ck + '.zip', device='cpu')
    else:
        model = PPO.load(ck + '.zip', device='cpu')
    base = dock = 0; modes = {}; coll_base = coll_dock = 0   # 🆕 碰撞计数(验早接管不引入碰撞·term=='collision')
    for sc, pp in pool:
        rb = run_ep(mk_env, model, tf, sc, pp, False)
        rd = run_ep(mk_env, model, tf, sc, pp, True)
        base += rb['reached']; dock += rd['reached']
        coll_base += int(rb['term'] == 'collision'); coll_dock += int(rd['term'] == 'collision')
        if not rb['reached']:                                # 按 RL单独失败模式记 dock 是否救活
            m = rb['mode']; modes[m] = modes.get(m, [0, 0])
            modes[m][0] += 1; modes[m][1] += int(rd['reached'])
    print(f"    [s{s} 碰撞] RL单独={coll_base}/{n}  +停车={coll_dock}/{n}", flush=True)   # 🆕 逐种子碰撞
    rows[s] = (base, dock)
    seed_modes[s] = {m: list(v) for m, v in modes.items()}   # 逐种子失败模式(落盘用)
    for m, (tot, resc) in modes.items():
        mt = mode_tab.setdefault(m, [0, 0]); mt[0] += tot; mt[1] += resc
    ms = ' '.join(f"{m}:{resc}/{tot}救" for m, (tot, resc) in sorted(modes.items()))
    print(f"{s:>4} | {base:>4}/{n} {dock:>4}/{n} {dock-base:>+5} | {ms}")

# ── 汇总 ──
base_pct = {s: 100.0 * b / n for s, (b, d) in rows.items()}
dock_pct = {s: 100.0 * d / n for s, (b, d) in rows.items()}
print(f"\n=== 汇总(n={n} 场景/种子) ===")
print(f"到达率% RL单独: {{{', '.join(f's{s}:{base_pct[s]:.1f}' for s in rows)}}}")
print(f"到达率% +停车:  {{{', '.join(f's{s}:{dock_pct[s]:.1f}' for s in rows)}}}")
if rows:
    print(f"IQM  RL单独={iqm(list(base_pct.values())):.1f}  +停车={iqm(list(dock_pct.values())):.1f}")
    print(f"崩(<10%)  RL单独={[s for s in rows if base_pct[s]<10]}  +停车={[s for s in rows if dock_pct[s]<10]}")
print("失败模式 × dock 救活率:")
for m, (tot, resc) in sorted(mode_tab.items()):
    print(f"  {m:>10}: {resc}/{tot} 救活 = {100*resc/tot:.0f}%" if tot else f"  {m}: 0")
if not MANIFEST:
    print("⚠️ 本机 glob=非标准集·数字不可比 eval;真数须服务器 STEP4E_MANIFEST 模式。")

# ── 结果落盘(OUT_DIR 设则写 .json 结构化 + .txt 表·断线不丢·`03` L179 harness审 H2-LOW) ──
if OUT_DIR and rows:
    import json as _json
    os.makedirs(OUT_DIR, exist_ok=True)
    _tag = 'dock_closedloop_' + ('disc_' if KIND == 'discrete' else '') + ('40set' if MANIFEST else 'sanity') + ('_fullphys' if FULL_PHYS else '')  # 臂+权限进文件名·连续/离散/满物理并行不撞
    _out = dict(
        source=SRC, seeds=SEEDS, n_scen=n, takeover_r=TAKEOVER_R,
        dock_authority=('full_physical' if FULL_PHYS else 'rl_box'),
        per_seed={str(s): dict(base=rows[s][0], dock=rows[s][1], n=n,
                               base_pct=round(base_pct[s], 2), dock_pct=round(dock_pct[s], 2),
                               modes=seed_modes.get(s, {})) for s in rows},
        iqm_base=round(iqm(list(base_pct.values())), 2), iqm_dock=round(iqm(list(dock_pct.values())), 2),
        crashed_base=[s for s in rows if base_pct[s] < 10], crashed_dock=[s for s in rows if dock_pct[s] < 10],
        mode_table={m: dict(total=t, rescued=r) for m, (t, r) in mode_tab.items()},
    )
    _jp = os.path.join(OUT_DIR, _tag + '.json')
    with open(_jp, 'w') as fh:
        _json.dump(_out, fh, ensure_ascii=False, indent=1)
    _tp = os.path.join(OUT_DIR, _tag + '.txt')
    with open(_tp, 'w') as fh:
        fh.write(f"停车模块闭环评估 | {SRC} | 接管半径{TAKEOVER_R} | dock权限={'满物理' if FULL_PHYS else 'RL箱(公平)'}\n")
        fh.write(f"{'seed':>4} {'RL单独':>8} {'+停车':>8} {'Δ':>6}  失败模式(RL单独)\n")
        for s in rows:
            _ms = ' '.join(f"{m}:{r}/{t}救" for m, (t, r) in sorted(seed_modes.get(s, {}).items()))
            fh.write(f"{s:>4} {rows[s][0]:>4}/{n} {rows[s][1]:>4}/{n} {rows[s][1]-rows[s][0]:>+6}  {_ms}\n")
        fh.write(f"\nIQM RL单独={_out['iqm_base']} +停车={_out['iqm_dock']} | 崩 RL单独={_out['crashed_base']} +停车={_out['crashed_dock']}\n")
        fh.write("失败模式×dock救活率: " + ' '.join(f"{m}:{r}/{t}={100*r/t:.0f}%" for m, (t, r) in sorted(mode_tab.items()) if t) + "\n")
    print(f"✅ 结果已落盘: {_jp} + {_tp}")
