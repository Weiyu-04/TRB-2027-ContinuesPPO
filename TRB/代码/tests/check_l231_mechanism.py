#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""L231 早停门工具：**机制中没中**（不是看指标好不好，是看根因有没有被打掉）。

═══ 为什么要单独一个工具 ═══
训练自带的 `trend` 只记 到达率/碰撞率/违规per局/紧急步%，**不记**我们真正要看的两件事：
  · 臂 A/C（Beta）：**贴满舵率**降没降 —— 这是"无界高斯+硬裁剪"根因被打掉的直接证据；
  · 臂 B/C（symmetric）：**盾判让路的步数占比**上没上去 —— 这是"盾几乎没识别到让路"被修掉的直接证据。
指标碰巧变好但机制没命中 = **不算数**（`03` L229-E 事先定死的判读规则第 ④ 条）。

═══ 用法 ═══
    # 单个存档（会自动找它的 vecnorm 与 sidecar）
    STEP4E_SDIR=<TRB>/scenarios python -B 代码/tests/check_l231_mechanism.py <ckpt_base 不带.zip> [更多...]
    # 例：50 万步早停门
    python -B 代码/tests/check_l231_mechanism.py 结果/checkpoints/Continuous-safe_s0_A231betaPpoS0
环境变量：
    L231_N=<场景数>   评多少个场景（默认 60·够看机制·几分钟）
    L231_KEYS=1,2,..  直接点名场景键（覆盖 L231_N）

═══ 判读（先定死·免得看到数字再找理由）═══
| 臂 | 机制中了 | 机制没中 |
|---|---|---|
| A / C（Beta） | 贴满舵率 **< 20%**（现状 73.5%）且确定性动作零越箱 | 贴满舵率仍 > 50% ⟹ 当场砍 |
| B / C（symmetric） | 让路**覆盖率** **≥ 60%** | 覆盖率仍 < 35% ⟹ 当场砍 |

paper 档基线实测：180 局样本 27%（68/249 步）· 60 局样本 12.5%（6/48 步·分母太小）
⟹ **跑这个工具务必 `L231_N≥150`**，否则覆盖率的分母只有几十步、噪声 ±10pt。

🔴 **2026-07-27 修门（原判据设错了·必须记着）**：本工具第一版拿"盾判让路步**占总步数**的比例 ≥2%"当判据，
   那个 2% 是我在**训练好的主线策略**上标定的（`03` L231-C2）。但半成品策略（50-100 万步）有两个混淆：
     ① **紧急态优先级最高**：`is_emergency → ρ5` 会**抢在让路之前**。半成品策略老是逼近碰撞，
        实测紧急步 **8~17%**（成熟主线只有 3.9%）⟹ 大量本该判让路的步被 ρ5 吃掉；
     ② 半成品策略到不了目标、局更长、态势分布整个不一样，"占总步数比例"的分母根本不可比。
   ⟹ 正确的判据是**覆盖率 = 盾判让路步 ÷ 瞬时让路谓词为真的步数**——它把"有多少让路态势"这个分母除掉了，
      量的正是这个修法**唯一**改变的东西。第一版用错分母，会把一个其实在起作用的修法误杀。

⚠️ 到达率在 50 万步时**本来就很低**（金标 10 种子在 50 万步是 0/7.5/0/0/0/0/2.5/0/2.5/0；100 万步是中位 0、最好 12.5）
   ⟹ **50 万步不看到达率**；到达率留到 150 万步那道门（`03` L229-E 事先定死的四条）。
⚠️ **紧急步% 是观察项**：半成品策略高属正常，但若到 150 万步仍 >10%，说明盾在频繁兜底 ⟹ 记进判读。
"""
import os
import sys
import json
import collections
import warnings

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))            # 代码/
import numpy as np                                     # noqa: E402

import run_step4e as R                                 # noqa: E402
from trb_env.usv_scenarios import load_scenario_pool   # noqa: E402
from trb_env import usv_colregs as _C                                                     # noqa: E402
from trb_env.usv_colregs import RHO_HEAD_ON, RHO_CROSSING, RHO_OVERTAKE, RHO_EMERGENCY   # noqa: E402
from trb_env.usv_env import A_NORMAL_ACCEL_MAX, A_NORMAL_OMEGA_MAX                        # noqa: E402

_BOX = np.array([A_NORMAL_ACCEL_MAX, A_NORMAL_OMEGA_MAX], dtype=float)
_GW = (RHO_HEAD_ON, RHO_CROSSING, RHO_OVERTAKE)
_EDGE = 0.999          # |u| ≥ 99.9% 箱 记作"贴满舵"（与 `03` L229-C / L231 反推口径一致）


def _pool():
    keys_env = os.environ.get("L231_KEYS", "").strip()
    if keys_env:
        keys = [k.strip() for k in keys_env.split(",") if k.strip()]
    else:
        n = int(os.environ.get("L231_N", "150"))   # 150 局 ⟹ 覆盖率的分母通常 200+ 步（60 局只有 ~50 步·噪声 ±10pt）
        # 用报数集（strict 563）的前 n 个：与最终报数同一批场景，机制读数才可比
        f = os.path.join(os.path.dirname(_HERE), "..", "结果", "结果0727-38臂同趟重评", "p1.json")
        if os.path.exists(f):
            keys = [str(k) for k in json.load(open(f, encoding="utf-8"))["strict键"][:n]]
        else:                                  # 退路：官方 2000 里取前 n 个（只影响绝对值·不影响"中没中"）
            keys = [str(i) for i in range(1, n + 1)]
    sdir = os.environ.get("STEP4E_SDIR") or os.path.join(os.path.dirname(_HERE), "..", "scenarios")
    paths = [os.path.join(sdir, f"T-{k}.xml") for k in keys]
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        raise SystemExit(f"🔒 一个场景都没找到（STEP4E_SDIR={sdir}）")
    return load_scenario_pool(paths), len(paths)


# ── 覆盖率的分母：瞬时让路谓词为真的步数 ──────────────────────────────────────────────
#   `ColregsStatechart.step` 每决策步被盾调用一次 ⟹ 在它外面包一层只读计数即可，
#   **不改任何返回值、不改任何行为**（动作仍完全由原状态机决定）。进程级 monkeypatch，
#   仅本工具内生效，仓库代码一行不动。
_TALLY = collections.Counter()
_ORIG_STEP = _C.ColregsStatechart.step


def _counting_step(self, s_l, s_m):
    r = _ORIG_STEP(self, s_l, s_m)
    _TALLY["steps"] += 1
    if (_C.crossing(s_l, s_m, self.t_horizon) or _C.head_on(s_l, s_m, self.t_horizon)
            or _C.overtake(s_l, s_m, self.t_horizon)):
        _TALLY["inst_gw"] += 1                      # 分母：这一步"客观上"处于让路态势（与评分器同款瞬时谓词）
        if r in _GW:
            _TALLY["covered"] += 1                  # 分子：盾也认出来了
    return r


_C.ColregsStatechart.step = _counting_step


def _sidecar(base):
    p = base + ".progress.json"
    if not os.path.exists(p):
        return {}
    try:
        return json.loads(open(p, encoding="utf-8").read()) or {}
    except Exception:
        return {}


def check(base, pool):
    sc = _sidecar(base)
    cfg = sc.get("config_sig") or {}
    act = cfg.get("act_dist", "gauss")
    gwe = cfg.get("gw_entry", "paper")
    steps = sc.get("num_timesteps", "?")

    applied, rhos = [], []

    class Spy:                       # 只旁录 applied/ρ，不改任何行为（走 replay_eval 的 policy_wrap 钩子）
        def __init__(self, m):
            self.m = m

        def __getattr__(self, k):
            return getattr(self.m, k)

        def predict(self, *a, **kw):
            return self.m.predict(*a, **kw)

    _TALLY.clear()
    _, per = R.replay_eval(base, "continuous", 0.0, pool, continuous_algo="ppo",
                           return_per=True, policy_wrap=Spy)
    inst_gw = _TALLY["inst_gw"]
    covered = _TALLY["covered"]
    coverage = covered / inst_gw if inst_gw else float("nan")
    # replay_eval 的 per 里带 rho_hist（态势步数）与 steps；applied 逐步序列在 evaluate 内部，
    # 这里用 rho_hist 算让路占比；贴满舵率用轨迹反推（Δψ/dt 恰等于施加的 ω·见 `03` L231-C）。
    hist = collections.Counter()
    tot_steps = 0
    for e in per:
        for k, v in (e.get("rho_hist") or {}).items():
            hist[int(k)] += int(v)
        tot_steps += int(e.get("steps") or 0)
    gw_frac = (hist[RHO_HEAD_ON] + hist[RHO_CROSSING] + hist[RHO_OVERTAKE]) / max(tot_steps, 1)
    em_frac = hist[RHO_EMERGENCY] / max(tot_steps, 1)

    # 贴满舵率：拿逐步轨迹反推 ω（θ̇=ω 精确·dt=10）。replay_eval 只对点名场景记轨迹 → 这里全记。
    _, per2 = R.replay_eval(base, "continuous", 0.0, pool, continuous_algo="ppo",
                            return_per=True, traj_idxs=list(range(len(pool))))
    W = []
    for e in per2:
        t = e.get("traj")
        if not t:
            continue
        psi = np.array([r["ego_psi"] for r in t], dtype=float)
        if len(psi) < 2:
            continue
        W.append((np.diff(psi) + np.pi) % (2 * np.pi) - np.pi)
    w = np.concatenate(W) / 10.0 if W else np.zeros(1)
    sat = float(np.mean(np.abs(w) >= A_NORMAL_OMEGA_MAX * _EDGE))
    zero = float(np.mean(np.abs(w) < 1e-9))
    flip = 0.0
    if len(w) > 1:
        s = np.abs(w) >= A_NORMAL_OMEGA_MAX * _EDGE
        flip = float(np.mean(s[:-1] & s[1:] & (np.sign(w[:-1]) != np.sign(w[1:]))))

    arr = 100.0 * sum(bool(e.get("reached")) for e in per) / max(len(per), 1)
    name = os.path.basename(base)
    print(f"\n■ {name}")
    print(f"   自描述: act_dist={act} · gw_entry={gwe} · 步数={steps} · 评了 {len(per)} 局 / {tot_steps} 步")
    print(f"   【机制·Beta】  贴满舵率 = {sat*100:6.2f}%   (现状高斯 73.5% · 中了 = <20%)"
          f"   恰居中 {zero*100:.2f}%   满舵翻面 {flip*100:.2f}%")
    _warn = "  ⚠️分母太小·噪声大" if inst_gw < 100 else ""
    print(f"   【机制·让路】  **覆盖率 = {coverage*100:5.1f}%**  = 盾认出 {covered} / 客观让路态势 {inst_gw} 步"
          f"   (paper 档基线 ~12~27% · 中了 = ≥60%){_warn}")
    print(f"                  （参考·分母不可比·别当判据）让路步占总步数 {gw_frac*100:.3f}%   紧急步 {em_frac*100:.2f}%")
    print(f"   （参考·50万步别当判据）到达率 {arr:.1f}%")
    verdict = []
    if act == "beta":
        verdict.append(("Beta 机制", "✅ 中" if sat < 0.20 else ("❌ 没中·当场砍" if sat > 0.50 else "🟡 灰区·看下一门")))
    if gwe == "symmetric":
        verdict.append(("让路机制", "✅ 中" if coverage >= 0.60 else ("❌ 没中·当场砍" if coverage < 0.35 else "🟡 灰区·看下一门")))
    else:
        verdict.append(("让路覆盖(paper 档·作参照)", f"{coverage*100:.1f}%"))
    for k, v in verdict:
        print(f"   ⟹ {k}: {v}")
    return dict(name=name, act_dist=act, gw_entry=gwe, steps=steps, sat=sat, zero=zero,
                flip=flip, gw_frac=gw_frac, em_frac=em_frac, arrival=arr, n=len(per),
                coverage=coverage, inst_gw=inst_gw, covered=covered)


if __name__ == "__main__":
    bases = [a[:-4] if a.endswith(".zip") else a for a in sys.argv[1:]]
    if not bases:
        raise SystemExit(__doc__)
    pool, n = _pool()
    print(f"场景池 {n} 局（机制读数用·不是报数）")
    out = [check(b, pool) for b in bases]
    print("\n" + "=" * 96)
    print(f"{'存档':<42}{'分布':>7}{'让路档':>11}{'贴满舵%':>9}{'让路覆盖%':>11}{'紧急步%':>9}{'到达%':>7}")
    for r in out:
        print(f"{r['name']:<42}{r['act_dist']:>7}{r['gw_entry']:>11}{r['sat']*100:9.2f}"
              f"{r['coverage']*100:11.1f}{r['em_frac']*100:9.2f}{r['arrival']:7.1f}")
    js = os.environ.get("L231_OUT")
    if js:
        json.dump(out, open(js, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"\n→ {js}")
