# -*- coding: utf-8 -*-
"""**单步求解耗时（实时性）** —— 论文必须有、项目唯一还缺的那个指标（`03` L243-续3）。

═══ 为什么写成独立脚本、而不是在盾里插桩 ═══════════════════════════════════════
盾（`usv_projection.py` / `usv_continuous_shield.py`）是**安全关键代码**，按 `CLAUDE.md` §2
动它要走最高审核标准；而这次实验**只有一次机会**，起跑前改热路径的风险远大于收益。
而且这是 **B 类量**（评估期量）——只要存档还在，**任何时候补测都行**，不阻塞起跑。
⟹ 本脚本**一行生产代码都不改**：直接把盾对象拿出来，喂真实回放出来的状态，掐表。

═══ 量什么（三层，分开报）═══════════════════════════════════════════════════════
  ① **盾（投影 QP）单次求解**：`ContinuousColregsProjection.safe_action(...)` 的墙钟
     —— 这就是论文里"实时性"要报的那个数
  ② **策略前向**：`model.predict(obs, deterministic=True)`
  ③ **① + ② + 环境物理** 的整步耗时
每层都报 **均值 / 中位 / p95 / p99 / 最大**（实时性看的是**尾部**，不是均值 —— 报均值等于没报）。

🔴 **诚实口径（写进论文）**：
  · 这是**单线程、CPU、本机**的墙钟，随机器而变 ⟹ 必须连**机器型号 + 线程设置**一起写。
  · 决策周期是 **dt = 10 秒**（`k_max=170` 步 ≈ 1700 秒）⟹ 只要 p99 ≪ 10 秒就满足实时性；
    别写成"比某某快 N 倍"——那要同机同口径才成立。
  · ρ 的分布会影响耗时（紧急态走 Alg.1、无冲突态投影退化成恒等）⟹ **按 ρ 拆开报**。
  · **他船不在窗内的步不解 QP**（`usv_continuous_shield.py:220` 短路）⟹ 那些步**不计入** ①，
    否则一堆 0 会把 p99 冲淡、把"实时性"说得比实际好听。

═══ 用法（服务器上·纯评估·不占训练卡）═══════════════════════════════════════════
    python3 -B 代码/tests/measure_solve_time.py <checkpoint 路径不带 .zip> [场景数]
    # 例：python3 -B 代码/tests/measure_solve_time.py 结果/checkpoints/Continuous-safe_s0_F240oursPpoS0 30
"""
import json
import os
import statistics as st
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_CODE)
sys.path.insert(0, _CODE)


class _Skip(Exception):
    """这一步他船不在窗内 ⟹ 盾短路、不解 QP ⟹ 不该算进求解耗时分布。"""


def pct(v, q):
    if not v:
        return float("nan")
    s = sorted(v)
    i = min(len(s) - 1, int(round(q * (len(s) - 1))))
    return s[i]


def report(name, v, unit="ms"):
    if not v:
        print(f"  {name:<28} —（没采到样本）")
        return
    print(f"  {name:<28} n={len(v):<7} 均值 {st.mean(v):7.3f} 中位 {st.median(v):7.3f} "
          f"p95 {pct(v,0.95):7.3f} p99 {pct(v,0.99):7.3f} 最大 {max(v):7.3f}  {unit}")


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    base = os.path.abspath(sys.argv[1])
    n_sc = int(sys.argv[2]) if len(sys.argv) > 2 else 30

    import numpy as np
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from trb_env import usv_action_dist as _uad          # Beta 档反序列化前必须先 import
    _ = _uad
    from trb_env.train import make_obs_transform
    from trb_env.usv_continuous_shield import ContinuousProjectionEnv
    from trb_env.usv_scenarios import load_scenario_pool

    # ── 存档旁的 sidecar 说了它自己是什么档（别靠环境变量猜·同 reeval 的回读纪律）──
    sc = {}
    pj = base + ".progress.json"
    if os.path.exists(pj):
        sc = (json.load(open(pj, encoding="utf-8")) or {}).get("config_sig") or {}
    gw = sc.get("gw_entry", "paper")
    print(f"【单步求解耗时】{os.path.basename(base)}")
    print(f"  存档自述：act_dist={sc.get('act_dist')} · gw_entry={gw} · dataset={sc.get('dataset')}")
    print(f"  线程设置：OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')} "
          f"（🔴 报数时必须连机器型号一起写 —— 这是墙钟，随机器变）\n")

    # ── 场景池：用官方测试集的前 n_sc 个（只为量耗时，不产任何指标）──
    sdir = os.environ.get("STEP4E_SDIR") or os.path.join(_ROOT, "scenarios")
    paths = [os.path.join(sdir, f"T-{i}.xml") for i in range(n_sc)]
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        raise SystemExit(f"🔒 {sdir} 里没找到 T-*.xml")
    pool = load_scenario_pool(paths)

    model = PPO.load(base + ".zip", device="cpu")
    t_shield, t_policy, t_step, by_rho = [], [], [], {}

    for sc_obj, pp in pool:
        env = ContinuousProjectionEnv(sc_obj, pp, gw_entry=gw)
        tf = None
        if os.path.exists(base + "_vecnorm.pkl"):
            vn = VecNormalize.load(base + "_vecnorm.pkl", DummyVecEnv([lambda: ContinuousProjectionEnv(sc_obj, pp, gw_entry=gw)]))
            vn.training = False
            tf = make_obs_transform(vn)
        obs, _ = env.reset()
        done = False
        while not done:
            o = tf(obs) if tf is not None else obs
            t0 = time.perf_counter()
            a, _ = model.predict(o, deterministic=True)
            t1 = time.perf_counter()
            # ① 盾单次求解：**照 `usv_continuous_shield.py:235` 的原样调**，不改任何参数
            try:
                s_obs = env._obs_vs()          # 逐字照抄 `usv_continuous_shield.py:220`（他船窗外时返 None）
                if s_obs is None:              # 他船不在窗内 ⟹ 盾走短路、不解 QP ⟹ 不计入求解耗时
                    raise _Skip()
                ts = time.perf_counter()
                env.proj.safe_action(env._ego_vs(), s_obs, np.asarray(a, dtype=float),
                                     env.env.dt, env.env.p, obs_width=env._obs_width)
                t_shield.append((time.perf_counter() - ts) * 1e3)
                by_rho.setdefault(int(getattr(env, "_rho", -1)), []).append(t_shield[-1])
            except _Skip:
                pass                                        # 无他船的步不算求解（否则把 0 掺进分布、把 p99 冲淡）
            except Exception as e:                          # 取内部状态的口子随版本会变 ⟹ 失败就只报②③
                if not t_shield and len(t_step) == 0:
                    print(f"  ⚠️ 盾单独计时取不到内部状态（{type(e).__name__}: {e}）"
                          " ⟹ 只报②策略前向 与 ③整步；盾的耗时 ≈ ③−② 上界")
            t2 = time.perf_counter()
            obs, _, term, trunc, info = env.step(a)
            t3 = time.perf_counter()
            t_policy.append((t1 - t0) * 1e3)
            t_step.append((t3 - t2 + t1 - t0) * 1e3)
            done = term or trunc

    print(f"  跑了 {len(pool)} 个场景 · 共 {len(t_step)} 步\n")
    report("① 盾（投影 QP）单次求解", t_shield)
    report("② 策略前向 predict", t_policy)
    report("③ 整步（②+盾+环境物理）", t_step)
    if t_shield and t_policy:
        print(f"\n  盾占整步的比例（中位）≈ {st.median(t_shield)/st.median(t_step)*100:.1f}%")
    if by_rho:
        print("\n  按态势 ρ 拆（0=无冲突 1=直航 2=对遇 3=交叉 4=追越 5=紧急）：")
        for r in sorted(by_rho):
            report(f"    ρ={r}", by_rho[r])
    dt_s = 10.0
    if t_step:
        print(f"\n  🔴 判据：决策周期 dt = {dt_s:.0f} 秒；整步 p99 = {pct(t_step,0.99):.3f} ms "
              f"= 周期的 {pct(t_step,0.99)/1e3/dt_s*100:.4f}% ⟹ "
              + ("**满足实时性**" if pct(t_step, 0.99) / 1e3 < dt_s else "🔴 **不满足**"))
    print("\n  ⚠️ 写作口径：这是单线程 CPU 墙钟，随机器而变 ⟹ 必须连机器型号 + 线程设置一起写；"
          "看尾部（p95/p99）不看均值；别写成『比某某快 N 倍』（要同机同口径才成立）。")


if __name__ == "__main__":
    main()
