"""
停船配方验证 / step4e 预跑脚本（本地 VSCode 直接跑，避开会话暂停杀后台）
========================================================================
【目的】用已产品化的"停船 setup 修复配方"（ent_coef=0.01 + VecNormalize(obs+reward)）训 Discrete-safe
       基线，看【到达率】是否随【场景多样性 + 训练预算】爬升 —— 决定是否值得上完整 step4e。

【背景】Phase 1 撞过"停船墙"：默认超参(ent_coef=0)下 RL 塌缩到停船、到达率 0%。诊断+实验定位根因=探索塌缩，
       配方 VecNormalize 已解停船（船跑满 episode + d_final 单调接近目标），但小代理实验(5-64场景)到达率仍低
       (0-20%)。本脚本用更多场景 + 满预算，验证到达是否真能爬上去（Krasowski 用 ~1400场景×10seed×3M 达 86%）。

【怎么跑】（trb 环境）
    /opt/miniconda3/envs/trb/bin/python -B 代码/run_validation.py
    # VSCode：右下角解释器选 /opt/miniconda3/envs/trb/bin/python，然后直接 Run
    # 首次自动下载训练/测试场景到 /tmp/trb_scenarios_pool/（gitlab BSD，每个~80KB）
    # 预计耗时 ~1.5 小时（200场景 × 3M步 × 8进程；按 CPU 核数调 N_ENVS）

【改规模】下方 CONFIG。想更接近 Krasowski 就调大 N_TRAIN_SCENARIOS（→~1400）；想四方对比改 ENV/COLREGS_WEIGHT。

【输出】每段打印 + 存到 结果/validation_result.txt：
    [步数] 训练集到达% / 测试集到达%(held-out 泛化) / 碰撞% / 违规 / 紧急步% / Ep长

【跑完发我】把 `结果/validation_result.txt` 的内容发给我（或直接复制终端那几行趋势）。

【怎么判读】
    · 测试集到达率随段明显爬升（→40-80%）= 配方+多样性成立 → 我整理完整 step4e 启动方案。
    · 碰撞率应恒 0%（盾的安全保证；若 >0 请务必告诉我，是要查的安全问题）。
    · 到达率卡在低位(~10%)不爬 = 多样性非主因，需查精确终端(goal 朝东±10°)/超参 —— 我再深挖（不动 Krasowski 忠实 reward 系数）。
"""
from __future__ import annotations
import os
import sys
import time

# ===================== CONFIG（按需改）=====================
N_TRAIN_SCENARIOS = 200          # 训练场景数（越多越接近 Krasowski ~1400）
N_TEST_SCENARIOS = 25            # held-out 测试场景数（测泛化 = Krasowski 报的 86% 那个指标）
TOTAL_STEPS = 3_000_000          # 总训练步（Krasowski / seed = 3M）
N_SEG = 10                       # 分段评估段数（看到达率趋势）
SEED = 0                         # 随机种子
N_ENVS = 8                       # 并行采样进程数（建议 = CPU 核数 - 2）
ENV = "shielded"                 # "shielded"=Discrete-safe(有盾) / "unshielded"=Base/RR(无盾)
COLREGS_WEIGHT = 1.0             # Discrete-safe / Rule-reward = 1.0 ；Base = 0.0
# =========================================================

_SDIR = "/tmp/trb_scenarios_pool"
_BASE = ("https://gitlab.lrz.de/tum-cps/commonocean-scenarios/-/raw/main/scenarios/"
         "HandcraftedTwoVesselEncounters_01_24/ZAM_AAA-1_20240121_T-{}.xml")


def _select_ids():
    """分散取样：训练 = 等距 N 个；测试 = 偏移半步的等距、与训练不重叠（held-out）。"""
    stride = max(1, 2000 // N_TRAIN_SCENARIOS)
    train = list(range(0, 2000, stride))[:N_TRAIN_SCENARIOS]
    tset = set(train)
    test = [i for i in range(stride // 2, 2000, stride) if i not in tset][:N_TEST_SCENARIOS]
    return train, test


def _download(ids):
    import urllib.request
    os.makedirs(_SDIR, exist_ok=True)
    paths, fail = [], []
    for n in ids:
        dst = f"{_SDIR}/T-{n}.xml"
        if not (os.path.exists(dst) and os.path.getsize(dst) > 1000):
            try:
                urllib.request.urlretrieve(_BASE.format(n), dst)
            except Exception:                                  # noqa: BLE001
                fail.append(n)
                continue
        paths.append(dst)
    return paths, fail


def main():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import VecNormalize
    from trb_env.train import ENT_COEF, VECNORM_KWARGS, POLICY_NET_ARCH, make_obs_transform
    from trb_env.usv_scenarios import load_scenario_pool, make_vec_env
    from trb_env.usv_shield import ShieldedUSVEnv, UnshieldedUSVEnv
    from trb_env.evaluate import evaluate

    env_cls = ShieldedUSVEnv if ENV == "shielded" else UnshieldedUSVEnv
    train_ids, test_ids = _select_ids()
    print(f"下载场景：训练 {len(train_ids)} + 测试(held-out) {len(test_ids)} → {_SDIR} …", flush=True)
    train_paths, f1 = _download(train_ids)
    test_paths, f2 = _download(test_ids)
    if f1 or f2:
        print(f"⚠️ {len(f1) + len(f2)} 个场景下载失败（已跳过；检查网络/gitlab 可达）", flush=True)
    train_pool = load_scenario_pool(train_paths)
    test_pool = load_scenario_pool(test_paths)
    train_eval = train_pool[:15]                               # 训练集子集评估（看是否在学）

    venv = make_vec_env(paths=train_paths, n_envs=N_ENVS, env_cls=env_cls,
                        env_kwargs=dict(colregs_weight=COLREGS_WEIGHT), subproc=True, seed=SEED)
    venv = VecNormalize(venv, gamma=0.99, **VECNORM_KWARGS)
    model = MaskablePPO("MlpPolicy", venv, policy_kwargs=dict(net_arch=POLICY_NET_ARCH),
                        seed=SEED, ent_coef=ENT_COEF, verbose=0, device="cpu")  # MLP 太小、GPU 反慢（D17）

    out = []
    def emit(s):
        print(s, flush=True)
        out.append(s)

    emit(f"# 停船配方验证 | {env_cls.__name__} colregs_weight={COLREGS_WEIGHT} | "
         f"{len(train_pool)}训练/{len(test_pool)}测试场景 | seed={SEED} | "
         f"ent_coef={ENT_COEF} + VecNormalize{dict(VECNORM_KWARGS)}")
    emit(f"{'步数':>9} | {'训练集到达%':>9} {'测试集到达%':>9} | {'碰撞%':>5} {'违规/局':>6} {'紧急步%':>6} {'Ep长s':>6}")
    seg = TOTAL_STEPS // N_SEG

    def fac(sc, pp):
        return env_cls(sc, pp, colregs_weight=COLREGS_WEIGHT)

    t0 = time.time()
    for c in range(N_SEG):
        print(f"  [段 {c + 1}/{N_SEG}] 训练 {seg} 步中…", flush=True)
        model.learn(total_timesteps=seg, reset_num_timesteps=(c == 0))
        venv.training = False                                  # 冻结归一化统计做评估
        tf = make_obs_transform(venv)
        agg_tr, _ = evaluate(fac, model, train_eval, obs_transform=tf)
        agg_te, _ = evaluate(fac, model, test_pool, obs_transform=tf)
        venv.training = True
        emit(f"{(c + 1) * seg:>9} | {agg_tr['到达率%']:>9.0f} {agg_te['到达率%']:>9.0f} | "
             f"{agg_te['碰撞率%']:>5.0f} {agg_te['违规次数/局']:>6.2f} "
             f"{agg_te['紧急步%']:>6.1f} {agg_te['Ep长s']:>6.0f}")
    emit(f"# 训练总耗时 {time.time() - t0:.0f}s（{TOTAL_STEPS / (time.time() - t0):.0f} fps）")

    model.save("/tmp/val_model.zip")
    venv.save("/tmp/val_vecnorm.pkl")
    venv.close()

    res_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "结果")
    os.makedirs(res_dir, exist_ok=True)
    out_path = os.path.join(res_dir, "validation_result.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print(f"\n✅ 完成。结果存到 {out_path}", flush=True)
    print("⭐ 把这个文件的内容发给我（或复制上面 [步数...] 那几行趋势）。", flush=True)

if __name__ == "__main__":
    main()
