#!/usr/bin/env python3
"""STEP4E_WARMSTART_CKPT / warmstart_ckpt 旋钮测试（热启动·探索侧治崩·`03` L190）。

背景：崩种子(s5/s6)整个 5M 步没爬出"满速打转-超时"吸引子 = **探索失败**(非奖励给错)→ 奖励治不了。
  治法=探索侧：热启动灌一个好策略(健康种子)当初始化·把崩种子放进好盆地（顶会 JSRL 2204.02372 / AWAC 2006.09359 式）。
  接线：STEP4E_WARMSTART_CKPT=<源ckpt base 路径> → 连续PPO臂 maker 建 model 后 apply_warmstart：
    ① 灌【源均值策略】(set_parameters·只 policy 跳优化器) ② log_std 重置回 maker in-box 初值(不继承源已训σ·保 F1 守卫)
    ③ 复制源 VecNormalize obs_rms(+ret_rms)（策略吃归一化 obs·不复制=喂错分布=白热启动）。默认 ""/None=不热启动=bit-identical。

验：
  T1 默认 bit-identical：不传 warmstart vs 显式 None·同 seed → policy max|Δ|=0（严格 <1e-9）。
  T2 热启动真落地：均值策略==源均值·且!=新建同seed(非空操作)·log_std==maker初值(非源σ)·venv.obs_rms==源obs_rms·caliber 仍过。
  T3 守卫(全 fail-fast·防静默错)：
     · 结构守卫：误指【SAC 存档】(键集不相交·SAC 是默认臂产同名 .zip+_vecnorm.pkl+也有 'policy' 键)→ 必 raise
       （对抗审 wf wy3rlm90p HIGH-1 实测坐实：set_parameters(exact_match=False)=load_state_dict(strict=False)·键集不相交时【静默不灌一个权重不报错】=随机初始化假装热启动）。
     · obs 维守卫：34维(augment_rho)源 → 27维目标 → 必 raise。
     · 缺文件守卫：源 .zip/_vecnorm.pkl 缺 → 必 raise。
  T4 config_conflict 混写守卫（`03` L190 D2 自审补漏·第37元·末元）：
     从零记录 vs 热启动run → 冲突；反向 → 冲突；同源 → 无冲突；异源(s1 vs s3) → 冲突；
     **旧记录(无 warmstart 字段)+默认run → 不误冲突**（向后兼容·不误伤现有金标 jsonl）。
     命门：热启动 vs 从零=两种不同训练流程·同TAG混写=汇总表静默混算=摧毁"全10种子统一施"方法论红线且从聚合数看不出来。
  T5 algo 守卫（对抗审 HIGH-2）：STEP4E_WARMSTART_CKPT 设了但 STEP4E_CONTINUOUS_ALGO≠ppo(默认 sac) → import-time fail-fast
     （否则 SAC 臂静默不施热启动、但 provenance 仍记路径=假溯源=违"训练流程如实可查"红线）。

运行：cd 代码 && PYTHONPATH=. STEP4E_MANIFEST=... STEP4E_BALANCED_DIR=... /opt/miniconda3/envs/trb/bin/python tests/test_warmstart_knob.py
"""
import os
import sys
import tempfile
import subprocess

import numpy as np
import torch as th

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import run_step4e as S
from trb_env.usv_sac_train import (make_continuous_safe_model, make_continuous_safe_ppo_model,
                                   assert_continuous_safe_ppo_caliber)

N_FAIL = 0


def ok(name, cond):
    global N_FAIL
    print(("[PASS] " if cond else "[FAIL] ") + name)
    if not cond:
        N_FAIL += 1


# ── 场景池（小集·冒烟够用；无 manifest 则跳过 T1-T3 只跑纯函数 T4）──
_MAN = os.environ.get("STEP4E_MANIFEST", "")
_BDIR = os.environ.get("STEP4E_BALANCED_DIR", "")
PATHS = None
if _MAN and os.path.exists(_MAN):
    _tr, _test, _i = S.load_manifest_split(_MAN, _BDIR or os.path.dirname(_MAN))
    PATHS = _test[:3]


def pol_all(m):
    return th.cat([p.detach().reshape(-1) for p in m.policy.parameters()])


def pol_mean(m):
    """均值策略签名（排除 log_std=探索σ·热启动灌均值不灌σ）。"""
    return th.cat([p.detach().reshape(-1) for n, p in m.policy.named_parameters() if "log_std" not in n])


def sig(v):
    return round(float(v.sum()), 6), round(float(v.norm()), 6), v.numel()


if PATHS:
    tmp = tempfile.mkdtemp()

    # ── 源模型（模拟真金标：已训 σ 超动作箱 → 验 log_std 重置生效）──
    src, srcv = make_continuous_safe_ppo_model(paths=PATHS, n_envs=2, seed=99, subproc=False)
    with th.no_grad():
        src.policy.log_std.data.copy_(th.log(th.tensor([0.062, 0.030])))   # =真金标 s1 实测 σ（超箱 ±0.048/±0.018）
    srcv.obs_rms.mean[:] = 7.0
    srcv.obs_rms.var[:] = 3.0
    base = os.path.join(tmp, "src_expert")
    src.save(base + ".zip")
    srcv.save(base + "_vecnorm.pkl")

    # ── T1 默认 bit-identical ──
    a, _ = make_continuous_safe_ppo_model(paths=PATHS, n_envs=2, seed=5, subproc=False)                       # 不传
    b, _ = make_continuous_safe_ppo_model(paths=PATHS, n_envs=2, seed=5, subproc=False, warmstart_ckpt=None)  # 显式 None
    _d = float((pol_all(a) - pol_all(b)).abs().max())
    ok(f"T1 默认 bit-identical：不传 vs 显式None 同seed policy max|Δ|={_d:.1e} <1e-9（默认关=整块不调）", _d < 1e-9)

    # ── T2 热启动真落地 ──
    w, wv = make_continuous_safe_ppo_model(paths=PATHS, n_envs=2, seed=5, subproc=False, warmstart_ckpt=base)
    ok("T2a 热启动后【均值策略】== 源均值策略（好盆地真灌入）", sig(pol_mean(w)) == sig(pol_mean(src)))
    ok("T2b 热启动后均值策略 != 新建同seed（非空操作·防静默假热启动）", sig(pol_mean(w)) != sig(pol_mean(a)))
    ok("T2c 热启动后 log_std == maker in-box 初值（重置探索σ·保 F1）",
       np.allclose(w.policy.log_std.detach().numpy(), a.policy.log_std.detach().numpy()))
    ok("T2d 热启动后 log_std != 源已训σ（不继承源探索水平）",
       not np.allclose(w.policy.log_std.detach().numpy(), src.policy.log_std.detach().numpy()))
    ok("T2e 热启动后 venv.obs_rms == 源 obs_rms（归一化坐标系真复制·否则策略喂错分布=白热启动）",
       np.allclose(np.asarray(wv.obs_rms.mean), np.asarray(srcv.obs_rms.mean))
       and np.allclose(np.asarray(wv.obs_rms.var), np.asarray(srcv.obs_rms.var)))
    _cal = True
    try:
        assert_continuous_safe_ppo_caliber(w, wv)
    except Exception as e:
        _cal = False
        print("   caliber 失败:", e)
    ok("T2f 热启动后 caliber 仍过（算法/net/gamma/colregs/σ-in-box 未被热启动破）", _cal)

    # ── T3 守卫 ──
    sac, sacv = make_continuous_safe_model(paths=PATHS, n_envs=1, seed=77, subproc=False)   # SAC 臂·同 obs 维 27
    sb = os.path.join(tmp, "src_sac")
    sac.save(sb + ".zip")
    sacv.save(sb + "_vecnorm.pkl")
    _raised = False
    try:
        make_continuous_safe_ppo_model(paths=PATHS, n_envs=2, seed=5, subproc=False, warmstart_ckpt=sb)
    except ValueError as e:
        _raised = ("结构" in str(e) or "键集" in str(e))
    ok("T3a 结构守卫：误指 SAC 存档（键集不相交）→ fail-fast（非静默假热启动·HIGH-1）", _raised)

    s34, s34v = make_continuous_safe_ppo_model(paths=PATHS, n_envs=2, seed=88, subproc=False, augment_rho=True)
    b34 = os.path.join(tmp, "src_aug34")
    s34.save(b34 + ".zip")
    s34v.save(b34 + "_vecnorm.pkl")
    _raised = False
    try:
        make_continuous_safe_ppo_model(paths=PATHS, n_envs=2, seed=5, subproc=False, warmstart_ckpt=b34)
    except (ValueError, RuntimeError):
        _raised = True
    ok("T3b obs 维守卫：34维(augment)源 → 27维目标 → fail-fast", _raised)

    _raised = False
    try:
        make_continuous_safe_ppo_model(paths=PATHS, n_envs=2, seed=5, subproc=False, warmstart_ckpt="/nonexistent/foo")
    except FileNotFoundError:
        _raised = True
    ok("T3c 缺文件守卫：源 .zip/_vecnorm.pkl 缺 → fail-fast", _raised)
else:
    print("[SKIP] T1-T3 需 STEP4E_MANIFEST/STEP4E_BALANCED_DIR（纯函数 T4/T5 仍跑）")

# ── T4 config_conflict 混写守卫（纯函数·无需场景）──
_rec = dict(steps=5_000_000, n_total=200, pool_size=None, n_seg=10)          # 旧记录：无 warmstart 字段
ok("T4a 向后兼容：旧记录(无warmstart字段)+默认run → 无冲突（不误伤现有金标 jsonl）",
   S.config_conflict([dict(_rec)], 5_000_000, 200, None, 10) == set())
ok("T4b 守卫：从零记录 + 热启动run 同TAG → 冲突（硬拒混写两种训练流程）",
   S.config_conflict([dict(_rec)], 5_000_000, 200, None, 10, warmstart_ckpt="/p/s1") != set())
_recw = [dict(_rec, warmstart_ckpt="/p/s1")]
ok("T4c 守卫：热启动记录 + 从零run 同TAG → 冲突", S.config_conflict(_recw, 5_000_000, 200, None, 10) != set())
ok("T4d 同源热启动 → 无冲突（不误报）",
   S.config_conflict(_recw, 5_000_000, 200, None, 10, warmstart_ckpt="/p/s1") == set())
ok("T4e 异源(s1 vs s3) → 冲突（不同源=不同训练流程）",
   S.config_conflict(_recw, 5_000_000, 200, None, 10, warmstart_ckpt="/p/s3") != set())
ok("T4f 🆕【指纹身份】同路径但源内容被换(sha 不同) → 冲突（第2轮审 HIGH#1：路径是指针·只比路径则同路径换源静默混写=破「全10种子统一同源」红线；run_step4e 传 _WARMSTART_ID=路径#zipsha+vnsha）",
   S.config_conflict([dict(_rec, warmstart_ckpt="/p/s1#aaaa+bbbb")], 5_000_000, 200, None, 10,
                     warmstart_ckpt="/p/s1#cccc+dddd") != set())
ok("T4g 🆕【指纹身份】同路径同内容 → 无冲突（不误报）",
   S.config_conflict([dict(_rec, warmstart_ckpt="/p/s1#aaaa+bbbb")], 5_000_000, 200, None, 10,
                     warmstart_ckpt="/p/s1#aaaa+bbbb") == set())
# 🆕 第2轮审 NIT：哨兵串位用例——所有硬编码元组末尾恒为 (…,False,None,None) → arr_slack 与 warmstart 位置对调仍会全绿。
#   用不同哨兵值钉死【两者在元组里的相对位置】：记录 arr_slack='SENT'/warmstart=None vs cur arr_slack=None/warmstart='SENT' 必须冲突。
ok("T4h 🆕 哨兵串位：arr_slack 与 warmstart 位置对调 → 必冲突（锁死元组位置·防未来加旋钮串位=比错字段的灾难性静默错判）",
   S.config_conflict([dict(_rec, arr_slack_start_deg="SENT", warmstart_ckpt=None)], 5_000_000, 200, None, 10,
                     arr_slack_start_deg=None, warmstart_ckpt="SENT") != set())

# ── T5 algo 守卫（import-time·子进程验）──
_env = dict(os.environ)
_env.pop("STEP4E_CONTINUOUS_ALGO", None)          # 默认 sac
_tmpck = tempfile.mkdtemp()
open(os.path.join(_tmpck, "x.zip"), "wb").close()
open(os.path.join(_tmpck, "x_vecnorm.pkl"), "wb").close()
_env["STEP4E_WARMSTART_CKPT"] = os.path.join(_tmpck, "x")
_p = subprocess.run([sys.executable, "-c", "import run_step4e"],
                    cwd=os.path.join(os.path.dirname(__file__), ".."), env=_env,
                    capture_output=True, text=True)
ok("T5 algo 守卫：warmstart 设了 + 默认 algo=sac → import-time fail-fast（防静默不施+假溯源·HIGH-2）",
   _p.returncode != 0 and ("热启动仅连续 PPO 臂" in (_p.stdout + _p.stderr)))

print("\n" + ("=" * 50))
print("✅ 全部通过" if N_FAIL == 0 else f"❌ {N_FAIL} 项失败")
sys.exit(1 if N_FAIL else 0)
