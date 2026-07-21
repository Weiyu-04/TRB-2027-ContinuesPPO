#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""位置-only 训练策略的【严格门再评】+ 双指标 + 平凡解(绕圈扫框)守卫（`03` L186·复审 C3 抓的缺口）。

背景（为什么需要这一步）：
  run_posonly.sh 训练时 STEP4E_GOAL_IGNORE_ORIENT=1（位置-only 门）→ 回合【进框即终止】。
  →该 run 自己 final_per 里的 in_box_aligned_steps（严格·位置+朝向）被【截断】压到近 0（船没机会对齐就结束了）
    = 假性偏低·不能当"位置-only 训练也对不齐"的证据（会误读）。
  拿【公平】严格到达率 = 必须【另跑一次严格门 eval】：goal_ignore_orientation=False → 回合不在进框时结束、
    船有机会对齐或超时 → reached 才是真严格到达率。本脚本做这件事。

一次严格门 eval 同时给三样（金标诊断同款·evaluate 已算好）：
  · 严格到达%   = mean(reached)                 位置+朝向±9.74°（忠实 CommonOcean is_reached）
  · 位置-only%  = mean(in_box_steps>0)          位置进目标区（忽略朝向·忠实原文字面 "goal area"）
  · 平凡解守卫  = 对【位置进框但没严格到达】的局，看 max_speed / speed_reversals / speed_at_min
                 → 若多为"满速+高反转"=船在【绕圈扫框】(平凡解·[[methodology-correctness-not-just-syntax]])·
                   位置-only"成功"含金量低·须靠停车 stage-2 真停稳才算数。

用法(本机复算 / 服务器均可·先同步整个代码文件夹):
  PYTHONPATH=代码 \
  STEP4E_MANIFEST=$(pwd)/balanced_pool/manifest_hocr_200.json \
  STEP4E_BALANCED_DIR=$(pwd)/balanced_pool STEP4E_SDIR=$(pwd)/scenarios \
  CKPT_DIR=$(pwd)/结果/checkpoints CKPT_TMPL='Continuous-safe_s{s}_posonly_ppo_s{s}' SEEDS='5 6 2' \
  python 代码/posonly_strict_reeval.py
  （CKPT_DIR 指向 run_posonly.sh 产物；健康对照 s2 的 posonly 到达%应≈其严格金标~65%=坐实"位置-only 门不虚高"）

⚠️ 本脚本【纯 eval·不训练·不碰金标训练配方】。严格门=goal_ignore_orientation=False（默认真门）。
   盾/cone/vfloor/augment 逐字对齐金标(shield=True/cone=None/vfloor=2.0/augment=False)→ 与训练同分布。
"""
import os, sys, math, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))          # 代码/
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from trb_env.train import make_obs_transform
from trb_env.usv_continuous_shield import ContinuousProjectionEnv
from trb_env.usv_scenarios import load_scenario_pool
from trb_env.evaluate import evaluate_continuous
from run_step4e import load_manifest_split

CKPT_DIR  = os.environ["CKPT_DIR"]
CKPT_TMPL = os.environ.get("CKPT_TMPL", "Continuous-safe_s{s}_posonly_ppo_s{s}")
SEEDS     = [int(x) for x in os.environ.get("SEEDS", "5 6 2").split()]
MANIFEST  = os.environ["STEP4E_MANIFEST"]
_bdir     = os.environ.get("STEP4E_BALANCED_DIR") or os.path.dirname(MANIFEST)
_tr, test_paths, _i = load_manifest_split(MANIFEST, _bdir)
pool = load_scenario_pool(test_paths)
print(f"位置-only 策略·严格门再评 | 测试集 n={len(pool)} | 种子={SEEDS} | 门=严格(位置+朝向±9.74°·回合不在进框时终止)")
print(f"  金标严格到达%(参照): s0=92.5 s1=90 s2=65 s3=72.5 s4=82.5 s5=2.5 s6=0 s7=77.5 s8=70 s9=87.5\n")

def mk_strict_env(sc, pp):
    # 逐字对齐金标 eval env·唯一区别 = goal_ignore_orientation 默认 False = 严格真门（回合进框不终止·可对齐/超时）
    return ContinuousProjectionEnv(sc, pp, shield=True, goal_cone_half=None, goal_v_floor=2.0,
                                   augment_rho=False, goal_ignore_orientation=False)

print(f"{'seed':>4} | {'严格到达%':>8} {'位置only%':>9} {'差(门损)':>8} | 平凡解守卫(位置进框但没严格到达的局)")
print("-" * 100)
for s in SEEDS:
    base = os.path.join(CKPT_DIR, CKPT_TMPL.format(s=s))
    if not (os.path.exists(base + ".zip") and os.path.exists(base + "_vecnorm.pkl")):
        print(f"{s:>4} | ❌ 缺 checkpoint {base}(.zip/_vecnorm.pkl)"); continue
    # 重建 obs_transform（saved vecnorm·同 replay_eval 路径·augment=False→27维）
    _bv = DummyVecEnv([lambda: mk_strict_env(pool[0][0], pool[0][1])])
    _vn = VecNormalize.load(base + "_vecnorm.pkl", _bv); _vn.training = False
    tf = make_obs_transform(_vn)
    model = PPO.load(base + ".zip", device="cpu")
    agg, per = evaluate_continuous(mk_strict_env, model, pool, obs_transform=tf)
    n = len(per)
    strict = sum(1 for e in per if e["reached"])
    posonly = sum(1 for e in per if e.get("in_box_steps", 0) > 0)
    # 平凡解守卫：位置进框但没严格到达的局(=门损局)——它们是"干净接近对不上朝向"还是"满速绕圈扫框"?
    gate_loss = [e for e in per if e.get("in_box_steps", 0) > 0 and not e["reached"]]
    if gate_loss:
        spins = sum(1 for e in gate_loss if e.get("max_speed_ms", 0) >= 8.5 and e.get("speed_reversals", 0) >= 10)
        mr = np.median([e.get("speed_reversals", 0) for e in gate_loss])
        msp = np.median([e.get("max_speed_ms", 0) for e in gate_loss])
        mhd = np.median([e.get("heading_err_at_min_deg", 0) or 0 for e in gate_loss])
        guard = f"门损{len(gate_loss)}局: 疑绕圈扫框{spins}(满速≥8.5+反转≥10)·反转中位{mr:.0f}·峰速中位{msp:.1f}·朝向误差中位{mhd:.0f}°"
    else:
        guard = "无门损局(位置到=严格到·干净)"
    coll = sum(1 for e in per if e["collided"])
    print(f"{s:>4} | {100*strict/n:>7.1f}% {100*posonly/n:>8.1f}% {100*(posonly-strict)/n:>+7.1f}% | {guard}  [碰撞{coll}]")

print("\n判读(方法论·[[orientation-removal-justified-paper-not-gospel]] + [[methodology-correctness-not-just-syntax]]):")
print("  · 主信号=位置-only%(=训练里程碑口径):崩种子 s5/s6 追平健康对照 s2 → '去朝向治崩'假设坐实。")
print("  · 严格到达%=公平严格数(回合不截断·可对齐/超时)·两个数都报=诚实底线(谁也说不出藏)。")
print("  · 🔴平凡解守卫:若崩种子位置-only% 高【但门损局多是'绕圈扫框'(满速+高反转)】→ 位置-only'成功'含金量低=")
print("     RL 只学会'扫过框'·真正停稳摆正要靠停车 stage-2→最终报【完整两阶段(RL+停车)在双判据】·别把 RL 单独位置-only 当终值。")
