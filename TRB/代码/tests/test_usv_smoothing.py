#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""`trb_env/usv_smoothing.py` 回归（`03` L228）——训练期与评估期共用的转向平滑算子。

**本文件最要紧的一条 = T4「训练期算子与评估期包装器逐位一致」**：
整个"把限速放进训练循环"的实验，前提就是**训练时施加的和评估时施加的是同一个东西**
（train what you deploy）。两边一旦漂移，这个实验从根上就废了，而且**从聚合数字上完全看不出来**。

跑：python 代码/tests/test_usv_smoothing.py
"""
import os
import sys

import types

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.dirname(_HERE)
for _p in (_CODE, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from trb_env.usv_smoothing import apply_lowpass, apply_slew_limit, clamp_reference, smooth_yaw  # noqa: E402

W = 0.018
_fail = 0


def ok(name, cond, extra=""):
    global _fail
    if not cond:
        _fail += 1
    print(f"[{'PASS' if cond else 'FAIL'}] {name}  {extra if not cond else ''}")


# ── T1 不施加档必须【逐位】等价（对照组干净性 + 默认关 bit-identical 的基础）──
u0 = np.array([0.031, 0.0123456789])
ok("T1a 两个旋钮都不给 → 原样返回同一对象（默认关=一行都不执行）", smooth_yaw(u0, 0.01, W) is u0)
_bad = [(kw, prev) for kw in ({"slew_frac": 1.0}, {"lowpass_alpha": 1.0})
        for prev in (None, -W, W, 0.0, 0.03, -0.03)
        if float(smooth_yaw(np.array([0.031, 0.0123456789]), prev, W, **kw)[1]) != 0.0123456789]
ok("T1b 系数=1.0 / α=1.0 → 转向逐位不变（含参考在箱边/物理满程）", not _bad, str(_bad[:3]))
ok("T1c 油门那一维永不被改动",
   all(float(smooth_yaw(np.array([0.031, x]), 0.0, W, slew_frac=0.25)[0]) == 0.031
       for x in (-W, 0.0, W)))

# ── T2 永不越箱（`03` L222-#5 那个自引入 bug 的靶）──
_over = [(kw, prev, raw) for kw in ({"slew_frac": f} for f in (0.25, 0.33, 0.5))
         for prev in (0.03, -0.03, 0.024, -0.024)
         for raw in (W, -W, 0.0, 0.009)
         if abs(float(smooth_yaw(np.array([0.0, raw]), prev, W, **kw)[1])) > W + 1e-9]
_over += [(kw, prev, raw) for kw in ({"lowpass_alpha": a} for a in (0.3, 0.5, 0.7))
          for prev in (0.03, -0.03) for raw in (W, -W, 0.0)
          if abs(float(smooth_yaw(np.array([0.0, raw]), prev, W, **kw)[1])) > W + 1e-9]
ok("T2a 参考取物理满程 ±0.03 时输出仍在 ±0.018 内（参考先夹回箱内）", not _over, str(_over[:3]))
ok("T2b clamp_reference 把物理满程投影回 RL 箱",
   clamp_reference(0.03, W) == W and clamp_reference(-0.03, W) == -W and clamp_reference(0.007, W) == 0.007)

# ── T3 旋钮真的起作用（防"改完等于没改"）──
ok("T3a 半箱限速：满左→满右要两步",
   abs(apply_slew_limit(-W, W, 0.5, W) - 0.0) < 1e-12 and apply_slew_limit(-W, 0.0, 0.5, W) == -W)
ok("T3b 低通是凸组合（落在参考与原始之间）",
   min(0.0, W) <= apply_lowpass(W, 0.0, 0.5) <= max(0.0, W))
ok("T3c 首步（参考=None）不施加", apply_slew_limit(W, None, 0.25, W) == W and apply_lowpass(W, None, 0.3) == W)

# ── T4 🔴 训练期算子 vs 评估期包装器【逐位一致】（本文件的命门）──
os.environ.setdefault("REEVAL_MANIFEST_DIRS", os.path.join(os.path.dirname(_CODE), "balanced_pool"))
import importlib  # noqa: E402

RO = importlib.import_module("reeval_official")


class _Model:
    def __init__(self):
        self.w = 0.0

    def predict(self, obs, deterministic=True, **kw):
        return np.array([0.0, self.w]), None


class _Env:
    """假 env。🔴 `last_action` 必须是**每个实例独立**的：第一版写成类属性，
    结果四个档位的循环互相串状态（上一档最后一步的施加值漏进下一档的第 0 步），
    表现为"低通对不上、限速对得上"——查了两轮才定位。**契约测试连自己的脚手架一起考。**"""

    def __init__(self):
        self.env = types.SimpleNamespace(last_action=np.array([0.0, 0.0]))


def _contract(make_pol, kwname, val):
    m, e = _Model(), _Env()
    pol = make_pol(m, val)
    pol.bind_env(e)
    rng = np.random.default_rng(0)
    # 🔴 首步参考 = 0.0，不是 None：真 env 在 reset 时把 `last_action` 置 [0,0]（舵居中），
    #   所以首步是"相对居中舵位"施加平滑——这在物理上正确（真舵不可能一步从居中打到满舵）。
    #   第一版这里写成 None，于是 α=0.3/0.5/0.7 各出现 1 处不一致；查下来是**脚手架错**、不是代码漂移
    #   （限速档恰好因为首步幅度小于窗口而没暴露 ⟹ 低通把它兜出来了 = 这条契约测试的价值）。
    prev, diff = 0.0, 0
    for _ in range(400):
        raw = float(rng.uniform(-W, W))
        m.w = raw
        got = float(pol.predict(np.zeros(27, np.float32))[0][1])
        exp = float(smooth_yaw(np.array([0.0, raw]), prev, W, **{kwname: val})[1])
        diff += (got != exp)
        # 模拟盾"偶尔改写施加值"（含越箱），下一步的参考取施加值
        applied = got if rng.random() > 0.1 else float(rng.choice([0.03, -0.03, 0.024]))
        e.env.last_action = np.array([0.0, applied])
        prev = applied
    return diff


_d = sum(_contract(RO.YawSlewLimitPolicy, "slew_frac", f) for f in (0.25, 0.33, 0.5, 1.0))
ok("T4a 限速：训练期算子 与 评估期包装器 1600 次调用逐位一致（含盾改写/越箱施加值）", _d == 0, f"不同 {_d} 次")
_d2 = sum(_contract(RO.YawLowPassPolicy, "lowpass_alpha", a) for a in (0.3, 0.5, 0.7, 1.0))
ok("T4b 低通：同上", _d2 == 0, f"不同 {_d2} 次")

# ── T5 fail-fast：两个旋钮不许同开（同开则平顺度改善归因不清）──
try:
    smooth_yaw(np.array([0.0, 0.0]), 0.0, W, slew_frac=0.5, lowpass_alpha=0.5)
    ok("T5 两个旋钮同开 → 拒绝", False)
except ValueError:
    ok("T5 两个旋钮同开 → 拒绝", True)

print("\n" + ("✅ 全部 PASS" if _fail == 0 else f"❌ {_fail} 项 FAIL"))
sys.exit(1 if _fail else 0)
