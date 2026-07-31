#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""通过门 4 + 门 5 正式实验（纯评估 · 不碰 代码/ · 不占训练机）。

三个场景池：
  A 官方场景        —— 对照组，说明"温和基准上兜底几乎不触发"
  B 对抗·紧急密集   —— T_c ∈ [80,240]s，真碰撞航向且落在紧急预测视界内 ⟹ 逼出紧急通道与兜底
  C 对抗·让路       —— T_c ∈ [250,400]s，真碰撞航向但超出紧急视界 ⟹ 产生真正的让路落点

三项对照：
  ① 门 5：A vs B 的兜底分支触发率（放松 / 碰撞风险最小化 / 退化 / 紧急控制器）
  ② 让路入口：C 上 paper vs symmetric（对标原设定 vs 本文改进）
  ③ 门 4：C 上终端约束 off / discrete / certv2 三档（用 symmetric 保证有让路落点）
"""
import os
import sys
import json
import argparse
from collections import Counter

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_TRB = os.environ.get("TRB_ROOT", "/home/user/TRB-2027-ContinuesPPO/TRB")
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_TRB, "代码"))

from make_adversarial import generate                          # noqa: E402
from trb_env.usv_scenarios import load_scenario_pool           # noqa: E402
from trb_env.usv_continuous_shield import ContinuousProjectionEnv  # noqa: E402

HOLD = np.array([0.0, 0.0], dtype=np.float32)   # 标称=保向保速，把盾的行为完全暴露出来


def run(pairs, *, gw_entry="paper", terminal_mode=None, max_steps=80):
    kw = dict(gw_entry=gw_entry)
    if terminal_mode is not None:
        kw.update(recursive_feasibility=True, terminal_mode=terminal_mode)
    src, rho = Counter(), Counter()
    n_step = n_ep = n_coll = 0
    for sc, pp in pairs:
        try:
            env = ContinuousProjectionEnv(sc, pp, shield=True, **kw)
            obs, _ = env.reset(seed=0)
        except Exception:
            continue
        n_ep += 1
        for _ in range(max_steps):
            try:
                obs, r, term, trunc, info = env.step(HOLD)
            except Exception:
                break
            n_step += 1
            if info.get("source"):
                src[info["source"]] += 1
            if info.get("rho") is not None:
                rho[info["rho"]] += 1
            if term or trunc:
                if (info.get("flags") or {}).get("collision"):
                    n_coll += 1
                break
    gw = sum(v for k, v in rho.items() if k in (2, 3, 4))
    return dict(episodes=n_ep, steps=n_step, collisions=n_coll,
                giveway_steps=gw, source=dict(src), rho={str(k): v for k, v in rho.items()},
                gw_entry=gw_entry, terminal_mode=terminal_mode)


def show(name, r):
    t = max(1, r["steps"])
    p = {k: round(100 * v / t, 2) for k, v in sorted(r["source"].items(), key=lambda x: -x[1])}
    print(f"  {name:38s} 回合{r['episodes']:3d} 步{r['steps']:5d} 碰撞{r['collisions']:3d} "
          f"让路步{r['giveway_steps']:4d} | {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="每池场景数")
    ap.add_argument("--out", default=os.path.join(_HERE, "gate45_final.json"))
    args = ap.parse_args()
    import glob
    paths = sorted(glob.glob(os.path.join(_TRB, "scenarios", "T-*.xml")),
                   key=lambda q: int(q.split("T-")[1].split(".")[0]))

    out = {}
    print("【池 A】官方场景（对照）")
    A = load_scenario_pool(paths[:args.n])
    out["A_official"] = run(A)
    show("官方场景", out["A_official"])

    print("\n【池 B】对抗·紧急密集（T_c 80–240s）")
    advB = generate(paths[:8], betas=list(range(-170, 180, 10)),
                    T_cs=[80., 120., 180., 240.], v_ms=[3., 5., 7.])
    B = [(g["scenario"], g["pp"]) for g in advB[::max(1, len(advB)//args.n)][:args.n]]
    out["B_adversarial_emergency"] = run(B)
    show("对抗·紧急密集", out["B_adversarial_emergency"])

    print("\n【池 C】对抗·让路（T_c 250–400s，超出紧急视界）")
    advC = generate(paths[:8], betas=list(range(-80, 10, 10)),
                    T_cs=[250., 300., 350., 400.], v_ms=[4., 6.])
    gwC = [g for g in advC if (g["meta"]["crossing"] or g["meta"]["head_on"])
           and not g["meta"]["is_emergency"]]
    C = [(g["scenario"], g["pp"]) for g in gwC[:args.n]]
    print(f"   可用 {len(gwC)} 个，取 {len(C)} 个")

    print("\n  ② 让路入口对照（对标原设定 vs 本文改进）")
    for e in ("paper", "symmetric"):
        r = run(C, gw_entry=e)
        out[f"C_entry_{e}"] = r
        show(f"让路入口={e}", r)

    print("\n  ③ 门 4：终端约束三档（gw_entry=symmetric）")
    for tm in (None, "discrete", "certv2"):
        r = run(C, gw_entry="symmetric", terminal_mode=tm)
        out[f"C_term_{tm}"] = r
        show(f"终端约束={tm}", r)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\n产物 → {args.out}")


if __name__ == "__main__":
    main()
