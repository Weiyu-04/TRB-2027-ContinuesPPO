#!/usr/bin/env python3
"""B2 课程难度分档（攻种子不稳·`03` 课程学习设计 B1）。

原理（对症崩塌根因=弱进门信号）：
  r_goal 是位移势（一条到目标轨迹累加 ≈ C_REACH·起点距离 = 1.5·D）；终端进门奖励 +50 固定。
  → +50 占单局回报 ≈ 50/(1.5·D)。D 越短 → +50 占比越大 → 进门梯度越强 → 越容易学会"真进门"。
  崩塌种子=掉进"门口瞎晃赚位移奖励、不进门"的坑（进门信号被淹没）。
  课程：先喂【短距=进门信号强】场景让所有种子先学会"一定进门"，再上长距 → 治崩塌根因。

本脚本=零训练算力·只读场景（floor/probe 跑过即已缓存到 STEP4E_SDIR/T-{n}.xml）。
按【起点→目标距离】给每类(head_on/crossing)的 train/test 排序，取下 FRAC 档(短距=易)，
写出 manifest_curric_easy.json（**全集真子集**：易 train ⊂ 全 train、易 test ⊂ 全 test·无泄漏）。

用法（服务器）：
  STEP4E_SDIR=$HOME/trb/scenarios python 代码/make_curriculum.py
环境变量：
  CURRIC_SRC        源 manifest（默认 ~/trb/balanced_pool/manifest_hocr_200.json·小集 HO/CR）
  CURRIC_OUT        输出（默认 ~/trb/balanced_pool/manifest_curric_easy.json）
  CURRIC_EASY_FRAC  易档占比（默认 0.34=下 1/3·短距）
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from classify_scenarios import read_scenario      # 复用项目场景读取（返回 goal center·不重造）

SDIR = os.environ.get("STEP4E_SDIR", os.path.expanduser("~/trb/scenarios"))
SRC = os.environ.get("CURRIC_SRC", os.path.expanduser("~/trb/balanced_pool/manifest_hocr_200.json"))
OUT = os.environ.get("CURRIC_OUT", os.path.expanduser("~/trb/balanced_pool/manifest_curric_easy.json"))
FRAC = float(os.environ.get("CURRIC_EASY_FRAC", "0.34"))

man = json.load(open(SRC))
out = {}
miss = 0
print(f"源 manifest = {SRC}")
print(f"场景缓存    = {SDIR}")
print(f"易档占比    = {FRAC}  (易 = 起点→目标 短距 = 进门信号强)\n")

for typ in ("head_on", "crossing"):
    out[typ] = {"source": man[typ].get("source", "old-T")}
    for split in ("train", "test"):
        rows = []                                  # (T-id, 起点→目标距离)
        for n in [int(x) for x in man[typ][split]]:
            p = f"{SDIR}/T-{n}.xml"
            if not (os.path.exists(p) and os.path.getsize(p) > 1000):
                miss += 1
                continue
            try:
                ego_p, _, _, gc, *_ = read_scenario(p)
                if gc is None:
                    continue
                rows.append((n, float(np.linalg.norm(ego_p - gc))))
            except Exception as e:                 # noqa: BLE001
                print(f"  ⚠️ 读 T-{n} 失败: {e}")
        rows.sort(key=lambda r: r[1])              # 按距离升序（短在前=易）
        n_easy = max(1, int(round(len(rows) * FRAC)))
        out[typ][split] = [str(n) for n, _ in rows[:n_easy]]   # 易档=最短的 n_easy 个
        ds = [d for _, d in rows]
        if ds:
            cut = rows[n_easy - 1][1]
            print(f"  {typ}/{split}: n={len(ds)}  距离 min={min(ds):.0f}/中位={np.median(ds):.0f}/max={max(ds):.0f}m"
                  f"   → 易档 ≤{cut:.0f}m 取 {n_easy} 个")

if miss:
    print(f"\n⚠️ {miss} 个场景缺缓存（先跑过 floor/probe 会缓存到 STEP4E_SDIR；或核对 STEP4E_SDIR 路径）")

out["overtaking"] = {"source": "none", "train": [], "test": []}   # HO/CR-only
out["note"] = (f"curriculum-easy stage: bottom {FRAC} by start->goal distance from "
               f"{os.path.basename(SRC)}; 易=短距=进门信号强（治弱进门信号崩塌）")
json.dump(out, open(OUT, "w"), ensure_ascii=False, indent=1)

n_tr = len(out["head_on"]["train"]) + len(out["crossing"]["train"])
n_te = len(out["head_on"]["test"]) + len(out["crossing"]["test"])
print(f"\n✅ 写出 {OUT}")
print(f"   易档 train = {n_tr}  (head_on {len(out['head_on']['train'])} + crossing {len(out['crossing']['train'])})")
print(f"   易档 test  = {n_te}  (head_on {len(out['head_on']['test'])} + crossing {len(out['crossing']['test'])})")
print("   下一步(B3·便宜验证)：在会崩种子上【只训易档】看崩塌降不降（命令待距离分布出来后给）。")
