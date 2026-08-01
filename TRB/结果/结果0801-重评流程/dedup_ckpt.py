#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""合并多机产物之后的【同名存档去重】—— 默认只看不动，加 APPLY=1 才真搬。

为什么必须做：
  · select_best_ckpt.py:_runs() 把结果存进 out[basename] —— 按文件名建字典、不去重不告警，
    同名分处两个目录时**静默保留后扫到的那个**；
  · run_reeval_all.sh 闸门 0 找存档用 find ... -print -quit —— **第一个命中就返回**。
  两处都可能取到【半截训练】的那一份，而表面完全正常、事后查不出来。

规则（不带旋钮）：同名的几份里**段数最多的留下**，其余整套挪进 结果/_孤儿档/<原目录名>/。
  段数从各自的 .progress.json 里的 seg_done+1 读；读不到的一律判为 -1（一定输）。
  段数相同（跨机逐位一致时会这样）⟹ 保留路径字典序最小的那份，其余照样挪走，并标注出来。

用法：
    python3 dedup.py /root/trb/结果            只看
    APPLY=1 python3 dedup.py /root/trb/结果    真搬（挪走，不删）
"""
import json
import os
import shutil
import sys
import collections

SUFFIXES = (".zip", "_vecnorm.pkl", ".progress.json")


def seg_of(zip_path):
    sc = zip_path[:-len(".zip")] + ".progress.json"
    try:
        d = json.load(open(sc, encoding="utf-8"))
        return int(d.get("seg_done", -1)) + 1
    except Exception:
        return -1


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    root = os.path.abspath(sys.argv[1])
    apply_ = os.environ.get("APPLY") == "1"
    # 🔴 隔离区**必须放在 `结果/` 之外**（2026-08-01 实测）：
    #   · `select_best_ckpt.py:_runs()` 是 `glob(root/**/checkpoints/*.progress.json, recursive=True)`
    #     —— `**` 匹配任意深度 ⟹ 放在 `结果/_孤儿档/` 里照样被扫到，且它按 basename 建字典、
    #     **后扫到的覆盖前面的** ⟹ 挪走的半截档可能反杀回来，等于没隔离；
    #   · `check_formal_integrity.py` 第 ⑧ 项同理，会把隔离区里的又判成重名（硬伤）；
    #   · 放 `~/trb/_孤儿档/` 也不行 —— `run_reeval_all.sh` 的
    #     `REEVAL_CKDIRS=$ROOT/*/*/checkpoints` 正好能匹配到 `~/trb/_孤儿档/<原目录>/checkpoints`。
    #   ⟹ 默认挪到 `~/trb` 的**上一级**（即 `/root/_孤儿档`），三个工具都够不着。
    quarant = os.environ.get("QUARANTINE") or os.path.abspath(
        os.path.join(root, os.pardir, os.pardir, "_孤儿档"))

    found = collections.defaultdict(list)
    for dirpath, dirnames, filenames in os.walk(root):
        if os.path.basename(dirpath) != "checkpoints":
            continue
        if os.sep + "_孤儿档" + os.sep in dirpath + os.sep:
            continue
        for f in filenames:
            if f.endswith(".zip") and "_F240" in f:
                found[f].append(os.path.join(dirpath, f))

    dup = {k: v for k, v in found.items() if len(v) > 1}
    print(f"扫到 {len(found)} 个不同的正式存档名 · 其中 {len(dup)} 个出现多次")
    if not dup:
        print("✅ 没有重名，什么都不用做。")
        return 0

    print()
    print("每组里【段数最多】的留下，其余挪走（不是删）：")
    moves = []
    ties = 0
    for name in sorted(dup):
        cands = sorted((seg_of(p), p) for p in dup[name])
        best_seg = max(c[0] for c in cands)
        winners = [p for s, p in cands if s == best_seg]
        keep = sorted(winners)[0]
        if len(winners) > 1:
            ties += 1
        print(f"  {name}")
        for s, p in sorted(cands, key=lambda t: (-t[0], t[1])):
            mark = "✅ 保留" if p == keep else "→  挪走"
            print(f"      {s:>3} 段   {mark}   {os.path.relpath(p, root)}")
            if p != keep:
                moves.append(p)

    print(f"⟹ 保留 {len(dup)} 份 · 挪走 {len(moves)} 份"
          + (f" · 其中 {ties} 组段数相同（按路径字典序保留第一份）" if ties else ""))
    if not apply_:
        print("\n（只看模式。确认无误后：APPLY=1 python3 dedup.py " + root + "）")
        return 0

    n_file = 0
    for zp in moves:
        d = os.path.dirname(zp)
        base = zp[:-len(".zip")]
        name = os.path.basename(base)
        # 原目录名（checkpoints 的上一级），用来在隔离区里保留出处
        origin = os.path.basename(os.path.dirname(d)) or "根"
        dst_dir = os.path.join(quarant, origin, "checkpoints")
        os.makedirs(dst_dir, exist_ok=True)
        for suf in SUFFIXES:
            src = base + suf
            if os.path.exists(src):
                shutil.move(src, os.path.join(dst_dir, name + suf))
                n_file += 1
        segdir = os.path.join(d, "segments")
        if os.path.isdir(segdir):
            dst_seg = os.path.join(dst_dir, "segments")
            os.makedirs(dst_seg, exist_ok=True)
            for f in sorted(os.listdir(segdir)):
                if f.startswith(name + "@s"):
                    shutil.move(os.path.join(segdir, f), os.path.join(dst_seg, f))
                    n_file += 1
    print(f"\n✅ 已挪走 {n_file} 个文件 → {quarant}（**是挪走不是删**，随时可回溯）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
