# -*- coding: utf-8 -*-
"""对标论文臂重训后的**训练可复现性检查**（`03` L236-D① 的白拿收益·零依赖）。

同种子 + 同配方重训一遍（`run_ds_reseg.sh`），把新的 `trend` 与 0702 那趟逐段比：
  · **完全相同** ⟹ 训练是确定性的 ⟹ 全项目所有"同种子配对"比较的地基被坐实（这是好消息，值钱）。
  · **不相同**   ⟹ 同种子也会漂 ⟹ **所有配对结论都要加噪声带**（这是必须知道的坏消息）。
两种结果都要如实记进 `03`，别只记好的那种。

顺带：把每颗种子的**峰值段**和**末段**都打出来 —— 这正是 L236-B 那个"练好了又崩回去"问题的量。

跑法：  python3 -B 代码/tests/check_ds_reproducible.py
        NEW_DIR=结果/xxx python3 -B 代码/tests/check_ds_reproducible.py    # 新产物不在默认位置时
"""
import glob
import json
import os
import sys

_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.abspath(os.path.join(_CODE, "..", "结果"))
OLD_DIR = os.environ.get("OLD_DIR", os.path.join(RES, "结果0702-地基第1版-12:18"))
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0 1 2 3 4").replace(",", " ").split()]


def _trend(path):
    with open(path, encoding="utf-8") as f:
        rec = json.loads(f.read().strip().split("\n")[-1])
    return rec["trend"], rec


def _find_new(seed):
    """新产物可能落在 结果/ 直挂，也可能被按批次归档进子目录 ⟹ 两种都找。"""
    pats = [os.path.join(RES, f"step4e_partial_dsSegS{seed}.jsonl"),
            os.path.join(RES, "*", f"step4e_partial_dsSegS{seed}.jsonl")]
    if os.environ.get("NEW_DIR"):
        pats.insert(0, os.path.join(os.path.abspath(os.environ["NEW_DIR"]), f"step4e_partial_dsSegS{seed}.jsonl"))
    for p in pats:
        hit = sorted(glob.glob(p))
        if hit:
            return hit[0]
    return None


def main():
    print("===== 对标论文臂（Discrete-safe）重训 · 训练可复现性检查 =====")
    print(f"  旧产物目录：{OLD_DIR}")
    n_same = n_diff = n_miss = 0
    for s in SEEDS:
        old_p = os.path.join(OLD_DIR, f"step4e_partial_discStdW0_s{s}.jsonl")
        new_p = _find_new(s)
        if not os.path.exists(old_p) or not new_p:
            print(f"\n  s{s}: ⚠️ 找不到产物（旧={os.path.exists(old_p)} 新={bool(new_p)}）—— 跳过")
            n_miss += 1
            continue
        old_t, old_r = _trend(old_p)
        new_t, new_r = _trend(new_p)
        oa = [x["到达率%"] for x in old_t]
        na = [x["到达率%"] for x in new_t]
        same = oa == na
        n_same += same
        n_diff += (not same)
        print(f"\n  ── s{s} ── {'✅ 逐段完全相同' if same else '🔴 不相同'}")
        print(f"     旧(0702) : " + " ".join(f"{x:>5.1f}" for x in oa))
        print(f"     新(重训) : " + " ".join(f"{x:>5.1f}" for x in na))
        if not same:
            print(f"     逐段差   : " + " ".join(f"{b-a:>+5.1f}" for a, b in zip(oa, na)))
        pk_o, pk_n = max(oa), max(na)
        print(f"     峰值段 旧 {oa.index(pk_o)+1}（{pk_o:.1f}）· 新 {na.index(pk_n)+1}（{pk_n:.1f}）"
              f"   末段 旧 {oa[-1]:.1f} · 新 {na[-1]:.1f}"
              f"   {'← 🔴 这颗就是 L236-B 那颗“练好了又崩回去”' if pk_o - oa[-1] > 20 else ''}")
        # 配方漂没漂（除 keep_segments 外不该有别的差）
        SKIP = {"steps", "train_s", "eval_s", "fps", "ckpt", "curves", "trend",
                "final", "final_per", "eval_pct", "keep_segments"}
        keys = (set(old_r) | set(new_r)) - SKIP
        drift = [k for k in sorted(keys) if repr(old_r.get(k, "<缺>")) != repr(new_r.get(k, "<缺>"))]
        # 老 schema 缺的新键不算漂（那是后来才加进 run_config 的自描述键）
        drift = [k for k in drift if k in old_r]
        print(f"     配方漂移 : {drift if drift else '无（除 keep_segments 外逐键相同）'}")

    print("\n===== 结论 =====")
    print(f"  逐段完全相同 {n_same} / 比上了 {n_same + n_diff} 颗" + (f"（另有 {n_miss} 颗没产物）" if n_miss else ""))
    if n_diff == 0 and n_same:
        print("  ✅ 训练确定性坐实 ⟹ 全项目『同种子配对』比较的地基是硬的，可以写进论文方法节。")
    elif n_diff:
        print("  🔴 同种子也会漂 ⟹ **所有配对结论都要加噪声带**；先量出漂多少，再决定要不要多种子重复。")
        print("     （别只记好消息 —— 这条必须如实进 `03`。）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
