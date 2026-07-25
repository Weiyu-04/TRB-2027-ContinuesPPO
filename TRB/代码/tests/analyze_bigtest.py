#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""均衡大集（`manifest.json` 测试 600）闭环 eval 结果的【泄漏感知 + 分型】判读器（`03` L209/L211）。

吃 `shield_certv2_eval.py --run` 产出的 jsonl（`gen_bigtest.jsonl` / `ws_bigtest.jsonl` …），出：
  ① 总体 / 分会遇类型（对遇·交叉·追越）
  ② **训练里真没见过 vs 训练期反复看过** 的对照
     —— 那 40 个 manifest 测试场景 ⊂ 大集测试集，而它们训练期每段都被评过、源种子是看着它挑的（`03` L192）
     ⟹ 不剔掉就是自我污染；剔掉后剩 **360 个对遇/交叉** 才是"真·一眼没见过"。
  ③ 逐种子（盯方差与崩·`03` 红线：只押聚合量、逐种子表加脚注）
  ④ 安全（碰撞/违规/动作来源分布）+ 失败去向 + 平滑度
  ⑤ 两个文件对比模式（新 vs 旧·逐种子配对 + 符号检验）

**场景序 → 类型/T-id 的映射**（`load_manifest_split` 拼接序：head_on.test + crossing.test 走 `_download`【保序】，
再接 overtaking.test）⟹ scn_idx 0-199 对遇 / 200-399 交叉 / 400-599 追越。
🔴 脚本会**自动做一致性验证**：追越段紧急步% 必须显著高于前两段（`03` L211：追越 ρ5 吸收态 ~84% vs 对遇/交叉 ~5%），
   对不上就报警——防止 manifest 换了、或下载缺额导致错位却看不出来。

用法：
  python 代码/tests/analyze_bigtest.py 结果/结果-A3-0723/gen_bigtest.jsonl
  python 代码/tests/analyze_bigtest.py 新.jsonl --vs 旧.jsonl          # 两个 run 逐种子配对对比
  可选 --manifest-dir（默认自动找 balanced_pool）
"""
from __future__ import annotations
import argparse
import json
import math
import os
import statistics as st
import sys
from collections import Counter, defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_TRB = os.path.dirname(os.path.dirname(_HERE))          # …/TRB


def find_balanced(explicit=None):
    for d in ([explicit] if explicit else []) + [
            os.path.join(_TRB, "balanced_pool"), "balanced_pool", "/root/trb/balanced_pool", "../balanced_pool"]:
        if d and os.path.exists(os.path.join(d, "manifest.json")):
            return d
    raise SystemExit("🔒 找不到 balanced_pool（要 manifest.json + manifest_hocr_200.json）→ 用 --manifest-dir 指定。")


def load_maps(bdir):
    """→ (scn_idx→类型, scn_idx→T-id 或 None, 训练键集合, 验证键集合)。"""
    big = json.load(open(os.path.join(bdir, "manifest.json"), encoding="utf-8"))
    sm = json.load(open(os.path.join(bdir, "manifest_hocr_200.json"), encoding="utf-8"))
    ho = [int(x) for x in big["head_on"]["test"]]
    cr = [int(x) for x in big["crossing"]["test"]]
    ot = [os.path.basename(str(x)) for x in big["overtaking"]["test"]]
    typ, tid = {}, {}
    for i, t in enumerate(ho):
        typ[i], tid[i] = "对遇", t
    for i, t in enumerate(cr):
        typ[200 + i], tid[200 + i] = "交叉", t
    for i, _f in enumerate(ot):
        typ[400 + i], tid[400 + i] = "追越", None
    tr = {int(x) for x in sm["head_on"]["train"]} | {int(x) for x in sm["crossing"]["train"]}
    va = {int(x) for x in sm["head_on"]["test"]} | {int(x) for x in sm["crossing"]["test"]}
    return typ, tid, tr, va


def em_pct(r):
    s = r.get("ep_src") or {}
    tot = sum(s.values())
    return (100.0 * s.get("emergency", 0) / tot) if tot else 0.0


def se(p, n):
    return 100 * math.sqrt(max(p / 100 * (1 - p / 100), 0.0) / n) if n else 0.0


def summ(rows, label, indent="  "):
    n = len(rows)
    if not n:
        print(f"{indent}{label:<26} （空）")
        return None
    a = 100.0 * sum(bool(r["arrived"]) for r in rows) / n
    print(f"{indent}{label:<26} n={n:>4}  到达 {a:5.2f}%±{se(a, n):.2f}  "
          f"碰撞 {100.0 * sum(bool(r['collided']) for r in rows) / n:.2f}%  "
          f"违规/局 {sum(r.get('violations', 0) for r in rows) / n:5.2f}"
          f"(让路{sum(r.get('giveway_violations', 0) for r in rows) / n:.2f}/直航{sum(r.get('standon_violations', 0) for r in rows) / n:.2f})  "
          f"紧急步 {sum(em_pct(r) for r in rows) / n:5.1f}%  兜底步/局 {sum(r.get('n_fallback', 0) for r in rows) / n:5.1f}")
    return a


def analyze(path, typ, tid, tr, va, quiet=False):
    """→ {'per_seed_unseen': {seed: 到达率}, 'unseen': [...], ...}（供对比模式复用）。"""
    L = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    modes = sorted({r.get("mode") for r in L})
    seeds = sorted({r["seed"] for r in L})
    idxs = {r["scn_idx"] for r in L}
    if not quiet:
        print("=" * 104)
        print(f"【文件】{path}")
        print(f"【规模】{len(L)} 局 = {len(seeds)} 种子 {seeds} × {len(idxs)} 场景（档位 {modes}）")

    # 🔴 映射一致性自检：追越段紧急步% 必须远高于对遇/交叉（`03` L211 机制特征）
    blk = {k: [em_pct(r) for r in L if lo <= r["scn_idx"] < hi]
           for k, (lo, hi) in {"对遇": (0, 200), "交叉": (200, 400), "追越": (400, 600)}.items()}
    m = {k: (sum(v) / len(v) if v else float("nan")) for k, v in blk.items()}
    ok = (m["追越"] > 40.0 > max(m["对遇"], m["交叉"]))
    if not quiet:
        print(f"【映射自检】紧急步% 对遇 {m['对遇']:.1f} / 交叉 {m['交叉']:.1f} / 追越 {m['追越']:.1f} → "
              + ("✅ 与 `03` L211 的吸收态特征吻合，分型可信" if ok else
                 "🔴 追越段没有出现高紧急步 → 场景序可能错位（manifest 换了？下载缺额？）**别信下面的分型**"))
        print("=" * 104)

    unseen = [r for r in L if typ.get(r["scn_idx"]) != "追越" and tid.get(r["scn_idx"]) not in va]
    seen = [r for r in L if tid.get(r["scn_idx"]) in va]
    leak_tr = [r for r in L if tid.get(r["scn_idx"]) in tr]

    if not quiet:
        print("\n■ 总体（含追越）")
        summ(L, "全部")
        print("\n■ 按会遇类型")
        by = defaultdict(list)
        for r in L:
            by[typ.get(r["scn_idx"], "未知")].append(r)
        for t in ("对遇", "交叉", "追越"):
            summ(by[t], t)
        summ(by["对遇"] + by["交叉"], "对遇+交叉（官方集口径）")

        print("\n■ 🔴 真没见过 vs 训练期看过（判泛化的关键对照）")
        a_un = summ(unseen, "真没见过（360×种子）")
        a_se = summ(seen, "训练期看过的 40")
        if a_un is not None and a_se is not None:
            print(f"    → 差 {a_se - a_un:+.2f}pt（正=看过的更高=验证集乐观偏差）")
        if leak_tr:
            print(f"    ⚠️ 训练场景泄漏进本池 {len(leak_tr)} 局（应为 0！）")

        print("\n■ 逐种子（对遇+交叉·只押聚合量·逐种子仅供盯崩/方差）")
        for s in seeds:
            u = [r for r in unseen if r["seed"] == s]
            rows = [r for r in L if r["seed"] == s]
            def rate(sub):
                return 100.0 * sum(bool(x["arrived"]) for x in sub) / len(sub) if sub else float("nan")
            print(f"    s{s}: 真没见过 {rate(u):5.2f}%   对遇 {rate([r for r in rows if typ.get(r['scn_idx']) == '对遇']):5.2f}%"
                  f"   交叉 {rate([r for r in rows if typ.get(r['scn_idx']) == '交叉']):5.2f}%"
                  f"   追越 {rate([r for r in rows if typ.get(r['scn_idx']) == '追越']):5.2f}%")

    per_seed = {}
    for s in seeds:
        u = [r for r in unseen if r["seed"] == s]
        if u:
            per_seed[s] = 100.0 * sum(bool(r["arrived"]) for r in u) / len(u)
    if per_seed and not quiet:
        v = sorted(per_seed.values())
        k = len(v)
        lo = int(math.floor(0.2 * k))
        iqm = sum(v[lo:k - lo]) / max(k - 2 * lo, 1)
        print(f"\n■ 聚合（真没见过·逐种子）：{[round(x, 2) for x in per_seed.values()]}")
        print(f"    Mean {st.mean(v):.2f}   IQM {iqm:.2f}   "
              f"std {st.stdev(v):.2f}   min {min(v):.2f}   max {max(v):.2f}" if k > 1 else f"    单种子 {v[0]:.2f}")

    if not quiet:
        print(f"\n■ 安全（全 {len(L)} 局）：碰撞 {sum(bool(r['collided']) for r in L)} 局")
        src = Counter()
        for r in L:
            src.update(r.get("ep_src") or {})
        tot = sum(src.values()) or 1
        print("    动作来源：" + "  ".join(f"{k}={v}({100 * v / tot:.1f}%)" for k, v in src.most_common()))
        ok_rows = [r for r in L if r["arrived"] and typ.get(r["scn_idx"]) != "追越"]
        if ok_rows:
            print("\n■ 平滑/控制（对遇+交叉·到达局）")
            for kk in ("ctrl_jerk_norm_mean", "yaw_incr_mean", "accel_incr_mean", "subgrid_yaw_frac", "subgrid_accel_frac"):
                vals = [r[kk] for r in ok_rows if r.get(kk) is not None]
                if vals:
                    print(f"    {kk:<24} {sum(vals) / len(vals):.4f}")
    return {"per_seed_unseen": per_seed, "n": len(L), "path": path}


def compare(a, b):
    """两个 run 的逐种子配对对比（a=新·b=旧）。"""
    common = sorted(set(a["per_seed_unseen"]) & set(b["per_seed_unseen"]))
    print("\n" + "=" * 104)
    print(f"■ 配对对比（真没见过的场景上）：{os.path.basename(a['path'])}  vs  {os.path.basename(b['path'])}")
    if not common:
        print("  两个文件没有共同种子 → 无法配对")
        return
    d = [a["per_seed_unseen"][s] - b["per_seed_unseen"][s] for s in common]
    for s in common:
        print(f"    s{s}: {a['per_seed_unseen'][s]:6.2f}%  vs {b['per_seed_unseen'][s]:6.2f}%   Δ {a['per_seed_unseen'][s] - b['per_seed_unseen'][s]:+6.2f}pt")
    up = sum(1 for x in d if x > 0)
    print(f"    → 平均 Δ {st.mean(d):+.2f}pt   {up} 升 / {sum(1 for x in d if x < 0)} 降 / {sum(1 for x in d if x == 0)} 平（n={len(d)}）")
    try:
        from scipy import stats as sp
        print(f"    符号检验 p={sp.binomtest(up, len(d)).pvalue:.3f}   Wilcoxon p={sp.wilcoxon([a['per_seed_unseen'][s] for s in common], [b['per_seed_unseen'][s] for s in common]).pvalue:.3f}")
    except Exception as e:                                  # noqa: BLE001 —— scipy 缺失不该拖垮判读
        print(f"    （统计检验跳过：{e}）")
    print("  ⚠️ n=5 种子的检验功效很低，**别把 p 值当定论**；主看逐种子方向是否一致 + 幅度。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--vs", default=None, help="对比的另一个 jsonl（旧 run）")
    ap.add_argument("--manifest-dir", default=None)
    args = ap.parse_args()
    bdir = find_balanced(args.manifest_dir)
    typ, tid, tr, va = load_maps(bdir)
    a = analyze(args.jsonl, typ, tid, tr, va)
    if args.vs:
        b = analyze(args.vs, typ, tid, tr, va)
        compare(a, b)


if __name__ == "__main__":
    sys.exit(main())
