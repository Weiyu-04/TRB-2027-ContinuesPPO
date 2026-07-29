# -*- coding: utf-8 -*-
"""正式实验 · **验证集挑最佳存档**（`03` L240 · user 2026-07-29 拍板的口径）。

═══ 为什么要它 ═══════════════════════════════════════════════════════════════
主存档是**覆盖式**的：只留最后一段。而 `03` L236-B 实测：46 条臂里有一条
（`Discrete-safe s0`）**第 7 段已到 100%、最后两段崩回 5%**，我们评的偏偏是末段
⟹ 评到了它最差的时刻。开了 `STEP4E_KEEP_SEGMENTS=1` 之后每段都留档，这个工具
就是把"用哪一段"这件事从**人工挑**变成**可复算的产物**。

═══ 选取规则（**先定死·不带任何旋钮·对全部臂完全一致**）═══════════════════════
  1. 候选 = 这条臂全部段（分段副本齐全的那些）
  2. **单一指标 = 验证集到达率**，取最高的那一段
  3. **平局取段号更小的**（更早 = 更保守 = 更少的选择偏倚）
  4. 就这三条，没了

🔴 **为什么不加"碰撞率不能太高"之类的硬门槛**：任何阈值都是旋钮，都会被问"这个数
   怎么来的"。单一指标 + 明确平局规则是教科书做法，一句话就能写进论文。
   代价是理论上可能选到"到达高但碰撞也高"的段 ⟹ **本工具会把这种情况直接报警打出来**
   （规则不变，但人必须看见），并且**强制要求同时报末段存档那一版**（见下）。

🔴 **必须同时报末段存档版**：`03` L236-A 实算，验证集 N=100 时到达率的二项标准误
   ≈ 4 个百分点；在 20 个候选里取最大，若真值持平，期望上抬 ≈ 1.87σ ≈ **7.5 个百分点**。
   这个选择偏倚**比我们专门立规矩防的 1.6pt 跨趟抖动大 4 倍多**。
   ⟹ 论文里必须两版都报，并把这段偏倚量化写出来。本工具同时输出两版清单。

🔴 **绝不能用测试集挑存档** —— 那是拿考题选答案。这里读的 `trend` 是训练期在
   **官方 1400 里切出的 100 个验证场景**上评的，与测试 600 零交集。

═══ 🆕 预算截取（`03` L242 · user 2026-07-29 要求"弹性、不返工"）═══════════════
`BUDGET_SEG=N` ⟹ **只在前 N 段里挑**。因为分段存档每 507,904 步留一份，
"如果 8M 大家就够了，那就截到 8M 报数"这件事**天然可做、零返工**：

    BUDGET_SEG=16 …   ⟹ 报「8.13M 步预算」（16 × 507,904 = 8,126,464）
    BUDGET_SEG=20 …   ⟹ 报「10.16M 步预算」（默认 = 全部段）

🔴 **同一张表里所有臂必须用同一个 BUDGET_SEG**，否则就不是同预算比较了。
🔴 报数时预算写实际值（段数 × 507,904），别写名义的"10M"。

═══ 用法 ═══════════════════════════════════════════════════════════════════
    python3 -B 代码/tests/select_best_ckpt.py <结果目录> [输出.json]
    BUDGET_SEG=16 python3 -B 代码/tests/select_best_ckpt.py <结果目录> 表_8M.json
产物 = 一份 JSON（进版本控制），含每条臂每颗种子选中的段号、该段与末段的指标、
以及可直接喂给 `reeval_official.py` 的 `REEVAL_CKPTS` 字符串。
"""
import glob
import json
import os
import re
import sys

METRIC = "到达率%"          # 单一选取指标（验证集）
SEG_STEPS = 507904          # 每段实际步数（= ceil(名义/16384)×16384·PPO 粒度 2048×8）
WATCH = ("碰撞率%", "违规次数/局")   # 不参与选取，但异常要报警


#: 🔴 `03` L243-续8（复审 D 线 O1）：**只挑正式臂**。`_runs()` 原来是全库 glob ⟹ 本地实测捞到
#   **270 条历史臂、正式臂 0 条**；服务器上历史臂也开着 `KEEP_SEGMENTS` ⟹ 它们的副本存在
#   ⟹ 会被塞进 `REEVAL_CKPTS`，`PASS=formal_best` 就会连着几十条**小集 563 臂**一起评
#   ⟹ 泄漏取并集、分母掉回 563。虽然 `load_pass` 的分母闸挡得住，但要**烧掉一两小时机时才炸**。
ARM_FILTER = os.environ.get("ARM_FILTER", "F240")      # 置空 = 不过滤（复现历史产物时用）


def _runs(root):
    """找出每个 run 的末段 sidecar（它的 trend 是完整的全部段）。"""
    out = {}
    for p in glob.glob(os.path.join(root, "**", "checkpoints", "*.progress.json"), recursive=True):
        if os.sep + "segments" + os.sep in p:          # 分段副本的 sidecar 只含到该段为止，不用
            continue
        if ARM_FILTER and ARM_FILTER not in os.path.basename(p):
            continue
        base = p[:-len(".progress.json")]
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        if d.get("trend"):
            out[os.path.basename(base)] = (base, d)
    return out


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    root = os.path.abspath(sys.argv[1])
    runs = _runs(root)
    if not runs:
        raise SystemExit(f"🔒 {root} 下没找到任何带 trend 的 checkpoint sidecar")
    cap = int(os.environ.get("BUDGET_SEG", "0"))     # 0 = 不截取（用全部段）
    sel, warn, missing = {}, [], []
    print(f"# 验证集挑最佳存档 · {os.path.basename(root)}\n")
    print(f"规则：验证集**{METRIC}**最高；平局取**更早**的段。无其它旋钮。")
    if cap:
        print(f"🔴 **预算截取**：只在前 {cap} 段里挑 ⟹ 报数预算 = {cap} × {SEG_STEPS:,} = "
              f"**{cap*SEG_STEPS:,} 步**（同一张表所有臂必须用同一个 BUDGET_SEG）。")
    print()
    print(f"{'run':<44}{'段/共':>8}{'选中'+METRIC:>12}{'末段'+METRIC:>12}{'抬升':>8}   备注")
    for name in sorted(runs):
        base, d = runs[name]
        tr = d["trend"][:cap] if cap else d["trend"]      # 🆕 预算截取
        vals = [t.get(METRIC) for t in tr]
        if any(v is None for v in vals):
            missing.append(name)
            continue
        best = max(range(len(tr)), key=lambda i: (vals[i], -i))   # 最大值·平局取更小 i
        # 🔴 `03` L243-续8（独立复审 D 线抓出·**差一 bug**）：段号是 **0 起**的，不是 1 起。
        #   `run_step4e.py:1247/1575` 是 `for c in range(n_seg)`，`:1270/:1598` 传 `seg_done=c`，
        #   `:831` 命名 `@s{seg_done:02d}` ⟹ 磁盘上是 `@s00 … @s{n_seg-1}`。
        #   而 `trend.append(row)` 在 `save_segment_checkpoint` **之前** ⟹ 存第 c 段时 trend 有 c+1 条
        #   ⟹ **`trend[i]` 与 `@s{i:02d}` 是同一段**。
        #   原来写 `best+1` 的后果有两条，都很贵：
        #     ① 评的是"最佳段的**下一段**"——论文写"验证集最佳存档"，评的却是别的模型；
        #     ② `best = len(tr)-1`（末段最好）⟹ `@s{n_seg}` 永不存在 ⟹ 判"副本缺失" ⟹
        #        **该 run 从最佳存档清单里静默消失**，而"一路在爬、末段最好"正是最该报的 run。
        #   ⚠️ 这个 bug 被一个**假夹具**掩盖过：`test_keep_segments.py:116` 直接调 `archive(base, 1)`，
        #      绕开了真正的调用方，所以那条断言给的是假保证。
        seg_no = best                                              # 段号 0 起，与 seg_done=c 一致
        segp = os.path.join(os.path.dirname(base), "segments", f"{name}@s{seg_no:02d}")
        ok = os.path.exists(segp + ".zip")
        if not ok:
            missing.append(f"{name}（选中第 {seg_no} 段〔0 起〕，但副本不存在：{segp}.zip）")
        else:
            # 🔴 硬校验：副本自带的 sidecar 的 trend 长度必须 == best+1，否则就不是这一段的模型。
            #   这一条直接把"段号对不对"变成可验证的，而不是靠人记住 0 起还是 1 起。
            try:
                _n = len((json.load(open(segp + ".progress.json", encoding="utf-8")) or {}).get("trend") or [])
            except Exception as _e:
                _n = -1
            if _n != best + 1:
                raise SystemExit(
                    f"🔒 {name}：选中第 {seg_no} 段（0 起），但副本 sidecar 的 trend 长度 = {_n}，应为 {best+1}"
                    " ⟹ **段号与存档对不上，选出来的不是那一段的模型**，中止。"
                    "（若是老产物用 1 起命名，请先统一命名，别改这里的判据。）")
        note = "" if ok else "🔴 副本缺失"
        # 报警：选中段在不参与选取的指标上明显差于末段
        for w in WATCH:
            a, b = tr[best].get(w), tr[-1].get(w)
            if a is not None and b is not None and a > b + max(0.5, abs(b) * 0.5):
                warn.append(f"{name}: 选中段的「{w}」= {a:.3f}，比末段 {b:.3f} 明显差")
                note = (note + " ⚠️ 见报警") if note else "⚠️ 见报警"
        sel[name] = {"seg": seg_no, "n_seg": len(tr), "path": segp, "副本存在": ok,
                     "选中": tr[best], "末段": tr[-1]}
        print(f"{name:<44}{f'{seg_no}/{len(tr)}':>8}{vals[best]:>12.2f}{vals[-1]:>12.2f}"
              f"{vals[best]-vals[-1]:>+8.2f}   {note}")

    up = [v["选中"][METRIC] - v["末段"][METRIC] for v in sel.values()]
    if up:
        print(f"\n【选择偏倚实测】平均抬升 {sum(up)/len(up):+.2f} 点 · 最大 {max(up):+.2f} 点 · "
              f"{sum(1 for x in up if x > 0)}/{len(up)} 条被抬高")
        print("  ⚠️ 这就是为什么必须同时报末段存档那一版（`03` L236-A：20 个候选取最大，理论期望上抬 ≈7.5 点）。")
    if warn:
        print("\n🔴 报警（规则不变，但你必须看见）：")
        for w in warn:
            print("   ·", w)
    if missing:
        print(f"\n🔴 缺产物 {len(missing)} 条（**分段存档没跑起来的话这里会全中**）：")
        for m in missing[:10]:
            print("   ·", m)

    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(root, "best_ckpt_selection.json")
    json.dump({"规则": f"验证集 {METRIC} 最高，平局取更早的段；无其它旋钮",
               "指标": METRIC, "预算段数": cap or "全部",
               "预算步数": (cap * SEG_STEPS) if cap else "全部段",
               "选取": sel,
               "REEVAL_CKPTS_最佳存档": ",".join(v["path"] for v in sel.values() if v["副本存在"]),
               "REEVAL_CKPTS_末段存档": ",".join(runs[k][0] for k in sel),
               "臂过滤": ARM_FILTER or "（无）",
               "⚠️": "两串都只含通过 ARM_FILTER 的臂；最佳存档串还额外只含副本齐全的。"},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n[已写入] {out}")
    print("  两串 REEVAL_CKPTS 都在里面：**最佳存档**与**末段存档**各评一趟，论文两版都报。")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
