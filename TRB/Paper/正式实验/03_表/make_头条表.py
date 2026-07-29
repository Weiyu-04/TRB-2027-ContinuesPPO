# -*- coding: utf-8 -*-
"""头条表生成（`Paper/正式实验/03_表/`）—— 只读原始重评产物，**一个数字都不手抄**。

按 `03` L234-E④ 的订正，**同时出两版**：
  · **全部种子**（含没练成的）—— 这是主表
  · **仅练成的种子**（到达 ≥ 50）—— 崩种子会污染逐局平均，而且污染方向**每条臂还不一样**
    （对标的崩种子违规偏高、金标的崩种子违规偏低）⟹ 两版都报才诚实

跑法：
    python3 -B Paper/正式实验/03_表/make_头条表.py <重评目录> [输出.md]
    # 例：python3 -B Paper/正式实验/03_表/make_头条表.py 结果/结果0729-59臂同趟重评
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _common as C   # noqa: E402

COLS = [("n", "n", "{:d}"), ("健康", "健康", None), ("到达%", "到达", "{:.2f}"),
        ("碰撞%", "碰撞率", "{:.3f}"), ("违规/局", "违规", "{:.3f}"), ("让路", "让路", "{:.3f}"),
        ("直航", "直航", "{:.3f}"), ("转艏Δ", "转艏", "{:.5f}"), ("油门Δ", "油门", "{:.4f}"),
        ("紧急%", "紧急", "{:.2f}")]


def table(ba, healthy_only):
    hdr = "| 臂 | " + " | ".join(c[0] for c in COLS) + " |"
    sep = "|---|" + "---|" * len(COLS)
    lines = [hdr, sep]
    for name, _, in_head in C.ARM_SPECS:
        if not in_head or name not in ba:
            continue
        m = C.metrics(ba[name], healthy_only=healthy_only)
        if m is None:
            lines.append(f"| {name} | " + " | ".join("—" for _ in COLS) + " |")
            continue
        cells = []
        for _, k, fmt in COLS:
            cells.append(f"{m['健康']}/{m['n']}" if k == "健康" else fmt.format(m[k]))
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def per_seed_block(ba):
    out = ["\n## 逐种子明细（到达%·strict）\n"]
    for name, _, _ in C.ARM_SPECS:
        if name not in ba:
            continue
        m = C.metrics(ba[name])
        pairs = " · ".join(f"s{s}:{a:.2f}" for s, a in zip(m["种子"], m["逐种子到达"]))
        out.append(f"- **{name}**（n={m['n']}·练成 {m['健康']}）：{pairs}")
    return "\n".join(out)


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    d = os.path.abspath(sys.argv[1])
    rows, n_strict, n_grp = C.load_pass(d, expect_strict=C.FORMAL['n_strict'] if '正式' in os.path.basename(d) else None)     # fail-closed：分母/重名/锚点任一不过直接抛错
    ba = C.by_arm(rows)
    md = [f"# 头条表 · {os.path.basename(d)}",
          "",
          f"> 由 `Paper/正式实验/03_表/make_头条表.py` 从 `{os.path.basename(d)}/g*.json` **自动生成**，"
          "一个数字都没有手抄（协议 §1 铁律）。",
          f"> {C.budget_note(n_strict, C.FORMAL['steps'], C.FORMAL['ckpt'])}",
          f"> 自查通过：{n_grp} 组 strict 键**逐位相同**且 =={n_strict} · {len(rows)} 条臂无重名 · 锚点 {len(rows)}/{len(rows)} 全过。",
          "",
          "## 主表（全部种子）", "", table(ba, False), "",
          "## 稳健性：仅练成的种子（到达 ≥ 50）", "",
          "> `03` L234-E④：崩种子会污染逐局平均，且**污染方向每条臂不一样**（对标的崩种子违规偏高、金标的偏低）"
          "⟹ 凡逐局平均的指标必须两版都报。**写作定式**：先报「两边各练成几颗」，再报「只比练成的种子」。",
          "", table(ba, True), "",
          per_seed_block(ba), ""]
    txt = "\n".join(md)
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), f"表1_头条表_{os.path.basename(d)}.md")
    open(out, "w", encoding="utf-8").write(txt)
    print(txt)
    print(f"\n[已写入] {out}")


if __name__ == "__main__":
    main()
