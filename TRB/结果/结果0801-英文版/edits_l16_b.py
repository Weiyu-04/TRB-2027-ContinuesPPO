# -*- coding: utf-8 -*-
"""2026-08-01 later-16 · 第 B 批：再腾 4 行，把图 7 放回去。

第 A 批之后第 20 页用了 39/54 行 ⟹ 空 15 行，而图 7 要 19 行，**还差 4 行**。
本批只挑 GPT 报告里**逐处核实确有重复**、且删掉不损失任何作用域声明的地方。

🔴 特别注意「收敛/未收敛」这条：全稿说了四遍，但**不是同一个断言**——
  · §5.2 说的是**主表**（未收敛按原样计入，限制到收敛后排序不变）→ 保留
  · §5.3 说的还是**主表** → 删（与 §5.2 同一件事）
  · §5.4 说的是**消融那一组**（另一批 run）→ 保留，删了就丢了一个不同的断言
  · §5.5 又抄了一遍 §5.2 → 删
GPT 建议 §5.3 §5.4 §5.5 三处全删，那会把 §5.4 的断言一起删掉，**没有采纳**。

用法：cd Paper/01_论文稿 && python3 ../../结果/结果0801-英文版/edits_l16_b.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tex_edit import Doc  # noqa: E402


def main():
    d = Doc("801-paper-英文版.tex")

    # §5.2 Checkpoint selection：加粗预告句，后面三句用数字把这件事说完了
    d.sub("is never accessed.\n\\textbf{The rule is itself biased.} The validation set holds only $100$ scenarios,",
          "is never accessed. The validation set holds only $100$ scenarios,")

    # §5.3：与 §5.2 同一个断言（主表的未收敛 run 怎么处理），第二次
    d.sub("shield versus Continuous, shield pair. A nonconverged run shifts the per-episode\n"
          "averages, and the direction of the shift differs among configurations; restricting the\n"
          "table to converged runs leaves the ordering unchanged.",
          "shield versus Continuous, shield pair.")

    # §5.5：逐字重复 §5.2 那一句
    d.sub("increase the number of runs meeting the criterion, as well as the final numbers.\n"
          "Nonconverged runs are counted in the main table as\nthey are.",
          "increase the number of runs meeting the criterion.")

    # §5.6 Online cost：二次规划的规模在摘要与 §4.2.2 已各写一次；
    #   机器描述里"32 个虚拟核"与"只用一个线程"自相矛盾，留一个线程那句就够
    d.sub("\\noindent\\textbf{Online cost.}\\ The projection is one quadratic program with two "
          "variables and at most six constraints.\n"
          "The timings below come from one thread of an AMD EPYC 9654 processor, on the Linux\n"
          "instance with $32$ allocated virtual cores that ran the evaluation. They cover $1{,}872$",
          "\\noindent\\textbf{Online cost.}\\ The timings below come from one thread of an "
          "AMD EPYC 9654 processor. They cover $1{,}872$")

    # §3.1：回合长度在 §5.1 场景说明里还写一次，那里更该有
    d.sub("\\citl{krasowski2024}{Krasowski and Althoff 2024}. An episode lasts at most $170$ steps,\n"
          "which is $1{,}700$~s. The circumscribed radius",
          "\\citl{krasowski2024}{Krasowski and Althoff 2024}. The circumscribed radius")
    d.sub("$300$~m long, and an episode lasts at most $170$ decision steps.",
          "$300$~m long, and an episode lasts at most $170$ decision steps, which is $1{,}700$~s.")

    # §3.2：连着两句都在说「观测维数固定」，两句都在说「只支持一条他船」
    d.sub("sector keeps the dimension independent of the number of target vessels. This keeps the "
          "observation dimension fixed, but it does not extend the shield beyond one\n"
          "target vessel. The shield deployed here is implemented for a single target vessel, and\n"
          "it raises an error rather than running in a degraded mode when more are present.",
          "sector keeps the observation dimension fixed. The shield is implemented for a single "
          "target vessel, and it raises an error rather than running in a degraded mode when more "
          "are present.")
    d.save()


if __name__ == "__main__":
    main()
