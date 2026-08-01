# -*- coding: utf-8 -*-
"""2026-08-01 later-16 · 第 A 批：删元话语与真重复 + 收段首加粗标签。

来源：GPT 的 68 条删减报告 + user 转来的「怎么去掉 AI 味」判据。
**没有照单全收**：先跑 `ai_flavor.py` 与对标论文同口径量了六项，
只对**量出来确实中招**的两项（句首结构、段首加粗标签）和**逐处核实确有重复**的地方动手。
驳回的条目与理由记在 `03` L243-续63。

本批做三件：
  ① 删纯元话语（讲论文自己怎么排版的句子，不是论文内容）
  ② 删逐处核实过的**真**重复（同一段内第三次说同一件事）
  ③ 段首加粗标签 31 → 22：**整句型的**标签要么去粗变成正常首句、要么整个删掉；
     名词短语型的（Metrics / Fallback / Assumptions …）全部保留

用法：cd Paper/01_论文稿 && python3 ../../结果/结果0801-英文版/edits_l16_a.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tex_edit import Doc  # noqa: E402


def main():
    d = Doc("801-paper-英文版.tex")

    # ══════════ ① 元话语：讲论文自己怎么排版，不是论文内容 ══════════
    # §4 开头：图注已经说了图 1 是什么；命题的结构由三个加粗标签自明
    d.sub("Figure~\\ref{fig:overview} summarizes the mechanism and its two bypasses.\n"
          "\\noindent\\textbf{A note on the guarantees.}\\ Each proposition below is placed after the\n"
          "mechanism it applies to, and is given as a\n"
          "statement, its assumptions, a proof sketch, and its scope. The scope paragraph names the\n"
          "cases it does not cover.",
          "")
    # §5.1：告诉读者结论会再说一遍，属于导航话
    d.sub("therefore cannot be tested empirically here. The conclusions state this.",
          "therefore cannot be tested empirically here.")
    # §5.5 开头：第一句是图注内容；第二句留，但去掉 "as labeled per panel" 的排版说明
    d.sub("Figure~\\ref{fig:curves} combines training-rollout metrics, validation-checkpoint\n"
          "metrics, and convergence counts, as labeled per panel. It reports learning speed among\n"
          "the converged runs and the number of runs that meet the convergence criterion.",
          "Figure~\\ref{fig:curves} reports learning speed among the converged runs and the number "
          "of runs that meet the convergence criterion.")

    # ══════════ ② 逐处核实过的真重复 ══════════
    # §4.1：引言与 §3.3 已各说一次「凸集可投影所以不必枚举」，§4.2 还要正式定义
    d.sub("union. A set of this shape can be entered by projection, so the commands never have to be enumerated.",
          "union.")
    # §4.1：「满足阈值的最小转艏率」已经含「不能再低」
    d.sub("Under the adopted $20^\\circ$-within-$40$~s definition this is the smallest turn rate\n"
          "that meets the threshold, and for that definition the value cannot be lowered. A command\n"
          "must still satisfy the required direction and the conditions of the state machine.",
          "Under the adopted $20^\\circ$-within-$40$~s definition this is the smallest turn rate "
          "that meets the threshold.")
    # §4.2.3：上一句已说紧急控制器给出命令，本句只是换个说法再讲一遍
    d.sub("law converts each mode into a control command. Its purpose is a defined behavior once\n"
          "the emergency situation has been declared.",
          "law converts each mode into a control command.")
    # §4.3：同一段里「盾不是事后修正」正反说了三次，删第一次（后面两处措辞更精确）
    d.sub("The shield is not applied after the policy as an after-the-fact correction. The projection of\n"
          "Equation~\\eqref{eq:qp} is placed inside the environment transition.",
          "The projection of Equation~\\eqref{eq:qp} is placed inside the environment transition.")
    # §5.1：首句已给出 test = 600，末句再推一次
    d.sub("The two sets are disjoint, so the evaluation sample size is $600$.",
          "The validation and test sets are disjoint.")
    # §5.7：「把可辩护的主张收到一个维度上」在同一节说了三次，删第一次和第三次
    d.sub("(Table~\\ref{tab:ext}). The comparison identifies the dimensions that the three geometric\n"
          "controllers evaluated here already cover.",
          "(Table~\\ref{tab:ext}).")
    d.sub("The claim against the three geometric baselines evaluated here rests on this dimension\nalone. Ours records",
          "Ours records")
    # §5.7：原始值 0.53 对 4.49 已经说明差距，倍数是同一件事的第二遍
    d.sub("function at its lowest-violation setting, which is a factor of $8.5$.",
          "function at its lowest-violation setting.")
    # §6.2：前一句刚说「不随网格加密单调缩小」，这半句原样再说一次
    d.sub("so for these two grids the overshoot is not monotone in the spacing. A continuous",
          "A continuous")
    # §5.2：第二句只是说「图注会写明出处」——图注本来就写了；第一句有信息，并进上一段
    d.sub("the same under both rules.\n\n"
          "\\noindent\\textbf{Reading the figures.}\\ Violation counts in the figures are the sum of\n"
          "stand-on and give-way violations. Each caption states whether the panel comes from the\n"
          "test set or from training.",
          "the same under both rules. Violation counts in the figures are the sum of stand-on and "
          "give-way violations.")

    # ══════════ ③ 段首加粗标签：整句型的收掉 ══════════
    # (a) 标签本身是一句完整的话 ⟹ 去掉加粗，让它当正常首句，信息一个字不丢
    for lab in ("The counting window and the shield do not coincide.",
                "Feasibility at later steps is not claimed.",
                "That gap does not shrink monotonically as the grid is refined."):
        d.sub("\\noindent\\textbf{%s}\\ " % lab, "%s " % lab)
    # (b) 标签只是把后一句概括一遍 ⟹ 整个删掉
    for lab in ("Quantifying a readily apparent action.",
                "The unshielded continuous configuration.",
                "Collision outcomes relative to the guarantee scope.",
                "What the two refinements address."):
        d.sub("\\noindent\\textbf{%s}\\ " % lab, "")
    d.save()


if __name__ == "__main__":
    main()
