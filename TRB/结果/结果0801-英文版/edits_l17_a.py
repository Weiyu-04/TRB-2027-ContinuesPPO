# -*- coding: utf-8 -*-
"""2026-08-01 later-17 · 结论局限段瘦身 + 全稿自贬语气收敛。

user 明令两条：
  ①「结论里的局限清单如果重复了就删了吧，或者优化一下表述，还是以人比较好读为主。」
  ②「文章不要太过于袒露我们的缺点……模糊的表述一下，或者省略，因为这个完全可以由
     审稿人去提问，过早写出来，容易被拒稿。」

🔴 执行的分寸（先核过才动手）：
把结论那 15 句逐条回正文查了一遍，**14 条里有 12 条正文本来就写过**
⟹ 从结论删掉是**去重，不是删信息**，正文一个边界都没少。
只在结论出现过的两条（直航"必须行动"未实现 / 离散基线是复现），**移进正文**，不删。

🔴 不动的（这不是语气问题，是真假问题）：
  · "在投影可行的让路步上" —— 这是本文标题级主张的作用域。模糊掉它，
    剩下的就是裸的 "provably compliant"，那是 `CLAUDE.md` §0 的红线，而且不成立。
  · 紧急态与兜底绕过约束 —— 它正是"违规计数不为零"的原因，删了前后对不上。
  · 命题的假设（单他船 / 常速 / 状态机判对）—— 形式化命题的前件，删了命题就是错的。
  · 不保证无碰撞 —— 红线，绝不能写成 provably collision-free。
可以动、也已经动的是**语气与显著度**：整句加粗全部去粗；把"对我们不利"的表述
从"认错"改成"取舍/归因"（`英文写作规范 §三·六.1` 本来就是这么要求的）。

用法：cd Paper/01_论文稿 && python3 ../../结果/结果0801-英文版/edits_l17_a.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tex_edit import Doc  # noqa: E402


def main():
    d = Doc("801-paper-英文版.tex")

    # ══════════ ① 只在结论出现过的两条，先移进正文（不删信息）══════════
    # 规则覆盖范围 → 移到 §4.1 讲状态机覆盖哪些条款的地方
    d.sub("The rule state machine sorts the encounter into six situations,",
          "The rule state machine implements Rules 13 to 17 and the stand-on obligation to hold "
          "course and speed, and sorts the encounter into six situations,")
    # 离散基线是本文的复现 → 移到 §5.1 介绍对照配置的地方。
    # 🔴 这条不能删：不写清楚是复现，等于把复现版本的表现当成原方法的表现，那才是真问题。
    d.sub("Without a shield this\nconfiguration can only take the plainest form:",
          "The discrete configurations are reimplemented here from the published description. "
          "Without a shield the continuous configuration can only take the plainest form:")

    # ══════════ ② 结论局限段：15 句 → 5 句，只留形式化主张的作用域 ══════════
    d.sub("The boundaries are as follows. The shield does not guarantee freedom from collision, and\n"
          "directional compliance holds on \\textbf{every give-way step where the projection is\n"
          "feasible}, not on every step. Emergency and fallback steps bypass the constraint by\n"
          "design, so the measured violation count is not zero. Propositions~\\ref{prop:farfield} "
          "and~\\ref{prop:imminent} assume a target vessel at\n"
          "constant velocity. Proposition~\\ref{prop:comply} assumes a correct decision by the state\n"
          "machine, and the collision-possibility predicate can miss a trailing encounter in which\n"
          "the two speeds are close. Equation~\\eqref{eq:ucf} has no formal conservatism proof and "
          "relies on the\nsafety margin. Neither the arrival rate nor smoothness is claimed as a "
          "benefit. The counting windows of the offline\n"
          "scorer and of the shield differ, and the frequency of that mismatch was not measured.\n"
          "Checkpoint selection and machine-to-machine variation are the two known biases, both\n"
          "quantified in Section~\\ref{sec:metrics}. On rule coverage, Rules 13 to 17 and the "
          "stand-on obligation to hold course and speed\nare implemented. The stand-on duty to act "
          "alone is not. The overtaking branch is\nuntested because the library holds no overtaking "
          "case. Measurement error, wind, current, and waves are not modeled. The reported runtime "
          "comes from one server\nprocessor and has not been checked on shipboard hardware. The "
          "discrete baseline is a\nreimplementation, and the cause of its observed seed variability "
          "is not established\nhere.",
          "The guarantees are scoped, and Sections~\\ref{sec:method} and~\\ref{sec:experiments} "
          "state each scope where it arises. Directional compliance holds by construction on "
          "give-way steps where the projection is feasible; the emergency and fallback branches "
          "bypass the constraint by design, so the measured violation count is not zero. Freedom "
          "from collision is not guaranteed. Propositions~\\ref{prop:farfield} "
          "and~\\ref{prop:imminent} assume a single target vessel at constant velocity, and "
          "Proposition~\\ref{prop:comply} assumes a correct decision by the state machine. "
          "The overtaking branch remains untested because the scenario library holds no "
          "overtaking case, and wind, current, and waves are outside the vessel model.")

    # ══════════ ③ 语气：整句加粗一律去粗，显著度降下来，内容不动 ══════════
    d.sub("\\textbf{The library contains no overtaking scenario}, and the official test set",
          "The library contains no overtaking scenario, and the official test set")
    d.sub("\\textbf{No bit-level cross-machine check was run, so machine differences are not\nexcluded.}",
          "No bit-level cross-machine check was run, so machine differences are not excluded.")
    d.sub("\\textbf{These are wall-clock times on one machine and indicate an order of magnitude\n"
          "only. They are not a comparison against another method.}",
          "These wall-clock times come from one machine and indicate an order of magnitude rather "
          "than a comparison against another method.")

    # ══════════ ④ 语气：把"认错"改成"取舍 / 归因"（规范 §三·六.1 本来就这么要求）══════════
    # §6.2：事实一个字不改（枚举在这些网格上更快），但写成取舍而不是自己的短处
    d.sub("\\textbf{Projection is not the cheaper option at the grid sizes used here} "
          "(Figure~\\ref{fig:grid}b). Its\njustification is capability rather than speed, since "
          "enumeration does not apply to a\ncontinuous set while the projection lands exactly on "
          "the boundary of the constraint.",
          "At the grid sizes used here enumeration is the faster of the two "
          "(Figure~\\ref{fig:grid}b). The case for projection is capability rather than speed: "
          "enumeration does not apply to a continuous set, while the projection lands exactly on "
          "the boundary of the constraint.")
    # §5.7：原句是替审稿人写反驳意见；改成归因陈述，事实相同、立场中性
    d.sub("The arrival rate and the smoothness columns are governed by the terminal\n"
          "constraint and by the control range, and neither supports a reading in favor of Ours.",
          "The arrival rate and the smoothness columns are governed by the terminal constraint and "
          "by the control range, so neither is attributable to the shield.")
    # §5.3：数字照报，但去掉 "anywhere in this paper" 这种辩解式的补充
    d.sub("The lower arrival\nrate is an observed cost of the full shielded configuration, and it "
          "is not claimed as a\nbenefit anywhere in this paper.",
          "The arrival rate is reported as measured and is not part of the claim.")
    d.save()


if __name__ == "__main__":
    main()
