# -*- coding: utf-8 -*-
"""2026-08-01 later-10 · 第 E 批：把前四批改出来的问题收干净。

前四批把 >30 词的句子从 0 句改成了 18 句、摘要 296→320 词，还踩了一条红线
（"a true zero collision probability" 命中 `zero collision`）。本批全部收拾：
  · 18 句超长句逐句拆开（规范 §一：>30 词一句都不要有）
  · 摘要压回 300 词以内（一个事实都不删，只压措辞）
  · 红线 zero collision 换说法
  · §4.1 的 "following \\citl{...}" 是挂靠句式（规范 §三·六 明令不要），改事实性归属

用法：cd Paper/01_论文稿 && python3 ../../结果/结果0801-英文版/edits_l10_e.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tex_edit import Doc  # noqa: E402

K24 = "\\citl{krasowski2024}{Krasowski and Althoff 2024}"


def main():
    d = Doc("801-paper-英文版.tex")

    # ==================== 摘要压回 300 词以内 ====================
    d.sub("on the give-way steps where the required correction is feasible.",
          "on give-way steps where the correction is feasible.")
    d.sub("The give-way clause is written as a half-plane constraint on the plane of control commands.",
          "The give-way clause is written as a half-plane constraint on the control plane.")
    d.sub("A quadratic program with two variables and at most six linear constraints projects the "
          "policy control command onto the nearest point of that set.",
          "A quadratic program with two variables and six linear constraints projects the policy "
          "control command onto the nearest point of that set.")
    d.sub("In the ablation the bounded action distribution lowers the median turn increment by "
          "$65\\%$ and the violation count by $41\\%$, at a paired signed-rank $p=0.008$.",
          "In the ablation the bounded action distribution lowers the median turn increment by "
          "$65\\%$ and the violation count by $41\\%$ (paired signed-rank $p=0.008$).")
    d.sub("The safety shield is independent of training, so it can sit above an existing controller "
          "and be checked one step at a time.",
          "The safety shield is independent of training and can sit above an existing controller.")

    # ==================== 红线 ====================
    d.sub("A zero observed rate on the $100$ validation scenarios cannot establish a true zero "
          "collision probability.",
          "A zero count on the $100$ validation scenarios cannot establish that the true rate is "
          "zero.")

    # ==================== 超长句逐句拆 ====================
    d.sub("Future work will wire the terminal constraint of the fallback maneuver into the deployed "
          "shield and verify it in closed loop, extend directional compliance to several target "
          "vessels with a conflict resolution rule, and implement the far-field test as a "
          "constant-time early exit.",
          "Three directions follow. The terminal constraint of the fallback maneuver should be wired "
          "into the deployed shield and verified in closed loop. Directional compliance should be "
          "extended to several target vessels with a conflict resolution rule. The far-field test "
          "should be implemented as a constant-time early exit.")
    d.sub("The convention does not fix the side for "
          "overtaking, so the state machine selects it from the relative heading of the two vessels "
          "and returns the sign $\\sigma\\in\\{+1,-1\\}$; the selected side is then the only one "
          "allowed on that step.",
          "The convention does not fix the side for overtaking. The state machine selects it from the "
          "relative heading of the two vessels and returns the sign $\\sigma\\in\\{+1,-1\\}$. Only "
          "the selected side is allowed on that step.")
    d.sub("What the construction does give is an object "
          "that can be checked step by step: the classification of the situation, the direction of "
          "the half-plane, and the feasibility of the projection, none of which depends on the "
          "training outcome.",
          "The construction does give an object that can be checked step by step. A reviewer can "
          "check the classification of the situation, the direction of the half-plane, and the "
          "feasibility of the projection. None of the three depends on the training outcome.")
    d.sub("Among the converged runs the unshielded configurations hold a collision rate near "
          "$1.8\\%$ throughout training and more than $2.4$ violations per episode, while the "
          "shielded configurations sit at a collision rate of zero and near $0.5$ violations per "
          "episode.",
          "Among the converged runs the unshielded configurations hold a collision rate near "
          "$1.8\\%$ throughout training and more than $2.4$ violations per episode. The shielded "
          "configurations sit at a collision rate of zero and near $0.5$ violations per episode.")
    d.sub("The present formulation states the collision-cone radius and the "
          "half-width of the speed set explicitly, makes entry into a give-way situation symmetric "
          "(Section~\\ref{sec:sym}), and maps each classified situation to a constraint set on "
          "continuous control commands.",
          "The present formulation states the collision-cone radius and the half-width of the speed "
          "set explicitly. It makes entry into a give-way situation symmetric "
          "(Section~\\ref{sec:sym}). It then maps each classified situation to a constraint set on "
          "continuous control commands.")
    d.sub("The first four rows are the action box, the fifth is the compliant-direction half-plane of "
          "Equation~\\eqref{eq:ucolregs}, which is a single half-plane in every situation, and the "
          "sixth is the collision-free constraint of Equation~\\eqref{eq:ucf}.",
          "The first four rows are the action box. The fifth is the compliant-direction half-plane of "
          "Equation~\\eqref{eq:ucolregs}, which is a single half-plane in every situation. The sixth "
          "is the collision-free constraint of Equation~\\eqref{eq:ucf}.")
    d.sub("To the best of the authors' "
          "knowledge, among the maritime studies listed there, a per-step hard guarantee of "
          "directional rule compliance has been demonstrated only on a discrete action space "
          + K24 + ".",
          "To the best of the authors' knowledge, a per-step hard guarantee of directional rule "
          "compliance has been demonstrated only on a discrete action space, among the maritime "
          "studies listed there " + K24 + ".")
    # 规范 §三·六：following X 是挂靠句式，改成事实性归属；同时拆开 34 词
    d.sub("The collision-cone radius is set to three times the length of the target vessel, following "
          + K24 + ", which detects an approach with less than about two target-vessel lengths of "
          "clearance. The check horizon is $420$~s.",
          "The collision-cone radius is three times the length of the target vessel, as reported by "
          + K24 + ". This detects an approach with less than about two target-vessel lengths of "
          "clearance. The check horizon is $420$~s.")
    d.sub("A give-way turn that must be readily apparent has a smallest admissible turn rate, and on "
          "a fixed grid the smallest available compliant turn rate is the nearest grid point at or "
          "beyond it.",
          "A give-way turn that must be readily apparent has a smallest admissible turn rate. On a "
          "fixed grid the smallest available compliant turn rate is the nearest grid point at or "
          "beyond it.")
    d.sub("These reproduce the span of the $7\\times7$ discrete action grid reported by " + K24 +
          ", so the two action spaces carry the same control authority and differ only in resolution.",
          "These reproduce the span of the $7\\times7$ discrete action grid reported by " + K24 +
          ". The two action spaces therefore carry the same control authority and differ only in "
          "resolution.")
    d.sub("The validation set holds only $100$ scenarios, and "
          "the binomial standard error of the arrival rate is about $4$ points, so taking the maximum "
          "over $20$ candidates lifts the figure on that set.",
          "The validation set holds only $100$ scenarios, and the binomial standard error of the "
          "arrival rate is about $4$ points. Taking the maximum over $20$ candidates therefore lifts "
          "the figure on that set.")
    d.sub("The premise that the safe control set must be enumerated can "
          "therefore be replaced by the premise that it is convex, which applies a projection-based "
          "safety shield to maritime COLREGs under continuous control.",
          "The premise that the safe control set must be enumerated can therefore be replaced by the "
          "premise that it is convex. That step applies a projection-based safety shield to maritime "
          "COLREGs under continuous control.")
    d.sub("In the emergency-release predicate of that work, the "
          "printed distance clause reads as an upper bound while the accompanying text describes a "
          "distance that is large enough; the present implementation follows the text.",
          "In the emergency-release predicate of that work, the printed distance clause reads as an "
          "upper bound. The accompanying text describes a distance that is large enough. The present "
          "implementation follows the text.")
    d.sub("A terminal constraint on the fallback maneuver is one route to "
          "that property, but the closed-loop integration is not finished, so it is stated as a "
          "design direction and not as a guarantee.",
          "A terminal constraint on the fallback maneuver is one route to that property. The "
          "closed-loop integration is not finished, so it is stated as a design direction and not as "
          "a guarantee.")
    d.sub("With that margin the constraint acts at a center "
          "distance of $590$--$840$~m, with a median of $714$~m over this library, while hull contact "
          "needs a much smaller distance.",
          "With that margin the constraint acts at a center distance of $590$--$840$~m, with a median "
          "of $714$~m over this library. Hull contact needs a much smaller distance.")
    d.sub("Across the $72$ runs the arrival rate at the selected checkpoint exceeds the one at "
          "the last checkpoint by a median of $0.2$ points, and it is lower for $31$ of them.",
          "Across the $72$ runs the arrival rate at the selected checkpoint exceeds the one at the "
          "last checkpoint by a median of $0.2$ points. It is lower for $31$ of them.")
    d.sub("\\item Three scoped safety statements (Sections~\\ref{sec:collfree} to~\\ref{sec:fallback}): "
          "a sufficient far-field condition for one-step collision freedom, directional compliance on "
          "feasible give-way steps, and a one-sided criterion for imminent unavoidable collision. "
          "Each carries its assumptions, and the uncovered cases are named.",
          "\\item Three scoped safety statements (Sections~\\ref{sec:collfree} "
          "to~\\ref{sec:fallback}). They are a sufficient far-field condition for one-step collision "
          "freedom, directional compliance on feasible give-way steps, and a one-sided criterion for "
          "imminent unavoidable collision. Each carries its assumptions, and the uncovered cases are "
          "named.")
    d.sub("Apart from the entropy coefficient, which is set to $0.01$, and the bounded action "
          "distribution, every hyperparameter uses its default value in the implementation library "
          "and is left untuned.",
          "Two settings are chosen here: the entropy coefficient is $0.01$ and the action "
          "distribution is bounded. Every other hyperparameter uses its default value in the "
          "implementation library and is left untuned.")
    d.save()


if __name__ == "__main__":
    main()
