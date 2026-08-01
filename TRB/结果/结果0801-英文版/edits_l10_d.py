# -*- coding: utf-8 -*-
"""2026-08-01 later-10 · 第 D 批改稿：§5.7 / §6 / §7 / 致谢 / 图注。

🔴 本批含 user 拍板的**整节删除 §6.3**（对自主船舶投入营运的意义）：三份报告都判它越界
   （型式认可 / 事故审查 / 责任认定 / 现役船载部署，稿里一条证据都没有），
   而 `英文写作规范 §九` 的「已砍内容登记」本来就写着这一节已砍 —— 实际它还留在稿里。
   它承重的那点意思（这一层可以逐步核查）并进 §6.1 一句话。

🔴 §6 两个小标题改回 `英文写作规范 §六` 的定标（6.2 = Projection Versus Enumeration），
   同时解决报告4 #36 说的「交易式标题」。

用法：cd Paper/01_论文稿 && python3 ../../结果/结果0801-英文版/edits_l10_d.py
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tex_edit import Doc  # noqa: E402


def main():
    d = Doc("801-paper-英文版.tex")

    # ==================== §5.7 ====================
    # 报告3 A13 / 报告4 #34：自问自答式转场 + 裁判式措辞
    d.sub("(Table~\\ref{tab:ext}). The question here is not whether a learning method is stronger. It "
          "is which dimensions a geometric method already covers.",
          "(Table~\\ref{tab:ext}). The comparison identifies the dimensions that the three geometric "
          "controllers evaluated here already cover.")
    # 报告2 #34：三个具体控制器推不到 geometric methods 整体；aim only 带贬低色彩
    d.sub("The answer narrows the defensible claim to one dimension. The collision rate does not "
          "separate the two families, since the barrier function shield "
          "also reaches $0.00\\%$ here. The arrival rate and the smoothness columns are governed by "
          "the terminal constraint and by the control range, and neither supports a reading in "
          "favor of Ours. The real difference is in COLREGs violations, and almost all of it "
          "comes from stand-on violations. These controllers aim only at avoiding contact, so they keep maneuvering while the rules "
          "require course and speed to be held.",
          "The comparison narrows the defensible claim to one dimension. The collision rate does not "
          "separate the two families, since the barrier function shield also reaches $0.00\\%$ here. "
          "The arrival rate and the smoothness columns are governed by the terminal constraint and by "
          "the control range, and neither supports a reading in favor of Ours. The largest observed "
          "difference is in COLREGs violations, and almost all of it comes from stand-on violations. "
          "These three controllers impose collision-avoidance constraints but no explicit "
          "course-and-speed constraint, so they keep maneuvering while the rules require course and "
          "speed to be held.")
    d.sub("The claim of this paper against geometric methods "
          "rests on this dimension alone.",
          "The claim against the three geometric baselines evaluated here rests on this dimension "
          "alone.")
    # 报告2 #34：96.8--99.7% 没有对应的表或图 ⟹ 说明它的来源，别当成一个凭空出现的数
    d.sub("These controllers have no terminal alignment stage. They reach the goal "
          "position in $96.8\\%$ to $99.7\\%$ of episodes and then fail the terminal heading gate.",
          "These controllers have no terminal alignment stage. Across the seven settings in this "
          "table they reach the goal position in $96.8\\%$ to $99.7\\%$ of episodes and then fail the "
          "terminal heading gate.")
    # 报告2 #34：Ours 的盾与紧急控制器可以用到执行器量程，「同等控制权限」只对策略输出成立
    d.sub("The policy range "
          "is $\\pm0.048$ and $\\pm0.018$, the same authority as the trained policies, and the "
          "actuator limits are $\\pm0.24$ and $\\pm0.03$.",
          "The policy range is $\\pm0.048~\\mathrm{m/s^2}$ and $\\pm0.018$~rad/s, which matches the "
          "output range of the trained policies, and the actuator limits are "
          "$\\pm0.24~\\mathrm{m/s^2}$ and $\\pm0.03$~rad/s. For Ours the shield and the emergency "
          "controller may use the actuator limits, so the executed range is not identical.")

    # ==================== §6.1 ====================
    d.sub(r"\subsection{Two Points about the Mechanism}",                 # 报告4 #35：标题不说明是哪两点
          r"\subsection{Continuous-Action Refinements and Guarantee Scope}")
    # 报告2 #35：symmetric entry 是状态机进入规则，离散系统同样会有 ⟹ 不能说两者都是连续动作专属
    # 报告4 #35：repairs two defects 把连续动作写成制造了缺陷
    d.sub("\\noindent\\textbf{Both refinements belong to "
          "the continuous action space.}\\ A quantized action space has no samples pushed past the "
          "action box, and no channel whose "
          "differential entropy is unbounded above. Both problems appear only once a COLREGs shield "
          "is moved into a continuous action space. The ablation therefore repairs two defects "
          "created by that move, rather than tuning two hyperparameters.",
          "\\noindent\\textbf{What the two refinements address.}\\ A quantized action space has no "
          "samples pushed past the action box, and no distribution whose differential entropy is "
          "unbounded above. The bounded action distribution therefore addresses an issue specific to "
          "continuous actions. Symmetric entry addresses the state machine instead, and the same "
          "asymmetry can arise on a discrete action space. The ablation evaluates two implementation "
          "effects rather than two hyperparameters.")
    # 报告2 #35：clearable set 全文未定义；报告3 E13：recursive feasibility 对交通读者不直观
    # 并入 §6.3 承重的那点意思（逐步可核查），其余整节删
    d.sub("\\noindent\\textbf{Recursive feasibility is not claimed.}\\ A "
          "reactive barrier function shield has no recursive feasibility proof under a $10$~s "
          "zero-order hold. The clearable set of this work suggests one route to that property. The closed-loop "
          "integration is not finished, so it is stated as a design direction and not as a "
          "guarantee.",
          "\\noindent\\textbf{Feasibility at later steps is not claimed.}\\ No proof is given that a "
          "feasible control command remains available at every subsequent decision step under a "
          "$10$~s zero-order hold. A terminal constraint on the fallback maneuver is one route to "
          "that property, but the closed-loop integration is not finished, so it is stated as a "
          "design direction and not as a guarantee. What the construction does give is an object "
          "that can be checked step by step: the classification of the situation, the direction of "
          "the half-plane, and the feasibility of the projection, none of which depends on the "
          "training outcome.")

    # ==================== §6.2 ====================
    d.sub(r"\subsection{What Projection Buys and What It Costs}",         # 规范 §六 定标 + 报告4 #36
          r"\subsection{Projection Versus Enumeration}")
    d.sub("Both sides of the exchange were measured directly, in exact arithmetic for the "      # 报告3 A14
          "resolution and on one machine for the cost.",
          "This section compares control resolution and computation time.")
    # 报告2 #5 / 报告4 #36：always overshoots 与 does not hold for this task 都是无限定的绝对判断
    d.sub("A continuous command reaches that value exactly. "
          "A finite grid can only take the nearest grid point at or beyond it, so it always overshoots. "
          "On the $7\\times7$ grid reproduced here, the smallest available compliant turn rate is "
          "$37.5\\%$ above that value.",
          "A continuous command reaches that value exactly. On a fixed grid the smallest available "
          "compliant turn rate is the first grid point at or beyond it, which coincides with the "
          "threshold only when the two divide. On the $7\\times7$ grid reproduced here it is "
          "$37.5\\%$ above that value.")
    d.sub("A $5\\times5$ grid overshoots by $3.1\\%$ while a $7\\times7$ grid overshoots by $37.5\\%$. "
          "The common statement that a finer grid approaches "
          "continuous control therefore does not hold for this task. Only a continuous command with "
          "projection reaches the smallest admissible turn rate.",
          "A $5\\times5$ grid overshoots by $3.1\\%$ while a $7\\times7$ grid overshoots by "
          "$37.5\\%$, so for these two grids the overshoot is not monotone in the spacing. A "
          "continuous command with projection reaches the smallest admissible turn rate for any "
          "threshold.")
    # 报告2 #36：正文没有给出枚举的计时、硬件与重复次数 ⟹ 不能说 measured directly
    d.sub("The cost is in computation, and it runs against intuition. "
          "Enumerating and masking $K\\times K$ commands costs time that grows with $K^2$. Even at "
          "$K=121$, which is $14{,}641$ commands, it stays well below the time "
          "of one projection. \\textbf{Projection is not the cheaper option.} Its justification is "
          "capability rather than speed. Enumeration has no meaning on a continuous set, while "
          "projection is feasible and lands exactly on the boundary of the constraint.",
          "The cost is in computation. Enumerating and masking $K\\times K$ commands scales with "
          "$K^2$, whereas the projection solves a fixed two-variable quadratic program. "
          "\\textbf{Projection is not the cheaper option at the grid sizes used here.} Its "
          "justification is capability rather than speed, since enumeration does not apply to a "
          "continuous set while the projection lands exactly on the boundary of the constraint.")

    # ==================== 🔴 整节删 §6.3（user 拍板）====================
    d.sub("\\subsection{What This Means for Putting Autonomous Vessels into Service}\n\n"
          "Making compliance a projectable hard constraint changes more than the behavior of the "
          "policy. It changes how this layer can be reviewed. When compliance is induced by "
          "training, verification is statistical: a sample of scenarios is run, a violation "
          "frequency is reported, and the conclusion moves with every version of the policy. When "
          "compliance holds by construction, the object of verification becomes the constraint "
          "itself. A reviewer checks whether the state machine classifies the situation correctly, "
          "whether the half-plane points the way the clause requires, and whether the projection "
          "lands inside the feasible set. All three can be checked step by step and none depends on "
          "the training outcome, so the reviewer does not need to reproduce the training run. For "
          "type approval and accident review this is the more workable form.\n\n"
          "The mechanism also does not require the existing control system to be replaced. It "
          "accepts a desired control command from any source and returns the nearest compliant "
          "one. It can therefore sit as a compliance layer above an existing autopilot or "
          "path-following controller. Its online cost is one quadratic program per step and it "
          "needs no dedicated hardware, so deployment on current shipboard platforms is realistic.",
          "", wrap=False)

    # ==================== §7 ====================
    # 报告3 A15 / 报告4 #38：either buys ... or buys 是胜负式二元框架
    d.sub("Existing work either buys a provable form "
          "of compliance with a discrete action space and masking, or buys control resolution with "
          "continuous commands and a soft reward. This paper shows that the exchange is not necessary. "
          "The give-way clause is exactly a half-plane in the control plane. The premise that the "
          "safe set must be enumerated can therefore be replaced by the premise that it is convex. A "
          "projection-based safety shield is therefore brought to maritime COLREGs under continuous "
          "control. Directional compliance on every give-way step where the projection is feasible holds "
          "by construction, independently of the training outcome and of the seed. Any rule that can "
          "be written as a convex constraint in the action space can follow the "
          "same route.",
          "Discrete masking and continuous reward-based formulations emphasize different combinations "
          "of command-level enforcement and control resolution. Under the action-space formulation "
          "adopted here, the implemented directional give-way requirement is a half-plane constraint "
          "on the control command. The premise that the safe control set must be enumerated can "
          "therefore be replaced by the premise that it is convex, which applies a projection-based "
          "safety shield to maritime COLREGs under continuous control. Directional compliance on "
          "every give-way step where the projection is feasible holds by construction, independently "
          "of the training outcome and of the seed. The same projection structure may apply to other "
          "rules that admit a convex representation in the action space, subject to correct "
          "classification and feasibility.")
    d.sub("The discrete shielded baseline records $1.16$ and the strongest geometric "     # 报告3 D5 / 报告4 #40
          "baseline records $4.95$.",
          "The discrete shielded baseline records $1.16$ and the geometric baseline with the fewest "
          "violations per episode records $4.95$.")
    # 报告3 B26：maneuvering target vessel 不能 void 一个命题
    d.sub("Propositions "
          "\\ref{prop:farfield} and \\ref{prop:imminent} both assume that the target vessel holds its "
          "velocity, and a maneuvering target vessel voids them.",
          "Propositions \\ref{prop:farfield} and \\ref{prop:imminent} both assume that the target "
          "vessel holds its velocity, and they do not apply when the target vessel maneuvers.")
    # 报告2 #21：主表并没有报两个版本；报告2 #37：种子脆弱性没有做原实现与重实现的实验拆分
    d.sub("Two biases in the reported numbers are known, namely checkpoint selection and "
          "machine-to-machine variation. The main table is therefore reported in two versions, and "
          "every configuration in one table is evaluated on one machine in one pass.",
          "Two biases in the reported numbers are known, namely checkpoint selection and "
          "machine-to-machine variation. Both are quantified in Section~\\ref{sec:metrics}, and every "
          "configuration in one table is evaluated on one machine in one pass.")
    d.sub("The discrete baseline is a "
          "reimplementation, and its seed fragility is attributed to that reimplementation.",
          "The discrete baseline is a reimplementation, and the cause of its observed seed "
          "variability is not established here.")
    # 报告2 #38：局限段漏掉了正文已经暴露的五类边界
    d.sub("The overtaking branch is implemented but the scenario "
          "library contains no overtaking case, so it is untested.",
          "The overtaking branch is implemented but the scenario library contains no overtaking case, "
          "so it is untested. Five further boundaries follow from the formulation. Proposition "
          "\\ref{prop:comply} assumes a correct decision by the state machine, and the "
          "collision-possibility predicate can miss a trailing encounter in which the two speeds are "
          "close. The counting window of the offline scorer and the window of the shield differ, and "
          "the frequency of that mismatch was not measured. The state of the target vessel is assumed "
          "to be available without measurement error, and wind, current, and waves are not modeled. "
          "The near-field constraint of Equation~\\eqref{eq:ucf} has no formal conservatism proof and "
          "relies on the safety margin. The reported runtime comes from one server processor and "
          "$1{,}872$ steps, and it has not been checked on shipboard hardware.")
    d.sub("Three lines of work follow. The terminal constraint of the backup maneuver should be "   # A16 + C9
          "wired into the deployed shield and verified in closed loop. Directional compliance "
          "should be extended to several target vessels with a conflict resolution rule. The "
          "far-field test should be implemented as an $O(1)$ early exit.",
          "Future work will wire the terminal constraint of the fallback maneuver into the deployed "
          "shield and verify it in closed loop, extend directional compliance to several target "
          "vessels with a conflict resolution rule, and implement the far-field test as a "
          "constant-time early exit.")

    # ==================== 致谢 ====================
    # 报告1 #4 / 报告3 B27 / 报告4 #18：不能 thank a citation；框架与场景集要分清
    d.sub("The authors thank the CommonOcean benchmark and \\citl{krasowski2024}{Krasowski and "
          "Althoff 2024} for the public scenario library and the formalization of the rules. Both "
          "made a comparison under one condition possible.",
          "The authors acknowledge the CommonOcean benchmark framework "
          "\\citl{krasowski2022}{Krasowski and Althoff 2022} and the public scenario set and "
          "temporal-logic rule formalization of \\citl{krasowski2024}{Krasowski and Althoff 2024}. "
          "These public resources made an evaluation on a common scenario set and a common "
          "rule-classification basis possible.")

    # ==================== 图注 ====================
    d.sub("The bars are wide because several configurations contain a diverged seed.",   # 报告3 B22
          "The error bars are wide because several configurations include a nonconverged run.")
    # 报告2 #26/#27：Figure 4 混了三种口径，曲线只含收敛的 run，正文解释必须带这个条件
    d.sub("Panels (a) to (c) and (f) are recorded once per rollout on the training scenarios, so they "
          "carry exploration noise. Panel (d) is measured on the validation set once per checkpoint, "
          "which is why it has fewer points. Panel (e) counts all eight seeds.",
          "Panels (a) to (c) and (f) are recorded once per trajectory batch on the training "
          "scenarios, so they carry the variation caused by sampled actions. Panel (d) is measured on "
          "the validation set once per checkpoint, which is why it has fewer points. Panel (e) counts "
          "all eight runs. Because panels (a) to (d) and (f) exclude nonconverged runs, they describe "
          "learning speed among the converged runs only.")
    # 报告3 B34（37 词）+ B35（31 词）：长句拆开
    d.sub("Within each encounter type of the test set, the "
          "scenario shown is the one whose closest approach is smallest when both vessels hold their "
          "initial velocity, which is $7$~m for T-$1848$ and $2$~m for T-$76$.",
          "For each encounter situation the scenario was selected by the smallest closest approach "
          "under constant velocity. That distance is $7$~m for T-$1848$ and $2$~m for T-$76$.")
    d.sub("The open circle is the start of the own vessel, the "
          "square is the start of the target vessel, and the dashed line is the recorded track of the "
          "target vessel.",
          "The open circle marks the start of the own vessel. The square marks the start of the "
          "target vessel. The dashed line shows the recorded track of the target vessel.")
    d.save()


if __name__ == "__main__":
    main()
