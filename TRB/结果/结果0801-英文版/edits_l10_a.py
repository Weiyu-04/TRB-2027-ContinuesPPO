# -*- coding: utf-8 -*-
"""2026-08-01 later-10 · 第 A 批改稿：摘要 / §1 / §2 / 表 1。

每一条都注明依据：GPT 四份复查报告的条号，或本窗口对**原始 PDF / 源码**的核实结论。
🔴 凡与仓库既定决策冲突的外部意见，一律不照做，理由写在注释里。

用法：cd Paper/01_论文稿 && python3 ../../结果/结果0801-英文版/edits_l10_a.py
"""
import os
import re
import sys
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tex_edit import Doc  # noqa: E402


def main():
    d = Doc("801-paper-英文版.tex")

    # ==================== 摘要 ====================
    # 报告4 #2 + user 拍板：Objectives 的口径大于证明范围，补作用域（标题不动）
    d.sub("This paper builds a per-step guarantee of directional compliance for continuous control.",
          "This paper builds a per-step guarantee of directional compliance for continuous control, "
          "on the give-way steps where the required correction is feasible.")
    # 报告4 #3：small 是无尺度评价词，正文已有变量数与约束数，直接量化
    d.sub("A small quadratic program projects the command of the policy onto the nearest point of that set.",
          "A quadratic program with two variables and at most six linear constraints projects the "
          "policy control command onto the nearest point of that set.")
    # 报告2 #17 / 报告4 #20：single-variable ladder 与表 2 表注的 companion variables 直接冲突
    d.sub("Nine configurations form a single-variable ladder, each trained with eight seeds.",
          "Nine configurations are compared, each trained with eight seeds.")
    # 报告3 D1 / 报告4 #4：gains are significant 无对象无数值。数取自官方 600（03 L243-续51 A）
    d.sub("The ablation gains are significant under a paired signed-rank test.",
          "In the ablation the bounded action distribution lowers the median turn increment by "
          "$65\\%$ and the violation count by $41\\%$, at a paired signed-rank $p=0.008$.")
    # 报告3 C6：safety mechanism 不是锁死术语
    d.sub("this is the first projection-based safety mechanism that enforces directional COLREGs "
          "compliance on continuous control.",
          "this is the first projection-based safety shield that enforces directional COLREGs "
          "compliance on continuous control.")
    # 报告3 A9 / 报告4 #5：空洞元话语
    d.sub("The paper also states what is proved and what is not.", "")
    # 报告4 #6：liability assessment / without special hardware 无证据；改成实测数字
    d.sub("The compliance check is independent of training. It can sit above an existing controller "
          "as a compliance layer for functional verification and liability assessment. The online cost "
          "is one small quadratic program per step, which runs in real time without special hardware.",
          "The safety shield is independent of training, so it can sit above an existing controller "
          "and be checked one step at a time. The online cost is one quadratic program per step, "
          "with a median solve time of $4.11$~ms against a decision period of $10$~s.")

    # ==================== §1 ====================
    # 报告3 A1 / 报告4 #7 电影化开头 + 报告3 B1 一句套两个 that 从句
    d.sub("Two vessels meet at sea. Each turns to starboard, and they pass clear of one another. What "
          "makes this work is not the judgment of either bridge team. It is a set of rules that both "
          "sides follow, and that both sides know the other will follow. When no one is on board, that "
          "predictability becomes a technical requirement.",
          "At sea, collision avoidance is coordinated by a shared rule set. Each vessel is expected to "
          "act as the rules prescribe, and each can expect the same of the other. When no one is on "
          "board, that predictability becomes a technical requirement.")
    # 报告3 C13：正文自身从未给出 COLREGs 这个缩写的来源，后文却一直用它
    d.sub("must therefore satisfy the International Regulations for Preventing Collisions at Sea "
          "\\citl{imo1972}{IMO 1972}.",
          "must therefore satisfy the International Regulations for Preventing Collisions at Sea, "
          "or COLREGs \\citl{imo1972}{IMO 1972}.")
    # 报告2 #4：两条路线未限定为「学习型」，与 §2.1 的第三类模型驱动方法自相矛盾
    # 报告4 #8：缺陷式二元对立 → 中性取舍
    # 报告4 #9：规范 §三·六 明写 Introduction 只描述路线不点名 ⟹ 删这处引用
    # 报告2 #5：no finite grid / every compliant turn overshoots 是无限定的绝对判断
    # 报告3 B2：The price is continuous control 搭配错（读成「代价是连续控制」）
    d.sub("Existing work follows two routes, and each gives up one property. The first discretizes "
          "the action space and masks the non-compliant commands \\citl{krasowski2024}{Krasowski and "
          "Althoff 2024}. Compliance then holds by construction, and it does not depend on the "
          "training outcome or on the seed. The price is continuous control. Quantization fixes the "
          "control resolution at the grid spacing. A give-way turn that must "
          "be readily apparent has a smallest admissible turn rate, and no finite grid lands on it, "
          "so every compliant turn overshoots. The second route keeps continuous control and writes the "
          "rules into the reward as a penalty \\citm{\\cl{meyer2020}{Meyer et al. 2020}; "
          "\\cl{heiberg2022}{Heiberg et al. 2022}; \\cl{mller2026}{M\\\"uller et al. 2026}}. Control "
          "resolution is preserved and the guarantee is lost, because a reward shapes expected "
          "behavior and cannot exclude any single violating command. Section~\\ref{sec:related} sets "
          "out both routes in detail.",
          "Learning-based work on directional compliance follows two routes, and the two keep "
          "different properties. The first discretizes the action space and masks the non-compliant "
          "commands. Compliance then holds by construction, and it does not depend on the training "
          "outcome or on the seed. Quantization fixes the control resolution at the grid spacing. A "
          "give-way turn that must be readily apparent has a smallest admissible turn rate, and on a "
          "fixed grid the smallest available compliant turn rate is the nearest grid point at or "
          "beyond it. The second route keeps continuous control and writes the rules into the reward "
          "as a penalty \\citm{\\cl{meyer2020}{Meyer et al. 2020}; "
          "\\cl{heiberg2022}{Heiberg et al. 2022}; \\cl{mller2026}{M\\\"uller et al. 2026}}. The "
          "control resolution is preserved. A reward acts on expected behavior, so it excludes no "
          "single violating command. Section~\\ref{sec:related} sets out both routes in detail.")
    # 报告1 #16 / 报告4 #11：空白领域声明必须带 hedge；fills it / change of premise 是宣言式
    # 报告3 A4：三短句模板 —— 但「枚举 → 凸投影」是叙事主线的核心一环（规范 §一），压缩不删
    # 报告3 C7：safe set → safe control set（锁死术语）
    d.sub("The intersection of continuous action spaces, provable directional compliance and COLREGs "
          "has therefore been empty. This paper fills it with a projection-based safety shield. A rule "
          "state machine returns the half-plane of compliant directions. That half-plane is intersected "
          "with the action box and with a one-step collision-free condition, which gives a convex safe "
          "control set. A quadratic program projects the continuous command of the policy onto the "
          "nearest point of that set. The step that makes this possible is a change of premise. Action "
          "masking needs the safe set to be enumerable. Projection needs it only to be convex. The "
          "shield sits inside the environment transition, so the policy learns under its own corrected "
          "commands.",
          "To the best of the authors' knowledge, continuous action spaces, provable directional "
          "compliance and COLREGs have not been combined before. This paper addresses that "
          "combination with a projection-based safety shield. A rule state machine returns the "
          "half-plane of compliant directions. That half-plane is intersected with the action box and "
          "with a one-step collision-free condition, which gives a convex safe control set. A "
          "quadratic program projects the continuous control command of the policy onto the nearest "
          "point of that set. Projection replaces enumeration because it needs the safe control set "
          "to be convex rather than enumerable. The shield sits inside the environment transition, so "
          "the policy learns under its own corrected commands.")
    # 报告3 A5：机械式章节导航，无新信息
    d.sub("The second concern of this paper is the scope of what is proved. The word provable covers",
          "The word provable covers")

    # ==================== 贡献列表 ====================
    # 报告4 #19：贡献 2 实际对应 §4.2.1--4.2.3，且没带出假设；报告3 A6：三项过度对称
    d.sub("\\item A graded safety statement (Section~\\ref{sec:collfree}). One-step collision freedom in "
          "the far field is a strict result. Directional compliance on every feasible give-way step "
          "holds by construction. Imminent unavoidable collision is flagged by a sufficient "
          "condition that never raises a false alarm. Each statement carries its assumptions, and the "
          "uncovered cases are named.",
          "\\item Three scoped safety statements (Sections~\\ref{sec:collfree} to~\\ref{sec:fallback}): "
          "a sufficient far-field condition for one-step collision freedom, directional compliance on "
          "feasible give-way steps, and a one-sided criterion for imminent unavoidable collision. "
          "Each carries its assumptions, and the uncovered cases are named.")
    # 报告2 #17 / 报告4 #20：贡献 3 自称 separates the mechanisms，下一句就承认三处分不开
    d.sub("\\item An experimental design that separates the mechanisms (Section~\\ref{sec:setup}). Nine "
          "configurations form one connected single-variable ladder. Three steps of that ladder carry "
          "a companion variable that cannot be separated, and those are listed with the ladder.",
          "\\item An experimental design that supports mechanism-level comparison "
          "(Section~\\ref{sec:setup}). Nine configurations form one connected ladder. Three steps of "
          "that ladder carry a companion variable that cannot be separated, and those are listed with "
          "the ladder.")

    # ==================== §2.1 ====================
    # 规范 §六 定标是 Action Shielding，并明写要避开裸的 provable compliance（红线）
    d.sub(r"\subsection{Safe Reinforcement Learning and Provable Compliance}",
          r"\subsection{Safe Reinforcement Learning and Action Shielding}")
    d.sub("The velocity obstacle uses the relative velocity cone to select an avoiding velocity, and "
          "it was given COLREGs semantics \\citl{kuwata2014}{Kuwata et al. 2014} and later extended to "
          "more general encounters \\citl{huang2019}{Huang et al. 2019}.",   # 报告3 B28（34 词）
          "The velocity obstacle selects an avoiding velocity from the relative velocity cone. "
          "\\citl{kuwata2014}{Kuwata et al. 2014} gave it COLREGs semantics, and "
          "\\citl{huang2019}{Huang et al. 2019} extended it to more general encounters.")
    d.sub("Deep reinforcement learning moved the policy into the data "     # 报告3 B4：隐喻式搭配
          "\\citl{sarhadi2022}{Sarhadi et al. 2022}.",
          "Deep reinforcement learning learns the policy from data "
          "\\citl{sarhadi2022}{Sarhadi et al. 2022}.")
    d.sub("Most of that work discretizes the command, into a finite set of rudder angles for confined "  # B29（36 词）+ B3 多余逗号
          "water \\citl{shen2019}{Shen et al. 2019}, or into compliant avoidance timing and paths "
          "\\citm{\\cl{zhao2019}{Zhao and Roh 2019}; \\cl{chun2021}{Chun et al. 2021}}.",
          "Most of that work discretizes the control command. \\citl{shen2019}{Shen et al. 2019} use a "
          "finite set of rudder angles for confined water. "
          "\\citm{\\cl{zhao2019}{Zhao and Roh 2019}; \\cl{chun2021}{Chun et al. 2021}} discretize "
          "avoidance timing and paths instead.")
    d.sub("A second group keeps continuous control and writes COLREGs as a soft penalty that decays "  # B30（33 词）
          "with bearing and range \\citl{meyer2020}{Meyer et al. 2020}, or as a tunable risk gate "
          "\\citl{heiberg2022}{Heiberg et al. 2022}.",
          "A second group keeps continuous control. \\citl{meyer2020}{Meyer et al. 2020} write COLREGs "
          "as a soft penalty that decays with bearing and range, and "
          "\\citl{heiberg2022}{Heiberg et al. 2022} write them as a tunable risk gate.")
    d.sub("has also been shown on underactuated craft \\citl{cheng2018}{Cheng and Zhang 2018}.",  # C2 craft
          "has also been shown on underactuated vessels \\citl{cheng2018}{Cheng and Zhang 2018}.")
    d.sub("This group is usually built on proximal policy optimization "                         # B31（40 词）
          "\\citl{schulman2017}{Schulman et al. 2017}, and a bounded Beta distribution "
          "\\citl{chou2017}{Chou et al. 2017} can parameterize the continuous command so that samples "
          "are not clipped at the boundary of the action box.",
          "This group is usually built on proximal policy optimization "
          "\\citl{schulman2017}{Schulman et al. 2017}. A bounded Beta distribution "
          "\\citl{chou2017}{Chou et al. 2017} can parameterize the continuous control command. Its "
          "samples are then not clipped at the boundary of the action box.")

    # ==================== §2.2 ====================
    # 报告1 #10：CMDP 的期望约束需原始出处。Altman 1999 已核实存在（Chapman & Hall/CRC）
    d.sub("Writing safety as a constrained Markov decision process and solving it with a Lagrangian "
          "method guarantees constraint satisfaction in expectation, which again excludes no single command.",
          "Constrained Markov decision processes state safety as an expected cumulative cost "
          "\\citl{altman1999}{Altman 1999}. A constraint of that form is satisfied in expectation, so "
          "it again excludes no single control command.")
    d.sub("A realizable form of it has been "        # 报告3 E7：realizable 是形式化验证圈的黑话
          "extended to continuous action spaces for general tasks, without maritime rules "
          "\\citl{kim2024}{Kim et al. 2024}.",
          "A continuous-space form of it has been developed for general tasks, without maritime rules "
          "\\citl{kim2024}{Kim et al. 2024}.")
    # 报告4 #12：has the same form / What is added 把本文写成对 Dalal 的局部增补
    d.sub("The closest mechanism is a safety shield that linearizes the constraint at the current "
          "state. It then projects the command of the policy onto the resulting half-space in "
          "closed form \\citl{dalal2018}{Dalal et al. 2018}. The projection used here has the same form. "
          "What is added is the encoding of the directional give-way clause as a half-plane in the "
          "control plane. The enforced object is then rule compliance rather than a state bound.",
          "\\citl{dalal2018}{Dalal et al. 2018} linearize a state constraint at the current state and "
          "project the control command onto the resulting half-space in closed form. In the present "
          "formulation the feasible set combines the actuator limits, a directional COLREGs "
          "constraint, and a one-step collision-free constraint. The projected quantity is therefore "
          "a rule-compliant control command rather than a state bound.")
    # 🔴 三条一起改，全部对着 Krasowski 原文核过（本窗口 fitz 抽取）：
    #   报告1 #3  场景是 2024 构造的：“we constructed 2000 CommonOcean benchmarks [64] and randomly
    #             split them in a 70 % training and 30 % testing set”（2022 只提供框架）
    #   报告1 #7  说“列为未来工作”的是 Krasowski 自己（§VII.d），Kochdumper 只是它举的手段
    #   报告1 #16 单篇文献撑不起“整个领域空白”，要 hedge
    d.sub("Table~\\ref{tab:related} places this work on four dimensions and shows where the gap is. "
          "A hard guarantee of rule compliance in a maritime setting had been demonstrated only on "
          "a discrete action space \\citl{krasowski2024}{Krasowski and Althoff 2024}. That work "
          "provides the scenario library and the state machine thresholds used here. Action "
          "projection under continuous control is listed there as future work "
          "\\citl{kochdumper2023}{Kochdumper et al. 2023}.",
          "Table~\\ref{tab:related} places this work on four dimensions. To the best of the authors' "
          "knowledge, among the maritime studies listed there, a per-step hard guarantee of "
          "directional rule compliance has been demonstrated only on a discrete action space "
          "\\citl{krasowski2024}{Krasowski and Althoff 2024}. That work also constructed the scenario "
          "set used here, inside the CommonOcean benchmark framework "
          "\\citl{krasowski2022}{Krasowski and Althoff 2022}, and reports the state machine thresholds "
          "adopted here. It identifies action correction under continuous control as future work, and "
          "names action projection \\citl{kochdumper2023}{Kochdumper et al. 2023} as one mechanism.")
    # 报告1 #9：Markgraf 全文 reachable set / over-approximation / quadratic program 各 0 次（本窗口核）
    #          它真正讲的是环境内投影 vs 策略内投影 + action aliasing ⟹ 归属改到它真支撑的地方
    # 报告3 B32（33 词）+ E8（higher-order set representation 含义不透明）一并解决
    d.sub("The reachable set is represented by a "
          "convex over-approximation and the correction by a quadratic program "
          "\\citl{markgraf2026}{Markgraf et al. 2026}, rather than by a higher-order set representation, "
          "which keeps the per-step cost small.",
          "\\citl{markgraf2026}{Markgraf et al. 2026} compare projection placed inside the environment "
          "with projection placed inside the policy, and analyze how each affects the policy gradient. "
          "Here the safe control set is a convex region of the control plane and the correction is a "
          "quadratic program in two variables, which keeps the per-step cost small.")
    # 报告1 #8：De Lellis 全文 linear kernel / arrival / terminal condition 各 0 次（本窗口核）
    #          它讲的是用奖励塑形达成 settling time 与 steady-state error ⟹ 归属改正，实现选择归本文
    d.sub("A linear kernel with arrival as a terminal "
          "condition is used instead \\citl{de2024}{De Lellis et al. 2024}.",
          "Shaping can instead be designed against a stated control requirement "
          "\\citl{de2024}{De Lellis et al. 2024}. Here a linear distance kernel is used, with arrival "
          "as a terminal condition.")
    # 报告3 B33：全文最长句（49 词，逗号连接三个独立结论）+ C5：filter → safety shield
    d.sub("Including the filter during training and penalizing "
          "the correction improves sample efficiency \\citl{pizarro2025}{Pizarro Bejarano et al. 2025}, "
          "a sufficiently permissive filter does not cost asymptotic optimality "
          "\\citl{oh2025}{Oh et al. 2025}, and a safety term that opposes the goal term causes stalling "
          "short of the goal \\citl{grover2023}{Grover et al. 2023}.",
          "Applying the safety shield during training and penalizing the correction improves sample "
          "efficiency \\citl{pizarro2025}{Pizarro Bejarano et al. 2025}. A sufficiently permissive "
          "shield does not cost asymptotic optimality \\citl{oh2025}{Oh et al. 2025}. A safety term "
          "that opposes the goal term causes stalling short of the goal "
          "\\citl{grover2023}{Grover et al. 2023}.")

    # ==================== 表 1 ====================
    # 报告2 #7：表注写 every decision step，与命题 2 的覆盖范围（状态机判对 + 让路 + 投影可行 + 非紧急）冲突
    d.sub("\\caption{Position of related work. \\capnote{A per-step hard guarantee means the class of "
          "commands the method can exclude at \\emph{every} decision step, not constraint satisfaction "
          "in expectation.}}",
          "\\caption{Position of related work. \\capnote{A per-step hard guarantee means the class of "
          "control commands the method can exclude at each \\emph{covered} decision step, not "
          "constraint satisfaction in expectation. For Ours the covered steps are the give-way steps "
          "on which the projection is feasible; emergency and fallback steps are excluded.}}")
    # 报告3 C15：QP 在表 1 处尚未给全称（§4.2.2 才定义）
    d.sub(r"\textbf{Ours} & \textbf{Continuous} & \textbf{Projection (QP)} & \textbf{Direction} & $\checkmark$ \\",
          r"\textbf{Ours} & \textbf{Continuous} & \textbf{Projection (quadratic program)} & \textbf{Direction} & $\checkmark$ \\",
          wrap=False)
    # 报告1 #18：Lee 已正式发表于 Mechatronics 卷 117（2026），KAIST 论文库核实
    d.sub(r"\cl{lee2025}{Lee et al. 2025}", r"\cl{lee2025}{Lee et al. 2026}", expect=2, wrap=False)
    d.save()

    # ==================== 参考文献 ====================
    r = Doc("_refs.tex")
    r.sub(r"\refitem{\hypertarget{ref:alshiekh2018}{}Alshiekh, Mohammed,",
          "\\refitem{\\hypertarget{ref:altman1999}{}Altman, Eitan. 1999. \\textit{Constrained Markov "
          "Decision Processes}. Boca Raton: Chapman \\& Hall/CRC.}\n\n"
          "\\refitem{\\hypertarget{ref:alshiekh2018}{}Alshiekh, Mohammed,", wrap=False)
    # 🔴 文章号未能独立核实 ⟹ 只写已核实的卷号，宁缺勿编（GPT 给的 103495 不采信）
    r.sub("Lee, Changyu, Jinwook Park, and Jinwhan Kim. 2025. ``Efficient COLREGs-Compliant Collision "
          "Avoidance Using Turning Circle-Based Control Barrier Function.'' arXiv:2504.19247.",
          "Lee, Changyu, Jinwook Park, and Jinwhan Kim. 2026. ``Efficient COLREGs-Compliant Collision "
          "Avoidance Using Turning Circle-Based Control Barrier Function.'' \\textit{Mechatronics} "
          "117. arXiv:2504.19247.", wrap=False)
    r.save()


if __name__ == "__main__":
    main()
