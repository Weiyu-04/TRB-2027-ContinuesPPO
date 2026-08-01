# -*- coding: utf-8 -*-
"""2026-08-01 later-10 · 第 C 批改稿：§5 实验（口径 / 归因 / 未兑现的承诺）。

🔴 本批最要紧的是**把「另有报告」的四处承诺兑现掉**。20 页装不下额外的表，
   改法是用 `make_ckpt_versions.py` 从重评产物算出数字、压成正文一两句：
     · 官方测试集 600 上，最佳存档比末段存档的到达率差：72 条 run 中位 **+0.17 点**，
       IQR [-1.12, +2.25]，其中 **39 条更高 / 31 条更低 / 2 条持平**；
     · 只算收敛种子（验证集口径）时，Ours 违规仍是九条里最低（0.53/局），8/8 收敛；
     · 收敛判据按验证集算与按测试集算，九条配置的收敛种子数**完全一致**。
   🔴 原文「That is four times the drift of about 1.6 points」两处都有问题：
     7.5/1.6 = 4.69 不是 4 倍（报告2 #22），且 **1.6 这个数在仓库里查无出处**
     （04:585 与 03 L243-续45 D 记的跨机器抖动是 ±0.5 点）⟹ 整句删掉，换成实测的两版差。

用法：cd Paper/01_论文稿 && python3 ../../结果/结果0801-英文版/edits_l10_c.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tex_edit import Doc  # noqa: E402


def main():
    d = Doc("801-paper-英文版.tex")

    # ==================== §5.1 ====================
    # 报告1 #3：2,000 个场景是 Krasowski 2024 构造的（原文 "we constructed 2000 CommonOcean
    #          benchmarks [64] and randomly split them in a 70 % training and 30 % testing set"），
    #          2022 提供的是基准框架。报告3 B16：数据集用 contains 不用 holds
    d.sub("The experiments use the two-vessel encounter library of the CommonOcean benchmark "
          "\\citl{krasowski2022}{Krasowski and Althoff 2022}, which holds $2{,}000$ scenarios. Each "
          "scenario fixes the initial pose and speed of the controlled vessel. It also fixes the "
          "initial state and the planned track of the target vessel. It ends with a goal region that "
          "has a position gate, a heading gate, and a time limit.",
          "The experiments use the $2{,}000$ two-vessel encounters constructed by "
          "\\citl{krasowski2024}{Krasowski and Althoff 2024} inside the CommonOcean benchmark "
          "framework \\citl{krasowski2022}{Krasowski and Althoff 2022}. Each scenario fixes the "
          "initial pose and speed of the own vessel. It also fixes the initial state and the planned "
          "track of the target vessel. Each scenario defines a goal region with a position gate, a "
          "heading gate, and a time limit.")
    # 报告2 #17 / 报告4 #27：与表 2 表注的三处 companion variables 直接冲突
    d.sub("Table~\\ref{tab:arms} lists nine configurations. They form one connected ladder, and each "
          "level changes a single variable from the level before it. The contribution of each mechanism "
          "can therefore be read on its own.",
          "Table~\\ref{tab:arms} lists nine configurations. They form one connected ladder. Most "
          "adjacent comparisons change one declared variable; the three bundled steps listed with the "
          "table have to be read jointly.")
    # 报告3 B17：is on / is off 口语化；steps can be attributed 缺被归因的对象
    d.sub("\\textbf{Notes.} A check mark is on, a cross is off, and a dash is not applicable. Rows run "
          "from the discrete baseline to Ours. Three steps of the ladder carry a companion "
          "variable that cannot be separated. The action space changes together with the policy "
          "distribution, the continuous-only shaping is a bundle of four items, and the bounded "
          "distribution also changes the initial exploration scale. Those three steps can only be "
          "attributed as a whole.",
          "\\textbf{Notes.} A check mark denotes enabled, a cross denotes disabled, and a dash "
          "denotes not applicable. Rows run from the discrete baseline to Ours. Three steps of the "
          "ladder carry a companion variable that cannot be separated. The action space changes "
          "together with the policy distribution, the continuous-only shaping is a bundle of four "
          "items, and the bounded distribution also changes the initial exploration scale. The "
          "effects of those three steps can only be read jointly with their companion variables.")
    # 报告2 #18 / 报告4 #28：weakest 是贬义等级判断，且推不出 conservative lower bound
    d.sub("The gain attributed to the shield is therefore measured against the weakest form of "
          "continuous action",
          "The shield comparison therefore uses the plainest continuous configuration")

    # ==================== §5.2 ====================
    # 报告2 #19：episode duration 与「每个指标按会遇态势分解」都没有报
    d.sub("The main metrics are the arrival rate, the collision rate, the COLREGs violations per "
          "episode, the share of steps under emergency control, and the episode duration. The "
          "violation count is split into give-way and stand-on violations. Control quality, safety "
          "margin, shield behavior and the single-step solve time are also reported, and every "
          "metric is broken down by encounter situation.",
          "The main metrics are the arrival rate, the collision rate, the COLREGs violations per "
          "episode, and the share of steps under emergency control. The violation count is split "
          "into give-way and stand-on violations. Control quality appears in "
          "Table~\\ref{tab:main}, and the behavior of the shield and the single-step solve time in "
          "Section~\\ref{sec:res-shield}.")
    # 报告2 #25：offline scorer 与盾的窗口不一致且错配频率未测，中间缺「指标如何反映机制」的桥
    d.sub("How often the mismatch occurs was not measured.",
          "How often the mismatch occurs was not measured. The violation counts below are therefore "
          "an empirical trajectory-level measure, and not a direct test of "
          "Proposition~\\ref{prop:comply}.")
    d.sub("Machine time did not allow the same during training.",       # 报告3 B19：the same 指代过远
          "Available machine time did not allow every training run to be placed on one machine.")
    d.sub("The candidates are those twenty checkpoints, the criterion "  # 报告3 B20：一句并列四件事
          "is the highest arrival rate on the validation set, ties go to the earlier checkpoint, and "
          "the test set is never touched.",
          "The twenty saved checkpoints are the candidates. The criterion is the highest arrival rate "
          "on the validation set. Ties go to the earlier checkpoint. The test set is never accessed.")
    # 🔴 报告2 #21 + #22：两版并报是空承诺，且 1.6 查无出处。换成实测（见文件头）
    d.sub("\\textbf{The rule is itself biased, so two versions are reported.} The validation set holds "
          "only $100$ scenarios, and the binomial standard error of the arrival rate is about $4$ "
          "points. Taking the maximum over $20$ candidates lifts the reported value by about $7.5$ "
          "points in expectation. That is four times the drift of about $1.6$ points that the "
          "same-machine rule was written to prevent. The best-checkpoint version and the "
          "last-checkpoint version are therefore both reported.",
          "\\textbf{The rule is itself biased.} The validation set holds only $100$ scenarios, and "
          "the binomial standard error of the arrival rate is about $4$ points, so taking the maximum "
          "over $20$ candidates lifts the figure on that set. On the held-out test set the effect is "
          "small. Across the $72$ runs the arrival rate at the selected checkpoint exceeds the one at "
          "the last checkpoint by a median of $0.2$ points, and it is lower for $31$ of them. The "
          "ordering of the configurations and the outcome of the paired tests are the same under "
          "both rules.")
    d.sub("are recorded during training and are labelled as such.",     # 报告3 E2：英式拼写
          "are recorded during training and are labeled as such.")
    # 报告2 #20：§5.2 总括说全文只用符号检验，而 §5.4 用的是符号秩检验
    d.sub("\\noindent\\textbf{Statistics.}\\ The paired test is a per-seed paired sign test, whose "
          "smallest two-sided value with $8$ seeds is $p=2/2^{8}=0.0078$.",
          "\\noindent\\textbf{Statistics.}\\ The nine-configuration comparisons use a per-seed paired "
          "sign test, whose smallest two-sided value with $8$ seeds is $p=2/2^{8}=0.0078$. The "
          "$2\\times2$ ablation comparisons use a per-seed paired Wilcoxon signed-rank test.")
    # 🔴 报告2 #21：converged-seeds-only version 没有这张表 ⟹ 换成实算结论
    d.sub("Diverged seeds are counted in the main table as they are, and a "
          "converged-seeds-only version is also reported.",
          "Diverged seeds are counted in the main table as they are. Restricting the table to "
          "converged runs leaves the ordering unchanged, and Ours keeps the lowest violation count at "
          "$0.53$ per episode. The convergence criterion selects the same runs whether it is applied "
          "on the validation set or on the test set.")

    # ==================== §5.3 ====================
    d.sub("Three readings follow from Table~\\ref{tab:main}. First, Ours has the lowest violation count",
          "Ours has the lowest violation count")                        # 报告3 A10：总-分三点模板
    d.sub("at $p=0.0078$. Second, Ours is the only shielded configuration that reaches the "
          "convergence criterion on all $8$ seeds. Third, the arrival rate of Ours is $88.2\\%$, which "
          "is below the three unshielded configurations and below the discrete safe baseline. That gap "
          "is the cost of the shield and it is not claimed as a benefit anywhere in this paper. "
          "Table~\\ref{tab:main} is also reported in a converged-seeds-only version, because a diverged "
          "seed pollutes the per-episode averages and the direction of that pollution differs by "
          "configuration. Per-seed raw values accompany the table.",
          "at $p=0.0078$. Ours is also the only shielded configuration that reaches the convergence "
          "criterion on all $8$ seeds. Its arrival rate is $88.2\\%$, which is below the three "
          "unshielded configurations and below the discrete shielded baseline. The lower arrival rate "
          "is an observed cost of the full shielded configuration, and it is not claimed as a benefit "
          "anywhere in this paper. The isolated shield comparison is the Continuous, no shield versus "
          "Continuous, shield pair. A nonconverged run shifts the per-episode averages, and the "
          "direction of the shift differs among configurations; restricting the table to converged "
          "runs leaves the ordering unchanged.")

    # ==================== §5.4 ====================
    d.sub("Neither can be separated statistically from "                # 报告3 B23：Neither 先行对象不清
          "the bounded distribution alone, at $p=0.74$ and $p=0.11$. The two changes therefore divide "
          "the work. Smoothness comes entirely from the bounded action distribution, and both "
          "contribute to compliance.",
          "The combined configuration does not differ significantly from the bounded-only "
          "configuration in either turn increment or violations per episode, at $p=0.74$ and "
          "$p=0.11$. Within this ablation set the bounded action distribution accounts for the "
          "observed reduction in turn increments, while both changes are associated with lower "
          "measured violation counts. Their effects cannot be separated from the declared companion "
          "variable, nor by violation type.")
    d.sub("A converged-seeds-only version is given for this group as well.",
          "Restricting this group to converged runs leaves the direction of every comparison "
          "unchanged.")

    # ==================== §5.5 ====================
    # 报告3 B24 + A11 + 报告2 #26：Figure 4 不是单一训练口径
    d.sub("Figure~\\ref{fig:curves} is taken from training. It answers two questions that do not need "
          "the re-evaluation, namely whether the shield costs learning speed, and how many seeds of "
          "each configuration reach a working policy.",
          "Figure~\\ref{fig:curves} combines training-rollout metrics, validation-checkpoint metrics, "
          "and convergence counts, as labeled per panel. It reports learning speed among the "
          "converged runs and the number of runs that meet the convergence criterion.")
    d.sub("First, there is a cost, and it appears in learning speed.",
          "There is a cost, and it appears in learning speed.")
    # 报告3 A12 / 报告4 #31：give what is bought 是交易式隐喻，且 (c) 训练 (d) 验证口径不同
    d.sub("(Figure~\\ref{fig:curves}b). Panels (c) and (d) give what is bought. The unshielded "
          "configurations hold a collision rate near $1.8\\%$ throughout training and more than "
          "$2.4$ violations per episode. The shielded configurations sit at a collision rate of zero "
          "and near $0.5$ violations per episode. The shield removes risk from the outcome, "
          "and it narrows the exploration space at the same time. Slower learning is the direct "
          "consequence of that narrowing.",
          "(Figure~\\ref{fig:curves}b). Panel (c) reports the collision rate on the training "
          "rollouts and panel (d) the violation count on the validation set. Among the converged "
          "runs the unshielded configurations hold a collision rate near $1.8\\%$ throughout "
          "training and more than $2.4$ violations per episode, while the shielded configurations "
          "sit at a collision rate of zero and near $0.5$ violations per episode. The shielded "
          "configurations therefore show lower collision and violation metrics together with slower "
          "improvement in arrival rate, which is consistent with a narrower explored region of the "
          "action space.")
    # 报告4 #30 / 报告2 #28：working policy 口语化；lower the chance that training fails 是概率外推
    d.sub("Second, reliability differs a great deal among the shielded configurations. Under the "
          "convergence rule declared in advance, all $8$ seeds of Ours reach a working policy, and it "
          "is the only shielded configuration for which this holds.",
          "Seed-level convergence differs among the shielded configurations. Under the convergence "
          "rule declared in advance, all $8$ runs of Ours meet the $50\\%$ validation-arrival "
          "criterion, and it is the only shielded configuration for which this holds.")
    d.sub("(Figure~\\ref{fig:curves}e). The two changes therefore improve more than the final numbers. "
          "They also lower the chance that training itself fails. Diverged seeds are counted as they "
          "are, and a converged-seeds-only version is given.",
          "(Figure~\\ref{fig:curves}e). In this eight-seed experiment the two changes therefore "
          "increase the number of runs meeting the criterion, as well as the final numbers. "
          "Nonconverged runs are counted in the main table as they are.")
    d.sub(r"\noindent\textbf{A counterexample that has to be stated.}\ On the test set",   # 报告4 #32
          r"\noindent\textbf{Collision outcomes relative to the guarantee scope.}\ On the test set")
    # 报告2 #29：100 场景里一次碰撞 = 1%，不是「低于样本分辨率」；且表 3 的 0.00% 是测试集种子中位
    d.sub("The "
          "earlier figure of zero came from the validation set of $100$ scenarios, where a single "
          "collision is below the resolution of the sample.",
          "A zero observed rate on the $100$ validation scenarios cannot establish a true zero "
          "collision probability. The $0.00\\%$ in Table~\\ref{tab:main} is the median across eight "
          "test-set runs, and two of those runs record one collision each.")

    # ==================== §5.6 ====================
    # 报告2 #30：§4.2.3 已写明兜底也无保证 ⟹ emergency 不是唯一；碰撞来源不可归因就不能说它 confirm 了什么
    d.sub("The emergency branch is the only part of this method with no guarantee, so its use is "
          "reported separately.",
          "The emergency branch and the fallback carry no guarantee, so their use is reported "
          "separately.")
    d.sub("Collisions are rare enough that their source cannot be attributed with confidence. Ours "
          "records two collisions in $4{,}800$ test episodes. If a collision falls under emergency "
          "control, then it occurs on a step where the last resort had already been engaged and had "
          "failed. Those steps "
          "lie outside the scope of Proposition~\\ref{prop:farfield} by definition, since that "
          "proposition covers the far field, non-emergency, single-step case. This is not a "
          "counterexample to it, but it does confirm that the emergency branch is an empirical "
          "fallback.",
          "Collisions are rare enough that their source cannot be attributed with confidence. Ours "
          "records two collisions in $4{,}800$ test episodes. Neither contradicts "
          "Proposition~\\ref{prop:farfield} unless its far-field condition held at the step that "
          "produced the collision.")
    # 报告2 #31 / 报告4 #33：只给了中位与四分位，推不出「所有修正都在十分之一以内」；
    #                       且修正可能来自任何一条起作用的约束，不只是规则那条
    d.sub("At the end of training its median is $0.082$ and its interquartile range is "
          "$[0.065,\\,0.102]$. The gap between the command of the policy and what the rules require "
          "therefore stays within about a tenth of the control range.",
          "At the end of training its median is $0.082$ and its interquartile range is "
          "$[0.065,\\,0.102]$. The correction can be caused by any of the active shield constraints, "
          "not only by the compliant-direction half-plane.")
    d.sub("They record an order of magnitude.",                          # 报告3 B25：搭配不自然
          "They indicate the order of magnitude of the computation time on that machine.")
    d.save()


if __name__ == "__main__":
    main()
