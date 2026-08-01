# -*- coding: utf-8 -*-
"""2026-08-01 later-10 · 第 B 批改稿：§3 / §4（数学与事实层，最重的一批）。

🔴 本批最重的一条是**追越的合规集**：论文式 (4) 写成两个半平面的**并集**（非凸），
   而 §4.2.2 又说安全集是凸多边形、QP 最多六条约束 —— 纸面上自相矛盾。
   查代码坐实：`usv_projection.py:315-323` 先由 `usv_colregs.py:878 get_turning_act`
   按**相对朝向**把转向侧定死，再设**单个**半平面 ⟹ 实现是凸的，是论文写漏了选边这一步。
   （同一结论 0731 那轮复审的 7.4 条也独立得出过，见 `05_外部复审/README_核实笔记.md`。）

其余每条注明依据。用法：cd Paper/01_论文稿 && python3 ../../结果/结果0801-英文版/edits_l10_b.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tex_edit import Doc  # noqa: E402

K24 = "\\citl{krasowski2024}{Krasowski and Althoff 2024}"


def main():
    d = Doc("801-paper-英文版.tex")

    # ==================== §3.1 ====================
    d.sub("positive counter-clockwise", "positive counterclockwise")     # 报告3 E20 美式拼写
    # 报告3 A7 / 报告4 #21：Two reasons fix this level of detail 是预告句 + 生硬搭配；
    # 报告2 #2：模型辩护不推进主线，压掉一半，末句局限保留
    d.sub("Two reasons fix this level of detail. The give-way rules govern changes of course and of "
          "speed, and Equation~\\eqref{eq:dyn} takes both as control commands. The rules can therefore "
          "be written as constraints on $u$ with nothing in between, and that is what lets the whole "
          "safety mechanism live in the action space. A finer hydrodynamic model would add "
          "coefficients that differ by vessel and are hard to identify, and the rule constraints would "
          "then be entangled with vessel parameters. Wind, current, and waves are not modeled.",
          "This level of detail is chosen because the give-way rules govern changes of course and of "
          "speed, and Equation~\\eqref{eq:dyn} takes both as control commands. The rules can therefore "
          "be written directly as constraints on $u$, which is what lets the safety shield live in "
          "the action space. A finer hydrodynamic model would add coefficients that differ by vessel "
          "and are hard to identify. Wind, current, and waves are not modeled.")
    # 报告1 #11：船型、动力学限制、决策周期与回合长度都继承自对标工作的模型，本段没有出处
    d.sub("\\noindent\\textbf{Vessel and occupancy.}\\ The subject is a large container vessel of length "
          "$175$~m and beam $25.4$~m. Its limits are $a_{\\max}=0.24~\\mathrm{m/s^2}$, "
          "$\\omega_{\\max}=0.03$~rad/s, and $v_{\\max}=9.5$~m/s. The decision period is $10$~s, and an "
          "episode lasts at most $170$ steps, which is $1{,}700$~s. The circumscribed radius of the "
          "hull is $88.4$~m and the inscribed radius is $12.7$~m.",
          "\\noindent\\textbf{Vessel and occupancy.}\\ The own vessel is a large container vessel of "
          "length $175$~m and beam $25.4$~m. Its limits are $a_{\\max}=0.24~\\mathrm{m/s^2}$, "
          "$\\omega_{\\max}=0.03$~rad/s, and $v_{\\max}=9.5$~m/s, and the decision period is $10$~s. "
          "The vessel model, these limits and the decision period are those reported by " + K24 +
          ". An episode lasts at most $170$ steps, which is $1{,}700$~s. The circumscribed radius of "
          "the hull is $88.4$~m and the inscribed radius is $12.7$~m, both computed from the stated "
          "rectangular hull dimensions.")
    # 报告4 #22：格言式断句，技术含义要读者自己还原
    d.sub("The vessel maneuvers more slowly than it decides. A safety constraint must look ahead for "
          "that reason alone.",
          "The maneuver time therefore exceeds one decision period, so the safety constraint has to "
          "look ahead.")

    # ==================== §3.2 ====================
    d.sub("That solves the dimension only. The shield "     # 报告3 B5：搭配不成立
          "deployed here is implemented for a single target vessel, and it raises an error rather than "
          "running in a degraded mode when more are present.",
          "This keeps the observation dimension fixed, but it does not extend the shield beyond one "
          "target vessel. The shield deployed here is implemented for a single target vessel, and it "
          "raises an error rather than running in a degraded mode when more are present.")
    # 报告1 #12：动作箱半宽与 7x7 网格的对应关系继承自对标配置，本段没有出处
    d.sub("These are exactly the span of the $7\\times7$ grid used by the discrete configurations, so "
          "the two action spaces carry the same control authority and differ only in resolution.",
          "These reproduce the span of the $7\\times7$ discrete action grid reported by " + K24 +
          ", so the two action spaces carry the same control authority and differ only in resolution.")

    # ==================== §3.3 ====================
    # 报告2 #9 / 报告4 #23：把本文的成对范围写成「公约本身的范围」，且语气防御
    d.sub("Assumption (A2) is not a simplification adopted for convenience. It is the scope of the "
          "convention. The rules are written for a pair of vessels "
          "\\citl{imo1972}{IMO 1972}. Three or more vessels can meet at once, and their obligations can then "
          "conflict. One case is an obligation to hold course for one vessel and to give way to "
          "another. The convention does not say which obligation prevails.",
          "Assumption (A2) applies the pairwise obligations of the convention "
          "\\citl{imo1972}{IMO 1972} to one controlled vessel and one target vessel. Three or more "
          "vessels can meet at once, and their obligations can then conflict. One case is an "
          "obligation to hold course for one vessel and to give way to another. The convention does "
          "not say which obligation prevails, and this paper does not define a precedence rule.")
    # 报告2 #8：实际控制器在紧急态与空集兜底时主动绕过这些约束 ⟹ 「for all t」不是本文求解的问题
    d.sub("\\pi:\\ S\\rightarrow\\mathbb{R}^2\n"
          "     \\quad\\text{such that}\\quad\n"
          "     u_t\\in U_{\\mathrm{colregs}}(\\rho_t)\\cap U_{\\mathrm{cf}}(s_t)\n"
          "     \\quad \\text{for all } t.",
          "\\pi:\\ S\\rightarrow\\mathbb{R}^2\n"
          " \\quad\\text{such that}\\quad\n"
          " u_t\\in U_{\\mathrm{colregs}}(\\rho_t)\\cap U_{\\mathrm{cf}}(s_t)\n"
          " \\quad \\text{on every covered step}.", wrap=False)
    d.sub("Equation~\\eqref{eq:problem} is not the same as writing compliance into the reward. It "
          "constrains every command that is executed. It does not ask for an expected value to be "
          "optimal.",
          "A covered step is a non-emergency step on which the constrained set is non-empty. "
          "Emergency steps and steps with an empty set are handled separately in "
          "Section~\\ref{sec:fallback}, and they are excluded from every guarantee in this paper. "
          "Equation~\\eqref{eq:problem} is not the same as writing compliance into the reward. It "
          "constrains the command that is executed rather than an expected value.")
    # 报告4 #24：not compatible / costs both 把离散方法写成缺陷；报告3 C10：This paper 作主语
    d.sub("That route needs enumerable actions, so it is not compatible with continuous control. "
          "Rudder and propulsion are continuous, and quantizing them costs both smoothness and "
          "resolution. This paper replaces enumeration by projection.",
          "That route operates on an enumerable set of commands. Rudder and propulsion are "
          "continuous, so quantizing them fixes the available control levels and their spacing, and a "
          "different enforcement operator is needed. Enumeration is therefore replaced by projection.")

    # ==================== §4.1 ====================
    # 报告3 E10：collision cone 对非海事读者不透明；报告3 B8：speed closes a distance 搭配不自然
    d.sub("The speed set of the own vessel must intersect the collision cone of the target "
          "vessel. The relative speed must also close the current center distance in time.",
          "The speed set of the own vessel must intersect the collision cone of the target vessel, "
          "that is, the set of relative velocities that lead to contact. The relative motion must "
          "also reduce the current center distance within the check horizon.")
    # 报告1 #5：原文是 rm = 3 lm 用于**碰撞锥半径**，几何后果是净空不足约两个船长（本窗口读原文核实）
    d.sub("The occupancy of the target vessel is inflated by three times its length, and the check "
          "horizon is $420$~s.",
          "The collision-cone radius is set to three times the length of the target vessel, following "
          + K24 + ", which detects an approach with less than about two target-vessel lengths of "
          "clearance. The check horizon is $420$~s.")
    # 报告2 #12：§4.1 给的是**修改前**的非对称进入规则，最终规则要到 §4.3.3 才出现
    # 报告3 B7："about to hold" / "already holds" 缺主语且不平行
    d.sub("Entry into a give-way situation has two parts. A give-way predicate must not hold now. It "
          "must hold at every decision step of a $60$~s reaction window under constant-velocity "
          "extrapolation. The first part separates ``about to hold'' from ``already holds'', and it "
          "creates one asymmetry that Section~\\ref{sec:sym} treats.",
          "Entry into a give-way situation has two routes. Under the persistence route the predicate "
          "must not hold now, and it must hold at every decision step of a $60$~s reaction window "
          "under constant-velocity extrapolation. Under the immediate route the obligation is entered "
          "when the predicate already holds at the current step. The persistence route alone "
          "distinguishes an obligation that is about to apply from one that already applies, which "
          "leaves the asymmetry that Section~\\ref{sec:sym} removes.")
    # 🔴🔴 追越：并集非凸，与 §4.2.2 的凸性结论矛盾。实现是先选边再设【单个】半平面（见文件头）
    d.sub("\\{\\omega\\le-\\omega_{\\mathrm{turn}}\\}\\ \\text{or}\\ \\{\\omega\\ge+\\omega_{\\mathrm{turn}}\\}, & \\rho=\\rho_4,",
          "\\{\\sigma\\,\\omega\\le-\\omega_{\\mathrm{turn}}\\}, & \\rho=\\rho_4,", wrap=False)
    d.sub("Head-on and crossing require a turn to starboard. Overtaking allows a turn to either side. "
          "Stand-on holds course and speed. Each set is a half-plane or an interval on the plane of "
          "control commands.",
          "Head-on and crossing require a turn to starboard. The convention does not fix the side for "
          "overtaking, so the state machine selects it from the relative heading of the two vessels "
          "and returns the sign $\\sigma\\in\\{+1,-1\\}$; the selected side is then the only one "
          "allowed on that step. Stand-on holds course and speed. Each set is therefore a single "
          "half-plane or an interval on the plane of control commands, and none of them is a union.")
    # 报告1 #6 / 报告2 #13：20 度与 40 秒是所采用形式化里的阈值，不是公约的法定下限
    d.sub("This is the smallest compliant turn rate. Any turn at least this large complies, and "
          "the bound cannot be lowered.",
          "Under the adopted $20^\\circ$-within-$40$~s definition this is the smallest turn rate that "
          "meets the threshold, and for that definition the value cannot be lowered. A command must "
          "still satisfy the required direction and the conditions of the state machine.")
    # 报告4 #16：follow / Three details are settled here 把本文写成在对方状态机上修补细节
    d.sub("The predicate structure and the numerical thresholds follow the temporal-logic "
          "formalization of the maritime rules " + K24 + ". Three "
          "details are settled here. The inflation radius and the half-width of the speed set are "
          "stated explicitly. Entry into a give-way situation is made symmetric "
          "(Section~\\ref{sec:sym}). The predicate set is then mapped to a constraint set on continuous "
          "commands.",
          "Encounter classification uses the temporal-logic predicates and the numerical thresholds "
          "reported by " + K24 + ". The present formulation states the collision-cone radius and the "
          "half-width of the speed set explicitly, makes entry into a give-way situation symmetric "
          "(Section~\\ref{sec:sym}), and maps each classified situation to a constraint set on "
          "continuous control commands.")

    # ==================== §4.2.1 ====================
    # 报告2 #6：前文称式(6)是真实无碰集合的**凸子集**，后文又承认它既非内近似也非外近似 —— 自相矛盾
    # 报告3 A8：First/Second/Third 模板；报告3 E11：二维平面上说 hyperplane 抬高抽象度
    d.sub("Written directly, that condition is not convex. We replace it by a convex subset of "
          "itself, built in three steps.",
          "Written directly, that condition is not convex. It is replaced by a convex surrogate "
          "constraint, built as follows.")
    d.sub("Third, the end point is expanded to first order in the command,",
          "The end point is then expanded to first order in the command,")
    d.sub("First, the occupancy of the target vessel after one step is over-approximated by a disk.",
          "The occupancy of the target vessel after one step is over-approximated by a disk.")
    d.sub("Second, let $n$ be the unit vector from the target vessel to the nominal end point of the "
          "own vessel, and let $\\delta$ be that distance.",
          "Let $n$ be the unit vector from the target vessel to the nominal end point of the own "
          "vessel, and let $\\delta$ be that distance.")
    d.sub("The end point must lie on the safe side of the separating hyperplane.",
          "The end point must lie on the safe side of the separating line.")
    # 报告3 B9：what does 的指代不完整；报告3 E12：linearization residual 未解释
    # 报告2 #6：明写严格单步无碰只在命题 1 的远场条件下证明
    d.sub("Equation~\\eqref{eq:ucf} alone therefore does not "
          "establish conservatism. The margin inside the safety distance is what does. With that "
          "margin the constraint acts at a center distance of $590$--$840$~m, with a median of "
          "$714$~m over this library. Hull contact needs a much smaller distance. The margin absorbs "
          "the linearization residual. The cost is a tighter constraint that rejects some commands "
          "which are in fact safe.",
          "Equation~\\eqref{eq:ucf} alone therefore does not establish conservatism. The margin "
          "inside the safety distance provides it. With that margin the constraint acts at a center "
          "distance of $590$--$840$~m, with a median of $714$~m over this library, while hull contact "
          "needs a much smaller distance. The margin absorbs the error introduced by the linear "
          "approximation. The cost is a tighter constraint that rejects some commands which are in "
          "fact safe. Strict one-step collision freedom is proved only under the far-field condition "
          "of Proposition~\\ref{prop:farfield}.")
    d.sub("That bound is not implied by the dynamics and is stated separately; the "   # 报告3 B10 分号硬连
          "largest value measured over the $2{,}000$ scenarios of the library is $7.10$~m/s.",
          "That bound is not implied by the dynamics and is stated separately. The largest value "
          "measured over the $2{,}000$ scenarios of the library is $7.10$~m/s.")

    # ==================== §4.2.2 ====================
    d.sub("The first four rows are the action box, the fifth is the compliant-direction half-plane, "
          "and the sixth is the collision-free constraint of Equation~\\eqref{eq:ucf}.",
          "The first four rows are the action box, the fifth is the compliant-direction half-plane of "
          "Equation~\\eqref{eq:ucolregs}, which is a single half-plane in every situation, and the "
          "sixth is the collision-free constraint of Equation~\\eqref{eq:ucf}.")

    # ==================== §4.2.3 ====================
    # 报告3 B11：主语变成目标船，导致「目标船执行直线逃逸」
    d.sub("A target vessel astern that acceleration can clear takes a "
          "straight-line escape inside the reaction window. Every other case tracks a point that "
          "moves with the target vessel. One position-tracking law turns all three into a control "
          "command.",
          "If acceleration alone can clear a target vessel astern, the controller follows a "
          "straight-line escape inside the reaction window. Every other case tracks a point that "
          "moves with the target vessel. One position-tracking law converts each of the three modes "
          "into a control command.")
    # 报告2 #14：emergency 分支是按紧急态直接触发的，并没有先证明「无合规可行命令」
    d.sub("This controller offers no guarantee. Its purpose is a "
          "defined behavior once no compliant feasible command remains.",
          "This controller offers no guarantee. Its purpose is a defined behavior once the emergency "
          "situation has been declared.")
    # 报告3 B12：that 的先行内容不明确，一句写两层失败处理
    d.sub("Otherwise the direction constraint is relaxed, and the collision risk is "
          "minimized if that also fails.",
          "Otherwise the direction constraint is relaxed. If no feasible control command is found "
          "even then, the fallback minimizes the collision risk.")
    # 报告2 #14：§5.6 只报总体分支占比，没有报这些专门构造用例的分支级结果
    d.sub("These branches almost never fire on this scenario library. With course and speed held, "
          "the median closest center distance is about $1{,}300$~m, so the safe control set is "
          "rarely empty. Purpose-built conflict cases exercise the branches, and their outcome is "
          "reported with the shield measurements.",
          "These branches almost never fire on this scenario library. With course and speed held, "
          "the median closest center distance is about $1{,}300$~m, so the safe control set is rarely "
          "empty. Purpose-built conflict cases were used during development to exercise the branches; "
          "they are synthetic and are not part of the reported results.")
    # 报告2 #30：兜底也无保证，别让 §5.6 写成 emergency 是唯一无保证的部分
    d.sub("The emergency branch offers no guarantee. It can still decide whether the situation was "
          "already beyond recovery.",
          "Neither the emergency branch nor the fallback offers a guarantee. The emergency branch can "
          "still decide whether the situation was already beyond recovery.")
    d.sub("After the envelope was hardened, $1{,}500$ further fuzz cases produced none "   # 报告3 E18
          "either.",
          "After the envelope was hardened, $1{,}500$ further randomized stress-test cases produced "
          "none either.")

    # ==================== §4.3 ====================
    # 报告2 #15：实际执行动作是分段函数，式(11) 只写了投影分支
    d.sub("rather than the $P(\\cdot\\mid s_t,u_t)$ of a post hoc filter. The policy learns on the "
          "transition it will actually meet, so it adapts to the projection during training instead of "
          "meeting an unfamiliar filter afterwards.",
          "rather than the $P(\\cdot\\mid s_t,u_t)$ of a correction applied after training. "
          "Equation~\\eqref{eq:shielded-mdp} describes the covered steps. On an emergency step the "
          "applied command comes from the emergency controller, and on a step with an empty safe "
          "control set it comes from the fallback (Section~\\ref{sec:fallback}). The policy is "
          "trained on the transition it will meet at deployment, so it adapts to the projection "
          "during training rather than to a shield introduced afterwards.")
    d.sub("The shield is not applied after the policy as a post hoc correction.",        # 报告3 E14
          "The shield is not applied after the policy as an after-the-fact correction.")

    # ==================== §4.3.1 ====================
    d.sub("A sparse terminal term scores arrival, timeout, stalling and collision, and charges a",
          "A sparse terminal term scores arrival, timeout, stalling, and collision, and charges a")
    # 报告1 #2 / 报告4 #17：from (citation) 不是规范的叙述式引用；are taken from 显得整套照搬
    # 报告1 #13：指称原文符号错误必须给可核查位置（我们代码 usv_colregs.py:489 记的正是这一支）
    d.sub("The five terms are taken from " + K24 + ", with three "
          "declared deviations. The approach term uses the difference of distances between two "
          "steps. The soft compliance term is rescaled, because its original calibration sits at a "
          "scale about fifty times smaller than these scenarios. One clause of the "
          "emergency-release test is read by its physical meaning rather than its literal sign, "
          "since the two disagree.",
          "For comparability with the discrete shielded baseline, the reward uses the five-term "
          "structure reported by " + K24 + ". The present implementation differs in three declared "
          "respects. The approach term uses the difference of distances between two steps. The soft "
          "compliance term is rescaled, because its original calibration sits at a scale about fifty "
          "times smaller than these scenarios. In the emergency-release predicate of that work, the "
          "printed distance clause reads as an upper bound while the accompanying text describes a "
          "distance that is large enough; the present implementation follows the text.")

    # ==================== §4.3.2 ====================
    # 报告1 #14：common implementations 是版本相关的软件行为断言。已下载 2.3.2 核实：
    #   on_policy_algorithm.py:193 把裁剪后的动作喂 env，:226 把**未裁剪**的动作存进 buffer 算梯度
    d.sub("First, common implementations feed the clipped action to the environment and "
          "the unclipped action to the policy gradient.",
          "First, in Stable-Baselines3 2.3.2 the action passed to the environment is clipped, while "
          "the action stored for the policy gradient is the unclipped sample.")
    # 报告3 B14：channel 隐喻要读者回推「两条通道」指什么
    d.sub("The standard deviation therefore has a one-way channel to "
          "grow, which pushes still more samples outside the box.",
          "The standard deviation can therefore keep growing without improving the reward, which "
          "pushes still more samples outside the box.")
    d.sub("Actions can then never leave "
          "the box, clipping becomes the identity map, both channels close, the smoothness penalty is "
          "differentiable everywhere, and the entropy is bounded above.",
          "Actions can then never leave the box, clipping becomes the identity map, both effects are "
          "removed, the smoothness penalty is differentiable everywhere, and the entropy is bounded "
          "above.")
    # 报告4 #25：cannot be dropped 是绝对命令，corrupt 带情绪
    d.sub("The offset of one keeps both shape parameters above unity, and that constraint "
          "cannot be dropped. Below unity the Beta density is U-shaped and diverges at the two ends, "
          "and the mode then sits at an endpoint. Because evaluation always takes the mode, such a "
          "policy would silently corrupt every reported number, and the aggregate metrics would give "
          "no sign of it.",
          "The offset of one keeps both shape parameters above unity. Below unity the Beta density is "
          "U-shaped and diverges at the two ends, and the mode then sits at an endpoint. Because "
          "evaluation always takes the mode, such a policy would return a boundary action at every "
          "step, and the aggregate metrics would give no sign of it.")

    # ==================== §4.3.3 ====================
    # 报告2 #10：用于论证对称进入必要性的对抗用例明确违反 (A5)，要写明它们在形式化范围之外
    d.sub("These cases are "
          "synthetic and illustrate the mechanism, so they are reported apart from the main results.",
          "These cases violate Assumption (A5) by construction. They lie outside the formal scope of "
          "this paper and are used only as an implementation diagnostic, so they are reported apart "
          "from the main results.")

    # ==================== §4.3.4 ====================
    # 报告3 B15：takes the default 搭配生硬；报告1 #15：sb3-contrib 不能靠 Raffin 2021 支撑
    d.sub("every hyperparameter takes the default of the implementation library and is "
          "left untuned. The libraries are Stable-Baselines3 2.3.2 "
          "\\citl{raffin2021}{Raffin et al. 2021} and sb3-contrib 2.3.0, and the advantage is computed "
          "by generalized advantage estimation \\citl{schulman2016}{Schulman et al. 2016}.",
          "every hyperparameter uses its default value in the implementation library and is left "
          "untuned. The libraries are Stable-Baselines3 2.3.2 "
          "\\citl{raffin2021}{Raffin et al. 2021} and its companion package sb3-contrib 2.3.0, and "
          "default refers to the released source of those two versions. The advantage is computed by "
          "generalized advantage estimation \\citl{schulman2016}{Schulman et al. 2016}.")
    d.sub("Running normalization of the observation and the "     # 报告4 #26：is not optional 是命令式口吻
          "reward is not optional. Without it the large-scale approach term in "
          "Equation~\\eqref{eq:reward} drives the policy to a degenerate solution that holds station.",
          "All configurations use running normalization of the observation and the reward. Without "
          "it, the large-scale approach term in Equation~\\eqref{eq:reward} drove the tested "
          "implementation to a degenerate solution that holds station.")
    d.save()


if __name__ == "__main__":
    main()
