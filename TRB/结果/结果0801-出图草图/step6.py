# -*- coding: utf-8 -*-
"""环节 2-D：§4.2 与 §4.3 压缩，并补一块伪代码。

4.2.1 单步无碰撞约束   717 → 约 400
4.2.2 安全动作集与投影  196 → 约 200 ＋ 伪代码
4.2.3 兜底            1130 → 约 420
4.3.2 回报函数         648 → 约 380
4.3.3 有界动作分布      503 → 约 340
4.3.5 训练配置         461 → 约 250
"""
import re

P = "/home/user/TRB-2027-ContinuesPPO/TRB/Paper/01_论文稿/729-paper-中文版.tex"
s = open(P, encoding="utf-8").read()
L = s.split("\n")
log = []


def sect(name):
    i = next(k for k, l in enumerate(L) if l.startswith("\\subsubsection{" + name + "}"))
    j = i + 1
    while j < len(L) and not re.match(r"\\(sub)?(sub)?section\{", L[j]):
        j += 1
    return "\n".join(L[i:j])


def block(seg, label):
    ls = seg.split("\n")
    j = next(k for k, l in enumerate(ls) if "\\label{" + label + "}" in l)
    i = j
    while "\\begin{equation}" not in ls[i]:
        i -= 1
    k = j
    while "\\end{equation}" not in ls[k]:
        k += 1
    return "\n".join(ls[i:k + 1])


# ── 伪代码宏包 ──────────────────────────────────────────────────────────────
if "algorithm" not in s:
    s = s.replace(r"\usepackage{caption}",
                  "\\usepackage{caption}\n\\usepackage{algorithm}\n"
                  "\\usepackage[noend]{algpseudocode}\n"
                  "\\captionsetup[algorithm]{labelsep=space,labelfont=bf,font=small}", 1)
    log.append("载入 algorithm + algpseudocode")

# ── 4.2.1 ───────────────────────────────────────────────────────────────────
old = sect("单步无碰撞约束")
EQ_UCF = block(old, "eq:ucf")
new = r"""\subsubsection{单步无碰撞约束}

对当前状态与他船占据，我们要求施加的控制在下一个决策步末不致使两船体相交。该条件直接写出是非凸的，故取一个保守的凸内近似，构造分三步。先把他船在一个决策步之后的占据外近似为圆盘，圆心取恒速外推、半径已含安全裕度，本船以外接圆过近似，于是“两船体不交”的一个充分条件是两圆心距不小于两半径之和 $d_{\mathrm{safe}}$。再取分离方向 $n$ 为由他船指向本船标称落点的单位向量、$\delta$ 为该距离，其中标称落点由裁剪进动作箱的策略输出算得。最后把落点沿控制作一阶展开，$p_e(u)\approx p_{\mathrm{nom}}+J(u-u_{\mathrm{nom}})$，并要求落点位于分离超平面的安全侧，即得一条关于控制的线性不等式
__EQ_UCF__

一阶展开本身既非内近似也非外近似，其余项符号随曲率变化，故仅由式~\eqref{eq:ucf} 并不能直接断言保守，使该断言成立的是安全距离中预留的裕度。叠加裕度后，无碰撞约束实际施加在约 $590$--$840$~m 的中心距上，按本场景库的他船尺寸实算中位约 $714$~m，而两船真正接触所需的距离要小得多。裕度因此吸收了线性化残差，代价是约束偏紧，会拒绝一些实际安全的控制。
"""
s = s.replace(old, new.replace("__EQ_UCF__", EQ_UCF), 1)
log.append("4.2.1 单步无碰撞约束 717 → 约 400 字（d_safe 与 n,δ 两式并入正文叙述）")

# ── 4.2.2 ＋ 伪代码 ─────────────────────────────────────────────────────────
old = sect("安全动作集与投影算子")
EQ_SET = block(old, "eq:safeset")
EQ_QP = block(old, "eq:qp")
EQ_AB = block(old, "eq:Ab")
new = r"""\subsubsection{安全动作集与投影算子}

安全动作集取三者之交
__EQ_SET__
它是 $\mathbb{R}^2$ 中的凸多边形。给定策略给出的期望控制，实际执行的控制取其在该集合上的欧氏投影
__EQ_QP__
可行集是若干半平面之交，可统一写成 $U_{\mathrm{safe}}=\{u:Au\le b\}$，其中前四行为动作箱、第五行为合规方向半平面、第六行为式~\eqref{eq:ucf} 的无碰撞约束。这是一个两变量、至多六条约束的二次规划，规模极小，可在每个决策步在线求解。算法~\ref{alg:shield} 给出整层的执行次序。

\begin{algorithm}[htbp]
\caption{投影式安全盾的单步执行}
\label{alg:shield}
\small
\begin{algorithmic}[1]
\Require 本船状态 $s$、他船状态、策略输出 $u_{\mathrm{policy}}$
\Ensure 实际执行的控制 $u_{\mathrm{safe}}$
\State $\rho \gets$ 规则状态机判定的会遇态势
\If{$\rho=\rho_5$}\Comment{紧急态自成一路，不构造约束、不求解}
  \State \Return 紧急控制器输出
\EndIf
\State $U_{\mathrm{colregs}} \gets$ 由 $\rho$ 按式~\eqref{eq:ucolregs} 取用
\State $U_{\mathrm{cf}} \gets$ 由式~\eqref{eq:ucf} 线性化得到
\State $U_{\mathrm{safe}} \gets U_{\mathrm{box}}\cap U_{\mathrm{colregs}}\cap U_{\mathrm{cf}}$
\If{$U_{\mathrm{safe}}\neq\varnothing$}
  \State \Return $\Pi_{U_{\mathrm{safe}}}(u_{\mathrm{policy}})$\Comment{式~\eqref{eq:qp} 的二次规划}
\EndIf
\State \Return 兜底控制\Comment{依次为退化、放松方向约束、碰撞风险最小化}
\end{algorithmic}
\end{algorithm}
"""
s = s.replace(old, new.replace("__EQ_SET__", EQ_SET).replace("__EQ_QP__", EQ_QP), 1)
log.append("4.2.2 补入算法 1（伪代码），Au≤b 的矩阵式并入正文叙述")

# ── 4.2.3 兜底 ──────────────────────────────────────────────────────────────
old = sect("安全动作集为空时的兜底")
new = r"""\subsubsection{安全动作集为空时的兜底}

安全动作集为空时投影无解，此时须由一条不提供保证的替代规则给出控制，本文称之为兜底。兜底的处置按态势分支，而非按一条固定优先级依次尝试，可分为两条互斥的通道。

\textbf{紧急态通道。}\ 紧急态自成一路，该态不构造无碰撞约束、也不求解式~\eqref{eq:qp}，而是直接交给紧急控制器。该控制器是一个有状态的三模式几何律，其生命周期对应一次紧急事件。$\mathrm{ahead}$ 模式在他船位于本船正前方且两船航向近反向时启用，跟踪一个由初始快照确定的固定目标点，该点位于初始位置沿规定转向一侧的横向方向上；$\mathrm{stern}$ 模式在他船位于本船后方扇区且沿纵轴加速可确定解除紧急时启用，在反应窗内直线加速脱离、其后停止施控；其余情形走 $\mathrm{base}$ 模式，跟踪一个随他船当前位置移动的目标点。三种模式都经同一个位置跟踪律换算为控制量。该控制器不提供任何保证，它的作用是在已无合规可行解时给出一个确定的行为，并由性质~\ref{prop:imminent} 的证书判定该局面是否本就不可避。

\textbf{其余态势的兜底。}\ 非紧急态只在安全动作集为空时兜底，且先判一种退化情形。若线性化给不出可用的分离方向，即本船标称落点与他船膨胀占据的圆心重合，则返回箱内投影；否则依次尝试放松方向约束、以及最小化碰撞风险。占据仅仅已经相交并不触发退化，那种情形照常走后两支。放松与碰撞风险最小化都在执行器物理量程上求解并整块丢弃合规约束，故其转艏率可以超出策略动作箱，这一点在统计动作饱和率时按各自的箱分别计算。

上述分支的完备性在实现上是必需的，但在本文使用的公开基准上兜底几乎不发生。我们实测该基准的场景，本船若保持航向航速，与他船的最近中心距中位数约 $1300$~m，只有极少数算例会真正逼近，安全动作集因而基本不会为空。为使这些只在极端态势下才启用的分支得到检验，我们另行构造真冲突算例，结果见第~\ref{sec:res-shield} 节。
"""
s = s.replace(old, new, 1)
log.append("4.2.3 兜底 1,130 → 约 420 字（模式判据式与 stern 控制律移出）")

open(P, "w", encoding="utf-8").write(s)
print("\n".join(log))
