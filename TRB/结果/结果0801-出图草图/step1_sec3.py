# -*- coding: utf-8 -*-
"""环节 1：重写 §3 为 Problem Formulation and Environment Modeling。

动作：
  ① 节名改为 Problem Formulation and Environment Modeling
  ② 三个小节：3.1 记号与本船运动学模型 / 3.2 观测与动作空间 / 3.3 问题表述
  ③ 「观测」从 §4.3.2 搬进 3.2（回报留在方法节，那本来就是方法）
  ④ 「两个动作箱」的区分从 §4.2 搬进 3.2（它是环境属性，不是方法）
  ⑤ 明写不建模风/流/浪，并在局限一节补一条（user 问的「环境力建模」的诚实答复）
  ⑥ 散文收紧到目标字数

🔴 两张表原文逐字保留，不重排。
"""
import re

P = "/home/user/TRB-2027-ContinuesPPO/TRB/Paper/01_论文稿/729-paper-中文版.tex"
s = open(P, encoding="utf-8").read()
L = s.split("\n")
log = []


def grab_table(caption):
    """连同前后的分隔注释一起取出整块 table 环境，逐字保留。"""
    i = next(k for k, l in enumerate(L) if "\\caption{" + caption + "}" in l)
    a = i
    while not L[a].lstrip().startswith("\\begin{table}"):
        a -= 1
    b = i
    while "\\end{table}" not in L[b]:
        b += 1
    return "\n".join(L[a:b + 1])


TAB_NOTATION = grab_table("符号表")
TAB_PARAMS = grab_table("模型与安全盾参数")
log.append(f"两张表逐字取出：符号表 {TAB_NOTATION.count(chr(10))+1} 行 · 参数表 {TAB_PARAMS.count(chr(10))+1} 行")

SEC3 = r"""\section{Problem Formulation and Environment Modeling}
\label{sec:problem-setup}

\subsection{记号与本船运动学模型}
\label{sec:model}

全文所用记号汇总于表~\ref{tab:notation}。角度一律折算到 $(-\pi,\pi]$，并约定右舷为正、左舷为负，故转艏率 $\omega<0$ 表示右转。

%------------------------------------------------------------------------------
__TAB_NOTATION__
%------------------------------------------------------------------------------

受控船舶建模为受偏航约束的质点，状态 $s=[p_x,p_y,\theta,v]^{\!\top}\in\mathbb{R}^4$，以纵向加速度 $a$ 与转艏率 $\omega$ 为控制输入：
\begin{equation}
 \dot s=\bigl[\,v\cos\theta,\ \ v\sin\theta,\ \ \omega,\ \ a\,\bigr]^{\!\top},
 \qquad u=[a,\omega]^{\!\top},
 \label{eq:dyn}
\end{equation}
控制受箱约束 $|a|\le a_{\max}$、$|\omega|\le\omega_{\max}$，航速受 $0\le v\le v_{\max}$ 限制，决策以零阶保持方式每 $\Delta t$ 施加一次。航速界在整个决策步积分\emph{完成之后}才施加，故步内航速可短暂超出 $v_{\max}$；单步位移界因此必须计入步内加速，取 $v_{\max}\Delta t+\tfrac12 a_{\max}\Delta t^2$，步末航速上界记为 $v_{\mathrm{bnd}}=v_{\max}+a_{\max}\Delta t$。安全盾内部用于验证后备机动的积分则在积分过程中即施加饱和，是更保守的模型；全文凡涉及单步位移界的推导一律取前者。

选择这一模型层级出于两点考虑。其一，COLREGs 的让路条款规定的正是航向与航速的变化，而式~\eqref{eq:dyn} 把两者直接作为控制输入，规则因而可以不经中间变量地写成对 $u$ 的约束，这是本文全部安全机制得以落在动作空间里的前提。其二，更精细的水动力模型会引入依船而异、难以辨识的系数，使规则约束与船型参数纠缠，反而削弱结论的普适性。本文不建模风、流、浪等环境扰动，其影响列入第~\ref{sec:limit} 节。

\textbf{建模对象与占据表示。}\ 我们以一条大型集装箱船为对象，几何、操纵性能界与决策周期见表~\ref{tab:params}。其中转艏率上限折合仅约 $1.7^\circ/\mathrm{s}$，一次 $20^\circ$ 的明显转向需十余秒才能完成，\emph{慢于一个决策周期}；这一“操纵慢于决策”的特征既是安全约束必须前瞻的原因，也是后文递归可行性得以成立的几何基础。船体占据用矩形表示，并按用途取两个圆近似：\emph{外接圆} $\mathrm{disk}(R_{\mathrm{circ}})$ 过近似船体，用于构造“一定不碰”的充分条件；\emph{内切圆} $\mathrm{disk}(r_{\mathrm{insc}})$ 欠近似船体，用于构造“一定会碰”的充分条件。两个方向各取一个圆，是两侧结论都单侧保守的关键。他船同样以矩形占据表示，需要外推时取恒速外推，该假设的代价在第~\ref{sec:guarantee} 节逐条标注。

%------------------------------------------------------------------------------
__TAB_PARAMS__
%------------------------------------------------------------------------------

\subsection{观测与动作空间}
\label{sec:obsact}

观测为 $27$ 维实向量：受控船状态 $4$ 维（航速、艏向、纵向加速度、转艏率）；目标相关 $5$ 维（距离、剩余步数、艏向与目标朝向区间之差、纵向偏差、横向偏差）加 $1$ 维“是否已在目标框外”的指示位；交通态势 $12$ 维，即前 / 左 / 右 / 后四个方位扇区各取该扇区内最近他船的三个量（距离、相对方位、相邻两个决策步之间的距离增量，该增量以米为单位、未除以 $\Delta t$）；终止相关 $5$ 维（超时、出界、停住、碰撞、到达）。观测层输出原始物理量，归一化在训练包装层完成。以扇区为单位组织交通信息，使观测维数不随他船数目改变；但这只解决了维数问题——本文部署的安全盾按单他船实现，遇到多于一艘他船时显式报错而非降级运行，因此多他船既未训练也未评估。

动作为二维连续量 $u=(a,\omega)$，此处需要区分两个箱。策略的动作空间是较窄的\emph{常规操作箱} $\{|a|\le a_{\mathrm{op}}\}\cap\{|\omega|\le\omega_{\mathrm{op}}\}$，其半宽恰等于对标工作所用 $7\times7$ 量化网格的张成范围，使两种动作空间的控制权限相同、仅分辨率不同；\emph{执行器物理量程箱} $\mathcal{U}_{\mathrm{box}}=\{|a|\le a_{\max}\}\cap\{|\omega|\le\omega_{\max}\}$ 更宽，只留给安全盾与紧急控制器。放宽是必要的：安全盾必须能在需要时动用超出策略权限的操纵能力，否则一旦策略的窄箱与无碰撞约束不相容，本来可解的局面会被误判为不可行。其代价是安全盾实际施加的控制可以落在策略动作箱之外，这一点在观测中如实回传，并在统计动作饱和率时按各自的箱分别计算。

\subsection{问题表述}
\label{sec:problem}

我们考虑开阔水域的双船会遇，并作如下假设：

\begin{enumerate}\itemsep1pt
 \item[(A1)] 开阔水域，无航道分隔、无静态障碍、无岸线约束；
 \item[(A2)] 场景中有一条受控船与一条他船，均为机动船；
 \item[(A3)] 受控船动力学由式~\eqref{eq:dyn} 描述，控制以 $\Delta t$ 为周期零阶保持；
 \item[(A4)] 他船当前状态无量测误差地可得；
 \item[(A5)] 会遇初始时刻不存在任何已生效的让路义务。
\end{enumerate}

假设 (A2) 不是为简化而作的让步，而是 COLREGs 本身的适用边界：该规则集是\emph{成对}制定的，当三船以上同时相遇、且各自的义务互相冲突时（例如对一条船须保向保速、对另一条须避让），公约本身并未规定如何取舍。多他船情形下各结论如何合成、以及在何处必然失效，见第~\ref{sec:guarantee} 节。

令 $\rho_t$ 为 $t$ 时刻的会遇态势，$\mathcal{U}_{\mathrm{colregs}}(\rho_t)\subseteq\mathbb{R}^2$ 为该态势下规则允许的控制集合（第~\ref{sec:rulesets} 节给出其形式），$\mathcal{U}_{\mathrm{cf}}(s_t)$ 为无碰撞约束集，所求为一个策略
\begin{equation}
 \pi:\ \mathcal{S}\rightarrow\mathbb{R}^2
 \quad\text{使得}\quad
 u_t\in\mathcal{U}_{\mathrm{colregs}}(\rho_t)\cap\mathcal{U}_{\mathrm{cf}}(s_t)
 \quad \forall t.
 \label{eq:problem}
\end{equation}
式~\eqref{eq:problem} 与“把合规写进奖励”有本质区别：它要求\emph{每一步执行的动作}都落在集合内，而非要求某个期望值达到最优。

当动作集有限时，式~\eqref{eq:problem} 有一个直接解法，即枚举全部动作、逐个检验、屏蔽不合规者。这条路要求动作可枚举，因而与连续控制不相容，而船舶的舵与推进本是连续量，量化会同时带来抖动与精度损失。本文的核心思路是把“可枚举”这一前提替换为“可投影”：让路条款在动作空间中恰好表现为半平面约束（第~\ref{sec:rulesets} 节），无碰撞条件可写成线性不等式（第~\ref{sec:proj} 节），故式~\eqref{eq:problem} 的可行集是二维空间中若干半平面与一个矩形之交，为凸多边形；向凸集的投影存在且唯一，可由一个规模极小的二次规划求得。

"""
SEC3 = SEC3.replace("__TAB_NOTATION__", TAB_NOTATION).replace("__TAB_PARAMS__", TAB_PARAMS)

# ── 替换整节 ────────────────────────────────────────────────────────────────
a = next(i for i, l in enumerate(L) if l.startswith(r"\section{Problem Formulation}"))
b = next(i for i, l in enumerate(L) if l.startswith(r"\section{Methodology}"))
while a > 0 and L[a - 1].startswith("%---"):
    a -= 1
old = "\n".join(L[a:b])
s = s.replace(old, SEC3, 1)
log.append(f"§3 整节替换：{b-a} 行 → {SEC3.count(chr(10))+1} 行")

# ── §4.3.2「观测与回报」→「回报函数」，并删掉已搬走的观测段 ─────────────────
s = s.replace(r"\subsubsection{观测与回报}", r"\subsubsection{回报函数}", 1)
s = "\n".join(l for l in s.split("\n") if not l.startswith("观测为 $27$ 维实向量"))
log.append("§4.3.2 改名为「回报函数」，观测段已搬走")

# ── §4.2 里重复的「两个箱」表述收成一句指路 ──────────────────────────────────
i = next((k for k, l in enumerate(s.split("\n")) if "这里需要区分两个箱" in l), None)
if i is not None:
    lines = s.split("\n")
    lines[i] = re.sub(r"这里需要区分两个箱.*$",
                      r"两个动作箱的区分（策略的常规操作箱与执行器物理量程箱）见第~\\ref{sec:obsact} 节。",
                      lines[i])
    s = "\n".join(lines)
    log.append("§4.2 的「两个箱」段收成一句指路，避免与 3.2 重复")

# ── 局限一节：补 label + 环境扰动这一条 ──────────────────────────────────────
s = s.replace("\\subsection{局限}\n\\begin{enumerate}",
              "\\subsection{局限}\n\\label{sec:limit}\n\\begin{enumerate}", 1)
s = s.replace(
    r" \item \textbf{可证明档 $=$ 中。}",
    " \\item \\textbf{未建模环境扰动。}本文的动力学不含风、流、浪等外力，"
    "他船亦按恒速外推。真实海况下这些扰动会同时影响本船落点与他船预测，"
    "从而侵蚀单步无碰所依赖的位移界；将其纳入需要把可达集从确定性过近似换成带扰动的鲁棒过近似。\n"
    " \\item \\textbf{可证明档 $=$ 中。}", 1)
log.append("局限：补 \\label{sec:limit} + 未建模环境扰动一条")

open(P, "w", encoding="utf-8").write(s)
print("\n".join(log))
