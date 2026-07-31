# -*- coding: utf-8 -*-
"""环节 2-A：重写 §4.6「安全性保证与作用域」3,841 字 → 约 1,000 字。

user 要求：证明可以移出去，但正文留一个精简版，且四条性质要「上下对称」。
做法：四条一律用同一套四段式 —— 陈述（prop 环境）/ 假设 / 证明 / 作用域，
      长度与顺序一致，扫一眼就知道是同一类东西。
完整推导（R_box 分段式、可清障集的三条认证条件、辅助结论与凸性论证、
三步证明的展开）移入 Paper/02_理论推导/，正文不再展开。
"""
import re

P = "/home/user/TRB-2027-ContinuesPPO/TRB/Paper/01_论文稿/729-paper-中文版.tex"
s = open(P, encoding="utf-8").read()
L = s.split("\n")

a = next(i for i, l in enumerate(L) if l.startswith(r"\subsection{安全性保证与作用域}"))
b = next(i for i, l in enumerate(L) if l.lstrip().startswith(r"\begin{table}") and
         any("可证明安全层" in L[j] for j in range(i, min(i + 4, len(L)))))
old = "\n".join(L[a:b])

NEW = r"""\subsection{安全性保证与作用域}
\label{sec:guarantee}

前几节给出的是机制的构造，本节说明它能够\emph{严格}保证什么。四条性质按同一格式给出，即陈述、假设、证明要点与作用域，其中作用域一栏写明该性质\emph{没有}覆盖的情形。全文中“证明”一律指严格引理，存在性级与经验级结论逐条标注。完整推导见补充材料。

\begin{prop}[远场单步无碰]\label{prop:farfield}
设本船与单个他船的中心距为 $d$。若
\begin{equation}
 d \;\ge\; d_{\mathrm{safe}} + v_{\mathrm{obs,max}}\Delta t + \rho_{\mathrm{ego}},
 \label{eq:farfield}
\end{equation}
则一个决策步之后，两船以外接圆保守占据必不相交。其中 $d_{\mathrm{safe}}$ 为两船外接圆半径之和，$\rho_{\mathrm{ego}}$ 为本船单步最大位移。
\end{prop}

\noindent\textit{假设。}\ 单他船，多他船须对每个目标分别满足；他船速度不超过 $9.5~\mathrm{m/s}$，该上界不由式~\eqref{eq:dyn} 约束，须独立给出，我们在全库 $2000$ 个场景、$342{,}000$ 个障碍状态上实测，最大值为 $7.10~\mathrm{m/s}$；他船在步内恒速。

\noindent\textit{证明要点。}\ 由三角不等式，步后中心距不小于 $d$ 减去两船各自的单步位移上界，故不小于 $d_{\mathrm{safe}}$，两外接圆不交，船体亦不交。

\noindent\textit{作用域。}\ 三角不等式与方向无关，故对遇即为最坏情形，几何上没有缺口，这是可证明层中最稳的一条。代入本场景库的常数后阈值约为 $764$~m。它是\emph{单步}充分条件，既非前向不变性，也不覆盖近场；尚未实现为提前退出的快路径。

\begin{prop}[每一让路步的方向合规]\label{prop:comply}
在被状态机判为让路态势、且投影二次规划可行的每一个决策步上，实际执行的 $u_{\mathrm{safe}}$ 必落在式~\eqref{eq:ucolregs} 所定义的合规方向半平面内。
\end{prop}

\noindent\textit{假设。}\ 单他船；状态机的态势判定正确。后者继承碰撞可能性谓词的一处已知保守偏差，该谓词以两船中心距为准，对速度接近的尾随会遇存在漏判区间。

\noindent\textit{证明要点。}\ 合规半平面以硬区间约束进入二次规划，投影解必然满足它。该结论由构造成立，不依赖任何额外引理，因而与训练结果和随机种子无关。

\noindent\textit{作用域。}\ 不覆盖紧急态，其判据在结构上旁路合规约束；不覆盖安全动作集为空而转入兜底的步。多他船下若两条他船要求相反的让路一侧，两个半平面之交为空，该步只能落入兜底并记为一次违规。

\begin{prop}[碰撞不可避的保守判据]\label{prop:imminent}
设单个恒速他船，$R_{\mathrm{box}}(t)$ 为本船可达中心的过近似集，$O(t)$ 为他船船体矩形。若
\begin{equation}
 \exists\,t^\ast\in[0,T]:\quad R_{\mathrm{box}}(t^\ast)\subseteq\bigl(O(t^\ast)\oplus\mathrm{disk}(r_{\mathrm{insc}})\bigr),
 \label{eq:imminent}
\end{equation}
则碰撞不可避，任何可行控制序列都会在 $t^\ast$ 与该他船相撞。
\end{prop}

\noindent\textit{假设。}\ 单个恒速他船；他船船体尺寸不得高估，若调用方传入大于真实值的船宽，判据将不再可靠；判据形如存在性命题，在有限时域 $120$~s 上以 $241$ 个等距时刻求值。

\noindent\textit{证明要点。}\ 三步。其一，速度矢量变化率有界，二重积分给出 $R_{\mathrm{box}}$ 的纵向与横向半轴，故它过近似真可达中心；其二，内切圆欠近似船体；其三，若可达中心整体落入膨胀后的他船占据，则任何控制都无法避开。

\noindent\textit{作用域。}\ 该判据是\emph{可靠}的，即只漏报不误报，但不完备，不触发并不意味着可避。它是末端分类器而非早预警，也不覆盖多障碍与机动他船。验证方面，六路对抗审计在 $20{,}000$ 个随机场景与 $735$ 个精选场景上零假阳，包络硬化后在基准集上追加 $1500$ 例模糊测试仍为零假阳，且判据非空。

\begin{prop}[可清障集的控制不变性，\emph{暂定}]\label{prop:inv}
称控制序列为状态 $s$ 的已认证逃逸，若它逐步可执行、尾段恒速直行、且沿途与他船的距离经可靠证书判定为不减。记全部具有已认证逃逸的状态之集为可清障集 $A_{\mathrm{clr}}$。对每个 $s\in A_{\mathrm{clr}}$，施加其逃逸序列的首个控制一个决策步后所得 $s'$ 仍属于 $A_{\mathrm{clr}}$。
\end{prop}

\noindent\textit{假设。}\ 单个恒速他船；$A_{\mathrm{clr}}$ 为真实可清障集的内近似；证书要求尾段恒\emph{速}，仅航向不变不足以保证距离函数的凸性。

\noindent\textit{证明要点。}\ 施加首个控制之后，原逃逸序列的其余部分就是同一条已通过证书的物理轨迹的后半段，因而仍是新状态的已认证逃逸。合规版本另需首步落在合规半平面内，该构造在让路后继上成立。

\noindent\textit{作用域。}\ \emph{暂定}。$A_{\mathrm{clr}}$ 的控制不变性已证、证书经长时域模糊测试为零假阳、终端约束模块已单测，但把该约束接进部署中的安全盾并在训练所得策略上验证闭环尚未完成。因此本文只声明“已证可清障集不变、并据此设计了终端约束”，不声明“已实现前向不变的安全盾”。

\noindent\textbf{关于多他船。}\ 本文的实证范围是双船会遇，故不展开推广，仅指出一点作用域事实。碰撞自由在数学上是对全部他船的合取，故性质 1 与性质 3 这类充分条件按逐目标合取即可保守推广；方向合规则不能，原因见性质 2 的作用域一栏。这不是本方法的缺陷，而是 COLREGs 自身的边界，该规则集成对制定，对多船冲突并未给出规范。

"""

s = s.replace(old, NEW, 1)
open(P, "w", encoding="utf-8").write(s)
han = len(re.findall(r"[一-鿿]", NEW))
print(f"§4.6 重写完成：{len(re.findall(chr(91)+'一-鿿'+chr(93), old))} 字 → {han} 字")
print("四条性质格式一致：陈述（prop）/ 假设 / 证明要点 / 作用域")
