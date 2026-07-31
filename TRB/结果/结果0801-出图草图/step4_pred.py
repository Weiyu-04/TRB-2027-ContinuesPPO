# -*- coding: utf-8 -*-
"""环节 2-B：重写 §4.1.1「会遇态势的谓词化」1,898 字 → 约 800 字。

移出正文：方位扇区四谓词、航向关系四谓词、碰撞锥的完整构造、持续性条件的形式化。
留在正文：六类态势、四类义务态势的合成式、合规动作集（半平面）、ω_turn 及 37.5%。
另按 user 要求：正文不再出现「符号 = 数值」的写法，一律用「取 / 为」。
🔴 沿用声明保留（复审要求补的，且后面「本文新增三项」要靠它才立得住），只压缩措辞。
"""
import re

P = "/home/user/TRB-2027-ContinuesPPO/TRB/Paper/01_论文稿/729-paper-中文版.tex"
s = open(P, encoding="utf-8").read()
L = s.split("\n")

a = next(i for i, l in enumerate(L) if l.startswith(r"\subsubsection{会遇态势的谓词化}"))
b = next(i for i, l in enumerate(L) if l.startswith(r"\subsection{安全动作集与投影}"))
old = "\n".join(L[a:b])

# 三块要原样保留的公式
def grab(label):
    m = re.search(r"\\begin\{equation\}[^\\]*?(?:.|\n)*?\\label\{" + re.escape(label) + r"\}\s*\n\\end\{equation\}", old)
    if not m:
        m = re.search(r"\\begin\{equation\}(?:.|\n)*?\\label\{" + re.escape(label) + r"\}(?:.|\n)*?\\end\{equation\}", old)
    assert m, label
    return m.group(0)


EQ_SIT = grab("eq:situations")
EQ_UC = grab("eq:ucolregs")
EQ_TURN = grab("eq:omegaturn")

NEW = r"""\subsubsection{会遇态势的谓词化}

谓词结构、阈值取值与判定优先序沿用 \citp{Krasowski and Althoff}{2021} 的时序逻辑形式化及其在 \citp{Krasowski and Althoff}{2024} 中的实现。本文在此之上新增三项，即碰撞可能性谓词中膨胀半径与速度集半宽的显式取值、让路态入口的对称化（第~\ref{sec:sym} 节），以及把这套谓词接到连续动作的约束集上。

规则状态机把当前会遇归入六类，
\begin{equation*}
\rho\in\{\rho_0\ \text{无冲突},\ \ \rho_1\ \text{直航},\ \ \rho_2\ \text{对遇},\ \ \rho_3\ \text{交叉},\ \ \rho_4\ \text{追越},\ \ \rho_5\ \text{紧急}\},
\end{equation*}
判定由三组谓词合取而成。第一组是方位扇区，按公约的舷灯分界把他船相对本船的方位划入前、左、右、后四区，前扇区半角取 $5^\circ$，舷侧扇区外边界取 $112.5^\circ$。第二组是航向关系，区分两船航向近反平行、近同向，以及他船朝向偏左与偏右，追越航向带半角取 $67.5^\circ$。第三组是碰撞可能性，要求本船速度集与他船碰撞锥相交，且两船相对速度足以在检查时域内消耗掉当前中心距；碰撞锥以本船位置为顶点，对按膨胀半径放大的他船占据作外切，再沿他船速度平移得到。膨胀半径按他船真实船长取其三倍，速度集半宽取 $1~\mathrm{m/s}$，检查时域取 $420$~s，前两个取值在所沿用的文献中未明确给出，此处一并写明。

三组谓词合成四类有义务的态势，
__EQ_SIT__
其余情形归入 $\rho_0$。追越与“本船被追越”两支都需交换两船参数，因为追越要求本船位于他船的后扇区。四个谓词在几何上可能同时成立，故按上式的书写次序取优先。

状态机的转移由持续性条件触发。从无冲突态进入让路态，要求某个让路谓词当前\emph{尚未}成立、却在长度为 $60$~s 的反应窗内逐个决策步恒速外推后持续成立。其中“当前尚未成立”这一合取项是必要的，它把“即将持续成立”与“已经成立”区分开，但也带来一处不对称，详见第~\ref{sec:sym} 节。紧急态另由一个更强的集合预测判据进入，预测时域取 $180$~s。

\textbf{合规动作集。}\ 每一类态势对应一个显式的控制约束集，
__EQ_UC__
两个动作箱的区分见第~\ref{sec:obsact} 节。$\rho_2$ 与 $\rho_3$ 要求向右转，$\rho_4$ 允许择一侧转向，$\rho_1$ 要求保向保速，因而全部落在控制量平面上的\emph{半平面或区间}形式。这正是后文能用投影替代枚举的原因。

\textbf{“足够明显”的量化。}\ 公约要求让路动作的幅度大到能被对方及时察觉。把“明显”量化为在机动时段内航向改变不少于给定阈值，机动时段取 $40$~s、航向阈值取 $20^\circ$，即得
__EQ_TURN__
它是合规转艏率的\emph{下确界}，任何幅度不小于它的转向都合规，且该界不可再降。

由此得到一个可以直接算出来的结构性差异。若动作空间被量化为 $7\times7$ 的 $49$ 点网格，转艏轴取值为 $0$、$\pm0.006$、$\pm0.012$ 与 $\pm0.018~\mathrm{rad/s}$（量化动作空间另设第 $50$ 个动作，它不是网格点，而是紧急态下由紧急控制器在线算出的状态相关动作），则 $0.006~\mathrm{rad/s}$ 那一档在 $40$~s 内只转过 $13.75^\circ$，不足 $20^\circ$，网格上最小的合规档位因而是 $0.012~\mathrm{rad/s}$，比该下确界高出 $37.5\%$。需要限定的是，这一比例是\emph{本文所复现的这一网格}的性质，而非动作离散化的固有性质，更细的网格可以缩小差距，代价是动作数与屏蔽开销的增长。也就是说，量化动作空间每执行一次合规让路，转艏幅度都必须超出规则要求约三成七，代价是多余的航向偏离与额外的舵动作。连续动作可以精确取到该下确界，这一优势与训练过程无关，纯由动作空间的分辨率决定。

"""
NEW = NEW.replace("__EQ_SIT__", EQ_SIT).replace("__EQ_UC__", EQ_UC).replace("__EQ_TURN__", EQ_TURN)

s = s.replace(old, NEW, 1)

# 对称化小节里对 eq:persistent 的引用改成文字（该式已移出正文）
s = s.replace("若一次会遇在\\emph{初始时刻}就已满足让路谓词，则式~\\eqref{eq:persistent} 的必要合取项 $\\neg X$ 恒不成立",
              "若一次会遇在\\emph{初始时刻}就已满足让路谓词，则持续性条件中“当前尚未成立”这一必要合取项恒不成立", 1)
# 让路态入口对称化小节补 label（上面新写的正文两处引用它）
s = s.replace("\\subsubsection{让路态入口的对称化}\n", "\\subsubsection{让路态入口的对称化}\n\\label{sec:sym}\n", 1)

open(P, "w", encoding="utf-8").write(s)
h_old = len(re.findall(r"[一-鿿]", old))
h_new = len(re.findall(r"[一-鿿]", NEW))
print(f"§4.1.1 重写：{h_old} 字 → {h_new} 字")
print("移出正文：方位扇区四谓词、航向关系四谓词、碰撞锥完整构造、持续性条件形式化")
print("保留：六类态势、义务态势合成式、合规动作集（半平面）、ω_turn、37.5% 论证")
