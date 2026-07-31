# -*- coding: utf-8 -*-
"""环节 4：综述对比表 + 多船内容瘦身 + 场景库一节收紧。

user 2026-08-01：
  · 文献综述做成一张汇总表，最能对比出本文的位置
  · 多船的内容没必要占篇幅
  · 场景库与对照配置太啰嗦，不是每件事都要解释前因后果

🔴 多船内容的取舍：删细节、留作用域。
   保留 (A2) 单他船假设本身、一句「COLREGs 成对制定」的理由（这是挡审稿人
   「为什么只做双船」的必要防守）、以及盾遇多船显式报错这条实现事实。
   删掉 §4.6 专门的「关于多他船」整段与表 5 表注里的数学推广讨论。
"""
import re

P = "/home/user/TRB-2027-ContinuesPPO/TRB/Paper/01_论文稿/729-paper-中文版.tex"
s = open(P, encoding="utf-8").read()
h = lambda t: len(re.findall(r"[一-鿿]", t))
log = []

# ── ① 综述对比表 ────────────────────────────────────────────────────────────
TABLE = r"""
%------------------------------------------------------------------------------
\begin{table}[htbp]
 \caption{相关工作在四个维度上的定位}
 \label{tab:related}
 \centering\footnotesize
 \setlength{\tabcolsep}{4pt}
 \begin{tabular}{@{}lllcc@{}}
  \toprule
  代表工作 & 动作空间 & 合规施加方式 & 逐步硬保证 & 含学习 \\
  \midrule
  \citp{Kuwata et al.}{2014} & 连续 & 速度障碍几何构造 & 几何充分条件 & --- \\
  \citp{Johansen et al.}{2016} & 离散分支 & 代价函数择优 & --- & --- \\
  \cit{Lee et al. 2025; Patil et al. 2026} & 连续 & 控制障碍函数 / 凸优化 & 状态量 & --- \\
  \addlinespace
  \citp{Meyer et al.}{2020} & 连续 & 软性奖励 & --- & $\checkmark$ \\
  \citp{Heiberg et al.}{2022} & 连续 & 软性奖励 $+$ 风险门控 & --- & $\checkmark$ \\
  \citp{Müller et al.}{2026} & 连续 & 软性奖励 $+$ 证伪训练 & --- & $\checkmark$ \\
  \addlinespace
  \citp{Alshiekh et al.}{2018} & 离散 & 动作屏蔽 & 状态量 & $\checkmark$ \\
  \citp{Dalal et al.}{2018} & 连续 & 线性化后投影 & 状态量 & $\checkmark$ \\
  \citp{Vaaler et al.}{2024} & 连续 & 预测式安全滤波 & 仅碰撞 & $\checkmark$ \\
  \addlinespace
  \citp{Krasowski and Althoff}{2024} & 离散（$49$ 点） & 动作屏蔽 & \textbf{方向合规} & $\checkmark$ \\
  \textbf{本文} & \textbf{连续} & \textbf{投影（二次规划）} & \textbf{方向合规} & $\checkmark$ \\
  \bottomrule
 \end{tabular}

 \vspace{4pt}\footnotesize
 \textbf{表注。}“逐步硬保证”一列指该方法在\emph{每一个决策步}上能排除掉哪一类动作，而非期望意义下的约束满足。
 前三行为非学习方法。本文与 \citp{Krasowski and Althoff}{2024} 的差别只在动作空间一列，
 而与其余连续动作工作的差别在保证一列。本文的方向合规保证限于投影可行的让路步，作用域见第~\ref{sec:guarantee} 节。
\end{table}
%------------------------------------------------------------------------------
"""
anchor = "\\subsection{安全强化学习与可证明合规}"
assert anchor in s
s = s.replace(anchor, TABLE + "\n" + anchor, 1)
log.append("① 新增表「相关工作在四个维度上的定位」（11 行，本文与对标只差动作空间一列）")

# 表已承载对比，正文里重复的对比句删掉
for a, b in [
    ("由此形成一处结构性空缺。海事场景中规则合规的硬保证此前只在离散动作空间上实现过，其代表是把 COLREGs 形式化为时序逻辑谓词与状态机、并在 $49$ 个离散动作上施加动作屏蔽的工作 \\citp{Krasowski and Althoff}{2024}，该工作也提供了本文使用的场景库与状态机阈值，并把连续动作下的动作投影列为未来方向 \\citp{Kochdumper et al.}{2023}。沿该方向的后续工作转入连续动作，却以软性奖励结合证伪式训练来逼近合规，没有给出硬保证 \\citp{Müller et al.}{2026}。本文在连续控制上给出方向合规的构造式保证，并以 zonotope 加二次规划投影 \\citp{Markgraf et al.}{2026} 替代多项式 zonotope 表示以降低单步求解开销。",
     "表~\\ref{tab:related} 汇总了上述工作在四个维度上的定位，由此可见一处结构性空缺。海事场景中规则合规的硬保证此前只在离散动作空间上实现过 \\citp{Krasowski and Althoff}{2024}，该工作也提供了本文使用的场景库与状态机阈值，并把连续动作下的动作投影列为未来方向 \\citp{Kochdumper et al.}{2023}；沿该方向的后续工作虽转入连续动作，却以软性奖励逼近合规，没有给出硬保证 \\citp{Müller et al.}{2026}。本文在连续控制上给出方向合规的构造式保证，并以 zonotope 加二次规划投影 \\citp{Markgraf et al.}{2026} 替代多项式 zonotope 表示以降低单步求解开销。"),
]:
    if a in s:
        s = s.replace(a, b, 1); log.append("   正文里与表重复的对比句收紧")

# ── ② 多船瘦身 ──────────────────────────────────────────────────────────────
# 删 §4.6 专门那段
m = re.search(r"\\noindent\\textbf\{关于多他船。\}.*?(?=\n\n|\n%---)", s, re.S)
if m:
    s = s[:m.start()] + s[m.end():]
    log.append(f"② 删 §4.6「关于多他船」整段（{h(m.group(0))} 字）")
# 表 5 表注里的多船讨论换成一句
old_note = ("\\textbf{表注。}性质 1 与性质 3 的\\textbf{数学}作用域可按目标逐一合取地推广到多他船（碰撞自由是对全部他船的合取），\n"
            " 性质 2 不能（方向要求相反时两个半平面的交为空）。但本文\\textbf{部署}的安全盾按单他船实现，遇到多于一艘他船时显式报错，\n"
            " 故上述推广未经实现，也没有多他船的实验数据支撑。表中凡提“多他船”一律只指数学作用域。")
new_note = "\\textbf{表注。}本文部署的安全盾按单他船实现，遇到多于一艘他船时显式报错，故全表结论均限于双船会遇。"
if old_note in s:
    s = s.replace(old_note, new_note, 1); log.append("   表 5 表注的多船讨论收成一句")
# 性质 1 假设、性质 2 作用域里的多船句收短
s = s.replace("单他船，多他船须对每个目标分别满足；", "单他船；", 1)
s = s.replace("多他船下若两条他船要求相反的让路一侧，两个半平面之交为空，该步只能落入兜底并记为一次违规。", "", 1)
s = s.replace("多他船下随 $N$ 下降；方向冲突 $\\to$ 兜底（COLREGs 成对性边界）；", "", 1)
s = s.replace("$\\forall i$ 合取；", "", 1)
s = s.replace("$\\exists i$ 可合成、可避性不可合成；", "", 1)
s = s.replace("远场单步无碰为严格结论且可按目标合取推广到多他船；", "远场单步无碰为严格结论；", 1)
s = s.replace("已证（初等引理；多他船合取不额外保守）", "已证（初等引理）", 1)
log.append("   性质 1/2 与表 5 各处的多船旁支一并清掉，只保留 (A2) 假设与「成对制定」那一句理由")

# ── ③ 场景库与对照配置收紧 ──────────────────────────────────────────────────
L = s.split("\n")
a = next(i for i, l in enumerate(L) if l.startswith(r"\subsection{场景库与对照配置}"))
b = next(i for i, l in enumerate(L) if l.lstrip().startswith(r"\subsubsection{九种配置"))
old = "\n".join(L[a:b])
NEW = r"""\subsection{场景库与对照配置}
\label{sec:setup}

实验在 CommonOcean 基准 \citp{Krasowski and Althoff}{2022} 的双船会遇场景库上进行，共 $2000$ 个场景。每个场景规定受控船的初始位姿与航速、他船的初始状态与既定航迹、以及受控船的目标区域（含位置门、朝向门与时限）；会遇几何程序化生成，覆盖不同的相对方位、航向交角与接近速率。他船船长 $200$--$300$~m，回合上限 $170$ 个决策步。

按状态机的几何类型谓词对全库分类，交叉与对遇两类占满全库，\textbf{追越场景数为零}，官方测试集中交叉 $395$ 个、对遇 $205$ 个。式~\eqref{eq:ucolregs} 的追越一支因而无法在本场景库上得到实证检验，结论中如实列出。

场景库的官方划分为训练 $1400$ 与测试 $600$。我们把官方训练侧再切为 $1300$ 个训练场景与 $100$ 个验证场景，验证集只用于训练期监控与存档选取，测试集在训练与调参全程不被访问。训练集与测试集零交集，经独立复算核验，故评估样本量即为 $600$。本文此前的探索性实验使用过一个较小的场景子集，其样本量与此处不同，两套样本量下的数值不可放入同一张表。

"""
s = s.replace(old, NEW, 1)
log.append(f"③ 场景库与对照配置 {h(old)} → {h(NEW)} 字")

open(P, "w", encoding="utf-8").write(s)
print("\n".join(log))
