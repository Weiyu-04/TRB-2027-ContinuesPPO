# -*- coding: utf-8 -*-
"""环节 2-C：中文斜体改粗体 + 对标论文的表述改为「先我们、后引用」。

user 2026-08-01：
  · 文中还有斜体 —— 中文排版不用斜体表强调，应改粗体（拉丁文的期刊名保留斜体）
  · 别写得像跟在对标论文后面 —— 改叙述角度，但引用本身保留
"""
import re

P = "/home/user/TRB-2027-ContinuesPPO/TRB/Paper/01_论文稿/729-paper-中文版.tex"
s = open(P, encoding="utf-8").read()
log = []

# ── ① 含中文的 \emph / \textit → \textbf（拉丁文的保持原样：参考文献里的期刊名等）──
def de_italic(m):
    inner = m.group(2)
    if re.search(r"[一-鿿]", inner):
        de_italic.n += 1
        return r"\textbf{" + inner + "}"
    return m.group(0)


de_italic.n = 0
s = re.sub(r"\\(emph|textit)\{([^{}]*)\}", de_italic, s)
log.append(f"① 含中文的斜体 → 粗体：{de_italic.n} 处（参考文献里的拉丁文期刊名保持斜体）")

# ── ② 对标论文的表述：先讲我们做了什么，引用降为从句 ────────────────────────
REPHRASE = [
    # 谓词化开篇
    ("谓词结构、阈值取值与判定优先序沿用 \\citp{Krasowski and Althoff}{2021} 的时序逻辑形式化及其在 "
     "\\citp{Krasowski and Althoff}{2024} 中的实现。本文在此之上新增三项，即碰撞可能性谓词中膨胀半径"
     "与速度集半宽的显式取值、让路态入口的对称化（第~\\ref{sec:sym} 节），以及把这套谓词接到连续动作的约束集上。",
     "我们把会遇态势形式化为一组关于两船相对几何的谓词，并由此定义每类态势下规则允许的控制集；"
     "谓词的结构与阈值遵循海事规则的时序逻辑形式化 \\citp{Krasowski and Althoff}{2021}。"
     "本文在此之上明确给出碰撞可能性谓词中的膨胀半径与速度集半宽，对让路态入口作对称化处理"
     "（第~\\ref{sec:sym} 节），并把整组谓词接到连续动作的约束集上。"),
    # 回报函数
    ("回报结构沿用 \\citp{Krasowski and Althoff}{2024} 的设定，除本节末声明的三处偏离外逐项相同。回报由五项相加而成，",
     "回报由五项相加而成，其结构与 \\citp{Krasowski and Althoff}{2024} 一致以保证可比，另有三处偏离在本节末声明。"),
    # 主指标
    ("主指标沿用 \\citp{Krasowski and Althoff}{2024} 的定义，以保证可比。",
     "主指标的定义与 \\citp{Krasowski and Althoff}{2024} 一致，以保证结果可比。"),
    # 表注里的「对标」
    ("碰撞如实报，与对标的差异按 Fisher 检验陈述，不写“零碰撞”。",
     "碰撞如实报，与离散动作基线的差异按 Fisher 检验陈述，不写“零碰撞”。"),
]
for a, b in REPHRASE:
    if a in s:
        s = s.replace(a, b, 1)
        log.append(f"② 改写：{a[:26]}…")
    else:
        log.append(f"② ⚠️ 未匹配：{a[:26]}…")

open(P, "w", encoding="utf-8").write(s)
print("\n".join(log))
