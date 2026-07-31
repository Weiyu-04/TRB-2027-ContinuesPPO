# -*- coding: utf-8 -*-
"""环节 1.5-A：去花体符号 + 表格不置顶 + 3.1 改名 + 局限并进结论。

user 2026-08-01 的意见：
  · 符号别太花，优先常见样式（点名了花体 U）
  · 表不要一律置顶，放该放的位置
  · 3.1「记号与本船运动学模型」→「运动学模型」
  · 局限不单独成节，压成结论末尾一小段；且不必提环境力建模
"""
import re

P = "/home/user/TRB-2027-ContinuesPPO/TRB/Paper/01_论文稿/729-paper-中文版.tex"
s = open(P, encoding="utf-8").read()
log = []

# ── ① 去花体 ────────────────────────────────────────────────────────────────
# 🔴 \mathcal{P}（安全动作集）不能变成 P —— 转移核 P(·|s,u) 已占用这个字母。
#    改用 U_safe：与 U_box / U_colregs / U_cf 同族，且与元素 u_safe 大小写呼应。
n_p = s.count(r"\mathcal{P}")
s = s.replace(r"\Pi_{\mathcal{P}(s_t,\rho_t)}", r"\Pi_{U_{\mathrm{safe}}(s_t,\rho_t)}")
s = s.replace(r"\Pi_{\mathcal{P}}", r"\Pi_{U_{\mathrm{safe}}}")
s = s.replace(r"\mathcal{P}(s,\rho)", r"U_{\mathrm{safe}}(s,\rho)")
s = s.replace(r"\mathcal{P}", r"U_{\mathrm{safe}}")
log.append(f"① 安全动作集 \\mathcal{{P}} → U_safe（{n_p} 处）—— 避开与转移核 P(·|s,u) 撞车")

for a, b, why in ((r"\mathcal{U}", "U", "控制集族"), (r"\mathcal{O}", "O", "他船占据"),
                  (r"\mathcal{S}", "S", "状态空间"), (r"\mathcal{C}", "C", "碰撞锥"),
                  (r"\mathcal{V}", "V", "速度集")):
    n = s.count(a)
    if n:
        s = s.replace(a, b); log.append(f"   {a} → ${b}$（{n} 处，{why}）")

# 🔴 既有冲突：可清障集 $A$ 与安全集约束矩阵 $A,b$ 同名。约束矩阵的 Au≤b 是通用写法，
#    故改可清障集为 A_clr。
n_a = len(re.findall(r"可清障集", s))
s = re.sub(r"可清障集 \$A\$", r"可清障集 $A_{\\mathrm{clr}}$", s)
s = re.sub(r"(?<![A-Za-z_{])\$A\$(?=[^a-zA-Z]{0,3}(的|上|中|∈|\\in|归属|不变))", r"$A_{\\mathrm{clr}}$", s)
s = s.replace(r"$s\in A$", r"$s\in A_{\mathrm{clr}}$")
s = s.replace(r"A\cap U_{\mathrm{colregs}}", r"A_{\mathrm{clr}}\cap U_{\mathrm{colregs}}")
s = s.replace("& 可清障集", "& 可清障集")          # 符号表那格单独处理
s = s.replace(r"$A$ & 可清障集", r"$A_{\mathrm{clr}}$ & 可清障集")
log.append(f"   🔴 可清障集 $A$ → $A_{{clr}}$（原与安全集约束矩阵 $A,b$ 同名，{n_a} 处提及）")

# ── ② 表格不再一律置顶 ──────────────────────────────────────────────────────
n_t = len(re.findall(r"\\begin\{table\}\[t\]", s))
s = s.replace(r"\begin{table}[t]", r"\begin{table}[htbp]")
log.append(f"② {n_t} 张表 [t] → [htbp]，允许就地排版而非一律顶到页首")

# ── ③ 3.1 改名 ──────────────────────────────────────────────────────────────
s = s.replace(r"\subsection{记号与本船运动学模型}", r"\subsection{运动学模型}", 1)
log.append("③ 3.1「记号与本船运动学模型」→「运动学模型」")

# ── ④ 局限：删节，压成结论末尾一段 ──────────────────────────────────────────
m = re.search(r"\\subsection\{局限\}\s*\n\\label\{sec:limit\}\s*\n\\begin\{enumerate\}.*?\\end\{enumerate\}\n*", s, re.S)
assert m, "没找到局限节"
s = s[:m.start()] + s[m.end():]
log.append("④ 删掉 \\subsection{局限} 整节")

LIMIT_PARA = (
    "\n最后说明本文结论的边界。安全盾提供的保证不包含无碰撞，性质 4 仍为暂定；"
    "方向合规成立于\\emph{每一个投影可行的让路步}而非每一步，紧急态在结构上旁路该约束，"
    "实测每局违规数非零（\\TBD）。性质 1、3、4 均建立在他船恒速外推之上，机动他船会使保证失效，"
    "他船速度上界亦为本基准上的经验值（全池最大 $7.10$~m/s）。到达率与平顺度都不作为效能主张，"
    "前者若无统计上可辨的优势即如实报为与离散基线相当，后者因动作增量惩罚只在连续配置中启用而不可跨动作空间归因。"
    "少数种子存在训练不稳定（\\TBD），这是训练问题而非场景难度。"
    "报数上有两个已知偏倚，最佳存档选择偏倚与跨机器跨轮次抖动，故主表两版并报且强制同机同轮评估。"
    "规则覆盖上，本文实现第 13--17 条的方向要求与直航保向保速，未实现直航船的自行避碰义务及其时机判定。"
    "离散基线为对 \\citp{Krasowski and Althoff}{2024} 的重新实现（其海事评估代码未公开），"
    "故该配置的种子脆弱性归因于本文的复现而非原作者。\n")

anchor = "以及把远场判据实现为 $O(1)$ 的提前退出，进一步压低单步求解耗时。"
assert anchor in s
s = s.replace(anchor, anchor + "\n" + LIMIT_PARA, 1)
log.append(f"   压成结论末尾一段（{len(re.findall(chr(91)+'一-鿿'+chr(93), LIMIT_PARA))} 字），环境力建模按 user 意见不列入")

# 清掉指向局限节的引用
s = s.replace("本文不建模风、流、浪等环境扰动，其影响列入第~\\ref{sec:limit} 节。",
              "本文亦不建模风、流、浪等环境扰动。")
s = s.replace("相应的实证检验需要多他船场景数据，见局限一节。", "相应的实证检验需要多他船场景数据。")
s = re.sub(r"见第~\\ref\{sec:limit\} 节", "见结论", s)
log.append("   清掉全部指向局限节的引用")

open(P, "w", encoding="utf-8").write(s)
print("\n".join(log))
