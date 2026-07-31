# -*- coding: utf-8 -*-
"""环节 5：图注瘦身 + 浮动体排版修正。

user：图注字太多；表格图片为什么会挤在某一页。

图注原则：只说图上画了什么、以及非显然的编码约定；解读与告诫放正文。
🔴 但两类内容必须留在图注里，因为它们是防误读的：
   · 口径隔离（验证集 100 场景 / 训练期遥测，不是测试集 600）
   · 挑图规则（场景人工挑、种子集合事先声明）
"""
import re

P = "/home/user/TRB-2027-ContinuesPPO/TRB/Paper/01_论文稿/729-paper-中文版.tex"
s = open(P, encoding="utf-8").read()
h = lambda t: len(re.findall(r"[一-鿿]", t))
log = []

# ── ① 浮动体排版：放宽每页可容纳的浮动体比例，减少「攒够一页再一起放」──────
if "\\topfraction" not in s:
    s = s.replace(r"\usepackage[noend]{algpseudocode}",
                  "\\usepackage[noend]{algpseudocode}\n"
                  "% 浮动体排版：默认参数偏保守，会把图表攒成整页一起排 ⟹ 放宽\n"
                  "\\renewcommand{\\topfraction}{0.9}\n"
                  "\\renewcommand{\\bottomfraction}{0.7}\n"
                  "\\renewcommand{\\textfraction}{0.08}\n"
                  "\\renewcommand{\\floatpagefraction}{0.8}\n"
                  "\\setcounter{topnumber}{3}\n"
                  "\\setcounter{totalnumber}{4}", 1)
    log.append("① 放宽浮动体排版参数（topfraction 0.7→0.9、textfraction 0.2→0.08 等），"
               "默认值会把图表攒成整页一起排")

# ── ② 六张图注瘦身 ──────────────────────────────────────────────────────────
CAPS = {
 "方法总览": r"""\textbf{方法总览。}$27$ 维观测经策略网络给出期望控制 $u_{\mathrm{policy}}$，其输出分布为支撑有界的 Beta 分布。规则状态机把当前会遇判入六类态势，安全动作集 $U_{\mathrm{safe}}$ 由执行器量程、合规方向半平面与单步无碰约束求交而成，二次规划把 $u_{\mathrm{policy}}$ 投影为 $u_{\mathrm{safe}}$ 后送入环境。该投影并入环境转移，故安全盾位于训练回路之内。两条旁路互斥，紧急态直接交紧急控制器（$u_{\mathrm{em}}$），其余态势仅在 $U_{\mathrm{safe}}=\varnothing$ 时兜底（$u_{\mathrm{fb}}$）。右下角为动作平面上三个约束集的示意。""",
 "安全—效率权衡": r"""安全—效率权衡。横轴为到达率，纵轴为每局 COLREGs 违规数（对数轴），误差棒为按种子重采样的自助法 $95\%$ 区间，左上角为理想区；内嵌小图纵轴换为碰撞率。形状编码：圆为连续动作、三角为离散动作、方块为外部几何基线；实心为带盾、空心为无盾。外部基线的到达率须按第~\ref{sec:res-ext} 节的限定阅读。""",
 "两项改进的贡献拆解": r"""两项改进的贡献拆解（$2\times2$ 消融，同种子配对）。横轴为四条消融配置，纵轴分别为转艏增量、每局让路违规与到达率。每点一颗种子，细线为同种子配对连线，粗线为中位数，误差棒为按种子重采样的自助法 $95\%$ 区间。""",
 "样本效率与训练可靠性": r"""样本效率与训练可靠性，横轴为训练步数。（a）四种主对照配置的验证集到达率，实线为跨种子中位数、阴影为四分位距；虚线段表示该配置尚有种子在训练，参与统计的种子数已在线端标出。（b）同种子配对，消融·两项都不改 $\rightarrow$ 本文方法。（c）到达率达到收敛判据的种子数。（d）值函数可解释方差。\textbf{纵轴为 $100$ 个验证场景上的数值，与表~\ref{tab:main} 的官方测试集口径不同。}""",
 "盾的行为随训练演化": r"""安全盾的行为随训练演化，取自训练期遥测。（a）本文方法中每一步控制由哪一支产生（对数纵轴）。（b）动作饱和率，转艏轴与加速度轴分开，有界分布、无界高斯与离散网格叠于同图；离散网格的动作上界恰等于连续动作箱半宽，故两者共用同一条打满判据。（c）投影修正量，按动作箱半宽归一化。（d）会遇态势占比（对数纵轴）。\textbf{本图为训练期采集，与表~\ref{tab:main} 的评估口径不同。}""",
 "同一场景下四种方法": r"""同一场景下四种方法的轨迹对比（场景 T-\TBD，种子 s\TBD）。本船轨迹按会遇态势着色，他船轨迹为灰色，圆点间隔一个决策步；虚线圆为性质~\ref{prop:farfield} 的远场阈值。\textbf{本图为定性插图，不承担任何定量声明；场景为人工挑选，种子集合（s0/s1/s2）在跑之前已声明。}""",
}
for key, new in CAPS.items():
    m = re.search(r"\\caption\{[^{}]*?" + re.escape(key), s)
    if not m:
        # caption 内含嵌套花括号，改用逐字符配平
        m = next((mm for mm in re.finditer(r"\\caption\{", s)
                  if key in s[mm.end():mm.end() + 400]), None)
    if not m:
        log.append(f"② ⚠️ 未找到图注：{key}"); continue
    i = m.end(); d = 1
    while d and i < len(s):
        if s[i] == "{": d += 1
        elif s[i] == "}": d -= 1
        i += 1
    old = s[m.end():i - 1]
    s = s[:m.end()] + new + s[i - 1:]
    log.append(f"② 图注「{key}」{h(old)} → {h(new)} 字")

open(P, "w", encoding="utf-8").write(s)
print("\n".join(log))
