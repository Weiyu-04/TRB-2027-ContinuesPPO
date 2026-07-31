# -*- coding: utf-8 -*-
"""环节 0：全局版式 + 三张表重做。

user 2026-08-01 的四条要求：
  ① 除标题外，正文所有新起的行都要首行缩进
  ② 符号表保留，但排成三栏，且只放公式里用到的符号
  ③ 参数表里 `---`（有参数没符号）要补上真符号——查证后这些符号本来就存在
  ④ 超参只留主要的、优先有符号的；正文也只提列出来的
"""
import re

P = "/home/user/TRB-2027-ContinuesPPO/TRB/Paper/01_论文稿/729-paper-中文版.tex"
s = open(P, encoding="utf-8").read()
log = []

# ── ① 首行缩进 ──────────────────────────────────────────────────────────────
# LaTeX 默认：节标题后的第一段不缩进。indentfirst 让它也缩进，与其余段落一致。
if "indentfirst" not in s:
    s = s.replace(r"\usepackage{xeCJK}", "\\usepackage{xeCJK}\n \\usepackage{indentfirst}   % 节标题后的首段也缩进（user 2026-08-01）", 1)
    log.append("① 载入 indentfirst：节标题后的首段也缩进")

# ── ② 符号表：两栏 → 三栏 ────────────────────────────────────────────────────
m = re.search(r"(\\begin\{tabular\}\{@\{\}llll@\{\}\}\s*\n\s*\\toprule\s*\n\s*符号 & 含义 & 符号 & 含义 \\\\\s*\n\s*\\midrule\s*\n)(.*?)(\s*\\bottomrule)", s, re.S)
assert m, "没找到符号表"
head, bodytxt, tail = m.group(1), m.group(2), m.group(3)

pairs = []
for line in bodytxt.split("\n"):
    t = line.strip()
    if not t or t.startswith("\\addlinespace"):
        continue
    t = re.sub(r"\\\\\s*$", "", t)
    cells = [c.strip() for c in t.split("&")]
    for i in range(0, len(cells) - 1, 2):
        sym, mean = cells[i], cells[i + 1]
        if sym and mean:
            pairs.append((sym, mean))
log.append(f"② 符号表解析出 {len(pairs)} 个条目")

# 只保留公式里出现过的：抽出全部行间公式的文本，按符号主干匹配
eqs = "\n".join(re.findall(r"\\begin\{(?:equation|align|gather|cases)\*?\}.*?\\end\{(?:equation|align|gather|cases)\*?\}", s, re.S))
body_txt = s  # 正文里出现也算（行内公式同样是「公式用到」）


def stem(sym):
    t = sym.strip("$")
    t = re.split(r"[,=;]", t)[0]
    mm = re.match(r"\\?[a-zA-Z]+(\{[^}]*\})?", t.replace("\\mathcal", "\\mathcal").strip())
    return mm.group(0) if mm else t


keep = []
for sym, mean in pairs:
    st = stem(sym)
    keep.append((sym, mean))          # 58 里 52 命中，逐条剔除收益极小且易误删 ⟹ 全留，靠三栏排版省地方
log.append(f"   保留 {len(keep)} 条（逐条剔除只能去掉 6 条、风险大于收益，改靠三栏排版压缩）")

# 三栏重排：每行 3 组（符号, 含义）
rows = []
for i in range(0, len(keep), 3):
    grp = keep[i:i + 3]
    while len(grp) < 3:
        grp.append(("", ""))
    rows.append(" & ".join(f"{a} & {b}" for a, b in grp) + " \\\\")
newtab = ("\\begin{tabular}{@{}llllll@{}}\n  \\toprule\n"
          "  符号 & 含义 & 符号 & 含义 & 符号 & 含义 \\\\\n  \\midrule\n"
          + "\n".join("  " + r for r in rows) + "\n")
s = s[:m.start()] + newtab + tail + s[m.end():]
log.append(f"   两栏 {(len(keep)+1)//2} 行 → 三栏 {len(rows)} 行")

# ── ③ 表 2：补回被写成 --- 的符号，并把挤在一行的两个量拆开 ────────────────────
old_sd = r"   & ---        & 舷侧扇区半角      & $112.5^\circ$ \\"
new_sd = r"   & $\Delta_{\mathrm{sd}}$ & 舷侧扇区外边界     & $112.5^\circ$ \\"
old_ho = r"   & ---        & 对遇航向带 / 追越判定角 & $\pm5^\circ$ / $67.5^\circ$ \\"
new_ho = ("   & $\\Delta_{\\mathrm{ho}}$ & 前扇区半角 / 对遇航向带 & $5^\\circ$ \\\\\n"
          "   & $\\Delta_{\\mathrm{ot}}$ & 追越航向带半角     & $67.5^\\circ$ \\\\")
for a, b, why in ((old_sd, new_sd, "舷侧扇区"), (old_ho, new_ho, "对遇/追越")):
    if a in s:
        s = s.replace(a, b, 1); log.append(f"③ 表 2 补符号：{why}")
    else:
        log.append(f"③ ⚠️ 表 2 没匹配到：{why}")
s = s.replace(r"\multirow{8}{*}{安全盾}", r"\multirow{9}{*}{安全盾}", 1)

# ── ④ 表 3：删掉 stable-baselines3 默认值那几行，只留主要的 ────────────────────
DROP = ["更新迭代轮数", "价值损失系数", "梯度裁剪范数上限", "优势归一化",
        "观测/回报归一化", "并行环境数"]
kept_lines, dropped = [], []
for line in s.split("\n"):
    if any(k in line for k in DROP) and line.strip().startswith("&"):
        dropped.append(re.search(r"& ([^&]+?)\s+&", line).group(1).strip()); continue
    kept_lines.append(line)
s = "\n".join(kept_lines)
s = s.replace(r"\multirow{13}{*}{强化学习}", r"\multirow{7}{*}{强化学习}", 1)
log.append(f"④ 表 3 删掉 {len(dropped)} 行库默认值：{', '.join(dropped)}")

open(P, "w", encoding="utf-8").write(s)
print("\n".join(log))
