# -*- coding: utf-8 -*-
"""环节 1.5-B：表 2 并进表 1（消除重复）。

user：表 2 与表 1 的「符号 / 含义」两列确实重复 ⟹ 删表 2，把「取值」并进表 1。
"""
import re

P = "/home/user/TRB-2027-ContinuesPPO/TRB/Paper/01_论文稿/729-paper-中文版.tex"
s = open(P, encoding="utf-8").read()
log = []

# ── ① 抽出表 2 的「符号 → 取值」，然后整块删掉 ────────────────────────────────
t2 = re.search(r"\\begin\{table\}\[htbp\]\s*\n\s*\\caption\{模型与安全盾参数\}.*?\\end\{table\}\n*", s, re.S)
assert t2, "没找到表 2"
VAL = {}
for line in t2.group(0).split("\n"):
    if "&" not in line or "\\toprule" in line or "textbf{类别}" in line:
        continue
    cells = [c.strip() for c in re.sub(r"\\\\\s*$", "", line).split("&")]
    cells = [c for c in cells if c and not c.startswith("\\multirow")]
    if len(cells) >= 3 and cells[-3].startswith("$"):
        VAL[cells[-3]] = cells[-1]
log.append(f"① 从表 2 抽出 {len(VAL)} 条取值，随后删表 2 整块")
s = s.replace(t2.group(0), "", 1)

# ── ② 表 1 → 符号 | 含义 | 取值 × 两栏 ──────────────────────────────────────
old_open = "\\setlength{\\tabcolsep}{3.2pt}\\begin{tabular}{@{}llllll@{}}"
i0 = s.index(old_open)
i1 = s.index("\\midrule", i0) + len("\\midrule\n")
i2 = s.index("\\bottomrule", i1)
body = s[i1:i2]

pairs = []
for line in body.split("\n"):
    t = re.sub(r"\\\\\s*$", "", line.strip())
    if not t:
        continue
    cells = [c.strip() for c in t.split("&")]
    for k in range(0, len(cells) - 1, 2):
        if cells[k] and cells[k + 1]:
            pairs.append((cells[k], cells[k + 1]))
log.append(f"   表 1 原有 {len(pairs)} 条")

have = {p[0] for p in pairs}
extra = [(k, "船体尺度") for k in VAL if k not in have]
pairs += extra
if extra:
    log.append(f"   表 2 独有、补进表 1 的：{[k for k, _ in extra]}")

hit = sum(1 for k, _ in pairs if k in VAL)
rows = []
for k in range(0, len(pairs), 2):
    grp = pairs[k:k + 2]
    while len(grp) < 2:
        grp.append(("", ""))
    rows.append(" & ".join(f"{a} & {b} & {VAL.get(a, '')}" for a, b in grp) + r" \\")

new = ("\\setlength{\\tabcolsep}{3.4pt}\\begin{tabular}{@{}llllll@{}}\n  \\toprule\n"
       "  符号 & 含义 & 取值 & 符号 & 含义 & 取值 \\\\\n  \\midrule\n"
       + "\n".join("  " + r for r in rows) + "\n  ")
s = s[:i0] + new + s[i2:]
log.append(f"② 并表完成：{len(pairs)} 条 · {hit} 条带数值 · {len(rows)} 行 · 两栏「符号|含义|取值」")

s = s.replace(r"\caption{符号表}", r"\caption{符号与参数}", 1)
# 表 2 的表注里有一条实质内容（ω_turn 的推导指路），移进表 1 的表注
s = s.replace("\\end{tabular}\n\\end{table}",
              "\\end{tabular}\n\n \\vspace{4pt}\\footnotesize\n"
              " \\textbf{表注。}$\\omega_{\\mathrm{turn}}$ 为合规转艏率的下确界，其推导见式~\\eqref{eq:omegaturn}。"
              "空白的取值栏表示该符号不对应固定数值。\n\\end{table}", 1)
log.append("   表 2 表注里 ω_turn 的推导指路移进表 1")

open(P, "w", encoding="utf-8").write(s)
print("\n".join(log))
