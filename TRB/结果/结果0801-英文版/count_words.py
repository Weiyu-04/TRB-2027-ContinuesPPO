# -*- coding: utf-8 -*-
"""TRB 标题页的 Word Count / Total Number of Pages —— **从编译好的 PDF 数，不手数**。

TRB 规矩：`Word Count = 正文词数 + 表数 × 250`。
本脚本的口径（写死在这里，改了要同步改论文标题页）：
  · **算**：摘要 + 正文（§1 到最后一个带编号的节）+ 致谢等收尾小节。
  · **不算**：标题页本身、参考文献表、每页左边距的行号、页码。
  · 表数从 tex 里数 `\begin{table}`。

跑法：python3 结果/结果0801-英文版/count_words.py Paper/01_论文稿/801-paper-英文版.tex
"""
import os
import re
import sys


def main():
    tex = sys.argv[1]
    pdf = os.path.splitext(tex)[0] + ".pdf"
    import fitz
    d = fitz.open(pdf)
    n_pages = d.page_count
    words = 0
    started = stopped = False
    for page in d:
        txt = page.get_text()
        if not started:
            if "ABSTRACT" in txt:
                started = True
                txt = txt[txt.index("ABSTRACT"):]
            else:
                continue
        if not stopped and "REFERENCES" in txt:
            txt = txt[:txt.index("REFERENCES")]
            stopped = True
        elif stopped:
            continue
        #: 去掉行号与页码（纯数字的 token）与页眉
        txt = txt.replace("Tang, Xue, Yang, and Li", " ")
        words += sum(1 for w in txt.split() if not re.fullmatch(r"[\d,.]+", w))
    n_tab = len(re.findall(r"\\begin\{table\}", open(tex, encoding="utf-8").read()))
    print(f"正文词数（摘要+正文+收尾，不含参考文献/行号/页码）= {words}")
    print(f"表数 = {n_tab} · 折算 {n_tab} × 250 = {n_tab*250}")
    print(f"合计 = {words + n_tab*250}")
    print(f"总页数 = {n_pages}")
    print()
    print("标题页照抄这两行：")
    print(f"  Word Count: {words:,} words + {n_tab} table(s) $\\times$ 250 = {words + n_tab*250:,} words")
    print(f"  Total Number of Pages: {n_pages}")


if __name__ == "__main__":
    main()
