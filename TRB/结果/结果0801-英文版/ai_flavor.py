# -*- coding: utf-8 -*-
"""量「AI 味」——把网上那几条常见判据全部变成可量的数，并与**对标论文**同口径比。

起因（user 2026-08-01 later-16）：GPT 交了一份 68 条的删减报告，同时 user 转来一组
网上流传的「怎么去掉 AI 味」判据。判据本身有道理，但**说我们中不中招得先量**，
不能凭感觉改——否则就是拿一份没依据的意见去改一份有依据的稿子。

量六样，全部与 Krasowski & Althoff 2024（本文对标论文、真人写的、同一圈子）同口径比：
  ① 句长分布（AI 常见毛病：句子普遍长、从句多）
  ② 句子开头的结构多样性（AI 常见毛病：清一色 "The X + 谓语"）
  ③ 破折号 / 冒号 / 括号的密度（AI 常见毛病：破折号成瘾）
  ④ 段落长度的整齐度（AI 常见毛病：每段一样长）
  ⑤ 显性连接词密度（However / Therefore / Moreover / Furthermore …）
  ⑥ 段首加粗提示词（run-in heading）的数量与分布

跑法：python3 结果/结果0801-英文版/ai_flavor.py
"""
import collections
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OURS_PDF = os.path.join(ROOT, "Paper/01_论文稿/801-paper-英文版.pdf")
OURS_TEX = os.path.join(ROOT, "Paper/01_论文稿/801-paper-英文版.tex")
REF_PDF = os.path.join(ROOT, "参考资料/2402.08502v2.pdf")

CONNECTIVES = ["However", "Therefore", "Moreover", "Furthermore", "In addition",
               "Additionally", "Nevertheless", "Consequently", "Thus", "Hence",
               "In contrast", "On the other hand", "It is worth noting",
               "Notably", "Importantly", "Overall", "In summary"]


def pdf_body(path, start_marker, stop_marker, drop=()):
    """从 PDF 抽正文。行号/页码这类纯数字 token 去掉。"""
    import fitz
    d = fitz.open(path)
    out, started, stopped = [], False, False
    for page in d:
        t = page.get_text()
        if not started:
            i = t.find(start_marker)
            if i < 0:
                continue
            started = True
            t = t[i:]
        if not stopped:
            i = t.find(stop_marker)
            if i >= 0:
                t, stopped = t[:i], True
        elif stopped:
            continue
        for s in drop:
            t = t.replace(s, " ")
        out.append(t)
    return "\n".join(out)


#: 🔴 从 PDF 抽出来的"句子"里混着表体、算法框和图注 —— 它们没有句号，会跟前后的真句子
#  粘成一条 150-200 词的怪物，把均值整个抬起来。第一次量对标论文得到「中位 21 · 均值 26.8」，
#  与 `英文写作规范 §一` 早先记录的「中位 12 · 均值 11.0」差一倍多，根因就是这个。
#  ⟹ 下面这些标志一出现就丢掉该句；数字/符号占比过高的也丢（那是表格残渣）。
JUNK = re.compile(r"\b(TABLE|Algorithm \d|Input:|Output:|Setup |Parameter Value|"
                  r"Fig\. \d+[.:]|[A-Z]{4,} [A-Z]{4,})")


def is_prose(s):
    if JUNK.search(s):
        return False
    w = s.split()
    if len(w) < 4:
        return False
    numish = sum(1 for x in w if re.search(r"\d", x) or not re.search(r"[A-Za-z]", x))
    return numish / len(w) < 0.30


def sentences(text):
    """切句。缩写、小数、方程/图表编号不算句末；表体/算法框的残渣直接丢。"""
    t = re.sub(r"\s+", " ", text)
    t = re.sub(r"\b(Fig|Eq|No|vs|cf|e\.g|i\.e|et al|Dr|Prof|Sec|approx)\.", r"\1<DOT>", t)
    t = re.sub(r"(\d)\.(\d)", r"\1<DOT>\2", t)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(\u201c])", t)
    return [p for p in (x.replace("<DOT>", ".").strip() for x in parts) if is_prose(p)]


def opening_shape(s):
    """句子开头的结构分类——看的是"这句用什么起头"，不是第一个单词。"""
    w = s.split()
    if not w:
        return "other"
    w0 = w[0].strip("(\u201c").rstrip(",")
    if w0 in ("The", "A", "An", "This", "That", "These", "Those", "Its", "Their", "It"):
        return "限定词起头"
    if re.match(r"^[\d$(]", w0):
        return "数字/符号起头"
    if w0.endswith("ing"):
        return "分词起头"
    if w0 in ("To", "For", "In", "On", "At", "With", "Without", "Under", "Across",
              "Among", "After", "Before", "By", "From", "Between", "During", "Within"):
        return "介词短语起头"
    if w0 in ("If", "When", "Because", "Since", "Although", "While", "Whereas",
              "As", "Once", "Where", "Unless", "Given"):
        return "从句起头"
    if w0 in CONNECTIVES or " ".join(w[:2]).rstrip(",") in CONNECTIVES:
        return "连接词起头"
    return "其它（含专名/代词/动词等）"


def paragraphs_tex(tex):
    """从 .tex 抽正文段落：只要 §1 之后、参考文献之前，且不在图表/公式/算法环境里。"""
    s = open(tex, encoding="utf-8").read()
    s = s[s.index("\\section{Introduction}"):s.index("\\section*{Acknowledgements}")]
    for env in ("figure", "table", "algorithm", "equation", "align", "itemize", "enumerate"):
        s = re.sub(r"\\begin\{%s\*?\}.*?\\end\{%s\*?\}" % (env, env), "\n\n", s, flags=re.S)
    s = re.sub(r"(?m)^\s*%.*$", "", s)                       # 注释行
    s = re.sub(r"\\(sub)*section\*?\{[^}]*\}", "\n\n", s)     # 节标题
    s = re.sub(r"\\label\{[^}]*\}", " ", s)
    out = []
    for p in re.split(r"\n\s*\n", s):
        p = re.sub(r"\s+", " ", p).strip()
        if len(p.split()) >= 12:
            out.append(p)
    return out


def report(name, body):
    ss = sentences(body)
    L = [len(x.split()) for x in ss]
    print("\n" + "=" * 74)
    print("【%s】句数 %d" % (name, len(ss)))
    print("  ① 句长：中位 %.0f · 均值 %.1f · >25 词 %.1f%% · >30 词 %.1f%%"
          % (statistics.median(L), statistics.mean(L),
             100 * sum(1 for x in L if x > 25) / len(L),
             100 * sum(1 for x in L if x > 30) / len(L)))
    sh = collections.Counter(opening_shape(x) for x in ss)
    top = sh.most_common(1)[0]
    print("  ② 句首结构：最集中的一类 %s 占 %.1f%%（共 %d 类）"
          % (top[0], 100 * top[1] / len(ss), len(sh)))
    for k, v in sh.most_common():
        print("       %-22s %5.1f%%" % (k, 100 * v / len(ss)))
    per1k = lambda n: 1000.0 * n / max(1, len(body.split()))
    print("  ③ 标点密度（每千词）：破折号 %.1f · 冒号 %.1f · 括号 %.1f · 分号 %.1f"
          % (per1k(len(re.findall(r"—|--(?![\d])", body))),
             per1k(body.count(":")), per1k(body.count("(")), per1k(body.count(";"))))
    hits = [(c, len(re.findall(r"(?<![A-Za-z])" + c + r"[ ,]", body))) for c in CONNECTIVES]
    tot = sum(n for _, n in hits)
    print("  ⑤ 显性连接词：共 %d 处，每千词 %.1f —— %s"
          % (tot, per1k(tot), ", ".join("%s×%d" % (c, n) for c, n in hits if n)) or "无")
    return L


def main():
    ours = pdf_body(OURS_PDF, "INTRODUCTION", "REFERENCES", drop=("Tang, Xue, Yang, and Li",))
    ours = re.sub(r"(?m)^\s*\d{1,2}\s*$", " ", ours)          # 行号独占一行
    ref = pdf_body(REF_PDF, "I. INTRODUCTION", "REFERENCES")

    report("本文 正文", ours)
    report("对标论文 Krasowski & Althoff 2024 正文", ref)

    # ④ 段落长度整齐度：只对本文量（对标论文的 PDF 分栏，切段不可靠）
    ps = paragraphs_tex(OURS_TEX)
    W = sorted(len(p.split()) for p in ps)
    print("\n" + "=" * 74)
    print("【④ 段落长度】本文 %d 段：最短 %d · 四分位 %d–%d · 中位 %d · 最长 %d 词"
          % (len(W), W[0], W[len(W) // 4], W[3 * len(W) // 4], statistics.median(W), W[-1]))
    print("     变异系数 %.2f（越接近 0，越像「每段裁成一样长」）"
          % (statistics.pstdev(W) / statistics.mean(W)))

    # ⑥ 段首加粗提示词
    s = open(OURS_TEX, encoding="utf-8").read()
    s = s[s.index("\\section{Introduction}"):s.index("\\section*{Acknowledgements}")]
    for env in ("figure", "table", "algorithm"):
        s = re.sub(r"\\begin\{%s\}.*?\\end\{%s\}" % (env, env), "", s, flags=re.S)
    runin = re.findall(r"\\textbf\{([A-Z][^}]{2,60}?)\.?\}\\?[ ]*\\?", s)
    runin = [r for r in runin if len(r.split()) <= 7 and not r.endswith("%")]
    print("\n【⑥ 段首加粗提示词】共 %d 个" % len(runin))
    for r in runin:
        print("     ·", r)


if __name__ == "__main__":
    sys.exit(main())
