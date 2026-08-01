#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""英文正式版一致性检查 —— 规范见 `Paper/01_论文稿/英文写作规范.md`。

用法：python3 结果/结果0801-英文版/check_en.py Paper/01_论文稿/801-paper-英文版.tex
退出码：0 = 全过 · 1 = 有硬伤 · 2 = 只有提醒

查六样：
  ① 禁用术语变体（user 2026-08-01：用词必须高度统一，不得变化）
  ② 超长句（对标论文正文 391 句里 >30 词的有 0 句 ⟹ 我们也不该有）
  ③ 红线措辞（provably collision-free / 裸 provably compliant / zero collisions …）
  ④ 摘要词数（≤300）与第一人称（官方 TAD 样例摘要 we=0、our=0）
  ⑤ COLREGs 拼写变体
  ⑥ 悬空承诺（technical report / appendix / supplementary —— TRB 不许附录，技术报告也还没做）
"""
import os
import re
import sys

N_HARD = 0
N_WARN = 0


def hard(msg):
    global N_HARD
    N_HARD += 1
    print("  ❌ " + msg)


def warn(msg):
    global N_WARN
    N_WARN += 1
    print("  ⚠️ " + msg)


def ok(msg):
    print("  ✅ " + msg)


#: 禁用变体 → 应该用什么。键是正则（词边界由脚本加），值是替代词。
BANNED = {
    r"\bships?\b": "vessel(s)",
    r"\bboats?\b": "vessel(s)",
    r"\bego vessels?\b": "own vessel",
    r"\bownships?\b": "own vessel",
    r"\bown ships?\b": "own vessel",
    r"\bobstacle vessels?\b": "target vessel",
    r"\bother vessels?\b": "target vessel",
    r"\bintruders?\b": "target vessel",
    r"\bsafety filters?\b": "safety shield",
    r"\bsafety layers?\b": "safety shield",
    r"\bsafe action set\b": "safe control set",
    r"\badmissible set\b": "safe control set",
    r"\bcontrol space\b": "action space",
    r"\bcontrol inputs?\b": "control command",
    r"\bactuator bounds\b": "actuator limits",
    r"\bencounter types?\b": "encounter situation",
    r"\bencounter scenarios?\b": "encounter situation",
    r"\bsuccess rate\b": "arrival rate",
    r"\binfractions?\b": "violation",
    r"\bsnapshots?\b": "checkpoint",
    r"\bhead on\b": "head-on",
    r"\bover-taking\b": "overtaking",
}

#: 红线：出现即硬伤
REDLINE = [
    (r"provably collision[- ]free", "红线：绝不写 provably collision-free"),
    (r"provably compliant(?!\s+(direction|in the))", "红线：裸的 provably compliant，必须写 provable directional compliance"),
    (r"zero collisions?\b", "红线：绝不写 zero collisions"),
    (r"guarantees? safety\b", "红线：guarantees safety 过强，写清作用域"),
    (r"\balways safe\b", "红线：always safe"),
    (r"complete guarantee", "红线：complete guarantee"),
    (r"technical report", "🔴 技术报告还没做出来（user 2026-08-01：先给我过目再引用）⟹ 现在一句都不许提"),
    (r"\bappendix\b|\bappendices\b", "TRB 不允许附录"),
    (r"supplementary material", "TRB 不允许补充材料"),
]

#: 挂靠句式（弱化对标论文；文献综述节除外，故只报提醒）
LEANING = [r"following \\cit", r"\bas in \\cit", r"consistent with \\cit", r"in line with \\cit"]

#: 🔴 自夸措辞（user 2026-08-01：表达尽量谦虚，不要引起读者不适）
#   只查【自夸的搭配】，不查裸的 significant —— 统计意义上的 significant 是合法的。
BOAST = [
    r"significantly (out)?perform", r"significantly better", r"significantly superior",
    r"far superior", r"vastly", r"dramatically", r"remarkabl", r"outstanding",
    r"impressive", r"excellent (result|performance)", r"state[- ]of[- ]the[- ]art",
    r"clearly demonstrat", r"proves? that (our|the proposed)", r"we solve the problem",
    r"\bsuperior to\b", r"greatly (improv|reduc|enhanc)",
]
#: 🔴 贬低他人的措辞（user 2026-08-01：对方很可能就是审稿人，差别写成取舍、不要写成缺陷）
CRITIQUE = [
    r"fails? to\b", r"cannot handle", r"suffers? from", r"the drawbacks? of",
    r"a weakness of", r"is unable to", r"\bineffective\b", r"\binadequate\b",
    r"\bpoorly\b", r"does not work", r"\bflawed?\b", r"\bnaive(ly)?\b",
]

#: 对标论文的引用键；只准出现在这些节里
ANCHOR_KEY = "krasowski2024"
ANCHOR_OK_SECTIONS = ("Related Work", "Experiments")


def strip_tex(s):
    """把 LaTeX 命令与数学环境剥掉，只留下自然语言，避免误报。"""
    s = re.sub(r"(?m)^\s*%.*$", " ", s)                 # 整行注释
    s = re.sub(r"(?<!\\)%.*$", " ", s, flags=re.M)      # 行尾注释
    s = re.sub(r"\$[^$]*\$", " NUM ", s)                # 行内数学
    s = re.sub(r"\\begin\{(equation|align|algorithm|tabular\*?|table|figure)\}.*?\\end\{\1\}", " ", s, flags=re.S)
    s = re.sub(r"\\(cite|citl|citm|cl|ref|label|hyperlink|hypertarget)\s*\{[^}]*\}", " CITE ", s)
    s = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", s)  # 其余命令
    s = re.sub(r"[{}~\\]", " ", s)
    return re.sub(r"\s+", " ", s)


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    p = sys.argv[1]
    if not os.path.exists(p):
        raise SystemExit(f"🔒 找不到 {p}")
    raw = open(p, encoding="utf-8").read()

    # 正文 = \begin{document} 之后；摘要 = Abstract 到 Keywords 之间
    doc = raw[raw.index(r"\begin{document}"):] if r"\begin{document}" in raw else raw
    m = re.search(r"\\section\*\{Abstract\}(.*?)\\noindent\\textbf\{Keywords", doc, re.S)
    abstract = m.group(1) if m else ""
    # 🔴 正文 = 摘要的 Keywords 之后（标题页的作者块不是句子，算进去会把中位句长打成 100+）
    k = doc.find(r"\noindent\textbf{Keywords")
    body = doc[doc.index("\\newpage", k):] if k > 0 and "\\newpage" in doc[k:] else doc

    text = strip_tex(doc)
    low = text.lower()

    print(f"【英文版一致性检查】{os.path.basename(p)}")

    print("\n① 术语一致性（禁用变体）")
    bad = 0
    for pat, should in BANNED.items():
        hits = re.findall(pat, low)
        if hits:
            hard(f"出现 {len(hits)} 次 {sorted(set(hits))} → 应统一为「{should}」")
            bad += 1
    if not bad:
        ok(f"{len(BANNED)} 条禁用变体一个都没出现")

    print("\n② 句长（对标论文正文 >30 词的句子有 0 句）")
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", strip_tex(body)) if len(s.split()) >= 4]
    lens = [len(s.split()) for s in sents]
    if lens:
        lens_sorted = sorted(lens)
        med = lens_sorted[len(lens_sorted) // 2]
        longs = [s for s in sents if len(s.split()) > 30]
        over20 = sum(1 for x in lens if x > 20)
        print(f"     正文 {len(sents)} 句 · 中位 {med} 词 · >20 词 {over20} 句（{100*over20/len(lens):.0f}%）")
        if longs:
            hard(f"{len(longs)} 句超过 30 词，逐句拆开：")
            for s in longs[:5]:
                print(f"        · ({len(s.split())} 词) {s[:110]}…")
        else:
            ok("没有超过 30 词的句子")
        if med > 16:
            warn(f"中位句长 {med} 词，目标 ≤16（对标论文是 12）")
    else:
        warn("正文还没有内容，跳过句长检查")

    print("\n③ 红线措辞")
    red = 0
    for pat, why in REDLINE:
        if re.search(pat, low):
            hard(f"{why} —— 命中「{re.search(pat, low).group(0)}」")
            red += 1
    if not red:
        ok(f"{len(REDLINE)} 条红线全部没踩")
    lean = [x for pat in LEANING for x in re.findall(pat, doc)]
    if lean:
        warn(f"{len(lean)} 处挂靠句式（following/as in/consistent with + 引用）—— 文献综述节可以，正文要改")

    print("\n③·五 语气（谦虚，不自夸）")
    boast = []
    for pat in BOAST:
        for m in re.finditer(pat, low):
            boast.append(text[max(0, m.start()-45):m.start()+45])
    if boast:
        hard(f"{len(boast)} 处自夸措辞 —— 让数字说话，形容词删掉：")
        for b in boast[:5]:
            print(f"        · …{b.strip()}…")
    else:
        ok(f"{len(BOAST)} 类自夸措辞一个都没有")
    n_novel = len(re.findall(r"\bnovel\b", low))
    if n_novel > 1:
        warn(f"novel 出现 {n_novel} 次 —— 只该在 Novelty 那一段出现一次")
    if re.search(r"(?<!knowledge, )this is the first|we are the first", low):
        hard("无 hedge 的 'the first' —— 写成 'To the best of the authors\' knowledge, this is the first…'")

    print("\n③·六 弱化对标论文（只准出现在 Related Work / Experiments）")
    secs = [(m.start(), m.group(1)) for m in re.finditer(r"\\section\*?\{([^}]*)\}", doc)]
    def sec_of(pos):
        cur = "(标题页/摘要)"
        for pp, nn in secs:
            if pp <= pos:
                cur = nn
            else:
                break
        return cur
    hits = [(m.start(), sec_of(m.start())) for m in re.finditer(ANCHOR_KEY, doc)]
    outside = [(p_, s_) for p_, s_ in hits if not any(a in s_ for a in ANCHOR_OK_SECTIONS)]
    print(f"     对标论文共引 {len(hits)} 处")
    if outside:
        warn(f"{len(outside)} 处在允许之外的节：{sorted(set(s_ for _, s_ in outside))} —— 逐处确认是否确有借用；"
             "确有就留引用但改成事实性归属，没有就删")
    elif hits:
        ok("全部落在 Related Work / Experiments 内")

    crit = []
    for pat in CRITIQUE:
        for m in re.finditer(pat, low):
            crit.append(text[max(0, m.start()-60):m.start()+50])
    if crit:
        warn(f"{len(crit)} 处可能在讲他人方法的毛病 —— 逐处看：差别要写成【取舍】不是【缺陷】")
        for cbit in crit[:4]:
            print(f"        · …{cbit.strip()}…")

    print("\n④ 摘要（≤300 词 · 零第一人称）")
    if abstract:
        at = strip_tex(abstract)
        aw = [w for w in re.split(r"\s+", at) if w.strip(" .,;:")]
        n = len(aw)
        (ok if n <= 300 else hard)(f"摘要 {n} 词（硬上限 300）")
        fp = re.findall(r"\b(we|our|us)\b", at.lower())
        if fp:
            hard(f"摘要出现 {len(fp)} 处第一人称 {sorted(set(fp))} —— 官方 TAD 样例摘要 we=0、our=0，改成被动或 'This paper …'")
        else:
            ok("摘要零第一人称，与官方样例一致")
        for h in ("Objectives", "Methods", "Findings", "Novelty", "Practical Applications"):
            if h not in abstract:
                hard(f"摘要缺小标题「{h}」")
    else:
        warn("没找到摘要节，跳过")

    print("\n⑤ COLREGs 拼写")
    variants = set(re.findall(r"\bCOLREG[Ss]?\b|\bColRegs?\b|\bColregs?\b", text)) - {"COLREGs"}
    (hard if variants else ok)(
        f"出现拼写变体 {sorted(variants)}，全文只用 COLREGs" if variants else "全文只用 COLREGs")
    if re.search(r"\bCOLREGs\b", text) and "International Regulations for Preventing Collisions at Sea" not in text:
        hard("COLREGs 从没给过全称 —— 首次出现必须写 International Regulations for Preventing Collisions at Sea (COLREGs)")

    print("\n" + "=" * 78)
    if N_HARD:
        print(f"❌ {N_HARD} 处硬伤（+{N_WARN} 处提醒）—— 先改再往下写。")
        return 1
    print(f"✅ 全过（{N_WARN} 处提醒）。")
    return 2 if N_WARN else 0


if __name__ == "__main__":
    sys.exit(main())
