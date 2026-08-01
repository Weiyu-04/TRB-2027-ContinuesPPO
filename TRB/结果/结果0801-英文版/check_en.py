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

#: 🔴 圈外术语（user 2026-08-01：「要么是专业的名词，要么就是大家都熟知的表述」）
#   这些词在形式化验证 / 优化圈里都是真术语，但**不是 TRB 读者的常用词**，
#   而且 sound 的日常义（"合理的"）跟术语义（"只漏报不误报"）不一样，会被读反。
#   左边=禁用，右边=改成什么（中文稿本来就是右边那个说法）。
OUTSIDER = {
    r"\bcertificates?\b":      "改说 criterion / test，并把「只漏报不误报」明写出来（中文稿：保守判据）",
    r"\bcertifie[sd]\b":       "改说 applies to / covers",
    r"\bsound(ness)?\b":       "术语义会被读成日常义「合理的」；改说 one-sided + 明写 never raises a false alarm",
    r"\bzonotopes?\b":         "TRB 读者不认；改说 over-approximation / a convex set",
    r"\binfimum\b":            "闭集上可取到；直接说 smallest",
    r"\bsupremum\b":           "直接说 largest",
    r"forward invarian":         "改说 does not extend to the steps after it",
    r"\bepigraph\b":           "改写成不用它的说法",
    r"\bsurjectiv|\binjectiv":  "改写成不用它的说法",
}

#: 对标论文的引用键；只准出现在这些节里
ANCHOR_KEY = "krasowski2024"
ANCHOR_OK_SECTIONS = ("Related Work", "Experiments")


def strip_tex(s):
    """把 LaTeX 命令与数学环境剥掉，只留下自然语言，避免误报。"""
    s = re.sub(r"(?m)^\s*%.*$", " ", s)                 # 整行注释
    s = re.sub(r"(?<!\\)%.*$", " ", s, flags=re.M)      # 行尾注释
    s = re.sub(r"\$[^$]*\$", " NUM ", s)                # 行内数学
    # 🔴 行间公式与各级标题都是**句子边界**：不这么处理，标题会被粘进下一句、
    #   跨公式的半句会被算成一整句，句长检查就全是假阳性（2026-08-01 实测）。
    s = re.sub(r"\\begin\{(equation|align|algorithm|tabular\*?|table|figure)\*?\}.*?\\end\{\1\*?\}", " EQUATION. ", s, flags=re.S)
    s = re.sub(r"\\(sub)*section\*?\{([^}]*)\}", r" \2. ", s)
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

    # 正文 = \begin{document} 之后
    doc = raw[raw.index(r"\begin{document}"):] if r"\begin{document}" in raw else raw
    # 🔴 摘要锚点：原来是 `\noindent\textbf{Keywords`，而关键词已按 2027 投稿清单从摘要页删掉
    #    ⟹ 锚点失效、abstract 变空、④ 整项**静默跳过**（2026-08-01 later-10 发现，
    #    从 later-8 起就没真跑过）。改成按结构性锚点定位：\section*{Abstract} → 下一个 \newpage。
    #    同 `03` L243-续55 B 的教训：锚点要选节命令，不要选内容行。
    m = re.search(r"\\section\*\{Abstract\}(.*?)\\newpage", doc, re.S)
    if m is None:   # 兼容旧稿
        m = re.search(r"\\section\*\{Abstract\}(.*?)\\noindent\\textbf\{Keywords", doc, re.S)
    abstract = m.group(1) if m else ""
    # 🔴 正文 = 摘要的 Keywords 之后（标题页的作者块不是句子，算进去会把中位句长打成 100+）
    # 🔴 正文起点：优先按 §1 定位。原来靠 Keywords 行定位，而关键词已按 2027 投稿清单
    #    从摘要页删掉（清单：摘要页除摘要外不得有其它信息）⟹ 那个锚点会失效，
    #    失效后 body 退化成整份文档，标题页的作者块会被当成句子、句长检查全是假阳性。
    k = doc.find(r"\section{Introduction}")
    if k < 0:
        k = doc.find(r"\noindent\textbf{Keywords")
        body = doc[doc.index("\\newpage", k):] if k > 0 and "\\newpage" in doc[k:] else doc
    else:
        body = doc[k:]

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

    print("\n②·五 破折号（user 2026-08-01：不要使用破折号）")
    # 🔴 在【原始文本】上查，不能用 strip_tex 后的：`$590$--$840$` 会被替换成 `NUM -- NUM`，
    #   数字区间是合法的，那样查会误报。只剥注释。
    rawb = re.sub(r"(?m)^\s*%.*$", " ", body)
    rawb = re.sub(r"(?<!\\)%.*$", " ", rawb, flags=re.M)
    em = re.findall(r"---", rawb) + re.findall(r"(?<=[a-zA-Z]) -- (?=[a-zA-Z])", rawb)
    if em:
        hard(f"{len(em)} 处破折号 —— 换成句号另起一句，或冒号/逗号（`--` 只准用于数字区间）")
    else:
        ok("正文没有破折号")

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

    print("\n③·七 圈外术语（不是 TRB 读者的常用词）")
    outs = []
    for pat, fix in OUTSIDER.items():
        n = len(re.findall(pat, low))
        if n:
            outs.append((n, pat, fix))
    if outs:
        hard(f"{len(outs)} 类圈外术语 —— 换成读者熟的说法：")
        for n, pat, fix in outs:
            print(f"        · {pat}  出现 {n} 次 → {fix}")
    else:
        ok(f"{len(OUTSIDER)} 类圈外术语一个都没有")

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
        hard("没找到摘要节 —— 锚点又失效了，别让这一项静默跳过（`03` L243-续57）")

    print("\n④·二 承诺没兑现（正文说「另有报告」，稿里却没有那张表）")
    # 2026-08-01 later-10：GPT 三份报告同时点到这条。TRB 不许附录 ⟹
    # 稿里没有的东西就不能说「也报了」，要么给出数字、要么删掉承诺。
    PROMISE = [r"(?:are|is) also reported", r"both reported", r"accompany the table",
               r"reported in two versions", r"reported separately in"]
    pr = [m_.group(0) for pat in PROMISE for m_ in re.finditer(pat, low)]
    if pr:
        warn(f"{len(pr)} 处「另有报告」承诺 {sorted(set(pr))} —— 逐处确认稿里真有对应的表/数字，"
             "没有就把数字写进正文或删掉承诺")
    else:
        ok("没有悬空的「另有报告」承诺")

    print("\n④·三 拼写与体例（美式拼写 · 度符号 · 引用逗号）")
    BRIT = {r"\blabell(ed|ing)\b": "labeled / labeling", r"\bbehaviour": "behavior",
            r"\banalys(e|ed|ing)\b": "analyze / analyzed / analyzing",
            r"\bnormalis": "normaliz", r"\bmodelling\b": "modeling",
            r"\bcounter-clockwise\b": "counterclockwise", r"\bcentre\b": "center"}
    bs = [(len(re.findall(pat, low)), pat, fix) for pat, fix in BRIT.items()
          if re.search(pat, low)]
    if bs:
        hard(f"{len(bs)} 类英式拼写 —— 全文统一美式：")
        for n_, pat, fix in bs:
            print(f"        · {pat} 出现 {n_} 次 → {fix}")
    else:
        ok(f"{len(BRIT)} 类英式拼写一个都没有")
    # 度符号：`$20^\circ$` 对；`$20\ ^\circ$` / `20 ^\circ` 错（度符号不留空格，同百分号）
    degsp = re.findall(r"\d\s*(?:\\,|\\ |~|\s)\s*\^\\circ", raw)
    (hard if degsp else ok)(
        f"{len(degsp)} 处度符号与数字之间有空格 —— 度符号不留空格（同百分号）"
        if degsp else "度符号都紧跟数字")
    # 🔴 引用体例 = **(Author Year)，作者与年份之间不加逗号**。
    #    依据是**官方 Word 样例正文实测**（`_模板_官方Word原版_TAD_AMSamplePaper.docx` 的
    #    document.xml 里写的是 `(Smith 2020)` / `(Smith and Jones 2020)` / `(Smith et al. 2020)`），
    #    也与 0731 外部复审 C14 一致（Chicago 作者-年份制无逗号），上一轮已按此删过逗号。
    #    ⚠️ 2026-08-01 later-10：GPT 复查报告第 1 条声称"TRB 指南示例使用逗号"并要求全文加逗号，
    #    **经官方样例核对为错**，已驳回。此检查就是防这一条被再犯一次。
    rawnc = re.sub(r"(?m)^\s*%.*$", " ", raw)          # 注释里有宏用法示例，别当正文查
    withcomma = [t for t in re.findall(r"\\cit[lmp]?\{[^}]*\}\{([^}]*)\}", rawnc)
                 if re.search(r"[A-Za-z.]\s*,\s+(1[89]|20)\d\d[a-z]?\b", t)]
    (hard if withcomma else ok)(
        f"{len(withcomma)} 处引用多了逗号 —— 官方样例是 (Author Year) 不加逗号："
        f"{sorted(set(withcomma))[:4]}"
        if withcomma else "引用都是 (Author Year)，与官方样例一致")

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
