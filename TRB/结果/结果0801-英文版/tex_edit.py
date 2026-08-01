# -*- coding: utf-8 -*-
"""正文改写工具 —— 空白无关匹配（03 L243-续52 D 的教训：别拿带换行的整段做字面替换）。

用法：
    from tex_edit import Doc
    d = Doc(path)
    d.sub("原句 原样贴，换行随便", "新句")     # 按空白切开、用 \\s+ 连成正则去找
    d.save()                                    # 替换后 textwrap 回 88 列

每次 sub 都会：① 断言恰好命中 1 处（0 处或多处直接抛错，绝不静默）；② 记录日志。
"""
import re
import textwrap


def ws_pattern(s: str) -> str:
    """把待匹配串按空白切开、每段转义、用 \\s+ 连起来。"""
    parts = [re.escape(p) for p in s.split()]
    return r"\s+".join(parts)


class Doc:
    def __init__(self, path):
        self.path = path
        self.text = open(path, encoding="utf-8").read()
        self.log = []

    def count(self, old: str) -> int:
        return len(re.findall(ws_pattern(old), self.text))

    def sub(self, old: str, new: str, expect: int = 1, wrap: bool = True):
        pat = ws_pattern(old)
        n = len(re.findall(pat, self.text))
        if n != expect:
            raise AssertionError(
                f"命中 {n} 处，期望 {expect} 处。\n待匹配：{' '.join(old.split())[:160]}"
            )
        if wrap and len(new) > 88 and "\n" not in new:
            new_out = "\n".join(textwrap.wrap(new, 88, break_long_words=False,
                                              break_on_hyphens=False))
        else:
            new_out = new
        # 🔴 repl 是**函数**时 re.sub 不解释反斜杠转义 ⟹ 绝不能再 .replace("\\","\\\\")，
        #    那会把 \citl 写成 \\citl（2026-08-01 later-10 踩过，全稿 36 处）。直接原样返回。
        self.text = re.sub(pat, lambda m: new_out, self.text)
        self.log.append((n, " ".join(old.split())[:70], " ".join(new.split())[:70]))
        return self

    def drop(self, old: str, expect: int = 1):
        """整段删除（连同前后多余空白收成一个换行）。"""
        return self.sub(old, "", expect=expect, wrap=False)

    def save(self):
        # 删空留下的连续空行收成一个
        self.text = re.sub(r"\n[ \t]*\n[ \t]*\n+", "\n\n", self.text)
        open(self.path, "w", encoding="utf-8").write(self.text)
        for n, a, b in self.log:
            print(f"  ✓ {a}\n    → {b or '（删）'}")
        print(f"共 {len(self.log)} 处")
