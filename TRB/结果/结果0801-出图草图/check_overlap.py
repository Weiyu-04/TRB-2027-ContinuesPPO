# -*- coding: utf-8 -*-
"""出图排版预检：**刻度文字互相重叠** 与 **文字跑出框线**，两件都用渲染后的实际包围盒算。

起因（2026-08-01 later-14）：user 连着两轮抓到同一类问题 ——
「图 2 的 cde 子图 x 轴文字重叠了」「e 图上标记崩塌的红字有个超出边界了」
「图 4 的 c 图 X 轴字体太小」。这些**静态读代码看不出来**，只有渲染后量包围盒才知道。
`CLAUDE.md` 绘图铁律要求「交付前先跑规范自带的静态预检」——本文件是它缺的那一半。

跑法：python3 结果/结果0801-出图草图/check_overlap.py
返回码 0 = 全过；非 0 = 有重叠 / 出框，终端逐条列出是哪张图哪一格。
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                            # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

PAD = 0.5          # 两块文字之间至少留 0.5 pt，纯相切不算重叠
OUT_TOL = 1.0      # 文字探出框线 1 pt 以内算贴边，不报


def _boxes(fig, ax, texts):
    r = fig.canvas.get_renderer()
    out = []
    for t in texts:
        if not t.get_text().strip():
            continue
        try:
            bb = t.get_window_extent(renderer=r)
        except Exception:
            continue
        out.append((t.get_text().replace("\n", "\\n")[:26], bb))
    return out


def check_fig(fig, name, problems):
    fig.canvas.draw()
    for k, ax in enumerate(fig.axes):
        tag = "%s · axes#%d" % (name, k)

        # ① 同一条轴上的刻度文字两两不许相交
        for which, ticks in (("x", ax.get_xticklabels()), ("y", ax.get_yticklabels())):
            bs = _boxes(fig, ax, ticks)
            for i in range(len(bs)):
                for j in range(i + 1, len(bs)):
                    (t1, b1), (t2, b2) = bs[i], bs[j]
                    ov = b1.expanded(1, 1).intersection(b1, b2)
                    if ov is not None and ov.width > PAD and ov.height > PAD:
                        problems.append("%s %s刻度重叠：%r × %r（重叠 %.1f×%.1f pt）"
                                        % (tag, which, t1, t2, ov.width, ov.height))

        # ② 画在数据区里的注记不许探出框线
        axbb = ax.get_window_extent()
        notes = _boxes(fig, ax, ax.texts)
        for txt, bb in notes:
            dx = max(axbb.x0 - bb.x0, bb.x1 - axbb.x1)
            dy = max(axbb.y0 - bb.y0, bb.y1 - axbb.y1)
            if dx > OUT_TOL or dy > OUT_TOL:
                problems.append("%s 注记出框：%r（横向探出 %.1f pt，纵向探出 %.1f pt）"
                                % (tag, txt, max(dx, 0), max(dy, 0)))

        # ③ 注记之间也不许相压（红色 "N collapsed" 那一排就是并排放的）
        for i in range(len(notes)):
            for j in range(i + 1, len(notes)):
                (t1, b1), (t2, b2) = notes[i], notes[j]
                ov = b1.intersection(b1, b2)
                if ov is not None and ov.width > PAD and ov.height > PAD:
                    problems.append("%s 注记互压：%r × %r（重叠 %.1f×%.1f pt）"
                                    % (tag, t1, t2, ov.width, ov.height))


def main():
    import paper_style as PS
    import runs_data as R
    import make_fig23_draft as F23
    import make_fig45_draft as F45

    problems = []
    PS.apply()

    # 让出图脚本照常画，但把 PS.save 换成"顺手体检一遍"，一行业务代码都不改
    real_save = PS.save

    def spy(fig, name, outdirs, also_png=True):
        check_fig(fig, name, problems)
        return real_save(fig, name, outdirs, also_png)

    PS.save = F23.PS.save = F45.PS.save = spy
    try:
        D = R.load_reeval("正式-最佳")
        F23.fig2(D)
        F23.fig3(D)
        F45.main()
    finally:
        PS.save = F23.PS.save = F45.PS.save = real_save
        plt.close("all")

    if problems:
        print("❌ 预检未过，共 %d 条：" % len(problems))
        for p in problems:
            print("   ·", p)
        return 1
    print("✅ 预检通过：刻度文字无重叠、注记无出框。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
