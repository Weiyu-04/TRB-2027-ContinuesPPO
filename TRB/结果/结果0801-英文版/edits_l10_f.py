# -*- coding: utf-8 -*-
"""2026-08-01 later-10 · 第 F 批：腾版面 + 补图 5 / 图 6 的配套分析。

页数账（03 L243-续53 B）：10 pt 一整页约 54 行、一行约 14 词。改完前五批是 21 页，
第 21 页只有 21 行文献尾巴 ⟹ 要腾约 21 行 ≈ 290 词，再加上本批要补的图注分析
（报告2 #32/#33），实际要腾约 350 词。

按 `英文写作规范 §十`「不写只对我们自己有意义的运维细节」逐处压：
  · §5.1 探索期那次小重评的 73% —— 运维细节，且明写「不是正式实验的设定」
  · §5.2 训练预算里 5/3 颗种子怎么分机器 —— 运维细节；结论（跨机差异未排除）保留
  · §2.2 势函数整形那一段（报告2 #1 判可选删）—— 压掉一半，四条引用全留
  · §5.6 单机计时的三句免责 —— 压成一句
  · §4.2.3 假阳性统计两句 —— 压成一句

同时补上报告2 #32/#33 要的东西：图 5 (b)(d) 各一句、图 6 三条轨迹的定性解读与三颗种子的出处。

用法：cd Paper/01_论文稿 && python3 ../../结果/结果0801-英文版/edits_l10_f.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tex_edit import Doc  # noqa: E402


def main():
    d = Doc("801-paper-英文版.tex")

    # ── §5.1：删探索期小重评的运维细节；报告3 B18：step can be negative 指代不明
    d.sub("The unbounded Gaussian also pushes many samples past the action "
          "box. In a small re-evaluation during the exploration phase, which is not the setting of "
          "the formal experiment, about $73\\%$ of the decision steps of that policy sat against the "
          "boundary. The shield comparison therefore uses the plainest continuous configuration. For "
          "the same reason the step from the discrete "
          "baseline to that configuration can be negative, and it should be read as a conservative "
          "lower bound.",
          "The unbounded Gaussian also pushes many samples past the action box. The shield "
          "comparison therefore uses the plainest continuous configuration, and the measured "
          "difference between the discrete baseline and that configuration can be negative. The "
          "comparison is specific to this baseline and is not a general bound on the effect of "
          "shielding.")

    # ── §5.2：训练预算的分机细节是运维内容；结论（跨机差异未排除）必须留
    d.sub("Available machine time did not allow every training run to be placed on one machine. For "
          "$5$ of the $8$ seeds "
          "all configurations were trained on one machine, and the other $3$ seeds are spread over two "
          "or three machines. Those machines run the same image and the same training code, with a "
          "byte-identical fingerprint of the training path, and with the thread count locked to one.",
          "Training itself was spread over several machines, which run the same software image and "
          "the same training code, verified by a checksum, with the thread count locked to one.")

    # ── §2.2：势函数整形段压掉一半（报告2 #1 判可选删；四条引用全留）
    d.sub("A directional constraint tightens the feasible set in close quarters and can cost "
          "terminal efficiency. Potential-based shaping recovers it without changing the set of "
          "optimal policies \\citl{ng1999}{Ng et al. 1999}. A distance potential alone leaves a "
          "stalling mode near the goal \\citl{trott2019}{Trott et al. 2019}, which a single "
          "continuous potential does not remove \\citl{mller2025}{M\\\"uller and Kudenko 2025}. "
          "Shaping can instead be designed against a stated control requirement "
          "\\citl{de2024}{De Lellis et al. 2024}. Here a linear distance kernel is used, with arrival "
          "as a terminal condition. The effect of a shield on performance has been studied separately.",
          "A directional constraint tightens the feasible set in close quarters and can cost terminal "
          "efficiency. Potential-based shaping recovers it without changing the set of optimal "
          "policies \\citl{ng1999}{Ng et al. 1999}, but a distance potential alone leaves a stalling "
          "mode near the goal \\citm{\\cl{trott2019}{Trott et al. 2019}; "
          "\\cl{mller2025}{M\\\"uller and Kudenko 2025}}. Shaping can instead be designed against a "
          "stated control requirement \\citl{de2024}{De Lellis et al. 2024}. Here a linear distance "
          "kernel is used, with arrival as a terminal condition.")

    # ── §5.6：单机计时的三句免责压成一句
    d.sub("\\textbf{These are wall-clock times on one machine. They indicate the order of magnitude "
          "of the computation time on that machine. They are not a comparison against any other "
          "method, which would need the same machine and the same protocol.}",
          "\\textbf{These are wall-clock times on one machine and indicate an order of magnitude "
          "only. They are not a comparison against another method.}")

    # ── §4.2.3：假阳性统计两句压成一句
    d.sub("On $20{,}000$ random scenarios and $735$ selected scenarios it produced no false "
          "positive. After the envelope was hardened, $1{,}500$ further randomized stress-test cases "
          "produced none either.",
          "On $20{,}000$ random scenarios, $735$ selected scenarios, and $1{,}500$ randomized "
          "stress-test cases it produced no false positive.")

    # ══════════ 报告2 #32：图 5 的 (b)(d) 两格没有任何配套分析，且 (d) 的 7.00% 与正文 6.2% 口径不明
    d.sub("It also narrows during training\n(Figure~\\ref{fig:shield}c).",
          "It also narrows during training (Figure~\\ref{fig:shield}c). Panel (b) reports the share "
          "of commands that sit against the boundary of the action box, which stays below the "
          "unshielded configuration throughout. Panel (d) splits the projection steps by situation "
          "over the whole of training, so its emergency share of $7.00\\%$ is a training-wide "
          "average and is not the same quantity as the $6.2\\%$ measured at the end of training.")

    # ══════════ 报告2 #33：图 6 没有实质解读，三颗种子也没有在别处声明过
    d.sub("Figure~\\ref{fig:traj} shows tracks from two encounters of the test set, one head-on and "
          "one crossing.",
          "Figure~\\ref{fig:traj} shows tracks from two encounters of the test set, one head-on and "
          "one crossing. Three seeds were declared in advance for trajectory recording, and $s_0$ is "
          "the first of them. In both encounters the own vessel is the stand-on vessel. Ours holds a "
          "straighter track while the two discrete configurations turn away and back, which is the "
          "behavior that the stand-on violation count measures. Panel (c) shows that all three keep a "
          "similar closest approach, so the difference is in the course held rather than in the "
          "clearance achieved.")
    d.save()


if __name__ == "__main__":
    main()
