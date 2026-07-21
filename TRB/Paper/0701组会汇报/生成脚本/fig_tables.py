#!/usr/bin/env python3
"""论文级三线表（无底色·无竖线·仅顶/中/底三横线·英文·Times New Roman）。"""
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman'],
                     'mathtext.fontset': 'stix', 'savefig.dpi': 300})

def three_line_table(headers, rows, colw, fname, fontsize=10.5, title=None):
    n_rows = len(rows) + 1
    fig_w = sum(colw) * 1.05
    fig_h = 0.42 * n_rows + (0.4 if title else 0.15)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h)); ax.axis('off')
    ax.set_xlim(0, sum(colw)); ax.set_ylim(0, n_rows)
    xs = [sum(colw[:i]) for i in range(len(colw))]  # 各列左边界
    def cellx(i): return xs[i] + colw[i]/2
    y_top = n_rows - 0.15
    lw = 1.3
    # 三线
    ax.plot([0, sum(colw)], [n_rows-0.1, n_rows-0.1], 'k-', lw=lw)          # 顶线
    ax.plot([0, sum(colw)], [n_rows-1.05, n_rows-1.05], 'k-', lw=0.8)       # 表头下中线
    ax.plot([0, sum(colw)], [0.1, 0.1], 'k-', lw=lw)                        # 底线
    # 表头
    for i, h in enumerate(headers):
        ax.text(cellx(i), n_rows-0.6, h, ha='center', va='center', fontsize=fontsize, fontweight='bold')
    # 数据
    for r, row in enumerate(rows):
        y = n_rows - 1.55 - r
        for i, cell in enumerate(row):
            ax.text(cellx(i), y, str(cell), ha='center', va='center', fontsize=fontsize)
    if title:
        ax.text(sum(colw)/2, n_rows+0.35, title, ha='center', va='center', fontsize=fontsize, style='italic')
    fig.savefig(fname, bbox_inches='tight', pad_inches=0.08)
    print('✅', fname)

# 表1 主对比
three_line_table(
    ['Method', 'Arrival\n(median %)', 'IQM', 'Failure\nRate', 'Collision\n(%)', 'Violations\n/ep', 'Control\nJerk', 'Control\nEffort', 'Path\n(m)'],
    [['Continuous (Shield + well-B)', '82.5', '80.8', '1/5', '0', '1.53', '1.583', '1.308', '4228'],
     ['Continuous (+ survival cost)', '85.0', '78.3', '1/5', '0', '1.48', '1.494', '1.309', '4333'],
     ['Discrete (Krasowski baseline)', '90.0', '87.5', '1/5', '0', '1.06', '0.849', '0.895', '4225']],
    colw=[3.6, 1.35, 0.9, 1.0, 1.1, 1.1, 1.05, 1.05, 0.95],
    fname='/Users/weiyutang/Desktop/TRB/Paper/0701组会汇报/table_main_comparison.png',
    title='Table 1. Continuous vs. discrete on the frozen HO/CR set (5 seeds, 40 held-out scenarios).')

# 表2 c_step 系数
three_line_table(
    ['Survival Cost $c_{step}$', 'Failure-Seed Arrival (%)', 'Episode Length (s)'],
    [['0 (baseline)', '0', '1700 (timeout)'],
     ['0.3', '70', '557'],
     ['0.5', '85', '582'],
     ['0.75', '100', '526']],
    colw=[2.6, 2.6, 2.4],
    fname='/Users/weiyutang/Desktop/TRB/Paper/0701组会汇报/table_cstep.png',
    title='Table 2. Per-step survival cost recovers the failure seed (monotonic).')
