#!/usr/bin/env python3
"""训练曲线图（Nature 风格·Times New Roman·英文·连续 vs 离散·多指标一行）。"""
import json, glob
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---- Nature 风格全局 ----
mpl.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman'], 'mathtext.fontset': 'stix',
    'font.size': 9, 'axes.labelsize': 10, 'axes.titlesize': 10.5, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.linewidth': 0.8, 'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'legend.fontsize': 8.5, 'legend.frameon': False, 'figure.dpi': 300, 'savefig.dpi': 300,
    'lines.linewidth': 1.6,
})
C_CONT = '#0072B2'   # 连续(ours) 蓝
C_DISC = '#D55E00'   # 离散(baseline) 橙

HOCR = '/Users/weiyutang/Desktop/TRB/结果/结果0629-14:37-数据集-继续测试'
BAL  = '/Users/weiyutang/Desktop/TRB/结果/结果0629-8:48-数据集'

def last(p): return [json.loads(l) for l in open(p) if l.strip()][-1]

def trend_stack(dir_, tag, key, ns=5):
    """堆叠 5 种子的 trend[key] → (steps, mean, std)。"""
    rows = []; steps = None
    for s in range(ns):
        c = glob.glob(f'{dir_}/step4e_partial_{tag}_s{s}.jsonl')
        if not c: continue
        tr = last(c[0])['trend']
        steps = np.array([t['step'] for t in tr]) / 1e6
        rows.append([t[key] for t in tr])
    a = np.array(rows, dtype=float)
    return steps, a.mean(0), a.std(0)

def curve_stack(dir_, tag, key, ns=5):
    """堆叠 5 种子的 curves[key]（按 step 对齐）→ (steps, mean, std)。"""
    per = {}
    for s in range(ns):
        c = glob.glob(f'{dir_}/step4e_partial_{tag}_s{s}.jsonl')
        if not c: continue
        for pt in last(c[0])['curves']:
            if pt.get(key) is not None:
                per.setdefault(pt['step'], []).append(pt[key])
    steps = sorted(k for k, v in per.items() if len(v) >= 3)
    m = np.array([np.mean(per[k]) for k in steps]); sd = np.array([np.std(per[k]) for k in steps])
    return np.array(steps) / 1e6, m, sd

def band(ax, x, m, sd, color, label):
    ax.plot(x, m, color=color, label=label)
    ax.fill_between(x, m - sd, m + sd, color=color, alpha=0.18, linewidth=0)

fig, axes = plt.subplots(1, 5, figsize=(15, 2.9))

# 1) Episode reward (curves ep_rew_mean)
ax = axes[0]
band(ax, *curve_stack(HOCR, 'probeHOCRA', 'ep_rew_mean'), C_CONT, 'Continuous (Ours)')
band(ax, *curve_stack(BAL, 'probeBalDisc_Discrete-safe', 'ep_rew_mean'), C_DISC, 'Discrete (Baseline)')
ax.set_ylabel('Episode Reward'); ax.set_title('(a) Reward')

# 2) Arrival rate
ax = axes[1]
band(ax, *trend_stack(HOCR, 'probeHOCRA', '到达率%'), C_CONT, 'Continuous (Ours)')
band(ax, *trend_stack(BAL, 'probeBalDisc_Discrete-safe', '到达率%'), C_DISC, 'Discrete (Baseline)')
ax.set_ylabel('Arrival Rate (%)'); ax.set_title('(b) Arrival Rate'); ax.set_ylim(-3, 103)

# 3) Collision rate
ax = axes[2]
band(ax, *trend_stack(HOCR, 'probeHOCRA', '碰撞率%'), C_CONT, 'Continuous (Ours)')
band(ax, *trend_stack(BAL, 'probeBalDisc_Discrete-safe', '碰撞率%'), C_DISC, 'Discrete (Baseline)')
ax.set_ylabel('Collision Rate (%)'); ax.set_title('(c) Collision Rate'); ax.set_ylim(-0.5, 5)

# 4) Violations per episode
ax = axes[3]
band(ax, *trend_stack(HOCR, 'probeHOCRA', '违规次数/局'), C_CONT, 'Continuous (Ours)')
band(ax, *trend_stack(BAL, 'probeBalDisc_Discrete-safe', '违规次数/局'), C_DISC, 'Discrete (Baseline)')
ax.set_ylabel('Violations per Episode'); ax.set_title('(d) Rule Violations')

# 5) Episode length
ax = axes[4]
band(ax, *trend_stack(HOCR, 'probeHOCRA', 'Ep长s'), C_CONT, 'Continuous (Ours)')
band(ax, *trend_stack(BAL, 'probeBalDisc_Discrete-safe', 'Ep长s'), C_DISC, 'Discrete (Baseline)')
ax.set_ylabel('Episode Length (s)'); ax.set_title('(e) Episode Length')

for ax in axes:
    ax.set_xlabel(r'Training Steps ($\times10^6$)')
axes[0].legend(loc='lower right')
fig.tight_layout()
out = '/Users/weiyutang/Desktop/TRB/Paper/0701组会汇报/fig_training_curves.png'
fig.savefig(out, bbox_inches='tight')
print('✅ 保存', out)
