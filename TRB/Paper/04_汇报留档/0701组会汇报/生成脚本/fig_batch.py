#!/usr/bin/env python3
"""批量: (1) 治抖效果预览 (2) 种子失败成因 + c_step 系数效果。Nature 风格·英文。"""
import json, glob, statistics
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman'], 'mathtext.fontset': 'stix',
    'font.size': 9, 'axes.labelsize': 10, 'axes.titlesize': 10.5, 'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
    'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'legend.fontsize': 8.5, 'legend.frameon': False, 'savefig.dpi': 300, 'lines.linewidth': 1.8,
})
C_CONT, C_DISC, C_ACC = '#0072B2', '#D55E00', '#009E73'
def last(p): return [json.loads(l) for l in open(p) if l.strip()][-1]
def mn(xs): xs=[x for x in xs if x is not None]; return sum(xs)/len(xs) if xs else float('nan')

# ============ 图1: 治抖效果预览 (jerk vs w + arrival vs w) ============
JIT = '/Users/weiyutang/Desktop/TRB/结果/结果0627-22:36-抖动诊断'
ws = ['0', '0.25', '0.5', '1.0']; wv = [0, 0.25, 0.5, 1.0]
jerk_m, arr_m = [], []
for w in ws:
    js, ars = [], []
    for s in [1, 2, 3]:
        c = glob.glob(f'{JIT}/step4e_partial_diagRate_w{w}_s{s}.jsonl')
        if not c: continue
        fp = last(c[0])['final_per']; reached = [e for e in fp if e['reached']]
        ars.append(100*len(reached)/len(fp))
        jr = mn([e['ctrl_jerk_norm_mean'] for e in reached])
        if jr == jr: js.append(jr)
    jerk_m.append(mn(js)); arr_m.append(mn(ars))
# 用公平同种子 s1,s2 的 jerk 降幅口径也可; 这里用 reached mean
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
ax[0].plot(wv, jerk_m, 'o-', color=C_CONT)
ax[0].set_xlabel(r'Action-Rate Penalty Weight $w$'); ax[0].set_ylabel('Control Jerk'); ax[0].set_title('(a) Smoothness vs Penalty')
drop = (jerk_m[0]-jerk_m[-1])/jerk_m[0]*100
ax[0].annotate(f'-{drop:.0f}% @ w=1.0', (wv[-1], jerk_m[-1]), textcoords='offset points', xytext=(-70, 8), fontsize=8.5, color=C_CONT)
ax[1].plot(wv, arr_m, 's-', color=C_ACC)
ax[1].set_xlabel(r'Action-Rate Penalty Weight $w$'); ax[1].set_ylabel('Arrival Rate (%)'); ax[1].set_title('(b) Arrival Preserved'); ax[1].set_ylim(0, 100)
fig.tight_layout()
fig.savefig('/Users/weiyutang/Desktop/TRB/Paper/0701组会汇报/fig_dither_preview.png', bbox_inches='tight')
print('✅ fig_dither_preview | jerk', [round(x,3) for x in jerk_m], 'arr', [round(x,1) for x in arr_m])

# ============ 图2: 种子失败成因 + c_step 系数效果 ============
fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.2))
# (a) 奖励构成: 到达 vs 未到达 的接近奖励 + 进门奖励(+50 极小)
approach_reach, approach_wander, entry = 5880, 4659, 50
bars = ax[0].bar(['Reach\n(success)', 'Loiter\n(failure)'], [approach_reach, approach_wander],
                 color=[C_CONT, C_DISC], width=0.55, label='Approach reward')
ax[0].bar(['Reach\n(success)'], [entry], bottom=[approach_reach], color='#E69F00', width=0.55, label='Goal-entry bonus (+50)')
ax[0].set_ylabel('Cumulative Reward'); ax[0].set_title('(a) Why Some Seeds Never Enter')
ax[0].legend(loc='lower center', fontsize=7.8)
ax[0].annotate('entry bonus = 0.84% of total\n(too weak to pull policy in)',
               (0, approach_reach+entry), textcoords='offset points', xytext=(-8, 12), fontsize=7.8, ha='center')
ax[0].set_ylim(0, 7200)
# (b) c_step 系数 → 失败种子 s1 到达率
cs = [0, 0.3, 0.5, 0.75]; s1 = [0, 70, 85, 100]
ax[1].plot(cs, s1, 'o-', color=C_CONT, markersize=6)
ax[1].set_xlabel(r'Per-Step Survival Cost $c_{step}$'); ax[1].set_ylabel('Arrival Rate of Failure Seed (%)')
ax[1].set_title('(b) Survival Cost Recovers Failure Seed'); ax[1].set_ylim(-3, 105)
for x, y in zip(cs, s1):
    ax[1].annotate(f'{y}%', (x, y), textcoords='offset points', xytext=(4, -12 if x==0 else 6), fontsize=8)
fig.tight_layout()
fig.savefig('/Users/weiyutang/Desktop/TRB/Paper/0701组会汇报/fig_seed_recovery.png', bbox_inches='tight')
print('✅ fig_seed_recovery')
