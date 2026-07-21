#!/usr/bin/env python3
"""方法框架图: 观测→PPO→期望动作→[投影盾: 状态机+约束集+QP]→安全动作→环境。Nature 风格·英文。"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

mpl.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman'],
                     'mathtext.fontset': 'stix', 'savefig.dpi': 300})

fig, ax = plt.subplots(figsize=(12, 4.3)); ax.axis('off')
ax.set_xlim(0, 12); ax.set_ylim(0, 4.3)

def box(x, y, w, h, text, fc='#EAF2FB', ec='#0072B2', fs=10, lw=1.3, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.08',
                                fc=fc, ec=ec, lw=lw))
    ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=fs,
            fontweight='bold' if bold else 'normal')

def arrow(x1, y1, x2, y2, text=None, color='k'):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=14,
                                 lw=1.3, color=color, shrinkA=2, shrinkB=2))
    if text:
        ax.text((x1+x2)/2, (y1+y2)/2+0.18, text, ha='center', va='bottom', fontsize=8.5, style='italic')

# 主流程 (下排)
box(0.2, 1.7, 1.7, 0.9, 'Environment\n(CommonOcean)', fc='#F3F3F3', ec='#555555')
box(2.5, 1.7, 1.6, 0.9, 'Observation', fc='#F3F3F3', ec='#555555')
box(4.7, 1.7, 1.7, 0.9, 'PPO Policy\n(RL agent)', fc='#F3F3F3', ec='#555555')
# 投影盾 (大框, 高亮 = 核心创新)
ax.add_patch(FancyBboxPatch((7.1, 0.55), 3.9, 3.2, boxstyle='round,pad=0.02,rounding_size=0.1',
                            fc='#EAF2FB', ec='#0072B2', lw=2.0, linestyle='-'))
ax.text(9.05, 3.5, 'Projection Safety Shield  (our contribution)', ha='center', va='center',
        fontsize=10.5, fontweight='bold', color='#0072B2')
box(7.35, 2.5, 3.4, 0.72, 'COLREGs State Machine\n$\\rho$: encounter type + compliant direction', fs=8.8)
box(7.35, 1.55, 3.4, 0.72, 'Action Constraint Set\n$U_{box}\\cap U_{colregs}\\cap U_{collision\\text{-}free}$', fs=8.8)
box(7.35, 0.7, 3.4, 0.62, r'QP Projection:  $\min\ \frac{1}{2}\|u-u_{des}\|^2$', fs=9.2, bold=True)
box(11.1, 1.7, 0.0, 0.0, '', fc='none', ec='none')  # spacer

# 箭头
arrow(4.1, 2.15, 4.7, 2.15)                              # obs -> policy
arrow(6.4, 2.15, 7.1, 2.15, r'$u_{des}$')               # policy -> shield
# 盾内竖向流
arrow(9.05, 2.5, 9.05, 2.27)                            # state machine -> constraint
arrow(9.05, 1.55, 9.05, 1.32)                           # constraint -> qp
# 盾出 -> 环境 (回环, 上排)
arrow(9.05, 3.75, 9.05, 4.05); arrow(9.05, 4.05, 1.05, 4.05); arrow(1.05, 4.05, 1.05, 2.6, r'$u_{safe}$')
ax.text(5, 4.22, 'safe control executed in environment', ha='center', fontsize=8.5, style='italic')
# 环境 -> 观测
arrow(1.9, 2.15, 2.5, 2.15)

fig.savefig('/Users/weiyutang/Desktop/TRB/Paper/0701组会汇报/fig_framework.png', bbox_inches='tight', pad_inches=0.1)
print('✅ fig_framework')
