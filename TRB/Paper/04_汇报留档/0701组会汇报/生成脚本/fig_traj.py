#!/usr/bin/env python3
"""轨迹 + 控制输入时序图（Nature 风格·连续 vs 离散·同一对遇场景·平滑度卖点）。"""
import json, glob, math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman'], 'mathtext.fontset': 'stix',
    'font.size': 9, 'axes.labelsize': 10, 'axes.titlesize': 10.5, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'legend.fontsize': 8.5, 'legend.frameon': False, 'savefig.dpi': 300, 'lines.linewidth': 1.6,
})
C_CONT, C_DISC = '#0072B2', '#D55E00'
HOCR = '/Users/weiyutang/Desktop/TRB/结果/结果0629-14:37-数据集-继续测试'
BAL  = '/Users/weiyutang/Desktop/TRB/结果/结果0629-8:48-数据集'
def last(p): return [json.loads(l) for l in open(p) if l.strip()][-1]
def get_traj(path, idx):
    for e in last(path)['final_per']:
        if e.get('scenario_idx') == idx and e.get('traj'):
            return e
    return None

IDX = 0
ec = get_traj(f'{HOCR}/step4e_partial_probeHOCRA_s3.jsonl', IDX)          # 连续 健康
ed = get_traj(f'{BAL}/step4e_partial_probeBalDisc_Discrete-safe_s0.jsonl', IDX)  # 离散 健康

def arrs(e):
    tj = e['traj']
    x = np.array([p['ego_x'] for p in tj]); y = np.array([p['ego_y'] for p in tj])
    psi = np.array([p['ego_psi'] for p in tj]); v = np.array([p['ego_v'] for p in tj])
    n = len(tj); dt = e['ep_len_s'] / max(n-1, 1)
    t = np.arange(n) * dt
    def wrap(d):
        return (d + np.pi) % (2*np.pi) - np.pi
    a = np.diff(v) / dt                       # 加速度 = Δv/Δt
    w = wrap(np.diff(psi)) / dt               # 转艏率 = Δψ/Δt
    return x, y, t, a, w, e['goal_geom']['center'], tj

xc, yc, tc, ac, wc, gc, tjc = arrs(ec)
xd, yd, td, ad, wd, gd, tjd = arrs(ed)

fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))

# (a) 轨迹 x-y
ax = axes[0]
ax.plot(xc, yc, color=C_CONT, label='Continuous (Ours)')
ax.plot(xd, yd, color=C_DISC, label='Discrete (Baseline)', linestyle='--')
ax.scatter([xc[0]], [yc[0]], c='k', s=25, marker='o', zorder=5)
ax.annotate('Start', (xc[0], yc[0]), textcoords='offset points', xytext=(6, 6), fontsize=8)
# 目标框
from matplotlib.patches import Rectangle
gg = ec['goal_geom']; cx, cy = gg['center']
ax.scatter([cx], [cy], c='g', s=40, marker='*', zorder=5)
ax.annotate('Goal', (cx, cy), textcoords='offset points', xytext=(6, -12), fontsize=8, color='g')
ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.set_title('(a) Trajectory'); ax.legend(loc='best'); ax.set_aspect('equal', adjustable='datalim')

# (b) 加速度 a(t)
ax = axes[1]
ax.plot(tc[1:], ac, color=C_CONT, label='Continuous (Ours)')
ax.plot(td[1:], ad, color=C_DISC, label='Discrete (Baseline)', linestyle='--')
ax.set_xlabel('Time (s)'); ax.set_ylabel(r'Acceleration $a$ (m/s$^2$)'); ax.set_title('(b) Throttle Command')

# (c) 转艏率 ω(t)
ax = axes[2]
ax.plot(tc[1:], wc, color=C_CONT, label='Continuous (Ours)')
ax.plot(td[1:], wd, color=C_DISC, label='Discrete (Baseline)', linestyle='--')
ax.set_xlabel('Time (s)'); ax.set_ylabel(r'Yaw Rate $\omega$ (rad/s)'); ax.set_title('(c) Rudder Command')

fig.tight_layout()
out = '/Users/weiyutang/Desktop/TRB/Paper/0701组会汇报/fig_trajectory_control.png'
fig.savefig(out, bbox_inches='tight')
print('✅ 保存', out, '| 连续步数', len(tjc), '离散步数', len(tjd))
