#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""概念示意图(中文标注·Nature风白底)：SE-RL/SP-RL 对比、27维观测空间、COLREGs 四类会遇。
内容全部 fact-based(见 content-research 调研 + 代码)。中文用黑体 STHeiti, 符号用 STIX。"""
import matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Polygon
import numpy as np
mpl.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Songti SC'],
                     'mathtext.fontset':'stix','savefig.dpi':300,'axes.unicode_minus':False})
OUT='/Users/weiyutang/Desktop/TRB/Paper/0701组会汇报/'
CJK='STHeiti'
ACC='#156082'; NAVY='#0E2841'; ORANGE='#E97132'; GRAY='#555555'; LB='#DCE6F1'; LO='#FCE3D2'

def box(ax,x,y,w,h,txt,fc='white',ec=NAVY,fs=11,cjk=True,bold=False,tc='black'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.01,rounding_size=0.02',
                                fc=fc,ec=ec,lw=1.4))
    fam=CJK if cjk else 'serif'
    ax.text(x+w/2,y+h/2,txt,ha='center',va='center',fontsize=fs,family=fam,color=tc,
            wrap=True)

def arrow(ax,x1,y1,x2,y2,ec=NAVY,style='-|>',lw=1.6,ls='-'):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle=style,mutation_scale=14,
                                 lw=lw,color=ec,ls=ls,shrinkA=2,shrinkB=2))

# ============ 图A: 安全盾在环境里(SE-RL) vs 在策略里(SP-RL) ============
fig,axs=plt.subplots(1,2,figsize=(11,3.9));
for ax in axs: ax.set_xlim(0,10); ax.set_ylim(0,7); ax.axis('off')
# --- 左: SE-RL (盾即环境) 我们采用 ---
ax=axs[0]
ax.text(5,6.7,'安全盾在环境里 (SE-RL)   我们采用',ha='center',fontsize=12.5,family=CJK,color=ACC)
box(ax,3.6,5.2,2.8,0.9,'策略 $\\pi_\\theta$',fc=LB,fs=12)
# 安全环境虚框
ax.add_patch(Rectangle((0.6,1.4),8.8,2.8,fc='#F5F7FA',ec=ACC,lw=1.4,ls='--'))
ax.text(1.0,3.95,'安全环境',ha='left',fontsize=10.5,family=CJK,color=ACC)
box(ax,1.2,2.1,3.4,1.2,'安全盾 $\\Phi$\n(投影)',fc=LO,fs=11)
box(ax,5.4,2.1,3.0,1.2,'系统\n(动力学)',fc='white',fs=11)
arrow(ax,5.0,5.2,4.7,3.3)        # 策略 -> 盾 (动作u)
ax.text(4.0,4.3,'动作 $u_t$',ha='right',fontsize=10,family=CJK,color=GRAY)
arrow(ax,4.6,2.7,5.4,2.7)        # 盾 -> 系统 (safe action)
ax.text(5.0,3.05,'$u^\\phi_t$',ha='center',fontsize=10,color=GRAY)
arrow(ax,6.9,3.3,5.6,5.2)        # 系统 -> 策略 (x,r)
ax.text(7.0,4.3,'$x_{t+1}, r_t$',ha='left',fontsize=10,color=GRAY)
ax.text(5,0.7,'梯度不穿过盾, 盾对策略是环境的一部分',ha='center',fontsize=10.5,family=CJK,color=NAVY)
# --- 右: SP-RL (盾在策略里) ---
ax=axs[1]
ax.text(5,6.7,'安全盾在策略里 (SP-RL)',ha='center',fontsize=12.5,family=CJK,color=GRAY)
ax.add_patch(Rectangle((0.6,4.55),8.8,2.05,fc='#F5F7FA',ec=GRAY,lw=1.4,ls='--'))
ax.text(1.0,6.42,'策略(含可微安全盾)',ha='left',fontsize=10.5,family=CJK,color=GRAY)
box(ax,1.2,5.0,3.2,1.1,'策略 $\\pi_\\theta$',fc=LB,fs=11)
box(ax,5.2,5.0,3.2,1.1,'安全盾 $\\Phi$\n(可微层)',fc=LO,fs=11)
arrow(ax,4.4,5.55,5.2,5.55)
box(ax,3.4,1.9,3.2,1.2,'环境\n(动力学)',fc='white',fs=11)
arrow(ax,6.8,5.0,5.6,3.1)
ax.text(6.9,4.0,'安全动作',ha='left',fontsize=10,family=CJK,color=GRAY)
arrow(ax,3.4,3.1,2.0,5.0)
ax.text(2.1,4.0,'$x_{t+1}, r_t$',ha='right',fontsize=10,color=GRAY)
ax.text(5,0.7,'梯度穿过盾回传, 盾是策略的一部分',ha='center',fontsize=10.5,family=CJK,color=GRAY)
fig.text(0.5,0.015,'重画自 Markgraf 等 2026 (arXiv:2509.12833) 图1',ha='center',fontsize=9,family=CJK,color=GRAY)
fig.savefig(OUT+'fig_serl_sprl.png',bbox_inches='tight',pad_inches=0.12,transparent=True); print('✅ serl_sprl')

# ============ 图B: 27维观测空间(忠实复现 Krasowski) ============
fig,ax=plt.subplots(figsize=(11,3.4)); ax.set_xlim(0,27); ax.set_ylim(-0.5,3.6); ax.axis('off')
ax.text(13.5,3.35,'策略观测 = 27 维实数向量 (逐字复现 Krasowski & Althoff 2024)',ha='center',fontsize=12,family=CJK,color=ACC)
groups=[('本船',4,LB),('目标',6,'#E8F0DE'),('他船',12,LO),('终止标志',5,'#EDE3F0')]
x=0
for name,n,fc in groups:
    ax.add_patch(Rectangle((x,2.35),n,0.85,fc=fc,ec=NAVY,lw=1.4))
    ax.text(x+n/2,2.775,f'{name} · {n}维',ha='center',va='center',fontsize=11,family=CJK,color=NAVY)
    x+=n
legend=[('本船 (4)','速度 v · 艏向 θ · 上一步加速度 · 上一步转艏率'),
        ('目标 (6)','到目标距离 · 剩余步数 · 艏向角差 · 纵向偏移 · 横向偏移 · 偏离标志'),
        ('他船 (12)','前 / 左 / 右 / 后 4 扇区 × (距离 · 相对方位 · 距离变化率)'),
        ('终止 (5)','超时 · 出界 · 停船 · 碰撞 · 到达')]
y=1.75
for k,v in legend:
    ax.text(0.3,y,k,ha='left',va='center',fontsize=9.5,family=CJK,color=NAVY)
    ax.text(4.0,y,v,ha='left',va='center',fontsize=9.5,family=CJK,color=GRAY)
    y-=0.42
ax.text(13.5,-0.35,'感知距离 8000 m 以内检测他船, 超出填默认远场值; 输出原始物理量交训练层归一化',ha='center',fontsize=8.5,family=CJK,color=GRAY)
fig.savefig(OUT+'fig_obs_space.png',bbox_inches='tight',pad_inches=0.12,transparent=True); print('✅ obs_space')

# ============ 图C: COLREGs 四类会遇 + 让路/直航角色 ============
def ship(ax,x,y,ang,color,scale=0.42):
    ang=np.deg2rad(ang)
    pts=np.array([[1.4,0],[-0.7,0.7],[-0.7,-0.7]])*scale
    R=np.array([[np.cos(ang),-np.sin(ang)],[np.sin(ang),np.cos(ang)]])
    pts=pts@R.T+np.array([x,y])
    ax.add_patch(Polygon(pts,closed=True,fc=color,ec='black',lw=1.0))
def turnarrow(ax,x,y,ang,color):  # 合规右转示意
    a=np.deg2rad(ang)
    ax.add_patch(FancyArrowPatch((x,y),(x+0.9*np.cos(a),y+0.9*np.sin(a)),
                 arrowstyle='-|>',mutation_scale=11,lw=1.6,color=color,
                 connectionstyle='arc3,rad=-0.5'))

fig,axs=plt.subplots(1,4,figsize=(12,3.2))
titles=['对遇  Rule 14','交叉  Rule 15','追越  Rule 13','直航  Rule 17']
roles=['本船让路 · 右转','本船让路 · 右转','本船让路 · 避让','本船直航 · 保向保速']
for i,(ax,ti,ro) in enumerate(zip(axs,titles,roles)):
    ax.set_xlim(-2.2,2.2); ax.set_ylim(-2.4,2.6); ax.axis('off'); ax.set_aspect('equal')
    ax.text(0,2.35,ti,ha='center',fontsize=12,family=CJK,color=NAVY)
    if i==0:   # 对遇: 本船向上, 他船向下正对
        ship(ax,0,-1.0,90,ACC); ship(ax,0,1.0,270,ORANGE); turnarrow(ax,0,-1.0,90,ACC)
    elif i==1: # 交叉: 本船向上, 他船从右向左
        ship(ax,0,-1.0,90,ACC); ship(ax,1.2,0.3,180,ORANGE); turnarrow(ax,0,-1.0,90,ACC)
    elif i==2: # 追越: 本船在后更快, 从后追上
        ship(ax,0,-1.2,90,ACC); ship(ax,0.1,0.6,90,ORANGE); turnarrow(ax,0,-1.2,90,ACC)
    else:      # 直航: 本船保向, 他船从左来(他船让)
        ship(ax,0,-1.0,90,ACC); ship(ax,-1.2,0.3,0,ORANGE)
        ax.annotate('',xy=(0,0.6),xytext=(0,-0.4),arrowprops=dict(arrowstyle='-|>',color=ACC,lw=1.8))
    ax.text(0,-2.15,ro,ha='center',fontsize=10,family=CJK,color=ACC)
fig.text(0.28,0.02,'本船(蓝)',ha='center',fontsize=9.5,family=CJK,color=ACC)
fig.text(0.42,0.02,'他船(橙)',ha='center',fontsize=9.5,family=CJK,color=ORANGE)
fig.text(0.72,0.02,'让路船须明显机动(累计转向≥20°), 直航船须保向(累计转向<10°)',ha='center',fontsize=9.5,family=CJK,color=GRAY)
fig.savefig(OUT+'fig_encounter.png',bbox_inches='tight',pad_inches=0.12,transparent=True); print('✅ encounter')
print('done')
