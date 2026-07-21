#!/usr/bin/env python3
"""渲染数学公式为干净图片(白底·Times New Roman/STIX·投PPT)。"""
import matplotlib as mpl, matplotlib.pyplot as plt
mpl.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Songti SC'],'mathtext.fontset':'stix','savefig.dpi':300,'axes.unicode_minus':False})
# 西文/数字/公式用 Times/STIX, 中文注释缺字回退到 Songti(宋体), 两者衬线协调
OUT='/Users/weiyutang/Desktop/TRB/Paper/0701组会汇报/'

# 图1: 基础奖励函数 (Krasowski) — 顶部加中文小标题命名五项(黑体), 公式行不变
fig=plt.figure(figsize=(9.4,3.9)); fig.patch.set_alpha(0)
fig.text(0.035,0.94,'基础奖励(Krasowski 原始系数)     五项之和: 稀疏事件 + COLREGs软惩罚 + 目标进展 + 速度带 + 参考线偏离',
         fontsize=11.5, ha='left', va='center', fontfamily='STHeiti')
lines=[
 (r'$r_t = r_{\mathrm{sparse}} + r_{\mathrm{colregs}} + r_{\mathrm{goal}} + r_{\mathrm{velocity}} + r_{\mathrm{deviate}}$', 0.76, 14),
 (r'$r_{\mathrm{sparse}} = c_{\mathrm{time}}\mathbb{1}_{\mathrm{time}} + c_{\mathrm{goal}}\mathbb{1}_{\mathrm{goal}} + c_{\mathrm{stop}}\mathbb{1}_{\mathrm{stop}} + c_{\mathrm{coll}}\mathbb{1}_{\mathrm{coll}} + \cdots$', 0.58, 12.5),
 (r'$c_{\mathrm{goal}}{=}{+}50,\ \ c_{\mathrm{time}}{=}{-}25,\ \ c_{\mathrm{stop}}{=}{-}40,\ \ c_{\mathrm{coll}}{=}{-}50$', 0.44, 11.5),
 (r'$r_{\mathrm{goal}} = c_{\mathrm{reach}}\,(d_{t-1}-d_t),\qquad c_{\mathrm{reach}}=1.5,\quad d_t=\|p_t-p_{\mathrm{goal}}\|$', 0.28, 12.5),
 (r'$r_{\mathrm{colregs}} = -\sum_{\mathrm{obs}} \frac{1}{1+e^{\gamma_\theta|\varphi|}}\,\alpha_x\,\exp(\min((\zeta_v\hat v_y-\zeta_x)d,0)$', 0.10, 12.5),
]
for t,y,s in lines:
    fig.text(0.035,y,t,fontsize=s,ha='left',va='center')
fig.savefig(OUT+'fig_math_reward.png',bbox_inches='tight',pad_inches=0.15,transparent=True); print('✅ reward math')

# 图2: c_step 非PBRS 证明
fig=plt.figure(figsize=(9.2,3.0)); fig.patch.set_alpha(0)
lines=[
 (r'$\mathrm{Per\text{-}step\ survival\ cost:}\quad r_{\mathrm{sparse}} \leftarrow r_{\mathrm{sparse}} - c_{\mathrm{step}}$', 0.88, 14),
 (r'$\sum_{t=0}^{T-1}\gamma^{t}(-c_{\mathrm{step}}) = -c_{\mathrm{step}}\,\dfrac{1-\gamma^{T}}{1-\gamma}\ \ \Rightarrow\ \ \mathrm{depends\ on\ episode\ length\ }T$', 0.60, 13.5),
 (r'$\mathrm{(non\text{-}PBRS:\ changes\ the\ optimal\ policy,\ penalizes\ long\ loitering)}$', 0.34, 11.5),
 (r'$\mathrm{vs.\ PBRS:}\ \ \sum_t\gamma^{t}F = \gamma^{T}\Phi(s_T)-\Phi(s_0)\ \ \mathrm{(endpoints\ only\Rightarrow cannot\ change\ optimum)}$', 0.10, 11.5),
]
for t,y,s in lines:
    fig.text(0.04,y,t,fontsize=s,ha='left',va='center')
fig.savefig(OUT+'fig_math_cstep.png',bbox_inches='tight',pad_inches=0.15,transparent=True); print('✅ cstep math')

def _draw(fig, lines):
    """'z'=纯中文行(黑体 STHeiti), 'm'=纯公式行(Times/STIX)。中文与公式分行, 绝不混排。"""
    for t, y, s, k in lines:
        if k == 'z':
            fig.text(0.035, y, t, fontsize=s, ha='left', va='center', fontfamily='STHeiti')
        else:
            fig.text(0.035, y, t, fontsize=s, ha='left', va='center')

# 图3: 投影式安全盾数学机制 (第一类核心·公式照 方法数学公式.md §2 · 中文注释与公式分行避免豆腐块)
fig=plt.figure(figsize=(9.6,3.4)); fig.patch.set_alpha(0)
_draw(fig, [
 ('投影式安全盾   把期望动作以最小改动投影到既合规又无碰撞的可行集', 0.92, 12, 'z'),
 (r'$u_{\mathrm{safe}} = \arg\min_{u}\ \frac{1}{2}\,\|u - u_{\mathrm{des}}\|^{2}\qquad \mathrm{s.t.}\ \ u \in U_{\mathrm{box}} \cap U_{\mathrm{colregs}} \cap U_{\mathrm{cf}}$', 0.72, 13.5, 'm'),
 (r'$U_{\mathrm{box}} = \{\,u:\ |a|\leq a_{\max},\ |\omega|\leq\omega_{\max}\,\}$', 0.50, 12, 'm'),
 (r'$U_{\mathrm{cf}} = \{\,u:\ g_\tau^{\top} u \leq h_\tau,\ \forall\tau\,\}$', 0.30, 12, 'm'),
 ('U_box 动作上限箱     U_colregs 合规方向集(状态机按相遇态势给轴对齐约束)     U_cf 无碰撞集(线性化分离超平面, 留安全裕度)', 0.08, 10, 'z'),
])
fig.savefig(OUT+'fig_math_shield.png',bbox_inches='tight',pad_inches=0.15,transparent=True); print('✅ shield math')

# 图4: 两类塑形数学 (第二类·well_B 已保留 + c_step 主攻·势函数塑形 vs 非势函数塑形)
fig=plt.figure(figsize=(9.4,3.7)); fig.patch.set_alpha(0)
_draw(fig, [
 ('进门势塑形 well_B     势函数塑形, 策略不变(Ng 1999, 不改变最优策略)', 0.93, 12, 'z'),
 (r'$\Phi(s) = w_B\,\mathrm{prox}(d)\,\mathrm{align}(\psi),\ \ \mathrm{prox}(d)=\max(0,1-d/R_{\mathrm{near}}),\ \ \mathrm{align}(\psi)=\frac{1}{2}(1+\cos(\psi-\theta_c))$', 0.77, 10.5, 'm'),
 (r'$F(s,s^{\prime}) = \gamma\,\Phi(s^{\prime}) - \Phi(s),\qquad w_B=200,\ \ R_{\mathrm{near}}=500$', 0.61, 10.5, 'm'),
 ('每步生存成本 c_step     非势函数塑形, 改变最优(惩罚长回合游荡)', 0.42, 12, 'z'),
 (r'$r_{\mathrm{sparse}} \leftarrow r_{\mathrm{sparse}} - c_{\mathrm{step}},\qquad \sum_{t}\gamma^{t}(-c_{\mathrm{step}}) = -c_{\mathrm{step}}\,\dfrac{1-\gamma^{T}}{1-\gamma}$', 0.26, 10.5, 'm'),
 (r'$\sum_{t}\gamma^{t}F = \gamma^{T}\Phi(s_T)-\Phi(s_0)$', 0.13, 10.5, 'm'),
 ('对比: 折扣累积依赖回合长度 T 故改变最优, 势函数塑形只依赖首末状态故不改', 0.02, 9.5, 'z'),
])
fig.savefig(OUT+'fig_math_shaping.png',bbox_inches='tight',pad_inches=0.15,transparent=True); print('✅ shaping math')

# 图5: 船舶模型与参数 (第一类固定基础·参数照 usv_dynamics/termination 代码核实)
fig=plt.figure(figsize=(9.6,2.9)); fig.patch.set_alpha(0)
_draw(fig, [
 ('偏航受限点质量模型   集装箱船 SR108   忠实沿用 Krasowski, 不改', 0.90, 12, 'z'),
 (r'$x=[\,p_x,\ p_y,\ \theta,\ v\,]\qquad u=[\,a,\ \omega\,]\qquad \dot{x}=[\,v\cos\theta,\ v\sin\theta,\ \omega,\ a\,]$', 0.60, 13, 'm'),
 ('状态 = 位置 · 艏向 θ · 速度 v (4维)        控制 = 加速度 a · 转艏率 ω (2维, 连续)', 0.34, 10.5, 'z'),
 ('船长 175 m · 船宽 25.4 m · 决策步长 10 s · 速度 0–9.5 m/s · 加速度上限 0.24 · 转艏率上限 0.03', 0.08, 10.5, 'z'),
])
fig.savefig(OUT+'fig_ship.png',bbox_inches='tight',pad_inches=0.15,transparent=True); print('✅ ship')
