#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M1 停车控制器 v4 —— 复审 wru3dhitu 修 3 处:
  ①冲过门(along<0/门在身后)掉头恢复(v3 会一路跑飞·致命bug)
  ②降速度增益 kv·dt≤1 治倒车/抖动 + 用 clip_velocity(v≥0·真环境口径)
  ③朝向对齐留 buffer
诚实全空间验证:门西 + 门东overshoot + 紧预算 + 朝向余量分布。"""
import math, numpy as np, sys
sys.path.insert(0,'/Users/weiyutang/Desktop/TRB/代码')
from trb_env.usv_dynamics import make_vessel_params, step, DECISION_DT
from trb_env.usv_env import A_NORMAL_OMEGA_MAX
P=make_vessel_params(); A_MAX,W_MAX=P.a_max,P.w_max
def wrap(a): return (a+math.pi)%(2*math.pi)-math.pi

def dock_controller(state, goal, theta_g=0.0, wmax=W_MAX,
                    Ld=140.0, k_align=math.radians(22), v_run=2.6, v_turn=0.7, v_creep=1.2,
                    kv=0.06, close_dist=110.0, close_cross=28.0):
    px,py,th,v=state; gx,gy=goal
    c,s=math.cos(theta_g),math.sin(theta_g)
    dx,dy=gx-px,gy-py
    along=dx*c+dy*s; cross=-dx*s+dy*c
    heading_err=wrap(theta_g-th)
    if along >= 0.0:                                    # 门在前方=正常接近
        la=max(0.0, along-Ld) if along>Ld else 0.0
        tx=gx-la*c; ty=gy-la*s
        desired=math.atan2(ty-py, tx-px)
        if along<close_dist and abs(cross)<close_cross: # 近门+在线→对准门朝向
            desired=theta_g
    else:                                               # ①门在身后(冲过门)→掉头指回门·别锁门朝向
        desired=math.atan2(gy-py, gx-px)                # 指向门中心(在西/身后)=掉头重新进近
    corr=wrap(desired-th)
    omega=float(np.clip(corr/DECISION_DT, -wmax, wmax))
    # ②速度调度(kv 小·禁倒车靠 clip_velocity):大扭头/掉头→慢;近门在线→蠕行;否则跑
    if along<0.0 or abs(corr)>k_align: v_t=v_turn
    elif along<close_dist:             v_t=v_creep
    else:                              v_t=v_run
    a=float(np.clip(kv*(v_t-v), -A_MAX, A_MAX))
    return np.array([a, omega])

def make_gate(gx,gy,tg=0.0,L=400.0,Wd=60.0,tol=0.17): return dict(gx=gx,gy=gy,th=tg,L=L,W=Wd,tol=tol)
def in_gate(s,g):
    px,py,th,v=s; c,ss=math.cos(g['th']),math.sin(g['th']); dx,dy=px-g['gx'],py-g['gy']
    return (abs(dx*c+dy*ss)<=g['L']/2) and (abs(-dx*ss+dy*c)<=g['W']/2) and (abs(wrap(th-g['th']))<=g['tol'])
def rollout(s0,goal,gate,budget,tg=0.0,wmax=W_MAX):
    s=np.array(s0,float); margin=None
    for k in range(budget):
        if in_gate(s,gate):
            return True,k,math.degrees(g_margin(s,gate))
        s=step(s, dock_controller(s,goal,tg,wmax=wmax), DECISION_DT, P, clip_velocity=True)  # v≥0 真口径
    return False,budget,None
def g_margin(s,gate): return gate['tol']-abs(wrap(s[2]-gate['th']))

if __name__ == '__main__':                                  # 🔴修·`03` L178 复审(agent C·LOW/WRONG)：开环网格挪进 __main__，
    # 否则 `from dock_controller_v4 import dock_controller`(闭环 harness line16)会触发这 3600 rollout 副作用打印+浪费秒数(每次 import 重跑)。纯网格自测走 `python dock_controller_v4.py`。
    gx,gy,tg=5000.0,1122.0,0.0; gate=make_gate(gx,gy,tg)
    for wmax,wname in [(W_MAX,'满物理±0.03'),(A_NORMAL_OMEGA_MAX,'RL同款±0.018')]:
        print(f'\n===== 权限={wname} · 完整全空间(门西 along>0 + 门东overshoot along<0) =====')
        tot=cap=0; margins=[]; fw=fe=cw=ce=0
        for along0 in [-300,-150,-60, 60,100,150,200,300,400]:   # 负=门东overshoot·正=门西
            for cr in [-60,-30,0,30,60]:
                for hd in [0,45,90,135,180,-45,-90,-135]:
                    for v0 in [0.5,2.0,4.0,6.0,9.0]:
                        px=gx-along0; py=gy+cr; th=wrap(math.radians(hd))
                        ok,k,mg=rollout([px,py,th,v0],(gx,gy),gate,60,tg,wmax)
                        tot+=1; cap+=ok
                        east=along0<0
                        if east: ce+=1; fe+=ok
                        else:    cw+=1; fw+=ok
                        if ok and mg is not None: margins.append(mg)
        print(f'  全空间捕获 {cap}/{tot}={100*cap/tot:.1f}%')
        print(f'   门西(正常接近) {fw}/{cw}={100*fw/cw:.1f}%  |  门东(冲过门overshoot) {fe}/{ce}={100*fe/ce:.1f}%')
        if margins:
            margins.sort(); import statistics as st
            print(f'   朝向余量°: min={margins[0]:.2f} p5={margins[max(0,len(margins)//20)]:.2f} 中位={st.median(margins):.2f}')
