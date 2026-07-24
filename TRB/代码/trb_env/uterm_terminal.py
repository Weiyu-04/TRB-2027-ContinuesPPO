#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""U_term 终端约束 · SOUND 版核心（Prop4 v2 backup-maneuver·独立复审 2026-07-25 任务A）。

【它是什么】盾的"落点 s' ∈ A(可清障集)"终端检查的【纯·可本机单测】soundness 核心：
  s∈A ⟺ ∃ certified 直行尾脱离序列 m·block1-SOUND cert_v2 判 m 对 s 永久清障(引理1 恒速直行尾凸性)。
  盾走一受盾步得 s'·若 s'∈A(∃ certified 脱离·可要求合规首步) ⟹ 递归可行 ⟹ A 前向不变 ⟹ 可证明前向不变无碰(A 上·单 CV)。

【为何独立成 pure 模块】soundness 关键逻辑必须【本机可单测】(usv_projection/usv_dynamics 依赖 vesselmodels·本机跑不了)。
  本模块只 numpy/shapely/math·不 import 任何 vesselmodels 件 → CI 可跑 + 对抗审可复现。
  动力学【解耦为 integrate_fn 回调】：盾传【官方 usv_dynamics 版】·单测传【自包含 RK4】·同一份 cert 逻辑两处复用。

【soundness 锚】cert_v2 = block1 Lipschitz 清障证书(已复审 SOUND·L198/L200·本机 0 假放行 fuzz 坐实)
  + 引理1 恒速直行尾(ω=0∧a=0)凸性 → 过 CPA ⟹ 永久清。F1 修(a=0 非仅 ω=0)已焊。

【契约·别踩】
  · integrate_fn(ego0, segments, T, h) -> (ts[N], traj[N,4], omega_seg[N-1])·10s 边界钳 v(执行口径)。
  · cert_v2 只对【恒速直行尾】判永久清(恒转/加速尾 straight_tail=False → 恒 False·sound 保守)。
  · 机动族时长网格【已加密】(L200-C·5..120s)·粗网格会把合规率压出假象。
"""
import math
import numpy as np
from shapely.geometry import Polygon

# 常量与官方一致（usv_dynamics/usv_colregs·Krasowski 偏航受限）
A_MAX, W_MAX, V_MAX = 0.24, 0.03, 9.5
L_SHIP, W_SHIP = 175.0, 25.4
R_CIRC = 0.5 * math.hypot(L_SHIP, W_SHIP)   # 88.4
DECISION_DT = 10.0


def _rect(cx, cy, th, l, w):
    hl, hw = 0.5 * l, 0.5 * w
    c, s = math.cos(th), math.sin(th)
    return Polygon([(cx + x * c - y * s, cy + x * s + y * c)
                    for x, y in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))])


def first_unsafe_t(ts, traj, obs0, obs_len, obs_wid, h, omega_seg):
    """block1 SOUND Lipschitz 清障证书（= block3.clearance_profile / reclassify.first_unsafe_t·L200 逐点相等验过）。
    L = max(|v_k|,|v_{k+1}|)+a_max·h（ego 速上界·兜 10s 钳前 overshoot）+ |ω|·R_circ（转艏在外接圆的贡献）+ |v_m|（他船）。
    子区间真距下界 (d_k+d_{k+1}−L·h)/2 ≤0 ⟹ 该子区间可能撞。返回首次不安全时刻或 None（全程清）。"""
    ts = np.asarray(ts, float); traj = np.asarray(traj, float)
    vm = float(obs0[3]); om = (math.cos(obs0[2]), math.sin(obs0[2]))
    N = len(ts); ds = np.empty(N)
    for k in range(N):
        t = ts[k]; ex, ey, eth, _ = traj[k]
        oc = (obs0[0] + vm * om[0] * t, obs0[1] + vm * om[1] * t)
        ds[k] = _rect(ex, ey, eth, L_SHIP, W_SHIP).distance(_rect(oc[0], oc[1], obs0[2], obs_len, obs_wid))
    hh = ts[1] - ts[0]
    vv = traj[:, 3]
    v_seg = np.maximum(np.abs(vv[:-1]), np.abs(vv[1:])) + A_MAX * hh
    ws = np.abs(np.asarray(omega_seg, float))
    w_term = ws * R_CIRC if len(ws) == N - 1 else W_MAX * R_CIRC   # 采样不匹配 → 全局上界（sound）
    L = v_seg + w_term + abs(vm)
    ilb = (ds[:-1] + ds[1:] - L * hh) / 2.0
    unsafe = np.where(ilb <= 0.0)[0]
    return float(ts[1:][unsafe[0]]) if len(unsafe) else None


def _tail_distances(ts, traj, obs0, obs_len, obs_wid):
    vm = float(obs0[3]); om = (math.cos(obs0[2]), math.sin(obs0[2]))
    ds = np.empty(len(ts))
    for k in range(len(ts)):
        t = ts[k]; ex, ey, eth, _ = traj[k]
        oc = (obs0[0] + vm * om[0] * t, obs0[1] + vm * om[1] * t)
        ds[k] = _rect(ex, ey, eth, L_SHIP, W_SHIP).distance(_rect(oc[0], oc[1], obs0[2], obs_len, obs_wid))
    return ds


def cert_v2(ts, traj, obs0, obs_len, obs_wid, segments, h, H):
    """修正证书（引理1 前提·Prop4 v2）。certified_perm = clears_H ∧ 恒速直行尾 ∧ 过CPA(永久清充分条件)。"""
    assert obs_len > 0.0 and obs_wid > 0.0, f"他船尺寸非法 {obs_len}x{obs_wid}"
    fut = first_unsafe_t(ts, traj, obs0, obs_len, obs_wid, h, _omega_from(segments, ts))
    clears_H = (fut is None) or (fut > H)
    # F1(CRITICAL)：引理1 凸性前提=恒速平移 → 尾段 ω=0 【且 a=0】（加速尾抛物线路径·g 非凸·假 sound）
    straight_tail = abs(segments[-1][1]) < 1e-9 and abs(segments[-1][0]) < 1e-9
    ds = _tail_distances(ts, traj, obs0, obs_len, obs_wid)
    t_tail = sum(d for a, w, d in segments[:-1] if d is not None)
    tail_idx = np.where(np.asarray(ts) >= t_tail - 1e-9)[0]
    if len(tail_idx) >= 2:
        tail_ds = ds[tail_idx]
        past_cpa = bool(tail_ds[-1] > tail_ds[-2] + 1e-9 and int(np.argmin(tail_ds)) < len(tail_ds) - 1)
    else:
        past_cpa = False
    return dict(certified_perm=bool(clears_H and straight_tail and past_cpa),
                clears_H=bool(clears_H), straight_tail=bool(straight_tail),
                past_cpa=bool(past_cpa), first_unsafe_t=fut)


def _omega_from(segments, ts):
    """逐子区间 |ω| 上界（与 traj 采样对齐·供 first_unsafe_t 的 w_term）。"""
    ts = np.asarray(ts, float)
    if len(ts) < 2:
        return np.array([])
    out = []
    for k in range(len(ts) - 1):
        t = ts[k]; acc = 0.0; w = segments[-1][1]
        for a, ww, dur in segments:
            if dur is None:
                w = ww; break
            if t < acc + dur - 1e-9:
                w = ww; break
            acc += dur
        out.append(abs(w))
    return np.asarray(out, float)


def straight_tail_family():
    """[(name, segments, first_omega)]·末段恒速直行(ω=0∧a=0)。时长网格【加密】(L200-C·5..120s)。
    first_omega<0=右转(starboard)·>0=左转(port)·=0=纯直行。"""
    A, W = A_MAX, W_MAX
    fam = []
    DURS = (5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 90.0, 100.0, 110.0, 120.0)
    for w in (-W, +W):
        for t1 in DURS:
            fam.append((f"turn{w:+.3f}_{int(t1)}s", [(0.0, w, t1), (0.0, 0.0, None)], w))
            fam.append((f"dec{w:+.3f}_{int(t1)}s", [(-A, w, t1), (0.0, 0.0, None)], w))
        for t1 in (20.0, 40.0, 60.0):
            fam.append((f"accdec{w:+.3f}_{int(t1)}s", [(+A, w, t1), (-A, 0.0, 20.0), (0.0, 0.0, None)], w))
    for a in (-A, 0.0, +A):
        fam.append((f"straight_a{a:+.2f}", [(a, 0.0, None)], 0.0))
    return fam


def _tail_after(segments, dt):
    """机动 segments 走 dt 秒后的尾巴序列（首段裁掉 dt）。"""
    out = []; used = 0.0; started = False
    for a, w, dur in segments:
        if dur is None:
            out.append((a, w, None)); started = True; continue
        if used + dur <= dt + 1e-9:
            used += dur; continue
        if not started and used < dt:
            out.append((a, w, dur - (dt - used))); started = True; used = dt
        else:
            out.append((a, w, dur))
    if not out:
        out = [(segments[-1][0], segments[-1][1], None)]
    return out


def state_in_A(ego, obs, obs_len, obs_wid, integrate_fn, H=120.0, h=0.5, require_omega_sign=0):
    """s∈A? 找首个 cert_v2 永久清障的直行尾脱离序列。require_omega_sign: -1右/+1左/0任意（合规约束）。
    返回 (name, segments) 或 (None, None)。integrate_fn(ego, segs, T, h)->(ts,traj,oseg)。"""
    for name, segs, w0 in straight_tail_family():
        if require_omega_sign < 0 and not (w0 < -1e-9):
            continue
        if require_omega_sign > 0 and not (w0 > 1e-9):
            continue
        ts, traj, _ = integrate_fn(ego, segs, H, h)
        if cert_v2(ts, traj, obs, obs_len, obs_wid, segs, h, H)["certified_perm"]:
            return name, segs
    return None, None


def successor_in_A(s_ego_next, s_obs_next, obs_len, obs_wid, integrate_fn,
                   H=120.0, h=0.5, require_omega_sign=0, backup_hint=None):
    """终端检查：后继 s'（已由盾/dynamics 算好）是否 ∈A（∃ certified 脱离·可要求合规首步）。
    backup_hint = 上一步的 certified 备份序列 m*（O(1) 优化·Prop4 保证其走一步后的尾巴对 s' 仍 certified）：
      先试 m* 的尾巴·命中免全族搜；miss 才全族搜（守 sound·hint 只加速不改判）。
    返回 (in_A: bool, backup_segs)。"""
    # O(1) 尝试：backup_hint 的尾巴（若其首步满足合规约束）
    if backup_hint is not None:
        w0 = backup_hint[0][1]
        ok_sign = (require_omega_sign == 0) or (require_omega_sign < 0 and w0 < -1e-9) or (require_omega_sign > 0 and w0 > 1e-9)
        if ok_sign:
            ts, traj, _ = integrate_fn(s_ego_next, backup_hint, H, h)
            if cert_v2(ts, traj, s_obs_next, obs_len, obs_wid, backup_hint, h, H)["certified_perm"]:
                return True, backup_hint
    name, segs = state_in_A(s_ego_next, s_obs_next, obs_len, obs_wid, integrate_fn,
                            H=H, h=h, require_omega_sign=require_omega_sign)
    return (segs is not None), segs


# ── 自包含 RK4（单测用·10s 边界钳·生产口径·不依赖 vesselmodels）───────────────────
def integrate_local_rk4(ego0, segments, T, h=0.5, dt=0.1):
    def seg_at(t):
        acc = 0.0
        for a, w, dur in segments:
            if dur is None:
                return a, w
            if t < acc + dur - 1e-9:
                return a, w
            acc += dur
        return segments[-1][0], segments[-1][1]
    def rhs(x, a, w):
        v, th = x[3], x[2]
        return np.array([v * math.cos(th), v * math.sin(th), w, a])
    nsub = int(round(h / dt))
    x = np.asarray(ego0, float).copy(); ts = [0.0]; out = [x.copy()]; oseg = []
    n = int(round(T / h))
    for i in range(n):
        a, w = seg_at(i * h)
        a = float(np.clip(a, -A_MAX, A_MAX)); w = float(np.clip(w, -W_MAX, W_MAX)); oseg.append(abs(w))
        for _ in range(nsub):
            k1 = rhs(x, a, w); k2 = rhs(x + 0.5 * dt * k1, a, w); k3 = rhs(x + 0.5 * dt * k2, a, w); k4 = rhs(x + dt * k3, a, w)
            x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        t = (i + 1) * h
        if abs(t / DECISION_DT - round(t / DECISION_DT)) < 1e-9:
            x[3] = float(np.clip(x[3], 0.0, V_MAX))
        ts.append(t); out.append(x.copy())
    return np.array(ts), np.array(out), np.array(oseg)
