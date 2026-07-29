# -*- coding: utf-8 -*-
"""`_RolloutStats` 回归（`03` L239·正式实验起跑前补的训练期采集·零依赖）。

被测：训练 rollout 期从 `info` 里采集的那一层。**这是 A 类量 —— 跑完补不回来**，
所以起飞前必须确认它真的采到、且真的不扰训练。

四条必须守住：
  ① **只读**：不修改传进来的 infos（SB3 之后还要用 terminal_observation）
  ② **恒定内存**：固定大小累加器，不每步 append（10M 步 × 8 环境 = 8000 万次调用）
  ③ **除数带出去**：`03` L216-D —— "没采到" 和 "采到了但为 0" 必须分得开
  ④ **出错不炸**：采集异常绝不打断训练

跑法：  python3 -B 代码/tests/test_rollout_stats.py
"""
import copy
import os
import sys

_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_PASS = N_FAIL = 0


def check(desc, cond):
    global N_PASS, N_FAIL
    if cond:
        N_PASS += 1
        print(f"  ✅ {desc}")
    else:
        N_FAIL += 1
        print(f"  ❌ {desc}")


def _load():
    """只把这两个符号抠出来跑 —— import run_step4e 会拉起 torch 全家桶还有一堆环境闸。"""
    src = open(os.path.join(_CODE, "run_step4e.py"), encoding="utf-8").read()
    a = src.index("class _RolloutStats:")
    b = src.index("def _accumulate_ep_returns(cb):")
    ns = {}
    exec(compile(src[a:b], "<rollout_stats>", "exec"), ns)
    return ns["_RolloutStats"], ns["_feed_rollout_stats"]


W = 0.018   # A_NORMAL_OMEGA_MAX


def mk(w, ua=None, src="projection", rho=0, flags=None, rp=None):
    d = {"source": src, "rho_acting": rho, "u_desired": [0.0, w],
         "u_applied": ua if ua is not None else [0.0, w],
         "reward_parts": rp or {"goal": 1.0, "sparse": 0.0}}
    if flags:
        d["flags"] = flags
    return d


def main():
    RS, feed = _load()
    print("===== _RolloutStats 回归 =====")

    print("\n【① 只读：绝不修改传进来的 infos】")
    st = RS()
    infos = [mk(0.018), mk(-0.018, ua=[0.1, 0.0])]
    before = copy.deepcopy(infos)
    st.feed(infos, [False, False], W)
    check("feed 之后 infos 逐字节未变", infos == before)

    print("\n【② 打满舵率 / 反转率 / 盾改写量算得对】")
    st = RS()
    # 单环境：+满舵 → −满舵 → +满舵 = 2 次反转，3 步全打满
    for w in (0.018, -0.018, 0.018):
        st.feed([mk(w)], [False], W)
    s = st.snapshot()
    check(f"打满舵率 = 3/3 = 1.0（得 {s['roll_yaw_sat_frac']}）", s["roll_yaw_sat_frac"] == 1.0)
    check(f"反转率 = 2/2 相邻对 = 1.0（得 {s['roll_yaw_reversal_rate']:.4f}）",
          abs(s["roll_yaw_reversal_rate"] - 1.0) < 1e-9)
    check(f"|Δ转艏| 均值 = 0.072/2 相邻对（得 {s['roll_yaw_incr_mean']:.5f}）",
          abs(s["roll_yaw_incr_mean"] - 0.036) < 1e-9)
    st = RS()
    st.feed([mk(0.001, ua=[0.05, 0.011])], [False], W)     # 盾把动作改写了 0.05 + 0.010
    s = st.snapshot()
    check(f"盾改写量 = 0.05+0.010 = 0.060（得 {s['roll_shield_corr_mean']:.4f}）",
          abs(s["roll_shield_corr_mean"] - 0.06) < 1e-9)
    check("小舵角不计入打满舵", s["roll_yaw_sat_frac"] == 0.0)

    print("\n【③ 终止旗只在 done 那一步计，且各档分开】")
    st = RS()
    st.feed([mk(0.0, flags={"goal": True, "collision": False})], [False], W)   # 未 done ⟹ 不计
    st.feed([mk(0.0, flags={"goal": True, "collision": False})], [True], W)    # done ⟹ 计
    st.feed([mk(0.0, flags={"goal": False, "collision": True})], [True], W)
    s = st.snapshot()
    check(f"episode 数 = 2（得 {s['roll_eps']}）", s["roll_eps"] == 2)
    check(f"到达 1 次 / 碰撞 1 次（得 {s['roll_ep_flags']}）",
          s["roll_ep_flags"].get("goal") == 1 and s["roll_ep_flags"].get("collision") == 1)

    print("\n【④ 盾归口 / 态势 / 奖励分量】")
    st = RS()
    for sc, rh in (("projection", 0), ("projection", 3), ("emergency", 5)):
        st.feed([mk(0.0, src=sc, rho=rh, rp={"goal": 2.0})], [False], W)
    s = st.snapshot()
    check(f"盾归口 projection 2 / emergency 1（得 {s['roll_source']}）",
          s["roll_source"] == {"projection": 2, "emergency": 1})
    check(f"态势直方图 {{0:1,3:1,5:1}}（得 {s['roll_rho']}）", s["roll_rho"] == {"0": 1, "3": 1, "5": 1})
    check(f"奖励分量取均值 goal=2.0（得 {s['roll_reward_parts']['goal']}）",
          abs(s["roll_reward_parts"]["goal"] - 2.0) < 1e-9)

    print("\n【⑤ 🔴 除数必须带出去（L216-D：没采到 vs 采到了但为 0）】")
    st = RS()
    s = st.snapshot()
    check("一步没采到 ⟹ 打满舵率是 None（不是 0.0）", s["roll_yaw_sat_frac"] is None)
    check("一步没采到 ⟹ roll_steps = 0（分母自带）", s["roll_steps"] == 0)
    st = RS()
    st.feed([mk(0.0)], [False], W)                     # 采到了，但舵角为 0
    s = st.snapshot()
    check("采到了但确实为 0 ⟹ 打满舵率是 0.0（不是 None）", s["roll_yaw_sat_frac"] == 0.0)
    check("  且分母带出来了 roll_n_act = 1", s["roll_n_act"] == 1)

    print("\n【⑥ 离散臂：没有 u_desired 也不能崩，其余字段照收】")
    st = RS()
    st.feed([{"flags": {"goal": True}, "reward_parts": {"goal": 1.0}, "rho": 2}], [True], W)
    s = st.snapshot()
    check("动作类字段为 None（离散臂结构上没有·不是 0.0）", s["roll_yaw_sat_frac"] is None)
    check("  且分母 roll_n_act = 0（明示「没测」）", s["roll_n_act"] == 0)
    check("终止旗照收", s["roll_ep_flags"].get("goal") == 1)
    check("态势照收（rho 兜底 rho_acting 缺失）", s["roll_rho"] == {"2": 1})

    print("\n【⑦ 恒定内存：8000 万次调用不能线性增长】")
    st = RS()
    for i in range(20000):
        st.feed([mk(0.018 if i % 2 else -0.018)], [i % 137 == 0], W)
    check("累加器没有 list 型成员（全是计数器/字典）",
          not any(isinstance(getattr(st, k), list) for k in st.__slots__))
    check(f"_prev_sign 只按并行环境数增长（1 个环境 ⟹ 1 条·得 {len(st._prev_sign)}）", len(st._prev_sign) == 1)
    s = st.snapshot()
    check(f"2 万步照样算对（步数 {s['roll_steps']}）", s["roll_steps"] == 20000)
    check("snapshot 之后清零", st.n == 0 and not st.src)

    print("\n【⑧ 出错不炸训练】")
    class _CB:
        locals = {"infos": [{"u_desired": "这不是数组"}], "dones": [False]}
    cb = _CB(); cb._roll = RS()
    try:
        feed(cb, W)
        check("喂进畸形数据不抛异常（整段 try/except 兜住）", True)
    except Exception as e:
        check(f"喂进畸形数据不抛异常 —— 但抛了 {type(e).__name__}", False)
    class _CB2:
        locals = {}
    cb2 = _CB2(); cb2._roll = RS()
    try:
        feed(cb2, W); check("locals 里没有 infos 也不抛", True)
    except Exception:
        check("locals 里没有 infos 也不抛", False)

    print(f"\n===== {N_PASS} PASS · {N_FAIL} FAIL =====")
    return 1 if N_FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
