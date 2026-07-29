# -*- coding: utf-8 -*-
"""`_RolloutStats` 回归测试（L243-续8 重写后）—— **零依赖**：把类源码抠出来单独 exec，
不 import `run_step4e`（那会拖 torch/sb3）。

为什么必须有这个文件：这套采集器采的是 **A 类量（训练期量）**——跑完就没了、拿存档补不回来，
而这次正式实验**只有一次机会**。起跑前重写了它（两轴统计 + 离散臂动作还原 + 逐环境兜异常 +
episode 边界断开），必须逐条证明新逻辑对，不能"跑起来没报错"就算数。

跑法：  python3 -B 代码/tests/test_rollout_stats_v2.py
"""
import ast
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.dirname(_HERE)
_SRC = os.path.join(_CODE, "run_step4e.py")

# ── 把 _RolloutStats 类源码原样抠出来（不执行 run_step4e 的任何模块级代码）──
_tree = ast.parse(open(_SRC, encoding="utf-8").read())
_cls = next((n for n in _tree.body if isinstance(n, ast.ClassDef) and n.name == "_RolloutStats"), None)
assert _cls is not None, "找不到 _RolloutStats 类"
_src_lines = open(_SRC, encoding="utf-8").read().splitlines()
_cls_src = "\n".join(_src_lines[_cls.lineno - 1:_cls.end_lineno])

# 真常量（与 trb_env/usv_env.py 逐字同源；这里硬写是为了零依赖，下面 T0 会去核对）
A_ACC = (-0.048, -0.032, -0.016, 0.0, 0.016, 0.032, 0.048)
A_OMEGA = (-0.018, -0.012, -0.006, 0.0, 0.006, 0.012, 0.018)
DISCRETE_ACTIONS = tuple((a, w) for a in A_ACC for w in A_OMEGA)
IDX_EMERGENCY = 49
A_NORMAL_ACCEL_MAX = 0.048
A_NORMAL_OMEGA_MAX = 0.018

_ns = {"DISCRETE_ACTIONS": DISCRETE_ACTIONS, "IDX_EMERGENCY": IDX_EMERGENCY,
       "A_NORMAL_ACCEL_MAX": A_NORMAL_ACCEL_MAX}
exec(compile(_cls_src, "<_RolloutStats>", "exec"), _ns)
RS = _ns["_RolloutStats"]

W, A = A_NORMAL_OMEGA_MAX, A_NORMAL_ACCEL_MAX
_fail = []


def check(name, cond, extra=""):
    print(("  ✅ " if cond else "  ❌ ") + name + (("  ← " + str(extra)) if (extra and not cond) else ""))
    if not cond:
        _fail.append(name)


def approx(x, y, tol=1e-9):
    return x is not None and abs(x - y) <= tol


print("【T0】常量与真源一致（否则本文件测的是假东西）")
_uenv = open(os.path.join(_CODE, "trb_env", "usv_env.py"), encoding="utf-8").read()
check("A_ACC 与 usv_env.py 逐字一致", "A_ACC: tuple = (-0.048, -0.032, -0.016, 0.0, 0.016, 0.032, 0.048)" in _uenv)
check("A_OMEGA 与 usv_env.py 逐字一致", "A_OMEGA: tuple = (-0.018, -0.012, -0.006, 0.0, 0.006, 0.012, 0.018)" in _uenv)
check("DISCRETE_ACTIONS 构造式一致",
      "DISCRETE_ACTIONS: tuple = tuple((a, w) for a in A_ACC for w in A_OMEGA)" in _uenv)
check("离散网格 49 个点", len(DISCRETE_ACTIONS) == 49)
check("离散网格 |ω| 上界 == 连续箱半宽（⟹ 打满判据两臂同一条线）",
      approx(max(abs(w) for _, w in DISCRETE_ACTIONS), A_NORMAL_OMEGA_MAX))
check("离散网格 |a| 上界 == 连续箱半宽", approx(max(abs(a) for a, _ in DISCRETE_ACTIONS), A_NORMAL_ACCEL_MAX))

print("\n【T1】🔴 离散臂现在真的采得到动作统计（原实现恒 None = 论文头号卖点的对照采不到）")
r = RS()
# 6 步，2 个并行环境：全选 idx=48 = (a=+0.048, ω=+0.018) = 两轴都打满
idx_full = DISCRETE_ACTIONS.index((0.048, 0.018))
for _ in range(3):
    r.feed([{"flags": {}}, {"flags": {}}], [False, False], W, actions=[idx_full, idx_full], a_box=A)
s = r.snapshot()
check("roll_n_act = 6（离散臂不再是 0）", s["roll_n_act"] == 6, s["roll_n_act"])
check("转艏打满率 = 100%", approx(s["roll_yaw_sat_frac"], 1.0), s["roll_yaw_sat_frac"])
check("加速度打满率 = 100%", approx(s["roll_acc_sat_frac"], 1.0), s["roll_acc_sat_frac"])
check("反转率 = 0（一直同向）", approx(s["roll_yaw_reversal_rate"], 0.0), s["roll_yaw_reversal_rate"])
check("|Δω| 均值 = 0（一直同值）", approx(s["roll_yaw_incr_mean"], 0.0), s["roll_yaw_incr_mean"])
check("离散臂没有盾改写量（n_corr=0 ⟹ 该列 None，不是 0）",
      s["roll_n_corr"] == 0 and s["roll_shield_corr_mean"] is None)

print("\n【T2】离散臂 bang-bang（左右打满来回切）—— 反转率必须是 1.0")
r = RS()
i_l = DISCRETE_ACTIONS.index((0.0, -0.018))
i_r = DISCRETE_ACTIONS.index((0.0, 0.018))
for k in range(8):
    r.feed([{"flags": {}}], [False], W, actions=[i_r if k % 2 == 0 else i_l], a_box=A)
s = r.snapshot()
check("步数 8 / 相邻对 7", s["roll_n_act"] == 8 and s["roll_n_pair"] == 7, (s["roll_n_act"], s["roll_n_pair"]))
check("转艏反转率 = 1.0（7/7）", approx(s["roll_yaw_reversal_rate"], 1.0), s["roll_yaw_reversal_rate"])
check("|Δω| 均值 = 0.036", approx(s["roll_yaw_incr_mean"], 0.036, 1e-12), s["roll_yaw_incr_mean"])
check("加速度轴恒 0 ⟹ 加速度反转率 = 0", approx(s["roll_acc_reversal_rate"], 0.0), s["roll_acc_reversal_rate"])

print("\n【T3】🔴 紧急槽 idx49 单列计数、不混进 (a,ω) 统计（它的值由调度器算、不在格点上）")
r = RS()
r.feed([{"flags": {}}], [False], W, actions=[IDX_EMERGENCY], a_box=A)
r.feed([{"flags": {}}], [False], W, actions=[idx_full], a_box=A)
s = r.snapshot()
check("roll_n_em = 1", s["roll_n_em"] == 1, s["roll_n_em"])
check("roll_n_act = 1（紧急步没被算成策略动作）", s["roll_n_act"] == 1, s["roll_n_act"])

print("\n【T4】🔴 连续臂：两轴分开 + 盾改写量拆两轴（原来 m/s² 与 rad/s 直接相加 = 量纲混算）")
r = RS()
# 策略想要 (0.048, 0.018)，盾改成 (0.000, 0.000) ⟹ Δa=0.048（满箱）、Δω=0.018（满箱）
r.feed([{"flags": {}, "u_desired": (0.048, 0.018), "u_applied": (0.0, 0.0), "source": "projection"}],
       [False], W, a_box=A)
s = r.snapshot()
check("corr_a = 0.048（m/s²）", approx(s["roll_shield_corr_a_mean"], 0.048), s["roll_shield_corr_a_mean"])
check("corr_w = 0.018（rad/s）", approx(s["roll_shield_corr_w_mean"], 0.018), s["roll_shield_corr_w_mean"])
check("归一化合量 = 1.0（两轴各改满一箱）", approx(s["roll_shield_corr_norm_mean"], 1.0),
      s["roll_shield_corr_norm_mean"])
check("兼容列 roll_shield_corr_mean 走归一化口径", approx(s["roll_shield_corr_mean"], 1.0))
check("roll_n_src = 1（有 source 的步数带出来了）", s["roll_n_src"] == 1, s["roll_n_src"])
# 反证：只改加速度轴时，旧口径会把它当成"盾大改了"，新口径只报 0.5
r = RS()
r.feed([{"u_desired": (0.048, 0.0), "u_applied": (0.0, 0.0)}], [False], W, a_box=A)
s = r.snapshot()
check("只改加速度轴 ⟹ 归一化 = 0.5（转向轴没被动）", approx(s["roll_shield_corr_norm_mean"], 0.5),
      s["roll_shield_corr_norm_mean"])
check("两轴可分：corr_w = 0", approx(s["roll_shield_corr_w_mean"], 0.0))

print("\n【T5】🔴 episode 边界必须断开（否则上一局末步与下一局首步被当相邻对 ⟹ 系统性高估抖动）")
r = RS()
r.feed([{"flags": {"goal": True}, "u_desired": (0.0, 0.018)}], [False], W, a_box=A)   # 局内第 1 步
r.feed([{"flags": {"goal": True}, "u_desired": (0.0, 0.018)}], [True], W, a_box=A)    # 局末（done）
r.feed([{"flags": {}, "u_desired": (0.0, -0.018)}], [False], W, a_box=A)              # 新局第 1 步
s = r.snapshot()
check("相邻对 = 1（只有局内那一对，跨局那一对被断开）", s["roll_n_pair"] == 1, s["roll_n_pair"])
check("反转率 = 0（跨局的方向翻转没被算成抖动）", approx(s["roll_yaw_reversal_rate"], 0.0),
      s["roll_yaw_reversal_rate"])
check("roll_eps = 1", s["roll_eps"] == 1, s["roll_eps"])
# 反证：若不断开，这里会是 2 对、反转率 0.5
r = RS()
for d, w in ((False, 0.018), (False, 0.018), (False, -0.018)):
    r.feed([{"flags": {}, "u_desired": (0.0, w)}], [d], W, a_box=A)
s2 = r.snapshot()
check("反证有区分力：同样三步但不含 done ⟹ 2 对 / 反转率 0.5",
      s2["roll_n_pair"] == 2 and approx(s2["roll_yaw_reversal_rate"], 0.5),
      (s2["roll_n_pair"], s2["roll_yaw_reversal_rate"]))

print("\n【T6】🔴 dones 拿不到时**不计** episode（原来把每一步都当结束 ⟹ roll_eps 膨胀约 170 倍）")
r = RS()
for _ in range(10):
    r.feed([{"flags": {"time": True}}], None, W, a_box=A)
s = r.snapshot()
check("roll_eps = 0（宁可缺，不要错）", s["roll_eps"] == 0, s["roll_eps"])
check("roll_steps 照常 = 10", s["roll_steps"] == 10, s["roll_steps"])

print("\n【T7】🔴 逐环境兜异常 + 计数（原来整段 try/except：一个环境抛了，同一步后面的环境全丢且无痕迹）")


class _Boom(dict):
    def get(self, k, *a):
        if k == "reward_parts":
            raise RuntimeError("模拟第 2 个环境的 info 有毒")
        return dict.get(self, k, *a)


r = RS()
r.feed([{"flags": {}, "u_desired": (0.0, 0.018)},
        _Boom(flags={}),
        {"flags": {}, "u_desired": (0.0, 0.018)},
        {"flags": {}, "u_desired": (0.0, 0.018)}], [False] * 4, W, a_box=A)
s = r.snapshot()
check("排在毒环境后面的两个环境照样采到（n_act = 3）", s["roll_n_act"] == 3, s["roll_n_act"])
check("异常留了计数 roll_n_err = 1（不再静默）", s["roll_n_err"] == 1, s["roll_n_err"])

print("\n【T8】只读性：feed 绝不修改传进来的 info（SB3 之后还要用 terminal_observation）")
import copy
_infos = [{"flags": {"goal": True}, "reward_parts": {"r_goal": 1.0}, "source": "projection",
           "rho_acting": 3, "u_desired": (0.01, 0.002), "u_applied": (0.01, 0.001),
           "terminal_observation": [1, 2, 3]}]
_before = copy.deepcopy(_infos)
RS().feed(_infos, [True], W, actions=[5], a_box=A)
check("infos 逐字未变", _infos == _before)

print("\n【T9】快照后清零（跨窗口不串）+ 空窗口不炸")
r = RS()
r.feed([{"flags": {}, "u_desired": (0.048, 0.018)}], [False], W, a_box=A)
r.snapshot()
s = r.snapshot()
check("空窗口 roll_steps = 0", s["roll_steps"] == 0)
check("空窗口动作类列全 None（不是 0）",
      s["roll_yaw_sat_frac"] is None and s["roll_acc_sat_frac"] is None
      and s["roll_shield_corr_mean"] is None and s["roll_r_alias_mean"] is None)

print("\n【T10】分母一律带出（L216-D：没采到 vs 采到了是 0 必须分得开）")
_need = {"roll_steps", "roll_eps", "roll_n_act", "roll_n_pair", "roll_n_corr",
         "roll_n_err", "roll_n_em", "roll_n_src"}
check("8 个分母/计数全在快照里", _need <= set(RS().snapshot().keys()),
      _need - set(RS().snapshot().keys()))

print("\n【T11】内存恒定（累加器不随步数增长）")
r = RS()
for k in range(4000):
    r.feed([{"flags": {}, "u_desired": (0.0, 0.018)}, {"flags": {}, "u_desired": (0.0, -0.018)}],
           [False, False], W, a_box=A)
check("_prev_sign 只有 n_envs=2 个条目", len(r._prev_sign) == 2, len(r._prev_sign))
check("没有任何 list 型累加器", not any(isinstance(getattr(r, f), list) for f in RS.__slots__))

print("\n" + "=" * 70)
if _fail:
    print(f"❌ {len(_fail)} 条没过：")
    for f in _fail:
        print("   ·", f)
    sys.exit(1)
print("✅ 全部通过 —— 训练期采集器（A 类量·跑完补不回）的新逻辑逐条成立")
