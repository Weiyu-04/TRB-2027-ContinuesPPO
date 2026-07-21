"""
动力学模块冒烟测试 —— 全部断言用闭式解 / 物理直觉，fact-based。
跑：/opt/miniconda3/envs/trb/bin/python 代码/tests/test_dynamics.py

闭式解（常值 a, ω；状态 [px,py,θ,v]）：
  v(t)=v0+a·t,  θ(t)=θ0+ω·t
  匀速直线(ω=0,a=0): x=x0+v0·t
  纯加速(θ0=0,ω=0):  x=x0+v0·t+½a·t² , v=v0+a·t
  纯转向(a=0,θ0=0):  x=(v0/ω)·sin(ω·t), y=(v0/ω)(1-cos(ω·t)), θ=ω·t
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from trb_env.usv_dynamics import step, make_vessel_params, wrap_to_pi, PAPER_V_MAX

p = make_vessel_params()
DT = 10.0
_fail = 0


def check(name, got, exp, tol=1e-3):
    global _fail
    got = np.asarray(got, dtype=float)
    exp = np.asarray(exp, dtype=float)
    ok = np.all(np.abs(got - exp) <= tol)
    if not ok:
        _fail += 1
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: got={np.round(got,4)} exp={np.round(exp,4)}")


# 0) 参数核对 Table II（fact-based）
assert p.l == 175, f"l={p.l}≠175"
assert p.a_max == 0.24, f"a_max={p.a_max}≠0.24"
assert p.w_max == 0.03, f"w_max={p.w_max}≠0.03"
assert p.v_max == 9.5, f"v_max={p.v_max}≠9.5(论文降值)"
print("[PASS] 参数 = SR108 集装箱船 l=175 / a_max=0.24 / w_max=0.03 / v_max=9.5(论文§VII降值)")

# 1) 匀速直线：v0=5, 无控制, 10s → x+50, 其余不变
check("① 匀速直线", step([0, 0, 0, 5], [0, 0], DT, p), [50, 0, 0, 5])

# 2) 纯加速：a=0.02, v0=0, θ=0 → v=0.2, x=½·0.02·100=1.0
check("② 纯加速", step([0, 0, 0, 0], [0.02, 0], DT, p), [1.0, 0, 0, 0.2])

# 3) 纯转向：ω=0.01, v=5, θ0=0 → θ=0.1; x=(5/0.01)sin(0.1)=49.9167; y=(5/0.01)(1-cos0.1)=2.4979
check("③ 纯转向", step([0, 0, 0, 5], [0, 0.01], DT, p),
      [(5/0.01)*np.sin(0.1), (5/0.01)*(1-np.cos(0.1)), 0.1, 5], tol=1e-2)

# 4) 加速度限幅：a=1.0→clip 0.24, v=5+2.4=7.4(<v_max不截), x=5·10+½·0.24·100=62
check("④ 加速度限幅 a_max", step([0, 0, 0, 5], [1.0, 0], DT, p), [62, 0, 0, 7.4], tol=1e-2)

# 5) 转艏率限幅：ω=1.0→clip 0.03 → θ=0.3
g = step([0, 0, 0, 5], [0, 1.0], DT, p)
check("⑤ 转艏率限幅 w_max (θ)", [g[2]], [0.3])

# 6) v 默认不 clip（忠实 yp RHS）：v0=9, a=0.24 → v=9+2.4=11.4（不截）
g = step([0, 0, 0, 9], [0.24, 0], DT, p)
check("⑥ v 默认不限速(忠实yp)", [g[3]], [11.4])

# 7) v clip 开启时才截：同上 + clip_velocity=True → v=9.5
g = step([0, 0, 0, 9], [0.24, 0], DT, p, clip_velocity=True)
check("⑦ v clip 开启→9.5", [g[3]], [9.5])

# 8) θ wrap：θ0=3.0, ω=0.03 → 3.3 > π → wrap 到 3.3-2π
g = step([0, 0, 3.0, 5], [0, 0.03], DT, p)
check("⑧ θ wrap 到[-π,π]", [g[2]], [wrap_to_pi(3.3)])

# 9) 负向转向（左转 ω>0 右手系? 按 RHS θ̇=ω）+ 负加速度：a=-0.01,ω=-0.02
#    θ=-0.2, v=5-0.1=4.9；只校验 θ、v（位置交给闭式解一致性）
g = step([0, 0, 0, 5], [-0.01, -0.02], DT, p)
check("⑨ 负 a/ω：θ,v", [g[2], g[3]], [-0.2, 4.9], tol=1e-2)

print("\n" + ("✅ 全部 PASS" if _fail == 0 else f"❌ {_fail} 项 FAIL"))
sys.exit(1 if _fail else 0)
