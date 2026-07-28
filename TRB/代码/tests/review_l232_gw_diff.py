# -*- coding: utf-8 -*-
"""L232-G 第 4 条独立复审：两个开关的【默认档逐位等价】—— 状态机差分实跑。

L231-D 的证法是"改动前后各跑一趟 8000 步冒烟、比网络权重"。本脚本换一条更直接、
且不需要 torch/SB3/commonocean 的证法：

  把**改动前**（commit fcb72a6 的父提交）的 `ColregsStatechart` 和**改动后**默认档
  `ColregsStatechart(gw_entry='paper')` 同时导进来，喂同一批会遇几何，逐步比 ρ。
  —— `gw_entry` 是这两把钥匙里**唯一会改变环境/盾行为**的那把；另一把 `act_dist=gauss`
     在源码层就是恒等（`policy_for('gauss')` 返回的还是字符串 "MlpPolicy"、policy_kwargs 逐键相同），
     不需要跑。

⚠️ 几何要挑过：`crossing/head_on` 要求 CV 外推**真的会撞上**（collision_possible），
   而 is_emergency(T_PRED=180s) 又会抢占 ⟹ 让路态 ρ2/3/4 的窗口本来就窄，
   随机撒状态基本只能撒出 ρ0/ρ5，测不到那条改动分支。本脚本按 CPA 反解几何、
   并把大部分样本压在 t_horizon(420s) 边界附近，实测能覆盖到 ρ1/ρ2/ρ3/ρ4。

跑法（需 numpy / shapely / commonocean-vessel-models）：
    python3 -B 代码/tests/review_l232_gw_diff.py
环境变量：OLD_ROOT（改动前 trb_env 的父目录，默认从 git 现取）· NCASE · NSTEP
"""
import os
import random
import subprocess
import sys
import tempfile

import numpy as np

SWITCH_COMMIT = "fcb72a6"     # 落两把钥匙的那个提交；它的父提交 = 改动前
NCASE = int(os.environ.get("NCASE", "500"))
NSTEP = int(os.environ.get("NSTEP", "45"))


def materialize_old(repo_root):
    """把改动前那版 trb_env 取到临时目录（不动工作区）。"""
    d = tempfile.mkdtemp(prefix="trb_old_")
    tar = subprocess.run(["git", "archive", SWITCH_COMMIT + "^", "TRB/代码/trb_env"],
                         cwd=repo_root, stdout=subprocess.PIPE, check=True).stdout
    subprocess.run(["tar", "-x", "-C", d, "--strip-components=2"], input=tar, check=True)
    return d


def load_colregs(root):
    """把指定目录下的 trb_env 作为 `trb_env` 顶层包导入（先清 sys.modules 免得两版互相盖）。"""
    sys.path.insert(0, root)
    for m in [k for k in list(sys.modules) if k.startswith("trb_env")]:
        del sys.modules[m]
    import trb_env.usv_colregs as C
    sys.path.pop(0)
    return C


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, "..", "..", ".."))
    new_root = os.path.abspath(os.path.join(here, ".."))
    old_root = os.environ.get("OLD_ROOT") or materialize_old(repo)

    OLD = load_colregs(old_root)
    NEW = load_colregs(new_root)
    OldChart, VSo = OLD.ColregsStatechart, OLD.VesselState
    NewChart, VSn = NEW.ColregsStatechart, NEW.VesselState
    print(f"改动前 : {OLD.__file__}   有 gw_entry 参数? "
          f"{'gw_entry' in OldChart.__init__.__code__.co_varnames}")
    print(f"改动后 : {NEW.__file__}   有 gw_entry 参数? "
          f"{'gw_entry' in NewChart.__init__.__code__.co_varnames}\n")

    mk = lambda VS, x, y, th, v: VS(position=np.array([x, y], float), orientation=th,
                                    velocity=v, length=50.0)

    def adv(s, dt=10.0):
        p = s.position + np.array([np.cos(s.orientation), np.sin(s.orientation)]) * s.velocity * dt
        return VSo(position=p, orientation=s.orientation, velocity=s.velocity, length=s.length)

    def gen(rng):
        """按最近会遇点(CPA)反解初始几何：60% 压在让路窗口，40% 泛化（含远距擦过/紧急）。"""
        v_e, v_o = rng.uniform(4.0, 9.5), rng.uniform(3.0, 9.0)
        if rng.random() < 0.6:
            t_c, m = rng.uniform(390.0, 560.0), rng.uniform(5.0, 160.0)
        else:
            t_c, m = rng.uniform(200.0, 900.0), rng.uniform(0.0, 1500.0)
        kind = rng.choice(["head_on", "cross_r", "cross_l", "overtake"])
        if kind == "head_on":
            th_o = np.pi + rng.uniform(-0.25, 0.25)
        elif kind == "cross_r":
            th_o = np.pi / 2 + rng.uniform(-0.6, 0.6)
        elif kind == "cross_l":
            th_o = -np.pi / 2 + rng.uniform(-0.6, 0.6)
        else:
            th_o, v_o = rng.uniform(-0.25, 0.25), max(1.0, v_e - rng.uniform(1.5, 3.5))
        p_cpa = np.array([v_e * t_c, rng.choice([1.0, -1.0]) * m])
        p0 = p_cpa - np.array([np.cos(th_o), np.sin(th_o)]) * v_o * t_c
        return mk(VSo, 0.0, 0.0, 0.0, v_e), mk(VSo, float(p0[0]), float(p0[1]), float(th_o), float(v_o))

    def run(c_old, c_new, ego0, obs0):
        c_old.reset(); c_new.reset()
        so, sm, out = ego0, obs0, []
        for t in range(NSTEP):
            if t:
                so, sm = adv(so), adv(sm)
            cp = lambda s: VSn(position=s.position.copy(), orientation=s.orientation,
                               velocity=s.velocity, length=s.length)
            out.append((int(c_old.step(so, sm)), int(c_new.step(cp(so), cp(sm)))))
        return out

    rng = random.Random(20260728)          # 固定种子 ⟹ 同一批几何喂两版，差异只可能来自代码
    cases = [gen(rng) for _ in range(NCASE)]

    hist, mism, tot = {}, 0, 0
    for ego, obs in cases:
        for a, b in run(OldChart(), NewChart(), ego, obs):   # 新版不传参 ⟹ 默认 'paper'
            tot += 1
            hist[a] = hist.get(a, 0) + 1
            mism += (a != b)
    gw = sum(hist.get(k, 0) for k in (2, 3, 4))
    print(f"【会遇几何差分 · {NCASE} 段 × {NSTEP} 步 = {tot} 步】")
    print(f"  改动前走过的 ρ 分布 = {dict(sorted(hist.items()))}"
          f"   (0=无冲突 1=直航 2=对遇 3=交叉 4=追越 5=紧急)")
    print(f"  真进过让路态 ρ2/3/4 = {gw} 步（{100 * gw / tot:.1f}%）· 直航 ρ1 = {hist.get(1, 0)} 步"
          f"   ← 覆盖到改动分支才算数")
    print(f"  🔴 默认档逐步 ρ 不一致 = {mism} / {tot}")

    mism_sym = sum(1 for ego, obs in cases
                   for a, b in run(OldChart(), NewChart(gw_entry="symmetric"), ego, obs) if a != b)
    print(f"  对照·开 symmetric 后不一致 = {mism_sym} / {tot}（{100 * mism_sym / tot:.1f}%）"
          f"   ← 必须 >0，否则『开关有效』本身就是假的")

    ok = (mism == 0 and gw > 0 and mism_sym > 0)
    print("\n判定：" + ("✅ 默认档逐步完全一致 · 让路态被真实覆盖 · 开关打开确实改行为"
                       if ok else "❌ 不通过"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
