"""test_plot_results —— Node L L2 绘图（trb_env/plot_results.py）守护。
离线可跑（合成记录·matplotlib Agg 后端·无需训练/场景夹具）。
核：① 按方聚合 mean±std 数值正确 ② 终值统计正确 ③ load/group 容错 ④ make_all_figures 产非空 PNG+PDF 不崩 ⑤ 边界（缺方/单种子/空）。
绘图是【纯下游消费者】（只读 jsonl 产图·不碰钱图计算）→ 守护重点 = 数据→图的【聚合数值正确】+ 不崩，而非像素级。"""
import os
import sys
import json
import math
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from trb_env import plot_results as P

_SQRT200 = math.sqrt(200.0)   # ddof=1 样本 std of [10,30] / [20,40]（非 ddof0 的 10）

_fail = 0


def ok(name, cond):
    global _fail
    print(("[PASS] " if cond else "[FAIL] ") + name)
    if not cond:
        _fail += 1


def _row(step, arr, coll, viol, emer, ep):
    return {"step": step, "到达率%": arr, "碰撞率%": coll, "违规次数/局": viol,
            "紧急步%": emer, "Ep长s": ep}


def _rec(party, kind, w, seed, trend):
    return {"party": party, "kind": kind, "colregs_weight": w, "seed": seed,
            "final": dict(trend[-1]), "trend": trend}


# 合成：Base 2 种子 + Discrete-safe 2 种子，trend 已知值（便于核 mean±std）
_RECS = [
    _rec("Base", "unshielded", 0.0, 0,
         [_row(100, 10.0, 5.0, 2.0, 0.0, 100.0), _row(200, 20.0, 4.0, 1.0, 0.0, 90.0)]),
    _rec("Base", "unshielded", 0.0, 1,
         [_row(100, 30.0, 7.0, 4.0, 0.0, 110.0), _row(200, 40.0, 6.0, 3.0, 0.0, 95.0)]),
    _rec("Discrete-safe", "shielded", 1.0, 0,
         [_row(100, 50.0, 0.0, 1.0, 10.0, 120.0), _row(200, 60.0, 0.0, 0.5, 12.0, 100.0)]),
    _rec("Discrete-safe", "shielded", 1.0, 1,
         [_row(100, 70.0, 0.0, 0.5, 8.0, 115.0), _row(200, 80.0, 0.0, 0.3, 9.0, 98.0)]),
]

# ① group_by_party
_g = P.group_by_party(_RECS)
ok("① group_by_party：Base/Discrete-safe 各 2 种子",
   set(_g) == {"Base", "Discrete-safe"} and len(_g["Base"]) == 2 and len(_g["Discrete-safe"]) == 2)

# ② agg_trend 数值正确（按 step 对齐多种子 mean±std）
_steps, _mean, _std = P.agg_trend(_g["Base"], "到达率%")
# step100: seeds [10,30] → mean20 std√200(ddof=1) ; step200: [20,40] → mean30 std√200
ok("② agg_trend Base 到达率% [ddof=1 样本std]：steps[100,200]·mean[20,30]·std[√200,√200]（与 Table III 同口径·非 ddof0 的 10）",
   _steps.tolist() == [100.0, 200.0]
   and np.allclose(_mean, [20.0, 30.0]) and np.allclose(_std, [_SQRT200, _SQRT200]))

# ③ final_stats 终值跨种子 mean±std（ddof=1）
_fm, _fs = P.final_stats(_g["Base"], "到达率%")   # final 到达率 = trend 末点 = [20,40] → mean30 std√200
ok("③ final_stats Base 终值到达率% [ddof=1]：mean30·std√200（柱状 error bar 对得上 Table III ±值）",
   abs(_fm - 30.0) < 1e-9 and abs(_fs - _SQRT200) < 1e-9)
_fm2, _fs2 = P.final_stats(_g["Discrete-safe"], "碰撞率%")   # 有盾终值碰撞 [0,0] → 0±0
ok("③b final_stats Discrete-safe 终值碰撞率%：0±0（有盾零碰撞）",
   abs(_fm2) < 1e-9 and abs(_fs2) < 1e-9)

# ④ load_records 容错（坏行跳过）
_tmpf = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8")
for r in _RECS:
    _tmpf.write(json.dumps(r, ensure_ascii=False) + "\n")
_tmpf.write("{坏行 not json}\n")          # 坏行
_tmpf.write("\n")                          # 空行
_tmpf.close()
_loaded = P.load_records(_tmpf.name)
ok("④ load_records 容错：读 4 条有效记录、跳过坏行/空行（与 run_step4e.read_records 同口径）",
   len(_loaded) == 4 and {r["party"] for r in _loaded} == {"Base", "Discrete-safe"})

# ⑤ make_all_figures 产非空 PNG+PDF 不崩
_outdir = tempfile.mkdtemp(prefix="trb_l2fig_")
_gret = P.make_all_figures(_tmpf.name, _outdir)
_files = {f: os.path.getsize(os.path.join(_outdir, f)) for f in os.listdir(_outdir)}
ok("⑤ make_all_figures 产 four_panel + money_figure 的 PNG+PDF 各非空（>1KB·不崩）",
   all(n in _files and _files[n] > 1000 for n in
       ("four_panel.png", "four_panel.pdf", "money_figure.png", "money_figure.pdf")))

# ⑥ 边界：单种子（std=0 不崩）/ 缺方（grouped 只含子集不崩）/ 空记录
_gsingle = P.group_by_party([_RECS[0]])    # 只 Base 1 种子
_s1, _m1, _sd1 = P.agg_trend(_gsingle["Base"], "到达率%")
ok("⑥ 边界单种子：agg_trend std 全 0（单样本）、mean==该种子值、不崩",
   np.allclose(_sd1, 0.0) and np.allclose(_m1, [10.0, 20.0]))
_outdir2 = tempfile.mkdtemp(prefix="trb_l2fig2_")
P.plot_four_panel(_gsingle, os.path.join(_outdir2, "single"))   # 仅 1 方 1 种子绘图不崩
ok("⑥b 边界缺方/单种子绘图：plot_four_panel 仅含 Base 一方仍出图不崩",
   os.path.getsize(os.path.join(_outdir2, "single.png")) > 1000)
_empty = P.group_by_party([])
ok("⑥c 边界空记录：group_by_party([])=={}、agg_trend 空方返回空 steps（不崩）",
   _empty == {} and len(P.agg_trend([], "到达率%")[0]) == 0)

# ⑦ 跨模块统计口径一致（防漂移·MAJOR 修复守护）：在【干净有限数值输入】下 _mean_std 与 run_step4e.agg_mean_std【逐位相同】
#    = 钱图误差棒 / 终值柱状 与 Table III 的 ±值【同源 ddof=1·N1→0·空→nan】，审稿人对得上。
#    ⚠️ 仅干净数值组逐位等价：plot 侧另滤 None/NaN/bool（防御性、与 agg_mean_std 有意分歧），故此处喂的 5 组全干净浮点；
#       钱图实际数据流恒为有限 float（evaluate agg 算术均值），分歧路径不可达。
import run_step4e as _S
_consistent = True
for _vals in ([], [5.0], [1.0, 2.0, 3.0], [10.0, 30.0], [0.0, 0.0, 0.0, 7.0]):
    _a = P._mean_std(_vals); _b = _S.agg_mean_std(_vals)
    if not _vals:
        _consistent = _consistent and all(x != x for x in _a) and all(x != x for x in _b)  # 两边都 (nan,nan)
    else:
        _consistent = _consistent and abs(_a[0] - _b[0]) < 1e-12 and abs(_a[1] - _b[1]) < 1e-12
ok("⑦ 跨模块口径：_mean_std ≡ run_step4e.agg_mean_std〔干净有限数值输入下·ddof=1·N1→0·空→nan·5 组逐位〕= 钱图误差棒 = Table III ±值同源",
   _consistent)

# ⑦c bit-identical 锁不变量（`03` L59·堵 var**0.5 vs math.sqrt 的 1 ULP 分歧·防回退）：
#    旧 ⑦ 的 5 组方差全是完全平方/0、var**0.5 与 math.sqrt 恰好逐位相等 → 系统性测不出该分歧（等价变异不翻 FAIL）。
#    用一个【确定性会撞 1 ULP 分歧】的输入（var 非完全平方）+ struct.pack 字节级断言，锁死 _mean_std 必用 math.sqrt。
import struct as _struct
_ulp_vals = [976.5504486605118, 268.7761451643785]   # var**0.5 末位 ...51 / math.sqrt ...52（diff~5.7e-14）
_pa = P._mean_std(_ulp_vals); _pb = _S.agg_mean_std(_ulp_vals)
ok("⑦c bit-identical（ULP 反例·防 var**0.5 回退）：_mean_std std 与 agg_mean_std 在【会撞 1 ULP 分歧】输入上 struct.pack 字节级逐位相同",
   _struct.pack(">d", _pa[1]) == _struct.pack(">d", _pb[1])
   and abs(_pa[0] - _pb[0]) < 1e-12)

# ⑦b None/NaN 健壮（MINOR 修复·防 float(None) 抛错）：trend 含 None 指标值 → 跳过该点不崩
_recN = _rec("Base", "unshielded", 0.0, 9,
             [{"step": 100, "到达率%": None, "碰撞率%": 5.0, "违规次数/局": 1.0, "紧急步%": 0.0, "Ep长s": 90.0}])
_mn, _sn = P._mean_std([None, 10.0, float("nan"), 30.0])   # 滤 None/NaN → [10,30] → mean20·std√200
ok("⑦b None/NaN 健壮：_mean_std 滤 None/NaN（[None,10,nan,30]→mean20·std√200·不抛 TypeError）",
   abs(_mn - 20.0) < 1e-9 and abs(_sn - _SQRT200) < 1e-9)

# ⑧ agg_trend source="curves" 读 ep_rew_mean·滤 None（早期无完成 episode=None；连续臂恒有/离散 LOG_CURVES=1）
_recsC = [
    {"party": "Continuous-safe", "kind": "continuous", "seed": 0,
     "final": _row(200, 90.0, 0.0, 0.5, 5.0, 100.0),
     "trend": [_row(100, 80.0, 0.0, 1.0, 6.0, 110.0), _row(200, 90.0, 0.0, 0.5, 5.0, 100.0)],
     "curves": [{"step": 100, "ep_rew_mean": None}, {"step": 200, "ep_rew_mean": -1500.0}]},
    {"party": "Continuous-safe", "kind": "continuous", "seed": 1,
     "final": _row(200, 88.0, 0.0, 0.6, 5.0, 102.0),
     "trend": [_row(100, 78.0, 0.0, 1.1, 6.0, 112.0), _row(200, 88.0, 0.0, 0.6, 5.0, 102.0)],
     "curves": [{"step": 100, "ep_rew_mean": None}, {"step": 200, "ep_rew_mean": -1400.0}]},
]
_gc = P.group_by_party(_recsC)
_cs, _cm, _csd = P.agg_trend(_gc["Continuous-safe"], "ep_rew_mean", source="curves")
# step100 两种子均 None → 被滤、该 step 不出现；step200: [-1500,-1400] → mean -1450·std √5000(ddof=1)
ok("⑧ agg_trend source=curves 读 ep_rew_mean·滤 None：steps[200]（step100 全 None 被滤）·mean[-1450]·std[√5000]",
   _cs.tolist() == [200.0] and abs(_cm[0] - (-1450.0)) < 1e-9 and abs(_csd[0] - math.sqrt(5000.0)) < 1e-9)

# ⑨ _band_bounds 下界裁剪（#3·L58）：率/计数 lower=0.0 裁负值区；回报 lower=None 不裁
_lo0, _hi0 = P._band_bounds(np.array([2.0]), np.array([5.0]), lower=0.0)        # 2-5=-3 → 裁到 0
_loN, _hiN = P._band_bounds(np.array([-1450.0]), np.array([70.0]), lower=None)  # 回报负·不裁
_loU, _hiU = P._band_bounds(np.array([5.0]), np.array([1.0]), lower=10.0)       # lower>上界：探针锁"上界 mean+std=6 永不被裁"（防误裁上界·堵等价变异 F-1）
ok("⑨ _band_bounds：lower=0.0→下界 max(mean-std,0)=0(不探负·上界不动)；lower=None→保留负下界；lower>上界探针→上界 mean+std=6 不被裁",
   abs(_lo0[0] - 0.0) < 1e-12 and abs(_hi0[0] - 7.0) < 1e-12
   and abs(_loN[0] - (-1520.0)) < 1e-12 and abs(_hiN[0] - (-1380.0)) < 1e-12
   and abs(_hiU[0] - 6.0) < 1e-12)

# ⑩ warn_misaligned_grids（#2/#4·L58）：种子 step 网格一致→[]；不一致(中途改 NSEG 模拟)→返回该方名（钱图数值表不受影响）
_align_ok = P.warn_misaligned_grids(_g, source="trend")    # _g(Base/DS) 各种子网格一致 → []
_mis = P.group_by_party([
    _rec("Base", "unshielded", 0.0, 0, [_row(100, 10.0, 5.0, 2.0, 0.0, 100.0), _row(200, 20.0, 4.0, 1.0, 0.0, 90.0)]),
    _rec("Base", "unshielded", 0.0, 1, [_row(100, 30.0, 7.0, 4.0, 0.0, 110.0)]),   # 仅 step100 → 网格不齐
])
_align_bad = P.warn_misaligned_grids(_mis, source="trend")
ok("⑩ warn_misaligned_grids：网格一致→[]；不一致→返回该方名['Base']（暴露中途改 NSEG·#2/#4·钱图数值表不受影响）",
   _align_ok == [] and _align_bad == ["Base"])

# ⑪ plot_training_return：连续臂有 curves→出非空图、返回画出方数=1；无 curves 方自动跳过
_outdir3 = tempfile.mkdtemp(prefix="trb_l2ret_")
_nd = P.plot_training_return(_gc, os.path.join(_outdir3, "tr"))
ok("⑪ plot_training_return：连续臂 ep_rew_mean 曲线出非空 PNG+PDF·返回画出方数=1（离散无 curves 跳过·回报负不裁 band）",
   _nd == 1 and os.path.getsize(os.path.join(_outdir3, "tr.png")) > 1000
   and os.path.exists(os.path.join(_outdir3, "tr.pdf")))

# ⑫ plot_example_trajectories（CAT5·D42-L2-续）：final_per 带 traj 的局 → ego 航迹按 ρ 着色+他船+goal★ 出图；无 traj 局跳过
_traj = [{"ego_x": float(i), "ego_y": float(2 * i), "ego_psi": 0.0,
          "obs_x": (None if i < 2 else float(20 - i)), "obs_y": (None if i < 2 else float(i)),
          "obs_psi": 0.0, "step": i, "rho": (0 if i < 3 else (3 if i < 6 else 5))} for i in range(8)]
_recT = [{"party": "Continuous-safe", "kind": "continuous", "seed": 0,
          "final": _row(200, 90.0, 0.0, 0.5, 5.0, 100.0), "trend": [_row(200, 90.0, 0.0, 0.5, 5.0, 100.0)],
          "final_per": [{"scenario_idx": 7, "reached": True, "goal": [12.0, 14.0], "traj": _traj},
                        {"scenario_idx": 8, "reached": False, "goal": None}]}]   # 第二局无 traj → 跳过
_gT = P.group_by_party(_recT)
_exs = P._iter_traj_examples(_gT["Continuous-safe"], 3)
_outdir4 = tempfile.mkdtemp(prefix="trb_l2traj_")
_nfig = P.plot_example_trajectories(_gT, _outdir4)
ok("⑫ plot_example_trajectories：final_per 带 traj 的局→ego 航迹按 ρ 着色+他船+goal★ 出非空图（无 traj 局跳过·D42-L2-续）",
   len(_exs) == 1 and _exs[0][1] == 7 and _exs[0][3] == [12.0, 14.0] and _nfig == 1
   and os.path.getsize(os.path.join(_outdir4, "traj_examples_Continuous-safe.png")) > 1000
   and os.path.exists(os.path.join(_outdir4, "traj_examples_Continuous-safe.pdf")))
# ⑫b plot_example_trajectories 边界：无任何 traj 的 grouped → 0 图、不崩
ok("⑫b plot_example_trajectories 边界：grouped 无 traj（_RECS 无 final_per/traj）→ 返回 0 图、不崩",
   P.plot_example_trajectories(P.group_by_party(_RECS), tempfile.mkdtemp(prefix="trb_l2traj0_")) == 0)

# 清理
import shutil
os.unlink(_tmpf.name)
shutil.rmtree(_outdir, ignore_errors=True)
shutil.rmtree(_outdir2, ignore_errors=True)
shutil.rmtree(_outdir3, ignore_errors=True)
shutil.rmtree(_outdir4, ignore_errors=True)

# ⑬ plot_training_diagnostics：curves 内部诊断仪表盘（SAC Q/α·PPO KL/std·自适应只画有数据的指标·无 curves 跳过）
_outdir5 = tempfile.mkdtemp()
def _crec(seed, algo, curves):
    _f = _row(200, 10.0, 0.0, 1.0, 0.0, 90.0)
    return {"party": "Continuous-safe", "kind": "continuous", "colregs_weight": 0.0, "seed": seed,
            "final": _f, "trend": [_f], "continuous_algo": algo, "curves": curves}
def _sac_curves(off):
    return [{"step": s, "ep_rew_mean": -100.0 + s / 1000 + off,
             "critic_loss": 0.5 - s / 2e6, "ent_coef": 0.05 * (1 - s / 3e6)} for s in (50000, 100000, 150000)]
_diag_g = P.group_by_party([_crec(0, "sac", _sac_curves(0)), _crec(1, "sac", _sac_curves(10))])
_ndiag = P.plot_training_diagnostics(_diag_g, _outdir5)
ok("⑬ plot_training_diagnostics：SAC 2 种子产 1 图（PNG 非空·多种子 band 不崩）",
   _ndiag == 1 and os.path.getsize(os.path.join(_outdir5, "training_diagnostics_Continuous-safe.png")) > 10000)
_present = [lab for k, lab, c in P._DIAG_METRICS
           if len(P.agg_trend(_diag_g["Continuous-safe"], k, source="curves")[0]) > 0]
ok("⑬b 自适应：SAC 画 critic_loss/ent_coef·不画 PPO 专属（approx_kl 无数据→不画不崩）",
   "Critic loss / Q (SAC)" in _present and "Entropy temperature alpha (SAC)" in _present
   and "Approx. KL (PPO)" not in _present)
ok("⑬c 无 curves 的方→跳过返回 0（离散 LOG_CURVES=0·不崩）",
   P.plot_training_diagnostics(P.group_by_party([_rec("Base", "unshielded", 0.0, 0,
       [_row(100, 10.0, 5.0, 2.0, 0.0, 100.0)])]), _outdir5) == 0)
# ⑬d plot_run_summary（参考图风格四宫格·每指标独立配色·含到达率面板·user 2026-06-25）：产图不崩 + Arrival 是面板之一
ok("⑬d plot_run_summary 产 run_summary 图（参考图风格·SUMMARY_PANELS 含 Arrival·每指标独立色）·不崩",
   P.plot_run_summary(_diag_g, _outdir5) >= 1
   and any("Arrival rate" in _p[2] for _p in P.SUMMARY_PANELS)
   and os.path.exists(os.path.join(_outdir5, "run_summary_Continuous-safe.png")))
shutil.rmtree(_outdir5, ignore_errors=True)

print()
print("✅ 全部 PASS" if _fail == 0 else f"❌ {_fail} 项 FAIL")
sys.exit(1 if _fail else 0)
