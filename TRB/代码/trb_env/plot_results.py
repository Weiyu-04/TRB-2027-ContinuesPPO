"""TRB Node L L2 出版级绘图（读 run_step4e 的 partial jsonl → 四宫格训练曲线 + 钱图）。
====================================================================================
**纯下游消费者**：只读 `结果/step4e_partial{tag}.jsonl` 记录、产 PNG/PDF，**不碰 evaluate/run_step4e
的钱图【计算】路径**（钱图数值仍由 evaluate.ViolationCounter 等算、本模块只把已算好的 final/trend 画出来）。

数据来源（run_step4e 每条 (party,seed) 记录·train_eval_one[_continuous]）：
  · record["final"]  = 末段 Table III 五/六列 {step, 到达率%, 碰撞率%, 违规次数/局, 紧急步%, [兜底步%], Ep长s}
  · record["trend"]  = [上面那种 row, ...]（每训练分段一个点 = 学习曲线）
  · record["curves"] = [{step, ep_rew_mean, actor_loss, ...}]（CAT6 内部曲线·连续臂恒有/离散臂 LOG_CURVES=1 才有）
多种子 = 同 party 多条记录 → 按 step 对齐求 mean±std（种子方差 error band / error bar）。

出版级 rcParams（D42 ③⑥）：Times New Roman / 300dpi(投稿可 600) / 全英文标签无抽象名词 / 矢量 PDF + 高分 PNG 双出。
matplotlib `Agg` 后端（无显示·服务器/CI 安全）。Times New Roman 缺失自动回退 DejaVu Serif（仍出图、不崩）。
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")                       # 无显示后端（screen 后台 / CI 安全）
import matplotlib.pyplot as plt              # noqa: E402

# 四方显示名 + 颜色 + 标记（键 = run_step4e PARTIES 的 party 名；Continuous-safe = 我方主角）
# 配色 = 冷暖对比方案（user 2026-06-19 选 C）：冷(灰/钢蓝)对暖、Discrete(青绿) vs Continuous(亮红) 区分最清、我方主角红最跳。
PARTY_STYLE = {
    "Base":            {"label": "Baseline",               "color": "#7F8C8D", "marker": "o"},   # 蓝灰
    "Rule-reward":     {"label": "Rule-reward",            "color": "#2E86C1", "marker": "s"},   # 钢蓝
    "Discrete-safe":   {"label": "Discrete-safe",          "color": "#17A589", "marker": "^"},   # 青绿
    "Continuous-safe": {"label": "Continuous-safe (ours)", "color": "#E74C3C", "marker": "D"},   # 亮红（我方·hero）
}
PARTY_ORDER = ["Base", "Rule-reward", "Discrete-safe", "Continuous-safe"]

# 四宫格指标 → 出版英文标签（无抽象名词·审稿人直读，D42 ⑥）
PANEL_METRICS = [
    ("到达率%",     "Arrival rate (%)"),
    ("碰撞率%",     "Collision rate (%)"),
    ("违规次数/局", "COLREGs violations per episode"),
    ("紧急步%",     "Emergency-control steps (%)"),
]

# 单 run/方 训练摘要四宫格（参考图风格·user 2026-06-25：每【指标】独立配色·非单色/全红）。
# (指标键, 取数源, 出版英文标签, 颜色, band 下界裁剪)；Episode reward 读 curves(ep_rew_mean)·其余读 trend。
SUMMARY_PANELS = [
    ("ep_rew_mean", "curves", "Episode reward",     "#1F4E96", None),  # 蓝
    ("到达率%",      "trend",  "Arrival rate (%)",   "#1E8449", 0.0),   # 绿
    ("Ep长s",       "trend",  "Episode length (s)", "#A23BB5", None),  # 品红
    ("碰撞率%",      "trend",  "Collision rate (%)", "#C0392B", 0.0),   # 红
]


def set_pub_style(dpi: int = 300):
    """出版级 rcParams（Times New Roman / 高 dpi / serif 数学字体）。每次绘图前调=幂等。
    字号 = user 2026-06-19 选"再大一档"（标题16/标签15/刻度13/图例13·配略缩的 figsize 使字占比更醒目）。"""
    plt.rcParams.update({
        "figure.dpi": dpi, "savefig.dpi": dpi,
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.titlesize": 16, "axes.labelsize": 15,
        "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 13,
        "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
        "lines.linewidth": 2.4, "savefig.bbox": "tight",
    })


def load_records(path: str) -> list:
    """读 partial jsonl 所有记录（坏行跳过·容错·与 run_step4e.read_records 同口径）。"""
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except Exception:
                continue
    return recs


def group_by_party(records: list) -> dict:
    """{party 名: [该方各种子记录]}（绘图按方聚合·跨种子求方差）。"""
    g: dict = {}
    for r in records:
        g.setdefault(r.get("party"), []).append(r)
    return g


def _mean_std(values):
    """多种子 (均值, **样本标准差 ddof=1**) —— 在【干净有限数值输入】下**与 run_step4e.agg_mean_std 逐位同值（bit-identical）**
    （= Table III ±值来源·"对齐 USV 方法论" CLAUDE §6）：空→(nan,nan)、单值→(v,**0.0** 非 nan)、N≥2 用 ddof=1。
    ⚠️ std 必须用 `math.sqrt(var)`（**不是 `var ** 0.5`**）才与 agg_mean_std 逐位相同：二者末位差 1 ULP（`var**0.5` 走
    `float.__pow__`、`math.sqrt` 走 C `sqrt`），约 0.13% 的干净浮点会非逐位等价（`03` L59）。test_plot_results ⑦c 锁此不变量。
    ⚠️ 另顺带滤 None/NaN/bool（防 float(None) 抛错 + 非数值点跳过）= plot 侧防御性加固，对 None/NaN/bool 输入与
    agg_mean_std【有意分歧】(后者会抛 TypeError / nan 传染)——但钱图实际数据流恒为有限 float，两者在该路径上逐位同值。"""
    vals = [float(v) for v in values
            if isinstance(v, (int, float)) and not isinstance(v, bool) and v == v]   # v==v 滤 NaN
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(vals) / n
    if n == 1:
        return m, 0.0
    var = sum((v - m) ** 2 for v in vals) / (n - 1)            # 样本标准差 ddof=1（同 agg_mean_std·钱图误差棒对得上 Table III）
    return m, math.sqrt(var)                                   # math.sqrt（非 var**0.5）= 与 agg_mean_std 逐位同值（`03` L59）


def agg_trend(party_recs: list, metric: str, source: str = "trend"):
    """按 step 对齐该方多种子的 source[metric] → (steps, mean, std〔ddof=1〕)。
    source="trend"（学习曲线·默认·向后兼容）或 "curves"（CAT6 内部曲线·如 ep_rew_mean·连续臂恒有/离散 LOG_CURVES=1）。
    种子网格一致（同 total_steps/n_seg）→ 同 step 聚多种子；缺该 step 的种子不计入该点（按 step 值聚·不错位）。
    滤 None（curves 的 ep_rew_mean 早期无完成 episode=None；trend 恒为有限 float·此滤对其无副作用=向后兼容）。
    ⚠️ 若各种子 step 网格不一致（如中途改 STEP4E_NSEG 续跑同 jsonl）→ 部分 step 种子数不齐、band 会误导：
       用 warn_misaligned_grids() 先检测、make_all_figures 会告警（#2/#4 防御·`03` L58）。"""
    by_step: dict = {}
    for r in party_recs:
        for row in r.get(source, []):
            if metric in row and row[metric] is not None:
                by_step.setdefault(row["step"], []).append(row[metric])
    steps = sorted(by_step)
    _ms = [_mean_std(by_step[s]) for s in steps]
    mean = np.array([m for m, _ in _ms])
    std = np.array([s for _, s in _ms])
    return np.array(steps, dtype=float), mean, std


def warn_misaligned_grids(grouped: dict, source: str = "trend", metric: str = "到达率%") -> list:
    """检测每方各种子的 step 网格是否一致（不一致 = 可能中途改 STEP4E_NSEG 续跑同 jsonl → 学习曲线某些 step
    种子数不齐、mean±std band 会误导·#2/#4 `03` L58）。打印告警、返回不齐的方名列表（供测试）。
    ⚠️ 钱图【数值表 Table III】按 (party,seed) 取终值·**不受此影响**；仅学习曲线/趋势子图可视化受影响。"""
    misaligned = []
    for party, recs in grouped.items():
        grids = set()
        for r in recs:
            g = tuple(sorted(row["step"] for row in r.get(source, [])
                             if metric in row and row[metric] is not None))
            if g:                                            # 仅有数据的种子参与比对
                grids.add(g)
        if len(grids) > 1:
            misaligned.append(party)
            print(f"⚠️ plot_results: 方 '{party}' 各种子 step 网格不一致（{len(grids)} 种·source={source}）"
                  f"——疑中途改 STEP4E_NSEG 续跑同 jsonl；学习曲线 band/均值线会误导"
                  f"（钱图数值表不受影响·`03` L58 #2/#4）。", flush=True)
    return misaligned


def _band_bounds(mean, std, lower=None):
    """种子方差 band 的 (下界, 上界)=mean∓std；lower 非 None → 下界裁到 lower（率/计数指标用 0.0·防 band 探入
    负值区·#3 `03` L58）。ep_rew_mean 回报可负 → lower=None 不裁。"""
    lo = mean - std
    if lower is not None:
        lo = np.clip(lo, lower, None)
    return lo, mean + std


def _fill_band(ax, steps, mean, std, color, lower=None):
    """画 mean±std 方差 band（下界按 _band_bounds 可选裁剪）。"""
    lo, hi = _band_bounds(mean, std, lower)
    ax.fill_between(steps, lo, hi, color=color, alpha=0.15)


def final_stats(party_recs: list, metric: str):
    """该方终值（record["final"][metric]）跨种子 (mean, std〔ddof=1〕)（柱状图 error bar）。无数据→(nan,nan)。"""
    vals = [r["final"][metric] for r in party_recs
            if r.get("final") and metric in r["final"]]
    return _mean_std(vals)


def _save(fig, out_base: str):
    """双出：高分 PNG + 矢量 PDF（投稿用）。"""
    fig.savefig(out_base + ".png")
    fig.savefig(out_base + ".pdf")
    plt.close(fig)


def plot_four_panel(grouped: dict, out_base: str):
    """2×2 四宫格：到达/碰撞/违规/紧急，每格四方学习曲线（mean line + 种子方差 band）vs 训练步。"""
    set_pub_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 5.9))
    for ax, (metric, ylabel) in zip(axes.flat, PANEL_METRICS):
        for party in PARTY_ORDER:
            if party not in grouped:
                continue
            st = PARTY_STYLE[party]
            steps, mean, std = agg_trend(grouped[party], metric)
            if len(steps) == 0:
                continue
            ax.plot(steps, mean, color=st["color"], marker=st["marker"],
                    markersize=6, label=st["label"])
            _fill_band(ax, steps, mean, std, st["color"], lower=0.0)   # 率/计数 ≥0 → 裁负值区（#3·L58）
        ax.set_xlabel("Training steps")
        ax.set_ylabel(ylabel)
    axes.flat[0].legend(loc="best", frameon=False)
    fig.tight_layout()
    _save(fig, out_base)


def plot_money_figure(grouped: dict, out_base: str, metric: str = "到达率%",
                      metric_label: str = "Arrival rate (%)"):
    """钱图：(a) 学习曲线（指标 vs 步·四方·方差 band）+ (b) 终值柱状（四方 mean±std error bar）。"""
    set_pub_style()
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(8.4, 3.9))
    # (a) 学习曲线
    for party in PARTY_ORDER:
        if party not in grouped:
            continue
        st = PARTY_STYLE[party]
        steps, mean, std = agg_trend(grouped[party], metric)
        if len(steps) == 0:
            continue
        axL.plot(steps, mean, color=st["color"], marker=st["marker"],
                 markersize=6, label=st["label"])
        _fill_band(axL, steps, mean, std, st["color"], lower=0.0)   # 到达率 ≥0 → 裁负值区（#3·L58）
    axL.set_xlabel("Training steps")
    axL.set_ylabel(metric_label)
    axL.set_title("(a) Learning curves")
    axL.legend(loc="best", frameon=False)
    # (b) 终值柱状 + 种子方差 error bar
    parties = [p for p in PARTY_ORDER if p in grouped]
    means = [final_stats(grouped[p], metric)[0] for p in parties]
    stds = [final_stats(grouped[p], metric)[1] for p in parties]
    colors = [PARTY_STYLE[p]["color"] for p in parties]
    labels = [PARTY_STYLE[p]["label"] for p in parties]
    x = np.arange(len(parties))
    axR.bar(x, means, yerr=stds, color=colors, alpha=0.85, capsize=4,
            error_kw={"elinewidth": 1.0})
    axR.set_xticks(x)
    axR.set_xticklabels(labels, rotation=20, ha="right")
    axR.set_ylabel("Final " + metric_label)
    axR.set_title("(b) Final performance")
    fig.tight_layout()
    _save(fig, out_base)


def plot_training_return(grouped: dict, out_base: str) -> int:
    """训练回报曲线：ep_rew_mean（**原始非归一化** episode 回报·callback 自算·`03` L54-续）vs 训练步·四方。
    连续臂(SAC hero)恒有 curves；离散臂仅 STEP4E_LOG_CURVES=1 才有 → 无 curves 的方自动跳过。
    ⚠️ 回报可负 → band **不裁**下界（lower=None·区别于率/计数图）。返回画出的方数（0=无任一方有 curves·仍出空图不崩）。"""
    set_pub_style()
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    n_drawn = 0
    for party in PARTY_ORDER:
        if party not in grouped:
            continue
        st = PARTY_STYLE[party]
        steps, mean, std = agg_trend(grouped[party], "ep_rew_mean", source="curves")
        if len(steps) == 0:                          # 该方无 curves / ep_rew_mean 全 None → 跳过
            continue
        ax.plot(steps, mean, color=st["color"], marker=st["marker"], markersize=6, label=st["label"])
        _fill_band(ax, steps, mean, std, st["color"], lower=None)    # 回报可负·不裁（#3 区别于率/计数）
        n_drawn += 1
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Episode return (raw)")
    ax.set_title("Training return")
    if n_drawn:
        ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    _save(fig, out_base)
    return n_drawn


# ρ 态势 → 颜色（CAT5 例图航迹着色）：ρ0 常规/keep(灰)·ρ1-4 给路机动(暖色渐变)·ρ5 紧急(红·RHO_EMERGENCY)。
_RHO_COLORS = {0: "#95A5A6", 1: "#F1C40F", 2: "#E67E22", 3: "#D35400", 4: "#8E44AD", 5: "#E74C3C"}


def _iter_traj_examples(party_recs: list, max_examples: int) -> list:
    """从该方各种子 record 的 final_per 收集【带 traj 的】episode（最多 max_examples 个）。
    产 (seed, scenario_idx, traj, goal, reached)。生产须 STEP4E_TRAJ_IDXS 开（默认 0,1,2）→ final_per 选中局才有 traj。"""
    out = []
    for r in party_recs:
        seed = r.get("seed")
        for ep in (r.get("final_per") or []):
            tj = ep.get("traj")
            if tj:
                out.append((seed, ep.get("scenario_idx"), tj, ep.get("goal"), ep.get("reached")))
                if len(out) >= max_examples:
                    return out
    return out


def _plot_one_traj(ax, traj: list, goal, reached):
    """单局轨迹：ego 航迹(细线 + 按 ρ 着色散点) + 他船航迹(虚线) + ego/他船起点 + 目标★。纯读 traj 列表。"""
    ex = [p["ego_x"] for p in traj]
    ey = [p["ego_y"] for p in traj]
    rh = [p.get("rho") for p in traj]
    ax.plot(ex, ey, "-", color="#34495E", lw=1.0, alpha=0.5, zorder=1)            # ego 航迹细线
    for rho in sorted({r for r in rh if r is not None}):                          # 按 ρ 分组散点（图例每 ρ 一项）
        xs = [ex[i] for i in range(len(traj)) if rh[i] == rho]
        ys = [ey[i] for i in range(len(traj)) if rh[i] == rho]
        ax.scatter(xs, ys, s=18, color=_RHO_COLORS.get(rho, "#95A5A6"),
                   label=f"ρ{rho}", zorder=3, edgecolors="none")
    ox = [p["obs_x"] for p in traj if p.get("obs_x") is not None]                 # 他船航迹（仅预测窗内步）
    oy = [p["obs_y"] for p in traj if p.get("obs_y") is not None]
    if ox:
        ax.plot(ox, oy, "--", color="#7F8C8D", lw=1.4, alpha=0.8, zorder=2, label="obstacle")
        ax.plot(ox[0], oy[0], "^", color="#7F8C8D", ms=9, zorder=4)               # 他船起点
    ax.plot(ex[0], ey[0], "o", color="#2C3E50", ms=9, zorder=5, label="ego start")
    if goal is not None:
        ax.plot(goal[0], goal[1], "*", color="#27AE60", ms=20, zorder=6, label="goal")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")


def plot_example_trajectories(grouped: dict, out_dir: str, max_examples: int = 3) -> int:
    """CAT5 示例轨迹图（每方一图·subplots=各代表局）：ego 航迹按 ρ 着色 + 他船航迹 + 目标★，展示给路/迎面/追越避碰。
    无 traj 的方跳过（生产须 STEP4E_TRAJ_IDXS 开·默认 0,1,2 → final_per 选中局带 traj）。返回产出图数。
    ρ 着色：ρ0 常规/keep(灰)·ρ1-4 给路机动(暖色)·ρ5 紧急(红)。"""
    set_pub_style()
    n_figs = 0
    for party in PARTY_ORDER:
        if party not in grouped:
            continue
        examples = _iter_traj_examples(grouped[party], max_examples)
        if not examples:
            continue
        ncol = len(examples)
        fig, axes = plt.subplots(1, ncol, figsize=(4.2 * ncol, 4.0), squeeze=False)
        for ax, (seed, idx, traj, goal, reached) in zip(axes[0], examples):
            _plot_one_traj(ax, traj, goal, reached)
            ax.set_title(f"sc{idx} · seed{seed} · {'reached' if reached else 'not reached'}")   # 纯 ASCII（避 ✓ 字形缺失 UserWarning·投稿图洁净）
        h, lbls = axes[0][0].get_legend_handles_labels()        # 图例去重（合并同 ρ/标记）
        _seen = {}
        for hi, li in zip(h, lbls):
            _seen.setdefault(li, hi)
        fig.legend(_seen.values(), _seen.keys(), loc="lower center",
                   ncol=min(len(_seen), 6), frameon=False, bbox_to_anchor=(0.5, -0.04))
        fig.suptitle(f"{PARTY_STYLE[party]['label']} — example trajectories", y=1.02)
        fig.tight_layout()
        _save(fig, os.path.join(out_dir, f"traj_examples_{party}"))
        n_figs += 1
    return n_figs


# 训练内部诊断指标目录（curves 字段 → 出版英文标签, band 下界裁剪 None=不裁/0.0=非负量裁0）。
# 按算法【自动只画该 run 有数据的指标】：SAC 出 Q/α/actor，PPO 出 KL/clip/policy_std/熵/value，离散 PPO 另有 explained_variance/ret_rms_var。
_DIAG_METRICS = [
    ("ep_rew_mean",        "Episode return (raw)",              None),   # 通用·逐回合回报滚动均值（= user 要的"episode 变化下的奖励"）
    ("critic_loss",        "Critic loss / Q (SAC)",             0.0),    # SAC 专属·Q 值损失（user 要的 Q-value 类）
    ("actor_loss",         "Actor loss (SAC)",                  None),   # SAC 专属
    ("ent_coef",           "Entropy temperature alpha (SAC)",   0.0),    # SAC 专属·自动熵温度 α
    ("ent_coef_loss",      "Entropy-temp loss (SAC)",           None),
    ("approx_kl",          "Approx. KL (PPO)",                  0.0),    # PPO 专属·更新步长健康度（发散预警）
    ("clip_fraction",      "Clip fraction (PPO)",               0.0),    # PPO 专属
    ("entropy_loss",       "Policy entropy loss (PPO)",         None),   # PPO 专属·探索（熵塌缩预警）
    ("value_loss",         "Value loss (PPO)",                  0.0),    # PPO 专属
    ("policy_std",         "Policy std / exploration (PPO)",    0.0),    # PPO 专属·动作标准差（探索量·崩溃预警）
    ("explained_variance", "Explained variance (PPO)",          None),  # 离散 PPO·critic 拟合度
    ("ret_rms_var",        "Reward running variance (VecNorm)",  0.0),   # 离散臂·reward 归一化方差（L39 诊断信号）
]


def _party_algo(party_recs: list) -> str:
    """该方算法标签（图题用）：连续臂读 continuous_algo（sac/ppo）；离散臂 = MaskablePPO。"""
    for r in party_recs:
        a = r.get("continuous_algo")
        if a:
            return a.upper()
    return "MaskablePPO"


def plot_training_diagnostics(grouped: dict, out_dir: str) -> int:
    """每方训练内部诊断仪表盘（读 curves·CAT6 内部曲线）：逐回合回报 ep_rew_mean + 算法专属量 vs 训练步。
    **自适应**：每方只画【该 run 有数据】的指标（SAC→Q/α/actor·PPO→KL/clip/policy_std/熵/value·离散另有 explained_variance/ret_rms_var）。
    多种子 → mean±std band（种子方差·诊断 s4 型"学了又崩"看 approx_kl 飙/policy_std 崩/熵塌）。无 curves 的方跳过。返回产出图数。
    x 轴=训练步（1 步=10s 仿真·回合≈50–170 步·见图题）。"""
    set_pub_style()
    n_figs = 0
    for party in PARTY_ORDER:
        if party not in grouped:
            continue
        present = []
        for key, label, clip in _DIAG_METRICS:
            steps, mean, std = agg_trend(grouped[party], key, source="curves")
            if len(steps) > 0:                       # 仅画该方有数据的指标（SAC/PPO 字段集不同→自适应）
                present.append((label, clip, steps, mean, std))
        if not present:                              # 该方无 curves（离散 LOG_CURVES=0）→ 跳过
            continue
        ncol = min(3, len(present))
        nrow = math.ceil(len(present) / ncol)
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.3 * nrow), squeeze=False)
        flat = axes.flatten()
        st = PARTY_STYLE[party]
        for ax, (label, clip, steps, mean, std) in zip(flat, present):
            ax.plot(steps, mean, color=st["color"], lw=2.2)
            _fill_band(ax, steps, mean, std, st["color"], lower=clip)   # 多种子方差 band（单种子 std=0=仅线）
            ax.set_ylabel(label)
            ax.set_xlabel("Training steps")
        for ax in flat[len(present):]:               # 隐藏多余子图（指标数非 3 的倍数时）
            ax.set_visible(False)
        n_seed = len(grouped[party])
        fig.suptitle(f"{st['label']} — training diagnostics ({_party_algo(grouped[party])}, {n_seed} seed) "
                     f"· 1 step = 10 s · episode ~50-170 steps", y=1.0)
        fig.tight_layout()
        _save(fig, os.path.join(out_dir, f"training_diagnostics_{party}"))
        n_figs += 1
    return n_figs


def plot_run_summary(grouped: dict, out_dir: str) -> int:
    """每方训练摘要四宫格（参考图风格·`SUMMARY_PANELS`·每指标独立配色）：Episode reward / Arrival rate /
    Episode length / Collision rate vs 训练步。Episode reward 读 curves(ep_rew_mean)·其余读 trend。
    多种子→mean±std band（单种子=仅线，同参考图）。无数据的指标格隐藏。Times serif·浅网格。返回产出图数。
    （user 2026-06-25 风格参考：不同指标用不同颜色·而非单色/全红。）"""
    set_pub_style()
    n_figs = 0
    for party in PARTY_ORDER:
        if party not in grouped:
            continue
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), squeeze=False)
        flat = axes.flatten()
        any_data = False
        for ax, (key, source, label, color, clip) in zip(flat, SUMMARY_PANELS):
            steps, mean, std = agg_trend(grouped[party], key, source=source)
            if len(steps) == 0:                      # 该指标无数据（如离散 LOG_CURVES=0 无 ep_rew）→ 隐藏该格
                ax.set_visible(False)
                continue
            any_data = True
            ax.plot(steps, mean, color=color, lw=2.4)
            _fill_band(ax, steps, mean, std, color, lower=clip)   # 多种子方差 band（单种子 std=0=仅线）
            ax.set_ylabel(label)
            ax.set_xlabel("Training steps")
        if not any_data:
            plt.close(fig)
            continue
        n_seed = len(grouped[party])
        st = PARTY_STYLE.get(party, {"label": party})
        fig.suptitle(f"{st['label']} — training summary ({_party_algo(grouped[party])}, {n_seed} seed) "
                     f"· 1 step = 10 s · episode ~50-170 steps", y=1.0)
        fig.tight_layout()
        _save(fig, os.path.join(out_dir, f"run_summary_{party}"))
        n_figs += 1
    return n_figs


def make_all_figures(jsonl_path: str, out_dir: str) -> dict:
    """主入口：读 jsonl → 四宫格 + 钱图（到达率）+ 训练回报曲线（ep_rew_mean）+ 训练内部诊断仪表盘（SAC/PPO 专属量）+ CAT5 示例轨迹 落 out_dir。返回 grouped。"""
    os.makedirs(out_dir, exist_ok=True)
    grouped = group_by_party(load_records(jsonl_path))
    warn_misaligned_grids(grouped, source="trend")               # #2/#4：学习曲线种子网格不齐告警（钱图数值表不受影响）
    plot_four_panel(grouped, os.path.join(out_dir, "four_panel"))
    plot_money_figure(grouped, os.path.join(out_dir, "money_figure"))
    plot_training_return(grouped, os.path.join(out_dir, "training_return"))
    plot_training_diagnostics(grouped, out_dir)                  # 每方训练内部诊断（SAC Q/α · PPO KL/clip/std/熵 · 自适应·无 curves 跳过）
    plot_run_summary(grouped, out_dir)                           # 每方训练摘要四宫格（参考图风格·每指标独立配色：reward/arrival/length/collision）
    plot_example_trajectories(grouped, out_dir)                  # CAT5 示例轨迹（无 traj 的方自动跳过）
    return grouped


if __name__ == "__main__":                  # 手动：python -m trb_env.plot_results <jsonl> <out_dir>
    import sys
    _jsonl = sys.argv[1] if len(sys.argv) > 1 else "结果/step4e_partial_4way.jsonl"
    _out = sys.argv[2] if len(sys.argv) > 2 else "结果/figures"
    g = make_all_figures(_jsonl, _out)
    print(f"✅ 绘图完成 → {_out}/（four_panel + money_figure + training_return + traj_examples_* · PNG+PDF）；方={sorted(g)}")
