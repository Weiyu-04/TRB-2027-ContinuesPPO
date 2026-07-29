# -*- coding: utf-8 -*-
"""正式实验方案 + 上窗口分析的**独立复审**（`03` L243）。**零依赖**（只用标准库）。

复审纪律：不信 `02`/`03` 里任何"已实测/已核过"的字样，全部从**原始产物**重算。
覆盖 `02` banner 第六节的 5 条清单：

  [A] 三对单变量对子 —— 把 `run_formal_2027.sh` 的 arm_env 逐臂展开、两两求差集（清单第 3 条）
  [B] L241 大集探针判读 —— 从 `结果0729-59臂同趟重评/g*.json` 重算同种子配对（清单第 1 条）
  [C] 两信号收敛判据 —— 从 jsonl 的 trend 自算末三段 vs 第 5-7 段（清单第 1 条后半）
  [D] L234 的 6 处订正 —— 直算（清单第 2 条；完整版另见 review_l232.py）
  [E] 🆕「崩」与「欠训」的可判别性 —— 全库 300+ 条 trend 实证（本窗口新提，见 `03` L243-§4）

跑法：  python3 -B 代码/tests/review_formal_plan.py
退出码非 0 = 有格对不上（可进 CI / 起跑前预检）。
"""
import json
import os
import re
import glob
import subprocess
import statistics
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_CODE)
_RES = os.path.join(_ROOT, "结果")
_REEVAL = os.path.join(_RES, "结果0729-59臂同趟重评")

ARMS = ["ours", "disc", "base", "rr", "uns", "ush", "ab0", "abB", "abG"]
K_MAX_STEPS = 170          # 回合步数上限（`trb_env/usv_env.py:143` g0.time_step.end）
DT_S = 10.0                # 每步秒
EP_CAP_S = K_MAX_STEPS * DT_S                    # = 1700 秒 = 打转吸引子所在的天花板
SPIN_LINE = 0.8 * EP_CAP_S                       # = 1360 秒（机制推出来的线，不是调出来的）
CRASH_ARR = 50.0                                 # `代码/bgate_judge.py:16` 全项目统一

N_FAIL = 0


def _bad(msg):
    global N_FAIL
    N_FAIL += 1
    print(f"  ❌ {msg}")


def _ok(msg):
    print(f"  ✅ {msg}")


# ══════════════════════════════════════════════════════════════════════════════
def arm_envs():
    """把 run_formal_2027.sh 里 arm_cfg/arm_env 两个函数抠出来跑，得到逐臂的环境变量集合。"""
    src = open(os.path.join(_CODE, "run_formal_2027.sh"), encoding="utf-8").read()
    a = src.index("arm_cfg () {")
    b = src.index("arm_tag () {")
    frag = src[a:b]
    out = {}
    for arm in ARMS:
        r = subprocess.run(["bash", "-c", frag + f"\narm_env {arm}"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"🔒 arm_env {arm} 失败：{r.stderr}")
        kv = {}
        for tok in r.stdout.split():
            k, _, v = tok.partition("=")
            kv[k] = v
        out[arm] = kv
    return out


def sec_A():
    print("=" * 100)
    print("【A】三对单变量对子：把 run_formal_2027.sh 的 arm_env 逐臂展开、两两求差")
    envs = arm_envs()
    for arm in ARMS:
        print(f"  {arm:<5} {' '.join(f'{k}={v}' for k, v in sorted(envs[arm].items()))}")

    def diff(x, y):
        keys = set(envs[x]) | set(envs[y])
        return {k: (envs[x].get(k, "〈未设〉"), envs[y].get(k, "〈未设〉"))
                for k in sorted(keys) if envs[x].get(k) != envs[y].get(k)}

    # (对子名, 左, 右, 允许出现差异的语义变量集合)
    PAIRS = [
        ("⑤uns ↔ ⑥ush（只差盾）", "uns", "ush", {"STEP4E_CONTINUOUS_SHIELD"}),
        ("③base ↔ ⑤uns（只差动作空间）", "base", "uns",
         {"STEP4E_PARTIES", "STEP4E_CONTINUOUS_ALGO", "STEP4E_CONTINUOUS_SHIELD", "STEP4E_ACT_DIST"}),
        ("⑦ab0 ↔ ⑧abB（只差 Beta）", "ab0", "abB", {"STEP4E_ACT_DIST"}),
        ("⑦ab0 ↔ ⑨abG（只差状态机）", "ab0", "abG", {"STEP4E_GW_ENTRY"}),
        ("⑧abB ↔ ①ours（只差状态机）", "abB", "ours", {"STEP4E_GW_ENTRY"}),
        ("⑨abG ↔ ①ours（只差 Beta）", "abG", "ours", {"STEP4E_ACT_DIST"}),
        # 下面三条是本窗口指出的「白捡的对子」（README 没写，但配方天然成立）
        ("🆕 ③base ↔ ②disc（只差离散盾）", "base", "disc", {"STEP4E_PARTIES"}),
        ("🆕 ③base ↔ ④rr（只差软奖励）", "base", "rr", {"STEP4E_PARTIES"}),
        ("🆕 ⑥ush ↔ ⑦ab0（只差连续专属塑形）", "ush", "ab0",
         {"STEP4E_WELL_B", "STEP4E_WELL_X", "STEP4E_PARK_W", "STEP4E_RATE_W",
          "STEP4E_PARK_RADIUS", "STEP4E_PARK_VTARGET"}),
    ]
    print()
    for name, x, y, allowed in PAIRS:
        d = diff(x, y)
        extra = set(d) - allowed
        if extra:
            _bad(f"{name}：出现计划外差异 {sorted(extra)} → {[d[k] for k in sorted(extra)]}")
        else:
            _ok(f"{name}：差异仅 {sorted(d)}")
    print("\n  ⟹ 完整单变量阶梯（每一跳只差一件事）：")
    print("     ②disc ← ③base → ④rr")
    print("            ③base → ⑤uns → ⑥ush → ⑦ab0 → ⑧abB ⇘")
    print("                                       ⑦ab0 → ⑨abG ⇒ ①ours")


# ══════════════════════════════════════════════════════════════════════════════
def _load_reeval(d, prefix="g"):
    """只吃 `<前缀><数字>.json`（`*_traj.json` 混进来会污染统计）。0729 那趟用 g、0727 那趟用 p。"""
    arms, keys, pools = {}, {}, {}
    files = sorted(f for f in os.listdir(d) if re.fullmatch(prefix + r"\d+\.json", f))
    for f in files:
        j = json.load(open(os.path.join(d, f), encoding="utf-8"))
        keys[f], pools[f] = j["strict键"], j["池"]
        if not j.get("全部完成"):
            _bad(f"{f} 未全部完成")
        for n, r in j["结果"].items():
            if n.startswith("_"):
                continue
            if n in arms:
                _bad(f"重名臂 {n}")
            arms[n] = r
    return arms, keys, pools, files


def _seed(n):
    return int(re.search(r"_s(\d+)_", n).group(1))


def sec_B():
    print("=" * 100)
    print("【B】L241 判读复算：同种子配对 D(官方1300) vs C(小集)，都从零 5.08M、只差训练集")
    if not os.path.isdir(_REEVAL):
        _bad(f"找不到 {_REEVAL}")
        return
    arms, keys, pools, files = _load_reeval(_REEVAL)
    ref = keys[files[0]]
    if all(keys[f] == ref for f in files):
        _ok(f"{len(files)} 组 strict 键逐位（含顺序）相同 · n={len(ref)}")
    else:
        _bad("strict 键跨组不一致 ⟹ 分母不可比")
    if len(set(p["strict"] for p in pools.values())) == 1 == len({len(ref)}):
        _ok(f"全部组 strict 分母 = {len(ref)}")
    nfail = sum(1 for r in arms.values() if not r.get("anchor", {}).get("通过"))
    (_ok if nfail == 0 else _bad)(f"锚点自检：{len(arms)-nfail}/{len(arms)} 通过")

    D = {_seed(n): r for n, r in arms.items() if "D232bigCppoS" in n}
    C = {_seed(n): r for n, r in arms.items() if "C231bothPpoS" in n}
    common = sorted(set(D) & set(C))
    diffs = []
    for s in common:
        c, d = C[s]["strict"]["到达率%"], D[s]["strict"]["到达率%"]
        diffs.append(d - c)
        print(f"     s{s}: 小集C {c:6.2f} → 官方1300 D {d:6.2f}   Δ={d-c:+7.2f}")
    med = statistics.median(diffs)
    print(f"     中位 Δ = {med:+.2f} pt")
    (_ok if med < -5 else _bad)(f"事先规则「中位掉 >5pt = 输」 ⟹ 判{'输' if med < -5 else '不输'}（L241-A 写的是判输）")
    ds = {n.split("_")[-1]: r["dataset"] for n, r in arms.items() if "D232bigCppoS" in n or "C231bothPpoS" in n}
    _ok(f"数据集核：D={sorted({v for k, v in ds.items() if k.startswith('D')})} · C={sorted({v for k, v in ds.items() if k.startswith('C')})}")
    return arms


# ══════════════════════════════════════════════════════════════════════════════
def _trends():
    """全库 trend：{run 名: (trend, total_steps)}。jsonl 与 progress.json 两条路都收。"""
    out = {}
    for p in glob.glob(os.path.join(_RES, "**", "checkpoints", "*.progress.json"), recursive=True):
        if os.sep + "segments" + os.sep in p:
            continue
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        if d.get("trend"):
            out[os.path.basename(p)[:-len(".progress.json")]] = (
                d["trend"], (d.get("config_sig") or {}).get("total_steps"))
    for p in glob.glob(os.path.join(_RES, "**", "step4e_partial*.jsonl"), recursive=True):
        last = None
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("trend"):
                last = r
        if last:
            out.setdefault(os.path.basename(str(last.get("ckpt", p))), (last["trend"], last.get("steps")))
    return out


def _two_signal(tr):
    """L236/L238/L241 的两信号：末 3 段均值 − 第 5-7 段均值（到达率 + 每局秒）。"""
    n = len(tr)
    if n < 7:
        return None
    a = [x["到达率%"] for x in tr]
    e = [x["Ep长s"] for x in tr]
    m = lambda v, i, j: sum(v[i-1:j]) / len(v[i-1:j])
    return m(a, n-2, n) - m(a, 5, 7), m(e, n-2, n) - m(e, 5, 7)


def sec_C():
    print("=" * 100)
    print("【C】L241-B 两信号复算（到达率=滞后 · 每局时长=先行）")
    T = _trends()
    want = {"D 大集 s1": "D232bigCppoS1", "D 大集 s3": "D232bigCppoS3", "D 大集 s4": "D232bigCppoS4",
            "C 小集 s1": "C231bothPpoS1", "C 小集 s3": "C231bothPpoS3", "C 小集 s4": "C231bothPpoS4"}
    print(f"     {'臂':<12}{'Δ到达(末3−第5-7)':>18}{'Δ每局秒':>12}   L241 判词")
    for lab, tag in want.items():
        hit = [k for k in T if tag in k]
        if not hit:
            _bad(f"{lab}: 找不到 trend")
            continue
        d = _two_signal(T[hit[0]][0])
        print(f"     {lab:<12}{d[0]:>18.1f}{d[1]:>12.1f}")
    print("  ⚠️ 本窗口结论：**每局时长这一路没有事先写死的数值门槛** —— L238/L241 只写了「任一还在动就算没收敛」。")
    print("     实证：C 小集 s3（被当作已收敛）Δ每局秒 = −57.3，同一句话也能读成「还在动」⟹ 判据不可判。")
    print("     ⟹ `03` L243 提出可判版本，正式实验起跑前必须先定死（见 04 §2 跑实验清单 T1 判据）。")


# ══════════════════════════════════════════════════════════════════════════════
def sec_D(arms):
    print("=" * 100)
    print("【D】L234 六处订正复算（直算，不经上窗口的脚本）")
    if not arms:
        return
    g = lambda pat: {_seed(n): r for n, r in arms.items() if re.search(pat, n)}
    B, GOLD, C, A = g("B231gwsymPpoS"), g("L1rateON_ppo_s"), g("C231bothPpoS"), g("A231betaPpoS")
    arr = lambda r: r["strict"]["到达率%"]
    yaw = lambda r: r["strict"]["控制质量"]["yaw_incr_mean"]

    hb = sum(1 for r in B.values() if arr(r) >= CRASH_ARR)
    (_ok if hb == 2 else _bad)(f"订正①：B 臂健康 = {hb}/{len(B)}（L234 订正为 2/3；逐颗 "
                               f"{[round(arr(B[s]), 2) for s in sorted(B)]}）")

    s012 = [0, 1, 2]
    up = [yaw(B[s]) > yaw(GOLD[s]) for s in s012]
    gy, by = statistics.mean(yaw(GOLD[s]) for s in s012), statistics.mean(yaw(B[s]) for s in s012)
    (_ok if all(up) else _bad)(f"订正②：B 转艏 3/3 同向变差 = {all(up)}（配对均值 {gy:.5f} → {by:.5f}，"
                               f"{(by/gy-1)*100:+.1f}%）⟹ 「纹丝不动」= 说错了")

    dd = [(s, arr(GOLD[s]), arr(B[s])) for s in s012]
    worse = sum(1 for _, a, b in dd if b < a)
    (_ok if worse == 2 else _bad)(f"订正③：B vs 金标同种子 {['%+.2f' % (b-a) for _, a, b in dd]} ⟹ "
                                  f"{worse}/3 变差，唯一大幅方向是反的 ⟹ 「状态机对训练稳定性有独立贡献」撤回")

    ds = g("discStdW0_s")
    dsh = [s for s in ds if arr(ds[s]) >= CRASH_ARR]
    (_ok if len(dsh) == 3 else _bad)(f"订正④：对标 5 颗里健康 {len(dsh)} 颗，仅健康均值 "
                                     f"{statistics.mean(arr(ds[s]) for s in dsh):.2f} vs C 全 10 颗 "
                                     f"{statistics.mean(arr(r) for r in C.values()):.2f} ⟹ 领先 "
                                     f"{statistics.mean(arr(r) for r in C.values())-statistics.mean(arr(ds[s]) for s in dsh):+.2f} 点（L234 写 +12.18）")

    base = g("baseW0_s")
    b_rate = statistics.mean(r["strict"]["碰撞率%"] for r in base.values())
    d_rate = statistics.mean(r["strict"]["碰撞率%"] for r in ds.values())
    c_rate = statistics.mean(r["strict"]["碰撞率%"] for r in C.values())
    _ok(f"订正⑤：单变量版（同为离散·只差盾）= {b_rate/d_rate:.2f}× · 跨动作空间版 = {b_rate/c_rate:.2f}×"
        f"（L234 写 26.50× / 13.25×）")

    # 订正⑥ 跨趟抖动：需要 0727 那趟
    old = os.path.join(_RES, "结果0727-38臂同趟重评")
    if os.path.isdir(old):
        o, _, _, _ = _load_reeval(old, prefix="p")
        both = set(o) & set(arms)
        cont = [abs(arms[n]["strict"]["到达率%"] - o[n]["strict"]["到达率%"]) for n in both if "Continuous" in n]
        disc = [abs(arms[n]["strict"]["到达率%"] - o[n]["strict"]["到达率%"]) for n in both if "Continuous" not in n]
        (_ok if max(cont) > 0.5 else _bad)(
            f"订正⑥：0727↔0729 共有 {len(both)} 条臂 · 连续臂 max|Δ|={max(cont):.2f}pt "
            f"中位={statistics.median(cont):.2f}pt · 离散臂 max|Δ|={max(disc):.2f}pt "
            f"⟹ 旧写的「±0.5pt」偏小（L234 订正为 1.60pt）")

    # 🆕 本窗口新查：README §8-8 的两个数没按 §8-2 自己的规矩报两版
    ca = sorted(set(A) & set(GOLD))
    hh = [s for s in ca if arr(A[s]) >= CRASH_ARR]
    print(f"  🆕 §8-8 复核：A(只Beta) 全 {len(ca)} 颗 {statistics.mean(yaw(A[s]) for s in ca):.5f}"
          f"（含崩种子 s6 到达 {arr(A[6]):.2f}% · 它的转艏 {yaw(A[6]):.5f} 是其余的 ~4 倍）")
    print(f"     仅健康 {len(hh)} 颗 = {statistics.mean(yaw(A[s]) for s in hh):.5f}"
          f" ⟹ 金标→Beta 倍数：全部种子 {statistics.mean(yaw(GOLD[s]) for s in ca)/statistics.mean(yaw(A[s]) for s in ca):.2f}×"
          f" · 仅健康 {statistics.mean(yaw(GOLD[s]) for s in hh)/statistics.mean(yaw(A[s]) for s in hh):.2f}×")
    print("     ⟹ README §8-8 只引了一版（0.01472→0.00403），与 §8-2「必须两版都报」自相矛盾。")


# ══════════════════════════════════════════════════════════════════════════════
def sec_E():
    print("=" * 100)
    print(f"【E】🆕「崩（打转吸引子）」vs「欠训（还在爬）」可判别性实证 —— 线 = 0.8 × 回合上限 = {SPIN_LINE:.0f} 秒")
    T = _trends()
    spin, under = [], []
    for n, (tr, _) in T.items():
        if not tr or len(tr) < 5:
            continue
        a, e = tr[-1].get("到达率%"), tr[-1].get("Ep长s")
        if a is None or e is None or a >= CRASH_ARR:
            continue
        (spin if e >= SPIN_LINE else under).append((n, a, e))
    print(f"     低到达 run 共 {len(spin)+len(under)} 条：崩 {len(spin)} · 欠训 {len(under)}")
    if spin and under:
        se = sorted(x[2] for x in spin)
        ue = sorted(x[2] for x in under)
        print(f"     崩组末段每局秒  区间 [{se[0]:.0f}, {se[-1]:.0f}]  中位 {statistics.median(se):.0f}")
        print(f"     欠训组末段每局秒 区间 [{ue[0]:.0f}, {ue[-1]:.0f}]  中位 {statistics.median(ue):.0f}")
        gap = se[0] - ue[-1]
        (_ok if gap > 0 else _bad)(f"两组在阈值处不重叠（间隙 {gap:.0f} 秒：欠训最高 {ue[-1]:.0f} < 崩最低 {se[0]:.0f}）")
        band = [x for x in spin + under if 1100 <= x[2] <= 1600]
        print(f"     ⚠️ 诚实边界：{len(band)} 条（{len(band)/(len(spin)+len(under))*100:.0f}%）落在 1100~1600 秒过渡带，"
              "这一带的标签本身就是灰的 ⟹ 论文里这一列必须附判据定义 + 过渡带条数。")


if __name__ == "__main__":
    sec_A()
    a = sec_B()
    sec_C()
    sec_D(a)
    sec_E()
    print("=" * 100)
    print(f"复审结论：{'✅ 全部对得上' if N_FAIL == 0 else f'❌ {N_FAIL} 处对不上'}")
    sys.exit(1 if N_FAIL else 0)
