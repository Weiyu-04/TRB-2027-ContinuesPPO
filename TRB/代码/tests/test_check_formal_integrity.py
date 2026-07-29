# -*- coding: utf-8 -*-
"""`check_formal_integrity.py` 的**变异测试**（L243-续8）—— 零依赖。

为什么必须有：这个体检脚本是本次「只有一次机会」的采集里**唯一的中途保险**。
上一轮我给它写过 7 条变异测试并且全绿，但独立复审仍在 8 项里挑出 6 项**有真失败却照样打绿灯**。
教训很直白：**造一棵健康的产物树、看它打绿**，证明不了任何事；必须**逐条把树弄坏，看它是不是真的红**。

跑法：  python3 -B 代码/tests/test_check_formal_integrity.py
"""
import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_CHK = os.path.join(_HERE, "check_formal_integrity.py")
_PY = sys.executable

ARMS = {
    "ours": ("Continuous-safe", "_F240oursPpoS"), "disc": ("Discrete-safe", "_F240discPpoS"),
    "base": ("Base", "_F240basePpoS"), "rr": ("Rule-reward", "_F240rrPpoS"),
    "uns": ("Continuous-safe", "_F240unsPpoS"), "ush": ("Continuous-safe", "_F240ushPpoS"),
    "ab0": ("Continuous-safe", "_F240ab0PpoS"), "abB": ("Continuous-safe", "_F240abBPpoS"),
    "abG": ("Continuous-safe", "_F240abGPpoS"),
}
CONT = {a for a, (p, _) in ARMS.items() if p == "Continuous-safe"}
NSEG, SEEDS = 20, [0, 1]

_SHAPE_FULL = {"well_shaping_weight": 200.0, "xtrack_weight": 200.0,
               "park_weight": 20.0, "rate_weight": 1.0,
               "park_radius": 400.0, "park_v_target": 4.0}
_SHAPE_NONE = {k: 0.0 for k in _SHAPE_FULL}
_ACT = {"ours": "beta", "abB": "beta", "abG": "gauss", "ab0": "gauss", "uns": "gauss", "ush": "gauss"}
_GW = {"ours": "symmetric", "abG": "symmetric"}
_SHAPE = {"ours": _SHAPE_FULL, "ab0": _SHAPE_FULL, "abB": _SHAPE_FULL, "abG": _SHAPE_FULL}


def _sig(arm, seed):
    c = {"seed": seed, "total_steps": 10_000_000, "n_seg": NSEG,
         "dataset": "manifest_official_1300.json", "n_envs": 8, "gamma": 0.99}
    if arm in CONT:
        c["act_dist"] = _ACT[arm]
        c["gw_entry"] = _GW.get(arm, "paper")
        c.update(_SHAPE.get(arm, _SHAPE_NONE))
    else:
        c["kind"] = "shielded" if arm == "disc" else "unshielded"
        c["colregs_weight"] = 0.0 if arm == "base" else 1.0
    return c


def _curves(arm, n_win=200):
    """每条 run 200 个窗口。盾开的臂 roll_source 走盾档，uns 走 unshielded。"""
    src = {"unshielded": 80} if arm == "uns" else {"no_obstacle": 60, "projection": 15, "emergency": 5}
    out = []
    for k in range(n_win):
        r = {"step": (k + 1) * 16384, "roll_steps": 16384, "roll_eps": 96,
             "roll_n_act": 16384, "roll_n_pair": 16300, "roll_n_err": 0,
             "roll_source": dict(src), "roll_rho": {"0": 15000, "3": 1384},
             "roll_yaw_sat_frac": 0.3, "roll_acc_sat_frac": 0.2,
             "roll_n_corr": (16384 if (arm in CONT and arm != "uns") else 0)}
        out.append(r)
    return out


def build(root, *, seeds=SEEDS, seg_done=NSEG - 1, finished=True):
    ck = os.path.join(root, "checkpoints")
    seg = os.path.join(ck, "segments")
    os.makedirs(seg, exist_ok=True)
    for arm, (party, tag) in ARMS.items():
        for s in seeds:
            base = os.path.join(ck, f"{party}_s{s}{tag}{s}")
            rec = {"party": party, "kind": "continuous" if arm in CONT else "discrete",
                   "colregs_weight": 0.0, "seed": s, "seg_done": seg_done,
                   "num_timesteps": (seg_done + 1) * 507904,
                   "total_steps": 10_000_000, "n_seg": NSEG, "trend": [{}] * (seg_done + 1),
                   "config_sig": _sig(arm, s), "curves": _curves(arm)}
            open(base + ".progress.json", "w", encoding="utf-8").write(json.dumps(rec, ensure_ascii=False))
            open(base + ".zip", "wb").write(b"x")
            for c in range(seg_done + 1):
                for suf in (".zip", "_vecnorm.pkl", ".progress.json"):
                    open(os.path.join(seg, f"{os.path.basename(base)}@s{c:02d}{suf}"), "wb").write(b"x")
            if finished:
                open(os.path.join(root, f"step4e_partial{tag}{s}.jsonl"), "w",
                     encoding="utf-8").write(json.dumps({"party": party, "seed": s}) + "\n")


def run(root, **env):
    e = dict(os.environ)
    e.update({"SEEDS": " ".join(str(s) for s in SEEDS), "NSEG": str(NSEG), "ARMS": " ".join(ARMS)})
    e.update({k: str(v) for k, v in env.items()})
    p = subprocess.run([_PY, "-B", _CHK, root], capture_output=True, text=True, env=e)
    return p.returncode, p.stdout + p.stderr


_fail = []


def case(name, mutate, *, expect_hard=True, env=None, want_text=None):
    root = tempfile.mkdtemp(prefix="chk_")
    try:
        build(root)
        if mutate:
            mutate(root)
        rc, out = run(root, **(env or {}))
        got_hard = (rc == 1)
        good = (got_hard == expect_hard) and (want_text is None or want_text in out)
        print(("  ✅ " if good else "  ❌ ") + name + f"   [rc={rc}]")
        if not good:
            _fail.append(name)
            print("\n".join("        " + l for l in out.splitlines()[-22:]))
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _edit(root, arm, seed, fn):
    party, tag = ARMS[arm]
    p = os.path.join(root, "checkpoints", f"{party}_s{seed}{tag}{seed}.progress.json")
    d = json.load(open(p, encoding="utf-8"))
    fn(d)
    open(p, "w", encoding="utf-8").write(json.dumps(d, ensure_ascii=False))


print("【基准】一棵健康的树必须打绿（否则下面所有变异都没意义）")
case("健康树 ⟹ 不报硬伤", None, expect_hard=False)

print("\n【A4】只跑一条臂时不该被自己判死（文档里 ARMS=\"ours\" 那条命令原来必定自杀）")


def _only_ours(root):
    for arm, (party, tag) in ARMS.items():
        if arm == "ours":
            continue
        for s in SEEDS:
            for suf in (".progress.json", ".zip"):
                p = os.path.join(root, "checkpoints", f"{party}_s{s}{tag}{s}{suf}")
                os.path.exists(p) and os.remove(p)


case("ARMS=\"ours\" + 树上只有 ours ⟹ 不报硬伤", _only_ours, expect_hard=False, env={"ARMS": "ours"})
case("反证：同一棵树不收窄 ARMS ⟹ 报缺臂", _only_ours, expect_hard=True,
     want_text="已开工的种子里缺")

print("\n【R5】跑到一半崩掉的 run 必须被抓出来（原来看上去和正常在跑的一模一样）")


def _crashed(root):
    _edit(root, "ab0", 1, lambda d: d.update(seg_done=8))
    seg = os.path.join(root, "checkpoints", "segments")
    for f in os.listdir(seg):
        if "_F240ab0PpoS1" in f and int(f.split("@s")[1][:2]) > 8:
            os.remove(os.path.join(seg, f))


case("已写 jsonl（收工了）却只跑了 9/20 段 ⟹ 硬伤", _crashed, want_text="已经收工却没跑满")


def _running(root):
    _crashed(root)
    os.remove(os.path.join(root, "step4e_partial_F240ab0PpoS1.jsonl"))


case("同样只跑了 9 段但**还没写 jsonl**（还在跑）⟹ 不算硬伤", _running, expect_hard=False)

print("\n【R4·最要紧】盾到底开没开 —— config_sig 里根本看不见，只能从 roll_source 看")


def _shield_on_for_uns(root):
    _edit(root, "uns", 0, lambda d: [c.update(roll_source={"no_obstacle": 60, "projection": 20})
                                     for c in d["curves"]])


case("uns 该关盾却采到盾档 ⟹ 硬伤", _shield_on_for_uns, want_text="盾没关掉")


def _shield_off_for_ours(root):
    _edit(root, "ours", 0, lambda d: [c.update(roll_source={"unshielded": 80}) for c in d["curves"]])


case("ours 该开盾却全是 unshielded ⟹ 硬伤", _shield_off_for_ours, want_text="盾没开")


def _no_source(root):
    _edit(root, "ush", 0, lambda d: [c.update(roll_source={}) for c in d["curves"]])


case("开盾臂 roll_source 整个空 ⟹ 硬伤", _no_source, want_text="roll_source 整个是空的")

print("\n【O3】对子必须逐颗种子查（原来只查第一颗 ⟹ 第 5 颗被改坏看不见）")


def _pair_broken_on_second_seed(root):
    _edit(root, "abB", 1, lambda d: d["config_sig"].update(act_dist="gauss", gamma=0.95))


case("对子在**第 2 颗**种子上出现计划外差异 ⟹ 硬伤", _pair_broken_on_second_seed,
     want_text="计划外差异")

print("\n【O4】A 类量：从「曾经非零」深到「全程都有」")


def _thin(root):
    def f(d):
        for c in d["curves"][1:]:
            c["roll_n_act"] = 0
    _edit(root, "ours", 0, f)


case("动作统计只有第 1 个窗口有 ⟹ 至少要报提醒（原来打绿灯）", _thin, expect_hard=False,
     want_text="动作统计只有")


def _all_act_zero(root):
    _edit(root, "disc", 0, lambda d: [c.update(roll_n_act=0) for c in d["curves"]])


case("离散臂 roll_n_act 全 0 ⟹ 硬伤（新逻辑下离散臂也该有）", _all_act_zero, want_text="roll_n_act 全 0")


def _err(root):
    _edit(root, "ab0", 0, lambda d: d["curves"][3].update(roll_n_err=17))


case("采集抛过异常 ⟹ 报提醒（不再静默）", _err, expect_hard=False, want_text="roll_n_err>0")


def _no_corr(root):
    _edit(root, "ush", 1, lambda d: [c.update(roll_n_corr=0) for c in d["curves"]])


case("开盾臂 roll_n_corr 全 0（盾改写量采不到）⟹ 硬伤", _no_corr, want_text="roll_n_corr 全 0")

print("\n【O2】重名：真正的风险是三台机器合并后同名分处两个子目录")


def _dup_across_dirs(root):
    ck = os.path.join(root, "checkpoints")
    ck2 = os.path.join(root, "B机", "checkpoints")
    os.makedirs(ck2, exist_ok=True)
    for f in os.listdir(ck):
        if f.startswith("Continuous-safe_s0_F240oursPpoS0"):
            shutil.copy2(os.path.join(ck, f), os.path.join(ck2, f))


case("同名存档分处两个子目录 ⟹ 硬伤（原检查恒为真、什么都拦不住）", _dup_across_dirs,
     want_text="在盘上出现多次")

print("\n【既有项回归】别把老的查法改坏了")
case("预算不一致（一条 run 的 total_steps 不同）⟹ 硬伤",
     lambda r: _edit(r, "rr", 0, lambda d: d.update(total_steps=5_000_000)),
     want_text="total_steps")
case("NSEG 与要求的对不上 ⟹ 硬伤",
     lambda r: _edit(r, "rr", 0, lambda d: d.update(n_seg=30)), want_text="NSEG=20 对不上")
case("训练集不是官方 1300 ⟹ 硬伤",
     lambda r: _edit(r, "base", 0, lambda d: d["config_sig"].update(dataset="strided")),
     want_text="dataset")
case("分段副本缺了一份 ⟹ 硬伤",
     lambda r: os.remove(os.path.join(r, "checkpoints", "segments",
                                      "Continuous-safe_s0_F240oursPpoS0@s07.zip")),
     want_text="分段副本不全")
case("curves 整个没有（LOG_CURVES 忘了开）⟹ 硬伤",
     lambda r: _edit(r, "abG", 0, lambda d: d.update(curves=[])), want_text="curves 都没有")
case("ARMS 里有未知臂名 ⟹ 硬伤", None, env={"ARMS": "ours 打错了"}, want_text="未知臂名")

print("\n" + "=" * 78)
if _fail:
    print(f"❌ {len(_fail)} 条没过：")
    for f in _fail:
        print("   ·", f)
    sys.exit(1)
print("✅ 全部通过 —— 体检脚本对这 20 种坏法都真的会红，不是打绿灯")
