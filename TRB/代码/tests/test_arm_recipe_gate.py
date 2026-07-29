# -*- coding: utf-8 -*-
"""闸门 2「配方真落地了吗」对账逻辑的**变异测试**（L243-续8 A8）—— 零依赖。

背景：起跑脚本的闸门 2 原来只查周边设施（分段存档写没写、训练集对不对、采集在不在），
**恰恰不查定义这 9 条臂的那几个开关**。失败场景：哪天改名/重构让 `STEP4E_ACT_DIST` 不再被读到
（值合法所以 fail-fast 不响）⟹ 9 条臂冒烟全绿，两天烧完才发现 `ours`/`abB` 跑的其实是高斯
⟹ 2×2 消融塌成两组重复，两个对子直接作废。

本文件做两件事：
  ① 把 shell 里的 `arm_cfg` 抽出来实跑展开，核对 9 条臂的配方本身没写错（对子干不干净）；
  ② 把闸门 2 内嵌的那段对账 python 抠出来，喂**正确的**和**被做过手脚的** run_metadata，
     证明它对前者放行、对后者拦停。

跑法：  python3 -B 代码/tests/test_arm_recipe_gate.py
"""
import json
import os
import re
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_SH = os.path.join(os.path.dirname(_HERE), "run_formal_2027.sh")
_PY = sys.executable
ARMS = ["ours", "disc", "base", "rr", "uns", "ush", "ab0", "abB", "abG"]
_fail = []


def check(name, cond, extra=""):
    print(("  ✅ " if cond else "  ❌ ") + name + (("   ← " + str(extra)) if (extra and not cond) else ""))
    if not cond:
        _fail.append(name)


# ── 把 shell 里的 arm_cfg / arm_env / arm_field 抽出来，在真 bash 里跑 ──
_src = open(_SH, encoding="utf-8").read()


def _slice(start_marker, end_marker):
    i = _src.index(start_marker)
    j = _src.index(end_marker, i)
    return _src[i:j]


_FUNCS = _slice("arm_cfg () {", "N_RUN=")


def sh(expr):
    p = subprocess.run(["bash", "-c", _FUNCS + "\n" + expr], capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip() or f"rc={p.returncode}")
    return p.stdout.strip()


print("【一】9 条臂的配方本身（实跑 shell 展开，不看注释）")
env = {a: dict(kv.split("=", 1) for kv in sh(f'arm_env {a}').split()) for a in ARMS}
for a in ARMS:
    check(f"{a} 能展开出配方", bool(env[a]), env[a])
tags = {a: sh(f'arm_tag {a}') for a in ARMS}
check("9 个 TAG 互不相同", len(set(tags.values())) == 9, tags)
check("9 个 TAG 都含 ppo（否则 run_step4e 的 PPO 隔离闸会拦）",
      all("ppo" in t.lower() for t in tags.values()), tags)

PAIRS = [("uns", "ush", {"STEP4E_CONTINUOUS_SHIELD"}),
         ("ab0", "abB", {"STEP4E_ACT_DIST"}),
         ("ab0", "abG", {"STEP4E_GW_ENTRY"}),
         ("abB", "ours", {"STEP4E_GW_ENTRY"}),
         ("abG", "ours", {"STEP4E_ACT_DIST"}),
         # base ↔ rr 的单变量是 **r_colregs 权重**，而它是由"方"决定的（Base=0.0 / Rule-reward=1.0，
         # `usv_shield.py:196-198` 同一个类、只有 colregs_weight 不同）⟹ 环境串上体现为 PARTIES 一项。
         ("base", "rr", {"STEP4E_PARTIES"})]
for x, y, want in PAIRS:
    dx, dy = env[x], env[y]
    diff = {k for k in set(dx) | set(dy) if dx.get(k) != dy.get(k)}
    check(f"{x} ↔ {y} 只差 {sorted(want) or '（环境串完全相同·差异在 run_step4e 内部）'}",
          diff == want, f"实际差 {sorted(diff)}")
check("ab0/abB/abG/ours 构成完整 2×2（Beta × 对称让路）",
      {(env[a].get("STEP4E_ACT_DIST"), env[a].get("STEP4E_GW_ENTRY", "paper"))
       for a in ("ab0", "abB", "abG", "ours")}
      == {("gauss", "paper"), ("beta", "paper"), ("gauss", "symmetric"), ("beta", "symmetric")})
check("无盾臂 uns 的连续专属塑形全 0（否则 run_step4e:224 会 fail-fast）",
      all(env["uns"].get(k, "0") in ("0", "0.0") for k in
          ("STEP4E_PARK_W", "STEP4E_RATE_W", "STEP4E_WELL_B", "STEP4E_WELL_X")), env["uns"])

print("\n【二】闸门 2 的对账逻辑：正确的放行、做过手脚的拦停")
# 从 shell 里抠出那段内嵌 python（`"$PY" -c "` 与收尾 `"` 之间）
m = re.search(r'"\$PY" -c "\n(import json,sys\n.*?)\n" "\$MET"', _src, re.S)
assert m, "没在起跑脚本里找到闸门 2 的对账 python —— 结构变了，本测试要同步改"
GATE_PY = m.group(1)

_SHAPE_FULL = {"well_shaping_weight": 200.0, "xtrack_weight": 200.0,
               "park_weight": 20.0, "rate_weight": 1.0}
_SHAPE_NONE = {k: 0.0 for k in _SHAPE_FULL}


def meta_for(arm):
    e = env[arm]
    full = "STEP4E_WELL_B=200" in sh(f'arm_env {arm}')
    d = {"n_envs": 8, "log_curves": True, "keep_segments": True,
         "gw_entry": e.get("STEP4E_GW_ENTRY", "paper")}
    d.update(_SHAPE_FULL if full else _SHAPE_NONE)
    if "STEP4E_ACT_DIST" in e:
        d["act_dist"] = e["STEP4E_ACT_DIST"]
    if "STEP4E_CONTINUOUS_SHIELD" in e:
        d["continuous_shield"] = (e["STEP4E_CONTINUOUS_SHIELD"] == "1")
    return d


def gate(arm, meta):
    fd, p = tempfile.mkstemp(suffix=".json")
    os.write(fd, json.dumps({"run_config": meta}, ensure_ascii=False).encode())
    os.close(fd)
    try:
        args = [_PY, "-c", GATE_PY, p, sh(f'arm_field {arm} SHIELD'), sh(f'arm_field {arm} ACT'),
                sh(f'arm_field {arm} GW'), sh(f'arm_field {arm} SHAPE'), arm]
        r = subprocess.run(args, capture_output=True, text=True)
        return r.returncode, r.stdout + r.stderr
    finally:
        os.remove(p)


for a in ARMS:
    rc, out = gate(a, meta_for(a))
    check(f"{a}：配方与真落地一致 ⟹ 放行", rc == 0, out.strip()[:200])

print("\n  ── 做手脚（每一条都是真会毁掉一个对子的改法）──")
MUT = [
    ("ours 的 Beta 被静默换成高斯（消融 2×2 塌成两组重复）", "ours", {"act_dist": "gauss"}),
    ("ours 的对称让路没生效（回落 paper）", "ours", {"gw_entry": "paper"}),
    ("uns 的盾没关掉（『盾值多少』这一对作废）", "uns", {"continuous_shield": True}),
    ("ush 的盾没开", "ush", {"continuous_shield": False}),
    ("ab0 漏了 park 塑形（⑦vs⑥ 不再是那套组合）", "ab0", {"park_weight": 0.0}),
    ("ab0 漏了 rate 塑形", "ab0", {"rate_weight": 0.0}),
    ("ush 被误开了塑形（⑤⑥ 孪生对不再逐字同配方）", "ush", {"well_shaping_weight": 200.0}),
    ("并行环境数被改（rollout 网格错位、两臂叠不上图）", "disc", {"n_envs": 4}),
    ("内部曲线没开（离散臂 A 类量全丢）", "base", {"log_curves": False}),
    ("离散臂被塞了 symmetric（它根本不吃这个键）", "rr", {"gw_entry": "symmetric"}),
]
for name, arm, patch in MUT:
    m2 = meta_for(arm)
    m2.update(patch)
    rc, out = gate(arm, m2)
    check(name + " ⟹ 拦停", rc != 0, out.strip()[:160])

# keep_segments 走的是 grep，不在这段 python 里；单独证明它查的是 run_metadata 而不是 jsonl
print("\n【三】keep_segments 查的是 run_metadata（不是 jsonl —— 那里根本没有这个键）")
check("闸门 2 的 keep_segments 用 $MET", 'keep_segments": true\'' in _src and '"$MET"' in _src)
check("$MET 指向 run_metadata<TAG>.json", 'MET="$RES_DIR/run_metadata${T}.json"' in _src)

print("\n" + "=" * 74)
if _fail:
    print(f"❌ {len(_fail)} 条没过：")
    for f in _fail:
        print("   ·", f)
    sys.exit(1)
print("✅ 全部通过 —— 9 条臂的配方干净，且闸门 2 对 10 种「开关没真落地」都会拦")
