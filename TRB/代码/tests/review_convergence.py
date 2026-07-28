# -*- coding: utf-8 -*-
"""收敛性 + 末段崩溃扫描（`03` L236）—— 回答两个报数口径问题，零烧卡、只用标准库。

  ① **5M 到期时各臂练透了没有**？（决定报数表能不能说成"收敛后性能"）
  ② **有没有臂"练好了又崩回去"**？（我们评的是末段存档 ⟹ 这种臂会被评到最差时刻）

⚠️ 用的是**训练期 40 场景里程碑数**，**只判"收没收敛/有没有崩"，一个数都不进任何报数表**（L232 铁律）。
判据用【3 段一块】（120 局·二项 SE≈4.5pt）而非单段（40 局·SE≈7.9pt）——单段跳 5~10 点全在噪声里。

跑法：  python3 -B 代码/tests/review_convergence.py
"""
import json, os, statistics as st
R = "/home/user/TRB-2027-ContinuesPPO/TRB/结果"
DIRS = [R+"/结果0702-地基第1版-12:18", R+"/结果0710-22:00-10种子最优方案",
        R+"/结果0727-大集指标提升", R+"/结果0728-beta测试训练", R+"/结果0727-大集探针"]
def tr(tag, s):
    for d in DIRS:
        for pat in (f"step4e_partial_{tag}_s{s}.jsonl", f"step4e_partial_{tag}S{s}.jsonl"):
            p = os.path.join(d, pat)
            if os.path.exists(p):
                return [t["到达率%"] for t in json.loads(open(p, encoding="utf-8").read().strip().split("\n")[-1])["trend"]]
    return None
ARMS = [("金标", "L1rateON_ppo", list(range(10))), ("A 只Beta", "A231betaPpo", [0,1,2,5,6]),
        ("B 只状态机", "B231gwsymPpo", [0,1,2]), ("C 两个都上", "C231bothPpo", list(range(10)))]
SE = 4.5   # 120 局的二项标准误（p≈0.9 时更小，取保守值）
print(f"判据：末3段均值 − 第5-7段均值 > {2*SE:.0f} 点（=2 个标准误）才算『还在明显上升』\n")
print(f"{'臂/种子':<14}{'第5-7段':>9}{'末3段':>9}{'涨幅':>8}   判定")
tot = {}
for nm, tag, seeds in ARMS:
    n_up = n_all = 0; ups = []
    for s in seeds:
        t = tr(tag, s)
        if t is None: continue
        a, b = st.mean(t[4:7]), st.mean(t[-3:])
        # 崩掉的种子（末3段<20）不参与"还能不能练出来"的判断，它们是另一回事
        if b < 20: tag2 = "   〔崩·另论〕"
        elif b - a > 2*SE: tag2 = "   🔴 还在明显上升"; n_up += 1; ups.append(s)
        else: tag2 = "   已走平"
        if b >= 20: n_all += 1
        print(f"  {nm+' s'+str(s):<12}{a:>9.1f}{b:>9.1f}{b-a:>+8.1f}{tag2}")
    tot[nm] = (n_up, n_all, ups)
    print()
print("【小结·只数没崩的种子】")
for nm, (u, n, ups) in tot.items():
    print(f"  {nm:<12} 还在明显上升 {u}/{n}" + (f"  ← 种子 {ups}" if ups else ""))

# ── ② 末段崩溃扫描（`03` L236-B）───────────────────────────────────────────────
print("\n" + "="*96)
print("【② 末段崩溃扫描】判据：训练途中峰值段 − 最后一段 > 20 点 = 『练好了又掉下来』")
print("   为什么要扫：我们评的是**末段存档** ⟹ 这种臂会被评到它最差的时刻 = 单方向不公平。")
DISC = [("Base 无盾","baseW0",list(range(5))), ("Rule-reward","rrW0",list(range(5))),
        ("Discrete-safe 对标","discStdW0",list(range(5))), ("大集探针","wsBIGppo",[1,3,4])]
hits, n_scanned = [], 0
for nm, tag, seeds in ARMS + DISC:
    for s_ in seeds:
        t = tr(tag, s_)
        if not t: continue
        n_scanned += 1
        pk = max(t); drop = pk - t[-1]
        if drop > 20:
            hits.append((nm, s_, t.index(pk)+1, pk, t[-1], drop, t))
if hits:
    print(f"  {'臂/种子':<22}{'峰值段':>7}{'峰值':>7}{'末段':>7}{'跌幅':>7}   曲线")
    for nm, s_, i, pk, last, drop, t in hits:
        print(f"  🔴{nm+' s'+str(s_):<20}{i:>7}{pk:>7.1f}{last:>7.1f}{drop:>7.1f}   " + " ".join(f"{x:.0f}" for x in t))
else:
    print("  （无）")
print(f"\n  ⟹ 扫了 {n_scanned} 条臂，命中 {len(hits)} 条。")
print("  ⚠️ 分段存档是覆盖式的（save_segment_checkpoint docstring:『覆盖最新』）")
print("     ⟹ 峰值那个存档已经没了，想拿真数只能重训那颗种子（`03` L236-D③）。")
