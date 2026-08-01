# -*- coding: utf-8 -*-
"""两版存档 + 只算收敛种子 —— **只读重评产物，一个数字都不手抄**。

回答 §5.2 / §5.5 / §7 里承诺过、但 20 页稿里放不下表的三件事：
  ① 最佳存档版 vs 末段存档版（官方测试集 600）差多少；
  ② 只算收敛种子时，头条结论变不变；
  ③ 收敛判据按【验证集】算（anchor.记录值，n=100）与按测试集算，收敛种子数一不一样。

用法：python3 结果/结果0801-出图草图/make_ckpt_versions.py
"""
import json, os, re
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
B = os.path.join(ROOT, "Paper", "正式实验", "02_重评产物")
PAT = re.compile(r"F240([A-Za-z0-9]+?)Ppo[Ss](\d+)")
CONV = 50.0  # 收敛判据：验证集到达率 ≥ 50%（与 §5.2 声明一致）

ROWS = [("base", "Discrete baseline"), ("rr", "Discrete, soft reward"),
        ("disc", "Discrete, masking"), ("uns", "Continuous, no shield"),
        ("ush", "Continuous, shield"), ("ab0", "Ablation reference"),
        ("abB", "Ablation, bounded only"), ("abG", "Ablation, symmetric only"),
        ("ours", "Ours")]


def load(pas):
    d = json.load(open(os.path.join(B, pas, "all.json"), encoding="utf-8"))
    out = {}
    for k, v in d.items():
        m = PAT.search(k)
        if m:
            out.setdefault(m.group(1), {})[int(m.group(2))] = v
    return out


def arr(r):  return r["strict"]["到达率%"]
def vio(r):  return r["strict"]["违规次数/局"]
def col(r):  return r["strict"]["碰撞率%"]
def yaw(r):  return r["strict"]["控制质量"]["yaw_incr_mean"]
def val(r):  return r.get("anchor", {}).get("记录值")      # 验证集（n=100）到达率


def sign_test(a, b):
    """逐种子配对符号检验的双侧 p（小者为优；返回 胜-负-p）。"""
    from math import comb
    w = sum(1 for x, y in zip(a, b) if x < y)
    l = sum(1 for x, y in zip(a, b) if x > y)
    n = w + l
    if n == 0:
        return w, l, 1.0
    k = min(w, l)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return w, l, min(1.0, p)


def main():
    BEST, LAST = load("正式-最佳"), load("正式-末段")

    print("=" * 78)
    print("① 两版存档在官方测试集 600 上的对照（每格 = 8 颗种子的中位数）")
    print("=" * 78)
    print(f"{'配置':<26}{'到达 best':>10}{'到达 last':>10}{'差':>8}"
          f"{'违规 best':>10}{'违规 last':>10}")
    gaps = []
    for tag, name in ROWS:
        ab = st.median([arr(x) for x in BEST[tag].values()])
        al = st.median([arr(x) for x in LAST[tag].values()])
        vb = st.median([vio(x) for x in BEST[tag].values()])
        vl = st.median([vio(x) for x in LAST[tag].values()])
        gaps.append(ab - al)
        print(f"{name:<26}{ab:>10.1f}{al:>10.1f}{ab-al:>+8.1f}{vb:>10.2f}{vl:>10.2f}")
    print(f"\n九条配置的到达率差：中位 {st.median(gaps):+.1f} 点 · "
          f"范围 {min(gaps):+.1f} 到 {max(gaps):+.1f} 点")

    # 头条结论在末段版下是否成立
    print("\n" + "=" * 78)
    print("② 头条结论在【末段存档版】下是否仍成立（违规次数/局，越小越好）")
    print("=" * 78)
    for pas, D in (("正式-最佳", BEST), ("正式-末段", LAST)):
        ours = [vio(D["ours"][s]) for s in sorted(D["ours"])]
        line = []
        for tag, name in ROWS:
            if tag == "ours":
                continue
            oth = [vio(D[tag][s]) for s in sorted(D[tag])]
            w, l, p = sign_test(ours, oth)
            line.append(f"{name}: {w}-{l} p={p:.4f}")
        med = st.median(ours)
        rank = sorted(st.median([vio(x) for x in D[t].values()]) for t, _ in ROWS)
        print(f"\n[{pas}] Ours 违规中位 {med:.2f} · 九条里排第 {rank.index(med)+1}")
        for x in line:
            print("   ", x)

    # 收敛判据两种口径
    print("\n" + "=" * 78)
    print("③ 收敛种子数：按【验证集】(anchor.记录值, n=100) vs 按【测试集】(strict, n=600)")
    print("=" * 78)
    print(f"{'配置':<26}{'验证集口径':>12}{'测试集口径':>12}")
    for tag, name in ROWS:
        vs = [val(x) for x in BEST[tag].values()]
        cv = sum(1 for a in vs if a is not None and a >= CONV)
        ct = sum(1 for x in BEST[tag].values() if arr(x) >= CONV)
        flag = "" if cv == ct else "   ← 不一致"
        print(f"{name:<26}{cv:>10}/8{ct:>11}/8{flag}")

    # 只算收敛种子
    print("\n" + "=" * 78)
    print("④ 只算收敛种子（验证集口径）时的主表数字（官方测试集 600 · 最佳存档）")
    print("=" * 78)
    print(f"{'配置':<26}{'n':>3}{'到达':>8}{'碰撞':>8}{'违规':>8}{'转艏':>9}")
    conv_med = {}
    for tag, name in ROWS:
        keep = [x for x in BEST[tag].values()
                if val(x) is not None and val(x) >= CONV]
        if not keep:
            print(f"{name:<26}{0:>3}  —")
            continue
        conv_med[tag] = (st.median([arr(x) for x in keep]),
                         st.median([col(x) for x in keep]),
                         st.median([vio(x) for x in keep]),
                         st.median([yaw(x) for x in keep]))
        a, c, v, y = conv_med[tag]
        print(f"{name:<26}{len(keep):>3}{a:>8.1f}{c:>8.2f}{v:>8.2f}{y:>9.4f}")
    order = sorted(conv_med, key=lambda t: conv_med[t][2])
    print("\n只算收敛种子时，违规从小到大：",
          " < ".join(dict(ROWS)[t] for t in order[:4]), "…")

    # 选择偏倚：两版在测试集上的实测差（论文该报的第三个数，见 03 L243-续45 B）
    print("\n" + "=" * 78)
    print("⑤ 选择偏倚的三个数（03 L243-续45 B：三个数出处不同，不许互相冒充）")
    print("=" * 78)
    lifts = []
    for tag, _ in ROWS:
        for s in BEST[tag]:
            if s in LAST[tag]:
                lifts.append(arr(BEST[tag][s]) - arr(LAST[tag][s]))
    print(f"  · 官方测试集 600 上，最佳存档比末段存档高：均值 {st.mean(lifts):+.2f} 点 · "
          f"中位 {st.median(lifts):+.2f} 点 · {len(lifts)} 条 run")
    print(f"    （被抬高的 {sum(1 for x in lifts if x > 0)} 条 / 被压低的 "
          f"{sum(1 for x in lifts if x < 0)} 条）")
    print("  · 验证集上的实测抬升（选存档工具自报）= +4.03 点（03 L243-续45 B）")
    print("  · 理论期望 = 二项 SE 4 点 × 20 个候选取最大 ≈ +7.5 点（03 L236-A）")


if __name__ == "__main__":
    main()
