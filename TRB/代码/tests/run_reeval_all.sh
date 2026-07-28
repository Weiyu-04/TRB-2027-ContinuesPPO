#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════════════════
# 全臂【同一趟】重评 → strict 563 头条表（`04 §2` 的报数纪律）
#
# 为什么要有这个脚本（别手敲 REEVAL_CKPTS）：
#   ① `04 §2` 写死：**必须显式点名每一条臂**。不点名 = 自动发现会捞到 277 个存档，其中 54 个 sidecar
#      的 dataset=None、35 个连 sidecar 都没有 ⟹ 撞 fail-closed 闸 ⟹ 整趟评不了；硬绕过更糟（分母静默从
#      563 掉到 312，且没有任何告警）。臂名写错一个字母就漏评一条，手敲极易出错 ⟹ 集中在这里维护。
#   ② **同分母纪律**：所有臂必须落在同一个 strict 563 上。本脚本分组并行后会**逐组核对 strict 键列表
#      逐位相同**，不同就中止（而不是"看着都是 563 就当没事"）。
#   ③ **跨机器纪律**：连续臂的投影 QP 在不同机器/BLAS 上有 ±0.5pt 级数值抖动（离散臂逐位一致）。
#      ⟹ **一张报数表里的每一条臂，必须同一台机器、同一趟评**。别把两台机器的数拼一张表。
#
# 用法：
#   bash 代码/tests/run_reeval_all.sh [并行组数] [输出目录]
#   例：bash 代码/tests/run_reeval_all.sh 8 /root/trb/结果/结果-全臂重评
#
# 耗时估算：单条臂约 8 分钟（单线程）。56 条臂 ÷ 8 组并行 ≈ 1 小时。
# ══════════════════════════════════════════════════════════════════════════════════════════
set -uo pipefail

CODE_DIR="${CODE_DIR:-/root/trb/代码}"
ROOT="$(cd "$CODE_DIR/.." && pwd)"
PY="${PY:-/root/miniconda3/bin/python}"
NGROUP="${1:-8}"
OUT_DIR="${2:-$ROOT/结果/结果-全臂重评}"
mkdir -p "$OUT_DIR"

# ─────────────────────────── 要评的臂（56 条·显式点名）───────────────────────────
ARMS=()
# 【待跑】L232 大集探针·新配方（C 配方 + 官方 1300 · 从零 5M · 脚本 代码/run_l231_bigset.sh）
#   🔴 跑完后【把下面这行取消注释】再评——它自己不贡献泄漏，单独评会得 strict 600，
#      与既有 56 臂同趟评才仍是 563（同分母才可同表）。没跑完就取消注释 ⟹ 闸门 0 会拦"找不到存档"。
# for s in 1 3 4;                   do ARMS+=("Continuous-safe_s${s}_D232bigCppoS${s}"); done      # 3
# 【新】L231 三臂（从零 5.08M · Beta / 修状态机 / 两个都上）
for s in 0 1 2 5 6;               do ARMS+=("Continuous-safe_s${s}_A231betaPpoS${s}"); done      # 5
for s in 0 1 2;                   do ARMS+=("Continuous-safe_s${s}_B231gwsymPpoS${s}"); done     # 3
for s in 0 1 2 3 4 5 6 7 8 9;     do ARMS+=("Continuous-safe_s${s}_C231bothPpoS${s}"); done      # 10
# 【旧】对照组
for s in 0 1 2 3 4 5 6 7 8 9;     do ARMS+=("Continuous-safe_s${s}_L1rateON_ppo_s${s}"); done    # 10 金标(从零)
for s in 0 1 2 3 4 5 6 7 8 9;     do ARMS+=("Continuous-safe_s${s}_wsHOCRppo_s${s}"); done       # 10 主线(热启动)
for s in 1 3 4;                   do ARMS+=("Continuous-safe_s${s}_wsBIGppo_s${s}"); done        # 3  大集探针
for s in 0 1 2 3 4;               do ARMS+=("Discrete-safe_s${s}_discStdW0_s${s}"); done         # 5  对标论文
for s in 0 1 2 3 4;               do ARMS+=("Base_s${s}_baseW0_s${s}"); done                     # 5  无盾
for s in 0 1 2 3 4;               do ARMS+=("Rule-reward_s${s}_rrW0_s${s}"); done                # 5  软奖励
N=${#ARMS[@]}

echo "===== [闸门 0] 环境与存档 ====="
[ -d "$CODE_DIR" ] || { echo "❌ CODE_DIR 不存在：$CODE_DIR"; exit 1; }
"$PY" -c "import torch,stable_baselines3,sb3_contrib,numpy,scipy,shapely,osqp,cvxpy" \
  || { echo "❌ 依赖缺失（照 代码/requirements.txt 装）"; exit 1; }
# ⚠️ 存档目录深度两种都要兼容（本机演练抓出来的）：
#   服务器原生：<root>/结果/checkpoints/           （run_step4e 恒写这里）
#   主窗口本机：<root>/结果/<子目录>/checkpoints/  （按批次归档过）
# ⟹ 找存档一律用 find（不写死层数），REEVAL_CKDIRS 也同时给两种通配。
CKDIRS="$ROOT/*/checkpoints:$ROOT/*/*/checkpoints"
MISS=0
for a in "${ARMS[@]}"; do
  f=$(find "$ROOT" -path "*/checkpoints/$a.zip" -print -quit 2>/dev/null)
  [ -n "$f" ] || { echo "  ❌ 找不到存档：$a"; MISS=$((MISS+1)); }
done
[ "$MISS" -eq 0 ] || { echo "❌ 缺 $MISS 条臂的存档 → 补齐再跑（少一条头条表就画不全）"; exit 1; }
echo "  ✅ $N 条臂的存档全在"

echo "===== [闸门 0.5] 还原存档时间戳（锚点自检的前提）====="
# 存档若经过 git/打包传输，文件 mtime 会被重写 ⟹ reeval 的【锚点自检】会因"sidecar 与 ckpt 不同步"
# 而**整条跳过**。锚点是防"评估配置配错→静默给出错数"的关键防线（`03` L192 那个洞），不能丢。
# 这里按 sidecar 自己记录的 zip_mtime 还原（**先硬校 zip_size 逐字节相同**，不同则跳过不动、并报警）。
# 原生环境本来就不会有这个问题，跑这步无副作用。
"$PY" - "$ROOT" <<'PYEOF'
import json, os, sys, glob
root = sys.argv[1]
ok = bad = miss = 0
cands = (glob.glob(os.path.join(root, "*", "checkpoints", "*.progress.json"))
         + glob.glob(os.path.join(root, "*", "*", "checkpoints", "*.progress.json")))
for p in cands:
    try:
        fp = (json.load(open(p, encoding="utf-8")) or {}).get("ckpt_fingerprint") or {}
    except Exception:
        miss += 1; continue
    z = p[:-len(".progress.json")] + ".zip"
    if not (fp.get("zip_mtime") and os.path.exists(z)):
        miss += 1; continue
    if os.stat(z).st_size != fp.get("zip_size"):
        print("  ⚠️ size 不符，跳过不动：", os.path.basename(z)); bad += 1; continue
    os.utime(z, (fp["zip_mtime"], fp["zip_mtime"])); ok += 1
print(f"  时间戳还原：{ok} 个成功 · {bad} 个 size 不符(未动) · {miss} 个无指纹/无 zip")
PYEOF

echo "===== [闸门 1] 分 $NGROUP 组并行评 $N 条臂 ====="
# 轮转分组：每组都会同时含【小集训练的臂】(贡献泄漏) 与其它 ⟹ 每组的泄漏并集一致 ⟹ 每组 strict 都是 563。
# （大集探针那 3 条自己不贡献泄漏，单独评会得 600；混在组里就仍是 563 —— 这正是"必须同趟"的原因。）
for ((g=0; g<NGROUP; g++)); do
  LIST=""
  for ((i=g; i<N; i+=NGROUP)); do LIST="${LIST}${ARMS[$i]},"; done
  LIST="${LIST%,}"
  [ -n "$LIST" ] || continue
  echo "  组 $g: $(echo "$LIST" | tr ',' '\n' | wc -l) 条"
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  STEP4E_SDIR="$ROOT/scenarios" STEP4E_CODE_DIR="$CODE_DIR" \
  REEVAL_MANIFEST_DIRS="$ROOT/balanced_pool" REEVAL_CKDIRS="$CKDIRS" \
  REEVAL_POOL=official REEVAL_CKPTS="$LIST" REEVAL_TRAJ_KEYS="1,100,1006,1016" \
  REEVAL_OUT="$OUT_DIR/g$g.json" REEVAL_TRAJ_OUT="$OUT_DIR/g${g}_traj.json" \
    "$PY" -B "$CODE_DIR/tests/reeval_official.py" > "$OUT_DIR/g$g.log" 2>&1 &
done
wait
echo "  全部组结束"

echo "===== [闸门 2] 同分母核对 + 合并 ====="
"$PY" - "$OUT_DIR" "$N" <<'PYEOF'
import json, os, sys, glob
out_dir, n_expect = sys.argv[1], int(sys.argv[2])
files = sorted(glob.glob(os.path.join(out_dir, "g*.json")))
if not files:
    raise SystemExit("🔒 一个结果文件都没有 → 看 g*.log")
ref = None; merged = {}; bad = []
for f in files:
    d = json.load(open(f, encoding="utf-8"))
    keys = d.get("strict键")
    if ref is None:
        ref = keys
    elif keys != ref:                       # 🔴 不是"都等于563就行"，是【键列表逐位相同】
        bad.append(os.path.basename(f))
    for k, v in (d.get("结果") or {}).items():
        if not k.startswith("_"):
            merged[k] = v
if bad:
    raise SystemExit(f"🔒 这些组的 strict 键列表与其它组【不一致】：{bad} ⟹ 分母不同、不可同表。中止。")
print(f"  ✅ 各组 strict 键列表逐位相同 · n = {len(ref)}（必须 == 563）")
if len(ref) != 563:
    raise SystemExit(f"🔒 strict 分母 = {len(ref)} ≠ 563 ⟹ 泄漏剔除口径不对，别信这趟数字。")
print(f"  ✅ 合并得 {len(merged)} 条臂（期望 {n_expect}）")
if len(merged) != n_expect:
    print(f"  ⚠️ 条数对不上，缺的臂：看各组 g*.log 里的 ❌")
json.dump(merged, open(os.path.join(out_dir, "all.json"), "w", encoding="utf-8"), ensure_ascii=False)
print(f"  → {os.path.join(out_dir, 'all.json')}")
PYEOF

echo
echo "===== 完成。把整个 $OUT_DIR 目录回传即可 ====="
