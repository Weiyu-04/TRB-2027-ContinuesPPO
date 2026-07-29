#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════════════════
# 全臂【同一趟】重评 → 同分母头条表（`04 §2` 的报数纪律）
#   分母由 REEVAL_EXPECT_STRICT 显式声明：小集训练的臂 = 563；正式实验全部官方 1300 = 600。
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
#   例：bash 代码/tests/run_reeval_all.sh 8 /root/trb/结果/结果-全臂重评          # legacy 59 条(563)
#   正式实验两趟（`03` L243-§7·**必须同一台机器背靠背跑**）：
#     PASS=formal_last bash 代码/tests/run_reeval_all.sh 8 /root/trb/结果/正式-末段
#     BUDGET_SEG=30 python3 -B 代码/tests/select_best_ckpt.py 结果 结果/_best_seg30.json
#     PASS=formal_best SELJSON=/root/trb/结果/_best_seg30.json \
#       bash 代码/tests/run_reeval_all.sh 8 /root/trb/结果/正式-最佳
#     PASS=traj bash 代码/tests/run_reeval_all.sh 3 /root/trb/结果/正式-轨迹全集    # 约 72 分钟 · 约 176 MB
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

# ══════════════════════════════════════════════════════════════════════════════════════════
# 🆕 `03` L243-§7：**正式实验的两趟重评**（默认仍是 legacy = 原来那 59 条历史臂，一字未动）
#
#   PASS=legacy       （默认）59 条探索期臂 · 小集训练 ⟹ strict **563**
#   PASS=formal_last  正式 9 条臂 × N 颗种子 · **末段存档** ⟹ strict **600**
#   PASS=formal_best  正式 9 条臂 × N 颗种子 · **验证集最佳存档**（吃 select_best_ckpt.py 的产物）⟹ **600**
#   PASS=traj         🆕 **轨迹专趟**：9 条臂 × TRAJ_SEEDS（默认 s0/s1/s2）· **全部 600 个场景都采轨迹**
#                     ⟹ 出 Fig.4 时**你想画哪个场景就画哪个**，不必回头重跑评估
#
# 🔴 三条为什么必须这么切（复审抓出来的，别自己简化）：
#   ① **分母不同**：正式臂全部在官方 1300 上训、与测试 600 零交集 ⟹ 谁都不贡献泄漏 ⟹ 600。
#      而历史 59 条是 563。**两边的数绝对不能进同一张表**，所以做成两个 PASS，不是一个。
#   ② **最佳存档那趟扫不到分段副本**：本脚本靠 `find "*/checkpoints/$a.zip"`，而副本在
#      `checkpoints/segments/` 子目录里 ⟹ 必须由 `select_best_ckpt.py` 给出**显式全路径**
#      （`reeval_official.py:292` 支持直接给路径）。⟹ formal_best 从它的产物 JSON 里读。
#   ③ 🔴 **无盾臂必须单独一个进程组**：`run_step4e.py:955` 的 `shield=` 取的是**进程级模块变量**、
#      **不从存档 sidecar 回读**（对比 `gw_entry` 是回读的）⟹ 混在同组里会把**整组的盾都关掉**。
#      ⟹ 下面把 `F240unsPpo` 摘出来，用 `STEP4E_CONTINUOUS_SHIELD=0` 单跑；其余组走默认（有盾）。
#      同机同趟仍然守得住（同机重跑逐位可复现已坐实），只是多起一个进程组。
# ══════════════════════════════════════════════════════════════════════════════════════════
PASS="${PASS:-legacy}"
FORMAL_SEEDS="${FORMAL_SEEDS:-0 1 2 3 4 5 6 7 8 9 10 11}"
#: 正式 9 条臂的 (party, TAG) —— 与 `代码/run_formal_2027.sh` 的 arm_cfg 一一对应，改那边这边要同步
FORMAL_SPEC=(
  "Continuous-safe:_F240oursPpoS"   "Discrete-safe:_F240discPpoS"
  "Base:_F240basePpoS"              "Rule-reward:_F240rrPpoS"
  "Continuous-safe:_F240unsPpoS"    "Continuous-safe:_F240ushPpoS"
  "Continuous-safe:_F240ab0PpoS"    "Continuous-safe:_F240abBPpoS"
  "Continuous-safe:_F240abGPpoS"
)
NOSHIELD_TAG="_F240unsPpoS"          # 唯一需要 shield=0 的臂

# 🆕 `03` L243-续3（user 2026-07-29：论文要「多算法轨迹放一张图」）：轨迹场景从 **4 个扩到 23 个**。
#   原来只采 4 个 ⟹ 事先不知道哪几个画出来有代表性，挑不出来就得**重跑一趟评估**。
#   一条轨迹约 11 KB ⟹ 23 个场景 × 108 条臂 ≈ 27 MB，**基本免费** ⟹ 多采、事后再选。
# 🔴 **按【几何】分层挑，绝不按【结果】挑**（看了成绩再决定画哪个场景 = 变相 cherry-pick）：
#   只用会遇类型（场景自身属性、与算法无关）分层，每类用 `stride_pick` **确定性等距**取 10 个，
#   外加强制保留旧口径那 4 个（新老两趟的轨迹图可直接对照）。
#   推导与复核：`python3 -B 代码/tests/pick_traj_keys.py --verify "<下面这串>"`
#   （该脚本自带与 `03` L111 全库分类的硬比对：head-on 709 / crossing 1291，对不上直接中止）
TRAJ_KEYS="${TRAJ_KEYS:-1,5,100,190,217,371,410,540,608,714,852,920,992,1006,1016,1142,1181,1326,1382,1546,1602,1764,1802}"

# ─────────────────────────── 要评的臂（59 条·显式点名）───────────────────────────
ARMS=()
# 【已跑完 2026-07-28·`03` L238】L232 大集探针·新配方（C 配方 + 官方 1300 · 从零 5M · 脚本 代码/run_l231_bigset.sh）
#   🔴 它自己【不贡献泄漏】⟹ 单独评会得 strict 600，与既有 56 臂**同趟**评才仍是 563（同分母才可同表）。
#   轮转分组已验：8/6/4 组下每组都仍含贡献泄漏的臂 ⟹ 每组分母都是 563（起跑后仍要逐组核一遍再出表）。
for s in 1 3 4;                   do ARMS+=("Continuous-safe_s${s}_D232bigCppoS${s}"); done      # 3  大集探针(新配方)
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
# ── 正式实验两趟：把 ARMS 整个换掉（legacy 时上面那份原样保留）─────────────────────
EXPECT_STRICT="${REEVAL_EXPECT_STRICT:-563}"
ARMS_NOSHIELD=()
case "$PASS" in
  legacy) ;;
  formal_last)
    ARMS=(); EXPECT_STRICT="${REEVAL_EXPECT_STRICT:-600}"
    for spec in "${FORMAL_SPEC[@]}"; do
      P="${spec%%:*}"; T="${spec##*:}"
      for s in $FORMAL_SEEDS; do
        if [ "$T" = "$NOSHIELD_TAG" ]; then ARMS_NOSHIELD+=("${P}_s${s}${T}${s}")
        else                                ARMS+=("${P}_s${s}${T}${s}"); fi
      done
    done ;;
  formal_best)
    ARMS=(); EXPECT_STRICT="${REEVAL_EXPECT_STRICT:-600}"
    SELJSON="${SELJSON:-$ROOT/结果/_best_seg30.json}"
    [ -f "$SELJSON" ] || { echo "❌ PASS=formal_best 需要 select_best_ckpt.py 的产物：$SELJSON"; \
                           echo "   先跑：BUDGET_SEG=<段数> python3 -B 代码/tests/select_best_ckpt.py 结果 $SELJSON"; exit 1; }
    # 从产物里读【最佳存档】的显式全路径（副本缺失的它自己就不会列进来）
    mapfile -t _SEL < <("$PY" -c "
import json,sys
d=json.load(open(sys.argv[1],encoding='utf-8'))
s=d.get('REEVAL_CKPTS_最佳存档') or ''
print('\n'.join(x for x in s.split(',') if x.strip()))" "$SELJSON")
    [ "${#_SEL[@]}" -gt 0 ] || { echo "❌ $SELJSON 里没有可用的最佳存档路径（分段副本没落地？）"; exit 1; }
    for p in "${_SEL[@]}"; do
      case "$p" in *"$NOSHIELD_TAG"*) ARMS_NOSHIELD+=("$p") ;; *) ARMS+=("$p") ;; esac
    done ;;
  traj)
    # 🆕 `03` L243-续4（user 2026-07-29：「场景可视化我还得手动去挑，你别提前替我选死了」）：
    #   **全部 600 个测试场景都采轨迹** ⟹ 出 Fig.4 时想画哪个画哪个，不必回头重跑评估。
    #   成本实测：单条轨迹 ≈ 11.1 KB ⟹ 9 臂 × 3 种子 × 600 场景 ≈ **176 MB**、约 72 分钟（纯评估）。
    #   🔴 只取少数几颗种子（默认 s0/s1/s2）——轨迹图是**定性插图**，不承担任何定量声明；
    #      种子多了只是把 176 MB 变成 705 MB，换不来更多信息。
    #   🔴 **种子必须事先声明**（默认 0 1 2），别跑完了再挑一颗好看的种子说"就用这颗"。
    #      挑场景是你的自由（图是插图）；但**图注必须写明用的是哪颗种子、哪个场景号**（README §8-16）。
    ARMS=(); EXPECT_STRICT="${REEVAL_EXPECT_STRICT:-600}"
    TRAJ_SEEDS="${TRAJ_SEEDS:-0 1 2}"
    for spec in "${FORMAL_SPEC[@]}"; do
      P="${spec%%:*}"; T="${spec##*:}"
      for s in $TRAJ_SEEDS; do
        if [ "$T" = "$NOSHIELD_TAG" ]; then ARMS_NOSHIELD+=("${P}_s${s}${T}${s}")
        else                                ARMS+=("${P}_s${s}${T}${s}"); fi
      done
    done
    # 全池 600 个键：与 `run_step4e.make_split(2000, 0.30, 0)` **同一套逻辑**（已独立核过·`03` L243-§1）
    TRAJ_KEYS=$("$PY" -c "
import random
ids=list(range(2000)); random.Random(0).shuffle(ids)
print(','.join(str(k) for k in sorted(ids[:600])))")
    echo "  轨迹专趟：全部 $(echo "$TRAJ_KEYS" | tr ',' '\n' | wc -l) 个测试场景 × 9 臂 × 种子[$TRAJ_SEEDS]" ;;
  *) echo "❌ PASS 只接受 legacy / formal_last / formal_best / traj，得 '$PASS'"; exit 1 ;;
esac
export REEVAL_EXPECT_STRICT="$EXPECT_STRICT"
N=${#ARMS[@]}
N_UNS=${#ARMS_NOSHIELD[@]}
N_ALL=$(( N + N_UNS ))
echo "═══ 重评 PASS=$PASS ═══ 有盾/离散 $N 条 + 无盾单独组 $N_UNS 条 = $N_ALL 条 · 期望 strict=$EXPECT_STRICT"

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
# 🆕 formal_best 传进来的是**全路径**（分段副本在 checkpoints/segments/ 子目录，find 那条扫不到）
#    ⟹ 含 "/" 的按路径直接查文件，其余照旧按名字 find。
for a in "${ARMS[@]}" ${ARMS_NOSHIELD[@]+"${ARMS_NOSHIELD[@]}"}; do
  case "$a" in
    */*) [ -f "$a.zip" ] || { echo "  ❌ 找不到存档（路径）：$a.zip"; MISS=$((MISS+1)); } ;;
    *)   f=$(find "$ROOT" -path "*/checkpoints/$a.zip" -print -quit 2>/dev/null)
         [ -n "$f" ] || { echo "  ❌ 找不到存档：$a"; MISS=$((MISS+1)); } ;;
  esac
done
[ "$MISS" -eq 0 ] || { echo "❌ 缺 $MISS 条臂的存档 → 补齐再跑（少一条头条表就画不全）"; exit 1; }
echo "  ✅ $N_ALL 条臂的存档全在"

echo "===== [闸门 0.5] 还原存档时间戳（锚点自检的前提）====="
# 存档若经过 git/打包传输，文件 mtime 会被重写 ⟹ reeval 的【锚点自检】会因"sidecar 与 ckpt 不同步"
# 而**整条跳过**。锚点是防"评估配置配错→静默给出错数"的关键防线（`03` L192 那个洞），不能丢。
# 这里按 sidecar 自己记录的 zip_mtime 还原（**先硬校 zip_size 逐字节相同**，不同则跳过不动、并报警）。
# 原生环境本来就不会有这个问题，跑这步无副作用。
"$PY" - "$ROOT" <<'PYEOF'
import json, os, sys, glob
root = sys.argv[1]
ok = bad = miss = 0
# 🆕 `03` L243：**分段副本目录也要收**。最佳存档那一趟评的就是 segments/ 里的副本，
#    它们的 mtime 一样会被传输重写 ⟹ 不还原则锚点自检整条跳过 = 丢掉防"评估配错"的关键防线。
cands = (glob.glob(os.path.join(root, "*", "checkpoints", "*.progress.json"))
         + glob.glob(os.path.join(root, "*", "*", "checkpoints", "*.progress.json"))
         + glob.glob(os.path.join(root, "*", "checkpoints", "segments", "*.progress.json"))
         + glob.glob(os.path.join(root, "*", "*", "checkpoints", "segments", "*.progress.json")))
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
  REEVAL_POOL=official REEVAL_CKPTS="$LIST" REEVAL_TRAJ_KEYS="$TRAJ_KEYS" \
  REEVAL_KEEP_PER="${REEVAL_KEEP_PER:-1}" \
  REEVAL_OUT="$OUT_DIR/g$g.json" REEVAL_TRAJ_OUT="$OUT_DIR/g${g}_traj.json" \
    "$PY" -B "$CODE_DIR/tests/reeval_official.py" > "$OUT_DIR/g$g.log" 2>&1 &
done
# 🔴 `03` L243-§7③：**无盾臂必须单独一个进程组**（`run_step4e.py:955` 的 shield 取进程级模块变量、
#    不从存档回读 ⟹ 混进上面任一组会把**整组的盾都关掉**，而且不会报错）。
#    组号接着上面往下排 ⟹ 产物仍是 g*.json ⟹ 闸门 2 的同分母核对会把它一起核进去。
if [ "$N_UNS" -gt 0 ]; then
  UG=$NGROUP
  ULIST=$(printf "%s," "${ARMS_NOSHIELD[@]}"); ULIST="${ULIST%,}"
  echo "  组 $UG【无盾·单独进程·STEP4E_CONTINUOUS_SHIELD=0】: $N_UNS 条"
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  STEP4E_CONTINUOUS_SHIELD=0 \
  STEP4E_SDIR="$ROOT/scenarios" STEP4E_CODE_DIR="$CODE_DIR" \
  REEVAL_MANIFEST_DIRS="$ROOT/balanced_pool" REEVAL_CKDIRS="$CKDIRS" \
  REEVAL_POOL=official REEVAL_CKPTS="$ULIST" REEVAL_TRAJ_KEYS="$TRAJ_KEYS" \
  REEVAL_KEEP_PER="${REEVAL_KEEP_PER:-1}" \
  REEVAL_OUT="$OUT_DIR/g$UG.json" REEVAL_TRAJ_OUT="$OUT_DIR/g${UG}_traj.json" \
    "$PY" -B "$CODE_DIR/tests/reeval_official.py" > "$OUT_DIR/g$UG.log" 2>&1 &
fi
wait
echo "  全部组结束"

echo "===== [闸门 2] 同分母核对 + 合并 ====="
"$PY" - "$OUT_DIR" "$N_ALL" <<'PYEOF'
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
# 🔴 分母**不再写死**（`03` L240）：以往写死 563，是因为所有臂都在小集上训、与官方测试 600 撞了
#   23 训练 + 14 验证 = 37 个场景。正式实验全部改在官方 1300 上训，与测试 600 **零交集** ⟹ 分母 = 600。
#   ⟹ 由环境变量 REEVAL_EXPECT_STRICT 显式声明本趟期望多少；**必须显式**，不许"看到多少算多少"。
_EXPECT = int(os.environ.get("REEVAL_EXPECT_STRICT", "0"))
print(f"  ✅ 各组 strict 键列表逐位相同 · n = {len(ref)}"
      + (f"（本趟声明期望 {_EXPECT}）" if _EXPECT else "（⚠️ 未声明期望值，只查了组间一致）"))
if _EXPECT and len(ref) != _EXPECT:
    raise SystemExit(f"🔒 strict 分母 = {len(ref)} ≠ 声明的 {_EXPECT} ⟹ 口径不对"
                     "（同趟里混进了别的训练集的臂？），别信这趟数字。")
if not _EXPECT:
    print("  ⚠️ 没设 REEVAL_EXPECT_STRICT ⟹ 分母没有被守卫。正式实验必须设"
          "（小集口径 563 / 官方 1300 口径 600）。")
print(f"  ✅ 合并得 {len(merged)} 条臂（期望 {n_expect}）")
if len(merged) != n_expect:
    print(f"  ⚠️ 条数对不上，缺的臂：看各组 g*.log 里的 ❌")
json.dump(merged, open(os.path.join(out_dir, "all.json"), "w", encoding="utf-8"), ensure_ascii=False)
print(f"  → {os.path.join(out_dir, 'all.json')}")
PYEOF

echo
echo "===== 完成。把整个 $OUT_DIR 目录回传即可 ====="
