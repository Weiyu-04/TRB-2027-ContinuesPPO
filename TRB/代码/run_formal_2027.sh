#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════════════════
# TRB 2027 **正式实验**（`03` L240 定稿方案 · `Paper/正式实验/README.md`）
#
#   9 条臂 × 10 颗种子（0-9）× 10.16M 步（20 段 × 507,904）· 官方 1300 训练集 · 分段存档全程开
#
# 用法：
#   SEEDS="0 1 2 3 4" bash run_formal_2027.sh 4        # A 机
#   SEEDS="5 6 7 8 9" bash run_formal_2027.sh 4        # B 机
#   ARMS="ours disc" SEEDS="0" bash run_formal_2027.sh 1    # 只跑某几条臂（调试/补跑）
#
# 🔴 **按种子分机器，绝不按臂分机器**：同种子配对比较是我们的主统计工具，必须落在同一台机器内。
#    按臂分机器 = 机器变成混淆变量，整套对比作废。
# 🔴 **评估另说**：跑完全部臂后，重评必须**一台机器一趟**跑完（`04 §2`）。
# ══════════════════════════════════════════════════════════════════════════════════════════
set -uo pipefail

CODE_DIR="/root/trb/代码"
[ -d "$CODE_DIR" ] || { echo "❌ CODE_DIR 不存在：$CODE_DIR"; exit 1; }
RES_DIR="$(cd "$CODE_DIR/.." && pwd)/结果"
PY="/root/miniconda3/bin/python"
MANIFEST="$HOME/trb/balanced_pool/manifest_official_1300.json"
BALANCED="$HOME/trb/balanced_pool"
SDIR="$HOME/trb/scenarios"

SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
ARMS="${ARMS:-ours disc base rr uns ush ab0 abB abG}"
KMAX="${1:-4}"                       # 每 run 8 环境 ⟹ 32 核最多 4 路（`04 §1` 定核纪律）

# ── 全臂共用（**任何一条臂都不许改这些**）──────────────────────────────────────
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export STEP4E_SMOKE=0
export STEP4E_STEPS=10000000 STEP4E_NSEG=20        # ⟹ 每段 507,904 步（与既有 5.08M 臂同网格）· 20 个存档候选
export STEP4E_NTOTAL=2000                          # manifest 模式下只影响非 manifest 路径；设 2000 保底不 striding
export STEP4E_MANIFEST="$MANIFEST" STEP4E_BALANCED_DIR="$BALANCED" STEP4E_SDIR="$SDIR"
export STEP4E_NENVS=8 STEP4E_ENT_START=0.01 STEP4E_ENT_END=0.01 STEP4E_ENT_FRAC=0.6
export STEP4E_LOG_CURVES=1                         # 🔴 默认是关的！离散臂漏了就永远没有内部曲线（A 类量）
export STEP4E_KEEP_SEGMENTS=1                      # 🔴 分段存档 —— "验证集挑最佳存档"这个定稿口径的唯一依据
export STEP4E_RECORD_EVERY=25000                   # 与既有 5M 臂同网格（默认会变成每 5 万步）
export PY RES_DIR

# ── 逐臂配方（**只列与共用项不同的**；空 = 全用默认）──────────────────────────
#   ⚠️ TAG 必须含 ppo（`run_step4e.py:405-410` 的 PPO 隔离闸），否则起跑即拦。
arm_cfg () {
  case "$1" in
    # ① 主角：两把钥匙 + 盾 + 连续专属塑形
    ours) echo "TAG=_F240oursPpoS PARTIES=Continuous-safe SHIELD=1 ACT=beta GW=symmetric SHAPE=full" ;;
    # ② 对标论文（离散 + 动作屏蔽）· ③④ 两条离散基线
    disc) echo "TAG=_F240discPpoS PARTIES=Discrete-safe  SHIELD=- ACT=-    GW=-         SHAPE=none" ;;
    base) echo "TAG=_F240basePpoS PARTIES=Base           SHIELD=- ACT=-    GW=-         SHAPE=none" ;;
    rr)   echo "TAG=_F240rrPpoS   PARTIES=Rule-reward    SHIELD=- ACT=-    GW=-         SHAPE=none" ;;
    # ⑤⑥ 孪生对：**逐字相同、只差盾** ⟹ 唯一干净回答"盾值多少"的一对
    #    极简配方（高斯/paper/零塑形）是无盾臂**唯一合法**形态（run_step4e.py:223/233/243 三道闸）
    #    且与离散 Base 的奖励口径一致 ⟹ ⑤vs③ 干净回答"离散换连续值多少"
    uns)  echo "TAG=_F240unsPpoS  PARTIES=Continuous-safe SHIELD=0 ACT=gauss GW=paper   SHAPE=none" ;;
    ush)  echo "TAG=_F240ushPpoS  PARTIES=Continuous-safe SHIELD=1 ACT=gauss GW=paper   SHAPE=none" ;;
    # ⑦⑧⑨ 消融三臂（都带盾 + 全套塑形，只差两把钥匙的开关组合）
    ab0)  echo "TAG=_F240ab0PpoS  PARTIES=Continuous-safe SHIELD=1 ACT=gauss GW=paper   SHAPE=full" ;;
    abB)  echo "TAG=_F240abBPpoS  PARTIES=Continuous-safe SHIELD=1 ACT=beta  GW=paper   SHAPE=full" ;;
    abG)  echo "TAG=_F240abGPpoS  PARTIES=Continuous-safe SHIELD=1 ACT=gauss GW=symmetric SHAPE=full" ;;
    *) echo ""; ;;
  esac
}

arm_env () {   # 把 arm_cfg 的紧凑描述展开成 env 赋值串
  local cfg; cfg=$(arm_cfg "$1"); [ -n "$cfg" ] || { echo "❌ 未知臂 $1" >&2; return 1; }
  local TAG PARTIES SHIELD ACT GW SHAPE; eval "$cfg"
  local e="STEP4E_PARTIES=$PARTIES"
  [ "$PARTIES" = "Continuous-safe" ] && e="$e STEP4E_CONTINUOUS_ALGO=ppo"
  [ "$SHIELD" != "-" ] && e="$e STEP4E_CONTINUOUS_SHIELD=$SHIELD"
  [ "$ACT" != "-" ]    && e="$e STEP4E_ACT_DIST=$ACT"
  # gw_entry=paper 是默认；只在 symmetric 时显式设（设 paper 也可，但少一个变量少一处出错）
  [ "$GW" = "symmetric" ] && e="$e STEP4E_GW_ENTRY=symmetric"
  if [ "$SHAPE" = "full" ]; then     # 连续专属塑形（逐字 = 既有金标/C 臂配方）
    e="$e STEP4E_WELL_B=200 STEP4E_SHAPING_RADIUS=500 STEP4E_WELL_X=200 STEP4E_XTRACK_RADIUS=80"
    e="$e STEP4E_PARK_W=20 STEP4E_PARK_RADIUS=400 STEP4E_PARK_VTARGET=4 STEP4E_RATE_W=1.0"
  else                               # 零塑形：离散三臂的既有口径，也是无盾臂唯一合法形态
    e="$e STEP4E_WELL_B=0 STEP4E_SHAPING_RADIUS=500 STEP4E_WELL_X=0 STEP4E_XTRACK_RADIUS=80"
    e="$e STEP4E_PARK_W=0 STEP4E_RATE_W=0"
  fi
  echo "$e"
}
arm_tag () { local cfg; cfg=$(arm_cfg "$1"); local TAG PARTIES SHIELD ACT GW SHAPE; eval "$cfg"; echo "$TAG"; }
arm_party () { local cfg; cfg=$(arm_cfg "$1"); local TAG PARTIES SHIELD ACT GW SHAPE; eval "$cfg"; echo "$PARTIES"; }

N_RUN=$(( $(echo $ARMS|wc -w) * $(echo $SEEDS|wc -w) ))
echo "═══ TRB 2027 正式实验 ═══  臂=[$ARMS]  种子=[$SEEDS]  ⟹ $N_RUN 个 run · 并发≤$KMAX"
echo "    10.16M 步 · 20 段 · 官方 1300 训练集 · 分段存档开 · 内部曲线开"

echo "===== [闸门 0] 代码已同步（含本轮三项新功能）====="
for pat in "STEP4E_KEEP_SEGMENTS" "def _archive_segment" "class _RolloutStats" "STEP4E_ACT_DIST" "STEP4E_GW_ENTRY"; do
  grep -q "$pat" "$CODE_DIR/run_step4e.py" || { echo "❌ run_step4e.py 里没有【$pat】＝代码没同步 → 同步整个 代码 文件夹再跑"; exit 1; }
done
[ -f "$CODE_DIR/trb_env/usv_action_dist.py" ] || { echo "❌ 缺 trb_env/usv_action_dist.py"; exit 1; }
grep -q "gw_entry=self._sc.gw_entry" "$CODE_DIR/trb_env/usv_projection.py" || { echo "❌ usv_projection.py 是旧版"; exit 1; }
echo "  ✅ 三项新功能（分段存档 / rollout 采集 / 两把钥匙）都在"

echo "===== [闸门 0.2] 记录代码指纹（可复现性·`03` L240）====="
( cd "$CODE_DIR/.." && git rev-parse HEAD 2>/dev/null && git status --porcelain 2>/dev/null | head -5 ) \
  > "$RES_DIR/_formal_code_fingerprint.txt" 2>&1 || true
for f in run_step4e.py trb_env/usv_projection.py trb_env/usv_colregs.py trb_env/usv_continuous_shield.py trb_env/evaluate.py; do
  "$PY" -c "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest()[:16], sys.argv[1])" \
    "$CODE_DIR/$f" >> "$RES_DIR/_formal_code_fingerprint.txt"
done
cat "$RES_DIR/_formal_code_fingerprint.txt"; echo "  ✅ 指纹已落盘（论文方法节要写）"

echo "===== [闸门 0.3] 清单自洽（六项必须全 ✅）====="
"$PY" -B "$CODE_DIR/make_official_manifest.py" --check "$MANIFEST" || { echo "❌ 清单校验没过 → 别烧卡"; exit 1; }

echo "===== [闸门 0.6] 存档不撞名（$N_RUN 个目标名 vs 既有全部存档）====="
CLASH=0
for A in $ARMS; do
  T=$(arm_tag "$A"); P=$(arm_party "$A")
  for S in $SEEDS; do
    HIT=$(find "$(cd "$CODE_DIR/.." && pwd)" -path "*/checkpoints/${P}_s${S}${T}${S}.zip" -print -quit 2>/dev/null)
    [ -z "$HIT" ] || { echo "  ❌ 已存在：$HIT"; CLASH=$((CLASH+1)); }
  done
done
[ "$CLASH" -eq 0 ] || { echo "❌ $CLASH 个存档会被【静默覆盖】→ 换 TAG 或先归档，别烧"; exit 1; }
echo "  ✅ $N_RUN 个目标存档名均不冲突"

echo "===== [闸门 1] 预下载场景（官方 1400·缓存则秒过）====="
STEP4E_DOWNLOAD_ONLY=1 STEP4E_SEEDS=0 STEP4E_TAG=_predl_f240 "$PY" -B "$CODE_DIR/run_step4e.py" \
  || { echo "❌ 预下载失败（查网络）"; exit 1; }

echo "===== [闸门 2] 逐臂冒烟（每条臂 1 次·验配方真落地 + 分段存档真写出来）====="
cd "$CODE_DIR"
for A in $ARMS; do
  T="_smkF240$A"; SMK="$RES_DIR/step4e_partial${T}.jsonl"; SEGDIR="$RES_DIR/checkpoints/segments"
  rm -f "$SMK"; rm -f "$SEGDIR"/*"${T}"@s* 2>/dev/null
  # shellcheck disable=SC2046
  env $(arm_env "$A") STEP4E_SMOKE=1 STEP4E_STEPS=8000 STEP4E_NSEG=2 STEP4E_SEEDS=0 STEP4E_TAG="$T" \
    "$PY" -B run_step4e.py > "$RES_DIR/_${T}.log" 2>&1 \
    || { echo "  ❌ 臂 $A 冒烟跑崩（看 $RES_DIR/_${T}.log）"; tail -15 "$RES_DIR/_${T}.log"; exit 1; }
  # 🔴 三样必须在 jsonl 里看得见（防"设了开关其实没生效"，`03` L192 那族坑）
  grep -q '"keep_segments": true' "$SMK" || { echo "  ❌ 臂 $A：keep_segments 没生效"; exit 1; }
  grep -q 'official_1300'         "$SMK" || { echo "  ❌ 臂 $A：训练集不是官方 1300"; exit 1; }
  grep -q '"roll_steps"'          "$SMK" || { echo "  ❌ 臂 $A：rollout 采集没生效（A 类量会全丢）"; exit 1; }
  # 🔴 最关键：分段副本必须**真的躺在磁盘上**，且份数 == NSEG。这条至今从没在真实 run 里验证过。
  NZ=$(ls "$SEGDIR"/*"${T}"@s*.zip 2>/dev/null | wc -l)
  [ "$NZ" -eq 2 ] || { echo "  ❌ 臂 $A：segments/ 里 .zip 有 $NZ 份（应 2 = NSEG）⟹ 分段存档没落地，别烧全量"; exit 1; }
  rm -f "$SEGDIR"/*"${T}"@s*                    # 只清本冒烟 TAG 的（**绝不整目录删**）
  echo "  ✅ 臂 $A：配方对 + 分段存档真落地 + rollout 采集在"
done

echo "===== [闸门 3] 起 $N_RUN 个 run ====="
run_one () {
  local A="$1" S="$2"; local T; T="$(arm_tag "$A")$S"
  # shellcheck disable=SC2046
  env $(arm_env "$A") STEP4E_SEEDS="$S" STEP4E_TAG="$T" "$PY" -B run_step4e.py > "$RES_DIR/${T}.log" 2>&1 \
    && echo "  [完] $T" || echo "  [⚠️失败] $T（看 $RES_DIR/${T}.log）"
}
for S in $SEEDS; do          # 外层种子、内层臂 ⟹ 同一颗种子的各臂时间上靠近，机器状态更接近
  for A in $ARMS; do
    run_one "$A" "$S" &
    while [ "$(jobs -rp | wc -l)" -ge "$KMAX" ]; do sleep 20; done
  done
done
wait
echo "===== 全部结束 ====="
echo "存档：$(ls "$RES_DIR"/checkpoints/*_F240*.zip 2>/dev/null | wc -l) 个 · 分段副本：$(ls "$RES_DIR"/checkpoints/segments/*_F240*.zip 2>/dev/null | wc -l) 个（应 = run 数 × 20）"
echo
echo "🔴 跑完的次序（别跳步）："
echo "  ① python3 -B 代码/tests/select_best_ckpt.py 结果  —— 用验证集挑最佳存档，产出可复算的清单"
echo "  ② 同趟重评【两趟】：最佳存档一趟 + 末段存档一趟（论文两版都报·`03` L236-A 的选择偏倚）"
echo "     REEVAL_EXPECT_STRICT=600 —— 全部臂都在官方 1300 上训，与测试 600 零交集 ⟹ 分母是 600 不是 563"
echo "  ③ 🔴 无盾臂（uns）**必须单独一个进程评**：replay 时 shield 取的是进程级环境变量、不从存档回读"
echo "     （run_step4e.py:955）⟹ 混在同组里会把整组的盾都关掉。见 04 §2。"
