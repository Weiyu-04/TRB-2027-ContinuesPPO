#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════════════════
# 对标论文臂（Discrete-safe）**重训 5 颗种子 · 开分段存档**  ——  `03` L236-D① / `Paper/正式实验` §4①
#
#   臂  `_dsSegS$S`   = 逐字复刻 0702 那趟的 Discrete-safe 配方，**唯一差别 = 开 STEP4E_KEEP_SEGMENTS=1**
#
# 【为什么必须做】`03` L236-B 实测：全部 46 条臂里**只有** `Discrete-safe s0` 出现"练好了又崩回去"
#   （训练期 20→58→88→98→95→98→**100**→88→22→**5**，峰值在第 7 段），而我们评的是**末段存档**
#   ⟹ 评到了它**最差的时刻**（strict 563 = 3.02%）。灵敏度：s0 按峰值附近算，对标均值 48.63 → 66~68，
#   我们的领先 **+43.5 → +24~26 点**。而分段存档是**覆盖式**的 ⟹ 那个好存档**已经没了、只能重训**。
#   🔴 **做这件事会让我们自己的数字变难看** —— 做它是为了诚实（审稿人必问"末段存档还是最好存档"）。
#
# 【顺带白拿一个检查】同种子 + 同配方重训，可以直接验**训练是不是逐位可复现**：
#   新 trend 若与 0702 那条逐段相同 ⟹ 全项目所有"同种子配对"比较的地基被坐实；
#   若不同 ⟹ 是个必须知道的坏消息（那所有配对结论都要加噪声带）。**两种结果都有价值。**
#
# 【🔴 绝不能直接改 run_strided.sh / 复用旧 TAG 跑】旧 TAG `discStdW0_s$S` 一旦复用，
#   run_step4e 会**直接覆盖**既有 `Discrete-safe_s{S}_discStdW0_s{S}.zip` —— 那是头条表里对标论文那条臂的
#   全部存档，覆盖是**静默**的、事后无法从存档看出来。⟹ 独立脚本 + 独立 TAG `_dsSegS$S`，闸门 0.6 硬查。
#
# 用法：  bash run_ds_reseg.sh [并发上限]        # 默认种子 0 1 2 3 4；并发默认 3（每 run NENVS=8 ⟹ 24 核）
#         SEEDS="0 1" bash run_ds_reseg.sh 2
# ══════════════════════════════════════════════════════════════════════════════════════════
set -uo pipefail

CODE_DIR="/root/trb/代码"
[ -d "$CODE_DIR" ] || { echo "❌ CODE_DIR 不存在：$CODE_DIR"; exit 1; }
RES_DIR="$(cd "$CODE_DIR/.." && pwd)/结果"
PY="/root/miniconda3/bin/python"
MANIFEST="$HOME/trb/balanced_pool/manifest_hocr_200.json"    # ← 逐字 = 0702 那趟（dataset: manifest_hocr_200.json）
BALANCED="$HOME/trb/balanced_pool"
SDIR="$HOME/trb/scenarios"

SEEDS="${SEEDS:-0 1 2 3 4}"
KMAX="${1:-3}"                                   # 并发（每 run NENVS=8）；重评在跑时用 3 = 24 核，给重评留 8 核

# ---- 配方：逐字复刻 0702 记录的 run_config（`结果0702-地基第1版-12:18/step4e_partial_discStdW0_s0.jsonl`）----
#   colregs_weight=1.0 由 PARTIES 表内置（Discrete-safe → 1.0），**不要**用 STEP4E_COLREGS_WEIGHT 覆盖（那是连续臂专用）。
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export STEP4E_SMOKE=0 STEP4E_NTOTAL=200 STEP4E_STEPS=5000000 STEP4E_NSEG=10 STEP4E_LOG_CURVES=1
export STEP4E_MANIFEST="$MANIFEST" STEP4E_BALANCED_DIR="$BALANCED" STEP4E_SDIR="$SDIR"
export STEP4E_ENT_START=0.01 STEP4E_ENT_END=0.01 STEP4E_ENT_FRAC=0.6 STEP4E_NENVS=8
export STEP4E_WELL_B=0 STEP4E_SHAPING_RADIUS=500 STEP4E_WELL_X=0 STEP4E_XTRACK_RADIUS=80
export STEP4E_PARTIES=Discrete-safe
export STEP4E_KEEP_SEGMENTS=1                    # ← 本脚本存在的**唯一理由**
export PY RES_DIR

echo "===== [闸门 0] 路径 + 代码已同步（含分段存档功能）====="
[ -d "$RES_DIR" ]  || { echo "❌ RES_DIR 不存在：$RES_DIR"; exit 1; }
[ -f "$MANIFEST" ] || { echo "❌ manifest 不存在：$MANIFEST"; exit 1; }
grep -q "STEP4E_KEEP_SEGMENTS" "$CODE_DIR/run_step4e.py" \
  || { echo "❌ run_step4e.py 里没有 STEP4E_KEEP_SEGMENTS ＝【代码没同步】→ 先同步整个 代码 文件夹再跑（否则白跑：不会有分段副本）"; exit 1; }
grep -q "def _archive_segment" "$CODE_DIR/run_step4e.py" \
  || { echo "❌ run_step4e.py 里没有 _archive_segment ＝旧版·先同步"; exit 1; }
echo "  ✅ 路径对 · 分段存档功能在"

echo "===== [闸门 0.6] 🔴 存档不撞名（**绝不能覆盖对标论文那条臂的既有存档**）====="
CLASH=0
for S in $SEEDS; do
  for T in "_dsSegS$S" ; do
    HIT=$(find "$(cd "$CODE_DIR/.." && pwd)" -path "*/checkpoints/Discrete-safe_s${S}${T}.zip" -print -quit 2>/dev/null)
    [ -z "$HIT" ] || { echo "  ❌ 已存在同名存档：$HIT"; CLASH=$((CLASH+1)); }
  done
  OLD=$(find "$(cd "$CODE_DIR/.." && pwd)" -path "*/checkpoints/Discrete-safe_s${S}_discStdW0_s${S}.zip" -print -quit 2>/dev/null)
  [ -n "$OLD" ] && echo "  · 旧存档在（本脚本不会碰它）：$(basename "$OLD")"
done
[ "$CLASH" -eq 0 ] || { echo "❌ $CLASH 个存档会被覆盖 → 换 TAG，别烧"; exit 1; }
echo "  ✅ $(echo $SEEDS|wc -w) 个目标存档名均不冲突，且旧存档不会被碰"

echo "===== [闸门 1] 预下载场景（小集 200·缓存则秒过）====="
STEP4E_SMOKE=0 STEP4E_DOWNLOAD_ONLY=1 STEP4E_SEEDS=0 STEP4E_TAG=_predl_dsseg "$PY" -B "$CODE_DIR/run_step4e.py" \
  || { echo "❌ 预下载失败（查网络）"; exit 1; }

echo "===== [闸门 2] 冒烟 1 次（~2min）·验【分段副本真落地 + 配方没漂】====="
cd "$CODE_DIR"
T=_smkDsSeg
SMK="$RES_DIR/step4e_partial${T}.jsonl"
SEGDIR="$RES_DIR/checkpoints/segments"
rm -f "$SMK"; rm -rf "$SEGDIR"
STEP4E_SMOKE=1 STEP4E_STEPS=8000 STEP4E_NSEG=2 STEP4E_SEEDS=0 STEP4E_TAG="$T" \
  "$PY" -B run_step4e.py > "$RES_DIR/_${T}.log" 2>&1 \
  || { echo "❌ 冒烟跑崩（看 $RES_DIR/_${T}.log）"; tail -20 "$RES_DIR/_${T}.log"; exit 1; }
grep -q '"keep_segments": true' "$SMK" || { echo "❌ 冒烟里 keep_segments 不是 true ⟹ 开关【静默没生效】·别烧全量"; exit 1; }
grep -q '"party": "Discrete-safe"' "$SMK" || { echo "❌ 冒烟臂不是 Discrete-safe·别烧全量"; exit 1; }
grep -q 'manifest_hocr_200' "$SMK"       || { echo "❌ 冒烟训练集不是 manifest_hocr_200 ⟹ 配方漂了·别烧全量"; exit 1; }
# 🔴 最关键：分段副本必须真的躺在 segments/ 里，而且【段数 == NSEG】
NSEG_FILES=$(ls "$SEGDIR"/*.zip 2>/dev/null | wc -l)
[ "$NSEG_FILES" -eq 2 ] || { echo "❌ segments/ 里 .zip 有 $NSEG_FILES 份（应为 2 = NSEG）⟹ 分段存档没真落地·别烧全量"; ls -la "$SEGDIR" 2>&1 | head; exit 1; }
ls "$SEGDIR" | head -8
echo "  ✅ 冒烟全过：keep_segments=true + 臂对 + 训练集对 + segments/ 里 2 份副本"
rm -rf "$SEGDIR"                                  # 清掉冒烟的副本，免得和全量的混在一起

echo "===== [闸门 3] 起 $(echo $SEEDS|wc -w) 个 run（从零 5M · 并发≤$KMAX）====="
run_one () {
  local S="$1"
  local T="_dsSegS$S"
  STEP4E_SEEDS="$S" STEP4E_TAG="$T" "$PY" -B run_step4e.py > "$RES_DIR/${T}.log" 2>&1 \
    && echo "  [完] $T" || echo "  [⚠️失败] $T（看 $RES_DIR/${T}.log）"
}
for S in $SEEDS; do
  run_one "$S" &
  while [ "$(jobs -rp | wc -l)" -ge "$KMAX" ]; do sleep 20; done
done
wait
echo "===== 全部结束 ====="
echo "分段副本份数：$(ls "$RES_DIR"/checkpoints/segments/Discrete-safe_s*_dsSegS*.zip 2>/dev/null | wc -l)（应 = 种子数 × 10）"
echo
echo "🔴 跑完两件事（都零烧卡）："
echo "  ① 训练可复现性检查：python3 -B 代码/tests/check_ds_reproducible.py"
echo "     —— 新 trend 与 0702 那条逐段比。相同 ⟹ 训练确定性坐实；不同 ⟹ 所有配对结论要加噪声带。"
echo "  ② 存档选取口径要【对所有臂统一】才能报：分段副本只是让『最好存档』这个口径**变得可算**，"
echo "     并不等于可以只给对标论文换口径。定稿时一并拍（见 Paper/正式实验 README §2/§4①）。"
