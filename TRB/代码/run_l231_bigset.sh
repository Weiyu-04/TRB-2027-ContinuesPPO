#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════════════════
# L232 大集探针（**新配方重开**）：C 配方（Beta + 对称让路入口）+ **官方 1300 训练集** · **从零 5M**。
#
#   臂 D  `_D232bigCppoS$S`   STEP4E_ACT_DIST=beta + STEP4E_GW_ENTRY=symmetric + manifest_official_1300
#
# 【为什么重开】0727 的大集判输（同种子配对中位掉 11.55pt·`03` L226/L231）是在**旧配方**下判的，
# 诊断成因 = "250 万步摊到 1300 个场景 ⟹ 每场景访问次数只剩 1/14 = 欠训"。而 C 配方把学习速度提了
# 一个量级（从零 5.08M 到达 92.11 vs 旧配方同条件 58.61·`03` L232-A）⟹ **那条"欠训"的理由很可能不成立了**。
#
# 【与小集 C 臂的关系 = 单变量】本脚本配方**逐字 = run_l231_arms.sh 的 C 臂**，
# 唯一差别就是训练集（manifest_hocr_200 → manifest_official_1300）+ NTOTAL + TAG。
#   ⟹ 判读 = 同种子配对：`_D232bigCppoS$S`  vs  `_C231bothPpoS$S`（都从零 5.08M·同配方·同步网格）。
#   ⚠️ 诚实口径：换官方 1300 时**场景数量（94→1300）与相遇类型配比（1:1 → 36%:64%）同时变**，不是纯单变量，
#      写作时必须如实说（`03` L223 已记）。
#
# 【为什么必须用清单、不能"不设 STEP4E_MANIFEST 走默认官方口径"】（`03` L223-A·这条救回过一趟白烧）
# 默认模式**本来就是**官方 1400/600，但那样 config_sig.dataset 会记成 "strided"，而 tests/reeval_official.py
# 有一道 fail-closed 闸专拦它（分不清 strided-200 还是 strided-2000 ⟹ 泄漏剔不干净）⟹ **跑完了评不了**。
# 用自描述清单就没这个问题，训练代码一行不用改。
#
# 【🔴 为什么另起一个脚本、而不是改 run_l231_arms.sh】那个脚本里 MANIFEST/NTOTAL/TAG **全写死不吃环境变量**：
#   · 只改 MANIFEST 不改 TAG ⟹ **种子 1/3/4 的小集 C 臂存档会被大集的直接覆盖**（头条表 10 颗种子当场少 3 颗）；
#   · 什么都不改直接 ARMS=C 跑 ⟹ 只是把小集 C 臂重跑一遍（数据集根本没换）。
#   ⟹ 独立脚本 + 独立 TAG 前缀 `_D232big`，与既有 56 条臂**任何一条都不同名**（起飞前闸门 0.6 会硬查）。
#
# 【清单本身的口径】官方 1400 训练 / 600 测试的官方划分不动；把官方那 1400 再切成
#   **1300 训练 + 100 验证**（训练期评估只用这 100 个）⟹ 训练与训练期评估**全程不碰**报数用的官方 600。
#   实测校验：训练 1300 + 验证 100 = 1400 · 训练∩官方600 = 0 · 验证∩官方600 = 0 · 对遇验证 36 / 交叉验证 64。
#
# 用法：  bash run_l231_bigset.sh [并发上限]                 # 默认种子 1 3 4（= 与 0727 旧配方大集探针同种子）
#         SEEDS="0 2 5" bash run_l231_bigset.sh 3
# ══════════════════════════════════════════════════════════════════════════════════════════
set -uo pipefail

CODE_DIR="/root/trb/代码"
[ -d "$CODE_DIR" ] || { echo "❌ CODE_DIR 不存在：$CODE_DIR"; exit 1; }
RES_DIR="$(cd "$CODE_DIR/.." && pwd)/结果"       # run_step4e 恒写到 <代码>/../结果·自动派生·防路径不一致
PY="/root/miniconda3/bin/python"
MANIFEST="$HOME/trb/balanced_pool/manifest_official_1300.json"   # ← 差别①：官方 1400 里切 1300训/100验
BALANCED="$HOME/trb/balanced_pool"
SDIR="$HOME/trb/scenarios"

SEEDS="${SEEDS:-1 3 4}"                          # 默认 3 颗探针 = 与 0727 旧配方大集同种子（多一条纵向对照）
KMAX="${1:-3}"                                   # 并发上限（每 run NENVS=8 ⟹ 3 路 = 24 核）

# ---- 配方：逐字 = run_l231_arms.sh 的 C 臂（= 金标 run_leg1_rate.sh + 两个开关）----
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
# 🔴 NTOTAL=2000（差别②）：本脚本走 manifest 模式，NTOTAL 只影响**非 manifest 路径**；设 2000 是为了让
#    run_step4e 的 _pool_eff = None（POOL 2000 不 > n_total 2000）⟹ 万一有人把 MANIFEST 注掉，也仍落在官方口径
#    而不是悄悄 striding 抽样。与 run_warmstart_big.sh 同款（`03` L226-Q 复审补进清单的那条）。
export STEP4E_SMOKE=0 STEP4E_NTOTAL=2000 STEP4E_STEPS=5000000 STEP4E_NSEG=10 STEP4E_LOG_CURVES=1
#    STEPS/NSEG 逐字同 C 臂（5M / 10 段 ⟹ 每段 500k）：学习曲线能与 C 臂、与金标叠在同一张图上。
export STEP4E_MANIFEST="$MANIFEST" STEP4E_BALANCED_DIR="$BALANCED" STEP4E_SDIR="$SDIR"
export STEP4E_WELL_B=200 STEP4E_SHAPING_RADIUS=500 STEP4E_WELL_X=200 STEP4E_XTRACK_RADIUS=80
export STEP4E_PARK_W=20 STEP4E_PARK_RADIUS=400 STEP4E_PARK_VTARGET=4
export STEP4E_RATE_W=1.0                          # 治抖 ON（金标同款·**不能关**：Beta 的机制就是"让这个罚项终于有着力点"）
export STEP4E_CONTINUOUS_ALGO=ppo STEP4E_ENT_START=0.01 STEP4E_ENT_END=0.01 STEP4E_NENVS=8
export STEP4E_PARTIES=Continuous-safe
export STEP4E_ACT_DIST=beta STEP4E_GW_ENTRY=symmetric   # ← C 配方的两把钥匙（`03` L231/L232）
export PY RES_DIR

echo "===== [闸门 0] 路径 + 代码已同步（含 L231 新开关）====="
[ -d "$RES_DIR" ]  || { echo "❌ RES_DIR 不存在：$RES_DIR"; exit 1; }
[ -f "$MANIFEST" ] || { echo "❌ manifest 不存在：$MANIFEST → 先同步 balanced_pool/manifest_official_1300.json"; exit 1; }
[ -f "$CODE_DIR/trb_env/usv_action_dist.py" ] || { echo "❌ 缺 trb_env/usv_action_dist.py＝没同步新模块 → 先【同步整个 代码 文件夹】再跑"; exit 1; }
grep -q "STEP4E_ACT_DIST" "$CODE_DIR/run_step4e.py"  || { echo "❌ run_step4e.py 无 STEP4E_ACT_DIST＝旧版·先同步"; exit 1; }
grep -q "STEP4E_GW_ENTRY" "$CODE_DIR/run_step4e.py"  || { echo "❌ run_step4e.py 无 STEP4E_GW_ENTRY＝旧版·先同步"; exit 1; }
grep -q "gw_entry" "$CODE_DIR/trb_env/usv_colregs.py" || { echo "❌ usv_colregs.py 无 gw_entry＝旧版·先同步"; exit 1; }
grep -q "gw_entry=self._sc.gw_entry" "$CODE_DIR/trb_env/usv_projection.py" || { echo "❌ usv_projection.py 前瞻未继承档位＝旧版·先同步"; exit 1; }
echo "  ✅ 路径对 · 新模块在 · 两个开关都在 · 前瞻一致性补丁在"

echo "===== [闸门 0.3] 清单自洽（六项必须全 ✅）====="
"$PY" -B "$CODE_DIR/make_official_manifest.py" --check "$MANIFEST" \
  || { echo "❌ 清单校验没过 → 别烧卡"; exit 1; }

echo "===== [闸门 0.6] 存档不撞名（防覆盖已有 56 条臂）====="
# 🔴 这道闸是本脚本存在的理由：一旦 TAG 与既有臂重名，run_step4e 会**直接覆盖** checkpoints/*.zip，
#    而覆盖是静默的、事后无法从存档里看出来 ⟹ 头条表会凭空少几颗种子。起飞前硬查。
CLASH=0
for S in $SEEDS; do
  T="_D232bigCppoS$S"
  HIT=$(find "$(cd "$CODE_DIR/.." && pwd)" -path "*/checkpoints/Continuous-safe_s${S}${T}.zip" -print -quit 2>/dev/null)
  [ -z "$HIT" ] || { echo "  ❌ 已存在同名存档：$HIT ⟹ 再跑会覆盖它"; CLASH=$((CLASH+1)); }
done
[ "$CLASH" -eq 0 ] || { echo "❌ $CLASH 个存档会被覆盖 → 换 TAG 或先归档，别烧"; exit 1; }
echo "  ✅ $(echo $SEEDS|wc -w) 个目标存档名均不与既有臂冲突"

echo "===== [闸门 1] 预下载场景（官方 1400 全量·缓存则秒过）====="
STEP4E_SMOKE=0 STEP4E_DOWNLOAD_ONLY=1 STEP4E_SEEDS=0 STEP4E_TAG=_predl_l232bigppo "$PY" -B "$CODE_DIR/run_step4e.py" \
  || { echo "❌ 预下载失败（查网络）"; exit 1; }

echo "===== [闸门 2] 冒烟 1 次（~2-3min）·验【两个开关 + 大集清单】三样都真落地 ====="
cd "$CODE_DIR"
T=_smkD232bigppo
SMK="$RES_DIR/step4e_partial${T}.jsonl"          # ⚠️ TAG 自带前导下划线（命名是 step4e_partial<TAG>.jsonl·别多写下划线）
rm -f "$SMK"
STEP4E_SMOKE=1 STEP4E_STEPS=8000 STEP4E_NSEG=1 STEP4E_SEEDS=0 STEP4E_TAG="$T" \
  "$PY" -B run_step4e.py > "$RES_DIR/_${T}.log" 2>&1 \
  || { echo "❌ 冒烟跑崩（看 $RES_DIR/_${T}.log）"; tail -20 "$RES_DIR/_${T}.log"; exit 1; }
grep -q '"act_dist": "beta"'      "$SMK" || { echo "❌ 冒烟没见 act_dist=beta → 开关【静默没生效】·别烧全量"; exit 1; }
grep -q '"gw_entry": "symmetric"' "$SMK" || { echo "❌ 冒烟没见 gw_entry=symmetric → 开关【静默没生效】·别烧全量"; exit 1; }
grep -q '"rate_weight": 1.0'      "$SMK" || { echo "❌ 冒烟未见 rate_weight=1.0 → 治抖没真开（Beta 的机制就靠它）·别烧全量"; exit 1; }
# 🔴 最关键的一条：数据集真的换了没有。dataset 记的是清单名 ⟹ 没换就会看见 hocr_200 / strided。
grep -q 'official_1300' "$SMK" || {
  echo "❌ 冒烟里没见 official_1300 ⟹ **训练集根本没换**（那就等于把小集 C 臂重跑一遍）·别烧全量"
  echo "   实际记到的 dataset ↓"; grep -o '"dataset": *"[^"]*"' "$SMK" | head -3; exit 1; }
echo "  ✅ 冒烟全过：beta + symmetric + 治抖ON + 训练集=官方1300"

echo "===== [闸门 3] 起 $(echo $SEEDS|wc -w) 个 run（从零 5M · 并发≤$KMAX）====="
run_one () {
  local S="$1"
  local T="_D232bigCppoS$S"
  STEP4E_SEEDS="$S" STEP4E_TAG="$T" "$PY" -B run_step4e.py > "$RES_DIR/${T}.log" 2>&1 \
    && echo "  [完] $T" || echo "  [⚠️失败] $T（看 $RES_DIR/${T}.log）"
}
for S in $SEEDS; do
  run_one "$S" &
  while [ "$(jobs -rp | wc -l)" -ge "$KMAX" ]; do sleep 20; done
done
wait
echo "===== 全部结束 ====="
ls -la "$RES_DIR"/checkpoints/Continuous-safe_s*_D232big*.zip 2>/dev/null | wc -l
echo
echo "🔴 评估纪律：这条臂自己【不贡献泄漏】，单独评会得 strict 600、既有 56 臂是 563 ⟹ 600≠563，"
echo "   单独评了把数字塞进主表就是错的。必须把它加进 代码/tests/run_reeval_all.sh 的 ARMS 数组，"
echo "   与既有 56 条臂【同一趟】评（同趟取泄漏并集 ⟹ 仍是 563 ⟹ 同分母可比）。"
