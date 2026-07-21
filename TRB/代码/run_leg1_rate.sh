#!/usr/bin/env bash
# ============================================================================
# 第一条腿：治抖版(rate ON) × 10 种子 × 5M · 小场景库 hocr_200 · 带新健康仪表
# ----------------------------------------------------------------------------
# 目的：坐实"治抖版零学废"是真稳还是运气(10 种子·非 5)；同时新仪表量每个种子多指标健康。
# 配方 = 逐字对齐 0707 rateHOCRB（唯一开关：治抖 rate 权重设为 1.0）·全 5M=和离散同预算。
# 对照 = 复用现有 5 种子朴素版旧数据(rate=0)作粗对照（省一半算力）。
#
# 用法（服务器上·先同步【整个 代码 文件夹】到服务器）：
#   改下面 2 行路径 → bash run_leg1_rate.sh [并发上限K=10]
# 内置 3 道闸门：① 路径+代码新(有 speed_reversals=新仪表已同步) ② 场景预下载
#   ③ 冒烟 1 种子·grep 证实 rate_weight=1.0 真开 + 新仪表出现（防静默 off 白烧）→ 才放全量。
# ⚠️ 训练吃 CPU 不吃 GPU；10 种子 × n_envs=8 = 80 核；服务器能同时跑 10 个则一批~4-5h。
# 断点续：被杀了重跑同一条命令、自动跳过已完成种子（run_step4e 分段续跑）。
# ============================================================================
set -uo pipefail

# ==== 只改这 1 行（你服务器放 run_step4e.py 的目录）====
CODE_DIR="/root/trb/代码"
# =======================================
[ -d "$CODE_DIR" ] || { echo "❌ CODE_DIR 不存在：$CODE_DIR"; exit 1; }
RES_DIR="$(cd "$CODE_DIR/.." && pwd)/结果"       # run_step4e 恒写到 <代码>/../结果·自动派生·防路径不一致
PY="/root/miniconda3/bin/python"
MANIFEST="$HOME/trb/balanced_pool/manifest_hocr_200.json"
BALANCED="$HOME/trb/balanced_pool"
SDIR="$HOME/trb/scenarios"
SEEDS="0 1 2 3 4 5 6 7 8 9"                      # 10 种子
KMAX="${1:-10}"                                  # 并发上限（服务器一次能跑几个·默认 10）

# ---- 治抖版配方（逐字=0707 rateHOCRB·唯一治疗开关 RATE_W=1.0）----
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export STEP4E_SMOKE=0 STEP4E_NTOTAL=200 STEP4E_STEPS=5000000 STEP4E_NSEG=10 STEP4E_LOG_CURVES=1
export STEP4E_MANIFEST="$MANIFEST" STEP4E_BALANCED_DIR="$BALANCED" STEP4E_SDIR="$SDIR"
export STEP4E_WELL_B=200 STEP4E_SHAPING_RADIUS=500 STEP4E_WELL_X=200 STEP4E_XTRACK_RADIUS=80
export STEP4E_PARK_W=20 STEP4E_PARK_RADIUS=400 STEP4E_PARK_VTARGET=4
export STEP4E_RATE_W=1.0                          # ← 治抖 ON（唯一与朴素版的区别）
export STEP4E_CONTINUOUS_ALGO=ppo STEP4E_ENT_START=0.01 STEP4E_ENT_END=0.01 STEP4E_NENVS=8
export STEP4E_PARTIES=Continuous-safe
export PY RES_DIR

echo "===== [闸门 0] 路径 + 代码已同步（含新仪表 speed_reversals）====="
[ -d "$CODE_DIR" ] || { echo "❌ CODE_DIR 不存在：$CODE_DIR"; exit 1; }
[ -d "$RES_DIR" ]  || { echo "❌ RES_DIR 不存在：$RES_DIR"; exit 1; }
[ -f "$MANIFEST" ] || { echo "❌ manifest 不存在：$MANIFEST（改脚本 MANIFEST 行）"; exit 1; }
grep -q "speed_reversals" "$CODE_DIR/trb_env/evaluate.py" || { echo "❌ evaluate.py 无 speed_reversals=没同步新仪表→跑出来缺健康指标·先【同步整个 代码 文件夹】再跑"; exit 1; }
grep -q "STEP4E_RATE_W" "$CODE_DIR/run_step4e.py" || { echo "❌ run_step4e.py 无 STEP4E_RATE_W=旧版·先同步"; exit 1; }
cd "$CODE_DIR"
echo "  ✅ 路径对·新仪表在·rate 开关在"

echo "===== [闸门 1] 预下载场景（缓存则秒过）====="
# ⚠️ 须带诊断 TAG(_predl_ppo·含 ppo)：本脚本 export CONTINUOUS_ALGO=ppo + SMOKE=0 → run_step4e:273 诊断 TAG 闸门在模块加载即触发(先于 DOWNLOAD_ONLY 判断)·漏 TAG=预下载崩(2026-07-09 L172 修·L1 首烧抓)。DOWNLOAD_ONLY 早退不记录·TAG 无副作用。
STEP4E_SMOKE=0 STEP4E_DOWNLOAD_ONLY=1 STEP4E_SEEDS=0 STEP4E_TAG=_predl_ppo "$PY" -B run_step4e.py || { echo "❌ 预下载失败（查网络/gitlab）"; exit 1; }

echo "===== [闸门 2] 冒烟 1 种子（~1-2min）·验 rate 真开=1.0 + 新仪表出现（防静默白烧）====="
SMK="$RES_DIR/step4e_partial_L1smoke_ppo.jsonl"; rm -f "$SMK"
STEP4E_SMOKE=1 STEP4E_STEPS=8000 STEP4E_SEEDS=0 STEP4E_TAG=_L1smoke_ppo "$PY" -B run_step4e.py > "$RES_DIR/_L1smoke.log" 2>&1 || { echo "❌ 冒烟跑崩（看 结果/_L1smoke.log）"; exit 1; }
grep -q '"rate_weight": 1.0' "$SMK" || { echo "❌ 冒烟未见 rate_weight=1.0 → 治抖没真开·别烧全量（查代码同步/参数）·看 $SMK"; exit 1; }
grep -q 'speed_reversals' "$SMK"    || { echo "❌ 冒烟未见 speed_reversals → 新仪表没生效·别烧全量（同步整个代码文件夹）"; exit 1; }
echo "  ✅ 冒烟确认：rate_weight=1.0（治抖真开）+ speed_reversals（新仪表在）"

echo "===== [闸门 3] 起 10 种子全量 5M（并发≤$KMAX·每种子 ~4-5h）====="
run_one() {
  local S="$1"
  STEP4E_SEEDS="$S" STEP4E_TAG="_L1rateON_ppo_s$S" "$PY" -B run_step4e.py > "$RES_DIR/_L1rateON_ppo_s$S.log" 2>&1 \
    && echo "  [完] s$S" || echo "  [⚠️失败] s$S（看 结果/_L1rateON_ppo_s$S.log）"
}
export -f run_one
printf '%s\n' $SEEDS | xargs -P "$KMAX" -I{} bash -c 'run_one "$@"' _ {}

N=$(ls "$RES_DIR"/step4e_partial_L1rateON_ppo_s*.jsonl 2>/dev/null | wc -l | tr -d ' ')
echo "===== 完成：产出 $N/10 个种子结果 ====="
echo "把 结果/step4e_partial_L1rateON_ppo_s*.jsonl 传回本地给主窗口分析（多指标健康 + 稳不稳 + 收敛没）。"
echo "被杀/失败：重跑同一条命令自动续跑未完成种子。"
