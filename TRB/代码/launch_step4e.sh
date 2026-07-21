#!/usr/bin/env bash
# ============================================================================
# step4e 多核并行启动器（服务器上把 4方×N种子 任务并发跑，压 wall-clock）
# ----------------------------------------------------------------------------
# 用法：  bash 代码/launch_step4e.sh [最大并发任务数 K] [每任务进程数 NENVS] [种子,逗号]
# 例：    bash 代码/launch_step4e.sh 4 8 0,1,2
#         → 12 个 (方,种子) 任务[4方×3种子]、最多 4 个并发、每任务 8 核（离散）/1 核（连续 SAC）
#
# ⚠️ 吃 CPU 不吃 GPU；核心预算 ≈ K × NENVS。先跑 SMOKE 量 fps 再定 K/NENVS。
# 并行安全：各任务 flock 追加同一 partial、table3 原子写、下载 PID 隔离。
# 断点续：任务被杀，重跑同一条 launch 命令即跳过已完成、补未完成。
# 可用环境变量微调内层（如 STEP4E_NTOTAL/STEPS/NSEG/TAG）——会传到各任务。
# ============================================================================
set -u
# ⚡ 每进程限 1 线程（防 numpy/BLAS 在多核机上 8worker×N核 线程互抢核拖慢；脚本内也 setdefault，这里双保险）
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export STEP4E_PY="${STEP4E_PY:-python}"   # 默认用 PATH 里的 python（服务器 base 环境）；本地可 STEP4E_PY=/全路径 覆盖
HERE="$(cd "$(dirname "$0")" && pwd)"
export STEP4E_SCRIPT="$HERE/run_step4e.py"
export STEP4E_LOGDIR="$HERE/../结果/step4e_logs"

K="${1:-3}"                         # 最大并发任务数
export STEP4E_LAUNCH_NENVS="${2:-8}"   # 每任务 SubprocVecEnv 进程数（≈每任务占核）
SEEDS="${3:-0,1,2}"                 # 种子（逗号分隔）
export STEP4E_ALL_SEEDS="$SEEDS"    # 全种子集 → 各单方/单种子子进程 run_metadata 记全 seeds（非单分片，L53 修）
# 慢的先起 → 不落关键路径末尾、压 wall-clock。Continuous-safe(SAC·n_envs=1·off-policy) 单 env 最慢(~9h/种子估)→最先起；
# 有盾 Discrete-safe 次之；无盾 Base/RR 快 ~3.3×、后面填空。
# ⚠️ Continuous-safe 内部用 N_ENVS_SAC(默认 1，run_step4e.py)、【不吃】本 launcher 的 NENVS → 只占 1 核（NENVS 是给离散 PPO 的）；
#    核预算：离散三方 ≈ NENVS 核/任务、连续臂 ≈ 1 核/任务。要给连续臂多核须 STEP4E_SAC_NENVS（但改 SAC 学习动态，03 L48）。
PARTIES=("Continuous-safe" "Discrete-safe" "Rule-reward" "Base")

mkdir -p "$STEP4E_LOGDIR"
echo "=== step4e 并行：并发上限 K=$K × 每任务 NENVS=$STEP4E_LAUNCH_NENVS 核 ≈ $((K * STEP4E_LAUNCH_NENVS)) 核；种子 $SEEDS ==="
echo "全量模式（STEP4E_SMOKE=0）；日志 → 结果/step4e_logs/；结果 → 结果/table3.txt"

# 1) 预下载全场景（单进程，避免多任务并发首下竞争；已缓存秒过）
echo "[1/3] 预下载场景 …"
STEP4E_SMOKE=0 STEP4E_DOWNLOAD_ONLY=1 STEP4E_SEEDS="$SEEDS" "$STEP4E_PY" -B "$STEP4E_SCRIPT" || {
  echo "❌ 预下载失败（查网络/gitlab）"; exit 1; }

# 2) 并发起 (party,seed) 任务（导出函数 + xargs -P 控并发，避免脆弱的嵌套引号）
run_one() {
  local party="$1" seed="$2"
  local safe="${party// /_}"
  local log="$STEP4E_LOGDIR/${safe}_s${seed}${STEP4E_TAG:-}.log"   # 带臂 tag → 各臂日志互不覆盖（03 L26）
  echo "  [起] $party seed=$seed → 结果/step4e_logs/${safe}_s${seed}${STEP4E_TAG:-}.log"
  if STEP4E_SMOKE=0 STEP4E_NENVS="$STEP4E_LAUNCH_NENVS" STEP4E_PARTIES="$party" STEP4E_SEEDS="$seed" \
       "$STEP4E_PY" -B "$STEP4E_SCRIPT" > "$log" 2>&1; then
    echo "  [完] $party seed=$seed"
  else
    echo "  [⚠️失败] $party seed=$seed（看 $log）"
  fi
}
export -f run_one

echo "[2/3] 并发训练（每个 (方,种子) 一个进程，并发≤$K）…"
for p in "${PARTIES[@]}"; do
  for s in ${SEEDS//,/ }; do
    printf '%s\t%s\n' "$p" "$s"
  done
done | xargs -P "$K" -L1 bash -c 'run_one "$1" "$2"' _

# 3) 聚合 → table3.txt
echo "[3/3] 聚合 Table III …"
STEP4E_SMOKE=0 STEP4E_AGG=1 STEP4E_SEEDS="$SEEDS" "$STEP4E_PY" -B "$STEP4E_SCRIPT"
echo "✅ 全部完成。把 结果/table3.txt 发回核对。被杀了就再敲同一条 launch 命令（自动续跑）。"
