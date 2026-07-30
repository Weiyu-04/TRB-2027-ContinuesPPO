#!/bin/bash
# 外部几何基线正式一趟（官方分母 600）—— 三个方法并行，各占一核
cd /home/user/TRB-2027-ContinuesPPO/TRB
SCRATCH=/tmp/claude-0/-home-user-TRB-2027-ContinuesPPO/c08a189c-71ba-54a8-929d-9cbfd3a0d5ca/scratchpad
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export BASELINE_MANIFEST_DIRS="$PWD/balanced_pool" REEVAL_MANIFEST_DIRS="$PWD/balanced_pool"
export STEP4E_SDIR="$PWD/scenarios" REEVAL_SDIR="$PWD/scenarios"
export BASELINE_LEAK_MANIFESTS=manifest_official_1300.json   # 🔴 官方 1300 训练集 ⟹ 泄漏 0 ⟹ 分母 strict 600
export BASELINE_BOX=rl,full BASELINE_TUNE_SRC=train
export BASELINE_LEAK_ACK=1   # 🔴 分母从旧口径 563 换成正式实验的 600，脚本要求显式放行 + 论文写清口径
for M in vo cbf pd; do
  BASELINE_METHODS=$M BASELINE_OUT="$SCRATCH/bl/baselines_${M}.json" \
    python3 -B 代码/m1_dock_wip/run_baselines_official.py --run > "$SCRATCH/bl/${M}.log" 2>&1 &
done
wait
echo "===== 三个方法全部跑完 ====="
grep -h "strict（与四臂同分母）" "$SCRATCH"/bl/*.log | head -30
