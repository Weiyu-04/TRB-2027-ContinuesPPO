#!/usr/bin/env bash
# 位置-only 训练测试（`03` L185·user 2026-07-13）——验"去朝向硬门 → 崩种子不再被逼绕圈、能学会"。
# 假设：崩种子的"高速绕圈"是被严格朝向门(进区却对不上朝向→回合不结束→逼继续机动)诱发的；
#       改成【1_goal 只判位置到达目标区域】(忠实原文字面 "reached the goal area") → 进区即终止 → 绕圈消失 → 崩种子学得会。
# 配方 = 逐字对齐 golden L1rateON（well_B200/xtrack200/park20/rate1.0/ppo/ent0.01/5M/NSEG10）+ 唯一开关 STEP4E_GOAL_IGNORE_ORIENT=1 + distinct TAG。
# 评测同配置(run_step4e eval fac 同传)→里程碑"到达%"=位置-only到达率；final_per 仍记 in_box_aligned_steps=位置+朝向严版→两指标都可报。
# 测试种子 = 崩 s5/s6 + 健康对照 s2（看 s2 位置-only 到达率是否≈严格~65%·崩种子是否追上）。
# 用法(服务器·先同步整个代码文件夹): bash 代码/run_posonly.sh 5    # 自动后台·结果/posonly_s5/train.log
#   32 独占核可并行: 分别 bash 代码/run_posonly.sh 5 / 6 / 2 (3种子×8env=24核·宽松)。
#   ⚠️看里程碑早停:NSEG=10=每50万步打一次到达%·若崩种子 s5/s6 到 ~2-3M 已明显爬到健康位→假设坐实·可 Ctrl-C/pkill 早停不必等满 5M。
set -u
PY=/root/miniconda3/bin/python    # 服务器 AutoDL·绝对路径（本机测试改 /opt/miniconda3/envs/trb/bin/python）
ROOT="$(cd "$(dirname "$0")/.." && pwd)"        # =~/trb
SELF="$ROOT/代码/run_posonly.sh"

run_one() {
  local S="$1"
  local TAG="_posonly_ppo_s${S}"
  local OUT="$ROOT/结果/posonly_s${S}"
  local CK="$ROOT/结果/checkpoints/Continuous-safe_s${S}${TAG}"
  cd "$ROOT"
  echo "▶ s${S} 位置-only 训练开始 · GOAL_IGNORE_ORIENT=1 · 5M步 · TAG=${TAG} · $(date)"
  # 预检:代码同步了吗(GOAL_IGNORE_ORIENT 开关在)
  grep -q "STEP4E_GOAL_IGNORE_ORIENT" "$ROOT/代码/run_step4e.py" || { echo "❌ run_step4e.py 无 STEP4E_GOAL_IGNORE_ORIENT=旧版·先同步整个代码文件夹再跑"; exit 1; }
  env \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    STEP4E_SMOKE=0 STEP4E_NTOTAL=200 STEP4E_STEPS=5000000 STEP4E_NSEG=10 STEP4E_LOG_CURVES=1 \
    STEP4E_MANIFEST="$ROOT/balanced_pool/manifest_hocr_200.json" \
    STEP4E_BALANCED_DIR="$ROOT/balanced_pool" STEP4E_SDIR="$ROOT/scenarios" \
    STEP4E_WELL_B=200 STEP4E_SHAPING_RADIUS=500 STEP4E_WELL_X=200 STEP4E_XTRACK_RADIUS=80 \
    STEP4E_PARK_W=20 STEP4E_PARK_RADIUS=400 STEP4E_PARK_VTARGET=4 \
    STEP4E_RATE_W=1.0 \
    STEP4E_CONTINUOUS_ALGO=ppo STEP4E_ENT_START=0.01 STEP4E_ENT_END=0.01 STEP4E_NENVS=8 \
    STEP4E_GOAL_IGNORE_ORIENT=1 \
    STEP4E_PARTIES=Continuous-safe STEP4E_SEEDS="$S" STEP4E_TAG="$TAG" \
    "$PY" -B "$ROOT/代码/run_step4e.py"
  echo "=== 训练结束·收产物 → $OUT · $(date) ==="
  mkdir -p "$OUT"
  cp -v "$CK.zip" "$CK""_vecnorm.pkl" "$CK.progress.json" "$ROOT/结果/run_metadata${TAG}.json" "$OUT/" 2>&1
  cp -v "$ROOT/结果/step4e_partial${TAG}.jsonl" "$OUT/" 2>&1 || true
  echo "✅ s${S} 全部完成 · $(date) · 文件夹 $OUT 自成一体·可传回本地判读"
}

if [ $# -ge 1 ]; then
  S="$1"
  OUT="$ROOT/结果/posonly_s${S}"
  mkdir -p "$OUT"
  if [ "${_POSONLY_BG:-}" != "1" ]; then
    _POSONLY_BG=1 nohup bash "$SELF" "$S" > "$OUT/train.log" 2>&1 &
    echo "✅ s${S} 已后台启动 (PID $!·关终端也不断)"
    echo "   文件夹: $OUT"
    echo "   看进度: tail -f $OUT/train.log   (NSEG=10=每50万步一条'到达%'里程碑·5M全量~6-7h)"
    echo "   ⚠️看里程碑早停:s5/s6 到达%到~2-3M明显爬健康→假设坐实·可 pkill -9 -f run_step4e 早停·不必等满。"
    exit 0
  fi
  run_one "$S"
else
  echo "用法: bash 代码/run_posonly.sh <seed>   (自动后台·结果/posonly_s<seed>/train.log)"
  echo "  测试种子: 5 6(崩) + 2(健康对照)。32核可3个并行。看里程碑到达%(=位置-only)崩种子是否追平健康。"
  echo "  例: bash 代码/run_posonly.sh 5"
fi
