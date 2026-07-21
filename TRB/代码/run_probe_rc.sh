#!/usr/bin/env bash
# 逆向课程【终端捕获探针】(方案C-B·`03` L182/L184·对抗审后定稿)——B 的 go/no-go。
# 问题:把船【固定生在近门 start_frac=0.2】短训,PPO 能否学会终端捕获?
#   学得会(崩种子近门 delta 追平【已过地板的】健康对照) → 崩是远场探索/坏盆地 → 退火课程可攻 → 烧全量。
#   学不会(崩种子近门仍绕圈·且健康对照过地板) → 终端捕获本身坏 → 课程救不了 → 转终端控制器 M1。
#   健康对照也趴(<~50%) → 1M 欠训/近门奖励教不会任何种子=inconclusive·非 GO 非 NO-GO(L184 健康地板守卫)。
# 配方 = 0710 L1rateON 逐字(well_B200/xtrack200/park20/rate1.0/ppo/ent0.01) + 【近门起点 frac=0.2·v=6】+ 短训 1M + distinct TAG。
#
# 🟢 本版改动(`03` L184·user 2026-07-13 要"打上就跑·文件夹里自动有 log"):
#   - 每个种子写进独立文件夹 结果/probeRC_s<seed>/,log 用【直接重定向】固定落 train.log(不靠会坏的内部 tee)。
#   - 单次调用【自动后台化】(nohup 自我 re-exec):打上就返回·关终端也不断·打印 tail 命令。
#   - 训练完把 checkpoint 三件 + metadata + partial jsonl 【收进同一文件夹】=自成一体·直接传回本地判读。
#   - ⚠️ 小服务器【一次只跑 1 个种子】(8 env 满速~80min);别并行(2+ 种子会超额抢核·各跑半速·反而更慢·L183)。
#
# 用法(服务器·同步代码后·从 ~/trb 或任意目录都行):
#   bash 代码/run_probe_rc.sh 2          # 跑健康对照 s2(自动后台·decisive)
#   跑完(或过几分钟)看进度: tail -f 结果/probeRC_s2/train.log
#   s2 跑完再单跑: bash 代码/run_probe_rc.sh 5   /   bash 代码/run_probe_rc.sh 6
set -u
PY=/root/miniconda3/bin/python    # 服务器 AutoDL·绝对路径（本机测试改 /opt/miniconda3/envs/trb/bin/python）
ROOT="$(cd "$(dirname "$0")/.." && pwd)"        # =~/trb(脚本在 代码/ 下·..=项目根)
SELF="$ROOT/代码/run_probe_rc.sh"

run_one() {
  # 真正干活:训练(输出天然进父级重定向的 train.log)→ 收产物进文件夹。
  local S="$1"
  local TAG="_probeRC_ppo_s${S}"
  local OUT="$ROOT/结果/probeRC_s${S}"
  local CK="$ROOT/结果/checkpoints/Continuous-safe_s${S}${TAG}"
  cd "$ROOT"
  echo "▶ s${S} 训练开始 · 近门 frac=0.2 v=6 · 1M步 · TAG=${TAG} · $(date)"
  env \
    STEP4E_SMOKE=0 STEP4E_NTOTAL=200 STEP4E_STEPS=1000000 STEP4E_NSEG=20 STEP4E_LOG_CURVES=1 \
    STEP4E_MANIFEST="$ROOT/balanced_pool/manifest_hocr_200.json" \
    STEP4E_BALANCED_DIR="$ROOT/balanced_pool" STEP4E_SDIR="$ROOT/scenarios" \
    STEP4E_WELL_B=200 STEP4E_SHAPING_RADIUS=500 STEP4E_WELL_X=200 STEP4E_XTRACK_RADIUS=80 \
    STEP4E_PARK_W=20 STEP4E_PARK_RADIUS=400 STEP4E_PARK_VTARGET=4 \
    STEP4E_RATE_W=1.0 \
    STEP4E_CONTINUOUS_ALGO=ppo STEP4E_ENT_START=0.01 STEP4E_ENT_END=0.01 \
    STEP4E_START_FRAC=0.2 STEP4E_START_V=6 \
    STEP4E_PARTIES=Continuous-safe STEP4E_SEEDS="$S" STEP4E_TAG="$TAG" \
    "$PY" -B "$ROOT/代码/run_step4e.py"
  echo "=== 训练结束·收产物 → $OUT · $(date) ==="
  mkdir -p "$OUT"
  cp -v "$CK.zip" "$CK""_vecnorm.pkl" "$CK.progress.json" "$ROOT/结果/run_metadata${TAG}.json" "$OUT/" 2>&1
  cp -v "$ROOT/结果/step4e_partial${TAG}.jsonl" "$OUT/" 2>&1 || true
  echo "✅ s${S} 全部完成 · $(date) · 文件夹 $OUT 自成一体(含 train.log + checkpoint 三件 + metadata)·可直接传回本地判读"
}

if [ $# -ge 1 ]; then
  S="$1"
  OUT="$ROOT/结果/probeRC_s${S}"
  mkdir -p "$OUT"
  if [ "${_PROBE_BG:-}" != "1" ]; then
    # 自动后台化:re-exec 自己·全部输出重定向进文件夹 train.log(固定路径·不依赖 tee/screen/locale)。
    _PROBE_BG=1 nohup bash "$SELF" "$S" > "$OUT/train.log" 2>&1 &
    echo "✅ s${S} 已后台启动 (PID $!·关终端也不断)"
    echo "   文件夹: $OUT"
    echo "   日志:   $OUT/train.log"
    echo "   看进度: tail -f $OUT/train.log     (约 4-5 分钟出第一条 '50000步 | 到达...' 里程碑)"
    echo "   ⚠️ 小服务器一次只跑 1 个种子·别再并行起第二个(会超额抢核变慢·L183)。等这个跑完(~80min)再跑下一个。"
    exit 0
  fi
  run_one "$S"          # 到这=已在后台的自我调用里·输出进父级 train.log
else
  echo "用法: bash 代码/run_probe_rc.sh <seed>     (自动后台·文件夹 结果/probeRC_s<seed>/ 里有 train.log)"
  echo "  ⚠️ 小服务器一次只跑 1 个(满速~80min)·别并行。决定性顺序: 先 2(健康对照)→ 再 5 / 6(崩种子)。"
  echo "  例: bash 代码/run_probe_rc.sh 2"
  echo ""
  echo "跑完判读(本机或服务器·CKPT_DIR 指文件夹·健康地板守卫已在脚本内):"
  echo "  CKPT_DIR=<各probeRC_s文件夹或结果/checkpoints> CKPT_TMPL='Continuous-safe_s{s}_probeRC_ppo_s{s}' \\"
  echo "  SEEDS='2 5 6' PROBE_FRAC=0.2 START_V=6 STEP4E_MANIFEST=... <python> 代码/m1_dock_wip/probe_capture_eval.py"
fi
