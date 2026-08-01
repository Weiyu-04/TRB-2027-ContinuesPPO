#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════════════════
# 正式实验【收尾重评】一条龙 —— 分阶段跑，每阶段可单独重来
#
#   本脚本**不含任何新逻辑**，只是把 `04 §八` 的命令按正确次序、填好正确的值调一遍，
#   免得手敲出错（终端里粘贴带中文注释的命令会被 zsh 当成参数，已踩过）。
#
# 用法（在 /root/trb 下，一步一步来，每步看完输出再进下一步）：
#     bash 结果/结果0801-重评流程/run_final_reeval.sh check     # 只读体检，几分钟
#     bash 结果/结果0801-重评流程/run_final_reeval.sh best      # 挑最佳存档，几秒
#     bash 结果/结果0801-重评流程/run_final_reeval.sh last      # 第一趟重评，约 1.2 小时
#     bash 结果/结果0801-重评流程/run_final_reeval.sh bestpass  # 第二趟重评，约 1.2 小时
#     bash 结果/结果0801-重评流程/run_final_reeval.sh traj      # 轨迹专趟（图 6 要用），约 1.2 小时
#
#   🔴 last 与 bestpass **必须同一台机器背靠背跑**（`04:384`、`run_reeval_all.sh:15`）——
#      连续臂的投影二次规划在不同机器/BLAS 上有 ±0.5 点级抖动，两趟拼不同机器 = 数字不可比。
#   🔴 一律放 screen 里跑：screen -dmS reeval bash 结果/结果0801-重评流程/run_final_reeval.sh last
# ══════════════════════════════════════════════════════════════════════════════════════════
set -uo pipefail

ROOT="${ROOT:-/root/trb}"
CODE_DIR="$ROOT/代码"
RES="$ROOT/结果"
PY="${PY:-/root/miniconda3/bin/python}"

#: 实际跑完的种子（仓库清点坐实：8 颗 × 9 臂 = 72 条 run）
SEEDS="${SEEDS:-0 1 2 3 4 5 8 9}"
ARMS_ALL="ours disc base rr uns ush ab0 abB abG"
STAGE="${1:-}"

cd "$ROOT" 2>/dev/null || { echo "❌ 进不去 $ROOT（用 ROOT=/你的路径 覆盖）"; exit 1; }
[ -d "$CODE_DIR" ] || { echo "❌ 找不到 $CODE_DIR"; exit 1; }

banner () { echo; echo "══════════ $* ══════════"; }

case "$STAGE" in

check)
  banner "第 0 步 · 代码指纹（必须与冻结基准逐字相同）"
  T=$(cd "$CODE_DIR" && find run_step4e.py trb_env -name '*.py' 2>/dev/null \
      | LC_ALL=C sort | xargs sha256sum 2>/dev/null | sha256sum | cut -c1-16)
  echo "  本机 TRAIN = $T"
  echo "  冻结基准   = 198d2c8a61c8f8ba"
  if [ "$T" = "198d2c8a61c8f8ba" ]; then
    echo "  ✅ 训练/评估代码没漂，与训出这批存档时逐字相同"
  else
    echo "  🔴 对不上！这台机的 代码/ 与训练时不是同一份 ⟹ 评出来的数不可比。先查清楚再跑。"
    exit 1
  fi

  banner "第 0.5 步 · 存档到底有多少、有没有同名的（合并多机之后最容易出事的地方）"
  echo "  主存档（正式臂）："
  find "$RES" -path "*/checkpoints/*_F240*.zip" ! -path "*/segments/*" 2>/dev/null | wc -l
  echo "  分段副本："
  find "$RES" -path "*/checkpoints/segments/*_F240*.zip" 2>/dev/null | wc -l
  echo
  echo "  🔴 同名存档检查（同名分处两个目录 ⟹ select_best_ckpt 按 basename 建字典、会静默取其一）："
  find "$RES" -path "*/checkpoints/*_F240*.zip" ! -path "*/segments/*" 2>/dev/null \
    | sed 's|.*/||' | sort | uniq -d | while read -r d; do
        echo "  ❌ 重名：$d"
        find "$RES" -name "$d" ! -path "*/segments/*" 2>/dev/null | sed 's/^/       /'
      done
  echo "  （上面没有 ❌ 就说明没有重名）"

  banner "第 1 步 · 完整性体检（退出码 1 = 有硬伤，先修）"
  SEEDS="$SEEDS" ARMS="$ARMS_ALL" NSEG=20 \
    "$PY" -B "$CODE_DIR/tests/check_formal_integrity.py" "$RES"
  RC=$?
  echo
  echo "  体检退出码 = $RC   （0 = 全过 · 1 = 硬伤 · 2 = 只有提醒）"
  echo "  🔴 记下它最后一行打的『BUDGET_SEG 上限 = N 段』，下一步要用。"
  exit $RC
  ;;

best)
  banner "第 2 步 · 用验证集挑最佳存档（规则不带旋钮：单一指标=验证集到达率·平局取更早的段）"
  echo "  BUDGET_SEG = ${BUDGET_SEG:-<未设·用全部段>}"
  echo "  🔴 同一张表所有臂必须用同一个值；不确定就用体检打出来的那个上限。"
  BUDGET_SEG="${BUDGET_SEG:-0}" "$PY" -B "$CODE_DIR/tests/select_best_ckpt.py" "$RES" "$RES/_best.json"
  echo
  echo "  产物 → $RES/_best.json"
  ;;

last)
  banner "第 3 步 · 重评【末段存档】那一趟（约 1.2 小时）"
  echo "  种子 = $SEEDS · 分母期望 600 · 无盾臂 uns 会自动单独起一个进程组"
  PASS=formal_last FORMAL_SEEDS="$SEEDS" \
    bash "$CODE_DIR/tests/run_reeval_all.sh" 8 "$RES/正式-末段" 2>&1 | tee "$RES/_reeval_last.log"
  ;;

bestpass)
  banner "第 4 步 · 重评【验证集最佳存档】那一趟（约 1.2 小时 · 必须与上一趟同机背靠背）"
  [ -f "$RES/_best.json" ] || { echo "❌ 先跑 best 那一步生成 $RES/_best.json"; exit 1; }
  PASS=formal_best FORMAL_SEEDS="$SEEDS" SELJSON="$RES/_best.json" \
    bash "$CODE_DIR/tests/run_reeval_all.sh" 8 "$RES/正式-最佳" 2>&1 | tee "$RES/_reeval_best.log"
  ;;

traj)
  banner "第 5 步 · 轨迹专趟（图 6 要用 · 全部 600 个场景 × 9 臂 × 3 颗种子 · 约 1.2 小时 · 约 176 MB）"
  echo "  🔴 种子集合 s0/s1/s2 是**事先声明**的，别跑完再挑好看的那颗（`04:408`）"
  PASS=traj TRAJ_SEEDS="0 1 2" \
    bash "$CODE_DIR/tests/run_reeval_all.sh" 3 "$RES/正式-轨迹全集" 2>&1 | tee "$RES/_reeval_traj.log"
  ;;

*)
  sed -n '2,20p' "$0"
  echo
  echo "❌ 第一个参数要是 check / best / last / bestpass / traj 之一"
  exit 1
  ;;
esac
