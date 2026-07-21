#!/usr/bin/env bash
# Step-0 · s2 replay 预检+跑（在【服务器】上跑·eval-only·不训练·不烧 GPU）。
# 用法：改下面 2 行为你服务器的真实路径 → bash run_s2_replay.sh
# 安全网：① 先验证同步的代码是新的(含改动·否则拒·防跑旧代码白跑) ② 自动发现 s2 checkpoint 基名+EXPECT
#         ③ replay_dump 内置【复现闸门】逐位比原 jsonl·不符即中止绝不 dump 垃圾。任何一步不对=loud 中止·exit 非 0。
set -euo pipefail

# ==== 只改这 2 行（你服务器上的真实路径）====
CODE_DIR="/root/trb/代码"                    # 服务器上放 run_step4e.py / evaluate.py / replay_dump.py 的目录
RES_DIR="/root/trb/结果"                      # 服务器上结果目录（含 checkpoints/ 和 结果0709*/）
# ===========================================
PY="/root/miniconda3/bin/python"
MANIFEST="/root/trb/balanced_pool/manifest_hocr_200.json"

echo "===== [预检 0] 路径存在 ====="
[ -d "$CODE_DIR" ] || { echo "❌ CODE_DIR 不存在：$CODE_DIR"; exit 1; }
[ -d "$RES_DIR" ]  || { echo "❌ RES_DIR 不存在：$RES_DIR"; exit 1; }
[ -f "$MANIFEST" ] || { echo "❌ manifest 不存在：$MANIFEST（改脚本里的 MANIFEST）"; exit 1; }

echo "===== [预检 1] 同步的代码是新的（含本次改动·防跑旧代码白跑）====="
grep -q "_approach_diag" "$CODE_DIR/trb_env/evaluate.py"      || { echo "❌ evaluate.py 无 _approach_diag → 未同步 Step-0 仪表·先 sync"; exit 1; }
grep -q "return_per"      "$CODE_DIR/run_step4e.py"           || { echo "❌ run_step4e.py 无 return_per → 未同步·先 sync"; exit 1; }
[ -f "$CODE_DIR/replay_dump.py" ]                             || { echo "❌ replay_dump.py 不在 $CODE_DIR → 先 sync"; exit 1; }
echo "  ✅ 三处改动都在"

echo "===== [预检 2] 发现 s2 checkpoint 基名 ====="
# checkpoint 名 = {party}_s{seed}{TAG}=Continuous-safe_s2_<TAG含dwelloff>.zip；seed 槽 _s2_ 紧跟 party（在 dwelloff 前）
CKZIP="$(ls "$RES_DIR"/checkpoints/ 2>/dev/null | grep -iE 'Continuous-safe_s2_.*dwelloff.*\.zip$' || true)"
N=$(printf '%s\n' "$CKZIP" | grep -c . || true)
[ "$N" = "1" ] || { echo "❌ s2 dwelloff checkpoint .zip 命中 $N 个（期望 1）：'$CKZIP' → 手动指定 REPLAY_CKPT 重跑"; ls "$RES_DIR"/checkpoints/ | grep -i dwelloff || true; exit 1; }
CKBASE="$RES_DIR/checkpoints/${CKZIP%.zip}"
[ -f "${CKBASE}_vecnorm.pkl" ] || { echo "❌ 缺 ${CKBASE}_vecnorm.pkl（复现须归一化统计）"; exit 1; }
echo "  ✅ checkpoint 基名：$CKBASE"

echo "===== [预检 3] 发现 EXPECT 原 jsonl（复现闸门用·须唯一且非空）====="
EXPECT_HITS="$(find "$RES_DIR" -name 'step4e_partial_dwelloffHOCRppo_s2.jsonl' 2>/dev/null || true)"
NE=$(printf '%s\n' "$EXPECT_HITS" | grep -c . || true)
[ "$NE" = "1" ] || { echo "❌ EXPECT 命中 $NE 个（期望 1）于 $RES_DIR：'$EXPECT_HITS' → 手动确认后设 REPLAY_EXPECT_JSONL 重跑（多个=散在多目录·0=不在服务器·请从本地 scp 上来）"; exit 1; }
EXPECT="$EXPECT_HITS"
[ -s "$EXPECT" ] || { echo "❌ EXPECT 是空文件：$EXPECT"; exit 1; }
echo "  ✅ EXPECT：$EXPECT"

OUT="$RES_DIR/s2_replay_step0.jsonl"
echo "===== [跑] replay_dump（eval-only·输出 $OUT）====="
cd "$CODE_DIR"
REPLAY_CKPT="$CKBASE" \
REPLAY_MANIFEST="$MANIFEST" \
REPLAY_EXPECT_JSONL="$EXPECT" \
REPLAY_OUT="$OUT" \
REPLAY_ALGO="ppo" \
"$PY" -B replay_dump.py
echo ""
echo "✅ 完成。把 $OUT 传回本地（scp 到 结果/ 下）给主窗口分析。"
