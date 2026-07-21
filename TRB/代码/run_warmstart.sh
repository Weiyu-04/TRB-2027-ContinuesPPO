#!/usr/bin/env bash
# ============================================================================
# 热启动治崩：全 10 种子【统一从同一金标源(s1)】热启动 × 5M · 小场景库 hocr_200
# ----------------------------------------------------------------------------
# 目的：崩种子(s5/s6)整个 5M【从没爬出"满速打转-超时"吸引子=探索失败】(非奖励给错·`03` L190 A)
#       → 灌一个好策略(金标 s1)当初始化·把崩种子放进好盆地。顶会主流(JSRL 2204.02372 / AWAC 2006.09359)。
# 配方 = 逐字对齐 run_leg1_rate.sh(=金标 L1rateON)·**唯一新增开关 STEP4E_WARMSTART_CKPT**。
#       配方若写错 → 代码【当场 fail-fast】("热启动源与本 run 语义配置不一致"·读源 sidecar 比对·非靠人肉)。
#
# 🔴 方法论红线(user 2026-07-16 已认同"热启动=训练手段·如实描述即可·非不公平")：
#   · **全 10 种子【统一】从同一好源**(非只救 s5/s6=10种子不独立=挑数据)。s1→s1 = 续训·报告标明。
#   · provenance 已自动记【源路径+内容指纹】→ 跑完用下方 grep **自证 10 种子同源**。
#   · **绝不 claim「从零稳定收敛」**（用了热启动就如实写）。诚实报 IQM+崩率+CI(rliable)。
#
# 用法（服务器·先同步【整个 代码 文件夹】）：改下面 1 行 CODE_DIR → bash run_warmstart.sh [并发K=10]
# 断点续：被杀了重跑同一条命令、自动跳过已完成种子。
# ⚠️ 训练吃 CPU 不吃 GPU；10 种子 × n_envs=8 = 80 核；32 核则 K=4 分批（脚本自动排队·总时长拉长）。
# ============================================================================
set -uo pipefail

# ==== 只改这 1 行（你服务器放 run_step4e.py 的目录）====
CODE_DIR="/root/trb/代码"
# =======================================
[ -d "$CODE_DIR" ] || { echo "❌ CODE_DIR 不存在：$CODE_DIR"; exit 1; }
cd "$CODE_DIR" || exit 1
RES_DIR="$(cd "$CODE_DIR/.." && pwd)/结果"       # run_step4e 恒写到 <代码>/../结果
ROOT="$(cd "$CODE_DIR/.." && pwd)"
mkdir -p "$RES_DIR" || { echo "❌ 建不了结果目录：$RES_DIR"; exit 1; }   # 防结果目录不存在→下面写 log 重定向失败(服务器 layout 可能与本地不同)
PY="/root/miniconda3/bin/python"
MANIFEST="$ROOT/balanced_pool/manifest_hocr_200.json"
BALANCED="$ROOT/balanced_pool"
SDIR="$ROOT/scenarios"
SEEDS="0 1 2 3 4 5 6 7 8 9"
KMAX="${1:-10}"

# ---- 热启动源 = 金标 s1（健康·实测 90%）----
SRC_NAME="Continuous-safe_s1_L1rateON_ppo_s1"
FROZEN_DIR="$ROOT/ws_src"
FROZEN="$FROZEN_DIR/$SRC_NAME"

echo "===== [准备] 冻结源 ckpt（防跑到一半被覆盖=换源静默混写·03 L190 D3 HIGH#1）====="
if [ ! -f "$FROZEN.zip" ]; then
  # 🔎 自动发现金标源（服务器目录名可能与本地不同 → 别硬编码路径·全盘搜 ckpt 名）
  SRC_ZIP="$(find "$ROOT" -name "${SRC_NAME}.zip" -not -path "*/ws_src/*" 2>/dev/null | head -1)"
  if [ -z "$SRC_ZIP" ]; then
    echo "❌ 找不到金标源 ${SRC_NAME}.zip（在 $ROOT 下全盘搜过）"
    echo "   → 你服务器上有哪些连续PPO金标 ckpt？跑这条看看："
    echo "     find $ROOT -name 'Continuous-safe_s*_L1rateON_ppo_s*.zip' | head"
    echo "   → 若金标 run 没传到服务器：从本地上传【整个 结果0710-22:00-10种子最优方案/checkpoints/ 目录】"
    echo "   → 若想换别的健康源(如 s0)：改本脚本 SRC_NAME= 那一行"
    exit 1
  fi
  SRC_GOLD="${SRC_ZIP%.zip}"
  echo "  🔎 发现源：$SRC_GOLD"
  mkdir -p "$FROZEN_DIR"
  for f in ".zip" "_vecnorm.pkl" ".progress.json"; do      # ⚠️ .progress.json 必带=源配置校验靠它
    if [ ! -f "${SRC_GOLD}${f}" ]; then
      echo "❌ 源缺 ${SRC_GOLD}${f}"
      [ "$f" = ".progress.json" ] && echo "   ⓘ 缺 sidecar → 源配置校验做不了(会 warning 放行·但你就失去了'配方对不对'的自动拦截)·建议从本地补传这个文件"
      exit 1
    fi
    cp "${SRC_GOLD}${f}" "${FROZEN}${f}" || exit 1
  done
  chmod -w "$FROZEN".* 2>/dev/null || true                 # 只读=杜绝被覆盖
  echo "  ✅ 已冻结到 $FROZEN_DIR（只读）"
else
  echo "  ✅ 冻结源已存在（复用）：$FROZEN.zip"
fi
"$PY" - <<PYEOF
import hashlib
for s in (".zip", "_vecnorm.pkl"):
    h = hashlib.sha256()
    with open("$FROZEN" + s, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""): h.update(c)
    print(f"  源指纹 {s:14s} sha256[:16] = {h.hexdigest()[:16]}")
PYEOF

# ---- 配方：逐字 = run_leg1_rate.sh（金标 L1rateON）+ 唯一新增 WARMSTART ----
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export STEP4E_SMOKE=0 STEP4E_NTOTAL=200 STEP4E_STEPS=5000000 STEP4E_NSEG=10 STEP4E_LOG_CURVES=1   # 🔴 NSEG=10 逐字=金标(别改20)：n_seg 改动=step 网格变→学习曲线无法与金标直接叠(`03` L58#2)·且破"唯一开关"原则。500k 粒度足够看出滑回(金标 trend 就是这个粒度看清崩种子全程趴0的)
export STEP4E_MANIFEST="$MANIFEST" STEP4E_BALANCED_DIR="$BALANCED" STEP4E_SDIR="$SDIR"
export STEP4E_WELL_B=200 STEP4E_SHAPING_RADIUS=500 STEP4E_WELL_X=200 STEP4E_XTRACK_RADIUS=80
export STEP4E_PARK_W=20 STEP4E_PARK_RADIUS=400 STEP4E_PARK_VTARGET=4
export STEP4E_RATE_W=1.0                                   # 治抖 ON（金标同款）
export STEP4E_CONTINUOUS_ALGO=ppo STEP4E_ENT_START=0.01 STEP4E_ENT_END=0.01 STEP4E_NENVS=8
export STEP4E_PARTIES=Continuous-safe
export STEP4E_WARMSTART_CKPT="$FROZEN"                     # ← 唯一新增开关（全 10 种子统一同源）
export PY RES_DIR

echo "===== [起飞] 10 种子 × 5M（并发≤$KMAX·每种子 ~4-5h）====="
echo "  源 = $FROZEN"
echo "  ⓘ 配方写错/源配置不符 → 代码当场 fail-fast（不会白烧）"
run_one() {
  local S="$1"
  STEP4E_SEEDS="$S" STEP4E_TAG="_wsHOCRppo_s$S" "$PY" -B run_step4e.py > "$RES_DIR/_wsHOCRppo_s$S.log" 2>&1 \
    && echo "  [完] s$S" || echo "  [⚠️失败] s$S（看 结果/_wsHOCRppo_s$S.log）"
}
export -f run_one
printf '%s\n' $SEEDS | xargs -P "$KMAX" -I{} bash -c 'run_one "$@"' _ {}

N=$(ls "$RES_DIR"/step4e_partial_wsHOCRppo_s*.jsonl 2>/dev/null | wc -l | tr -d ' ')
echo "===== 完成：产出 $N/10 个种子结果 ====="
echo ""
echo "===== 🔴 同源自证（方法论命门·必看）====="
FPS=$(grep -ho '"warmstart_src_fp": *{[^}]*}' "$RES_DIR"/step4e_partial_wsHOCRppo_s*.jsonl 2>/dev/null | sort -u)
NFP=$(printf '%s' "$FPS" | grep -c . )                     # printf 不补换行 → 空串恒 0（echo "" 会算成 1 行·踩过）
[ -n "$FPS" ] && echo "$FPS"
if [ "$NFP" = "1" ]; then
  echo "  ✅ $N 个种子【同一指纹】= 统一同源坐实（可写进论文的机器审计证据）"
elif [ "$NFP" = "0" ]; then                                # 🔴 别把"没数据"误诊成"换源"（dry-run 抓出的误报）
  echo "  ⓘ 无指纹记录 = 还没产出结果（$N/10 个种子完成）→ 不是换源问题；先看 结果/_wsHOCRppo_s*.log 查失败原因"
else
  echo "  🔴🔴 出现 $NFP 个【不同指纹】= 源被换过 → 该批数据【作废】·查 $FROZEN_DIR 是否被覆盖"
fi
echo ""
echo "回传给主窗口：结果/step4e_partial_wsHOCRppo_s*.jsonl + 结果/checkpoints/*_wsHOCRppo_s*.progress.json（含 trend 全程曲线）"
echo "被杀/失败：重跑同一条命令自动续跑未完成种子。"
