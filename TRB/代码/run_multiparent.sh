#!/usr/bin/env bash
# ============================================================================
# 多父策略【嵌套】设计：3 个不同水平的健康父策略 × 各 3 个微调种子 = 9 run · 5M · hocr_200
# ----------------------------------------------------------------------------
# 目的（补 `03` L191-B ② / L192 的最大软肋"有效样本量≈1"）：
#   热启动的 10 种子全从【同一个】事后挑的父策略(s1)衍生 → 有效 n≈1 → IQM 窄带只是
#   "一个父策略的微调抖动带"·撑不起"种子方差小/方法稳"。要把【父策略挑得好】与【方法真有效】
#   拆开 → 必须从【多个不同水平的父策略】各衍生几个种子（嵌套：父=组·微调种子=组内重复）。
#
# 设计（user 2026-07-21 拍）：
#   父 A = s0（金标 92.5·最好）  父 B = s7（77.5·中等）  父 C = s2（65·偏低的健康种子）
#   每父 × 微调种子 {0,1,2} → 9 run。同一批种子号跨父 = 配对/分块可分析（父 vs 种子方差分解）。
#
# 配方 = 逐字对齐 run_warmstart.sh（=金标 L1rateON + 热启动开关）·唯一变量 = 【父源 ckpt + TAG + 种子】。
#   配方写错 → 代码当场 fail-fast（读源 sidecar config_sig 比对·非人肉）。3 个父都是金标 L1rateON →
#   语义配置与本 run 一致 → 源配置校验通过；3 个父权重不同 → 指纹不同（下方自证）。
#
# 🔴 方法论红线（同 run_warmstart.sh）：
#   · 判读【多指标·不挑种子·报 IQM+崩率+CI(rliable)】；父间差异 = 真信号（"挑得好"的量化）。
#   · provenance 自动记【每 run 的父源路径+内容指纹】→ 跑完自证：同父 3 种子同指纹、异父指纹互异。
#   · 绝不 claim「从零稳定收敛」。
#
# 用法（服务器·先同步【整个 代码 文件夹】）：改下面 1 行 CODE_DIR → bash run_multiparent.sh [并发K=9]
#   断点续（诚实·对抗审订正）：只有【整段跑完】的 run 会被跳过；中途被杀的 run 重跑会【整个从头重热启动+重训】
#     （非分段续训·且会覆盖被杀那次留下的中间 checkpoint）→ 长时间无人值守烧多天前先知情此代价。
#   训练吃 CPU 不吃 GPU；9 run × n_envs=8 = 72 核。
#   ⚠️ 需要服务器上有 s0/s2/s7 三个金标 ckpt（金标 run `结果0710-22:00-10种子最优方案/checkpoints/`）。
#      缺哪个脚本会 fail-fast 并告诉你补传哪个文件。
# ============================================================================
set -uo pipefail

# ==== 只改这 1 行（你服务器放 run_step4e.py 的目录）====
CODE_DIR="/root/trb/代码"
# =======================================
[ -d "$CODE_DIR" ] || { echo "❌ CODE_DIR 不存在：$CODE_DIR"; exit 1; }
cd "$CODE_DIR" || exit 1
ROOT="$(cd "$CODE_DIR/.." && pwd)"
RES_DIR="$ROOT/结果"                                    # run_step4e 恒写到 <代码>/../结果
mkdir -p "$RES_DIR" || { echo "❌ 建不了结果目录：$RES_DIR"; exit 1; }
PY="/root/miniconda3/bin/python"
MANIFEST="$ROOT/balanced_pool/manifest_hocr_200.json"
BALANCED="$ROOT/balanced_pool"
SDIR="$ROOT/scenarios"
KMAX="${1:-9}"
FROZEN_DIR="$ROOT/ws_src_multiparent"                   # 独立冻结目录（不碰 run_warmstart 的 ws_src）
mkdir -p "$FROZEN_DIR" || { echo "❌ 建不了冻结目录：$FROZEN_DIR"; exit 1; }

# ---- 父策略清单：plabel  金标ckpt名（无扩展名）----
#   plabel 只用于 TAG/自证分组；ckpt 名 = 金标 L1rateON 命名。
PARENTS_LABELS=("0" "7" "2")
PARENTS_CKPTS=(
  "Continuous-safe_s0_L1rateON_ppo_s0"
  "Continuous-safe_s7_L1rateON_ppo_s7"
  "Continuous-safe_s2_L1rateON_ppo_s2"
)
FT_SEEDS=("0" "1" "2")                                  # 每父的微调种子（同一批跨父=配对）

# ---- 数组一致性断言（改父策略时防写错·对抗审 NIT）----
[ "${#PARENTS_LABELS[@]}" = "${#PARENTS_CKPTS[@]}" ] || { echo "❌ PARENTS_LABELS(${#PARENTS_LABELS[@]}) 与 PARENTS_CKPTS(${#PARENTS_CKPTS[@]}) 长度不一致（改数组须一一对应）"; exit 1; }
_dup=$(printf '%s\n' "${PARENTS_LABELS[@]}" | sort | uniq -d)
[ -z "$_dup" ] || { echo "❌ PARENTS_LABELS 有重复标签：$_dup（TAG 会撞）"; exit 1; }

# ============================================================================
# [准备] 逐父：自动发现 + 冻结只读 + 指纹（防跑到一半被覆盖=换源静默混写·`03` L190 D3 HIGH#1）
# ============================================================================
declare -A FROZEN_OF                                    # ckptname -> 冻结路径(无扩展)
echo "===== [准备] 冻结 3 个父策略源 ckpt ====="
miss=0
for i in "${!PARENTS_CKPTS[@]}"; do
  NAME="${PARENTS_CKPTS[$i]}"
  PL="${PARENTS_LABELS[$i]}"
  FROZEN="$FROZEN_DIR/$NAME"
  FROZEN_OF["$NAME"]="$FROZEN"
  if [ -f "$FROZEN.zip" ] && [ -f "${FROZEN}_vecnorm.pkl" ] && [ -f "${FROZEN}.progress.json" ]; then
    echo "  ✅ 父 s$PL 冻结源已存在（三文件齐·复用）：$FROZEN.zip"
    continue
  fi
  # 残缺冻结（上次三文件复制中途被 kill·对抗审 fix）→ 清掉重来（只读位可能挡 rm·先加回写）
  chmod +w "$FROZEN".* 2>/dev/null || true
  rm -f "$FROZEN.zip" "${FROZEN}_vecnorm.pkl" "${FROZEN}.progress.json" "${FROZEN}".*.tmp 2>/dev/null || true
  # 🔎 全盘搜金标源（别硬编码路径·服务器目录名可能不同）·排除已冻结目录避免自发现
  SRC_ZIPS="$(find "$ROOT" -name "${NAME}.zip" -not -path "$FROZEN_DIR/*" -not -path "$ROOT/ws_src/*" 2>/dev/null)"
  NCAND=$(printf '%s' "$SRC_ZIPS" | grep -c .)
  if [ "$NCAND" = "0" ]; then
    echo "  ❌ 父 s$PL 找不到金标源 ${NAME}.zip（在 $ROOT 全盘搜过）"
    miss=1; continue
  fi
  if [ "$NCAND" -gt 1 ]; then                            # 🔴 多个同名源=可能选错父静默污染嵌套设计（自证查不出"选错但用得一致"·对抗审 MEDIUM）
    echo "  ❌ 父 s$PL 发现【$NCAND 个】同名源 ${NAME}.zip → 无法判断用哪个（恐选错源）："
    printf '%s\n' "$SRC_ZIPS" | sed 's/^/       /'
    echo "     → 删掉多余/旧副本只留 1 个再跑。"
    miss=1; continue
  fi
  SRC_GOLD="${SRC_ZIPS%.zip}"
  echo "  🔎 父 s$PL 发现源：$SRC_GOLD"
  ok=1
  for f in ".zip" "_vecnorm.pkl" ".progress.json"; do   # ⚠️ .progress.json 必带=源配置校验靠它
    if [ ! -f "${SRC_GOLD}${f}" ]; then
      echo "     ❌ 源缺 ${SRC_GOLD}${f}"
      [ "$f" = ".progress.json" ] && echo "        ⓘ 缺 sidecar → 失去'配方对不对'的自动拦截·建议补传"
      ok=0
    fi
  done
  if [ "$ok" != "1" ]; then miss=1; continue; fi
  # 原子冻结（对抗审 fix）：先全复制到 .tmp·三个都成功再 rename → 半途被 kill 只留 .tmp（不会被误判"已冻结"）
  for f in ".zip" "_vecnorm.pkl" ".progress.json"; do cp "${SRC_GOLD}${f}" "${FROZEN}${f}.tmp" || exit 1; done
  for f in ".zip" "_vecnorm.pkl" ".progress.json"; do mv "${FROZEN}${f}.tmp" "${FROZEN}${f}" || exit 1; done
  chmod -w "$FROZEN".* 2>/dev/null || true              # 只读=挡非 root 误覆盖（注：root 无视权限位·非硬保证）
  echo "     ✅ 已冻结到 $FROZEN_DIR（三文件原子落地·只读）"
done
if [ "$miss" = "1" ]; then
  echo ""
  echo "❌ 有父策略源缺失 → 不起飞（避免只跑一部分父=毁掉嵌套设计）。"
  echo "   → 看服务器上有哪些金标连续PPO ckpt："
  echo "     find $ROOT -name 'Continuous-safe_s*_L1rateON_ppo_s*.zip' | sort"
  echo "   → 从本地补传缺的父到服务器（整个 结果0710-22:00-10种子最优方案/checkpoints/ 目录最省事）。"
  echo "   → 想换父：改本脚本 PARENTS_CKPTS/PARENTS_LABELS 两个数组（保持一一对应）。"
  exit 1
fi

echo ""
echo "===== 源指纹（3 个父应【互不相同】=真的是 3 个不同父策略）====="
for i in "${!PARENTS_CKPTS[@]}"; do
  NAME="${PARENTS_CKPTS[$i]}"; PL="${PARENTS_LABELS[$i]}"; FROZEN="${FROZEN_OF[$NAME]}"
  "$PY" - "$FROZEN" "$PL" <<'PYEOF'
import hashlib, sys
frozen, pl = sys.argv[1], sys.argv[2]
outs=[]
for s in (".zip", "_vecnorm.pkl"):
    h = hashlib.sha256()
    with open(frozen + s, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""): h.update(c)
    outs.append(f"{s}={h.hexdigest()[:16]}")
print(f"  父 s{pl}: " + "  ".join(outs))
PYEOF
done

# ============================================================================
# [配方] 逐字 = run_warmstart.sh（金标 L1rateON + 热启动）· 唯一变量 = 父源/TAG/种子
# ============================================================================
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export STEP4E_SMOKE=0 STEP4E_NTOTAL=200 STEP4E_STEPS=5000000 STEP4E_NSEG=10 STEP4E_LOG_CURVES=1   # NSEG=10 逐字金标·别改20
export STEP4E_MANIFEST="$MANIFEST" STEP4E_BALANCED_DIR="$BALANCED" STEP4E_SDIR="$SDIR"
export STEP4E_WELL_B=200 STEP4E_SHAPING_RADIUS=500 STEP4E_WELL_X=200 STEP4E_XTRACK_RADIUS=80
export STEP4E_PARK_W=20 STEP4E_PARK_RADIUS=400 STEP4E_PARK_VTARGET=4
export STEP4E_RATE_W=1.0                                                    # 治抖 ON（金标同款）
export STEP4E_CONTINUOUS_ALGO=ppo STEP4E_ENT_START=0.01 STEP4E_ENT_END=0.01 STEP4E_NENVS=8
export STEP4E_PARTIES=Continuous-safe
export PY RES_DIR
# ⚠️ STEP4E_WARMSTART_CKPT / STEP4E_SEEDS / STEP4E_TAG 逐 run 设（见 run_one）·不在此 export

# ---- 构造 run 清单：每行 = "plabel ckptname seed" ----
RUNS=()
for i in "${!PARENTS_CKPTS[@]}"; do
  for S in "${FT_SEEDS[@]}"; do
    RUNS+=("${PARENTS_LABELS[$i]} ${PARENTS_CKPTS[$i]} $S")
  done
done

echo ""
echo "===== [起飞] ${#RUNS[@]} run（3 父 × 3 种子·并发≤$KMAX·每 run ~4-5h）====="
echo "  ⓘ 配方写错/源配置不符 → 代码当场 fail-fast（不会白烧）"
run_one() {
  local PL="$1" NAME="$2" S="$3"
  local FROZEN="$FROZEN_DIR/$NAME"
  local TAG="_mpP${PL}_ppo_s${S}"                       # 含 ppo 过 PPO 诊断闸门·父+种子唯一=不撞文件/不混写
  STEP4E_WARMSTART_CKPT="$FROZEN" STEP4E_SEEDS="$S" STEP4E_TAG="$TAG" \
    "$PY" -B run_step4e.py > "$RES_DIR/${TAG}.log" 2>&1 \
    && echo "  [完] 父s$PL 种子$S" || echo "  [⚠️失败] 父s$PL 种子$S（看 结果/${TAG}.log）"
}
export -f run_one
export FROZEN_DIR
printf '%s\n' "${RUNS[@]}" | xargs -P "$KMAX" -L1 bash -c 'run_one "$@"' _

# ============================================================================
# [自证] 同父 3 种子应【同指纹】；3 父之间应【指纹互异】
# ============================================================================
echo ""
echo "===== 🔴 嵌套同源自证（方法论命门·必看）====="
allfps=""
for i in "${!PARENTS_LABELS[@]}"; do
  PL="${PARENTS_LABELS[$i]}"
  FPS=$(grep -ho '"warmstart_src_fp": *{[^}]*}' "$RES_DIR"/step4e_partial_mpP${PL}_ppo_s*.jsonl 2>/dev/null | sort -u)
  NFP=$(printf '%s' "$FPS" | grep -c .)
  NDONE=$(ls "$RES_DIR"/step4e_partial_mpP${PL}_ppo_s*.jsonl 2>/dev/null | wc -l | tr -d ' ')
  if [ "$NFP" = "1" ]; then
    echo "  ✅ 父 s$PL：$NDONE/3 种子·【同一指纹】= 同源坐实"
    allfps="$allfps$FPS"$'\n'
  elif [ "$NFP" = "0" ]; then
    echo "  ⓘ 父 s$PL：无指纹记录（$NDONE/3 完成）→ 还没产出·先看 结果/_mpP${PL}_ppo_s*.log 查失败"
  else
    echo "  🔴🔴 父 s$PL：出现 $NFP 个不同指纹 = 源被换过 → 该父数据【作废】·查 $FROZEN_DIR"
  fi
done
UNIQ_PARENTS=$(printf '%s' "$allfps" | grep -c .)
if [ "$UNIQ_PARENTS" -ge 2 ]; then
  DISTINCT=$(printf '%s' "$allfps" | grep . | sort -u | grep -c .)
  if [ "$DISTINCT" = "$UNIQ_PARENTS" ]; then
    echo "  ✅ $UNIQ_PARENTS 个父策略指纹【互不相同】= 真的是不同父（嵌套设计成立）"
  else
    echo "  🔴 有父策略指纹相同 = 你可能把同一个 ckpt 填成了两个父·查 PARENTS_CKPTS 数组"
  fi
fi
echo ""
echo "回传给主窗口：结果/step4e_partial_mpP*_ppo_s*.jsonl + 结果/checkpoints/*_mpP*_ppo_s*.progress.json（含 trend）"
echo "被杀/失败：重跑同一条命令跳过【已整段跑完】的 run；中途被杀的 run 会整个从头重跑（非分段续·会覆盖中间 checkpoint）。"
