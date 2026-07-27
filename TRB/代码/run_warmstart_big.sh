#!/usr/bin/env bash
# ============================================================================
# 【两段式·第二段】小集训出的起点 → 在【官方 1400 训练集】上继续训练（`03` L223·user 2026-07-26 提的方案）
# ----------------------------------------------------------------------------
# 想解决什么：现在最大的软肋是训练集只有 94 个场景（官方是 1400）。
#   早前"直接在大集上从零训"失败过（成功率中位数 55%），诊断的病因是**欠拟合**——场景太多、每个练的遍数不够。
#   而"从已训好的策略继续训练"恰好消掉这个病因（策略已经会开船，只需适应更多场景）⟹ 那条负结果**不否决**本方案。
#
# 与 run_warmstart.sh 的差别【五处】，配方（奖励/盾/算法/熵/并行）逐字不动：
#   ① 数据集：manifest_hocr_200(94训) → **manifest_official_1300**(官方1400 里切 1300训/100验)
#      🔴 2026-07-27 独立复审更正（`03` L229-G）：**下面写的"只换训练集的单变量对照"是 overclaim。**
#      实际同时变了两样：场景【数量】94→1300 **和** 会遇类型【配比】对遇:交叉 = 1:1 → 468:832(36%:64%)。
#      不是错误（官方测试 600 本来也是 ~1:2，按官方配比训反而更贴测试分布），但**汇报/论文措辞必须写
#      "换到官方训练集（数量与配比同时变化）"**，不许写"单变量"。
#   ② 步数：5M → **2.5M** · NSEG 10 → 5（每段仍 500k·步网格不变）
#   ③ 场景总数：STEP4E_NTOTAL 200 → **2000**（🔴 2026-07-27 独立复审 L226-Q 补进本清单：
#      原文写"只有四处"却没列它。manifest 模式下 `_pool_eff=None`、`make_split` 不被调用 ⟹ 无实质影响，
#      但它确实进 config_sig 签名与 Table III 表头 ⟹ 差异清单必须列全，否则下次核对会漏。)
#   ④ TAG：_wsHOCRppo_s$S → **_wsBIGppo_s$S**（不同 TAG=不会去续现有那批·且 dataset 进 config_conflict 会硬拦混写）
#   ⑤ 种子：默认只跑 **3 颗探针**（**1 3 4**）；打算上全量就 `SEEDS="0 1 2 3 4 5 6 7 8 9" bash ...`
#
# 🔴🔴 为什么默认种子是 1 3 4 而不是 0 1 2（2026-07-27 独立复审 L226-R·**这条决定对照成不成立**）：
#   本脚本 2.5M 的用意是"与现有热启动那批对齐 ⟹ 只换训练集的单变量对照"。但**现有那批不是一个步数**——
#   逐个读 progress.json 实测：s1/s3/s4/s5/s7 = 2,539,520 步（seg_done=4）· s0/s2/s6/s8/s9 = 3,047,424 步（seg_done=5）。
#   本 run 的终点恰是 2,539,520（5 段 × 507,904）⟹ 只有前一组能构成同步数配对。
#   原默认 0 1 2 里有**两颗**（s0/s2）的对照方多训 20%，而每段 checkpoint 是覆盖式的、2.5M 那个已被覆盖找不回
#   ⟹ 事后没法在评估层补齐。混杂幅度与判读门槛（±2pt）同量级 ⟹ 必须在起飞前把种子选对。
#
# 🔴 为什么要用清单、而不是"不设 STEP4E_MANIFEST 走默认官方口径"：
#   默认模式确实就是官方 1400/600（`03` L207-B），但那样 `config_sig.dataset` 会记成 "strided"，
#   而 `代码/tests/reeval_official.py` 有一道 fail-closed 闸专拦它（分不清 strided-200 还是 strided-2000
#   ⟹ 泄漏剔不干净）⟹ **跑完了评不了**。用自描述清单就没这个问题，且训练代码一行不用改。
#   顺带：清单的"验证集"是从官方 1400 里切的 100 个 ⟹ **训练期评估全程不碰报数用的官方 600**。
#
# 🔴 方法论红线（沿用 run_warmstart.sh）：
#   · 全部种子【统一】从同一个起点（非挑种子）· provenance 自动记源路径+内容指纹 · 跑完自证同源
#   · **绝不 claim「从零稳定收敛」**；论文写"两段式：小集训起点 → 官方全集续训"，累计算力如实报
#
# 🔴 评估纪律（跑完必看）：这条臂**必须与既有四臂放在同一趟 reeval 里评**。
#   单独评它会得到 strict=600（它自己不贡献泄漏），与其它臂的 563 **不同分母**，塞进同一张表就是错的。
#   同趟评时泄漏取并集 ⟹ 仍然 563 ⟹ 同分母可比（本机已验）。
#
# 用法（服务器·先同步【整个 代码 文件夹】+ balanced_pool/manifest_official_1300.json）：
#   改下面 1 行 CODE_DIR → **务必用 screen 后台起**（本脚本自己【不】起 screen·L226-S）：
#     screen -dmS wsbig bash -c 'cd /root/trb/代码 && bash run_warmstart_big.sh 3'
#   看进度：screen -r wsbig（退出 Ctrl+A 再 D）｜ tail -f 结果/*wsBIGppo*.log
# 断点续【语义务必看清·2026-07-27 独立复审 L226-S 更正】：重跑同一条命令只会**跳过已经整颗跑完**的种子
#   （done_keys 读的是"整颗跑完才写进 jsonl"的那条记录）。`run_step4e` 里**没有中途续训**——
#   段循环恒从第 0 段开始 ⟹ **跑到一半被杀的那颗种子会从头重训**，已烧的步数作废。
#   ⟹ 所以【必须 screen 后台】（见下方 用法）。项目自己 `02` 也记着"Layer-2 自动续训缓做"。
# ⚠️ 训练吃 CPU 不吃 GPU；每种子 n_envs=8 ⟹ 3 种子 = 24 核。
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
MANIFEST="$ROOT/balanced_pool/manifest_official_1300.json"   # ← 差别①：官方 1400 里切 1300训/100验
BALANCED="$ROOT/balanced_pool"
SDIR="$ROOT/scenarios"
SEEDS="${SEEDS:-1 3 4}"        # ← 差别⑤：默认 3 颗探针【1 3 4】= 对照方恰在 2,539,520 步的那组（L226-R）；上全量用 SEEDS="0 1 2 3 4 5 6 7 8 9" bash ...
KMAX="${1:-3}"

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
# 🔴 NTOTAL=2000：让 run_step4e 的 _pool_eff = None（POOL 2000 不 > n_total 2000）⟹ 不做 striding 抽样。
#    本脚本走 manifest 模式，NTOTAL 只影响非 manifest 路径，设 2000 是为了万一有人把 MANIFEST 注掉时仍落在官方口径。
# 🔴 STEPS=2500000 / NSEG=5：**每段仍是 500k**（2.5M/5 = 5M/10）⟹ 里程碑的 step 网格与金标、与现有热启动**完全一致**，
#    学习曲线可以直接叠（`03` L58#2 要的就是这条：别改段大小，段数可以少）。2.5M 是为了与现有热启动那批 2.54M 对齐 = 单变量对照。
export STEP4E_SMOKE=0 STEP4E_NTOTAL=2000 STEP4E_STEPS=2500000 STEP4E_NSEG=5 STEP4E_LOG_CURVES=1
# 差别②：2.5M/NSEG=5（每段仍 500k=与金标同步网格）· 差别③：TAG=_wsBIGppo
export STEP4E_MANIFEST="$MANIFEST" STEP4E_BALANCED_DIR="$BALANCED" STEP4E_SDIR="$SDIR"
export STEP4E_WELL_B=200 STEP4E_SHAPING_RADIUS=500 STEP4E_WELL_X=200 STEP4E_XTRACK_RADIUS=80
export STEP4E_PARK_W=20 STEP4E_PARK_RADIUS=400 STEP4E_PARK_VTARGET=4
export STEP4E_RATE_W=1.0                                   # 治抖 ON（金标同款）
export STEP4E_CONTINUOUS_ALGO=ppo STEP4E_ENT_START=0.01 STEP4E_ENT_END=0.01 STEP4E_NENVS=8
export STEP4E_PARTIES=Continuous-safe
export STEP4E_WARMSTART_CKPT="$FROZEN"                     # ← 唯一新增开关（全 10 种子统一同源）
export PY RES_DIR

echo "===== [起飞] 种子 [$SEEDS] × 2.5M · 官方 1300 训练集（并发≤$KMAX）====="
echo "  源 = $FROZEN"
echo "  ⓘ 配方写错/源配置不符 → 代码当场 fail-fast（不会白烧）"
run_one() {
  local S="$1"
  STEP4E_SEEDS="$S" STEP4E_TAG="_wsBIGppo_s$S" "$PY" -B run_step4e.py > "$RES_DIR/_wsBIGppo_s$S.log" 2>&1 \
    && echo "  [完] s$S" || echo "  [⚠️失败] s$S（看 结果/_wsBIGppo_s$S.log）"
}
export -f run_one
printf '%s\n' $SEEDS | xargs -P "$KMAX" -I{} bash -c 'run_one "$@"' _ {}

N=$(ls "$RES_DIR"/step4e_partial_wsBIGppo_s*.jsonl 2>/dev/null | wc -l | tr -d ' ')
echo "===== 完成：产出 $N 个种子结果 ====="
echo ""
echo "===== 🔴 同源自证（方法论命门·必看）====="
FPS=$(grep -ho '"warmstart_src_fp": *{[^}]*}' "$RES_DIR"/step4e_partial_wsBIGppo_s*.jsonl 2>/dev/null | sort -u)
NFP=$(printf '%s' "$FPS" | grep -c . )                     # printf 不补换行 → 空串恒 0（echo "" 会算成 1 行·踩过）
[ -n "$FPS" ] && echo "$FPS"
if [ "$NFP" = "1" ]; then
  echo "  ✅ $N 个种子【同一指纹】= 统一同源坐实（可写进论文的机器审计证据）"
elif [ "$NFP" = "0" ]; then                                # 🔴 别把"没数据"误诊成"换源"（dry-run 抓出的误报）
  echo "  ⓘ 无指纹记录 = 还没产出结果（已完成 $N 个种子·本次要跑 [$SEEDS]）→ 不是换源问题；先看 结果/_wsBIGppo_s*.log 查失败原因"
else
  echo "  🔴🔴 出现 $NFP 个【不同指纹】= 源被换过 → 该批数据【作废】·查 $FROZEN_DIR 是否被覆盖"
fi
echo ""
echo "回传给主窗口：结果/step4e_partial_wsBIGppo_s*.jsonl + 结果/checkpoints/*_wsBIGppo_s*.progress.json（含 trend 全程曲线）"
echo "被杀/失败：重跑同一条命令会跳过【已整颗跑完】的种子；跑到一半被杀的那颗【从头重训】(无中途续训·L226-S)。"
