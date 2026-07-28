#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════════════════
# L231 三臂实验：治我们输掉的两个指标（转艏增量 / 违规per局）。**从零 5M · 小集 manifest_hocr_200**。
#
#   臂 A  `_A231betaPpoS$S`   只换动作分布：STEP4E_ACT_DIST=beta            → 治【转艏增量】
#   臂 B  `_B231gwsymPpoS$S`  只改状态机让路入口：STEP4E_GW_ENTRY=symmetric → 治【违规/局】的让路那半
#   臂 C  `_C231bothPpoS$S`   两个都上                                      → 看叠加
#
# 对照【不用跑】= 已有金标 `Continuous-safe_s*_L1rateON_ppo_s*`（从零 5.08M · 10 种子 · 同配方同数据集）。
# 配方逐字 = `run_leg1_rate.sh`（金标），**唯一差别 = 上面那两个开关**（连 NSEG=10 的步网格都不动，
# 这样三臂的学习曲线能与金标直接叠在同一张图上·`03` L58#2）。
#
# 立项依据（全部本机实测·`03` L230 / L231·零烧卡）：
#   · 真实观测下策略确定性均值 **76~84% 的步在动作箱外**（中位 2.0~3.2× 半箱）⟹ 压 σ / 退火熵 / 事后滤波全无效，
#     **必须换有界分布**（L231-C1）。
#   · 状态机让路入口改对称后，同轨迹上让路覆盖 **3.4~5.6×**、对评分器口径的覆盖率 27%→94%（L231-C2）。
#   · 两个开关默认关时**训练出的网络权重与改动前逐元素差 = 0.000e+00**（L231-D）。
#
# ⚠️ 换分布 ⟹ 旧存档灌不进去 ⟹ 三臂**都必须从零**（本脚本不设 WARMSTART）。
# ⚠️ TAG 三臂互不相同 ⟹ 各写各的 jsonl，不会 config_conflict 混写。
#
# 用法：  bash run_l231_arms.sh [并发上限]                          # 默认 = A/B/C 三臂 × 种子 0 1 2
#         SEEDS="3 4 5 6 7 8 9" ARMS="C" bash run_l231_arms.sh 7   # 只补 C 臂 7 颗种子
#         SEEDS="5 6" ARMS="A" bash run_l231_arms.sh 2              # 只补 A 臂那两颗崩种子
# ══════════════════════════════════════════════════════════════════════════════════════════
set -uo pipefail

CODE_DIR="/root/trb/代码"
[ -d "$CODE_DIR" ] || { echo "❌ CODE_DIR 不存在：$CODE_DIR"; exit 1; }
RES_DIR="$(cd "$CODE_DIR/.." && pwd)/结果"       # run_step4e 恒写到 <代码>/../结果·自动派生·防路径不一致
PY="/root/miniconda3/bin/python"
MANIFEST="$HOME/trb/balanced_pool/manifest_hocr_200.json"
BALANCED="$HOME/trb/balanced_pool"
SDIR="$HOME/trb/scenarios"

# 🆕 种子与臂都可用环境变量覆盖（不用改文件）：
#   SEEDS="3 4 5 6 7 8 9"  ARMS="C"  bash run_l231_arms.sh 7
#   ARMS 取值 = A/B/C 的任意组合（空格分隔）；只冒烟被选中的臂。
SEEDS="${SEEDS:-0 1 2}"
ARMS="${ARMS:-A B C}"
KMAX="${1:-9}"                                   # 并发上限（32 核 + 每 run NENVS=8 ⟹ 9 路≈2.3× 超订·与既往 10 路同量级）
for _a in $ARMS; do
  case "$_a" in A|B|C) ;; *) echo "❌ ARMS 只能是 A/B/C 的组合，得到 '$_a'"; exit 1 ;; esac
done

# ---- 配方：逐字 = run_leg1_rate.sh（金标 L1rateON），差异只有各臂自己的那一个开关 ----
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export STEP4E_SMOKE=0 STEP4E_NTOTAL=200 STEP4E_STEPS=5000000 STEP4E_NSEG=10 STEP4E_LOG_CURVES=1
export STEP4E_MANIFEST="$MANIFEST" STEP4E_BALANCED_DIR="$BALANCED" STEP4E_SDIR="$SDIR"
export STEP4E_WELL_B=200 STEP4E_SHAPING_RADIUS=500 STEP4E_WELL_X=200 STEP4E_XTRACK_RADIUS=80
export STEP4E_PARK_W=20 STEP4E_PARK_RADIUS=400 STEP4E_PARK_VTARGET=4
export STEP4E_RATE_W=1.0                          # 治抖 ON（金标同款·**不能关**：Beta 立项的机制就是"让这个罚项终于有着力点"）
export STEP4E_CONTINUOUS_ALGO=ppo STEP4E_ENT_START=0.01 STEP4E_ENT_END=0.01 STEP4E_NENVS=8
export STEP4E_PARTIES=Continuous-safe
export PY RES_DIR

echo "===== [闸门 0] 路径 + 代码已同步（含 L231 新开关）====="
[ -d "$RES_DIR" ]  || { echo "❌ RES_DIR 不存在：$RES_DIR"; exit 1; }
[ -f "$MANIFEST" ] || { echo "❌ manifest 不存在：$MANIFEST"; exit 1; }
[ -f "$CODE_DIR/trb_env/usv_action_dist.py" ] || { echo "❌ 缺 trb_env/usv_action_dist.py＝没同步新模块 → 先【同步整个 代码 文件夹】再跑"; exit 1; }
grep -q "STEP4E_ACT_DIST" "$CODE_DIR/run_step4e.py"  || { echo "❌ run_step4e.py 无 STEP4E_ACT_DIST＝旧版·先同步"; exit 1; }
grep -q "STEP4E_GW_ENTRY" "$CODE_DIR/run_step4e.py"  || { echo "❌ run_step4e.py 无 STEP4E_GW_ENTRY＝旧版·先同步"; exit 1; }
grep -q "gw_entry" "$CODE_DIR/trb_env/usv_colregs.py" || { echo "❌ usv_colregs.py 无 gw_entry＝旧版·先同步"; exit 1; }
grep -q "gw_entry=self._sc.gw_entry" "$CODE_DIR/trb_env/usv_projection.py" || { echo "❌ usv_projection.py 前瞻未继承档位＝旧版·先同步"; exit 1; }
echo "  ✅ 路径对 · 新模块在 · 两个开关都在 · 前瞻一致性补丁在"

echo "===== [闸门 0.5] 新开关回归套件（45 项·~1-2 分钟·不烧卡）====="
#   ⚠️ 必须 `env -u` 清掉本脚本 export 的 STEP4E_*：自检里会 import run_step4e，而它模块级有 PPO 隔离闸门
#      （TAG 为空 + CONTINUOUS_ALGO=ppo → SystemExit）。本机演练实测撞到过，别删这行 env -u。
env -u STEP4E_CONTINUOUS_ALGO -u STEP4E_TAG -u STEP4E_SMOKE -u STEP4E_PARTIES -u STEP4E_STEPS -u STEP4E_NSEG \
    STEP4E_SDIR="$SDIR" "$PY" -B "$CODE_DIR/tests/test_act_dist_gw_entry.py" > "$RES_DIR/_l231_selftest.log" 2>&1 \
  || { echo "❌ 新开关回归没全绿 → 别烧卡，看 $RES_DIR/_l231_selftest.log"; tail -5 "$RES_DIR/_l231_selftest.log"; exit 1; }
tail -1 "$RES_DIR/_l231_selftest.log"

echo "===== [闸门 1] 预下载场景（缓存则秒过）====="
STEP4E_SMOKE=0 STEP4E_DOWNLOAD_ONLY=1 STEP4E_SEEDS=0 STEP4E_TAG=_predl_l231ppo "$PY" -B "$CODE_DIR/run_step4e.py" \
  || { echo "❌ 预下载失败（查网络）"; exit 1; }

echo "===== [闸门 2] 冒烟 3 臂各 1 次（~2-3min）·验开关【真落地】而不是设了没生效 ====="
cd "$CODE_DIR"
smoke_one () {   # $1=TAG $2..=开关
  local T="$1"; shift
  local SMK="$RES_DIR/step4e_partial${T}.jsonl"   # ⚠️ TAG 自带前导下划线（run_step4e 的命名是 step4e_partial<TAG>.jsonl）——本机演练抓出来的：多写一条下划线 → 闸门 2 永远误报"开关没生效"
  rm -f "$SMK"
  env "$@" STEP4E_SMOKE=1 STEP4E_STEPS=8000 STEP4E_NSEG=1 STEP4E_SEEDS=0 STEP4E_TAG="$T" \
    "$PY" -B run_step4e.py > "$RES_DIR/_${T}.log" 2>&1 \
    || { echo "❌ 冒烟跑崩：$T（看 $RES_DIR/_${T}.log）"; tail -20 "$RES_DIR/_${T}.log"; exit 1; }
  echo "$SMK"
}
for _a in $ARMS; do
  case "$_a" in
    A) SMK="$(smoke_one _smkA231ppo STEP4E_ACT_DIST=beta)" || exit 1
       grep -q '"act_dist": "beta"'  "$SMK" || { echo "❌ 臂A 冒烟没见 act_dist=beta → 开关【静默没生效】·别烧全量"; exit 1; }
       grep -q '"gw_entry": "paper"' "$SMK" || { echo "❌ 臂A 的 gw_entry 应为 paper（单变量）"; exit 1; } ;;
    B) SMK="$(smoke_one _smkB231ppo STEP4E_GW_ENTRY=symmetric)" || exit 1
       grep -q '"gw_entry": "symmetric"' "$SMK" || { echo "❌ 臂B 冒烟没见 gw_entry=symmetric → 开关【静默没生效】·别烧全量"; exit 1; }
       grep -q '"act_dist": "gauss"'     "$SMK" || { echo "❌ 臂B 的 act_dist 应为 gauss（单变量）"; exit 1; } ;;
    C) SMK="$(smoke_one _smkC231ppo STEP4E_ACT_DIST=beta STEP4E_GW_ENTRY=symmetric)" || exit 1
       grep -q '"act_dist": "beta"'      "$SMK" || { echo "❌ 臂C 冒烟没见 act_dist=beta·别烧全量"; exit 1; }
       grep -q '"gw_entry": "symmetric"' "$SMK" || { echo "❌ 臂C 冒烟没见 gw_entry=symmetric·别烧全量"; exit 1; } ;;
  esac
  grep -q '"rate_weight": 1.0' "$SMK" || { echo "❌ 冒烟未见 rate_weight=1.0 → 治抖没真开（Beta 的机制就靠它）·别烧全量"; exit 1; }
done
echo "  ✅ 选中的臂（$ARMS）冒烟全过：开关逐条真落地 + 治抖 ON"

echo "===== [闸门 3] 起 $(echo $SEEDS|wc -w) 种子 × $(echo $ARMS|wc -w) 臂($ARMS) = $(( $(echo $SEEDS|wc -w) * $(echo $ARMS|wc -w) )) 个 run（5M·并发≤$KMAX）====="
run_one () {
  local ARM="$1" S="$2"; shift 2
  local T
  case "$ARM" in
    A) T="_A231betaPpoS$S" ;;
    B) T="_B231gwsymPpoS$S" ;;
    C) T="_C231bothPpoS$S" ;;
  esac
  env "$@" STEP4E_SEEDS="$S" STEP4E_TAG="$T" "$PY" -B run_step4e.py > "$RES_DIR/${T}.log" 2>&1 \
    && echo "  [完] $T" || echo "  [⚠️失败] $T（看 $RES_DIR/${T}.log）"
}
export -f run_one
export STEP4E_KEEP_SEGMENTS=1                     # 🆕 `03` L236-D②（user 2026-07-28 拍板"以后所有训练都开"）：每段存档另留一份到 checkpoints/segments/
#   为什么：主存档是覆盖式的（每段盖掉上一段）⟹ Discrete-safe s0 那种"第 7 段 100%、末两段崩回 5%"的存档事后拿不回来、只能重训。
#   开了之后想换"最好存档"口径 / 查崩溃前后 / 补学习曲线上的真实测试集数，都不必重训。副本在**子目录**⟹ 不会被重评自动发现（要评须显式点名）。
for S in $SEEDS; do
  for _arm in $ARMS; do
    case "$_arm" in
      A) run_one A "$S" STEP4E_ACT_DIST=beta & ;;
      B) run_one B "$S" STEP4E_GW_ENTRY=symmetric & ;;
      C) run_one C "$S" STEP4E_ACT_DIST=beta STEP4E_GW_ENTRY=symmetric & ;;
    esac
    while [ "$(jobs -rp | wc -l)" -ge "$KMAX" ]; do sleep 20; done
  done
done
wait
echo "===== 全部结束 ====="
ls -la "$RES_DIR"/checkpoints/Continuous-safe_s*_[ABC]231*Ppo*.zip 2>/dev/null | wc -l
