#!/usr/bin/env bash
# ============================================================================
# TRB 第二条腿 A/B：C_REACH 温和剂量扫(0.5/0.35)  ——  2 臂 × 5 种子{0,2,4,5,6}=10 run
#   `03` L175/L176/L177 · user 2026-07-11 拍方向 ·
#   ⚠️【rate_dock 臂已砍·`03` L177 诊断复审 REFUTED】：连续 RL 策略转向 ω 被动作箱硬卡在
#     ±0.018rad/s(=10.31°/步·usv_continuous_shield.py:115-117·故意=离散臂同权限)·策略已打满饱和·
#     物理 0.03(17.19°)只有紧急控制器够到=RL 策略够不着·rate_dock(改奖励)动不了动作箱上界=无物可放。
#     真因=朝向捕获/目标承诺(价值)·c_reach 已证是对症杠杆(s5 2.5→97.5)。故只烧 c_reach 扫。
#   逐字对齐 0710 golden(治抖ON=rate_weight1.0 · well_B200 · well_X200 · park20 ·
#     ppo · ent0.01 · 5M · manifest_hocr_200 · 40测试)——唯一变量 = c_reach。
#   对照臂 = 第一条腿基线 C_REACH=1.5（已有 `结果0710-22:00-10种子最优方案`·别重跑）。
#   ⚠️ 已逐字预检(本机 config-parse·3臂旋钮落地+rate_dock护栏·2026-07-11)。
#   ⚠️ 服务器【必须先同步整个 代码/ 文件夹】(含 rate_dock 门控代码=A/B探针后新增·服务器旧码没有！
#      + run_step4e 新护栏)。distinct TAG(含 ppo)·别复用旧 TAG 续跑(config_sig 含 c_reach→会重启白烧)。
# ============================================================================
# 用法：
#   bash 代码/run_leg2_scan.sh                 # 无参 → 一次性起全部 15 个 screen（见底部并发说明）
#   bash 代码/run_leg2_scan.sh <arm> <seed>    # 单跑一个(screen 内部调用·或手动排障)
#      arm ∈ {creach05, creach035}   （ratedock 保留在函数里但【已砍·不进默认启动】·`03` L177）
# ============================================================================
set -u
cd "$(dirname "$0")/.." || { echo "cd 到 ~/trb 失败"; exit 1; }   # 代码/ 的上一级(=~/trb·结果/ 落这)
mkdir -p 结果
PY=/root/miniconda3/bin/python    # 服务器 AutoDL·screen 非交互不激活 conda → 绝对路径（本机测试改 /opt/miniconda3/envs/trb/bin/python）
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

# ---- 单臂单种子（前台跑·由 screen 包裹）----
run_one () {
  local ARM="$1" S="$2"
  local TAG ARMENV
  case "$ARM" in
    creach05)  TAG="_L2creach05_ppo_s$S";  ARMENV=(STEP4E_C_REACH=0.5) ;;
    creach035) TAG="_L2creach035_ppo_s$S"; ARMENV=(STEP4E_C_REACH=0.35) ;;
    ratedock)  TAG="_L2ratedock_ppo_s$S";  ARMENV=(STEP4E_DOCK_R=350 STEP4E_RATE_DOCK=0.0) ;;   # 【已砍·`03` L177 REFUTED·留作手动排障勿默认烧·rate_dock 动不了 RL 转向箱上界】
    *) echo "未知 arm: $ARM (须 creach05|creach035|ratedock)"; return 1 ;;
  esac
  env \
    STEP4E_SMOKE=0 STEP4E_NTOTAL=200 STEP4E_STEPS=5000000 STEP4E_NSEG=10 STEP4E_LOG_CURVES=1 \
    STEP4E_MANIFEST=$HOME/trb/balanced_pool/manifest_hocr_200.json \
    STEP4E_BALANCED_DIR=$HOME/trb/balanced_pool STEP4E_SDIR=$HOME/trb/scenarios \
    STEP4E_WELL_B=200 STEP4E_SHAPING_RADIUS=500 STEP4E_WELL_X=200 STEP4E_XTRACK_RADIUS=80 \
    STEP4E_PARK_W=20 STEP4E_PARK_RADIUS=400 STEP4E_PARK_VTARGET=4 \
    STEP4E_RATE_W=1.0 \
    STEP4E_CONTINUOUS_ALGO=ppo STEP4E_ENT_START=0.01 STEP4E_ENT_END=0.01 \
    STEP4E_PARTIES=Continuous-safe STEP4E_SEEDS=$S STEP4E_TAG=$TAG \
    "${ARMENV[@]}" \
    "$PY" -B 代码/run_step4e.py 2>&1 | tee "结果/${TAG}.log"
}

# ---- 有参 = 单跑；无参 = 起全部 15 screen ----
if [ "$#" -ge 2 ]; then
  run_one "$1" "$2"
  exit 0
fi

SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
for ARM in creach05 creach035; do        # 只 2 个 c_reach 臂（rate_dock 已砍·`03` L177）
  for S in 0 2 4 5 6; do
    screen -dmS "l2_${ARM}_s${S}" bash "$SELF" "$ARM" "$S"
  done
done
echo "已起 10 个 screen(2臂×5种子)。screen -ls 查看；tail -f 结果/_L2creach0*_ppo_s*.log 看进度。"
# ============================================================================
# 并发说明：10 run × PPO n_envs=8 = 80 核当量 > 32vCPU(≈2.5× 超订)——【与第一条腿 10 并发同款】跑通过。
#   接受超订一次起全(同第一条腿)；据 top 看 CPU%·>90% 持续爬得慢就杀几个等等再起。
#   续跑：PPO 支持同 TAG 从 checkpoint 续(别删 ckpt)；断网/被杀重敲同一条=自动续。
# 回传：结果/step4e_partial_L2{creach05,creach035}_ppo_s{0,2,4,5,6}.jsonl(+log) → 主窗口配对判读。
# ============================================================================
