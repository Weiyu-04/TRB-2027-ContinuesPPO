#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════════════════
# TRB 2027 **正式实验**（`03` L240 定稿方案 · `Paper/正式实验/README.md`）
#
#   9 条臂 × N 颗种子 × 10.16M(NSEG=20) 或 15.24M(NSEG=30) 步（每段恒 507,904）· 官方 1300 训练集 · 分段存档全程开
#
# 用法：
#   SPEED=1                    bash run_formal_2027.sh 4   # 先测速（9 臂各 ~6.5 万步·跑完即退·不起全量）
#   NSEG=30 ARMS="ours" SEEDS="0 1 2 3" bash run_formal_2027.sh 4   # T1：只起主线
#   NSEG=30 SEEDS="0 1 2 3"            bash run_formal_2027.sh 4   # 全 9 臂 × 该机负责的种子列
#   RESUME=1 <同一条命令>                                        # 断了续跑（撞名闸放行·配方须逐字相同）
#
# 🔴 **按种子分机器，绝不按臂分机器**：同种子配对比较是我们的主统计工具，必须落在同一台机器内。
#    按臂分机器 = 机器变成混淆变量，整套对比作废。
# 🔴 **评估另说**：跑完全部臂后，重评必须**一台机器一趟**跑完（`04 §2`）。
# ══════════════════════════════════════════════════════════════════════════════════════════
set -uo pipefail

CODE_DIR="/root/trb/代码"
[ -d "$CODE_DIR" ] || { echo "❌ CODE_DIR 不存在：$CODE_DIR"; exit 1; }
RES_DIR="$(cd "$CODE_DIR/.." && pwd)/结果"
PY="/root/miniconda3/bin/python"
MANIFEST="$HOME/trb/balanced_pool/manifest_official_1300.json"
BALANCED="$HOME/trb/balanced_pool"
SDIR="$HOME/trb/scenarios"

SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
ARMS="${ARMS:-ours disc base rr uns ush ab0 abB abG}"
KMAX="${1:-4}"                       # 每 run 8 环境 ⟹ 32 核最多 4 路（`04 §1` 定核纪律）

# ── 全臂共用（**任何一条臂都不许改这些**）──────────────────────────────────────
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export STEP4E_SMOKE=0
# 🆕 `03` L243-改①：段数可覆盖，但**只接受 20 / 30 两档**（保每段步数逐字相同 = 同网格 = 可叠图、可续训）。
#   20 段 → STEP4E_STEPS=10,000,000 ⟹ seg = 10000000//20 = 500,000
#   30 段 → STEP4E_STEPS=15,000,000 ⟹ seg = 15000000//30 = 500,000   ← **与 20 段逐字相同**
#   两档 seg 都被 SB3 按 rollout(2048×8=16384) 取整到 **507,904** ⟹ 30 段跑到第 20 段时，模型状态与 20 段跑法逐位相同
#   （前提：lr 退火关 + ent_start==ent_end + 惩罚退火关 —— 本脚本三条都满足，逐条读码见 `03` L243-改①）。
#   ⟹ 「30 段起跑、够了就在第 20 段停/按 BUDGET_SEG=20 报数」是**零代码成本**的双向弹性，不必等接力训练模块。
# 🔴 同一张报数表里全部臂必须同 NSEG。改它 = 改预算 = 必须 user 拍板。
NSEG="${NSEG:-20}"
case "$NSEG" in
  20) _TOT=10000000 ;;
  30) _TOT=15000000 ;;
  *)  echo "❌ NSEG 只接受 20 或 30（其他值会改掉每段步数 507,904 = 破坏同网格/可叠图/可续训）"; exit 1 ;;
esac
export STEP4E_STEPS=$_TOT STEP4E_NSEG=$NSEG        # ⟹ 每段 507,904 步（与既有 5.08M 臂同网格）· $NSEG 个存档候选
export STEP4E_NTOTAL=2000                          # manifest 模式下只影响非 manifest 路径；设 2000 保底不 striding
export STEP4E_MANIFEST="$MANIFEST" STEP4E_BALANCED_DIR="$BALANCED" STEP4E_SDIR="$SDIR"
export STEP4E_NENVS=8 STEP4E_ENT_START=0.01 STEP4E_ENT_END=0.01 STEP4E_ENT_FRAC=0.6
export STEP4E_LOG_CURVES=1                         # 🔴 默认是关的！离散臂漏了就永远没有内部曲线（A 类量）
export STEP4E_KEEP_SEGMENTS=1                      # 🔴 分段存档 —— "验证集挑最佳存档"这个定稿口径的唯一依据
# 🆕 `03` L243-续3：设成 **16384 = 2048 × 8**（PPO 一个 rollout 的步数）。
#   原因：**两条臂的 A 类量采集窗口本来不一样** —— 离散臂的 `_CurveLogger` 是每个 rollout 结束时
#   快照一次（16,384 步），连续臂的 `_SACCurveLogger` 是按 `record_every` 快照（原来 25,000 步）
#   ⟹ 两臂的 `roll_*` 曲线不在同一个横轴网格上，**要叠进同一张学习曲线图就得先重采样**。
#   设成 16384 之后两臂同网格、可直接叠图。**只影响记录密度，不碰训练**（PPO 不吃这个键，
#   它只喂连续臂的曲线记录器；`record_every` 也不在 `config_sig` 里 ⟹ 不影响续训匹配）。
export STEP4E_RECORD_EVERY=16384
export PY RES_DIR

# ── 逐臂配方（**只列与共用项不同的**；空 = 全用默认）──────────────────────────
#   ⚠️ TAG 必须含 ppo（`run_step4e.py:405-410` 的 PPO 隔离闸），否则起跑即拦。
arm_cfg () {
  case "$1" in
    # ① 主角：两把钥匙 + 盾 + 连续专属塑形
    ours) echo "TAG=_F240oursPpoS PARTIES=Continuous-safe SHIELD=1 ACT=beta GW=symmetric SHAPE=full" ;;
    # ② 对标论文（离散 + 动作屏蔽）· ③④ 两条离散基线
    disc) echo "TAG=_F240discPpoS PARTIES=Discrete-safe  SHIELD=- ACT=-    GW=-         SHAPE=none" ;;
    base) echo "TAG=_F240basePpoS PARTIES=Base           SHIELD=- ACT=-    GW=-         SHAPE=none" ;;
    rr)   echo "TAG=_F240rrPpoS   PARTIES=Rule-reward    SHIELD=- ACT=-    GW=-         SHAPE=none" ;;
    # ⑤⑥ 孪生对：**逐字相同、只差盾** ⟹ 唯一干净回答"盾值多少"的一对
    #    极简配方（高斯/paper/零塑形）是无盾臂**唯一合法**形态（run_step4e.py:223/233/243 三道闸）
    #    且与离散 Base 的奖励口径一致 ⟹ ⑤vs③ 干净回答"离散换连续值多少"
    uns)  echo "TAG=_F240unsPpoS  PARTIES=Continuous-safe SHIELD=0 ACT=gauss GW=paper   SHAPE=none" ;;
    ush)  echo "TAG=_F240ushPpoS  PARTIES=Continuous-safe SHIELD=1 ACT=gauss GW=paper   SHAPE=none" ;;
    # ⑦⑧⑨ 消融三臂（都带盾 + 全套塑形，只差两把钥匙的开关组合）
    ab0)  echo "TAG=_F240ab0PpoS  PARTIES=Continuous-safe SHIELD=1 ACT=gauss GW=paper   SHAPE=full" ;;
    abB)  echo "TAG=_F240abBPpoS  PARTIES=Continuous-safe SHIELD=1 ACT=beta  GW=paper   SHAPE=full" ;;
    abG)  echo "TAG=_F240abGPpoS  PARTIES=Continuous-safe SHIELD=1 ACT=gauss GW=symmetric SHAPE=full" ;;
    *) echo ""; ;;
  esac
}

arm_env () {   # 把 arm_cfg 的紧凑描述展开成 env 赋值串
  local cfg; cfg=$(arm_cfg "$1"); [ -n "$cfg" ] || { echo "❌ 未知臂 $1" >&2; return 1; }
  local TAG PARTIES SHIELD ACT GW SHAPE; eval "$cfg"
  local e="STEP4E_PARTIES=$PARTIES"
  [ "$PARTIES" = "Continuous-safe" ] && e="$e STEP4E_CONTINUOUS_ALGO=ppo"
  [ "$SHIELD" != "-" ] && e="$e STEP4E_CONTINUOUS_SHIELD=$SHIELD"
  [ "$ACT" != "-" ]    && e="$e STEP4E_ACT_DIST=$ACT"
  # gw_entry=paper 是默认；只在 symmetric 时显式设（设 paper 也可，但少一个变量少一处出错）
  [ "$GW" = "symmetric" ] && e="$e STEP4E_GW_ENTRY=symmetric"
  if [ "$SHAPE" = "full" ]; then     # 连续专属塑形（逐字 = 既有金标/C 臂配方）
    e="$e STEP4E_WELL_B=200 STEP4E_SHAPING_RADIUS=500 STEP4E_WELL_X=200 STEP4E_XTRACK_RADIUS=80"
    e="$e STEP4E_PARK_W=20 STEP4E_PARK_RADIUS=400 STEP4E_PARK_VTARGET=4 STEP4E_RATE_W=1.0"
  else                               # 零塑形：离散三臂的既有口径，也是无盾臂唯一合法形态
    e="$e STEP4E_WELL_B=0 STEP4E_SHAPING_RADIUS=500 STEP4E_WELL_X=0 STEP4E_XTRACK_RADIUS=80"
    e="$e STEP4E_PARK_W=0 STEP4E_RATE_W=0"
  fi
  echo "$e"
}
arm_tag () { local cfg; cfg=$(arm_cfg "$1"); local TAG PARTIES SHIELD ACT GW SHAPE; eval "$cfg"; echo "$TAG"; }
arm_party () { local cfg; cfg=$(arm_cfg "$1"); local TAG PARTIES SHIELD ACT GW SHAPE; eval "$cfg"; echo "$PARTIES"; }

N_RUN=$(( $(echo $ARMS|wc -w) * $(echo $SEEDS|wc -w) ))
echo "═══ TRB 2027 正式实验 ═══  臂=[$ARMS]  种子=[$SEEDS]  ⟹ $N_RUN 个 run · 并发≤$KMAX"
echo "    $NSEG 段 × 507,904 = $((NSEG*507904)) 步 · 官方 1300 训练集 · 分段存档开 · 内部曲线开"

echo "===== [闸门 0] 代码已同步（含本轮三项新功能）====="
for pat in "STEP4E_KEEP_SEGMENTS" "def _archive_segment" "class _RolloutStats" "STEP4E_ACT_DIST" "STEP4E_GW_ENTRY"; do
  grep -q "$pat" "$CODE_DIR/run_step4e.py" || { echo "❌ run_step4e.py 里没有【$pat】＝代码没同步 → 同步整个 代码 文件夹再跑"; exit 1; }
done
[ -f "$CODE_DIR/trb_env/usv_action_dist.py" ] || { echo "❌ 缺 trb_env/usv_action_dist.py"; exit 1; }
grep -q "gw_entry=self._sc.gw_entry" "$CODE_DIR/trb_env/usv_projection.py" || { echo "❌ usv_projection.py 是旧版"; exit 1; }
echo "  ✅ 三项新功能（分段存档 / rollout 采集 / 两把钥匙）都在"

echo "===== [闸门 0.2] 记录代码指纹（可复现性·`03` L240）====="
( cd "$CODE_DIR/.." && git rev-parse HEAD 2>/dev/null && git status --porcelain 2>/dev/null | head -5 ) \
  > "$RES_DIR/_formal_code_fingerprint.txt" 2>&1 || true
for f in run_step4e.py trb_env/usv_projection.py trb_env/usv_colregs.py trb_env/usv_continuous_shield.py trb_env/evaluate.py; do
  "$PY" -c "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest()[:16], sys.argv[1])" \
    "$CODE_DIR/$f" >> "$RES_DIR/_formal_code_fingerprint.txt"
done
cat "$RES_DIR/_formal_code_fingerprint.txt"; echo "  ✅ 指纹已落盘（论文方法节要写）"

echo "===== [闸门 0.3] 清单自洽（六项必须全 ✅）====="
"$PY" -B "$CODE_DIR/make_official_manifest.py" --check "$MANIFEST" || { echo "❌ 清单校验没过 → 别烧卡"; exit 1; }

echo "===== [闸门 0.6] 存档不撞名（$N_RUN 个目标名 vs 既有全部存档）====="
# 🆕 `03` L243-§6（复审抓出）：这道闸原本让「断了再敲一次同一条命令续跑」**跑不起来** ——
#   run_step4e 每段都写主存档，所以只要跑过一段，目标 .zip 就已存在 ⟹ 本闸判撞名 ⟹ exit 1。
#   而 `04 §1` 白纸黑字写着"被杀/断网就再敲一次同一条启动命令，已完成的自动跳过、续跑"。
#   ⟹ 加 RESUME=1 显式放行（**默认仍是硬拦**，一个字节的默认行为都没变）。
#   放行时不静默：把每个已存在的存档跑到第几段、目标几段，逐条打出来给人看。
RESUME="${RESUME:-0}"
CLASH=0
for A in $ARMS; do
  T=$(arm_tag "$A"); P=$(arm_party "$A")
  for S in $SEEDS; do
    HIT=$(find "$(cd "$CODE_DIR/.." && pwd)" -path "*/checkpoints/${P}_s${S}${T}${S}.zip" -print -quit 2>/dev/null)
    [ -z "$HIT" ] && continue
    if [ "$RESUME" = "1" ]; then
      PRG="${HIT%.zip}.progress.json"
      # seg_done 是 0 起的段下标 ⟹ 已完成段数 = seg_done+1；同时打 num_timesteps 与目标，肉眼可核
      DONE=$("$PY" -c "import json,sys
d=json.load(open(sys.argv[1],encoding='utf-8'))
c=d.get('config_sig') or {}
print(f\"{int(d.get('seg_done',-1))+1}/{d.get('n_seg', c.get('n_seg','?'))} 段 · {d.get('num_timesteps','?')}/{d.get('total_steps', c.get('total_steps','?'))} 步\")" "$PRG" 2>/dev/null || echo "读不到 progress.json")
      echo "  ↻ 续跑：${P}_s${S}${T}${S}  已完成 $DONE"
    else
      echo "  ❌ 已存在：$HIT"; CLASH=$((CLASH+1))
    fi
  done
done
if [ "$RESUME" = "1" ]; then
  echo "  ⚠️ RESUME=1：撞名闸已放行。续跑靠 run_step4e 的 Layer-2 config_sig 全等匹配 ——"
  echo "     🔴 本次的 STEP4E_STEPS / NSEG / 逐臂配方必须与上次**逐字相同**，任一不同会【从 0 重训】而不报错。"
else
  [ "$CLASH" -eq 0 ] || { echo "❌ $CLASH 个存档会被【静默覆盖】→ 换 TAG / 先归档 / 或确认是续跑后加 RESUME=1，别烧"; exit 1; }
  echo "  ✅ $N_RUN 个目标存档名均不冲突"
fi

echo "===== [闸门 1] 预下载场景（官方 1400·缓存则秒过）====="
STEP4E_DOWNLOAD_ONLY=1 STEP4E_SEEDS=0 STEP4E_TAG=_predl_f240 "$PY" -B "$CODE_DIR/run_step4e.py" \
  || { echo "❌ 预下载失败（查网络）"; exit 1; }

echo "===== [闸门 2] 逐臂冒烟（每条臂 1 次·验配方真落地 + 分段存档真写出来 + **实测内存**）====="
cd "$CODE_DIR"
# 🆕 `03` L243-续2（user 2026-07-29 说一台机能跑 10 路 ⟹ 内存成了新的头号风险）：
#   `make_vec_env(subproc=True)` 让**每个 worker 各自加载整份场景池**（`usv_scenarios.py:159-166`）
#   ⟹ 单 run 内存 ≈ n_envs × 场景数 × 单场景内存。项目自己实测过 **0.20 MB/场景**
#   （`03` L233：0.20 × 1300 × 8 worker × 3 种子 ≈ 6.2 GB ⟹ **单 run ≈ 2.1 GB 只算场景池**，
#   加上每进程的 torch/numpy 基线，真实占用要大得多）。K=10 时是 10 倍。
#   OOM 的后果特别恶劣：worker 被杀 → 那条 run 静默失败 → `run_one` 只打一行 [⚠️失败] 就继续
#   ⟹ 两天后才发现少了几条臂。**所以这里不估、直接量**：冒烟跑的就是真数据集 + 真 NENVS，
#   它的峰值常驻内存 ≈ 正式 run 的峰值（场景池在启动时就全载进来了）。
RSSLOG="$RES_DIR/_formal_rss.txt"; : > "$RSSLOG"
( while :; do
    ps -eo rss=,args= 2>/dev/null | awk '/run_step4e\.py/ && !/awk/ {s+=$1} END{print s+0}' >> "$RSSLOG"
    sleep 2
  done ) & RSS_SAMPLER=$!
trap 'kill "$RSS_SAMPLER" 2>/dev/null' EXIT
for A in $ARMS; do
  T="_smkF240$A"; SMK="$RES_DIR/step4e_partial${T}.jsonl"; SEGDIR="$RES_DIR/checkpoints/segments"
  rm -f "$SMK"; rm -f "$SEGDIR"/*"${T}"@s* 2>/dev/null
  # shellcheck disable=SC2046
  env $(arm_env "$A") STEP4E_SMOKE=1 STEP4E_STEPS=8000 STEP4E_NSEG=2 STEP4E_SEEDS=0 STEP4E_TAG="$T" \
    "$PY" -B run_step4e.py > "$RES_DIR/_${T}.log" 2>&1 \
    || { echo "  ❌ 臂 $A 冒烟跑崩（看 $RES_DIR/_${T}.log）"; tail -15 "$RES_DIR/_${T}.log"; exit 1; }
  # 🔴 三样必须在 jsonl 里看得见（防"设了开关其实没生效"，`03` L192 那族坑）
  grep -q '"keep_segments": true' "$SMK" || { echo "  ❌ 臂 $A：keep_segments 没生效"; exit 1; }
  grep -q 'official_1300'         "$SMK" || { echo "  ❌ 臂 $A：训练集不是官方 1300"; exit 1; }
  grep -q '"roll_steps"'          "$SMK" || { echo "  ❌ 臂 $A：rollout 采集没生效（A 类量会全丢）"; exit 1; }
  # 🆕 `03` L243-§5（复审抓出）：光看 `roll_steps` **拦不住** —— `_feed_rollout_stats` 整段 try/except，
  #   若取 u_desired 那截第一步就抛，`roll_steps` 照样非零、而**动作类 A 类量全程为空**，跑完三天才发现。
  #   ⟹ 连续臂必须另验 `roll_n_act > 0`（离散臂结构上没有 u_desired，恒 0 才对，不能要求它非零）。
  if [ "$(arm_party "$A")" = "Continuous-safe" ]; then
    "$PY" -c "
import json,sys
n=0
for line in open(sys.argv[1],encoding='utf-8'):
    line=line.strip()
    if not line: continue
    r=json.loads(line)
    for c in (r.get('curves') or []):
        n=max(n,int(c.get('roll_n_act') or 0))
sys.exit(0 if n>0 else 1)" "$SMK" \
      || { echo "  ❌ 臂 $A：roll_n_act 全程 0 ⟹ 打满舵率/盾改写量这些 A 类量【采不到】（采集被 try/except 静默吞了）"; exit 1; }
  fi
  # 🔴 最关键：分段副本必须**真的躺在磁盘上**，且份数 == NSEG。这条至今从没在真实 run 里验证过。
  NZ=$(ls "$SEGDIR"/*"${T}"@s*.zip 2>/dev/null | wc -l)
  [ "$NZ" -eq 2 ] || { echo "  ❌ 臂 $A：segments/ 里 .zip 有 $NZ 份（应 2 = NSEG）⟹ 分段存档没落地，别烧全量"; exit 1; }
  rm -f "$SEGDIR"/*"${T}"@s*                    # 只清本冒烟 TAG 的（**绝不整目录删**）
  echo "  ✅ 臂 $A：配方对 + 分段存档真落地 + rollout 采集在"
done
kill "$RSS_SAMPLER" 2>/dev/null; trap - EXIT

echo "===== [闸门 2.5] 内存够不够跑 $KMAX 路（**实测·不是估的**）====="
PEAK_KB=$(sort -n "$RSSLOG" | tail -1)
[ -n "$PEAK_KB" ] && [ "$PEAK_KB" -gt 0 ] || { echo "  ⚠️ 没采到内存样本（ps 输出为空？）→ 跳过本闸，**起飞后自己盯 free -g**"; PEAK_KB=0; }
if [ "$PEAK_KB" -gt 0 ]; then
  AVAIL_KB=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)
  TOTAL_KB=$(awk '/^MemTotal:/{print $2}' /proc/meminfo)
  NEED_KB=$(( PEAK_KB * KMAX ))
  # 留 15% 余量：page cache + 分段存档写盘 + 段末评估的临时峰值
  SAFE_KB=$(( AVAIL_KB * 85 / 100 ))
  KFIT=$(( SAFE_KB / PEAK_KB ))
  printf "  单 run 峰值常驻 %.1f GB（冒烟实测·真数据集真 NENVS=%s）\n" "$(echo "$PEAK_KB" | awk '{print $1/1048576}')" "${STEP4E_NENVS:-8}"
  printf "  本机 MemTotal %.0f GB · MemAvailable %.0f GB ⟹ 留 15%% 余量后可跑 **%s 路**\n" \
    "$(echo "$TOTAL_KB" | awk '{print $1/1048576}')" "$(echo "$AVAIL_KB" | awk '{print $1/1048576}')" "$KFIT"
  printf "  本次要 %s 路 ⟹ 需 %.0f GB\n" "$KMAX" "$(echo "$NEED_KB" | awk '{print $1/1048576}')"
  if [ "$NEED_KB" -gt "$SAFE_KB" ]; then
    echo "  ❌ **内存不够**：$KMAX 路会 OOM。OOM 的后果是 worker 被杀 → 那条 run 静默失败 →"
    echo "     两天后才发现少了几条臂（`run_one` 只打一行 [⚠️失败] 就继续跑）。"
    echo "     ⟹ 改用 KMAX=$KFIT 重跑本脚本；或降 STEP4E_NENVS（⚠️ 它进 config_sig，"
    echo "        中途改会让已跑的 run 从 0 重来，**要改就在起跑前一次改到位**）。"
    [ "${MEM_ACK:-0}" = "1" ] || exit 1
    echo "  ⚠️ MEM_ACK=1 强行放行 —— 你自己盯 free -g 和 dmesg 里的 oom-kill。"
  else
    echo "  ✅ 内存够（$KMAX ≤ $KFIT 路）"
  fi
fi

# ══════════════════════════════════════════════════════════════════════════════════════════
# 🆕 [闸门 2.5] 测速档（`03` L243-§7·复审抓出的最大排期风险）—— `SPEED=1` 才跑，跑完就退出，不起全量
#
# 🔴 为什么必须有它：`02` 写的"对标 ≈8.8h / 离散无盾 ≈4.6h"**全是外推，不是实测**。
#    全库 fps 翻一遍的事实：官方 1300 训练集上**只有 `ours` 这个配方被真跑过**（375 fps · D 探针 3 条 run）。
#    而离散臂在大池子上唯一的实测是 0624 的 `Discrete-safe / n_train=1400 / fps=130` ⟹ 折 10.16M = **21.7 小时**，
#    是 8.8h 的 2.5 倍。更要命的是方向相反：**连续臂换大集变快（270→375）、离散臂换大集变慢（236→130）**
#    ⟹ 拿连续臂的比例去推离散臂没有任何依据。排期照 8.8h 排，很可能整整差一天半。
#
# 成本：9 条臂 × 约 6.5 万步，按真实并发跑 ⟹ 约 20~40 分钟。相对 3~4 天的全量，等于零。
# ⚠️ 它**必须按真实并发跑**（KMAX 与全量同值）——单跑测出来的 fps 是没有争抢的假数字。
# ══════════════════════════════════════════════════════════════════════════════════════════
if [ "${SPEED:-0}" = "1" ]; then
  echo "===== [闸门 2.5] 测速档（每臂 ~6.5 万步 · 并发 $KMAX · 跑完即退出，不起全量）====="
  SPD_SEED="${SPEED_SEED:-0}"
  spd_one () {
    local A="$1"; local T="_probeSpd${A}"
    # shellcheck disable=SC2046
    env $(arm_env "$A") STEP4E_STEPS=60000 STEP4E_NSEG=1 STEP4E_KEEP_SEGMENTS=0 \
      STEP4E_SEEDS="$SPD_SEED" STEP4E_TAG="$T" \
      "$PY" -B run_step4e.py > "$RES_DIR/_spd_${A}.log" 2>&1 || echo "  [⚠️测速失败] $A（看 $RES_DIR/_spd_${A}.log）"
  }
  for A in $ARMS; do
    spd_one "$A" &
    while [ "$(jobs -rp | wc -l)" -ge "$KMAX" ]; do sleep 5; done
  done
  wait
  echo
  echo "  臂     fps    训练h/10.16M(20段)   训练h/15.24M(30段)   单段评估s   ⟹ 单run总h(20段)  (30段)"
  for A in $ARMS; do
    "$PY" -c "
import json,sys,os
p=os.path.join(sys.argv[2], 'step4e_partial_probeSpd%s.jsonl' % sys.argv[1])
r=None
for line in open(p,encoding='utf-8'):
    line=line.strip()
    if line and '\"fps\"' in line: r=json.loads(line)
if not r: raise SystemExit('  %-6s 没拿到 fps（看日志）' % sys.argv[1])
f=float(r['fps']); ev=float(r.get('eval_s') or 0.0)
t20=20*507904/f/3600; t30=30*507904/f/3600
print('  %-6s %5.0f %14.2f %20.2f %13.1f %16.2f %8.2f' % (sys.argv[1], f, t20, t30, ev, t20+20*ev/3600, t30+30*ev/3600))
" "$A" "$RES_DIR" 2>&1 | tail -1
  done
  echo
  echo "  🔴 把这张表填进 04 §2 的《跑实验清单》再定列分配。**别用外推值排期。**"
  echo "  🔴 测速产物 TAG 是 _probeSpd*，与正式 TAG(_F240*) 不撞、也不会被 ARM_SPECS 收进报数表；确认后可删。"
  exit 0
fi

echo "===== [闸门 3] 起 $N_RUN 个 run ====="
run_one () {
  local A="$1" S="$2"; local T; T="$(arm_tag "$A")$S"
  # shellcheck disable=SC2046
  env $(arm_env "$A") STEP4E_SEEDS="$S" STEP4E_TAG="$T" "$PY" -B run_step4e.py > "$RES_DIR/${T}.log" 2>&1 \
    && echo "  [完] $T" || echo "  [⚠️失败] $T（看 $RES_DIR/${T}.log）"
}
# 🔴 `03` L243-续2（user 2026-07-29：两天硬期限 + 数据采集只有一次机会）：
#   **一个实验被时间掐断，只有三种断法** ——
#     少了【臂】     ⟹ 比较根本做不成
#     各臂【步数不齐】⟹ 不是同预算比较，也做不成
#     少了【种子】   ⟹ 只是统计力弱一点，**所有比较照样成立**
#   ⟹ 必须让它断在**种子**这条轴上。做法 = **一颗种子的 9 条臂一起跑、跑完再开下一颗**
#      （`COLUMN=1`，默认开）。时间到了直接 Ctrl+C，手里是 N 列**完整的**格子，不是一堆半成品。
#   ⚠️ 关掉它（COLUMN=0）会让下一颗种子的臂提前顶上空闲槽位 ⟹ 掐断时出现半列 ⟹ 那半列全废。
COLUMN="${COLUMN:-1}"
COL_DONE=0
for S in $SEEDS; do          # 外层种子、内层臂 ⟹ 同一颗种子的各臂时间上靠近，机器状态更接近
  for A in $ARMS; do
    run_one "$A" "$S" &
    while [ "$(jobs -rp | wc -l)" -ge "$KMAX" ]; do sleep 20; done
  done
  if [ "$COLUMN" = "1" ]; then
    wait                                          # 本列 9 条臂全部落地，才开下一列
    COL_DONE=$((COL_DONE+1))
    echo "  ───── 种子 $S 这一【列】跑完（第 $COL_DONE 列）─────"
    # 🔴 第一列跑完立刻体检：**系统性问题一定在第一列就露头**。这时候停下来还剩一天多能补救；
    #    等两天后再查，就只剩重跑一条路，而时间不够。第二列起只报告、不拦（宁可留下数据）。
    if "$PY" -B "$CODE_DIR/tests/check_formal_integrity.py" "$RES_DIR" 2>&1 | tail -40; then :; else
      if [ "$COL_DONE" -eq 1 ]; then
        echo "  ❌❌ **第一列体检没过 —— 停在这里，别让它继续跑满两天。**"
        echo "     修完之后：RESUME=1 <同一条命令> 接着跑（已完成的自动跳过）。"
        exit 1
      fi
      echo "  ⚠️ 体检有问题（第 $COL_DONE 列）——已记录，继续跑；跑完务必按上面的清单处理。"
    fi
  fi
done
wait
echo "===== 全部结束 ====="
echo "存档：$(ls "$RES_DIR"/checkpoints/*_F240*.zip 2>/dev/null | wc -l) 个 · 分段副本：$(ls "$RES_DIR"/checkpoints/segments/*_F240*.zip 2>/dev/null | wc -l) 个（应 = run 数 × $NSEG）"
echo
echo "🔴 跑完的次序（别跳步）："
echo "  ① python3 -B 代码/tests/select_best_ckpt.py 结果  —— 用验证集挑最佳存档，产出可复算的清单"
echo "  ② 同趟重评【两趟】：最佳存档一趟 + 末段存档一趟（论文两版都报·`03` L236-A 的选择偏倚）"
echo "     REEVAL_EXPECT_STRICT=600 —— 全部臂都在官方 1300 上训，与测试 600 零交集 ⟹ 分母是 600 不是 563"
echo "  ③ 🔴 无盾臂（uns）**必须单独一个进程评**：replay 时 shield 取的是进程级环境变量、不从存档回读"
echo "     （run_step4e.py:955）⟹ 混在同组里会把整组的盾都关掉。见 04 §2。"
