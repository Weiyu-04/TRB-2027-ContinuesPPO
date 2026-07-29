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
# 🔴 L243-续8 A7：KMAX 原来一个校验都没有。两个真会发生的坑：
#   ① 闸门 2.5 内存不够时会亲口建议「改用 KMAX=$KFIT」，而 KFIT 是整数除法、可能是 **0**
#      ⟹ `while [ $(jobs -rp|wc -l) -ge 0 ]` 恒真 ⟹ 起完第一个 run 就**永远 sleep**，两天只跑出 1 个。
#   ② 打成 `10x` 之类 ⟹ `[ N -ge 10x ]` 报错返回假 ⟹ **限流整个失效，108 个 run 同时起飞**，机器当场拖垮。
case "$KMAX" in ''|*[!0-9]*|0) echo "❌ 并发数（第 1 个参数）须是 ≥1 的整数，得到：'$KMAX'"; exit 1 ;; esac
mkdir -p "$RES_DIR" "$RES_DIR/checkpoints/segments" || { echo "❌ 建不出 $RES_DIR（权限？）"; exit 1; }

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
# 🆕 L243-续8：取某条臂配方里的单个字段（闸门 2 的"开关真落地了吗"对账要用）
arm_field () {
  local cfg; cfg=$(arm_cfg "$1"); [ -n "$cfg" ] || { echo "❌ 未知臂 $1" >&2; return 1; }
  local TAG PARTIES SHIELD ACT GW SHAPE; eval "$cfg"
  case "$2" in TAG) echo "$TAG";; PARTIES) echo "$PARTIES";; SHIELD) echo "$SHIELD";;
               ACT) echo "$ACT";; GW) echo "$GW";; SHAPE) echo "$SHAPE";; *) echo "";; esac
}
arm_tag () { local cfg; cfg=$(arm_cfg "$1"); [ -n "$cfg" ] || { echo "❌ 未知臂 $1" >&2; return 1; }; local TAG PARTIES SHIELD ACT GW SHAPE; eval "$cfg"; echo "$TAG"; }
arm_party () { local cfg; cfg=$(arm_cfg "$1"); [ -n "$cfg" ] || { echo "❌ 未知臂 $1" >&2; return 1; }; local TAG PARTIES SHIELD ACT GW SHAPE; eval "$cfg"; echo "$PARTIES"; }

# 🔴 L243-续8 A11：臂名打错时原来**不会立刻拦** —— `arm_env` 返回空串 ⟹ `env` 展开成空 ⟹
#   `run_step4e.py` 收不到 STEP4E_PARTIES ⟹ **默认跑全部 4 方**，而报错信息指向完全无关的地方。
#   这里在最前面就把 ARMS 逐个验一遍。
for _A in $ARMS; do
  [ -n "$(arm_cfg "$_A")" ] || { echo "❌ 未知臂名【$_A】。合法臂：ours disc base rr uns ush ab0 abB abG"; exit 1; }
done
for _S in $SEEDS; do
  case "$_S" in ''|*[!0-9]*) echo "❌ 种子须是非负整数，得到：'$_S'"; exit 1 ;; esac
done

N_RUN=$(( $(echo $ARMS|wc -w) * $(echo $SEEDS|wc -w) ))
echo "═══ TRB 2027 正式实验 ═══  臂=[$ARMS]  种子=[$SEEDS]  ⟹ $N_RUN 个 run · 并发≤$KMAX"
echo "    $NSEG 段 × 507,904 = $((NSEG*507904)) 步 · 官方 1300 训练集 · 分段存档开 · 内部曲线开"

echo "===== [闸门 0] 代码已同步（含本轮三项新功能）====="
for pat in "STEP4E_KEEP_SEGMENTS" "def _archive_segment" "class _RolloutStats" "STEP4E_ACT_DIST" "STEP4E_GW_ENTRY"; do
  grep -q "$pat" "$CODE_DIR/run_step4e.py" || { echo "❌ run_step4e.py 里没有【$pat】＝代码没同步 → 同步整个 代码 文件夹再跑"; exit 1; }
done
[ -f "$CODE_DIR/trb_env/usv_action_dist.py" ] || { echo "❌ 缺 trb_env/usv_action_dist.py"; exit 1; }
grep -q "gw_entry=self._sc.gw_entry" "$CODE_DIR/trb_env/usv_projection.py" || { echo "❌ usv_projection.py 是旧版"; exit 1; }
# 🔴 L243-续8（F 线复审 R1）：`04 §5` 早就记过这个教训 —— 「没被验的里面正好包括 metrics_subgrid.py
#   （修浮点误判的那个），它要是没同步成功，预检不会报警，离散臂那个本该恒 0 的次网格率会**安安静静地
#   又给出错数**」。而这条闸此前只 grep 了 usv_projection.py 一个字符串，metrics_subgrid.py 全流程无人看守。
grep -q "_GRID_TOL" "$CODE_DIR/trb_env/metrics_subgrid.py" 2>/dev/null \
  || { echo "❌ trb_env/metrics_subgrid.py 缺 _GRID_TOL ＝ 是 7-27 之前的旧版（次网格率会静默给错数）→ 同步整个 代码/ 再跑"; exit 1; }
grep -q "n_pairs_giveway" "$CODE_DIR/trb_env/evaluate.py" \
  || { echo "❌ trb_env/evaluate.py 是旧版（缺 n_pairs_giveway）→ 同步整个 代码/ 再跑"; exit 1; }
grep -q "A_NORMAL_OMEGA_MAX" "$CODE_DIR/run_step4e.py" \
  && grep -q "from trb_env.usv_env import" "$CODE_DIR/run_step4e.py" \
  || { echo "❌ run_step4e.py 没有导入动作常量 ＝ 是 L243-续8 之前的旧版（曲线 callback 第一步就 NameError 崩）→ 同步整个 代码/ 再跑"; exit 1; }
grep -q 'device=os.environ.get("STEP4E_DEVICE", "cpu")' "$CODE_DIR/trb_env/usv_sac_train.py" \
  || { echo "❌ trb_env/usv_sac_train.py 是旧版（连续 PPO 没锁 CPU，会偷偷跑到 GPU 上）→ 同步整个 代码/ 再跑"; exit 1; }
echo "  ✅ 三项新功能（分段存档 / rollout 采集 / 两把钥匙）都在；次网格率/评估口径/动作常量/CPU 锁 也都是新版"

echo "===== [闸门 0.2] 记录代码指纹（可复现性·`03` L240）====="
( cd "$CODE_DIR/.." && git rev-parse HEAD 2>/dev/null && git status --porcelain 2>/dev/null | head -5 ) \
  > "$RES_DIR/_formal_code_fingerprint.txt" 2>&1 || true
for f in run_step4e.py trb_env/usv_projection.py trb_env/usv_colregs.py trb_env/usv_continuous_shield.py trb_env/evaluate.py; do
  "$PY" -c "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest()[:16], sys.argv[1])" \
    "$CODE_DIR/$f" >> "$RES_DIR/_formal_code_fingerprint.txt"
done
# 🔴 L243-续8（F 线 R1）：单文件指纹只证明「这几个文件是新的」，不证明「代码是新的」。
#   run_step4e.py 一个人就 import 了 21 个 trb_env 模块，指纹表一个都没覆盖。补一条**整棵树**的指纹：
#   它不判对错（没有基准值可比），但**三台机器打出来必须一模一样** —— 这一条就能把「某台漏同步了一个文件」
#   当场照出来，而这正是本项目栽过的那个坑。
TREE_FP=$(cd "$CODE_DIR" && find . -name '*.py' -o -name '*.sh' | LC_ALL=C sort \
          | xargs sha256sum 2>/dev/null | sha256sum | cut -c1-16)
echo "TREE $TREE_FP  (整棵 代码/ 树·*.py + *.sh)" >> "$RES_DIR/_formal_code_fingerprint.txt"
cat "$RES_DIR/_formal_code_fingerprint.txt"
echo "  ✅ 指纹已落盘（论文方法节要写）"
echo "  🔴 **三台机器的 TREE 值必须逐字相同** —— 不同 = 有机器漏同步了文件，现在停下来同步，别烧。"

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
      echo "  ↻ 已有存档：${P}_s${S}${T}${S}  上次跑到 $DONE"
      # 🔴 L243-续8（A/B/E 三线独立复审都抓到）：**段级续训在 run_step4e.py 里根本不存在**。
      #   全文只有一处会加载模型（热启动，要显式给 STEP4E_WARMSTART_CKPT），训练永远新建模型；
      #   唯一的"跳过"是整条 (方,种子) 已经写完 jsonl 记录。⟹ 跑到一半被杀，再敲同一条命令 = **从 0 重训**。
      #   更脏的是：新一趟会从 @s00 逐段覆盖旧副本，若又在中途被杀，segments/ 里就是**两趟训练混在同一批文件名下**，
      #   而 select_best_ckpt 按段号挑，可能挑到已判废那趟的权重。⟹ 放行时**主动清掉本 run 的旧副本**。
      rm -f "$(dirname "$HIT")/segments/$(basename "${HIT%.zip}")"@s* 2>/dev/null
    else
      echo "  ❌ 已存在：$HIT"; CLASH=$((CLASH+1))
    fi
  done
done
if [ "$RESUME" = "1" ]; then
  echo "  ⚠️ RESUME=1：撞名闸已放行。**先看清楚续跑的真实粒度**（L243-续8 复审订正，此前这里写错了）："
  echo "     · 已经**整条跑完**并写了 jsonl 记录的 (臂,种子) ⟹ 自动跳过，不重跑。"
  echo "     · 跑到一半被杀的 ⟹ 🔴 **从第 0 步重训**（代码里没有段级续训这个功能）。"
  echo "     · 上面列出的旧分段副本已经清掉了 —— 避免两趟训练的存档混在同一批文件名下。"
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
echo "  ⏳ 提示：这一闸每条臂要跑 2 段 + **官方 600 场景全量评估**，9 条臂串行 ⟹ 约 **1~1.5 小时**。"
echo "     敲完命令别走开等第一个正式 run —— 一个多小时之后才会有。"
# 🔴 L243-续8 A2（**实测坐实**）：原来的采样器是 `ps -eo rss=,args= | awk /run_step4e\.py/`，
#   而内存的大头全在 SubprocVecEnv 的 8 个 worker 里。SB3 的 SubprocVecEnv 默认用 **forkserver**
#   （没有则 spawn），这两种启动方式下 worker 的 `ps args` 是
#   `python3 -c from multiprocessing.forkserver import main; ...` —— **里面根本没有 run_step4e.py**。
#   我在本机把 fork / forkserver / spawn 三种都跑了一遍：3 个各占 150MB 的 worker + 父进程，
#   这条 awk 在 forkserver 下只数出 13,096 KB，真实 python 总量 542,548 KB ⟹ **低估 41 倍**。
#   ⟹ 这道闸在任何机器、任何 KMAX 下都只会打印「✅ 内存够」，它宣称"不估、直接量"，实际一个 worker 都没量到。
#   改成**按进程树求和**（从本脚本的 PID 往下收所有 python 后代），三种启动方式都实测通过。
SELF_PID=$$
sample_tree () {
  ps -eo pid=,ppid=,rss=,comm= 2>/dev/null | awk -v root="$1" '
    { pid[NR]=$1; ppid[NR]=$2; rss[NR]=$3; cmd[NR]=$4; n=NR }
    END{ d[root]=1
         for (it=0; it<16; it++) { ch=0
           for (i=1;i<=n;i++) if (d[ppid[i]] && !d[pid[i]]) { d[pid[i]]=1; ch=1 }
           if (!ch) break }
         s=0
         for (i=1;i<=n;i++) if (d[pid[i]] && pid[i]!=root && cmd[i] ~ /^python/) s+=rss[i]
         print s+0 }'
}
RSSLOG="$RES_DIR/_formal_rss.txt"; : > "$RSSLOG"
( while :; do sample_tree "$SELF_PID" >> "$RSSLOG"; sleep 2; done ) & RSS_SAMPLER=$!
trap 'kill "$RSS_SAMPLER" 2>/dev/null' EXIT
for A in $ARMS; do
  T="_smkF240$A"; SMK="$RES_DIR/step4e_partial${T}.jsonl"; SEGDIR="$RES_DIR/checkpoints/segments"
  MET="$RES_DIR/run_metadata${T}.json"                     # 🆕 静态元数据（keep_segments / 各开关真落地的值都在这里）
  rm -f "$SMK" "$MET"; rm -f "$SEGDIR"/*"${T}"@s* 2>/dev/null
  # shellcheck disable=SC2046
  env $(arm_env "$A") STEP4E_SMOKE=1 STEP4E_STEPS=8000 STEP4E_NSEG=2 STEP4E_SEEDS=0 STEP4E_TAG="$T" \
    "$PY" -B run_step4e.py > "$RES_DIR/_${T}.log" 2>&1 \
    || { echo "  ❌ 臂 $A 冒烟跑崩（看 $RES_DIR/_${T}.log）"; tail -15 "$RES_DIR/_${T}.log"; exit 1; }
  # 🔴 L243-续8 A1（**今天这个脚本起不来的元凶**）：`keep_segments` 只写进
  #   `结果/run_metadata<TAG>.json`（`run_step4e.py:1938` 的 write_run_metadata），
  #   **逐 run 的 jsonl 记录里没有这个键**（我解包现存 jsonl 核过：55 个顶层键，没有它）。
  #   原来在 jsonl 里 grep 它 ⟹ 闸门 2 跑完第一条臂就 exit 1，9 条臂一条都验不完。
  grep -q '"keep_segments": true' "$MET" || { echo "  ❌ 臂 $A：keep_segments 没生效（查 $MET）"; exit 1; }
  grep -q 'official_1300'         "$SMK" || { echo "  ❌ 臂 $A：训练集不是官方 1300"; exit 1; }
  grep -q '"roll_steps"'          "$SMK" || { echo "  ❌ 臂 $A：rollout 采集没生效（A 类量会全丢）"; exit 1; }
  # 🔴 L243-续8 A8：上面这几条查的都是**周边设施**，恰恰**不查定义这 9 条臂的那几个开关**。
  #   失败场景：哪天改名/重构让 STEP4E_ACT_DIST 不再被读到（值合法所以 fail-fast 不响）⟹ 9 条臂冒烟**全绿**，
  #   两天烧完才发现 ours/abB 跑的其实是高斯 ⟹ 2×2 消融塌成两组重复，两个对子直接作废。
  #   ⟹ 拿 arm_cfg 的配方与 run_metadata 里"真落地的值"逐项对账，不符就停。
  #   （run_metadata 是 write_run_metadata 落的静态元数据，离散/连续臂都写，键最全。）
  _EXP_SHIELD="$(arm_field "$A" SHIELD)"; _EXP_ACT="$(arm_field "$A" ACT)"
  _EXP_GW="$(arm_field "$A" GW)";         _EXP_SHAPE="$(arm_field "$A" SHAPE)"
  "$PY" -c "
import json,sys
m=json.load(open(sys.argv[1],encoding='utf-8'))
rc=(m.get('run_config') or m)
exp_shield,exp_act,exp_gw,exp_shape,arm=sys.argv[2:7]
bad=[]
def eq(key,got,want):
    if got!=want: bad.append(f'{key}: 真落地={got!r} 应为={want!r}')
# 让路入口档：'-' 表示这条臂不设（离散臂）⟹ 应为默认 'paper'
eq('gw_entry', rc.get('gw_entry'), 'paper' if exp_gw in ('-','paper') else exp_gw)
if exp_act != '-':   eq('act_dist', rc.get('act_dist'), exp_act)
if exp_shield != '-':eq('continuous_shield', rc.get('continuous_shield'), exp_shield=='1')
# 连续专属塑形：full ⟹ 四件都开；none ⟹ 四件都关（离散臂结构上恒 0，同样成立）
full = (exp_shape=='full')
for k,v_full in (('well_shaping_weight',200.0),('xtrack_weight',200.0),('park_weight',20.0),('rate_weight',1.0)):
    eq(k, float(rc.get(k) or 0.0), v_full if full else 0.0)
eq('n_envs', int(rc.get('n_envs') or 0), 8)
eq('log_curves', bool(rc.get('log_curves')), True)
if bad:
    print(f'  ❌ 臂 {arm} 配方对账不符：'); [print('     ·',b) for b in bad]; sys.exit(1)
" "$MET" "$_EXP_SHIELD" "$_EXP_ACT" "$_EXP_GW" "$_EXP_SHAPE" "$A" \
    || { echo "     ⟹ 这条臂的开关**没真落地**，跑下去消融会作废。停。"; exit 1; }
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
  # 🔴 L243-续8 A6：冒烟的**主存档**原来留在 `checkpoints/` 同层没人清。
  #   `reeval_official.py:discover_ckpts()` 是 `glob(<dir>/*.zip)` + 有同名 _vecnorm.pkl 就收 ⟹
  #   将来不带 REEVAL_CKPTS 做全量重评时，表里会多出 9 条**只训了 3 万步**的"臂"，
  #   数字看起来像崩掉的正式臂。⟹ 验完就地清掉（只清本冒烟 TAG，绝不整目录删）。
  rm -f "$RES_DIR"/checkpoints/*"${T}"*.zip "$RES_DIR"/checkpoints/*"${T}"*_vecnorm.pkl \
        "$RES_DIR"/checkpoints/*"${T}"*.progress.json 2>/dev/null
  echo "  ✅ 臂 $A：配方逐项对账通过 + 分段存档真落地 + rollout 采集在（冒烟产物已清）"
done
kill "$RSS_SAMPLER" 2>/dev/null; trap - EXIT

echo "===== [闸门 2.6] 内存够不够跑 $KMAX 路（**实测·不是估的**）====="
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
    echo "     ⟹ 改用 KMAX=$(( KFIT > 0 ? KFIT : 1 )) 重跑本脚本；或降 STEP4E_NENVS（⚠️ 它进 config_sig，"
    echo "        中途改会让已跑的 run 从 0 重来，**要改就在起跑前一次改到位**）。"
    [ "${MEM_ACK:-0}" = "1" ] || exit 1
    echo "  ⚠️ MEM_ACK=1 强行放行 —— 你自己盯 free -g 和 dmesg 里的 oom-kill。"
  else
    echo "  ✅ 内存够（$KMAX ≤ $KFIT 路）"
  fi
fi

# ══════════════════════════════════════════════════════════════════════════════════════════
# 🆕 [闸门 2.7] 测速档（`03` L243-§7·复审抓出的最大排期风险）—— `SPEED=1` 才跑，跑完就退出，不起全量
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
  echo "===== [闸门 2.7] 测速档（每臂 ~6.5 万步 · 并发 $KMAX · 跑完即退出，不起全量）====="
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
  # 🔴 L243-续8 A6：`ARM_SPECS` 收不到它们，但 `reeval_official.py:discover_ckpts()` 的
  #   `glob(<dir>/*.zip)` **收得到** ⟹ 就地清掉，别留给将来的全量重评当"新臂"。
  rm -f "$RES_DIR"/checkpoints/*_probeSpd*.zip "$RES_DIR"/checkpoints/*_probeSpd*_vecnorm.pkl \
        "$RES_DIR"/checkpoints/*_probeSpd*.progress.json 2>/dev/null
  echo "  ✅ 测速存档已清（否则会被全量重评当成只训了 6 万步的「新臂」收进表）"
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
    # 🔴 L243-续8 A4/A5/A9（复审抓出的三个坑，都会让这道保险变成杀手）：
    #   A4 体检脚本的臂清单原来是**硬编码 9 条**，它把"已开工种子里缺的臂"判成硬伤 ⟹
    #      文档里自己写的 `ARMS="ours"` 那条命令跑完第一列必定自杀，而屏幕上的理由是"体检没过"。
    #      ⟹ 把本次真正要跑的 ARMS / SEEDS / NSEG 传给它，让它只查该查的。
    #   A5 体检脚本约定 `0=全过 / 1=有硬伤 / 2=只有提醒`，而这里原来只分"0 / 非 0" ⟹
    #      一条 run 少跑了几段（只是提醒）也会被判成硬伤，**整轮 108 个 run 停在第 1 列**。
    #   A9 `| tail -40` 保留的是**末尾** 40 行，切掉的恰好是最需要人眼看的 ①②两节。⟹ 不截断。
    CHK_LOG="$RES_DIR/_formal_integrity_col${COL_DONE}.txt"
    ARMS="$ARMS" SEEDS="$SEEDS" NSEG="$NSEG" \
      "$PY" -B "$CODE_DIR/tests/check_formal_integrity.py" "$RES_DIR" > "$CHK_LOG" 2>&1
    CHK_RC=$?
    cat "$CHK_LOG"
    if [ "$CHK_RC" -eq 1 ] && [ "$COL_DONE" -eq 1 ]; then
      echo "  ❌❌ **第一列体检查出硬伤 —— 停在这里，别让它继续跑满两天。**（全文见 $CHK_LOG）"
      echo "     🔴 修完之后**不要**指望 RESUME=1 续上半截 run —— 代码里没有段级续训，"
      echo "        跑到一半被杀的那条会从 0 重训；只有整条跑完的才会被跳过。"
      exit 1
    elif [ "$CHK_RC" -eq 1 ]; then
      echo "  ⚠️ 体检查出硬伤（第 $COL_DONE 列）——已记录到 $CHK_LOG，继续跑；跑完务必按清单处理。"
    elif [ "$CHK_RC" -eq 2 ]; then
      echo "  ⓘ 体检只有提醒、无硬伤（第 $COL_DONE 列）——继续跑。"
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
