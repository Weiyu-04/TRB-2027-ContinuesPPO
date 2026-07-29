#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════════════════
# TRB 2027 正式实验 · **起飞前预检**（`03` L243-续5）—— 三台各跑一次 · **纯只读、不写任何东西**
#
# 为什么要它：`run_formal_2027.sh` 自己有七道闸，但那些闸在**脚本已经跑起来之后**才生效。
#   这份预检管的是更前面一层：**代码到底同没同步到最新、机器扛不扛得住**。
#   本项目栽过的坑：服务器上跑的是旧代码副本（`03` L237 就提醒过"服务器跑的是 /root/trb/代码 的副本"）。
#   ⟹ 用**文件指纹逐字对**，不靠"我同步过了"这句话。
#
# 跑法（三台各一次）：  bash 代码/tests/preflight_formal.sh
# 退出码 0 = 全过可起飞 · 非 0 = 有几处不过
# ══════════════════════════════════════════════════════════════════════════════════════════
cd "$(dirname "$0")/../.." || exit 1                 # 定位到 trb 根目录（代码/tests/ 的上上层）
F=0; ok(){ echo "  ✅ $1"; }; no(){ echo "  ❌ $1"; F=$((F+1)); }
echo "═══ 起飞前预检 · $(hostname) · $(pwd) ═══"

# 🔴 L243-续8（F 线复审 R1/R2）：原来这一段只验 **6 个文件**的 sha256。但 `run_step4e.py` 一个人就
#   `from trb_env.xxx import` 了 21 个模块，一个都没进表；今天新改的 `select_best_ckpt.py`（`BUDGET_SEG`
#   是今天才加的，服务器若是旧版**不认这个变量、不报错、正常退出**，报数口径整个错掉）、`_common.py`、
#   三个出表出图脚本也都不在表里。⟹「6 个文件是最新的」不等于「代码是最新的」，而这正是
#   `04 §5` 已经记过的那个教训（没被验的里面正好有 metrics_subgrid.py，它错了会安安静静给错数）。
#   ⟹ 改成**整棵树**的指纹：一个数覆盖所有 .py/.sh，漏同步任何一个文件都会当场变色。
[ "${BASH_VERSINFO[0]:-0}" -ge 3 ] || { echo "❌ 需要 bash 运行本脚本（不要用 sh）"; exit 1; }

echo "── ① 代码是不是最新版（**整棵树**逐字对·不靠“我同步过了”）──"
# 基准 = 主窗口 main 分支的树指纹。**本文件自身被排除在外**（否则改基准值会改掉树指纹 = 自指死循环）。
BASE_CODE=6c3f08d943356b1d          # 代码/ 下全部 *.py + *.sh（不含本文件）
BASE_PAPER=ed3f8c3c97f88226        # Paper/正式实验/ 下全部 *.py（出表/出图/统计）
tree_fp () {   # $1=目录  $2=排除的相对路径（可空）
  ( cd "$1" 2>/dev/null || exit 0
    find . \( -name '*.py' -o -name '*.sh' \) ${2:+! -path "./$2"} | LC_ALL=C sort \
      | xargs sha256sum 2>/dev/null | sha256sum | cut -c1-16 )
}
G_CODE=$(tree_fp 代码 tests/preflight_formal.sh)
G_PAPER=$(tree_fp "Paper/正式实验")
[ "$G_CODE" = "$BASE_CODE" ] && ok "代码/ 整棵树逐字一致（$G_CODE）" \
  || no "代码/ 树指纹 $G_CODE ≠ 应为 $BASE_CODE  ← **有文件没同步**（下面会逐个点名）"
[ "$G_PAPER" = "$BASE_PAPER" ] && ok "Paper/正式实验/ 整棵树逐字一致（$G_PAPER）" \
  || no "Paper/正式实验/ 树指纹 $G_PAPER ≠ 应为 $BASE_PAPER  ← 出表/出图脚本没同步"

# 树对不上时，逐个文件点名到底是哪几个 —— 光说"树不一致"没法动手
if [ "$G_CODE" != "$BASE_CODE" ] || [ "$G_PAPER" != "$BASE_PAPER" ]; then
  echo "     ── 逐文件指纹（拿去和主窗口的对，不一样的就是没同步的）──"
  ( find 代码 "Paper/正式实验" \( -name '*.py' -o -name '*.sh' \) 2>/dev/null | LC_ALL=C sort \
    | xargs sha256sum 2>/dev/null | cut -c1-16,65- | sed 's/^/       /' )
fi

echo "── ② 新功能的开关都在（同步残缺时指纹已经会红，这里是双保险）──"
for pat in 'RESUME="${RESUME:-0}"' 'SPEED:-0' 'COLUMN="${COLUMN:-1}"' 'NSEG 只接受 20 或 30' 'roll_n_act' '内存够不够跑'; do
  grep -qF "$pat" 代码/run_formal_2027.sh && ok "起跑脚本含【$pat】" || no "起跑脚本缺【$pat】"
done
grep -q 'traj)' 代码/tests/run_reeval_all.sh && ok "重评脚本含轨迹专趟" || no "重评脚本缺轨迹专趟"
grep -q 'KEEP_PER' 代码/tests/reeval_official.py && ok "重评含逐局明细落盘" || no "重评缺逐局明细落盘"
# 🆕 L243-续8 起飞前复审修掉的几处，逐个点名（树指纹已经覆盖，这里是给人看的可读版）
grep -qF 'from trb_env.usv_env import' 代码/run_step4e.py \
  && ok "run_step4e 已导入动作常量（旧版会在曲线 callback 第一步 NameError 崩）" \
  || no "run_step4e 缺动作常量导入 ← **旧版·9 条臂一条都跑不出来**"
grep -qF 'device=os.environ.get("STEP4E_DEVICE", "cpu")' 代码/trb_env/usv_sac_train.py \
  && ok "连续 PPO 已锁 CPU" || no "连续 PPO 没锁 CPU ← 6 条连续臂会偷偷跑到 GPU"
grep -qF 'sample_tree' 代码/run_formal_2027.sh \
  && ok "内存闸按进程树求和（旧版看不见 8 个采样进程、恒说『够』）" || no "内存闸还是旧版（等于没有）"
grep -qF 'run_metadata${T}.json' 代码/run_formal_2027.sh \
  && ok "闸门 2 的 keep_segments 查对了文件" || no "闸门 2 还在 jsonl 里查 keep_segments ← **一条臂都过不去**"
grep -qF 'REEVAL_ENVCFG_ACK=1' 代码/tests/run_reeval_all.sh \
  && ok "重评的无盾组带了 ACK（否则整组当场死）" || no "重评无盾组缺 ACK"
grep -qF 'seg_no = best' 代码/tests/select_best_ckpt.py \
  && ok "挑最佳存档的段号没差一" || no "挑最佳存档段号差一 ← 评的是下一段的模型"
grep -qF 'BUDGET_SEG' 代码/tests/select_best_ckpt.py \
  && ok "挑最佳存档认 BUDGET_SEG" || no "select_best_ckpt 不认 BUDGET_SEG ← 报数口径会静默错掉"
grep -qF '"F240" not in ck' "Paper/正式实验/_common.py" \
  && ok "出表脚本不会把正式臂贴成探索期标签" || no "出表脚本会把正式臂认成探索期臂 ← 两代实验混一行"

echo "── ③ 数据与依赖 ──"
[ -f balanced_pool/manifest_official_1300.json ] && ok "训练清单在" || no "缺 balanced_pool/manifest_official_1300.json"
N=$(ls scenarios/T-*.xml 2>/dev/null | wc -l); [ "$N" -ge 2000 ] && ok "场景 $N 个" || no "场景只有 $N 个（应 ≥2000）"
PY=/root/miniconda3/bin/python; [ -x "$PY" ] || PY=$(command -v python3)
DEPS_OK=0
if "$PY" -c "import torch,stable_baselines3,sb3_contrib,numpy,scipy,shapely,osqp,cvxpy,vesselmodels" 2>/dev/null; then
  DEPS_OK=1
  ok "依赖齐（python $("$PY" -c 'import sys;print(sys.version.split()[0])') · numpy $("$PY" -c 'import numpy;print(numpy.__version__)')）"
else
  no "依赖缺 —— 逐个点名："
  for m in torch stable_baselines3 sb3_contrib numpy scipy shapely osqp cvxpy vesselmodels commonocean; do
    "$PY" -c "import $m" 2>/dev/null && echo "       ✓ $m" || echo "       ✗ $m  ← 缺这个"
  done
  echo "     装法：pip install --no-cache-dir -r 代码/requirements.txt"
  echo "     （torch 若镜像已带就别重装；commonocean-io 卡在 antlr4 时见 requirements.txt 里的踩坑记）"
fi
"$PY" -B 代码/make_official_manifest.py --check balanced_pool/manifest_official_1300.json >/dev/null 2>&1 \
  && ok "清单六项自洽（训练 1300 + 验证 100 · 与测试 600 零交集）" || no "清单校验没过 → 别烧卡"

echo "── ④ 机器资源 ──"
CORE=$(nproc); MEM=$(awk '/MemTotal/{printf "%.0f",$2/1048576}' /proc/meminfo)
AV=$(awk '/MemAvailable/{printf "%.0f",$2/1048576}' /proc/meminfo)
DF=$(df -BG --output=avail . 2>/dev/null | tail -1 | tr -dc 0-9)
echo "     核数 $CORE · 内存 ${MEM}G（可用 ${AV}G）· 磁盘可用 ${DF}G"
# 🔴 磁盘门槛按【实测】定，不拍脑袋（`03` L243-续6）：
#   单 run 实测 ≈ 12 MB（主存档 0.2 + progress 0.7 + 20 段副本 3.6 + 分段 progress 累计 7.6）
#   每台 36 run（12 种子 ÷ 3 台 = 4 列 × 9 臂）⟹ **训练本身只要约 0.5 GB**。
#   真正吃盘的是【装依赖】（torch 2~3G + 其余）与 pip 缓存 ⟹ 门槛随"依赖装没装"分两档。
NEED_TRAIN=2                                   # 训练产物 0.5G，给 4 倍余量
if [ "$DEPS_OK" = "1" ]; then NEED=$NEED_TRAIN; WHY="依赖已装 ⟹ 只需训练产物的空间"
else NEED=8; WHY="依赖还没装 ⟹ 要留 torch 等约 4~5G 的安装空间"; fi
[ "${DF:-0}" -ge "$NEED" ] && ok "磁盘够（需 ≥${NEED}G · $WHY）" || {
  no "磁盘只剩 ${DF}G < ${NEED}G（$WHY）"
  echo "     盘被谁吃了（前 8 名）："
  du -shx /root/* 2>/dev/null | sort -h | tail -8 | sed 's/^/       /'
  echo "     常见可删：~/.cache/pip（pip 缓存）· /root/miniconda3/pkgs（conda 包缓存）· 旧的 结果/ 批次目录"
}
# 粗判：单 run 光场景池 ≈ 8 worker × 1300 × 0.20MB ≈ 2.1G，加 torch 基线保守按 3.5G/run 估
# 🆕 L243-续8：并发数不再写死 10 —— 用 `KMAX=<你起飞时要用的并发数> bash ...` 让预检和起飞命令同一个数。
K="${KMAX:-10}"
NEED=$(( K * 35 / 10 ))
[ "${AV:-0}" -ge "$NEED" ] && ok "内存粗判够跑 $K 路（需约 ${NEED}G）" \
  || no "可用 ${AV}G < 粗估 ${NEED}G ⟹ $K 路很可能 OOM（精确判定见起跑脚本闸门 2.6 的实测值）"
[ "$CORE" -ge $(( K * 2 )) ] && ok "核数 $CORE 够跑 $K 路（每 run 8 个采样进程，超订也能跑但会变慢）" \
  || echo "  ⚠️ 核数 $CORE 对 $K 路偏少（每 run 8 个采样进程）⟹ fps 会掉，排期要按实测值重算"
command -v screen >/dev/null && ok "screen 可用" || no "没装 screen（长训必须 screen 后台）"

echo "── ⑤ 会不会撞名 ──"
# 🔴 L243-续8（F 线 R4）：原来把主存档和分段副本混在一个数里报。两者的处理方式**完全不同**：
#   主存档撞名 —— RESUME=1 放行是合理的（本来就要覆盖）；
#   `segments/` 里的陈年副本 —— **必须清掉**，否则上一趟跑到第 18 段、这一趟从 0 重训只跑到第 5 段时，
#   @s06..@s17 是**上一趟**的权重，而 `select_best_ckpt.py` 按段号挑、分不出新旧，可能挑到已判废那趟的模型。
#   （起跑脚本的闸门 0.6 现在会在 RESUME=1 放行时自动清，这里是双保险 + 让人看见。）
C=$(find . -path "*/checkpoints/*_F240*.zip" ! -path "*/segments/*" 2>/dev/null | wc -l)
CS=$(find . -path "*/checkpoints/segments/*_F240*@s*.zip" 2>/dev/null | wc -l)
[ "$C" -eq 0 ] && ok "无 F240 旧主存档" || echo "  ⚠️ 已有 $C 个 F240 主存档 ⟹ 续跑请加 RESUME=1；残留请先归档"
[ "$CS" -eq 0 ] && ok "无 F240 陈年分段副本" \
  || echo "  🔴 segments/ 里有 $CS 个 F240 陈年副本 ⟹ **续跑前必须清掉**（两趟训练混在同一批文件名下，挑最佳存档会挑串）"

echo
[ "$F" -eq 0 ] && echo "🟢 预检全过（$(hostname)）—— 可以起飞" || echo "🔴 有 $F 处不过（$(hostname)）—— 先修，别起飞"
exit "$F"
