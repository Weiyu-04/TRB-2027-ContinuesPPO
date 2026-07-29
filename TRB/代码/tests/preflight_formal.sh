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

echo "── ① 代码是不是最新版（指纹逐字对·不靠"我同步过了"）──"
# 基准 = 主窗口 main 分支 79c64f2 的 sha256 前 16 位。改了代码就要同步更新这张表。
declare -A W=( [代码/run_formal_2027.sh]=f534836d56d4e616 [代码/tests/run_reeval_all.sh]=99bcc869f45e4160
 [代码/tests/check_formal_integrity.py]=6eeef5596833357d [代码/tests/judge_t1_budget.py]=65081b23562660f9
 [代码/tests/reeval_official.py]=cea5010ebe372267 [代码/run_step4e.py]=03c36ea97f5da21e )
for f in "${!W[@]}"; do
  [ -f "$f" ] || { no "缺文件 $f"; continue; }
  g=$(sha256sum "$f" | cut -c1-16)
  [ "$g" = "${W[$f]}" ] && ok "$f" || no "$f 指纹 $g ≠ 应为 ${W[$f]}  ← 没同步到最新"
done

echo "── ② 新功能的开关都在（同步残缺时指纹已经会红，这里是双保险）──"
for pat in 'RESUME="${RESUME:-0}"' 'SPEED:-0' 'COLUMN="${COLUMN:-1}"' 'NSEG 只接受 20 或 30' 'roll_n_act' '内存够不够跑'; do
  grep -qF "$pat" 代码/run_formal_2027.sh && ok "起跑脚本含【$pat】" || no "起跑脚本缺【$pat】"
done
grep -q 'traj)' 代码/tests/run_reeval_all.sh && ok "重评脚本含轨迹专趟" || no "重评脚本缺轨迹专趟"
grep -q 'KEEP_PER' 代码/tests/reeval_official.py && ok "重评含逐局明细落盘" || no "重评缺逐局明细落盘"

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
NEED=$((10 * 35 / 10))
[ "${AV:-0}" -ge "$NEED" ] && ok "内存粗判够跑 10 路（需约 ${NEED}G）" \
  || no "可用 ${AV}G < 粗估 ${NEED}G ⟹ 10 路很可能 OOM（精确判定见闸门 2.5 的实测值）"
command -v screen >/dev/null && ok "screen 可用" || no "没装 screen（长训必须 screen 后台）"

echo "── ⑤ 会不会撞名 ──"
C=$(find . -path "*/checkpoints/*_F240*.zip" 2>/dev/null | wc -l)
[ "$C" -eq 0 ] && ok "无 F240 旧存档" || echo "  ⚠️ 已有 $C 个 F240 存档 ⟹ 续跑请加 RESUME=1；残留请先归档"

echo
[ "$F" -eq 0 ] && echo "🟢 预检全过（$(hostname)）—— 可以起飞" || echo "🔴 有 $F 处不过（$(hostname)）—— 先修，别起飞"
exit "$F"
