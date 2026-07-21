"""
step4e 四方对比 → Table III（本地 / 服务器多核并行；**避会话暂停杀后台 → user 直接跑**）
=====================================================================================
忠实复现 Krasowski 2024 §VII Table III 的【四方】对比（钱图）：
    Base            = UnshieldedUSVEnv, colregs_weight=0.0   （r_colregs 关；式(10) 去 colregs 项，MaskablePPO）
    Rule-reward     = UnshieldedUSVEnv, colregs_weight=1.0   （r_colregs 软约束；式(10) 全量、无盾，MaskablePPO）
    Discrete-safe   = ShieldedUSVEnv,   colregs_weight=1.0   （As(ρ) 安全盾 = action masking，MaskablePPO）
    Continuous-safe = ContinuousProjectionEnv, colregs_weight=0.0  （SAC + 连续投影盾；合规靠投影约束(档位A 经验性·非档位B provable 硬保证)、丢 r_colregs；
                      Phase 3 Node C-C2 接入 = train_eval_one_continuous，算法换连续=同环境对比 D2/D37-B/D38）

四方同口径（钱图命门，03 D2/D22/L46/L47）：四方共享 同 env 物理 / 同场景池+划分(SPLIT_SEED) / 同评估(ViolationCounter,
    pre-step rho_acting 紧急步%) / 同款 VecNormalize(obs+reward,clip_obs10) / gamma=0.99（同进算法与 VecNorm）/ net[64,64]。
    离散三方 = make_vec_env → VecNormalize → MaskablePPO(ent_coef 熵退火)；
    Continuous-safe = make_continuous_safe_model(SAC, ent_coef='auto' 无退火) → ContinuousProjectionEnv（连续投影盾）。
VecNormalize/网络结构常量从 `trb_env.train` 导入（VECNORM_KWARGS / POLICY_NET_ARCH）→ 与产品化入口同步。
ent_coef（仅离散）= 熵退火 ENT_START→ENT_END（前 ENT_FRAC 步线性退完；START==END 即关退火复现旧常量 0.01 配方，03 D25）。
**三条【有意】差异（其余逐位一致，03 L21/L46/L47）= ① 盾/投影 on/off（env_cls）② 算法 MaskablePPO(离散) vs SAC(连续)（D2）
③ r_colregs on/off（colregs_weight：RR/Discrete=1 有、Base/Continuous=0 无）**。Continuous-safe 另有诊断列「兜底步%」(jsonl)。

【⚠️ 跑通 ≠ 效果好】冒烟只证【脚本正确 + 三方能训】；【数值对齐 Krasowski】
   （Base 违规≈2.65 / RR≈2.24 / Discrete-safe 到达≈86%、违规≈0）= Phase 1 通过门，靠满量真跑判定。
【⚠️ 吃 CPU 不吃 GPU】MLP 2×64 太小、GPU 闲置 → 服务器要【高核心数 CPU】，别租 GPU（03 D17）。

【单机怎么跑】（trb 环境）
    # 1) 先冒烟（默认 SMOKE=True，~分钟级，确认三方跑通、不停船、Base/RR 出碰撞/违规锚点、看 fps）：
    /opt/miniconda3/envs/trb/bin/python -B 代码/run_step4e.py
    # 2) 冒烟没问题 → ⭐把下方 CONFIG 的 SMOKE 改成 False，再跑全量（重启只需同一条命令、自动续跑）：
    /opt/miniconda3/envs/trb/bin/python -B 代码/run_step4e.py

【⚡ 服务器多核并行跑法（把 wall-clock 压到单任务下限）】
   9 个 (方,种子) 任务互相独立 → 并发起多进程、每个 ~N_ENVS 核。
   · 任务分片：STEP4E_PARTIES（选方，如 'Discrete-safe'）+ STEP4E_SEEDS（选种子，如 '0'）。
   · 并行安全：各任务追加同一 partial 用 flock 串行化、table3 原子写、下载 PID 隔离 → 并发不打架。
   · 预下载 / 纯聚合：STEP4E_DOWNLOAD_ONLY=1（只下场景不训）/ STEP4E_AGG=1（只读 partial 出 table3 不训）。
   · ⭐ 一键启动器：`bash 代码/launch_step4e.sh K NENVS 种子`（K=最大并发任务数、每任务 NENVS 核）。
   · ⚠️ 先量速度：上服务器先跑一次 SMOKE（默认），把打印的【Discrete-safe fps】发我 → 我据此定 K 和核心数、保证 <5h。

【🔬 归因式 mini-run 协议（2026-06-13 审核加固，03 D25 + 审核 #1-#4）】
   退火 fix 未经服务器验证、且漏掉替代解释 → mini-run 必须能【归因】、不能只跑退火。
   各消融臂用独立 STEP4E_TAG 隔离输出（table3 头自记 ent/clip 并对混配告警）。各臂 3方×3种子×1.2M×200场景：
   · 对照臂（控制变量命门）= 常量 0.01：STEP4E_ENT_START=0.01 STEP4E_ENT_END=0.01 STEP4E_TAG=_c01
       → 回答"退火 vs 常量谁让 DS seed1/2 离地"（否则离地了也不知是退火还是种子方差）。
       ⚠️ 重点对照常量臂里【RR seed1/2 是否也塌】：RR(无盾)也塌 = 配方种子脆弱；仅 DS(有盾)塌 = 坐实盾放大（补 D25 缺口）。
   · 退火臂（提出的 fix）= 默认 0.03→0.005：STEP4E_TAG=_a03（想试更高起点：STEP4E_ENT_START=0.05 STEP4E_TAG=_a05）。
   · clip 臂（验替代解释 #1）= STEP4E_CLIP_REWARD=50：对已知塌的 DS seed1 单跑
       ⚠️【直跑 run_step4e.py、别套 launch_step4e.sh】——launcher 会 per-task 覆盖 PARTIES/SEEDS 成全矩阵、白烧算力；
       ⚠️【L49#1 护栏】CLIP_REWARD 仅施离散臂、连续臂走默认 10 → 须显式 STEP4E_CLIP_REWARD_ACK=1 确认（仅离散诊断、勿含 Continuous-safe），否则 fail-fast：
       STEP4E_SMOKE=0 STEP4E_CLIP_REWARD=50 STEP4E_CLIP_REWARD_ACK=1 STEP4E_PARTIES=Discrete-safe STEP4E_SEEDS=1 STEP4E_TAG=_clip50 python -B 代码/run_step4e.py
       → 若松 clip 离地而退火不，则主因是归一化 clip 削平稀疏 +50 到达信号、非探索 → fix 方向要换。
   · normR 臂（验替代解释 #2 真身 = 奖励归一化的 std 除法，03 L27）= 常量 ent + STEP4E_NORM_REWARD=0（与 c01 只差 norm_reward、干净隔离；直跑别套 launcher）：
       STEP4E_SMOKE=0 STEP4E_ENT_START=0.01 STEP4E_ENT_END=0.01 STEP4E_NORM_REWARD=0 STEP4E_PARTIES=Discrete-safe STEP4E_SEEDS=1 STEP4E_TAG=_normR0 python -B 代码/run_step4e.py
       → 对照 c01（同常量 ent、norm_reward=True、s1=0%）：若 s1 离地 = 奖励归一化(÷return-std)是元凶 → 配方改"只归一化观测"；若还冻 = 元凶更深（盾探索）。
   · 锚点保护 = 核对 Base/RR 违规列（钱图锚 2.65/2.24）在退火 vs 常量下是否一致；带偏则无盾两方退回常量 0.01。
   · ⚠️ 熵在盾零自由度态(ρ1/ρ5)恒 0 → 退火只经 ρ0 起作用、碰不到盾；写作勿把"盾放大塌缩"与"熵退火解之"挂钩。
【环境变量】STEP4E_SMOKE / STEPS / NTOTAL / SEEDS / NSEG / NENVS / TAG / PARTIES / AGG / DOWNLOAD_ONLY / SDIR
            / ENT_START / ENT_END / ENT_FRAC（熵退火）/ CLIP_REWARD（+CLIP_REWARD_ACK=1 离散诊断 opt-in，L49#1）/ NORM_REWARD（消融）/ DEVICE / OMP
【输出】结果/step4e_partial{tag}.jsonl（每 party×seed 一行，增量/断点续/并行安全）+ 结果/table3{tag}.txt（聚合 Table III）。
【跑完发我】把 结果/table3.txt 内容发我，我核对是否过 Phase 1 通过门（对齐 Krasowski）。
"""
from __future__ import annotations
import json
import math
import os
import random
import sys
import time

# ⚡ 限制每进程 BLAS/OMP/torch 线程 = 1（必须在 import numpy/torch 前设、放最前）：
# SubprocVecEnv 已用【多进程】并行采样；若每个 worker 进程的 numpy/torch 再各自开满核心线程，
# 8 worker × N核线程 会在多核机上互抢核 → 严重拖慢（32+ 核云服务器尤甚，RL 常见坑）。
# 每进程 1 线程 → 并行度来自 8 个 worker 进程本身，最优。STEP4E_OMP 可覆盖（一般别动）。
_omp = os.environ.get("STEP4E_OMP", "1")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, _omp)

# ===================== CONFIG（按需改；环境变量可覆盖，不必改文件）=====================
SMOKE = os.environ.get("STEP4E_SMOKE", "1") != "0"   # 默认先冒烟；全量改这里 False（或 STEP4E_SMOKE=0）

# --- 全量（对齐 Krasowski §VII）---
N_TOTAL_FULL = 2000          # 用前 N 个 HandcraftedTwoVesselEncounters（论文 ~2000）
TEST_FRAC = 0.30             # 70% 训练 / 30% held-out 测试（论文 70/30）
STEPS_FULL = 3_000_000       # 每 seed 3M（论文 3M）
SEEDS_FULL = [0, 1, 2]       # 先 3 种子出初版；可扩到 range(10)（论文 10 seed）
N_SEG_FULL = 6               # 分段评估段数（看趋势；每段末 eval 一次 held-out）

# --- 冒烟（快确认三方端到端 + 不停船 + 锚点方向对 + 量 fps）---
N_TOTAL_SMOKE = 60
STEPS_SMOKE = 60_000
SEEDS_SMOKE = [0]
N_SEG_SMOKE = 2

N_ENVS = int(os.environ.get("STEP4E_NENVS", "8"))    # 离散三方(on-policy PPO)每任务 SubprocVecEnv 进程（≈每任务占核数）
# ⚠️ Continuous-safe(SAC=off-policy) 单独定 n_envs（审查 perf MEDIUM）：SAC train_freq=1/gradient_steps=1 下 n_envs>1
# 会【摊薄梯度更新数】(3M env步 → 3M/n_envs 次 update)、改学习动态 → 默认 1（SAC 标准、最 sample-efficient/每env步）；
# 服务器要并行提速可调大 STEP4E_SAC_NENVS 但须同知会改了 SAC 学习动态（非纯加速、与离散 n_envs 语义不同）。
N_ENVS_SAC = int(os.environ.get("STEP4E_SAC_NENVS", "1"))
# ⚠️ Continuous-safe-PPO(on-policy) 须靠并行 rollout 提采样效率（红队 MAJOR：原连续分派硬编 N_ENVS_SAC=1·PPO 拿到 1=丢并行效率=PPO 立项动机作废）→ 默认 = 离散 N_ENVS(8)
N_ENVS_PPO = int(os.environ.get("STEP4E_PPO_NENVS", str(N_ENVS)))
def continuous_n_envs(algo):
    """连续臂 n_envs 选择（可测·红队 MAJOR 回归）：PPO on-policy→N_ENVS_PPO(并行 rollout)·SAC off-policy→N_ENVS_SAC。"""
    return N_ENVS_PPO if str(algo).lower() == "ppo" else N_ENVS_SAC
SPLIT_SEED = 0               # train/test 划分种子（固定 → 四方/多 run/并行任务用同一 held-out 集，可比）
POOL_SIZE = int(os.environ.get("STEP4E_POOL", "2000"))   # 场景库总数；n_total<POOL 时从全库均匀铺开抽样（避免缩小实验聚集采样=偏向失败代理，03 L29）；n_total≥POOL 无影响
# --- 种子稳健性 fix：ent_coef 熵退火（2026-06-13，诊断=种子分裂"游荡超时"局部最优，2 agent 验证 + 本地复现）---
# 前期高熵踹出"游荡避让不到目标"盆地、后期退到≈旧配方让已学成种子收敛。末值 0.005≈旧 0.01 → 对已学成种子/无盾 RR 低风险。
# START==END 即关退火（恒定，复现旧配方：STEP4E_ENT_START=STEP4E_ENT_END=0.01）。**仍不动 Krasowski reward**（合规）。
ENT_START = float(os.environ.get("STEP4E_ENT_START", "0.03"))   # 退火起点（高熵逃局部最优）
ENT_END = float(os.environ.get("STEP4E_ENT_END", "0.005"))      # 退火终点（≈旧配方、收敛锐化）
ENT_FRAC = float(os.environ.get("STEP4E_ENT_FRAC", "0.6"))      # 前 frac 比例总步数退完，之后恒 END
_GAMMA = float(os.environ.get("STEP4E_GAMMA", "0.99"))          # 折扣因子【单一真相源】：同进 VecNormalize/PPO/SAC + 修法A PBRS（`03` L81·任一不同源破 Ng 不变性）
_WELL_B = float(os.environ.get("STEP4E_WELL_B", "0.0"))         # 修法A 进门势强度 well_B（默认 0.0=关=与现状逐位等价 bit-identical；A/B 用 200=4×C_GOAL·四方对称透传四臂）
_WELL_X = float(os.environ.get("STEP4E_WELL_X", "0.0"))         # 对症 横向进带势强度 well_X（`03` L88·默认 0.0=关=逐字节不变；治终端横向 cross-track 进不了窄带；起点 200·A/B 标定·四方对称透传）
_XTRACK_RADIUS = float(os.environ.get("STEP4E_XTRACK_RADIUS", "80.0"))   # 对症 横向势半径 R_lat（默认 80m·覆盖实测失败 |e_cross|=32-44m+带半宽 30m+余量·可调 A/B 扫）
_PARK_W = float(os.environ.get("STEP4E_PARK_W", "0.0"))         # 想法B 终端保速势 well_park（`03` L109·默认 0.0=关=逐字节不变；治"终端横向修正时过早减速停带外"；连续臂专属·不接离散=不开挂；A/B 扫 ~10/20/30·50 偏强）
_PARK_RADIUS = float(os.environ.get("STEP4E_PARK_RADIUS", "400.0"))   # 想法B 近目标作用半径 R_park（默认 400m）
_PARK_VTARGET = float(os.environ.get("STEP4E_PARK_VTARGET", "4.0"))   # 想法B 目标机动速度（speed_frac 封顶·成功局 ~3.5-4.5·L109）
_STEP_COST = float(os.environ.get("STEP4E_STEP_COST", "0.0"))   # 修法C 每步生存成本 c_step（`03` L123·默认 0.0=关=与现状逐位等价 bit-identical；连续臂专属·非PBRS 真改最优·治游荡局部最优；A/B 用 0.5·扫 {0.3,0.5,0.75}；离散复现臂不传=忠实 Krasowski·盾硬拒）
_DWELL_W = float(os.environ.get("STEP4E_DWELL_W", "0.0"))       # r_dwell 入库赤字滞留成本系数 c_dwell（`03` L161/L162·默认 0.0=关=bit-identical；连续臂专属·非PBRS 真改最优·治 corr≈0 终端入库病；A/B 用 0.5·扫 {0.3,0.5,0.75}；离散盾硬拒）
_DWELL_WLAT = float(os.environ.get("STEP4E_DWELL_WLAT", "90.0"))   # r_dwell 横向 cross-track 赤字尺度 W_DWELL（默认 90m·A/B 可调·>此值饱和零梯度=复审 L162⑥ 须盯尾部）
_DWELL_HDG = float(os.environ.get("STEP4E_DWELL_HDG", "0.52"))     # r_dwell 朝向赤字尺度 H_DWELL（默认 0.52rad≈30°）
_DWELL_R = float(os.environ.get("STEP4E_DWELL_R", "250.0"))        # r_dwell 近场作用半径 R_DWELL（默认 250m·带外 r_dwell=0）
_DWELL_B = float(os.environ.get("STEP4E_DWELL_B", "0.0"))          # r_dwell 终端入库锚 B_DWELL（默认 0·真入库 +B·补"回避近场躲成本"病·A/B 现病象时开 ~50·仅 c_dwell>0 时生效）
if not (math.isfinite(_DWELL_W) and _DWELL_W >= 0.0):             # 防御 fail-fast（同 _COLREGS_W_CONT·`03` L162⑥ 提早报错·别等 RewardFunction 深处）
    raise SystemExit(f"🔒 STEP4E_DWELL_W 须有限非负,得 {_DWELL_W}")
# 🆕 第二条腿修法（`03` L172·连续臂专属·默认关=bit-identical·治崩塌"corr≈0 脱钩 + 过路惩罚泄漏入库精修段"）：
#   _C_REACH=重标 r_goal 系数（Rung1·默认=C_REACH=1.5=现状·降到 0.2 压近常数回报量级→抬 +50 归一化占比→治 corr≈0）；
#   _DOCK_R+_V_DOCK=泊位精修门（Rung2·默认关·泊位区内把速度地板 V_LOW 降到 v_dock·治"入库减速被罚"泄漏·【降非清零】防停门口）。
#   默认引用模块常量（防常量漂移=bit-identical 与 RewardFunction 默认同源）·仅连续臂 maker 透传（离散不传=忠实 Krasowski typo-fix）·进 config_conflict/config_sig/jsonl/run_config。
from trb_env.usv_reward import C_REACH as _C_REACH_DEF, V_LOW as _V_LOW_DEF
_C_REACH = float(os.environ.get("STEP4E_C_REACH", repr(_C_REACH_DEF)))    # Rung1 重标 r_goal 系数·默认=1.5=现状·降救脱钩
_DOCK_R = float(os.environ.get("STEP4E_DOCK_R", "0.0"))                   # Rung2 泊位精修门半径 R_dock·默认 0=关·区内减免速度地板
_V_DOCK = float(os.environ.get("STEP4E_V_DOCK", repr(_V_LOW_DEF)))        # Rung2 区内残余速度地板 v_dock·默认=V_LOW=2.5=不变
# 🆕 L190（user 2026-07-16）：热启动源 ckpt（探索侧治崩·JSRL 2204.02372/AWAC 2006.09359 式·灌源均值策略+源 vecnorm stats·log_std 重置回 in-box 初值）。
#   路径不含扩展名·须 <base>.zip+<base>_vecnorm.pkl 同在。默认 ""=不热启动=bit-identical(严格验 max|Δ|=0)。**仅连续 PPO 臂**(SAC maker 无此形参→algo≠ppo 时 fail-fast·见下 :329)。
#   进 config_conflict(第37元·防【热启动 vs 从零 两种训练流程】同TAG静默混写=汇总表混算) 不进 config_sig(=保续训 bit-identical·同 shield/augment/cone 既定口径·A/B 用 distinct TAG)。
#   ⚠️ **四方钱图混写(同 cone/augment/shield 结构性模式·自审补漏)**：热启动连续臂【无法】与从零离散臂共写同一 jsonl（离散记录 warmstart=None ≠ cur 路径 → config_conflict 阻断 SystemExit）
#     → 四方对比须【热启动连续臂独立 TAG 单跑·绘图/聚合层合并两 jsonl】·别 launcher 顶层 export 混写。
#   ⚠️ **方法论红线**：全10种子【统一】从同一好源热启动(非只救崩种子=10种子不独立)·provenance 如实记(jsonl+run_config)·**绝不 claim"从零稳定收敛"若热启动了**([[seed-variance-report-iqm-not-cherrypick]])。
#   ⚠️ **不查语义配置**：仅 obs 维/policy 键集守卫；源须与本 run 同 shield/augment_rho/goal_cone/colregs_weight=【烧前预检责任】(`03` L190 E)。
_WARMSTART_CKPT = os.environ.get("STEP4E_WARMSTART_CKPT", "").strip()
if _WARMSTART_CKPT:
    _WARMSTART_CKPT = os.path.realpath(_WARMSTART_CKPT)   # 🆕 路径归一化（第2轮审 NIT）：同一源的 相对/绝对/软链 写法不再被误判成"异源冲突"
    if not (os.path.exists(_WARMSTART_CKPT + ".zip") and os.path.exists(_WARMSTART_CKPT + "_vecnorm.pkl")):
        raise SystemExit(f"🔒 STEP4E_WARMSTART_CKPT 指向的源 ckpt 缺 .zip 或 _vecnorm.pkl: {_WARMSTART_CKPT}（须两者同在·预检别白烧）")


def _warmstart_fingerprint(base: str):
    """🆕 L190 第2轮审 HIGH#1（"指针 vs 标量"不对称·第1轮15agent+自审均漏）：源 ckpt 内容指纹。

    🔴 为什么必须有：其它旋钮的值是【标量=自身即身份】(shield=True/cone=45°)；warmstart 的值是【路径=指针】·指向【可变文件】。
      只记路径 → 源文件被覆盖/重生成时【10 种子记录看起来完全一样】·config_conflict 结构上不可能发现
      → user 定的「全10种子统一从同一好源」方法论红线【无守卫且事后不可证】。
    → 记 sha256(前16位·zip+vecnorm 各一) 当【真身份】：进 provenance（可机器审计"10种子同源"）+ 进 config_conflict（换源即冲突·哪怕路径没变）。
    """
    import hashlib
    out = {}
    for _sfx, _k in ((".zip", "zip_sha256"), ("_vecnorm.pkl", "vecnorm_sha256")):
        _p = base + _sfx
        try:
            _h = hashlib.sha256()
            with open(_p, "rb") as _f:
                for _chunk in iter(lambda: _f.read(1 << 20), b""):
                    _h.update(_chunk)
            out[_k] = _h.hexdigest()[:16]
        except OSError as _e:                      # 读失败 → 显式标记（不静默当"无指纹"=退回裸路径身份）
            raise SystemExit(f"🔒 热启动源 ckpt 指纹计算失败(读不了 {_p}): {_e}")
    return out


_WARMSTART_FP = _warmstart_fingerprint(_WARMSTART_CKPT) if _WARMSTART_CKPT else None
# 🆕 config_conflict / provenance 用的【源身份】= 路径+内容指纹（非裸路径·防同路径换源静默混写）
_WARMSTART_ID = (f"{_WARMSTART_CKPT}#{_WARMSTART_FP['zip_sha256']}+{_WARMSTART_FP['vecnorm_sha256']}"
                 if _WARMSTART_CKPT else None)
if not (math.isfinite(_C_REACH) and _C_REACH >= 0.0):
    raise SystemExit(f"🔒 STEP4E_C_REACH 须有限非负,得 {_C_REACH}")
if not (math.isfinite(_DOCK_R) and _DOCK_R >= 0.0):
    raise SystemExit(f"🔒 STEP4E_DOCK_R 须有限非负,得 {_DOCK_R}")
if _DOCK_R > 0.0 and not (0.48 < _V_DOCK <= _V_LOW_DEF):                  # 同 RewardFunction 守卫：下界 0.48=离零保守正地板(防停船墙复活)·上界 V_LOW(禁抬高地板反罚)。⚠️订正 `03` L176：旧"0.48=单步max减速"错 5×(真=a_max·dt=2.4)·0.48 仅作离零正裕度
    raise SystemExit(f"🔒 STEP4E_DOCK_R>0 时 STEP4E_V_DOCK 须 0.48<v≤{_V_LOW_DEF},得 {_V_DOCK}")
_ALIAS_W = float(os.environ.get("STEP4E_ALIAS_W", "0.0"))       # 动作混叠惩罚 w（Markgraf 式20·`03` L97·默认 0.0=关=逐位等价；A/B 用 Markgraf 扫值 {0.1,0.5,1.0,2.0}·仅连续臂有盾→只接 ContinuousProjectionEnv）
_RATE_W = float(os.environ.get("STEP4E_RATE_W", "0.0"))         # action-rate 平滑惩罚 w（治 bang-bang 抖动·`03` L98·默认 0.0=关=逐位等价；仅连续臂·与 alias 正交[时间抖动 vs 空间分歧]）
# 🆕 第二条腿 rank1（`03` L173）：泊位精修门控治抖 r_rate。默认 off(None)=bit-identical·设值(如 0/0.1)=船进泊位区(‖ego−goal‖≤STEP4E_DOCK_R)时把治抖罚降到该值=放行入库急打舵对齐窄朝向门（治"接近后朝向捕获失败"·L171 点名却在 L172 漏做的那半）。复用 DOCK_R 作区半径(须 DOCK_R>0)·仅连续臂。
_RATE_DOCK_RAW = os.environ.get("STEP4E_RATE_DOCK", "").strip()
_RATE_DOCK = None if _RATE_DOCK_RAW == "" else float(_RATE_DOCK_RAW)
if _RATE_DOCK is not None:
    if not (math.isfinite(_RATE_DOCK) and _RATE_DOCK >= 0.0):
        raise SystemExit(f"🔒 STEP4E_RATE_DOCK 须有限非负（区内治抖罚权重），得 {_RATE_DOCK}")
    if _DOCK_R <= 0.0:
        raise SystemExit(f"🔒 STEP4E_RATE_DOCK={_RATE_DOCK} 需 STEP4E_DOCK_R>0（泊位区半径·复用 r_velocity 门同区）·得 DOCK_R={_DOCK_R}")
_COLREGS_W_CONT = float(os.environ.get("STEP4E_COLREGS_WEIGHT", "0.0"))   # 连续臂 r_colregs 权重·默认0.0=现状bit-identical·A/B=1.0·仅连续臂
if not (math.isfinite(_COLREGS_W_CONT) and _COLREGS_W_CONT >= 0.0):
    raise SystemExit(f"🔒 STEP4E_COLREGS_WEIGHT 须有限非负,得 {_COLREGS_W_CONT}")
_CONTINUOUS_SHIELD = os.environ.get("STEP4E_CONTINUOUS_SHIELD", "1").strip().lower() not in ("0", "", "false", "no", "off")   # 🆕 连续臂 SE-RL 盾开关（P0·L146）·默认 True=有盾=现状bit-identical·设0/off/false=连续无盾臂(施RL原动作·解耦崩塌混淆+why-RL C臂)·仅连续臂·进 config_conflict(防同TAG混shield静默跳过·不进config_sig=保续训bit-identical)·A/B 用 distinct TAG
if not _CONTINUOUS_SHIELD and any(v > 0.0 for v in (_PARK_W, _STEP_COST, _RATE_W, _ALIAS_W, _DWELL_W)):   # 🆕 P0 对抗审：无盾臂开连续专属 shaping(park/c_step/rate/alias/dwell·离散无盾拿不到)→"vs 离散无盾"解耦口径污染→fail-fast。⚠️此处只挡【常量】_RATE_W/_ALIAS_W/_DWELL_W；【退火】rate/alias_anneal_end 定义在 L160-161（此行之后）→退火通道在惩罚退火块另挡（L147 复审 C1 补·防退火把 rate_weight 从 0 ramp 起绕过本行）
    raise SystemExit(f"🔒 连续无盾臂(STEP4E_CONTINUOUS_SHIELD=0)不可同开连续专属 shaping：park={_PARK_W}/c_step={_STEP_COST}/rate={_RATE_W}/alias={_ALIAS_W}/dwell={_DWELL_W} 须全 0（离散无盾 Base/RR 拿不到这些项·带了则解耦崩塌归因被污染）。")
if not _CONTINUOUS_SHIELD and (_C_REACH != _C_REACH_DEF or _DOCK_R > 0.0 or _RATE_DOCK is not None):   # 🆕 第二条腿修法同理（`03` L172/L173）：无盾臂改 c_reach/开泊位门/开 rank1 治抖门→与离散无盾(默认口径)不对称=污染 why-RL 解耦→fail-fast
    raise SystemExit(f"🔒 连续无盾臂(STEP4E_CONTINUOUS_SHIELD=0)不可改 c_reach({_C_REACH}≠默认{_C_REACH_DEF}) 或开泊位门(dock_radius={_DOCK_R}) 或开 rank1 治抖门(rate_dock={_RATE_DOCK})：离散无盾用默认奖励口径·带了则解耦崩塌归因被污染。")
# 🆕 ρ0 朝目标锥（统一态势盾·方案①·`03` L145/L147·PhaseC 标定）：空旷水域(ρ0)也约束动作朝目标锥防游荡（崩铁证 97% ρ0 空旷游荡）。
#   STEP4E_GOAL_CONE_HALF=Φ【半角·度】(默认 off=None=逐位等价现状 bit-identical·锥关→goal_cone_action 返回 None→u_safe=u_desired)·内部转弧度传盾（ContinuousColregsProjection 期望 (0,π] rad）。
#   仅连续臂（离散/无盾臂拿不到锥）·进 config_conflict（防同TAG混锥配置静默跳过）·不进 config_sig（=保续训 bit-identical·同 continuous_shield 口径·A/B/sweep 用 distinct TAG）。
#   ⚠️ 四方钱图混写（复审 L147 ①④·同 lr_anneal L149 那条 caveat）：cone-on 连续臂【无法】与 cone-off 离散臂共写同一 jsonl——config_conflict 会因离散记录省略 goal_cone_half_deg(归 None) ≠ cur 值而【阻断 SystemExit】（park/c_step/shield/lr_anneal/well_B 全同款结构性模式·非锥特有）。→ PhaseD4 四臂钱图须 cone-on 连续臂【独立 TAG 单跑·绘图/聚合层合并两 jsonl】，不可靠 launcher 顶层 export 混写。PhaseC 连续臂-only（每 Φ 独立 TAG）不受影响。
_GOAL_CONE_RAW = os.environ.get("STEP4E_GOAL_CONE_HALF", "off").strip().lower()
if _GOAL_CONE_RAW in ("off", "", "none", "no", "0", "false"):
    _GOAL_CONE_HALF_DEG = None
else:
    try:
        _GOAL_CONE_HALF_DEG = float(_GOAL_CONE_RAW)
    except ValueError:
        raise SystemExit(f"🔒 STEP4E_GOAL_CONE_HALF 须为 'off' 或半角度数 ∈(0,180]（如 30/45/60），得 {_GOAL_CONE_RAW!r}")
    if not (math.isfinite(_GOAL_CONE_HALF_DEG) and 0.0 < _GOAL_CONE_HALF_DEG <= 180.0):
        raise SystemExit(f"🔒 STEP4E_GOAL_CONE_HALF（锥半角·度）须 ∈(0,180]，得 {_GOAL_CONE_HALF_DEG}")
_GOAL_CONE_HALF_RAD = math.radians(_GOAL_CONE_HALF_DEG) if _GOAL_CONE_HALF_DEG is not None else None   # 传盾用弧度；度=用户面/记录/config_conflict 口径（避免转换浮点噪声致假冲突）
from trb_env.usv_dynamics import PAPER_V_MAX as _PAPER_V_MAX   # v_max 单一真相源（论文 §VII=9.5·盾层 goal_v_floor 上界据此）
_GOAL_V_FLOOR = float(os.environ.get("STEP4E_GOAL_V_FLOOR", "2.0"))   # 锥内保底机动速度（仅锥开时用·默认 2.0=盾默认）
if not (math.isfinite(_GOAL_V_FLOOR) and 0.0 <= _GOAL_V_FLOOR <= _PAPER_V_MAX):   # 上界=v_max（复审⑤·与 Φ 双层守卫对称；否则 >v_max 在 Subproc worker 盾层晚爆 EOFError·父进程拿不到根因）
    raise SystemExit(f"🔒 STEP4E_GOAL_V_FLOOR（锥内保底速度）须 ∈ [0, v_max={_PAPER_V_MAX}]，得 {_GOAL_V_FLOOR}")
if not _CONTINUOUS_SHIELD and _GOAL_CONE_HALF_DEG is not None:   # 无盾臂 step 短路(if not self.shield)→不推状态机/不投影→锥(盾内 ρ0 分支机制)无从施加=inert·同 shaping fail-fast 保守口径（防误配）
    raise SystemExit("🔒 连续无盾臂(STEP4E_CONTINUOUS_SHIELD=0)不可同开 ρ0 朝目标锥（STEP4E_GOAL_CONE_HALF）：锥是盾内机制·无盾臂 step 短路不施（避免误以为在探锥）。")
# 🆕 腿1（`03` L150/L152）：态势感知观测增广 STEP4E_AUGMENT_RHO（ρ one-hot(6)+give_way_dir(1) 进连续臂 obs 27→34·让策略看见此刻态势→治抖/治违规·非治崩[15-35%·L150]）。
#   默认 off=27维=bit-identical·仅连续臂（透传 ContinuousProjectionEnv·内层 USVEnv 27维一字不动=离散臂忠实）·进 config_conflict 不进 config_sig（续训 bit-identical·A/B 用 distinct TAG）。
#   ⚠️ 四方钱图混写：augment-on 连续臂(34维) 无法与 augment-off 离散/无盾臂(27维) 共写同一 jsonl（config_conflict 阻断·同 cone/park/shield 结构性模式）→ PhaseD 须独立 TAG·绘图层合并·别 launcher 顶层 export 混写。
_AUGMENT_RHO = os.environ.get("STEP4E_AUGMENT_RHO", "0").strip().lower() in ("1", "true", "yes", "on")
if _AUGMENT_RHO and not _CONTINUOUS_SHIELD:   # 无盾臂 self._rho 恒 NO_CONFLICT=常数零信息→增广无意义+破四方 obs 维度平价（离散臂 27维·同 cone 无盾 fail-fast 保守口径）
    raise SystemExit("🔒 连续无盾臂(STEP4E_CONTINUOUS_SHIELD=0)不可同开态势感知增广（STEP4E_AUGMENT_RHO）：无盾臂 ρ 恒 NO_CONFLICT=常数零信息·增广无意义且破 obs 维度平价（离散臂 27维）。")
# 🆕 B1（`03` L153）：到达门朝向容差课程 STEP4E_ARR_SLACK_START（训练放宽终端朝向门起始半松弛量·度·退火到 0）。
#   崩塌根因=+50 窄门(±9.74°)探索撞不中→放宽门让崩种子撞中 +50 学会终端·退火回真门。
#   🔴【评估恒真门=诚实红线】：eval env（fac/replay_eval）不挂退火 callback、从不调 set_arrival_slack → 恒 slack=0=真门（报的到达率永远在真 ±9.74° 门上）。
#   默认 off=None=不安装退火 callback/env 收 0=训练【字节级不变】。仅连续臂（train_eval_one_continuous 透传 maker→term_checker）·进 config_conflict 不进 config_sig（A/B 用 distinct TAG·同 augment/cone）。
#   盾开关无关（有盾/无盾连续臂都有 term_checker·B1 是到达门课程非盾机制）→【不加 shield fail-fast】（与 shaping/cone 不同·B1 须对称施加于盾×崩塌解耦臂）。
_ARR_SLACK_START_RAW = os.environ.get("STEP4E_ARR_SLACK_START", "off").strip().lower()
if _ARR_SLACK_START_RAW in ("off", "", "none", "no", "0", "false"):
    _ARR_SLACK_START_DEG = None
else:
    try:
        _ARR_SLACK_START_DEG = float(_ARR_SLACK_START_RAW)
    except ValueError:
        raise SystemExit(f"🔒 STEP4E_ARR_SLACK_START 须为 'off' 或起始半松弛度数 ∈(0,60]（如 30/45），得 {_ARR_SLACK_START_RAW!r}")
    if not (math.isfinite(_ARR_SLACK_START_DEG) and 0.0 < _ARR_SLACK_START_DEG <= 60.0):   # 上界 60°（对【窄门】如 HOCR ±0.17=宽19.5° 安全:AngleInterval 崩溃点~78.8°;退火意图上限 45°。⚠️M2:宽门 goal 安全上界更低[=(π−0.05−真门宽)/2]→60° 未必够→由 term_checker clamp【按真 goal 宽】在 build 期 fail-fast 兜底[非静默/非 mid-train]·大集启用 B1 前须核 goal 朝向宽度）
        raise SystemExit(f"🔒 STEP4E_ARR_SLACK_START（到达门朝向容差起始·度）须 ∈(0,60]，得 {_ARR_SLACK_START_DEG}")
_ARR_SLACK_START_RAD = math.radians(_ARR_SLACK_START_DEG) if _ARR_SLACK_START_DEG is not None else None   # 传 env 用弧度；度=用户面/记录/config_conflict 口径（避免转换浮点噪声致假冲突·同锥）
_ARR_SLACK_ANNEAL_FRAC = float(os.environ.get("STEP4E_ARR_SLACK_ANNEAL_FRAC", "0.65"))   # slack 退火到 0 的步数比例（默认 0.65=前 65% 退完·后 35% 用真门让策略收敛真精度·`03` L153）
if not (math.isfinite(_ARR_SLACK_ANNEAL_FRAC) and 0.0 < _ARR_SLACK_ANNEAL_FRAC <= 1.0):
    raise SystemExit(f"🔒 STEP4E_ARR_SLACK_ANNEAL_FRAC 须 ∈(0,1]，得 {_ARR_SLACK_ANNEAL_FRAC}")
# 🆕 逆向起点课程（方案C-B·`03` L181·Florensa 2017·攻崩种子高速绕圈坏盆地）：STEP4E_START_FRAC = 训练时 ego 生到 goal+frac·(init−goal)。
#   frac<1=更靠泊位门（f→0贴门·教终端捕获）。**探针用固定值**（如 0.05·测 PPO 能否学会捕获=B 的 go/no-go）；完整课程后续加 Florensa 式自适应成功率退火。
#   默认 1.0=真起点=bit-identical；**评估 fac（:753）恒不传 → 恒真起点=诚实红线（同 arrival_slack）**。仅连续臂。start_v=课程重生速度（覆盖真实到门速度分布含高速·否则学假终端·L181 弱点②）。
_START_FRAC = float(os.environ.get("STEP4E_START_FRAC", "1.0"))
if not (math.isfinite(_START_FRAC) and 0.0 < _START_FRAC <= 1.0):
    raise SystemExit(f"🔒 STEP4E_START_FRAC 须 ∈(0,1]（1=真起点/→0贴门/0退化），得 {_START_FRAC}")
_START_V_RAW = os.environ.get("STEP4E_START_V", "off").strip().lower()
_START_V = None if _START_V_RAW in ("off", "", "none", "no") else float(_START_V_RAW)
if _START_V is not None and not (math.isfinite(_START_V) and _START_V >= 0.0):
    raise SystemExit(f"🔒 STEP4E_START_V 须为 'off' 或非负速度（m/s），得 {_START_V_RAW!r}")
if _START_V is not None and _START_FRAC >= 1.0:   # 🔴修(L182 对抗审 LOW)：start_v 仅课程 frac<1 有意义（frac>=1 走 bit-identical 分支用真 init 速度）→挡 silent no-op（镜像 rate_dock 需 dock_radius>0）
    raise SystemExit(f"🔒 STEP4E_START_V={_START_V} 但 STEP4E_START_FRAC={_START_FRAC}>=1（真起点）：start_v 仅逆向课程 frac<1 有意义·别静默失效。设 STEP4E_START_FRAC<1 或去掉 START_V。")
# 🆕 L185（user 2026-07-13）：训练目标【去朝向硬门】STEP4E_GOAL_IGNORE_ORIENT=1 → 1_goal 只判位置到达目标区域（忠实原文字面 "reached the goal area"·治崩种子被朝向门逼出的高速绕圈=两阶段 stage-1）。
#   默认 0=严格真门（位置+朝向±9.74°）=bit-identical。训练/评测【同配置】(eval fac/765 同传=主指标 reached 与训练一致)；评测层 in_box_aligned_steps 仍记位置+朝向严版→两指标都可报。连续臂专属·用 distinct TAG 防与严门 run 混写。
_GOAL_IGNORE_ORIENT = os.environ.get("STEP4E_GOAL_IGNORE_ORIENT", "0").strip() == "1"
_SHAPING_RADIUS = float(os.environ.get("STEP4E_SHAPING_RADIUS", "500.0"))   # 修法A 近场势半径 R_near（默认 500m·仅目标近场给梯度·不干扰避碰段）
# 学习率退火（`03` L88·治连续臂晚期漂移/跨种子方差）：默认 off=不安装 schedule/callback=训练【字节级不变】。
#   设为终点 lr（如 0 / 1e-6 / 3e-5）→ 从该臂当前恒定 lr（PPO=sb3默认3e-4 / SAC=STEP4E_LR）线性退火到终点、前 FRAC 比例步退完后恒终点。
#   分段训练鲁棒：用累积 num_timesteps（LRAnnealSchedule 忽略 SB3 锯齿 progress·见类 docstring）。四臂可对称施加（默认关·开则进 config_conflict/钱图须对称或披露）。
_LR_ANNEAL_RAW = os.environ.get("STEP4E_LR_ANNEAL", "off").strip().lower()
if _LR_ANNEAL_RAW in ("off", "", "none", "no", "false"):
    _LR_ANNEAL_END = None
else:
    try:
        _LR_ANNEAL_END = float(_LR_ANNEAL_RAW)                # 复审 LOW：解析失败给友好 🔒（非裸 ValueError·与其余旋钮口径一致）
    except ValueError:
        raise SystemExit(f"🔒 STEP4E_LR_ANNEAL 须为 'off' 或有限非负 lr 数值（如 0 / 1e-6 / 3e-5），得 {_LR_ANNEAL_RAW!r}")
_LR_ANNEAL_FRAC = float(os.environ.get("STEP4E_LR_ANNEAL_FRAC", "1.0"))      # 前 frac 比例总步数退完、之后恒终点（默认 1.0=全程退到终点）
if _LR_ANNEAL_END is not None:
    if not (math.isfinite(_LR_ANNEAL_END) and _LR_ANNEAL_END >= 0):   # 复审 MEDIUM：挡 inf/nan（inf<0/nan<0 均 False 会漏过→optimizer 拿 nan lr 静默烧出 nan 模型·同 target_q_clip 的 isfinite 守卫口径）
        raise SystemExit(f"🔒 STEP4E_LR_ANNEAL（学习率退火终点 lr）须为【有限非负】数值，得 {_LR_ANNEAL_END}")
    if not (0.0 < _LR_ANNEAL_FRAC <= 1.0):
        raise SystemExit(f"🔒 STEP4E_LR_ANNEAL_FRAC 须 ∈ (0,1]，得 {_LR_ANNEAL_FRAC}")
# 惩罚权重退火（penalty anneal·`03` L103·治"惩罚从第0步压脆弱种子=先学躲再没机会学到达"·先无罚让起飞·再 ramp 加罚）：
#   默认 off=不安装 callback=训练【字节级不变】。STEP4E_{RATE,ALIAS}_ANNEAL_END=退火【终点】权重（如 0.25/1.0）→从 0 经
#   hold-then-ramp 升到该值（覆盖常量 _RATE_W/_ALIAS_W·退火 on 时 factory 收初值=start=0）；'off'=不退火（用常量）。
#   RAMP_START_FRAC（默认 0.65=3.25M/5M·起飞实测 s3 站稳~3.5M·L103-续）=hold 段占比；ANNEAL_FRAC（默认 0.25=1.25M）=ramp 段占比。
def _parse_anneal_end(_envname):
    _raw = os.environ.get(_envname, "off").strip().lower()
    if _raw in ("off", "", "none", "no", "false"):
        return None
    try:
        _v = float(_raw)
    except ValueError:
        raise SystemExit(f"🔒 {_envname} 须为 'off' 或有限非负数（惩罚退火终点权重·如 0.25/1.0），得 {_raw!r}")
    if not (math.isfinite(_v) and _v >= 0):       # 同 LR 退火 isfinite 守卫（挡 inf/nan 经 env_method 进子进程毒化奖励）
        raise SystemExit(f"🔒 {_envname}（惩罚退火终点权重）须【有限非负】，得 {_v}")
    return _v
_RATE_ANNEAL_END = _parse_anneal_end("STEP4E_RATE_ANNEAL_END")
_ALIAS_ANNEAL_END = _parse_anneal_end("STEP4E_ALIAS_ANNEAL_END")
_PENALTY_RAMP_START_FRAC = float(os.environ.get("STEP4E_PENALTY_RAMP_START_FRAC", "0.65"))
_PENALTY_ANNEAL_FRAC = float(os.environ.get("STEP4E_PENALTY_ANNEAL_FRAC", "0.25"))
if _RATE_ANNEAL_END is not None or _ALIAS_ANNEAL_END is not None:
    if not _CONTINUOUS_SHIELD:   # 🆕 P0 复审 C1(L147)：连续无盾臂(shield=0)不可同开惩罚退火。L125 只挡【常量】_RATE_W/_ALIAS_W；退火从 0 ramp 起、常量恒 0→绕过 L125。而 r_rate 施加(usv_continuous_shield.py:195 `if self.rate_weight>0`)【无 source 守卫】→退火把 rate_weight 抬起后【在无盾臂真施加】=离散无盾 Base/RR 拿不到的连续专属 shaping→污染"盾诱发崩塌"与 why-RL 的解耦归因（端到端实证:shield=0+RATE_ANNEAL_END 原会漏过 import）。(alias 退火 L175 有 source=='projection' 守卫→无盾臂 inert，但同挡=与 L125 已挡常量 _ALIAS_W 的保守口径一致。)
        raise SystemExit(f"🔒 连续无盾臂(STEP4E_CONTINUOUS_SHIELD=0)不可同开惩罚退火：rate_anneal_end={_RATE_ANNEAL_END} / alias_anneal_end={_ALIAS_ANNEAL_END} 须全不设（离散无盾拿不到连续专属 shaping；rate 退火尤其=r_rate 无 source 守卫会在无盾臂真施加·污染解耦归因）。")
    if not (0.0 <= _PENALTY_RAMP_START_FRAC < 1.0):
        raise SystemExit(f"🔒 STEP4E_PENALTY_RAMP_START_FRAC 须 ∈ [0,1)，得 {_PENALTY_RAMP_START_FRAC}")
    if not (0.0 < _PENALTY_ANNEAL_FRAC <= 1.0):
        raise SystemExit(f"🔒 STEP4E_PENALTY_ANNEAL_FRAC 须 ∈ (0,1]，得 {_PENALTY_ANNEAL_FRAC}")
    # 🔒 防呆（`03` L108·复审主窗口+A3 抓的 MINOR footgun）：同一惩罚【常量 STEP4E_{RATE,ALIAS}_W≠0 + 退火 _ANNEAL_END】同设=矛盾配置。
    #   退火 ON 时 factory 收初值 0（:756 _rate_w_init=0.0）、callback 经 set_penalty_weight 覆盖式从 0 ramp 到终点 → 常量分量被【完全丢弃】
    #   从未进训练；但 jsonl（:929）/config_sig（:863）仍记 rate_weight=_RATE_W=幻影元数据误标（账本记了个没真用上的数·破"jsonl 自描述"）。
    #   fail-fast 挡掉歧义组合（镜像 PPO ent 退火不对称那道门 :761）→ 强制二选一：要退火别设常量 _W，要常量别设 _ANNEAL_END。
    if _RATE_ANNEAL_END is not None and _RATE_W != 0.0:
        raise SystemExit(f"🔒 STEP4E_RATE_W={_RATE_W}（常量 rate 惩罚）与 STEP4E_RATE_ANNEAL_END={_RATE_ANNEAL_END}（rate 退火终点）不可同设："
                         "退火 ON 时常量被静默丢弃（factory 从 0 起 ramp）、但 jsonl 仍会记常量值=元数据误标。"
                         "要退火→【别设 STEP4E_RATE_W】（默认 0）；要常量→【别设 STEP4E_RATE_ANNEAL_END】。")
    if _ALIAS_ANNEAL_END is not None and _ALIAS_W != 0.0:
        raise SystemExit(f"🔒 STEP4E_ALIAS_W={_ALIAS_W}（常量 alias 惩罚）与 STEP4E_ALIAS_ANNEAL_END={_ALIAS_ANNEAL_END}（alias 退火终点）不可同设："
                         "退火 ON 时常量被静默丢弃（factory 从 0 起 ramp）、但 jsonl 仍会记常量值=元数据误标。"
                         "要退火→【别设 STEP4E_ALIAS_W】（默认 0）；要常量→【别设 STEP4E_ALIAS_ANNEAL_END】。")
# 🔒 第二条腿 rank1 防呆（`03` L176 对抗复审补·2 独立 agent CONFIRMED）：rate_dock 门控【只在 r_rate 施加时生效】——
#   usv_continuous_shield.py:247 `if self.rate_weight>0.0` 短路整个 r_rate 块(含 rate_dock 门)，故 rate_weight 全程 0（既没设常量 STEP4E_RATE_W 又没开 rate 退火）时 rate_dock=纯静默 no-op（记进 jsonl 却从不施加=白烧一次+provenance 误标开了治抖门）。
#   本项目对同类 silent no-op(dock_radius=0)已 fail-fast(:148)→此支须对称挡。⚠️ 守卫【只能在此】(同握 _RATE_W 与 _RATE_ANNEAL_END)——不能加进盾构造期，因【惩罚退火】合法配置下构造期 rate_weight 本就是 0(:1001 起 ramp)、盾无法区分"真 no-op"vs"退火后抬起"。
if _RATE_DOCK is not None and _RATE_W == 0.0 and _RATE_ANNEAL_END is None:
    raise SystemExit(f"🔒 STEP4E_RATE_DOCK={_RATE_DOCK}（泊位区治抖门）需 rate_weight 生效：请设 STEP4E_RATE_W>0（常量治抖罚·如 1.0）或 STEP4E_RATE_ANNEAL_END（rate 退火）。"
                     f"否则 usv_continuous_shield 的 r_rate 块被 `rate_weight>0` 短路整段跳过 → rate_dock 纯静默空转（记进 jsonl 却从不施加=白烧+provenance 误标）。")
LOG_CURVES =os.environ.get("STEP4E_LOG_CURVES", "0") not in ("0", "", "false", "no")   # L39 决定性诊断：opt-in 记录每 rollout 的 VecNormalize reward 滚动方差 + PPO 内部曲线（区分 setup-artifact vs 内禀；默认关、不影响既有 run）
# Node L CAT5 示例轨迹（D42-L2-续·`03` L58）：【末段】评估对这些【测试场景索引】记逐步 ego/他船航迹(+ρ)供给路/迎面/追越例图。
# 默认 "0,1,2"（"第一次跑全记齐不重复跑"·D42-Lschema⑦）；STEP4E_TRAJ_IDXS="off"/"none"/"" → None=生产钱图零轨迹（同 L1b 默认）。
# additive：仅【末段 c==n_seg-1】对【这几个索引】记 traj，钱图列已证逐位不变(L1b CAT5)；落进 final_per、绝不入钱图 5/6 列聚合。
_TRAJ_IDXS_ENV = os.environ.get("STEP4E_TRAJ_IDXS", "0,1,2").strip()
if _TRAJ_IDXS_ENV.lower() in ("", "off", "none"):
    TRAJ_EXAMPLE_IDXS = None                          # 关闭=生产钱图零轨迹记录（同 L1b 默认）
else:
    try:                                              # 友好 fail-fast（非法环境变量早爆+清晰提示·非裸 traceback）
        TRAJ_EXAMPLE_IDXS = frozenset(int(x) for x in _TRAJ_IDXS_ENV.split(",") if x.strip() != "")
    except ValueError:
        raise SystemExit(f"⚠️ STEP4E_TRAJ_IDXS 非法（须逗号分隔非负整数如 '0,1,2'，或 off/none 关闭）：得到 {_TRAJ_IDXS_ENV!r}")
# =====================================================================================

# 四方 = (显示名, env 类型, colregs_weight)。Continuous-safe(SAC+连续投影盾) = Phase 3 Node C 接入（C2）。
PARTIES = [
    ("Base",            "unshielded", 0.0),
    ("Rule-reward",     "unshielded", 1.0),
    ("Discrete-safe",   "shielded",   1.0),
    ("Continuous-safe", "continuous", 0.0),   # SAC + ContinuousProjectionEnv；colregs_weight 默认 0.0(可经 STEP4E_COLREGS_WEIGHT 覆盖·A/B 复活 r_colregs)（丢 r_colregs，D37-B/Node B）
]

_SDIR = os.environ.get("STEP4E_SDIR", "/tmp/trb_scenarios_pool")   # 场景缓存；/tmp 太小可 STEP4E_SDIR 指数据盘
_BASE = ("https://gitlab.lrz.de/tum-cps/commonocean-scenarios/-/raw/main/scenarios/"
         "HandcraftedTwoVesselEncounters_01_24/ZAM_AAA-1_20240121_T-{}.xml")
# 🆕 均衡数据集模式（`03` L113-L115·BLOCKER 追越补救）：设 STEP4E_MANIFEST=manifest.json → 用【均衡数据集】(对遇+交叉+追越各≈667·
#   含同一条线超车)·预定义分层 70/30·覆盖三大 give-way；不设(默认)→ 完全走原 strided-T 选取(逐字节不变)。
#   head-on/crossing 走 T-id 下载复用·overtaking 走 STEP4E_BALANCED_DIR/OT-*.xml(默认=manifest 同目录·须上传)。
_MANIFEST = os.environ.get("STEP4E_MANIFEST", "").strip()
_BALANCED_DIR = os.environ.get("STEP4E_BALANCED_DIR", "").strip()
_DATASET_SIG = os.path.basename(_MANIFEST) if _MANIFEST else "strided"   # 自描述：均衡 manifest 名 / "strided"（jsonl 记·续跑/钱图溯源·防混）
_RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "结果")
# 输出按模式隔离：SMOKE→*_smoke、FULL→无后缀；STEP4E_TAG 可显式覆盖。
_TAG = os.environ.get("STEP4E_TAG", "_smoke" if SMOKE else "")
# 🔒 PPO opt-in 隔离闸门（红队 MEDIUM·L67-续3）：连续臂走 PPO 是诊断/A-B（与 SAC 同 party 名 "Continuous-safe"）→ 用生产 TAG
#   会静默覆盖 SAC 的 (party,seed) record、污染四方钱图且表面看不出。强制 PPO 用诊断 TAG（含 ppo/diag/probe/ab）或显式 ACK。
_CONTINUOUS_ALGO = os.environ.get("STEP4E_CONTINUOUS_ALGO", "sac").lower()
if _CONTINUOUS_ALGO == "ppo" and not SMOKE:
    _diag_tag = any(k in _TAG.lower() for k in ("ppo", "diag", "probe", "ab"))
    if not _diag_tag and os.environ.get("STEP4E_CONTINUOUS_ALGO_ACK") != "1":
        raise SystemExit(
            f"🔒 STEP4E_CONTINUOUS_ALGO=ppo 但 TAG='{_TAG}' 非诊断 TAG（须含 ppo/diag/probe/ab）→ 会静默覆盖 SAC 的"
            " Continuous-safe 钱图记录。请用诊断 TAG（如 STEP4E_TAG=_ppo_ab）跑 PPO A/B；确需覆盖请 STEP4E_CONTINUOUS_ALGO_ACK=1。")
# 🆕 L190 修（对抗审 wf wy3rlm90p HIGH-2）：热启动仅连续 PPO 臂接线（SAC maker 无 warmstart_ckpt 形参）→
#   设了 STEP4E_WARMSTART_CKPT 却用默认 algo=sac = 热启动静默丢弃、但 provenance 仍记路径=假溯源(违"训练流程如实可查"红线)→ fail-fast。
if _WARMSTART_CKPT and _CONTINUOUS_ALGO != "ppo":
    raise SystemExit(f"🔒 STEP4E_WARMSTART_CKPT 已设但 STEP4E_CONTINUOUS_ALGO='{_CONTINUOUS_ALGO}'≠ppo → 热启动仅连续 PPO 臂支持·"
                     "SAC 臂会静默不施+provenance 假记录。请设 STEP4E_CONTINUOUS_ALGO=ppo（+诊断TAG）或清空 STEP4E_WARMSTART_CKPT。")
_PARTIAL = os.path.join(_RESULT_DIR, f"step4e_partial{_TAG}.jsonl")
_TABLE3 = os.path.join(_RESULT_DIR, f"table3{_TAG}.txt")
_CKPT_DIR = os.path.join(_RESULT_DIR, "checkpoints")            # L1c：每 (party,seed) 训后存 model+VecNorm（不重跑总保险·D42-Lschema CAT1）


# ---------------- 纯逻辑（可导入测试；无副作用）----------------
def make_split(n_total: int, test_frac: float, split_seed: int = 0, pool_size: int | None = None):
    """种子化 shuffle 后切 70/30 → (train_ids, test_ids)。分散（非尾块）、互不重叠、可复现。

    pool_size: 给定且 > n_total → 从 [0, pool_size) **跨全库均匀铺开**抽 n_total 个再切（避免缩小实验
               聚集采样前 N 个 = "偏向失败的代理"，03 L29；n=200,pool=2000 → 0,10,…,1990 = run_validation
               的多样选取）。None（默认）→ 用前 n_total 个（向后兼容旧行为）。n_total ≥ pool_size → 等价
               range(n_total)（全量无影响）。`i*pool//n` 公式对任意 n_total 都铺开、无"中段又聚集"边角。
    审核 L21 M1/M3：assert 防边界塌空 + 重叠泄漏。
    """
    if not (0 < test_frac < 1):
        raise ValueError(f"test_frac 须 ∈ (0,1)，得到 {test_frac}")
    if pool_size and pool_size > n_total:
        ids = [i * pool_size // n_total for i in range(n_total)]   # 跨全库均匀铺开 → 多样性（03 L29）
        assert len(set(ids)) == n_total, f"等距选取索引重复（n_total={n_total}, pool_size={pool_size}）——不应发生"
    else:
        ids = list(range(n_total))                                # 默认：前 n_total 个（向后兼容）
    random.Random(split_seed).shuffle(ids)        # 固定种子 → 分散且可复现
    n_test = int(round(n_total * test_frac))
    if not (0 < n_test < n_total):
        raise ValueError(f"划分非法：n_total={n_total} → n_test={n_test}（需 0<n_test<n_total；"
                         f"N_TOTAL 太小或 test_frac 极端）")
    test = sorted(ids[:n_test])
    train = sorted(ids[n_test:])
    assert not (set(train) & set(test)), "train/test 重叠（泄漏）——不应发生"
    assert len(train) + len(test) == n_total, "train+test 数不等于 n_total——不应发生"
    return train, test


def anneal_ent_coef(start, end, anneal_steps, num_timesteps):
    """ent_coef 线性退火当前值：num_timesteps=0→start；≥anneal_steps→end（端点 clamp）。

    与 _EntAnneal callback 共用此纯逻辑（可单元测试、无需起 PPO；03 D25 + 审核加固）。
    anneal_steps≤0 视为 1 防除零；START==END → 恒返 start（关退火、复现旧常量配方）。
    返回值恒在 [min(start,end), max(start,end)] 内（不 overshoot）。
    """
    n = max(1.0, float(anneal_steps))
    p = min(1.0, max(0.0, float(num_timesteps) / n))
    return float(start) + (float(end) - float(start)) * p


def resolve_vecnorm_kwargs(base, clip_reward_env, norm_reward_env=None):
    """合成 VecNormalize kwargs：clip_reward_env（STEP4E_CLIP_REWARD）非空则覆盖 clip_reward；
    norm_reward_env（STEP4E_NORM_REWARD）= '0'/'false'/'no' 则关掉奖励归一化（norm_reward=False）。

    返回 (kwargs_dict, 有效 clip_reward)。不修改入参 base。
    - clip 默认空 → 吃 sb3 默认 10.0（= 已验证配方 D22）；'50' → clip_reward=50.0（消融，03 审核#1）。
    - norm_reward 默认不设 → 保持 base 的 True（= 已验证配方）；'0' → False
      （消融：关奖励归一化，验"reward 除以 return-std 压平稀疏 +50 → 种子分裂"假设，03 L27/审核#2 真身）。
    ⚠️ clip_reward≤0 会把喂 PPO 的 reward 静默归零/恒定化 → 防呆硬拒（fail-fast 优于静默烧算力，03 审核 Agent B）。
    """
    kw = dict(base)
    if clip_reward_env:
        cr = float(clip_reward_env)
        if cr <= 0:
            raise ValueError(f"STEP4E_CLIP_REWARD 须 >0（得到 {cr}）：clip_reward≤0 会归零/恒定化 reward → 训练学不到信号")
        kw["clip_reward"] = cr
    if norm_reward_env is not None and str(norm_reward_env).strip().lower() in ("0", "false", "no"):
        kw["norm_reward"] = False
    return kw, kw.get("clip_reward", 10.0)


def env_cls_of(kind: str):
    """'shielded' → ShieldedUSVEnv（有盾）；'unshielded' → UnshieldedUSVEnv（Base/RR）；
    'continuous' → ContinuousProjectionEnv（Continuous-safe = SAC + 连续投影盾，C2）。"""
    from trb_env.usv_shield import ShieldedUSVEnv, UnshieldedUSVEnv
    if kind == "continuous":
        from trb_env.usv_continuous_shield import ContinuousProjectionEnv
        return ContinuousProjectionEnv
    if kind == "shielded":
        return ShieldedUSVEnv
    if kind == "unshielded":
        return UnshieldedUSVEnv
    raise ValueError(f"env 类型须 ∈ {{shielded, unshielded, continuous}}，得到 {kind!r}")


def select_parties(spec, parties=None):
    """STEP4E_PARTIES='Base,Discrete-safe' → 过滤三方（保序）；None/空 → 全部。并行分片用。"""
    parties = list(PARTIES if parties is None else parties)
    if not spec or not spec.strip():
        return parties
    want = [s.strip() for s in spec.split(",") if s.strip()]
    valid = {p[0] for p in parties}
    bad = [w for w in want if w not in valid]
    if bad:
        raise ValueError(f"未知 party {bad}；合法 = {sorted(valid)}")
    return [p for p in parties if p[0] in want]


def agg_mean_std(values):
    """多种子聚合 → (均值, 样本标准差 ddof=1)。空→(nan,nan)；单值→(v,0.0)。"""
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"))
    m = sum(values) / n
    if n == 1:
        return (m, 0.0)
    var = sum((v - m) ** 2 for v in values) / (n - 1)   # 样本标准差（多种子统计，对齐 USV 方法论）
    return (m, math.sqrt(var))


def read_records(partial_path: str):
    """读 partial jsonl 所有记录（坏行跳过、容错）。"""
    recs = []
    if os.path.exists(partial_path):
        with open(partial_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return recs


def done_keys(partial_path: str):
    """读已完成的 (party, seed) 集（断点续跑跳过）。"""
    return {(r["party"], int(r["seed"])) for r in read_records(partial_path)
            if "party" in r and "seed" in r}


def config_conflict(records, total_steps: int, n_total: int, pool_size: int | None = None, n_seg: int | None = None,
                    well_shaping_weight: float = 0.0, shaping_radius: float = 500.0, gamma: float = 0.99,
                    lr_anneal_end: float | None = None, lr_anneal_frac: float = 1.0,
                    xtrack_weight: float = 0.0, xtrack_radius: float = 80.0,
                    alias_weight: float = 0.0, rate_weight: float = 0.0,
                    rate_anneal_end: float | None = None, alias_anneal_end: float | None = None,
                    penalty_ramp_start_frac: float = 0.65, penalty_anneal_frac: float = 0.25,
                    dataset: str = "strided",
                    park_weight: float = 0.0, park_radius: float = 400.0, park_v_target: float = 4.0,
                    c_step: float = 0.0,
                    c_dwell: float = 0.0, w_dwell: float = 90.0, h_dwell: float = 0.52, dwell_radius: float = 250.0, b_dwell: float = 0.0,
                    c_reach: float = 1.5, dock_radius: float = 0.0, v_dock: float = 2.5,
                    rate_dock: float | None = None,
                    continuous_shield: bool = True,
                    goal_cone_half_deg: float | None = None, goal_v_floor: float = 2.0,
                    augment_rho: bool = False,
                    arr_slack_start_deg: float | None = None,
                    warmstart_ckpt: str | None = None):
    """断点续跑配置守卫：已存记录的 (steps,n_total,pool_size,n_seg) 与当前不符 → 返回冲突集（main 据此报错，避免混配置）。

    纳入 pool_size → 不同场景【选取方式】(strided 跨全库 vs 聚集前N) 不会被静默混进同一 Table III（03 L29）。
    纳入 n_seg（`03` L58 #2）→ 不同分段数 = 不同 step 网格（step_done=(c+1)*(total//n_seg)）；中途改 STEP4E_NSEG
       续跑同 jsonl 会产生【种子间不对齐的 step 网格】→ 学习曲线 agg_trend 的 mean±std band 误导（plot 侧
       warn_misaligned_grids 另有运行时告警·此处为续跑期【硬拒】）。钱图【数值表】按 (party,seed) 取终值不受影响，
       但学习曲线子图是论文图 → 一并拒混。
    旧记录无某字段（pool_size/n_seg）→ 取 None，与带该字段的新配置判为不同（正确：不可混；C3 前删 stale jsonl 即可）。
    纳入 well_shaping_weight/shaping_radius/gamma（修法A 进门势·`03` L82）→ well_B A/B 各臂用同 TAG 续跑被【硬拒】，
       防 well_B=0 与 well_B=200 记录静默混进同一 jsonl（=断点续跑 skip 已完成 (party,seed) 致漏跑+混表·接线复审 MEDIUM）。
       缺字段（旧记录）→ 归一化为系统默认（0.0/500.0/0.99·=接线前隐含值），不误判同配置 run 为冲突（保旧 jsonl/测试兼容）。
    纳入 lr_anneal_end/lr_anneal_frac（学习率退火·`03` L88）→ 退火 on/off 或不同终点/比例不混进同一 jsonl。
       lr_anneal_end 缺字段（旧记录）/关闭 → None（=恒定 lr·=退火前隐含值）·不与 float 误等；frac 缺 → 1.0。
    纳入 xtrack_weight/xtrack_radius（对症 横向进带势·`03` L88）→ well_X on/off 或不同 R_lat 不混进同一 jsonl。
       缺字段（旧记录）→ 归一化为系统默认（0.0/80.0·=接线前隐含值），不误判（保旧 jsonl/测试兼容）。
    """
    prior = {(r.get("steps"), r.get("n_total"), r.get("pool_size"), r.get("n_seg"),
              float(r.get("well_shaping_weight", 0.0)), float(r.get("shaping_radius", 500.0)), float(r.get("gamma", 0.99)),
              r.get("lr_anneal_end"), float(r.get("lr_anneal_frac", 1.0)),   # lr_anneal_end 可 None(关)/float(终点)·直接比不 float()-wrap
              float(r.get("xtrack_weight", 0.0)), float(r.get("xtrack_radius", 80.0)),   # 对症 横向进带势（`03` L88）
              float(r.get("alias_weight", 0.0)), float(r.get("rate_weight", 0.0)),   # 动作混叠(L97)+action-rate(L98) on/off 不混进同一 jsonl·缺字段旧记录归一化 0.0 保兼容
              r.get("rate_anneal_end"), r.get("alias_anneal_end"),   # 惩罚退火 on/off/终点不混进同一 jsonl（None=关·`03` L103·直接比不 float-wrap）
              float(r.get("penalty_ramp_start_frac", 0.65)), float(r.get("penalty_anneal_frac", 0.25)),   # 退火 hold/ramp 比例·缺字段旧记录归一化默认(=接线前隐含)保兼容
              str(r.get("dataset", "strided")),   # 🆕 数据集模式（均衡 manifest 名/strided·`03` L116 二审 MEDIUM 修）→ strided 与 manifest 记录【绝不静默混进同一 jsonl】(尤 FULL 默认两者 n_total=2000 数值签名相同·只靠 TAG 兜底不够)·旧记录缺→"strided"(=接线前隐含·兼容)
              float(r.get("park_weight", 0.0)), float(r.get("park_radius", 400.0)), float(r.get("park_v_target", 4.0)),   # 想法B Φ_park 终端保速势 on/off 不混进同一 jsonl（`03` L111/L112 二审 MINOR 补·连续臂专属·缺字段旧记录归一化默认 0.0/400.0/4.0 保兼容）
              float(r.get("c_step", 0.0)),   # 修法C 每步生存成本 on/off 不混进同一 jsonl（`03` L123·连续臂专属·非PBRS·缺字段旧记录归一化默认 0.0 保兼容）
              float(r.get("c_dwell", 0.0)), float(r.get("w_dwell", 90.0)), float(r.get("h_dwell", 0.52)), float(r.get("dwell_radius", 250.0)), float(r.get("b_dwell", 0.0)),   # r_dwell 入库赤字滞留成本 on/off/参数 不混进同一 jsonl（`03` L161/L162·连续臂专属·非PBRS·缺字段旧记录归一化默认保兼容·须与 cur 同序）
              float(r.get("c_reach", 1.5)), float(r.get("dock_radius", 0.0)), float(r.get("v_dock", 2.5)),   # 🆕 第二条腿修法 c_reach/dock_radius/v_dock on/off 不混进同一 jsonl（`03` L172·连续臂专属·缺字段旧记录/离散归一化 1.5/0.0/2.5=接线前隐含 保兼容·须与 cur 同序）
              r.get("rate_dock"),   # 🆕 第二条腿 rank1 泊位门控治抖（`03` L173·连续臂专属·None-able 直接比不 float-wrap·缺字段旧记录/离散→None=off 隐含·须与 cur 同序）
              bool(r.get("continuous_shield", True)),   # 🆕 P0 SE-RL 盾 on/off 不混进同一 jsonl（L146·连续臂专属·防同TAG翻shield静默跳过；缺字段旧记录/离散记录→True=有盾隐含·四方正常run 不误触）
              r.get("goal_cone_half_deg"), float(r.get("goal_v_floor", 2.0)),   # 🆕 ρ0 朝目标锥 Φ(度)/v_floor on/off 不混进同一 jsonl（PhaseC·L147·连续臂专属·缺字段旧记录/离散→None=关隐含·v_floor 仅锥开有意义·缺归默认 2.0 保兼容）
              bool(r.get("augment_rho", False)),   # 🆕 腿1(L150/L152)：态势感知增广 on/off 不混进同一 jsonl（连续臂专属·缺字段旧记录/离散→False=关隐含·obs 27维 vs 34维 混写=脏钱图）
              r.get("arr_slack_start_deg"),   # 🆕 B1(L153)：到达门朝向课程 slack 起始度 on/off 不混进同一 jsonl（连续臂专属·缺字段旧记录/离散/off→None=关隐含·度=canonical 口径避浮点噪声）
              r.get("warmstart_ckpt"))   # 🆕 L190(自审补漏·15agent对抗审也漏)：热启动源 on/off 不混进同一 jsonl（连续PPO臂专属·此为第37元·末元·缺字段旧记录/离散/off→None=从零训练隐含）
                                          #   🔴 命门:热启动 vs 从零是【两种不同训练流程】·同TAG混写=汇总表静默混算=摧毁"全10种子统一施"方法论红线且从聚合数看不出来→硬拒(同 shield/augment/cone 既定模式)
             for r in records if r.get("steps") is not None}
    cur = (total_steps, n_total, pool_size, n_seg,
           float(well_shaping_weight), float(shaping_radius), float(gamma),
           lr_anneal_end, float(lr_anneal_frac),
           float(xtrack_weight), float(xtrack_radius),
           float(alias_weight), float(rate_weight),
           rate_anneal_end, alias_anneal_end,
           float(penalty_ramp_start_frac), float(penalty_anneal_frac),
           str(dataset),
           float(park_weight), float(park_radius), float(park_v_target),
           float(c_step),
           float(c_dwell), float(w_dwell), float(h_dwell), float(dwell_radius), float(b_dwell),   # r_dwell（`03` L161/L162·连续臂专属·非PBRS·须与 prior r.get 同序）
           float(c_reach), float(dock_radius), float(v_dock),   # 🆕 第二条腿修法（`03` L172·连续臂专属·须与 prior r.get 同序）
           rate_dock,   # 🆕 第二条腿 rank1 泊位门控治抖（`03` L173·None-able·须与 prior r.get 同序）
           bool(continuous_shield),   # 🆕 P0 SE-RL 盾 on/off（L146）
           goal_cone_half_deg, float(goal_v_floor),   # 🆕 ρ0 朝目标锥 Φ(度)/v_floor（PhaseC·L147）
           bool(augment_rho),   # 🆕 腿1(L150/L152)：态势感知增广 on/off（连续臂专属·此为第35元·`03` L176 订正:旧"25→26"注释stale·后续插了 c_dwell组5+c_reach组3+rate_dock1→元组现共36·按位置比较不按注释）
           arr_slack_start_deg,   # 🆕 B1(L153)：到达门朝向课程 slack 起始度 on/off（连续臂专属·第36元·`03` L176 订正:旧"26→27"stale）
           warmstart_ckpt)   # 🆕 L190(自审补漏)：热启动源 on/off（连续PPO臂专属·此为第37元·末元·须与 prior r.get 同序）
    return prior - {cur} if prior and prior != {cur} else set()


# ---------------- 并行安全 I/O ----------------
def append_record(path, rec):
    """并行安全增量写一条 jsonl 记录（fcntl.flock 串行化追加，多任务并发写同文件不打架/不黏行）。"""
    import fcntl
    line = json.dumps(rec, ensure_ascii=False) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(line)
            f.flush()
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def write_atomic(path, content):
    """原子写文本（.<pid>.tmp + os.replace）：并发写同一 table3 互不见半成品、last-writer-wins 完整文件。"""
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(content)
    os.replace(tmp, path)


def _download(ids, workers=16):
    """并行下载场景（16 路并发 + 30s 超时 + 进度 + 原子写）。

    场景在 gitlab.lrz.de（德国），跨境下 2000 个文件顺序很慢 → 多线程并发 + 超时防卡死。
    STEP4E_DL_WORKERS 可调并发数。已缓存（>1KB）的跳过。
    """
    import socket
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor
    socket.setdefaulttimeout(30)                               # urllib 默认无超时 → 单文件坏连接会永久挂；强制 30s
    workers = int(os.environ.get("STEP4E_DL_WORKERS", workers))
    os.makedirs(_SDIR, exist_ok=True)

    def _one(n):
        dst = f"{_SDIR}/T-{n}.xml"
        if os.path.exists(dst) and os.path.getsize(dst) > 1000:
            return (dst, None)                                 # 已缓存
        try:
            tmp = f"{dst}.{os.getpid()}.{n}.tmp"               # 进程+id 隔离 + 原子 rename：并发/被杀都安全
            urllib.request.urlretrieve(_BASE.format(n), tmp)
            os.replace(tmp, dst)
            return (dst, None)
        except Exception:                                      # noqa: BLE001
            return (None, n)

    paths, fail = [], []
    total = len(ids)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for j, (p, f) in enumerate(ex.map(_one, ids)):         # ex.map 保序
            if p:
                paths.append(p)
            if f is not None:
                fail.append(f)
            if total > 100 and (j + 1) % 200 == 0:             # 大批量打进度（否则看着像卡住）
                print(f"  …下载 {j + 1}/{total}（失败 {len(fail)}）", flush=True)
    return paths, fail


def load_manifest_split(manifest_path, balanced_dir=None):
    """均衡数据集模式（`03` L113-L115）：从 manifest.json 读【预定义分层 70/30】训练/测试路径。
       返回 (train_paths, test_paths, info)。head-on/crossing 走 _download(T-id·复用 gitlab 缓存)·
       overtaking 走 balanced_dir/OT-*.xml（须上传到服务器·缺则 fail-fast）。任何缺失硬停（不静默缩水污染钱图）。"""
    import json as _json
    if not os.path.exists(manifest_path):
        raise SystemExit(f"🔒 STEP4E_MANIFEST 指向的文件不存在：{manifest_path}")
    bdir = balanced_dir or os.path.dirname(os.path.abspath(manifest_path))
    with open(manifest_path) as _fh:
        man = _json.load(_fh)
    for _typ in ("head_on", "crossing", "overtaking"):           # 结构校验（fail-fast·防 manifest 残缺静默漏一类）
        if _typ not in man or "train" not in man[_typ] or "test" not in man[_typ]:
            raise SystemExit(f"🔒 manifest 缺 {_typ}.train/test（须含 head_on/crossing/overtaking 三类各 train+test）")

    def _resolve(split):
        # head-on + crossing：旧 T-id → 下载（复用缓存）
        t_ids = [int(x) for x in man["head_on"][split]] + [int(x) for x in man["crossing"][split]]
        t_paths, t_fail = _download(t_ids)
        # overtaking：新造 OT 文件 → balanced_dir（须存在）
        ot_paths, ot_miss = [], []
        for f in man["overtaking"][split]:
            p = os.path.join(bdir, os.path.basename(str(f)))     # basename 防 manifest 写绝对路径越界
            (ot_paths if os.path.exists(p) and os.path.getsize(p) > 1000 else ot_miss).append(p if os.path.exists(p) else f)
        return t_paths, t_fail, ot_paths, ot_miss, len(t_ids), len(man["overtaking"][split])

    tr_t, tr_tf, tr_ot, tr_otm, tr_tn, tr_otn = _resolve("train")
    te_t, te_tf, te_ot, te_otm, te_tn, te_otn = _resolve("test")
    # 追越文件缺失 = 实验有效性硬伤（追越是 BLOCKER 补救的核心）→ 硬停
    if tr_otm or te_otm:
        raise SystemExit(
            f"🔒 均衡池缺【追越 OT 场景】：训练缺 {len(tr_otm)} + 测试缺 {len(te_otm)}（找于 {bdir}）。\n"
            f"   追越是 BLOCKER 补救核心·缺失=覆盖不全·中止。请把 OT-*.xml 上传到 {bdir}（或设 STEP4E_BALANCED_DIR）。\n"
            f"   缺失样例：{(tr_otm + te_otm)[:3]}")
    # 测试集 T-id 下载缺额守卫（同 strided 模式·>5% 硬停防 Table III 失真）
    if len(te_t) < te_tn * 0.95:
        raise SystemExit(f"🔒 均衡池测试集 head-on/crossing 下载缺额过大：{len(te_t)}/{te_tn}（缺 {len(te_tf)}）→ 中止（查网络/gitlab）。")
    if tr_tf:
        print(f"⚠️ 均衡池训练集 head-on/crossing {len(tr_tf)} 个下载失败 → 用 {len(tr_t)}/{tr_tn}（降级但有效）", flush=True)
    train_paths = tr_t + tr_ot
    test_paths = te_t + te_ot
    # 🆕 test_meta（type+file·平行 test_paths·L146 分层彻底堵洞）：类型由【id 本身】查 manifest 得（非靠位置），
    #    故与 te_t 的下载丢弃/顺序无关、恒正确；下游 evaluate 返回后盖进 final_per record → 分层零位置再推导。
    _ho_ids = {int(x) for x in man["head_on"]["test"]}
    _cr_ids = {int(x) for x in man["crossing"]["test"]}
    _ov = _ho_ids & _cr_ids                                       # 对抗审建议:把"两类 test id 互斥"隐性契约显式化(否则重叠 id 被判 head_on 吞 crossing)
    if _ov:
        raise SystemExit(f"🔒 manifest head_on/crossing test id 重叠 {sorted(_ov)[:5]}… → 类型标签歧义，中止（两类须互斥）。")
    def _type_of_tid(path):
        b = os.path.basename(path)
        try:
            i = int(b[2:-4]) if b.startswith("T-") and b.endswith(".xml") else None
        except ValueError:
            i = None
        if i in _ho_ids: return "head_on"
        if i in _cr_ids: return "crossing"
        return "unknown"
    test_meta = [{"type": _type_of_tid(p), "file": os.path.basename(p)} for p in te_t] \
              + [{"type": "overtaking", "file": os.path.basename(p)} for p in te_ot]
    info = dict(n_train=len(train_paths), n_test=len(test_paths),
                train_breakdown=f"head-on/crossing {len(tr_t)} + overtaking {len(tr_ot)}",
                test_breakdown=f"head-on/crossing {len(te_t)} + overtaking {len(te_ot)}",
                test_meta=test_meta)   # 🆕 平行 test_paths
    return train_paths, test_paths, info


def _stamp_scenario_meta(seg_per, scenario_meta):
    """把 scenario_type + scenario_file 盖进逐 episode 诊断 record（L146 分层彻底堵洞·additive）。
    在 evaluate/evaluate_continuous 返回【后】调用 → 不改 agg（钱图列逐位不变）、只丰富 final_per 诊断行。
    scenario_meta[i] 对应 test_pool[i] 对应 record 的 scenario_idx=i（同序，manifest 模式才有；strided=None 跳过）。"""
    if not scenario_meta or not seg_per:
        return
    for e in seg_per:
        si = e.get("scenario_idx")
        if isinstance(si, int) and 0 <= si < len(scenario_meta):
            e["scenario_type"] = scenario_meta[si]["type"]
            e["scenario_file"] = scenario_meta[si]["file"]


# ---------------- L1c checkpoint 可重放（不重跑总保险·D42-Lschema CAT1）+ Layer-1 每段崩溃数据安全（L80-续4）----------------
def _atomic_save(save_fn, final_path):
    """原子存盘（L80-续4·D2 复审）：save_fn 写到 `final.<pid>.tmp` → os.replace 到 final（同目录原子）。
    ⚠️ tmp 必带【非空后缀】(`.tmp`)：否则 sb3 model.save 见 Path.suffix=='' 会静默追加 `.zip`、致 os.replace 找不到源文件崩。
    `final_path` 已带 `.zip`/`.pkl` 后缀 → tmp 后缀=`.tmp` 非空 → sb3 不追加，写到精确 tmp 路径。
    原子性：os.replace 成功前旧 final 始终完整可用；半写被杀只留孤儿 tmp（旧 final 仍在）→ 绝不"两个都没"。"""
    tmp = f"{final_path}.{os.getpid()}.tmp"
    save_fn(tmp)
    os.replace(tmp, final_path)


def save_checkpoint(model, venv, name, seed, ckpt_dir):
    """存 model(.zip) + VecNormalize 统计(_vecnorm.pkl) → {ckpt_dir}/{方}_s{seed}{tag}。
    【不重跑总保险 + Layer-1 每段崩溃数据安全】：run 死 / 要重算聚合 / 复算诊断时可重载重 eval、无需重训。
    **纯 additive·只读**——save/venv.save 实证只做 self.__dict__ 浅拷 + state_dict 读取、不调 set_random_seed/不 advance 任何 RNG
    （L80-续4 D3 实证）→ 训练逐位不受影响（每段调用同样只读·与末段一次同性质）。**原子写**（_atomic_save·D2 复审）→ 半写被杀不留损坏 ckpt。返回 base 路径（无后缀）。"""
    from stable_baselines3.common.vec_env import VecNormalize
    os.makedirs(ckpt_dir, exist_ok=True)
    base = os.path.join(ckpt_dir, f"{name.replace(' ', '_')}_s{seed}{_TAG}")
    _atomic_save(model.save, base + ".zip")                    # sb3 序列化 policy（网络权重）·原子
    if isinstance(venv, VecNormalize):
        _atomic_save(venv.save, base + "_vecnorm.pkl")         # obs_rms/ret_rms 快照（eval obs 归一化复现需）·原子
    return base


def write_progress(base, *, name, kind, weight, seed, seg_done, num_timesteps,
                   total_steps, n_seg, trend, config_sig, curves=None, seg_per=None):
    """Layer-1 续训进度 + 【增量诊断数据】（L80-续4·D2 复审 = commit barrier；增量诊断 = memory realtime-diagnostic + user 2026-06-24 要求）。
    【在 save_checkpoint 原子写完 .zip + .pkl 之后】调用，progress.json 作为【唯一提交点】最后原子写（内记 ckpt 指纹 mtime+size）。
    **增量诊断**：除 trend（到达率等里程碑）外，把 curves（critic_loss/熵/approx_kl/ret_rms_var 等 WHY 量·随训练累积）+ seg_per（本段逐 episode 诊断）也每段写入
    → 训练【中途/被杀】都能拿到结构化诊断数据分析根因、不必等整 run 跑完（四臂同款·离散连续都调）。Layer-2 续训只读 trend/num_timesteps/config_sig/指纹，curves/seg_per 是诊断附加（Layer-2 忽略）。
    progress 文件名带 (party,seed,tag) 隔离（base 已含 名_s种子_tag）→ 并行多任务不撞。"""
    zip_path = base + ".zip"
    fp = None
    if os.path.exists(zip_path):
        _st = os.stat(zip_path)
        fp = {"zip_mtime": _st.st_mtime, "zip_size": _st.st_size}
    rec = {"party": name, "kind": kind, "colregs_weight": weight, "seed": seed,
           "seg_done": int(seg_done), "num_timesteps": int(num_timesteps),
           "total_steps": int(total_steps), "n_seg": int(n_seg),
           "trend": trend, "config_sig": config_sig, "ckpt_fingerprint": fp,
           "curves": curves, "seg_per": seg_per,                 # 增量诊断（curves=至今全量 WHY 曲线·seg_per=本段逐局）→ 中途可拉取分析
           "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    write_atomic(base + ".progress.json", json.dumps(rec, ensure_ascii=False))


def save_segment_checkpoint(model, venv, name, kind, weight, seed, ckpt_dir, *,
                            seg_done, num_timesteps, total_steps, n_seg, trend, config_sig,
                            curves=None, seg_per=None):
    """Layer-1 每段存档（L80-续4 ⑥）：原子存 model+VecNorm（覆盖最新）→ 最后写 progress.json 当 commit 点（提交顺序定死·D2 Q2）。
    progress.json 含【增量诊断】curves+seg_per（中途可拉取·四臂同款）。调用方包 try/except（存盘失败不崩训练·同 :540/:697 容错纪律）。"""
    base = save_checkpoint(model, venv, name, seed, ckpt_dir)   # 先原子写 .zip + .pkl（commit 前两步）
    write_progress(base, name=name, kind=kind, weight=weight, seed=seed,
                   seg_done=seg_done, num_timesteps=num_timesteps, total_steps=total_steps,
                   n_seg=n_seg, trend=trend, config_sig=config_sig,
                   curves=curves, seg_per=seg_per)                                # 最后写 .progress.json = 提交点（含增量诊断）
    return base


def _read_continuous_algo(base):
    """从 checkpoint 旁的 `.progress.json`（config_sig.continuous_algo）读连续臂算法（'ppo'/'sac'）→ 决定 replay 用哪个 loader。
    progress.json 由 Layer-1 每段 save_segment_checkpoint 写（含 config_sig·`03` L80-续6）。缺 sidecar/缺字段/读失败 → None
    （调用方默认 'sac'·向后兼容原行为）。纯只读·容错（损坏 json 不崩 replay）。"""
    pj = base + ".progress.json"
    if os.path.exists(pj):
        try:
            cfg = json.loads(open(pj, encoding="utf-8").read()).get("config_sig") or {}
            a = cfg.get("continuous_algo")
            return a.lower() if isinstance(a, str) else None
        except Exception:
            return None
    return None


def replay_eval(base, kind, weight, test_pool, *, continuous_algo=None, return_per=False):
    """从 checkpoint（base.zip + base_vecnorm.pkl）重载 model+VecNorm → eval（不重训）→ agg。
    供【不重跑总保险】+ smoke 验"存→重放逐位复现 final"。eval 确定性(deterministic=True) → 同 model+同 VecNorm 快照
    +同 test_pool ⟹ agg 逐位复现训练末段 final。obs_transform 由 saved vecnorm 经 VecNormalize.load 重建
    （colregs_weight 只影响 reward、不影响 eval 指标，传同值仅为 env 构造平价）。
    return_per：默认 False=返回 agg（向后兼容·所有既有调用点不变）；True=返回 (agg, per)=(聚合, 逐 episode final_per)
                → 供 replay-dump 诊断（拿逐局 Step-0 进近标量等·evaluate 已算好只是原来丢弃）。additive·不改 agg 复现语义。
    🔧 连续臂算法分派（`03` L108·复审抓的钱图地雷）：连续臂主臂已从 SAC 换成 PPO（L69）·但本函数原硬编 load_sac_for_eval
       → 对 PPO checkpoint【崩】（zip 结构/policy 类不符）。修=按算法分派：① 显式 continuous_algo 实参优先（钱图调用方从 jsonl
       record['continuous_algo'] 透传）② 缺则读 checkpoint 旁 progress.json 的 config_sig.continuous_algo（sidecar 自描述）
       ③ 仍未知默认 'sac'（向后兼容原行为·守护⑤ SAC smoke 不变）。'ppo'→PPO.load（plain PPO·同 reeval_2000.py）·否则→load_sac_for_eval。"""
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from trb_env.train import make_obs_transform
    env_cls = env_cls_of(kind)
    tf = None
    _pkl = base + "_vecnorm.pkl"
    if os.path.exists(_pkl):                                    # 重建 obs_transform：saved vecnorm 载入最小 base venv（test_pool[0] 取 obs 空间）
        _sc0, _pp0 = test_pool[0]
        _bv_extra = dict(augment_rho=_AUGMENT_RHO) if kind == "continuous" else {}   # 🆕 腿1：vecnorm-load base venv 的 obs 空间须与训练同维（34/27）·否则载 34-obs_rms 进 27-space=形状崩/静默错（cone 不改 obs 维度不需此·augment 必须·L151 train/eval 同构造）
        _bv = DummyVecEnv([lambda: env_cls(_sc0, _pp0, colregs_weight=weight, **_bv_extra)])
        _vn = VecNormalize.load(_pkl, _bv)
        _vn.training = False
        tf = make_obs_transform(_vn)
    if kind == "continuous":
        from trb_env.usv_continuous_shield import ContinuousProjectionEnv
        from trb_env.evaluate import evaluate_continuous
        _algo = (continuous_algo or _read_continuous_algo(base) or "sac").lower()   # 显式优先→sidecar→默认 sac(向后兼容)
        if _algo == "ppo":                                         # 连续臂主臂（L69）·plain PPO·load_sac_for_eval 对它崩（02 残余④/钱图地雷）
            from stable_baselines3 import PPO
            model = PPO.load(base + ".zip", device="cpu")          # 同 reeval_2000.py 的 PPO.load 路径
        else:                                                      # SAC（脚注臂）·鲁棒载跳优化器（BRO wd>0 checkpoint 不崩·L67-续8/二审 BRO-3）
            from trb_env.usv_sac_train import load_sac_for_eval
            model = load_sac_for_eval(base + ".zip", device="cpu")
        agg, per = evaluate_continuous(lambda sc, pp: ContinuousProjectionEnv(sc, pp, shield=_CONTINUOUS_SHIELD, goal_cone_half=_GOAL_CONE_HALF_RAD, goal_v_floor=_GOAL_V_FLOOR, augment_rho=_AUGMENT_RHO, goal_ignore_orientation=_GOAL_IGNORE_ORIENT), model, test_pool, obs_transform=tf)   # 🆕 P0 盾开关 + ρ0 锥 + 腿1 态势增广（replay eval 须与训练同 shield/cone/augment·靠同 env 变量）
    else:
        from sb3_contrib import MaskablePPO
        from trb_env.evaluate import evaluate
        model = MaskablePPO.load(base + ".zip", device="cpu")
        agg, per = evaluate(lambda sc, pp: env_cls(sc, pp, colregs_weight=weight), model, test_pool, obs_transform=tf)
    return (agg, per) if return_per else agg


# ---------------- ep_rew_mean 原始 episode 回报追踪（Node L·替代 Monitor·不碰共享 make_vec_env）----------------
def _accumulate_ep_returns(cb):
    """curve callback 的 _on_step 每步调：用 `VecNormalize.get_original_reward()` 累积【原始(非归一化)】episode 回报，
    on done 压入 cb._ep_returns（窗口 mean=ep_rew_mean）。**等价 Monitor 的 rollout/ep_rew_mean 但不需包 Monitor**
    （Monitor 须改共享 make_vec_env、四方+scenarios 测试都用、PPO mask 经 Monitor.__getattr__ 转发 + .env 属性冲突风险）。
    纯只读（读 get_original_reward / locals['dones']）→ 不改训练（同 callback 既有只读性·L51-续 bit-identical）。"""
    _vn = cb.model.get_vec_normalize_env()
    _raw = _vn.get_original_reward() if _vn is not None else cb.locals.get("rewards")
    _dones = cb.locals.get("dones")
    if _raw is None or _dones is None:
        return
    acc = cb._ep_acc
    if len(acc) != len(_raw):                                  # 首步/ n_envs 变化 → 重置累积器
        acc = [0.0] * len(_raw); cb._ep_acc = acc
    for i in range(len(_raw)):
        acc[i] += float(_raw[i])
        if _dones[i]:                                          # episode 结束 → 压入完成回报、清该 env 累积
            cb._ep_returns.append(acc[i]); acc[i] = 0.0


def _ep_rew_mean(cb, window=100):
    """最近 window 个完成 episode 的原始回报均值（无完成 episode → None·同 Monitor 早期行为）。"""
    w = cb._ep_returns[-window:]
    return (sum(w) / len(w)) if w else None


# ---------------- 单 (party, seed) 训练 + 分段评估 ----------------
def train_eval_one(name, kind, weight, seed, train_paths, test_pool, *,
                   total_steps, n_seg, n_envs, subproc=True, ckpt_dir=None, scenario_meta=None):
    """训一个 (party, seed)（停船配方 + 分段评估）→ 返回末段五列指标 + 趋势 + 训练耗时/fps。

    复用 trb_env.train 的配方常量（同口径）；镜像已验证的 run_validation 结构。
    """
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import VecNormalize
    from trb_env.train import VECNORM_KWARGS, POLICY_NET_ARCH, make_obs_transform
    from trb_env.usv_scenarios import make_vec_env
    from trb_env.evaluate import evaluate

    env_cls = env_cls_of(kind)
    venv = make_vec_env(paths=train_paths, n_envs=n_envs, env_cls=env_cls,
                        env_kwargs=dict(colregs_weight=weight, gamma=_GAMMA,
                                        well_shaping_weight=_WELL_B, shaping_radius=_SHAPING_RADIUS,
                                        xtrack_weight=_WELL_X, xtrack_radius=_XTRACK_RADIUS),   # 对症 横向进带势（`03` L88·四方对称）
                        subproc=subproc, seed=seed)
    # clip_reward 默认吃 sb3 的 10.0（= 已验证配方 D22）；STEP4E_CLIP_REWARD 设则覆盖（消融）。
    # 审核#1：稀疏 +50 终端被 return-std 归一化后硬 clip ±10、疑为种子分裂诱因之一 → 放松 clip 做对照。
    # 仅归一化 setup 参数（同 VecNormalize 本身论文未明示）、不动 Krasowski reward 系数。
    _vn_kwargs, _clip_reward_eff = resolve_vecnorm_kwargs(
        VECNORM_KWARGS, os.environ.get("STEP4E_CLIP_REWARD"), os.environ.get("STEP4E_NORM_REWARD"))
    _norm_reward_eff = _vn_kwargs.get("norm_reward", True)
    venv = VecNormalize(venv, gamma=_GAMMA, **_vn_kwargs)
    # ⚠️ 强制 CPU：MLP 2×64 太小，放 GPU 每步 CPU↔GPU 拷贝开销 >> 计算 → 反而更慢（吃 CPU 不吃 GPU，D17）。
    # CUDA torch 服务器若不指定，sb3 device="auto" 会误用 GPU 拖慢。STEP4E_DEVICE 可覆盖（一般别动）。
    model = MaskablePPO("MlpPolicy", venv, policy_kwargs=dict(net_arch=POLICY_NET_ARCH),
                        seed=seed, ent_coef=ENT_START, gamma=_GAMMA, verbose=0,   # 起点 ENT_START；callback 逐步退到 ENT_END
                        # gamma 显式 0.99 同进 MaskablePPO 与 VecNormalize（L21②/审查 four-way F1：原赖 sb3 默认恰=0.99=脆弱、显式化）
                        device=os.environ.get("STEP4E_DEVICE", "cpu"))

    class _EntAnneal(BaseCallback):
        """ent_coef 线性退火 start→end（前 anneal_steps 步退完，之后恒 end）。
        model.ent_coef 是普通 float、loss 里直接乘 → rollout 间改即时生效；
        model.num_timesteps 跨段累计（仅第 0 段 reset_num_timesteps）→ 全局进度正确。
        """
        def __init__(self, s, e, anneal_steps):
            super().__init__()
            self._s, self._e, self._n = float(s), float(e), max(1, int(anneal_steps))

        def _on_step(self):
            self.model.ent_coef = anneal_ent_coef(self._s, self._e, self._n, self.model.num_timesteps)
            return True

    ent_cb = _EntAnneal(ENT_START, ENT_END, total_steps * ENT_FRAC)

    class _CurveLogger(BaseCallback):
        """L39 决定性诊断（opt-in STEP4E_LOG_CURVES=1）：每 rollout 记录 VecNormalize reward 滚动方差
        ret_rms.var（关键信号：塌种子是否早期被游荡轨迹推高、把稠密 r_goal 信号除没）+ PPO 内部曲线
        （explained_variance/approx_kl/entropy_loss/clip_fraction/value_loss），区分 setup-artifact vs 内禀。
        读 self.model.logger.name_to_value（上次 train() 持久值）。⚠️ 解读注意（复审 A MINOR）：
        ① logger 在每段 learn() 内每 dump 清空 → **每段首 rollout** 的 train/* 项为 None（非仅全程首个）；
        ② ret_rms_var 是本 rollout 末当前值、train/* 是上一 rollout 的 train() 结果（滞后 1 rollout）；
        ③ entropy_loss 对【有盾】run 受 mask 几何结构性压低(ρ5单动作→熵≡0)、不反映 ent_coef → 熵判据只用【无盾】RR run 当锚（复审 B MAJOR-1）。
        CartPole+MaskablePPO 单元验证 bit-exact 零扰动（2026-06-16c）。跨段累积（同一实例每段复用、records 不清）。"""
        def __init__(self):
            super().__init__()
            self.records = []
            self._ep_acc = []                              # ep_rew_mean：每 env 当前 episode 原始回报累积
            self._ep_returns = []                          # 已完成 episode 原始回报（窗口 mean）

        def _on_rollout_end(self):
            _vn = self.model.get_vec_normalize_env()
            _ret_var = (float(_vn.ret_rms.var) if _vn is not None
                        and getattr(_vn, "ret_rms", None) is not None else None)
            _nv = self.model.logger.name_to_value

            def _g(k):
                v = _nv.get(k)
                return float(v) if v is not None else None
            self.records.append({
                "step": int(self.model.num_timesteps),
                "ret_rms_var": _ret_var,                       # reward 滚动方差（核心信号、无滞后）
                "ep_rew_mean": _ep_rew_mean(self),             # 原始 episode 回报滚动均值（callback 自算·替 Monitor·Node L L54-续）
                "ep_rew_mean_logger": _g("rollout/ep_rew_mean"),  # sb3 logger 值（需 Monitor·当前 None，保留对照）
                "explained_variance": _g("train/explained_variance"),
                "approx_kl": _g("train/approx_kl"),
                "entropy_loss": _g("train/entropy_loss"),
                "clip_fraction": _g("train/clip_fraction"),
                "value_loss": _g("train/value_loss"),
                "policy_gradient_loss": _g("train/policy_gradient_loss"),   # advantage 尺度间接代理（复审 B MEDIUM-1）
            })

        def _on_step(self):
            _accumulate_ep_returns(self)                   # 每步累积原始 episode 回报（ep_rew_mean·只读不扰训练）
            return True

    curve_cb = _CurveLogger() if LOG_CURVES else None
    # LR 退火（`03` L88·默认 off=不安装=训练字节级不变）：装累积式 schedule + sync callback（替 SB3 锯齿 progress）
    _lr_anneal_cb = None
    _lr_anneal_start = None
    if _LR_ANNEAL_END is not None:
        from trb_env.usv_sac_train import LRAnnealSchedule, LRAnnealSyncCallback
        _lr_anneal_start = float(model.lr_schedule(1.0))                    # 当前恒定 lr（MaskablePPO=sb3默认3e-4）·稳健提取不硬编
        _lr_sched = LRAnnealSchedule(_lr_anneal_start, _LR_ANNEAL_END, _LR_ANNEAL_FRAC * total_steps)
        model.learning_rate = _lr_sched   # 同设 learning_rate→SB3 load 时 _setup_lr_schedule(get_schedule_fn) 保住退火(否则重建回恒定起点·Layer-2 续训静默重置·冒烟坐实)
        model.lr_schedule = _lr_sched     # 训练期直接用(非 lambda 包裹·返回 python float·无 GH#1900 numpy 问题)；load 后 SB3 按 learning_rate 重建为 lambda(sched)·值等价
        _lr_anneal_cb = LRAnnealSyncCallback(_lr_sched)
    _cbs = [ent_cb] + ([curve_cb] if curve_cb is not None else []) + ([_lr_anneal_cb] if _lr_anneal_cb is not None else [])
    learn_cb = _cbs[0] if len(_cbs) == 1 else _cbs                          # 单个时传 callback 本体（保 OFF 路径与旧版逐位一致·不套 CallbackList）

    def fac(sc, pp):
        return env_cls(sc, pp, colregs_weight=weight)

    seg = max(1, total_steps // n_seg)
    trend = []
    last = None
    t_train = 0.0
    t_eval = 0.0                                   # 评估耗时（L23：eval 单线程顺序跑 = 多核空转疑似源 → 量它定 NSEG/核）
    _config_sig = {"kind": kind, "colregs_weight": weight, "total_steps": total_steps, "n_seg": n_seg,   # Layer-1 记入 progress·Layer-2 续训用作匹配(白名单全等·任一不符从0)
                   "n_envs": n_envs, "algo": "MaskablePPO", "ent_start": ENT_START, "ent_end": ENT_END,
                   "ent_frac": ENT_FRAC, "clip_reward": _clip_reward_eff, "norm_reward": _norm_reward_eff,
                   "gamma": _GAMMA, "well_shaping_weight": _WELL_B, "shaping_radius": _SHAPING_RADIUS,   # 修法A 进门势=影响训练的配置→进续训匹配白名单（`03` L81）
                   "xtrack_weight": _WELL_X, "xtrack_radius": _XTRACK_RADIUS,   # 对症 横向进带势=影响训练→进续训匹配白名单（`03` L88）
                   "dataset": _DATASET_SIG,   # 🆕 数据集模式（均衡/strided·`03` L113-L115·影响场景池→自描述+防混）
                   "lr_anneal_end": _LR_ANNEAL_END, "lr_anneal_frac": _LR_ANNEAL_FRAC}   # LR 退火=影响训练的配置→进续训匹配白名单（`03` L88·off 时 end=None）
    for c in range(n_seg):
        _t = time.time()
        model.learn(total_timesteps=seg, reset_num_timesteps=(c == 0), callback=learn_cb)
        t_train += time.time() - _t                # 只计训练时间（不含评估），fps 才准
        venv.training = False                      # 冻结归一化统计做评估
        tf = make_obs_transform(venv)
        _te = time.time()
        _tj = TRAJ_EXAMPLE_IDXS if c == n_seg - 1 else None   # CAT5：仅末段对少数代表场景记示例轨迹(additive·钱图列逐位不变·落 final_per·D42-L2-续)
        agg, seg_per = evaluate(fac, model, test_pool, obs_transform=tf, traj_idxs=_tj)   # seg_per：末段=最终模型逐 episode 诊断(Node L 落盘·D42-Lschema①)
        _stamp_scenario_meta(seg_per, scenario_meta)   # 🆕 盖 scenario_type/file（L146·additive·不改 agg）
        t_eval += time.time() - _te
        venv.training = True
        step_done = (c + 1) * seg
        row = {"step": step_done, **{k: agg[k] for k in
               ("到达率%", "碰撞率%", "违规次数/局", "紧急步%", "Ep长s")}}
        trend.append(row)
        last = row
        print(f"    [{name} seed{seed}] {step_done:>8}步 | 到达 {agg['到达率%']:>5.1f}% "
              f"碰撞 {agg['碰撞率%']:>4.1f}% 违规 {agg['违规次数/局']:>5.2f} "
              f"紧急 {agg['紧急步%']:>4.1f}% Ep {agg['Ep长s']:>5.0f}s", flush=True)
        if ckpt_dir is not None:                               # Layer-1 每段存档（L80-续4·崩溃数据安全·additive 只读·被杀不丢模型/数据可 replay_eval）
            try:
                save_segment_checkpoint(model, venv, name, kind, weight, seed, ckpt_dir,
                                        seg_done=c, num_timesteps=model.num_timesteps,
                                        total_steps=total_steps, n_seg=n_seg, trend=trend, config_sig=_config_sig,
                                        curves=(curve_cb.records if curve_cb is not None else None), seg_per=seg_per)  # 增量诊断（离散臂）
            except Exception as _se:
                print(f"    ⚠️ 每段 checkpoint 存盘失败（不影响训练，继续）：{_se}", flush=True)
    try:                                                       # L1c：训后存 ckpt（additive·venv.close 前）；存盘失败【不丢训练结果】（同 write_run_metadata 容错纪律）
        _ckpt_base = save_checkpoint(model, venv, name, seed, ckpt_dir) if ckpt_dir is not None else None
    except Exception as _ce:
        _ckpt_base = None
        print(f"    ⚠️ checkpoint 存盘失败（不影响训练结果，继续）：{_ce}", flush=True)
    venv.close()
    fps = (seg * n_seg) / t_train if t_train > 0 else 0.0
    _wall = t_train + t_eval
    _eval_pct = round(100 * t_eval / _wall, 1) if _wall > 0 else 0.0
    if fps:
        print(f"    [{name} seed{seed}] ✅ 训练 {t_train:.0f}s = {fps:.0f} fps + 评估 {t_eval:.0f}s"
              f"（评估占 wall {_eval_pct:.0f}%；满量 3M 训练预计 {3_000_000 / fps / 3600:.1f}h/方/种子）\n"
              f"      ⓘ L23 定核：评估占比高 = 单线程 eval 空转多核 → 降 NSEG / 减评估场景数比加核更直接", flush=True)
    record = {"party": name, "kind": kind, "colregs_weight": weight, "seed": seed,
              "final": last, "trend": trend, "train_s": round(t_train, 1), "fps": round(fps),
              "eval_s": round(t_eval, 1), "eval_pct": _eval_pct,
              "ent_start": ENT_START, "ent_end": ENT_END, "ent_frac": ENT_FRAC,
              "clip_reward": _clip_reward_eff, "norm_reward": _norm_reward_eff,
              "gamma": _GAMMA, "well_shaping_weight": _WELL_B, "shaping_radius": _SHAPING_RADIUS,   # 修法A 进门势配置自描述（`03` L81）
              "xtrack_weight": _WELL_X, "xtrack_radius": _XTRACK_RADIUS,   # 对症 横向进带势配置自描述（`03` L88）
              "dataset": _DATASET_SIG, "manifest": (_MANIFEST or None),   # 🆕 数据集模式（均衡/strided·`03` L113-L115·写进 jsonl record 供钱图/续跑溯源）
              "lr_anneal_end": _LR_ANNEAL_END, "lr_anneal_frac": _LR_ANNEAL_FRAC, "lr_anneal_start": _lr_anneal_start,   # LR 退火自描述（`03` L88·off 时 end/start=None）
              "final_per": seg_per, "ckpt": _ckpt_base}   # Node L：逐 episode 诊断(CAT2/3/4) + checkpoint base(L1c·不重跑总保险)
    if curve_cb is not None:                       # L39 决定性诊断：内部曲线（reward 滚动方差 + PPO 内部量）
        record["curves"] = curve_cb.records
    return record


# ---------------- Continuous-safe 臂（SAC + 连续投影盾，Node B+C1；C2 接入四方）----------------
def train_eval_one_continuous(seed, train_paths, test_pool, *, total_steps, n_seg, n_envs, subproc=True, ckpt_dir=None, scenario_meta=None):
    """训一个 Continuous-safe (seed)：SAC + ContinuousProjectionEnv + 连续投影盾 → 末段六列 + 趋势 + fps。

    与离散 train_eval_one【同口径】（同场景池 / 同款 VecNormalize(obs+reward,clip10) / 同分段评估 / 同
    ViolationCounter / 同 gamma=0.99 / 同 make_obs_transform eval 归一化）；唯一【有意差异】(D2/D37-B/D38) =
      · 算法 = SAC（自带最大熵 ent_coef='auto'、**无 ent 退火**——SAC 不靠 masking 清零熵，L39）
      · env = ContinuousProjectionEnv（连续投影盾，Node A）
      · colregs_weight 默认 0.0(可经 STEP4E_COLREGS_WEIGHT 覆盖·A/B 复活 r_colregs)（丢 r_colregs、合规靠投影约束(档位A 经验性·非档位B provable 硬保证)，D37-B；make_continuous_safe_model 内 probe 断言）
      · 评估 = evaluate_continuous（按 source 三类归口、含「兜底步%」列；不调 action_masks，C1）
    subproc=False 仅供测试（DummyVecEnv 免 spawn）；生产同离散走 subproc=True。
    """
    from trb_env.train import make_obs_transform, VECNORM_KWARGS
    from trb_env.usv_continuous_shield import ContinuousProjectionEnv
    from trb_env.usv_sac_train import make_continuous_safe_model
    from trb_env.evaluate import evaluate_continuous
    from stable_baselines3.common.callbacks import BaseCallback

    # 同款 norm_reward（随 STEP4E_NORM_REWARD，与离散臂一致）；clip_reward 走默认 10（STEP4E_CLIP_REWARD 是离散种子诊断 knob、不施连续臂）
    _norm_reward_eff = resolve_vecnorm_kwargs(
        VECNORM_KWARGS, None, os.environ.get("STEP4E_NORM_REWARD"))[0].get("norm_reward", True)
    _learning_starts = int(os.environ.get("STEP4E_LEARNING_STARTS", "5000"))   # L63 Fix③：默认 5000(原 SAC 100)·A/B 可设 100=旧基线
    _gc = os.environ.get("STEP4E_GRAD_CLIP", "1.0")                            # L65 Fix：critic/actor 梯度裁剪范数·默认 1.0
    _max_grad_norm = None if _gc.lower() in ("off", "none") else float(_gc)    #   STEP4E_GRAD_CLIP=off → 退化 vanilla SAC(A/B 基线复现发散)
    _lr = float(os.environ.get("STEP4E_LR", "1e-4"))                           # L65 Fix：学习率·默认 1e-4(原 3e-4)·稳化自举
    _tqc = os.environ.get("STEP4E_TARGET_Q_CLIP", "off")                       # L67 Fix：target-Q 值裁剪天花板·默认 off(保 L65 行为)
    # ⚠️ 值按【VecNormalize 归一化后】尺度（合法 |Q|~5-15）·非 raw（~6000）：填 raw 大值=永不咬=静默 no-op（红队 B③）；A/B 荐 ~50-1000
    _target_q_clip = None if _tqc.lower() in ("off", "none", "") else float(_tqc)  #   STEP4E_TARGET_Q_CLIP=<正数> → cap bootstrap Q 高估发散(对症·A/B)
    _critic_ln = os.environ.get("STEP4E_CRITIC_LAYERNORM", "0") not in ("0", "", "off", "false", "no")  # L67-续2：critic LayerNorm(根因·BRO)·默认关
    _n_critics = int(os.environ.get("STEP4E_N_CRITICS", "2"))                  # L67-续2：REDQ-lite·默认 2=sb3
    _tau = float(os.environ.get("STEP4E_TAU", "0.005"))                        # L67-续2：目标平滑·默认 0.005=sb3
    _critic_wd = float(os.environ.get("STEP4E_CRITIC_WD", "0.0"))              # L67-续7：完整 BRO=LayerNorm + critic AdamW 权重衰减·默认 0(纯 LN)
    # L76：SAC gradient_steps 旋钮（速度/UTD 控制·sb3 原生·不碰 D38 熵守卫）。默认 'auto'→=n_envs=保 UTD=1
    #   （n_envs=1 时→1=现状完全不变；n_envs=16 时→16=UTD 仍 1·只靠并行采样提速·不改学习动态）。
    #   设【正整数】< n_envs → 降 UTD（每样本少训·换 fps·改学习动态·须当独立 A/B·非纯加速·`03` L76 复审 BLOCKER）。
    _gs_env = os.environ.get("STEP4E_SAC_GRADIENT_STEPS", "auto").lower()
    _gradient_steps = n_envs if _gs_env in ("auto", "", "n_envs") else int(_gs_env)
    if _gradient_steps < 1:
        raise SystemExit(f"🔒 STEP4E_SAC_GRADIENT_STEPS 须 ≥1（或 'auto'=n_envs），得 {_gs_env!r}")
    if _critic_ln and abs(_lr - 1e-4) < 1e-12:                                 # LN-6（深核·footgun）：LayerNorm 卖点=让【高 lr】既稳又学；只设 LN 忘了 LR=3e-4→跑 lr=1e-4(本就"稳但不学")=归因混淆
        print(f"⚠️ STEP4E_CRITIC_LAYERNORM=1 但 STEP4E_LR={_lr}（默认低 lr·L66 已证 lr=1e-4 不靠 LN 本就稳）→ 测不出 LayerNorm 卖点"
              "（让高 lr 既稳又学）。主推 A/B 应同设 STEP4E_LR=3e-4，并对照 vanilla+3e-4 / vanilla+1e-4 三方拆出 LN 边际贡献。", flush=True)
    # 惩罚退火（`03` L103）：退火 ON 时 factory 收【初值=schedule t=0 值=0】(=hold 段)·callback 再 ramp 到终点；OFF 时收常量 _ALIAS_W/_RATE_W。
    _alias_w_init = 0.0 if _ALIAS_ANNEAL_END is not None else _ALIAS_W
    _rate_w_init = 0.0 if _RATE_ANNEAL_END is not None else _RATE_W
    _arr_slack_init = _ARR_SLACK_START_RAD if _ARR_SLACK_START_RAD is not None else 0.0   # 🆕 B1（`03` L153）：maker 收起始 slack（退火起点·off 时 0.0=bit-identical·退火 callback 从此值 ramp 到 0；eval fac 另建收默认 0=真门）
    _algo = os.environ.get("STEP4E_CONTINUOUS_ALGO", "sac").lower()            # L67-续3/节点2：连续臂算法·默认 sac·"ppo"=并行 hedge(on-policy 无 off-policy 发散·A/B 诊断 TAG·不污染四方 metadata)
    if _algo == "ppo":
        from trb_env.usv_sac_train import make_continuous_safe_ppo_model
        # F3（深核 MEDIUM）：PPO ent_coef 须随【离散臂同款 ent 配置】走、别硬编——四方全 PPO 须同 ent 口径（唯一差异=连续vs离散动作）。
        if ENT_START != ENT_END:                                       # 离散臂退火(START≠END)·PPO 退火 callback 未接线 → fail-fast 防口径静默不对称
            raise SystemExit(f"🔒 连续 PPO 臂的 ent 退火(START={ENT_START}≠END={ENT_END})未接线 → 与离散臂 ent 口径不对称会污染'连续vs离散'对照。"
                             " 四方 PPO A/B 请用【常量 ent】(STEP4E_ENT_START==STEP4E_ENT_END·=生产 run_c3.sh 配方)，或先给 PPO 接退火 callback。")
        model, venv = make_continuous_safe_ppo_model(                  # SAC 专属 knob(learning_starts/grad_clip/lr/target_q/layernorm/n_critics/tau)不适用 PPO
            paths=train_paths, n_envs=n_envs, seed=seed, subproc=subproc,
            gamma=_GAMMA, norm_reward=_norm_reward_eff,
            well_shaping_weight=_WELL_B, shaping_radius=_SHAPING_RADIUS,   # 修法A 进门势 PBRS（`03` L81·四方对称·gamma 同源 _GAMMA）
            xtrack_weight=_WELL_X, xtrack_radius=_XTRACK_RADIUS,   # 对症 横向进带势 PBRS（`03` L88·四方对称）
            park_weight=_PARK_W, park_radius=_PARK_RADIUS, park_v_target=_PARK_VTARGET,   # 想法B 终端保速势 PBRS（`03` L109·连续臂专属·不接离散=不开挂）
            c_step=_STEP_COST,   # 修法C 每步生存成本（`03` L123·连续臂专属·非PBRS 真改最优·治游荡局部最优）
            c_dwell=_DWELL_W, w_dwell=_DWELL_WLAT, h_dwell=_DWELL_HDG, dwell_radius=_DWELL_R, b_dwell=_DWELL_B,   # r_dwell 入库赤字滞留成本（`03` L161/L162·连续臂专属·非PBRS·治 corr≈0 终端入库病）
            c_reach=_C_REACH, dock_radius=_DOCK_R, v_dock=_V_DOCK,   # 🆕 第二条腿修法（`03` L172·连续臂专属·默认关 bit-identical·治 corr≈0 脱钩 + 过路惩罚泄漏）
            alias_weight=_alias_w_init, rate_weight=_rate_w_init,      # 动作混叠(L97)+action-rate(L98)·退火 ON 时收初值 0(L103)·OFF 时常量
            rate_dock=_RATE_DOCK,   # 🆕 第二条腿 rank1（`03` L173）：泊位精修门控治抖·默认 None=off=bit-identical·静态不退火
            colregs_weight=_COLREGS_W_CONT,                            # r_colregs 权重·默认0.0=现状·A/B 复活 Meyer 式26（仅连续臂）
            shield=_CONTINUOUS_SHIELD,                                 # 🆕 P0：SE-RL 盾开关（默认 True=bit-identical·0=连续无盾臂）
            goal_cone_half=_GOAL_CONE_HALF_RAD, goal_v_floor=_GOAL_V_FLOOR,   # 🆕 ρ0 朝目标锥（PhaseC·弧度传盾·默认 None=关=bit-identical）
            augment_rho=_AUGMENT_RHO,   # 🆕 腿1(L150/L152)：态势感知增广透传 maker（默认 False=bit-identical）
            start_frac=_START_FRAC, start_v=_START_V,   # 🆕 逆向起点课程（`03` L181）：起点系数透传训练 maker（默认 (1.0,None)=真起点=bit-identical·评估 fac:753 恒不传=真起点诚实红线）
            goal_ignore_orientation=_GOAL_IGNORE_ORIENT,   # 🆕 L185：去朝向硬门透传训练 maker（默认 False=严格真门=bit-identical·eval fac 同传=主指标一致）
            arrival_heading_slack=_arr_slack_init,   # 🆕 B1(`03` L153)：到达门朝向课程起始 slack 透传 maker（默认 0.0=真门=bit-identical·退火 callback 覆盖）
            warmstart_ckpt=(_WARMSTART_CKPT or None),   # 🆕 L190：热启动源 ckpt 透传训练 maker（默认 None=不热启动=bit-identical·灌源 policy+源 vecnorm·探索侧治崩）
            ent_coef=ENT_START)                                        # =离散臂常量 ent（config-driven·非硬编 0.01·F3 口径平价）
    else:
        model, venv = make_continuous_safe_model(
            paths=train_paths, n_envs=n_envs, seed=seed, subproc=subproc,
            gamma=_GAMMA, norm_reward=_norm_reward_eff,                  # 同口径 VecNormalize（gamma 同进 SAC 与 VecNorm，Node B 断言）
            well_shaping_weight=_WELL_B, shaping_radius=_SHAPING_RADIUS,   # 修法A 进门势 PBRS（`03` L81·四方对称·gamma 同源 _GAMMA）
            xtrack_weight=_WELL_X, xtrack_radius=_XTRACK_RADIUS,   # 对症 横向进带势 PBRS（`03` L88·四方对称）
            park_weight=_PARK_W, park_radius=_PARK_RADIUS, park_v_target=_PARK_VTARGET,   # 想法B 终端保速势 PBRS（`03` L109·连续臂专属·不接离散=不开挂）
            c_step=_STEP_COST,   # 修法C 每步生存成本（`03` L123·连续臂专属·非PBRS 真改最优·治游荡局部最优）
            c_dwell=_DWELL_W, w_dwell=_DWELL_WLAT, h_dwell=_DWELL_HDG, dwell_radius=_DWELL_R, b_dwell=_DWELL_B,   # r_dwell 入库赤字滞留成本（`03` L161/L162·连续臂专属·非PBRS·治 corr≈0 终端入库病）
            c_reach=_C_REACH, dock_radius=_DOCK_R, v_dock=_V_DOCK,   # 🆕 第二条腿修法（`03` L172·连续臂专属·默认关 bit-identical·治 corr≈0 脱钩 + 过路惩罚泄漏）
            alias_weight=_alias_w_init, rate_weight=_rate_w_init,      # 动作混叠(L97)+action-rate(L98)·退火 ON 时收初值 0(L103)·OFF 时常量
            rate_dock=_RATE_DOCK,   # 🆕 第二条腿 rank1（`03` L173）：泊位精修门控治抖·默认 None=off=bit-identical·静态不退火
            learning_starts=_learning_starts,                          # L63 Fix③ knob（STEP4E_LEARNING_STARTS·供 F2 A/B 逐个验修法）
            max_grad_norm=_max_grad_norm, learning_rate=_lr,           # L65 Fix（StableSAC 梯度裁剪 + lr↓·STEP4E_GRAD_CLIP/STEP4E_LR）
            target_q_clip=_target_q_clip,                              # L67 Fix（target-Q 值裁剪·对症·STEP4E_TARGET_Q_CLIP·默认关）
            use_critic_layernorm=_critic_ln, n_critics=_n_critics, tau=_tau,  # L67-续2（LayerNorm critic 根因 + REDQ-lite + tau）
            critic_weight_decay=_critic_wd,                            # L67-续7（完整 BRO·critic AdamW 权重衰减·STEP4E_CRITIC_WD·默认 0）
            gradient_steps=_gradient_steps,                            # L76：UTD 控制（auto=n_envs 保 UTD=1·并行提速不改学习；<n_envs=降 UTD 换 fps）
            colregs_weight=_COLREGS_W_CONT,                            # r_colregs 权重·默认0.0=现状·A/B 复活 Meyer 式26（仅连续臂）
            shield=_CONTINUOUS_SHIELD,                                 # 🆕 P0：SE-RL 盾开关（默认 True=bit-identical·0=连续无盾臂）
            goal_cone_half=_GOAL_CONE_HALF_RAD, goal_v_floor=_GOAL_V_FLOOR,   # 🆕 ρ0 朝目标锥（PhaseC·弧度传盾·默认 None=关=bit-identical）
            augment_rho=_AUGMENT_RHO,   # 🆕 腿1(L150/L152)：态势感知增广透传 maker（默认 False=bit-identical）
            start_frac=_START_FRAC, start_v=_START_V,   # 🆕 逆向起点课程（`03` L181）：起点系数透传训练 maker（默认 (1.0,None)=真起点=bit-identical·评估 fac:753 恒不传=真起点诚实红线）
            goal_ignore_orientation=_GOAL_IGNORE_ORIENT,   # 🆕 L185：去朝向硬门透传训练 maker（默认 False=严格真门=bit-identical·eval fac 同传=主指标一致）
            arrival_heading_slack=_arr_slack_init,   # 🆕 B1(`03` L153)：到达门朝向课程起始 slack 透传 maker（默认 0.0=真门=bit-identical·退火 callback 覆盖）
            device=os.environ.get("STEP4E_DEVICE", "cpu"))             # SAC ent_coef 不传 → 'auto'（Node B 守护）

    # LR 退火（`03` L88·默认 off=不安装=训练字节级不变）：PPO/SAC 通用·装累积式 schedule + sync callback（替 SB3 分段锯齿 progress）
    _lr_anneal_cb = None
    _lr_anneal_start = None
    if _LR_ANNEAL_END is not None:
        from trb_env.usv_sac_train import LRAnnealSchedule, LRAnnealSyncCallback
        _lr_anneal_start = float(model.lr_schedule(1.0))                # 当前恒定 lr（PPO=sb3默认3e-4 / SAC=_lr）·稳健提取不硬编
        _lr_sched = LRAnnealSchedule(_lr_anneal_start, _LR_ANNEAL_END, _LR_ANNEAL_FRAC * total_steps)
        model.learning_rate = _lr_sched   # 同设 learning_rate→SB3 load 时 _setup_lr_schedule(get_schedule_fn) 保住退火(否则重建回恒定起点·Layer-2 续训静默重置·冒烟坐实)
        model.lr_schedule = _lr_sched     # 训练期直接用(非 lambda 包裹·返回 python float·无 GH#1900 numpy 问题)；load 后 SB3 按 learning_rate 重建为 lambda(sched)·值等价
        _lr_anneal_cb = LRAnnealSyncCallback(_lr_sched)

    # 惩罚退火（`03` L103·默认 off=不安装=训练字节级不变）：通用 schedule（rate_weight/alias_weight）+ sync callback 每步把 value() 推到各子 env。
    _penalty_anneal_cb = None
    if _RATE_ANNEAL_END is not None or _ALIAS_ANNEAL_END is not None:
        from trb_env.usv_sac_train import PenaltyAnnealSchedule, PenaltyAnnealSyncCallback
        _ramp_start = _PENALTY_RAMP_START_FRAC * total_steps
        _anneal_steps = _PENALTY_ANNEAL_FRAC * total_steps
        _pen_scheds = {}
        if _RATE_ANNEAL_END is not None:                               # start=0（hold 段无罚让起飞）→ end=终点权重
            _pen_scheds["rate_weight"] = PenaltyAnnealSchedule(0.0, _RATE_ANNEAL_END, _ramp_start, _anneal_steps)
        if _ALIAS_ANNEAL_END is not None:
            _pen_scheds["alias_weight"] = PenaltyAnnealSchedule(0.0, _ALIAS_ANNEAL_END, _ramp_start, _anneal_steps)
        _penalty_anneal_cb = PenaltyAnnealSyncCallback(_pen_scheds, venv)   # 持训练 venv（非 eval·fac 另建）

    # 🆕 B1 到达门朝向课程退火（`03` L153·默认 off=不安装=训练字节级不变）：slack start→0（量化 n_levels·仅连续臂）。
    #   🔴【诚实红线】eval env（下方 fac / replay_eval）【不挂本 callback、从不调 set_arrival_slack】→ 恒 slack=0=真门（报的到达率永远在真 ±9.74° 门上）。
    _arr_slack_anneal_cb = None
    if _ARR_SLACK_START_RAD is not None:
        from trb_env.usv_sac_train import ArrivalSlackAnnealSchedule, ArrivalSlackAnnealSyncCallback
        _arr_slack_sched = ArrivalSlackAnnealSchedule(_ARR_SLACK_START_RAD, _ARR_SLACK_ANNEAL_FRAC * total_steps)
        _arr_slack_anneal_cb = ArrivalSlackAnnealSyncCallback(_arr_slack_sched, venv)   # 持【训练 venv】（eval fac 另建·恒真门·不被本 callback 触碰）

    def fac(sc, pp):
        return ContinuousProjectionEnv(sc, pp, shield=_CONTINUOUS_SHIELD, goal_cone_half=_GOAL_CONE_HALF_RAD, goal_v_floor=_GOAL_V_FLOOR, augment_rho=_AUGMENT_RHO, goal_ignore_orientation=_GOAL_IGNORE_ORIENT)   # colregs_weight 默认 0.0（Node B/L44 footgun 修复）；🆕 P0 盾开关 + ρ0 锥 + 腿1 态势增广（eval 须与训练同 shield/cone/augment）；🔴 B1：**不传 arrival_heading_slack=恒真门**（诚实红线·评估不放水）

    class _SACCurveLogger(BaseCallback):
        """连续臂(SAC=论文 hero 臂)训练曲线（Node L CAT6·D42-Lschema④：原连续臂 model.learn 零 callback=hero 内部曲线未记）。
        按固定步距抽样记 SAC 内部量（actor/critic loss + ent_coef[α 自动熵温度] + ent_coef_loss + learning_rate + ep_rew_mean[需 Monitor·当前 None]）。
        SAC off-policy：_on_step 每环境步触发 → 按 record_every 步距抽样（避免每步记爆列表）；读 model.logger.name_to_value
        （上次 train() 持久值·dump 后清空 → 该键 None 则跳过）。纯读 logger、不改训练（同离散 _CurveLogger·L39 验证 bit-exact）；跨段累积。"""
        def __init__(self, record_every):
            super().__init__()
            self.records = []
            self._every = max(1, int(record_every))
            self._last = -1
            self._ep_acc = []                                # ep_rew_mean：每 env 当前 episode 原始回报累积
            self._ep_returns = []                            # 已完成 episode 原始回报（窗口 mean）

        def _on_step(self):
            _accumulate_ep_returns(self)                     # 每步累积原始 episode 回报（ep_rew_mean·只读不扰训练）
            t = self.model.num_timesteps
            if t - self._last >= self._every:
                self._last = t
                _nv = self.model.logger.name_to_value

                def _g(k):
                    v = _nv.get(k)
                    return float(v) if v is not None else None
                self.records.append({
                    "step": int(t),
                    "actor_loss": _g("train/actor_loss"),
                    "critic_loss": _g("train/critic_loss"),
                    "ent_coef": _g("train/ent_coef"),                # SAC 自动熵温度 α（'auto' → 学习得到）
                    "ent_coef_loss": _g("train/ent_coef_loss"),
                    "learning_rate": _g("train/learning_rate"),
                    # PPO(连续 hedge)内部量（SAC 臂这些键 None·PPO 臂 actor/critic_loss/ent_coef None）——双算法各记自己的、防诊断盲点（红队 LOW·L67-续3）
                    "pg_loss": _g("train/policy_gradient_loss"), "value_loss": _g("train/value_loss"),
                    "entropy_loss": _g("train/entropy_loss"), "approx_kl": _g("train/approx_kl"),
                    "clip_fraction": _g("train/clip_fraction"), "policy_std": _g("train/std"),
                    "ep_rew_mean": _ep_rew_mean(self),               # 原始 episode 回报滚动均值（callback 自算·替 Monitor·Node L L54-续）
                    "ep_rew_mean_logger": _g("rollout/ep_rew_mean"), # sb3 logger 值（需 Monitor·当前 None，保留对照）
                })
            return True

    sac_curve_cb = _SACCurveLogger(record_every=max(1, int(os.environ.get("STEP4E_RECORD_EVERY", str(total_steps // 200)))))   # ~200 抽样点（与 total_steps 自适应）；STEP4E_RECORD_EVERY 诊断覆盖（L82-续·隔离 record_every 是否扰训练）

    seg = max(1, total_steps // n_seg)
    trend = []
    last = None
    t_train = 0.0
    t_eval = 0.0
    # 🆕 L190 第2轮审 MEDIUM（源语义配置校验·别再甩"人肉预检责任"）：源 ckpt 旁的 .progress.json sidecar【现成记着源的 config_sig】→
    #   拿它与本 run 的 _config_sig 取【两边都有的语义键】交集比对·不一致 fail-fast（防"有盾源灌进无盾run/异奖励配方"=obs维相同→维度守卫抓不到的【静默方法论错】·agent 实跑坐实零报错通过）。
    #   sidecar 缺失/读失败/无 config_sig → 显眼 warning 不硬拒（老档缺键=既定可接受口径·同 c_reach）。
    if _WARMSTART_CKPT and _algo == "ppo":
        _SEMANTIC_KEYS = ("kind", "colregs_weight", "gamma", "norm_reward", "well_shaping_weight", "shaping_radius",
                          "xtrack_weight", "xtrack_radius", "park_weight", "park_radius", "park_v_target",
                          "c_step", "c_dwell", "w_dwell", "h_dwell", "dwell_radius", "b_dwell",
                          "c_reach", "dock_radius", "v_dock", "alias_weight", "rate_weight", "rate_dock",
                          "continuous_algo")   # 影响【策略语义/环境动力学】的键（不含 total_steps/n_seg/n_envs/seed/dataset 等 run 规模键=允许不同）
        _sp = _WARMSTART_CKPT + ".progress.json"
        try:
            with open(_sp, "r", encoding="utf-8") as _f:
                _src_sig = (json.load(_f) or {}).get("config_sig") or {}
        except (OSError, ValueError):
            _src_sig = None
        if not _src_sig:
            print(f"⚠️ 热启动源无 sidecar config_sig（{_sp} 缺/坏）→ 【无法校验源配置是否与本 run 同语义】·"
                  f"源须与本 run 同 shield/colregs/奖励配方=烧前预检责任（`03` L190 E）", flush=True)
        else:
            _cur_sig_probe = dict(kind="continuous", colregs_weight=_COLREGS_W_CONT, gamma=_GAMMA, norm_reward=_norm_reward_eff,
                                  well_shaping_weight=_WELL_B, shaping_radius=_SHAPING_RADIUS,
                                  xtrack_weight=_WELL_X, xtrack_radius=_XTRACK_RADIUS,
                                  park_weight=_PARK_W, park_radius=_PARK_RADIUS, park_v_target=_PARK_VTARGET,
                                  c_step=_STEP_COST, c_dwell=_DWELL_W, w_dwell=_DWELL_WLAT, h_dwell=_DWELL_HDG,
                                  dwell_radius=_DWELL_R, b_dwell=_DWELL_B, c_reach=_C_REACH, dock_radius=_DOCK_R,
                                  v_dock=_V_DOCK, alias_weight=_ALIAS_W, rate_weight=_RATE_W, rate_dock=_RATE_DOCK,
                                  continuous_algo=_algo)
            _mism = [(k, _src_sig.get(k), _cur_sig_probe.get(k)) for k in _SEMANTIC_KEYS
                     if k in _src_sig and k in _cur_sig_probe and _src_sig.get(k) != _cur_sig_probe.get(k)]
            if _mism:
                raise SystemExit("🔒 热启动源与本 run【语义配置不一致】(源策略在不同环境/奖励下训出→灌进来=静默方法论错·维度守卫抓不到)：\n"
                                 + "\n".join(f"    {k}: 源={s!r} ≠ 本run={c!r}" for k, s, c in _mism)
                                 + f"\n  源={_sp}\n  → 改用同配置的源 ckpt，或显式对齐本 run 配方（`03` L190 E 烧前预检）。")
            print(f"✅ 热启动源配置校验通过（sidecar 交集 {len([k for k in _SEMANTIC_KEYS if k in _src_sig])} 键全等）: {_WARMSTART_CKPT}", flush=True)
    _config_sig = {"kind": "continuous", "colregs_weight": _COLREGS_W_CONT, "total_steps": total_steps, "n_seg": n_seg,   # Layer-1 记入 progress·Layer-2 续训仅 PPO(SAC 入口 fail-fast 跳续·replay buffer 不随 save 存)
                   "n_envs": n_envs, "continuous_algo": _algo, "norm_reward": _norm_reward_eff, "clip_reward": None,
                   "gamma": _GAMMA, "well_shaping_weight": _WELL_B, "shaping_radius": _SHAPING_RADIUS,   # 修法A 进门势=影响训练的配置→进续训匹配白名单（`03` L81）
                   "xtrack_weight": _WELL_X, "xtrack_radius": _XTRACK_RADIUS,   # 对症 横向进带势=影响训练→进续训匹配白名单（`03` L88）
                   "park_weight": _PARK_W, "park_radius": _PARK_RADIUS, "park_v_target": _PARK_VTARGET,   # 想法B 终端保速势=影响训练→进续训匹配白名单（`03` L109·连续臂专属）
                   "c_step": _STEP_COST,   # 修法C 每步生存成本=影响训练→进续训匹配白名单（`03` L123·连续臂专属·非PBRS）
                   "c_dwell": _DWELL_W, "w_dwell": _DWELL_WLAT, "h_dwell": _DWELL_HDG, "dwell_radius": _DWELL_R, "b_dwell": _DWELL_B,   # r_dwell 入库赤字滞留成本=影响训练→进续训匹配白名单（`03` L161/L162·连续臂专属·非PBRS）
                   "c_reach": _C_REACH, "dock_radius": _DOCK_R, "v_dock": _V_DOCK,   # 🆕 第二条腿修法=影响训练→进续训匹配白名单（`03` L172·连续臂专属·⚠️老 ckpt 续训缺此键会从0重启·本项目走 fresh 5M+distinct TAG 不续老 ckpt=可接受）
                   "dataset": _DATASET_SIG,   # 🆕 数据集模式（均衡/strided·`03` L113-L115·影响场景池→自描述+防混）
                   "alias_weight": _ALIAS_W, "rate_weight": _RATE_W, "rate_dock": _RATE_DOCK,   # 动作混叠(L97)+action-rate 平滑(L98)+rank1 泊位门控治抖(L173)=影响训练→进续训匹配白名单·连续臂专属
                   "rate_anneal_end": _RATE_ANNEAL_END, "alias_anneal_end": _ALIAS_ANNEAL_END,   # 惩罚退火=影响训练→进续训匹配白名单（`03` L103·off 时 end=None）
                   "penalty_ramp_start_frac": _PENALTY_RAMP_START_FRAC, "penalty_anneal_frac": _PENALTY_ANNEAL_FRAC,
                   "ent_start": ENT_START, "ent_end": ENT_END,
                   "learning_rate": (None if _algo == "ppo" else _lr),
                   "max_grad_norm": (None if _algo == "ppo" else _max_grad_norm),
                   "learning_starts": (None if _algo == "ppo" else _learning_starts),
                   "target_q_clip": (None if _algo == "ppo" else _target_q_clip),
                   "critic_layernorm": (None if _algo == "ppo" else _critic_ln),
                   "n_critics": (None if _algo == "ppo" else _n_critics),
                   "tau": (None if _algo == "ppo" else _tau),
                   "critic_wd": (None if _algo == "ppo" else _critic_wd),
                   "gradient_steps": (None if _algo == "ppo" else _gradient_steps),
                   "lr_anneal_end": _LR_ANNEAL_END, "lr_anneal_frac": _LR_ANNEAL_FRAC}   # LR 退火=影响训练→进续训匹配白名单（`03` L88·off 时 end=None）
    _cont_cbs = [sac_curve_cb]                                           # OFF 全关→只本体(不套 CallbackList·保字节级不变)·ON→加 LR/惩罚退火同步
    if _lr_anneal_cb is not None:
        _cont_cbs.append(_lr_anneal_cb)
    if _penalty_anneal_cb is not None:
        _cont_cbs.append(_penalty_anneal_cb)
    if _arr_slack_anneal_cb is not None:                                 # 🆕 B1 到达门课程退火同步（off 时不 append=字节级不变·同 penalty/LR）
        _cont_cbs.append(_arr_slack_anneal_cb)
    _cont_cb = sac_curve_cb if len(_cont_cbs) == 1 else _cont_cbs
    for c in range(n_seg):
        _t = time.time()
        model.learn(total_timesteps=seg, reset_num_timesteps=(c == 0), callback=_cont_cb)   # SAC 无 ent 退火；curve callback 记内部曲线(CAT6)·_lr_anneal_cb 同步累积步给 lr_schedule
        t_train += time.time() - _t
        venv.training = False                                            # 冻结归一化统计做评估（同离散）
        tf = make_obs_transform(venv)
        _te = time.time()
        _tj = TRAJ_EXAMPLE_IDXS if c == n_seg - 1 else None   # CAT5：仅末段对少数代表场景记示例轨迹(additive·钱图列逐位不变·落 final_per·D42-L2-续)
        agg, seg_per = evaluate_continuous(fac, model, test_pool, obs_transform=tf, traj_idxs=_tj)   # 末段逐 episode 诊断(Node L 落盘·含投影修正量/source 六档/CPA)
        _stamp_scenario_meta(seg_per, scenario_meta)   # 🆕 盖 scenario_type/file（L146·additive·不改 agg）
        t_eval += time.time() - _te
        venv.training = True
        step_done = (c + 1) * seg
        row = {"step": step_done, **{k: agg[k] for k in
               ("到达率%", "碰撞率%", "违规次数/局", "紧急步%", "兜底步%", "Ep长s")}}
        trend.append(row)
        last = row
        print(f"    [Continuous-safe seed{seed}] {step_done:>8}步 | 到达 {agg['到达率%']:>5.1f}% "
              f"碰撞 {agg['碰撞率%']:>4.1f}% 违规 {agg['违规次数/局']:>5.2f} "
              f"紧急 {agg['紧急步%']:>4.1f}% 兜底 {agg['兜底步%']:>4.1f}% Ep {agg['Ep长s']:>5.0f}s", flush=True)
        if ckpt_dir is not None:                               # Layer-1 每段存档（L80-续4·崩溃数据安全·additive 只读；SAC/PPO 均存档·SAC 续训缓但崩溃后可 replay_eval）
            try:
                save_segment_checkpoint(model, venv, "Continuous-safe", "continuous", _COLREGS_W_CONT, seed, ckpt_dir,
                                        seg_done=c, num_timesteps=model.num_timesteps,
                                        total_steps=total_steps, n_seg=n_seg, trend=trend, config_sig=_config_sig,
                                        curves=sac_curve_cb.records, seg_per=seg_per)  # 增量诊断（连续臂·SAC/PPO 同款）
            except Exception as _se:
                print(f"    ⚠️ 每段 checkpoint 存盘失败（不影响训练，继续）：{_se}", flush=True)
    try:                                                       # L1c：训后存 ckpt（additive·venv.close 前）；存盘失败【不丢训练结果】（同 write_run_metadata 容错纪律）
        _ckpt_base = save_checkpoint(model, venv, "Continuous-safe", seed, ckpt_dir) if ckpt_dir is not None else None
    except Exception as _ce:
        _ckpt_base = None
        print(f"    ⚠️ checkpoint 存盘失败（不影响训练结果，继续）：{_ce}", flush=True)
    venv.close()
    fps = (seg * n_seg) / t_train if t_train > 0 else 0.0
    _wall = t_train + t_eval
    _eval_pct = round(100 * t_eval / _wall, 1) if _wall > 0 else 0.0
    if fps:
        print(f"    [Continuous-safe seed{seed}] ✅ 训练 {t_train:.0f}s = {fps:.0f} fps + 评估 {t_eval:.0f}s"
              f"（评估占 wall {_eval_pct:.0f}%；满量 3M 预计 {3_000_000 / fps / 3600:.1f}h/种子）", flush=True)
    return {"party": "Continuous-safe", "kind": "continuous", "colregs_weight": _COLREGS_W_CONT,
            "continuous_shield": _CONTINUOUS_SHIELD, "seed": seed,   # 🆕 P0 盾开关自描述（provenance·钱图/复现溯源·不进 config_sig 保续训 bit-identical）
            "goal_cone_half_deg": _GOAL_CONE_HALF_DEG, "goal_v_floor": _GOAL_V_FLOOR,   # 🆕 ρ0 朝目标锥自描述（PhaseC·L147·连续臂专属·off=None=现状·config_conflict 据此识锥混配·度=canonical 口径）
            "augment_rho": _AUGMENT_RHO,   # 🆕 腿1(L150/L152)：态势感知增广自描述（连续臂专属·off=False=现状·config_conflict 据此识 27/34维混写）
            "goal_ignore_orientation": _GOAL_IGNORE_ORIENT,   # 🆕 L185/L186：训练目标去朝向硬门自描述（连续臂专属·off=False=严格真门·True=位置-only·纯 provenance 不进 config_sig=续训 bit-identical·判读位置-only 烧卡结果据此识判据·posonly 用 distinct TAG 故不与金标混文件）
            "warmstart_ckpt": (_WARMSTART_CKPT or None),   # 🆕 L190：热启动源 ckpt 路径自描述（连续PPO臂专属·off=None=从零训练·纯 provenance 不进 config_sig=续训 bit-identical·⚠️训练流程如实可查=方法论诚实命门·别 claim"从零稳定"若热启动了）
            "warmstart_src_fp": _WARMSTART_FP,   # 🆕 第2轮审HIGH#1：源【内容指纹】(zip+vecnorm sha256前16)·路径是指针会变→指纹才是真身份→判读时校验10种子 fp 全等=「统一同源」从口头承诺变【可机器审计证据】
            "arr_slack_start_deg": _ARR_SLACK_START_DEG,   # 🆕 B1(L153)：到达门朝向课程 slack 起始度自描述（连续臂专属·off=None=现状·config_conflict 据此识课程混配·度=canonical·eval 恒真门不受影响）
            "final": last, "trend": trend, "train_s": round(t_train, 1), "fps": round(fps),
            "eval_s": round(t_eval, 1), "eval_pct": _eval_pct,
            "ent_start": None, "ent_end": None, "ent_frac": None,          # SAC 自带最大熵、无退火 → 不入 ent 配置集
            "clip_reward": None, "norm_reward": _norm_reward_eff,
            "gamma": _GAMMA, "well_shaping_weight": _WELL_B, "shaping_radius": _SHAPING_RADIUS,   # 修法A 进门势配置自描述（`03` L81）
            "xtrack_weight": _WELL_X, "xtrack_radius": _XTRACK_RADIUS,     # 对症 横向进带势配置自描述（`03` L88/L90·补连续臂对称漏记·config_conflict 守卫据此识 well_X 混配）
            "park_weight": _PARK_W, "park_radius": _PARK_RADIUS, "park_v_target": _PARK_VTARGET,   # 想法B 终端保速势配置自描述（`03` L109·连续臂专属）
            "c_step": _STEP_COST,   # 修法C 每步生存成本配置自描述（`03` L123·连续臂专属·非PBRS）
            "c_dwell": _DWELL_W, "w_dwell": _DWELL_WLAT, "h_dwell": _DWELL_HDG, "dwell_radius": _DWELL_R, "b_dwell": _DWELL_B,   # r_dwell 入库赤字滞留成本配置自描述（`03` L161/L162·连续臂专属·非PBRS·config_conflict 据此识 dwell 混配）
            "c_reach": _C_REACH, "dock_radius": _DOCK_R, "v_dock": _V_DOCK,   # 🆕 第二条腿修法配置自描述（`03` L172·连续臂专属·config_conflict 据此识 c_reach/泊位门 混配）
            "dataset": _DATASET_SIG, "manifest": (_MANIFEST or None),   # 🆕 数据集模式（均衡/strided·`03` L113-L115·写进 jsonl record 供钱图/续跑溯源）
            "alias_weight": _ALIAS_W, "rate_weight": _RATE_W, "rate_dock": _RATE_DOCK,   # 动作混叠(L97)+action-rate 平滑(L98)+rank1 泊位门控治抖(L173)配置自描述·连续臂专属·on/off 不混进同一 jsonl
            "rate_anneal_end": _RATE_ANNEAL_END, "alias_anneal_end": _ALIAS_ANNEAL_END,   # 惩罚退火自描述（`03` L103·off 时 end=None）
            "penalty_ramp_start_frac": _PENALTY_RAMP_START_FRAC, "penalty_anneal_frac": _PENALTY_ANNEAL_FRAC,
            "algo": ("PPO" if _algo == "ppo" else "StableSAC"),            # L67-续3：如实记连续臂算法（PPO opt-in 时勿假记 StableSAC·D6-F5 元数据诚实）
            "continuous_algo": _algo,                                      # 显式 sac/ppo（A/B 数据自描述·诊断可追溯）
            "learning_starts": (None if _algo == "ppo" else _learning_starts),   # SAC 专属·PPO 不适用
            "max_grad_norm": (None if _algo == "ppo" else _max_grad_norm),
            "learning_rate": (None if _algo == "ppo" else _lr),           # SAC 专属调参·PPO 用 sb3 默认(非调参·不记防 schedule 序列化)
            "lr_anneal_end": _LR_ANNEAL_END, "lr_anneal_frac": _LR_ANNEAL_FRAC, "lr_anneal_start": _lr_anneal_start,   # LR 退火自描述（`03` L88·off 时 end/start=None·PPO 退火 start=sb3默认3e-4）
            # L67-续2/续：SAC 稳化 knob 进 record，A/B 产物【自描述】哪个臂开了哪个修法（否则只能靠 TAG 文件名分辨=脆弱·preflight 抓出）
            "critic_layernorm": (None if _algo == "ppo" else _critic_ln),
            "n_critics": (None if _algo == "ppo" else _n_critics),
            "tau": (None if _algo == "ppo" else _tau),
            "target_q_clip": (None if _algo == "ppo" else _target_q_clip),
            "critic_weight_decay": (None if _algo == "ppo" else _critic_wd),   # L67-续7：完整 BRO knob 进 record（A/B 自描述）
            "gradient_steps": (None if _algo == "ppo" else _gradient_steps),   # L76：SAC UTD 控制（=n_envs 保 UTD=1；<n_envs 降 UTD）·自描述
            "n_envs_sac_run": (None if _algo == "ppo" else n_envs),            # L76：本 run 真实 n_envs（配 gradient_steps 反推 UTD=gs/n_envs）
            "curves": sac_curve_cb.records,             # Node L CAT6：SAC hero 臂内部训练曲线（actor/critic loss + α）
            "final_per": seg_per, "ckpt": _ckpt_base}   # Node L：逐 episode 诊断(CAT2/3/4) + checkpoint base(L1c·不重跑总保险)


# ---------------- Table III 聚合 ----------------
def build_table3(records):
    """records = [单(party,seed)结果...] → 每 party 在多种子上五列 均值±std 的 Table III 文本。

    先按 (party,seed) 去重保留最后一条（防并发双跑/续跑重训导致同 key 重复 → 抬高 seed 计数、拉偏均值）。
    """
    cols = [("到达率%", "到达率%", "{:.1f}"), ("碰撞率%", "碰撞率%", "{:.1f}"),
            ("违规次数/局", "违规/局", "{:.2f}"), ("紧急步%", "紧急步%", "{:.1f}"),
            ("Ep长s", "Ep长s", "{:.0f}")]
    dedup = {}                                              # (party,seed) → 最后一条记录
    for r in records:
        if "party" in r and "seed" in r and r.get("final"):
            dedup[(r["party"], int(r["seed"]))] = r
    by_party = {}
    for r in dedup.values():
        by_party.setdefault(r["party"], []).append(r)
    lines = []
    lines.append(f"{'Party':<14} | " + " | ".join(f"{disp:>12}" for _, disp, _ in cols)
                 + " | seeds")
    lines.append("-" * 96)
    for name, kind, weight in PARTIES:
        rs = by_party.get(name, [])
        if not rs:
            lines.append(f"{name:<14} | （未完成）")
            continue
        cells = []
        for key, _disp, fmt in cols:
            vals = [x["final"][key] for x in rs if x.get("final")]
            m, s = agg_mean_std(vals)
            # Base/RR 无盾 → 无紧急控制器 → 紧急步 = "–"（Table III 约定）
            if key == "紧急步%" and kind == "unshielded":
                cells.append(f"{'–':>12}")
            else:
                cells.append(f"{(fmt.format(m) + '±' + fmt.format(s)):>12}")
        lines.append(f"{name:<14} | " + " | ".join(cells) + f" | {len(rs)}")
    return "\n".join(lines)


def aggregate_and_write(seeds, mode, n_total, total_steps, *, elapsed=None):
    """读 partial → 按 seeds 过滤 → build_table3 → 原子写 table3{tag}.txt，返回全文。"""
    sel = [r for r in read_records(_PARTIAL) if int(r.get("seed", -1)) in seeds]
    table3 = build_table3(sel)
    n_tr = max((int(r.get("n_train", 0)) for r in sel), default=0) or "?"
    n_te = max((int(r.get("n_test", 0)) for r in sel), default=0) or "?"
    # 自记选取方式（03 L29 + L116 二审 MINOR 修：先看 dataset·别把均衡 manifest 误标 strided）。
    _datasets = {str(r.get("dataset", "strided")) for r in sel}
    _nonstrided = {d for d in _datasets if d != "strided"}
    _pools = {int(r.get("pool_size") or 0) for r in sel}
    _pos = {p for p in _pools if p > 0}
    if len(_datasets) > 1:                                  # strided 与 均衡 manifest 混进同表 → 告警（与 config_conflict 守卫呼应·防混）
        _sel = f"⚠混数据集{sorted(_datasets)}（strided+均衡 manifest 混表、归因无效，应按 STEP4E_TAG 隔离）"
    elif _nonstrided:                                       # 均衡数据集模式：显 manifest 名（非 strided·`03` L113-L116 覆盖三大 give-way 含追越）
        _sel = f"均衡数据集 manifest({sorted(_nonstrided)[0]})"
    elif len(_pos) > 1 or (_pos and 0 in _pools):
        _sel = f"⚠混选取{sorted(_pools)}（strided+聚集混表、归因无效，应按 STEP4E_TAG 隔离）"
    elif _pos:
        _sel = f"strided跨全库(pool={max(_pos)})"
    elif n_total >= POOL_SIZE:
        _sel = f"全库(N={n_total})"
    else:
        _sel = "前N聚集(⚠非全库子集)"
    et = f"# 本次 run 耗时 {elapsed:.0f}s（续跑/并行场景非累计）\n" if elapsed is not None else ""
    # 自记录 ent/clip 配置 → table3 自描述；多种配置混入同表 = 归因无效 → 显式告警（03 审核加固）
    cfgs = sorted({(r.get("ent_start"), r.get("ent_end"), r.get("ent_frac"),
                    r.get("clip_reward"), r.get("norm_reward", True))
                   for r in sel if r.get("ent_start") is not None})
    if not cfgs:
        cfg_line = "# ent/clip/normR 配置：（旧记录未标 → 未知）\n"
    else:
        cfg_strs = [f"ent {c[0]}→{c[1]}@前{c[2]} / clip±{c[3]} / normR={c[4]}" for c in cfgs]
        warn = ("  ⚠️【混配置！同表含多种 ent/clip/normR → 归因无效，应按 STEP4E_TAG 隔离各消融臂】"
                if len(cfgs) > 1 else "")
        cfg_line = "# ent/clip/normR 配置：" + " ; ".join(cfg_strs) + warn + "\n"
    # clip_reward 跨臂对称性独立校验（L49 #1）：上面 cfgs 以 `ent_start is not None` 过滤掉了连续臂（ent_start=None）→
    # 漏检 clip 不对称；此处含【全部臂】（连续臂 clip_reward=None→有效默认 10）。任一臂 clip 与众不同 = 四方
    # VecNormalize reward clip 口径分叉（无意=钱图污染）→ 在表头显式告警（与 ent/normR 混配告警对齐）。
    _clips = {float(r.get("clip_reward") or 10.0) for r in sel}
    if len(_clips) > 1:
        cfg_line += (f"#   ⚠️【clip_reward 跨臂不对称：{sorted(_clips)} → 四方 reward clip 口径分叉、钱图无效；"
                     f"CLIP_REWARD 仅施离散臂、正式四方 run 勿设（03 L49 #1）】\n")
    _wellbs = {float(r.get("well_shaping_weight") or 0.0) for r in sel}   # 修法A 进门势跨臂/跨记录一致性（`03` L82·含连续臂·与 ent/clip 混配告警对齐）
    if len(_wellbs) > 1:
        cfg_line += (f"#   ⚠️【well_shaping_weight 跨臂/跨记录不一致：{sorted(_wellbs)} → 修法A 进门势混配、归因无效；"
                     f"well_B A/B 须按 STEP4E_TAG 隔离各臂（03 L82）】\n")
    _lranns = {("off" if r.get("lr_anneal_end") is None else float(r.get("lr_anneal_end"))) for r in sel}   # LR 退火跨臂/跨记录一致性（`03` L88·与 well_B/ent/clip 混配告警对齐）
    if len(_lranns) > 1:
        cfg_line += (f"#   ⚠️【lr_anneal_end 跨臂/跨记录不一致：{sorted(map(str, _lranns))} → 学习率退火混配、归因无效；"
                     f"LR 退火 A/B 须按 STEP4E_TAG 隔离各臂（03 L88）】\n")
    _wellxs = {float(r.get("xtrack_weight") or 0.0) for r in sel}   # 对症 横向进带势跨臂/跨记录一致性（`03` L88·与 well_B 混配告警对齐）
    if len(_wellxs) > 1:
        cfg_line += (f"#   ⚠️【xtrack_weight 跨臂/跨记录不一致：{sorted(_wellxs)} → 对症 横向进带势混配、归因无效；"
                     f"well_X A/B 须按 STEP4E_TAG 隔离各臂（03 L88）】\n")
    _aliasw = {float(r.get("alias_weight") or 0.0) for r in sel}   # 动作混叠惩罚跨臂/跨记录一致性（Markgraf 式20·`03` L97·与 well_B/well_X 混配告警对齐）
    if len(_aliasw) > 1:
        cfg_line += (f"#   ⚠️【alias_weight 跨臂/跨记录不一致：{sorted(_aliasw)} → 动作混叠惩罚混配、归因无效；"
                     f"alias A/B 须按 STEP4E_TAG 隔离各臂（03 L97）】\n")
    _ratew = {float(r.get("rate_weight") or 0.0) for r in sel}   # action-rate 平滑惩罚跨臂/跨记录一致性（`03` L98·与 alias 混配告警对齐）
    if len(_ratew) > 1:
        cfg_line += (f"#   ⚠️【rate_weight 跨臂/跨记录不一致：{sorted(_ratew)} → action-rate 平滑惩罚混配、归因无效；"
                     f"rate A/B 须按 STEP4E_TAG 隔离各臂（03 L98）】\n")
    header = (f"# step4e Table III [{mode}] | N_TOTAL={n_total} 70/30 选取={_sel} "
              f"| 实际 训练 {n_tr} / 测试 {n_te} | steps={total_steps} "
              f"| seeds={sorted(seeds)} | 均值±样本std\n"
              f"# 四方同口径配方：同 env/场景/评估(ViolationCounter)/VecNormalize(obs+reward·clip 见下行配置)/gamma{_GAMMA}；"
              f"离散三方=MaskablePPO+ent_coef（配置见下行，03 D22/D25/L21）、Continuous-safe=SAC(ent_coef auto·无退火)+连续投影盾"
              f"(丢 r_colregs·紧急步%两臂 pre-step rho_acting 同口径·另有「兜底步%」列见 jsonl，C2/L46)\n" + cfg_line + et)
    out = header + "\n" + table3 + "\n"
    write_atomic(_TABLE3, out)
    return out


def clip_reward_guard(parties, clip_set, clip_ack):
    """L49 #1 口径护栏（仅训练模式调用）：STEP4E_CLIP_REWARD 是【离散种子诊断 knob】、只施离散臂（连续臂
    train_eval_one_continuous 硬走默认 clip=10）。含 continuous 臂 + 设 CLIP_REWARD → 四方 clip_reward 不对称必停；
    不含 continuous（如 launcher 离散子任务）+ 设 CLIP_REWARD + 无 ACK → 疑 leaked export 污染离散臂、停
    （旧 print 护栏仅在 continuous in parties 时触发、被 launcher 拆单方任务旁路=复审证伪）。返回 None=通过、否则 raise SystemExit。"""
    has_cont = any(p[1] == "continuous" for p in parties)
    if clip_set and (has_cont or not clip_ack):
        raise SystemExit(
            "❌【口径护栏 L49#1】STEP4E_CLIP_REWARD 是【离散种子诊断 knob】，只施离散臂、连续臂走默认 clip=10 → "
            + ("同 run 含 Continuous-safe 臂时四方 clip_reward 不对称、钱图静默污染。\n" if has_cont
               else "未显式确认即设此变量（疑 shell 残留 export 污染离散臂；launcher 拆单方任务时旧 print 护栏不触发）。\n")
            + "  · 正式四方 run：请【勿设】STEP4E_CLIP_REWARD（`unset STEP4E_CLIP_REWARD` 清残留 export）。\n"
            + "  · 确需离散种子诊断：STEP4E_PARTIES 单跑离散臂 + 独立 STEP4E_TAG（如 _clip50）+ 显式 "
              "STEP4E_CLIP_REWARD_ACK=1 确认（仅离散臂、勿含 Continuous-safe）。")


def main():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from trb_env.usv_scenarios import load_scenario_pool

    n_total = int(os.environ.get("STEP4E_NTOTAL", N_TOTAL_SMOKE if SMOKE else N_TOTAL_FULL))
    total_steps = int(os.environ.get("STEP4E_STEPS", STEPS_SMOKE if SMOKE else STEPS_FULL))
    n_seg = int(os.environ.get("STEP4E_NSEG", N_SEG_SMOKE if SMOKE else N_SEG_FULL))
    if "STEP4E_SEEDS" in os.environ:
        seeds = [int(x) for x in os.environ["STEP4E_SEEDS"].split(",") if x.strip() != ""]
    else:
        seeds = SEEDS_SMOKE if SMOKE else SEEDS_FULL
    parties = select_parties(os.environ.get("STEP4E_PARTIES"))
    mode = "SMOKE（冒烟）" if SMOKE else "FULL（非SMOKE；场景数看 N_TOTAL，非必=全2000）"

    # 输入校验（fail-fast）
    if n_seg < 1:
        raise SystemExit(f"STEP4E_NSEG 须 ≥1，得到 {n_seg}")
    if N_ENVS < 1:
        raise SystemExit(f"STEP4E_NENVS 须 ≥1，得到 {N_ENVS}")
    if not seeds:
        raise SystemExit("seeds 为空（检查 STEP4E_SEEDS，如 '0,1,2'）——无可训练种子")
    # 纯聚合模式（并行任务全跑完后出 table3，不训练）
    if os.environ.get("STEP4E_AGG", "0") != "0":
        out = aggregate_and_write(seeds, mode, n_total, total_steps)
        print("\n" + out, flush=True)
        print(f"✅ 仅聚合完成 → {_TABLE3}", flush=True)
        return

    # 预下载模式（并行前先单进程下全场景，避免多任务首下竞争）
    if os.environ.get("STEP4E_DOWNLOAD_ONLY", "0") != "0":
        if _MANIFEST:                                            # 均衡模式：下 head-on/crossing T-id + 校验 OT 文件齐（追越是本地上传）
            tp, te, _i = load_manifest_split(_MANIFEST, _BALANCED_DIR or None)
            print(f"✅ 预下载完成（均衡模式）：训练 {len(tp)} + 测试 {len(te)}（{_i['train_breakdown']} | {_i['test_breakdown']}）→ {_SDIR} + OT 池", flush=True)
        else:
            train_ids, test_ids = make_split(n_total, TEST_FRAC, SPLIT_SEED, pool_size=POOL_SIZE)
            tp, _ = _download(train_ids)
            te, _ = _download(test_ids)
            print(f"✅ 预下载完成：训练 {len(tp)}/{len(train_ids)} + 测试 {len(te)}/{len(test_ids)} → {_SDIR}", flush=True)
        return

    # 审查护栏（L49 #1，2026-06-17d：four-way F2 print→fail-fast）：STEP4E_CLIP_REWARD 是【离散种子诊断 knob】，
    # 只施离散臂（连续臂 train_eval_one_continuous 硬走默认 clip=10）→ 正式四方 run 误设 = 离散/连续 clip_reward
    # 不对称、钱图静默污染。旧护栏（仅 print + 仅 continuous in parties 触发）被复审证伪：launcher 把每方拆成单方
    # 子任务、离散臂子任务 parties 不含 continuous → 护栏对真正会被污染的离散臂【根本不触发】。故改 fail-fast：
    #   · 含 continuous 臂 + 设 CLIP_REWARD → 必停（四方不对称、任何情况都无效）；
    #   · 不含 continuous（如 launcher 离散子任务）+ 设 CLIP_REWARD + 无 ACK → 停（堵 leaked export 静默污染离散臂）。
    #   · 离散种子诊断须 opt-in：STEP4E_CLIP_REWARD_ACK=1 + 单跑离散臂 + 独立 TAG。（仅训练模式校验；AGG/预下载不训练故跳过）
    clip_reward_guard(parties, bool(os.environ.get("STEP4E_CLIP_REWARD")),
                      os.environ.get("STEP4E_CLIP_REWARD_ACK", "0") not in ("0", "", "false", "no"))

    _clip_disp = os.environ.get("STEP4E_CLIP_REWARD") or "10(默认)"
    _nr = os.environ.get("STEP4E_NORM_REWARD")
    _nr_disp = "False(消融·关奖励归一化)" if _nr and str(_nr).strip().lower() in ("0", "false", "no") else "True(默认)"
    _ent_disp = (f"{ENT_START}(恒定·关退火)" if ENT_START == ENT_END
                 else f"{ENT_START}→{ENT_END}@前{ENT_FRAC:.0%}步")
    print(f"=== step4e [{mode}] | parties={[p[0] for p in parties]} | N_TOTAL={n_total} test_frac={TEST_FRAC} "
          f"| steps={total_steps} seeds={seeds} n_seg={n_seg} n_envs={N_ENVS} "
          f"| ent_coef={_ent_disp} clip_reward={_clip_disp} norm_reward={_nr_disp} ===", flush=True)

    # 跨模式告警（兜底：重启/分片时模式设错→写错文件→从头重训）
    _other = os.path.join(_RESULT_DIR, "step4e_partial.jsonl" if SMOKE else "step4e_partial_smoke.jsonl")
    if os.path.exists(_other) and read_records(_other):
        print(f"⚠️ 检测到【{'FULL' if SMOKE else 'SMOKE'}】模式进度文件 {os.path.basename(_other)}（非当前模式）。"
              f"当前 = {mode}，将写 {os.path.basename(_PARTIAL)}。\n"
              f"   若你本想【续跑】那个 run：FULL 请改源码 SMOKE=False（推荐）或加 STEP4E_SMOKE=0 前缀。", flush=True)

    if _MANIFEST:                                                # 🆕 均衡数据集模式（`03` L113-L115·覆盖三大 give-way 含追越）
        train_paths, test_paths, _man_info = load_manifest_split(_MANIFEST, _BALANCED_DIR or None)
        train_ids = [os.path.basename(p) for p in train_paths]   # 均衡模式：用文件名作场景 id（run_metadata 复现性·替 strided T-id）
        test_ids = [os.path.basename(p) for p in test_paths]
        print(f"均衡数据集模式（manifest）：训练 {_man_info['n_train']}（{_man_info['train_breakdown']}）+ "
              f"测试 {_man_info['n_test']}（{_man_info['test_breakdown']}）← {_MANIFEST}", flush=True)
        n_train_req, n_test_req = _man_info['n_train'], _man_info['n_test']
    else:
        train_ids, test_ids = make_split(n_total, TEST_FRAC, SPLIT_SEED, pool_size=POOL_SIZE)
        print(f"分散抽样：训练 {len(train_ids)} + held-out 测试 {len(test_ids)}（互不重叠、非尾块）→ 缓存 {_SDIR} …", flush=True)
        train_paths, f1 = _download(train_ids)
        test_paths, f2 = _download(test_ids)
        # 测试集缩水 = Table III 泛化数字静默失真 → 硬停（缺额 >5%）；训练集缩水 = 降级但有效 → 仅告警
        if len(test_paths) < len(test_ids) * 0.95:
            raise SystemExit(
                f"⚠️ 测试集下载缺额过大：可用 {len(test_paths)}/{len(test_ids)}（缺 {len(f2)}）。\n"
                f"   测试集缩水会让 Table III 到达/碰撞/违规数字失真 → 中止。\n"
                f"   请检查网络/gitlab 后重跑（已下的会缓存复用，只补缺的；或先 STEP4E_DOWNLOAD_ONLY=1 预下）。")
        if f1:
            print(f"⚠️ 训练集 {len(f1)} 个下载失败 → 训练用 {len(train_paths)}/{len(train_ids)} 个（降级但有效，继续）", flush=True)
        n_train_req, n_test_req = len(train_ids), len(test_ids)
    test_pool = load_scenario_pool(test_paths)
    _scenario_meta = _man_info.get("test_meta") if _MANIFEST else None   # 🆕 平行 test_pool（manifest 模式才有·L146 分层堵洞）
    n_train_actual, n_test_actual = len(train_paths), len(test_pool)
    print(f"实际可用：训练 {n_train_actual}/{n_train_req} / 测试 {n_test_actual}/{n_test_req}", flush=True)

    os.makedirs(_RESULT_DIR, exist_ok=True)
    # 有效 pool：不 striding（全量 n≥库 / POOL 未生效）时记 None → 与改动前旧记录一致、不误判"选取冲突"（Agent A MINOR-1）
    _pool_eff = POOL_SIZE if POOL_SIZE > n_total else None
    conflict = config_conflict(read_records(_PARTIAL), total_steps, n_total, _pool_eff, n_seg,
                               _WELL_B, _SHAPING_RADIUS, _GAMMA, _LR_ANNEAL_END, _LR_ANNEAL_FRAC,
                               _WELL_X, _XTRACK_RADIUS, _ALIAS_W, _RATE_W,
                               _RATE_ANNEAL_END, _ALIAS_ANNEAL_END, _PENALTY_RAMP_START_FRAC, _PENALTY_ANNEAL_FRAC,
                               dataset=_DATASET_SIG,
                               park_weight=_PARK_W, park_radius=_PARK_RADIUS, park_v_target=_PARK_VTARGET,   # 惩罚退火进守卫（`03` L103）+ dataset 进守卫（`03` L116 二审 MEDIUM）+ park 进守卫（`03` L111/L112 二审 MINOR·Φ_park on/off 不混·连续臂专属）
                               c_step=_STEP_COST,
                               c_dwell=_DWELL_W, w_dwell=_DWELL_WLAT, h_dwell=_DWELL_HDG, dwell_radius=_DWELL_R, b_dwell=_DWELL_B,   # r_dwell 进守卫（`03` L161/L162·防同TAG混 dwell 配置静默跳过·连续臂专属）
                               c_reach=_C_REACH, dock_radius=_DOCK_R, v_dock=_V_DOCK, rate_dock=_RATE_DOCK,   # 🆕 第二条腿修法进守卫（`03` L172/L173·防同TAG混 c_reach/泊位门/rank1 配置静默跳过·连续臂专属）
                               continuous_shield=_CONTINUOUS_SHIELD,   # 修法C 进守卫（L123）+ 🆕 P0 SE-RL 盾 on/off 进守卫（L146·防同TAG翻shield静默跳过）
                               goal_cone_half_deg=_GOAL_CONE_HALF_DEG, goal_v_floor=_GOAL_V_FLOOR,   # 🆕 ρ0 朝目标锥进守卫（PhaseC·L147·防同TAG混锥配置静默跳过）
                               augment_rho=_AUGMENT_RHO,   # 🆕 腿1(L150/L152)：态势感知增广进守卫（防同TAG混 27/34维静默跳过）
                               arr_slack_start_deg=_ARR_SLACK_START_DEG,   # 🆕 B1(L153)：到达门朝向课程进守卫（防同TAG混课程配置静默跳过·度口径）
                               warmstart_ckpt=_WARMSTART_ID)   # 🆕 L190(自审补漏+第2轮审HIGH#1)：热启动进守卫·比较用【路径+内容指纹】非裸路径（防①热启动vs从零同TAG混写 ②同路径换源静默混写=破"全10种子统一同源"红线）
    if conflict:                                            # 配置守卫：拒绝把不同配置结果混进同一 Table III
        raise SystemExit(
            # 🆕 第2轮审 NIT 修：字段名单补全 37 名（原只列 32=漏 c_reach/dock_radius/v_dock/rate_dock/warmstart_ckpt → 打印 37 值对不上号）
            f"⚠️ {_PARTIAL} 含【不同配置】的结果 {sorted(conflict, key=str)}（签名 37 元=steps,n_total,pool_size,n_seg,well_B,shaping,gamma,lr_anneal_end,lr_anneal_frac,xtrack_weight,xtrack_radius,alias_weight,rate_weight,rate_anneal_end,alias_anneal_end,penalty_ramp_start_frac,penalty_anneal_frac,dataset,park_weight,park_radius,park_v_target,c_step,c_dwell,w_dwell,h_dwell,dwell_radius,b_dwell,c_reach,dock_radius,v_dock,rate_dock,continuous_shield,goal_cone_half_deg,goal_v_floor,augment_rho,arr_slack_start_deg,warmstart_ckpt[路径#指纹]），"
            f"与当前 (steps={total_steps}, n_total={n_total}, pool_size={_pool_eff}, n_seg={n_seg}) 不符。\n"
            f"   → 请删除该文件后重跑（避免 Table III 混配置/学习曲线 step 网格错位），或用 STEP4E_TAG 换输出文件。")

    # Node L L1a：run 起跑落盘静态元数据（动力学/COLREGs/reward/训练/动作/论文叙事全参 + 本次 run_config）——
    # "第一次跑全记齐不重复跑"(D42-Lschema⑦)。logging 失败【绝不中断训练】(try/except 吞)。并行/续跑多进程重写同名=幂等(静态部分一致)。
    try:
        from trb_env.run_metadata import write_run_metadata
        _meta_path = os.path.join(_RESULT_DIR, f"run_metadata{_TAG}.json")
        # seeds 权威写盘（L53 修）：launcher 把四方拆【单方/单种子】子进程、各以本任务单种子分片写同名 run_metadata{tag}.json
        # → 旧 "seeds": seeds 只记 last-writer 单种子。launcher 经 STEP4E_ALL_SEEDS 传【全种子集】→ 每个子进程都记全 seeds
        # （直接调用 run_step4e 不经 launcher 时 STEP4E_ALL_SEEDS 未设 → 回退本地 seeds=本就全量、正确）。
        _meta_seeds = ([int(x) for x in os.environ.get("STEP4E_ALL_SEEDS", "").split(",") if x.strip() != ""]
                       or seeds)
        # L59：run 实际 ent/clip/norm 配方写进 run_config（= 本 run 真相源·env 驱动·常量或退火）；
        # 否则元数据只有 train.py 静态默认(ENT_COEF=0.01)、与实际退火/消融不符=论文誊抄风险（`03` L59 复审 MINOR）。
        from trb_env.train import VECNORM_KWARGS as _VNK
        _vn_kw_meta, _clip_eff_meta = resolve_vecnorm_kwargs(
            _VNK, os.environ.get("STEP4E_CLIP_REWARD"), os.environ.get("STEP4E_NORM_REWARD"))
        write_run_metadata(_meta_path, run_config={
            "tag": _TAG, "parties": [p[0] for p in PARTIES], "seeds": _meta_seeds,
            "n_total": n_total, "total_steps": total_steps, "n_seg": n_seg,
            "pool_size": _pool_eff, "split_seed": SPLIT_SEED, "test_frac": TEST_FRAC,
            "n_envs": N_ENVS, "n_envs_sac": N_ENVS_SAC,
            "continuous_algo": _CONTINUOUS_ALGO,                         # 红队 MEDIUM：连续臂算法(sac/ppo)进 metadata（否则 sidecar 无从分辨·与 per-record algo 对账）
            "n_train": n_train_actual, "n_test": n_test_actual,
            "train_ids": list(train_ids), "test_ids": list(test_ids),   # L1a补：场景 ID 选择（复现性·make_split 输出·非下载后子集）
            # 离散臂实际 ent 配方（ENT_START==ENT_END=常量·否则前 ENT_FRAC 退火）；连续臂 SAC=auto 无退火。clip/norm=有效 VecNormalize 配方。
            "ent_start": ENT_START, "ent_end": ENT_END, "ent_frac": ENT_FRAC,
            "ent_schedule_discrete": ("constant" if ENT_START == ENT_END else "anneal"),
            "well_shaping_weight": _WELL_B, "shaping_radius": _SHAPING_RADIUS,   # 修法A 进门势（补 run_config 自描述·原仅 per-record 记·`03` L88）
            "xtrack_weight": _WELL_X, "xtrack_radius": _XTRACK_RADIUS,   # 对症 横向进带势（run_config 自描述·`03` L88）
            "park_weight": _PARK_W, "park_radius": _PARK_RADIUS, "park_v_target": _PARK_VTARGET,   # 想法B 终端保速势（run_config 自描述·`03` L109·连续臂专属）
            "c_step": _STEP_COST,   # 修法C 每步生存成本（run_config 自描述·`03` L123·连续臂专属·非PBRS）
            "c_dwell": _DWELL_W, "w_dwell": _DWELL_WLAT, "h_dwell": _DWELL_HDG, "dwell_radius": _DWELL_R, "b_dwell": _DWELL_B,   # r_dwell 入库赤字滞留成本（run_config 自描述·`03` L161/L162·连续臂专属·非PBRS·provenance 完整）
            "c_reach": _C_REACH, "dock_radius": _DOCK_R, "v_dock": _V_DOCK, "rate_dock": _RATE_DOCK,   # 🆕 第二条腿修法（run_config 自描述·`03` L172/L173·连续臂专属·provenance 完整）
            "warmstart_ckpt": (_WARMSTART_CKPT or None), "warmstart_src_fp": _WARMSTART_FP,   # 🆕 L190 热启动源 ckpt 路径+【内容指纹】（run_config 自描述·连续PPO臂专属·off=None=从零·provenance 命门=训练流程如实可查·指纹=真身份防同路径换源·第2轮审HIGH#1）
            "continuous_shield": _CONTINUOUS_SHIELD,   # 🆕 P0 SE-RL 盾 on/off（run_config 自描述·L146·连续臂专属·provenance 完整）
            "goal_cone_half_deg": _GOAL_CONE_HALF_DEG, "goal_v_floor": _GOAL_V_FLOOR,   # 🆕 ρ0 朝目标锥 Φ(度)/v_floor（run_config 自描述·PhaseC·L147·连续臂专属）
            "augment_rho": _AUGMENT_RHO,   # 🆕 腿1(L150/L152)：态势感知增广（run_config 自描述·连续臂专属）
            "arr_slack_start_deg": _ARR_SLACK_START_DEG, "arr_slack_anneal_frac": _ARR_SLACK_ANNEAL_FRAC,   # 🆕 B1(L153)：到达门朝向课程 slack 起始度+退火比例（run_config 自描述·连续臂专属·off=None）
            "start_frac": _START_FRAC, "start_v": _START_V,   # 🆕 逆向起点课程（`03` L181）：起点系数+课程重生速度（run_config 自描述·连续臂专属·1.0=真起点·评估恒真起点不受影响）
            "goal_ignore_orientation": _GOAL_IGNORE_ORIENT,   # 🆕 L185：训练目标去朝向硬门（run_config 自描述·连续臂专属·False=严格真门位置+朝向·True=位置-only到达区域·训练/评测同配置）
            "dataset": _DATASET_SIG, "manifest": (_MANIFEST or None),   # 🆕 数据集模式（均衡 manifest 名 / strided·`03` L113-L115·覆盖三大 give-way）
            "lr_anneal_end": _LR_ANNEAL_END, "lr_anneal_frac": _LR_ANNEAL_FRAC,  # 学习率退火（`03` L88·off 时 end=None）
            "lr_schedule": ("constant" if _LR_ANNEAL_END is None else "anneal"),
            "clip_reward": _clip_eff_meta, "norm_reward": _vn_kw_meta.get("norm_reward", True),
            "log_curves": LOG_CURVES,                                    # L59：离散臂训练曲线是否记录（事后可发现 curves 缺失）
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        print(f"📋 静态元数据 → {os.path.basename(_meta_path)}（论文参数全记齐）", flush=True)
    except Exception as _e:                                  # logging 不得影响训练
        print(f"⚠️ run_metadata 落盘失败（不影响训练，继续）：{_e}", flush=True)

    done = done_keys(_PARTIAL)
    if done:
        print(f"断点续跑：已完成 {len(done)} 个 (party,seed)，跳过。", flush=True)

    t0 = time.time()
    for name, kind, weight in parties:                     # 仅本任务分片选中的方
        for seed in seeds:
            if (name, seed) in done:
                continue
            print(f"\n>>> {name} (env={kind}, colregs_weight={weight}) seed={seed} 训练中…", flush=True)
            if kind == "continuous":                       # Continuous-safe = SAC 或 PPO（opt-in·L67-续3）+ 连续投影盾（C2）
                _cont_nenvs = continuous_n_envs(_CONTINUOUS_ALGO)              # 红队 MAJOR：PPO→N_ENVS_PPO(默认8·并行)·SAC→N_ENVS_SAC(默认1)
                rec = train_eval_one_continuous(seed, train_paths, test_pool,
                                                total_steps=total_steps, n_seg=n_seg, n_envs=_cont_nenvs,
                                                ckpt_dir=_CKPT_DIR, scenario_meta=_scenario_meta)  # L1c：存 checkpoint；🆕 scenario_meta 盖 type/file
            else:                                          # Base/RR/Discrete-safe = MaskablePPO（离散）
                rec = train_eval_one(name, kind, weight, seed, train_paths, test_pool,
                                     total_steps=total_steps, n_seg=n_seg, n_envs=N_ENVS,
                                     ckpt_dir=_CKPT_DIR, scenario_meta=_scenario_meta)  # L1c：存 checkpoint；🆕 scenario_meta 盖 type/file
            rec.update(steps=total_steps, n_total=n_total, pool_size=_pool_eff,  # 配置签名（有效 pool：不striding→None）
                       n_seg=n_seg,                                              # `03` L58 #2：分段数进签名（防中途改 NSEG 续跑→step 网格错位污染学习曲线）
                       n_train=n_train_actual, n_test=n_test_actual)
            append_record(_PARTIAL, rec)                    # 并行安全（flock）

    out = aggregate_and_write(seeds, mode, n_total, total_steps, elapsed=time.time() - t0)
    print("\n" + out, flush=True)
    print(f"✅ 完成。Table III → {_TABLE3}（partial → {_PARTIAL}）", flush=True)
    print("⭐ 把 结果/table3.txt 发我核对是否过 Phase 1 通过门（对齐 Krasowski）。", flush=True)


if __name__ == "__main__":
    main()
