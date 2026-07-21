#!/usr/bin/env python
"""Step-0 replay-dump 驱动（eval-only·不重训）：载入已存 checkpoint → 重跑评测（deterministic·同 test_pool）
→ 把逐 episode final_per（含 Step-0 进近 4 标量 min_goal_dist_m/heading_err_at_min_deg/in_box_steps/in_box_aligned_steps）
dump 成 jsonl，以看清那些【训练时只存了 3 条示例轨迹(idx0/1/2)】的失败局机制（游荡 vs 门口捕获 vs 横向进不了框）。

⚠️ 忠实性红线：evaluate 的 eval-env 配置(shield/cone/v_floor/augment/colregs)由 run_step4e 模块级全局决定
   → 本驱动【默认不设】这些 env=用 run_step4e 默认(=dwelloff/probeHOCRB s2 基线配置:shield True/cone None/v_floor 2.0/
   augment False/colregs 0)。若被复现的 run 用了非默认，须在跑本驱动前显式 export 同样 STEP4E_* env，否则复现不出。
   **复现闸门(REPLAY_EXPECT_JSONL)**：dump 前用原 jsonl 逐位校验(agg 6 列 + 逐局 reached/end_state)——
   不逐位相同=配置/manifest/algo 与原 run 不一致=【中止·绝不 dump 垃圾】。这是本驱动的安全网，务必给 EXPECT。

env：
  REPLAY_CKPT          checkpoint base 路径(无后缀·须有 .zip + _vecnorm.pkl [+ .progress.json])
  REPLAY_MANIFEST      manifest 路径(建 test_pool·须与原 run 同·如 /root/trb/balanced_pool/manifest_hocr_200.json)
  REPLAY_OUT           输出 jsonl 路径(已存在/撞 EXPECT/checkpoint→拒·防截断毁原数据)
  REPLAY_ALGO          连续臂算法(默认 ppo)
  REPLAY_EXPECT_JSONL  原 run 的 step4e_partial_*.jsonl(复现闸门·【强制】·缺则拒跑)
  STEP4E_BALANCED_DIR  均衡池本地 OT 目录(hocr 无追越→通常不需)
  ——若原 run 用了非默认 eval-env 配置：额外 export STEP4E_CONTINUOUS_SHIELD / STEP4E_GOAL_CONE_HALF /
    STEP4E_GOAL_V_FLOOR / STEP4E_AUGMENT_RHO / STEP4E_COLREGS_WEIGHT 与原 run 同。
用法：见 04；命令由主窗口逐字预检后给。
"""
import os, sys, json
from collections import Counter
import statistics as st


def _die(msg):
    print(f"\n❌ replay_dump 中止：{msg}", flush=True)
    sys.exit(2)


CKPT = os.environ.get("REPLAY_CKPT", "").strip()
MANIFEST = os.environ.get("REPLAY_MANIFEST", "").strip()
OUT = os.environ.get("REPLAY_OUT", "").strip()
ALGO = os.environ.get("REPLAY_ALGO", "ppo").strip().lower()
EXPECT = os.environ.get("REPLAY_EXPECT_JSONL", "").strip()
BALANCED = os.environ.get("STEP4E_BALANCED_DIR", "").strip() or None

if not CKPT or not MANIFEST or not OUT or not EXPECT:
    _die("须设 REPLAY_CKPT / REPLAY_MANIFEST / REPLAY_OUT / REPLAY_EXPECT_JSONL"
         "（EXPECT=复现闸门·【强制】·否则泄漏的 STEP4E_ export 会静默 dump 错配置数据·无法确证对齐）")
if not os.path.exists(CKPT + ".zip"):
    _die(f"checkpoint 不存在：{CKPT}.zip")
if not os.path.exists(CKPT + "_vecnorm.pkl"):
    _die(f"vecnorm 不存在：{CKPT}_vecnorm.pkl（缺则 obs 未归一化=策略看错分布=轨迹发散=复现失败）")
if not os.path.exists(MANIFEST):
    _die(f"manifest 不存在：{MANIFEST}")
if not os.path.exists(EXPECT):
    _die(f"REPLAY_EXPECT_JSONL 不存在：{EXPECT}")
# OUT 防误覆盖：open(w) 无条件截断→若 OUT 误指向原 partial/checkpoint/EXPECT 会静默毁原数据 → 已存在或撞受保护文件即拒
_outr = os.path.realpath(OUT)
_protected = {os.path.realpath(p) for p in (EXPECT, CKPT + ".zip", CKPT + "_vecnorm.pkl", CKPT + ".progress.json")}
if os.path.exists(OUT):
    _die(f"REPLAY_OUT 已存在：{OUT}（拒覆盖·换新路径或先手动删）")
if _outr in _protected:
    _die(f"REPLAY_OUT 与受保护文件(EXPECT/checkpoint)同路径：{OUT}（拒·防截断毁原数据）")

# manifest 须在 import run_step4e 前进 env（_DATASET_SIG 等模块级读；load_manifest_split 显式传参不依赖，但对齐口径）
os.environ.setdefault("STEP4E_MANIFEST", MANIFEST)
import run_step4e as S
from trb_env.usv_scenarios import load_scenario_pool

# 复现前打印 eval-env 配置 → 人工核对与原 run 同（闸门是自动的·这行给人看）
print(f"[replay_dump] eval-env 配置：shield={S._CONTINUOUS_SHIELD} cone={S._GOAL_CONE_HALF_RAD} "
      f"v_floor={S._GOAL_V_FLOOR} augment={S._AUGMENT_RHO} colregs_w={S._COLREGS_W_CONT} algo={ALGO}", flush=True)

# 建 test_pool（与主 run 同路径：load_manifest_split → load_scenario_pool → 同 40 局同序）
train_paths, test_paths, man_info = S.load_manifest_split(MANIFEST, BALANCED)
test_pool = load_scenario_pool(test_paths)
print(f"[replay_dump] test_pool={len(test_pool)} 局（{man_info['test_breakdown']}）← {MANIFEST}", flush=True)

# 重跑评测（return_per=True·additive·evaluate 已算好逐局标量）
agg, per = S.replay_eval(CKPT, "continuous", S._COLREGS_W_CONT, test_pool, continuous_algo=ALGO, return_per=True)
S._stamp_scenario_meta(per, man_info.get("test_meta"))   # 盖 scenario_type/file（replay_eval 未调·crossing vs head_on 分组需·additive 不改 agg/复现闸门）
print(f"[replay_dump] 复现 agg 到达率%={agg.get('到达率%')} 碰撞率%={agg.get('碰撞率%')} n_per={len(per)}", flush=True)

# ── 复现闸门（EXPECT 强制·上方已校验存在）：与原 jsonl 逐位校验（不同=配置/池/algo 不对=中止·绝不 dump 垃圾）──
# EXPECT 须是【单种子】原始 jsonl（空/多行/缺键 → 干净中止·非难看崩溃·守 docstring 承诺）
_lines = [l for l in open(EXPECT).read().splitlines() if l.strip()]
if not _lines:
    _die(f"REPLAY_EXPECT_JSONL 空文件/无有效行：{EXPECT}")
if len(_lines) > 1:
    _die(f"REPLAY_EXPECT_JSONL 有 {len(_lines)} 行（多种子合并文件？）→ 须指向【单种子】的 "
         f"step4e_partial_..._s<seed>.jsonl（闸门须对单个种子逐位比·多行会抓错行）")
try:
    orig = json.loads(_lines[-1])
except Exception as _e:
    _die(f"REPLAY_EXPECT_JSONL 末行不是合法 json：{EXPECT}（{type(_e).__name__}）")
ofin = orig.get("final")
ofp = orig.get("final_per")
if ofin is None or ofp is None:
    _die(f"REPLAY_EXPECT_JSONL 缺 final / final_per 键：{EXPECT}（不是 step4e 原始 jsonl？闸门无从比对·拒）")
COLS = ("到达率%", "碰撞率%", "违规次数/局", "紧急步%", "兜底步%", "Ep长s")
if not any(k in ofin for k in COLS):
    _die(f"REPLAY_EXPECT_JSONL 的 final 无任何钱图列（{COLS[:2]}...）：{EXPECT}（格式异常·agg 闸门无从比对·拒）")
bad = [k for k in COLS if k in ofin and agg.get(k) != ofin.get(k)]
if bad:
    _die(f"agg 不逐位复现原 final（差异列 {bad}）→ 配置/manifest/algo 与原 run 不一致，未 dump。\n"
         f"  复现 {[(k, agg.get(k)) for k in bad]}\n  原   {[(k, ofin.get(k)) for k in bad]}")
if len(per) != len(ofp):
    _die(f"per 局数 {len(per)} ≠ 原 final_per {len(ofp)} → 池不一致，未 dump")


def _epkey(e):                                            # 逐局身份：reached + 终端态（6 位=同轨迹判据·配置错→整轨迹不同非 1-ULP）
    es = e.get("end_state")
    if not es:
        return (bool(e.get("reached")),)
    return (bool(e.get("reached")),
            round(es["px"], 6), round(es["py"], 6), round(es["psi"], 6), round(es["v"], 6))


mism = [i for i in range(len(per))
        if per[i].get("scenario_idx") != ofp[i].get("scenario_idx") or _epkey(per[i]) != _epkey(ofp[i])]
if mism:
    _die(f"逐局 reached/end_state 不逐位复现原 final_per（{len(mism)}/{len(per)} 局不符·首 idx {mism[:5]}）→ 未 dump")
print(f"[replay_dump] ✅ 复现闸门通过：agg 6 列 + 全 {len(per)} 局 reached/end_state 与原 jsonl 逐位相同 "
      f"→ 新增 Step-0 标量可信", flush=True)

# dump per → jsonl（单行 record：溯源信息 + agg + final_per[含 Step-0 标量]）
rec = {"party": "Continuous-safe", "kind": "continuous", "replay_of": os.path.basename(CKPT),
       "manifest": MANIFEST, "continuous_algo": ALGO, "reproduced": True, "agg": agg, "final_per": per}
os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
with open(OUT, "w") as f:
    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
print(f"[replay_dump] ✅ dump {len(per)} 局 → {OUT}", flush=True)

# ── Step-0 失败局机制小结（reached=False 的分类·主窗口/人工快速看）──
fails = [e for e in per if not e.get("reached")]


def _cls(e):
    ib, ia, md = e.get("in_box_steps"), e.get("in_box_aligned_steps"), e.get("min_goal_dist_m")
    if ib and ib > 0 and ia == 0:
        return "门口捕获(进框·朝向没对上)"
    if ia and ia > 0:
        return "门内已对齐未达(卡时间门/未定住?)"
    if ib == 0 and md is not None and md <= 205:
        return "横向进不了框(cross-track·到门口出窄带)"
    if md is not None and md > 300:
        return "游荡(从未接近·最近>300m)"
    return "其它/未知"


c = Counter(_cls(e) for e in fails)
print(f"\n[replay_dump] 失败局机制小结（reached=False·{len(fails)}/{len(per)}）：{dict(c)}", flush=True)
mds = [e.get("min_goal_dist_m") for e in fails if e.get("min_goal_dist_m") is not None]
if mds:
    print(f"  min_goal_dist_m 分布：中位 {st.median(mds):.0f}m · 内100m {sum(1 for x in mds if x <= 100)}/{len(mds)} · "
          f"内300m {sum(1 for x in mds if x <= 300)}/{len(mds)} · >300m {sum(1 for x in mds if x > 300)}/{len(mds)}", flush=True)
# 分场景类型（对遇 vs 交叉·验 head_on→crossing 机制是否一致）
byt = {}
for e in fails:
    byt.setdefault(e.get("scenario_type", "?"), Counter())[_cls(e)] += 1
for t, cc in byt.items():
    print(f"  [{t}] {dict(cc)}", flush=True)
