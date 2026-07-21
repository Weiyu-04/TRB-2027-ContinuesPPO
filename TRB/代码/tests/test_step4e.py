"""
run_step4e.py 纯逻辑冒烟测试 —— 守护 step4e 四方对比的 Table-III 攸关逻辑（不跑训练）。
跑：/opt/miniconda3/envs/trb/bin/python -B 代码/tests/test_step4e.py

覆盖：make_split（分散/无泄漏/可复现/边界）、agg_mean_std（均值±样本std/空/单值）、
      env_cls_of、PARTIES（论文三方）、build_table3（Base/RR 紧急=–、多种子聚合）、
      done_keys（断点续跑解析/容错）。
训练编排（download/train_eval_one/main）非单元可测 → 由 run_step4e 端到端冒烟覆盖。
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import run_step4e as S

_fail = 0


def ok(name, cond, extra=""):
    global _fail
    if cond:
        print(f"[PASS] {name}")
    else:
        _fail += 1
        print(f"[FAIL] {name}  {extra}")


def raises(fn, exc=Exception):
    try:
        fn()
        return False
    except exc:
        return True
    except Exception:                              # noqa: BLE001
        return False


# ---------------- make_split ----------------
tr, te = S.make_split(2000, 0.30, 0)
ok("① make_split(2000,0.3) 尺寸 = 1400/600", len(tr) == 1400 and len(te) == 600, (len(tr), len(te)))
ok("② train/test 无重叠（无泄漏）", not (set(tr) & set(te)))
ok("③ 并集 = 全 2000（不丢场景）", sorted(tr + te) == list(range(2000)))

# 分散性：测试集应散布全程（5 段每段都有），非尾部连续块（审核 L21 M3 修正点）
seg_counts = [sum(1 for i in te if lo <= i < lo + 400) for lo in range(0, 2000, 400)]
ok("④ 测试集分散（5 段每段都有、非尾块）", all(c > 0 for c in seg_counts), seg_counts)
# 反例守护：尾部连续块 [1400..1999] 会让前 3 段=0 → 上面断言会 FAIL（确认断言真守护分散性）
tail_block = list(range(1400, 2000))
tail_counts = [sum(1 for i in tail_block if lo <= i < lo + 400) for lo in range(0, 2000, 400)]
ok("⑤ 守护有效性：尾块分布前段为 0（证 ④ 非平凡）", tail_counts[:3] == [0, 0, 0], tail_counts)

# 可复现 + 种子相关
tr2, te2 = S.make_split(2000, 0.30, 0)
ok("⑥ 同种子可复现", tr == tr2 and te == te2)
tr3, te3 = S.make_split(2000, 0.30, 1)
ok("⑦ 异种子不同划分", te != te3)

# 边界 assert（审核 L21 M1）
ok("⑧ n_total=1 拒（test 塌空）", raises(lambda: S.make_split(1, 0.30, 0), ValueError))
ok("⑨ test_frac=0 拒", raises(lambda: S.make_split(2000, 0.0, 0), ValueError))
ok("⑩ test_frac=1 拒", raises(lambda: S.make_split(2000, 1.0, 0), ValueError))
# 小 n_total 仍合法（冒烟用）
trs, tes = S.make_split(12, 0.30, 0)
ok("⑪ 小 n_total=12 合法（冒烟）", len(trs) + len(tes) == 12 and len(tes) >= 1 and not (set(trs) & set(tes)),
   (len(trs), len(tes)))

# strided 跨全库选取（03 L29：避免缩小实验聚集采样前 N 个 = 偏向失败的代理）。子标号 ⑪a-f 避免与下方 agg 的 ⑫-⑰ 撞号
trd, ted = S.make_split(200, 0.30, 0, pool_size=2000)
ok("⑪a strided(200,pool=2000) 跨全库（max>199、非聚集前200）", max(trd + ted) > 199, max(trd + ted))
ok("⑪b strided(200,pool=2000) = run_validation 选取 0,10,…,1990", sorted(trd + ted) == list(range(0, 2000, 10)))
ok("⑪c strided 200 个全 distinct + 尺寸 140/60",
   len(set(trd + ted)) == 200 and len(trd) == 140 and len(ted) == 60, (len(trd), len(ted)))
# 反例守护：默认 pool=None 仍聚集前200（max≤199）→ 证 ⑪a 非平凡、strided 真改了行为
trc, tec = S.make_split(200, 0.30, 0)
ok("⑪d 守护：默认 pool=None 仍聚集(max≤199) → 证 strided 真生效", max(trc + tec) <= 199, max(trc + tec))
# 边角：n_total 在 (pool/2,pool) 也铺开、不退化成聚集前 N
tre, tee = S.make_split(1500, 0.30, 0, pool_size=2000)
ok("⑪e 边角 strided(1500,pool=2000) 仍铺开(max>1500、非聚集到1499)", max(tre + tee) > 1500, max(tre + tee))
# full：n_total≥pool → pool 无影响，行为不变（与 ①③ 等价）
trf, tef = S.make_split(2000, 0.30, 0, pool_size=2000)
ok("⑪f 全量 strided(2000,pool=2000) 并集==range(2000)（无影响）", sorted(trf + tef) == list(range(2000)))


# ---------------- agg_mean_std ----------------
m, s = S.agg_mean_std([1, 2, 3])
ok("⑫ agg [1,2,3] → 均值 2", abs(m - 2.0) < 1e-12, m)
ok("⑬ agg [1,2,3] → 样本 std 1.0（ddof=1，非总体 0.816）", abs(s - 1.0) < 1e-12, s)
m, s = S.agg_mean_std([5.0])
ok("⑭ agg 单值 → (5, 0.0)", m == 5.0 and s == 0.0, (m, s))
m, s = S.agg_mean_std([])
ok("⑮ agg 空 → (nan, nan)", math.isnan(m) and math.isnan(s))
m, s = S.agg_mean_std([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
ok("⑯ agg 已知样本方差（μ=5, s=2.138…）", abs(m - 5.0) < 1e-12 and abs(s - 2.13809) < 1e-4, (m, s))


# ---------------- env_cls_of / PARTIES ----------------
from trb_env.usv_shield import ShieldedUSVEnv, UnshieldedUSVEnv

ok("⑰ env_cls_of('shielded') = ShieldedUSVEnv", S.env_cls_of("shielded") is ShieldedUSVEnv)
ok("⑱ env_cls_of('unshielded') = UnshieldedUSVEnv", S.env_cls_of("unshielded") is UnshieldedUSVEnv)
ok("⑲ env_cls_of 非法 kind 拒", raises(lambda: S.env_cls_of("xxx"), ValueError))
from trb_env.usv_continuous_shield import ContinuousProjectionEnv as _CPE
ok("⑲b env_cls_of('continuous') = ContinuousProjectionEnv（C2 连续臂）", S.env_cls_of("continuous") is _CPE)
ok("⑳ PARTIES = 四方（Base/RR/Discrete-safe/Continuous-safe + 正确 env/weight；C2 加连续臂）",
   S.PARTIES == [("Base", "unshielded", 0.0), ("Rule-reward", "unshielded", 1.0),
                 ("Discrete-safe", "shielded", 1.0), ("Continuous-safe", "continuous", 0.0)], S.PARTIES)

# select_parties（并行分片：STEP4E_PARTIES 选方）
ok("⑳a select_parties(None) = 全三方", S.select_parties(None) == S.PARTIES)
ok("⑳b select_parties('') = 全三方（空 → 不分片）", S.select_parties("") == S.PARTIES)
ok("⑳c select_parties('Discrete-safe') = 仅有盾",
   S.select_parties("Discrete-safe") == [("Discrete-safe", "shielded", 1.0)])
ok("⑳d select_parties 多选 + 保 PARTIES 序（非输入序）",
   S.select_parties("Discrete-safe,Base") == [("Base", "unshielded", 0.0),
                                               ("Discrete-safe", "shielded", 1.0)])
ok("⑳e select_parties 非法名拒", raises(lambda: S.select_parties("XXX"), ValueError))
ok("⑳f select_parties('Continuous-safe') = 仅连续臂（C2 分片可单选）",
   S.select_parties("Continuous-safe") == [("Continuous-safe", "continuous", 0.0)])


# ---------------- build_table3 ----------------
def _rec(party, kind, weight, seed, **cols):
    base = {"到达率%": 0.0, "碰撞率%": 0.0, "违规次数/局": 0.0, "紧急步%": 0.0, "Ep长s": 0.0}
    base.update(cols)
    return {"party": party, "kind": kind, "colregs_weight": weight, "seed": seed, "final": base}


recs = [
    _rec("Base", "unshielded", 0.0, 0, **{"违规次数/局": 2.6, "紧急步%": 0.0}),
    _rec("Base", "unshielded", 0.0, 1, **{"违规次数/局": 2.7, "紧急步%": 0.0}),
    _rec("Discrete-safe", "shielded", 1.0, 0, **{"到达率%": 86.0, "碰撞率%": 0.0, "紧急步%": 4.0}),
    _rec("Continuous-safe", "continuous", 0.0, 0, **{"到达率%": 90.0, "碰撞率%": 0.0, "紧急步%": 3.0}),
]
t3 = S.build_table3(recs)
ok("㉑ Base(无盾) 紧急步 = '–'", "–" in t3)
ok("㉒ Base 多种子聚合成行（seeds=2）", "Base" in t3 and "| 2" in t3)
ok("㉓ Discrete-safe(有盾) 紧急步显示数值（非 –）",
   any("Discrete-safe" in ln and "4.0" in ln and "–" not in ln for ln in t3.splitlines()))
ok("㉓b Continuous-safe(连续投影) 紧急步显示数值（非 –；C2 接入四方表）",
   any("Continuous-safe" in ln and "3.0" in ln and "–" not in ln for ln in t3.splitlines()))
# 未提供的 party 应标"（未完成）"而非崩
ok("㉔ 缺 Rule-reward → 标（未完成）不崩", "（未完成）" in t3)
# 守护有效性：unshielded 的紧急步若误显示 0.0±0.0 而非 –，则下面断言会 FAIL
base_line = next(ln for ln in t3.splitlines() if ln.startswith("Base"))
ok("㉕ 守护：Base 行确含 – 且不含紧急数值（证 ㉑ 非平凡）", "–" in base_line)

# 去重守护：同 (party,seed) 重复（并发双跑/续跑重训）→ 只算一次、取最后一条（防计数虚高/均值拉偏）
dup = [_rec("Base", "unshielded", 0.0, 0, **{"违规次数/局": 2.6}),
       _rec("Base", "unshielded", 0.0, 0, **{"违规次数/局": 9.9}),   # 同 (Base,0) 重复、值不同 → 应取此(最后)
       _rec("Base", "unshielded", 0.0, 1, **{"违规次数/局": 2.7})]
base_d = next(ln for ln in S.build_table3(dup).splitlines() if ln.startswith("Base"))
ok("㉕b 去重：同 (party,seed) 重复 → seeds=2 非 3", "| 2" in base_d, base_d)
ok("㉕c 去重保留最后一条（均值用 9.9 非 2.6 → (9.9+2.7)/2=6.30）", "6.30" in base_d, base_d)


# ---------------- done_keys（断点续跑）----------------
import json
import tempfile

with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as tf:
    tf.write(json.dumps({"party": "Base", "seed": 0, "final": {}}) + "\n")
    tf.write(json.dumps({"party": "Discrete-safe", "seed": 2, "final": {}}) + "\n")
    tf.write("坏行不是 json\n")                     # 容错：坏行跳过不崩
    tf.write(json.dumps({"没有 party 键": 1}) + "\n")  # 容错：缺键跳过
    _p = tf.name
dk = S.done_keys(_p)
ok("㉖ done_keys 解析 (party,seed) 集", dk == {("Base", 0), ("Discrete-safe", 2)}, dk)
ok("㉗ done_keys 容错坏行/缺键不崩", isinstance(dk, set))
ok("㉘ done_keys 不存在的文件 → 空集", S.done_keys("/tmp/_nope_step4e_xyz.jsonl") == set())
os.unlink(_p)

# append_record（并行安全写，flock 串行追加）round-trip + 与 done_keys 配合
with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as tfa:
    _pa = tfa.name
S.append_record(_pa, {"party": "Base", "seed": 0, "final": {"到达率%": 1.0}})
S.append_record(_pa, {"party": "Base", "seed": 1, "final": {"到达率%": 2.0}})
rr = S.read_records(_pa)
ok("㉘a append_record round-trip（2 条可读回、不黏行）", len(rr) == 2 and rr[0]["party"] == "Base", len(rr))
ok("㉘b append_record + done_keys 配合", S.done_keys(_pa) == {("Base", 0), ("Base", 1)})
os.unlink(_pa)


# ---------------- read_records + config_conflict（配置守卫，防 Table III 混配置）----------------
with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as tf2:
    tf2.write(json.dumps({"party": "Base", "seed": 0, "steps": 3_000_000, "n_total": 2000}) + "\n")
    tf2.write(json.dumps({"party": "Rule-reward", "seed": 0, "steps": 3_000_000, "n_total": 2000}) + "\n")
    tf2.write("坏行不是 json\n")
    _p2 = tf2.name
recs = S.read_records(_p2)
ok("㉙ read_records 读记录 + 跳坏行", len(recs) == 2, len(recs))
ok("㉚ config_conflict 同配置 → 无冲突（正常续跑）", S.config_conflict(recs, 3_000_000, 2000) == set())
# 签名 37 元组 (…,c_step,c_dwell,w_dwell,h_dwell,dwell_radius,b_dwell,c_reach,dock_radius,v_dock,rate_dock,continuous_shield,goal_cone_half_deg,goal_v_floor,augment_rho,arr_slack_start_deg,warmstart_ckpt)·warmstart_ckpt=🆕L190 热启动源 on/off(连续PPO臂专属·末元·缺字段旧记录/离散/off→None=从零训练隐含·防【热启动 vs 从零 两种训练流程】同TAG静默混写)·rate_dock=🆕rank1 泊位门控治抖(L173·None-able·插 v_dock 后 shield 前·缺字段/离散→None=off)：c_step=修法C（L123）；r_dwell 5 元（c_dwell/w_dwell/h_dwell/dwell_radius/b_dwell=🆕 入库赤字滞留成本·`03` L161/L162·连续臂专属·缺字段旧记录归一化 0.0/90.0/0.52/250.0/0.0=关隐含·插 c_step 后 continuous_shield 前）；c_reach/dock_radius/v_dock=🆕 第二条腿修法 3 元（`03` L172·连续臂专属·缺字段旧记录/离散归一化 1.5/0.0/2.5=接线前隐含·插 b_dwell 后 continuous_shield 前）；continuous_shield=🆕 P0 SE-RL 盾 on/off（L146）；goal_cone_half_deg/goal_v_floor=🆕 ρ0 朝目标锥（L147·连续臂专属·缺字段/离散记录归一化 None/2.0=锥关隐含）；augment_rho=🆕 腿1 态势感知增广 on/off（L150/L152·连续臂专属·缺字段/离散→False=关隐含）；arr_slack_start_deg=🆕 B1 到达门朝向课程 slack 起始度 on/off（L153·连续臂专属·缺字段/离散/off→None=关隐含·度=canonical 口径）
#   旧记录缺字段归一化：pool_size/n_seg→None·well_B/gamma→默认·lr_anneal→(None,1.0)·xtrack→(0.0,80.0)·alias→0.0·rate→0.0·惩罚退火→(None,None,0.65,0.25)·park→(0.0,400.0,4.0)·dwell→(0.0,90.0,0.52,250.0,0.0)（L58 n_seg·L82 well_B·L88 lr_anneal+xtrack·L97 alias·L98 rate·L103 惩罚退火·L116 dataset·L111/L112 park·L161/L162 dwell）
ok("㉛ config_conflict 异 steps → 报冲突（拦截混配置）",
   S.config_conflict(recs, 1_000_000, 2000) == {(3_000_000, 2000, None, None, 0.0, 500.0, 0.99, None, 1.0, 0.0, 80.0, 0.0, 0.0, None, None, 0.65, 0.25, "strided", 0.0, 400.0, 4.0, 0.0, 0.0, 90.0, 0.52, 250.0, 0.0, 1.5, 0.0, 2.5, None, True, None, 2.0, False, None, None)})
ok("㉜ config_conflict 异 n_total → 报冲突", S.config_conflict(recs, 3_000_000, 200) == {(3_000_000, 2000, None, None, 0.0, 500.0, 0.99, None, 1.0, 0.0, 80.0, 0.0, 0.0, None, None, 0.65, 0.25, "strided", 0.0, 400.0, 4.0, 0.0, 0.0, 90.0, 0.52, 250.0, 0.0, 1.5, 0.0, 2.5, None, True, None, 2.0, False, None, None)})
ok("㉝ config_conflict 空记录 → 无冲突（首次跑）", S.config_conflict([], 3_000_000, 2000) == set())
# pool_size 纳入签名：strided(pool=2000) vs 聚集(无pool) 判为不同配置（03 L29 防混选取静默进同表）
_recs_p = [{"party": "Base", "seed": 0, "steps": 3_000_000, "n_total": 200, "pool_size": 2000}]
ok("㉝b config_conflict 异 pool_size → 报冲突（拦 strided/聚集 混表）",
   S.config_conflict(_recs_p, 3_000_000, 200, None) == {(3_000_000, 200, 2000, None, 0.0, 500.0, 0.99, None, 1.0, 0.0, 80.0, 0.0, 0.0, None, None, 0.65, 0.25, "strided", 0.0, 400.0, 4.0, 0.0, 0.0, 90.0, 0.52, 250.0, 0.0, 1.5, 0.0, 2.5, None, True, None, 2.0, False, None, None)})
ok("㉝c config_conflict 同 pool_size → 无冲突（正常续跑）",
   S.config_conflict(_recs_p, 3_000_000, 200, 2000) == set())
# n_seg 纳入签名（`03` L58 #2）：中途改 STEP4E_NSEG 续跑同 jsonl → step 网格错位污染学习曲线 → 硬拒
_recs_ns = [{"party": "Base", "seed": 0, "steps": 3_000_000, "n_total": 200, "pool_size": 2000, "n_seg": 6}]
ok("㉝h config_conflict 异 n_seg → 报冲突（拦中途改 NSEG 续跑·step 网格错位·`03` L58 #2）",
   S.config_conflict(_recs_ns, 3_000_000, 200, 2000, 3) == {(3_000_000, 200, 2000, 6, 0.0, 500.0, 0.99, None, 1.0, 0.0, 80.0, 0.0, 0.0, None, None, 0.65, 0.25, "strided", 0.0, 400.0, 4.0, 0.0, 0.0, 90.0, 0.52, 250.0, 0.0, 1.5, 0.0, 2.5, None, True, None, 2.0, False, None, None)})
ok("㉝i config_conflict 同 n_seg → 无冲突（正常续跑）",
   S.config_conflict(_recs_ns, 3_000_000, 200, 2000, 6) == set())
# well_shaping_weight 纳入签名（修法A 进门势·`03` L82）：同 TAG 下 well_B=0 与 well_B=200 续跑 → 硬拒（防静默 skip+混表·接线复审 MEDIUM）
_recs_wb = [{"party": "Base", "seed": 0, "steps": 3_000_000, "n_total": 200, "pool_size": 2000, "n_seg": 6,
             "well_shaping_weight": 0.0, "shaping_radius": 500.0, "gamma": 0.99}]
ok("㉝j config_conflict 异 well_shaping_weight → 报冲突（修法A A/B 防 well_B 混表·L82）",
   S.config_conflict(_recs_wb, 3_000_000, 200, 2000, 6, well_shaping_weight=200.0) == {(3_000_000, 200, 2000, 6, 0.0, 500.0, 0.99, None, 1.0, 0.0, 80.0, 0.0, 0.0, None, None, 0.65, 0.25, "strided", 0.0, 400.0, 4.0, 0.0, 0.0, 90.0, 0.52, 250.0, 0.0, 1.5, 0.0, 2.5, None, True, None, 2.0, False, None, None)})
ok("㉝k config_conflict 同 well_shaping_weight → 无冲突（正常续跑）",
   S.config_conflict(_recs_wb, 3_000_000, 200, 2000, 6, well_shaping_weight=0.0) == set())
# lr_anneal_end/frac 纳入签名（学习率退火·`03` L88）：退火 on/off 或异终点续跑同 jsonl → 硬拒（防混配·与 well_B 对齐）
_recs_lr = [{"party": "Base", "seed": 0, "steps": 3_000_000, "n_total": 200, "pool_size": 2000, "n_seg": 6,
             "well_shaping_weight": 0.0, "shaping_radius": 500.0, "gamma": 0.99}]   # 无 lr_anneal 字段=旧记录·归一化 (None,1.0)
ok("㉝l config_conflict 异 lr_anneal_end（旧记录无=None vs 当前退火到 0.0）→ 报冲突（防退火/恒定混表·L88）",
   S.config_conflict(_recs_lr, 3_000_000, 200, 2000, 6, lr_anneal_end=0.0) == {(3_000_000, 200, 2000, 6, 0.0, 500.0, 0.99, None, 1.0, 0.0, 80.0, 0.0, 0.0, None, None, 0.65, 0.25, "strided", 0.0, 400.0, 4.0, 0.0, 0.0, 90.0, 0.52, 250.0, 0.0, 1.5, 0.0, 2.5, None, True, None, 2.0, False, None, None)})
ok("㉝m config_conflict 同 lr_anneal（均关=None/1.0）→ 无冲突（正常续跑·旧记录兼容）",
   S.config_conflict(_recs_lr, 3_000_000, 200, 2000, 6) == set())
_recs_lr2 = [{"party": "Base", "seed": 0, "steps": 3_000_000, "n_total": 200, "pool_size": 2000, "n_seg": 6,
              "well_shaping_weight": 0.0, "shaping_radius": 500.0, "gamma": 0.99,
              "lr_anneal_end": 0.0, "lr_anneal_frac": 1.0}]
ok("㉝n config_conflict 同 lr_anneal_end=0.0（两记录同退火）→ 无冲突",
   S.config_conflict(_recs_lr2, 3_000_000, 200, 2000, 6, lr_anneal_end=0.0, lr_anneal_frac=1.0) == set())
ok("㉝o config_conflict 异 lr_anneal_frac（0.5 vs 1.0）→ 报冲突（防退火比例混表）",
   S.config_conflict(_recs_lr2, 3_000_000, 200, 2000, 6, lr_anneal_end=0.0, lr_anneal_frac=0.5)
   == {(3_000_000, 200, 2000, 6, 0.0, 500.0, 0.99, 0.0, 1.0, 0.0, 80.0, 0.0, 0.0, None, None, 0.65, 0.25, "strided", 0.0, 400.0, 4.0, 0.0, 0.0, 90.0, 0.52, 250.0, 0.0, 1.5, 0.0, 2.5, None, True, None, 2.0, False, None, None)})
# xtrack_weight/xtrack_radius 纳入签名（对症 横向进带势·`03` L88）：well_X on/off 或异 R_lat 续跑同 jsonl → 硬拒
_recs_wx = [{"party": "Base", "seed": 0, "steps": 3_000_000, "n_total": 200, "pool_size": 2000, "n_seg": 6,
             "well_shaping_weight": 0.0, "shaping_radius": 500.0, "gamma": 0.99}]   # 无 xtrack 字段=旧记录·归一化 (0.0,80.0)
ok("㉝s config_conflict 异 xtrack_weight（旧记录无=0.0 vs 当前 200）→ 报冲突（防 well_X 混表·L88）",
   S.config_conflict(_recs_wx, 3_000_000, 200, 2000, 6, xtrack_weight=200.0) == {(3_000_000, 200, 2000, 6, 0.0, 500.0, 0.99, None, 1.0, 0.0, 80.0, 0.0, 0.0, None, None, 0.65, 0.25, "strided", 0.0, 400.0, 4.0, 0.0, 0.0, 90.0, 0.52, 250.0, 0.0, 1.5, 0.0, 2.5, None, True, None, 2.0, False, None, None)})
ok("㉝t config_conflict 同 xtrack（均关 0.0/80.0·旧记录兼容）→ 无冲突",
   S.config_conflict(_recs_wx, 3_000_000, 200, 2000, 6) == set())
ok("㉝u config_conflict 异 xtrack_radius（80 vs 60）→ 报冲突（防 R_lat 混表）",
   S.config_conflict([{**_recs_wx[0], "xtrack_weight": 200.0, "xtrack_radius": 80.0}], 3_000_000, 200, 2000, 6, xtrack_weight=200.0, xtrack_radius=60.0)
   == {(3_000_000, 200, 2000, 6, 0.0, 500.0, 0.99, None, 1.0, 200.0, 80.0, 0.0, 0.0, None, None, 0.65, 0.25, "strided", 0.0, 400.0, 4.0, 0.0, 0.0, 90.0, 0.52, 250.0, 0.0, 1.5, 0.0, 2.5, None, True, None, 2.0, False, None, None)})
# 惩罚退火纳入签名（`03` L103）：退火 on/off 或异终点/比例续跑同 jsonl → 硬拒（防混配·与 alias/rate/lr_anneal 对齐）
_recs_pa = [{"party": "Base", "seed": 0, "steps": 3_000_000, "n_total": 200, "pool_size": 2000, "n_seg": 6,
             "well_shaping_weight": 200.0, "shaping_radius": 500.0, "gamma": 0.99}]   # 无惩罚退火字段=旧记录·归一化 (None,None,0.65,0.25)
ok("㉝v config_conflict 异 rate_anneal_end（旧记录无=None vs 当前退火到 1.0）→ 报冲突（防退火/恒定 rate 混表·L103）",
   S.config_conflict(_recs_pa, 3_000_000, 200, 2000, 6, well_shaping_weight=200.0, rate_anneal_end=1.0)
   == {(3_000_000, 200, 2000, 6, 200.0, 500.0, 0.99, None, 1.0, 0.0, 80.0, 0.0, 0.0, None, None, 0.65, 0.25, "strided", 0.0, 400.0, 4.0, 0.0, 0.0, 90.0, 0.52, 250.0, 0.0, 1.5, 0.0, 2.5, None, True, None, 2.0, False, None, None)})
ok("㉝w config_conflict 同惩罚退火（均关=None/默认）→ 无冲突（正常续跑·旧记录兼容）",
   S.config_conflict(_recs_pa, 3_000_000, 200, 2000, 6, well_shaping_weight=200.0) == set())
ok("㉝x config_conflict 异 penalty_ramp_start_frac（0.65 vs 0.5·均退火 rate=1.0）→ 报冲突（防 ramp 比例混表）",
   S.config_conflict([{**_recs_pa[0], "rate_anneal_end": 1.0, "penalty_ramp_start_frac": 0.65, "penalty_anneal_frac": 0.25}],
                     3_000_000, 200, 2000, 6, well_shaping_weight=200.0, rate_anneal_end=1.0, penalty_ramp_start_frac=0.5)
   == {(3_000_000, 200, 2000, 6, 200.0, 500.0, 0.99, None, 1.0, 0.0, 80.0, 0.0, 0.0, 1.0, None, 0.65, 0.25, "strided", 0.0, 400.0, 4.0, 0.0, 0.0, 90.0, 0.52, 250.0, 0.0, 1.5, 0.0, 2.5, None, True, None, 2.0, False, None, None)})
# dataset 纳入签名（`03` L116 二审 MEDIUM）：strided 与 均衡 manifest 续跑同 jsonl → 硬拒（尤 FULL 默认两者 n_total=2000 数值签名相同·只靠 TAG 物理隔离不够·二审两 agent 抓）
_recs_ds = [{"party": "Base", "seed": 0, "steps": 3_000_000, "n_total": 2000, "n_seg": 10}]   # 无 dataset 字段=旧 strided 记录·归一化 "strided"
ok("㉝y config_conflict 异 dataset（旧 strided vs 当前均衡 manifest）→ 报冲突（防 strided/均衡混表·L116）",
   S.config_conflict(_recs_ds, 3_000_000, 2000, None, 10, dataset="balanced.json")
   == {(3_000_000, 2000, None, 10, 0.0, 500.0, 0.99, None, 1.0, 0.0, 80.0, 0.0, 0.0, None, None, 0.65, 0.25, "strided", 0.0, 400.0, 4.0, 0.0, 0.0, 90.0, 0.52, 250.0, 0.0, 1.5, 0.0, 2.5, None, True, None, 2.0, False, None, None)})
ok("㉝z config_conflict 同 dataset（均衡 manifest 同名）→ 无冲突（正常续跑）",
   S.config_conflict([{**_recs_ds[0], "dataset": "balanced.json"}], 3_000_000, 2000, None, 10, dataset="balanced.json") == set())
# park_weight/park_radius/park_v_target 纳入签名（想法B Φ_park 终端保速势·`03` L111/L112 二审 MINOR 补·连续臂专属）：同 TAG 下 park_w=0 与 park_w=20 续跑 → 硬拒（防 Φ_park A/B 混表·与 well_B/well_X/alias/rate 对齐）
_recs_pk = [{"party": "Continuous-safe", "seed": 0, "steps": 5_000_000, "n_total": 200, "pool_size": 2000, "n_seg": 10,
             "well_shaping_weight": 200.0, "shaping_radius": 500.0, "gamma": 0.99}]   # 无 park 字段=旧记录·归一化 (0.0,400.0,4.0)
ok("㉝pk1 config_conflict 异 park_weight（旧记录无=0.0 vs 当前 20）→ 报冲突（防 Φ_park A/B 混表·L111/L112）",
   S.config_conflict(_recs_pk, 5_000_000, 200, 2000, 10, well_shaping_weight=200.0, park_weight=20.0)
   == {(5_000_000, 200, 2000, 10, 200.0, 500.0, 0.99, None, 1.0, 0.0, 80.0, 0.0, 0.0, None, None, 0.65, 0.25, "strided", 0.0, 400.0, 4.0, 0.0, 0.0, 90.0, 0.52, 250.0, 0.0, 1.5, 0.0, 2.5, None, True, None, 2.0, False, None, None)})
ok("㉝pk2 config_conflict 同 park（均关 0.0/400.0/4.0·旧记录兼容）→ 无冲突（正常续跑）",
   S.config_conflict(_recs_pk, 5_000_000, 200, 2000, 10, well_shaping_weight=200.0) == set())
ok("㉝pk3 config_conflict 异 park_v_target（4.0 vs 3.0·均 park_w=20）→ 报冲突（防 V_target 混表）",
   S.config_conflict([{**_recs_pk[0], "park_weight": 20.0, "park_radius": 400.0, "park_v_target": 4.0}],
                     5_000_000, 200, 2000, 10, well_shaping_weight=200.0, park_weight=20.0, park_v_target=3.0)
   == {(5_000_000, 200, 2000, 10, 200.0, 500.0, 0.99, None, 1.0, 0.0, 80.0, 0.0, 0.0, None, None, 0.65, 0.25, "strided", 20.0, 400.0, 4.0, 0.0, 0.0, 90.0, 0.52, 250.0, 0.0, 1.5, 0.0, 2.5, None, True, None, 2.0, False, None, None)})
# c_step 纳入签名（修法C 每步生存成本·`03` L123·连续臂专属·非PBRS）：同 TAG 下 c_step=0 与 c_step=0.5 续跑 → 硬拒（防修法C A/B 混表·与 well_B/park 对齐）
ok("㉝cs1 config_conflict 异 c_step（旧记录无=0.0 vs 当前 0.5）→ 报冲突（防修法C A/B 混表·L123）",
   S.config_conflict(_recs_pk, 5_000_000, 200, 2000, 10, well_shaping_weight=200.0, c_step=0.5)
   == {(5_000_000, 200, 2000, 10, 200.0, 500.0, 0.99, None, 1.0, 0.0, 80.0, 0.0, 0.0, None, None, 0.65, 0.25, "strided", 0.0, 400.0, 4.0, 0.0, 0.0, 90.0, 0.52, 250.0, 0.0, 1.5, 0.0, 2.5, None, True, None, 2.0, False, None, None)})
ok("㉝cs2 config_conflict 同 c_step（均关 0.0·旧记录兼容）→ 无冲突（正常续跑）",
   S.config_conflict(_recs_pk, 5_000_000, 200, 2000, 10, well_shaping_weight=200.0, c_step=0.0) == set())
# ㉝p 复审 MAJOR 回归守护：冲突集同时含 OFF(lr_anneal_end=None) 与 ON(float) 记录 → main 报错处 sorted(conflict)
#   会因 None 与 float 不可比抛 TypeError 崩成裸 traceback；修法 sorted(conflict, key=str) 必须能排不崩。
_recs_mix = [{"party": "A", "seed": 0, "steps": 3_000_000, "n_total": 200, "pool_size": 2000, "n_seg": 6,
              "well_shaping_weight": 0.0, "shaping_radius": 500.0, "gamma": 0.99, "lr_anneal_end": None, "lr_anneal_frac": 1.0},
             {"party": "B", "seed": 1, "steps": 3_000_000, "n_total": 200, "pool_size": 2000, "n_seg": 6,
              "well_shaping_weight": 0.0, "shaping_radius": 500.0, "gamma": 0.99, "lr_anneal_end": 0.0, "lr_anneal_frac": 1.0}]
_conf_mix = S.config_conflict(_recs_mix, 3_000_000, 200, 2000, 6, lr_anneal_end=3e-5)   # 当前第三配置 → prior 两条(None/0.0)都进冲突集
_mix_ok = len(_conf_mix) == 2

def _plain_sorted_crashes():                            # 变异坐实：裸 sorted 在混 None/float 元组集上必崩（证明守护非空过）
    sorted(_conf_mix)
try:
    _plain_sorted_crashes(); _plain_crashed = False
except TypeError:
    _plain_crashed = True
ok("㉝p config_conflict 冲突集混 None/float(lr_anneal_end) → sorted(key=str) 不崩 + 裸 sorted 确崩（复审 MAJOR 守护·变异坐实）",
   _mix_ok and _plain_crashed and isinstance(sorted(_conf_mix, key=str), list) and len(sorted(_conf_mix, key=str)) == 2)
os.unlink(_p2)

# ㉝q 复审 MEDIUM 安全守护（subprocess·防静默烧 nan 模型）：STEP4E_LR_ANNEAL=inf/nan/foo → import 期 🔒 SystemExit(exit≠0)
import subprocess as _sub
_code = "import sys; sys.argv=['x']; import run_step4e"
_runp = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 代码/ 目录（run_step4e.py 所在）
def _import_rc(val):
    _e = dict(os.environ); _e["STEP4E_LR_ANNEAL"] = val
    return _sub.run([sys.executable, "-c", _code], cwd=_runp, env=_e, capture_output=True, text=True).returncode
ok("㉝q STEP4E_LR_ANNEAL=inf/nan/foo → import 期 🔒 SystemExit(exit≠0·防 inf/nan 静默流入 optimizer 烧 nan 模型)",
   _import_rc("inf") != 0 and _import_rc("nan") != 0 and _import_rc("foo") != 0)
ok("㉝r STEP4E_LR_ANNEAL=off/0/3e-5 → import 通过(exit=0·合法值不误拦)",
   _import_rc("off") == 0 and _import_rc("0") == 0 and _import_rc("3e-5") == 0)


# ---------------- aggregate_and_write 表头"选取="渲染（Agent B MINOR-1 补回归：strided/全库/聚集/混选取）----------------
_F = {"到达率%": 90.0, "碰撞率%": 0.0, "违规次数/局": 1.0, "紧急步%": 5.0, "Ep长s": 500.0}


def _hdr(records, n_total, total_steps=3_000_000):
    """跑 aggregate_and_write 取表头第一行（临时文件隔离，不碰真 结果/）。"""
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as _tf:
        for r in records:
            _tf.write(json.dumps(r) + "\n")
        _pp = _tf.name
    _op, _ot = S._PARTIAL, S._TABLE3
    S._PARTIAL, S._TABLE3 = _pp, _pp + ".t3"
    try:
        out = S.aggregate_and_write({r["seed"] for r in records}, "FULL（测试）", n_total, total_steps)
    finally:
        S._PARTIAL, S._TABLE3 = _op, _ot
        for _f in (_pp, _pp + ".t3"):
            os.path.exists(_f) and os.unlink(_f)
    return out.split("\n")[0]


_strided = [{"party": "Discrete-safe", "seed": s, "steps": 3_000_000, "n_total": 200,
             "pool_size": 2000, "n_train": 140, "n_test": 60, "final": _F} for s in (0, 1, 2)]
ok("㉝d 表头 strided 记录 → 'strided跨全库(pool=2000)'", "选取=strided跨全库(pool=2000)" in _hdr(_strided, 200))
_full = [{"party": "Discrete-safe", "seed": 0, "steps": 3_000_000, "n_total": 2000,
          "n_train": 1400, "n_test": 600, "final": _F}]                       # 无 pool_size = 全量(n≥库)
ok("㉝e 表头 全量(无pool,n=2000) → '全库'（非误导'聚集'）", "全库" in _hdr(_full, 2000))
_clustered = [{"party": "Discrete-safe", "seed": 0, "steps": 1_200_000, "n_total": 200,
               "n_train": 140, "n_test": 60, "final": _F}]                     # 旧聚集子集(无pool,n<库)
ok("㉝f 表头 旧聚集子集(无pool,n=200) → '前N聚集(⚠非全库子集)'", "前N聚集" in _hdr(_clustered, 200))
_mixed = [_strided[0], {"party": "Discrete-safe", "seed": 1, "steps": 1_200_000, "n_total": 200,
                        "n_train": 140, "n_test": 60, "final": _F}]            # strided(s0) + 聚集(s1) 混
ok("㉝g 表头 混选取 → '⚠混选取'告警（对齐 ent/clip 混配告警）", "⚠混选取" in _hdr(_mixed, 200))


# ---------------- anneal_ent_coef（熵退火纯逻辑，03 D25 + 审核加固：唯一新增训练旋钮）----------------
# p=0→START；p=0.5→中点；p≥1→END(clamp)；START==END→恒定(关退火、对照臂命门)。
_N = 1_800_000   # = 3M × 0.6（FULL 退火步数）
ok("㉞ anneal num=0 → START=0.03", abs(S.anneal_ent_coef(0.03, 0.005, _N, 0) - 0.03) < 1e-12)
ok("㉞a anneal num=N/2 → 中点 0.0175（D25 line345 声称值）",
   abs(S.anneal_ent_coef(0.03, 0.005, _N, _N / 2) - 0.0175) < 1e-12, S.anneal_ent_coef(0.03, 0.005, _N, _N / 2))
ok("㉞b anneal num=N → END=0.005", abs(S.anneal_ent_coef(0.03, 0.005, _N, _N) - 0.005) < 1e-12)
ok("㉞c anneal num=2N（超界）→ clamp END 不 overshoot",
   abs(S.anneal_ent_coef(0.03, 0.005, _N, 2 * _N) - 0.005) < 1e-12, S.anneal_ent_coef(0.03, 0.005, _N, 2 * _N))
_vals = [S.anneal_ent_coef(0.03, 0.005, _N, t) for t in range(0, _N + 1, _N // 10)]
ok("㉞d 单调不增", all(_vals[i] >= _vals[i + 1] - 1e-15 for i in range(len(_vals) - 1)), _vals)
# 含超界点（1.5N/2N/10N）→ 直接守护上 clamp：去 min(1.0,…) 则超界值跌破 0.005（变负）→ FAIL
_vals_ext = _vals + [S.anneal_ent_coef(0.03, 0.005, _N, k * _N) for k in (1.5, 2, 10)]
ok("㉞e 全程（含超界）∈ [0.005, 0.03]（不 overshoot、直接守护 clamp）",
   all(0.005 - 1e-12 <= v <= 0.03 + 1e-12 for v in _vals_ext), _vals_ext)
ok("㉞f START==END=0.01 → 恒 0.01（关退火=复现旧常量配方）",
   all(abs(S.anneal_ent_coef(0.01, 0.01, _N, t) - 0.01) < 1e-12 for t in (0, _N // 3, _N, 2 * _N)))
ok("㉞g anneal_steps=0 不除零崩（num=0→START、num>0→END）",
   abs(S.anneal_ent_coef(0.03, 0.005, 0, 0) - 0.03) < 1e-12 and abs(S.anneal_ent_coef(0.03, 0.005, 0, 5) - 0.005) < 1e-12)
# 反例守护：若公式误写成线性【升】则 anneal(N)>anneal(0) → 下面断言 FAIL（证 ㉞b/方向非平凡）
ok("㉞h 守护：退火是【降】非升", S.anneal_ent_coef(0.03, 0.005, _N, _N) < S.anneal_ent_coef(0.03, 0.005, _N, 0))


# ---------------- resolve_vecnorm_kwargs（clip_reward 消融旋钮，03 审核#1：默认=已验证配方）----------------
_vbase = {"norm_obs": True, "norm_reward": True, "clip_obs": 10.0}
_kw, _eff = S.resolve_vecnorm_kwargs(_vbase, None)
ok("㉟ clip None → 不加键、有效=10.0（sb3 默认=已验证配方 D22）", "clip_reward" not in _kw and _eff == 10.0, (_kw, _eff))
# 直接守护无副作用：传【真值】"50" 后入参 base 仍不应被注入 clip_reward（变异 kw=base 会让此行直接 FAIL）
S.resolve_vecnorm_kwargs(_vbase, "50")
ok("㉟a 传真值后不修改入参 base（无副作用，直接守护）", "clip_reward" not in _vbase, _vbase)
_kw2, _eff2 = S.resolve_vecnorm_kwargs(_vbase, "50")
ok("㉟b clip '50' → clip_reward=50.0 覆盖（消融臂）", _kw2.get("clip_reward") == 50.0 and _eff2 == 50.0, (_kw2, _eff2))
_kw3, _eff3 = S.resolve_vecnorm_kwargs(_vbase, "")
ok("㉟c clip ''（空串）→ 视为未设、吃默认 10.0", "clip_reward" not in _kw3 and _eff3 == 10.0)
# 防呆：clip≤0 静默归零/恒定化 reward → 硬拒（审核 Agent B 实证 clip=0 喂 PPO 的 reward 全 0）
ok("㉟d clip '0' → 拒（防归零 reward）", raises(lambda: S.resolve_vecnorm_kwargs(_vbase, "0"), ValueError))
ok("㉟e clip '-5' → 拒（防恒定化 reward）", raises(lambda: S.resolve_vecnorm_kwargs(_vbase, "-5"), ValueError))
ok("㉟f clip 'abc'（非数字）→ 干净 ValueError 不静默", raises(lambda: S.resolve_vecnorm_kwargs(_vbase, "abc"), ValueError))
# norm_reward 旋钮（03 L27：验"奖励归一化 std 除法"是否种子分裂元凶；默认不动=已验证配方）
_kwN, _ = S.resolve_vecnorm_kwargs(_vbase, None, "0")
ok("㉟g normR '0' → norm_reward=False（关奖励归一化）", _kwN.get("norm_reward") is False, _kwN)
ok("㉟h normR '0' 不改入参 base（无副作用）", _vbase.get("norm_reward") is True)
_kwT, _ = S.resolve_vecnorm_kwargs(_vbase, None, None)
ok("㉟i normR None → 保持 base 的 True（=已验证配方）", _kwT.get("norm_reward") is True)
_kwT2, _ = S.resolve_vecnorm_kwargs(_vbase, None, "1")
ok("㉟j normR '1'（非关键字）→ 不关（仍 True）", _kwT2.get("norm_reward") is True)
ok("㉟k normR 'false'/'no' 也认（大小写不敏感）",
   S.resolve_vecnorm_kwargs(_vbase, None, "false")[0].get("norm_reward") is False
   and S.resolve_vecnorm_kwargs(_vbase, None, "NO")[0].get("norm_reward") is False)


# ---------------- clip_reward_guard（L49 #1：CLIP_REWARD 仅离散诊断、误用 fail-fast；four-way F2 print→fail-fast）----------------
_p_disc = [("Discrete-safe", "shielded", 1.0), ("Base", "unshielded", 0.0)]
_p_4way = _p_disc + [("Continuous-safe", "continuous", 0.0)]
ok("㊱ clip 未设 → 任何 parties 通过（不 raise）",
   S.clip_reward_guard(_p_4way, False, False) is None and S.clip_reward_guard(_p_disc, False, True) is None)
ok("㊱a clip 设 + 含 Continuous-safe + ACK=True → 仍 raise（四方 clip 不对称任何情况都无效）",
   raises(lambda: S.clip_reward_guard(_p_4way, True, True), SystemExit))
ok("㊱b clip 设 + 含 Continuous-safe + ACK=False → raise",
   raises(lambda: S.clip_reward_guard(_p_4way, True, False), SystemExit))
ok("㊱c clip 设 + 仅离散臂 + ACK=False → raise（堵 leaked export 静默污染离散臂；launcher 拆单方旧 print 护栏旁路）",
   raises(lambda: S.clip_reward_guard(_p_disc, True, False), SystemExit))
# ㊳ 连续臂 n_envs 选择（红队 MAJOR 回归·L67-续3）：PPO 须拿 N_ENVS_PPO(并行 rollout)·SAC 拿 N_ENVS_SAC，别恒 1
ok("㊳ continuous_n_envs('ppo')==N_ENVS_PPO(默认=离散 N_ENVS) 且 !=N_ENVS_SAC（PPO 不被恒锁 1=丢并行效率）",
   S.continuous_n_envs("ppo") == S.N_ENVS_PPO and S.N_ENVS_PPO == S.N_ENVS and S.continuous_n_envs("ppo") != S.N_ENVS_SAC)
ok("㊳a continuous_n_envs('sac')==N_ENVS_SAC（off-policy 默认 1·不被 PPO 改）",
   S.continuous_n_envs("sac") == S.N_ENVS_SAC and S.continuous_n_envs("SAC") == S.N_ENVS_SAC)
# ㊳b 【call-site 回归·L67-续8 二审 F-2】㊳/㊳a 只测纯函数 continuous_n_envs，【不】保证 run() 的连续分派 call-site 真把它接进
# train_eval_one_continuous(n_envs=…)——把 run_step4e:954 改回硬编 N_ENVS_SAC，㊳/㊳a 仍全绿（'测了 helper 没测 dispatch' 盲区·
# 正是 L67-续3 教训②自己警告的复发）。此处用 AST 结构性守护 call-site：连续臂调用的 n_envs= 实参必须【由 continuous_n_envs(…) 赋值】，
# 非常量、非 N_ENVS_SAC/N_ENVS 直填。变异 call-site（硬编 n_envs）→ 本测试翻 FAIL（已本地变异坐实 load-bearing）。
import ast as _ast


def _continuous_callsite_wires_dispatch():
    tree = _ast.parse(open(S.__file__, encoding="utf-8").read())
    calls = [n for n in _ast.walk(tree)
             if isinstance(n, _ast.Call) and isinstance(n.func, _ast.Name)
             and n.func.id == "train_eval_one_continuous"]
    if not calls:
        return False, "未找到 train_eval_one_continuous 调用"
    nenvs_vars = set()
    for c in calls:
        kw = next((k for k in c.keywords if k.arg == "n_envs"), None)
        if kw is None:
            return False, "train_eval_one_continuous 调用缺 n_envs= 实参"
        if not isinstance(kw.value, _ast.Name):
            return False, "n_envs= 实参非变量（疑硬编常量）"
        if kw.value.id in ("N_ENVS_SAC", "N_ENVS"):
            return False, f"n_envs= 直填 {kw.value.id}（call-site 硬编·PPO 会丢并行 rollout）"
        nenvs_vars.add(kw.value.id)
    assigns = [n for n in _ast.walk(tree) if isinstance(n, _ast.Assign)]
    for vn in nenvs_vars:
        if not any(vn in {t.id for t in a.targets if isinstance(t, _ast.Name)}
                   and isinstance(a.value, _ast.Call) and isinstance(a.value.func, _ast.Name)
                   and a.value.func.id == "continuous_n_envs"
                   for a in assigns):
            return False, f"n_envs 变量 {vn} 非由 continuous_n_envs(...) 赋值（疑硬编）"
    return True, ""


_cs_ok, _cs_msg = _continuous_callsite_wires_dispatch()
ok("㊳b 【call-site 回归】run() 连续分派 train_eval_one_continuous(n_envs=…) 实参来自 continuous_n_envs(...)（非硬编·F-2）",
   _cs_ok, _cs_msg)


# ㊳c 【结构不变量·L90】凡自描述 well_shaping_weight 的记录 dict（离散臂 record + 连续臂 return + SAC/PPO 各支）
#   必须也自描述 xtrack_weight/xtrack_radius——否则 config_conflict 守卫对该臂的 well_X 混配【失明】
#   （L89⑧/L90④：连续臂 return@861 曾漏记 → well_X=200 记录 xtrack_weight=None → 守卫归一化 0.0 → 与 well_X=0 不可辨）。
#   变异：删任一记录的 xtrack_weight 键 → 本测翻 FAIL（已本地变异坐实 load-bearing）。
def _records_selfdescribe_xtrack():
    tree = _ast.parse(open(S.__file__, encoding="utf-8").read())
    bad = []
    for n in _ast.walk(tree):
        if not isinstance(n, _ast.Dict):
            continue
        keys = {k.value for k in n.keys if isinstance(k, _ast.Constant) and isinstance(k.value, str)}
        if "well_shaping_weight" in keys:                  # 配置自描述记录（修法A well_B + 对症 well_X 须成对）
            miss = {"xtrack_weight", "xtrack_radius"} - keys
            if miss:
                bad.append((getattr(n, "lineno", "?"), sorted(miss)))
    return (not bad), (f"自描述 well_B 但漏 xtrack 的记录 dict（行号,缺键）: {bad}" if bad else "")


_xt_ok, _xt_msg = _records_selfdescribe_xtrack()
ok("㊳c 【结构不变量】凡记 well_shaping_weight 的记录 dict 必也记 xtrack_weight/xtrack_radius（防连续臂 well_X 混配失明·L89⑧/L90）",
   _xt_ok, _xt_msg)
ok("㊱d clip 设 + 仅离散臂 + ACK=True → 通过（合法离散种子诊断 opt-in）",
   S.clip_reward_guard(_p_disc, True, True) is None)

# ---------------- aggregate_and_write clip_reward 跨臂不对称告警（L49 #1：补 cfg 去重以 ent_start 过滤漏检连续臂的缺口）----------------
def _agg_full(records, n_total, total_steps=3_000_000):
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as _tf:
        for r in records:
            _tf.write(json.dumps(r) + "\n")
        _pp = _tf.name
    _op, _ot = S._PARTIAL, S._TABLE3
    S._PARTIAL, S._TABLE3 = _pp, _pp + ".t3"
    try:
        return S.aggregate_and_write({r["seed"] for r in records}, "FULL（测试）", n_total, total_steps)
    finally:
        S._PARTIAL, S._TABLE3 = _op, _ot
        for _f in (_pp, _pp + ".t3"):
            os.path.exists(_f) and os.unlink(_f)

_clipmix = [
    {"party": "Discrete-safe", "seed": 0, "steps": 3_000_000, "n_total": 200, "pool_size": 2000,
     "n_train": 140, "n_test": 60, "clip_reward": 50.0, "ent_start": 0.01, "ent_end": 0.01,
     "ent_frac": 0.6, "norm_reward": True, "final": _F},
    {"party": "Continuous-safe", "seed": 0, "steps": 3_000_000, "n_total": 200, "pool_size": 2000,
     "n_train": 140, "n_test": 60, "clip_reward": None, "ent_start": None, "final": _F}]   # 连续臂 clip None→有效10
ok("㊲ clip 跨臂不对称(离散50 vs 连续None→10) → 表头告警（cfg 去重以 ent_start 过滤会漏检连续臂、本校验补上）",
   "clip_reward 跨臂不对称" in _agg_full(_clipmix, 200))
_clipsym = [{**r, "clip_reward": None} for r in _clipmix]                                  # 四方都默认 10 = 正式 run 平价
ok("㊲a clip 对称(都默认10) → 无不对称告警（正式四方 run 平价、不误报）",
   "clip_reward 跨臂不对称" not in _agg_full(_clipsym, 200))


# ---- Node L 守护（L52 MEDIUM·user 拍板"先加固"·破例补 tiny SAC smoke）----------------------------
# 连续臂(SAC hero) record 的 final_per(逐 episode 诊断 CAT2/3/4) + curves(SAC 内部训练曲线 CAT6) 当前【无持久测试】：
# build_table3 钱图不读它们、完成型 SMOKE 也不断言其内容 → 未来误删 record['final_per']/['curves'] 行或 callback
# 退化为不 append，会【静默通过 85 条 + 完成型 SMOKE】，直到 3M×多种子 SAC hero 臂跑完才发现训练曲线/逐 episode
# 诊断永久丢失（违 user『第一次全记齐不重复跑』、数小时重算）。故此处真跑 tiny SAC（~5s）断言两字段存在且结构对。
# 离线无场景夹具 → SKIP（非 FAIL，同 sac_train/continuous_shield 端到端块约定）。
import glob as _glob
_pool_dir = next((d for d in ("/tmp/trb_scenarios_pool", "/tmp/trb_scenarios")
                  if _glob.glob(os.path.join(d, "*.xml"))), None)
if _pool_dir is None:
    print("[SKIP] Node L 守护：无场景夹具（/tmp/trb_scenarios_pool|trb_scenarios）→ 离线跳过（非 FAIL）")
else:
    _ran = False
    import tempfile as _tf
    _ckdir = _tf.mkdtemp(prefix="trb_nodeL_ckpt_")        # L1c checkpoint 落临时目录（不污染 结果/checkpoints）
    try:
        from trb_env.usv_scenarios import load_scenario_pool
        _xmls = sorted(_glob.glob(os.path.join(_pool_dir, "*.xml")))
        _tpool = load_scenario_pool(_xmls[6:9])
        os.environ["STEP4E_LEARNING_STARTS"] = "50"   # L63 Fix③：默认 5000>tiny smoke 的 400 步=纯 warmup 无曲线；smoke 须设 <total_steps 才真训练出 actor_loss（守护④正测此）
        os.environ["STEP4E_CRITIC_LAYERNORM"] = "1"    # L67-续8（二审 BRO-4 覆盖盲区）：smoke 跑【完整 BRO】(LayerNorm + critic AdamW 权重衰减=生产主臂 _diagBRO 配置)
        os.environ["STEP4E_CRITIC_WD"] = "1e-3"        #   →守护⑤ replay 真覆盖 wd>0 路径（裸 SAC.load 会崩·load_sac_for_eval 修后才绿·BRO-3）
        _recL = S.train_eval_one_continuous(0, _xmls[:6], _tpool, total_steps=400,
                                            n_seg=1, n_envs=1, subproc=False, ckpt_dir=_ckdir)
        _ran = True
    except (FileNotFoundError, ImportError) as _e:             # 真夹具/环境问题（场景文件缺/依赖缺）→ SKIP，不误判 FAIL
        print(f"[SKIP] Node L 守护：tiny SAC 跑不起来（夹具/环境·文件/依赖缺）→ 跳过（非 FAIL）：{type(_e).__name__}: {_e}")
    except Exception as _e:                                     # 🔴 代码错(TypeError/AttributeError/断链等)→ 必须 FAIL·绝不吞成 SKIP（复审 wvbzr5av3 教训：maker 断链 TypeError 曾被 except Exception 吞成假绿=本项目"测试绿≠没问题"活教材）
        ok(f"Node L 守护 tiny SAC 端到端不崩于【代码错】（防断链/形参缺·得 {type(_e).__name__}: {_e}）", False)
    if _ran:
        _fp = _recL.get("final_per"); _cv = _recL.get("curves")
        ok("Node L 守护① 连续臂 record 含【非空 final_per】（逐 episode 诊断·防 record['final_per'] 行被静默删）",
           isinstance(_fp, list) and len(_fp) > 0)
        ok("Node L 守护② 连续臂 record 含【非空 curves】（SAC hero 训练曲线·防 callback 退化/record['curves'] 行被静默删）",
           isinstance(_cv, list) and len(_cv) > 0)
        ok("Node L 守护③ final_per 逐 episode 带 CAT4 诊断键 proj_correction_mean/source_counts + source 计数真非空（防只剩 CAT2/3 或诊断空转）",
           isinstance(_fp, list) and len(_fp) > 0
           and all(("proj_correction_mean" in p and "source_counts" in p) for p in _fp)
           and any(sum(p["source_counts"].values()) > 0 for p in _fp))      # 值校验：source 真被逐步计数（非仅键在）
        ok("Node L 守护④ curves 带 SAC 内部键 step/actor_loss 且 actor_loss 真被捕获非全 None（防 logger 读断/total_steps≤learning_starts 致曲线静默全空·红队 L53 MEDIUM）",
           isinstance(_cv, list) and len(_cv) > 0
           and all(("step" in c and "actor_loss" in c) for c in _cv)
           and any(c.get("actor_loss") is not None for c in _cv))            # 值校验：SAC 内部量真被捕获（健康 run ~199/300 点满足）
        ok("Node L 守护⑨ 连续臂 record 自描述【SAC 稳化 knob】critic_layernorm/n_critics/tau/target_q_clip/critic_weight_decay 全记入（L67-续2/续7·A/B 产物可分辨配置·别只靠 TAG 文件名·preflight 抓出此缺陷）",
           all(k in _recL for k in ("critic_layernorm", "n_critics", "tau", "target_q_clip", "critic_weight_decay")))
        ok("Node L 守护⑥ curves 的 ep_rew_mean 原始 episode 回报真被捕获非全 None（callback 自算·替 Monitor·防 get_original_reward 路径断·L54-续）",
           isinstance(_cv, list) and len(_cv) > 0
           and any(c.get("ep_rew_mean") is not None for c in _cv))           # 值校验：≥1 训练 episode 完成且原始回报记进 curves
        try:                                                                 # 守护⑤ L1c：checkpoint 存→重载→eval 逐位复现 final（不重跑总保险）
            _agg2 = S.replay_eval(_recL["ckpt"], "continuous", 0.0, _tpool)
            _cols = ("到达率%", "碰撞率%", "违规次数/局", "紧急步%", "兜底步%", "Ep长s")
            _replay_ok = (_recL.get("ckpt") is not None
                          and all(_agg2.get(k) == _recL["final"].get(k) for k in _cols))
        except Exception as _e2:
            _replay_ok = False
            print(f"  (Node L 守护⑤ replay 异常：{_e2})")
        ok("Node L 守护⑤ checkpoint 存→重载→eval【逐位复现 final 六列】（不重跑总保险·L1c·D42-Lschema CAT1）", _replay_ok)
        # 守护⑤c（`03` L108·复审抓的钱图地雷）：连续臂主臂=PPO（L69）·但 replay_eval 原硬编 load_sac_for_eval→对 PPO checkpoint【崩】。
        #   修后按算法分派（显式实参>progress.json sidecar>默认 sac）。此守护训 tiny 连续 PPO→replay_eval 应走 PPO.load 分支逐位复现 final。
        _ppo_replay_ok = False
        _ppo_per_ok = False                                                  # 守护⑤c-per：return_per=True→(agg,per)·per 带 Step-0 进近标量·默认仍单 agg（向后兼容）
        _prev_algo = os.environ.get("STEP4E_CONTINUOUS_ALGO")
        _prev_ent = (S.ENT_START, S.ENT_END)                                 # PPO 臂要求常量 ent（START==END·生产 run_rate_anneal 配方）·否则 :761 fail-fast
        try:                                                                 # subproc=False 同守护⑤/⑦（测试顶层无 __main__ guard）
            os.environ["STEP4E_CONTINUOUS_ALGO"] = "ppo"                      # 连续臂用 PPO 训（train_eval_one_continuous 在 :769 读此 env）
            S.ENT_START = S.ENT_END = 0.01                                    # 常量 ent（=生产四方 PPO 配方·过 PPO ent 退火不对称守卫）
            _recP = S.train_eval_one_continuous(0, _xmls[:6], _tpool, total_steps=400,
                                                n_seg=1, n_envs=1, subproc=False, ckpt_dir=_ckdir)
            _aggP_explicit = S.replay_eval(_recP["ckpt"], "continuous", 0.0, _tpool, continuous_algo="ppo")  # 显式 algo 路径
            _aggP_sidecar = S.replay_eval(_recP["ckpt"], "continuous", 0.0, _tpool)                          # 不传→读 sidecar 认 ppo（progress.json）
            _colsP = ("到达率%", "碰撞率%", "违规次数/局", "紧急步%", "兜底步%", "Ep长s")
            _ppo_replay_ok = (_recP.get("ckpt") is not None
                              and _recP.get("continuous_algo") == "ppo"       # 训的真是 PPO（非误判）
                              and all(_aggP_explicit.get(k) == _recP["final"].get(k) for k in _colsP)   # 显式 algo 逐位复现
                              and all(_aggP_sidecar.get(k) == _recP["final"].get(k) for k in _colsP))   # sidecar 认算法 逐位复现
            # 守护⑤c-per（Step-0 replay-dump 依赖）：return_per=True→(agg,per)·per 逐 episode 带 Step-0 进近 4 标量·agg 与单返回逐位同·默认仍返回单 agg dict
            _aP, _perP = S.replay_eval(_recP["ckpt"], "continuous", 0.0, _tpool, continuous_algo="ppo", return_per=True)
            _S0KEYS = ("min_goal_dist_m", "heading_err_at_min_deg", "in_box_steps", "in_box_aligned_steps", "speed_at_min_ms", "max_speed_ms", "speed_reversals")
            _ppo_per_ok = (isinstance(_perP, list) and len(_perP) == len(_tpool)
                           and all(all(k in e for k in _S0KEYS) for e in _perP)                 # per 每局带 Step-0 进近标量（evaluate 已算·return_per 透出·丢 per→翻 FAIL）
                           and all(_aP.get(k) == _aggP_explicit.get(k) for k in _colsP)         # return_per 的 agg == 单返回 agg（同复现·不改语义）
                           and isinstance(_aggP_explicit, dict))                                # 默认 return_per=False 仍返回单 agg dict（非 tuple·向后兼容既有调用点）
        except Exception as _e2p:
            print(f"  (Node L 守护⑤c 连续 PPO replay 异常：{_e2p})")
        finally:
            S.ENT_START, S.ENT_END = _prev_ent                              # 复原 ENT 全局（不污染后续测试）
            if _prev_algo is None:
                os.environ.pop("STEP4E_CONTINUOUS_ALGO", None)               # 复原 env（不污染后续测试）
            else:
                os.environ["STEP4E_CONTINUOUS_ALGO"] = _prev_algo
        ok("Node L 守护⑤c checkpoint 存→replay_eval【连续 PPO 主臂】逐位复现 final（`03` L108·原 load_sac_for_eval 对 PPO 崩=钱图地雷·修后 PPO.load 分支·显式 algo+sidecar 两路都验）", _ppo_replay_ok)
        ok("Node L 守护⑤c-per replay_eval(return_per=True)→(agg,per)·per 逐局带 Step-0 进近 4 标量·agg 与单返回逐位同·默认仍单 agg dict（向后兼容·Step-0 replay-dump 依赖·丢 per/破签名即翻 FAIL）", _ppo_per_ok)
        _disc_ok = False                                                     # 守护⑦ D1-F5：补【离散臂】replay 覆盖（守护⑤仅测连续臂 SAC.load·离散 MaskablePPO.load 分支原仓库零守护=覆盖不对称）
        try:                                                                 # subproc=False：测试顶层无 __main__ guard、SubprocVecEnv spawn 会重导入本文件 → 离散臂走 DummyVecEnv
            _recD = S.train_eval_one("Discrete-safe", "shielded", 1.0, 0, _xmls[:6], _tpool,
                                     total_steps=400, n_seg=1, n_envs=1, subproc=False, ckpt_dir=_ckdir)
            _aggD = S.replay_eval(_recD["ckpt"], "shielded", 1.0, _tpool)
            _colsD = ("到达率%", "碰撞率%", "违规次数/局", "紧急步%", "Ep长s")  # 离散 final 五列（无连续臂的"兜底步%"）
            _disc_ok = (_recD.get("ckpt") is not None
                        and all(k in _aggD and k in _recD["final"] for k in _colsD)   # 列齐才比（防两边同缺列→None==None 空真·2 对抗 agent 同提）
                        and all(_aggD.get(k) == _recD["final"].get(k) for k in _colsD))
        except Exception as _e3:
            print(f"  (Node L 守护⑦ 离散 replay 异常：{_e3})")
        ok("Node L 守护⑦ 【离散臂】checkpoint 存→重载→eval【逐位复现 final 五列】（D1-F5·MaskablePPO.load replay 分支守护·与守护⑤连续臂对称）", _disc_ok)
        # 守护⑧ D42-L2-续 CAT5：末段评估对 TRAJ_EXAMPLE_IDXS(默认0,1,2)记示例轨迹→落 final_per（端到端·防 traj_idxs 接线被删/钱图来源回退零记录）
        _want = set(S.TRAJ_EXAMPLE_IDXS) if S.TRAJ_EXAMPLE_IDXS else set()
        ok("Node L 守护⑧ CAT5：final_per 中 scenario_idx∈TRAJ_EXAMPLE_IDXS 的局带【非空 traj + goal=[x,y]】（末段记示例轨迹·钱图列不受影响·D42-L2-续）",
           isinstance(_fp, list) and len(_fp) > 0 and len(_want) > 0
           and any(p.get("traj") for p in _fp)                                    # 至少一局真记到（_tpool 3 局 ⊂ 默认{0,1,2}）
           and all((p.get("traj") and isinstance(p.get("goal"), list) and len(p["goal"]) == 2)
                   for p in _fp if p.get("scenario_idx") in _want))               # 选中局必带 traj + goal（接线断→翻 FAIL）
    import shutil as _sh
    _sh.rmtree(_ckdir, ignore_errors=True)                                   # 清理 L1c 临时 checkpoint


# ---------------- P3(L147 复审补测) _stamp_scenario_meta（分层盖章·additive·idx 键控·no-op 守卫·此前 0 committed 用例）----------------
_p3_meta = [{"type": "head_on", "file": "T-0.xml"}, {"type": "crossing", "file": "T-9.xml"}]
_p3_recs = [{"scenario_idx": 0, "reached": True, "violations": 2},
            {"scenario_idx": 1, "reached": False, "violations": 0}]
S._stamp_scenario_meta(_p3_recs, _p3_meta)
ok("P3 _stamp 按 scenario_idx 盖 scenario_type/file",
   _p3_recs[0]["scenario_type"] == "head_on" and _p3_recs[0]["scenario_file"] == "T-0.xml"
   and _p3_recs[1]["scenario_type"] == "crossing" and _p3_recs[1]["scenario_file"] == "T-9.xml")
ok("P3 _stamp additive（既有键 reached/violations 不动·钱图 agg 不受影响）",   # .get()：非 additive(抹键)→干净 [FAIL] 而非 KeyError 中止后续断言（变异审 MINOR·L147）
   _p3_recs[0].get("reached") is True and _p3_recs[0].get("violations") == 2 and _p3_recs[1].get("reached") is False)
# no-op：meta=None/空（strided 模式）→ 不盖任何键（钱图 bit-identical 前提）
_p3_r2 = [{"scenario_idx": 0, "reached": True}]
S._stamp_scenario_meta(_p3_r2, None)
S._stamp_scenario_meta(_p3_r2, [])
ok("P3 _stamp meta=None/空 → no-op（strided 不盖·钱图 bit-identical）", "scenario_type" not in _p3_r2[0])
# 越界/负/非 int idx / 缺 idx → 该条 no-op（不 IndexError·不误盖）
_p3_r3 = [{"scenario_idx": 5}, {"scenario_idx": -1}, {"scenario_idx": "x"}, {"reached": True}]
S._stamp_scenario_meta(_p3_r3, _p3_meta)   # meta 仅 2 项
ok("P3 _stamp 越界(5)/负(-1)/非int('x')/缺 idx → 该条不盖·不崩",
   all("scenario_type" not in e for e in _p3_r3))
ok("P3 _stamp seg_per 空 → no-op 不崩", S._stamp_scenario_meta([], _p3_meta) is None)


print()
if _fail == 0:
    print("✅ 全部 PASS")
else:
    print(f"❌ {_fail} 项 FAIL")
    sys.exit(1)
