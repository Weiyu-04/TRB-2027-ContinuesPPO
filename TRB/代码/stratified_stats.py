"""分层统计工具（user 2026-07-03 要求:每个指标都要【总 + 分场景类型 + 分难度档 + 每场景成功数】）。
纯下游消费者:读 manifest(定类型顺序) + 场景池(算 min-CPA 难度) + 各臂 final_per jsonl → 打印总+分层。
用法: python stratified_stats.py <manifest.json> <场景池目录> <arm1.jsonl> [arm2.jsonl ...]
  (arm 文件名含臂名即可;多种子=同臂多文件·跨种子聚合)

🔴 对齐契约（2026-07-04 复审修·L146）：jsonl 的 scenario_idx = eval 期 test_pool 位置序，
  test_pool = load_manifest_split 的 te_t(=_download(head_on+crossing)·【丢弃下载失败项】) + te_ot(overtaking)。
  build_scenario_meta 必须【镜像这套解析+丢弃】(缓存存在 >1000B 才算成功)才能与 eval 位置对齐；
  否则任一 head_on/crossing 下载失败 → 其后全部标签错位。main() 另加【对齐硬断言】：
  distinct scenario_idx 数 ≠ len(meta) 或越界 → 中止(防 strided jsonl 喂 balanced manifest / eval期丢失 等污染)。
"""
import sys, os, json, glob, re
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from commonocean.common.file_reader import CommonOceanFileReader

DT = 10.0
DIFF_TIERS = [("很难<562m", 0, 562), ("难562-1000m", 562, 1000), ("中1-2km", 1000, 2000), ("易>2km", 2000, 1e18)]

def keepcourse_mincpa(path):
    sc, pps = CommonOceanFileReader(path).open(); pp = list(pps.planning_problem_dict.values())[0]
    ob = list(sc.dynamic_obstacles)
    if not ob: return None
    ob = ob[0]; ist = pp.initial_state
    p0 = np.asarray(ist.position, float); v0 = float(ist.velocity); ori0 = float(ist.orientation)
    vel = np.array([np.cos(ori0), np.sin(ori0)]) * v0; dm = 1e18
    for t in range(0, 171):
        s = ob.initial_state if t == 0 else ob.prediction.trajectory.state_at_time_step(t)
        if s is None: break
        dm = min(dm, np.linalg.norm((p0 + vel*t*DT) - np.asarray(s.position, float)))
    return dm

def _present(p):
    """镜像 run_step4e._download 的缓存判定（存在且 >1000B 才算成功·否则视为被丢弃）。"""
    return os.path.exists(p) and os.path.getsize(p) > 1000

def build_scenario_meta(manifest_path, pool, balanced_dir=None):
    """scenario_idx → (type, difficulty_tier)。顺序【必须镜像 eval 的 test_pool】：
       te_t = _download(head_on test + crossing test)【丢弃下载失败/缺失项】，te_ot = overtaking(balanced_dir)。
       与旧版差异（bug 修）：① head_on/crossing 缺文件时【跳过】(镜像 _download 丢弃)而非占位→保持位置对齐；
       ② overtaking 走 balanced_dir(=manifest 同目录)非场景池 pool。"""
    man = json.load(open(manifest_path))
    bdir = balanced_dir or os.path.dirname(os.path.abspath(manifest_path))
    order = []  # (type, id, path)·顺序=te_t(head_on+crossing·丢弃缺失) + te_ot(overtaking)
    for typ in ("head_on", "crossing"):                    # T-id → 场景池；缺失=eval 期 _download 丢弃=跳过
        for i in man[typ]["test"]:
            p = os.path.join(pool, f"T-{i}.xml")
            if _present(p):
                order.append((typ, i, p))
    for i in man["overtaking"]["test"]:                    # OT → balanced_dir（eval 追加在 te_t 之后）
        p = os.path.join(bdir, os.path.basename(str(i)))
        if _present(p):
            order.append(("overtaking", i, p))
    meta = []
    for typ, i, p in order:
        cpa = None
        try: cpa = keepcourse_mincpa(p)
        except Exception: cpa = None
        tier = next((nm for nm, lo, hi in DIFF_TIERS if cpa is not None and lo <= cpa < hi), "未知")
        meta.append({"type": typ, "cpa": cpa, "tier": tier})
    return meta

METRICS = [("到达率%", "reached", True, 100), ("碰撞率%", "collided", True, 100),
           ("违规/局", "violations", False, 1), ("jerk航向抖", "ctrl_jerk_norm_mean", False, 1),
           ("路径m", "path_len_m", False, 1), ("CPA净空m", "cpa_clearance_m", False, 1)]

def arm_name(f): return os.path.basename(f).replace("step4e_partial_", "").replace(".jsonl", "")

def arm_base(nm):
    """臂名去种子后缀分组。只裁末尾 _s<digits>（bug 修：旧 rsplit('_s') 会误裁 baseW0_shield→baseW0）。"""
    m = re.match(r"^(.*)_s\d+$", nm)
    return m.group(1) if m else nm

def agg_group(rows, meta, idxs):
    """一组场景(idxs)的各指标聚合(跨该组场景+跨种子)。"""
    out = {"n": len(idxs)}
    reached_mask = {}
    for label, key, is_bool, scale in METRICS:
        vals = []
        for r in rows:                       # 每种子一个 final_per
            for e in r:
                si = e.get("scenario_idx")
                if si in idxs:
                    v = e.get(key)
                    if v is None: continue
                    if key in ("ctrl_jerk_norm_mean", "path_len_m", "cpa_clearance_m") and not e.get("reached"):
                        continue             # 平滑/路径/CPA 只在到达局算(公平)
                    vals.append(float(v))
        out[label] = (np.mean(vals)*scale if vals else float("nan"))
    # 每场景成功数(到达)
    succ = 0; tot = 0
    for r in rows:
        for e in r:
            if e.get("scenario_idx") in idxs:
                tot += 1; succ += 1 if e.get("reached") else 0
    out["成功/总(局)"] = f"{succ}/{tot}"
    return out

def _resolve_scenario_file(fname, pool, balanced_dir=None):
    """按 basename 找场景文件：先 pool（head_on/crossing 的 T-*.xml），再 balanced_dir（overtaking 的 OT-*.xml 只在这·C2/L147）。缺→None。
    修 meta_from_records 原只 join(pool, file) 致 OT 全 tier='未知'（追越难度分层整体消失·C2 复审 L147）。"""
    if not fname:
        return None
    b = os.path.basename(str(fname))
    for d in (pool, balanced_dir):
        if d:
            p = os.path.join(d, b)
            if os.path.exists(p):
                return p
    return None


def meta_from_records(arms, pool, balanced_dir=None):
    """优先路径（L146·新 run）：record 自带 scenario_type/scenario_file → 直接建 meta·【类型维零位置推导·无错位洞】。
    返回 meta 或 None（record 无 scenario_type=旧 run → 回退位置法）。
    ⚠️ tier(难度)维仍需按 scenario_file 现算 CPA→须 balanced_dir 出口解 OT（否则 OT 难度全'未知'·C2/L147·非头条钱图/类型分层来源）。"""
    rec = {}   # scenario_idx -> {type, file}
    for rows in arms.values():
        for r in rows:
            for e in r:
                si, st = e.get("scenario_idx"), e.get("scenario_type")
                if isinstance(si, int) and st is not None:
                    if si in rec and rec[si]["type"] != st:
                        sys.exit(f"🔴 scenario_idx {si} 在不同臂/种子 scenario_type 不一致({rec[si]['type']} vs {st}) → 混了不同测试集，中止。")
                    rec.setdefault(si, {"type": st, "file": e.get("scenario_file")})
    if not rec:
        return None
    n = max(rec) + 1
    meta = []
    for i in range(n):
        m = rec.get(i)
        if m is None:
            meta.append({"type": "unknown", "cpa": None, "tier": "未知"}); continue
        cpa = None
        fp = _resolve_scenario_file(m.get("file"), pool, balanced_dir)   # 🆕 C2(L147)：先 pool 再 balanced_dir·否则 OT-*.xml 找不到→追越难度分层全"未知"
        if fp:
            try: cpa = keepcourse_mincpa(fp)
            except Exception: cpa = None
        tier = next((nm for nm, lo, hi in DIFF_TIERS if cpa is not None and lo <= cpa < hi), "未知")
        meta.append({"type": m["type"], "cpa": cpa, "tier": tier})
    return meta

def main():
    manifest, pool = sys.argv[1], sys.argv[2]
    arm_files = sys.argv[3:]
    # 按臂名分组多种子（先读·供 record-meta 优先路径用）
    arms = {}
    for f in arm_files:
        base = arm_base(arm_name(f))
        d = json.loads([l for l in open(f) if l.strip()][-1])
        arms.setdefault(base, []).append(d.get("final_per", []))
    # 🆕 C2(L147) soft 守卫：各臂 scenario_idx 覆盖范围不一致=可能混了不同测试集/某臂 eval 丢数据→分层不可比。
    #   record 路径是 manifest-free 设计(无"应有 n 个"的外部基数参照)→无法硬断言；对【跨臂范围不一致】warn(不硬停·防误杀合法子集比较·L146 残余假阴性的可见化缓解)。
    _cov = {}
    for _b, _rows in arms.items():
        _s = {e.get("scenario_idx") for _r in _rows for e in _r if isinstance(e.get("scenario_idx"), int)}
        if _s: _cov[_b] = (min(_s), max(_s), len(_s))
    if len(set(_cov.values())) > 1:
        print(f"  ⚠️ 各臂 scenario_idx 覆盖不一致 {_cov} → 疑混了不同测试集/某臂 eval 丢数据·分层数可能不可比（record 路径残余·L147·非硬停）")
    # 🟢 优先：record 自带 scenario_type/file（L146·新 run·类型维零位置推导·无错位洞）；否则回退位置法 + 对齐硬断言
    meta = meta_from_records(arms, pool, balanced_dir=os.path.dirname(os.path.abspath(manifest)))   # 🆕 C2(L147)：给 OT-*.xml 出口(在 manifest 目录=balanced_dir)
    if meta is not None:
        print("（✅ 用 record 自带 scenario_type/file·零位置推导·无错位洞·L146）")
    else:
        meta = build_scenario_meta(manifest, pool)   # 旧 run 无 scenario_type → 位置法（OT 目录默认=manifest 同目录）
        # 🔴 对齐硬断言（【基数+范围】级·非成员级·L146 诚实标限）：仅位置法回退时需要。
        #   ⚠️ 残余假阴性(对抗审 a17b06e0)：eval 丢的文件≠分析期缺的文件但【数量恰好相等】→ 断言放行仍错位一格。
        #   彻底堵洞=用新 run（record 带 scenario_type，走上面优先路径）。
        observed = {e.get("scenario_idx") for rows in arms.values() for r in rows for e in r if e.get("scenario_idx") is not None}
        if observed:
            omin, omax, od = min(observed), max(observed), len(observed)
            if omax >= len(meta) or od != len(meta):
                sys.exit(
                    f"🔴 对齐失败·中止：jsonl scenario_idx 覆盖 [{omin},{omax}] distinct={od} ≠ meta 长度 {len(meta)}。\n"
                    f"   ① 该 jsonl 来自不同 manifest / strided run；② eval 期下载失败与本地缓存不一致；③ 错的场景池。\n"
                    f"   本断言仅【基数+范围】级；等数不同成员错位仍可穿透 → 正解=用新 run(record 自带 scenario_type)。")
    n = len(meta)
    unknown = sum(1 for m in meta if m['tier'] == "未知")
    print(f"测试集 {n} 场景 · 类型分布:", {t: sum(1 for m in meta if m['type']==t) for t in ('head_on','crossing','overtaking')})
    print(f"难度分布:", {nm: sum(1 for m in meta if m['tier']==nm) for nm,_,_ in DIFF_TIERS}, f"| 未知(缺CPA)={unknown}")
    if unknown:
        print(f"  ⚠️ {unknown} 个场景难度=未知（文件缺失/CPA 算失败）→ 已计入类型分层与总、但不在难度分层出现（覆盖面缺口·勿当'每档赢'覆盖全集）")
    all_idx = list(range(n))
    for base, rows in arms.items():
        print("\n" + "="*70); print(f"臂: {base}  ({len(rows)} 种子)")
        # 总
        print(f"  【总】", {k: (round(v,2) if isinstance(v,float) else v) for k,v in agg_group(rows, meta, all_idx).items()})
        # 分类型
        for typ in ("head_on", "crossing", "overtaking"):
            idxs = [i for i in all_idx if meta[i]["type"]==typ]
            if idxs: print(f"  【{typ:9}】", {k:(round(v,2) if isinstance(v,float) else v) for k,v in agg_group(rows, meta, idxs).items()})
        # 分难度
        for nm,_,_ in DIFF_TIERS:
            idxs = [i for i in all_idx if meta[i]["tier"]==nm]
            if idxs: print(f"  【{nm:11}】", {k:(round(v,2) if isinstance(v,float) else v) for k,v in agg_group(rows, meta, idxs).items()})

if __name__ == "__main__":
    main()
