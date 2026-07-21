"""
stratified_stats.py 单元冒烟（L147 复审补测：此前 0 committed 用例）。
覆盖本会话新增/复审焦点：C2 OT-tier balanced_dir 出口（_resolve_scenario_file）+ C3 arm_base 只裁末尾 _s<digits>
+ meta_from_records 类型维键控/旧 run 回退/混集 type 冲突 fail-fast。
跑：/opt/miniconda3/envs/trb/bin/python -B 代码/tests/test_stratified_stats.py
（纯逻辑 + 临时文件·不需真场景池；CPA 缺文件降级 tier='未知'不影响 type 维断言。）
"""
import sys, os, tempfile, shutil

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import stratified_stats as ss

_fail = 0


def ok(name, cond):
    global _fail
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        _fail += 1


# ---- C3: arm_base 只裁末尾 _s<digits>（不误裁名字里的 _s...·L146 bug 修守护）----
ok("C3 arm_base 裁末尾 _s<digits>：discStdW0_s0→discStdW0", ss.arm_base("discStdW0_s0") == "discStdW0")
ok("C3 arm_base 多位种子：floorBig_s10→floorBig", ss.arm_base("floorBig_s10") == "floorBig")
ok("C3 arm_base 不误裁名字含 _s：baseW0_shield→原样(非 baseW0)", ss.arm_base("baseW0_shield") == "baseW0_shield")
ok("C3 arm_base 无种子后缀→原样：nofix→nofix", ss.arm_base("nofix") == "nofix")

# ---- C2: _resolve_scenario_file 先 pool 再 balanced_dir（OT-*.xml 只在 balanced_dir）----
_pd = tempfile.mkdtemp()
_bd = tempfile.mkdtemp()
open(os.path.join(_pd, "T-1.xml"), "w").write("x")     # 内容无所谓·_resolve 只查 exists
open(os.path.join(_bd, "OT-9.xml"), "w").write("x")
ok("C2 _resolve 先 pool 命中 T-1", os.path.basename(ss._resolve_scenario_file("T-1.xml", _pd, _bd) or "") == "T-1.xml")
ok("C2 _resolve pool 无 → balanced_dir 命中 OT-9", os.path.basename(ss._resolve_scenario_file("OT-9.xml", _pd, _bd) or "") == "OT-9.xml")
ok("C2 _resolve 无 balanced_dir → OT 找不到（=修前 tier 未知根因）", ss._resolve_scenario_file("OT-9.xml", _pd) is None)
ok("C2 _resolve 缺文件/空名 → None（不崩）",
   ss._resolve_scenario_file("nope.xml", _pd, _bd) is None and ss._resolve_scenario_file(None, _pd) is None)

# ---- meta_from_records：类型维键控（非位置）+ 旧 run 回退 None + 同 idx 跨臂 type 冲突 fail-fast ----
_arms_new = {"a": [[{"scenario_idx": 0, "scenario_type": "head_on", "scenario_file": "T-1.xml"},
                    {"scenario_idx": 1, "scenario_type": "overtaking", "scenario_file": "OT-9.xml"}]]}
_meta = ss.meta_from_records(_arms_new, _pd, balanced_dir=_bd)
ok("meta_from_records 类型由 record 键控（idx0=head_on / idx1=overtaking·非靠位置）",
   _meta is not None and _meta[0]["type"] == "head_on" and _meta[1]["type"] == "overtaking")
_arms_old = {"a": [[{"scenario_idx": 0, "reached": True}]]}     # 无 scenario_type=旧 run
ok("meta_from_records 旧 run(无 scenario_type)→返回 None（回退位置法）", ss.meta_from_records(_arms_old, _pd) is None)
_arms_conflict = {"a": [[{"scenario_idx": 0, "scenario_type": "head_on"}]],
                  "b": [[{"scenario_idx": 0, "scenario_type": "crossing"}]]}   # 同 idx 冲突 type


def _raises_sysexit():
    try:
        ss.meta_from_records(_arms_conflict, _pd)
        return False
    except SystemExit:
        return True
    except Exception:
        return False


ok("meta_from_records 同 idx 跨臂 type 冲突 → SystemExit（防混不同测试集）", _raises_sysexit())

shutil.rmtree(_pd, ignore_errors=True)
shutil.rmtree(_bd, ignore_errors=True)
print()
print("✅ 全部 PASS" if _fail == 0 else f"❌ {_fail} 项 FAIL")
sys.exit(1 if _fail else 0)
