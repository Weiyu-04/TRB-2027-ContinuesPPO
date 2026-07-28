# -*- coding: utf-8 -*-
"""`STEP4E_KEEP_SEGMENTS` 回归（`03` L236-D②·user 2026-07-28 拍板"以后所有训练都开"）。

被测的东西 = 分段存档保留：主存档是覆盖式的（每段盖掉上一段），开了这个开关就把每一段
另存一份到 `checkpoints/segments/`，以后想换"最好存档"口径 / 查崩溃前后，不必重训。

三条必须守住（按重要性排）：
  ① **默认关 ⟹ 一个字节都不多写**（本项目所有旋钮的铁律：默认档逐位等价现状）。
  ② **副本绝不与主存档同层** —— `reeval_official.discover_ckpts()` 是 `glob(<dir>/*.zip)` + 有同名
     `_vecnorm.pkl` 就收、**非递归**；同层就会被【静默】收成新臂，56 条臂的表当场变几百条。
  ③ **mtime 必须原样保留** —— sidecar 里 `ckpt_fingerprint` 记的是 (zip_mtime, zip_size)，对不上
     重评的【锚点自检】就整条跳过，而锚点是防"评估配错→静默给错数"的关键防线（`03` L192）。

跑法：  python3 -B 代码/tests/test_keep_segments.py       （零依赖：不 import run_step4e，避免拖 torch）
"""
import json
import os
import shutil
import sys
import tempfile

_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _CODE)

N_PASS = N_FAIL = 0


def check(desc, cond):
    global N_PASS, N_FAIL
    if cond:
        N_PASS += 1
        print(f"  ✅ {desc}")
    else:
        N_FAIL += 1
        print(f"  ❌ {desc}")


def _load_archive_fn():
    """只把 `_archive_segment` 抠出来单独执行 —— import run_step4e 会拉起 torch/sb3 全家桶且有
    一堆 fail-fast 环境闸，测一个纯文件操作函数不值得付那个代价。"""
    src = open(os.path.join(_CODE, "run_step4e.py"), encoding="utf-8").read()
    start = src.index("def _archive_segment(")
    end = src.index("def save_segment_checkpoint(")
    ns = {"os": os, "shutil": shutil}
    exec(compile(src[start:end], "<_archive_segment>", "exec"), ns)
    return ns["_archive_segment"]


def _mk_ckpt(d, name, mtime):
    base = os.path.join(d, name)
    for suf, body in ((".zip", b"PK\x03\x04fake-model"),
                      ("_vecnorm.pkl", b"fake-vecnorm"),
                      (".progress.json", json.dumps({"seg_done": 3}).encode())):
        open(base + suf, "wb").write(body)
        os.utime(base + suf, (mtime, mtime))
    return base


def main():
    archive = _load_archive_fn()
    print("===== STEP4E_KEEP_SEGMENTS 回归 =====")

    print("\n【① 语义：环境变量怎么解析】")
    src = open(os.path.join(_CODE, "run_step4e.py"), encoding="utf-8").read()
    ln = [l for l in src.splitlines() if l.startswith("_KEEP_SEGMENTS =")]
    check("_KEEP_SEGMENTS 只有一处定义", len(ln) == 1)
    check("默认值是 '0'（= 关 = 逐位等价现状）", '"STEP4E_KEEP_SEGMENTS", "0"' in ln[0])
    check("接受 1/true/yes/on 四种写法", all(v in ln[0] for v in ('"1"', '"true"', '"yes"', '"on"')))
    # 只在 save_segment_checkpoint 的函数体里核（`def _archive_segment(base, seg_done)` 那行也含同样的字串，
    # 全文 index 会误命中定义处 ⟹ 必须先切出函数体）
    body = src[src.index("def save_segment_checkpoint("):src.index("def _run_arm_") if "def _run_arm_" in src
               else src.index("def save_segment_checkpoint(") + 3000]
    check("关的时候整块跳过（调用被 if _KEEP_SEGMENTS 守着）",
          "if _KEEP_SEGMENTS:" in body and body.index("if _KEEP_SEGMENTS:") < body.index("_archive_segment(base, seg_done)"))
    check("副本失败被 try/except 兜住（存盘失败不崩训练·同 :540/:697 纪律）",
          "except Exception as _e:" in body[body.index("if _KEEP_SEGMENTS:"):])
    check("归档发生在 write_progress **之后**（主存档先提交·副本绝不抢在前面）",
          body.index("curves=curves, seg_per=seg_per)") < body.index("if _KEEP_SEGMENTS:"))
    check("keep_segments 进 run_config（provenance 自描述）", '"keep_segments": _KEEP_SEGMENTS' in src)
    check("keep_segments **不进** config_sig（它一个权重都不改·进了会让老 ckpt 续训从 0 重启）",
          '"keep_segments"' not in src[src.index('"act_dist": _ACT_DIST, "gw_entry": _GW_ENTRY,   # 🆕 L230：动作分布档'):][:800])

    print("\n【② 真跑一次：三件套是否完整落到子目录、mtime 是否保住】")
    d = tempfile.mkdtemp(prefix="ckseg_")
    MT = 1600000000.0
    base = _mk_ckpt(d, "Continuous-safe_s3_C231bothPpoS3", MT)
    dst = archive(base, 7)
    seg_dir = os.path.join(d, "segments")
    check("副本落在 segments/ 子目录里", os.path.dirname(dst) == seg_dir)
    check("副本名带段号 @s07（两位·可排序）", os.path.basename(dst).endswith("@s07"))
    for suf in (".zip", "_vecnorm.pkl", ".progress.json"):
        check(f"三件套齐全：{suf}", os.path.exists(dst + suf))
    check("zip 内容逐字节相同", open(dst + ".zip", "rb").read() == open(base + ".zip", "rb").read())
    check("🔴 zip 的 mtime 原样保留（锚点自检要用 ckpt_fingerprint 对 mtime）",
          abs(os.stat(dst + ".zip").st_mtime - MT) < 1e-6)
    check("没留下 .tmp 残骸", not any(f.endswith(".tmp") for f in os.listdir(seg_dir)))

    print("\n【③ 🔴 承重：副本不会被 reeval 的自动发现扫到】")
    import glob
    top = [z for z in glob.glob(os.path.join(d, "*.zip")) if os.path.exists(z[:-4] + "_vecnorm.pkl")]
    check("主目录同层只发现得到 1 个存档（= 主存档本身，副本扫不到）", len(top) == 1)
    check("  且它就是主存档", top and top[0] == base + ".zip")
    check("run_reeval_all.sh 的 find 口径（*/checkpoints/<臂名>.zip）也匹配不到副本",
          "@s07" not in os.path.basename(base))
    # 直接核 reeval 源码里那行 glob 确实是非递归的（改成 ** 或 recursive=True 就会开始扫子目录 ⟹ 这条会红）
    _ro = open(os.path.join(_CODE, "tests", "reeval_official.py"), encoding="utf-8").read()
    _fn = _ro[_ro.index("def discover_ckpts("):_ro.index("def resolve_ckpts(")]
    check("🔴 reeval 的自动发现仍是非递归 glob（一旦有人改成递归，副本就会被静默收成新臂）",
          'glob.glob(os.path.join(d, "*.zip"))' in _fn and "recursive" not in _fn and "**" not in _fn)

    print("\n【④ 多段累积：段与段互不覆盖】")
    for seg in (1, 2, 10):
        _mk_ckpt(d, "Continuous-safe_s3_C231bothPpoS3", MT + seg)
        archive(base, seg)
    zips = sorted(os.path.basename(z) for z in glob.glob(os.path.join(seg_dir, "*.zip")))
    check("4 段各留一份（@s01/@s02/@s07/@s10），不互相覆盖", len(zips) == 4)
    check("  段号补零 ⟹ 文件名字典序 = 训练顺序", zips == sorted(zips) and zips[0].endswith("@s01.zip"))
    shutil.rmtree(d, ignore_errors=True)

    print(f"\n===== {N_PASS} PASS · {N_FAIL} FAIL =====")
    return 1 if N_FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
