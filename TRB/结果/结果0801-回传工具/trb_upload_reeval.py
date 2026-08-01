#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""重评 + 轨迹产物：盘点 + 分类 + 上传（先只盘点，加 --go 才真的动文件）

与同目录的 `trb_upload.py` 分工：
    trb_upload.py         【训练期】产物（F240* 的 progress.json / table3 / jsonl / log）
    trb_upload_reeval.py  【评估期】产物（正式-末段 / 正式-最佳 / 正式-轨迹全集 / _best.json / 各种 log）← 本文件

用法（Mac 上，两行）：
    cd ~/TRB-2027-ContinuesPPO && git pull origin main
    python3 TRB/结果/结果0801-回传工具/trb_upload_reeval.py --go

不加 --go 只做只读盘点，一个字节都不改。想指定源目录就当参数传：
    python3 TRB/结果/结果0801-回传工具/trb_upload_reeval.py ~/Downloads/结果-正式重评+轨迹 --go

═══ 它防住的坑（都是本项目踩过的）═══════════════════════════════════════════
  ① 模型权重（zip/pkl/pth）几百 MB 进 git —— 一律跳过，只搬 json/jsonl/txt/log/csv/md
  ② 单文件 > 90 MB —— GitHub 硬上限 100 MB，超了整次 push 会被拒；先拦下来、单独报
  ③ `.gitignore` 静默吞文件（`03` L243-续43 B）—— 提交前用 `git status --ignored` 单独列出
  ④ 命令里夹中文注释被 zsh 当参数（`03` L243-续43 C）—— 全部逻辑在脚本里，用户只敲两行
  ⑤ 轨迹文件混进重评目录（`03` L243-续8 D 线 R4）—— 分类时 `*_traj.json` 单独归口
  ⑥ 传上来的重评目录其实是残的 —— 复制前先自查 `g*.json` 的 strict 键长度与组间一致性、
     `all.json` 的臂数，缺了当场报，别等出表时才发现
"""
import collections
import json
import os
import re
import shlex
import shutil
import subprocess
import sys

HOME = os.path.expanduser("~")
#: 仓库不在默认位置时用 `TRB_REPO=/你的路径` 覆盖（本脚本的自测也走这个口子）
REPO = os.environ.get("TRB_REPO") or os.path.join(HOME, "TRB-2027-ContinuesPPO")
DEST = os.path.join(REPO, "TRB", "Paper", "正式实验", "02_重评产物")

KEEP_EXT = {".json", ".jsonl", ".txt", ".log", ".csv", ".md"}
DROP_EXT = {".zip", ".pkl", ".pth", ".pt", ".npz", ".npy", ".ckpt", ".tgz", ".tar", ".gz"}
SKIP_DIR = {".git", "Library", "node_modules", "__pycache__", ".Trash", "Applications"}
BIG_MB = 90.0                      # 单文件红线：GitHub 硬上限 100 MB

#: 分类规则（顺序有意义：轨迹先判，否则 "正式-轨迹全集" 里的 g0.json 会被当成重评那两趟）
BUCKETS = [
    ("正式-轨迹全集", ("轨迹", "traj")),
    ("正式-末段",     ("末段", "formal_last", "-last")),
    ("正式-最佳",     ("最佳", "formal_best", "-best")),
]
OTHER = "_其他"

REPORT = []


def P(*a):
    line = " ".join(str(x) for x in a)
    print(line)
    REPORT.append(line)


def sh(cmd, cwd=None):
    p = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    return p.returncode, p.stdout.strip(), p.stderr.strip()


def current_branch():
    """在**当前分支**上收发，别写死 main —— Claude 的改动可能在 claude/* 分支上。"""
    rc, out, _ = sh("git rev-parse --abbrev-ref HEAD", cwd=REPO)
    return out.strip() if rc == 0 and out.strip() and out.strip() != "HEAD" else "main"


def walk(root, maxdepth=6):
    root = os.path.abspath(root)
    base = root.rstrip("/").count("/")
    for dirpath, dirnames, filenames in os.walk(root):
        if dirpath.count("/") - base >= maxdepth:
            dirnames[:] = []
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIR and not d.startswith(".")]
        if os.path.abspath(dirpath).startswith(os.path.abspath(REPO)):
            dirnames[:] = []
            continue
        for f in filenames:
            if not f.startswith("."):
                yield os.path.join(dirpath, f)


def find_sources(given):
    """没给路径就自己找：Downloads / Desktop / Documents 下名字像重评产物的目录。"""
    if given:
        return [os.path.abspath(g) for g in given]
    hits = []
    for top in ("Downloads", "Desktop", "Documents"):
        d = os.path.join(HOME, top)
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            p = os.path.join(d, name)
            if not os.path.isdir(p) or name.startswith("."):
                continue
            if ("重评" in name or "轨迹" in name or "reeval" in name.lower()
                    or name.startswith("正式-")):
                hits.append(p)
    return hits


def bucket_of(path, src_root):
    """按【相对源目录的那一段路径】判归属，别拿 ~/Downloads 这种上层目录名去撞。"""
    rel = os.path.relpath(path, src_root).lower()
    name = os.path.basename(path).lower()
    if name.endswith("_traj.json") or name.endswith("_traj.jsonl"):
        return "正式-轨迹全集"
    for b, pats in BUCKETS:
        if any(x.lower() in rel for x in pats):
            #: `_best.json` 是【挑存档的结果文件】，不是"最佳那趟重评"，别混进去
            if b == "正式-最佳" and os.path.basename(path).lower() in ("_best.json", "_best_seg30.json"):
                return OTHER
            return b
    return OTHER


def scan(src_roots):
    """→ {bucket: [(绝对路径, 源根, 字节数)]}, 跳过的权重数与字节数"""
    got = collections.defaultdict(list)
    nweight = wbytes = 0
    for r in src_roots:
        if not os.path.isdir(r):
            P(f"  ⚠️ 不存在，跳过：{r}")
            continue
        for p in walk(r):
            ext = os.path.splitext(p)[1].lower()
            if ext in DROP_EXT:
                nweight += 1
                wbytes += os.path.getsize(p)
                continue
            if ext not in KEEP_EXT:
                continue
            got[bucket_of(p, r)].append((p, r, os.path.getsize(p)))
    return got, nweight, wbytes


def verify_reeval_dir(files, bucket):
    """自查一趟重评是不是完整的：g*.json 的 strict 键长度 / 组间是否逐位相同 / all.json 臂数。

    🔴 轨迹专趟的期望值与另两趟不同：它只跑 TRAJ_SEEDS（默认 s0/s1/s2）⟹ 9 臂 × 3 颗 = 27 条，
       且真正的产物是 `g*_traj.json`。拿 72 去卡它会当场误报。
    """
    is_traj = bucket == "正式-轨迹全集"
    n_expect = 27 if is_traj else 72
    gs = sorted(f for f, _, _ in files
                if re.fullmatch(r"g\d+\.json", os.path.basename(f)))
    alls = [f for f, _, _ in files if os.path.basename(f) == "all.json"]
    ntraj = sum(1 for f, _, _ in files if os.path.basename(f).endswith("_traj.json"))
    if is_traj:
        pre = [f"  轨迹文件 g*_traj.json {ntraj} 个" + ("  ✅" if ntraj else "  🔴 一个都没有，图 6 画不了")]
    else:
        pre = []
    if not gs:
        return pre + ["  ⚠️ 没有 g*.json —— 这一趟可能没跑完，或者只传了一部分"]
    out = list(pre)
    ref = None
    bad = []
    for f in gs:
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception as e:
            out.append(f"  🔴 读不动 {os.path.basename(f)}：{type(e).__name__}")
            continue
        keys = d.get("strict键")
        if ref is None:
            ref = keys
        elif keys != ref:
            bad.append(os.path.basename(f))
    n = len(ref) if ref else 0
    out.append(f"  组文件 {len(gs)} 个 · strict 分母 = {n}"
               + ("  ✅" if n == 600 else f"  🔴 正式实验应当是 600"))
    if bad:
        out.append(f"  🔴 这些组的 strict 键列表与其它组不一致：{bad} ⟹ 分母不同、不可同表")
    if alls:
        try:
            m = json.load(open(alls[0], encoding="utf-8"))
            out.append(f"  all.json 合并得 {len(m)} 条臂"
                       + ("  ✅" if len(m) == n_expect else f"  🔴 期望 {n_expect} 条"))
        except Exception as e:
            out.append(f"  🔴 all.json 读不动：{type(e).__name__}")
    elif not is_traj:
        out.append("  🔴 没有 all.json —— 合并那一步没跑成（看同目录的 g*.log）")
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    go = "--go" in sys.argv
    force_big = "--force-big" in sys.argv

    src_roots = find_sources(args)
    if not src_roots:
        raise SystemExit("🔒 没找到重评产物目录。把目录路径当参数传给我，例如：\n"
                         "   python3 TRB/结果/结果0801-回传工具/trb_upload_reeval.py "
                         "~/Downloads/结果-正式重评+轨迹")
    P("扫描目录：")
    for r in src_roots:
        P("  " + r)

    P("\n══ 1. 盘点 ══")
    got, nweight, wbytes = scan(src_roots)
    if not got:
        raise SystemExit("🔒 一个可搬的产物都没扫到（只认 json/jsonl/txt/log/csv/md）")
    total = 0
    for b in [x[0] for x in BUCKETS] + [OTHER]:
        files = got.get(b) or []
        if not files:
            continue
        mb = sum(s for _, _, s in files) / 1048576
        total += mb
        P(f"\n  【{b}】{len(files)} 个文件 · {mb:.1f} MB")
        for line in (verify_reeval_dir(files, b) if b != OTHER else []):
            P(line)
        if b == OTHER:
            for f, _, s in sorted(files)[:20]:
                P(f"    {os.path.basename(f):<40} {s/1024:.0f} KB")
            if len(files) > 20:
                P(f"    …… 还有 {len(files)-20} 个")
    P(f"\n  合计要搬 {total:.1f} MB · 跳过权重/压缩包 {nweight} 个（{wbytes/1048576:.0f} MB，不进 git）")

    big = [(f, s) for fs in got.values() for f, _, s in fs if s / 1048576 > BIG_MB]
    if big:
        P(f"\n  🔴 有 {len(big)} 个文件超过 {BIG_MB:.0f} MB（GitHub 硬上限 100 MB）：")
        for f, s in big[:10]:
            P(f"     {s/1048576:.0f} MB  {f}")
        if not force_big:
            P("     ⟹ 默认【跳过】这些文件，其余照传。确定要传就加 --force-big（很可能被 GitHub 拒）。")

    if not go:
        P("\n（这是只读盘点，什么都没动。确认无误后加 --go 真正上传。）")
        return

    if not os.path.isdir(REPO):
        raise SystemExit(f"🔒 本机没有仓库 {REPO} —— 先 git clone，或者在有仓库的机器上跑本脚本")

    br = current_branch()
    P(f"\n══ 2. 同步仓库（分支 {br}）══")
    rc, out, err = sh(f"git pull origin {shlex.quote(br)}", cwd=REPO)
    P("  " + (out or err).replace("\n", "\n  "))
    if rc != 0:
        raise SystemExit("❌ git pull 没成功，先解决它再跑本脚本")

    P("\n══ 3. 分类复制 ══")
    ncopy = nskip = 0
    nbytes = 0
    for b, files in got.items():
        for f, root, s in files:
            if s / 1048576 > BIG_MB and not force_big:
                nskip += 1
                continue
            rel = os.path.relpath(f, root)
            #: 源目录里已经有 正式-末段/ 这层的，就不要再套一层同名目录
            parts = [x for x in rel.split(os.sep)[:-1]
                     if not any(y.lower() in x.lower() for _, ps in BUCKETS for y in ps)]
            dst = os.path.join(DEST, b, *parts, os.path.basename(f))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(f, dst)
            ncopy += 1
            nbytes += s
    P(f"  复制 {ncopy} 个（{nbytes/1048576:.1f} MB）· 因超大跳过 {nskip} 个")

    P("\n══ 4. 提交前检查：有没有文件被 .gitignore 悄悄挡下 ══")
    rc, out, _ = sh("git status --porcelain --ignored " + shlex.quote(DEST), cwd=REPO)
    ign = [l[3:] for l in out.splitlines() if l.startswith("!!")]
    if ign:
        P(f"  🔴 有 {len(ign)} 个文件被忽略，前 10 个：")
        for x in ign[:10]:
            P("     " + x)
        P("  → 把这份清单发给 Claude，别自己 -f 强加")
    else:
        P("  ✅ 没有文件被忽略")

    P("\n══ 5. 提交并推送 ══")
    sh("git add -A " + shlex.quote(DEST), cwd=REPO)
    rc, out, _ = sh("git diff --cached --name-only", cwd=REPO)
    files = out.splitlines()
    P(f"  真正进入本次提交的文件：{len(files)} 个")
    for x in files[:20]:
        P("     " + x)
    if len(files) > 20:
        P(f"     …… 还有 {len(files)-20} 个")
    if files:
        sh('git commit -m "回传正式实验重评与轨迹产物"', cwd=REPO)
        rc, out, err = sh(f"git push -u origin {shlex.quote(br)}", cwd=REPO)
        P("  " + (out or err).replace("\n", "\n  "))
        if rc != 0:
            P("  🔴 push 没成功。把上面这段发给 Claude。")
    else:
        P("  没有新东西要提交。")

    #: 盘点报告本身也提交，Claude 那边不用你复制粘贴终端输出
    rp = os.path.join(DEST, "_盘点报告_重评.txt")
    os.makedirs(DEST, exist_ok=True)
    open(rp, "w", encoding="utf-8").write("\n".join(REPORT) + "\n")
    sh("git add -A " + shlex.quote(rp), cwd=REPO)
    sh('git commit -m "回传：重评产物盘点报告"', cwd=REPO)
    sh(f"git push -u origin {shlex.quote(br)}", cwd=REPO)
    print(f"\n盘点报告已提交：{rp}")


if __name__ == "__main__":
    main()
