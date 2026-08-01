#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TRB 训练产物：盘点 + 上传（先只盘点，加 --go 才真的动文件）

用法：
    python3 ~/trb_upload.py                 只盘点，一个字节都不改
    python3 ~/trb_upload.py --go            盘点 + 复制 + 提交 + 推送
    python3 ~/trb_upload.py /路径A /路径B    指定要扫的目录（默认扫 Downloads/Desktop/Documents）

设计上防住的三个坑（都是本项目踩过的）：
  ① 只搬轻量证据（json/jsonl/txt/log/csv），模型权重（zip/pkl/pth）一律不搬 —— 几百 MB 不进 git
  ② 覆盖保护：同名 progress.json 只有在【段数更多】时才覆盖，防"半截档盖掉完整档"
  ③ 提交前把【被 .gitignore 悄悄挡下】的文件单独列出来 —— git add 对被忽略的文件不报错
"""
import os, re, sys, json, shlex, shutil, subprocess, collections
# 🔴 2026-08-01 later-3 修：原来 git 的路径参数用 `json.dumps(DEST)` 包引号，而 DEST 里带中文，
#   json 默认把非 ASCII 转义成 \uXXXX ⟹ 传给 shell 的是字面量 "TRB/Paper/正式…"，
#   `git add` 匹配不到任何文件、还不报错 ⟹ **复制完了却一个文件都没提交**。改用 shlex.quote。

HOME = os.path.expanduser("~")
REPO = os.path.join(HOME, "TRB-2027-ContinuesPPO")
DEST = os.path.join(REPO, "TRB", "Paper", "正式实验", "01_训练产物")

KEEP_EXT = {".json", ".jsonl", ".txt", ".log", ".csv"}
DROP_EXT = {".zip", ".pkl", ".pth", ".pt", ".npz", ".npy", ".ckpt"}
SKIP_DIR = {".git", "Library", "node_modules", "__pycache__", ".Trash", "Applications"}
MAXDEPTH = 7

ARMS = ["ours", "disc", "base", "rr", "uns", "ush", "ab0", "abB", "abG"]
PAT = re.compile(r"F240([A-Za-z0-9]+?)Ppo[Ss](\d+)")


REPORT_LINES = []


def P(*a):
    line = " ".join(str(x) for x in a)
    print(line)
    REPORT_LINES.append(line)


def sh(cmd, cwd=None):
    p = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    return p.returncode, p.stdout.strip(), p.stderr.strip()


def walk(root, maxdepth=MAXDEPTH, skip_repo=False):
    root = os.path.abspath(root)
    base = root.rstrip("/").count("/")
    for dirpath, dirnames, filenames in os.walk(root):
        if dirpath.count("/") - base >= maxdepth:
            dirnames[:] = []
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIR and not d.startswith(".")]
        if skip_repo and os.path.abspath(dirpath).startswith(os.path.abspath(REPO)):
            dirnames[:] = []
            continue
        for f in filenames:
            yield os.path.join(dirpath, f)


def is_artifact(name):
    return ("F240" in name or "f2027" in name) and os.path.splitext(name)[1] in (KEEP_EXT | DROP_EXT)


def seg_of(path):
    """主存档 progress.json 里的已完成段数；读不到返回 None"""
    try:
        d = json.load(open(path, encoding="utf-8"))
        return int(d.get("seg_done", -1)) + 1
    except Exception:
        return None


def scan(roots, label):
    """→ {(arm,seed): {'seg':n, 'seg_files':n, 'table3':bool, 'jsonl':bool, 'log':bool, 'src':dir}}"""
    inv = collections.defaultdict(lambda: {"seg": 0, "seg_files": 0, "table3": False,
                                           "jsonl": False, "log": False, "src": set()})
    files = []
    for r in roots:
        if not os.path.isdir(r):
            continue
        for p in walk(r, skip_repo=(label == "本机")):
            n = os.path.basename(p)
            if not is_artifact(n):
                continue
            files.append(p)
            m = PAT.search(n)
            if not m:
                continue
            arm, seed = m.group(1), int(m.group(2))
            if arm not in ARMS:
                continue
            k = (arm, seed)
            e = inv[k]
            e["src"].add(os.path.dirname(p))
            if n.endswith(".progress.json"):
                if "@s" in n:
                    e["seg_files"] += 1
                else:
                    s = seg_of(p)
                    if s and s > e["seg"]:
                        e["seg"] = s
            elif n.startswith("table3_"):
                e["table3"] = True
            elif n.startswith("step4e_partial_"):
                e["jsonl"] = True
            elif n.endswith(".log"):
                e["log"] = True
    P(f"  {label}：扫到 {len(files)} 个产物文件，{len(inv)} 个 (臂,种子)")
    return inv


def table(inv, title):
    P(f"\n===== {title} =====")
    P(f"  {'臂':<6}{'种子':<5}{'段数':<6}{'分段副本':<9}{'table3':<8}{'jsonl':<7}{'log'}")
    seeds = sorted({s for _, s in inv})
    for arm in ARMS:
        for s in seeds:
            e = inv.get((arm, s))
            if not e:
                continue
            flag = "🔴" if e["seg"] < 20 else "  "
            P(f"{flag}{arm:<6}{s:<5}{e['seg']:<6}{e['seg_files']:<9}"
                  f"{'有' if e['table3'] else '—':<8}{'有' if e['jsonl'] else '—':<7}"
                  f"{'有' if e['log'] else '—'}")
    full = sum(1 for e in inv.values() if e["seg"] >= 20)
    P(f"  —— 满 20 段 {full} 条 / 共 {len(inv)} 条")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    go = "--go" in sys.argv
    roots = args or [d for d in
                     [os.path.join(HOME, x) for x in ("Downloads", "Desktop", "Documents")]
                     + ["/root/trb/结果", os.path.join(HOME, "trb", "结果")]
                     if os.path.isdir(d)]
    P("扫描目录：" + " ".join(roots))

    if not os.path.isdir(REPO):
        P(f"\nⓘ 本机没有仓库（{REPO}）—— 判定为【服务器】，只做盘点。")
        src = scan(roots, "本机")
        table(src, "这台机器上有什么")
        P("\n把上面整张表发给 Claude；要上传就先把这些目录下载到 Mac 再跑本脚本。")
        return

    P("══ 0. 同步仓库（必须成功，否则 .gitignore 的修复不生效）══")
    rc, out, err = sh("git pull origin main", cwd=REPO)
    P("  " + (out or err).replace("\n", "\n  "))
    if rc != 0:
        sys.exit("❌ git pull 没成功，先解决它再跑本脚本")

    P("\n══ 1. 盘点 ══")
    src = scan(roots, "本机")
    rep = scan([DEST], "仓库")
    nolog = not any(e["log"] for e in src.values())
    if nolog:
        P("  ⚠️ 本机这些目录里【一个训练日志都没有】——日志在服务器 `结果/*.log`，值得一并下载")
    table(src, "本机有什么")
    table(rep, "仓库已有什么")

    P("\n===== 差异（本机比仓库多的才需要上传）=====")
    todo = []
    for k, e in sorted(src.items()):
        r = rep.get(k)
        why = []
        if not r:
            why.append("仓库完全没有")
        else:
            if e["seg"] > r["seg"]:
                why.append(f"段数更多 {e['seg']}>{r['seg']}")
            if e["table3"] and not r["table3"]:
                why.append("多 table3")
            if e["jsonl"] and not r["jsonl"]:
                why.append("多 jsonl")
            if e["log"] and not r["log"]:
                why.append("多 log")
        if why:
            todo.append((k, why))
            P(f"  🆕 {k[0]} s{k[1]}: " + " / ".join(why))
    if not todo:
        P("  ✅ 本机没有仓库里缺的东西 —— 产物已经齐了，不用再传")

    if not go:
        P("\n（这是只读盘点。确认无误后加 --go 真正上传：python3 ~/trb_upload.py --go）")
        return

    P("\n══ 2. 复制（只搬轻量证据，权重一律不搬）══")
    ncopy = nskip_w = nskip_old = 0
    nbytes = 0
    for r in roots:
        if not os.path.isdir(r):
            continue
        for p in walk(r, skip_repo=True):
            n = os.path.basename(p)
            if not is_artifact(n):
                continue
            ext = os.path.splitext(n)[1]
            if ext in DROP_EXT:
                nskip_w += 1
                continue
            # 目标子目录名 = 源路径里第一个含 F240 产物的那层目录名
            parts = os.path.abspath(p).split(os.sep)
            top = next((x for x in parts if "结果" in x or "auto" in x or "result" in x.lower()), None)
            sub = top or "结果auto-本机"
            rel = p.split(sub, 1)[1].lstrip(os.sep) if sub in p else n
            dst = os.path.join(DEST, sub, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if os.path.exists(dst) and n.endswith(".progress.json") and "@s" not in n:
                a, b = seg_of(p), seg_of(dst)
                if a is not None and b is not None and a <= b:
                    nskip_old += 1
                    continue
            shutil.copy2(p, dst)
            ncopy += 1
            nbytes += os.path.getsize(dst)
    P(f"  复制 {ncopy} 个（{nbytes/1048576:.1f} MB）· 跳过权重 {nskip_w} 个 · 跳过更旧的存档 {nskip_old} 个")

    P("\n══ 3. 提交前检查：有没有文件被 .gitignore 悄悄挡下 ══")
    rc, out, _ = sh("git status --porcelain --ignored " + shlex.quote(DEST), cwd=REPO)
    ign = [l[3:] for l in out.splitlines() if l.startswith("!!")]
    if ign:
        P(f"  🔴 有 {len(ign)} 个文件被忽略，前 10 个：")
        for x in ign[:10]:
            P("     " + x)
        P("  → 把这份清单发给 Claude，别自己 -f 强加")
    else:
        P("  ✅ 没有文件被忽略")

    P("\n══ 4. 提交并推送 ══")
    sh("git add -A " + shlex.quote(DEST), cwd=REPO)
    rc, out, _ = sh("git diff --cached --name-only", cwd=REPO)
    files = out.splitlines()
    P(f"  真正进入本次提交的文件：{len(files)} 个")
    for x in files[:25]:
        P("     " + x)
    if len(files) > 25:
        P(f"     …… 还有 {len(files)-25} 个")
    if not files:
        P("  没有新东西要提交，结束。")
        return
    sh('git commit -m "回传训练产物（含训练日志）"', cwd=REPO)
    rc, out, err = sh("git push origin main", cwd=REPO)
    P("  " + (out or err).replace("\n", "\n  "))
    P("  ✅ 完成" if rc == 0 else "  🔴 push 失败，把上面这段发给 Claude")


def write_report():
    import socket, time
    out = os.path.join(REPO, "TRB", "结果", "结果0801-回传工具", "_盘点报告.txt")
    try:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write("机器：%s\n时间：%s\n\n" % (socket.gethostname(),
                    time.strftime("%Y-%m-%d %H:%M:%S")))
            f.write("\n".join(REPORT_LINES) + "\n")
        print("\n📄 盘点报告已写到 TRB/结果/结果0801-回传工具/_盘点报告.txt")
        return out
    except Exception as e:
        print("\n⚠️ 报告写不出来：%s" % e)
        return None


if __name__ == "__main__":
    try:
        main()
    finally:
        if os.path.isdir(REPO):
            rp = write_report()
            if rp and "--go" in sys.argv:
                sh("git add -A " + shlex.quote(rp), cwd=REPO)
                rc, out, _ = sh("git diff --cached --name-only", cwd=REPO)
                if out.strip():
                    sh('git commit -m "回传：盘点报告"', cwd=REPO)
                    rc, o, e = sh("git push origin main", cwd=REPO)
                    print("  推送报告：" + ("✅ 成功" if rc == 0 else "🔴 失败\n  " + (o or e)))
