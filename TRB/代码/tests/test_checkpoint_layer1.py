"""Layer-1 每段 checkpoint 离线冒烟（L80-续4 ⑥ + S5 范式·不需 commonocean/场景）。
覆盖新增 helper：_atomic_save（原子写+tmp 非空后缀+覆盖）/ write_progress（commit barrier+指纹+config_sig）/
save_checkpoint（原子化）/ save_segment_checkpoint（先 ckpt 后 progress 的提交顺序）。
⚠️ 不覆盖【每段存 vs 不存 bit-identical】（需场景+训练=服务器 smoke·见 04 §诊断）；本套只验存盘机制正确性。
跑：python3 tests/test_checkpoint_layer1.py
"""
import os
import sys
import json
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import run_step4e as R

_n = 0
_fail = 0


def ok(desc, cond):
    global _n, _fail
    _n += 1
    if cond:
        print(f"[PASS] {desc}")
    else:
        _fail += 1
        print(f"[FAIL] {desc}")


# ---- ① _atomic_save：写到 .<pid>.tmp（非空后缀）→ os.replace → final；不留 tmp ----
with tempfile.TemporaryDirectory() as d:
    final = os.path.join(d, "x.zip")
    seen_paths = []

    def _fake_save(p):
        seen_paths.append(p)
        with open(p, "w") as f:
            f.write("DATA-v1")

    R._atomic_save(_fake_save, final)
    ok("① _atomic_save 写出 final 文件", os.path.exists(final))
    ok("① final 内容正确", open(final).read() == "DATA-v1")
    ok("① save_fn 收到的是 tmp 路径（非 final·证 tmp 间接）", seen_paths[0] != final)
    ok("① tmp 后缀非空(.tmp·防 sb3 静默追加 .zip)", os.path.splitext(seen_paths[0])[1] == ".tmp")
    ok("① tmp 路径基于 final（同目录·os.replace 原子前提）", os.path.dirname(seen_paths[0]) == d)
    ok("① 不留孤儿 tmp", not os.path.exists(seen_paths[0]))

    # ② 覆盖：第二次写覆盖旧 final（last-writer-wins·原子）
    def _fake_save2(p):
        with open(p, "w") as f:
            f.write("DATA-v2")
    R._atomic_save(_fake_save2, final)
    ok("② 覆盖最新：final 更新为新内容", open(final).read() == "DATA-v2")

# ---- ③ write_progress：commit barrier·记 ckpt 指纹(mtime+size)+config_sig+全字段 ----
with tempfile.TemporaryDirectory() as d:
    base = os.path.join(d, "Discrete-safe_s1_test")
    with open(base + ".zip", "wb") as f:
        f.write(b"X" * 1234)                       # 假 ckpt（量指纹用）
    sig = {"kind": "shielded", "total_steps": 3000000, "n_seg": 6, "algo": "MaskablePPO"}
    trend = [{"step": 500000, "到达率%": 12.3}]
    R.write_progress(base, name="Discrete-safe", kind="shielded", weight=1.0, seed=1,
                     seg_done=0, num_timesteps=500000, total_steps=3000000, n_seg=6,
                     trend=trend, config_sig=sig)
    pp = base + ".progress.json"
    ok("③ progress.json 写出", os.path.exists(pp))
    prog = json.load(open(pp))
    ok("③ seg_done/num_timesteps 正确", prog["seg_done"] == 0 and prog["num_timesteps"] == 500000)
    ok("③ total_steps/n_seg 正确", prog["total_steps"] == 3000000 and prog["n_seg"] == 6)
    ok("③ config_sig 原样落盘", prog["config_sig"] == sig)
    ok("③ trend 落盘", prog["trend"] == trend)
    ok("③ ckpt 指纹 size 与真 ckpt 一致(1234)", prog["ckpt_fingerprint"]["zip_size"] == 1234)
    ok("③ ckpt 指纹含 mtime", "zip_mtime" in prog["ckpt_fingerprint"])
    ok("③ party/seed 落盘", prog["party"] == "Discrete-safe" and prog["seed"] == 1)

    # ③b ckpt 不存在 → 指纹 None（fail-safe·不崩）
    base2 = os.path.join(d, "nockpt_test")
    R.write_progress(base2, name="X", kind="shielded", weight=1.0, seed=0,
                     seg_done=0, num_timesteps=1, total_steps=2, n_seg=1, trend=[], config_sig={})
    ok("③b ckpt 缺失时指纹=None（不崩）", json.load(open(base2 + ".progress.json"))["ckpt_fingerprint"] is None)

# ---- ④ save_checkpoint 原子化：fake model + 非 VecNormalize venv（跳 venv.save）----
with tempfile.TemporaryDirectory() as d:
    class _FakeModel:
        def save(self, p):
            with open(p, "w") as f:
                f.write("MODEL")

    base = R.save_checkpoint(_FakeModel(), object(), "Discrete-safe", 2, d)   # venv=object() 非 VecNormalize → 跳 .pkl
    ok("④ save_checkpoint 返回 base（无后缀）", base.endswith("Discrete-safe_s2" + R._TAG))
    ok("④ model 原子写到 base.zip", os.path.exists(base + ".zip") and open(base + ".zip").read() == "MODEL")
    ok("④ 非 VecNormalize venv → 不写 _vecnorm.pkl", not os.path.exists(base + "_vecnorm.pkl"))
    ok("④ 不留孤儿 .tmp", not any(x.endswith(".tmp") for x in os.listdir(d)))

# ---- ⑤ save_segment_checkpoint：提交顺序(ckpt 先·progress 后)·progress 指纹指向真 ckpt ----
with tempfile.TemporaryDirectory() as d:
    class _FakeModel:
        def save(self, p):
            with open(p, "wb") as f:
                f.write(b"M" * 999)

    sig = {"kind": "shielded", "n_seg": 6}
    base = R.save_segment_checkpoint(_FakeModel(), object(), "Discrete-safe", "shielded", 1.0, 3, d,
                                     seg_done=2, num_timesteps=1500000, total_steps=3000000,
                                     n_seg=6, trend=[{"step": 1}], config_sig=sig)
    ok("⑤ ckpt.zip 与 progress.json 都在", os.path.exists(base + ".zip") and os.path.exists(base + ".progress.json"))
    prog = json.load(open(base + ".progress.json"))
    ok("⑤ progress 指纹 size 指向真 ckpt(999)", prog["ckpt_fingerprint"]["zip_size"] == 999)
    ok("⑤ progress seg_done/num_timesteps 正确", prog["seg_done"] == 2 and prog["num_timesteps"] == 1500000)
    # 提交顺序：progress.json mtime >= ckpt.zip mtime（progress 最后写=commit 点）
    ok("⑤ 提交顺序：progress 不早于 ckpt（commit barrier）",
       os.stat(base + ".progress.json").st_mtime >= os.stat(base + ".zip").st_mtime)
    ok("⑤ 不留孤儿 .tmp", not any(x.endswith(".tmp") for x in os.listdir(d)))

# ---- ⑥ 增量诊断：curves + seg_per 随每段写入 progress.json（四臂同款·中途可拉取分析）----
with tempfile.TemporaryDirectory() as d:
    class _FakeModel:
        def save(self, p):
            with open(p, "wb") as f:
                f.write(b"M")
    fake_curves = [{"step": 16384, "critic_loss": 0.5, "ep_rew_mean": -100.0},
                   {"step": 32768, "critic_loss": 0.3, "ep_rew_mean": 50.0}]
    fake_per = [{"reached": False, "steps": 170}, {"reached": True, "steps": 53}]
    base = R.save_segment_checkpoint(_FakeModel(), object(), "Continuous-safe", "continuous", 0.0, 1, d,
                                     seg_done=1, num_timesteps=1000000, total_steps=5000000, n_seg=10,
                                     trend=[{"step": 1000000}], config_sig={}, curves=fake_curves, seg_per=fake_per)
    prog = json.load(open(base + ".progress.json"))
    ok("⑥ 增量诊断 curves 随段写入 progress.json", prog["curves"] == fake_curves)
    ok("⑥ 增量诊断 seg_per（逐局）随段写入", prog["seg_per"] == fake_per)
    # 向后兼容：curves/seg_per 缺省=None 不崩
    base2 = R.save_segment_checkpoint(_FakeModel(), object(), "Base", "unshielded", 0.0, 0, d,
                                      seg_done=0, num_timesteps=1, total_steps=2, n_seg=1, trend=[], config_sig={})
    prog2 = json.load(open(base2 + ".progress.json"))
    ok("⑥ curves/seg_per 缺省→None（向后兼容·不崩）", prog2["curves"] is None and prog2["seg_per"] is None)

print()
print(f"{'✅ 全部 PASS' if _fail == 0 else '❌ ' + str(_fail) + ' FAIL'}（{_n} 项）")
sys.exit(1 if _fail else 0)
