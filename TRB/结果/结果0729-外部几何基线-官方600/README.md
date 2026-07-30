# 外部几何基线 · 官方测试集 600 · 2026-07-29（首次真跑）

> **这是项目历史上第一次真正跑通三条外部几何基线**（此前所有窗口都只有脚本、没有产物）。
> 跑在云端容器（32 vCPU / 60 GB / 无 GPU），三个方法各占一核并行，库版本对齐 `requirements.txt`。

## 一、口径（写论文必须照抄这一段）

| 项 | 值 |
|---|---|
| 报数池 | 官方 1400/600 划分的**测试 600**（`split_seed=0`, `test_frac=0.3`, 池 2000） |
| 分母 | `n=600`，`clean=600`，`strict=600` —— **三者相等** |
| 训练泄漏 / 验证泄漏 | **0 / 0**（`BASELINE_LEAK_MANIFESTS=manifest_official_1300.json`） |
| 会遇类型 | crossing 395 · head-on 205 · **overtake 0**（官方 2000 池本身没有追越） |
| 调参池 | 官方**训练集**抽 100 例（`BASELINE_TUNE_SRC=train`），与报数池零交集 |
| 动作箱 | 两档都报：`rl`（±0.048/±0.018，与本文策略同权限）· `full`（±0.24/±0.03 物理满程） |
| 单机线程 | `OMP/OPENBLAS/MKL_NUM_THREADS=1`（单线程墙钟，耗时数字只能同机同口径比） |
| 脚本版本 | `script_rev=rb2-2026-07-27` · `runner_rev=b3-2026-07-27` · `reeval_rev=r13-2026-07-27` |

⚠️ **`BASELINE_LEAK_ACK=1` 是必须的**：脚本把预期分母硬编成旧小池的 clean 577 / strict 563，
正式实验换成官方 600 之后它会 fail-closed 直接中止。这是**正确的设计**（防止分母被静默改掉），
正式一趟必须显式放行并在论文里写清分母口径。

## 二、结果（strict 600 · 各方法自己最好的那一档）

| 方法 | 箱 | 到达% | 碰撞% | 违规/局 | 其中让路违规 | 其中直航违规 | 转艏Δ | 油门Δ | jerk | 每局秒 |
|---|---|---|---|---|---|---|---|---|---|---|
| CBF | rl | **70.17** | **0.00** | 4.950 | 0.467 | 4.483 | 0.002424 | 0.001841 | 0.1708 | 0.181 |
| CBF | rl (margin=175) | 69.83 | 0.00 | **4.488** | 0.447 | 4.042 | 0.002354 | 0.001846 | 0.1670 | 0.185 |
| CBF | full | 69.33 | 0.00 | 6.482 | 0.515 | 5.967 | 0.000582 | 0.001056 | 0.0503 | 0.187 |
| PD | rl | 58.83 | 18.00 | 9.785 | 0.435 | 9.350 | 0.004922 | 0.001825 | 0.3098 | 0.189 |
| PD | full | 57.67 | 16.67 | 8.097 | 0.625 | 7.472 | 0.000323 | 0.001217 | 0.0405 | 0.191 |
| VO | — | **未跑完**（被本窗口 kill 让核） | | | | | | | | 0.45 s/局（实测速度） |

CBF 最优档参数：`a1=0.02, a2=0.02`（margin 0 或 175 两版都在）。PD 无可调参数。

## 三、三条对写作最要紧的结论

1. **CBF 碰撞 0.00%** —— 它在这个基准上和我们打平。⟹ **卖点不能是碰撞率，也不能是到达率**，
   只能是 **COLREGs 违规**。这一条推翻了起飞前"CBF 非自动 0 碰撞"的旧判断（那是小池 40 例的结论）。
2. **违规几乎全是直航违规**（CBF 4.48/4.95 里 4.04/4.48 是直航；PD 9.79 里 9.35 是直航）。
   几何控制器"不撞"的代价是**该保向保速的时候仍在机动**。这是我们相对它们的真实差距所在。
3. **不许说我们比几何控制器平顺**。PD-full 的转艏增量 0.000323、CBF-full 0.000582，
   比我们探索期的 0.00256 小一个量级。正确写法："平顺度相当，同时违规更低"。

⚠️ 另一条口径事实（复审 wf_3d34bb96-609 独立查实）：这三条基线走无盾路径、`态势步数合计` 全落在
`ρ0`（`{"0": 53339, "1..5": 0}`），却仍然报出非零违规 ⟹ **违规计数用的是与盾的 ρ 无关的离线裸态势
谓词**。论文的指标定义节必须把这个评分器口径单独写出来，否则读者会误以为违规是盾判的。

## 四、复现（VO 那一档给 user 重跑用）

```bash
cd /home/user/TRB-2027-ContinuesPPO/TRB   # 或服务器上的 /root/trb
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export BASELINE_MANIFEST_DIRS="$PWD/balanced_pool" REEVAL_MANIFEST_DIRS="$PWD/balanced_pool"
export STEP4E_SDIR="$PWD/scenarios" REEVAL_SDIR="$PWD/scenarios"
export BASELINE_LEAK_MANIFESTS=manifest_official_1300.json
export BASELINE_BOX=rl,full BASELINE_TUNE_SRC=train BASELINE_LEAK_ACK=1
BASELINE_METHODS=vo BASELINE_OUT=$PWD/结果/结果0729-外部几何基线-官方600/baselines_vo.json \
  python3 -B 代码/m1_dock_wip/run_baselines_official.py --run
```

VO 实测 **0.45 s/局**（CBF 0.15、PD 0.15），扫参档位比另两个多 ⟹ 是三者里最慢的一条，
单核约需 2–3 小时。**建议直接并进训练结束后的重评那一趟**（那时核是空的），不必单独占机器。

## 五、库版本（与 `requirements.txt` 对齐，装反了跑不起来）

```
python 3.11.15 · numpy 1.24.4 · scipy 1.15.3 · shapely 2.1.2
commonocean-io 2025.1 · commonroad-io 2023.1   ← 必须钉 2023.1
```

两个装依赖的坑（下个窗口别再踩）：
- `antlr4-python3-runtime==4.9.3` 的 wheel 构建会失败 ⟹ 手工把纯 Python 源码摊进 `site-packages`，
  再 `pip install --no-deps commonocean-io`。
- `commonroad-io 2026.1` 把 `commonroad.geometry.shape` 挪了位置 ⟹ 必须钉 `commonroad-io==2023.1`。
- 不要装 matplotlib / cvxpy：它们强拉 `numpy>=1.25`，会把 1.24.4 顶掉。

## 六、文件清单

| 文件 | 内容 |
|---|---|
| `baselines_cbf.json` | CBF 全量（调参 6 档 + 最终 3 档 × strict/clean/全部 + 按会遇类型分型）|
| `baselines_pd.json` | PD 全量（无调参 + 最终 2 档）|
| `baselines_vo.json` | VO **部分**（`全部完成: false`，只有调参档，最终档缺）|
| `cbf.log` `pd.log` `vo.log` | 三条的完整标准输出 |
| `run_full.sh` | 本次真跑用的脚本（逐字保存）|
