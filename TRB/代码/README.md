# 代码 / 环境与可复用清单（Phase 0 环境摸底产出）

> 2026-06-09 本地环境摸底结果，**已过独立 agent 对抗复核**（0 错误、0 过度声明）。新窗口要跑代码先看这里。
> 全部本地（user 指示"尽量本地，实在不行最后才上服务器"）；真正烧 GPU 的训练到 Phase 3 再议。

## 1. conda 环境 `trb`（独立，隔离 USV 的 DRL 环境）

- 创建：`conda create -n trb python=3.10`（实测 Python 3.10.20）。
- 解释器：`/opt/miniconda3/envs/trb/bin/python`。
- 已装并验证：
  - **commonocean-io 2025.1**（连带 commonroad-io 2023.1 / numpy 1.24.4 / matplotlib 3.6.3 / scipy 1.15.3 / shapely / commonocean-vessel-models 1.0.0）。
  - **cvxpy 1.7.5 + osqp 1.1.1 + clarabel + scs**（投影 QP 的免费求解器）。
  - **pypdf 6.13.1**（抽论文 PDF 文本核施工参数；2026-06-09 观测件时装）。
- 重装：`/opt/miniconda3/envs/trb/bin/pip install commonocean-io cvxpy osqp pypdf`。

## 2. CommonOcean（环境基座）摸底结论

- **是什么**：海事场景层 I/O 库——读/写/可视化场景 + 规划问题 + 障碍 + 预测。子模块：common / planning / prediction / scenario / visualization。
- **不是什么**：⚠️ **不含 RL gym 环境**。Krasowski 的"基于 CommonOcean + 偏航受限动力学的仿真环境"是她在 commonocean-io 之上自建、**未作为包发布** → **gym 封装我们 Phase 1 自建**（参考论文 §VI：27 维 obs + 奖励 + 终止 + 49 离散动作；我们改连续）。
  - 注：包内有名叫 `Environment` / `EnvironmentFactory` / `EnvironmentXMLNode` 的类，但那是**海事水域环境（水域/障碍）XML 读写**，**不是 RL gym 环境**——别被名字误导（grep `Environment` 会撞见）。
- **核心 API（已实测签名）**：
  - `Scenario(dt, scenario_id, ...)`
  - `DynamicObstacle(obstacle_id, obstacle_type, obstacle_shape, initial_state, prediction, depth)` —— prediction 支持 `TrajectoryPrediction` 和 **`SetBasedPrediction`**（⭐ Krasowski 的集合预测内建可用）。
  - `PlanningProblem(planning_problem_id, initial_state, goal_region, ...)`；State 复用 `commonroad.scenario.state.State`。
- **场景文件（Phase 1 step1 已定位 + 验证加载）**：
  - 源 = **`gitlab.lrz.de/tum-cps/commonocean-scenarios`**（**BSD 许可，可用**，与 serl-sprl 无 license 不同）。
  - Krasowski 基准路径 = `scenarios/HandcraftedTwoVesselEncounters_01_24/ZAM_AAA-1_20240121_T-{0..1999}.xml`（2000 个手工两船相遇场景）。
  - 实测加载 `T-0`：`CommonOceanFileReader(path).open()` → `dt=10.0s` ✓、**1 动态障碍**(MOTORVESSEL + TrajectoryPrediction) + **0 静态** ✓（两船开阔海）、1 规划问题(初始 position/orientation/velocity + goal)。**格式与论文 §VII 完全对上。**
  - 全量批下载（~160MB）**到 step 4 复现基线时做**（clone 或 sparse-checkout 该子目录）；录制 AIS 那 49 个另在仓库其他子目录。

## 3. serl-sprl（投影方法论参考）摸底结论

- 已 clone 到 `代码/serl-sprl/`（**仅供研究**，见许可证警告）。Python 3.10。
- 投影/集合代码位置（实测存在、非空）：`serl_sprl/sets.py`（394 行，含真实 `class Zonotope`）、`serl_sprl/projection/`（projection_helpers / projection_wrappers / base）、`serl_sprl/envs/safe_region.py`。
- ⭐ **有 SAC + 投影实现**：`serl_sprl/sb3_contrib/algorithms/sac/sacdiffproj.py`（405 行，`class SACDiffProj`；+ td3/ppo/a2c diffproj）、`serl_sprl/sb3_contrib/{serl,sprl}_policies.py`（SE-RL / SP-RL 封装）；`benchmarks/seeker/sac/`（海事最近类比，含 serl/sprl 各版本）。
- 🔴 **许可证警告（plan 级，已彻底核）**：**全仓无任何 LICENSE / 版权 / SPDX 声明**（pyproject 是未填完模板、README 空、git 史无 license；grep 命中全是假阳性 `TimeLimit`/`limit=` 之类）→ 法律上"保留所有权利"，**只能读、不能照抄 / 再发布**。
- → **决定（`03` L2）**：投影层**自己重写**（数学在论文附录 A.1 + 这些代码里已看懂），用免费求解器，论文里引用 Markgraf 2026，**不 vendor 它的代码进我们仓库**。
- 依赖与求解器（影响"装不装/怎么重写"）：
  - 依赖重（torch 2.8 + 商业 **Gurobi** + wandb 等）→ **不装它全栈、不跑它例子**（无必要 + Gurobi 要授权 + 无 license）。
  - ⭐ 旁证：serl-sprl 投影**硬编码 `solver=cp.GUROBI`**（`projection/base.py`、`sets.py:317`）→ 坐实它依赖商业 Gurobi，更印证"自己用免费 OSQP 重写"的必要。
  - 另两个专属依赖 `cvxpylayers`（做**可微投影**，SP-RL 才用）+ 自研 `continuoussets`：我们走 **SE-RL + action aliasing 惩罚**路线（§5），**不碰 cvxpylayers**（可微投影是 SP-RL 专属）。

## 4. 投影第三层可行性（已实测，agent 复跑复现）

最近点投影 QP 用**免费 OSQP** 跑通：不安全动作 `[a=0.20, ω=0.02]`（左转，违反右转令）→ 投影到 `[0.20, −0.01]`（右转，满足 a/ω 限幅 + COLREGS 右转半空间约束），status=optimal。trb 环境 `cp.installed_solvers()=['CLARABEL','OSQP','SCIPY','SCS']`，**无 Gurobi 也解得出**。**第三层不需 Gurobi。**

## 5. 「可复用 vs 要重写」清单

| 模块 | 来源 | 结论 |
|---|---|---|
| 海事场景 / 障碍 / 规划问题 / 集合预测 | commonocean-io | ✅ 直接用（已装）|
| Krasowski 基准场景文件 | CommonOcean 网站 | ⬇️ Phase 1 下载 |
| RL gym 环境（obs/reward/termination/step）| — | 🔨 Phase 1 自建（库不含；参考论文 §VI 27 维 obs）|
| COLREGS 状态机（相遇判定 + 让路方向）| Krasowski §IV + Table I/IV | 🔨 照论文重写（参数已锁，见文献核实笔记 ②）|
| zonotope 安全集 + 最近点投影 QP | serl-sprl（只读）+ 论文附录 A.1 | 🔨 自己重写（无 license），免费 OSQP（已验证）|
| SE-RL 接线（盾当环境一部分）| serl-sprl serl_policies（只读）+ Markgraf §5.1 | 🔨 自己重写，SAC 主体不改 |
| action aliasing 奖励惩罚 `w‖u−uφ‖²` | Markgraf §7.1 | 🔨 直接加进 reward（w∈{0.1,0.5,1,2} 待调）|
| SAC 求解器 | stable-baselines3 | ✅ 直接用（Phase 3 装 sb3）|
| 档位 B 不变集（RCI / reach-avoid）| Schäfer 2024 算法 | 🔨 Phase 4 为两船 + COLREGS 构造（最难，用可达性背景）|
| 评估：多种子 + 配对检验 + 自助法 CI + 钱图 | USV 项目方法论 | ✅ 迁移（见 §6）|

## 6. 从 USV 项目迁移的评估方法论

USV 项目打磨过的这套直接迁移（思路 + 部分脚本可改）：≥5 种子重跑、配对符号检验 / Wilcoxon、自助法 95% 置信区间、钱图绘制、PI 严谨（不 cherry-pick、报均值 + 方差）。对得上：Krasowski 用 10 seed + 自助法 CI；Markgraf 用 7 runs×10 seeds + Kruskal-Wallis。

---
**独立 agent 复核（2026-06-09）**：派 general-purpose agent 用真实命令逐条实测（pip list / import / inspect.signature / find / grep / 实跑 OSQP 投影），结论 **未发现任何错误或过度声明**；本文 3 处精确化（§2 Environment 命名 / §3 Gurobi 硬编码 + cvxpylayers）即采纳其旁注。
