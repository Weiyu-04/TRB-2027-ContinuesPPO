# Phase 4 · U_term 可算终端约束设计 + A3 真态探针规格 + B1 CBF基线规格
> 2026-07-23 · workflow wf_27032c4d（4方案设计+评审+证伪·2 Prop4复审·1综合·11agent·90万tok·未烧卡）。全量输出 `scratch/设计workflow_完整输出.txt`。

## 一 · U_term 推荐设计 = 方案②(backup-maneuver 信赖域 + O(1) 事后重认证)
**四方案评审**：②胜出（sound4/tract5/rl2/colregs4/cost5·唯一无 soundness 致命洞）。①CBF-clearance-barrier=弃作独立机制（可算形 soundness 只锚经验 κ·无 certified 二阶余项界·"可证明"落空）·但**留作写作/分析层**（标量 h + α 不变性证明是干净可发的 framing）。③gap1补集=弃作独立落地（唯一 sound 路 oracle+veto 不维持前向不变）·但"固定模板 clears⟹后继∈A"当②的证书本体。④reach-avoid V表=弃作 soundness 层（DP 未收敛·{V≥ε}⊇真不变集·U_term 可空）·**留作层0 µs 快筛**（只拒不放·V 欠估永不假放行）。

**②为什么同时 sound+可算+不杀策略**（核心洞察）：把 soundness 从"集合刻画"挪到**事后对单条拼接机动 u*⊕m*_tail 重认证**，绕开两个死结：
- (a) 尾闭合死结：②不依赖 ∃m∈族M·block1 证书认证任意序列·验证对象=执行对象=同一条被 SOUND 证书判过的轨迹 → 尾闭合问题消失（=修 Prop4 v1 洞2）。
- (b) 可算性 OPEN：成员判定不再是"∃清障机动"非凸存在量词·而是"这一条过不过 1 次证书"·O(1)。

**三层**（详见 `Paper/命题4_..._草稿_0723.md` v2）：层0 µs快筛(V表/gap1·只拒)→层1 内联信赖域盒(替换disk半平面·给RL凸域)→层2 事后重认证(唯一soundness锚·退回u₀兜底)。**funnel多备份**恢复自由度。

**🔴 三条落地必钉（省任一=静默不安全）**：
1. **replace 非 augment**：U_term-QP 路径信赖域盒**替换** disk 单步半平面(L560·562m外接圆)·非取交（否则 u₀ 被排除·恒可行破）。碰撞安全全交层2证书。
2. **verify-after 是唯一 soundness 锚**：绝不砍层2只留内联盒(κ=1 实测3/5不sound)。
3. **证书扩交付闭合率**：C 须输出 [t_pass,H] ṙ<0+过顶裕度（引理1 前提·非经验事实）。

## 二 · A3 真让路态探针规格（服务器纯 eval·不烧训练卡·待 user 拍板+逐字预检+screen）
仿 block3 探针体例（load rollout 落盘态·CPU 跑证书+分类·无梯度）。目的=把门1/门2/U_term 非空率从**合成分布**搬到**盾真实访问的态分布**·出真 go/no-go。
- **收态**：受盾 SAC（中途 ckpt 可·无须收敛）eval rollout·每步落 (s_ego,s_obs,ρ,give_way_dir,QP source,u_nom,u_applied,m*存在?)。筛 ρ2/3/4 让路态为主(各≥150)·ρ1 stand-on 单收(测 in-extremis)·ρ5 单收(测 A 外率)。**分层**：bearing×rel-course×range×速度比 + **L_obs 分箱(追越必富采·yaw饱和+方向合规最吃紧)**。总 N≥600。
- **门1 递归可行(测机制非全量重分类·修 v1 循环)**：对 s∈A 走一受盾步得 s'·三查隔离：(1a)同一尾巴 m*_tail 能 certify s'；(1b)收缩视界 H−Δ；(1c)引理1 闭合(H 处 ṙ<0+直行尾+过CPA margin·证书直接输出)。**报逐机制通过率**。GO≥99%(各≥99%)·<99%逐条查因(stale backup? reset漏? A须收紧?)。
- **门2 方向弹性**：让路态查**合规方向** certified backup 存在(head_on/crossing右转·overtake按get_turning_act)。报 A∩U_colregs 非空率·三态分开。GO≥85-90%·head_on≥95%·低于则扩残余集诚实刻画。
- **U_term 非空率**：A-成员率(真让路态)≥95% + 层1+2 产出非空可行域率(u* 过重认证率)。A 外态显式列率(落 emergency)。
- **附核**：真态 RK4 vs odeint~1e-8·gap#1 真态触发率分距离。

## 三 · B1 CBF-QP 外部基线规格（同 benchmark 同指标·纯 eval·兑现"别假设去测"）
- **实现**：标准海事 CBF-QP 动作滤波·h=‖p_ego−p_obs‖²−d_safe²(相对度2→HOCBF/指数CBF)·min‖u−u_nom‖² s.t. HOCBF+U_box·**同一 u_nom 喂两盾**(公平)。+一档 COLREGs-CBF(非对称barrier惩罚port passing)。**引一篇已发表公式·别自造**。ablation:含/不含 COLREGs 项。
- **同**：同场景池/动力学(10s ZOH步末钳v·单CV→多障碍+追越)/u_nom/eval harness(CPU纯eval·建 trb_env 内共用 dynamics/statechart)。
- **指标**：**裸船体**碰撞率(非d_safe)·COLREGs方向合规率·介入率·到达率·路径效率·最小距离分布。
- **🔴 测它是不是真0碰撞·别假设**：CBF 前向不变只在连续时间+精确模型+安全集真控制不变下成立·本设定破三条：(a)10s ZOH→决策时刻满足≠步内(inter-sample)安全·**须步内细积分复检最小距离**；(b)CBF-QP 同有 P=∅→fallback 期无保证·报 QP 可行率；(c)yaw饱和+v钳非线性→ḣ 线性外推失真。**预期 CBF-QP 有非零碰撞·报碰撞率带CI·不 claim 0**。大池(~2000-80000场景)+追越分层+多障碍报可行率。
- **差异化对照**：(i)CBF {h≥0}=解析手工barrier·非certified可清障集且非尾闭合；(ii)CBF-QP 10s ZOH 下无递归可行性证明·压经验步内安全=本文取代的；(iii)同场景实测对撞对照。

## 四 · 开放风险（靠 A3/B1 出真值）
真分布 A-成员率/方向弹性可能低于合成85%→残余集或更大；多障碍 A 不合成更弱；机动他船破 CV→证书假"清"→真撞（写"CV世界内"）；引理1 A_H⊆A_∞ 仍单 CV 限定。
