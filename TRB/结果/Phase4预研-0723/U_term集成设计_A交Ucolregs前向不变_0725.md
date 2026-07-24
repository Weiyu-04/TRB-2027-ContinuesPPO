# U_term 集成设计 + A∩U_colregs 前向不变分析（2026-07-25·任务A设计节·未动核心代码前）

> 目的 = 把 Prop4 v2 的 backup-maneuver 终端约束【真正接进盾】(兑现"可证明前向不变盾")+ 补 A∩U_colregs 前向不变这个唯一未证的洞。
> **纪律**：基础设施核心(usv_projection.py)=最高审核标准 → 本文=先设计+理论·下一节点才动代码(冒烟测试+≥2对抗agent审+措辞收紧)。
> 依据：`Paper/命题4_前向不变递归可行_草稿_0723.md` v2.1 + `U_term设计_A3探针_B1基线_规格.md` + 本窗口独立复审 L200(门2 in-A=100%定值)。

## 0 · 关键发现:盾【已有】递归可行性机件·但默认关·且用的是【旧离散验证】非 cert_v2
读 `usv_projection.py` 查实(更正 L200 F-A 的措辞精度)：
- `ContinuousColregsProjection.__init__(recursive_feasibility=False)` **默认关**。A3 collect/训练用的 `ContinuousProjectionEnv(...)` **没开** → 部署/训练盾跑的是 `safe_action`→`project_qp`(U_box∩U_colregs∩U_collision-free 单步 QP)·**终端检查整块跳过**(L364-366)。所以 L200 F-A"部署盾=裸单步投影"的结论【成立】。
- 但机件【存在】：`_terminal_feasible`(L424)=开则加"落点 s' 是否【存在】合规脱身机动"(存在性级递归可行·As(s')≠∅)。**它用的是 `encounter_action_verification`(usv_colregs:855·离散候选转向×`build_st`/`maneuver_verified`·dt_sim=0.5 Euler·漂移1-3m·靠 dobs_safety=2·l_obs 大膨胀吸收)**——**非 block1 SOUND cert_v2**。docstring 自认"影响 provably 措辞精度(D5)"。
- ⟹ **U_term 任务的真内容 = ①打开 recursive_feasibility ②把 `_terminal_feasible` 的判据从【离散 encounter_action_verification】升级成【block1-SOUND cert_v2 backup-maneuver】·③打通 A∩U_colregs**。不是从零造·是"开+换 sound 判据"。

## 1 · U_term 集成设计（最小侵入 sound 版·推荐先做这版·非直接上"replace disk"大改）
**核心洞察**：前向不变 = "每步落点 s'∈A"(A=可清障集·成员=∃ cert_v2 certified 直行尾脱离)。盾单步 QP 已保【当步】无碰(disk 半平面)；再加"落点∈A"终端门 = 每步都留着一条 SOUND 永久清障退路 ⟹ 轨迹永不进不可避撞态 = **可证明前向不变无碰(A 上·单 CV)**。

- **落地 = 在 `project_qp` 的终端检查处·用 cert_v2 版判 s'∈A**(替 encounter_action_verification)：
  - 新增 `_terminal_feasible_certv2(s_ego, s_obs, u_applied, dt, vessel_params)`：
    1. `s' = dynamics.step(s_ego, u_applied, dt)`(复用官方·忠实)；`s_obs' = predict_state_cv(s_obs, dt)`(CV)。
    2. 判 `s'∈A`：对 cert_v2 直行尾脱离族(加密 89 条·L200-C)找【任意方向】首个 certified_perm → ∃ ⟹ s'∈A。
       - **O(1) 优化(可选·后做)**：维护当前 certified 备份序列 m*·先试 m* 走一步后的尾巴 m'_tail 是否仍 certified(Prop4 保证在)→ O(1)·命中则免搜族。miss 才全族搜。**m* stateful → episode reset()(D13 头号静默错误类·必须 reset)**。
    3. 返回 ∃ ⟹ True(放行 u_applied)·否则 False(退兜底·同 P=∅ 出口)。
  - **加配置开关** `terminal_mode ∈ {'off'(默认·bit-identical), 'discrete'(现 encounter_action_verification), 'certv2'(新 SOUND)}`·默认 'off' 保 bit-identical(现状不破)·eval/证明性跑用 'certv2'。
- **soundness 锚 = cert_v2(block1 Lipschitz·已复审 SOUND + 本窗口 0 假放行 fuzz 坐实)**·比旧 encounter_action_verification(dt_sim=0.5 Euler·~3m 漂移)严格 → **这才配得上"provably"**。
- **"replace disk 单步半平面→信赖域盒"(规格钉死1)= 更大改·留作后续**(它把碰撞安全全交 layer2 证书·收益=给 RL 更大凸域·但风险高)。**第一版不做**·先用"现单步 QP(disk) + cert_v2 终端门"这个最小 sound 集成(已足够兑现前向不变 claim)。
- **测试**：①本机单测 `_terminal_feasible_certv2`(自包含 RK4·喂 A 内/A 外态·验判定正确)②bit-identical(terminal_mode='off' max|Δ|=0)③服务器闭环纯 eval(待 user 拍·非烧训练卡)看盾接 cert_v2 终端门后碰撞率/介入率/到达率(对照 off)。

## 2 · A∩U_colregs 前向不变分析（唯一未证的洞·本节给【可证明的作用域】+ 诚实残余）
**问题**：A∩U_colregs = {s: ∃ certified 脱离·其【首步】在合规让路向(give-way ω 号)}。s∈A∩U_colregs 施合规 u0 后·s'=F(s,u0) 是否仍 ∈A∩U_colregs(仍有合规 certified 脱离)?

**为什么不能照搬 Prop4 尾巴论证**：Prop4 证 A 不变靠"同一序列的尾巴仍 certified"。但合规备份 m=(转 t1 秒·后直行)·尾巴 m'=(转 t1−Δ 秒·后直行) 的首步——若 t1>Δ 仍是转(合规)·**若 t1≤Δ 则尾巴首步=直行(ω=0)·未必满足合规硬半空间(严格 starboard ω<0)** → 尾巴论证在"转向本步走完"时断。

**可证明的作用域(分情形)**：
- **情形1(t1>Δ·转向未走完)= ✅ 可证**：合规备份 m 首步 u0 转 starboard·尾巴 m'=(转 t1−Δ·后直行) 首步仍 starboard=合规·且由 Prop4 尾巴论证 m' 对 s' 仍 certified(同一物理轨迹后半段) ⟹ **s'∈A∩U_colregs**。∴ **只要合规备份的转向相扩过当前决策步·A∩U_colregs 前向不变成立**。
- **情形2(t1≤Δ·转向本步走完·s' 已过转)**：尾巴=直行·首步非严格 starboard。此时 s' 已转离+开始分离(过 CPA)·**态势通常 de-escalate**(ρ' → stand-on/no-conflict·give-way 硬约束消失·合规平凡满足)。**若 ρ' 仍 give-way**(未 de-escalate)则须 s' 有【新】合规 certified 脱离——**这是唯一未闭合的理论缝**。
- **经验闭合(本窗口 L200 定值)**：**门2 in-A 条件率 = 三态全 100%(全 860 真让路态)** ⟹ 情形2 的缝在真分布上【不咬】(每个可清障让路态都有合规脱离·包括新的)。
- **🟢🟢 数值坐实(2026-07-25·verify_colregs_invariance.py·241 有合规备份的真让路态)**：**情形2 = 0**(全部落情形1)！即【每个】真让路态的合规备份·其转向都扩过一个决策步(t1>Δ=10s·大船慢转 1.72°/s 天然需持续转>10s)。逐条验:①走一步后尾巴仍 certify s' = **241/241(100%)**(Prop4 尾巴论证坐实)②尾巴首步仍合规 = **241/241(100%)**(情形1 断言坐实)③s' 重新有【新】合规 certified 脱离 = **241/241(100%)**(冗余稳)。→ **情形2(未闭合的理论缝)在真分布上【根本不触发】·可证明的情形1 机制覆盖 100%** ⟹ **A∩U_colregs 前向不变 = 由可证明机制(情形1)+ 经验普适(情形2=0·再加 fresh 重入 100%)【实质闭合】**(local RK4·待官方 --gates)。
- **实操闭合(更强·可选)**：备份策略【总选转向相够长的 m*】(族含转到 120s=12 决策步·转到清为止再直行)·使情形1 覆盖整个接近段·情形2 只在"已清且分离大"时触发(那时 ρ' 必 de-escalate) ⟹ **A∩U_colregs 前向不变在"备份转向覆盖接近段"这个可满足条件下成立**。这是干净的可写命题(比"一般证明"弱·但诚实且够论文)。

**⟹ 论文命题(诚实收紧·据数值坐实更新)**：
> "在可清障集 A 上·受盾策略经 cert_v2 backup-maneuver 终端约束 ⟹ **可证明前向不变无碰**(单 CV·A 上)。让路态:合规备份的 starboard 相扩过当前决策步时(情形1)·后继仍在 A∩U_colregs(合规且安全前向不变·尾巴论证严证);**且在真让路态分布上·合规备份的转向恒 >一个决策步(大船慢转天然)·故情形1 覆盖 100%(情形2=0·数值坐实 241/241)+ 后继恒重入合规脱离(门2 in-A=100%)** ⟹ A∩U_colregs 前向不变【实质成立】(local RK4·待官方 --gates)。残余仅 A 成员边界(~0-3% 不可清障·落 fallback)·非合规冲突。"

## 3 · 实施状态 + 盾适配器 ready-to-apply spec（本机可做的已做·盾核心待服务器冒烟）
### 已完成(本机·已提交 main)
- **✅ soundness 核心 `代码/trb_env/uterm_terminal.py`**(纯·不依赖 vesselmodels)：cert_v2 + straight_tail_family(加密89) + state_in_A/successor_in_A(O(1) backup 复用)。**本机单测 6 项全过**(`代码/tests/test_uterm_terminal.py`)：引理1/state_in_A/hint==全搜/合规过滤/**SOUND fuzz 0 假放行**/**first_unsafe_t==block3.clearance_profile(L198 SOUND)逐点相等**。
- **✅ A∩U_colregs 前向不变理论+数值闭合**(§2)。

### 🔴 盾适配器(usv_projection.py)= 服务器活(本机无 vesselmodels·不可冒烟·不盲改核心)
拟加(全在 `recursive_feasibility=True ∧ terminal_mode='certv2'` 门后·默认 off=bit-identical by construction)：
```python
# __init__ 加: terminal_mode: str = "discrete"（validate ∈ {'discrete','certv2'}）; self.terminal_mode=...
# project_qp L366 改为 dispatch:
if self.recursive_feasibility and not self._terminal_ok(s_ego, s_obs, u_safe, rho, dt, vessel_params):
    return _fallback()
# 新增:
def _terminal_ok(self, s_ego, s_obs, u_applied, rho, dt, vp):
    if self.terminal_mode == 'certv2':
        return self._terminal_feasible_certv2(s_ego, s_obs, u_applied, rho, dt, vp)
    return self._terminal_feasible(s_ego, s_obs, u_applied, rho, dt, vp)   # 现离散(back-compat)
def _terminal_feasible_certv2(self, s_ego, s_obs, u_applied, current_rho, dt, vp):
    # s'/s_obs'/ρ'(当前ρ播种·同 _terminal_feasible); ρ'∈{0,1,5}→True(同现·经验兜底诚实标)
    # ρ'∈{2,3,4}: ego_vec/obs_vec; sign = -1 if ρ'∈{head_on,crossing} else 0(overtake 松)
    # integ = lambda e,segs,T,h: self._integrate_maneuver_official(e,segs,T,h,vp)  # usv_dynamics.step 分段+10s钳
    # return uterm.successor_in_A(ego_vec,obs_vec,olen,owid,integ,H=self._sc.t_horizon,h=self.terminal_dt_sim,require_omega_sign=sign)[0]
```
### 🔴🔴 服务器session须先定的 OPEN(本机定不了·会影响 soundness/数值)
1. **他船宽 obs_wid 来源**：`VesselState` 【无 width 字段】(只 length·usv_colregs:80)·project_qp 收不到真宽。两选:
   (a) **保守 w=obs_len**(=现 `_vessel_circumradius` 默认·SOUND 但悲观·会比本窗口 dense_full_gates 用真宽 25-44m 低估 A/门率)·无需改 API；
   (b) **真宽 plumb**(env 有 obstacle_shape.width → 经 safe_action→project_qp→terminal 传入·recover 高 率)·须改 API(小侵入)。**先跑 (a) 保 sound·再评是否值得 (b)**。
2. **episode reset**：若上 O(1) backup m* 缓存(stateful)→必 episode 边界 reset(D13 头号静默错误类)。第一版可【不缓存】(每步全族搜·免 stale·慢但 sound)·先求对再优化。
3. **integrate_maneuver_official 复用**：`block3.integrate_maneuver_official` 已是官方分段积分(需 vesselmodels)·可抽到 uterm 或盾内复用·别再写一份(divergence 风险)。
### 服务器 smoke + eval 计划(待 user 拍·纯 eval 不烧训练卡)
1. bit-identical: terminal_mode 默认(recursive_feasibility=False) max|Δ|=0 vs 现状。
2. 单测 _terminal_feasible_certv2(喂 A 内/外态·对 uterm 本机结果)。
3. ≥2 对抗 agent 审(集成/时序/契约/width 口径/reset)。
4. 闭环纯 eval(certv2 vs off)看碰撞率/介入率/到达率/门通过率。
5. 过了→把情形1 命题写进 `Paper/命题4草稿` §COLREGs 叠加(替"【尚未证】")。

*(本文=设计+理论节·未动 usv_projection.py 核心代码·未烧任何卡。)*
