export const meta = {
  name: 'trb-independent-review',
  description: 'Independent adversarial review of TRB 2026-07-22~23 window: block1 false-cert fix, gap#1 soundness, undecided 21-27%, is_emergency over-trigger, strategic call',
  phases: [
    { title: '独立复核', detail: '4 条独立线：block1+gap#1 数学 / 声称③ / 声称② / 战略④' },
    { title: '对抗证伪', detail: '对每条裁决派怀疑者试图refute' },
    { title: '完备性批评', detail: '窗口和主复审都漏了什么' },
  ],
}

const REPO = '/home/user/TRB-2027-ContinuesPPO/TRB'
const SCRATCH = '/tmp/claude-0/-home-user-TRB-2027-ContinuesPPO/c66f9aab-d514-56eb-b3c3-8a5123b55141/scratchpad'

const COMMON = `
你在对一个海事避碰 RL 项目(TRB 2027 投稿)做【独立对抗复审】。上一个工作窗口(2026-07-22~23)自己承认它的探针曾被对抗审抓出一个 CRITICAL 假认证。**绝不信它任何"把握"·别 rubber-stamp·尽量证伪**。
本机环境：python3 有 numpy/scipy/shapely，但【没有 vesselmodels】(装不了)。对常控 (a,ω) 的官方 yp 动力学 f=[cosθ·v, sinθ·v, ω, a] 解析可积，用 RK4/欧拉 dt≤0.1 与官方 odeint 误差~1e-8，可本机重建。
关键文件(用 Read 读)：
- 探针: ${REPO}/代码/m1_dock_wip/block3_partition_probe.py (清障判据 clearance_profile ~L86, keep_course_min_dist ~L146, 机动族 ~L193, 分类 classify_state ~L252, selftest ~L601)
- gap#1 证书: ${REPO}/代码/trb_env/usv_projection.py (imminent_unavoidable_certificate ~L705, _reach_params ~L681, _lateral_reach_bound ~L692)
- is_emergency: ${REPO}/代码/trb_env/usv_colregs.py:435 ; 动力学: ${REPO}/代码/trb_env/usv_dynamics.py
- 原始态数据(可本机读+复算): ${REPO}/结果/结果-block3-0722/{block3_adv_states.jsonl(1363), block3_rho5_states.jsonl(423 金标), block3_synthetic_states.jsonl(300 合成真对撞)}
  每行字段: ego=[x,y,heading,v], obs=[x,y,heading,v], obs_len, obs_wid（ego 恒为 SR108 175×25.4；obs 用数据真尺寸）
- 窗口自记裁决: ${REPO}/03_决策与教训.md 第 4017-4088 行(L195/L196/L197/L197-B)
- 主复审已跑的独立脚本(可参考/复用/挑错): ${SCRATCH}/{recompute_kc.py, reproduce_falsecert.py, reclassify.py}
你可以写自己的 python 脚本到 ${SCRATCH}/ 下跑。返回值=结构化裁决(不是给人看的汇报)。`

const SCHEMA = {
  type: 'object',
  properties: {
    verdicts: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          claim: { type: 'string', description: '被审的具体声称' },
          verdict: { type: 'string', enum: ['SOLID', 'OVERCLAIM', 'WRONG', 'UNCERTAIN'] },
          confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
          evidence: { type: 'string', description: '你亲自跑/推的证据(含数字/脚本名)，别转述窗口' },
          key_risk: { type: 'string', description: '这条裁决最可能错在哪(留给证伪者)' },
        },
        required: ['claim', 'verdict', 'confidence', 'evidence', 'key_risk'],
      },
    },
    summary: { type: 'string' },
  },
  required: ['verdicts', 'summary'],
}

const VERIFY_SCHEMA = {
  type: 'object',
  properties: {
    target_claim: { type: 'string' },
    refuted: { type: 'boolean', description: 'true=你成功找到裁决的漏洞/反例；false=裁决站得住' },
    finding: { type: 'string', description: '你试图 refute 的具体动作与结果(跑了什么/推了什么)' },
    corrected_verdict: { type: 'string', enum: ['SOLID', 'OVERCLAIM', 'WRONG', 'UNCERTAIN'] },
  },
  required: ['target_claim', 'refuted', 'finding', 'corrected_verdict'],
}

const finders = [
  {
    key: 'MATH',
    prompt: `${COMMON}

【你的任务：独立重推 block1 清障判据 + gap#1 证书的 soundness】
不要相信"600 对抗 0 假认证"这类测试佐证——独立从数学上判两件事：
(1) 清障判据(clearance_profile)：区间下界 (d_k+d_{k+1}−L·h)/2，其中修正版 L_seg = max(|v_k|,|v_{k+1}|)+a_max·h + |ω_seg|·R_circ + |v_m|。
    问：在【生产口径动力学】(步内 v 可冲负、10s 边界才钳、a=±0.24、ω=±0.03)下，L_seg 是否真是子区间 [t_k,t_{k+1}] 上 |d'(t)| 的合法上界？
    重点查：① a_max·h 项是否足够兜住"10s 边界钳前的 v overshoot"(采样点是钳后值)——构造一个采样点在钳后、但子区间内 v 峰值远超采样值的例子试试。② ω 项用 R_circ=88.4 是否够(船体任意点到中心最大距离)。③ 他船项 |v_m| 是否够(他船 CV)。目标=造出一个【修正版仍假认证】的反例(clears=True 但真撞)，或严格论证造不出。
(2) gap#1 证书(imminent_unavoidable_certificate)：对每个 t 查 ego 可达箱(纵 ½a_eff·t²、横 L_lat(t))4 角是否都落进 O(t)⊕disk(r_insc=½·25.4)。
    问：R_box 是否真过近似 ego 全可达中心集？a_eff=√(a_max²+(v_bnd·ω_max)²) 与 b_lat 的推导是否 sound？传入 obs_width=真宽(非 under)是否破坏 soundness？造一个 gap#1 假阳(说不可避但其实可避)的反例，或论证造不出。
两件都要给出：亲自跑的脚本/亲手推的不等式。返回 verdicts(每件一条)。`,
  },
  {
    key: 'CLAIM3',
    prompt: `${COMMON}

【你的任务：审声称③ = "is_emergency 大幅过触发·真冲突罕见"】
窗口称：对抗基线 1363 个 ρ5 态里 97.6%(1330) 是"假紧急"(keep-course 不做机动已安全)、真对撞只 15(1.1%)；金标 423 态"0 真冲突"。
你要：
(1) 本机独立复算 keep-course 分桶(三个文件)。keep-course=ego 恒速恒向、obs 恒速 CV，全程最小船体距 ≤0 记真对撞。核对 15/1330 与金标数。
(2) 【关键 nuance】读 is_emergency(usv_colregs.py:435)：它把他船当【可机动的增长圆盘(reach_radius_pm)+大外接圆】、视界 180s；而 keep-course 把他船当【纯 CV 矩形】。这个建模不对称下，"假紧急"这个标签是否公道？是不是 is_emergency 故意保守(安全谓词本就该保守)？"过触发"的论文诚实料框法会不会反而误导?
(3) 独立复核主复审的一个发现：金标其实有 2 个 kc≤0(不是窗口说的0)+3 个 near-miss(<57m)，L196"最近10态57-210m"是120欠采样错。跑出来核实这2个态是不是真的、是不是同一次相遇的长视界擦碰。判这个错的 materiality(定性"罕见"是否仍成立)。
返回 verdicts。`,
  },
  {
    key: 'CLAIM2',
    prompt: `${COMMON}

【你的任务：审声称② = "方向 A 未决 21-27%"】
窗口三分区(真对撞子集)：合成 n=300 → avoid 65.7%/unavoid(gap#1) 13.7%/undec 20.7%；对抗 n=15 → 53.3/20.0/26.7。
结论"方向 A 可行但不完整，~1/4 未决"。你要判 21-27% 靠不靠谱：
(1) 【soundness 方向】清障判据是 sound 充分条件(无假 clear)、gap#1 也是 sound 充分条件(无假 unavoid)。那么 undecided = "两个 sound 充分条件都没覆盖" → undecided 是【真硬骨头的上界还是下界】？换句话说 21-27% 是被高估还是低估？想清楚这个方向对"可靠性"的含义。
(2) 【family artifact 测试】主复审重建了分类流水线(${SCRATCH}/reclassify.py)，对比"窗口族(~20机动)"vs"加浓族(~90机动)"的 undecided。读它的结果(可能在 ${SCRATCH}/../tasks/ 或自己重跑 python3 reclassify.py 100)。若加浓族 undecided 大幅下降→undecided 主要是机动族弱的产物(不可靠)；若不降→接近场景天花板。你独立判断，必要时自己再加浓机动族(长序列/两段减速转向)重测。
(3) 【分布/样本】合成态是窗口自构造的对撞几何(gen_synthetic_conflicts ~L164)、非真基准分布；对抗真对撞 n=15。评估这两个对"21-27%可推广性"的杀伤。
(4) L197-B 称密搜130控制发现未决里96%是场景天花板/4%机动族弱——这是窗口单次经验搜索。你信不信这个 96%？
返回 verdicts。`,
  },
  {
    key: 'STRATEGY',
    prompt: `${COMMON}

【你的任务：steelman + attack 战略判断④】
窗口结论(L197 D)："真冲突本基准罕见 → 方向 A practical 价值低 → 天平推向 demonstration+诚实刻画为主线；方向 A 作 demonstrated 扩展；多障碍才是真舞台。"
背景：投 TRB 2027 Presentation 类(非 NeurIPS/CDC)，摘要 deadline ~8月初(现 2026-07-23，约1.5周)。卖点=连续动作 ∩ 可证明方向合规 ∩ COLREGs 三重交集(此前空白)。项目已有：可证明层章节(命题1/2/3)已成型、金标策略 IQM 98/0碰撞、热启动治崩。方向 A=把紧急态从"检测"做成"可证明恢复层"(需探针+执行接管工程+治OOD反噬+烧卡)。
你要：
(1) Steelman "demonstration 为主线"：为什么在 deadline+已有料下这是对的。
(2) Attack：这个判断会不会下得太早/太自我设限？(a)"方向 A practical 价值低"只基于【单障碍+好策略】——多障碍下真险多，会不会其实价值高？(b) demonstration-only 会不会被审稿人判"薄/A+B缝合"(导师原批评)？三重交集空白到底够不够撑一篇？(c) is_emergency 过触发+真冲突罕见，对论文是【利好诚实料】还是【削弱 motivation(没险可救何必要盾)】？
(3) 给一个平衡建议：主线怎么定、方向 A 摆哪、deadline 前该烧不该烧卡。
你可以读 ${REPO}/02_进展与交接.md 顶部 banner + ${REPO}/Paper/ 下的 tex/md 了解现状。返回 verdicts(把每个战略判断当一条 claim 裁 SOLID/OVERCLAIM/WRONG/UNCERTAIN)。`,
  },
]

phase('独立复核')
const results = await pipeline(
  finders,
  (f) => agent(f.prompt, { label: `finder:${f.key}`, phase: '独立复核', schema: SCHEMA, effort: 'high' })
            .then(r => ({ key: f.key, ...r })),
  // 对抗证伪：对该 finder 每条 verdict 派一个怀疑者
  (r) => {
    if (!r || !r.verdicts) return { key: r?.key, verdicts: [], verifies: [] }
    return parallel(r.verdicts.map(v => () =>
      agent(`${COMMON}

【你是证伪者】另一个 agent 对下面这条声称下了裁决。**你的唯一目标是 refute 它**(找反例/找它没跑的口径/找它推错的地方)。默认怀疑=refuted，除非你亲自验证后它确实站得住。
被审声称: ${v.claim}
它的裁决: ${v.verdict} (信心${v.confidence})
它的证据: ${v.evidence}
它自认最可能错在: ${v.key_risk}
你亲自去跑/推(可写脚本到 ${SCRATCH})，然后给 refuted(true/false) + corrected_verdict。`,
        { label: `refute:${r.key}`, phase: '对抗证伪', schema: VERIFY_SCHEMA, effort: 'high' })
    )).then(verifies => ({ key: r.key, verdicts: r.verdicts, verifies: verifies.filter(Boolean) }))
  }
)

phase('完备性批评')
const FINDINGS = results.filter(Boolean).map(r => {
  const vs = (r.verdicts || []).map((v, i) => {
    const ref = (r.verifies || [])[i]
    return `  · [${r.key}] ${v.claim} → ${v.verdict}(${v.confidence})｜证伪:${ref ? (ref.refuted ? 'REFUTED→' + ref.corrected_verdict : '站得住') : 'n/a'}`
  }).join('\n')
  return `[${r.key}] ${r.summary}\n${vs}`
}).join('\n\n')

const critic = await agent(`${COMMON}

【你是完备性批评者】前面 4 条线复审了：block1/gap#1 数学、is_emergency过触发、未决21-27%、战略。
主复审已独立坐实：① block1 反向假认证修对了(重现+0.133假认证/修后clears=False) ③ 对抗97.6%假紧急精确复现 + 金标其实有2个真对撞态(窗口欠采样说成0)。
下面是 4 条线各自的裁决(含证伪结果):
${FINDINGS}
你的任务：指出【窗口和主复审都可能漏掉的东西】。想这些角度：
- 有没有一个 modality/口径没人跑？(比如 ego 也是 SR108 的假设有没有被验、obstacle CV 外推超出场景真实时长的问题、is_emergency 的 reach_radius_pm 参数、gap#1 的 n_grid=241 分辨率会不会漏 t*)
- 三个数据文件本身有没有可疑处(重复态、退化态、seed 覆盖)？
- 有没有哪条"已被证伪/降级"其实主复审判错了方向？
- 对写 demonstration 论文，哪些数字是"能写进去"的硬料、哪些绝不能写？
返回 verdicts(每条一个你新发现的 gap 或对前面裁决的再校正)。`,
  { label: 'completeness-critic', phase: '完备性批评', schema: SCHEMA, effort: 'high' }
    // 注：FINDINGS 占位在下方 replace
)

return { finders: results, critic }
