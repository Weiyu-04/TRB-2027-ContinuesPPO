export const meta = {
  name: 'trb-paper-review',
  description: 'TRB 中文稿复审：代码主线是否全部落进论文 + 语言风格 + 诚实红线 + 体例篇幅',
  phases: [
    { title: 'CodeCoverage', detail: '按模块切片，逐片检查论文有没有写到' },
    { title: 'PaperAudit',   detail: '语言风格 / 诚实红线 / TRB 体例 三路并行' },
    { title: 'Verify',       detail: '对每条发现派独立证伪者' },
    { title: 'Synthesize',   detail: '汇总成可执行清单' },
  ],
}

const ROOT = '/home/user/TRB-2027-ContinuesPPO/TRB'
const PAPER = `${ROOT}/Paper/01_论文稿/729-paper-中文版.tex`

const FINDINGS = {
  type: 'object',
  required: ['findings'],
  properties: {
    findings: {
      type: 'array',
      items: {
        type: 'object',
        required: ['severity', 'what', 'evidence', 'fix'],
        properties: {
          severity: { type: 'string', enum: ['BLOCKER', 'MAJOR', 'MINOR'] },
          what:     { type: 'string' },
          evidence: { type: 'string' },
          fix:      { type: 'string' },
        },
      },
    },
  },
}

const VERDICT = {
  type: 'object',
  required: ['stands', 'why'],
  properties: {
    stands: { type: 'boolean' },
    why:    { type: 'string' },
  },
}

const COMMON = `
你在复审一篇投 TRB 2027 的**中文**论文稿：${PAPER}
项目根目录 ${ROOT}。相关约束（必读）：
  · ${ROOT}/CLAUDE.md §0 的诚实红线：能严证的只有命题1(远场单步无碰)+命题2(每让路步方向合规)
    +命题3(紧急迫近不可避可靠证书)；**绝不许**出现 provably collision-free / 裸 provably compliant /
    完整可证明覆盖；命题4 仍是 PROVISIONAL。
  · ${ROOT}/Paper/正式实验/README.md = 正式实验定稿方案（九种配置/官方划分/指标/诚实口径 16 条）。
  · ${ROOT}/Paper/02_理论推导/ = 四条命题与失败机理的推导底稿。
  · TRB 硬约束：正文约 7500 词、图表各折 250 词、**不允许附录**（完整证明移交外部技术报告）。
只报**你能给出文件:行号证据**的问题。拿不准的标 MINOR 并说明不确定在哪。不要报风格偏好之类的空话。
`

phase('CodeCoverage')
const SLICES = [
  { key: 'env-dyn-obs', files: '代码/trb_env/usv_env.py, usv_dynamics.py, usv_observation.py, usv_termination.py',
    focus: '环境/动力学/观测/终止判据：动作空间构造(含紧急槽)、观测 27 维布局、到达判据、回合上限、速度裁剪语义' },
  { key: 'colregs-shield', files: '代码/trb_env/usv_colregs.py, usv_projection.py, usv_continuous_shield.py',
    focus: '状态机态势判定与常数、U_colregs 各态势约束、无碰约束构造、投影 QP、兜底链各级、盾的介入语义' },
  { key: 'reward-dist', files: '代码/trb_env/usv_reward.py, usv_action_dist.py, usv_smoothing.py',
    focus: '奖励五分量与偏离原文之处、Beta 分布(含 α,β≥1 与众数问题)、平滑/塑形各项' },
  { key: 'protocol', files: '代码/run_formal_2027.sh, 代码/run_step4e.py, Paper/正式实验/README.md, Paper/正式实验/_common.py',
    focus: '实验协议：九种配置配方、官方划分与分母、步数/种子/并发、存档选取、指标采集、达标/发散判据、统计方法' },
]

const coverage = await parallel(SLICES.map(sl => () => agent(
  `${COMMON}
【你的切片】${sl.files}
【关注点】${sl.focus}

任务：**逐个读这些代码文件**（含 docstring 与关键常量），列出其中属于**方法主线**的机制与参数；
然后到论文里核对：**方法论节(Methodology)与实验节(Experimental Design)有没有把它写出来**。
判定标准 = 一个只读论文的复现者能否知道这件事存在、并把它实现对。
只报**论文缺了或写错了**的（给代码行号 + 论文行号）。**注意**：诊断性/探索期/未在正式实验启用的开关**不算缺**
（例如默认关闭且九种配置都不设的旋钮）——你必须先确认它在正式实验里真的生效，再报。`,
  { label: `coverage:${sl.key}`, phase: 'CodeCoverage', schema: FINDINGS })))

phase('PaperAudit')
const AUDITS = [
  { key: 'style', prompt:
    `审**中文学术写作质量**：① 是否还有项目内部行话（例：钱图/臂/钥匙/练成/崩/趟/口径/打满舵/贴边/阶梯）；
     ② 是否有"跟对标论文对齐"的口吻残留（例：忠实沿用/我们也用了/与原文一致）——方法论节必须讲我们自己的方法；
     ③ 术语是否前后一致、英文术语首次出现是否有中文界定；④ 是否有口语化、断言过强、或读起来像笔记而非论文的句子；
     ⑤ 段落是否过长、加粗是否滥用。给行号。` },
  { key: 'honesty', prompt:
    `审**诚实红线**：逐句扫全文，找① 任何超出 CLAUDE.md §0 允许范围的保证性表述；
     ② 命题4 是否处处标为暂定（对照 ${ROOT}/Paper/02_理论推导/命题4_前向不变递归可行_草稿_0723.md 开头的 v2.1 三条更正：
     "A 控制不变"≠"盾已前向不变" / 残余≈0 不是 46% 也不是 5% / 门2 测的是"存在"不是"每步 by construction"）；
     ③ 是否有把探索期口径的数字与正式口径混用；④ 定量声明是否都挂了必要限定；⑤ 是否有该声明而未声明的偏离。给行号。` },
  { key: 'venue', prompt:
    `审**TRB 体例与篇幅**：参考论文在 ${ROOT}/Paper/03_参考文献/（实测 25 页 / 正文约 5300 词 / 6 图 4 表 /
     33 条数字式参考文献 / 每行带行号）。核对：① 我们的图表数与额度（每张折 250 词、总 7500）是否算得过来，
     给出当前正文字数与折算词数的估计；② 章节次序与命名是否符合该体例；③ 参数是否该进表而散在正文里
     （user 明确要求参数集中成表）；④ 有没有该删的内容（例：无实验支撑的理论推广）；⑤ 无附录约束下哪些内容该移交技术报告。给行号。` },
]

const audits = await parallel(AUDITS.map(a => () => agent(
  `${COMMON}\n【你的职责】${a.prompt}`,
  { label: `audit:${a.key}`, phase: 'PaperAudit', schema: FINDINGS })))

phase('Verify')
const all = [...coverage, ...audits].filter(Boolean).flatMap(r => r.findings || [])
log(`初审共 ${all.length} 条发现，开始逐条证伪`)

const KEEP = all.filter(f => f.severity !== 'MINOR')
const verified = await parallel(KEEP.map(f => () => agent(
  `${COMMON}
【要证伪的发现】
  严重度：${f.severity}
  内容：${f.what}
  证据：${f.evidence}
  建议改法：${f.fix}

你的任务是**尽力证伪它**。回到论文与代码亲自核：证据里的行号对不对？论文是不是其实已经在别处写了
（换了措辞、或写在表注/图注/其他小节）？这个机制在正式实验里是否真的生效？
拿不准时**倾向判它不成立**（stands=false）——我们宁可漏一条，也不要把假发现塞进论文修改清单。`,
  { label: `verify:${f.severity}`, phase: 'Verify', schema: VERDICT })
  .then(v => ({ ...f, verdict: v }))))

const confirmed = verified.filter(Boolean).filter(f => f.verdict?.stands)
const refuted  = verified.filter(Boolean).filter(f => !f.verdict?.stands)
log(`证伪后：成立 ${confirmed.length} 条 · 被推翻 ${refuted.length} 条 · MINOR 未验 ${all.length - KEEP.length} 条`)

phase('Synthesize')
const synth = await agent(
  `${COMMON}
下面是经过独立证伪之后**成立**的发现（JSON）：
${JSON.stringify(confirmed, null, 1)}

另有未经证伪的 MINOR 级发现：
${JSON.stringify(all.filter(f => f.severity === 'MINOR'), null, 1)}

任务：整理成一份**可执行的修改清单**，用中文，给主窗口照着改。要求：
  · 按"必须改 / 建议改 / 记录备查"三档排序，每条写清：改哪个文件的哪一处、改成什么、为什么；
  · **合并重复**（多路复审常报同一件事）；
  · 明确指出哪些发现**互相冲突**、以及你建议怎么取舍；
  · 末尾单列一节"代码里有、论文里没有"的清单（这是本次复审最重要的产出）；
  · 不要复述没被确认的东西。`,
  { label: 'synthesize', phase: 'Synthesize' })

return { 初审条数: all.length, 成立: confirmed.length, 被推翻: refuted.length, 清单: synth }
