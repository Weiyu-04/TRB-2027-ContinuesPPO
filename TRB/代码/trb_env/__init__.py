"""
TRB RL 环境包（连续动作空间 COLREGs 合规避碰）。

逐件搭建，每件 fact-based 对论文公式 + 写冒烟测试 + 多轮独立 agent 审核，过了才进下一件。
模块清单（按搭建顺序，均带 usv_ 前缀 = 这是 USV 仿真环境的组件）：
  - usv_dynamics.py     ✅ 本船动力学（偏航受限模型 eq.1，直接用 Krasowski 官方 vesselmodels）
  - usv_observation.py  ✅ 27 维观测（ObservationBuilder）
  - usv_reward.py       ✅ 奖励 eq.10（RewardFunction，5 分量；r_colregs 尺度待 step4 标定）
  - usv_termination.py  ✅ 5 终止条件（TerminationChecker）
  - usv_env.py          ✅ gym 接线 USVEnv（reset/step；step2 完成，环境能跑 episode）
  - usv_colregs.py      ⏳ COLREGs（step3）：(a.1) 几何原语 ✅ / (a.2) 态势分类 + (a.3) 状态机 ⬜
对应论文事实见 `../../参考资料/文献核实笔记.md`。
"""
