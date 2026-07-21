# M1 停车模块 WIP（`03` L178+**L179 订正**·2026-07-12·**未集成·未定论·健康种子停短有效/崩种子绕圈无力**）

停车入库控制器实验件（新窗口接手用）。**不是生产代码·未进 trb_env**。诚实状态见 `02` banner + `03` L179（订正 L178）。

## 🔴 L179 订正（新窗口两阶段独立复审·派对抗 agent）
- **L178 D④"停车模块不够有效"=以偏概全**：那是【只测崩种子 s6】（高速绕圈=停车最弱场景）得 0/13→2/13。全 10 种子闭环重跑证：**对健康种子【停短失败】高度有效**（s2/s8 **7/13→12/13**·s7 7→13），只对崩种子【绕圈】无力（s5/s6 0→2-4）。失败模式交叉表：停短救 72%·绕圈救 28%。**准确框架=停车把健康种子抬到接近可用·没真修好崩种子**。
- **L178 D③"停车用±0.018·无confound"=WRONG**：闭环 harness 实调 dock_controller 默认 **wmax=±0.03**（1.67×RL），且加速度硬编 **A_MAX=0.24=RL箱5×**（无 amax 形参）。**但 confound 结论(不靠多给权限)HOLDS**：combined-cap（omega+accel 同时限 RL 箱 ±0.018/±0.048）→ s2/s8 照样 12/13 → 救援真在控制逻辑（速度调度避 f_stopped 早终止+蠕行对齐）。
- **修法待做**：dock_controller 加 amax 形参默认 A_NORMAL_ACCEL_MAX；集成两维都限 RL 箱+显式披露。

## 文件
- `dock_controller_v4.py` — 停车控制器（纯追踪进近走廊+速度调度+冲过门掉头恢复）。开环诚实:门西82-94%/冲过头16%/平滑vs捕获权衡。⚠️开环网格会骗人（合成冷启网格≠真实闭环状态分布·闭环才是真）。**L179 已把开环网格挪进 `if __name__=='__main__'`**（否则 import 触发 3600 rollout 副作用）。
- `closed_loop_dock.py` — 闭环机制验 harness（load 策略·真环境有盾·近门无冲突停车接管·测到达率）。**⚠️现用任意 glob 取40≠训练eval那40=数字不可比·服务器跑前须改用 manifest 测试集拆分**（复用 run_step4e.py:568 `load_manifest_split`）。本机13场景=脚本 sanity；真数须服务器40测试集。
- `openloop_dock_feasibility.py` — 早期开环可行性探针（留史）。

## 跑
- 网格自测：`PYTHONPATH=代码 python 代码/m1_dock_wip/dock_controller_v4.py`
- 闭环（本机sanity）：`PYTHONPATH=代码 REPLAY_CKPT=<ckpt无后缀> SCN_GLOB='<场景glob>' python 代码/m1_dock_wip/closed_loop_dock.py`
- 服务器真数（待改 harness+user 拍）：全10种子×40测试集×两臂·dock 限 RL 箱·出逐种子 base vs dock+失败模式分类。
