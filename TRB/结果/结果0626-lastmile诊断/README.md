# last-mile 根因诊断产物（`03` L88②·2026-06-25/26）

> 修法A(well_B=200) 后连续 PPO 臂【残余失败】根因的数据钉死。供新窗口二审/复现。

## 文件
- `reeval_lastmile.py` —— 重评脚本（eval-only·本地·不训练）。对现有 10 个 wb200 checkpoint（`../结果0625-奖励改造第2次/checkpoints/`）用 `PPO.load`（**非 load_sac_for_eval·对 PPO 崩**）+ `evaluate_continuous(traj_idxs=all)` 重评 60 测试场景，用 faithful 法（Polygon 顶点 + AngleInterval 角差·== 官方 `is_reached`·见 `代码/tests/test_usv_evaluate.py` ㊳d/e）把每局失败分解成 位置门/朝向门/时间门 miss。
- `reeval_wb200_fails.json` —— 69 个 wb200 失败 episode 明细（seed/末点 px,py,psi,v/term_flags/in_pos,in_ori,in_t/goal_geom）。

## 复现（新窗口二审·~5min·本地·eval-only·不烧训练）
```bash
cd 代码 && /opt/miniconda3/envs/trb/bin/python -B ../结果0626-lastmile诊断/reeval_lastmile.py
# 需 commonocean 本地可用 + 能从 gitlab 下 60 测试场景(~2min·已缓存则秒过)
```

## 结论（钉死·faithful 法自检 mism=0=分解==官方 is_reached）
- **97% 失败 = f_stopped**（终端 v→0 刹停·中位 0.000）·碰撞 ~0·超时 ~2/69。
- **94% 位置门 miss**（船到正确 x·刹停在 60m 窄 y 带外·|e_cross|=32-44m 到中心线=y带缺口 2-14m+带半宽 30m）。
- **59% 朝向门 miss**（兼朝向偏出 ±9.7°）·**0% 时间门 miss**。仅位置外 28 / 仅朝向外 4 / 都外 37。
- 目标区全 69 同几何：400(x)×60(y) 矩形·θ_c=0·朝向门 ±0.17rad·时间门 0-170。
- **根因 = 终端横向(cross-track)进窄带失败·非奖励吸引子**（reward 已重罚停车）→ 对症 = Φ_xtrack 横向进带势（`03` L88⑥）。

⚠️ 这批 checkpoint 在 `../结果0625-奖励改造第2次/checkpoints/`（wb200_s0..4）。重评只读、不改它们。
