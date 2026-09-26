# 二维全图解析对照：用户授权续跑

日期：2026-09-26。用户在阅读首检查点状态后明确要求继续推进。已从 map_id=0、full 补能变体的 1 决策检查点恢复，启动 600 仿真秒同一运行；后台进程 PID 83205，原始 stdout/stderr 写入 [runner.log](../artifacts/dual_constraint_2d_fullmap_baseline_20260926/map000_full/runner.log)。

仅做一次续跑启动健康核查：可恢复[状态文件](../artifacts/dual_constraint_2d_fullmap_baseline_20260926/map000_full/status.json)已前进至累计 21 决策、10.5 仿真秒；电量 72.19778013625444，统一碰撞 0，安全代价 0，能量过滤事件 0，完成目标 0；进程当时正常运行。这个检查点不能解读为 600 秒正式结果。

运行器每新增 20 决策写原始事件与检查点，结束时写 summary.json。根据仓库固定工作流，启动健康确认后交还运行，不监控到完成，也不自动启动其它地图、部分补能变体、MPC 或训练。后续读取状态应以当前 status.json 和 summary.json 是否存在为准，不能把这里的启动快照当成实时状态。
