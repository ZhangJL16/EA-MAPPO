# 完整原始窗口筛查：启动交接

日期：2026-09-29。预定 `27×10×3=810` 个独立完整 `T` 回放，三个固定方法分别为最近任务且可补能就补能、FIFO 且可补能就补能、原 B5 reserve-SJF。新 seeds、目标量、同信息规则与禁止的推断见 `FULL_WINDOW_PILOT_PROTOCOL.md`。不是训练，也不是 5% 风险认证。

聚焦测试 6/6 通过。新执行路径的真实导航 10 秒 smoke 完成（50 policy steps）；一条原始完整 `T` job 保存并通过 collector，随后同分片续跑成功跳过旧行并保存下一条。8 worker 的正式可续跑会话已启动，第一次统一健康检查 `deliverables/full_window_startup_health.json` 严格校验 **20/810** 条，`complete=false`；worker 日志没有 Traceback/ERROR/Exception。检查时 `free -h` 显示约 17 GiB 可用 RAM、0 Swap 使用。首个样本曾耗尽，但 20 条只用于启动完整性，不做风险或吞吐判断。

运行输出在 `runs/full_window_pilot_v1/`，manifest 固定 runner、原校准与环境 provenance；每个完整 episode 的行和事件轨迹原子保存，分片文件锁防止重复 worker。运行中不得再次启动相同分片。查看校验状态的一行命令（在本项目目录）：`MPLCONFIGDIR=/tmp/mpl-uav-feasibility /home/zjl/mappo/.venv/bin/python checks/collect_full_window_pilot.py`。如会话中断，先确认旧 worker 停止，再用 `MPLCONFIGDIR=/tmp/mpl-uav-feasibility /home/zjl/mappo/.venv/bin/python checks/start_full_window_pilot.py --workers 8 --wait` 续跑。collector 只在全部 810 行通过时给 27×3 格汇总；未完成时不输出方法胜负。

按根 `AGENTS.md`，已完成首 checkpoint 健康检查后交接可续跑运行；不在本轮监控到完成，不自动启动认证、学习或其他长实验。
