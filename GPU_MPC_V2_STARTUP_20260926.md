# GPU-MPC v2 启动记录

2026-09-26 12:24:53 CST 启动，预设截止 20:24:53 CST。运行目录：
`artifacts/gpu_mpc_8h_campaign_v2_20260926/`，计划 48 个配对作业。

此前的 v1 运行因终点剩余时间短于物理步而将 `hold_steps=0` 送入安全证书，
于第一张验证图中断。v1 原始记录保持原样。v2 的源码修复、两个旧源码 SHA、
56 个 CPU 验证作业以及 GPU 断点复制来源，见
`HORIZON_GUARD_MIGRATION_20260926.json` 和两个 v2 目录的
`migration_record.json`。

启动前 4 项针对性单元测试通过。GPU H4/map 32 从旧的 560 决策断点恢复，
补跑 20 决策后于 600 仿真秒正常结束：10 个目标、统一碰撞数 0、耗尽 0。
旧事件日志全部作为 v2 日志的逐字节前缀保留。

首个**新**作业 H8/map 32 已保存 40 决策检查点：20 仿真秒、1 个目标、
统一碰撞数 0、耗尽 0、无错误文件。启动健康检查通过，随后交还自主运行；
这里不声称 48 个作业已经完成，也不自动把验证结果提升为测试结论。

本次使用 `/home/zjl/mappo/.venv/bin/python`，其 PyTorch 2.7.1+cu128
可见 RTX 5060 Laptop GPU。独立项目原 `.venv` 缺少 `torch` 和
`stable-baselines3`；解释器路径和依赖版本均由矩阵 manifest 记录。

查看运行：`artifacts/gpu_mpc_8h_campaign_v2_20260926/status.json`；
每条轨迹保留 `checkpoint.pkl`、`events.jsonl`、`summary.json` 与 `run.log`。
