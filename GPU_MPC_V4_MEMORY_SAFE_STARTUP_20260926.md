# 8-worker GPU-MPC v4 启动与内存记录

2026-09-26 12:58:14 CST 从已停止的 v3 断点启动；8 小时墙钟上限
2026-09-26 20:58:14 CST。运行目录：
`artifacts/gpu_mpc_parallel_v4_scale_20260926/`。

用户曾要求先关闭长运行、验证速度和 24 GB RAM 余量后再运行。
v3 停止记录为 `artifacts/gpu_mpc_parallel_v3_20260926/user_stop_record.json`；
v4 单独保留新 manifest、来源迁移清单和每作业断点，未覆盖 v3。

短时吞吐基准：8 张图各 20 决策，1 worker 90.36 秒，8 workers
14.90 秒（6.06 倍）。16 图测试中，16 workers 比 8 更快，但
峰值独占 RAM 为 12950 MiB，系统最低可用 RAM 仅 8346 MiB；
8 workers 对应 6477 MiB 与 14845 MiB，Swap 均为零。
因此本机 v4 默认/最高 8 workers，RAM 总量低于 32 GiB 时拒绝
超过 8 workers；启动前还要求 `MemAvailable >=12 GiB`。
原始测量：`profile_output/scale_benchmark.json`、
`memory_benchmark_8.json` 和 `memory_benchmark_16.json`。

只改变独立地图作业的调度并发；`gpu2d/`、`dual_constraint_2d/`、
`nav3d/` 的实验源码与 v3 相同。8-worker 短测中对应事件日志
逐字节相同。独立 v4 候选的路线去重/几何快速判断暂留在
`profile_output/v4_candidate_only.patch`，**没有应用到本次正式运行**。

启动时复制 v3 的 8 个已有作业目录，其中 5 个完成、3 个部分或未开始；
迁移 SHA 在 v4 `seed_migration.json`。首检查点时 8 个作业活跃，
H4/map 35 和 H8/map 34 已各新增至少 40/60 个决策，事件日志仍
完整保留 v3 字节前缀。新开始的 test/map 40–42 作业各有 40 决策
检查点；无错误文件。检查时系统可用 RAM 约 14 GiB，显存占用
约 2 GiB，Swap 为零。该检查只证明启动健康，不是完整结果。

后续仅从各作业 `summary.json` 和 v4 `status.json` 判断完成状态；
本次不监控到结束、不自动启动其他训练或实验。
