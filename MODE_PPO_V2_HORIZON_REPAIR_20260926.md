# 二维模式条件 PPO：时间终点修复与续跑记录

日期：2026-09-26。此记录只涉及独立二维合成训练，不改变服务器三维实验。

## 故障与边界

用户要求从 v1 的 512 步检查点续跑。第一次后台续跑立即触发
`committed return exceeded execution budget`，进程退出，v1 状态仍为
512 步。复现时 map 15 的训练环境停在 `time_s=599.9507001226987`，
距离 600 秒终点只剩 `0.04929987730133689` 秒，少于一个 `0.05` 秒
物理步。原始训练环境把可执行步数算成 0，却没有推进到时间终点；
Gym 适配器的返站循环因而反复读取同一个状态。

修复只在可执行步数为 0 时将仿真时间置为 horizon 并结束 episode，
不执行额外飞行、不改变能量、位置、速度或碰撞规则。针对该余量的
回归测试与原有训练环境、Gym 适配器及动作掩码测试共 **7/7 通过**。
这是运行终止语义的错误修复，不能算作学习性能改进。

## 检查点迁移

保留 v1 原始目录不改动。由于训练环境源码 SHA 已改变，新运行使用
`artifacts/mode_ppo_v2_20260926/seed101/` 和
`dual_constraint_2d_mode_masked_ppo_v2_horizon_close` 协议。旧检查点
前没有触发该终点错误，因此复制了原始 512 步模型与 episode 记录；
`migration.json` 记录来源与目标 SHA。恢复本来就会重置 Gym episode
和随机流，不宣称逐位续接同一轨迹。

- v2 manifest SHA-256：`79afc0a4bba64dda75cfd42c954e44323186f0419d2f0587b7db53ac97eb49e2`
- 迁移记录 SHA-256：`40d59e2680f044877fd574abf995c8cdb240aa4af669aaa6018c011ff9df10a7`
- v2 的 1,024 步模型 SHA-256：`1fa5b5cb33487c8e2acf2f231633daaea1bade084dda505101af39db308472d4`

迁移后在 CUDA 上完成 512→1,024 步，写出新检查点。随后用户明确
授权续跑，seed 101 的剩余预算已在后台启动。运行 PID 与日志位于
同一输出目录的 `runner.pid`、`resume.log`；训练结束时会写
`status.json` 的 `complete=true`。这里只确认启动健康，不以早期训练
episode 评判方法。

单行查看状态：

`cd /home/zjl/uav_learning_research && cat artifacts/mode_ppo_v2_20260926/seed101/status.json && tail -n 8 artifacts/mode_ppo_v2_20260926/seed101/resume.log`
