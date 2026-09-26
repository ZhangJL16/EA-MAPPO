# 二维 PPO v4：长时信用分配训练启动记录

日期：2026-09-26。seed 101，γ=1，GAE λ=1，51,200 高层决策步预算；
训练图 8–31，CUDA 设备。此轮与已冻结 v3 只改变 λ。协议见
`MODE_PPO_V4_LONG_CREDIT_PROTOCOL_20260926.md`。

- 相关测试 8/8 通过，新增训练和评价入口静态编译通过。
- 第一份 512 步检查点存在且可继续；核查时运行状态为 1,536 步、
  5 个完整 episode，进程 PID 108811 仍运行。
- manifest 为 γ=1、λ=1，全部源码 SHA 与当前文件一致；CUDA 设备
  上有该训练进程。尚未进行验证图评价，不将启动轨迹当作效果证据。
- manifest SHA-256：`4fc012ecc2c671b2f33971628e7df162dcd46599d85fbd3d9ac1635aa628aa5f`。
  512 步模型 SHA-256：`d71b253ef85c4153bb6959afa5dd5367e254d615841c7a5e0f575a35cb773eb9`。

输出目录为 `artifacts/mode_ppo_v4_lambda1_20260926/seed101/`，
训练会按 512 步间隔持续保存；若中断，以相同训练命令恢复。查看当前状态
的一行命令：

`cd /home/zjl/uav_learning_research && cat artifacts/mode_ppo_v4_lambda1_20260926/seed101/status.json`

按当前仓库的实验启动约束，在确认首检查点健康后交还运行；不监控到完成，
也不自动启动 seed 202/303 或验证图。
