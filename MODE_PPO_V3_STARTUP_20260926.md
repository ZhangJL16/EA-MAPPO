# 二维固定时窗 PPO v3：主条件首检查点

日期：2026-09-26。`gamma=1.0`、seed 101、预算 51,200 高层决策步。
只完成 **512 步首检查点**，未启动后续长训练、验证图或确认图。

- 固定 600 秒内任务终点改为 `terminated=True`；本机 SB3
  `DummyVecEnv` 返回 `TimeLimit.truncated=False`。该变更只在新的
  `FiniteHorizonGym` 中实现，旧评价/物理环境未改。
- 相关测试 **10/10 通过**；64 步短训练在 32 步中断后可恢复到
  64 步，新评价入口能执行一个决策步。完整训练恢复仍不保证随机流
  逐位等价。
- 实际 512 步模型可在 `cuda:0` 重新加载，`gamma=1.0`。目前 1 个
  完整训练 episode（map 15、600 仿真秒、0 个目标、25 次显式充电、
  0 碰撞）。这个样本不能评价策略效果。
- 检查点位于 `artifacts/mode_ppo_v3_gamma1_20260926/seed101/`；
  manifest SHA-256：`f92c8f607627901467a2a17dc21a0f66416bce7061e5d60d1ff0264ec1e4cbab`；
  512 步模型 SHA-256：`44772c110cdffd529bedb0a52d67329b90ab703d1d9b98b4150c12f55b287493`。

按仓库实验启动约束，确认首检查点健康后不自动推进剩余长训练。
同参数同输出目录去掉 `--max-new-timesteps 512` 即可续跑；单行命令
见 `MODE_PPO_V3_FINITE_HORIZON_PROTOCOL_20260926.md`。
