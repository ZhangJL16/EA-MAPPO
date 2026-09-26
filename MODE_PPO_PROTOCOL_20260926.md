# 二维模式条件 PPO 对照协议

日期：2026-09-26。研究范围仅为独立合成二维项目；服务器三维实验不受影响。

## 要检验的问题

旧 PPO 的 13 个动作在站内和空中有不同含义：站内 0–9 均为出发，空中
10–12 均被解释为飞行动作。旧 PPO 在 24 个验证运行中零次主动充电，
且站内出发被安全层拒绝 25,651 次。事后站内监督器将平均完成数从
1.208 提高到 3.708，但不属于学到的行为。此对照只改变策略的动作
支持集，检验消除别名后 PPO 能否自行学习充电与适时返站。

## 冻结比较

- 环境、能耗、碰撞、返站安全层、目标生成、奖励、网络宽度、PPO 超参数
  均沿用旧 PPO；只有 `ModeMaskedPolicy` 改变动作支持集。
- 训练图 8–31，seed 101/202/303，每个 seed 51,200 个高层决策步。
  验证图 32–39；新确认图预留 56–71。验证期间使用同一执行安全层，
  每图 600 秒仿真时长。结果以图为配对单位，报告每个 seed 的原始值。
- 空中允许动作 0–9；站内允许动作 4，以及当前电量严格低于目标电量
  的 10/11/12。掩码只编码物理动作语义，不替策略判断任务能量可行性。
  原始训练仍允许耗尽、碰撞修复与试错。
- 主指标为每 600 秒完成目标数；同时报告碰撞、实际耗尽、安全层接管、
  站内拒绝出发、显式充电次数、空中主动返站次数、运行成本。不得仅凭
  训练启动检查点或单个 seed 声称改进。
- 主要比较对象为旧 PPO、旧 PPO 加非学习站内监督器、`route_full` 和
  已冻结的 GPU-MPC H4/H8。后两者拥有不同规划预算，应明确标注。

## 可恢复执行

训练程序 `python3 -m learning2d.train_mode_ppo` 在 `artifacts/` 下写入
manifest、逐 episode 记录、模型和状态文件。使用相同参数及输出目录
再次执行即可继续；恢复不保证逐位等价，因为 Gym episode 和随机流在
重启时重置。`--max-new-timesteps 512` 只运行第一个检查点。

单行启动命令示例：

`cd /home/zjl/uav_learning_research && python3 -m learning2d.train_mode_ppo --output artifacts/mode_ppo_v1_20260926/seed101 --seed 101 --total-timesteps 51200 --n-steps 512 --checkpoint-steps 512 --max-new-timesteps 512 --device cuda`

单行续跑命令示例：

`cd /home/zjl/uav_learning_research && python3 -m learning2d.train_mode_ppo --output artifacts/mode_ppo_v1_20260926/seed101 --seed 101 --total-timesteps 51200 --n-steps 512 --checkpoint-steps 512 --device cuda`

验证入口 `python3 -m learning2d.evaluate_mode_ppo` 会拒绝未完成训练的
模型，并为每张图记录模型、训练 manifest 和评价源码的 SHA。正式验证
前先冻结全部三个训练 seed；不得用验证图挑选 seed 或改超参数。
