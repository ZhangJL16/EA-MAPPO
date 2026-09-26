# 二维 PPO v4：GAE λ=1 单变量诊断

日期：2026-09-26。完整预设见
[`refine-logs/EXPERIMENT_PLAN.md`](refine-logs/EXPERIMENT_PLAN.md)。
v3 γ=1/λ=0.95 的 seed 101 在验证图 32–39 只完成 1 单，并在 map 36/39
反复离站、返站、充满。原始训练环境重放也出现该循环，故不是安全层单独
制造的动作模式。

本轮只把 GAE λ 从 0.95 改为 1.0；γ=1、固定时窗 terminal、24 张训练图、
8 张验证图、51,200 高层步、seed 101、13 动作掩码、网络与 PPO 其余参数、
奖励、碰撞语义和共同评价安全层均与 v3 相同。主假设是更长的资格迹
能改善延迟目标完成奖励的信用分配；λ=1 可能增加方差，不保证改善。

训练输出固定为 `artifacts/mode_ppo_v4_lambda1_20260926/seed101/`，
每 512 步保存模型和状态；只用 51,200 步最终模型评价，不选中途最优。
启动与续跑命令均为同一行：

`cd /home/zjl/uav_learning_research && python3 -m learning2d.train_long_credit_ppo --output artifacts/mode_ppo_v4_lambda1_20260926/seed101 --seed 101 --total-timesteps 51200 --n-steps 512 --checkpoint-steps 512 --device cuda`

完成后验证图 32–39 用 `learning2d.evaluate_long_credit_ppo`，并以
`analysis2d.raw_frozen_replay` 重放 map 34/36/39。只有两个零单循环均消失
且 8 图总完成数至少达到旧 v2 seed 101 的 7 单，才考虑扩大种子；
这仍不是胜过路线 61 单或 H4 68 单的最终研究主张。
