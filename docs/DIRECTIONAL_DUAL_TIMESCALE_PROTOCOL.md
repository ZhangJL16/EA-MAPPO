# PPO-Lagrangian：dual时间尺度单因素验证

日期2026-09-05，在启动前冻结协议。不是新增算法或能耗正式实验。

启动后状态记录：后台进程组202678、trainer202681。初次健康检查完成首组2174步，
λ=.0095，checkpoint0001已保存，更新有限，无error.json。24项相关测试通过，
Ruff/compile检查通过，指定项目解释器后Pyright0错误。以下实验设计未因运行结果改动。
后台日志`artifacts/directional_lagrangian_slowdual_pair_20260905_v1.background.log`。
后续不持续监控，由用户请求时再读取结果。

最终状态补记：已完成2668.36秒。slowdual确定性26/13/11、随机22/16/12，
对照25/24/1与23/26/1；顺序安全到达/接触/超时。恢复导航但未达标。
后续用户授权的超时研究见`docs/DIRECTIONAL_COST_TRACE_PROTOCOL.md`，不是本轮自动触发。

## 证据与假设

原64组比较中，约束组0/50安全到达，对照25/50。冻结重放在约束组第16/32/64
更新前测到加权成本/奖励梯度范数10.8/18.9/43.0。对应成本估计当组R²<0。
假设：dual增长过快，放大尚不可靠的成本梯度，阻碍导航学习。
证据不排除稀疏标签、GAE/MC差异、部分可观测性或CMDP本身难以满足。

依据：[Stooke等ICML2020 §7](https://proceedings.mlr.press/v119/stooke20a/stooke20a.pdf)
说明奖励单位缩放要求乘子及controller步长考虑相应缩放。
本任务reward_scale=.01，选择dual_lr=.01作为相对旧1慢100倍的有依据干预，
不是声称.01是数学最优，也不是复现该论文的全部PID算法。

## 唯一改变

- 原脚本及源文件完全不改，只使用已有`--dual-lr 0.01`参数。
- 不改reward_scale=.01、cost_limit=.05、γ_C=1、MC成本估计、奖励GAE=.95。
- 保留结构化全LiDAR、同网络、同两价值头、同physics与first-contact终止。
- 从相同随机初始化开始，不续训已失败的高乘子分支。
- 同64×8登记训练任务；控制组也复跑，检查算法以外的复现漂移。
- 两组真实步数、梯度次数、时间单独记录，不宣称等步数。
- 此次及上次唯一意图干预为dual_lr；上次结果是已查看的开发比较。

## 执行

```bash
uv run --no-project --python .venv/bin/python python scripts/run_directional_lagrangian_pair.py \
  --output-dir artifacts/directional_lagrangian_slowdual_pair_20260905_v1 \
  --cohorts 64 --eval-tasks 50 --dual-lr 0.01
```

lagrangian和control顺序跑，每组保存每个完整cohort的checkpoint/optimizer/RNG。
评估确定性50、随机50；与原比较同seed，非新独立最终测试。预估30—60分钟，
按任务数完成，不设破坏续跑的墙钟kill；若策略接近期限上限会更久。
仅检查启动、首批checkpoint和有限数值，随后交给用户，不监控后续成绩。
暂停仍用PAUSE或SIGTERM，等待当前完整cohort保存；续跑同命令加`--resume`。

## 解释与停止边界

1. 控制组须核对旧控制的初始权重、训练记录和最终表现；明显漂移先排查复现。
2. 主要目标是检验能否解除0成功的退化，且接触变化必须同时报告。若只恢复
   导航但碰撞仍接近控制，则只支持“降低干扰”，不能称为安全研究成功。
3. 碰撞减少但超时增加、到达减少，也不算通过。继续保持三类结果互斥统计。
4. 若仍几乎无法到达，不继续扩大预算或网格扫dual_lr；优先研究成本优势估计
   和安全探索覆盖。GAE(.95)可成为下一项独立干预，但不会和本轮混改。
5. 即使明显改善，也需新的训练seed、未查看场景及正式评价；不自动500k/500测试。
6. 本轮结束程序退出。无自动后续训练，无安全保证或论文创新声明。
