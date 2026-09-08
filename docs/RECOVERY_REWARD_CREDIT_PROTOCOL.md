# 恢复导航的整程 actor 信用检验

2026-09-06，启动前预注册。用户锁定的碰撞处理、奖励与统一计数不变。

## 依据与假设

恢复PPO在同50开发场景显著退化；32个训练组无记录错位。对cohort
65/81/89/96精确重放32任务，结果、步数、碰撞次数和raw return全部匹配。
GAE(.95)与完整回报优势的相关仅.324/.338/.206/.253；对应actor梯度余弦
.335/-.018/.207/.551。价值相对MC return的R²为.257/.206/.413/.149。
这支持“长任务的短迹信用可能不可靠”作为待检假设，不证明MC更优。

对完整任务，令V_T=0，完整回报和MC优势为

    G_t=sum_(k=t)^(T-1) r_k,    A_t^MC=G_t-V(s_t).

它避免GAE(.95)依赖未来不精确V的bootstrap，但方差通常更高。依据PPO原论文
的裁剪目标和GAE的bias-variance关系，本轮只替换actor中的reward advantage。
reward critic仍用原GAE(.95)lambda-return标签；网络、optimizer、value loss、
PPO clip、KL规则、epochs与所有采样条件均不改。因此检验的是actor信用，
不是“MC版PPO”整体替换，也没有安全定理或收敛保证。

## 配对设计

- 共同源：原control checkpoint64；复制相同模型、optimizer和RNG。
- GAE对照trace=.95保持原actor advantage数组逐位不重算。
- MC干预trace=1，由GAE递推恒等式恢复逐步奖励，逐回合反向累加；每条
  回报和日志raw return×.01核对，critic标签对象保持不变。
- 两支各追加16组×8完整任务，cohort64→80；seed483600513..640。
- 最后每支det/stoch各50个相同开发任务；不是正式500或新训练seed。
- 每组保存模型、optimizer、RNG和任务记录；SIGTERM/PAUSE在组边界恢复。
- 预计两支训练和200评估约20—60分钟，不自动延长或升档。

主要看总到达、无碰撞到达、超时和统一碰撞次数。MC若优于GAE，只支持
该起点/预算下整程actor信用更合适；若二者都退化，则优先改为标准固定步
rollout或处理迁移初始化，而不是继续扫trace。任何一支都不能因训练任务
表现或单一开发fork自动成为新基线。
