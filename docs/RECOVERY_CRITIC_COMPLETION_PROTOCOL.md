# PPO 价值训练预算解耦实验

2026-09-06。用户授权深入全文调研并启动更长实验；启动仅做必要检查。

## 问题与证据

最近 actor-MC 对比已完成。确定性到达 MC24/50、GAE33/50，无碰撞到达
14/50、18/50；随机到达29/50、32/50，无碰撞到达18/50、20/50。
不继续扫 actor trace。

现有 `update_reward_only` 的 KL 超限会同时停止 actor 和 critic。配置的
10 epochs 是上限，实际完成量按 `gradient_steps / ceil(N/256)` 统计：

| 运行 | KL 停止批数 | critic 平均等效遍数 | 中位数 |
| --- | ---: | ---: | ---: |
| recovery PPO 32批 | 29/32 | 2.0583 | 0.9027 |
| GAE 对照16批 | 15/16 | 1.3896 | 0.6162 |
| MC actor16批 | 16/16 | 0.2065 | 0.1337 |

这是更新量事实；还不能据此认定它是导航失败的唯一原因。旧诊断发现
critic 相对整程 MC return 的 R² 仅 .15—.41；固定 lambda-return 的低训练
MSE 也不能替代未来新轨迹上的价值准确性。

## 文献依据和数学范围

全文重点阅读记录与本地 PDF：
`literature-search-20260906-ppo-critic-training/papers.md`。

- PPO §5 / Algorithm1 使用多遍 minibatch 更新；共享参数时描述联合损失。
- PPG §2、§3.2—3.3、补充C/D分别研究 actor/value 的数据复用量。其视觉
  特征蒸馏不是本轮方法；Procgen 的收益不直接作为无人机效果证据。
- Spinning Up 官方 PPO 源码中 policy 的 KL break 与后续 value loop 分开。
- CUP 附录C Algorithm1同样分别列出策略与价值拟合；FOCOPS Algorithm1
  则保留了共同内循环/停止，说明不存在所有论文一致的实现惯例。

设固定轨迹终点 T 的 value 为0，gamma=1，lambda=.95。对于两个价值函数
V 与 V*，令 e_t=V(s_t)-V*(s_t)，二者用相同奖励构建 GAE，则有限和恒等式为

    A_t(V)-A_t(V*) = -e_t
      + (1-lambda) sum_{j=1}^{T-t-1} lambda^(j-1) e_{t+j}.

证明：把 delta_t(V)-delta_t(V*)=e_{t+1}-e_t 代入 GAE，并对相邻项消去，
使用 e_T=0。若所有 |e_t|<=epsilon，差异不超过2epsilon。
这是已知 GAE 的代数性质，不是新定理，也不是本轮网络已满足的一致误差界。
它解释了 value 误差会进入后续 actor 更新；多拟合标签不必然缩小真实 e_t。

本轮唯一干预：原联合 PPO 更新完整执行原代码；若 KL 提前停止，只追加
critic-only SGD，直到 critic 总步数达到 `10*ceil(N/256)`。这是10遍等效更新
预算；补充阶段重新独立打乱，不能声称每条样本恰好使用10次。
actor 在追加阶段无梯度且参数保持逐位相同。critic 与 actor 的编码器本已
分离，不添加网络。共享 Adam 对没有梯度的 actor 参数不更新其动量/参数。

## 固定设置

- 全部使用用户锁定的 RecoveryCohort；碰撞修复、速度清零、固定-0.42连续
  惩罚和-1.2首次惩罚遵守锁定协议。
  接触不结束任务；只有统一碰撞计数，cost按接触策略步取0/1。
- 结构化128×8双通道LiDAR、剩余时间输入、网络、gamma=1、GAE=.95、
  Adam3e-4、batch256、value系数.5、clip.2、KL阈值.03、奖励×.01均保留。
- 未启用cost loss或dual。这是安全导航所需的reward-PPO训练机制实验；
  不把惩罚奖励当作约束安全保证，更不报告能耗研究完成。
- 两个随机流重复各有两个分支：critic_complete 与 joint_stop_control。
  四支均从原control checkpoint64开始，恢复同一模型和Adam状态；它们是
  两个续训随机流，不是两次独立从零初始化。
- 每分支96个完整cohort×8任务=768训练任务（比上轮每支128任务增加6倍）。
  每组最多4000步，实际交互量受任务表现影响，单支上限3,072,000新步。
- 同一重复内场景seed与采样/打乱的主RNG seed相同。重复1场景从813600001
  开始；重复2从813700001开始。训练RNG按cohort固定，不受中途评估影响。
  额外critic阶段使用自己的局部NumPy随机源。

## 评估与解释

- 追加32/64/96批时，各做50个固定开发场景确定性评估，仅记录曲线。
  任何结果不触发停训、换配置、挑checkpoint或自动升档。
- 最终checkpoint160：每分支在793800001..793800500共500个新场景做
  确定性测试，再做50个已有开发场景随机评估。以最终checkpoint为主结果。
- 共四分支、3072训练任务、2000次新场景测试和800次开发评估。四支共用
  500个最终场景，因此不能把2000次测试当成2000个独立场景。
- 核心指标：到达率、无碰撞到达率、超时、平均统一碰撞数；同时保存原始
  回报、能耗记录、路径比、actor/critic更新数和固定标签MSE。
- 事后按场景做配对比较并逐重复报告；两个训练随机流不足以支撑广泛结论。
- 若固定标签MSE降低但任务指标仍差，则此修改没有解决导航问题。下一步
  优先处理采样/初始化，不能据此继续无上限增加critic epochs。

## 运行与恢复

预计四分支训练和评估合计约4—7小时，实际随回合长度和机器负载变化。
每cohort保存模型、Adam、RNG及日志；暂停信号或根目录PAUSE在边界停下。
评估按小组保存，恢复跳过已保存场景。原保存/恢复函数复用，新优化器路径
有聚焦恢复测试；不额外跑一条恢复性能gate。无硬时限、无自动升档。

启动命令：

```bash
.venv/bin/python scripts/run_recovery_critic_completion.py \
  --source artifacts/directional_lagrangian_slowdual_pair_20260905_v1/control/checkpoint_0064 \
  --output-dir artifacts/recovery_critic_completion_20260906_v1 \
  --cohorts 96 --replicates 2 --eval-tasks 500 --dev-tasks 50
```

续跑使用相同命令加 `--resume`；存在PAUSE时先由启动操作移走该标记。
确认首个checkpoint有限、actor追加阶段不变和进程运行后，交接等待用户。
