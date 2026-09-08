# 恢复导航 PPO 基线续训：2026-09-06

状态：启动前预注册。本实验不是新安全算法或能耗训练。
碰撞机制严格服从 AGENTS.md / LOCKED_COLLISION_RECOVERY_PROTOCOL.md。

## 证据与最小问题

冻结评估200任务完成（580.02s）。随机动作下原对照到达36/50、无碰撞
23/50、平均碰撞704.02次；慢乘子到达27/50、无碰撞22/50、平均488.76次。
这两个旧模型均未在碰撞后继续的任务中训练。不能由此得出新环境不可学，
也不能归因于奖励衰减；本轮不修改任何奖励项来追求指标。

原collect_cohort的terminal_gae(costs, vc, 1)实际上已经给出正确的累计
成本标签。真正不兼容的是按是否曾碰撞更新的概率乘子、cost_trace中
“只在最后一步碰撞”的重构，以及成本价值必须在[0,1]内的诊断。
尤其旧control并非纯PPO：虽然乘子为0，仍有二元成本value辅助损失。
把新上千次碰撞直接当标签继续该辅助损失，可能改变共享reward critic的
表示和梯度规模；本轮直接不训练、不使用这个旧成本头，不假称它已经是
新计数critic。旧头留在checkpoint中只为权重格式兼容，不增加任何网络。

## 当前与未来的数学问题

固定每步c_t=1[任意碰撞]，真实统计C=sum(c_t)，无折扣、无类别拆分。
若未来做CMDP，应明确E[C]≤d的计数量纲；旧E[是否碰撞]≤.05与之不等价。
本轮暂不自行设定用户未指定的安全预算d，而建立reward-only PPO基线：

    maximize E[sum_t r_t]
    loss = -mean min(ratio * A_R, clip(ratio,.8,1.2)*A_R)
           + .5 mean(V_R - target_R)^2

r_t含用户锁定的碰撞惩罚，训练统一乘.01，与此前相同；γ_R=1、GAE=.95。
碰撞转移有下一状态，不能在碰撞处清零bootstrap；只有到达/4000步期限
清零。完整任务组采样不增加自动重置任务。MC计数标签只用于一致性检查：
G_t=sum_(k=t) c_k，可>1；检查G_0=总次数，G_t-G_(t+1)∈{0,1}。
成本统计照常保存，但不参与本轮损失；这不是约束满足保证。

依据：[PPO原论文](https://arxiv.org/abs/1707.06347)使用裁剪替代目标进行
小批次更新；[OmniSafe PPOLag官方实现](https://raw.githubusercontent.com/PKU-Alignment/omnisafe/main/omnisafe/algorithms/on_policy/naive_lagrange/ppo_lag.py)
用回合累计成本更新乘子，并组合reward/cost优势。本轮只建立前者对照，
不声称复现PPOLag、获得安全证明或提出定理。

## 固定实验设计

- 源：directional_lagrangian_slowdual_pair_20260905_v1/control/checkpoint_0064。
- 复制actor/reward value、优化器、RNG；保持已验证CuDNN确定性设置。
- 旧cost_net不参与loss，其参数必须保持不变；不重置已有导航能力。
- 追加32组，每组8个完整任务（256任务），cohort64→96。
- 训练seed483600513..483600768，24障碍、5米到达、4000策略步期限不变。
- Adam学习率3e-4，PPO clip .2，最多10epochs，batch256，KL>.03停止当组
  后续梯度更新；不是停止整个训练。保留原优势归一化和梯度裁剪.5。
- 每组保存权重/优化器/RNG及原子任务记录。组边界可暂停、可续跑。
- 最后确定性50+随机50，开发seed593800001..593800050，按完整组保存。
- 对照是相同RecoveryCohort下已完成的冻结control，不再跑首次碰撞环境。
- 预计约20—60分钟，以首组实测速率为准；最多新增1,024,000实际策略步。
  不自动升档500k/长训，不设破坏恢复的硬墙钟kill，不监控到实验结束。

## 判读与局限

主要报告随机动作下总到达、无碰撞到达、统一平均/分位数碰撞次数及超时，
确定性补充。不能仅凭到达提高就宣布安全，也不能仅凭平均次数下降忽略
停滞或少数极端轨迹。全体任务参与统计，不只统计成功任务的碰撞数。
若改善，只能支持“在正确恢复任务上续训可改善该起点”，不能分离环境
转移改变与移除旧辅助损失各自贡献，更不能宣称已超过约束RL方法。
若无改善，下一步分析真实计数cost-to-go及reward策略梯度的冲突，而不是
修改用户碰撞协议。单训练起点、已查看开发场景不支持普遍或正式500结论。

## 启动、恢复与历史澄清

原始R3 SAC（历史提交7e30d8f）明确collision_terminal=False，并非首次碰撞
终止。首次碰撞终止来自后续RACT及directional PPO；本协议所说“旧模型”
指本轮两个PPO模型，不包含原始R3。R3还包含HOCBF/bridge等机制，不能把
其表现差异只归因于终止条件。本轮不移植或堆叠这些模块。

```bash
uv run --no-project --python .venv/bin/python python scripts/train_recovery_ppo.py \
  --source artifacts/directional_lagrangian_slowdual_pair_20260905_v1/control/checkpoint_0064 \
  --output-dir artifacts/recovery_ppo_baseline_20260906_v1 \
  --extra-cohorts 32 --eval-tasks 50
```

同一命令加--resume可恢复，SIGTERM在组边界保存后停止。原始PPO/R3源码
与checkpoint不覆盖。训练日志记录真实额外交互步数而非将256任务叫作256步。
数值检查允许有限期限尺度的两个float32 ULP误差，仅用于验证成本标签的
舍入；实际碰撞次数仍是整数，未修改接触阈值、奖励或任何任务标准。

验证补记：38项相关测试通过，Ruff/Pyright通过。GPU预试验a分两次运行
64→65→66；独立b一次运行64→66。两个checkpoint的模型与优化器逐位相同，
任务记录及除时间外更新指标完全相同，16个唯一训练任务无重复。
单元测试还验证eval完整组保存与恢复跳过、旧成本张量不影响reward-only
更新、成本头无梯度、真实累计次数可大于1。预试验不作效果结论。
