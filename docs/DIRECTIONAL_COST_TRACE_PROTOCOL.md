# 成本信用分配：MC vs GAE-RTG 配对续训

> 已被用户2026-09-05的碰撞后继续回合要求取代，禁止据此启动新实验。
> v1主实验已暂停隔离；v2只完成预试验，未启动主实验。当前协议见
> LOCKED_COLLISION_RECOVERY_PROTOCOL.md。以下保留为历史研究记录。

2026-09-05，启动前协议；碰撞安全研究，不是能耗实验或新算法声明。

实施验证补记：29项相关测试通过，Ruff/compile/Pyright通过。
预试验两支从checkpoint64各更新到65，再用--resume恢复到66；
fork权重一致、每支16个唯一任务、更新日志恰为65/66，checkpoint66落盘。
预试验eval_tasks=0，只验证计算与恢复，不用其任务结果挑选参数或宣称安全提升。
正式开发比较仍按下述32追加组、每模式50任务执行。

## 为什么不只延长超时上限

慢乘子策略完整200任务重放全部匹配原seed、结果、步数和raw return。
其确定性11个超时中10个在最后500步（100秒）净前进不足50米；
只有seed593800043仍前进940.8米。2个为持续低速，7个满足无效移动诊断标记
（末段路程>100米且净进度/路程<.1，标记可重叠，不是新的成功标准）。
随机12个超时均满足这一无效移动标记，其中8个曾到60米内，4个曾到约5—6米。
这些现象既包括靠近目标不收敛，也包括远处停滞，不能全归因于路太长。
末段最小几何净空仍42.6米以上（随机分支最低）；未发生接触，并非碰撞后物理修复。
这不保证任意后续动作可安全通行。

完整报告及逐任务轨迹：`artifacts/directional_slowdual_timeout_audit_20260905_v1/`。

## 单因素假设

当前同一个超时任务的MC成本目标全为0。它能准确描述任务没碰撞，
但不能单靠该结果区分轨迹中的有效靠近动作与无效徘徊动作。
奖励和成本均需要价值估计完成信用分配。此前奖励trace=.95、成本trace=1，
不是成熟PPO-Lag常见的对称GAE设置。

假设：保持碰撞风险目标不变，仅改变actor成本优势的时间信用分配，
可改善导航/安全取舍。**超时轨迹不能证明此假设，必须做干预对照。**

参考[OmniSafe onpolicy buffer的gae-rtg分支](https://raw.githubusercontent.com/PKU-Alignment/omnisafe/main/omnisafe/common/buffer/onpolicy_buffer.py)，
该分支用GAE更新actor，但以return-to-go拟合critic；
[PPOLag配置](https://raw.githubusercontent.com/PKU-Alignment/omnisafe/main/omnisafe/configs/on-policy/PPOLag.yaml)
使用cost trace=.95。这里保留γ_C=1的有限任务适配，不照搬其.99折扣、obs标准化或其他超参。

对一条完成任务，V_C(s_T)=0：

\[
\Delta^C_t=c_t+\hat V_C(s_{t+1})-\hat V_C(s_t),\qquad
\hat A^{C,\ell}_t=\sum_{j=0}^{T-t-1}\ell^j\Delta^C_{t+j}.
\]

- MC对照ℓ=1，保持原数组不变以避免重计算舍入改变控制。
- GAE干预ℓ=.95，只替换actor的成本优势。
- critic标签始终`G_C=sum c`，γ_C=1，首次碰撞轨迹全1，超时/到达轨迹全0。
- dual始终使用完整任务接触均值，η=.01、δ=.05，不把晚碰撞折扣掉。
- 环境、碰撞修复、5米goal tolerance、4000步期限、LiDAR输入及网络均不改。

### 数学边界

ℓ<1不改变上述MC风险标签，但用不精确V bootstrap会引入actor梯度偏差。
沿同一有限轨迹，设e_t为估计V与参考V之差且e_T=0，优势误差为

\[
-e_t+(1-\ell)\sum_{j=1}^{T-t-1}\ell^{j-1}e_{t+j},
\]

因此若|e|≤ε，绝对值≤2ε；这一界是代数界，不证明本网络ε小。
ℓ=1消去未来bootstrap误差但仍有baseline项。总体policy-gradient还需要
状态/历史条件及采样假设，不能把近似观测critic当成真实MDP价值。
[GAE原论文式16—18](https://arxiv.org/abs/1506.02438)明确讨论偏差—方差取舍。
不把更低样本优势标准差当作性能保证，也不声称提出新定理。

## 配对续训及预算

从`directional_lagrangian_slowdual_pair_20260905_v1/lagrangian/checkpoint_0064`
分叉，两支复制相同actor、两value头、optimizer、乘子.26675及RNG。
MC对照不是λ=0对照，而是原慢乘子方法继续训练。
每支再32个完整cohort（256任务），64→96；训练seed483600513..483600768。
不复用开发场景作梯度数据。不同策略导致交互和梯度步数不同，须分别报告。
追加预算最多每支1,024,000实际步；不是必须跑满，也不按旧79.6万步重头计数。
两支最后各做确定性50和随机50开发任务，原seed593800001..593800050；
这是配对开发比较，不能声称未查看正式测试。

```bash
uv run --no-project --python .venv/bin/python python scripts/run_directional_cost_trace_pair.py \
  --source artifacts/directional_lagrangian_slowdual_pair_20260905_v1/lagrangian/checkpoint_0064 \
  --output-dir artifacts/directional_cost_trace_pair_20260905_v1 \
  --extra-cohorts 32 --eval-tasks 50
```

顺序运行gae95、mc_control；不并发两份训练。预计20—60分钟，任务变长可更久。
每个完整cohort保存optimizer/RNG及日志；暂停信号在组边界处理。
续跑同命令加`--resume`。不设破坏恢复的硬墙钟kill。
初始健康和checkpoint验证后不监控结果，完成自动退出，无后续自动长训。

## 结果判读

主要看随机策略safe_goal/contact/timeout三项；确定性作为部署动作方式的补充。
若GAE优于自己的起点却不优于MC续训，不能归因于GAE。
只降接触而增加超时不算解决；只增到达而提高接触同样报告其代价。
开发50任务、单训练fork不足以宣布普遍改进；改善后还要新训练seed和未查看场景。
若无改善，不继续盲扫trace/叠模块；重新评估近目标训练覆盖、价值误差及观测混叠。
本轮不增加reward项、不放宽goal tolerance、不缩小任务难度。
