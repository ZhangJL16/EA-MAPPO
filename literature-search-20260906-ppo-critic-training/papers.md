# PPO 价值估计与安全优化：全文重点阅读

2026-09-06。目的：解释本项目恢复导航中PPO续训退化，选择一个可直接实施的
干预。保存9篇论文及PPG补充材料的原始PDF和逐页文本于 `fulltext/`。
阅读范围是下面列出的正文与附录重点；不声称逐行审完全部证明。
仅以原始论文、官方实现支持判断。

## 决策

采用“actor到KL界后，critic补足原定更新量”的单因素实验。
这是借鉴已有实现和数据复用研究，不能包装为新的安全RL方法或完整PPG。
文献没有证明它必然修复本项目；本地可直接观测的证据是critic实际更新量
远低于配置上限，且此前整程价值准确性较弱。

## 论文与阅读范围

评分为本次阅读范围内的主观优先级，I/C/E分别是洞见、完整性、数值证据
（1—5）；数值证据评分来自原文协议覆盖，绝非复现实验评分。

| 论文 | 年份/来源 | 类型 | I/C/E | 阅读重点与适配判断 |
| --- | --- | --- | --- | --- |
| [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) | 2017 arXiv | pure method | 4/4/4 | §5、Algorithm1、§6.1、附录A表3。固定长度rollout、多遍SGD；原文联合loss解释以共享参数为条件。当前完整cohort采样是项目变体。 |
| [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) | ICLR2016 | pure method | 5/4/4 | §2 Proposition1、§3 Eq10—16、§4 value fitting。MC与GAE的偏差方差取舍依赖value精度；长trace不自动更好。 |
| [Implementation Matters in Deep Policy Gradients](https://arxiv.org/abs/2005.12729) | ICLR2020 | pure method | 4/4/4 | §3九项实现差异、Figure1和Table1、§4 trust-region分析。提醒具体优化机制会改变算法表现；不支持把所有技巧同时搬过来。 |
| [What Matters In On-Policy Reinforcement Learning?](https://arxiv.org/abs/2006.05990) | ICLR2021版本 | pure method | 4/5/5 | §3.2—3.6，附录B.1/B.2及基准配置。分离网络、value尺度、GAE、数据复用均有消融。更宽value或归一化只列候选，不同时实施。 |
| [Phasic Policy Gradient](https://proceedings.mlr.press/v139/cobbe21a.html) | ICML2021 | pure method | 5/5/4 | §2 Algorithm1、§3.1—3.4、补充A—D。政策和value可以有不同数据复用量。Procgen中辅助特征共享作用大；本项目不照搬蒸馏头。 |
| [Constrained Policy Optimization](https://proceedings.mlr.press/v70/achiam17a.html) | ICML2017 | pure method | 5/5/4 | §3—5差异界、§6.1—6.3及Algorithm1、§8实验。实际算法有近似、恢复和cost shaping；样本可行性不是零碰撞保证。 |
| [First Order Constrained Optimization in Policy Space](https://papers.nips.cc/paper/2020/hash/af5d5ef24881f3c3049a7b9bfe74d58b-Abstract.html) | NeurIPS2020 | pure method | 4/4/4 | §2定义、§3 Theorem1/优化、Algorithm1、§4任务。仍需reward/cost advantage；原伪代码内循环有KL停止，不能说所有安全RL都解耦critic。 |
| [Constrained Update Projection Approach to Safe Policy Optimization](https://arxiv.org/abs/2209.07089) | NeurIPS2022版本 | pure method | 4/5/4 | §3 generalized difference、§4 Theorem2/实现、附录C Algorithm1（PDF22页）。先改善reward再投影；最后value拟合独立列出。当前gamma1不能直接代入其折扣界。 |
| [Embedding Safety into RL: A New Take on Trust Region Methods](https://proceedings.mlr.press/v267/milosevic25a.html) | ICML2025 | pure method | 5/5/4 | Algorithm1、Theorem4.4/4.5及附录C.2证明、Figure3/Discussion。安全域几何很有价值，但依赖准确cost估计；不作为本轮即插即用安全模块。 |

## 对方法、证明和实验的具体核对

### 1. 价值训练能否独立于策略停止

PPG的政策期分别优化PPO surrogate与value MSE；其辅助期还有特征共享。
§3.2在控制value复用量时改变policy复用量；§3.3讨论过拟合与数据复用。
补充C表明value额外优化可以安排在政策期，补充D讨论低复用PPO的value
训练不足。上述结果来自Procgen，不能据此选取无人机的最优epoch。

[Spinning Up PPO官方源码](https://spinningup.openai.com/en/latest/_modules/spinup/algos/pytorch/ppo/ppo.html)
的update中，policy loop在KL过大时break，随后仍执行train_v_iters次value
更新。它使用独立Adam、MC/bootstrap return等其他设置；本轮只借鉴训练
预算分离，保留项目原有GAE标签、Adam状态和联合更新，形成可解释对照。

### 2. GAE误差是如何进入policy梯度的

GAE §3 Eq10指出，TD residual在真实value下是对应advantage的无偏估计；
近似value则会引入误差。Eq14—16从k步回报推导lambda加权迹。lambda1
消除中间bootstrap，但整程随机回报方差可很高。本地MC失败与此相容，
尚不能从相容性推出因果。详细有限轨迹恒等式写入实验协议。

### 3. 为什么不直接更换安全算法

CPO实际实现§6承认采样与局部近似误差，使用line search、不可行时恢复
以及cost上界塑形。冻结接触cost=0/1时不能悄悄把塑形后的量称为原cost。
FOCOPS理论的非参数最优更新在可行起点假设下推导，网络投影与有限样本
还会带来误差；选择第一阶算法可以减少求解负担，但不能消除critic问题。
CUP Theorem2给出含误差与折扣分母的界，Algorithm1采用GAE value标签；
把这里有限任务gamma1直接塞进带1/(1-gamma)的公式没有数学依据。

C-TRPO的Theorem4.4针对连续时间C-NPG流、安全域和regular参数化；附录
C.2利用occupancy空间的Hessian gradient flow证明不变性。有限步、有限
样本神经网络C-TRPO不是该理想化流。Algorithm1还有不安全时cost恢复。
Figure3比较8任务×5种子、1000万步的最终cost/reward及累计违规，而Discussion
明确指出cost advantage/value准确性会影响divergence估计。其对未来能耗
可行域研究有启发，但目前还没有本项目的可行初始化和估计误差保证。

### 4. 采样与初始化仍是备选解释

PPO Algorithm1是固定步片段，当前cohort收集每批8个完整任务。What Matters
§3.5说明片段长度、batch规模与sample reuse会改变学习；§3.2还发现初始化
影响很大。因而完整cohort和历史first-contact checkpoint迁移均不能排除。
本轮保留两者以单独检验critic预算；若不能改善，将优先评估标准连续rollout
或从零初始化，停止在trace/epoch上无止境微调。

## 后续研究关系

PPO训练稳定性：已有研究覆盖数据复用和优化实现，本轮是应用诊断，直接
新颖性弱。碰撞约束：CPO/FOCOPS/CUP已覆盖多种policy-space更新；重点
是有限任务、多次碰撞计数下的适配。能耗可行性：C-TRPO的安全域几何可以
借鉴，但返航可行域与累计接触预算不同，需要重新定义状态、目标和误差界。

推荐基准仍是本项目固定环境下的匹配对照；Procgen/MuJoCo/Safety Gym用来
理解论文证据边界，不能替代无人机的500场景评估。
