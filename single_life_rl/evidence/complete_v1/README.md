# Single-Life v1 完整结果与审阅入口

**已按冻结协议一次性完成：200 条物理 mission + 47,000 个学习 job。**
完成时间：2026-09-21 00:20:24（Asia/Shanghai）。运行源码：`c85b38a`。
无缺失、无重复、无工程错误；没有根据中间成绩改参数或选择实例。

- [协议](../../SINGLE_LIFE_PROTOCOL_V1.md)
- [定理与证明草稿](../../paper/THEORY.md)
- [程序生成的全方法汇总](REPORT.md)
- [主指标表](summary.csv)、[图表 PDF](scaling.pdf)、[图表 PNG](scaling.png)
- [正确/错误证书与细分结果](review_summary.json)
- [追加文献风险审计](../related_work_addendum.md)

## 先明确安全含义

主算法的目标是 **sup_M P(ever catastrophe during learning and deployment) <= .05**。
预算只分配一次，不随回站重置；第一版 delta_alea=0，所有风险来自 epistemic error。
证明依据是 anytime-valid likelihood-ratio confidence sequence 与 robust-safe action set。
下面的有限窗口频率仅是 calibration check，不用于推导或替代理论保证。
严格零灾难不可学习性单独陈述；不再使用已被反例否定的原零灾难 iff。

## Synthetic 主结果

这里“正确证书”使用模拟真实模型核验，和原汇总 `Certified`（仅表示算法已给出证书）不同。

| SafeRefine primary suite | Runs | Catastrophe | 正确证书 | 错误证书 | 未发证书 |
|---|---:|---:|---:|---:|---:|
| Binary（含 kappa=0） | 2000 | 58 | 1542 | 58 | 400 |
| Recursive Unlock | 1600 | 52 | 1548 | 20 | 32 |
| Nuisance | 500 | 15 | 485 | 10 | 5 |
| Random trees | 5000 | 165 | 4835 | 28 | 137 |

所有 SafeRefine catastrophe 均伴随真实模型此前被排除；读回审计检查通过。
证书错误概率与灾难概率不同：错误解锁可能在最终发证前就导致灾难。
Nuisance 的不同 m 使用相同 relevant truths/streams，结果配对，不能把五个 m
复制出来的 relevant outcomes 当成 500 个独立安全试验。

- **Binary kappa=0**：400/400 不解锁，0 catastrophe，无证书，regret=16000。
- **Binary kappa>0**：1600 次均给出证书，其中 58 次错误并在部署时 catastrophe。
  kappa=.02/.04/.08/.16 的平均发证时间分别为 884.14 / 231.73 / 60.13 / 15.13。
  这与逆信息量的预期趋势一致，不是对一般下界的证明。
- **Recursive**：StaticSafeID 1600/1600 无法发出完整策略证书；SafeRefine 正确认证1548/1600。
  其余52条保留为失败，不被丢弃。
- **Nuisance**：m=0/4/8/16/32，SafeRefine 平均 exploration 均为170.79；
  FullModelID 为170.79 / 614.79 / 1196.29 / 2404.69 / 5058.55。
  两者使用相同安全预算与 relevant-first design，区别是 FullModelID 继续到 singleton。
- **Random**：SafeRefine 与 FullModelID 在无 nuisance 的这组完全一致；
  StaticSafeID 5000/5000 无法完整认证。随机化范围是平衡三层树的标签、噪声和时长，
  不能将它表述为任意拓扑/任意 MDP 的实证覆盖。

## Lifetime delta 敏感性

预先冻结的 binary positive-kappa 子集，每行1600次；同实例/seed跨delta配对。

| delta | Catastrophe | 经验比例 | 平均发证时间 |
|---:|---:|---:|---:|
| .1 | 116 | 7.25% | 203.18 |
| .05 | 58 | 3.625% | 297.78 |
| .01 | 12 | .75% | 483.37 |
| .001 | 0 | 0% | 730.63 |

0/1600 不证明真实风险为0或小于.001。`raw_run.tar.gz` 中的 `aggregate/cells.json`
保留固定条件的Wilson区间；不同模型/参数混合池的区间仅作描述。
图中 random-tree restricted mean 把所有未认证情况（包括灾难）赋到原horizon，
因此包含终止风险的影响，不能把散点斜率直接当成理论复杂度常数。

## UAV：没有算法优势，保留这一结果

200条物理mission中195成功、5次navigation timeout；未补抽样。
B=q60(E)=39.28778762567276，r=B/109.05=0.36027315566870943。

| theta | Safe arms | 最优 mission ID |
|---:|---:|---:|
| .85 | 141 | 89 |
| .925 | 128 | 89 |
| 1.0 | 117 | 89 |
| 1.075 | 105 | 89 |
| 1.15 | 97 | 89 |

全部模型具有相同的最优mission，而且它对最坏模型也安全。
因此 SafeRefine 500/500 在 t=0 认证共同最优策略，探索时间0。
Oracle、WorstCaseRobust、SafeRefine、CertaintyEquivalent 的 reward/throughput 相同，
全部0 catastrophe；平均regret约0.38来自T截止的不足一个完整cycle。

**该UAV实例没有重现在线解锁带来的性能收益。** 安全任务集合虽随theta不同，
但影响最优决策的不确定性在这里为空。不得更改reward、battery或mission library来制造gap。
另一个预先指出的限制仍然成立：完整cycle时钟能反算theta，不能声称存在Gaussian-only
的能量识别难度。这是半经验再生arm模型，不是原persistent-UAV队列实验或真实机器人安全证明。

## 理论与新颖性状态

草稿给出有限、可重复experiment库中的任意高置信度可学习性 iff，以及分层Bernoulli
测试族的下/上界。**没有完成一般未知MDP或任意自适应experiment graph的匹配复杂度定理。**
也没有因实验成立而确认新颖性。既有工作覆盖安全集扩张、模型—可行域耦合和信息复杂度；
仍需逐条定理比较和独立数学审阅，不能声称已达到Level 5。

## 证据和重建

- `freeze.json`：运行Git、全源码/物理依赖哈希、完整config及job-plan hash。
- `mission_library.json`：全部200条物理结果，包括timeout。
- `raw_run.tar.gz`：完整运行输出；包含47,000逐run结果、固定job plan、各worker状态、
  全汇总、物理checkpoint及mission manifest。仅排除进程锁/临时文件。
- `raw_inventory.json` / `archive_verification.json`：47,176个member逐个读回验证。
- `SHA256SUMS`：本目录交付内容校验和。

工程检查：15项针对性检查通过；真实飞行2个policy steps的磁盘恢复等价smoke通过；
16/16首批检查点哈希通过；collect再次核对全部47,000 job身份与覆盖。
归档保留原运行配置和源码哈希。若重放需要检出运行提交c85b38a并恢复匹配的依赖和模型；
后来记录结果的Git提交不等于运行源码提交。没有新训练或后续实验启动。
