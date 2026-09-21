# Retained-counterpart / residual-queue archived audit v1

**简单的 retained-counterpart 局部量没有近乎完整地解释吞吐排序。** 留下耗时较短 counterpart 有一定描述性吻合，但 pair 等权吻合率为 62.5%，root 等权后为 54.0%；其他指标也未给出统一方向。本轮不支持把“counterpart 更顺路”确立为主导因果机制，也不证明长期规划必要。

## 范围与数据完整性

只读取上一轮已提交的 `post_task_evidence.jsonl`、`branch_features.jsonl` 和 `pairs.jsonl`，没有运行新 simulation、训练、改环境或恢复执行快照。无需公式证明工具，未安装 Lean。

固定群体为同一 root 内 time+battery joint-matched 且首任务期间 arrival IDs 相同的全部 **77 对**，来自 **50 roots、45 source/regime/seed runs、10 个 seeds**。保留 36 个等吞吐对；41 对吞吐不同，涉及 29 roots。17 个原 Atlas 未决 roots 仍不进入完整案例群体。

逐对提取 branch i 完成 i 后留下的任务 j，及 branch j 完成 j 后留下的任务 i，共 154 条已归档 oracle candidate 查询。77/77 对均满足：

- 首任务期间 arrival IDs 相同；
- accepted arrival IDs 相同；
- 排除 counterpart 后，其余 residual task 的 ID 和坐标完全相同。

因此这次不是仅匹配 queue 长度；队列内容差异确实只在 counterpart。但完成位置、完成时刻、资源及 controller 状态仍未被单独干预。

154 条 counterpart 分支为 128 success、25 energy depletion、1 navigation failure。实际 task cost 仅在 task 完成时使用；完整 task+return time / return reserve 仅在返航成功时使用。失败后观测到的短耗时或低能耗保留为截断记录，不能当作便宜任务。

## 对吞吐方向的解释程度

下表以 **41 个吞吐不同的配对** 为固定分母。“吻合”表示预先指定的单调方向选中了 N(W) 较高侧；“相反”表示较低侧；同值不强行打破平局，缺失不插补。

| 简单方向假设 | 吻合 | 相反 | 特征同值 | 截断/不可用 |
|---|---:|---:|---:|---:|
| counterpart safe 更好 | 2 | 2 | 37 | 0 |
| counterpart 实际任务耗时更短 | **25** | **15** | 0 | 1 |
| counterpart task+return 更短 | 13 | 16 | 0 | 12 |
| counterpart 实际任务能耗更低 | 23 | 17 | 0 | 1 |
| counterpart return reserve 更大 | 16 | 13 | 0 | 12 |
| counterpart 全队列 predicted-SJF rank 更靠前 | 11 | 13 | 17 | 0 |
| counterpart oracle-safe 队列 rank 更靠前 | 7 | 10 | 12 | 12 |
| counterpart 实际成为下一动作 | 4 | 6 | 31 | 0 |

SJF rank 使用冻结预测时间、task ID 打破平局；另导出实际已完成任务耗时排名，竞争候选耗时不完整时保持 null。rank 是在各自 post queue 内计算的相对名次，不是跨队列的绝对价值。

“实际任务耗时更短”在可排序分叉对中为 25/40=62.5%；先在每个 root 内算吻合率，再对有可排序分叉的 roots 等权平均，为 **54.02%**。两者差异说明重复配对权重会影响描述。它不是交叉验证预测准确率，也没有独立样本显著性结论。

所有 77 对的等 N、同特征、缺失分母均保存在 summary.json。例如任务耗时规则还在 34 个等 N 对上给出不同排序，1 个等 N 对上同值，另 1 个等 N 对有截断值；不能只读 25/40 而忽略覆盖率。

| 来源 | 全部配对 / 分叉 | 耗时更短吻合 / 可排序分叉 | root 等权吻合率 |
|---|---:|---:|---:|
| B4-.75 | 47 / 29 | 17/28 | 48.25% |
| B5 | 30 / 12 | 8/12 | 65.00% |

其他量的方向也会随来源变化。例如较短 task+return 在 B4 来源为 7/20，在 B5 来源为 6/9。没有依据分层结果重新挑选指标或方向；完整电池分层保留在 summary.json，稀少 strata 不做强推断。

## counterpart 安全与实际选择

| 条件 | 配对 | roots | 分叉对 | 分叉 roots |
|---|---:|---:|---:|---:|
| 两侧 counterpart 都安全 | 61 | 42 | 29 | 23 |
| 只有一侧安全 | 6 | 5 | 4 | 4 |
| 两侧均不安全 | 10 | 8 | 8 | 6 |
| 两侧下一步都选择 counterpart | 17 | 16 | 8 | 7 |

在两侧都安全的 29 个分叉对中，较短 counterpart 飞行吻合 19/29，较短 task+return 为 13/29，较大 return reserve 为 16/29。即使排除不同 safety 状态，仍没有近乎完美的局部排序。

两侧下一步都选择 counterpart 的 17 对只表示归档的下一动作相同类型（分别为 i→j、j→i）；并非保证随后完整状态合并。选择顺序仍影响终点、时间、电量和期间到达。不能把 8 个分叉对描述成“消除了两任务序列之后的一切区别”。

## c(i,j) 与 c(j,i) 的成本差

这里成本来自冻结 controller 的真实归档分支，不是欧氏距离代替飞行成本。

| 测量 | 两侧完整可用的对数 | 绝对差中位数 | 绝对差均值 |
|---|---:|---:|---:|
| counterpart 任务时间（秒） | 75 | 5.10 | 8.89 |
| counterpart 任务能耗（原能量单位） | 75 | 1.128 | 1.902 |
| counterpart task+return 时间（秒） | 61 | 16.30 | 20.25 |
| 单独 return 时间（秒） | 61 | 13.65 | 16.99 |

task+return 的差不全是 A→B / B→A 非对称：末端任务不同，返航腿也不同。counterpart 欧氏距离差中位数仅 1.746 米，理想任务坐标之间的欧氏距离本来对称；这里使用实际完成端点，所以小距离差还包含到达容差，不能包装成大尺度有向路由结构。

再把首任务计入，比较 `first_time + counterpart_task_time`，较短的一侧只在 23/40 个可排序分叉对中吞吐更高；比较 `first_time + counterpart_task_return_time` 为 16/29。因此当前也不能用一个两任务总时间规则几乎完整解释 gap。

## 本轮能说什么、不能说什么

可以说：在这个同 arrival、time/battery 近似匹配的归档群体中，retained-counterpart 的已测试单调局部量没有给出强而统一的吞吐排序；实际飞行方向成本确实有差异，但其存在不等于它造成了主要吞吐差异。

不能说：counterpart / residual geometry 完全没有价值；所有 handcrafted index 都不行；相互作用或非线性组合已经被排除；或者因此证明问题需要更长规划。我们没有拟合多变量模型，也没有干预位置、队列或时间。局部量失效与长时域必要性之间不存在本轮已证明的逻辑蕴含。

所有分析是回顾性描述。配对共享首动作、root 来自同一 run、seeds 跨 regime/source 共享；不把 77 对或 154 查询当独立重复，不报告 IID p 值或置信区间。方向和变量在本次 counterpart 统计之前写入 PLAN.md，但 N(W) 在前轮已知，不称实验前预注册。

**当前判断：这个简单 counterpart 主因假设尚未得到支持；主导机制仍未识别。** 后续 multi-step queue/charge/phase 交互仍是候选，而不是自动成立的答案。本轮未启动后续实验。

## 可复算证据

```bash
python persistent_uav_throughput_v1/evidence/atlas_retained_counterpart_v1/analyze.py
python persistent_uav_throughput_v1/evidence/atlas_retained_counterpart_v1/verify.py
```

本轮只需要仓库内上一轮提交的 JSONL 文件；不需要本机大快照、原始 worker 文件、模型或 simulator。

- [PLAN.md](PLAN.md)：固定分析口径及方向。
- [counterpart_pairs.jsonl](counterpart_pairs.jsonl)：77 对完整提取、候选原始记录、预测/实际排名、截断标记及 source hash。
- [summary.json](summary.json)：所有方向及来源/电池、both-safe、both-next 分层；保留同 N 与缺失。
- [analyze.py](analyze.py)：标准库离线提取与统计。
- [integrity.json](integrity.json)：输入、代码、计划与输出哈希。
- [verify.py](verify.py)、[validation.json](validation.json)：独立按 counterpart ID 查找证据、截断核对、方向计数复算、交换两侧不变性检查。

校验全部通过。没有改动原 Atlas、matched-pair 阈值或先前结果文件。
