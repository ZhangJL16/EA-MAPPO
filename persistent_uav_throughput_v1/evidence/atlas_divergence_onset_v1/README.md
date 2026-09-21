# Divergence-onset audit：真实 continuation 事件对齐

**主要时间结构是“动作序列早分叉，完成数反复追平，终局差较晚稳定”，不是一个已经识别的晚期因果触发器。**

41 对中，38 对在前两个 downstream decisions 就选择不同动作；所有 41 对的累计完成数差都曾重新归零，中位数追平 7 次，32 对发生过领先方反转。因此不能把第一次计数差当成最终 gap 的起因，也不能把终局领先直到很晚才稳定解释成长期规划必要。

本轮只读原始 continuation traces，未启动仿真、训练或新策略。

## 样本、对齐与可视化

固定上一轮全部 41 个 same-arrival、time+battery matched 且 N(W) 不同的 pairs：29 roots、28 source/regime/seed runs，复用 70 份不同的实际 B 轨迹。未增加、删减或按 onset 筛选 pair。

打开 [交互时间线](timeline_viewer.html)，可选择全部 41 对，查看累计任务数阶梯线、实际 task/return/charge/arrival/overflow/decision 事件，拖动时间光标查看等待队列、完成任务集合及最近决策快照。HTML 自包含，可下载后直接离线打开。

两种对齐分别报告：

1. **Downstream decision ordinal**：不计 root 的强制首动作，按第 1、2…个后续决策比较 `(kind,task_id)`。不因 reason 字符串不同认定动作不同。
2. **共同绝对时间**：按原物理 0.05 秒网格处理 <=1e-6 的浮点误差，同一时刻两侧事件合批比较。队列在 serve 决策时移除任务，在 accepted arrival 时加入；逐个核验实际 predecision queue。

时间均从 root 起算，窗口 W=1308.6 秒。等待队列不包含正在执行的任务，所以其变化不能等同于整个 unfinished workload 改变。只用实际 top-level events，没有混入 oracle 嵌套分支的 arrival 或 task completion。

## 1. 动作身份与决策时刻

| 首次 downstream 动作身份不同 | 配对数 |
|---|---:|
| 第 1 个决策 | 31 |
| 第 2 个决策 | 7 |
| 第 3 个决策 | 2 |
| 前 7 个身份相同，尾部第 8 个仅一侧存在 | 1 |

因此 **38/41（92.68%）** 在前两个后续决策已选择不同动作，涉及 26 roots。最后一例是 `B4_0.75_r17_s920260924_d00040` 的首任务 15/39：它是序列长度不同，不能写成“两侧第八步选择了不同动作”。

所有 41 对的第一个 downstream decision 时间已经不同。这与首任务耗时仅近似匹配相符，不是额外发现。初始 action 不同也会天然留下不同 counterpart，所以上表本身不能证明短 horizon 策略充分或最终输赢已在此决定。

## 2. 计数差不是单一的 onset

| 定义 | 41 对中的中位 elapsed（秒） |
|---|---:|
| 首次完成数不相等 | 134.15 |
| 两个首任务都完成之后首次完成数不相等 | 176.55 |
| 最终赢家此后不再被追平或反超 | **1194.50** |
| 完成数差此后一直等于最终数值差 | **1268.45** |

首个计数差 **41/41 都发生在较慢首任务完成之前**，只是首任务完成的相位差。最初领先侧仅在 24/41 对中成为最终赢家。

所有 41 对都至少重新追平一次，追平次数中位数为 **7**；**32/41** 对发生过领先方反转（忽略中间零差时刻后比较非零差符号）。

“稳定领先”通过整条轨迹回顾定义：找最早时间，使之后所有完成事件时刻到 W 的差值符号都等于最终符号且非零。“最终数值差稳定”要求差值本身不再改变。它们分别距离截止时间中位数约 114.1 秒和 40.15 秒。阶梯计数交替追平与固定截止时间都会使这些时间靠近窗口末尾，**不是 causal decision time**。

在稳定领先事件批之前，两侧已选 downstream decisions 数量取较大者：

| 已选决策数（严格早于事件批） | 配对数 |
|---|---:|
| ≤2 | 3 |
| 3–5 | 5 |
| ≥6 | 33 |

每侧计数分别保存在 pair_audits.json。排除了在该完成时刻才新选的下一动作，避免 off-by-one。这个表是回顾性过程描述，不能拿来证明“33 对需要 horizon≥6”。

## 3. 补能与队列何时分叉

- 所有 41 对第一次 recharge request 的绝对时间就不同；两侧时差的中位绝对值为 **34.75 秒**。
- 所有 41 对第一个完整 charging cycle 的时长也不同；中位绝对差为 **4.90 秒**。
- charger arrival、charge complete 分别对齐并记录，不把返航请求等同于开始充电。未完成 charge 和缺少对应 cycle 有单独字段，未记作 0。
- root 强制服务一开始，原始 waiting queue ID 就因移除不同首任务而不同。这是预期差别。
- 排除 pair 的两个原始 task IDs 后，其余 waiting queue 首次不同的 elapsed 中位数为 **148.25 秒**；限制两项首任务都已完成后观察，中位数为 **153.70 秒**。这也会受到一侧较早开始下一项服务、任务从 waiting queue 移出的影响，不能视为独立的 queue-composition 干预。
- 整个窗口的外生 arrival IDs 和时刻逐项一致，但 **27/41** 对在后续曾出现 accepted/overflow 判定不同。

最终稳定领先之前，至少一侧已经：提出 recharge 37/41、完成一次 charge 35/41、出现配对 arrival acceptance 差异 19/41。这里使用严格早于的事件关系；这些条件不是互斥机制类别，也不是因果归因率。

## 4. 相同完成任务集合并不代表完整状态重合

**25/41** 对在两项首任务完成后，曾拥有相同且非空的“从 root 起已完成任务 ID 集合”，随后仍走向不同终局。**11/41** 对在最终稳定领先事件之前的瞬间，完成任务 ID 集合还相同。

这回答了“是否只是任务顺序不同”的一部分：确实有轨迹曾追到相同任务集合。但位置、电量、决策时刻及待执行任务仍可能不同，所以不能说 simulator states 已合并，也不能把后来的差异称为从同一个完整状态自发分叉。

## 5. 两种来源

| 来源 | pairs | 前两个 downstream 决策动作已不同 | 稳定领先中位秒数 | 曾重新完成同一任务集合 |
|---|---:|---:|---:|---:|
| B4-.75 | 29 | 28 | 1198.60 | 17 |
| B5 | 12 | 10 | 1182.85 | 8 |

电池分层完整保留在 summary.json；本群体没有 B=2 的分叉 pair。41 pairs 来自此前按终局差筛选的样本，不能推断全部安全动作的 onset 频率。pair 共享动作/roots，seeds 跨 regime 和来源共享，不作独立样本推断、p 值或置信区间。

## 当前结论与不能下的结论

真实、可重复的时间模式是：**多数动作序列很早就不同，但吞吐领先方会反复变化；补能和 queue acceptance 差异穿插其中，最终计数差在截止窗口附近才停止变化。**

这次定位了过程，尚未识别一个足以解释最终输赢的事件。不能根据早期 action mismatch 宣布 short-horizon 足够，也不能根据晚期稳定领先宣布 long-horizon 必要。要知道某个决策是否“决定了最终 gap”，仅靠观察对齐无法识别；本轮没有执行改变该决策的干预。

因此没有把 pairs 强行分入“立即决定 / 2–3 步决定 / recharge 决定 / 随机晚期放大”四个因果类别。原轨迹在冻结 controller 和相同外生流下比较，晚期波动也不能无依据称为新的随机扰动。

## 状态与复核边界

每个 onset 保存精确重建的 queue IDs、完成 IDs、累计 overflow，以及事件前最近一次 predecision observation 和快照年龄。**未记录时刻的精确 battery/position 没有插值或捏造**；查看快照时必须同时看 timestamp/age。原始实际事件单独导出，支持逐条人工核对。

- [PLAN.md](PLAN.md)：对齐定义和解释边界。
- [pair_audits.json](pair_audits.json)：41 对完整 onset、补能周期与前序状态。
- [timelines.json](timelines.json)：合并时间网格上的计数、队列与决策数。
- [actual_event_traces.json](actual_event_traces.json)：70 份不同 B 分支的实际事件，不含嵌套 oracle。
- [summary.json](summary.json)、[integrity.json](integrity.json)、[validation.json](validation.json)。
- [analyze.py](analyze.py)、[build_viewer.py](build_viewer.py)、[verify.py](verify.py)。

完整重提取需要本机原 Atlas 的原始 worker B 文件；输出中的实际事件足够独立复核时间线、计数与 onset。70 份原始文件 SHA256 全部匹配；3646 项逐时刻完成数独立复算通过，稳定领先及最终 gap 的最早后缀边界核验通过。

```bash
python persistent_uav_throughput_v1/evidence/atlas_divergence_onset_v1/analyze.py
python persistent_uav_throughput_v1/evidence/atlas_divergence_onset_v1/build_viewer.py
python persistent_uav_throughput_v1/evidence/atlas_divergence_onset_v1/verify.py
```

交互页面通过 JavaScript 语法检查及 41 对渲染逻辑 DOM-stub smoke；未做浏览器截图验收。未修改原实验或前轮结果，未启动后续实验。
