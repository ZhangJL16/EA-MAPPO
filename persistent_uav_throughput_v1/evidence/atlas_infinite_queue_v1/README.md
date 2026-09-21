# Infinite-queue intervention v1 — 启动交接

状态：**已启动；16/16 workers 首个非零、可恢复检查点健康检查 PASS。尚无完整干预科学结果。**

- 冻结运行源码：`9297b6fae0d1d4c0251ce0034dfeecfbc5930c7a`；清单时间 `2026-09-21T17:33:50.815209+08:00`。
- 固定 41 个已选 matched divergent pairs，29 个历史 roots、70 个唯一 continuation；共享首动作仅计算一次。
- 原始完整 root 快照恢复，队列容量从 root 起设为完整冻结任务流大小的足够上界；不恢复历史丢单。当前与嵌套 oracle 都不会因容量而丢弃未来到达。
- 首动作、原绝对时间任务流、SAC/map/energy/battery/charging、root-origin 1308.6 秒窗口及原 Oracle-Safe SJF 不变。
- 4 项针对性工程测试通过；一个真实 root 的有限容量首任务事件 exact match、干预首任务物理结果 exact match、父状态隔离和嵌套磁盘恢复通过。
- 16 个进程 command/PID 对应，单线程；检查点可反序列化，冻结 contract 一致，队列容量与统计数组一致，观察有限，实际新 overflow=0，training updates=0。

[冻结协议](../../INFINITE_QUEUE_INTERVENTION_V1.md)、[manifest](manifest.json)、[jobs](jobs.json)、[pairs](pairs.json)。
[工程测试](engineering_tests.json)、[真实 smoke](engineering_smoke.json)、[启动健康](startup_health.json)。

首次工程 smoke 的恢复字节比较存在过强判据，尚未启动正式 workers 时已修正为父状态各自不变与恢复观察/事件相同；记录见 [engineering_correction.json](engineering_correction.json) 和旧工程 manifest。参数、任务集合、环境动力学没有因此改变。

## 后续收集

运行命令和 PID 见 [launch.json](launch.json)；停止后的原参数续跑命令见 [resume_commands.txt](resume_commands.txt)，不得与原分片并行写入。
独立 [supervisor](supervisor.json) 只等待全部分片完成或错误；完成后自动 collect。此回合仅检查启动，不持续监视中间科学成绩。

输出将包含 continuations.json、pair_comparisons.json、summary.json、integrity.json。比较旧/新 signed gap、absolute gap、消失/缩小/保持/增大/反转，并报告所有 unknown 和 failure。No-safe-continuation 不记零、不记死亡；只有两侧新窗口均有结果时才做对应配对变化，并用相同子集的原始 gap 对照。

这是解除未来有限队列约束的总干预效应，候选任务集合与调度选择也会随之改变；不是唯一 admission 中介路径的归因比例。41 对原本按分叉筛选，不是总体自然状态分布。不按中间结果砍样本、换 controller 或启动下一干预。

worker 原始事件和完整快照保存在本地被忽略的 worker_*/，启动提交不代表完整原始数据外部备份。
