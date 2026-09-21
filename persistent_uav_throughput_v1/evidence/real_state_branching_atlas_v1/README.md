# Real-State Counterfactual Branching Atlas v1

状态：**已冻结并启动；16-worker 首次检查点健康检查 PASS。科学结果尚未汇总。**
记录时间：2026-09-21T09:38:08.918098+08:00。

- 运行源码：`6d328f8e8746a14a5d4d4d8cc4cd8c1a79856280`；[协议](../../REAL_STATE_BRANCHING_ATLAS_V1.md)。
- 两组归档共 540 个 run，B4-.75 1496 个 member、B5 672 个 member 全部校验。
- 固定 216 roots，54 个 cell 每个 4 个，无短缺与补抽。
- A：740 task→return + 216 direct-return = 956 个固定分支。
- B：由 A 的 >=2 safe-task 条件唯一决定；每个 safe first action 都执行。
- 同一 root 的窗口固定为 [t_root, t_root+1308.6s]，包含首任务。
- 10 项工程检查通过；两种来源的真实历史重放、分支和嵌套恢复 smoke 通过。
- 16 workers 均为单线程；检查点哈希、历史 prefix、流哈希和有限观察已核验。
- 只完成首次健康检查，不依据中间成绩调整运行。训练更新始终为 0。

## 记录与最终输出

[manifest.json](manifest.json) 冻结参数、依赖和代码；[roots.json](roots.json) 冻结抽样。
[startup_health.json](startup_health.json)、[engineering_smoke.json](engineering_smoke.json)
记录启动证据。原始输入来自已提交的 B4/B5 archive，工作副本和大型状态快照保存在
本目录的 inputs/、worker_*/ 中，不将活动 checkpoint 当作已归档科学结果。

16-worker 运行命令与 PID 见 launch.json；精确续跑命令见 resume_commands.txt。
只在相应 worker 已停止后续跑。不要并发写同一个分片。

supervisor.json 中的独立汇总进程仅检查状态：全部完成后自动执行 collect，生成
one_step_branches.jsonl、continuation_branches.jsonl、summary.json、integrity.json。
遇到错误保留 error.json 和最后有效快照，不删 root、不更换 seed。

`no_safe_continuation` 是诊断中止；未来 N(W)/F(W) 保留 unknown，不能冒充死亡或零吞吐。
完整结果应同时检查 multi-safe 比例、未决比例、safe-vs-safe gap、三个 local scalar
对照、实际 B5 安全误判/排序、navigation/depletion 与 overflow，不能仅报告 oracle gap。
