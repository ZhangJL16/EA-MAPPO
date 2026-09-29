# Persistent UAV 整窗可行性与公平对照：独立研究项目

日期：2026-09-29。此目录是独立范围的新研究项目；`/home/zjl/mappo` 的 regenerative-control 结项资产与 `persistent_uav_throughput_v1/` 实验资产只读引用，不改其源码、参数、模型、历史结果或 Git 证据。当前任务由用户在上一轮[信息边界裁决](/home/zjl/uav_throughput_identifiability_round_20260929/deliverables/RESEARCH_DECISION.md)后明确要求继续推进。

目标是把下一步变成可复核检查：对相同可部署观测与相同离线/在线计算权限，先查是否存在达到初始分布下完整原始 `T` 的 `P(actual energy depletion)≤0.05` 的固定高层参考策略，再比较其实际完成任务数与强规划/学习策略。导航 timeout 单独报告；不把 `task→immediate return`、1308.6 秒 Atlas 窗口、每步可返回性或 calibration 飞行当成完整 `T` 标签。不凭旧 B4/B5 的 10 个 validation seeds 作风险保证。

本轮先审历史完整结果和实际代码接口，冻结信息/风险/策略/数据/统计合同及最小可执行方案；不从历史计划自动启动旧实验或训练。任何新运行都应在本项目自己的结果目录，遵守父目录 AGENTS.md 的冻结碰撞规则、最小启动检查和首个 checkpoint 后交还用户要求。

协作角色：A01 审完整 `T` policy 与安全参考可行性；A02 审数据/抽样/同时置信界和 common-random-number 陷阱；A03 审同信息同预算规划/学习对照与最小新数据接口；A04 第一轮独立攻击；A05 第二轮独立复核。共享 `board/`，允许零方法候选，原始结果优先于历史计划。相关论文论断只用已保存的合法全文并如实记阅读范围；数学未经 Lean 不称形式证明。

冻结问题边界：单 UAV、外生 Poisson 持续到达、初始 3 单/最多 5 等待单、单地点服务、单站全充、固定完整 `T`、实际完成数。当前实现非空队列不能 idle，满电站内不能 recharge，服务/回站低层冻结。若需要更改任一语义以生成 safe fallback，那是**新问题设定**，必须单独标注，不能和原始任务混算。
