# 二维双约束实验跟踪

日期：2026-09-26。状态字段以实际原始结果和检查点为准；本文件不替代运行器状态。

| Run ID | 阶段 | 系统/变体 | 地图 | 状态 | 证据 |
|---|---|---|---|---|---|
| HIST-map000-full | 历史诊断 | 路线＋补满 | 0 | DONE | dual_constraint_2d/BASELINE_MAP000_RESULT_20260926.md |
| M0-code-freeze | 启动 | 原始训练环境、MPC、RL、矩阵调度 | 8–55 | DONE | STATIC_MATRIX_RUN_20260926.md；v1 manifest；针对性测试和 512 决策 smoke 通过 |
| M1-train-3seeds | 训练 | 学习策略 × 3 种子 | 8–31 | RUNNING | artifacts/dual_constraint_2d_static_matrix_v1_20260926/train/ |
| M2-validation | 验证 | 学习与 MPC 配置 | 32–39 | QUEUED | 训练完成后自动运行 |
| M3-static-test | 主比较 | 补满、部分补能、MPC、学习 | 40–55 | QUEUED | 验证完成后自动运行 |
| M4-coupling | 诊断 | 安全层干预与计算代价 | 40–55 | TODO | 待生成事件分析 |
| M5-queue | 条件后续 | 多目标队列 | 待冻结 | NOT AUTHORIZED FOR RUN | 仅当前任务负结果后单独启动 |
