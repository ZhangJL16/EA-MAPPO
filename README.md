# 单无人机在线配送新研究：问题识别资料

本目录与已结项的 `/home/zjl/mappo` regenerative-control 研究分开。现有解析三维导航 v1/v2、1 km 场景的地图/服务点/假设充电组件，以及独立的合成二维双约束环境。截至 2026-09-26，1 km 静态导航资格已完成 96/96；二维已进行 PPO 训练与规划器诊断，结果和边界需按各自报告解释。**完整的三维持续取送货、能量安全与订单调度环境尚未实现。** v2 的 60 m 导航资格不能替代 1 km 或未来动态订单的资格测试。

- [评价指标与公平对比协议](EVALUATION_PROTOCOL.md)
- [区分能耗预测与调度决策的三组实验](DISCRIMINATING_EXPERIMENTS.md)
- [解析三维导航＋CBF 首版及其工程 smoke](NAVIGATION3D.md)
- [v1 导航资格结果与失败诊断](QUALIFICATION_REPORT_20260923.md)
- [v2 航段跟踪修复与完整多地图结果](NAV3D_V2_REPORT_20260923.md)
- [v2 同类地图低层冻结边界与逐文件校验清单](NAV3D_V2_FREEZE_20260923.md)
- [冻结低层后的能耗与订单接口](ENERGY_ORDER_PHASE_20260923.md)
- [四旋翼机型、运动上限与能耗参数调研](PLATFORM_ENERGY_SURVEY_20260923.md)
- [文献机型对照与 1 km 配送场景重估](PLATFORM_SCALE_REASSESSMENT_20260923.md)
- [已落地的 1 km 候选机型与地图配置](delivery_1km/README.md)
- [1 km 新尺度导航资格协议](DELIVERY_1KM_NAVIGATION_PROTOCOL_20260923.md)
- [1 km 导航资格首检查点记录](DELIVERY_1KM_STARTUP_20260923.md)
- [1 km 导航 v1 修复协议与完整资格结果](DELIVERY_1KM_NAVIGATION_V1_REPORT_20260925.md)
- [能耗与订单阶段已确认边界](ENERGY_PHASE_DECISIONS_20260925.md)
- [参数化飞后能耗计量器与待冻结参数](ENERGY_ACCOUNTING_MODEL_20260925.md)
- [已确认的能耗参数基准与逐项扫描](ENERGY_PARAMETER_PROPOSAL_20260925.md)
- [96 条冻结航迹 × 19 组参数的飞后能耗诊断结果](ENERGY_OAT_RESULT_20260925.md)
- [飞前路线摘要的内部跨地图估计诊断](PREFLIGHT_ESTIMATOR_RESULT_20260925.md)
- [二维避碰与返站能量双约束的新研究边界](DUAL_CONSTRAINT_2D_REFRAMING_20260925.md)
- [独立的二维双约束模块](dual_constraint_2d/README.md)
- [二维全图解析对照首检查点](dual_constraint_2d/BASELINE_STARTUP_20260926.md)
- [二维全图解析对照 map 0 完整轨迹](dual_constraint_2d/BASELINE_MAP000_RESULT_20260926.md)
- [M100 官方与社区仿真平台核查及已选路线](M100_SIMULATOR_OPTIONS_20260925.md)
- 文献全文及本地提取文本保留在本机 `literature/`，不随 GitHub 代码分支发布；公开依据见各研究说明中的论文链接。
- [可用于后续实验规划的 Ascend 服务器资源快照](ASCEND_SERVER_INVENTORY_20260926.md)
- [16-worker GPU-MPC 运行中断与内存核查](GPU_MPC_V5_INTERRUPTION_20260926.md)
- [GPU-MPC H4/H8 完整 48 作业结果](GPU_MPC_V6_FINAL_RESULT_20260926.md)
- [下一阶段公平比较与 ARM 三维准备计划](NEXT_PHASE_PLAN_20260926.md)
- [独立 GitHub 分支与 ARM64 三维实验交接](SERVER_ARM_3D_HANDOFF_20260926.md)

用户当前先研究[二维碰撞与返站能量双约束](dual_constraint_2d/README.md)：在未见过的同类静态地图上，单无人机逐步选择局部航向/速度和补能，要求执行后仍有避碰且电量足够的返站方案；使用**明确标注的合成机制环境**、连续单目标任务、PPO 与规划基线检验机制。M100/1 km 与先前三维取送货方案保留为后续外部验证/扩展，不把它们的评价结果当成二维双约束结果。二维已实现确定性模型内的可执行返站轨迹核验，但尚无形式化或现实安全证明；最新运行状态以对应实验目录的 manifest、summary 和中断记录为准。
