# 当前任务状态

2026-09-20：项目整理与清理。研究分支已关闭，无待续跑训练任务。

- [x] 保留最终 frozen SAC、P0 原始检查、P1–P2 压缩证据、完整 P2-A2 树和证书。
- [x] 删除已结束实验的旧续跑快照、旧 return-only energy-head 权重及特征缓存。
- [x] 将历史首页、计划、笔记和 AGENTS 原文存入 `docs/project_history_20260920/`。
- [x] 当前入口改为 `PROJECT_CLOSEOUT.md`，不再展示历史启动指令。
- [x] 父目录剩余旧研究资产归入 `archive/parent_workspace_20260920/`，保留依赖及恢复映射。

下一阶段：用户在新目录开展独立 problem selection。没有选定新方法或启动新实验。
历史任务见 `docs/project_history_20260920/task_plan.md`，不作为当前授权。

## 2026-09-20 独立 F0–F2 审计授权

- [x] 以用户指定 `envs/UAVEnergyDeliverySAC.py` 完成 F0，见 `FULL_PROBLEM_AUDIT.md`。
- [x] 16个reset状态核查：单目标、空订单池；零训练、零policy steps。
- [ ] F1 长轨迹任务选择统计：当前实现没有多候选订单，未执行。
- [ ] F2 完整调度性能比较：缺少调度state/action接口，未执行；未擅加机制。

当前无后台运行。后续需完整调度入口或新的机制实现授权，不自动进入F3。

## 2026-09-20 后续用户授权：PersistentUAVThroughput-v1

- [x] 修订规格：仅actual depletion风险；空队列forced wait；独立物理calibration冻结27个regime。
- [x] 独立实现 `/home/zjl/persistent_uav_throughput_v1/`：环境、公共随机流、B0–B5和恢复检查点。
- [x] 16项针对性检查、冻结SAC小型执行与真实飞行中恢复等价检查通过。
- [x] 启动1000条物理calibration，首轮checkpoint健康核查通过，运行交还用户。
- [ ] calibration完整结果与参数冻结：由该运行产出，不监控到完成。
- [ ] 正式baseline validation/evaluation：尚未启动，等待完整calibration产物后显式推进。

未训练网络；未启动MPC、OracleSafe-SJF或small oracle。此授权不重开旧P2-A2研究。

## 2026-09-20 最新审阅修订

- [x] 用户要求新代码放回当前仓库的 `persistent_uav_throughput_v1/`，不使用嵌套Git/submodule。
- [x] 空队列允许recharge/idle；站内满电且无任务才强制idle；非空队列仍无idle。
- [x] 保存原始v1.1 source bundle、manifest、16项检查及93/1000中断前检查点。
- [x] 发现并修复固定地图起点被旧生成保护距离误拒的问题，不改地图或1000对输入。
- [x] 22项针对性检查、真实serve→serve→recharge→serve与完整恢复结果等价通过。
- [x] 资格筛查规则已在本轮读取结果前写定；pilot已开始，故不称采集前预注册。
- [ ] 当前同一1000对任务从保留检查点继续，完成后提交资格报告及27-regime manifest。

正式B0–B5入口已暂停，等待用户审查calibration/qualification后重新授权。
