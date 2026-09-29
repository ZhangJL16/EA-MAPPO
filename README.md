# 持续任务流 UAV：当前研究入口

当前活跃研究位于 [`research_projects/persistent_uav_feasibility_20260929/`](research_projects/persistent_uav_feasibility_20260929/CHARTER.md)：在原始固定地图、冻结导航和 27 个 regime 下，检验单机持续到达任务的选单、补能与完整运行窗口内的实际耗尽风险。完整模型规范见 [`MINIMAL_PERSISTENT_THROUGHPUT_SPEC.md`](MINIMAL_PERSISTENT_THROUGHPUT_SPEC.md)，只读物理与历史实验证据在 [`persistent_uav_throughput_v1/`](persistent_uav_throughput_v1/README.md)。当前实验代码和结果都写在独立 `research_projects/` 目录，不改旧物理源码。

旧 regenerative-control 分支已经结项；历史结论简述见 [`PROJECT_CLOSEOUT.md`](PROJECT_CLOSEOUT.md)。2026-09-29 按用户要求清理了与当前路线无关的旧实验、归档和大部分论文库；删除清单及恢复边界见 [`docs/cleanup_20260929/`](docs/cleanup_20260929/status.json)。清理清单只有路径和哈希，**不是备份**。当前冻结导航模型、所需运行源码、原 27 regime、B4/B5 与 Atlas/Sweep 证据以及必要的文献索引保留。

当前完整窗口固定策略筛查仍是 CPU 工作负载，服务器交接见当前项目的迁移说明。不要从旧历史计划自动启动训练或其他实验。
