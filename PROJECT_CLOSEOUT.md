# 已关闭的 regenerative-control 分支

这个分支于 2026-09-20 结项，不再作为当前研究路线。原问题是单 UAV 在有限安全任务树内，于服务后选择继续或返站充电；冻结低层导航。旧结果中，VB 的长期吞吐量为 `0.010780521145` 单/秒，有限 safe-tree 参考最优为 `0.010874671029` 单/秒，比例 `99.13422775%`，达到当时预设的 `>=98%` 停止阈值。该比例仅适用于旧有限模型；不能外推为当前持续随机到达任务流的学习上限。

用户随后另立 PersistentUAVThroughput-v1，保留单机、冻结导航、充电站，但加入持续随机到达、最多五项待办和固定运行窗口。它是当前研究所依赖的只读源码与证据，位于 [`persistent_uav_throughput_v1/`](persistent_uav_throughput_v1/README.md)。2026-09-29 的活跃复检在 [`research_projects/persistent_uav_feasibility_20260929/`](research_projects/persistent_uav_feasibility_20260929/CHARTER.md)。两题的初始分布、动作语义和结论不混用。

2026-09-29 用户要求直接清理仓库的其他路线。旧 `archive/`、旧 regenerative 运行目录、旧 Lean 构建和无关论文库已删除；没有可声称的外部备份。此前删除记录保留在 [`docs/cleanup_20260920/deletion_manifest.json`](docs/cleanup_20260920/deletion_manifest.json)，本次逐文件哈希清单保留在 [`docs/cleanup_20260929/`](docs/cleanup_20260929/status.json)。原 Git 历史是否仍含部分 tracked 内容与本地删除清单是两个概念；未跟踪旧文件不能据哈希恢复。
