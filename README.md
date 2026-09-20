# 项目结项：持续作业无人机与 regenerative control

**状态：研究已关闭，作为实验与复现资产保留。** 下一阶段在新目录重新选择研究问题；此目录不自动启动训练、实验或第三次 pivot。

先读 [项目总结与迁移说明](PROJECT_CLOSEOUT.md)。

父目录旧资产现已全部收回 [archive/](archive/README.md)，包括外部实验环境、参考源码和结项包。
全局隐藏工具目录保留原位；新研究可另开干净目录。

| 入口 | 内容 |
|---|---|
| [最终实证结论](docs/VIABILITY_BOUNDARY_THEORY_20260919.md) | VB 达到当前 safe-tree 最优吞吐量的 99.13%，触发预设 KILL |
| [新颖性审计](docs/VIABILITY_BOUNDARY_NOVELTY_AUDIT_20260919.md) | 理论是经典工具的短特化，独立论文主线关闭 |
| [原始推导](DERIVATION_PACKAGE.md) | 冻结保存，支持材料，不代表活跃研究计划 |
| [压缩实验证据](research/regenerative_control/evidence) | P1、P1.1、P2-A1、P2-A2、证书与文献核验 |
| [清理记录](docs/cleanup_20260920/deletion_manifest.json) | 已删除文件、大小、SHA256 与保留范围 |
| [历史首页与计划](docs/project_history_20260920) | 清理前原文；其中相对链接按原仓库根目录解释 |

当前导航模型保留在 `artifacts/hocbf_correction_sac_20260908_v1/sac/checkpoint_000131072/model.zip`。
Python/Lean 依赖保留在本机；迁移包不含它们，也不含 Git 历史。
