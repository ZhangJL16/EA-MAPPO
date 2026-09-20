# 项目结项总结与迁移说明

日期：2026-09-20。状态：整个 regenerative-control 研究分支关闭。
这是现有资产交接，不是新项目立项，也不授权重新训练或开展第三次 pivot。

## 研究问题与真实任务

单无人机不断送货。充满电后强制做第一单；每次送完，在不知道下一单类型时决定
再做一单（C）或回去充电（R）。下一单为三种任务之一，各概率 1/3。
送货完成不 reset；充电有返航、停靠、overhead 和补能时间。电量、位置和时间连续。
R 不抽任务、不丢任务、不提前读取 task RNG。耗尽电量是失败，不免费救援。
低层 SAC 冻结，所有策略共享；固定受限地图和当前任务族。

## 最终结果与停止原因

主指标为任务价值/模拟时间；每单价值为 1，所以即任务数/秒。
精确参考是当前有限 safe task-prefix tree 的最优解，不是连续 UAV 世界的全局最优解。
C 只有在三种下一任务均能完成且立即返航成功时才合法；VB 在 C 合法时总是继续。

| 指标 | 结果 |
|---|---:|
| 最优长期吞吐量 | 0.010874671029 单/秒 |
| VB 长期吞吐量 | 0.010780521145 单/秒 |
| VB / optimum | 99.13422775% |
| 冻结的 KILL 阈值 | >=98% |
| distinct safe-but-return-optimal physical state | 1 |
| VB-only 模型证书的相对最优下界 | 98.61853328% |

36 个 cycle nodes；另 10 个 candidate subtree nodes 不算独立主树证据。
安全性由同一 robust admissibility 检查限定；导航 timeout 与耗能失败分开记录。
早期导航资格是 18/18 成功、0 接触的局部 smoke，不是全场景可靠性认证。

Learning 主线关闭：简单规则已达最优的 99.13%，复杂方法最大收益空间不足 0.9%。
理论主线关闭：Wald/renewal ratio、average-reward performance difference 和
prefix mass domination 足以还原核心命题，未识别不可替代的新技术步骤。
证书需要模型和 counterfactual R 成本，不是仅靠 VB 实际轨迹的统计认证。
用户对独立理论线最终评价 Level 1/5；可作 supporting analysis / thesis material。

这些结果只适用于当前模型，未隔离地图因果效应，不能断言所有持续作业问题都简单。
本项目保留负面证据，不通过更换任务、随机性或多机来挽救旧故事。

## 保留资产

- 代码：`research/regenerative_control/`、导航与 simulator 源码、`runtime_support/`。
- frozen SAC：`artifacts/hocbf_correction_sac_20260908_v1/sac/checkpoint_000131072/model.zip`。
  SHA256：`fb79482ae7d832b40137ff1a080ee84912eabb95867e118aca80c846f6ae62be`。
- P0：`artifacts/regenerative_p0_20260919/`，含 18 条检查与旧能耗头库存说明。
- P1/P1.1/P2-A1：`research/regenerative_control/evidence/` 内的压缩原始记录与报告。
- P2-A2：`artifacts/regenerative_p2a2_20260919/` 完整保留，包括快照；
  portable 副本位于 `research/regenerative_control/evidence/vb_theory_20260919/`。
- 推导：`DERIVATION_PACKAGE.md`，原文未改变。
- 新颖性审计：`docs/VIABILITY_BOUNDARY_NOVELTY_AUDIT_20260919.md`。
- 历史根目录文档原文：`docs/project_history_20260920/`。原相对链接仍按仓库根解释。
- 其余研究源码、Git 历史、Python 与 Lean 依赖保留在旧目录，不自动迁移为新课题。

## 本次实际删除

共 892 个文件，590,736,199 字节（563.37 MiB），详见
[逐文件删除清单](docs/cleanup_20260920/deletion_manifest.json)。

- P1/P1.1/P2-A1 已结束运行的 `.pkl` 续跑快照；压缩 outcome 证据保留。
- `new_navigation_energy_heads_20260908_v1/features.pt`：旧特征缓存。
- 同目录 `latest.pt`：旧 return-only mean/q95/defective 头；不是 frozen SAC。

这些文件已实际删除，清单中的哈希不是备份。对应旧运行不能原地续跑，
历史 `audit_p0.py` 也不能再重新读取已删除 energy head；应使用已保存的 energy_inventory。
保留的压缩证据足以审阅/重建相应 outcome 分析，但不声称完整恢复旧 pickle 状态。
P2-A2 完整树及其快照没有删除。未清理其他课题的原始数据。

核验：删除前 61 个 portable evidence 哈希全部通过；frozen SAC 哈希匹配。
本次不启动任何 simulator 或训练。依赖按默认保留，因此旧目录体积仍主要由依赖占用。

## 搬到新文件夹

准备的结项包路径为 `/home/zjl/mappo_closeout_20260920.tar.gz`，旁有 SHA256 校验文件。
内容为本次提交的 tracked source/docs/evidence、最终 SAC、P0 原始检查、完整 P2-A2。
不含 `.git`、`.venv`、Lean `.lake`、已删除产物及其他未跟踪旧实验目录。

建议新研究目录只先带本总结作为背景；旧资产作为单独 reference 保存，避免复制全部历史计划。
若需要复核结果，将结项包解压到一个独立的空 reference 目录；新研究不要沿用旧 AGENTS 的任务计划。
压缩包包含模型，但不包含依赖环境；重新执行导航需按 requirements 安装兼容依赖。
新课题名称、方向和代码骨架本轮没有替用户预设。
