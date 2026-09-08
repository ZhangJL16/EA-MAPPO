# Task Plan: 独立审阅 Return-to-Charge 方案（2026-09-06）

## Goal
深度审阅仓库论文方案，评价安全导航要求合理性与 oral 级创新潜力，输出修正路线文档。

## Phases
- [x] Phase 1: 通读仓库核心方案文档（研究卡片、论文、协议、gate 结果、notes/task_plan）
- [x] Phase 2: 检查实验失败证据（JSEB/R3/RACT/directional PPO/recovery PPO/dual-viability 诊断）
- [x] Phase 3: 并行文献调研 ×3 子代理（能耗返航决策 / 安全RL达标线 / oral 创新构成）
- [x] Phase 4: 独立一手验证（Back to Base、RC-PPO、RAPCPO、stoppability 系列、S2 引文评价、ICLR 2027 截稿日）
- [x] Phase 5: 撰写 REVIEW_AND_ROADMAP.md 并整合三份子代理报告
- [x] Phase 6: 交付（chat 总结 + 文件索引）

## Key Questions
1. 安全导航 gate（98%/零碰撞/路径比≤1.1）合理吗？ → 偏严于领域惯例但可作内部工程 gate；错误在于被当成论文科学前提。
2. 满足后是 oral 级核心创新吗？ → 否。导航是平台；oral 取决于 executed-interface 可辨识性定理的锋利度 + 预测性实验。
3. 用户核心问题（能耗自适应返航）在文献中位置？ → 固定 SOC 阈值被引文评价为能量 CBF 线公认局限；学习充电时机已有 Learning to Recharge 等；未占位=executed-interface+killed 首达资源律+不可逆返航的组合。
4. 下一步怎么走？ → 三个月计划（ICML 2027/RSS 2027 主攻）；ICLR 2027 截稿 09-25，oral 证据链 19 天内不可完成，默认跳过。

## Decisions Made
- 独立目录 `independent-review-20260906/` 与 codex 并行，不改动其产物。
- 子代理 web_search 失效时改用 arXiv/S2/OpenAlex/OpenReview/Crossref 直连（已验证可行）。
- 最终建议：平台 gate 与部署 gate 分离；恢复策略修复优先；随机性注入；Oracle headroom 重启；pair-vs-interface 机制先行；SIRP 按需启用；理论收缩+加锋；related work 补齐最近邻居。

## Errors Encountered
- web_search modsearch 全引擎失败 → 通知子代理改 curl 直连。
- S2 频繁 429（部分 ID 解析到错误论文，rcrl/sailr 引文不可用）→ 用 OpenAlex/子代理数据补位，报告已注明。
- arXiv API 间歇 429/502 → 降低频率、重试、转 OpenAlex。
- 两个 bash 长 sleep 被 300s 上限截断 → 改用 list_agents 检查子代理状态。

## Status
**全部阶段完成**。交付物见 REVIEW_AND_ROADMAP.md（主报告）、navigation-gate-benchmark-report-20260906.md、oral-feasibility-literature-report.md、literature-search-20260906-adaptive-return-to-charge/。
