# 检索笔记（2026-09-06，adaptive return-to-charge）

## 工具与端点
- web_search / modsearch：不可用（父代理确认）。
- arXiv API：http://export.arxiv.org/api/query（search_query 与 id_list）。
- Semantic Scholar：paper/search（早期大量 429）、paper/{id} 与 /citations（fields=title,year,venue,contexts,externalIds；慢速后台抓取，16–22s 间隔）。
- OpenAlex：works 搜索 + filter=cites:/title.search/title_and_abstract.search + DOI 查询。
- Crossref：query.bibliographic 精确题名核对。
- URL 核验：HTTP HEAD/GET 状态码（IEEE DOI 返回 202 视为有效）。

## 执行的查询（摘要）
- arXiv abs 查询 20 组：return-to-base；abort+battery；persistent monitoring+RL+energy；recharging+RL+UAV；energy sufficiency+CBF；energy-to-go；remaining battery+return；CVaR+energy+planning；chance-constrained+battery；safe return+UAV；remaining flight time；battery-aware+RL；energy-aware+navigation+RL；reach-avoid+battery；energy budget+learning；charging decision+RL；return+energy threshold+robot；mission abort 等（其中 4 组因 429 未完成，由 OpenAlex 补位）。
- OpenAlex 相关性检索 25+ 组（见 raw/O*.json、T_*.json、U*.json、V*.json、W*.json）。
- S2 单篇引文挖掘 10 篇关键论文（btb、persist_ral、persist_tcst、energy_autonomy_tro、esu_cbf、es_cbf_auro、rcppo、learn2recharge、rapcpo、task_persist_icra22），含引用上下文 contexts。
- OpenAlex cites 列表 10 篇关键论文。

## 证据文件
- raw/*.json / *.xml：全部原始检索结果（含摘要、引用上下文）。
- raw/oa_ids.json、oa_cites_*.json：关键论文的 OpenAlex 引用图。
- s2_*_citations.json：S2 引用者+contexts。
- REPORT.md、papers.md：最终输出。

## 已排除/降权
- MDPI/Research Square/Zenodo/Preprints.org 渠道（除非仅作发现线索）。
- 电网 EV 充电、通信资源分配等非导航类工作（仅列代表性）。
- 2026 预印本：采信但标注【不确定】。

## 未完成/失败项
- S2 search 端点大部分查询 429（改用 OpenAlex/arXiv 补位）。
- arXiv B1–B4（chance-constrained energy planning / risk-aware battery / optimal stopping battery / mission abort）429 未补跑；对应主题已由 OpenAlex（MAP 文献、CC-MDP、CVaR 谱系）覆盖。
- Google Scholar、IEEE Xplore 全文未读取。
- RAPCPO ICML 2026 收录状态未能验证。

## 补记（2026-09-07）：全文精读升级
应父代理要求，把"引文评价"从 S2 contexts 片段升级为对引用论文的全文精读：
- ar5iv 全文（正文抽取后检索引用段）：2604.03405（Steering with Contingencies）、2602.00868（Safe Stochastic Explorer）、2601.02686（Learning to Nudge）、2511.08419（Probabilistic Safety Guarantee）、2509.19597（From Space to Time）、2605.11975（RAPCPO）、2607.05683（Order Pickers）、2309.03157（Learning to Recharge）、2310.06933（Eclares 预印本）。
- arXiv PDF（pypdf 抽取）：2609.02358（Humanoid Safe Stop）。
- 被反爬/无 OA 全文，维持 S2 contexts 粒度：meSch（IROS 2025）、Adaptive ergodic（AuRo 2025，Springer PDF 拦截）、ES-CBF（AuRo 2025 非 OA）、energy tank（RA-L 2024，IEEE ielx7 PDF 拦截）。
- 关键修正：Back to Base 的 S2 引用者中 4/5 为作者同组（UCSD Herbert/Yip 组）或合著（Caltech+UCSD），唯一独立引用仅背景性质 → 原"引文评价"降级为同组自评；独立第三方实证评价暂不存在。
- 新增全文级证据：RAPCPO 对 RC-PPO 的表 1 数值与"instability"评价；Eclares 对 CBF 线的 13 倍速度/模型假设评价与 3% SoC 返站实验；Humanoid Safe Stop 对经典 HJ 的扩展性评价；Order Pickers 的 FixedThreshold/HighLow 基线与 +6% 完成率。
- 逐段摘录：raw/fulltext/_citation_passages.txt（早期）与 REPORT.md B 节 ★ 项（最终版）。
