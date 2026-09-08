# 独立调研报告：安全导航达标线 vs 领域惯例（2026-09-06）

调研方式说明：本会话 `web_search` 工具不可用（modsearch 全部引擎失败，家长已确认），改用 arXiv API、Semantic Scholar、OpenAlex、OpenReview API2、CrossRef 直连检索（合计 40+ 次查询），并下载核验 20+ 篇论文全文/HTML/PDF（含 C-TRPO/ProSh/CRAX/SafeOR-Gym 全文复核）。所有数值均来自原文表格，检索来源见文末。未能验证处已显式标注。第二轮已补齐 E 节六项短板。

## A) 对项目 navigation gate 的判断：偏严，且是"跨范畴"的严格

项目 gate：500 任务 overall success ≥0.98、分桶 ≥0.95、平均路径比 ≤1.10、障碍碰撞步 =0、边界接触步率 <0.01（冻结 R3：0.84 / 1.63 / 0 碰撞，FAIL）。逐项对照领域惯例：

**1. 碰撞验收：领域通行的是"cost budget"，不是零碰撞。** Safety-Gymnasium（NeurIPS 2023）的通行协议是 CMDP 预算：`cost_limit=25`，cost 定义是"每步接触障碍/hazard 计 1"，评测 = 10 轮评估 × ≥3 种子，报告"相对 cost_limit 归一化成本"（1.0=恰好花完预算）。顶会论文自己就大量超预算：SafeMPO（ICLR 2026，10M 步，B=25）在 SafetyCarGoal/CarButton/PointGoal 的平均 cost 为 32.23/30.87/32.92（原文作者承认"所有方法 cost≈30 vs 预算 25 是方法局限"，唯一合规的 RCPO 回报最低）；Langevin（ICML 2024，2.5M 步）表 2 中 CPO 在 PointGoal1 cost 46.0、CarGoal1 45.4。即"每集平均几十个碰撞步"在领域内被当作正常结果报告，零碰撞步是比通行标准严一个量级的要求。

**2. 成功率：学习型导航在未见/高杂波静态环境中 60–88% 是常态，98% 属极严。** 代表性数字（均为作者报告）：ReachNav（arXiv 2605.14174，静态障碍+感知不确定，Table I）TD3 success 76.7%±4.7 且 collision 15.0%±7.6，SAC-Lagrangian 64.0%±4.9 / 36.0%±4.9 / avg cost 3.52；Time-optimal safe RL（arXiv 2406.19646）未见高杂波场景 66.7%、已见场景才 100%；Safety-shielded 高速飞行（arXiv 2602.08653）87.5%→25%（3→9 m/s）；LiDAR 点云飞行（arXiv 2503.00496）4 m/s 时约 80%（其余方法 <40%）。100% success 只出现在受控/已见/低速设置（CurricRacing arXiv 2602.24030：3 条真实赛道 100%；TimeOptimal 的已见 Split-S 100%）。此外领域惯例是 success 与 collision 分开报（ReachNav 里 TD3 76.7% success 同时 15% collision），而项目 gate 的"success"是"无碰撞到达"这一更严的合取指标。

**3. 路径比 ≤1.10：严于文献校准线，但单项最接近可行。** 路径比 1.10 对成功轨迹等价于 SPL≥0.909。SPL 提出者 Anderson et al.（arXiv 1807.06757）明确写："SPL 是相当严苛的度量；在未见过的较复杂环境中，SPL≈0.5 就是一个好的导航水平。" 领域报告的成功 SPL 常在 0.5–0.9。项目 R3 历史最好 1.188/96% success，gate 1.10 只比自身历史再收紧约 8%——单项不是天文数字，但叠加 0.98 success 后是双重约束。

**4. 零碰撞 + shield-on 的地位：** 安全 filter 文献（SRLF arXiv 2410.06852"collision-free tracking"；CBF-RL arXiv 2510.14959；Safety Filtering While Training arXiv 2410.11671）确实以"shield 保证无碰撞执行"为卖点，且公开讨论 shield-on 的保守性/分布偏移问题（SFWT 原话：filter 的保守性造成 certified/uncertified 环境间的分布偏移）。领域共识：shield-on 是"安全执行"证据，**不是**"策略本身学会避障"的证据（项目 47/50 shield-on vs 26/24 shield-off 的数据正好是这一区分的例证）。Back to Base（arXiv 2501.02620）甚至把 filter 用作训练期安全重置。所以 gate 的"碰撞步=0"若指 shield-on 执行，与领域一致且可辩护；若要求 shield-off 策略零碰撞，则比领域任何惯例都严。

**5. 边界接触率 <0.01：** 未找到任何文献先例，是项目自造指标，无法做领域对标（标注：无法对标）。

**总体判断：** 五个条件组合起来比 safe RL 顶会论文的通行达标线（B=25 预算 + 归一化 cost 报告 + success/collision 分报）**明显偏严**；比导航 RL 论文的典型 success 报告线（60–88%）**偏严一档以上**。作为内部工程 gate 并非荒谬——静态 24 柱、4000 步期限、shield-on 属于"已见/简单"区间，0.98 有达到先例（CurricRacing 100%）；但它不是领域通行标准，不应被当作"论文必须达到的领域基线"来解读。当前 R3 0.84 的失败主因（方向丢失的编码器缺陷）是工程问题，与 gate 定标是否合理是两个独立问题。

## B) safe RL 导航论文典型达标线（均经原文核验）

| 论文（环境） | 方法 | success | cost/碰撞 | 预算 |
|---|---|---|---|---|
| Ray et al. 2019 OpenAI（Safety Gym SGPoint/SGCar，3 seeds） | PPO-Lagrangian | 不报 | violation 0.054/0.004（相对 PPO=1）；cost rate 0.278/0.238 | 训练终点，cost rate 为评估协议 |
| Safety-Gymnasium NeurIPS 2023（CarGoal1，≥3 seeds×10 轮，B=25） | FOCOPS 原实现 / SafePO 实现 | 不报（报归一化回报） | 归一化 cost 2.45/0.93；PointGoal1 SafePO 版 1.32（超） | 回报/成本归一化，无 success 指标 |
| Langevin ICML 2024（PointGoal1 / CarGoal1，2.5M 步，B=25） | PPO-Lag | 不报 | 16.48±10.95 / 19.50±16.92（合规）；FOCOPS 38.08±12.63（超）/ 20.09±9.20（贴线） | 2.5M |
| SafeMPO ICLR 2026（SafetyCarGoal/CarButton/PointGoal，10M 步，B=25） | SafeMPO | 不报 | 32.23/30.87/32.92（均超预算，作者承认） | 10M |
| Multiplicative Value IROS 2023（Point Robot Navigation 1D LiDAR，10 seeds×100 eps） | FOCOPS / SAC-Lag | 不报 | FOCOPS 50k viol 64±24 → 100k 30±19；SAC-Lag 0 | 50k–300k；Gazebo 真实导航 FOCOPS 不如 SAC/PPO |
| UAV River Following IFAC 2024（Unity 无人机河面跟随，EpC≤1） | FOCOPS | 不报 | FOCOPS 三难度回报最高、Easy 最低成本、cost rate=基线 65%（正面评价） | 训练期指标 |
| GUARD ICLR 2024（Goal_Point_8Hazards，2 seeds，arXiv 2305.13681） | TRPO-Lag / TRPO / CPO | 不报 | 平均 episodic cost 之和 M̄c=2.50 / 7.46 / 3.24；训练期 cost rate ρ̄c=0.0034/0.0067/0.0036 | 报告协议=J̄r、M̄c、ρ̄c 三指标，无 success/碰撞计数 |
| MAVRL RA-L 2024（杂乱环境深度图，10 seeds，Table I 经 PDF 提取） | PPO+记忆隐空间 | 0.500±0.037 ～ 0.707±0.051（最佳 checkpoint） | 不单报 | DOI 10.1109/LRA.2024.3522778 |
| ReachNav arXiv 2605.14174（静态障碍+感知不确定，Table I） | TD3 / SAC-Lag | 76.7%±4.7 / 64.0%±4.9 | 碰撞率 15.0%±7.6 / 36.0%±4.9；avg cost 3.52 | —（另报验证式 Safety(Succ.) 98.4%，非经验零碰撞） |
| Time-optimal Safe RL arXiv 2406.19646 | PPO+安全奖励 | 66.7%（未见）/ 100%（已见） | 不单报 | — |
| Shielded 高速飞行 arXiv 2602.08653 | RL+模型安全机制 | 87.5/75/42.5/25%（3–9 m/s） | 不单报 | — |
| Flying on Point Clouds arXiv 2503.00496 | LiDAR RL | ~80%（4 m/s，其余<40%） | 碰撞即终止 | — |
| CurricRacing arXiv 2602.24030 | 课程 RL | 100%（3 条真实赛道，8 m/s） | 不单报 | 含硬件在环 |

## C) FOCOPS / PPO-Lagrangian 在导航任务上的引文评价汇总（不看原论文自述）

- **FOCOPS：引文评价分化，无"广泛有效"共识，且强实现/任务依赖。**
  - Safety-Gymnasium（NeurIPS 2023）表 1 专列同名算法不同实现：CarGoal1 原实现 0.79/2.45，SafePO 实现 0.52/0.93（更合规但回报更低）；PointGoal1 SafePO 实现 cost 1.32 超预算。
  - Langevin（ICML 2024）：PointGoal1 38.08 超预算、CarGoal1 20.09 贴线——"大部分基线在 2.5M 步内难以找到满足预算的策略"。
  - Multiplicative Value（IROS 2023）：Point Robot Navigation 不如 SAC（viol 64→30 vs 0）；Gazebo 真实导航不如 PPO/SAC；Car Racing 无法过第一弯（reward −2±2）。
  - 正面：UAV River Following（IFAC 2024）称 FOCOPS"唯一在回报与成本率双超基线"（回报 1.4×、cost rate 65%）；CRAX（arXiv 2606.20376）Goal Level 2 回报 73.3/cost 24.6 合规（但 L1 25.4、L3 26.6 超，且正文 500M/附录 100M 预算不一致）。
  - CUP 原始比较（ICML 2021，家长前轮已核对表 1）：FOCOPS Ant cost 101.31/预算 103 合规、Hopper 102.30/83 超——同一张表内一半合规一半超。
- **PPO-Lagrangian：导航上常是更稳、低回报的合规基线。** Ray et al. 2019 原文以 0.24（SG18 回报）换 0.026 violation；Langevin 表 2 中 PointGoal 16.48、CarGoal 19.50 均合规且回报（11.36/12.38）高于 FOCOPS（6.01/7.91）。但 SafeMPO 表 1 中所有 Lagrangian 类基线在三个导航任务同样普遍超 25。**两者都没有"测试期零碰撞"的任何证据**——它们优化的都是平均预算。
- 其他方法速记（家长前轮已核，我复核出处）：Sauté 被后续引文描述为"更保守、部分 Goal 任务回报很低"（ProSh arXiv 2510.15720）与"较强基线"（CRAX）并存；C-TRPO（ICML 2025，PMLR v267，已验证链接 200）报告 TRPO-Lag/FOCOPS/CUP 高回报 ≠ 低训练违规；SafeMPO（ICLR 2026，我读原文）自述 cost≈30 vs 25。

## D) 实验纪律建议（对齐领域惯例）

1. **种子与轮数**：≥3 种子 × 多轮评测是基准下限（Safety-Gymnasium 3 seeds×10 轮；Multiplicative 10 seeds×100 eps；OmniSafe arXiv 2305.09304 默认多种子实验网格）。500 任务单集足够大，但建议报告跨种子。
2. **训练预算**：安全导航文献 2.5M–10M 环境步（Langevin 2.5M、SafeMPO/C-TRPO 10M、OmniSafe 1e7）。项目 500k gate 预算处于领域最低端，0.98 目标在此预算下尤其苛刻；若保留 500k，报告时应与文献预算差距一起声明。
3. **cost 三件套**：mean episodic cost（对照 budget）、violation probability（首次碰撞率）、training cost rate（训练期 regret，Ray et al. 2019 的论证）。只报均值会掩盖尾部风险——ReachNav 明确批评"平均累积 cost 掩盖尾风险"。
4. **成功与安全分报**：领域惯例是 success 与 collision 分开（ReachNav Table I）；"无碰撞到达"作为复合指标可用但必须说明它是严格于任何 safe RL 基线的合取。
5. **已知失败模式**：① 双低退化（过度保守 → 低 success + 仍不零碰撞：SAC-Lag 64%/36%；Sauté 保守；SafeMPO 讨论中 CPPOPID 合规但回报负、RCPO 合规但回报最低）；② 训练期超预算（C-TRPO 报告的 Lagrangian 类方法）；③ 归一化 cost<1 被误读为"很少碰撞"（实为"累计接触步数在预算内"）；④ shield-on 被误读为策略学会安全。
6. **指标语义对齐**：Safety-Gym cost=累计接触步、预算 25≈允许每集平均 25 个接触步；成功=到达目标（可带碰撞）。项目 gate 与文献不可直接换算，比较时须换算或声明。

## E) 不确定性与检索局限（第二轮补齐后更新）

1. **引用图**：S2 对 FOCOPS 索引严重不全（其 arXiv 版记录仅 4 条施引，重试确认）；S2 的 paper/search 端点持续 429，后台重试任务运行 30+ 分钟仍无产出后终止，改用 OpenAlex 兜底。OpenAlex：FOCOPS 52 条施引全量列出，其中导航/无人机/机器人直接相关 8 条（EPO-S SMC 2023、UAV River Following IFAC 2024、Multiplicative Value IROS 2023、Protective Policy Transfer ICRA 2020、航天器避撞 AST 2024、自动驾驶安全状态增强 2023/2025 等）；CPO/CUP/Sauté/Safety-Gymnasium 的 S2 引用列表（300/91/163/201 条）此前已抓取。
2. **出版状态核验（已补）**：ShieldFlight=IEEE RA-L 2026；SFWT=RA-L 2025；MAVRL=RA-L 2024（DOI 10.1109/LRA.2024.3522778）；FOPC=IROS 2025（DOI 10.1109/iros60139.2025.11246821）；TimeOptimal=Unmanned Systems 期刊 2026（DOI 10.1142/s230138502650007x）；C-TRPO=ICML 2025（PMLR 267，arXiv 注释确认）。**仍为预印本/评审状态未确认**：ReachNav（2605.14174）、CurricRacing（2602.24030）、CRAX（2606.20376）、SRLF（2410.06852，CrossRef 无记录）、ProSh（2510.15720，AAMAS 2026 extended abstract 状态未独立确认）。
3. **MAVRL 数字已补齐**：pdfplumber 提取 Table I——success rate 0.500±0.037（当前帧）至 0.707±0.051（当前+过去帧），10 种子最佳 checkpoint，已加入 B 表。
4. **GUARD 协议已补齐**：72 组结果统一三指标——平均 episode return J̄r、平均 episode cost 之和 M̄c、全训练期平均 cost rate ρ̄c；收敛末 epoch、2 seeds；例：Goal_Point_8Hazards TRPO-Lag M̄c=2.50 vs TRPO 7.46。**无 success rate、无碰撞计数**——确认领域"达标"口径是成本指标而非成功率，已加入 B 表。
5. **中文文献**：CNKI 返回 302（需登录）、百度学术触发安全验证页，均无法检索——中文文献标注未能验证。
6. **家长前轮表值独立复核（已补）**：C-TRPO（2411.02957v4 全文）：10M 步 × 5 seeds × 8 任务（4 Navigation+4 Locomotion），原文"TRPO-Lag., FOCOPS, and CUP perform well in terms of reward, but poorly in terms of cost regret""TRPO-Lag … oscillates more around the threshold during training"；ProSh（2510.15720v2 全文）：6 个 Safety-Gymnasium 任务（含 SafetyPointGoal/SafetyCarGoal 导航套件）、3 seeds、OmniSafe 超参，原文"PPO-Saute being in most cases the only algorithm that is safe during training""FOCOPS and CPO show less stable safety profiles, with costs that fluctuate even after extended training"；CRAX v2（2606.20376v2 全文）：FOCOPS Goal L1/L2/L3 = 37.2/25.4、73.3/24.6、63.6/26.6（预算 25，仅 L2 合规），正文 500M 步 × 5 seeds，v2 未再出现 v1 附录 100M 字样（版本间不一致仍标注）。家长前轮结论全部与原文一致，无矛盾。
7. **新增佐证**：AAAI 2023《Evaluating Model-Free RL toward Safety-Critical Tasks》（DOI 10.1609/aaai.v37i12.26786）明确指领域"缺乏在每个决策步都满足安全约束的高质量评估"（state-wise 口径）并推出 USL/SafeRL-Kit——项目 gate 的逐步碰撞=0 属于该少数派口径，主流仍是预算口径。
8. 其余关键数值（Langevin 表 2、Safety-Gymnasium 表 1、Ray 2019 表 1-2、SafeMPO 表 1、ReachNav 表 I、Multiplicative 表 I、Anderson SPL 定义、OmniSafe 1e7 协议、SafeOR-Gym 表 3）均为本会话直读原文或家长前轮直读并已复核出处，未见矛盾。

## 来源 URL

- Safety-Gymnasium: https://proceedings.neurips.cc/paper_files/paper/2023/hash/3c557a3d6a48cc99444f85e924c66753-Abstract-Datasets_and_Benchmarks.html ; arXiv https://arxiv.org/abs/2310.12567
- Ray, Achiam, Amodei 2019: https://cdn.openai.com/safexp-short.pdf
- Langevin (ICML 2024): https://proceedings.mlr.press/v235/lei24a.html
- SafeMPO (ICLR 2026): https://proceedings.iclr.cc/paper_files/paper/2026/file/1fa0c4e5a7e189729230d018b229abc7-Paper-Conference.pdf
- CUP: arXiv https://arxiv.org/abs/2209.07089 ; NeurIPS 2022 https://papers.nips.cc/paper_files/paper/2022/file/3ba7560b4c3e66d760fbdd472cf4a5a9-Supplemental-Conference.pdf
- Multiplicative Value (IROS 2023): https://arxiv.org/abs/2303.04118
- UAV River Following (IFAC 2024): https://arxiv.org/abs/2409.08511 ; DOI 10.1016/j.ifacol.2024.10.090
- ReachNav: https://arxiv.org/abs/2605.14174
- Time-optimal Safe RL: https://arxiv.org/abs/2406.19646
- Shielded 高速飞行: https://arxiv.org/abs/2602.08653
- Flying on Point Clouds: https://arxiv.org/abs/2503.00496
- CurricRacing: https://arxiv.org/abs/2602.24030
- SRLF: https://arxiv.org/abs/2410.06852
- Safety Filtering While Training: https://arxiv.org/abs/2410.11671
- Back to Base: https://arxiv.org/abs/2501.02620
- OmniSafe: https://arxiv.org/abs/2305.09304
- Anderson et al. (SPL): https://arxiv.org/abs/1807.06757
- C-TRPO (ICML 2025): https://proceedings.mlr.press/v267/milosevic25a.html
- CRAX: https://arxiv.org/abs/2606.20376
- ProSh: https://arxiv.org/abs/2510.15720
- GUARD: https://arxiv.org/abs/2305.13681
- VO-Safe (ICRA 2024): DOI 10.1109/ICRA57147.2024.10611487
- Evaluating model-free RL toward safety-critical tasks (AAAI 2023): DOI 10.1609/aaai.v37i12.26786
