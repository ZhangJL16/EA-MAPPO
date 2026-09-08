# Notes: 独立审阅 — Return-to-Charge / 安全导航 / oral 级创新判断

日期：2026-09-06。本目录是独立于 codex 研究的第二视角审阅。

## 仓库现状速览（已读文档）

### 顶层目标
- 论文：ICLR 2027 投稿《When Should an Agent Return? Reliable Resource-to-Go under Closed-Loop Composition Shift》
- 顶层问题：资源受限智能体何时应不可逆停止任务并返航/回充（stranding–throughput 权衡）
- 核心 ML 问题（当前版本）：policy∘safety-filter 组合（"executed interface"）下，charger Resource-to-Go 的可辨识性与选择性分布预测；错误只在"返回边界"附近改变决策
- 理论包（已完成本地证明，未独立审计）：可辨识性/不可辨识性、occupancy-weighted 长时域误差传输、返回边界稳定性、SSP 截断、EIRR/风险商等 33 条结果（被 protocol 自己批评为 theorem sprawl）
- 实验：全部 PENDING。原因：导航前置 gate 全部失败（R1–R5）

### 用户核心诉求（来自 ORIGINAL_GOAL 与多次决策）
- 在合适的能耗选择返航，而不是固定阈值或仅靠远近评估
- 想要 AI 顶会（ICLR）oral 级论文

### 安全导航现状（2026-09-05~06 活跃执行中）
- JSEB 500k gate：success 0.84（要求 0.98）、路径比 1.63（要求 ≤1.10）、碰撞/边界通过 → FAIL
- 冻结 R3：shield-on 47/50 安全到达；shield-off 26/24 → 自主避碰能力不足
- reach-avoid SAC 微调两条分支（hard/continuous 50k）：安全到达 7/50、1/50 → 导航塌缩
- 发现旧 LiDAR 全局 mean/max 池化丢失绝对方向信息（四方位诊断：MSE 0.50 vs 0.000258，角误差 90° vs 0.75°）
- 新路线：从随机初始化，方向保留编码器（有序 2×16 读出）+ SB3 PPO 试跑：262144 步、50 任务 → 29 到达 / 20 接触 / 1 超时（2026-09-05）
- 当前执行：PPO-Lagrangian 慢乘子对照（directional_lagrangian_slowdual_pair）完成 → 冻结评估 200 任务：随机动作下原对照到达 36/50、无碰撞 23/50、平均碰撞 704 次
- 2026-09-06 启动：recovery_ppo_baseline（去掉旧成本头辅助损失的 reward-only PPO 续训 32 组 256 任务；进行中 cohort 90/96）
- 安全合同（冻结）：首次接触即任务失败；J_C=E[Σc_t]=P(接触)；目标 J_R 最大化且 J_C≤δ（δ 未定，0.05 为探索配置）；不折扣、有限期限

### 能量/返航决策方向的自我审计结论（重要）
- uav_safety_energy_novelty_audit：电池回充可行性 CBF（Candidate C）与 Notomista persistification、Fouad 2023、Dan 2021 直接重叠 → novelty rejected
- DUAL_VIABILITY_LITERATURE_REVIEW（2026-09-05）：Recovery RL、Leave No Trace、SAILR、Reach-Avoid RL、RCRL、Back to Base (L4DC 2025)、Consumption MDP、CRC、置信序列均为先例；剩余新颖点只能是：executed-interface 下三对象（完成可行、立即返航可行、单步 option 保留）的成对反事实识别 + 预算单调 option-preservation value + 整轨迹风险定理 + 实证避免"过早放弃任务"
- research-question-card.md：最小决定性研究 = 310 锚点 × (1 立即返航 + 10 冻结候选动作各 1 步 HOCBF 后返航) = 3410 rollouts，比较与"完成+返航"分支的语义不一致
- RETURN_TO_CHARGE_ICLR_RESEARCH_PROTOCOL §10.5：oral 收缩为三个主定理块（可辨识边界、商-瞬态信息相位、对偶停时 Pareto 有限样本证书）；"如果 Oracle headroom <5% 则停止 learned-manager 论文；纯理论路线更适配 COLT/AISTATS"
- ccfa 评审（2026-08-28）：3/10 reject at current evidence；正鉴定理近似重述 exact-agreement 假设；无任何实验

### 我的初步判断（待文献佐证）
1. 安全导航 gate 要求（98% 成功、零碰撞、路径比≤1.1）在数值上并不离谱（静态障碍+shield 评估时不少论文报告类似量级），但"零碰撞/98% 成功"作为**前置必要条件**把整条能源线卡死，属于工程排序问题，而非论文科学问题。
2. 即使安全导航达标，它只是实验平台（benchmark/precondition），不是 oral 级核心创新——仓库自己已正确判断这一点。
3. 用户关心的核心问题（自适应返航决策）目前存在两种候选路线：
   - 路线 A（现 paper）：executed-interface Resource-to-Go 预测 + 可辨识性理论 + 选择性预测（ML 视角，理论重、实验待补）
   - 路线 B（research-question-card 最新）：option-preservation Q^κ_op + 预算单调 critic + 整轨迹风险（更贴近用户"保留返航选项"的直觉，但先例多、区分度待证）
4. 风险：Back to Base (L4DC 2025)、Recovery RL、predictive safety network (Guo&Bürger)、DCRL 等已占据"资源安全 + 返航"的大量位置；纯"自适应返航决策"若无新机制会被视为组合。

## 待文献确认的问题（已全部确认，见各子代理报告）
- 领域内"return-to-charge 决策"的最新 SOTA 与评价 → `literature-search-20260906-adaptive-return-to-charge/REPORT.md`
- safe RL 导航的典型达标线（评估本 gate 严苛度）→ `navigation-gate-benchmark-report-20260906.md`
- oral 级"理论+单仿真器"论文的接受先例 → `oral-feasibility-literature-report.md`

## 最终新增关键事实（2026-09-06 独立核验）
- ICLR 2027：abstract 截止 2026-09-18，全文 2026-09-25（iclr.cc 核实）→ 19 天内 oral 证据链不可完成。
- 论文 related work 欠引：Back to Base、RC-PPO、RAPCPO、Recovery RL、SAILR、RCRL、Hsu RSS21、Sauté、Consumption MDP、energy-sufficiency CBF、2026 stoppability 系列全部缺失（grep 确认）→ 最便宜的必修项。
- Back to Base 引文（S2）：5 篇均为机制借用/扩展，批评 HJR 扩展性；OpenAlex 记 0 引。
- 2026 直接竞争：Humanoid Safe Stop via Learned Stoppability Value（Long, Abbeel, Sreenath, Shi, Liu 等，arXiv 2609.02358）；Learning Safe-Stoppability Monitors（arXiv 2026）；Steering with Contingencies（arXiv 2604.03405）；Safe Stochastic Explorer（arXiv 2602.00868）。
- Recovery RL 引文（S2，40 条）：2026 年大量工作作为 baseline/扩展（UniIntervene、AeroDPO、SafeExplorer、CRRL、TALB-MAPPO…），"分离式恢复"已成事实标准底座，不能作 novelty。
- R3 电池标定 + 100 轮耐力 PASS；能量-界面机制证据：executed-interface 汇总改善 total-energy MAE 3.90% [0.62%,5.14%]（不可部署，作为机制正信号保留）。
- recovery PPO 续训终结果：det 8 到达/4 无碰撞/42 超时、均值碰撞 818.5 → 比基线（25 无碰撞）更差；codex 已做 GAE/MC 符号分歧诊断（27-38% 分歧，MC value R² 0.15-0.41）。
- 三个子代理独立收敛：gate 偏严、位置错误；A 线 (a) 是唯一未占位点；现状不可 oral、全过预期 poster、oral 需耦合定理+分离构例+下界+预测性实验。

## 子代理 1 全文精读补充（第二轮交付，已并入 REVIEW_AND_ROADMAP.md §4.2）
- Back to Base 引文 4/5 为 UCSD 同组自评（From Space to Time=原班人马；Safe Stochastic Explorer/Learning to Nudge=UCSD；Steering with Contingencies=Caltech Ames+UCSD）；唯一独立引用（UNH 2511.08419）仅背景一句 → "HJR 扩展性差"等不能当独立评价；该线几乎无独立实证评价。
- RAPCPO 表1硬数字：同预算 RC-PPO 到达 62.29%/73.98% vs RAPCPO 78.49%/88.67%；"RC-PPO similarly exhibits instability"（随机基准）；"restricted to deterministic settings"。
- Eclares 全文：CBF 0.0551 ms/次 vs eware 30.2 ms/次（13x），但 "only assume simple robot and battery models"；实验 3% SoC 安全返站 vs 无滤波耗尽坠机。
- Order Pickers 全文：基线 FixedThreshold/HighLow；PPO 完成率 +6%；"charge only when battery falls below a fixed threshold … myopic" —— "固定阈值 vs 上下文充电决策"最直接公开实证（离散、无安全滤波、预印本）。
- Steering with Contingencies 动机句明确 "remain within reach of charging stations as its battery depletes"（r-of-p 组合 reach-avoid）；Humanoid Safe Stop 为独立组（Long/Abbeel/Sreenath/Liu 等）的"可停止性值"。
- meSch/Adaptive ergodic/ES-CBF/energy tank 无 OA 全文，评价仍为 S2 上下文粒度；MAP 文献正文未读。

## 子代理 2 第二轮补齐（已并入 REVIEW_AND_ROADMAP.md §2.2）
- FOCOPS 引文：S2 严重欠索引（arXiv 版仅 4 条）；OpenAlex 52 条全量列出。
- 出版状态核验：ShieldFlight=RA-L 2026；SFWT=RA-L 2025；MAVRL=RA-L 2024；FOPC=IROS 2025；TimeOptimal=Unmanned Systems 2026；C-TRPO=ICML 2025；仍预印本：ReachNav、CurricRacing、CRAX、ProSh。
- MAVRL：杂波环境 10 种子 success 0.500–0.707；GUARD 协议：只报 episode return/平均 episode cost/训练期 cost rate，无 success/碰撞计数。
- AAAI 2023 Evaluating Model-Free RL toward Safety-Critical Tasks：领域缺逐步满足约束的高质量评估（USL/SafeRL-Kit）→ 本项目逐步碰撞=0 是少数派 state-wise 口径，可引用。
- C-TRPO/ProSh/CRAX/SafeOR-Gym 表值直读复核通过，与仓库前轮结论无矛盾。
