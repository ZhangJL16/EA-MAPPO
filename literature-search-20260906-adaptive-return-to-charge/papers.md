# 关键论文明细（标题/作者/年份/venue/核心方法/与自适应返航的关系/评价来源）

"自述"=论文自身 claim；"引文评价"=检索到的引用该论文的文献片段（S2 contexts / OpenAlex 引用图）。置信度同 REPORT.md 图例。

## 能量 CBF / persistification
| 论文 | 作者 | 年份 | Venue | 核心方法 | 与返航决策关系 | 评价 |
|---|---|---|---|---|---|---|
| Persistification of Robotic Tasks Using Control Barrier Functions | Notomista, Ruf, Egerstedt | 2018 | IEEE RA-L (10.1109/LRA.2018.2789848) | 电池 SOC≥下限为 CBF 不变集，最小干预任务控制器 | 原型：能耗水平决定干预（隐含返航/减速） | 引文评价：仅无障碍环境（ES-CBF 2025）；模型/电池假设简单但计算快13倍（Eclares 2024）；单积分器模型（meSch 2025） |
| Persistification of Robotic Tasks | Notomista, Egerstedt | 2020 | IEEE TCST (10.1109/TCST.2020.2978913) | 期刊版，同上框架 | 同族 | S2 引用 51；后续覆盖/能量工作多为背景引用 |
| Energy Autonomy for Robot Systems With Constrained Resources | Fouad, Beltrame | 2022 | IEEE T-RO (10.1109/TRO.2022.3175438) | 多机器人共享充电站，能量 CBF | 回充可行性保证 | 引文评价：固定 SoC 阈值+单积分器（Adaptive ergodic search 2025）；被 meSch 批评限于单积分器 |
| Energy Sufficiency in Unknown Environments via CBF | Fouad, Beltrame | 2023 | arXiv 2306.15115 | planner 无关的 CBF 能量层+真机实验 | 叠加在路径规划器上的回充保证 | 引文评价：ES-CBF 称其仅无障碍环境 |
| ES-CBF | Fouad 等 | 2025 | Autonomous Robots (10.1007/s10514-025-10203-w) | 能量充分性 CBF 扩展到采样规划器（RRT*）、有障碍 | 显式回充保证 | 自述：克服前作无障碍限制 |
| A Novel Safety-Aware Energy Tank Formulation Based on CBF | Michel, Saveriano, Lee | 2024 | IEEE RA-L (10.1109/lra.2024.3389556) | 二阶能量罐+CBF，处理功率限 | 能量/功率约束 | 无负面引文记录（较新） |
| Eclares / meSch / Adaptive ergodic search | Naveed 等 | 2024/2025/2025 | ICRA 2024 (10.1109/icra57147.2024.10611286)；IROS 2025 (10.1109/iros60139.2025.11247605)；AuRo 2025 (10.1007/s10514-025-10215-6) | 能量感知调度：决定谁/何时回充 | 回充时机的显式调度（非学习门控） | ★Eclares 全文核实（arXiv 2310.06933）：CBF 法"computationally efficient; however, they only assume simple robot and battery models"；0.0551ms（单积分器）vs eware 30.2ms，换算快 13 倍；其电池动力学带控制输入项 "as opposed to the worst-case approximation"；有 eware 时 3% SoC 安全返站，无则耗尽坠机。meSch/Adaptive ergodic 为 S2 片段粒度（无 OA 全文） |
| Energy-Aware Coordination with Mobile Charging Stations | Nasif, Notomista | 2025 | MED (10.1109/med64031.2025.11073197) | 移动充电站+能量 CBF 协调 | 回充可行性 | 引用了 T-RO 2022 的能量动力学 |

## 回航/中止决策
| 论文 | 作者 | 年份 | Venue | 核心方法 | 与返航决策关系 | 评价 |
|---|---|---|---|---|---|---|
| Back to Base: Hands-Off Learning via Safe Resets with Reach-Avoid Safety Filters | Begzadić, Shinde, Tonkens, Hirsch, Ugalde, Yip, Cortés, Herbert | 2025 | L4DC (PMLR v283 begzadic25a；arXiv 2501.02620) | 学到的 HJR reach-avoid 值做安全过滤器，时限内返回基地，自动 reset | **最接近**：放弃任务返回基地的状态级切换 | 自述：hands-off 训练。★全文核实：S2 的 5 篇引用者中 4 篇为同组 UCSD（含直接后续 From Space to Time 2509.19597；Steering with Contingencies 2604.03405 为 Caltech+UCSD 合著）；唯一独立引用（UNH 2511.08419）仅 §1 背景。**无独立第三方实证评价**；"HJR 扩展性差"类评语系同组自评，已降级 |
| Mission abort policy (MAP) 系列 | Levitin, Finkelstein, Qiu 等 | 2017–2023 | RESS/EJOR/IEEE TR（如 10.1016/j.ress.2019.106496；10.1109/tr.2022.3172377；10.1111/risa.12886） | 退化/冲击模型下最优中止阈值（最优停时） | "何时中止任务"的数学理论 | 学科封闭；无学习、无导航 |
| Recovery Policies for Safe Exploration of Lunar PSR | 2023 | arXiv 2307.16786 | 太阳能月球车+随机故障的 recovery 策略 | 能量时空约束下的返回策略 | 自述为主，无引文评价 |
| Energy as a Concealable State in Adversarial UAV Patrolling | 2026 | arXiv 2608.26518 | 周期回基地+能量阈值+对抗 | 能量阈值语义 | 预印本，无评价 |
| There's No Place Like Home | Warren 等 | 2018 | arXiv 1809.05757 | GPS 失效时视觉教-重复紧急返航 | 紧急返航（感知侧） | 经典工作 |

## 能量感知 RL（学习的充电决策）
| 论文 | 作者 | 年份 | Venue | 核心方法 | 与返航决策关系 | 评价 |
|---|---|---|---|---|---|---|
| Learning to Recharge: UAV CPP through DRL | Theile, Bayerlein, Caccamo, Sangiovanni-Vincentelli | 2023 | arXiv 2309.03157 | PPO+地图观测，充电旅程并入覆盖策略 | 学习"何时充电"（网格地图层） | 被 14 篇后续工作引用为 recharge-CPP 基线，未见负面评价；非连续控制 |
| DRL Enabled Persistent Surveillance, Energy-Aware UAV-UGV | 2025 | arXiv 2502.02666 | 规划框架决定充电时机 | 回充调度 | 自述为主 |
| RL-Based Energy-Aware CPP for Precision Agriculture | 2026 | arXiv 2601.16405 | SAC+CNN+LSTM+充电站 | 学习充电决策 | 预印本，无评价 |
| Risk-Aware Energy-Constrained UAV-UGV Cooperative Routing (Ra-DRL) | 2025 | ICRA (10.1109/icra55743.2025.11128262) | 注意力 transformer RL 路由+风险 | 风险感知回充路由 | 新，无引文评价 |
| Hierarchical DRL Backscattering Data Collection, Multiple UAVs | 2020 | IEEE IoT-J (10.1109/jiot.2020.3024666) | 高层决策：继续收集或返航充电 | 二元返航决策（离散） | 高引用(140+)，未见机制批评 |
| Deep RL for Dynamic Battery Management of Autonomous Order Pickers | Defryn 组 | 2026 | arXiv 2607.05683 | PPO 学习充电决策（站选择+充电时长）vs 固定规则 | **与"固定阈值 vs 学习"直接对话** | ★全文核实：基线 FixedThreshold（固定上下阈值）与 HighLow（上阈值随队列长度变）；自述固定阈值启发式"myopic"、完成率 +6%；预印本，无第三方评价 |
| UBTP: Enhancing Sustainable Data Collection | 2024 | IEEE IoT-J (10.1109/jiot.2024.3461853) | 回程充电路径联合优化 | 返航路径规划 | 较新 |
| Optimizing Smart Wireless Charging … Battery Life Prediction | 2025 | IEEE TVT (10.1109/tvt.2025.3598601) | 电池寿命预测+充电决策 | 状态依赖充电决策 | 较新 |
| Trajectory design under insufficient UAV energy (staged actor-critic) | 2025 | J. Systems Architecture (10.1016/j.sysarc.2025.103566) | 分阶段 actor-critic 处理能量不足 | 阶段化返航 | 新 |
| Behavior Tree Battery-Aware Inspection | Rocamora 等 | 2024 | ICUAS (10.1109/icuas60882.2024.10557083) | 行为树电池感知 | 规则化返航 | 低引用 |
| Energy-Aware Navigation in Maze-Like Environments | Crespi 等 | 2026 | LNCS (10.1007/978-3-032-21811-7_3) | 探索/回充平衡 | 学习回充策略（小会） | 无评价 |
| Multi-UAV Monitoring with Priorities and Limited Energy | 2015 | ICAPS (10.1609/icaps.v25i1.13695) | 规划 | 早期能量受限监测 | 经典 |
| Constrained DRL for Energy Sustainable Multi-UAV NOMA IoT | 2020 | IEEE JSAC (10.1109/jsac.2020.3018804) | 约束 DRL 能量可持续 | 通信侧 | 高引用 |
| Energy-Aware Autonomous UAV Navigation via DRL (DQN/PPO/SAC, Battery-Constrained Reward) | 2026 | Research Square/Zenodo (10.5281/zenodo.20502034) | 电池约束奖励 | 低质量渠道，仅列不采信 | 【不确定】 |

## Reach-avoid / 约束 RL 外壳
| 论文 | 作者 | 年份 | Venue | 核心方法 | 关系 | 评价 |
|---|---|---|---|---|---|---|
| RC-PPO (Solving Minimum-Cost Reach Avoid using RL) | So 等 | 2024 | NeurIPS (hash 3750e99b；arXiv 2410.22600) | 增广动力学+HJ，确定性最小代价 reach-avoid | 把返航写成 reach-avoid | ★全文核实（RAPCPO 2605.11975 表1）：同迭代预算 PointGoal/FixedWing 到达率 62.29%/73.98% vs RAPCPO 78.49%/88.67%；"restricted to deterministic settings"；随机基准下 "similarly exhibits instability"（均为独立组实证评价） |
| RAPCPO | Pan, Wu, Xue, Xue | 2026 | arXiv 2605.11975 | 概率≥p 的随机 reach-avoid+期望代价 | 返航成功概率约束 | ICML 2026 状态未验证；无引文评价 |
| Safety and Liveness via Reach-Avoid RL | Hsu 等 | 2021 | RSS (arXiv 2112.12288) | 无 Lagrange 目标的 reach-avoid RL | 理论基础 | 高引用(104) |
| RCRL | Yu 等 | 2022 | ICML (PMLR v162 yu22d) | 最大可行集学习 | 可行集 | — |
| RESPO | 2023 | NeurIPS (dca63f26) | 迭代可达性估计 | 可行区 | — |
| QCRL | 2022 | NeurIPS (2a07348a) | 代价分位数约束 | 尾部风险 | — |
| Sauté RL | Sootla 等 | 2022 | ICML (PMLR v162 sootla22a) | 预算状态增广 | 电池状态增广非新 | — |
| DCRL | Qin 等 | 2021 | ICML (PMLR v139 qin21a) | 密度约束（含EV充电） | 资源约束 | — |
| Recovery RL | Thananjeyan 等 | 2021 | RA-L (IEEE 9392290) | 任务/恢复策略分离 | 恢复策略 | 与冻结策略门控最像的机制之一 |
| Almost-Sure Constraints | Castellano 等 | 2022 | L4DC (PMLR v168 castellano22a) | 最小安全预算不动点 | 预算 | — |
| Consumption MDP | 2020 | CAV (Springer 978-3-030-53291-8_22) | 有限资源+reload 的定性综合 | 电池/reload 语义 | — |
| Provably Safe RL for Stochastic Reach-Avoid (entropy reg.) | 2026 | arXiv 2601.08646 | OFU+熵正则，有限样本界 | 理论 | 预印本 |
| Joint Chance Constrained Safe-Optimal Control | 2026 | arXiv 2606.30829 | 联合机会约束 | 理论 | 预印本 |
| Predictive Safety Network | Choudhry 等 | 2019 | CoRL | 资源预测+中央安全策略 | 资源预测层次 | 引用少(2) |
| Steering with Contingencies: Combinatorial Stabilization and Reach-Avoid Filters | Lishkova, Ong, Tonkens, Ames 等 | 2026 | arXiv 2604.03405 | r-of-p 备选目标同时可达的组合 reach-avoid 过滤器 | **"返航期权保持"最新相邻**：动机句即 "remain within reach of charging stations as its battery depletes" | ★全文核实；引用 B2B 为"单 reset 区域"过滤器代表；MIP 编码 "scale poorly with the number of constraints" |
| Humanoid Safe Stop via Learned Stoppability Value | Long, Abbeel, Sreenath, Horowitz, Shi, Liu | 2026 | arXiv 2609.02358 | 应急停止的 reach-avoid 形式化：stop 策略+可停止性估计器 | 可恢复性（stoppability）估计 = 返航可行性的同类对象 | ★全文核实（独立组）："Classical HJ methods scale poorly with dimension, while learned certificates and reach-avoid RL approximate these values with neural networks" |

## 风险感知 / CVaR / 机会约束
| 论文 | 作者 | 年份 | Venue | 核心方法 | 关系 | 评价 |
|---|---|---|---|---|---|---|
| Risk-Aware Motion Planning via CVaR-Constrained Optimization | Hakobyan, Yang | 2019 | RA-L (10.1109/lra.2019.2929980) | CVaR 约束两阶段规划 | 能耗尾部风险工具 | 高引用(113)，成熟基线 |
| STEP | 2021 | RSS (arXiv 2103.02828) | 随机可穿越性评估 | 风险图 | 高引用(117) |
| DR-CVaR-Based Safety Filtering | 2024 | ICRA (10.1109/icra57147.2024.10611276) | 分布鲁棒 CVaR 安全滤波 | 滤波器侧 | 新 |
| Risk-aware UAV-UGV Rendezvous CC-MDP | 2022 | arXiv 2204.04767 | 机会约束 MDP+随机能耗+移动充电 | 随机能耗返航 | 自述为主 |
