# 学习与规划：研究定位与全文依据

日期：2026-09-23。用途：新课题讨论；不重开关闭项目，不启动训练，不修改观测契约。

## 检索和阅读范围

UnderMind 检索与元数据定位五篇论文，下载四篇 PDF；UVFA 在 UnderMind 无 PDF，转到官方 PMLR 下载。五篇完整 PDF 已导入 Windows Zotero 分类“学习与规划：AI方法基础”（U6LD68UM），附件 MD5 验证通过。元数据以官方会议记录为准，避免把预印本年份当正式年份。LOOP 是 CoRL 2021，PMLR 出版年份 2022；TD-MPC2 是 ICLR 2024，预印本 2023。

实际核读下列 PDF 正文段落，不声称逐页读完全部附录或独立核验所有定理。未执行 Lean；以下都是对既有论文的转述和未验证的研究建议。

## 论文与可迁移部分

1. Hansen, Su, Wang. Temporal Difference Learning for Model Predictive Control. ICML 2022. https://proceedings.mlr.press/v162/hansen22a.html
   - PDF pp.3–4，§3–4，Eq.(3)、Algorithm 1：有限时域预测奖励加终端 Q，在线滚动规划执行第一动作；学习潜在动力学、奖励和价值。p.6 核读基线和实验范围，包括有限预算 MPC:sim。
   - 支持学习与规划结合的领域定位；不支持照搬后即获得无人机能量保证，也不证明能优于充分求解的真实模型规划。
2. Sikchi, Zhou, Held. Learning Off-Policy with Online Planning. CoRL 2021 / proceedings 2022. https://proceedings.mlr.press/v164/sikchi22a.html
   - PDF pp.3–6，§3–6：H-step lookahead + learned terminal value；Theorem 1 陈述以模型误差、价值误差和规划误差为条件的界限。未验证附录证明。
   - §5.1 actor divergence：收集数据的规划策略和进行价值更新的 actor 不一致可能导致误差，提出 ARC。
   - §5.2 区分离线静态数据与在线交互；Safe-LOOP 约束规划窗口内的模型成本，并非自动提供整个配送窗口零耗尽保证。
3. Schaul, Horgan, Gregor, Silver. Universal Value Function Approximators. ICML 2015. https://proceedings.mlr.press/v37/schaul15.html
   - PDF pp.2–5，§2–4：V(s,g)/Q(s,a,g)，共享状态和目标的表示；实验包含未见状态目标组合和目标插值。
   - 原始形式的转移模型固定、目标改变；电池容量和到达率还可能改变转移或合法动作，因此迁移到 V(s,context) 是拟议扩展，不是原文已经证明的能力。
4. Berkenkamp, Turchetta, Schoellig, Krause. Safe Model-based Reinforcement Learning with Stability Guarantees. NeurIPS 2017. https://proceedings.neurips.cc/paper/2017/hash/766ebcd59621e305170616ba3d3dac32-Abstract.html
   - PDF pp.2–4，§2–3：确定性离散动力学、已知模型+未知误差、Lipschitz、校准置信区间、初始安全策略、Lyapunov 吸引域。
   - 支持带明确假设的学习与安全证书结合。稳定到平衡点与反复充电配送不同，不能直接移植为电量安全定理。
5. Hansen, Su, Wang. TD-MPC2: Scalable, Robust World Models for Continuous Control. ICLR 2024. https://www.tdmpc2.com/
   - PDF pp.3–4 §3：任务嵌入 e 条件下的世界模型、奖励、终端价值和策略先验；p.7 多任务训练与 held-out task finetuning 实验。
   - 证明该领域有多任务模型学习路线；多任务训练与新任务微调不等于任意未见资源配置零样本安全泛化。

## 当前定位建议

应用问题：随机动态取送货与补能调度。AI 方法主线：reinforcement learning with planning / learned terminal value functions for MPC。如果学习转移模型，属于 model-based RL；若转移模型已知，仅学习价值，可更准确称 value learning for model-based planning / approximate dynamic programming。不必为了 MBRL 标签学习已知物理。

安全 RL 描述约束维度；offline RL 描述固定训练数据条件；off-policy 不等于 offline；先训练再部署不自动属于 offline RL。希望跨配置使用，可研究 contextual/multitask RL 与泛化，但尚未选定具体算法。高层服务动作持续时间不同，后续建模需考虑实际时间、剩余 horizon、动作期间到达的订单，不直接套每订单一步的固定折扣目标。

用户已选：独立随机到达；可见已到达订单；希望跨容量/负载/充电速度泛化但逐步引入；允许规划器选动作；学习为核心贡献。已知地图为讨论候选，不能视作已修改原 LiDAR-only 约束。

可检验研究问题：有限在线规划预算下，能否学习跨资源配置的后继状态价值，帮助规划器处理任务顺序、位置价值和部分充电的长期耦合？这不是新颖性结论。现有论文证明方法谱系存在，不证明新的方法优越、理论成立或符合会议录用要求。

优先比较现有终端价值学习/规划方法；用相同信息、计算预算和安全机制隔离学习贡献。仅集成 TD-MPC + 安全过滤 + UAV，不能自称新的通用方法。
