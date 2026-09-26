# 补能策略退化：全文文献与代码诊断

日期：2026-09-26。使用用户已连接的 UnderMind 工作区
`e8dceead-a3eb-4789-aa94-992f328239fa` 完成专题深度检索，并用
`read_pdfs` 阅读下列论文全文。Bischoff、Theile、Dou、Pardo、Fouad、
Yu 的 PDF 也下载到本地 `literature/charging_learning_20260926/`；
该目录按仓库忽略规则不进入 Git。检索共有 182 条候选，**没有声称
逐篇阅读**。本记录只使用已核实的相关论文，不从摘要推断实验效果。

## 对当前失败最有解释力的证据

1. **固定时窗被误作外加截断。** Pardo et al., *Time Limits in
   Reinforcement Learning* (ICML 2018, arXiv:1712.00378) 区分“固定
   时间本身就是任务目标”与“只是训练采样截断”：前者在终点不应
   bootstrap，且应观测剩余时间。我们的观察已有剩余时间，
   `DualConstraintGym.step` 却把 600 秒终点标为 `truncated=True`；
   本机 SB3 2.8.0 的 `on_policy_algorithm.py` 会对这种转移额外加入
   `gamma * V(terminal_observation)`。这是已核实的训练定义错误，
   但不能单独证明它造成全部充电退化。
2. **动作可用性掩码不等于安全可行性。** Theile et al., *Learning to
   Recharge* (arXiv:2309.03157), Sec. IV-A/Eqs. 16–17 使用从下一
   状态到着陆点的最低电量要求构建 invariant mask；其 Fig. 5 对比
   有效动作掩码与返站可行性掩码，后者降低坠机并帮助学习。我们的
   `ModeMaskedPolicy` 只消除了站内/空中动作别名，未约束低电量出发。
   我们已有可执行返站证书，可在以后明确标注的安全训练条件中试用；
   不能把论文的离散网格安全结论直接当作连续二维或三维保证。
3. **反复补能与状态循环需要诊断记忆。** 同文 Sec. IV-C/Eq. 20
   使用衰减位置历史，Table III 报告了去除历史后的循环和成功率差异。
   当前 PPO 观察有剩余时间，却没有“上次返站后是否推进目标”等历史。
   这是候选解释，不应在修正终止标记前同时加入新特征。
4. **变时长动作需要按时间计价。** Precup, Sutton & Singh,
   *Theoretical Results on Reinforcement Learning with Temporally Abstract
   Options* (ECML 1998, DOI:10.1007/BFb0026709), Eqs. 1–2 用
   `gamma^T` 处理一个 option 持续 T 个原始步的后续价值。我们当前
   PPO 每个高层决策统一折扣 0.995，一次数十秒充电和 0.5 秒飞行
   都只折扣一次；这不等于按物理时间折扣。若目标是 600 秒内的
   **未折扣**完成数，可比较 `gamma=1`、正确终止的条件；若使用
   `gamma<1`，应实现并测试半马尔可夫回报，而非只改一个奖励系数。
5. **奖励/动作设计的权衡是已知现象。** Bischoff et al.,
   *Reinforcement Learning for AMR Charging Decisions* (arXiv:2505.11136),
   Sec. 4–5 对 PPO 比较不同奖励、动作和充电中断设计。其低于 20%
   必须充电是预设固定阈值，不适合直接照搬到本研究；更合适的是
   用路线与能耗证书导出的状态依赖可行性。其结果也提示强引导设计
   可能牺牲泛化，必须用新地图测试。
6. **返站安全是单独的约束层。** Fouad et al., *Energy Sufficiency in
   Unknown Environments via Control Barrier Functions* (arXiv:2306.15115),
   Eq. 15/Thm. 1 将剩余能量与返站路径成本关联；Yu et al.,
   *Reachability Constrained Reinforcement Learning* (ICML 2022,
   arXiv:2205.07536) 强调逐状态可达安全而非仅期望总成本。
   两文为“有电安全返站”提供理论邻域，但它们的模型/假设不同于
   当前仿真；现有可执行返站证书仍须独立审计，不能借论文转述为保证。

## 下一步实验的顺序

先做一个只改**固定时窗终止与折扣定义**的学习对照：保留全部地图、
能耗、碰撞修复、动作掩码、安全层及 51,200 步预算，使用正确的
finite-horizon `terminated`。主条件用 `gamma=1` 对齐未折扣固定时窗
完成数；另设 `gamma=0.995` 以区分“终止修复”与“折扣修复”。两者
都是待检验方法，不能根据早期训练曲线挑选验证地图或最优检查点。

若仍失败，再逐项测可执行返站证书辅助训练、位置/返站循环记忆、
规划器示范初始化，以及按物理时长折扣的半马尔可夫 PPO。安全训练
条件可允许仿真试错，但正式评价保持同一安全层，分别报告原始策略
失败率与执行后安全率。任何新方法都应与强解析路线和 GPU-MPC 比较。

参考链接：

- Pardo et al. (ICML 2018): https://arxiv.org/abs/1712.00378
- Theile et al. (arXiv 2023): https://doi.org/10.48550/arXiv.2309.03157
- Bischoff et al. (arXiv 2025): https://doi.org/10.48550/arXiv.2505.11136
- Precup et al. (ECML 1998): https://doi.org/10.1007/BFb0026709
- Fouad et al. (arXiv 2023): https://doi.org/10.48550/arXiv.2306.15115
- Yu et al. (ICML 2022): https://doi.org/10.48550/arXiv.2205.07536
