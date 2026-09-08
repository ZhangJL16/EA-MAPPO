# 文献调研报告：Return-to-Charge 方案能否成为 ICLR 主会 oral 级核心创新

调研日期：2026-09-06。检索方式说明：本会话 web_search 工具不可用（modsearch 引擎全部失败），改用 curl 直查 iclr.cc/nips.cc/icml.cc 官方 virtual 页面（oral 列表）、OpenReview API（venue/评审字段）、arXiv API、Crossref；Semantic Scholar 与 OpenAlex 全程 429 限流，dblp 握手超时——这些来源的引文上下文评价未能获取，已在下文标注。

## A) oral 级论文的核心创新形态与证据标准（真实例子，均已在官方页面/OpenReview venue 字段验证为 Oral）

1. **纯理论+零实验型**：Span-Based Optimal Sample Complexity for Weakly Communicating and General Average Reward MDPs（NeurIPS 2024 Oral，https://nips.cc/virtual/2024/oral/97954，arXiv 2403.11477）；Computationally Efficient RL under Linear Bellman Completeness for Deterministic Dynamics（ICLR 2025 Oral，https://iclr.cc/virtual/2025/oral/31796，arXiv 2406.11810）。二者均以"解决开放问题+匹配上下界"取胜，PDF 中无实验节。→ 证明"理论深度可以单独支撑 oral"，但前提是非平凡的新定理（open-problem 级），不是已知结果的组合。
2. **理论+小规模仿真**：Ordering-based Conditions for Global Convergence of Policy Gradient（NeurIPS 2023 Oral，https://nips.cc/virtual/2023/oral/73818）；Online RL in Linearly q^π-Realizable MDPs（NeurIPS 2023 Oral，https://nips.cc/virtual/2023/oral/73864）；Conformal Meta-learners for ITE（NeurIPS 2023 Oral，arXiv 2308.14895，仅 IHDP/ACIC 半合成数据）；Statistical Efficiency of Distributional TD（NeurIPS 2024 Oral，https://nips.cc/virtual/2024/oral/97962）。
3. **理论/机制+MuJoCo 级实验**：Flat Reward in Policy Parameter Space Implies Robust RL（ICLR 2025 Oral，OpenReview venue 字段验证，https://openreview.net/forum?id=4OaO3GjP7k）；METRA（ICLR 2024 Oral，https://iclr.cc/virtual/2024/oral/19745；MuJoCo+Kitchen+DMC，4–8 seeds）；OMPO: Unified Framework for RL under Policy and Dynamics Shifts（ICML 2024 Oral，https://icml.cc/virtual/2024/oral/35525；MuJoCo Hopper/Walker2d/Ant/Humanoid+Meta-World，含 dynamics shift）。
4. **与本方案同族工具的口碑先例**：Amortized Control of Continuous State Space Feynman-Kac Model（ICLR 2025 Oral，https://openreview.net/forum?id=8zJRon6k5v，用多边际 Doob h-transform）——说明 Doob/Feynman-Kac 语言在 ICLR oral 有位置；Probabilistic Learning to Defer（ICLR 2025 Oral，https://openreview.net/forum?id=zl0HLZOJC9）——选择性预测/拒绝已拿过 oral。

形态结论：oral 的通行组合是"一个干净的非平凡定理（最好带匹配界）+ 可信的有限实验"，或"open-problem 级纯理论"。实验规模常见为 3–5 个 MuJoCo 类环境、4–8 seeds；半合成 2 域也够（NeurIPS 2023 例子）。

## B) (a)–(e) 与最近工作的重叠/差距

- **(a) executed-interface resource-to-go 可辨识性**。最近：Khan et al., OPE Beyond Overlap（ICML 2024，https://proceedings.mlr.press/v235/khan24b.html）；Chen & Jiang（UAI 2022，https://proceedings.mlr.press/v180/chen22g.html）；Duan et al. 受限 χ² 覆盖（ICML 2020，https://proceedings.mlr.press/v119/duan20b.html）；Katdare et al. 环境迁移 OPE（CoRL 2023，https://proceedings.mlr.press/v229/katdare23a.html）。差距：现有可辨识性针对折扣/有限时域回报的密度比；未见"policy∘safety filter 复合接口 + 被杀首达资源泛函"的可辨识性充要条件。这是 (a) 唯一可能的新意，但必须给出"普通 support vs 尾部相关 support"分离的不可辨识性定理，否则被 Khan et al. 覆盖。
- **(b) 长时域误差 occupancy 分解与分布 OPE**。最近：Huang et al., Off-Policy Risk Assessment（AISTATS 2022，https://proceedings.mlr.press/v151/huang22b.html，DR CDF+匹配 minimax——最大碰撞点）；Distributional OPE with Bellman Residual Minimization（AISTATS 2025，arXiv 2402.01900）；A Principled Path to Fitted Distributional Evaluation（NeurIPS 2025 Spotlight，https://openreview.net/forum?id=Gte3F0ONhr）；DualDICE（NeurIPS 2019，arXiv 1906.04733）/GenDICE（arXiv 2002.09072）/CoinDICE（NeurIPS 2020）/KROPE（ICML 2025，https://proceedings.mlr.press/v267/pavse25a.html）。差距：固定/随机但非被杀的水平线 vs 项目的 killed Feynman-Kac 首达对象；普通 occupancy 分解无新意，需证明"风险倾斜 resolvent 分解严格更尖锐"。
- **(c) 返回边界稳定性**。最近：Audibert–Tsybakov 2007（https://arxiv.org/abs/0708.2321）；Belomestny 2011 停时率（https://arxiv.org/abs/0909.3570）；Kim & Zubizarreta 比较引理（ICML 2023，https://proceedings.mlr.press/v202/kim23ab.html）；Fujimoto et al., Bellman error ≠ value error（ICML 2022，https://proceedings.mlr.press/v162/fujimoto22a.html）。差距：margin→决策转换是经典工具，"边界附近误差才改变决策"作为独立定理无新意，只有与 (a)/(b) 的对象耦合并给出匹配下界才可能成立。
- **(d) SIRP/选择性预测/保形**。最近：SelectiveNet（ICML 2019，https://proceedings.mlr.press/v97/geifman19a.html）；Learning to Defer（ICLR 2025 Oral，同上）；Conformal Off-Policy Prediction in Bandits（NeurIPS 2022，https://doi.org/10.52202/068431-2285）；Conformal OPE in MDPs（CDC 2023，arXiv 2304.02574）；Conformal Prediction Beyond the Horizon（NeurIPS 2025 Poster，https://openreview.net/forum?id=RIkHzQbpeR，arXiv 2510.26026）；Tibshirani 加权保形（NeurIPS 2019，arXiv 1904.06019）。差距：组件全部成熟且已有 oral；SIRP 必须证明决策效用（stranding–throughput 增益）或与 executed-interface 的耦合，否则只是组合。
- **(e) Doob 变换/风险 OPE/Perron 微扰**。最近：ACSSM（ICLR 2025 Oral，Doob h-transform）；Meyer 1994 与 Ipsen–Meyer 1994（SIAM J. Matrix Anal. 谱敏感性）；Karmakar–Bhatnagar 2021（Perron-Frobenius 微扰界）；Blanchet–Glynn–Zheng 2016（QSD CLT）；Fei et al.（NeurIPS 2020/2021，风险-样本权衡）。差距：perron/killed 谱微扰与指数效用全部经典（工作区 2026-08-30 审计已列明），唯一存活空间是四者耦合（复合接口+随机停时+谱临界+不可逆决策），目前无文献占位但必须证明确实"更尖锐"。

## C) 直接判断

**现状（导航 gate 未过、实验全 pending）下不可成为 oral；即使 gate 全过，按现有证据预期是 poster 级核心（novelty 3/5、reject 风险主要来自组合贡献观感），oral 仅在 (a)+(b)+(c) 成功耦合为一个"新对象"定理并附匹配下界时才有可能。** 理由：(1) (b)–(e) 的每个组件都有密集的顶会/oral 先例，组件本身无新意；(2) 相比 A 组理论 oral（open-problem 级），本方案的定理目标是"条件化转移/稳定性定理"，弱一档；(3) 唯一的新对象——executed interface 上的被杀资源首达分布与不可逆返航——在文献中确实未被占据（RC-PPO/RAPCPO/Back-to-Base 均不是这个对象，见 E 项），这是真实空间；(4) 但审稿人会要求它"改变算法设计或数据收集规则"（工作区蓝图已列此条件）。满足 (a) 的充要条件定理+一个可预测失败的实验，oral 概率中等偏上；若只交付 (b)/(d) 组合，则明确 poster 级。

## D) 达到 oral 的证据链（缺一不可）

1. 前置 gate：导航通过 + Oracle headroom kill test（现有协议已定）。
2. 至少一条**严格更尖锐**定理：风险倾斜 resolvent/occupancy 分解优于普通 occupancy 界的实例与证明。
3. 不可辨识性分离定理：普通支持重叠但尾部相关支持不重叠的构例。
4. 匹配下界（packing/Le Cam），或明确声明无下界并降级贡献措辞。
5. 实验：mean 预测准但返航尾部失败、且方法事前预测该失败的案例；stranding–throughput Pareto 有置信区间；**第二个域或第二个安全滤波器**（或明确有界声明——ICLR 指引要求"convincingly demonstrate new knowledge"，未强制真实机器人，但安全措辞必须限域）。
6. SIRP 与 ensemble/conformal 基线做风险-覆盖/AURC 对决，输则删除（工作区评审已定）。
7. 诚实 AI-use 声明（工作区 2026-09-01 评审将其列为 desk-reject 风险）。

## E) 不确定性与检索局限

- web_search 不可用，本报告证据来自官方 virtual 页面/OpenReview/arXiv/Crossref 的直接抓取，可信度高但覆盖面窄于原计划 12–20 次搜索。
- Semantic Scholar、OpenAlex 全程 429：**RC-PPO 等论文的引用上下文评价未能获取**；替代证据为其 NeurIPS 2024 OpenReview 评审（rating 5 与 7，肯定理论+实验验证）与 RAPCPO（ICML 2026 regular）的延续。
- 若干实验规模细节（Flat Reward 论文的环境/种子数、RC-PPO 种子数、"Learning MDPs from Features"的 NeurIPS 2021 归属）**未能逐项验证**。
- 与资源/返航最近工作：RC-PPO（NeurIPS 2024 Poster，https://openreview.net/forum?id=jzngdJQ2lY；确定性动力学，Safety Hopper/HalfCheetah+WindField 四旋翼仿真+FixedWing）、RAPCPO（ICML 2026，https://openreview.net/forum?id=ReCCYnyfXa）、Back to Base（L4DC 2025，https://proceedings.mlr.press/v283/begzadic25a.html）、Consumption MDP 综合（CAV 2020，https://link.springer.com/chapter/10.1007/978-3-030-53291-8_22）、QCPO（NeurIPS 2022）、Sauté RL（ICML 2022）、Energy-Sufficiency CBF（arXiv 2306.15115）。这些均处理预算/可达语义，未占据"executed-interface + 被杀首达 + 不可逆返航"这一联合对象——但该空白是"机会信号"，不是"无竞争证明"。
- 2026 年 oral 列表（223 篇）明显 LLM 化，RL 理论 oral 供给收紧，投稿时机对理论型工作不利。
