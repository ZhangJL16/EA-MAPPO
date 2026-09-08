# 独立审阅报告：方案现状、安全导航要求合理性、oral 级创新潜力与修正路线

- 日期：2026-09-06
- 审阅人：独立研究代理（与 codex 研究并行，位于 `independent-review-20260906/`）
- 依据：仓库全部核心文档 + 实验产物 + 独立文献检索（arXiv/S2/OpenAlex/PMLR）+ 3 个并行文献调研子代理（结果见 `literature/` 与第 6 节）
- 状态：**全部完成**。三个文献调研子代理报告均已并入（§2.2 / §3.3 / §4.2 / §6）。

---

## 0. 执行摘要（结论先行）

1. **安全导航要求的合理性：数值不算离谱，但定位错了。** 98% 成功率/零碰撞/路径比≤1.10 是"部署级平台验收"标准，不是 ICLR 论文的科学前提。这篇论文的叙事是"HOCBF 保碰撞安全、learned 模块不做安全证书"——它只需要一个"能完成任务的导航平台"（shield-on 成功率 ≥85% 即可支撑下游实验），不需要"裸策略零碰撞"。当前 fail-closed 链把整条能源/返航研究卡在一个与论文贡献无关的工程问题上。
2. **就算安全导航达标，它本身也不是 oral 级核心创新**——仓库自己的 audit（`uav_safety_energy_novelty_audit.md`、`SAFE_NAVIGATION_FROM_SCRATCH_REDESIGN.md` §8）已经正确承认这一点：FOCOPS/PPO-Lagrangian 是成熟方法，有限期事件成本恒等式是标准事实。
3. **当前论文（Resource-to-Go + executed-interface 可辨识性 + 返回边界稳定性）有 oral 潜力的骨架，但当前形态不足以构成 oral 核心创新**：ccfa 预审 3/10 的三大死穴仍在——(a) 正鉴定理接近"重述 exact-agreement 假设"；(b) 33 条定理 sprawl，主干不突出；(c) 零实验 + 单一确定性仿真器支撑不了 stranding 尾事件的任何 claim。满足安全导航 gate 只解决前置，不解决这三条。
4. **用户关心的核心问题（"在合适的能耗选择返航"）在文献中已被大量占据**（Back to Base L4DC 2025、RC-PPO NeurIPS 2024、RAPCPO 2026、Recovery RL、Sauté RL、消费 MDP、energy-sufficiency CBF、以及 2026 年最新的 learned stoppability/option 系列）。剩余可辩护的空间是**窄而具体的**（见第 7 节）。
5. **仓库最新实验给了一个决定性的负结果**（2026-09-05 dual-viability 诊断，3410 rollouts）：单步动作条件的"option 保留"标签几乎完全由状态决定（动作敏感性 p=.776；能量预算 Kendall W=.017），且冻结恢复策略返航成功率只有 88.7%（275/310）——即"动作级 gate"路线被数据否定，"先修复返航策略 + 学状态级 V_R + 后继模型评估动作"是正确收缩。
6. **推荐路线**：把论文收缩为"**返航决策的统计可靠性**"而不是"自适应返航方法"：核心对象 = `b_t ≤ Û_C(x_t) + m ⇒ 返航` 这个不可逆决策的**有限样本/轨迹级校准**，创新点 = executed-interface 下的可辨识性边界 + 返回边界稳定性 + 选择性预测，并把它与 2025-2026 的新竞争工作（Back to Base 系、stoppability 系、RAPCPO）显式区分。同时给一个务实备选：如果 8 周内拿不到 Oracle headroom 实证，降档投稿（CoRL/RA-L 或 TMLR），理论部分单发 COLT/AISTATS 风格短文。
7. **日历硬约束（2026-09-06 核实 iclr.cc）**：ICLR 2027 abstract 截止 **2026-09-18**，全文截止 **2026-09-25**（AoE）。距今 12/19 天。完整的 oral 证据包（导航→电池标定→Oracle headroom→pair/interface→ensemble→Pareto→第二域）**在 19 天内物理上不可完成**；本轮 ICLR 投稿只能是"理论 + 协议 + 极小实证切片"，按 3/10 预审状态大概率被拒。因此本报告默认建议：**跳过 ICLR 2027 正面对撞，执行 3 个月计划，主攻 ICML 2027（约 2027-01 截稿）或 RSS/CoRL 2027（春季）**；若用户坚持本轮 ICLR，则按 §7.4 的最小可行切片执行并接受概率结果。

---

## 1. 仓库方案现状复盘（已核实的事实）

### 1.1 顶层问题与论文
- 顶层问题（用户冻结）：资源受限智能体**何时不可逆地停止任务并返航/回充**，权衡 stranding（返晚了）与 throughput（返早了）。
- 当前论文《When Should an Agent Return? Reliable Resource-to-Go under Closed-Loop Composition Shift》，目标 ICLR 2027（`paper/iclr2027/`）。
- 核心 ML 对象：policy∘safety-filter 的 executed interface 下，charger Resource-to-Go 的**可辨识性**与**选择性分布预测**；错误只在"返回边界"附近改变决策。
- 系统结构：冻结 SAC 导航 + HOCBF 碰撞安全 + TelemetryCostModel 能耗 + Resource-to-Go 估计器 + 可靠性机制 + 单向 ReturnManager。

### 1.2 理论状态
- 本地证明包：33 条结果（`RETURN_TO_CHARGE_DERIVATION_PACKAGE.md`），含 4 个主定理（可辨识/不可辨识/有限时域传输/返回边界稳定）+ EIRR/风险商/停时 Pareto 证书族。
- 自身 protocol 的批评：theorem sprawl；正鉴定理近乎同义反复（ccfa 意见）；第 27 定理 Assouad 构造修复了部分下限，但仍缺"任意 pushed overlap 下的一般下限"。
- 独立证明审计：未完成。

### 1.3 实验状态（全部门 gate 未通过）
- R0–R5 导航修复：500k 步 × 500 任务，全部 FAIL。最近一次（R5）：success 0.938 / 桶 0.860 / 路径比 1.098 / 碰撞步 2497。
- JSEB 500k：success 0.84、路径比 1.63，碰撞/边界通过 → 失败的是导航能力不是碰撞。
- 冻结 R3：shield-on 47/50 安全到达；shield-off 26/24 → 自主避碰不足；SAC+HOCBF 交互 stall（11/11 冻结态诊断：0/11 在 4000 步内到充电器）。
- reach-avoid SAC 两条微调臂：安全到达 7/50、1/50 → 导航塌缩（价值函数越界 38.77%/45.31%）。
- 发现并修复：旧 LiDAR 全局池化丢失绝对方向（四方位探针：90° vs 0.75° 误差）。
- 方向保留 + 普通 PPO 试跑：29/50 到达（20 接触）。
- PPO-Lagrangian 对照（64 cohorts）：Lag 11 安全到达 vs 控制 22；慢乘子：26/13/11；cost-trace 修复未获批（用户冻结原始碰撞语义：非终止、固定 -1.2/-0.42、统一碰撞计数）。
- 2026-09-06 recovery PPO 续训（刚完成）：确定性 50 任务 8 到达/4 无碰撞/42 超时，平均碰撞 818.5 次 → **更差**。
- 结论：目前**没有任何一条"从当前起点训练"的安全导航路线能同时保持到达能力并降碰撞**。这是 4 条独立路线的聚合证据（不是单次失败的巧合）。

### 1.4 返航决策侧实验
- **已通过的环节（此前文档常被忽视）**：R3 500 任务电池标定完成（容量 383.35→精化 304.95）；100 轮连续工作耐力验证 PASS（100 耗尽、0 截尾）；概率语义审计：当前仿真器确定性（Var(Z_C)=0）。
- **机制层面的早期正信号（值得保留）**：500k 冻结策略能量轨迹内，能量–horizon 相关 0.9802（因此 interface 分析必须匹配 horizon）；executed-interface 汇总对 total-energy MAE 相对强混杂模型改善 3.90% [0.62%, 5.14%]（block bootstrap），最高干预分位数残差尾部方差更大——支持"条件尾部风险"机制，但不可部署（未来轨迹汇总），且 R3 内无 held-out 组合。
- Oracle Decision Headroom（Stage B）未跑成：320 循环/点 × 5 点的正式运行在 20 分钟时因 Oracle 克隆撞 4000 步上限崩掉；诊断确认是 SAC-HOCBF 交互 stall（11 个失败态：0/11 到充电器；去 HOCBF 7/11、换启发式 8/11、双换 9/11）。
- 2026-09-05 dual-viability 诊断（310 锚点 × 11 分支 = 3410 rollouts，5,797s）**已完成**，结果决定性：
  - 立即返航成功率 88.71%（CI 84.70–91.77）：冻结恢复策略不可作为高可靠证书；
  - option 标签几乎纯状态决定：21–22/310 结构混合锚点；动作敏感性 p=.776；能量 Friedman p=3.33e-7 但 Kendall W=.017（可忽略）；
  - 历史"完成+返航"标签与"option 保留"标签不可互换（+44.97pp 差异）→ 旧标签作废；
  - 决定：不训练动作条件 Q_op；先修恢复策略；学状态级 V_R（结构头 + 能量头）+ 后继模型。
- 电池标定 / ETG 学习 / 可靠性 / 返回 Pareto：标定与耐力已完成；ETG 学习及之后全部未执行（被导航 gate 阻断或 Oracle 未跑）。

### 1.5 文献自我审计结论（仓库已有，值得信任）
- `uav_safety_energy_novelty_audit.md`：电池回充可行性 CBF、能量梯度投影、聚合 HOCBF 全部有直接先例 → NOVELTY NOT SUPPORTED（加权 2.88/5）。
- `DUAL_VIABILITY_LITERATURE_REVIEW.md`：Recovery RL / Leave No Trace / SAILR / Reach-Avoid RL / RCRL / Back to Base / Consumption MDP / CRC 均为先例。
- `literature-search-20260903-*`：RC-PPO（NeurIPS 2024）、RAPCPO（2026）直接占据"最小成本 reach-avoid + 概率证书"；Sauté RL 占据"电池入状态"；消费 MDP 占据"reload 状态 + 反复可达"。
- 剩余 gap（仓库自己的收缩）：**有限样本/统计证书** + **executed-interface 可辨识性** + **停时资源律（失败质量在无穷）**。

---

## 2. 安全导航要求是否合理？

### 2.1 要求是什么
500 任务固定评估：overall success ≥0.98；每距离桶 ≥0.95；平均路径比 ≤1.10；边界接触步率 <1%；障碍碰撞步 = 0。训练预算 500k 环境步。

### 2.2 判断：数值"从严但不荒唐"，结构"定位错误"

**合理的一面**
- 静态已知障碍、固定场景分布、训练集与测试集同分布时，98% 到达 + 路径比≤1.1 并非不可能（文献中同分布静态导航常报 90–100%）；500 任务、分桶、互斥结果（到达/接触/超时）、路径比带样本数——这套统计纪律本身是好的。
- "碰撞步=0"作为**平台安全验收**（shield-on 评估下）合理——HOCBF 有几何保证，JSEB 500k 确实做到了 0 碰撞。

**不合理的一面（关键）**
1. **它把论文不需要的东西设成了论文的前提。** 论文的 safety story 是"HOCBF 保碰撞，learned 模块不是证书"。它需要的是 (i) 会完成任务的导航器、(ii) 标定的能耗模型、(iii) 随机性。裸策略零碰撞从来不是贡献点，也不该是 gate。
2. **确定性 + 零碰撞 gate 与"返航决策"科学问题正交。** 返航决策的统计内容（分布预测、可靠性、选择性）需要的是**随机性来源**（风、执行噪声、动态障碍、未见组合），不是导航完美。当前 gate 通过与否都不产生该科学内容。
3. **路径比 ≤1.10 与 98% 成功率是"部署质量"语汇**，适合作为论文 limitations/平台描述，不适合作为研究启动条件。历史 R5 已到 0.938/1.098，距 gate 一步之遥，却被判"整条能源线禁止"——这是流程误伤。
4. **对新手算法的预算不公平**：500k 步对从随机初始化的连续控制 + 1024 射线 + 有限期限任务偏小（文献主流 2.5M–10M 步，SafeMPO 用 10M、Langevin 对照用 2.5M）。当然，预算只是次要因素——当前多条路线连"到达能力"都退化了，说明问题更深（见下）。

**独立文献核验（子代理 2，报告见 `navigation-gate-benchmark-report-20260906.md`）——gate 相对领域惯例"跨范畴偏严"**
- 碰撞验收：领域通行是 **cost budget**（Safety-Gymnasium cost_limit=25，cost=每接触步 1；SafeMPO ICLR 2026 在 10M 步下三任务平均 cost 32.23/30.87/32.92 全超预算，作者仍作为正常结果报告）。"零碰撞步"比通行标准严一个量级；shield-off 零碰撞在文献中无先例。
- 成功率：未见/高杂波静态环境的端到端学习导航典型 **60–88%**（ReachNav TD3 76.7% success 且 15.0% 碰撞、SAC-Lag 64.0%/36.0%；Time-optimal 未见 66.7%；ShieldFlight 87.5%→25% @3→9 m/s）；100% 只出现在已见/低速/受控设置。0.98 属极严。
- 路径比 ≤1.10 ≈ SPL≥0.909，严于 Anderson et al. 2018 校准线（未见复杂环境 SPL≈0.5 即好水平）；但这是项目中唯一接近可行的单项（R3 曾 1.188）。
- shield-on 被领域认可为"安全执行"证据，但不等于策略学会避障（Safety Filter vs. Task 讨论 filter 保守性/分布偏移）。
- 边界接触率 <0.01：无文献对标，项目自造指标。
- **总判：作为内部工程 gate 可辩护（静态 24 柱 + shield 属"已见/简单"区间），但比 safe RL 顶会达标线明显偏严；它不应被当作"领域基线"，更不能作为能源/返航研究的科学前提。**
- **补充核验（子代理 2 第二轮，已并入其报告 E 节）**：(i) 预印本出版状态——ShieldFlight=RA-L 2026、SFWT=RA-L 2025、MAVRL=RA-L 2024、FOPC=IROS 2025、TimeOptimal=Unmanned Systems 2026、C-TRPO=ICML 2025；仍为预印本：ReachNav、CurricRacing、CRAX、ProSh 等；(ii) MAVRL Table I 杂波环境 10 种子 success 仅 0.500–0.707；(iii) GUARD 72 组结果统一按 episode return/平均 episode cost/全训练期 cost rate 报告、无 success rate 与碰撞计数——领域达标口径确认是成本指标；(iv) AAAI 2023《Evaluating Model-Free RL toward Safety-Critical Tasks》明指领域"缺乏逐决策步满足约束的高质量评估"并推出 USL/SafeRL-Kit——本项目"逐步碰撞=0"属该少数派 state-wise 口径，主流是预算口径（可在论文 D 节引用）；(v) C-TRPO/ProSh/CRAX/SafeOR-Gym 对 FOCOPS/PPO-Lag 的表值全部直读原文复核通过（C-TRPO："TRPO-Lag./FOCOPS/CUP perform well in reward but poorly in cost regret"；ProSh："PPO-Saute 多数情况下唯一训练期安全，FOCOPS/CPO 安全曲线更不稳定"；SafeOR-Gym：FOCOPS 可行性 3/9、TRPO-Lag 5/9）。

### 2.3 更深的问题：为什么 4 条路线全失败（我的诊断）
- 旧 R3 的价值函数训练合同（碰撞不终止、+100/-1.2）根本不学避碰——这不是"训练不够"。
- reach-avoid SAC 微调把"存活值"替换了导航回报 → 信用分配塌缩 + 完成选择偏差（10k MC 只含完成轨迹）。
- 新 PPO 路线把"首次接触即终止"改为用户后来冻结的"接触不终止"语义，cost=每接触步 1——从随机初始化学"减少接触"的梯度信号被 reward 中 -1.2/-0.42 的固定惩罚淹没；PPO-Lagrangian 的乘子学不到有效信号（MC cost 中心化后常数 advantage 被消掉，replay R² 为负）。**当前的奖励/成本合同与"想学的行为"不一致**，这是比"算法选型"更根本的问题。
- 加上 HOCBF shield-off 评估下的 stall/绕障，说明环境本身（24 个大圆柱 + 100m 感知 + 20m/s）对端到端学习并不友好。

### 2.4 建议的 gate 修正（可执行）
- 把 gate 拆成两个：**平台 gate**（论文需要）与**部署 gate**（工程需要）。
  - 平台 gate（放行下游实验）：shield-on 500 任务 success ≥0.85（或 CI 下限 ≥0.80）、路径比 ≤1.5、分桶无 <0.7 的桶、边界接触 <1%、碰撞步=0（shield-on 下本来就是 0）。再加一项：50 个未见场景（新 seed）。
  - 部署 gate（未来工程）：维持 98%/1.10/零碰撞，但明确它是"部署质量"，不阻塞论文实验。
- 增加**随机性**：为下游返航决策实验加入可复现的风/执行噪声/动态障碍（固定 seed 分布），否则 q90/q95、stranding 尾、可靠性全部没有对象（blueprint §5.2 自己承认确定性下 Var(Z_C)=0）。
- 导航训练优先修**奖励/成本合同**而不是再换算法：在用户冻结的碰撞语义下，把 cost 定义为 1{接触步}（不惩罚已修复位移），把到达奖励与无接触到达绑定，给"减速+绕行"的中间行为塑造 reward（progress 奖励按与最近障碍距离加权），并考虑课程（先无碰撞走廊后全场景）。若两周内 reward-only PPO 仍不达 60% 到达，接受"冻结 R3 + HOCBF"作为平台（它 shield-on 有 47/50），把精力全部移到返航决策线——这才是论文本体。

---

## 3. 就算满足要求，能构成 oral 级核心创新吗？

### 3.1 直接回答
**不能。** 安全导航（无论 FOCOPS/PPO-Lagrangian 实现得多好）是"采用成熟方法 + 应用适配"。仓库自己已冻结该结论（`SAFE_NAVIGATION_FROM_SCRATCH_REDESIGN.md` §8："采用成熟 FOCOPS 不作为原创贡献……论文创新仍需在可验证的能量安全问题上建立"）。它至多是论文的实验章节里"平台已建立"的一句话。

### 3.2 那什么才算 oral 级核心创新？三条可辩护的候选（含我的评价）
**候选 A：executed-interface 可辨识性 + 返回边界稳定性 + 选择性预测（现论文主线）**
- 优势：决策中心的科学问题（不可逆返航）、理论-实验链条完整、与现有方法（PCM、world model、ensemble、OPE）有清晰的 baseline 对阵表。
- 风险：(i) 正鉴定理的同义反复批评（ccfa）；(ii) "interface support vs pair novelty" 的经验定律必须先于方法成立（oracle 分析），否则整个故事塌；(iii) 单仿真器 + 确定性 → 尾事件 claim 无对象。
- oral 条件（blueprint §10 已列，我认为还应加）：**两个动力学域 + 两个滤波器族 + 多策略**；**Oracle headroom 存在**；**SIRP 在 CRPS/AURC/适应效率上打赢 executed-WM+Deep Ensemble**（否则删掉 SIRP，论文变理论+基准，这也可以，但要按这个类别写）。

**候选 B：option-preservation 语义（research-question-card 的 Q_op 线）**
- 优势：最贴近用户直觉（"别做毁掉返航选项的动作"）。
- 风险：**已经被 2026 年新工作正面冲击**——"Humanoid Safe Stop via Learned Stoppability Value"（Long, Abbeel, Sreenath, Shi, Liu 等，arXiv 2609.02358）就是"学停得下来的值"；"Steering with Contingencies"（arXiv 2604.03405）做 r-of-p 备份可达集；Back to Base 做返航 CBF。且**本地数据已否定动作级 Q_op 的增量信息**（动作 p=.776）。除非改成"能量预算条件下的 stop/return 值 + 跨预算单调 + 统计校准"，且明确与上述 2026 工作区分，否则此路不通。

**候选 C：返航决策的有限样本/统计证书（EIRR/停时 Pareto 族，protocol §10.5 收缩后的 3 定理包）**
- 优势：数学上最"硬"，与消费 MDP/RAPCPO 等区分最大。
- 风险：审稿人会说"这是 COLT/AISTATS 的题"；ICLR 要的是"insight + 实证"。protocol 自己判定：Oracle headroom <5% 则纯理论线只适合 COLT/AISTATS。
- oral 条件：证明一个**非平凡的上下界相位转移**（quotient coverage × 指数瞬态 × 停时 margin）+ 至少一个实证域验证相位预测。

### 3.3 我的结论（子代理 3 独立调研后的定稿判断）
- 子代理 3（报告：`oral-feasibility-literature-report.md`，12 个经官方页面/OpenReview venue 字段验证的 oral 例子）结论与本报告一致且更硬：
  - **oral 的通行形态**：(i) open-problem 级纯理论（NeurIPS 2024 oral Span-Based Sample Complexity、ICLR 2025 oral Linear Bellman Completeness——均无实验）；(ii) 理论 + 半合成/小实验（NeurIPS 2023 oral Conformal Meta-learners 只用 IHDP/ACIC）；(iii) 理论 + MuJoCo 级实验（ICLR 2024 oral METRA：MuJoCo+Kitchen+DMC、4–8 seeds；ICML 2024 oral OMPO 含 dynamics shift）。**单一仿真器本身不是拒绝理由，但安全措辞必须限域 + 第二域/滤波器。**
  - **(a) executed-interface 可辨识性是唯一未被占据的位点**：现有可辨识性工作（Khan et al. ICML 2024 "OPE Beyond Overlap"、Duan et al. ICML 2020）不覆盖"policy∘safety filter 复合接口 + 被杀首达资源泛函"的充要条件。
  - **(b)(c)(d)(e) 密集重叠**：(b) 与 Huang et al. AISTATS 2022（DR CDF+matching minimax）、NeurIPS 2025 Spotlight Fitted Distributional Evaluation 重叠；(c) margin→决策是 Audibert–Tsybakov/Belomestny 经典工具；(d) SelectiveNet、Learning to Defer（ICLR 2025 **Oral**）、Conformal Beyond the Horizon（NeurIPS 2025）已成熟——SIRP 若不能证明决策效用或 interface 耦合，只是组合；(e) Perron/killed 谱微扰与指数效用全为经典。
  - **直接判断**：现状（gate 未过、零实验）不可 oral；**gate 全过也预期 poster 级**；oral 仅在 (a)+(b)+(c) 耦合成一个"新对象"定理（严格更尖锐的 resolvent/occupancy 分解 + 尾部相关 support 的不可辨识性分离构例 + 匹配下界）+ 可预测失败的实验（mean 准但返航尾部失败、方法事前预测到）+ 第二域/滤波器时才有可能。
- 结合三个子代理与本审阅：**"满足安全导航要求"既不构成也不解锁 oral 创新；oral 与否由 (a) 的定理锋利度与预测性实验决定。** 最现实的 oral 组合是 A（可辨识性）为主干 + C（更尖锐界与下界）做主定理 + B（option 保持/返航尾部）做实验对象——即"决策中心的统计可靠性"论文。若 A 的 oracle headroom 不成立（5% 线）或分离构例做不出，诚实降档：ICLR poster 级或 CoRL/RA-L 系统版 + 理论短文单发。

---

## 4. 用户的真问题在文献中的位置（我的独立检索，子代理报告并入后更新）

### 4.1 已核实的最接近工作（全部经一手来源验证）
| 工作 | 年份/venue | 与"自适应返航决策"的关系 | 引文评价（截至检索日） |
|---|---|---|---|
| Back to Base（Begzadic et al.） | L4DC 2025 / arXiv 2501.02620 | 值函数型 reach-avoid 安全滤波，minimal 修改名义控制器并回到充电目标；SAC 实验 | S2 引文 5 条，全部为 2025–2026 扩展（Steering with Contingencies、Safe Stochastic Explorer、From Space to Time 等，多为同组或同脉络）；"extend/baseline"性质 |
| RC-PPO（So, Cheng, Fan） | NeurIPS 2024 / arXiv 2410.22600 | 直接用 RL 解最小成本 reach-avoid（HJ 联系、增广动力学），MuJoCo 上成本降最多 57% | 引文 1（OpenAlex）；RAPCPO 将其扩展为随机版 |
| RAPCPO（Pan, Wu, Xue, Xue） | arXiv 2605.11975（仓库标注 ICML 2026，需以正式 proceedings 为准） | 随机动力学 + 概率 reach-avoid 证书 RAPC + 期望成本最小化，"最近的当前结果" | 太新，尚无独立引文评价；只能读原文 |
| Recovery RL（Thananjeyan et al.） | RA-L 2021 / arXiv 2010.15920 | 任务/恢复策略分离 + 安全 critic | S2 引文 40 条；2026 年大量工作将其作为 baseline/benchmark（SafeExplorer、CRRL、AeroDPO、Runtime Safety Filtering、TALB-MAPPO 等），说明其框架已成为该子领域的事实标准底座——"分离式恢复"本身不再构成新颖性 |
| Humanoid Safe Stop via Learned Stoppability Value（Long et al.） | arXiv 2609.02358（2026-09） | **学"停得下来"的概率值 + HJ 可达性值**，任务无关迁移 | 太新（同月），无引文评价；与候选 B 直接撞车 |
| Steering with Contingencies | arXiv 2604.03405（2026-04） | r-of-p 备份可达性（CLF/HJ），模型已知 | 太新，无引文评价 |
| Sauté RL（Sootla et al.） | ICML 2022 | 剩余安全预算入状态 → "电池入状态"非新 | 领域内广泛使用/评价，多为"简单但保守" |
| Consumption MDP（Blahoudek et al.） | CAV 2020 | 电池 + reload 状态 + 反复几乎必然可达的形式化对象 | 形式方法圈内的标准对象 |
| Energy-Sufficiency CBF（Fouad et al.） | arXiv 2306.15115 | 未知环境能量充足性 CBF（依赖外部 planner） | OpenAlex 引文 0（弱引用） |
| RCRL（Yu et al.） | ICML 2022 | 可行集 + 时间上最坏值（资源预算可入状态） | 需进一步核实引文评价（S2 ID 匹配失败，待子代理结果） |
| SAILR（Wagener et al.） | ICML 2021 | 干预值语义（baseline 下的成本动作值） | S2 ID 匹配失败，待补 |

### 4.2 未占位但拥挤的缝隙（子代理 1 检索 + 全文精读后的更新判断）
- 子代理 1 独立检索（31+ 篇核验，报告：`literature-search-20260906-adaptive-return-to-charge/REPORT.md`；后续全文精读 11 篇，证据在 `raw/fulltext/`）的结论：
  - **最接近工作已确认**：Back to Base（learned HJR reach-avoid 返航过滤器）是状态级切换——无电池状态、无动作级 option 保持。
  - **重大修正（全文精读）**：Back to Base 的 5 篇 S2 引用者中 **4/5 是作者同组**（From Space to Time 即 Tonkens/Shinde/Begzadić/Herbert 原班后续；Safe Stochastic Explorer、Learning to Nudge 同为 UCSD 组；Steering with Contingencies 为 Caltech Ames + UCSD 合著）；唯一独立引用（UNH，arXiv 2511.08419）仅背景一句。**因此此前记录的"HJR 扩展性差"等评价是同组自评，不能作为独立第三方评价——该方向目前几乎没有独立实证评价。** 双重含义：(i) 写 related work 时不能引这些当独立批评；(ii) "无人做过独立评价的返航滤波线"本身是空位证据，但 UCSD 组迭代极快，竞争窗口正在关闭。
  - **能量 CBF 谱系的公认局限就是固定 SOC 阈值**：ES-CBF (AuRo 2025) 批其"only obstacle-free"；Eclares (ICRA 2024) 批其"simple robot and battery models"；meSch (IROS 2025) 批其"single-integrator tailored"；Adaptive ergodic search (AuRo 2025) 把该线归类为"relies on fixed SoC thresholds"。注：meSch/Adaptive ergodic/ES-CBF 无 OA 全文，其条目为 S2 引用上下文粒度；Eclares 为全文核验（另含量化：CBF 0.0551 ms/次 vs eware 30.2 ms/次，及"有能量感知滤波 3% SoC 安全返站 vs 无则耗尽坠机"实验）。
  - **energy-aware RL 线**：Learning to Recharge (Theile 2023, arXiv 2309.03157) 是"把充电时机学进策略"的事实基线（14 篇后续引用）；2026 Order Pickers 预印本（AMR 学习充电决策，arXiv 2607.05683）**全文核验**：基线就是 FixedThreshold 与 HighLow（阈值随订单队列变化），PPO 学充电站选择 + 充电时长，完成率 +6%，并明言"routing a robot to charge only when its battery falls below a fixed threshold … myopic"——这是"固定阈值 vs 上下文相关充电决策"最直接的公开实证，但为离散决策、无安全滤波器、预印本。
  - **RAPCPO 全文硬数字（可入基线表）**：同迭代预算下 RC-PPO 到达率 62.29%/73.98% vs RAPCPO 78.49%/88.67%（PointGoal/FixedWing）；随机基准下"RC-PPO similarly exhibits instability"、被指"restricted to deterministic settings"。
  - **最新相邻工作（全文核验）**：Steering with Contingencies（arXiv 2604.03405）动机句即"remain within reach of charging stations as its battery depletes"——r-of-p 备选目标可达的组合 reach-avoid 过滤器，是"返航 option 保持"的最近相邻；Humanoid Safe Stop（arXiv 2609.02358，独立组）学"可停止性值"作为应急停止判据——与"返航可行性门控"同构的可恢复性估计对象。
  - **MAP（mission abort policy）可靠性工程线**（Levitin/Finkelstein/Qiu 等）：最优中止阈值/停时理论，解析可靠性模型，与机器人学习无交叉（正文未精读，仅元数据）。
  - **未占位组合（本检索范围内）**：冻结导航策略 + HOCBF 执行接口 + 电池状态之上，学习**动作条件化**的返航 option 保持门控 + 整轨迹/多充电循环的风险界。Back to Base 是状态值切换（确定性 HJR、无电池）、能量 CBF 是固定 SOC 阈值（需能量模型）、A3 线是端到端重学而非冻结策略门控。
  - **但对这个"未占位"要打折**：(i) 2026 年 learned stoppability/contingency 系列正在从无能量维度逼近同一语义；(ii) 本仓库自己的 3410-rollout 诊断已显示动作级 option 标签增量信息微弱（p=.776）——"动作条件化"这个具体卖点可能被数据否定，需收缩为"状态级 V_R + 后继模型 + 轨迹级校准"。
- "executed-interface（policy∘filter）下的资源律可辨识性"——两个子代理均未见直接竞争者，仍是 A 线最独特的位点。

---

## 5. 对 codex 当前方案的总体评价

**做对的事**
- 实验纪律（fail-closed gate、预注册、互斥指标、原子产物、可恢复性）达到发表级甚至超出多数论文。
- 文献纪律（不轻信原文、追前向引用、区分 B/E/I/M 引用类型）与用户要求一致，且已落地为文件。
- 自我否定机制有效：三周内连续否掉 6+ 条失败路线并给出数据化死因，没有把负结果包装成正结果。
- research-question-card 的收缩（Q_op → V_R + 后继模型）方向正确，且有 3410-rollout 数据支持。

**做错/危险的事**
- gate 的定位错误（第 2 节）——把工程验收当科学前提，导致 8 月中旬以来大量周期消耗在导航修复上（R0-R5、R7、R3-RACT、directional PPO 四代尝试），论文的返航决策实验长期零推进。
- 理论 sprawl（33 结果）与主定理不锋利（ccfa 的同义反复批评未解决）。
- **related work 严重欠引最近的邻居**（2026-09-06 抽查 `sections/02_related_work.tex` + bib）：未引 Back to Base (L4DC 2025)、RC-PPO (NeurIPS 2024)、RAPCPO (2026)、Recovery RL、SAILR、RCRL、Reach-Avoid RL (Hsu RSS 2021)、Sauté RL、Consumption MDP、energy-sufficiency CBF、2026 年 learned stoppability/contingency 系列。其中 Back to Base 与 learned stoppability 是**直接同题工作**。ICLR 审稿人若撞见这些（几乎必然），"作者不知道最近工作"即可致命。这是当前论文最便宜、最必须修的问题。
- 主线摇摆：8 月末至今在 "预测/可辨识性"（A）与 "option 保留/可行性"（B）之间来回切换，二者需要不同的实验平台（A 需要 2 策略×2 滤波器 + 随机性；B 需要可靠的恢复策略 + 能量网格）。当前两个平台都不存在，而实验资源继续投在导航上。
- 用户冻结的碰撞语义与"学避碰"目标之间存在未解决的张力（第 2.3 节），目前以"再试一种训练变体"的方式推进，有无限调试风险；需要止损条件（我建议：2 周内 reward-only PPO 不达 60% shield-off 到达，即冻结"R3+HOCBF 平台"决策，不再回头）。

---

## 6. 文献调研子代理报告（全部完成）

- [x] 子代理 2：安全 RL 基准与达标线 — `navigation-gate-benchmark-report-20260906.md`（要点并入 §2.2；**含第二轮补齐**：FOCOPS 引文图 OpenAlex 兜底 52 条、2026 预印本出版状态逐条核验、MAVRL/GUARD 数字提取、C-TRPO/ProSh/CRAX 表值直读复核）
- [x] 子代理 1：能耗自适应返航决策 — `literature-search-20260906-adaptive-return-to-charge/REPORT.md`（要点并入 §4.2；**含 11 篇全文精读的后续升级**：Back to Base 引文 4/5 为同组自评、RAPCPO/Order Pickers/Eclares 硬数字、contingency/stoppability 全文证据）
- [x] 子代理 3：oral 级创新点构成 — `oral-feasibility-literature-report.md`（要点并入 §3.3）

三个子代理独立收敛：**gate 偏严但可辩护、位置错误；"自适应返航"空间拥挤但 (a) executed-interface 可辨识性是唯一未占位点；现状不可 oral，gate 全过预期 poster，oral 需耦合新对象定理 + 分离构例 + 匹配下界 + 预测性实验。**（全文精读补充：Back to Base 线几乎无独立评价——空位更大，但 UCSD 同组迭代极快，窗口期更短。）

---

## 7. 修正后的研究路线（推荐）

### 目标标定（2026-09-06 用户修订）：Spotlight 级 = 4 个干净定理 + 多域多基线实验
- 用户目标从 oral 调整为 **Spotlight（ICML 2027 主会 spotlight 级）**：有数学定理支撑 + 较多实验。
- 参照模板（同族、同形态）：*A Principled Path to Fitted Distributional Evaluation*（NeurIPS 2025 Spotlight，理论 + 分布 OPE 实验）、SDAC（NeurIPS 2023）、*Statistical Efficiency of Distributional TD*（NeurIPS 2024 Oral，理论+实验）。
- Spotlight 分水岭 = **不可辨识性分离构例（普通 support 重叠但尾部/目标 support 不重叠）**；"严格更锐界 + 匹配下界"从 oral 强制项降级为加分项（可留到 rebuttal）。
- 实验侧从"够用"升级为"较多"：第二域为**强制项**，另加第三轻量域验证定律。

### Spotlight 级证据包清单
- **理论（4 主定理，其余进附录）**：T1 充要条件可辨识性；T2 分离构例不可辨识性（分水岭）；T3 occupancy 加权有限时域传输界（+SSP 截断）；T4 返回边界决策稳定性。每定理必须写明"它否定了哪个替代做法"。
- **实验**：主域 UAV 2 策略×2 滤波器 leave-one-pair-out；第二域 2D 地面车燃料域；第三域 2D grid 资源世界（定律最小组件）；第二滤波器族（HOCBF vs MPC/规则 shield）；预测基线 5 件（直接 MC/TD、PCM、executed-WM、WM+ensemble、SIRP 按需）+ 决策基线 8 件（Oracle/SOC/距离/切换分类器/planner/CMDP 等）；headline ≥3 seeds + CI；AURC/风险-覆盖；配对 stranding–throughput Pareto；机制实验（pair vs interface 因子分解，oracle 先行）+ "事前预测到尾部失败"案例；消融（occupancy 加权、截断、abstention 阈值、适应预算）。

### 路线 R（推荐主路线）："返航决策的统计可靠性"论文
1. **平台决策（1–2 周内做出，不再拖）**：以 `R3 冻结 + HOCBF` 为导航平台（shield-on 47/50 已够用），同步并行 (i) 修复恢复策略（当前 88.7% 返航成功率是下游一切的前提：目标 ≥97%）；(ii) 按第 2.4 节修正平台 gate 并放行下游实验。
2. **随机性注入**：给仿真器加可复现的风/执行噪声/动态障碍分布（seed 固定），重跑 Var(Z_C) audit，升级到 stochastic Resource-to-Go 语义。没有这步，q90/q95、stranding、选择性预测全是空谈。
3. **Oracle headroom（决定性一役）**：5 点 × 320 循环（修好 4000 步 stall 与原子写），阈值：Oracle 对 SOC/距离规则的 throughput 增益 ≥5% 且 stranding 不劣化。FAIL 则整条 learned-manager 线停（protocol 已约定）。
4. **机制先行**：leave-one-pair-out 2×2 组合（2 策略 × 2 滤波器）factorial，检验"pair novelty < interface extrapolation"定律——在学任何方法前用 oracle 误差做。这是 A 线生死点。
5. **方法按需启用**：先 executed-WM + Deep Ensemble；只有当它 AURC 打不过才做 SIRP。**禁止**预先实现 SIRP 当 novelty 占位。
6. **理论收缩与加锋（spotlight 标定）**：4 主定理包，其余 30 条进附录；正鉴定理补可检查的有限样本代理（经验可分性测度），回应同义反复批评。**spotlight 分水岭 = 不可辨识性分离构例（T2）**——必须做出来；加分项（不强求）：风险倾斜 resolvent/occupancy 分解严格更锐的实例、匹配下界（可留 rebuttal，做不出则明确声明无下界并降级措辞）。
7. **领域证据**：第二动力学域（如 2D 地面车 + 燃料/电量，重用 MuJoCo/HighwayEnv 类环境）+ 第二滤波器族。单 UAV 仿真器本身不是拒稿理由（METRA oral 即 MuJoCo-only），但"资源/返航安全"措辞必须限域，第二域/滤波器是审稿人的硬预期。
8. **下游决策表**：统一 ReturnManager 下对比 Oracle / SOC / 距离-能量 / 直接切换分类器 / WM+ensemble / SIRP / receding-horizon planner / CMDP（CPO 或 RAPCPO 适配）。报告 stranding–throughput Pareto + 边界带错误指标。**关键实验（子代理 3 指出）**：构造"mean 预测准但返航尾部失败"的案例，并展示方法**事前**预测到该失败——这是 (a)-(c) 耦合的可证伪实验。

### 路线 S（务实降档路线，与 R 并行准备）
- 系统/机器人版：`采样数据 HOCBF 安全滤波 + 能量感知返航的实测基准研究`（CoRL / RA-L / ICRA）：把已做的 1,024 行 timing、3.56% 轨迹能耗、stall 因果诊断（SAC-HOCBF 交互）做成严谨系统论文——这是仓库目前**唯一有数据**的贡献。
- 理论版：3 定理包 + 27 号 Assouad 构造 → COLT/AISTATS/ALT 短文（若 R 的实证线失败）。

### 决策树（何时放弃/切换）
- Oracle headroom <5% → 停 learned-manager；转 S 或纯理论。
- pair-vs-interface 定律不成立 → A 线死，转 B 线（能量预算单调 V_R + 轨迹级校准）或 S。
- 恢复策略修不好（<95%）→ B 线的"返航选项"对象无意义，只能做 S。
- 全部失败 → 诚实接受：当前平台最合适产出是 S（系统研究）。Spotlight 的达成条件是：Oracle headroom 通过 + pair-vs-interface 定律成立 + T2 分离构例做出来 + 第二/第三域复现；四者齐备时 spotlight 概率主观估计 40–60%，任一缺失则落到 poster 或 S 线。

### 7.4 若坚持 ICLR 2027（9-25 截稿）的最小可行切片
只做能放进 19 天的一件事：**pair-vs-interface 的 oracle 机制检验**（2 策略 × 2 滤波器的 leave-one-pair-out，纯 oracle 误差，不学任何方法）+ 定理 3 个主块的独立证明审计 + 诚实写明"方法结果 pending"。这比空表 3/10 略强，但仍是高风险投递。配套写一段 limitation："本稿是理论+协议；完整实证见后续版本"（审稿人一般不给 credit）。

### 7.5 3 个月日历（默认推荐：ICML 2027 / RSS 2027 主攻）
- 2026-09（本月）：平台决策 + 恢复策略修复 + 随机性注入 + Oracle Stage-B 重启（§8 清单）。
- 2026-10：pair-vs-interface 机制检验（oracle）；若通过 → 启动 5 个最小预测 baseline（IQN/直接 MC、PCM、executed-WM、WM+ensemble、SIRP 原型）。
- 2026-11：可靠性 gate（AURC/risk-coverage）；下游决策表（统一 ReturnManager 下 8 个决策器）；第二域 + 第二滤波器族复制。
- 2026-12：3-5 seeds 完整重跑 + 可复现包 + 写作；2027-01 投 ICML 2027。同期把 HOCBF+energy 系统数据整理成 RA-L/ICRA 短文（保底产出）。

---

## 8. 立即行动清单（接下来 1 周）

1. [ ] 停掉"再来一种导航训练变体"的循环；执行 §2.4 平台 gate 修正并写协议文档。
2. [ ] 恢复策略修复实验（当前 R3 返航 88.7% → 目标 ≥97%）：诊断 35 个失败锚点（17 超时/16 碰撞/2 边界）的失败模式；用"返航专用重训练或几何规则 + HOCBF"修复；配 310 锚点复测。
3. [ ] 随机性注入 + Var(Z_C) audit 重跑（1–2 天）。
4. [ ] Oracle Stage-B 重启（修 stall 诊断 + 原子持久化），拿到第一个真实的 headroom 数字。
5. [ ] 投稿日历决策（已核实：ICLR 2027 abstract 09-18 / 全文 09-25）：默认跳过本轮 ICLR，执行 §7.5 三个月计划主攻 ICML 2027；如坚持 ICLR，按 §7.4 切片执行。
6. [ ] 与用户确认三选一：R（统计可靠性主线）/ S（系统降档）/ 理论单发；避免再摇摆。

---

## 附录 A：本目录文件索引
- `notes.md`：仓库阅读笔记与证据引用
- `literature/`：独立检索原始数据（S2 引文 JSON、arXiv/OpenAlex 查询结果）
- `REVIEW_AND_ROADMAP.md`：本报告
