# 安全强化学习：前向引用与批判性适配审查

日期：2026-09-05。研究范围：沿上一轮候选的后续引用追踪实际使用，而不是用引用数量投票。筛查 15 项论文/工具候选，保留以下 10 项核心证据。没有运行或修改训练代码。

## 结论先行

1. **FOCOPS 保留为候选，不能指定为已知最适合的唯一方案。** 后续导航实验有成功也有失败，且实现版本会明显改变成本—收益位置。
2. **SafeMPO 下调为理论参考。** 原文三项导航实验在预算 25 下的平均成本全部超过 25；其理论启发不能冒充经验可行性。
3. **PPO-Lagrangian 应成为同成本合同的必要简单对照。** 仅比较普通 PPO 与 FOCOPS，会同时改变是否约束和策略更新规则，无法分清收益来自哪里。旧 R7 的动作修正成本不是这个对照。
4. **不排除 SAC 路线。** 有更接近原始 LiDAR 导航的后续论文报告 SAC 优于其 PPO/FOCOPS 配置；但不能照搬其额外模块或真实机器人成功率。
5. 从头训练、完整输入、首次物理接触指标仍保留。暂不加入 ProSh、Langevin 采样、课程或新的能耗网络。

## 如何认定“引用后用得怎么样”

- **B — background**：参考文献或 related work 提及，不代表复现成功。
- **E — evaluated**：明确作为实验基线；必须一起记录任务、预算和结果。
- **I — implementation reuse**：明确采用作者/库实现；不自动证明实现完全等价。
- **M — mechanism criticism/extension**：分析或改进机制；作者的因果解释仍须和实测事实分开。
- 没查到引用边，记为未确认。索引收录、作者主页和学校新闻不算独立复现。

下文所谓“安全”一律按论文实际成本预算解释；多数不是每次导航零碰撞。

## 核心证据与反证

### F01 — Safety-Gymnasium / SafePO，NeurIPS 2023 Datasets and Benchmarks

[论文](https://proceedings.neurips.cc/paper_files/paper/2023/hash/3c557a3d6a48cc99444f85e924c66753-Abstract-Datasets_and_Benchmarks.html)，[PDF](https://proceedings.neurips.cc/paper_files/paper/2023/file/3c557a3d6a48cc99444f85e924c66753-Paper-Datasets_and_Benchmarks.pdf)。关系：FOCOPS/CPO/PPO-Lagrangian/PID/CUP → 本文 **E/I**，正文 §5–6、表 1–2 和参考文献已核对。

**观测事实：** 表 1 专门比较同名算法不同实现。CarGoal1 中，FOCOPS 原实现的归一化 reward/cost 为 0.79/2.45，SafePO 实现为 0.52/0.93；cost 归一化分母为预算 25。后者成本合规但收益也降低，不能只说“修复实现全面提高性能”。表 2 中 SafePO FOCOPS 在 PointGoal1 的归一化成本为 1.32，仍超预算。

**批判：** 表中收益不是到达率；成本不是首次碰撞概率。其广泛评测支持任务依赖和实现依赖，不支持一个算法普适。不要把重实现差异完全归因于某个代码 bug，配置/训练协议也可能参与。

**迁移意义：** 必须记录参考实现、优势归一化、成本缩放、KL 方向和预算；算法名字本身不是充分实验描述。

### F02 — CUP，NeurIPS 2022

[正文与附录](https://papers.nips.cc/paper_files/paper/2022/file/3ba7560b4c3e66d760fbdd472cf4a5a9-Supplemental-Conference.pdf)。关系：FOCOPS/CPO/PCPO → CUP **E/M**，§1、§6 和表 1。

**观测事实：** 表 1 的 FOCOPS：Ant 成本 101.31，预算 103；Hopper 成本 102.30，预算 83。CUP 在 Hopper 的成本为 79.98。相同一篇比较就同时包含 FOCOPS 合规和超标的情况。

**批判：** CUP 自身的改进依据是 GAE surrogate 与一阶投影，而不是物理轨迹安全证书。后续 C-TRPO/SafeMPO 的实验仍报告 CUP 在部分导航任务超预算。原作者实验优胜不能终止审查。

**迁移意义：** CUP 是替代更新规则，不是可以不加区分叠在 FOCOPS 上的“安全模块”。

### F03 — A Multiplicative Value Function for Safe and Efficient Reinforcement Learning，IROS 2023

[全文](https://arxiv.org/pdf/2303.04118)，[作者代码](https://github.com/nikeke19/Safe-Mult-RL)。关系：FOCOPS/SAC/PPO-Lagrangian → 本文 **E/M**，§V-A/B、§VI；IROS 收录标题亦在会议目录核对。

**观测事实：** 比较包含局部占据图导航和原始 1D LiDAR 的 Gazebo Jackal 导航；作者额外调整了 FOCOPS 的 lambda、batch size 和 KL target。FOCOPS 在 Point Robot Navigation 最终超过其 PPO 乘法版本，但不如 SAC；Gazebo 的两个评测点不如 PPO，作者同时承认没有训练到收敛。Car Racing 中 SAC 与 FOCOPS 都无法通过首个弯道。

**批判新方法自身：** 未裁剪 SAC Mult 也出现过度保守、高超时；作者明确承认训练期安全无法保证。真实机器人版本训练 4M 步，另加观测/动作噪声、较小目标区域与 cross-attention；七个目标区域执行四轮，不是大规模部署认证。受追赶时雷达后方盲区仍可能导致碰撞。

**迁移意义：** 这是重要的场景近邻，支持分别学习导航与风险、检查样本效率和观测盲区。但其乘法价值与我们失败的 killed-kernel 分支不是同一目标，不能把任一方成败当成另一方复现结论。也不能因一篇论文有效就立即重启乘法分支。

### F04 — Langevin Policy for Safe Reinforcement Learning，ICML 2024

[论文](https://proceedings.mlr.press/v235/lei24a.html)，[PDF](https://raw.githubusercontent.com/mlresearch/v235/main/assets/lei24a/lei24a.pdf)。关系：FOCOPS/CPO/CUP/PPO-Lagrangian → LAC **E/I**；表 1–3、§6–7、附录 D。

**观测事实：** 2.5M 步，表 2 的 FOCOPS 在 PointGoal 的 reward/cost 为 6.01/38.08，在 CarGoal 为 7.91/20.09；两者预算均 25。PPO-Lagrangian 对应 11.36/16.48 和 12.38/19.50。这是普通拉格朗日方法在部分导航设置中不弱于 FOCOPS 的具体反例，不是全任务排名。

**批判新方法自身：** LAC 使用 Langevin 采样及生成器，论文承认生成器近似和 Gaussian 表达能力局限；其消融中的长采样链会增加计算。FOCOPS 来自作者库，而其他部分基线来自不同库；作者声称默认参数公平，不能等同于等额调参或等墙钟公平。

**迁移意义：** 不因为其表格更好就引入多步采样，先测控制周期和训练预算是否承受得住。

### F05 — C-TRPO，ICML 2025

[正式论文页](https://proceedings.mlr.press/v267/milosevic25a.html)，[所读作者全文版本](https://arxiv.org/html/2411.02957v4)。关系：CPO/PID/FOCOPS/CUP → C-TRPO **E/M**；§3.2、§5、附录 D.3。

**观测事实：** 八项任务、五个种子、每次 10M 步，比较最终成本与训练累计超预算量。作者报告 TRPO-Lag/FOCOPS/CUP 的高回报并不意味着低训练违规。C-TRPO 自身的表现依赖成本优势估计精度。

**批判：** 理想安全集合内部的几何不变性，不是随机初始化的神经网络全过程安全。实现明确允许估计误差导致离开安全集合，并用 recovery+hysteresis 恢复；成本下降阶段可能牺牲探索。它相对 CPO 不增加主要计算负担，因此不能笼统称它“计算昂贵不可用”；相对我们的一阶候选仍需实测 Hessian-vector/line-search 开销。

**前向证据边界：** 本轮未确认足够可靠的独立后续数值复现来为 C-TRPO 本身背书。查到的引用片段和一个未来卷期记录未进入实测证据。

### F06 — SafeMPO，ICLR 2026

[正式论文](https://proceedings.iclr.cc/paper_files/paper/2026/file/1fa0c4e5a7e189729230d018b229abc7-Paper-Conference.pdf)。关系：CVPO/CPO/CRPO/CUP/PID → SafeMPO **M**；FOCOPS/CPO/CUP/PID → SafeMPO **E**。核对 §1–2、§3.3–5 和表 1。

**有用批评：** 当初始策略不安全、探索尚未找到安全高收益区域时，急于恢复/投影可能锁在安全但无收益的区域；CVPO 的预算衰减速度会影响局部可行性。这是机制分析，不是证明每个神经网络实现必然陷入该情形。

**必须降级的实证：** 10M 步、预算 B=25 时，SafeMPO 在 CarGoal/CarButton/PointGoal 的平均成本分别 **32.23/30.87/32.92**。均超过预算。用 B=20 训练的行也不能一概称为合规：成本 25.68/33.37/23.00，至少还需区分训练预算 20 与外部评测线 25。

**迁移意义：** 保留逐步改善可行性的思想；撤回“优先升级到 SafeMPO 就更可能可靠满足约束”的倾向。其方差更小、回报更好和成本合规是不同命题。采用 Retrace、E/M 步骤和额外内循环也需要完整复现，不是替换一个 loss 即可。

§5 中作者承认边界偏移和人工降低预算不是根本解决方案。页 10 注 6 还说明其 CPPOPID 某些轨迹通过走到任务区域外取得低成本、低收益。因此“其他基线合规”也必须检查如何合规；这不是本项目位置越界的证据。

**前向证据边界：** 本轮未找到可核验的独立后续采用/复现论文；检索结果主要是原文及聚合页。不是声称该方法没有任何引用。

### F07 — CRAX，arXiv 2026-06 预印本

[全文 v1](https://arxiv.org/html/2606.20376v1)。关系：FOCOPS/PID/Sauté/PPO-Lagrangian → CRAX **E/I**；§5.1、表 5、附录 C，参考文献 [35–37,47]。

**正面证据：** 作者将 FOCOPS 列为较强基线；表 5 Goal Level 2 的 reward/cost 为 73.3/24.6，预算 25。

**反证与质量检查：** 同表 Goal Level 1/3 的成本为 25.4/26.6，并非所有导航难度都合规。§5 实验设置写 500M 步，附录 C 表 4 写总步数 100M，存在应向代码/作者核实的预算不一致。两者都远大于 500k。运行平台为 H100、2048 并行环境，不能据速度结果推断当前笔记本完成时间。

**任务不等价：** Goal 会在到达后重采样目标，某些 hazard 可穿过并按接近程度记成本，没有单次成功终止。它对算法表现的支持不能直接移植到首次碰撞即失败的无人机。

### F08 — ProSh，arXiv 长文；AAMAS 2026 extended abstract

[所读长文 v2](https://arxiv.org/html/2510.15720v2)，[作者出版状态](https://sites.google.com/site/gaspardohlmann)，[AAMAS DOI](https://doi.org/10.65109/KCVZ6904)。关系：FOCOPS/CPO/PID/Sauté → ProSh **E/M**；§4–5。

**观测事实：** 三种子比较中，FOCOPS/CPO 仍有成本振荡；Sauté 常更保守且部分 Goal 任务回报很低。ProSh 自身在 Goal 中也并非所有运行都严格合规；均值低于阈值不等于每条轨迹安全。

**批判：** 长文安全界依赖 backup critic 的全局误差控制；当前项目没有这种误差证书。部署输出包含分布 shield 和 backup actor，不能拿 shield 后结果证明主网络可独立安全导航。它也引入主/备用两套 actor-critic，不满足第一版最小对照需求。

**元数据警告：** 学校新闻把它列在 ICLR 新闻下；作者主页明确写 AAMAS extended abstract，长文投稿 NeurIPS 2026。本审查不把它写成已核验的 ICLR 2026 完整论文，也不把 extended abstract 当成长文全部定理经过同等审查。

### F09 — SafeOR-Gym，TMLR 2026；所读 arXiv v2

[全文](https://arxiv.org/html/2506.02255v2)，[TMLR PDF 记录](https://openreview.net/pdf?id=3EREfePUvi)。关系：CPO/CRPO/FOCOPS/PID/SAC-Lagrangian → SafeOR-Gym **E/I**；§5 与参考文献。

**观测事实：** 包含能源储存、调度等复杂约束；表 3 中 FOCOPS 满足预算的环境计数为 3，TRPO-Lag 为 5。训练和评测也可能不同，GridStorage 中报告 SACPID/SACLag 成本差异，Blending 中也报告 FOCOPS 差异。

**批判：** 论文后段将 on-policy 训练—评估差异概括为保守性，应与前述 FOCOPS 成本差异一起看，不能据概括句排除估计失准。其作者对 replay/非光滑成本的机制解释未必是隔离因果实验；OR 中混合动作、修复和约束量纲也与无人机不同。

**迁移意义：** 能源/资源问题并不因采用成熟 CMDP 算法就自动解决。可借用按约束分项检查及训练—评测差异诊断，不直接迁移总预算 25。

### F10 — FSRL，作者维护的开源工具，非独立算法论文

[项目](https://github.com/liuzuxin/FSRL)。关系：CPO/FOCOPS/CVPO/PID → FSRL **I**。README 明确强调实现与超参数会影响约束满足，并报告其 CVPO 加速及简单任务分钟级训练。

**批判：** 这是维护者报告，不是本项目已复现的速度或安全证据；不同代码库常使用不同优势归一化、采集与更新配置。它提示做实现交叉核对，不支持盲换依赖或承诺当前场景十分钟训练完成。

## 上轮候选逐项处置

| 候选 | 本轮引用使用证据 | 当前处置 / 可反驳条件 |
|---|---|---|
| CPO | F01/F02/F04/F05/F06/F08/F09 | 保留严肃基线地位；成本估计和恢复阶段需审计，不把二阶优化直接判死刑。 |
| FOCOPS | F01–F09 中多项 E/I；结果混合 | 保留候选，不认定优于标准 PPO-Lagrangian；用同合同对照决定。 |
| PID-Lagrangian | F01/F05/F06/F07/F08/F09 | 抑制振荡有证据，但不能纠正错误成本对象；先不在新主线额外叠 PID。 |
| CRPO | F06 机制讨论、F09 实测 | 不用 dual 不等于无超参数或无停滞；unsafe-start 可能长期只优化成本。 |
| CUP | F01/F04/F05/F06 后续实测 | 不是稳定合规的通用解；作为替代更新，不叠加。 |
| CVPO | F06 直接机制批评、F10 实现 | 优先核对 E-step 可行性与预算衰减，不以样本效率掩盖初始化问题。 |
| Sauté | F07/F08 后续实测相冲突 | 用于资源预算有概念适配，不能因此认定当前碰撞任务有效；必须说明预算更新与违规惩罚。 |
| Safety-Gymnasium | F04/F05/F06/F08 使用 | 作为代码/基准参考，不作为本场景合规证明；先对齐终止和成本语义。 |
| CAL | 本轮尚无核验的独立后续适配实测 | 上轮原文机制证据保留，不增加“已被广泛验证”的结论；集成成本暂不划算。 |
| FCSRL | 本轮未完成可靠前向实测链 | 暂缓；只有观测/表征成为主要瓶颈才深入该分支，不把引用缺口当方法失败。 |
| 安全课程 SCG | F07 正文有一般课程引用，不把它误认成直接验证 SCG | 暂缓；若无约束基线也无法探索到目标，再检查课程。 |
| C-TRPO | F05 是它引用旧方法的研究；自身独立后续复现未确认 | 理论与现有实验值得保留，不能声称后续广泛验证。 |
| ActSafe | F05 related work 提及，属 B 而非复现 | 不将背景引用升级为可部署实证；模型及不确定性假设是迁移前提。 |
| SafeMPO | F06 引用旧方法；自身后续独立实测未确认 | 降为理论参考，原文均值超标是直接限制。 |

## 对当前项目的具体影响

**先建立候选对照，不预定赢家。** 在既定首次接触、完整 LiDAR、同导航奖励合同下，以标准 PPO-Lagrangian 和 FOCOPS 为最小安全候选；普通 PPO 用来检查基础学习能力。三者分开、顺序运行，不是一个模型同时叠三个算法。SAC-Lagrangian 留作 on-policy 学习明显不足时的备选，不先全部扫一遍。

**新增安全机制必须对应可测的失败。** 如果只是 dual 震荡，再考虑 PID；如果是风险估计失真，换 dual 优化器不是充分修复；如果是在安全但无收益区域停滞，再考察初始化/探索；若 blind spot 或刹停距离超出感知信息，先检查可观测性和控制能力。

**算力和样本效率必须一起看。** 论文存在 2.5M、4M、10M 乃至 100M/500M 规模，不支持 50k 足够，也不意味着必须让用户跑这些预算。第一阶段应按当前 measured throughput、有效完整任务数和学习趋势分配可恢复的小预算；没有证据就不自动扩大。

**成功证据分三层。** 成本估计可信、更新确实改善风险—收益、冻结模型在独立任务上无碰撞到达。任何一层失效，都不能用下一层的平均 loss 或 shield-on 表现替代。

## 质量评分与边界

评分在 papers.csv：I=数学/机制洞见，C=方法和证据完整性，N=数值证据强度，1–5；是本轮阅读范围内的暂定质量判断，不是接收概率，也不等同于项目适配。纯基准 N 为 N/A benchmark。F01 覆盖广且有重实现比较，F07 加速与任务覆盖有用但预算不一致，F09 包含现实 OR 约束但与 UAV 迁移距离较大。所有结果均为作者报告，本项目未复现。
