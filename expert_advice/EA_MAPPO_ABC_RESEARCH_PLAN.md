# EA-MAPPO：交付 A / B / C 的仓库审计与执行计划

**编制日期：2026-09-19。**
**已审计远端快照：** `ZhangJL16/EA-MAPPO`，默认分支 `master`，commit `fff2b026ca3e4ea6b68cacc3964971b0749bfeb3`，提交时间 `2026-09-16T15:52:35Z`。
**本次性质：** 只读、定向源码审计 + 文献核对 + 新计划；没有修改 GitHub，没有克隆或运行该仓库，没有执行测试、重新计算原证书、访问本地服务器或读取 CONFIRM 数据。

## 0. 决策摘要

保留现有 separation 稿件和证据，不继续用更多 framing 代替研究。建立一个**独立版本**的有限预算反馈协议学习工作线：

- **A：问题与基准。** 从两个测量通道、容量 3/4 的固定 fixture，升级到配置驱动的可组合实验图；明确公开信息、反馈释放时间、可行性、有限时域比较者及训练/测试划分。
- **B：方法。** 先建立独立强基线与简单 channel-coverage 修正版，然后实现“公共 belief + 预算条件 + 图结构 + 自回归合法协议生成”的候选方法。小问题用精确 Bayes teacher，规模化问题不枚举全部路径。该组合是待检验的研究假设，不预先称为新范式。
- **C：证据。** 小实例数学一致性、规模化问题的泛化/计算比较、一个已有实验设计任务的外部适配。主要评价有限预算任务质量和计算成本，不用单条 `P(T)/log T` 曲线替代预期风险。

**最先需要完成的是 A0/A1 与独立基线接口，而不是新的定理或大规模训练。**

本计划中的实例数、网络尺寸、周数均为工程排期建议，不是 ICML 官方门槛，也不是已授权的实验配置。当前只批准了制定计划；训练、DEV/CONFIRM 访问、写入 GitHub 均不由本文件自动授权。

## 1. 实际审计发现

下表来自上述 commit 的 GitHub 文件内容；“没有找到”限于读取范围，不能当作整个本地工作区的不存在证明。

| 实际对象 | 发现 | 后续处理 |
|---|---|---|
| `README.md` | 仍以 Paired-Advantage Identification/PSPS 为当前工作介绍，含旧运行状态 | 只在新研究入口 PR 中修正导航；不要改写旧实验记录 |
| `task_plan.md`、`notes.md` | 最新顶层工作已是 separation、诊断桥、AI relevance；负校准保留 | 作为当前远端研究状态的优先依据 |
| `AGENTS.md` | 碰撞机制、合法观测与 startup/resume 边界明确；反对无穷预检链 | 新线只做必要 focused tests、tiny smoke、首 checkpoint；不恢复旧 UAV 任务 |
| `experiments/bundling_calibration_core.py` | `HYPOTHESES`、`POSES`、`CELLS` 固定；`catalogue` 仅接受 B∈{3,4}；两通道 A/B；所有动作时间/消耗为 1 | 是回归 fixture，不是通用 benchmark；保留原文件，新模块独立实现 |
| 同文件 `Learner` | 初始化及 forced sampling 覆盖所有路径；`select(time)` 无最终部署 horizon 输入 | 可以解释旧有限预算现象；不能当成 budget-conditioned controller |
| 同文件 `cost_aware_reference` | 与 resource-path 使用相同 allocation 与 certificate/coverage 调度，只换 scalar scan 实现 | 改称实现一致性 reference；新实验必须有算法独立的强对照 |
| `tests/test_bundling_calibration.py` | 显式断言两个 reference 动作完全一致；还有安全、封存、续跑 identity 测试 | 保留这些回归测试；不能以它们代替算法比较 |
| `scripts/run_bundling_calibration.py` | 固定 seed、truth、128/4096 checkpoint；sealed hashes 只覆盖 core 与 runner；只输出固定 fixture | 不直接改造成 sweep；新 runner 必须能配置任务、训练 seed、噪声、预算并封存完整依赖闭包 |
| `scripts/analyze_bundling_calibration.py` | 明确单 seed、无 CI；硬编码通道均值；检查 T=4096 | 保留原结果，新写多实例/多 seed 分层分析器 |
| `scripts/verify_bundling_finite_budget_theory.py` | 精确 rational Bellman、primitive low Bayes、policy mixture 证书；固定 3 hypotheses | 可作为 teacher 验证锚点；需参数化，不能直接称为通用最优 teacher |
| `scripts/audit_resource_separation_bridge.py` | 独立编码的归一化 belief/risk vector 复算，未导入原 verifier | 适合保留双实现对照；不等同于外部证明审查 |
| `requirements.uv.txt` | 混有 PySC2/SMAC、UAV、优化和 GPU 依赖；Torch 2.7.1+cu128 | 为新线新建最小环境；不动 `.venv`，不混装 DAD 的旧依赖 |
| 根目录 tree、指定 artifact fetch | 该快照无根 `artifacts/` 条目；`strict_benefit_certificate.json` 指定路径返回 404 | 本地存在性未知；请用户提供已访问证据的只读清单，或在新目录生成带新 provenance 的复算结果 |

### 1.1 现有资产的用途

**保留为证据：** 原定理包、prior-art ledger、负校准、封存 manifest、源代码和原 trace。

**迁移为回归测试：** 容量/反馈开关语义、扣费后补给、pending sortie、配对流、精确 T=12 证书。

**迁移为候选 teacher/参照：** 参数化后的精确 Bayes Bellman；最优路径/已知模型有限预算计算。

**不直接复用为新方法：** MAPPO/SAC/UAV low-level policy、全路径 forced coverage、与自身相同的 reference、硬编码分析器。

### 1.2 不能由仓库阅读确认的内容

- 本地未提交/未推送版本、实际 GPU/内存/进程、`.venv` 可运行性。
- 已忽略 artifact 的完整性和是否与原 hash 对应。
- 手写证明是否完全正确；本次未做盲审或重跑 verifier。
- 新算法优势、新颖性、Spotlight 概率。完成代码和计划不构成这些结论。

## 2. 文献依据与方法边界

本轮重新读取了 ICML 2026 reviewer instructions、ICML 2025 官方 Spotlight 列表、PDNAR 与 probabilistic factorial design 的 PMLR 页面，以及 DAD/Step-DAD 的论文页和作者 README。

- **PDNAR（ICML 2025 Spotlight）**：借助经典算法结构和小实例最优解，检验规模/OOD 泛化。对本计划的启发是“小 teacher → 可复用计算”，不是把 GNN 本身当创新。
- **Probabilistic Factorial Experimental Design（ICML 2025 Spotlight）**：从可操作的实验接口提出设计问题，给出结构性理论及模拟证据。启发是 A 必须定义清楚可控接口。
- **DAD（ICML 2021）**：已有离线学习 design policy、线上低延迟决策。不能以“历史到实验的网络”作为首创。
- **Step-DAD（ICML 2025）**：已有离线训练加测试时 infer/refine。线上计算和 posterior 更新成本必须纳入比较。本计划没有把 DAD/Step-DAD 标为已核实 Spotlight。
- **OSSB（NeurIPS 2017）**：已有结构化 allocation 理论。相同 scheduling wrapper 的两个实现不是独立算法。
- **Goal-oriented OED（SIAM/ASA JUQ 2026）**：预测目标导向实验设计已有直接工作。不能以“不最大化所有参数熵”为唯一新颖性。

这些文献提供设计依据，不提供“做完以下数量的实验即可 Spotlight”的规则。

## 3. 研究问题与论文边界

### 3.1 唯一主问题

> 在已知操作/资源约束、未知观测模型标签、明确有限部署预算下，能否学习一个不依赖完整路径枚举的反馈协议生成器，并在未见过的实验图与预算下，以相同信息和计算预算获得更好的任务决策质量？

需要同时检验两个子主张：

1. 相对于完整目录方法，组合协议生成能改善可扩展性。
2. 相对于强近似规划和学习式实验设计，结构与有限预算续策训练能带来决策质量/计算 trade-off 优势。

**不是本阶段主张：** 无限生存、未知动力学安全、完整 belief 最小化、基础 Blackwell 理论、POMDP 新范式、真实 UAV 部署、通用 minimax 最优、增加容量对所有预算必有实际收益。

### 3.2 两条工作线不混淆

- **Frozen theory line**：保留既有存在性/稳健性证书与 original calibration。
- **New method line**：单独 run/version/manifest；研究泛化、可计算性与有限任务质量。旧主定理作为动机和精确 fixture，不能自动覆盖新任务。

建议工作目录 `research/feedback_protocol/`，建议分支 `research/feedback-protocol-v1`。两者均未创建。

## 4. 交付 A：可复用的问题与基准

### A0：恢复与来源登记（0.5–1.5 人日）

**操作：** 在本地只读运行随包附带的 inventory 脚本；提交本地 HEAD、dirty 文件名、核心源码 hash、已访问 bundling artifacts 是否存在、可用硬件信息。

**不要：** `git reset --hard`、覆盖旧 `.venv`、读取旧 CONFIRM、把未推送实验冒充远端快照。

**输出：** `repository_baseline.json`、`evidence_inventory.json`、明确 remote/local commit 差异。

**验收：** 明确能否恢复原 receipt；若没有，记录缺失，不重新制造“原始”文件。新复算写新目录，并带自己的生成时间/hash。

### A1：冻结问题 contract（2–3 人日）

`ProblemSpec` 必须包括：

| 字段 | 含义 |
|---|---|
| `graph` | 已知状态/操作节点与有向边 |
| `duration`, `energy` | 每个 primitive 的已知非负资源消耗、正时间消耗 |
| `reset_rule` | debit-before-reload，补给/清理时间是否计入 |
| `hypothesis_model` | 公共模型族及先验；隐藏 label 单独归 evaluator |
| `measurement_channels` | 哪个 primitive 产生哪些反馈；共享参数映射 |
| `protocol_rule` | 每次 batch 可测次数、许可次序、反馈何时可供决策 |
| `capacity`, `calendar_budget` | 初始资源与有限总预算；不同概念分开 |
| `terminal_rule` | 原 fixture 保持原语义；新主实验建议合法结束且回到 reset |
| `objective` | cumulative task reward 或明确的 terminal loss；不能混用 |
| `split_id`, `instance_hash` | 划分与谱系；clone/reskin 不算新实例 |

**第一版控制语义：** 只在 reset 点选择完整 committed batch；batch 内可以记录测量，但不能根据尚未完成 batch 的奖励改协议。autoregressive 解码是在出发前进行的计算，不是获得新物理反馈。实验完成后更新 belief 并重新规划。

**公共信息 API：** `PublicProblem`、`ObservationHistory`、`PlannerState` 与 `PrivateTruth` 四者分离。禁止把 evaluator、true label、oracle value、teacher private rollout 或未来 RNG 流交给 learner。

**验收：** 仅通过配置即可复现原四 cells；四 cells 的原结果仍保持为原版本。新协议时间、反馈、终态都能从 event log 重建。

### A2：配置驱动 simulator 和生成器（3–5 人日）

先实现有限模型类的图式巡检/批实验，不引入新物理引擎。模型可在节点/通道间共享参数；实例生成时应保证模型族公开而 truth 隐藏。

建议开发参数，仅用于资源 profiling 后冻结：

- 精确层：2–4 个测量通道、3–8 个 hypotheses、T=12/18/24；超过 solver 限额标注未解。
- 训练层：3–6 个测量通道，多个 graph/template 和 public priors，T=24/48/96。
- OOD 层：8/12 个通道，未见 topology、费用比例或 T=192/384；不要将全部 OOD 难度一次叠加。
- 容量不是永远取 3/4：按合法路径所需资源生成多个水平；始终检查旧动作嵌入与 return feasibility。

以上大小不是求解可行性的保证。精确求解要设置结点/时间/内存限额，不准 timeout 后把近似解标为 optimum。

两类实例必须分开：

1. **机制子集：** 通过公开模型解析检查或精确计算，选择跨容量 oracle execution 不变的成对实例。选择规则在采样前固定，报告接受/拒绝比例。
2. **总体子集：** 不以“容量有收益”或“oracle 值相同”筛选，评价任务族整体性能。

机制结论不外推到总体；总体比较使用各实例自己的 comparator。

### A3：oracle 与评价对象（2–4 人日）

新线主指标建议：

`R(instance, theta, T) = V_known(instance, theta, T, terminal_rule) - E[realized cumulative reward]`。

同时保留 cumulative task reward。原理论中的 `T*rho` 是独立的旧指标；只有证明两个 comparator 一致的条件下才等同。

- **已知真值 oracle：** 评价 reference，不能生成未知模型 learner 的动作标签。
- **Bayes teacher：** 给定公开 prior/posterior 计算反馈策略，合法用于训练；不是逐 truth 最佳动作。
- **Minimax witness/teacher：** 仅在实际求解过的有限 policy game 中使用，记录上下界和 gap；不将 Bayes 解改名为 minimax。
- **大实例：** 若 `V_known` 只有区间 [L,U]，报告 policy regret 区间 [L−J,U−J]，不能把一个较弱 heuristic 当 exact oracle。

**验收：** 小实例 primitive enumeration 与 macro solver 一致；Bayes/minimax/oracle 三种对象在配置与输出中不可互换。

### A4：日志、封存与拆分（2–3 人日）

新 event schema：instance id、model-training seed、noise stream id、decision epoch、time、remaining budget、resource、合法动作、committed protocol、released feedback、public posterior、decision reason、在线计算时间和模型调用数。

- 保存 pending protocol、模型/优化器/环境 RNG。
- 数据生成、teacher、训练、评估使用独立 RNG namespace。
- 同一 graph 的全部 capacity/horizon 变体属于同一 train/dev/test 分区，防止图泄漏。
- 主模型测试给新图和新 noise；另开结构 OOD，不伪装 IID。
- source seal 覆盖 simulator、adapter、mask、inference、learner、config、lockfile、teacher manifest，不只两个 Python 文件。
- 不导入旧 `review_bundle`、UAV checkpoint 或旧 DEV/CONFIRM registry。

## 5. 交付 B：有限预算协议学习方法

### B0：独立基线先落地（3–5 人日，可与 A2 并行）

| Baseline | 角色 | 实现和公平性 |
|---|---|---|
| frozen original | 历史诊断，不作为唯一主要对手 | 不修改，保留旧结果 |
| public-channel coverage + cost allocation | 最小强修正 | 基于全公共 hypothesis class 选择覆盖通道，不能使用 true-model lambda |
| independent cost-aware structured allocation | 邻近理论对照 | 独立依据算法描述实现；调参资源对齐；不得调用同一 wrapper 然后宣称独立算法 |
| posterior sampling / cost-aware greedy | 简单而强的 reference | 相同 posterior、合法操作与反馈 |
| budget-limited lookahead / tree search | 有限预算主要对照 | 控制线上时间、分支/模型调用数量；出界必须标注 |
| exact Bayes/minimax | 小实例计算参照 | 不要求在大规模场景强行运行；记录适用大小 |
| generic recurrent/attention policy | 同目标同容量模型对照 | 与 proposed 同训练目标/数据/模型量，区分结构价值 |

不要先全部做大型神经 baseline。第一项有信息价值的新实验应包含 simple channel coverage 与 finite-budget planning。

### B1：teacher 数据（3–5 人日）

参数化已有 `exact_response`/Bayes recursion，但保留原两套固定证书实现用于交叉检查。

一条 teacher 记录包括：

- 公开 graph、prior/posterior、资源与预算；
- 候选 complete protocol 的 Bayes Q 或可认证区间；
- ties/近似最优集合；合法 mask；求解结点数、时间、精度、provenance。

**禁止：** 利用真实 theta 单独求 optimal action 再当作未知环境策略标签。

训练状态不能只来自最优 teacher 的单一路径。混合 teacher、简单策略和当前 student 的合法历史；全部通过公共模型计算 posterior。若做数据聚合，记录新增 teacher 查询/计算成本。

**验收：** theta label 从 actor 输入剔除；固定相同公共 history 改 evaluator truth 不应改变 policy 输出。测试 task 的 teacher 数据不进入训练。

### B2：候选方法（5–8 人日）

第一版不用 opaque history encoder 重新讨论最小充分表示。有限假设类先用**精确公共 posterior**，将不确定性学习与协议学习分开。

结构建议：

1. graph/channel/model-set encoder 处理节点特征、共享观测结构、公共模型后验；
2. budget-conditioned autoregressive decoder 在出发前产生下一个 committed batch；
3. prefix mask 确保完成该前缀后仍能按已知规则回到 reset，并满足终态预算；
4. learnable continuation critic 估计该 batch 反馈后剩余预算的任务价值；
5. 可选固定 beam width 的搜索，计入线上算力；同时报告纯 forward 与带搜索版本。

输入不含 truth、未来 reward、已封存 C 或已知模型 oracle。路线中 feedback 的 joint law 来自公共模型，不额外调用真实实验对象。

**核心收益假设：** 结构化 decoder 不必遍历完整 catalogue，并能在未见图/容量下利用共享测量与剩余预算。这个假设未获证据，不称“新范式已建立”。

训练优先次序：

- 第一步：小实例 Bayes imitation / regret-weighted Q distillation；ties 不强迫任意动作成为唯一真值。
- 第二步：只有 imitation 的分布偏移确实存在时，加入模拟训练中的 on-policy 数据聚合或共同任务效用 fine-tuning。
- minimax 训练不是默认前提：先得到 Bayes/平均任务性能，额外报告有限类 worst-case，不将前者叫 minimax guarantee。

### B3：决定原创性的消融

| 对照 | 能排除什么 |
|---|---|
| exact posterior + generic autoregressive policy | 不是额外 inference accuracy 或 decoder 形式本身 |
| 相同结构，去掉剩余预算输入 | 有限预算条件是否真的重要 |
| 相同结构，EIG 目标 vs task utility | 目标贡献与结构贡献分开 |
| 相同目标，完整 path scoring vs compositional decoding | 可扩展性是不是来自不枚举 |
| 相同搜索预算，无 learned critic | 改善是否只是额外搜索 |
| public-channel-coverage correction | 是否简单修正原 wrapper 就足够 |

不使用去掉安全 mask 后频繁失效作为主要胜利。安全规则已知，应给各方法同等的合法动作接口。

### B4：理论工作边界

当前不要求新的 universal minimax theorem。最小应有：算法定义完整、已知图上的 feasibility 证明、复杂度账目、teacher/approximate planner 误差范围。

若推导性能误差，应在明确的状态分布、价值逼近误差与 planner 条件下完成，不从监督 MSE 小直接推出 regret 小。未证明时报告经验值，不塞一个未核验 `O(epsilon T)` 公式。

**方法线判断：** 若 simple coverage 或同算力近似规划已经达到 proposed 的效果，应缩小/停止算法创新声明，而不是新增术语或弱化 baseline；理论稿继续保留。

## 6. 交付 C：可比较、可复现的证据

### C1：精确层验证

- 原 T=12 证书与开放盒仍是原结果；不改数值、不重新选择 prior 为新算法加分。
- 新接口仅做 regression：固定 public prior 的 Bayes expected value、known-model value、旧 control 语义一致。
- 新方法在小有限类可做 exact policy evaluation；无法精确时进行多噪声轨迹估计，并标记 stochastic estimate。
- 每个 hypothesis 单列平均风险，不能对所有轨迹取 max 冒充 `max_theta E[R]`。

### C2：规模与泛化

预注册学习曲线和计算曲线，至少分别回答：

1. 固定 online time，谁的 finite-budget task value 更好？
2. 固定任务表现，谁的训练/推理成本更低？
3. 图/通道数增大，完整目录何时不可处理，structured decoder 如何退化？
4. 容量或 horizon 不在训练支持中，表现是否稳定？
5. 无组合收益的对照下是否不出现虚构提升？

指标包括 reward/regret、terminal loss（若适用）、online latency、simulator/likelihood calls、teacher nodes/time、训练 GPU 小时、内存、episode safety、return completion、measurement/transit/reset 占比。

不要把只看 `Reg/log T` 当作 main figure。它可作为固定类渐近诊断，不能证明有限预算收益。

### C3：一个外部任务——优先 Source Location Finding

选择理由：DAD 与 Step-DAD 作者代码均有 source-location finding，便于恢复原任务，再明确增加资源协议。不是因为它叫“机器人”，也不是医疗数据。

分两层：

**Native reproduction：** 原模型/先验/噪声、原连续设计空间、原 EIG 评价，先确认接入和基线实现可信。严格记录偏离作者配置之处。

**Resource extension：** 新增候选 sensing graph、travel/setup/reset 费用与有限预算；terminal 定位误差或明确决策效用为主结果。连续源参数与旧 finite Bernoulli hypotheses 不同，因此旧 theorem 不自动适用。

重要限制：

- 原 DAD 是可微设计策略；改成离散路径 decoder 后可能要 score-function 或其他训练估计，不能把它叫未修改的官方 DAD。
- 报告 `DAD-style resource adaptation`、`Step-DAD-style resource adaptation`，列出与原算法差异；native 结果保持单列。
- 增加费用/图是人工实验条件，应标为资源扩展模拟任务；无真实采集依据时不称“真实传感系统验证”。
- 先只做这一个外部任务，不同时引入因果图、UAV、LLM、实验室平台。
- 如资源约束导致与 DAD 适配过大，明确比较的是 shared policy backbone + infer/refine思想；不得用错误梯度训练一个弱版本来“击败 DAD”。

### C4：统计与预算 contract

**重复层级：** 训练初始化 seed → 独立 task/graph → capacity/horizon paired cells → observation noise。相同图的多个 horizon 不是独立问题样本。

**建议流程：**

- 开发 pilot 可先用 2 个训练 seed、16 个独立 task、每 task 16–32 个噪声 rollout，用于 profiling/方差估计；这不是正式效果结论。
- 正式训练 seed 数、task 数和 rollout 数据方差、effect size 与计算限制冻结；建议预算草案是 5 个训练 seed、每测试层 32–64 个独立 task，具体数值待 pilot。
- 不因显著性不足自动补 seed 或只延长有利 horizon。需要变更时登记为第二探索轮，新确认数据独立。
- 报告 mean/median 和合适区间；分层或 crossed bootstrap 正确保留配对与 trained-model 依赖；多 primary contrasts 预注册 multiplicity 处理。
- 对 continuous source 参数，不称 finite-class worst-case evaluation 为 continuum minimax。

**成本对齐：** 将 offline teacher、training、online planning、Step-DAD fine-tuning、likelihood calls 分栏。硬件相同/可换算，不把 offline 成本藏起来。

预算计算使用测量值：

`CPU hours = sum(job_count * measured_median_seconds)/3600`。

`GPU hours = sum(training_seeds * methods * measured_hours_per_training)`。

另加预先批准的失败重试与日志空间上限；不是填入未经测量的承诺值。

初期工程假设为一台 32–64 GB RAM 机器与一张 16–24 GB GPU 可用于小规模开发；不是实际硬件核验，也不是正式实验可行性的保证。先通过 inventory 和 batch profiling 决定。

## 7. 建议目录与 PR 划分

以下均为**拟新增路径**，当前仓库不存在这些入口：

```text
research/feedback_protocol/
  pyproject.toml
  lock/                         # 与原 .venv 隔离
  src/fpl/
    problem.py                  # PublicProblem / PrivateTruth 分离
    environment.py              # graph, resource, clock, feedback semantics
    belief.py
    safety.py
    protocols.py                # committed protocol + pending execution
    teachers/exact_bayes.py
    teachers/minimax_small.py
    teachers/bounded_search.py
    policies/channel_cover.py
    policies/structured_reference.py
    policies/posterior_sampling.py
    policies/generic_policy.py
    policies/protocol_decoder.py
    adapters/location_finding.py
    logging/manifest.py
    evaluation/metrics.py
    evaluation/statistics.py
    cli.py
  configs/{fixture,train,pilot,confirm,ood}/
  tests/
  README.md
```

新数据建议放 `artifacts/fpl_v1/` 或外部 artifact store，并有独立 `dataset_manifest.json`、`training_manifest.json`、`evaluation_manifest.json`。由于现有 ignore 规则包括模型/数组/CSV，不能默认这些会被 git 推送；选择小 fixture 跟踪、大文件独立只读发布，保留 hash。

PR 划分：

1. **PR-A0** 研究入口与 evidence inventory：不改实验科学代码。
2. **PR-A1** 通用 simulator、public/private API、fixture regression。
3. **PR-A2** 教师接口、独立 baseline、统计 schema。
4. **PR-B1** 模型、teacher 数据管道与 ablation switches。
5. **PR-C1** 可恢复 runner、正式数据拆分与 pilot。
6. **PR-C2** 外部任务 adapter、训练/验证报告。
7. **PR-PAPER** 仅在有数据后调整主张与正文。

不要把这些 PR 全部一次写入 master。原 frozen source hashes 不变。

## 8. 6–8 周建议排期与决定点

假设：1 名博士生主导，0.5 名工程协作者或等效 agent 支持，可使用上述小规模算力。总时长是估计；没有运行 profiling 前不能保证。

| 阶段 | 时间建议 | 主要交付 | 完成意味着什么 |
|---|---|---|---|
| 仓库与问题冻结 | 第 1 周前半 | A0/A1 | 资料、权限、目标清楚；不是 scientific pass |
| 通用 fixture 与独立 baseline | 第 1–2 周 | A2/A3/B0 | 能进行真实比较，不再只有相同 wrapper |
| teacher 与候选方法 | 第 3–4 周 | B1/B2 | 一套可运行、无 truth 泄漏的学习协议 |
| profiling pilot | 第 4–5 周 | C1/C2 开发版 | 明确预算、方差、是否值得规模化 |
| 外部任务恢复与适配 | 第 5–6 周 | C3 | 原任务复现与资源扩展分开 |
| 冻结确认与结果审查 | 第 6–8 周 | C4 + 报告 | 依据效应/机制/泛化判定主张 |

只保留三个研究决定点：

- **能否建立可信比较？** 若不能，修接口与 baseline，不做效果宣传。
- **结构方法是否有独立能力收益？** 若只有简单 coverage 修正有效，记录并缩小算法故事；不再发明新 theoretical gate。
- **收益是否在未见任务/相同预算下保持？** 若不能，保留理论/诊断论文，不把 benchmark 构造等同于新范式。

这些是项目决策点，不是人为定义的 ICML 录用门槛。

## 9. 论文证据与主张映射

| 想写的主张 | 必须具备 | 不能用什么替代 |
|---|---|---|
| 资源配置影响有限学习风险 | 已有精确 separation，独立 proof audit | 更漂亮的图 |
| 学习协议可扩展 | 新 graph/size 上的 runtime+quality 比较 | 多个名字不同的同构 toy |
| finite-budget 训练有价值 | 相同结构/训练量下对照 EIG、myopic 和 task-utility baseline | 仅击败 UCB |
| 组合结构有价值 | 与全路径/非结构架构同算力对照，和 coverage 修正比较 | 新增可选动作后无条件多探索 |
| 可用于外部 ML 任务 | native reproduction 与资源扩展实验，明确真实/模拟 | 第二个文字映射 |
| “新范式” | 更广问题接口、实际新能力、可复用方法和独立证据 | 名字、模块数量、或 prior-art 没搜到 |

定稿按 soundness、originality、significance、presentation 分别陈述证据。不再给文档完成度兑换 4.0/4.5 的分数。

## 10. 需要用户完成、授权或协调的工作

### 10.1 本地工作区与算力

运行随包提供的只读 inventory，不运行任何研究脚本；检查输出是否含不愿分享的文件名。需要你确认：本地工作区路径、HEAD/未推送差异、已访问 artifact 是否保留、GPU/内存可用预算、可投入人时、计划投稿年份/截止日期。

不需要提供 SSH key、PAT、服务器密码或任何秘钥。

### 10.2 原始证据

目前远端没有原 `artifacts` 树。你应在本地保存原始 manifest、sealed prediction、T=4096 的 20 个 checkpoints、分析结果及 strict certificate；提供只读清单/hash 即可。不要发送旧 CONFIRM，不要把重新运行生成的文件标为原记录。

### 10.3 外部代码与版权

DAD/Step-DAD 采用隔离环境。固定 author repository commit；阅读其 LICENSE 与数据条款再复制/修改/发布，保留 attribution。README 可读不代表所有代码和数据可以无条件再分发。

### 10.4 独立审查

委托未参与构造的人检查：完整 policy class、terminal qualification、Bayes/minimax 差别、最大固定实例系数、finite-hypothesis upper 的条件。另请熟悉 Bayesian experimental design/structured bandit 的研究者评估 proposed 与 DAD、Step-DAD、goal-oriented OED 和有限预算规划的差异。

我能做源码审查、推导复核和新实现对照，但不能把这些自称为外部独立审查。

### 10.5 执行授权

每次执行授权要绑定：代码 commit、config hash、数据分区、运行/算力上限、停止点、是否允许结果分析、是否允许恢复。

不逐步设置无穷 performance gate。focused tests + 新入口 tiny smoke + 首 checkpoint integrity；通过后按你的明确授权完成目标运行。与旧任务绝不自动串联。

## 11. 第一轮应具体做什么

**目标是完成 A0/A1 和一个独立比较骨架。**

1. 本地 inventory，确认缺失 artifacts 和可用算力。
2. 独立分支/工作目录，建立新 ProblemSpec 与 finite-time comparator contract。
3. 参数化 simulator，原 fixture 作为不变回归。
4. 加入 channel coverage 与 budgeted Bayes lookahead，避免把简单解决方案跳过。
5. 输出新 runner/config 设计与必要 focused tests 清单，随后申请小规模执行权限。

这轮不要求新反例、新 theorem、完整 neural training 或 UAV。

## 12. 来源

### 仓库（均以冻结 SHA 为准）

链接前缀：`https://github.com/ZhangJL16/EA-MAPPO/blob/fff2b026ca3e4ea6b68cacc3964971b0749bfeb3/`

- `AGENTS.md`、`README.md`、`task_plan.md`、`notes.md`
- `experiments/bundling_calibration_core.py`
- `scripts/run_bundling_calibration.py`
- `scripts/analyze_bundling_calibration.py`
- `scripts/verify_bundling_finite_budget_theory.py`
- `scripts/audit_resource_separation_bridge.py`
- `tests/test_bundling_calibration.py`
- `.gitignore`、`requirements.uv.txt`

### 外部主来源

- ICML 2026 reviewer instructions: https://icml.cc/Conferences/2026/ReviewerInstructions
- ICML 2025 官方 Spotlight 列表: https://icml.cc/virtual/2025/events/2025SpotlightPosters
- PDNAR: https://proceedings.mlr.press/v267/he25r.html
- Probabilistic factorial design: https://proceedings.mlr.press/v267/shyamal25a.html
- DAD: https://proceedings.mlr.press/v139/foster21a.html
- DAD author code: https://github.com/ae-foster/dad
- Step-DAD: https://proceedings.mlr.press/v267/hedman25a.html
- Step-DAD author code: https://github.com/marcelhedman/stepdad
- OSSB: https://proceedings.neurips.cc/paper/2017/hash/e19347e1c3ca0c0b97de5fb3b690855a-Abstract.html
- Goal-oriented OED, 2026: https://epubs.siam.org/doi/10.1137/24M1649344

外部 code README 已读取，未安装、运行或逐行审计其算法。文献核对为有边界的研究规划，不是全球新颖性清关。
