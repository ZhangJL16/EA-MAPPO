# 原始 SAC 环境完整问题审计

日期：2026-09-20。依据用户本次独立授权，仅审 F0–F2；旧 regenerative-control 结项不变。
权威对象：`envs/UAVEnergyDeliverySAC.py`，不是其父类 `UAVEnergyDelivery.py`。

## 结论

**这个文件实现的是单目标连续导航与返航切换，不是多候选任务 pickup–delivery scheduling。**
它没有实现用户所描述的完整持续配送问题。之前 99.134% 不能外推到完整配送系统；
同时，当前 SAC 环境也不能直接提供所要求的多任务调度实验。
这不是“完整问题已被 heuristic 解决”的证明，更不是新的 oracle ratio。

## F0：逐项核对

行号均针对本次未修改的 `envs/UAVEnergyDeliverySAC.py`。

| 项目 | 实际语义 | 源码位置 |
|---|---|---|
| Task pool | 父类初始化参数为单机、单订单；随后 `orders=[]`。运行维护一个 `current_task_point`，没有等待订单队列 | 568–585、698–720 |
| Arrival | reset、任务完成、充电完成时按需生成目标；可选 timeout rollover 也生成目标。没有独立随时间到达的订单过程 | 978–986、1206–1223、1251–1272、2349–2354 |
| Pickup / delivery | 一个三维目标点；没有订单级 pickup/dropoff 两阶段或取货状态 | 750–756、3009–3060 |
| Heterogeneity | 随机位置导致距离、导航时间及耗能变化；每次完成奖励相同，默认100。没有任务价值字段 | 486、1200–1217、2567 |
| Deadline | 无订单 deadline/expiration。`max_steps_per_task` 是导航超时；energy-estimate 的 deadline feasibility 是模型可达期限语义，不是订单期限 | 482–484、1224–1243、1657起 |
| Payload | 无载荷状态、载荷速度或能耗耦合 | reset、step、telemetry cost |
| Charger | 固定一个充电站、单机，无站点选择或竞争 | 582–585、750–752 |
| Charging time | 返航飞行耗时；到站立即将能量设为capacity，额外充电时间为0。父类 charging_rate 不在此覆盖后的 step 中用于逐步补能 | 1218–1223、2268–2356 |
| Partial charge | 无充电动作或目标SOC，到站直接补满 | 2349–2354 |
| Waiting | 无高层 wait。零加速度不等于等待：有残余速度、基础耗能、导航超时，也不会等来新候选订单 | 1041起、telemetry cost |
| Abandonment | TASK 可被返航承诺打断；到站后抽新目标，旧目标未排回队列。可选 workload 模式在任务超时后换目标。没有独立取消动作/取消费 | 1067–1071、1251–1272、2349–2354 |
| Episode | NAVIGATION/TD_PRETRAINING 完成单目标即terminated；ENERGY_MANAGED 完成目标不reset，位置/电量/时钟连续，但速度清零并抽新目标 | 1196–1217、1324–1325 |
| Horizon | 存在环境内部任务及episode step guards，触发truncated；并非只有外部固定T cutoff。battery-validation特例绕过episode guard | 1224–1250、1324–1337 |
| Navigation | 构造环境不会加载/冻结SAC。`enable_phase_two`要求绑定导航provider；调用方需保证权重冻结。保留模型依赖2056维的导航包装观察，裸环境默认观察仅7维 | 700–718、798起、858–875 |
| Energy | 对物理子步实际速度和加速度积分，不是仅由距离决定；只在ENERGY_MANAGED扣实际电量。capacity默认None，NAVIGATION能量1是占位值 | 473–474、627–642、1109–1140、1576–1590 |
| Objective | 源码reward是导航进度、速度、完成奖励及时间/接触/干预罚项之和；`tasks_completed`可用于外部N(T)评价，但没有现成完整调度目标实现 | 2545–2581 |

默认能耗（synthetic units）每个物理子步为

\[
\Delta e=\Delta t\left(0.06+0.005\sum_j|v_j|+0.005\sum_j a_j^2\right).
\]

一般配置另有乘子与每子步simulation_error；依据
`runtime_support/review_bundle/envs/navigation/telemetry_cost.py:11–68`。
不把synthetic单位解释成真实Wh。

### State 与 action

物理及任务状态可写为
\(s=(x,v,e,g,m,t,k,n,c,\mathcal M,\mathrm{RNG},\ldots)\)：
当前位置/速度/电量、唯一目标、TASK或CHARGER_COMMITTED模式、时间、当前任务步数、
完成数、cycle状态、地图及随机状态。地图是simulator状态，不能直接输入deployable actor。
裸SAC观察为归一化速度3、目标方向3、距离1；开启LiDAR后再加距离和hit flags。
冻结checkpoint的导航wrapper另加剩余option时间；电量与已知充电站可供高层使用。

底层Gym动作严格是 \(u\in[-1,1]^3\)，映射为三轴加速度。
现有ReturnManager的高层选择是

\[
\mathcal A_{HL}(s)=
\begin{cases}
\{\mathrm{continue\ current\ target},\mathrm{commit\ return}\}, & m=\mathrm{TASK},\\
\{\mathrm{continue\ return}\}, & m=\mathrm{CHARGER\_COMMITTED}.
\end{cases}
\]

这是控制选项集合，不是已经证明能量安全的admissible集合。
具体manager用规则选commit与否（1792–1840、2122–2127）；不会选择订单编号。
返航承诺在到站前不可撤销。它不是 \(\{serve_1,\ldots,serve_K,recharge,wait\}\)。

## F1：结构与状态证据

可复现脚本：[audit_structure.py](research/full_delivery_audit/audit_structure.py)。
原始记录：[reset_state_census.json](research/full_delivery_audit/reset_state_census.json)。
运行命令：`PYTHONPATH=.:runtime_support .venv/bin/python research/full_delivery_audit/audit_structure.py`。

检查16个默认环境reset states：目标数恒为1，legacy订单列表长度恒为0。
这是初始化状态核查，**不是长轨迹分布、任务完成率或能量可行性验证**。
未运行策略；action、successor、执行时间、能耗估计均为null，不能填造为已观测结果。
默认phase没有有限电池，不能据此报告安全可行任务数量。

从状态转移源码可进一步确定：整个该实现不存在多目标候选池的扩展路径。
因此任意状态的可选择任务数最多1；能量可行数至多1，具体是0还是1未验证。
此结构结论无需用长轨迹去估计。D2的长期arrival/latency统计、D3的实测安全可行率未完成。

## F2：baseline适用性审计

| 方法 | 在当前原始环境中的状态 |
|---|---|
| Nearest-task / shortest-job / energy-greedy / throughput-greedy | 只有同一个目标可选；排序动作完全退化，无可比较的任务选择 |
| Reserve-safe nearest / shortest | 在同一feasibility估计与返航规则下退化为同一个策略；仍需估计task+return安全，当前未验证 |
| Charge-threshold + greedy | 可以比较返航时机，但属于单目标C/R管理，不是完整调度baseline |
| 1-step lookahead / MPC-H=2,3,5 | 没有多个已知待办任务可作序列搜索；对未知目标采样做C/R规划是另一项实验，不可冒称完成调度MPC |
| Oracle / near-oracle | 本轮不执行，属于后续F3 |

**本轮没有baseline性能表，也没有新的N(T)、失败率或最优比值。**
F2完整调度运行被实际state/action接口缺失阻断；增加订单池、两阶段配送、
独立arrival或付费/部分充电都会改变机制，与本次“不加新机制”冲突，故没有擅自实现。
另：在所有任务收益均为1、使用同一时长估计时，shortest-job和throughput-greedy本就等价。

## 与之前99.134%实验的关系

| 维度 | 此SAC原始文件 | 已结项P2-A2 |
|---|---|---|
| 高层决策时机 | 可在当前目标执行过程中检查返航，默认每policy step | 完成任务后，在未知下一任务前C/R |
| 任务生成 | 随机三维目标；按完成/充电/特定timeout触发 | 冻结有限三任务类型序列树 |
| 充电 | 到站瞬时补满 | paid recharge，含停靠/overhead/补能时间 |
| 安全集 | 取决于绑定的ReturnManager/estimator | robust immediate-return safe tree |
| 多候选pickup–delivery | 没有 | 也没有覆盖完整调度问题 |

因此两个结论必须同时保留：旧99.134%有严格适用边界；这个原始文件没有展示
用户期待的额外task-selection自由度。不能据此证明两者最优值相同或RL没有价值。

## 完整性与交接

只新增本报告和独立audit目录；未改环境、reward、碰撞规则、模型或历史证据。
裸SAC继承父类位置修复并保留分类型接触计数；新的实际飞行须沿用已有
`LockedRecoveryPlant`的全速度清零及统一policy-step计数包装，不能直接复用旧统计。
本次没有飞行、训练或后台任务；不需要将零步状态检查冒充startup-health。

F0完成；F1仅完成结构核查与reset census；F2仅完成适用性审计，未运行性能比较。
如要继续用户描述的完整调度研究，需要另行明确授权实现缺失的任务机制，或提供已有的
完整调度wrapper入口。当前不自行进入F3–F10，也不重开旧研究。
