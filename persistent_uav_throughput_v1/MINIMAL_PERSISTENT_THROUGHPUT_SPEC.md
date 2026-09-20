# PersistentUAVThroughput-v1：最小持续任务吞吐量环境规格

## Material Passport

- 日期：2026-09-20；修订v1.2。状态：**已授权修复空队列语义、完成calibration及资格审计；B0–B5运行待用户另行批准**。
- 依据：用户本轮最小环境要求、`FULL_PROBLEM_AUDIT.md`、保留的导航/能耗代码。
- 产物类型：环境语义规格；没有性能证据、训练结果或新颖性结论。
- 版本边界：新的问题定义；不修改旧 `UAVEnergyDeliverySAC.py`，不继承旧99.134%结论。

## 1. 问题与范围

一个无人机在固定时间内，从持续到达的待办任务中选择执行任务或回站充满电，空队列时选择补能或idle，
在能量约束下最大化完成任务数：

\[
\max_\pi J_T(\pi)=\mathbb E_\pi[N_{\mathrm{completed}}(T)].
\]

**本提案采用单地点服务任务**：飞到任务位置即完成一项服务。不同任务的执行时间差异
来自导航距离、方向和路径；不另造随机服务时长或不同任务价值。
没有取货点、携货状态或取送先后约束，因此准确名称是“持续空间任务调度”，
不能声称已经模拟完整物流 pickup–delivery。这是本规格供用户审阅的一项明确简化。

只保留一个充电站、一个无人机、静态地图、同质任务价值、全充电。
无deadline、payload、通信决策、部分充电、任务抢占、分时电价或多机竞争。
不预设该问题需要RL，也不把制造heuristic失败作为设计目标。

## 2. State 与可用信息

完整运行状态为

\[
s_t=(x_t,v_t,e_t,\mathcal Q_t,j_t,m_t,t,N_t),
\]

其中 \(j_t\) 是正在执行的任务或空值，
\(m_t\in\{\mathrm{IDLE, SERVING, RETURNING, CHARGING, WAITING, FAILED}\}\)。
仿真还保存固定地图、外生到达流游标、低层导航及随机状态，用于精确续跑。
它们不因此成为策略可观测信息。

高层可读取当前位置、速度、在线剩余电量、当前时间/剩余时间、所有等待任务的ID、
位置与到达时间，以及已知充电站位置。低层沿用冻结SAC的goal-relative观察及LiDAR。
策略不能读取未来到达、未来任务位置、仿真障碍中心/半径或精确地图路径代价。
部署用时间/能耗估计只能使用上述可用输入；模型估计不是环境真值。

## 3. Task arrival process 与 queue

任务记录为 \(\tau_i=(id_i,a_i,y_i)\)：唯一ID、外生到达时间、服务位置。
每项任务价值均为1，无过期时间。

| 项目 | 提案语义 |
|---|---|
| 初始条件 | \(t=0\) 在站内静止、满电；等待队列有3项任务 |
| 到达过程 | 齐次Poisson过程，\(a_{i+1}-a_i\sim\mathrm{Exp}(\lambda)\)，\(\lambda>0\)固定 |
| 任务位置 | 从固定地图的合法服务区域独立同分布采样；不依赖当前无人机位置或所选策略 |
| 队列容量 | 最多5项等待任务；正在执行的任务另计，最多1项 |
| 满队列到达 | 新任务直接拒收，记为overflow；不挤出旧任务，不允许策略挑选拒收对象 |
| 空队列 | 允许出现；不即时补任务，也不强制维持至少2项 |
| 队列变化 | 仅由外生到达、开始服务时取走所选任务引起；充电或完成不会触发抽样 |

初始任务与未来任务采用同一位置分布。地图几何仅用于环境生成合法目标，
不为任何策略提供privileged clearance；不按某个baseline的成败筛选目标。

所有方法共享**完全相同的外生到达时间、任务位置与ID流**。
即使到达时队列已满，也消耗该条外生事件并记录拒收；不能推迟到下一次有空位时再抽取。
流可提前生成以复现，但只在到达时对策略公开。不同策略接受的任务集合可以不同，
这是有限队列的结果，不是随机流不同。

“2–5个candidate tasks”指初始化和运行中允许多个候选任务共存，
不是每个时刻保证至少2项。容量5是最小计算规模约束；overflow须单独报告，
不能把队列截断造成的差距隐去或包装成普遍调度机制。

## 4. Action 与决策时机

仅在IDLE决策点选择

\[
\mathcal A(s_t)=\{\mathrm{serve}(i):i\in\mathcal Q_t\}
\cup\{\mathrm{recharge}\},\qquad\mathcal Q_t\ne\varnothing.
\]

空队列且未处于站内满电时，动作集合为
\(\mathcal A_0(s)=\{\mathrm{recharge},\mathrm{idle}\}\)。
站内满电且空队列时仅自动执行idle，不调用scheduler。

IDLE包括初始化、完成一项任务、完成全充电，以及空队列等待后的任务到达。
任务执行、返航和充电期间处理新到达，但不中断当前动作。
因此这是动作持续时间不等的高层决策过程，不是每个物理步重选任务。

- `serve(i)`：立即将任务从等待队列移到当前任务槽，交给同一个冻结导航器执行。
- `recharge`：导航至唯一充电站，然后充至满电；不能途中撤销或提前离站。
- 空队列可选recharge或idle；idle持续至下一次到达、能量耗尽或T，以先到者为准。
  未来事件时间不提前暴露；非空队列仍不能选择idle。
  站内已满电且空队列时没有决策，强制idle。

站内已满电时屏蔽`recharge`，避免零时间动作循环。其他位置即使满电仍可选择回站。
空队列允许利用空闲期返航补能；充电仍需显式recharge，不能靠idle自动补电。
不存在取消已接任务、免费换单或立即reset动作。
任务ID有效性与模式限制形成**语法动作集**；不得把未经证明的能量估计混称为真实安全动作集。

## 5. Task service semantics

从当前实际位置飞向 \(y_i\)，进入沿用的目标容差区域、且尚未发生能量失败时完成。
计数加1，当前任务清空，位置保持，速度按既有到达服务语义清零，进入IDLE。
不传送、不补电、不重置时钟、队列或地图；不自动接下一项任务。

服务时间等于实际导航耗时；v1额外驻点服务时间为0。
同一任务从不同前序任务的终点出发，可以有不同的时间和能耗。
不得使用距离除以标称速度代替真实执行时间；该式只可作为明确标注的baseline估计。

导航沿用冻结SAC、物理步长、动作限制、LiDAR和已冻结的碰撞恢复机制。
接触不结束任务，修复瞬间全速度清零；干净步后的boundary/obstacle原始罚项各为−1.2，
上一policy step有任意接触时各为−0.42，干净步恢复；两类同时发生则罚项相加。
安全cost为任意接触指示量，统一collision count每policy step最多计1次。
新统计不输出两个分开的碰撞计数；collision-free arrival依据统一计数为0。
保留低层既有reward，不把其累计值作为高层吞吐量目标。

## 6. Recharge duration 与能量动力学

飞行能耗沿用既有TelemetryCostModel及事先冻结的配置：每个物理子步用实际速度、
实际加速度和时间计算并扣除能量。单位为synthetic simulation energy units。
不改变能耗模型来调出期望结果。

设容量为 \(B>0\)，回站到达电量为 \(e_{\mathrm{arr}}\)。到站后静止，以恒定
**净入电池速率** \(r>0\) 充电：

\[
e(t+u)=\min(B,e_{\mathrm{arr}}+ru),\qquad
\tau_{\mathrm{charge}}=\frac{B-e_{\mathrm{arr}}}{r}.
\]

\[
\tau_{\mathrm{recharge\ action}}=
\tau_{\mathrm{return\ flight}}+\tau_{\mathrm{charge}}.
\]

有电量缺口时充电时间严格为正；不存在instant refill或额外人为固定overhead。
净速率已包含站内辅助耗电，不重复扣一次飞行基础功率。
充电期间时钟推进、订单持续到达，队列可能溢出；充满之前不能接单。

站外等待以静止悬停计，按同一能耗模型的零速度/零加速度成本扣电。
站内idle也不自动补能；补能只能通过`recharge`发生。
等待使用零速度驻留的高层抽象，不等同于向有残余速度的飞行器发送零加速度。
高层决策点均由服务/充电/等待后的静止状态进入，不引入途中免费刹车动作。

## 7. Energy constraint 与失败

取 \(e_{\min}=0\)，运行要求非失败状态中 \(e_t>0\)；到0立即进入吸收失败状态。
绝不免费救援、传送或reset。失败后的剩余评价时间内完成数保持不变。
同一步到达目标并耗尽电量时，失败优先，不计该任务完成。

v1唯一能量风险约束为实际观察到的物理电量耗尽：

\[
\Pr_\pi(\mathrm{energy\ depletion\ within}\ [0,T])\le\delta.
\]

初始共同风险预算固定为 \(\delta=0.05\)，不由baseline结果选择。
实测失败率不是概率保证，报告样本数与不确定性；不以少量零失败宣称安全。
recharge failure指补能动作在回站途中发生能量或导航失败；T截断单列为censored。
navigation failure、recharge failure与terminal residual battery均单列报告。

**returnability仅为未来可选的offline privileged diagnostic**，不进入v1可接受策略定义、
环境动作mask、终止条件或主约束。当前不运行counterfactual simulator branching。
deployable reserve-SJF可使用可计算的task+return能量估计，但它不是环境真实安全oracle。

## 8. Termination / evaluation horizon

\(T>0\) 是从初始化起所有方法相同的**模拟时间cutoff**，不是完成固定任务数。
任务完成或充满电不会结束episode；到达T时截断当前动作，保留终态。
仅统计在 \(t\le T\) 已完成的任务；未完成任务不计部分奖励，未用电量无终端奖励。
若T截在充电中，只增加实际经过时间对应的电量，不能提前记满电。

T时不要求实际回站或反事实返航能力，也不赠送评价窗外的返回时间。
允许策略在T前使用剩余电量提高完成数，只要没有实际耗尽；这属于本版有限时域目标。

能源耗尽为terminated。既有导航执行上限若被触发，则记为navigation failure，
吸收至T，不丢单后换任务继续刷分；其阈值对全部方法一致并写入配置。
物理接触本身不是terminated，也不触发换任务。

同一时刻按以下顺序处理：物理/能量更新及失败判定、有效任务完成或充满事件、
按ID排序的到达事件、T截断、下一次高层决策。
连续到达时刻可统一向上量化到物理时间网格；保留原时刻和有效时刻，对所有方法一致。
T必须对齐物理网格，充电末步按实际缺口封顶。

## 9. Primary metric 与公平性

Primary：每个外生seed下的 \(N_{\mathrm{completed}}(T)\)，再估计其期望。
失败run仍包含在主统计中，不能只平均成功run。
固定T下可附报 \(N(T)/T\)，但不能用不等时长的完成率替代N(T)。

必须同时报告：depletion、recharge failure、navigation failure、
统一collision count、队列长度分布、拒收率、实际飞行/充电/等待时间和剩余电量。
这些不通过加权和进入主reward；若后续需要高层reward接口，仅用完成计数增量，
能量作为约束单独处理。

所有方法共享T、地图、初始3项任务、外生到达流、B、r、能耗参数、导航器及执行上限。
不同策略引起的路径、队列和拒收差异是被比较的结果。
本轮B0–B5只能看当前已到达任务，不偷看未来流。MPC与oracle不在首轮实现授权内。
若将来reference能预知未来，它只是clairvoyant上界，不能直接冒充同信息条件下的最优策略。

## 10. 配置边界与审阅重点

本规格定义参数化环境，不伪造尚未确定的物理容量或实验负载。
\(K_{\max}=5,K_0=3\)、单站、全充电、单地点服务是本次明确提案；
其余实例参数按第11节先验calibration协议冻结；不存在可调的主动等待间隔。

用户已认可单地点服务、容量5、全充电与不抢占。不能悄悄扩展成取送货、主动拒单或部分充电。

概念上，本版本增加的决策只来自：多个已到达任务间的选择、真实外生任务流、
占用运行时间的补能。没有证据表明这些结构必然使简单策略失效。

## 11. 与调度结果独立的参数calibration协议

先执行1000条独立x→y冻结SAC导航任务，不运行任何scheduler。
固定一张原尺度地图（4000×4000×400、24个障碍）和合法服务空间，
位置在合法区域独立均匀采样；不按导航成功或baseline表现重新选图、缩短距离或换seed。
地图、输入pair列表、checkpoint和源码哈希在开始前写入manifest。
1000是预先选定样本量，不看结果后扩到5000来筛选有利样本。

记录每条实际时间、能耗、是否到达、统一接触数、导航timeout。
成功任务给出T_job与E_job的q25、median、q75、q90；失败保留耗费时间/能耗和原因，
不能把timeout的部分耗能当成成功任务总代价。用成功样本的条件median定义典型任务尺度，
同时报告成功比例和条件统计的局限；没有成功样本时不能生成参数。
失败样本不能从后续环境绩效中删除，也不能据此默默改变物理任务分布。

令m_T、m_E分别为成功任务时间和能耗的条件median。冻结全因子27个regime：

| 无量纲量 | 预先固定取值 | 物理参数 |
|---|---|---|
| 满电典型任务数 B_tilde | 2、4、6 | B = B_tilde × m_E |
| 从空到满充电时长比例 charge_tilde | 0.5、1、2 | r = B / (charge_tilde × m_T) |
| offered load eta | 0.5、0.9、1.2 | lambda = eta / m_T |
| 评价窗 | 100个典型任务时间 | T = 100 × m_T，向上对齐物理时间网格 |

eta是相对于未计补能的典型导航时间的offered load，不声称等于实际队列利用率。
实际queue occupancy、empty fraction与overflow rate必须报告；充电和失败会改变service rate。
不同regime用同一物理地图和导航器，所有baseline使用同一个冻结参数文件。
不得依据算法gap删去某些regime。启动工作只推进calibration，不自动启动27组baseline sweep。

## 12. 首轮baseline边界

| 方法 | 任务排序与补能规则 |
|---|---|
| B0 FIFO | 到达时间最早；共享固定SOC阈值0.25补能 |
| B1 nearest | 欧氏距离最近；共享固定SOC阈值0.25补能 |
| B2 shortest-time | 预计时间最短；共享固定SOC阈值0.25补能 |
| B3 energy-greedy | 预计能耗最低；共享固定SOC阈值0.25补能 |
| B4 threshold+SJF | 预计时间最短；阈值仅在独立validation seeds从0.1、0.25、0.5、0.75中选 |
| B5 deployable reserve-SJF | 在预计task+return能耗小于当前电量的候选中选预计时间最短；无此候选则补能 |

B0–B3共享阈值仅为完整策略的透明定义，不保证安全。站内满电无法再次补能时，
所有方法必须选任务；B5若全部预计不可行，选预计总能耗最小任务并记录fallback，
不能凭不存在的wait动作滞留。该运行仍计入depletion统计。
平局均按task ID处理，不访问未来任务。
时间与能耗估计采用calibration位置差的简单非负线性回归（截距、水平距离、垂直距离），
不训练神经网络、不输入地图几何，也不查询仿真未来分支。模型误差须承认，不冠以“Safe”。
B4 validation按满足实测depletion率预算的配置中最高平均N(T)选；若无配置满足则明确标为
validation不可行，不自动选一个并宣称安全；validation样本率也不是统计安全证书。

OracleSafe-SJF、MPC和small near-exact reference均留待用户后续授权。
未来98%检验的分子必须是deployable simple scheduler；特权诊断不能混入其中。

新实现位于独立项目 `/home/zjl/mappo/persistent_uav_throughput_v1/`，旧源码与历史结果只读复用。
用户本轮明确授权完成当前1000条calibration、修复软件初始化错误并提交审计证据。
随后先做导航资格审查；任何B0–B5、MPC或oracle运行都须用户后续批准。

## 13. 导航资格、状态与证据

资格分析遵循 NAVIGATION_QUALIFICATION_PROTOCOL.md：先报告总体/固定几何分组成功率及Wilson区间、失败原因与失败耗时比例，出现预定暂停信号就暂停调度实验。规则在本轮读取结果前写定，但pilot已开始，因此不能声称采集前预注册。无暂停信号也只进入REVIEW_REQUIRED，不自动通过。

B0–B3为共享SOC阈值的task-ranking diagnostic，不作为最终strongest-simple；B4/B5及未来MPC另行检验。所有预留baseline在空队列时共享透明规则：低于自身SOC阈值且可充电时recharge，否则idle；这只是实现定义，本轮不运行它们。

原始v1.1源码、失败检查点与错误保留在evidence/original_v1_1；第94号pair的合法起点被旧地图生成保护距离误拒，是软件初始化问题，不计navigation failure。修复不改变固定地图、1000对输入或飞行物理；续跑迁移须保留原manifest及逐文件SHA的对应关系。
