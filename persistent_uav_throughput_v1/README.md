# PersistentUAVThroughput-v1

2026-09-21 当前授权：[Real-State Counterfactual Branching Atlas v1](REAL_STATE_BRANCHING_ATLAS_V1.md)。
从 B5 和 B4-.75 归档按固定哈希抽取实际决策状态，执行 task+return census 与配对 Oracle-Safe SJF continuation。
所有首动作统一从 root 起评价 1308.6 秒，计入首任务；环境、任务流、导航器不变，不训练。
Atlas 已完成 216 roots、956 个一步分支和 456 个 continuation；见 [完整结果](evidence/real_state_branching_atlas_v1/RESULTS.md)。
145/216 roots 存在多个安全任务；128 个完整案例中 82 个出现首动作吞吐差异，17 个 roots 仍有诊断中止导致的未知窗口结果。
现有日志的 [matched-pair 分解](evidence/atlas_matched_pairs_v1/README.md) 已完成：时间与电量联合匹配 79 对，42 对仍有吞吐分叉；未识别主导因果机制，未新增 simulation。
[Retained-counterpart 离线分析](evidence/atlas_retained_counterpart_v1/README.md) 已完成：77 对同 arrival 匹配，简单 counterpart 成本/余量/排名没有近乎完整解释吞吐排序；未新增仿真。
[Divergence-onset audit](evidence/atlas_divergence_onset_v1/README.md) 已完成：41 对多数动作序列早分叉，完成数反复追平；提供离线交互时间线，未识别因果决定时刻。
以下为历史证据与授权记录。

最新授权：只做 [Oracle Recoverability Census](ORACLE_RECOVERABILITY_CENSUS.md)，
固定 205 个历史 failure-predecessor 状态、561 个有界分支；先精确回放恢复完整 simulator state，
再检查 task+immediate-return 或合法的 immediate-return。不运行完整 OracleSafe policy、MPC 或训练。
205/205 状态、561/561 分支已完成并通过 manual collect，见
[完整 Census 结果与原始状态证据](evidence/oracle_census_complete_20260920/README.md)。
A：12/67 有安全任务；B：47/48 立即返航成功；C：59/90 有安全替代任务、28/90 仅返航成功。
这里仅指冻结 controller 下的 bounded continuation；未启动新 policy 或训练。
历史 [Census 启动交接](evidence/oracle_census_startup_20260920/README.md) 保留。

最新离线分析：[Stranding Decomposition](evidence/stranding_decomposition_20260920/README.md)。
166 次返航耗尽分为 67 次 predicted-infeasible fallback、37 次任务低估余量翻转、
40 次任务完成后等待造成的余量翻转、22 次返航预测 false-safe；另有 53 次任务耗尽和 8 次等待耗尽。
仅分析已有日志，未启动新 simulation/policy。不能把所有失败归为估计误差，
也不能把预测负余量等同于真实不可恢复。

最新结果：**B5 reserve-SJF validation 已于 2026-09-20 20:59 完成全部 270 次**，
27/27 个 regime 未满足经验耗尽率预算；总体 depletion 227/270、导航失败 42/270、
补能失败 166/270，平均完成任务 14.61。见 [完整 B5 结果与原始诊断记录](evidence/b5_validation_complete_20260920/README.md)。

本轮授权（审阅 c158cc07 后）：只运行 **B5 reserve-SJF validation，27×10=270 次**。
冻结估计器、参数及零额外 margin 不变；暂缓 B0–B3 与 evaluation，禁止训练及最终 kill test。
新增逐决策预测和失败阶段日志，用于后续 feasibility / estimation / planning 分解。
实现与运行入口见 [B5 validation 说明](B5_VALIDATION.md)。
16 个单线程 CPU 分片均已完成；历史启动核验见
[B5 启动交接与证据](evidence/b5_validation_startup_20260920/README.md)。

2026-09-20 ARM 最新结果：B4 validation 已完成全部 1080 次，27 个 regime 均无候选阈值满足预设耗尽率预算。完整结果、原始轨迹、校验和及迁移修复记录见 [ARM 验证结果](evidence/arm_validation_20260920/README.md)。尚未启动 evaluation 或训练；估计误差审计与估计/规划分解仍待完成。

历史本地状态：本地B4在382/1080因等待网格浮点检查中断；v1.4已修复，34项测试及现场回放通过。
本地不续跑，ARM端同步后在新目录完整运行；见 [ARM_DIAGNOSTIC_HANDOFF.md](ARM_DIAGNOSTIC_HANDOFF.md)。

独立的新问题版本：持续空间任务调度，固定T内最大化完成数，实际电量耗尽风险为约束。
单地点服务、单机、单站、初始3项/最多5项等待任务、外生Poisson到达、计时全充电。
有任务时只选serve/recharge；空队列可选recharge/idle，站内满电时强制idle。站内idle不耗电也不自动充电，站外idle保留悬停耗电。无取送货、非空队列主动wait、部分充电或网络训练。

完整语义见 [MINIMAL_PERSISTENT_THROUGHPUT_SPEC.md](MINIMAL_PERSISTENT_THROUGHPUT_SPEC.md)。
这里复用 `/home/zjl/mappo` 的冻结导航资产与Python依赖，旧仓库物理代码和历史结果不修改。
这不是独立打包后的portable环境；运行前须保留该只读依赖。

## 已实现

- `persistent_uav/environment.py`：可在任意step返回处恢复的任务/补能状态机，真实物理飞行耗能。
- `persistent_uav/navigation.py`：冻结SAC、2056维观察、LiDAR HOCBF、非终止碰撞恢复、统一接触计数。
- `persistent_uav/streams.py`：与策略无关的外生流，固定ID，溢出拒收照常消耗事件。
- `persistent_uav/baselines.py`：FIFO、nearest、shortest-time、energy-greedy、threshold+SJF、reserve-SJF。
- `persistent_uav/calibration.py`：固定1000条无调度导航任务；完成后按条件median生成27个regime。
- `persistent_uav/evaluation.py`：独立validation/evaluation seeds、B4阈值选择、每run N(T)/失败/overflow统计。
- `persistent_uav/storage.py`：原子checkpoint、校验和、进程互斥、源码/模型/依赖一致性检查。

导航初始化后，服务与充电均不reset。原环境的自动目标生成和瞬时充电分支不用于运行。
冻结模型SHA256：`fb79482ae7d832b40137ff1a080ee84912eabb95867e118aca80c846f6ae62be`。
仅加载actor；不加载optimizer或energy critic。

## 运行入口

在本目录执行，使用现有Python环境：

```bash
/home/zjl/mappo/.venv/bin/python -m unittest discover -s tests -v
/home/zjl/mappo/.venv/bin/python -m persistent_uav.cli smoke --output artifacts/engineering_smoke_20260920
```

smoke里的容量、估计系数和1秒cutoff都是工程检查fixture，不是calibration或baseline结果。
已有输出目录不会被新运行覆盖。

历史物理calibration命令（已完成，不需重跑）：

```bash
/home/zjl/mappo/.venv/bin/python -m persistent_uav.cli calibrate --output artifacts/calibration_20260920
```

每100个policy steps或60秒保存checkpoint。退出信号SIGINT/SIGTERM在当前policy step结束后
保存并退出；最近两代checkpoint保留。不要用SIGKILL作为正常暂停方式。
续跑使用同一命令加 `--resume`；活动进程存在时互斥锁拒绝第二个写入进程。
`--stop-after-checkpoint` 可用于显式的分段运行。训练更新数始终为0。

calibration结束只生成 `jobs.json`、`statistics.json`、`frozen_regimes.json`，不启动任何baseline。
统计明确区分成功任务条件分位数与失败任务；不因失败换地图或筛seed。
1000条完整结果生成之前，baseline CLI拒绝运行正式评价。

用户已批准修复dock-idle后的B0–B5 diagnostic。先运行B4 validation，完成后再执行evaluation；不自动串联两个阶段：

```bash
/home/zjl/mappo/.venv/bin/python -m persistent_uav.cli baselines --frozen evidence/v1_2/calibration/frozen_regimes.json --calibration-compatibility evidence/v1_3/calibration_compatibility.json --output artifacts/diagnostic_validation_20260920 --split validation --regimes all
/home/zjl/mappo/.venv/bin/python -m persistent_uav.cli baselines --frozen evidence/v1_2/calibration/frozen_regimes.json --calibration-compatibility evidence/v1_3/calibration_compatibility.json --output artifacts/evaluation --split evaluation --regimes all --threshold-file artifacts/diagnostic_validation_20260920/thresholds.json
```

validation固定10个seed、4个阈值；evaluation固定另外20个seed。选择规则使用实测depletion率预算，
不构成统计安全保证。结果附Wilson区间；例如少量零失败不能证明风险≤5%。
无满足validation预算的B4阈值时，明确标记该regime的B4不可用，保留validation失败结果。
`--regimes 13`可选择中心regime作分段运行，但不得事后只发表有利regime。

## 解释边界

部署估计为calibration的简单非负线性拟合，输入只有水平/垂直位移与截距。
没有地图路径真值或未来分支。B0–B3使用共同固定25% SOC补能阈值；B4只在validation调阈值；
B5用task+return的估计作排序/补能规则，不声称保证安全。
空队列允许补能或idle，站内已满电时强制idle；非空队列仍禁止idle。
电量耗尽后任务数固定至T，失败run保留；未实际返航的returnability不参与准入。

MPC、OracleSafe-SJF、小规模oracle、98% kill test及任何神经训练均未实现/未运行。
最新用户已通过导航资格审阅，并批准修复dock-idle后的B0–B5 diagnostic；首轮结果不能触发最终98% kill test。检查首个checkpoint后交还运行，不自动推进后续阶段。

## 当前证据位置

- `evidence/original_v1_1/`：原始源码bundle（commit 1a13bf0）、16项检查、原smoke、93/1000失败前检查点与原始错误。
- `evidence/v1_2/tests.txt`：修订后22项逐项测试结果。
- `evidence/v1_2/continuing_smoke/`：真实serve→serve→recharge→serve事件轨迹与全结果恢复等价证据。
- `NAVIGATION_QUALIFICATION_PROTOCOL.md`：读取结果前制定的资格筛查规则；并非采集前预注册。

代码现位于当前仓库的 `persistent_uav_throughput_v1/`，没有嵌套Git仓库或submodule；新代码和精简证据由父仓库直接跟踪。原始独立实现的Git历史另以bundle保留。

## 本轮完成状态

1000/1000 calibration已完成，988成功、12次navigation timeout。详见[CALIBRATION_REPORT.md](CALIBRATION_REPORT.md)。修复后的运行保存在 `artifacts/calibration_20260920_repaired/`，精简完整证据在 `evidence/v1_2/calibration/`。原运行目录因初始化错误中断，只作为历史保留。当前不需要续跑calibration。自动资格报告保留历史REVIEW_REQUIRED原文；用户随后明确批准进入scheduling diagnostics。

## v1.3 dock-idle修复与诊断入口

`evidence/v1_3/`保存29项测试、更新的真实continuing trace和calibration兼容记录。
站内70%电量idle保持70%；满电idle保持满电；显式recharge仍需占用时间。
复用的calibration manifest逐字节不变；新diagnostic manifest记录新源码及兼容依据。
B4 validation为27×4 thresholds×10 seeds=1080个持续任务流run。
每run固定T≈10905秒，不是一个单目标飞行；首个checkpoint时不要求完成一个run。
当前只启动该validation阶段；B0–B5 evaluation尚未启动。

解释结果前仍需补estimator的MAE/RMSE/R²、energy underprediction和positive residual tail审计。
10个validation seeds只支持经验风险筛选，不支持5% chance-constraint认证。
