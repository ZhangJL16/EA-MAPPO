# PersistentUAVThroughput-v1

独立的新问题版本：持续空间任务调度，固定T内最大化完成数，实际电量耗尽风险为约束。
单地点服务、单机、单站、初始3项/最多5项等待任务、外生Poisson到达、计时全充电。
有任务时只选serve/recharge；空队列可选recharge/idle，站内满电时强制idle。无取送货、主动wait、部分充电或网络训练。

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

物理calibration（本轮唯一启动的研究运行）：

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

以下为后续实现入口示例；**当前代码拒绝baseline运行，须资格审查及用户后续批准**：

```bash
/home/zjl/mappo/.venv/bin/python -m persistent_uav.cli baselines --frozen artifacts/calibration_20260920/frozen_regimes.json --output artifacts/validation --split validation --regimes all
/home/zjl/mappo/.venv/bin/python -m persistent_uav.cli baselines --frozen artifacts/calibration_20260920/frozen_regimes.json --output artifacts/evaluation --split evaluation --regimes all --threshold-file artifacts/validation/thresholds.json
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
最新用户授权完成calibration并提交资格报告；不自动推进baseline。当前baseline入口明确拒绝运行，等待用户资格审查后批准。

## 当前证据位置

- `evidence/original_v1_1/`：原始源码bundle（commit 1a13bf0）、16项检查、原smoke、93/1000失败前检查点与原始错误。
- `evidence/v1_2/tests.txt`：修订后22项逐项测试结果。
- `evidence/v1_2/continuing_smoke/`：真实serve→serve→recharge→serve事件轨迹与全结果恢复等价证据。
- `NAVIGATION_QUALIFICATION_PROTOCOL.md`：读取结果前制定的资格筛查规则；并非采集前预注册。

代码现位于当前仓库的 `persistent_uav_throughput_v1/`，没有嵌套Git仓库或submodule；新代码和精简证据由父仓库直接跟踪。原始独立实现的Git历史另以bundle保留。
