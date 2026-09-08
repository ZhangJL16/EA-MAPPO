# 可部署观测 v2：2026-09-06

用户同意合并雷达标志与加入资源信息，随后明确授权直接启动，覆盖之前的
启动前暂停要求。只做必要检查和启动健康检查，不监控完成、不自动追加实验。

## 改动和边界

新接口为1039维：速度3、当前目标方向3与对数距离1、雷达距离1024、原有
回合剩余步数比例1、充电站相对方向3与对数距离1、电量比例1、返航指令1、
上一步统一碰撞标志1。精确切片见 deployable_observation.py 的 FIELDS。

保留全部1024条射线及其顺序和归一化，不降采样。去除距离阈值生成的命中
通道。卷积输入由2×8×128变成1×8×128；仍是两层卷积、方位循环填充、
俯仰边界填充、2×16有序均值/最大池化。保留7维目标分支，额外8维上下文
直接拼接，输出2088维。无新增注意力、循环网络、critic或安全过滤器。

在下一次决策的观测中，上一策略步碰撞来自 agent.collided；不能读取
agent.prev_collided，否则会错一拍。充电站信息只依赖已知站点与定位。
不输入圆柱半径、中心、真实净空、由全图算的障碍物方向或隐藏扰动。
旧模型和采集脚本不修改；新模型从头训练，不能直接续旧2056维checkpoint。

## 电量来源与可支持的结论

NAVIGATION阶段并不扣减agent.energy。新观测明确采用：

    E_obs = max(0, capacity * initial_soc - cumulative_realized_energy)

累计量来自已有TelemetryCostModel.realized_cost逐步积分，不是未来能耗标签。
部署时应由同量纲电池遥测替换。训练阶段电量账本到零不改变动力学或终止，
每个导航任务重置账本；这不是连续电池航次实验。

容量383.35430890654663沿用
artifacts/r3_fixed_baseline_energy_chain/battery_calibration.json的合成能量单位。
该文件navigation-valid标志为false；不能把这个容量来源当作新策略通过安全
门槛或具备30分钟续航的证据。此处仅是明确来源的仿真预算标尺，不重开校准gate。

本轮保留原任务采样与initial_soc=1，不改奖励，因此是**观测变更的导航实验**，
不是学习自主返航或能量可持续性。返航指令位在本轮task-only训练中为0，
尚无返航模式训练覆盖。reset接口支持mission_mode="return"和initial_soc覆盖，
返航时目标必须是充电站，矛盾目标会报错；不会加隐藏阈值自动切换。
已承诺返航模式不额外表示未完成任务序列。真正的资源决策实验需要另行明确
任务/返航采样与资源相关学习目标，仅加电量输入并不能学出节能能力。

本轮联合改变通道与上下文，不能据一个结果归因某个字段有效；没有测速证据
前不宣称速度提升。已知充电站方向也不等于知道可通行路线或返航能耗。

## 训练与可恢复性

复用上一SAC基线：gamma=.99、学习率3e-4、tau=.005、batch256、replay200000、
warmup5000、8workers、train_freq=1、gradient_steps=-1、auto_0.01熵系数、
target_entropy=-3、独立actor/critic编码器、256×256头、seed0。
计划524288步；不是强制运行时限，不设置到点杀进程。物理和碰撞奖励完全不改。

每4096步一完整训练块，首块和第二块、之后每32768步，以及暂停和结束时保存。
继承已有完整checkpoint代码，保存模型/优化器/replay/RNG/worker状态，包含
电量账本与指令。SIGTERM/SIGINT或输出目录PAUSE文件会在完整块边界暂停。
同命令加--resume续跑；用户要求续跑时才移除其PAUSE请求。不会载入旧观测模型。

默认脚本仅准备，显式--start才启动：

```bash
uv run --no-project --python .venv/bin/python python scripts/run_deployable_observation_v2.py \
  --output-dir artifacts/deployable_observation_v2_sac_20260906_v1 \
  --battery-capacity 383.35430890654663 \
  --capacity-source artifacts/r3_fixed_baseline_energy_chain/battery_calibration.json \
  --start
```

训练结束等待用户，不自动评估/升级。后续正式导航评估仍须使用历史固定五档
距离500任务，不能用随机seed替换。本轮没有连续能量任务安全的结论。

## 必要检查

13个聚焦测试通过：完整距离保持、无用标志污染不影响输出、站点/电量数值、
新旧物理和奖励一致、上一碰撞时间对齐与清零后继续、返航目标一致、单通道
有序编码及梯度、完整checkpoint恢复后下一更新逐参数一致、准备不启动、一次
16步子进程执行smoke。仅第三方依赖弃用警告。启动后只检查首个已学习checkpoint。

启动交接：后台Python PID6740；8192步checkpoint已完整保存，actor/critic各更新
3192次，loss/entropy均为有限值，无ERROR哨兵。第二训练块106.47秒，其中采样
7.02秒。该耗时仅为启动窗口观测，不是全程速度或性能结论。此后不监控完成。
