# 内部派生命中通道消融 v3（2026-09-07）

## 证据与问题

固定500任务：旧SAC-raw到达496、安全到达423；v2-raw到达461、安全到达372。
按源场景配对：旧模型安全而v2不安全85例，反向34例，净损失51任务。
安全率差−10.2个百分点，分距离层配对bootstrap95%区间[−14.4,−6.0]个百分点。
这描述两个冻结seed0模型，不是跨训练种子结论，也不能将整组改动的差异归因
于单通道。v2的39次超时中30次发生碰撞；这些任务贡献95481/110611次碰撞。
共同到达458任务上，v2平均能耗28.0677 vs25.3905，不能以选择不同成功子集
掩盖路径/能耗恶化。完整数值、分析限制和两幅图见
artifacts/deployable_v2_fixed500_analysis_20260907_v1/。

用户询问HOCBF后已核对：此处旧SAC与v2在训练和固定500评测中都未开启HOCBF；
曾经的SAC+HOCBF是另一组返航接口对照，不能混为基线。v3同样不启用HOCBF，
避免一次改变表示与控制接口。这不是拒绝今后评估过滤器，而是隔离当前变量。

## 唯一实验变量

维持v2的1039维外部观测、全部1024条归一化距离、资源上下文、任务采样、
奖励、碰撞恢复、SAC超参数和524288步预算。从头训练seed0，不续训失败权重。
只把第一卷积层的输入从r变成(r,h(r))，h_i(r)=1[r_i<1]。
max-range100m的float32距离与边界邻点测试均与原命中标志一致；不声称任意
量程/量化格式无条件等价。没有读取雷达之外的真值或增加外部传输维度。

## 数学依据与初始化控制

h是r的确定性函数，故sigma(r,h(r))=sigma(r)：它不增加观察到的信息。
但固定容量、有限更新次数下，显式阈值基函数可能改变优化难度，这只是
待检验假设，不是性能保证或新的安全定理。

将新首层写成 W_r*r + W_h*h(r)+b。初始化时复制v2的W_r与b，令W_h=0，
其他参数保持不变，则初始表示函数与v2相同（实际浮点输出以1e-6容差验证）。
新通道从零权重仍可获得梯度；新增仅144权重/encoder，不增加网络层数。
构造新卷积时保留RNG状态，验证SAC actor/critic/target中所有原有参数完全
相同，避免增加通道改变后续随机初始化。环境轨迹在学习后自然可能分化。

正结果只能支持此表示在当前设置中的作用，不能宣称此前退化原因已被唯一
识别；重复种子和其他输入的单独对照仍是算法级因果解释所需。负结果应降低
“删命中通道是主因”的可信度，不能再靠无限延长本消融训练来找最好结果。

## 实现与运行

独立encoder：experiments/directional_navigation/derived_hit_observation.py。
独立runner：scripts/run_deployable_derived_hit_v3.py，训练函数AST与v2完全一致，
复用未变的save/load checkpoint代码及同一个DeployableRecovery环境。
保留所有v2及旧SAC产物，不编辑历史类。资源输入仍仅是导航上下文，本实验
没有学习自主返航、预算约束或能源可持续性的目标，不能声称解决能耗安全。

```bash
uv run --no-project --python .venv/bin/python python scripts/run_deployable_derived_hit_v3.py \
  --output-dir artifacts/deployable_derived_hit_v3_sac_20260907_v1 \
  --battery-capacity 383.35430890654663 \
  --capacity-source artifacts/r3_fixed_baseline_energy_chain/battery_calibration.json \
  --start
```

4项必要测试通过：初始函数/RNG保持与新增梯度、量程边界重建、完整SAC参数
及训练/checkpoint代码一致性、一次16步子进程smoke。checkpoint逻辑未变，
复用已有精确恢复测试，不添加恢复gate。8worker、GPU、计划524288步，参考
前两轮约4–5小时但不承诺精确时限；按完整块保存，SIGTERM/SIGINT或PAUSE可
恢复性暂停，同命令加--resume续跑。只查首个学习checkpoint后交接，不自动
接评测，不监控到完成。后续固定500必须沿用同样的五距离档与碰撞规则。

启动交接：Python PID101139，8192步checkpoint完整，actor/critic各3192次更新，
指标有限、模型/replay/worker状态文件齐全，无ERROR，进程正常。外部1039维、
内部双通道合同已核对；此后停止监控，等待用户查看结果。
