# 二维解析对照：首检查点

日期：2026-09-26。此运行属于独立合成二维项目，不属于已结项的 regenerative-control 分支。全图解析对照采用同一可见图最短路线与局部 CBF 跟踪，仅看当前目标；返站能量与碰撞执行过滤复用环境安全层。两个补能变体分别为 full（回站补满）和 reference_minimum（参考站→当前目标→站能耗加固定 5% 容量余量，至多补满）。后者的 5% 是显式假设，尚未做敏感性分析；两者都不是最优 MPC，也未比较学习策略。

[baseline.py](baseline.py)实现对照策略，[baseline_runner.py](baseline_runner.py)按固定 manifest 保存原始事件、状态与可信本地 pickle 检查点。运行器默认只推进一个高层决策；恢复时核对标定文件与执行源码哈希，防止混合不同版本。新检查点路径经针对性测试验证：连续执行 3 步与先执行 1 步、恢复后执行 2 步的事件和状态完全一致。

本次只启动 map_id=0、full、600 仿真秒配置的**首个 0.5 仿真秒决策**。结果为：成功离站，电量 84.72690610868959，统一碰撞 0，安全代价 0，能量过滤事件 0，任务完成 0；安全层对 10 个低层步的控制作了调整，不能把它们算成碰撞。这个检查点只证明运行入口健康，不是 600 秒结果，也不支持任何吞吐量结论。

原始[manifest](../artifacts/dual_constraint_2d_fullmap_baseline_20260926/map000_full/manifest.json)、[事件](../artifacts/dual_constraint_2d_fullmap_baseline_20260926/map000_full/events.jsonl)、[状态](../artifacts/dual_constraint_2d_fullmap_baseline_20260926/map000_full/status.json)及[可恢复检查点](../artifacts/dual_constraint_2d_fullmap_baseline_20260926/map000_full/checkpoint.pkl)均已写入。Manifest SHA256 为 365143e301ea2b0792dadc81be5d074d7d8ab4f1b99272947a60dbd172672aec；首检查点 SHA256 为 2dd62abaf489217e0a83b8be9d59475f5271e9fe61f28b2d201526072634a67f。

本轮二维针对性测试 **18/18 通过**；源码编译检查通过。测试覆盖两种补能动作、目标不预览、manifest 拒绝混用与检查点恢复等价。尚未运行 reference_minimum 的完整轨迹，也未启动地图比较或神经训练。

需要继续该同一运行时的一行命令：

`cd /home/zjl/uav_learning_research && .venv/bin/python -m dual_constraint_2d.baseline_runner --output artifacts/dual_constraint_2d_fullmap_baseline_20260926/map000_full --map-id 0 --charge-mode full --max-new-decisions 20`

本次按仓库实验启动约束在首检查点后停止，没有继续监控至完成。
