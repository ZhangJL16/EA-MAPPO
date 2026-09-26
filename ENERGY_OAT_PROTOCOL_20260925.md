# 1 km 飞后能耗假设敏感性：执行协议

日期：2026-09-25，首次读取这 19 组批量结果前固定。用户已选择继续沿用轻量纯软件环境，并确认[19 组基准+逐项端点](ENERGY_PARAMETER_PROPOSAL_20260925.md)。DJI 官方 M100 飞控模拟器需要实机，当前不可作为环境。模型、来源与边界见[能耗计量器说明](ENERGY_ACCOUNTING_MODEL_20260925.md)。

输入固定为已经完成的[96 条 1 km 导航资格航迹](artifacts/delivery_1km_nav_v1_20260923/manifest.json)。对每条轨迹分别以载荷 `0/0.25/0.50 kg` 计算 19 组模型功率，共 `96×3×19=5472` 条**模型标签**；它们不是 5472 次独立飞行。每条输出积分 Wh、最大瞬时功率、最大所需推力/倾角、假设推力或厂家倾角/升降速度违反步数和首违反步。跨参数比较时，轨迹、地图、起终点和低层控制完全相同。可用比例在生成标签时同时作用于厂家悬停锚点功率与电池可用容量；当前只输出飞行 Wh，不作实际 SOC 判定。

主分析对每种载荷分别报告：基准飞行 Wh 分布；各个单参数改变相对基准的配对 Wh 差；任何轨迹变为不可执行的比例；变化是否会使后续“能否完成取货→送达→回站”判断反转。最后一项此轮还不能计算，因为订单链与飞前估计尚未建立。全体 96 条已被用于导航资格，故**不能**用此轮误差当作未来飞前能耗预测器的未见地图测试结果。Rodrigues 数据只作外部数量级和载荷趋势核对，不直接拟合主模型。

程序在 `artifacts/delivery_1km_energy_oat_20260925/` 写入源码哈希、参数值、导航 manifest 与逐航迹 SHA、逐条结果和可恢复 checkpoint。首轮仅运行到 2 条航段检查点，检查标签可读、源哈希与参数匹配，再交还用户，不自动续跑全体或开始订单实验。没有任何神经训练。

执行补记（2026-09-25）：上述首轮检查点已经健康通过；随后用户明确批准按提案完整运行 19 组模型内诊断，因此同一批次从 checkpoint 续跑至 96/96。完整性复核及配对统计见[结果报告](ENERGY_OAT_RESULT_20260925.md)和[机器可读摘要](artifacts/delivery_1km_energy_oat_20260925/summary.json)。没有启动订单实验或神经训练。

一行启动命令：`cd /home/zjl/uav_learning_research && .venv/bin/python -m delivery_1km.energy_audit --output artifacts/delivery_1km_energy_oat_20260925 --stop-after-checkpoint`

一行续跑命令：`cd /home/zjl/uav_learning_research && .venv/bin/python -m delivery_1km.energy_audit --output artifacts/delivery_1km_energy_oat_20260925 --resume --stop-after-checkpoint`
