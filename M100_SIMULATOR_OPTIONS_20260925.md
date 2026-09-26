# M100 仿真环境可用性核查

日期：2026-09-25。此页只比较平台能力，不改变当前导航资格结果，也不擅自迁移实验。

| 路线 | 已核实能力 | 对当前研究的限制 |
|---|---|---|
| DJI 官方 M100 飞控模拟器 + Assistant 2 / SDK | DJI 官方文档明确 M100 可运行模拟器；模拟器位于机载飞控，Assistant 2 经 USB 连接 M100，SDK 能发送控制并读取模拟状态。 | 需要 M100 实机/飞控；官方资料没有提供可直接修改飞控动力学、载货耗电、订单队列和复杂障碍地图的仿真源码。适合作硬件在环控制验证，尚不是现成的批量配送研究环境。 |
| 社区 M100 ROS/Gazebo 模型 | 可编辑 ROS/Gazebo 世界、模型与传感器。 | 核对的两个 M100 实现均明确要求连接真实 M100 和 DJI PC/SDK 模拟器，属硬件在环桥接；不能直接作为无实机的现成纯软件 M100。 |
| PX4 SITL + Gazebo 自定义模型 | 官方 PX4 文档支持无实机软件在环、自定义模型/世界与传感器。 | 默认机型不是 DJI M100；若要称“M100 参数化模型”，需自行设置并核验机体、推进、电池和控制接口。迁移会改变低层动力学，必须另开资格测试。 |

用户已确认**没有 M100 实机**，因此官方 M100 飞控模拟器目前不可直接使用。平台路线已选定：先沿用通过 96/96 导航资格的轻量纯软件环境，完成能耗与订单诊断；之后把 PX4/Gazebo 作为**单独的交叉验证**，届时自建 M100 参数化模型并重新核查低层。当前未迁移仿真平台，也不能称现有模型为 DJI 官方仿真。

来源：[DJI SDK 模拟器说明](https://developer.dji.com/mobile-sdk/documentation/application-development-workflow/workflow-testing.html)、[DJI OSDK M100 硬件连接说明](https://developer.dji.com/cn/onboard-sdk/documentation/development-workflow/hardware-setup.html)、[hku_m100_gazebo](https://github.com/caochao39/hku_m100_gazebo)、[dji_ros_simulator](https://github.com/TareqAlqutami/dji_ros_simulator)、[PX4 Gazebo SITL 官方文档](https://docs.px4.io/main/en/simulation/)。
