# 地图来源与高低层接口核查

UnderMind 检索下载 Ole16、Zho19、Koo18；正式年份分别 IROS 2017、IEEE RA-L 2019、ICLR 2019。三篇 PDF 已存 Zotero 分类 地图导航与学习调度（FTEZVEHZ），MD5 通过。

阅读覆盖：Voxblox §VII（PDF pp.7–8）立体相机+IMU，机载位姿估计、增量 TSDF/ESDF、4Hz更新与重规划，未知体素保守处理；Fast-Planner §VI–VII（PDF pp.6–8），VLP16+LOAM点云与定位、IMU融合，另一个实验为预建地图+OptiTrack，不能据此声称实际城市配送已验证；Attention Learn to Solve Routing Problems §3–4（PDF pp.2–5），节点特征、注意力编码、自回归节点选择、REINFORCE训练与rollout baseline。未逐页审核全文所有附录或证明。

作者 mav_voxblox_planning 仓库 README Download maps 提供真实场景扫描地图，包含 ESDF、骨架、稀疏图，并演示读取保存地图规划。https://github.com/ethz-asl/mav_voxblox_planning/blob/master/README.md

结论：离线传感器建图并保存、运行时定位/更新，是有技术依据的导航配置；不是从仿真器直接读取完美障碍真值。高层只接收规划代价可以作为系统接口假设，但这些文献不直接验证完整的能量、动态订单、时限、部分充电方案。代价特征仍包含地图衍生信息，必须更改并记录原 LiDAR-only 契约后才能实施，所有基线共享。

创新判断：已知地图导航、高低层分工、注意力选单都不应单独声称创新。本次为有针对性的代表文献核查，不是穷尽新颖性检索。候选问题为能耗/到达时间估计误差下的在线订单选择和部分补能；尚无现有方法失败证据，不启动实现或实验。
