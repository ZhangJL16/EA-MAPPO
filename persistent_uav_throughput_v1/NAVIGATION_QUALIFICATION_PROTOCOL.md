# Navigation qualification：v1.2 分析与停止规则

2026-09-20。本文件在本轮读取pilot的统计量、逐条outcome及error内容之前写定。
Pilot已启动，故这是对已有数据的pre-analysis protocol，不能称为采集前预注册。
不根据结果调整以下分组、停止规则、地图或采样支持集。

1. 原计划1000对x→y全部保留。软件错误不是navigation failure；单独说明恢复路径，不删除未完成pair。
2. 报告成功比例及双侧95% Wilson区间；时间/能耗q25、q50、q75、q90明确只针对成功任务。
   失败任务实际耗费另报，不能把censored cost当成功mission cost。
3. 固定分组：水平距离[0,1000)、[1000,2000)、[2000,3000)、[3000,+inf)；
   竖直绝对距离[0,60)、[60,180)、[180,+inf)。
   obstacle exposure采用直线投影与障碍圆柱的目标安全半径膨胀区域相交数量：0、1、>=2。
   该量仅为offline privileged geometry diagnostic，不输入actor或部署估计，也不等于实际路径碰撞概率。
4. 每组报告总数、成功数、失败原因、成功比例Wilson区间；不合并不利组。
   另报统一接触数、实际HOCBF fallback和失败任务消耗的pilot时间比例。
5. 明确暂停信号：整体导航失败率Wilson下界>5%；或任何n>=30的上述组失败率下界>10%；
   或失败任务消耗>=10%的全部pilot执行时间。最后一项是物理pilot中的浪费时间诊断，
   不是正式throughput loss的无偏估计。阈值仅是导航confound筛查，不是安全证书。
6. 出现任一信号：PAUSE_SCHEDULING，交由用户决定支持集/低层controller，不能自动缩图、挑seed或换controller。
   没出现信号：REVIEW_REQUIRED，仍须用户审阅；不自动授权B0–B5。
   如未完成1000条则INCOMPLETE，不做通过结论。
7. 不论筛查结果如何，都不能仅凭calibration断言navigation failure不影响policy ranking。
   后续正式评价须按policy分别报告N(T)、实际耗尽、导航失败、recharge failure、overflow、
   各模式耗时与失败后的剩余时间；保存逐事件记录，以便将这些描述性归因与真正反事实的
   scheduling loss区分。重叠原因不得相加伪造严格加性throughput gap decomposition。
8. B0–B3为共享25%SOC的task-ranking diagnostic；B4/B5才是首轮较强部署对手。
   MPC、小reference和OracleSafe诊断仍未授权。本轮不运行任何baseline。
