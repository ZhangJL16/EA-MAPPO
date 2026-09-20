# Calibration 完成与导航资格报告

## Material Passport

2026-09-20；1000/1000完成；源码15fe98e；验证状态VERIFIED（本地执行及完整性核验）。
资格状态：**REVIEW_REQUIRED**，不是批准baseline。neural updates=0，baseline runs=0。

## 主要结果

导航成功988/1000 = **98.8%**，Wilson 95%区间 **97.91%–99.31%**。
失败12次均为navigation timeout；统一接触总数1。
成功任务的条件统计如下，能耗是synthetic units，不是Wh：

| 指标 | q25 | q50 / median | q75 | q90 |
|---|---:|---:|---:|---:|
| 时间（秒） | 69.5875 | 109.0500 | 147.5125 | 181.1200 |
| 能耗 | 14.6838 | 22.9262 | 30.8417 | 38.2620 |

失败任务耗时占全部pilot时间 **7.990%**。
其中timeout后曾记录接触：1次；没有记录接触：11次。
这是共现分类，不能据此断言timeout由接触造成。
横向距离、竖向距离、离线障碍暴露分组及各组区间见
[NAVIGATION_QUALIFICATION.md](evidence/v1_2/calibration/NAVIGATION_QUALIFICATION.md)和qualification.json。

## 参数与资格

m_T = 109.05000000秒；m_E = 22.92621828。
27个regime全部按原定3×3×3比例生成并逐项核验，无按算法结果筛选。
实际manifest的T为10905.05000000秒（按存储的浮点median向上对齐物理网格）。
完整参数见[frozen_regimes.json](evidence/v1_2/calibration/frozen_regimes.json)。

未触发整体/分组失败率或失败耗时比例的暂停信号；仍须用户审阅才能启动B0–B5。
本结果只覆盖这张地图上的预定随机x→y分布，不能证明station-return或scheduler诱导的
长期轨迹同样可靠，也不能证明navigation failure不会影响policy ranking。
障碍暴露为privileged直线段几何诊断，不是实际路径LiDAR暴露，更不是部署输入。

## 修复与证据完整性

原运行因固定地图起点被旧地图生成保护距离误拒而中断，属于1次软件初始化错误，
不计入12次navigation timeout。修复后沿用同一地图与1000对输入；原完成的93条结果
逐项相等，1000个任务ID唯一且完整，详见integrity.json及migration.json。

修复后22项检查通过；真实serve→serve→recharge→serve保持位置、电量和时间连续，
仅初始化reset一次；飞行中恢复后的完整事件轨迹、结果与终态相同。
源码、原16项检查和原始中断证据均已进入当前仓库。
复现入口与资产哈希见[REPRODUCIBILITY.md](REPRODUCIBILITY.md)。

## 统计解释核查（11/11）

| 风险 | 核查与处理 |
|---|---|
| Simpson聚合反转 | 同时报告边际分组，不把边际关系解释成独立效应 |
| 生态谬误 | 不从单地图推断其他地图或所有实际航线 |
| Berkson选择偏差 | 成功条件分位数明确标注，不代表无条件总代价 |
| Collider条件化 | 不用成功筛选后的T/E关系作因果结论 |
| 忽略基率 | 成功率以全部1000条为分母 |
| 均值回归 | 不选择极端样本或换seed后声称改善 |
| 幸存者偏差 | 12次失败原始记录保留，失败耗时单列 |
| 多重搜索 | 固定分组全报；区间为pointwise，不是同时置信保证 |
| 分析选择自由度 | 规则是pilot已启动后的pre-analysis，不冒称事前预注册 |
| 相关与因果 | 分组差异是描述性关系，不作因果归因 |
| 反向因果 | 几何诊断由预定输入定义，不用结果反选输入 |

没有新98% kill-test结论；没有运行B0–B5、MPC、OracleSafe或small reference。
