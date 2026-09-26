# 1 km 候选配送场景参数

入口：[`scenario.py`](scenario.py)、[`sites.py`](sites.py)、[`charging.py`](charging.py)、[`energy.py`](energy.py)。本模块确定机型质量、电池标称容量、运动上限、确定性地图与服务点生成，并提供**明确标为假设**的部分充电曲线及参数化飞后 Wh 计量器；尚未冻结能耗参数，也未实现持续订单队列或能源安全决策。研究依据见[机型与尺度重估](../PLATFORM_SCALE_REASSESSMENT_20260923.md)和[能耗计量器说明](../ENERGY_ACCOUNTING_MODEL_20260925.md)。

- 地图 `1000 × 1000 × 100 m`；水平速度上限 `7 m/s`，竖直上限 `4 m/s`，水平／竖直加速度上限 `4/3 m/s²`。速度是控制上限，并非假设每个航段匀速飞完。
- M100＋一块 TB47D：机体含电池 `2.355 kg`、标称电量 `99.9 Wh`、最大起飞质量 `3.6 kg`，来自[DJI 产品规格](https://www.dji.com/matrice100)与[SDK 质量表](https://developer.dji.com/onboard-sdk/documentation/introduction/osdk-hardware-introduction.html)。设备增加 `0.50 kg`、货物 `0/0.25/0.50 kg` 是**候选研究设置**；最重总质量 `3.355 kg`。电池可用 SOC 与能耗模型尚未校准。
- `make_world(map_id)` 用独立种子生成 **24 根有限高、可从上方越过的圆柱**；障碍物高度分布在 `15–75 m`，低／中／高各 8 根。半径 `12.5–30 m` 是原 4 km 地图 `50–120 m` 的 1/4；**只缩放平面尺寸，没有把旧环境“贯穿全高”的障碍物保留**。圆柱互不重叠，离站点和边界有生成余量。身体半径 `0.4 m` 保持实际长度。这是**新分布**，不能称原 SAC 或冻结 v2 的已通过地图。
- 每张地图有**一座**地面充电站，XY 由地图 ID 的独立随机种子确定，边界至少留 100 m。`make_map(map_id).station_m` 返回其上方 2 m、满足 v2 规划净空的**飞行交接点**；不表示 UAV 已停靠或可自动补能。后续配送状态机须单独建模下降、停靠、显式充电和再次起飞。
- 订单的取货与送达点各自随机生成，取货点不充电。`sample_site(case, site_id, kind)` 可生成距地面或圆柱屋顶 2 m 的服务点，并核查从充电站有静态地图航路；它代表空中悬停取放的几何点，货物交接时间/能耗待建模。高层不能由此获得地图或特权路线。订单流、地面/屋顶混合比例和 deadline 仍须在评价前冻结。
- [`CHARGING_SENSITIVITY`](charging.py) 提供乐观、基准、保守三条**假设曲线**。三者都以 M100 TB47D [官方标配充电器 100 W 额定功率](https://www.dji.com/matrice100)为输入上限；转换效率、开始降功率的 SOC 和近满电功率均非实测。`seconds_to_soc` 显式返回从当前 SOC 充到目标 SOC 的时间；站内 idle 仍为零补电。
- 暂留 `800 s` 导航守卫，仅用于识别卡住的航段；它不是订单期限。生成时间、截止时间、到达率、充电功率、可用 SOC 与功率系数均尚未固定。

一行验证命令：`cd /home/zjl/uav_learning_research && .venv/bin/python -m unittest discover -s tests -p 'test_delivery_1km_*.py' -v`
