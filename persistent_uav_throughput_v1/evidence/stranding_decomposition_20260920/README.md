# B5 Stranding Decomposition — 纯离线分析

**结论：预测误差确实存在，但不能把 B5 失败主要归结为单一的 estimator uncertainty。**
219 次飞行阶段 depletion 的互斥描述分组为：67 次接了已判不可行的 fallback 任务，
37 次任务能耗低估使预测余量翻转，40 次任务完成后等待耗电使预测余量翻转，
22 次返航出发预测仍可行但实际耗尽，53 次任务出发预测可行但任务自身耗尽。
实际终点相对目标点造成的返航估计变化没有单独解释本次翻转。

输入是已发布 B5 270 次运行的 JSON/event/decision telemetry；**新增 simulation = 0**，
未加载任何 pickle，未调用导航器、Oracle、policy 或模型训练，未修改 estimator、margin、环境或历史结果。
分析覆盖 3,945 条完成的服务飞行与 1,085 次实际返航决策，不只抽取失败样本。

## 1. 样本、定义与可复核等式

270 个 run 的结局：227 depletion、42 navigation failure、1 正常活到 horizon。
227 depletion 中 task=53、return=166、waiting=8；recharge failure 与 return depletion 重叠。

对每次返航耗尽，按事件索引匹配最后一条完成的 task 及随后返航。166/166 均匹配成功；
两者之间只有即时返航或 idle，没有另一条服务/补能飞行。所有预测使用原 frozen estimator
系数离线重算，并与已保存预测逐项比较。

定义（均为能量单位，正的 residual/shift 表示侵蚀余量）：

```
m_pre = e_before - Ehat_task - Ehat_return(goal)
E_task_actual = e_before - e_post
Delta_task = E_task_actual - Ehat_task
Delta_endpoint = Ehat_return(actual_endpoint) - Ehat_return(goal)
m_post = e_post - Ehat_return(actual_endpoint)

m_post = m_pre - Delta_task - Delta_endpoint

Delta_wait = e_post - e_return_departure
Delta_later_return_estimate = Ehat_return(departure_position) - Ehat_return(actual_endpoint)
m_departure = m_post - Delta_wait - Delta_later_return_estimate
```

166 条记录的第一条等式最大绝对误差为 `7.11e-15`，第二条为 `0`，检查容差 `1e-7`。
等待期间位置均未变化，`Delta_later_return_estimate=0`。
这是对已执行路径的会计分解，不是重新执行修正后的 policy 的反事实实验。
这里的“余量”始终是 **point-estimator margin**；其符号不是实际 recoverability 的证明。

## 2. 166 次返航 depletion 的互斥分组

| 类别 | 数量 | 占 166 次 | 直接证据 |
|---|---:|---:|---|
| 前一任务本就是 predicted-infeasible fallback | 67 | 40.4% | 出发时 m_pre≤0，`full_station_infeasible_estimate_fallback` |
| 任务能耗低估足以解释正→非正的 m_post | 37 | 22.3% | m_pre>0，m_pre−Delta_task≤0，m_post≤0 |
| 可行估计任务完成后，等待耗电造成余量翻转 | 40 | 24.1% | m_pre>0，m_post>0，等待后 m_departure≤0 |
| 返航出发仍是正预测余量，实际返航却耗尽 | 22 | 13.3% | m_departure>0，实际 depletion |
| **合计** | **166** | **100%** | |

分类优先标记 fallback，以免将本就被拒绝的预测说成 false-safe。
因此还有一个重叠统计：**42** 次在任务结束时 m_post>0，后来等待后变成非正，
其中 **2 次**属于前面的 fallback，剩余 **40 次**属于第三行。
这一点解释了为什么不能将 67、37、42、22 直接相加。

原有 `144` 次 nonpositive margin at return 的准确组成是 **67 + 37 + 40**。
所以不能把这 144 次统称为“前一个可行估计任务的能耗低估”。

### 2.1 任务能耗低估：37 条

37/166（22.3%）的返航耗尽符合任务能耗低估单独足以解释余量翻转；
若只看前一任务估计可行的 99 次返航耗尽，则为 37/99（37.4%）。

- 出发 m_pre 中位数：1.895。
- 任务能耗 residual 中位数：+6.903；范围 +0.268 至 +57.130。
- 终点导致的返航估计 shift 范围：−0.0601 至 +0.0655。
- 37 次都没有随后等待耗电。
- 这 37 次中，没有一例仅凭 endpoint shift 就足以翻转；若会计上令 task residual 为 0，
  全部 m_post 都仍为正（最小约 +0.00435）。这不等于现实中替换 estimator 必然成功。

在所有 166 次返航耗尽中，endpoint shift 范围仅 −0.0703 至 +0.0777。
本样本没有 endpoint-only 或 task+endpoint 联合作用才足够的翻转。
这只涉及该线性模型的终点位置输入变化，不能排除真实路径几何对实际返航能耗的影响。

### 2.2 等待机制：42 条重叠统计，40 条非 fallback 主分类

42 条记录在任务完成时仍有正的估计返航余量；等待耗电中位数 **23.403**，范围 **4.404–78.225**。
它们的前一任务能耗 residual 中位数为 **−1.197**，并非普遍低估。
已记录的空队列规则只在 SOC<0.25 时选择 recharge，否则 idle；站外 idle 保留悬停成本。
因此任务完成时的 reserve 并不会在 idle 中得到持续保护。

另有 **8 次直接 waiting depletion**；其中 7 次在开始 idle 时还具有正的预测返航余量。
这些记录提示空队列行为是单独的机制，不能被“预测任务能耗不准”覆盖。
本次没有修改该规则，也没有测量更改它会改善多少。

### 2.3 返航本身的 false-safe：22 条

返航实际出发时的预测余量为正，中位数 **2.636**，范围 **0.0866–11.7518**，但全部耗尽。
其前一服务任务 residual 中位数为 +1.185，范围 −1.508 至 +4.917；因此不依赖前一任务
出现巨大的 underprediction，实际返航本身也能违反 point prediction。
失败返航只能得到耗尽前已消耗能量与预测误差的下界，不能把它当作完整返航能耗。

### 2.4 Fallback：67 条，全部在 battery_tilde=2

前一任务出发时，B5 本来就没有把该任务判成 reserve-feasible。
满电站内不能再 recharge、非空队列不能 idle，现有规则选择估计 task+return 总能耗最小的任务。
这与“估计可行却失败”不同。

例如 regime=0、seed=920260920 的最后一条服务任务：m_pre=−24.161，
实际 task residual=−0.436（估计反而偏高），任务完成后 m_post=−23.694，随后返航耗尽。
这条轨迹不能被叙述为正 reserve 被 task underprediction 吞掉。
但 estimator 认为不可行不是所有候选在真实 simulator 中都不可行的证明；未做 Oracle branching。

## 3. 53 次 task depletion：保持 censoring 语义

53/53 在出发时都被判为 task+return feasible，0 次是 fallback。
飞行未完成，所以不存在可直接观测的完整 `E_task_actual`，也不存在该任务的 m_post；
这些字段不做插补。只记录耗尽前消耗的能量，作为完成该 leg 所需能量的下界。

`consumed_energy - predicted_task_energy` 下界：中位数 **34.452**，范围 **13.417–123.065**。
在预测可行的任务上，已经实际花掉的能量就远超任务预测，构成直接的 false-safe 证据。
不能把这 53 条删掉后，仅凭成功任务 residual 的均值声称模型准确。

## 4. 完整的 219 次飞行 depletion 与电池分层

| 互斥类别 | 总数 | battery_tilde=2 | =4 | =6 |
|---|---:|---:|---:|---:|
| Predicted-infeasible fallback 后返航耗尽 | 67 | 67 | 0 | 0 |
| 任务 residual 足以翻转，随后返航耗尽 | 37 | 6 | 20 | 11 |
| 任务完成后等待翻转，随后返航耗尽（非 fallback） | 40 | 0 | 23 | 17 |
| 返航预测 false-safe | 22 | 7 | 8 | 7 |
| 任务预测 false-safe | 53 | 9 | 21 | 23 |
| **飞行 depletion 小计** | **219** | **89** | **72** | **58** |
| 直接等待 depletion | 8 | 1 | 4 | 3 |
| **全部 depletion** | **227** | **90** | **76** | **61** |

112/219（51.1%）属于 task/return false-safe 或 task residual 足以解释翻转这三类；
其余 107/219 是 fallback 或后续等待主分类。**这个分组不是“112 条可由更好 estimator 挽救”
的因果估计**，机制可能重叠，修改 policy 后的状态分布也可能改变。
因此当前证据不支持“绝大多数失败都只是 estimator uncertainty”这一单因结论。

## 5. 预测负余量也不是实际不可恢复证明

全部 1,085 次实际返航的观测表：

| 出发时预测返航余量 | 成功到站 | 电量耗尽 |
|---|---:|---:|
| >0 | 880 | 22 |
| ≤0 | **39** | 144 |

所有 3,945 条完成的 task 中有 59 条从正 m_pre 变成非正 m_post；
下一动作均为立即 recharge，**22 次仍成功到站，37 次耗尽**。
因此“预测 margin≤0”不能直接写成“已经进入真实不可恢复区域”。
这也说明不能将记录中的 estimated feasibility 当成 Oracle ground truth。
本表只覆盖 B5 实际选择的返航状态，不代表所有候选任务或所有 policy 的风险。

## 6. 成功任务 residual 的选择偏差

3,945 条已完成服务任务 residual：均值 −0.094，中位数 −0.648，q95=+3.988，最大 +57.130。
这些是 success-conditioned、on-policy、同 seed/不同 regime 可能重复的 leg 记录；
既不包括 53 条 depletion 的完整未知成本，也不包含 42 条 timeout 的完整未知成本。
不把 pooled legs 视为独立样本，不用它们直接构造全任务风险证书。

## 7. 逐条复核示例

数值保留三位小数；下表每行是同一条记录，各列可作加减，不能将组中位数混合做等式。

| 来源（archive member） | m_pre | Delta_task | Delta_endpoint | m_post | 等待耗电 | m_departure |
|---|---:|---:|---:|---:|---:|---:|
| worker_00/run_00010.json | 3.550 | 14.276 | −0.060 | −10.667 | 0 | −10.667 |
| worker_00/run_00011.json | 5.526 | −1.625 | −0.020 | 7.172 | 29.559 | −22.387 |
| worker_01/run_00002.json | 4.596 | 1.190 | −0.046 | 3.452 | 0 | 3.452 |
| worker_00/run_00000.json | −24.161 | −0.436 | −0.031 | −23.694 | 0 | −23.694 |

全部 269 条提前失败的记录在 `failure_cases.json`，包含 source member、regime、seed、
任务出发/完成/返航/失败的事件索引，可回溯上一提交的原始归档。
`starting_battery` 指失败 leg 的起始电量；返航分解另用 `e_before` 指前一 task 的起始电量。

## 8. 复现与检查

```bash
python3 persistent_uav_throughput_v1/scripts/analyze_b5_stranding.py \
  --evidence persistent_uav_throughput_v1/evidence/b5_validation_complete_20260920 \
  --frozen persistent_uav_throughput_v1/evidence/v1_2/calibration/frozen_regimes.json \
  --output /tmp/b5_stranding_fresh_output

python3 -m unittest discover -s persistent_uav_throughput_v1/tests -p test_stranding.py -v
```

输出目录必须不存在。纯标准库；归档 checkpoint 只读取字节验证哈希，不反序列化。
已验证发布包 SHA256、全部 672 个归档成员、270 个预定 job、raw/summary 一致、
全部候选能耗重算、3,945 条完成任务耗能与日志一致、旧 failure audit 计数一致、分解等式一致。
7 项针对性测试覆盖 task/endpoint/joint 翻转、等待、fallback 重叠、return false-safe、
失败 leg censoring 和 goal-radius 边界。

文件：`summary.json` 汇总；`failure_cases.json` 269 条失败分解；
`completed_task_legs.json.gz` 3,945 条完成服务飞行；`return_legs.json.gz` 1,085 次返航；
`input_integrity.json` 输入/分析器哈希与检查；`SHA256SUMS` 本目录交付文件校验和。

**本轮没有启动新 policy、Oracle-Feasibility、MPC、训练或 kill test。**
下一研究决策应同时考虑 estimator false-safe、fallback 暴露的可行性疑问与等待时的余量保护，
而不是把这个结果直接包装为 planning difficulty 或 estimator-only story。
