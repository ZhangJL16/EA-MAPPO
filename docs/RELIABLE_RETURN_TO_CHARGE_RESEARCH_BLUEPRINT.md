# Reliable Return-to-Charge：整体研究方案与证据路线

> 更新时间：2026-08-27  
> 文档性质：统一研究蓝图、理论义务与实验 Gate 说明  
> 当前结论状态：研究路线已收束；导航修复 R1 的正式评估仍在进行；能耗、Oracle 返航价值和后续方法结论均为 `PENDING`

## 1. 一句话研究问题

资源有限的自主智能体应该在什么时候停止继续执行任务，并不可逆地返回充电站？

本研究的机器学习难点不是单纯预测能耗，而是：

> 当导航策略与独立安全过滤器共同决定真实执行轨迹，并可能发生策略、过滤器或环境分布变化时，如何估计并判断当前返航 Resource-to-Go 是否可信，从而尽可能晚地返航，同时避免因返航过晚而耗尽电量？

最终优化目标是：

\[
\boxed{
\text{在相同低 stranding rate 下完成更多任务，或在相同任务吞吐下获得更低 stranding rate}
}
\]

因此论文的主结果应是 **stranding--throughput Pareto frontier**，而不是 Energy-to-Go 的 MAE 单独下降。

---

## 2. 故事背景

### 2.1 实际任务

无人机持续执行随机任务流：

```text
TASK 1 -> TASK 2 -> TASK 3 -> ...
```

电池有限时，每个决策时刻必须选择：

```text
CONTINUE TASK
或
CHARGER_COMMITTED
```

一旦进入 `CHARGER_COMMITTED`，返航决定在当前 battery cycle 内不可撤销：

```text
TASK -> CHARGER_COMMITTED -> CHARGER -> recharge -> TASK
```

过早返航会浪费可用电量并降低任务吞吐；过晚返航会导致 energy exhaustion 和 stranding。

### 2.2 为什么固定 SOC 阈值不够

相同剩余电量下，返航难度可能完全不同：

- 充电站距离不同；
- UAV 当前速度和方向不同；
- 障碍物造成的绕行不同；
- HOCBF 可能触发制动、投影和恢复；
- 导航策略更新后，实际 occupancy 和路径效率不同。

固定 `SOC < 30% -> RETURN` 可能在简单状态下过度保守，在困难状态下又过晚。

### 2.3 为什么仅使用距离也不够

简单模型

\[
\widehat E_C=d_C\bar e_{\mathrm{meter}}
\]

忽略了速度、加速度、障碍绕行、安全干预、策略变化和路径历史所决定的未来执行轨迹。

### 2.4 为什么普通 Energy predictor 仍可能失效

系统真正执行的链条是：

\[
\pi
\rightarrow a^{\mathrm{nom}}
\rightarrow \Pi
\rightarrow u^{\mathrm{exec}}
\rightarrow x'
\rightarrow c
\rightarrow Z_C.
\]

其中：

- \(\pi\) 是导航策略；
- \(\Pi\) 是 HOCBF 或其他安全算子；
- \(a^{\mathrm{nom}}\) 是 SAC 原始动作；
- \(u^{\mathrm{exec}}\) 是安全过滤后的真实动作；
- \(c\) 是 TelemetryCostModel 给出的真实逐步资源消耗；
- \(Z_C\) 是到达充电站之前的累计 Resource-to-Go。

如果只在旧策略、旧安全算子或旧 occupancy 上训练 predictor，模型可能在新组合上看起来置信度很高，但已经 stale。

---

## 3. 论文定位

### 3.1 顶层任务

**Reliable Return-to-Charge / Return-to-Replenishment Decision**。

### 3.2 核心 ML 问题

**Long-horizon Resource-to-Go prediction and reliability under compositional closed-loop shift**。

### 3.3 应用实例

主实例是带静态障碍物、LiDAR、SAC 和 HOCBF 的 UAV continuous delivery。

### 3.4 不应采用的弱定位

论文不应被描述成：

- 一个更聪明的 SOC 阈值；
- 一个 UAV 电池工程系统；
- 给 Energy Critic 拼接几个 HOCBF feature；
- 证明 HOCBF 一定增加能耗；
- 用神经网络替代 HOCBF 安全证书。

### 3.5 期望的完整科学链

\[
(\pi,\Pi)\text{ shift}
\Rightarrow
\text{executed-interface extrapolation}
\Rightarrow
\text{Resource-to-Go error}
\Rightarrow
\text{return-boundary error}
\Rightarrow
\text{stranding or premature return}.
\]

计划验证的两条经验规律是：

1. **Prediction law**：pair novelty 本身不是主要失败变量，executed-interface extrapolation 与 hitting horizon 才控制长期 Resource-to-Go 误差。
2. **Decision law**：真正造成任务失败的是 return boundary 附近的 Resource-to-Go 低估，而不是所有状态上的平均预测误差。

---

## 4. 固定系统结构

```text
Navigation policy
    -> nominal action a_nom

Hard safety operator HOCBF
    -> executed action u_exec

Physical dynamics + TelemetryCostModel
    -> next state x' + realized energy c

Resource-to-Go estimator
    -> required return energy estimate

Reliability mechanism, if justified
    -> epistemic margin / abstention / adaptation

ReturnManager
    -> CONTINUE or absorbing CHARGER_COMMITTED
```

固定职责如下：

- SAC 学习目标导航和基于 LiDAR 的避障倾向；
- HOCBF 始终是最终碰撞安全层；
- Energy 模块预测真实 executed closed loop 的资源需求；
- ReturnManager 决定何时返航；
- learned Jacobian、Energy predictor 和 reliability score 都不是安全证书；
- 到达 charger 后充满电并继续任务；
- energy exhaustion 是真实失败 terminal；
- charger reached 结束 battery cycle，但不终止 continuous mission。

---

## 5. 当前环境与概率语义

### 5.1 物理环境

- 地图：4000 m × 4000 m × 400 m；
- 单 UAV；
- 当前正式导航环境：24 个静态障碍物；
- 障碍半径：50--120 m；
- LiDAR：128 方位角 × 8 垂直层，range 与 valid mask；
- policy dt：0.2 s；
- physics dt：0.05 s；
- 水平最大速度：20 m/s；
- 垂直最大速度：5 m/s；
- 水平最大加速度：5 m/s²；
- 垂直最大加速度：3 m/s²；
- 能量单位：`synthetic_simulation_energy_units`。

### 5.2 当前正式预测对象

固定部署信息、策略、安全算子、障碍物和初始状态后，当前模拟器没有独立未来随机扰动过程。现有 clone-rollout audit 得到：

\[
\operatorname{Var}(Z_C\mid I_t,\pi,\Pi)=0.
\]

因此当前正式对象是：

\[
\boxed{
\text{deterministic point Resource-to-Go}
+
\text{epistemic prediction reliability}
}
\]

当前不能把模型输出宽度称为真实物理 q95/q99 tail。

未来只有在加入具有物理依据、时间相关并可复现的风、执行器误差、部分可观测状态或动态障碍过程后，且 repeated fixed-condition clone rollouts 显示非退化条件分布，才升级到 stochastic Resource-to-Go。

### 5.3 统一保守需求量

确定性环境下：

\[
U_C(x)=\widehat E_C(x)+\Delta_{\mathrm{epi}}(x).
\]

若未来 stochastic probability Gate 通过：

\[
U_C(x)=\widehat q_{1-\delta}(Z_C\mid x)+\Delta_{\mathrm{epi}}(x).
\]

统一返航规则为：

\[
\boxed{
b_t\le U_C(x_t)+m
\Longrightarrow
\textsf{CHARGER\_COMMITTED}
}
\]

其中 \(b_t\) 是剩余能量，\(m\) 是 reserve。

---

## 6. 所提出的方法框架

整个系统暂称 **Reliable Return-to-Charge (RRC)**。预测与可靠性子模块可暂称 **Selective Interface Resource Prediction (SIRP)**，但 SIRP 只有在普通 executed-action ensemble 不够时才允许实现。

### 6.1 Navigation：结构化 LiDAR SAC

SAC observation 保持 2055D 外部 contract：

```text
7D goal/velocity features
+ 1024 LiDAR ranges
+ 1024 LiDAR valid mask
```

策略内部不再使用 flat MLP 直接处理所有输入，而是：

```text
7D local goal/velocity -> goal MLP
2 × 8 × 128 LiDAR -> circular azimuth 2D convolution
feature fusion -> SAC actor/critic
```

该结构利用 LiDAR 的局部邻接与方位角环形拓扑。

### 6.2 Hard Safety：HOCBF

每个物理子步：

```text
a_nom -> HOCBF projection -> u_exec -> dynamics
```

训练 reward 包含小的 intervention penalty，使 SAC 有动力减少依赖安全投影；但零碰撞仍属于 `SAC + HOCBF` executed system，不能宣称纯 SAC 已获得安全证明。

### 6.3 Realized Energy

每个 physics substep 使用实际推进后的速度与 realized acceleration：

\[
c_t=mathrm{TelemetryCostModel}(v_t,a_t^{\mathrm{real}},\Delta t).
\]

边界或碰撞 projection 产生的人工速度跳变不计为 propulsion acceleration。

### 6.4 Resource-to-Go Estimator

目标是预测：

\[
T_C=\inf\{t\ge0:x_t\in G_C\},
\qquad
Z_C=\sum_{t=0}^{T_C-1}c_t.
\]

候选估计器按复杂度逐级增加：

1. distance × average energy per meter；
2. direct supervised Monte-Carlo point ETG；
3. frozen/online TD ETG；
4. PCM-Executed；
5. executed-action probabilistic world model；
6. executed-action world model + Deep Ensemble；
7. SIRP，仅当第 6 项不能充分识别长期失败时。

### 6.5 Reliability Mechanism

如果 ensemble 不够，SIRP 不使用任意 OOD score，而估计 cross-fitted local predictive error：

\[
\widehat\epsilon_K(x,u),
\]

再沿目标 composition 的 rollout occupancy 聚合：

\[
\widehat R_H
=
\sum_{t=0}^{H-1}
\mathbb E_{\widehat d_t^{\pi,\Pi}}
[\widehat\epsilon_K(x_t,u_t)]
+\widehat\epsilon_{\mathrm{trunc}}.
\]

该值用于：

- 低风险：正常给出 ETG；
- 中风险：扩大 epistemic margin；
- 高风险：abstain 或请求少量 target data 后 adaptation。

### 6.6 ReturnManager

所有估计器共用相同高层决策接口，防止把 predictor 与 threshold policy 混淆。

当前已实现：

- `FixedSOCThresholdReturnManager`；
- `DistanceEnergyReturnManager`；
- `QuantileEnergyReturnManager`，在当前确定性环境中应解释为 learned ETG + margin manager。

正式比较还需要：

- Oracle ReturnManager；
- direct learned switching classifier；
- executed-WM/ensemble/SIRP 接入相同 threshold rule；
- 独立 end-to-end CMDP track。

---

## 7. 研究阶段与 Fail-Closed Gate

### Stage 0：导航能力修复

当前 R0--R4 设计：

| ID | LiDAR encoder | Jacobian bridge | 目的 |
| --- | --- | --- | --- |
| R0 | flat MLP | 原始 bridge | 已归档失败基线 |
| R1 | structured LiDAR | OFF | 判断表征是否足够 |
| R2 | structured LiDAR | 原始 uniform replay bridge | 判断 Jacobian 是否提供额外价值 |
| R3 | structured LiDAR | recency bridge | 判断 policy drift 是否是问题 |
| R4 | structured LiDAR | noise-coupled recency bridge | 判断 stochastic action mismatch 是否是问题 |

每个候选训练 500,000 environment transitions，并接受相同 500-task 固定评估。

Navigation Gate：

- overall success ≥ 98%；
- 每个距离 bucket success ≥ 95%；
- mean path ratio ≤ 1.10；
- boundary-contact step rate < 1%；
- obstacle-collision steps = 0。

没有候选通过时，停止 downstream Energy/return 实验。

### Stage 1：冻结导航策略

通过导航 Gate 后：

- 冻结 SAC；
- 固定 HOCBF 配置；
- 不用 energy reward 继续改变导航；
- 保存精确 checkpoint、SHA、环境配置与评估任务集合。

### Stage 2：Battery Calibration

使用冻结导航策略完成 500 个分层单任务，只累计 TelemetryCostModel realized energy：

\[
\bar P
=
\frac{\sum_i E_i}{\sum_i T_i}.
\]

目标 endurance 为 30 分钟时：

\[
B_{\mathrm{cal}}
=
\bar P\times1800\ \mathrm{s}.
\]

同时报告 20/30/40 分钟候选容量。Calibration 不使用 TD prediction，不训练 SAC，不写 TD replay。

### Stage 3：Held-out Battery Validation

使用独立 seed 连续执行 TASK 直到 depletion，检查：

- mean depletion time；
- tasks before depletion；
- distance before depletion；
- observed endurance 相对 30 分钟目标误差。

工程 sanity interval 为 ±20%。失败时默认停止后续正式实验。

### Stage 4：Oracle Decision Headroom Gate

在训练复杂 ETG 模型前，Oracle 从同一 simulator snapshot 比较两条未来：

```text
RETURN NOW
FINISH CURRENT TASK -> service stop -> RETURN
```

Oracle 与 SOC threshold、distance heuristic 比较 stranding--throughput frontier。

预注册 Gate：

- common stranding ceiling：0.05；
- 使用 Wilson 95% upper bound；
- 每个 frontier 点至少 100 个独立 battery cycles；
- Oracle 在相同安全约束下，相对最佳 eligible heuristic 的 throughput 至少提高 5%。

若 Oracle 没有足够 headroom，停止 ICLR return-to-charge 方法线，不训练 PCM、world model 或 SIRP。

### Stage 5：Frozen-policy Energy-to-Go 学习

使用冻结后的合格导航策略重新采集数据：

```text
random start
-> random goal / charger goal
-> frozen SAC
-> HOCBF
-> realized energy trajectory
```

当前 TD baseline 使用 exactly 500,000 collection transitions。Phase 1 早期、导航尚未成熟的数据不得混入正式 TD replay。

### Stage 6：连续配送返航决策

在相同 frozen policy、HOCBF、battery capacity 和任务流下比较：

1. fixed SOC；
2. distance-energy heuristic；
3. direct switching classifier；
4. Frozen-TD ETG；
5. Online-TD ETG；
6. Oracle；
7. 后续 PCM/executed-WM/ensemble/SIRP。

所有 predictor-based 方法使用同一 ReturnManager 规则和 reserve sweep。

### Stage 7：Closed-loop Composition Shift

只有 Oracle Gate 通过且简单 ETG baseline 有决策价值后，才运行：

\[
2\ \text{policies}
\times
2\ \text{safety operators}
\]

完整 leave-one-pair-out。

必须独立控制：

- pair novelty；
- executed-interface extrapolation；
- hitting horizon。

主实验应使用 high/medium/low interface extrapolation × matched horizon 的 factorial design，不能把距离、障碍难度和 support 混在一起。

### Stage 8：第二 Domain / 第二 Safety Family

ICLR oral-level generality 至少需要：

- 第二 dynamics/resource domain，例如 ground robot charging；
- 与 HOCBF 不同的安全算子 family，例如 MPSC、MPC shield 或 reachability shield；
- 多个 policies；
- 相同 pair-OOD/interface-ID/interface-OOD 三阶段规律复现。

### Stage 9：End-to-End CMDP Track

CPO、Sauté、SDAC 等允许重新训练完整 constrained policy，训练权限与 frozen-policy modular track 不同。

因此报告两条独立赛道：

- **Modular/frozen-policy track**：固定 \(\pi,\Pi\)，只比较 return mechanism；
- **End-to-end constrained-control track**：允许 CMDP 方法重训策略。

不得把两者混为完全相同的训练条件。

---

## 8. 理论目标与当前证明状态

### 8.1 基本假设

主要理论在以下条件下成立：

1. 状态、动作和资源空间是标准 Borel 空间；
2. predictor 的输入是部署时可获得的充分信息状态；
3. 目标 policy \(\pi\) 与 safety operator \(\Pi\) 已知或可查询；
4. plant/resource primitive 通过 executed action 接口
   \(K(dx',dc\mid x,u)\) 分解；
5. composition 比较中 primitive kernel 共享；
6. charger hitting problem 是 proper SSP，且 \(\mathbb E[T_C]<\infty\)；
7. target executed-interface occupancy 上 primitive 可识别；
8. transport theorem 额外需要 one-step coupling 和 continuation-law regularity；
9. energy-risk implication 额外需要 deployed calibration 与 decision-interval overshoot bound。

### Theorem 1：Executed-Interface Compositional Identifiability

**命题。** 即使 target pair \((\pi^\star,\Pi^\star)\) 从未作为完整轨迹生成组合出现，只要 policy 和 filter 可查询、shared primitive 在 target executed occupancy 上可识别、并满足 proper SSP 条件，则 target closed-loop path law 与 charger Resource-to-Go law 唯一可识别。

核心结论：

\[
\boxed{
\text{pair-level trajectory overlap is not necessary}
}
\]

但 component 分别出现过不等于 target interface 已覆盖。

**当前状态：** 在补充 measurable-space、proper-SSP 和 target-interface identification 假设后可证明；现有 proof package 已给出路径测度与截断极限证明。

### Theorem 2：Non-Identification Outside Interface Support

**命题。** 如果 target composition 以正概率访问 source 数据完全未覆盖的 interface region \(B\)，则可以构造两个 primitive kernels：它们在全部 source 数据上 observationally equivalent，但在 \(B\) 上产生不同资源消耗，因此 target Resource-to-Go law 不同。

结论：

\[
\boxed{
\text{interface-OOD is a fundamental identification obstruction}
}
\]

除非额外引入已知物理参数化、平滑性或安全 target probing，否则不能普遍 zero-shot 识别。

**当前状态：** 在明确模型类后可证明；已有构造性反例。

### Theorem 3：Finite-Horizon Distributional Transport

在 one-step primitive coupling 与 continuation-law Lipschitz 条件下：

\[
W_1(\mathcal Z_{0:H},\widehat{\mathcal Z}_{0:H})
\le
\sum_{t=0}^{H-1}
\mathbb E_{d_t^{\pi,\Pi,K}}
[\epsilon_t(x_t,u_t)].
\]

该定理说明长期误差由 target occupancy 下的局部 executed-interface error 累积，而不是由“pair 是否见过”这一标签直接决定。

它也为 reliability statistic 提供结构依据。

**当前状态：** 只能在显式 coupling 和 future-return regularity 假设下证明；coverage 本身不足以推出该 bound。HOCBF active-set 不连续可能破坏全局 Lipschitz 条件，需要 piecewise 或 total-variation 扩展。

### Theorem 4：Proper-SSP Truncation Bound

若 per-step resource 有界且 \(T_C\) 一阶矩有限：

\[
W_1(\mathcal Z_C,\mathcal Z_{C,H})
\le
c_{\max}\mathbb E[(T_C-H)_+].
\]

这把长期误差分成：

```text
interface/model error
+ hitting-time tail / truncation error
```

**当前状态：** 可证明。

### Theorem 5：Return-Boundary Decision Stability

定义 oracle 与 learned decision：

\[
d^\star=\mathbf 1\{b\le U+m\},
\qquad
\widehat d=\mathbf 1\{b\le\widehat U+m\}.
\]

若

\[
|\widehat U-U|\le\varepsilon,
\]

则：

\[
d^\star\ne\widehat d
\Longrightarrow
|b-m-U|\le\varepsilon.
\]

也就是说，prediction error 只有在 return boundary 的 \(\varepsilon\)-band 内才能改变 threshold decision。

**研究意义：** 全局 MAE 不是最关键的任务指标；boundary-weighted underestimation 才直接关联 late return 和 stranding。

**当前状态：** 精确阈值命题，可证明。

### Corollary 1：Conditional Return-Failure Bound

若 accepted prediction 满足：

\[
\Pr(Z_C\le U_t\mid\text{accepted})\ge1-\delta,
\]

且 reserve 覆盖相邻检查之间的电量消耗和 requirement drift：

\[
m\ge\Delta_{\mathrm{check}},
\]

则在排除非能量失败模式后，commit 后到达 charger 前 energy exhaustion 的条件概率至多为 \(\delta\)。

**当前状态：** 需要 deployed-composition calibration 与实测 overshoot bound；当前 telemetry 已实现，但正式数据尚未产生。该结论不是无条件安全证书。

### Corollary 2：Task-Then-Return Risk Allocation

若 task 与后续 return 位于同一 joint mission probability space，且：

\[
\Pr(E_T\le U_T)\ge1-\delta_T,
\qquad
\Pr(E_R\le U_R)\ge1-\delta_R,
\]

则无需独立性：

\[
\Pr(E_T+E_R\le U_T+U_R)
\ge1-\delta_T-\delta_R.
\]

因此：

\[
q_{.95}(E_T)+q_{.95}(E_R)
\]

不能被称为 joint mission q95。两个 95% component bounds 通过 union bound 只能给出声明的 90% 下界。要获得整体 95%，应直接预测 joint mission，或分配例如 2.5% + 2.5% 风险预算。

**当前状态：** 可证明。

### 当前明确不能证明的命题

- 小 \(W_1\) 自动推出 q95/q99 稳定；
- 任意 closed-loop shift 下 conformal coverage 保持不变；
- safety intervention 总是增加能耗；
- ensemble disagreement 等于真实 interface support；
- learned ETG 是 hard energy-safety certificate；
- 当前 deterministic simulator 存在真实 aleatoric q95；
- 单 UAV + 单 HOCBF 结果足以支持一般 compositional claim。

---

## 9. 实验方案与 Claim 对齐

| 科学问题 | 实验 | 主要对比 | 成功标准 |
| --- | --- | --- | --- |
| 导航是否足够可靠 | R1--R4 500k + 500 tasks | R0 flat/JSEB | 通过固定 navigation Gate |
| 完美 ETG 是否有决策价值 | Oracle headroom | SOC、distance | 同一 stranding ceiling 下 throughput ≥5% 提升 |
| TD 是否能逼近返航资源 | Frozen/Online TD | distance、MC direct | ETG error 与 decision utility 同时改善 |
| pair-OOD 是否本质困难 | unseen pair / interface-ID | direct、PCM、executed-WM | executed-interface model 仍可 zero-shot |
| interface extrapolation 是否控制误差 | matched support × horizon | ensemble | error 随 interface risk 和 horizon 系统变化 |
| reliability 是否能识别失败 | risk--coverage | ensemble threshold、simple kNN | AURC 与 severe underestimation 优于强基线 |
| 误差是否真正影响任务 | common ReturnManager | SOC、distance、direct switcher、CMDP | Pareto frontier 左上移动 |
| 是否可泛化 | 第二 domain/filter | 同一 baseline suite | 三个 regime 重复出现 |

### 9.1 Prediction / Mechanism Table

必要 baseline：

1. distance × average energy；
2. MC-supervised direct ETG；
3. TD ETG；
4. PCM-Nominal 与 PCM-Executed；
5. generic executed-action probabilistic WM；
6. executed-WM + bootstrap Deep Ensemble；
7. proposed SIRP，若 justified；
8. simulator Oracle。

所有 model-based 方法必须得到公平信息：若 proposed 可查询目标 \(\pi\) 和 \(\Pi\)，其他相应 baseline 也必须可查询；若 proposed 看见 \(u^{exec}\)，executed-WM 和 PCM-Executed 也必须看见。

### 9.2 Decision Table

必要决策对比：

1. SOC 20/30/40% thresholds；
2. distance-energy return；
3. direct high-level continue/return switcher；
4. direct ETG return；
5. PCM return；
6. executed-WM return；
7. ensemble-conservative return；
8. SIRP/RRC；
9. Oracle；
10. 独立 CMDP track。

### 9.3 Headline Metrics

任务层：

- stranding / energy-exhaustion rate；
- tasks per battery cycle；
- tasks per simulated hour；
- charger return success；
- unused energy at charger；
- premature-return rate；
- paired stranding--throughput Pareto frontier。

机制层：

- ETG MAE、RMSE、bias；
- underestimation rate 与 magnitude；
- boundary-weighted underestimation；
- horizon-stratified error；
- risk--coverage 与 AURC；
- stochastic Gate 通过后才使用 CRPS、q90/q95 pinball 和 coverage。

统计层：

- 每个 frontier 点至少 100 independent battery cycles；
- Wilson confidence interval 处理 rare-event stranding；
- paired cycle-level bootstrap；
- navigation candidate 至少 3 seeds 才能形成研究 claim；
- test composition 不允许调 threshold 或超参数。

---

## 10. 与已有实验的关系

### 10.1 旧 77D / 71D / 7D 路线

- 旧 77D frozen SAC 保留为历史兼容 checkpoint；
- 71D multi-scale navigation 是已归档失败实验；
- 7D 4×4 relative-goal SAC 研究局部尺度迁移，但不是当前 4 km 障碍物主实验；
- 这些结果不能直接用于当前 obstacle-aware return claim。

### 10.2 UAV Energy Delivery V3

V3 提供了当前仍然复用的基础：

- 单 UAV continuous task stream；
- TelemetryCostModel；
- battery calibration；
- TASK / CHARGER_COMMITTED 生命周期；
- Goal-conditioned Energy TD；
- charging、trajectory logging 和 rendering。

当前工作不是推翻 V3，而是在其上加入 obstacle-aware navigation、HOCBF、严格 Gate 和 return-decision research framing。

### 10.3 旧 100k obstacle-aware Energy runs

这些实验使用中间 100k navigation checkpoint，并明确绕过 navigation readiness Gate，只能作为 exploratory control。

它们可以用于发现：

- frozen-policy Energy 数据流程是否可运行；
- stale critic、calibration data 和 Phase 2 pipeline 的实现问题。

但不能证明：

- 导航已合格；
- Energy estimator 已正式有效；
- 能够可靠返航；
- Jacobian 比原方法优越。

### 10.4 旧 JSEB 500k

旧 R0 正式结果：

- overall success：0.84；
- minimum bucket success：0.82；
- mean path ratio：1.63125；
- obstacle collision steps：0；
- boundary safety：通过。

结论是“执行安全但导航未 ready”，因此 downstream calibration/Oracle/TD 被正确停止。

### 10.5 当前 R1--R4

当前比较只改变结构化 LiDAR 与 Jacobian bridge 机制，HOCBF、障碍物、reward、physics 和评估任务保持一致。

因此：

- R1 相对 R0 的改善主要检验 structured representation；
- R2--R4 才检验 Jacobian 是否有独立贡献；
- HOCBF 零碰撞不能被计为 actor 自主学习成功；
- 必须报告 nominal-safe action、intervention 和 emergency rates。

---

## 11. 当前证据快照

截至本文档写作时：

- Stage A pluggable ReturnManager：已实现，targeted regression 通过；
- mission component risk semantics：已修正；
- deterministic probability-semantics audit：完成；
- Oracle clone runner：已实现，smoke 通过，但无正式 headroom 结果；
- decision-interval overshoot telemetry：已实现，正式数据待产生；
- 理论 proof package：已有五个 theorem/proposition 与两个 corollary 草案，但独立证明审查尚未完成；
- R1 structured-LiDAR navigation：500k checkpoint 已生成；
- R1 500-task formal evaluation：运行中，尚无最终 Gate 结论；
- R2--R4：等待 R1 队列；
- formal battery calibration：等待 navigation Gate；
- formal Oracle headroom：等待 calibration；
- formal TD 与 return frontier：等待 Oracle Gate；
- PCM/executed-WM/ensemble/SIRP：Gate 前禁止启动；
- 第二 domain/filter 与 CMDP：待后续 pilot 证明方向值得继续。

任何训练中 success、smoke、GIF 或 `RUNNING.json` 都不是正式研究结论。

---

## 12. Go / No-Go 决策规则

### Gate N：Navigation

- 失败：停止 Energy/return 正式链，继续修复导航；
- 通过：冻结选中 checkpoint。

### Gate B：Battery Calibration

- calibration success <98% 或 endurance sanity 失败：停止并诊断；
- 通过：进入 Oracle。

### Gate O：Oracle Headroom

- Oracle 不比简单 heuristic 提供 ≥5% eligible throughput headroom：停止 ICLR return 方法线；
- Oracle 明显更好：证明 ETG information 有决策价值。

### Gate P：Prediction Pilot

- executed-WM 不能在 pair-OOD/interface-ID transfer：检查 Markov/interface 假设；
- interface extrapolation 与 matched-horizon error 无稳定关系：停止 compositional oral 核心；
- ensemble 已与 proposed 一样识别 failure：删除复杂 SIRP，保留理论/benchmark 路线；
- SIRP 稳定改善 AURC、boundary underestimation 和 adaptation sample efficiency：继续完整方法线。

### Gate G：Generality

- 第二 domain/filter 不复现规律：降级为 robotics/autonomous-systems claim；
- 跨 domain/filter 稳定复现：具备一般 ML claim 的证据形态。

---

## 13. ICLR Oral-Level 证据门槛

无法保证 oral，但要达到 oral-candidate evidence standard，至少需要：

1. 跨系统发现稳定新规律：pair novelty 不是主要失败变量，interface extrapolation × horizon 才是；
2. 证明 prediction failure 在 return boundary 附近转化为 stranding/premature return；
3. proposed 超过最危险简单基线 `executed-action WM + Deep Ensemble`；
4. 在相同 stranding 下显著提高 throughput，或在相同 throughput 下显著降低 stranding；
5. theory 与 experiment 一一对应，而不是 appendix 装饰；
6. 至少两种 dynamics/resource domains、两类真正不同的 safety operators、多 policies；
7. 提供可复用的 `Policy × Safety Operator × Interface Coverage × Horizon` benchmark；
8. 公平比较 direct switcher、PCM、executed-WM、ensemble 和 CMDP；
9. 三 seeds 以上、完整置信区间和预注册 stop rules；
10. 诚实保留 negative result：如果简单方法足够，删除复杂模块而不是制造增益。

---

## 14. 下一步执行顺序

严格按以下顺序推进：

1. 完成 R1 500-task evaluation；
2. 顺序完成 R2--R4 seed-0 candidate-killing comparison；
3. 对仍有希望的候选补至少 3 seeds；
4. 冻结通过 Navigation Gate 的最终策略；
5. 正式 500-task battery calibration；
6. 正式 100-run held-out battery validation；
7. 正式 Oracle headroom Pareto test；
8. Oracle PASS 后采集 exactly 500k Energy TD transitions；
9. 完成 SOC / distance / Frozen-TD / Online-TD / Oracle decision frontier；
10. 决策价值成立后实现最小 prediction pilot：MC direct、PCM-Executed、executed-WM、ensemble；
11. 只有 ensemble 不够时实现 SIRP；
12. 完成 2-policy × 2-filter leave-one-pair-out；
13. 增加第二 domain 和第二 safety family；
14. 完成 CMDP end-to-end track；
15. 独立 proof audit、artifact audit、claim audit 后再形成论文结论。

---

## 15. 权威配套文档

- `docs/RETURN_TO_CHARGE_ICLR_RESEARCH_PROTOCOL.md`：正式研究与实验协议；
- `docs/RETURN_TO_CHARGE_COMPLETION_LEDGER.md`：逐项完成状态；
- `docs/RETURN_TO_CHARGE_DERIVATION_PACKAGE.md`：统一推导路线；
- `docs/RETURN_TO_CHARGE_PROOF_PACKAGE.md`：当前 theorem/proof 草案；
- `docs/INTERFACE_SUPPORT_DERIVATION_PACKAGE.md`：executed-interface identifiability 推导；
- `docs/JSEB_NAVIGATION_GATE_REPAIR_PROTOCOL.md`：R0--R4 导航修复协议；
- `docs/JSEB_500K_NAVIGATION_GATE_RESULT.md`：旧 R0 正式失败结果；
- `docs/UAV_ENERGY_DELIVERY_V3.md`：V3 环境与两阶段能耗基础。

本文件负责把这些内容组织成单一论文故事，不替代各文档中的精确代码合同、证明细节或实验 artifact。
