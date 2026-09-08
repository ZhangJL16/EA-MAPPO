# 既有研究资产复用审查：先保留方向信息，再比较安全算法

日期：2026-09-05。范围：代码、历史证据、无梯度诊断；未修改训练代码、旧 checkpoint 或启动新训练。

## 1. 本次改变了什么判断

此前设计说“借用现有已验证的 LiDAR 编码形式”，表述过强。现有测试验证了张量切分、卷积环绕等变性、反向传播和模型保存，却没有验证最终表示能否区分障碍物方位。本次代码审查发现：**卷积本身保留方位，但最终全局 mean/max 池化消除了绝对水平角信息**。这影响 R3，也影响调用同一提取器的 R7，不是 PPO 独有问题。

因此复用双通道观测、局部卷积和环绕处理，**不原样复用最终全局池化**。这是一项有明确机制证据的表示修正，不是再追加安全模块。它尚未证明可以提高闭环到达率，更不能解释历次失败的全部原因。

## 2. 实际编码合同及数学问题

实现：`experiments/jacobian_energy_bridge/features.py`。

- 输入 2055 维：速度 3、目标方向 3、归一化距离 1、1024 个距离、1024 个命中标记。
- 雷达按 `[2,8,128]` 排列，两个 3×3 卷积，通道 2→16→32。
- 水平角 circular padding，垂直方向 replicate padding。
- 目标/速度经 MLP 得到 32 维；雷达全局均值 32 维和全局最大值 32 维，拼成 96 维。
- 提取器参数量 6256。目标在池化后才融合，没有在空间特征图中提供方向锚点。

令 T_k 表示雷达在 128 个水平角上循环平移 k 格，C 为卷积映射，P 为全局均值/最大值池化。对于当前实现，在精确算术下：

\[
C(T_k L)=T_k C(L),\qquad P(T_k Z)=P(Z).
\]

因此对固定目标/速度特征 g：

\[
z(g,T_k L)=[G(g),P(C(T_k L))]=z(g,L).
\]

任何仅接收 z 的 actor 或价值头都无法恢复被消除的角度。随机策略也只能产生相同条件动作分布。若两个这样的状态需要不同最优动作，增加训练步数或换 PPO/SAC 不能让此表示同时区分它们。

限制：这不等于策略完全不使用雷达。它仍可按障碍物密度、近距离特征等减速；保守动作可能同时对多个方位安全。也不保证所有真实场景都成严格旋转配对。但它不能可靠地表达“相同目标与速度下，这个障碍物在前方还是侧方”这个区别。

仿真中的射线方向是固定世界坐标方向，见 `review_bundle/safety/collision/geometry3d.py:Lidar3DModel`；目标方向、速度和加速度也使用相应坐标。这里是**只旋转障碍物而固定目标与速度**，不是同时旋转全部状态的合理物理对称性。局部特征等变是合理的，过早将其变成不变标量表示则不适合此动作接口。

这是当前实现的直接代数性质，不作为新定理或论文创新声称。

## 3. 已执行的诊断

### 3.1 随机权重检查

固定 torch seed 20260905，4 个输入，循环位移 1、16、32、64 格。旧提取器最大差异 ≤2.99e-8；R7 随机 actor 高斯均值差异 ≤9.32e-10。此合成张量检查仅验证代数性质，不是环境成功率实验。

### 3.2 冻结 R3 + 真实射线几何

读取已训练 500k checkpoint，不修改权重，不做环境运动或梯度更新：

`artifacts/r3_sac_accelerated_reproduction_seed0_500k_formal500_20260902_v1/phase1_navigation/checkpoint_transition_500000.zip`

设置无人机位置 (2000,2000,200)，速度 (10,0,0)，目标 (2500,2000,200)。分别放置一个半径 50 的圆柱，其中心在无人机水平位置的 +x、+y、−x、−y 方向 70 米处。每次重新调用当前仿真 `_update_lidar()`，不是伪造角度标签。此为专门构造的诊断场景，不是原 24 障碍物任务分布的性能估计。

| 与 +x 场景比较 | +x | +y | −x | −y |
|---|---:|---:|---:|---:|
| 原始雷达最大绝对差 | 0 | 1 | 1 | 1 |
| 最终 96 维编码最大差 | 0 | 0 | 0 | 0 |
| 确定性 nominal action 最大差 | 0 | 5.96e-8 | 0 | 5.96e-8 |
| 同一卷积图改用有序 2×16 mean/max 读出的最大差 | 0 | 2.0341 | 2.0341 | 2.0341 |

+y 距离图与 +x 图循环平移 32 格的最大差为 0。动作的浮点末位差异不是可用的方向识别能力。此处是进入 HOCBF 之前的网络动作，不能与 shield 执行动作混淆。

来源 SHA-256：

- 提取器：`322049a88a2cf9dae3d21e249cbb57e6c5c6bb64340a4f99642d352ce4776c27`
- R3 checkpoint：`a9d270fc106db71f52c84486190d94d740b99cb770ff7d5b6f2326c78e4cb4fc`

### 3.3 为什么此前测试没有发现

`tests/test_jacobian_energy_bridge.py:test_structured_lidar_convolution_preserves_azimuth_wrap_equivariance` 检查的是池化前特征图，正确通过；并没有要求池化后 actor 能区分旋转后的障碍物。因此“已有测试通过”和“当前发现方位信息损失”不矛盾。

本轮重跑提取器/R7/首次接触旧分支相关四个测试文件：56 passed。该数字是现有实现回归通过，不是新方向读出已实现或学习有效。

另外重跑当前环境的向量化雷达等价、非终止碰撞、非终止边界、四子步累计、边界修复不制造推进能耗五项测试：5 passed（80 deselected）。合计 61 项现有测试通过；只有依赖库弃用警告。native QP 全套本轮未重跑，其先前等价证据见 `docs/R7_RUNTIME_ACCELERATION.md`，不将历史测试数冒充本轮结果。

## 4. 资产复用决定

| 资产 | 决定 | 复用范围与限制 |
|---|---|---|
| 128×8 距离+命中双通道 | 保留 | 不减原始射线；保留索引顺序、归一化、命中与零距离区别 |
| 环形水平/有界垂直卷积 | 保留形式 | 已有环绕等变及梯度测试；从随机权重训练，不锁定 R3 权重 |
| 全局 mean/max 读出 | 替换候选 | 当前方向不变性已直接验证；旧实现保留用于复现，不能原地改变历史 checkpoint 含义 |
| R7 actor/critic 隔离、tanh latent/logprob | 按接口复用 | 可复用分离参数和似然计算；不能继承其 PID、修正量成本和折扣成本语义 |
| R3 导航 reward 和基准权重 | 分开处理 | reward 分项作为经验起点；旧权重仅冻结参照，不等于自主安全策略或新策略教师 |
| 向量化射线/并行环境 | 保留 | 当前 `_update_lidar` 为 NumPy 几何批处理，不是 GPU 雷达；保留完整分辨率和四个物理子步 |
| C/native HOCBF QP 加速 | 保留可选路径 | 用于 shield-on 参照；shield-off 主训练不能把这个加速当作主要收益，也不能把 solver 等价当安全证明 |
| 碰撞/边界位置修复与速度归零 | 保留物理实现 | 新 wrapper 使用首次接触失败、接触优先，子步接触不可丢失；不整块复用旧 reach-avoid reward |
| 分层任务生成、哈希、每任务记录 | 保留机制 | 新开发/测试集分离，改为无接触到达/接触/超时互斥；正式集不重复调参 |
| 历史 gate 分类函数 | 不直接用作新决策 | `audit_navigation_gate_inventory.py` 仍含 .98/.95/1.10 等历史人为阈值，只能解释旧结果，不是定理或新部署要求 |
| R7 原子 checkpoint/优化器/RNG | 改造复用 | 当前未保存活动模拟器状态和半程任务；不能声称逐位续跑，需新合同版本与显式恢复语义 |
| forked-action 快照校验/配对数据/scene bootstrap | 保留研究工具 | 复用边界浮点容差、合法性拒绝、anchor hash、场景级拆分；全状态快照只用于离线诊断，不泄漏给 actor |
| 能量累计、失败原子、预算单调建模 | 留作后续能耗线 | 复用定义与测量工具；新导航策略改变 continuation policy 后，旧能耗标签不能直接当新策略标签 |
| 旧能耗 predictor/残差 critic/桥接损失 | 暂不接入 | 独立证据不足，或语义绑定冻结 R3+shield；不把全部历史模块堆到安全导航上 |

补充：`review_bundle/envs/navigation/lidar.py` 和 `envs/certified_uav/lidar.py` 还有 32 射线的旧导航接口，不是当前 1024 射线实现；不能因为文件名叫 lidar 就替换过来。

## 5. 能耗方向已有结果如何利用

不能只保留“LiDAR 帮过预测”的正面描述：

1. `artifacts/r3_resource_aware_encoder_gate_seed270001_20260903_v1/RESULT.json` 状态为 `DO_NOT_PROMOTE_RESOURCE_AWARE_ENCODER`；主要模型没有达到注册的相对几何基线 Brier/AUROC 条件。不能称为已验证的可迁移能耗编码器。
2. `artifacts/independent_lidar_action_confirmation_20260904_v1/RESULT.json` 的独立 75 场景结果：直接 LiDAR/action 模型 Brier 0.08526，几何/action 0.09132；但配对场景 Brier 差异 95% 区间 [-0.02311,0.01680] 跨零。联合危险误判率从 2.9844% 升到 4.0156%，差异区间 [-0.00651,0.03596]。登记判定为 `DO_NOT_MODIFY_R3_ACTOR`。

这不是证明 LiDAR 对能耗没有价值，而是说明平均预测改善不足以支持安全闭环接入。保留这些配对数据、评价方式和负结果；优先审计输入表示是否丢失“障碍物相对目标/动作的方向”。这可能同时影响绕行与动作条件能耗识别，但目前只是机制假设，未隔离验证。

这些标签对应八步宏动作后使用冻结 R3+HOCBF 的 continuation。更换导航策略后必须重采、重估或明确另一个 estimand，不能直接继承校准结论。

## 6. 下一版最小方案及实验顺序

首先修订表示合同，再比较优化算法，避免拿方向受限网络反复测试 PPO 是否有效。

1. **仅改读出。** 第一候选将同一卷积图的全局 1×1 mean/max 换成固定有序 2×16 网格读出，再接普通 MLP。它在本次诊断能区分四个方位；不增加规划器、规则动作或世界模型。原始 128×8 射线不变，但特征图仍有压缩，不能宣称保存了全部几何信息或已选择最优网格。
2. **实现前明确代价。** 该候选总特征从 96 变 2080，256 宽首层每个 trunk 多 507904 个参数；actor/critic 各一个 trunk 合计多 1015808。不能说改池化完全没有容量与耗时变化。若表示消融要归因于方向，需容量匹配控制；PPO-Lagrangian 与 FOCOPS 比较则必须使用完全相同网络。
3. **短诊断优先。** 增加池化后方向辨识、固定目标改变障碍物、固定雷达改变目标、梯度及保存恢复测试；测前向/反向和每任务采集耗时。有序读出“不同”不是“动作正确”，再用受控近障碍任务检查学到的方向响应。此处新增测试与正式实现尚未执行。
4. **再做导航学习与约束对照。** 按既定合同先普通 PPO 检查学习性，再匹配标准 PPO-Lagrangian / FOCOPS。首个约束方法不是定论，SAC-Lagrangian 仍作备选；不把两种算法叠在一起。相同首次物理接触成本与任务回报，shield-off 主评估，shield-on 辅助。
5. **最后接能耗。** 只有自主导航有稳定证据后，再重用能量/返航定义和场景配对工具建立策略对应的资源模型。避免让新的安全算法、方向表示和能耗损失同时改变后无法归因。

本轮没有新训练结果，不以诊断通过代替实验成功。旧提取器与历史模型保持原样。

## 7. 关键诊断复现

在项目根目录运行（无训练、无模型写入；仅建立内存中的诊断环境）：

```bash
uv run --no-project --python .venv/bin/python python - <<'PY'
import numpy as np
import torch
from stable_baselines3 import SAC
from torch.nn import functional as F
from envs.UAVEnergyDeliverySAC import (
    UAVEnergyDeliverySACEnv, SACTrainingPhase, StaticCylinderObstacle,
)
torch.set_num_threads(1)
path = ('artifacts/r3_sac_accelerated_reproduction_seed0_500k_formal500_20260902_v1/'
        'phase1_navigation/checkpoint_transition_500000.zip')
model = SAC.load(path, device='cpu')
env = UAVEnergyDeliverySACEnv(
    phase=SACTrainingPhase.NAVIGATION, lidar_enabled=True,
    lidar_horizontal_sectors=128, lidar_vertical_sectors=8,
    num_obstacles=0, cbf_enabled=False,
)
env.reset(seed=20260905)
env.agent.pos = np.array([2000.,2000.,200.], dtype=np.float32)
env.agent.vel = np.array([10.,0.,0.], dtype=np.float32)
goal = np.array([2500.,2000.,200.], dtype=np.float32)
rows = []
for center in ((2070.,2000.),(2000.,2070.),(1930.,2000.),(2000.,1930.)):
    env.obstacles = [StaticCylinderObstacle(np.array(center), 50.)]
    env.num_obstacles = 1
    env._update_lidar()
    rows.append(env.sac_observation_for_goal(goal))
x = torch.from_numpy(np.stack(rows))
with torch.no_grad():
    extractor = model.actor.features_extractor
    features = extractor(x)
    _, lidar = extractor.split_observation(x)
    maps = extractor.lidar_feature_map(lidar)
    ordered = torch.cat([
        F.adaptive_avg_pool2d(maps,(2,16)).flatten(1),
        F.adaptive_max_pool2d(maps,(2,16)).flatten(1),
    ], 1)
    actions = model.actor(x, deterministic=True)
    for name, values in [('input', x[:,7:]), ('features',features),
                         ('actions',actions), ('ordered',ordered)]:
        print(name, (values-values[:1]).abs().amax(1).tolist())
env.close()
PY
```
