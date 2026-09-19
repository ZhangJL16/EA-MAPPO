# 原主线关闭；Pivot A 理论推导首版

**用户已确认 KILL。原 high-level regenerative learning 主线停止。**
不训练 PPO / MAPPO / average-reward AC / Transformer，不增加任务分布、随机性或多智能体
来扩大 gap。不再采随机 UAV states。P0–P2 环境、冻结 SAC、全部原始证据保留。

## 已完成 P2-A2 的归档结论

| 指标 | 值 |
|---|---:|
| 完整树 | 36 cycle nodes + 10 candidate nodes |
| Oracle rho | 0.0108746710290 |
| VB rho | 0.0107805211448 |
| VB / Oracle | **0.991342277482** |
| 数值区间 | [0.991342276571, 0.991342278394] |
| 判定 | **KILL（冻结阈值 ≥0.98）** |
| 真正不同的 early-return physical state | **1** |
| 该状态的 optimal / VB cycle 访问质量 | 1/9 / 1/9 |
| candidate 与该 cycle state 的物理字段 | 完全相同，已核对 |

唯一差异状态前缀是 `100 → 300`。VB 在这里多做一个后缀，带来
1.4444444 个期望任务、148.2934031 秒额外时间，边际产出率0.00974045。
按1/9访问质量计，VB 每 cycle 多0.1604938任务、16.4770448秒。
这说明存在 opportunity-cost correction，但实际总收益差只有0.86577%。

完整结果、read-only structural summary、全部46个 node receipts 和 root 已归档到
`research/regenerative_control/evidence/vb_theory_20260919`。旧 startup evidence 保持不变。

## 本轮完成的理论工作

新问题：**When Is the Viability Boundary Optimal for Persistent Regenerative Control?**

主文件：[DERIVATION_PACKAGE.md](../DERIVATION_PACKAGE.md)。
状态：**COHERENT AFTER REFRAMING / EXTRA ASSUMPTION**。

1. **精确最优性：**任务的 reward/time/energy 标记 IID，可以异质且相互相关；
   在看到下一任务前决定是否执行，并且 recharge time 为固定开销加累计能耗的线性项时，
   VB 精确最优。证明使用可预测任务纳入的 Wald 恒等式及 VB 最大化任务数。
   不要求不同任务 reward/time 相同。
2. **非线性残差界：**保留 IID job marks，补能在 affine 部分上叠加有界非负残差时，
   给出以残差量级和 cycle 工作量衡量的近似最优界。
3. **一般有限树证书：**定义 VB 后缀相对立即返航的收益 A、额外时间 Delta，
   `G = A - rho_VB * Delta`。所有可达合法 C 节点 G≥0 是 VB 最优的充要条件。
   给出完整 stopping-prefix coupling 证明，不要求任务成本 IID。
4. **近似最优性：**VB 访问概率加权的负 G 总负荷控制整个 throughput gap。
   在现有36-node树上，只评估 VB 得到 **98.6185%** 的最优收益率下界；
   该计算没有读取 oracle rho/action，也没有调用优化 solver。
5. **Overhead 条件与解析例子：**推导固定转移树上的 overhead 阈值；
   给出 reward、time、energy 均不同且 recharge 为二次函数的解析模型。
   只是代数推导，没有改动或重跑 UAV。

## 必须保留的范围限制

UAV 的 task label IID **不等于** realized navigation cost IID。任务成本依赖上一任务
结束后的实际位置和速度，返航/停靠也有几何成本。因此精确 IID 定理是一个简化模型类
的结论，不能直接宣称它证明了 UAV 的 exact VB optimality。

UAV 采用一般树证书：唯一负 G 节点的 tail rate 为0.00974045，低于 rho_VB；
负荷为0.01713730，cycle时间下界113.48039秒，得到98.6185%的独立近似最优证书。
这强化了原 KILL，而不是恢复算法动机。

## Theorem kill-test 状态

数学上已超出“所有任务 reward/time 相同”的退化特例。
**新颖性仍未通过：**主要结论是 Wald / renewal / optimal-stopping 的直接应用或推论，
不能把一个短的 verification identity 包装成 Spotlight theorem。
限定范围的标准文献来源列于推导包；本轮没有做穷尽 prior-art search。
当前可以保留这一理论问题，但不能宣称已形成一篇新论文或批准更多实验。

## 核验与复现

- 本轮新 UAV runs：**0**；新训练更新：**0**。
- 只运行已有结果的报告脚本及新的标准库代数脚本。
- 核对已有 result 的 KILL 数值区间、candidate physical duplication、
  唯一负 tail 到全局 gap 的恒等式。
- 异质、非线性解析例子使用有理数核对阈值两侧和等号，不是 simulator sweep。
- 原 P2-A2 的7项测试结果保留；本轮未重复运行导航测试或采样实验。

```bash
python research/regenerative_control/theory_certificate.py \
  --input artifacts/regenerative_p2a2_20260919 \
  --output research/regenerative_control/evidence/vb_theory_20260919/certificate.json
```

没有自动启动下一研究阶段。新理论是否值得继续，取决于后续明确的新颖性和可扩展性判断，
不能用增加环境复杂度来替代。
