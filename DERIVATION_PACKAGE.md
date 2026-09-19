# Derivation Package

## Target

**Pivot A：When Is the Viability Boundary Optimal for Persistent Regenerative Control?**

目标是给出一个允许任务奖励、时间、耗能变化的模型类，在其中证明 VB 的最优性；
再给出偏离该模型时的近似最优证书，解释已完成 UAV 树中的唯一 early-return state。
这是推导包，不是论文或已确认的新定理贡献。

原算法主线已按预注册标准 **KILL**：VB/Oracle = 0.9913422775 ≥ 0.98。
该结论不因本推导而改变。不训练 high-level RL，不新增 UAV 实验，不更换任务分布。

## Status

**COHERENT AFTER REFRAMING / EXTRA ASSUMPTION**。

- 数学目标成立：下面给出异质 IID job + affine recharge 的精确结果、非线性残差界、
  一般有限树的 VB 证书及固定 overhead 条件，并附证明。
- 必须区分 **IID task labels** 和 **IID realized job costs**。UAV 只有前者；
  不能直接把精确 IID 定理套到 UAV。UAV 使用后面的 state-dependent tree 证书。
- 新颖性未确立。核心工具是经典 Wald / renewal / optimal-stopping；
  本文不将这些工具的应用命名为新的 community-level finding。

## Invariant Object

所有部分围绕同一目标：

\[
\rho_\pi=\frac{\mathbb E W_\pi}{\mathbb E D_\pi},
\qquad J_\rho(\pi)=\mathbb E[W_\pi-\rho D_\pi],
\]

其中 \(W_\pi\) 是完整 recharge-completion→recharge-completion cycle 的任务收益，
\(D_\pi>0\) 是对应总时间。不是平均单个任务的 reward/time，也不是 finite-episode return。

## Assumptions

**共同假设 G（所有结论）：**

1. 补能完成到同一 canonical H；未来任务过程重新独立，使用相同的 cycle policy。
   每个 cycle 有强制首任务，之后只可 C 或 R，没有任务筛选、等待或其他任务选择。
2. C 在抽取下一个任务之前决定。任务结果出现后，才进入下一个 D-state。
3. 使用已冻结的 robust immediate-return safe set：所有下一任务都能完成并立即 R，
   C 才合法；否则只能 R。R 在所有合法可达 D-state 均安全。
   这里的 VB 是该指定 safe set 的边界，不声称是一般控制系统的最大 viability kernel。
4. 所有合法 cycle 有统一有限任务数上界；本包正式范围采用有界任务收益、任务时间和补能时间，
   期望 cycle 时间有正下界。仅假设可积的推广未在这里证明。
   资源耗费有正下界及容量有限是充分实现方式。仅“逐次耗费 >0”本身不保证统一有限上界。
5. 状态包含影响未来的全部信息；在有限树中直接使用任务前缀和真实物理状态。
   策略可根据已知过去做决定，但不能看未来任务/RNG。收益非负且 cycle 期望收益为正。

**附加假设 I（仅 Proposition 1–2 的简化模型）：**

- 任务标记 \(X_i=(r_i,\tau_i,a_i)\) 跨任务 IID；\(a_i\) 为资源耗费，
  \(0<a_{\min}\le a_i\le a_{\max}<\infty\)。三分量可相关，\(r_i/\tau_i\) 可变化。
- 给定任务标记后，其任务耗时/耗能不依赖此前位置、速度或资源余量。
- R 所需固定资源 reserve 已从工作容量中扣除；返回没有额外的位置依赖资源限制。
  首任务在满资源 H 总可安全执行。VB 在剩余工作资源至少为 \(a_{\max}\) 时继续。
- 基本补能时间为 \(\beta+\gamma\sum_{i=1}^{N_\pi}a_i\)，\(\beta,\gamma\ge0\)。
  固定返回能耗的线性补能时间也可吸收到 \(\beta\) 中。
  \(c=\mu_\tau+\gamma\mu_a>0\)，其中 \(\mu_X=\mathbb E X_i\)。

**附加假设 F（一般树的 overhead corollary）：**

\(R_\beta(s)=\beta+g(s)\)，且改变 \(\beta\) 不改变任务转移、收益、安全集合或 H。
例如不存在 deadline、时间变动 arrival、overhead 期间额外耗尽等效应。
这里只研究这一数学参数族，不实际修改 UAV 环境。

## Notation

| 符号 | 定义 |
|---|---|
| \(B\) | VB policy：C 合法就继续，否则 R |
| \(N_\pi,n_\pi\) | cycle 任务数及其期望 |
| \(R(s)\) | 当前立即 R 到 H 的时间；**不是 reward** |
| \(A_B(s)\) | 从 D-state s 按 VB 到 H 的剩余期望任务收益 |
| \(U_B(s)\) | 同一 VB 后缀的剩余期望总时间，包含最终 R |
| \(\Delta_B(s)\) | \(U_B(s)-R(s)\)，相对立即 R 的额外时间 |
| \(G_B(s)\) | \(A_B(s)-\rho_B\Delta_B(s)\)，VB 后缀 excess |
| \(p_B(s)\) | 一轮 VB cycle 访问前缀 s 的概率 |
| \(\bar\delta(s)\) | \(\mathbb E[\tau(s,J)+R(s'_J)]-R(s)\) |
| \(\mathcal D_B\) | \(\sum_s p_B(s)[-G_B(s)]_+\)，VB 后缀缺口负荷 |

上述 \(A_B,U_B,G_B\) 都是 **单个固定策略 VB** 的评估量，不需要先求 \(\rho^*\)。
边界节点 VB 直接 R，因此 \(A_B=\Delta_B=G_B=0\)。

## Derivation Strategy

先用“不预知下一任务”推出可预测纳入指标的期望求和恒等式，证明一个足够广的精确模型类。
然后让补能包含非线性残差，控制该残差能产生的最优性差距。
最后回到一般 state-dependent finite tree，用停止前缀耦合推出可由 VB 自身验证的证书，
不把 UAV 的任务标签 IID 错认成成本 IID。

## Derivation Map

1. 共同目标：renewal ratio → excess reward。
2. I 假设下：可预测任务纳入 → 三个 Wald 恒等式 → 固定开销摊薄 → VB exact optimum。
3. 增加非负补能残差 → affine envelope → VB near-optimality bound。
4. 一般树：任一策略是 VB 路径的停止前缀 → VB tail excess identity → 最优性充要条件。
5. tail 证书 → additive overhead threshold / 稀少负 tail 的误差界。
6. 在已有 36-node 数据上做确定性代数核对；不采样、不运行 simulator。

## Main Derivation

### Step 1 — Identity：可预测纳入，而不是任务收益率相同

令 \(I_i=\mathbf1\{N_\pi\ge i\}\)。是否执行第 i 个任务，在看到 \(X_i\) 前已经决定，
所以 \(I_i\) 对过去可测，与 \(X_i\) 独立。统一有限上界保证可交换有限求和与期望：

\[
\mathbb E\sum_{i=1}^{N_\pi}r_i=\mu_r n_\pi,\quad
\mathbb E\sum_{i=1}^{N_\pi}\tau_i=\mu_\tau n_\pi,\quad
\mathbb E\sum_{i=1}^{N_\pi}a_i=\mu_a n_\pi. \tag{1}
\]

证明以 reward 为例：\(\sum_i\mathbb E[I_i r_i]
=\sum_i\Pr(N_\pi\ge i)\mu_r=\mu_r\mathbb E N_\pi\)。
分量间相关不影响该论证；\(N_\pi\) 可以依赖所有已完成任务的标记。
这是 Wald 型恒等式的有限版本，**不是新定理**。
标准工具参考：[Gallager / MIT 的 renewal rewards 与 Wald 教学资料](https://ocw.mit.edu/courses/6-262-discrete-stochastic-processes-spring-2011/resources/lecture-12-renewal-rewards-stopping-trials-and-walds-inequality/)。

### Step 2 — Proposition 1：异质 IID jobs + affine recharge 下 VB 最优

在 G+I 下：

\[
\rho_\pi
=\frac{\mu_r n_\pi}{(\mu_\tau+\gamma\mu_a)n_\pi+\beta}
=\frac{\mu_r}{c+\beta/n_\pi}. \tag{2}
\]

**Claim：**\(\rho_B=\rho^*\)，允许 \(r,\tau,a\) 都变化并相关。

**Proof：**在同一无限 IID 标记序列上耦合任意合法策略与 VB。在前者第一次 R 前，
二者做完全相同的任务。VB 不会在 C 合法时提前停，故逐路径 \(N_\pi\le N_B\)，
从而 \(n_\pi\le n_B\)。式 (2) 随 \(n_\pi\) 单调不减，得结论。
若 \(\beta>0\)，期望任务数严格减少会严格降低收益率；若 \(\beta=0\)，所有合法
cycle policy 的收益率均为 \(\mu_r/c\)，VB 是最优策略之一。□

**Interpretation：**不预知未来任务使 reward、work time 和耗能的期望都按任务数缩放；
控制只剩下摊薄每次补能的固定开销。因此任务异质性本身不必产生 early-return 价值。
这一结论比“各任务 reward/time 相同”强，但证明是标准恒等式的短推论，不自动具备论文新颖性。

### Step 3 — Proposition 2：非线性补能残差的近似最优界

保留 IID job marks 和相同 safe set，改为

\[
\tau_R=\beta+\gamma\sum_{i=1}^{N_\pi}a_i+u(S_{N_\pi}),
\qquad 0\le u(s)\le U<\infty. \tag{3}
\]

残差可依赖终态/历史，允许非线性 recharge；它不能改变前面的 job mark law 或合法动作。
令 \(\bar u_B=\mathbb E_B u(S_{N_B})\)。则

\[
\boxed{\frac{\rho_B}{\rho^*}
\ge\frac{c n_B+\beta}{c n_B+\beta+\bar u_B}
\ge\frac{c n_B+\beta}{c n_B+\beta+U}.} \tag{4}
\]

**Proof：**对任一策略，\(u\ge0\) 给出
\(\rho_\pi\le\mu_r/(c+\beta/n_\pi)
\le\mu_r/(c+\beta/n_B)\)。VB 自己的实际收益率为
\(\mu_r/[c+(\beta+\bar u_B)/n_B]\)。相除得到第一界，再用 \(\bar u_B\le U\)。□

因而 \(U\le\epsilon(c n_B+\beta)/(1-\epsilon)\) 是 VB 达到
\((1-\epsilon)\)-optimal 的充分条件。无需求 optimal controller。
例如容量限制 \(d\le E\)，\(g(d)=\gamma d+\eta d^2\)、\(\eta\ge0\) 时，
可取 \(U=\eta E^2\)。这是全模型类的解析界，不是新 UAV 参数实验。

**限制：**若残差可为负、返航可绕过资源限制、或任务执行成本随位置改变，不能直接使用式 (4)。
给一个巨大 U 虽然形式上有效，也可能完全无信息；需要报告界的紧度。

### Step 4 — Identity：一般 state-dependent stopping 的局部 correction

回到 G 的一般树。令 \(F_\rho(s)\) 为从 D 到 H 的最优 excess，
\(K_\rho(s)=F_\rho(s)+\rho R(s)\ge0\)，因为 R 总是合法。
代入 Bellman 方程得到，在 C 合法时：

\[
Q_\rho(s,C)-Q_\rho(s,R)
=\bar r(s)-\rho\bar\delta(s)+\mathbb E K_\rho(s'_J). \tag{5}
\]

在 \(\rho=\rho^*\) 下，该式直接回答 C 何时优于 R：右侧 ≥0。
最后一项表示完成下一任务后仍能选择最佳停止时机的价值。
**此式是标准 optimal-stopping 改写，不是独立 novelty，也不是下面证书的前提。**
相关标准框架：[Ferguson 的 Markov optimal-stopping 讲义](https://www.math.ucla.edu/~tom/Stopping/sr4.pdf)。

### Step 5 — Proposition 3：只评估 VB 的精确最优性证书

在有限 prefix tree 中：

\[
\boxed{B\text{ 从 H 最优}\iff G_B(s)\ge0
\quad\text{对每个 VB 以正概率到达的合法 C 节点。}} \tag{6}
\]

**Proof（先证恒等式）：**沿同一 VB 任务序列耦合任意合法策略 \(\pi\)，
记它选择 R 的前缀为 \(S_{\sigma_\pi}\)。此前两策略完全一致。
条件于该前缀，VB 比 \(\pi\) 多拿的期望 reward 为 \(A_B(s)\)，多花的期望时间为
\(\Delta_B(s)\)。利用 \(J_{\rho_B}(B)=0\)，得

\[
J_{\rho_B}(\pi)=-\mathbb E_\pi G_B(S_{\sigma_\pi}). \tag{7}
\]

若全部 G 非负，任一策略的 excess 都 ≤0，即 \(\rho_\pi\le\rho_B\)。
反之，若某个正概率可达 s 的 G<0，构造“按 VB 运行，但到 s 即 R”的合法策略，
其它路径仍按 VB。其 excess 为 \(-p_B(s)G_B(s)>0\)，故优于 VB。□

该条件允许任意 state-dependent、异质 job costs 和 reward。代价是需要知道整个
VB 可达模型并评估 tail；它不是不依赖 transition knowledge 的 deployment rule。

### Step 6 — Corollary：固定 overhead 下的精确阈值

增加 F，并假设每个可达合法 C 节点 \(A_B(s)>0,\Delta_B(s)>0\)。定义

\[
\lambda_{\min}=\min_s\frac{A_B(s)}{\Delta_B(s)},\quad
\bar W_B=\mathbb E W_B,\quad D_B^0=\mathbb E D_B(\beta)-\beta.
\]

其中 \(\bar W_B\) 是期望 cycle reward。
VB 的树不随 overhead 改变；\(A_B\) 不变，而 U 和 R 各增加同一个 \(\beta\)，
故 \(\Delta_B\) 也不变。式 (6) 等价于
\(\bar W_B/(D_B^0+\beta)\le\lambda_{\min}\)，所以

\[
\boxed{B\text{ 最优}\iff\beta\ge\beta_{\rm crit}
=\max\{0,\bar W_B/\lambda_{\min}-D_B^0\}.} \tag{8}
\]

这不是所有“recharge 越慢 VB 越好”的普遍定理。只改固定 overhead 且不影响状态转移
是必要范围限制。若改变 recharge rate、capacity 或 deadlines，树、\(\Delta_B\) 或
safe set 也可能改变，不能沿用同一个阈值。若 \(\Delta_B\le0\)，用原始 G 证书，
不能盲目除以增量时间。

### Step 7 — Proposition 4：稀少负 tail 如何控制整体 optimality gap

取任意可证明的 \(T_{\min}>0\)，满足所有合法策略 \(\mathbb E D_\pi\ge T_{\min}\)。
在有限树上定义 \(\mathcal D_B=\sum_s p_B(s)[-G_B(s)]_+\)。则

\[
0\le\rho^*-\rho_B\le\frac{\mathcal D_B}{T_{\min}},\qquad
\boxed{\frac{\rho_B}{\rho^*}\ge
\left(1+\frac{\mathcal D_B}{\rho_B T_{\min}}\right)^{-1}.} \tag{9}
\]

**Proof：**由式 (7)，\(J_{\rho_B}(\pi)\le
\mathbb E[-G_B(S_{\sigma_\pi})]_+\)。在同一 VB 路径上，\(\pi\) 停在 s 的概率
不超过 VB 访问 s 的概率，因此该期望 ≤\(\mathcal D_B\)。
再除以 \(\mathbb E D_\pi\ge T_{\min}\)，取最优策略，得到绝对界；代数整理得相对界。□

这是保守上界：一轮只在一个前缀停止，而求和可能计入同一路径上的多个负 tail。
若只有一个负 tail，负荷就是 \(p_B(s)[-G_B(s)]_+\)：小 gap 可来自小缺口、低访问概率，
或二者兼有；不需要宣称所有可行 C 都严格最优。

**Lemma（无需 optimal controller 的时间下界）：**如果所有合法 C 节点
\(\bar\delta(s)\ge0\)，可取强制首任务后立即 R 的期望时间作为 \(T_{\min}\)。

**Proof：**对任一停止前缀，telescoping 给出
\(D_\pi=\tau_0+R(S_0)+\sum_{k<\sigma_\pi}
[\tau_k+R(S_{k+1})-R(S_k)]\)。继续指示量在抽下一任务前可测，故取期望后每项
为非负访问质量乘 \(\bar\delta(s)\)。因此最短期望时间就是立即 R 策略达到的下界。□

### Step 8 — Analytical example：奖励、时间、耗能均不同，且 recharge 非线性

**这只是解析例子，不是新 UAV 环境或实验。**工作容量3，任务 IID，各概率1/2：

\[
(r,\tau,a)=(1,1,1)\quad\text{或}\quad(3,4,2).
\]

robust C 要求剩余资源 ≥2，且 \(\tau_R=\beta+d^2/2\)，d 为累计耗费。
VB 的 terminal paths 是 AA、AB、B，概率分别1/4、1/4、1/2；因此

\[
\mathbb E W_B=3,\quad \mathbb E D_B=\beta+51/8.
\]

唯一可选 C 节点是首任务 A 后：\(A_B=2\)、\(\Delta_B=21/4\)，
故 \(\lambda_{\min}=8/21\)、\(\beta_{\rm crit}=3/2\)。
首任务后总是 R 的收益率为 \(2/(\beta+15/4)\)。
当 \(\beta<3/2\)，它优于 VB；当 \(\beta>3/2\)，VB 最优；等号时二者并列。
两种任务的 reward/time 为1和3/4，资源耗费也不同。
这展示条件没有退化成“所有任务收益率相同”，但不构成模型类新颖性的证明。

### Step 9 — Existing UAV tree：何处违反条件，差距为何小

输入只有已完成的 root 和 **36 个 cycle-node receipts**；10 个 candidate 节点不进入计算。
复现脚本不导入 simulator 或 actor：

```bash
python research/regenerative_control/theory_certificate.py \
  --input artifacts/regenerative_p2a2_20260919 \
  --output research/regenerative_control/evidence/vb_theory_20260919/certificate.json
```

仅评估 VB 得到：

| Quantity | Value |
|---|---:|
| VB cycle reward | 2.1399176955 |
| VB cycle time | 198.4985388677 |
| rho_B | 0.010780521145 |
| 合法 C 节点 | 11 |
| G_B<0 的 cycle 节点 | 1 |
| 该节点任务前缀 | 100 → 300 |
| A_B(s) | 1.4444444444 |
| Delta_B(s) | 148.2934031059 |
| tail rate | 0.009740449772 |
| G_B(s) | −0.154235723373 |
| p_B(s) | 1/9 |
| D_B defect load | 0.017137302597 |
| T_min：首任务后立即 R | 113.4803902571 |
| 不使用 rho-star 得到的 rho-star 上界 | 0.010931536686 |
| 不使用 oracle 的 VB/optimum 下界 | **0.9861853328** |

唯一失败条件是该状态的 VB tail rate < rho_B。它有正 mission reserve，
但继续到边界的边际产出率过低。原 exact oracle 的 Delta_Q = −0.16819753
是在 **rho-star** 下计算；不要把它与 **rho_B** 下的 G_B = −0.15423572 混为一谈。

为什么 Proposition 1 不能直接套到 UAV？即使任务标签 IID，飞行耗时/能耗仍取决于
上一 delivery 的位置与残余速度；返航/停靠时间也有几何依赖。
因此 realized marks 不是 state-independent IID，R 也不只是累计耗能的 affine 函数。
这是假设不满足，不是通过拟合一个 residual 就能假装恢复定理。
一般树的证书不需要这些 IID-cost 假设。

精确已完成结果为 VB/Oracle = **0.9913422775**，高于该证书的98.6185%下界。
这里“独立”仅指计算不依赖 optimal-policy 求解；数据仍是同一棵树，不是独立实验或样本。
在本树中只有一个负 tail，oracle 恰好在那里停止，故
\(\rho^*-\rho_B=\mathcal D_B/182.0214940781\)。
VB 多做 \(1.4444444444/9=0.1604938272\) 个任务，却多耗
\(148.2934031059/9=16.4770447895\) 秒；其比值就是0.00974045。

式 (8) 在**保持本树所有转移不变**的纯数学参数族上给出
\(\beta_{\rm crit}=31.19539166\)，当前固定 overhead 为10。
这只是解析 corollary 的数值代入，**没有改 overhead、没有重跑，也不作为救活原主线的实验**。

## Remarks and Interpretation

1. 原主线 KILL 保持成立。数学上“safe 未必 optimal”是真的，但目前真实 gap 小且集中于一个状态。
2. Proposition 1 提供结构解释：在未知下一任务的 IID-mark 模型中，异质性被期望求和恒等式处理；
   affine recharge 使控制主要决定固定开销的摊薄。不要将此解释偷换成 UAV 已满足全部假设。
3. Proposition 2 的残差界与 Proposition 4 的访问加权 tail 缺口界是两种不同范围的证书；
   前者要求 IID realized costs，后者允许 state-dependent tree。UAV 的98.62%来自后者。
4. 误差界不是独立于模型的学习方法。当前证书仍用完整安全树；计算可扩展性未证明。

## Boundaries and Non-Claims

- 不证明 VB 在所有 persistent control problems 中接近最优。
- 不把“resource 正余量”当作完整 navigation viability；树的 safe set 同时包含导航可行性。
- 不推广到有预览筛选、任务改派、共享 charger、时间变化 process 的问题。
- 不将过去的算法线改名为理论线继续训练；没有新实验授权。
- 上述 propositions 是本次写出的有条件数学推导，**不是已通过 peer review 的原创定理**。
- 标准 renewal identity / Wald / ratio-to-excess / stopping verification 不能作为主要 novelty。
  Ratio-to-excess 的经典来源：[Dinkelbach (1967)](https://pubsonline.informs.org/doi/10.1287/mnsc.13.7.492)。
- 本次限定范围的文献核对确认标准工具来源，但不是穷尽 prior-art search；不能宣布“首次”。

## Open Risks

**Theorem kill-test 判读：**“只能在所有任务 reward/time 相同下成立”的数学退化风险已排除：
Proposition 1–2 和解析例子明确允许 reward/time/resource 异质、随机序列和部分非线性补能。
但是证明很短，依赖标准结果，**理论新颖性 gate 尚未通过**。

目前不能以此宣布 Spotlight potential 回升。若需要后续研究，应先确认是否存在超出这些
标准推论的实质贡献，以及是否能在不枚举完整树的情况下得到有用、紧的结构界。
本轮不展开新的环境或算法实验，也不自动开启下一阶段。
