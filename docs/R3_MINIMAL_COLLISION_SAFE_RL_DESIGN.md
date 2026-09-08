# R3 最小碰撞安全强化学习设计

日期：2026-09-05  
状态：实现核验中；先运行冻结 R3 的接口开发实验，尚未启动新策略训练

## 1. 结论

当前最适合 R3 的第一步不是 CPO、PPO-Lagrangian、Recovery RL、再加一个 safety critic，也不是继续调碰撞罚分。推荐一个**单一 Bellman 目标替换**：

> 用 R3 现有的 twin Q 估计“在首次碰撞或触边前到达目标”的 reach-avoid 值；用物理加速度的实际修正量构造连续人工生存权重，使安全信号在真正接触前进入 TD 目标。仅在输出为有效凸投影的分支，修正量才有到可行集距离的解释。

项目内暂称 **R3-RACT**（Reach-Avoid Constraint Termination）。它是 reach-avoid Bellman 语义、CaT 随机终止原语和项目已有 sampled-data HOCBF 的场景化组合，不主张算法所有权。

它不增加网络、cost critic、恢复策略、规划器、MPC、规则树或能耗损失：

- 保留 R3 structured-LiDAR actor 和连续三维动作；
- 保留 twin Q 的数量与结构，但重置参数，因为估计对象改变；
- 关闭旧 Jacobian bridge，而不是把新损失叠上去；
- 将已有 HOCBF 切换到项目已经实现的 sampled-data robust 模式；
- HOCBF 同时提供训练约束语义和最终执行防线，但不包装成新学习模块。

这里“最适合”指：在当前代码、观测、checkpoint 与失败证据下，它的假设最少、目标最对齐、新增对象最少；不表示 learned actor 获得无条件零碰撞证明。

## 2. R3 的真正问题

R3 的正式 500-task 结果是：

| 指标 | 观测值 |
|---|---:|
| 成功率 | 0.960 |
| 平均路径比 | 1.1880 |
| 有障碍碰撞的 episode | 3/500 |
| 障碍碰撞步 | 2,242/394,853 |
| HOCBF intervention step rate | 0.3587 |
| HOCBF emergency-brake step rate | 0.1516 |

其中一个失败 episode 有 2,240 个碰撞步。环境碰撞后把无人机移到最近合法位置、速度置零，然后继续 episode。碰撞不是 terminal：

\[
\texttt{terminated}
=\mathbf 1\{\text{energy exhausted or task completed}\}.
\]

到达奖励是 \(+100\)，首次障碍碰撞是 \(-1.2\)，连续碰撞缩放后仅 \(-0.42\)。因此训练目标允许“碰撞—环境修复—继续拿任务回报”，而正式指标要求“一次接触即失败”。这首先是价值对象错误，不是网络容量不足。

在有效投影分支，环境执行

\[
u_t^{\mathrm{exec}}=\Pi_{\mathcal C(s_t)}(u_t^{\mathrm{nom}}).
\]

紧急制动、QP fallback 和后续径向限幅不必满足这个等式。replay 中 SAC 对名义动作估值，但多个不安全名义动作可能被映射到近似相同的执行动作并得到近似相同的后继状态，形成 action aliasing。它可能促使 actor 依赖 shield；这仍是待检验的机制假设，不是已证因果结论。

旧 Jacobian bridge 也有实测覆盖瓶颈：R3 训练中几何预筛有效率约 86.44%，但 actor-to-anchor trust region 后参与损失的平均比例仅 24.61%。它还是一步局部模仿，不能把失去未来安全到达机会的影响沿轨迹传播。因此本方案替换 bridge，不增加 bridge 权重。

## 3. 正确的任务对象

令 \(G\) 为目标集合，\(F=F_{\mathrm{obs}}\cup F_{\mathrm{wall}}\) 为障碍与边界接触集合：

\[
\tau_G=\inf\{t:X_t\in G\},\qquad
\tau_F=\inf\{t:X_t\in F\}.
\]

基础目标是

\[
V_\gamma^\pi(s)
=\mathbb E_s^\pi[
\gamma^{\tau_G-1}\mathbf 1\{\tau_G<\tau_F,\ \tau_G<\infty\}],
\qquad 0<\gamma<1.
\tag{1}
\]

这里仅从非终止初态讨论，首次到达状态的索引为 \(\tau_G\ge1\)，下一步成功的 payoff 为 1，所以指数是 \(\tau_G-1\)。未成功的路径贡献定义为 0。它把任何首次碰撞轨迹计为零，并偏好更快的安全到达。由于实际评估有 4,000-step 截止而 actor 输入没有剩余时域，式 (1) 是无限时域折扣替代，不冒充精确 finite-horizon probability；正式评估仍单独报告时限内 safe-goal success。

## 4. 唯一连续约束

当前环境的真实障碍是圆柱，filter 使用 top-K LiDAR 点构造局部点球代理。下面只对球形代理 \(i\) 推导，不能据此声称覆盖整根圆柱、漏检表面或边界。令相对位置 \(r_i=p-o_i\)、相对速度 \(w_i=v-v_i^o\)、膨胀安全半径 \(R_i\)：

\[
h_i(p)=r_i^\top r_i-R_i^2.
\]

在双积分器 \(\dot p=v,\dot v=u\) 下，sampled-data 二阶 HOCBF 约束为

\[
2r_i^\top u+2w_i^\top w_i-2r_i^\top a_i^o
+(k_1+k_2)2r_i^\top w_i+k_1k_2h_i
\ge \rho_i(\Delta),
\tag{2}
\]

其中 \(\rho_i(\Delta)\) 是零阶保持区间 \(\Delta=0.05s\) 内的残差上界。ordinary HOCBF 相当于令 \(\rho_i=0\)，不能推出采样间安全。

让已有 sampled-data HOCBF、执行器和 next-velocity 行组成闭凸集

\[
\mathcal C_\Delta(s)=\{u:A_\Delta(s)u\ge b_\Delta(s)\}.
\tag{3}
\]

网络输出虽在 \([-1,1]^3\)，环境会径向裁剪水平分量，再按水平/垂直上限 \(A_h=5,A_v=3\) 缩放。因此不能把网络坐标中的变化直接当作 QP 投影距离。对每个物理子步 \(j\)，以实际物理加速度定义

\[
d_{t,j}
=\frac{\|u^{nom}_{t,j}-u^{exec}_{t,j}\|_2}{2\sqrt{A_h^2+A_v^2}}
\in[0,1].
\tag{4}
\]

它是实际修正强度，不是碰撞概率。仅当集合非空、闭凸，且最终执行输出确实是该物理坐标下的精确投影时，

\[
d_{t,j}=0
\iff u^{nom}_{t,j}\in\mathcal C_\Delta(s_{t,j}).
\tag{5}
\]

第一版不加入 TTC、clearance、碰撞预测器、多 cost、手工区域或 energy term。HOCBF slack 只作诊断。

## 5. 人工生存权重与 Bellman 递推

令 \(c_t\in\{0,1\}\) 表示 transition 首次发生真实碰撞/触边，且碰撞优先于成功。定义

\[
I_t=\sum_{j=1}^{m_t}d_{t,j}^2\Delta_{physics},\qquad
q_t=(1-c_t)\exp[-\kappa I_t].
\tag{6}
\]

仅真实碰撞或触边使 \(q_t=0\)。QP infeasible、fallback、emergency 单独记录，仍使用有限的实际修正强度，不能把可恢复制动等同于失败。\(q_t\) 是训练用人工生存权重，不是估计的真实无碰撞概率。一个 policy step 通常有四个 0.05s 物理子步；先逐子步平方积分，不能先平均动作再平方。对同一物理轨迹的纯时间划分具有一致性，但改变仿真步长可能改变轨迹，不能声称普遍不变。

令 \(g_t=\mathbf1\{s_{t+1}\in G,\ c_t=0\}\)。基础 Bellman 算子为

\[
(\mathcal T_q^\pi Q)(s,a)
=\mathbb E\!\left[
q_t\left(
g_t+\gamma(1-g_t)
\mathbb E_{a'\sim\pi(\cdot|s_{t+1})}Q(s_{t+1},a')
\right)\right].
\tag{7}
\]

\(\kappa=0\) 是只修正首次碰撞语义的 hard reach-avoid control；\(\kappa>0\) 才检验连续投影距离是否带来额外学习价值。

### 严格保持动作排序的密集进展信号

纯终点信号在长距离任务上较稀疏。不能重新叠加一串 progress/velocity reward，也不能给每个生存步加正常数，因为后者会鼓励拖延。令

\[
\beta_t=\gamma q_t(1-g_t),
\]

选有界状态势函数

\[
\Phi(s)=-\lambda_\Phi\,\mathrm{clip}
\left(\frac{d_G(s)}{D_G},0,1\right),
\]

并在吸收态设 \(\Phi=0\)。训练 target 为

\[
\widetilde r_t
=q_tg_t+\beta_t\Phi(s_{t+1})-\Phi(s_t),
\tag{8}
\]

\[
y_t=\widetilde r_t+
\beta_t\min_{j=1,2}
\widetilde Q_{\bar\theta_j}(s_{t+1},a'_{t+1}).
\tag{9}
\]

不要再把整个 \(\widetilde r_t\) 乘一次 \(q_t\)。代入可得

\[
\widetilde Q_q^\pi(s,a)=Q_q^\pi(s,a)-\Phi(s),
\tag{10}
\]

所以同一状态下所有动作只减去相同常数，动作排序完全不变。距离势只改善 TD 信号密度，不改变式 (7) 的目标。

actor 仍用两个现有 critic：

\[
\max_\theta\;
\mathbb E_{s,a\sim\pi_\theta}
[\min_j\widetilde Q_{\theta_j}(s,a)
+\alpha\mathcal H(\pi_\theta(\cdot|s))].
\tag{11}
\]

但 differential entropy 不进入式 (7)–(9) 的 survival target。连续策略微分熵可为负，把普通 SAC mixed-sign soft value 整体乘 \(q_t\) 时，“更危险必然更低值”并不成立。第一版把熵仅作为 actor 探索正则，并在 fine-tune 后半程退火到零。

\(\gamma\) 从 R3 开发轨迹的典型成功步数 \(L_{50}\) 冻结：

\[
\gamma=q_{1/2}^{1/L_{50}},
\tag{12}
\]

例如令典型成功保留 \(q_{1/2}=0.5\) 权重。不得看正式 test 调参。

\(\kappa\) 只取一个开发数据校准值。令 R3 robust 接口下安全成功开发轨迹的整程积分为 \(H=\sum_t I_t\)，取全部安全成功轨迹的第 90 百分位 \(H_{90}\)。定义“该高修正量分位的整程权重减半”：

\[
\kappa=\frac{\log 2}{H_{90}}.
\tag{13}
\]

这是启动短训练前的显式方案修订，不是事后声称预注册。接口实验给出的旧非零中位数方案 \(\kappa=0.372132\) 使 49 条安全成功轨迹中的 7 条权重低于 0.01，最小值为 \(2.94\times10^{-5}\)。新方案 \(H_{90}=15.1510\)、\(\kappa=0.0457492\) 在同一开发集的最小整程权重为 0.2773；因 \(I_t\le0.2\)，单步非接触权重至少约 0.99089。它只减弱连续约束强度，未对权重做截断，也未改变 Bellman 形式；新策略的长程权重仍须记录，开发范围外不保证该下界。正式测试不参与选择。

## 6. 可证明性质

### 命题 1：收缩与唯一不动点

固定 \(\pi\) 和 \(q\in[0,1]\) 后，\(\mathcal T_q^\pi\) 在 sup norm 下至多是 \(\gamma\)-收缩，因此存在唯一不动点 \(Q_q^\pi\in[0,1]\)。式 (8)–(9) 的 shaped 算子具有相同 continuation 系数，也收缩，且不动点满足式 (10)。

### 命题 2：killed-process 解释

令 \(\tau_B\) 为条件生存概率 \(q_t\) 生成的人工吸收时间，则

\[
Q_q^\pi(s,a)=
\mathbb E[
\gamma^{\tau_G-1}
\mathbf1\{\tau_G<\tau_F\wedge\tau_B,\ \tau_G<\infty\}
\mid S_0=s,A_0=a].
\tag{14}
\]

因此

\[
0\le Q_q^\pi(s,a)
\le
\mathbb E[
\gamma^{\tau_G-1}
\mathbf1\{\tau_G<\tau_F,\ \tau_G<\infty\}
\mid s,a].
\tag{15}
\]

这是 value 的保守顺序，不是概率校准。\(Q_q=0.9\) 不能称为 90% 真实安全成功率。

### 命题 3：对 \(\kappa\) 的单调性

对固定策略、固定轨迹分布和无熵非负基础目标，若 \(\kappa_2\ge\kappa_1\)，则逐轨迹有 \(q^{(\kappa_2)}_t\le q^{(\kappa_1)}_t\)，从而

\[
Q_{\kappa_2}^\pi(s,a)\le Q_{\kappa_1}^\pi(s,a).
\tag{16}
\]

该命题不意味着神经训练后的实际碰撞率随 \(\kappa\) 单调，也不适用于把熵混入 survival target 的版本。

### 条件式 hard-safety 链条

若 learned actor 在所有访问物理子步满足 \(d_{t,j}=0\)，输出始终是有效投影，且 sampled-data HOCBF 的安全初始集、模型、完整约束覆盖、intersample bound 和递归可行性假设都成立，则 actor 名义动作自身满足相应 sampled-data HOCBF 条件，final filter 退化为 identity。当前 top-K 点球代理并未验证完整几何覆盖，训练与有限 rollout 也不能证明全状态前提；这里不是当前环境的安全证书。

### 定理与训练实现的边界

上述收缩和恒等式针对完整 Markov 状态或历史/信念状态以及精确策略评估。R3 使用单帧 LiDAR 的 2055 维观测，未证明充分 Markov；非线性函数逼近、twin-Q 的取小估计及离策略训练也没有由上述定理获得收敛保证。无熵 critic 加 actor 熵正则是 **SAC 衍生的离策略 actor–critic**，不是未修改的标准 SAC。移除旧 bridge、改变 critic 对象并非仅调一个 SAC 超参数。

无限时域替代目标下，纯时间截断必须从真实终止观测 bootstrap，不能从自动 reset 的新观测 bootstrap；真实接触和成功才令 continuation 为零。两臂必须统一这个约定。人工 survival 权重只进入 TD 系数，不随机结束物理采样。

## 7. 为什么不选其他方法

| 方法 | 新增对象 | 不选作第一步的原因 |
|---|---:|---|
| CPO | cost value + on-policy trust region | 替换已能训练的 SAC；只约束平均成本 |
| SAC/PID-Lagrangian | cost critic + dual state | 又增加 critic 与阈值动态；不直接修正首次碰撞对象 |
| Recovery RL | safety critic + recovery policy | 当前 R3 recovery 仅 275/310 安全，不能作可靠前提 |
| SAILR | cost-Q + backup rule | 与已有 HOCBF 干预层重叠 |
| RCRL/reachability | reachability critic | 语义强，但不是最小修改 |
| learned CBF | barrier network + validation | 已知动力学/几何下重复造模 |
| hard ET-MDP | 0 | 必须作 \(\kappa=0\) 对照；单独使用时 shield 下碰撞样本过稀 |
| **R3-RACT** | **0** | 直接修正价值对象并使用接触前连续信号 |

## 8. 最小实现边界

保留：

- R3 actor 与 structured-LiDAR encoder 权重；
- 连续三维动作与 twin-Q 结构；
- 同一障碍、任务、地图、并行环境和正式测试；
- 中间 checkpoint 与 uv 环境。

替换：

1. shield loss weight 设为 0，不再训练 Jacobian bridge；
2. 仅当接口开发实验通过后，sampled-data robust HOCBF 才成为共同训练/执行接口；
3. actor/encoder 从 R3 加载，twin Q 与 target Q 重置；
4. replay 记录 nominal action、executed action、逐子步修正积分 \(I_t\)、\(q_t\)、goal hit 与 first contact；
5. critic target 使用式 (8)–(9)；
6. actual collision 在学习语义中吸收；包装器在当步物理修复后结束 episode，不收集接触后的成功轨迹。原环境碰撞后挪到合法位置、速度归零的行为保持不变。

禁止顺手加入 energy critic、第三 critic、GRU、attention 重构、额外 LiDAR channel、MPC、A*、APF、waypoint、恢复策略或安全策略切换。

旧 ordinary-HOCBF transition 不能仅重打标签后复用，因为换成 robust shield 后真实 successor 通常已经改变。

## 9. 最小、可归因实验

按顺序做，不并行扩展算法树：

1. **接口核验**：冻结 R3，分别用 ordinary 与 sampled-data robust HOCBF，在同一开发任务上测 success、path、QP infeasible、intervention、emergency 与 first contact。先确认 robust 接口不是停飞器。
2. **共同底座，\(\kappa=0\)**：R3 actor 初始化、critic reset、bridge off、首次碰撞吸收、相同 potential shaping。
3. **唯一变化，\(\kappa>0\)**：检验 projection survival weight 的增量。两组除 \(\kappa\) 外完全一致。

学习臂先各 50k transitions，每 10k 保存 checkpoint。只有开发 Gate 通过才做多 seed 500k 和 500-task 正式测试。

主要判断量：

- safe-goal success：时限内到达且 episode 无首次接触；
- task success 与 path ratio，防止用停飞换安全；
- nominal 实际物理修正率、修正量均值与 P90；
- HOCBF intervention、fallback 与 emergency rate；
- shield-on 是部署结果；shield-off 仅诊断神经网络自主避障；
- episode-level first contact 为主，collision steps 仅诊断卡死。

否决条件：

- \(\kappa>0\) 相对 \(\kappa=0\) 没有 paired safe-goal 改善；
- 实际修正量下降但 task success/path 明显退化；
- QP infeasible/emergency 上升，说明 robust 集与任务不兼容；
- 改善全部来自 robust shield，shield-off actor 无改善且 intervention 不降；
- 生存权重乘积普遍接近零，长期目标信号被消灭；
- 结果只在一个 seed 或距离桶成立；
- replay 中 \(q_t\) 不能从名义动作、filter 版本和约束重算一致。

## 10. 论文边界

安全模块应诚实写成已有原理的场景化适配。潜在研究问题是：

> 安全投影造成的 action aliasing 是否会阻止 off-policy TD 学会 nominal admissibility，而 constraint-terminated reach-avoid Bellman 目标能否在不新增 safety critic 的情况下解除这种依赖？

即使成功，也不声称：

- projection distance 或 \(q_t\) 是真实碰撞概率；
- learned actor 在所有状态零碰撞；
- finite simulation 等于真实无人机安全；
- HOCBF 在 QP infeasible、感知漏检或模型失配时仍有保证；
- 安全到达自动解决能耗安全。

## 11. 主要依据

- [CaT, IROS 2024](https://doi.org/10.1109/IROS58592.2024.10802334)
- [SoloParkour, CoRL 2024](https://proceedings.mlr.press/v270/chane-sane25a.html)
- [Safe Exploration by Solving Early Terminated MDP](https://arxiv.org/abs/2107.04200)
- [Reach-Avoid RL, RSS 2021](https://roboticsproceedings.org/rss17/p077.html)
- [RCRL, ICML 2022](https://proceedings.mlr.press/v162/yu22d.html)
- [PID Lagrangian, ICML 2020](https://proceedings.mlr.press/v119/stooke20a.html)
- [Recovery RL, RA-L 2021](https://arxiv.org/abs/2010.15920)
- [Potential-based shaping, ICML 1999](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf)
- [HOCBF, IEEE TAC 2022](https://doi.org/10.1109/TAC.2021.3105491)
- [Sampled-data CBF, IEEE L-CSS 2022](https://doi.org/10.1109/LCSYS.2021.3076127)

完整筛选记录：literature-search-20260905-r3-collision-safe-rl/。

## 12. 本次实现与开发审计协议

`scripts/run_r3_collision_interface_audit.py` 冻结 accelerated R3 的 500k checkpoint，用开发 task seed 510001、world seed 520001 起始的 50 个分层任务，逐任务配对 ordinary / robust 两臂，共 100 条轨迹。两臂逐条验证障碍布局及初始观测哈希相同，只切换 sampled-data robust 标志。使用 8 个单线程环境 worker、中心 CUDA 批量推理，保留完整 2055 维观测、原物理步长和原任务时限。首次真实接触即按失败结束开发 episode。

开发 gate 在启动前固定：robust 安全到达最多比 ordinary 少 1 个任务、接触数不增加、fallback 步率增量不超过 5 个百分点、至少半数任务安全到达、有非零连续修正信号。这是防止明显失配接口进入昂贵训练的**工程筛选**，不是定理推导的阈值、显著性检验或论文达标标准。通过也只允许考虑配对短训练，不能宣称方法有效。

结果逐 episode 原子保存，重启跳过完成记录；正在执行的 episode 会重跑，不冒充精确中途状态恢复。启动与恢复使用同一命令，源码/模型哈希不一致则拒绝混入原结果。创建 `PAUSE_REQUEST` 可在下一批向量步停止；恢复前移除该请求文件。`RESULT.json` 自动汇总接口判断，但不会自动训练。当前输出目录：`artifacts/r3_collision_interface_audit_50pairs_20260905_v1/`。

最小目标代数实现位于 `experiments/r3_collision/core.py`，4 项单元测试覆盖物理动作径向别名、子步累计、一旦接触优先失败及势函数抵消。8-step smoke 只验证进程/接口/落盘，不用于成功率结论。

## 13. 配对短训练实现协议

入口 `scripts/run_r3_reach_avoid_pair.py`，算法与环境包装器为
`experiments/r3_collision/training.py`。两臂均为 R3 actor 初始化、原双 Q
全量重置、bridge off、robust HOCBF，保留完整 2055 维观测。仅连续臂使用
式 (13) 的正 \(\kappa\)，hard 臂取零。关闭不再使用的 projection geometry
日志计算，不改变滤波约束、物理仿真或神经输入。

每臂预算 50k 环境 transitions，其中前 10k 冻结 actor：用原 R3 随机策略
采样，不用均匀随机动作，使用完整成功/接触轨迹的递推累计 shaped return
初始化 critic。超时轨迹没有可知的无限时域 MC 标签，故不参与此初始化；
这会有完成轨迹选择偏差，因此它仅是 warm-start，不能称为全状态无偏值估计。
后 40k 只使用式 (9) 的 TD 目标和 actor 更新，不叠加 MC 辅助损失。
若初始化期间没有获得可用完整轨迹/更新，拒绝从随机 critic 开始更新 actor。

两臂统一：critic LR 3e-4、actor LR 3e-5、batch 256、每个环境 transition
一次梯度更新、原 tau=0.005，actor 熵系数初始 0.001 并在策略更新前半段
线性退火至零。critic 无熵项；动作/状态势不增加网络。两臂依次运行，单个
作业使用 8 worker + CUDA，避免抢占同一 GPU。

每 10k 保存一套 model（含 optimizer）+ replay + RNG，并原子发布目录与
LATEST 指针。每 1k 检查暂停请求；续跑保留优化进度与样本，但重置未完成
仿真 episode，清空它们的待完成 MC 索引，不声称逐位一致的环境状态恢复。
完成记录日志可能含故障后回退 checkpoint 的旧诊断行，正式比较仅用单独
原子保存的最终评估轨迹，不从训练日志拼接正式样本。

两臂训练完成后，与冻结 R3 一起在新的 50 个分层开发任务上分别做 shield-on
和 shield-off 评估（共 300 条轨迹）。任务/世界种子为 710001/720001（区别于
联调使用的 610001/620001），所有
模型逐任务核对初始观测与障碍布局一致。shield-off 仅仿真诊断，不是真实
部署。该单 seed 短实验只判断是否值得继续研究，不自动升级 500k 或正式 500 测试。
