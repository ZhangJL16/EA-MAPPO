# Viability-boundary theory: novelty kill audit

Date: 2026-09-19. Audited revision: `217be6d7`.

**Verdict: KILL the current standalone theory-paper story under the user's frozen
novelty criterion.** Prop. 1/3/4 and the overhead threshold reduce to standard
identities plus elementary consequences of the restricted stopping-prefix class.
No irreducible new technical step was identified. This is a contribution decision,
not a claim that every displayed formula has already appeared verbatim elsewhere.

原 learning 主线仍 KILL：VB/Oracle = 0.9913422775。现有数学保留为
thesis insight / supporting analysis，不作为已通过 novelty 的 ICML 主贡献。
本轮没有增加命题、运行 UAV、训练网络或自动启动 statistical-certificate pivot。

## 1. 审计范围与来源

只核对用户指定的四条文献路线及现有命题。阅读六篇指定论文的相关正文；
另核对 Gallager 的 Wald 讲义。Breiman 原章未直接取得，明确使用现代论文的
定位与转述，不冒充已读原始 Theorem 10.5。下表 PDF 页码均从 1 起算。

| ID | 已核对的 primary source | 定位 | 对本次审计的作用 |
|---|---|---|---|
| W | [Gallager, MIT 6.262 Lecture 15, 2011](https://ocw.mit.edu/courses/6-262-discrete-stochastic-processes-spring-2011/61c15adb62248b208b3cf78fdf5c1b08_MIT6_262S11_lec15.pdf) | PDF pp. 4–5, Wald's equality；PDF pp. 2–3, renewal reward | Prop. 1 的标准期望求和工具；不是近期原创性来源 |
| N | [Neely, Fast Learning for Renewal Optimization in Online Task Scheduling, JMLR 22(279), 2021](https://jmlr.org/papers/volume22/20-813/20-813.pdf) | §1 pp. 1–2；§3.3 Lemma 1, Eq. (22), p. 12；Theorems 1–2 | ratio/excess 转换是已有工具；其在线算法先看 task type 再选 mode，信息时序与我们不同，不能直接声称包含 Prop. 1 |
| I | [Van Foreest & Kilic, An intuitive approach to inventory control with optimal stopping, EJOR 311(3):921–924, 2023](https://pure.rug.nl/ws/portalfiles/portal/781978779/1-s2.0-S0377221723004204-main.pdf) | §3, p. 922；Lemmas 4.3–4.4, p. 923；Eq. (6), p. 922 | renewal 问题转为单参数 cycle stopping；库存阈值不是我们 VB 阈值的直接先例 |
| Z | [Zhang & Ross, On-Policy Deep Reinforcement Learning for the Average-Reward Criterion, ICML 2021](https://proceedings.mlr.press/v139/zhang21q/zhang21q.pdf) | §2 Assumptions 1–2, PDF p. 2；§4.1 Lemma 1, Eq. (4), p. 4；Lemma 2 和 Theorem 1 | 最直接的 average-reward PDL 定位；其 Kemeny/divergence bound 不是我们 reduction 所需的定理 |
| M | [Murthy, Moharrami & Srikant, Performance Bounds for Policy-Based Average Reward Reinforcement Learning Algorithms, NeurIPS 2023](https://papers.neurips.cc/paper_files/paper/2023/file/3da8e709fa1a7d9e23bee89d3c25b5b4-Paper-Conference.pdf) | Assumption 3.1, PDF p. 5；Theorem 3.3、Corollary 3.4, p. 6 | approximate PI 的误差界；不是 Prop. 4 公式的直接出处，也不被我们的精确树证书严格加强 |
| S | [Busatto-Gaston et al., Bi-Objective Lexicographic Optimization in Markov Decision Processes with Related Objectives, arXiv:2305.09634v2, 2023](https://arxiv.org/pdf/2305.09634) | §5, Theorems 2–3；Theorem 3, PDF p. 18 | 已研究 safety 概率优先、再优化 safe paths 上的 conditional mean payoff；不等同于本项目的 robust immediate-return admissibility |
| V | [La Rocca, Saveriano & Del Prete, VBOC: Learning the Viability Boundary of a Robot Manipulator using Optimal Control, arXiv:2305.07535, 2023](https://arxiv.org/pdf/2305.07535) | Abstract、§II–III | viability-boundary 计算与学习的既有背景；不提供我们的 stopping certificate |

I 的参考文献将 Breiman 定位为 **Stopping-rule problems**, *Applied
Combinatorial Mathematics*, Chapter 10, pp. 284–319 (1964)，不是泛指其概率教材。
I §3 引用其 §§10.13–10.14 / Theorem 10.5；本审计只确认这一间接定位。

来源限制：academic-search MCP 未挂载，改用 proceedings/JMLR/arXiv/大学官方资源。
ScienceDirect 和 Groningen 原下载路径访问受阻；正文经官方 `pure.rug.nl` PDF 的
web reader 读取。没有把搜索摘要当作 theorem 全文。六份可下载 PDF 的 SHA256 和
全部定位记录在 [source ledger](../research/regenerative_control/evidence/vb_novelty_20260919/sources.json)。
第三方论文全文不加入仓库；本地解析使用已有 pypdf（pdftotext 不可用）。

## 2. 逐命题 reduction verdict

| 现有结果 | 最接近的标准结果 / reduction | 是否严格更强 | 剩余技术步骤 | 决定 |
|---|---|---|---|---|
| Prop. 1 | W 的 Wald identity + renewal ratio | 未建立；属于附加 IID/additive 假设下的特化 | 预抽样决策使 inclusion predictable；VB 逐路径最大化任务数；ratio 单调 | KILL as main novelty |
| Prop. 3 | Z Lemma 1 的 advantage/PDL 原理 + stopping verification | 否；只优化同一 safe stopping class | 以 cycle excess 实现 SMDP 版本；负 advantage 节点做一次提前停止 | KILL as main novelty |
| Prop. 4 | 同一 PDL + unnormalized prefix occupancy domination | 未建立一般意义的严格加强 | stop mass ≤ VB visit mass；除以时间下界；改写成乘法界 | KILL as main novelty |
| overhead threshold | Prop. 3 + 固定开销的 ratio 敏感性；I 为 renewal/stopping 背景 | 否 | U 与 R 中相同 overhead 抵消；解一个标量不等式 | KILL as main novelty |
| Prop. 2（防止绕行） | Prop. 1 的 affine upper envelope | 否 | 去掉非负残差得到最优率上界，再与 VB 相除 | 保留 supporting bound |

“严格更强”必须在明确共同假设下比较结论/代价；不能因为符号更针对 UAV，
或不出现 Kemeny 常数，就宣称强于一般 average-reward RL 定理。

## 3. Prop. 1：三个 Wald 恒等式已经完成核心工作

以下是**现有命题到标准工具的还原**，不是新命题。
采用原包的有限 cycle、IID vector marks 和相同 safe set 假设。
令 \(I_i=1\{N_\pi\ge i\}\)。决策不观察下一 job，故 \(I_i\) 对过去可测，
与当前完整 mark 独立；即使 reward/time/energy 三分量相关仍有

\[
\mathbb E\sum_{i=1}^{N_\pi}X_i=\mathbb E[X_1]\mathbb E[N_\pi].
\]

这是 W 的 Wald 工具在向量 filtration 下逐分量应用；有界 cycle 可直接有限求和证明，
无需新增 optional-stopping 技术。把三分量代入原 affine recharge 定义便得到

\[
\rho_\pi=\frac{\mu_r n_\pi}{c n_\pi+\beta},\qquad
c=\mu_\tau+\gamma\mu_a>0.
\]

共享同一 job 序列时 \(N_\pi\le N_B\)；\(\beta\ge0\) 使上述比率随 n 单调不减。
这两点完成 Prop. 1。异质性得到允许是有用的适用性澄清，不产生新的证明障碍。
N 的任务信息时序不同，故判 KILL 的依据是这份显式还原，**不是**把 N 的学习定理
错认成一个 VB optimality theorem。

## 4. Prop. 3：G_B 就是 differential action advantage contrast

沿用原包符号：\(A_B\) 是剩余 reward，\(U_B\) 是剩余 time，\(R(s)\)
是立即返航 time。为避免把 reward 和 advantage 混同，标准 advantage 记为
\(\mathcal A_B\)。以 H 的 bias 为零，定义

\[
h_B(H)=0,\qquad h_B(s)=A_B(s)-\rho_B U_B(s),
\]
\[
Q_B(s,a)=\bar r(s,a)-\rho_B\bar\tau(s,a)
                 +\mathbb E[h_B(s')\mid s,a].
\]

在合法 C 节点，B 选择 C，故 \(Q_B(s,C)=h_B(s)\)；而
\(Q_B(s,R)=-\rho_B R(s)\)。因此精确地有

\[
Q_B(s,C)-Q_B(s,R)=A_B(s)-\rho_B(U_B(s)-R(s))=G_B(s),
\]
\[
\mathcal A_B(s,C)=0,\qquad\mathcal A_B(s,R)=-G_B(s).
\]

边界节点只能 R，advantage 为零；H 的强制 transition 也由
\(\mathbb E_B[W_B-\rho_BD_B]=0\) 得到零 advantage。
这正是固定策略的 bias / Bellman residual，**没有引入额外 regenerative state variable**。

Z Lemma 1 给出单位步长 MDP 的两策略 PDL。这里必须保留 macro duration，
不能把 \(\rho_B\bar\tau\) 错写成 \(\rho_B\)。在一个有限 regeneration cycle
上对上式望远镜求和，H 首尾 bias 抵消，直接得到同一 PDL 的 cycle 形式：

\[
(\rho_\pi-\rho_B)\mathbb E D_\pi
=\mathbb E_\pi\sum_k\mathcal A_B(S_k,a_k)
=-\sum_s q_\pi(s)G_B(s),
\]

其中 \(q_\pi(s)=\Pr_\pi(\text{本 cycle 在前缀 }s\text{ 停止})\)。
这就是原包式 (7)，不是独立于 performance difference 的新机制。

所有 G 非负时，各竞争策略的 rate 不超过 B；若某个 B 正概率访问的合法节点 G<0，
只在该节点改为 R，其余遵循 B，cycle excess 为 \(-p_B(s)G_B(s)>0\)。
这完成充分性和必要性；正访问概率限定用于保证从 H 出发的严格 improvement。

**适用边界：**Z Lemma 1 假设 aperiodic unichain；其主要 Kemeny bound 更强地
要求 ergodicity。我们的 early-stop policy 会使部分树节点不可达，macro chain
也未必非周期。因此不能把整条 Z theorem 原封不动套用。上面的有限 cycle
望远镜求和在原包假设内解决这一问题，不需要修改环境或加入随机重启。
这是 elementary specialization，未形成新的收敛、学习或测度技术。

## 5. Prop. 4：只剩 prefix 支配、时间归一化和代数

所有竞争策略在首次 R 前都与 B 相同，因而逐前缀

\[
q_\pi(s)\le p_B(s).
\]

把它代入上一节的 PDL：

\[
(\rho_\pi-\rho_B)\mathbb E D_\pi
\le\sum_s p_B(s)[-G_B(s)]_+=\mathcal D_B.
\]

再用原命题已经要求的 \(\mathbb E D_\pi\ge T_{\min}>0\)，取最优策略并整理：

\[
\rho^*\le\rho_B+\mathcal D_B/T_{\min},\qquad
\rho_B/\rho^*\ge(1+\mathcal D_B/(\rho_BT_{\min}))^{-1}.
\]

这已经完整还原 Prop. 4。原包给出的 \(T_{\min}\) 构造也只是对
\(\tau+R(s')-R(s)\) 做 cycle telescoping，在其 conditional mean 非负时取首任务后立即 R。
乘法证书来自除以正的 rate upper bound，没有额外最优策略消除定理。

必须修正审稿 shorthand：这里支配的是**每周期未归一化的 stop/visit mass**，
不是一般的 \(d_\pi(s)\le d_B(s)\)。即使某前缀每周期两策略均必达，
更短 cycle 的策略每单位时间仍会更频繁到达它。归一化分母正是
\(\mathbb E D_\pi\)，不能省略。原推导包使用 p_B，是正确的；无需改其证明。

prefix 结构确实比任意两策略的 distribution mismatch 更容易控制，
但原因是竞争策略不能改道、不能超过 B 的停止前缀。它是很强的 policy-class 限制。
Z/M 的一般策略界与此不同；M Theorem 3.3 还处理近似评估/改进误差，
不是我们证书的同义公式。**正确判定是短 corollary，而不是宣称逐字重复某个 prior theorem，
也不是宣称已在共同模型类中严格改进 generic bound。**

## 6. Overhead：固定树上的一个标量不等式

原假设 F 要求改变 \(\beta\) 不影响转移、reward、安全集合或 H。
于是 \(A_B\) 固定，U 与 R 中的 \(\beta\) 抵消，\(\Delta_B\) 固定。
对原包限定的正 A、正 Delta 节点，Prop. 3 等价于

\[
\frac{\bar W_B}{D_B^0+\beta}\le\min_s A_B(s)/\Delta_B(s).
\]

解这个不等式就是已有 \(\beta_{\rm crit}\)。无新技术步骤。
I 的库存 replenishment K 是成本，原包的 beta 是时间；未找到它直接陈述
这一相同 threshold，也不据此声称公式完全已有。novelty KILL 来自上述代数还原。
本轮没有利用 threshold 去修改环境或重跑参数。

## 7. “Baseline-only”能力边界

| 声称 | 当前证据 |
|---|---|
| 不输入 optimal policy / rho-star | 成立；现有脚本由 root + 36 cycle-node receipts 计算 |
| 只需固定 B 的 tail，而不求最优 tail | 成立；这也是标准 advantage 评估的含义 |
| 不需完整模型 / 不枚举 safe tree | 不成立；当前仍完整评估 represented safe tree |
| 只执行 B 即可得到所有输入 | **未成立**；合法 C 内部节点上的立即 R 成本是 counterfactual，脚本从独立 R branch 读取 |
| finite-sample、model-free、1-delta certificate | 未提供；没有 confidence bound 或未知 return cost 的识别论证 |
| 严格减少状态复杂度 / sample complexity | 未建立；显式树 evaluation 与固定 rho 的 optimal-stopping DP 都遍历该树，尚无复杂度分离 |

实现核对：`research/regenerative_control/theory_certificate.py` 在每个节点读取
`model.tau_R` 并与 `R.row.elapsed_time` 比较，然后计算 VB tails。
因此“baseline-only”目前应读作 **baseline-policy evaluation with known branch costs**，
不能读作仅有 baseline behavior trajectories。已有 98.6185% 证书依然有效，
但这一区别不能被一个新的措辞或 sample-complexity 占位公式消除。

## 8. Gate outcome 与保留内容

用户的 frozen test 是：如果全部核心证明仅需经典定理加 1–2 个结构观察，
就关闭 theory-paper story。本次 reduction 满足该条件，故 **novelty gate FAIL / KILL**。
结论不是“再添几个 theorem 就能继续”，也不是“相同公式未检出所以通过”。

- 保留原推导包、36-node 原始证据、frozen SAC、环境与代码。
- 保留结论：safe continuation 未必 rate-optimal，但当前任务族 VB 已有 99.13% optimal throughput。
- 保留证书作为可复核的特化分析，不宣传为新的 model-free ML theory。
- 当前 level 仍 2/5；不宣称 paper-ready 或 venue novelty。
- 不自动进入用户列举的 A/B/C 新路线；任何新 scientific question 需单独界定。

本审计不是全领域穷尽综述，也不证明未来所有 regenerative certificates 都无新颖性。
它回答的是当前四个 contribution candidates，且以可显式写出的 reduction 为 KILL 依据。

## 9. Integrity check

原 `DERIVATION_PACKAGE.md` 保持逐字不变，SHA256:
`7f27a3465c7f92e03013011ff2076d670e8573f6013b51e5f015a458f34fd416`。
只新增审计与来源记录，并更新当前研究状态；未改 runtime、证据或 checkpoint。
该结论是数学/文献审计，无需启动 simulator 或训练来验证。
