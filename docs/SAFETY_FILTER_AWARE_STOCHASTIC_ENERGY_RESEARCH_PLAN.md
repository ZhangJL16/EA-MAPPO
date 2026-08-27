# Safety-Filter-Aware Stochastic Energy-to-Go: Overall Research Plan

## 中文执行摘要

当前主线不是继续包装“Energy Critic + HOCBF 特征 + 阈值返航”，而是研究：

> 当学习导航策略与优化型硬安全过滤器共同决定真实执行轨迹时，如何可靠预测到达任务点或充电站所需的随机累计能耗，并在策略、障碍物和安全过滤器变化后保持可适应性。

进一步面向 ICLR oral-level evidence，核心问题收紧为：闭环 policy-filter pair 从未联合出现时，什么最小 executed-action interface 以及什么 target-support 条件足以识别其长期资源分布。`pair-level trajectory overlap` 可以不是必要条件，但 `target interface/occupancy support` 不能被省略。普通 executed-action world model 是必须击败或解释的最强朴素基线。

系统职责保持分离：导航策略产生 nominal action，HOCBF 负责最终碰撞安全，Energy 模块预测 executed safe policy 下的 Energy-to-Go 分布，高层仅依据剩余能量、预测上界和 reserve 决定是否单向 commit 到 charger。学习模块不被描述为安全证书。

研究的关键不是证明“安全干预一定增加能耗”，而是识别干预是否通过制动、绕行、重新加速、到达时间变化和后续干预序列产生可预测的长期能耗后果。该影响可能为正、为负，或完全由当前 Markov 状态解释。只有实验确认历史干预在控制当前状态后仍提供额外信息，才保留 latent intervention-history state。

理论部分需要完成：executed-policy 下的 undiscounted SSP/Bellman 语义；Markov 充分时历史冗余、存在有限记忆时的增广状态条件；HOCBF 固定 active-set 内的局部投影敏感性；安全干预对即时能耗的有符号局部影响；factorized rollout 的 population semantics；功率、动力学、mode、hitting-time 和 policy/filter shift 的误差分解；条件校准上界的决策含义；以及 one-way charger commitment 的 no-chattering 性质。明确不证明深度 TD 全局收敛、任意环境必达、全局 QP 可微或 distribution shift 下无条件 coverage。

实验首先做 intervention-aligned consequence analysis，而不是直接训练更大网络；随后依次验证历史是否必要、direct return 与 factorized safety-workload rollout 的差异、不同 safety representation 的价值、policy/filter/obstacle shift 下的失效、recent-window adaptation 与结构化 adaptation 的比较，最后才把估计器接入 TASK→CHARGER 决策。所有核心结论必须来自多 seed、完整轨迹隔离、held-out evaluation 和等预算比较。

ICLR 门槛是发现可复现且可迁移的学习问题，并证明结构化 safety-workload modeling 超过参数匹配黑盒、简单 recent-window retraining 和版本标签。如果优势只存在于单一 UAV/HOCBF 配置，则应诚实转向 ICRA/IROS/RA-L 或 autonomous-systems 论文，而不是强行制造通用理论 novelty。

## 1. Document Status

This document defines the current primary research direction for the UAV project.
It consolidates the useful parts of the existing navigation, HOCBF, energy-model,
distributional-learning, and policy-aware experiments without promoting pilot or
running-experiment values into formal paper claims.

The target is an ICLR-level machine-learning contribution. The UAV application is
the main testbed, but the intended scientific object is broader than one simulator:

> Long-horizon resource prediction when the executed trajectory is jointly induced
> by a learned policy and an optimization-based hard safety filter.

Current status:

- HOCBF remains the final hard collision-safety layer.
- The learned policy, safety context, Jacobian features, and Energy models are not
  safety certificates.
- The existing Jacobian Energy Bridge is retained as a baseline and negative/partial
  result, not assumed to be the final method.
- Existing frozen-policy energy learning remains a control condition.
- Formal claims remain `PENDING` until multi-seed held-out evidence is complete.

## 2. Research Motivation

### 2.1 Smartphone analogy

Smartphone battery management separates two questions:

1. **Battery state:** how much usable energy remains?
2. **Time-to-empty:** how long will that energy last under uncertain future usage?

Mature mobile systems do not assume a fixed future power rate. They use recent
behavior, workload modes, historical usage, and periodic re-estimation. Some work
models time-to-empty as a first-passage distribution under stochastic mode
switching.

The UAV analogue is:

1. **Battery state:** remaining usable energy (e_t).
2. **Energy-to-go:** energy required to reach a task goal or charger under the
   future executed safe policy.

The analogy is useful but not itself a novelty claim. Smartphone workloads are
largely app/user driven. UAV safety workload is endogenous: the navigation policy,
obstacle geometry, perception, and HOCBF projection jointly determine future
executed actions and energy.

### 2.2 Core UAV difficulty

Let the nominal navigation policy be

\[
a_t^{\mathrm{nom}} \sim \pi_\theta(o_t,g),
\]

and let the hard safety filter return

\[
u_t^{\mathrm{exec}}
=
\Pi_H(x_t,a_t^{\mathrm{nom}},\xi_t),
\]

where \(\xi_t\) denotes obstacle/perception/filter context.

The realized step energy is

\[
c_t
=
P(x_t,u_t^{\mathrm{exec}},w_t)\,\Delta t_t,
\]

where \(w_t\) may include wind, payload, battery condition, or other physical
context.

The safety-filter-aware stochastic Energy-to-Go is

\[
Z_E^{\pi_\theta,\Pi_H}(x_t,g)
=
\mathcal L\!\left[
\sum_{k=t}^{T_g-1}c_k
\;\middle|\;
x_t,g,\pi_\theta,\Pi_H
\right],
\]

with goal hitting time

\[
T_g=\inf\{k\ge t:x_k\in G(g)\}.
\]

The key challenge is that the safety filter changes both the current energy cost
and the future state distribution through braking, redirection, path detours, and
active-set changes.

## 3. Primary Research Question

The primary research question is:

> In safety-filtered sequential systems, when and why does a structured model of
> future safety workload provide more reliable long-horizon resource prediction
> than direct return learning, especially under obstacle, filter, and policy shift?

The project must not reduce this question to:

```text
LiDAR/HOCBF features
+ Energy Critic
+ threshold switch
```

That formulation is an obvious combination of established components.

## 4. Scientific Objects to Study

### 4.1 Battery state

Battery state is separate from Energy-to-Go. It includes:

- measured remaining energy;
- usable capacity;
- capacity uncertainty;
- aging and temperature effects when introduced.

State of charge is used by the high-level decision rule. It is not required as an
input to the pure goal-conditioned Energy-to-Go model.

### 4.2 Realized power model

For the current simulator, `TelemetryCostModel` provides the realized physical
cost using executed velocity, realized propulsion acceleration, and actual
transition duration.

The simulator research should not train a neural network merely to reproduce a
known deterministic power equation. A future real-UAV version may use

\[
P_{\mathrm{real}}
=
P_{\mathrm{physics}}
+
\Delta P_\omega(\text{telemetry},w_t)
\]

to learn a residual correction.

### 4.3 Safety workload

Safety workload summarizes how the filter alters future control. Candidate
observable features include:

- nominal and executed actions;
- intervention vector and magnitude;
- active-constraint count;
- normalized slack and dual statistics;
- projection Jacobian rank and singular values;
- time-to-collision and local obstacle geometry;
- LiDAR embedding;
- filter and policy version;
- recent intervention history.

The primary mode vocabulary is:

```text
NOMINAL
PROJECTED
STRONG_PROJECTED
EMERGENCY_BRAKE
```

Do not reintroduce the abandoned recovery-chain architecture. Raw active-set IDs
are not the preferred representation because they are combinatorial and depend on
constraint ordering. Continuous set/geometry summaries are the primary input.

### 4.4 Direct return model

The direct model learns

\[
Z_{E,\mathrm{direct}}^{\pi,\Pi}(x,g,c_H),
\]

where \(c_H\) is the current safety context. Candidate implementations include:

- Monte-Carlo return regression;
- scalar TD;
- standard quantile TD;
- direct history-conditioned return prediction.

Direct return learning is simple and avoids model-rollout error, but it can become
policy/filter specific and stale after the executed policy changes.

### 4.5 Factorized safety-workload model

The factorized alternative learns a stochastic model of future executed dynamics
or future safety workload:

\[
p_\psi(x_{t+1},m_{t+1},I_{t+1}
\mid x_t,m_t,I_t,g,\pi,\Pi),
\]

then performs stochastic rollouts and integrates the realized power model:

\[
E_g^{(n)}
=
\sum_{k=t}^{T_g^{(n)}-1}
P(x_k^{(n)},u_k^{\mathrm{exec},(n)},w_k^{(n)})\Delta t_k.
\]

The rollout distribution produces quantities such as

\[
q_{0.50},\ q_{0.90},\ q_{0.95},\ q_{0.99}.
\]

The model must earn its complexity by outperforming direct return learning under
declared distribution shifts. Factorized rollout is not assumed to be superior:
its errors can compound with horizon.

### 4.6 Possible intervention-history state

A candidate history state is

\[
z_{t+1}=F_z(z_t,I_t),
\]

where \(I_t\) summarizes safety intervention.

This is a hypothesis, not a physical fact. Unlike smartphone radio tail power,
the consequences of UAV braking may be fully mediated by the current position,
velocity, goal, and obstacle state. If the current observation is Markov sufficient,
intervention history should add no information.

The history state is retained only if experiments establish

\[
I(E_g;I_{t-L:t}\mid x_t,g,c_{H,t},\pi,\Pi)>0.
\]

Otherwise the project must conclude that the apparent tail is state-mediated and
remove the latent history component.

### 4.7 High-level energy decision

For charger goal \(g_c\), a conservative decision uses

\[
e_t^-
\ge
U_E^{\pi,\Pi}(x_t,g_c)+m,
\]

where \(U_E\) is a held-out calibrated or otherwise explicitly justified upper
estimate and \(m\) is a reserve.

The high-level state remains:

```text
TASK
CHARGER_COMMITTED
```

Commitment is one-way within a battery cycle. Collision constraints remain hard
constraints and are never traded against energy by a weighted scalar reward.

## 5. Research Hypotheses

### H1. Safety-context necessity

Ignoring current safety context causes systematic Energy-to-Go error when the
filter materially changes executed actions and trajectories.

### H2. Intervention-history necessity

Recent intervention history improves held-out prediction after controlling for the
current observable state.

H2 is allowed to be false. A negative result means that intervention consequences
are adequately represented by the current state.

### H3. Hybrid-structure advantage

A structured safety-mode model improves tail prediction over a parameter-matched
black-box sequence model using the same raw observations.

### H4. Direct-versus-factorized separation

Direct return learning is competitive in-distribution, while factorized safety-
workload rollout transfers better under some policy/filter/environment shifts.

This separation must be discovered empirically, not assumed.

### H5. Recent-window adaptation

Recent current-policy data repairs stale Energy-to-Go estimates more efficiently
than indiscriminately mixing trajectories from old and new policies.

### H6. Structured transfer beyond simple refitting

Safety-aware structure provides value beyond:

- current-policy windowing;
- policy-version tags;
- policy fingerprints;
- increased model capacity;
- increased training data.

### H7. Decision utility

Improved upper-tail Energy-to-Go prediction reduces energy exhaustion or return
failure without producing excessive unused battery, premature charger commitment,
or unacceptable task-throughput loss.

## 6. Theory Program

The theory should clarify semantics and failure boundaries. It must not fabricate
global convergence or unconditional physical-safety guarantees.

### T1. Executed-policy SSP semantics

Define Energy-to-Go under the executed policy \(\tilde\pi=\Pi_H\circ\pi\) as an
undiscounted stochastic-shortest-path cost. Derive

\[
Q_E^{\tilde\pi}(x,a,g)
=
\mathbb E\left[
c(x,\Pi_H(x,a))
+
V_E^{\tilde\pi}(x',g)
\right],
\]

with terminal condition

\[
V_E^{\tilde\pi}(x,g)=0
\quad\text{for }x\in G(g).
\]

State the nonnegative-cost, finite-hitting-time/proper-policy, observation, and
stationarity assumptions required for finite values.

### T2. Markov history-redundancy proposition

Prove that if:

- the physical state is Markov sufficient;
- the nominal policy and safety filter are memoryless given the state;
- step cost is state/action Markov;
- the goal and policy/filter identity are fixed;

then past intervention history is conditionally redundant for future Energy-to-Go:

\[
E_g\perp I_{0:t-1}
\mid x_t,g,\pi,\Pi_H.
\]

This proposition prevents an unjustified latent-tail claim.

### T3. Augmented-state Markovization

If perception latency, actuator dynamics, filter hysteresis, thermal state, or other
finite-memory processes violate T2, specify an augmented state

\[
\bar x_t=(x_t,z_t)
\]

and prove the conditions under which \(\bar x_t\) restores Markov semantics.

### T4. Fixed-active-set projection sensitivity

Reuse the existing fixed-active-set HOCBF-QP Jacobian derivation. State precisely
that the projection is locally affine only while the effective active set and
constraint data remain fixed. No differentiability is claimed at active-set or
action-map switching boundaries.

### T5. Immediate realized-energy sensitivity

For a quadratic acceleration-power component and a fixed active set, derive the
local signed change in immediate energy caused by a nominal-action perturbation.
Do not assume every intervention increases energy.

### T6. Factorized rollout semantics

Define the stochastic rollout distribution and show that, with the true transition,
mode, and power models, its hitting-cost distribution equals
\(Z_E^{\pi,\Pi}\).

This is a population semantics statement, not a deep-learning convergence theorem.

### T7. Finite-horizon prediction-error decomposition

Under bounded cost, Lipschitz test functions, and a finite or truncated horizon,
derive an error decomposition separating:

- instantaneous power-model error;
- executed-dynamics error;
- safety-mode/intervention-model error;
- hitting-time truncation error;
- policy/filter mismatch.

The result should explain when direct return regression can outperform rollout and
when factorization may transfer better.

### T8. Policy/filter-shift value difference

Use a cost performance-difference identity or a restricted local transport result
to characterize

\[
J_E(\pi',\Pi')-J_E(\pi,\Pi).
\]

The analysis must include future occupancy shift. A one-step projection Jacobian
alone is not a derivative of the complete hitting-cost trajectory.

### T9. Calibrated upper-bound implication

Under an explicit held-out exchangeability or calibration assumption, show that

\[
\Pr(E_g\le U_E(x,g))\ge1-\alpha_E
\]

and

\[
e_t^-\ge U_E(x_t,g_c)+m
\]

imply a corresponding bound on energy insufficiency after immediate charger
commitment. The theorem must remain conditional on calibration validity.

### T10. One-way commitment

Retain the simple latch result:

\[
N_{\mathrm{TASK}\rightarrow\mathrm{CHARGER}}
\le1
\]

per battery cycle. This prevents chattering but is not a novelty claim.

### Theory non-claims

The theory must explicitly refuse the following claims:

- arbitrary unknown-environment reachability;
- global convergence of deep TD or deep stochastic rollout models;
- global differentiability of HOCBF projection;
- recursive feasibility when the bounded HOCBF-QP is infeasible;
- real-UAV Joule/Wh accuracy before physical calibration;
- unconditional quantile coverage under policy, filter, or environment shift;
- existence of a latent intervention tail without empirical evidence;
- equivalence between low collision count and learned-policy safety.

## 7. Experimental Program

### E0. Data and semantics audit

Before training a new architecture:

- verify nominal and executed actions are both logged;
- verify realized acceleration and substep energy semantics;
- verify HOCBF slacks, duals, Jacobian diagnostics, fallback, and obstacle context;
- verify complete goal segments and true backward return-to-go labels;
- tag policy checkpoint, filter configuration, obstacle layout, and seed;
- separate training, calibration, and final evaluation streams.

### E1. Intervention-aligned consequence analysis

This is the first decisive experiment.

Align trajectories at intervention events and measure over horizons
\(1,5,10,20,50,\ldots\):

- realized power and cumulative energy;
- speed and acceleration;
- path deviation and path ratio;
- goal hitting time;
- future intervention occupancy;
- Energy-to-Go prediction residual.

Use state-matched non-intervention samples and, where valid, cloned simulator
rollouts. Report signed effects: an intervention can reduce immediate power while
increasing total energy through delay or detour.

Output gate:

```text
INTERVENTION_CONSEQUENCE_OBSERVED = TRUE/FALSE/UNCLEAR
```

### E2. History-necessity experiment

Compare parameter-matched predictors:

1. current state only;
2. current state plus current HOCBF context;
3. current state plus the last \(L\) intervention records;
4. generic recurrent model with the same history;
5. proposed structured history state.

Evaluate whether history improves held-out performance after stratifying by current
distance, velocity, obstacle geometry, and intervention severity.

Output gate:

```text
SAFETY_HISTORY_ADDS_INFORMATION = TRUE/FALSE
```

If false, remove the latent history mechanism.

### E3. Fixed-policy estimator comparison

Use one frozen navigation policy and one fixed HOCBF configuration. Compare:

- distance times empirical energy per metre;
- physical/telemetry power integration on supplied trajectories;
- supervised Monte-Carlo return regression;
- scalar TD;
- standard quantile TD;
- direct safety-context return model;
- factorized safety-workload rollout;
- oracle simulator rollout when simulator state is available.

The oracle rollout separates learned-model error from irreducible uncertainty.

### E4. Safety-mode representation ablation

Compare:

- no safety context;
- raw LiDAR/context concatenation;
- intervention magnitude only;
- slack/dual summary;
- Jacobian summary;
- continuous set-encoded constraint geometry;
- raw active-set ID as a diagnostic only;
- full proposed structured representation.

All learned baselines must be parameter-count and data-budget matched.

### E5. Distribution-shift evaluation

Train on one controlled distribution and test separately on:

- unseen static obstacle layouts;
- obstacle density and radius shift;
- charger-distance shift;
- HOCBF gain/top-K/filter-strength shift;
- sensing noise or delay;
- wind, payload, and battery-aging proxies;
- navigation-policy checkpoint shift;
- a second safety-filter family when available.

Do not combine all shifts into one aggregate score. Report which shift breaks which
model.

### E6. Recent-window and policy-aware adaptation

After changing the policy/filter pair, compare:

- frozen stale estimator;
- mixed old/new replay;
- recent current-policy window;
- explicit version-tagged replay;
- alternating current-policy refit;
- policy fingerprint;
- proposed structured transfer/update.

The policy fingerprint is retained only if it improves over simpler current-window
and version-tag baselines.

### E7. Direct-versus-factorized decision experiment

Test the central paper claim:

> Direct return learning and factorized safety-workload rollout have different
> in-distribution and shift behavior.

Required outputs:

- in-distribution accuracy;
- OOD deterioration ratio;
- adaptation sample efficiency;
- rollout horizon error growth;
- inference/runtime cost;
- tail underestimation;
- calibration stability.

### E8. Energy-management utility

Only after estimation gates pass, use the estimator for one-way charger commitment.
Compare:

- fixed SOC threshold;
- distance/energy-per-metre rule;
- direct Energy-to-Go switch;
- factorized Energy-to-Go switch;
- calibrated upper-bound variants.

Measure:

- task completions per battery cycle;
- charger arrival and return success;
- energy exhaustion/stranding;
- remaining energy at commitment and charger arrival;
- unused reserve;
- switch timing;
- task throughput;
- collision and HOCBF fallback/intervention statistics.

### E9. Generality requirement for an ICLR claim

A single UAV simulator plus one HOCBF implementation is likely a robotics/systems
paper. An ICLR-level claim should additionally demonstrate at least one of:

- a second robot/dynamics domain;
- a second optimization-based safety filter;
- a controlled general benchmark for shielded resource prediction;
- a theorem-backed separation repeatedly verified across diverse tasks.

Dynamic obstacles are a valuable robustness test but are not required before the
static-obstacle mechanism is established.

## 8. Metrics

### 8.1 Point prediction

- MAE;
- RMSE;
- median absolute error;
- signed bias;
- relative error where the denominator is stable.

### 8.2 Safety-oriented energy metrics

- underestimation rate;
- severe underestimation rate;
- worst underestimation;
- mean underestimation magnitude;
- q50/q90/q95/q99 empirical coverage;
- upper-bound violation rate;
- calibration error;
- bound width and tightness.

### 8.3 Horizon and mode diagnostics

- error by goal distance;
- error by hitting horizon;
- error by intervention magnitude;
- error by mode and mode-transition count;
- error by active-set/Jacobian regime;
- TD bootstrap drift;
- factorized rollout compounding error.

### 8.4 Navigation and safety

- task completion;
- path ratio \(=\) actual path / straight-line distance;
- completion steps and hitting time;
- collision and near-collision;
- HOCBF intervention/fallback rate and magnitude;
- nominal-safe action rate;
- boundary-contact metrics.

### 8.5 Persistent mission utility

- tasks per battery cycle;
- charger-return success;
- stranded/exhausted rate;
- unused energy at charger;
- premature-return rate;
- task throughput.

## 9. Experimental Discipline

Formal results require:

- at least three seeds, with five preferred for main tables;
- fixed held-out evaluation streams;
- equal environment and update budgets;
- no transition leakage across complete trajectories;
- immutable config and exact command;
- git SHA, code hash, checkpoint hash, and device information;
- raw JSON/JSONL plus derived tables;
- interrupted runs marked invalid for formal comparison;
- separate reporting of pilot, validation, and formal results;
- no claim based only on reward, collision count, or smoke tests.

## 10. Decision Gates

### Gate A: Does the phenomenon exist?

Proceed only if safety filtering causes a reproducible Energy-to-Go error pattern
or tail change beyond ordinary distance/horizon effects.

### Gate B: Is history necessary?

Retain the latent intervention-history state only if it improves held-out prediction
after controlling for the current state.

### Gate C: Does structured modeling add value?

Proceed only if the structured model beats parameter-matched raw-context and generic
sequence baselines, not merely a context-free critic.

### Gate D: Does it beat simple adaptation?

The method must improve over recent-window retraining and policy/filter version
tagging. Otherwise the complex representation is unnecessary.

### Gate E: Does prediction improve decisions?

Better MAE alone is insufficient. The estimator must improve tail reliability or
charger-return utility without unacceptable conservatism.

### Gate F: Is the result ICLR-level?

An ICLR submission requires a general learning insight, a defensible novelty delta,
and evidence beyond one hand-designed UAV configuration. If this gate fails but the
system result is strong, redirect to ICRA/IROS/RA-L or an autonomous-systems venue.

## 11. Expected Contribution if All Gates Pass

The strongest defensible contribution would be:

1. identification of a reproducible safety-filter-induced failure mode in long-
   horizon resource prediction;
2. a structured stochastic safety-workload representation based on executed-action
   and projection geometry rather than simple feature concatenation;
3. a direct-versus-factorized analysis explaining in-distribution accuracy,
   transfer, and tail reliability;
4. a policy/filter adaptation protocol that prevents stale Energy-to-Go semantics;
5. evidence that improved estimation produces safer and less conservative charger
   decisions;
6. theory limited to executed-policy SSP semantics, Markov/history conditions,
   local projection geometry, rollout-error decomposition, and conditional
   calibration implications.

## 12. Explicit Failure/Pivot Outcomes

The project must accept the following outcomes:

- If distance/power integration remains competitive, TD is not the primary method.
- If intervention history adds no information, remove the latent tail state.
- If factorized rollout suffers larger compounding error, retain direct return
  learning and report the negative result.
- If current-policy recent-window refitting matches the proposed transfer method,
  do not claim a new policy-aware representation.
- If upper-tail calibration fails under shift, do not call the system energy safe.
- If the contribution remains UAV/HOCBF specific, pursue a robotics or autonomous-
  systems paper rather than forcing an ICLR framing.

## 13. Immediate Next Work

The next work should be performed in this order:

1. freeze and audit the existing HOCBF trajectory logs;
2. implement intervention-aligned consequence analysis;
3. test state-only versus safety-context versus intervention-history prediction;
4. decide whether a latent history state is justified;
5. implement the smallest direct and factorized baselines;
6. run fixed-policy held-out comparison;
7. run obstacle/filter/policy shift tests;
8. only then design online adaptation and charger switching experiments;
9. perform a new closest-work and hostile-review gate before paper framing.

Do not begin with a new large network or another formal million-step run. The first
scientific decision is whether safety intervention creates a predictable long-
horizon energy phenomenon that is not already explained by the current Markov
state and simple baselines.

## 14. Relationship to Existing Repository Documents

- `docs/JACOBIAN_SAFETY_ENERGY_BRIDGE.md`: current Jacobian-based implementation,
  pilot evidence, and non-claim boundary.
- `docs/uav_safety_energy_theory_derivation.md`: HOCBF, sampled-data, and energy-
  objective derivations.
- `docs/uav_safety_energy_method_comparison.md`: existing filter baselines and
  Candidate A/B/C decisions.
- `docs/policy_aware_uav_energy_estimation_literature_review.md`: closest energy,
  policy-aware, and two-timescale literature.
- `docs/energy_td_root_cause_report.md`: prior TD semantic and bootstrap failure
  analysis.

This document is the current roadmap. Existing documents remain evidence and
provenance; they must not be interpreted as proof that the new safety-workload
hypotheses are already established.

## 15. ICLR Oral-Level Interface-Support Upgrade

### 15.1 Corrected central claim

The stronger research target is not merely that a modular model can recombine a
seen policy and a seen safety filter. If the exact interface is known, this can be
an obvious application of a standard executed-action world model.

The corrected central question is:

> What is the minimal identifiable interface for compositional long-horizon
> resource prediction under policy-dependent safety intervention, and when does
> this interface permit reliable distributional prediction without pair-level
> closed-loop trajectory overlap?

The intended claim is therefore:

```text
pair-level trajectory support is not always necessary,
but target executed-interface support remains necessary
unless additional extrapolation structure is assumed.
```

### 15.2 Formal trajectory law

Let \(x_t\) be the physical state, \(o_t\) the observation, \(a_t^{\rm nom}\)
the nominal action, \(u_t^{\rm exec}\) the executed action, \(\xi_t\) the
geometry/perception/filter context, and \(c_t\) the realized resource cost. The
closed-loop trajectory law is

\[
\rho_0(dx_0)
\prod_{t\ge0}
O_{\mathcal E}(do_t\mid x_t)
\pi(da_t^{\rm nom}\mid o_t,g)
K_{\Pi_\phi}(du_t^{\rm exec}\mid x_t,a_t^{\rm nom},\xi_t)
P_\omega(dx_{t+1},dc_t\mid x_t,u_t^{\rm exec}).
\]

For a deterministic, fully observed HOCBF-QP,

\[
K_{\Pi_\phi}
=
\delta_{\Pi_\phi(x_t,a_t^{\rm nom},\xi_t)}.
\]

Geometry/context shift and physical shift must remain distinct:

- obstacle layout, density, and perception modify \(\xi_t\), filter behavior,
  and closed-loop occupancy while plant dynamics may remain shared;
- wind, payload, actuator efficiency, and battery aging modify
  \(P_\omega(x',c\mid x,u)\).

The first paper should prioritize policy-filter composition under shared plant
dynamics. Physical-context composition is a declared extension, not silently
included in the same invariance assumption.

### 15.3 Interface-support identifiability proposition

For a target pair \((\pi^\star,\Pi^\star)\), define the executed closed-loop
kernel

\[
K_{\mathcal E}^{\pi^\star,\Pi^\star}(dx',dc\mid x)
=
\int
\pi^\star(da^{\rm nom}\mid o,g)
K_{\Pi^\star}(du^{\rm exec}\mid x,a^{\rm nom},\xi)
P_{\mathcal E}(dx',dc\mid x,u^{\rm exec}).
\]

The target Energy-to-Go distribution is identifiable without observing complete
trajectories from the target pair only under conditions including:

1. \(\pi^\star\) is known, queryable, or consistently estimated;
2. \(K_{\Pi^\star}\) is known, queryable, or consistently estimated;
3. the shared executed dynamics/cost interface \(P_{\mathcal E}\) is identifiable
   on the support induced by the target occupancy
   \(d^{\pi^\star,\Pi^\star}(x,u^{\rm exec})\);
4. the interface is genuinely invariant across the policy/filter pairs being
   composed;
5. the target SSP is proper and its hitting-cost distribution exists.

The support condition must be explicit:

\[
\operatorname{supp}
d^{\pi^\star,\Pi^\star}(x,u^{\rm exec})
\subseteq
\operatorname{supp}
d_{\rm identifiable}(x,u^{\rm exec}).
\]

Thus, seeing each policy and filter component separately is insufficient. A new
composition can induce a previously unseen state-executed-action region. Without
interface support or additional smoothness/model assumptions, its Energy-to-Go is
not identifiable.

### 15.4 Required strongest baseline

The primary baseline is a generic executed-action world model:

\[
\widehat P(x_{t+1},c_t\mid x_t,u_t^{\rm exec}).
\]

At evaluation it invokes the actual policy and filter:

\[
a_t^{\rm nom}\sim\pi,
\qquad
u_t^{\rm exec}=\Pi(x_t,a_t^{\rm nom},\xi_t),
\]

then rolls out \(\widehat P\). It receives no Jacobian, raw active-set ID, or
special safety representation.

If this baseline matches the proposed operator-aware method, the defensible result
is that the executed-action interface itself is sufficient. The paper must not
claim that a specialized safety representation caused compositional
generalization. Algorithmic novelty then requires either discovering a more useful
interface, improving uncertainty under weak support, or reducing target adaptation
data beyond this baseline.

### 15.5 Distribution-level theory requirement

The mean SSP transport identity is useful but insufficient because the empirical
target is the Energy-to-Go distribution and upper-tail quantities. For a bounded
or suitably transient SSP, the theory should seek a distributional perturbation
bound such as

\[
W_1(Z_E',Z_E)
\le
C_{\rm dyn}\epsilon_{\rm dyn}
+C_{\Pi}\epsilon_{\Pi}
+C_{\pi}\epsilon_{\pi}
+C_c\epsilon_c
+\epsilon_{\rm trunc}.
\]

Every constant and metric requires explicit regularity, horizon, and properness
conditions. A small Wasserstein distance does not by itself guarantee a small
q99 error. Quantile error requires an additional local anti-concentration or
positive-density condition near the target quantile; otherwise the theory should
stop at distributional distance and calibrated coverage statements.

### 15.6 Clean compositional experiments

The first pilot should use two policies and two filters under a fixed plant. For
each obstacle distribution, train on three policy-filter pairs and hold out the
fourth, rotating all four possible held-out pairs rather than selecting one
favorable split.

For a genuine three-factor diagnostic, use the parity split over
\((\pi,\Pi,\mathcal E)\in\{0,1\}^3\):

```text
train: 000, 011, 101, 110
test:  001, 010, 100, 111
```

Each individual component and every pairwise projection appears in training, but
each test triple is unseen. This tests higher-order composition rather than simple
component OOD. Geometry-context and physical-dynamics shifts must not be mixed in
the same first experiment.

Required baselines are:

1. distance times energy per metre;
2. direct MC and quantile return models;
3. direct safety-context return model;
4. generic recurrent/history model;
5. policy-conditioned model;
6. pair-conditioned policy-filter black-box model;
7. nominal-action world model;
8. generic executed-action world model;
9. proposed operator-aware factorized model;
10. recent-window adaptation;
11. oracle simulator rollout.

### 15.7 Oral-level decision gates

An oral-candidate framing requires all of the following empirical outcomes:

1. a reproducible long-horizon resource failure associated with safety-filtered
   closed-loop shift after controlling for ordinary geometry and horizon;
2. a clear identification of which interface is sufficient and where target
   support failure breaks prediction;
3. improvement over the generic executed-action world model and policy-conditioned
   model, or a scientifically strong result showing that the simple interface is
   already sufficient;
4. distribution-level evidence on q95/q99 underestimation and calibration, not
   only mean MAE;
5. held-out compositional generalization and few-shot recovery across multiple
   policies, filters, seeds, and at least two dynamics domains;
6. theory and experiments using the same distributional target and support
   assumptions;
7. a benchmark/protocol that makes policy-filter composition and interface support
   measurable beyond the UAV application.

Failure to beat the executed-action world model is not a failed experiment, but it
changes the contribution from a new operator-aware algorithm to a minimal-interface
or benchmark result. Failure to show cross-domain regularity should trigger a
robotics/autonomous-systems framing rather than an ICLR oral claim.

### 15.8 Claim-oriented AI baseline matrix

Baselines must be selected by the scientific claim they attack. The project must
not run unrelated top-conference methods merely to create a long comparison table.

| Claim under test | AI method family | Required empirical baseline or comparison |
|---|---|---|
| Policy-induced model shift | Policy-conditioned modeling | PCM-style policy-conditioned dynamics/cost model |
| Compositional generalization | Modular/causal world models | Empirical adaptation where assumptions match; otherwise theorem-level comparison to modular solutions and WM3C |
| Interface sufficiency | Generic model learning | Nominal-action and executed-action probabilistic world models |
| Unsupported rollout reliability | Pessimistic/uncertainty-aware model learning | Ensemble-UQ, MOPO-style uncertainty inflation, MOReL-style rejection, COMBO-inspired conservative correction adapted to prediction rather than policy optimization |
| Return-distribution representation | Distributional RL | Fixed quantile regression plus IQN-style direct Energy-to-Go; FQF-style model only if needed |
| Coverage and selective reliability | Conformal/selective prediction | CQR, weighted conformal under valid covariate-shift assumptions, and risk-coverage/selective-prediction baselines |

MOPO, MOReL, and COMBO are not headline same-task algorithms because their native
objective is offline policy optimization. Their support-handling principles should
be adapted into prediction baselines with identical data, models, target policies,
and executed-action access.

All baselines that can legitimately query the known target policy and safety
operator must receive that same access. The proposed method may not win merely
because it is given the real HOCBF operator while competitors receive only a filter
ID.

### 15.9 Three-regime experimental law

The central experiment must independently control:

\[
\text{pair novelty}
\times
\text{interface support}
\times
\text{hitting horizon}.
\]

The desired scientific output is a regime map rather than one aggregate ranking:

1. **Pair-OOD, interface-ID:** unseen policy-filter pair, but target executed
   occupancy remains covered. A correct interface model should transfer.
2. **Interface boundary:** target rollouts approach weakly covered regions. Error
   and uncertainty should rise in a detectable way.
3. **Interface OOD:** target rollout leaves identifiable support. Structure-free
   zero-shot reliability is unavailable; a valid system must abstain, widen its
   bound under stated assumptions, or collect adaptation data.

The first pilot uses only five decisive baselines:

1. direct quantile Energy-to-Go;
2. PCM-style model;
3. generic executed-action probabilistic world model;
4. executed-action world model plus ensemble uncertainty;
5. support-aware interface world model.

The full baseline suite is justified only after this pilot demonstrates a stable
separation.

### 15.10 Reliability semantics and calibration boundaries

An operational support score is not identical to mathematical support. In a
continuous high-dimensional space, density, nearest-neighbor distance, latent
distance, ensemble disagreement, local error prediction, and conformal
nonconformity measure different objects. The method must determine which score
predicts actual rollout or Energy-to-Go failure.

Do not use

\[
1-\prod_t(1-r_t)
\]

as a trajectory failure probability unless the required conditional independence
or calibrated hazard assumptions are established. Safer initial alternatives are
an explicit union bound, maximum/soft-maximum risk score, or a separately trained
trajectory-level reliability model evaluated on held-out complete trajectories.

CQR provides finite-sample marginal coverage under exchangeability. Weighted
conformal methods under covariate shift require the relevant weighted
exchangeability/likelihood-ratio assumptions and approximately invariant
conditional outcome law. A policy-filter change can alter the conditional
Energy-to-Go law itself, so these guarantees must not be transferred to general
composition shift. In that regime conformal methods are baselines and diagnostics,
not automatically valid certificates.

Required selective-prediction metrics include:

- risk-coverage curves and area under the risk-coverage curve;
- q95/q99 underestimation conditioned on accepted predictions;
- abstention/unknown rate;
- interval width and empirical coverage;
- adaptation data required to move an abstained region into the accepted region.

## 16. Selected External References

- Smart Battery Data Specification: present-rate versus rolling-average time-to-
  empty semantics: <https://sbs-forum.org/specs/sbdata10.pdf>
- Li et al., smartphone lifetime prediction using current-session and historical
  usage: <https://arxiv.org/abs/1801.04069>
- Google Pixel Adaptive Battery behavior: <https://support.google.com/pixelphone/answer/7015477?hl=en>
- Apple Adaptive Power behavior: <https://support.apple.com/zh-cn/123707>
- Smartphone stochastic hybrid TTE model: <https://arxiv.org/abs/2605.09367>
- Smartphone instantaneous/delayed drain decomposition: <https://www.mdpi.com/2079-9292/15/15/3257>
- Choudhry et al., UAV deep energy model and CVaR risk:
  <https://arxiv.org/abs/2105.15189>
- Fouad et al., CBF energy sufficiency:
  <https://arxiv.org/abs/2306.15115>
- Policy-conditioned environment models:
  <https://proceedings.mlr.press/v235/chen24g.html>
- Schug et al., modular solutions and compositional identification, ICLR 2024:
  <https://openreview.net/forum?id=H98CVcX1eh>
- Wang and Huang, composable causal world models, ICLR 2025:
  <https://proceedings.iclr.cc/paper_files/paper/2025/hash/79d86433c2acd12b6fa98553435d226e-Abstract-Conference.html>
- MOPO, uncertainty-penalized model-based offline RL:
  <https://proceedings.neurips.cc/paper/2020/hash/a322852ce0df73e204b7e67cbbef0d0a-Abstract.html>
- MOReL, unknown-region pessimism:
  <https://www.microsoft.com/en-us/research/publication/morel-model-based-offline-reinforcement-learning/>
- COMBO, conservative model-based offline RL:
  <https://proceedings.neurips.cc/paper/2021/hash/f29a179746902e331572c483c45e5086-Abstract.html>
- Implicit Quantile Networks:
  <https://proceedings.mlr.press/v80/dabney18a.html>
- Fully Parameterized Quantile Functions:
  <https://proceedings.neurips.cc/paper/2019/hash/f471223d1a1614b58a7dc45c9d01df19-Abstract.html>
- Conformalized Quantile Regression:
  <https://proceedings.neurips.cc/paper/2019/hash/5103c3584b063c431bd1268e9b5e76fb-Abstract.html>
- Conformal prediction under covariate shift:
  <https://proceedings.neurips.cc/paper/2019/hash/8fb21ee7a2207526da55a679f0332de2-Abstract.html>
- SelectiveNet and risk-coverage evaluation:
  <https://proceedings.mlr.press/v97/geifman19a.html>
- Safe RL using intervention:
  <https://proceedings.mlr.press/v139/wagener21a.html>
