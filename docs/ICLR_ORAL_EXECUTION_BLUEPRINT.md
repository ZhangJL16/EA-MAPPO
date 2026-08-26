# ICLR Oral-Level Execution Blueprint

> **Framing update:** the canonical top-level protocol is now
> `docs/RETURN_TO_CHARGE_ICLR_RESEARCH_PROTOCOL.md`. Selective Resource-to-Go
> prediction remains the ML mechanism, while the final task is the irreversible
> continue-versus-return-to-charge decision and its stranding--throughput frontier.

## 1. Verdict

The current direction is scientifically promising, but oral-level evidence is
conditional on a decisive pilot. The primary ML problem is:

> **Selective distributional prediction under compositional closed-loop shift:**
> when a policy and a safety operator jointly generate the executed trajectory,
> when is the long-horizon return distribution of an unseen composition still
> identifiable, and can a predictor detect loss of reliability before target
> returns are observed?

UAV Energy-to-Go is the primary testbed. HOCBF remains the hard collision-safety
layer. The learned return and reliability models are not safety certificates.

## 2. One-Sentence Falsifiable Claim

> Unseen policy-filter pairs need not cause long-horizon prediction failure when
> their target trajectories remain covered through an identifiable executed-action
> interface; failure is instead governed by interface extrapolation and hitting
> horizon, and these factors can support selective prediction and efficient target
> adaptation.

The claim is falsified if pair identity predicts failure better than interface
extrapolation, if reliability scores do not rank actual return error, if a generic
ensemble already matches the proposed method, or if adaptation gains do not repeat
across compositions and domains.

## 3. Critical Missing Definition: What Is Random?

Distributional Energy-to-Go requires an explicit probability space. In a fully
deterministic simulator with deterministic policy, static obstacles, deterministic
HOCBF, deterministic plant dynamics, and fixed initial state, Energy-to-Go is a
point mass. Quantile modeling would then represent estimator uncertainty rather
than an aleatoric return distribution.

The project must separate:

### 3.1 Aleatoric return uncertainty

Randomness in the physical or closed-loop trajectory, such as:

- stochastic initial conditions;
- wind or disturbance realizations;
- sensing and actuation noise;
- moving-obstacle behavior;
- stochastic policy actions when declared;
- payload or operating-context variation drawn from a declared distribution.

For model parameter \(\theta\), this defines

\[
Z_E\mid\theta
=
\mathcal L\left[
\sum_{t<T_g}c_t
\middle|
\pi,\Pi,\omega
\right].
\]

### 3.2 Epistemic model uncertainty

Uncertainty caused by finite data, interface extrapolation, model misspecification,
or an unseen composition. An ensemble approximates uncertainty over model
parameters or predictions:

\[
\Theta\mid\mathcal D.
\]

The aleatoric q95 of \(Z_E\mid\theta\) and an epistemic upper confidence band are
different quantities and must be logged, evaluated, and calibrated separately.

### 3.3 Headline quantiles

Use q90 and q95 as primary tail metrics. Treat q99 as supplementary unless each
condition has enough independent stochastic rollouts to estimate it reliably.
Include CRPS for full-distribution quality and pinball loss for quantile quality.

## 4. Scientific Positioning

| Layer | Position |
|---|---|
| Primary field | Sequential prediction and model-based RL under structured distribution shift |
| Core problem | Compositional generalization, distributional return prediction, selective prediction |
| Adjacent fields | Offline/model-based uncertainty, off-policy evaluation, risk-sensitive RL, CMDP |
| Application | Resource-aware autonomous systems |
| Main testbed | Safety-filtered UAV Energy-to-Go |
| Downstream utility | Charger commitment and mission completion |

CMDP methods optimize policies under constraints. The core paper instead predicts
the return distribution of a fixed policy-filter composition and decides whether
that prediction is trustworthy. CMDP algorithms belong in the downstream decision
table, not the core prediction table.

## 5. Final Method Skeleton: SIRP

`SIRP` is a provisional name for **Selective Interface Return Prediction**. A
novelty/name search is required before publication.

### 5.1 Executed-interface probabilistic model

\[
a_t^{\rm nom}\sim\pi,
\qquad
u_t^{\rm exec}=\Pi(x_t,a_t^{\rm nom},\xi_t),
\]

\[
(x_{t+1},c_t)
\sim
\widehat K_\theta(
x_{t+1},c_t
\mid
x_t,u_t^{\rm exec},w_t).
\]

Stochastic rollout to the goal produces \(\widehat Z_E\). Every legitimate
model-based baseline receives the same target policy, safety operator, executed
actions, data budget, and rollout budget.

### 5.2 Local interface-error estimator

Theory should produce a local error term \(\epsilon_K(x,u)\). The practical model
estimates it through cross-fitted source-composition residuals using candidate
signals such as:

- ensemble disagreement;
- data density or action coverage;
- latent/interface distance;
- one-step dynamics and cost residual predictors;
- operator-induced target occupancy;
- hitting-horizon and truncation diagnostics.

The method must not equate any single proxy with mathematical support.

### 5.3 Target-occupancy-weighted reliability

The bound-inspired trajectory score is

\[
\widehat R_H
=
\sum_{t=0}^{H-1}
\mathbb E_{\widehat d_t^{\pi^\star,\Pi^\star}}
[\widehat\epsilon_K(x_t,u_t)]
+\widehat\epsilon_{\rm trunc}(H).
\]

Do not interpret \(1-\prod_t(1-r_t)\) as a probability without conditional hazard
assumptions. Additive aggregation matches the planned error decomposition.

### 5.4 Selective prediction

The output is \((\widehat Z_E,\widehat R_H)\):

```text
low reliability risk    -> emit distribution
intermediate risk       -> emit with wider epistemic band
high risk               -> abstain and request target adaptation
```

No unconditional target-domain selective-risk guarantee is claimed. Formal
coverage requires declared calibration data and exchangeability/transport
assumptions.

## 6. Theory: Four Theorems and Two Corollaries

### Theorem 1. Interface-compositional identifiability

If the target policy and safety operator are known/queryable, the shared
executed-action kernel is identified over the target occupancy support, and the SSP
is proper, then the target hitting-cost distribution is identifiable even when the
policy-filter pair was never jointly observed.

### Theorem 2. Interface-support necessity

If the target composition reaches an unobserved executed-interface region, construct
two primitive kernels that agree on all observed data but induce different target
Energy-to-Go distributions. Pair novelty is not the fundamental obstruction;
interface non-identifiability is.

### Theorem 3. Finite-horizon distributional transport

For an \(H\)-step truncated SSP, derive under explicit regularity assumptions:

\[
D(Z_{E,H},\widehat Z_{E,H})
\le
C
\sum_{t<H}
\mathbb E_{d_t^{\pi,\Pi}}
[\epsilon_K(x_t,u_t)]
+\epsilon_\pi+\epsilon_\Pi+\epsilon_c.
\]

This theorem motivates occupancy-weighted local error rather than generic Euclidean
OOD distance.

### Theorem 4. Proper-SSP truncation extension

Under bounded cost and a hitting-time tail condition, control

\[
D(Z_E,Z_{E,H})
\]

using a term such as \(c_{\max}\mathbb E[(T_g-H)_+]\). The total error separates
interface prediction, horizon amplification, and hitting-time truncation.

### Corollary 1. Quantile stability

Quantile error requires a CDF-distance result plus local anti-concentration or a
positive-density condition near q90/q95. Wasserstein closeness alone does not imply
stable q99.

### Corollary 2. Energy-decision implication

Conditional on valid calibration,

\[
\Pr(E_g\le U_E)\ge1-\alpha,
\qquad
e_t^-\ge U_E+m
\]

imply the corresponding bound after immediate charger commitment. No unconditional
coverage is claimed under arbitrary composition shift.

## 7. Claim-Oriented Baseline Matrix

### 7.1 Core prediction table

| Claim | Decisive competitor |
|---|---|
| Direct distributional return | MC quantile, IQN; FQF if pilot survives |
| Policy shift | PCM |
| Pair composition | Pair-conditioned and matched modular model |
| Executed interface | Generic executed-action probabilistic world model |
| Epistemic reliability | Deep Ensemble |
| Unsupported rollout | MOPO/MOReL/COMBO-inspired prediction mechanisms |
| Selectivity | Ensemble threshold and SelectiveNet-style rejector |
| Calibration | CQR and weighted conformal only under valid assumptions |
| Adaptation | Equal-budget recent-window refit |

### 7.2 Downstream decision table

Only after prediction gates pass, compare charger/resource decisions against fixed
SOC/distance rules, CVaR energy planning, and selected CMDP/risk-sensitive methods
such as CPO or SDAC. Do not compare prediction MAE directly with a policy-optimization
algorithm.

## 8. Decisive Pilot

### 8.1 Methods

Run only:

1. IQN-style direct Energy-to-Go;
2. PCM-style model;
3. generic executed-action probabilistic world model;
4. executed-action world model plus Deep Ensemble;
5. SIRP prototype.

### 8.2 Composition protocol

Use \(2\) policies by \(2\) safety filters. Rotate all four leave-one-pair-out
splits. For every held-out pair, construct test regimes with high, medium, and low
executed-interface extrapolation while keeping ordinary task difficulty controlled.

### 8.3 Two notions of interface reliability

For scientific diagnosis, simulator access may define an **oracle interface
coverage/error variable**. The deployed method must use only a **learned reliability
proxy**. Do not report oracle support as if the algorithm observed it.

### 8.4 Metrics

- CRPS;
- q90/q95 pinball loss and empirical coverage;
- severe underestimation;
- Wasserstein or energy distance;
- risk-coverage curve and AURC;
- abstention rate;
- horizon-binned error;
- adaptation trajectories required to recover a fixed risk level.

### 8.5 Kill criteria

Stop the oral-method route if any condition persists across the pilot:

- executed-action world model cannot transfer in the pair-OOD/interface-ID regime;
- ensemble matches SIRP on CRPS, q95 risk, AURC, and adaptation efficiency;
- learned reliability does not rank actual distributional error;
- apparent return distribution is degenerate or only epistemic;
- the three regimes cannot be separated after controlling distance and horizon.

## 9. Full Evidence Package After Pilot Survival

1. Expand the AI baseline suite by claim.
2. Add a second dynamics domain with natural resource cost.
3. Add a second safety-filter family, not merely another HOCBF gain.
4. Run the \(2\times2\times2\) parity split for higher-order composition.
5. Use at least three seeds, with five for headline results.
6. Add few-shot target adaptation under equal data budgets.
7. Run downstream charger/resource decisions only after estimator reliability passes.
8. Release a benchmark independently controlling pair novelty, interface
   extrapolation, and hitting horizon.

## 10. Oral-Level Evidence Standard

The project becomes an oral candidate only if it establishes all of:

1. a cross-domain empirical law that pair novelty is a weak predictor of failure
   relative to interface extrapolation and hitting horizon;
2. a minimal-interface identifiability and impossibility theory aligned with the
   experiment;
3. a reliability method that beats executed-action world model plus Deep Ensemble;
4. selective distributional improvements in CRPS, q95 risk, and AURC;
5. reproducibility across multiple policies, filters, environments, and seeds;
6. a reusable benchmark and fair information access for every competitor;
7. downstream utility without claiming that the predictor itself is a safety
   certificate.

No experiment can guarantee an oral decision. This document defines an oral-level
evidence standard, not an acceptance probability.

## 11. Immediate Work Order

1. Freeze the research statement and probability-space definition.
2. Audit whether current simulator trajectories contain genuine aleatoric variation.
3. Define oracle interface coverage for analysis only.
4. Build the five minimal pilot baselines with matched information.
5. Generate all four leave-one-pair-out splits.
6. Construct high/medium/low interface-extrapolation test regimes.
7. Run the pilot before adding domains, filters, or full baseline suites.
8. Continue only if the kill criteria are passed.
