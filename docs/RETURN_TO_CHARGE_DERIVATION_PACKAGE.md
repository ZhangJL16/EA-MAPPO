# Derivation Package: Reliable Return-to-Charge

## Target

Derive a coherent theory line connecting:

1. a policy and a hard safety operator;
2. the executed closed-loop Resource-to-Go to a charger;
3. prediction under an unseen policy--filter composition;
4. the irreversible `CONTINUE` versus `RETURN` decision.

The immediate target is a set of paper-ready theorem statements and proof
obligations. This document does **not** claim that all statements already have
complete proofs or that the current deterministic simulator supplies a
non-degenerate conditional return distribution.

## Status

**COHERENT AFTER REFRAMING / EXTRA ASSUMPTION**

The theory is coherent when the invariant object is the charger hitting-resource
of the executed closed loop, not a generic q95 output. Identifiability requires
target executed-interface coverage. Distributional transport requires an explicit
regularity/coupling condition. A return-failure statement additionally requires a
calibrated upper requirement and a bound on decision-interval overshoot.

## Invariant Object

For charger set \(G_C\), define the hitting time and accumulated resource

\[
T_C=\inf\{t\ge 0:x_t\in G_C\},
\qquad
Z_C=\sum_{t=0}^{T_C-1}c_t.
\]

The distribution

\[
\mathcal Z_C^{\pi,\Pi}(x)
=
\mathcal L(Z_C\mid x_0=x,\pi,\Pi)
\]

is the top-level prediction object. In a deterministic system it is a point mass
and the relevant learned uncertainty is epistemic. In a declared stochastic
system it may also have an aleatoric tail.

The decision object is the one-way stopping rule

\[
d_t\in\{\textsf{CONTINUE},\textsf{RETURN}\},
\qquad
d_t=\textsf{RETURN}
\Longrightarrow
d_{t+k}=\textsf{RETURN}\quad(k\ge 0)
\]

until the charger is reached and a new battery cycle begins.

## Assumptions

- **A1 (deployment information).** \(x_t\) contains the information available to
  the predictor at deployment. Randomness remaining after conditioning on this
  information is aleatoric; uncertainty caused by finite data or extrapolation is
  epistemic.
- **A2 (queryable components).** The target navigation policy \(\pi\) and safety
  operator \(\Pi\) are known or can be queried without receiving target return
  labels.
- **A3 (executed interface).** Nominal action \(a\), executed action \(u\), and the
  primitive transition/resource kernel satisfy
  \[
  a\sim\pi(\cdot\mid x),\quad
  u\sim K_\Pi(\cdot\mid x,a),\quad
  (x',c)\sim K(\cdot\mid x,u).
  \]
- **A4 (shared primitive).** Across the policy--filter compositions considered in
  the first theory result, \(K\) is shared. A physics or battery-model shift is a
  separate extension and cannot be hidden inside a policy/filter composition.
- **A5 (proper SSP with first moment).** \(T_C<\infty\) almost surely under the
  target composition, \(\mathbb E[T_C]<\infty\), and
  \(0\le c_t\le c_{\max}<\infty\). The finite first moment is needed for finite
  \(W_1\) statements, not merely for almost-sure goal reaching.
- **A6 (interface identifiability).** For the positive result, \(K\) is identified
  on the support of the target occupancy over \((x,u)\). Merely observing each
  policy and each filter somewhere is not sufficient.
- **A7 (transport regularity).** For the finite-horizon error result, one-step
  target/model couplings and future-return Lipschitz moduli exist as stated below.
  This is an additional theorem assumption, not a consequence of A1--A6.
- **A8 (decision timing).** Return decisions are evaluated at declared policy-step
  intervals. Any energy consumed and requirement drift between checks must be
  bounded when a feasibility implication is claimed.
- **A9 (calibration).** Any probabilistic return guarantee is conditional on an
  explicitly valid calibration statement for the deployed composition or accepted
  subset. No arbitrary-shift coverage guarantee is assumed.

## Notation

- \(x\): Markov deployment state or declared sufficient information state.
- \(a\): nominal action sampled from the navigation policy.
- \(u\): action executed after the safety operator.
- \(K_\Pi(du\mid x,a)\): deterministic or stochastic safety-operator kernel.
- \(K(dx',dc\mid x,u)\): shared executed-action transition/resource kernel.
- \(Q^{\pi,\Pi}\): closed-loop kernel obtained by composing \(\pi\), \(K_\Pi\),
  and \(K\).
- \(d_t^{\pi,\Pi}(x,u)\): target executed-interface occupancy at time \(t\).
- \(b_t\): remaining resource; \(m\): reserve.
- \(U_C(x)\): exact or conservative charger Resource-to-Go requirement.
- \(\widehat U_C(x)\): learned requirement used by the ReturnManager.
- \(H\): finite rollout horizon; \(Z_{C,H}=\sum_{t<\min(T_C,H)}c_t\).
- \(W_1\): Wasserstein-1 distance when its moment conditions hold.

## Derivation Strategy

Compose the known policy and safety operator through the executed-action
interface, identify the resulting path law, then map path-law error to accumulated
resource error. Finally, treat the ReturnManager as a threshold classifier and
derive exactly where prediction error can change its decision. Truncation and
probabilistic calibration enter as separate assumptions rather than being hidden
inside a generic “q95” label.

## Derivation Map

1. Define the closed-loop kernel by exact composition of \(\pi\), \(K_\Pi\), and
   \(K\).
2. Show that identification of \(K\) over target executed occupancy identifies the
   target path law and hence the hitting-resource law.
3. Construct two observationally equivalent primitive kernels outside source
   interface support to prove the necessity boundary.
4. Use a stepwise coupling and a Lipschitz future-return condition to unroll a
   finite-horizon \(W_1\) error recurrence.
5. Bound infinite-horizon truncation separately by the hitting-time tail.
6. Compare oracle and learned threshold decisions algebraically; disagreement is
   confined to a boundary band proportional to requirement error.
7. Add calibration and decision-interval overshoot only for a conditional return
   failure implication.
8. Compose task and return component bounds with a union bound; never relabel
   \(q_{.95}(X)+q_{.95}(Y)\) as \(q_{.95}(X+Y)\).

## Main Derivation

### Step 1. Closed-loop executed kernel — identity

The one-step closed-loop kernel is

\[
Q^{\pi,\Pi}(dx',dc\mid x)
=
\int \pi(da\mid x)
\int K_\Pi(du\mid x,a)
K(dx',dc\mid x,u).
\]

For a deterministic policy and filter, the two inner measures reduce to Dirac
measures. This identity makes the scientific interface explicit: the primitive
plant/resource model receives \(u\), not the nominal \(a\).

### Step 2. Interface-compositional identifiability — proposition

Let the target pair \((\pi^\star,\Pi^\star)\) be unseen jointly. If A2--A6 hold,
then \(Q^{\pi^\star,\Pi^\star}\) is identified on every state reached before
\(T_C\). Consequently, every finite path-cylinder probability is identified.
Under A5, the induced hitting-resource law

\[
\mathcal Z_C^{\pi^\star,\Pi^\star}(x)
\]

is identified by the limit of its truncated path laws.

The key conclusion is limited but useful:

\[
\boxed{\text{pair-level trajectory overlap is not necessary}}
\]

provided target **executed-interface** support is identified. Component identity
coverage alone does not imply this condition.

### Step 3. Interface-support necessity — counterexample proposition

Suppose the target occupancy reaches a measurable interface region \(B\) with
positive probability, while the source data assign zero probability to \(B\).
Construct \(K_1\) and \(K_2\) that agree outside \(B\). On \(B\), let both kernels
transition to the charger but assign different bounded terminal costs
\(c_1\ne c_2\). Then the complete source-data laws are identical under \(K_1\)
and \(K_2\), while the target hitting-resource laws differ with positive
probability.

Therefore no source-observation-only estimator can uniformly identify the target
Resource-to-Go over this class:

\[
\boxed{\text{interface-OOD is an identification obstruction.}}
\]

This is an impossibility boundary, not a claim that every practical OOD point must
fail.

### Step 4. Finite-horizon transport — proposition with extra regularity

Let \(e_t(x)=W_1(\mathcal Z_{t:H}(x),\widehat{\mathcal
Z}_{t:H}(x))\). Assume a coupling of target and learned one-step kernels such that

\[
\mathbb E\left[
|c-\widehat c|+L_{t+1}d(x',\widehat x')
\mid x,u
\right]
\le \epsilon_K(x,u),
\]

where \(L_{t+1}\) is a valid Lipschitz modulus for the learned/target future-return
map. Using the Bellman path decomposition and the coupling definition of \(W_1\),

\[
e_t(x)
\le
\mathbb E_{a,u}[\epsilon_K(x,u)]
+
\mathbb E_{x'}[e_{t+1}(x')].
\]

Unrolling gives

\[
W_1(\mathcal Z_{C,H},\widehat{\mathcal Z}_{C,H})
\le
\sum_{t=0}^{H-1}
\mathbb E_{d_t^{\pi,\Pi}}[\epsilon_K(x_t,u_t)],
\]

for the exact-query setting of A2. Approximate policy or filter models require
their own coupling bounds and may add further terms, but those terms are not
asserted without a separate derivation. This proposition motivates an
occupancy-weighted local predictive-error score. It does **not** establish that
ensemble disagreement, density, or a learned residual predictor equals
\(\epsilon_K\).

### Step 5. Proper-SSP truncation — exact bound

Since costs are nonnegative and bounded,

\[
0\le Z_C-Z_{C,H}
\le c_{\max}(T_C-H)_+.
\]

Using the natural coupling of a trajectory with its own truncation,

\[
W_1(\mathcal Z_C,\mathcal Z_{C,H})
\le
c_{\max}\,\mathbb E[(T_C-H)_+].
\]

The total prediction error therefore separates into interface/model error and a
hitting-time-tail term. Long horizon is not itself a novelty claim; it is an
explicit amplifier and truncation source.

### Step 6. Return-boundary decision stability — exact proposition

Define the oracle and learned commitment rules at a decision instant:

\[
d^\star=\mathbf 1\{b\le U_C(x)+m\},
\qquad
\widehat d=\mathbf 1\{b\le\widehat U_C(x)+m\}.
\]

If \(|\widehat U_C(x)-U_C(x)|\le\varepsilon\), then

\[
d^\star\ne\widehat d
\Longrightarrow
|b-m-U_C(x)|\le\varepsilon.
\]

Proof is by interval ordering: outside this band, both thresholds lie on the same
side of \(b\). Thus global MAE is not the decision-relevant statistic; harmful
underestimation near the return boundary is the direct mechanism for late return.

### Step 7. Conditional return-failure implication — corollary with timing condition

Assume a deployed upper requirement satisfies

\[
\Pr(Z_C\le U_C(x)\mid\text{prediction accepted})\ge 1-\delta.
\]

If the agent commits at state \(x_t\) with

\[
b_t\ge U_C(x_t),
\]

then, excluding non-energy failure modes, the probability of energy exhaustion
before charger arrival is at most \(\delta\). A threshold-crossing implementation
obtains the premise only if reserve covers check-interval energy and requirement
drift. For an overshoot bound \(\Delta_{\rm check}\), a sufficient engineering
condition is

\[
m\ge\Delta_{\rm check}.
\]

The current experiments must measure this overshoot; the ReturnManager alone is
not a hard energy-safety certificate.

### Step 8. Task-plus-return risk allocation — exact probability inequality

Let \(E_1\) be task-completion resource and \(E_2\) be subsequent return resource,
defined on the same joint mission probability space. If

\[
\Pr(E_1\le U_1)\ge1-\delta_1,
\qquad
\Pr(E_2\le U_2)\ge1-\delta_2,
\]

then without an independence assumption, the union bound gives

\[
\Pr(E_1+E_2\le U_1+U_2)
\ge1-\delta_1-\delta_2.
\]

Therefore two 95% component upper bounds guarantee only the declared lower bound
of 90%, not a joint mission q95. A direct joint mission rollout/model is preferred;
otherwise component risk budgets must sum to the mission risk budget.

## Remarks and Interpretation

- The executed action is a sufficient interface only if A1/A3 are valid. Hidden
  actuator, battery, wind, or perception state may require additional context.
- Policy conditioning may still help under insufficient state representations or
  finite-capacity approximation, but it is not required by the ideal Markov
  factorization.
- A learned trajectory reliability score should approximate the transport terms
  using cross-fitted errors. Training residuals are optimistically biased.
- The mission-level objective is a stranding--throughput Pareto frontier. Prediction
  metrics explain the mechanism but are not the final utility.
- `CONTINUE` versus `RETURN` is an optimal-stopping/one-way-switching view. The
  current threshold managers are baselines, not a proof of optimal stopping.

## Boundaries and Non-Claims

- No theorem states that a safety intervention always increases energy.
- No theorem states that pair novelty alone causes failure.
- No learned energy estimator is a collision-safety or energy-safety certificate.
- No q90/q95 headline is valid until repeated fixed-information clone rollouts show
  non-degenerate conditional aleatoric variation.
- The current static deterministic environment supports point Resource-to-Go plus
  epistemic reliability, not a physical aleatoric tail claim.
- Wasserstein closeness alone does not imply stable q95/q99. A quantile corollary
  needs a CDF bound and local anti-concentration/positive-density assumption.
- The support necessity result does not forbid extrapolation under stronger
  parametric or physical structure; such structure must be declared.
- The CMDP comparison has different training privileges and belongs in a separate
  end-to-end decision track.

## Open Risks

- The formal Oracle headroom Gate may show that perfect Resource-to-Go information
  has insufficient mission decision value over SOC/distance heuristics.
- The 500k navigation checkpoint may fail the preregistered navigation Gate before
  calibration or Oracle evaluation begins.
- A generic executed-action world model plus Deep Ensemble may already match any
  proposed reliability mechanism.
- Interface extrapolation may be confounded with charger distance, obstacle
  density, or hitting horizon unless experiments use matched strata.
- Decision-interval overshoot may dominate estimator error near the return boundary.
- The local transport regularity assumption may be too strong for discontinuous
  safety-filter active-set changes; a weaker total-variation or piecewise analysis
  may be required.
- A second safety family and second dynamics/resource domain remain necessary for
  an oral-level generality claim.
