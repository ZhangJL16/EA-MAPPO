# Learned Safe and Energy-Sustainable Navigation: Derivation Package

Date: 2026-09-03  
Relation to project goal: this is a learned-policy extension of
`ORIGINAL_GOAL.txt`; it does not replace the frozen R3 Resource-to-Go evidence
path or relabel its running experiments.

## Target

Construct a neural continuous-control policy that learns when and how to pursue
tasks versus return to recharge, while controlling the *joint* probability of:

1. avoiding collision and boundary failure;
2. reaching a charger before a finite deadline; and
3. consuming no more than the currently available battery reserve.

The mathematical target is not a lower expected-energy score.  It is a
state-and-budget-dependent feasibility probability that can be estimated by a
neural distributional critic and audited independently of any HOCBF execution
filter.

## Status

**COHERENT AS A METHOD; NOT YET A STANDALONE NOVELTY.**

The broad idea is not novel.  Deterministic minimum-cost reach-avoid is covered
by RC-PPO (NeurIPS 2024); stochastic probabilistic reach-avoid plus expected
cost is covered by RAPCPO (ICML 2026); quantile cost constraints are covered by
QCRL; finite batteries and reload states are covered by consumption-MDP theory.

The residual candidate is the learned distribution of one stopped,
extended-real resource-to-recharge variable.  Its finite part is energy used
before a safe charger hit and its atom at \(+\infty\) is collision or deadline
failure.  This gives a direct certificate for the joint event rather than a
weighted reward or a collection of marginal constraints.  Equation-level
inspection of RAPCPO shows that, at any fixed budget, the same event can be
represented as reach-avoid on a battery-augmented state.  Therefore the CDF
object is useful because it solves all budgets at once and exposes survivor
bias, but it is not by itself an oral-level theorem.  The harder residual
target is a calibrated quantitative consumption-Büchi learner for continuous
unknown dynamics, with an explicit approximation-error to lifecycle-risk bound.

## Invariant Object

For a policy \(\pi\), state \(z\), and remaining horizon \(h\), define

\[
Y_h^\pi(z)=
\begin{cases}
\displaystyle\sum_{t=0}^{\tau_C-1} e(Z_t,A_t,Z_{t+1}),
& \tau_C\le h\ \text{and}\ \tau_C<\tau_U,\\[1ex]
+\infty, & \text{otherwise}.
\end{cases}
\]

Here \(C\) is the charger set, \(U\) is the unsafe set, \(\tau_C\) and
\(\tau_U\) are first hitting times, and \(e\ge0\) is realized transition
energy.  The invariant object is its finite-threshold CDF

\[
F_h^\pi(y\mid z)=\Pr\!\left(Y_h^\pi(z)\le y\right),\qquad y<+\infty.
\]

This is exactly the probability of safe, timely recharge within resource
budget \(y\).  In particular,

\[
\Pr(Y_h^\pi=+\infty)
=1-\Pr(\tau_C\le h,\ \tau_C<\tau_U).
\]

The failure atom is essential: conditioning an energy predictor only on
successful returns would make failed trajectories disappear and can make an
unsafe policy look energy-efficient.

## Assumptions

**A1 (Markov state).** The physical observation is made Markov at the chosen
control rate by including all variables needed by the environment model or by
using a recurrent belief state.  The augmented state is

\[
Z_t=(X_t,B_t,G_t,C_t,H_t,K_t),
\]

where \(X_t\) is physical/perceptual state, \(B_t\) remaining charge,
\(G_t\) the active task goal, \(C_t\) charger information, \(H_t\) remaining
deadline, and \(K_t\) the recharge-cycle index when lifecycle allocation is
used.

**A2 (Transition resource).** Realized energy is measurable, non-negative, and
bounded per transition: \(0\le e_t\le e_{\max}<\infty\).  Before recharge,
\(B_{t+1}=B_t-e_t\).

**A3 (Terminal sets).** Charger and unsafe sets are measurable and disjoint.
For the finite-horizon return problem they are treated as terminal.  A new
excursion begins only after a completed recharge.

**A4 (Finite excursion horizon).** Every certificate is stated for an explicit
remaining horizon \(h\le H\).  No undiscounted infinite-horizon Bellman
contraction is assumed.

**A5 (Policy class).** \(\pi(a\mid z)\) is a stochastic continuous policy.  A
learned task/return gate may be represented as a latent hybrid action, but is
not required by the theorems.

**A6 (Deployment distribution).** Any learned-CDF guarantee is restricted to
the declared state distribution or calibration set.  Out-of-distribution
generalization is not automatic.

**A7 (Calibration event).** Where used, a lower confidence CDF
\(\underline F_h\) satisfies \(\underline F_h(y\mid z)\le
F_h^\pi(y\mid z)\) simultaneously on the certified domain with confidence
at least \(1-\alpha\).

**A8 (Extra-energy replicability).** For budget monotonicity of the *optimal*
viability set, a controller with more battery can emulate any controller with
less battery, for example by passing a clipped virtual budget to the policy.

## Notation

| Symbol | Meaning |
| --- | --- |
| \(X_t\) | physical/perceptual navigation state |
| \(B_t\) | remaining battery energy |
| \(G_t,C_t\) | task goal and charger representation |
| \(H_t\) | remaining steps or time before mandatory recharge |
| \(U,C\) | unsafe and charger sets |
| \(e_t\) | realized energy on one executed transition |
| \(Y_h^\pi\) | extended-real resource-to-safe-recharge random variable |
| \(F_h^\pi\) | CDF of \(Y_h^\pi\) at finite thresholds |
| \(r\) | non-spendable reserve/margin |
| \(q_{1-\delta}\) | upper \((1-\delta)\)-quantile, with empty set mapped to \(+\infty\) |
| \(\delta_k\) | allowed joint failure probability on recharge cycle \(k\) |

## Strategy

1. Augment the state by battery, deadline, goals, and optionally lifecycle risk
   index; this restores the Markov property for consumable resources.
2. Learn the *full stopped law* of energy-to-safe-recharge.  Do not write an
   additive Bellman equation for a static VaR; static quantiles are generally
   not time-consistent.
3. Extract the joint feasibility probability directly as
   \(F_h^\pi(B-r\mid z)\).
4. Train the task policy under a lower-confidence CDF constraint.  HOCBF may
   remain during data collection, but raw actions must also be evaluated.
5. Compose finite-excursion certificates across recharge cycles with a
   summable risk allocation when making a lifecycle claim.

### Operational action-conditioned slice

The passive state-only CDF experiments are now rejected empirically: three
learned heads failed to beat the locked geometry baseline.  The next executable
object is therefore the pre-action critic

\[
Q_h^\mu(z,b,a)
=\Pr_\mu\!\left(
 \tau_C<\tau_U,\ \tau_C\le h,\
 \sum_{k=t}^{\tau_C-1}e_k\le b
 \mid Z_t=z,A_t=a
\right),
\]

where \(\mu\) is the declared continuation/behavior policy.  For the current
offline Gate, \(\mu\) is the frozen-R3 grouped-intervention distribution; the
notation must not be read as an arbitrary-policy counterfactual guarantee.
The corresponding one-step recursion is

\[
Q_h^\mu(z,b,a)
=\int
\mathbf 1_{\{e(z,a,z')\le b\}}
\begin{cases}
1,&z'\in C,\\
0,&z'\in U\text{ or }h=1,\\
\displaystyle\int Q_{h-1}^\mu(z',b-e,a')\,\mu(da'\mid z'),&\text{otherwise}
\end{cases}
P(dz'\mid z,a).
\tag{O1}
\]

The neural implementation represents the budget axis by ordered logit knots:

\[
\ell_0=g_0(z,a),\qquad
\ell_j=\ell_0+\sum_{i=1}^{j}\operatorname{softplus}(g_i(z,a))/K,
\qquad
\widehat Q(z,b,a)=\sigma(\operatorname{Interp}_b(\ell)).
\tag{O2}
\]

Hence \(b_1\le b_2\Rightarrow\widehat Q(z,b_1,a)\le
\widehat Q(z,b_2,a)\) holds architecturally.  The actor-relevant gradient
\(\nabla_a\widehat Q\) is present, unlike in the rejected trajectory-level
predictors.  Training combines stopped Monte-Carlo event labels with the
one-step residual from (O1); the no-action and no-Bellman variants are required
mechanism controls.

This operational slice does not change the theorem assumptions.  In
particular, the existing grouped dataset retains only compact per-step state
and has five collision rollouts out of 750.  It can test energy/deadline
rechargeability, but a positive result cannot establish collision-tail safety.
That requires a subsequent near-obstacle forked-action collection with
per-state perception.  Actor training is gated on both stages.

## Derivation Map

\[
\text{augmented state}
\Longrightarrow
\text{stopped extended-real resource law}
\Longrightarrow
\text{distributional Bellman recursion}
\Longrightarrow
\text{finite-battery joint certificate}
\Longrightarrow
\text{calibrated neural constraint}
\Longrightarrow
\text{multi-cycle risk composition}.
\]

## Main Derivation

### 1. State augmentation

Let the unaugmented controlled kernel be
\(P(dx'\mid x,a,\xi)\), where \(\xi\) denotes task/charger context.  Define

\[
\begin{aligned}
B' &= B-e(x,a,x'),\\
H' &= H-1,\\
G' &= \Gamma(G,x'),\\
K' &= K+\mathbf 1\{x'\in C\},
\end{aligned}
\]

with a recharge reset \(B'=B_{\max}\) only when the previous excursion has
terminated at \(C\).

**Proposition 1 (Markovization).** Under A1--A3, the augmented process
\(Z_t\) is controlled Markov even though \(X_t\) alone need not encode the
remaining feasible resource.

**Proof sketch.** Every update of \(B,H,G,K\) is a measurable function of the
current augmented state, action, and next physical state.  Thus the conditional
law of \(Z_{t+1}\) depends on the history only through \((Z_t,A_t)\).  This is
standard augmentation, not a novelty claim.

### 2. Exact joint-event semantics

For any finite \(y\), the event represented by the invariant object is

\[
\{Y_h^\pi(z)\le y\}
=\left\{
\tau_C\le h,\ \tau_C<\tau_U,\
\sum_{t=0}^{\tau_C-1}e_t\le y
\right\}.
\]

**Proposition 2 (Failure-atom identity).** For every policy and finite horizon,

\[
F_h^\pi(y\mid z)
=\Pr_\pi\!\left(
\tau_C\le h,\ \tau_C<\tau_U,
\sum_{t<\tau_C} e_t\le y\mid Z_0=z
\right).
\]

**Proof.** It follows immediately by taking the inverse image of
\(( -\infty,y]\) under the piecewise definition of \(Y_h^\pi\).  All failed
trajectories map to \(+\infty\) and therefore cannot belong to the finite
sublevel set.

This identity is more useful than predicting
\(E[\sum e_t\mid\text{success}]\), which has no control over either the energy
tail or the missing success mass.

### 3. Finite-horizon distributional recursion

For a terminal charger state and finite \(y\), set

\[
F_h^\pi(y\mid z)=\mathbf 1\{y\ge0\},\qquad x\in C.
\]

For an unsafe state or exhausted horizon outside the charger,

\[
F_h^\pi(y\mid z)=0,
\qquad x\in U\ \text{or}\ (h=0, x\notin C).
\]

For a safe non-charger state and \(h\ge1\), define

\[
F_h^\pi(y\mid z)
=\int_{a}\!\pi(da\mid z)
  \int_{x'}\!P(dx'\mid z,a)
F_{h-1}^\pi\!\left(y-e(z,a,x')\mid z'\right),
\tag{1}
\]

with \(F(\tilde y)=0\) for \(\tilde y<0\), and with terminal handling applied
to \(x'\in C\cup U\).

**Theorem 1 (Distributional first-passage recursion).** Under A1--A4,
Equation (1) uniquely gives the finite-threshold CDF of \(Y_h^\pi\) by backward
recursion.

**Proof.** The terminal cases are the definition at \(h=0\).  Assume the claim
holds for \(h-1\).  Condition on the first action and transition.  On every
nonterminal continuation, \(Y_h=e_0+Y_{h-1}'\); on charger and unsafe
transitions the boundary conditions give respectively zero remaining resource
and failure mass.  The tower property yields (1).  Finite backward induction
gives uniqueness.

**Important non-substitution.** The quantile itself generally does not obey
\(q(Y_h)=e_0+q(Y_{h-1})\) under stochastic mixtures.  The critic must learn a
distribution/CDF (or a dynamically consistent risk measure), not bootstrap a
static quantile as though it were an expectation.

### 4. Battery-feasibility certificate

Define the required \((1-\delta)\)-resource quantile

\[
R_{h,\delta}^\pi(z)
=\inf\{y\ge0:F_h^\pi(y\mid z)\ge1-\delta\},
\]

where the infimum of the empty set is \(+\infty\).

**Theorem 2 (Joint safe-recharge certificate).** If

\[
B-r\ge R_{h,\delta}^\pi(z),
\tag{2}
\]

then

\[
\Pr_\pi\!\left(
\tau_C\le h,\ \tau_C<\tau_U,
\sum_{t<\tau_C}e_t\le B-r
\mid Z_0=z\right)\ge1-\delta.
\tag{3}
\]

Conversely, (3) implies \(F_h^\pi(B-r\mid z)\ge1-\delta\), and hence (2)
whenever the quantile convention is well behaved at atoms (or with the usual
right-continuous generalized inverse).

**Proof.** Equation (2), monotonicity/right-continuity of a CDF, and Proposition
2 give (3).  The reverse direction follows from the generalized-inverse
definition.

Because \(e_t\ge0\), the terminal energy bound also implies that the battery
does not cross the reserve earlier in the excursion.  If recuperation makes
individual \(e_t\) negative, this implication fails and pathwise prefix
resource must be included explicitly.

### 5. Calibrated learned certificate

Let a neural critic output a lower confidence CDF \(\underline F_{\psi,h}\).

**Theorem 3 (Approximation-margin transfer).** On the calibration event A7, if

\[
\underline F_{\psi,h}(B-r\mid z)\ge1-\delta,
\tag{4}
\]

then the true policy satisfies the joint event (3).  Therefore (4) is a
certificate with statistical confidence at least \(1-\alpha\) on the declared
domain.

An equivalent deterministic error form is: if
\(\sup_{z,y}|\widehat F-F_h^\pi|\le\varepsilon\), require

\[
\widehat F_{\psi,h}(B-r\mid z)\ge1-\delta+\varepsilon.
\tag{5}
\]

**Proof.** From A7,
\(F_h^\pi\ge\underline F_{\psi,h}\ge1-\delta\).  For (5), use
\(F_h^\pi\ge\widehat F-\varepsilon\).

This theorem deliberately separates the exact probabilistic result from the
difficult statistical problem.  A neural estimate without a validated
lower-bound construction is a score, not a certificate.

### 5.1 Approximate Bellman-subsolution certificate

Let \(\mathcal T_h^\pi\) denote the positive CDF operator in Equation (1), and
let \(L_h(z,B)\in[0,1]\) be a learned lower-score family with correct terminal
boundary \(L_0\le F_0^\pi\).  Suppose the audited one-sided residual satisfies

\[
L_h(z,B)\le
(\mathcal T_h^\pi L_{h-1})(z,B)+\varepsilon_h
\quad\text{uniformly on the certified domain}.
\tag{5a}
\]

**Theorem 3a (Residual-to-probability margin).** Under A1--A4,

\[
F_h^\pi(B-r\mid z)
\ge L_h(z,B)-\sum_{j=1}^{h}\varepsilon_j.
\tag{5b}
\]

Consequently, requiring

\[
L_h(z,B)\ge1-\delta+\sum_{j=1}^{h}\varepsilon_j
\tag{5c}
\]

is sufficient for the joint certificate.

**Proof.** The claim is true at the terminal boundary.  Assume it holds at
\(h-1\).  Positivity and constant preservation of the Markov expectation give

\[
\mathcal T_h^\pi L_{h-1}
\le \mathcal T_h^\pi F_{h-1}^\pi
   +\sum_{j=1}^{h-1}\varepsilon_j
=F_h^\pi+\sum_{j=1}^{h-1}\varepsilon_j.
\]

Combine this with (5a) and rearrange.  Backward induction completes the proof.
This turns a one-sided Bellman-residual audit into a usable actor margin.  An
average TD loss cannot substitute for the required uniform or high-confidence
one-sided residual bound.

### 5.2 Equivalence to augmented reach-avoid

For a fixed resource threshold \(y\), augment the physical state with a virtual
resource \(D_0=y\), \(D_{t+1}=D_t-e_t\), and expand the unsafe set to
\(U_y=U\cup\{D<0\}\).  Then

\[
F_h^\pi(y\mid z)
=\Pr_\pi(\tau_C\le h,\ \tau_C<\tau_{U_y}
\mid Z_0=(z,D_0=y)).
\tag{5d}
\]

This identity means that RAPCPO or another reach-avoid learner can, in
principle, solve one chosen battery threshold after state augmentation.  Our
distributional formulation amortizes over all \(y\), represents the failure
atom explicitly, and supports quantile queries, but those are differentiators
to test—not permission to claim a new reachability principle.

### 6. Upward closure and its limitation

Define optimal finite-horizon viability

\[
V_h^*(z,B)=\sup_{\pi\in\Pi_B}
\Pr_\pi(Y_h\le B-r\mid z),
\qquad
\mathcal V_{h,\delta}=\{(z,B):V_h^*(z,B)\ge1-\delta\}.
\]

**Proposition 3 (Budget monotonicity).** Under A2 and A8, if
\((z,B_1)\in\mathcal V_{h,\delta}\) and \(B_2\ge B_1\), then
\((z,B_2)\in\mathcal V_{h,\delta}\).

**Proof.** At \(B_2\), emulate the \(B_1\) policy using virtual budget
\(B_1\).  The same trajectory distribution is feasible and the event
\(Y_h\le B_1-r\) is contained in \(Y_h\le B_2-r\).

This proposition does **not** imply that an arbitrary battery-conditioned
neural policy is monotone: a poorly learned policy may take greater risks when
given more charge.  Monotonicity must be architectural, regularized, or checked.

### 7. Recharge-cycle risk composition

Let \(A_k\) be failure of the joint event during excursion \(k\), and let
\(\mathcal H_{k-1}\) be the history at its start.

**Theorem 4 (Lifecycle composition without independence).** Suppose that on
every history that has survived to cycle \(k\),

\[
\Pr(A_k\mid\mathcal H_{k-1})\le\delta_k
\quad\text{almost surely}.
\tag{6}
\]

Then for any \(N\),

\[
\Pr\left(\bigcup_{k=1}^{N}A_k\right)
\le\sum_{k=1}^{N}\delta_k.
\tag{7}
\]

If \(\sum_{k=1}^{\infty}\delta_k\le\bar\delta\), then

\[
\Pr\left(\exists k\ge1:A_k\right)\le\bar\delta.
\tag{8}
\]

**Proof.** Partition first failure by cycle.  For the event \(S_{k-1}\) of no
earlier failure,
\[
\Pr(A_k\cap S_{k-1})
=\mathbb E[\mathbf1_{S_{k-1}}
\Pr(A_k\mid\mathcal H_{k-1})]\le\delta_k.
\]
Summing disjoint first-failure events yields (7).  Taking \(N\to\infty\) and
using continuity from below gives (8).  No cycle independence is used.

A concrete allocation is

\[
\delta_k=\frac{6\bar\delta}{\pi^2 k^2},
\qquad \sum_{k=1}^{\infty}\delta_k=\bar\delta.
\]

This supplies a mathematically valid meaning of “energy-sustainable over a
lifecycle.”  It also exposes a hard limitation: a fixed \(\delta>0\) per cycle
only gives \(N\delta\) over \(N\) cycles and cannot support a nontrivial
infinite-lifecycle claim.  A decreasing risk schedule can itself become
infeasible under irreducible aleatoric noise; that feasibility must be tested.

### 8. Learning objective and minimal architecture

Let \(Q_R(z,a)\) be the ordinary SAC task-return critic and let
\(F_{\psi,h}(y\mid z,a)\) be one conceptual distributional feasibility critic.
The constrained actor problem is

\[
\max_\theta\;
\mathbb E[Q_R(Z,A)+\alpha\mathcal H(\pi_\theta(\cdot\mid Z))]
\quad\text{s.t.}\quad
\underline F_{\psi,h}(B-r\mid Z,A)\ge1-\delta_K.
\tag{9}
\]

A trainable surrogate is

\[
\mathcal L_{\text{actor}}
=\mathbb E\!\left[
\alpha\log\pi_\theta(A\mid Z)-Q_R(Z,A)
+\lambda(Z)
\big(1-\delta_K-\underline F_{\psi,h}(B-r\mid Z,A)\big)_+
\right].
\tag{10}
\]

Equation (10) is an optimization mechanism, not by itself a certificate; the
certificate comes from Theorem 3 and held-out calibration.

The minimal project architecture is therefore:

1. the existing structured LiDAR/goal encoder;
2. battery, reserve, remaining deadline, charger goal, and cycle-risk inputs;
3. standard twin SAC task critics (one conceptual reward critic, duplicated for
   bias control);
4. one conceptual distributional feasibility critic representing both finite
   energy and the failure atom; and
5. one actor, optionally with a learned latent TASK/RETURN mixture head.

There is no mathematical requirement for three unrelated safety critics.  A
separate collision critic is redundant if collision is encoded in the failure
atom, although twin target networks may still be used for numerical stability.

### 9. Practical representation of the failure atom

Finite quantile atoms alone cannot numerically represent \(+\infty\).  Two
valid implementations are:

- a categorical distribution with an explicit terminal-failure atom; or
- a structured head
  \(F(y\mid z)=p_{\mathrm{succ}}(z)
  F_E(y\mid z,\mathrm{succ})\).

The second representation must be trained jointly.  Training
\(F_E(\cdot\mid\mathrm{succ})\) only on successful trajectories without also
learning/calibrating \(p_{\mathrm{succ}}\) reintroduces survivor bias.

### 10. Why marginal constraints are insufficient

Separate constraints such as

\[
\Pr(\text{no collision})\ge1-\delta_c,
\qquad
\Pr(E\le B-r)\ge1-\delta_e
\]

do not equal the desired joint probability.  A union bound gives at best
\(1-\delta_c-\delta_e-\delta_h\) after adding the deadline event, and the bound
can be loose.  Likewise,

\[
q_{0.95}(X)+q_{0.95}(Y)\ne q_{0.95}(X+Y)
\]

in general.  The stopped law avoids both composition errors by estimating the
mission event directly.

## Remarks

### Connection to the original R3 Resource-to-Go line

The original project estimates executed closed-loop Resource-to-Go for a frozen
navigation policy and tests whether a return commitment rule has real headroom.
That remains a valid diagnostic and supplies training labels/teacher data.  The
new line changes the policy itself and should begin only after the diagnostic
Gate is read.  Oracle reservation can initialize or upper-bound learning, but
must not remain part of the deployed policy if the claim is end-to-end learning.

### Role of HOCBF

The HOCBF layer may safely collect experience and act as a high-quality teacher.
To make the stronger learned-navigation claim, report at least:

- raw actor collision and joint recharge feasibility;
- filtered actor collision and joint recharge feasibility;
- filter intervention rate and squared action correction; and
- performance after removing the filter at evaluation.

A safe filtered trajectory proves safety of the composed controller, not of the
neural policy.

### Candidate falsifiable hypotheses

- H1: the joint CDF critic is better calibrated for energy outage than an
  expected-energy critic at equal task success.
- H2: failure-atom training reduces conditional-survivor bias versus fitting
  energy only on successful returns.
- H3: a budget-conditioned actor dominates fixed SOC, distance-reserve, and
  oracle-switch candidates on task/return Pareto frontiers.
- H4: raw-policy safety improves during shielded training as measured by falling
  intervention and raw-vs-filtered performance gap.
- H5: per-cycle empirical failure tracks the declared \(\delta_k\) schedule in
  held-out multi-cycle tests.

## Boundaries / Non-Claims

1. State augmentation, distributional RL, quantile constraints, reachability
   critics, recovery policies, and reload-state MDPs are prior art.
2. Theorem 2 is an exact consequence of the proposed random-variable
   definition; its value lies in selecting the correct joint object, not in a
   difficult proof.
3. Theorem 3 requires a genuine lower-confidence CDF.  Ordinary validation
   accuracy or an uncalibrated neural quantile is insufficient.
4. Theorem 4 is a risk-composition result, not proof that the required
   \(\delta_k\) remains feasible forever.
5. Finite-horizon certification is not almost-sure infinite-horizon safety.
6. HOCBF-filtered deployment cannot be used as evidence that the raw network
   learned collision safety.
7. The theory assumes non-negative per-step consumption.  Regenerative braking
   requires a prefix-minimum resource formulation.
8. Continuous LiDAR partial observability may violate A1; a recurrent belief
   state or a carefully scoped empirical claim is then necessary.
9. No oral-level novelty claim is warranted until full-text overlap checking,
   calibrated estimation, strong baselines, and multiple-seed lifecycle tests
   succeed.

## Open Risks

- **Nearest-paper risk:** RAPCPO is accepted at ICML 2026.  Full-text inspection
  confirms that it learns a reach-avoid critic, a cost critic, and a
  successful-hit compensation factor, then optimizes a policy-dependent
  surrogate rather than enforcing its exact certificate at every update.
- **Formal-methods risk:** consumption/energy MDP literature may already define
  a mathematically equivalent resource-to-reload threshold object for finite
  models.
- **Statistical risk:** obtaining a useful uniform lower CDF bound in a
  high-dimensional continuous observation space may be too conservative.
- **Optimization risk:** a statewise CDF constraint can destabilize SAC or
  collapse task exploration without a curriculum/teacher replay stage.
- **Data risk:** rare collision and energy-outage tails require targeted
  sampling; ordinary replay is unlikely to calibrate them.
- **Systems risk:** a policy can exploit simulator-specific energy or LiDAR
  artifacts, so domain randomization and held-out physics are necessary.
- **Evaluation risk:** a 500-episode single-cycle evaluation is inadequate for
  a lifecycle theorem; exact small-MDP checks plus long multi-cycle stress tests
  are both required.

## Research Decision After Equation-Level Audit

The extended-real CDF is retained as the correct implementation object, but is
demoted from “the core novelty” to “the representation that makes the real gap
testable.”  A paper-level contribution now requires all three of the following:

1. a nontrivial approximation theorem such as Theorem 3a strengthened from a
   uniform assumption to a finite-sample, distribution-shift-aware bound;
2. quantitative repeated-recharge control in continuous unknown dynamics,
   beyond finite known consumption MDPs and finite-state temporal-logic
   learning; and
3. evidence that the raw neural actor—not only a safety filter—learns the joint
   battery/reach/avoid boundary.

Without item 1, the work is an integration paper.  Without items 2 and 3, it is
already substantially covered by RAPCPO plus state augmentation.
