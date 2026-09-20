# Pre-freeze mathematical audit

Historical audit retained. The user subsequently selected lifetime delta-safe
learning plus strict-safety impossibility. The revised definitions and proofs are
in ../paper/THEORY.md; this document records the objections, not a current pause.

This document is written before new experiment outcomes. It does not change the
user's numerical parameters or assert the proposed theorems. The main issue is
correctness of the theorem statement, not an empirical performance gate.

## 1. Positive KL does not permit zero-error finite-time safety unlocking

Consider the proposed binary fork with 0 < kappa < 1/2. Every finite probe
observation string has positive probability under both models. More generally,
assume finite-history laws under all commonly safe experiments are mutually
absolutely continuous, including all learner-observable rewards, durations,
states and internal randomization.

Let sigma be the first time a learner selects either specialized exploit. Before
sigma, all selected actions are common-safe. If P_-(sigma=n, A_n=a_+) > 0,
the learner has positive catastrophe probability under M_-. Uniform almost-sure
safety therefore requires this probability to be zero. Absolute continuity of
the pre-action histories implies P_+(sigma=n, A_n=a_+) = 0 as well. Interchanging
the models gives the same result for a_-. Taking a countable union over finite
n shows that neither specialized exploit can ever be selected in either model.
Randomization does not evade the argument: the learner uses the same conditional
action kernel in both models.

If all common-safe actions earn at most 0.2 per unit time and the model-specific
oracle earns 1, regret is at least 0.8 T. This holds for **all positive proposed
kappa values**, despite positive KL and singleton terminal blocks under the
proposed observational-law refinement. It refutes the stated zero-catastrophe
iff theorem under the currently listed assumptions. Infinite-sequence consistent
identification does not supply a finite safe time at which to unlock an action.

Possible repairs require an explicit change of theorem assumptions/guarantee:

- Zero catastrophe: support-based knowledge, and finite-time zero-error evidence
  for safety-relevant exclusions (e.g. suitably distinguishable observations).
  The current noisy examples do not satisfy this condition.
- Lifetime risk at most delta: an anytime-valid confidence set with
  P(exists n: M* not in V_n) <= delta. Execute only experiments safe for all of V_n;
  then catastrophe is contained in confidence failure, if experiment safety and
  regeneration assumptions truly hold. This is high-probability, not zero-risk.
  Certification correctness and safety should share one simultaneous coverage
  event; naive fixed-time tests reused at arbitrary stopping times do not suffice.

The user has been asked which semantics to adopt. Neither is silently assumed.

## 2. Proposed Gamma is zero at the recursive root

At V={00,01,10,11}, only p0 and the uninformative robust baseline are common-safe.
Models 00 and 01 have equal laws for every such experiment. They require different
specialized safe exploits, hence form a decision-relevant pair. For every design
w, that pair's weighted KL is zero. Thus the proposed min over D(V) is zero,
regardless of kappa1>0. Yet distinguishing the first bit can unlock p1 under a
high-probability safety formulation.

Consequently, Gamma(V)=0 is not global impossibility for recursively learnable
problems under that formulation. A stage design could use pairs in distinct
current observational blocks; within-block unresolved pairs must be deferred.
This is a different complexity. A pathwise sum of local complexities is not
already an information-theoretic lower bound for all adaptive strategies.
A global worst-pair nonadaptive design is also not automatically an
instance-specific lower bound: adaptive designs can depend on the true model.

For a fixed experiment menu and disjoint acceptable-answer sets, a valid starting
point is the stopped-history change-of-measure constraint

    sum_e E_M[N_e] KL(P_M^e || P_M'^e) >= kl(1-delta, delta).

Its assumptions include the full observation law, admissible stopping, and valid
probability control for the separating recommendation event. Expected time uses
E_M[tau_e], not automatically max_M' E_M'[tau_e]. History-dependent menus require
an adaptive design argument; neither the proposed universal lower bound nor a
matching upper bound has been proved here.

## 3. Pairwise policy overlap is insufficient

Acceptable-policy sets {a,b}, {b,c}, {a,c} have nonempty pairwise intersections,
but empty total intersection. Then D(V) defined by disjoint pairs is empty while
common_optimal_policy(V) must return false. These sets are realizable with safe
stationary deterministic arms: in each model give its two listed arms reward 1
and the third reward 0; for randomized policies the epsilon-optimal sets also
intersect pairwise but have no common member when epsilon < 1/3.

Decision-relevant complexity needs an answer-specific alternative set (or a
suitable set-valued formulation), not just disjoint model pairs. This is also
why multiple-correct-answer identification literature is essential.

## 4. Regeneration and experiment composition need explicit assumptions

Avoiding C before returning to G does not imply returning to G by H. A policy
may time out outside G forever without catastrophe. Require a return guarantee
by a bounded horizon, or a certified recovery continuation counted in duration.
If G has multiple states, returning to a different point need not allow the same
next experiment. Either index experiments and information by starting state, or
provide a common regenerative-state law/repositioning policy with counted costs.
Do not replace physical regeneration by an uncharged simulator reset.

## 5. UAV observation leakage

Under the requested deterministic duration law, a known library arm reveals

    theta = r * (T_cycle - T_i) / E_i

from exact cycle duration alone. The defined full trajectory observation includes
time. Exact battery depletion or charging duration leaks the same quantity.
Gaussian noise in Y_i cannot conceal these other channels. Observational laws
with different deterministic durations are mutually singular, not the finite-KL
Gaussian-only laws. Masking the clock/energy requires an explicit partially
observed benchmark definition; it cannot be silently done in the implementation.
Alternatively keep the natural full observation and honestly analyze one-cycle
identification. The numerical library and battery rule can stay unchanged.

Further, normalized energy observations Y_i/E_i ~ N(theta, 0.03^2) have the same
per-cycle KL for every arm. Optimal library arm identity need not vary with theta;
if one arm is robust-safe and optimal for all five models, report certification
at time zero. Never change battery/reward/library to create an algorithm gap.

## 6. Other pre-freeze implementation choices

- m=32 nuisance bits means K=2^34 if enumerated. Use an exact factorized model/set
  representation, and describe its extra structure rather than claiming explicit
  enumeration is computationally small.
- Define epsilon, probe rewards/durations, exploit durations, nuisance observation
  channel, randomized tree distribution, actual seeds, ties, finite-horizon partial
  cycles and absorbing catastrophe accounting before outcomes.
- Common-policy overlap is not generally a transitive equivalence relation; a
  reported policy-equivalence-class count requires a precise definition.
- Use symbolic/log likelihoods or support metadata: floating-point underflow must
  never be treated as zero-probability evidence of model elimination.
- Record censored certification times as censored; do not regress only successful
  certifications against 1/Gamma and ignore unresolved or infinite cases.

No theorem target, implementation completeness or protocol freeze is claimed by
this audit. These are issues to resolve before the requested one-shot execution.
