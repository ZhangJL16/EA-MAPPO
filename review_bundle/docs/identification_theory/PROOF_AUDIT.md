# Proof Audit

## Target

Audit whether the identification package proves a new safety-induced-censoring theory, or only valid special cases of existing support and safe-exploration results.

## Status

`COHERENT AFTER REFRAMING; NOVELTY BLOCKED`.

## Invariant Object

The invariant used in the strongest proof is statistical distinguishability of two full-data environments under a uniformly safe data-acquisition policy:

$$
\operatorname{TV}\left(P_S^{\mathrm{obs}},P_U^{\mathrm{obs}}\right).
$$

The safety budget bounds the probability of executing the only distinguishing action, which bounds total variation and therefore binary testing accuracy.

## Dependency Map

1. Proposition 1 depends on a structural zero for one action and a model class that can vary unsupported outcomes.
2. Corollary 1 depends on equal observed laws and disjoint radius-$1/2$ success events.
3. Proposition 2 depends on deterministic commitment and equal charger-mode suffixes.
4. Proposition 3 depends on a valid augmented-state Markov representation.
5. Proposition 4 depends on coordinate-separable updates and no execution outside the admitted set.
6. Theorem 1 depends on a safe/catastrophic model pair, equality before first probe, uniform $\beta$-safety, a coupling inequality for total variation, and the two-point testing bound.

## Claim-by-Claim Audit

| Claim | Mathematical status | Hidden assumption check | Counterexample boundary | Novelty status |
|---|---|---|---|---|
| Rejected-action non-identification | Valid | Requires zero execution probability and no structural coupling | Positive randomized probe support or correct model can identify | Standard no-overlap |
| Minimax error $\ge 1/2$ under identical laws | Valid | Estimator may be randomized; common law handled | Any informative auxiliary observation breaks equality | Standard two-point argument |
| Post-commitment continuation non-identification | Valid | Charger suffix must be equal; continuation never executed | Randomized continuation restores support | History-action no-overlap |
| Two-level reduction to augmented MDP | Valid when augmented state is sufficient | Filter/learner memory must be included | Hidden non-Markov variables invalidate the simple reduction but not necessarily non-identification | Representation, not contribution |
| Coordinate-separable lock-in | Valid | No cross-action update is essential | Smooth/shared models can update untried actions | Trivial conservative trap |
| Uniform safety–identification lower bound | Valid | Catastrophic outcome on first probe and uniform safety are essential | Safe side channel, weaker model class, or informative proxy can defeat bound | Correct but generic safe-bandit lower bound |
| Trajectory-level embedding | Valid | Continuation must be the sole distinguishing action | Additional pre-stop sensor information can distinguish | No new theorem |
| Probe-count energy bound | Valid identity | Each probe must cost at least $d_E$ | Variable/zero-cost probes alter count | Standard knapsack accounting |

## Detailed Audit of Theorem 1

### Step 1 — Safety-to-probe probability

In $\mathcal{M}_U$, the first execution of $b$ causes failure with probability one. Therefore

$$
P_U(A)=P_U(\text{failure})\le\beta.
$$

The inequality direction is correct.

### Step 2 — Equality of first-probe probability

The models and algorithmic randomness are coupled identically until the first execution of $b$. The event of deciding to execute $b$ is determined before observing its outcome. Therefore

$$
P_S(A)=P_U(A).
$$

This step would fail if the learner had an environment-dependent pre-probe side channel; the assumption excludes it.

### Step 3 — Total variation

Under the coupling, observed datasets are equal on $A^c$. The coupling characterization gives

$$
\operatorname{TV}(P_S^{\mathrm{obs}},P_U^{\mathrm{obs}})
\le P(A)
\le\beta.
$$

No equality is claimed.

### Step 4 — Testing error

For equal priors, binary testing error is

$$
\frac{1-\operatorname{TV}(P_S^{\mathrm{obs}},P_U^{\mathrm{obs}})}{2}.
$$

The maximum model-specific error is at least the average error, yielding $(1-\beta)/2$. Quantifiers are consistent with an infimum over uniformly $\beta$-safe learners.

## Formula-Derivation Classification

- Observation indicators: **definitions**.
- Augmented-MDP representation: **proposition conditional on state sufficiency**.
- Static observational equivalence: **construction/proposition**.
- Safety-to-TV-to-testing chain: **theorem**.
- “Self-reinforcing conservatism”: **interpretation of Proposition 4**, not a separate theorem.
- Resource-aware sample complexity: **not derived**; only a deterministic count bound is available.

## Fatal Audit Finding

There is no unresolved mathematical `FATAL` or `CRITICAL` issue in the claims as scoped. The unresolved fatal issue is scientific novelty:

- the static result is no-overlap;
- the dynamic result embeds a two-action cautious bandit;
- positive safe probing is directly occupied by SafeOpt, SafeMDP, active safe learning, and ActSafe;
- resource coupling is a budgeted-MDP/BwK constraint.

Thus a correct proof package does not justify a new ICLR theory route.

## Prohibited Upgrades

Do not call these results:

- a complete identification theory for persistent UAVs;
- a new minimax frontier;
- a proof of safe learnability;
- a guarantee for real UAV deployment;
- a theorem that uses both censoring levels essentially.
