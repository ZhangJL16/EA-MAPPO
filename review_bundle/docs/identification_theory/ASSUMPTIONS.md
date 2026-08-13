# Assumptions

## Baseline Observation Assumptions

**A1 — Sequential consistency.** If action $a$ is physically executed at time $t$, the logged outcome equals $Y_t(a)$.

**A2 — No interference across independent sorties.** Potential outcomes for one sortie are not changed by simultaneous unmodeled agents. Battery aging or environment drift must instead be included in state or declared as shift.

**A3 — Measurable intervention.** The proposal, filter, and commitment decisions are measurable with respect to their declared information sets. In particular, $\tau_c$ is a stopping time.

**A4 — Logged execution.** $A_t^P$, $A_t^X$, $F_t$, commitment status, and realized outcomes of $A_t^X$ are logged without ambiguity.

**A5 — Markov sufficiency only when invoked.** If a Markov reduction is used, the augmented state includes physical state, battery, mode, and learner/filter memory needed to make the transition and decision kernels Markov.

## Deliberately Absent Assumptions

**A6 — No global positivity.** The filter and commitment rule may assign zero probability to target history-action pairs.

**A7 — No cross-action smoothness by default.** Outcomes of rejected actions need not be related to outcomes of executed neighbors.

**A8 — No correct simulator by default.** Simulated labels are not physical ground truth unless simulator validity is separately established.

**A9 — No independent censoring by default.** Commitment may depend on state, battery, uncertainty, and predicted continuation consequences.

**A10 — No automatic safety of probes.** A probe is an ordinary physical action whose collision and energy consequences are unknown before execution.

## Assumptions for the Static Lower Bound

**S1 — Rich model class.** The class contains two environments that agree on all outcomes under deployed support and differ on at least one unsupported action or continuation.

**S2 — Fixed deployed mechanism.** The same proposal/filter/commitment algorithm, including its randomization law, is used in both environments.

No smoothness, parametric coupling, or auxiliary labels are available.

## Assumptions for the Dynamic Safety–Identification Bound

**D1 — Catastrophic probe pair.** There is an untested action $b$ and two environments that are observationally identical before the first execution of $b$. In one environment $b$ is safe; in the other, its first execution causes the declared failure event with probability one.

**D2 — Uniform deployment safety.** The algorithm must keep the failure probability at most $\beta$ in every environment in the class.

**D3 — Honest identification.** The final declaration about $b$ must be correct in both environments, not only in the actually realized one.

These assumptions create an intentionally stark lower bound. They do not model graded collision probabilities, smooth geometry, or simulator information.

## Structures That Can Restore Learnability

Any positive result must explicitly introduce at least one of:

- positive execution/continuation propensity;
- Lipschitz, RKHS, monotonicity, linearity, or other model structure;
- an initial safe set and safe reachability/returnability relation;
- a trusted baseline or shield model;
- a validated simulator or digital twin;
- an instrumental intervention or shadow variable;
- auxiliary data from a different missingness mechanism;
- a bounded-risk probe protocol;
- a known relationship between charger and task dynamics.

These are identification assumptions, not facts implied by persistent operation.
