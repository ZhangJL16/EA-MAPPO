# Reviewer A — Theory and Proof Audit

## Score

- Mathematical soundness after repair: **6.0 / 10 for conditional lemmas**
- End-to-end safety theorem: **2.0 / 10**
- Confidence: **0.95**
- Recommendation: **Reject as an end-to-end certificate**

## Round-1 Critical Findings

1. Ego rollout error was absent from the clearance margin.
2. The future-tube API accepted a bare box without acknowledging the required
   true-current-state-containment premise.
3. The code implemented a switchable-acceleration lower bound while the revised
   theorem concerned bounded jerk.
4. The document conflated the exact projected polytope with its coordinatewise
   interval hull.
5. The one-hold argument was described as invariant induction although its
   premises were re-assumed at each sample.
6. The jerk lower bound needed pointwise minimax quantifiers and admissibility
   under velocity/acceleration bounds.
7. Scalar radius ordering was insufficient for differently centered tubes.

## Repaired Status

- T1 now distinguishes the exact set from its interval hull and proves both
  contractions for nonempty common-model sets.
- T2 explicitly requires independently valid present-state containment and a
  bounded measurable future jerk.
- T3 includes exact ego realization or a pathwise ego-error tube and requires
  full-hold verification.
- The repeated-hold statement is now a union of independently certified holds,
  not a recursive-feasibility or invariant-set theorem.
- Separate helper functions implement second-order acceleration switching and
  third-order bounded-jerk lower bounds.
- The jerk impossibility statement uses
  `for all tau, for all predictor, exists sign` quantifiers.
- Search-set monotonicity now assumes full obstacle-tube inclusion.
- `INFEASIBLE` is explicitly excluded from robust certification metrics.

The same reviewer performed a final narrow recheck of the fail-open repair and
returned **RESOLVED**.

## Final Proof Status

- **0 FATAL**
- **0 CRITICAL**
- **0 unresolved MAJOR after the fail-open metric repair**
- T1--T5 and Proposition 1: **conditionally valid and standard**
- Adaptive maximum-feasible-window certification T6: **disproved**
- Recursive feasibility: **not proved**

The analytic subresults can support implementation documentation. They do not
establish that the proposed adaptive history selector is safe or that a
certified action exists each cycle.
