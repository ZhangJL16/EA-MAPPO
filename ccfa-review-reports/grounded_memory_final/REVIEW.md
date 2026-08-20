# Final Hostile Review: Control-Grounded Structured Memory

## Mode

Scientific, theory/novelty/methodology panel with AC synthesis.

## Paper Summary

The candidate routes typed Object, Safety, Route, and Energy memories through a fixed graph. Learned modules propose historical constraints and route/energy messages; an analytic verifier and hard HOCBF remain authoritative.

## Reviewer A — Theory

- **Score:** 6/10
- **Confidence:** 5/5
- **Strength:** The valid-intersection and learned-message non-interference statements are correctly scoped.
- **Fatal concern:** Feasibility of a proposed historical intersection is not evidence that it contains truth. The theorem requires independently trusted sensor and association premises.
- **Exact counterexample:** A wrong but mutually consistent track history can define a nonempty smaller polytope that excludes the target obstacle.
- **Score-change condition:** A deployable association/error verifier plus a nontrivial finite-sample contraction guarantee not inherited from set-membership estimation.

## Reviewer B — Novelty

- **Score:** 3/10
- **Confidence:** 5/5
- **Fatal concern:** The method is set-membership estimation plus learned constraint screening plus an independent verifier.
- **Evidence:** C3 generic GRU, C4 FOGM, and C5 oracle have identical contraction metrics; all-history is tighter and only about 1.15 ms slower than FOGM.
- **Exact counterexample to differentiation:** Replacing FOGM with deterministic all-history intersection preserves containment and improves set tightness.
- **Score-change condition:** A theorem-level property that fails for standard constraint pruning/active-set learning but holds for typed grounded routing.

## Reviewer C — Methodology

- **Score:** 5/10
- **Confidence:** 5/5
- **Strength:** The final diagnostic uses 100,122 real safety-filtered transitions, trajectory splits, and three training seeds.
- **Major concern:** FOGM route and Safe Energy-to-Go prediction are worse than simpler baselines in all three seeds.
- **Closed-loop concern:** No-memory has the best path ratio; explicit hysteresis has lower intervention, fewer switches, and lower energy than FOGM.
- **Score-change condition:** A pre-registered downstream metric with 3/3 stable improvement and a matched simple-memory baseline.

## Area Chair Synthesis

- **Overall:** 4/10, Reject
- **Theory:** 6/10
- **Novelty:** 3/10
- **Methodology:** 5/10
- **Novelty total:** 11/30
- **Fatal/critical unresolved:** 2 — independent association certification; no theorem-level separation from set-membership/constraint-pruning work.

The narrow H40 energy-overhead prediction improvement is acknowledged but does not offset the negative Route, total Energy-to-Go, and closed-loop results. The long experiment gate is not met.

## Final Classification

`ENGINEERING_ONLY`

`LONG_EXPERIMENT = NOT_LAUNCHED`

