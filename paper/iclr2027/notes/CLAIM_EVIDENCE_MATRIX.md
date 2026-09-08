# Claim--Evidence Matrix

| Claim | Source / test | Status | Paper location | Falsification condition | Allowed wording now | Allowed after PASS |
| --- | --- | --- | --- | --- | --- | --- |
| Target Resource-to-Go is identifiable without pair overlap under target executed-interface identification | Executed-interface identification theorem / proof package | PROVED LOCALLY; independent audit pending | Sec. 4, App. A | proof audit finds a path-measure or null-set gap | “Under the stated queryability, shared-primitive, target-interface, and proper-SSP assumptions, we prove…” | same after independent proof PASS |
| Resource-to-Go is not identifiable outside interface support without extra structure | Non-identification proposition | PROVED LOCALLY; independent audit pending | Sec. 4, App. A | counterexample model class invalid for declared setting | “Over a structure-free class, no estimator can uniformly identify…” | same after proof PASS |
| Long-horizon error is bounded by occupancy-weighted local transport plus SSP tail | Occupancy-weighted transport theorem | PROVED only under coupling/Lipschitz assumptions | Sec. 5, App. A | continuation regularity fails or primitive mismatch omitted | “Under explicit coupling and Lipschitz assumptions…” | no broader wording without new theory |
| Prediction changes threshold decisions only near the return boundary | Return-boundary stability theorem | PROVED LOCALLY; independent audit pending | Sec. 5, App. A | algebraic proof error | “Bounded error can change this threshold only…” | same after proof PASS |
| Pair novelty is weaker than interface extrapolation empirically | leave-one-pair-out factorial | PENDING | Intro, RQ2, Results | pair identity dominates after support/horizon control | “We investigate whether…” | “Error tracks interface extrapolation after…” with audited effect |
| Oracle Resource-to-Go improves decisions | Oracle Gate | PENDING | Abstract, RQ1, Results | <5% throughput gain at matched stranding | “We test whether…” | exact audited improvement with CI |
| Reliability detects severe future underestimation | risk--coverage Gate | PENDING | RQ3, Results | no AURC gain over ensemble/kNN | “We evaluate whether…” | named improvement and CI |
| Reliable Resource-to-Go improves stranding--throughput frontier | paired cycle evaluation | PENDING | Abstract, RQ4, Results | frontier is not improved | “The planned evaluation tests…” | exact Pareto claim with provenance |
| Phenomenon generalizes beyond primary simulator/filter | second domain/family | PENDING | RQ5, Discussion | regimes do not replicate | “requires replication” | scoped two-setting conclusion |
| SIRP is necessary | residual mechanism Gate | NOT YET JUSTIFIED | Method | ensemble explains failures | “gated candidate” only | retain only after predeclared Gate PASS |

## Forbidden current wording

- “We show empirically that interface extrapolation governs failure.”
- “Our method prevents stranding” or “guarantees energy safety.”
- “q95/q99 Resource-to-Go” as a physical tail in the current deterministic simulator.
- “SIRP outperforms ensembles.”
- “The paper is domain-general.”
