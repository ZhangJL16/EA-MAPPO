# Rebuttal Round 2

## Outcome

The panel's theory assessment is accepted. The novelty objection is not refuted. The project enters `RESEARCH_PROBLEM_REDESIGN` rather than claiming readiness.

| Concern | Classification | Evidence/action |
| --- | --- | --- |
| failure mass can make conditional energy coverage vacuous | FIXED_BY_PROBLEM_REFORMULATION | defective law explicitly reports success mass; experiments must report empty/failure rates |
| augmented properness remains assumed | ACCEPTED_LIMITATION | A4 and T10 prohibit universal reachability claims |
| continuous-action C3 is too narrow | RESEARCH_DIRECTION_BLOCKER | next redesign must derive a measurable boundary-mass transport result or remove C3 as central evidence |
| BMDP already conditions on risk budgets | ACCEPTED_LIMITATION | budget augmentation and Bellman construction removed from novelty claims |
| SDAC/QCRL cover distributional multiple constraints | ACCEPTED_LIMITATION | these become mandatory matched baselines |
| policy-coupled conformal already covers action-dependent prediction | ACCEPTED_LIMITATION | calibration is an ingredient, not a contribution |
| no strong theorem beyond C1--C3 | RESEARCH_DIRECTION_BLOCKER | triggers redesign around endogenous filter-update target drift |
| current code does not instantiate `(b,eta)` defective learner | REQUIRES_EXPERIMENT | no empirical/theory claim is inferred from existing E1 |
| censoring and version overlap unresolved algorithmically | FIXED_BY_ALGORITHM | next design must specify target-drift-aware collection/reuse and censoring treatment before implementation |
| E1 cannot validate coupling | REFUTED_BY_EXISTING_PROOF | canonical docs explicitly classify E1 as a collision-free special case only |

## Author Position

The strongest replacement—BMDP/SDAC/QCRL plus policy-coupled conformal and an absorbing failure state—currently covers the broad formulation. The only defensible next move is to ask whether learned collision-filter updates create a distinct, quantifiable return-target transport problem with an actionable reuse/calibration rule. If that also reduces to standard Markov-kernel perturbation and off-policy conformal prediction, the route must be declared blocked.
