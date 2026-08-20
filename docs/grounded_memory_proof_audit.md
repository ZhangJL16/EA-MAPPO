# Grounded Memory Proof Audit

## Audited statements

### Modular error bound

- Status: valid under explicit separability and Lipschitz assumptions.
- Gap: does not establish recursive information-state accuracy.
- Prior-art classification: existing composition of approximate-information-state/value-continuity arguments.

### Certified-channel non-interference

- Status: valid when the certified constraint set is independent of learned messages and the solver returns a feasible action.
- Gap: safety still fails if the certified set does not contain truth, data association is wrong, or sampled-data premises fail.
- Prior-art classification: existing modular safety-filter and measurement-robust CBF result.

### Uncertainty-set contraction to action-set expansion

- Status: valid for nested **certified** sets and support-function robust constraints.
- Gap: a learned narrow set is not certified merely because its empirical error is small.
- Prior-art classification: standard robust optimization monotonicity.

### Action-set expansion to intervention reduction

- Status: valid pointwise for the same nominal action and objective.
- Gap: does not imply trajectory-level path or energy improvement.
- Prior-art classification: immediate optimization corollary.

## Counterexample audit

| Counterexample | Conditional theorem survives? | End-to-end candidate survives? |
|---|---:|---:|
| object identity swap | only if certificate invalidates/reset | not demonstrated |
| new obstacle | only if instantiated in certified channel | not demonstrated |
| long dropout | yes if interval expansion remains valid | likely becomes infeasible/conservative |
| abrupt maneuver outside jerk bound | no | no |
| random learned route message | yes | performance may degrade |
| random learned energy message | yes for collision | energy behavior may degrade |
| learned memory zeroed | yes through certified baseline | benefit disappears |
| unsound learned radius contraction | no | no |

## Audit conclusion

- `PROOF_DRAFT_VALID_FOR_NARROWED_CLAIMS = TRUE`
- `THEORETICAL_NOVELTY_SUPPORTED = FALSE`
- `CENTRAL_MISSING_THEOREM = learned grounding -> sound certified information contraction`

## Final verified-contraction audit

- The bounded-Taylor historical strip is algebraically correct under bounded jerk.
- Intersection containment is valid only when each accepted strip independently contains truth.
- LP feasibility checks mutual consistency, not truth containment.
- A learned or stale association can produce a nonempty, internally consistent intersection that excludes the target obstacle.
- Therefore `association_verified` and the sensor bound must come from a trusted evidence path; the current simulator supplies an oracle flag only.
- The proposed recall-to-contraction theorem is not valid without additional geometric assumptions because log-volume gain for arbitrary polyhedral intersections is not established as submodular.
- The 500-snapshot test verifies the declared bounded simulator slice, not deployment-wide certification.

## Final status

- `VERIFIED_INTERSECTION_LEMMA = VALID_CONDITIONALLY`
- `LEARNED_PROPOSAL_NON_INTERFERENCE = VALID_ONLY_WITH_SOUND_VERIFIER_PREMISES`
- `CONTRACTION_RATE_THEOREM = NOT_ESTABLISHED`
- `INDEPENDENT_ASSOCIATION_CERTIFICATE = MISSING`
- `THEORETICAL_NOVELTY_SUPPORTED = FALSE`
