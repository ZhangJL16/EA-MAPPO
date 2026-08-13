# Author Response and Revision Ledger — Round 1

**PROVENANCE COPY:** canonical continuing-work classification is in `REBUTTAL_ROUND_1.md`.

Mode: revision ledger plus evidence-bounded response. No experimental result is claimed.

## Overall Response

We agree with the panel's central diagnosis: the component combination is not novel, and the current package is not CCF-A ready. The response therefore narrows the contribution to a falsifiable coupling and converts every empirical defense into a required experiment. The theory response only defends conditional identities actually proved.

## Concern Ledger

| ID | Concern | Classification | Response / action | Evidence | Status |
| --- | --- | --- | --- | --- | --- |
| A1 | Adaptive action selection can destroy marginal coverage | FIXED_BY_THEORY | A10 and distributional derivation require selected-action, simultaneous, or policy-coupled calibration; implementation must test candidate-set size | `ASSUMPTIONS.md`, `DISTRIBUTIONAL_ENERGY_DERIVATION.md` | CLOSED IN THEORY |
| A2 | Charger policy may be improper | ACCEPTED_LIMITATION | Properness is a declared domain assumption; empirical failure states must be reported | A4, T2, `CLAIM_BOUNDARIES.md` | CLOSED AS LIMITATION |
| A3 | Repeated energy checks inflate risk | FIXED_BY_THEORY | T3 is a commitment-time implication; repeated claims require anytime calibration not presently claimed | `JOINT_SAFETY_THEOREM.md` | CLOSED IN THEORY |
| A4 | Local observations may be non-Markov | FIXED_BY_DESIGN | New critics accept history/context; memoryless variant is an ablation, not theorem default | A1 | IMPLEMENTATION REQUIRED |
| B1 | Distance baseline may replace TD | REQUIRES_EXPERIMENT | E1 compares distance×Wh/m, MC, scalar TD, distributional TD, calibrated TD | `NOVELTY_MATRIX.md` | OPEN |
| B2 | CMDP/MaxSafe may replace dual timescales | FATAL_IF_UNRESOLVED | Add matched single-timescale/action-mask baseline to E3; route fails if effect is unchanged | `NOVELTY_MATRIX.md` | OPEN |
| B3 | State-only value may replace action-conditioned value | REQUIRES_EXPERIMENT | Add state-only `V_E(s')` ablation with identical capacity/data | E1/E3 design | OPEN |
| B4 | Separate charger planner may replace shared-policy coupling | REQUIRES_EXPERIMENT | Add collision-unaware/separate-return-policy ablation when E2 competence is available | E3 design | OPEN |
| B5 | Policy-coupled conformal work dominates calibration framing | ACCEPTED_LIMITATION | No new conformal-theory claim; use it as closest work and selection-aware protocol | `CLOSEST_WORK.md` | CLOSED BY CLAIM NARROWING |
| C1 | Completed-return survivor bias | FIXED_BY_DESIGN | Preserve censoring/failure labels; never treat truncation as completed return; report completion strata | `TD_DERIVATION.md` | IMPLEMENTATION REQUIRED |
| C2 | Unequal action-search budgets | FIXED_BY_DESIGN | Shared candidate action stream and equal inference/candidate budget across action-filter methods | experiment config contract | IMPLEMENTATION REQUIRED |
| C3 | Calibration leakage | FIXED_BY_DESIGN | Four disjoint streams: train, calibration, validation, final held-out evaluation | experiment protocol | IMPLEMENTATION REQUIRED |
| C4 | Policy drift changes target | FIXED_BY_DESIGN | Freeze navigation in first round; record policy hash; joint fine-tuning deferred | A5, TD data semantics | IMPLEMENTATION REQUIRED |
| C5 | Stopping improvement may be early-commit confound | REQUIRES_EXPERIMENT | Log trigger state/time/energy/path and compare reversible, hysteresis, commitment at matched thresholds | E5 | OPEN |
| AC1 | Obvious combination | FATAL_IF_UNRESOLVED | No rhetorical rebuttal. Only B1--B4 causal results can close it; otherwise narrow to negative/benchmark study or terminate route | Reviewer B ledger | OPEN |
| AC2 | No evidence | REQUIRES_EXPERIMENT | Formal runs only after tests; smoke is excluded | experiment gate | OPEN |

## Exact Theory Changes Already Made

- Added bounded/integrable stage-cost premise to the finiteness theorem.
- Restricted calibration claims to selected-action/simultaneous/matched-distribution modes.
- Restricted the energy probability implication to the actual commitment decision.
- Added empty-feasible-set and partial-observability failure boundaries.
- Removed any independent novelty claim for quantile TD, conformal filtering, action masking, SSP, or commitment.

## Promises Avoided

- no promise that experiments will support the route;
- no promise of acceptance or score improvement;
- no “first,” “guaranteed,” or “provably safe” wording;
- no claim that background runs are results before completion and evaluation.

## Round-2 Trigger

A theory-only Round 2 is unnecessary because no theorem-level FATAL remains. A results-aware CCFA review is mandatory after immutable E1--E6 artifacts exist. Novelty remains fatal for a paper until the replacement attacks are experimentally resolved.
