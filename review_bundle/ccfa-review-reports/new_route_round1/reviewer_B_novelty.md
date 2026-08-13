# Reviewer B — Novelty / Closest-Work Attack

Mode: scientific proposal review with public closest-work search
Materials: `CLOSEST_WORK.md`, `NOVELTY_MATRIX.md`, active theory

## Summary and Score

The original “collision critic + energy critic + threshold” idea is an obvious combination. Quantile TD, distributional safe RL, conformal safety prediction/filtering, action masks, continuous shields, SSP learning, and persistent charging all have direct precedents. The reframed post-action charger-return semantics is specific and testable, but currently looks like a careful application rather than a non-obvious algorithmic contribution.

**Overall: 3/10 (reject). Confidence: 4/5.**

## Scorecard

| Dimension | Score | Confidence | Evidence basis | Deduction / score-change condition |
| --- | ---: | ---: | --- | --- |
| Novelty | 2/5 | 4/5 | Closest-work matrix, especially QR-DQN, calibrated safety chances, MaxSafe, policy-coupled coverage | Requires decisive mechanism evidence and stronger exact-work screening |
| Soundness | 4/5 | 3/5 | Conditional theorem package | Soundness does not create novelty |
| Evidence | 1/5 | 5/5 | No experiments | Required factorial ablations |
| Significance | 3/5 | 3/5 | Important deployment problem | Significance depends on measurable operational envelope |
| Clarity | 4/5 | 4/5 | Claims are narrow and falsifiable | Keep “not a new critic/calibration theorem” explicit |
| Reproducibility | 3/5 | 2/5 | Planned structure only | Requires runnable release |
| Ethics / Limitations | 5/5 | 4/5 | Strong non-claim discipline | No deduction |

## Fatal Concern

**FATAL_IF_UNRESOLVED: novelty collapse.** If E1--E6 show only that a better predictor plus conservative threshold reduces failures, the paper is an engineering composition of established components.

## Exact Replacement Attacks

1. Replace `U_E` with shortest-path distance times a high quantile Wh/m. If performance is unchanged, TD distribution learning is unnecessary.
2. Replace separate risks with MaxSafe/CMDP action masking plus battery in state. If joint performance is unchanged, dual timescales are exposition, not mechanism.
3. Replace post-action `Q_E(s,a)` with state-only `V_E(s)` evaluated after transition. If unchanged, action conditioning adds no value.
4. Use a separate charger planner rather than the shared goal policy. If unchanged, shared-policy coupling is not a contribution.
5. Calibrate logged actions but vary candidate-set size. If violations grow, the claimed calibration is selection-biased and policy-coupled conformal work dominates the framing.

## Score-Change Conditions

- Raise to 5: E1--E6 establish at least one non-substitutable mechanism under equal budgets, and full-text search finds no exact prior instantiation.
- Raise to 6: the method supplies a generally reusable selected-action return-cost calibration protocol beyond the UAV case.
- Keep at 3 or lower: only aggregate safety gains are shown without causal ablations.
