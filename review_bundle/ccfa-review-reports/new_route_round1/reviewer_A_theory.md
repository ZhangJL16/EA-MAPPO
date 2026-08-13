# Reviewer A — Theory / Mathematical Soundness

Mode: scientific proposal review
Venue assumption: CCF-A-level ML/robotics conference, method paper
Materials: active files in `docs/new_theory/`; no experimental results

## Summary and Score

The SSP energy definition is now semantically precise and the proof scope is substantially narrower than a safety certificate. The main theorems are conditional implications. I find no fatal algebraic error, but the calibration mode after adaptive action selection and the empirical domain of charger-policy properness are decisive.

**Overall: 6/10 (borderline positive on theory only). Confidence: 4/5.**

## Scorecard

| Dimension | Score | Confidence | Evidence basis | Deduction / score-change condition |
| --- | ---: | ---: | --- | --- |
| Novelty | 2/5 | 3/5 | SSP/TD/conformal ingredients are established | Theory novelty is weak; raise only with a distinct policy-coupled result or mechanism |
| Soundness | 4/5 | 4/5 | T1--T6 and proof audit | Conditional coverage mode must be fixed in implementation |
| Evidence | 1/5 | 5/5 | No results exist | Requires calibration/properness tests and E1--E6 |
| Significance | 4/5 | 3/5 | Persistent UAV stranding/collision problem | Requires demonstrated operational-envelope effect |
| Clarity | 4/5 | 4/5 | Stable notation and explicit non-claims | Clarify selected-action calibration protocol in algorithm spec |
| Reproducibility | 3/5 | 3/5 | Planned provenance only | Raise after immutable configs/tests/scripts exist |
| Ethics / Limitations | 5/5 | 4/5 | `CLAIM_BOUNDARIES.md` | Maintain real-UAV non-claim |

## Fatal Concerns

None at theorem level after the added assumptions. A future manuscript becomes fatally unsound if it converts marginal conformal coverage into pointwise/adaptive action safety.

## Major Concerns and Exact Counterexamples

1. **Adaptive selection counterexample.** Two action groups each contribute equally to calibration; all undercoverage is concentrated in group B. The selector always chooses B because its predicted bound is lower. Marginal 95% coverage can coexist with arbitrarily poor selected-action coverage. Required fix: calibrate the selector output or use simultaneous/policy-coupled coverage.
2. **Improper charger policy.** In a cul-de-sac, collision avoidance can circle indefinitely with positive energy cost. The Bellman equation remains formal while `Q_E=infinity`. Required fix: declare an empirical proper-policy domain and report failure states.
3. **Repeated energy checks.** A 5% violation bound invoked 100 adaptively chosen times is not a 5% sortie guarantee. Required fix: make T3 a commitment-time statement or implement anytime spending.
4. **Partial observability.** Identical local scans can correspond to distinct global charger routes. A memoryless critic cannot represent both return distributions. Required fix: history/belief input or restrict environment aliasing.

## Score-Change Conditions

- Raise to 7: selected-action calibration is implemented and tested; empirical properness/failure-domain audit is explicit.
- Lower to 3: paper uses “provably safe,” hides empty feasible sets, or reports deep-TD convergence.
