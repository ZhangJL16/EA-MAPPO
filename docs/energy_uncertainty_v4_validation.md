# Energy Uncertainty v4 Validation

## Decision

- `FRESH_V4_OFFLINE_GATE_PASSED = FALSE`
- `SHORT_PHASE2_100K_STATUS = SKIPPED_BY_PREREGISTERED_GATE`
- `FORMAL_PHASE2_500K_STATUS = NOT_STARTED`
- `DOES_ADAPTIVE_UNCERTAINTY_IMPROVE_DELIVERY_EFFICIENCY = INCONCLUSIVE`
- `RECOMMENDED_FINAL_500K_CONFIG = NONE`

The selected Goal and Mission uncertainty estimators both passed their overall
95% trajectory-coverage checks, but both failed preregistered, sufficiently
populated subgroups. The protocol therefore correctly skipped the matched 100k
Phase 2 comparison. No post-v4 tuning is permitted on this test set.

## Frozen Inputs And Provenance

The run reused the existing point estimator and SAC without retraining.

| Component | Frozen artifact | SHA-256 |
| --- | --- | --- |
| MC point estimator | `artifacts/uav_energy_delivery_mc_formal_20260817_154321/energy_model/best_validation.pt` | `86b371ca92d6cdd337dc44a67e84f3242decd5661921be0f2fe1d5e4fce8f92b` |
| SAC | `artifacts/uav_energy_delivery_v3_formal_20260816_004619/phase1_navigation/checkpoint_transition_500000.zip` | `df9a4354420a1a043b60d704292c6fe06ddd2127e937d3d9c72278b287c1ff0a` |
| Frozen Mission risk model | `artifacts/uav_energy_adaptive_uncertainty_20260817_214303/models/mission_heteroscedastic_laplace.pt` | `268b2994453bd84407cf6510fe6382f2afa36533ed4da444685351271f771da6` |

The preregistration was frozen before fresh-v4 collection. Fresh-v4 used goal
seed `720001`, mission seed `730001`, 3,000 Goal trajectories, and 2,000 Mission
trajectories. The leakage audit confirms disjoint trajectory identifiers and
distinct seeds across training, validation, calibration, diagnostics, and v4.

## Goal Undercoverage Diagnosis

The undercoverage is heterogeneous and trajectory-dependent rather than a
single global offset. On the development diagnostic set, physical features most
associated with trajectory maximum underestimation were path ratio
(`Spearman rho=0.3963`), minimum distance to a boundary (`rho=-0.2494`), maximum
acceleration (`rho=0.2468`), boundary-contact duration (`rho=0.2020`), flight
time (`rho=0.1961`), and initial goal distance (`rho=0.1715`). Fresh-v4
reporting-only correlations reproduced the pattern: path ratio `0.4191`, maximum
acceleration `0.2821`, minimum boundary distance `-0.2751`, maximum consecutive
boundary contacts `0.2364`, mean remaining distance `0.1833`, and P90 remaining
distance `0.1886`.

TASK trajectories were more difficult than charger-directed trajectories. In
fresh v4, trajectory maximum underestimation had mean `0.4091` and P95 `1.0908`
for TASK distances 2,500--4,000 m, and mean `0.5027` and P95 `1.2177` above
4,000 m. Corresponding charger-directed means were approximately `0.2490` and
`0.2476` in the available long-distance buckets.

State-level positive underestimation also widened with remaining distance. Its
P95 was `0.1428` at 0--100 m, `0.1893` at 1,500--2,500 m, `0.3962` at
2,500--4,000 m, and `0.4772` above 4,000 m. Maximum observed underestimation in
the latter two remaining-distance buckets was `2.4234` and `2.2863`.

These results support `7D_STATE_SUFFICIENCY = PARTIALLY_INSUFFICIENT`. Adding
compact absolute-position and boundary context improved the diagnostic 95%
coverage from `94.80%` to `95.05%`, improved the worst primary subgroup from
`91.56%` to `92.21%`, and reduced mean width from `0.5643` to `0.5502` relative
to the otherwise matched G3 state7 model. The improvement is real but small;
the remaining tail cannot be attributed only to static position aliasing.
Trajectory/path effects, acceleration, and boundary interaction remain relevant
unobserved context for the per-state risk model.

## Controlled Goal Ablation

All figures below are from the architecture-diagnostic split at nominal 95%,
before fresh-v4 evaluation.

| Method | Features | Overall coverage | Worst primary coverage | Mean width | Parameters | Latency us/state |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| G1 group conformal | state7 | 95.10% | 92.21% | 0.7046 | 0 | 0 |
| G2 group | state7 | 93.80% | 90.26% | 0.4638 | 17,665 | 0.0356 |
| G2 group | compact position/boundary | 94.55% | 91.56% | 0.4904 | 18,433 | 0.0420 |
| G3 group | state7 | 94.80% | 91.56% | 0.5643 | 18,052 | 0.0713 |
| G3 group | compact position/boundary | 95.05% | 92.21% | 0.5502 | 18,820 | 0.0655 |

No candidate passed every diagnostic subgroup. G3 with compact position and
boundary context was frozen because it gave the best safety/tightness tradeoff,
not because it had already established the final safety gate.

## Fresh-v4 Goal Results

The selected method was `g3_compact_position_boundary_group`: positive residual
quantile prediction followed by one-sided complete-trajectory group conformal
calibration. Point-model MAE was `0.0885`, RMSE was `0.1661`, and mean signed
bias was `-0.00394`.

At nominal 95%:

| Group | Trajectories | Coverage | Wilson 95% CI | Mean width |
| --- | ---: | ---: | ---: | ---: |
| Overall | 3,000 | 95.30% | [94.48%, 96.00%] | 0.5486 |
| TASK | 1,154 | 94.54% | [93.08%, 95.71%] | 0.7200 |
| CHARGER | 923 | 95.12% | [93.54%, 96.34%] | 0.4028 |
| 100--500 m | 693 | 95.38% | [93.55%, 96.71%] | 0.5263 |
| 500--1,500 m | 693 | 95.96% | [94.22%, 97.19%] | 0.4749 |
| 1,500--2,500 m | 693 | 95.67% | [93.89%, 96.95%] | 0.4806 |
| 2,500--4,000 m | 691 | 94.65% | [92.71%, 96.09%] | 0.5016 |
| >4,000 m | 230 | 93.91% | [90.04%, 96.34%] | 0.8070 |
| TASK x 2,500--4,000 m | 231 | 93.07% | [89.05%, 95.69%] | 0.6981 |

The worst preregistered primary subgroup was TASK x 2,500--4,000 m at 93.07%.
TASK, both long-distance groups, and several TASK/CHARGER intersections failed
the point target of 95% despite adequate sample counts.

### Goal Safety--Efficiency Tradeoff

| Nominal level | Overall trajectory coverage | Worst subgroup | Mean width | P95 width | Mean conservatism |
| --- | ---: | ---: | ---: | ---: | ---: |
| 90% | 90.27% | 87.45% | 0.4337 | 0.7932 | 0.4298 |
| 95% | 95.30% | 93.07% | 0.5486 | 1.0255 | 0.5447 |
| 97.5% | 98.00% | 96.54% | 0.7346 | 1.4517 | 0.7306 |
| 99% | 99.57% | 97.84% | 1.0808 | 1.9172 | 1.0769 |

The 97.5% setting clears the empirical 95% subgroup target on v4, but selecting
it after inspecting v4 would violate preregistration. It is therefore only a
reported tradeoff, not a validated replacement configuration.

## Fresh-v4 Mission Results

The frozen Mission heteroscedastic Laplace model retained better overall
tightness than summing two component upper bounds, but failed the longest
distance group.

At nominal 95%:

| Group | Missions | Coverage | Wilson 95% CI | Mean width |
| --- | ---: | ---: | ---: | ---: |
| Overall | 2,000 | 96.60% | [95.71%, 97.31%] | 1.0602 |
| 100--500 m | 400 | 97.00% | [94.83%, 98.28%] | 1.0341 |
| 500--1,500 m | 400 | 98.00% | [96.10%, 98.98%] | 1.0148 |
| 1,500--2,500 m | 400 | 98.25% | [96.43%, 99.15%] | 1.0250 |
| 2,500--4,000 m | 400 | 96.00% | [93.60%, 97.52%] | 1.0574 |
| >4,000 m | 400 | 93.75% | [90.94%, 95.73%] | 1.0942 |

The mean Mission conservatism cost was `1.1316`. The offline unnecessary-return
proxy was `0.3695%`, and the estimated task-acceptance rate was `80.02%`.

At the same nominal 95%, summing Goal component bounds reached 99.60% overall
coverage but widened the mean bound from `1.0602` to `1.3846`, increased mean
conservatism from `1.1316` to `1.4560`, and increased the unnecessary-return
proxy from `0.3695%` to `0.4788%`. This supports keeping a distinct Mission risk
model, but it does not rescue the failed >4,000 m gate.

### Mission Safety--Efficiency Tradeoff

| Nominal level | Overall coverage | Worst distance coverage | Mean width | P95 width | Unnecessary-return proxy |
| --- | ---: | ---: | ---: | ---: | ---: |
| 90% | 92.35% | 89.50% | 0.7815 | 0.9680 | 0.2830% |
| 95% | 96.60% | 93.75% | 1.0602 | 1.3270 | 0.3695% |
| 97.5% | 97.90% | 95.25% | 1.2150 | 1.5621 | 0.4219% |
| 99% | 98.90% | 97.75% | 1.3866 | 1.8399 | 0.4795% |

## Phase 2 Decision

The preregistered rule allowed the matched 100k Phase 2 experiment only after
both fresh-v4 offline gates passed. Goal and Mission both failed subgroup gates,
so `phase2_100k/SKIPPED.json` records
`fresh_v4_offline_gate_failed`. Consequently there are no valid operational
numbers for tasks/1,000 transitions, returns, exhaustion, or charger-arrival
SOC. Offline unnecessary-return proxies cannot substitute for that causal
comparison.

## Test Evidence

- Focused adaptive uncertainty, MC energy, conformal, TD-diagnostic, and UAV SAC
  tests: `108 passed`.
- `review_bundle/tests/new_route`: `58 passed`.
- Root `tests`: `206 passed`, `1 failed`, with the sole failure in the unrelated
  legacy certified-control acceptance test
  `test_open_trace_closes_execution_invariants`. This route does not modify or
  repair that superseded architecture.

## Future Safety Extension

Obstacles, LiDAR, and CBF were intentionally excluded. Adding a collision-safety
filter changes the executed trajectory distribution and therefore invalidates
the current energy calibration without further evidence. The next energy study
would need newly generated safe-trajectory MC data and retraining or fine-tuning
of both point and risk models. Candidate context should begin with compact safe
path length, detour ratio, clearance, and CBF-intervention statistics rather
than raw high-dimensional LiDAR.

## Final Answers

- `GOAL UNDERCOVERAGE ROOT CAUSE`: heterogeneous trajectory tail errors linked
  to TASK/long-distance regimes, path inefficiency, boundary proximity/contact,
  and acceleration; static 7D state aliasing is contributory but not sufficient
  to explain the residual tail.
- `7D STATE SUFFICIENCY`: `PARTIALLY_INSUFFICIENT`.
- `BEST GOAL RISK METHOD`: G3 positive-residual quantile with compact
  position/boundary context and one-sided trajectory group conformal.
- `BEST MISSION RISK METHOD`: frozen heteroscedastic Laplace Mission model.
- `100K PHASE2`: `SKIPPED_BY_PREREGISTERED_GATE`.
- `DOES ADAPTIVE UNCERTAINTY IMPROVE DELIVERY EFFICIENCY`: `INCONCLUSIVE`.
- `RECOMMENDED FINAL 500K CONFIG`: none; do not start a final 500k Phase 2 from
  this v4 result.
