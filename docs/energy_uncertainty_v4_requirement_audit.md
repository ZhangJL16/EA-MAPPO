# Energy Uncertainty v4 Requirement Audit

| Requirement | Status | Authoritative evidence |
| --- | --- | --- |
| Keep the MC point estimator | Complete | `PREREGISTERED_CONFIG.json` records `retrained=false`; checkpoint SHA-256 remains `86b371...f92b`. |
| Do not retrain SAC | Complete | `COMPLETED.json` records `sac_retrained=false`; checkpoint SHA-256 remains `df9a435...1ff0a`. |
| Separate Goal and Mission uncertainty | Complete | `SeparatedGoalMissionRiskEstimator` exposes independent Goal-context and Mission-context paths; switching tests use `return_now_upper95` and direct `mission_upper95`. |
| Freeze Mission adaptive Laplace | Complete | Preregistration points to the prior model and calibration hashes and records no v4-stage retraining or tuning. |
| Avoid hand-expanded margins/reserve changes | Complete | Goal corrections are calibration-derived; reserve remains 10% in preregistration and environment tests. |
| Diagnose trajectory maximum underestimation | Complete | Development and fresh-v4 trajectory residual CSVs contain mean/std/P90/P95/P99/max, dynamics, path, position, boundary, and consecutive-contact fields. |
| Analyze remaining-distance effects | Complete | Reporting-only `state_residual_by_remaining_distance.csv` and fresh-v4 correlation output were added without changing model selection or calibration. |
| Test 7D information loss | Complete | A/B/C/D feature modes cover state7, absolute position, six directional boundary distances, and compact position/boundary context. |
| Keep obstacles/LiDAR/CBF off | Complete | v4 collection retains the obstacle-free environment; no collision module was introduced. |
| Evaluate G1/G2/G3 | Complete | `selection/goal_candidate_matrix.csv` contains 17 fixed candidate/calibration combinations and complexity metrics. |
| Learn positive residual for G3 | Complete | `positive_underestimation_target` and `PositiveResidualQuantileModel` implement point/risk separation. |
| Keep Goal/Mission switching semantics | Complete | Environment uses immediate return upper first, then direct Mission upper, with unchanged reserve semantics and one-way commitment. |
| Measure unnecessary return | Complete | Mission evaluation reports unnecessary-return proxy rates at all four risk levels; operational measurement is intentionally unavailable because 100k was gated off. |
| Evaluate 90/95/97.5/99% | Complete | Fresh-v4 Goal and Mission JSONs contain all four levels with coverage and width metrics. |
| Freeze before v4 | Complete | `PREREGISTERED_CONFIG.json` predates v4 collection and fixes models, hashes, features, calibration, seeds, groups, reserve, and thresholds. |
| Use untouched v4 | Complete | Leakage audit reports distinct seeds and disjoint IDs; v4 was not used for selection. |
| Collect at least 3,000 Goal / 2,000 Mission trajectories | Complete | Fresh-v4 manifests and leakage audit record exactly 3,000 and 2,000 successful trajectories. |
| Report Goal primary groups and uncertainty | Complete | Goal JSON and readiness report contain overall, TASK, CHARGER, distance, intersections, counts, and Wilson 95% intervals. |
| Report Mission primary groups | Complete | Mission JSON and readiness report contain overall and five 400-mission distance groups with Wilson intervals. |
| Compare safety, efficiency, accuracy, complexity | Complete | Validation report includes coverage, worst groups, widths, conservatism, unnecessary-return proxy, point errors, parameters, and latency. |
| Run 100k only after offline PASS | Complete | Offline gate failed; `phase2_100k/SKIPPED.json` records `fresh_v4_offline_gate_failed`. |
| Do not run formal 500k Phase 2 | Complete | `COMPLETED.json` records `formal_500k_phase2_launched=false`; no matching process is running. |
| Add future safety extension | Complete | Validation report states the required data regeneration and compact safe-trajectory context path. |
| Add and run tests | Complete with unrelated legacy failure | Focused suite: 108 passed. New-route suite: 58 passed. Full root suite: 206 passed and one unrelated superseded certified-control acceptance test failed. |
| Produce explicit final decision | Complete | Validation report sets both offline-gate and final-500k decisions and explains why delivery-efficiency evidence remains inconclusive. |

## Completion Boundary

The objective is complete as a falsification result, not as a successful safety
gate. The preregistered 95% candidate failed important fresh-v4 subgroups, so
the required scientific action is to stop before the 100k and 500k Phase 2
runs. Running either would violate the stated protocol rather than complete it.
