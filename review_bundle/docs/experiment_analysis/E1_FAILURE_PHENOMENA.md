# E1 Failure Phenomena

Date: 2026-08-13

## Evidence Rule

`OBSERVED` requires a valid comparison artifact or a direct protocol/runtime fact. `INSUFFICIENT_DATA` is used when the interrupted seed-0 collection can at most suggest a pattern. No item is promoted from a partial checkpoint to a paper claim.

| ID | Phenomenon | Status | Evidence |
|---|---|---|---|
| F1 | Distance × energy/m systematically underestimates in some states | INSUFFICIENT_DATA | No distance model was fitted or evaluated. Partial seed 0 has high energy–distance correlation (0.9603), but no held-out residual analysis. |
| F2 | Scalar TD has good mean error but severe tail underestimation | INSUFFICIENT_DATA | Scalar TD never ran. |
| F3 | Distributional TD calibrates globally but fails by distance/SOC/horizon regime | INSUFFICIENT_DATA | Distributional fitting and calibration never ran; SOC has essentially no variation. |
| F4 | Calibration achieves coverage by becoming too conservative | INSUFFICIENT_DATA | No calibrated bounds or task-throughput outcomes exist. |
| F5 | Earlier commitment starves low-SOC/far-return training data | INSUFFICIENT_DATA | The collector commits after one task action for every trajectory, so there is no commitment-time variation. Directly observed: initial SOC is always 1.0 and final mean SOC is 0.99953, making low-SOC learning impossible under this protocol. |
| F6 | TD bootstrap error accumulates with charger hitting horizon | INSUFFICIENT_DATA | TD never ran and no error-versus-horizon result exists. |
| F7 | Energy value is highly sensitive to the navigation policy | INSUFFICIENT_DATA | Each seed uses one frozen policy; no within-seed policy intervention exists. |
| F8 | An old energy critic fails after a policy checkpoint change | INSUFFICIENT_DATA | No trained energy critic exists and no policy-drift evaluation ran. |
| F9 | Similar Euclidean distances can require materially different return energy | INSUFFICIENT_DATA | Partial trajectories show nonzero path/energy variation, but only 43 invalid trajectories and no preregistered matched-distance analysis. |
| F10 | Energy learning improves with sorties or saturates early | INSUFFICIENT_DATA | No estimator checkpoint or data-scaling curve exists. |

## Runtime and Protocol Failures Actually Observed

1. **OBSERVED — detached-run lifecycle failure:** all recorded PIDs are dead while stale `RUNNING`/`INITIALIZED` markers remain; logs are empty and no completion/failure summary exists.
2. **OBSERVED — finite-energy protocol mismatch:** the experiment uses a 1000-unit nonterminating navigation budget; all 43 partial initial SOC values equal 1.0 and final mean SOC is 0.9995265.
3. **OBSERVED — switching question not instantiated:** immediate charger commitment is hard-coded after one task action, with no threshold baselines or learned stopping decision.
4. **OBSERVED — no formal estimator evidence:** none of B0–B4 reached fitting, calibration, evaluation, or checkpoint output.

These protocol/runtime failures are currently more important than any hypothesized learning failure.
