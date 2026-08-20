# Memory Closed-Loop Experiments

## Status

The bounded controlled experiment is complete. No formal 500k run was started.

Final integrity-remediated artifact:
`artifacts/memory_closed_loop_controlled_final_audited_20260820/`.

- `60,152` actual controlled transitions, below the `100,000` ceiling;
- `12` matched scenarios: four motion families times three seeds;
- all nine executed-source hashes match the current files;
- Git base SHA: `657b08b3768c90966b32813a97b708bebf7fe175`;
- `formal_500k = false`.
- learned/Kalman/IMM development radii use the validation split; the held-out
  test split does not configure closed-loop behavior;
- all calibration/test NPZ files, predictions, checkpoints, and 13 executed
  source modules have recorded SHA-256 hashes; the estimator training-source
  hash matches its checkpoint artifact.

Two earlier final-like artifacts are explicitly invalid and excluded:

- `artifacts/memory_closed_loop_controlled_final_20260820_000136/` failed the
  proof audit;
- `artifacts/memory_closed_loop_controlled_final_round4_20260820_002216/`
  failed the fail-closed runtime audit.
- `artifacts/memory_closed_loop_controlled_integrity_fixed_20260820/` is
  invalid because a temporary post-hoc metric patch changed the estimator
  source hash; its `INVALID.json` forbids claims.

## Completed theorem counterexample slice

Artifact:
`artifacts/memory_counterexamples_tracking_delay_200k_20260820_002409/summary.json`.

- 100,000 bounded-residual states: prediction containment 100%, correction containment 100%, zero under-bound cases.
- 100,000 deliberately out-of-bound abrupt states: pre-reset containment 0.001% (`99,999` under-bound cases).
- Reset containment conditional on the declared base set: 100%; unconditional containment: 7.777%, because the deliberately generated states usually violated that base set too.
- 100,000 robust directional-HOCBF random uncertainty draws: zero negative-psi2 cases.
- Position radius after 0/1/2/3/5 dropout frames: 0.1000/0.1545/0.2260/0.3265/0.6625 per axis.
- A crossing identity swap separated by 0.08 m passes the single-track innovation gate, proving that innovation alone cannot certify association.

The same artifact adds the sensor-delay and 2/4/8/16/32-obstacle tracking
diagnostics. These are algebra/counterexample checks, not closed-loop
performance results.

## Controlled protocol

- Fresh train/validation/test motion sequences with disjoint seeds.
- Motion benchmark regimes: constant velocity, constant acceleration,
  sinusoidal, piecewise acceleration, sudden velocity change, sudden direction
  change, stop-go, crossing-like turn, and dropout burst. Identity swap is a
  separate handcrafted counterexample, not a train/validation/test regime.
- Prediction horizons: 0.25, 0.5, 1.0, and 2.0 seconds.
- Estimation baselines: current-only, CV-KF, CA-KF/EKF, IMM, MHE, raw-history MLP, ego-compensated MLP, RNN, GRU, LSTM, TCN, SSM, Physics-GRU, and explicit physical-memory residual model.
- Closed-loop baselines: exact obstacle state, noisy current-only LiDAR, Kalman robust HOCBF, IMM robust HOCBF, GRU robust HOCBF, Physics-GRU robust HOCBF, and the analytic-tube candidate.
- The exact-state Candidate A remains an oracle-style controller upper baseline.

## Metrics

Estimator: velocity/acceleration MAE, future-position MAE, abrupt adaptation
delay, tube coverage, full orthotope width, P95 directional support, and
under-bound count. The original 5k artifact did not store future-position MAE;
the exact post-hoc computation is recorded separately rather than mislabeling
directional support as P95 full width.

Safety/control: collisions, intersample violations, minimum clearance, HOCBF feasibility, uncertified fallback, intervention fraction/norm, freeze, success, path ratio, and energy.

Compute: memory update, tube propagation, QP, and end-to-end P99.

## Final controlled results

All learned/Kalman/IMM boxes in this table are empirical, model-selection-
independent calibration boxes and are explicitly `certification_valid=False`.
Zero simulated collision is therefore not a deterministic safety guarantee.

| Method | Success | Collision | Infeasible | Uncertified | Under-bound | Intervention | Freeze | Success path ratio | Success energy | P99 total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A exact-state Candidate A | 1.000 | 0 | 0.0252 | 0.0252 | 0 | 0.0284 | 0.0057 | 0.9890 | 2.5317 | 12.5861 ms |
| B current only | 0.583 | 0 | 0.3691 | 1.000 | 0.3701 | 0.3980 | 0.1779 | 1.0020 | 2.9988 | 25.4735 ms |
| C CA Kalman | 0.583 | 0 | 0.3697 | 1.000 | 0.0160 | 0.3787 | 0.1976 | 0.9908 | 2.6558 | 24.8767 ms |
| D IMM | 0.583 | 0 | 0.3697 | 1.000 | 0.0237 | 0.3710 | 0.1983 | 0.9907 | 2.6358 | 24.0862 ms |
| E ego L16 MLP | 0.667 | 0 | 0.3313 | 1.000 | 0.3428 | 0.3423 | 0.1496 | 0.9933 | 2.6198 | 24.6742 ms |
| F GRU | 0.667 | 0 | 0.3334 | 1.000 | **0.5184** | 0.3168 | 0.1606 | 0.9889 | 2.4632 | 21.9489 ms |
| G Physics-GRU | 0.583 | 0 | 0.3692 | 1.000 | 0.0423 | 0.3703 | 0.1657 | 0.9903 | 2.5827 | 19.3628 ms |
| H contractive memory | 0.583 | 0 | 0.3649 | 1.000 | 0.0538 | 0.3765 | 0.1788 | 0.9908 | 2.6607 | 27.4091 ms |
| I analytic interval observer | 0.000 | 0 | 1.000 | 1.000 | 0 | 1.000 | 1.000 | n/a | n/a | 24.2806 ms |

Minimum clearance for the oracle exact-state baseline was `2.6845 m`. Its
`2.5204%` infeasible/uncertified rate means the full trace is not wholly
certified even though it had no simulated collision. The only deterministic
interval candidate had no under-bounds but was operationally unusable: every
step fell back, every rollout froze, and no task succeeded.

The experiment does **not** establish an energy or intervention advantage at
equal deterministic certification. Apparent learned-method energy values are
conditioned on successful rollouts and come from uncertified empirical boxes.
They cannot be compared to the broad analytic interval as a same-safety causal
effect.

## Held-out future-position diagnostic

Artifact: `artifacts/memory_estimation_position_metrics_20260820_v2/summary.json`.
This is a hash-bound post-hoc computation on the unchanged 1,000 held-out test
predictions; it adds no trajectories and performs no training or model
selection. At 1 s, position MAE is `1.8708` current-only, `0.5500` CA-Kalman,
`0.6224` IMM, `0.6421` ego L16 MLP, `0.7576` GRU, `0.6951` Physics-GRU, and
`0.7307` contractive memory. CA-Kalman is best on this metric, so recurrence is
not supported by the future-position diagnostic either.

## Counterexample protocol

The completed slice contains 100,000 bounded states and 100,000 deliberately
out-of-bound abrupt states, plus dropout, crossing, association swap, sensor
delay, and multi-obstacle tracking checks. Cases outside assumptions are
recorded as assumption violations, not silently counted as certified.

## Budget

No formal 500k run. The controlled closed-loop ceiling is 100k transitions or 5000 matched trajectories.
