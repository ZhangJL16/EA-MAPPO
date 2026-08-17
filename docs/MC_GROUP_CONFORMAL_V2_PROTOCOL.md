# MC Energy-to-Go Group-Conformal V2 Protocol

## Frozen components

The corrected experiment reuses, without retraining:

- the 500k frozen navigation SAC;
- the existing 7D MC-supervised Energy-to-Go point estimator;
- the calibrated 30-minute synthetic battery capacity.

The point-estimator checkpoint SHA256 is
`86b371ca92d6cdd337dc44a67e84f3242decd5661921be0f2fe1d5e4fce8f92b`.

## Goal-trajectory calibration

For calibration goal trajectory `i`, define

`R_i = max_t(G_i,t - E_hat_i,t)`.

For `n` scores and target coverage `1-alpha`, use rank

`k = min(n, ceil((n+1)(1-alpha)))`.

The global, goal-type, and initial-distance-bucket margins are fitted only on
calibration trajectories. At deployment the selected margin is

`Delta = max(Delta_global, Delta_goal_type, Delta_distance_bucket)`.

The final upper bound is `E_hat + max(0, Delta)`. Pointwise state coverage is
reported only as a secondary diagnostic. The primary metric is the fraction of
complete held-out goal trajectories for which every state is covered.

## Mission calibration

A mission sample contains a complete frozen-policy task segment followed by a
zero-velocity task-endpoint-to-charger segment. At every state on the task
segment,

`G_mission = G_task + G_return_after_task`

and

`E_hat_mission = E_hat_task + E_hat_return_after_task`.

Mission calibration uses the maximum mission underprediction over each
complete mission and fits global and task-distance margins. Formal switching
uses the calibrated mission upper bound directly. The sum of separately
calibrated component bounds remains a diagnostic and is not interpreted as a
95% mission bound.

## Data separation

The v2 calibration and final tests use distinct trajectory ids and never-used
seeds. The old calibration split may be combined with 2,000 new
intersection-stratified calibration trajectories. The final goal test contains
2,000 new trajectories. Mission calibration and mission test each use separate
stratified task-return sets and independent seeds.

## Corrected Phase2

The corrected Phase2 runner is single-environment and uses a 500,000-transition
budget. Its emergency guard must be strictly greater than the entire formal
budget; the formal configuration uses 600,000 policy steps. Consequently a
normal run cannot trigger a guard reset. Completed recharge cycles, guard
partials, other partial battery segments, and Gym episodes are counted
separately.
