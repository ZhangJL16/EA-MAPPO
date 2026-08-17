# MC-Supervised Energy-to-Go Protocol

## Status

This protocol supersedes the divergent Quantile TD estimator as the primary
energy estimator. The Quantile TD implementation and artifacts remain retained
as a failed baseline.

## Target

For every successful frozen-policy goal trajectory with realized per-transition
energy costs `e_0, ..., e_T`, the supervised label is

`G_t = sum_{k=t}^T e_k`.

There is no bootstrap target, target network, discount factor, or next-state
prediction. The 7D input is normalized velocity, goal unit direction, and
linear goal distance divided by the fixed map diagonal. The target used by the
network is `G_t / calibrated_battery_capacity`; reported predictions are
converted back to synthetic simulation energy units.

## Data Separation

The train, validation, upper-bound calibration, and held-out test sets use
disjoint complete trajectories and independent seeds. Each split contains
random task goals, random-state charger goals, and zero-velocity task-endpoint
to charger goals. Validation, calibration, and held-out test sets are stratified
over the five physical-distance buckets used by the navigation protocol.

## One-Sided Upper Bound

For each calibration trajectory `j`, define the trajectory score

`R_j = max_t (G_{j,t} - E_hat(s_{j,t}, g_j))`.

For `n` calibration trajectories and target coverage `1-alpha`, use rank

`k = min(n, ceil((n+1)(1-alpha)))`.

The deployed margin is `Delta = max(0, R_(k))`, where `R_(k)` is the kth order
statistic. The switching bound is `E_upper95 = E_hat + Delta`. The independent
held-out test set is used only to report coverage and is never used to choose
`Delta`.

## Switching

The one-way TASK-to-CHARGER decision uses three separately evaluated bounds:

- current state to current task;
- zero-velocity task endpoint to charger;
- current state to charger.

The mission bound is the sum of the first two bounds. The conformal margin and
the battery reserve remain distinct: the former addresses estimator residuals,
while the latter is an operational reserve equal to 10% of calibrated battery
capacity.
