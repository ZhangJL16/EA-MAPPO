# Large-Scale Geometry + LiDAR/Action Residual Critic Gate

Date: 2026-09-04  
Protocol: `LARGE_SCALE_LIDAR_RESIDUAL_CRITIC_GATE_V1`

## Purpose

Test whether fixed-state LiDAR and proposed action explain safe-rechargeability
variation that remains after a strong geometric model.  The experiment is
deliberately bounded and may authorize an actor pilot, but cannot itself make a
closed-loop safety claim.

## Collection

- 75 scene seeds selected equally from the five locked distance buckets;
- four source-trajectory anchors per scene, balanced across task and return
  legs where available;
- ten registered raw eight-step action prefixes per anchor;
- unchanged deterministic R3 plus HOCBF continuation after the raw prefix;
- 300 anchors and 3,000 paired branches maximum;
- atomic branch files and exact resume semantics.

Collection must first pass `LIDAR_FORKED_ACTION_CAUSAL_DATA_GATE_V1`.  A failed
data Gate stops the chain before model fitting.

## Estimand

For anchor state (x), raw macro-action (a), energy budget (b), and the
declared frozen continuation,

\[
Q(x,a,b)=\Pr(S=1,\ E_{\mathrm{stop}}\le b\mid x,a),
\]

where (S) requires collision/boundary avoidance, task completion when still
on the task leg, and charger arrival before the remaining deadlines.

## Representation

The geometry context contains normalized position/velocity, task and charger
direction/distance, leg and remaining horizon, nearest cylinder clearance,
proposed action, deviation from R3, action alignment to the active goal, action
alignment to the nearest obstacle, and queried budget.

The perceptual context is the frozen 96-dimensional feature produced by the
confirmed R3 structured-LiDAR encoder from the exact anchor observation.  No
branch outcome, executed future action, realized energy, terminal reason, or
candidate-name label is an input.

## Compared models

1. `geometry_action`: monotone budget critic on geometry and action.
2. `lidar_no_action`: geometry state plus frozen LiDAR embedding, with current
   action fields removed.
3. `lidar_action_direct`: direct monotone critic on geometry, action, and
   frozen perception.
4. `geometry_lidar_action_residual`: frozen fitted geometry knot logits plus a
   zero-initialized perceptual/action residual.  Geometry is exactly recovered
   at epoch zero; residual knot increments remain positive.

## Splits and selection

Scene seed is the independent unit.  Its stable rank in the sorted set of
sampled scene seeds, modulo three, defines the outer test fold; a deterministic
subset of remaining ranked scenes is used for early stopping and temperature
calibration.  This rank mapping is necessary because collection scene IDs are
bucket-stratified and intentionally sparse.  All ten actions, four anchors, and
all budget queries from a scene stay together.

Every model is trained on the same fixed normalized budgets from 0.01 to 0.70
of calibrated capacity.  The residual candidate includes epoch zero in model
selection.  Temperature is selected only on inner-validation scenes.

## Metrics and promotion

Primary: scene-averaged held-out Brier score across all budget queries.

Secondary: AUROC, ECE, probability-0.90 dangerous false-safe rate, exact budget
monotonicity, matched-action sensitivity, mixed-anchor ranking, and each
distance bucket.

Promotion requires:

- residual pooled Brier at least 2% below `geometry_action`;
- residual beats geometry on at least two of three outer folds;
- no budget-monotonicity violation;
- nonzero matched-action sensitivity;
- residual dangerous false-safe rate no worse than geometry.

Any failure keeps the R3 actor frozen.  Passing authorizes a separate
closed-loop actor experiment; it is not itself evidence that the learned actor
is safe.
