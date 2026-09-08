# Conservative Hazard Fusion Feasibility Gate

Date: 2026-09-04  
Protocol: `CONSERVATIVE_HAZARD_FUSION_GATE_V1`

## Purpose

Test whether a neural LiDAR/action hazard can selectively remove unsafe
optimism from a geometry rechargeability critic while preserving useful action
coverage.  This is a method-development Gate on already inspected data, not
fresh confirmation and not actor training.

## Data and split

- Combine both completed 75-scene fork datasets: 150 unique scenes, 600
  anchors, and 6,000 paired action branches.
- Use five outer folds by stable scene rank.  In each fold, 60 scenes fit model
  weights, 30 select epochs/temperature, 30 calibrate the risk threshold, and
  30 are tested.  Roles rotate; no scene crosses roles within a fold.
- All 15 battery queries, ten actions, and four anchors remain clustered by
  scene.  The resulting out-of-fold prediction covers every scene once.

## Models

1. `geometry_action`: monotone geometry parent.
2. `lidar_action_direct`: unconstrained direct fusion control.
3. `geometry_direct_intersection`: pointwise minimum of the first two.
4. `geometry_only_hazard`: learned one-sided hazard without LiDAR, testing
   whether a larger geometry head alone explains gains.
5. `lidar_action_hazard`: primary one-sided neural hazard using geometry,
   frozen R3 LiDAR embedding, and proposed action.

For the primary,

\[
q_h=\sigma(\operatorname{logit}(q_g)-\operatorname{softplus}(r_\theta(z))).
\]

The geometry parent is frozen.  Hazard training uses binary cross-entropy with
logits; validation Brier selects the epoch, including the near-zero initial
hazard.

## Scene-level risk calibration

At target \(\alpha=0.05\), define each scene loss as the fraction of its
infeasible action-budget queries declared safe.  Search a fixed 0.0005-spaced
threshold grid and select the lowest threshold satisfying

\[
\frac{n}{n+1}\widehat R_n(t)+\frac1{n+1}\le\alpha.
\]

The calibration split alone chooses this threshold.  Report held-out
scene-mean false-safe loss and total query coverage.  Complete abstention is
valid for risk but has zero utility and cannot pass the Gate.

## Promotion to fresh confirmation

All conditions are necessary:

- primary out-of-fold Brier is at least 2% below geometry;
- primary wins Brier in at least three of five test folds;
- zero probability-dominance and budget-monotonicity violations;
- dangerous false-safe at threshold 0.90 is no worse than geometry;
- held-out CRC scene risk is at most 0.05;
- CRC coverage is at least two absolute percentage points above geometry.

A pass authorizes collection of new task-seed scenes with the architecture,
risk target, and metrics frozen.  A fail stops this branch before new physics
simulation.  Neither outcome modifies R3.
