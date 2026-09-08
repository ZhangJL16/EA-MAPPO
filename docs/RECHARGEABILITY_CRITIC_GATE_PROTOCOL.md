# Action-Conditioned Rechargeability Critic Gate

Date: 2026-09-03  
Protocol: `ACTION_CONDITIONED_RECHARGEABILITY_GATE_V1`

## Question

Can an action-conditioned, budget-monotone critic predict safe and timely
recharge under the executed physical-energy budget better than a locked
geometry-only model on unseen scene seeds?

This is a learnability Gate.  It does not train or modify R3, certify deployment
safety, or establish collision-tail performance.

## Estimand

For pre-action state (x_t), proposed raw action (a_t), normalized available
budget (b), and remaining horizon (h),

\[
Q_h(x_t,b,a_t)=\Pr\!\left(
 \tau_C<\tau_U,\ \tau_C\le h,\
 \sum_{k=t}^{\tau_C-1}e_k\le b
 \mid x_t,a_t
\right).
\]

The continuation distribution is the fixed grouped-intervention behavior
policy represented in the collected data.  The current Gate does not identify
arbitrary-policy counterfactual values.

## Data and independence

- Source: completed 150-scene, five-intervention grouped R3 run.
- Independent unit: scene seed.
- Correlated augmentation: intervention, transition, suffix, and budget query.
- Outer split: `scene_index mod 3`; all augmentation stays with its scene.
- Inner validation: a deterministic subset of the two non-test scene folds.
- Pre-action reconstruction: regenerate the deterministic task from the locked
  task seed, invert the saved active-goal direction and log-distance, and verify
  reconstructed task/return distances against rollout metadata.

## Leakage boundary

Inputs may contain current physical state, task and charger geometry, remaining
horizon, frozen/proposed action, intervention descriptor, and previous-step
action/energy/filter history.  They may not contain current-step executed
action, current realized energy, current HOCBF result/slack, collision outcome,
future path length, suffix energy, or terminal result.

## Models

1. `geometry`: monotone-budget critic using only velocity, task/charger
   distances, leg, horizon, past scalar burden, and intervention descriptor.
2. `no_action`: full context with current frozen/proposed action masked.
3. `no_bellman`: full action-conditioned context trained only on stopped
   Monte-Carlo labels.
4. `full`: action-conditioned context with Monte-Carlo supervision and
   one-step finite-horizon Bellman consistency.

Every neural CDF uses non-negative logit increments across fixed budget knots,
so monotonicity in budget is architectural rather than a penalty.

## Targets

For a successful mission-to-charger rollout, the Monte-Carlo label is one iff
the realized suffix energy does not exceed the queried budget.  Every suffix of
a collision, deadline, or task failure has label zero.  The Bellman target is

\[
y_t(b)=
\begin{cases}
0,&b<e_t,\\
1,&\text{safe charger terminal and }b\ge e_t,\\
0,&\text{failure terminal},\\
Q_{h-1}(x_{t+1},b-e_t,a_{t+1}),&\text{otherwise}.
\end{cases}
\]

## Primary evidence and promotion

- Primary: scene-averaged held-out Brier score, pooled across three outer folds.
- Secondary: ECE, dangerous false-safe rate at probability 0.90, held-out
  Bellman residual, and matched-scene action sensitivity.
- Promotion requires all of:
  - full pooled Brier at least 2% below geometry;
  - full Brier below geometry on at least two of three folds;
  - full held-out Bellman residual below `no_bellman`;
  - nonzero action sensitivity on matched initial scene states;
  - no split overlap or budget-monotonicity violation.

Failure stops actor work.  Passing authorizes a new LiDAR-complete forked-action
collector and actor pilot; it does not itself authorize a safety claim.

## Known limitation

Only five of 750 source rollouts contain obstacle collision and only the initial
full LiDAR observation was retained.  Therefore this Gate mainly tests
energy/deadline rechargeability.  Collision-tail learning requires a later
near-obstacle, forked-action dataset with per-state LiDAR or frozen perceptual
features, and is explicitly outside any positive claim from this Gate.
