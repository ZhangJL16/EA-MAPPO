# Independent LiDAR/Action Rechargeability Confirmation

Date: 2026-09-04  
Protocol: `INDEPENDENT_LIDAR_ACTION_CONFIRMATION_V1`

## Question and frozen hypothesis

The development Gate found that a direct monotone critic using geometry,
proposed action, and the frozen R3 structured-LiDAR embedding improved both
Brier score and the registered dangerous false-safe statistic relative to the
geometry/action baseline.  Because this model was a control in that Gate, the
finding is exploratory.  This protocol freezes it as the primary candidate and
tests it once on untouched scene seeds.

The estimand remains

\[
Q(x,a,b)=\Pr(S=1, E_{\rm stop}\le b\mid x,a),
\]

under the same eight-step raw macro-action followed by frozen R3 plus HOCBF.

## Independent confirmation data

- Development scenes: offsets 0--14 within each of five source distance
  buckets (75 scenes), already collected.
- Confirmation scenes: offsets 15--29 within every bucket (75 disjoint scenes).
- Four anchors and ten registered action branches per scene, giving 3,000 new
  confirmation branches.
- No confirmation outcome may influence model weights, normalization,
  temperature, architecture, or thresholds.
- Anchor identity is bound to the canonical NPZ row by ID and observation hash;
  legacy integer metadata indices are not trusted.

## Frozen predictors

Use the three already fitted development-fold members from
`lidar_residual_critic_gate_75scenes_20260904_v2`.  For each member, reuse its
stored normalization, weights, and temperature without refitting.  Average the
three probabilities.  Convex averaging preserves monotonicity in budget.

Primary: `lidar_action_direct`.  Baseline: `geometry_action`.  Mechanism
controls: `lidar_no_action` and `geometry_lidar_action_residual`.

## Metrics

Primary accuracy is confirmation-scene-averaged Brier score over the 15 fixed
budget queries.  At the frozen 0.90 safe-declaration threshold report:

- joint dangerous false-safe probability
  \(P(\hat Q\ge .9,Y=0)\);
- conditional false-safe rate \(P(\hat Q\ge .9\mid Y=0)\);
- unsafe-among-accepted rate \(P(Y=0\mid\hat Q\ge .9)\);
- declaration coverage \(P(\hat Q\ge .9)\).

Also report AUROC, ECE, exact budget monotonicity, matched-action sensitivity,
each distance bucket, and every frozen ensemble member.  Scene is the only
resampling unit.  A deterministic 10,000-sample paired scene bootstrap gives a
95% interval for direct-minus-geometry Brier and dangerous false-safe rate.

## Promotion rule

Actor-pilot authorization requires all of:

1. direct ensemble Brier is at least 2% below geometry;
2. the 95% paired-scene bootstrap upper bound for the Brier difference is below
   zero;
3. direct joint dangerous false-safe is no worse than geometry, and the 95%
   upper bound on their difference is at most +0.005 absolute;
4. direct wins Brier for at least two of the three frozen members;
5. zero budget-monotonicity violations and nonzero matched-action sensitivity;
6. all 75 confirmation scenes and all 3,000 branches pass the upstream data
   Gate.

Failure preserves R3 unchanged.  Passing authorizes only a separately bounded
closed-loop actor pilot; it is not a deployment-safety guarantee.
