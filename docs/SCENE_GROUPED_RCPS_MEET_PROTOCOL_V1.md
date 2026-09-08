# Scene-grouped RCPS meet protocol V1

Status: frozen before collecting calibration or confirmation data.

## Question

Can the already frozen certified-meet rechargeability score accept more
state-action-budget queries than the geometry-only score while providing a
finite-sample certificate that its scene-averaged dangerous false-safe risk is
at most 5%?

This Gate changes neither the R3 navigation actor nor either frozen critic. It
only replaces the earlier empirical-risk threshold with risk-controlling
calibration. The construction follows *Risk-Controlling Prediction Sets*
(https://arxiv.org/abs/2101.02703) and its public HBB implementation
(https://github.com/aangelopoulos/rcps), within the broader learn-then-test
calibration framework (https://arxiv.org/abs/2110.01052).

## Independent unit and loss

The independent unit is a newly generated scene, not a branch or a budget
query. For score `p`, acceptance threshold `lambda`, and scene `i`, define

```text
L_i(lambda) = mean[ 1{p >= lambda} | query is infeasible in scene i ].
```

If a scene contains no infeasible query, its loss is defined as zero. Thus
`L_i` lies in `[0,1]` and decreases monotonically with `lambda`. The target is

```text
R(lambda) = E_scene[L_i(lambda)] <= alpha = 0.05.
```

This estimand deliberately gives each scene equal weight and prevents the 15
budget labels and 40 paired branches per scene from being mistaken for 600
independent samples.

## Frozen score family

- Frozen model: `artifacts/certified_meet_fusion_freeze_150scenes_20260904_v2/frozen_meet.pt`.
- Baseline score: geometry-action critic probability.
- Primary score: pointwise minimum of geometry-action and lidar-action-direct
  probabilities.
- The pointwise minimum must never exceed either constituent score and must
  have zero budget-monotonicity violations.

## Calibration rule

For each fixed threshold on

```text
{0, 0.0005, ..., 1.0, +infinity},
```

compute the Hoeffding--Bentkus--empirical-Bennett 95% upper confidence bound
`U_HBB(lambda)` from the scene losses. Scan from conservative to permissive and
select the smallest threshold whose entire more-conservative suffix satisfies

```text
U_HBB(lambda') <= 0.05 for every lambda' >= lambda.
```

This is the nested RCPS selection rule. There is no post-hoc threshold choice,
bootstrap substitution, branch-level resampling, or model refit.

## Data split and immutable seeds

The already inspected 75-scene confirmation set is reclassified as development
evidence only and is not reused for calibration or confirmation.

- Calibration: 450 new stratified scenes; task seed `510001`, world seed
  `520001`; 4 anchors per scene, 10 registered actions per anchor, 15 budgets.
- Confirmation: 150 further new stratified scenes; task seed `610001`, world
  seed `620001`; the same registered action/budget design.
- Both sources store exact simulator pre-action positions and velocities.
- The calibration model, thresholds, protocol, source manifests, and result
  files are SHA-256 bound before confirmation.

## Calibration promotion Gate

All checks are required:

1. both paired-action data Gates pass;
2. exact simulator snapshots are present;
3. meet dominance violations are zero;
4. meet and geometry HBB certificates are non-vacuous;
5. meet HBB upper risk bound is at most 0.05;
6. meet calibrated coverage exceeds geometry calibrated coverage by at least
   0.02 absolute.

If calibration fails, confirmation is not run.

## Confirmation Gate

Using only the frozen calibration thresholds on the 150 untouched scenes:

1. meet scene-averaged false-safe risk is at most 0.05;
2. meet coverage is at least geometry coverage plus 0.02;
3. the paired 10,000-resample, scene-level 95% interval for the coverage
   difference has lower endpoint above zero;
4. pointwise meet dominance and budget monotonicity have zero violations;
5. fixed-0.90 dangerous false-safe rate is no worse than geometry.

The HBB statement is the formal calibration certificate. Confirmation is an
untouched external validation of transfer and utility, not a second opportunity
to tune the threshold.

## Interpretation

Promotion supports a calibrated learned rechargeability guard around the fixed
R3 actor. It does not establish real-world UAV safety, robustness to arbitrary
distribution shift, or superiority of the navigation actor itself.
