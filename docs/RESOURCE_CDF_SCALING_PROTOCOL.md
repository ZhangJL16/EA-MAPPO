# Resource-CDF Boundary-Balanced Scaling Protocol

## Question

The first 30-task pilot could not distinguish whether its negative result came
from insufficient optimization, too few independent environments, or attempting
to learn directly from a 2066-dimensional input.  It also contained no finite
trajectory requiring more than one calibrated battery, so its one-capacity CDF
query collapsed to deadline classification.

This experiment asks whether the stopped resource-to-recharge distribution is
learnable on held-out task seeds after explicitly covering battery thresholds
and reusing the frozen R3 navigation representation.

## Frozen elements

- R3 SAC checkpoint and structured LiDAR encoder.
- Accelerated HOCBF/runtime semantics that passed the energy equivalence Gate.
- Telemetry energy definition and 304.953884 calibrated capacity.
- Task generator, five distance buckets, and maximum 4000-policy-step rollout.
- Extended-real target: finite energy on timely task-then-recharge completion;
  failure atom on collision or deadline failure.

## Experimental factors

1. Independent data scale: 300 tasks, 60 per distance bucket.
2. Representation:
   - `raw`: task observation, compact charger state, absolute position, deadline;
   - `frozen_encoder`: frozen R3 encoder applied separately to task and charger
     observations, then concatenated with position and deadline;
   - `compact`: task kinematics/goal, charger kinematics/goal, position, deadline.
3. Optimization duration: 25, 100, and 400 epochs.
4. Battery query: 0.08, 0.12, 0.16, 0.20, 0.24, 0.28, 0.32, and 0.36 calibrated
   capacity.  These are evaluation queries on one learned distribution, not
   separately trained classifiers.

## Split and unit of analysis

Three outer folds hold out complete task seeds.  Every suffix snapshot and all
deadline/battery queries from one task remain in the same fold.  Suffixes expand
the training support but are never counted as independent seeds.  The primary
descriptive metrics pool out-of-fold task queries; no significance claim is made
from repeated queries within a task.

## Baselines and metrics

- Fold-local prevalence baseline.
- Geometry/deadline/budget logistic regression.
- AUROC and AUPRC for discrimination.
- Brier score, log loss, and ten-bin ECE for probabilistic quality.
- Per-horizon, per-budget, and per-distance-bucket breakdowns.
- Training-duration and representation ablations under identical folds.

## Fixed exploratory Gate

The primary configuration is `frozen_encoder@400` and promotes only if:

1. both capacity-feasibility classes contain at least 200 held-out queries;
2. AUROC is at least 0.75;
3. ECE is at most 0.15;
4. Brier score is at least 2% lower than the geometry baseline.

A pass authorizes a small on-policy integration pilot.  A failure diagnoses the
dominant factor from the representation and epoch matrix; it does not by itself
disprove the mathematical CDF formulation.

## Evidence limitations

This is an exploratory learnability experiment.  It does not establish a
calibrated lower confidence bound, raw-policy collision safety, improvement over
RC-PPO/RAPCPO, repeated-recharge lifecycle safety, or an oral-level contribution.
