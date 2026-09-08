# Resource-Aware Encoder Learnability Gate

## Status and scope

This is a preregistered exploratory representation-learning Gate.  It asks
whether a trainable copy of the R3 structured LiDAR encoder can predict the
finite-horizon resource CDF on unseen task seeds.  It does not update the R3
actor, establish closed-loop improvement, or certify lifecycle safety.

## Locked inputs

- Source dataset:
  `artifacts/r3_resource_cdf_boundary_balanced_scaling_300tasks_seed240001_20260903_v1`
- Navigation initialization: the immutable R3 500,000-transition checkpoint.
- Evaluation queries: battery fractions 0.08, 0.12, 0.16, 0.20, 0.24, 0.28,
  0.32, and 0.36 at horizons 1,000, 2,000, 3,000, and 4,000.
- The run manifest records SHA-256 hashes for the source result, checkpoint,
  protocol, and implementation.

## Task-disjoint evidence design

For outer fold `k`, tasks satisfying `task_index mod 3 = k` form the untouched
test set (100 tasks; 20 per distance bucket).  Within the remaining 200 tasks,
a deterministic hash rank is computed separately inside every distance bucket.
The first six tasks become the early-stopping validation set, the next six the
temperature-calibration set, and the remaining 28 the selection-training set.
After the best epoch is fixed, the model is reinitialized and trained for that
many epochs on selection-training plus validation tasks.  The calibration and
test tasks remain unseen by gradient updates.

All suffix snapshots from a task remain in the same partition.  Training loss
weights are inverse to the number of snapshots per task.  Evaluation uses only
the four initial-state horizon records for each test task.  The 32
budget-by-horizon queries from one task are repeated measurements and are not
treated as independent experimental seeds.

## Threshold-aligned resource distribution

Uniform finite-energy bin edges are augmented with every evaluation budget.
Consequently, each queried battery threshold is exactly an atom boundary rather
than cutting through a coarse bin.  A separate final class records the
collision/deadline failure atom.  Let `p_j(s)` be the softmax probability of
finite atom `j` and `e_j` its upper edge.  The learned CDF query is

`F_theta(b | s) = sum_{j: e_j <= b} p_j(s)`.

This construction enforces monotonicity in `b` by architecture and removes the
label/atom mismatch present when an evaluation budget lies inside a uniform
energy bin.

## Model and objectives

The model copies the R3 structured feature extractor, applies the same shared
encoder to the task-goal and charger-goal observations, concatenates both
embeddings with normalized absolute-position/deadline context, and predicts the
resource atoms.  The deployed R3 actor and checkpoint are never mutated.

Three variants are evaluated:

1. `frozen_ce`: frozen R3 encoder and categorical NLL only.
2. `trainable_ce`: trainable R3-initialized encoder and categorical NLL only.
3. `trainable_calibrated`: trainable R3-initialized encoder with
   `NLL / log(number_of_atoms) + mean_budget_Brier`.

For the primary variant only, a scalar softmax temperature is selected on the
separate calibration tasks by minimum task-averaged Brier score.  Both calibrated
and uncalibrated test predictions are retained.

## Optimization and stopping rule

- Maximum 500 epochs; evaluate every five epochs after epoch 25.
- Stop after 30 validation checks without a Brier improvement of at least
  `1e-5` (at most 150 additional epochs after the best checkpoint).
- Head learning rate `3e-4`; encoder learning rate `1e-4`; AdamW weight decay
  `1e-4`; batch size 128.
- Best checkpoint is the earliest epoch attaining the minimum task-averaged
  validation Brier.  Fold/variant results are atomic and resumable.

## Baselines, metrics, and Gate

The locked geometry/deadline/budget logistic baseline is refit on each outer
development partition and evaluated on the identical held-out tasks.  The
constant-prevalence baseline is also retained.

Primary configuration: calibrated `trainable_calibrated`, aggregated over the
three outer folds.  Metrics are AUROC, AUPRC, Brier score, log loss, ECE, and
accuracy, with horizon, budget, and distance-bucket breakdowns.

Promotion to a closed-loop policy pilot requires all of:

- at least 200 positive and 200 negative held-out queries;
- Brier score at least 2% below the geometry baseline;
- ECE at most 0.10;
- AUROC no more than 0.01 below the geometry baseline;
- Brier improvement over geometry on at least two of three outer folds.

Any pass remains exploratory.  A failure blocks actor modification and directs
work toward the target/observation semantics rather than longer training.

