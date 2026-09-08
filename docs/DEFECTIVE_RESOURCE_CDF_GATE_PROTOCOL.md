# Defective Continuous Resource-CDF Gate

## Question

Can an identifiable two-head neural model predict safe-return feasibility on
unseen R3 task seeds better than the locked geometry/deadline/budget logistic
baseline?

The Gate is passed only by held-out predictive evidence.  Training completion,
lower training loss, or attractive calibration alone are not success.

## Locked data and partitions

- Source:
  \`artifacts/r3_resource_cdf_boundary_balanced_scaling_300tasks_seed240001_20260903_v1\`.
- All 300 tasks, four horizons, eight battery thresholds, and the immutable R3
  checkpoint are unchanged.
- Outer test fold: \`task_index mod 3\`, giving 100 test tasks per fold.
- Within the 200 development tasks, deterministic bucket-stratified hashing
  assigns six tasks per distance bucket to early stopping, six per bucket to
  calibration, and 28 per bucket to gradient training.
- After epoch selection, refit uses gradient-training plus validation tasks.
  Calibration and test tasks never receive gradient updates.
- Suffix rows are training augmentation.  Metrics use initial-state rows only.

## Identifiable predictive family

For state/deadline context \(z\) and positive budget \(b\),

\[
\widehat F_\theta(b\mid z)
=(1-q_\theta(z))
\Phi\!\left(\frac{\log b-\mu_\theta(z)}{\sigma}\right).
\]

The network predicts only failure logit and conditional log-energy location.
The positive scale \(\sigma\) is a single learned smoothing parameter, not a
state-dependent physical variance.  A separate calibration split selects three
global corrections: failure-logit temperature, log-energy shift, and scale
multiplier.

## Variants

1. \`compact_hurdle\`: standardized 18-dimensional
   task/charger/position/deadline observation through an MLP.
2. \`encoder_hurdle\`: trainable R3-initialized dual-goal LiDAR encoder plus
   normalized position/deadline context.
3. \`fusion_no_cdf\`: encoder features plus the direct compact observation
   skip, trained without the budget-CDF Brier term.
4. \`fusion_hurdle\`: the same fusion architecture with the complete loss; this
   is the fixed primary model.

The compact skip is an observation pathway, not a hand-coded return rule.  The
geometry baseline remains external and is never supplied as a privileged
decision score.

## Objective and optimization

The full loss is

\[
\mathcal L
=\mathcal L_{\mathrm{failure\ BCE}}
+\mathcal L_{\mathrm{finite\ log-energy\ Huber}}
+2\mathcal L_{\mathrm{budget\ Brier}}.
\]

All three terms are task weighted.  \`fusion_no_cdf\` sets the final coefficient
to zero.  The finite-energy Huber transition is 0.25 log units.

- AdamW, head learning rate \(3\times10^{-4}\), encoder learning rate
  \(10^{-4}\), weight decay \(10^{-4}\), batch size 128.
- Validate every five epochs from epoch five.
- Maximum 400 epochs; stop after 30 validation checks without an improvement
  of at least \(10^{-5}\).
- Reinitialize and refit for the selected epoch count before separate
  calibration.
- Fold/variant outputs and model checkpoints are atomic; resume skips complete
  units.

## Evidence

Primary metrics: Brier, AUROC, ECE, AUPRC, log loss, and accuracy for the
finite-budget event.  Mechanism diagnostics additionally report failure
probability metrics and conditional finite log-energy MAE/RMSE.

Promotion requires:

- at least 200 positive and 200 negative held-out budget queries;
- primary Brier at least 2% below geometry;
- primary ECE at most 0.10;
- primary AUROC no more than 0.01 below geometry;
- primary Brier below geometry in at least two of three outer folds.

Even a pass promotes only to a closed-loop policy pilot.  It is not formal
safety evidence or an oral-level novelty claim.

