# Certified Meet Fusion: Fresh-Scene Confirmation Protocol V2

Date: 2026-09-04  
Protocol: `CERTIFIED_MEET_FUSION_CONFIRMATION_V2`

## Preflight correction

V1 was retired before any fresh scene was generated.  An evaluator integration
smoke on previously inspected data showed that requiring both
\(R_\wedge\le\alpha\) and \(R_\wedge\le R_g\) changes the declared constraint
from \(\alpha\) to the random, generally smaller baseline risk.  That is not
the selective-prediction objective.  V2 therefore maximizes coverage subject
to the fixed risk target \(\alpha=0.05\); it reports, but does not gate on, the
difference from a baseline that may leave part of the risk budget unused.  No
fresh task or world seed was inspected before this correction.

## Question and non-claim

Can two learned, budget-monotone rechargeability critics provide a more useful
safe-declaration set when combined by the lattice meet

\[
q_\wedge(x,a,b)=\min\{q_g(x,a,b),q_d(x,a,b)\},
\]

where `g` is the geometry/action critic and `d` is the frozen-R3
LiDAR/action critic?  This tests an energy-safety decision layer.  It does not
retrain R3, certify collision avoidance, or establish end-to-end mission gain.

## Structural claims

For a common threshold \(t\), define
\(A_j(t)=\{(x,a,b):q_j(x,a,b)\ge t\}\).  Then

\[
A_\wedge(t)=A_g(t)\cap A_d(t),\qquad
q_\wedge\le q_g,\qquad q_\wedge\le q_d.
\]

Thus the meet's joint false-safe set at the same threshold is a subset of
either branch's false-safe set.  If both critics are nondecreasing in budget,
their minimum is nondecreasing.  These deterministic facts do not imply that
two separately calibrated thresholds have ordered risks.

## Development freeze

- Use the two already inspected, disjoint 75-scene fork datasets: 150 scenes,
  600 anchors, and 6,000 action branches.
- Stable scene rank modulo five assigns 90 complete scenes to training, 30 to
  validation, and 30 to CRC calibration, stratified across distance buckets.
- Validation alone selects epochs and temperatures for one geometry/action
  and one LiDAR/action `MonotoneBudgetCritic`.
- Calibration selects one threshold per score at \(\alpha=0.05\).  Scene loss
  is the fraction of infeasible action-budget queries declared safe, with

\[
\widehat R^+(t)=\frac{n}{n+1}\widehat R(t)+\frac1{n+1}.
\]

- States, normalization, temperatures, budget grid, thresholds, source hashes,
  split IDs, protocol hash, and confirmation criteria are frozen before fresh
  simulation.

## Fresh scene contract

- Generate 75 stratified tasks with task seed `420001` and static-world seed
  `430001`, distinct from development seeds `310001` and `320001`.
- Keep the R3 checkpoint, environment, nominal task/return trajectory, four
  anchors per scene, ten raw eight-step macro-actions per anchor, and frozen-R3
  plus HOCBF continuation unchanged: 300 anchors and 3,000 branches.
- The paired-data Gate must pass.  The fresh set is then evaluated exactly once
  and cannot select any model or decision component.

## Registered confirmation decision

All conditions are necessary:

1. all 75 scenes, 300 anchors, and 3,000 branches are complete and the
   paired-data Gate passes;
2. meet probability dominance and budget monotonicity have zero violations;
3. meet test scene-mean false-safe risk at its frozen CRC threshold is at most
   0.05;
4. meet safe-declaration coverage is at least 0.02 higher than geometry at
   geometry's own frozen CRC threshold;
5. a 10,000-draw paired-scene bootstrap 95% interval for meet-minus-geometry
   coverage has lower endpoint above zero;
6. at the common fixed threshold 0.90, meet joint dangerous false-safe is no
   worse than geometry.

The geometry risk and paired risk difference remain mandatory diagnostics.
Brier, AUROC, ECE, distance buckets, conditional false-safe, and
unsafe-among-accepted are also mandatory diagnostics, not promotion gates.

A pass authorizes a bounded closed-loop mission-retargeting pilot with R3 still
frozen.  A fail stops this branch and preserves R3 unchanged.

