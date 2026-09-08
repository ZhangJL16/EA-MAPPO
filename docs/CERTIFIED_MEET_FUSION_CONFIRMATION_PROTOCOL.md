# Certified Meet Fusion: Fresh-Scene Confirmation Protocol

Date: 2026-09-04  
Protocol: `CERTIFIED_MEET_FUSION_CONFIRMATION_V1`

## Question and non-claim

Can two learned, budget-monotone rechargeability critics provide a more useful
safe-declaration set when their evidence is combined by the lattice meet

\[
q_\wedge(x,a,b)=\min\{q_g(x,a,b),q_d(x,a,b)\},
\]

where `g` is the geometry/action critic and `d` is the frozen-R3
LiDAR/action critic?  This tests an energy-safety decision layer.  It does not
retrain R3, certify collision avoidance, or establish an end-to-end mission
improvement.

The preceding one-sided hazard Gate is immutable and failed its registered
coverage condition.  Its result may motivate this new protocol but is not
fresh evidence for it.

## Structural claims

For a common threshold \(t\), let
\(A_j(t)=\{(x,a,b):q_j(x,a,b)\ge t\}\).  Pointwise,

\[
A_\wedge(t)=A_g(t)\cap A_d(t),
\qquad q_\wedge\le q_g,
\qquad q_\wedge\le q_d.
\]

Consequently, at that same threshold the joint false-safe set of the meet is
a subset of either branch's false-safe set.  If both critics are nondecreasing
in battery budget, their pointwise minimum is also nondecreasing.  These are
deterministic implementation properties, not distribution-free claims about
thresholds calibrated separately for different scores.

## Development freeze

- Inputs are the two already inspected, disjoint 75-scene fork datasets (150
  scenes, 600 anchors, 6,000 action branches).
- Within every distance bucket, stable scene rank modulo five assigns three
  roles to training, one to validation, and one to CRC calibration: 90/30/30
  complete scenes.  No anchor, action, or battery query crosses roles.
- Training fits one geometry/action and one LiDAR/action
  `MonotoneBudgetCritic`.  Validation selects epochs and temperature.
- The 30-scene calibration role selects one threshold per score at
  \(\alpha=0.05\).  Scene loss is the fraction of infeasible action-budget
  queries declared safe.  The finite-sample corrected empirical risk is

\[
\widehat R^+(t)=\frac{n}{n+1}\widehat R(t)+\frac{1}{n+1}.
\]

- Model states, normalization, temperatures, budget grid, thresholds, source
  hashes, and split scene IDs are frozen before fresh simulation starts.

## Fresh scene contract

- Generate 75 new stratified tasks with task seed `420001` and static-world
  seed `430001`.  These seeds differ from development (`310001`, `320001`).
- Use the unchanged R3 checkpoint, environment reconstruction, nominal task
  then return trajectory, four registered anchors per scene, ten registered
  raw eight-step macro-actions per anchor, and frozen-R3 plus HOCBF
  continuation.  Expected size is 300 anchors and 3,000 branches.
- The upstream paired-data Gate must pass exactly as currently implemented.
- The fresh dataset is evaluated once with frozen models and thresholds.  It
  cannot select weights, temperatures, thresholds, architectures, or metrics.

## Registered confirmation decision

All conditions are necessary:

1. all 75 scenes, 300 anchors, and 3,000 branches are complete and pass the
   paired-data Gate;
2. meet probability dominance and budget monotonicity have zero numerical
   violations;
3. meet test scene-mean false-safe risk at its frozen CRC threshold is at most
   0.05;
4. meet test risk is no larger than geometry test risk at geometry's own
   frozen CRC threshold;
5. meet safe-declaration coverage is at least 0.02 higher than geometry;
6. a 10,000-draw paired-scene bootstrap 95% interval for meet-minus-geometry
   coverage has lower endpoint above zero;
7. at fixed threshold 0.90, meet joint dangerous false-safe is no worse than
   geometry.

Brier, AUROC, ECE, distance buckets, conditional false-safe, and
unsafe-among-accepted are mandatory diagnostics but not promotion gates.  This
is intentional: the scientific object is selective decision risk at useful
coverage, while Brier averages errors from queries that deployment may reject.

A pass authorizes a bounded closed-loop mission-retargeting pilot with R3 still
frozen.  A fail stops this branch and preserves R3 unchanged.

