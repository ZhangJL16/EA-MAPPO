# Paired C/R advantage identification protocol

Date designed: 2026-09-14. Status: pre-freeze protocol draft. It becomes a
formal preregistration only when its exact source/contract hashes and fresh
world manifests are externally anchored before access. It does not authorize
METHOD-TRAIN.

## Scientific order

The target is the paired advantage

`Delta_i,r = U_i(C,r) - U_i(R,r)`

under the same downstream disturbance seed. The study asks two ordered
questions:

1. **Gate H — decision heterogeneity:** does the conditional paired advantage
   vary stably enough across legal decision anchors that anchor-specific action
   choice improves over one constant action?
2. **Gate I — incremental legal-history information:** only if H passes, does
   full legal history predict paired advantage out of world better than the
   latest legal frame, and does that improvement affect selected utility?

Failure of H means no stable decision heterogeneity was established. Passing H
but failing I means heterogeneity exists but is not predictably encoded by the
tested legal history beyond the latest frame. Only passing both permits a new
method-design stage; neither gate starts training automatically.

## Isolation and collection

- Use fresh physical worlds disjoint from every CMI v1–v4 SMOKE/DEV/CONFIRM
  manifest: 96 DEV worlds and 128 unopened CONFIRM worlds.
- Use the locked steps 256/768 eligibility law and 64 paired-CRN resamples per
  action and anchor. No outcome-dependent replacement, extension, or repeated
  look is allowed.
- Preserve the locked collision-recovery protocol. Contact never terminates an
  episode; utility uses one unified collision count per policy step. New
  collision-free statistics are derived from `collision_count == 0`.
- Legal inputs remain LiDAR-derived geometry, known charger-relative state,
  online battery/task state, and legal response/action history. Simulator map,
  obstacle centers/radii, seeds, future outcomes, and branch state are forbidden
  predictors.
- The existing recyclable four-world coordinator may be reused only after exact
  equivalence and first-checkpoint health; one process owns ordered access and
  durable writes. Automatic analysis, confirmation, method design, and training
  are false.

## Replicate-split identification

Replicates are deterministically split by parity into two 32-pair halves. Every
statistic is computed twice, swapping discovery and evaluation halves, then
averaged. Physical worlds are the bootstrap and cross-fitting unit.

For Gate H, the discovery half selects C when its mean paired advantage is
strictly positive and R otherwise. The held-out half estimates selected utility.
The comparison action is the best constant action estimated only from outer-
training worlds. This identifies stable, decision-relevant heterogeneity without
using a learned state predictor.

For Gate I, the prediction target is paired advantage directly, not two
separate action values. Six outer world folds and four inner world folds are
fixed by world hash. Latest and full channels use the already specified compact
legal features. Estimators remain `LINEAR_RIDGE` and the existing 256-feature
`RFF_RIDGE` ensemble with ridge grid `[0.1,1,10,100]`; no deeper or larger model
is introduced. Models train on one replicate half and are scored against the
other, with the half roles swapped.

## Frozen DEV gates

All checks are conjunctive. Strict boundaries fail.

### Gate H

1. Raw hashes, source/world bindings, 64 paired seeds per action, finite
   outcomes, utility, and unified collision accounting pass; at least 90 worlds
   and 85 anchors per step contribute.
2. Paired-advantage MC-SE median is at most 0.10 and p95 at most 0.15.
3. The swap-averaged held-out selector gain over the cross-fitted best constant
   action has a 20,000-draw world-bootstrap 95% lower bound above zero and point
   estimate at least 0.05 utility.
4. In each discovery half, both R and C are selected at at least 5% of anchors
   and in at least five physical worlds.
5. Noise-corrected anchor-level advantage variance has a world-bootstrap 95%
   lower bound above zero. This supports heterogeneity but cannot replace the
   held-out decision-value requirement.

### Gate I, interpreted only if Gate H passes

1. The four primary contrasts use one 20,000-draw physical-world multiplier
   bootstrap and one simultaneous 95% max-t critical value: full minus latest
   advantage-MSE improvement, full minus cross-fitted constant-advantage MSE
   improvement, full minus latest selected utility, and full minus cross-fitted
   best-constant-action utility. Every simultaneous lower bound must exceed zero.
2. Full versus latest and full versus constant advantage-MSE improvements must
   each be at least 5% relative to the corresponding latest/constant MSE. Full
   selected-utility gain over the best constant action must additionally have a
   point estimate of at least 0.05 utility. This prevents a full model that only
   beats an inadequate latest model from passing.
3. Full/latest selected actions disagree at at least 5% of anchors; otherwise
   no incremental decision effect is established.
4. `LINEAR_RIDGE` and `RFF_RIDGE` each have positive full-minus-latest MSE
   improvement. Their improvement estimates must have the same sign.
5. Both legal representations and all predictions are finite. DEV-only PCA
   support has at least eight positive singular values; confirmation support
   radius is 1.15 times the DEV 99th-percentile leave-one-world-out distance.

### Machine-executable CONFIRM support rule

Support is evaluated separately for latest and full using the frozen DEV PCA
state and radius. Every CONFIRM anchor remains in all prediction, utility, and
effect statistics; no anchor is dropped and there is no per-anchor abstention.
For each channel, at least 95% of all CONFIRM anchors and at least 90% within
each frozen anchor-step stratum (256 and 768) must lie at or inside the radius.
The validator derives these four proportions directly from the complete raw
manifest. If any threshold fails, the whole Gate I result is
`INCONCLUSIVE_SUPPORT`; no predictive-information verdict is issued. The
thresholds, radii, and treatment of out-of-support anchors cannot be changed
after CONFIRM access.

## Confirmation and stopping

If either DEV gate fails, terminate without opening CONFIRM. Do not retune the
thresholds, feature map, estimator, folds, worlds, or resample count on that DEV
split. If both pass, freeze final DEV models, analysis hashes, support radii,
the complete DEV access head, and the untouched 128-world CONFIRM manifest in a
separately externally anchored child. Confirmation repeats H and I without
model refitting and requires every corresponding check to pass.

Passing confirmation permits only a new method-design proposal. METHOD-TRAIN
still requires a separate user-authorized contract and startup.

## Role of prior evidence

CMI-v4 may be used only for design and power diagnostics. Its post-hoc paired-
advantage result cannot satisfy either formal gate and cannot be pooled with the
fresh experiment.

## Operational lineage note (v3)

The v2 lineage stopped after 15 DEV accesses because its diagnostic RMS divided
the sum over safety-filter invocations by policy steps. Collision repair can
invoke that filter more than once in one policy step, so the diagnostic could
spuriously exceed the frozen per-invocation disturbance bound even though every
realized disturbance remained bounded. V2 is invalid for Gate H/I and is never
pooled. V3 counts actual filter invocations in the RMS denominator and uses 226
new physical worlds disjoint from all earlier CMI and paired-advantage lineages;
all scientific estimands, thresholds, folds, features, and stopping rules above
remain unchanged.
