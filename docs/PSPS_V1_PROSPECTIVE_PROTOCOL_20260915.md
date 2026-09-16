# PSPS-v1 prospective protocol

Date frozen: 2026-09-15. The Progress/Stall Predictive State (PSPS-v1)
experiment is an independent lineage. It uses prior PAI DEV results for design
only and never accesses or reuses the old PAI CONFIRM split.

## Claim and ordered gates

At each eligible step-256 or step-768 anchor, collect 64 C/R pairs with the
same post-anchor CRN disturbance seed. Gate O asks whether a prespecified compact
legal progress/stall state predicts the probability that C completes the current
task before the 12,000-policy-step branch guard. Only if Gate O passes is Gate D
evaluated: whether a threshold policy driven only by that probability improves
held-out paired utility over a matched latest-frame policy and the best constant
action. Strict failure of either DEV gate terminates the lineage without opening
PSPS CONFIRM.

## Fresh allocation and locked plant

- Excluded smoke: 2 worlds; DEV: 96 worlds; unopened CONFIRM: 128 worlds.
- All 226 worlds are generated before access and must be disjoint from every
  registered CMI and PAI world.
- Anchors, eligibility, history length 64, stride 4, C/R options, frozen SAC
  navigator, AR(1) post-filter action disturbance, and utility match PAI-v3.
- Contact remains nonterminal. Repair position and continue, zero all velocity
  at repair, use the fixed -1.2/-0.42 reward rule, and count one unified contact
  and one safety cost at most once per policy step. Branch utility remains
  `task_increment - 2*operational_failure - 0.25*I[collision_count>0]`.
- Automatic analysis, CONFIRM opening, and METHOD-TRAIN are false.

## Frozen deployment inputs

No simulator geometry, obstacle center/radius, world/branch/disturbance seed,
snapshot, future outcome, recurrent state, RFF feature, neural encoder, or
generic pooled history representation may enter either model.

The matched latest channel has exactly ten scalars:

1. battery fraction;
2. task distance divided by workspace diagonal;
3. charger distance divided by workspace diagonal;
4. task-clock fraction;
5. prefix-step fraction;
6. latest normalized velocity projected toward the task;
7. latest normalized horizontal speed;
8. latest normalized vertical velocity;
9. latest LiDAR hit fraction; and
10. latest minimum hit-masked normalized LiDAR range.

PSPS-v1 contains those ten scalars plus exactly ten legal trend scalars:

1. net task-distance progress per policy step;
2. fraction of sampled intervals with task-distance change at least -0.001 m;
3. longest consecutive run of those non-progress intervals divided by 63;
4. mean normalized velocity projected toward the task;
5. 10th percentile of normalized velocity projected toward the task;
6. mean normalized horizontal speed;
7. fraction of samples with normalized horizontal speed below 0.10;
8. joint stall fraction: non-progress interval whose ending speed is below 0.10;
9. battery depletion per policy step divided by synthetic capacity; and
10. mean normalized LiDAR hit fraction.

Task distance is decoded from the legal SAC observation. Charger distance and
battery are online legal measurements. The history spans the same frozen 64
samples at stride four. No feature may be added, removed, rescaled, or tuned
after outcome access.

## Frozen probability estimator

For each channel, fit one binomial logistic ridge with unpenalized intercept,
standardized inputs, coefficient L2 penalty, probability clipping at
`[1e-6,1-1e-6]`, six outer folds and four inner folds fixed by physical-world
hash. Select alpha from `[0.1,1,10,100]` by inner held-out binary log loss;
larger alpha wins an exact tie. Replicates split by parity into two 32-sample
halves. Fit on one half and evaluate on the other, swap halves, then average.
The constant probability is the outer-training C completion frequency.

## Gate O: observability

The event is `C.task_increment >= 1`, which necessarily occurs before a C
branch can commit return and therefore before a `branch_guard` terminal.

All conditions are conjunctive:

1. complete raw/hash/CRN/utility/collision audit; at least 90 worlds and at
   least 85 eligible anchors in each anchor-step stratum;
2. in each discovery half, C-completion prevalence lies in `[0.05,0.95]`;
3. two primary paired contrasts—latest Brier minus PSPS Brier and constant
   Brier minus PSPS Brier—have simultaneous 95% lower bounds strictly above
   zero under the frozen 20,000-draw whole-world max-t bootstrap;
4. PSPS reduces Brier loss by at least 5% relative to both baselines; and
5. all fitted values and probabilities are finite and PSPS has at least eight
   positive singular values on DEV.

Any failure stops immediately. Gate D is not interpreted or emitted as a
scientific result when Gate O fails.

## Gate D: decision value, conditional on O

Within each outer training set, create inner-world cross-fitted completion
probabilities and choose a threshold from `{0.05,0.10,...,0.95}` maximizing
discovery-half utility. Choose the larger, more conservative threshold on an
exact tie. At outer test anchors choose C only when predicted completion
probability is strictly above the threshold; otherwise choose R. Latest and
PSPS use identical fitting and threshold-selection rules. The constant policy
uses the higher-utility action on outer training worlds, with R on an exact tie.

All conditions are conjunctive:

1. PSPS-minus-latest and PSPS-minus-best-constant held-out paired utility have
   simultaneous 95% lower bounds strictly above zero under a separate frozen
   20,000-draw whole-world max-t bootstrap;
2. both point gains are at least 0.05 utility;
3. PSPS and latest actions disagree on at least 5% of anchors; and
4. in each half swap, PSPS selects each action on at least 5% of anchors and
   in at least five worlds.

## Confirmation and permanent denials

Only a DEV result passing every Gate O and Gate D condition may create a
hash-linked, externally anchored PSPS CONFIRM child. CONFIRM uses the unopened
128 PSPS worlds, frozen DEV standardizers, coefficients, alphas, thresholds,
support state, and no refit. It repeats the two O and two D contrasts with the
same strict simultaneous rules and requires at least 95% overall and 90% per
anchor-step support under a DEV-frozen PCA radius (up to 20 components, 1.15
times the 99th percentile leave-one-world-out distance).

The old PAI-v3 CONFIRM manifest is permanently denied to this lineage, whether
for tuning, pooling, diagnostics, or fallback. A validator must fail if an old
PAI CONFIRM access ever appears. PSPS-v1 success authorizes only a research
conclusion; METHOD-TRAIN always requires a separate user-authorized contract.

## Result schema

Gate O reports per-channel Brier score, log loss, completion prevalence, both
simultaneous contrasts and relative reductions. Gate D, only when interpreted,
reports both selected-utility contrasts, simultaneous intervals, action mix,
world coverage, disagreement, and frozen thresholds. Every artifact records
world/anchor counts, source/input hashes, bootstrap seeds, access counts, and
`method_train_authorized=false`.
