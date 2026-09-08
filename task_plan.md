# Task Plan: Return-to-Charge Research Gates

## Active 2026-09-08: new-navigation return-energy learning

User: continue subsequent research. Navigation adaptation131072 and paired
fixed500 evaluation both complete. Working budget explicitly announced:
500000 additional energy-data environment transitions; frozen new navigator.

- [x] Audit completed paired navigation results (descriptive, one training seed).
- [x] Implement isolated resumable new-policy return-energy data collection;
  preserve current2056 sensor interface and locked collision recovery.
- [x] Define scene-disjoint mean/quantile and defective-law learning comparison,
  with correct future-contact labels and right-censored budget tails.
- [x] Focused tests and one tiny execution/resume smoke, launch500k collection.
- [x] Verify first committed startup health, then hand off. Model fitting is
  subsequent work on completed labels, not already running or automatically queued.

Protocol: docs/NEW_NAVIGATION_RETURN_ENERGY_500K_PROTOCOL.md. Four tests passed.
Formal artifact new_navigation_return_energy_500k_20260908_v1, detachedPID96663.
First8192 checkpoint verified:16 completed episodes,397 retained observations,
1959 pending transitions with full worker state; completed+pending=8192.
Source hashes match, checkpoint size79840691 verified, actor observations finite
2056, unified labels consistent, no ERROR, navigation_updates0. Startup complete;
STOP MONITORING and await user. Collection will stop at500000, no auto fitting.
First launcher incorrectly resolved the venv Python symlink to system Python,
which failed at numpy import before creating any run output. Relaunched with
the absolute .venv/bin/python path (without following its symlink); no experiment
data was lost or changed. Initial failure remains in the external console log.

Do not claim correction loss causally isolated (no matched lambda0 run), full
mission energy sustainability, or a safety theorem. No shared runtime edits.
Error during read-only discovery: guessed stochastic_return.py did not exist;
rg found stochastic_anchor_return.py. No file was changed by that failed read.

## Active 2026-09-08: queue fixed500, then new-navigation energy research

User explicitly requests post-training500-task evaluation and subsequent energy
experiments on the new navigation model (“500k那个”). This authorizes the
requested evaluation successor, not arbitrary automatic model promotion.

- [x] Add separate resumable raw/HOCBF fixed500 evaluator (same500 tasks/5bins).
- [x] Add/verify detached successor that waits only for completed131072 checkpoint;
  leave live trainer and its hashed source unchanged; stop on pause/error.
- [x] Reconcile historical500k energy pipeline with current observation/collision
  contract and recent defective-return-law design (porting still required).
- [ ] Clarify what500k counts; asynchronous question pending, working interpretation
  is500000 additional frozen-navigation energy-stage transitions.
- [x] Queue requested evaluation, verify waiter health, hand off (do not await completion).

Queue PID19195 healthy WAITING_FOR_TRAINING, manifest/code hashes verified;
training PID9399 alive at53248/131072 adaptation transitions during handoff.
Formal evaluation not yet started, no energy experiment launched. Six focused
tests passed plus one nonformal5-task×2-step×2-arm actual checkpoint smoke.
Protocol: docs/HOCBF_FIXED500_AND_ENERGY_HANDOFF_20260908.md.

Energy-budget clarification sent asynchronously; do not silently launch an old
2055-D/JSEB energy pipeline on the current2056-D model. No energy run launched yet.

## Active 2026-09-07: HOCBF correction-supervised SAC — LAUNCH AUTHORIZED

User superseded the pre-launch stop with “请继续实验”. Finish focused checks,
then launch and hand off after the first learned checkpoint, without monitoring.
Skills: daily-coding (minimal changes/tests), planning-with-files (durable handoff).

- [x] Audit reusable SAC/checkpoint/HOCBF interfaces and freeze supervision semantics.
- [x] Implement isolated old-architecture SAC + fresh valid HOCBF correction loss.
- [x] Implement explicit prepare/start runner, warm start and full resumability.
- [x] Run focused unit/integration checks (tiny test fixtures only), document commands.
- [x] Launch one correction-supervised adaptation, verify first learned checkpoint, hand off.

Constraints: retain locked nonterminal collision repair/rewards/unified counts;
LiDAR-only obstacle observations; no new critic/encoder/energy module; keep
nominal-action SAC replay semantics. Old artifacts and source remain intact.
Fresh teacher queries must match observation/action/time; invalid fallback
outputs are not valid targets. Historical Jacobian bridge is not blindly reused.
Handoff2026-09-08: detached PID9399, artifacts/hocbf_correction_sac_20260908_v1.
Committed8192 adaptation steps; actor/critic each3192 updates, finite metrics,
complete model/replay/worker checkpoint, source/config hashes match, no ERROR.
Fresh supervision399 updates/395nonzero,3192queries/3176valid/1496corrected;
16solver-invalid targets excluded. Latest block110.97s, teacher queries2.94s
cumulative.131072 planned additional steps; no automatic evaluation or promotion.
Startup complete: STOP MONITORING, await user.14 focused tests/compile passed.
Errors: first focused suite8passed/1failed: test incorrectly assumed a near-wall
native HOCBF query was feasible. Both native/teacher reject it with existing
stagnated_infeasible_or_ill_conditioned solver result; no unsafe teacher accepted.
Add explicit fallback rejection/alignment coverage; do not weaken validity rules.
Documentation patch initially used a nonmatching wrapped line; no files changed
by that failed patch. Reapplied using the actual text. No runtime source changed.

## Active 2026-09-07: final v3 → fixed500 evaluation

- [x] Confirm v3 completed524288 and no prior v3 formal evaluation exists.
- [x] Isolate a versioned v3 evaluator with the identical500-task batch/resume
  logic, task seeds, model inference, observation interface and collision rules.
- [x] Focused version/unchanged-loop check and one tiny final-model smoke only.
- [x] Start formal evaluation; confirm first committed batch; hand off.

Handoff: Python PID20663; deployable_derived_hit_v3_fixed500_20260907_v1.
Startup verification 200/500 committed, immutable task-prefix/safety semantics
validated, formal v3 contract confirmed, process live, no ERROR. Stop monitoring;
no automatic analysis or follow-up training. See docs/DERIVED_HIT_V3_FIXED500_PROTOCOL.md.

User explicitly approved the fixed500 next step. No training, HOCBF change,
performance gates, automatic follow-up, or claim of energy sustainability.

## 2026-09-07 second interrupted v3 continuation

User again requested continuation. Revalidated no live trainer (exact process
arguments), no PAUSE/ERROR, checkpoint393216 files and source hashes intact;
stale log401408 was not a checkpoint. Preserved that interrupted log/status
under distinct401408 filenames, then resumed the unchanged run with --resume.
Current detached PID3841: full state restored, fresh TRAINING status393216,
GPU active73%, no console error at startup. This is restored-running evidence,
not a claim of a newly completed block/checkpoint. Stop monitoring and hand off.

## 2026-09-07 interrupted v3 resumed (user: 继续现有任务)

- Verified no live v3 trainer; stale status417792 did not establish liveness.
- Preserved interrupted training records/status separately in the run root.
- Source/config and model/replay/state file sizes matched checkpoint393216.
- Resumed the same run using --start --resume, detached Python PID3998.
- Startup health: completed397312, actor/critic each392312 updates, finite
  recorded metrics, no ERROR, live process. First resumed block141.94seconds.
- Latest full checkpoint remains393216; the unchanged32768-step schedule next
  saves425984 (or at requested pause). Do not claim a new model checkpoint at397312.
- Hand off now; do not monitor completion or auto-start evaluation. Original
  ReturnManager/probability/Oracle-headroom/Pareto goal remains active and is
  not considered achieved by this navigation ablation.

## Active request: completed v2 fixed500 → isolated encoding ablation (2026-09-07)

- [x] Validate paired fixed500 evidence and produce analysis bundle with two
  figures, raw effect sizes, conditional task-level uncertainty, and limitations.
- [x] Add a versioned extractor reconstructing a threshold hit channel from
  existing normalized ranges; preserve1039-D deployable observation/context.
- [x] Focused tests plus one tiny runner smoke, reusing unchanged checkpoint
  mechanism. No performance pre-gates, no physics/reward modifications.
- [x] Launch same-budget scratch SAC ablation, verify learned checkpoint, hand off.

Handoff: Python PID101139, artifacts/deployable_derived_hit_v3_sac_20260907_v1.
8192-step checkpoint complete,3192 actor and3192 critic updates, finite metrics,
source contract confirms1039 external dimensions and2 internal channels,
model/replay/state files complete, no ERROR, process alive. Stop monitoring.
User HOCBF question resolved from source: both prior fixed500 SAC and v2/v3
have filter disabled; separate raw/HOCBF return experiments are not this control.

Observed v2:461 arrivals/372 safe vs baseline496/423;39 vs4timeouts. No claim
that missing hit-channel caused this: joint input changes and one seed confound
attribution. Next ablation isolates encoder input representation vs v2, not
another energy-management claim. Existing finalized artifacts stay immutable.

## Active request 2026-09-07: analyze v2 and advance evidence

- [x] Verify completed524288-step training, inspect descriptive training logs
  and comparable baseline; do not claim a formal win from training episodes.
- [x] Add isolated, resumable v2 evaluation on the immutable500 five-distance
  tasks; preserve all legacy training/evaluation artifacts.
- [x] Focused evaluator tests and one tiny execution smoke only.
- [x] Start evaluation, confirm first committed batch, then hand off.

Handoff: evaluator Python PID93949; independent output
artifacts/deployable_observation_v2_fixed500_20260907_v1. Startup snapshot32/500
committed, source task alignment and unified safe-goal semantics validated,
process alive, no ERROR. No training/automatic promotion. Stop monitoring.

Core evaluation evidence is missing, so results-analysis is in read-only audit
mode: no inferential winner claim or publication plots/report. Next research
step is measurement, not adding architecture modules or extending training.
Training descriptives suggest contact-heavy tails despite high goal arrival.
Do not equate availability of battery inputs with learned resource management.

## Active request: deployable observations v2 — LAUNCH AUTHORIZED (2026-09-06)

- [x] Audit online energy/goal/contact sources and preserve legacy interfaces.
- [x] Add versioned distance-only observations and single-channel structured
  encoder with battery, charger, mission mode, and previous-contact context.
- [x] Integrate an explicit preparation/launch interface.
- [x] Run focused tests and a tiny execution-path check; document protocol.
- [x] Launch after user's updated authorization; learned-checkpoint health
  only, then hand off without monitoring to completion.

Handoff: detached Python PID6740, artifacts/deployable_observation_v2_sac_20260906_v1.
Committed8192 transitions; actor and critic3192 optimizer updates each; finite
loss/entropy metrics, complete model/replay/worker-state files, process alive,
no ERROR sentinel. Last block106.47s (7.02s collection). Stop monitoring now.

No reward/physics changes, no privileged obstacle features, no modifications to
ongoing experiments. Observation availability is not evidence of learned energy
sustainability. Existing plan/history below is preserved.

User superseded the pre-launch stop: “这次不用停在启动前了，直接启动即可”.
13 tests passed, including one tiny subprocess smoke and exact checkpoint
continuation for the new worker state. Protocol docs/DEPLOYABLE_OBSERVATION_V2.md.

Error log: initial plan patch used a generic heading that did not match; no
file was changed by that failed patch. Retried against the actual heading.
One later combined documentation patch also failed on a notes heading; the
patch was atomic and no target changed. Retried against read headings.
Final handoff patch also retried after an unmatched notes line; no source or
run artifact was affected by the failed atomic documentation patch.
Startup note: shell-background launch did not persist (no manifest/status and
no live trainer). Relaunched via a detached subprocess session; inspect actual
checkpoint health before claiming success. No training artifacts were lost.

## Current Research: Defective Return Law Under Executed-Interface Shift (2026-09-06)

The 310-anchor raw/HOCBF diagnostic is complete and exposes a real safety--
throughput--energy trade-off.  The next research object is the distribution of
energy-to-safe-recharge with a failure atom, not another navigation variant.

- [x] DR1: Produce a scene-clustered paired analysis bundle for the completed
  raw/HOCBF diagnostic; quantify safe-return gain, arrival loss, energy shift,
  intervention exposure, distance dependence, and catastrophic tails.
- [x] DR2: Freeze a Research Question Card that distinguishes ordinary finite
  mean-energy regression from a defective resource first-passage law
  \(Y=E\) on collision-free timely return and \(Y=+\infty\) otherwise.
- [x] DR3: Implement one physically specified stochastic execution law without
  editing the shared UAV environment or the frozen collision mechanism.
- [ ] DR4: Collect repeated matched raw/HOCBF return rollouts from preserved
  anchors, using scene/anchor/replicate keyed disturbances and atomic resume.
- [x] DR5: Run only focused changed-path tests, one tiny execution smoke, and
  one first-batch health check; then hand the run back without monitoring or
  automatically fitting/promoting a predictor.

### Live collection handoff

- Strict analysis:
  `artifacts/sac_anchor_return_interface_analysis_20260906_v1/analysis-output`.
- Research Question Card:
  `docs/DEFECTIVE_RETURN_LAW_RESEARCH_QUESTION_CARD.md`.
- Frozen collection protocol:
  `docs/STOCHASTIC_DEFECTIVE_RETURN_COLLECTION_PROTOCOL.md`.
- Formal artifact: `artifacts/stochastic_defective_return_310x8_20260906_v1`;
  launcher PID 97592, Python PID 97595.
- Scale: 310 anchors × 8 disturbance replicates × 2 matched interfaces = 4,960
  atomic trajectories.  Raw is collected first, then HOCBF; no model fitting
  or promotion is chained after collection.
- Startup health: 8/2,480 raw trajectories committed, eight active workers,
  finite disturbance/energy/outcome fields, and no error sentinel.  Stop
  monitoring now and wait for the user.

### Locked comparison and stopping logic

- Primary unit for interface inference is the source scene; multiple anchors
  in one scene remain clustered.  Anchor-level discordance is descriptive.
- Structural safe return means charger arrival by 4,000 steps with total
  unified collision count exactly zero.  Contact never terminates execution.
- The stochastic law must be explicit, reproducible, and applied after the
  learned/filter action as physical execution error; policy sampling alone is
  not treated as aleatoric environment uncertainty.
- This is a development feasibility dataset.  It may justify later fresh-scene
  train/calibration/test collection, but cannot itself confirm a paper claim.
- If repeated outcomes remain almost deterministic within anchor/interface,
  stop distribution-model fitting and redesign the disturbance/domain.  If
  meaningful boundary variation appears, compare mean-only and defective-law
  predictors in the next user-authorized phase.

## Current Research: Independent-Review Reconciliation (2026-09-06)

The scratch SAC/PPO comparison is complete.  The next decision must reconcile
that new evidence with `independent-review-20260906` before any additional
training variant is started.

- [x] IR1: Read the independent roadmap, navigation benchmark report, oral
  feasibility report, and research notes; separate literature-backed claims
  from reviewer judgment.
- [x] IR2: Produce a paired, distance-stratified analysis bundle for the frozen
  500-task SAC/PPO evaluation, including training-seed limitations and actual
  PPO optimization diagnostics.
- [x] IR3: Classify every actionable independent-review recommendation as
  retained, revised by new evidence, or rejected.
- [x] IR4: Select one smallest experiment that directly unlocks the energy/
  return research object; do not add another navigation-module stack or a
  preliminary gate chain.
- [x] IR5: Implement only the selected experiment's necessary path, run focused
  checks plus one tiny smoke if new, and confirm one resumable startup
  checkpoint before handing the run back to the user.

### Live handoff

- Analysis bundle: `artifacts/recovery_sac_ppo_scratch_analysis_20260906_v1`.
- Review reconciliation: `docs/INDEPENDENT_REVIEW_RECONCILIATION_20260906.md`.
- Frozen next protocol: `docs/SAC_ANCHOR_RETURN_INTERFACE_PROTOCOL.md`.
- Formal artifact: `artifacts/sac_anchor_return_interface_310_20260906_v1`;
  launcher PID 82149, Python PID 82152.
- Startup health: 34/310 raw-interface anchor rows atomically committed by the
  single allowed check, eight workers active, finite outcomes, no error or
  completion sentinel.  The two-anchor CUDA smoke had completed both raw and
  HOCBF arms.  Stop monitoring now and wait for the user.

### Decision constraints

- The collision-recovery and experiment-startup contracts in `AGENTS.md` are
  immutable.
- The 500 evaluation tasks are paired observations conditional on one trained
  initialization per algorithm.  They support task-set and distance-stratum
  comparisons, not an algorithm-level multi-seed superiority claim.
- A navigation threshold may describe platform fitness, but it cannot be used
  as an oral-level novelty claim or an indefinite blocker on energy research.
- The completed 3,410-rollout action-conditioned option diagnostic is negative
  evidence against another action-level `Q_op` experiment under the old
  executed interface.  The next experiment should instead test the return
  platform or the state-level/resource first-passage object.

## Current Execution: Certified Meet Fusion Fresh Confirmation (2026-09-04)

The one-sided neural hazard development Gate passed six of seven registered
checks but improved CRC coverage over geometry by only 0.85 percentage points,
below the locked two-point requirement.  That branch is not promoted.  A
mechanism control, the pointwise meet of the geometry/action and direct
LiDAR/action neural critics, instead exposed a stronger safety--coverage point
and now receives one preregistered fresh-scene confirmation.

- [x] CM1: Derive the one-sided hazard and meet structural properties, including
  explicit non-claims for separately calibrated thresholds.
- [x] CM2: Complete the fivefold 150-scene hazard Gate.  Preserve invalid v1
  (calibration-semantics mismatch) and valid v2 (`DO_NOT_COLLECT_FRESH_CONFIRMATION`).
- [x] CM3: Freeze the v2 selective-prediction estimand, 90/30/30 development
  split, neural critics, CRC thresholds, seeds, metrics, and promotion rule.
- [x] CM4: Implement model freeze, fresh evaluator, nominal-only source mode,
  atomic chained execution, and focused tests; 32/32 pass.
- [x] CM5: Run new-seed nominal-source and fork smoke.  Structural reconstruction
  and branch checks pass; smoke outcome-variation checks are intentionally
  non-evidence because branch continuations are censored at four steps.
- [ ] CM6: Complete the live 75-scene fresh chain (75 nominal rollouts, 300
  anchors, 3,000 fork branches, then one frozen evaluation) and apply every
  registered Gate without changing R3.

### Locked evidence contract

- Primary score: `min(geometry_action, lidar_action_direct)`, where both inputs
  are learned budget-monotone critics and R3 is frozen.
- Structural theorem: at a common threshold the meet acceptance set is the
  intersection of the two branch acceptance sets, so its false-safe set is a
  subset of either; pointwise minimum preserves budget monotonicity.
- Statistical target: maximize safe-declaration coverage subject to
  scene-level false-safe risk at most 0.05, with a CRC threshold selected on 30
  development calibration scenes.
- Fresh confirmation uses task seed 420001 and world seed 430001, disjoint from
  development.  Promotion requires test risk <=0.05, >=2-point coverage gain,
  paired-scene coverage CI lower bound >0, same-threshold danger no worse, and
  zero dominance/monotonicity violations.
- V1 of this confirmation protocol was retired before fresh collection because
  it incorrectly required both the fixed risk target and risk no larger than a
  baseline that could under-use that budget.  V2 fixes the estimand and is
  immutable during the live run.

### Live execution

- Initial unified session: `87525`; resumed fork/evaluation session: `68753`.
- Health check at 37.47 s: 30/75 nominal rollouts complete, eight active, 37
  pending, valid `RUNNING.json`, no failure sentinel.  Per user instruction,
  no further monitoring occurs until requested.
- Nominal collection completed 75/75 in 234.85 s.  The first fork process
  stopped at 1,373 saved branches when compact float32 reconstruction placed
  `scene_034_anchor_02` 49.6 micrometers below the 0.5 m altitude boundary;
  the worker's established ulp-bounded sanitization produced the correct 0.5 m
  state and therefore a different exact hash.  Preparation now performs the
  identical projection before hashing.  Resume preserved all atomic branches,
  migrated the unstarted anchor to exactly 0.5 m, and was verified live at
  1,374/3,000 with eight workers and no current failure marker.
- Protocol: `docs/CERTIFIED_MEET_FUSION_CONFIRMATION_PROTOCOL_V2.md`.
- Frozen model: `artifacts/certified_meet_fusion_freeze_150scenes_20260904_v2`.

## Current Execution: Independent LiDAR/Action Confirmation (2026-09-04)

The development Gate showed that direct action-conditioned LiDAR fusion was
the only learned variant that improved both Brier and dangerous false-safe over
geometry.  Because it was registered as a control, it cannot be promoted from
that result.  The next Gate freezes those three fitted fold members and tests
them once on the untouched half of the source scene inventory.

- [x] IC1: Lock the new primary, untouched scene set, frozen predictors,
  scene-level statistics, conditional safety metrics, and promotion rule.
- [x] IC2: Add disjoint per-bucket scene offsets and repair future anchor index
  provenance; verify the exact unused scene slice with a 50-branch smoke.
- [x] IC3: Implement frozen three-member ensemble evaluation, scene-paired
  bootstrap intervals, conditional false-safe and unsafe-among-accepted rates,
  exact hashes, and terminal result semantics.
- [x] IC4: Collect the remaining 75 scenes (300 anchors, 3,000 branches).  Run
  independent evaluation automatically only after the upstream data Gate
  passes.
- [x] IC5: Apply the preregistered rule.  Direct LiDAR/action improves point
  Brier by 6.64% and all three frozen members win, but the paired-scene Brier
  interval crosses zero and dangerous false-safe is worse.  Final decision is
  `DO_NOT_MODIFY_R3_ACTOR`; no actor pilot is authorized.

### Locked confirmation evidence contract

- Development and confirmation scene sets are disjoint 75/75 halves within
  each of the five distance buckets; no confirmation outcome selects weights,
  normalization, temperatures, thresholds, or architecture.
- Primary: the frozen three-member `lidar_action_direct` probability ensemble.
  Baseline: the matching frozen `geometry_action` ensemble.  Residual and
  no-action variants remain mechanism controls.
- Primary accuracy: confirmation-scene-averaged Brier.  Safety reports joint
  dangerous false-safe, false-safe conditional on infeasibility,
  unsafe-among-accepted, and safe-declaration coverage at the fixed 0.90
  threshold.
- Promotion requires >=2% Brier improvement, paired-scene bootstrap Brier upper
  bound <0, dangerous false-safe point estimate no worse with 95% upper
  difference <=+0.005, two of three member wins, exact budget monotonicity,
  nonzero matched-action sensitivity, and a passing 3,000-branch data Gate.
- The protocol is
  `docs/INDEPENDENT_LIDAR_ACTION_CONFIRMATION_PROTOCOL.md`.  Passing authorizes
  only a later bounded actor pilot; failing preserves R3 unchanged.

### Errors encountered

- The initial confirmation collection stopped after 2,493 atomic files when
  two not-yet-started scene-137 anchors reconstructed 25.7 and 51.0 micrometers
  below the 0.5 m legal altitude.  This is float32 reconstruction noise, not a
  physical boundary event.  Snapshot positions now admit at most eight
  float32 ulps at world scale (about 3.8 mm here), project that numerical excess
  to the legal interior, and still reject a 1 cm violation.
- Resume migrated only the two wholly unstarted anchors, regenerated their
  observation hashes and action candidates, and preserved all existing branch
  files.  The relevant suite passes 20/20.  One-time resumed health check found
  both heights exactly 0.5 m, 2,513/3,000 branches complete, eight active, no
  failure sentinel, and a live CUDA-registered process.
- The resumed collection completed 3,000/3,000 and passed all upstream checks.
  Frozen evaluation completed without error on 75 disjoint confirmation
  scenes.  Direct ensemble Brier was 0.085261 versus geometry 0.091324 (6.64%
  lower), but paired-scene direct-minus-geometry 95% bootstrap CI was
  [-0.023109, +0.016800].  Dangerous false-safe was 4.016% versus 2.984%, with
  difference CI [-0.006512, +0.035956], failing both registered safety checks.
- The generalization failure is concentrated at >4,000 m: direct Brier 0.19358
  versus geometry 0.17125, and joint dangerous false-safe 11.92% versus 7.17%.
  Direct improved Brier in all three middle-distance buckets but slightly lost
  at 100--500 m.  This blocks a global actor integration claim.

## Current Execution: Large-Scale Geometry + LiDAR Residual Gate (2026-09-04)

The 15-scene forked-action pilot passed its data Gate, including a directional
collision signal for actions aimed toward obstacles.  The authorized next step
is a bounded large-scale collection followed automatically by a held-out
critic comparison.  This is still a feasibility experiment, not actor training.

- [x] LS1: Lock scale, scene unit, baseline matrix, primary metric, residual
  parameterization, and stop condition before starting the run.
- [x] LS2: Implement the forked-dataset loader, frozen R3 LiDAR embedding,
  geometry features, and all-budget labels without branch-outcome leakage.
- [x] LS3: Implement three-fold geometry, LiDAR-no-action, direct LiDAR-action,
  and geometry-plus-LiDAR/action residual critics.  Include epoch zero in
  residual model selection so the full model cannot be promoted by degrading
  its locked geometry parent.
- [x] LS4: Add focused tests and one critic-only smoke using the completed
  15-scene pilot.
- [x] LS5: Launch a resumable chain: collect 75 stratified scenes (300 anchors,
  3,000 branches), run the critic Gate after a passing data Gate, health-check
  once, then stop monitoring.
- [x] LS6: Apply the locked promotion rule.  The repaired residual improves
  scene-averaged Brier by 8.75% and wins two folds, but its dangerous
  false-safe rate is 4.06% versus 3.92% for geometry.  Therefore the formal
  decision is `DO_NOT_MODIFY_R3_ACTOR`; actor integration is not authorized.

### Locked large-scale evidence contract

- Scale: 75 independent scenes, 15 per distance bucket; four task/return
  anchors and ten fixed-state action branches per scene.
- Runtime expectation: approximately 1--2 hours of fork simulation plus
  minutes of GPU critic fitting; no experiment configuration may exceed ten
  hours by expected pilot throughput.
- Geometry baseline: task distance, charger distance, leg/deadline, velocity,
  nearest-obstacle clearance, proposed action, goal/obstacle alignment, and
  budget.
- Frozen perception: the confirmed R3 structured-LiDAR encoder; actor weights
  remain immutable.
- Primary model: a monotone budget CDF whose geometry logits are frozen and
  whose LiDAR/action residual starts identically at zero.  Positive knot
  increments preserve budget monotonicity after residual adaptation.
- Controls: geometry only; LiDAR/state without current action; direct
  LiDAR+action without the locked geometry parent; full residual model.
- Split: complete scene seeds by outer fold; anchors, actions, and budget
  queries from a scene never cross folds.  Inner scene validation selects epoch
  and temperature.
- Primary metric: scene-averaged Brier over fixed battery-budget queries.
  Secondary metrics: AUROC, ECE, dangerous false-safe rate, budget monotonicity,
  mixed-anchor ranking, and distance-bucket breakdown.
- Promotion: full residual Brier at least 2% below geometry pooled and below it
  on at least two folds, zero monotonicity violations, nonzero matched-action
  sensitivity, and no worse dangerous false-safe rate.  Otherwise actor work
  remains blocked.

### Errors encountered

- The first critic smoke exposed empty validation partitions.  Collection
  scene IDs encode the distance bucket (for example 0, 30, 60), while the
  original splitter treated those provenance IDs as dense indices and applied
  modulo arithmetic directly.  The splitter now operates on each unique
  scene's stable sorted rank.  A sparse-ID regression test verifies nonempty,
  mutually disjoint train/validation/test scene sets in all three folds.
- After the split repair, 14/14 focused tests and the full four-variant CUDA
  smoke completed.  The smoke retains only three epochs and 15 scenes, so its
  scores are integration evidence only and cannot satisfy LS6.
- The first 75-scene critic attempt exposed two additional implementation
  defects.  `anchor_index` was assigned before append as `len(anchors)-1`, so
  metadata indices lagged the canonical NPZ row by one; the physical branch
  outcomes remain valid, but the first cache and fold 0 are quarantined.  The
  loader now binds anchor IDs to list rows and verifies every observation hash,
  while new collections write `len(anchors)`.  The residual inverse-softplus
  used `log(expm1(x))`, which overflowed for observed scaled increments up to
  153.26; it now uses the stable identity `x + log(-expm1(-x))`.
- Regression and integration verification passes 16/16 tests.  The repaired
  CUDA smoke completed all folds with canonical anchors 0--299 and finite
  checkpoints.  Formal v2 also completed all three folds without a failure
  sentinel.


## Current Execution: LiDAR Forked-Action Causal Data Gate (2026-09-04)

The action-conditioned critic Gate completed but did not promote: its Brier was
0.05275 versus 0.04232 for geometry, while removing current action improved it
to 0.04988.  Bellman consistency helped its own residual, so the missing object
is not another loss coefficient.  The observational trajectories do not
identify the effect of one action at a fixed state and contain only five
collision rollouts.  The next experiment changes the data, not the head.

- [x] FA1: Lock the causal unit, action set, execution interface, leakage rule,
  data metrics, and stop conditions before collection.
- [x] FA2: Add a fork-reset worker operation that reconstructs the same static
  scene and exact pre-action physical state without replaying the prefix.
- [x] FA3: Select scene-grouped task/return anchors using nominal R3 trajectories,
  oversampling low-clearance LiDAR states without treating anchors as
  independent test samples.
- [x] FA4: From every anchor execute the same ten registered macro-actions raw
  for eight policy steps, then restore frozen R3 plus the unchanged HOCBF
  continuation and measure safe recharge, collision, deadline, and energy.
- [x] FA5: Verify focused tests and one small smoke.  If healthy, launch the
  resumable 15-scene pilot and stop monitoring after one process/data check.
- [ ] FA6: Train a LiDAR+action critic only if the paired-data Gate has adequate
  executed-action diversity and within-anchor outcome/resource variation.

### Locked evidence semantics

- Independent statistical unit: scene seed.  Anchors, actions, budgets, task
  and return legs from one scene must remain in one fold.
- Causal contrast: all branches at an anchor share position, velocity, task,
  charger, obstacle layout, LiDAR, and continuation policy; only the registered
  initial macro-action changes.
- Raw interface: HOCBF is disabled only for the eight repeated candidate-action
  steps.  Any collision or boundary contact makes the branch unsafe.  HOCBF is
  restored for the frozen-R3 continuation so the label isolates the proposed
  action rather than an uncontrolled future policy.
- Candidate set: nominal, brake, nominal +/- 0.45 on each action axis, toward
  nearest obstacle, and away from nearest obstacle.  Clipping and the actual
  executed action are recorded.
- Pilot scale: 15 scenes stratified equally over the five distance buckets,
  four anchors per scene, ten branches per anchor (600 branches maximum).
- Data Gate: exact scene/anchor match, at least five distinct executed actions
  for at least 80% of anchors, unsafe prevalence between 2% and 40%, and paired
  safety or >=5% successful-energy variation for at least 20% of anchors.
- This is a data Gate, not publication evidence and not actor training.  A fail
  changes anchor/action sampling; it does not authorize a larger neural model.

### Errors encountered

- The first forked-data test expected terminal leg endpoints to be eligible
  anchors, contradicting the locked protocol.  The test now checks the nearest
  interior state on each leg.
- The first implementation of distinct-action counting inverted its pairwise
  separation condition.  It now adds a representative only when it is at least
  0.05 away from every previous representative.
- The formal pilot stopped after 73 atomic branches because a reconstructed
  float32 velocity exceeded the 20 m/s horizontal norm by about 1.6e-6.  This
  is integration-rounding noise already observed in the earlier CDF work, not
  a physical divergence.  Fork reset now projects only excess within a
  machine-epsilon-derived tolerance and still rejects material overspeed.

The corrected focused suite passes 16/16.  The 5-scene/50-branch smoke matched
every anchor observation and obstacle digest exactly, retained action diversity,
and reconstructed the source observation prefix within 5.96e-8.  Its four-step
censoring intentionally makes all labels non-evidence and is not a failed data
Gate.

**FA DATA GATE COMPLETED AND PROMOTED.**  The repaired run completed all 600
branches in 925.5 s after resume at
`artifacts/lidar_forked_action_data_gate_15scenes_20260904_v1`.  Every registered
Gate passed: exact paired observation/layout hashes, 100% action-diverse
anchors, 7.83% unsafe prevalence, and 46.67% paired safety-or-energy variation.
Collision prevalence was 3.67% and boundary prevalence 1.50%.  Only 5/60
anchors contain both safe and unsafe branches, so FA6 is authorized as a small
learnability Gate, not as actor training or a collision-safety claim.

## Current Execution: Action-Conditioned Rechargeability Gate (2026-09-03)

The three passive resource-prediction Gates are now terminal and all lost to
the locked geometry baseline.  The grouped counterfactual collector completed
750/750 rollouts and passed its data-diversity checks, but its trajectory-level
learned predictor also lost to geometry.  This rules out more tuning of a
passive energy head on the same estimand.  The next object is the pre-action
probability of safe, timely recharge under a finite energy budget.

- [x] RQ1: Freeze the scientific pivot: energy safety means budgeted
  reach-avoid-to-recharge, not low expected consumption and not a fixed SOC
  threshold.
- [x] RQ2: Recover aligned pre-action physical state and both goals from the
  locked task seed plus the saved active-goal direction/log-distance.  The
  maximum initial-position reconstruction error is 0.00098 m.  Retain a fresh
  LiDAR-complete collection as a mandatory later collision-tail stage.
- [x] RQ3: Implement a budget-monotone action-conditioned rechargeability
  critic and the finite-horizon one-step Bellman target, with focused tests for
  monotonicity, terminal cases, action dependence, and scene-disjoint splits.
- [x] RQ4: Run a resumable Gate comparing the critic with geometry-only,
  no-action, and no-Bellman ablations on held-out scene seeds.
- [ ] RQ5: Only if RQ4 passes, connect the calibrated lower confidence value to
  an actor update.  No R3 actor weight changes are authorized by a failed Gate.

### Locked estimand and Gate

- Estimand: `Q_h(x, b, a) = P(safe charger hit before unsafe/deadline and
  cumulative executed energy <= b | x, a)` under the declared continuation
  policy and intervention distribution.
- Statistical unit: scene seed.  Every step, budget query, suffix, intervention,
  and task/return leg from a scene remains in one fold.
- Input-time rule: a row contains only information available immediately before
  the proposed action.  Current-step energy, current HOCBF result, future path
  length, suffix energy, and terminal result are label-only fields.
- Primary score: scene-averaged Brier score on held-out scenes, with dangerous
  false-safe rate and calibration reported at the decision boundary.
- Promotion: the full critic must improve pooled geometry Brier by at least 2%,
  improve on at least two of three outer folds, retain nonzero matched-action
  sensitivity, and reduce held-out Bellman residual versus the no-Bellman
  ablation.  Smoke data cannot promote the method.
- Runtime: use `.venv` through `uv`, write atomic/resumable outputs, and keep
  every launched pilot below ten hours by construction.  Perform one health
  check after launch, then stop monitoring until the user asks.

**RQ4 COMPLETED, NOT PROMOTED.**  The formal 150-scene Gate completed all 12
units at `artifacts/rechargeability_critic_gate_seed350001_20260903_v2`.
Full Brier 0.05275 lost to geometry 0.04232, and no-action was better than full
at 0.04988.  Bellman residual improved, but observational action conditioning
did not generalize; actor training remains blocked on the forked-action Gate.

## Current User Priority: Learned Safe and Energy-Sustainable Navigation (2026-09-03)

The user asked to continue toward a neural policy that learns collision-safe,
energy-sustainable navigation rather than relying on a hand-authored return
threshold or route planner.  This is an extension of, not a silent replacement
for, the authoritative `ORIGINAL_GOAL.txt` Resource-to-Go line.

### Learned-Sustainability Phases

- [x] L1: Search exact-overlap literature across safe RL, distributional RL,
  reach-avoid control, stochastic shortest paths, and replenishable-resource
  formal methods.
- [x] L2: Identify the nearest prior art and reject already-occupied novelty
  claims, especially deterministic/stochastic minimum-cost reach-avoid.
- [x] L3: Define a joint random object that represents collision, return
  deadline, and finite battery without an arbitrary weighted reward.
- [x] L4: Derive its finite-horizon distributional recursion, certificate,
  monotonicity conditions, approximation margin, and repeated-recharge risk
  composition theorem.
- [x] L5: Materialize a literature evidence packet and a separate derivation
  package with explicit non-claims and implementation consequences.
- [x] L6: Audit the running Phase 6b.8 Gate and select one diagnostic
  energy-head experiment rather than changing the actor before learnability is
  established.
- [x] L7: Complete the extended-real resource-CDF learnability pilot.  It
  completed 30 tasks but did not promote: 12/120 initial queries were negative,
  CDF AUROC was 0.434, and the geometry baseline was stronger at 0.725.
- [x] L8: Run the boundary-balanced scaling experiment that separates data
  scale, frozen-navigation representation, and optimization duration before any
  on-policy actor change.  It completed all 300 tasks and rejected promotion:
  the fixed `frozen_encoder@400` model had AUROC 0.882, Brier 0.169, and ECE
  0.160 versus geometry-baseline Brier 0.101.  Training beyond 25 epochs worsened
  calibration rather than repairing underfitting.
- [x] L9: Test a jointly trainable resource-aware encoder with nested
  task-disjoint validation, validation-Brier early stopping, and an explicit
  query-calibration loss.  Do not modify the deployed R3 actor unless this
  supervised representation Gate beats the locked geometry baseline on held-out
  task seeds.  The run completed but did not promote: calibrated Brier was
  0.124 versus 0.101 for geometry, and all three outer folds were worse.
- [x] L10: Replace the over-parameterized categorical head with an identifiable
  defective continuous CDF: one failure-mass head, one conditional log-energy
  location head, and one globally calibrated scale.  Evaluate compact,
  resource-encoder, fusion, and no-Brier variants under the unchanged outer
  test folds before any actor modification.  The 12/12-unit formal Gate
  completed in 5.3 minutes but did not promote: the primary fusion-CDF model
  had Brier 0.1203 and AUROC 0.8989 versus geometry at 0.1013 and 0.9233; every
  outer fold was worse.  Compact inputs were strongest among learned variants,
  and the no-CDF fusion beat the fusion CDF, so further head tuning is rejected.
- [ ] L11: Build grouped counterfactual trajectory evidence from the frozen R3
  navigation system: for each independent scene seed, execute multiple
  stochastic action interventions from the same initial state and retain
  action, collision margin, completion/return outcome, and physical energy.
  Keep complete scene groups disjoint across train/validation/test.
- [ ] L12: Establish a data Gate before policy training: verify within-scene
  action diversity, non-degenerate safety/energy outcomes, deterministic replay
  metadata, physical energy accounting, and measurable conditional signal over
  geometry-only prediction.
- [ ] L13: Add the smallest joint neural-policy experiment that consumes the
  counterfactual groups and optimizes navigation return subject to learned
  collision and resource costs.  Compare frozen R3, R3 fine-tuning without the
  new signal, and the full joint objective on held-out scene seeds.
- [ ] L14: Run focused tests and one non-evidence smoke; only if L12 passes,
  launch a fresh resumable formal experiment with atomic scene checkpoints and
  perform a single process/metric health check.

### Grouped Counterfactual Safe-Energy Experiment (2026-09-03)

- Independent statistical unit: scene/task seed.  Action interventions,
  horizons, budgets, and suffixes within one scene are correlated augmentation,
  never separate test samples.
- Primary data question: within the same initial scene, do feasible policy
  interventions induce enough variation in collision/return outcome and
  physical energy to identify an action-conditioned learning target?
- Locked baselines: frozen R3; geometry-only risk/energy predictor; neural model
  trained on ungrouped observational trajectories; joint model without energy;
  joint model without safety.
- Data Gate: no train/test scene overlap; at least three distinct executed
  intervention classes per retained scene; both safe and unsafe outcomes and a
  nontrivial finite-energy spread; action-conditioned predictor must beat the
  geometry-only predictor on held-out scenes before R3 actor weights change.
- Runtime contract: use the existing uv environment, atomic per-scene outputs,
  resumable checkpoints, no wall-clock kill, and no claim based on smoke data.

**ACTIVE, FORMAL DATA GATE RUNNING.**  The grouped collector, matched-obstacle
task-to-return retarget path, correlated intervention process, physical-energy
records, grouped cross-fitting, and terminal Gate are implemented.  The focused
suite passes 90/90.  A 25-rollout real-R3 smoke exercised 7,109 transitions,
including five return legs, with zero matched-initial-observation discrepancy;
resume also completed successfully.  The 150-scene x five-intervention formal
run is active at
`artifacts/r3_grouped_counterfactual_safe_energy_gate_seed310001_20260903_v1`.
Its single health check observed 15/750 atomically completed rollouts, eight
active workers, a live CUDA process, and no failure sentinel.  Do not modify or
launch actor training until this fixed Gate reaches a promotable terminal result.

### Defective Continuous Resource-CDF Gate (2026-09-03)

- [x] D1: Audit target identifiability and write the exact defective-CDF
  derivation, including why state-dependent variance is not identified here.
- [x] D2: Lock the task-disjoint protocol, strong baseline, ablations, early
  stopping, calibration split, and promotion criteria.
- [x] D3: Implement the compact two-head model and learned LiDAR-fusion model,
  with task-weighted failure, finite-energy, and budget-probability losses.
- [x] D4: Verify formula properties and split integrity with focused tests, run
  one real-data smoke, then launch the resumable full Gate and health-check it
  once.

**ACTIVE, NOT MONITORED.**  The combined focused suite passes 29/29 under uv,
the 75-task real-data smoke completed, and the full 300-task run writes to
`artifacts/r3_defective_resource_cdf_gate_seed290001_20260903_v1`.  Its one-time
health check observed `compact_hurdle` fold 0 atomically complete (best epoch
10, stopped epoch 160), a compute-active process, and no failure sentinel.
Success still means beating geometry on untouched task seeds; completing a run
is not success.

### Resource-Aware Encoder Gate (2026-09-03)

- [x] P1: Lock the train/validation/test task split, primary metric, baselines,
  training objective, early-stopping rule, and promotion Gate in a protocol.
- [x] P2: Implement a trainable copy of the R3 structured encoder and the
  categorical resource head without mutating the R3 checkpoint or actor.
- [x] P3: Add focused unit tests for task-disjoint splits, cumulative budget
  probabilities, calibration loss, checkpoint selection, and resume semantics.
- [x] P4: Run the minimum smoke test, then launch the full exploratory run in a
  fresh resumable output directory and perform one health check only.

**COMPLETED, NOT PROMOTED.**  The full run at
`artifacts/r3_resource_aware_encoder_gate_seed270001_20260903_v1` completed all
nine variant-folds.  The primary improved calibration to ECE 0.035, but its
Brier 0.124 and AUROC 0.896 remained below geometry (0.101 and 0.923).  Each
primary fold selected the minimum allowed epoch 25 and later validation scores
worsened, ruling out additional epochs as the repair under this model.

### Boundary-Balanced Scaling Experiment (2026-09-03)

- Primary question: can a resource-CDF head generalize to unseen task seeds when
  training data cover the battery boundary and the 2055-dimensional LiDAR input
  is compressed by the frozen R3 navigation encoder?
- Dataset: 300 independent task/obstacle seeds, 60 per distance bucket; four
  deadline horizons; eight counterfactual battery fractions from 0.08 to 0.36.
  All split operations group by task seed.  Suffix snapshots are training
  augmentation, not independent evaluation units.
- Mechanism matrix: raw 2066-dimensional input, frozen dual-goal R3 encoder
  features, and an 18-dimensional compact structured input.  Each is evaluated
  at 25, 100, and 400 epochs under the same folds and categorical resource bins.
- Main model fixed before execution: frozen dual-goal R3 encoder at 400 epochs.
  Simple baselines are fold-local prevalence and geometry/deadline/budget
  logistic regression.
- Promotion requires sufficient positive/negative boundary queries, held-out
  AUROC >= 0.75, ECE <= 0.15, and Brier score at least 2% below the geometry
  baseline.  Results remain exploratory and cannot establish closed-loop or
  lifecycle safety.
- Runtime contract: uv environment, no wall-clock kill, per-task atomic data,
  epoch checkpoints, fresh versioned output, and no modification of the frozen
  R3 checkpoint or HOCBF runtime.
- Completed output:
  `artifacts/r3_resource_cdf_boundary_balanced_scaling_300tasks_seed240001_20260903_v1`.
  It completed in 1.35 h with 5,798 positive and 3,802 negative budget queries.
  The compact 25-epoch model was the best learned calibration point (Brier
  0.122), but it still underperformed the geometry baseline (Brier 0.101).
- Smoke error: the reduced smoke matrix contains only `compact@2`, while the
  terminal decision initially assumed the formal `frozen_encoder@400` key; its
  deliberately short rollouts also contained no finite energy for `nanmax`.
  Resolution: make the smoke primary explicit and treat an empty finite-energy
  set as `null`, without changing the fixed formal primary or Gate.

### Chained Learnability Pilot (2026-09-03)

- Phase 6b.8 v5 completed all three seed families and fifteen candidate cycles
  without a worker failure.  This closes the corrected-rule mechanism preflight,
  but the selected three seeds remain non-formal descriptive evidence.
- Seed 110004 is descriptive warning evidence, not a Gate result.  SOC 0.20 and
  both distance points each returned safely after completing two tasks.  Both
  Oracle reserve points committed at the initial charger state and completed
  zero tasks because the 4,000-step task-then-return deadline was classified
  infeasible.  This Oracle cannot be blindly distilled as a teacher.
- The selected follow-on is
  `extended_real_resource_cdf_learnability_pilot_v1`, running at
  `artifacts/r3_resource_cdf_learnability_pilot_seed140001_20260903_v1`.
  It consumed the completed `COMPLETED_DIAGNOSTIC.json`; collection is resumable
  per task and has no wall-clock timeout.
- The pilot freezes R3, samples 30 independent task/obstacle seeds over all five
  distance buckets, reuses each rollout under 1k/2k/3k/4k deadline queries, and
  fits one 32-bin finite-energy distribution plus an explicit failure atom.
  Three-fold evaluation holds out complete task seeds.  The comparison set is
  a constant predictor and a geometry/deadline logistic baseline.
- Promotion requires class support, held-out AUROC at least 0.70, ECE at most
  0.20, and at least a two-percent Brier improvement over the best simple
  baseline.  These are internal exploration thresholds, not theorem or paper
  evidence.  Even a pass does not establish closed-loop improvement or
  lifecycle safety.

### Learned-Sustainability Decision

**MATHEMATICALLY COHERENT, BUT CURRENT OBJECT IS NOT STANDALONE NOVELTY.**  RC-PPO
(NeurIPS 2024) already solves deterministic minimum-cost reach-avoid, and
RAPCPO (accepted ICML 2026) extends this to stochastic reach-avoid probability
with expected cost.  Consumption-MDP theory (CAV 2020) already models batteries,
atomic recharge states, and repeated almost-sure objectives in finite known
MDPs.  Therefore, neither "safe navigation plus lower energy" nor "battery in
the state" is a contribution.  The remaining defensible candidate is learning
the full finite-horizon distribution of an extended-real resource-to-recharge
random variable, whose mass at infinity jointly records collision/deadline
failure, and using calibrated lower CDF bounds to train a budget-conditioned
continuous policy.  Equation-level inspection shows that a fixed-budget query
of this CDF is equivalent to reach-avoid on a battery-augmented state, so this
representation alone is insufficient for an oral-level claim.  The remaining
research target is a finite-sample approximation-error-to-lifecycle-risk theorem
for quantitative repeated recharge under continuous unknown dynamics, plus raw
policy evidence separating it from RC-PPO/RAPCPO/QCPO/SDAC.  Until that exists,
the proposal is an integration direction rather than a core theoretical result.

## Current User Priority: Accelerated Energy-Equivalence Gate and Phase 6b.8 (2026-09-02)

The user accepted R3 as the frozen conditional navigation baseline and authorized
the next experiment.  First establish that the accelerated safety/runtime path
preserves the energy/return closed-loop estimand; only a passing equivalence
check may launch the corrected three-seed viability-order preflight.

### Energy-Continuation Phases

- [x] E1: Inventory the existing equivalence tests, Phase 6b.8 runner, stopped
  artifacts, and immutable R3/energy contracts.
- [x] E2: Run isolated array-path Python-QP versus native-QP trajectory
  equivalence plus same-solver object-versus-array constraint snapshots under
  `uv`, and retain versioned evidence.
- [x] E3: If and only if E2 passes, launch a fresh Phase 6b.8 run on seeds
  110004, 110012, and 110013 with the corrected v7 viability rule, one worker,
  two retries, checkpoint-safe atomic outputs, and a sub-10-hour limit.
- [x] E4: Verify one healthy background progress signal, then stop active
  monitoring until the user asks for results.

### Energy-Continuation Invariants

1. The frozen R3 checkpoint, battery capacity, task seeds, candidate family,
   corrected v7 decision semantics, and scientific Gates are unchanged.
2. Acceleration may change implementation/runtime only; paired executed states,
   actions, safety decisions, energy increments, and terminal semantics must
   agree within a declared numerical tolerance.
3. Failed equivalence evidence blocks Phase 6b.8; it may not be waived by a
   speed improvement.
4. A fresh versioned output directory is required; terminal v2/v4 artifacts are
   never overwritten or relabeled.

### Energy-Continuation Status

**E2 v2 PASSED; E3 LAUNCHED AND LEFT UNMONITORED.**  The corrected equivalence
artifact at `artifacts/r3_energy_runtime_equivalence_isolated_seed110004_20260902_v2`
reports identical Python/native QP traces, zero maximum action delta, two tasks,
charger reached, and no collision.  The fail-closed chain therefore launched
the three-seed Phase 6b.8 v5 run at
`artifacts/r3_oracle_headroom_stage_b_viability_order_preflight_v5_accelerated`.
Per user instruction, do not inspect or alter that process until asked.

### Energy-Continuation Errors

- The first focused pytest invocation used the nonexistent class name
  `UAVSafetyEnergyFilterTests`, so pytest collected neither requested HOCBF
  test.  Inspection showed both tests belong to `ProjectionTests`; rerun with
  the corrected node IDs before treating the regression Gate as passed.
- Two attempts to detach the conditional shell with `setsid`/`nohup` did not
  survive the command executor despite returning transient wrapper PIDs; they
  created no v5 directory and changed no evidence.  Resolution: run the
  conditional Gate chain in persistent unified session `2985`.
- E2 v1 changed ordinary HOCBF into a zero-weight energy-aware formulation to
  force the scalar builder.  Although the mathematical objective differs only
  by a positive scale, the fixed iterative tolerance is not scale-invariant;
  the test therefore conflated three factors and cannot reject acceleration.
  It also wrote both `COMPLETED.json` and `FAILED.json`.  Resolution: isolate
  QP-backend trajectory equivalence from constraint-builder snapshots and emit
  exactly one terminal sentinel in v2.
- The first conditional launcher used nested shell quoting that was altered by
  the command transport, producing `syntax error: unexpected end of file`.
  Resolution: launch the corrected verifier and Phase 6b.8 as one direct
  fail-closed unified command chain, without a nested watcher shell.

## Current User Priority: R7 Exact-Semantics Runtime Acceleration (2026-09-01)

The user authorized implementation of the previously researched acceleration
path. The change must preserve the sampled-data robust HOCBF constraint set and
the resulting safety-filter solution; increasing GPU utilization is not itself
the objective.

### Runtime-Acceleration Phases

- [x] A1: Compare the historical R3 SAC execution schedule and safety-filter
  contract against current R7.
- [x] A2: Add an array-native sampled-data robust LiDAR/HOCBF path with no
  per-hit Python obstacle construction.
- [x] A3: Add an exact constraint-generation QP path for the three-dimensional
  projection problem, with a full-feasibility termination check.
- [x] A4: Add scalar/vectorized and reference/optimized equivalence tests,
  including infeasible and fallback cases.
- [x] A5: Run focused uv regressions and a deterministic microbenchmark without
  modifying the active experiment process.

### Runtime-Acceleration Invariants

1. No approximate top-k truncation is introduced into the current R7 contract.
2. The sampled-data residual formula, actuator constraints, solver tolerance,
   fallback behavior, and scientific diagnostics remain unchanged.
3. A constraint-generation candidate is accepted only after checking every
   original inequality; otherwise another violated constraint is added.
4. The active background run is historical code-in-memory and is not restarted
   or relabeled by source edits.

### Runtime-Acceleration Status

**A5 COMPLETE.** Historical SAC evidence shows that its higher GPU duty came
from 495,000 replay-gradient updates and a lighter ordinary-HOCBF/top-16 safety
contract, not GPU physics. R7 now has an array-native robust-HOCBF path, exact
constraint generation, and an optional native C implementation of the unchanged
three-variable QP sweep. The collision/projection suite passed 80 tests and the
environment/R7 trainer suite passed 96 tests. Fixed near-obstacle policy steps
fell from roughly 78--236 ms to 8--18 ms. The existing background workers remain
on their imported old code; only a newly started/resumed process uses this path.

## Current User Priority: ICLR Paper Readiness and Story Audit (2026-09-01)

The active task is to assess the current `paper/iclr2027/` manuscript against
ICLR scientific, evidence, writing, and format expectations, then sharpen the
research story so that collision safety is a mature, best-fit execution layer
and energy safety is the paper's central research contribution. This is an
assessment-and-idea-optimization pass: pending evidence must remain pending,
and no collision component may be promoted as novel without source-backed
support.

### Paper-Audit Phases

- [x] P1: Locate the manuscript, evidence ledgers, prior reviews, and current build.
- [x] P2: Extract claims, method roles, empirical support, theory scope, and venue compliance.
- [x] P3: Produce a strict ICLR readiness review with calibrated scores and repair conditions.
- [x] P4: Produce an optimized collision-as-infrastructure / energy-as-contribution research blueprint.
- [x] P5: Verify the report against the manuscript, build logs, citations, and unresolved result gates.

### Paper-Audit Invariants

1. Collision safety is evaluated for suitability and coverage, not automatically claimed as novel.
2. Energy/resource safety owns the central problem, mechanism, theory, and discriminating evidence.
3. Red `RESULT PENDING` markers remain authoritative until their declared gates pass.
4. Existing experimental artifacts are evidence only when provenance and admissibility ledgers allow them.
5. Readiness and development potential are reported separately.

### Paper-Audit Status

**P5 COMPLETE.** The strict review and story blueprint were checked against the
frozen JSON, completion ledger, manuscript source, existing PDF/build log,
literature packet, and ICLR 2027 policy pages. No manuscript prose or pending
result was changed. Fresh compilation remains unavailable in this environment;
that limitation is recorded below.

## Current User Priority: R7 PPO Navigation Repair (2026-09-01)

The user has explicitly moved the active implementation priority from the
paused R3 return-to-charge Gate to repairing the failed learned-navigation PPO.
R3 evidence and Gate order remain preserved, but no R3 experiment may be
restarted until the user returns to that objective.

### R7 Goal

Replace SAC only at the optimizer/return-estimation layer while preserving the
successful R3 structured LiDAR perception, navigation environment, reward, and
HOCBF execution layer. Use the established CPPO-PID CMDP algorithm and require
the existing navigation energy and safety Gates without an explicit route
planner or hand-authored waypoint controller.

### R7 Phases

- [x] R7.1: Convert the completed SAC--PPO failure diagnosis into falsifiable code-level repair requirements.
- [x] R7.2: Select a recognized CMDP method and lock the minimal swap contract: R3 modules plus CPPO-PID, one reward value and one scalar-cost value.
- [x] R7.3: Implement R7 as a new version without modifying or relabeling the failed R6 artifact.
- [x] R7.4: Pass unit, probability-ratio, GAE, checkpoint, and environment regression tests under uv.
- [ ] R7.5: Run a bounded smoke/development preflight and require actual goal successes rather than numerical-health proxies. (196k pilot active)
- [x] R7.5a: Survey collision-safety methods and select CPPO-PID plus robust sampled-data HOCBF-QP as the mature, non-core safety stack.
- [x] R7.5b: Before the next promoted run, enable sampled-data robustness and replace binary shield activation cost by squared action-correction cost.
- [ ] R7.6: Conditionally resume the healthy 196k pilot checkpoint to a total of 500k only if the locked promotion Gate passes; training has no wall-clock limit and remains checkpoint-resumable.
- [ ] R7.7: If and only if the 500k run completes, automatically run the immutable 500-task formal evaluation (100 tasks per distance bucket), with no wall-clock limit and atomic progress every 20 completed tasks.

### R7 Status

**R7.5 PILOT RUNNING IN BACKGROUND.** The sampled-data HOCBF / squared-correction
cost implementation passed 103 relevant tests. A CUDA checkpoint-resume smoke
also completed from 256 to 512 total transitions with model, optimizer, PID, and
random-state restoration. The initial pipeline reached 118,000 transitions in about one hour and was
intentionally interrupted at the user's request to replace its insufficient
70-minute stage limit. It atomically saved
`checkpoint_interrupted_000118000.pt`; no completed rollout was lost. The new
versioned pipeline
`artifacts/r7_sampled_hocbf_pilot_then_500k_seed7002_resume118k_v2` restored that
checkpoint, then the user removed all wall-clock limits. It was intentionally
interrupted at 128,000 and saved another exact checkpoint. The active unbounded
pipeline
`artifacts/r7_sampled_hocbf_pilot_500k_formal500_resume128k_unbounded_v3`
restored 128,000 and reached 130,000 on its first `HEALTHY` update: finite
losses, CUDA actor, zero collisions, and 8.8% rollout intervention. After a
stratified 25-task evaluation, it will resume to exactly
500,000 total transitions only if every locked success, efficiency, safety, and
intervention criterion passes. A successful 500k run automatically starts a
500-task formal evaluation with 100 tasks per distance bucket. Formal progress
is atomically stored every 20 completed tasks and resumes by exact source-task
index. No stage has a wall-clock timeout. Do not poll while it runs; inspect
only when the user asks.

R6 is frozen as failed descriptive evidence: 262,144
transitions, 64 completed training episodes, zero successes, and every episode
ending at the 4,000-step limit. The accidental R3 Phase 6b.8 v5 launch was
stopped after seconds with no seed result and is excluded from evidence.
Literature selection chose ICML 2020 PID-Lagrangian PPO / CPPO-PID over CPO and
FOCOPS because it retains the standard PPO clipped update, uses first-order
optimization, and has both a primary paper and a maintained OmniSafe reference
implementation. R7 will not retain R6's recurrent attention network, three value
heads, zero intervention budget, or mismatched safety distillation.

### R7 Errors Encountered

- A combined regression command initially named a nonexistent
  `tests/test_uav_energy_parallel.py`, so pytest collected no tests. The valid
  suite names were discovered with `rg --files`; the corrected R7/environment/
  projection command then passed 103 tests.

## Goal

Preserve `ORIGINAL_GOAL.txt` as the authoritative research objective and advance
the irreversible return-to-charge study in its declared Gate order, with
stranding--throughput Pareto improvement as the end-to-end criterion.

## Phases

- [x] Phase 1: Freeze the user-selected R3 navigation platform and bind its identity.
- [x] Phase 2: Extract ReturnManager behavior and audit probability semantics.
- [x] Phase 3: Calibrate battery capacity on the frozen R3 checkpoint.
- [x] Phase 4: Reject the right-censored v1 endurance result and implement continuous-workload depletion semantics.
- [x] Phase 5: Obtain and audit the formal 100-run v2 endurance artifact.
- [ ] Phase 6: Run formal Oracle Decision Headroom against SOC and distance baselines if and only if Phase 5 passes.
- [x] Phase 6a: Replace the failed formal runner with a resumable, bounded-runtime implementation and launch it under a verified 9.5-hour hard limit while retaining 320 cycles per point.
- [ ] Phase 6b: Diagnose and resolve the optimized-v2 Oracle rollout-budget failure without weakening the scientific Gate or exceeding the 10-hour wall-clock budget.
- [x] Phase 6b.1: Establish paired causal evidence that the frozen SAC--HOCBF executed loop, rather than a slightly short timeout, causes the recurrent return stalls.
- [x] Phase 6b.2: Replace rollout-budget exceptions by tagged finite-deadline completion requirements and propagate them through ReturnManager, coupling audits, and schema v6.
- [x] Phase 6b.3: Reproduce one historical failure state under v6 and verify that the five-candidate family terminates without a rollout-budget exception.
- [x] Phase 6b.4: Stop and invalidate formal v3 after detecting that task timeout and an under-sized global guard could pre-empt the battery-cycle estimand.
- [x] Phase 6b.5: Implement continuous task rollover, a certified physical-exhaustion guard, exact truncation diagnostics, and pass the related regression suite.
- [x] Phase 6b.6: Re-run the three historical v3 failure seeds as a bounded non-formal preflight and verify all families complete without invalid truncation.
- [x] Phase 6b.7: Diagnose the preflight Oracle-stranding mechanism, replace the invalid branch-maximum decision by a non-nested macro-action viability rule, migrate evidence to schema v7, and verify code/theory regressions.
- [x] Phase 6b.8: Re-run the three mechanism seeds under the corrected viability rule; all three completed without reproducing the worker failure.  The result is mechanism/preflight evidence only, not the fresh 320-seed formal headroom test.
- [ ] Phase 7: Apply the Oracle kill test; continue learned ETG/EIRR experiments only if decision headroom is demonstrated.
- [ ] Phase 8: Establish paired stranding--throughput Pareto evidence and complete the original goal audit.

## Gate Invariants

1. R3 checkpoint, policy, HOCBF/filter, dynamics, and capacity remain frozen.
2. Endurance estimand is `time_to_true_energy_exhaustion_under_continuous_task_workload`.
3. Task failure may roll over to a new task but may not reset position, battery,
   simulation time, or cumulative distance.
4. A valid endurance artifact requires 100/100 true energy-exhaustion endpoints,
   zero censored runs, and the predeclared engineering tolerance.
5. Oracle/Pareto execution cannot start from a running, partial, legacy-v1, or
   failed endurance artifact.
6. Navigation success/path/collision statistics remain descriptive strata, not
   invented theorem assumptions or retroactive R3 selection criteria.
7. For operational horizon H, completed-goal Resource-to-Go is finite only when
   the goal is hit by H; otherwise the deadline-completion requirement is +infinity.
8. A truncated rollout prefix is diagnostic evidence, never a finite substitute
   for a missed completed-goal requirement.
9. Deadline infeasibility does not assert infinite-horizon unreachability.
10. Formal JSON distinguishes finite, infinite, and not-evaluated Oracle values.

## Evidence

- Original objective: `ORIGINAL_GOAL.txt`
- Frozen checkpoint: `artifacts/jseb_navigation_repair_r3r4_uv_seed0_20260828_193904/R3/phase1_navigation/checkpoint_transition_500000.zip`
- Fixed-platform contract: `artifacts/r3_fixed_baseline_energy_chain/navigation_platform_contract.json`
- Calibration: `artifacts/r3_fixed_baseline_energy_chain/battery_calibration.json`
- Invalid v1 stop record: `artifacts/r3_fixed_baseline_energy_chain/STOPPED_BATTERY_ENDURANCE_INVALID.json`
- Failed v2 attempt 1: `artifacts/r3_fixed_baseline_energy_chain_continuous_endurance/`
- Failed-tolerance v2 attempt 2 (100/100 depletion, zero censoring):
  `artifacts/r3_fixed_baseline_energy_chain_continuous_endurance_attempt2/`
- Continuous-workload recalibration attestation:
  `artifacts/r3_fixed_baseline_energy_chain_recalibrated_v2/battery_calibration.json`
- Completed disjoint-seed confirmatory attempt 3 (Gate PASS):
  `artifacts/r3_fixed_baseline_energy_chain_continuous_endurance_attempt3_recalibrated_confirmatory/`
- Terminal failed formal Oracle Decision Headroom run:
  `artifacts/r3_oracle_headroom_stage_b_formal/`
- Terminal bounded failure diagnostic:
  `artifacts/r3_oracle_headroom_stage_b_diagnostic_v1/`
- Completed paired stall-causality ablation:
  `artifacts/r3_return_rollout_stall_ablation_v1/`
- Completed v6 historical-failure preflight:
  `artifacts/r3_oracle_headroom_stage_b_deadline_semantics_preflight_v1/`
- Invalid/stopped formal v3 (partial files retained; never resume or cite):
  `artifacts/r3_oracle_headroom_stage_b_formal_deadline_v3/`
- Claim/evidence deliverable: `docs/EIRR_CLAIM_EVIDENCE_MATRIX.md`

## Errors Encountered

- The README's Windows TeX Live `latexmk.exe` path is absent in this environment,
  and no local `latexmk`/`pdflatex` executable is installed. Resolution: treat the
  newer existing `paper/iclr2027/build/main.pdf` and its log as the build evidence,
  run source-level reference/label checks, and do not claim a fresh compilation.
- The prose checker accepts one file path despite its apparent multi-file/stdin
  interface: the first multi-path call printed usage, and the `-` stdin call was
  treated as a filename. Resolution: concatenate read-only input through
  `/dev/stdin`; the successful check reported one semicolon-density advisory.
- A verification search referenced a nonexistent `paper/iclr2027/result_sheets`
  directory. Resolution: the authoritative pending result cells are in
  `paper/iclr2027/sections/08_results.tex` and related section sources; no file was
  modified by the failed read-only search.
- A first build-log audit used a regular expression containing an invalid escaped
  `\\hbox` token for ripgrep. Resolution: repeat the read-only audit with fixed
  string searches; it confirms eight underfull boxes and no overfull/undefined
  citation/reference signal in the existing log.
- `pdfinfo` is not installed, and a source scan first used the nonexistent plural
  path `paper/iclr2027/appendices`. Resolution: read the existing LaTeX log's
  terminal record (`13 pages`) and use the actual singular
  `paper/iclr2027/appendix/` directory for the final static scan.

- V1 endurance validation mixed true-depletion endpoints with 41 task-limit
  endpoints. Resolution: retain it as invalid censoring evidence and create a
  versioned continuous-workload estimand/contract in a new directory.
- A legacy regression expected battery validation to truncate at the Phase-2
  emergency guard. Resolution: split ordinary Phase-2 truncation from continuous
  validation, which records but ignores that guard until true depletion.
- `LaVineLeo/Paper-novelty-design` could not be installed because the authenticated
  GitHub account lacks repository access. This does not block the experimental Gate.
- A read-only prerequisite audit was first invoked as `uv run ... -c`, which uv
  rejected because the executable name was omitted. Resolution: invoke
  `uv run --no-project --python .venv/bin/python python -c ...`; no experiment
  state was changed.
- `ara-rigor-reviewer` could not produce an official Seal report because this
  repository is not a Level-1-validated ARA and has no required `PAPER.md`,
  `logic/*`, or `trace/exploration_tree.yaml` structure. Resolution: do not
  fabricate `level2_report.json`; use its semantic questions only as an informal
  checklist and keep `ccf-experiment-designer` as the applicable evidence owner.
- The first formal v2 continuous-endurance rerun terminated in parallel worker 4:
  task rollover called `_sample_task_point(self.agent.pos)`, whose reference-point
  validator rejected a physically preserved UAV position inside an obstacle
  clearance region. This is an implementation failure, not an endurance Gate
  verdict. Resolution: retained the failed run, separated reference geometry
  validation from candidate-goal clearance validation, added a regression,
  passed the 146-test critical path, and launched attempt 2 in a new directory.
- Formal v2 attempt 2 completed all 100 true-depletion endpoints with zero
  censoring, but mean endurance was 2262.761 s (relative error +25.709%), outside
  the predeclared +/-20% tolerance. Resolution: retain it as a scientific Gate
  FAIL, repurpose its full 100-run sample for capacity refinement only, compute
  `C_refined = C_source * 1800 / mean(T_source) = 304.9538842289515`, and require
  an independent 100-run confirmation on disjoint seeds 91001--91100. The new
  refinement attestation is hash-bound to both source artifacts and the validator
  fails closed on seed-split drift.
- Disjoint-seed attempt 3 completed 100/100 true-depletion endpoints with zero
  censoring at mean endurance 1668.366 s (relative error -7.313%), inside the
  predeclared +/-20% tolerance. Its `COMPLETED.json` binds the formal validation
  SHA-256 `70914a81ec4a7a5a62c44045a5d3852af58f9f91cc01ddde8b233d6a4ff13f2d`;
  Phase 5 therefore passes and authorizes the Oracle Decision Headroom Gate.
- The direct Oracle Stage-B entry point initially lacked project-root path
  initialization and failed during `--help`, before creating output. Resolution:
  add the same repository-root import bootstrap used by the other scripts; the
  prerequisite audit then passed with zero failures, the P0 probability audit
  passed deterministic point-ETG semantics, and 64 critical tests passed.
- The first formal Oracle run oversubscribed CPU threads (six spawned CUDA
  workers, roughly 73 threads and 1.6 GiB RSS each), underutilized the GPU, and
  lost worker jobs after SIGSEGV while blocked in `Pool.map`. The user imposed a
  hard wall-clock budget below 10 hours. Resolution in progress: the invalid
  process group was terminated intact; preserve its partial audit directory,
  add resumable atomic seed results and worker-failure handling, cap per-worker
  CPU threads, benchmark the unchanged workload, and launch a fresh versioned
  run only if the measured projection is below the budget.
- The first LiDAR fast-path equivalence test compared timing diagnostics for
  exact dataclass equality and failed because optimized and reference runtimes
  necessarily differ. Resolution: exclude only wall-clock/deadline fields while
  retaining exact comparisons for actions, feasibility, constraint counts,
  slacks, solver result, and all scientific diagnostics.
- The Stage-B default-contract regression still expected the superseded six
  workers after the bounded-runtime runner raised the default to twelve
  single-thread workers. Resolution: update the contract assertion and add the
  new one-Torch-thread invariant; sample size and candidate family are unchanged.
- The first shared-Oracle-cache unit fixture left the default charger roughly
  2.4 km away while retaining a 500-step test cap, so the reference rollout
  correctly exceeded its budget before exercising the cache. Resolution: place
  the fixture charger 100 m away as in the existing cache tests; production
  rollout limits and semantics are unchanged. The formal manifest assertion was
  also updated from six to twelve workers.
- The first in-process speed benchmark set Torch inter-op threads itself and
  then invoked the worker initializer, which correctly rejected a second
  `set_num_interop_threads` call after Torch work began. Resolution: let the
  production initializer own thread configuration in the repeated benchmark;
  spawned formal workers are unaffected.
- The first real R3 two-candidate benchmark failed after roughly nine minutes
  because a long Oracle return clone inherited the ordinary navigation
  `max_steps_per_task` and terminated with `task_step_limit` before the declared
  4000-step Oracle guard. Resolution: Oracle clones now set their task limit to
  at least `oracle_max_policy_steps + 1`, so the sole rollout guard is the
  preregistered Oracle budget; add a regression proving a rollout can exceed the
  live task limit without changing the live environment.
- The subsequent SOC=0.20 benchmark exposed the second inherited guard:
  `phase2_episode_limit` terminated the clone as
  `episode_emergency_step_guard`. Resolution: raise both clone-local task and
  episode limits to `oracle_max_policy_steps + 1`; the Oracle's own loop remains
  the only 4000-step failure boundary.
- A focused clone trace showed the hard-coded `phase=2` denotes
  `TD_PRETRAINING`, so the active guard is actually
  `phase1_episode_max_policy_steps`; raising only the phase-2 field was
  insufficient. Resolution: place task, phase-1, and phase-2 clone guards all
  beyond the Oracle loop budget and assert the active
  `episode_policy_step_limit` in regression.
- The optimized-v2 five-point formal run terminated after about 20 minutes with
  `model-based energy rollout exceeded max_policy_steps`. It produced zero
  atomic seed-family results because one candidate exception aborts the whole
  family before persistence. The exception did not record the failing seed,
  method/parameter, rollout role, start/goal/final state, or progress trace.
  Resolution in progress: preserve the failed run, add structured per-rollout
  and per-seed failure artifacts, reproduce under a short bounded diagnostic,
  and classify navigation infeasibility versus a termination-semantics defect
  before changing any Oracle budget or Gate rule.
- The instrumented 12-seed reproduction terminated with 11 structured rollout
  failures and one complete five-candidate seed. Nine failures were return-now
  and two return-after-task; every failure exhausted 4000 steps, and 8/11 made
  nonpositive progress over the last 100 steps. Zero collisions/boundary
  contacts but 85.1--100% HOCBF substep intervention show a persistent executed-
  loop conflict, not evidence that the budget is merely slightly short.
  Resolution in progress: run paired same-state policy-by-HOCBF ablations before
  assigning causality or changing Oracle infeasibility semantics.
- `jq` was unavailable during the read-only diagnostic aggregation. Resolution:
  use the uv-managed Python interpreter for structured JSON aggregation; no
  artifact was modified and no dependency was added for a one-off audit.
- Three attempts to patch the derivation package failed before modifying the
  file because JavaScript interpolation consumed TeX backslashes/backticks.
  Resolution: use a raw string payload without embedded backticks; the final
  patch applied cleanly and no partial document corruption occurred.
- The first v6 regression pass found two stale fixtures: the coupling fixture
  omitted the new deadline tags and a result-writer assertion still expected
  schema v5. Resolution: migrate both fixtures; 56 schema/decision tests pass.
- Two legacy Oracle-shadow tests returned finite prediction objects predating
  the deadline tag. Resolution: finite legacy deterministic shadows default to
  deadline-feasible while new estimators must provide explicit tags; the full
  160-test energy/decision selection passes.
- The v6 formal validator initially conflated an intentionally stopped Oracle
  shadow with a finite exact requirement. Resolution: use a three-state tag
  (`true` infinite, `false` finite, `null` not evaluated), require exact fields
  only while `oracle_shadow_active=true`, and add active-infinite/inactive tests.
- A read-only result audit passed `parse_constant` to `Path.read_text` instead of
  `json.loads`, causing a local TypeError before parsing. Resolution: pass the
  hook to `json.loads`; the strict parse then passed and found no non-finite JSON.
- Formal v3 exposed a mission-level termination defect: a 4000-step TASK timeout
  truncated the whole battery sortie, while the inherited 20000-step Phase-2
  guard was below the 25413-step worst-case physical exhaustion bound implied by
  capacity 304.953884 and the certified 0.012 energy/step minimum. Resolution:
  stop v3, forbid resume/citation, roll a timed-out TASK goal to the next keyed
  task without resetting physical state, set the global guard to
  `max(source, cycles*ceil(capacity/min_step_energy)+1)`, preserve fail-closed
  behavior, and record the exact truncation reason. The related 163-test suite
  passes.
- The corrected task-rollover preflight completed all three historical failure
  seeds and all fifteen candidate cycles in 1752.9 seconds with ten task
  rollovers and zero emergency-guard truncations. It therefore validates the
  termination repair, but is non-formal and selected on prior failure.
- The same preflight exposed a distinct ReturnManager logic defect. For Oracle
  seeds 110012 and 110013, the last pre-commit states had direct return
  deadline-infeasible while task--then--return remained deadline-feasible with
  large positive resource margin. The old immediate-infeasibility-first rule
  nevertheless committed and both cycles stranded. Resolution: formalize the
  non-nested hybrid feasible sets, continue on a certified task--then--return
  route even when direct return is uncertified, use direct return only as a
  post-commit certificate, and bump the evidence contract to v7. The corrected
  code/schema/environment suite passes 165 tests and the corrected EIRR
  finite-law suite passes 7 tests.

## Status

**ORIGINAL GOAL PRESERVED; PHASE 6b.8 PAUSED BY USER AND SUPERSEDED AS THE
CURRENT PRIORITY BY R7.** The first viability-order
preflight directory is terminal but incomplete: seed 110004 finished, while a
native Python worker segfault caused seeds 110012 and 110013 to remain absent.
This is infrastructure failure, not Oracle-headroom evidence. The diagnostic
entry point now permits an explicit one-worker execution with atomic seed-family
retries. A fresh corrected three-seed mechanism preflight was started at
`artifacts/r3_oracle_headroom_stage_b_viability_order_preflight_v4_threadcapped_single_worker`
with a three-hour hard limit. Its worker health artifact proves one
OS/Torch/inter-op thread and all six native thread environments equal one. At
the user's 2026-09-01 18:00 Asia/Shanghai stop request it had completed zero of
three atomic seed families, so the process group was interrupted and a
`STOPPED_BY_USER.json` marker was written. Restart Phase 6b.8 from the same three
seeds when requested. No fresh formal 320-seed run is authorized until this
preflight terminates successfully. R3 remains the frozen conditional baseline;
the R6 navigation experiment is separate descriptive development evidence.

## R6 End-to-End Learned Navigation Repair (2026-09-01)

### Goal

Create a new versioned navigation branch in which the neural policy itself learns
goal-directed collision avoidance and recovery, without A*/Theta*, waypoint, PD,
or hand-coded deadlock switching. Keep HOCBF only as a training-time safety teacher
and emergency execution projection, then launch a bounded formal training run only
after short preflight evidence shows healthy learning and artifact production.

### Phases

- [x] R6.1: Audit the existing observation/action/environment/training interfaces and historical R3 failure replay hooks.
- [x] R6.2: Specify the recurrent obstacle-set actor-critic and constrained PPO/distillation objective with explicit evidence semantics.
- [x] R6.3: Implement the versioned R6 model, rollout/trainer, configuration, checkpointing, and metrics without modifying R3 behavior.
- [x] R6.4: Add deterministic unit tests for shapes, recurrence boundaries, executed-action likelihood semantics, dual updates, and intervention loss.
- [x] R6.5: Run a small uv-managed smoke/preflight and reject launch on NaN, missing gradients, invalid action bounds, or missing progress/intervention metrics.
- [x] R6.6: Launch one bounded background experiment (hard wall-clock limit below 10 hours), verify that it is alive and producing healthy logs, then stop active goal work pending the user's request.

### R6 Invariants

1. R3 code paths, checkpoint, capacity calibration, and evidence artifacts are immutable baselines.
2. No explicit route planner, waypoint state machine, PD controller, or heuristic deadlock branch may supply navigation actions.
3. The actor consumes observations and goal/resource state and directly emits the nominal continuous action.
4. If a safety filter changes an action, PPO likelihoods are defined on the sampled nominal action while transitions use the executed action; the intervention is exposed to the recurrent state and optimized through an explicit distillation/constraint term.
5. A successful navigation claim requires decreasing safety-filter intervention in addition to task success; high success under near-total intervention is not learned navigation.
6. The formal launch must have an external timeout below 10 hours, versioned outputs, atomic checkpoints, and no reuse of an existing artifact directory.

### R6 Errors Encountered

- The first PPO unit test retained an autograd graph through its synthetic
  rollout hidden-state fixture, so a second recurrent minibatch attempted to
  backpropagate through a freed collection graph. Production collection already
  serializes detached hidden states. Resolution: make the optimizer boundary
  explicit by detaching every sequence's initial hidden state inside
  `ppo_update`, preventing either tests or future collectors from leaking a
  behavior-policy graph into PPO optimization.
- The first CUDA smoke showed one spawned evaluation worker above 200% CPU even
  though `_worker_main` set OMP/MKL limits. Those variables were assigned after
  NumPy/SciPy import in spawned interpreter initialization and omitted OpenBLAS.
  Resolution: set OMP, MKL, OpenBLAS, and NumExpr limits at the R6 module top
  before numerical-library import, and mirror the complete set in the worker.
- The repaired-thread smoke remained finite, but early squared value loss grew
  from 3.53 to 60.28 as progress returns changed scale, with unclipped gradient
  norms reaching 56.36. Gradient clipping prevented instability, but a critic-
  dominated shared encoder would make a long run low-value. Resolution: replace
  MSE by an equally weighted per-head Huber loss and retain global norm clipping.

### R6 Status

**INTERRUPTED BY USER AFTER TRAINING, DURING FINAL EVALUATION.** Implementation
and 91 relevant regressions pass. R6 completed the exact 262,144-transition
training budget and persisted the final checkpoint, but its 50-task final
evaluation had not completed after about 3 h 20 min and produced no atomic final
evaluation artifact. At the user's 2026-09-01 18:00 Asia/Shanghai request, the
dedicated process group was interrupted. `INTERRUPTED.json` and
`checkpoint_interrupted_000262144.pt` preserve the terminal training state. Do
not retrain; resume by loading this checkpoint in a bounded evaluation-only path.

## R6 SAC-vs-PPO Failure Diagnosis (2026-09-01)

### Goal

Explain with code- and artifact-level evidence why frozen R3 SAC learned useful
navigation while the new recurrent constrained PPO reached zero successful
training episodes, and identify which missing PPO mechanisms or implementation
choices must be tested before any retraining.

### Phases

- [x] D1: Align SAC and PPO evidence at comparable environment-transition budgets.
- [x] D2: Audit PPO probability, recurrence, GAE/value, constraint, safety-distillation, and termination semantics against the executed environment loop.
- [x] D3: Quantify optimizer/sample reuse, exploration, representation, reward, and horizon differences.
- [x] D4: Rank causal hypotheses by evidence and specify minimal falsifying experiments without changing code.
- [x] D5: Write `docs/R6_SAC_PPO_FAILURE_ANALYSIS.md` and pause for user review.

### Diagnostic Invariants

1. Numerical stability is not navigation success.
2. A 262,144-transition PPO development run cannot be called a fair winner/loser
   comparison against 500,000-transition SAC without an aligned-budget trace.
3. Training episodes are correlated trajectory observations from one seed; they
   are descriptive failure evidence, not independent seeds for significance.
4. A live final evaluation is read-only during diagnosis unless the user
   explicitly requests that it be interrupted.

### Diagnostic Status

**DIAGNOSIS COMPLETE; PAUSED FOR USER REVIEW.** The aligned trace rules out raw
transition budget as the primary cause. The report identifies optimizer/gradient
suppression, a degenerate zero-budget intervention dual, stochastic-teacher
mismatch, deterministic-mean collapse, spatial representation loss, and short
credit horizon as the ranked causes. No training code or active evaluation
process was modified.

### Phase 6b.8 Infrastructure Finding

- `r3_oracle_headroom_stage_b_viability_order_preflight_v2` is terminal
  `REPRODUCED_FAILURE`, with only seed 110004 persisted and seeds 110012/110013
  missing. No per-seed Python failure file exists.
- The kernel recorded `python3` SIGSEGV at 2026-09-01 11:24:11 while the
  three-worker diagnostic was active. The diagnostic wrapper had forced
  `formal_seed_retries=0`, so atomic resume could not recover the missing seed
  families.
- The preflight CLI now accepts `--evaluation-num-envs` and
  `--formal-seed-retries`. Phase 6b.8 will use one worker and two retries without
  changing seeds, candidate points, resource semantics, or Gate thresholds.
- The combined ReturnManager/environment, finite-deadline Oracle, probability
  semantics, v7 record, and Stage-B inference selection passes under uv (`166
  passed`, 187 third-party deprecation warnings). The new worker-isolation test
  is included in that count.
- The first single-worker launch was stopped before producing a seed result
  because the child still had 47 OS threads: its thread limits were applied in
  the ProcessPool initializer, after NumPy/Torch import. It is marked
  `STOPPED_INVALID_WORKER_PARALLELISM` and cannot be resumed or cited.
- Stage-B now sets OMP/MKL/OpenBLAS/NumExpr/BLIS/VecLib limits before numerical
  imports and writes a worker-health artifact after policy initialization. The
  v4 worker reports exactly one OS thread, one Torch thread, and one inter-op
  thread; Phase 6b.8 therefore ran with the intended isolation contract. It was
  stopped on the user's request before any seed family completed and must be
  restarted from the same three-seed preflight.

## 2026-09-02 R7 Accelerated Restart at 196k

- [x] Let old v3 reach its next exact atomic endpoint and save
  `checkpoint_transition_000196000.pt` before terminating its process group.
- [x] Fully deserialize and validate the model, actor/critic optimizers, PID,
  NumPy/Torch CPU/CUDA RNG, update 98, and seed cursor 720133. Checkpoint
  SHA-256: `bca3fe9de29462b197de5492d1dc09969cabf86bb59040881c4689d5edcc5c97`.
- [x] Add an explicit `--promoted-resume-checkpoint` recovery path so an exact
  completed-pilot checkpoint continues to 500k without replaying transitions.
- [x] Restore SIGINT handling for detached training and hash the HOCBF Python
  source, native C source, and compiled shared library in the run manifest.
- [x] Launch v5 without timeouts and verify first-update health, source hashes,
  all eight native-library mappings, and the 200k periodic checkpoint.

**ACTIVE, NOT MONITORED.** Detached v5 has PGID 120325 and writes to
`artifacts/r7_sampled_hocbf_promoted500k_formal500_accelerated_resume196k_v5`.
It resumes from 196k to 500k and then automatically starts the resumable formal
500-task evaluation. First-update health is `HEALTHY`; observed collection
throughput was 643--960 transitions/s versus about 23--28 before restart. Per
user instruction, do not poll until requested.

## 2026-09-02 Accelerated R3 SAC Reproduction

### Goal

Reproduce the original R3 SAC navigation result from random initialization with
the numerically equivalent accelerated HOCBF runtime, then automatically run the
same immutable 500-task formal navigation evaluation.

### Phases

- [x] A1: Recover and hash-lock the original R3 training/evaluation contract.
- [x] A2: Audit whether the existing R3 launcher can activate the accelerated
  HOCBF path without changing SAC, reward, task, seed, or budget semantics.
- [x] A3: Add only the versioned orchestration/provenance needed for a fresh
  accelerated 500k train followed by the immutable formal 500-task evaluation.
- [x] A4: Run focused regression and short CUDA health checks.
- [x] A5: Launch the versioned detached run, verify first log/health and
  native-library loading once, then stop monitoring until requested.

### Invariants

1. Training starts from random initialization; no R3/R7 checkpoint is loaded.
2. R3 SAC architecture, hyperparameters, seed, task generator, reward, 500k
   transition budget, and evaluation task source remain unchanged.
3. The only intended simulator change is the already differential-tested
   array/native implementation of the same sampled-data HOCBF equations.
4. Formal evaluation contains exactly 500 immutable source tasks, 100 in each
   distance bucket, and is resumable by source index.
5. Output paths are fresh and source/checkpoint/task hashes are recorded.

### Status

**ACTIVE, NOT MONITORED.** The 13-field contract audit and 110 focused tests
pass. The detached fresh run has PGID 204721, trainer PID 204724, and output
`artifacts/r3_sac_accelerated_reproduction_seed0_500k_formal500_20260902_v1`.
All eight workers map `_qp_native.so`. Its first scheduled log at 8k is finite,
uses eight environments, has zero collision/boundary contacts, and reports
16.7875% HOCBF intervention. It will train to 500k and then automatically run
the immutable formal 500-task evaluation. Do not poll until the user asks.

### Errors Encountered

- Invoking the standalone script through `uv --no-project` omitted the project
  root from Python imports and raised `ModuleNotFoundError: envs`. Resolution:
  all smoke/formal commands explicitly set `PYTHONPATH=/home/zjl/mappo` and use
  the project `.venv` through uv.

## 2026-09-04 scene-grouped finite-sample energy-safety calibration

### Finding

- [x] Complete the fresh 75-scene certified-meet confirmation without changing
  the frozen R3 actor or critics.
- [x] Diagnose the failed 5% Gate at the scene level.
- [x] Replace empirical thresholding with a preregistered RCPS/HBB upper-risk
  calibration rule.
- [x] Store exact simulator pre-action position and velocity in all new nominal
  sources; preserve bounded observation decoding only as a legacy fallback.
- [x] Validate exact-state fork reconstruction and the risk-bound code by tests
  and a five-scene end-to-end smoke.
- [ ] Collect 450 entirely new calibration scenes and freeze non-vacuous meet
  and geometry thresholds only if every calibration Gate passes.
- [ ] If calibration promotes, run the chained 150-scene untouched confirmation.

### Locked interpretation

The first fresh meet Gate failed at 6.486% empirical scene false-safe risk
versus the locked 5% target, despite a 6.096-point coverage gain and zero meet
dominance/monotonicity violations. A post-hoc meet threshold near 0.815 would
hit 5% empirically, but is not evidence. With only 75 independent scenes, the
95% HBB upper bound is 6.060% even for an all-zero loss vector, so the original
Gate could not produce a finite-sample 5% certificate in principle.

Protocol `docs/SCENE_GROUPED_RCPS_MEET_PROTOCOL_V1.md` therefore freezes 450
new calibration scenes (task/world seeds 510001/520001), a 5% risk target and
5% failure probability, a fixed 2,001-point threshold grid, and a further 150
untouched confirmation scenes (seeds 610001/620001). The inspected 75 scenes
are development-only and will not be reused.

**ACTIVE, NOT MONITORED.** At the user's request the eight-worker chain (old
PGID 219370) was stopped after 15,952/18,000 atomic calibration branches and
replaced by the compute-equivalent 12-worker resume chain, PGID 301762. No
completed branch was deleted or recomputed; at most the eight in-flight,
uncommitted branches were restarted. The resumed pool has 12 active workers,
no failure sentinel, no swap use, and preserves every model, seed, state,
candidate-action, horizon, and Gate parameter. It was then recoverably paused
at 16,692/18,000 branches (92.73%); zero workers remain and 1,308 branches will
be reconstructed from the frozen anchors on resume. Integrity hashes matched
before restart. The chain resumed from exactly 16,692 saved branches with 12
workers under PGID 2915; the pause marker was cleared, no failure sentinel
exists, and no swap is in use. Log:
`artifacts/rcps_meet_confirmation_chain_12w_resume_20260904_v2.log`. Per user
instruction, do not poll again until requested.

The 450-scene RCPS calibration subsequently passed, but its shell promotion
guard incorrectly passed `-c` to uv itself and stopped before confirmation.
Both full/resume guards now invoke `.venv/bin/python -c` correctly. A
confirmation-only chain hash-validated the frozen calibration and launched the
150 untouched scenes under PGID 13216 with 12 workers; it does not revisit the
18,000 calibration branches. Initial health: 5/150 nominal rollouts committed,
12 active, 133 pending, no failure sentinel, and zero swap. Log:
`artifacts/rcps_meet_confirmation_only_12w_20260904_v1.log`. Do not poll until
requested.

The first confirmation fork stopped at 3,069/6,000 files on an exact-hash
mismatch. Root cause was non-idempotent float32 velocity canonicalization: a
radial projection could round a few ulps above 20 m/s and be changed on the
next call. The sanitizer now performs float64 radial scaling plus a minimal
representable inward correction and is bitwise idempotent. A second issue was
resume compatibility for a partially completed legacy anchor: workers now
validate, but never recanonicalize, a registered hash-bound snapshot.

The real failing vector, 721 boundary directions, and registered-snapshot
worker behavior are covered; the focused suite passes 21/21. Both actual
failure anchors reproduce with exact 2055-dimensional equality and zero max
error. Resume PGID 29193 is active with 12 workers; it retained all 3,069
files, advanced to 3,071 at the health check, and has no failure sentinel or
swap use. Log: `artifacts/rcps_meet_confirmation_fork_12w_resume_20260905_v2.log`.
Do not poll until requested.

## 2026-09-05 bounded RCPS closed-loop pilot

### Completed prerequisite

- [x] Independent 150-scene confirmation completed and passed all seven frozen
  Gates. Certified Meet kept scene false-safe risk at 2.0133% with 50.1456%
  coverage, versus geometry's 1.9465% and 46.4089%; paired coverage gain 95% CI
  `[2.9722, 4.4723]` points. This supports a bounded pilot, not a worst-case or
  adaptive closed-loop certificate.
- [x] Reproduce the 26-dimensional online geometry/action feature contract
  against all 6,000 confirmation branches: exact array equality, zero maximum
  absolute error.
- [x] Freeze the 75-case/three-arm pilot protocol before observing outcomes.
- [x] Add and test the resumable closed-loop runner. The five-bucket smoke
  completed 15/15 censored records and reopened with `--resume` without
  recomputation.
- [x] Start the formal pilot without modifying the R3 actor or HOCBF.

### Status

**ACTIVE, NOT MONITORED.** The formal paired pilot runs under PGID 111248 at
`artifacts/rcps_closed_loop_pilot_75cases_20260905_v1`, with 12 simulator
workers and CUDA actor/critic inference. The one-time health check reached
11/225 atomically completed rollouts, with 12 active, 202 pending, no failure
sentinel, zero swap, and about 8 GiB available RAM. It compares ungated,
geometry Gate, and Certified Meet Gate on 75 new cases with five balanced
energy budgets. Do not poll again until the user asks.

### Terminal judgment

**COMPLETE; DO NOT PROMOTE.** All 225 records completed in 625.5 s and pass
provenance/pairing validation. Meet achieved 5/75 task completions, 71/75 safe
returns, and 2/75 unsafe terminals; geometry achieved 3/75, 72/75, and 1/75;
ungated achieved 72/75, 64/75, and 10/75. Meet fails the frozen
unsafe-not-worse-than-geometry Gate by one case. More importantly, both Gates
collapse task utility: Meet returns immediately in 56/75 and eventually in
70/75 cases, with no completion at distances >=1500 m.

The mechanism is now identified: the offline target certifies the queried task
action plus frozen task continuation and later return, whereas rejection was
deployed as if it certified immediate return. Cases 23, 42, 57, and 72 disprove
that implication. Repeated pointwise thresholding also compounds into a nearly
always-stop trajectory rule. Next work must learn separate continuation and
return-now viability functions and use trajectory-level stopping control. The
strict analysis bundle is under the pilot artifact's `analysis-output/`.

## 2026-09-05 post-pilot dual-viability correction

### Research question

Can an energy-aware UAV preserve task utility by learning the probability that
the **executed next action leaves a safe-return option available**, rather than
using task-completion-then-return feasibility as a proxy for immediate-return
or recursive safety?

### Locked distinctions

1. `completion_viability`: candidate action, continued task execution, then
   return. This is the label used by the failed RCPS pilot.
2. `return_now_viability`: from the current state, immediately execute the
   frozen R3 policy toward the charger while preserving current velocity.
3. `option_preservation`: execute one candidate through the unchanged HOCBF,
   then immediately execute frozen R3 toward the charger while preserving the
   resulting velocity.

Only item 3 has the correct action-conditioned semantics for deciding whether a
task action preserves the option to fall back on item 2. None of the three is
silently treated as a physical certificate.

### Phases

- [x] P1: Audit the failed 75-case pilot and identify the label/deployment
  implication error.
- [x] P2: Cross-check recovery-policy, reach-avoid, feasible-set,
  consumption-MDP, and finite-sample risk-control literature.
- [x] P3: Freeze the corrected invariant and research question before new
  outcomes are observed.
- [x] P4: Implement a resumable paired counterfactual collector over the exact
  150-scene confirmation anchors.
- [x] P5: Pass focused unit tests and a small end-to-end smoke.
- [x] P6: Launch the development-only 3,410-rollout diagnostic (310
  return-now plus 3,100 one-action-then-return branches) and perform one health
  check. Do not monitor afterward.
- [x] P7: After completion, quantify label discordance, action sensitivity,
  recoverability coverage, and budget-dependent failure modes before deciding
  whether fresh training/calibration data are justified.

### Non-negotiable evidence rules

- The inspected 150-scene confirmation set is mechanism-development data only;
  it cannot become new confirmation evidence.
- Candidate actions pass through the same HOCBF execution interface used at
  deployment; no raw eight-step hold is used.
- Retargeting preserves current velocity. Resetting velocity to zero would
  answer a different and easier problem.
- RCPS/CRC calibration of independent examples is not described as an
  adaptive closed-loop guarantee. A fixed learned controller must later be
  evaluated at the whole-trajectory level.
- No large training run starts from this phase. The diagnostic must first show
  that the corrected labels are nondegenerate and distinct from the old label.

### Status

**DIAGNOSTIC COMPLETE — REPAIR RECOVERY FIRST.** All 3,410 rollouts completed
without censoring in 5,797.6 s. Immediate R3 return was structurally safe for
275/310 anchors (88.71%); its 35 failures include 17 step-limit failures, 16
collisions, and two boundary contacts. One-action-then-return was safe for
2,766/3,100 branches (89.23%), but only 21/310 anchors had mixed structural
outcomes across the ten candidates and at most 22/310 were mixed at any energy
budget. Candidate structural rates do not differ (Cochran Q=5.633, p=.776),
while candidate energy differences have negligible effect size (Kendall
W=.017). The old completion label is materially different at intermediate
budgets because it consumes 13.18% of capacity on average versus 7.93% for the
corrected option arm. Decision: do not fit a standalone binary action critic;
repair the R3 recovery baseline, then learn state return viability plus a
successor model and separate continuous energy and structural-failure heads.
Strict analysis is under the experiment's `analysis-output/`. The previous
bounded RCPS gate remains `DO_NOT_PROMOTE`.

## 2026-09-05 R3 minimal collision-safe RL literature and theory phase

### Goal

Select one mathematically coherent collision-safety mechanism that can modify
the frozen R3 SAC baseline without turning the system into a stack of unrelated
modules. The chosen mechanism must address the observed R3 recovery failures
and remain compatible with later energy-return viability work.

### Constraints

- Keep R3 SAC, structured LiDAR observation, continuous three-dimensional
  action, environment collision semantics, and checkpoint compatibility unless
  evidence shows one of them is the root blocker.
- Add at most one core learned safety object. Existing HOCBF may be retained as
  an execution interface or replaced, but it is not counted as a new research
  contribution and must not hide unsafe learned behavior.
- Do not introduce MPC, a third critic, a hand-coded behavior tree, waypoint
  rules, or simultaneous energy and collision redesign in the first test.
- Distinguish training-time safety improvement from deployment-time formal
  guarantees. No empirical critic is called an absolute certificate.

### Phases

- [x] L1: Audit the exact R3 reward, transition, collision, HOCBF, recovery, and
  checkpoint contract.
- [x] L2: Run a standard primary-source literature search spanning constrained
  policy optimization, Lagrangian actor-critic, recovery/backup policies,
  reach-avoid dynamic programming, safety-index/CBF learning, and shielding.
- [x] L3: Screen each family against the five project constraints and identify
  the closest prior art and novelty ceiling.
- [x] L4: Derive the smallest viable mathematical intervention and its failure
  conditions in `DERIVATION_PACKAGE.md`.
- [x] L5: Write the canonical literature folder and minimal R3 design document;
  challenge the design once for prior-art overlap and once for evidence risk.

### Status

**COMPLETE — DESIGN ONLY.** No training was started. The selected candidate is
R3-RACT: a zero-new-network, constraint-terminated reach-avoid replacement for
the existing R3 Bellman/bridge objective. An independent GPT-6 theory review
forced explicit handling of signed rewards, entropy, time horizon, projection
semantics, and experiment attribution. Canonical output:
'docs/R3_MINIMAL_COLLISION_SAFE_RL_DESIGN.md'.

## 2026-09-05 R3 collision interface implementation phase

- [x] Check actual normalized-to-physical action mapping and emergency output;
  replace nominal-coordinate distance with per-substep physical correction.
- [x] Correct terminal discount indexing, distinguish physical failure from
  recoverable braking, and calibrate hazard over full successful trajectories.
- [x] Implement/test the scalar reach-avoid and shaping primitives independently
  of the old environment; no actor/critic architecture changes made.
- [x] Implement paired ordinary/robust frozen-R3 audit with first-contact
  episode semantics, identical-scene hashes and atomic completed-episode resume.
- [x] Run short end-to-end CUDA/multiprocess smoke, then launch 50 pairs in the
  background. Startup checked: CUDA, 8 workers, 49/100 records at 59.4 seconds.
- [x] User-requested review of completed audit; decide whether robust interface
  is a viable common base. Do not silently promote on a development gate.
- [x] Implement matched SAC-derived reach-avoid trainer and test replay,
  timeout bootstrap, critic reset and checkpoint/replay resume if gate permits.
- [ ] Run matched kappa=0 / kappa>0 short training, then reassess before scaling.

Status: **BACKGROUND INTERFACE AUDIT STARTED; NO NEW POLICY TRAINING YET.**
Directory: `artifacts/r3_collision_interface_audit_50pairs_20260905_v1/`.
Process launched at 12:59:03 Asia/Shanghai; uv PID 67747, Python PID 67750.
Do not monitor to completion; wait for the user to request results. The scalar
objective tests do not validate neural learning. Complete 2055-dimensional
observations and the original simulator/time limits are preserved.

## 2026-09-05 matched short-training implementation

User authorized the next step after audit completion: ordinary/robust both
49/50 safe goals, no contacts; robust interface passed development screening.

- [x] Quantify full-trajectory and per-step hazard attenuation; freeze a
  defensible development-only scale before training, with change rationale.
- [x] Build a SAC-derived two-Q trainer with physical-contact terminal semantics,
  correct terminal-observation bootstrap, actor warm start and critic warmup.
- [x] Verify Bellman algebra, replay continuation, actor/critic initialization,
  finite updates, and checkpoint/replay resume with focused tests and smoke.
- [x] Launch sequential matched 50k-transition arms, save every 10k plus replay;
  evaluate paired held-out development cases after training, not formal 500-test.
- [x] Confirm startup health, document exact resume command and stop monitoring.

Status: **BACKGROUND TRAINING HEALTHY — waiting for user, not monitoring.**
Output: `artifacts/r3_reach_avoid_pair_50k_each_20260905_v1/`.
Launched 13:21:47 Asia/Shanghai, uv PID83096 / Python PID83099. Startup check
at 6000 transitions: 3576 MC critic updates, eight complete episodes, finite
critic loss7.08e-5; actor correctly frozen until10k. Resume command is recorded
in the experiment README. No results/efficacy claims from this startup check.

Implementation evidence: 8 focused tests pass, including MC telescoping,
timeout terminal-observation bootstrap and model/optimizer/replay resume.
A real-environment smoke completed both 5k arms and 30 evaluations. Both arms
had 3274 MC initialization updates and 1000 actor updates; actor and initial
critic hashes matched exactly across arms. All update metrics were finite.
Batch-256 CUDA benchmark: 0.03617 seconds/update, about 237 MiB peak allocated
PyTorch memory (not total GPU reservation). Estimated full job roughly 1–2 h,
not a deadline guarantee. Do not add a hard timeout or scale to 500k.

Recorded implementation correction: the v1 smoke RESULT's `smoke_only` field
was false because it used full episode limits; its directory/budget identify
it as smoke. Added an explicit smoke flag in runner v2 and annotated the old
artifact without modifying raw results. New 50k evaluation uses 710001/720001,
not smoke's 610001/620001.

## 2026-09-05 failure-driven safe/CMDP literature redesign

The 50k pair finished: shield-on safe goals hard7/50, continuous1/50,
frozen R3 47/50. User explicitly permits from-scratch learning and no longer
requires R3 as architecture/initialization, only as an empirical reference.

- [x] Diagnose common failure signatures from saved replay/checkpoints/logs;
  distinguish numerical execution from value-learning validity.
- [x] Search 15–30 primary candidates, verify 8–15 relevant top-conference
  papers, and inspect closest official implementations.
- [x] Select a minimal coherent baseline (including from-scratch options),
  state constraint semantics, theoretical limitations and attribution.
- [x] Write evidence packet and next implementation protocol. Do not launch
  another long run merely because the previous code executes without errors.

Status: **RESEARCH / DIAGNOSTIC COMPLETE — no new training launched.**

Evidence packet: literature-search-20260905-safe-rl-from-scratch/ (17 screened,
14 retained; variable reading depth explicitly recorded). Design:
docs/SAFE_NAVIGATION_FROM_SCRATCH_REDESIGN.md. Exact first-contact finite-task
cost identities added to DERIVATION_PACKAGE.md, without neural-safety claims.

Next implementation: isolated from-scratch finite-task FOCOPS plus matched PPO;
physical contact cost, utility retained, explicit terminal/rollout semantics,
shield-off simulation evaluation. R3 is reference only. First implement/test
the contract and measure runtime; no automatic 500k or formal500 launch.

## 2026-09-05 forward-citation critical audit

User asks to trace papers citing the shortlisted safe-RL methods and critically
assess actual usage. Research-only turn; no training/code changes authorized
by this literature request alone.

- [x] Verify forward citation edges, separating background citations from
  implementation reuse, experimental comparison and concrete criticism.
- [x] Read downstream evidence, including negative results, benchmark budgets,
  cost semantics and implementation variants; examine recent candidates fairly.
- [x] Write a reusable critical evidence ledger and update the recommendation
  only where source-supported findings justify a change.

Status: **FORWARD-CITATION AUDIT COMPLETE — no training started.**

Packet: literature-search-20260905-safe-rl-forward-citations/. Fifteen candidate
records screened, ten evidence records retained (including a tool). Verified
background/evaluated/reuse/criticism distinctions; independent follow-up
evidence remains unconfirmed for SafeMPO/C-TRPO themselves and several
secondary candidates, explicitly recorded rather than filled with assumptions.

Recommendation supersedes the single-FOCOPS leaning above: standard
PPO-Lagrangian and FOCOPS under identical event-cost contract, plain PPO as
learning sanity reference. SafeMPO downgraded to theory due to its own Table1
budget overruns; SAC remains a viable fallback. Design doc updated, no source
code or model changes. Access errors/version conflicts logged in search-notes.

## 2026-09-05 existing asset reuse audit

User requests continued research with critical reuse of structured LiDAR and
other existing results. Scope: inspect implementation and evidence, not launch
training or silently inherit old safety assumptions.

- [x] Map encoders, simulation acceleration, evaluation and checkpoint assets.
- [x] Check interfaces, historical evidence and relevant existing tests.
- [x] Record keep/adapt/defer decisions and update minimal design.

Status: ASSET AUDIT COMPLETE. No training launched or production code changed.

Deliverable: docs/SAFE_NAVIGATION_ASSET_REUSE_AUDIT.md. Important correction:
structured LiDAR global mean/max readout removes absolute azimuth, despite
equivariant intermediate convolutions. Frozen R3 physical-ray four-bearing
probe: raw observations differ, final embeddings exactly equal, nominal
actions differ only <=5.97e-8. This is not proof of the sole cause of failures.
Reuse channels/convolutions/simulation, change readout candidate before matched
safe-RL comparison. Ordered 2x16 readout distinguishes probe but is untrained
and adds parameters; do not claim learning success. 61 existing tests passed.
Historical energy confirmation failed promotion; preserve data/metrics, not
blindly transfer models or policy-dependent calibration.

## 2026-09-05 directional readout implementation and preflight

User approves proceeding. Preserve old code/checkpoints; implement isolated
directional extractor and tests, verify controlled learning and runtime before
navigation rollout. Use uv; no automatic long safe-RL/energy experiment stack.

- [x] Implement independently named ordered readout with observation contract.
- [x] Test bearing information, goal sensitivity, gradients and serialization.
- [x] Run bounded, parameter-accounted learnability/runtime diagnostic.
- [x] If preflight supports proceeding, implement first-contact plain-PPO
  navigation pilot and check resumable execution; otherwise report evidence.
- [x] Record results and next boundary in a durable implementation report.

Status: IMPLEMENTING DIRECTIONAL PREFLIGHT.

Preflight correction: checkpoint test initially compared automatic CUDA model
to CPU-loaded model with bitwise equality (difference ~1e-8). Fixed the test
to use CPU for both sides; this tests serialization, not cross-device rounding.
No change to model math or relaxed serialization assertion.

Second test correction: the original physical repair places the UAV exactly
on an inflated obstacle surface; the spawn sampler requires strictly greater
clearance. Test now checks nonpenetration plus zero velocity, rather than
mistaking the sampler predicate for the repair contract. Physics unchanged.
Pilot review corrections: clear SB3's queued startup seed so wrapper keyed
tasks/cursors govern resume; bind evaluation records to checkpoint transition
count so an extended training run cannot reuse an older model's evaluation.

Full-geometry preflight: 8192 transitions completed in ~13 s before final
formatting; second4096 batch5.807s (collection2.928/update2.878), finite KL/loss.
Smoke resumed128→256 with optimizer update count4→8 and episode cursors4→8.
Style audit found24 import/dict-literal findings in new files, to be fixed
mechanically before final source freeze and a new versioned pilot. Save the
initial model too, enabling later unchanged-seed initial-policy evaluation.

Final quality checks:153 regression tests passed,12 new tests rerun after type
adaptation; Ruff clean; Pyright18 framework-contract typing findings resolved
with typed base references/Box checks/array narrowing, then0 errors. No old
training code or weights modified. Ordered bearing test MSE~.000258 vs matched
globalpool~.500098; not navigation efficacy evidence. Implementation report:
docs/DIRECTIONAL_PPO_PILOT.md.

Detached final-source pilot v3 launched: target262144 from scratch, eight
environments,24obstacles,shield-off, then50 independent development episodes.
Awaiting only initial health verification, not experiment outcomes.

Status: IMPLEMENTATION VERIFIED / PILOT RUNNING — hand off without monitoring.
Initial v3 health check: trainerPID157454, processgroup157451,24576transitions,
checkpoint16384saved, noerror.json, finiteupdatevalues, KL.01554. Model1211111
parameters. Timing~5.94s/4096 afterstartup. Do not interpret this healthcheck as
navigation/safety success. User will request result analysis after completion.

## 2026-09-05 next step: minimal matched PPO-Lagrangian

Completed plain PPO262144/50dev:29safe_goal,20contact,1timeout, path1.3515;
total363.18s. User approves next research step. Do not change old artifacts.

- [x] Audit available contact evidence and implement reproducible event detail.
- [x] Verify reference Lagrangian objective and cost/terminal/dual statistics.
- [x] Implement isolated two-value PPO-Lagrangian and matched lambda-zero control.
- [x] Test signs, terminal targets, cohort completeness, checkpoint and runtime.
- [x] Launch bounded resumable comparison if checks pass; hand off after health
  verification without waiting for outcomes.

Status: CONTRACT AND IMPLEMENTATION REVIEW.

Contact replay completed: all50outcomes and episode lengths match frozen run;
18obstacle and2boundary contacts. Audit is not independent new performance data.
Design refinement: use full fixed-policy task cohorts for BOTH training arms,
MCcosttargets(gamma1,lambda_cost1), GAEreward.95; bothfit same two-value network,
onlylambda feedbackdiffers. Compare same32registeredcohorts, not equalsteps;
record actualphysics/gradient/walltime. Earlyfinish slots are parked(no extra
physics/tasks). Constrainttarget.05 remains exploratory, no safetyguarantee.
Six new tests initiallypassed. Mechanicalsmoke2cohorts each executed, exact
matching loss/weights when bothlambda0. Style20findings auto-fixed; nine typing
findings addressed with observation-shape narrowing and explicit predictions.

Verification complete:161tests passed,Ruff/Pyrightclean,compileallpassed. Full
preflight firstcohort2174physicalsteps and all8contacts => lambda0→.95; resume
secondcohort accumulated4287steps =>lambda1.9,control0. Exactinitialhashmatch.
SecondcohortcostVrange~.95–1.029 recorded(notcalibratedrisk); firstcohortconstant
costadvantage means normalization makes initialactorupdates essentiallysame,
expected rather than multiplier-signbug. No claims of learnedsafety yet.

Before main launch, budget refined32→64cohorts perarm to retain prior64policy
update cycles. Eacharm512registeredtasks, max2,048,000actualsteps; actualsteps
andwalltimepolicydependent, not equaltransitioncomparison. Evaldet50+stoch50
perarm; first3devseeds usedmechanicalpreflight, not claimeduntouchedformaltest.
Protocol: docs/DIRECTIONAL_PPO_LAGRANGIAN_PROTOCOL.md.

Status at launch (superseded by completion below): COMPARISON RUNNING / INITIAL HEALTH VERIFIED. Rootprocessgroup183290,
trainerPID183307. Lagrangiancompleted5cohorts/23438transitions,checkpoint0005
saved,lambda4.75,finiteupdates,noerror.json. Initialcohorts stillallcontacts;
executionverification is NOT evidence of learnedsafety. Nextcontrol scheduled
automatically; no ongoing monitoring or final-outcome waiting this turn.

## 2026-09-05 completed-pair failure diagnosis

User approves research after completed comparison: Lagrangian deterministic
0/50 safe versus matched control25/50; no new long training by default.

- [x] Validate final artifacts and quantify training outcome trajectories.
- [x] Audit objective/estimator and replay frozen checkpoints to measure
  cost signal, reward/cost competition and critic error without training.
- [x] Check primary literature against the observed failure mechanism.
- [x] Save reproducible analysis, figures, limitations and a minimal next
  experiment proposal; do not claim causal proof from a single run.

Status: DIAGNOSIS COMPLETE / VERIFYING SINGLE-FACTOR FOLLOW-UP.

All12checkpoint replays(96tasks) match original outcomes/lengths; weights unchanged.
Measured pre-update lambda-weighted cost/reward actor-gradient ratios10.79/18.88/
43.01 atlag16/32/64. Cost current-batchR²negative, but control64also negative;
cannot blame critic range alone. Lagzero-cost training81timeouts/92no-contact.
Common1/(1+lambda) denominator cancels under jointnormalization (up toepsilon).
Read primaryPID-Lag§7 andGAEpaper; selected onlydual_lr1→.01 test, notPIDstack.
Protocol docs/DIRECTIONAL_DUAL_TIMESCALE_PROTOCOL.md records limitations and
noautomaticpromotion. Preserve alloldsourcecontracts andweights.
Diagnostictool initially tried unavailable localruff; resolved viauvxruff,
without editing project Python dependencies. SPGpaperHTML unavailable; not
claimedfullyread or used as implementation authority.

- [x] Verify analysis algebra and legacy directional tests; launch single-factor
  slowdual pair, save checkpoint and confirm finite initial execution only.

Status: SLOWDUAL PAIR RUNNING / HAND OFF WITHOUT MONITORING.
Newartifact directional_lagrangian_slowdual_pair_20260905_v1, processgroup202678,
trainer202681. Firstcohort2174steps,lambda.0095,finiteupdates/checkpoint0001.
24tests pass,Ruffclean,compilepass,Pyright0 usingexplicit.venv interpreter.
InitialPyrightusedtoolenvironmentandreported8missingimports; fixedinterpreter
selection, notsourceimports. Matplotlib/protobuf deprecationwarningsonly.
Do not wait for results or auto-start downstream training.

## 2026-09-05 slowdual timeout investigation

Slowdual pair completed2668.36s: lagdet26safe/13contact/11timeout and stoch22/
16/12; matchedcontrol25/24/1and23/26/1 exactlyreproducespreviouscontrol.
User approves next research. Determine whether slowdualtimeouts reflect stalls,
circling, longpaths or near-goal failures before choosing a new intervention.

- [x] Validate paired outcomes and inspect reward/termination semantics.
- [x] Replay frozen evaluation with trajectory diagnostics; match original
  outcomes/lengths, preserve actor inputs and simulator behavior.
- [x] Identify supported mechanism and mathematical implication; verify any
  literature basis with primary sources.
- [x] Save evidence and a minimal next experiment; launch only if justified
  and tested, then hand off without outcome monitoring.

Status: TIMEOUT DIAGNOSIS IN PROGRESS.

200evalreplays matchedseed/outcome/steps/rawreturn; weightsunchanged.
Det10/11timeoutstailnetprogress<50m; stoch12/12tailpath>100mwithnet/path<.1.
Noevidenceofcollisionrepairasreason. Somefarstalls,someneargoalmisses;
5.2—5.8mnear-missesmustnotcountassuccess(goalradius5unchanged).
ContactdifferenceHolmadjustedp.139,bothmodes; notconfirmatorysafetyimprovement.
Report: artifacts/directional_slowdual_timeout_audit_20260905_v1/analysis-report.md.
Nextcostactortrace.95vsMC1fromsametrainedcheckpoint64,32newcohortseach.
CriticMCtargets/dual/physics/inputs/rewardunchanged. BasedonOmniSafegae-rtg
andGAEbias-variancetheory; notyetprovedtimeoutcauseorsolution.
29tests passed. Pyright6logginguniontypeerrorsfixedwithruntimedictcheck,
then0errors; nochangephysics. Ruffclean. Onecohortpairpreflightcompleted,
checkingresume65→66beforemainlaunch. Oldsourcefilespreserved.

Resumevalidated:botharms65→66withoptimizer/RNGrestored,16uniqueepisodes,
cohortlogs[65,66]exact,matchedforkweights. No evalperformanceclaims(preflight0tasks).
Main32additionalcohortpairlaunched:
artifacts/directional_cost_trace_pair_20260905_v1.
Status: BACKGROUND PAIR LAUNCHED / INITIAL HEALTH CHECK ONLY.

## 2026-09-05 user freezes collision contract (supersedes preceding launch)

The v1 cost-trace launch was paused and quarantined after a GPU backend
reproducibility discrepancy. Corrected preflights v2a/v2b completed; no v2
main run started. User rejects first-contact termination and freezes reward:
first contact -1.2, consecutive contact fixed -0.42, one clean step resets;
boundary/obstacle penalties additive; contact cost stays 1 per policy step.

- [x] Implement independent recovery environment and immutable reward guards.
- [x] Test reward sequence, repair/zero velocity, continuation and task metrics.
- [x] Register and launch frozen-policy recovery audit (not new training).
- [x] Verify initial progress/checkpoint, hand off without result monitoring.

Status: RECOVERY AUDIT LAUNCHED / INITIAL HEALTH CHECK. Historical source and
results preserved; no new first-contact training is authorized.
User additionally requests unified collision counts only; new logs do not split
boundary and obstacle. 34tests passed; Ruffclean; Pyright0. All historical source
hashes match original slowdual contract. Detached processgroup38284/trainer38287,
artifact directional_recovery_audit_20260905_v1. Smoke tested pause/resume across
arms. Main first two groups committed16tasks, weights unchanged, finite records,
no error file; initial health check passed. No outcome monitoring afterward.
Status: RECOVERY AUDIT RUNNING / HAND OFF AND WAIT FOR USER.

## 2026-09-06 train under locked recovery semantics

Recovery audit COMPLETE200tasks/580.02s. Frozen policies recover some contacts
but repeat contact heavily: stochastic slowdual27goal/22safe/488.76mean count,
control36goal/23safe/704.02mean count. User approves next research.

- [x] Inspect cost assumptions and verify primary PPO/PPOLag implementations.
- [x] Implement pure reward-PPO recovery baseline and count-target audit.
- [x] Validate updates, frozen unused cost head, checkpoints and resumption.
- [x] Launch bounded32cohort continuation plus100developer evaluations; check
  initial progress only, then wait for user.

Decision: old collector already sums costs correctly. Probability dual update,
binary cost_trace helper and unit-interval value diagnostics are NOT appropriate
for repeated counts. Do not invent a user safety budget. First establish proper
reward-only PPO baseline with unchanged locked contact penalties and inputs.
Source control checkpoint64; unused binary cost head has no loss/gradient.
This is baseline establishment, not evidence a safety method works.

R3 historical clarification verified at git7e30d8f: original SAC collision_terminal
false; later RACT and directional PPO used first-contact termination. Do not
describe original R3 as first-contact trained or conflate it with PPO control.
New runner train_recovery_ppo.py uses frozen recovery environment, no reward
edits. Preflight a checkpoint65 completed, resumed66 completed; checking against
independent uninterrupted b65→66. Cost net receives no gradients and stays fixed.
All stored costs are unified counts. PPO KL stops updates for a group, not run.

38tests pass, Ruffclean, Pyright0. Preflight a resumed64→65→66 versus independent
b uninterrupted64→66: models AND optimizers bitwise equal at65and66; episodes
and nontiming update metrics exact.16unique tasks, no duplicate resume records.
Locked recovery/audit source hashes unchanged. Training/eval sources checkpointed
in manifest. Background main launched recovery_ppo_baseline_20260906_v1, checking
first group only. No safety/performance claims from these preflight tasks.

Main processgroup54052/trainer54055 healthy. checkpoint65 committed16786new
steps,24.13s,finiteupdate,unusedcostheadunchanged. Maincheckpoint65 model exactly
matches preflight. Noerrorfile. Status: RECOVERY PPO TRAINING RUNNING / HAND OFF.
Do not monitor results; user will request review after completion.

## 2026-09-06 recovery PPO regression investigation

Baseline COMPLETE1182.32s,581518newsteps,32cohorts. Det8goal/4safe/42timeout,
stoch16goal/9safe/34timeout, vs frozen control36/25/14 and36/23/14.
User approves investigation. Locked environment sources remain unchanged.

- [x] Replay selected complete training batches and validate exact correspondence.
- [x] Measure value error, GAE-vs-MC signal, policy drift and update statistics.
- [x] Save paired statistics/figures and evidence-limited mechanism conclusions.
- [x] Choose a bounded single-factor follow-up only if evidence supports it;
  verify startup and hand off without monitoring its results.

Initial descriptive finding: last8groups29/64goals vs44—48/64earlier; last8
groups3436updates vs258/742/545precedingblocks. Different scene seeds mean this
is not a controlled learning curve. Need replay before blaming update count.

Replayed complete cohorts65/81/89/96 exactly. Reward-value MC R²=.15—.41;
GAE/MC correlation=.21—.34 and actor-gradient cosine=-.018—.551. This supports
testing actor credit assignment but does not prove MC is better. Update count
alone was not supported as the cause (Spearman rho=.202,p=.267). Analysis:
artifacts/recovery_ppo_regression_analysis_20260906_v1/analysis-report.md.

Single-factor pair changes only actor reward trace: MC(lambda1) versus exact
GAE(.95) control; reward critic target, networks, optimizer, collision/reward
contract and data seeds stay fixed. New GAE runner reproduced prior checkpoint65
model+optimizer bitwise. Focused23tests, Ruff and Pyright pass. Pause/resume was
verified across checkpoint boundaries. Main artifact
artifacts/recovery_reward_credit_pair_20260906_v1 launched as PID81355:16
cohorts/128tasks per arm plus det/stoch50 per arm. First MC checkpoint65 exactly
matches preflight and is finite; no error sentinel. Handed off without outcome
monitoring.

User fixed a lean-startup rule on 2026-09-06: no growing multi-gate chains.
Use only focused changed-code checks, at most one new-path smoke, one first-
checkpoint health check, and resume validation only when resume code changes.
Extra pre-gates require an identified risk, stopping rule, and explicit user
approval. This rule is recorded in AGENTS.md and the locked protocol.

## 2026-09-06 long PPO critic-budget experiment

User authorizes deeper full-paper reading and longer self-launched experiments.
MC actor pair completed1295.52s: det24/14 vs GAE33/18 goals/safe, stoch29/18
vs32/20. No support for switching actor to MC. Mechanism audit found KL also
stops critic: actual mean value passes1.3896 for GAE and.2065 for MC, despite
configured10 epochs. This is a candidate explanation, not an established cause.

- [x] Download9 primary papers plus PPG supplement; read relevant methods,
  theorem premises/proofs and experimental ablations, alongside official PPO code.
- [x] Implement critic-budget completion after the unchanged PPO joint update;
  preserve all actor updates, architecture, reward, GAE and contact semantics.
- [x] Verify focused invariants, exact control path, optimizer restoration and
  saved-evaluation skipping. Nine focused tests pass; Ruff/Pyright clean.
- [x] Launch96cohorts×2arms×2continuation random streams, with500 final held-out
  scenarios per arm and descriptive development evaluations every32cohorts.
- [x] Confirm first saved checkpoint and hand off without outcome monitoring.

Protocol: docs/RECOVERY_CRITIC_COMPLETION_PROTOCOL.md.
Literature: literature-search-20260906-ppo-critic-training/papers.md.
Background artifact: artifacts/recovery_critic_completion_20260906_v1, PID95809.
Expected4—7h for the registered schedule, without hard time cutoff or automatic
promotion. All four arms share historical initialization; do not describe this
as independent from-scratch seeds. No additional pre-experiment gate chain.

Startup health verified through saved checkpoint66:12 actor steps and740 value
steps (10 equivalent passes); extra phase leaves actor unchanged. Fixed-label
MSE .12043→.04009, finite metrics, no error sentinel. This verifies execution
only; it does not establish navigation improvement. Hand off and wait for user.

## 2026-09-06 scratch SAC/PPO matched-environment comparison

- [x] Audit previous result: critic completion reverses effect across streams;
  historical R3 success includes HOCBF intervention and is not a pure-policy baseline.
- [x] Register common recovery environment, directional LiDAR, reward scale,
  gamma and held-out seeds; no new safety modules or performance pre-gates.
- [x] Implement native SB3 SAC/PPO runner with full-state resumable checkpoints.
- [x] Focused tests, including changed resume path, and one tiny execution smoke.
- [x] Launch preplanned pair, verify first updated checkpoint, hand off.

Each algorithm: scratch seed0,524288transitions,8workers,500matched final scenes.
SAC first then PPO, not performance-conditioned promotion. This first seed is a
diagnostic comparison, not evidence of algorithm-level superiority. Contract:
docs/RECOVERY_SAC_PPO_SCRATCH_PROTOCOL.md. Do not alter locked environment files.

USER HOLD before formal startup (2026-09-06): do not launch until explicitly
called again. No scratch comparison training process has been started.
Native baseline runner and continuous-recovery/checkpoint utility drafted.
PPO now collects512steps/worker (4096total) before an update; gamma.99 in BOTH
new arms. Historical gamma1 files are deliberately unchanged.
Focused tests:7passed/1failed. Both SAC and PPO CPU next-update checkpoint
continuations match uninterrupted parameters/observations bitwise. Failing test
fixture requests x=.1 outside the allowed interior; fix its start point before
rerunning, not the environment. Ruff invocation unavailable on PATH; pending
static check and tiny real CUDA runner smoke. Do NOT mistake passing unit
continuation tests for a complete runner/startup validation.
Pending audit: immutable checkpoint behavior if a saved step is requested twice;
eval resume/zero-task smoke handling; post-warmup SAC time estimate. No smoke,
formal launch or first-checkpoint health check has occurred. Respect user hold.

HOLD RELEASED by user.8focused tests pass; py_compile passes. Ruff/Pyright are
not installed in this uv environment. One256-step real CUDA smoke completed
both SAC and PPO paths, including replay/checkpoint and zero-task evaluation.
Initial ordinary nohup child was reaped before Python startup (no run state was
created); relaunched in an independent session without changing configuration.

Formal artifact: artifacts/recovery_sac_ppo_scratch_20260906_v1, supervisor
PID4693 / Python PID4713 at launch. SAC checkpoint8192 is committed and healthy:
3192actor +3192critic updates; finite actor/Q/entropy metrics; model26.8MB,
replay3.294GB, worker/RNG state331MB, file sizes committed in metadata. Measured
second block109.83s (7.01s collection,102.82s update), giving a rough SAC-only
training estimate about5h including remaining sampling/checkpoints, not a
completion promise. Do not monitor results; user will ask.

Idea optimization registered before outcomes:
docs/RECOVERY_SAC_PPO_DECISION_LEDGER.md. It derives the approximately25.4x
maximum sample-presentation difference and pre-registers outcome-conditioned
single-factor follow-ups. Energy novelty remains the action-conditioned,
resource-augmented return-option problem; the standard baseline choice is not
claimed as innovation.

## 2026-09-06 correction to the formal500 evaluation

User correctly recalled the historical distance-stratified protocol. The first
runner had registered ordinary random seeds instead, a protocol regression.
User confirmed correction. A PAUSE request stopped SAC at a complete520192-step
checkpoint before any evaluation row existed; no training progress was lost.

- [x] Verify legacy immutable file: seed170001,500unique tasks, exactly100 in
  100–500,500–1500,1500–2500,2500–4000,>4000m; SHA-256
  6169ef6ec56e38d9c95260a1351bd40cfc8af0e62b6672dac93013494e00692d.
- [x] Add versioned migration runner without changing original manifest or
  checkpoint metadata. Preserve source-index world seeds and explicit starts,
  goals and zero velocities; interleave only execution order.
- [x] Keep locked nonterminal collision recovery and unified contact reporting.
  Add per-distance arrival/safe-arrival/timeout/unified-contact/path/energy.
- [x] Run10 focused tests and py_compile. Five-task real evaluation smoke covers
  all buckets; existing exact SAC/PPO resume tests remain passing.
- [x] Resume training, commit final SAC checkpoint524288, and verify the first
  stratified evaluation batch saved. Migration process Python PID61964.

Versioned runner: scripts/resume_recovery_sac_ppo_stratified.py. Migration audit:
artifacts/recovery_sac_ppo_scratch_20260906_v1/EVALUATION_PROTOCOL_MIGRATION.json.
At handoff status was EVALUATING_STRATIFIED/SAC,16/500. This count is startup
health only; no intermediate efficacy analysis or completion monitoring.
