# Correction-SAC fixed500 successor and energy handoff

## Authorization and execution

User request: finish current learning, add the 500-task test, then validate the
previous energy designs on the newly trained navigation model ("500k那个").
This supersedes the earlier no-automatic-evaluation setting for this one
explicitly requested successor only. It does not change the running trainer,
extend its budget, or automatically select/promote another navigation variant.

- Training: `artifacts/hocbf_correction_sac_20260908_v1`, warm start from524288,
  then131072 additional correction-supervised transitions.
- Queue: `artifacts/hocbf_correction_followup_20260908_v1`, detached PID19195.
- Formal evaluation: `artifacts/hocbf_correction_fixed500_20260908_v1`.
- Queue waits for `TRAINING_COMPLETE_AWAITING_USER` AND the exact131072 final
  checkpoint. Evaluator verifies the training manifest, source hashes, model
  size and checkpoint metadata. No intermediate checkpoint is formal evidence.
- Errors, training pause, missing training process, changed source/plan, or a
  queue pause stop the successor. It never restarts training automatically.
- After the requested evaluation, stop at `COMPLETE_AWAITING_USER`; no automatic
  energy training is implemented or launched by this queue.

## Fixed500 evaluation

Reuse the immutable original task file, hash
`6169ef6ec56e38d9c95260a1351bd40cfc8af0e62b6672dac93013494e00692d`.
Five distance buckets:100–500,500–1500,1500–2500,2500–4000,>4000;100 tasks each.
Original scene seeds, starts, velocities and goals are unchanged. Horizon4000,
24 obstacles,2056-D observation; no battery-limited mission added to this
navigation benchmark. Do not interpret its accumulated energy as proof of
sustainable operation.

Freeze one final model; deterministic inference with no optimizer updates.
Evaluate raw actions and HOCBF execution on the SAME500 tasks, sequentially:
500 unique tasks,1000 total trajectories. This separates filter-assisted system
performance from filter-free navigation skill; it does NOT alone identify the
causal effect of correction supervision (that needs matched lambda0 training).

Keep locked nonterminal collision recovery, zero velocity at repair, fixed
reward discount, and one unified collision count. Report goal arrival,
zero-contact arrival, timeout, contacts, successful-path ratio, energy, and
HOCBF intervention/fallback/emergency steps, both overall and by distance.

Eight parallel workers share batched CUDA inference. A finished worker parks
until the next task group. Commit atomic ordered task records after each group;
resume validates the exact saved task prefix. An interrupted group, at most8
tasks, is replayed; earlier committed groups and completed arms are retained.

## Pause and resume

`SIGTERM` or a `PAUSE` marker in the queue root pauses the successor; during
evaluation it forwards SIGTERM to the evaluator. A PAUSE in the evaluation root
also stops at its atomic batch boundary. Training pause remains independent.
Do not indiscriminately kill all Python processes. Remove only an explicitly
identified PAUSE marker when the user requests resumption.

```bash
.venv/bin/python scripts/queue_hocbf_fixed500.py \
  --source artifacts/hocbf_correction_sac_20260908_v1 \
  --output-dir artifacts/hocbf_correction_followup_20260908_v1 \
  --evaluation-dir artifacts/hocbf_correction_fixed500_20260908_v1 --resume
```

An advisory lock prevents duplicate successors. Resume is explicit; a paused
training process must first be resumed using its unchanged original command.

## Energy stage: reconciled plan, not a launched experiment

Budget clarification was sent asynchronously. Working interpretation:500000
ADDITIONAL energy-stage environment transitions with navigation frozen, not an
extension of navigation adaptation. Until clarified, no budget mutation or
old-pipeline launch is performed. The historical runner explicitly defines
`--phase2-energy-transitions` as frozen-navigation mission transitions used for
energy-model learning; optimizer update count is a different quantity.

Reuse methods, not incompatible labels or checkpoints:

1. Bind the exact newly trained navigation checkpoint and the actual execution
   interface. Primary deployment candidate is new navigation+HOCBF; raw execution
   is a labeled diagnostic. Hash frozen policy before/after energy collection.
2. Adapt the previous frozen-policy energy collection to the current2056-D
   observations and locked recovery protocol. Do not directly invoke the old
   JSEB/earliest-R3 runner, assume its encoder matches, reuse its battery
   calibration as validated for this navigator, or mix old-policy return labels
   into new-policy ground truth without an explicit transfer experiment.
3. Reuse structured LiDAR. Deployable energy inputs may include sensor-derived
   geometry/encoder features, velocity, relative charger location and measured
   battery. Never use simulator obstacle centers/radii/full-map clearance.
4. Compare previous ideas as separate arms, not a pile of navigation modules:
   fixed battery threshold; finite mean/MC energy-to-return regression;
   quantile-TD conservative cost-to-go; and the recent factorized safe-return
   resource distribution. Share declared data/splits where statistically
   appropriate, and report their actual differing data requirements.
5. For the distributional arm, retain
   `Y=realized energy` on zero-contact timely return, `Y=+infinity` on failure;
   learn `P(Y<=battery | observations, executed interface)` through structural
   return probability times conditional finite-energy CDF. This distinguishes
   navigation failure from merely expensive successful return. No finite cap
   is silently treated as the true energy of failure.
6. New-policy rollouts supply labels. Partition by scene before fitting;
   related anchors/replicates stay together. The fixed500 navigation test is not
   energy training data. Budget-truncated unfinished trajectories are censored,
   not labeled as completed returns or structural failures. Keep their state
   resumable and account separately for any extra label-collection rollout cost.
7. Compare dangerous false-safe decisions, safe return, mission utility,
   battery exhaustion and calibration, not MAE alone. A subsequent closed-loop
   energy benchmark needs its own declared battery and mission protocol; the
   navigation500 test is not interchangeable with that benchmark.

Porting the collector/decision loop and freezing those experimental details
remain required work before energy launch. Do not reinstate historical hundreds
of pre-performance gates: use only focused changed-path checks, one tiny smoke,
and first committed startup health. Neither a factorization identity nor a
good500-task navigation score establishes a novel theorem or oral-level result.

## Verification

-6 focused tests passed: raw/HOCBF worker parking, both batch-resume paths,
  immutable task-prefix rejection, successor terminal/budget handling, resume
  command construction.
- One actual committed32768 checkpoint smoke:5 tasks×2 steps×2 arms on CPU,
  both complete. Explicit nonformal artifact:
  `artifacts/hocbf_correction_fixed500_smoke_20260908_v1`.
- New Python files compile. All running-training source hashes unchanged.
- Queue startup health confirmed:PID19195 alive,WAITING_FOR_TRAINING, bound
  manifest/code hashes match, no ERROR; trainingPID9399 alive at53248/131072.
  Formal evaluation has not started. No waiting for completion in this turn.
