# Matched-environment scratch SAC/PPO diagnostic

Registered 2026-09-06 before running. No efficacy claim is presupposed.

## Motivation and scope

Previous critic-completion final500 results reversed between two continuation
streams (safe arrival3.8% vs36.2%, and52.6% vs21.8% for modification/control).
It is not a demonstrated improvement. Historical R3's96% arrival used HOCBF
on35.9% of evaluation policy steps; it does not establish pure SAC superiority.

Test native SB3 SAC and PPO from scratch under a shared environment. This is
a baseline-family comparison, NOT a one-factor explanation of past failures.
The new collector and scratch initialization differ from past cohort fine-tuning.
No claim that these unconstrained baselines guarantee collision safety.

## Fixed common conditions

- Reuse RecoveryCohort/LockedRecoveryPlant unchanged. Contact repairs position,
  zeros all velocity at repair, never terminates alone. Raw penalties remain
  -1.2 / fixed-.42 with additive simultaneous penalties; binary undiscounted
  contact cost and one unified count per policy step. No contact-generated
  progress/velocity/goal reward. Cost is logged, not optimized separately.
- No HOCBF, projection, rule controller, energy reward or added loss/head.
-24obstacles, horizon4000, goal radius5m, physics and LiDAR unchanged.
- Full2056-dimensional observation:7state +2×8×128LiDAR +remaining time.
  Reuse directional structured extractor, ordered2×16readout. Separate actor
  and critic encoders; downstream widths256×256. Intrinsic SAC twin Q heads
  versus PPO V head are not described as equal parameter counts.
- Reward raw×.01, gamma.99 for both (R3 discount; prior custom PPO used1).
  This does not discount the recorded collision count/cost.
- lr3e-4,batch256,8workers, initialization seed0;524288transitions per arm.
  Per-worker training scenario schedule993600001+i+8k, identical ordered
  streams, but task lengths imply different numbers of consumed scenes.
- SAC: replay200000, warmup5000, tau.005, one gradient update per collected
  transition (train_freq1vector step, gradient_steps=-1), auto entropy target-3.
  Initial alpha.01 is in scaled reward units; R3's raw-reward alpha started1.
- PPO:512steps/worker,10epochs, GAE.95,clip.2,target KL.02,entropy coefficient0,
  value coefficient.5,gradient norm.5,initial action std.5. Native SB3 algorithm,
  no critic-completion phase. Native Gaussian/PPO and tanh-Gaussian/SAC differ.

## Evaluation, interpretation and timing

After each arm,500deterministic held-out scenes993800001..993800500; no training
on these seeds, no checkpoint selection by their results. Report arrival,
collision-free arrival, timeout, unified contacts, energy and successful-path
ratio. Save every8completed evaluation tasks; partial groups replay on resume.
Report environment transitions, actual actor/critic optimizer steps and wall
time, because equal interactions do not imply equal compute. One initialization
only: descriptive diagnostic, not broad method superiority or an oral claim.

The pair is preplanned SAC then PPO, not automatic promotion based on results.
No hard wall-time termination; checkpoints support interruption. Runtime must
be estimated from measured post-warmup speed rather than assumed GPU speed.
Only focused tests, one tiny new-path smoke and first trained checkpoint health
check; then hand off without watching performance or completion.

## Recovery

Checkpoint after initial8192steps, every32768steps, final, or requested pause.
Pause is honored between4096-transition training blocks; SAC saves replay,
model/optimizers/temperature, all worker environment states including unfinished
tasks, random generators and last observation. PPO saves the same minus replay
at a completed rollout/update boundary. Checkpoints are immutable step folders;
latest pointer is atomically replaced only after all files and metadata exist.
Crash before pointer commit resumes prior committed checkpoint, discarding
uncommitted episode/update log rows. Resume checks source/argument hashes.
Send SIGTERM or create PAUSE to request a checkpointed pause; restart with the
same command plus --resume after removing a user-created PAUSE request.

Primary method sources: SAC https://arxiv.org/abs/1812.05905 (sections4.2,5),
PPO https://arxiv.org/abs/1707.06347 (sections3–5). Local installed SB3 source
was inspected for loss, optimizer saving, learn-boundary and force_reset behavior.
Prior fulltext readings: literature-search-20260906-ppo-critic-training/.

## Registered evaluation correction, 2026-09-06

The initial runner incorrectly generated500 ordinary random tasks. The user
recalled and confirmed the pre-existing immutable navigation protocol before
any evaluation began. Training was checkpoint-paused; no random-evaluation row
exists. Training checkpoints and hyperparameters remain unchanged.

The final evaluation instead reuses the exact historical task file with seed
170001 and SHA-256
`6169ef6ec56e38d9c95260a1351bd40cfc8af0e62b6672dac93013494e00692d`:
100tasks in each of100–500,500–1500,1500–2500,2500–4000 and>4000m.
Start, goal, zero initial velocity and per-source-index world seed are fixed.
Execution order is interleaved by bucket only to make resumable prefixes
balanced. Both algorithms use the same500 scenes and deterministic actions.

This restores distance comparability but does not restore obsolete collision
semantics or split collision reporting. Evaluation still uses locked nonterminal
repair, zero velocity at repair, and one unified per-policy-step contact count.
The versioned migration runner is
`scripts/resume_recovery_sac_ppo_stratified.py`; the old training manifest and
checkpoint metadata are retained unchanged.
