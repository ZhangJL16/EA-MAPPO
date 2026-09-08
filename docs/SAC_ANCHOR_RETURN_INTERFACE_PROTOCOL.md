# Scratch SAC anchor-return executed-interface diagnostic

Frozen 2026-09-06 before collecting outcomes. Development mechanism evidence only; not confirmation.

## Question

Does the newly trained scratch SAC repair the old R3 return-to-charger bottleneck on the same 310 preserved task-leg anchors, and does the standard HOCBF materially change the resulting resource first-passage law?

## Fixed inputs

- Policy: scratch SAC seed 0 at exactly 524,288 transitions, selected before this anchor diagnostic from the completed matched SAC/PPO experiment.
- Anchors: all 310 task-leg anchors in `rcps_meet_confirmation_fork_150scenes_20260904_v1`; positions, velocities, obstacles, charger, scene seed, and distance labels are unchanged.
- Two matched interfaces: deterministic raw SAC action; deterministic SAC proposal passed through the repository's existing default HOCBF. No learned parameter differs between arms.
- Return leg: 4,000 policy-step horizon, 24 obstacles, full 2×8×128 LiDAR plus remaining-time observation, no finite-battery termination.

## Immutable collision behavior

Contact never terminates the return episode. Position is repaired, all velocity components are zeroed at the repair instant, and the same charger target continues. Boundary and obstacle events contribute one unified contact indicator at most once per policy step. Collision-free arrival is exactly charger arrival with unified contact count zero. No boundary/obstacle split statistic is saved.

The reward implementation remains raw −1.2 per applicable contact after a clean step (including the first contact after reset) and −0.42 after any-contact preceding policy step, with a clean policy step resetting the next contact to −1.2. Reward is observational here; no training occurs.

## Outcomes and interpretation

For each interface report charger arrival, collision-free charger arrival, timeouts, per-task unified contacts, contact-step rate, energy-to-charger, path ratio among arrivals, HOCBF intervention rate, and distance-bucket breakdown. Each anchor is the paired unit. There is no numerical promotion gate.

- High arrival with a rare contact/timeout tail supports proceeding to a state-level defective resource-to-charger distribution under stochastic dynamics.
- Broad failure in both arms means return-target navigation must be trained before energy modeling.
- Better collision-free arrival but worse arrival/energy under HOCBF is direct evidence that the executed interface changes the return distribution and must be represented explicitly.

The old 3,410-rollout diagnostic terminated on first contact and split collision types. Its files remain legacy evidence and are not overwritten. This version intentionally estimates current nonterminal recovery behavior and writes a fresh artifact.

## Execution discipline

Every completed anchor is an atomic row. Interruption discards only active incomplete trajectories; `--resume` reruns those anchors and preserves committed rows. Run only focused changed-path tests, one two-anchor smoke, and one first-batch formal startup-health check. Do not monitor to completion or automatically trigger further training.
