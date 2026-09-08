# Grouped Counterfactual Safe–Energy Data Gate

## Question

Can the frozen R3 observation and a controlled change to its action process
predict mission safety and physical energy better than geometry plus the same
intervention descriptor?  This must be established before changing R3's actor.

## Experimental unit and replay contract

The independent unit is a complete scene seed: start, task goal, charger,
initial velocity, and static obstacle layout.  Five policy interventions are
executed from the same initial scene.  Their task and return legs reuse the
identical obstacle layout.  Interventions within a scene are correlated
counterfactual measurements and never cross train/test boundaries.

The frozen R3 action is \(a_t^0\).  Each behavior action is

\[
  a_t^{(j)}=\operatorname{clip}
  \left(g_j a_t^0+\sigma_j\eta_t^{(j)},-1,1\right),\qquad
  \eta_{t+1}^{(j)}=\rho_j\eta_t^{(j)}+
  \sqrt{1-\rho_j^2}\,\epsilon_t^{(j)}.
\]

The fixed interventions are nominal, conservative, low residual, medium
residual, and high residual.  Their random streams are keyed by scene and
intervention, so scheduling and resume order cannot change a trajectory.
These are data interventions, not deployed navigation rules.

## Recorded quantities

Every rollout records the initial R3 observation, per-step compact goal state,
R3 action, intervened action, HOCBF-executed action, physical energy increment,
progress, contact indicators, intervention norm, nominal/executed barrier
slack, and leg identity.  The episode record contains task success, return
success, joint mission success, energy, path ratio, collision/contact counts,
and safety-intervention statistics.

## Locked split and baselines

Outer fold is `scene_index mod 3`.  Validation scenes are selected only inside
the two training folds.  All normalization and early stopping use training and
validation scenes only.

- Geometry baseline: normalized task distance, task-to-charger distance, and
  the same action-intervention descriptor used by the learned model.
- Learned representation: frozen 96-dimensional R3 structured-LiDAR embedding
  plus the intervention descriptor.
- Identical MLP fitting and early stopping are used for both inputs, so the
  comparison tests representation signal rather than model capacity.

Primary outcomes are mission energy fraction and nominal-unsafe step fraction.
Mission success is secondary because the hard safety layer may make failures
rare.

## Pre-actor data Gate

The Gate passes only if all conditions hold:

1. every planned scene–intervention rollout is atomically complete;
2. matched initial observations agree to `1e-6` within each scene;
3. non-nominal interventions produce nonzero executed behavior deviation;
4. at least 25% of eligible scenes show at least 5% mission-energy spread among
   successful counterfactuals;
5. at least 25% of scenes show at least 0.05 spread in nominal-unsafe fraction
   or a collision/success outcome change;
6. the learned representation lowers scene-held-out energy MAE by at least 2%
   relative to geometry;
7. the learned representation lowers scene-held-out nominal-unsafe MAE by at
   least 2% relative to geometry.

Smoke runs validate only code paths.  Passing this Gate authorizes a subsequent
joint-policy pilot; it is not evidence that a learned policy is safe or
energy-sustainable.

## Runtime and recovery

Collection uses the existing parallel UAV workers, batched GPU policy
inference, native HOCBF path, and the existing physical telemetry energy model.
Each rollout is written atomically.  `--resume` skips valid completed rollouts.
There is no wall-clock kill; progress and terminal sentinels are explicit.
