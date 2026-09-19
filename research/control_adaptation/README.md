# Pretrained control adaptation and retention — public-task study

Active research question: with few NEW deployment transitions, how can an
already useful controller adapt to actuator-response changes while retaining
nominal capability? No novel algorithm, forgetting phenomenon or safety guarantee
is assumed. Budget-reading is paused; AFPP/V3/full-vector escalation stay stopped.

## Current scope

This first delivery restores the public task, implements a reusable interface,
starts **base SAC pretraining**, and validates checkpoint recovery. It does NOT
yet deliver adaptation curves: these require competent base policies first.
No long training is automatically promoted to another seed or adaptation stage.

Startup is now paused at 1,573 transitions. See
[`CONTROL_ADAPTATION_STARTUP`](../../docs/CONTROL_ADAPTATION_STARTUP_20260919.md).
Do not resume long training before resolving the native reset-distribution issue.
Run only one PyBullet environment per process: the pinned upstream omits the
client ID in its damping setter. Native-equivalence regression runs serially.
The constructor also overwrites YAML randomization with additive class defaults:
effective initial z is 1 + U(.1,1.5), sometimes above the z=2 terminal boundary.
Neither issue has been silently patched in upstream. This limits baseline readiness.

## Upstream task and deliberate changes

Pinned [safe-control-gym](https://github.com/learnsyslab/safe-control-gym/tree/6b5391d014f36fdfa0f9d22d92c77387e5274308)
commit `6b5391d014f36fdfa0f9d22d92c77387e5274308`.
Read the official `examples/rl/config_overrides/quadrotor_2D/quadrotor_2D_track.yaml`:
2D figure-eight tracking, 50Hz control, 1000Hz PyBullet, 5-second episodes,
randomized native initial conditions, native exponential tracking reward and
constraints. No initial-state narrowing or reward shaping. Out-of-bound failure
and time truncation are kept distinct. No old UAV collision repair is imported.
That freeze continues to apply to the OLD navigation environment, not this
separate public benchmark's native terminal rules.

The public benchmark state and next reference (12 floats) are augmented for ALL
future learned methods by previous physical state (6), last commanded action (2)
and history-valid flag (1): 21 public observations. No hidden gain, actual applied
thrust, physical-parameter dictionary or symbolic model is passed to the actor.
Policy APIs take observation only. This is full-state control, not LiDAR navigation.

SAC is **Stable-Baselines3 2.8.0**, not claimed as an exact reproduction of the
author's SAC training stack. Hidden [128,128], fixed entropy=.2, lr=.001,
batch=256, gamma=.99, tau=.005, 1,000 warmup transitions follow the supplied
SAC settings where applicable. Differences: one environment instead of four,
episode-batched updates rather than per 100 transitions, replay cap=200k,
21-dimensional short-history observation instead of 12. One update per collected
transition after warmup. Small MLP runs on one CPU thread; GPU is not required.
No claims compare this variant against the author's published scores.

## Deployment conditions (prospective DEV only)

`u_applied = clip(gain * u_command_physical, physical_bounds)` uses the native
action-disturbance hook AFTER denormalization and BEFORE physical clipping/PWM.
Nominal, [.95,.95], [1.05,1.05], [.95,1.05] are listed before results.
The normalized action box corresponds to hover ±10%; a 20% thrust loss can make
hover impossible. We therefore check `(1/gain - 1)/0.1` is in [-1,1]. This is a
necessary STATIC check, not proof that the trajectory is feasible. Report
saturation, tracking and failure alongside a traditional-control diagnostic.
No condition is selected/dropped according to a learned method's performance.

## Planned comparison — NOT implemented/running yet

Same independently pretrained seeds 5101/5102/5103, same base checkpoint per
method, same new-interaction budgets [0,1000,5000,10000]. Compare frozen SAC,
plain fine-tuning, explicitly old-replay retention, frozen-base residual learning,
and estimated input calibration. The latter must estimate from charged new
interactions, not read the hidden gain. Known-gain inversion may ONLY appear as
an explicitly privileged feasibility diagnostic, never as the deployable method.

Old-replay critic mixing and actor behavior anchoring are distinct choices; the
first comparison must specify which is used and report old samples/updates.
Do not silently mix transition dynamics or count old data as free new data.
Same adapted policy goes to E0 and E_delta with no environment-identity switch.
No old-environment resets are allowed to update it during retention evaluation.
All warmup, failures and partial episodes count toward N; strict adaptation-budget
handling still needs implementation. Base pretraining is episode-bounded and may
overshoot its requested checkpoint/budget by at most 249 transitions.

Report new return vs N, nominal return change, position RMSE, completion/termination,
violation and saturation counts, interactions and optimizer/compute costs. RMSE
over a failed short episode is not comparable without its length/failure flag.
Evaluation transitions remain separate from replay and are reported separately.
No safety theorem or novelty claim follows from this infrastructure.

## Installation and commands

Upstream checkout: `/home/zjl/safe_control_gym_reference_20260919` (unmodified).
Active isolated environment: `/home/zjl/control_adaptation_py311_20260919`.
Install CPU Torch 2.7.1 from the official PyTorch CPU index, then use
`requirements-runtime.txt` and `constraints.txt`. No optional MPC/Mosek/GP stack
is needed for the native environment and nominal LQR diagnostic.
Python 3.12 source-build attempt failed; its unused environment is preserved at
`/home/zjl/control_adaptation_env_20260919`. The active Python 3.11 environment
does not depend on the old UAV environment. No system package or old venv changed.

From the repository root:

```bash
PYTHONPATH=research/control_adaptation/src /home/zjl/control_adaptation_py311_20260919/bin/python -m unittest discover -s research/control_adaptation/tests -v
PYTHONPATH=research/control_adaptation/src /home/zjl/control_adaptation_py311_20260919/bin/python -m cadapt.train --config research/control_adaptation/configs/study_v1.json --reference /home/zjl/safe_control_gym_reference_20260919 --output /home/zjl/control_adaptation_base_5101_20260919 --seed 5101 --stop-after-checkpoint
```

The training command starts/resumes to one next saved checkpoint. Omit
`--stop-after-checkpoint` only to run toward the declared 200k base budget
(one-hour invocation watchdog pauses after the next complete episode).
No automatic adaptation job is chained. SIGINT/SIGTERM requests a boundary save.

Save contains model/actor/critics/optimizer/temperature, replay buffer, random
states, full episode statistics, source/config bindings and deterministic next
episode reset index. Atomic latest pointer only promotes complete directories.
Resume after a crash uses the last complete checkpoint; lost work must not be
called charged adaptation data. Existing run without checkpoint fails visibly.

Evaluation, once a checkpoint is selected for a stated purpose:

```bash
PYTHONPATH=research/control_adaptation/src /home/zjl/control_adaptation_py311_20260919/bin/python -m cadapt.evaluate --config research/control_adaptation/configs/study_v1.json --reference /home/zjl/safe_control_gym_reference_20260919 --checkpoint /ABSOLUTE/CHECKPOINT_DIRECTORY --output /ABSOLUTE/NEW_REPORT.json
```

Without `--checkpoint`, this evaluates nominal-model discrete LQR using the same
normalized action box and public next reference. It is a conventional diagnostic,
not system identification or an already completed strong calibration baseline.
