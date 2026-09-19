# Control adaptation: startup only, 2026-09-19

Active problem: adapt a useful pretrained controller to changed actuator response
with few new interactions while retaining nominal capability. No proposed algorithm,
forgetting result, safety guarantee, or venue-readiness claim is established.
Budget-reading paused; AFPP/Teacher V3/full-vector escalation remain stopped.

## Implemented and run

Isolated code: `research/control_adaptation`; upstream safe-control-gym commit
`6b5391d014f36fdfa0f9d22d92c77387e5274308`, unmodified checkout
`/home/zjl/safe_control_gym_reference_20260919`.
Python 3.11.15 environment `/home/zjl/control_adaptation_py311_20260919` uses
CPU Torch 2.7.1, SB3 2.8.0, Gymnasium 1.2.3, NumPy 1.26.4, PyBullet 3.2.7,
CasADi 3.7.2. The failed Python 3.12/PyBullet source-build attempt is preserved;
the old project environment was not modified.

Native figure-eight 2D task, native reward/termination, 21 public observation
components (native 12 plus previous physical state/action/valid bit). No hidden
gain or simulator model enters actor inputs. This is SB3 SAC with documented
training differences, NOT reproduction of the author's SAC scores.

Five focused tests pass: timeout semantics, gain/hover check, native parity and
history/privacy, reset reconstruction, and exact uninterrupted-versus-resumed
training tensors/history. Native parity initially failed; diagnosis below, not
a relaxed tolerance, resolved it. Training was inadvertently started before that
test result was gated; its single-client execution was subsequently checked and
does not exercise the identified multi-client fault. No training result was discarded.

First real checkpoint (paused, no background training):

| Item | Observed |
| --- | ---: |
| Base seed | 5101 |
| Collected transitions / replay entries | 1,573 |
| Optimizer updates | 617 |
| Completed episodes | 47 |
| Parameter finite check | true |
| Last ten episode mean return | 9.3336 |
| Last ten mean length | 26.2 |

Checkpoint `/home/zjl/control_adaptation_base_5101_20260919/checkpoint_1573`
contains model/optimizers, replay, RNG, indexed reset state, episode history and
source/config bindings. This is a startup-health result, NOT useful-controller
qualification. Only one seed has started; no adaptation update has run.

## Two concrete upstream findings

1. `base_aviary.py:226` calls `changeDynamics` without `physicsClientId`. Two
   simultaneous clients therefore do not receive the same damping settings.
   Native parity passes when tested serially. Actual trainer/evaluator uses one
   environment at a time. Do not parallelize clients in the same process.
2. `Quadrotor.__init__` overwrites supplied `INIT_STATE_RAND_INFO` after the
   superclass initializes it. `_randomize_values_by_info` ADDS perturbations.
   Consequently this pinned task has effective z = 1 + U(.1,1.5), while terminal
   altitude is 2. YAML ranges alone do not describe the effective distribution.
   Diagnostic evaluation seed 8101 resets at z=2.2031674385. Frozen startup SAC
   and nominal-model LQR both terminate after one step in all four gain conditions.

These eight evaluation transitions are separate from training replay. Reports:
`/home/zjl/control_adaptation_startup_sac_20260919.json` and
`/home/zjl/control_adaptation_startup_lqr_20260919.json`.
This smoke diagnoses reset semantics; it cannot establish controller ranking,
tracking feasibility, actuator sensitivity, or retention. No seed was replaced.
No reset distribution was silently narrowed and no upstream file changed.

## Next action and boundaries

Pause here under repository startup instructions. Resolve and explicitly version
the initialization semantics before committing long base training. A correction
must apply identically to all controllers, retain this startup evidence, and use
a new run lineage rather than silently resume with changed physics/initialization.
Then establish competent multiple-seed base checkpoints before adaptation.

Ordinary fine-tuning, replay retention, frozen-base residual learning and estimated
input calibration are specified but NOT implemented or compared yet. Strict
adaptation interaction caps also remain pending. Static hover feasibility is only
necessary, not a tracking certificate. Nominal LQR here is not system identification.
No DEV/FPL/CONFIRM assets were opened for this new task; no GPU training, remote
push or commit was performed. Research-workflow skill informed separation of
startup evidence, interaction accounting and scientific claims.
