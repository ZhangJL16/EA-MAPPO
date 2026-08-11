# Certified Recovery Rollout Teacher at 2x Flight Energy

## Scope

This protocol is independent of the frozen solved navigation artifacts under
`artifacts/phase1_sb3_sac_1m_gpu/` and the historical 1x energy artifacts under
`artifacts/phase2_sb3_sac_energy_open_1m/`. It creates only:

- `artifacts/phase2_sb3_sac_energy_open_1m_2x_unguided/`
- `artifacts/phase2_sb3_sac_energy_open_1m_2x_recovery_guided/`

The 2x condition multiplies `EnergyModel.realized_cost(...)` after computing the
base flight cost. It does not change `dt`, `v_max`, `a_max`, battery capacity, or
charging rate. The energy reward penalty uses this actual multiplied flight cost.
Thus 1x versus 2x changes both endurance and the numerical energy-cost reward term.

## Certificate Binding

Regenerate the 2x numerical recovery atlas with:

```bash
cd review_bundle
./.venv/bin/python scripts/generate_sb3_energy_certificate.py \
  --scenario random_persistent_open.json \
  --flight-energy-multiplier 2.0 \
  --output certificates/random_persistent_open_energy_x2.json
```

The descriptor binds the multiplier, energy calibration version/hash, recovery
manifest hash, atlas hash, terminal recovery certificate, and frozen-κ version.
Runtime loading is fail closed. A 1x artifact is rejected by a 2x runtime.

## Guided Initialization

The task teacher must be an existing solved Standard SB3 SAC checkpoint supplied by
`--navigation-teacher-model`; there is no random-policy or PID fallback. The oracle
reuses the certified recovery atlas, frozen κ, certified station hold, and departure
gate. Oracle values never enter the clean 77-dimensional actor observation.

A successful demonstration must contain:

```text
task SAC -> certified margin band -> frozen κ -> station -> certified hold
-> departure gate -> task SAC resumes -> original pending goal completed
```

Station arrival, charging onset, and departure are not demonstration success. Only
complete trajectories enter the standard SB3 replay buffer. Only states controlled
by frozen κ supervise the actor deterministic output. Actor warm start uses MSE on
normalized κ actions and verifies that critic, target critic, and entropy coefficient
remain unchanged.

The inserted buffer is therefore **outcome-conditioned initialization data**, not an
unqualified unbiased sample of a stochastic Bellman kernel. Every tuple is generated
by the same environment, but filtering on future full-cycle success may change the
conditional successor law given an aliased `(observation, action)`. The current use is
a deterministic-simulator initialization heuristic; no convergence, unbiasedness, or
asymptotic improvement theorem is claimed.

After initialization, `model.learn()` has no online teacher, κ, verifier, or active
hold override.

## Matched Controls

Both conditions use Standard SB3 SAC 2.8.0, 2x flight energy, and identical clean
environment, reward, observation, dynamics, battery, charger, seeds, checkpoints,
and held-out protocol.

- `unguided`: no replay prefill and no actor warm start.
- `recovery_guided`: complete-cycle prefill and actor-only κ warm start.

## Evaluation Authority

- `POLICY_ONLY`: no teacher, κ override, verifier recovery override, or active hold.
  Natural physical charging remains enabled.
- `SYSTEM_WITH_KAPPA`: no teacher; verified action support, frozen κ, certified hold,
  departure gate, and the multiplier-bound 2x certificate are active.

`SYSTEM_WITH_KAPPA` cannot prove that the actor learned charging behavior.
Every checkpoint writes separate deterministic/stochastic, full/low-SOC, and
execution-authority result files.

## Formal Commands

The launchers reject existing output directories and run seeds serially on one GPU.

```bash
cd review_bundle
PHYSICAL_GPU_INDEX=0 bash scripts/launch_sb3_energy_2x_unguided_1m.sh
PHYSICAL_GPU_INDEX=0 bash scripts/launch_sb3_energy_2x_recovery_guided_1m.sh
```

The guided launcher uses each seed's solved teacher at
`artifacts/phase1_sb3_sac_1m_gpu/seedN/checkpoint_step_1000000.zip`. A portable
alternative root can be supplied with `NAVIGATION_TEACHER_ROOT`.

No learned charge/return head, station-direction reward, low-energy return bonus,
charging bonus, full-charge bonus, or distance-to-station shaping is added. SAC
remains the continuous task policy; learned safety fields remain future risk
estimators; the certified layer and frozen κ remain independent strict fallback.
