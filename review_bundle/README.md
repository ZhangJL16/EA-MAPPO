# Staged Energy Management for Persistent UAV Navigation

The active engineering route keeps navigation and energy management separate. A frozen goal-conditioned SAC selects actions for the active goal. The Energy Module estimates charger-return energy and decides whether TASK mode can continue or must irreversibly commit to the CHARGER goal. Collision code is retained but is not expanded in the current stage.

## Active Sources

- Theory and claim boundaries: `docs/new_theory/`
- Navigation baseline environment: `envs/navigation/`
- Safety models: `safety/collision/`, `safety/energy/`, `safety/calibration/`, `safety/switching/`
- Frozen goal-conditioned policy adapter: `agents/goal_conditioned_sac/`
- Stage A and transfer interfaces: `experiments/energy_transfer/`
- Superseded diagnostic E1: `experiments/new_route/`
- Tests: `tests/new_route/`

## Retained Baseline

`artifacts/phase1_sb3_sac_1m_gpu/` is the completed three-seed Standard SAC navigation baseline. Its checkpoint contract is 77 observation dimensions and 3 continuous actions. The large resource budget and telemetry-cost feature are retained only for checkpoint compatibility; the artifact does not establish finite-energy safety.

Run the focused baseline and safety tests:

```bash
PYTHONPATH=. /home/zjl/mappo/.venv/bin/python -m pytest tests/new_route -q
```

The first E1 protocol is superseded because it switched to charger after one task action and used only the navigation compatibility energy. Inspect it only as provenance:

```bash
python scripts/check_new_route_runs.py
```

Do not relaunch into existing immutable run directories and do not interpret incomplete runs.

## Active Stages

- **Stage A — 4×4 bootstrap:** varied task prefixes, random charger points, complete frozen-SAC charger returns, per-transition return-to-go supervision, and B0–B3 training.
- **Stage B — 16×16 target interface:** separate `target_open_16x16.json`; no claim yet that the 4×4 policy is competent at this scale.
- **Stage C — autonomous return:** separate operational energy plus one-way TASK→CHARGER switching.
- **Stage D — online adaptation:** planned target-domain updates from real 16×16 charger returns; not yet implemented.

Run a small Stage A smoke with a fresh output directory:

```bash
PYTHONPATH=. /home/zjl/mappo/.venv/bin/python scripts/run_stage_a_energy_bootstrap.py \
  --seed 0 \
  --checkpoint artifacts/phase1_sb3_sac_1m_gpu/seed0/checkpoint_step_1000000.zip \
  --output-dir /tmp/stage_a_smoke \
  --sorties 10 --task-prefix-max-steps 12 \
  --max-return-steps 500 --training-updates 10 --device cpu
```

This command is a smoke test, not a formal experiment.

## Historical Provenance

`DERIVATION_PACKAGE.md`, `docs/theory/`, and pre-migration CCFA reports are explicitly marked `LEGACY / SUPERSEDED`. They are retained for research provenance only and are not active method authority.

## Current Scientific Status

The prior general-theory routes remain blocked and are not being extended. Current work is empirical systems engineering driven by staged energy experiments. A same-data TD repair restored B2/B3 value scale and terminal semantics, but B0 remains best and B3 upper quantiles remain severely under-covered. Stage A is not ready for a formal run or 16x16. See `docs/experiment_analysis/ENERGY_ROUTE_REFACTOR_AUDIT.md`, `docs/experiment_analysis/STAGE_A_IMPLEMENTATION_STATUS.md`, `docs/experiment_analysis/STAGE_A_VALIDATION_RESULTS.md`, and `docs/experiment_analysis/STAGE_A_TD_REPAIR_RESULTS.md`.
