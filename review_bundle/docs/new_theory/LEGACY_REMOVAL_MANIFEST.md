# Legacy Removal Manifest

Status: ACTIVE MIGRATION MANIFEST
Repository root: `review_bundle/` in `ZhangJL16/EA-MAPPO`
Route decision: the persistent certified-control, recoverability, and Generator-safety route is superseded and is not an active dependency.

## Classification Rules

- `DELETE`: remove executable code, tests, active documentation, scripts, and artifacts whose meaning depends on the abandoned route.
- `KEEP_BASELINE`: retain the completed Standard SAC navigation artifact and the minimum code/config/scripts needed to load and reproduce it.
- `KEEP_LEGACY_THEORY`: retain research-provenance documents only after marking them `LEGACY / SUPERSEDED`; no active code, test, README, or new theorem may cite them as authority.
- `REFACTOR_BASELINE_DEPENDENCY`: move a baseline dependency into a concept-clean module, preserve checkpoint/observation/action/reward/seed semantics, then delete its old certified-UAV source.

## Code and Runtime

| Path | Action | Reason / replacement |
| --- | --- | --- |
| `cert_runtime/**` | DELETE | Certified execution, zonotope/Generator geometry, watchdog, WCET, support action, trainer, and authority runtime belong exclusively to the superseded route. |
| `certificates/**` | DELETE | Frozen persistent/2x certificate payloads are stale route artifacts. |
| `learned_fields/**` | DELETE | Dual learned proposal fields and proposal ranker implement the abandoned collision/recovery-energy field design. |
| `calibration/**` | DELETE | Existing calibration package calibrates certified plant/runtime bounds; new statistical calibration is implemented under `safety/calibration/`. |
| `experiments/**` | DELETE | Existing runners, agents, and first-round configs evaluate Generator/certified-control experiments. New experiments live under `experiments/new_route/`. |
| `envs/certified_uav/sb3_persistent_navigation.py` | REFACTOR_BASELINE_DEPENDENCY | Extract the phase-1 checkpoint-compatible navigation environment to `envs/navigation/`; do not retain finite-energy/charging subclasses. |
| `envs/certified_uav/config.py` | REFACTOR_BASELINE_DEPENDENCY | Extract only dynamics/world constants needed by the navigation baseline. |
| `envs/certified_uav/dynamics.py` | REFACTOR_BASELINE_DEPENDENCY | Extract double-integrator dynamics without certification language. |
| `envs/certified_uav/energy.py` | REFACTOR_BASELINE_DEPENDENCY | Preserve only the non-binding synthetic per-step telemetry formula needed for checkpoint reward/observation regression; no safety semantics. |
| `envs/certified_uav/lidar.py` | REFACTOR_BASELINE_DEPENDENCY | Extract local sensing without certificate-ray conversion. |
| `envs/certified_uav/obstacles.py` | REFACTOR_BASELINE_DEPENDENCY | Extract geometric simulation primitives as environment dynamics, not certified safety. |
| `envs/certified_uav/scenario.py` and baseline JSON inheritance | REFACTOR_BASELINE_DEPENDENCY | Replace certificate-heavy scenario parser with clean navigation scenario files while preserving `random_persistent_open.json` values. |
| `envs/certified_uav/state.py` | REFACTOR_BASELINE_DEPENDENCY | Extract simulator state under neutral naming. |
| `envs/certified_uav/__init__.py` | DELETE | Public API exposes the superseded certificate/recovery/charging architecture. |
| `envs/certified_uav/acceptance.py`, `actuator.py`, `adapters.py`, `charging.py`, `mission_certificate.py`, `persistent_certificate.py`, `persistent_task.py`, `persistent_wrapper.py`, `plant_env.py`, `random_persistent.py`, `recoverability.py`, `recovery_atlas.py`, `runtime_wrapper.py`, `task_wrapper.py`, `telemetry.py`, `terminal.py` | DELETE | Certified execution, charging/recovery lifecycle, authority, and terminal-certificate implementation. |
| `envs/certified_uav/scenarios/**`, `envs/certified_uav/persistent_scenarios/**` | DELETE after extraction | Old scenario schema carries corridor/certificate/recovery fields; retain only clean new-route scenario data. |

## Scripts

| Path | Action | Reason / replacement |
| --- | --- | --- |
| `scripts/sb3_navigation_harness.py` | REFACTOR_BASELINE_DEPENDENCY | Change imports to `envs.navigation`; preserve phase-1 evaluation semantics. |
| `scripts/train_sb3_navigation_baseline.py`, `train_sb3_persistent_sac.py`, `run_one_sb3_sac_1m_gpu.sh`, `launch_sb3_sac_1m_gpu.sh`, `evaluate_sb3_sac_gif.py`, `summarize_sb3_navigation_baselines.py`, `setup_uv_env.sh` | KEEP_BASELINE | Minimum phase-1 training/loading/evaluation/provenance entry points; names that imply persistence are refactored where practical. |
| `scripts/train_sb3_ddpg_navigation.py`, `train_sb3_ppo_navigation.py`, `run_one_sb3_ddpg_1m.sh`, `run_one_sb3_ppo_1m.sh`, `launch_sb3_ddpg_1m.sh`, `launch_sb3_ppo_1m.sh`, `launch_sb3_ppo_ddpg_1m.sh` | DELETE | Non-SAC comparison scripts are not required to reproduce the retained Standard SAC artifact. |
| `scripts/*energy*`, `scripts/*recovery*`, `scripts/*certificate*`, `scripts/*certified*`, `scripts/*persistent*`, `scripts/*kappa*`, `scripts/*generator*`, `scripts/*authority*`, `scripts/*two_x*`, `scripts/*2x*` except the explicitly retained baseline entry | DELETE | Old finite-energy, recovery, certificate, Generator, teacher, authority, and 2x routes. |
| Other old audit/comparison/table scripts | DELETE | Their inputs and metrics are defined by removed route artifacts. New route receives dedicated scripts. |
| `scripts/check_new_route_runs.py`, `evaluate_new_route.py`, `build_new_route_tables.py`, `run_ccfa_review_after_results.py` | NEW | Future-session completion, evaluation, tables, and results-aware CCFA review entry points. |

## Tests

| Path | Action | Reason / replacement |
| --- | --- | --- |
| `tests/test_sb3_persistent_navigation.py`, `test_sb3_sac_training_protocol.py`, relevant SAC checkpoint loading checks | REFACTOR_BASELINE_DEPENDENCY | Move only phase-1 observation/action/reward/seed and checkpoint compatibility assertions to `tests/new_route/`. |
| All existing `tests/test_*.py` after baseline assertions are extracted | DELETE | Current suite tests certificate, Generator, recovery, lifecycle, dual fields, old energy navigation, or obsolete experimental tooling. |
| `tests/new_route/**` | NEW | TD targets, SSP terminal, quantiles, calibration, energy accounting, risk budget, commitment, determinism, checkpoint loading, and baseline regression. |

## Artifacts

| Path | Action | Reason |
| --- | --- | --- |
| `artifacts/phase1_sb3_sac_1m_gpu/**` | KEEP_BASELINE | Completed three-seed Standard SAC navigation competence baseline and checkpoints. Do not overwrite or rerun 1M by default. |
| Every other existing directory/file under `artifacts/` | DELETE | Old Generator/certified/recovery/energy/random-persistent/smoke/theory/runtime artifacts are stale for the new route. |
| `artifacts/new_route/**` | NEW | Immutable configs, raw JSON/JSONL, hashes, logs, PIDs, and run markers for the new route. |

## Documentation

| Path | Action | Reason |
| --- | --- | --- |
| `DERIVATION_PACKAGE.md` | KEEP_LEGACY_THEORY | Historical derivation provenance; mark superseded. |
| `docs/theory/**` | KEEP_LEGACY_THEORY | Preserve proof/red-team/mock-review provenance only; prepend an explicit legacy banner to every retained file. |
| Existing `ccfa-review-reports/**` | KEEP_LEGACY_THEORY | Historical review provenance only; directory-level legacy notice required. |
| `README.md` | REWRITE | Must describe only the new route and retained baseline, while linking old theory solely as superseded provenance. |
| `docs/new_theory/**` | NEW | Sole active theory source for Data-Driven Dual-Timescale Safety. |

## Dirty-Tree Handling

Before migration, 17 modified files were detected, all in old theory or superseded persistent certificate/recovery/lifecycle code. Theoretical edits are retained as legacy provenance; executable/test edits are removed under the user's destructive-migration authorization. Git history remains untouched.

## Baseline Gate Before Deletion

The old certified-UAV package may be deleted only after:

1. the phase-1 clean environment reproduces observation and action spaces;
2. seeded reset and transition outputs match the old implementation on the retained open scenario;
3. reward components match under the saved baseline configuration;
4. `SAC.load` reads all three 1M checkpoints;
5. a short deterministic held-out rollout succeeds without retraining.
