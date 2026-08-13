# Data-Driven Dual-Timescale Safety for Persistent UAVs

This directory contains the reconstructed research route. The active method learns two non-scalarized objects: short-horizon collision risk and post-action energy-to-charger return distributions under a shared goal-conditioned navigation policy.

## Active Sources

- Theory and claim boundaries: `docs/new_theory/`
- Navigation baseline environment: `envs/navigation/`
- Safety models: `safety/collision/`, `safety/energy/`, `safety/calibration/`, `safety/switching/`
- Frozen goal-conditioned policy adapter: `agents/goal_conditioned_sac/`
- New experiments: `experiments/new_route/`
- Tests: `tests/new_route/`

## Retained Baseline

`artifacts/phase1_sb3_sac_1m_gpu/` is the completed three-seed Standard SAC navigation baseline. Its checkpoint contract is 77 observation dimensions and 3 continuous actions. The large resource budget and telemetry-cost feature are retained only for checkpoint compatibility; the artifact does not establish finite-energy safety.

Run the focused baseline and safety tests:

```bash
PYTHONPATH=. /home/zjl/mappo/.venv/bin/python -m pytest tests/new_route -q
```

The first formal energy-estimation experiment was launched before the final theory audit. Inspect it only in a results-analysis session:

```bash
python scripts/check_new_route_runs.py
```

Do not relaunch into existing immutable run directories and do not interpret incomplete runs.

## Historical Provenance

`DERIVATION_PACKAGE.md`, `docs/theory/`, and pre-migration CCFA reports are explicitly marked `LEGACY / SUPERSEDED`. They are retained for research provenance only and are not active method authority.

## Current Scientific Status

The conditional theory is coherent, but the final blind panel found fatal novelty overlap with Budgeted MDP, distributional constrained RL, off-policy prediction, and selective-label learning. The theory route is marked `RESEARCH_DIRECTION_BLOCKED`; see `docs/new_theory/RESEARCH_DIRECTION_BLOCKED.md`. `THEORY_ICLR_READY` and `PAPER_ICLR_READY` are both false. E1 cannot resolve this blocker because it studies only the collision-free energy-estimation special case.
