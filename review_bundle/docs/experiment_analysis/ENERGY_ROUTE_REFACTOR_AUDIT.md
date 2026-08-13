# Energy Route Refactor Audit

Date: 2026-08-14

## Fixed Research Route

The active engineering route is fixed:

1. frozen 4×4×2 goal-conditioned SAC performs navigation only;
2. the Energy Module learns charger-return energy from completed telemetry trajectories;
3. a separate operational-energy account decides TASK continuation versus one-way CHARGER commitment;
4. 4×4 Stage A bootstraps the estimator;
5. a separate 16×16 target environment will test autonomous return and later online adaptation;
6. Collision Module remains unchanged in this phase.

No SAC retraining, reward modification, checkpoint modification, collision expansion, or new theory work is part of this refactor.

## Conflict Audit

| Current source | Conflict with route | Resolution |
|---|---|---|
| `envs/navigation/environment.py` | Station was fixed to the scenario and external goal changes used private assignment. | Preserve defaults; add reset-time random/configured station, counterfactual goal observation, and explicit external-goal handoff. |
| `NavigationEnv.state.energy` | 1000-unit energy is embedded in the 77-dimensional checkpoint contract. | Keep unchanged; introduce independent operational-energy accounting. |
| `experiments/new_route/e1_energy.py` | Executes one task action and immediately commits to charger; no prefix diversity. | Keep as `SUPERSEDED` provenance; new Stage A is independent. |
| old E1 artifacts | Partial seed-0 data and stale process markers cannot support comparison. | Preserve unchanged and excluded; never import them into Stage A. |
| `safety/switching/commitment.py` | Boolean feasibility latch lacks the numerical energy boundary. | Retain it; add a dedicated `EnergySwitchController`. |
| energy critics | Models and gamma=1 targets exist, but no valid staged collector/trainer feeds them. | Reuse the model primitives in a new Stage A pipeline. |
| target environment | No separate 16×16 scenario exists. | Add an open 16×16 scenario while leaving the 4×4 scenario untouched. |
| Collision Module | Present but outside current priority. | Keep unchanged; no new collision code or experiment. |

## File Classification

### KEEP

- `artifacts/phase1_sb3_sac_1m_gpu/**`: frozen 1M navigation checkpoints and provenance.
- `agents/goal_conditioned_sac/**`: frozen policy adapter.
- `safety/energy/accounting.py`, `critics.py`, `td.py`: energy supervision and model primitives.
- `safety/collision/**`: unchanged and inactive for Stage A.
- all historical E1 artifacts and experiment-analysis records.

### MODIFY

- `envs/navigation/environment.py`: configurable charger and public goal-query/handoff APIs without default-contract changes.
- `envs/navigation/__init__.py`: export operational-energy layer.
- `safety/energy/__init__.py`: export checkpoint predictors.
- `safety/switching/__init__.py`: export energy boundary controller.
- `experiments/new_route/e1_energy.py`: explicit superseded marker only.
- `experiments/new_route/provenance.py`: include the new experiment package in code hashing.
- `scripts/launch_new_route_e1.sh`: fail closed instead of relaunching the invalid protocol.
- `README.md`: document the staged energy route.

### ADD

- `envs/navigation/operational_energy.py`: finite research-energy accounting separated from SAC observation energy.
- `envs/navigation/scenarios/target_open_16x16.json`: Stage B open-world target interface.
- `safety/energy/inference.py`: loadable scalar and quantile energy predictors.
- `safety/switching/energy.py`: numerical one-way stopping rule.
- `experiments/energy_transfer/`: Stage A collection/training and Stage C managed-goal interface.
- `scripts/run_stage_a_energy_bootstrap.py`: immutable Stage A runner.
- focused tests for operational energy, bootstrap data, inference, switching, and 16×16 compatibility.

### SUPERSEDED, NOT DELETED

- `experiments/new_route/e1_energy.py`;
- `scripts/launch_new_route_e1.sh`;
- `artifacts/new_route/e1_energy/**`.

## Minimal Architecture

```text
FrozenGoalConditionedSAC
    receives unchanged 77-D observation and active goal
                |
NavigationEnv --+--> telemetry per-step cost
                |
OperationalEnergyWrapper
    separate finite remaining_energy / SOC
                |
Energy critic on charger-conditioned (state, action)
                |
EnergySwitchController
    TASK if remaining > predicted_return + reserve
    CHARGER_COMMITTED otherwise, absorbing within sortie
```

The Energy Module never enters the SAC reward and never emits SAC actions.
