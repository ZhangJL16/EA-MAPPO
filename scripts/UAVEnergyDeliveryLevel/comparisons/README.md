# UAVEnergyDeliveryLevel Comparison Runs

All scripts use `.venv/bin/python` by default and write outputs under ignored
local artifact directories:

- Logs: `logs/uav_energy_delivery_comparisons/<run_name>/`
- Models: `model_runs/comparisons/<run_name>/`

Pass any normal training flags after the script name. For example:

```bash
scripts/UAVEnergyDeliveryLevel/comparisons/train_ours_full.sh --gpu_id 0 --n_steps 600000
```

Set `DRY_RUN=1` to print the command without launching training.

## Implemented Training/Evaluation Files

| Method | Script | Repository implementation |
|---|---|---|
| Greedy + Threshold-Charge | `train_greedy_threshold_charge.sh` | heuristic evaluator |
| Energy-aware Greedy | `train_energy_aware_greedy.sh` | heuristic evaluator |
| Auction + Threshold-Charge | `train_auction_threshold_charge.sh` | heuristic evaluator + environment auction |
| IPPO | `train_ippo.sh` | `main.py`, `policy/ippo.py` local-observation critics |
| MAPPO | `train_mappo.sh` | `main.py`, `policy/mappo.py` |
| MAPPO-Lagrangian | `train_mappo_lagrangian.sh` | `main.py`, `policy/mappo_lagrangian.py` |
| MAPPO + Safety Shield | `train_mappo_safety_shield.sh` | `main.py`, `policy/mappo.py`, `policy/mappo_safety.py` |
| MADDPG | `train_maddpg.sh` | `main.py --alg RGMComm`, `policy/maddpg.py` |
| MATD3 | `train_matd3.sh` | `main.py --alg MATD3RGMComm`, `policy/matd3.py` |
| MACPO | `train_macpo.sh` | `main.py`, `policy/macpo.py` |
| HSD | `train_hsd.sh` | HSD-style generic hierarchical baseline on `main_level.py` |
| Hierarchical MAPPO | `train_hmappo_basic.sh` | `main_level.py`, `level_policy/mappo.py` |
| H-MAPPO without Energy-aware Design | `train_hmappo_wo_energy_aware_design.sh` | `main_level.py`, `level_policy/mappo.py` |
| Ours w/o Hierarchy | `train_ours_wo_hierarchy.sh` | flat `mappo_safe` substitute |
| Ours w/o Energy Constraint | `train_ours_wo_energy_constraint.sh` | current H-MAPPO with energy shield/loss disabled |
| Ours w/o Charging Resource Modeling | `train_ours_wo_charging_resource_modeling.sh` | current H-MAPPO with charge queue disabled |
| Ours w/o Safety Layer | `train_ours_wo_safety_layer.sh` | current H-MAPPO with safety guard disabled |
| Ours w/o Auction Module | `train_ours_wo_auction_module.sh` | current H-MAPPO with auction disabled and local greedy order selection |
| Ours w/o High-level Temporal Abstraction | `train_ours_wo_high_level_temporal_abstraction.sh` | current H-MAPPO with `hmappo_meta_period=1` |
| Ours Full | `train_ours_full.sh` | current full method |

## Notes

`HSD` is implemented as a runnable HSD-style hierarchical baseline because this
repository does not contain a separate published HSD implementation. It removes
the proposed energy-aware, auction, charging-resource, and safety modules while
keeping generic hierarchical subgoal learning.
