# Stage A Implementation Status

Date: 2026-08-14

## Status

`STAGE_A_CODE_COMPLETE = TRUE`

`STAGE_A_VALIDATION_RUN_COMPLETED = TRUE`

`STAGE_A_FORMAL_EXPERIMENT_RUN = FALSE`

`LEARNED_MODEL_ADDS_VALUE = FALSE`

`READY_FOR_16X16 = FALSE`

Stage A now collects completed charger-return trajectories after varied task prefixes and trains B0-B3 from sortie-separated data. The initial 10-sortie smoke was followed by a 150-sortie validation run. A same-data TD repair subsequently restored B2 and B3 value scale and terminal semantics, but B0 still outperforms every learned model and B3 upper quantiles remain severely under-covered. See `STAGE_A_VALIDATION_RESULTS.md` for the original validation and `STAGE_A_TD_REPAIR_RESULTS.md` for the repair audit.

## Implemented Supervision

Each reset samples a start, task goal, and charger in the unchanged 4×4×2 open world. Frozen SAC follows task goals for a scheduled prefix spanning short to long durations. The experiment then changes the active goal to the charger and records the complete return.

Every charger-return transition stores:

- 77-dimensional charger-conditioned state;
- 3-dimensional frozen-SAC action;
- next state;
- telemetry per-step energy;
- accumulated return energy;
- exact return-energy-to-go label;
- Euclidean charger distance;
- step and remaining path length;
- operational energy/SOC before and after the transition;
- task-prefix length and trajectory context.

Only completed charger returns enter estimator training. Splits are performed by whole sortie before transition expansion.

## Methods

- B0: Euclidean distance × train-split empirical energy/m;
- B1: Monte-Carlo return regression;
- B2: scalar TD critic with gamma=1;
- B3: monotone quantile TD critic at 0.50/0.90/0.95/0.99.

The runner writes immutable config, checkpoint SHA-256, code/Git hash, exact split, raw JSONL hash, model hashes, result JSON, and `COLLECTION_COMPLETED`/`COMPLETED` or `FAILED` markers.

## Smoke Evidence

Command:

```bash
PYTHONPATH=. /home/zjl/mappo/.venv/bin/python scripts/run_stage_a_energy_bootstrap.py \
  --seed 0 \
  --checkpoint artifacts/phase1_sb3_sac_1m_gpu/seed0/checkpoint_step_1000000.zip \
  --output-dir /tmp/ea_mappo_stage_a_smoke_20260814_02 \
  --sorties 10 \
  --task-prefix-min-steps 0 \
  --task-prefix-max-steps 12 \
  --max-return-steps 500 \
  --operational-energy-capacity 3.0 \
  --training-updates 10 \
  --device cpu
```

Observed smoke facts:

| Item | Value |
|---|---:|
| Completed / censored returns | 10 / 0 |
| Return-to-go supervision transitions | 357 |
| Prefix range / unique lengths | 0–12 / 10 |
| Train / calibration / test sorties | 6 / 2 / 2 |
| Unique charger positions | 10 |
| Commitment operational SOC range | 0.9508–1.0000 |
| Return-energy range | 0.2812–0.7007 |
| Model checkpoints reload successfully | yes |
| Formal-result eligible | no |

The smoke establishes executable data and training plumbing only. Ten optimization updates and two test sorties cannot rank B0–B3.

## Validation Evidence

The completed validation artifact is:

`artifacts/energy_transfer/stage_a_validation_seed0_20260814_v2/`

Key facts:

| Item | Value |
|---|---:|
| Completed / censored returns | 150 / 0 |
| Return-to-go supervision transitions | 5,712 |
| Prefix range / unique lengths | 0-160 / 150 |
| Train / calibration / test sorties | 90 / 30 / 30 |
| Train / calibration / test transitions | 3,412 / 1,170 / 1,130 |
| B0 held-out MAE | 0.05509 |
| Best learned held-out MAE (B1) | 0.06140 |
| B2 held-out MAE | 0.12098 |
| B3 q0.50 held-out MAE | 4.92040 |
| Data coverage sufficient | yes |
| Sortie leakage | none |
| Formal-result eligible | no |

The learned models do not establish extra value over B0. In particular, B3's 100% empirical coverage at every quantile is caused by extreme overprediction and is not reasonable calibration. No formal run was launched because the validation did not pass the success gate.

## TD Repair Follow-Up

The repair reused the exact raw trajectories and split without recollection. With the same 800-update budget, B2 MAE changed from `0.12098` to `0.06365` and B3 median MAE from `4.92040` to `0.07131`. Terminal MAE changed from `0.07056` to `0.01725` for B2 and from `4.31169` to `0.02809` for B3.

B3 q0.50 coverage is now `44.69%`, so the vacuous all-100% scale collapse is gone. Its q0.90/q0.95/q0.99 coverage is only `52.12%/55.40%/59.91%`; distributional safety semantics remain unresolved. B0 MAE remains best at `0.05509`. Therefore `LEARNED_MODEL_ADDS_VALUE` and `READY_FOR_16X16` remain false.

## Stage C Interface

`decide_managed_goal` forms a charger-conditioned observation without changing TASK mode, obtains the charger action from frozen SAC, queries a return-energy predictor, and applies `EnergySwitchController` using the separate operational energy account. At the boundary it mutates the active navigation goal to charger exactly once; commitment is absorbing until a new sortie.

## Before Any Formal or 16x16 Experiment

Still required:

1. do not treat repaired B3 quantiles as upper bounds while q0.90/q0.95/q0.99 remain under-covered;
2. decide whether the distance-dominated 4x4 setting warrants any further Energy Critic work before a formal multi-seed run;
3. establish stable held-out value over B0 before approving a formal scale;
4. only after the complete Stage A gate passes, consider 16x16 competence, operational capacity, switching baselines, and target-domain adaptation;
5. continue to defer obstacles and Collision Module.
