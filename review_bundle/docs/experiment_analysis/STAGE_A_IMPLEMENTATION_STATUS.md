# Stage A Implementation Status

Date: 2026-08-14

## Status

`STAGE_A_CODE_COMPLETE = TRUE`

`STAGE_A_FORMAL_EXPERIMENT_RUN = FALSE`

Stage A now collects completed charger-return trajectories after varied task prefixes and trains B0–B3 from sortie-separated data. Only a 10-sortie smoke was run; its estimator scores are not paper results.

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

## Stage C Interface

`decide_managed_goal` forms a charger-conditioned observation without changing TASK mode, obtains the charger action from frozen SAC, queries a return-energy predictor, and applies `EnergySwitchController` using the separate operational energy account. At the boundary it mutates the active navigation goal to charger exactly once; commitment is absorbing until a new sortie.

## Before 16×16 Experiments

Still required:

1. choose physically interpretable 16×16 operational capacity, reserve, and telemetry scaling;
2. measure whether the 4×4 frozen policy can reliably reach long-distance 16×16 open-world goals—dimensional compatibility alone is not navigation competence;
3. implement the Stage B/C rollout runner and switching baselines: fixed SOC, distance, learned scalar, learned quantile;
4. define target-domain replay and update cadence for Stage D without train/eval leakage;
5. preregister fixed target train/calibration/test seeds and multi-seed budgets;
6. run only smoke tests before any formal three-seed launch;
7. defer obstacles and Collision Module until open-world energy adaptation is causally resolved.
