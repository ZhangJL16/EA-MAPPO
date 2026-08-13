# E1 Provenance Audit

Date: 2026-08-13

## Verdict

`E1_PROVENANCE_VALID = FALSE`

`E1_INVALID_RESTART_REQUIRED = TRUE`

All three configured runs are excluded from formal comparison. The one available partial file is retained only as an engineering and protocol diagnostic.

## Process and Artifact Snapshot

The process state was checked once. No E1 PID was alive at that snapshot.

| Seed | Recorded PID | Marker | Raw sorties | Log | E1 checkpoints | Result/summary | Formal status |
|---:|---:|---|---:|---:|---|---|---|
| 0 | 3413153 | stale `RUNNING` | 43/100 | 0 bytes | none | none | INVALID |
| 1 | 3413204 | stale `RUNNING` | 0/100 | 0 bytes | none | none | INVALID |
| 2 | 3413280 | stale `INITIALIZED` | 0/100 | 0 bytes | none | none | INVALID |

There is no `COMPLETED.json` or `FAILED.json` for any seed. The Python runner writes `FAILED.json` for caught exceptions, so the absence of that marker is consistent with external process termination, but the exact terminating signal is not recoverable from the empty logs. It is not evidence of a scientific result.

## Configuration Consistency

The three immutable configs agree on all intended common fields:

- seeds: 0, 1, 2;
- 100 collection sorties per seed;
- 800 maximum charger-return steps;
- 800 estimator updates;
- frozen seed-matched Standard SAC checkpoint at step 1,000,000;
- CPU estimator execution;
- sortie-level 60/20/20 train/calibration/test split;
- undiscounted energy target, `gamma_energy = 1.0`;
- conformal level `alpha_energy = 0.05`;
- Git SHA `70a89569204bc037cbd984732fdc64727c75b288`;
- code hash `021d221845db64e620e6a68235b1ccd3352a25309df01ea7842ab2fed847e852`.

The current aggregate code hash still equals the recorded hash. Each baseline checkpoint SHA-256 also matches its config:

| Seed | Checkpoint SHA-256 |
|---:|---|
| 0 | `56f50be556888a2b567ea76b3c9726319187802e1327a78db48082de195c6fce` |
| 1 | `5b5b81b67817fd928c460ecd1b4b3788804ad1fdc748cf7a7a1e02556f3f9520` |
| 2 | `c23d3b0d9fa557ecdb842720a0dbc1e9a709aa88b254bdc574525bda43b323ac` |

No artifact overwrite occurred: initialization refuses a nonempty output directory. There is no evidence of checkpoint corruption, code drift, NaN, divergence, disk exhaustion, or a caught Python exception. Because training never produced metrics or checkpoints, absence of NaN evidence must not be read as evidence of stable training.

## Scientific Protocol Failure

The interrupted process is not the only problem. The configured experiment cannot answer the declared finite-energy and switching questions:

1. `NavigationEnv` used its default `navigation_energy_capacity = 1000.0`, whose documented semantics are a nonterminating checkpoint-compatibility budget, not finite-resource operation.
2. Every episode executes one task-goal action and then immediately replaces the goal with the charger. There is no learned stopping time and no task-continuation phase.
3. No fixed-SOC, distance-threshold, learned-threshold, or calibrated-threshold switching comparison is implemented.
4. The config does not persist the exact train/calibration/test sortie indices or a separate fixed switching-evaluation seed stream.
5. Environment factors needed to interpret energy heterogeneity—wind, payload, aging, vertical regime, and policy version changes—are absent or fixed.
6. The result schema omits median error, relative error, severe/worst underestimation, conditional errors, bound slack, and switching outcomes requested for E1.

Therefore a simple engineering relaunch of the same command would remain scientifically invalid.

## Partial Seed-0 Diagnostic

These numbers describe the 43 completed return trajectories already written before interruption. They are not a train/test comparison and cannot select a method.

| Diagnostic | Value |
|---|---:|
| Completed / censored trajectories | 43 / 0 |
| Return steps, mean / median / range | 38.19 / 37 / 9–57 |
| Return energy, mean / median / range | 0.4735 / 0.4603 / 0.1092–0.7127 |
| Successor charger distance, mean / range | 2.2718 / 0.2998–3.8594 m |
| Return path length, mean / range | 2.1143 / 0.1354–3.7593 m |
| Initial SOC | 1.0000 for every trajectory |
| Final SOC, mean / range | 0.9995265 / 0.9992873–0.9998908 |
| Corr(return energy, Euclidean distance) | 0.9603 |
| Corr(return energy, path length) | 0.9624 |

The almost constant SOC confirms that the collected data do not exercise finite-energy behavior. The high distance correlation is a reason to include a strong distance baseline, not evidence that it matches or beats TD.

## Required Corrected Protocol

Before restart, E1-v2 must preregister and test:

- a finite battery capacity and initial-SOC distribution that actually exercises return margins;
- task prefixes before charger commitment, with no obstacle complexity;
- exact disjoint collection, calibration, prediction-test, and switching-evaluation seeds;
- the five estimator methods and a single shared data budget;
- fixed-SOC and distance switching baselines;
- conditional tail metrics and one-way switching telemetry;
- progress/checkpoint markers and signal-safe detached execution;
- fresh immutable output directories, never mixed with the invalid E1 artifacts.

E2 and E3 must remain stopped until a corrected E1 protocol passes tests and produces valid multi-seed evidence.
