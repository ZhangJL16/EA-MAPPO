# Grounded Memory Real Diagnostic

## Protocol

- Frozen navigation SAC checkpoint SHA-256: `df9a4354420a1a043b60d704292c6fe06ddd2127e937d3d9c72278b287c1ff0a`.
- Accepted dataset: `artifacts/grounded_memory_real_dataset_20260820_v4/`.
- Controlled transitions: **100,122**.
- Accepted trajectories: **180**; rejected feasibility-precondition trajectories: **16**, recorded separately.
- Families: static, constant velocity, accelerating, turning, abrupt change, dropout, dense multi-obstacle, and boundary interaction.
- Split: whole trajectory, 108 train / 36 validation / 36 test.
- Accepted slice: 0 collision steps, 0 infeasible steps, 0 unsafe fallback steps.
- Training seeds: 11, 23, 37.

Earlier `v1`-`v3` artifacts are explicitly marked invalid for hard-safety comparison because they retained collision or infeasible fallback trajectories.

## Certified Contraction

Artifact: `artifacts/grounded_memory_contraction_20260820_v2/contraction_diagnostic.json`.

Evaluation uses 500 held-out whole-trajectory snapshots. All methods use the same base bounds and analytic set-membership verifier.

| Method | Containment | Geometric volume ratio to C0 | Mean width sum | Accepted constraints | Mean compute |
|---|---:|---:|---:|---:|---:|
| C0 current only | 1.000 | 1.000000 | 230.168 | 0.000 | 16.54 ms |
| C1 fixed history | 1.000 | 0.000158 | 61.431 | 3.466 | 18.66 ms |
| C2 all history | 1.000 | 0.000101 | 56.344 | 14.220 | 19.74 ms |
| C3 generic GRU + verifier | 1.000 | 0.016264 | 100.179 | 3.952 | 18.62 ms |
| C4 FOGM + verifier | 1.000 | 0.016264 | 100.179 | 3.952 | 18.59 ms |
| C5 oracle valid-history | 1.000 | 0.016264 | 100.179 | 3.952 | 18.52 ms |

All methods had zero false accepted constraints. The simple all-history intersection is both the tightest and only 1.15 ms slower than FOGM on average. Generic GRU, FOGM, and the oracle select the same constraints on this slice. Therefore grounding does shrink the certified set, but **learned structured grounding adds no contraction or practical compute advantage over simple verified baselines**.

## True Route Labels

Labels are backward/future calculations from the executed safety-filtered trajectory:

- previous and future avoidance side;
- future safe path length and detour ratio;
- future goal progress;
- future intervention count;
- freeze indicator;
- corridor class;
- curvature;
- time to clear the obstacle.

They do not reconstruct LiDAR and do not use synthetic proxy trajectories.

### Offline prediction

Held-out test samples: 16,000.

| Method | Route accuracy, mean over seeds | Macro-F1 tendency |
|---|---:|---|
| Current safety state | **0.8059** | best |
| Generic GRU history | 0.7563 | second |
| FOGM route/safety/energy | 0.6721 | worst learned method |

FOGM route accuracy is below generic GRU in all three seeds by 0.0753, 0.0849, and 0.0923.

### Matched closed loop

Each method uses the same exact-state sampled-data HOCBF. There are 48 rollouts per method.

| Method | Success | Collision steps | Mean path ratio | Mean intervention | Route switches | Mean energy |
|---|---:|---:|---:|---:|---:|---:|
| No route memory | 1.000 | 0 | **1.01037** | 2.4529 | 0.000 | 5.9867 |
| Explicit side hysteresis | 1.000 | 0 | 1.01498 | **2.2283** | **3.375** | **5.9822** |
| Generic GRU | 1.000 | 0 | 1.01868 | 2.4694 | 9.563 | 6.1157 |
| FOGM route | 1.000 | 0 | 1.01856 | 2.4339 | 16.500 | 6.1387 |

FOGM is not a downstream winner: it has worse path and energy than no memory, and far more switching than explicit hysteresis. Its zero infeasible-filter steps are not unique; explicit hysteresis also has zero and has better intervention and energy.

## True Energy Labels

The dataset records:

- `E_nominal`: telemetry energy for the nominal SAC action;
- `E_safe`: telemetry energy for the executed safety-filtered action;
- `E_overhead = E_safe - E_nominal`;
- future 40-step overhead;
- terminal Monte-Carlo `E_safe` energy-to-go on successful trajectories.

### Held-out energy-to-go

Mean MAE over three seeds:

| Method | Safe Energy-to-Go MAE |
|---|---:|
| Distance/velocity linear baseline | **0.26743** |
| Current safety-state MLP | 0.43620 |
| Generic GRU history | 0.50444 |
| Existing MC Energy-to-Go network | 0.55708 |
| FOGM route/safety/energy | 0.68027 |

FOGM is worse than generic GRU in all three seeds by 0.1046, 0.1952, and 0.2277 MAE.

FOGM does improve the short-horizon 40-step overhead MAE in all three seeds (approximately 0.0183-0.0192 versus generic 0.0215-0.0231). This is a real but narrow auxiliary-prediction effect. The overhead target itself has mean `-0.0326` and standard deviation `0.0600`, and the improvement does not translate to path, intervention, or total-energy gains. Closed-loop energy ranking was therefore not activated.

## Three-Seed Direction

- Certified contraction: C4 equals C3 and C5 in 3/3 seeds; C2 is tighter.
- Route prediction: FOGM worse than generic in 3/3 seeds.
- Safe Energy-to-Go: FOGM worse than generic in 3/3 seeds.
- H40 overhead prediction: FOGM better than generic in 3/3 seeds, but with no downstream control effect.
- Closed-loop path/energy: FOGM is not the best method in any seed.

## Diagnostic Conclusion

The two missing bridges were tested rather than assumed:

1. Historical grounding can produce a smaller valid set, but simple set-membership intersection dominates learned proposal selection.
2. Real Route/Energy labels can be learned, but FOGM does not yield downstream route or mission-energy improvement.

The result does not meet the long-run gate.

