# Frozen single-life protocol v1 results

Integrity: 47000/47000 unique jobs, no missing/duplicates.

Primary delta = 0.05. Lifetime safety is a theorem under the stated model assumptions; finite-horizon frequencies and Wilson intervals are calibration checks, not proofs. Pooled heterogeneous-cell intervals are descriptive. Uncertified lives enter restricted means at their horizon, including catastrophe.

| Suite | Method | Runs | Catastrophe | Certified | Mean regret | Mean exploration |
|---|---|---:|---:|---:|---:|---:|
| binary | Oracle | 2000 | 0 | 2000 | 0.00 | 0.00 |
| binary | RobustOnly | 2000 | 0 | 0 | 16000.00 | 0.00 |
| binary | SafeRefine | 2000 | 58 | 1600 | 3961.73 | 238.23 |
| binary | StaticSafeID | 2000 | 58 | 1600 | 3961.73 | 238.23 |
| binary | UnsafeMLE | 2000 | 1000 | 0 | 10000.00 | 0.00 |
| nuisance | FullModelID | 500 | 15 | 495 | 2976.89 | 1889.02 |
| nuisance | SafeRefine | 500 | 15 | 495 | 1632.26 | 170.79 |
| random | FullModelID | 5000 | 165 | 4863 | 1891.74 | 982.95 |
| random | Oracle | 5000 | 0 | 5000 | 0.00 | 0.00 |
| random | SafeRefine | 5000 | 165 | 4863 | 1891.74 | 982.95 |
| random | StaticSafeID | 5000 | 0 | 0 | 24034.26 | 255.77 |
| recursive | Oracle | 1600 | 0 | 1600 | 0.00 | 0.00 |
| recursive | RobustOnly | 1600 | 0 | 0 | 24000.00 | 0.00 |
| recursive | SafeRefine | 1600 | 52 | 1568 | 1273.93 | 385.86 |
| recursive | StaticSafeID | 1600 | 0 | 0 | 24000.00 | 180.42 |
| recursive | UnsafeMLE | 1600 | 1200 | 0 | 22500.00 | 0.00 |
| uav | CertaintyEquivalent | 500 | 0 | 0 | 0.38 | 0.00 |
| uav | Oracle | 500 | 0 | 500 | 0.38 | 0.00 |
| uav | SafeRefine | 500 | 0 | 500 | 0.38 | 0.00 |
| uav | WorstCaseRobust | 500 | 0 | 0 | 0.38 | 0.00 |

No result triggered parameter changes or a follow-on experiment. See cells.json for stratification and results.json for every life, including failures.

UAV observes the full deterministic cycle clock; support filtering can identify theta after one completed safe mission. Gaussian-only KL scaling is not claimed. No task distribution or battery adjustment is permitted if optimal arms coincide.

The matching information bound is established only for the staged Bernoulli witness family in the theory document, not for arbitrary unknown MDPs.
