# Energy Risk Root Cause and Final v5 Solution

## Executive Decision

- Fresh v5 status: **PASS**.
- Formal 500k Phase2: **NOT STARTED**.
- Frozen SAC and MC point estimator were not retrained.
- Selected Goal construction: K2 suffix-future-max positive residual model, compact decision-time context, direct Goal-type x distance Mondrian calibration at 97.50%.
- Selected Mission construction: frozen heteroscedastic Laplace model plus direct distance Mondrian correction at 99.00%.

## Statistical Gate Audit

Raw subgroup empirical coverage >=95% is **not a well-designed sole gate**. At n=231 and true coverage 95%, the probability of observing empirical coverage >=95% is only 0.5122. The v4 worst Goal group had 215/231 covered with exact 95% interval [89.00%, 95.99%].

The preregistered v5 gate therefore combines:

1. direct finite-sample split-conformal construction inside each predefined physical group;
2. fresh empirical stress tests;
3. one-sided binomial undercoverage tests at p=0.95 with Holm family-wise correction.

Failure to reject undercoverage is not proof of arbitrary conditional coverage. The guarantee is predefined finite-group conditional coverage under within-group exchangeability, not arbitrary conditional coverage at every state x.

To avoid calibration-selection bias, model architecture, features, and alpha were frozen before collecting the independent final calibration split (2500 Goal trajectories and 1000 Mission trajectories). These labels fitted only the fixed Mondrian quantiles and were disjoint from train, validation, v2/v4 diagnostic, and fresh v5 data.

## Root Cause

The original 7D state is **partially insufficient for tail risk**, while remaining sufficient for the point estimate. Adding absolute position reduced point MAE from 0.086787 to 0.082379, only 5.08%, below the preregistered 10% materiality threshold. Therefore the original point model was retained.

The deployable missing information is compact position/boundary context and a target aligned with future worst underestimation. K2's predicted risk had Pearson/Spearman correlation 0.550/0.693 with suffix maximum underestimation, stronger than K1's 0.445/0.561. On development data at 97.5% construction, K2 achieved overall 97.35%, worst group 96.42%, and mean width 0.6950.

Post-hoc path ratio, future boundary contact, future steps, and future acceleration were used only for diagnosis and were prohibited as model inputs. The top-100 forensics found long-distance and boundary-interaction concentration, but these realized-future labels are unavailable at decision time.

Short rollout context did not improve the worst group: H=10/25/50 worst coverage was 91.00%/88.00%/87.00%. The deterministic full-rollout oracle was nearly exact (MAE 1.09e-06) but required 0.452s per state and is not the learned deployment estimator.

## Coverage-Efficiency Frontier

The construction level is the declared risk tolerance, not a manually added post-test margin. Development results for the selected K2 method were:

| Construction Coverage | Overall Coverage | Worst Primary Group | Mean Width | P95 Width | Unnecessary-Return Proxy |
|---:|---:|---:|---:|---:|---:|
| 0.900 | 89.20% | 85.71% | 0.4377 | 0.9602 | 0.10% |
| 0.950 | 94.30% | 91.56% | 0.5464 | 1.1731 | 0.13% |
| 0.975 | 97.35% | 96.42% | 0.6950 | 1.5548 | 0.17% |
| 0.990 | 99.05% | 98.05% | 0.9959 | 2.0388 | 0.25% |

The 97.5% construction was frozen before final calibration and fresh v5 because the 95% construction under-covered difficult development groups, while 99% increased mean width from 0.6950 to 0.9959. This is an alpha-level selection on development data, not v5 tuning.

## Mission Tail

Task and return residuals were essentially uncorrelated (Pearson -0.0009, Spearman 0.0057); the long-distance failure was therefore treated as distance-specific Mission calibration rather than a correlated-component theorem. Development performance at the selected 99% construction was overall 99.00%, worst distance group 97.50%, mean width 1.3076.

## Fresh v5 Goal Results

Point MAE: 0.087083. Mean/P95 bound width: 0.613955/1.334575. Overall whole-trajectory coverage: 97.42%. Worst interaction: TASK|2500-4000 at 95.05%.

| Group | n | Covered | Coverage | Exact 95% CI | One-sided p at 0.95 |
|---|---:|---:|---:|---:|---:|
| distance:100-500 | 1155 | 1129 | 97.75% | [96.72%, 98.52%] | 1 |
| distance:1500-2500 | 1154 | 1134 | 98.27% | [97.34%, 98.94%] | 1 |
| distance:2500-4000 | 1152 | 1120 | 97.22% | [96.10%, 98.09%] | 0.9999 |
| distance:500-1500 | 1155 | 1123 | 97.23% | [96.11%, 98.10%] | 0.9999 |
| distance:>4000 | 384 | 365 | 95.05% | [92.38%, 97.00%] | 0.5514 |
| goal_type:CHARGER | 3077 | 3009 | 97.79% | [97.21%, 98.28%] | 1 |
| goal_type:TASK | 1923 | 1862 | 96.83% | [95.94%, 97.57%] | 1 |
| interaction:CHARGER|100-500 | 770 | 748 | 97.14% | [95.71%, 98.20%] | 0.9988 |
| interaction:CHARGER|1500-2500 | 769 | 756 | 98.31% | [97.13%, 99.10%] | 1 |
| interaction:CHARGER|2500-4000 | 768 | 755 | 98.31% | [97.12%, 99.10%] | 1 |
| interaction:CHARGER|500-1500 | 770 | 750 | 97.40% | [96.02%, 98.41%] | 0.9997 |
| interaction:TASK|100-500 | 385 | 381 | 98.96% | [97.36%, 99.72%] | 1 |
| interaction:TASK|1500-2500 | 385 | 378 | 98.18% | [96.29%, 99.27%] | 0.9997 |
| interaction:TASK|2500-4000 | 384 | 365 | 95.05% | [92.38%, 97.00%] | 0.5514 |
| interaction:TASK|500-1500 | 385 | 373 | 96.88% | [94.62%, 98.38%] | 0.9722 |
| interaction:TASK|>4000 | 384 | 365 | 95.05% | [92.38%, 97.00%] | 0.5514 |
| overall | 5000 | 4871 | 97.42% | [96.94%, 97.84%] | 1 |

Goal Holm stress gate: **PASS**.

## Fresh v5 Mission Results

Point MAE: 0.243392. Mean/P95 bound width: 1.436682/1.943879. Overall complete-Mission coverage: 98.87%. Worst distance group: 1500-2500 at 98.17%.

| Group | n | Covered | Coverage | Exact 95% CI | One-sided p at 0.95 |
|---|---:|---:|---:|---:|---:|
| distance:100-500 | 600 | 596 | 99.33% | [98.30%, 99.82%] | 1 |
| distance:1500-2500 | 600 | 589 | 98.17% | [96.74%, 99.08%] | 1 |
| distance:2500-4000 | 600 | 593 | 98.83% | [97.61%, 99.53%] | 1 |
| distance:500-1500 | 600 | 594 | 99.00% | [97.84%, 99.63%] | 1 |
| distance:>4000 | 600 | 594 | 99.00% | [97.84%, 99.63%] | 1 |
| overall | 3000 | 2966 | 98.87% | [98.42%, 99.21%] | 1 |

Mission Holm stress gate: **PASS**.

## Safety-Efficiency and Reserve Attribution

Fresh Goal unnecessary-return proxy: 0.16%. Fresh Mission unnecessary-return proxy: 0.39%.

Historical 20k decisions were reserve dominated: 100.00% of switches were attributed to reserve, and reserve was roughly 31.0x the uncertainty margin. Therefore a narrower uncertainty bound is not expected to improve throughput unless it changes a decision after the fixed reserve is applied.

## Paired 100k Phase2

| Seed | Baseline Tasks | Candidate Tasks | Baseline Exhaustion | Candidate Exhaustion | Baseline Unnecessary | Candidate Unnecessary | Baseline Arrival SOC | Candidate Arrival SOC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 970001 | 195 | 195 | 0 | 0 | 0.000 | 0.083 | 0.151 | 0.152 |
| 970002 | 207 | 207 | 0 | 0 | 0.154 | 0.154 | 0.156 | 0.156 |
| 970003 | 196 | 196 | 0 | 0 | 0.000 | 0.000 | 0.137 | 0.137 |

Mean tasks/1000: baseline 1.9933, candidate 1.9933.
Aggregate tasks: baseline 598, candidate 598; energy exhaustion: 0 vs 0.
Successful returns: 37 vs 37; unnecessary returns: 2/37 vs 3/37.
Arrival SOC (return-weighted): baseline 0.1481, candidate 0.1485.
Reserve-related switches: baseline 35/37, candidate 34/37; pure uncertainty-margin switches: 0 vs 0.
Task-stream pairing audit: 597/598 completed goals matched, with 2/3 seeds exact. Seed 970001 had one sampler-dependent goal mismatch; the other two streams were exact.

Adaptive risk improves delivery efficiency: **NO MEASURABLE IMPROVEMENT**.

## Final Architecture and Scope

Future obstacle/CBF integration should keep the point network on motion state plus goal geometry and expose a compact safe-trajectory context interface to the risk model. Raw LiDAR should not be inserted into the Energy network without evidence. Energy MC data must be collected from the actually executed CBF-filtered trajectory.

The result supports marginal and predefined finite-group conditional statements under exchangeability. It does not support arbitrary conditional coverage, distribution-shift-free guarantees, or physical UAV safety without calibration.

## Recommended Final 500k Configuration

- Frozen SAC checkpoint: `df9a4354420a1a043b60d704292c6fe06ddd2127e937d3d9c72278b287c1ff0a`.
- Frozen 7D point checkpoint: `86b371ca92d6cdd337dc44a67e84f3242decd5661921be0f2fe1d5e4fce8f92b`.
- Goal: K2 suffix risk + compact decision context + direct Mondrian, construction coverage 97.50%.
- Mission: frozen Laplace + direct distance Mondrian, construction coverage 99.00%.
- Safety target: 95%; reserve: 10% of calibrated capacity.
- Formal 500k Phase2 remains **NOT STARTED** pending explicit user approval.
