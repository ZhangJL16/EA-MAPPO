# Navigation qualification

Status: **REVIEW_REQUIRED**; baseline runs = 0; neural updates = 0.

Success: 988/1000 (98.800%); Wilson 95% CI: [0.9791427315245017, 0.9931223521935965].

Quantiles below condition on navigation success; failed routes remain in the evidence.

| Metric | q25 | q50 | q75 | q90 |
|---|---:|---:|---:|---:|
| time | 69.587500 | 109.050000 | 147.512500 | 181.120000 |
| energy | 14.683832 | 22.926218 | 30.841672 | 38.262026 |

Failure reasons: {'navigation_timeout': 12}.
Pilot time consumed by failed jobs: 7.990%; this is not an additive scheduling loss estimate.

| Diagnostic | Range | n | Failures | Success rate |
|---|---|---:|---:|---:|
| horizontal_distance | [0, 1000) | 186 | 4 | 97.85% |
| horizontal_distance | [1000, 2000) | 331 | 4 | 98.79% |
| horizontal_distance | [2000, 3000) | 329 | 2 | 99.39% |
| horizontal_distance | [3000, inf) | 154 | 2 | 98.70% |
| vertical_distance | [0, 60) | 321 | 4 | 98.75% |
| vertical_distance | [60, 180) | 443 | 5 | 98.87% |
| vertical_distance | [180, inf) | 236 | 3 | 98.73% |
| privileged_chord_exposure | [0, 1) | 611 | 6 | 99.02% |
| privileged_chord_exposure | [1, 2) | 288 | 6 | 97.92% |
| privileged_chord_exposure | [2, inf) | 101 | 0 | 100.00% |

Stopping signals: none; user review remains required

Obstacle exposure is a privileged straight-chord geometry diagnostic, not actor input or realized flight exposure.
Calibration cannot prove navigation failure does not dominate future policy rankings. No baseline was run.
