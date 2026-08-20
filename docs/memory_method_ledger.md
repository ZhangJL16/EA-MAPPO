# Memory Method Ledger

All prediction/tube values use the held-out 1,000-trajectory test split of the
5,000 matched-trajectory benchmark. Tube entries are marginal q95 empirical
orthotopes at the 1 s horizon and are not deterministic certificates. Closed-
loop values use only
`artifacts/memory_closed_loop_controlled_final_audited_20260820/`.

| ID | Method | Explicit memory | Uncertainty | Proof status | Joint MAE | q95 coverage / width | Closed-loop success / fallback / under-bound | Success energy | P99 total | Closest prior | Novelty / 30 | Decision |
|---|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---|
| A | Current only | position | empirical base box | no deterministic learned-box proof | 1.2044 | .959 / 25.2910 | .583 / 1.000 / .370 | 2.9988 | 25.4735 ms | robust sensing | 2 | baseline |
| B | Raw L16 MLP | none | conformal only | none | .8985 | .949 / 13.0867 | not selected for closed loop | n/a | .0036 ms estimator | sequence regression | 1 | reject |
| C | Ego L16 MLP | motion vectors | conformal only | radial identifiability only | **.5303** | .961 / **8.3024** | .667 / 1.000 / .343 | 2.6198 | 24.6742 ms | motion compensation | 2 | engineering baseline |
| D | Vanilla RNN | latent | conformal only | none | .5964 | .964 / 9.6522 | not selected | n/a | .0026 ms estimator | recurrent regression | 1 | reject |
| E | GRU | latent | development max-residual box in loop | none | .5982 | .949 / 9.3619 | .667 / 1.000 / **.518** | 2.4632 | 21.9489 ms | recurrent regression | 1 | reject |
| F | LSTM | latent | conformal only | none | .5895 | .965 / 10.2958 | not selected | n/a | .0025 ms estimator | recurrent regression | 1 | reject |
| G | TCN | finite latent history | conformal only | none | .6165 | .963 / 10.7977 | not selected | n/a | .0089 ms estimator | temporal convolution | 1 | reject |
| H | SSM | latent dynamics | conformal only | hidden stability only | .9195 | .962 / 25.5142 | not selected | n/a | .0194 ms estimator | stable sequence models | 1 | reject |
| I | Physics-GRU | p/v/a nominal | development max-residual box in loop | nominal only | .6683 | .960 / 9.9987 | .583 / 1.000 / .042 | 2.5827 | 19.3628 ms | KalmanNet | 3 | reject |
| J | Physics-GRU + interval | p/v/a | analytic orthotope | conditional on an independent residual bound | same as I | radius cannot shrink without a smaller valid bound | no additional certified feasible set | no guarantee | same order as I | KalmanNet + interval observer | 4 | reject |
| K | J + reset | J + epoch/consistency | analytic/base reset | conditional fresh-base proof | n/a | broad base set after reset | fail-closed rule, not contribution | no guarantee | analytic | switched robust observer | 4 | reject novelty |
| L | Explicit analytic memory | p/v/a centers/radii, innovation, mode, misses, track epoch | deterministic orthotope | conditional observer/tube/HOCBF proof | analytic | bounded-slice containment 1.0; width too broad for control | **.000 / 1.000 / .000** | n/a | 24.2806 ms | observer/set-valued CBF | 5 | reject: unusable |
| M | Contractive physical memory | L + contractive residual hidden state | development box plus analytic fallback | hidden contraction only | .8362 | .960 / 10.2082 | .583 / 1.000 / .054 | 2.6607 | 27.4091 ms | contracting RNN | 3 | reject |

Classical same-data references: CV-KF joint MAE `.7828`; CA-KF/EKF `.8390`;
IMM `.7772` with q95 `.967/9.6142`; MHE `.8945`. CA-KF closed-loop
success/fallback/under-bound is `.583/1.000/.016`; IMM is
`.583/1.000/.024`. Every learned/Kalman/IMM closed-loop box is explicitly
uncertified.

## Size and deployment ledger

Physics-GRU hidden sizes 16/32/64/128 yield joint MAE
`.8269/.6734/.6059/.5892`; contractive memory yields
`.8905/.8393/.7997/.7181`. Both remain worse than the ego L16 MLP. All
single-estimator P99 inference values in the size ablation are below 1 ms, but
the contractive model's full-loop P99 memory-update path reaches `23.7764 ms`.

## Search closure

All pre-registered families A–M are either baselines or rejected. The final
route does not satisfy the novelty gate or the equal-certification empirical
consequence gate.
