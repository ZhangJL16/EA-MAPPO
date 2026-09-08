# Pending Results and Gate Map

| Slot | Upstream gate | Required artifact | Claim unlocked only after PASS |
| --- | --- | --- | --- |
| NAVIGATION GATE | R1--R4 fixed protocol, 3 seeds | checkpoint evaluations and gate JSON | navigation is capable enough for downstream study |
| BATTERY CALIBRATION | navigation PASS | held-out calibration curves | simulator resource accounting is fit for study |
| ORACLE HEADROOM | battery PASS | paired Oracle vs SOC/distance cycles | exact Resource-to-Go has decision utility |
| ETG PILOT | Oracle PASS | direct/TD/WM prediction artifacts | learned prediction is feasible |
| PAIR VS INTERFACE | pilot PASS | factorial leave-one-pair-out report | interface risk, not pair identity, governs failure |
| ENSEMBLE VS RELIABILITY | mechanism PASS | risk--coverage/AURC report | reliability detects severe underestimation |
| RETURN PARETO | reliability PASS | paired 100-cycle/point frontier | method improves stranding--throughput tradeoff |
| SECOND DOMAIN/FILTER | primary PASS | replicated factorial and frontier | generality beyond the primary setting |
| CMDP TRACK | protocol-matched budget | CPO/CVPO/SDAC or selected track | comparison to direct constrained control |

## Current evidence that cannot fill result slots

- The historical JSEB 500k R0 navigation evaluation failed the fixed navigation Gate despite passing collision/boundary predicates.
- Short smoke runs validate mechanics only.
- The current R1 process is ongoing and has no audited final evaluation.

## Reviewer-driven additions

- Reviewer A: direct switcher and matched-budget constrained/distributional RL are mandatory.
- Reviewer B: executed-WM + Deep Ensemble must be beaten before SIRP is retained.
- Reviewer C: a second domain or materially different safety-operator family is mandatory for broad autonomy claims.
- Closest-work repair: add a Predictive-Safety-Network-style resource-safety head, DCRL when constraint semantics are comparable, and a receding-horizon planner with the same declared primitive; document incompatibilities instead of silently omitting a baseline.
