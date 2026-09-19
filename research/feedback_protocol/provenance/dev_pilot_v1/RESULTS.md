# Frozen DEV quality–compute pilot

1920 scheduled records; statuses: `{'completed': 1776, 'reference_unresolved': 144}`.
Four root groups, 12 repeated capacity/horizon conditions; no final test or training.

## Per-condition comparison

Risk below is Monte Carlo Bayes risk (not exact expected risk). Exact reference risk
comes from rational Bayes value. Approximate methods use the LARGE shared budget.

| Root | B | T | Exact Bayes risk | Beam MC | VOI MC | PS MC | Cover MC |
|---|---:|---:|---:|---:|---:|---:|---:|
| iid-dev-0 | 3 | 4 | 2.0000 | 2.0000 | 2.0000 | 2.0000 | 3.5000 |
| iid-dev-0 | 3 | 8 | 3.5000 | 3.6250 | 3.6250 | 4.0000 | 6.2500 |
| iid-dev-0 | 5 | 4 | 2.0000 | 2.0000 | 2.0000 | 2.0000 | 4.0000 |
| iid-dev-0 | 5 | 8 | 3.5000 | 3.6250 | 3.6250 | 4.0000 | 5.3750 |
| iid-dev-1 | 3 | 4 | 2.0000 | 2.0000 | 2.0000 | 2.0000 | 3.7500 |
| iid-dev-1 | 3 | 8 | 4.0000 | 4.0000 | 4.0000 | 4.0000 | 6.2500 |
| iid-dev-1 | 5 | 4 | 2.0000 | 2.0000 | 2.0000 | 2.0000 | 4.0000 |
| iid-dev-1 | 5 | 8 | 4.0000 | 4.0000 | 4.0000 | 4.0000 | 4.5000 |
| scale-dense-dev-0 | 7 | 8 | 4.0000 | 4.0000 | 4.0000 | 4.0000 | 7.1250 |
| scale-dense-dev-0 | 7 | 12 | unresolved | 6.0000 | 5.0000 | 6.0000 | 7.6250 |
| scale-sparse-dev-0 | 7 | 8 | 4.0000 | 4.0000 | 4.0000 | 4.0000 | 7.1250 |
| scale-sparse-dev-0 | 7 | 12 | 5.2500 | 3.0000 | 3.0000 | 6.0000 | 10.2500 |

## Reference solve costs

| Root / B / T | Bayes status | States | Offline seconds | Minimax status |
|---|---|---:|---:|---|
| iid-dev-0 / 3 / 4 | exact | 17 | 0.000738 | exact |
| iid-dev-0 / 3 / 8 | exact | 105 | 0.004500 | unresolved |
| iid-dev-0 / 5 / 4 | exact | 29 | 0.000740 | exact |
| iid-dev-0 / 5 / 8 | exact | 266 | 0.017780 | unresolved |
| iid-dev-1 / 3 / 4 | exact | 13 | 0.000470 | exact |
| iid-dev-1 / 3 / 8 | exact | 57 | 0.002836 | unresolved |
| iid-dev-1 / 5 / 4 | exact | 17 | 0.000684 | exact |
| iid-dev-1 / 5 / 8 | exact | 86 | 0.008387 | unresolved |
| scale-dense-dev-0 / 7 / 8 | exact | 1008 | 0.163523 | unresolved |
| scale-dense-dev-0 / 7 / 12 | unresolved | 2000 | 0.379631 | unresolved |
| scale-sparse-dev-0 / 7 / 8 | exact | 194 | 0.010059 | unresolved |
| scale-sparse-dev-0 / 7 / 12 | exact | 847 | 0.067158 | unresolved |

## Interpretation limits

- DEV only; no neural training; no final-test access
- Root graph is the sampling unit. Capacity/horizon variants are repeated conditions.
- Four noise replicates and two fixed policy seeds: descriptive pilot, not population significance.
- Worst-hypothesis risk is max of estimated means; finite-replicate upward selection bias remains.
- MC SE is conditional on fixed policy seeds and task, not an instance-generalization interval.
- Reference-unresolved cells remain visible; compare Bayes gaps only on explicitly solved subsets.
- Empirical Pareto status is noise/timing dependent; wall limits are cooperative.
- A zero empirical MC SE from four draws is not zero true uncertainty.
- Root aggregates with unresolved references cover different numbers of conditions; do not compare mismatched averages.
- No parameter/seed/horizon/policy changes were made after outcome inspection.
- This is internal rational replay, not external independent expert validation.

Raw evidence archive SHA-256: `b634bf71a2e05125c42a7e525ea220881be075f61e6f39b89d1380fd02614f7a`.
