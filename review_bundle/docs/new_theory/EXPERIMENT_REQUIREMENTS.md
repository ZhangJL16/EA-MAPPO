# Experiment Requirements for the Coupled Theory

No experiment in this file is a result. The running E1 remains a collision-free special-case estimator study and must not be interrupted or reinterpreted as evidence for the coupled operator.

## Required Coupling Tests

1. **Budget conditioning:** compare `Z_E(x,a)` against `Z_E(x,a,b)` with identical data/model capacity.
2. **Version conditioning:** freeze versus update `eta`; measure target drift near admission boundaries.
3. **Defective law:** compare completed-return-only fitting against explicit success-mass plus finite-return modeling.
4. **Failure retention:** collisions, empty supports, timeouts, and interrupted trajectories remain separate raw strata.
5. **Selector calibration:** logged-action split conformal versus selector-output or simultaneous candidate-set calibration at matched candidate counts.
6. **Matched general baselines:** multi-cost CMDP, chance-constrained/CVaR policy, distributional cost RL, geometric RTH threshold, and independent collision/energy critics.
7. **Representation counterexample:** instantiate C1 and verify that a budget-agnostic critic incurs the predicted irreducible mismatch.
8. **Margin stability:** stratify return-target drift by empirical collision-admission margin and test C2/C3 qualitatively.
9. **Transport gate:** compare freeze/reuse, full recollection, generic importance-weighted/off-policy reuse, and C4--C6 transport-or-recalibrate decisions.
10. **Boundary-mass calibration:** evaluate whether held-out upper bounds on `rho` actually dominate observed return-law and coverage drift.
11. **Commitment probes:** compare routine learned-stopping logs, randomized probe logs, and oracle fully labeled simulation data.
12. **Positivity sweep:** vary probe rates and verify CDF error versus effective probe count `Np` and task-throughput loss.
13. **Selection bias:** construct outcome-dependent stopping to demonstrate I2, then test propensity-aware correction only where assumptions hold.

## Falsification Criteria

The coupled claim fails if a budget-agnostic independent critic, matched multi-cost CMDP/CVaR method, or ordinary collision filter plus RTH threshold matches failure mass, energy-tail coverage, empty-set behavior, and throughput under equal policy/data/candidate budgets.

It also fails if conditioning on `b,eta` only increases capacity without improving the exact budget-stratified estimands, or if survivor-bias correction has no measurable effect because failure mass is negligible in the target regime.

## Data and Evaluation Units

- train/calibration/validation/final evaluation split by sortie or independent environment stream;
- immutable `eta`, policy hash, candidate generator, allocator, and calibration hash per sample;
- explicit right-censoring indicators;
- fixed held-out budgets and shift strata;
- 5 seeds for primary comparisons where feasible, never fewer than 3;
- equal environment transitions, candidate evaluations, and inference budget.
