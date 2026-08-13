# TD Derivation

## Target and Status

Derive learnable targets for the budget-conditioned defective charger-return law while separating population semantics, tabular theory, nonlinear approximation, and calibration. Status: COHERENT AFTER REFORMULATION.

## E1 Special-Case Baselines

When collision is absent and the charger policy is fixed, compare distance times empirical Wh/m, Monte-Carlo return regression, scalar TD with `gamma=1`, and distributional TD. These estimate the finite special-case law in `ENERGY_SSP_DERIVATION.md`; E1 cannot validate the coupled operator.

## Coupled Replay Tuple

Each coupled transition must record

`(x_t,b_t,a_t,d_t,c_t,B_{t+1},x_{t+1},b_{t+1},terminal,eta)`.

Collision, empty support, charger hit, timeout, and logging interruption are distinct outcomes. `eta` hashes the base policy, collision estimator/calibrator, candidate selector, and allocator. Mixing versions without a justified off-policy correction changes the target.

## Scalar Mean Target on Finite Returns

For finite collision-free returns under fixed `eta`, the ordinary target is

`y_t=c_t+(1-terminal) E_{(a',d')~kappa_eta}[Q_bar(x',a',d',b-d)]`.

This has `gamma=1`. It estimates a conditional finite-return mean only when failure/censoring semantics are modeled separately. Discarding failed trajectories and calling this an unconditional return value is survivor bias.

## Defective-Law Factorization

Represent the finite sub-CDF as

`F_E^eta(z|w)=p_S^eta(w) F_fin^eta(z|w,S)`,

where `w=(x,a,d,b)`, `p_S=P(T_c<min(T_B,T_empty)|w)`, and `F_fin` is the finite-energy CDF conditional on success. The missing mass is `1-p_S`.

This factorization is an identity. It motivates two learned heads:

1. a success/failure-mass head trained with all completed, collision, empty-support, and right-censored trajectories using a declared survival/censoring method;
2. a conditional finite-return distribution head trained on valid finite-return supervision, with censoring-aware objectives when completion is not observed.

The two outputs reconstruct one defective law; they are not two compensating safety rewards.

## Population Recursion for Failure Mass

For fixed `kappa_eta`, let `r_eta(w)=P(Z_tilde_E=+infinity|w)`. Formally,

`r_eta(x,a,d,b)`

`= P(B'=1 or empty at S' | x,a)`

`+ E[1{B'=0, support nonempty, S' not terminal} r_eta(S',A',D',b-d)]`.

This is a population Bellman identity. A finite timeout is not automatically the event `Z_tilde_E=+infinity`; it is right-censoring unless the horizon is part of the estimand.

## Approximation Levels

1. **Population semantics:** the mean, failure-mass, and distributional first-step identities are exact under A1--A12.
2. **Tabular/proper special case:** standard absorbing-SSP stochastic approximation may converge with sufficient visitation and suitable stepsizes.
3. **Realizable supervised special case:** completed i.i.d. return labels support standard regression analysis for the chosen function class.
4. **Nonlinear TD:** target networks/replay are optimization heuristics; no general convergence theorem is claimed.
5. **Calibration:** a separate statistical layer; low TD loss does not imply coverage.

## Policy/Filter Drift

Proposition C2 shows that arbitrarily small changes in `U_C` can cause an order-one target change at an admission boundary. Therefore online updates require one of:

- freeze `eta` during collection/training/calibration windows;
- condition the critic on version/context and retain support overlap;
- recollect after support changes;
- use an explicitly justified off-policy estimator and recalibrate.

Proposition C3 permits reuse only on a region with a verified admission margin and unchanged induced kernel.

## Non-Claims

- No global convergence for deep scalar, survival, or quantile TD.
- No exchangeability claim for correlated replay transitions.
- No claim that completed-return-only training estimates missing mass.

## Identification Before TD

TD bootstrapping does not repair absent counterfactual support. If task opportunities in a region are never followed by charger commitment, I1 shows their charger-return law is unidentified regardless of TD architecture. Version-matched randomized commitment probes or another justified off-policy design must supply support before TD generalization is evaluated.
