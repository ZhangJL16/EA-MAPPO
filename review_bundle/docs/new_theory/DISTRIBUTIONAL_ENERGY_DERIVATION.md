# Distributional Energy Derivation

## Target

Learn the finite-energy part and missing mass of the budget-conditioned charger-return law, then construct a statistically scoped upper prediction bound.

## Status

COHERENT CONDITIONALLY. The distributional Bellman law is exact; neural approximation and calibration remain assumption-dependent.

## T3 — Coupled Distributional Identity

For `w=(x,a,d,b)` and residual `b'=b-d`,

`Z_tilde_E^eta(w) =_D +infinity`

on immediate collision or empty filtered support, and otherwise

`c(x,a,S') + Z_tilde_E^eta(S',A',D',b')`,

with `(A',D')~kappa_{C,eta}(.|S',b')`. At charger terminal it equals zero. The identity is undiscounted.

## Why a Finite Quantile Head Is Not the Whole Object

A neural head cannot emit mathematical `+infinity` robustly. The primary model therefore estimates:

- `p_S^eta(w)`, the finite-return success mass;
- ordered quantiles `q_tau^eta(w|S)` of finite energy conditional on successful collision-free charger arrival.

The reconstructed sub-CDF is `p_S F_fin`. Reporting conditional quantiles without `p_S` hides collision, empty-support, and non-arrival failures.

## Quantile TD for the Finite Component

For transitions known to remain on a finite-return path, quantile targets sample the **joint successor mixture**

`Y=c+(1-terminal) Z_fin^eta(S',A',D',b-d)`.

Pairwise quantile Huber regression projects this mixture. Adding corresponding quantiles directly is not exact for dependent/random sums. Monotone heads prevent crossing. The result approximates `F_fin`; it does not estimate `p_S`.

## Calibration Population

Freeze `eta` and the selector. Let a deployment-selected tuple be `W=(X,A,D,B,eta)` and let `S` denote successful collision-free charger arrival in the energy-unconstrained reference rollout. Suppose independent calibration sorties and the future selected tuple are exchangeable **conditional on `S=1` and declared context**. For base quantile `q_hat`, define scores

`R_i=E_i-q_hat(W_i)`

on successful finite returns and use the finite-sample split-conformal order statistic. Then

`P(E_new<=U_E^eta(W_new) | S_new=1) >= 1-alpha_E`.

This is marginal over the matched successful selected-action population. It is not statewise coverage, does not estimate `P(S=1)`, and is not anytime-valid.

## T4 — Calibrated Energy Implication

Assume the conditional coverage statement above, `e^- >= U_E^eta(W)+m`, and `P(e<e^-)<=beta_e`. Then

`P(S=1 and E+m>e) <= alpha_E P(S=1) + beta_E + beta_e`

and hence also `<=alpha_E+beta_E+beta_e`.

**Proof.** On the valid battery event, feasibility implies `E+m>e` only if `E>U_E`. Multiply the conditional exceedance bound by `P(S=1)` and union-bound calibration- and measurement-validity failures.

This theorem deliberately excludes `S=0`. Collision/failure mass is bounded separately by the risk ledger and success-mass model. A paper may alternatively calibrate the full extended law, but a finite conformal bound then exists only when the calibration quantile does not fall in its infinite mass.

## Censoring and Rare Tails

- Charger completion provides a finite label.
- Collision or empty support provides failure-mass supervision, not a truncated energy label.
- Timeout is right-censoring unless the estimand declares timeout as failure.
- Rare catastrophic energy tails absent from calibration cannot receive distribution-free subgroup coverage. The allowed claim is only for the exchangeable population and nominal finite-sample level.

## Shift and Adaptive Selection

Policy/filter version changes, candidate-set optimization, payload, age, wind, and sensor changes can invalidate coverage. Accepted options are selector-output calibration, simultaneous candidate-set methods, weighted conformal under a stated density-ratio assumption, or sequential methods with their actual theorem. A heuristic rolling window is not automatically valid.

## Non-Claims

- Quantile TD does not itself supply an upper bound.
- `mean+k sigma` is not used as a theorem-level bound.
- No unconditional coverage under arbitrary drift.
- No inference from E1 to collision-coupled return laws.
- No identification of unobserved task-action charger returns from ordinary logs without probe positivity/ignorability or a justified off-policy model.
