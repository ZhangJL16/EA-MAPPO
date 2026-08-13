# Commitment-Censored Charger-Return Identification

## Target

Determine whether the post-action charger-return distribution is identifiable from ordinary repeated sorties, where the return label is observed only when the system actually commits to the charger.

## Status

COHERENT AFTER FINAL PROBLEM REFORMULATION.

## Invariant Object

At a task decision opportunity, let

`W=(x,a,d,b,eta)`

and define the potential extended outcome

`Z_tilde^c(W)`

as the defective charger-return energy that would result under the intervention “execute `(a,d)` and commit to the charger continuation from the successor.” Let `M in {0,1}` indicate that this commitment intervention is actually executed and followed until a declared terminal/censoring outcome.

The target CDF is

`F_c(z|W)=P(Z_tilde^c(W)<=z | W)`.

Routine data observe `W,M` and observe the return/failure outcome only when `M=1`. One-step task telemetry when `M=0` does not reveal the counterfactual charger suffix.

## Assumptions for Identification

**I-A1 (consistency).** If `M=1`, the observed extended outcome equals `Z_tilde^c(W)` under the same versioned continuation semantics.

**I-A2 (probe ignorability).** `M` is conditionally independent of `Z_tilde^c(W)` given `W` for the identification population. This is guaranteed by a randomized commitment probe drawn after `W` is fixed, not by a learned threshold that uses hidden information.

**I-A3 (positivity).** The known probe propensity `p(W)=P(M=1|W)` satisfies `p(W)>=p_min>0` on the claimed domain.

**I-A4 (independent analysis units).** Finite-sample concentration treats sortie-level opportunities or independently randomized blocks as units. Correlated transition opportunities are not silently i.i.d.

**I-A5 (probe admissibility boundary).** A probe is attempted only within the declared immediate collision/battery measurement protocol. This does not guarantee successful return; all collision, empty-support, non-arrival, and censoring outcomes remain labels for the defective law.

## I1 — Positivity Failure Implies Non-Identification

**Theorem.** If there is a measurable stratum `H` with positive probability and `P(M=1|W in H)=0`, then `F_c(.|W in H)` is not nonparametrically identifiable from the observed data law.

**Proof.** Construct two environments that agree on the distribution of `W,M` and every outcome observed when `M=1`, but assign different charger-continuation energy/collision laws after `W in H`. Since `M=0` almost surely on `H`, neither law is observed there. The complete observed-data distributions are identical while the target CDFs differ. No estimator can distinguish the two targets from those data.

This directly refutes the claim that arbitrary routine task telemetry supplies all action-conditioned charger-return labels.

## I2 — Selection on the Unobserved Outcome Also Breaks Identification

Even with `P(M=1|W)>0`, if commitment depends on unobserved `Z_tilde^c` after conditioning on `W`, the observed committed-return distribution can differ arbitrarily from the target distribution.

**Counterexample.** Let potential energy be binary, low/high with equal probability. In one model commit only on low outcomes; in another alter the population fraction and selection probabilities so the same committed low/high frequencies and same overall commitment rate are observed. The latent target distributions differ. Therefore positivity without ignorability is insufficient.

## I3 — Identification by Randomized Commitment Probes

Under I-A1--I-A3, for any finite threshold `z`,

`F_c(z|W) = E[M 1{Z_tilde_obs<=z}/p(W) | W]`.

**Proof.** Conditional on `W` and the potential outcome, ignorability gives `E[M|W,Z_tilde^c]=p(W)`. Consistency substitutes the observed outcome on `M=1`. Taking conditional expectation yields the identity.

The same identity estimates finite success mass by taking `z` to the largest finite reporting threshold, while failure/censoring categories are estimated from their observed probe outcomes. A timeout remains right-censoring unless the estimand declares it a failure.

## I4 — Finite-Grid Concentration for Fixed-Propensity Probes

Consider `N` independent opportunities in one declared stratum with constant randomized probe probability `p>=p_min`. For finite thresholds `z_1,...,z_K`, define the Horvitz--Thompson CDF estimate

`F_hat(z_j)=N^{-1} sum_i M_i 1{Z_i<=z_j}/p`.

Then with probability at least `1-delta`, simultaneously for all `j`,

`|F_hat(z_j)-F_c(z_j)|`

`<= sqrt(2 log(2K/delta)/(N p)) + log(2K/delta)/(3 N p)`.

**Proof.** Each summand is unbiased, lies in `[0,1/p]`, and has variance at most `1/p`. Apply Bernstein's inequality to each threshold and union-bound over `K`.

This is a finite-grid identification rate, not a continuous-state deep-learning guarantee. It makes the supervision-throughput cost explicit: small probe propensity degrades sample efficiency approximately as `1/sqrt(Np)`.

## I5 — Throughput/Supervision Tradeoff

With constant probe rate `p`, the expected number of early task terminations caused by probes among `N` opportunities is `Np`. I4 therefore exposes a noncompensable data-collection tradeoff: reducing probe commitments reduces direct task interruption but increases uncertainty in the charger-return CDF. No reward shaping removes this information constraint.

## Algorithmic Consequence

Use **predictable randomized commitment probes** on a declared audit subset:

1. freeze `eta` for the probe block;
2. after observing `W`, draw `M~Bernoulli(p(W))` with logged propensity;
3. if `M=1`, irreversibly commit and retain finite, collision, empty-support, non-arrival, and censoring outcomes;
4. estimate the defective law with propensity-aware methods;
5. keep ordinary learned stopping data separate because it is selectively labeled;
6. calibrate only on an independent probe/evaluation stream or use a theorem justified for the adaptive stream.

This is not a special recovery controller. It changes only when task termination is sampled during data collection; charger motion still uses the shared goal-conditioned policy.

## Boundaries and Non-Claims

- I1--I3 are standard missing-data/potential-outcome identification logic specialized to charger commitment.
- I4 is a standard inverse-propensity concentration result on a finite grid.
- Safe real-world probing is not guaranteed; probes are limited to the declared empirical admission protocol.
- Continuous high-dimensional generalization, adaptive propensity optimization, and conformal coverage require additional results.
- If closest work already covers the same sequential probe/return-identification problem, this reformulation is not novel.

## Open Risk

The mathematical results may be too standard for ICLR novelty. Their value is to expose a previously hidden data-identification flaw and to define a correct experiment. If hostile review finds no stronger distinction, the research direction is blocked rather than promoted.
