# Filter-Induced Target Transport

## Target

Quantify when charger-return data/calibration from collision-filter version `eta` can be reused after updating the learned filter to `eta'`.

## Status

COHERENT AFTER PROBLEM REFORMULATION / EXTRA ASSUMPTIONS.

## Invariant Object

The invariant is the path law of the extended charger return under the induced augmented-state kernels, not neural parameter distance:

`P_eta^{y}` and `P_eta'^{y}`, `y=(x,b)`.

The relevant drift is the total-variation distance between path/return laws caused by changed action support.

## Assumptions

1. Environment transition and energy laws are unchanged between versions; only the charger action/spend kernel changes. Environment shift is a separate term.
2. Both versions use the same measurable augmented state and absorbing outcomes.
3. `kappa_eta(.|y)` and `kappa_eta'(.|y)` are probability kernels including an explicit failure action for empty support.
4. A coupled trajectory uses maximal action coupling while paths agree and common transition/cost randomness after matched actions.
5. Claims using a finite horizon state it; unbounded claims require finite expected absorption and integrable kernel discrepancy.

## Notation

Define local kernel drift

`epsilon(y)=TV(kappa_eta(.|y),kappa_eta'(.|y))`.

Let `T` be absorption under the reference version while the coupled paths agree. Define

`rho_eta->eta'(y0)=min(1, E_eta^{y0}[sum_{t<T} epsilon(Y_t)])`.

For a collision-bound perturbation size `gamma`, define the admission-boundary set

`B_gamma={y: some proposed (a,d) satisfies |U_C,eta^1(x,a)-d|<=gamma}`.

## Derivation Map

1. Maximal coupling mismatches actions at state `y` with probability `epsilon(y)`.
2. Until the first mismatch, both trajectories and accumulated energy are identical.
3. Union/tower accounting bounds mismatch-before-absorption by expected cumulative local kernel drift.
4. If filter support can change only inside `B_gamma`, path-law drift is bounded by boundary visitation.
5. Total-variation control transfers any fixed coverage event with additive degradation.

## C4 — Path/Return-Law Transport Bound

**Proposition.** Under the assumptions above,

`TV(P_eta^{y0},P_eta'^{y0}) <= rho_eta->eta'(y0)`.

The same bound holds for the pushforward defective return laws:

`TV(Law_eta(Z_tilde_E),Law_eta'(Z_tilde_E)) <= rho_eta->eta'(y0)`.

**Proof.** Couple the two kernels maximally whenever the histories agree. At common state `Y_t`, conditional mismatch probability is `epsilon(Y_t)`. If no mismatch occurs before absorption, common environment randomness gives identical states, costs, terminal outcome, and extended return. The coupling inequality bounds total variation by the probability of any mismatch. Conditional union bound and tower expectation give

`P(any mismatch before T) <= E_eta[sum_{t<T} epsilon(Y_t)]`.

Cap at one. Total variation cannot increase under the measurable map from paths to extended returns.

This result is directional because the expectation uses the reference path law. A symmetric report uses the minimum of valid directional bounds only when both have been estimated, or the maximum for a conservative bidirectional certificate.

## C5 — Boundary-Mass Corollary

Suppose the base policy, candidate proposal, allocations, tie-breaking, and environment remain fixed, and

`sup_{x,a}|U_C,eta'^1(x,a)-U_C,eta^1(x,a)| <= gamma`.

Then admissibility cannot change outside `B_gamma`, so `epsilon(y)=0` there. Consequently,

`TV(Law_eta(Z_tilde_E),Law_eta'(Z_tilde_E))`

`<= P_eta^{y0}(exists t<T: Y_t in B_gamma)`

for deterministic support filters coupled identically outside the boundary set. More generally C4 applies with the measured `epsilon(y)` for stochastic/continuous selectors.

**Proof.** Outside `B_gamma`, every old admission inequality has margin greater than the maximal perturbation, so its sign is unchanged. Thus the induced kernels agree. Before the first boundary visit, coupled paths cannot diverge. The coupling inequality gives the result.

C2 shows this dependence is necessary: without controlling boundary visitation, arbitrarily small `gamma` can produce order-one return-law drift.

## C6 — Coverage Transport

Let `A_U` be any fixed measurable success event derived from the old prediction rule, for example

`A_U={Z_tilde_E <= U_eta(y0)}`.

If `P_eta(A_U)>=1-alpha` and `TV(P_eta,P_eta')<=rho`, then

`P_eta'(A_U)>=1-alpha-rho`.

**Proof.** Total variation bounds the absolute probability difference of every measurable event by `rho`.

Thus old selected-action coverage can be transferred only with an explicit drift debit. If `alpha+rho>=1`, the transferred statement is vacuous. This does not update the numeric bound `U_eta`; it only states its degraded coverage under the new path law.

## Algorithmic Consequence — Transport-or-Recalibrate Gate

For each filter update:

1. freeze old and candidate new versions;
2. estimate or upper-bound local kernel drift and boundary-hit probability on an independent audit stream;
3. if a high-confidence upper bound `rho_bar` is below a declared transport tolerance, retain the old bound only at degraded level `alpha+rho_bar`;
4. otherwise invalidate old energy calibration and recollect/recalibrate under `eta'`;
5. never infer `rho_bar` from training loss or parameter distance.

This is a falsifiable data-reuse rule. Its benefit is zero when updates frequently alter support along charger paths.

## Boundaries and Non-Claims

- C4 is a policy-kernel perturbation/coupling result; generic perturbation theory may subsume it.
- C5 is specific to learned hard admission boundaries but assumes unchanged proposal/allocation machinery.
- Estimating a valid `rho_bar` is a separate statistical problem.
- No arbitrary-shift, pointwise conformal, or deep-learning convergence guarantee follows.
- If closest work already gives this boundary-mass-to-return-coverage transport result, it is not novel.

## Open Risks

- Continuous action proposals can have diffuse support changes not captured by a finite candidate margin.
- The directional reference occupancy may miss regions reached only under `eta'`; bidirectional audit is safer.
- Long expected hitting times make cumulative kernel-drift bounds vacuous.
