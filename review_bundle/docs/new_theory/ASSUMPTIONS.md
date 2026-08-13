# Assumptions

## State, Terminal, and Costs

**A1 (state sufficiency).** The augmented input `x_t` is Markov for motion, collision, and energy laws under the deployment regime. If raw local observations alias different charger routes, `x_t` must be a history or belief. A memoryless observation is not assumed sufficient by notation.

**A2 (terminals).** Charger set `G_c` and collision set `G_B` are measurable. Charger entry terminates return accounting with zero future cost. Collision, empty filtered support, and non-arrival are represented by `+infinity` in the extended return.

**A3 (energy cost).** `c_t>=0` almost surely. Finiteness results use bounded stage cost `c_t<=c_max<infinity` or the explicitly weaker integrability condition stated in the theorem.

**A4 (augmented properness domain).** On a declared domain `D_eta` of `(x,b)`, the fixed risk-budgeted charger kernel `kappa_{C,eta}` has nonempty support until absorption and reaches either charger or collision in finite expected time. Any stronger claim of almost-sure charger arrival is stated separately. This is an SSP domain assumption, not recoverability.

**A5 (stationary version).** During a theorem/evaluation window, transition law and version tuple `eta` are fixed. Changing the base policy, collision estimator, calibration set, candidate generator, or allocation rule changes the target law and requires a new version, recollection/reweighting justification, and recalibration.

## Sequential Collision Accounting

**A6 (atomic hazard).** The recursively spent event is the one-transition swept-body collision event `B_{t+1}`. A predictor with overlapping `H_C>1` windows may be used for screening but cannot be subtracted step-by-step from `b_t` as if windows were disjoint. Exact block recursion is allowed only for non-overlapping macro-decisions.

**A7 (near collision).** Near-collision is an auxiliary prediction/evaluation label unless a separate event and risk budget are declared. It is not silently merged with physical collision in theorem statements.

**A8 (selection-valid upper hazard).** On a validity event `G_C`, the executed selected action satisfies

`P(B_{t+1}=1 | F_t,a_t) <= U_C^1(x_t,a_t)`

for every pre-collision decision. This requires pointwise, simultaneous candidate-set, or policy-coupled selected-action validity; ECE and marginal classifier accuracy are insufficient.

**A9 (predictable spending).** `d_t` is `F_t`-measurable, `U_C^1(x_t,a_t)<=d_t<=b_t`, and `b_{t+1}=b_t-d_t` after a noncollision transition. Along every realized sortie, `sum_{t<T} d_t<=Delta_C` for the relevant stopping time `T`.

## Coupled Charger Kernel

**A10 (same base policy).** `kappa_{C,eta}` is constructed by restricting or projecting `pi_theta(.|x,g_c)` using learned collision admissibility and budget allocation. It is not a separate recovery policy. The exact selector and normalization are part of `eta`.

**A11 (support semantics).** If no `(a,d)` satisfies the filter at `(x,b)`, execution records `NO_ADMISSIBLE_CHARGER_ACTION`. Theory assigns this branch infinite extended return; it does not choose a least-violating action and call it safe.

## Energy Distribution and Calibration

**A12 (target match).** Training and calibration targets correspond to the same `eta`, budget input, first action/spend, and charger continuation as deployment. Completed returns, collisions, empty-support episodes, and timeouts remain distinguishable; failed trajectories are not discarded from success-mass estimation.

**A13 (finite-return calibration mode).** A finite `U_E^eta` receives only the precise coverage mode justified by data: exchangeable marginal selected-action coverage, simultaneous candidate-set coverage, or an explicitly conditional/PAC result. Ordinary split conformal does not become pointwise or anytime-valid after adaptive search.

**A14 (shift scope).** Exchangeability, weighted-covariate-shift, or sequential-validity assumptions are stated for the actual deployment stream. Policy drift, payload, wind, battery aging, and sensor changes either enter `x,eta`, trigger recalibration, or suspend the coverage claim. Arbitrary shift has no unconditional result.

**A15 (battery lower bound).** On event `G_e`, `e_t^-<=e_t`, with failure probability at most `beta_e`. Reserve `m` covers only explicitly listed consumption. Optimistic telemetry bias invalidates the energy implication.

## Approximation

**A16 (no nonlinear convergence claim).** Deep scalar/quantile TD estimates a population target. Convergence is claimed only for declared finite tabular/proper or realizable special cases.

**A17 (calibration dependence).** Correlated TD replay items are not treated as exchangeable calibration examples. Calibration splits by independent sortie, episode block, or another unit justified for the dependence structure.

## Filter-Version Transport

**A18 (transport isolation).** C4--C6 isolate a change in the induced charger action kernel. Transition, energy, sensor, payload, wind, and battery laws remain fixed; otherwise their kernel drift must be included rather than attributed to the collision filter.

**A19 (kernel-level audit).** Reuse decisions are based on a valid upper bound for induced-kernel/path drift or admission-boundary visitation. Neural parameter distance and training loss are not transport metrics.

**A20 (coverage event frozen).** C6 transports the probability of a fixed old-version event. If the prediction rule is retuned after seeing new-version outcomes, a new statistical argument is required.

## Commitment-Censored Identification

**A21 (potential-return consistency).** A committed observed outcome equals the potential charger-return outcome under the logged `W,eta`.

**A22 (randomized probe ignorability).** On the identification stream, the commitment probe is randomized after `W` is fixed and is conditionally independent of the potential return given `W`.

**A23 (probe positivity).** Logged propensity satisfies `p(W)>=p_min>0` on every domain where nonparametric identification is claimed.

**A24 (selective routine logs).** Learned stopping decisions are not assumed ignorable. They are excluded from simple identification/calibration unless a valid selection/off-policy correction is supplied.
