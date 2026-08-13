# Dual-Budget Bellman Semantics

## Target

Derive the recursive law and feasibility implication for physical energy and a noncompensable collision-risk ledger.

## Status

COHERENT AFTER PROBLEM REFORMULATION / EXTRA ASSUMPTIONS.

## Derivation Strategy

1. augment the charger SSP state by residual collision budget;
2. use a one-transition swept collision event so risk spends are sequentially meaningful;
3. propagate an extended energy law whose infinite mass records collision, empty support, or non-arrival;
4. calibrate energy only for the exact selected-action/versioned population;
5. derive joint failure accounting without weighted scalarization.

## Distributional Bellman Operator

Let `S'~P(.|x,a)`, let `B'` be the swept-transition collision indicator, and set `b'=b-d`. For charger terminal `x in G_c`, `Z_tilde_E=0`. Otherwise,

`Z_tilde_E^eta(x,a,d,b) =_D`

`+infinity`, if `B'=1` or `D_eta(S',b')` is empty;

`c(x,a,S') + Z_tilde_E^eta(S',A',D',b')`, otherwise,

where `(A',D')~kappa_{C,eta}(.|S',b')`.

This is an equality in distribution under one common path law. Quantiles are not added independently; quantile TD is a projection of this distributional target.

For a probability measure `mu` on `[0,+infinity]`, the corresponding operator pushes safe successor mass through the energy shift and sends collision/empty-support mass to `delta_{+infinity}`. No discount is introduced.

## T3 — Distributional Fixed-Point Semantics

**Conditional-proved.** Under A1--A5 and A10--A11, the pathwise extended return constructed above is a fixed point of the distributional operator. If the finite augmented SSP is absorbing under the fixed kernel, finite-horizon truncated returns converge almost surely to the extended return; their laws converge weakly on the compactified space `[0,+infinity]`.

**Proof sketch.** First-step decomposition gives the operator identity. Couple every truncation on the same trajectory. Nonnegative cumulative cost is monotone until absorption; failure maps to `+infinity`. The pointwise limit is the declared extended return. Almost-sure convergence implies weak convergence on the compactified state space. Uniqueness outside an absorbing/proper finite setting is not claimed.

## T7 — Dual-Budget Feasibility

Define

`A_DB(x,e,b) = {(a,d): U_C^1(x,a)<=d<=b and U_E^eta(x,a,d,b)+m<=e^-}`.

Assume `(a,d)` is selected from this set, future charger actions follow `kappa_{C,eta}`, collision spending remains within residual budget, and the energy bound has the selected-action validity stated in A13.

Let `F_C` denote any collision before charger, `F_empty` empty filtered support, and `F_E` the event that required finite return energy plus reserve exceeds true usable battery on a collision-free charger path. If:

- `P(F_C | G_C) <= b` by T5;
- `P(F_empty)<=beta_empty` on the declared domain;
- `P(F_E)<=alpha_E+beta_E+beta_e` under T4 and A15;

then

`P(F_C union F_empty union F_E) <= b + beta_C + beta_empty + alpha_E + beta_E + beta_e`,

capped at one.

No independence is used. Energy and collision budgets do not offset each other. A finite `U_E` is disallowed when the selected-action calibration protocol cannot distinguish finite return mass from failure/censoring.

## Two-Resource Stopping Boundary

For task candidates define the continuation set

`A_task_DB(x,e,b) = A_task(x) intersection A_DB(x,e,b)`.

For charger candidates define `A_charger_DB` analogously using the same base policy with goal `g_c`. The commitment time is the first decision at which task continuation has no declared dual-budget-feasible action while charger commitment still has one. This stopping rule consumes neither resource by scalar preference; it reacts to loss of feasibility in either ledger.

## Boundaries

- This is an augmented-state chance-constrained SSP construction; that general device is not novel.
- The theorem is conditional on selection-valid bounds and the declared support/properness domain.
- The operator may be discontinuous when learned collision bounds cross action-admission thresholds.
- There is no general deep fixed-point convergence claim.
