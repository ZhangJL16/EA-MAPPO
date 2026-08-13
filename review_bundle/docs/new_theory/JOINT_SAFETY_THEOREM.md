# Joint Safety Theorem

## Dual-Budget Action Set

Define

`A_DB(x,e,b) = {(a,d): U_C^1(x,a)<=d<=b and U_E^eta(x,a,d,b)+m<=e^-}`.

The energy bound is indexed by the residual collision budget and induced charger-kernel version. This is not the old independent intersection `U_C(s,a)` with `U_E(s,a)`.

## T6 — Action-Level Joint Feasibility

If `(a,d) in A_DB`, then two separate implications hold under their stated validity events:

1. immediate swept-collision hazard is at most `d`;
2. conditional on collision-free successful charger continuation under `kappa_eta`, the energy exceedance probability is at most `alpha_E` plus declared calibration and battery-bound failures.

Neither implication offsets the other. The set definition alone is elementary; the coupled, budget-indexed target and its calibration are the research object.

## T7 — Continuation Failure Bound

Under A1--A17, T4, and T5, executing an admitted action and then committing to `kappa_eta` gives

`P(collision before charger OR empty support OR finite-return energy insufficiency)`

`<= b + beta_C + beta_empty + alpha_E + beta_E + beta_e`,

capped at one, on the declared augmented properness domain. If non-arrival probability is not included in `beta_empty`, add an explicit `beta_nonarrival`; do not hide it inside properness prose.

**Proof.** T5 bounds collision under residual spending. T4 bounds energy insufficiency on successful collision-free paths. Empty support and non-arrival are separate events. Apply the union bound. Independence is unnecessary.

## Empty Set

If `A_DB` is empty, the method records infeasibility. It neither executes the least-violating action nor invokes an undeclared recovery mechanism. The theorem is not an existence theorem for feasible actions.

## Non-Vacuity Check

Every reported bound must include its numerical right-hand side, empirical finite-return mass, and frequency of empty sets. A bound greater than or equal to one, or a method that nearly always returns no action, is formally valid but scientifically vacuous.

## Final Research Status

T6--T7 remain conditional accounting results. The conjunction/union-bound structure is not novel, and the route is blocked from an ICLR theory claim by the overlaps in `RESEARCH_DIRECTION_BLOCKED.md`.
