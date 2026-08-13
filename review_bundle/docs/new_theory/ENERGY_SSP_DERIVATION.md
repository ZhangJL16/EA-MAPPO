# Energy SSP Derivation

## Target

Derive the undiscounted energy-to-charger SSP as the finite, collision-independent special case of the coupled operator.

## Status

COHERENT AS A SPECIAL CASE; INSUFFICIENT AS THE PRIMARY CONTRIBUTION.

## Invariant Object

For fixed charger continuation `pi_c` that is independent of collision budget,

`Z_E(x,a)=sum_{k=0}^{T_c-1} c_k`.

The canonical route instead uses `Z_tilde_E^eta(x,a,d,b)`. The simpler `Z_E` remains useful for E1 and for checking the energy learner before collision coupling is introduced.

## T1 — Bellman Identity

For nonterminal `x`, `S'~P(.|x,a)`, and `A'~pi_c(.|S')`,

`Z_E(x,a) =_D c(x,a,S') + Z_E(S',A')`.

Taking expectations for nonnegative costs gives

`Q_E(x,a)=E[c(x,a,S')+V_E(S')]`,

where `V_E(x')=E_{a'~pi_c}[Q_E(x',a')]`. At `G_c`, future return is zero. This is an SSP identity with `gamma=1`; `gamma<1` is a different estimand.

## T2 — Finiteness

If `c_t<=c_max<infinity` and `E[T_c|x,a]<infinity`, then

`0<=Q_E(x,a)<=c_max E[T_c|x,a]<infinity`.

The proof is the pathwise inequality `sum_{t<T_c}c_t<=c_max T_c` followed by expectation. It proves no reachability fact; properness is assumed on a declared domain.

Heavy-tailed hitting time can give almost-sure arrival but infinite expected energy. Almost-sure termination alone is therefore insufficient for T2.

## Relation to the Coupled Object

If collision probability is zero, filtered support is never empty, and `kappa_{C,eta}` does not depend on `b`, then `Z_tilde_E^eta` is finite almost surely and reduces to `Z_E`. If collision filtering changes future actions, the unindexed object is misspecified by Proposition C1.

## Non-Claims

- No sup-norm contraction for the undiscounted operator in general.
- No charger-policy properness outside the declared domain.
- No collision semantics in the E1 special case.

## Final Research Status

T1--T2 remain valid supporting semantics. They are established SSP facts and do not constitute the blocked route's novelty.
