# Collision Risk Derivation

## Target

Obtain an action-level calibrated swept-collision hazard and compose it over a variable-length, adaptively controlled sortie.

## Event Semantics

The theorem-level atomic event is

`B_{t+1}=1` iff the swept vehicle body intersects an obstacle during the executed transition from `t` to `t+1`.

Near-collision `N_{t+1}` is auxiliary unless given its own budget. A model may predict an `H_C>1` event for screening, but overlapping windows are not treated as independent or subtracted repeatedly from the remaining budget. Exact `H_C>1` accounting requires non-overlapping macro-action blocks.

Inputs are local perception/history, kinematics, candidate action, validity masks, and declared context. Privileged geometry is limited to simulator labels/evaluation.

## Calibrated Upper Hazard

The deployed selector requires, on validity event `G_C`,

`P(B_{t+1}=1 | F_t,a_t) <= U_C^1(x_t,a_t) <= d_t`.

The first inequality must cover the adaptively selected action. Aleatoric and epistemic estimates may construct the score, but neither ensemble variance nor ECE is itself this bound.

## T5 — Sequential Collision-Risk Composition

Let `L` be the charger/timeout decision horizon that would stop the process absent collision, and let actual execution stop at `T=min(L,T_B)`. Before collision, choose predictable spends `d_t>=0` satisfying

`sum_{t<T} d_t <= Delta_C` almost surely.

Then on `G_C`,

`P(T_B<=L | G_C) <= E[sum_{t<T} d_t | G_C] <= Delta_C`.

If `P(G_C^c)<=beta_C`, then

`P(T_B<=L) <= Delta_C+beta_C`.

**Proof.** Let `I_t` indicate survival without collision to decision `t`. The first-collision events are disjoint, so

`P(T_B<=L|G_C)=E[sum_{t<T} I_t P(B_{t+1}=1|F_t,G_C)]`.

Selection-validity and admission bound each conditional hazard by `d_t`. Since `I_t<=1`, predictable spending yields the result. No independence between steps is required.

This stopped-hazard proof is at least as tight as union-bounding overlapping prediction windows and remains valid for variable sortie length.

## Allocation

Valid examples include finite-horizon equal spending, geometric spending, and state-adaptive predictable spending. A fixed per-step epsilon over an unbounded sortie is not a finite sortie guarantee.

## Distribution Shift

The theorem is conditional on `G_C`. Shift detection may suspend action admission or trigger recalibration; it cannot make a false conditional hazard bound true. Calibration validity, sensor faults, and selector effects must be separately reported.

## Final Research Status

T5 remains a valid conditional accounting theorem. Selected-action hazard calibration and risk spending are established ingredients, not an ICLR-level novelty claim.
