# Reviewer A — Theory (Round 2)

## Decision

**Score: 7/10 (theory package only). Confidence: 4/5.**

The reformulation fixes the largest semantic defect: collision-filter failures can no longer disappear from an energy dataset, and the return law is indexed by the budget and policy/filter version that generate it. T1--T9 are mathematically scoped rather than presented as deep-learning guarantees.

## Reject-First Audit

### FATAL

None in the canonical theorem statements.

### CRITICAL

None after three repairs: atomic swept hazards replace overlapping-window spending; non-arrival/empty support receive explicit missing mass; successful-return calibration is separated from success probability.

### MAJOR

1. **Conditional-success calibration is operationally weak.** A finite `U_E` can be tight while success mass is tiny. The joint method must estimate/calibrate missing mass and refuse vacuous regimes.
2. **A4 remains a strong domain assumption.** Finite expected absorption under the augmented kernel is not learned or proved.
3. **T5 validity is selector-level.** A marginal collision classifier cannot instantiate A8.
4. **C3 is a finite-candidate strict-margin theorem.** Continuous action implementations need a measurable support/margin analogue or must state C3 as a discretized candidate-selector result.

### MINOR

The `beta_empty` and `beta_nonarrival` accounting should use one canonical notation in any paper draft.

## Strongest Counterexample

At a physical state with two charger actions, a calibrated collision-bound change of `10^-6` crosses the admission threshold for the only short route. The finite-return energy law jumps from one joule to one hundred joules, while a replay critic trained under the old filter remains perfectly low-loss on stale data. C2 correctly exposes this; any implementation that mixes versions invalidates the target.

## Closest Theory

- Budget-dependent state/policy and Bellman operators already appear in Carrara et al.: https://arxiv.org/abs/1903.01004
- Multiple distributional constraints already appear in Kim et al.: https://arxiv.org/abs/2301.10923
- Policy-coupled coverage is directly addressed by Zheng and Jin: https://arxiv.org/abs/2607.02206

## Score-Change Conditions

- Raise to 8: prove a nontrivial continuous-action target-transport/stability result that connects collision admission-boundary mass to return-law and calibration drift.
- Lower to 4: omit failure mass, calibrate individual replay transitions as i.i.d., or call A4 a reachability guarantee.
