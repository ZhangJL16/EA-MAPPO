# Formal MC Run 20260817 Metric Correction

## Historical artifact

This correction applies to
`artifacts/uav_energy_delivery_mc_formal_20260817_154321/`. The historical
artifact is retained unchanged.

## Correct cycle interpretation

The original summary reported `battery_cycles_completed = 75`. Inspection of
all underlying segment records shows that this number combined two different
events:

- 50 completed charger arrivals followed by an actual recharge;
- 25 guard-truncated partial battery segments.

The corrected terminology is therefore:

```text
completed_recharge_cycles = 50
guard_truncated_segments = 25
partial_battery_segments = 25
```

An emergency-guard partial segment is not a completed battery cycle.

## Artificial battery-reset audit

At every 20,000-policy-step guard event the environment returned
`truncated=True`. The Phase2 runner then called `reset()`. Runtime replay of the
exact reset path confirms that every reset:

- restored remaining energy to the full calibrated capacity
  `378.72626091628933`;
- restored remaining-energy fraction to `1.0`;
- reset the environment-local battery-cycle id to `0`.

Therefore:

```text
FORMAL_PHASE2_HAS_ARTIFICIAL_BATTERY_RESETS = TRUE
```

All 25 event-level before/after records are preserved in
`docs/formal_mc_run_20260817_guard_reset_audit.json`.

## What remains valid

The following are direct observed events and remain valid functional evidence:

- 50/50 attempted autonomous returns reached the charger;
- zero energy-exhaustion terminals were observed;
- zero charger-loop events were observed;
- 577 tasks were completed after at least one real recharge in the active Gym
  episode;
- the MC point estimator held-out metrics and the frozen-SAC navigation results
  are unaffected by the Phase2 accounting error.

## Confounded metrics

The old aggregate `tasks/1000 transitions`, energy-utilization summary,
battery-cycle count, and exhaustion-rate denominator are not accepted as the
final continuous-battery benchmark because the 25 guard resets supplied full
batteries without physical charger arrivals. The corrected formal run uses a
single-environment guard strictly greater than the full 500,000-transition
budget, so normal execution cannot trigger a guard reset.
