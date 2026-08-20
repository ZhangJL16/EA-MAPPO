# Recoverable Safety Final Decision

## Decision

```text
FAILURE_PHENOMENON_STABLE = TRUE
ROUTE_A_GENERAL_THEORY_GO = FALSE
ROUTE_B_GENERAL_THEORY_GO = FALSE
ROUTE_C_GENERAL_THEORY_GO = FALSE
RECOVERABILITY_CONTROLLER_IMPLEMENTATION_GO = FALSE
FORMAL_MULTI_SEED_EXPERIMENT_GO = FALSE
ROBOTICS_BENCHMARK_DIRECTION = CONDITIONAL
```

## Why the Failure Study Passes

- Two independent searches found hundreds of positive-to-negative margin events.
- All ten required scenario families exist, and seven produce events.
- Matched perception modes reveal repeatable differences.
- Hard states preserve state, obstacle, sensing, HOCBF, actuator-set, margin, TTC, clearance, braking and active-constraint evidence.

## Why the Proposed Theory Fails the Novelty Gate

- Route A is a robust predecessor/viability construction.
- Route B is a horizon-one predictive feasibility condition.
- Backup CBF and Predictive CBF already provide stronger invariant-set or horizon-based mechanisms.
- Robust sampled-data and interval-analysis CBF work already handles bounded uncertainty and sample-and-hold margins.
- Measurement-robust and occlusion-aware work already handles conservative perception uncertainty and contingency behavior.
- Route C is contradicted as the dominant explanation by the current event data.

## Why Long Experiments Were Stopped

The stop criteria require halting when closest work has a stronger theorem, a candidate is mathematically equivalent to existing predictive/backup/viability methods, or online computation misses 20 Hz. All three conditions apply. No new network, controller, or 500k experiment was started.

## What Is Still Scientifically Valuable

The dataset exposes a practical distinction that is often hidden by aggregate collision metrics:

- the raw HOCBF action set can remain feasible;
- sampled-data strengthening can remove all certified actions;
- the filter then spends many steps in constraint-violating fallback;
- no immediate collision may occur, so collision rate alone misses the loss of safety authority;
- perception-history choices materially alter row-disappearance failures.

This supports a negative-result or benchmark paper only after implementing closest-work baselines and adding outcome-level task, deadlock, energy and collision evidence.

## Minimum Reopening Condition

Reopen the method route only if a concrete candidate satisfies all of:

1. a theorem/property not reducible to robust predecessor, Predictive CBF, Backup CBF, DTCBF, or standard robust CBF;
2. uncertainty sets demonstrably contain true obstacles during dropout;
3. significantly fewer positive-to-negative entries than faithful closest-work baselines;
4. nonzero progress without freeze/deadlock;
5. collision and energy outcomes measured on matched scenarios;
6. P95 and P99 computation within the 50 ms deadline.

