# Recoverable Safety Counterexamples

## Scenario Families

The generator contains all required families:

1. two-obstacle squeeze;
2. symmetric corridor;
3. crossing dynamic obstacles;
4. obstacle emerging from occlusion;
5. temporary LiDAR whole-object dropout;
6. obstacle disappears and reappears;
7. suddenly accelerating obstacle;
8. high-speed UAV approach with bounded braking;
9. boundary plus obstacle conflict;
10. simultaneous vertical and horizontal avoidance conflict.

Dynamic obstacle specifications enforce bounded speed, acceleration, and jerk. Acceleration events model abrupt but bounded behavior. The UAV uses repository timing, velocity limits, acceleration limits, policy hold, sensing period, filter period, and frozen SAC actions.

## Search Event

Each event is deduplicated at the first negative entry and must have a positive margin within the configured preceding lookahead window. In the matched artifact all detected entries have horizon one:

\[
\rho_t>0,\quad \rho_{t+1}<0.
\]

## Hard-State Record

Each JSONL record contains:

- UAV position, velocity, current action;
- true and perceived obstacle position, velocity, acceleration and radius;
- explicit UAV-obstacle relative velocities;
- visibility and stale-age sensing history;
- strengthened HOCBF rows, bounds, identifiers and active constraints;
- exact cylindrical physical margin set plus the filter's 32-facet actuator and velocity-limit rows;
- current `rho` and successor `rho` sequence;
- TTC, clearance, required braking and available braking;
- feasibility/fallback flags;
- sampled one-step counterfactual action and its successor margin;
- failure-cause labels.

The exact `rho` diagnostic optimizes over the physical cylindrical input set. The existing QP filter uses a 32-facet inscribed actuator polygon plus state-dependent velocity-limit rows. The artifact records both representations and the actual finite rows used by the filter, rather than silently treating the polygon as the physical cylinder.

## Matched Results

Artifact: `artifacts/uav_recoverable_safety_matched250x4_seed1_v3_20260820/`

| Quantity | Result |
| --- | ---: |
| Scenario instances | 260 |
| Perception modes | 4 |
| Matched rollouts | 1,040 |
| Filter steps | 83,200 |
| Positive-to-negative events | 451 |
| Events with sampled alternative | 356 (78.94%) |
| Raw/unstrengthened margin nonnegative | 451 (100%) |
| Braking authority lost at entry | 0 |
| Multi-obstacle conflicts | 40 |
| Constraint-disappearance events | 131 |
| Explicit reappearance events | 4 |
| Physical collision rollouts | 0 |
| Near-collision steps (`clearance <= 2m`) | 0 |
| Minimum rollout clearance | 6.773 m |
| Fallback steps | 8,973 |
| Fallback steps violating generated constraints | 8,973 |

## Perception Comparison

| Perception mode | Rollouts with event | Rate |
| --- | ---: | ---: |
| Current frame | 151/260 | 58.08% |
| Hold last | 104/260 | 40.00% |
| Constant velocity | 97/260 | 37.31% |
| Bounded acceleration | 99/260 | 38.08% |

Whole-object dropout falls from 26/26 current-frame failures to 0/26 for every history mode. Crossing obstacles and sudden acceleration remain difficult for every mode. This supports the constraint-disappearance mechanism but rejects the claim that simple uncertainty inflation solves future infeasibility.

## Cause Taxonomy

The mutually exclusive headline cause counts are:

- sampled-data residual strengthening: 280;
- constraint disappearance during dropout/occlusion: 127;
- multiple obstacle constraint conflict: 40;
- obstacle reappearance: 4.

Flags are also retained separately, so disappearance, stale sensing, active multi-obstacle constraints, and raw-margin status remain auditable even when one headline label is selected.

## Interpretation Limits

- `fallback_satisfies_constraints=False` means the returned action violates at least one generated safety row; it does not itself mean collision.
- The sampled alternative uses realized next obstacle states and is an oracle diagnostic, not an online controller.
- The scenarios are short hard-state diagnostics, not task-completion experiments.
- Collision and energy improvements have not been demonstrated.
