# E1 Research Decision

Date: 2026-08-13

## Answers

1. **Is E1 complete?** No. All three processes are dead; no seed has a completion marker or result summary.
2. **Which seeds are valid?** None. Seed 0 stopped after 43/100 raw trajectories; seeds 1 and 2 produced no raw trajectories.
3. **Which method is best?** Unknown. None of B0–B4 ran to a result.
4. **Is TD necessary?** Unclear. `TD_LEARNING_ADVANTAGE_NOT_ESTABLISHED`.
5. **Is distributional TD necessary?** Unclear. `ENERGY_DISTRIBUTION_LEARNING_SUPPORTED = FALSE` because there is no evidence, not because it was disproved.
6. **Did calibration improve safety metrics?** Not evaluated.
7. **Did learned switching beat SOC/distance rules?** Not evaluated; these switching baselines are not implemented in E1.
8. **Was the method overconservative?** Unknown; no bound or throughput result exists.
9. **Is there a new empirical failure phenomenon?** No learning phenomenon is established. The established failures are experiment lifecycle, finite-energy protocol mismatch, absent switching semantics, and absent evaluation outputs.
10. **Best paper category now?** If corrected experiments become positive, the credible route is robotics/autonomous systems or UAV systems, not ML theory. No paper-level empirical claim is currently supported.
11. **Next minimum experiment?** A corrected E1-v2 in the same obstacle-free setting, with finite battery/SOC coverage, task prefixes, exact held-out seeds, B0–B4, fixed-SOC/distance switching baselines, tail metrics, and one-way switching telemetry.
12. **Should E2 start?** No.
13. **Why not E2?** The causal energy-only stage has neither valid provenance nor valid results, and the navigation/energy protocol needs correction first.
14. **Why not E3?** Joint collision+energy evaluation would destroy attribution before either component has valid standalone evidence.

## Data-Driven Interpretation

The only numerical signal currently available is diagnostic: across 43 interrupted seed-0 return trajectories, return energy correlates strongly with Euclidean charger distance (0.9603) and path length (0.9624). This makes a strong distance baseline scientifically mandatory and raises the possibility that the current deterministic open world is too simple for TD to add value. It does **not** establish equivalence because no held-out method comparison exists.

The immediate decision is therefore not `CONTINUE_ENERGY_TD`, `REDESIGN_ENERGY_ENVIRONMENT`, or `INVESTIGATE_OBSERVED_FAILURE_PHENOMENON`. The data are too incomplete for those choices.

## Corrected E1-v2 Gate

The restart must use new immutable artifact directories and must not reuse the old raw file in formal aggregates. Before launch it must pass tests for:

- finite initial energy and SOC accounting without changing the retained baseline default contract;
- exact split and evaluation seed persistence;
- all five estimator outputs and requested safety-tail metrics;
- charger terminal and gamma=1 targets;
- fixed-SOC, distance, learned, and calibrated switching rules;
- one-way commitment with at most one TASK→CHARGER switch;
- interruption markers and resumable collection checkpoints.

Only after a valid three-seed E1-v2 should the project decide among:

- distance model sufficient → test realistic heterogeneous energy factors before retaining TD;
- TD only improves prediction → position as a UAV energy-estimation component;
- stable tail/calibration/policy-drift failure → run a minimal replication experiment before defining a new research problem.

## Binary Decision

`CURRENT_RESEARCH_DECISION = E1_INVALID_RESTART_REQUIRED`

`FINAL_E1_CONCLUSION = PENDING`

`E2 = NOT_STARTED`

`E3 = NOT_STARTED`
