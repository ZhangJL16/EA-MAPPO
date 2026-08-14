# Archived 71D Multi-Scale Navigation Failure

## Status

`ARCHIVED_IMPLEMENTATION_FAILURE`

The 71-dimensional multi-scale SAC implementation remains preserved at commit `1a57998ecfb238b9f1ee3ccda7f24cc1fffb48a7`. Its failed seed-0 run remains under `artifacts/scale_invariant_sac_1m/seed0/` and must not be resumed, deleted, overwritten, or used for formal comparison.

## Failure

- Requested budget: 1,000,000 steps
- Failure step: 20,627
- Failure class: implementation error in distance-bin feasibility
- Exception: `RuntimeError('empty physical distance interval for bin 8-12')`
- Checkpoints produced: none
- Zero-shot evaluation produced: none

The feasibility test admitted an 8-12 m bin using one maximum-distance calculation, while the sampler later multiplied that maximum by 0.98. Near the feasibility boundary, the adjusted upper endpoint became no larger than 8 m. This was a sampler-contract bug, not evidence for or against SAC scale generalization.

## Partial Metrics Are Non-Results

At 20,000 steps, the training stream had 38 completed goals across 55 ended episodes, finite actor/critic losses, and approximately balanced sampled-bin counts. These are interrupted training diagnostics only. The artifact is invalid for navigation-quality or generalization claims.

## Superseding Route

The active route no longer continues the 71D multi-scale policy. It uses a separate 7D relative-goal policy trained only in 4x4, then freezes that checkpoint for 8x8 and 16x16 zero-shot evaluation. The old 77D baseline and this 71D failed route remain historical provenance.
