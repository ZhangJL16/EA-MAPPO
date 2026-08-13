# Scale-Invariant Goal-Conditioned SAC Protocol

## Status

This protocol replaces the failed attempt to transfer the old 4x4 frozen SAC directly to 16x16. It does not modify or supersede the reproducibility contract of the old 77-dimensional environment or any historical Stage A/B artifact.

`OLD_NAVIGATION_77D_COMPATIBILITY = TRUE`

## Navigation Responsibility

The new SAC answers only: given local motion/perception and the current goal, how should the UAV fly toward that goal?

It does not observe or decide battery sufficiency, charger identity, return commitment, or collision-safety authority. A task goal and a charger are represented identically as `current_goal`.

## Observation Contract

`SCALE_INVARIANT_NAVIGATION_71D_V1` contains:

| Field | Dimension | Definition |
|---|---:|---|
| Normalized velocity | 3 | componentwise physical velocity / fixed `v_max` |
| Relative goal direction | 3 | `(goal - position) / max(||goal-position||, eps)` |
| Relative goal distance | 1 | `clip(||goal-position|| / 6 m, 0, 1)` |
| Normalized LiDAR distance | 32 | physical range reading / fixed 6 m LiDAR range |
| LiDAR valid mask | 32 | sensor-validity bits |

Excluded fields are absolute position, absolute goal position, station position, and state of charge.

The fixed 6 m goal-distance scale is tied to the actual physical LiDAR horizon in `NavigationConfig`, not map size. It preserves distance resolution over the local sensing/control horizon. Beyond 6 m, the distance feature saturates and the goal direction remains available for long-range steering.

## Environment Distribution

- XY world width is sampled continuously from `Uniform(4 m, 16 m)` at every episode reset.
- Height remains 2 m.
- Start and goal positions respect a 0.30 m boundary margin.
- Initial velocity is uniformly sampled within 10% of each fixed componentwise velocity limit.
- A feasible physical-distance bin is selected uniformly per episode: 0.5-2, 2-4, 4-8, 8-12, or >12 m.
- The >12 m bin is used only when the sampled world can physically support it.
- Each episode ends on first goal completion or at 800 steps, so the next episode resamples world, start, goal, velocity, and distance bin.
- The world is obstacle-free; no Collision Module or finite-energy termination is active.

The named 4/8/12/16 m scales are evaluation anchors inside the training support. The 6/10/14 m evaluations are exact-scale interpolation checks: they are not out-of-support OOD tests under continuous training randomization.

## Fixed Physics

Across every world size:

- `v_max = [0.30, 0.30, 0.12] m/s`
- `a_max = [0.18, 0.18, 0.08] m/s^2`
- `dt = 0.2 s`
- `lidar_range = 6.0 m`
- `body_radius = 0.05 m`
- `goal_radius = 0.20 m`

## Reward

The old map-dependent observation is not reused. The new reward uses physical-unit semantics:

- progress: `distance_before_m - distance_after_m`, weight 1.0;
- signed velocity toward current goal, weight 0.1;
- per-step time cost 0.01;
- first goal-entry bonus 10.0;
- boundary-contact penalty 1.2;
- energy-cost weight 0.0;
- control penalty 0.0.

The same physical transition therefore receives the same reward semantics regardless of map width.

## Evaluation Protocol

The independent evaluator loads a checkpoint and reports completion, steps, path efficiency, final distance, timeout, boundary contacts, and distance-bin success on:

- anchor scales: 4, 8, 12, 16 m;
- exact-scale interpolation checks: 6, 10, 14 m.

No 1M evaluation result exists until the background training has completed and a later session explicitly runs the evaluator.

## Research Route

Scale-invariant SAC training -> cross-map navigation validation -> freeze new SAC -> recollect 4x4 return-energy trajectories -> Energy Critic training -> 16x16 zero-shot energy transfer -> target adaptation -> autonomous switching -> obstacle environment -> Collision Module.
