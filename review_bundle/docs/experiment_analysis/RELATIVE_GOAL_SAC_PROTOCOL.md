# 4x4 Relative-Goal SAC Protocol

## Active Route

`4X4_RELATIVE_GOAL_SAC`

The navigation policy is trained only in a 4x4x2 m obstacle-free world. After training, its checkpoint is frozen and evaluated without updates in 4x4, 8x8, and 16x16 worlds. Navigation SAC fine-tuning in 8x8 or 16x16 is forbidden.

The old 77D frozen baseline and archived 71D multi-scale implementation remain reproducible historical routes.

## 7D Observation

The policy observes only:

| Field | Dimension | Definition |
|---|---:|---|
| Normalized velocity | 3 | componentwise velocity / fixed `v_max` |
| Relative goal direction | 3 | `(goal-position) / max(||goal-position||, eps)` |
| Relative goal distance | 1 | `clip(||goal-position|| / 2m, 0, 1)` |

Absolute position, absolute goal, station, state of charge, LiDAR distances, and LiDAR validity are excluded. Internal simulator LiDAR may continue to exist for other modules, but no LiDAR value is read by the SAC observation.

## Near/Far Training Distribution

One policy is trained under two sampling strata:

- `NEAR`: 0.30 m to 2.00 m;
- `FAR`: 2.00 m to the shared effective physical maximum of the 4x4 sampler;
- selection probability: 0.50 NEAR / 0.50 FAR.

The sampler computes one effective interval table. Feasibility checks and sampled endpoints use the same table. Start points are sampled from an axis-wise interval constructed from the selected displacement, so every returned start/goal pair is legal by construction and then revalidated.

## Fixed Physics and Reward

Training and evaluation retain fixed `v_max`, `a_max`, `dt`, body radius, goal radius, and 2 m world height. Reward uses physical distance progress, velocity toward the goal, a small time cost, goal completion bonus, and boundary penalty. Energy reward weight is zero.

## Training and Evaluation Separation

- Training data: 4x4 only.
- Periodic training evaluation: 4x4 only; monitoring only, never target-scale model selection.
- Post-training frozen evaluation: 4x4, 8x8, and 16x16.
- Evaluation performs no optimizer, replay-buffer, checkpoint, or policy update.
- Evaluation reports completion, timeout, final distance, steps, path efficiency, boundary contact, velocity saturation, action-goal alignment, far progress, negative-progress runs, and overshoot events.

## Research Gate

No Energy Critic recollection or autonomous switching begins until completed frozen evaluation determines whether `NAVIGATION_ZERO_SHOT_4_TO_16_VALID` is true or false. A false result must be reported without 16x16 navigation fine-tuning.
