# UAV Safety + Energy Joint Action Filter: Controlled Experiment Report

## Status and Scope

This report records the completed prototype experiments for the frozen-SAC
deployment safety layer. No final 500k safety training run was started. The
experiments are deterministic controlled rollouts and solver diagnostics, not
multi-seed RL training results.

The obstacle-free persistent-energy baseline remains unchanged:

- 500,000 Phase-2 transitions;
- 978 completed tasks;
- 63/63 autonomous charger returns;
- zero energy exhaustion and zero charger loops;
- mean charger-arrival SOC 14.12%.

Evidence: `artifacts/uav_energy_phase2_final_500k_20260818_185117_tmux/COMPLETED.json`.

## Protocol and Provenance

Main artifact: `artifacts/uav_safety_filter_1000_controlled_20260819_022302/`.

- 125 held-out scenarios from six fixed families: sparse, medium, dense,
  narrow-passage, long-distance, and vertical-detour;
- eight methods evaluated on identical starts, goals, physics, and geometry;
- exactly 1,000 scenario-method rollouts;
- 603,037 method transitions;
- 544.47 s wall time and 3,629.12 CPU seconds;
- peak resident memory 529.77 MB;
- CPU execution; an NVIDIA GeForce RTX 5060 was available but unused;
- base Git SHA `657b08b3768c90966b32813a97b708bebf7fe175`;
- the prototype worktree was dirty, and the manifest records a source SHA-256.

The dirty-worktree flag prevents treating these artifacts as immutable formal
paper results. It does not invalidate the controlled comparisons because the
complete source hash, command, configurations, and raw rollout CSV are saved.

The aggregate 603,037 transition count is the sum over all eight methods, not a
policy-training budget. The required 1,000 deterministic method rollouts include
long scenes and intentionally failing one-step/aggregate methods that ran to
their fixed timeout. No optimizer or replay buffer was updated, and no 500k
safety-training experiment was launched.

### Scenario separation

The filter itself is not trained on obstacle scenes. Six hand-designed
development scenes and low-count seed-0 pilots were used for implementation and
weight selection. The main and fixed-weight comparisons use a separate
deterministic generator seed (`20260819`) with 125 held-out scenarios distributed
round-robin across all six families. The adaptive normalization ablation uses a
separate analysis seed. These held-out scenes were never used to update SAC or
the energy estimator. A future trained CMDP comparison still requires immutable
train/validation/test geometry manifests; that training experiment was not run.

## Main 1,000-Rollout Comparison

| Method | Rollouts | Success | Collision rollout rate | Minimum clearance | Mean energy | Successful path ratio | Infeasible steps | Mean filter ms | Worst-rollout P95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen SAC, no filter | 125 | 1.000 | 0.912 | -33.103 | 3.7430 | 1.0001 | 0 | 0.000 | 0.000 |
| One-step projection | 125 | 0.328 | 0.912 | -33.493 | 15.6349 | 0.9981 | 143,886 | 11.098 | 37.575 |
| Continuous second-order HOCBF | 125 | 1.000 | 0.184 | -0.112 | 4.8039 | 1.0337 | 570 | 1.137 | 43.258 |
| Aggregate HOCBF | 125 | 0.720 | 0.584 | -24.012 | 9.1293 | 1.0233 | 54,437 | 4.207 | 45.487 |
| Continuous Candidate A, weight 0.3 | 125 | 1.000 | 0.160 | -0.064 | 4.3569 | 1.0469 | 110 | 0.753 | 13.668 |
| Continuous Candidate A+B, unit direction, weight 0.3 | 125 | 1.000 | 0.200 | -0.075 | 4.4399 | 1.0358 | 131 | 1.640 | 7.329 |
| Sampled-data HOCBF | 125 | 1.000 | **0.000** | 0.636 | 4.8251 | 1.0388 | 73 | 1.005 | 31.194 |
| Sampled-data Candidate A+B, unit direction, weight 0.3 | 125 | 1.000 | **0.000** | 0.669 | 4.4634 | 1.0386 | 4 | 1.670 | 9.396 |

The main run establishes three facts. First, the obstacle-free nominal policy is
not safe in obstacle scenes despite completing every task. Second, continuous
HOCBF checks leave intersample collisions under the simulated 20 Hz hold. Third,
the sampled-data residual correction eliminates observed collision rollouts on
this exact-geometry benchmark. One-step projection and smooth aggregation are
not acceptable primary safety methods in these scenarios.

## Candidate A: Instantaneous Energy Objective

Artifact:
`artifacts/uav_safety_energy_ablation_fixed_weight01_heldout_20260819_023653/`.

The clean attribution comparison uses 125 identical held-out scenarios and a
fixed nondimensional energy weight of 0.1.

| Method | Success | Collision rate | Mean energy | Path ratio | Infeasible steps |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sampled-data HOCBF | 1.000 | 0.000 | 4.8251 | 1.03884 | 73 |
| + Candidate A | 1.000 | 0.000 | 4.6535 | 1.04249 | 34 |
| + Candidate A + fixed Candidate B | 1.000 | 0.000 | 4.6138 | 1.03973 | 38 |

Candidate A versus sampled-data HOCBF:

- paired mean energy difference: -0.17160 synthetic units (-3.56%);
- 95% bootstrap interval: [-0.20715, -0.12942];
- lower energy in 121/125 paired scenarios;
- two-sided paired sign-test p = 4.71e-31;
- path-ratio difference: +0.00365 (+0.35%), interval
  [-0.00071, 0.00940];
- mean intervention magnitude: +6.98%;
- collision and task-success rates unchanged at 0% and 100%.

Candidate A therefore offers a measured energy reduction, but not free
dominance: it modestly increases intervention and does not establish improved
path efficiency.

The matched telemetry decomposition provides a more specific mechanism. Relative
to no safety filter, sampled-data HOCBF added 1.0820 total energy per rollout:
0.5815 from the acceleration-energy component and 0.5006 from the remaining
time/velocity/base-power components. Candidate A added 0.9104 total energy:
0.3685 acceleration and 0.5419 remaining components. Thus Candidate A recovered
0.2129 acceleration-energy units while slightly increasing the non-acceleration
detour/time component by 0.0413. The saving is consistent with its physical
objective and is not explained by a shorter path.

### Feasibility, fallback, and latency for the selected method

Across 59,948 selected-method transitions:

- HOCBF residual violation steps: 25;
- bounded-QP infeasible steps: 34;
- emergency-braking fallback steps that did not satisfy every retained row: 34;
- collision steps: 0;
- mean total filter latency: 0.861 ms;
- median total filter latency: 0.699 ms;
- worst-rollout P90/P95/P99: 3.697/7.593/38.953 ms;
- maximum observed latency: 54.082 ms;
- 50 ms deadline misses: 6 (0.0100%).

The fallback never reuses the unsafe nominal SAC action; it applies bounded
emergency braking. However, because braking does not satisfy all rows in these
infeasible states, those transitions are outside the forward-invariance
proposition. Zero observed collisions is an empirical outcome, not proof that
the fallback is safe. There is no slack relaxation in the primary QP, so
infeasibility is exposed rather than hidden.

## Candidate B: Energy-to-Go Gradient

Gradient audit artifacts:

- `artifacts/uav_safety_energy_gradient_audit_small_20260819_020314.json`;
- `artifacts/uav_safety_energy_action_gradient_audit_20260819_020407.json`.

For the action-level finite-difference audit, median relative error was 0.299%,
P95 was 1.94%, P99 was 16.07%, and 90.8% of states were below 1%. This supports
using the point-predictor derivative as a local diagnostic. The nondifferentiable
conformal/Mondrian correction was excluded.

Fixed-scale Candidate B versus Candidate A:

- mean energy difference: -0.03964 (-0.85%);
- 95% bootstrap interval: [-0.10624, 0.00020];
- lower in 57 pairs and higher in 68 pairs;
- paired sign-test p = 0.371;
- path ratio improved by 0.00276, but this does not establish an energy effect.

Unit-direction Candidate B at weight 0.3 versus Candidate A:

- mean energy difference: +0.03466 (+0.78%, worse);
- 90/125 scenarios used more energy;
- intervention magnitude increased 7.86%;
- path ratio improved 1.16%.

**Decision: Candidate B does not add supported energy value and is rejected as
the primary mechanism.** Correct local gradients are insufficient when their
scale, off-policy meaning, and interaction with hard constraints do not produce
a reproducible trajectory-energy improvement.

## Candidate C: Energy Viability Constraint

Candidate C was derived but intentionally not deployed. Generic
battery-returnability barriers overlap persistification and energy-sufficiency
CBF prior work. In this system they also duplicate the already validated
TASK/CHARGER commitment layer and would depend on a statistical learned upper
bound rather than exact deterministic energy dynamics. No evidence justified
adding the resulting QCQP/SOCP constraint.

## Sampled-Data and Sensor Robustness

Artifact: `artifacts/uav_safety_sensor_robustness_30_20260819_024350/`.

Thirty rollouts were run for each condition using the selected sampled-data
Candidate-A filter.

| Condition | Success | Collision rollout rate | Minimum clearance | Mean energy |
| --- | ---: | ---: | ---: | ---: |
| Exact geometry, 10 Hz LiDAR | 1.000 | 0.000 | 0.676 | 4.7437 |
| Bounded radial noise, no added margin | 1.000 | 0.000 | 0.580 | 4.7580 |
| Bounded radial noise + 1.5 m margin | 1.000 | 0.000 | 2.056 | 4.7995 |
| 10% whole-obstacle dropout | 0.967 | 0.333 | -11.355 | 5.5978 |
| Exact geometry, 5 Hz LiDAR | 1.000 | 0.000 | 0.676 | 4.7437 |
| Noise + dropout + 5 Hz + margin | 0.967 | 0.100 | -14.457 | 5.6564 |

The slower exact LiDAR condition is unchanged because obstacles are static and
the simulator retains exact primitive centers; it is not evidence for moving
obstacles or real delayed perception. The dropout result is the decisive
boundary: the filter cannot constrain an obstacle that is absent from the
perception set.

## Multi-Obstacle Representation and Top-K

The aggregate barrier is computationally attractive but failed safety in
symmetric and dense scenes. It is therefore rejected as the primary safety
representation.

The top-K stress artifact uses 100 trials per obstacle count at m = 32, 64,
128, and 256 with K = 8. TTC and nominal HOCBF-slack ranking had zero trials
with an omitted violated row. Distance-only ranking omitted a violated row in
23--30% of trials; closing-speed-only ranking did so in 10--20%. The selected
priority is HOCBF slack, with TTC as the interpretable alternative. This is an
active-row diagnostic, not a proof that arbitrary row dropping preserves
forward invariance.

## Compute Scaling

The 30-repeat row-scaling diagnostic isolates constraint construction and solve
cost. At 1,024 rows:

| Method | Mean ms | P95 ms | P99 ms | Max ms | Mean solve ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full HOCBF QP | 27.899 | 31.931 | 36.489 | 37.806 | 1.566 |
| Top-K-16 QP | 29.665 | 31.050 | 33.279 | 34.142 | 0.232 |
| Aggregate QP | 23.868 | 29.355 | 33.162 | 33.709 | 0.199 |
| Aggregate one-row closed form | 23.047 | 24.740 | 25.536 | 25.801 | 0.068 |

All feasible synthetic scaling samples met the 50 ms deadline. Top-K reduces
solver time but not total latency because the prototype still constructs and
sorts all rows. The aggregate closed-form diagnostic omits actuator rows and is
not a deployable safety result. Rare rollout deadline misses remain associated
with infeasibility tails and runtime cold starts, so this is empirical 20 Hz
feasibility rather than a hard real-time guarantee.

## CMDP Baselines

CPO, FOCOPS, and PID-Lagrangian were selected as the representative learned
baselines. They were not implemented in this controlled frozen-policy phase.
A fair comparison requires independent obstacle-aware training with the same
compact obstacle information, action limits, maps, reward, collision cost, and
interaction budget. Therefore:

- no superiority over CMDP methods is claimed;
- no CMDP number is imputed from another paper;
- expected CMDP-cost satisfaction remains semantically distinct from the
  conditional pointwise HOCBF statement.

## Final Research Decision

1. **Best existing safety method:** obstacle-wise sampled-data second-order
   HOCBF-QP with bounded actuator and next-velocity constraints.
2. **Best energy-aware variant:** the same filter plus normalized instantaneous
   acceleration-energy Candidate A at weight 0.1.
3. **Rejected mechanisms:** one-step projection, smooth aggregate as the primary
   shield, learned energy-gradient Candidate B, and hard energy-viability
   Candidate C.
4. **Novelty:** not supported as a generic method. The defensible result is a
   robotics/autonomous-systems measurement: sampled-data correction was required
   for observed collision elimination, while a simple physical secondary cost
   recovered 3.56% energy.
5. **Formal 500k safety experiment:** **NOT STARTED**.

## Recommended Next Experiment

Do not start a final 500k run to promote Candidate B. If the project proceeds,
the next evidence should be a clean-commit, multi-seed robotics benchmark of the
selected existing-method variant, with matched obstacle-aware CPO/FOCOPS or
PID-Lagrangian training, fixed train/validation/test maps, bounded-noise and
dropout regimes, and the current frozen obstacle-free energy baseline retained.
The paper should be scoped as a systems comparison unless a new failure-driven
mechanism emerges.
