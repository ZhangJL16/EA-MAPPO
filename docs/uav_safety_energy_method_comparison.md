# UAV Safety + Energy Joint Action Filter: Method Comparison

## Fixed System Contract

- Nominal controller: frozen 7D goal-conditioned SAC at 5 Hz.
- Safety layer: 20 Hz acceleration filter.
- Physics: 20 Hz translational double integrator with horizontal and vertical actuator limits.
- Sensor: 3D LiDAR, 128 horizontal by 8 vertical sectors, 100 m fixed range, nominally refreshed at 10 Hz.
- Energy: `TelemetryCostModel` evaluated on realized executed safe acceleration.
- High-level energy logic: existing TASK/CHARGER_COMMITTED one-way switching remains unchanged.

## Method Table

| ID | Method | Safety object | Solve | Pointwise claim | Energy role | Primary weakness |
| --- | --- | --- | --- | --- | --- | --- |
| B0 | Frozen SAC without filter | none | actor forward pass | none | telemetry only | intentionally unsafe obstacle baseline |
| B1 | One-step reachable projection | linearized next-position halfspaces | 3D polyhedral QP | one-step geometric approximation only | none | misses braking horizon and intersample collision |
| B2 | Second-order HOCBF-QP | one linear HOCBF row per active primitive | convex QP | continuous-time forward invariance if feasible and assumptions hold | none | many constraints; sampled-data gap |
| B3 | Aggregate HOCBF | log-sum-exp smooth aggregate | one-row closed form or small QP | for the represented aggregate safe set under valid approximation | none | approximation conservatism and direct prior-art overlap |
| B4-A | Instantaneous energy-aware HOCBF | B2 constraints | strictly convex QP | same collision claim as B2 | quadratic realized-acceleration proxy | may save acceleration but increase detour energy |
| B4-B | Energy-to-go gradient HOCBF | B2 constraints | strictly convex QP with linear energy term | same collision claim as B2 | local long-horizon energy sensitivity | gradient may be noisy or meaningless off-policy |
| B4-C | Collision plus energy viability | B2 plus convex quadratic battery row | convex QCQP/SOCP if representable | collision plus conditional battery-set invariance | hard returnability | direct prior-art overlap and likely duplicate conservatism |
| B5 | CPO | expected discounted collision cost | trust-region CMDP update | expected-cost constraint, not pointwise invariance | optional second cost | expensive and unsafe during learning |
| B6 | FOCOPS | expected discounted collision cost | first-order constrained update | expected-cost constraint | optional second cost | no deterministic deployment shield |
| B7 | PID-Lagrangian or CVPO | expected collision cost | primal-dual or variational update | expected-cost constraint | optional second cost | tuning and fairness complexity |

## Safety Semantics

### B1: one-step projection

B1 constrains an approximate next position. It is a useful cheap baseline but does not encode stopping distance. A collision-free next sample does not imply collision-free intersample motion or future feasibility.

### B2: second-order HOCBF

B2 is the strongest existing baseline for this action contract because acceleration appears in the second derivative of squared clearance. Its claim remains conditional on:

1. correct obstacle geometry and UAV radius;
2. initial membership in the nested HOCBF sets;
3. a feasible bounded acceleration at every control update;
4. execution of the solved acceleration;
5. a sampled-data condition covering the 50 ms hold and sensor age.

### B3: aggregate HOCBF

B3 trades exact intersection semantics for one smooth aggregate row. It is attractive when primitive count is high, but it must report the aggregate approximation gap and cannot silently claim all individual inequalities are satisfied.

## Multi-LiDAR Strategy

The recommended first implementation is not 1,024 raw QP rows. It uses:

1. ray-to-primitive extraction for spheres, cylinders, and boxes when simulator geometry is available;
2. raw point spheres only as a perception fallback;
3. danger ranking by current HOCBF slack, which already combines clearance and radial closing velocity;
4. nearest/top-K rows for B2;
5. log-sum-exp aggregation as B3;
6. full raw-point scaling only as a compute stress test.

Distance-only top-K is not the primary selector because a farther rapidly closing obstacle can be more urgent than a nearer receding obstacle.

## CMDP Baseline Selection

The selected high-quality baselines are:

1. **CPO** for canonical trust-region CMDP behavior;
2. **FOCOPS** for a scalable first-order constrained policy baseline;
3. **PID-Lagrangian** for a practical dual-control baseline.

CVPO is a strong substitute if a maintained implementation is more reliable than PID-Lagrangian. Reachability-Constrained RL is optional because a faithful implementation adds multiple learned objects and a different state-wise semantics.

CMDP policies must receive the same compact obstacle representation, action limits, maps, starts/goals, reward, and collision cost. They require independent obstacle-aware training and therefore are not part of the frozen-SAC controlled prototype. Their expected-cost result must be reported separately from B2's conditional pointwise guarantee.

### CMDP cost and fairness protocol

Future CPO/FOCOPS/PID-Lagrangian runs must report both of the following without
selecting whichever makes a baseline look weaker:

1. `C1`, a binary collision transition cost using the same geometric overlap
   definition as the filter benchmark;
2. `C2`, a continuous near-obstacle cost
   \(\max(0,d_{\mathrm{warning}}-d_{\min})/d_{\mathrm{warning}}\), with the
   warning distance fixed before training.

Every CMDP policy receives the same compact obstacle geometry or LiDAR encoder,
acceleration action space, actuator limits, obstacle maps, navigation reward,
starts/goals, and evaluation streams. Training violations, expected discounted
cost, realized collisions, and minimum clearance are separate outputs. A CMDP
training constraint is not equivalent to deployment-time forward invariance,
and the frozen-SAC HOCBF experiment is not evidence of safe learning.

An optional energy CMDP may add a second expected cost, but it cannot replace
the collision constraint with a weighted reward. No CMDP was trained in this
prototype phase; the protocol exists to prevent an unfair future comparison.

## Energy Candidate Decisions

### Candidate A

**Selected as the energy-aware engineering variant.** With fixed nondimensional weight 0.1, the sampled-data version reduced held-out mean trajectory energy from 4.8251 to 4.6535 (-3.56%; paired 95% bootstrap interval [-0.2071, -0.1294]) while retaining 125/125 task successes and zero observed collision rollouts. Mean path ratio increased from 1.03884 to 1.04249 and intervention magnitude increased by 6.98%, so this is a measured Pareto trade-off rather than dominance. It is convex, interpretable, and not a novelty claim.

### Candidate B

**Rejected as the primary mechanism.** The retained MC point predictor passed the local finite-difference audit, but gradient correctness did not produce reliable incremental energy savings. At fixed physical scaling, Candidate B changed energy by -0.0396 relative to Candidate A (-0.85%), with a 95% bootstrap interval [-0.1062, 0.0002] and sign-test p=0.371. Unit-direction normalization made mean energy 0.78% worse and increased intervention magnitude by 7.86%. The calibrated correction is nondifferentiable and was correctly excluded. Candidate B may remain only as a negative ablation.

### Candidate C

**Not implemented as a deployment constraint.** It overlaps battery-persistification prior art, duplicates the existing high-level return switch, and would combine deterministic collision semantics with a statistically calibrated learned energy object. Retain only as a documented theoretical comparison unless a future barely-sufficient-battery study demonstrates value beyond high-level switching.

## Current Recommended Method

Use **obstacle-wise sampled-data-aware second-order HOCBF-QP with HOCBF-slack or TTC-aware active-row selection, plus fixed-scale Candidate A at weight 0.1**. Keep all hard collision rows separate from the soft energy objective. Do not use the aggregate barrier as the primary safety representation: it failed catastrophically in symmetric/dense scenes despite lower compute cost. Do not include Candidate B or Candidate C in the deployed method.

This recommendation is an existing-method systems variant. Its supported finding is narrower: in the current controlled UAV benchmark, sampled-data strengthening was necessary to eliminate observed intersample collisions, and a simple physically normalized acceleration-energy term recovered 3.56% trajectory energy without sacrificing observed collision safety. It is not a new generic CBF theorem.

## Feasibility and Fallback Boundary

The prototype does not silently execute the nominal SAC action when the bounded
QP is infeasible. It executes actuator-limited emergency braking and records
both `fallback_used` and whether the braking action satisfies every retained
row. This is a fail-safer engineering fallback, not a certified backup policy.

For the selected held-out Candidate-A run, 34 of 59,948 transitions were marked
infeasible and the braking fallback violated at least one retained row on those
34 transitions. No collision was observed, but the HOCBF theorem does not cover
those transitions. A future deployment claim requires either a recursively
feasible safety construction, a verified backup controller, or an explicit
initial-state restriction that excludes these states.
