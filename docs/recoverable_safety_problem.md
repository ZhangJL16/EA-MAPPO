# Recoverable Safe UAV Control under Intermittent Perception

## Status

- Date: 2026-08-20
- Frozen navigation policy: `artifacts/uav_energy_delivery_v3_formal_20260816_004619/phase1_navigation/checkpoint_transition_500000.zip`
- Neural training in this study: none
- Failure phenomenon stable: **YES**
- New general recoverability theory: **NO-GO after closest-work attack**
- Robotics failure-analysis/benchmark direction: **CONDITIONAL**, not yet a method paper

## Problem Definition

For the current sampled-data second-order HOCBF constraints

\[
A_i(x,\mathcal O)u\ge b_i(x,\mathcal O),\qquad u\in\mathcal U,
\]

the implemented pointwise feasibility margin is

\[
\rho(x,\mathcal O)=
\max_{u\in\mathcal U}\min_i\left(A_i(x,\mathcal O)u-b_i(x,\mathcal O)\right).
\]

The input set is the exact cylindrical set used by the controller,

\[
\mathcal U=\{u:\|u_{xy}\|_2\le a_{xy}^{\max},\ |u_z|\le a_z^{\max}\}.
\]

Therefore `rho >= 0` is equivalent to non-emptiness of the current strengthened HOCBF action set. It is only a pointwise diagnostic. It does not imply

\[
\rho(x_t,\mathcal O_t)\ge0
\Longrightarrow
\rho(x_{t+1},\mathcal O_{t+1})\ge0.
\]

The tested failure event is a negative entry

\[
\rho_t>0,\qquad \rho_{t+1}<0,
\]

after applying the existing frozen-SAC plus sampled-data HOCBF action. The true obstacle state is used only for offline diagnosis. The online filter receives the selected perception mode.

## Reproducible Counterexamples

The matched diagnostic contains 260 scenario instances under four perception modes, giving 1,040 matched rollouts and 83,200 filter steps:

- 451 rollouts contain a positive-to-negative margin transition.
- Current-frame perception: 151/260 rollouts, 58.08%.
- Hold-last: 104/260, 40.00%.
- Constant-velocity propagation: 97/260, 37.31%.
- Bounded-acceleration propagation: 99/260, 38.08%.
- 356/451 events have at least one sampled alternative action whose realized one-step successor has nonnegative strengthened margin.
- 451/451 negative-entry states retain nonnegative *unstrengthened* HOCBF margin.
- 0/451 negative entries are classified as loss of available braking authority.
- 40/451 involve multiple active obstacle constraints.
- 131/451 contain obstacle-constraint disappearance; four are explicit reappearance events.

The independent current-frame search over 1,010 scenarios found 556 events in 555 rollouts. This independently establishes that the event is not confined to the ten hand-written instances.

Artifacts:

- `artifacts/uav_recoverable_safety_random1000_seed0_v2_20260820/`
- `artifacts/uav_recoverable_safety_matched250x4_seed1_v3_20260820/`

## Why the Current HOCBF Loses Feasibility

The dominant empirical mechanism is not an already impossible raw HOCBF constraint. Every matched negative entry remains feasible under the unstrengthened continuous-time row. The infeasibility is introduced by the sampled-data residual strengthening needed to account for sample-and-hold evolution.

Intermittent perception is an important amplifier:

1. Current-frame filtering removes an obstacle row when the object is not observed.
2. The nominal controller can accelerate or turn without that row.
3. Reacquisition, or evaluation against the true obstacle set, restores the row.
4. The strengthened sampled-data row can then make the admissible action set empty.

History propagation helps whole-object dropout, but it does not solve crossing, sudden-acceleration, squeeze, or high-speed cases. Bounded-acceleration inflation was slightly worse than constant-velocity propagation overall in this matched sample, so it cannot be claimed as a monotonic improvement.

The experiment recorded no physical collisions and a minimum rollout clearance of 6.77 m, but it recorded 8,973 fallback steps, all marked as not satisfying every generated HOCBF constraint. This means the demonstrated problem is currently **loss of filter feasibility/control authority**, not a demonstrated increase in collision rate.

## Candidate Mathematical Object

The most data-aligned candidate is the robust future-feasibility margin

\[
R_1(x,\mathcal O)=
\max_{u\in\mathcal U_{\mathrm{sd}}(x,\mathcal O)}
\min_{w\in\mathcal W(x,\mathcal O)}
\rho\!\left(F_h(x,u,w),\Psi_h(\mathcal O,w)\right),
\]

where `U_sd` is the current sampled-data HOCBF action set, `F_h` is the sample-and-hold successor, `W` contains admissible obstacle and sensing evolution, and `Psi_h` updates the conservative obstacle-information set.

Requiring `R_1 >= eta >= 0` would be a one-step robust predecessor condition. It is strictly more informative than current `rho` in the empirical sense that 356 negative entries have a sampled action that changes the sign of the one-step successor margin.

This object is **not claimed as novel**. It is a direct robust predecessor/future-feasibility construction.

## Closest Prior Work

The central mathematical ideas are already covered by:

- [Predictive Control Barrier Functions](https://arxiv.org/abs/2105.10241): predictive feasibility, terminal CBFs, always-feasible auxiliary problem, and recovery toward a feasible set.
- [Backup Control Barrier Functions](https://arxiv.org/abs/2104.11332): online forward integration under a backup policy to define a control-invariant set and guarantee CBF-QP feasibility.
- [Discrete-time CBFs for Guaranteed Recursive Feasibility in NMPC](https://arxiv.org/abs/2309.09268): DTCBF/qDTCBF terminal certificates for recursive feasibility under input constraints.
- [Robust CBFs for Sampled-Data Systems](https://arxiv.org/abs/2309.08050): high-order sampled-data barriers under bounded disturbances and measurement errors.
- [CBF Meets Interval Analysis](https://arxiv.org/abs/2110.00915): sampled-data CBF margins computed from reachable overapproximations under measurement and actuation uncertainty, including a Crazyflie experiment.
- [Measurement-Robust CBFs](https://proceedings.mlr.press/v155/dean21a.html) and [MR-CBF with backup sets](https://arxiv.org/abs/2104.14030): safety under uncertain state estimates.
- [Disturbance-Robust Backup CBFs](https://arxiv.org/abs/2409.07700): robust tubes around backup-policy flows and robust invariant sets.
- [Occlusion-Aware Contingency Safety-Critical Planning](https://arxiv.org/abs/2502.06359): forward reachable sets for occluded agents plus coordinated exploration/fallback trajectories.

## Novelty Risks

1. Route A is standard robust predecessor/viability analysis.
2. Route B is one-step predictive safety and is covered in stronger horizon-based forms by predictive CBF and DTCBF work.
3. Route C is not the dominant mechanism in the data: braking authority is not lost at any of the 451 negative entries.
4. Propagating bounded obstacle uncertainty is covered by robust sampled-data, interval-analysis, measurement-robust, and occlusion-aware methods.
5. The sampled one-step action search is not online-ready: median 237.2 ms and P95 1.166 s versus a 50 ms deadline.
6. The current experiment demonstrates conservative infeasibility and fallback, not collision, task failure, or energy benefit.

## Next Experiment

Do **not** implement a newly named recoverability controller or run long experiments. The only defensible next experiment is a closest-work benchmark:

1. implement or faithfully reproduce Predictive CBF/Backup CBF and a robust sampled-data or measurement-robust baseline;
2. use the same matched hard scenarios and true uncertainty containment checks;
3. compare negative-entry rate, fallback, collision, progress, deadlock, energy overhead, and P95/P99 latency;
4. proceed only if a new approximation has a property not already supplied by those baselines and satisfies the 20 Hz tail deadline.

## GO / NO-GO

- Stable failure phenomenon: **GO**.
- Route A as new theory: **NO-GO**.
- Route B as new theory: **NO-GO**.
- Route C on current evidence: **NO-GO**.
- Long candidate experiment: **NO-GO**.
- Negative-result benchmark or robotics systems study: **CONDITIONAL GO**, contingent on closest-work implementations and outcome-level evidence.
