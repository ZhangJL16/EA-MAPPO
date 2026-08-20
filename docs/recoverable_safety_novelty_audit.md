# Recoverable Safety Novelty Audit

## Candidate Route A — Robust Predecessor

- Proposed object: `S_rho intersect Pre_W(S_rho)` and finite predecessor iterations.
- Exact difference from prior work: none at theorem level.
- Closest work: viability kernels, Backup CBF, disturbance-robust Backup CBF.
- Empirical fit: could exclude hard states, but was not implemented after the novelty stop rule triggered.
- Decision: **NO-GO** as new theory.

## Candidate Route B — Future-Feasibility Margin

- Proposed object: `max_u min_w rho(F(x,u,w))`.
- Exact difference from prior work: horizon-one scalarization of a robust predecessor.
- Closest work: Predictive CBF, DTCBF/qDTCBF recursive feasibility, robust sampled-data CBF.
- Empirical fit: 356/451 hard states admit a sampled realized-successor action with nonnegative next margin.
- Fatal limitations: the search uses realized future obstacle states; median latency 237.2 ms and P95 1.166 s; no robust guarantee; no collision outcome improvement.
- Decision: **NO-GO** as new theory or online method in its current form.

## Candidate Route C — Braking/Escape Authority

- Proposed object: available future authority minus required avoidance authority.
- Exact difference from prior work: potentially interpretable multi-obstacle bound, but no completed derivation.
- Closest work: input-limited CBF synthesis, braking invariant sets, backup CBF.
- Empirical fit: 0/451 negative-entry events lose braking authority according to the implemented radial diagnostic.
- Decision: **NO-GO** for this failure dataset. The current evidence does not justify centering the paper on braking authority.

## Intermittent-Perception Combination

Combining obstacle reachable sets with recoverability is not sufficient novelty. Measurement-robust CBF, interval-analysis sampled-data CBF, robust Backup CBF, dynamic CBF-MPC, and occlusion-aware contingency planning already cover the constituent mechanism with stronger formal or system evidence.

The remaining specific finding is that whole-object row disappearance and conservative sampled-data strengthening interact to produce large numbers of filter-infeasible fallback steps despite positive raw HOCBF margin. This is a diagnostic/benchmark contribution, not yet a new controller.

## Claim Ledger

| Claim | Status | Evidence |
| --- | --- | --- |
| Current pointwise feasibility does not imply next-step feasibility | Supported empirically | 451 matched negative entries |
| Failure is stable across random scenarios | Supported | 555/1,010 current-frame rollouts |
| Perception history reduces dropout failures | Supported in matched scenarios | 151 current-frame vs 97–104 history-mode events |
| Bounded-acceleration propagation is best | Refuted | 99 events vs 97 for constant velocity |
| Failures are dominated by braking authority loss | Refuted at event entry | 0/451 |
| Current failure creates collisions | Not supported | 0 collision rollouts |
| A one-step alternative often exists | Supported as oracle diagnostic | 356/451 |
| Sampled action search is 20 Hz deployable | Refuted | P95 1.166 s |
| Proposed robust predecessor is novel | Refuted by prior art | predictive/backup/DTCBF literature |
| A new recoverability controller improves outcomes | Not tested | candidate stopped before method stage |

## Final Novelty Position

The general control-theory candidate is covered. The defensible research opportunity is limited to:

1. a reproducible intermittent-perception feasibility-loss benchmark;
2. a mechanism study separating raw HOCBF feasibility from sampled-data strengthened feasibility;
3. a systems comparison of closest-work methods under 20 Hz tail latency, progress and energy constraints.

That opportunity becomes a method contribution only if a future implementation has a precise property that Predictive CBF, Backup CBF, robust sampled-data CBF, and occlusion-aware planning do not already provide.
