# Memory Theory Novelty Attack

## Strongest rejection memo

The route is an incremental composition of structured recurrent estimation,
set-valued/interval observation, sampled-data robust HOCBFs, and standard convex
feasible-set monotonicity. The recurrent nominal is not required by the proof:
deleting it and retaining the same independently declared residual bound leaves
the observer/tube/HOCBF guarantee unchanged. The method therefore fails the
requested memory-specific theorem gate.

## Closest theorem-level prior

The closest work is Hugo Matias and Daniel Silvestre, *Safe Navigation under
Uncertain Obstacle Dynamics using Control Barrier Functions and Constrained
Convex Generators*, arXiv:2601.07715 (2026). It already combines:

1. finite-horizon guaranteed set-valued obstacle estimation;
2. propagation of the obstacle set over a sampled control interval;
3. conversion of the set flow into CBF constraints;
4. QP safety filtering;
5. uncertain linear obstacle dynamics and second-order strict-feedback agents.

The proposed orthotope, constant-jerk dynamics, directional support, and UAV
specialization are narrower implementation choices, not an incommensurate
theorem conclusion.

Supporting direct overlap includes Oruganti, Naghizadeh, and Ahmed,
*Robust Control Barrier Functions for Sampled-Data Systems*, arXiv:2309.08050,
and Zhang, Walters, and Xu, *Control Barrier Function Meets Interval Analysis*,
arXiv:2110.00915. Earlier observer/perception CBF work additionally covers the
observer-error-to-barrier connection.

## Candidate theorem deletion test

Delete the GRU/contractive residual center and use the analytic nominal with the
same residual bound. The deterministic containment theorem, future tube,
robust HOCBF, safe-set inclusion, and QP-optimum monotonicity all survive. Thus
the theorem does not depend on recurrent memory. The recurrent component is an
engineering estimator, not the source of the guarantee.

## Empirical consequence attack

- Ego L16 MLP is the best held-out point estimator (`MAE 0.5303`), better than
  every recurrent candidate, including H128 Physics-GRU (`0.5892`).
- At 1 s q95, ego L16 has width `8.3024`; GRU `9.3619`, Physics-GRU `9.9987`,
  contractive memory `10.2082`.
- Learned/Kalman/IMM closed-loop boxes are uncertified on every step and show
  nonzero under-bound rates; GRU under-bounds `51.84%` of steps.
- The deterministic analytic interval has zero under-bounds in the bounded
  slice but success `0`, fallback `1`, and freeze `1`.
- Therefore no result supports the intended chain “recurrent memory -> smaller
  certified set -> less intervention -> lower energy.”

## Hostile score

| Axis | Score / 10 | Reason |
|---|---:|---|
| Correctness | 6 | Conditional ideal-arithmetic proofs pass the exact-source audit, but the final oracle trace still has 2.52% uncertified fallback and the usable learned boxes are not certificates. |
| Distinctness | 1 | Direct 2026 guaranteed-set-flow-to-CBF prior subsumes the theorem object; recurrence is proof-irrelevant. |
| Consequence | 2 | Safe-set/QP monotonicity is standard and no equal-certification intervention or energy gain is observed. |
| **Total** | **9 / 30** | Required gate is 22/30. |

## Verdict

`NOVELTY_GE_22_OF_30 = FALSE`

`DEFENSIBLE_THEORY_PAPER_CONTRIBUTION = FALSE`

`THEORETICAL_NOVELTY_SUPPORTED = FALSE`

Every pre-registered family A–M is now either an empirical baseline, falsified
by the matched-data/closed-loop evidence, or theorem-equivalent to prior work.
The target-mode stop condition is therefore the negative closure condition, not
discovery of a new theory contribution.
