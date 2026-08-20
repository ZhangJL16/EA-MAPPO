# Reviewer B — Novelty and Closest-Work Attack

## Score

- Novelty: **2.5 / 10**
- Confidence: **0.90**
- Recommendation: **Reject as paper-core theory**

## Summary

There are literal differences from the closest work, but no substantive
positive theorem-level novelty. Fixed-time nesting across history-window length
differs from time-indexed parameter-set contraction, and the exact
bounded-jerk formula is UAV-specific. Those differences are immediate
specializations of set-membership and reachable-tube machinery. The only clean
residual statement is the causal future-jerk lower bound, which is too
elementary and negative to support the paper core.

## Fatal Concerns

1. **No recursive feasibility.** One-hold safety assumes a certified action is
   available; no multi-obstacle invariant or recoverable set is constructed.
2. **Broken contraction-to-search implication.** Radius ordering does not imply
   inclusion when predicted tube centers differ. Full tube inclusion, or
   `center shift + small radius <= large radius`, is required.
3. **No quantitative search theorem.** The Bernoulli expression contains an
   unknown safe mass and does not connect history-set diameter to candidate
   count.
4. **Model inconsistency in the old lower bound.** Instant opposite
   accelerations are incompatible with a common present acceleration under the
   declared bounded-jerk model. Only the future-jerk branch survives.
5. **Closest work is structurally stronger.** Adaptive MPSC supplies a
   terminal safe set and recursive safety; motion-primitive literature supplies
   stronger completeness and cost-resolution results.

## Exact Non-Equivalence

| Current object | Literal difference | Why it does not rescue novelty |
|---|---|---|
| `X_t^(L2) subset X_t^(L1)` | obstacle state, fixed time, varying window length | direct feasible-set restriction monotonicity |
| deterministic jerk tube | bounded physical rather than conformal uncertainty | standard reachability propagation |
| one-hold induction | pathwise deterministic rather than average probabilistic coverage | weaker than recursive safety because action existence is assumed |
| `1-(1-p)^N` | safe-candidate mass rather than optimizer box volume | elementary identity with unknown `p` |
| `j_bar tau^3 / 6` lower bound | exact third-order causal specialization | useful boundary, insufficient positive mechanism |

## Closest Primary Work

- Adaptive Model Predictive Safety Certification, arXiv:2109.13033.
- Set Membership Based Nonlinear Model Predictive Control, European Journal of
  Control 2023, DOI 10.1016/j.ejcon.2023.100857.
- Adaptive Conformal Prediction for Motion Planning among Dynamic Agents,
  L4DC 2023.
- SODA-MPC, L4DC 2025.
- Sampling-Based Optimal Kinodynamic Planning with Motion Primitives,
  arXiv:1809.02399.

## Score-Change Conditions

- Soundness-only improvement: prove full tube inclusion and remove model
  inconsistency; novelty remains incremental.
- Material novelty improvement: prove a matching minimax upper/lower result for
  the smallest causal valid tube, or construct a new computable
  multi-obstacle recoverable set guaranteeing an action every cycle.
- More UAV experiments alone do not change the novelty score.

