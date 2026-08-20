# Recoverable Safety Literature Review

## Search Scope

Primary-source search covered 2020–2026 work on recursive-feasibility CBFs, predictive and backup CBFs, sampled-data robustness, bounded input, measurement uncertainty, dynamic obstacles, occlusion, and intermittent observations. Stable arXiv or official PMLR pages are used below.

## Cluster 1 — Predictive and Recursive Feasibility

- [Predictive Control Barrier Functions](https://arxiv.org/abs/2105.10241) constructs an always-feasible predictive auxiliary problem, uses terminal CBF structure, and proves stabilization/recovery toward the feasible set.
- [DTCBFs for Guaranteed Recursive Feasibility in Nonlinear MPC](https://arxiv.org/abs/2309.09268) uses DTCBF/qDTCBF terminal certificates to guarantee recursive feasibility with limited input.
- [Backup Control Barrier Functions](https://arxiv.org/abs/2104.11332) defines an online implicit invariant set through forward integration under a backup policy and guarantees QP feasibility.

**Effect on this project:** Route A and the general recursive-feasibility objective are covered. Route B is a horizon-one special case of predictive/discrete-time safety reasoning unless it provides a new computable bound or information structure.

## Cluster 2 — Sampled-Data Safety

- [Control Barrier Functions in Sampled-Data Systems](https://arxiv.org/abs/2103.03677) develops sufficient forward-invariance conditions under piecewise-constant control.
- [Safety of Sampled-Data Systems via Approximate Discrete-Time Models](https://arxiv.org/abs/2203.11470) connects sampled-data CBFs and practical safety through approximate discrete-time models.
- [Robust CBFs for Sampled-Data Systems](https://arxiv.org/abs/2309.08050) treats bounded disturbances and measurement errors, including high-relative-degree constraints.
- [CBF Meets Interval Analysis](https://arxiv.org/abs/2110.00915) computes reachable-overapproximation margin terms for sampled-data systems with measurement and actuation uncertainty and demonstrates them on a Crazyflie.

**Effect:** The empirical observation that sample-and-hold strengthening can be conservative is useful, but a generic robust sampled-data correction is not new.

## Cluster 3 — Measurement and Perception Robustness

- [Guaranteeing Safety of Learned Perception Modules via MR-CBFs](https://proceedings.mlr.press/v155/dean21a.html) defines measurement-robust CBF inputs under bounded perception error.
- [Measurement-Robust CBFs with Backup Sets](https://arxiv.org/abs/2104.14030) unifies measurement robustness and backup-set invariant synthesis.
- [Dynamic Output-Feedback Barrier Pairs](https://arxiv.org/abs/2308.00326) develops barrier-based safety under partial state information and uncertain MIMO dynamics.

**Effect:** Using a conservative observation set instead of a point estimate is established. A learned prediction cannot be presented as a hard certificate without independent coverage assumptions.

## Cluster 4 — Dynamic Obstacles and Occlusion

- [Dynamic CBF-MPC for LiDAR Obstacle Avoidance](https://arxiv.org/abs/2209.08539) predicts dynamic obstacle motion with a Kalman filter and inflates uncertainty ellipses.
- [Occlusion-Aware Contingency Safety-Critical Planning](https://arxiv.org/abs/2502.06359) propagates reachable sets for occluded agents and jointly optimizes exploration and fallback trajectories.

**Effect:** Constraint persistence, obstacle-set propagation, and contingency behavior during occlusion are not unoccupied ideas. A UAV implementation alone is an application difference.

## Cluster 5 — Bounded Input and Robust Backup

- [Safe Control Under Input Limits with Neural CBFs](https://proceedings.mlr.press/v205/liu23e.html) explicitly targets input saturation and safe-set synthesis for systems including a quadcopter-pendulum.
- [Disturbance-Robust Backup CBFs](https://arxiv.org/abs/2409.07700) uses expanding robust tubes around backup flows to define robust invariant sets.

**Effect:** Route C would need an analytic multi-obstacle authority result beyond generic input-constrained invariance. Current events do not exhibit authority loss at negative entry.

## Opportunity Map

| Area | Status | Remaining defensible gap |
| --- | --- | --- |
| One-step robust predecessor | Covered central claim | None without a new tractable approximation/property |
| Predictive future margin | Covered central claim | Real-time approximation with a rigorously distinct guarantee |
| Bounded uncertainty propagation | Crowded/covered | Benchmarking conservatism and tail latency under whole-object dropout |
| Intermittent-perception hard-state dataset | Benchmark gap | Reproducible matched protocol spanning row disappearance and reacquisition |
| Sampled-data residual infeasibility | Mechanism/negative-result gap | Quantify when robust strengthening removes all input authority while raw rows remain feasible |
| UAV deployment at 20 Hz | Deployment/system gap | Tail-latency-safe implementation with outcome-level collision/progress evidence |

## Literature Verdict

The exact proposed theory line is not novel enough for an ICLR/robotics theory contribution. The strongest remaining opportunity is a benchmark or systems paper that exposes the conservatism, perception-history tradeoff, and computational tails of existing filters while faithfully implementing the closest baselines.

