# History-Conditioned Coarse-to-Fine Sampling Design

## Control-cycle contract

The feature is default-off and does not alter the legacy UAV environment. A safety cycle consumes the current state, causal history, SAC nominal action, goal, and obstacle observations. It returns the first action of a certified sequence or an explicitly labeled fallback.

Flow:

    history + current state + SAC action
      -> diverse proposal mixture
      -> cheap heuristic / certified-reject screen
      -> true clipped physics rollout
      -> continuous-time multi-step certificate
      -> hard progress admissibility
      -> rollout + terminal energy ranking
      -> execute first action

Learning proposes. Physics simulates. The certificate decides.

## Proposal families

| ID | Source | Purpose | Safety role |
|---|---|---|---|
| P1 | Correlated random perturbation around SAC | Local diversity and coverage baseline | None |
| P2 | History control extrapolation | Preserve smooth recent motion | None |
| P3 | Braking sequence | Guaranteed proposal diversity near high closing speed | None until certified |
| P4 | Goal-directed sequence | Progress mode | None until certified |
| P5 | Obstacle-normal escape | Rapid clearance mode | None until certified |
| P6 | Shift previous safe sequence | Warm start and recursive-feasibility candidate | Requires re-verification |
| P7 | CLF-guided sequence | Reduce freeze | Secondary progress only |
| P8 | CBF-gradient sequence | Aim samples toward barrier recovery | Proposal guidance only |
| P9 | Learned history-conditioned proposal | Compress useful region | Never certification |
| P10 | Mixture distribution | Prevent single-mode collapse | None |

Horizons are H in {5, 10, 20, 40}. History lengths are L in {2, 4, 8, 16}. The initial prototype uses P1–P8 and P10. P9 remains disabled until non-learning baselines establish a measurable proposal bottleneck.

## Coarse stages

Two modes are kept separate:

1. HEURISTIC_RANK ranks candidates by approximate clearance, progress proxy, and energy proxy. It has no pruning-safety guarantee.
2. CERTIFIED_REJECT rejects only when a deterministic approximation-error bound proves a collision at a coarse sample. This mode has the theorem in history_safe_trajectory_theory.md.

No arbitrary weighted score controls final safety. Final ranking is lexicographic:

1. exact certified safety;
2. progress admissibility;
3. minimum evaluated trajectory-energy upper cost;
4. tie-break by larger progress and clearance.

## Candidate counts

| Stage | Nominal count | Work |
|---|---:|---|
| A | 1,000–100,000 | Generate action sequences |
| B | all | Approximate waypoint / feature scoring |
| C | top 1–5% | True batched physics rollout |
| D | 5–100 | Exact quartic interval verification |
| E | certified subset | CLF and energy ranking |

The controlled benchmark uses at least 100,000 initial states and 1,000 coarse sequences per state. A hard subset uses at least 10,000 sequences per state. Full physics and exact verification are deliberately restricted to survivors.

## History feature provenance

| Feature | Provenance | Future leakage allowed? |
|---|---|---|
| Position, velocity, realized acceleration | directly measured | no |
| Executed safe control | directly measured | no |
| LiDAR ranges and valid mask | directly measured | no |
| Relative position/velocity/acceleration | physically derived | no |
| Closing speed, TTC, stopping/delay distance | physically derived | no |
| HOCBF slack, feasibility margin | physically derived | no |
| Motion-compensated range flow | physically derived from causal frames | no |
| Learned proposal latent | learned | no |
| Ground-truth future obstacle state | future information | forbidden from proposal input |

## Real-time measurement

Each cycle records proposal, coarse screen, physics, exact certificate, CLF, energy, and total latency. Required summaries are mean, P50, P90, P95, P99, maximum, and 50 ms deadline miss rate. CPU scalar, NumPy vectorized, Torch CPU, and Torch CUDA are compared. CUDA reporting includes host-to-device, kernel/compute, device-to-host, and end-to-end time.

## Rejection gates

The candidate route is rejected if any of the following remains after bounded optimization: unsafe neural dependence, future leakage, low safe recall against high-cost reference, material freeze/path degradation, significant energy regression, or P99 above 50 ms with no credible batching/pruning path.
