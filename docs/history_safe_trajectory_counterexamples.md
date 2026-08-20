# Counterexamples for History-Conditioned Safe Trajectory Control

Each case is a falsification target. A theorem or implementation that does not survive the case is narrowed or rejected.

| ID | Counterexample | Broken claim | Required response | Status |
|---|---|---|---|---|
| C1 | A static obstacle is crossed between two safe sampled waypoints. | Waypoint collision checks imply continuous safety. | Use the quartic minimum on every physics hold. | CLOSED by exact verifier. |
| C2 | A coarse approximate point has clearance 0.2, error bound 0.3, while the true point has clearance 0.4. | Negative lower clearance bound justifies rejection. | Lower bounds certify acceptance; only a negative upper clearance bound proves rejection. | CLOSED by theorem correction. |
| C3 | A safe corridor occupies arbitrarily small action-sequence volume. | Finite random samples have deterministic safe recall. | Require robust feasible volume/density assumptions or report empirical recall only. | OPEN empirical risk. |
| C4 | Learned proposal collapses around the nominal SAC action and omits a braking/escape mode. | Learning always accelerates safe search. | Preserve deterministic diverse proposals and compare against random/CEM/MPPI. | DESIGN MITIGATION. |
| C5 | An obstacle changes velocity immediately after the final history frame. | History guarantees future obstacle motion. | Bound obstacle acceleration/change or downgrade to probabilistic perception claim. | UNRESOLVED dynamic-stage assumption. |
| C6 | LiDAR drops all rays in the obstacle-normal direction. | Temporal flow remains informative. | Mark invalid sectors and enlarge uncertainty/reject deterministic claim. | DESIGN MITIGATION. |
| C7 | Ego translation causes a static obstacle range to shrink. | Raw range flow equals obstacle motion. | Compensate the projected UAV displacement. | CLOSED by test. |
| C8 | Boundary projection zeroes velocity after propulsion. | Projected velocity jump is propulsion acceleration and energy. | Compute realized propulsion acceleration before boundary projection. | CLOSED in rollout semantics. |
| C9 | A one-horizon-safe path ends in a dead end. | Finite-horizon safety implies recursive feasibility. | Require terminal backup invariant set or state only finite-horizon safety. | CLAIM NARROWED. |
| C10 | All safe candidates increase CLF near a narrow detour. | Hard CLF decrease is always compatible with safety. | Safety dominates; permit bounded CLF slack/fallback and report freeze. | UNAVOIDABLE conflict. |
| C11 | Minimum finite-horizon energy chooses hover because terminal energy is absent. | Rollout-only energy gives task-efficient behavior. | Add frozen terminal energy-to-go and enforce progress first. | CLOSED in objective design. |
| C12 | Terminal energy model was trained on obstacle-free trajectories. | Its upper quantile remains calibrated after safety detours. | Use only as diagnostic now; retrain/recalibrate on safely executed obstacle trajectories later. | OPEN. |
| C13 | Speed clipping makes commanded acceleration nonzero but realized acceleration zero. | Command acceleration equals physical energy acceleration. | Roll out clipping and use realized acceleration. | CLOSED by test. |
| C14 | A learned trajectory lies within its historical residual envelope on training data but not after policy shift. | Empirical residual maximum is a universal tube. | Require set-membership assumptions or conditional/marginal statistical wording. | CLAIM REJECTED. |
| C15 | History length 16 increases inference latency and contains stale obstacle motion. | Longer history is monotonically better. | Ablate L=2,4,8,16 on recall, error, and P99 latency. | OPEN empirical question. |
| C16 | Candidate horizon ends before collision, while no safe continuation exists. | Short horizon plus exact verifier is enough. | Terminal backup/reachability condition or report finite-horizon-only safety. | CLAIM NARROWED. |
| C17 | Coarse weighted score ranks a high-clearance but no-progress hover above a narrow advancing path. | Weighted score encodes final control priorities. | Coarse score is computational only; final choice is safety then progress then energy. | CLOSED by selector. |
| C18 | GPU transfer dominates a small candidate batch. | GPU batching is always faster. | Report end-to-end transfer and kernel overhead by batch size. | OPEN benchmark. |

## Rejected theorem shortcuts

1. **History implies recoverability.** False without a terminal invariant set.
2. **A calibrated neural tube is deterministic.** False; calibration coverage is conditional on its statistical assumptions.
3. **Safe lower-bound pruning cannot remove feasible candidates.** False for rejection; it confuses sufficient acceptance with necessary safety.
4. **Candidate-set optimality is global optimality.** False unless the candidate set covers the complete feasible set with a quantified radius.
5. **CBF-guided samples are certified because they use CBF features.** False unless every executed interval satisfies an actual certificate.
