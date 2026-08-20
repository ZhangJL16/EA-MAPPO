# UAV Safety + Energy Joint Action Filter: Novelty Audit

## Verdict

**NOVELTY STATUS: NOT SUPPORTED.**

The current package is a technically coherent deployment architecture, but the literature search does not support a new generic CBF, aggregate-CBF, or battery-viability claim. The remaining learned energy-to-go gradient hypothesis has now been falsified as the primary mechanism: a fixed-scale held-out ablation did not establish incremental trajectory-energy savings, and unit-direction normalization increased energy. A 1,000-state finite-difference audit supports local derivative correctness for the point predictor, but local correctness is not a contribution and did not create a reliable systems benefit.

Search basis: 49 primary papers in `uav_safety_energy_literature_matrix.csv`, including 41 papers from 2020--2026 and direct closest work in HOCBFs, sampled-data CBFs, point-cloud safety, CMDPs, and battery-persistence CBFs.

## Candidate Cards

| Candidate | Mechanism | Strongest closest work | Remaining delta | Current decision |
| --- | --- | --- | --- | --- |
| Standard second-order HOCBF-QP | Project frozen-SAC acceleration into bounded collision-safe set | Nguyen and Sreenath 2016; Xiao and Belta 2022 | Application-specific 3D implementation only | Existing method and required baseline |
| Dense multi-obstacle aggregate HOCBF | Replace many obstacle rows by smooth log-sum-exp row | Wang et al. L4DC 2025; Glotfelter et al. 2017 | A sampled-data perception-specific error bound could differ, but is absent | Generic novelty rejected |
| Candidate A: instantaneous energy-aware HOCBF-QP | Add normalized quadratic acceleration-energy term to minimal-deviation objective | Standard CBF-QP plus energy-aware planning and energy-tank CBF work | Exact UAV cost integration and Pareto evidence | Sound ablation; novelty weak |
| Candidate B: energy-to-go gradient projection | Add local action sensitivity of learned long-horizon energy to the safe objective | Safe action projection; energy-aware planning; learned value-gradient control families | Frozen-policy MC energy gradient inside hard sampled-data 3D HOCBF was not found identically | Rejected: no reliable incremental energy benefit |
| Candidate C: battery-returnability viability barrier | Constrain battery minus reserve minus return energy as a CBF | Notomista et al. 2018/2021; Fouad et al. 2023; Dan et al. 2021 | Learned calibrated return bound and this exact simulator | Direct conceptual overlap; novelty rejected |
| Full system integration | Frozen SAC plus LiDAR HOCBF plus calibrated energy switching | Point-cloud CBF; layered safety filters; energy persistification | Joint systems benchmark with latency and energy accounting | Potential robotics/systems contribution only if measurements expose a new bottleneck |

## Strongest Prior-Art Attacks

### Attack 1: “The safety method is standard HOCBF-QP.”

- Exact claim attacked: second-order spherical-obstacle inequality followed by minimum-deviation action projection.
- Evidence: ECBF/HOCBF theory already handles higher relative degree; CBF-QP already supplies minimal intervention under input bounds.
- Result: fatal to a method-novelty claim, not to correctness or utility.
- Score-change condition: a distinct sampled-data theorem or active-constraint mechanism must prove a property unavailable from direct HOCBF application.

### Attack 2: “The aggregate method is Wang et al. 2025 in a UAV wrapper.”

- Exact claim attacked: log-sum-exp reduction of many LiDAR obstacle barriers followed by closed-form correction.
- Evidence: Wang et al. directly studies multi-constraint safe RL with log-sum-exp CBF approximation and a closed-form action correction.
- Result: generic aggregate novelty is closed.
- Score-change condition: derive and validate an error bound coupling primitive extraction, LiDAR staleness, and aggregate approximation while preserving sampled-data safety.

### Attack 3: “Quadratic propulsion cost is an obvious secondary objective.”

- Exact claim attacked: Candidate A.
- Evidence: adding a positive-semidefinite quadratic cost to a strictly convex CBF-QP preserves convexity but does not create a new safety principle.
- Result: Candidate A is useful engineering and an ablation, not a central contribution.
- Score-change condition: show a non-obvious physically normalized choice that consistently lowers total trajectory energy without degrading hard safety, path ratio, or feasibility across dense scenes.

### Attack 4: “Battery-returnability CBF already exists.”

- Exact claim attacked: Candidate C and names such as “Energy-Viable HOCBF.”
- Evidence: persistification and energy-sufficiency papers already construct forward-invariant resource/charging conditions, including drone collision-plus-charging certificates.
- Result: fatal to generic novelty. Learned return-energy calibration changes the estimator, not the fundamental CBF object.
- Score-change condition: none for the generic formulation. Keep only if a controlled ablation shows operational value beyond the existing high-level one-way switch.

### Attack 5: “The full package is module recombination.”

- Exact claim attacked: frozen SAC + HOCBF + propulsion term + learned energy value + switching.
- Evidence: every component family is established separately, and current results do not show an emergent failure or capability that requires their exact coupling.
- Result: current conference-method readiness is low.
- Score-change condition: identify a reproducible failure of minimum-deviation HOCBF under the UAV energy model and show Candidate B fixes it with a verified causal mechanism.

## Independent Reviewer Panel

### Field Expert

- Score tendency: 3/5.
- Positive signal: dense 3D LiDAR at 20 Hz with frozen-policy intervention is an important deployment setting.
- Rejection-grade concern: the problem setting is specific, but the current formula family is already populated.
- Change condition: demonstrate an energy/safety/latency phenomenon not explained by standard HOCBF and aggregate-CBF work.
- Confidence: high.

### Method Expert

- Score tendency: 3/5 for soundness, 2/5 for conceptual novelty.
- Positive signal: the collision derivation and convexity checks are coherent under the translational double-integrator abstraction.
- Rejection-grade concern: Candidate B depends on a differentiable and locally meaningful energy estimator. The retained MC point model passed a local finite-difference audit (action-gradient median relative error 0.299%, P95 1.94%), but P99 was 16.07% and the conformal correction remains nondifferentiable and excluded.
- Change condition: pass 1,000-state finite-difference checks and show lower total trajectory energy rather than only lower instantaneous acceleration cost.
- Confidence: high.

### Experiment Expert

- Score tendency: 3/5 for systems evidence, 2/5 for method novelty.
- Positive signal: 1,000 matched method rollouts expose clear safety differences, paired held-out ablations quantify the energy trade-off, and 1,024-row timing was measured with 30 repeats.
- Rejection-grade concern: Candidate B failed its incremental-effect gate; sensor-dropout stress also shows the guarantee depends on obstacle observability. CMDP baselines were selected but not trained, so no learned-baseline superiority claim is available.
- Change condition: a future paper should either add high-quality matched CMDP training or explicitly scope itself as a robotics systems evaluation of sampled-data safety filtering.
- Confidence: high.

### AC / Venue Expert

- Score tendency: 2/5 for a general ML/control-method paper; 3/5 potential for robotics/systems after strong results.
- Positive signal: an end-to-end measured safety layer over a fixed high-performing policy can be valuable to robotics deployment readers.
- Rejection-grade concern: without a theorem-level delta or a newly exposed systems bottleneck, reviewers will classify this as integration.
- Change condition: either produce a distinct sampled-data/perception theorem or reposition around a reproducible real-time safety-energy systems finding.
- Confidence: medium-high.

### Skeptical Prior-Art Expert

- Score tendency: 1/5 for Candidate C, 2/5 for Candidate A, 3/5 for Candidate B.
- Rejection-grade concern: new terminology would hide established persistification, energy-sufficiency, and aggregate-CBF mechanisms.
- Change condition: restrict claims to measured behavior and stop using generic novelty language unless Candidate B survives both gradient and closest-work checks.
- Confidence: high.

## Scorecard for the Combined Candidate

| Dimension | Weight | Score | Confidence | Deduction | Repair condition |
| --- | ---: | ---: | ---: | --- | --- |
| Problem importance | 12 | 4 | 4 | Real-time safe deployment is important, but the exact bottleneck is not yet measured | Show a dense-LiDAR energy/latency failure |
| Novelty | 14 | 2 | 4 | HOCBF, aggregation, and energy viability have direct prior art | Candidate B must exhibit a theorem-level or causal empirical delta |
| Conceptual innovation | 12 | 2 | 4 | Present method is mostly module composition | Isolate one indispensable mechanism |
| Method soundness | 14 | 3 | 4 | Collision part is sound; learned energy gradient and sampled-data guarantee remain incomplete | Validate gradients and adopt an exact sampled-data bound |
| Elegance | 8 | 3 | 4 | Layered architecture is clear but Candidate C duplicates switching | Remove redundant protection unless ablation proves value |
| Feasibility | 8 | 4 | 4 | Three-dimensional action and existing policy make controlled experiments tractable | Demonstrate reliable solver/fallback integration |
| Experimental convincibility | 10 | 3 | 4 | Controlled and paired evidence is strong for the prototype, but no multi-seed CMDP comparison exists | Add matched trained baselines and broader perception tests |
| Venue fit | 8 | 3 | 3 | Better robotics/systems fit than general RL theory at present | Ground contribution in system measurements |
| Timeliness | 6 | 4 | 4 | Perception-aware and sampled-data safe control are active | Compare directly with recent aggregate/perception-aware methods |
| Risk-adjusted acceptance potential | 8 | 2 | 3 | Novelty collapse is the strongest unresolved risk | Produce decisive Candidate B evidence or pivot to existing-method study |

Weighted score after controlled experiments: **2.88/5**. This is a diagnostic score, not an acceptance probability.

Current conference readiness: **low**.

Development potential: **medium** as a rigorous robotics/autonomous-systems study; low as a generic new-CBF theory paper without further evidence.

## Claim Gate

Permitted now:

1. the second-order HOCBF derivation is valid under the stated model;
2. the prototype implements several established action-filter variants;
3. Candidate A is a convex energy-aware secondary objective that reduced held-out trajectory energy by 3.56% in this benchmark at matched zero observed collisions;
4. Candidate B did not establish incremental energy savings and Candidate C was not deployed;
5. Candidate C has high prior-art overlap;
6. sampled-data strengthening eliminated observed collisions in the exact-geometry controlled benchmark, subject to the stated assumptions.

Not permitted now:

1. “first” or “novel” joint filter;
2. deterministic safety under noisy or stale LiDAR;
3. energy optimality from an instantaneous quadratic penalty;
4. battery-returnability novelty;
5. superiority over CMDP methods without matched training and evaluation;
6. robustness to whole-object LiDAR dropout.

## Decision

The controlled falsification is complete. Candidate B did not reliably reduce complete-trajectory energy beyond Candidate A, so it is rejected. Select the best existing sampled-data HOCBF variant, retain Candidate A as a systems-level energy-efficiency improvement, and report **NO SUFFICIENT NOVELTY YET**. The current work is suitable as a rigorous robotics/autonomous-systems prototype and benchmark study, not as a new generic CBF or safe-RL method claim.
