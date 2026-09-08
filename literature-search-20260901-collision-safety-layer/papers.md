# Literature Search: Collision-Safety Layer for the ICLR Return-to-Charge Study

Date: 2026-09-01

Search purpose: choose a current, defensible collision-safety layer without making collision avoidance the paper's research contribution

Target venue/family: ICLR 2027 / AI, ML, robotics

Source-quality policy: primary proceedings, publisher pages, DOI records, and arXiv for clearly labeled preprints; MDPI and snippet-only records excluded

## Summary

- Closest-work clusters: high-order CBFs; sampled-data/robust CBFs; composite LiDAR CBFs; predictive and learned safety filters.
- Best fit for the current simulator: a **sampled-data robust HOCBF with input constraints, inter-sample residual compensation, exact swept-clearance audit, and fail-safe backup**.
- Strongest recent quadrotor reference: Harms et al.'s ICRA 2025 Composite CBF, which scales to dense LiDAR constraints and has indoor/outdoor flight evidence.
- Important distinction: the most recent method is not automatically the best project choice. The present plant exposes a known translational double-integrator model, acceleration input, fixed sample-and-hold interval, static obstacles, and bounded inputs. Those assumptions align directly with sampled-data HOCBFs already implemented in the repository.
- Novelty risk: ordinary continuous-time HOCBF cannot be described as certified in the deployed sampled implementation unless inter-sample, sensing, input-feasibility, initialization, and fallback conditions are all verified.
- Recommended action: freeze the collision layer as infrastructure before any energy calibration or Oracle/Pareto study. Do not add a learned collision method unless the environment itself becomes unknown or model-mismatched.

## Paper Table

| # | Title | Year | Venue/source | Link | Type | Insight | Completeness | Numeric evidence | Overall | Why it matters here |
| --- | --- | ---: | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| 1 | High-Order Control Barrier Functions | 2022 | IEEE TAC | [DOI](https://doi.org/10.1109/TAC.2021.3105491) | theory/proof | 5 | 5 | 3 | A | Canonical relative-degree foundation for position constraints under acceleration control; insufficient by itself for sampled implementation claims. |
| 2 | Control Barrier Functions in Sampled-Data Systems | 2022 | IEEE L-CSS | [DOI](https://doi.org/10.1109/LCSYS.2021.3076127) | theory/proof | 5 | 5 | 3 | A | Directly addresses piecewise-constant control and fixed sampling, the main mismatch in the current ordinary-HOCBF execution. |
| 3 | Robust Control Barrier Functions for Sampled-Data Systems | 2024 | IEEE L-CSS / author preprint | [arXiv](https://arxiv.org/abs/2309.08050) | theory + method | 5 | 4 | 3 | A | Adds bounded disturbance and measurement-error handling for high-relative-degree sampled systems. |
| 4 | Safe Quadrotor Navigation Using Composite Control Barrier Functions | 2025 | ICRA | [DOI](https://doi.org/10.1109/ICRA55743.2025.11127368) | method + system | 5 | 5 | 5 | A | Strong recent hardware-aligned comparator: dense onboard LiDAR, many constraints, adversarial nominal commands, indoor/outdoor quadrotor tests. |
| 5 | A Predictive and Sampled-Data Barrier Method for Safe and Efficient Quadrotor Control | 2025 | arXiv preprint | [arXiv](https://arxiv.org/abs/2510.05456) | method + theory | 4 | 4 | 3 | B / current signal | Combines MPC and sampled-data HOCBF for quadrotors, but is a preprint and would add a major online-planning subsystem outside the energy-paper focus. |
| 6 | Time-Varying Soft-Maximum Barrier Functions for Safety in Unmapped and Dynamic Environments | 2025 | IEEE TCST | [DOI](https://doi.org/10.1109/TCST.2025.3587198) | method + system | 5 | 4 | 4 | A | Relevant if the project moves to unknown or dynamic environments with periodic LiDAR; unnecessary for the present known-static simulator. |
| 7 | Safety Filters for Black-Box Dynamical Systems by Learning Discriminating Hyperplanes | 2024 | L4DC / PMLR | [PMLR](https://proceedings.mlr.press/v242/lavanakul24a.html) | pure method | 5 | 4 | 4 | A | Strong modular learned-filter alternative when dynamics are genuinely black box; its training/verification burden is not justified by the current known plant. |
| 8 | A Predictive Safety Filter for Learning-Based Control of Constrained Nonlinear Dynamical Systems | 2021 | Automatica | [DOI](https://doi.org/10.1016/j.automatica.2021.109597) | method + theory | 5 | 5 | 4 | A | Canonical MPC safety-filter alternative with backup trajectories and uncertainty, but incurs online optimization and model/terminal-set obligations. |
| 9 | LiDAR-Based Online Control Barrier Function Synthesis for Safe Navigation in Unknown Environments | 2024 | IEEE RA-L | [DOI](https://doi.org/10.1109/LRA.2023.3339059) | method + system | 4 | 4 | 4 | B | Useful when obstacle geometry is unknown and a barrier must be synthesized from LiDAR online; evaluated on unicycle robots rather than this quadrotor model. |
| 10 | Safe Control Under Input Limits with Neural Control Barrier Functions | 2023 | CoRL / PMLR | [PMLR](https://proceedings.mlr.press/v205/liu23e.html) | pure method | 4 | 4 | 4 | B | Highlights that input saturation can invalidate nominal CBF safety and supplies a learned remedy; current project can enforce its known bounds directly. |
| 11 | Sampling-Based Safe Reinforcement Learning for Nonlinear Dynamical Systems | 2024 | AISTATS / PMLR | [PMLR](https://proceedings.mlr.press/v238/suttle24a.html) | method + theory | 4 | 4 | 4 | B | Samples policies directly from CBF-constrained safe action sets and includes quadcopter avoidance; useful baseline concept, but changes the policy-learning mechanism. |
| 12 | Data-Driven Safety Filters: Hamilton-Jacobi Reachability, Control Barrier Functions, and Predictive Methods for Uncertain Systems | 2023 | IEEE Control Systems Magazine | [DOI](https://doi.org/10.1109/MCS.2023.3291885) | survey | 5 | 5 | N/A | A | Authoritative map of filter families and their uncertainty/verification tradeoffs. |

## Clusters

### Sampled-data and robust HOCBF

- Representative papers: Xiao and Belta; Breeden et al.; Oruganti et al.; Gao et al.
- Already solved in the literature: relative-degree handling, fixed-sample controllers, and bounded inter-sample/state-error compensation under explicit assumptions.
- Remaining project task: instantiate those assumptions for this simulator, prove QP/fallback feasibility on the declared initial set, and audit exact swept clearance.
- Effect on the paper: collision safety becomes an inherited execution contract, not a novelty claim.

### Dense LiDAR/composite CBF

- Representative papers: Harms et al.; Safari and Hoagg; Keyumarsi et al.
- Already solved in the literature: scalable aggregation of many spatial constraints and online construction/composition of perception-derived safe sets.
- Project-specific caveat: the repository's existing `aggregate_hocbf` prototype is not evidence for Harms et al.'s Composite CBF and performed poorly in its controlled prototype benchmark. A faithful implementation would require a new verification track.
- Effect on the paper: cite Composite CBF as the strongest recent quadrotor comparator and use it only if dense raw-LiDAR scalability is a declared requirement.

### Predictive and learned filters

- Representative papers: Wabersich and Zeilinger; Lavanakul et al.; Liu et al.; Suttle et al.
- Already solved in the literature: backup-trajectory safety filtering, learned input half-spaces, learned feasible barrier functions, and safe-action policy parameterizations.
- Project-specific caveat: these methods add learning, terminal-set, model-uncertainty, or policy-optimization burdens that compete with the central energy-safety contribution.
- Effect on the paper: keep as related work or a stress baseline, not as the default collision subsystem.

## Opportunity Map

| Cluster | Status | Open gap for this project | Recommended direction | Evidence needed | Risk |
| --- | --- | --- | --- | --- | --- |
| Ordinary HOCBF | covered central claim | continuous-time certificate does not establish sampled execution safety | replace with sampled-data robust HOCBF contract | invariant-set initialization, inter-sample bound, input feasibility, zero unsafe fallback | high if unchanged |
| Sampled-data HOCBF | crowded but open | project-specific sensing/fallback contract | use as fixed infrastructure | paired frozen-policy benchmark plus formal stress suite | low-medium |
| Composite CBF | strong recent method | faithful implementation and sample-data robustness not yet integrated | strong external baseline; optional implementation only if needed | reproduce ICRA formulation, runtime, collision and fallback audit | medium-high |
| Predictive safety filter | covered central claim | terminal safe set and runtime budget | do not make central; optional baseline | matched model/information/compute contract | high cost |
| Learned/GP barrier | mechanism gap only under unknown dynamics/maps | current environment does not need learned collision certification | omit from core method | new uncertainty setting would be required | scope drift |

## Citation And Positioning Cautions

- Do not call HOCBF, sampled-data HOCBF, Composite CBF, or a QP projection novel in this paper.
- Do not equate zero observed collisions with a certificate.
- Do not call a filter the hard-safety authority if fallback actions can violate its constraints or if the frozen evaluation records collisions.
- State every guarantee condition: safe-set initialization, model fidelity, observation coverage, bounded sensing/actuation error, input feasibility, sample-and-hold period, solver success, and backup validity.
- Position collision filtering as a controlled source of executed-interface shift whose resource consequences motivate the energy-safety problem.

