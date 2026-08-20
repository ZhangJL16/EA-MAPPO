# Formula-Level Novelty Matrix: Feasibility, Sampling, and Energy

## Decision rule

A difference in application, notation, or solver is not a theory contribution. A
candidate survives only if its mathematical object, guarantee, assumptions, and
computable mechanism are jointly non-equivalent to the closest prior result.

| Our equation or mechanism | Closest prior equation or mechanism | Same / different | Assumption difference | Guarantee difference | Computational difference | Energy coupling difference | Novelty decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `rho(x)=max_{u in U} min_i(A_i u-b_i)` | Exact scalar feasible-input interval and its width in [Fikri 2026](https://arxiv.org/abs/2608.10872); compatibility sets in [Tan and Dimarogonas 2022](https://arxiv.org/abs/2209.02284) | Same role; our expression is the standard multivariable max-min generalization | Cylindrical 3D acceleration set instead of scalar input / generic compact input | Exact pointwise nonemptiness only in all cases | Three-control-variable SOCP-like problem; dual simplex certificate | None | **Not novel as a feasibility concept** |
| `rho_1=sigma_U(a)-b` with `sigma_U(a)=a_xy_bar*||a_xy||+a_z_bar*|a_z|` | Support-function characterization of bounded-input linear inequalities | Mathematically identical convex analysis | Product of a disk and interval gives a UAV-specific closed form | Exact single-row feasibility | O(1) | None | **Useful closed form, not theory novelty** |
| `rho=min_{lambda in simplex}[sigma_U(A^T lambda)-b^T lambda]` | Multiple-CBF compatibility and viability-domain constructions in [Breeden and Panagou 2023](https://arxiv.org/abs/2210.01354); generic minimax/Farkas duality | Equivalent convex dual certificate | Our control dimension is fixed at three | Pointwise certificate, weaker than their viability-domain guarantee | Tiny simplex optimization and KKT recovery | None | **Not novel** |
| Add `dot rho+alpha(rho-rho_min)>=0` | Feasibility-CBF construction in [Xiao, Belta, and Cassandras 2022](https://arxiv.org/abs/2011.08248) | Same construction family | Our `rho` is nonsmooth and multi-obstacle | We cannot prove compatibility of the added constraint; their result requires a compatible feasibility condition | Adds another online constraint | None | **Rejected: circular without an invariant domain** |
| Controlled-invariant set on which all CBF rows remain compatible | Viability domain in [Breeden and Panagou 2023](https://arxiv.org/abs/2210.01354) | Same guarantee target | UAV double integrator and spherical obstacles are a special case | No stronger guarantee obtained | Could exploit 3D geometry, but no new set construction was derived | None | **Covered** |
| Backup trajectory / terminal backup-safe set | Backup-CBF safety and RL shield in [Rabiee and Safari 2025](https://proceedings.mlr.press/v283/rabiee25a.html); uniform feasible smoothed backup CBF in [Alan and De Schutter 2025](https://arxiv.org/abs/2511.13499); robust adaptive backup CBF in [Das et al. 2026](https://arxiv.org/abs/2607.20842) | Same mechanism family | Static known spheres would be easier than their settings | No new backup-invariance theorem was obtained | Requires flow rollout or verified terminal set | Energy-optimal backup was not solved in closed form | **Covered unless a new closed-form energy backup theorem is found; none was** |
| `h_brake=||r||-d_safe-v_c^2/(2 sigma_U(e))` | Braking-distance/backup-set construction, including [Braking within Barriers](https://arxiv.org/abs/2510.15797), and backup-CBF invariant subsets | Exact one-dimensional stopping calculation is the same physical construction | Cylindrical 3D acceleration support and spherical obstacle geometry | Exact only for fixed radial direction; separate obstacle tests do not certify a joint invariant set | O(1) per obstacle, but joint compatibility remains unresolved | None | **Physical diagnostic, not a new recursive-feasibility theorem** |
| Forward-invariant control-authority scalar | Readiness barrier on actuator authority in [Silano 2026](https://arxiv.org/abs/2608.16335) | Different physical object, same high-level strategy | Rotor-allocation authority versus obstacle-CBF action compatibility | Their authority set is proven invariant; ours is not | Their metric has closed-form structure; ours is an online max-min | None | **Terminology/strategy overlap; no advantage for our candidate** |
| Recursive feasibility by preserving a dynamic safety margin | CBF reference governor in [Nakano, Garone, and Notomista 2026](https://arxiv.org/abs/2604.04001) | Different controller architecture, same recursive-feasibility objective | Prestabilized reference-governor setting versus direct acceleration filter | Their design is feasible by construction; ours lacks an equivalent invariant reference dynamics | Their update is closed form | None | **Our guarantee is weaker** |
| Exact ZOH sphere clearance: minimize quartic `||r+v tau+0.5a tau^2||^2-d^2` over one hold | Sampling-aware CBF Taylor enclosure in [Liu, Xiao, and Belta 2025](https://arxiv.org/abs/2511.11897); ZOCBF in [Tan et al. 2024](https://arxiv.org/abs/2411.17079); sampled-data quadrotor HOCBF in [Gao et al. 2025](https://arxiv.org/abs/2510.05456) | Exact special-case verifier differs from generic bounds | Static spherical primitive, exact double-integrator ZOH | Exact safety verification for a **given** action; no convex safe-action synthesis theorem | Endpoints plus real roots of a cubic, O(1) per obstacle | None | **UAV-specific exact diagnostic, not a new general sampled-data theorem** |
| Two-stage progress then minimum `u^T R_E u` | Pointwise optimal feedback over a safe action map; performance-aware CBF work such as [Manda et al. 2025](https://proceedings.mlr.press/v270/manda25a.html) | Standard constrained convex optimization | Physical synthetic telemetry quadratic and one-step progress floor | Only pointwise energy optimality | LP plus convex quadratic solve | Explicit physical acceleration energy | **Optimization property is tautological, not novel** |
| `Delta P=2u_nom^T R_E d+d^T R_E d` and spectral norm upper bound | Quadratic expansion and projection-distance bounds | Mathematically identical | `d` is a safety-filter intervention | Instantaneous overhead only; no sign or mission-energy guarantee | O(1) diagnostic once the action is known | Explicit acceleration-power term | **Useful accounting identity, not novelty** |
| One-step smooth upper bound `Ehat(F(u)) <= Ehat(F(u0))+g^T delta+L/2||B delta||^2` | Descent lemma / verified neural smoothness bounds | Identical theorem under Lipschitz gradient | Current estimator contains ReLU and normalized goal-direction singularities | Conditional upper bound only; no usable verified `L` for the deployed estimator | Would be a convex QCQP if a finite useful bound existed | Learned Energy-to-Go | **Standard result and presently inapplicable** |
| Bellman supersolution `Vbar>=c+Vbar(F)` | Dynamic programming supersolution / Lyapunov telescoping certificate | Identical principle | Learned MC regressor plus conformal residual does not verify the Bellman inequality | Could yield a finite-horizon energy bound if verified; verification was not achieved | Requires global/local residual verification | Strong trajectory coupling | **Potentially strong but not constructed; no claim** |

## Closest-work conclusion

The strongest overlap is not one paper containing every implementation detail.
It is the fact that each proposed theorem-level mechanism is already a standard
member of an established family:

1. pointwise feasibility is compatibility/support-function duality;
2. recursive feasibility requires a viability, backup, terminal, or governor
   invariant construction;
3. sampled-data invariance is already addressed by sampling-aware and zero-order
   barriers;
4. pointwise energy minimization is the defining property of constrained
   optimization;
5. one-step value upper bounds are the descent lemma and require verified
   smoothness absent from the current ReLU estimator.

No formula-level difference survives the novelty standard in the task.
