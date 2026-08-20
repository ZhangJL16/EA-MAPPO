# UAV Safety + Energy Joint Action Filter: Derivation Package

## Target

Derive a deployment-time action filter for the existing frozen goal-conditioned SAC policy. The filter receives the nominal acceleration command, current translational state, obstacle information, and optional energy context, and returns an executed acceleration command. Collision constraints are hard constraints within the stated model. Energy terms are either secondary optimization objectives or conditional constraints whose validity depends on the energy model.

## Status

**COHERENT AFTER REFRAMING / EXTRA ASSUMPTION.** The second-order collision HOCBF and its safe-action projection are coherent. The instantaneous propulsion objective is a coherent convex objective. A battery-returnability barrier is mathematically coherent only when the return-energy function is differentiable and conservative on the deployed state distribution; the current Mondrian correction is piecewise and statistically calibrated, so it cannot support a deterministic battery-invariance theorem. Direct prior work also makes a generic energy-sufficiency CBF non-novel.

## Symbol Table

| Symbol | Dimension | Unit | Meaning |
|---|---:|---:|---|
| \(p\) | 3 | m | UAV position |
| \(v\) | 3 | m/s | UAV translational velocity |
| \(u\) | 3 | m/s² | realized propulsion acceleration |
| \(u_{\rm nom}\) | 3 | m/s² | frozen-SAC nominal acceleration |
| \(c_i\) | 3 | m | obstacle-center or local sphere-center estimate |
| \(v_{o,i}\) | 3 | m/s | obstacle velocity |
| \(a_{o,i}\) | 3 | m/s² | obstacle acceleration |
| \(r_i=p-c_i\) | 3 | m | relative position |
| \(v_{r,i}=v-v_{o,i}\) | 3 | m/s | relative velocity |
| \(d_i\) | 1 | m | effective center-to-center safety distance |
| \(h_i\) | 1 | m² | squared-distance barrier |
| \(\dot h_i\) | 1 | m²/s | first barrier derivative |
| \(\ddot h_i\) | 1 | m²/s² | second barrier derivative |
| \(k_1,k_2\) | 1 | 1/s | linear HOCBF gains |
| \(B\) | 1 | synthetic energy unit | remaining battery |
| \(P(v,u)\) | 1 | synthetic energy unit/s | telemetry power model |
| \(E_R(x)\) | 1 | synthetic energy unit | return-to-charger energy estimate |
| \(R_E\) | 3×3 | energy·s³/m² | acceleration-squared power matrix |
| \(\Delta_s\) | 1 | s | safety-filter sample interval, nominally 0.05 s |
| \(\tau\) | 1 | s | bounded sensing/processing/hold delay |

## Invariant Objects

1. **Collision object:** the intersection of obstacle-wise sets \(\mathcal C_i=\{(p,v):h_i(p)\ge 0,\;\psi_{1,i}(p,v)\ge0\}\).
2. **Energy object:** the battery-returnability margin \(h_E=B-B_{\rm res}-E_R(x)\), treated separately from collision invariance.

The two objects must not be merged into a weighted scalar reward. Collision remains a hard constraint; energy affects a secondary objective or a separately audited constraint.

## Assumptions

1. The controlled plant for the proof is the translational double integrator
   \[
   \dot p=v,\qquad \dot v=u.
   \]
   This is not a full quadrotor attitude model.
2. The realized propulsion acceleration satisfies
   \[
   \lVert u_{xy}\rVert_2\le a_{xy}^{\max},\qquad |u_z|\le a_z^{\max}.
   \]
3. Static-obstacle proofs use exact obstacle geometry. Moving-obstacle proofs require measured \(v_o\) and either measured \(a_o\) or a declared acceleration bound.
4. Continuous-time forward invariance requires locally Lipschitz closed-loop inputs and satisfaction of the HOCBF condition throughout time. A sampled implementation needs an additional sampled-data condition; checking only at sample times is insufficient.
5. Energy coefficients are the synthetic `TelemetryCostModel` coefficients. No Joule or Wh interpretation is claimed.
6. Any theorem involving learned \(E_R\) is conditional on differentiability, approximation error bounds, and the stated calibration/shift assumptions.

## Derivation Map

1. Double-integrator dynamics and squared-distance geometry give a relative-degree-two barrier.
2. Linear class-\(\mathcal K\) functions give an affine inequality in physical acceleration.
3. Multiple obstacle inequalities plus a conservative polyhedral actuator approximation give a small convex QP; the exact horizontal acceleration ball gives an SOCP-representable problem.
4. A soft-min aggregate barrier gives one affine HOCBF constraint but shrinks the certified set and can degenerate through gradient cancellation.
5. The acceleration-squared term in `TelemetryCostModel` gives a positive-semidefinite quadratic energy objective.
6. A differentiable return-energy approximation gives a first-order energy-to-go action term.
7. A battery-returnability barrier gives a convex quadratic inequality, but its deterministic interpretation fails for the current piecewise statistical upper bound.

## 1. Second-Order Collision HOCBF

### 1.1 Static spherical obstacle

For obstacle center \(c_i\) and effective safety distance \(d_i\), define

\[
r_i=p-c_i,
\qquad
h_i(p)=r_i^\top r_i-d_i^2.
\]

The safe geometric set is \(h_i\ge0\). Since \(c_i\) is static,

\[
\dot r_i=v,
\qquad
\dot h_i=2r_i^\top v.
\]

The input does not appear in \(\dot h_i\), so the relative degree with respect to acceleration input \(u\) is two. Differentiating again,

\[
\ddot h_i
=2\dot r_i^\top v+2r_i^\top\dot v
=2v^\top v+2r_i^\top u.
\]

Choose linear extended class-\(\mathcal K\) functions

\[
\alpha_1(h)=k_1h,
\qquad
\alpha_2(\psi_1)=k_2\psi_1,
\qquad k_1,k_2>0.
\]

Then

\[
\psi_{0,i}=h_i,
\qquad
\psi_{1,i}=\dot h_i+k_1h_i,
\]

and

\[
\begin{aligned}
\psi_{2,i}
&=\dot\psi_{1,i}+k_2\psi_{1,i}\\
&=\ddot h_i+k_1\dot h_i+k_2(\dot h_i+k_1h_i)\\
&=\ddot h_i+(k_1+k_2)\dot h_i+k_1k_2h_i.
\end{aligned}
\]

Substitution yields

\[
2r_i^\top u
+2\lVert v\rVert_2^2
+2(k_1+k_2)r_i^\top v
+k_1k_2(\lVert r_i\rVert_2^2-d_i^2)
\ge0.
\]

Equivalently,

\[
a_i^\top u\ge b_i,
\]

with

\[
a_i=2r_i,
\]

\[
b_i=-2\lVert v\rVert_2^2
-2(k_1+k_2)r_i^\top v
-k_1k_2(\lVert r_i\rVert_2^2-d_i^2).
\]

### 1.2 Unit check

- \(h_i\): m².
- \(\dot h_i\): m²/s.
- \(k_1h_i\): (1/s)m² = m²/s.
- \(\ddot h_i\): m²/s².
- \(k_2\psi_{1,i}\): (1/s)(m²/s) = m²/s².
- \(r_i^\top u\): m·m/s² = m²/s².

Every term in \(\psi_2\) therefore has unit m²/s².

**Proposition 1 (continuous-time collision invariance).** Assume the double-integrator model is exact, obstacle position/velocity/acceleration are known, the initial state satisfies \(h_i\ge0\) and \(\psi_{1,i}\ge0\) for every obstacle, and a locally Lipschitz executed input satisfies every HOCBF inequality continuously. Then the intersection of the nested obstacle safe sets is forward invariant.

This is the standard relative-degree-two HOCBF result. It does not include sample-and-hold, sensor error, dropped constraints, or post-solve actuator projection.

### 1.3 Moving obstacle

Let

\[
r_i=p-p_{o,i},\qquad
v_{r,i}=v-v_{o,i},\qquad
a_{r,i}=u-a_{o,i}.
\]

Then

\[
\dot h_i=2r_i^\top v_{r,i},
\]

\[
\ddot h_i=2v_{r,i}^\top v_{r,i}+2r_i^\top(u-a_{o,i}).
\]

The affine constraint becomes

\[
2r_i^\top u\ge
2r_i^\top a_{o,i}
-2\lVert v_{r,i}\rVert_2^2
-2(k_1+k_2)r_i^\top v_{r,i}
-k_1k_2(\lVert r_i\rVert_2^2-d_i^2).
\]

If obstacle acceleration is unknown but \(\lVert a_{o,i}\rVert_2\le\bar a_{o,i}\), then

\[
r_i^\top a_{o,i}\le\lVert r_i\rVert_2\bar a_{o,i}.
\]

Replacing \(2r_i^\top a_{o,i}\) by \(2\lVert r_i\rVert_2\bar a_{o,i}\) gives a worst-case robust sufficient condition. It is conservative and requires a credible acceleration bound.

## 2. Safe Action Set and Projection

For active obstacle rows, stack \(a_i^\top\) into \(A(x)\) and \(b_i\) into \(b(x)\). The exact physical input set is

\[
\mathcal U=
\left\{u:\lVert u_{xy}\rVert_2\le a_{xy}^{\max},\ |u_z|\le a_z^{\max}\right\}.
\]

The state-dependent safe set is

\[
\mathcal U_{\rm safe}(x)=\{u\in\mathcal U:A(x)u\ge b(x)\}.
\]

The baseline projection is

\[
u^*=\arg\min_{u\in\mathcal U_{\rm safe}(x)}
\frac12(u-u_{\rm nom})^\top W(u-u_{\rm nom}),
\qquad W\succ0.
\]

The exact horizontal norm constraint makes this a convex QCQP/SOCP-representable problem, not a pure QP. A conservative inscribed polygon for the horizontal disk converts it to a QP while guaranteeing the true norm limit.

The implementation also prevents the environment's velocity saturation from changing the solved acceleration. For one safety interval, it enforces the conservative polygonal constraints

\[
n_j^\top(v_{xy}+u_{xy}\Delta_s)\le
v_{xy}^{\max}\cos(\pi/N_f),
\]

together with

\[
-v_z^{\max}\le v_z+u_z\Delta_s\le v_z^{\max}.
\]

The polygon is inscribed in the true horizontal velocity disk. Therefore any accepted next velocity satisfies the physical norm limit, though feasible velocities near the circular boundary can be rejected. Without this state-dependent constraint, environment-side speed clipping would change the realized acceleration and invalidate a guarantee about the QP command.

### 2.1 Single halfspace closed form

Ignoring additional active actuator constraints, for \(a^\top u\ge b\),

\[
u^*=u_{\rm nom}
\quad\text{if }a^\top u_{\rm nom}\ge b.
\]

Otherwise the KKT conditions give

\[
u^*=u_{\rm nom}
+\frac{b-a^\top u_{\rm nom}}{a^\top W^{-1}a}W^{-1}a.
\]

This is the weighted orthogonal projection onto the boundary hyperplane. Once an actuator boundary is also active, this expression is generally no longer the solution.

### 2.2 One-step reachable supporting-halfspace baseline

Under constant acceleration for \(\Delta_s\),

\[
r^+=r+v\Delta_s+\tfrac12u\Delta_s^2.
\]

The exact requirement \(\lVert r^+\rVert\ge d\) is nonconvex in \(u\). Let \(n\) be the unit vector of the nominal predicted relative position. The sufficient supporting-halfspace condition

\[
n^\top r^+\ge d
\]

gives

\[
n^\top u\ge
\frac{2}{\Delta_s^2}
\left[d-n^\top(r+v\Delta_s)\right].
\]

This certifies only the chosen one-step supporting halfspace. It does not imply continuous-time forward invariance and is therefore an experimental baseline, not a replacement theorem for HOCBF.

## 3. Multiple Obstacles

### 3.1 Raw constraints and top-K selection

For nominal action \(u_{\rm nom}\), define nominal HOCBF slack

\[
s_i=a_i^\top u_{\rm nom}-b_i.
\]

Negative slack means the nominal action violates the barrier. Sorting by increasing \(s_i\) is directly induced by the HOCBF inequality and incorporates distance, relative radial velocity, and braking authority through \(b_i\). It is preferable to distance-only ranking. Dropping constraints is not safety preserving unless omitted obstacles are independently proved inactive over the hold interval.

### 3.2 Smooth aggregate barrier

Define the conservative soft minimum

\[
h_{\rho}=-\frac1\rho\log\sum_{i=1}^m e^{-\rho h_i},\qquad \rho>0.
\]

Because \(h_{\rho}\le\min_i h_i\), \(h_{\rho}\ge0\) implies every \(h_i\ge0\). Let

\[
w_i=\frac{e^{-\rho h_i}}{\sum_j e^{-\rho h_j}}.
\]

Then

\[
\dot h_{\rho}=\sum_iw_i\dot h_i,
\]

and

\[
\ddot h_{\rho}
=\sum_iw_i\ddot h_i
-\rho\left[
\sum_iw_i\dot h_i^2-\left(\sum_iw_i\dot h_i\right)^2
\right].
\]

The variance term is nonpositive. Since each \(\ddot h_i\) is affine in \(u\), the aggregate HOCBF condition remains affine in \(u\):

\[
2\left(\sum_iw_ir_i\right)^\top u+\eta_{\rho}(x)\ge0.
\]

This reduces the obstacle constraints to one row. It is conservative, and symmetric obstacles can make \(\sum_iw_ir_i\) nearly zero, destroying control authority in the aggregate row. Aggregate feasibility and minimum individual clearance must therefore be measured, not assumed.

## 4. Sampled Data, LiDAR Age, and Solver Delay

The continuous-time HOCBF theorem does not automatically hold under 20 Hz zero-order hold. Define

\[
\tau_{\rm total}=\tau_{\rm sensor}+\tau_{\rm processing}+\tau_{\rm solve}+\tau_{\rm actuation}.
\]

Two distinct effects must not be double counted:

1. **Geometry uncertainty:** a stale or noisy obstacle estimate changes the unknown true center. If the center error is bounded by \(\epsilon_c\), use a geometric inflation \(d_i=d_{\rm geometry}+\epsilon_c\).
2. **Inter-sample evolution:** even with exact geometry, the HOCBF residual can decrease between sample instants. A sampled-data theorem must bound this change or enforce a discrete/zero-order barrier condition.

A provisional center-error bound of the form

\[
\epsilon_c\le\epsilon_p+\epsilon_v\tau+\tfrac12\bar a_o\tau^2
\]

is a geometry-estimation bound, not by itself a sampled-data invariance proof. For a continuous residual \(F(x,u)=\psi_2(x,u)\), if a valid local bound

\[
|F(x(t),u_k)-F(x_k,u_k)|\le\nu_k(\tau)
\]

holds over the hold interval, then enforcing

\[
F(x_k,u_k)\ge\nu_k(\Delta_s)
\]

is a sufficient robust condition for \(F\ge0\) during that interval. Computing a non-vacuous \(\nu_k\) requires state, velocity, input, obstacle-motion, and model bounds. Until that bound is implemented and tested, the prototype reports sampled-data robustness as unverified.

For the static or constant-acceleration obstacle model over one hold, write

\[
F(r,v_r,u)=2\lVert v_r\rVert^2-2r^\top a_o
+2r^\top u+2(k_1+k_2)r^\top v_r+k_1k_2(\lVert r\rVert^2-d^2).
\]

Let \(\bar A_u=\sqrt{(a_{xy}^{\max})^2+(a_z^{\max})^2}\),
\(\bar A_r=\bar A_u+\lVert a_o\rVert\), and \(T=\Delta_s\). Then

\[
\Delta_v=\bar A_rT,
\qquad
\Delta_r=\lVert v_r\rVert T+\tfrac12\bar A_rT^2,
\]

bound the relative velocity and position changes. Along the hold interval use

\[
\bar V_r=\lVert v_r\rVert+\Delta_v,
\qquad
\bar R=\lVert r\rVert+\Delta_r.
\]

The gradients of \(F\) satisfy

\[
\left\lVert\frac{\partial F}{\partial r}\right\rVert
\le
2\lVert a_o\rVert+2(k_1+k_2)\bar V_r+2k_1k_2\bar R+2\bar A_u=:L_r,
\]

\[
\left\lVert\frac{\partial F}{\partial v_r}\right\rVert
\le4\bar V_r+2(k_1+k_2)\bar R=:L_v.
\]

Therefore a computable local residual bound is

\[
\nu_k(T)=L_r\Delta_r+L_v\Delta_v.
\]

The robust sampled constraint is

\[
a_i^\top u\ge b_i+\nu_k(T).
\]

**Proposition 5 (conditional sampled-hold safety).** Suppose the obstacle acceleration is known and constant over each hold, the UAV input is constant and bounded by the declared actuator set, the model is the exact translational double integrator, the initial nested HOCBF conditions hold, and the strengthened constraint above is feasible and enforced at every sample. Then \(F(t)\ge0\) throughout every hold interval, so the continuous-time HOCBF condition is maintained and the collision safe set is forward invariant.

The proof is the mean-value inequality applied to \(F\), followed by the continuous-time HOCBF theorem. This bound is deliberately conservative. It does not cover unbounded LiDAR error, variable obstacle acceleration without a jerk/error bound, dropped constraints, or the aggregate soft-min barrier. The prototype implements it behind `sampled_data_robust`; aggregate use is rejected rather than assigned an unsupported bound.

## 5. Instantaneous Energy-Aware Projection

The telemetry model has power

\[
P(v,u)=P_{\rm fixed}+c_v^\top|v|+u^\top R_Eu,
\]

where \(R_E=\operatorname{diag}(c_a)\succeq0\). Over one safety interval, the action-dependent energy is approximately \(\Delta_su^\top R_Eu\).

Consider

\[
J_A(u)=
\frac12(u-u_{\rm nom})^\top W(u-u_{\rm nom})
+\lambda_E\Delta_su^\top R_Eu.
\]

Its Hessian is

\[
H_A=W+2\lambda_E\Delta_sR_E.
\]

If \(W\succ0\), \(R_E\succeq0\), and \(\lambda_E\ge0\), then \(H_A\succ0\), so the objective is strictly convex. This does not prove lower trajectory energy: reducing instantaneous acceleration can increase flight time or detour length.

**Proposition 3 (convexity and uniqueness).** Under the preceding matrix assumptions, Candidate A has a unique minimizer over any nonempty convex intersection of affine HOCBF rows, polygonal acceleration limits, and polygonal one-step velocity limits. With the exact horizontal norm limits it remains a strictly convex SOCP-representable problem.

For interpretable scaling, use dimensionless terms

\[
\bar J_A=
\frac12\frac{\lVert u-u_{\rm nom}\rVert_W^2}{A_{\rm char}^2}
+\lambda_E
\frac{\Delta_su^\top R_Eu}{E_{\rm char}},
\]

where \(A_{\rm char}\) and \(E_{\rm char}\) are fixed physical characteristic scales recorded in configuration. The remaining \(\lambda_E\) is a Pareto parameter, not a guarantee parameter.

## 6. Energy-to-Go Gradient Objective

For state \(x=(p,v)\), the constant-acceleration discrete approximation is

\[
p^+=p+v\Delta_s+\tfrac12u\Delta_s^2,
\qquad
v^+=v+u\Delta_s.
\]

For differentiable \(\widehat E(x,g)\), Taylor expansion around a reference action gives

\[
\widehat E(x^+,g)\approx
\widehat E(\bar x^+,g)
+g_E^\top(u-\bar u),
\]

with

\[
g_E=
\tfrac12\Delta_s^2\nabla_p\widehat E
+\Delta_s\nabla_v\widehat E.
\]

Here \(\nabla_p\widehat E\) has units energy/m and
\(\nabla_v\widehat E\) has units energy·s/m. Hence \(g_E\) has units
energy·s²/m and \(g_E^\top u\) has units energy. It can only be added to the
dimensionless intervention objective after division by a declared energy
scale or through a weight carrying the reciprocal units.

The candidate objective is

\[
J_B(u)=J_A(u)+\lambda_Gg_E^\top u.
\]

The linear term preserves convexity. The current energy estimator consumes normalized goal-relative features, so gradients must include the feature Jacobian. Clipping, zero-distance direction normalization, and piecewise Mondrian corrections create nondifferentiable locations. The point-model gradient must pass finite-difference checks before use; the conformal correction must not be silently differentiated.

### 6.3 Numerical gradient audit

The implemented gradient differentiates only the retained MC point predictor. It
does **not** differentiate the piecewise conformal/Mondrian correction. On 1,000
random physical states, a local central-difference audit using 0.1 m position and
0.01 m/s velocity perturbations produced median relative error 0.169%, P95
5.87%, and P99 22.13%. The larger tail is consistent with ReLU activation and
feature-clipping boundaries. The action-gradient audit at the 50 ms safety
interval, with a 0.1 m/s2 finite-difference action perturbation, produced median
relative error 0.299%, P95 1.94%, P99 16.07%, and 90.8% of states below 1%.

These numbers support Candidate B as a locally checked **soft objective
ablation**. They do not support a smooth global energy theorem, a calibrated
energy guarantee, or differentiation through the deployed upper confidence
correction. The exact audit artifacts are
`artifacts/uav_safety_energy_gradient_audit_small_20260819_020314.json` and
`artifacts/uav_safety_energy_action_gradient_audit_20260819_020407.json`.

## 7. Battery-Returnability Viability Barrier

Define

\[
h_E(x,B)=B-B_{\rm res}-E_R(x).
\]

With

\[
\dot B=-P(v,u),
\]

the derivative is

\[
\dot h_E
=-P(v,u)-\nabla_pE_R^\top v-\nabla_vE_R^\top u.
\]

Imposing \(\dot h_E+k_Eh_E\ge0\) gives

\[
u^\top R_Eu+\nabla_vE_R^\top u
\le
k_Eh_E-P_0(v)-\nabla_pE_R^\top v.
\]

The left side is convex when \(R_E\succeq0\), so its sublevel set is convex. Combined with affine collision constraints, the problem is a convex QCQP and may be SOCP-representable after completing the square when the quadratic admits a suitable factorization.

Every term in this inequality has power units: \(u^\top R_Eu\),
\(\nabla_vE_R^\top u\), \(P_0\), \(\nabla_pE_R^\top v\), and
\(k_Eh_E\) are all synthetic energy units per simulation second.

**Proposition 4 (conditional battery-returnability invariance).** If \(E_R\) is continuously differentiable and is a deterministic conservative return-energy function on the deployed domain, battery dynamics equal the declared power model, the initial state satisfies \(h_E\ge0\), and the energy inequality is feasible and continuously enforced, then \(\{h_E\ge0\}\) is forward invariant. The current learned Mondrian bound does not satisfy these deterministic premises, so this proposition is not a claim about the deployed estimator.

### Critical limitations

1. `Persistification of Robotic Tasks Using Control Barrier Functions` (RA-L 2018) already encodes battery sufficiency as a forward-invariant CBF set while minimizing deviation from a nominal task controller.
2. `Energy Sufficiency in Unknown Environments via Control Barrier Functions` (2023) explicitly places an energy-sufficiency CBF layer over a generic planner.
3. The current \(E_R^{\rm upper}\) is learned and Mondrian-calibrated. Its coverage is statistical and matched-distribution dependent, not a deterministic pointwise upper bound.
4. Mondrian bin corrections are piecewise and generally nondifferentiable at bin boundaries.
5. The existing TASK/CHARGER switch already acts on return-energy sufficiency. Adding a hard viability barrier can duplicate reserve and calibration conservatism.

Therefore Candidate C is **not a supported novelty claim**. It remains an ablation hypothesis only, preferably enabled during `CHARGER_COMMITTED` or tested against high-level switching alone.

## 8. Feasibility Under Bounded Acceleration

For one row \(a=2r\), the maximum attainable left side under the exact actuator set is

\[
\max_{u\in\mathcal U}a^\top u
=2a_{xy}^{\max}\lVert r_{xy}\rVert_2
+2a_z^{\max}|r_z|.
\]

The single obstacle constraint is feasible if and only if this value is at least \(b\). Multiple individually feasible constraints can still be jointly infeasible. Infeasibility must be surfaced explicitly. Emergency braking, hover, or an escape direction is not automatically safe; each fallback needs a separate viability argument.

**Proposition 2 (single-row bounded-input feasibility).** For one HOCBF row and the exact acceleration set, feasibility is equivalent to

\[
b\le2a_{xy}^{\max}\lVert r_{xy}\rVert+2a_z^{\max}|r_z|.
\]

For multiple rows, feasibility is equivalent to nonemptiness of the full convex intersection including acceleration and next-velocity limits; individual row tests are necessary but not sufficient. The implementation therefore reports solver infeasibility and whether the fallback satisfies every retained row.

## 9. Guarantee Boundaries

### Supported under assumptions

- Continuous-time collision-set forward invariance for exact geometry, the double-integrator model, valid initial HOCBF conditions, feasible bounded inputs, and continuous satisfaction of every active obstacle HOCBF.
- Strict convexity and uniqueness of the energy-aware quadratic projection when \(W\succ0\).
- Convexity of the candidate energy-viability inequality when \(R_E\succeq0\) and \(E_R\) is treated as differentiable at the evaluated state.

### Not supported

- Real quadrotor safety under attitude dynamics.
- No-collision guarantees from noisy or dropped 3D LiDAR without a verified perception-error bound. The implemented local sampled-hold residual bound covers exact static geometry and bounded ZOH dynamics only; it does not convert imperfect perception into exact geometry.
- Safety after dropping LiDAR constraints with top-K selection unless omitted constraints are certified inactive.
- Deterministic energy-returnability from conformal coverage.
- A universal lower-trajectory-energy theorem for either energy soft objective.
- Novelty of a generic collision-CBF plus battery-CBF optimization.

## 10. Prototype Selection

The controlled implementation compares:

1. one-step supporting-halfspace projection;
2. obstacle-wise second-order HOCBF with a small convex projection solver;
3. soft-min aggregate HOCBF;
4. HOCBF plus normalized instantaneous energy objective;
5. HOCBF plus the locally audited MC energy-to-go action gradient;
6. sampled-data-strengthened HOCBF;
7. sampled-data-strengthened HOCBF plus the same gradient objective.

Candidate B is enabled only as a falsifiable soft-cost ablation after the
finite-difference audit. Candidate C remains disabled because its novelty is
directly overlapped and its learned-bound premises do not support deterministic
viability. This ordering keeps collision feasibility and sampled-data safety
separate from learned-energy claims.

### 10.1 Post-experiment decision

The controlled results select the following deployment prototype:

\[
\min_{u\in\mathcal U}
\frac12\|u-u_{\mathrm{SAC}}\|_W^2
+\lambda_P\Delta t\,u^\top R_Eu
\quad\text{s.t. sampled-data-strengthened obstacle-wise HOCBF rows.}
\]

The selected development weight is the fixed, nondimensionalized
\(\lambda_P=0.1\) setting. On 125 held-out scenarios it retained 100% task
success and zero observed collision rollouts while reducing mean realized
trajectory energy from 4.8251 to 4.6535 synthetic units. The paired mean
difference was -0.1716 with a 95% bootstrap interval of
[-0.2071, -0.1294]. This is empirical evidence for Candidate A in the tested
distribution, not an energy-optimality theorem.

Candidate B is not selected. With a fixed physical gradient scale, its
incremental energy difference relative to Candidate A was -0.0396, but the 95%
bootstrap interval [-0.1062, 0.0002] crossed zero and the paired sign test was
not significant. Unit-direction normalization increased mean energy by 0.78%.
Thus local gradient correctness did not translate into a reliable additional
trajectory-energy benefit.

Candidate C remains a non-deployed theoretical comparison. It neither survived
the prior-art attack nor obtained an operational ablation that would justify
duplicating the existing high-level TASK/CHARGER commitment mechanism.

### 10.2 Perception and sampling boundary after robustness tests

Exact static geometry at 10 Hz and the implemented sampled-hold residual bound
produced zero observed collisions in the controlled test. Clipped radial range
error with a matching 1.5 m perception margin also produced zero observed
collisions in 30 short rollouts. In contrast, dropping an entire obstacle with
10% probability produced a 33.3% collision-rollout rate; the combined
noise/dropout/5 Hz condition produced 10.0%.

Accordingly, the proposition covers bounded geometry error only when the error
set is explicitly contained in the inflated obstacle. It does not cover missed
objects, unknown primitives, or real LiDAR segmentation failure. No deterministic
perception guarantee is claimed.
