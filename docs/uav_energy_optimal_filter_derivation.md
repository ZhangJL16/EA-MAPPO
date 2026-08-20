# Energy-Optimal Selection Inside a Fixed Certified Action Set

## Physical stage cost

The repository's `TelemetryCostModel` has the structure

\[
P(v,u)=P_{\rm fixed}+c_v^\top|v|+u^\top R_Eu,
\]

where the diagonal acceleration coefficients are nonnegative. Over one hold interval \(\Delta t\), the acceleration-dependent contribution is

\[
c_{\rm acc}(u)=\Delta t\,u^\top R_Eu.
\]

The units are synthetic energy units because the telemetry coefficients have not been physically calibrated to joules or watt-hours. \(R_E\succeq0\), so the action-dependent term is convex. It is strictly convex and the minimizer is unique only when \(R_E\succ0\) on the feasible directions.

### Unit audit

Acceleration (u) has units `m/s^2`.  Entries of (R_E) therefore have units
`synthetic_power / (m/s^2)^2`, so (u^\top R_Eu) is synthetic power and
(\Delta t\,u^\top R_Eu) is synthetic energy.  The feasibility margin
(A_iu-b_i) is instead `m^2/s^2`; it cannot be added to energy without an
explicit conversion coefficient, which is one reason the old weighted sum has
no invariant physical meaning.

For the optional learned-value bound, (B(u-u_0)) is a six-dimensional
physical-state increment with position components in metres and velocity
components in `m/s`.  A scalar Euclidean (L_E) is dimensionally meaningful
only after declaring a nondimensional state normalization/metric.  The deployed
feature map supplies such normalization numerically, but its ReLU and direction
singularities prevent the required verified smoothness bound.  We therefore do
not attach physical certificate units to an empirical network Hessian.

## Definition: feasibility-preserving progress floor

Let \(\mathcal U_s(x)\) be a fixed nonempty certified action set and let \(d_g\) be a unit goal-progress direction. Define

\[
\pi_{\max}(x)=\max_{u\in\mathcal U_s(x)}d_g^\top u,
\]

and

\[
\pi_{\rm req}(x)=\min\{d_g^\top u_{\rm nom},\pi_{\max}(x)\}.
\]

The energy action is

\[
u_E=\arg\min_u u^\top R_Eu
\]

subject to

\[
u\in\mathcal U_s(x),\qquad d_g^\top u\ge\pi_{\rm req}(x).
\]

This removes the arbitrary weighted sum between nominal deviation and physical acceleration energy. It does not remove the modeling choice that acceleration-aligned goal progress is the desired one-step performance quantity.

## Proposition 1: feasibility of the progress-constrained second stage

If \(\mathcal U_s(x)\ne\varnothing\), the second stage is feasible.

### Proof

Let \(u_\pi\) attain \(\pi_{\max}\), which exists by compactness. By definition, \(d_g^\top u_\pi=\pi_{\max}\ge\pi_{\rm req}\), so \(u_\pi\) is feasible for the second stage. ∎

## Proposition 2: pointwise energy dominance

For any comparator \(u_c\) satisfying the same hard constraints and progress floor,

\[
u_E^\top R_Eu_E\le u_c^\top R_Eu_c.
\]

### Proof

This is the defining optimality property of \(u_E\) over the common feasible set. ∎

This is a pointwise statement. It does not imply lower mission energy because the two controllers generally induce different states, path lengths, flight times, and fixed-power accumulation.

## Proposition 3: instantaneous safety-filter energy overhead bound

Let (u_s=u_n+d), where (u_n) is the nominal action and (u_s) is any
filtered action.  For the acceleration-power term (P_a(u)=u^\top R_Eu),

\[
P_a(u_s)-P_a(u_n)
=2u_n^\top R_Ed+d^\top R_Ed.
\]

If (R_E\succeq0), then

\[
P_a(u_s)-P_a(u_n)
\le \lambda_{\max}(R_E)
\left(2\|u_n\|_2\|d\|_2+\|d\|_2^2\right).
\]

Over a hold interval Δt, the same right-hand side multiplied by Δt
bounds the acceleration-energy overhead.  If (u_s) is the Euclidean
projection of (u_n) onto a closed safe set, then

\[
\|d\|_2=\operatorname{dist}(u_n,\mathcal U_s),
\]

so the bound is computable from the filter intervention norm.  For a weighted
projection, the analogous statement follows after norm equivalence with the
declared projection metric.

### Proof

Expand the quadratic at (u_n+d).  Cauchy--Schwarz and
(\|R_E\|_2=\lambda_{\max}(R_E)) give

\[
2u_n^\top R_Ed\le
2\lambda_{\max}(R_E)\|u_n\|_2\|d\|_2,
\]

and (d^\top R_Ed\le\lambda_{\max}(R_E)\|d\|_2^2).  The projection identity
is the definition of distance to the set. ∎

This is an upper bound on *instantaneous intervention overhead*, not a sign
guarantee: the filtered action can consume either more or less acceleration
power than the nominal action.  It also says nothing about flight time,
base-power accumulation, or total mission energy.

## Counterexample: pointwise acceleration energy need not reduce total energy

Suppose each step costs \(0.6+u^2\). A one-step action \(u=1\) reaches the goal with energy \(1.6\). Two actions \(u=0.5\) each have lower instantaneous acceleration energy, but total energy is

\[
2(0.6+0.5^2)=1.7>1.6.
\]

The base-power term makes slower progress more expensive. Therefore the strongest unconditional claim is pointwise acceleration-energy optimality under an explicit progress constraint.

## One-step Energy-to-Go upper bound

For exact ZOH double-integrator dynamics,

\[
\begin{bmatrix}p^+\\v^+\end{bmatrix}
=
\begin{bmatrix}p+v\Delta t\\v\end{bmatrix}
+B u,
\qquad
B=\begin{bmatrix}\frac12\Delta t^2I\\\Delta t I\end{bmatrix}.
\]

If a differentiable value function \(E\) has \(L_E\)-Lipschitz gradient on a convex region containing the complete segment between the reference next state and the candidate next state, the descent lemma gives

\[
E(x^+(u))\le E(x^+(u_0))
+\nabla E(x^+(u_0))^\top B(u-u_0)
+\frac{L_E}{2}\|B(u-u_0)\|_2^2.
\]

Since

\[
B^\top B=(\Delta t^2+\tfrac14\Delta t^4)I,
\]

the quadratic upper term is positive semidefinite. Adding \(\Delta t,u^\top R_Eu\) yields a convex quadratic objective when the state-to-network input map is affine and the declared \(L_E\) is valid.

## Why this bound is not currently certified for the deployed estimator

The frozen MC estimator is a two-hidden-layer ReLU network followed by Softplus. ReLU gradients jump at activation boundaries, so there is no finite global gradient-Lipschitz constant. Moreover, its actual 7D input contains relative goal direction and distance; the direction map is nonlinear and singular as goal distance approaches zero. A verified local theorem would require all of:

1. a trust region that does not cross any ReLU activation boundary;
2. a positive lower bound on goal distance over that region;
3. a verified Hessian/Jacobian bound for the complete physical-state-to-feature-to-network composition;
4. runtime rejection when any condition fails.

Without those checks, using an empirical Hessian norm or a conformal residual as \(L_E\) would mix deterministic and statistical semantics. The current prototype therefore does **not** claim certified one-step Energy-to-Go upper-bound optimality.

## Supersolution route

A finite-horizon energy bound would follow if a function \(\bar V_E\) satisfied

\[
\bar V_E(x,g)\ge c_E(x,u)+\bar V_E(F(x,u),g)
\]

for every applied action until the goal. Telescoping would give

\[
\sum_{k=0}^{T-1}c_E(x_k,u_k)\le\bar V_E(x_0,g).
\]

The current MC predictor and conformal upper residual do not verify this Bellman inequality. Held-out coverage is not a uniform Bellman-residual certificate. Consequently, no finite-horizon deterministic energy theorem is currently available.

## Experimental falsification of the two-stage realization

The valid pointwise propositions did not produce a useful closed-loop filter.
On 334 matched adversarial scenarios, the two-stage realization achieved only
`69/334` task success, mean path ratio `3.4560`, and mean realized energy
`15.9561`. Standard sampled-data HOCBF achieved `333/334`, path ratio `1.0360`,
and energy `4.8466`. The candidate retained 15 infeasible/uncertified fallback
steps and its worst-rollout P99 latency was `145.24 ms`.

This is consistent with the theory boundary: preserving a one-step acceleration
progress floor does not control flight time, detour, base-power accumulation,
or future feasibility. The candidate is therefore rejected even though its
pointwise optimization statement is correct.
