# UAV Joint Feasibility and Recursive-Feasibility Derivation

## Scope

This note studies the translational sampled-data model

\[
\dot p=v,\qquad \dot v=u,
\]

with the physical acceleration set

\[
\mathcal U=\{u\in\mathbb R^3:\|u_{xy}\|_2\le a_{xy},\ |u_z|\le a_z\}.
\]

It separates three questions that are often conflated:

1. Is the current HOCBF action set nonempty?
2. Does a selected current action keep the next action set nonempty?
3. Is collision safety preserved continuously between samples?

Only the first question is answered by the feasibility margin below.

## Obstacle-wise HOCBF row

For a static spherical obstacle with relative position \(r=p-p_o\), relative velocity \(w=v-v_o\), and safe radius \(d\), let

\[
h=r^\top r-d^2,
\qquad
\psi_1=\dot h+k_1 h.
\]

For known obstacle acceleration \(a_o\),

\[
\ddot h=2w^\top w+2r^\top(u-a_o).
\]

The exponential second-order HOCBF condition

\[
\ddot h+(k_1+k_2)\dot h+k_1k_2h\ge0
\]

is equivalent to the affine row

\[
A_i(x)u\ge b_i(x),
\]

where

\[
A_i(x)=2r_i^\top,
\]

\[
b_i(x)=-\left(2w_i^\top w_i-2r_i^\top a_{o,i}
+(k_1+k_2)\dot h_i+k_1k_2h_i\right).
\]

The row has units of metres and \(A_i u-b_i\) has units \(\mathrm{m^2/s^2}\). Therefore a common obstacle-wise slack margin also has units \(\mathrm{m^2/s^2}\). Actuator rows must not be mixed into the same scalar slack without a unit conversion; instead, actuator limits define the domain \(\mathcal U\).

## Definition 1: joint control-authority margin

For nonempty active obstacle set \(I(x)\), define

\[
\rho(x)=\max_{u\in\mathcal U}\min_{i\in I(x)}\{A_i(x)u-b_i(x)\}.
\]

Equivalently,

\[
\begin{aligned}
\rho(x)=\max_{u,\eta}\quad &\eta\\
\text{s.t.}\quad&A_i(x)u-b_i(x)\ge\eta,\quad i\in I(x),\\
&u\in\mathcal U.
\end{aligned}
\]

This is a diagnostic of current control authority, not yet a barrier certificate.

## Proposition 1: exact feasibility equivalence

Let

\[
\mathcal U_{\rm safe}(x)=\{u\in\mathcal U:A(x)u\ge b(x)\}.
\]

Then

\[
\mathcal U_{\rm safe}(x)\ne\varnothing
\iff \rho(x)\ge0.
\]

### Proof

If \(\mathcal U_{\rm safe}(x)\ne\varnothing\), choose any feasible \(u\). Every slack is nonnegative, so their minimum is nonnegative and the maximum over \(u\) is nonnegative. Conversely, if \(\rho(x)\ge0\), compactness of \(\mathcal U\) and continuity of the finite minimum imply that a maximizer \(u^\star\) exists. Its minimum slack is \(\rho(x)\ge0\), hence every row is satisfied. ∎

## Proposition 2: single-obstacle support-function form

For one row \(a^\top u\ge b\),

\[
\rho_1(x)=\sigma_{\mathcal U}(a)-b,
\]

where

\[
\sigma_{\mathcal U}(a)
=a_{xy}\|a_{xy\text{-components}}\|_2+a_z|a_z\text{-component}|.
\]

To avoid overloaded symbols, if the physical limits are \(\bar a_{xy},\bar a_z\) and the row is \(q=(q_x,q_y,q_z)\), this is

\[
\sigma_{\mathcal U}(q)=\bar a_{xy}\sqrt{q_x^2+q_y^2}+\bar a_z|q_z|.
\]

### Proof

The domain is the Cartesian product of a two-dimensional Euclidean ball and a scalar interval. Support functions add over Cartesian products. Cauchy--Schwarz gives the horizontal maximum \(\bar a_{xy}\|q_{xy}\|_2\), attained by aligning \(u_{xy}\) with \(q_{xy}\). The interval maximum is \(\bar a_z|q_z|\), attained at the endpoint with matching sign. ∎

## Proposition 3: minimax dual and infeasibility witness

Let \(\Delta_m=\{\lambda\ge0:\mathbf1^\top\lambda=1\}\). Then

\[
\rho(x)=\min_{\lambda\in\Delta_m}
\left[\sigma_{\mathcal U}(A(x)^\top\lambda)-b(x)^\top\lambda\right].
\]

### Proof

For any slack vector \(s\), \(\min_i s_i=\min_{\lambda\in\Delta_m}\lambda^\top s\). Thus

\[
\rho=\max_{u\in\mathcal U}\min_{\lambda\in\Delta_m}
\lambda^\top(Au-b).
\]

The sets are convex and compact and the integrand is bilinear. Sion's minimax theorem permits exchange of max and min. Maximizing the inner linear form over \(u\) gives the support function. ∎

Consequently, any \(\lambda\in\Delta_m\) satisfying

\[
\sigma_{\mathcal U}(A^\top\lambda)-b^\top\lambda<0
\]

is a separating infeasibility witness. This is the compact-set analogue of a Farkas certificate.

## Proposition 4: regularity

Assume a fixed finite active index set on an open region \(D\), compact \(\mathcal U\), and locally Lipschitz \(A_i,b_i\). Then \(\rho\) is locally Lipschitz on \(D\), hence continuous and differentiable almost everywhere.

### Proof

For \(f(x,u)=\min_i(A_i(x)u-b_i(x))\), each row is locally Lipschitz uniformly over compact \(\mathcal U\). A finite minimum preserves the common local Lipschitz bound. For any \(x,y\),

\[
|\max_u f(x,u)-\max_u f(y,u)|
\le\max_u|f(x,u)-f(y,u)|.
\]

Thus \(\rho\) is locally Lipschitz. Rademacher's theorem gives almost-everywhere differentiability. ∎

If the primal--dual saddle point \((u^\star,\lambda^\star)\) is unique and the active structure is stable, the envelope derivative is

\[
\nabla\rho(x)=\sum_i\lambda_i^\star
\left[\nabla_x(A_i(x)u^\star)-\nabla_x b_i(x)\right].
\]

At ties, actuator support-axis changes, or changing obstacle active sets, classical differentiability need not hold. The Clarke generalized gradient is contained in the convex hull of limiting gradients from neighboring stable active structures. This characterization does not itself create a feasible control law.

## Counterexample 1: pointwise feasibility is not recursive feasibility

Consider the sampled scalar system

\[
x_{k+1}=x_k+0.2u_k,\qquad u_k\in[-1,1],
\]

with current safety row \(u_k\ge x_k\). Its margin is \(\rho(x)=1-x\). At \(x_k=0.9\), \(\rho=0.1>0\), and \(u_k=1\) is feasible. Nevertheless, \(x_{k+1}=1.1\) and \(\rho(x_{k+1})=-0.1\). Therefore

\[
\rho(x_k)\ge0\not\Rightarrow\rho(x_{k+1})\ge0.
\]

The implication fails before any question of smoothness.

## Counterexample 2: a feasibility-CBF can be circular

For the same row and continuous dynamics \(\dot x=u\), \(\rho=1-x\). Enforcing

\[
\dot\rho+\rho\ge0
\]

gives \(u\le1-x\). At \(x=0.75\), the original safety row requires \(u\ge0.75\), while the feasibility-CBF requires \(u\le0.25\). The augmented optimization is empty although the original safety action set \([0.75,1]\) is nonempty.

## Consequence for the proposed theorem

The theorem "\(\rho\ge0\) plus a \(\rho\)-CBF guarantees recursive feasibility" is **invalid without an independently controlled-invariant domain or a certified backup action that satisfies both layers**. Adding a second CBF merely moves the feasibility obligation upward. A valid recursive-feasibility theorem must start from one of:

1. an explicitly verified controlled-invariant subset;
2. a backup-recoverable set with an admissible backup policy;
3. a discrete-time viability condition \(\exists u:\rho(F(x,u))\ge0\);
4. an MPC terminal invariant set.

All four are established control constructions. The scalar margin remains useful as a real-time diagnostic and adversarial-search objective, but not as a new recursive-feasibility theorem by itself.

## Braking-aware candidate and why it is not the missing theorem

Let (e=r/\|r\|_2) point from a spherical obstacle toward the UAV and let

\[
v_c=\max\{0,-e^\top w\}
\]

be the radial closing speed.  The largest acceleration available in the
outward radial direction under the cylindrical actuator set is the support
function

\[
a_b(e)=\sigma_{\mathcal U}(e)
=\bar a_{xy}\|e_{xy}\|_2+\bar a_z|e_z|.
\]

For a *fixed* radial direction, a static obstacle, and (a_b(e)>0), constant
maximum outward braking stops the radial motion after

\[
t_{\rm stop}=\frac{v_c}{a_b(e)},\qquad
d_{\rm stop}=\frac{v_c^2}{2a_b(e)}.
\]

This motivates the dimensionally consistent candidate

\[
h_{\rm brake}(r,w)=\|r\|_2-d_{\rm safe}
-\frac{v_c^2}{2a_b(e)}.
\]

Every term in (h_{\rm brake}) has units of metres.  The formula is exact for
the one-dimensional fixed-direction stopping problem and gives useful physical
intuition: a geometrically collision-free state may already be outside the
bounded-input stopping set.

It does **not** supply the requested multi-obstacle sampled-data recursive-
feasibility theorem.  During a 3D maneuver, (e) changes; tangential velocity
affects future radial motion; the action maximizing outward acceleration for
one obstacle can oppose that for another; zero-order hold and velocity
saturation alter the reachable stopping trajectory; and sensing error changes
the inferred radial geometry.  In particular, two nearby obstacles with
opposing outward normals can each satisfy their scalar stopping-distance test
while admitting no common braking action.  Thus separate conditions
(h_{{\rm brake},i}\ge0) do not imply nonemptiness of the joint action
intersection.

The braking construction is also a specialization of established braking
barrier, viability-kernel, and backup-CBF ideas.  Without an explicitly
computed multi-obstacle controlled-invariant backup set, it is neither a new
barrier principle nor a proof of online recursive feasibility.  We therefore
retain it only as a hard-state generator and diagnostic and reject it as the
core theorem candidate.

## Numerical certificate implementation

The implementation solves the primal max-min problem and the simplex dual. The
dual support objective is nonsmooth when its horizontal or vertical direction
vanishes, so direct SLSQP can stop near a nonoptimal kink. The implementation
therefore recovers dual weights from primal KKT conditions: nonzero weights are
restricted to minimum-slack rows and their weighted direction is constrained to
the normal cone of the cylindrical input set. This is a small linear
feasibility problem and supplies an independent dual certificate.

Across 100,000 random joint cases, the final implementation had zero primal or
dual certificate failures, zero feasibility-sign mismatches, and maximum
primal-dual gap `9.97e-8`. Mean/P95/P99 solve times were
`2.994/7.081/32.456 ms`. Three cases required the validation-only global dual
fallback; the maximum was `393.65 ms`, and 19 cases exceeded 50 ms. These
measurements support typical 20 Hz diagnostic use but explicitly disprove a
hard worst-case 20 Hz claim for the certificate implementation.

## Computational dependence on obstacle count

For one obstacle, the support margin is (O(1)): one 2D norm, one absolute
value, and scalar arithmetic.  Exact one-hold sphere verification is also
(O(1)) per obstacle because it evaluates two endpoints and the real roots of
one cubic derivative.  Checking all (K) independent quartics is therefore
(O(K)), although using them as simultaneous synthesis constraints remains
nonconvex.

The joint primal has four decision variables ((u_x,u_y,u_z,\eta)) and (K)
affine safety rows plus the cylindrical actuator constraint.  Its decision
dimension is constant, but each objective/constraint evaluation is (O(K)).
The minimax dual has (K) simplex variables; a generic dense Newton or KKT
linear solve can cost (O(K^3)), and an active-set method has no useful
worst-case polynomial iteration claim in this implementation.  The KKT
recovery LP is likewise (K)-dimensional.

The lexicographic candidate performs (1) one max-margin solve and (2) one convex
energy QP over the retained rows.  It is therefore two serial optimizations,
not one, and its observed latency reflects that composition.  Selecting top
(K) from (M) perceived obstacles can be implemented in (O(M\log K)), but
dropped obstacles are no longer covered by the joint certificate.  Top-(K)
is consequently a compute heuristic, not a theorem-preserving complexity
reduction unless a separate proof shows every dropped row redundant.
