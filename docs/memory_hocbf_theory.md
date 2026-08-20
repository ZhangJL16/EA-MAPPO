# Memory-Aware Robust HOCBF Theory

## Directional barrier

For a fixed unit support direction \(n\), UAV position \(p_u\), obstacle position
\(p_o\), and finite combined physical radius \(d\ge0\), define

\[
h_n=n^T(p_u-p_o)-d.
\]

If \(h_n\ge0\), then \(\|p_u-p_o\|_2\ge d\). This supporting halfspace is sufficient, not necessary, for spherical separation.

For double-integrator UAV input \(u\) and obstacle acceleration \(a_o\),

\[
\ddot h_n=n^T(u-a_o).
\]

The exponential relative-degree-two condition is

\[
\psi_2=n^T(u-a_o)+(k_1+k_2)n^T(v_u-v_o)+k_1k_2h_n\ge0.
\]

## Robust interval tightening

Let obstacle position, velocity, and acceleration errors lie in orthotopes with radii \(r^p,r^v,r^a\). Using their support functions, a sufficient robust constraint is

\[
\begin{aligned}
n^Tu &- n^T\hat a_o +(k_1+k_2)n^T(v_u-\hat v_o)\\
&+k_1k_2(n^T(p_u-\hat p_o)-d)\\
&\ge \sigma_{\mathcal E_a}(n)
+(k_1+k_2)\sigma_{\mathcal E_v}(n)
+k_1k_2\sigma_{\mathcal E_p}(n).
\end{aligned}
\]

## Theorem 2: robust output-feedback directional HOCBF

Assume \(k_1,k_2>0\); \(p_u,v_u,p_o,v_o\) are absolutely continuous;
their displayed double-integrator dynamics hold almost everywhere; inputs and
obstacle jerk are measurable and bounded; the UAV state is exact; \(d\ge0\) is
constant; each nonempty compact observer interval contains the true obstacle
state; and the unit support direction is fixed over the enforcement interval.
Define the robust lower bounds
\(\underline h_n=\hat h_n-\sigma_{\mathcal E_p}(n)\) and
\(\underline\psi_1=\hat{\dot h}_n-\sigma_{\mathcal E_v}(n)+k_1\underline h_n\).
If these two bounds are exactly nonnegative and the controller enforces the
tightened condition almost everywhere, then the true directional barrier remains
nonnegative, hence collision separation is maintained in ideal real arithmetic.

### Proof

For each uncertain term, its adverse projection is bounded by the corresponding support function. Therefore the tightened nominal inequality lower-bounds the true \(\psi_2\) by zero for every state in the interval. The standard comparison argument for the cascade \(\dot h=\psi_1-k_1h\), \(\dot\psi_1=\psi_2-k_2\psi_1\), with nonnegative initial values, yields \(\psi_1(t)\ge0\) and \(h(t)\ge0\). Since Euclidean norm dominates projection on a unit vector, \(\|p_u-p_o\|_2\ge n^T(p_u-p_o)\ge d\). \(\square\)

## Theorem 3: safe-action-set monotonicity

For a common actuator set \(\mathcal U_{act}\), define
\(\mathcal U_{safe}(\mathcal E)=\{u\in\mathcal U_{act}:n^Tu\ge b(\mathcal E)\}\).
Let nonempty compact uncertainty sets satisfy
\(\mathcal E_1^q\subseteq\mathcal E_2^q\) for \(q\in\{p,v,a\}\). For fixed
nominal state, gains, support direction, and actuator set, the robust safe-action
sets satisfy

\[
\mathcal U_{safe}(\mathcal E_1)\supseteq\mathcal U_{safe}(\mathcal E_2).
\]

### Proof

Support functions are monotone under set inclusion, so the right-hand tightening for \(\mathcal E_1\) is no larger than for \(\mathcal E_2\). Every action satisfying the latter inequality satisfies the former. \(\square\)

## Corollary: minimum-intervention objective monotonicity

For the same extended-real objective \(J(u)\) and actuator constraints, with \(\inf\varnothing=+\infty\),

\[
\inf_{u\in\mathcal U_{safe}(\mathcal E_1)}J(u)
\le
\inf_{u\in\mathcal U_{safe}(\mathcal E_2)}J(u).
\]

This proves monotonicity of the pointwise infimum. It does not require either set
to be nonempty because \(\inf\varnothing=+\infty\), and it does not prove that
realized mission energy decreases because future states and interventions change.

## Theorem 4: sampled-data hold strengthening

For a fixed support direction and ZOH UAV input, the true directional condition satisfies

\[
\dot\psi_2=-n^Tj_o+(k_1+k_2)n^T(u-a_o)+k_1k_2n^T(v_u-v_o).
\]

If the declared state intervals, input bounds, and obstacle jerk bound imply \(|\dot\psi_2|\le L_k\) throughout a hold of length \(\Delta\), enforcing

\[
\psi_2(t_k)\ge L_k\Delta
\]

implies \(\psi_2(t)\ge0\) for every \(t\in[t_k,t_k+\Delta]\), because

\[
\psi_2(t)\ge\psi_2(t_k)-L_k(t-t_k)\ge0.
\]

The implementation uses a norm/support-function upper bound on \(L_k\) and has
randomized grid regressions plus deterministic edge tests. These tests are not a
proof over continuous time; the displayed Lipschitz argument provides that
implication. Closed-loop safety still requires valid initial robust
\(h,\psi_1\), feasible constraints, and observer/jerk assumptions at every hold.
The controlled experimental runner in
`experiments/memory_safety/closed_loop.py` propagates the observer at every
control hold, checks exact nonnegative preconditions and feasibility, and labels
fallback or empirical maximum-residual boxes uncertified. This helper is not
wired into the repository's production collision filter.

The QP is solved against constraints strengthened by a fixed numerical reserve
and is called certifiable only after convergence and a post-solve check showing
strictly positive slack in the original constraints. Solver-level feasibility
within tolerance is not itself treated as certification. The controlled
simulator advances obstacle position and velocity under the same constant
realized jerk semantics used by the theorem, and collision diagnostics minimize
the resulting degree-six squared-distance polynomial over each hold by checking
the real roots of its degree-five derivative.

The exact-obstacle-state oracle baseline uses this same directional,
jerk-aware sampled-data constraint and the same reserve/post-solve check with
zero observer radii. It does not use the older constant-obstacle-acceleration
filter path, and it loses the certified label if simulator velocity saturation
changes the commanded acceleration.

The acceleration term uses the throughout-hold bound

\[
\|a_o(t)\|\le \|\hat a_o(t_k)\|+\|r^a_k\|_2+\|\bar j_{true}\|_2\Delta,
\]

not merely the acceleration bound at the sample. Here \(\bar j_{true}\) bounds
the total true obstacle jerk. It is distinct from the observer residual bound:
if \(|\hat j|\preceq\bar j_{nom}\) and
\(|j_o-\hat j|\preceq\bar d\), a valid choice is
\(\bar j_{true}=\bar j_{nom}+\bar d\). This extra jerk-growth term is required
because \(a_o\) appears both directly in \(\dot\psi_2\) and through the
relative-velocity growth over the hold.

## Probabilistic baseline boundary

Learned/Kalman/IMM empirical maximum-residual boxes are evaluated as
uncertified controller baselines only. Their constraints may be computed for
comparison, but `certification_valid=False` throughout: finite calibration-set
containment does not imply distribution-free or repeated-time closed-loop
coverage.
