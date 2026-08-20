# Theory Candidate V1: Audited Statements and Rejected Claims

## Scope

This document records the strongest statements that survived proof and
counterexample search. It deliberately does not name a new method. The valid
statements are mathematically useful, but the novelty audit finds them to be
existing convex-analysis, sampled-data, and constrained-optimization results
specialized to the UAV.

## Assumptions

### A1. Translational model

During one safety hold,

\[
\dot p=v,\qquad \dot v=u,
\]

and `u` is constant. This is needed for the affine HOCBF row and quartic exact
inter-sample trajectory. With attitude lag, jerk limits, or model error, the
quartic certificate is invalid unless these effects are bounded.

### A2. Bounded acceleration

\[
\mathcal U=\{u:\|u_{xy}\|_2\le \bar a_{xy},\ |u_z|\le \bar a_z\}.
\]

Compactness is needed for attainment of the feasibility margin. Different
actuator geometry changes the support function.

### A3. Static, exactly known spherical primitives

Obstacle `i` has fixed center and certified radius `d_i`. This is needed for
the exact quartic clearance calculation used here. Moving obstacles add their
relative velocity and acceleration; perception dropout invalidates any
constraint for the missing obstacle.

### A4. Fixed active obstacle set for classical derivatives

The active set is finite. Local Lipschitz regularity survives finite active-set
changes, but ordinary differentiability need not. Clarke generalized gradients
are needed at ties.

### A5. Physical energy stage model

The action-dependent energy over a hold is

\[
c_{acc}(u)=\Delta t\,u^\top R_Eu,\qquad R_E\succeq0.
\]

The units are synthetic simulation energy, not joules or watt-hours. Fixed
power and velocity terms affect trajectory energy even when they do not affect
the action minimizer at one state.

### A6. Smooth value only for the optional upper-bound lemma

The composed map `u -> Ehat(F(x,u),g)` must have `L_E`-Lipschitz gradient on a
declared trust region. The deployed ReLU estimator does not satisfy this
globally, and no useful verified local constant was established. Therefore the
optional lemma is not deployed as a certificate.

## Definitions

### Definition 1: obstacle-wise HOCBF row

For `r_i=p-p_i`, `h_i=r_i^T r_i-d_i^2`, and gains `k_1,k_2>0`, the
second-order exponential HOCBF is equivalent to

\[
A_i(x)u\ge b_i(x),\qquad A_i(x)=2r_i^\top.
\]

The complete expression for `b_i` is given in
`docs/uav_recursive_feasibility_derivation.md`.

### Definition 2: pointwise safe-action set

\[
\mathcal U_s(x)=\{u\in\mathcal U:A(x)u\ge b(x)\}.
\]

### Definition 3: joint feasibility margin

\[
\rho(x)=\max_{u\in\mathcal U}\min_i(A_i(x)u-b_i(x)).
\]

Its units are `m^2/s^2`; actuator limits define its domain and are not mixed
as dimensionless rows.

### Definition 4: exact one-hold clearance

For relative state `(r,w)` and constant relative acceleration `a`,

\[
q(\tau)=\|r+w\tau+\tfrac12a\tau^2\|^2-d^2,
\qquad \tau\in[0,T].
\]

Define `q_min` as the minimum over both endpoints and all real roots of the
cubic `q'(tau)` inside `[0,T]`.

### Definition 5: progress-preserving energy action

For a nonempty fixed hard set `U_s`, goal direction `d_g`, and nominal action
`u_n`, let

\[
\pi_{max}=\max_{u\in\mathcal U_s}d_g^\top u,
\qquad
\pi_{req}=\min\{d_g^\top u_n,\pi_{max}\}.
\]

Then

\[
u_E\in\arg\min u^\top R_Eu
\quad\text{s.t.}\quad
u\in\mathcal U_s,\ d_g^\top u\ge\pi_{req}.
\]

## Lemma 1: affine HOCBF constraint

Under A1--A3, the second-order exponential HOCBF condition is affine in `u`.

### Proof

`h_dot=2r^T w` and `h_ddot=2w^T w+2r^T u` for a static obstacle. Substitution
into `h_ddot+(k_1+k_2)h_dot+k_1k_2h>=0` isolates `2r^T u` on the left. SymPy
verified the identity in 100,000-state validation artifact
`artifacts/uav_theory_candidate_validation_20260819/report_100k_all_final.json`. ∎

## Lemma 2: exact single-row authority

For one row `a^T u>=b`,

\[
\rho_1=\bar a_{xy}\|a_{xy}\|_2+\bar a_z|a_z|-b.
\]

### Proof

The input set is the Cartesian product of a Euclidean disk and an interval.
Their support functions add. Cauchy--Schwarz and endpoint selection attain both
terms. ∎

## Theorem 1: exact pointwise joint-feasibility certificate

For finite `A,b` and compact convex `U`,

\[
\mathcal U_s(x)\ne\varnothing\iff\rho(x)\ge0,
\]

and

\[
\rho(x)=\min_{\lambda\ge0,\mathbf1^\top\lambda=1}
\left[\sigma_{\mathcal U}(A^\top\lambda)-b^\top\lambda\right].
\]

### Proof

The first equivalence follows because a nonnegative minimum slack is exactly
simultaneous row feasibility and the maximum is attained. For the dual, write
`min_i s_i=min_{lambda in simplex} lambda^T s`; compact convex domains and a
bilinear integrand satisfy Sion's minimax theorem. The maximization over `u`
is the support function. ∎

### Corollary 1: separating witness

Any simplex weight with negative dual objective proves the intersection empty.

### Remark

This theorem is pointwise. It is not recursive feasibility.

## Theorem 2: exact static-sphere inter-sample verification

Under A1--A3, a fixed action is collision-free with respect to a static sphere
throughout one ZOH interval iff `q_min>=0`.

### Proof

`q` is a quartic polynomial and continuous on a compact interval. Every global
minimum occurs at an endpoint or at an interior stationary point. `q'` is
cubic, so evaluating the endpoints and all real roots of `q'` in the interval
is exhaustive. ∎

### Corollary 2

Endpoint safety is insufficient. The numerical search found 74 cases among
100,000 where both endpoints were safe but the exact interior minimum was
unsafe.

### Limitation

The theorem verifies a selected action. The set of actions satisfying the
quartic minimum condition was not shown convex, and no new recursively feasible
action synthesis theorem follows.

## Proposition 1: second-stage feasibility and pointwise energy dominance

If `U_s` is nonempty, Definition 5 is feasible. For every comparator satisfying
the same hard constraints and progress floor,

\[
u_E^\top R_Eu_E\le u_c^\top R_Eu_c.
\]

### Proof

An optimizer of `pi_max` satisfies the progress floor by construction, proving
feasibility. The inequality is the defining property of the energy minimizer. ∎

### Failure case

This does not imply finite-horizon energy dominance. With per-step cost
`0.6+u^2`, one step at `u=1` costs `1.6`, while two lower-action-energy steps at
`u=0.5` cost `1.7`.

## Conditional Lemma: one-step Energy-to-Go upper bound

If A6 holds on a trust region and `B=[0.5T^2 I; T I]`, then for
`delta=u-u_0`,

\[
\widehat E(F(x,u),g)\le
\widehat E(F(x,u_0),g)+g_E^\top\delta+
\frac{L_E}{2}\|B\delta\|^2.
\]

### Proof

Apply the descent lemma to the next-state value and use the exact affine
next-state difference `F(x,u)-F(x,u_0)=B delta`. ∎

### Deployment status

**Rejected for the current estimator.** ReLU gradient jumps make a global
gradient-Lipschitz constant unavailable, and no verified useful local constant
was produced. Conformal coverage is statistical and cannot replace this
deterministic smoothness premise.

## Invalid theorem: recursive feasibility from `rho`

The implication

\[
\rho(x_k)\ge0\Rightarrow\rho(x_{k+1})\ge0
\]

is false. The scalar counterexample `x_{k+1}=x_k+0.2u_k`, `u>=x`, `u in
[-1,1]` has `rho(0.9)=0.1` and a feasible action `u=1`, but the next margin is
`-0.1`.

Adding a barrier on `rho` is also circular: at `x=0.75`, the original row can
require `u>=0.75` while the `rho` barrier requires `u<=0.25`. Therefore a valid
recursive theorem still needs an independently verified invariant/backup/
viability/terminal construction.

## Algorithmic prototype and falsification

The implemented two-stage filter instantiates Definition 5 after a sampled-data
HOCBF hard set. On 334 adversarial scenarios (1002 method trajectories):

- success was `69/334` for the two-stage candidate versus `333/334` for both
  sampled-data baselines;
- mean path ratio was `3.4560` versus `1.0360` for standard HOCBF;
- mean energy was `15.9561` versus `4.8466`;
- 15 infeasible/uncertified fallback steps remained;
- worst-rollout P99 filter latency was `145.24 ms`, exceeding the 20 Hz budget.

A second adversarial search over 100,000 physically parameterized UAV states
found 58,059 empty sampled-data action sets.  Of these, 10,225 still satisfied
the standard initial HOCBF-domain conditions `h>=0` and `psi1>=0`.  The margin
therefore exposes a genuine bounded-authority failure domain but does not close
it.  The 1,000 hardest states and full obstacle/velocity provenance are stored
in `artifacts/uav_hard_feasibility_set_20260819_v3/`.

Thus the computable candidate does not satisfy the required practical closure.

## Final status

| Claim | Proof status | Novelty status |
| --- | --- | --- |
| Exact pointwise margin and dual | Valid | Existing convex/CBF compatibility result |
| Exact one-hold static-sphere verifier | Valid | Special-case diagnostic within populated sampled-data CBF literature |
| Recursive feasibility from `rho` | Failed | Counterexample |
| Certified backup recursive feasibility | Not constructed | Existing backup/viability route |
| Pointwise physical action-energy optimality | Valid | Definitional constrained optimization |
| One-step learned-value upper bound | Conditional only | Standard descent lemma; assumptions unmet |
| Finite-horizon energy certificate | Not constructed | No verified supersolution |

No core theorem remains both valid and defensibly novel.
