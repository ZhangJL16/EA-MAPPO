# Derivation Package: Reliable Return-to-Charge

## Target

Derive a coherent theory line connecting:

1. a policy and a hard safety operator;
2. the executed closed-loop Resource-to-Go to a charger;
3. prediction under an unseen policy--filter composition;
4. the irreversible `CONTINUE` versus `RETURN` decision.

The immediate target is a set of paper-ready theorem statements and proof
obligations. This document does **not** claim that all statements already have
complete proofs or that the current deterministic simulator supplies a
non-degenerate conditional return distribution.

## Status

**COHERENT AFTER REFRAMING / EXTRA ASSUMPTION**

The theory is coherent when the invariant object is the charger hitting-resource
of the executed closed loop, not a generic q95 output. Identifiability requires
target executed-interface coverage. Distributional transport requires an explicit
regularity/coupling condition. A return-failure statement additionally requires a
calibrated upper requirement and a bound on decision-interval overshoot.

## Invariant Object

For charger set \(G_C\), define the hitting time and accumulated resource

\[
T_C=\inf\{t\ge 0:x_t\in G_C\},
\qquad
Z_C=\sum_{t=0}^{T_C-1}c_t.
\]

The distribution

\[
\mathcal Z_C^{\pi,\Pi}(x)
=
\mathcal L(Z_C\mid x_0=x,\pi,\Pi)
\]

is the top-level prediction object. In a deterministic system it is a point mass
and the relevant learned uncertainty is epistemic. In a declared stochastic
system it may also have an aleatoric tail.

The decision object is the one-way stopping rule

\[
d_t\in\{\textsf{CONTINUE},\textsf{RETURN}\},
\qquad
d_t=\textsf{RETURN}
\Longrightarrow
d_{t+k}=\textsf{RETURN}\quad(k\ge 0)
\]

until the charger is reached and a new battery cycle begins.

## Assumptions

- **A1 (deployment information).** \(x_t\) contains the information available to
  the predictor at deployment. Randomness remaining after conditioning on this
  information is aleatoric; uncertainty caused by finite data or extrapolation is
  epistemic.
- **A2 (queryable components).** The target navigation policy \(\pi\) and safety
  operator \(\Pi\) are known or can be queried without receiving target return
  labels.
- **A3 (executed interface).** Nominal action \(a\), executed action \(u\), and the
  primitive transition/resource kernel satisfy
  \[
  a\sim\pi(\cdot\mid x),\quad
  u\sim K_\Pi(\cdot\mid x,a),\quad
  (x',c)\sim K(\cdot\mid x,u).
  \]
- **A4 (shared primitive).** Across the policy--filter compositions considered in
  the first theory result, \(K\) is shared. A physics or battery-model shift is a
  separate extension and cannot be hidden inside a policy/filter composition.
- **A5 (proper SSP with first moment).** \(T_C<\infty\) almost surely under the
  target composition, \(\mathbb E[T_C]<\infty\), and
  \(0\le c_t\le c_{\max}<\infty\). The finite first moment is needed for finite
  \(W_1\) statements, not merely for almost-sure goal reaching.
- **A6 (interface identifiability).** For the positive result, \(K\) is identified
  on the support of the target occupancy over \((x,u)\). Merely observing each
  policy and each filter somewhere is not sufficient.
- **A7 (transport regularity).** For the finite-horizon error result, one-step
  target/model couplings and future-return Lipschitz moduli exist as stated below.
  This is an additional theorem assumption, not a consequence of A1--A6.
- **A8 (decision timing).** Return decisions are evaluated at declared policy-step
  intervals. Any energy consumed and requirement drift between checks must be
  bounded when a feasibility implication is claimed.
- **A9 (calibration).** Any probabilistic return guarantee is conditional on an
  explicitly valid calibration statement for the deployed composition or accepted
  subset. No arbitrary-shift coverage guarantee is assumed.
- **A10 (exponential transience).** For every risk parameter
  \(\lambda\) used by the certificate, the killed Feynman--Kac operator
  \(M_\lambda\) defined below is a bounded positive operator and
  \(I-M_\lambda\) has a bounded inverse. In a finite state space,
  \(\rho(M_\lambda)<1\) is sufficient. In a general state space this condition
  must be supplied by an exponential drift/transience argument; ordinary
  properness and \(\mathbb E[T_C]<\infty\) do not imply it.
- **A11 (risk-tilted domination).** For a positive source-to-target result, the
  target risk-tilted *executed-interface* measure
  \(\bar\eta_{\nu,\lambda}\) at each declared \(\lambda\) is absolutely
  continuous with respect to the source executed-interface measure
  \(\bar\rho_s\). Its density ratio belongs to the normed class used in the
  bound. This is stronger and more decision-specific than ordinary state overlap.
- **A12 (return-boundary margin).** For a fast decision-rate result, the deployed
  decision-state law \(\mu_D\) obeys, for some \(C_D<\infty\), \(\kappa>0\), and
  small \(t\),
  \[
  \mu_D\{|b-m-U_C(x)|\le t\}\le C_Dt^\kappa.
  \]
  This assumption is not needed for pointwise decision stability; it converts a
  requirement-estimation bound into a population disagreement or regret rate.
- **A13 (operational completion horizon).** The deployment protocol declares a
  finite policy-step deadline \(H\) for completing a goal under the frozen
  executed closed loop. Failure to hit by \(H\) means *deadline infeasibility*;
  it is not asserted to mean global or infinite-horizon unreachability. The same
  \(H\), goal set, policy, safety operator, dynamics, and stopping tolerance are
  used across methods whenever deadline-feasibility outcomes are compared.
- **A14 (two-macro-action objective).** At a task-mode decision epoch the
  admissible macro actions are \(R\), commit to the charger now, and \(M\),
  complete the current task and then return after the declared task-service
  reset. The decision is lexicographic: preserve a certified
  task--then--return route when one exists; otherwise take a certified direct
  return; if neither route is certified, enter an explicitly uncertified
  emergency fallback. This is a two-option viability oracle, not a globally
  optimal planner over arbitrary controllers.

## Notation

- \(x\): Markov deployment state or declared sufficient information state.
- \(a\): nominal action sampled from the navigation policy.
- \(u\): action executed after the safety operator.
- \(K_\Pi(du\mid x,a)\): deterministic or stochastic safety-operator kernel.
- \(K(dx',dc\mid x,u)\): shared executed-action transition/resource kernel.
- \(Q^{\pi,\Pi}\): closed-loop kernel obtained by composing \(\pi\), \(K_\Pi\),
  and \(K\).
- \(d_t^{\pi,\Pi}(x,u)\): target executed-interface occupancy at time \(t\).
- \(b_t\): remaining resource; \(m\): reserve.
- \(U_C(x)\): exact or conservative charger Resource-to-Go requirement.
- \(\widehat U_C(x)\): learned requirement used by the ReturnManager.
- \(H\): finite rollout horizon; \(Z_{C,H}=\sum_{t<\min(T_C,H)}c_t\).
- \(\mathcal E_H(x,g)\in[0,\infty]\): extended-real resource required to hit
  goal set \(G_g\) by deadline \(H\); it is \(+\infty\) when the frozen closed
  loop does not hit \(G_g\) by \(H\).
- \(R_H(x,g)=\mathbf 1\{T_g\le H\}\): finite-horizon reachability indicator.
- \(C_M(x,b)=\mathbf1\{b>m+\mathcal E_H^{\rm cont}(x)\}\): certified
  task--then--return macro-action.
- \(C_R(x,b)=\mathbf1\{b>m+\mathcal E_H^{\rm ret}(x)\}\): certified direct
  return macro-action.
- \(W_1\): Wasserstein-1 distance when its moment conditions hold.
- \(M_\lambda\): killed Feynman--Kac operator for exponential accumulated
  resource before charger hitting.
- \(r_\lambda\): terminal-entry term of the multiplicative return recursion.
- \(\psi_\lambda(x)=\mathbb E_x[e^{\lambda Z_C}]\): exponential Resource-to-Go
  value, with \(\psi_\lambda(x)=1\) for \(x\in G_C\).
- \(R_\lambda=(I-M_\lambda)^{-1}\): risk resolvent.
- \(\eta^X_{\nu,\lambda}=\nu R_\lambda\): non-normalized risk-tilted state
  occupation measure for initial distribution \(\nu\).
- \(\bar\eta_{\nu,\lambda}\): measure obtained by disintegrating every
  \(\eta^X_{\nu,\lambda}\) state visit through the target policy, safety kernel,
  and primitive transition/resource kernel; it lives on
  \((x,a,u,x',c)\).
- \(\bar w_\lambda=d\bar\eta_{\nu,\lambda}/d\bar\rho_s\): risk-tilted
  executed-interface density ratio when A11 holds.

## Derivation Strategy

Compose the known policy and safety operator through the executed-action
interface, identify the resulting path law, then map path-law error to accumulated
resource error. Finally, treat the ReturnManager as a threshold classifier and
derive exactly where prediction error can change its decision. Truncation and
probabilistic calibration enter as separate assumptions rather than being hidden
inside a generic “q95” label.

## Derivation Map

1. Define the closed-loop kernel by exact composition of \(\pi\), \(K_\Pi\), and
   \(K\).
2. Show that identification of \(K\) over target executed occupancy identifies the
   target path law and hence the hitting-resource law.
3. Construct two observationally equivalent primitive kernels outside source
   interface support to prove the necessity boundary.
4. Use a stepwise coupling and a Lipschitz future-return condition to unroll a
   finite-horizon \(W_1\) error recurrence.
5. Bound infinite-horizon truncation separately by the hitting-time tail.
6. Compare oracle and learned threshold decisions algebraically; disagreement is
   confined to a boundary band proportional to requirement error.
7. Add calibration and decision-interval overshoot only for a conditional return
   failure implication.
8. Compose task and return component bounds with a union bound; never relabel
   \(q_{.95}(X)+q_{.95}(Y)\) as \(q_{.95}(X+Y)\).
9. Replace a generic tail predictor by the killed Feynman--Kac fixed point for
   \(\psi_\lambda\) and state the exact exponential-transience requirement.
10. Apply the resolvent identity to express global exponential-value error as a
    risk-tilted occupation integral of local executed-interface error.
11. Define risk-tilted concentrability and prove the matching no-domination
    identification obstruction by a two-kernel construction.
12. Convert simultaneous upper log-MGF bounds into a return-energy chance bound;
    separate aleatoric tail probability from epistemic model failure.
13. Compose task and return exponential values exactly at a stopping time, without
    assuming component independence.
14. Add a boundary-margin condition to convert estimator error into irreversible
    stopping disagreement and regret rates.
15. Derive the neural critic, density-ratio moment equation, loss weighting, and
    data-acquisition consequences without claiming known risk-sensitive losses as
    new.
16. Separate finite truncated consumption from deadline-completion Resource-to-Go,
    then separate the task--return stopping boundary from the direct-return
    safety certificate when hybrid goal switching makes their feasible sets
    non-nested.

## Main Derivation

### Step 1. Closed-loop executed kernel — identity

The one-step closed-loop kernel is

\[
Q^{\pi,\Pi}(dx',dc\mid x)
=
\int \pi(da\mid x)
\int K_\Pi(du\mid x,a)
K(dx',dc\mid x,u).
\]

For a deterministic policy and filter, the two inner measures reduce to Dirac
measures. This identity makes the scientific interface explicit: the primitive
plant/resource model receives \(u\), not the nominal \(a\).

#### Step 1a. Current telemetry-resource decomposition — exact identity

The current simulator uses fixed step duration \(\Delta t\) and diagonal
nonnegative coefficients. With

\[
P_0=P_{\rm base}+P_{\rm compute}+P_{\rm communication},
\quad
A_v=\operatorname{diag}(\alpha_v),
\quad
A_u=\operatorname{diag}(\alpha_u),
\]

its realized step resource is exactly

\[
c_t
=\Delta t\left[
P_0+\alpha_v^\top|v_t|+u_t^\top A_u u_t
\right].
\]

Therefore the charger Resource-to-Go decomposes as

\[
\boxed{
Z_C
=\Delta t P_0T_C
+\Delta t\sum_{t<T_C}\alpha_v^\top|v_t|
+\Delta t\sum_{t<T_C}u_t^\top A_u u_t.}
\]

This is not an approximation. It proves that a policy or filter can alter energy
through three channels: charger hitting time, the induced velocity path, and the
executed-action quadratic term. In the present default configuration,
\(P_0=0.06\), while all velocity and acceleration coefficients are \(0.005\).

Let the safety correction be \(\Delta_t=u_t-a_t\). The instantaneous quadratic
change relative to nominal action is

\[
u_t^\top A_u u_t-a_t^\top A_u a_t
=2a_t^\top A_u\Delta_t+\Delta_t^\top A_u\Delta_t.
\]

Hence a safety intervention increases the action-dependent energy term if and
only if

\[
2a_t^\top A_u\Delta_t+\Delta_t^\top A_u\Delta_t>0.
\]

Braking corrections can make the cross term negative and reduce instantaneous
energy, even while detours or slower progress increase \(T_C\). This identity
rules out any blanket theorem that intervention magnitude alone must increase
energy. A useful model must retain correction alignment, executed action, and
hitting-time effects rather than using an unsigned intervention score as a cost.

### Step 2. Interface-compositional identifiability — proposition

Let the target pair \((\pi^\star,\Pi^\star)\) be unseen jointly. If A2--A6 hold,
then \(Q^{\pi^\star,\Pi^\star}\) is identified on every state reached before
\(T_C\). Consequently, every finite path-cylinder probability is identified.
Under A5, the induced hitting-resource law

\[
\mathcal Z_C^{\pi^\star,\Pi^\star}(x)
\]

is identified by the limit of its truncated path laws.

The key conclusion is limited but useful:

\[
\boxed{\text{pair-level trajectory overlap is not necessary}}
\]

provided target **executed-interface** support is identified. Component identity
coverage alone does not imply this condition.

### Step 3. Interface-support necessity — counterexample proposition

Suppose the target occupancy reaches a measurable interface region \(B\) with
positive probability, while the source data assign zero probability to \(B\).
Construct \(K_1\) and \(K_2\) that agree outside \(B\). On \(B\), let both kernels
transition to the charger but assign different bounded terminal costs
\(c_1\ne c_2\). Then the complete source-data laws are identical under \(K_1\)
and \(K_2\), while the target hitting-resource laws differ with positive
probability.

Therefore no source-observation-only estimator can uniformly identify the target
Resource-to-Go over this class:

\[
\boxed{\text{interface-OOD is an identification obstruction.}}
\]

This is an impossibility boundary, not a claim that every practical OOD point must
fail.

### Step 4. Finite-horizon transport — proposition with extra regularity

Let \(e_t(x)=W_1(\mathcal Z_{t:H}(x),\widehat{\mathcal
Z}_{t:H}(x))\). Assume a coupling of target and learned one-step kernels such that

\[
\mathbb E\left[
|c-\widehat c|+L_{t+1}d(x',\widehat x')
\mid x,u
\right]
\le \epsilon_K(x,u),
\]

where \(L_{t+1}\) is a valid Lipschitz modulus for the learned/target future-return
map. Using the Bellman path decomposition and the coupling definition of \(W_1\),

\[
e_t(x)
\le
\mathbb E_{a,u}[\epsilon_K(x,u)]
+
\mathbb E_{x'}[e_{t+1}(x')].
\]

Unrolling gives

\[
W_1(\mathcal Z_{C,H},\widehat{\mathcal Z}_{C,H})
\le
\sum_{t=0}^{H-1}
\mathbb E_{d_t^{\pi,\Pi}}[\epsilon_K(x_t,u_t)],
\]

for the exact-query setting of A2. Approximate policy or filter models require
their own coupling bounds and may add further terms, but those terms are not
asserted without a separate derivation. This proposition motivates an
occupancy-weighted local predictive-error score. It does **not** establish that
ensemble disagreement, density, or a learned residual predictor equals
\(\epsilon_K\).

### Step 5. Proper-SSP truncation — exact bound

Since costs are nonnegative and bounded,

\[
0\le Z_C-Z_{C,H}
\le c_{\max}(T_C-H)_+.
\]

Using the natural coupling of a trajectory with its own truncation,

\[
W_1(\mathcal Z_C,\mathcal Z_{C,H})
\le
c_{\max}\,\mathbb E[(T_C-H)_+].
\]

The total prediction error therefore separates into interface/model error and a
hitting-time-tail term. Long horizon is not itself a novelty claim; it is an
explicit amplifier and truncation source.

#### Step 5a. Deadline-completion Resource-to-Go — definition and exact propositions

The finite truncated consumption

\[
Z_{g,H}=\sum_{t=0}^{\min(T_g,H)-1}c_t
\]

is always finite under bounded per-step cost. If \(T_g>H\), however,
\(Z_{g,H}\) is only the resource spent before censoring; it is **not** the
resource required to complete the goal. Define instead the extended-real
deadline-completion requirement

\[
\boxed{
\mathcal E_H(x,g)
=
\begin{cases}
\displaystyle\sum_{t=0}^{T_g-1}c_t, & T_g\le H,\\[4pt]
+\infty, & T_g>H.
\end{cases}}
\tag{5a.1}
\]

Equation (5a.1) is a definition on the frozen executed closed loop, not an
estimate and not a claim of infinite-horizon unreachability. It prevents the
category error

\[
T_g>H
\quad\not\Rightarrow\quad
\text{the finite prefix cost }Z_{g,H}\text{ is a valid ETG.}
\]

For deterministic dynamics and a fixed goal-conditioned closed loop, if
\(H'\ge H\), then in the extended order

\[
\boxed{\mathcal E_{H'}(x,g)\le \mathcal E_H(x,g).}
\tag{5a.2}
\]

Indeed, a trajectory reaching by \(H\) has the same hitting cost under both
horizons; a trajectory first reaching in \((H,H']\) changes from \(+\infty\) to
a finite value; and a trajectory missing both deadlines remains \(+\infty\).
Thus horizon sensitivity must be reported rather than hidden by raising \(H\)
after observing failures.

Let \(g_T\) be the current task and \(g_C\) the charger. With \(x_T\) denoting
the task-hitting state when it exists, define

\[
\mathcal E_H^{\rm ret}(x)=\mathcal E_H(x,g_C),
\]

\[
\mathcal E_H^{\rm cont}(x)
=
\mathcal E_H(x,g_T)
+
\mathcal E_H(x_T,g_C),
\tag{5a.3}
\]

where addition is in \([0,\infty]\). Consequently, if either component of the
continue route misses its deadline, \(\mathcal E_H^{\rm cont}=+\infty\). This is
an exact compositional rule for deadline feasibility; it is separate from the
probabilistic quantile-composition issue in Step 8.

Let the certified macro-action indicators be

\[
C_M(x,b)=\mathbf1\{b>m+\mathcal E_H^{\rm cont}(x)\},
\qquad
C_R(x,b)=\mathbf1\{b>m+\mathcal E_H^{\rm ret}(x)\}.
\tag{5a.4}
\]

The deadline-aware irreversible rule under A14 is

\[
d_H(x,b)=
\begin{cases}
\textsf{CONTINUE}, & C_M(x,b)=1,\\
\textsf{RETURN}, & C_M(x,b)=0,\ C_R(x,b)=1,\\
\textsf{EMERGENCY\_RETURN\_UNCERTIFIED},
& C_M(x,b)=0,\ C_R(x,b)=0.
\end{cases}
\tag{5a.5}
\]

The last branch is intentionally not called a safe-return certificate. It says
that neither declared macro action certifies recovery; the operational system
may still commit to its least-bad emergency return, while the experiment records
the state as recovery-infeasible rather than silently using a truncated prefix.
If direct return is finite but task--then--return is infinite, (5a.5) commits
immediately.

Crucially, the converse nesting need not hold. Goal commitment changes the
goal-conditioned policy, while task completion applies a hybrid service map
\(J_T\) that sets velocity to zero before return. Therefore neither dynamics nor
triangle inequality implies

\[
C_M(x,b)\le C_R(x,b).
\]

Define the **viability-restoration region**

\[
\boxed{
\mathcal V_{\rm restore}
=\{(x,b):C_M(x,b)=1,\ C_R(x,b)=0\}.}
\tag{5a.6}
\]

On this region, an immediate-infeasibility-first rule chooses an uncertified
direct return even though the declared task--then--return route is certified.
Rule (5a.5) instead continues. Under deterministic exact macro rollouts, this
weakly dominates the old rule in certificate order and additionally completes
one task before charger arrival. It does not claim that direct return is
unreachable at an unbounded horizon.

Observed low progress over a recent window may be logged as a **stalled**
mechanism label but is not needed for (5a.1)--(5a.4) and is not a proof of global
unreachability. A run that is still progressing at \(H\) and a run trapped in a
local equilibrium are both deadline-infeasible under (5a.1), with distinct
diagnostic causes.

### Step 6. Return-boundary decision stability — exact proposition

Define the oracle and learned commitment rules at a decision instant:

\[
d^\star=\mathbf 1\{b\le U_C(x)+m\},
\qquad
\widehat d=\mathbf 1\{b\le\widehat U_C(x)+m\}.
\]

If \(|\widehat U_C(x)-U_C(x)|\le\varepsilon\), then

\[
d^\star\ne\widehat d
\Longrightarrow
|b-m-U_C(x)|\le\varepsilon.
\]

Proof is by interval ordering: outside this band, both thresholds lie on the same
side of \(b\). Thus global MAE is not the decision-relevant statistic; harmful
underestimation near the return boundary is the direct mechanism for late return.

### Step 7. Conditional return-failure implication — corollary with timing condition

Assume a deployed upper requirement satisfies

\[
\Pr(Z_C\le U_C(x)\mid\text{prediction accepted})\ge 1-\delta.
\]

If the agent commits at state \(x_t\) with

\[
b_t\ge U_C(x_t),
\]

then, excluding non-energy failure modes, the probability of energy exhaustion
before charger arrival is at most \(\delta\). A threshold-crossing implementation
obtains the premise only if reserve covers check-interval energy and requirement
drift. For an overshoot bound \(\Delta_{\rm check}\), a sufficient engineering
condition is

\[
m\ge\Delta_{\rm check}.
\]

The current experiments must measure this overshoot; the ReturnManager alone is
not a hard energy-safety certificate.

### Step 8. Task-plus-return risk allocation — exact probability inequality

Let \(E_1\) be task-completion resource and \(E_2\) be subsequent return resource,
defined on the same joint mission probability space. If

\[
\Pr(E_1\le U_1)\ge1-\delta_1,
\qquad
\Pr(E_2\le U_2)\ge1-\delta_2,
\]

then without an independence assumption, the union bound gives

\[
\Pr(E_1+E_2\le U_1+U_2)
\ge1-\delta_1-\delta_2.
\]

Therefore two 95% component upper bounds guarantee only the declared lower bound
of 90%, not a joint mission q95. A direct joint mission rollout/model is preferred;
otherwise component risk budgets must sum to the mission risk budget.

### Step 9. Killed Feynman--Kac Resource-to-Go — known foundation

For \(x\notin G_C\), define the positive killed operator and terminal term

\[
(M_\lambda f)(x)
=
\mathbb E_x\!\left[
e^{\lambda c_0}f(x_1)\mathbf 1\{x_1\notin G_C\}
\right],
\]

\[
r_\lambda(x)
=
\mathbb E_x\!\left[
e^{\lambda c_0}\mathbf 1\{x_1\in G_C\}
\right].
\]

Both expectations use the target closed-loop kernel from Step 1, so they compose
the nominal policy, safety operator, and primitive executed-action kernel. The
Markov property gives the multiplicative recursion

\[
\psi_\lambda=r_\lambda+M_\lambda\psi_\lambda.
\]

Under A10, the unique bounded solution is

\[
\boxed{
\psi_\lambda
=(I-M_\lambda)^{-1}r_\lambda
=\sum_{t\ge0}M_\lambda^t r_\lambda.}
\]

The series is the exponential analogue of an absorbing-chain fundamental
matrix. It accumulates paths that survive outside the charger for \(t\) steps and
then enter it. Exponential utility, this recursion, and its Chernoff connection
are prior art; they are the foundation, not the proposed novelty.

For comparison, at \(\lambda=0\) the ordinary mean-cost value \(v\) satisfies

\[
v=(I-Q)^{-1}g,
\]

where \(Q\) is the killed state kernel and \(g\) is one-step expected resource.
Thus an undiscounted TD implementation is theoretically meaningful only with an
SSP transience condition; it is not justified by importing a discounted
sup-norm contraction argument with \(\gamma=1\).

#### Step 9a. Resource--hitting-time sandwich — exact proposition

Suppose that before charger hitting the one-step resource is uniformly bounded
as

\[
0<c_-\le c_t\le c_+<\infty.
\]

Then pathwise

\[
c_-T_C\le Z_C\le c_+T_C,
\]

and for every \(\lambda\ge0\), monotonicity of the exponential gives

\[
\boxed{
\mathbb E_x[e^{\lambda c_-T_C}]
\le \psi_\lambda(x)
\le \mathbb E_x[e^{\lambda c_+T_C}].}
\]

Consequently, a finite hitting-time MGF at \(\lambda c_+\) is sufficient for
finite exponential Resource-to-Go, while divergence of the hitting-time MGF at
\(\lambda c_-\) is sufficient for divergence of \(\psi_\lambda\). If no
positive lower cost exists, only the upper implication remains. This proposition
shows that exponential Resource-to-Go can be primarily a hitting-horizon risk
measure when per-step resource varies little; an executed-interface contribution
must therefore be tested within matched horizon strata.

For the cumulant function \(K_x(\lambda)=\log\psi_\lambda(x)\), whenever
differentiation under the expectation is valid,

\[
K_x'(\lambda)=\mathbb E_{P_{x,\lambda}}[Z_C],
\qquad
K_x''(\lambda)=\operatorname{Var}_{P_{x,\lambda}}(Z_C),
\]

where \(dP_{x,\lambda}/dP_x=e^{\lambda Z_C}/\psi_\lambda(x)\). Convexity is
therefore exact, and growth of \(K_x''\) identifies a range in which empirical
exponential estimates become dominated by a small number of paths. This does not
replace an analytic exponential-drift proof for A10.

### Step 10. Executed-interface risk-resolvent perturbation — theorem candidate

Let a learned model induce \((\widehat M_\lambda,\widehat r_\lambda)\) and
\(\widehat\psi_\lambda=(I-\widehat M_\lambda)^{-1}\widehat r_\lambda\).
Subtracting the two fixed-point equations and resolving with the *target*
operator gives the exact identity

\[
\boxed{
\psi_\lambda-\widehat\psi_\lambda
=R_\lambda
\left[
(r_\lambda-\widehat r_\lambda)
+(M_\lambda-\widehat M_\lambda)\widehat\psi_\lambda
\right].}
\]

Define the local operator residual

\[
\Delta_\lambda
=(r_\lambda-\widehat r_\lambda)
+(M_\lambda-\widehat M_\lambda)\widehat\psi_\lambda.
\]

For an initial law \(\nu\), positivity yields

\[
|\nu(\psi_\lambda-\widehat\psi_\lambda)|
\le
\int |\Delta_\lambda(x)|\,
\eta^X_{\nu,\lambda}(dx),
\qquad
\eta^X_{\nu,\lambda}=\nu R_\lambda.
\]

This expression is sharper than generic one-step MAE: a local error matters in
proportion to the exponentially tilted mass of all pre-charger paths that reach
it. To expose the physical interface, define

\[
\begin{aligned}
\bar\eta_{\nu,\lambda}(dx,da,du,dx',dc)
={}&\eta^X_{\nu,\lambda}(dx)\,\pi(da\mid x)\\
&K_\Pi(du\mid x,a)K(dx',dc\mid x,u).
\end{aligned}
\]

This is the **target executed-interface risk occupation**. A policy ID or filter
ID is not a primitive causal input when A1--A4 hold; its mathematical role is to
induce this measure.

This resolvent identity is algebraic. The intended contribution is the theorem
package that instantiates it for unseen policy--filter compositions, supplies
verifiable norm conditions, and connects it to stopping decisions and finite
samples.

#### Step 10a. Exact discounted-occupancy reduction — identity

The finite-state proof package establishes an important prior-art boundary. For
any \(\gamma\in(\rho(M_\lambda),1)\), define

\[
v=\left(I-M_\lambda/\gamma\right)^{-1}\mathbf1
\]

and the Doob-scaled transient kernel

\[
P_{\lambda,\gamma}(i,j)
=\frac{M_\lambda(i,j)v_j}{\gamma v_i}.
\]

Adding cemetery probability \(1/v_i\) makes it stochastic, and the exact
similarity relation is

\[
\boxed{
(I-M_\lambda)^{-1}
=D_v(I-\gamma P_{\lambda,\gamma})^{-1}D_v^{-1}.}
\]

Thus finite-state risk occupation is a diagonally rescaled discounted occupancy
of a transformed killed chain. This means the adjoint ratio objective alone is
not new: DICE-style methods on the Doob chain are mandatory baselines. The
non-reduced difficulty is that \(v\) and the transformed target transition are
unknown and not directly sampled under source policy--filter compositions.

### Step 11. Risk-tilted overlap and identification boundary — theorem candidates

For any bounded positive critic \(h\), define its sample Bellman residual on an
executed transition \(z=(x,a,u,x',c)\) by

\[
\epsilon_{\lambda,h}(z)
=e^{\lambda c}
\left[
\mathbf 1\{x'\in G_C\}
+\mathbf 1\{x'\notin G_C\}h(x')
\right]-h(x).
\]

The fixed-point and resolvent identities imply the exact equality

\[
\nu(\psi_\lambda-h)
=\int \epsilon_{\lambda,h}(z)\,
\bar\eta_{\nu,\lambda}(dz).
\]

Let \(\bar\rho_s\) be the source measure on the same executed tuple and suppose
A11 holds. Then Cauchy--Schwarz gives

\[
\boxed{
|\nu(\psi_\lambda-h)|
\le
\|\bar w_\lambda\|_{L_2(\bar\rho_s)}
\|\epsilon_{\lambda,h}\|_{L_2(\bar\rho_s)},
\qquad
\bar w_\lambda
=\frac{d\bar\eta_{\nu,\lambda}}{d\bar\rho_s}.}
\]

The coefficient

\[
\mathfrak C_\lambda(\nu,\bar\rho_s)
=\|\bar w_\lambda\|_{L_2(\bar\rho_s)}
\]

is a risk-tilted executed-interface concentrability coefficient. It can be large
even when ordinary finite-horizon occupancy overlap looks acceptable, because
rare, long, high-resource paths are magnified by \(e^{\lambda Z_C}\).

The matching negative statement has the following proof target. If a measurable
interface region \(B\) has zero source mass but positive target risk-occupation
mass, choose two bounded primitive kernels \(K_0,K_1\) that agree on source
support and differ only on \(B\), with different terminal resource or charger
entry probabilities. Their complete source observation laws are identical while

\[
\theta_i=\nu\psi_{\lambda,i}
\]

differ. Hence every source-only estimator \(\widehat\theta\) obeys the two-point
lower bound

\[
\max_{i\in\{0,1\}}
\mathbb E_i|\widehat\theta-\theta_i|
\ge \frac{|\theta_1-\theta_0|}{2}.
\]

This is the formal reason that ensembles or arbitrary-shift conformal wrappers
cannot manufacture tail information where the target executed interface is not
identified. Strong parametric physics, bridge functions, or targeted queries may
restore partial or point identification, but each is an additional assumption.

The finite-state realization in
`docs/RETURN_TO_CHARGE_PROOF_PACKAGE.md` now proves two additional boundaries and
one positive finite-sample result.  First, ordinary source and target occupancy
can match exactly while the risk-occupation ratio at a merge state grows as
\(1-p+pe^{\lambda L}\).  Second, even two models with the same ordinary support
\(\{0,L\}\) require \(\Omega(e^{\lambda L})\) independent trajectories for
uniform constant-accuracy log-MGF estimation.  Third, an independent calibration
fold yields a clipped Bernstein certificate whose non-empirical terms are

\[
B\xi+\frac{BC_2}{\tau}
+B\sqrt{\frac{2C_2\log(2/\alpha)}{n}}
+\frac{2B\tau\log(2/\alpha)}{3n},
\]

where \(\xi\) bounds ratio \(L_1\) error and \(C_2\) bounds its second moment.
This makes the design implication precise: support, risk concentration, and
nuisance estimation are three separate obligations, and a small ordinary TD
loss addresses none of them by itself.

### Step 12. Log-MGF upper bounds to chance guarantees — corollary

For a deterministic finite grid \(\Lambda\subset(0,\infty)\), define the exact
Chernoff/EVaR requirement

\[
U_\delta(x)
=
\min_{\lambda\in\Lambda}
\frac{\log\psi_\lambda(x)+\log(1/\delta)}{\lambda}.
\]

The minimizing \(\lambda\) is selected from the return distribution, not from the
realized deployment resource. Markov's inequality then gives

\[
\Pr_x\{Z_C\ge U_\delta(x)\}\le\delta.
\]

Suppose training data produce simultaneous upper bounds

\[
\log\psi_\lambda(x)
\le \widehat L_\lambda^+(x)
\quad\text{for every }\lambda\in\Lambda
\]

with epistemic failure probability at most \(\alpha\), under a declared data and
deployment protocol. Then

\[
\widehat U_{\delta,\alpha}(x)
=
\min_{\lambda\in\Lambda}
\frac{\widehat L_\lambda^+(x)+\log(1/\delta)}{\lambda}
\]

satisfies the joint bound

\[
\Pr\{Z_C>\widehat U_{\delta,\alpha}(x)\}
\le\delta+\alpha.
\]

This explicitly separates aleatoric tail budget \(\delta\) from epistemic model
failure \(\alpha\). Establishing the simultaneous learned upper bound under
composition shift is a major proof obligation; ordinary ensemble spread is not
such a bound.

### Step 13. Exact task--return exponential composition — identity

Let \(\tau_T\) be the stopping time at which the task phase ends, let
\(E_T=\sum_{t<\tau_T}c_t\), and initiate the declared return controller from
\(x_{\tau_T}\). By the tower property and strong Markov property,

\[
\boxed{
\psi_\lambda^{\rm mission}(x)
=
\mathbb E_x\!\left[
e^{\lambda E_T}
\psi_\lambda^{\rm return}(x_{\tau_T})
\right].}
\]

This identity preserves dependence between task resource, terminal task state,
and return resource. It is a mathematically exact replacement for adding two
marginal q95 values. It also shows why task trajectories must cover the states
from which the return critic is queried.

### Step 14. Irreversible stopping boundary rates — proved population theorem

The corrected manager has one binary stopping boundary and one post-decision
certificate. Its binary action is exactly represented by

\[
U_t^{\rm eff}=U_t^{\rm task+return},
\qquad
s_t=b_t-m-U_t^{\rm eff},
\]

because it continues exactly when the full task--then--return macro action is
certified. \(U_t^{\rm return}\) separately certifies whether an actual
commitment is a safe return or an emergency fallback; it is not an additional
commit boundary in the non-nested hybrid system. Couple the
oracle and learned one-way rules until their first disagreement \(\tau_\Delta\).
The oracle-shadow law runs the oracle through cycle termination while evaluating
the fixed learned estimator on every oracle history. If a simultaneous error
event along this shadow path fails with probability at most \(\alpha\), then the
full proof in `docs/RETURN_TO_CHARGE_PROOF_PACKAGE.md` gives

\[
\boxed{
\Pr(\tau_\Delta<\tau_{\rm end})
\le
\alpha+
\mathbb E_\star\!\left[
\sum_{t<\tau_{\rm end}}
\mathbf1\{|s_t|\le\varepsilon_t\}
\right].}
\]

This expectation is explicitly under the oracle-shadow stopped law, resolving
the ambiguity in the earlier candidate statement. Under per-epoch margin bounds

\[
\mathbb P_\star(t<\tau_{\rm end},|s_t|\le r)
\le C_t r^\kappa,
\]

the first-disagreement probability is at most
\(\alpha+\sum_tC_t\varepsilon_t^\kappa\). Any cycle metric
\(F\in[0,B_F]\) therefore obeys

\[
|\mathbb EF^{\widehat d}-\mathbb EF^{d^\star}|
\le B_F
\left(\alpha+\sum_tC_t\varepsilon_t^\kappa\right).
\]

For stranding rate \(p\), bounded throughput \(q\le B_q\), and
\(\zeta_\Delta\) equal to the capped disagreement bound,

\[
p_{\widehat d}\le p_\star+\zeta_\Delta,
\qquad
q_{\widehat d}\ge q_\star-B_q\zeta_\Delta.
\]

Hence the learned rule preserves a population stranding ceiling \(\bar p\) and
throughput headroom \(g\) over the best eligible heuristic \(q_H\) whenever

\[
p_\star+\zeta_\Delta\le\bar p,
\qquad
q_\star-B_q\zeta_\Delta\ge(1+g)q_H.
\]

This is the first theorem in the chain that directly matches the original Pareto
success criterion. It remains a population result: formal Wilson bounds and
paired cycle uncertainty must be added by the experiment protocol. The learning
consequence is to focus sampling and loss weighting jointly on risk-tilted
interface mass and oracle-shadow return-boundary mass, not global prediction MAE.

### Step 15. Cross-fitted EIRR certificate to Oracle headroom — proved finite-state theorem

For each direct return or joint task--then--return branch, state, and risk
parameter, Theorem 11 can be applied with initial distribution \(e_x\). Allocate
calibration failure \(\alpha_{\rm cal}/Q\) over the finite global query set and
require the training-fold nuisance conditions to hold simultaneously. This gives

\[
|\psi_{k,\lambda}(x)-h_{k,\lambda}(x)|
\le c_{k,x,\lambda}
\]

for every query with epistemic failure probability at most
\(\alpha_E=\beta+\alpha_{\rm cal}\). When all lower endpoints are positive,
define

\[
\underline\psi=h-c,
\qquad
\overline\psi=h+c.
\]

Propagating these endpoints separately through the finite-grid EVaR functional
and selecting the task--then--return branch as the commitment boundary gives

\[
\underline U^{\rm eff}(x)
\le U^{\star,{\rm eff}}(x)
\le\widehat U^{+,{\rm eff}}(x),
\]

and therefore

\[
0\le
\widehat U^{+,{\rm eff}}(x)-U^{\star,{\rm eff}}(x)
\le
\varepsilon(x)
:=\widehat U^{+,{\rm eff}}(x)-\underline U^{\rm eff}(x).
\]

This converts the symmetric boundary band in Step 14 into the one-sided event

\[
0<s_t^\star\le\varepsilon(x_t),
\]

because a certified upper requirement can cause premature commitment but cannot
cause later commitment on the certificate event. The resulting bound is

\[
\Pr(\tau_\Delta<\tau_{\rm end})
\le
\alpha_E+
\mathbb E_\star\!\left[
\sum_{t<\tau_{\rm end}}
\mathbf1\{0<s_t^\star\le\varepsilon(x_t)\}
\right].
\]

If \(h_q\ge h_{\min}\), \(c_q\le a_n\le h_{\min}/2\), and
\(\lambda_{\min}=\min\Lambda\), interval propagation gives

\[
\varepsilon(x)
\le\frac{4a_n}{h_{\min}\lambda_{\min}}.
\]

Under a stopped margin exponent \(\kappa\), an actual
\(a_n=O(n^{-1/2})\) certificate therefore yields
\(O(\alpha_E+Hn^{-\kappa/2})\) first-decision disagreement and the same-order
stranding--throughput coordinate rectangle. The full formula, proof, and
non-claims are Theorem 13 in the proof package. Theorem 14 below supplies one
finite-tabular primitive plug-in construction for \(a_n\); it does not derive a
neural or direct learned-Doob rate.

The exact-risk comparator uses the same finite-grid risk functional. It must not
be identified with the formal simulator Oracle unless their requirements are
shown to coincide. In the deterministic conditional-return regime, both reduce
to a point Resource-to-Go comparison; under future stochastic disturbances they
need not coincide.

### Step 16. Finite-interface source-to-target rate — proved matching-order slice

Assume a finite state set and finite executed-action interface. At predictable
visits to each \((x,u^{\rm exec})\), estimate the primitive exponential cells

\[
K_\lambda(x,u,j)
=\mathbb E[e^{\lambda c}\mathbf1\{x'=j\}\mid x,u].
\]

An empirical Bernstein radius based on the cell sample variance is

\[
\eta_{\lambda,x,u,j}
=
\sqrt{\frac{2\widehat\sigma^2_{\lambda,x,u,j}\log(4D/\alpha)}{m_{xu}}}
+\frac{8e^{\lambda c_{\max}}\log(4D/\alpha)}{3(m_{xu}-1)}.
\]

The known target policy and filter induce weights \(\omega_x(u)\). Mixing the
cell radii by these weights yields computable operator radii

\[
\Delta_{M,\lambda}
=\max_x\sum_{j\in S}\sum_u\omega_x(u)\eta_{\lambda,x,u,j},
\]

\[
\Delta_{r,\lambda}
=\max_x\sum_u\omega_x(u)\eta_{\lambda,x,u,G_C}.
\]

With \(\widehat R_\lambda=(I-\widehat M_\lambda)^{-1}\), the robust condition

\[
\|\widehat R_\lambda\|_\infty\Delta_{M,\lambda}<1
\]

gives the fully explicit value radius

\[
c_\lambda^{\rm plug}
=
\frac{\|\widehat R_\lambda\|_\infty}
{1-\|\widehat R_\lambda\|_\infty\Delta_{M,\lambda}}
\left(
\Delta_{r,\lambda}
+\Delta_{M,\lambda}\|\widehat\psi_\lambda\|_\infty
\right).
\]

This radius can instantiate \(c_q\) in Step 15 without learning the unknown Doob
transform: estimate the shared primitive at the executed interface, then compose
the target policy/filter exactly.

The matching hard family makes the target always execute an informative
interface that the source executes only with probability \(\mu\). The two
primitive models differ only in a high-resource event with probabilities
\(e^{-\lambda L}\) and \(2e^{-\lambda L}\). If

\[
n\le\frac{e^{\lambda L}}{4\mu},
\]

every source-only target-MGF estimator has worst-case absolute error at least
\(9/32\). Thus constant accuracy requires

\[
n=\Omega(e^{\lambda L}/\mu).
\]

In this same family the exponentially weighted outcome has variance
\(\Theta(e^{\lambda L})\), so empirical Bernstein achieves the same
\(e^{\lambda L}/\mu\) order up to logarithms. This closes an honest finite-tabular
upper/lower slice. It also creates a strong stop condition: a learned DICE/EIRR
network needs a structural or computational advantage over this primitive
plug-in baseline, rather than merely reproducing its adjoint equation.

The lower bound also reaches the mission decision directly. Put the net remaining
resource halfway between the two exact single-risk requirements. The exact
manager then continues in the low-risk model and commits in the high-risk model.
Le Cam testing implies that below the same sample threshold every learned manager
makes one of these opposite decisions incorrectly with probability at least
\(3/8\). Embed premature commitment as unit normalized throughput regret in the
first model and late commitment as unit stranding excess in the second. Then

\[
\max\left\{
\mathbb E_0[q_0^\star-q_{\widehat d}],
\mathbb E_1[p_{\widehat d}-p_1^\star]
\right\}
\ge\frac38.
\]

Thus the exponential-risk/interface-coverage barrier is a lower bound on the
original Pareto decision, not only on an auxiliary prediction metric.

### Step 17. Learning consequence — derived design, not yet a theorem

Use a positive multi-risk critic \(\psi_\theta(x,\lambda)>0\) with terminal value
one and target

\[
Y_\lambda
=e^{\lambda c}
\left[
\mathbf 1\{x'\notin G_C\}\bar\psi(x',\lambda)
+\mathbf 1\{x'\in G_C\}
\right].
\]

The risk-tilted state measure is characterized by the adjoint balance equation

\[
\eta^X_{\nu,\lambda}
=\nu+\eta^X_{\nu,\lambda}M_\lambda.
\]

Let \(\bar\rho_s(dx,da,du,dx',dc)\) denote a source executed-transition
measure and \(\bar w_\lambda=d\bar\eta_{\nu,\lambda}/d\bar\rho_s\). Then, for
every test function \(f\), the corresponding source moment condition is

\[
\mathbb E_{\bar\rho_s}\!\left[
\bar w_\lambda(x,a,u)
\{f(x)-e^{\lambda c}\mathbf 1\{x'\notin G_C\}f(x')\}
\right]
=\mathbb E_\nu[f(x)].
\]

This suggests a DICE-like saddle estimator specialized to an unnormalized killed
Feynman--Kac occupation measure. A candidate critic loss is

\[
\mathcal L(\theta)
=
\mathbb E_{\bar\rho_s}\!\left[
\widehat{\bar w}_\lambda(x,a,u)
D_{\rm pos}(Y_\lambda\,\|\,\psi_\theta(x,\lambda))
\right],
\]

with cross-fitting between \(\widehat{\bar w}_\lambda\), targets, and critic
evaluation.
The positive-value divergence may use the existing Itakura--Saito method as a
baseline. The potentially new element is the executed-interface risk-resolvent
weight and its decision theorem, not the base divergence, policy conditioning,
or exponential Bellman target.

### Step 18. Risk-observable quotient plus stopped-boundary functional — frozen target

A focused continuous-interface audit rules out a direct “conditional mean
embedding plus resolvent perturbation” theorem as sufficient novelty. Conditional
mean operators, restricted-\(\chi^2\) OPE rates, function-class-induced state
aggregation, continuous bisimulation, minimax covariate-shift KRR, and
multiplicative-drift Feynman--Kac stability all have strong prior art.

Let \(q=(x,u)\), let \(\Lambda\) be the deployed finite risk grid, and let
\(\mathcal F_1\) be the unit ball of a multiplicatively Bellman-closed witness
class. Define

\[
\mathcal K_\lambda f(q)
=\mathbb E\!\left[e^{\lambda C}
\left\{\mathbf 1_{X'\in G_C}
+\mathbf 1_{X'\notin G_C}f(X')\right\}\mid q\right]
\]

and

\[
d_{\Lambda,\mathcal F}(q,q')
=\sup_{\lambda\in\Lambda,\,f\in\mathcal F_1}
\left|\mathcal K_\lambda f(q)-\mathcal K_\lambda f(q')\right|.
\]

The intended representation theorem must establish four linked claims while
keeping the one-step quotient separate from the nonlinear stopped-margin
functional:

1. an encoder that separates the zero-distance quotient exactly preserves the
   killed log-MGF grid and exact-risk return decision;
2. within-code distortion \(\varepsilon\) yields an explicit weighted-resolvent
   error and then the existing one-sided first-disagreement/Pareto rectangle;
3. an estimator attains a rate governed by risk-restricted quotient coverage or
   effective dimension rather than minimum raw continuous-interface density; and
4. a packing of risk-distinguishable quotient classes gives a matching lower
   bound on log-MGF or irreversible decision error.

This would yield a network/loss design rule: preserve conditional exponential
continuation features that are both risk-reachable and return-boundary-relevant;
permit invariance to other interface variation. It is not yet a theorem. If the
construction reduces exactly to existing restricted OPE on the known Doob chain,
the correct result is a negative equivalence theorem and the oral novelty claim
must be removed.

### Step 19. Two-layer orthogonal source-to-target score

The target execution kernel \(L\) is queryable, whereas the primitive killed
transition operator \(\mathcal K_\lambda\) is learned from source executed
interfaces. Collapsing them into one Bellman residual hides two different
support and approximation failures. For state continuation \(f\), interface
continuation \(g\), risk-state ratio \(v_\lambda\), and risk-interface ratio
\(w_\lambda\), define

\[
\mathcal S_\lambda(f,g;v,w)
=\nu f
+\mathbb E_{\rho_X}[v(X)(Lg(X)-f(X))]
+\mathbb E_{\rho_QK}
[w(Q)(\Gamma_{\lambda,f}(Y)-g(Q))].
\]

Theorem 17 proves two exact robustness routes: both ratios correct with arbitrary
\((f,g)\), or \(f=\psi_\lambda\) and
\(g=\mathcal K_\lambda\psi_\lambda\) with arbitrary ratios. For general
cross-fitted candidates its entire population bias is

\[
\mathbb E_{\rho_X}
[(\widehat v-v)(L\widehat g-\widehat f)]
+\mathbb E_{\rho_Q}
[(\widehat w-w)(\mathcal K_\lambda\widehat f-\widehat g)].
\]

This identity supplies a principled loss/estimator interface: learn and diagnose
the two residual blocks separately, and target products rather than raw
one-step MAE. It remains a specialization of standard orthogonal/doubly robust
OPE. The open oral-level question is whether killed exponential first passage
and the queryable execution layer yield a sharper structure-dependent rate,
matching lower bound, or irreversible-decision result unavailable from existing
DRL theory.

### Step 20. Canonical gradient removes the apparent second information layer

Under iid source primitives \(O=(Q,Y)\sim\rho_QP\), fixed \((\nu,L,\rho_Q)\),
and an unknown conditional primitive law \(P\), differentiating the killed
fixed point gives

\[
\dot\psi_\lambda
=(I-M_\lambda)^{-1}L
\mathbb E[\Gamma_{\lambda,\psi_\lambda}(Y)s(Q,Y)\mid Q].
\]

Consequently the canonical gradient is

\[
\varphi_\lambda(Q,Y)
=w_\lambda(Q)
\{\Gamma_{\lambda,\psi_\lambda}(Y)
-\mathcal K_\lambda\psi_\lambda(Q)\},
\]

and its squared norm is

\[
V_{\mathrm{eff},\lambda}
=\mathbb E_{\rho_Q}
[w_\lambda(Q)^2
\operatorname{Var}\{\Gamma_{\lambda,\psi_\lambda}(Y)\mid Q\}].
\]

At the true continuation pair, the state-composition correction from Step 19 is
zero pointwise. Thus that layer supplies double robustness when continuation
nuisances are inaccurate, but it is not extra efficient information when \(L\)
is known. Theorem 18 and its finite verifier close this semiparametric slice.
The result is mathematically useful but classically derived; novelty now requires
a new quotient/transience/stopped-boundary phase transition at the scale
\(\sqrt{V_{\mathrm{eff},\lambda}/n}\).

### Step 21. Exact scalar critical law and the multi-state target

For one noncharger state, multiplier \(a=e^{\lambda c}>1\), charger-hitting
probability \(p\), exact risk-quotient source mass \(\mu_h\), and
\(\Delta=1-a(1-p)>0\), Theorem 19 gives

\[
\psi=\frac{ap}{\Delta},\qquad
V_\psi=\frac{p(1-p)a^2(a-1)^2}{\mu_h\Delta^4},\qquad
V_{\log\psi}=\frac{(1-p)(a-1)^2}{\mu_h p\Delta^2}.
\]

This yields two distinct information thresholds: \(n\mu_h\Delta^4\) for
absolute raw-MGF estimation and \(n\mu_h\Delta^2\) for the log-risk used by the
ReturnManager. The verifier recovers exact normalized exponents \(-4\) and
\(-2\). The next proof target is no longer vague: for a finite positive killed
operator with simple Perron root \(1-\Delta\), uniformly separated remaining
spectrum, and nonzero primitive-noise projection onto its right Perron mode,
show \(V_\psi\asymp(\mu_h\Delta^4)^{-1}\) and
\(V_{\log\psi}\asymp(\mu_h\Delta^2)^{-1}\). Degenerate projection must appear as
a separate lower-order regime rather than being hidden by a generic bound.

### Step 22. Perron-mode phase split and the joint-limit boundary

For a finite irreducible killed operator with Perron root \(1-\Delta\), bounded
reduced resolvent, and normalized modes \((\ell_\Delta,z_\Delta)\), Theorem 20
uses

\[
(I-M_\Delta)^{-1}
=\frac{z_\Delta\ell_\Delta^\top}{\Delta}+H_\Delta
\]

inside the exact canonical gradient. The leading information coefficient is

\[
\mathcal C_0
=\sum_q
\frac{\{\ell_0(x)L_0(q\mid x)\}^2}{\rho_Q(q)}
\operatorname{Var}\{e^{\lambda C}\mathbf1_{X'\notin G_C}z_0(X')\mid q\}.
\]

When \(\mathcal C_0>0\), raw and log-risk efficiency variances scale as
\(\Delta^{-4}\) and \(\Delta^{-2}\). If the projected primitive is conditionally
deterministic on every leading interface, they instead scale no worse than
\(\Delta^{-2}\) and \(1\). Thus the spectral gap alone is insufficient: the
conditional-noise direction matters.

The fixed-\(\Delta\) LAN theorem and the \(\Delta\downarrow0\) variance theorem
are separate statements. A decision lower bound along \(\Delta_n\downarrow0\)
requires uniform triangular-array DQM/LAN and Lindeberg assumptions. This
correction prevents an iterated limit from being overclaimed as a joint theorem.

### Step 23. Critical information becomes a sampling rule

For interface acquisition cost \(\kappa(q)\), define the exact log-risk
sensitivity

\[
\alpha_\Delta(q)
=\frac{\eta^X_\Delta(x)L_\Delta(q\mid x)}{\theta_\Delta}
\sqrt{\operatorname{Var}(\Gamma_{\lambda,\psi_\Delta}\mid q)}.
\]

With independent stratum counts \(n_q\), the efficient variance is
\(\sum_q\alpha_\Delta(q)^2/n_q\). Theorem 21 solves the cost budget exactly:

\[
n_q^*\propto\frac{\alpha_\Delta(q)}{\sqrt{\kappa(q)}},
\qquad
\inf\mathcal V_\Delta
=\frac{\{\sum_q\alpha_\Delta(q)\sqrt{\kappa(q)}\}^2}{\mathsf B}.
\]

Near criticality, \(\Delta\alpha_\Delta(q)\to
\ell_0(x)L_0(q\mid x)\sigma_0(q)\). Hence the theorem-derived replay primitive
is left-mode target occupation times right-mode projected conditional noise,
cost-adjusted. A pilot whose sensitivities have uniform relative error
\(\varepsilon\) loses at most a factor
\((1+\varepsilon)/(1-\varepsilon)\) in second-stage efficiency. The unresolved
oral-level theorem is to obtain that relative error while learning all required
objects near criticality; the oracle allocation algebra is classical.

### Step 24. Allocation learning cancels the critical singularity

Let

\[
d_\Delta(q)=\ell_\Delta(x)L_\Delta(q\mid x)
\sqrt{\operatorname{Var}\{A(Y)^\top z_\Delta\mid q\}}.
\]

Under a fixed exact quotient, bounded primitives, a uniformly bounded Perron
reduced resolvent, and positive active projected variances, ordinary bounded-
moment concentration plus simple-eigenvalue perturbation gives

\[
\max_q|\widehat d_\Delta(q)/d_\Delta(q)-1|
=O_p(m_{\min}^{-1/2}).
\]

The exact canonical sensitivity obeys
\(\alpha_\Delta(q)=d_\Delta(q)\Delta^{-1}\{1+O(\Delta)\}\). Because the common
\(\Delta^{-1}\) cancels from allocation proportions, Theorem 22 obtains

\[
\frac{\mathcal V_\Delta(\widehat n)}{\mathcal V_\Delta(n^*)}
\le
\frac{B}{B-B_0}
\frac{1+K\{m_{\min}^{-1/2}+\Delta\}}
{1-K\{m_{\min}^{-1/2}+\Delta\}}
\]

up to logarithmic confidence factors. Thus a pilot can have
\(m_{\min}\Delta^2\to0\) and still learn the oracle design, even though the
final risk value requires \(B\Delta^2\to\infty\). Adaptive Neyman allocation is
classical; the candidate contribution is this near-critical scale separation and
the as-yet-unproved phase diagram when the effective Perron separation, projected-noise
floor, or quotient identifiability also collapses.

### Step 25. Critical information reaches the stopped Pareto boundary

For a frozen oracle-shadow query, let the regular requirement estimator have
sub-Gaussian variance proxy \(V_t\), and let the stopped score law satisfy
\(\Pr(t<\tau_{\rm end},|s_t|\le r)\le C_t r^\kappa\). A simultaneous radius

\[
\varepsilon_t
=\sqrt{2V_t\log(2J/\alpha)}
\]

inserted into Theorem 12 gives first-disagreement and Pareto-coordinate radius

\[
\alpha+
\{2\log(2J/\alpha)\}^{\kappa/2}
\sum_t C_tV_t^{\kappa/2},
\]

while locally boundary-weighted excess loss is controlled by
\(\sum_tC_tV_t^{(\kappa+1)/2}\) with the corresponding confidence factor.
For the scalar cost-aware critical design,

\[
V_\Delta^*
=\frac{\mathcal J_0+o(1)}{B\Delta^2}.
\]

Thus the Pareto rectangle and local excess-loss radii scale, up to logarithms,
as \((B\Delta^2)^{-\kappa/2}\) and
\((B\Delta^2)^{-(\kappa+1)/2}\). Theorem 23's adaptive variance ratio
\(R_\Omega\) enters with powers \(\kappa/2\) and
\((\kappa+1)/2\), respectively. A symmetric power-law margin with independent
Gaussian requirement error attains both powers exactly for the plug-in rule.
This closes the mathematical handoff to the original stranding--throughput
criterion, but it contracts only a valid uncertainty rectangle and does not
assert empirical Pareto dominance.

### Step 26. A shared replay design is a margin-powered compound allocation

For canonical query--stratum sensitivities \(a_{tq}\), one shared allocation
has \(V_t(n)=\sum_qa_{tq}^2/n_q\). The stopped-boundary objective is

\[
\mathcal R_p(n)=\sum_t\omega_tV_t(n)^p,
\quad
p=\kappa/2\ \text{or}\ (\kappa+1)/2.
\]

Theorem 25 proves this objective is convex for every \(p>0\), including the
otherwise delicate \(p<1\) regime, and derives the exact fixed point

\[
n_q^*\propto
\left[
\frac{\sum_t\omega_tV_t(n^*)^{p-1}a_{tq}^2}{c_q}
\right]^{1/2}.
\]

When every query shares the same critical gap, the common singularity cancels
from normalized replay priorities. With query-specific gaps, query \(t\) enters
with weight \(\Delta_t^{-2p}\), so the margin exponent determines how strongly
the sampler emphasizes the most nearly transient part of the stopped boundary.
This gives a neural/RL design primitive, but generic compound allocation and
active OPE design are prior art; learning the full sensitivity matrix with an
adaptive oracle guarantee remains the publishable-risk bottleneck.

### Step 27. Finite risk-witness quotient recovery fixes the representation target

For a finite raw interface alphabet, stack every declared terminal,
continuation, and first/second variance witness into \(W\), and define the
quotient by equality of \(\mu(u)=\mathbb E[W\mid u]\). Theorem 26 proves exact
recovery under separation \(\gamma\) at

\[
m_{\min}
=\widetilde O(B_W^2\gamma^{-2}),
\]

with a matching \(m\gamma^2=O(1)\) testing floor. A neural representation must
therefore decode these conditional witness moments within \(\gamma/4\);
reconstruction of raw inputs or Euclidean contrastive proximity is not a
substitute. This closes the finite representation target, but exact separated
classes are not an appropriate continuum assumption.

### Step 28. Continuous quotient learning creates a joint four-coordinate phase

Theorem 27 replaces exact continuum classes by a fixed finite set of
oracle-shadow interface queries in a declared intrinsic metric space. Under
local source mass \(c_Qh^{d_Q}\) and an \(\alpha\)-Hölder risk-witness mean, a
local structure-fold estimator has point-query radius

\[
\varepsilon_n
=\widetilde O\left(n^{-\alpha/(2\alpha+d_Q)}\right).
\]

Its learned moment pseudometric obeys the deterministic sandwich
\(|\widehat{\mathfrak d}_W-\mathfrak d_W|\le2\varepsilon_n\). Pooling at
threshold \(\rho_n\)
therefore creates operator bias
\(\tau_M\le C_M(\rho_n+2\varepsilon_n)\), without falsely assuming that
threshold proximity is transitive.

Positive-resolvent propagation amplifies this error by the risk-transience
margin \(\Delta_n^{-1}\), while the normalized critical replay priority
amplifies it by the Perron effective separation
\(\mathfrak g_n^{-1}\). Under stopped-margin exponent \(\kappa\), the
representation-induced disagreement and local boundary loss have orders

\[
\left\{\frac{\rho_n+\varepsilon_n}{\Delta_n}\right\}^{\kappa},
\qquad
\left\{\frac{\rho_n+\varepsilon_n}{\Delta_n}\right\}^{\kappa+1}.
\]

The resulting joint phase is

\[
\rho_n=o\{\min(\Delta_n,\mathfrak g_n)\},\qquad
n\Delta_n^{(2\alpha+d_Q)/\alpha}\to\infty,\qquad
n\mathfrak g_n^{(2\alpha+d_Q)/\alpha}\to\infty.
\]

Local Bernoulli Hölder bumps match the nonparametric exponent. Embedding them
into the scalar transient and two-state Perron constructions matches the
\(\Delta_n\) and \(\mathfrak g_n\) phase boundaries. Generic Hölder regression,
margin conversion, and spectral perturbation are classical; the candidate
contribution is the full
\(d_Q\)--\(\mathfrak g\)--\(\Delta\)--\(\kappa\) composition. ICML 2025
KROPE already occupies generic bisimulation-representation stability, so that
weaker headline is explicitly rejected.

### Step 29. The correct baseline boundary is risk-neutral failure but oracle-risk equivalence

Theorem 28 separates two superficially similar claims. If two terminal
interfaces have costs \(C=1\) and \(C\in\{0,2\}\) with equal probabilities,
their expected costs and next-state laws agree, so an expected-reward KROPE
representation may collapse them. Their exponential witnesses differ by

\[
\frac{1+e^{2\lambda}}2-e^\lambda
=\frac{(e^\lambda-1)^2}{2}>0,
\]

and a ReturnManager threshold between their log-MGFs produces different
actions. The risk-witness representation is therefore strictly necessary
relative to a risk-neutral baseline.

Likewise, nuisance raw interface coordinates can make target/source raw
\(\chi^2\) coverage grow as \(K-1\), while the exact risk quotient has
coefficient zero. This justifies quotient rather than raw support.

Neither fact separates the method from a correctly transformed baseline.
Theorem 6B maps the oracle killed-risk operator by positive diagonal similarity
to a discounted Markov chain. KROPE or DICE applied after that true transform is
therefore mandatory. The surviving research object is learning the unknown
transform from original first-passage primitives with a certificate satisfying
Theorem 27, not relabeling standard discounted OPE.

### Step 30. Unknown-transform learning is not itself the surviving object

Theorem 29 further corrects the last sentence of Step 29. Once the artificial
cemetery probability is retained, the Doob map is invertible:

\[
v_i=P_\gamma(i,\dagger)^{-1},
\qquad
M(i,j)=\gamma v_iP_\gamma(i,j)/v_j.
\]

For the same primitive estimate \(\widehat M\), direct resolvent plug-in and
Doob-chain plug-in give exactly the same value. Consequently a learned Doob
coordinate system cannot improve minimax information merely by
reparameterization.

Near transience its relative conditioning also remains \(1/\Delta\). In the
scalar family \(M=1-\Delta\), choosing
\(\gamma=1-c\Delta\) gives

\[
\frac{d\log v}{dM}
=\frac{1}{(1-c)\Delta}.
\]

Hence the viable object must be reframed again: the method must reduce effective
statistical complexity through the risk quotient or reduce decision loss
through stopped localization. Learning \(v\) and \(P_\gamma\) without one of
those gains is a coordinate change, not an Oral-level algorithmic contribution.

### Step 31. The valid dependent-data sample unit is a fresh primitive transition

Theorem 30 removes the iid-interface requirement without pretending replay
duplicates are data. At the \(k\)-th predictable hit of query \(z_t\),

\[
\xi_{t,k}=W_{\tau_{t,k}}-\mu(Z_{\tau_{t,k}})
\]

is a martingale difference even though the policy, state, and safety-filter
execution depend on the full past. The first \(m\) unique hits give

\[
\|\widehat\mu_{m,h}(z_t)-\mu(z_t)\|_\infty
\le L_\mu h^\alpha+
B_W\sqrt{\frac{2\log(2JD/\delta)}m}
\]

simultaneously over the declared queries. Therefore Theorem 27's joint phase
extends to adaptive chronological execution with \(m\) equal to fresh eligible
primitive transitions.

Sampling one stored transition \(R\) times leaves its empirical value unchanged
and creates no new martingale innovation. Using \(Rm\) in a confidence radius
would shrink it spuriously by \(R^{-1/2}\). The theorem-derived algorithmic rule
is consequently sharp: critical priorities may control which *new* transitions
to collect, while replay frequency may only control optimization and must not
enter certificate sample counts.

### Step 32. The sufficient prediction object is stopped-occupation \(L_p\) risk

Theorem 31 removes the need to estimate the transformed Energy-to-Go uniformly
over all states. If \(S\) is the exact stopped decision margin,
\(\widehat S=S-E\),

\[
\Pr(|S|\le t)\le C_0t^\kappa,
\qquad R_p=\mathbb E|E|^p,\quad p>1,
\]

then arbitrary dependence between \(S\) and \(E\) is allowed and

\[
\Pr(\text{first disagreement})
\lesssim C_0^{p/(p+\kappa)}R_p^{\kappa/(p+\kappa)},
\]

while local boundary-weighted loss obeys

\[
\mathcal L_\partial
\lesssim C_0^{(p-1)/(p+\kappa)}
R_p^{(\kappa+1)/(p+\kappa)}.
\]

An explicit symmetric boundary construction attains both powers, so no theorem
using only the same margin and \(p\)-moment assumptions can improve them. The
resulting algorithmic rule is to train and validate a cross-fitted
stopped-occupation weighted \(p\)-loss. Global MAE and repeated replay draws are
not evidence for that risk; oracle-shadow targets and weights must not leak
future deployment information.

### Step 33. Push both laws through the chart before measuring coverage

Theorem 32 turns the stopped-risk target into an estimable conditional object.
An independent structure fold fixes chart cells \(A_j\), with source masses
\(p_j\) and target stopped masses \(q_j\). The correct local overlap is

\[
\mathfrak C_h=\sum_{j:q_j>0}\frac{q_j}{p_j},
\]

where both laws are pushed through the chart before forming ratios. Under
bounded cross-fitted pseudooutcomes, chart distortion \(b_H\), Hölder score
regularity, and stopped nuisance remainder \(\tau_Y\), the source cell means
obey

\[
R_{2,\rm stop}
\lesssim
(b_H+L_Hh^\alpha)^2+\tau_Y^2
+\frac{B_S^2\log(J/\delta)}n\mathfrak C_h.
\]

For regular quotient dimension \(d_Q\), the score MSE rate is
\(n^{-2\alpha/(2\alpha+d_Q)}\) up to logarithms and declared nuisance terms.
Exponential transience multiplies the structural terms by \(\Delta^{-2}\), and
Theorem 31 then supplies the first-disagreement and boundary-loss powers.

The discrete nuisance example is exact: \(K\) irrelevant raw variants per
quotient cell multiply raw overlap by \(K\), while the pushed-forward quotient
coefficient is unchanged. A matched transformed OPE method that learns the same
quotient can inherit the same gain, so this is not yet an exclusive method
theorem.

### Step 34. Certified chart selection cannot create an information advantage by itself

On a finite executed-interface alphabet, an independent structure fold estimates
the stacked risk witness \(\mu(u)\). Uniform radius

\[
r_m=B_W\sqrt{\frac{2\log(2ND/\delta)}{m_{\min}}}
\]

implies that every candidate partition's empirical within-cell diameter differs
from its true diameter by at most \(2r_m\). Minimizing

\[
A_\Delta^2(\widehat\rho_k+2r_m)^2+V_k
\]

therefore achieves the oracle envelope

\[
A_\Delta^2\rho_{\widehat k}^2+V_{\widehat k}
\le
\min_k\{A_\Delta^2(\rho_k+4r_m)^2+V_k\}.
\]

An exact quotient candidate has chart contribution
\(O\{\log(ND/\delta)/(m_{\min}\Delta^2)\}\). This is a genuine learned-chart
rate for a finite candidate library, not for an arbitrary continuous encoder.

The same calculation kills a broader novelty claim. Given identical data and
candidate partitions, an unrestricted correct Doob/DICE/KROPE-side learner can
run the same selector. Invertibility of the cemetery-retaining Doob map makes
the direct and transformed decision-rule classes identical. Any strict
superiority theorem must therefore declare a computational, hypothesis-class,
regularization, or side-information asymmetry.

### Step 35. Oral theory contraction: one stopped-Pareto certificate, two information regimes

#### Target and status

The candidate paper-level object is not another estimator name or another
resource quantile. It is a finite-sample certificate from source
executed-interface data to the first irreversible manager disagreement and then
to the original stranding--throughput coordinates.

**Status: COHERENT AFTER REFRAMING / EXTRA ASSUMPTION.** The upper chain below
follows from Theorems 31--33. The localization powers are sharp given an
integrated score-error budget. Theorem 27 now contains a distributed Assouad
first-passage family that jointly realizes quotient dimension, transience,
stopped margin, and Pareto disagreement in the exact-chart, source-equals-target
strong-coverage slice. A matching minimax theorem for the *entire* pushed-
overlap, learned-chart, nuisance-bearing source-to-stopped-decision experiment is
still not proved.

#### Invariant object and normalized assumptions

Let \(\nu_{\rm stop}\) be the oracle-shadow law at the first decision query at
which exact and learned managers can diverge. Let \(S\) be the exact signed
ReturnManager score under this law and let

\[
E=\widehat S-S,
\qquad
R_{2,\rm stop}=\mathbb E_{\nu_{\rm stop}}E^2.
\tag{C.1}
\]

The measurable output is the coupled first-disagreement event
\(D=\{\widehat d\ne d^\star\}\), followed by deviations in stranding probability
and bounded throughput. The combined statement requires:

1. an independent structure fold that fixes or certifies the chart and cells;
2. an independent nuisance fold and an estimation fold of fresh primitive
   transitions, or an explicitly proved predictable first-hit martingale
   analogue;
3. source coverage \(p_j>0\) for every target-active stopped cell \(q_j>0\);
4. conditional invariance after the chart, chart error \(b_H\), pseudooutcome
   remainder \(\tau_Y\), and the boundedness/Hölder conditions of Theorem 32;
5. the stopped margin law
   \(\Pr_{\nu_{\rm stop}}(|S|\le t)\le C_0t^\kappa\) for
   \(0<t\le r_0\);
6. the absorbing ReturnManager coupling and a throughput coordinate with range
   at most \(B_q\).

Repeated replay draws are not fresh observations in item 2. Target stopped
weights or oracle-shadow scores computed from an outcome trajectory must be
cross-fitted or predictable; otherwise (C.1) is a training residual, not the
declared population risk.

#### Finite-sample map

Condition on the independent structure and nuisance folds and define the
Theorem 32 certificate

\[
\mathcal B_n(h,\delta)
=3(2b_H+L_Hh^\alpha)^2+3\tau_Y^2
+\frac{12B_S^2\log(4J/\delta)}{n}
  \mathfrak C_h,
\qquad
\mathfrak C_h=\sum_{j:q_j>0}\frac{q_j}{p_j}.
\tag{C.2}
\]

On its probability-\(1-\delta\) event,
\(R_{2,\rm stop}\le\mathcal B_n\). Theorem 31 with \(p=2\) then gives the
radius-valid, nonasymptotic bounds

\[
\boxed{
\Pr(D)
\le
\min\!\left\{1,
\inf_{0<t\le r_0}
\left(C_0t^\kappa+\frac{\mathcal B_n}{t^2}\right)
\right\},}
\tag{C.3}
\]

\[
\boxed{
\mathcal L_\partial
\le
\inf_{0<t\le r_0}
\left(C_0t^{\kappa+1}+\frac{\mathcal B_n}{t}\right).}
\tag{C.4}
\]

If the unconstrained optimizers lie below \(r_0\), they are

\[
t_D=\left(\frac{2\mathcal B_n}{\kappa C_0}\right)^{1/(\kappa+2)},
\qquad
t_L=\left(\frac{\mathcal B_n}{(\kappa+1)C_0}\right)^{1/(\kappa+2)}.
\tag{C.5}
\]

Substitution yields

\[
\Pr(D)
\le
K_D(\kappa)C_0^{2/(\kappa+2)}
\mathcal B_n^{\kappa/(\kappa+2)},
\tag{C.6}
\]

\[
K_D(\kappa)
=\left(\frac2\kappa\right)^{\kappa/(\kappa+2)}
+\left(\frac\kappa2\right)^{2/(\kappa+2)},
\tag{C.7}
\]

and

\[
\mathcal L_\partial
\le
K_L(\kappa)C_0^{1/(\kappa+2)}
\mathcal B_n^{(\kappa+1)/(\kappa+2)},
\tag{C.8}
\]

\[
K_L(\kappa)
=(\kappa+1)^{-(\kappa+1)/(\kappa+2)}
+(\kappa+1)^{1/(\kappa+2)}.
\tag{C.9}
\]

The absorbing first-disagreement coupling therefore gives the mission-level
rectangle

\[
|p_{\rm strand}(\widehat d)-p_{\rm strand}(d^\star)|
\le \Pr(D),
\qquad
|q(\widehat d)-q(d^\star)|
\le B_q\Pr(D).
\tag{C.10}
\]

This is the primary testable implication: lower cross-fitted stopped-score risk
must contract a preregistered Pareto uncertainty rectangle. It does not itself
prove empirical Pareto dominance.

#### The two regimes must not be silently merged

Theorem 27 works in a stronger finite-query uniform-error regime. If a score
certificate supplies \(\|\widehat S-S\|_\infty\le\eta_n\), then

\[
\Pr(D)\le C_0\eta_n^\kappa,
\qquad
\mathcal L_\partial\le C_0\eta_n^{\kappa+1}.
\tag{C.11}
\]

By contrast, (C.2) supplies only integrated squared error, so the valid powers
are \(\mathcal B_n^{\kappa/(\kappa+2)}\) and
\(\mathcal B_n^{(\kappa+1)/(\kappa+2)}\). Writing the faster powers from
(C.11) after an \(L_2\) bound would be a mathematical error. The shared
critical consistency scale in Theorems 27 and 32 does not make their statistical
experiments or loss transfers identical.

There is also a strict minimax warning. In the exact-chart, fixed-transience,
strong-density passive slice, classical Hölder plug-in classification has
boundary-weighted excess-risk order

\[
n^{-\alpha(\kappa+1)/(2\alpha+d_Q)}
\tag{C.11a}
\]

in the standard compatible-margin regime. Composing the global \(L_2\) rate
with (C.4) instead gives only

\[
n^{-2\alpha(\kappa+1)/\{(2\alpha+d_Q)(\kappa+2)\}},
\tag{C.11b}
\]

whose exponent is smaller by the factor \(2/(\kappa+2)\). This is not a
contradiction: Theorem 31 is sharp given only an integrated error budget, while
Hölder structure and strong local coverage give more information than that
budget records. Audibert--Tsybakov and the active-learning refinement audited in
`literature-search-20260830-stopped-lp-quotient-learning/` therefore rule out an
end-to-end minimax headline based only on (C.2)--(C.4).

#### Strong-coverage corollary: a second, faster valid route

The same cell proof supplies a uniform target-active certificate under a
stronger nuisance and coverage condition. Define

\[
p_{\min}=\min_{j:q_j>0}p_j,
\qquad
\tau_{Y,\infty}=\max_{j:q_j>0}|\bar b_{Y,j}|,
\tag{C.11c}
\]

and suppose the count event in Theorem 32 holds. Uniformly over the target-active
cells,

\[
\|\widehat S_h-S\|_{\infty,\nu_{\rm stop}}
\le
\varepsilon_{\infty,n}
:=2b_H+L_Hh^\alpha+\tau_{Y,\infty}
+2B_S\sqrt{\frac{\log(4J/\delta)}{np_{\min}}}.
\tag{C.11d}
\]

Consequently, whenever \(\varepsilon_{\infty,n}\le r_0\),

\[
\Pr(D)\le C_0\varepsilon_{\infty,n}^{\kappa},
\qquad
\mathcal L_\partial
\le C_0\varepsilon_{\infty,n}^{\kappa+1}.
\tag{C.11e}
\]

If \(p_{\min}\ge c_Ph^{d_Q}\), the chart is exact, and the uniform nuisance
vanishes, balancing (C.11d) gives

\[
\varepsilon_{\infty,n}
=O_{\mathbb P}\!\left[
\frac1\Delta
\left(\frac{\log n}{n}\right)^{\alpha/(2\alpha+d_Q)}
\right].
\tag{C.11f}
\]

At fixed \(\Delta\), (C.11e)--(C.11f) recover the classical passive
boundary-loss exponent (C.11a), up to logarithms. The classical source-equals-
target lower subclass makes that exponent minimax in its declared regular
regime; it does not establish a new generic classification rate.

The operational certificate should therefore take the smaller of two valid
routes:

\[
\Pr(D)
\le
\min\left\{
\inf_{0<t\le r_0}\left(C_0t^\kappa+\frac{\mathcal B_n}{t^2}\right),
C_0\varepsilon_{\infty,n}^{\kappa},
1
\right\}.
\tag{C.11g}
\]

The first route adapts to average pushed overlap \(\mathfrak C_h\) and tolerates
rare target-active cells by paying their weighted contribution. The second is
faster when every active cell has strong source coverage, but it can be vacuous
when a single \(p_j\) is small. This dual certificate is the correct mathematical
reason to maintain both an integrated stopped loss and calibrated per-cell
confidence radii; neither route uniformly dominates the other.

For a regular quotient of dimension \(d_Q\), Theorem 32 gives the score scale

\[
r_n
=\frac1\Delta
\left\{
\left(\frac{C_Q\log n}{n}\right)^{\alpha/(2\alpha+d_Q)}
+\rho_H
\right\}+\tau_Y,
\qquad
\mathcal B_n=O_{\mathbb P}(r_n^2),
\tag{C.12}
\]

up to fixed constants and the elementary square/sum inequalities. Consequently
the integrated-risk decision orders are

\[
\Pr(D)=O_{\mathbb P}
\left(r_n^{2\kappa/(\kappa+2)}\right),
\qquad
\mathcal L_\partial=O_{\mathbb P}
\left(r_n^{2(\kappa+1)/(\kappa+2)}\right).
\tag{C.13}
\]

#### Algorithm and loss consequences

Equations (C.2)--(C.13) impose four concrete design rules.

1. **Loss.** Train the signed requirement score with a cross-fitted
   stopped-occupation \(L_2\) loss. Global Energy MAE, an unweighted Bellman
   loss, or post-trajectory feature regression does not estimate (C.1).
2. **Representation.** Select the chart by the full certificate in (C.2), not
   reconstruction error alone. A smaller \(d_Q\) or \(\mathfrak C_h\) is useful
   only while \(b_H\) and \(\tau_Y\) remain controlled.
3. **Collection and replay.** Allocate *new* primitive transitions toward
   target-active cells that dominate \(q_j/p_j\) and the boundary risk. Replay
   may improve optimization but cannot increase \(n\) in (C.2).
4. **Comparison.** A matched Doob/DICE/KROPE learner must receive the same chart,
   folds, pseudooutcome information, and stopped loss. Theorem 33 proves that an
   unrestricted algorithm label cannot yield a strict statistical advantage.

#### Matching-lower-bound audit and non-claims

Theorem 31's construction proves that no transfer theorem using only a margin
law and an \(L_2\) budget can improve the exponents in (C.6) and (C.8). It does
**not** prove that every component of \(\mathcal B_n\) is jointly minimax.

The repaired Theorem 27 lower slice takes a standard strong-density
Hölder--margin Assouad hypercube and maps every signed score cell through the
exact inverse of the scalar killed first-passage requirement. A neighboring bit
changes the primitive termination probability by
\(\Theta(\Delta_ns_n)=\Theta(h_n^\alpha)\), so its \(n\)-sample KL is
\(O(nh_n^{d_Q+2\alpha})\), while the stopped active mass is
\(\Theta(s_n^\kappa)\). Thus, with

\[
h_n\asymp n^{-1/(2\alpha+d_Q)},
\qquad
s_n\asymp h_n^\alpha/\Delta_n,
\tag{C.14}
\]

one common exact-chart, source-equals-target first-passage family has lower
orders

\[
\Pr(D),\quad
\max\{\mathcal R_{\rm strand},\mathcal R_{\rm throughput}\}
\gtrsim \min\{1,s_n^\kappa\},
\qquad
\mathcal L_\partial\gtrsim\min\{1,s_n^{\kappa+1}\}.
\tag{C.15}
\]

This closes the joint \((d_Q,\Delta,\kappa)\) phase for the strong-coverage
uniform route, up to logarithms, and repairs the previous invalid jump from a
single-query Le Cam bump to a random stopped-margin law. It does not lower-bound
arbitrary pushed-overlap coefficients, chart selection error, cross-fitted
nuisance terms, or general source--target shift. For those terms the defensible
claim remains a sharp **risk-to-decision localization theorem composed with an
attainable estimator upper bound**, not a matching end-to-end minimax rate.
Likewise, (C.10) is a
stability rectangle rather than a theorem that the learned manager dominates a
heuristic, and none of these equations authorizes bypassing the navigation and
Oracle-headroom Gates.

## Diagnosis of the Current Estimator Families

### Monte Carlo return regression

With complete on-composition trajectories, the target

\[
Y^{\rm MC}=\sum_{t<T_C}c_t
\]

is unbiased for the conditional mean Resource-to-Go. It does not create a tail
certificate, and rare-event estimation needs trajectories that actually realize
the event. Estimating an event of probability \(p\) has relative sampling error
of order \((np)^{-1/2}\), so q95/q99 behavior can remain weak while mean MAE is
small. MC also gives no compositional transfer theorem when the target executed
interface is absent from training data.

### Mean TD and undiscounted bootstrapping

The formal recursion is an SSP equation with \(\gamma=1\), not a discounted MDP
contraction. Convergence requires properness, transience, or an appropriate
weighted norm. Bootstrapping reduces rollout variance but adds target/model bias,
and that bias is amplified by the ordinary resolvent \((I-Q)^{-1}\). Mean TD
cannot recover tail risk without an additional distributional or exponential
object.

### Quantile TD

Quantile TD is not covered by simply citing distributional Bellman contraction:
the quantile approximation operator itself need not be a contraction and can
have multiple fixed points. Sparse, nonuniform quantiles further require a
theory connecting their projection to the desired chance constraint. The current
project's unstable quantile component should therefore remain a baseline/failure
case, not the foundation of the theorem line.

### One-step world-model likelihood

A generic objective

\[
\min_\theta
\mathbb E_{\rho_s}
[-\log p_\theta(x',c\mid x,u)]
\]

weights errors by source frequency and likelihood geometry. Step 10 shows that
the downstream exponential return error is instead weighted by
\(\eta^X_{\nu,\lambda}\) and by \(\widehat\psi_\lambda\). Therefore excellent
one-step likelihood can coexist with a bad return decision. This is the precise
mathematical mismatch that a new loss should repair.

### Ensembles and calibration wrappers

Ensemble disagreement is an empirical epistemic score, not an upper bound on
\(|\Delta_\lambda|\). Conformal coverage needs exchangeability or a valid shift
weight; arbitrary policy--filter composition shift supplies neither. Both remain
useful empirical baselines, but neither overrides the Step 11 identification
boundary.

### Policy/filter conditioning and direct switching

Under A1--A4, the primitive kernel is conditioned on executed action \(u\), so
policy and filter identifiers are not intrinsically required for identification.
They can help only when state is insufficient, hidden filter state remains, or a
finite model benefits from context. A direct learned `RETURN/CONTINUE` switcher
may be strong in-domain, but it hides which tail resource law transfers and offers
no modular certificate. It is the appropriate end-to-end baseline against the
theorem-guided critic, not the main explanatory object.

## Remarks and Interpretation

- The executed action is a sufficient interface only if A1/A3 are valid. Hidden
  actuator, battery, wind, or perception state may require additional context.
- Policy conditioning may still help under insufficient state representations or
  finite-capacity approximation, but it is not required by the ideal Markov
  factorization.
- A learned trajectory reliability score should approximate the transport terms
  using cross-fitted errors. Training residuals are optimistically biased.
- The mission-level objective is a stranding--throughput Pareto frontier. Prediction
  metrics explain the mechanism but are not the final utility.
- `CONTINUE` versus `RETURN` is an optimal-stopping/one-way-switching view. The
  current threshold managers are baselines, not a proof of optimal stopping.
- The mean resolvent weights by expected visit count; the exponential resolvent
  weights paths by resource severity as well as visitation. Their empirical
  separation is a required mechanism experiment.
- The completed exploratory R3 dataset provides an initial falsification check,
  not formal Gate evidence: 930 trajectories and 370,572 executed-interface
  visits have energy--horizon correlation \(0.9802\). At normalized
  \(\beta=\lambda\,\mathrm{sd}(Z_C)=1\), terminal ESS falls to \(1.28\%\) and
  the top 5% of paths hold \(52.02\%\) of terminal tilted mass; risk-occupation
  ESS falls to \(1.20\%\), with the top 5% of paths holding \(56.01\%\) of
  visit mass. This supports severe risk concentration but also shows that horizon
  is a dominant confounder.
- The subsequent five-fold horizon-matched audit supplies narrow
  within-composition mechanism evidence. Adding aggregate executed-interface
  summaries to a strong post-trajectory control lowers held-out total-Energy MAE
  by \(3.9041\%\), with paired moving-block-bootstrap absolute improvement
  \(0.028031\) and 95% interval \([0.006198,0.051350]\). An approximate
  intervention-conditional permutation diagnostic gives upper-tail probability
  \(1/201\). Because duration, path length, and aggregate interventions are
  realized after the trajectory, this is neither a deployable ETG model nor a
  composition-transfer result.
- The signed-mean interpretation is falsified. In the highest intervention
  quintile the strong-model residual mean is \(-0.1108\), yet its standard
  deviation is \(1.3199\), q95 is \(2.0459\), and normalized beta-one log-MGF is
  \(0.7384\); in the lowest quintile these are \(0.7137\), \(1.1372\), and
  \(0.4270\), respectively. Beta-one residual-risk tilt enriches intervention by
  \(1.3538\times\). The resulting modeling obligation is to estimate the
  conditional exponential residual/tail law and its risk occupation, not to add
  an intervention count as a positive linear Energy penalty or merely rename it
  a structured input. Subgroup sign changes and near-flat Energy-rate MAE require
  the claim to remain heterogeneous and composition-specific.
- The literature and novelty map supporting Steps 9--17 is recorded in
  `literature-search-20260830-executed-interface-risk-resolvent/`.

## Boundaries and Non-Claims

- No theorem states that a safety intervention always increases energy.
- No theorem states that pair novelty alone causes failure.
- No learned energy estimator is a collision-safety or energy-safety certificate.
- No q90/q95 headline is valid until repeated fixed-information clone rollouts show
  non-degenerate conditional aleatoric variation.
- The current static deterministic environment supports point Resource-to-Go plus
  epistemic reliability, not a physical aleatoric tail claim.
- Wasserstein closeness alone does not imply stable q95/q99. A quantile corollary
  needs a CDF bound and local anti-concentration/positive-density assumption.
- The support necessity result does not forbid extrapolation under stronger
  parametric or physical structure; such structure must be declared.
- The CMDP comparison has different training privileges and belongs in a separate
  end-to-end decision track.
- Entropic risk, EVaR, Feynman--Kac recursions, multiplicative Bellman equations,
  policy-conditioned models, and Itakura--Saito exponential-value losses are not
  new contributions here.
- The risk-resolvent identity alone is an elementary operator identity. Novelty
  requires nontrivial composition-shift assumptions, finite-sample consequences,
  an impossibility boundary, and a stopping result.
- A finite log-MGF exists only on the \(\lambda\) range certified by A10. Heavy
  tails may force a truncated, polynomial-moment, or robust-risk alternative.
- Missing a finite deadline \(H\) is not claimed to imply infinite-horizon
  unreachability. Recent-window stagnation is a diagnostic label, not a theorem.
- The finite prefix cost \(Z_{g,H}\) is not a valid completed-goal ETG when
  \(T_g>H\); deadline infeasibility must propagate separately or as \(+\infty\).

## Open Risks

- The formal Oracle headroom Gate may show that perfect Resource-to-Go information
  has insufficient mission decision value over SOC/distance heuristics.
- All repair checkpoints R1--R5 failed the historical engineering navigation
  Gate, but the user subsequently fixed R3 as the conditional Energy-research
  platform. The continuous-workload endurance Gate now passes on 100/100 true
  depletion endpoints. The remaining Oracle blocker is deadline-completion
  semantics: the first formal attempt crashed when frozen R3+HOCBF missed the
  4000-step charger deadline, and paired diagnostics show policy/filter
  interaction rather than a mere runtime-budget issue.
- A generic executed-action world model plus Deep Ensemble may already match any
  proposed reliability mechanism.
- Interface extrapolation may be confounded with charger distance, obstacle
  density, or hitting horizon unless experiments use matched strata. The R3
  post-trajectory audit controls these within one composition and finds a small
  held-out total-Energy increment plus a stronger conditional tail increment,
  but composition transfer and deployable pre-decision estimation remain untested.
- Decision-interval overshoot may dominate estimator error near the return boundary.
- The local transport regularity assumption may be too strong for discontinuous
  safety-filter active-set changes; a weaker total-variation or piecewise analysis
  may be required.
- A second safety family and second dynamics/resource domain remain necessary for
  an oral-level generality claim.
- The closest unresolved prior-art risk is a Feynman--Kac occupation-density-ratio
  theorem that already specializes off-policy evaluation to exponential
  first-passage cost. A theorem-level search must remain active until submission.
- Risk-tilted ratios may have prohibitive variance. The finite-state clipped
  certificate exposes this through \(C_2\) and an explicit bias term, but useful
  nuisance-estimation rates and a trajectory-dependent extension remain open.
- The current deterministic simulator cannot validate an aleatoric tail theorem
  unless stochastic wind, battery, sensing, or actuation is declared and sampled.
