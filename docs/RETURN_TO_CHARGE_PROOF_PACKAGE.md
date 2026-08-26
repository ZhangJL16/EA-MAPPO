# Proof Package: Reliable Return-to-Charge

## Scope

This package formalizes the theory obligations in
`docs/RETURN_TO_CHARGE_DERIVATION_PACKAGE.md`.  Its invariant object is the
resource required by the **executed** closed loop to hit a charger set.  The
package does not claim that the current deterministic simulator has a
non-degenerate aleatoric return distribution, and it does not turn a learned
energy model into a hard safety certificate.

The package proves five corrected statements and two corollaries.  The main
correction relative to an informal simulation-lemma argument is that the
finite-horizon transport result requires an explicit coupling of the primitive
kernels and a Lipschitz condition on continuation-return laws.  The return
failure corollary also requires a bound on decision-interval overshoot.

## Status

**PROVABLE AFTER WEAKENING / EXTRA ASSUMPTION**

The interface-identification, non-identification, truncation, and threshold
claims are provable after making their measurable-space and timing assumptions
explicit.  The distributional transport claim is not justified from interface
coverage alone; it is proved below only under the coupling and continuation
regularity assumptions in A10--A12.  A generic q95-stability claim remains
unjustified without a separate CDF error bound and local anti-concentration
condition.

## Common Assumptions

- **A1 (measurable spaces).** The state, nominal-action, executed-action, and
  nonnegative resource spaces \(\mathsf X,\mathsf A,\mathsf U,\mathbb R_+\)
  are standard Borel spaces.  The charger set \(G_C\subseteq\mathsf X\) is
  measurable.
- **A2 (deployment information).** The state \(x\) is the information state
  available to the deployed predictor.  Randomness remaining after conditioning
  on \(x\) is aleatoric; uncertainty from finite data or extrapolation is
  epistemic.
- **A3 (queryable composition).** The target policy \(\pi(da\mid x)\) and safety
  operator \(K_\Pi(du\mid x,a)\) are known or queryable Markov kernels.
- **A4 (executed primitive).** The plant/resource primitive is a Markov kernel
  \(K(dx',dc\mid x,u)\).  The primitive receives the executed action \(u\), not
  the nominal action \(a\).
- **A5 (absorbing charger convention).** For \(x\in G_C\), the process remains in
  \(G_C\) and incurs zero further resource.  This convention lets finite-horizon
  sums be written without a separate stopping indicator.
- **A6 (shared primitive for composition claims).** The primitive kernel \(K\)
  is shared across the policy--filter compositions under comparison.  A plant,
  wind, payload, actuator, or battery-model shift is not hidden inside this
  assumption.
- **A7 (proper SSP with first moment).** For every initial state considered,
  the charger hitting time
  \[
  T_C=\inf\{t\geq 0:x_t\in G_C\}
  \]
  is almost surely finite and has \(\mathbb E[T_C]<\infty\).  Per-step resource
  satisfies \(0\leq c_t\leq c_{\max}<\infty\).
- **A8 (target-interface identification).** Let \(K_0\) be the true primitive and
  let \(\mathcal K(\mathcal D)\) be all primitive kernels observationally
  compatible with the source-data law.  For every
  \(\widetilde K\in\mathcal K(\mathcal D)\), \(\widetilde K\) and \(K_0\) agree
  up to \(d_t^{\pi^\star,\Pi^\star,K_0}\)-null sets for every target-executed
  interface visited before \(T_C\).  Observing each component somewhere is not
  enough.
- **A9 (exact component query).** In the first transport result, the learned
  rollout uses the same queryable \(\pi\) and \(K_\Pi\) as the target.  Only the
  primitive \(K\) is replaced by \(\widehat K\).
- **A10 (one-step primitive coupling).** For each \((x,u)\), there is a measurable
  coupling \(\gamma_{x,u}\) of
  \(K(\cdot\mid x,u)\) and \(\widehat K(\cdot\mid x,u)\).
- **A11 (continuation-law Lipschitz regularity).** For each finite-horizon time
  \(t+1\), the learned continuation-return law
  \(\widehat{\mathcal Z}_{t+1:H}(x)\) obeys
  \[
  W_1\!\left(
    \widehat{\mathcal Z}_{t+1:H}(x),
    \widehat{\mathcal Z}_{t+1:H}(y)
  \right)
  \leq L_{t+1}d(x,y)
  \]
  for a declared state metric \(d\) and finite constant \(L_{t+1}\), with
  \(L_H=0\) for the zero terminal continuation law.
- **A12 (finite local transport error).** Under \(\gamma_{x,u}\),
  \[
  \epsilon_t(x,u)
  :=
  \mathbb E_{\gamma_{x,u}}
  \left[
    |c-\widehat c|+L_{t+1}d(x',\widehat x')
  \right]
  <\infty.
  \]
- **A13 (decision-time calibration).** A probabilistic upper requirement
  \(U_t\) is used only on an explicitly declared accepted set and satisfies
  \(\Pr(Z_{C,t}\leq U_t\mid\text{accepted at }t)\geq1-\delta\) for the deployed
  composition.
- **A14 (bounded decision-interval overshoot).** Between consecutive return
  checks, the decrease in remaining resource plus the increase in required
  charger resource is at most \(\Delta_{\mathrm{check}}\).

## Notation

- \(a_t\): nominal policy action.
- \(u_t\): action after the safety operator.
- \(Q^{\pi,\Pi}_K\): closed-loop state/resource kernel generated by
  \(\pi,K_\Pi,K\).
- \(Z_C=\sum_{t=0}^{T_C-1}c_t\): charger Resource-to-Go.
- \(Z_{C,H}=\sum_{t=0}^{H-1}c_t\): truncated Resource-to-Go under A5.
- \(\mathcal Z_C(x)\): law of \(Z_C\) given \(x_0=x\).
- \(d_t^{\pi,\Pi,K}\): target occupancy distribution of \((x_t,u_t)\).
- \(b_t\): remaining resource at a return-decision instant.
- \(m\): nonnegative return reserve.
- \(U_t\): exact or calibrated conservative charger requirement.
- \(\widehat U_t\): learned requirement used by the ReturnManager.

## Dependency Map

1. Theorem 1 uses A1--A8 and the Ionescu--Tulcea construction of a path law.
2. Theorem 2 uses only measurable source/target occupancies and a two-kernel
   construction on an unobserved target interface region.
3. Theorem 3 uses A1--A5 and A9--A12, plus the coupling characterization and
   triangle inequality of \(W_1\).
4. Theorem 4 uses A5 and A7 with the natural coupling of a trajectory and its
   truncation.
5. Theorem 5 is an exact deterministic threshold argument.
6. Corollary 1 uses A13--A14 and the one-way commitment rule.
7. Corollary 2 is the union bound on a single joint mission probability space.

## Theorem 1: Executed-Interface Compositional Identifiability

### Claim

Fix an initial state \(x_0=x\).  Suppose A1--A8 hold for a target pair
\((\pi^\star,\Pi^\star)\), even if that pair was never jointly used to generate a
source trajectory.  Then the finite-dimensional path law of the target executed
closed loop is uniquely identified up to its charger hitting time.  Consequently,
the law \(\mathcal Z_C^{\pi^\star,\Pi^\star}(x)\) is uniquely identified.

### Status

**PROVABLE AFTER EXTRA ASSUMPTION**

The claim needs standard Borel spaces, a declared initial distribution, a proper
SSP, and target-interface rather than component-only identification.

### Proof Strategy

Compose the queryable policy and safety kernels with the identified primitive,
apply uniqueness of the induced path measure, and pass from truncated resource
sums to the hitting-resource limit.

### Proof

For any state \(x\), define

\[
Q^{\pi^\star,\Pi^\star}_K(dx',dc\mid x)
=
\int_{\mathsf A}\pi^\star(da\mid x)
\int_{\mathsf U}K_{\Pi^\star}(du\mid x,a)
K(dx',dc\mid x,u).
\]

By A1, A3, and A4, this is a well-defined Markov kernel.  Fix any
\(\widetilde K\in\mathcal K(\mathcal D)\).  A8 states that \(\widetilde K\) and
the true \(K_0\) agree
\(d_t^{\pi^\star,\Pi^\star,K_0}\)-almost everywhere on the interfaces reached by
the true target before \(T_C\).  Since \(\pi^\star\) and \(K_{\Pi^\star}\) are
fixed and queryable, integrating these agreeing primitive kernels gives the same
closed-loop kernel at true target-reached states, up to true target path-law null
sets.

Starting from the fixed measure \(\delta_x\), the Ionescu--Tulcea theorem gives a
unique probability measure on every finite path cylinder by iterated application
of \(Q^{\pi^\star,\Pi^\star}_K\).  Induction on the cylinder length shows that
\(K_0\) and \(\widetilde K\) induce the same finite path-law: the base
distribution is identical, and the next conditional kernel agrees at almost
every state reached under the common preceding marginal.  The induction also
shows that the \(\widetilde K\) marginal equals the \(K_0\) marginal at the next
time, so A8 continues to apply at every finite step.

For each \(H\), the truncated sum \(Z_{C,H}\) is a measurable function of a
finite path prefix.  Its pushforward law is therefore uniquely identified.  By
A5 and A7,

\[
0\leq Z_C-Z_{C,H}\leq c_{\max}(T_C-H)_+
\]

and \(\mathbb E[(T_C-H)_+]\to0\) as \(H\to\infty\), because \(T_C\) is
integrable.  Thus \(Z_{C,H}\to Z_C\) in \(L^1\), hence in distribution.  A
probability law has at most one weak limit, so the uniquely identified truncated
laws identify \(\mathcal Z_C^{\pi^\star,\Pi^\star}(x)\).  The proof never uses
joint source trajectories from the pair \((\pi^\star,\Pi^\star)\).  Therefore
pair-level overlap is not necessary under target-interface identification.  ∎

### Corrections or Missing Assumptions

- “Every policy and every filter was observed” is insufficient; A8 is required.
- The result is identification, not a finite-sample learning rate.
- It does not cover a changed primitive kernel.

## Theorem 2: Non-Identification Outside Observed Interface Support

### Claim

Suppose a measurable interface region \(B\subseteq\mathsf X\times\mathsf U\)
has zero probability under every source trajectory distribution but is reached
before the charger with positive probability under a target composition.  Over a
model class that permits two different bounded resource kernels on \(B\), no
estimator using only the source observations can uniformly identify the target
Resource-to-Go law.

### Status

**PROVABLE AS STATED AFTER DECLARING THE MODEL CLASS**

### Proof Strategy

Construct two primitive kernels that are observationally equivalent on every
source trajectory but assign different resource at the first target visit to
\(B\).

### Proof

Choose two distinct constants \(r_1,r_2\in[0,c_{\max}]\).  Let \(K_1\) and
\(K_2\) agree at every \((x,u)\notin B\).  On \(B\), let both kernels move to a
fixed charger state \(x_C\in G_C\), but let the incurred resource be respectively
\(r_1\) and \(r_2\):

\[
K_i(dx',dc\mid x,u)
=
\delta_{x_C}(dx')\delta_{r_i}(dc),
\qquad (x,u)\in B,quad i\in\{1,2\}.
\]

Because every source occupancy assigns probability zero to \(B\), replacing
\(K_1\) with \(K_2\) changes no source conditional law on an event with positive
source probability.  Hence the complete source-data distributions under the two
models are identical.  An estimator receiving only those observations must have
the same output distribution under \(K_1\) and \(K_2\).

Before the first target visit to \(B\), the two target processes have identical
kernels and therefore identical path laws.  Let \(\tau_B\) be that first visit.
By assumption, \(\Pr(\tau_B<T_C)>0\).  On this event the two processes have the
same accumulated resource before \(\tau_B\), then incur different terminal
resource \(r_1\neq r_2\).  Their target Resource-to-Go laws therefore differ.
No common estimator output can equal both distinct target laws, so no
source-observation-only estimator can identify the target law uniformly over
this model class.  ∎

### Corrections or Missing Assumptions

- This is a structure-free impossibility statement.  A known parametric physics
  law may permit extrapolation into \(B\), but that extra structure must be
  declared.
- It does not say every interface-OOD sample causes large practical error.

## Theorem 3: Finite-Horizon Distributional Transport

### Claim

Under A1--A5 and A9--A12, let
\(\mathcal Z_{t:H}(x)\) and
\(\widehat{\mathcal Z}_{t:H}(x)\) be the target and learned laws of resource
accumulated from time \(t\) to \(H-1\), starting from the same state \(x\).  Then

\[
W_1\!\left(
  \mathcal Z_{0:H}(x_0),
  \widehat{\mathcal Z}_{0:H}(x_0)
\right)
\leq
\sum_{t=0}^{H-1}
\mathbb E_{(x_t,u_t)\sim d_t^{\pi,\Pi,K}}
[\epsilon_t(x_t,u_t)].
\]

### Status

**PROVABLE AFTER EXTRA ASSUMPTION**

Interface coverage alone does not imply this bound.  A10--A12 are substantive
regularity assumptions, especially near discontinuous safety-filter active-set
changes.

### Proof Strategy

At each common starting state, couple true and learned primitive outcomes using
the same nominal and executed action.  Split the continuation discrepancy into
model error at the true next state and sensitivity to next-state displacement,
then unroll the recurrence under the true target occupancy.

### Proof

Define

\[
e_t(x)=W_1\!\left(
\mathcal Z_{t:H}(x),
\widehat{\mathcal Z}_{t:H}(x)
\right),
\qquad e_H(x)=0.
\]

Fix \(x\) at time \(t\).  Draw the same \(a\sim\pi(\cdot\mid x)\) and
\(u\sim K_\Pi(\cdot\mid x,a)\) for the target and learned one-step rollouts,
which is valid by A9.  Conditional on \((x,u)\), draw
\((x',c,\widehat x',\widehat c)\) from the coupling \(\gamma_{x,u}\) in A10.
For each coupled pair of next states, the triangle inequality gives

\[
\begin{aligned}
W_1\!\left(
  \mathcal Z_{t+1:H}(x'),
  \widehat{\mathcal Z}_{t+1:H}(\widehat x')
\right)
&\leq
W_1\!\left(
  \mathcal Z_{t+1:H}(x'),
  \widehat{\mathcal Z}_{t+1:H}(x')
\right)\\
&\quad+
W_1\!\left(
  \widehat{\mathcal Z}_{t+1:H}(x'),
  \widehat{\mathcal Z}_{t+1:H}(\widehat x')
\right)\\
&\leq e_{t+1}(x')+L_{t+1}d(x',\widehat x'),
\end{aligned}
\]

where the second inequality uses A11.  Couple the current resource sums by
adding \(c\) and \(\widehat c\) to coupled draws from these continuation laws.
The coupling characterization of \(W_1\) then yields

\[
e_t(x)
\leq
\mathbb E_{a,u}\mathbb E_{\gamma_{x,u}}
\left[
|c-\widehat c|
+L_{t+1}d(x',\widehat x')
+e_{t+1}(x')
\right].
\]

By A12,

\[
e_t(x)
\leq
\mathbb E_{a,u}[\epsilon_t(x,u)]
+\mathbb E_{x'\sim Q^{\pi,\Pi}_K(\cdot\mid x)}[e_{t+1}(x')].
\]

Apply this recurrence at \(t=0\), then repeatedly substitute the recurrence for
the last term.  The first marginal at each substitution is the true target
closed-loop kernel, so the resulting expectations are taken under the true
occupancies \(d_t^{\pi,\Pi,K}\).  Since \(e_H=0\), the final remainder vanishes,
giving the claimed sum.  ∎

### Corrections or Missing Assumptions

- The result does not prove that ensemble variance, density, or nearest-neighbor
  distance equals \(\epsilon_t\).
- Approximate policy or filter models require separately derived coupling terms.
- If continuation laws are discontinuous across filter active sets, A11 may fail;
  a piecewise or total-variation analysis is then required.

## Theorem 4: Proper-SSP Truncation Bound

### Claim

Under A5 and A7,

\[
W_1(\mathcal Z_C(x),\mathcal Z_{C,H}(x))
\leq
c_{\max}\mathbb E[(T_C-H)_+],
\]

and the right-hand side converges to zero as \(H\to\infty\).

### Status

**PROVABLE AS STATED**

### Proof

Couple \(Z_C\) and \(Z_{C,H}\) using the same realized trajectory.  By
nonnegativity and A5,

\[
0\leq Z_C-Z_{C,H}
=
\sum_{t=H}^{T_C-1}c_t
\leq c_{\max}(T_C-H)_+.
\]

The expected absolute difference under any coupling upper-bounds \(W_1\), so

\[
W_1(\mathcal Z_C(x),\mathcal Z_{C,H}(x))
\leq
\mathbb E[|Z_C-Z_{C,H}|]
\leq
c_{\max}\mathbb E[(T_C-H)_+].
\]

Because \(T_C\) is integrable, \((T_C-H)_+\downarrow0\) and is dominated by
\(T_C\).  Dominated convergence gives the stated limit.  ∎

## Theorem 5: Return-Boundary Decision Stability

### Claim

At a decision instant, define

\[
d^\star=\mathbf 1\{b\leq U+m\},
\qquad
\widehat d=\mathbf 1\{b\leq\widehat U+m\}.
\]

If \(|\widehat U-U|\leq\varepsilon\), then

\[
d^\star\neq\widehat d
\quad\Longrightarrow\quad
|b-m-U|\leq\varepsilon.
\]

### Status

**PROVABLE AS STATED**

### Proof

Let \(\theta=U+m\) and \(\widehat\theta=\widehat U+m\).  If the two indicators
differ, then \(b\) lies between \(\theta\) and \(\widehat\theta\), including the
appropriate threshold endpoint.  Therefore

\[
|b-\theta|
\leq
|\widehat\theta-\theta|
=
|\widehat U-U|
\leq\varepsilon.
\]

Since \(b-\theta=b-m-U\), the claimed implication follows.  ∎

### Interpretation Boundary

This theorem says where prediction error can alter the threshold decision.  It
does not say that every disagreement causes stranding, or that global MAE is
irrelevant to all other uses of the predictor.

## Corollary 1: Conditional Return-Failure Bound with Check Overshoot

### Claim

Suppose A13--A14 hold.  Let return checks occur at \(t-1\) and \(t\), and define
the true/calibrated feasibility margin \(D_s=b_s-U_s\).  Assume the one-way rule
commits at the first check satisfying \(b_s\leq U_s+m\).  If it did not commit at
\(t-1\), commits at \(t\), and \(m\geq\Delta_{\mathrm{check}}\), then
\(b_t>U_t\).  Conditional on prediction acceptance, and excluding non-energy
failure modes, the probability of energy exhaustion before reaching the charger
is at most \(\delta\).

### Status

**PROVABLE AFTER EXTRA ASSUMPTION**

### Proof

No commitment at \(t-1\) means

\[
D_{t-1}=b_{t-1}-U_{t-1}>m.
\]

By A14,

\[
(b_{t-1}-b_t)+(U_t-U_{t-1})
\leq\Delta_{\mathrm{check}}.
\]

Rearranging gives

\[
D_t
=D_{t-1}
-\big[(b_{t-1}-b_t)+(U_t-U_{t-1})\big]
>m-\Delta_{\mathrm{check}}
\geq0.
\]

Hence \(b_t>U_t\) at commitment.  A13 gives
\(\Pr(Z_{C,t}\leq U_t\mid\text{accepted at }t)\geq1-\delta\).  On the event
\(Z_{C,t}\leq U_t\), the strict inequality \(b_t>U_t\) implies
\(b_t>Z_{C,t}\), so energy exhaustion cannot occur before charger arrival even
under the current simulator convention that exact zero energy is exhaustion.
Therefore the conditional energy-exhaustion probability is at most \(\delta\).
∎

### Corrections or Missing Assumptions

- The statement is conditional on valid deployed-composition calibration.
- The check-interval bound must include both consumed energy and an increase in
  return requirement caused by motion, disturbances, or changed geometry.
- Collision safety remains the responsibility of the hard safety operator.

## Corollary 2: Task-Then-Return Risk Allocation

### Claim

Let task resource \(E_T\) and subsequent charger-return resource \(E_R\) be
defined on the same joint mission probability space.  If

\[
\Pr(E_T\leq U_T)\geq1-\delta_T,
\qquad
\Pr(E_R\leq U_R)\geq1-\delta_R,
\]

then, without assuming independence,

\[
\Pr(E_T+E_R\leq U_T+U_R)
\geq1-\delta_T-\delta_R.
\]

### Status

**PROVABLE AS STATED**

### Proof

If both component events occur, then their nonnegative upper bounds add:

\[
\{E_T\leq U_T\}\cap\{E_R\leq U_R\}
\subseteq
\{E_T+E_R\leq U_T+U_R\}.
\]

The union bound gives

\[
\begin{aligned}
\Pr(E_T+E_R>U_T+U_R)
&\leq
\Pr(E_T>U_T\ \text{or}\ E_R>U_R)\\
&\leq\delta_T+\delta_R.
\end{aligned}
\]

Taking complements proves the claim.  ∎

### Consequence

Adding two marginal q95 component bounds supports only a declared 90% lower
bound by this argument.  A 95% joint mission lower bound can instead use, for
example, two component bounds calibrated at 97.5%, or a directly modeled joint
mission distribution.

## Claim Not Yet Justified: Generic Quantile Stability

### Status

**NOT CURRENTLY JUSTIFIED**

A small \(W_1\) error does not by itself guarantee a small q95 or q99 error.  A
corrected quantile result needs a stronger CDF-distance bound and a local
anti-concentration condition such as a positive density lower bound around the
target quantile.  The present deterministic simulator can produce point-mass
conditional Resource-to-Go, so such a density assumption is not automatically
appropriate.  No quantile theorem should be claimed until the probability-
semantics audit determines the deployed stochastic object.

## Experiment-to-Theory Obligations

| Theory term | Required evidence | Current status |
| --- | --- | --- |
| Queryable \(\pi,K_\Pi\) | Exact frozen policy/filter provenance | Implemented in Stage B runners |
| Executed action interface | Log nominal and executed actions | Implemented; formal audit pending |
| Proper charger hitting | Charger reach and emergency-tail statistics | Pending formal Gate chain |
| Target interface coverage | Predeclared, non-label-based coverage proxy | Not implemented for Stage E |
| Local error \(\epsilon_t\) | Cross-fitted one-step predictive error | Not implemented; Stage C/F only |
| Hitting-tail term | Horizon-stratified \(T_C\) tail | Pending formal Oracle evaluation |
| Decision overshoot | Per-check consumption plus requirement drift | Implemented in switch and cycle telemetry; formal data pending |
| Calibration A13 | Accepted-set empirical calibration on target composition | Not established |
| Boundary law | Boundary-weighted late-return error versus mission failure | Stage D, pending Oracle Gate |

## Open Risks

- A11 may be false at discontinuous HOCBF active-set transitions.  A piecewise
  theorem or a total-variation formulation may be needed.
- A8 is an identification assumption, not an operationally observable support
  certificate.  The eventual reliability score must be evaluated empirically and
  must not be described as exact mathematical support.
- The Oracle headroom Gate may show that perfect Resource-to-Go information does
  not materially improve the stranding--throughput frontier.
- The current 500k navigation checkpoint may fail its prerequisite Gate.
- A generic executed-action probabilistic world model with a bootstrap ensemble
  may make a separate SIRP architecture unnecessary.
- The theorem package currently covers shared primitive dynamics only.  Wind,
  payload, battery aging, and actuator-model shifts require explicit context or a
  second identification analysis.
