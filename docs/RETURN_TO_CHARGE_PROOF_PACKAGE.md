# Proof Package: Reliable Return-to-Charge

## Scope

This package formalizes the theory obligations in
`docs/RETURN_TO_CHARGE_DERIVATION_PACKAGE.md`.  Its invariant object is the
resource required by the **executed** closed loop to hit a charger set.  The
package does not claim that the current deterministic simulator has a
non-degenerate aleatoric return distribution, and it does not turn a learned
energy model into a hard safety certificate.

The package proves five corrected base statements, two mission corollaries, nine
finite-state risk-resolvent statements, one sequential irreversible-stopping
theorem, one end-to-end certificate-to-headroom theorem, and one finite-interface
Bernstein-resolvent theorem with matching estimation and irreversible-decision
lower bounds, plus one fixed-linear risk-witness quotient theorem with an exact
identifiability/minimax dichotomy, one continuous local-mass
quotient--spectral--stopping phase theorem, and one exact risk-neutral
collapse/oracle-transform equivalence boundary, followed by an exact
unknown-Doob reparameterization/conditioning theorem and a predictable-interface
martingale quotient certificate, followed by a sharp stopped-margin localization
theorem from integrated \(L_p\) error and a cross-fitted stopped-occupation
quotient-regression theorem with an explicit pushed-forward overlap coefficient.
It is followed by a certified finite-library chart selector and a matched-
information impossibility theorem for algorithm-label separation. The main
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
8. Theorems 6--8 use the finite-state assumptions F1--F4 below and elementary
   positive-matrix/resolvent identities. Theorem 6B additionally uses a positive
   supersolution to reduce the risk resolvent to a discounted killed chain.
9. Theorem 9A uses an explicit pair of source-observationally-equivalent kernels;
   Theorem 9B uses a fully covered three-step chain to separate support from
   risk concentration; Theorem 9C applies a two-point total-variation lower
   bound to show exponential trajectory complexity despite common support.
10. Theorem 10 uses a simultaneous finite-grid upper log-MGF event, conditional
    independence of training and deployment randomness, and Markov's inequality.
11. Theorem 11 uses sample splitting, a clipped learned risk-occupation ratio,
    an \(L_1\) nuisance bound, a second-moment bound, and Bernstein's inequality.
12. Theorem 12 uses the corrected task--then--return stopping boundary, while
    treating direct-return feasibility as a separate post-commit certificate;
    it then applies oracle-shadow coupling up to first decision disagreement,
    a stopped boundary-occupation condition, and bounded cycle metrics.
13. Theorem 13 applies Theorem 11 simultaneously over a finite branch--state--risk
    query set, propagates positive MGF intervals through finite-grid EVaR, uses
    the task--then--return branch for commitment, and retains the direct-return
    branch as a safety certificate.
14. Theorem 14 estimates the primitive exponential kernel at predictably sampled
    executed interfaces, composes it through the queryable target policy/filter,
    applies a robust resolvent perturbation, and matches its exponential-risk and
    inverse-coverage order on an explicit two-model family.
15. Theorem 15 reduces the same indistinguishable pair to opposite exact-risk
    ReturnManager actions and then embeds the testing error into the two Pareto
    coordinates.

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
\qquad (x,u)\in B,\quad i\in\{1,2\}.
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

## Finite-State Executed-Interface Risk-Resolvent Extension

This section fixes a finite-state slice in which the exponential Resource-to-Go
claims can be proved without hiding functional-analytic assumptions.  The
general-state derivation remains in
`docs/RETURN_TO_CHARGE_DERIVATION_PACKAGE.md`; finite-state proofs do not by
themselves establish neural-network consistency or an oral-level contribution.

### Finite-state assumptions

- **F1 (finite killed closed loop).** The non-charger state set is
  \(S=\{1,\ldots,n\}\), and \(G_C\) is absorbing.  After composing the target
  policy and safety kernel, \(P_i(j,dc)\) is the joint kernel of next state
  \(j\in S\cup\{G_C\}\) and nonnegative step resource \(c\), starting from
  \(i\in S\).
- **F2 (proper hitting).** From every state considered,
  \(T_C<\infty\) almost surely.
- **F3 (exponential transience).** For every declared
  \(\lambda\in\Lambda\subset(0,\infty)\), the nonnegative matrix
  \[
  M_\lambda(i,j)=\int e^{\lambda c}P_i(j,dc),
  \qquad i,j\in S,
  \]
  has spectral radius \(\rho(M_\lambda)<1\).
- **F4 (learned resolvent).** When a learned operator is used,
  \(I-\widehat M_\lambda\) is invertible and
  \(\widehat\psi_\lambda=(I-\widehat M_\lambda)^{-1}\widehat r_\lambda\)
  is finite.
- **F5 (source domination for a positive error bound).** The target
  risk-occupation transition measure \(\bar\eta_{\nu,\lambda}\) defined below
  is absolutely continuous with respect to a declared source executed-transition
  measure \(\bar\rho_s\), with density
  \(\bar w_\lambda\in L_2(\bar\rho_s)\).
- **F6 (learned certificate protocol).** Training/calibration data \(D\) and a
  future deployment rollout are conditionally independent given the declared
  deployment information.  A finite-grid simultaneous upper log-MGF event holds
  with probability at least \(1-\alpha\), as stated in Theorem 10.

### Theorem 6: Killed Feynman--Kac resolvent representation

#### Claim

Define

\[
r_\lambda(i)=\int e^{\lambda c}P_i(G_C,dc).
\]

Under F1--F3, the exponential Resource-to-Go vector

\[
\psi_\lambda(i)=\mathbb E_i[e^{\lambda Z_C}]
\]

is finite and is the unique solution of

\[
\psi_\lambda=r_\lambda+M_\lambda\psi_\lambda.
\]

Equivalently,

\[
\boxed{
\psi_\lambda
=(I-M_\lambda)^{-1}r_\lambda
=\sum_{t=0}^{\infty}M_\lambda^t r_\lambda.}
\]

#### Status

**PROVABLE AS STATED UNDER F1--F3.**  This is a known Feynman--Kac/risk-sensitive
foundation, not the paper's novelty by itself.

#### Proof

For \(t\ge0\), repeated kernel multiplication shows

\[
(M_\lambda^t r_\lambda)(i)
=
\mathbb E_i\!\left[
e^{\lambda\sum_{k=0}^{t}c_k}
\mathbf 1\{T_C=t+1\}
\right].
\]

The identity follows by induction.  At \(t=0\), \(r_\lambda(i)\) is exactly the
weighted event of entering the charger after one transition.  Multiplication by
\(M_\lambda\) prepends one transient transition and its factor
\(e^{\lambda c_0}\), which proves the induction step.

The events \(\{T_C=t+1\}\) are disjoint and, by F2, their union has probability
one.  Since all summands are nonnegative, monotone convergence gives

\[
\mathbb E_i[e^{\lambda Z_C}]
=\sum_{t=0}^{\infty}(M_\lambda^t r_\lambda)(i).
\]

By F3 and finite dimensionality, the matrix Neumann series converges and equals
\((I-M_\lambda)^{-1}\).  The resulting vector is finite and satisfies the fixed
point equation.  If \(h\) is another finite solution, then
\((I-M_\lambda)(h-\psi_\lambda)=0\).  Invertibility of
\(I-M_\lambda\) gives \(h=\psi_\lambda\), proving uniqueness. ∎

### Theorem 6B: Exact Doob-scaled reduction to discounted occupancy

#### Claim

Fix \(\lambda\) and abbreviate \(M=M_\lambda\). Under F1--F3, choose any

\[
\gamma\in(\rho(M),1)
\]

and define the strictly positive vector

\[
v=\left(I-\frac{M}{\gamma}\right)^{-1}\mathbf 1
=\sum_{t=0}^{\infty}\left(\frac{M}{\gamma}\right)^t\mathbf1.
\]

Let \(D_v=\operatorname{diag}(v)\), introduce an artificial cemetery state
\(\dagger\), and define

\[
P_\gamma(i,j)
=\frac{M(i,j)v_j}{\gamma v_i},
\quad i,j\in S,
\qquad
P_\gamma(i,\dagger)=\frac1{v_i}.
\]

With \(P_{\gamma,S}\) denoting the \(S\times S\) block, \(P_\gamma\) is a
Markov kernel after making \(\dagger\) absorbing, and

\[
\boxed{
M=\gamma D_vP_{\gamma,S}D_v^{-1}.}
\]

Consequently,

\[
\boxed{
(I-M)^{-1}
=D_v(I-\gamma P_{\gamma,S})^{-1}D_v^{-1},}
\]

and

\[
\boxed{
\psi_\lambda
=D_v(I-\gamma P_{\gamma,S})^{-1}D_v^{-1}r_\lambda.}
\]

If \(\nu\) is an initial probability row vector, set

\[
c_\nu=\nu v,
\qquad
\widetilde\nu_i=\frac{\nu_i v_i}{c_\nu},
\]

and let the usual discounted occupancy restricted to \(S\) be

\[
d_{\widetilde\nu,\gamma}^S
=(1-\gamma)\widetilde\nu
(I-\gamma P_{\gamma,S})^{-1}.
\]

Then the risk state occupation is exactly

\[
\boxed{
\eta^X_{\nu,\lambda}(j)
=\frac{c_\nu}{1-\gamma}
\frac{d_{\widetilde\nu,\gamma}^S(j)}{v_j}.}
\]

#### Status

**PROVABLE AS STATED.** This is a finite-dimensional positive-operator/Doob
scaling, not a standalone novelty claim. Its purpose is a reduction audit: it
identifies exactly how much of EIRR is algebraically a discounted occupancy-ratio
problem and therefore must be compared with DICE methods.

#### Proof

Because \(\rho(M/\gamma)<1\), the Neumann series defining \(v\) converges,
is finite, and satisfies \(v\ge\mathbf1>0\). Moreover,

\[
Mv=\gamma(v-\mathbf1).
\]

Therefore each row of the proposed kernel sums to one:

\[
\sum_{j\in S}P_\gamma(i,j)+P_\gamma(i,\dagger)
=\frac{(Mv)_i}{\gamma v_i}+\frac1{v_i}
=\frac{v_i-1}{v_i}+\frac1{v_i}=1.
\]

The entrywise definition immediately gives
\(M=\gamma D_vP_{\gamma,S}D_v^{-1}\). Hence for every \(t\ge0\),

\[
M^t=D_v(\gamma P_{\gamma,S})^tD_v^{-1}.
\]

Summing the convergent Neumann series proves the resolvent identity, and
multiplication by \(r_\lambda\) proves the value identity. Finally,

\[
\begin{aligned}
\eta^X_{\nu,\lambda}
&=\nu D_v(I-\gamma P_{\gamma,S})^{-1}D_v^{-1}\\
&=\frac{c_\nu}{1-\gamma}
d_{\widetilde\nu,\gamma}^S D_v^{-1},
\end{aligned}
\]

which is the componentwise occupation formula. ∎

#### Interpretation boundary

The reduction does not make the learning problem disappear. Both \(v\) and the
Doob-scaled target kernel depend on the unknown target Feynman--Kac operator;
source data do not directly sample \(P_\gamma\). It does show that a proposed
ratio estimator cannot be called new merely for replacing the ordinary Bellman
adjoint with \(M_\lambda\): DualDICE/GenDICE-style estimators on this transformed
chain are mandatory baselines. Any remaining novelty must come from estimating
the transform under queryable policy--filter composition, first-passage killing,
or carrying its certified error to the irreversible return decision.

### Theorem 7: Exact risk-occupation residual representation

#### Claim

Let

\[
R_\lambda=(I-M_\lambda)^{-1},
\qquad
\eta^X_{\nu,\lambda}=\nu R_\lambda
\]

for an initial row distribution \(\nu\) on \(S\).  For any finite critic
\(h:S\to\mathbb R\), let

\[
(\mathcal T_\lambda h)(i)=r_\lambda(i)+(M_\lambda h)(i).
\]

Then

\[
\boxed{
\psi_\lambda-h
=R_\lambda(\mathcal T_\lambda h-h).}
\]

Disintegrate each risk-weighted state visit through the target policy, safety
kernel, and primitive kernel to obtain the finite executed-transition measure

\[
\begin{aligned}
\bar\eta_{\nu,\lambda}(di,da,du,dj,dc)
={}&\eta^X_{\nu,\lambda}(di)\,\pi(da\mid i)\\
&K_\Pi(du\mid i,a)K(dj,dc\mid i,u).
\end{aligned}
\]

For

\[
\epsilon_{\lambda,h}(i,a,u,j,c)
=e^{\lambda c}
\left[
\mathbf1\{j=G_C\}+\mathbf1\{j\in S\}h(j)
\right]-h(i),
\]

the initial-value error has the exact representation

\[
\boxed{
\nu(\psi_\lambda-h)
=\int\epsilon_{\lambda,h}\,d\bar\eta_{\nu,\lambda}.}
\]

Under F5,

\[
\boxed{
|\nu(\psi_\lambda-h)|
\le
\|\bar w_\lambda\|_{L_2(\bar\rho_s)}
\|\epsilon_{\lambda,h}\|_{L_2(\bar\rho_s)}.}
\]

#### Status

**PROVABLE AS STATED UNDER F1--F3 AND F5 FOR THE FINAL INEQUALITY.**

#### Proof

The fixed-point equation from Theorem 6 gives

\[
\begin{aligned}
(I-M_\lambda)(\psi_\lambda-h)
&=r_\lambda-(I-M_\lambda)h\\
&=r_\lambda+M_\lambda h-h\\
&=\mathcal T_\lambda h-h.
\end{aligned}
\]

Left multiplication by \(R_\lambda\) proves the first identity.  Multiplying by
\(\nu\) gives

\[
\nu(\psi_\lambda-h)
=\eta^X_{\nu,\lambda}(\mathcal T_\lambda h-h).
\]

Expanding the conditional expectation in \(\mathcal T_\lambda h-h\) and then
using the definition of \(\bar\eta_{\nu,\lambda}\) proves the integral identity.
Under F5, change measure using
\(d\bar\eta_{\nu,\lambda}=\bar w_\lambda d\bar\rho_s\), take absolute values,
and apply Cauchy--Schwarz. ∎

#### Interpretation boundary

The theorem prescribes the population weighting of Bellman residuals.  It does
not prove that an estimated density ratio or neural critic is consistent, and
the coefficient may be too large for useful finite-sample learning.

### Theorem 8: Exact learned-model perturbation

#### Claim

Under F1--F4,

\[
\boxed{
\psi_\lambda-\widehat\psi_\lambda
=R_\lambda
\left[
(r_\lambda-\widehat r_\lambda)
+(M_\lambda-\widehat M_\lambda)\widehat\psi_\lambda
\right].}
\]

For every subordinate matrix/vector norm,

\[
\|\psi_\lambda-\widehat\psi_\lambda\|
\le
\|R_\lambda\|
\left(
\|r_\lambda-\widehat r_\lambda\|
+\|M_\lambda-\widehat M_\lambda\|
\|\widehat\psi_\lambda\|
\right).
\]

#### Status

**PROVABLE AS STATED.**  The identity is elementary; novelty cannot rest on this
line alone.

#### Proof

Subtract the learned fixed-point equation from the target equation and add and
subtract \(M_\lambda\widehat\psi_\lambda\):

\[
\begin{aligned}
\psi_\lambda-\widehat\psi_\lambda
={}&r_\lambda-\widehat r_\lambda
+M_\lambda\psi_\lambda
-\widehat M_\lambda\widehat\psi_\lambda\\
={}&r_\lambda-\widehat r_\lambda
+M_\lambda(\psi_\lambda-\widehat\psi_\lambda)
+(M_\lambda-\widehat M_\lambda)\widehat\psi_\lambda.
\end{aligned}
\]

Move the middle term to the left and multiply by \(R_\lambda\).  The norm bound
follows from the triangle inequality and submultiplicativity. ∎

### Theorem 9A: Source-only minimax lower bound without interface support

#### Claim

Suppose two primitive kernels \(K_0,K_1\) induce exactly the same distribution
of every possible source dataset \(D\), while their target exponential
Resource-to-Go values are \(\theta_i=\nu\psi_{\lambda,i}\) with
\(\theta_0\ne\theta_1\).  Then every source-only estimator
\(\widehat\theta(D)\) satisfies

\[
\boxed{
\max_{i\in\{0,1\}}
\mathbb E_i|\widehat\theta-\theta_i|
\ge\frac{|\theta_1-\theta_0|}{2}.}
\]

The same statement holds for estimating \(\log\theta_i\), with separation
\(|\log\theta_1-\log\theta_0|/2\).

#### Explicit construction

Use transient states \(s,b\) and charger \(G_C\).  A source-only executed action
at \(s\) moves directly to \(G_C\) with zero resource, so the source never visits
the interface at \(b\).  Under the target composition, the transition from
\(s\) goes to \(b\) with probability \(p>0\) and to \(G_C\) otherwise, with
zero current resource.  At \(b\), kernel \(K_i\) moves to \(G_C\) with resource
\(c_i\in[0,c_{\max}]\), where \(c_0\ne c_1\).  The complete source laws are
identical, but

\[
\theta_i=1-p+pe^{\lambda c_i},
\]

so

\[
|\theta_1-\theta_0|
=p|e^{\lambda c_1}-e^{\lambda c_0}|.
\]

#### Status

**PROVABLE AS STATED.**

#### Proof

Because the source-data laws coincide, the random output
\(Y=\widehat\theta(D)\) has one common distribution under both models.  Pointwise
in \(Y\), the triangle inequality gives

\[
|Y-\theta_0|+|Y-\theta_1|\ge|\theta_1-\theta_0|.
\]

Taking expectation under the common output law shows that the sum of the two
risks is at least the separation, so their maximum is at least half the
separation.  Applying the same argument to \(\log\theta_i\) proves the log-MGF
version.  The displayed target value in the construction follows by conditioning
on whether \(b\) is reached. ∎

#### Consequence

The lower bound is independent of source sample size.  More source trajectories
cannot resolve an interface that the source law assigns zero probability.  The
exponential separation also makes the tail functional more sensitive than the
mean separation \(p|c_1-c_0|\), but this does not make exponential risk itself a
new concept.

### Theorem 9B: Perfect support does not control risk concentration

#### Claim and construction

There exists a finite proper chain for which source data cover every target
interface and the ordinary source and target occupation measures are identical,
yet the risk-occupation/source density ratio grows exponentially with a prefix
resource.

Let \(s\) transition to \(h\) with probability \(p\in(0,1)\) and to \(\ell\)
otherwise, at zero resource.  From \(h\), move to a merge state \(b\) with
resource \(L>0\); from \(\ell\), move to \(b\) with zero resource.  Finally,
\(b\) moves to \(G_C\) with zero resource.  Use this same chain for source and
target, and let \(\bar\rho_s\) be its ordinary unnormalized transition occupation
measure.  Then \(\bar\rho_s\) gives mass one to the transition out of \(b\),
whereas the risk-occupation measure gives that transition mass

\[
\bar\eta_{\nu,\lambda}(b)=1-p+pe^{\lambda L}.
\]

Thus

\[
\boxed{
\bar w_\lambda(b)=1-p+pe^{\lambda L},}
\]

which diverges as \(\lambda L\to\infty\), even though ordinary support and
ordinary occupancy match perfectly.

#### Status

**PROVABLE AS STATED.**

#### Proof

Every path visits \(b\), so its ordinary occupation mass is one.  A path that
reaches \(b\) through \(\ell\) has accumulated prefix resource zero and
contributes weight one.  A path reaching \(b\) through \(h\) has accumulated
prefix resource \(L\) and contributes \(e^{\lambda L}\).  Averaging over the two
paths yields \(1-p+pe^{\lambda L}\).  Dividing by the ordinary mass one gives the
stated density ratio. ∎

#### Interpretation boundary

Support controls identification, not statistical stability.  The example does
not show policy/filter shift; it deliberately shows that rare high-resource
prefixes can make risk learning difficult even on-policy.  Composition shift can
add a separate density-ratio burden.

### Theorem 9C: Common support can still require exponentially many trajectories

#### Claim

Fix \(L>0\) and \(\lambda>0\) with \(q=e^{-\lambda L}\le 1/4\).  Consider the
two proper one-step charger-hitting models

\[
P_i(Z_C=L)=p_i,
\qquad
P_i(Z_C=0)=1-p_i,
\qquad
p_0=q,\quad p_1=2q.
\]

Both models have exactly the same ordinary support \(\{0,L\}\).  Their
exponential values are

\[
\theta_0=\mathbb E_0[e^{\lambda Z_C}]=2-q,
\qquad
\theta_1=\mathbb E_1[e^{\lambda Z_C}]=3-2q.
\]

For any estimator \(\widehat\theta\) based on \(n\) independent trajectories,
if

\[
n\le \frac{e^{\lambda L}}{4},
\]

then

\[
\boxed{
\max_{i\in\{0,1\}}
\mathbb E_i|\widehat\theta-\theta_i|
\ge \frac{9}{32}.}
\]

For any estimator \(\widehat\ell\) of the log-MGF
\(\ell_i=\log\theta_i\), the same sample-size condition implies

\[
\boxed{
\max_{i\in\{0,1\}}
\mathbb E_i|\widehat\ell-\ell_i|
\ge \frac{3}{8}\log\frac{10}{7}.}
\]

Consequently, uniform constant-accuracy estimation over this common-support
family has trajectory complexity \(\Omega(e^{\lambda L})\).

#### Status

**PROVABLE AS STATED.**  The generic two-point method is classical; the
contribution candidate is its executed-interface, charger-hitting interpretation
and its connection to risk-occupation concentration and stopping reliability.

#### Proof

Couple one Bernoulli observation under \(P_0\) and \(P_1\) so that they disagree
only with probability \(p_1-p_0=q\).  Coupling the \(n\) coordinates
independently and applying the union bound gives

\[
\operatorname{TV}(P_0^n,P_1^n)\le nq\le\frac14.
\]

For any two scalar targets \(a_0,a_1\) and estimator \(A\), integration against
the common part of \(P_0^n\) and \(P_1^n\) gives the two-point inequality

\[
\max_i\mathbb E_i|A-a_i|
\ge
\frac{|a_1-a_0|}{2}
\left(1-\operatorname{TV}(P_0^n,P_1^n)\right).
\]

Here

\[
|\theta_1-\theta_0|=1-q\ge\frac34,
\]

so the first lower bound is at least
\((3/4)(3/4)/2=9/32\).  Moreover,

\[
\frac{\theta_1}{\theta_0}
=\frac{3-2q}{2-q}
\ge\frac{10}{7}
\qquad(q\le1/4),
\]

and applying the same inequality to \(a_i=\log\theta_i\) yields the second
bound.  Since \(1/q=e^{\lambda L}\), keeping either minimax risk below the
displayed constant requires a number of independent trajectories proportional
to \(e^{\lambda L}\). ∎

#### Interpretation boundary

Common support is enough for population identification but not finite-sample
learnability of exponential risk.  The lower bound concerns independent
trajectories and a nonparametric two-model class; additional structural knowledge
can reduce the complexity and must be declared explicitly.

### Theorem 10: Simultaneous learned log-MGF bound implies return chance bound

#### Claim

Let \(\Lambda\subset(0,\infty)\) be finite.  Suppose F6 holds and, for a fixed
deployment information state \(x\), the data-dependent functions
\(\widehat L_\lambda^+(x;D)\) satisfy

\[
\Pr_D\!\left(
\forall\lambda\in\Lambda:
\log\psi_\lambda(x)
\le\widehat L_\lambda^+(x;D)
\right)
\ge1-\alpha.
\]

Define

\[
\widehat\lambda(D,x)
\in\arg\min_{\lambda\in\Lambda}
\frac{\widehat L_\lambda^+(x;D)+\log(1/\delta)}{\lambda}
\]

and

\[
\widehat U_{\delta,\alpha}(x;D)
=
\frac{
\widehat L_{\widehat\lambda}^+(x;D)+\log(1/\delta)
}{\widehat\lambda}.
\]

Then a fresh deployment rollout satisfies

\[
\boxed{
\Pr_{D,Z_C}\!\left(
Z_C>\widehat U_{\delta,\alpha}(x;D)
\right)
\le\delta+\alpha.}
\]

#### Status

**PROVABLE AS STATED UNDER F6.**  Chernoff/EVaR is known; the substantive future
work is constructing the simultaneous upper log-MGF event under composition
shift.

#### Proof

Let \(E_D\) be the simultaneous upper-bound event.  Conditional on any dataset
\(D\in E_D\), the selected \(\widehat\lambda\) is fixed before the independent
deployment rollout.  Markov's inequality gives

\[
\begin{aligned}
\Pr\!\left(
Z_C>\widehat U_{\delta,\alpha}\mid D
\right)
&\le
e^{-\widehat\lambda\widehat U_{\delta,\alpha}}
\mathbb E[e^{\widehat\lambda Z_C}\mid x]\\
&=
\delta\exp\!\left(
\log\psi_{\widehat\lambda}(x)
-\widehat L_{\widehat\lambda}^+(x;D)
\right)\\
&\le\delta.
\end{aligned}
\]

On \(E_D^c\), bound the failure probability by one.  Averaging over \(D\)
yields

\[
\Pr(Z_C>\widehat U_{\delta,\alpha})
\le\delta\Pr(E_D)+\Pr(E_D^c)
\le\delta+\alpha.
\]

This also proves that data-dependent selection over the finite \(\lambda\) grid
is valid when the upper event is simultaneous.  Marginal per-\(\lambda\) bounds
alone require a multiple-comparison correction. ∎

### Theorem 11: Cross-fitted clipped risk-residual certificate

#### Claim

Let the source executed-transition sampling law \(\bar\rho_s\) in Theorem 7 be
a probability measure, and write

\[
w_\lambda=\frac{d\bar\eta_{\nu,\lambda}}{d\bar\rho_s}.
\]

Use a training fold to choose a critic \(h\) and a nonnegative ratio estimator
\(\widehat w_\lambda\).  Suppose that, on a training-data event \(E_{\rm tr}\)
of probability at least \(1-\beta\), these now-fixed functions satisfy

\[
|\epsilon_{\lambda,h}|\le B,
\qquad
\|\widehat w_\lambda-w_\lambda\|_{L_1(\bar\rho_s)}\le\xi,
\qquad
\mathbb E_{\bar\rho_s}[\widehat w_\lambda^2]\le C_2.
\]

Let \(X_1,\ldots,X_n\) be an independent calibration fold sampled i.i.d. from
\(\bar\rho_s\), and define

\[
\widetilde w_{\lambda,\tau}=\min\{\widehat w_\lambda,\tau\},
\qquad \tau>0.
\]

Then, with joint probability at least \(1-\alpha-\beta\),

\[
\boxed{
\begin{aligned}
|\nu(\psi_\lambda-h)|
\le{}&
\left|
\frac1n\sum_{k=1}^n
\widetilde w_{\lambda,\tau}(X_k)
\epsilon_{\lambda,h}(X_k)
\right|
+B\xi+\frac{BC_2}{\tau}\\
&+B\sqrt{\frac{2C_2\log(2/\alpha)}{n}}
+\frac{2B\tau\log(2/\alpha)}{3n}.
\end{aligned}}
\]

Choosing

\[
\tau_\star
=\sqrt{\frac{3C_2n}{2\log(2/\alpha)}}
\]

makes the sum of the clipping-bias and range terms equal

\[
2B\sqrt{\frac{2C_2\log(2/\alpha)}{3n}},
\]

so, apart from the nuisance error \(B\xi\) and the observed weighted residual,
the certificate shrinks at the parametric \(n^{-1/2}\) rate whenever \(C_2\) is
uniformly controlled.

#### Status

**PROVABLE AS STATED UNDER THE DECLARED SAMPLE-SPLITTING ASSUMPTIONS.**  Clipped
importance weighting and Bernstein bounds are standard.  The theorem's role is
to turn the exact executed-interface risk-resolvent identity into an auditable
loss/certificate and to expose the precise point at which Theorems 9B--9C make
the guarantee vacuous.

#### Proof

Condition on a training fold in \(E_{\rm tr}\).  Theorem 7 and the triangle
inequality give

\[
\begin{aligned}
|\nu(\psi_\lambda-h)|
&=|\mathbb E_{\bar\rho_s}[w_\lambda\epsilon_{\lambda,h}]|\\
&\le
|\mathbb E_{\bar\rho_s}[\widetilde w_{\lambda,\tau}
\epsilon_{\lambda,h}]|
+B\mathbb E_{\bar\rho_s}
|w_\lambda-\widetilde w_{\lambda,\tau}|.
\end{aligned}
\]

Because \(\widehat w_\lambda\ge0\),

\[
\begin{aligned}
\mathbb E|w_\lambda-\widetilde w_{\lambda,\tau}|
&\le \mathbb E|w_\lambda-\widehat w_\lambda|
+\mathbb E(\widehat w_\lambda-\tau)_+\\
&\le \xi+\frac{\mathbb E[\widehat w_\lambda^2]}{\tau}
\le \xi+\frac{C_2}{\tau},
\end{aligned}
\]

where \((z-\tau)_+\le z^2/\tau\) for \(z\ge0\).  On the independent calibration
fold, set

\[
Y_k=\widetilde w_{\lambda,\tau}(X_k)
\epsilon_{\lambda,h}(X_k).
\]

Then \(|Y_k|\le B\tau\),
\(\mathbb E[Y_k^2]\le B^2C_2\), and
\(|Y_k-\mathbb E Y_k|\le2B\tau\).  Bernstein's inequality therefore gives,
with conditional probability at least \(1-\alpha\),

\[
\left|
\mathbb E Y_k-\frac1n\sum_{k=1}^nY_k
\right|
\le
B\sqrt{\frac{2C_2\log(2/\alpha)}{n}}
+\frac{2B\tau\log(2/\alpha)}{3n}.
\]

Combining the displays proves the certificate.  Removing the conditioning costs
at most \(\beta\) by a union bound.  Finally, direct minimization of
\(C_2/\tau+2\tau\log(2/\alpha)/(3n)\) gives \(\tau_\star\) and the stated
rate. ∎

#### Interpretation boundary

The certificate is invalid if the same calibration fold is used to tune an
unrestricted critic, ratio, or \(\lambda\) without a simultaneous correction.
Trajectory dependence also cannot be hidden: i.i.d. transition sampling must be
implemented by a declared replay-sampling law or replaced by a trajectory-level
martingale/mixing bound.  A small empirical residual alone is not a certificate
when \(\xi\) or \(C_2\) is uncontrolled.

## Theorem 12: Irreversible first-disagreement and Pareto stability

### Claim

At each task-mode decision epoch define the effective exact and learned
**commitment** requirements

\[
U_t^{\rm eff}=U_t^{\rm task+return},
\qquad
\widehat U_t^{\rm eff}=\widehat U_t^{\rm task+return},
\]

and scores

\[
s_t=b_t-m-U_t^{\rm eff},
\qquad
\widehat s_t=b_t-m-\widehat U_t^{\rm eff}.
\]

Both oracle and learned rules commit to the charger exactly when their mission
score is nonpositive. Their direct-return requirements separately label the
commitment as certified or emergency; those requirements do not define the
binary stopping action when the hybrid feasible sets are non-nested. Charger
commitment is absorbing until cycle termination. Couple
the two closed loops using common primitive randomness while their decisions
agree, and let

\[
\tau_\Delta
=\inf\{t<\tau_{\rm end}:d_t^\star\ne\widehat d_t\}
\]

be their first disagreement. Define the **oracle-shadow law** \(\mathbb P_\star\)
by running the oracle rule through \(\tau_{\rm end}\) while evaluating the fixed
learned estimator on every oracle history. Suppose the simultaneous event

\[
G=\left\{
|\widehat U_t^{\rm eff}-U_t^{\rm eff}|\le\varepsilon_t
\text{ for every oracle-shadow }t<\tau_{\rm end}
\right\}
\]

satisfies \(\Pr(G^c)\le\alpha\). Then

\[
\boxed{
\Pr(\tau_\Delta<\tau_{\rm end})
\le
\alpha+
\mathbb E_\star\!\left[
\sum_{t<\tau_{\rm end}}
\mathbf1\{|s_t|\le\varepsilon_t\}
\right].}
\tag{12.1}
\]

If the stopped boundary-occupation condition

\[
\mathbb P_\star(t<\tau_{\rm end},|s_t|\le r)
\le C_t r^\kappa
\]

holds for every declared \(r\), then

\[
\boxed{
\Pr(\tau_\Delta<\tau_{\rm end})
\le
\alpha+\sum_t C_t\varepsilon_t^\kappa.}
\tag{12.2}
\]

Let \(F\in[0,B_F]\) be any bounded cycle metric computed under the common-random-
number coupling, with identical value whenever no decision disagreement occurs.
Then

\[
\boxed{
|\mathbb E F^{\widehat d}-\mathbb E F^{d^\star}|
\le
B_F\Pr(\tau_\Delta<\tau_{\rm end}).}
\tag{12.3}
\]

In particular, writing \(p\) for stranding rate and \(q\) for a throughput
metric bounded by \(B_q\), and defining the right-hand side of (12.1), capped at
one, as \(\zeta_\Delta\),

\[
\boxed{
p_{\widehat d}\le p_\star+\zeta_\Delta,
\qquad
q_{\widehat d}\ge q_\star-B_q\zeta_\Delta.}
\tag{12.4}
\]

Therefore, for population stranding ceiling \(\bar p\), best eligible heuristic
throughput \(q_H\), and required gain \(g\), the sufficient conditions

\[
\boxed{
p_\star+\zeta_\Delta\le\bar p,
\qquad
q_\star-B_q\zeta_\Delta\ge(1+g)q_H}
\tag{12.5}
\]

guarantee that the learned rule preserves the population Oracle-headroom
criterion.

Finally, let cycle loss lie in \([0,B_L]\). If on \(G\), conditional on first
disagreement at \(t\), its positive excess obeys

\[
\mathbb E[(L^{\widehat d}-L^{d^\star})_+
\mid\mathcal F_t,\tau_\Delta=t,G]
\le L_D|s_t|,
\]

then

\[
\boxed{
\mathbb E[(L^{\widehat d}-L^{d^\star})_+]
\le
B_L\alpha
+L_D\sum_t
\varepsilon_t
\mathbb P_\star(t<\tau_{\rm end},|s_t|\le\varepsilon_t).}
\tag{12.6}
\]

Under the stopped margin condition this becomes

\[
\mathbb E[(L^{\widehat d}-L^{d^\star})_+]
\le B_L\alpha+L_D\sum_tC_t\varepsilon_t^{\kappa+1}.
\tag{12.7}
\]

### Status

**PROVABLE AS STATED UNDER THE DECLARED COUPLING, SIMULTANEOUS ERROR, STOPPED
MARGIN, AND BOUNDED-METRIC ASSUMPTIONS.** The coupling and margin techniques are
standard. The useful project-specific content is the separation of the mission
stopping boundary from the direct-return certificate, the oracle-shadow stopped
law, and the population
uncertainty rectangle for the preregistered stranding--throughput Gate.

### Proof

The implementation continues exactly when the task--then--return macro action is
certified. This is equivalent to

\[
b_t-m-U_t^{\rm task+return}\le0,
\]

so the effective-score rule exactly represents the implemented commitment
decision. Direct-return feasibility is audited separately.

On \(G\), if first disagreement occurs at \(t\), the coupled histories through
that decision epoch are identical and equal to the oracle-shadow history. By
Theorem 5, disagreement implies \(|s_t|\le\varepsilon_t\). Hence

\[
\{\tau_\Delta<\tau_{\rm end}\}\cap G
\subseteq
\bigcup_{t<\tau_{\rm end}}\{|s_t|\le\varepsilon_t\}
\]

under \(\mathbb P_\star\). Splitting on \(G\), applying the union bound, and
writing the random union bound as a stopped occupation expectation proves
(12.1). Applying the per-epoch margin condition proves (12.2).

If no first disagreement occurs, both one-way rules generate the same complete
cycle under common primitive randomness, so \(F^{\widehat d}=F^{d^\star}\).
Otherwise their absolute difference is at most \(B_F\). Thus

\[
|\mathbb EF^{\widehat d}-\mathbb EF^{d^\star}|
\le\mathbb E|F^{\widehat d}-F^{d^\star}|
\le B_F\Pr(\tau_\Delta<\tau_{\rm end}),
\]

which proves (12.3). Choosing the stranding indicator gives its unit bound;
choosing bounded throughput gives the second coordinate, proving (12.4).
Substitution into the population Gate definition proves (12.5).

On \(G^c\), bound positive excess loss by \(B_L\). On \(G\), split by the
disjoint events \(\{\tau_\Delta=t\}\), apply the conditional boundary-loss
assumption, use \(|s_t|\le\varepsilon_t\), and enlarge each first-disagreement
event to the corresponding oracle-shadow boundary event. This gives (12.6), and
the stopped margin condition gives (12.7). ∎

### Interpretation boundary

- Equations (12.4)--(12.5) are population statements. Formal experiments still
  require cycle-level uncertainty, including the preregistered Wilson upper
  stranding statistic.
- Throughput must have an explicit protocol bound \(B_q\); an unbounded ratio is
  not covered.
- The simultaneous event must be valid along oracle-shadow histories. Marginal
  per-state calibration is insufficient without a sequential correction.
- The theorem does not create Oracle headroom. It only quantifies how much
  already-established headroom can be lost to return-boundary disagreement.
- Decision-interval overshoot remains governed by Corollary 1 and must be added
  to the reserve/calibration contract; it is not hidden inside \(\varepsilon_t\).

## Theorem 13: Cross-fitted EIRR certificate to conservative headroom

### Claim

Let \(k\in\mathcal K=\{R,M\}\) index the direct return and direct joint
task--then--return branches.  For each branch use a finite Markov information
state set \(S_k\), a common finite grid \(\Lambda\subset(0,\infty)\), and tail
budget \(\delta_k\in(0,1)\).  The state must include every variable needed to
make that branch Markov.  Define

\[
\psi_{k,\lambda}(x)
=\mathbb E_x[e^{\lambda Z_k}],
\qquad
Q=|\Lambda|\sum_{k\in\mathcal K}|S_k|.
\]

For every query \(q=(k,x,\lambda)\), apply Theorem 11 with
\(\nu=e_x\), a fixed critic \(h_{k,\lambda}\), its corresponding target
risk-occupation ratio, and calibration failure allocation
\(\alpha_{m cal}/Q\).  Suppose the training-fold nuisance conditions hold
**simultaneously** for all \(q\) on an event of probability at least
\(1-\beta\).  Writing \(B_{\epsilon,q},\xi_q,C_{2,q},\tau_q,n_q\) for the
quantities in Theorem 11 and \(X_{q,i}\) for its independent calibration sample,
define

\[
\begin{aligned}
c_q={}&
\left|\frac1{n_q}\sum_{i=1}^{n_q}
\widetilde w_{q,\tau_q}(X_{q,i})
\epsilon_q(X_{q,i})\right|
+B_{\epsilon,q}\xi_q+\frac{B_{\epsilon,q}C_{2,q}}{\tau_q}\\
&+B_{\epsilon,q}\sqrt{\frac{2C_{2,q}\log(2Q/\alpha_{\rm cal})}{n_q}}
+\frac{2B_{\epsilon,q}\tau_q\log(2Q/\alpha_{\rm cal})}{3n_q}.
\end{aligned}
\tag{13.1}
\]

Then, with probability at least
\(1-\alpha_E\), where
\(\alpha_E=\beta+\alpha_{\rm cal}\), simultaneously for all queries,

\[
\boxed{
|\psi_{k,\lambda}(x)-h_{k,\lambda}(x)|\le c_{k,x,\lambda}.}
\tag{13.2}
\]

Assume the certificate is positive, meaning
\(h_{k,\lambda}(x)-c_{k,x,\lambda}>0\) for every query.  Define

\[
\underline\psi_{k,\lambda}(x)=h_{k,\lambda}(x)-c_{k,x,\lambda},
\qquad
\overline\psi_{k,\lambda}(x)=h_{k,\lambda}(x)+c_{k,x,\lambda},
\]

and the lower, exact, and learned-upper finite-grid EVaR requirements

\[
\begin{aligned}
\underline U_k(x)
&=\min_{\lambda\in\Lambda}
\frac{\log\underline\psi_{k,\lambda}(x)+\log(1/\delta_k)}{\lambda},\\
U_k^\star(x)
&=\min_{\lambda\in\Lambda}
\frac{\log\psi_{k,\lambda}(x)+\log(1/\delta_k)}{\lambda},\\
\widehat U_k^+(x)
&=\min_{\lambda\in\Lambda}
\frac{\log\overline\psi_{k,\lambda}(x)+\log(1/\delta_k)}{\lambda}.
\end{aligned}
\tag{13.3}
\]

Propagate the actual ReturnManager commitment semantics using the mission branch:

\[
\underline U^{\rm eff}(x)=\underline U_M(x),
\quad
U^{\star,{\rm eff}}(x)=U_M^\star(x),
\quad
\widehat U^{+,{\rm eff}}(x)=\widehat U_M^+(x),
\]

and set

\[
\varepsilon(x)
=\widehat U^{+,{\rm eff}}(x)-\underline U^{\rm eff}(x).
\]

On the simultaneous event (13.2),

\[
\boxed{
0\le
\widehat U^{+,{\rm eff}}(x)-U^{\star,{\rm eff}}(x)
\le\varepsilon(x).}
\tag{13.4}
\]

Thus the learned certified manager is conservative relative to the exact-risk
manager.  Couple their irreversible decisions as in Theorem 12 and let

\[
s_t^\star=b_t-m-U^{\star,{\rm eff}}(x_t).
\]

The learned manager cannot make a later first commitment on the certificate
event.  Its first-disagreement probability satisfies the sharper one-sided
bound

\[
\boxed{
\Pr(\tau_\Delta<\tau_{\rm end})
\le
\alpha_E+
\mathbb E_\star\!\left[
\sum_{t<\tau_{\rm end}}
\mathbf1\{0<s_t^\star\le\varepsilon(x_t)\}
\right].}
\tag{13.5}
\]

Moreover, the probability of a later first commitment is at most \(\alpha_E\).
If

\[
\mathbb P_\star(t<\tau_{\rm end},0<s_t^\star\le r)
\le C_t r^\kappa
\]

and \(\bar\varepsilon_t\) is a deterministic upper bound on
\(\varepsilon(x_t)\) over oracle-shadow states reachable at epoch \(t\), then

\[
\boxed{
\zeta_E
=\min\!\left\{1,
\alpha_E+\sum_t C_t\bar\varepsilon_t^\kappa
\right\}}
\tag{13.6}
\]

bounds first disagreement.  Consequently, for stranding rate \(p\) and a
throughput metric \(q\in[0,B_{\rm thr}]\),

\[
\boxed{
p_{\widehat d}\le p_\star+\zeta_E,
\qquad
q_{\widehat d}\ge q_\star-B_{\rm thr}\zeta_E,}
\tag{13.7}
\]

and the population Oracle-headroom criterion is preserved whenever

\[
\boxed{
p_\star+\zeta_E\le\bar p,
\qquad
q_\star-B_{\rm thr}\zeta_E\ge(1+g)q_H.}
\tag{13.8}
\]

Finally, suppose uniformly that \(h_q\ge h_{\min}>0\),
\(c_q\le a_n\le h_{\min}/2\), and let
\(\lambda_{\min}=\min\Lambda\).  Then

\[
\boxed{
\bar\varepsilon_t
\le\frac{4a_n}{h_{\min}\lambda_{\min}},
\qquad
\zeta_E
\le\alpha_E+
\sum_t C_t
\left(\frac{4a_n}{h_{\min}\lambda_{\min}}\right)^\kappa.}
\tag{13.9}
\]

Hence a genuinely non-vacuous
\(a_n=O(n^{-1/2})\) simultaneous EIRR certificate implies decision disagreement
rate \(O(\alpha_E+Hn^{-\kappa/2})\) for bounded \(C_t\) and decision horizon
\(H\).  More generally, nuisance-rate and residual terms enter only through
the explicit \(a_n\) in (13.1).

### Status

**PROVABLE AS STATED IN THE FINITE-STATE, FINITE-RISK-GRID, CROSS-FITTED
REGIME.**  This is the first proved end-to-end composition in the package, but
it does not establish that \(a_n=O(n^{-1/2})\) holds for a learned Doob transform
or dependent trajectories.  Those remain the central statistical obligations.

### Proof

On the simultaneous training event, apply Theorem 11 to every one of the \(Q\)
queries with calibration failure probability \(\alpha_{\rm cal}/Q\).  A union
bound over calibration failures, followed by a union bound with the training
failure event, gives (13.2).  Independence among queries is not required.

On this event,

\[
0<\underline\psi_{k,\lambda}(x)
\le\psi_{k,\lambda}(x)
\le\overline\psi_{k,\lambda}(x).
\]

The logarithm is increasing.  Adding the same tail term, dividing by positive
\(\lambda\), and taking a minimum preserves the interval:

\[
\underline U_k(x)\le U_k^\star(x)\le\widehat U_k^+(x).
\]

Selecting the mission branch preserves the interval, which proves (13.4).
Notice that this argument remains valid when different risk parameters minimize
the lower, exact, and upper expressions. The direct-return branch has its own
simultaneous interval for certificate labeling but does not enter (13.4).

At any common history, define the learned score

\[
\widehat s_t^+=b_t-m-\widehat U^{+,{\rm eff}}(x_t).
\]

Equation (13.4) gives

\[
s_t^\star-\varepsilon(x_t)
\le\widehat s_t^+\le s_t^\star.
\]

If the exact-risk manager commits, \(s_t^\star\le0\), then the learned manager
also commits; a later first commitment is impossible on the certificate event.
If they first disagree there, necessarily
\(s_t^\star>0\) and \(\widehat s_t^+\le0\), which implies
\(0<s_t^\star\le\varepsilon(x_t)\).  Splitting on certificate failure and using
the oracle-shadow stopped union bound proves (13.5).  The margin condition gives
(13.6), while Theorem 12's bounded-metric coupling gives (13.7)--(13.8).

For the rate, set \(z=c_q/h_q\le a_n/h_{\min}\le1/2\).  Then

\[
\log\frac{h_q+c_q}{h_q-c_q}
=2\operatorname{artanh}(z)
\le4z
\le\frac{4a_n}{h_{\min}}.
\]

The interval width of a minimum is at most the largest component interval
width. Division by \(\lambda\ge\lambda_{\min}\) proves the first part
of (13.9); substitution into (13.6) proves the second. ∎

### Interpretation boundary

- The comparator is the **exact-risk manager using the same finite-grid risk
  functional**.  It equals the formal Stage-B simulator Oracle only when their
  requirements coincide; headroom cannot be transferred from a different
  comparator without evidence.
- A positive interval is mandatory.  If any \(h_q-c_q\le0\), the log-MGF
  certificate is vacuous and the method must abstain, collect coverage, or use a
  different bound.
- The theorem certifies no-later commitment, not lower stranding by itself.
  Earlier return can be declared safety-monotone only with an additional system
  property; the general Pareto rectangle deliberately does not assume it.
- The finite global query set avoids optional-stopping leakage.  Certifying only
  the states observed on the same evaluation path is not covered.
- Theorem 9C prevents a uniform polynomial-rate claim: rare high-resource paths
  can force exponential trajectory complexity, manifested here through large
  \(C_{2,q}\), nuisance error, or a nonshrinking observed residual.

## Theorem 14: Finite-interface transfer upper bound and matching lower bound

### Claim A: Bernstein--resolvent upper bound

Let \(S=\{1,\ldots,d\}\), let \(\mathcal U\) be a finite executed-action
interface, and let \(G_C\) be the absorbing terminal state.  The primitive
kernel is \(K(j,dc\mid i,u)\).  For every
\((i,u)\), suppose the source acquisition protocol reaches that interface at
least \(m_{iu}\) times almost surely.  The visit time is predictable before its
next primitive outcome is observed, and the same time-homogeneous Markov kernel
is used across visits and episode resets.  Thus the first \(m_{iu}\) primitive
outcomes at a fixed interface are independent draws from \(K(\cdot\mid i,u)\),
although the complete source trajectory is dependent.

For \(\lambda\in\Lambda\), define

\[
K_\lambda(i,u,j)
=\mathbb E[e^{\lambda c}\mathbf1\{x'=j\}\mid i,u],
\qquad j\in S\cup\{G_C\},
\]

and let \(\widehat K_\lambda\) be its sample mean.  Let
\(\widehat\sigma^2_{\lambda,i,u,j}\) be the usual sample variance with
denominator \(m_{iu}-1\), where \(m_{iu}\ge2\), and write

\[
W_\lambda=e^{\lambda c_{\max}},
\qquad
\]

and

\[
D=|\Lambda|\,|S|\,|\mathcal U|\,(|S|+1).
\]

For \(\alpha\in(0,1)\), define the cell radius

\[
\eta_{\lambda,i,u,j}
=
\sqrt{\frac{2\widehat\sigma^2_{\lambda,i,u,j}\log(4D/\alpha)}{m_{iu}}}
+\frac{8W_\lambda\log(4D/\alpha)}{3(m_{iu}-1)}.
\tag{14.1}
\]

Then, with probability at least \(1-\alpha\), simultaneously over all cells,

\[
\boxed{
|\widehat K_\lambda(i,u,j)-K_\lambda(i,u,j)|
\le\eta_{\lambda,i,u,j}.}
\tag{14.2}
\]

Let the queryable target policy--filter composition induce the known executed
mixture

\[
\omega_i(u)
=\int\pi(da\mid i)K_\Pi(u\mid i,a).
\]

Compose

\[
M_\lambda(i,j)=\sum_u\omega_i(u)K_\lambda(i,u,j),
\qquad
r_\lambda(i)=\sum_u\omega_i(u)K_\lambda(i,u,G_C),
\]

and define \(\widehat M_\lambda,\widehat r_\lambda\) analogously.  Set

\[
e_{\lambda,i,j}=\sum_u\omega_i(u)\eta_{\lambda,i,u,j},
\]

\[
\Delta_{M,\lambda}
=\max_i\sum_{j\in S}e_{\lambda,i,j},
\qquad
\Delta_{r,\lambda}
=\max_i e_{\lambda,i,G_C}.
\tag{14.3}
\]

On (14.2),

\[
\|M_\lambda-\widehat M_\lambda\|_\infty
\le\Delta_{M,\lambda},
\qquad
\|r_\lambda-\widehat r_\lambda\|_\infty
\le\Delta_{r,\lambda}.
\]

Suppose \(\widehat R_\lambda=(I-\widehat M_\lambda)^{-1}\) exists and the
fully data-computable robust condition

\[
\widehat\kappa_\lambda
=\|\widehat R_\lambda\|_\infty\Delta_{M,\lambda}<1
\tag{14.4}
\]

holds.  With
\(\widehat\psi_\lambda=\widehat R_\lambda\widehat r_\lambda\),

\[
\boxed{
\|\psi_\lambda-\widehat\psi_\lambda\|_\infty
\le
c_\lambda^{\rm plug}
:=
\frac{\|\widehat R_\lambda\|_\infty}
{1-\widehat\kappa_\lambda}
\left(
\Delta_{r,\lambda}
+\Delta_{M,\lambda}\|\widehat\psi_\lambda\|_\infty
\right).}
\tag{14.5}
\]

Thus \(c_\lambda^{\rm plug}\) is an explicit simultaneous radius that can be
inserted into Theorem 13 without estimating the unknown Doob transform itself.
It instead estimates the shared executed primitive and then composes the known
target policy/filter exactly.

If \(m_{iu}\ge m\),
\(\widehat\sigma^2_{\lambda,i,u,j}\le V_\lambda\), and the learned resolvent/value
norms remain bounded away from (14.4), then, suppressing dimensions and log
factors,

\[
c_\lambda^{\rm plug}
=O\!\left(
\sqrt{\frac{V_\lambda}{m}}
+\frac{W_\lambda}{m}
\right).
\tag{14.6}
\]

### Claim B: matching source-interface lower bound

Fix \(L>0\), \(\lambda>0\), and
\(q=e^{-\lambda L}\le1/4\).  Consider one transient state and two executed
interfaces \(u_0,u_1\).  A source transition executes \(u_1\) with probability
\(\mu\in(0,1]\) and otherwise executes \(u_0\).  Interface \(u_0\) always hits
the charger with zero resource.  Conditional on \(u_1\), model \(i\in\{0,1\}\)
hits the charger with resource \(L\) with probability
\(p_0=q\), respectively \(p_1=2q\), and with zero resource otherwise.  The
target composition always executes \(u_1\).

For \(n\) source transitions, every estimator \(\widehat\theta\) of the target
exponential Resource-to-Go satisfies, whenever

\[
n\le\frac{e^{\lambda L}}{4\mu},
\]

\[
\boxed{
\max_{i\in\{0,1\}}
\mathbb E_i|\widehat\theta-\theta_i|
\ge\frac{9}{32},
\qquad
\theta_0=2-q,
\quad
\theta_1=3-2q.}
\tag{14.7}
\]

Consequently, constant-accuracy source-to-target exponential-risk estimation
requires

\[
\boxed{
n=\Omega\!\left(\frac{e^{\lambda L}}{\mu}\right).}
\tag{14.8}
\]

In the same one-step family, the weighted outcome has range
\(W=e^{\lambda L}\) and variance \(\Theta(W)\), not \(\Theta(W^2)\).  Bernstein's
upper bound therefore needs \(m=O(W\log(1/\alpha))\) informative \(u_1\)
samples for constant error.  Standard binomial concentration converts this to
\(n=O(W\log(1/\alpha)/\mu)\) source transitions.  Hence (14.8) matches the
upper bound in its exponential-risk and inverse-interface-coverage order, up to
logs and constants, for this declared family.

### Status

**PROVABLE AS STATED FOR FINITE EXECUTED INTERFACES AND THE DECLARED PREDICTABLE
FIXED-COUNT SAMPLING PROTOCOL.**  Bernstein concentration, robust resolvent
perturbation, and the two-point method are standard.  The project-specific
content is their source policy--filter execution-interface composition,
first-passage exponential operator, and direct handoff to Theorem 13.

### Proof

At every predictable visit to \((i,u)\), the Markov property makes the next
primitive outcome conditionally independent of the past with common law
\(K(\cdot\mid i,u)\).  Induction over the first \(m_{iu}\) visits therefore
gives independent cell observations.  Each is bounded by \(W_\lambda\) and has
the displayed sample variance.  The two-sided empirical Bernstein inequality of
Maurer--Pontil, followed by a union bound over \(D\) cells, proves (14.2).

The target mixture weights are nonnegative and sum to one.  Applying the cell
bounds before this convex combination, then the row-sum definition of the
matrix infinity norm, proves (14.3).

Write \(\Delta=M_\lambda-\widehat M_\lambda\).  The exact factorization

\[
I-M_\lambda
=(I-\widehat M_\lambda)
(I-\widehat R_\lambda\Delta)
\]

implies

\[
R_\lambda
=(I-\widehat R_\lambda\Delta)^{-1}\widehat R_\lambda.
\]

Condition (14.4) makes the first inverse a convergent Neumann series and gives

\[
\|R_\lambda\|_\infty
\le
\frac{\|\widehat R_\lambda\|_\infty}
{1-\widehat\kappa_\lambda}.
\]

Substitution into Theorem 8's perturbation identity proves (14.5).  Equation
(14.6) follows from (14.1)--(14.5) under the declared uniform bounds.

For Claim B, couple one source observation under the two models.  They can differ
only when \(u_1\) is selected and the Bernoulli rare-event indicators differ, an
event of probability \(\mu q\).  Coordinatewise coupling and the union bound
give

\[
\operatorname{TV}(P_0^n,P_1^n)\le n\mu q\le\frac14.
\]

Under the target, direct calculation gives the displayed \(\theta_i\), with
\(|\theta_1-\theta_0|=1-q\ge3/4\).  The two-point inequality used in Theorem 9C
therefore yields

\[
\max_i\mathbb E_i|\widehat\theta-\theta_i|
\ge\frac{1-q}{2}
\left(1-\operatorname{TV}(P_0^n,P_1^n)\right)
\ge\frac{9}{32}.
\]

This proves (14.7)--(14.8).  Finally, for the rare weighted outcome,
\(p=1/W\) and its variance is
\(p(1-p)(W-1)^2=\Theta(W)\).  Substitution in Bernstein's radius and binomial
concentration for the number of source \(u_1\) visits proves the matching-order
upper statement. ∎

### Interpretation boundary

- The fixed-count predictable-interface protocol is stronger than arbitrary
  replay sampling. If an interface is visited only on a data-dependent survival
  event, naively conditioning on its realized count can bias a non-time-uniform
  interval.
- The theorem is tabular. Neural approximation, continuous actions, hidden
  filter state, and learned state abstractions require new complexity and
  misspecification terms.
- The plug-in route deliberately bypasses direct learned-Doob estimation. A
  DICE/EIRR estimator must beat its certificate width or computational scaling
  to justify additional machinery.
- The lower bound varies source coverage \(\mu\) and rare-resource severity
  jointly. It does not claim every physical environment realizes the hard
  two-model family.
- The matching statement is order-level for the explicit one-step family, not a
  general minimax theorem for all proper SSPs.

## Theorem 15: Coverage--risk lower bound for irreversible Pareto decisions

### Claim

Use the two-model source/target family from Claim B of Theorem 14.  Fix any
\(\delta\in(0,1)\) and the single-risk requirements

\[
U_i
=\frac{\log\theta_i+\log(1/\delta)}{\lambda},
\qquad i\in\{0,1\}.
\]

Because \(\theta_0<\theta_1\), we have \(U_0<U_1\).  At one irreversible
decision epoch choose the net remaining resource

\[
b-m=\frac{U_0+U_1}{2}.
\]

The exact-risk ReturnManager continues under model 0 and commits under model 1.
For any possibly randomized learned manager \(\widehat d(D)\) based on \(n\)
source transitions, if

\[
n\le\frac{e^{\lambda L}}{4\mu},
\]

then

\[
\boxed{
\max\left\{
\Pr_0(\widehat d={\rm COMMIT}),
\Pr_1(\widehat d={\rm CONTINUE})
\right\}
\ge\frac38.}
\tag{15.1}
\]

This testing problem admits a direct stranding--throughput embedding.  Normalize
throughput to \([0,1]\). Under model 0, continuing completes one task safely,
whereas committing completes no task; hence throughput regret equals the first
error indicator. Under model 1, committing returns safely, whereas continuing
strands; hence stranding excess equals the second error indicator. Therefore

\[
\boxed{
\max\left\{
\mathbb E_0[q_0^\star-q_{\widehat d}],
\mathbb E_1[p_{\widehat d}-p_1^\star]
\right\}
\ge\frac38.}
\tag{15.2}
\]

Consequently, no source-only return algorithm can uniformly preserve both
exact-risk Pareto coordinates to error below \(3/8\) on this family with
\(o(e^{\lambda L}/\mu)\) samples.

### Status

**PROVABLE AS STATED FOR THE EXPLICIT TWO-MODEL FAMILY.**  The testing reduction
is classical.  Its role is to show that the risk--coverage barrier is not merely
a prediction-metric pathology: it can force either premature return or
stranding in the original irreversible decision.

### Proof

The threshold lies strictly between \(U_0\) and \(U_1\).  Since the implemented
score is \(s=b-m-U\) and commitment occurs iff \(s\le0\), the two exact actions
are opposite.  Any learned action is therefore a binary test of the two source
laws.  Le Cam's testing inequality gives

\[
\Pr_0(\widehat d={\rm COMMIT})
+\Pr_1(\widehat d={\rm CONTINUE})
\ge1-\operatorname{TV}(P_0^n,P_1^n).
\]

Theorem 14 proves the right-hand side is at least \(3/4\) under the displayed
sample condition, so the larger of the two errors is at least \(3/8\), proving
(15.1).  In the declared embedding, these two indicators are exactly normalized
throughput regret and stranding excess, respectively, which proves (15.2). ∎

### Interpretation boundary

- Equation (15.2) is a minimax construction, not an empirical claim about the
  current simulator's Oracle headroom.
- The hard pair puts the decision threshold between two nearby risk models by
  construction. Environments with large observed decision margin can be much
  easier; that is exactly the role of Theorems 12--13's stopped margin law.
- The lower bound applies to every estimator or neural architecture using only
  the declared source data. Physical structure or active target-interface
  queries can change the information class and must be analyzed separately.

## Theorem 16: Finite linear risk-witness quotient — exact identifiability and rate

### Setup

Fix one risk level and one continuation witness using a fold independent of the
samples below. Let \(\phi(h(q))\in\mathbb R^d\) be an encoded-interface feature.
Conditional exponential witness observations obey the correctly specified
fixed-design Gaussian model

\[
Y=X\theta+\varepsilon,
\qquad
\varepsilon\sim N(0,\sigma^2 I_n),
\tag{16.1}
\]

where row \(i\) of \(X\) is \(\phi(h(q_i))^\top\). Let

\[
m=\int\phi(h(q))\,\bar\eta_{\nu,\lambda}(dq)
\quad\text{and}\quad
J(\theta)=m^\top\theta.
\tag{16.2}
\]

Here the target risk occupation and continuation witness are treated as fixed
nuisances. Thus \(J\) is the linear slice of the exact EIRR residual in
(RBQ.11), not a claim that those nuisances are known in the full problem.

### Claim A: exact identifiability dichotomy

The target functional \(J\) is identified from the source law for every
\(\theta\in\mathbb R^d\) if and only if

\[
\boxed{m\in\operatorname{Range}(X^\top).}
\tag{16.3}
\]

If (16.3) fails, then on the parameter ball
\(\Theta_B=\{\theta:\|\theta\|_2\le B\}\), every estimator satisfies

\[
\boxed{
\inf_{\widehat J}
\sup_{\theta\in\Theta_B}
\mathbb E_\theta(\widehat J-J(\theta))^2
\ge
B^2\|P_{\operatorname{Null}(X)}m\|_2^2.}
\tag{16.4}
\]

Thus supervised source fit, raw encoder dimension, and small training residual do
not imply target risk-functional identification.

### Claim B: exact minimax risk when identifiable

If (16.3) holds, define

\[
v_{X,m}=m^\top(X^\top X)^\dagger m,
\qquad
\widehat J
=m^\top(X^\top X)^\dagger X^\top Y.
\tag{16.5}
\]

Then

\[
\widehat J-J(\theta)\sim N(0,\sigma^2v_{X,m})
\tag{16.6}
\]

and, over the unrestricted identifiable Gaussian family,

\[
\boxed{
\inf_{\widetilde J}
\sup_{\theta\in\mathbb R^d}
\mathbb E_\theta(\widetilde J-J(\theta))^2
=\sigma^2v_{X,m}.}
\tag{16.7}
\]

Writing \(C_{X,m}=n v_{X,m}\), the exact standard-error scale is

\[
\sigma\sqrt{C_{X,m}/n}.
\tag{16.8}
\]

This is the finite linear form of a risk-restricted quotient leverage
coefficient. It depends on target risk occupation in feature space, not on the
minimum probability of every raw interface cell.

### Claim C: strict structural advantage over raw-support plug-in

Let the raw interface be \(q\in[-1,1]\) with
\(\phi(q)=(1,q)^\top\). Source data contain only balanced samples at
\(q=-1\) and \(q=1\), while the target risk occupation is a point mass at
\(q=0\). The raw target law is singular with respect to source interface
occupation, so a cellwise density ratio or primitive plug-in at \(q=0\) is not
identified. Nevertheless,

\[
X^\top X=nI_2,
\qquad m=(1,0)^\top,
\qquad C_{X,m}=1,
\tag{16.9}
\]

and (16.7) gives exact minimax risk \(\sigma^2/n\). The advantage is purchased
by the declared linear-witness structure; it does not arise from weighting alone.

### Claim D: matching local irreversible-decision floor

Assume \(v_{X,m}>0\) and define

\[
d=(X^\top X)^\dagger m/\sqrt{v_{X,m}},
\qquad
\theta_\pm=\pm\frac{\sigma}{2}d.
\tag{16.10}
\]

Then \(\|Xd\|_2=1\) and

\[
J(\theta_\pm)=\pm\frac{\sigma}{2}\sqrt{v_{X,m}}.
\]

Place an irreversible ReturnManager threshold between these two requirements.
For every possibly randomized decision rule \(\widehat d(Y)\), Gaussian testing
gives

\[
\boxed{
\max\{\Pr_{\theta_-}(\widehat d=+),
       \Pr_{\theta_+}(\widehat d=-)\}
\ge\Phi(-1/2)\approx0.3085.}
\tag{16.11}
\]

Consequently, the local decision resolution is exactly of order
\(\sigma\sqrt{C_{X,m}/n}\). Embedding the two signs as premature return and
stranding gives the same lower scale for one of the two normalized Pareto
coordinates.

### Status

**PROVABLE AS STATED FOR THE FIXED-DESIGN GAUSSIAN, FIXED-NUISANCE SLICE.**
The linear identifiability and minimax tools are classical. The theorem's role is
to replace the tabular minimum-coverage obstruction by a necessary-and-sufficient
feature-quotient condition and to state exactly what structural generalization
buys before reconnecting it to the irreversible decision.

### Proof

For Claim A, two parameters induce the same source law exactly when their
difference lies in \(\operatorname{Null}(X)\). The functional is constant on
every such equivalence class exactly when \(m\) is orthogonal to
\(\operatorname{Null}(X)\), which is (16.3). If it fails, take the unit vector

\[
u=P_{\operatorname{Null}(X)}m/
\|P_{\operatorname{Null}(X)}m\|_2
\]

and \(\theta_\pm=\pm Bu\). Their observation laws are identical and their
functional values are opposite with magnitude
\(B\|P_{\operatorname{Null}(X)}m\|_2\). Averaging the two squared errors gives
(16.4).

Under (16.3), substitution into (16.5) proves unbiasedness and (16.6), so the
right side of (16.7) is attainable. For the lower bound, take a compact SVD
\(X=USV^\top\) and write \(m=Vb\). A Gaussian prior
\(V^\top\theta\sim N(0,\tau^2I)\) has posterior covariance

\[
(\tau^{-2}I+\sigma^{-2}S^2)^{-1}.
\]

The Bayes risk for \(b^\top V^\top\theta\) converges as
\(\tau\to\infty\) to
\(\sigma^2b^\top S^{-2}b=\sigma^2v_{X,m}\). Minimax risk is at least every
Bayes risk, proving the reverse inequality in (16.7). Claim C follows by direct
calculation.

For Claim D, pseudoinverse identities under (16.3) give
\(d^\top X^\top Xd=1\) and \(m^\top d=\sqrt{v_{X,m}}\). The two observation
laws are Gaussians with common covariance \(\sigma^2I\) and means separated by
\(\sigma Xd\), whose Mahalanobis distance is one. The equal-prior optimal binary
testing error is therefore \(\Phi(-1/2)\). The maximum of the two directional
errors is no smaller than their average, proving (16.11). ∎

### Interpretation boundary

- This theorem does not estimate \(m\), the Doob/risk occupation, or the
  continuation witness. Their errors require cross-fitted nuisance terms.
- Gaussian homoskedastic noise is a proof slice, not a physical Energy model.
  Martingale, heteroskedastic, and dependent-trajectory extensions remain open.
- The singular-support example relies on correct linear extrapolation. Without
  structure, Theorems 9A and 14's non-identification conclusions remain.
- The theorem prescribes representation coverage and whitening/leverage control;
  it does not by itself prescribe a unique neural architecture.
- Claims A--D are classical linear-statistical ingredients specialized to the
  risk-occupation/irreversible-decision object, not standalone oral novelty.

## Theorem 17: Two-layer orthogonal EIRR score and exact product bias

Fix \(\lambda\), the initial law \(\nu\), the queryable target execution kernel
\(L(du\mid x)\), and the shared primitive transition law. Let

\[
\rho_Q(dx,du)=\rho_X(dx)B(du\mid x),
\qquad
\eta^X=\nu(I-M_\lambda)^{-1},
\qquad
\eta^Q=\eta^X L.
\tag{17.1}
\]

Assume \(\eta^X\ll\rho_X\) and \(\eta^Q\ll\rho_Q\), with square-integrable
non-normalized density ratios

\[
v_\lambda=\frac{d\eta^X}{d\rho_X},
\qquad
w_\lambda=\frac{d\eta^Q}{d\rho_Q}.
\tag{17.2}
\]

For a state continuation \(f\) and executed-interface continuation \(g\), set

\[
A_{f,g}(x)=Lg(x)-f(x),
\qquad
B_{\lambda,f,g}(q,Y)=\Gamma_{\lambda,f}(Y)-g(q),
\tag{17.3}
\]

where
\(\mathbb E[\Gamma_{\lambda,f}(Y)\mid Q=q]=\mathcal K_\lambda f(q)\).
Define

\[
\mathcal S_\lambda(f,g;v,w)
=\nu f
+\mathbb E_{\rho_X}[v(X)A_{f,g}(X)]
+\mathbb E_{\rho_QK}[w(Q)B_{\lambda,f,g}(Q,Y)].
\tag{17.4}
\]

All expectations below are assumed finite, and candidate ratios are measurable
with respect to the argument shown.

### Claim A: exact risk-ratio block

For every admissible \(f,g\),

\[
\boxed{
\mathcal S_\lambda(f,g;v_\lambda,w_\lambda)
=\theta_\lambda:=\nu\psi_\lambda.}
\tag{17.5}
\]

Thus the continuation functions may both be misspecified without changing the
population target when both risk-occupation ratios are correct.

### Claim B: exact continuation block

Let

\[
f^*=\psi_\lambda,
\qquad
g^*=\mathcal K_\lambda\psi_\lambda.
\tag{17.6}
\]

For arbitrary integrable ratio candidates \(\widetilde v,\widetilde w\),

\[
\boxed{
\mathcal S_\lambda(f^*,g^*;\widetilde v,\widetilde w)
=\theta_\lambda.}
\tag{17.7}
\]

Claims A--B are block double robustness. They do not claim exactness when only
one nuisance inside a block is correct.

### Claim C: exact two-layer product bias

For arbitrary candidates
\((\widehat f,\widehat g,\widehat v,\widehat w)\),

\[
\boxed{
\begin{aligned}
&\mathcal S_\lambda(\widehat f,\widehat g;
\widehat v,\widehat w)-\theta_\lambda\\
&=\mathbb E_{\rho_X}
[(\widehat v-v_\lambda)(L\widehat g-\widehat f)]\\
&\quad+\mathbb E_{\rho_Q}
[(\widehat w-w_\lambda)
(\mathcal K_\lambda\widehat f-\widehat g)].
\end{aligned}}
\tag{17.8}
\]

Consequently,

\[
\begin{aligned}
|\mathcal S_\lambda-\theta_\lambda|
&\le
\|\widehat v-v_\lambda\|_{2,\rho_X}
\|L\widehat g-\widehat f\|_{2,\rho_X}\\
&\quad+
\|\widehat w-w_\lambda\|_{2,\rho_Q}
\|\mathcal K_\lambda\widehat f-\widehat g\|_{2,\rho_Q}.
\end{aligned}
\tag{17.9}
\]

Under cross-fitting, iid validation transitions or an explicitly declared
episode-level CLT, and uniform \(2+\epsilon\) moments, making both products in
(17.9) \(o_p(n^{-1/2})\) is sufficient to remove first-order nuisance bias.
This is a rate condition, not a proved nuisance-learning theorem for replay data.

### Proof

Change of measure and conditional expectation give

\[
\begin{aligned}
\mathcal S_\lambda(f,g;v_\lambda,w_\lambda)
&=\nu f+\eta^X(Lg-f)+\eta^Q(\mathcal K_\lambda f-g)\\
&=\nu f+\eta^X(\mathcal T_\lambda f-f).
\end{aligned}
\tag{17.10}
\]

Writing \(\mathcal T_\lambda f=r_\lambda+M_\lambda f\), the occupation
balance \(\eta^X=\nu+\eta^X M_\lambda\) and fixed-point identity
\(\psi_\lambda=(I-M_\lambda)^{-1}r_\lambda\) imply

\[
\eta^X(\mathcal T_\lambda f-f)
=\eta^Xr_\lambda-\nu f
=\nu\psi_\lambda-\nu f,
\tag{17.11}
\]

which proves Claim A. For Claim B,
\(Lg^*=L\mathcal K_\lambda\psi_\lambda=\psi_\lambda=f^*\), while
\(\mathbb E[B_{\lambda,f^*,g^*}\mid Q]=0\). Both correction terms therefore
vanish even under arbitrary candidate ratios.

For Claim C, subtract (17.5), evaluated at the same
\((\widehat f,\widehat g)\), from (17.4), then condition on \(Q\) in the
primitive term. This gives (17.8) exactly. Applying Cauchy--Schwarz separately
under \(\rho_X\) and \(\rho_Q\) gives (17.9). ∎

### Status and interpretation boundary

**PROVABLE AS A POPULATION IDENTITY UNDER FIXED MEASURES AND QUERYABLE
COMPOSITION; FINITE EXAMPLE VERIFIED.** The verifier reports exact-ratio and
exact-model scores equal to target \(1.0369960217\); with both blocks perturbed,
actual bias \(0.04955025\) equals the two product terms and is below the L2 bound
\(0.09152815\).

- This theorem is a two-layer specialization of standard doubly robust and
  orthogonal off-policy estimation. Double robustness by itself is not an
  oral-level novelty claim.
- Its project-specific content is the separation of the queryable safety
  execution residual from the killed exponential primitive residual and their
  composition into charger-hitting risk occupation.
- It does not establish positivity from data, learn \(v,w,f,g\), handle arbitrary
  dependent replay, select \(\lambda\), or prove an Oracle/Pareto improvement.
- A closest-theorem audit against double reinforcement learning and marginalized
  occupancy-ratio estimators is mandatory before any novelty claim.

## Theorem 18: Canonical gradient and local information limit for stopped EIRR

Consider the Q13 observed-data model

\[
O=(Q,Y)\sim\rho_Q(dq)P(dY\mid q),
\]

where \(\nu\), the queryable target execution kernel \(L\), and the source
design \(\rho_Q\) are fixed, and only the shared primitive conditional law
\(P\) varies. Assume differentiability in quadratic mean, exponential
transience, and

\[
\mathbb E_{\rho_Q}
\left[w_\lambda(Q)^2
\operatorname{Var}\{\Gamma_{\lambda,\psi_\lambda}(Y)\mid Q\}
\right]<\infty.
\tag{18.1}
\]

For
\(\theta_\lambda(P)=\nu(I-M_{\lambda,P})^{-1}r_{\lambda,P}\), define

\[
\varphi_\lambda(Q,Y)
=w_\lambda(Q)
\{\Gamma_{\lambda,\psi_\lambda}(Y)
-\mathcal K_\lambda\psi_\lambda(Q)\}.
\tag{18.2}
\]

### Claim A: canonical gradient

For every regular conditional-law score \(s\) with
\(\mathbb E[s\mid Q]=0\),

\[
\left.\frac{d}{dt}\theta_\lambda(P_t)\right|_{t=0}
=\mathbb E[\varphi_\lambda(O)s(O)].
\tag{18.3}
\]

Moreover, \(\varphi_\lambda\) lies in the nonparametric conditional tangent
space, so it is the canonical gradient. If \(\rho_Q\) is also unknown,
\(\theta_\lambda\) is invariant in marginal-\(Q\) directions and (18.2)
remains canonical.

### Claim B: efficiency bound and the role of the two layers

The semiparametric efficiency bound is

\[
\boxed{
V_{\mathrm{eff},\lambda}
=\mathbb E_{\rho_Q}
\left[w_\lambda(Q)^2
\operatorname{Var}
\{\Gamma_{\lambda,\psi_\lambda}(Y)\mid Q\}
\right].}
\tag{18.4}
\]

At the true continuation block, the centered one-observation score from Theorem
17 equals \(\varphi_\lambda\); its state-composition correction is identically
zero. Therefore the state-ratio nuisance \(v_\lambda\) provides robustness to
misspecified continuation functions but is not a second component of the
canonical gradient when \(L\) is known.

### Claim C: local irreversible-decision floor

Suppose the normalized least-favourable score
\(s^*=\varphi_\lambda/\sqrt{V_{\mathrm{eff},\lambda}}\) generates a regular
bounded conditional submodel, or is approximable by such scores. For local
alternatives \(P_{\pm a/\sqrt n}\),

\[
\theta_\lambda(P_{\pm a/\sqrt n})
=\theta_\lambda(P)
\pm\frac{a\sqrt{V_{\mathrm{eff},\lambda}}}{\sqrt n}
+o(n^{-1/2}).
\tag{18.5}
\]

For any decision rule choosing the sign relative to a threshold at
\(\theta_\lambda(P)\), local asymptotic normality yields

\[
\liminf_{n\to\infty}
\inf_{\widehat d_n}
\max_{\sigma\in\{-,+\}}
\Pr_{P_{\sigma a/\sqrt n}}
\{\widehat d_n\ne\sigma\}
\ge\Phi(-a).
\tag{18.6}
\]

Thus \(\sqrt{V_{\mathrm{eff},\lambda}/n}\) is the exact local statistical
resolution scale before translating the two signs into premature return versus
stranding.

### Proof

For a conditional score \(s\), differentiating the primitive operator at fixed
\(f\) gives

\[
\dot{\mathcal K}_\lambda f(q)
=\mathbb E[\Gamma_{\lambda,f}(Y)s(q,Y)\mid Q=q].
\tag{18.7}
\]

Differentiating the fixed point and solving the linearized equation gives

\[
\dot\psi_\lambda
=(I-M_\lambda)^{-1}L
\dot{\mathcal K}_\lambda\psi_\lambda.
\tag{18.8}
\]

Left multiplication by \(\nu\), followed by
\(\eta^X=\nu(I-M_\lambda)^{-1}\), \(\eta^Q=\eta^XL\), and
\(w=d\eta^Q/d\rho_Q\), yields

\[
\dot\theta_\lambda[s]
=\mathbb E[w(Q)\Gamma_{\lambda,\psi_\lambda}(Y)s(Q,Y)].
\]

Subtracting \(\mathcal K_\lambda\psi_\lambda(Q)\) does not change the
expectation because \(s\) is conditionally centered, proving (18.2)--(18.3).
Membership in the full conditional tangent space makes this Riesz representer
canonical; its squared norm is (18.4). Theorem 17's first residual vanishes at
\((f^*,g^*)\), proving Claim B. Claim C is the ordinary least-favourable
one-dimensional LAN testing reduction with total local separation \(2a\). ∎

### Status and interpretation boundary

**PROVABLE IN THE DECLARED IID CONDITIONAL-PRIMITIVE MODEL; ASYMPTOTIC RESULT,
NOT A DEPENDENT-REPLAY THEOREM.** Canonical gradients, efficiency bounds, and LAN
testing reductions are classical. The specialization identifies the correct
Energy-risk information scale and removes an unnecessary claimed information
layer; it is not standalone oral novelty.

The focused audit in
`literature-search-20260830-two-layer-orthogonal-risk-ope/` finds close results
for DRL, DR risk-CDF estimation, restricted OPE, linear-MDP efficiency, and
off-environment ratio factorization. A paper-level theorem must therefore prove a
new phase transition or strict separation involving risk-observable quotient
coverage, approach to the killed transience boundary, and the stopped
ReturnManager margin—not merely (18.2).

## Theorem 19: Exact quotient--transience phase transition in a stopped-risk slice

Fix \(a=e^{\lambda c}>1\). Consider a single noncharger state whose target
executed risk-equivalence class terminates at the charger with probability \(p\)
and otherwise returns to the same state. Let \(\mu_h>0\) be the total source
design probability of interfaces constrained to share this primitive conditional
law, and assume

\[
\Delta:=1-a(1-p)>0.
\tag{19.1}
\]

Then the charger-hitting exponential value, raw-value efficiency bound, log-value
efficiency bound, and requirement efficiency bound are respectively

\[
\psi=\frac{ap}{\Delta},
\tag{19.2}
\]

\[
\boxed{V_\psi
=\frac{p(1-p)a^2(a-1)^2}{\mu_h\Delta^4},}
\tag{19.3}
\]

\[
\boxed{V_{\log\psi}
=\frac{(1-p)(a-1)^2}{\mu_h p\Delta^2},}
\tag{19.4}
\]

and, for \(U=(\log\psi+b)/\lambda\) with fixed \(b\),

\[
\boxed{V_U
=\frac{(1-p)(a-1)^2}
{\mu_h p\lambda^2\Delta^2}.}
\tag{19.5}
\]

Consequently, along triangular arrays with fixed \(a>1\) and
\(p\downarrow1-1/a\), the information scales are

\[
n\mu_h\Delta^4
\quad\text{for absolute raw-MGF estimation},
\qquad
n\mu_h\Delta^2
\quad\text{for log-risk and requirement estimation}.
\tag{19.6}
\]

The local requirement resolution is

\[
s_n
=\frac{a-1}{\lambda\Delta}
\sqrt{\frac{1-p}{n\mu_h p}},
\tag{19.7}
\]

and threshold alternatives separated by \(\pm z s_n\) have asymptotic minimax
sign error at least \(\Phi(-z)\) under the regular local model.

### Proof

The fixed-point equation is
\(\psi=ap+a(1-p)\psi\), which gives (19.2), while the risk occupation is
\(\eta=1/\Delta\). On the informative quotient class,
\(w=\eta/\mu_h=1/(\mu_h\Delta)\). Conditional on that class, the witness equals
\(a\) with probability \(p\) and \(a\psi\) with probability \(1-p\). Therefore

\[
\operatorname{Var}(\Gamma_{\lambda,\psi})
=p(1-p)a^2(1-\psi)^2
=\frac{p(1-p)a^2(a-1)^2}{\Delta^2}.
\]

Substitution in Theorem 18's
\(E_{\rho_Q}[w^2\operatorname{Var}(\Gamma\mid Q)]\) contributes source mass
\(\mu_h\) and proves (19.3). Dividing by \(\psi^2\), then by \(\lambda^2\),
proves (19.4)--(19.5). Equivalently,

\[
\frac{d\psi}{dp}=-\frac{a(a-1)}{\Delta^2},
\qquad
\frac{d\log\psi}{dp}=-\frac{a-1}{p\Delta},
\]

and the Bernoulli efficiency variance for \(p\) under source mass \(\mu_h\) is
\(p(1-p)/\mu_h\). Equation (19.6) follows because the remaining factors stay
bounded away from zero and infinity at the critical limit. Equation (19.7) is
\(\sqrt{V_U/n}\), and Theorem 18's least-favourable LAN reduction proves the
decision floor. ∎

### Quotient and novelty boundary

- \(\mu_h\) is pooled source mass only under an exact structural constraint that
  all interfaces in the class share the relevant primitive law. It cannot be
  obtained by declaring a learned encoder after looking at outcomes.
- A raw target interface may have zero source mass while \(\mu_h>0\), giving the
  same structural-support separation as Theorem 16. Without the equality
  constraint, the target primitive remains nonidentified.
- The fourth-power raw-MGF blow-up and second-power log-risk blow-up are exact in
  this slice. Bernoulli Fisher information, delta methods, and LAN are classical.
- The paper-level target was a Perron--Frobenius generalization: a simple leading
  eigenvalue \(1-\Delta\), nondegenerate primitive noise in its eigenmode, and
  quotient coverage should imply raw/log critical exponents with a distinct
  degenerate phase. Theorem 20 below now proves that finite critical-family
  statement; its novelty and joint-limit restrictions remain separate issues.

## Theorem 20: Perron-mode information phase split for stopped EIRR

Let \(\{M_\Delta:0<\Delta\le\Delta_0\}\) be a finite irreducible family of
nonnegative killed Feynman--Kac operators with simple Perron root
\(1-\Delta\). Normalize the positive left/right Perron vectors by

\[
\ell_\Delta^\top\mathbf1=1,
\qquad
\ell_\Delta^\top z_\Delta=1.
\tag{20.1}
\]

Assume they converge to positive limits, and that

\[
H_\Delta
=(I-M_\Delta)^{-1}
-\frac{z_\Delta\ell_\Delta^\top}{\Delta}
\tag{20.2}
\]

is uniformly bounded. Let \(r_\Delta\to r_0\), fix initial law \(\nu\), and set

\[
a_\Delta=\nu z_\Delta\to a_0>0,
\qquad
b_\Delta=\ell_\Delta^\top r_\Delta\to b_0>0.
\tag{20.3}
\]

Work after an exact, predeclared risk-observable interface quotient. For quotient
interface \(q=(x,\bar u)\), let \(L_\Delta(q\mid x)\) be its target execution
probability and \(\rho_{Q,\Delta}(q)\) its source design mass. Define

\[
Z_\Delta(q,Y)
=e^{\lambda C}\mathbf1\{X'\notin G_C\}z_\Delta(X'),
\qquad
\sigma_\Delta^2(q)
=\operatorname{Var}\{Z_\Delta(q,Y)\mid Q=q\},
\tag{20.4}
\]

and

\[
\mathcal C_\Delta
=\sum_q
\frac{\{\ell_\Delta(x)L_\Delta(q\mid x)\}^2}
{\rho_{Q,\Delta}(q)}
\sigma_\Delta^2(q).
\tag{20.5}
\]

Assume target-leading quotient support is source dominated, relevant conditional
second moments are uniformly bounded, and the quantities above converge.

### Claim A: Perron expansions

The value, risk-state occupation, and initial-law target obey

\[
\psi_\Delta
=\frac{b_\Delta}{\Delta}z_\Delta+O(1),
\qquad
\eta^X_\Delta
=\frac{a_\Delta}{\Delta}\ell_\Delta^\top+O(1),
\qquad
\theta_\Delta=\nu\psi_\Delta
=\frac{a_\Delta b_\Delta}{\Delta}+O(1).
\tag{20.6}
\]

### Claim B: nondegenerate critical information

If \(\mathcal C_\Delta\to\mathcal C_0\in(0,\infty)\), then Theorem 18's exact
efficient variances for the scalar initial-law target \(\theta_\Delta\) satisfy

\[
\boxed{
\Delta^4V_{\theta,\Delta}
\longrightarrow a_0^2b_0^2\mathcal C_0,}
\tag{20.7}
\]

\[
\boxed{
\Delta^2V_{\log\theta,\Delta}
\longrightarrow\mathcal C_0.}
\tag{20.8}
\]

For \(U_\Delta=(\log\theta_\Delta+c_0)/\lambda\),

\[
\boxed{
\Delta^2V_{U,\Delta}
\longrightarrow\mathcal C_0/\lambda^2.}
\tag{20.9}
\]

Thus the regular efficient standard-error scale is

\[
s_{n,\Delta}
=\frac{\sqrt{\mathcal C_0}}
{\lambda\sqrt n\,\Delta}\{1+o(1)\}.
\tag{20.10}
\]

For each fixed \(\Delta>0\), margins of order \(z s_{n,\Delta}\) inherit the
local sign-decision error floor \(\Phi(-z)\) from Theorem 18. For a joint
triangular array \(\Delta=\Delta_n\downarrow0\), impose in addition uniform DQM,
a uniform Lindeberg condition for the normalized least-favourable scores, and
LAN along the array. Only under those extra assumptions does the same decision
floor hold. Then \(n\Delta_n^2/\mathcal C_0\to\infty\) is exactly the condition
for the regular efficiency scale to vanish; it is not by itself an
unconditional consistency theorem. Inverse quotient coverage is already
contained in \(\mathcal C_0\).

### Claim C: projected-noise-degenerate phase

If, for every target-leading quotient interface and all sufficiently small
\(\Delta\), \(Z_\Delta(q,Y)\) is conditionally deterministic, then

\[
\boxed{
V_{\theta,\Delta}=O(\Delta^{-2}),
\qquad
V_{\log\theta,\Delta}=O(1).}
\tag{20.11}
\]

If \(\mathcal C_\Delta\to0\) without exact degeneracy, intermediate rates are
possible; no universal exponent is asserted.

### Proof

The spectral projector identity gives

\[
(I-M_\Delta)^{-1}
=\frac{z_\Delta\ell_\Delta^\top}{\Delta}+H_\Delta.
\]

Multiplication by \(r_\Delta\) and by \(\nu\) proves (20.6). The exact primitive
witness and risk-interface ratio can therefore be written

\[
\Gamma_{\lambda,\psi_\Delta}(Y)
=\frac{b_\Delta}{\Delta}Z_\Delta(q,Y)+W_\Delta(q,Y),
\tag{20.12}
\]

\[
w_\Delta(q)
=\frac{a_\Delta}{\Delta}
\frac{\ell_\Delta(x)L_\Delta(q\mid x)}
{\rho_{Q,\Delta}(q)}+O(1),
\tag{20.13}
\]

where \(W_\Delta\) has uniformly bounded conditional second moment. Expanding
the conditional variance in (20.12) gives

\[
\operatorname{Var}(\Gamma_{\lambda,\psi_\Delta}\mid q)
=\frac{b_\Delta^2}{\Delta^2}\sigma_\Delta^2(q)
+O(\Delta^{-1})+O(1).
\tag{20.14}
\]

Substitute (20.13)--(20.14) into
\(V_\psi=E_{\rho_Q}[w^2\operatorname{Var}(\Gamma\mid Q)]\). After multiplication
by \(\Delta^4\), the lower-order terms vanish and (20.5) yields (20.7). Since
\(\Delta\theta_\Delta\to a_0b_0\), division by \(\theta_\Delta^2\) proves
(20.8), and division by \(\lambda^2\) proves (20.9). Equation (20.10) follows as
the corresponding regular efficiency scale. Theorem 18 gives its decision
consequence pointwise in fixed \(\Delta\); applying it along
\(\Delta_n\downarrow0\) requires the additional uniform triangular-array LAN
assumptions stated after (20.10).

Under exact projected-noise degeneracy, the first term in (20.12) is
conditionally constant and disappears upon centering. Hence the conditional
witness variance is \(O(1)\), while (20.13) gives \(w^2=O(\Delta^{-2})\).
This proves (20.11). ∎

### Status and interpretation boundary

**PROVABLE AS A FINITE-STATE CRITICAL-FAMILY THEOREM UNDER THE EXPLICIT SPECTRAL,
MOMENT, AND EXACT-QUOTIENT ASSUMPTIONS.** A non-symmetric two-state verifier uses
the same Perron root and subleading eigenvalue in both regimes. In the
nondegenerate construction, tail slopes are \(-4.044\) and \(-2.039\), with raw
and log limiting-constant relative errors \(1.30\%\) and \(1.15\%\). In the
projected-noise-degenerate construction, projected variance is
\(1.97\times10^{-31}\), and slopes change to \(-2.023\) and \(-0.017\).

- The theorem concerns iid source primitives and a finite exact quotient; it does
  not establish learned quotient recovery or dependent-replay inference.
- Perron resolvent expansions, canonical gradients, and LAN are classical. The
  focused audit in
  `literature-search-20260830-critical-perron-eirr/` finds direct collisions with
  Markov-chain conditioning, killed-process perturbation, Feynman--Kac particle
  limits, QSD estimation CLTs, SSP hardness, and margin theory. The spectral
  exponents alone are therefore not a safe novelty claim.
- The theorem reaches a local irreversible sign-decision floor, not the formal
  empirical stranding--throughput Pareto Gate, which remains unavailable behind
  failed navigation.

## Theorem 21: Cost-aware critical-mode oracle and pilot stability

Adopt Theorems 18 and 20 and suppose the finite exact quotient interfaces are
independently sampleable. A primitive from interface \(q\) costs
\(\kappa(q)\in(0,\infty)\). For total acquisition budget \(\mathsf B\), define

\[
\alpha_\Delta(q)
=\frac{\eta^X_\Delta(x)L_\Delta(q\mid x)}{\theta_\Delta}
\sqrt{\operatorname{Var}\{
\Gamma_{\lambda,\psi_\Delta}(Y)\mid Q=q\}},
\qquad
S_\Delta=\sum_q\alpha_\Delta(q)\sqrt{\kappa(q)}.
\tag{21.1}
\]

### Claim A: exact costed oracle

For fixed positive stratum counts \(n_q\), the attainable canonical variance for
the scalar log-risk target is

\[
\mathcal V_\Delta(\boldsymbol n)
=\sum_q\frac{\alpha_\Delta(q)^2}{n_q}.
\tag{21.2}
\]

Subject to \(\sum_q\kappa(q)n_q\le\mathsf B\), its exact continuous-allocation
solution is

\[
\boxed{
n_q^*=\frac{\mathsf B}{S_\Delta}
\frac{\alpha_\Delta(q)}{\sqrt{\kappa(q)}},
\qquad
\inf_{\boldsymbol n}\mathcal V_\Delta(\boldsymbol n)
=\frac{S_\Delta^2}{\mathsf B}.}
\tag{21.3}
\]

The formula applies to positive-sensitivity strata. A zero-sensitivity stratum
can receive zero oracle second-stage mass only after a pilot or structural
certificate; integer rounding is asymptotically negligible when active counts
diverge.

### Claim B: critical Perron allocation

Under Theorem 20's nondegenerate critical-family assumptions, let

\[
d_0(q)=\ell_0(x)L_0(q\mid x)\sigma_0(q).
\tag{21.4}
\]

Then \(\Delta\alpha_\Delta(q)\to d_0(q)\) and

\[
\boxed{
\mathsf B\Delta^2
\inf_{\boldsymbol n}\mathcal V_\Delta(\boldsymbol n)
\longrightarrow
\left\{\sum_qd_0(q)\sqrt{\kappa(q)}\right\}^2.}
\tag{21.5}
\]

In particular, the asymptotic count priority is

\[
\boxed{
n_q^*\ \propto\
\frac{\ell_0(x)L_0(q\mid x)}{\sqrt{\kappa(q)}}
\sqrt{\operatorname{Var}\!left\{
e^{\lambda C}\mathbf1\{X'\notin G_C\}z_0(X')\mid q
\right\}}.}
\tag{21.6}
\]

### Claim C: pilot plug-in stability

Use an independent pilot costing \(\mathsf B_0<\mathsf B\), and let the remaining
budget be \(\mathsf B_1=\mathsf B-\mathsf B_0\). If, on a pilot event,

\[
\max_q\left|
\widehat\alpha(q)/\alpha_\Delta(q)-1
\right|\le\varepsilon<1,
\tag{21.7}
\]

then applying (21.3) with \(\widehat\alpha\) gives

\[
\boxed{
1\le
\frac{\mathcal V_\Delta(\widehat{\boldsymbol n})}
{S_\Delta^2/\mathsf B_1}
\le\frac{1+\varepsilon}{1-\varepsilon},
\qquad
\frac{\mathcal V_\Delta(\widehat{\boldsymbol n})}
{S_\Delta^2/\mathsf B}
\le
\frac{\mathsf B}{\mathsf B_1}
\frac{1+\varepsilon}{1-\varepsilon}.}
\tag{21.8}
\]

Thus \(\mathsf B_0/\mathsf B\to0\) and uniform pilot relative error \(o_p(1)\)
are sufficient for oracle efficiency.

### Proof

With design mass \(\rho_Q(q)=n_q/n\), Theorem 18's one-observation log-risk
bound divided by \(n\) is exactly (21.2). Cauchy--Schwarz yields

\[
\mathcal V_\Delta(\boldsymbol n)
\sum_q\kappa(q)n_q
\ge
\left\{\sum_q\alpha_\Delta(q)\sqrt{\kappa(q)}\right\}^2.
\]

Equality holds precisely when
\(n_q\propto\alpha_\Delta(q)/\sqrt{\kappa(q)}\), proving Claim A. Theorem 20
gives
\(\eta^X_\Delta L/\theta_\Delta\to\ell_0L_0/b_0\) and
\(\Delta\sqrt{\operatorname{Var}(\Gamma\mid q)}
\to b_0\sigma_0(q)\), proving Claims B.

For Claim C, set
\(p_q=\alpha_\Delta(q)\sqrt{\kappa(q)}/S_\Delta\) and
\(r_q=\widehat\alpha(q)/\alpha_\Delta(q)\). Direct substitution gives the
second-stage oracle ratio

\[
\left(\sum_qp_qr_q\right)
\left(\sum_q\frac{p_q}{r_q}\right).
\]

It is at least one by Cauchy--Schwarz and no larger than
\((1+\varepsilon)/(1-\varepsilon)\). Accounting for the pilot budget proves
(21.8). ∎

### Status and interpretation boundary

**PROVABLE AS AN EXACT FINITE-STRATUM DESIGN THEOREM; CRITICAL LIMIT AND PILOT
PERTURBATION VERIFIED.** The non-symmetric two-state verifier reports a
\(1.18\%\) relative error for the predicted critical constant at its smallest
gap. None of 2,000 random feasible allocations beats the closed-form oracle. With
pilot sensitivity errors bounded by \(10\%\), the observed second-stage variance
ratio is \(1.0068\), below the analytic bound \(1.2222\); including a \(5\%\)
pilot budget gives ratio \(1.0598\).

- The closed-form optimizer is cost-aware Neyman allocation, a classical design
  principle. Standalone novelty is not claimed.
- The project-specific result identifies the correct allocation statistic as
  left-Perron target occupation times right-Perron projected conditional noise,
  not TD error or raw energy variance.
- The theorem assumes exact quotient strata and independent designed sampling.
  The oral-level open problem is an adaptive oracle inequality that supplies
  (21.7) while estimating the quotient, both Perron modes, and conditional tail
  variance as \(\Delta_n\downarrow0\).
- It does not authorize changing the formal navigation/Pareto Gate, and it is not
  evidence of an empirical stranding--throughput improvement.

## Theorem 22: Singularity-free adaptive critical-mode design

Adopt Theorems 20--21 on a finite known exact quotient. Let

\[
A(Y)=e^{\lambda C}\mathbf1\{X'\notin G_C\}e_{X'},
\qquad
d_\Delta(q)
=\ell_\Delta(x)L_\Delta(q\mid x)
\sqrt{\operatorname{Var}\{A(Y)^\top z_\Delta\mid q\}}.
\tag{22.1}
\]

Assume uniformly in small \(\Delta\): \(A\) is bounded; the simple Perron
eigenprojector has bounded reduced spectral resolvent
\(S_\Delta=(M_\Delta-(1-\Delta)I)^\#\), with effective separation
\(\mathfrak g_\Delta=\|S_\Delta\|^{-1}\ge\mathfrak g_0>0\); normalized
left/right Perron vectors stay in a compact positive set; and each leading active
\(d_\Delta(q)\) is bounded below. Let an independent pilot contain at least
\(m_{\min}\) primitives in each active quotient interface. Build the empirical
operator, its normalized Perron vectors, projected conditional variances, and
\(\widehat d_\Delta(q)\).

### Claim A: uniform pilot priority rate

For finite constants \(K_p,m_0\), independent of \(\Delta\),

\[
\Pr\left[
\max_q\left|
\frac{\widehat d_\Delta(q)}{d_\Delta(q)}-1
\right|
>K_p
\sqrt{\frac{\log\{2|\mathcal A|(D+2)/\delta\}}{m_{\min}}}
\right]
\le\delta
\tag{22.2}
\]

for \(m_{\min}\ge m_0\).

### Claim B: exact-priority approximation

There is \(K_c<\infty\), independent of \(\Delta\), such that

\[
\max_q
\left|
\frac{\Delta\alpha_\Delta(q)}{d_\Delta(q)}-1
\right|
\le K_c\Delta.
\tag{22.3}
\]

Hence, defining

\[
\varepsilon_{m,\Delta}(\delta)
=K\left[
\sqrt{\frac{\log\{2|\mathcal A|(D+2)/\delta\}}{m_{\min}}}
+\Delta
\right],
\tag{22.4}
\]

the allocation based directly on \(\widehat d_\Delta\) has the same proportions
as one based on \(\widehat d_\Delta/\Delta\), and approximates the exact oracle
priorities within relative error \(\varepsilon_{m,\Delta}\) with probability at
least \(1-\delta\).

### Claim C: adaptive oracle inequality and scale separation

Let the pilot cost \(\mathsf B_0<\mathsf B\). If
\(\varepsilon_{m,\Delta}(\delta)<1\), then with probability at least
\(1-\delta\),

\[
\boxed{
\frac{\mathcal V_\Delta(\widehat{\boldsymbol n})}
{\inf_{\boldsymbol n:\,\sum_q\kappa(q)n_q\le\mathsf B}
\mathcal V_\Delta(\boldsymbol n)}
\le
\frac{\mathsf B}{\mathsf B-\mathsf B_0}
\frac{1+\varepsilon_{m,\Delta}(\delta)}
{1-\varepsilon_{m,\Delta}(\delta)}.}
\tag{22.5}
\]

Thus \(m_{\min}\to\infty\), \(\mathsf B_0/\mathsf B\to0\), and
\(\Delta\to0\) imply adaptive oracle efficiency even when

\[
m_{\min}\Delta^2\to0.
\tag{22.6}
\]

The final log-risk value still requires total information on the scale
\(\mathsf B\Delta^2\), as in Theorem 20. Learning the normalized sampling design
is therefore strictly easier, in this regime, than resolving the risk value.

### Proof

Every entry of the empirical continuation operator and every required
conditional first/second moment is an average of bounded variables. A union of
Hoeffding bounds makes their maximum error
\(O\{\sqrt{\log(|\mathcal A|D/\delta)/m_{\min}}\}\). On the compact family in the
assumptions, the reduced-resolvent perturbation formula makes the normalized
left/right eigenvectors locally Lipschitz functions of the matrix entries.
Conditional
variance is locally Lipschitz in its two moments and in \(z_\Delta\); the positive
projected-variance and Perron-component floors make the square root and the final
relative normalization Lipschitz. This proves (22.2).

The uniform resolvent expansion in Theorem 20 gives
\(\eta^XL/\theta=\ell L/b+O(\Delta)\). Its witness decomposition gives
\(\Delta\sqrt{\operatorname{Var}(\Gamma\mid q)}
=b\sigma_\Delta(q)+O(\Delta)\) under the active variance floor. Multiplication
proves (22.3). Equations (22.2)--(22.3) give (22.4), and Theorem 21's pilot
stability bound proves (22.5). The common \(1/\Delta\) factor cancels from every
allocation proportion, so (22.6) is compatible with oracle design consistency.
∎

### Status and interpretation boundary

**PROVABLE FOR A KNOWN FINITE QUOTIENT WITH UNIFORMLY CONDITIONED PERRON
EIGENPROJECTOR; MONTE CARLO CONSTRUCTION VERIFIED.** In the verifier,
\(m_\Delta=\lceil20/\Delta\rceil\) increases from 400 to 6,400 per interface
while \(m_\Delta\Delta^2\) decreases from `1.0` to `0.0625`. Nevertheless, the
median oracle variance ratio improves from `1.00228` to `1.00009`, and every
reported q90 ratio is below `1.011`.

- Adaptive Neyman allocation and pilot-estimated stratum variances are classical;
  the closest-work audit includes Étoré--Jourdain and cost-aware stratified
  design. Generic adaptive allocation is not novel.
- The candidate new statement is the **critical-scale separation**: the common
  killed-resolvent singularity cancels from the normalized design, so pilot
  allocation learning need not pay the \(\Delta^{-2}\) risk-value information
  cost when the Perron reduced resolvent remains bounded.
- This theorem still assumes a known exact quotient and active variance floors.
  Learned quotient recovery, collapsing effective Perron separation,
  projected-noise
  phase changes, and dependent online replay are not covered.
- The theorem provides a reusable sampler target, not a navigation/Pareto result.

## Theorem 23: Conditioned collapse phase and local allocation lower bound

Let Theorem 22's fixed quotient now vary in a triangular family. Define the
Perron effective separation

\[
\mathfrak g_\Delta
=\left\|(M_\Delta-(1-\Delta)I)^\#\right\|^{-1},
\tag{23.1}
\]

and, for \(\Sigma_\Delta(q)=\operatorname{Cov}\{A(Y)\mid q\}\),

\[
\sigma_\Delta(q)^2
=z_\Delta^\top\Sigma_\Delta(q)z_\Delta,
\qquad
\chi_\Delta(q)
=\frac{\|\Sigma_\Delta(q)z_\Delta\|}
{\sigma_\Delta(q)^2}.
\tag{23.2}
\]

Let \(\varkappa_\Delta(q)\) and \(b_\Delta(q)\) be respectively the standardized
fourth moment and standardized essential range of the centered projected
primitive \(A^\top z_\Delta\). Write \(\chi_*,\varkappa_*,b_*\) for active
maxima and \(\sigma_*^2\) for the active minimum projected variance. Allow a
learned/proposed quotient to introduce operator-moment bias \(\tau_M\) and
fixed-direction projected-standard-deviation relative bias \(\tau_\sigma\).

### Claim A: conditioned upper phase

For

\[
r_m(\delta)
=\sqrt{\frac{\log(c|\mathcal A|D/\delta)}{m_{\min}}},
\tag{23.3}
\]

define

\[
\begin{aligned}
\Omega_{m,\Delta}(\delta)
=K\Bigg[&
\frac{(1+\chi_*)\{r_m(\delta)+\tau_M\}}
{\mathfrak g_\Delta}
+\sqrt{\frac{\varkappa_*\log(c|\mathcal A|/\delta)}{m_{\min}}}\\
&+\frac{b_*^2\log(c|\mathcal A|/\delta)}{m_{\min}}
+\tau_\sigma
+\frac{\Delta}{\sigma_*^2}
\Bigg].
\end{aligned}
\tag{23.4}
\]

Under bounded primitive moments, compact positive Perron normalizations, and the
local smallness conditions needed by reduced-resolvent perturbation, if
\(\Omega_{m,\Delta}(\delta)<1\), the critical-mode pilot allocation obeys

\[
\boxed{
\frac{\mathcal V_\Delta(\widehat{\boldsymbol n})}
{\inf_{\boldsymbol n:\,\mathrm{cost}\le\mathsf B}
\mathcal V_\Delta(\boldsymbol n)}
\le
\frac{\mathsf B}{\mathsf B-\mathsf B_0}
\frac{1+\Omega_{m,\Delta}(\delta)}
{1-\Omega_{m,\Delta}(\delta)}}
\tag{23.5}
\]

with probability at least \(1-\delta\). Thus
\(\Omega_{m,\Delta}(\delta)\to0\) and
\(\mathsf B_0/\mathsf B\to0\) are sufficient for adaptive oracle efficiency.

### Claim B: local effective-separation/anisotropy necessity

There is a two-state positive-matrix/covariance slice for which a scalar bounded
pilot tilt \(t\) satisfies

\[
\left.\frac{d}{dt}\log d_t\right|_{t=0}
=\frac{1+\chi}{\mathfrak g}.
\tag{23.6}
\]

For alternatives \(t_\pm=\pm a\mathfrak g/(1+\chi)\), their critical priorities
have constant-order log separation, while \(m\) bounded Rademacher pilot draws
have KL

\[
O\left(\frac{ma^2\mathfrak g^2}{(1+\chi)^2}\right).
\tag{23.7}
\]

Therefore any allocation-priority estimator has a nonvanishing local two-point
error floor if

\[
\boxed{
m\mathfrak g^2/(1+\chi)^2=O(1).}
\tag{23.8}
\]

Likewise, uniform recovery under deterministic quotient/operator distortion
requires \((1+\chi)\tau_M/\mathfrak g\to0\).

### Claim C: projected-noise collapse has no universal exponent

Vanishing \(\sigma_*\) alone does not determine pilot difficulty.

- If \(Z=\sigma G\) with \(G\sim N(0,1)\), one-sample Fisher information for
  \(\log\sigma\) equals two for every \(\sigma>0\).
- If \(Z\sim\operatorname{Bernoulli}(p)\), then
  \(\sigma^2=p(1-p)\), standardized fourth moment is asymptotic to \(1/p\), and
  \(mp=O(1)\) leaves a nonvanishing probability of no observed rare event.

Hence the Gaussian scale family needs only \(m\to\infty\) for relative scale
estimation, whereas the rare-Bernoulli family needs
\(m\sigma^2\to\infty\). A universal phase statement must retain
\(\varkappa_*\), \(b_*\), or a declared primitive distribution class.

### Proof

On the joint pilot moment event, the operator perturbation has norm
\(O\{r_m+\tau_M\}\). The standard simple-eigenprojector derivative through the
group inverse in (23.1) gives

\[
\|\widehat z-z\|+\|\widehat\ell-\ell\|
\le K\{r_m+\tau_M\}/\mathfrak g_\Delta.
\]

For \(s_q(z)=\sqrt{z^\top\Sigma(q)z}\),
\(\nabla\log s_q(z)=\Sigma(q)z/(z^\top\Sigma(q)z)\), whose norm is exactly
\(\chi_\Delta(q)\). Bernstein concentration applied to the squared centered
projected primitive gives the two moment terms in (23.4); quotient bias adds
\(\tau_\sigma\). Finally, Theorem 20's witness decomposition gives

\[
\Delta^2\operatorname{Var}(\Gamma\mid q)
=b_\Delta^2\sigma_\Delta(q)^2+O(\Delta),
\]

so converting the exact canonical sensitivity to the leading critical priority
costs relative order \(\Delta/\sigma_*^2\). Theorem 21 then proves (23.5).

For Claim B take

\[
U=2^{-1/2}\begin{bmatrix}1&1\\1&-1\end{bmatrix},
\quad
M_t=U\begin{bmatrix}\rho&t\\t&\rho-\mathfrak g\end{bmatrix}U^\top,
\quad
\Sigma_\chi
=U\begin{bmatrix}\chi^{-2}&\chi^{-1}\\\chi^{-1}&1\end{bmatrix}U^\top.
\tag{23.9}
\]

The leading eigenvector angle has derivative \(1/\mathfrak g\) at zero. Direct
differentiation of the scale-invariant first-state priority gives (23.6). A
Rademacher variable with mean \(t\) has KL against mean \(-t\) equal to
\(t\log\{(1+t)/(1-t)\}=2t^2+O(t^4)\), so Le Cam's lemma yields (23.8). Claim C
follows from the normal scale likelihood and the exact Bernoulli central moments.
∎

### Status and interpretation boundary

**COHERENT AFTER REFRAMING: SUFFICIENT PHASE PROVED UNDER EXPLICIT MOMENT
COORDINATES; LOCAL SPECTRAL LOWER SLICE PROVED; NO UNIVERSAL NOISE-ONLY PHASE
CLAIMED.** The verifier checks (23.6) to normalized error below \(10^{-6}\). At
\(m=(1+\chi)^2/\mathfrak g^2\), bounded-pilot KL stays near `0.02` and the
log-priority separation stays near `0.20`. In the rare-Bernoulli slice with
\(mp=1\), the probability of no event remains between `0.3487` and `0.3656`,
while standardized fourth moment diverges.

- This theorem corrects the vague phrase “collapsing eigengap”: for nonnormal
  operators the relevant coordinate is the Perron reduced-resolvent norm, which
  includes eigenvector/pseudospectral conditioning.
- The \(\Delta/\sigma_*^2\) term is a dominance condition for the leading
  critical-mode proxy, not a lower bound for every possible exact allocation
  estimator.
- The effective-separation/anisotropy lower slice is locally matching. The
  variance-moment part is deliberately distribution-class dependent.
- Learned quotient estimation rates and dependent online replay remain open, and
  no empirical Pareto improvement is implied.

## Theorem 24: Critical information-to-stopped-Pareto transfer

Let \(t=1,\ldots,J\) index a finite declared collection of frozen
oracle-shadow decision queries. Write

\[
s_t=b_t-m-U_t^{\rm eff},
\qquad
E_t=\widehat U_t^{\rm eff}-U_t^{\rm eff}.
\]

Assume the estimator is trained on data independent of the oracle-shadow
deployment cycle and, at every query, has the conditional tail bound

\[
\Pr(|E_t|>r\mid\mathcal H_t)
\le2\exp\{-r^2/(2V_t)\}.
\tag{24.1}
\]

Suppose the stopped margin law is

\[
\Pr_\star(t<\tau_{\rm end},|s_t|\le r)
\le C_t r^\kappa.
\tag{24.2}
\]

### Claim A: finite-confidence Pareto transfer

For \(\alpha\in(0,1)\), define

\[
\varepsilon_t(\alpha)
=\sqrt{2V_t\log(2J/\alpha)}.
\tag{24.3}
\]

Then the first-disagreement probability satisfies

\[
\boxed{
\Pr(\tau_\Delta<\tau_{\rm end})
\le
\alpha+
\{2\log(2J/\alpha)\}^{\kappa/2}
\sum_{t=1}^J C_tV_t^{\kappa/2}.}
\tag{24.4}
\]

Consequently the stranding deviation is at most the right-hand side of (24.4),
and the deviation of a throughput coordinate bounded by \(B_q\) is at most
\(B_q\) times that quantity. If Theorem 12's conditional positive boundary-loss
assumption holds, then

\[
\boxed{
\mathbb E[(L^{\widehat d}-L^{d^\star})_+]
\le B_L\alpha+L_D
\{2\log(2J/\alpha)\}^{(\kappa+1)/2}
\sum_{t=1}^J C_tV_t^{(\kappa+1)/2}.}
\tag{24.5}
\]

### Claim B: critical design consequence

For one scalar risk query with total acquisition budget \(\mathsf B\), let
Theorem 21's oracle variance be \(V_\Delta^*\). In the nondegenerate critical
phase,

\[
V_\Delta^*
=\frac{\mathcal J_0+o(1)}{\mathsf B\Delta^2},
\qquad
\mathcal J_0
=\left\{\sum_q
\ell_0(x)L_0(q\mid x)\sigma_0(q)\sqrt{\kappa(q)}
\right\}^2.
\tag{24.6}
\]

If Theorem 23 supplies an adaptive variance ratio

\[
R_\Omega
=\frac{\mathsf B}{\mathsf B-\mathsf B_0}
\frac{1+\Omega_{m,\Delta}}{1-\Omega_{m,\Delta}},
\tag{24.7}
\]

then the adaptive design's certified first-disagreement/Pareto radius is at most
\(R_\Omega^{\kappa/2}\) times the oracle-design radius, and its boundary-loss
radius is at most \(R_\Omega^{(\kappa+1)/2}\) times the oracle-design radius.
Ignoring confidence logarithms, the corresponding critical powers are

\[
\boxed{
(\mathsf B\Delta^2)^{-\kappa/2}
\quad\text{and}\quad
(\mathsf B\Delta^2)^{-(\kappa+1)/2}.}
\tag{24.8}
\]

Thus the pilot may learn normalized priorities with
\(m\Delta^2\to0\), but the final irreversible decision still needs
\(\mathsf B\Delta^2\to\infty\) for a vanishing regular error bound.

### Claim C: exact Gaussian-shift tightness slice

At one decision epoch let the oracle score \(S\) have symmetric sign and

\[
\Pr(|S|\le r)=(r/r_0)^\kappa,
\qquad 0\le r\le r_0.
\tag{24.9}
\]

Let the plug-in score be \(S-vG\), where \(G\sim N(0,1)\) is independent of
\(S\). Conditional on \(G\), its disagreement probability is

\[
\frac12\min\{(v|G|/r_0)^\kappa,1\},
\tag{24.10}
\]

and, when disagreement costs \(L_D|S|\), its conditional excess loss is

\[
\frac{L_D\kappa}{2(\kappa+1)r_0^\kappa}
\min\{v|G|,r_0\}^{\kappa+1}.
\tag{24.11}
\]

Therefore, as \(v/r_0\to0\), disagreement and excess loss are exactly of orders
\(v^\kappa\) and \(v^{\kappa+1}\), respectively. The powers in (24.8) are tight
for this regular plug-in slice.

### Proof

Equation (24.1) and a union bound show that
\(|E_t|\le\varepsilon_t(\alpha)\) simultaneously with probability at least
\(1-\alpha\). Theorem 12 then gives (24.4) and (24.5) after substituting (24.2).
Theorem 21 gives the first equality in (24.6); its Perron expansion gives the
second. Applying the powers \(\kappa/2\) and \((\kappa+1)/2\) to the variance
ratio (24.7) proves Claim B.

For Claim C, if the realized error is \(e>0\), disagreement occurs precisely
for \(0<S\le\min(e,r_0)\); for \(e<0\), the symmetric negative interval applies.
Equation (24.9) gives (24.10). Integrating \(L_D|S|\) against the corresponding
power-law margin density gives (24.11). Gaussian absolute moments are finite,
so dominated convergence after division by \(v^\kappa\) or
\(v^{\kappa+1}\) proves the asserted exact powers. ∎

### Status and interpretation boundary

**PROVED UNDER A DECLARED SUB-GAUSSIAN REGULAR-ESTIMATOR TAIL AND STOPPED MARGIN;
THE PLUG-IN POWERS ARE EXACT ON A GAUSSIAN-SHIFT SLICE.**

- Margin conversion and Gaussian plug-in classification are classical. The
  project-specific object is the transfer from quotient-aware critical Perron
  information design to the actual absorbing two-branch ReturnManager and its
  stranding--throughput coordinates.
- The result contracts a valid Pareto *uncertainty rectangle*; it does not prove
  that realized stranding decreases or throughput increases.
- Equation (24.6) is a scalar-query design. A single shared replay distribution
  for many decision queries requires a multi-objective design theorem and cannot
  be obtained by independently minimizing every \(V_t\).
- Theorem 23's quotient biases must enter both \(\Omega\) and any centering
  remainder in (24.1). Dependent online replay needs a martingale, mixing, or
  regeneration replacement for the assumed tail bound.
- Claim C is a tight witness for the plug-in powers, not a universal minimax
  lower bound over every possible decision algorithm.

## Theorem 25: Shared stopped-margin Perron information design

Let \(t=1,\ldots,J\) index frozen oracle-shadow risk queries and
\(q=1,\ldots,Q\) index source quotient strata. Suppose a regular estimator has

\[
V_t(\boldsymbol n)
=\sum_{q=1}^Q\frac{a_{tq}^2}{n_q},
\tag{25.1}
\]

where \(a_{tq}\) is its canonical stratum sensitivity. Let \(\omega_t>0\) be
declared stopped-occupation weights, \(c_q>0\) be acquisition costs, and
\(p>0\). Define

\[
\mathcal R_p(\boldsymbol n)
=\sum_{t=1}^J\omega_tV_t(\boldsymbol n)^p,
\qquad
\sum_qc_qn_q\le\mathsf B.
\tag{25.2}
\]

Theorem 24's first-disagreement/Pareto bound induces
\(p=\kappa/2\); its local boundary-loss bound induces
\(p=(\kappa+1)/2\).

### Claim A: global convexity and the exact shared allocation

For every \(p>0\), (25.2) is convex on \(n_q>0\). Every active optimum spends
the full budget and is characterized by

\[
\boxed{
n_q^*
=\mathsf B
\frac{g_q(\boldsymbol n^*)}
{\sum_rg_r(\boldsymbol n^*)c_r},
\quad
g_q(\boldsymbol n)
=\left[
\frac{\sum_t\omega_tV_t(\boldsymbol n)^{p-1}a_{tq}^2}{c_q}
\right]^{1/2}.}
\tag{25.3}
\]

For \(p=1\), the fixed point disappears:

\[
\boxed{
n_q^*
=\mathsf B
\frac{\sqrt{\sum_t\omega_ta_{tq}^2/c_q}}
{\sum_r\sqrt{c_r\sum_t\omega_ta_{tr}^2}}.}
\tag{25.4}
\]

For \(J=1\), (25.3) reduces to Theorem 21's scalar cost-aware Neyman rule for
every \(p>0\).

### Claim B: common-critical cancellation and heterogeneous-critical emphasis

If all queries share a critical family and

\[
a_{tq,\Delta}=d_{tq}\Delta^{-1}\{1+o(1)\},
\tag{25.5}
\]

then the common \(\Delta^{-2p}\) factor cancels from the normalized allocation
in (25.3). If query \(t\) instead has its own \(\Delta_t\), then, writing
\(V_t=\Delta_t^{-2}\widetilde V_t\),

\[
g_q(\boldsymbol n)^2
\asymp
\frac1{c_q}
\sum_t\omega_t\Delta_t^{-2p}
\widetilde V_t(\boldsymbol n)^{p-1}d_{tq}^2.
\tag{25.6}
\]

Thus a shared sampler automatically emphasizes queries closer to exponential
transience, with exponent determined by the stopped margin objective.

### Proof

It suffices to prove convexity of
\(f(\boldsymbol n)=\{\sum_qb_q/n_q\}^p\) for \(b_q\ge0\). Put
\(S=\sum_qb_q/n_q\). For any direction \(u\),

\[
u^\top\nabla^2fu
=pS^{p-2}\left[
(p-1)\left\{\sum_q\frac{b_qu_q}{n_q^2}\right\}^2
+2S\sum_q\frac{b_qu_q^2}{n_q^3}
\right].
\tag{25.7}
\]

Cauchy--Schwarz makes the squared term no larger than
\(S\sum_qb_qu_q^2/n_q^3\). For \(p\ge1\), (25.7) is nonnegative directly. For
\(0<p<1\), multiplying the Cauchy bound by \(p-1\) gives a lower bound of
\((p+1)S\sum_qb_qu_q^2/n_q^3\) inside the brackets. Hence each summand and
therefore (25.2) is convex.

The objective is decreasing in every active \(n_q\), so the budget is tight.
Differentiating the Lagrangian gives

\[
\lambda c_q
=\frac p{n_q^2}
\sum_t\omega_tV_t^{p-1}a_{tq}^2.
\]

Normalizing this equation by the tight budget proves (25.3); setting \(p=1\)
proves (25.4). Under (25.5),
\(V_t=\Delta^{-2}\widetilde V_t\{1+o(1)\}\), so every squared score in (25.3)
has the same \(\Delta^{-2p}\) multiplier and it cancels. Keeping
query-specific \(\Delta_t\) gives (25.6). ∎

### Status and interpretation boundary

**EXACT CONTINUOUS-ALLOCATION THEOREM; NUMERICALLY VERIFIED; NOT STANDALONE
NOVELTY.**

- Convex multi-characteristic stratified allocation, compound optimal design,
  multiple-logger efficient OPE, and behavior-policy variance search are prior
  art. The candidate specialization is the stopped-margin exponent combined
  with killed-Perron sensitivities and the critical cancellation/failure phase.
- The replay target is (25.3), not TD error or an intervention-norm heuristic.
  In a learned implementation, sampling probabilities must be accompanied by
  the appropriate estimator correction; changing replay alone can change the
  estimand.
- Integer counts require rounding. Adaptive estimation of the full sensitivity
  matrix, stopped-occupation weights, and query-specific gaps remains open.
- If online strata are reachable only through survival-selected trajectories,
  (25.1) does not hold without a valid trajectory-level influence analysis.
- The theorem minimizes an upper-bound objective. It does not imply observed
  Pareto dominance or replace the original formal Gate.

## Theorem 26: Finite risk-observable quotient recovery and impossibility

Let \(\mathcal U\) be a finite raw executed-interface alphabet of size \(N\).
For every declared risk level, stack into
\(W(Y)\in[-B_W,B_W]^D\) all terminal/continuation coordinates and first/second
moments needed to construct the killed operators and projected conditional
variances. Define

\[
\mu(u)=\mathbb E[W(Y)\mid u],
\qquad
u\sim_Q u'\iff\mu(u)=\mu(u').
\tag{26.1}
\]

Assume distinct equivalence classes obey

\[
\min_{u\not\sim_Q u'}
\|\mu(u)-\mu(u')\|_\infty\ge\gamma>0.
\tag{26.2}
\]

An independent structure fold supplies \(m_u\) observations at raw interface
\(u\), with \(m_{\min}=\min_um_u\).

### Claim A: exact recovery

Let \(\widehat\mu(u)\) be the coordinatewise empirical mean and

\[
r_Q(\delta)
=B_W\sqrt{\frac{2\log(2ND/\delta)}{m_{\min}}}.
\tag{26.3}
\]

Join \(u,u'\) when
\(\|\widehat\mu(u)-\widehat\mu(u')\|_\infty<\gamma/2\), and return the graph's
connected components. If \(r_Q(\delta)<\gamma/4\), this estimator equals the
exact quotient with probability at least \(1-\delta\). Therefore

\[
\boxed{
m_{\min}
\gtrsim
\frac{B_W^2}{\gamma^2}
\log\frac{ND}{\delta}}
\tag{26.4}
\]

is sufficient. Conditional on recovery, independent allocation/value folds may
apply Theorems 23--25 with \(\tau_M=\tau_\sigma=0\).

### Claim B: matching separation lower bound

There is a two-interface one-witness Bernoulli family for which every quotient
estimator has nonvanishing worst-case error when \(m\gamma^2=O(1)\). In model
zero both witness laws are \(\operatorname{Bernoulli}(1/2)\); in model one the
second is \(\operatorname{Bernoulli}(1/2+\gamma)\). The correct quotients merge
and separate the two interfaces, respectively, while

\[
\operatorname{KL}(P_0^m,P_1^m)
=-\frac m2\log(1-4\gamma^2)
=2m\gamma^2+O(m\gamma^4).
\tag{26.5}
\]

Thus the \(\gamma^{-2}\) order in (26.4) is necessary up to logarithms and
constants.

### Claim C: neural representation corollary

If an encoder/decoder pair satisfies

\[
\sup_{u\in\mathcal U}
\|g_\theta(e_\theta(u))-\mu(u)\|_\infty<\gamma/4,
\tag{26.6}
\]

then the same \(\gamma/2\) threshold recovers the exact quotient from decoded
risk-witness moments. Consequently a theorem-aligned representation loss must
supervise the stacked conditional moments in (26.1); raw-input reconstruction,
action labels, or Euclidean contrastive proximity alone do not imply risk
quotient recovery.

### Proof

Hoeffding's inequality and a union bound over \(ND\) coordinates give

\[
\max_u\|\widehat\mu(u)-\mu(u)\|_\infty\le r_Q(\delta)
\]

with probability at least \(1-\delta\). On this event, same-class empirical
means differ by less than \(2r_Q<\gamma/2\), whereas different-class means
differ by more than \(\gamma-2r_Q>\gamma/2\). The graph therefore consists
exactly of the true equivalence classes, proving Claim A.

For Claim B, only the second interface changes between the two models. Direct
Bernoulli calculation gives (26.5). Pinsker followed by Le Cam's testing bound
keeps the sum of merge/separate errors bounded away from zero whenever
\(m\gamma^2\) is bounded. Claim C repeats Claim A's deterministic triangle
inequality with the decoded approximation in place of empirical means. ∎

### Status and interpretation boundary

**FINITE QUOTIENT RATE PROVED WITH MATCHING SEPARATION ORDER; NUMERICALLY
VERIFIED; NOT A CONTINUOUS-NEURAL GENERALIZATION THEOREM.**

- Mean concentration, graph clustering under separation, and the two-point lower
  bound are classical. The useful specialization is the exact witness stack that
  preserves the killed-risk operator, conditional covariance, and hence the
  Perron replay sensitivities.
- The separation \(\gamma\) or a consistent way to choose the empirical gap is
  required. Without separation, exact quotient labels are statistically
  nonregular; an approximate quotient must retain a nonzero \(\tau_M\) in
  Theorem 23.
- Structure selection and allocation/value estimation use independent folds.
  Reusing outcome data needs uniform post-selection control; naive reuse is not
  covered.
- In continuous interfaces, \(N\) must be replaced by metric entropy, RKHS
  effective dimension, neural complexity, or another declared regularity class.
- Claim C specifies a valid neural loss target; it does not say that a particular
  architecture can attain (26.6) from the current data.

## Theorem 27: Continuous quotient--spectral--stopping joint phase

Let \(z_1,\ldots,z_J\) be a fixed finite set of oracle-shadow executed-interface
queries in a metric space \((\mathsf Z,d_Z)\). An independent structure fold
contains iid observations \((Z_i,W_i)_{i=1}^n\), with
\(W_i\in[-B_W,B_W]^D\), and

\[
\mu(z)=\mathbb E[W\mid Z=z].
\tag{27.1}
\]

Assume \(\mu\) is \(\alpha\)-Hölder with constant \(L_\mu\), and that for every
query and \(0<h\le h_0\),

\[
\Pr\{d_Z(Z,z_t)\le h\}\ge c_Qh^{d_Q}.
\tag{27.2}
\]

The vector \(W\) contains the declared first/second risk-witness moments from
Theorem 26. Suppose their map to the killed operator and projected-variance
coordinates is \(C_M\)-Lipschitz. The query set, witness dictionary, bandwidth,
and pooling threshold are fixed independently of the structure fold.

### Claim A: attainable continuous point-query radius

Let \(\widehat\mu_h(z_t)\) be the average of \(W_i\) over
\(d_Z(Z_i,z_t)\le h\). If

\[
n c_Qh^{d_Q}\ge8\log(2J/\delta),
\tag{27.3}
\]

then, with probability at least \(1-\delta\),

\[
\boxed{
\max_{t\le J}\|\widehat\mu_h(z_t)-\mu(z_t)\|_\infty
\le
\varepsilon_n(h,\delta)
:=
L_\mu h^\alpha+
2B_W\sqrt{
\frac{\log(4JD/\delta)}{n c_Qh^{d_Q}}}.}
\tag{27.4}
\]

The balanced bandwidth and radius obey

\[
h_n\asymp
\left\{\frac{B_W^2\log(4JD/\delta)}
{c_QL_\mu^2n}\right\}^{1/(2\alpha+d_Q)},
\qquad
\varepsilon_n
=\widetilde O\{n^{-\alpha/(2\alpha+d_Q)}\}.
\tag{27.5}
\]

For \(\mathfrak d_W(z,z')=\|\mu(z)-\mu(z')\|_\infty\) and its plug-in version,

\[
|\widehat{\mathfrak d}_W(z_t,z_s)-\mathfrak d_W(z_t,z_s)|
\le2\varepsilon_n.
\tag{27.6}
\]

Thus pooling only query pairs with
\(\widehat{\mathfrak d}_W\le\rho_n\) has true moment distortion at most
\(\rho_n+2\varepsilon_n\), and

\[
\tau_{M,n}\le C_M(\rho_n+2\varepsilon_n).
\tag{27.7}
\]

### Claim B: joint critical and stopped-decision phase

Suppose the requirement-error amplification in Theorems 13 and 19 is at most
\(A_0/\Delta_n\), where \(\Delta_n\) is the exponential-transience margin.
Under Theorem 24's stopped-margin law,

\[
\Pr(\tau_\Delta<\tau_{\rm end})
\le\delta+
C\left\{\frac{\rho_n+\varepsilon_n}{\Delta_n}\right\}^{\kappa},
\tag{27.8}
\]

and its local boundary-weighted excess loss is at most

\[
B_L\delta+
C_L\left\{\frac{\rho_n+\varepsilon_n}{\Delta_n}\right\}^{\kappa+1}.
\tag{27.9}
\]

For Theorem 23's critical replay priority, the quotient-estimation contribution
to the adaptive-oracle radius is

\[
\Omega_{Q,n}
\le
K\frac{(1+\chi_*)C_M(\rho_n+2\varepsilon_n)}
{\mathfrak g_n},
\tag{27.10}
\]

where \(\mathfrak g_n\) is the Perron effective separation. Consequently, up to
logarithms and Theorem 23's separately retained variance-moment terms, the joint
vanishing phase is

\[
\boxed{
\rho_n=o\{\min(\Delta_n,\mathfrak g_n)\},\quad
n\Delta_n^{(2\alpha+d_Q)/\alpha}\to\infty,\quad
n\mathfrak g_n^{(2\alpha+d_Q)/\alpha}\to\infty.}
\tag{27.11}
\]

### Claim C: distributed first-passage Assouad necessity

The earlier point-query Le Cam bump is insufficient by itself for a random
stopped-law lower bound: it does not construct the \(s^\kappa\) mass of decision
queries near the boundary. The following single statistical experiment supplies
that missing coupling.

Assume the standard compatible-margin regime

\[
\alpha\kappa\le d_Q.
\tag{27.12}
\]

Fix a scalar killed-chain cost multiplier \(A=e^{\lambda c}>1\). For a
transience sequence \(\Delta_n\in(0,\Delta_0]\), put

\[
h_n=c_h n^{-1/(2\alpha+d_Q)},
\qquad
s_n=c_s\frac{h_n^\alpha}{\Delta_n},
\qquad
0<s_n\le r_0\wedge1.
\tag{27.13}
\]

Let

\[
p_{0,n}=\frac{A-1+\Delta_n}{A},
\qquad
\psi_{0,n}=\frac{A p_{0,n}}{\Delta_n},
\tag{27.14}
\]

and define two exact primitive termination probabilities by inverting the
one-state charger-hitting requirement:

\[
p_{\pm,n}
=
\frac{(A-1)\psi_{0,n}e^{\mp\lambda s_n}}
{A\{\psi_{0,n}e^{\mp\lambda s_n}-1\}}.
\tag{27.15}
\]

For

\[
U(p)=\lambda^{-1}\log
\frac{Ap}{1-A(1-p)},
\tag{27.16}
\]

equation (27.15) gives the exact score identity

\[
U(p_{0,n})-U(p_{\pm,n})=\pm s_n.
\tag{27.17}
\]

For sufficiently small fixed \(c_s,\Delta_0,r_0\), all three termination
probabilities stay in one compact subset of \((0,1)\), their transience gaps
are comparable to \(\Delta_n\), and constants independent of \(n\) satisfy

\[
c_-\Delta_ns_n
\le |p_{+,n}-p_{-,n}|
\le c_+\Delta_ns_n.
\tag{27.18}
\]

Take the standard strong-density Hölder--margin Assouad hypercube on
\([0,1]^{d_Q}\): it has \(m_n\asymp s_n^\kappa h_n^{-d_Q}\) disjoint active
cells of source and target mass \(\asymp h_n^{d_Q}\), score signs indexed by
\(\sigma\in\{-1,+1\}^{m_n}\), score magnitude \(s_n\) on their plateau cores,
and

\[
\nu_{\rm stop}\{0<|S_\sigma|\le t\}
\le C_0t^\kappa.
\tag{27.19}
\]

Map its signed score bumps through (27.15). Because (27.18) and (27.13) give
\(|p_{+,n}-p_{-,n}|=\Theta(h_n^\alpha)\), the resulting primitive conditional
law has an \(n\)-uniform \(\alpha\)-Hölder constant. Source observations are iid
context--primitive-transition pairs; the source and target context laws are the
same in this lower slice. If \(\sigma^{(j)}\) flips one active cell, bounded
Bernoulli curvature gives

\[
\operatorname{KL}
\{P_\sigma^n,P_{\sigma^{(j)}}^n\}
\le
C n h_n^{d_Q}|p_{+,n}-p_{-,n}|^2
\le Cc_h^{2\alpha+d_Q}.
\tag{27.20}
\]

Choosing \(c_h\) so that the last constant is below a fixed Assouad testing
threshold yields, for every learned manager based on the declared source data,

\[
\boxed{
\inf_{\widehat d}\sup_\sigma
\Pr_\sigma(\widehat d\ne d_\sigma^\star)
\ge c_D\min\{1,s_n^\kappa\},}
\tag{27.21}
\]

and

\[
\boxed{
\inf_{\widehat d}\sup_\sigma
\mathbb E_\sigma
[|S_\sigma|\mathbf1\{\widehat d\ne d_\sigma^\star\}]
\ge c_L\min\{1,s_n^{\kappa+1}\}.}
\tag{27.22}
\]

Apply Theorem 15's one-decision mission embedding independently on every active
cell. A wrong `RETURN` on an oracle-`CONTINUE` cell produces normalized
throughput regret, while a wrong `CONTINUE` on an oracle-`RETURN` cell produces
stranding excess. Their sum is the disagreement mass, so at least one coordinate
is half that mass. Hence

\[
\boxed{
\inf_{\widehat d}\sup_\sigma
\max\{\mathcal R_{\rm throughput},\mathcal R_{\rm strand}\}
\ge c_P\min\{1,s_n^\kappa\}.}
\tag{27.23}
\]

Since \(s_n\asymp n^{-\alpha/(2\alpha+d_Q)}/\Delta_n\), vanishing of these
lower scales requires

\[
n\Delta_n^{(2\alpha+d_Q)/\alpha}\to\infty.
\tag{27.24}
\]

Thus the quotient-dimension--transience--margin--Pareto phase is realized in one
common exact-chart, source-equals-target first-passage experiment. In a separate
two-state subfamily from Theorem 23, an indistinguishable operator bump changes
log priority by order \(h_n^\alpha/\mathfrak g_n\); the
\(\mathfrak g_n\) necessity remains a separate output family rather than being
silently folded into (27.21)--(27.23).

### Proof

For each query, Chernoff's lower-tail inequality and (27.2) imply that the
local count is at least \(nc_Qh^{d_Q}/2\). A union bound proves this
simultaneously under (27.3). Conditional on the selected covariates,
coordinatewise Hoeffding concentration gives the second term in (27.4), while
Hölder continuity bounds the ball-averaging bias by \(L_\mu h^\alpha\).
Balancing these terms proves (27.5).

The reverse triangle inequality proves (27.6); (27.7) follows from the
\(C_M\)-Lipschitz moment-to-operator map. The exact residual identity in
Theorem 13 and the positive-resolvent log-grid propagation turn (27.7) into a
requirement radius of order
\((\rho_n+\varepsilon_n)/\Delta_n\). Theorem 24 then gives (27.8)--(27.9).
Substitution into Theorem 23 gives (27.10), and (27.5) yields (27.11).

For Claim C, solving \(\psi=Ap/\{1-A(1-p)\}\) for \(p\) proves (27.15)--
(27.17). The derivative

\[
\frac{dp}{dU}
=-\frac{\lambda p\{1-A(1-p)\}}{A-1}
\]

and compactness of the local probability interval prove (27.18). The standard
Hölder--margin hypercube exists under (27.12); its active-cell count is feasible
because \(m_nh_n^{d_Q}\asymp s_n^\kappa\le1\), while
\(m_n\ge1\) in the nontrivial compatible regime. Equation (27.18) turns the
primitive bump height into \(\Theta(h_n^\alpha)\), so the conditional law remains
uniformly Hölder and the Bernoulli KL calculation gives (27.20). Assouad's lemma
lower-bounds the expected number of incorrectly decoded cell signs by a fixed
fraction of \(m_n\). Multiplying by cell mass and, for boundary loss, by
\(s_n\), proves (27.21)--(27.22). The cellwise mission embedding and
\(\max(x,y)\ge(x+y)/2\) prove (27.23). Rearranging \(s_n\to0\) proves
(27.24). The separate two-state priority construction is Theorem 23. ∎

### Status and interpretation boundary

**PROVED FOR A DECLARED IID LOCAL-MASS HÖLDER QUERY SLICE; DISTRIBUTED
FIRST-PASSAGE ASSOUAD LOWER PHASE PROVED IN THE EXACT-CHART SOURCE=TARGET SLICE;
NUMERICALLY VERIFIED; UNKNOWN-CHART NEURAL ENCODERS AND GENERAL COVARIATE SHIFT
REMAIN OPEN.**

- The theorem does not recover continuum equivalence classes. It estimates the
  decision-relevant pseudometric at a predeclared finite query set and uses a
  distortion sandwich; threshold proximity is not assumed transitive.
- Local averaging, Hölder minimax rates, Le Cam bumps, plug-in margin transfer,
  and spectral perturbation are classical. The candidate contribution is the
  joint quotient-dimension--Perron-separation--risk-transience--stopped-boundary
  phase and its two project-aligned outputs.
- ICML 2025 KROPE already establishes representation stability and Bellman
  completeness for bisimulation-based offline value learning. Generic stable
  representation learning is therefore not a novelty claim here.
- The intrinsic chart \(Z\), \(d_Q\), local-mass law, and iid structure fold are
  assumptions. A practical neural theorem must control chart/decoder selection,
  witness-class approximation, and survival-selected dependence.
- Formal Oracle headroom and empirical Pareto improvement remain unavailable
  until the selected R3 platform completes its version-2 100-run continuous-
  workload endurance validation. The first 100-run audit is retained as an
  invalid right-censored result: 41 runs stopped at a task-step limit instead of
  true energy exhaustion.

## Theorem 28: Risk-neutral collapse and oracle-transform equivalence boundary

### Claim A: a risk-neutral bisimulation representation can erase the return decision

Consider two executed interfaces that enter the charger after one step and have
the same expected resource cost. At interface \(q_0\), let \(C=1\) surely; at
\(q_1\), let \(C\in\{0,2\}\) with equal probabilities. Their expected immediate
costs and next-state laws are identical, so an expected-reward
policy-bisimulation/KROPE objective may assign zero distance. For every
\(\lambda>0\), however,

\[
\mathbb E[e^{\lambda C}\mid q_1]
-\mathbb E[e^{\lambda C}\mid q_0]
=\frac{1+e^{2\lambda}}2-e^\lambda
=\frac{(e^\lambda-1)^2}{2}>0.
\tag{28.1}
\]

Hence the risk-witness pseudometric separates the interfaces. Choosing a
ReturnManager requirement threshold strictly between their two log-MGFs makes
the exact return decisions different. Thus a risk-neutral representation loss
is not sufficient even in a one-step charger-hitting problem.

### Claim B: raw support can diverge while risk-quotient support stays exact

Let the raw interface be \(Q=(Z,V)\), where primitive risk witnesses depend on
\(Z\) but are conditionally independent of nuisance \(V\). In a finite
approximation with \(K\) nuisance categories, let the source make \(V\) uniform
and the target put all mass on one category. Then

\[
\chi^2(P_T^Q\|P_S^Q)=K-1,
\qquad
\chi^2(P_T^Z\|P_S^Z)=0.
\tag{28.2}
\]

The first quantity diverges as \(K\to\infty\), approaching a singular raw
target, while the risk-quotient conditional law and value remain identified.
This proves a strict coverage advantage over raw-interface importance
weighting. It does not separate EIRR from a baseline that first uses the same
valid quotient.

### Claim C: an oracle risk transform returns to standard discounted OPE

For every finite killed Feynman--Kac operator \(M_\lambda\) with
\(\rho(M_\lambda)<1\), Theorem 6B constructs a positive diagonal \(D_v\), a
discount \(\gamma\in(\rho(M_\lambda),1)\), and a Markov kernel
\(P_{\gamma,S}\) such that

\[
M_\lambda
=\gamma D_vP_{\gamma,S}D_v^{-1},
\qquad
(I-M_\lambda)^{-1}
=D_v(I-\gamma P_{\gamma,S})^{-1}D_v^{-1}.
\tag{28.3}
\]

Therefore an oracle supplied with the true risk transform can apply standard
discounted bisimulation/KROPE or DICE machinery to the transformed chain.
Claims A--B justify the risk witness and quotient against risk-neutral or raw
baselines, but they do not establish superiority over this oracle-transformed
baseline. Any remaining learning contribution must control estimation of the
unknown transform under source-to-target execution shift and carry that error
to the irreversible stopped boundary.

### Proof

Claim A is direct calculation; (28.1) is half a square. Claim B follows from
\(\sum_v P_T(v)^2/P_S(v)-1=K-1\), while quotienting out \(V\) makes source and
target \(Z\)-laws identical. Claim C is exactly the positive diagonal
similarity and occupancy reduction proved in Theorem 6B. ∎

### Status and interpretation boundary

**EXACT SEPARATION/EQUIVALENCE BOUNDARY PROVED AND NUMERICALLY VERIFIED.**

- Ordinary expected-reward KROPE is an invalid risk baseline for the theorem
  claim, but risk-transformed KROPE is mandatory.
- Raw-interface DICE may have infinite or exploding coverage while
  quotient-aware DICE remains valid; only the former is strictly separated.
- This theorem narrows rather than inflates novelty. The defensible open target
  is adaptive unknown-transform estimation plus the joint phase in Theorem 27.

## Theorem 29: Unknown Doob transform is a rate-invariant reparameterization

Fix \(\gamma\in(0,1)\) and let \(\mathcal M_\gamma\) be a finite-dimensional
class of nonnegative killed operators \(M\) with \(\rho(M)<\gamma\). Define

\[
v(M)=(I-M/\gamma)^{-1}\mathbf1,
\quad
P_M(i,j)=\frac{M(i,j)v_j}{\gamma v_i},
\quad
P_M(i,\dagger)=\frac1{v_i}.
\tag{29.1}
\]

### Claim A: exact parameter and plug-in equivalence

The map

\[
\mathfrak T_\gamma:M\longmapsto\{v(M),P_M\}
\tag{29.2}
\]

is one-to-one on \(\mathcal M_\gamma\). Its inverse is

\[
v_i=\frac1{P_M(i,\dagger)},
\qquad
M(i,j)=\gamma\,\frac{v_iP_M(i,j)}{v_j}.
\tag{29.3}
\]

For any admissible estimate \(\widehat M\), terminal vector \(\widehat r\), and
the transform \((\widehat v,\widehat P)=\mathfrak T_\gamma(\widehat M)\),

\[
\boxed{
(I-\widehat M)^{-1}\widehat r
=D_{\widehat v}
(I-\gamma\widehat P_S)^{-1}
D_{\widehat v}^{-1}\widehat r.}
\tag{29.4}
\]

Consequently, for any statistical experiment
\(\{\mathbb P_M:M\in\mathcal M_\gamma\}\), relabeling the parameter by
\(\mathfrak T_\gamma(M)\) leaves every minimax decision risk for a functional
of \(M\) unchanged. The raw observations, estimator class, and loss have not
changed; only the parameter coordinates have.

### Claim B: transform conditioning cannot remove the critical scale

The Fréchet derivative of \(v\) in direction \(E\) is

\[
Dv_M[E]
=(I-M/\gamma)^{-1}\frac{E}{\gamma}v.
\tag{29.5}
\]

If \(\|M\|_\infty/\gamma\le1-s\) for \(s>0\), then

\[
\|Dv_M[E]\|_\infty
\le\frac{\|E\|_\infty}{\gamma s^2}.
\tag{29.6}
\]

For \(\|\widehat M-M\|_\infty=e<\gamma s\), the finite perturbation bound is

\[
\|\widehat v-v\|_\infty
\le\frac{e}{s(\gamma s-e)}.
\tag{29.7}
\]

The scalar family \(M=\gamma(1-s)\) makes the conditioning exact:

\[
v=\frac1s,\qquad
\frac{\widehat v-v}{v}
=\frac{e}{\gamma s-e},
\qquad
\frac{d v}{dM}=\frac1{\gamma s^2},
\qquad
\frac{d\log v}{dM}=\frac1{\gamma s}.
\tag{29.8}
\]

Thus absolute transform-vector estimation pays an inverse-square slack, while
relative estimation pays an inverse slack. These singularities can cancel in
the final value through (29.4); bounding \(\widehat v\) and \(\widehat P\)
separately can therefore be strictly more pessimistic than direct resolvent
analysis.

### Claim C: near transience, the learned transform has the same phase

In the scalar stopped-risk model let \(M=1-\Delta\) and choose
\(\gamma=1-c\Delta\) for fixed \(c\in(0,1)\). Then

\[
\gamma-M=(1-c)\Delta,
\qquad
s=\frac{(1-c)\Delta}{\gamma}.
\tag{29.9}
\]

Equation (29.8) shows that consistent relative transform learning requires
\(e=o(\Delta)\). Direct log-risk estimation has

\[
\frac{d}{dM}\log\{r/(1-M)\}=\frac1\Delta,
\tag{29.10}
\]

and therefore has the same local requirement. If Theorem 27 supplies
\(e_n=\widetilde O(n^{-\alpha/(2\alpha+d_Q)})\), both coordinates require

\[
n\Delta^{(2\alpha+d_Q)/\alpha}\to\infty
\tag{29.11}
\]

up to logarithms. The stopped-margin powers in Theorem 27 are unchanged.

### Proof

Equation (29.3) directly inverts (29.1), proving injectivity. The diagonal
similarity in Theorem 6B applied to \(\widehat M\) proves (29.4). A bijective
parameter relabeling induces a bijection between decision rules written in the
two coordinate systems, proving minimax-risk invariance.

Differentiate
\((I-M/\gamma)v=\mathbf1\) to obtain (29.5). The Neumann bound
\(\|(I-M/\gamma)^{-1}\|_\infty\le1/s\) and
\(\|v\|_\infty\le1/s\) prove (29.6). The resolvent identity, together with
\(\|(I-\widehat M/\gamma)^{-1}\|_\infty
\le1/(s-e/\gamma)\), proves (29.7). Scalar substitution gives (29.8), and
(29.9)--(29.10) prove Claim C. ∎

### Status and interpretation boundary

**EXACT REPARAMETERIZATION AND CONDITIONING BOUNDARY PROVED; NUMERICALLY
VERIFIED; LEARNED-DOOB ALONE IS NOT A NOVELTY ROUTE.**

- A transformed objective may still improve computation, regularization, or
  optimization. Those are algorithm-dependent effects, not extra statistical
  information from the coordinate change.
- Oracle transformed KROPE/DICE remains a mandatory baseline. A learned
  transformed method must be compared with direct primitive plug-in using the
  same data and nuisance privileges.
- The viable theorem target is now narrower: quotient learning or
  decision-localized estimation must reduce effective complexity or loss. Merely
  estimating \(v\) and \(P_M\) cannot improve the minimax order.

## Theorem 30: Predictable-interface martingale quotient certificate

Let \((\mathcal H_i)_{i\ge0}\) be the chronological data filtration. Before
primitive outcome \(W_i\in[-B_W,B_W]^D\) is generated, suppose the executed
interface representation \(Z_i\) is \(\mathcal H_{i-1}\)-measurable and

\[
\mathbb E[W_i\mid\mathcal H_{i-1}]
=\mu(Z_i).
\tag{30.1}
\]

The policy, safety filter, and interface sequence may depend arbitrarily on past
outcomes. Fix query centers \(z_1,\ldots,z_J\) and bandwidth \(h\) independently
of the evaluated outcomes, and assume
\(\mu\) is \(\alpha\)-Hölder with constant \(L_\mu\).

For query \(t\), let \(\tau_{t,k}\) be the chronological index of its \(k\)-th
unique transition satisfying \(d_Z(Z_i,z_t)\le h\). On the event that every
query reaches \(m\) hits, define

\[
\widehat\mu_{m,h}(z_t)
=\frac1m\sum_{k=1}^m W_{\tau_{t,k}}.
\tag{30.2}
\]

### Claim A: dependence-robust simultaneous radius

With probability at least \(1-\delta\),

\[
\boxed{
\max_{t\le J}
\|\widehat\mu_{m,h}(z_t)-\mu(z_t)\|_\infty
\le
\varepsilon^{\rm mart}_{m,h}(\delta)
:=
L_\mu h^\alpha
+B_W\sqrt{\frac{2\log(2JD/\delta)}{m}}.}
\tag{30.3}
\]

No mixing coefficient is required. The bound uses dependence only through the
observed time needed to acquire \(m\) fresh eligible transitions.

### Claim B: the continuous joint phase survives chronological dependence

Substituting \(\varepsilon^{\rm mart}_{m,h}\) for \(\varepsilon_n\) in
Theorem 27 preserves its pseudometric sandwich, operator bias, stopped-margin,
and critical-priority conclusions. In particular,

\[
\rho+L_\mu h^\alpha+
B_W\sqrt{\frac{\log(JD/\delta)}m}
=o\{\min(\Delta,\mathfrak g)\}
\tag{30.4}
\]

is the quotient-estimation part of the joint phase. If a separate predictable
coverage condition supplies

\[
m\asymp n c_Qh^{d_Q},
\tag{30.5}
\]

balancing (30.3) recovers Theorem 27's
\(\widetilde O(n^{-\alpha/(2\alpha+d_Q)})\) rate and matching phase powers.
Since iid source transitions are a subclass of (30.1), Theorem 27's local lower
orders continue to apply.

### Claim C: replay duplicates carry zero additional certificate information

If one stored transition \(W_i\) is sampled \(R\) times by an optimizer, all
copies are measurable with respect to the same realized primitive outcome. They
do not create \(R\) martingale innovations and count once in (30.2). Replacing
\(m\) by \(Rm\) would shrink the reported radius by \(R^{-1/2}\) without
changing the estimate, and is invalid.

### Proof

For coordinate \(d\), define

\[
\xi_{t,k,d}
=W_{\tau_{t,k},d}-\mu_d(Z_{\tau_{t,k}}).
\tag{30.6}
\]

Predictability of the eligibility event and (30.1), followed by optional
sampling at the successive hit times, make
\((\xi_{t,k,d})_{k\le m}\) a martingale-difference sequence. Its conditional
range has length at most \(2B_W\). The martingale Hoeffding inequality gives

\[
\Pr\left(
\left|\frac1m\sum_{k=1}^m\xi_{t,k,d}\right|
>B_W\sqrt{\frac{2x}{m}}
\right)
\le2e^{-x}.
\tag{30.7}
\]

Set \(x=\log(2JD/\delta)\) and union bound over \(t,d\). Hölder continuity
bounds the difference between the average
\(m^{-1}\sum_k\mu(Z_{\tau_{t,k}})\) and \(\mu(z_t)\) by
\(L_\mu h^\alpha\), proving (30.3). Claim B is direct substitution into
Theorem 27. Claim C follows because replay indices expose no new primitive
random variable after the original transition is observed. ∎

### Status and interpretation boundary

**PROVED FOR FRESH CONDITIONAL PRIMITIVE INNOVATIONS UNDER PREDICTABLE
INTERFACES; NUMERICALLY VERIFIED; REPLAY DUPLICATES EXPLICITLY EXCLUDED.**

- The result covers history-dependent policy/safety execution without assuming
  stationary mixing. It certifies unique chronological environment
  transitions, not optimizer minibatch draws.
- Query centers, bandwidth, and the representation used for eligibility must be
  fixed on another fold or updated predictably from past data. Outcome-selected
  retrospective clustering needs post-selection control.
- Failure to obtain \(m\) hits is a coverage failure, not a concentration
  failure. It must be reported rather than repaired by replaying existing rows.
- Martingale Hoeffding is standard. The project-specific consequence is that the
  continuous quotient--Perron--stopping phase remains valid for adaptive
  executed trajectories with a precise statistical sample unit.

## Theorem 31: Sharp stopped-margin localization from integrated \(L_p\) error

Let \(S\) be the exact oracle-shadow score at a random stopped decision query,
with exact decision \(d^\star=\operatorname{sign}(S)\). Let
\(\widehat S=S-E\) and
\(\widehat d=\operatorname{sign}(\widehat S)\), using the same deterministic
tie convention. No independence between \(S\) and the estimation error \(E\)
is assumed. Suppose that, for \(0<t\le r_0\),

\[
\Pr(|S|\le t)\le C_0t^\kappa,
\qquad C_0>0,\quad \kappa>0,
\tag{31.1}
\]

and that for some \(p>1\),

\[
R_p:=\mathbb E|E|^p<\infty.
\tag{31.2}
\]

### Claim A: integrated error controls first disagreement

For every \(0<t\le r_0\),

\[
\boxed{
\Pr(\widehat d\ne d^\star)
\le C_0t^\kappa+\frac{R_p}{t^p}.}
\tag{31.3}
\]

Consequently, if

\[
t_D=\left(\frac{pR_p}{\kappa C_0}\right)^{1/(p+\kappa)}\le r_0,
\tag{31.4}
\]

then

\[
\Pr(\widehat d\ne d^\star)
\le C_D(p,\kappa)C_0^{p/(p+\kappa)}
R_p^{\kappa/(p+\kappa)}.
\tag{31.5}
\]

Here \(C_D(p,\kappa)\) is the finite constant obtained by substituting
(31.4) into (31.3). If \(t_D>r_0\), the valid statement is the infimum of
the right-hand side of (31.3) over \(0<t\le r_0\), not the unconstrained
closed form.

### Claim B: boundary-weighted decision loss has a faster power

Define the local decision loss

\[
\mathcal L_{\partial}
=\mathbb E\!left[|S|\mathbf1\{\widehat d\ne d^\star\}\right].
\tag{31.6}
\]

For every \(0<t\le r_0\),

\[
\boxed{
\mathcal L_{\partial}
\le C_0t^{\kappa+1}+\frac{R_p}{t^{p-1}}.}
\tag{31.7}
\]

Thus, when

\[
t_L=\left\{\frac{(p-1)R_p}{(\kappa+1)C_0}\right\}^{1/(p+\kappa)}
\le r_0,
\tag{31.8}
\]

\[
\mathcal L_{\partial}
\le C_L(p,\kappa)C_0^{(p-1)/(p+\kappa)}
R_p^{(\kappa+1)/(p+\kappa)}.
\tag{31.9}
\]

The same radius controls mission coordinates under the first-disagreement
coupling: stranding-probability deviation is at most (31.3), and the deviation
of any throughput coordinate bounded by \(B_q\) is at most \(B_q\) times
(31.3).

### Claim C: both exponents are sharp

Let \(|S|\) have distribution
\(\Pr(|S|\le t)=t^\kappa\) on \([0,1]\), with an independent symmetric sign.
For \(a\in(0,1)\), take

\[
E=2S\mathbf1\{|S|\le a\}.
\tag{31.10}
\]

Then \(\widehat S=-S\) exactly on \(|S|\le a\), so

\[
R_p=\frac{2^p\kappa}{p+\kappa}a^{p+\kappa},\qquad
\Pr(\widehat d\ne d^\star)=a^\kappa,
\tag{31.11}
\]

and

\[
\mathcal L_{\partial}
=\frac{\kappa}{\kappa+1}a^{\kappa+1}.
\tag{31.12}
\]

Therefore the powers \(\kappa/(p+\kappa)\) in (31.5) and
\((\kappa+1)/(p+\kappa)\) in (31.9) cannot be uniformly improved using only
(31.1)--(31.2).

### Proof

Opposite signs of \(S\) and \(S-E\) imply \(|S|\le|E|\). Splitting the
disagreement event at \(|E|=t\), applying (31.1) to the low-error part and
Markov's inequality to the high-error part proves (31.3). On the low-error
part of (31.6), \(|S|\le t\), giving \(C_0t^{\kappa+1}\). On its high-error
part, disagreement gives \(|S|\le|E|\), while
\(|E|\mathbf1\{|E|>t\}\le |E|^p/t^{p-1}\); this proves (31.7). Direct
differentiation gives (31.4) and (31.8). The density of \(|S|\) in Claim C is
\(\kappa s^{\kappa-1}\); integrating (31.10) gives (31.11)--(31.12). ∎

### Estimator and loss consequence

The theorem removes global uniform prediction error from the minimal decision
handoff. A neural estimator may instead target a cross-fitted stopped-occupation
loss

\[
\widehat R_p
=\frac1N\sum_{i=1}^N\omega_i
|\widehat S_i-S_i^{\rm shadow}|^p,
\tag{31.13}
\]

where oracle-shadow scores and occupation weights are evaluated on data not used
to fit the predictor. Future trajectory summaries cannot be used as deployable
features, and outcome-derived weights require sample splitting or a predictable
construction. Global one-step MAE, repeated replay draws, and an unweighted
reconstruction loss do not certify (31.2) under the stopped law.

### Status and interpretation boundary

**PROVED AND NUMERICALLY VERIFIED; STOPPED-OCCUPATION \(L_p\) ESTIMATION RATE
REMAINS OPEN.**

The sharp transfer is a margin-theory ingredient specialized to the absorbing
ReturnManager and its stranding--throughput coordinates; it is not standalone
Oral-level novelty. Its value is a strict design and evidence reduction: a
method need only beat correct Doob/DICE/KROPE baselines in stopped-occupation
\(L_p\) risk, rather than uniformly estimating the whole transformed value
function. A valid learned-chart or cross-fitted estimator attaining that risk
remains to be proved and tested.

## Theorem 32: Cross-fitted stopped-occupation quotient regression

Let \(P\) be the source law of executed-interface observations and let
\(\nu_{\rm stop}\) be the target oracle-shadow stopped-decision law. An
independent structure fold returns a measurable chart
\(H:\mathsf X\to\mathsf Z\) and a finite partition
\(\mathcal A_h=\{A_1,\ldots,A_J\}\) of its range, with cell diameter at most
\(h\). Because this fold is independent, condition on \((H,\mathcal A_h)\)
throughout.

Write

\[
p_j=P\{H(X)\in A_j\},
\qquad
q_j=\nu_{\rm stop}\{H(X)\in A_j\},
\tag{32.1}
\]

and require \(p_j>0\) whenever \(q_j>0\). Define the pushed-forward local
overlap coefficient

\[
\boxed{
\mathfrak C_h(H;P,\nu_{\rm stop})
=\sum_{j:q_j>0}\frac{q_j}{p_j}.}
\tag{32.2}
\]

This coefficient is computed after both laws are pushed through \(H\); using
raw-atom density ratios after aggregation would define a different and generally
overly pessimistic object.

Let \(S(X)\) be the exact signed ReturnManager score and suppose there is an
\(\alpha\)-Hölder function \(s_H\) with constant \(L_H\) such that

\[
\sup_x|S(x)-s_H\{H(x)\}|\le b_H.
\tag{32.3}
\]

Nuisances used to construct a bounded pseudooutcome \(Y\in[-B_S,B_S]\) are
trained on another independent fold. On the estimation fold, let

\[
\mathbb E_P[Y\mid X]=S(X)+b_Y(X).
\tag{32.4}
\]

For

\[
\bar b_{Y,j}=\mathbb E_P[b_Y(X)\mid H(X)\in A_j],
\tag{32.5}
\]

assume the stopped cellwise nuisance remainder satisfies

\[
\sum_jq_j\bar b_{Y,j}^2\le\tau_Y^2.
\tag{32.6}
\]

From \(n\) iid source-fold observations define the unweighted cell estimator

\[
\widehat S_h(x)
=\frac{\sum_{i=1}^nY_i\mathbf1\{H(X_i)\in A_j\}}
{\sum_{i=1}^n\mathbf1\{H(X_i)\in A_j\}},
\quad H(x)\in A_j.
\tag{32.7}
\]

The estimator is deliberately unweighted: under conditional invariance after
the certified chart, source samples consistently estimate each conditional cell
mean. The target stopping law enters the risk and overlap coefficient. Explicit
importance weights are needed only if conditional invariance fails or a
different target cell functional is requested.

### Claim A: finite-sample stopped-risk oracle bound

If

\[
np_{\min}\ge8\log(2J/\delta),
\qquad
p_{\min}=\min_{j:q_j>0}p_j,
\tag{32.8}
\]

then with probability at least \(1-\delta\),

\[
\boxed{
R_{2,\rm stop}(\widehat S_h)
:=\mathbb E_{\nu_{\rm stop}}
[(\widehat S_h(X)-S(X))^2]
\le
3(2b_H+L_Hh^\alpha)^2
+3\tau_Y^2
+\frac{12B_S^2\log(4J/\delta)}{n}
\mathfrak C_h.}
\tag{32.9}
\]

This is a conditional learned-chart theorem: \(H\) may be data-dependent, but
its construction must use the independent structure fold. The bound does not
yet derive \(b_H\) from a neural training algorithm.

### Claim B: intrinsic quotient rate and transience phase

Suppose the target-active partition obeys

\[
\mathfrak C_h\le C_Qh^{-d_Q},
\tag{32.10}
\]

and the exact chart has \(b_H=0\). Balancing the Hölder and stochastic terms
in (32.9) gives

\[
h_n\asymp
\left\{\frac{B_S^2C_Q\log n}{L_H^2n}\right\}^{1/(2\alpha+d_Q)},
\tag{32.11}
\]

and

\[
R_{2,\rm stop}(\widehat S_{h_n})
=O_{\mathbb P}\!\left[
\left(\frac{C_Q\log n}{n}\right)^{2\alpha/(2\alpha+d_Q)}
+\tau_Y^2
\right],
\tag{32.12}
\]

with constants depending on \(B_S,L_H\). If score range and smoothness are
amplified by exponential transience as
\(B_S,L_H=O(\Delta^{-1})\), and
\(b_H=O(\rho_H/\Delta)\), then

\[
R_{2,\rm stop}
=O_{\mathbb P}\!\left[
\frac1{\Delta^2}
\left{
\left(\frac{C_Q\log n}{n}\right)^{2\alpha/(2\alpha+d_Q)}
+\rho_H^2
\right}
+\tau_Y^2
\right].
\tag{32.13}
\]

For fixed \(C_Q\), consistency of the stochastic term in the joint critical
family requires

\[
\frac{n\Delta^{(2\alpha+d_Q)/\alpha}}{\log n}\to\infty,
\tag{32.14}
\]

matching the continuous quotient phase in Theorem 27.

### Claim C: irreversible decision and Pareto rates

Under Theorem 31's stopped margin with exponent \(\kappa\), put the right-hand
side of (32.9) equal to \(\mathcal R_{2,n}\). Then

\[
\Pr(\text{first manager disagreement})
=O(\mathcal R_{2,n}^{\kappa/(\kappa+2)}),
\tag{32.15}
\]

and

\[
\mathcal L_\partial
=O(\mathcal R_{2,n}^{(\kappa+1)/(\kappa+2)}).
\tag{32.16}
\]

The stranding coordinate has the same upper order as (32.15), while any bounded
throughput coordinate inherits that order times its range.

### Claim D: strict raw-versus-quotient coverage slice

Let the raw interface be \((Z,V)\), with \(J_Z\) equally likely values of \(Z\)
and \(K\) nuisance values of \(V\). Under the source law, \((Z,V)\) is uniform;
under the target stopped law, \(Z\) remains uniform and \(V=1\). If the score
and pseudooutcome law depend only on \(Z\), the raw singleton partition has

\[
\mathfrak C_{\rm raw}=KJ_Z,
\qquad
\mathfrak C_{\rm quotient}=J_Z.
\tag{32.17}
\]

Thus the raw local estimator pays an exact nuisance factor \(K\) that disappears
after the sufficient quotient. In a regular continuous slice, a raw ambient
dimension \(D>d_Q\) gives the declared local-smoother rate
\(n^{-2\alpha/(2\alpha+D)}\), while (32.12) gives
\(n^{-2\alpha/(2\alpha+d_Q)}\).

This is not a universal lower bound against adaptive transformed OPE. A
Doob/DICE/KROPE estimator supplied with, or capable of learning, the same
sufficient quotient may attain \(\mathfrak C_{\rm quotient}\) and the same
intrinsic rate.

### Proof

Condition on the structure and nuisance folds. A multiplicative Chernoff bound
and (32.8) give \(N_j\ge np_j/2\) simultaneously on target-active cells with
probability at least \(1-\delta/2\). Conditional on \(N_j\), the observations in
cell \(j\) are iid and bounded in \([-B_S,B_S]\). Hoeffding's inequality and a
union bound give

\[
|\widehat S_j-\mathbb E_P[Y\mid H(X)\in A_j]|
\le
B_S\sqrt{\frac{2\log(4J/\delta)}{N_j}}.
\tag{32.18}
\]

For a target point in the same cell, Hölder continuity and (32.3) bound the
difference between the source-cell mean of \(S\) and the target score by
\(2b_H+L_Hh^\alpha\). The pseudooutcome cell bias is \(\bar b_{Y,j}\).
Applying \((a+b+c)^2\le3(a^2+b^2+c^2)\), integrating with weights \(q_j\),
using (32.6), (32.18), and \(N_j\ge np_j/2\) proves (32.9). Balancing the two
terms under (32.10) proves (32.11)--(32.12); substituting the declared
transience scales proves (32.13)--(32.14). Theorem 31 proves Claim C. For Claim
D, each target-active raw atom has \(q/p=K\), whereas each pushed-forward
\(Z\)-cell has \(q/p=1\); summing gives (32.17). ∎

### Status and interpretation boundary

**PROVED FOR AN INDEPENDENTLY CERTIFIED FINITE PARTITION; NUMERICALLY VERIFIED;
LEARNED-CHART DISTORTION RATE OPEN.**

The literature audit in
`literature-search-20260830-stopped-lp-quotient-learning/` shows that margin
comparison, target-law regression, cross-fitted nuisance learning, finite-library
aggregation, restricted OPE, and KROPE are established ingredients. Theorem 32's
project-specific value is the exact pushed-forward overlap-to-transience-to-
irreversible-Pareto composition. Oral-level promotion still requires a learned
chart rate or information separation that a matched transformed baseline cannot
inherit.

## Theorem 33: Certified chart selection and matched-information equivalence

Let the raw executed-interface alphabet be finite,
\(\mathcal U=\{1,\ldots,N\}\), and let
\(W\in[-B_W,B_W]^D\) be the stacked risk-witness vector from Theorem 26, with

\[
\mu(u)=\mathbb E[W\mid u].
\tag{33.1}
\]

For every interface \(u\), an independent structure fold supplies \(m_u\)
conditionally independent witness samples. Put

\[
\widehat\mu(u)=\frac1{m_u}\sum_{i=1}^{m_u}W_{u,i},
\qquad
r_m(\delta)
=B_W\sqrt{\frac{2\log(2ND/\delta)}{m_{\min}}}.
\tag{33.2}
\]

Let \(\mathcal P_1,\ldots,\mathcal P_M\) be candidate partitions of
\(\mathcal U\). They may be generated from \(\widehat\mu\): the event below is
uniform over every pair of raw interfaces and therefore over every partition of
the finite alphabet. For partition \(k\), define true and empirical within-cell
witness diameters

\[
\rho_k
=\max_{u,v:\mathcal P_k(u)=\mathcal P_k(v)}
\|\mu(u)-\mu(v)\|_\infty,
\tag{33.3}
\]

\[
\widehat\rho_k
=\max_{u,v:\mathcal P_k(u)=\mathcal P_k(v)}
\|\widehat\mu(u)-\widehat\mu(v)\|_\infty.
\tag{33.4}
\]

On independent estimation/target folds, suppose \(V_k\ge0\) is a simultaneous
certificate for all nonrepresentation terms in the stopped-risk bound: pushed-
forward overlap, cell estimation, pseudooutcome nuisance, and any declared
approximation term. Let \(A_\Delta>0\) be the witness-distortion-to-score
amplification, with \(A_\Delta=O(\Delta^{-1})\) near exponential transience.

### Claim A: uniform chart-distortion certificate

With probability at least \(1-\delta\), simultaneously for every candidate
partition,

\[
\boxed{
|\widehat\rho_k-\rho_k|\le2r_m(\delta).}
\tag{33.5}
\]

Thus

\[
\operatorname{Cert}_k
=A_\Delta^2\{\widehat\rho_k+2r_m(\delta)\}^2+V_k
\tag{33.6}
\]

is a valid upper certificate for the risk proxy
\(A_\Delta^2\rho_k^2+V_k\).

### Claim B: finite-library selector oracle inequality

Choose

\[
\widehat k\in\arg\min_{k\le M}\operatorname{Cert}_k.
\tag{33.7}
\]

On the same event,

\[
\boxed{
A_\Delta^2\rho_{\widehat k}^2+V_{\widehat k}
\le
\min_{k\le M}
\left[
A_\Delta^2\{\rho_k+4r_m(\delta)\}^2+V_k
\right].}
\tag{33.8}
\]

If the candidate library contains the exact risk quotient
\(\rho_{k^\star}=0\), its chart-learning contribution is at most

\[
16A_\Delta^2r_m(\delta)^2
=\frac{32A_\Delta^2B_W^2}{m_{\min}}
\log\frac{2ND}{\delta}.
\tag{33.9}
\]

Near transience, vanishing chart contribution therefore requires

\[
\frac{m_{\min}\Delta^2}{\log(ND/\delta)}\to\infty.
\tag{33.10}
\]

This is the value/decision-resolution scale, not the weaker scale for learning
only normalized allocation proportions in Theorem 22. When distinct quotient
classes have witness separation \(\gamma\), exact quotient identification is
obtained once \(r_m<\gamma/4\), and Theorem 26's Bernoulli two-point family
shows that the order \(m_{\min}\gamma^2\) is necessary.

### Claim C: matched-information algorithm-label separation is impossible

Consider two unrestricted algorithm classes given exactly the same raw data,
structure fold, witness dictionary, candidate partitions, target composition,
and cemetery coordinate:

1. a direct killed-operator/chart class; and
2. a transformed class allowed to apply the invertible Doob map from Theorem 29
   before or after any measurable chart learner.

Let \(\mathscr D_{\rm direct}\) and \(\mathscr D_{\rm Doob}\) be the sets of
manager decision rules these classes can induce. Then

\[
\boxed{
\mathscr D_{\rm direct}=\mathscr D_{\rm Doob}.}
\tag{33.11}
\]

Consequently, for every parameter family \(\Theta\) and every loss depending
only on the coupled manager trajectory or its stranding--throughput coordinates,

\[
\inf_{d\in\mathscr D_{\rm direct}}
\sup_{\theta\in\Theta}\mathbb E_\theta L(d,\theta)
=
\inf_{d\in\mathscr D_{\rm Doob}}
\sup_{\theta\in\Theta}\mathbb E_\theta L(d,\theta).
\tag{33.12}
\]

Thus no matched-information minimax theorem can prove superiority merely from
the names “EIRR,” “Doob,” “DICE,” or “KROPE.” A strict result must impose and
justify a smaller hypothesis class, a computational restriction, different side
information, or a specific optimization/regularization constraint. Named
implementations remain legitimate empirical baselines, but they are not distinct
statistical experiments.

### Proof

Coordinatewise Hoeffding concentration and a union bound over \(N D\)
coordinates give

\[
\max_u\|\widehat\mu(u)-\mu(u)\|_\infty\le r_m(\delta).
\tag{33.13}
\]

For every pair \(u,v\), the reverse triangle inequality then bounds the error
of its estimated witness distance by \(2r_m\). Taking a maximum over any
partition cell proves (33.5), including partitions selected after observing
\(\widehat\mu\). Therefore the selected true proxy is at most its certificate;
optimality in (33.7), followed by
\(\widehat\rho_k+2r_m\le\rho_k+4r_m\), proves (33.8). Equations
(33.9)--(33.10) are substitutions, and the recovery boundary is Theorem 26.

For Claim C, take any direct algorithm, copy its chart learner and estimator in
the transformed class, and insert the invertible Doob coordinate map and its
inverse around the same estimated operator. Theorem 29 gives identical values
and therefore identical manager decisions. The converse removes those maps.
Hence the induced decision-rule sets are equal, which proves (33.12). ∎

### Status and interpretation boundary

**FINITE-LIBRARY LEARNED-CHART RATE PROVED AND VERIFIED; UNRESTRICTED MATCHED-
INFORMATION SEPARATION CLOSED NEGATIVELY.**

The theorem advances beyond treating \(b_H\) as wholly external: on a finite
risk-witness alphabet it learns among candidate charts with a sharp
\(m_{\min}^{-1/2}\) witness radius and a decision-scale transience phase. It
does not provide distribution-free generalization for an arbitrary continuous
neural encoder. More importantly, the same certificate is available to every
matched-information transformed method. The remaining Oral-level route is the
full first-passage/interface/stopping certificate as a task theorem, or a
carefully declared computational/inductive-bias separation—not an unrestricted
statistical superiority claim.

## Novelty Classification of the Extension

- Theorems 6 and 10 are known mathematical foundations specialized to the
  charger-hitting object.
- Theorem 6B explicitly reduces the finite risk resolvent to a diagonally scaled
  discounted killed-chain resolvent. It strengthens the prior-art boundary:
  DICE-style occupancy-ratio estimation is a mandatory baseline.
- Theorem 8 is an elementary resolvent identity.
- Theorem 9A is a standard two-point decision-theoretic argument with an explicit
  executed-interface first-passage construction.
- Theorem 9C is a classical two-point sample-complexity argument specialized to
  common-support exponential charger-hitting risk.
- Theorem 11 is a standard clipped importance-weight certificate specialized to
  the exact risk-occupation residual identity.
- Theorem 12 is a standard first-disagreement coupling specialized to the actual
  absorbing two-boundary ReturnManager and the stranding--throughput Gate; it is
  not a novelty claim by itself.
- Theorem 13 is a new package-level composition of standard ingredients. Its
  interval propagation and union bounds are not standalone novelty; the
  potentially publishable object is the full source-executed-interface EIRR
  certificate, one-sided irreversible decision consequence, and matching
  concentration barrier.
- Theorem 14 supplies a finite-tabular attainable rate and matching
  \(e^{\lambda L}/\mu\) lower order by estimating the shared primitive directly.
  Empirical Bernstein concentration, robust perturbation, and two-point testing
  are prior art; the theorem is an attainability/limitation result for the joint
  executed-interface object, not a standalone algorithmic novelty.
- Theorem 15 is a classical testing-to-decision reduction. Its contribution to
  the package is alignment: the \(e^{\lambda L}/\mu\) barrier now reaches the
  actual premature-return/stranding Pareto criterion rather than stopping at
  prediction error.
- Theorem 16 is classical Gaussian linear-functional theory specialized to a
  cross-fitted risk-witness quotient. Its useful content is the exact
  representation condition \(m\in\operatorname{Range}(X^\top)\), the leverage
  rate, and a singular-raw-support example that strictly separates structural
  quotient coverage from cellwise support. It is not standalone novelty.
- Theorem 17 gives an exact two-layer block-double-robust score and product-bias
  identity for queryable execution composed with a killed exponential primitive.
  Orthogonal/doubly robust OPE is established prior art; the theorem is an
  estimator-design lemma and cannot be sold as oral novelty without a strictly
  sharper risk-first-passage rate, lower bound, or decision consequence.
- Theorem 18 derives the canonical gradient and local information bound for the
  fixed-design shared-primitive stopped-risk model. This is standard
  semiparametric calculus specialized to the charger-hitting Feynman--Kac
  functional. Its main design consequence is negative: with known \(L\), the
  state-composition correction improves robustness but is not a distinct
  efficient-information component.
- Theorem 19 is the first package result to expose a joint quotient--coverage and
  exponential-transience information phase transition at the ReturnManager's
  actual log-risk scale. Its one-state proof uses classical Bernoulli information;
  oral-level novelty depends on a sharp multi-state spectral theorem and matching
  irreversible-decision consequence, not the scalar calculation alone.
- Theorem 20 proves that the scalar critical exponents extend to finite
  multi-state systems exactly when primitive uncertainty has nonzero conditional
  projection onto the critical right Perron mode. It also proves a different
  phase under exact projected-noise degeneracy. The spectral and semiparametric
  ingredients are classical. The focused mathematical audit finds direct
  collisions for generic spectral sensitivity, QSD perturbation/CLTs,
  Feynman--Kac particle limits, SSP hardness, and margin conversion. Novelty is
  therefore limited to the still-conditional quotient-plus-projected-noise-plus-
  stopped-boundary composition, not the exponents themselves.
- Theorem 21 converts the canonical gradient into an exact cost-aware
  critical-mode source allocation and gives a finite pilot perturbation bound.
  Its optimizer is a specialization of classical Neyman allocation. A defensible
  oral-level claim requires an adaptive oracle inequality while learning the
  quotient, both Perron modes, and projected tail variance near criticality; the
  closed form alone is not novel.
- Theorem 22 supplies that adaptive bound for a known finite quotient and exposes
  a sharper candidate: normalized design learning can remain oracle-efficient
  even when its pilot is below the \(m\Delta^2\) scale needed for risk-value
  resolution. Adaptive Neyman allocation remains prior art. Paper-level novelty
  now depends on proving the matching phase diagram when the secondary eigengap,
  projected noise, or quotient separation collapses.
- Theorem 23 replaces the vague secondary-eigengap condition by the nonnormal
  Perron reduced-resolvent scale, gives a conditioned sufficient joint phase,
  proves a locally matching spectral/anisotropy allocation lower slice, and
  shows why projected variance alone has no universal exponent. Eigenvector
  perturbation, adaptive stratification, empirical variance concentration, and
  Le Cam testing are prior art; only their near-critical coupling remains a
  candidate contribution.
- Theorem 24 transfers the critical information constant and adaptive-oracle
  ratio into stopped Pareto and boundary-loss rates. Generic margin conversion
  is prior art. Its project-specific value is that the information phase now
  reaches the absorbing ReturnManager's original stranding--throughput
  criterion; it still does not establish empirical Pareto improvement.
- Theorem 25 solves the shared multi-query objective induced by Theorem 24 and
  shows common-critical cancellation plus heterogeneous-critical emphasis.
  Multiobjective stratified allocation and optimal OPE data design are mature;
  the KKT formula alone is not oral-level novelty. Its value is as the reusable
  algorithmic corollary of the stopped-risk phase theorem.
- Theorem 26 closes exact finite quotient recovery at the sharp
  \(\gamma^{-2}\) separation order and turns the quotient into a conditional
  moment reconstruction target for neural encoders. Concentration, clustering,
  and testing are classical; the finite rate is a baseline and design corollary,
  not standalone oral novelty. The declared continuous Hölder slice is handled
  by Theorem 27; unknown-chart neural adaptation remains open.
- Theorem 27 proves a continuous local-mass Hölder point-query rate, a
  distortion-to-operator sandwich, its coupled Perron/transience/stopped-margin
  phase, and matching local bump orders. Nonparametric regression, plug-in
  classification margins, and spectral perturbation are classical. The only
  remaining candidate is their joint phase for risk-observable executed
  interfaces; KROPE makes generic bisimulation representation stability an
  occupied claim.
- Theorem 28 proves that expected-reward KROPE can collapse equal-mean,
  unequal-exponential-risk interfaces and that raw coverage can diverge while
  quotient coverage remains exact. It also closes the other side of the audit:
  oracle risk-transformed KROPE/DICE is algebraically a standard discounted
  baseline under Theorem 6B. The result is a novelty boundary, not a new
  algorithmic claim.
- Theorem 29 proves that learning the unknown Doob coordinates is itself a
  bijective reparameterization: direct and transformed plug-in values are
  identical for the same operator estimate, minimax decision risk is invariant,
  and relative transform conditioning retains the \(1/\Delta\) critical scale.
  Learned-Doob is therefore not a standalone rate-improvement route.
- Theorem 30 replaces iid transitions by adaptive chronological execution under
  a conditional primitive-law martingale. It preserves Theorem 27's rate when
  fresh local hit counts grow at the declared coverage rate and proves that
  replay duplicates cannot shrink a statistical certificate. Martingale
  Hoeffding is classical; the contribution is the exact sample-unit and phase
  handoff for policy--filter-dependent trajectories.
- Theorem 31 proves that stopped-occupation \(L_p\) error, rather than global
  sup-norm error, is sufficient for irreversible decisions. Its disagreement
  and boundary-loss powers are matched by an explicit construction. Generic
  margin localization is classical; the contribution candidate is only a
  learned risk-quotient estimator that realizes a strict stopped-risk advantage
  over matched Doob/DICE/KROPE baselines.
- Theorem 32 supplies a cross-fitted finite-partition estimator whose stopped
  \(L_2\) risk depends on the source/target overlap coefficient after pushforward
  through the chart. It proves intrinsic Hölder, transience, decision, and exact
  nuisance-multiplicity consequences. Covariate-shift regression, orthogonal
  learning, and margin comparison are occupied; the theorem is a project-specific
  composition, not yet an exclusive algorithmic advantage.
- Theorem 33 learns among finite candidate charts using uniform risk-witness
  diameter certificates and proves an oracle inequality at the
  \(m_{\min}\Delta^2\) decision scale. It also closes unrestricted
  matched-information separation negatively: the invertible Doob map makes the
  direct and transformed manager-decision classes identical. Concentration,
  structural selection, and reparameterization are standard; the result is a
  design/novelty boundary rather than an exclusive method claim.
- The candidate contribution is the combined theorem chain: executed-interface
  quotient/identification, risk-occupation residual weighting, separation of
  support from risk concentration, composition-shift estimation, and irreversible
  return-boundary consequences.
- No oral-level novelty is established yet. The finite tabular primitive plug-in
  chain and the continuous local-mass Hölder slice now have matching upper/local
  lower orders and stopped-decision consequences. Theorems 31--33 prove the exact
  decision-localization transfer and a certified-chart estimator advantage over
  a declared raw local model. Theorem 33 proves that no unrestricted adaptive
  matched baseline can be separated information-theoretically by algorithm
  label. The remaining theory route is either the combined task-specific
  first-passage/interface/stopping certificate, or an explicitly restricted
  computational/inductive-bias theorem that survives misspecification and
  dependent replay.
- The focused continuous-interface audit in
  `literature-search-20260830-continuous-eirr-theory/` shows that generic CME
  estimation, restricted-chi-square OPE, function-class aggregation,
  covariate-shift KRR, continuous bisimulation, and multiplicative-drift
  Feynman--Kac stability are already covered. The next theorem is therefore
  frozen as a four-part risk-observable interface quotient composed with a
  separate stopped-boundary functional: exact sufficiency, approximate
  irreversible-decision stability, a quotient-dependent attainable rate, and
  matching necessity. Theorem 27 now proves all four on a declared finite-query
  iid local-mass Hölder slice. The general unknown-chart neural and dependent-
  replay versions remain unproved. The algebra and phase derivation are in
  `docs/RISK_BOUNDARY_QUOTIENT_DERIVATION.md`; the verifier checks the intrinsic
  rate, distance sandwich, constant-KL bump, and critical phase powers.
- The theorem-level audit in
  `literature-search-20260830-executed-interface-risk-resolvent/closest-theorem-audit.md`
  finds that no individual finite-state statement is a safe novelty claim. The
  remaining candidate is their joint source-to-target executed-interface,
  killed-first-passage, and irreversible-stopping consequence.

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
| Within-composition interface mechanism | Horizon/path/goal/action-controlled held-out association and residual-tail audit | R3 exploratory support: 3.90% total-Energy MAE reduction; beta-one intervention enrichment 1.354x; sign-heterogeneous mean effect; not deployable or cross-composition evidence |
| Proper charger hitting | Charger reach and emergency-tail statistics | Pending formal Gate chain |
| Target interface coverage | Predeclared, non-label-based coverage proxy | Not implemented for Stage E |
| Local error \(\epsilon_t\) | Cross-fitted one-step predictive error | Not implemented; Stage C/F only |
| Hitting-tail term | Horizon-stratified \(T_C\) tail | Pending formal Oracle evaluation |
| Decision overshoot | Per-check consumption plus requirement drift | Implemented in switch and cycle telemetry; formal data pending |
| Calibration A13 | Accepted-set empirical calibration on target composition | Not established |
| Boundary law | Oracle-shadow stopped boundary mass, first disagreement, and coordinate deviation | Population theorem and tight finite-law verifier complete; formal mission data pending Oracle Gate |
| Simultaneous EIRR radius | Global branch--state--risk residual, nuisance, and positive-interval certificate | Algebra and tight finite-law verifier complete; learned dependent-trajectory rate absent |

## Open Risks

- A11 may be false at discontinuous HOCBF active-set transitions.  A piecewise
  theorem or a total-variation formulation may be needed.
- A8 is an identification assumption, not an operationally observable support
  certificate.  The eventual reliability score must be evaluated empirically and
  must not be described as exact mathematical support.
- The Oracle headroom Gate may show that perfect Resource-to-Go information does
  not materially improve the stranding--throughput frontier.
- All repair checkpoints R1--R5 failed the historical engineering navigation
  Gate. R3 is now the user-selected frozen conditional platform; its contract and
  calibration attestation do not rewrite that failure. The first endurance
  audit was invalid because 41/100 endpoints were task-limit censored; the
  version-2 continuous-workload audit is running and requires 100/100 true
  depletion endpoints. No further incremental navigation repair is authorized.
- A generic executed-action probabilistic world model with a bootstrap ensemble
  may make a separate SIRP architecture unnecessary.
- The R3 post-trajectory audit controls horizon, path geometry, goal type, nominal
  effort, and distance within one composition. Its small held-out total-Energy
  increment and stronger conditional-tail increment keep the mechanism alive,
  but do not establish composition transfer or deployable pre-decision estimation.
- The theorem package currently covers shared primitive dynamics only.  Wind,
  payload, battery aging, and actuator-model shifts require explicit context or a
  second identification analysis.
