# Risk-Observable Interface Quotient and Irreversible Boundary Derivation

## Target

Determine whether a continuous/neural executed-interface representation can be
given a theorem-level design rule for charger-hitting exponential Energy-to-Go,
and whether that rule is mathematically distinct from conditional mean embedding
or restricted off-policy evaluation after a Doob transform.

The immediate output is an exact algebraic equivalence boundary and a coherent
local-error-to-decision derivation. It is not yet a finite-sample neural theorem.

## Status

**COHERENT AFTER REFRAMING / EXTRA ASSUMPTION**

The original phrase “risk--boundary observable interface quotient” mixed two
different objects:

1. a one-step interface pseudometric induced by conditional exponential
   continuation expectations; and
2. a nonlinear stopped-margin functional that determines whether propagated
   error changes an irreversible decision.

They must remain separate and be composed by a theorem. The quotient alone is a
task-restricted conditional mean embedding and is not a safe novelty claim.

## Invariant Object

The invariant object is the exact target-composition charger-hitting exponential
value

\[
\psi_\lambda(x)
=\mathbb E_x^{\pi,\Pi,K}
\exp\!\left(\lambda\sum_{t<T_C}C_t\right),
\qquad \lambda\in\Lambda,
\]

together with the exact-risk ReturnManager margin built from the finite grid
\(\{\log\psi_\lambda/\lambda:\lambda\in\Lambda\}\). Prediction MAE, raw
interface distance, and encoder reconstruction error are proxies, not the
invariant object.

## Assumptions

- **Q1 (measurable composition).** State space \(\mathsf X\), executed-action
  space \(\mathsf U\), and outcome space are standard Borel. The charger set
  \(G_C\subseteq\mathsf X\) is measurable and absorbing.
- **Q2 (shared primitive).** The primitive
  \(K(dx',dc\mid x,u)\) is shared across policy--filter compositions. The target
  policy and safety execution kernel induce a queryable interface kernel
  \(L(dq\mid x)\), where \(q=(x,u)\).
- **Q3 (finite risk grid).** \(\Lambda\subset(0,\infty)\) is finite. No uniform
  statement over all positive \(\lambda\) is asserted.
- **Q4 (exponential transience).** For each \(\lambda\in\Lambda\), the killed
  positive operator \(M_\lambda\) has a positive bounded resolvent
  \(R_\lambda=(I-M_\lambda)^{-1}\) on a declared weighted-supremum or bounded
  function space. Write \(\kappa_\lambda=\|R_\lambda\|\).
- **Q5 (witness closure).** The continuation witness class \(\mathcal F\)
  contains the relevant true and encoded values and is closed under the
  conditional multiplicative continuation operations used below.
- **Q6 (positive values).** On queried decision states,
  \(\psi_\lambda,\widehat\psi_\lambda\ge m_\lambda>0\).
- **Q7 (fixed encoder for the algebra).** The encoder \(h:\mathsf X\times
  \mathsf U\to\mathsf Z\) is measurable and fixed independently of the sample
  used to evaluate the displayed identities. Learned encoders require
  cross-fitting or a joint empirical-process analysis.
- **Q8 (stopped-law coupling).** Oracle and encoded managers are coupled on the
  same primitive randomness until first disagreement, as in the existing
  irreversible-boundary theorem.
- **Q9 (source factorization and domination).** The source interface law is
  \(\rho_Q(dq)=\rho_X(dx)B(du\mid x)\). The target risk state/interface measures
  satisfy \(\eta^X_{\nu,\lambda}\ll\rho_X\) and
  \(\eta^Q_{\nu,\lambda}:=\eta^X_{\nu,\lambda}L\ll\rho_Q\), with square-integrable
  non-normalized ratios \(v_\lambda=d\eta^X/d\rho_X\) and
  \(w_\lambda=d\eta^Q/d\rho_Q\).
- **Q10 (queryable composition integration).** For a supplied interface function
  \(g\), \(L g(x)=\int g(x,u)L(du\mid x)\) is exactly computable or its separate
  Monte Carlo error is explicitly bounded. The identities below use exact
  integration.
- **Q11 (cross-fitting for rates).** Nuisance functions are trained outside the
  validation fold on which their score is averaged. Population identities do not
  require sample splitting; the conditional empirical-process statement does.
- **Q12 (sampling slice for the displayed root-\(n\) statement).** Validation
  observations are iid source transitions, or independent episodes whose
  episode-level scores satisfy a declared central limit theorem. Arbitrary
  survival-selected replay and overlapping transitions are not covered by the
  iid statement below; they require a martingale, regeneration, or mixing
  extension.
- **Q13 (semiparametric primitive-law slice).** For the efficiency calculation,
  iid observations are \(O=(Q,Y)\sim\rho_Q(dq)P(dY\mid q)\); \(\nu\), \(L\),
  and the source design law \(\rho_Q\) are fixed, while the shared conditional
  primitive \(P\) varies along differentiable-in-quadratic-mean submodels. If
  \(\rho_Q\) is also unknown, it is an ancillary nuisance because the target
  parameter depends only on \(P,\nu,L\). Required ratios and conditional witness
  variances are square-integrable.
- **Q14 (finite critical Perron family).** For \(\Delta\downarrow0\), the finite
  quotient-state killed operator \(M_\Delta\ge0\) is irreducible with simple
  Perron root \(1-\Delta\). Its left/right Perron vectors
  \((\ell_\Delta,z_\Delta)\) are normalized by
  \(\ell_\Delta^\top\mathbf1=1\) and
  \(\ell_\Delta^\top z_\Delta=1\), converge to positive limits, and the reduced
  resolvent
  \(H_\Delta=(I-M_\Delta)^{-1}-z_\Delta\ell_\Delta^\top/\Delta\) is uniformly
  bounded. The forcing \(r_\Delta\), primitive conditional second moments, fixed
  \(\nu\), target quotient kernel \(L\), and source quotient design \(\rho_Q\)
  converge; \(\nu z_0>0\), \(\ell_0^\top r_0>0\), and target quotient mass is
  dominated by \(\rho_Q\). Quotient pooling is permitted only under an exact
  shared-primitive constraint fixed before outcome inspection.
- **Q15 (optional uniform triangular-array LAN slice).** A joint statement with
  \(\Delta=\Delta_n\downarrow0\) is made only for a declared sequence of
  conditional primitive laws whose normalized least-favourable scores satisfy a
  uniform Lindeberg condition and whose conditional submodels are uniformly
  differentiable in quadratic mean, so that their log-likelihood ratios obey
  LAN along the triangular array. This is strictly stronger than Q13, which
  gives pointwise LAN for each fixed \(\Delta>0\).
- **Q16 (costed stratified primitive design).** The finite exact quotient
  interfaces can be sampled independently without changing their conditional
  primitive laws. One draw from interface \(q\) costs
  \(\kappa(q)\in(0,\infty)\), and the total acquisition budget is
  \(\mathsf B\). Conditional on a separate pilot sample, second-stage stratum
  counts are fixed and the regular stratified estimator attains the canonical
  bound. This is a designed-sampling theorem, not a statement about arbitrary
  survival-selected replay.
- **Q17 (uniformly learnable critical modes).** The quotient and target
  composition \(L\) are fixed and known. The primitive continuation vector
  \(A(Y)=e^{\lambda C}\mathbf1\{X'\notin G_C\}e_{X'}\) is uniformly bounded.
  The Perron eigenprojector has uniformly bounded reduced spectral resolvent
  \(S_\Delta=(M_\Delta-(1-\Delta)I)^\#\), equivalently the effective Perron
  separation \(\mathfrak g_\Delta=\|S_\Delta\|^{-1}\) is bounded below
  (although the distance \(\Delta\) from the Perron root to one vanishes). The
  normalized Perron vectors remain in a compact positive set, and every
  leading active priority
  \(d_\Delta(q)=\ell_\Delta(x)L_\Delta(q\mid x)\sigma_\Delta(q)\) is bounded
  below. An independent pilot supplies at least \(m_{\min}\) observations per
  active quotient stratum. Outcome-selected or learned quotient recovery is not
  included.
- **Q18 (conditioned collapsing family).** To study failure of Q17, write
  \(S_\Delta=(M_\Delta-(1-\Delta)I)^\#\) and
  \(\mathfrak g_\Delta=\|S_\Delta\|^{-1}\), allowing
  \(\mathfrak g_\Delta\downarrow0\). For
  \(\Sigma_\Delta(q)=\operatorname{Cov}\{A(Y)\mid q\}\), define
  \(\sigma_\Delta(q)^2=z_\Delta^\top\Sigma_\Delta(q)z_\Delta\), directional
  condition
  \(\chi_\Delta(q)=\|\Sigma_\Delta(q)z_\Delta\|/\sigma_\Delta(q)^2\),
  standardized fourth moment \(\varkappa_\Delta(q)\), and standardized range
  \(b_\Delta(q)\) of \(A^\top z_\Delta\). A proposed or learned quotient may
  contribute uniform operator-moment bias \(\tau_M\) and fixed-direction
  projected-standard-deviation relative bias \(\tau_\sigma\). These quantities,
  rather than \(\sigma\) alone, define the collapsing phase.
- **Q19 (stopped information transfer).** For each of finitely many frozen
  oracle-shadow decision queries, the risk-requirement estimator is trained on
  data independent of the deployment cycle and has a centered sub-Gaussian
  error with declared variance proxy \(V_t\). The oracle-shadow score satisfies
  \(\Pr(t<\tau_{\rm end},|s_t|\le r)\le C_t r^\kappa\). A regular stratified
  estimator attains Theorem 21's canonical variance up to a stated remainder.
  When positive excess loss is discussed, Theorem 12's local boundary-loss
  condition is also assumed. Shared multi-query replay design is not silently
  reduced to a collection of independently optimized scalar queries.
- **Q20 (shared multi-query design).** For \(J\) frozen stopped-boundary queries
  and \(Q\) source strata, the regular influence sensitivity of query \(t\) in
  stratum \(q\) is \(a_{tq}\), so
  \(V_t(\boldsymbol n)=\sum_q a_{tq}^2/n_q\). Positive weights \(\omega_t\)
  encode the declared stopped-occupation constants, and the acquisition costs
  are \(c_q>0\). The continuous allocation relaxation is used; integer rounding,
  adaptive estimation of \(a_{tq}\), and survival-selected replay are separate.
- **Q21 (finite learned quotient).** The raw executed-interface alphabet
  \(\mathcal U\) is finite. A bounded vector \(W(Y)\in[-B_W,B_W]^D\) stacks the
  terminal and continuation coordinates needed by every declared risk level,
  together with the first and second moments needed for projected variances.
  The exact risk-observable quotient is equality of
  \(\mu(u)=\mathbb E[W(Y)\mid u]\). Distinct quotient classes have
  \(\ell_\infty\) separation at least \(\gamma>0\). A structure fold supplies
  independent observations per raw interface; allocation and value estimation
  use separate folds.
- **Q22 (continuous local-mass quotient slice).** A fixed finite collection of
  oracle-shadow interface queries has representations
  (z_1,\ldots,z_J\) in a metric space \((\mathsf Z,d_Z)\). An independent
  structure fold contains iid pairs \((Z_i,W_i)\), where
  (W_i\in[-B_W,B_W]^D\),
  \(\mu(z)=\mathbb E[W\mid Z=z]\) is \(\alpha\)-Hölder with constant
  (L_\mu\), and, for (0<h\le h_0\),
  \(\Pr\{d_Z(Z,z_t)\le h\}\ge c_Qh^{d_Q}\). The map from the declared
  witness moments to the killed operator/variance coordinates is
  (C_M\)-Lipschitz. The queries, bandwidth, pooling threshold, and witness
  dictionary are fixed independently of this fold. This assumption covers a
  declared intrinsic Hölder quotient slice, not an arbitrary learned neural
  encoder or survival-selected replay.
- **Q23 (predictable chronological interface slice).** For the dependent
  extension, \(Z_i\) is known before fresh primitive outcome \(W_i\) and
  \(\mathbb E[W_i\mid\mathcal H_{i-1}]=\mu(Z_i)\). Query centers, bandwidth,
  and eligibility representation are fixed on another fold or updated from
  past-only information. Each query uses its first \(m\) unique eligible
  environment transitions. Optimizer replay copies are not new observations.
- **Q24 (stopped integrated-error slice).** Under the oracle-shadow stopped
  decision law, let \(S\) be the exact signed margin and
  \(\widehat S=S-E\) the plug-in margin. For \(0<t\le r_0\),
  \(\Pr(|S|\le t)\le C_0t^\kappa\), and for some \(p>1\),
  \(R_p=\mathbb E|E|^p<\infty\). The score and error may be arbitrarily
  dependent. A learned estimator must establish this moment on an independent
  stopped-occupation fold or through a separately valid predictable argument.
- **Q25 (cross-fitted stopped quotient regression).** An independent structure
  fold fixes a chart and finite partition; an independent nuisance fold fixes a
  bounded stopped-score pseudooutcome; an estimation fold supplies iid source
  rows. Conditional pseudooutcome bias has a declared stopped-cell \(L_2\)
  remainder. Source support covers every target-active pushed-forward cell. The
  chart has declared score distortion and Hölder regularity. This is not a rate
  theorem for the chart learner itself.
- **Q26 (finite chart library and matched information).** The raw interface
  alphabet and stacked bounded witness dictionary are finite. Per-interface
  structure samples estimate every witness coordinate. Candidate partitions may
  depend on those estimates, while their nonrepresentation stopped-risk terms
  are simultaneously certified on independent folds. Direct and transformed
  algorithm classes receive exactly the same data, candidate library, target
  composition, and cemetery coordinate.

## Notation

- \(q=(x,u)\): executed interface.
- \(Y=(X',C)\): primitive next-state/resource outcome.
- \(L(dq\mid x)\): queryable target policy--filter interface law.
- \(\mathcal F_1\): unit ball of the declared continuation witness class.
- \(\rho_s(dq,dY)\): source transition law.
- \(\nu\): target initial-state law.
- \(\mu_{\mathrm{stop}}\): oracle-shadow stopped decision-state occupation law.
- \(m(x)\): true ReturnManager continue margin, positive on `CONTINUE` states.
- \(v_\lambda(x)\): target risk-state/source-state density ratio.
- \(w_\lambda(q)\): target risk-interface/source-interface density ratio.

For \(f\in\mathcal F\), define the outcome witness

\[
\Gamma_{\lambda,f}(x',c)
=e^{\lambda c}\left[
\mathbf 1\{x'\in G_C\}
+\mathbf 1\{x'\notin G_C\}f(x')
\right]
\]

and the primitive conditional continuation operator

\[
\mathcal K_\lambda f(q)
=\mathbb E[\Gamma_{\lambda,f}(X',C)\mid q].
\]

After target composition,

\[
\mathcal T_\lambda f(x)
=\int \mathcal K_\lambda f(q)L(dq\mid x)
=r_\lambda(x)+M_\lambda f(x).
\]

The exact value is the fixed point
\(\psi_\lambda=\mathcal T_\lambda\psi_\lambda\).

## Derivation Strategy

1. Identify the proposed quotient exactly as an integral probability metric over
   conditional outcome laws.
2. Prove the exact sufficiency statement as a factorization identity.
3. Compare that identity with restricted CME/OPE to locate the non-novel part.
4. Introduce an arbitrary encoder-restricted approximation and derive its exact
   resolvent residual.
5. Propagate the local residual through log-MGF, EVaR/grid minimization, branch
   maximum, and the stopped irreversible margin.
6. State the finite-sample quotient-rate step separately as an unproved target;
   do not hide it inside the exact algebra.

## Derivation Map

1. \(d_{\Lambda,\mathcal F}\) depends only on conditional expectations of the
   witness family \(\Gamma_{\lambda,f}\): exact identity.
2. Zero distance on encoder fibers implies measurable factorization of
   \(\mathcal K_\lambda f\): proposition using Q1, Q5, and a measurable quotient
   condition.
3. Factorization plus exact encoded conditional expectations gives identical
   \(\mathcal T_\lambda\), \(\psi_\lambda\), and manager decisions: proposition.
4. For an approximate encoded operator, subtract the two fixed-point equations:
   exact resolvent identity using Q4.
5. Positivity gives a pointwise absolute-error envelope; Q6 converts it to a
   log-MGF envelope: proposition.
6. Finite-grid min and branch max are 1-Lipschitz in the sup coordinate norm:
   exact inequality.
7. The stopped-law coupling converts state-dependent requirement error to a
   first-disagreement functional: proposition using Q8.
8. A quotient effective-dimension rate needs a separate sampling theorem and is
   not proved by Steps 1--7.
9. Splitting target composition and primitive transition residuals produces an
   exact two-layer population score; change of measure and conditional centering
   give block double robustness and an exact product-bias identity.
10. Differentiate the stopped Feynman--Kac fixed point with respect to the
    primitive conditional law, propagate the derivative through the resolvent,
    and project it onto the conditional-law tangent space to obtain the canonical
    gradient and information bound.
11. Use the Perron spectral projector to expand the value, risk occupation,
    primitive witness variance, and efficient information near transience; split
    nondegenerate and projected-noise-degenerate regimes.
12. Optimize the cost-aware scalar-query information bound, learn its normalized
    critical-mode priorities, and characterize the conditioned collapse phase.
13. Transfer the resulting variance through the oracle-shadow stopped margin to
    first-disagreement, Pareto-coordinate, and boundary-loss rates.
14. Optimize one shared allocation for the margin-powered collection of query
    variances and expose how common or query-specific critical gaps enter it.
15. Recover a finite risk-observable quotient from conditional witness moments,
    prove its sharp separation rate, and state the neural representation target.
16. Replace finite exact recovery by a continuous local-mass Hölder slice,
    derive its pointwise quotient radius, pass it through Perron and transient
    amplification, and match both phase coordinates with local bump tests.
17. Separate risk-neutral representation failure and raw-support pessimism from
    the exact oracle-Doob equivalence that makes transformed KROPE/DICE mandatory.
18. Audit the unknown Doob transform as a statistical reparameterization and
    retain its exact near-transience conditioning rather than claiming a new rate.
19. Replace iid local averages by predictable first-hit martingale averages and
    identify unique chronological transitions as the certificate sample unit.
20. Localize the final decision directly: convert stopped-occupation \(L_p\)
    error to sharp disagreement and boundary-loss powers without requiring a
    global sup-norm value-function bound.
21. Estimate the stopped score on an independently certified chart, expose the
    exact pushed-forward source/target overlap coefficient, and propagate its
    intrinsic rate through transience and the irreversible boundary.
22. Learn among finite candidate charts by upper-certifying within-cell witness
    diameter, then audit whether the same selector can be copied by a matched
    transformed baseline.

## Main Derivation

### Step 1. The interface pseudometric is a restricted conditional-law IPM

Define

\[
d_{\Lambda,\mathcal F}(q,q')
=\sup_{\lambda\in\Lambda,\,f\in\mathcal F_1}
\left|
\mathcal K_\lambda f(q)-\mathcal K_\lambda f(q')
\right|.
\tag{RBQ.1}
\]

Let \(P_q\) be the conditional law of \(Y=(X',C)\) given \(q\), and let

\[
\mathcal G_{\Lambda,\mathcal F}
=\{\Gamma_{\lambda,f}:\lambda\in\Lambda,
f\in\mathcal F_1\}.
\]

Then, by substitution,

\[
d_{\Lambda,\mathcal F}(q,q')
=\sup_{g\in\mathcal G_{\Lambda,\mathcal F}}
\left|\int g\,dP_q-\int g\,dP_{q'}\right|.
\tag{RBQ.2}
\]

Equation (RBQ.2) is an **identity**. Thus the proposed distance is an integral
probability metric restricted to decision-relevant exponential continuation
witnesses. If \(\mathcal G\) is an RKHS unit ball, it is an MMD; if represented by
a finite feature family, it is a feature-mean distance. This part is not a new
risk measure or a new generic embedding theorem.

### Step 2. Exact quotient sufficiency

Assume

\[
h(q)=h(q')\quad\Longrightarrow\quad
d_{\Lambda,\mathcal F}(q,q')=0.
\tag{RBQ.3}
\]

Then \(\mathcal K_\lambda f\) is constant on every fiber of \(h\). Under the
standard-Borel/measurable-factorization condition, for each
\((\lambda,f)\) there exists a measurable \(k_{\lambda,f}\) such that

\[
\mathcal K_\lambda f(q)=k_{\lambda,f}(h(q)).
\tag{RBQ.4}
\]

If an encoded primitive reproduces these conditional expectations, its composed
operator satisfies

\[
\widehat{\mathcal T}_\lambda f(x)
=\int k_{\lambda,f}(h(q))L(dq\mid x)
=\mathcal T_\lambda f(x)
\quad\text{for all }f\in\mathcal F.
\tag{RBQ.5}
\]

Since \(\psi_\lambda\in\mathcal F\) and the fixed point is unique under Q4,

\[
\widehat\psi_\lambda=\psi_\lambda
\quad\text{for every }\lambda\in\Lambda.
\tag{RBQ.6}
\]

Every deterministic ReturnManager functional of this grid is therefore exactly
preserved. Equations (RBQ.3)--(RBQ.6) form a **proposition**, not merely an
interpretation.

### Step 3. Exact prior-art equivalence boundary

The map

\[
q\longmapsto
\left(g\longmapsto\mathbb E[g(Y)\mid q]\right),
\qquad g\in\mathcal G_{\Lambda,\mathcal F},
\tag{RBQ.7}
\]

is precisely a conditional mean operator restricted to the witness family
\(\mathcal G_{\Lambda,\mathcal F}\). Therefore:

- estimating (RBQ.7) by vector-valued/kernel regression is a restricted CME;
- measuring source--target difficulty only through its feature span is a
  restricted-\(\chi^2\)/aggregated-concentrability construction;
- identifying points with zero (RBQ.1) is a task-specific behavioral quotient.

This is an **equivalence classification**. Exact quotient sufficiency alone
cannot support an oral novelty claim. The project-specific object can only be the
subsequent composition with killed first passage, queryable safety execution, and
the irreversible stopped boundary, or a genuinely sharper attainable/necessary
rate.

### Step 4. Exact residual identity for an approximate encoder

Let \(\widehat{\mathcal K}_\lambda\) be any encoder-restricted conditional
operator and let

\[
\widehat\psi_\lambda
=\widehat{\mathcal T}_\lambda\widehat\psi_\lambda.
\]

Define the local encoded-interface residual

\[
\delta_{\lambda,h}(q)
=\mathcal K_\lambda\widehat\psi_\lambda(q)
-\widehat{\mathcal K}_\lambda\widehat\psi_\lambda(q)
\tag{RBQ.8}
\]

and its target-composed state residual

\[
b_{\lambda,h}(x)
=\int\delta_{\lambda,h}(q)L(dq\mid x).
\tag{RBQ.9}
\]

Subtracting the two fixed-point equations yields the **exact identity**

\[
\psi_\lambda-\widehat\psi_\lambda
=R_\lambda b_{\lambda,h}.
\tag{RBQ.10}
\]

For initial law \(\nu\), with
\(\eta^X_{\nu,\lambda}=\nu R_\lambda\) and the corresponding target executed
risk occupation \(\bar\eta_{\nu,\lambda}\),

\[
\nu(\psi_\lambda-\widehat\psi_\lambda)
=\eta^X_{\nu,\lambda}b_{\lambda,h}
=\bar\eta_{\nu,\lambda}\delta_{\lambda,h}.
\tag{RBQ.11}
\]

This shows why unweighted one-step encoder reconstruction is misaligned: the
exact scalar target weights the fiber residual by target exponential first-
passage occupation.

### Step 5. Pointwise propagated representation envelope

Because \(R_\lambda\) is positive,

\[
|\psi_\lambda-\widehat\psi_\lambda|(x)
\le e_{\lambda,h}(x)
:=R_\lambda|b_{\lambda,h}|(x).
\tag{RBQ.12}
\]

If every encoder fiber has witness distortion at most \(\varepsilon\), and the
encoded operator selects any representative/conditional mixture within the
fiber, then for the normalized witness containing
\(\widehat\psi_\lambda\),

\[
|b_{\lambda,h}(x)|
\le B_{\lambda,\widehat\psi}\,\varepsilon,
\qquad
e_{\lambda,h}(x)
\le \kappa_\lambda
B_{\lambda,\widehat\psi}\varepsilon.
\tag{RBQ.13}
\]

The scaling factor \(B_{\lambda,\widehat\psi}\) is required when the witness
class is normalized; it must not be silently set to one.

Using Q6 and the mean-value inequality for \(\log\),

\[
|\log\psi_\lambda(x)-\log\widehat\psi_\lambda(x)|
\le \frac{e_{\lambda,h}(x)}{m_\lambda}.
\tag{RBQ.14}
\]

### Step 6. Grid-risk and ReturnManager propagation

For a branch \(j\) and failure level \(\delta_j\), define the finite-grid
effective requirement

\[
U_j(x)=\min_{\lambda\in\Lambda}
\frac{\log\psi_{j,\lambda}(x)+\log(1/\delta_j)}{\lambda}.
\]

The elementary inequality

\[
|\min_i a_i-\min_i b_i|\le\max_i|a_i-b_i|
\]

and (RBQ.14) give

\[
|U_j(x)-\widehat U_j(x)|
\le g_j(x)
:=\max_{\lambda\in\Lambda}
\frac{e_{j,\lambda,h}(x)}{\lambda m_{j,\lambda}}.
\tag{RBQ.15}
\]

The commitment boundary uses the task--then--return branch. Direct-return risk
is retained as a separate post-commit certificate because hybrid goal switching
and the task-service reset make the branch feasible sets non-nested. Hence the
effective commitment-requirement error is

\[
g_h(x)=g_{\rm task+return}(x).
\tag{RBQ.16}
\]

Equations (RBQ.15)--(RBQ.16) are **exact inequalities** conditional on the
pointwise envelopes, not statistical coverage statements.

### Step 7. The boundary object is a stopped-margin functional, not a metric

Let \(m(x)>0\) be the exact manager's continue margin at an oracle-shadow
decision state. For a conservative encoded requirement, first disagreement can
occur only when

\[
0<m(x)\le g_h(x).
\]

Therefore the one-sided first-disagreement probability is bounded by

\[
\Pr(\widehat\tau<\tau^*)
\le
\mathfrak B_{\mu_{\mathrm{stop}}}(g_h)
:=\int
\mathbf 1\{0<m(x)\le g_h(x)\}
\,\mu_{\mathrm{stop}}(dx).
\tag{RBQ.17}
\]

The functional \(\mathfrak B\) is nonlinear and depends on the stopped law and
the true margin. It is not a pseudometric on one-step interfaces. Calling
\(d_{\Lambda,\mathcal F}\) itself “boundary weighted” would therefore mix
objects and obscure the sequential mechanism.

Combining (RBQ.17) with bounded cycle throughput and the existing stranding
coordinate coupling yields the established Pareto rectangle. This final step is
a **proposition under Q8**, not an empirical Pareto result.

### Step 8. General quotient statistical complexity — open obligation

For a fixed finite feature map \(\phi(h(q))\in\mathbb R^d\), define

\[
\Sigma_s=\mathbb E_{\rho_s}[\phi\phi^\top],
\qquad
m_{\nu,\lambda}
=\int\phi(h(q))\,\bar\eta_{\nu,\lambda}(dq),
\]

and the restricted coefficient

\[
C_{\lambda,\phi}
=m_{\nu,\lambda}^\top
\Sigma_s^\dagger m_{\nu,\lambda}
=\sup_{v:\,v^\top\Sigma_s v>0}
\frac{(v^\top m_{\nu,\lambda})^2}
{v^\top\Sigma_s v}.
\tag{RBQ.18}
\]

Equation (RBQ.18) is an **identity** for a restricted chi-square/leverage
coefficient. A future theorem may bound the stochastic part of
\(\bar\eta\delta\) by a term of order

\[
\widetilde O\!\left(
\sigma_\lambda
\sqrt{C_{\lambda,\phi}d_{\mathrm{eff}}/n}
\right),
\tag{RBQ.19}
\]

plus approximation, nuisance, and dependent-trajectory terms. Equation
(RBQ.19) is currently a **target form**, not a proved rate. It requires a fixed
sampling model, regularization rule, noise/moment condition, and a matching lower
bound. Existing restricted-OPE and covariate-shift KRR results make those details
mandatory and also create a strong prior-art collision.

### Step 9. Finite linear Gaussian witness slice — exact theorem

The fixed-design slice of (RBQ.19) is closed by Theorem 16 in the proof package.
For \(Y=X\theta+\varepsilon\), \(\varepsilon\sim N(0,\sigma^2I)\), and target
risk functional \(J=m^\top\theta\):

\[
J\text{ is identifiable}
\quad\Longleftrightarrow\quad
m\in\operatorname{Range}(X^\top).
\tag{RBQ.20}
\]

When identifiable, the exact minimax mean-squared error is

\[
\sigma^2m^\top(X^\top X)^\dagger m
=\frac{\sigma^2}{n}C_{X,m},
\qquad
C_{X,m}=n,m^\top(X^\top X)^\dagger m.
\tag{RBQ.21}
\]

When (RBQ.20) fails, the component
\(P_{\operatorname{Null}(X)}m\) yields two source-indistinguishable parameters
and a positive minimax floor. A balanced endpoint design with
\(\phi(q)=(1,q)\), source \(q\in\{-1,1\}\), and target \(q=0\) has singular raw
target support but \(C_{X,m}=1\), proving that declared linear witness structure
can strictly improve over raw-cell plug-in. A least-favourable Gaussian pair
then gives irreversible sign-decision error at least
\(\Phi(-1/2)\approx0.3085\) at the local
\(\sigma\sqrt{C_{X,m}/n}\) scale.

This is an **exact theorem for a fixed-nuisance linear slice**. It does not prove
the RKHS/neural/dependent-trajectory target (RBQ.19), but it closes the minimum
attainability/necessity example and makes the representation rule precise:
target risk-occupation feature means must lie in the source feature covariance
range, with small leverage.

### Step 10. Two-layer orthogonal score — exact population identity

The safety-composition layer and primitive-transition layer require separate
conditional residuals. For a state continuation \(f\) and an interface
continuation \(g\), define

\[
A_{f,g}(x)=L g(x)-f(x),
\qquad
B_{\lambda,f,g}(q,Y)=\Gamma_{\lambda,f}(Y)-g(q).
\tag{RBQ.22}
\]

Let \(\theta_\lambda=\nu\psi_\lambda\). Define the population score

\[
\mathcal S_\lambda(f,g;v,w)
=\nu f
+\mathbb E_{\rho_X}[v(X)A_{f,g}(X)]
+\mathbb E_{\rho_QK}[w(Q)B_{\lambda,f,g}(Q,Y)].
\tag{RBQ.23}
\]

With the exact ratios \((v_\lambda,w_\lambda)\), change of measure gives

\[
\begin{aligned}
\mathcal S_\lambda(f,g;v_\lambda,w_\lambda)
&=\nu f
+\eta^X_{\nu,\lambda}(Lg-f)
+\eta^Q_{\nu,\lambda}(\mathcal K_\lambda f-g)\\
&=\nu f
+\eta^X_{\nu,\lambda}(\mathcal T_\lambda f-f)\\
&=\nu f+\nu\psi_\lambda-\nu f\\
&=\theta_\lambda.
\end{aligned}
\tag{RBQ.24}
\]

The penultimate equality uses
\(\eta^X=\nu+\eta^X M_\lambda\) and
\(\eta^Xr_\lambda=\nu\psi_\lambda\). Equation (RBQ.24) is an **exact identity
for every admissible \(f,g\)**, not an approximation.

Now set

\[
f^*=\psi_\lambda,
\qquad
g^*=\mathcal K_\lambda\psi_\lambda.
\tag{RBQ.25}
\]

Then \(A_{f^*,g^*}(x)=0\) pointwise, and

\[
\mathbb E[B_{\lambda,f^*,g^*}(Q,Y)\mid Q]=0.
\]

Consequently, for arbitrary square-integrable ratio candidates
\(\widetilde v,\widetilde w\),

\[
\mathcal S_\lambda(f^*,g^*;\widetilde v,\widetilde w)
=\theta_\lambda.
\tag{RBQ.26}
\]

Equations (RBQ.24) and (RBQ.26) establish **block double robustness**:
the score is exact if either both ratio nuisances are exact or both continuation
nuisances are exact. Exactness of only one member inside either block is not
claimed.

### Step 11. Exact product-bias identity

For arbitrary candidates
\((\widehat f,\widehat g,\widehat v,\widehat w)\),
subtract (RBQ.24) evaluated at the same \((\widehat f,\widehat g)\). Conditional
expectation over \(Y\mid Q\) yields

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
\tag{RBQ.27}
\]

Because \(L g^*=f^*\) and \(\mathcal K_\lambda f^*=g^*\), the two residuals in
(RBQ.27) are precisely

\[
L(\widehat g-g^*)-(\widehat f-f^*),
\qquad
\mathcal K_\lambda(\widehat f-f^*)-(\widehat g-g^*).
\tag{RBQ.28}
\]

Thus no isolated first-order model or ratio error remains. This cancellation is
an **identity**, not an appeal to informal Neyman orthogonality.

### Step 12. L2 product bound and cross-fitted root-n condition

Cauchy--Schwarz applied separately to the two terms in (RBQ.27) gives

\[
\boxed{
\begin{aligned}
|\operatorname{Bias}(\widehat{\mathcal S}_\lambda)|
&\le
\|\widehat v-v_\lambda\|_{2,\rho_X}
\|L\widehat g-\widehat f\|_{2,\rho_X}\\
&\quad+
\|\widehat w-w_\lambda\|_{2,\rho_Q}
\|\mathcal K_\lambda\widehat f-\widehat g\|_{2,\rho_Q}.
\end{aligned}}
\tag{RBQ.29}
\]

On a validation fold \(O_i=(X_i,Q_i,Y_i)\), the cross-fitted estimator is

\[
\widehat\theta_\lambda
=\nu\widehat f
+\mathbb P_n\left[
\widehat v(X)A_{\widehat f,\widehat g}(X)
+\widehat w(Q)B_{\lambda,\widehat f,\widehat g}(Q,Y)
\right].
\tag{RBQ.30}
\]

Conditional on the training folds, Q12 makes its centered empirical term an
ordinary iid sample average (or supplies an episode-level CLT). Under a uniform
\(2+\epsilon\) moment condition and score \(L_2\) convergence, it is
\(O_p(n^{-1/2})\). Therefore a sufficient condition for negligible root-\(n\)
bias is

\[
\begin{aligned}
&\|\widehat v-v_\lambda\|_2
\|L\widehat g-\widehat f\|_2=o_p(n^{-1/2}),\\
&\|\widehat w-w_\lambda\|_2
\|\mathcal K_\lambda\widehat f-\widehat g\|_2=o_p(n^{-1/2}).
\end{aligned}
\tag{RBQ.31}
\]

The common \(n^{-1/4}\)-per-nuisance heuristic is one sufficient special case,
not a necessary rate. A full asymptotic-normality theorem additionally requires
fold aggregation, nondegenerate limiting variance, and control of estimated
\(\nu\), queryable-\(L\) integration, risk grid selection, and positivity.

### Step 13. Primitive-law pathwise derivative and canonical gradient

Under Q13, take a regular conditional submodel \(P_t(dY\mid q)\) through \(P\)
with score \(s(q,Y)\) satisfying

\[
\mathbb E[s(Q,Y)\mid Q]=0,
\qquad
\mathbb E[s(Q,Y)^2]<\infty.
\tag{RBQ.32}
\]

For fixed \(f\), differentiation under the conditional integral gives

\[
\dot{\mathcal K}_{\lambda}f(q)
=\mathbb E[\Gamma_{\lambda,f}(Y)s(q,Y)\mid Q=q].
\tag{RBQ.33}
\]

Differentiate
\(\psi_{\lambda,t}=L\mathcal K_{\lambda,t}\psi_{\lambda,t}\) at \(t=0\).
The derivative through the continuation argument is \(M_\lambda\dot\psi\), so

\[
(I-M_\lambda)\dot\psi_\lambda
=L\dot{\mathcal K}_\lambda\psi_\lambda,
\qquad
\dot\psi_\lambda
=R_\lambda L\dot{\mathcal K}_\lambda\psi_\lambda.
\tag{RBQ.34}
\]

For \(\theta_\lambda(P)=\nu\psi_\lambda\), equations (RBQ.33)--(RBQ.34) imply

\[
\begin{aligned}
\dot\theta_\lambda[s]
&=\eta^X_{\nu,\lambda}L
\dot{\mathcal K}_\lambda\psi_\lambda\\
&=\mathbb E_{\rho_QP}
\left[w_\lambda(Q)\Gamma_{\lambda,\psi_\lambda}(Y)s(Q,Y)\right]\\
&=\mathbb E_{\rho_QP}[\varphi_\lambda(Q,Y)s(Q,Y)],
\end{aligned}
\tag{RBQ.35}
\]

where conditional centering gives

\[
\boxed{
\varphi_\lambda(q,Y)
=w_\lambda(q)
\left\{
\Gamma_{\lambda,\psi_\lambda}(Y)
-\mathcal K_\lambda\psi_\lambda(q)
\right\}.}
\tag{RBQ.36}
\]

The nonparametric conditional-law tangent space is
\(\{s\in L_2(\rho_QP):\mathbb E[s\mid Q]=0\}\). The function
\(\varphi_\lambda\) belongs to this space and represents every pathwise
derivative in (RBQ.35). It is therefore the **canonical gradient** in the Q13
model. Its efficiency bound is

\[
\boxed{
V_{\mathrm{eff},\lambda}
=\mathbb E_{\rho_Q}
\left[
w_\lambda(Q)^2
\operatorname{Var}
\{\Gamma_{\lambda,\psi_\lambda}(Y)\mid Q\}
\right].}
\tag{RBQ.37}
\]

If the marginal design law \(\rho_Q\) is unknown, scores depending only on
\(Q\) are orthogonal to (RBQ.36), while \(\theta_\lambda\) has zero derivative
in those directions. Thus (RBQ.36) remains canonical in the product model.

Define the one-observation estimating function

\[
m_\lambda(O;f,g,v,w)
=\nu f+v(X)A_{f,g}(X)+w(Q)B_{\lambda,f,g}(Q,Y),
\]

whose expectation is \(\mathcal S_\lambda\). At the truth, its centered value
reduces exactly to

\[
m_\lambda(O;f^*,g^*,v_\lambda,w_\lambda)-\theta_\lambda
=\varphi_\lambda(Q,Y)
\tag{RBQ.38}
\]

because the state-composition correction is identically zero.
Hence \(v_\lambda\) is a robustness/computation nuisance, not an additional
component of the efficient influence function when \(L\) is known. Any regular
estimator in Q13 has asymptotic variance at least
\(V_{\mathrm{eff},\lambda}\), and a cross-fitted Theorem 17 estimator attains
the bound only if its empirical score converges in \(L_2\) to (RBQ.36) and its
two product remainders are \(o_p(n^{-1/2})\).

For a bounded least-favourable score
\(s^*=\varphi_\lambda/\sqrt{V_{\mathrm{eff},\lambda}}\), or bounded
approximations to it, local alternatives \(P_{\pm a/\sqrt n}\) satisfy

\[
\theta(P_{\pm a/\sqrt n})
=\theta(P)\pm
\frac{a\sqrt{V_{\mathrm{eff},\lambda}}}{\sqrt n}
+o(n^{-1/2}).
\tag{RBQ.39}
\]

A threshold placed at \(\theta(P)\) therefore gives the standard LAN testing
floor

\[
\liminf_{n\to\infty}
\inf_{\widehat d_n}
\max_{\sigma\in\{-,+\}}
\Pr_{P_{\sigma a/\sqrt n}}
\{\widehat d_n\ne\sigma\}
\ge\Phi(-a).
\tag{RBQ.40}
\]

Equations (RBQ.36)--(RBQ.40) are **standard semiparametric efficiency and LAN
consequences specialized to the killed charger-hitting functional**. They close
the correct information scale but are not a safe oral novelty claim. The open
mathematical target is a nonstandard phase transition or sharper bound caused by
quotient support, approach to the exponential-transience boundary, or the
stopped ReturnManager margin law.

### Step 14. Exact quotient--transience information phase transition

Consider one noncharger state and one target risk-equivalence class. Conditional
on that class, the next outcome reaches the charger with probability \(p\) and
otherwise returns to the noncharger state. Every step has multiplicative factor
\(a=e^{\lambda c}>1\). Let the total source probability of the equivalence class
be \(\mu_h>0\). Define the killed transience gap

\[
\Delta=1-a(1-p)>0.
\tag{RBQ.41}
\]

The exact fixed point and risk occupation are

\[
\psi=ap+a(1-p)\psi
=\frac{ap}{\Delta},
\qquad
\eta=\frac{1}{\Delta}.
\tag{RBQ.42}
\]

On the informative source class, the non-normalized occupation ratio is
\(w=1/(\mu_h\Delta)\). The primitive witness takes values \(a\) on termination
and \(a\psi\) on continuation. Since

\[
1-\psi=\frac{1-a}{\Delta},
\]

Theorem 18's efficient variance for the raw MGF is exactly

\[
\boxed{
V_{\psi}
=\frac{p(1-p)a^2(a-1)^2}
{\mu_h\Delta^4}.}
\tag{RBQ.43}
\]

For the log-MGF, the delta method gives

\[
\boxed{
V_{\log\psi}
=\frac{V_\psi}{\psi^2}
=\frac{(1-p)(a-1)^2}
{\mu_h p\Delta^2}.}
\tag{RBQ.44}
\]

If the deployed scalar requirement is
\(U=(\log\psi+b)/\lambda\), where \(b\) is fixed, then

\[
V_U=\frac{(1-p)(a-1)^2}
{\mu_h p\lambda^2\Delta^2}.
\tag{RBQ.45}
\]

Equations (RBQ.43)--(RBQ.45) can also be checked directly from Bernoulli Fisher
information. Since

\[
\frac{d\psi}{dp}=-\frac{a(a-1)}{\Delta^2},
\qquad
\frac{d\log\psi}{dp}=-\frac{a-1}{p\Delta},
\tag{RBQ.46}
\]

and only \(n\mu_h\) observations are informative, the efficient asymptotic
variances are the squared derivatives times \(p(1-p)/\mu_h\).

For fixed \(a>1\) and \(p\to1-1/a\) from above, all numerator factors remain
nonzero. Therefore:

\[
\begin{array}{ll}
n\mu_h\Delta^4\to\infty
&\text{is the raw-MGF absolute-consistency scale},\\
n\mu_h\Delta^2\to\infty
&\text{is the log-risk/requirement consistency scale}.
\end{array}
\tag{RBQ.47}
\]

At a ReturnManager threshold, local margins of order

\[
s_n
=\frac{a-1}{\lambda\Delta}
\sqrt{\frac{1-p}{n\mu_h p}}
\tag{RBQ.48}
\]

have a nonvanishing LAN sign-decision error. In particular, alternatives at
\(\pm z s_n\) inherit the \(\Phi(-z)\) lower floor from (RBQ.40).

The coverage \(\mu_h\) is the **total source mass of an exactly shared
risk-observable quotient class**, not necessarily the raw target-interface cell
mass. Hence a target raw cell may have zero source mass yet remain regularly
estimable if its primitive Bernoulli parameter is constrained to equal that of a
source-observed interface in the same class. Without that structural equality,
zero raw support remains nonidentified.

This slice establishes an exact joint phase transition among risk severity,
transience, quotient coverage, and irreversible decision resolution. The algebra
is new to this package but uses classical Bernoulli information and delta-method
arguments. An oral-level theorem still requires a multi-state generalization
showing when \(\Delta\) is replaced by the Perron spectral gap and when
conditional noise projected onto the Perron mode is nondegenerate.

### Step 15. Multi-state Perron information theorem

Under Q14, the resolvent decomposition is the **exact identity**

\[
R_\Delta=(I-M_\Delta)^{-1}
=\frac{z_\Delta\ell_\Delta^\top}{\Delta}+H_\Delta.
\tag{RBQ.49}
\]

Define

\[
a_\Delta=\nu z_\Delta,
\qquad
b_\Delta=\ell_\Delta^\top r_\Delta.
\tag{RBQ.50}
\]

Uniform boundedness of \(H_\Delta\) gives

\[
\psi_\Delta
=\frac{b_\Delta}{\Delta}z_\Delta+O(1),
\qquad
\eta^X_\Delta
=\frac{a_\Delta}{\Delta}\ell_\Delta^\top+O(1),
\qquad
\theta_\Delta=\nu\psi_\Delta
=\frac{a_\Delta b_\Delta}{\Delta}+O(1).
\tag{RBQ.51}
\]

Here and below, \(O(1)\) is uniform over the finite state/interface space. Work
on the exact risk-observable quotient interface \(q=(x,\bar u)\), with target
kernel \(L_\Delta(q\mid x)\) and source design mass \(\rho_{Q,\Delta}(q)\).
Define the leading projected primitive variable and its conditional variance

\[
Z_\Delta(q,Y)
=e^{\lambda C}\mathbf1\{X'\notin G_C\}z_\Delta(X'),
\qquad
\sigma_\Delta^2(q)
=\operatorname{Var}\{Z_\Delta(q,Y)\mid Q=q\}.
\tag{RBQ.52}
\]

The leading quotient-coverage/noise coefficient is

\[
\boxed{
\mathcal C_\Delta
=\sum_q
\frac{
\{\ell_\Delta(x)L_\Delta(q\mid x)\}^2
}{\rho_{Q,\Delta}(q)}
\sigma_\Delta^2(q).}
\tag{RBQ.53}
\]

Terms with zero target numerator are defined as zero; a positive numerator and
zero source denominator violates Q14 rather than contributing infinity silently.

From (RBQ.51), the exact primitive witness admits

\[
\Gamma_{\lambda,\psi_\Delta}(Y)
=\frac{b_\Delta}{\Delta}Z_\Delta(q,Y)
+W_\Delta(q,Y),
\tag{RBQ.54}
\]

where \(W_\Delta\) has uniformly bounded conditional second moment. Likewise,

\[
w_\Delta(q)
=\frac{a_\Delta}{\Delta}
\frac{\ell_\Delta(x)L_\Delta(q\mid x)}
{\rho_{Q,\Delta}(q)}+O(1).
\tag{RBQ.55}
\]

Assume \(\mathcal C_\Delta\to\mathcal C_0\in(0,\infty)\), and that the
conditional covariance between \(Z_\Delta\) and \(W_\Delta\) is uniformly
bounded. Substitution of (RBQ.54)--(RBQ.55) into the exact efficient variance
(RBQ.37) yields

\[
\boxed{
\Delta^4V_{\theta,\Delta}
\longrightarrow
a_0^2b_0^2\mathcal C_0.}
\tag{RBQ.56}
\]

Because \(\Delta\theta_\Delta\to a_0b_0\), the delta method then gives the
scale-free log-risk limit

\[
\boxed{
\Delta^2V_{\log\theta,\Delta}
\longrightarrow\mathcal C_0,
\qquad
\Delta^2V_{U,\Delta}
\longrightarrow\frac{\mathcal C_0}{\lambda^2}.}
\tag{RBQ.57}
\]

Thus the scalar exponents from Theorem 19 survive in a multi-state system, while
the constant becomes an exact product of quotient coverage and primitive noise
projected onto the critical right Perron mode. For pointwise ReturnManager risk,
take \(\nu=\delta_x\); positivity of \(z_0(x)\) guarantees the same log-risk
limit.

There is a distinct **projected-noise-degenerate regime**. If, for every
target-leading quotient interface and all sufficiently small \(\Delta\),
\(Z_\Delta(q,Y)\) is conditionally deterministic, then the centered leading
term in (RBQ.54) vanishes. Under the bounded-moment part of Q14,

\[
 V_{\theta,\Delta}=O(\Delta^{-2}),
\qquad
V_{\log\theta,\Delta}=O(1).
\tag{RBQ.58}
\]

Between (RBQ.56) and (RBQ.58), vanishing
\(\mathcal C_\Delta\) can create intermediate rates; no universal exponent is
claimed without specifying its decay. This degeneracy classification is
essential: the spectral gap alone does not determine Energy-risk difficulty.

For \(n\) iid quotient-interface primitives, the regular efficient
standard-error scale suggested by (RBQ.57) in the nondegenerate regime is

\[
s_{n,\Delta}
=\frac{\sqrt{\mathcal C_0}}
{\lambda\sqrt n\,\Delta}\{1+o(1)\}.
\tag{RBQ.59}
\]

For each fixed \(\Delta>0\), the least-favourable reduction in (RBQ.40)
converts margins of order \(s_{n,\Delta}\) into a nonvanishing
premature-return/stranding sign-error floor. Along a joint sequence
\(\Delta_n\downarrow0\), that same \(\Phi(-z)\) conclusion requires Q15; it does
**not** follow from pointwise LAN alone. Under Q15,
\(n\Delta_n^2/\mathcal C_0\to\infty\) is exactly the condition for the regular
efficiency scale in (RBQ.59) to vanish, not an unconditional consistency theorem
for arbitrary estimators. Source quotient coverage is already inside
\(\mathcal C_0\) through the inverse \(\rho_Q\) factor.

Equations (RBQ.49)--(RBQ.58) form a **finite-state critical-family proposition
under Q14**. Equation (RBQ.59) is its regular efficiency scale; its triangular-
array irreversible-decision consequence additionally requires Q15. None of
these statements is a neural or dependent-replay theorem. Perron resolvent
expansion and semiparametric efficiency are classical ingredients. The
potentially new paper-level object is their sharp composition with exact
risk-observable quotient coverage, the projected-noise phase split, and—only
under Q15—the irreversible stopped-margin lower consequence.

### Step 16. Cost-aware critical-mode information design

Under Q16, define the exact log-risk sensitivity of quotient interface \(q\) by

\[
\alpha_\Delta(q)
=
\frac{\eta^X_\Delta(x)L_\Delta(q\mid x)}{\theta_\Delta}
\sqrt{
\operatorname{Var}\{\Gamma_{\lambda,\psi_\Delta}(Y)\mid Q=q\}}
\ge0.
\tag{RBQ.60}
\]

If \(n_q>0\) independent primitives are acquired from every active stratum, the
attainable canonical variance for \(\log\theta_\Delta\) is

\[
\mathcal V_\Delta(\boldsymbol n)
=\sum_q\frac{\alpha_\Delta(q)^2}{n_q},
\qquad
\sum_q\kappa(q)n_q\le\mathsf B.
\tag{RBQ.61}
\]

This is the stratified form of (RBQ.37): with total sample size \(n\) and design
mass \(\rho_Q(q)=n_q/n\), division of the one-observation efficiency bound by
\(n\) gives (RBQ.61). Let

\[
S_\Delta
=\sum_q\alpha_\Delta(q)\sqrt{\kappa(q)}.
\tag{RBQ.62}
\]

Cauchy--Schwarz gives the exact budget lower bound

\[
\mathcal V_\Delta(\boldsymbol n)\mathsf B
\ge
\left(\sum_q
\frac{\alpha_\Delta(q)}{\sqrt{n_q}}
\sqrt{\kappa(q)n_q}\right)^2
=S_\Delta^2.
\tag{RBQ.63}
\]

Ignoring asymptotically negligible integer rounding, equality holds uniquely on
the positive-sensitivity strata at

\[
\boxed{
n_q^*
=\frac{\mathsf B}{S_\Delta}
\frac{\alpha_\Delta(q)}{\sqrt{\kappa(q)}},
\qquad
\inf_{\boldsymbol n}\mathcal V_\Delta(\boldsymbol n)
=\frac{S_\Delta^2}{\mathsf B}.}
\tag{RBQ.64}
\]

Zero-sensitivity strata receive no oracle second-stage mass; they still require a
pilot or an externally justified structural certificate before being removed.

Under Q14, let

\[
d_0(q)
=\ell_0(x)L_0(q\mid x)\sigma_0(q).
\tag{RBQ.65}
\]

For every leading nondegenerate interface, (RBQ.51)--(RBQ.54) imply
\(\Delta\alpha_\Delta(q)\to d_0(q)\). Consequently,

\[
\boxed{
\mathsf B\Delta^2
\inf_{\boldsymbol n}\mathcal V_\Delta(\boldsymbol n)
\longrightarrow
\left\{\sum_qd_0(q)\sqrt{\kappa(q)}\right\}^2,}
\tag{RBQ.66}
\]

and the limiting oracle count and cost priorities are respectively

\[
n_q^*\ \propto\
\frac{\ell_0(x)L_0(q\mid x)\sigma_0(q)}{\sqrt{\kappa(q)}},
\qquad
\frac{\kappa(q)n_q^*}{\mathsf B}\ \longrightarrow\
\frac{d_0(q)\sqrt{\kappa(q)}}
{\sum_jd_0(j)\sqrt{\kappa(j)}}.
\tag{RBQ.67}
\]

Thus neither TD error, visitation, intervention count, nor raw energy variance is
the correct standalone replay priority. The leading statistic is conditional
noise in the **right Perron continuation mode**, multiplied by target left-mode
occupation and adjusted for acquisition cost.

There is also a finite pilot-stability guarantee. Suppose a separate pilot gives
\(\widehat\alpha(q)>0\) on every active stratum and

\[
\max_q\left|\frac{\widehat\alpha(q)}{\alpha_\Delta(q)}-1\right|
\le\varepsilon<1.
\]

Allocate the remaining budget \(\mathsf B_1=\mathsf B-\mathsf B_0\) by (RBQ.64)
with \(\widehat\alpha\). Then, conditionally on the pilot event,

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
\tag{RBQ.68}
\]

Indeed, with weights
\(p_q=\alpha_\Delta(q)\sqrt{\kappa(q)}/S_\Delta\) and ratios
\(r_q=\widehat\alpha(q)/\alpha_\Delta(q)\), the first variance ratio is exactly
\((\sum_qp_qr_q)(\sum_qp_q/r_q)\). Cauchy--Schwarz gives its lower bound;
\(r_q\in[1-\varepsilon,1+\varepsilon]\) gives the upper bound. Hence a pilot with
\(\mathsf B_0/\mathsf B\to0\) and uniform relative error \(o_p(1)\) is
oracle-efficient. Proving those relative-error rates while learning the quotient,
Perron vectors, and conditional projected variances is a separate open theorem.

Equations (RBQ.60)--(RBQ.68) are a cost-aware Neyman-allocation theorem
specialized to the stopped log-EIRR canonical gradient. The allocation algebra is
classical; its role here is to expose the exact algorithm-design primitive implied
by the critical information geometry. It does not yet establish an adaptive
learned-quotient algorithm or a navigation Pareto improvement.

### Step 17. Singularity-free adaptive critical design

Theorem 21 appears at first to require estimating the divergent exact
sensitivity \(\alpha_\Delta(q)\). Q17 reveals a cancellation. Define

\[
d_\Delta(q)
=\ell_\Delta(x)L_\Delta(q\mid x)\sigma_\Delta(q),
\qquad
\sigma_\Delta(q)^2
=\operatorname{Var}\{A(Y)^\top z_\Delta\mid q\}.
\tag{RBQ.69}
\]

Uniform versions of (RBQ.51)--(RBQ.54), together with the active variance floor,
give a constant \(K_c<\infty\), independent of \(\Delta\), such that

\[
\max_{q\in\mathcal A}
\left|
\frac{\Delta\alpha_\Delta(q)}{d_\Delta(q)}-1
\right|
\le K_c\Delta.
\tag{RBQ.70}
\]

Now estimate the conditional continuation means and their first two projected
moments from the pilot, build \(\widehat M_\Delta\), normalize its leading left
and right eigenvectors as in Q14, and form

\[
\widehat d_\Delta(q)
=\widehat\ell_\Delta(x)L_\Delta(q\mid x)
\widehat\sigma_\Delta(q).
\tag{RBQ.71}
\]

Let \(D\) be the quotient-state dimension, \(Q=|\mathcal A|\), and

\[
r_m(\delta)
=\sqrt{\frac{\log\{2Q(D+2)/\delta\}}{m_{\min}}}.
\tag{RBQ.72}
\]

Hoeffding concentration of the bounded conditional first/second moments,
finite-dimensional perturbation theory for a uniformly conditioned simple
eigenvalue, and the active positivity floors imply constants \(K_p,m_0\),
uniform in small \(\Delta\), for which

\[
\Pr\left{
\max_{q\in\mathcal A}
\left|\frac{\widehat d_\Delta(q)}{d_\Delta(q)}-1\right|
>K_pr_m(\delta)
\right}
\le\delta,
\qquad m_{\min}\ge m_0.
\tag{RBQ.73}
\]

The important spectral quantity in (RBQ.73) is the norm of the Perron reduced
resolvent, not the distance from the Perron root to one. For normal matrices it
is the reciprocal eigengap; for nonnormal matrices it also captures
pseudospectral/eigenvector conditioning. Under Q17 it does not contain
\(1/\Delta\).

Because allocation is invariant to multiplying every priority by the same
positive number, use \(\widehat d_\Delta\) directly in (RBQ.64). Combining
(RBQ.70)--(RBQ.73), there is a uniform constant \(K\) such that, with probability
at least \(1-\delta\), the proxy priorities have relative error at most

\[
\varepsilon_{m,\Delta}(\delta)
=K\{r_m(\delta)+\Delta\}
\tag{RBQ.74}
\]

against the exact \(\alpha_\Delta\), after the irrelevant common factor
\(1/\Delta\) is restored. Whenever \(\varepsilon_{m,\Delta}<1\), Theorem 21
therefore gives the adaptive oracle inequality

\[
\boxed{
\frac{\mathcal V_\Delta(\widehat{\boldsymbol n})}
{\inf_{\boldsymbol n:\,\mathrm{cost}\le\mathsf B}
\mathcal V_\Delta(\boldsymbol n)}
\le
\frac{\mathsf B}{\mathsf B-\mathsf B_0}
\frac{1+\varepsilon_{m,\Delta}(\delta)}
{1-\varepsilon_{m,\Delta}(\delta)}}
\quad\text{with probability at least }1-\delta.
\tag{RBQ.75}
\]

Consequently, along a critical family,

\[
m_{\min}\to\infty,
\qquad
\mathsf B_0/\mathsf B\to0,
\qquad
\Delta\to0
\tag{RBQ.76}
\]

are sufficient for adaptive oracle efficiency. There is deliberately **no**
requirement \(m_{\min}\Delta^2\to\infty\). In contrast, Theorem 20 shows that
vanishing regular error for the final log-risk value requires
\(\mathsf B\Delta^2\to\infty\) up to its coverage/noise constant. Thus learning
where to sample is asymptotically easier than learning the near-critical risk
value itself.

This separation fails if the effective Perron separation collapses, an active projected
variance tends to zero, or quotient recovery is learned from the same outcomes.
Those cases require different perturbation rates and may eliminate the claimed
singularity-free pilot scale.

### Step 18. Conditioned collapse phase and local necessity

Under Q18, abbreviate the worst active quantities by

\[
\mathfrak g=\mathfrak g_\Delta,
\quad
\chi_*=\max_q\chi_\Delta(q),
\quad
\varkappa_*=\max_q\varkappa_\Delta(q),
\quad
b_*=\max_q b_\Delta(q),
\quad
\sigma_*^2=\min_q\sigma_\Delta(q)^2.
\tag{RBQ.77}
\]

Let

\[
r_m(\delta)
=\sqrt{\frac{\log\{cQD/\delta\}}{m_{\min}}},
\qquad
e_M=r_m(\delta)+\tau_M.
\tag{RBQ.78}
\]

The reduced-resolvent eigenvector perturbation equations give, on the bounded-
moment event and while \(e_M/\mathfrak g\) is sufficiently small,

\[
\|\widehat z-z\|+\|\widehat\ell-\ell\|
\le K_M\frac{e_M}{\mathfrak g}.
\tag{RBQ.79}
\]

For \(s_q(z)=\{z^\top\Sigma(q)z\}^{1/2}\), exact differentiation yields

\[
\nabla_z\log s_q(z)
=\frac{\Sigma(q)z}{z^\top\Sigma(q)z},
\qquad
\|\nabla_z\log s_q(z)\|=\chi_\Delta(q).
\tag{RBQ.80}
\]

Thus mode error is amplified by \(\chi_*\). At the true direction, an empirical-
Bernstein bound for the sample variance gives relative projected-standard-
deviation error of order

\[
h_m(\delta)
=K_V\left\{
\sqrt{\frac{\varkappa_*\log(cQ/\delta)}{m_{\min}}}
+\frac{b_*^2\log(cQ/\delta)}{m_{\min}}
\right\}.
\tag{RBQ.81}
\]

Combining the Perron component, directional standard deviation, fixed-direction
moment estimate, and quotient bias gives the deterministic/high-probability
priority perturbation scale

\[
E_{m,\Delta}(\delta)
=K\left[
\frac{(1+\chi_*)\{r_m(\delta)+\tau_M\}}{\mathfrak g}
+h_m(\delta)+\tau_\sigma
\right].
\tag{RBQ.82}
\]

When \(\sigma_*\) also vanishes, the critical approximation itself changes.
From (RBQ.54),

\[
\Delta^2\operatorname{Var}(\Gamma\mid q)
=b_\Delta^2\sigma_\Delta(q)^2+O(\Delta),
\]

so the relative approximation in (RBQ.70) becomes

\[
\max_q
\left|
\frac{\Delta\alpha_\Delta(q)}{d_\Delta(q)}-1
\right|
\le K_c\frac{\Delta}{\sigma_*^2}
\tag{RBQ.83}
\]

provided \(\Delta/\sigma_*^2\) is small. The full sufficient phase parameter is

\[
\boxed{
\Omega_{m,\Delta}(\delta)
=E_{m,\Delta}(\delta)
+K_c\frac{\Delta}{\sigma_*^2}.}
\tag{RBQ.84}
\]

If \(\Omega_{m,\Delta}(\delta)<1\), replacing
\(\varepsilon_{m,\Delta}\) by \(\Omega_{m,\Delta}\) in (RBQ.75) gives the same
adaptive oracle inequality. Hence \(\Omega_{m,\Delta}(\delta)\to0\) and
\(\mathsf B_0/\mathsf B\to0\) are sufficient for critical-proxy oracle
efficiency.

The effective-separation/anisotropy factor is locally necessary in a regular
two-state slice. Let

\[
U=2^{-1/2}
\begin{bmatrix}1&1\\1&-1\end{bmatrix},
\quad
M_t=U
\begin{bmatrix}\rho&t\\t&\rho-\mathfrak g\end{bmatrix}
U^\top,
\quad
\Sigma_\chi=U
\begin{bmatrix}\chi^{-2}&\chi^{-1}\\
\chi^{-1}&1\end{bmatrix}
U^\top.
\tag{RBQ.85}
\]

At \(t=0\), the scale-invariant priority for the first state satisfies the exact
local identity

\[
\left.\frac{d}{dt}\log d_t\right|_{t=0}
=\frac{1+\chi}{\mathfrak g}.
\tag{RBQ.86}
\]

A bounded Rademacher pilot coordinate with mean \(t\) has one-observation KL
between means \(t\) and \(-t\) equal to
\(t\log\{(1+t)/(1-t)\}=2t^2+O(t^4)\). Choosing

\[
t_\pm=\pm a\frac{\mathfrak g}{1+\chi}
\tag{RBQ.87}
\]

creates constant-order log-priority separation while the \(m\)-sample KL is
\(O\{ma^2\mathfrak g^2/(1+\chi)^2\}\). Le Cam's lemma therefore gives a
nonvanishing priority/allocation error floor whenever

\[
\boxed{
\frac{m\mathfrak g^2}{(1+\chi)^2}=O(1).}
\tag{RBQ.88}
\]

The same construction shows that deterministic quotient/operator bias must obey
\((1+\chi)\tau_M/\mathfrak g\to0\) if a critical-mode priority is to be recovered
uniformly.

There is **no universal small-\(\sigma\) sample exponent**. In a Gaussian scale
family \(Z=\sigma G\), Fisher information for \(\log\sigma\) is exactly two per
sample, independent of \(\sigma\). For \(Z\sim\operatorname{Bernoulli}(p)\),
\(\sigma^2=p(1-p)\), standardized fourth moment is asymptotic to \(1/p\), and
at \(mp=O(1)\) the probability of observing no rare event stays bounded away
from zero. The former needs only \(m\to\infty\) for relative scale estimation;
the latter needs \(mp\to\infty\), equivalently \(m\sigma^2\to\infty\). Thus
\(\varkappa_*\) and \(b_*\), not \(\sigma_*\) alone, are indispensable phase
coordinates.

Equation (RBQ.84) is a sufficient conditioned phase diagram. Equation (RBQ.88)
is a matching local necessity statement for the effective-separation/anisotropy
coordinate. No universal matching lower bound is claimed from
\((\sigma_*,\varkappa_*,b_*)\) without fixing a primitive distribution class.

### Step 19. Stopped margin converts information into Pareto rates

For a frozen oracle-shadow decision query \(t\), let
\(E_t=\widehat U_t-U_t\) and suppose

\[
\Pr(|E_t|>r\mid\mathcal H_t)
\le 2\exp\{-r^2/(2V_t)\}.
\tag{RBQ.89}
\]

For \(J\) declared queries, a union bound gives the simultaneous radius

\[
\varepsilon_t(\alpha)
=\sqrt{2V_t\log(2J/\alpha)}
\tag{RBQ.90}
\]

with failure probability at most \(\alpha\). Substitution into Theorem 12 yields

\[
\Pr(\tau_\Delta<\tau_{\rm end})
\le
\alpha+
\{2\log(2J/\alpha)\}^{\kappa/2}
\sum_{t=1}^J C_tV_t^{\kappa/2},
\tag{RBQ.91}
\]

and, under its local boundary-loss assumption,

\[
\mathbb E[(L^{\widehat d}-L^{d^\star})_+]
\le B_L\alpha+L_D
\{2\log(2J/\alpha)\}^{(\kappa+1)/2}
\sum_{t=1}^J C_tV_t^{(\kappa+1)/2}.
\tag{RBQ.92}
\]

The stranding and bounded-throughput deviations are at most respectively one
and \(B_q\) times the right-hand side of (RBQ.91). These are certified upper
radii, not a claim that their actual changes have a favorable sign.

For one scalar query, Theorem 21 and the Perron limit give

\[
V_\Delta^*
=\frac{\{\sum_q\alpha_\Delta(q)\sqrt{\kappa(q)}\}^2}{\mathsf B}
=\frac{\mathcal J_0+o(1)}{\mathsf B\Delta^2},
\quad
\mathcal J_0
=\left\{\sum_qd_0(q)\sqrt{\kappa(q)}\right\}^2.
\tag{RBQ.93}
\]

If Theorem 23 gives variance ratio \(R_\Omega\), then replacing \(V^*\) by the
adaptive design multiplies the certified first-disagreement/Pareto radius by at
most \(R_\Omega^{\kappa/2}\) and the boundary-loss radius by at most
\(R_\Omega^{(\kappa+1)/2}\). Therefore the critical rates are

\[
O\{(\mathsf B\Delta^2)^{-\kappa/2}\},
\qquad
O\{(\mathsf B\Delta^2)^{-(\kappa+1)/2}\},
\tag{RBQ.94}
\]

up to confidence logarithms and the coefficient \(\mathcal J_0\). Vanishing
decision error still requires \(\mathsf B\Delta^2\to\infty\); singularity-free
pilot allocation does not remove the information cost of resolving the risk
value.

These exponents are exact for a one-epoch Gaussian-shift plug-in slice. Let
\(E=vG\), \(G\sim N(0,1)\), independent of the oracle score, and for
\(0\le r\le r_0\) let

\[
\Pr(|S|\le r)=(r/r_0)^\kappa
\tag{RBQ.95}
\]

with symmetric sign. The plug-in action uses \(S-E\). Conditional on \(E=e\),

\[
\Pr\{S(S-E)\le0\mid E=e\}
=\frac12\min\{(|e|/r_0)^\kappa,1\}.
\tag{RBQ.96}
\]

If the disagreement loss is \(L_D|S|\), its conditional expectation is

\[
\frac{L_D\kappa}{2(\kappa+1)r_0^\kappa}
\min\{|e|,r_0\}^{\kappa+1}.
\tag{RBQ.97}
\]

As \(v/r_0\to0\), (RBQ.96)--(RBQ.97) are respectively asymptotic to constants
times \(v^\kappa\) and \(v^{\kappa+1}\). Hence the powers in (RBQ.94) cannot be
improved for this regular plug-in slice. This is not a universal minimax lower
bound over all decision algorithms.

### Step 20. One shared margin-aware Perron allocation

For \(p>0\), define the stopped multi-query information objective

\[
\mathcal R_p(\boldsymbol n)
=\sum_{t=1}^J\omega_t
\left\{\sum_{q=1}^Q\frac{a_{tq}^2}{n_q}\right\}^p,
\qquad
\sum_qc_qn_q\le\mathsf B.
\tag{RBQ.98}
\]

The Pareto first-disagreement bound uses \(p=\kappa/2\); the local
boundary-weighted loss bound uses \(p=(\kappa+1)/2\). For every \(p>0\),
\(\mathcal R_p\) is convex on the positive orthant. To see this, put
\(S(\boldsymbol n)=\sum_qb_q/n_q\). In direction \(u\),

\[
u^\top\nabla^2S^pu
=pS^{p-2}\left[
(p-1)\left\{\sum_q\frac{b_qu_q}{n_q^2}\right\}^2
+2S\sum_q\frac{b_qu_q^2}{n_q^3}
\right].
\tag{RBQ.99}
\]

Cauchy--Schwarz bounds the squared first sum by
\(S\sum_qb_qu_q^2/n_q^3\). When \(p<1\), multiplication by \(p-1<0\) reverses
the bound; in either case (RBQ.99) is nonnegative. Thus the KKT equations are
globally sufficient.

Every active optimum spends the full budget and obeys the fixed-point rule

\[
\boxed{
n_q^*
=\mathsf B
\frac{g_q(\boldsymbol n^*)}
{\sum_rg_r(\boldsymbol n^*)c_r},
\qquad
g_q(\boldsymbol n)
=\left[
\frac{\sum_t\omega_tV_t(\boldsymbol n)^{p-1}a_{tq}^2}{c_q}
\right]^{1/2}.}
\tag{RBQ.100}
\]

For \(p=1\), this becomes the closed form

\[
n_q^*
=\mathsf B
\frac{\sqrt{\sum_t\omega_ta_{tq}^2/c_q}}
{\sum_r\sqrt{c_r\sum_t\omega_ta_{tr}^2}}.
\tag{RBQ.101}
\]

When every query shares one critical gap and
\(a_{tq,\Delta}=d_{tq}\Delta^{-1}\{1+o(1)\}\), the common
\(\Delta^{-2p}\) factor cancels from (RBQ.100). Hence the normalized shared
allocation remains singularity-free under the same conditioning needed to learn
the \(d_{tq}\). If queries instead have distinct gaps \(\Delta_t\), its score is
asymptotically

\[
g_q^2
\asymp\frac1{c_q}
\sum_t\omega_t\Delta_t^{-2p}
\widetilde V_t^{p-1}d_{tq}^2,
\tag{RBQ.102}
\]

so queries nearer exponential transience receive more shared budget. Equation
(RBQ.100) is the theorem-derived replay/data-collection target: it combines
stopped-boundary occupation, critical left/right Perron sensitivity, conditional
tail noise, and acquisition cost. Multi-characteristic convex allocation and
compound optimal design are classical; only the stopped-risk/Perron coupling and
its adaptive phase can remain a contribution candidate.

### Step 21. Finite quotient recovery has a sharp separation boundary

Let \(N=|\mathcal U|\), and from \(m_u\) independent structure-fold samples form
\(\widehat\mu(u)\). With probability at least \(1-\delta\),

\[
\max_{u\in\mathcal U}
\|\widehat\mu(u)-\mu(u)\|_\infty
\le
r_Q(\delta)
=B_W\sqrt{\frac{2\log(2ND/\delta)}{m_{\min}}}.
\tag{RBQ.103}
\]

Connect \(u,u'\) when
\(\|\widehat\mu(u)-\widehat\mu(u')\|_\infty<\gamma/2\), and take connected
components. If \(r_Q<\gamma/4\), every within-class pair is connected and no
between-class pair is connected. Hence the exact quotient is recovered, and an
independent allocation fold may invoke Theorems 23--25 with
\(\tau_M=\tau_\sigma=0\) on the recovery event. The structure-fold requirement
is

\[
m_{\min}
\gtrsim
\frac{B_W^2}{\gamma^2}
\log\frac{ND}{\delta}.
\tag{RBQ.104}
\]

This order is necessary. With two raw interfaces and one Bernoulli witness,
compare

\[
H_0:\quad p_1=p_2=1/2,
\qquad
H_1:\quad p_1=1/2,\quad p_2=1/2+\gamma.
\tag{RBQ.105}
\]

The correct quotient merges the interfaces under \(H_0\) and separates them
under \(H_1\). For \(m\) observations from the second interface,

\[
\operatorname{KL}(P_0^m,P_1^m)
=-\frac m2\log(1-4\gamma^2)
=2m\gamma^2+O(m\gamma^4).
\tag{RBQ.106}
\]

Le Cam's inequality gives a nonvanishing quotient-recovery error whenever
\(m\gamma^2=O(1)\). Thus no network, clustering rule, or loss can uniformly
recover this quotient below the separation scale merely by changing its
architecture.

The neural implication is precise. An encoder/prototype head should approximate
\(\mu(u)\), not raw action identity: if

\[
\sup_u\|g_\theta(e_\theta(u))-\mu(u)\|_\infty<\gamma/4,
\tag{RBQ.107}
\]

then thresholding its decoded witness moments recovers the quotient. The loss
must supervise the stacked first/second risk-witness moments on a structure fold;
ordinary reconstruction or Euclidean contrastive proximity has no such
guarantee. In continuous interfaces, \(N\) must be replaced by metric entropy or
a declared function-class complexity, so (RBQ.104) is only the finite baseline.

### Step 22. Continuous quotient, Perron separation, and stopping form one joint phase

For query \(z_t\), let

\[
N_t(h)=\sum_{i=1}^n\mathbf 1\{d_Z(Z_i,z_t)\le h\},
\qquad
\widehat\mu_h(z_t)
=\frac{1}{N_t(h)}\sum_{i:d_Z(Z_i,z_t)\le h}W_i.
\tag{RBQ.108}
\]

When

\[
n c_Qh^{d_Q}\ge8\log(2J/\delta),
\tag{RBQ.109}
\]

Chernoff's inequality gives \(N_t(h)\ge n c_Qh^{d_Q}/2\) simultaneously,
except on an event of probability at most \(\delta/2\). Conditional
coordinatewise Hoeffding concentration and the Hölder bias then give, with
probability at least \(1-\delta\),

\[
\boxed{
\max_{t\le J}\|\widehat\mu_h(z_t)-\mu(z_t)\|_\infty
\le
\varepsilon_n(h,\delta)
:=L_\mu h^\alpha
+2B_W\sqrt{
\frac{\log(4JD/\delta)}{n c_Qh^{d_Q}}}.}
\tag{RBQ.110}
\]

Balancing the two terms yields

\[
h_n\asymp
\left\{\frac{B_W^2\log(4JD/\delta)}
{c_QL_\mu^2n}\right\}^{1/(2\alpha+d_Q)},
\qquad
\varepsilon_n
=\widetilde O\left(n^{-\alpha/(2\alpha+d_Q)}\right).
\tag{RBQ.111}
\]

This is a point-query statement on the declared risk quotient. Its dimension
is \(d_Q\), not the ambient raw-interface dimension, but that gain is an
assumption to be tested rather than a free consequence of using a neural
network.

Define the moment pseudometric

\[
\mathfrak d_W(z,z')=\|\mu(z)-\mu(z')\|_\infty,
\qquad
\widehat{\mathfrak d}_W(z,z')
=\|\widehat\mu_h(z)-\widehat\mu_h(z')\|_\infty.
\tag{RBQ.112}
\]

On the event (RBQ.110), the reverse triangle inequality gives the exact
sandwich

\[
|\widehat{\mathfrak d}_W(z_t,z_s)-\mathfrak d_W(z_t,z_s)|
\le2\varepsilon_n.
\tag{RBQ.113}
\]

Consequently, pooling only pairs with
\(\widehat{\mathfrak d}_W\le\rho_n\) introduces true witness distortion at most
\(\rho_n+2\varepsilon_n\). No connected-component claim is made: threshold
proximity need not be transitive in a continuum. The induced operator-moment
bias is therefore

\[
\tau_{M,n}\le C_M(\rho_n+2\varepsilon_n).
\tag{RBQ.114}
\]

Insert (RBQ.114) into the exact residual and log-grid propagation in
(RBQ.10)--(RBQ.16). If the relevant positive resolvent/log constants obey
\(A_{t,n}\le A_0/\Delta_n\), the representation contribution to every frozen
requirement radius is at most

\[
G_{t,n}\le
\frac{A_0C_M}{\Delta_n}(\rho_n+2\varepsilon_n).
\tag{RBQ.115}
\]

Under Q19's stopped margin, this gives the unconditional orders

\[
\Pr(\tau_\Delta<\tau_{\rm end})
\le \delta
+C\left\{
\frac{\rho_n+\varepsilon_n}{\Delta_n}
\right\}^{\kappa},
\tag{RBQ.116}
\]

and, under the local boundary-loss condition,

\[
\mathbb E[(L^{\widehat d}-L^{d^\star})_+]
\le B_L\delta
+C_L\left\{
\frac{\rho_n+\varepsilon_n}{\Delta_n}
\right\}^{\kappa+1}.
\tag{RBQ.117}
\]

The same quotient bias enters Theorem 23's learned-priority term as

\[
\Omega_{Q,n}
\le K\frac{(1+\chi_*)C_M
(\rho_n+2\varepsilon_n)}{\mathfrak g_n}.
\tag{RBQ.118}
\]

Thus, ignoring logarithms and the other explicitly retained pilot-moment terms,
a necessary sufficient-phase target for both the ReturnManager and normalized
critical replay design is

\[
\boxed{
\rho_n=o\{\min(\Delta_n,\mathfrak g_n)\},\qquad
n\Delta_n^{(2\alpha+d_Q)/\alpha}\to\infty,\qquad
n\mathfrak g_n^{(2\alpha+d_Q)/\alpha}\to\infty.}
\tag{RBQ.119}
\]

The statistical powers in (RBQ.119) are locally necessary. On
\(\mathsf Z=[0,1]^{d_Q}\), compare a Bernoulli conditional witness with mean
\(1/2\) to one with mean

\[
\frac12+a_n\phi\{(z-z_0)/h_n\},
\qquad a_n=c h_n^\alpha,
\tag{RBQ.120}
\]

where \(\phi\) is an \(\alpha\)-Hölder bump supported on the unit ball. The
two \(n\)-sample laws have

\[
\operatorname{KL}(P_0^n,P_1^n)
\le Cn h_n^{d_Q}a_n^2.
\tag{RBQ.121}
\]

Taking \(h_n\asymp n^{-1/(2\alpha+d_Q)}\) keeps (RBQ.121) bounded while the
pointwise moment separation is
\(a_n\asymp n^{-\alpha/(2\alpha+d_Q)}\). Le Cam's lemma therefore matches
(RBQ.111) up to logarithms. This single-query result alone does **not** imply a
random stopped-law margin lower bound.

For that step, Theorem 27 now uses a distributed strong-density
Hölder--margin Assouad hypercube under \(\alpha\kappa\le d_Q\). Put
\(s_n=a_n/\Delta_n\) and use
\(m_n\asymp s_n^\kappa h_n^{-d_Q}\) disjoint active cells. Each signed score
cell is mapped through the exact inverse of Theorem 19's scalar killed
requirement. Neighboring cell signs then change the primitive Bernoulli
termination probability by \(\Theta(\Delta_ns_n)=\Theta(a_n)\), so their
\(n\)-sample KL remains \(O(nh_n^{d_Q}a_n^2)\). Assouad's lemma supplies
expected Hamming error of order \(m_n\); multiplying by cell mass, and by
\(s_n\) for local boundary loss, gives

\[
\min\left\{1,
\left(\frac{a_n}{\Delta_n}\right)^\kappa\right\},
\qquad
\min\left\{1,
\left(\frac{a_n}{\Delta_n}\right)^{\kappa+1}\right\}.
\tag{RBQ.122}
\]

Theorem 15's cellwise one-decision embedding gives the same first order for at
least one normalized stranding/throughput coordinate. In a separate subfamily,
embedding the same indistinguishable bump as the
operator tilt in Theorem 23's two-state construction changes log priority by
order \(a_n/\mathfrak g_n\). Hence uniform recovery of both stopped decisions
and critical replay priorities cannot hold if either of the last two limits in
(RBQ.119) stays bounded.

Equations (RBQ.108)--(RBQ.122) are a proved iid local-mass Hölder upper slice,
a common exact-chart source-equals-target Assouad first-passage lower slice for
the \((d_Q,\Delta,\kappa)\) stopped/Pareto phase, and a separate priority lower
slice for \(\mathfrak g\). Generic nonparametric regression, plug-in margin
conversion, Assouad testing, and spectral perturbation are established
ingredients. Arbitrary pushed-overlap shift, neural encoders, unknown intrinsic
charts, and nuisance estimation remain outside the matching lower theorem.

### Step 23. Risk-neutral collapse does not imply superiority to risk-transformed OPE

Take two one-step charger-hitting interfaces with identical next charger state.
At \(q_0\), let \(C=1\); at \(q_1\), let \(C\) be zero or two with equal
probability. Their expected costs coincide, but

\[
\mathcal K_\lambda 1(q_1)-\mathcal K_\lambda 1(q_0)
=\frac{(e^\lambda-1)^2}{2}>0.
\tag{RBQ.123}
\]

Thus expected-reward bisimulation/KROPE may collapse a pair that the exact-risk
ReturnManager must distinguish. A threshold between the two log-MGFs witnesses
an actual action disagreement, not just prediction error.

For a separate coverage construction, let \(Q=(Z,V)\), make all risk witnesses
conditionally independent of nuisance \(V\) given \(Z\), let the source be
uniform over \(K\) nuisance values, and let the target use one. Then

\[
\chi^2(P_T^Q\|P_S^Q)=K-1,
\qquad
\chi^2(P_T^Z\|P_S^Z)=0.
\tag{RBQ.124}
\]

This proves a strict gap between raw-interface and risk-quotient coverage.
However, Theorem 6B also proves

\[
M_\lambda=\gamma D_vP_{\gamma,S}D_v^{-1}.
\tag{RBQ.125}
\]

An oracle given \(v\) and \(P_{\gamma,S}\) can therefore run ordinary discounted
KROPE/DICE on the transformed chain. Equations (RBQ.123)--(RBQ.124) reject
risk-neutral and raw baselines; (RBQ.125) prevents claiming algebraic
superiority over the correct oracle-risk baseline. The remaining estimator
question is whether the transform can be learned with the joint certified rate
in Step 22.

### Step 24. Learning Doob coordinates alone cannot improve the minimax order

For fixed \(\gamma>\rho(M)\), define

\[
v=(I-M/\gamma)^{-1}\mathbf1,\qquad
P_\gamma(i,j)=\frac{M(i,j)v_j}{\gamma v_i},\qquad
P_\gamma(i,\dagger)=\frac1{v_i}.
\tag{RBQ.126}
\]

This coordinate map is invertible:

\[
v_i=P_\gamma(i,\dagger)^{-1},
\qquad
M(i,j)=\gamma\frac{v_iP_\gamma(i,j)}{v_j}.
\tag{RBQ.127}
\]

It follows that a fixed raw-data experiment indexed by \(M\) contains exactly
the same statistical information when indexed by \((v,P_\gamma)\). For every
admissible \(\widehat M\),

\[
(I-\widehat M)^{-1}\widehat r
=D_{\widehat v}(I-\gamma\widehat P_{\gamma,S})^{-1}
D_{\widehat v}^{-1}\widehat r.
\tag{RBQ.128}
\]

Thus direct and transformed plug-in estimators are not merely asymptotically
equivalent: they are algebraically identical when built from the same primitive
estimate.

The transform can be ill-conditioned. Differentiating its defining equation
gives

\[
Dv_M[E]
=(I-M/\gamma)^{-1}(E/\gamma)v.
\tag{RBQ.129}
\]

If \(\|M\|_\infty/\gamma\le1-s\), then

\[
\|Dv_M[E]\|_\infty
\le\frac{\|E\|_\infty}{\gamma s^2}.
\tag{RBQ.130}
\]

In the scalar family \(M=\gamma(1-s)\),

\[
\frac{d v}{dM}=\frac1{\gamma s^2},
\qquad
\frac{d\log v}{dM}=\frac1{\gamma s}.
\tag{RBQ.131}
\]

Taking \(M=1-\Delta\) and
\(\gamma=1-c\Delta\), \(c\in(0,1)\), makes
\(\gamma s=(1-c)\Delta\). Relative Doob-vector learning and direct log-risk
learning therefore both require primitive operator error \(o(\Delta)\).
Theorem 27's
\(n\Delta^{(2\alpha+d_Q)/\alpha}\to\infty\) phase is invariant under the
coordinate change.

Equations (RBQ.126)--(RBQ.131) are exact identities/propositions. They rule out
“learned Doob” as a standalone statistical novelty. They do not rule out
optimization gains from a transformed parameterization or gains from a
strictly smaller learned risk quotient.

### Step 25. Predictable execution removes iid interface assumptions, not sample scarcity

Let \(Z_i\) be the executed interface known before fresh primitive witness
\(W_i\), and assume

\[
\mathbb E[W_i\mid\mathcal H_{i-1}]=\mu(Z_i).
\tag{RBQ.132}
\]

For fixed query \(z_t\), let \(\tau_{t,k}\) be the time of its \(k\)-th unique
transition inside radius \(h\). The stopped residuals

\[
\xi_{t,k}
=W_{\tau_{t,k}}-\mu(Z_{\tau_{t,k}})
\tag{RBQ.133}
\]

are martingale differences even when policy and safety execution depend on the
entire past. Martingale Hoeffding plus Hölder bias gives, simultaneously over
\(J\) queries and \(D\) coordinates,

\[
\max_t
\|\widehat\mu_{m,h}(z_t)-\mu(z_t)\|_\infty
\le
L_\mu h^\alpha+
B_W\sqrt{\frac{2\log(2JD/\delta)}m}
\tag{RBQ.134}
\]

with probability at least \(1-\delta\), provided every query reaches \(m\)
fresh hits. Substitution of (RBQ.134) into Step 22 preserves the joint phase.
If predictable visitation yields \(m\asymp nc_Qh^{d_Q}\), its balanced rate is
again \(\widetilde O(n^{-\alpha/(2\alpha+d_Q)})\).

A replayed transition is already measurable after its first observation.
Repeating it \(R\) times does not create \(R\) copies of (RBQ.133), so the
certificate count remains \(m\), not \(Rm\). Replay may change optimization,
but only new chronological environment transitions change statistical
coverage.

### Step 26. Stopped \(L_p\) localization replaces global uniform prediction

Opposite signs of \(S\) and \(S-E\) imply \(|S|\le|E|\). Therefore, for every
\(0<t\le r_0\), Q24 and Markov's inequality give

\[
\Pr\{\operatorname{sign}(S-E)\ne\operatorname{sign}(S)\}
\le C_0t^\kappa+\frac{R_p}{t^p}.
\tag{RBQ.135}
\]

Optimizing at

\[
t_D=\left(\frac{pR_p}{\kappa C_0}\right)^{1/(p+\kappa)}
\tag{RBQ.136}
\]

when \(t_D\le r_0\) yields the exact order

\[
\Pr(\text{first disagreement})
=O\!\left(
C_0^{p/(p+\kappa)}R_p^{\kappa/(p+\kappa)}
\right).
\tag{RBQ.137}
\]

The boundary-weighted loss admits the sharper split

\[
\mathbb E[|S|\mathbf1\{\text{disagreement}\}]
\le C_0t^{\kappa+1}+\frac{R_p}{t^{p-1}},
\tag{RBQ.138}
\]

and hence, at

\[
t_L=\left\{\frac{(p-1)R_p}{(\kappa+1)C_0}\right\}^{1/(p+\kappa)},
\tag{RBQ.139}
\]

has order

\[
O\!\left(
C_0^{(p-1)/(p+\kappa)}
R_p^{(\kappa+1)/(p+\kappa)}
\right).
\tag{RBQ.140}
\]

Neither power can be improved from Q24 alone. If
\(\Pr(|S|\le a)=a^\kappa\) on \([0,1]\) and
\(E=2S\mathbf1\{|S|\le a\}\), then the plug-in flips exactly on the boundary
band and

\[
R_p=\frac{2^p\kappa}{p+\kappa}a^{p+\kappa},\quad
\Pr(\text{disagreement})=a^\kappa,\quad
\mathcal L_\partial=\frac{\kappa}{\kappa+1}a^{\kappa+1}.
\tag{RBQ.141}
\]

Thus the theory-derived learning target is stopped-occupation \(L_p\) risk,
not global one-step MAE or uniform transformed-value error. This creates a
precise possible advantage over matched Doob/DICE/KROPE baselines: lower
integrated error where the irreversible decision is actually queried. It does
not yet prove that a learned quotient or neural estimator attains that
advantage. Oracle-shadow targets and occupation weights must be cross-fitted;
future trajectory summaries cannot be inserted into the deployed encoder.

### Step 27. Cross-fitted quotient regression exposes the correct overlap object

Let \(p_j\) and \(q_j\) be source and target stopped masses after both laws are
pushed through an independently learned chart and into cell \(A_j\). The local
coefficient relevant to an unweighted conditional-cell estimator is

\[
\mathfrak C_h
=\sum_{j:q_j>0}\frac{q_j}{p_j}.
\tag{RBQ.142}
\]

It is essential to push both laws forward before forming the ratio. Retaining
raw-atom ratios after aggregation would fail to remove nuisance variation and
would not describe the quotient experiment.

Under Q25, let the exact stopped score be within \(b_H\) of an
\(\alpha\)-Hölder chart function with constant \(L_H\), let the bounded
pseudooutcome have range \([-B_S,B_S]\), and let its stopped cellwise nuisance
remainder be \(\tau_Y\). The cell-mean estimator satisfies

\[
R_{2,\mathrm{stop}}
\le
3(2b_H+L_Hh^\alpha)^2+3\tau_Y^2
+\frac{12B_S^2\log(4J/\delta)}n\mathfrak C_h
\tag{RBQ.143}
\]

with probability at least \(1-\delta\), provided every target-active source
cell has \(np_j\gtrsim\log(J/\delta)\). No importance weight is required for
the conditional mean under chart-conditional invariance; the target stopping
law enters (RBQ.143) through \(q_j\) and evaluation risk.

If \(\mathfrak C_h\lesssim h^{-d_Q}\), balancing gives

\[
R_{2,\mathrm{stop}}
=O_{\mathbb P}\!\left[
\left(\frac{\log n}{n}\right)^{2\alpha/(2\alpha+d_Q)}
+b_H^2+\tau_Y^2
\right].
\tag{RBQ.144}
\]

When \(B_S,L_H=O(\Delta^{-1})\) and
\(b_H=O(\rho_H/\Delta)\), this becomes

\[
R_{2,\mathrm{stop}}
=O_{\mathbb P}\!\left[
\Delta^{-2}
\left\{
\left(\frac{\log n}{n}\right)^{2\alpha/(2\alpha+d_Q)}
+\rho_H^2
\right}+\tau_Y^2
\right].
\tag{RBQ.145}
\]

Theorem 31 then converts (RBQ.145) into disagreement power
\(\kappa/(\kappa+2)\) and boundary-loss power
\((\kappa+1)/(\kappa+2)\). In the discrete nuisance construction with \(K\)
raw nuisance values per sufficient quotient cell,

\[
\mathfrak C_{\rm raw}=KJ_Z,
\qquad
\mathfrak C_{\rm quotient}=J_Z.
\tag{RBQ.146}
\]

This proves an exact advantage over the declared raw partition. It does not
prove superiority to a Doob/DICE/KROPE method that learns the same chart.

### Step 28. Finite chart selection is learnable, but not method-exclusive

For raw interface \(u\), estimate its bounded witness vector \(\mu(u)\) from
\(m_u\) structure samples. Uniform Hoeffding concentration gives

\[
r_m(\delta)
=B_W\sqrt{\frac{2\log(2ND/\delta)}{m_{\min}}},
\qquad
\max_u\|\widehat\mu(u)-\mu(u)\|_\infty\le r_m.
\tag{RBQ.147}
\]

For every candidate partition \(k\), let \(\rho_k\) and
\(\widehat\rho_k\) be its maximum true and estimated within-cell witness
diameters. The pairwise reverse triangle inequality is uniform over all raw
pairs, so even data-dependent candidate partitions satisfy

\[
|\widehat\rho_k-\rho_k|\le2r_m.
\tag{RBQ.148}
\]

Let \(V_k\) be the independently certified nonrepresentation part of Theorem
32's stopped-risk bound and let \(A_\Delta\) amplify witness distortion to score
distortion. Selecting

\[
\widehat k\in\arg\min_k
\left[A_\Delta^2(\widehat\rho_k+2r_m)^2+V_k\right]
\tag{RBQ.149}
\]

gives the oracle inequality

\[
A_\Delta^2\rho_{\widehat k}^2+V_{\widehat k}
\le
\min_k\left[A_\Delta^2(\rho_k+4r_m)^2+V_k\right].
\tag{RBQ.150}
\]

If an exact quotient candidate is present, chart learning contributes at most
\(16A_\Delta^2r_m^2\). With \(A_\Delta=O(\Delta^{-1})\), its decision-scale
phase is

\[
\frac{m_{\min}\Delta^2}{\log(ND/\delta)}\to\infty.
\tag{RBQ.151}
\]

This resolves finite-library chart learning, but closes the hoped-for
unrestricted method separation. When the cemetery coordinate is retained, the
Doob map is invertible. A transformed algorithm with the same information can
run (RBQ.149), insert the Doob map and inverse around the same operator estimate,
and reproduce every manager decision. Hence

\[
\mathscr D_{\rm direct}=\mathscr D_{\rm Doob},
\tag{RBQ.152}
\]

and their minimax stranding--throughput decision risks are equal over every
common parameter family. A strict theorem now requires a declared restriction
on computation, hypothesis class, regularization, or side information; an
algorithm label is not a statistical restriction.

## Remarks and Interpretation

- The theorem-backed neural design primitive is not “add intervention norm.” It
  is preservation of the conditional witness vector
  \(\{\mathcal K_\lambda f:\lambda\in\Lambda,f\in\mathcal F\}\).
- A contrastive encoder loss is justified only if positive/negative pairs are
  generated according to (RBQ.1), not Euclidean proximity or policy/filter IDs.
- Cross-fitted risk occupation may weight the witness loss through (RBQ.11), while
  boundary emphasis must enter through (RBQ.17) or a differentiable surrogate.
- The R3 result is compatible with this structure: intervention strata mainly
  change conditional residual variance/tails and log-MGF, not a signed mean
  Energy penalty.
- Raw continuous-interface support can be infinite-dimensional while
  \(C_{\lambda,\phi}\) is finite. That possibility is standard restricted
  generalization; the project must prove that the chosen quotient preserves the
  charger-hitting decision and is strictly smaller in a nontrivial family.

## Boundaries and Non-Claims

- Exact quotient sufficiency is not claimed novel; it is a restricted conditional
  mean/operator factorization.
- A continuous local-mass Hölder point-query rate is proved in Step 22, but no
  distribution-free neural-encoder generalization rate is claimed.
- No claim is made that importance weighting is always necessary. In
  well-specified bounded covariate shift, unweighted regularized regression may
  already attain minimax rates.
- No boundary weighting can recover information lost by an encoder that collapses
  interfaces with positive (RBQ.1) distance.
- The stopped-margin functional does not certify collision safety or physical
  battery safety.
- Formal Oracle headroom and Pareto evidence remain unavailable behind the failed
  navigation prerequisite.

## Open Risks

- The factorization in (RBQ.4) needs an explicit measurable quotient lemma for the
  chosen encoder/range; continuous neural encoders are safer if the range is
  Polish and the conditional features are continuous on fibers.
- Bellman closure of a practical neural witness class may fail. Approximate
  closure must appear as a separate bias term.
- The risk occupation \(\bar\eta_{\nu,\lambda}\) depends on the unknown primitive,
  so estimating both it and the conditional witness operator can create a product
  nuisance rate.
- The score identity is a project-specific two-layer specialization of standard
  doubly robust/orthogonal off-policy estimation. It is not a safe standalone
  novelty claim until its closest theorem is audited; any publishable gain must
  come from the killed exponential first-passage, queryable execution layer, or
  a sharper structure-dependent rate and lower bound.
- The focused orthogonal-risk-OPE audit in
  `literature-search-20260830-two-layer-orthogonal-risk-ope/` finds direct
  collisions with DRL, DR risk-CDF estimation, MWL/MQL, restricted-chi-square
  OPE, and off-environment ratio factorization. Therefore neither (RBQ.23) nor
  (RBQ.36) is currently claimed as standalone novelty.
- Step 25 covers adaptive chronological trajectories through predictable
  conditional primitive innovations. Retrospective outcome-selected clustering
  and treating replay duplicates as fresh samples remain invalid without
  post-selection or new-data arguments.
- Step 26 proves a sharp deterministic handoff from stopped \(L_p\) error to
  mission decisions, but it does not supply the learned estimator's \(L_p\)
  rate. Reporting only training loss or replay-weighted minibatch error does not
  satisfy Q24.
- Step 27 makes the chart pushforward order explicit and supplies a conditional
  cross-fitted estimator rate. The chart distortion \(b_H\) is an input
  certificate, not a consequence of using a neural encoder.
- Step 28 closes the finite-library chart-rate gap but also rules out an
  unrestricted matched-information superiority theorem. Continuous neural
  generalization and computationally restricted separations remain open.
- If the full statistical theorem is exactly Duan-style restricted OPE on the
  known Doob chain, the honest output is an equivalence/negative theorem rather
  than an oral-level new learning theorem.
- ICML 2025 KROPE already proves stability and Bellman-completeness properties
  for bisimulation-based OPE representations. Step 22 therefore cannot be
  positioned as generic representation stability; only the joint
  quotient--critical-mode--stopped-boundary phase remains a candidate.
