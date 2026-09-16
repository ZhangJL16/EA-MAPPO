# Derivation Package: Attainability for Resource-Coupled Path Catalogues

## Target

Prove a matching upper for a natural, controlled subclass of the existing general
lower theorem. Resolve unknown-model attainability, including finite alternatives,
shared primitive feedback and unequal path durations. Audit OSSB reduction rather
than presenting any apparent mismatch as certified novelty.

## Status

COHERENT AFTER REFRAMING / EXTRA ASSUMPTION. Complete handwritten theorem draft
for a FINITE known hypothesis class with common finite reward supports and unique
optimal path types. A single horizon-free safe learner attains C_B(theta) at
every fixed hypothesis. No independent/formal proof audit or novelty certification.
The continuous-parameter five-state upper remains a separate specialized result.

## Invariant Object

The same C_B(theta) in GENERAL_RESOURCE_COUPLED_INFORMATION_LOWER_BOUND_20260916.md,
with calendar-time centered path costs, not per-excursion reward-rate gaps.
Capacity is fixed in the T-to-infinity limit. Freeze the base class before B.

## Assumptions

Use the previous lower theorem's finite known deterministic dynamics/cost,
single recharge, decreasing and safe reachable domain. Add ONLY:

1. Theta is a finite known collection of reward hypotheses. Each accessible
   primitive row i has a finite support Y_i contained in [0,1], common to all
   hypotheses, and probability v_{theta,i}(y)>0 for every y in that support.
   Known deterministic rows have a singleton support. Bernoulli laws indexed
   by a finite collection of interior parameters are included. This is not a
   theorem for every continuous exponential-family parameter class.
2. Conditional reward laws are stationary given the chosen row, as in the lower
   theorem. Equivalently, use independent IID streams for primitive rows,
   conditional on theta; equal laws/shared parameters across rows are permitted.
3. Quotient Theta by equality of all accessible row laws. This quotient is known
   at fixed B and does not discard any different optimal decisions: equal laws
   give equal expected reward for every safe path. Representatives are denoted
   again by Theta, with size K. At K=1 the known optimal path solves the problem.
4. Each representative theta has a unique optimal path TYPE p_theta. Paths with
   identical primitive-row count vectors may be merged by choosing one fixed
   ordering; their lengths, means and feedback information agree. If distinct
   types tie for optimal gain, this upper theorem does not apply. The earlier
   lower theorem still allows ties.

Do not add unknown support, stochastic transitions, partial observability or
changing catalogues during a mission. Safety is known and never learned here.

## Notation

P=P_B is the finite catalogue of complete legal charger-to-charger paths after
removing the duplicate types just specified; M=|P|, length ell_p<=L_B. All paths
are feasible at a full charger. Each path is committed before departure and
executed without reacting to the current excursion's rewards. The learner adapts
between excursions using ALL primitive observations, including certified ones.

For theta define rho_theta, path cost c_theta(p), row counts m_p and information
j_{theta,theta'}(p) exactly as in the lower theorem. Because the optimum is
unique, confusing alternatives are

\[
\mathcal A_\theta=\{\theta':p_{\theta'}\ne p_\theta,
                         j_{\theta,\theta'}(p_\theta)=0\}.
\]

Fix one deterministic optimizer lambda_theta of

\[
\min_{\lambda_p\ge0,p\ne p_\theta}\sum_p\lambda_pc_\theta(p)
\quad\text{s.t.}\quad
\sum_{p\ne p_\theta}\lambda_pj_{\theta,\theta'}(p)\ge1
\ (\theta'\in\mathcal A_\theta).
\tag{1}
\]

It exists and has objective C_B(theta): each confusing alternative differs on
some accessible row outside the optimal path, so some nonoptimal path supplies
positive KL. A sufficiently large allocation to all nonoptimal paths is feasible.
Positive nonoptimal costs make bounded objective level sets compact. With no
confusing alternatives, choose lambda=0. Allocation uniqueness is NOT assumed.

Different-optimum alternatives outside A_theta are distinguishable by playing
the optimal path for free at truth. Set

\[
D_\theta=\min_{\substack{p_{\theta'}\ne p_\theta\\
                         j_{\theta,\theta'}(p_\theta)>0}}
                         j_{\theta,\theta'}(p_\theta),
\qquad \kappa_\theta=2/D_\theta.
\]

If that set is empty put kappa_theta=0. This explicitly handles FREE information;
one must not require costly exploration against every different parameter.

Let L_t(theta) be the log likelihood of all rewards observed before primitive
time t, evaluated at a charger. MLE hat_theta_t uses fixed tie breaking. Define

\[
f(t)=\log(t+3)+2\log\log(t+3)+\log(K+1),
\quad \mathcal H_t=\{\theta':L_t(\hat\theta_t)-L_t(\theta')\le f(t)\}.
\tag{2}
\]

Let Q_p count EXPLORATION excursions of type p; certified excursions are not
counted in Q but their observations remain in L_t. Let j=sum_p Q_p. Set
eta_j=(j+1)^(-1/8). This buffer tends to zero; its choice is justified below,
not a guessed confidence calibration.

## Derivation Strategy

Finite-hypothesis likelihood certificate prevents costly exploitation mistakes.
Sublogarithmic forced sampling stabilizes the hypothesis at exploration indices.
Then track the exact resource-coupled optimum, with a separate zero-cost path
quota to eliminate nonconfusing alternatives. Count actual calendar-time cost.

## Derivation Map

1. Likelihood ratios give a summable false-certificate bound without an iid-cycle
   or random-horizon concentration shortcut.
2. Forced path counts give uniform-in-sample-count row likelihood concentration.
3. Correct-model allocation quotas imply a correct decision certificate.
4. An exploration-index counting argument bounds wrong allocations and overhead.
5. The existing lower bound applies to this uniformly efficient learner, giving
   equality; all within-excursion adaptive learners share that lower bound.

## Main Derivation

### Step 1 — theoretical learner

Initialize by executing every path once as exploration, giving Q_p=1. Then at
each return to the full charger at time t:

1. Compute hat_theta_t and H_t. If all hypotheses in H_t have the SAME optimal
   path type, execute that path as CERTIFIED; leave Q unchanged.
2. Otherwise enter EXPLORATION, using the current j and eta_j:
   - If min_p Q_p < sqrt(j), execute a least-counted path (forced sampling).
   - Else, if any p != p_hat has
     Q_p < (1+eta_j)lambda_hat,p f(t), execute one such deficient path.
   - Else, if Q_{p_hat} < kappa_hat f(t), execute p_hat (free-information quota).
   - Else execute a least-counted path (fallback).
   Increment Q for the executed exploration path, hence increment j.

All tie breaking is fixed. Each path is feasible; there is no oracle theta input,
no horizon T input, no latent state query, and no unsafe probing. Finite models
and their LP solutions are known offline; the selected model remains unknown.

### Step 2 — proposition: false certified exploitation has finite expected cost

Under true theta, for every theta' the transcript likelihood ratio
exp(L_t(theta')-L_t(theta)) is a mean-one nonnegative martingale at deterministic
primitive t. Common positive supports justify this assertion. The same learner
is used in both models; its action probabilities cancel. At any integer t,

\[
P_\theta(\theta\notin\mathcal H_t)
\le(K-1)e^{-f(t)}
\le\frac1{(t+3)\log^2(t+3)}.
\tag{3}
\]

This is a fixed-time likelihood-ratio union bound, not a confidence formula
borrowed from a scalar empirical-mean theorem. Sum (3) over all primitive t;
the actual charger decision times are a subset. If truth belongs to H_t,
a certificate can only select p_theta. Each wrongly certified path costs at
most max_p c_theta(p), a fixed constant. Thus wrongly certified paths have
finite expected count and regret. Whole-world clock includes every executed step.

### Step 3 — proposition: model and information estimates are reliable at query indices

The least-counted forced rule implies

\[
\min_p Q_p\ge\sqrt j-M-2
\tag{4}
\]

up to harmless initialization rounding. To see this, consider the last nonforced
exploration index r before j. Its counts were all at least sqrt(r). Subsequent
forced plays increase the minimum at least once per M plays. Use
sqrt(j)-sqrt(r)<=sqrt(j-r) and
sqrt(n)-n/M<=M/4. If there has been no nonforced play, apply the same counting
from the all-path initialization. Certificates do not change j or these counts.
Every accessible primitive row occurs in some path, hence has at least
g(j)=max(1,floor(sqrt(j)-M-2)) samples.

For a row i with unequal laws under theta,theta', the IID log-likelihood-ratio
increments Z=log(v_{theta,i}(Y)/v_{theta',i}(Y)) have positive mean I_i and are
bounded, because both the row support and Theta are finite. Define E_j as the
event that for ALL such row/model pairs and EVERY sample count n>=g(j),

\[
\left|\frac1n\sum_{k=1}^nZ_{i,k}-I_i\right|
\le\frac{\eta_j}{4}I_i.
\tag{5}
\]

Equal-law rows have identically zero increments. Hoeffding's inequality and a
geometric sum over n, followed by the finite row/model union, give, for large j,

\[
P_\theta(E_j^c)\le A(j+1)^{1/4}\exp(-a(j+1)^{1/4}),
\qquad \sum_jP_\theta(E_j^c)<\infty.
\tag{6}
\]

Here A,a>0 are instance constants. Indeed eta_j^2 g(j) is order j^(1/4), and the
geometric-sum prefactor is order eta_j^(-2)=j^(1/4). Finite early indices add a
finite constant. This event is uniform over ALL row sample counts above g(j);
it is valid at the random calendar time of query j. No optional-time substitution
into a fixed-n bound is used. Query indexing ensures the same j is charged only
once, irrespective of the possibly long certified period preceding it.

On E_j, the true model has greater likelihood than every different observable
representative; thus hat_theta_t=theta. In addition,

\[
L_t(\theta)-L_t(\theta')
\ge(1-\eta_j/4)\sum_iN_i(t)I_i(\theta,\theta').
\tag{7}
\]

### Step 4 — proposition: completed quotas imply a decision certificate

Suppose E_j holds and neither informative quota nor free-information quota is
deficient. Primitive observations include those in Q, so for every confusing
alternative,

\[
\sum_i N_i(t)I_i(\theta,\theta')
\ge\sum_{p\ne p_\theta}Q_pj_{\theta,\theta'}(p)
\ge(1+\eta_j)f(t).
\]

By (7), its log likelihood is more than f(t) below truth, since
(1-eta/4)(1+eta)>1 for 0<eta<=1. For every nonconfusing alternative with a
different optimum, the free quota supplies at least 2f(t) KL, again more than
f(t) in log likelihood by (7). All different-optimum models are absent from H_t.
Therefore H_t certifies p_theta. The fallback branch, when reached, must have
E_j fail. A wrong-model exploration also requires E_j fail.

### Step 5 — counting proof: exploration overhead is sublogarithmic

Let J_T be the total exploration count by time T, including initialization.
Let B_T count exploration indices with E_j failing. By (6),
E_theta B_T <= sum_j P_theta(E_j^c) < infinity, uniformly in T.

Forced plays of any one path increase its Q only while Q_p<sqrt(j), so their
number is at most sqrt(J_T)+1. With a correct model, each informative-quota
play of p increases Q_p only below 2lambda_theta,p f(T), and free-quota plays
increase Q_opt only below kappa_theta f(T). Counts from other branches cannot
invalidate this upper counting argument; they only fill a quota sooner.
Consequently, pathwise with instance constants A_0,A_1,

\[
J_T\le A_0+A_1f(T)+M\sqrt{J_T}+B_T.
\tag{8}
\]

Use M sqrt(J)<=J/2+M^2/2 to obtain E J_T=O(log T). Jensen's inequality now gives
expected forced plays O(sqrt(log T)); their bounded path costs are o(log T).
Incorrect-model and fallback costs are O(1) in expectation by (6).

For the EXACT leading constant, fix epsilon>0 and then a deterministic j_0
after which eta_j<=epsilon. Before j_0 there are only finitely many query plays.
After j_0, each correct informative play increases Q_p only below
(1+epsilon)lambda_theta,p f(T). Its total expected centered cost is at most

\[
(1+\epsilon)C_B(\theta)f(T)+O(1).
\tag{9}
\]

Correct free-quota and correctly certified paths cost zero at truth. Add forced,
bad-query and false-certificate costs, plus one bounded incomplete-path/bias
remainder from the lower document. Divide by log T, use f(T)/log T -> 1, and let
epsilon decrease to zero. This proves the upper below. The proof works also
when C_B(theta)=0: the remaining regret is o(log T).

### Main theorem — exact attainability

For this finite-hypothesis regenerative path-catalogue class, the learner above
is safe on its single continuing trajectory and, for EVERY theta, satisfies

\[
\boxed{\displaystyle
\lim_{T\to\infty}\frac{\operatorname{Reg}_\theta(T)}{\log T}
=C_B(\theta).}
\tag{10}
\]

Proof. Steps 2--5 prove limsup<=C_B(theta) and O(log T) regret at every hypothesis.
The same learner is therefore uniformly efficient on the whole class. Apply
the existing general lower theorem to it for liminf>=C_B(theta). QED.

### Corollary — within-excursion adaptivity does not improve the leading coefficient

The lower theorem covers arbitrary safe within-excursion adaptive learners;
the upper learner commits to a complete safe path at each departure. Thus both
classes have the SAME optimal instance-dependent leading coefficient C_B(theta)
under these finite-hypothesis assumptions. Feedback can affect finite-time
behavior, but need not be acted on before recharge to attain that coefficient.

This is an exact consequence, not a proof that all adaptive transcript laws
are equivalent to committed paths. It is also not a general minimax statement.

## Remarks and Interpretation

The allocation is genuinely coupled if several observations share travel and
resource costs; no armwise diagonal assumption is needed. Different hypotheses
may prescribe completely different allocation optimizers: one learner attains
each through likelihood selection, rather than selecting model-dependent
occupancies independently with an oracle. Finite Theta removes the need for
allocation continuity, not the need to control wrong exploitation.

Do not turn (10) into the previous continuous Bernoulli example's general upper:
that example already has its own upper proof. A finite subset of its class
is covered here, but its confusing set and coefficient may differ. The general
lower continues to cover both finite and continuous classes.

## OSSB Reduction Audit

The inspected 2017 paper explicitly extends its framework to vector independent
semi-bandit feedback in Section 3. Shared observations or vector feedback are
NOT a defensible reason for saying structured bandits cannot cover this setting.
Its displayed regret is per decision round, and Theorem 2 lists Bernoulli/Gaussian
observation conditions and a unique allocation optimizer. Our proof instead
uses finite likelihood hypotheses, permits nonunique LP optimizers, and tracks
calendar-time path cost c=rho ell-R. Hence its displayed upper is not literally
the theorem just proved for unequal path durations.

However, this is not a fundamental separation from structured learning. A
committed safe path is a finite structured experiment with feedback law
product_i nu_{theta,i}^{m_p(i)}, duration ell_p and expected total reward R_theta(p).
The GL-B program is exactly its cost-weighted structured allocation. When all
ell_p equal ell, c=ell(rho-R/ell), and the per-round allocation program becomes
the calendar-time program by this common scaling. When durations differ, this
same simple scaling no longer preserves the objective; weighted allocation is
needed. Unequal durations by themselves are not certified new learning theory.

The 1997 Graves--Lai paper is additional direct prior art: its publisher abstract
already describes asymptotically efficient learning of long-run average reward
over finite recurrent control-law sets. Its full assumptions have not been
audited here, so do NOT claim that the displayed OSSB mismatch establishes a
gap in the broader prior literature.

Verified primary sources:

- Combes, Magureanu and Proutiere (2017), Sections 3--6, especially Theorem 2:
  https://proceedings.neurips.cc/paper/2017/file/e19347e1c3ca0c0b97de5fb3b690855a-Paper.pdf
- Graves and Lai (1997), publisher abstract, DOI 10.1137/S0363012994275440:
  https://epubs.siam.org/doi/10.1137/S0363012994275440

## Boundaries and Non-Claims

- Attainability is now a class theorem, not solely five states or one symmetric
  true mean. It is instance-dependent asymptotic optimality, NOT gap-uniform
  minimax logarithmic regret or growing-B uniform asymptotics.
- The learner is a mathematically specified feasible procedure. No code or
  empirical claim is implied. Catalogue enumeration and LP computation may be
  expensive; no polynomial-in-log-B planning guarantee is given.
- Safety is precompiled. Information does not change physical safety in this
  model. Catalogues are fixed at each capacity, not nonstationary within a run.
- A maximum path length depending on B affects constants and remainders, even
  though the theorem identifies the fixed-B leading coefficient exactly.
- Lower + class-wide matching upper closes the attainability question. It does
  not by itself close the publication novelty question. The adaptivity corollary
  actually rules out within-cycle feedback as an automatic novelty escape.

## Open Risks

Independent proof audit remains required, especially exploration-index counting
and the uniform row-count likelihood event. There is no Lean probability proof.
Continuous-class allocation stability and tie handling are excluded rather than
asserted. More importantly, cost-weighted structured bandit/controlled-Markov
prior art may already subsume the result; do not label (10) a new principle or
Spotlight-ready on the strength of missing literal OSSB assumptions alone.
