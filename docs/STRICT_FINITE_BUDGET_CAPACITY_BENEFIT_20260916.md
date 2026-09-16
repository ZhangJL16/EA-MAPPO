# Derivation Package: Strict Finite-Budget Capacity Benefit

## Target

Resolve strictness on the EXISTING sealed calibration class, without new
hypotheses, route descriptors, trajectory samples, or changes to the frozen
learner. First check the requested F-based minimax fixed point. Then, if that
relaxation cannot separate the bounds, use exact finite-decision-game
certificates on the same class rather than construct another example.

The target is a finite-budget minimax inequality, not a positive observed
ordering of the frozen learner at T=4096.

## Status

**COHERENT AFTER REFRAMING. Strict finite-budget benefit is established by a
handwritten proof with exact rational computational certificates on this class.**

The requested fixed-point lower bound is valid, but comparison with the
previous frozen-learner upper envelope cannot separate at ANY T>8 here.
This is proved below, not inferred from failure of a numerical search.
Replacing that loose comparison with a finite Bayesian Bellman lower certificate
and an explicit safe policy mixture upper certificate yields

\[
\mathfrak R_{3,\mathrm{on}}(T)-\mathfrak R_{4,\mathrm{on}}(T)>1/20
\quad(T\in\{12,13,\ldots,24\}).
\]

The primitive-action low-capacity calculation allows within-excursion feedback
and all legal terminal states. The high-capacity witness is committed between
charger visits and is required to finish at the charger. Thus the strict bound
holds both for committed learners and for the larger physical safe-policy class.

This is not a Lean proof or an independent researcher audit. It does not establish
paper novelty, an efficient general minimax solver, or an empirical positive
result. No learner trajectory has been generated or resumed.

## Invariant Object

Keep expected calendar-time regret and the SAME base hypothesis class:

\[
\mathfrak R_{B,u}(T)=\inf_A\max_{\theta\in\Theta}
\left\{T\rho_\theta-\mathbb E_\theta^A\sum_{t<T}Y_t\right\}.
\]

u is the bundling permission; A is one safe unknown-model policy, not one policy
chosen separately for each true theta. Policies can be randomized. The infimum
is taken separately at each fixed T, so a horizon-specific theoretical witness
is allowed. No claim is made that one horizon-free policy attains all the finite
bounds simultaneously. At each certified T the witness ends at the charger and
can thereafter start the previous asymptotically efficient learner from scratch,
if asymptotic efficiency is additionally required.

## Assumptions

The public hypotheses, in certificate order, are exactly

\[
\theta_q=(1/20,1/20),\quad
\theta_A=(3/10,1/20),\quad
\theta_B=(3/10,19/20).
\]

Dock reward is Bernoulli with common known mean 1/20. Inspection rows A and B
are independent Bernoulli streams conditional on theta. All rewards are observed,
and all dynamics, costs, routes and reload rules are known. The true index is
unknown. The class itself is fixed before capacity or horizon is selected.

Each primitive action costs one unit of clock and resource. Debit before reload;
arrival at the charger replenishes to full capacity. The complete physical paths
are dock (length 1), empty (2), A (3), B (3), and additionally AB/BA (4) only at
capacity 4 with bundling on. A/B first inspection occurs on step 2; a second
inspection occurs on step 3. The prep/undock step is not a reload. Each site may
be inspected only once in a sortie. Bundling off requires return after the first
inspection. This is the existing benchmark, not altered timing or physics.

Every hypothesis has the same optimal path and gain in all four cells:
dock/1/20 at theta_q, A/1/10 at theta_A, B/19/60 at theta_B. Start at the charger.

## Notation

- c_theta(p)=rho_theta ell_p-R_theta(p): complete-path centered cost.
- R_B,theta^A(T): expected calendar regret of one learner A.
- F_B,theta(T;R): the previous acquisition LP with common envelope r_v(T)=R.
- Psi_B(T,R)=max_theta[F_B,theta(T;R)-L_B], with extended values permitted.
- pi: a PUBLIC prior on Theta; it is a design constant, not private truth.
- V_B(T,pi): maximal Bayesian expected finite-horizon reward under that prior.
- beta_B(T,pi)=T sum_theta pi_theta rho_theta-V_B(T,pi).
- w: unnormalized likelihood masses over hypotheses after an observation history.
- q(s,a,y): the appropriate public hypothesis-indexed outcome law, not a simulator
  query accessible to the agent. Mathematical verification enumerates its outcomes.

## Derivation Strategy

Necessary envelope fixed point -> prove looseness of the requested comparison ->
finite decision-game dual -> exact low-capacity Bellman certificate -> safe
high-capacity policy mixture -> strict finite-budget interval -> controls excluding
terminal exploitation and feedback-free execution as explanations.

## Derivation Map

1. Any class-wide envelope R must satisfy R>=Psi_B(T,R).
2. On this class, the largest instance-dependent coefficient is unchanged by B.
   Explicit feasible allocations prevent the old F-lower / wrapper-upper comparison
   from ever separating.
3. The finite policy game has a Bayesian dual, retaining the single-learner
   constraint that the theta-wise allocation relaxation drops.
4. Exact Bellman values and exact policy risk vectors provide a strict certificate.
5. Repeat the certificate check for each integer T in [12,24], without samples.
6. Verify capacity-only equivalence and feedback-free/terminal controls.

## Main Derivation

### 1. The requested minimax fixed point [proposition]

For T>2L_B define

\[
\underline R_B(T)=\inf\{R\ge0:R\ge\Psi_B(T,R)\}.
\]

At theta_q, every primitive reward mean is at most rho_q=1/20, so every safe
policy has R_q(T)>=0. Thus every learner's worst-case regret R is nonnegative.
Its own R is a valid common envelope on all hypotheses. The previous theorem
then gives R>=Psi_B(T,R), hence R>=underline R_B(T). Taking the learner infimum
proves

\[
\mathfrak R_B(T)\ge\underline R_B(T).
\]

The error bounds e_theta and q_v increase with R. In their informative range,
(1-e_theta)log(1/q_v) decreases; the clipped KL requirements therefore decrease.
The F constraints relax as R increases, so Psi is nonincreasing. Its feasible-R
set is upward closed. There is no need to assert equality at a fixed point or
silently assume a differentiable/unique root. This statement is scoped to the
previous committed-path F theorem; the later primitive-game certificate is wider.

At T=12 the original e_theta(T;0) is already >=1 for every hypothesis in both
cells, so all its information requirements are zero and underline R_B(12)=0.
Floating LP evaluation also gives zero at 4096; this latter numerical fact is
not used as a proof of strictness or of universal looseness.

### 2. Why F versus the existing upper can NEVER separate here [proposition]

Set

\[
a=\mathrm{kl}(1/20,3/10),\quad
d=\mathrm{kl}(3/10,1/20),\quad
b=\mathrm{kl}(1/20,19/20)=(9/10)\log19.
\]

Directly solving the existing allocation constraints gives

| hypothesis | C_3 | C_4 |
| --- | ---: | ---: |
| theta_q | (1/10)/a = 0.498691947 | (1/10)/a = 0.498691947 |
| theta_A | (1/4)/b = 0.094339798 | (1/20)/b = 0.018867960 |
| theta_B | 0 | 0 |

At theta_q, theta_A can only be distinguished by an A measurement. Both A-only
and AB cost 1/10 per A measurement there. The constraint against theta_B is
automatically met by the same A queries. At theta_B, optimal B paths supply free
information against both alternatives. Consequently the maximal coefficient
C_q=(1/10)/a is unchanged: the primary-truth coefficient is not a minimax coefficient.

For the low-capacity F, every required a_v(T;R) is at most log T: in the informative
case k_v<log T since delta_v<=1 and 2(R+3)>=6, and partial-KL credit only reduces it.

At theta_q, either e_q>=1 and F_q=0, or e_q<1 implies T>186. Choose
n_A=log T/a, all other counts zero. This meets both alternative constraints and
has cost C_q log T. Its duration 3 log T/a is at most T for T>=186.

At theta_A, either e_A>=1 and F_A=0, or T>126. Choose
n_A=log T/d, n_B=log T/b. These cover the free alternative theta_q and confusing
alternative theta_B respectively, at cost C_A,3 log T. Their total duration is
at most T for T>=126.

At theta_B, n_B=log T/b covers both alternatives at zero centered cost and fits
the duration budget for every T>6.

For explicit verification of these time inequalities: on [1/20,3/10], the
Bernoulli-KL curvature is at least 100/21, so a,d>=25/168 by integrating twice
from the minimum. Globally curvature>=4 gives b>=81/50. Then at T=186 the first
duration is <(504/25)*6<186; at T=126 the second is
<3*(168/25+50/81)*5<126. Use log186<6, log126<5, and the decreasing log T/T
thereafter. The zero-cost B allocation uses 3/b<=50/27 and fits for T>6.
Also a<19/56 by log(1+x)<=x, so C_q>28/95, whereas C_A,3<=25/162<C_q.

Thus, for EVERY R>=0 and T>6,

\[
\Psi_3(T,R)\le\max\{0,C_q\log T-3\},\qquad
\underline R_3(T)\le\max\{0,C_q\log T-3\}.
\]

The previous high-capacity worst-case upper expression includes its theta_q
bound C_q f(T)+nonnegative terms+L_4, with L_4=4 and f(T)>log T for T>8.
Therefore that PARTICULAR lower/upper comparison cannot separate at any T>8.
This is a limitation of the proposed certificate route, not absence of strict
capacity benefit. No additional expansion of its wrapper overhead is needed.

### 3. A finite decision-game characterization [proposition; general finite class]

For a known finite safe physical graph, finite hypothesis class and finite common
outcome supports, a fixed integer horizon has finitely many deterministic
history-contingent safe policies. Any randomized safe policy is a distribution
over these policies: preassign its random choices at each possible finite history.
Conversely, sampling one deterministic contingent policy is implementable without
knowing theta. The finite zero-sum game's linear-program dual gives

\[
\mathfrak R_B(T)=\max_{\pi\in\Delta(\Theta)}
\left[T\sum_\theta\pi_\theta\rho_\theta-V_B(T,\pi)\right].
\tag{Finite-game dual}
\]

This is a standard finite-game duality argument, not a claimed novel duality.
It is an EXACT single-learner characterization, unlike independently feasible
theta-wise acquisition allocations. It may be computationally expensive.

The Bayesian value is obtained by a physical-state Bellman recursion. With t
steps left and unnormalized likelihood masses w, set U_0(x,w)=0 and

\[
U_t(x,w)=\max_{a\in A_{\rm safe}(x)}
\left\{\sum_\theta w_\theta r_\theta(x,a)
+\sum_y U_{t-1}\left(F_B(x,a),(w_\theta\nu_{\theta,x,a}(y))_\theta\right)\right\}.
\]

It follows by conditioning on the next observed outcome. Induction gives its
positive homogeneity, so known common-law dock feedback can be integrated out
without losing Bayesian value. No normalization or hypothesis-index oracle is
needed. For rational primitive probabilities and gains the entire recursion is
rational. This includes within-excursion adaptive choices.

A sufficient strictness certificate is any low-capacity prior pi and ANY
implementable high-capacity mixture gamma satisfying

\[
\beta_{B_1}(T,\pi)>
\max_\theta\sum_k\gamma_k R^{A_k}_{B_2,\theta}(T).
\]

The LHS lower-bounds the low minimax value and the RHS upper-bounds the high
minimax value. This avoids assuming that independent allocations are jointly
attainable by one unknown-model learner.

### 4. An explicit T=12 strict certificate on the unchanged class [proposition]

Use the public low-capacity prior

\[
\pi=(1/2,9/20,1/20).
\]

Exact Bayesian Bellman maximization gives V_3(12,pi)=6067/8000. Therefore

\[
\mathfrak R_3(12)\ge 12\sum_\theta\pi_\theta\rho_\theta-6067/8000
=103/100-6067/8000=2173/8000=0.271625.
\]

The route-level exact recurrence evaluates 238 states. A separately implemented
primitive-action recurrence, with no path commitment, returns exactly the SAME
rational value. It allows all legal terminal positions. The selected low Bayesian
policy is particularly simple: dock once, execute A twice; if both observations
were zero use three dock services, otherwise execute A once more; finally take
a two-step partial A path. This policy's expected reward at mean m_A is

\[
1/20+4m_A+(3/20-m_A)(1-m_A)^2.
\]

Its three risk values are 1039/4000, 47/2000, 5247/2000. Direct substitution
checks these values independently of the Bellman code. Global Bayesian optimality,
which is essential for a lower bound, comes from the full Bellman maximization,
not merely evaluation of this selected policy.

For the high-capacity upper witness, define A_k as the Bayesian-optimal committed
charger-terminal policy at T=12 under these PUBLIC priors:

\[
\pi_1=(1/3,1/3,1/3),\quad
\pi_2=(71/125,421/1000,11/1000),\quad
\pi_3=(567/1000,421/1000,3/250).
\]

At a charger, maximize expected remaining reward over dock/A/B/AB, update the
finite-hypothesis posterior using actual past rewards, and forbid any departure
whose complete duration exceeds the remaining time. Thus every branch is safe
and ends at the charger. This defines the witness without private truth or C(theta).
One explicit initial randomization selects these policies with probabilities
(12/25,1/10,21/50).

Their exactly evaluated risk vectors, in theta_q/theta_A/theta_B order, are

| policy | theta_q | theta_A | theta_B |
| --- | ---: | ---: | ---: |
| A_1 | 349679/1600000 | 173067/800000 | 41029/800000 |
| A_2 | 17541/80000 | 6993/40000 | 58829/40000 |
| A_3 | 335961/1600000 | 178653/800000 | 80131/800000 |

The mixture risks are exactly

\[
\left(17201577/80000000,\ 8604621/40000000,\ 8550347/40000000\right).
\]

Hence

\[
\boxed{\mathfrak R_4(12)\le 8604621/40000000=0.215115525}
\]

and

\[
\boxed{\mathfrak R_3(12)-\mathfrak R_4(12)
\ge2260379/40000000=0.056509475>0.}
\]

The bounds are not assertions that either minimax value equals the displayed
certificate endpoint. Floating minimax optimization is unnecessary to verify
this T=12 certificate: all its priors and mixing weights are specified above.

### 5. A nonempty finite-budget interval [exact computational corollary]

The certificate file provides rational low priors, high policy priors, mixing
weights, per-hypothesis risks and gaps at every integer T in [12,24]. Each bound
is recomputed with exact fractions; the primitive low recurrence independently
agrees at every T. Every EXACT rational gap exceeds 1/20:

| T | low minimax lower | high minimax upper | guaranteed gap, rounded |
| ---: | ---: | ---: | ---: |
| 12 | .271625000 | .215115525 | .056509475 |
| 13 | .316473903 | .216586796 | .099887107 |
| 14 | .292932036 | .233517339 | .059414697 |
| 15 | .323881220 | .256225313 | .067655907 |
| 16 | .356650610 | .249722151 | .106928459 |
| 17 | .336245087 | .271430096 | .064814991 |
| 18 | .363540135 | .292370236 | .071169899 |
| 19 | .397147282 | .285371903 | .111775379 |
| 20 | .373409176 | .305483626 | .067925550 |
| 21 | .402596318 | .324933664 | .077662654 |
| 22 | .434741573 | .318017138 | .116724435 |
| 23 | .411256789 | .337008745 | .074248044 |
| 24 | .439415302 | .355021275 | .084394027 |

The inequalities use the saved fractions, not the rounded table. Floating
column generation was used to DISCOVER witnesses for T=13..24, not certify them.
The exact-only receipt-verification command never calls floating optimization.
This interval was mathematically searched; it is NOT a prospectively registered
or sampled empirical success interval. It must not be presented as one.

### 6. Execution, terminal and negative controls [propositions]

**Terminal control.** Every high witness ends at the charger, whereas the low
lower bound allows any legal terminal state. Thus favorable terminal non-return
cannot be the source of the certified high advantage. At T=12 and T=24, with a
charger-terminal restriction, the known-model optimum equals T rho_theta in
BOTH capacities at EVERY hypothesis: complete paths have nonnegative centered
cost and repetitions of the dock/A/B optimum fit exactly. This is equality of
finite-horizon execution performance, stronger than just equal asymptotic gain.

**Feedback-free control at T=12.** Consider high-capacity policies that ignore
inspection feedback and finish at the charger. Under public prior
(21/23,0,2/23), every complete path has weighted centered cost at least
(8/345) ell_p. Dock and AB achieve equality; A/B/empty have larger ratios.
Therefore every such feedback-free mixture has worst-case risk at least
12*(8/345)=32/115=.278260870. A mixture with weight 5/46 on one AB plus eight
dock steps, and 41/46 on three AB paths, achieves risk vector
(32/115,21/115,32/115). Thus this feedback-free value is EXACTLY 32/115,
strictly above the adaptive high witness's upper .215115525. Simply providing
a robust static bundled action does not account for that witness's performance.

**Capacity-only control.** With bundling off, both capacities have the identical
physical feedback experiment catalogue: after the first inspection only return
is allowed. At B=3, on and off are also identical because no second inspection
is resource-feasible. Virtual-counter simulation works in BOTH directions for
these catalogues, yielding, for every finite T,

\[
\mathfrak R_{4,\mathrm{off}}(T)=\mathfrak R_{3,\mathrm{off}}(T)
=\mathfrak R_{3,\mathrm{on}}(T).
\]

Thus the certified benefit requires the newly feasible shared-measurement
sortie, not capacity increase alone. Neither new reward laws nor new parameters
are introduced by that permission.

## Remarks and Interpretation

The strictness question has a constructive positive answer on the existing
class, including a finite interval: this is no longer an asymptotic-existence
argument or a wrapper-specific upper-bound story. The valid F relaxation is
too loose to certify this result; its failure should not be interpreted as
failure of the statistical mechanism.

There is also a conceptual correction: this class's MAXIMAL asymptotic C is
unchanged, even though the primary truth's C drops by 80%. Conditional on the
existing asymptotic upper theorem, the minimax leading coefficient at either
capacity is C_q. For completeness, the finite-class upper gives limsup<=C_q;
the F fixed point with the attained worst-case R=O(log T), theta_q versus theta_A,
gives required KL=log T-O(log log T), so liminf>=C_q. The finite strict result
therefore concerns acquisition/decision risk not captured by that maximal
leading coefficient alone. It does not prove a uniform all-horizon benefit.

The Bayesian dual is standard; the novel-status question concerns the resource
mechanism and its general significance, not inventing another duality or LP.
This package does not settle novelty against finite-budget structured experiments.

## Boundaries and Non-Claims

- No positive ordering for the frozen T=4096 primary learner is claimed.
- No inference that the high witness attains its bound at each individual truth
  relative to every low learner; the comparison is minimax, not pointwise.
- No result for unknown safety/support, stochastic transitions, POMDPs or UAVs.
- No scalable exact Bellman solver for large T is claimed; the verification tool
  restricts finite-game calculations to T<=36 to avoid uncontrolled computation.
- No single new horizon-free finite-optimal learner is implemented or tested.
- No posterior/prior selection uses the private truth index. Verification uses
  the known public collection of three laws, as the theoretical learner may.
- Exact rational checks are not independent formal machine proof or empirical
  validation. A prospective experiment would need separate authorization.

## Open Risks and Reproducibility

The handwritten finite-game argument and transcription of physical routes still
require independent proof review. Primitive versus macro Bellman agreement,
direct risk algebra, known-model execution equality and closed-terminal witness
checks reduce implementation risk but do not substitute for that review.

Verification source: `scripts/verify_bundling_finite_budget_theory.py`.
Tests: `tests/test_bundling_finite_budget_theory.py` (7 focused mathematical tests).
Authoritative exact receipt:
`artifacts/bundling_finite_budget_theory_20260916/strict_benefit_certificate.json`.
Intermediate discovery receipts in that directory are not experimental evidence.

Recompute the encoded certificate WITHOUT floating optimization or sampling:

```sh
.venv/bin/python scripts/verify_bundling_finite_budget_theory.py --verify-receipt artifacts/bundling_finite_budget_theory_20260916/strict_benefit_certificate.json
```

The receipt records source SHA256, public hypothesis order, exact rational
endpoints, and sampled_trajectories=0. All frozen benchmark sources and the
previous T=4096 outputs are left unchanged.
