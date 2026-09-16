# General Resource-Coupled Information Lower Bound

## Target

Replace the five-state example as the main mathematical object by a general
definition and a decision-derived asymptotic lower-bound theorem. Recover the
existing coupled result as Example 1, with its matching learner as a special-case
attainability result. Do not claim a general matching upper or minimax theorem.

## Status

Handwritten theorem and proof below, for finite deterministic Consumption MDPs
with shared unknown reward laws. The parameter class can be finite or infinite.
This is a generalization, not an independent proof audit or novelty certificate.
No algorithm implementation, training, UAV run, or DEV/CONFIRM access occurred.

A subsequent [class-wide attainability draft](RESOURCE_PATH_CATALOGUE_ATTAINABILITY_20260916.md)
proves a matching upper for finite reward hypotheses with common finite supports
and unique optimal path types. The general continuous/tie-allowing upper remains
open; the lower theorem below is unchanged.

## Invariant Object

Freeze a base class M_theta=(S,A,F,c,q,nu_theta), theta in Theta, once.
S,A,F,c,q,Theta and the reward-law parameterization do not depend on capacity B
or horizon T. Only the known deterministic counter changes under Lift_B.
The object is the instance-dependent cost of decision-relevant information:

\[
C_B(\theta)=\inf_{x\ge0}G_{B,\theta}(x)
\quad\text{subject to}\quad
\sum_i x_i I_i(\theta,\theta')\ge1
\quad(\theta'\in\operatorname{Alt}_B(\theta)).
\]

## Assumptions

1. S and A are finite; F(s,a) is a known deterministic successor. There is one
   reload state q. Costs c(s,a) are known nonnegative integers, and every base
   cycle has strictly positive total consumption (decreasing).
2. Debit c before transition; an action is forbidden if the debit would make
   the battery negative. On arrival at q the battery becomes B. Identify all
   charger copies with q_B=(q,B). Start there.
3. D_B is the nonempty safe reachable domain from q_B; actions are restricted
   to its viable, successor-closed safe action correspondence. Assume q_B has
   a safe continuation. Neither the domain nor the action restriction depends
   on theta, because F and costs are known.
4. At a primitive row i=(s,a), the observed reward has law nu_{theta,i} on [0,1],
   conditional on the entire past and that row. Its mean is r_theta(i).
   The row laws may share parameters. All additional observations are either
   known or included in the same outcome law and its KL. Thus no information
   channel is omitted. Conditional sampling is stationary, not an assumption
   that completed adaptive excursions are IID.
5. For every pair theta,theta', each accessible row has finite
   I_i(theta,theta')=KL(nu_{theta,i} || nu_{theta',i}). In particular the
   transcript laws admit the usual conditional KL chain rule. Infinite-KL
   cases require a separate formulation and are not claimed here.
6. Work at a fixed feasible B. A learner is safe and uniformly efficient on
   the SAME class Theta: Reg_theta(T)=o(T^a) for every theta and every a>0,
   where Reg_theta(T)=T rho_theta^* - E_theta sum_{t<T}reward_t.

S0's uniform recovery and p_min bounds may additionally be imposed to make the
bias bound uniform in B. They are not necessary for this fixed-B lower theorem.
The present theorem does NOT cover unknown stochastic transitions.

## Notation

Off q_B the finite lifted support graph is acyclic: a lifted cycle would project
to a zero-consumption base closed walk, contradicting decreasing. Consequently
every safe trajectory returns to q_B within a fixed finite bound L_B.

Let P_B be the finite catalogue of ALL complete legal q_B-to-q_B action paths,
with no internal q_B. Include a safe one-step charger loop if present. A path
need not be base-simple. Let m_p(i) be its primitive-row count, ell_p its length,
and R_theta(p)=sum_i m_p(i)r_theta(i). Define

\[
\rho_\theta^*=\max_{p\in\mathcal P_B}\frac{R_\theta(p)}{\ell_p},
\qquad c_\theta(p)=\rho_\theta^*\ell_p-R_\theta(p)\ge0,
\qquad \mathcal P_\theta^*=\{p:c_\theta(p)=0\}.
\]

Repeating a maximizing path achieves this gain. Conversely, expected primitive
reward counts of any adaptive safe learner decompose into complete path counts
plus one bounded partial path. A mixture cannot exceed the maximal path rate
asymptotically. This justifies the comparator even with within-excursion feedback.
Do NOT condition on a reward-selected path and assert that its expected reward
equals R_theta(p); the argument uses predictable primitive-row counts instead.

Canonical bias h_theta(q_B)=0 exists here by backward recursion on the off-q
DAG at this rho_theta^*. At q_B, the maximized centered complete-path value is
zero, so this recursion satisfies the charger Bellman equation too. Write

\[
d_\theta(x,a)=\rho_\theta^*+h_\theta(x)-r_\theta(s,a)
                 -h_\theta(F_B(x,a))\ge0,
\quad H_\theta=\operatorname{sp}_{D_B}(h_\theta).
\]

Telescoping on every complete deterministic path gives
sum_{(x,a) in p} d_theta(x,a)=c_theta(p). Set
j_{theta,theta'}(p)=sum_i m_p(i)I_i(theta,theta').

## Definition: decision-relevant confusing alternatives

Let F_theta^* be the union of primitive rows visited by theta-optimal paths.
Define

\[
\operatorname{Alt}_B(\theta)=\left\{\theta'\in\Theta:
\nu_{\theta',i}=\nu_{\theta,i}\ (i\in F_\theta^*),\quad
\mathcal P_\theta^*\cap\mathcal P_{\theta'}^*=\varnothing
\right\}.
\]

The first condition excludes information available at zero theta-regret cost;
the second makes the alternative require a different control decision. Different
gain alone is not enough. Different parameters alone are not enough. This is a
Graves--Lai-style confusing-alternative definition, not a new identification rule.
An alternative outside this set can still matter for other bounds; no claim of
necessity or general attainability is made by restricting to this set.

## Definition: resource-coupled acquisition geometry

For exploration row allocation x, define the extended-valued function

\[
G_{B,\theta}(x)=
\inf_{\lambda\ge0}\left\{
\sum_{p\notin\mathcal P_\theta^*}\lambda_p c_\theta(p):
\sum_{p\notin\mathcal P_\theta^*}\lambda_p m_p=x
\right\}.
\]

An empty feasible set has value infinity. Its domain is the cone of primitive
row-count vectors generated by nonoptimal safe excursions. G is convex,
positively homogeneous and polyhedral. It is piecewise linear, not a claim of
nonlinear smooth geometry. Shared travel and resource gates can make it
nonseparable across observation coordinates.

Lambda is a log-T-normalized number of excursions, not a probability vector;
there is no sum(lambda)=1 constraint. Optimal paths are omitted because their
KL against every confusing alternative is zero, even if executed O(T) times.
The underlying safe path catalogue is known; costs and the optimal-path exclusion
depend on theta. This distinction matters for an unknown-model upper bound.

Equivalently,

\[
C_B(\theta)=\inf_{\lambda_p\ge0, p\notin\mathcal P_\theta^*}
\sum_p\lambda_p c_\theta(p)
\quad\text{s.t.}\quad
\sum_p\lambda_p j_{\theta,\theta'}(p)\ge1
\ (\theta'\in\operatorname{Alt}_B(\theta)).
\tag{GL-B}
\]

For finite Theta this is a finite LP. Finite does not mean polynomial in log B:
the path catalogue can be large. For infinite Theta it is a semi-infinite program.
Empty alternatives give C=0; infeasible constraints give C=infinity.

## Derivation Strategy and Map

Safe excursion catalogue -> centered path costs -> low-regret decision test ->
transcript information requirement -> resource-constrained allocation lower bound.
The substantive missing implication in S2 is proved next, rather than assumed
as an externally supplied testing requirement.

## Lemma: low regret forces decision-relevant discrimination

For any theta,theta' with disjoint optimal path sets, a safe uniformly efficient
learner satisfies

\[
\liminf_{T\to\infty}
\frac{\operatorname{KL}(\mathbb P_\theta^{A,T}\Vert
                         \mathbb P_{\theta'}^{A,T})}{\log T}\ge1.
\tag{1}
\]

Proof. Define

\[
\gamma=\min_{p\in\mathcal P_B}
\frac{c_\theta(p)+c_{\theta'}(p)}{\ell_p}>0.
\]

Positivity follows from a finite catalogue, nonnegative costs, and no path
optimal in both models. On ANY realized safe action trajectory, let
S_theta(T)=sum_{t<T}d_theta(X_t,A_t). Complete paths cover at least T-L_B steps,
and the remaining slacks are nonnegative. Therefore, pathwise,

\[
S_\theta(T)+S_{\theta'}(T)\ge\gamma(T-L_B).
\tag{2}
\]

The stopped Bellman identity gives
E_theta S_theta(T)=Reg_theta(T)-E_theta h_theta(X_T)=o(T^a)+O(H_theta).
This sum is nonnegative. The event
A_T={S_theta(T)<=gamma T/2} is measurable from the action/state transcript.
It is a proof-side test indexed by theta, not extra knowledge supplied to the
learner. Markov's inequality gives

P_theta(A_T^c)=o(T^{a-1}),

and, using (2), for T>2L_B,

P_theta'(A_T)<=E_theta' S_theta'(T)/(gamma T/2-gamma L_B)=o(T^{a-1}).

Data processing bounds transcript KL below by binary KL for A_T. Its first
probability tends to one; its second is o(T^{a-1}). Thus the liminf divided by
log T is at least 1-a, for every a in (0,1). Let a decrease to zero. This proves
(1). The proof needs neither a unique optimal path nor independent excursions.
QED.

## Theorem: general resource-coupled information lower bound

Under the assumptions above, every safe uniformly efficient learner obeys

\[
\boxed{\displaystyle
\liminf_{T\to\infty}\frac{\operatorname{Reg}_\theta(T)}{\log T}
\ge C_B(\theta).}
\tag{3}
\]

Proof. Let K_p(T) count completed paths of type p. This variable may depend on
rewards observed inside that path. We use pathwise row-count identities and the
predictable conditional KL chain rule, NOT Wald's identity on K_p.
With constants depending on the fixed B and models but not T,

\[
\operatorname{Reg}_\theta(T)
=\sum_p c_\theta(p)E_\theta K_p(T)+O(1),
\tag{4}
\]

\[
\operatorname{KL}(\mathbb P_\theta^{A,T}\Vert\mathbb P_{\theta'}^{A,T})
=\sum_p j_{\theta,\theta'}(p)E_\theta K_p(T)+O(1).
\tag{5}
\]

For (4), sum Bellman slacks on complete paths, leave a partial path of length
at most L_B, and use the bounded terminal bias. Its error is bounded by
H_theta+L_B max d_theta. For (5), primitive-row counts equal completed-path row
counts plus at most L_B observations; their KL remainder is bounded by
L_B max_i I_i(theta,theta'). Adaptive learner action kernels cancel in the
chain rule because the learner is the same in both models.

For a confusing alternative, j(p)=0 on theta-optimal paths. Apply (1) and (5):

\[
\liminf_T\sum_{p\notin\mathcal P_\theta^*}
\frac{E_\theta K_p(T)}{\log T}\,j_{\theta,\theta'}(p)\ge1.
\tag{6}
\]

If the regret/log T liminf is infinite, (3) is immediate. Otherwise choose a
sequence achieving a finite liminf. Every nonoptimal path has strictly positive
c_theta(p), so (4) bounds all its normalized expected counts along this sequence.
Extract a convergent subsequence in the finite-dimensional nonoptimal-path space.
Its limit lambda satisfies EVERY constraint (6). This remains valid for an
infinite alternative class: pointwise convergence of the same finite vector
preserves each fixed constraint; no uniform KL convergence is asserted or needed.
By (4), the limiting objective equals the selected regret liminf. A feasible
objective is at least (GL-B)'s infimum. If the program is infeasible, the supposed
finite-liminf subsequence cannot exist. QED.

## Example 1: the five-state coupled frontier is a corollary

Use the fixed class in COUPLED_CONSUMPTION_FRONTIER_RESULT_20260916.md, with
true means (.1,.1), charger baseline r=.2, and capacities 3 and 4. Only baseline
is optimal; its known reward supplies no information about either unknown mean.
Thus confusing alternatives are precisely the fixed-class models in which a
measurement path defeats baseline. They have no common optimal path with truth.

Project primitive rows sharing the same Bernoulli parameter onto x_A,x_B.
The definition above gives G3=.5(x_A+x_B) and
G4=.5(x_A+x_B)-.4 min(x_A,x_B). With a=kl(.1,.6) and b=kl(.1,.4),
(GL-B) recovers C3=1/a and C4=.3/b as already solved there. Empty return-only
paths carry no information and cannot improve a positive-cost allocation.

The existing safe certify-or-reveal learner attains these two coefficients:
this supplies matching upper/lower for a structure class, not for every finite
Consumption MDP. The continuous-class Bernoulli thresholds are not silently
replaced by finite-Theta alternatives: a finite grid is covered by (3), but its
exact coefficient can differ. Allowing infinite Theta makes the original result
a literal corollary rather than a discretization analogy.

## Remarks: what has and has not generalized

- From five states/two means to arbitrary finite S,A and arbitrary shared reward
  parameterization with finite row KL, including finite Theta.
- From one special baseline-count test to disjoint-optimal-path discrimination.
- From armwise allocations to a cone of complete resource-feasible path counts.
- From nonadaptive sorties to arbitrary safe randomized history-dependent
  choices, including within-excursion feedback; no IID-cycle shortcut.
- NOT to unknown stochastic transitions, a general matching learner, finite-time
  minimax rates, or capacity-independent computation.

## Prior-work boundary and publication test

The confusing-alternative information program is classical in structured bandit
and controlled-Markov learning. Combes, Magureanu and Proutiere's 2017
*Minimal Exploration in Structured Stochastic Bandits* gives a structured-arm
allocation lower bound and OSSB attainability. This theorem does not establish
that unequal durations, primitive shared feedback and adaptive resource paths
fall outside all such extensions. A finite safe path catalogue can itself be
regarded as a structured experiment catalogue; adaptive path choice requires
the proof above, but that observation alone is not a novelty certificate.

Ok et al.'s 2018 *Exploration in Structured Reinforcement Learning* is direct
prior art for information/slack allocations. Its stated DEL upper-bound
assumptions cannot simply be imported here: a charger-loop policy is not an
every-policy irreducible chain. Failure of that upper theorem's assumptions does
not make the lower-bound idea new.

Sources inspected:

- https://proceedings.neurips.cc/paper/2017/file/e19347e1c3ca0c0b97de5fb3b690855a-Paper.pdf
- https://papers.neurips.cc/paper/8103-exploration-in-structured-reinforcement-learning.pdf

## Claim Boundaries and Open Risks

Generalization gate: a general definition and general lower theorem now have a
complete handwritten proof, and Example 1 has an existing matching special-case
learner. This closes the previously missing low-regret-to-testing implication.

Publication gate: OPEN. This is still potentially a resource-specific rendering
of Graves--Lai/structured experiment theory. A reusable resource-specific
characterization, tractable representation or broader attainable class would be
needed to claim an irreducible new principle. Do not rate Spotlight readiness
from mathematical generality alone.

Independent proof audit remains open. Fixed-B O(1) remainders can depend on B;
(3) is not uniform when B grows with T. S0 controls terminal bias under its
regularity but does not by itself control path-catalogue size or all partial-path
information remainders. No inference of a growing-B minimax theorem is allowed.
