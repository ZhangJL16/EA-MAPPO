# Derivation Package: S1 Universal Capacity Equivalence Kill Test

## Target

Test whether the S0 regularity assumptions imply regret-preserving equivalence
between capacities of the SAME fixed base parameter class. In particular test
R_B(T)<=C R_B0(T)+O(1), uniformly in T, without further assumptions.

## Status

COHERENT AFTER REFRAMING: universal equivalence is refuted by the explicit
fixed-class example below. This is an elementary feasibility-gated bandit
counterexample, not a proposed new paper contribution. The full S1 minimax
characterization for arbitrary regular classes remains OPEN. Handwritten,
AI-assisted proof; no independent audit or machine probability proof.

## Invariant Object

R_B(T)=inf over safe learners sup over theta in Theta of
T rho_B,theta^* - E_theta sum_{t=0}^{T-1} reward_t,
starting from full charger q_B. Theta is frozen once for every B and T.

## Assumptions

- Arrival-normalized recharge: debit c(s,a) first, require enough resource
  BEFORE recharge, then reset to B when the successor is q.
- B>=2 is an integer; a learner observes only the reward of its executed action.
  There is no generative oracle or reward observation for unsafe actions.
- Fixed base states {q,w}; q is the only reload. Fixed deterministic support,
  fixed known integer costs, and Bernoulli reward samples in [0,1].
- Theta={(sigma,Delta): sigma in {+,-}, 0<Delta<=1/4}, independent of B,T.
  Under +, (mu_1,mu_2)=(1/2+Delta,1/2); under -, the means are swapped.
  Samples are conditionally independent given the selected actions.
- Base transitions are known in this example. It is a subclass of the broader
  unknown-transition/reward setting: unknown rewards alone suffice to refute
  a universal statement covering that broader class. It does not establish an
  unknown-transition-specific obstruction.

## Notation

At q, a_0 has cost 1, reward 0, and successor q.
At q, a_1 and a_2 each have cost 2, reward Bernoulli(mu_i), and successor w.
At w, the only action u has cost 1, reward 0, and successor q.
All these primitives and the reward family are identical for all capacities.
For T=2n define N_i as the number of a_i executions within these T steps.
P_+ and P_- denote full adaptive transcript laws for a fixed learner.

## Derivation Strategy

Known safety -> capacity-dependent feasible experiments -> explicit AROE ->
adaptive two-environment testing lower bound -> standard bandit upper bound.
No approximation is used. Only the gap selected INSIDE fixed Theta depends on n.

## Derivation Map

1. Compute viable and reachable states under the fixed resource timing.
2. Check S0 regularity and compute the canonical bias at both capacities.
3. Show the small-capacity transcript is independent of theta and regret is zero.
4. For larger capacity, bound regret by missed optimal pulls, then use KL/testing.
5. Give a standard existing-policy upper bound without implementing an algorithm.
6. Separate feasibility-threshold non-equivalence from growing-B hardness.

## Main Derivation

### Proposition 1: all S0 regularity conditions hold

For B=2, a_i would enter (w,0). From there u cannot be executed safely, so a_i
is excluded from the viable action set at q_B. The reachable safe domain is
just {q_B}, and a_0 is its only action. In particular, the immediate debit 2
alone being feasible does NOT suffice for viability.

For every B>=3, q_B permits all three actions, and the reachable domain is
{q_B,(w,B-2)}. Every w representative immediately returns safely using u.
Thus recovery has H_R=1, minimum positive transition probability is p_0=1,
and every base cycle has strictly positive consumption. State/action and
unknown reward-parameter complexity are fixed. Safe scalar gain and AROE exist.

### Identity 2: canonical bias is uniformly bounded

For B=2, rho^*=0 and h(q_B)=0. For B>=3 put mu_* = max(mu_1,mu_2). Then

    rho^*=mu_*/2,  h(q_B)=0,  h(w,B-2)=-rho^*.

At w the AROE is rho+h(w)=0. At q it is
rho=max{0,mu_1+h(w),mu_2+h(w)}=mu_*-rho.
Consequently the canonical span is 0 at B=2 and mu_*/2<=3/8 at EVERY B>=3.
No large bias or rare transition is involved.

### Proposition 3: R_2(T)=0 exactly

Every safe trajectory uses only a_0, earns zero reward, and has optimal gain
zero. Its transcript has the SAME law for every theta. Hence minimax regret
is exactly zero for every horizon, not merely bounded.

### Proposition 4: R_B(2n)=Theta(sqrt(n)) for every B>=3

First prove the lower bound for arbitrary safe adaptive learners, including
learners that idle using a_0. Two successive a_i executions require an intervening
u, so N_1+N_2<=n in 2n steps, even if the final action is an unpaired a_i.
Expected reward equals mu_1 E N_1+mu_2 E N_2. Therefore, under either sign,

    Reg=mu_* E[n-N_1-N_2]+Delta E N_suboptimal
       >=Delta (n-E N_optimal),

because mu_*>=Delta. This explicitly accounts for lost time from idling.

Let A={N_1>=n/2}. In the + environment, on A^c the missed-optimal count is
at least n/2. In the - environment, on A, N_2<=n-N_1<=n/2. Hence

    Reg_+ + Reg_- >= Delta n/2 [P_+(A^c)+P_-(A)].

By the adaptive KL chain rule, all deterministic transitions and learner action
kernels contribute zero divergence. Each reward observation contributes at most
6 Delta^2, using kl(Ber(p),Ber(q))<= (p-q)^2/[q(1-q)] and
p,q in [1/2,3/4]. At most n such observations occur. Thus

    KL(P_+ || P_-)<=6 n Delta^2.

The testing inequality and Pinsker give

    P_+(A^c)+P_-(A)>=1-TV(P_+,P_-)
                         >=1-sqrt(3n Delta^2).

Select Delta=1/(8 sqrt(n)), which belongs to the fixed Theta for every n>=1.
Then at least one of the two environments has

    Reg >= (sqrt(n)/32) (1-sqrt(3)/8).

This is a class-specific information-theoretic lower bound; it is NOT the
general linear-mixture span lower bound applied to a subclass without proof.

For the upper bound, an existing two-arm bandit policy can select a_i at q,
always execute u next, and produce exactly n bandit pulls in 2n steps. Its
bandit regret is IDENTICAL to this lift's gain-based regret. For example the
existing EXP3 bound 2.7 sqrt(n K log K), with K=2, is O(sqrt(n)); the log K
is a constant here. This is only an existence reduction, not algorithm design
or execution. Primary source: Audibert and Bubeck, COLT 2009, Theorem 1:
https://www.microsoft.com/en-us/research/wp-content/uploads/2017/01/COLT09_AB.pdf
The theorem's adversarial bound also applies to independent Bernoulli rewards
by averaging over their realizations. Thus the stated upper/lower match in T.

### Corollary 5: universal multiplicative or O(1)-additive equivalence fails

For the SAME fixed Theta, R_2(2n)=0 whereas R_B(2n)>=c sqrt(n) for B>=3.
No finite T-independent C and additive constant can satisfy
R_B(T)<=C R_2(T)+O(1). A two-way learner transfer preserving expected regret
up to O(1) would contradict this minimax lower bound. The ratio with R_2 in
the denominator is undefined; it must not be reported as a numeric ratio.

For this example, all B>=3 ARE exactly equivalent: the two reachable states,
action availability, reward experiment and time costs are identical after
mapping (w,B-2) to (w,B'-2). Their transcript laws and regrets match exactly.
The counterexample does not refute equivalence above a sufficiently large
class-specific threshold, or an equivalence theorem under additional assumptions.

## Remarks and Interpretation

- Sanity audit: 30 exact Fraction capacity/AROE checks and exhaustive checks
  of 7,278 supported finite action histories (even horizons 2 through 12)
  passed. The latter check N_1+N_2<=n and the pointwise missed-pull/testing
  regret inequalities, including idling and final unpaired pulls. These checks
  do not verify the KL chain rule or the infinite-horizon/general theorem.
- Parameter dimension and span do not determine which unknown parameters are
  observable and relevant to the optimal safe policy. Capacity gates experiments.
- The example changes no base primitive with B, and uses p_0=1. This is a real
  capacity effect on the feasible statistical experiment, NOT base degeneracy.
- It is a threshold effect: zero regret becomes sqrt(T). There is no Omega(B)
  growth, no increasing learning horizon, and no new statistical technique.
- Failure of an exact/universal equivalence theorem does NOT force an
  Omega(B)-dependent lower bound. That proposed dichotomy is false.
- A B-free worst-case upper bound and unequal capacities' minimax risks can
  coexist. 'No extra leading capacity factor' and 'statistically equivalent'
  are distinct claims and need distinct theorem statements.
- Chae et al.'s leading span-controlled upper bound remains relevant but does
  not itself imply a multiplicative comparison with another class's risk:
  https://proceedings.mlr.press/v258/chae25a.html

## Boundaries and Non-Claims

No universal S1 rate for arbitrary Theta, no eventual-capacity equivalence, no
counter-specific growing-B lower bound, and no computational separation is proved.
No new algorithm, UAV run, training, CONFIRM access, or scientific Gate analysis.
This elementary obstruction is not Spotlight novelty. Reward sampling is
explicitly stochastic; a model revealing deterministic reward upon one execution
would NOT support this sqrt(T) lower bound.

## Open Risks

1. Independent audit of adaptive transcript KL and finite-horizon counts.
2. Next theorem needs a precise claim: uniform B-free achievable rate, or
   capacity comparison above an independently justified feasibility threshold.
3. General fixed Theta may have zero regret or other rates; a universal positive
   matching lower bound requires a specified nondegenerate embedded hard family.
4. Observation/reward-law assumptions and comparator timing must remain explicit.
5. Existing structured-RL reductions may already establish the desired upper;
   novelty must be audited before turning this obstruction into a paper story.
