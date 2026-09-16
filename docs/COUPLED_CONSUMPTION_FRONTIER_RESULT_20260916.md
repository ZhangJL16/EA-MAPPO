# Derivation Package: A Feasible Coupled Consumption Frontier

This package is now Example 1 and a matching special-case learner for
[the general resource-coupled lower theorem](GENERAL_RESOURCE_COUPLED_INFORMATION_LOWER_BOUND_20260916.md).
It is not promoted to a general matching theorem or standalone novelty claim.

## Target

Deliver a concrete minimal coupled class, solve its information allocation,
and give one safe unknown-model learner attaining the coefficient. Do not stop
at a framework, counterexample, numerical LP, or an unattainable oracle allocation.
The target is an instance-dependent asymptotic theorem, NOT minimax log regret.

## Status

COHERENT AS STATED under the explicit class below. Handwritten theorem draft:
nonseparable capacity-dependent allocation; decision-derived lower bound;
one confidence-certification learner attaining both stated coefficients.
No independent proof audit, machine probability proof or novelty certification.
The result is feasible and explicit; coupling alone is not sufficient ICML novelty.

## Invariant Object

The optimal log-calendar-time regret coefficient at the SAME fixed true model,
same optimal gain, same observable parameters, and different recharge capacities.

## Assumptions

There are five base states q,h_0,h_A,h_B,h_F. Only q reloads. All transitions,
costs and the safe support are known and deterministic. Every action consumes 1;
debit before reloading on arrival at q. q has baseline stay with deterministic
reward r=1/5, and depart to h_0 with reward 0. The other actions are:

| State | Action / successor | Observed reward |
|---|---|---|
| h_0 | measure A -> h_A | Bernoulli(mu_A) |
| h_0 | measure B -> h_B | Bernoulli(mu_B) |
| h_0 | return -> q | 0 |
| h_A | measure B -> h_F | Bernoulli(mu_B) |
| h_A | return -> q | 0 |
| h_B | measure A -> h_F | Bernoulli(mu_A) |
| h_B | return -> q | 0 |
| h_F | return -> q | 0 |

Samples of A and B are independent IID Bernoulli streams, shared across the
two base rows where each is measured. A primitive reward is always in [0,1].
The agent observes the reward at its measurement step, not only at recharge.

Freeze the base parameter class ONCE: mu in (0,1)^2 such that both finite
catalogues {baseline,A-only,B-only} and {baseline,A-only,B-only,AB} have unique
maximal reward/time rates. This excludes optimal ties only; nonoptimal ties
are allowed. It is a B- and T-independent open union of cells. This technical
restriction is used by the confidence-certification upper bound, which would
otherwise need additional tie handling. The true model is mu_A=mu_B=m=1/10.
The theorem treats B=3 and B=4; all primitives and Theta remain identical.

## Notation

The full catalogues have rates:

    baseline: r; A-only: mu_A/3; B-only: mu_B/3; AB: (mu_A+mu_B)/4.

An empty depart/return cycle has rate 0 and is dominated. AB and BA have the
same observation/cost law; the theoretical learner uses the fixed order AB.
At the true model, baseline uniquely maximizes both catalogues and rho^*=r.
Let k_A,k_B,k_AB be normalized expected counts of complete informative cycles,
and x_A=k_A+k_AB, x_B=k_B+k_AB the normalized measurement counts.
Normalization is division by log T. Incomplete-cycle errors are bounded constants.
Let a=kl(m,3r), b=kl(m,2r), with Bernoulli KL.

## Derivation Strategy

Resource gate permits joint probing -> shared travel cost -> coupled occupancy
projection -> multi-parameter decision alternatives -> closed-form coefficient ->
joint confidence certification with a safe revealing block -> matching upper.

## Derivation Map

1. Verify safety, S0 regularity and unchanged observability/gain.
2. Project ALL safe adaptive policies to cycle-count geometry up to O(1).
3. Derive required tests from low regret at the true and alternative models.
4. Solve the two capacity-dependent allocation problems.
5. Specify one theoretical learner for all unknown parameters in frozen Theta.
6. Prove uniform efficiency and its matching coefficients at the stated model.
7. Separate an actual coupled result from a claim of new semi-bandit theory.

## Main Derivation

### Proposition 1: safety and regularity, without new unknown parameters

At B=3, depart then one measurement leaves 1 unit: return is safe but measuring
again would leave 0 at h_F and prevent the continuing return. The second
measurement is therefore not viable. At B=4 the second measurement is safe.
Every noncharger state has an immediate known return action, so H_R=1 on the
reachable domain at both capacities; p_0=1. Every base cycle consumes positive
resource, and the off-charger base graph is acyclic. S<=5, A<=3.

At the true model, h(q_B)=0 and h(x)=-r at every reachable noncharger state
solve AROE: direct return gives h=-r, while any measurement followed by return
has centered value m-2r<-r. At q the baseline action gives scalar gain r;
depart has smaller value. Thus canonical span is exactly r=1/5 at both B.

Both unknown Bernoulli means are queryable at B=3 already, so the observation
quotient and accessible base parameter set do not improve at B=4. What improves
is the ability to acquire both observations in ONE excursion. Actual safety
does not depend on mu: learning cannot change the known safety kernel here.

### Proposition 2: cycle geometry includes adaptive within-cycle choices

Every completed nonempty excursion has category A-only, B-only, or AB. At most
one incomplete excursion remains at a deterministic calendar cutoff. Each
measurement count differs from its complete-category count by at most one.
Reward expectation is mu_i times the expected ACTUAL measurement count, since
that action is selected before observing its reward. It need not be mu_i times
a reward-dependent completed-category count exactly; replacing the two counts
introduces only O(1). Empty cycles only increase regret and reveal no information.

Consequently all safe policies, including policies choosing their second action
after seeing the first reward, have at the true model

    Reg(T)=(3r-m) E K_A+(3r-m) E K_B+(4r-2m) E K_AB
              +2r E K_empty+O(1).

Cycle durations likewise give time 3K_A+3K_B+4K_AB+2K_empty plus baseline
steps and a residual <=4. This shows that arbitrary adaptive within-cycle
policies cannot improve the average gain over the best catalogue ratio:
their expected reward and time are mixtures of the same category coefficients.
For a general mu use mu_A,mu_B in these coefficients. Thus the upper-bound
learner's nonadaptive catalogue restriction does NOT change the comparator.

At B=4, for given measurement totals x_A,x_B, shared-cycle regret is minimized
by k_AB=min(x_A,x_B), with remaining counts supplied by singles. The projected
cost is therefore

    G_4(x_A,x_B)=0.5(x_A+x_B)-0.4 min(x_A,x_B).

At B=3 the projected cost is G_3=0.5(x_A+x_B). The discount is the two shared
travel steps. This is a nonseparable, piecewise-linear cost; a measurement of
A or B is not an independent arm with a fixed marginal exploration cost.

### Proposition 3: decision-derived information constraints

The true baseline policy has no unknown outcomes. Decision-relevant alternatives
are those where baseline is strictly suboptimal:

    Alt_3={nu_A>3r OR nu_B>3r},
    Alt_4=Alt_3 union {nu_A+nu_B>4r}, intersected with frozen Theta.

Strict unique-optimal points are dense in these regions; excluded tie boundaries
do not change the infimum information distance.

For every uniformly efficient learner and fixed alternative nu, define the
transcript event A_T={number of baseline steps >=T/2}. At the true model all
informative/empty cycles have a strictly positive rate gap, so low regret
implies expected off-baseline time o(T^alpha) for every alpha>0. Hence
P_mu(A_T^c)=o(T^(alpha-1)). At nu, baseline steps have per-step loss
rho_nu^*-r>0, all complete-category losses are nonnegative, and the incomplete
cycle contributes only O(1). Low regret at nu therefore implies
P_nu(A_T)=o(T^(alpha-1)). Binary data processing, followed by alpha down to 0,
gives the necessary constraint

    liminf [E N_A kl(m,nu_A)+E N_B kl(m,nu_B)]/log T >=1.

Thus tests follow from the original control objective, not full parameter
identification. The information comes from ACTUAL primitive measurement counts.

For B=3 the constraints reduce to x_A a>=1 and x_B a>=1. For B=4 they also
require

    inf_{nu_A+nu_B>=4r} [x_A kl(m,nu_A)+x_B kl(m,nu_B)] >=1.

This last constraint is genuinely joint: alternatives can make the AB sortie
optimal without making either single sortie optimal. For fixed positive x it
is a strictly convex one-dimensional minimization at nu_A+nu_B=4r.

### Theorem A: explicit coupled lower coefficients

For any uniformly efficient learner on frozen Theta,

    liminf Reg_3(T)/log T >= C_3=1/a,
    liminf Reg_4(T)/log T >= C_4=0.3/b.

Proof for B=3: minimize 0.5(x_A+x_B) with the two diagonal constraints.

Proof for B=4: its cost G_4 is convex and symmetric, and its feasible information
set is convex and symmetric. Averaging (x_A,x_B) with (x_B,x_A) preserves
feasibility and cannot increase cost. An optimum is therefore symmetric x_A=x_B=x.
The joint information minimum is attained at nu_A=nu_B=2r by strict convexity.
Hence x>=1/min(a,2b), and G_4(x,x)=0.6x. At m=.1,r=.2,

    a-2b=0.1 log(8/3)>0,

so x=1/(2b), k_AB=x, k_A=k_B=0, and C_4=0.6/(2b).

For the lower bound's limit interchange, take a subsequence with bounded
Reg/log T; strictly positive true-model cycle costs bound all normalized cycle
counts. Extract a convergent subsubsequence. Every fixed alternative's necessary
constraint holds at its limit, hence this limit is feasible for the program.
An unbounded subsequence already exceeds the finite coefficient. This avoids
assuming uniform convergence over the continuum of alternatives.

Numerically:

    a=0.5506612476718906, b=0.22628916118535888,
    C_3=1.8159985003990078, C_4=1.3257373814482558.

The reduction is approximately 26.997%. Strict C_4<C_3 also follows analytically:
Pinsker gives b>=0.18, and log(8/3)<1 implies a=2b+0.1 log(8/3)<(10/3)b.
This is a change in a log-regret coefficient, not minimax scaling or unbounded B.

### One feasible unknown-model learner: certify or reveal

At B=3 a revealing block is one A-only and one B-only sortie (6 steps). At B=4
it is one AB sortie (4 steps). Initialize with one revealing block; every sample
count is then >=1. Keep actual A/B sample counts n_i and means hat_mu_i.

At each charger decision time t, form the joint likelihood confidence region

    H_t={nu in [0,1]^2: sum_i n_i kl(hat_mu_i,nu_i)<=f(t)},

using the following CHECKED multi-arm calibration from Kaufmann--Koolen (2021),
Theorem 7, two-sided Bernoulli version. Let C_exp be its exact Equation (10)
calibration (computable using the inverse of u-log u on [1,infinity)). Define

    z_t=log(t+3)+2 log log(t+3),
    f(t)=6 log(1+log(t+3))+2 C_exp(z_t/2).

The theorem gives P_mu(mu not in H_t)<=exp(-z_t), and sum_{t>=0} exp(-z_t)<infinity.
It also gives C_exp(x)/x->1, so f(t)/log t->1. The correction uses n_i<=t;
H_t's exact set can be conservative without changing its leading coefficient.
Sequential primitive measurements satisfy adaptive predictable sampling; no
independent action counts or false product-of-single-arm-CIs argument is used.

If one catalogue policy j is rate-optimal for EVERY nu in H_t, execute j for
one cycle/step. Otherwise execute a revealing block. If multiple policies are
universally optimal, use a fixed tie rule. The certification consists of at most
four affine rate comparisons on a two-dimensional convex KL ball; it requires
no true mu or oracle occupation. Revealing blocks and all catalogue policies
are pre-certified safe. The algorithm is theoretical only, not implemented.

### Proposition 4: uniform efficiency and matching at the stated model

Wrong certified exploitation requires mu not in H_t, so its expected number
of decisions over ALL time is bounded by sum_t exp(-z_t). Each lasts <=4;
its expected cumulative regret and extra samples are O(1).

Let R(T) be the number of revealing blocks. Both actual sample counts are at
least R up to initialization/end corrections. For any fixed mu in frozen Theta,
the true optimal catalogue policy is unique and its decision cell has a positive
KL distance from mu to its complement. Continuity gives D_epsilon>0 uniformly
for empirical means within epsilon of mu. If R>f(T)/D_epsilon, the KL confidence
set is entirely in this cell, so revealing cannot be required. Further blocks
must be charged to empirical deviations. Each revealing block increments both
counts; therefore each arm's sample index is charged at most once. IID Hoeffding
tails sum to O(epsilon^-2), even if additional certified sorties sampled an arm.
Hence E R(T)=O(log T), and Reg=o(T^alpha) for every fixed mu and alpha>0.
The unique-optimal restriction is important: this argument does not cover
arbitrary boundary ties using the same certification rule.

At the stated symmetric model the relevant distances are

    D_3=inf_Alt3 sum_i kl(m,nu_i)=a,
    D_4=inf_Alt4 sum_i kl(m,nu_i)=min(a,2b)=2b.

The same empirical-neighborhood/count argument gives
limsup E R(T)/log T<=1/D_B. Baseline certified exploitation has zero regret;
wrong certified exploitation has bounded expected regret. Revealing-block
costs at the true model are 1 at B=3 and .6 at B=4. Theorem A supplies the
opposite inequalities, and therefore this ONE model-independent learner attains

    lim Reg_3(T)/log T=C_3,
    lim Reg_4(T)/log T=C_4.

The coefficients concern the same true primitive model, gain .2, observable
parameter class and span .2. The 27% advantage is entirely acquisition geometry.

## Remarks and Interpretation

- Coupled Frontier Test: YES. Nonseparable shared-cost occupancy and genuinely
  multi-parameter alternatives both occur with five states and two unknown means.
- Attainability Test: YES in this frozen class at the stated model. A safe
  confidence-certification learner attains the solved frontier without an oracle.
- This is not a claim that ALL coupled acquisition is new. The catalogue is
  also a variable-duration combinatorial semi-bandit with shared observations.
  Existing structured bandit theory is direct adjacent prior art. The actual
  result is stronger than another feasibility-threshold example, but it still
  needs a novelty audit before becoming a main paper theorem.
- The statement 'resources matter ONLY when coupling exists' is not justified;
  earlier feasibility thresholds already affect learning. Coupling here changes
  quantitative information cost even when identifiability and gain are unchanged.
- No information-dependent safety learning is claimed: support/cost were known.
  Unknown safety would need new authorization and a different learnability model.
- All B>=4 saturate in this minimal model. No growing-in-B rate is claimed.

## Sources and Verification Basis

The following are PRIMARY sources, not claimed citations proving this draft:

| Source | Verified relevance / limitation |
|---|---|
| Kaufmann--Koolen, JMLR 2021, Theorem 7, Eq. (10) | Exact multi-arm adaptive confidence calibration used above; NOT guessed from individual KL-UCB bounds |
| Combes et al., NIPS 2015, Combinatorial Bandits Revisited | Semi-bandit information lower bounds and structure-aware exploration; coupling itself is prior art |
| Ok et al., NeurIPS 2018, Exploration in Structured RL, Theorem 4 | Requires every stationary policy to induce an irreducible chain, plus allocation uniqueness/continuity; cannot be directly invoked here because baseline stays at q |
| Hou et al., ICML 2023, Probably Anytime-Safe Stochastic Combinatorial Semi-Bandits | Lifetime high-probability variance budget, not deterministic consumption/reload safety; related but not the same theorem |

https://www.jmlr.org/papers/volume22/18-798/18-798.pdf
https://arxiv.org/abs/1502.03475
https://papers.neurips.cc/paper/8103-exploration-in-structured-reinforcement-learning.pdf
https://proceedings.mlr.press/v202/hou23d.html

Nature academic-search MCP was unavailable; its OpenAlex fallback script returned
HTTP 429. Publisher/conference and arXiv primary pages/PDFs were then checked.
No exhaustive bibliography or publication-novelty certification is claimed.

The analytic solution was cross-checked by a separate constrained numerical solve
of the two-dimensional information program, which found x_A=x_B=2.2095623024
and objective 1.3257373814. This is a numeric cross-check, not the proof.

## Boundaries and Non-Claims

Handwritten mathematical draft with a published concentration input. Independent
audit remains necessary. No minimax log T claim, arbitrary-CMDP attainability,
unknown transition result, computational hardness separation, ICML acceptance
prediction, algorithm source implementation, training, UAV, DEV/CONFIRM execution.

## Open Risks

1. Independently audit the adaptive complete-cycle projection's O(1) cutoff and
   low-regret-derived baseline test; do not assume completed counts are predictable.
2. Audit confidence threshold/certification and tie exclusion; leading coefficient
   relies on multi-arm sum-KL confidence, not rectangular marginal bounds.
3. The five-state result may remain an illustrative instance of known structured
   semi-bandit theory. A genuinely new theorem must demonstrate more than existence
   of a coupled polytope or a shared-travel discount.
4. No transfer to UAV or empirical throughput follows without a separate model.
