# Dynamic admissibility: pre-action information bounds

Date: 2026-09-21. Parent audit: `aed9c9f`.
Status: proof draft; no novelty clearance or general matching theorem claimed.
This document does not change protocol v1 or run any simulation.

## Results obtained and limits

1. A pre-action change-of-measure bound applies to **every** uniformly safe
   learner, not only learners using our likelihood confidence sets.
2. It yields a general ordered-elimination linear-program lower bound for finite
   menus, overlapping answer sets, deterministic conditional safety and bounded
   common-safe likelihood increments.
3. A three-model family has a matching asymptotic minimax constant. A precisely
   specified current-menu information design is arbitrarily worse at one model
   than an unlocking design. The example has competing information routes, not
   a chain of independently required bits.
4. The general lower bound is model-aware. A single implementable algorithm
   attaining it across all models has **not** been proved. Nonanticipativity is
   a substantive obstacle, not an omitted technicality.
5. The specific classical controlled-sensing theorems checked below do not
   immediately transfer after adding an absorbing catastrophe. This does not
   exclude other prior constrained-testing results or establish novelty.

## 1. Finite formulation and error convention

Models i belong to a finite set H. Query e belongs to a finite set E, costs a
known model-independent d_e in [d_min,d_max], and has a known safety indicator
s_i(e). If s_i(e)=1 it returns a live observation with law P_i^e; otherwise it
enters absorbing death. Observations are conditionally independent given the
model and query. For any pair safe under the same query, assume mutual absolute
continuity and a uniformly bounded log density ratio. A finite full-support
alphabet suffices. No regularity is imposed across a safe/unsafe pair: those
outcomes deliberately have disjoint live/dead support.

There is a finite answer set A and known nonempty acceptable subsets A(i).
An acceptable answer includes safe perpetual deployment; declaring an answer
ends information collection. No extra informative deployment is omitted from E.

Let F_i mean death, an invalid final answer, or nontermination. Work with

    P_i(F_i) <= delta, for every i.

This joint requirement implies both requested separate safety and correctness
bounds. Conversely, two separate delta bounds give joint 2 delta when finite
resolution is also required. The constant factor does not affect normalization
by h=log(1/delta). It does matter for finite-delta claims, so it is not suppressed
in the actual protocol.

T is cost until a declaration or death, and infinity if neither occurs. It is
**resolution cost**, not time to a successful certificate assigned infinity on
death. All lower bounds are valid for infinite expectations as well. Nontermination
is charged to failure; finite expected cost excludes it automatically.

Write D_ij(e)=KL(P_i^e || P_j^e) only when both models regard e as safe.

## 2. Why the quoted classical theorems do not directly settle this

[Nitinawarat, Atia and Veeravalli, Controlled Sensing for Multihypothesis
Testing](https://arxiv.org/pdf/1205.0858), Section II equation (2), assumes finite
second moments of every control's pairwise log likelihood ratio, implying
pairwise absolute continuity. A live-versus-certain-death control violates this.
Its hard risks, Section IV equation (15), concern erroneous declarations, not
probability of ever taking a fatal query. Removing positivity of all KL values
is not the same as permitting singular catastrophic controls.

[Nitinawarat and Veeravalli, Controlled Sensing for Sequential Multihypothesis
Testing with Controlled Markovian Observations and Non-Uniform Control
Cost](https://arxiv.org/pdf/1310.1844), Section 3 equation (3.1), assumes every
controlled transition probability is positive. An absorbing dead state has zero
transition probability back to live and violates that assumption. Its stationary
information/cost characterization therefore cannot simply be invoked unchanged.

These are concrete failed hypotheses of specific theorems, not a proof that the
whole controlled-sensing literature lacks a suitable extension. Small positive
escape probabilities would alter the irreversible-safety problem; a singular
limit would require its own uniform argument. No such limit is justified here.

## 3. Information must be paid before a dangerous query

For fixed i,j, stop a paired transcript immediately **before** the first query
unsafe in either model, or at declaration, whichever occurs first. Include the
selected exit-query label or declaration, but not its execution outcome. Denote
this stopped transcript by W. Before exit, every executed query is safe for
both models, so its log likelihood contribution is finite. Algorithm randomization
and query-selection kernels cancel in the likelihood ratio.

### Lemma 1: pre-exit KL

Under the usual stopped-chain integrability condition (or by localization),

    KL(Law_i(W) || Law_j(W))
      = sum_e E_i[N_e before exit] D_ij(e).

Let B be the event that the exit query is unsafe in j, or the declaration is
invalid in j. Uniform joint validity gives P_j(B)<=delta. If P_i(B)=p>=delta,

    sum_e E_i[N_e before exit] D_ij(e) >= kl(p,delta).

Proof: the chain rule counts only executed common-safe observations. The
predictable exit label contributes no extra conditional KL. Data processing to
B gives kl(p,P_j(B)); monotonicity for P_j(B)<=delta<=p gives the result.
For finite truncations, add a censored label and apply the same argument.

This must not be replaced by the KL of the dangerous query's *outcome*. That KL
can be infinite precisely because the other model dies. It is unavailable when
deciding whether the query can be executed.

### Lemma 2: high-probability gate barrier

Consider the first query unsafe in j, before any death in i. Let L_ij^- be the
log likelihood of its pre-action transcript under i versus j. Until this time
all previously executed queries on such paths are safe for both. For any eta>0,

    P_i(first unsafe-j query is selected,
        L_ij^- < (1-eta) h) <= delta^eta.

The same bound holds for declaring an answer invalid in j, provided no earlier
unsafe-j query was executed, using the terminal likelihood before deployment.

Proof: on the indicated event, dP_i/dP_j < exp((1-eta)h). The event under j is
contained in its lifetime failure event, of probability <=delta. Summing over
all finite stopping times yields delta exp((1-eta)h)=delta^eta. This accounts
for adaptive timing without assigning a new delta to each query.

This is a necessary probabilistic barrier for all valid algorithms. It does not
say every history must obey a deterministic confidence threshold. Algorithms may
spend their small failure probability on exceptional paths.

## 4. A general ordered-elimination lower bound

Fix a true model i. This is a model-aware relaxation, not an executable learner.
Choose an answer a in A(i) and an ordered list of distinct alternatives
j_1,...,j_r. It must include every j for which a is invalid. It may also include
answer-compatible models whose removal unlocks useful queries.

At stage k, before removing j_k, the retained set is

    V_k = H minus {j_1,...,j_(k-1)}.

Choose nonnegative query amounts x_(k,e) only for queries safe in all of V_k.
For every removal impose the accumulated information constraint

    sum_(l<=k) sum_e x_(l,e) D_(i,j_k)(e) >= 1.             (G_k)

All divergences in (G_k) are well-defined because j_k was retained throughout
those stages. Evidence collected in earlier stages is reused. Simultaneous
removals are represented by zero-length later stages. The cost is

    sum_k sum_e d_e x_(k,e).

Define C_i^gate as the minimum of these LP values over all such answers and
ordered lists. An empty list costs zero when a is acceptable for every model;
infeasible lists cost infinity. There are finitely many lists, though exponentially
many. This is a definition for analysis, not a proposed computationally efficient
algorithm.

### Theorem 1: universal asymptotic lower bound

Under Section 1's assumptions, for any sequence of uniformly delta-valid
learners,

    liminf_(delta->0) E_i[T_delta] / log(1/delta) >= C_i^gate.

The lower bound is uniform over learner choices at each delta. Consequently,

    liminf_(delta->0) inf_pi max_i E_i[T_delta] / log(1/delta)
      >= max_i C_i^gate.

### Proof

Fix a finite cost window B h and eta>0. There are at most B h/d_min observations
in this window. For each j, stop its evidence process before the first query
unsafe in i or j. On the stopped process,

    L_ij(n) = sum_(t<=n) D_ij(e_t) + martingale(n).

Bounded log increments give a uniform martingale second-moment bound O(h),
independent of the adaptive policy. Doob's inequality therefore makes the
maximum deviation divided by h vanish in probability, uniformly over all
learners. There are only finitely many j. Together with Lemma 2 and a union
bound, with probability 1-o(1) every first unsafe-j query or still-unexited
invalid-j declaration in the window has accumulated drift at least
(1-eta-o(1))h. Exclude actual failure in i, of probability <=delta.

On any remaining successful path in the window, list models when the first
query unsafe for them is selected. At final declaration a, append any unlisted
models with a not in A(j). Multiple models removed at one query may be ordered
arbitrarily with zero intervening duration. Prior to each removal all executed
queries were safe for all still-retained models. Partition their counts by these
stages. Queries after the last required removal may be omitted. Divide counts
by h(1-eta-o(1)). The result is feasible for the LP corresponding to that list
and answer. Thus the successful path's cost is at least

    h(1-eta-o(1)) C_i^gate.

To obtain a probability lower bound at any finite c<C_i^gate, apply this argument
with B>c and eta sufficiently small. Resolution before c h is impossible outside
the o(1) exceptional set. Hence P_i(T_delta>=c h)->1 and E_i[T_delta]/h>=c-o(1).
Let c increase to C_i^gate, including the infinite case. Uniform concentration
makes the conclusion valid for infima over policy families as well. QED.

### What this does and does not establish

This bound contains a temporal constraint missing from a static allocation:
information obtained after j's removal cannot finance the query that first
required j's removal. A single final occupancy vector discards this ordering.
It also handles overlapping answers through answer-specific invalid alternatives.

When every query is safe in every model, ordering ceases to restrict sampling.
For fixed a the LP reduces to the ordinary constraints
`sum_e x_e D_ij(e)>=1` for every j with a invalid, followed by minimizing over a.
That reduction is a necessary sanity check, not a novelty claim.

The lower bound does not show C_i^gate is always achievable. Its optimizer knows
i. Two models indistinguishable under the current common-safe menu may prefer
different next removals. An actual learner cannot pick the right plan before
obtaining distinguishing observations. An o(h) unrestricted pilot cannot fix
this when the pilot itself needs unsafe queries. Also, proving likelihood
tracking with optional stopping and controlling expected costs on erroneous
branches remains necessary for a general upper bound. We do not call the LP
an optimal complexity characterization.

## 5. An exactly solvable competing-route family

There are three models 0,1,2 and three unit-cost probes p,b,q. Answers are the
three labels, with A(i)={i}. All declarations are physically harmless but wrong
labels count as failure; acceptable deployment may equivalently be an
uninformative safe label-specific decision. There are no hidden informative
queries outside the table.

Fix 0<s<1/4. Define

    epsilon = KL(Ber(1/2-s) || Ber(1/2+s))
            = 2s log((1/2+s)/(1/2-s)),
    I = KL(Ber(1/4) || Ber(3/4)) = (1/2) log 3.

Both Bernoulli pairs are symmetric, so their reverse KL equals the forward KL.

| Model | p (safe for all) | b (safe for all) | q |
|---|---|---|---|
| 0 | Ber(1/2-s) | Ber(1/4) | safe: Ber(1/4) |
| 1 | Ber(1/2+s) | Ber(3/4) | certain catastrophe |
| 2 | Ber(1/2+s) | Ber(1/4) | safe: Ber(3/4) |

Probe p can directly establish answer 0, but is weak. Probe b provides no
information between 0 and 2, but excludes model 1 efficiently and thereby
allows q. There are three hypotheses, not all combinations of two independent
bits. In particular, p couples the answer distinction to the safety distinction
and supplies a competing direct route.

### Theorem 2: sharp pointwise and minimax constants

For fixed s with epsilon<I, define V_i(delta)=inf_pi E_i[T_delta], where pi must
be uniformly delta-valid over all three models. Then

    lim_(delta->0) V_0(delta)/h = min{1/epsilon, 2/I},
    lim_(delta->0) V_2(delta)/h = 2/I,

and

    lim_(delta->0) inf_pi max_i E_i[T_delta]/h = 2/I.

The last equality uses a single common learner, not different learners for
different true models. The infima in the first two displays are pointwise
performance criteria over the same uniform-validity class.

### Lower bound at model 0

Before q is available, p contributes (epsilon,epsilon) information against
(1,2); b contributes (I,0). If the learner reaches answer 0 without q, eliminating
2 requires at least 1/epsilon units in the normalized LP.

For a route using q, let x,y be p,b amounts before model 1 is removed. They obey

    epsilon x + I y >= 1.

If x>=1/epsilon, cost is already at least 1/epsilon. Otherwise, at least
1-epsilon x further evidence against 2 is needed. q's rate I exceeds p's
rate epsilon, while b contributes zero, so even the best continuation costs at
least (1-epsilon x)/I. Total cost is bounded below by

    x + (1-epsilon x)/I + (1-epsilon x)/I
      = x + 2(1-epsilon x)/I.

This affine function on [0,1/epsilon] has minimum min{1/epsilon,2/I}.
Discarding 2 before 1 already costs 1/epsilon and cannot improve the bound.
Theorem 1 gives the lower bound for arbitrary valid learners, not only staged
confidence algorithms.

### Lower bound at model 2

Against model 1, p has zero KL, and only b supplies information before q.
Removing 1 costs b amount at least 1/I. Against model 0, p supplies epsilon and
q supplies I; b supplies zero. Reaching answer 2 therefore costs at least an
additional 1/I when epsilon<I. A no-q route is no cheaper: it needs b amount
1/I and p amount 1/epsilon. Thus C_2^gate=2/I.

### Matching common learner

Use a two-sided likelihood test on b with threshold log(2/delta), comparing
model 1 against the observationally identical block {0,2}.

- If 1 is certified, declare 1.
- Otherwise execute q and perform a two-sided test of 0 versus 2, again with
  threshold log(2/delta). Under true model 1, the first q causes death immediately.

Each erroneous test has probability <=delta/2 by a likelihood martingale. Under
1, death is charged to the first test's erroneous block decision. Under 0 or 2,
both queries are safe and answer errors are covered by the two stage budgets.
Hence the learner is uniformly joint-delta-valid, without regenerating risk.

Bounded overshoots and positive drift give expected test lengths
`(h+O(1))/I`. Under 0 and 2, expected resolution cost is at most
`2h/I+O(1)`; under 1 it is at most `h/I+O(1)` plus the bounded erroneous fatal
query cost. This attains the minimax lower bound and the model-2 bound.

For the other model-0 endpoint, first test p to separate 0 from {1,2}. If it
selects 0, declare 0; otherwise use b to distinguish 1 from 2. Use delta/2 per
test. The algorithm is safe for all models and uniformly correct. Under model 0,
the second stage is reached with probability at most delta/2 and has conditional
mean O(h) using fresh independent samples; its contribution is o(h). The leading
model-0 cost is 1/epsilon. Choosing the better of these two predetermined valid
algorithms proves the model-0 upper bound. QED.

### Exactly which greedy design is arbitrarily worse

At target model 0 and the **current initial common-safe menu** {p,b}, the static
Chernoff max-min design chooses p alone:

    max_(w in [0,1]) min{w epsilon+(1-w) I, w epsilon}
      = epsilon, uniquely at w=1.

Consider the fully specified valid algorithm that pursues this target-0 design
until p resolves 0 versus {1,2}, and then uses the b fallback described above.
Its model-0 leading constant is 1/epsilon. The unlocking learner's is 2/I.
For epsilon<I/2, their ratio is

    I/(2 epsilon) -> infinity as s -> 0.

The order of limits is delta->0 for each fixed s, then s->0. The optimal
unlocking learner first selects a probe with zero minimum current discrimination
rate against the two alternatives, because it buys access to q.

This is an arbitrarily large gap for **this target-specific, current-menu
max-min design**. It is not a theorem about every policy called greedy, mutual
information maximization, posterior sampling, or a continually changing MLE
Chernoff policy. Those algorithms have not been analyzed here.

### A static true-safe rate still misses the cost

At true model 0, a relaxation that grants q immediately treats model 1 as
separated by q's live/dead support and distinguishes 2 at KL I. It suggests a
1/I leading time. The true uniformly safe minimax constant is 2/I, and for
small epsilon model 0 also costs 2/I. The missing 1/I is paid before q can be
used. This relaxation is deliberately **not** a physically admissible learner;
it demonstrates why a static oracle-safe allocation is insufficient, not an
application of a classical finite-KL theorem to singular observations.

## 6. What is still required for the requested general theorem

Theorem 1 is a general necessary bound; Theorem 2 is a sharp special case.
Neither proves a general equality involving arbitrary admissibility graphs and
overlapping answers. The finite LP already exhibits temporal information
constraints, but optimizing it separately under each i can anticipate information
the learner does not possess.

A matching candidate needs a nonanticipative information-control policy: models
with identical currently observable laws must induce the same decisions until
a distinguishing query is actually available and observed. It must also allow
answer selection without identifying every model. Proving compact fluid limits,
achievability, expected-cost control on error paths, and a uniform minimax
interchange cannot be replaced by merely writing a Bellman equation.

Specific next proof obligations, not new experiment authorization:

1. Determine whether max_i C_i^gate is tight or exhibit a nonanticipativity gap.
2. If it is not tight, define the shared information-policy object and prove its
   lower bound and implementable upper bound rather than adding heuristic terms.
3. Compare that quantitative statement with prior constrained controlled-sensing
   and safe BAI results. The use of likelihood martingales, change of measure,
   SPRT or multiple answers supplies no novelty by itself.

The supplied classical results fail explicit assumptions in our absorbing
setting, so the proposed immediate-transfer kill test has **not** succeeded.
No exhaustive literature exclusion has been established. General sharp
complexity remains open; the project is not upgraded to Level 4 on these proofs.
