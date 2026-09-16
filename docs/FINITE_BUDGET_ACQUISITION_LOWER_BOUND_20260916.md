# Derivation Package: Finite-Budget Acquisition Lower Bound

## Target

Test whether the existing fixed resource-path class supports an unavoidable
positive overhead beyond C_B(theta) log T. Do not alter the frozen learner,
calibration, hypotheses, or runtime sources. Derive a finite-time information
requirement from control performance rather than prescribing a testing task.

## Status

COHERENT AFTER REFRAMING / EXTRA ASSUMPTION.

The proposed universal additive-overhead theorem does not survive unchanged.
Asymptotic uniform efficiency does not impose a finite-time performance envelope;
certification cost also cannot be added to the allocation cost without proving
that it is a separate necessary expense. Below is an explicit finite-budget
necessary allocation bound under a stated performance envelope. It is a
handwritten derivation, not an independently audited attainability theorem or
a novelty certificate. No new experiments or learner changes occurred.

Subsequent resolution: [Strict finite-budget capacity benefit](STRICT_FINITE_BUDGET_CAPACITY_BENEFIT_20260916.md)
proves the minimax fixed-point implication, shows this particular F/frozen-upper
comparison cannot separate here, and supplies exact finite-game certificates
for strict benefit on the SAME calibration class at integer T=12..24. This does
not alter the F bound below or the negative frozen-learner T=4096 outcome.

## Invariant Object

For ONE unknown-model safe learner A on the same fixed class Theta, use

    R_theta(T) = T rho_theta - E_theta[sum_{t<T} reward_t].

All comparisons of models use this same learner, not separately optimized
model-specific policies. For each capacity, the learner must satisfy the stated
performance envelope at EVERY hypothesis at the specified horizon.

## Assumptions

Inherit the finite committed-path subclass of
RESOURCE_PATH_CATALOGUE_ATTAINABILITY_20260916.md: known deterministic dynamics,
known counter/reload rules, finite safe catalogue P_B, positive finite reward
supports, unique optimal path type p_theta, maximum duration L=L_B. Paths are
committed before their rewards, but choices between paths are adaptive.
All realized primitive rewards lie in [0,1]. Start at the charger.

Fix integer T>2L. Additional FINITE-TIME premise, not a consequence of uniform
efficiency: R_v(T)<=r_v(T) for each hypothesis v, with specified nonnegative
numbers r_v(T). A common envelope r(T) is a special case. Interesting bounds
require these numbers to be small compared with T times the relevant gaps.

## Notation

- c_theta(p)=rho_theta ell_p-R_theta(p), nonnegative complete-path cost.
- j_theta,v(p)=sum_i m_p(i) KL(nu_theta,i || nu_v,i).
- N_p: number of completed plays of p by T; n_p=E_theta N_p.
- Delta_theta=min_{p!=p_theta} c_theta(p)/ell_p>0, if nonoptimal paths exist.
- For an alternative v with p_v!=p_theta, delta_v=c_v(p_theta)/ell_p_theta>0.
- I_max(theta,v)=max_i KL(nu_theta,i || nu_v,i), finite.
- A_theta: confusing alternatives with p_v!=p_theta and j_theta,v(p_theta)=0.
- kl(a,b): binary relative entropy; all logarithms are natural.

If there is only one path, no different-optimum alternative exists; skip the
gap construction and use the empty-constraint bound. L, gaps and information
constants can depend on B; no uniform capacity claim is made without bounds.

## Derivation Strategy

Low regret -> high occupancy of the true optimal path -> a decision test with
explicit errors -> transcript KL -> a time-budgeted safe acquisition program.
Then inspect whether this program implies a positive additive correction to
the asymptotic coefficient. No asymptotic approximation enters the finite bound.

## Derivation Map

1. Compensator and committed-path decomposition bound complete-path costs.
2. Completed optimal-path duration defines a measurable test event.
3. Markov and binary entropy yield explicit finite-time KL requirements.
4. The last incomplete path costs at most L and carries at most L I_max KL.
5. Expected completed-path counts satisfy duration and information constraints.
6. These constraints price TOTAL acquisition, not a disjoint extra expense.

## Main Derivation

### 1. Complete-path cost [identity and bound]

The predictable reward compensator gives

    R_theta(T)=sum_p n_p c_theta(p)+E_theta[partial centered cost],
    |partial centered cost|<=L.

Consequently sum_p n_p c_theta(p)<=r_theta(T)+L. The same statement holds
under v with expectations taken under v. No conditioning on a reward-selected
path is used to replace its realized rewards by its expected path reward.

### 2. Low regret creates a decision test [proposition]

Let Z be the completed duration spent on p_theta and E={Z>=T/2}. Completed
paths cover at least T-L primitive steps. On E^c, completed nonoptimal duration
under theta is greater than T/2-L. Therefore

    P_theta(E^c) <= e_theta := (r_theta(T)+L)/(Delta_theta (T/2-L)).

Under alternative v, each complete p_theta play costs delta_v per unit duration.
On E its complete cost is at least delta_v T/2, giving

    P_v(E) <= q_v := 2(r_v(T)+L)/(delta_v T).

Use the informative case e_theta<1 and q_v<1; otherwise assign requirement
zero to that alternative. This event distinguishes decisions, not all parameters.

### 3. Explicit transcript information [proposition]

Set p=P_theta(E), q=P_v(E). Data processing gives transcript KL>=kl(p,q).
This follows directly by conditioning the likelihood ratio on E and E^c and
applying convexity on each part. Binary entropy is at most log 2, so

    kl(p,q) >= p log(1/q)-log 2
            >= (1-e_theta) log(1/q_v)-log 2.

Thus a valid nonnegative requirement is

    k_v(T;r)=[(1-e_theta) log(1/q_v)-log 2]_+.

Assign k_v=0 in the uninformative cases above. The conditional KL chain rule
counts primitive-row information under theta even for adaptive path choices.
Completed paths contribute sum_p n_p j_theta,v(p). The incomplete path has
at most L rows, hence contributes at most L I_max(theta,v). Therefore

    sum_p n_p j_theta,v(p) >= a_v(T;r),
    a_v(T;r)=[k_v(T;r)-L I_max(theta,v)]_+.

The upper bound on incomplete-path information is conservative. It is not a
claim that all that information can actually be acquired for free.

### 4. Finite-budget acquisition lower bound [proposition]

Define the following deterministic relaxation, using all different-optimum
alternatives v and their a_v (one may restrict to A_theta for a weaker bound):

    F_B,theta(T;r) = inf_{n_p>=0} sum_p n_p c_theta(p)
    subject to sum_p n_p ell_p <= T,
               sum_p n_p j_theta,v(p) >= a_v(T;r) for all v.

The actual expected completed counts are feasible. Hence EVERY learner
satisfying the finite-time class-wide envelope obeys

    R_theta(T) >= F_B,theta(T;r)-L.                         (1)

If this relaxed program is infeasible, no learner can satisfy the proposed
envelope at this horizon. Feasibility is not sufficient: different theta-specific
allocations need not be simultaneously attainable by one unknown-model learner.

For confusing alternatives, the optimal path contributes zero KL. Omitting
the duration constraint and imposing a common requirement a<=min_v a_v yields,
by positive homogeneity of the existing allocation program,

    R_theta(T) >= a C_B(theta)-L.                          (2)

This is a finite-time bound derived from regret. It is not C_B log T plus a
positive remainder. For r_v(T)=O(log T), its information requirement is of the
form log T-log log T plus gap/boundary constants, not log T plus an extra
positive coverage penalty. This observation describes the bound, not the exact
optimal finite-time regret.

### 5. Why the proposed universal positive remainder fails [proposition]

Uniform efficiency only constrains limits. Fix any finite T_0 and true hypothesis
theta_0. An algorithm can repeat the PUBLIC path p_theta_0 through the first
charger arrival at or after T_0, then start any uniformly efficient learner
from scratch. It does not read private truth; p_theta_0 is merely a hardcoded
guess. Its fixed finite initial prefix preserves uniform efficiency on every
hypothesis. At theta_0, at horizons that are multiples of ell_p_theta_0 within
that prefix, its expected regret is exactly zero.

Consequently, if C_B(theta_0)>0, a positive pointwise lower bound of the form
C_B(theta_0) log T+Omega(Gamma_B(T)) cannot hold for all uniformly efficient
learners at all specified finite horizons. This argument does NOT refute an
eventual learner-dependent asymptotic refinement or a finite-time minimax bound.
Those are different quantified statements and require different premises.

### 6. Certification is a total cost, not automatically an extra cost [proposition]

Suppose instead an algorithm stops at a charger and outputs the optimal path
with error at most delta<1/2 at EVERY hypothesis, with finite expected duration.
For E={output p_theta}, p>=1-delta and q<=delta under a different-optimum v.
The stopped transcript therefore needs KL>=kl(1-delta,delta). For committed
paths, its expected acquisition counts must satisfy

    sum_p n_p j_theta,v(p)>=kl(1-delta,delta).

Pricing these counts by sum_p n_p c_theta(p) gives a lower bound on the TOTAL
centered acquisition cost. For confusing alternatives without a duration
constraint, it is kl(1-delta,delta) C_B(theta). At delta of order 1/T it already
has leading term C_B(theta) log T. Defining Gamma as this full certification
cost and adding it to C_B log T counts the same necessary information twice.

A separate positive overhead requires an additional proved bottleneck not
already priced by these counts. Neither all-path forcing nor eta buffering is
such a problem-imposed obligation in the current model.

### 7. Finite-time simulation across capacities [proposition]

Let B_2>=B_1. Assume every old committed path remains feasible, has the SAME
duration and primitive feedback law, and rho_B2,theta=rho_B1,theta for EVERY
theta in the fixed class. Define the finite-time minimax regret over all safe
unknown-model committed-path learners by

    Rstar_B(T)=inf_A sup_theta R_B,theta^A(T).

For any learner A_1 at B_1, construct A_2 at B_2 with a virtual counter of
capacity B_1. Feed A_1 its virtual battery and the actual primitive rewards,
execute precisely its old committed paths, and reset the virtual budget at
each charger. The actual battery is never below the virtual budget before
recharge; its excess cannot invalidate any debit. Since reward laws and path
durations do not depend on the actual capacity, the primitive transcript supplied
to A_1 has exactly its B_1 law at EVERY theta. Use the same internal randomization.

Equal gains and identical reward transcripts imply, for every integer T,

    R_B2,theta^A2(T)=R_B1,theta^A1(T),
    Rstar_B2(T)<=Rstar_B1(T).                              (3)

No asymptotic limit, guessed truth or allocation oracle is used. Any class-wide
finite-time risk guarantee achievable at B_1 transfers to B_2 without extra
cost. The same construction transfers a uniformly efficient algorithm if that
restriction is imposed on both infima. It does not construct a reverse reduction
or prove strict improvement. If an extra path improves gain at ANY hypothesis,
this equal-regret statement no longer follows: the comparator difference must
be added, even if the gain at the calibration's true hypothesis is unchanged.

Thus, for the current unchanged-execution calibration, mandatory positive
NEW-PATH coverage overhead cannot be a fundamental price of increased capacity.
An improved frontier can still require a difficult identification tradeoff to
exploit; that is not the same assertion as unavoidable coverage of every path.

## Remarks and Interpretation

The useful result is (1): finite-budget control requirements imply a
capacity-dependent safe acquisition relaxation. It simultaneously includes
calendar duration, physical path cost, shared information and decision gaps.
It can rule out an asserted finite-time performance envelope without analyzing
the frozen exploration wrapper. It does not establish a new learning principle
or matching finite-time resource tradeoff by itself.

The current calibration cannot convert its measured forced-coverage cost into
an unavoidable lower bound. A valid finite-time capacity comparison still needs
a common risk/envelope definition, a lower bound at one capacity and an
attainable upper at the other. Two lower bounds, or two upper bounds, do not
prove the sign of the regret difference. More feasible committed paths can
always be ignored by a learner at a larger capacity; if comparators are unchanged
for EVERY hypothesis and old paths/feedback are preserved, extra catalogue
coverage is not mandatory for that learner. This statement does not cover a
capacity change that improves the comparator.

## Boundaries and Non-Claims

- No claim R=C log T+Gamma+o(...), Gamma>0, or necessary sqrt(log T) cost.
- No matching finite-time upper, minimax equality or strict positive capacity
  contrast. Equation (3) is a one-way minimax comparison under explicit nested
  experiment and all-hypothesis comparator assumptions.
- Fixed committed-path subclass, not general within-excursion adaptive control.
- No claim the previous upper envelope exactly explains expected performance;
  transcript accounting explains the observed frozen trajectory only.
- No new sampling, retuning, seeds, horizons, routing library or runtime edits.

## Open Risks

Independent audit of the finite-time event, count decomposition, stopped KL
conditions and boundary constants remains required. The stopped certification
claim uses common bounded likelihood increments and finite expected stopping
duration, so optional stopping cannot be invoked without those premises.
The relaxation can be loose through continuous expected allocations, conservative
partial-information credit, and lack of a single-learner attainability constraint.
Novelty against finite-time structured experiment design is not established.
These limitations prevent presenting this derivation as the missing paper core.
