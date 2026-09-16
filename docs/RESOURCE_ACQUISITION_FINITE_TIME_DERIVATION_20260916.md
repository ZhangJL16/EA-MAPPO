# Derivation Package: Finite-Time Acquisition Cost of the Frozen Learner

Isolated continuation of the referenced path-catalogue proof; previous derivation
packages and the frozen scientific source closure are not rewritten.

## Target

Derive an explicit expected-regret upper bound for the ALREADY FROZEN finite-
hypothesis path-catalogue learner. Separate allocation, all-path coverage,
likelihood certification and unfinished-excursion costs. Determine what this
bound can honestly say about the negative T=4096 calibration.

This is a proposition/derivation, not an algorithm change or new experiment.
The central object is calendar-time expected regret, not allocation-vector
distance or a single observed pseudo-regret. Do not assume an additive constant
K_B exists just because the leading coefficient is asymptotically correct.

## Status

**COHERENT AFTER REFRAMING.** The original requested form C_B log T + K_B is
not established by the existing theorem. A complete counting derivation below
instead gives explicit time-dependent sublogarithmic remainders for the frozen
procedure. Its probability input is stated and bounded separately. This is a
handwritten derivation; independent proof audit remains open.

No assertion is made that a constant remainder is mathematically impossible;
the derived upper bound is insufficient to establish it. No submission-level
novelty is claimed for unrolling this proof.

## Invariant Object

At a fixed true hypothesis theta and capacity B, let

    Reg_theta(T) = T rho_theta - E_theta[sum_{t<T} realized reward_t].

For a complete safe committed path p, define

    c_p = rho_theta ell_p - R_theta(p) >= 0.

This equals its deterministic conditional-mean pseudo-regret. Thus

    Reg_theta(T) = E_theta[sum_{completed paths} c_p] + E_theta[partial cost].

**Identity.** Because all durations/motion are known, path commitment precedes
its rewards, and primitive reward expectations agree with their public row laws,
the compensator gives this expected-regret identity. The partial cost can be
negative; its absolute value is at most the maximum path length L.

## Assumptions

Use the exact assumptions/learner of
`docs/RESOURCE_PATH_CATALOGUE_ATTAINABILITY_20260916.md`:

- Fixed B; finite known safe committed catalogue P, size M, maximum length L.
- Finite known observable hypothesis class, size K, common finite positive
  reward supports; stationary IID primitive row streams, shared laws allowed.
- Conditional means and realized primitive rewards in [0,1]. Known deterministic
  zero-reward rows and known common-law dock rows carry no discrimination KL.
- Unique optimal path type p_star; scalar gain is its reward rate.
- One initialization play of every path. Certificate first, then least-Q forced
  sampling, then buffered true/MLE LP quota, then free-information quota.
- Q counts only exploration paths, j=sum Q; eta_j=(j+1)^(-1/8).
- The original exploration-count concentration event E_j and false-certificate
  martingale construction are used; their finite constants are made explicit below.

No new mixing, safety knowledge, alternate confidence rule, selected seed, or
horizon is assumed. These bounds are not uniform over unbounded B/classes unless
their displayed constants are controlled uniformly.

## Notation

- lambda_p: a fixed true-model nonoptimal LP optimizer; lambda_p_star=0.
- C = sum_p c_p lambda_p, the same existing information lower-bound coefficient.
- S_c = sum_p c_p; c_max = max_p c_p; Lambda = sum_p lambda_p.
- kappa: existing true-model free-information quota coefficient, zero if absent.
- f(T)=log(T+3)+2log log(T+3)+log(K+1); all logarithms natural.
- J_T: complete exploration plays by T, including initialization.
- B_T: exploration indices with the concentration event E_j failing.
- b >= sup_T E B_T: finite concentration-error budget described in Step 5.
- s >= expected number of wrongly certified departures at all times.
- A_1 = 2 Lambda + kappa.
- V(T)=6M+M^2+2 A_1 f(T)+2b.
- A_eta = sum_{p != p_star} c_p lambda_p^(7/8), with 0^(7/8)=0.

All constants may depend on theta/catalogue/B, but not T. Actual observed
Q/path counts are not substituted for E J_T in a probability bound.

## Derivation Strategy

Use the same existing rule, not an outcome-adjusted replacement. Count each
kind of complete path, sharpen the buffer counting without taking epsilon to
zero, bound the exploration count, then return to calendar regret.

## Derivation Map

1. Complete-path accounting reduces expected regret to counts plus one remainder.
2. Forced plays are bounded by sqrt(J_T)+1 for EACH path.
3. LP quota plays are bounded by F_p+F_p^(7/8)+1, F_p=lambda_p f(T).
4. A coarse exploration bound controls E sqrt(J_T).
5. Explicit finite-support concentration budgets bound bad allocations and
   false certificates. No observed-data extrapolation is used.
6. Combine the bounds. State precisely why this does not yet guarantee a positive
   finite-budget capacity contrast.

## Main Derivation

### Step 1 — forced count [proposition / deterministic counting]

Let F_p(T) count forced plays of p. Before any such play its current exploration
count satisfies Q_p < sqrt(j) <= sqrt(J_T). Counts only increase, irrespective
of other reasons for selecting p. Therefore

    F_p(T) <= sqrt(J_T)+1,
    forced complete cost <= S_c [sqrt(J_T)+1].                 (1)

This term is not an additive T-independent constant in this bound. It covers
even zero-LP-allocation paths; dock/empty/B-only are examples at primary truth.

### Step 2 — buffered quota count [proposition / deterministic counting]

For a correctly estimated model's nonoptimal information-quota play, write
n=Q_p before that play, j>=n and F_p=lambda_p f(T). The rule implies

    n < [1+(j+1)^(-1/8)] F_p
      <= [1+(n+1)^(-1/8)] F_p.                               (2)

If n<F_p, the new count is at most F_p+1. If n>=F_p>0, then
(n+1)^(-1/8)<=F_p^(-1/8), so the new count is at most
F_p+F_p^(7/8)+1. If lambda_p=0 there are no such quota plays.

Other play types only increase Q_p, hence cannot create additional quota plays
below the same threshold. The number U_p(T) of correct-model quota plays obeys

    U_p(T) <= lambda_p f(T)+[lambda_p f(T)]^(7/8)+1.           (3)

Consequently their complete cost is at most

    C f(T) + A_eta f(T)^(7/8) + S_c.                        (4)

The f^(7/8) term is a proven upper envelope, NOT a proved lower bound on the
algorithm's actual overhead. It does not justify claiming unavoidable hardness.

### Step 3 — complete exploration count [proposition]

For a coarse count use eta<=1, giving at most 2lambda_p f(T)+1 correct
information-quota plays per nonoptimal path. Correct free-information plays
number at most kappa f(T)+1. Forced plays total at most M(sqrt(J_T)+1).
Initialization contributes M. Wrong-MLE quota/free-information/fallback plays
outside these categories require E_j failing, by the existing quota-certificate
argument, and contribute at most B_T. Double-counting bad forced plays is safe
for this upper bound. Thus

    J_T <= 3M + A_1 f(T) + M sqrt(J_T) + B_T.                (5)

Using M sqrt(J_T)<=J_T/2+M^2/2 gives

    J_T <= 6M+M^2+2 A_1 f(T)+2B_T,
    E J_T <= V(T),   E sqrt(J_T) <= sqrt(V(T)).              (6)

The last inequality is Jensen, not plugging an empirical J into an expectation.

### Step 4 — explicit upper bound [proposition]

Initialize complete cost S_c. Correct free-information and correctly certified
paths cost zero at truth. Bad-query cost is at most c_max B_T; wrongly certified
paths cost at most c_max times their number. Add (1), (4) and the partial bound L:

    Reg_theta(T)
      <= C f(T) + A_eta f(T)^(7/8)
         + S_c sqrt(V(T)) + 3S_c + c_max(b+s) + L.           (7)

This bound is valid also before initialization is finished, since incomplete
initialization can only remove some nonnegative complete-path terms.

Equivalently, write the explicit correction

    R_theta(T) = C[f(T)-log T] + A_eta f(T)^(7/8)
                 + S_c sqrt(V(T)) + 3S_c + c_max(b+s) + L

for T>=2. Then

    Reg_theta(T) <= C log T + R_theta(T),
    R_theta(T) = O(f(T)^(7/8)+sqrt(f(T))+log log T+1).        (8)

The expression is time dependent. The original limit theorem remains consistent:
all these terms are o(log T). That limit alone cannot strengthen (8) to a
constant-remainder result.

### Step 5 — finite, explicit error budgets [probability input]

For completeness, the original concentration argument can carry an explicit
series rather than an unnamed constant. Work with shared primitive row laws,
not duplicated battery-level copies. Let D be the number of unequal row/ordered
model pairs, I_min their minimum positive row KL, and W the maximum range of
the row log-likelihood-ratio increment. With a=I_min^2/(8W^2)>0, Hoeffding gives
for any such pair at sample count n:

    Pr(|empirical row logLR - row KL| > eta_j row KL/4)
      <= 2 exp(-a eta_j^2 n).

Use the documented forced-sampling minimum-row bound
g(j)=max(1,floor(sqrt(j)-M-2)) after full initialization. Union over D pairs
and ALL n>=g(j), not a fixed n substituted at a stopping time, gives

    Pr(E_j fails) <= min(1,
       2D exp[-a eta_j^2 g(j)]/[1-exp(-a eta_j^2)]).

One explicit valid choice is

    b = M + sum_{j=M}^infinity min(1,
        2D exp[-a eta_j^2 g(j)]/[1-exp(-a eta_j^2)]).         (9)

The finite M allowance handles initialization/early indices. Since eta_j^2
g(j) grows like j^(1/4), this series is finite; its constants can nonetheless
be enormous. It is not estimated from the successful likelihood of one seed.
For D=0 the class is observationally identical and has one optimal decision;
use the known optimal policy separately rather than dividing by W=0.

For false certificates the original finite-hypothesis likelihood martingale
bound gives, at deterministic primitive time t, probability at most
(K-1)exp[-f(t)]. The possible charger departures form a subset of these times:

    s <= (K-1)/(K+1) sum_{t=0}^infinity
                    1/[(t+3)log^2(t+3)]
      <= (K-1)/(K+1)[1/(3log^2 3)+1/log 3].                (10)

No new concentration claim about path-adaptive observations or unknown
transitions is being introduced. The original minimum-row-count lemma and
martingale hypotheses must survive independent audit for (7) to inherit them.

### Step 6 — calibration interpretation [identity + interpretation]

At primary truth, S_c is .50 in split and .55 in high/on, while M is4 versus5.
Both lambda-positive coordinates have value .377359. Thus the new AB reduces
C but adds a covered path and increases the wrapper's S_c/M-dependent bound.
The error budget (9) also depends on M through g(j). It is incorrect to claim
capacity only changes C and never the finite-time overhead.

Existing trace accounting, not a new experiment:

    split P = 4.00 complete cost + .10 partial cost = 4.10;
    high/on P = 4.40 complete cost - .10 partial cost = 4.30.

Every dock/empty/B count stays8 and AB adds8. This identifies the actual source
of the observed difference. Bound (7) is an expectation upper bound and can be
very loose; it is not a fitted explanation, an early-time prediction, or proof
that the observed difference has the same expectation across seeds.

### Step 7 — visibility needs two-sided control [proposition / algebra]

Comparing two regret UPPER BOUNDS cannot prove the high-capacity algorithm has
lower actual regret. Suppose separate valid finite-time inequalities exist:

    Reg_low(T) >= C_low log T - L_low(T),
    Reg_high(T) <= C_high log T + U_high(T).

Only then does

    Reg_low(T)-Reg_high(T)
      >= (C_low-C_high)log T - L_low(T)-U_high(T)             (11)

give a sufficient expected-benefit condition. A positive RHS certifies direction.
The existing ASYMPTOTIC lower bound supplies no usable explicit L_low(T), and
the current calibration is only one realization. Therefore this derivation
does not claim a certified crossover horizon or finite-time visibility theorem.

## Remarks and Interpretation

- A completed result is (7), not a new fictitious K_B theorem. Its explicit
  dependencies show where capacity can improve acquisition and worsen coverage.
- Calling the wrapper remainder O(1) would erase the very schedule being audited.
  Even C[f-log T] contains log log T in this proof's confidence rule.
- A public-class informative-channel coverage basis is a plausible algorithmic
  design change, but replacing Q_path forcing changes the proof. It does not
  inherit (5), g(j), or (7) automatically. No revised procedure is implemented.
- Reducing S_c/M or improving an upper bound is not itself proof of a finite-
  horizon gap, novelty or embodied relevance.
- The existing structured-bandit prior art already studies low-regret exploration
  under structured observations; no general claim that standard methods ignore
  physical acquisition is justified by this local derivation.
  Source checked: Combes, Magureanu and Proutiere (2017), official NeurIPS abstract,
  https://proceedings.neurips.cc/paper/2017/hash/e19347e1c3ca0c0b97de5fb3b690855a-Abstract.html.
  This source check establishes prior-art context, not proof or novelty verification.

## Boundaries and Non-Claims

This is an explicit unrolling of the existing fixed-class procedure, not a
capacity-uniform minimax characterization, matching finite-time lower bound,
new algorithm or empirical validation. No model/environment/source registry or
trajectory was edited; no seed/horizon/route/library was added.

Simply appending (7) to a paper cannot establish a new learning principle.
A substantive paper-level target would require sharp two-sided finite-budget
acquisition/certification costs for the SAME class, or a guarantee on the
paired regret difference. Until that is achieved, claim only the proved upper
envelope and observed overhead accounting, not "when physical resource becomes
statistical efficiency" as a finished theorem.

## Open Risks

1. Independent audit of the original minimum-row-count and certificate lemmas.
2. Tightness: neither f^(7/8) nor sqrt(f) is shown necessary here.
3. The concentration constant b can dominate; (7) need not be numerically useful.
4. Finite-time decision-derived lower/coupled contrast required for (11).
5. Public-class channel coverage changes need a new proof and distinct lineage.
6. Novelty beyond classical structured allocation/counting remains unestablished.
