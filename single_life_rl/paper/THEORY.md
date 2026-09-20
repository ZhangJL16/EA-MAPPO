# Safe Identifiability: When Is Reset-Free Reinforcement Learning Possible?

Pre-results manuscript skeleton and proofs, protocol v1. This is a research draft,
not a novelty claim or a proved characterization for arbitrary unknown MDPs.

## 1. Introduction

Informative actions can irreversibly end a learning trajectory. Physical
regeneration restores operational resources, but does not restore the statistical
risk budget. Distinguish uniform probability-one safety from uniform lifetime
probabilistic safety; their learnability boundaries are different even with two
known candidate models and one unknown bit.

## 2. Regenerative Single-Life Learning

For all environment, observation and algorithm randomness, strict safety is
sup_M P_M(tau_C < infinity)=0. Learning uses
sup_M P_M(tau_C < infinity)<=delta, with primary delta=.05. This is the entire
learning-and-deployment life, not per cycle. Set delta_epi=delta and delta_alea=0.

**Finite experiment abstraction used below.** There are finitely many candidate
models, experiments and deployment answers. The learner knows their conditional
laws but not the true model. Each safe experiment returns to the same operational
regenerative state (or a model-independent reusable start distribution) in bounded
positive time. Each unsafe experiment causes catastrophe with probability one.
Successive observations are conditionally independent given model and experiments.
The full observation includes reward, duration and every visible sensor. A timeout
outside the regenerative set is not a safe experiment. There is no free reset
within a life. The admissible menu must include every interaction the learner may
use to gather information, including informative exploitation.

An acceptable answer a belongs to A(M) if its indefinitely repeated deployment is
safe in M and epsilon-optimal for the specified regenerative objective. Answers
and acceptability are known given M; each A(M) is nonempty. For the main categorical
characterization assume a finite observation alphabet; bounded likelihood ratios
on common support suffice for the quantitative finite-mean claims. The Gaussian
sensor plus exact UAV clock is treated separately by support filtering.

Call a learner delta-successful if, uniformly over M, with probability >=1-delta
it avoids catastrophe for its entire life, stops with a certificate in finite time,
and deploys an answer in A(M). Safety alone is weaker than this joint requirement.

## 3. Strict-safety impossibility

**Theorem 1 (finite-history absolute-continuity obstruction).** In the binary fork,
if 0<=kappa<1/2 and common-safe observation laws have full support, no uniformly
zero-catastrophe learner ever selects either specialized action at a finite time.
If common-safe rewards are at most .2/time and the specialized oracle earns 1/time,
its regret is at least .8T.

**Proof.** Before the first specialized action, finite histories are mutually
absolutely continuous under the two models, including algorithm randomness.
A positive probability of selecting a_+ at some time under M_+ would therefore
imply positive probability under M_-, where it is catastrophic. Uniform zero
catastrophe forbids this. Apply the same argument to a_- and take a countable
union over finite times. Rewards then give the regret bound. QED.

Merely sharing some support is not enough for this theorem: exclusive outcomes
can provide zero-error evidence. Infinite-history singularity/consistent estimation
does not furnish a finite-time zero-error unlock when finite histories are equivalent.

## 4. Observational refinement and learnability

For V, define E_safe(V) as experiments safe for every model in V. Partition V by
equality of the complete laws under all experiments in E_safe(V). Recursively
partition each proper block, stopping when the partition has only one block.
Finiteness implies termination in at most K-1 strict refinements along a path.
This is a structural tree of law distinctions, not an assertion that finite noisy
samples determine a child block without error.

**Theorem 2 (arbitrarily high-confidence learnability in the finite experiment
abstraction).** There exists a delta-successful learner for every delta>0 if and
only if each terminal block B admits a common acceptable answer:
intersection_{M in B} A(M) is nonempty.

**Sufficiency.** At a node V, cycle through all its common-safe experiments while
maintaining model likelihoods. Use pairwise likelihood-ratio rejection with a
preassigned stage budget eta_l, sum_l eta_l<=delta. Under a true model M, each
alternative/true likelihood ratio is a nonnegative martingale (supermartingale
with singular supports) for adaptive predictable sampling; Ville's inequality
bounds the probability that any alternative crosses (|V|-1)/eta_l against M by
eta_l. Hence the true model is retained simultaneously at all times and stages
with probability >=1-delta. Distinct blocks differ on at least one repeatedly
sampled experiment. The strong law for its log-likelihood ratio (or exclusive
support observation) eventually eliminates all models outside the true block.
Enter the unique surviving block, then repeat. There are finitely many stages.
At a terminal block choose a common answer. All experiments and deployment are
safe on the same coverage event. Finite expected stage resolution time additionally
requires separation and appropriate likelihood moment bounds; almost-sure finite
termination on the coverage event is the characterization claim. QED.

**Necessity.** Let B be a terminal block without a common answer. Every experiment
common-safe for B has exactly the same observation law for all M in B. Couple
these common histories using one reference law, until the learner first selects
an experiment unsafe in at least one member of B. For each such exit path, at
least one member of B catastrophically fails with the same pre-exit path mass.
For paths with no exit and a finite recommendation, at least one member regards
the recommendation as unacceptable. Paths with neither exit nor recommendation
fail finite certification for every member. Summing failure probabilities over
M in B is therefore at least one. Some M has joint failure probability at least
1/|B|. Thus delta-success for arbitrarily small delta is impossible. QED.

This theorem is restricted to the declared experiment/answer library, known
model-conditional deterministic safety, reusable regenerative starts and complete
observation laws. It is not maximal viability in the original UAV simulator, and
is not a fixed-delta iff for every arbitrarily large model class. Novelty relative
to existing controlled testing/safe-exploration theory remains to be established.

Pairwise acceptable-answer overlap does NOT replace common intersection. The
sets {a,b},{b,c},{a,c} provide a counterexample. A general information design must
be answer-specific when acceptable answers overlap.

## 5. Lifetime safety and the implemented confidence sequence

**Proposition 3 (lifetime safety).** If P_M(M in V_t for all t)>=1-delta uniformly
in M, and each selected experiment and eventual deployment are safe for all
members of V_t, then sup_M P_M(tau_C<infinity)<=delta.

**Proof.** On the simultaneous coverage event every chosen action is safe for the
true model, including deployment. Catastrophe is a subset of its complement. QED.
The budget is never renewed at regeneration.

For bit j with p_+=.5+k and p_-=.5-k, use
L_j(n)=sum_{i<=n}(2Y_i-1) log(p_+/p_-). Certify 1 at L_j>=h_j and 0 at L_j<=-h_j,
h_j=log(1/alpha_j). Under the opposite true bit exp(+/-L_j) is a martingale.
Ville bounds the probability of a wrong crossing, over all n, by alpha_j.
Use alpha_0=delta in binary. Relevant tree bit j receives delta/2^(j+1).
In the nuisance suite the two relevant bits receive delta/2 and delta/4,
and each of m nuisance bits receives delta/(4m). Their sum is at most delta,
including all 34 possible relevant/nuisance coordinates. This avoids making
later nuisance bits artificially expensive through geometrically tiny budgets. Coordinates are
never reinitialized. In conditional tree experiments, argue until the first
incorrect certified ancestor: before that event only the correct-prefix channel
is sampled, and its likelihood martingale is valid. A wrong ancestor is already
charged to the same union of error events. This covers adaptive experiment counts.

The method exploits known product structure to represent 2^34 nuisance models
exactly. It does not enumerate billions of models or claim general factored-MDP
tractability. A float likelihood underflow is never evidence of impossible support.

## 6. Information limits and a matching witness-family bound

The original all-decision-pair Gamma at the recursive root is zero, because models
00 and 01 require different answers but p0 cannot distinguish them. It is not the
quantity plotted. Use current cross-block information for the implemented menu.

For symmetric Bernoulli separation,
I_j=KL(Ber(.5+k_j)||Ber(.5-k_j))=2k_j log((.5+k_j)/(.5-k_j)).
For independent single-bit probes currently available, the hardest block pair can
differ in only one bit. Maximizing min_j w_j I_j / sum_j w_j d_j gives
w_j proportional to 1/I_j and Gamma_stage=1/sum_j(d_j/I_j).
The implementation tracks these proportions. In the recursive tree only the
next path probe is informative and legal, so Gamma_stage=I_j/d_j.
Nuisance bits are not answer-relevant until FullModelID alone requires singleton
certification; both algorithms use the same relevant-first design ordering.

**Proposition 4 (binary identification lower bound).** A sequential test using
the safe binary probe, with error <=delta under each hypothesis and almost-sure
finite decision time N, satisfies
E_M[N] I >= kl(1-delta,delta).

**Proof.** The expected stopped log-likelihood equals E_M[N]I (localize stopping
and use standard integrability conditions). Data processing to the final binary
recommendation gives at least kl(1-delta,delta). Multiply by d for time. QED.
This is a testing lower bound, not a claim that every policy with allowed risky
shortcuts is forced to follow this menu.

**Proposition 5 (implemented binary upper bound).** Let a=log(p_+/p_-) and use
threshold h=log(1/alpha). The two-sided likelihood test resolves a bit with error
<=alpha and E[N] <= (h+a)/I for k>0.

**Proof.** The drift of the true-direction log-likelihood is I>0. Two-sided stopping
occurs no later than its first passage of +h. The latter has bounded increments,
finite mean, and overshoot at most a. Wald gives I E[N_+]=E[L(N_+)]<=h+a. The
likelihood martingale gives the error bound. QED.

For a depth-L independent staged identification family, with a required certified
branch choice before accessing the next probe, preassigned alpha_j as above gives
expected time to test resolution/termination bounded by
sum_j d_j(log(1/alpha_j)+a_j)/I_j, plus at most one bounded failed-execution cost.
On correct coverage all L decisions are correct and certify the true answer.
An expected *unconditional correct certification time* treating errors as infinity
is NOT finite and is not what this upper bound states.

For the corresponding constrained staged testing benchmark with uniform joint
correctness >=1-delta, flip just bit j while preserving all prior channel laws.
The stopped pre-choice history and the choice for bit j yield the same binary
change-of-measure bound. Summing expected probe times gives
sum_j d_j kl(1-delta,delta)/I_j. The implemented rule matches this benchmark to
risk-allocation/logarithmic terms for fixed depth. This is not a universal matching
bound for all adaptive experiment graphs, multiple acceptable answers or MDPs;
those stronger targets remain open in this draft. Raw Monte Carlo cannot prove them.

## 7. Algorithm

Safe Refinement maintains V, returns a common acceptable policy as soon as one
exists, otherwise gathers confidence-valid information from the current robust-safe
menu and unlocks experiments as V shrinks. StaticSafeID fixes its menu at the
initial common-safe set. RobustOnly never explores. FullModelID changes the stop
requirement to singleton (with the same safety budgets/design ordering). Oracle
knows the true model. UnsafeMLE / CertaintyEquivalent are explicitly unsafe foils,
not algorithms covered by Proposition 3.

## 8. Experiments and interpretation

See SINGLE_LIFE_PROTOCOL_V1.md. All parameters, seeds, designs, baseline ties,
timeouts and observation channels are fixed before production. Empirical
catastrophe rates and intervals are calibration checks only. Each recorded run
is a different single life; regeneration within it never resets risk. Censored
certification times remain in restricted means. A semi-empirical UAV clock can
identify its multiplier after one safe completed cycle; this structural outcome
must be reported even if it removes the proposed learning advantage.

The paper's significance and novelty have not been established by this draft.
The finite-menu characterization and specialized matching bound are narrower
than the initial arbitrary-RL ambition; no Level-5 achievement is claimed.
