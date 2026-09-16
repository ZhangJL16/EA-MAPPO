# Derivation Package: Single-Recharge Consumption Bandit Attainability

## Target

Answer the requested narrow Attainability Test with decision-derived alternatives
and one unknown-model learner, without implementing or running the learner.
Do not further expand quotient, occupancy or KL foundations.

## Status

COHERENT AFTER REFRAMING. Handwritten matching instance-dependent asymptotic
lower/upper theorem for deterministic-duration independent Bernoulli sorties.
The lower bound is for uniformly efficient learners, not an unrestricted pointwise
infimum. This is NOT a minimax-in-Theta logarithmic theorem or a novel resource-RL
result. Independent proof audit and machine verification are not claimed.

## Invariant Object

C_B(mu), the optimal log-calendar-time regret coefficient at a fixed positive-gap
instance, under the original safe average-reward comparator.

## Assumptions

Freeze K sorties i=1,...,K and Theta=(0,1)^K once for ALL capacities and horizons.
Each sortie begins and ends at the unique charger q, lasts a known integer
ell_i>=1, has known nonnegative integer stage consumptions of positive total
e_i>=1, and reveals one reward Y~Bernoulli(mu_i) at its terminal time. All
intermediate stages have zero reward and known deterministic transitions.
Successive samples of every arm are IID and independent across arms.

The learner chooses only at q; after choosing i, its acyclic corridor is forced
until the return to q. Thus arbitrary within-sortie abort/replan actions and
shared stochastic parameters between sorties are NOT in this class. It can be
encoded as a finite Consumption MDP with disjoint deterministic corridors and
one q. The base class, costs, supports, durations and reward law class do not
depend on B. Debit consumption before resetting on arrival at q.

The feasible arm set is F_B={i:e_i<=B}; assume it is nonempty. The learner must
choose from F_B, starts at q_B, and runs continuously without extra reset or
generative queries. The horizon is a calendar-time cutoff of this continuing
trajectory, not a termination/restart operation. Put ell_max=max_i ell_i.
All base cycles have positive consumption, p_0=1, and safe recovery has
H_R<=ell_max. For the stated instance the optimal rate arm is unique.

## Notation

rho_i=mu_i/ell_i; i_*=argmax_{i in F_B} rho_i; rho^*=rho_i*.
delta_i=ell_i rho^*-mu_i>0 for suboptimal feasible arms.
q_i=ell_i rho^*. A suboptimal arm is competitive if q_i<1.
kl(p,q)=p log(p/q)+(1-p)log((1-p)/(1-q)), with endpoint conventions.

N_i(T) counts COMPLETE sorties by calendar cutoff T; M_i(T) counts starts.
There is at most one started but incomplete sortie, so |M_i-N_i|<=1.
A learner is uniformly efficient if Reg_mu(T)=o(T^a) for EVERY fixed mu in
Theta and EVERY a>0, at the capacity under consideration. The same learner
mapping operates for all mu; it is not allowed a preselected oracle allocation.

## Derivation Strategy

Decision change by a single-arm alternative -> low-regret-induced test ->
sampling lower bound -> separable information allocation -> calendar-time
rate KL-UCB upper bound. Use a checked published concentration inequality;
do not guess an adaptive sampling bound.

## Derivation Map

1. Use full-cycle reward and duration bookkeeping to retain the terminal residual.
2. Derive decision-relevant testing events from uniformly efficient control.
3. Solve the necessary allocation problem explicitly.
4. Specify one implementable theoretical policy using no true mu/rho/gaps.
5. Bound its draws with predictable-sampling KL concentration and IID tails.
6. State matching scope and diagnose whether the geometry adds new science.

## Main Derivation

### Identity 1: reward and unfinished-cycle accounting

Let Z_T=T-sum_i ell_i N_i(T). Continuity and bounded deterministic durations
give 0<=Z_T<ell_max. Completion of the kth sample is determined by its selected
start and duration BEFORE seeing that sample; hence predictable sampling gives
E sum rewards=sum_i mu_i E N_i(T). Consequently

    Reg_mu(T)=sum_{i in F_B} delta_i E N_i(T)+rho^* E Z_T.

The residual is nonnegative and bounded by ell_max, independent of B. Do not
replace sample means inside arbitrary stopping sums without this predictability
argument. The optimal average gain is rho^*: time-weighted mixtures cannot beat
the maximal reward/duration ratio and repeating i_* attains it.

### Proposition 2: low regret forces decision-relevant discrimination

Fix a competitive suboptimal i and any v in (q_i,1). Construct mu^(i,v) by
changing ONLY mu_i to v. This lies in the SAME fixed Theta; i becomes uniquely
optimal. Unsafe arms are not alternatives requiring experimentation.

For transcript event A_T={ell_i N_i(T)<=T/2}, uniform efficiency and Identity 1
give, under mu, E N_i(T)=o(T^a) for every a>0. Markov's inequality therefore
gives P_mu(A_T^c)=o(T^{a-1}).

Under mu^(i,v), the smallest other-arm rate gap gamma>0. On A_T, completed
time on other arms is at least T/2-ell_max. Identity 1 pointwise in occupations
therefore implies
Reg_mu^(i,v)(T)>=gamma (T/2-ell_max) P_mu^(i,v)(A_T).
Uniform efficiency gives P_mu^(i,v)(A_T)=o(T^{a-1}).

Only the terminal rewards of complete i sorties differ between the two transcript
laws. The adaptive KL identity is E_mu N_i(T) kl(mu_i,v). Binary data processing
and kl(p,q)>=p log(1/q)-log(2), applied to A_T, imply

    liminf_{T->infinity} E_mu N_i(T) kl(mu_i,v)/log T >=1-a.

Taking a down to zero, then v down to q_i, yields

    liminf E_mu N_i(T)/log T >=1/kl(mu_i,q_i).

This derives the testing requirement from the CONTROL objective, not from an
added demand to identify every parameter. Merely changing optimal gain while
keeping all optimal decisions unchanged would not justify this argument.

### Theorem A: necessary information coefficient and closed-form allocation

For every uniformly efficient learner,

    liminf Reg_mu(T)/log T >= C_B(mu),
    C_B(mu)=sum_{i in F_B, i!=i_*, q_i<1}
                         delta_i/kl(mu_i,q_i).

Equivalently this is the solution of

    min_{eta_i>=0} sum_{i!=i_*} delta_i eta_i,
    subject to eta_i kl(mu_i,q_i)>=1 for each competitive suboptimal i.

The constraints come from all single-arm alternatives v>q_i, not a fabricated
finite-horizon log T test. Because arm samples are independent and sorties reset
to the SAME control state, the constraints decouple and the infimum is explicit.
Noncompetitive q_i>=1 arms cannot become strictly better by changing their
Bernoulli mean alone; no such exploration constraint is imposed.

This asymptotic allocation describes O(log T) exploratory sorties, whose total
known duration is o(T); remaining time is filled by i_*. It does not assert
arbitrary oracle allocations are achievable by an unknown-model learner.

### Theoretical learner: calendar-time rate KL-UCB

First complete every feasible arm once. At each subsequent charger decision
with elapsed integer calendar time t, use completed counts n_i and means hat_mu_i.
Define f(t)=log(t+3)+3 log(log(t+3)) and

    U_i(t)=sup{u in [hat_mu_i,1]: n_i kl(hat_mu_i,u)<=f(t)}.

Choose a feasible arm maximizing U_i(t)/ell_i, using any fixed tie rule; execute
it completely and update its statistic. This policy uses only known feasibility,
known duration and its own observed rewards, not mu, rho^*, delta or eta.
Every completed cycle resets the battery; e_i<=B guarantees pathwise safety.
Only a theoretical policy definition is authorized here; no source runner,
simulation, training or experiment is created.

### Proposition 3: the same learner attains the coefficient

At a charger decision, if U_i*/ell_i*>=rho^* and suboptimal i is selected,
then U_i>=q_i. Fix 0<epsilon<q_i-mu_i for a competitive i. If its empirical
mean is <=mu_i+epsilon, monotonicity of kl(p,q_i) in p<q_i implies

    n_i<=f(t)/kl(mu_i+epsilon,q_i)
         <=f(T)/kl(mu_i+epsilon,q_i).

Thus selections beyond this count require either optimal-arm underconfidence
or an empirical deviation hat_mu_i>mu_i+epsilon.

For the first event use Garivier--Cappe (2011), Theorem 10, predictable-sampling
bound P(U_i*(t)<mu_i*)<=e ceil(f(t)log(t+1)) exp(-f(t)). Calendar-time reward
indicators are predictable because durations are deterministic and choices
precede their rewards; unused latent IID samples can be supplied at other times
for applying the theorem. At q there is no pending reward, and all completed
samples are available. This bound is O(1/[t log t]); summing over all calendar
times <=T bounds the EXPECTED number of underconfidence decisions by O(log log T).

For the second event, each selected i increments its completed count before the
next decision. Every sample-count s is therefore charged at most once, and IID
Hoeffding gives sum_{s>=1} P(hat_mu_i,s>mu_i+epsilon)<=sum_s exp(-2s epsilon^2),
a finite constant for fixed epsilon. Including initialization,

    E M_i(T)<=f(T)/kl(mu_i+epsilon,q_i)+O(log log T)+O(epsilon^-2)+O(1).

The distinction M_i-N_i is at most one. Divide by log T, then let epsilon down
to zero to get limsup E N_i/log T<=1/kl(mu_i,q_i).

If q_i>1, i cannot be selected when the optimal index is confident, so its draws
are O(log log T). If q_i=1, a confident selection requires U_i=1, possible only
while all its observed rewards are successes. Once a failure is seen U_i<1
forever at finite times. The expected draws up to this failure are at most
1/(1-mu_i); further selections are charged to optimal underconfidence. Thus
noncompetitive arms also have o(log T) expected draws.

This gives O(log T) regret at every fixed instance with positive suboptimal gaps,
and hence uniform efficiency. Tied best arms need no discrimination between
themselves; the same count argument handles each strictly suboptimal arm, which
is enough for the uniform-efficiency premise on full Theta.

### Theorem B: matching attainability, with the correct quantifiers

For every fixed unique-best instance in this class, the SAME rate KL-UCB learner
is safe for all times and satisfies

    lim_{T->infinity} Reg_mu(T)/log T=C_B(mu).

If C_B(mu)>0 this is Reg=C_B(mu)log T+o(C_B(mu)log T). If C_B=0, the valid
statement is Reg=o(log T), NOT an undefined o(C_B log T) expression.

This is an asymptotic optimality statement among uniformly efficient algorithms.
It is NOT inf_A sup_mu Reg=Theta(C_B log T). For unrestricted pointwise inf_A
at a single mu an oracle-specialized constant-arm policy invalidates that lower
bound; for minimax over gaps approaching zero the log asymptotics are not uniform.

## Remarks and Interpretation

- Attainability Test: YES for this explicitly frozen class; decision-relevant
  lower + one adaptive learner's upper match. No extra testing task is assumed.
- Publication test: NOT established. Here capacity only removes arms from a
  known set. Once an arm is feasible, it can be repeated independently after
  every reload. The allocation geometry is diagonal, not a new coupled resource
  frontier. Unequal durations change thresholds and regret weights, but the
  proof is a duration-aware standard KL-UCB analysis.
- Thus this theorem must not be marketed as a new feasibility-gated Graves--Lai
  principle. It is a fully specified baseline and a check that the lower-bound
  framework can be attained when the control geometry actually decouples.
- Any novel extension would have to demonstrate a genuinely coupled acquisition
  constraint or a sharp structural improvement over existing bandit/RL results.
  No such extension is silently included or approved by this document.

## Sources and Verification Basis

Sanity checks: 636 exact Fraction full/partial-sortie calendar-time identities
passed for fixed durations (2,3,4), energy thresholds (2,4,5), means (.4,.5,.6)
and horizons 1--12. Numerical allocation checks give C_B=0, 4.899320, 6.810459
at B=2,4,5 respectively. These test bookkeeping and the separable formula,
not adaptive concentration, asymptotic optimality or theorem novelty.

Garivier and Cappe, COLT 2011, checked the primary PDF: Theorem 2 provides the
standard draw-count benchmark, and Theorem 10 provides the adaptive concentration
input used explicitly above. The unequal-duration/calendar-time proof steps are
spelled out here rather than attributed to a theorem about ordinary equal-time
arms. https://proceedings.mlr.press/v19/garivier11a/garivier11a.pdf

Adjacent prior art: budgeted bandit KL-UCB-SC already treats reward/cost rates
and asymptotic optimality for Bernoulli costs/rewards; the publisher abstract
was checked, not a claimed theorem-level full reduction audit.
https://www.jstage.jst.go.jp/article/transfun/E100.A/11/E100.A_2470/_article/-char/en

No new counterexample. No experiment, CONFIRM access, training, UAV or algorithm
implementation. Handwritten proof only; independent audit still required.

## Boundaries and Non-Claims

Unknown transitions, shared parameters, stochastic unbounded duration, interrupted
sorties, within-cycle planning and coupled safe occupation constraints are excluded.
No arbitrary Consumption-MDP matching theorem, finite-time exact minimax constant,
capacity-growing lower bound, computational separation, or Spotlight claim.

## Open Risks

1. Independently audit predictable reward timing and the calendar-time use of
   Theorem 10; early within-cycle observations would change this experiment.
2. Audit the low-regret-derived test and noncompetitive endpoint handling.
3. Whether coupled resource acquisition admits a novel attainable frontier is
   still unknown; this theorem does not establish that larger research direction.
