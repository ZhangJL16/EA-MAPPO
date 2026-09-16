# Derivation Package: S2 Feasibility-Gated Information and Regret

## Target

Build a positive, finite-time information/regret representation for a fixed
Consumption-MDP base class. Stop negative example search. Determine the correct
objects before asserting a matching capacity-dependent minimax theorem.

## Status

COHERENT AFTER REFRAMING. The observation quotient, occupancy representation,
KL identity, and regret identity below have handwritten proofs. The resulting
information-constrained lower bound applies to a specified testing requirement.
Matching minimax regret and attainable optimal information allocations remain
OPEN. This is not a claim of new learning theory or independent verification.

## Invariant Object

The finite-time information/regret frontier achievable by safe trajectories,
not the dimension of an identifiable parameter subset.

## Assumptions

Retain fixed base Theta, common known exact transition support, known integer
costs and resource timing, single recharge q, decreasing, viable q_B, and safely
reachable successor-closed D_B. Scalar optimal gain and canonical AROE bias exist.
The S0 uniform regularity supplies span(h_B,theta)<=H independent of B,theta.

The observed base outcome is (S_{t+1},Y_t), with joint conditional law
nu_theta(.|s,a), independent of battery and earlier history given (s,a).
Y_t is in [0,1]; its mean is r_theta(s,a). Transition and reward outcomes need
not be independent. The learner sees state, battery and its executed reward,
and has no external/generative experiment access. The initial state is q_B.
For the KL and optimization statements take finite pairwise row KL on safely
queryable pairs; equality/quotient statements need no KL assumption.
Safe policies may be adaptive, randomized, and history-dependent, but their
action rule is the SAME mapping under every possible theta.

## Notation

- X_t=(S_t,b_t), using q_B as the single normalized charger state.
- A_B(x): known viable actions in D_B; Ptilde_B,theta: lifted transition.
- E_B: base pairs (s,a) executed with positive probability by SOME finite safe
  supported history from q_B, followed by action a. This set is common to theta
  because exact support is common; mere local feasibility does not define E_B.
- theta ~_B theta': all finite safe-policy transcript laws agree.
- mu(x,a)=E_theta sum_{t<T} 1{X_t=x,A_t=a}, and
  N(s,a)=sum_{x:base(x)=s} mu(x,a).
- I_theta,theta'(s,a)=KL(nu_theta(.|s,a)||nu_theta'(.|s,a)).
- d_B,theta(x,a)=rho_B,theta^*+h_B,theta(x)-r_theta(s,a)
  -sum_{x'} Ptilde_B,theta(x'|x,a) h_B,theta(x') >=0.

## Derivation Strategy

Safety defines observable rows and feasible time-indexed occupancies. Apply the
same occupancy to information and Bellman-centered regret. Keep exact identities,
testing consequences, and unproved minimax attainability separate.

## Derivation Map

1. Safe experimental equivalence is a quotient, not Theta_B subset Theta.
2. Virtual-budget coupling gives nested accessible experiment sets.
3. Flow constraints characterize model-specific achievable expected occupancies.
4. Transcript KL and regret are linear functionals of those occupancies.
5. Testing requirements give information constraints, hence a regret-cost lower
   bound. No testing requirement is silently assumed to follow from low regret.

## Main Derivation

### Proposition 1: safe observation quotient is control sufficient

For common known support and fully observed base outcomes,

    theta ~_B theta' iff
    nu_theta(.|s,a)=nu_theta'(.|s,a) for every (s,a) in E_B.

For the forward implication, fix a supported base-state/action prefix under a
policy prescribing its actions irrespective of rewards. Reaching this queryable
pair is a positive-probability event; no exact continuously-valued reward prefix
is conditioned upon. Equal transcript laws give equal probabilities to the event;
conditioning on this event and its specified action gives equal next-outcome
laws. For the reverse implication, induct on transcript length. The same learner
action kernels and identical queryable outcome laws give identical next transcript
laws; the deterministic resource update creates no new statistical randomness.

Therefore for every fixed safe policy pi and every finite T, its expected
cumulative reward is the same under equivalent parameters. Its long-run expected
gain is also the same. Taking the supremum over the SAME safe-policy class gives
equal optimal gains and the same optimal-policy set within each equivalence class.

The correct object is Theta/~_B. Theta itself does not shrink with capacity.
An incomplete observation quotient is not automatically a learning obstruction:
its indistinguishable parameters already agree on EVERY safe-policy objective.
This is a sufficiency statement for the full experimental quotient, not a compact
learnable representation theorem.

### Proposition 2: capacity induces an experiment filtration

For B_2>=B_1, E_B1 subset E_B2. Simulate a B_1-safe supported history using a
virtual low-capacity budget, while the real budget resets to B_2 at each recharge.
Nonnegative extra resource preserves supported continuation safety. The same
base pair can therefore be safely queried at B_2. The virtual budget is memory,
not a modification of plant dynamics. Thus larger capacities refine the quotient.
Filtration of observable rows does NOT imply monotonicity of minimax regret:
the comparator and regret costs change as well.

### Proposition 3: exact model-specific occupancy representation

Let y_t(x,a)>=0 be time-indexed expected action probabilities, supported only on
viable pairs. They satisfy the exact flow equations

    sum_a y_0(x,a)=1{x=q_B},
    sum_a y_{t+1}(x,a)=sum_{z,a} y_t(z,a) Ptilde_B,theta(x|z,a),
        0<=t<T-1.

Let O_B,theta(T) be the projection mu(x,a)=sum_{t<T} y_t(x,a) of this polytope.
Every safe adaptive policy generates such y by taking its marginal action/state
probabilities. Conversely every feasible y is realized by the safe time-dependent
Markov rule pi_t(a|x)=y_t(x,a)/sum_a y_t(x,a), with arbitrary viable actions at
zero-probability states. Induction verifies its marginals. Hence this is an EXACT
expected occupancy characterization at a fixed theta, not just a stationary
flow relaxation. The construction may use the true model as an analytical witness.

Crucial limitation: feasible occupancies for different theta cannot in general
be selected independently by ONE unknown-model learner. This characterization
alone does not solve the joint minimax learning problem or provide an algorithm.

### Identity 4: information comes from shared base rows

For any one fixed safe learner A,

    KL(P_B,theta^{A,T} || P_B,theta'^{A,T})
       =sum_{s,a} N_theta^{A,T}(s,a) I_theta,theta'(s,a).

Apply the conditional KL chain rule at each time. Initial state and learner
action kernels are identical, hence contribute zero. Outcome law contributes
I at its selected base pair. The known battery update is deterministic and the
base successor is observed, so lifting neither duplicates nor hides this outcome
information. Summing and taking expectation gives the identity. No independence
of action counts or IID trajectory assumption is used.

### Identity 5: regret is safe occupancy cost plus a bounded boundary

Conditional on X_t,A_t, the definition of d gives

    rho^*-r_theta(S_t,A_t)
       =d_B,theta(X_t,A_t)+E_theta[h(X_{t+1})|history,A_t]-h(X_t).

Taking expectation and telescoping,

    Reg_B,theta^A(T)=sum_{x,a} mu_theta^{A,T}(x,a) d_B,theta(x,a)
                      +E_theta h_B,theta(X_T)-h_B,theta(q_B).

Since both endpoints lie in D_B and span(h)<=H,

    |Reg_B,theta^A(T)-<mu_theta^{A,T},d_B,theta>|<=H.

This is the exact decomposition currently supported by the model. The slack
cost includes suboptimal decisions and transit/recovery costs where appropriate;
it is not a pure prediction-error term. It depends on the capacity-specific
optimal comparator. No separate additive feasibility penalty is established.

### Proposition 6: an information-constrained cost lower bound

Fix theta and a set Alt of parameter alternatives. Suppose the learner, after
T steps, must satisfy specified pairwise testing requirements: for each theta'
in Alt, some transcript event A_theta' has
P_theta(A_theta')>=1-beta and P_theta'(A_theta')<=beta, where 0<beta<1/2.
Binary data processing implies

    sum_{s,a} N(s,a) I_theta,theta'(s,a)
          >= kl(1-beta,beta), for every theta' in Alt.

Define the capacity-dependent information cost

    C_B,theta(T,beta,Alt)=inf_{mu in O_B,theta(T)} <mu,d_B,theta>
      subject to sum_{s,a}(sum_{x:base(x)=s} mu(x,a)) I_theta,theta'(s,a)
                    >=kl(1-beta,beta), for every theta' in Alt.

Every learner meeting the stated tests therefore satisfies

    Reg_B,theta^A(T)>=C_B,theta(T,beta,Alt)-H.

Proof: its expected occupancy is feasible by Proposition 3 and Identity 4;
minimization bounds its slack cost from below; Identity 5 bounds the boundary.
An empty feasible set means no such testing procedure exists at that horizon.

This is a positive cost characterization for NECESSARY allocations in a
specified testing problem, NOT matching minimax regret. For regret lower bounds,
Alt and beta must first be derived from genuinely decision-relevant alternatives
and low-regret requirements. Demanding identification of irrelevant parameters
would impose a task the learner was never required to solve.

## Remarks and Interpretation

- Exact-arithmetic audit: 21,842 finite-prefix checks on the existing two-state
  example passed for the occupation/slack/boundary regret identity, including
  idling and terminal w states. This checks telescoping signs, not the general
  quotient, adaptive KL, occupancy attainability or minimax theorem. No new
  counterexample or learning algorithm was introduced by this diagnostic.
- Capacity enters E_B, the achievable occupancy polytope, and the comparator's
  slack cost. Primitive row-information functions do not change with B.
- E_B alone records whether a query is possible, not its attainable frequency,
  information per unit mission time, or recovery regret. Thus cardinality or
  injectivity of Theta/~_B alone cannot supply a quantitative regret theorem.
- If a larger-capacity or unconstrained reference gain rho_ref>=rho_B^* is
  chosen, the exact comparator identity is
  Reg_ref^A(T)=Reg_B^A(T)+T(rho_ref-rho_B^*).
  That latter term is a comparator loss, not an exploration penalty under the
  original safe comparator. It need not be O(1). An additive decomposition with
  a symbol Psi requires specifying this comparator and reference experiment.
- It is incorrect to describe existing RL theory as depending ONLY on parameter
  or state counts. Information-cost constrained exploration already has a long
  history. Checked primary sources include Graves--Lai (1997) and structured
  RL (NeurIPS 2018); the formulas above are foundational applications of those
  ideas, not a novel general principle merely because resource masks are added.
  https://statistics.stanford.edu/technical-reports/asymptotically-efficient-adaptive-choice-control-laws-controlled-markov-chains
  https://papers.neurips.cc/paper/8103-exploration-in-structured-reinforcement-learning

## Boundaries and Non-Claims

No additional negative construction. No implication 'full identifiability gives
B-free rate' or 'partial identifiability gives regret lower bound' is claimed.
No matching minimax characterization, new algorithm, experiment, training,
CONFIRM access, optimal compact planner, or Spotlight-level novelty claim.
Finite-time KL identities and flow equations are exact, but independent proof
audit and formal probability verification remain outstanding.

## Open Risks

1. Must establish a resource-specific simplification or sharp bound for C_B
   rather than just restate existing structured-RL information constraints.
2. Need a decision-relevant alternative set and a low-regret-to-test lemma before
   translating Proposition 6 into minimax regret. No unrestricted PAC-identification
   requirement may be substituted for the original control objective.
3. An attainability theorem must use one learner across Theta, not independent
   oracle occupancies. This is the substantive positive upper-bound gap.
4. Compact computation and matching rates are separate questions; explicitly
   lifted time-indexed occupancies are semantic objects, not poly(log B) algorithms.
