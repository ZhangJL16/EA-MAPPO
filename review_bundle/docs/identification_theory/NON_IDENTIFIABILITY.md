# Non-Identification Results

## Status

`PROVABLE AS STATED`, but the propositions are reductions to established no-support identification logic. Their correctness does not establish novelty.

## Proposition 1 — Rejected-Action Non-Identification

Let the model class contain a state $s_\star$ and proposed action $b$ such that the deployed mechanism executes $b$ with probability zero whenever $S_t=s_\star$. Suppose two environments $\mathcal{M}_0$ and $\mathcal{M}_1$ agree on the conditional law of every variable that can be observed under the deployed mechanism, but

$$
r_{C,\mathcal{M}_0}(s_\star,b)=0,
\qquad
r_{C,\mathcal{M}_1}(s_\star,b)=1.
$$

Then $r_C(s_\star,b)$ is not point identified from $\mathcal{D}_{\mathrm{obs}}$.

### Construction

Use one recurrent observed state $s_\star$ and two task actions $a$ and $b$. The filter always allows $a$ and always rejects $b$, replacing it with $a$. Under both environments, executing $a$ yields the same deterministic next state, zero collision label, and the same energy cost. The proposal and filter logs are also identical. Define the unexecuted potential outcome of $b$ to be collision-free under $\mathcal{M}_0$ and colliding under $\mathcal{M}_1$.

### Proof

The executed action is $a$ in both environments for every possible proposal. Therefore every realized transition, collision label, and energy cost has the same law. The proposal and filter mechanisms are fixed and have the same internal-randomization law, so

$$
P_{\mathcal{M}_0}^{\mathrm{obs}}
=P_{\mathcal{M}_1}^{\mathrm{obs}}.
$$

The target differs by construction. Hence the implication required for point identification fails. $\square$

## Corollary 1 — Minimax Estimation Lower Bound

For any possibly randomized estimator $\widehat r(\mathcal{D}_{\mathrm{obs}})$,

$$
\inf_{\widehat r}
\sup_{j\in\{0,1\}}
\Pr_{\mathcal{M}_j}
\left(
\left|\widehat r-r_{C,\mathcal{M}_j}(s_\star,b)\right|
\ge \frac{1}{2}
\right)
\ge \frac{1}{2}.
$$

### Proof

Because the observed-data laws are equal, $\widehat r$ has one common distribution $Q$ under both environments. The events

$$
B_0=\{|\widehat r|<1/2\},
\qquad
B_1=\{|\widehat r-1|<1/2\}
$$

are disjoint. Thus $Q(B_0)+Q(B_1)\le 1$, so at least one success probability is at most $1/2$. The corresponding error probability is at least $1/2$. $\square$

## Proposition 2 — Post-Commitment Continuation Non-Identification

Consider a finite-horizon process with a history $h_\star$ at time $t$ such that the deployed rule commits to the charger with probability one at $h_\star$. Suppose $\mathcal{M}_0$ and $\mathcal{M}_1$ agree on the complete pre-commitment law and on the charger-policy suffix, but the potential task-continuation return at $h_\star$ is $0$ under $\mathcal{M}_0$ and $1$ under $\mathcal{M}_1$. Then $V_{\mathrm{cont}}(h_\star)$ is not point identified.

### Proof

The two environments generate identical histories until $h_\star$. At $h_\star$, both execute the same deterministic commitment decision. Their charger-mode transitions and outcomes are equal by construction. The task-continuation kernel is never invoked, so changing it does not change the observed-data law. The target differs. $\square$

## Proposition 3 — Two-Level Censoring Reduction

Assume the augmented state $X_t=(S_t,E_t,Z_t,\Xi_t)$ is Markov, with $Z_t$ absorbing after commitment and $\Xi_t$ containing the filter/learner memory. Then identification of any target policy that assigns positive occupancy to rejected task actions or to task continuation after commitment requires identification outside the support of the deployed augmented behavior policy.

### Proof

The deployed system induces a behavior policy $\mu$ over augmented history-action pairs. If an action is rejected deterministically, the corresponding physical execution pair has $d^\mu(x,a)=0$. If the mode is charger-committed, task-continuation actions have $d^\mu(x,a)=0$ because the absorbing mode removes them from the action kernel. A target that uses either pair has positive target occupancy where behavior occupancy is zero. Therefore its value or local outcome functional includes a component outside observed support. Without a restriction linking that component to supported outcomes, Proposition 1 or 2 can be embedded at that pair. $\square$

## Why These Results Are Not a New Theorem Class

- Proposition 1 is the standard zero-propensity/selective-label construction.
- Proposition 2 is the same construction on a history-action suffix.
- Proposition 3 formalizes that the two indicators become occupancy zeros in an augmented MDP.
- The use of a learned filter explains how the support arose, but does not alter the observational-equivalence argument once the behavior law is fixed.

The propositions are useful diagnostics for the UAV system, but cannot support a claim stronger than “the application instantiates known no-support non-identification.”
