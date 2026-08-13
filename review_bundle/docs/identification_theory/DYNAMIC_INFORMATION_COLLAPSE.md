# Dynamic Information Collapse

## Target

Test whether a learned safety filter can create a theorem-level phenomenon stronger than static no-overlap: uncertainty causes rejection, rejection prevents data, and the resulting uncertainty prevents future expansion.

## Status

`COHERENT AS A DIAGNOSTIC; NOT NOVEL AS A GENERAL THEORY CLAIM`.

## Proposition 4 — Coordinate-Separable Conservative Lock-In

Let $\mathcal{A}$ be finite. For each action $a$, a learner maintains an upper safety-risk bound $U_t(a)$ and admits

$$
\widehat{\mathcal{A}}_t
=
\{a:U_t(a)\le\delta\}.
$$

Assume:

1. an action can be executed only if it belongs to $\widehat{\mathcal{A}}_t$;
2. $U_{t+1}(a)=U_t(a)$ whenever $a$ is not executed at time $t$;
3. for some action $b$, $U_0(b)>\delta$;
4. the true risk nevertheless satisfies $r_C(b)\le\delta$.

Then $b\notin\widehat{\mathcal{A}}_t$ for all $t$.

### Proof

At $t=0$, $b$ is inadmissible by assumption 3, so it is not executed by assumption 1. Assumption 2 gives $U_1(b)=U_0(b)>\delta$. Repeating the same implication inductively yields $U_t(b)=U_0(b)>\delta$ for every $t$. Therefore $b$ is never admitted despite being truly safe. $\square$

## Interpretation

This is a self-reinforcing conservatism loop, but assumption 2 makes the conclusion immediate. Structural generalization, shared parameters, a simulator, an instrument, or a safe neighboring probe can invalidate assumption 2. The proposition therefore identifies an implementation failure mode, not a broad lower bound.

## Theorem 1 — Uniform Safety Versus Identification

Let $b$ be an initially untested action. Consider two environments:

- $\mathcal{M}_S$: the first execution of $b$ is safe;
- $\mathcal{M}_U$: the first execution of $b$ causes the declared failure event with probability one.

Assume the two environments have identical observed laws until the first execution of $b$. Let algorithm $\mathcal{L}$ satisfy the uniform safety requirement

$$
\Pr_{\mathcal{M}}(\text{failure during learning})\le\beta
$$

for every $\mathcal{M}\in\{\mathcal{M}_S,\mathcal{M}_U\}$. Let $\theta_{\mathcal{M}_S}=S$, $\theta_{\mathcal{M}_U}=U$, and let $\widehat\theta\in\{S,U\}$ be the learner's declaration. Then

$$
\inf_{\mathcal{L}:\,\beta\text{-safe}}
\sup_{\mathcal{M}\in\{\mathcal{M}_S,\mathcal{M}_U\}}
\Pr_{\mathcal{M}}(\widehat\theta\neq\theta_{\mathcal{M}})
\ge
\frac{1-\beta}{2}.
$$

### Proof Strategy

Couple the two environments so that their histories are identical until $b$ is executed, then bound the total variation distance of the observed-data laws and apply the binary testing lower bound.

### Proof

Let $A$ be the event that $b$ is executed at least once. Under $\mathcal{M}_U$, failure occurs whenever $A$ occurs. Uniform safety therefore implies

$$
P_U(A)\le\beta.
$$

Before the first execution of $b$, the coupled histories and algorithmic randomization are identical. Hence the probability of reaching the first execution is the same in both environments:

$$
P_S(A)=P_U(A)\le\beta.
$$

On $A^c$, the complete observed data are identical under the coupling. Therefore

$$
\operatorname{TV}(P_S^{\mathrm{obs}},P_U^{\mathrm{obs}})
\le P_S(A)
\le\beta.
$$

For testing two simple hypotheses with equal prior probability, the minimum average error is

$$
\frac{1-\operatorname{TV}(P_S^{\mathrm{obs}},P_U^{\mathrm{obs}})}{2}
\ge\frac{1-\beta}{2}.
$$

The maximum of the two model-specific errors is at least their average, which proves the claim. $\square$

## Consequence

For small allowed failure probability $\beta$, uniformly safe learning cannot reliably distinguish an actually safe action from an observationally identical catastrophic action unless extra structure transfers information before executing it. The exact-safety case $\beta=0$ gives minimax error at least $1/2$.

## Necessary-Structure Corollary

To obtain identification error strictly below $(1-\beta)/2$, at least one premise of Theorem 1 must be broken. Examples include:

- positive safe-probe probability with a non-catastrophic observation channel;
- Lipschitz, linear, monotone, GP/RKHS, or mechanistic coupling to observed actions;
- a validated simulator or auxiliary dataset;
- a trusted baseline or safe seed enabling boundary approach;
- an instrument, shadow variable, or randomized intervention;
- a weaker safety requirement or restricted environment class.

This is a necessary disjunction, not a statement that any listed assumption is sufficient.

## Trajectory-Level Embedding

Replace $b$ by the decision “continue task for one additional step” at a commitment history. Let the unsafe environment strand the vehicle with probability one after continuation and let the safe environment complete the task without threatening return. If commitment is required to avoid stranding with probability at least $1-\beta$, the same coupling and testing bound applies.

This embedding does not create a separate theorem: continuation is simply the untested action in an augmented history state.

## Resource Coupling

If every execution of $b$ consumes at least $d_E>0$ battery and the available exploration allocation is $B_E$, then the number of probes satisfies

$$
N_b\le\left\lfloor\frac{B_E}{d_E}\right\rfloor.
$$

Combining this deterministic budget with a statistical testing lower bound can limit achievable accuracy. However, this is a direct composition of hypothesis-testing sample complexity with a knapsack constraint. No new minimax rate has been derived here.

## Closest-Work Reduction

- SafeMDP already states that safe exploration without regularity is impossible and proves expansion only of a structurally defined safely reachable set.
- SafeOpt and Active Learning with Safety Constraints study sample allocation for unknown safety boundaries.
- ActSafe explicitly chooses informative safe trajectories to expand a pessimistic safe policy set.
- Learning When Not to Learn proves an abstention-versus-catastrophic-feedback impossibility and recovers sublinear regret under Lipschitz/trusted-region assumptions.

Theorem 1 is mathematically valid, but its mechanism is not a theorem-level separation from these lines.
