# Censoring Mechanism

## Action-Level Censoring

For a task proposal $A_t^P=a$, the full potential outcome $Y_t(a)$ is observed only if

$$
M_t^A=M_t^T F_t=1.
$$

The relevant propensity is

$$
p_t^A(a\mid h,s)
=
\Pr(M_t^A=1\mid \mathcal{H}_t=h,S_t=s,A_t^P=a).
$$

If $p_t^A(a\mid h,s)=0$, inverse-propensity weighting cannot identify the outcome of $a$ at $(h,s)$. Logging the deterministic reason for rejection does not change this structural zero.

## Trajectory-Level Censoring

Define the counterfactual task-continuation suffix after a history $h_t$ as

$$
Y_{t:H}^{\mathrm{task}}
=
\{S_{k+1}^{\mathrm{task}},L_k^{C,\mathrm{task}},C_k^{E,\mathrm{task}}\}_{k=t}^{H}.
$$

If $t\ge\tau_c$, this suffix is not physically realized. Its observation propensity is

$$
p_t^T(h_t)=\Pr(\tau_c>t\mid\mathcal{H}_t=h_t).
$$

A deterministic stopping rule creates $p_t^T(h_t)=0$ on histories where it commits. The observed charger suffix is not a task-continuation label.

## Endogeneity

The censoring mechanism is endogenous in the operational sense that $F_t$ and $G_t$ are functions of learner state, uncertainty, and accumulated data. It therefore changes future state-action occupancy. This endogeneity has three distinct cases:

1. **Known adaptive selection.** The filter is fully logged and depends only on observed history. Selection probabilities may be known, but zero support remains fatal.
2. **MNAR selection.** The decision also depends on unobserved variables associated with the missing potential outcome. Identification needs additional assumptions such as proxies, shadow variables, or a structural model.
3. **Performative environment change.** Deployment changes the physical transition or outcome law, not merely which parts are observed. This is a stronger mechanism and is not assumed here.

The candidate problem primarily has case 1 with structural zeros. Calling it “endogenous” does not by itself move it beyond positivity/support failure.

## Two-Level Composition

The joint observation indicator for a proposed task outcome is

$$
M_t=M_t^T F_t.
$$

The pair $(M_t^T,F_t)$ is useful for diagnosis because the two mechanisms operate at different temporal scales. For point identification, however, their product determines whether the target outcome enters the likelihood. In an augmented state space, commitment is an absorbing action and filtering is an action-selection kernel. Hence the composition generally becomes a history-action support restriction.

## Resource Coupling

An executed probe can consume a vector of resources

$$
R_t^{\mathrm{probe}}
=
\bigl(C_t^E,\,L_t^C,\,1\bigr),
$$

representing energy, collision-risk expenditure, and one unit of remaining horizon. If remaining resources are $B_t$, the update is

$$
B_{t+1}=B_t-R_t^{\mathrm{probe}}.
$$

This coupling is operationally important, but it is representable by augmenting the state with $B_t$. A new theorem cannot rest only on the existence of this update because budgeted MDP and knapsack formulations already use it.

## Distinction From Administrative Censoring

The commitment time is informative for the task-continuation outcome because it is selected using energy and uncertainty. Standard independent-censoring estimators are therefore not justified automatically. Nevertheless, “informative stopping” plus structural zero continuation is still an MNAR/support problem unless an additional identifiable relation couples the stopped and observed suffixes.
