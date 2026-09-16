# Learning in Consumption MDPs — Reduction Kill Test

## Material Passport

- Origin skills: academic-research-suite / deep-research; nature-academic-search;
  formula-derivation
- Mode: theory scoping, literature verification, and derivation
- Date: 2026-09-16
- Status: **COHERENT AFTER REFRAMING / EXTRA ASSUMPTIONS**
- Immediate purpose: decide whether unknown Consumption-MDP learning is a new
  theorem problem or a corollary of average-reward linear-mixture RL
- Non-authorization: this document does not authorize a new algorithm, UAV
  experiment, or modification of the frozen v1a empirical gate

## Target

The original target was a finite-time regret theorem for an unknown Consumption
MDP that maximizes average reward while maintaining pathwise non-depletion.

The corrected immediate target is narrower:

> Determine exactly when a known-support Consumption MDP reduces to a safe
> average-reward linear-mixture MDP, identify the first assumption of the 2025
> bounded-span theorem that is not implied by Consumption structure, and isolate
> any remaining statistical or computational separation from generic RL.

## Current verdict

**Latest amendment — S0 capacity-free span under base regularity.** For the
active fixed-base, single-reload, decreasing subclass on its safely reachable
domain, the appended S0 handwritten proof establishes
sp(h_B,theta)<=rho_B,theta (S-1+H_R)/p_min(theta)^(S-1).
Thus uniform p_min>=p_0>0 and uniform safe recovery give a span bound independent
of capacity. S0's proposed Omega(B) witness is excluded under those assumptions.
An exact three-state example separates pointwise-in-theta from class-uniform
bounds when support probabilities can vanish. No minimax characterization or
independent/machine proof is claimed. Earlier sections record the prior audit;
the appended S0 package is the current theorem statement.

The broad claim

> “We give the first safe average-reward RL algorithm for unknown Consumption
> MDPs and avoid the battery-state blow-up”

is not yet a defensible main contribution.

With known transition support and known resource costs, exact safety can be
compiled into a robust viability kernel before learning. If the resulting safe
augmented MDP additionally satisfies the Bellman optimality condition and a
usable bias-span upper bound is known, its transition model is an exact linear
mixture with the same feature dimension as the base transition model. A
state-dependent-action adaptation of existing average-reward linear-mixture RL
then gives a conditional regret result.

The first substantive failure is that known support does **not** imply the
Bellman optimality condition. The safe augmented process may contain multiple
irreversible safe end components with different optimal gains. In that case,
unknown rewards alone can make sublinear regret impossible from a single
trajectory. This obstruction is generic multichain irreversibility, not a new
replenishment phenomenon.

One primary statistical question and one secondary computational boundary remain:

1. **Statistical complexity (primary):** with \(S,A,d\) fixed and the base
   primitives controlled independently of \(B\), can a known deterministic
   replenishing counter change the minimax regret?  Bias-span dependence is a
   diagnostic, not yet the answer.
2. **Computational complexity (secondary):** exact quantitative reachability is
   already NP-hard for binary-encoded Consumption MDPs.  A candidate reduction
   transfers this hardness to exact expected mean-payoff planning, provided
   target absorption preserves the source problem's global safety semantics.
   Compact optimistic planning should therefore be revisited only after this
   restriction is verified, and then only for a deliberately tractable subclass
   or approximation target.

The first is now the central candidate.

## Invariant object

The organizing object is not the battery level itself. It is the **safe
augmented controlled process** obtained after compiling support-level viability:

\[
  \widetilde{\mathcal M}_{\rm safe}
  = (\mathcal K, A_{\mathcal K}, \widetilde P, \widetilde r),
\]

where \(\mathcal K\) is the greatest pathwise viable subset of
\(\mathcal S\times\{0,\ldots,B\}\).

All reduction, Bellman, span, regret, and computation claims must be made about
this object. A formula involving only a scalar return-energy quantity \(D(s)\)
is a special representation of \(\mathcal K\), not the general invariant.

## Model and assumptions

Let

\[
  \mathcal C=(\mathcal S,\mathcal A,P,r,c,\mathcal R,B)
\]

be a finite Consumption MDP. For the reduction test:

- \(\mathcal S,\mathcal A,\mathcal R,B\) are known;
- \(c(s,a)\in\{0,\ldots,B\}\) is known and nonnegative;
- the transition support \(E(s,a)=\operatorname{Supp}P(\cdot\mid s,a)\)
  is known;
- transition probabilities are unknown;
- rewards are either known, or separately assumed to have a specified linear
  realizability model;
- interaction is one continuing trajectory;
- safety means non-depletion on every realized path with probability one.

Reload timing must be frozen as part of the model.  The CAV 2020 Consumption-MDP
convention resets the resource before taking an action *from* a reload state.
Under that convention, define

\[
  \bar b(s,b)=
  \begin{cases}
  B,&s\in\mathcal R,\\
  b,&s\notin\mathcal R,
  \end{cases}
  \qquad
  f_{\rm src}(s,b,a,s')=\bar b(s,b)-c(s,a).
\]

Some applications instead reset on *arrival* at a reload state.  To avoid
silently switching conventions, the remainder uses a deterministic transition
update

\[
  f(s,b,a,s')
\]

defined only after the action's resource cost has been shown feasible.  The
source-reload model above is the default special case.  An arrival-reload model
must explicitly state whether the transition cost is paid before the reset.
The derivation must be modified if cost is random or safety depends on an
unknown cost envelope.

## Derivation map

1. Compute the greatest robust viability kernel using only support and costs.
2. Restrict actions to those that keep every supported successor in the kernel.
3. Lift the unknown base transition model to the augmented safe process.
4. Show that a base linear-mixture model lifts without increasing feature
   dimension.
5. Audit every assumption of Chae et al. (2025).
6. Separate mechanical mismatches from substantive blockers.
7. Prove impossibility at the two knowledge extremes.
8. Retain only questions whose solution changes statistical or computational
   complexity relative to generic structured RL.
9. Test the exact-planning hardness transfer and record the restrictions it
   does and does not cover.

## Main derivation

### Step 1 — General viability is a greatest fixed point

For \(K\subseteq\mathcal S\times\{0,\ldots,B\}\), define

\[
\Gamma(K)=
\left\{
  (s,b):\exists a\in\mathcal A(s),\
  \ \forall s'\in E(s,a),\ f(s,b,a,s')\ge0,
  \ (s',f(s,b,a,s'))\in K
\right\}.
\]

Let

\[
  \mathcal K=\nu K.\,\Gamma(K)
\]

be the greatest fixed point. For \(x=(s,b)\in\mathcal K\), define

\[
A_{\mathcal K}(x)=
\left\{
a:\ \forall s'\in E(s,a),\ f(s,b,a,s')\ge0,
  \ (s',f(s,b,a,s'))\in\mathcal K
\right\}.
\]

**Proposition requiring proof.** Every policy selecting only
\(A_{\mathcal K}(x_t)\) remains in \(\mathcal K\) and never depletes the resource
on any supported path. Conversely, if an action has a supported successor
outside \(\mathcal K\), no almost-sure-safe policy may assign it positive
probability at that augmented state.

This formulation is preferable to prematurely writing

\[
D(s)=\min_a\left(c(s,a)+\max_{s'\in E(s,a)}D(s')\right).
\]

That scalar recursion requires additional conditions. In particular, not every
reload state is sustainably usable: one must identify a reload subset that can
return to itself indefinitely. Zero-consumption cycles also require explicit
treatment. Consumption-MDP synthesis handles these issues through sustainable
reload sets and fixed-point algorithms, not merely the boundary assignment
\(D(r)=0\) for every \(r\in\mathcal R\).

### Step 2 — Exact augmented kernel

On viable state-action pairs,

\[
\widetilde P((s',b')\mid(s,b),a)
=P(s'\mid s,a)\mathbf 1\{b'=f(s,b,a,s')\},
\qquad
\widetilde r((s,b),a)=r(s,a).
\]

This is an identity. Resource evolution adds no new unknown transition
parameter when \(c,\mathcal R,B\) are known.

### Step 3 — Linear-mixture lifting

Assume the base transition is a linear mixture

\[
P(s'\mid s,a)=\langle\phi(s,a,s'),\theta^\star\rangle,
\]

with known \(\phi\in\mathbb R^d\) and unknown \(\theta^\star\). Define

\[
\widetilde\phi((s,b),a,(s',b'))
=\phi(s,a,s')\mathbf 1\{b'=f(s,b,a,s')\}.
\]

Then exactly

\[
\widetilde P(x'\mid x,a)
=\langle\widetilde\phi(x,a,x'),\theta^\star\rangle.
\]

Thus the parameter dimension remains \(d\), not \(d(B+1)\).

For any \(F:\mathcal K\to[0,H]\), let

\[
G_{s,b,a}(s')=F(s',f(s,b,a,s')).
\]

The lifted value feature satisfies

\[
\widetilde\phi_F((s,b),a)
=\sum_{s'}\phi(s,a,s')G_{s,b,a}(s').
\]

Therefore the Chae et al. feature-norm assumption is inherited whenever the
base assumption holds for every bounded function \(G:\mathcal S\to[0,H]\).

**Important limitation.** For a generic tabular unknown kernel, one elementary
linear-mixture encoding can have dimension on the order of \(S^2A\). The lifting
removes explicit \(B\)-fold parameter duplication but does not automatically
give a competitive tabular regret bound.

### Step 4 — Conditional reduction theorem

The defensible target statement is conditional:

> **Conditional reduction proposition.** Suppose the support, costs, reload set,
> and capacity are known; \(\mathcal K\) is exactly computed; the lifted kernel
> satisfies the linear-mixture norm assumptions; rewards are known or satisfy a
> separately stated realizability model; the safe augmented MDP satisfies the
> Bellman optimality condition with state-independent optimal gain; and an upper
> bound \(H\ge2\operatorname{sp}(v^\star)\) is supplied. Then a version of
> UCLK-C whose maximization is restricted to \(A_{\mathcal K}(x)\) is pathwise
> non-depleting and inherits the bounded-span linear-mixture regret theorem.

This is not yet a publishable new theorem. The state-dependent action
correspondence requires a proof adaptation because the published algorithm uses
a common action space, but this appears mechanical: every optimistic model uses
the same known support and the same robustly viable actions.

### Step 5 — Assumption audit against Chae et al. (2025)

| Published requirement | Implied by known-support Consumption structure? | Verdict |
|---|---:|---|
| Known feature map | Only if a base linear/tabular representation is supplied | conditional |
| Unknown shared transition parameter | yes after lifting | pass |
| Feature-norm bound | inherited from the base feature model | pass |
| Known deterministic reward, or stated linear reward model | no | additional assumption |
| One common action set | no; safety gives \(A_{\mathcal K}(x)\) | minor proof adaptation |
| Bellman optimality with one \(J^\star\) for all states | no | **first substantive blocker** |
| Known \(H\ge2\operatorname{sp}(v^\star)\) | no | **second substantive blocker** |
| Computational time polynomial in explicit state count | yes, but explicit count is \(O(SB)\) | statistical reduction only |
| Optimistic exploration remains safe | yes if all candidate kernels share the known support and actions stay in \(A_{\mathcal K}\) | pass |

The published UCLK-C bound is approximately

\[
\widetilde O\!\left(
d\sqrt{HT}+H\sqrt{dT}+d^{7/4}HT^{1/4}
\right),
\]

and becomes \(\widetilde O(d\sqrt{\operatorname{sp}(v^\star)T})\) in the
leading regime when \(H\) is chosen proportional to the optimal bias span.

### Step 6 — Why Bellman optimality can fail

Known support does not make the safe augmented MDP communicating. Safe action
removal can leave several closed viable end components with different optimal
average rewards. Hence

\[
  J^\star(x)\ne J^\star(y)
\]

for viable states \(x,y\), so no single scalar \(J^\star\) satisfies the Bellman
system assumed by UCLK-C.

This is not merely technical. Consider a known-support safe initial state with
two safe actions. Each action irreversibly enters one of two closed reload
components. Two environments have identical transition support and swap which
component has reward one versus zero. Before the irreversible choice they are
indistinguishable. For at least one environment, any learner chooses the wrong
component with constant probability and suffers \(\Omega(T)\) regret.

Therefore:

\[
\boxed{
\text{known support + exact safety does not imply learnable average reward.}
}
\]

One must add a safe communication/repeated-exploration condition. Once a strong
enough safe Bellman/communication condition is added, the generic
linear-mixture reduction becomes correspondingly stronger.

### Step 7 — Unknown-support impossibility

Let a known recovery action be safe. Let another action have high reward and a
recoverable successor in model \(M_1\), but in indistinguishable model \(M_2\)
also have a positive-probability successor from which the remaining resource
cannot reach a reload state. Any algorithm satisfying zero violation in \(M_2\)
cannot try that action. Consequently it has linear regret in \(M_1\).

Thus, without an envelope or certificate ruling out unseen catastrophic
successors,

\[
\boxed{
\text{unknown safety-relevant support}
+\text{almost-sure safety}
+\text{sublinear regret}
}
\]

cannot generally coexist.

This lower bound should be retained as a learnability boundary, not sold as a
complete paper unless it is sharpened into a minimal necessary-and-sufficient
information theorem.

### Step 8 — Where \(B\) has and has not disappeared

The reduction removes \(B\) from the unknown transition parameter dimension:

\[
  d_{\rm lifted}=d_{\rm base}.
\]

It does **not** establish capacity-independent learning overall:

1. \(\operatorname{sp}(v^\star)\) may depend on \(B\);
2. the required span bound \(H\) is not given by the Consumption definition;
3. UCLK-C value iteration explicitly processes the lifted state space, whose
   size can be \(S(B+1)\);
4. a generic tabular linear-mixture encoding may itself be statistically loose.

Consequently, “no battery blow-up” must be split into two different claims:

\[
\text{parameter sharing across battery levels}
\quad\text{versus}\quad
\text{compact planning without enumerating battery levels}.
\]

The first follows from the reduction. The second does not.

### Step 8a — The obvious long-sortie construction has constant span

Cycle length growing with capacity does not by itself make the bias span grow.
Consider the deterministic safe cycle

\[
R\xrightarrow{0}W_B,
\qquad
W_b\xrightarrow{1}W_{b-1}\ (b\ge1),
\qquad
W_0\xrightarrow{0}R,
\]

where edge labels are rewards, each work transition consumes one unit, and
leaving \(R\) reloads the counter.  The cycle has length \(B+2\), total reward
\(B\), and gain

\[
g_B=\frac{B}{B+2}.
\]

Normalize \(h(R)=0\).  The deterministic Poisson equations give

\[
h(W_B)=g_B,
\qquad
h(W_{b-1})=h(W_b)+g_B-1,
\]

and hence

\[
h(W_0)=-g_B,
\qquad
\operatorname{sp}(h)=2g_B<2.
\]

Thus a \(\Theta(B)\)-long replenishment cycle can have uniformly bounded bias
span.  Any lower-bound witness must create a \(B\)-scale cumulative reward
imbalance while preserving safe communication and keeping the base dynamics
uniformly controlled; merely counting down the resource is insufficient.

### Step 9 — Candidate exact-planning hardness transfer

Fu et al. prove NP-hardness of exact maximal reachability probability in
binary-encoded Consumption MDPs.  The following reduction should be written as
a formal proposition before it is cited as a result.

Take a quantitative safe-reachability instance with target set \(G\).  Preserve
all pre-target states, costs, supports, capacity, and rewards, setting the
pre-target reward to zero.  Replace each first transition into \(G\) by a
transition into a new absorbing safe reload state \(g\), with a zero-cost
self-loop and reward one.  Policies and resource paths agree up to the first
target hit, and after a target hit the transformed path remains safe forever.
For every safe policy \(\pi\), bounded convergence gives

\[
\begin{aligned}
 \rho^\pi
 &=\lim_{T\to\infty}\frac1T
   \mathbb E_\pi\!\left[\sum_{t=0}^{T-1}r_t\right]\\
 &=\Pr_\pi(\Diamond G).
\end{aligned}
\]

The same identity holds if the objective is written as the expectation of the
sample-path limiting average: the limit is one on target-hitting paths and zero
otherwise.  This gives an NP-hardness transfer **if** the source hardness remains
valid when target states already admit a safe absorbing reload continuation, or
if converting them to such states does not enlarge the source feasible-policy
set.  That condition has not yet been verified from the accessible source text.

This result has three important boundaries:

1. the cited hardness is obtained from 0--1 Knapsack and does not rule out
   pseudo-polynomial dependence on \(B\);
2. exact NP-hardness does not by itself rule out an approximation scheme or the
   approximate planning accuracy sufficient for regret;
3. the absorbing-target construction is multichain, so it does not establish
   hardness for the safely communicating/Bellman-optimal subclass required by
   the statistical question below.

Conditional on that remaining preservation lemma, it demotes a *general exact*
compact planner, but does not settle compact planning in the eventual learning
subclass.

## Revised scientific question

The recommended primary question is now:

> **A known replenishing counter adds no unknown transition parameters.  Can it
> nevertheless change the minimax statistical complexity of average-reward
> learning?**

Fix once, independently of B, a parameter set \(\Theta\) and base class
\(M_\theta=(S,A,P_\theta,r_\theta,c,R)\). The feature map, support envelope,
costs, reward class, and parameter constraints are all B-independent. For every
\(B\ge B_0\), \(\mathsf{Lift}_B\) changes only the deterministic counter
capacity. Require the intended viability/Bellman/recovery properties for all
\(B,\theta\), rather than selecting a different parameter subclass at each B.
The central target is to characterize

\[
 \mathfrak R_B(T)
 =\inf_{\mathcal A}\sup_{\theta\in\Theta}
   \operatorname{Reg}^{\mathcal A}_{\mathsf{Lift}_B(M_\theta)}(T)
\]

as a function of capacity \(B\), with \(S,A,d\) fixed.

This is better than “Safe RL in Consumption MDPs” because it names the precise
separation that generic structured RL has not already granted: parameter count
versus effective learning horizon.

## Candidate contribution packages

### Package A — Statistical price of a known counter (primary)

1. First prove the stopped-bias identity and recovery lower bound in the
   appended derivation. Then analyze centered excursion imbalance in a fixed
   base lift; do not presuppose that span must grow.
2. Reject any witness whose apparent \(B\)-dependence is actually placed in a
   transition probability, reward scale, support change, growing base diameter,
   or cost description rather than in the known counter.
3. If such a clean span family exists, prove a Consumption-specific minimax
   lower bound.  Span growth alone is not a regret lower bound.
4. If the construction repeatedly fails, seek a structural upper bound

   \[
   \operatorname{sp}(v_B^\star)\le C H_{\rm reload}
   \]

   where \(H_{\rm reload}\) is independently defined and uniformly bounded in
   \(B\).
5. Match the resulting \(B\)-dependence, or independence, with an upper bound.

Either a \(B\)-dependent or a \(B\)-free characterization is publishable in
principle.  The key is that the conclusion must be minimax-statistical, not an
artifact of one generic span-based upper bound.

### Package B — Exact-planning hardness boundary (supporting result)

Formalize the quantitative-reachability reduction above, including the missing
safe-target restriction.  If it passes, use it to state why general exact
polynomial-in-\(\log B\) planning is not a viable default oracle.
Do not promote this supporting complexity result into the main RL contribution,
and do not apply it to the safely communicating or approximate cases without a
separate reduction.

### Package C — Tractable compact planning subclass (deferred)

Only after Package A yields a statistical story, identify a restricted class in
which approximate optimistic planning admits a compact counter representation.
This must explicitly evade the exact general-class hardness result.

## P0 kill gates

The main gates are S0 -> S1 -> S2. P0 is a nonblocking side question:

| Order | Required result | Kill / routing consequence |
|---|---|---|
| P0 | Formal exact reachability-to-mean-payoff reduction | reduction fails to preserve safety or the relevant objective |
| S0 | Clean capacity/span characterization in a safely communicating class | every \(\Omega(B)\) witness smuggles \(B\) into base dynamics, or the upper bound is only a restatement of generic span |
| S1 | Minimax regret lower bound or a proof of \(B\)-free minimax complexity | span behavior cannot be converted into a statistical statement |
| S2 | Matching or near-matching lifted upper bound | generic reduction leaves an unexplained polynomial gap |
| C0 | Compact approximate planning for the surviving statistical subclass | deferred unless S0--S2 establish a paper-level statistical result |

Do not design a new RL algorithm until S0 is complete and S1 supplies a
genuine information-theoretic statement.

## Fatal flaw

The current fatal risk is:

> After controlling base dynamics independently of capacity, the safely
> communicating Consumption subclass may have \(B\)-free minimax regret and a
> uniformly bounded span, making the statistical result a direct corollary of
> existing linear-mixture RL.  Conversely, an observed \(\Theta(B)\) span may
> fail to imply any \(B\)-dependent minimax regret.

If no clean lower bound, structural upper bound, or minimax separation emerges,
the theoretical direction should be closed.  “No paper has used the name
Consumption MDP” would then be a bibliography gap, not a scientific gap.

## Recommended working title

Do not yet use *Safe Reinforcement Learning in Consumption MDPs* as a committed
title. It overstates what remains open.

Use internally:

> **Does a Known Counter Cost Samples?**

If a characterization survives:

> **Learning with Succinct Replenishing Counters**  
> *The Statistical Price of Known Resource State*

## Boundaries and non-claims

- No claim is made that the literature search proves absolute absence of prior
  work.
- The linear-mixture lift is an exact representation identity, not by itself a
  new learning theorem.
- Known support makes viability computable; it does not guarantee reward
  learnability or Bellman optimality.
- The action-set adaptation and generic unknown-reward extension require formal
  proof before citing the published regret bound verbatim.
- Capacity-independent parameter dimension is not capacity-independent regret
  or computation.
- Bias-span growth is not by itself a minimax regret lower bound.
- Any capacity lower-bound family must keep base dynamics, support, reward
  range, and other horizon/mixing parameters uniformly controlled; otherwise
  it does not isolate the statistical price of the counter.
- The exact-planning NP-hardness transfer concerns the general multichain class.
  It does not establish hardness of approximation or hardness in the safely
  communicating subclass.
- The UAV option/SMDP setting is out of scope until the discrete-time theory
  survives these gates.

## Verified source boundary

- Chae et al. (2025) assume a known feature map, unknown transition parameter,
  Bellman optimality, and an input upper bound on twice the optimal bias span;
  their algorithm is polynomial in the explicit state size and their leading
  regret is controlled by feature dimension and span:
  https://proceedings.mlr.press/v258/chae25a.html
- Consumption-MDP safety depends on all supported successors, sustainable reload
  structure, and safe-action selection; compact counter selectors are established
  for qualitative synthesis:
  https://link.springer.com/chapter/10.1007/978-3-030-53291-8_22
- Exact quantitative reachability and repeated-reachability synthesis for
  Consumption MDPs is NP-hard in the binary-encoded input and the published
  algorithms use flattened MDPs with pruning/quotient reductions:
  https://doi.org/10.1016/j.ipl.2022.106342
- The efficient counter-strategy work is qualitative reachability/Büchi
  synthesis, not an online average-reward regret theorem:
  https://arxiv.org/abs/2105.02099
- Unknown mean-payoff learning under omega-regular constraints already assumes
  known support and, in its main result, a single-end-component structure:
  https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.CONCUR.2018.8
- Almost-sure budget learning already supplies fixed-point and sample-complexity
  machinery for feasibility, but not the proposed average-reward separation:
  https://proceedings.mlr.press/v168/castellano22a.html
- Factored-MDP regret already demonstrates that compact parameterization can
  avoid Cartesian-product statistical dependence in suitable models:
  https://arxiv.org/abs/1403.3741
- Known-model energy-constrained mean-payoff optimization has substantial
  structural and computational complexity; it cannot be summarized as a trivial
  finite-state planning step:
  https://arxiv.org/abs/1607.00678

## Open risks

The latest lemma-specific conditions, counterexamples, and verification scope
are recorded in the appended derivation package below.

1. The reachability-to-mean-payoff reduction still needs a source-definition
   audit: safe-policy quantification, first-hit target modification, and the
   exact expected mean-payoff convention must be matched line by line.
2. A \(B\)-dependent span witness may be impossible without also making a base
   hitting-time or mixing parameter depend on \(B\).  That would favor a
   capacity-free structural theorem rather than a lower bound.
3. Even a clean \(\Theta(B)\) span family may have \(B\)-free minimax regret,
   showing that generic span complexity is loose on this subclass.
4. The small deterministic \(S=2,A=2\) exploratory scan performed during this
   audit did not reveal linear span growth; its largest observed growth
   saturated near a constant.  This is a diagnostic only, not evidence of a
   theorem, because the scan used a restricted template and discounted-value
   approximation.
5. Known support may be physically unrealistic for the UAV option model; that is
   an application limitation, not a reason to weaken the theorem silently.
6. If reward/cost observations are stochastic, both safety and reward
   realizability assumptions must be restated from scratch.

# Derivation Package: Fixed-Base Capacity Lift and Stopped Bias

## Target

Derive the exact relation between optimal Bellman bias and centered excursions;
prove the recovery lower bound without assuming fast mixing of the lift.

## Status

**COHERENT AFTER REFRAMING / EXTRA ASSUMPTION.** A complete handwritten proof
of the stopped identity and recovery inequality follows. The bare excursion
equality needs additional conditions and is false for an arbitrary Bellman
solution even in a communicating, uniformly recoverable model. No machine proof
or independent proof review is claimed. AI-assisted derivation, not novelty.

## Invariant Object

The stopped Bellman decomposition, including terminal bias and Bellman slack.
Excursion duration is not substituted for centered reward imbalance.

## Assumptions

- Fix the same base class and parameter set Theta for every capacity B.
- Fix B, theta temporarily. The viable lifted MDP is finite and has nonempty
  safe action sets and a scalar Bellman optimal gain rho in [0,1].
- A finite function h solves rho+h(x)=max_a[r(x,a)+P_a h(x)].
  Here r is the conditional expected one-step reward in [0,1], and throughout
  this derivation r_t means r(X_t,A_t), NOT a possibly unbounded realized reward.
  A version using realized rewards additionally assumes they lie in [0,1] a.s.
  (or supplies another valid integrable domination).
- The stopped policies are safe, nonanticipating, possibly history-dependent.
- Stop at tau=inf{t>=0: X_t belongs to Q_B}, where Q_B is the set of actual
  reload-boundary augmented states under the frozen timing convention. For
  x in Q_B, tau=0. A cycle starting there uses tau^+ instead.
- For identities at tau, E_x^pi tau<infinity. Uniform recovery requires only
  that at least one such policy exists per x with expectation <=H_R, uniformly
  in B,theta,x. It does not require every policy to be proper.
- The full bias equality additionally requires an integrable Bellman-greedy
  policy reaching Q_B, or integrable policies with total expected slack ->0.

## Notation

Write Pi_hit(x) for the safe policies with finite expected tau. Define

\[
 d_h(x,a)=\rho+h(x)-r(x,a)-P_a h(x)\ge0,
 \quad m_Q=\min_{q\in Q_B}h(q),\quad M_Q=\max_{q\in Q_B}h(q).
\]

Define two different excursion objects:

\[
 G(x)=\sup_{\pi\in\Pi_{\rm hit}(x)}
 E_x^\pi\sum_{t<\tau}(r_t-\rho),
\]

\[
 F_h(x)=\sup_{\pi\in\Pi_{\rm hit}(x)}
 E_x^\pi\left[\sum_{t<\tau}(r_t-\rho)+h(X_\tau)\right].
\]

These are finite from below when recovery exists, and bounded above by the
Bellman inequality. Policies with infinite expected hitting time are excluded:
an infinite centered sum is not automatically a well-defined expectation.

## Derivation Strategy

Bounded-stop telescoping -> dominated convergence for each fixed lift ->
slack characterization -> terminal-bias bounds -> recovery lower bound.

## Derivation Map

1. Bellman slack gives an exact one-step conditional-expectation identity.
2. Sum to tau wedge n, using the predictable indicator t<tau.
3. Pass n->infinity using E tau<infinity and bounded h on a finite lift.
4. Optimize only over Pi_hit; equality requires vanishing total slack.
5. Apply one recovery policy, not a policy-uniform recurrence assumption.

## Main Derivation

### Proposition 1: exact stopped identity

For every pi in Pi_hit(x),

\[
\boxed{
 h(x)=E_x^\pi\left[
 \sum_{t<\tau}(r_t-\rho)+h(X_\tau)
 +\sum_{t<\tau}d_h(X_t,A_t)\right].
}
\]

Proof. Conditional on the history and chosen action,

\[
 E[r_t-\rho+h(X_{t+1})-h(X_t)\mid\mathcal F_t,A_t]
 =-d_h(X_t,A_t).
\]

Multiplying by 1{t<tau wedge n}, summing, and taking expectations gives the
claimed identity with tau replaced by tau wedge n. For fixed B,theta let
L=||h||_infinity. The centered sum is bounded in absolute value by tau, terminal
bias by L, and the slack sum by (1+2L)tau. Dominated convergence is therefore
valid when E tau<infinity. This argument uses no bound on L uniform in B and
does not assume the conclusion of S0. QED.

### Corollary 1: exact criterion for excursion representation

Nonnegative slack yields F_h(x)<=h(x). In fact,

\[
 h(x)-F_h(x)=\inf_{\pi\in\Pi_{\rm hit}(x)}
 E_x^\pi\sum_{t<\tau}d_h(X_t,A_t).
\]

Hence equality holds iff that infimum is zero. An integrable Bellman-greedy
recovery policy is sufficient, but uniform recoverability alone is not.
When equality holds,

\[
 G(x)+m_Q\le h(x)\le G(x)+M_Q.
\]

Only if the terminal reload bias is constant, normalized to zero, does this
reduce to h(x)=G(x). For a singleton reference q, use its hitting time rather
than set hitting, and prove greedy properness to q separately.

### Proposition 2: recovery bounds the negative side

Applying Proposition 1 to a recovery policy with E tau<=H_R gives

\[
 \boxed{h(x)\ge m_Q-\rho H_R\ge m_Q-H_R.}
\]

This conclusion requires neither greedy properness nor excursion equality.
With min_Q h=0, it yields min_x h(x)>=-H_R. Alternatively, a singleton reference
q with h(q)=0 yields the same bound if recovery is uniformly to q, not merely
to some member of Q_B. Consequently,

\[
 \operatorname{sp}(h)\le\max_x(h(x)-m_Q)+H_R.
\]

This localizes any B-growing span to the upper side relative to min_Q h.
It does not yet localize it to bare excursions G: a growing reload-boundary
oscillation M_Q-m_Q or a nonzero slack gap may also matter.

### Counterexample 1: multiple reload states cannot all be normalized to zero

Let both states A,D be reload states, with zero costs, one action, reward 1 at
A and 0 at D, and next-state probabilities (1/2,1/2) from either state. The
chain is communicating, rho=1/2, and every state hits Q immediately, so H_R=0.
The Bellman equations require h(A)-h(D)=1. Normalizing h(A)=0 gives h(D)=-1,
contradicting the proposed bound h>=-H_R relative to that arbitrary reference.
Normalizing min_Q h=0 fixes the bound but not the terminal-bias oscillation.

### Counterexample 2: uniform recovery does not imply equality h=G

Use two base states q (reload), z (nonreload), and two zero-cost actions:
at q, stay has reward 1 and enter z reward 0; at z, stay and return q both have
reward 1. Transitions are deterministic. The safe controlled graph is
communicating and recovery to q takes at most one step. Gain is 1 everywhere.
For every alpha in [0,1], h(q)=0,h(z)=alpha solves the optimal Bellman equation.
But every integrable return excursion from z has centered reward sum zero,
so G(z)=0. For alpha>0, equality fails: greedy stay never returns, and the
proper return action incurs slack alpha. Thus Bellman bias need not be unique
modulo an additive constant even in a communicating controlled MDP.

This example uses zero-cost loops, allowed in the general model but excluded
by some decreasing-CMDP subclasses. It is a warning about the stated general
assumptions, not a claim about those stricter subclasses. If using a canonical
or minimum-span Bellman bias, specify and justify that selection explicitly.

## Remarks and Interpretation

- The user's fixed-base lift is now the active minimax definition. Worst-case
  theta may still depend on B,T within the SAME fixed Theta, as in ordinary
  local testing alternatives; that alone is not illicit model-class drift.
  Any counter-specific claim must still explain why the lift, rather than an
  unconstrained base near-degeneracy, creates the additional difficulty.
- Full replenishment does not reset the base state to one common reference.
  Calling the identity 'stopped bias' avoids prematurely claiming iid
  regenerative cycles across multiple chargers.
- Chae et al.'s span-based result is a generic upper/lower theorem on its own
  class, not a lower bound for this fixed-base subclass. Verified primary source:
  https://proceedings.mlr.press/v258/chae25a.html
- No augmented diameter or mixing-time O(1) hypothesis has been introduced.
- Correct S0: characterize upper stopped centered value AND reload-boundary
  bias oscillation, or first specialize explicitly to a single reload reference
  with greedy properness. Neither route presupposes Omega(B).

## Boundaries and Non-Claims

The stopped identity is elementary Bellman/optional-stopping reasoning, not a
new paper contribution. Uniform recovery proves a one-sided bound, not a
capacity-free full span theorem. No minimax characterization is established.
No new algorithm, Lean expansion, UAV experiment, or formal v1a analysis ran.

## Open Risks

1. Need a B-uniform positive-side bound without assuming lifted mixing.
2. Multi-reload spread is deferred: the active subclass below has one charger.
3. Bias selection and greedy properness are resolved for the active subclass
   below; these remain issues only when returning to the general model.
4. Fixed-class minimax may be infinite/linear without additional learnability
   regularity; uniform recovery to a set alone does not ensure learnability.
5. S1 requires an actual subclass-specific information lower bound even after S0.

# Derivation Package: S-minus-1 Canonical Excursion on the Reachable Domain

## Target

Remove arbitrary Bellman-bias selection and unreachable-state inflation before
S0. Prove properness, canonical excursion representation, normalized uniqueness,
and resource monotonicity in the single-reload decreasing subclass.

## Status

**COHERENT AS STATED under the explicit assumptions below.** Complete
handwritten proofs are given. Exact-rational checks of an example supplement
the proof, but are not machine verification or independent expert review.
This is an AI-assisted foundational derivation, not a novelty claim.

## Invariant Object

The UNIQUE normalized AROE bias on the safe reachable lifted domain D_B, not an
arbitrarily selected solution on the whole viability kernel K_B.

## Assumptions

1. Fix one B-independent Theta and base model class
   M_theta=(S,A,P_theta,r_theta,c,{q}). States and actions are finite; exact
   support E(s,a), integer nonnegative costs, feature map, and parameter/reward
   classes are fixed independently of B. Capacities B>=B_0 are integers.
2. Decreasing means the base supported state-action graph has NO directed cycle
   whose total consumption is zero. It is enough for the proof that this holds
   on cycles avoiding q; the active class adopts the conventional stronger
   definition. Nonnegative costs make a zero-total cycle a zero-edge-cost cycle.
3. Use a normalized charger state q_B=(q,B). For x=(s,b), actions first debit
   c(s,a) from available b; require b>=c(s,a) BEFORE any reload. On transition
   into q record q_B; otherwise record (s',b-c(s,a)). This is the normalized
   decision-state encoding of instantaneous full reload. For the earlier
   source-reload convention, normalize all arrival copies of q before the next
   decision; their same outgoing kernel/reward makes them redundant copies.
   This is not permission to allow an unaffordable transition into the charger.
4. q_B belongs to the greatest robust viability kernel K_B. Safe policies select
   only A_K(x); illegal actions and depleted states are not part of the MDP.
5. D_B=K_B^reach consists of states attainable from q_B along some finite
   supported safe history with POSITIVE probability. This is existential safe
   reachability, not almost-sure reachability. Use the full safe action set at
   every x in D_B. Then every supported safe successor also belongs to D_B.
6. On D_B the scalar optimal gain rho_B and at least one finite AROE bias exist:
   rho_B+h(x)=max_{a in A_K(x)}[r_theta(s,a)+P_a h(x)]. Existence is kept explicit
   for this proposition; the support communication argument below explains why
   the finite-MDP communicating setting is appropriate, without assuming a
   B-independent diameter. Reward means r_t:=r_theta(S_t,A_t) in [0,1].
7. Optional uniform recovery: for each B,theta,x in D_B there is a safe policy
   to q_B with E tau_q<=H_R independent of B,theta,x. This is ONLY needed for
   the negative-side bound, not the four canonical/properness claims.

## Notation

Suppress theta,B inside local proofs. Let

\[
 \tau_q=\inf\{t\ge0:X_t=q_B\},\qquad
 \tau_q^+=\inf\{t\ge1:X_t=q_B\}.
\]

For x=q_B use tau_q=0 in the excursion representation. A nonempty charger
cycle uses tau_q^+ instead. Let Pi_safe(x) be all nonanticipating randomized
history-dependent policies selecting A_K along their histories.

The statistical diagnostics and minimax target are now

\[
 H_B^{\rm reach}=\sup_{\theta\in\Theta}
  \operatorname{sp}_{D_{B,\theta}}(h^\star_{B,\theta}),
 \qquad
 \mathfrak R_B(T)=\inf_{\mathcal A}\sup_{\theta\in\Theta}
  \operatorname{Reg}^{\mathcal A}_{\mathsf{Lift}_B(M_\theta),q_B}(T).
\]

The learner starts at q_B; it must remain safe uniformly over the fixed class.
If exact support is common, the viability and reachable domain are common too.

## Derivation Strategy

Nonreload support is acyclic after resource lifting -> pathwise finite stopping
-> stopped slack identity -> greedy attainment -> uniqueness -> virtual-budget
coupling for within-capacity monotonicity.

## Derivation Map

1. A cycle in the nonreload lift would have zero total consumption.
2. Every safe path therefore hits q within a bound depending on B, NOT an O(1)
   lifted mixing or diameter assumption.
3. All safe policies are integrable; the earlier stopped identity applies.
4. Finite-action Bellman maximizers have zero slack and are also proper.
5. The resulting centered-excursion RHS contains no unknown bias function.
6. Couple more-resource and less-resource paths until their common charger hit.

## Main Derivation

### Proposition S-minus-1a: all safe policies are bounded-proper

The supported lifted graph induced by D_B\{q_B} has no directed cycle.

Proof. Along a hypothetical cycle avoiding q, the battery never resets and
returns to its starting value. Thus the sum of nonnegative action costs along
the projected base closed walk is zero. Each edge cost is zero. Extracting a
simple cycle from that base closed walk contradicts decreasing. QED.

Let n_B=|D_B\{q_B}|. Every safe trajectory avoiding q has no repeated lifted
vertex, so from a nonreload x it hits q in at most n_B steps. Safe action sets
are nonempty and supported successors remain viable; a finite path cannot get
stuck at a dead end. Thus, on every supported safe path,

\[
 \tau_q\le n_B\le (S-1)(B+1)\quad(x\ne q_B),
 \qquad \tau_q^+\le1+n_B\quad(x=q_B).
\]

The bounds hold for every safe history-dependent randomized policy, not only
stationary or greedy ones. They establish bounded integrability for each fixed
B; they do NOT establish H_R=O(1) or a B-free excursion/span bound.

D_B is successor-closed under safe actions by its reachability definition.
Every x in D_B has a supported safe path back to q by bounded properness, and
q has a supported safe path to x by definition. Therefore the safe controlled
support graph on D_B is strongly connected. This is graph communication, not
a bound on the expected time required to reach a rare particular x.

### Proposition S-minus-1b: canonical excursion and unique normalized bias

Given scalar optimal gain rho_B, every AROE solution normalized by h(q_B)=0
satisfies

\[
 \boxed{
 h(x)=\sup_{\pi\in\Pi_{\rm safe}(x)}
 E_x^\pi\sum_{t=0}^{\tau_q-1}(r_\theta(S_t,A_t)-\rho_B).
 }
\]

Proof. Proposition S-minus-1a makes every safe policy admissible for the stopped
identity. Terminal state is q_B and terminal bias is zero. Nonnegative Bellman
slack therefore gives expected centered sum <=h(x) for every safe pi.
Choose a maximizing action at each state; finite nonempty safe action sets
permit a deterministic stationary Bellman-greedy selector. Its slack is zero
at each step, and S-minus-1a proves it reaches q with bounded stopping time.
The stopped identity then gives equality. At q_B both sides are zero since
tau_q=0. QED.

The RHS depends only on rho_B and the model, not h. Two normalized solutions
for the same optimal gain therefore coincide everywhere on D_B. Consequently
AROE bias is unique there up to a constant, and its span is not an arbitrary
selection artifact.

For a general model without this subclass, the fallback invariant is

\[
 H^\star(M;D)=\inf\{\operatorname{sp}_D(h):
                 (\rho^\star,h)\text{ solves AROE on }D\}.
\]

On a finite nonempty AROE solution set this infimum is attained: fix h(x_0)=0,
take a minimizing sequence with bounded span; then all coordinates are bounded,
and a convergent subsequence has an AROE-solving limit by continuity of the
finite Bellman operator. In the active canonical subclass the infimum simply
equals sp_D(h^star); no selection optimization remains.

### Proposition S-minus-1c: within-capacity resource monotonicity

For the SAME B,theta and two domain states x_i=(s,b_i) in D_B with s!=q and
b_2>=b_1,

\[
 \boxed{h^\star_B(s,b_2)\ge h^\star_B(s,b_1).}
\]

Proof. Take any safe lower-budget policy. Execute it from b_2 while tracking
a virtual budget initially b_1. Before q, couple the actions, randomization,
base successors, and reward histories. Base transition probabilities and reward
functions are battery-independent. The real budget exceeds the virtual one by
the fixed nonnegative difference b_2-b_1 until their common first hit of q.
Thus every action affordable to the virtual trajectory is affordable to the
real one. After q both budgets reset to B, and the virtual execution can use
the same safe continuation. This proves an infinite safe policy exists for the
real execution: its chosen actions have viable supported successors. Starting
from an x_2 already reachable, those successors are reachable as well, hence
the real execution stays in D_B.

The two coupled excursions have the same hitting time and centered reward sum;
rho_B is unchanged. Optimizing over lower-budget policies and using canonical
excursion representation proves the inequality. QED.

The notation Pi_safe(s,b_1) subset Pi_safe(s,b_2) is only an embedding via this
virtual-budget controller; the two sets of literal augmented histories differ.
No inference compares B_1 with B_2: changing capacity also changes rho_B.

For each fixed s, a maximum over its reachable battery levels is achieved at
the LARGEST REACHABLE level, which need not equal B. If (s,B) is unreachable,
it cannot be introduced into the span diagnostic. Reachable battery sets need
not be intervals.

### Corollary: recovery still isolates the negative side

If optional B-independent recovery holds, the earlier inequality specializes to

\[
 h^\star_B(x)\ge-\rho_B H_R\ge-H_R,
 \quad
 \operatorname{sp}_{D_B}(h^\star_B)
 \le\max_{x\in D_B}h^\star_B(x)+H_R.
\]

There is now neither terminal reload oscillation nor arbitrary bias slack gap.
S0 genuinely asks whether the positive canonical centered excursion can grow
with B. Properness alone only yields |h(x)|<=n_B, an O(SB) diagnostic bound.

## Remarks and Interpretation

- Exact-rational sanity example within the active subclass: q->w has cost 0,
  reward 0; w->w has cost 1, reward 1; w->q has cost 1, reward 0. Every base
  cycle consumes positively. For B>=1 the reachable domain is q_B and
  w_1,...,w_B. Its AROE gain and canonical bias are

  \[
  \rho_B=\frac{B-1}{B+1},\qquad h(q_B)=0,\qquad
  h(w_b)=(b-1)(1-\rho_B)-\rho_B.
  \]

  Working b-1 times before return attains the excursion supremum; returning
  immediately provides uniform recovery H_R=1. Direct substitution gives
  sp(h)=2 rho_B<2 and battery monotonicity. Fraction-arithmetic checks were
  executed for B=1,2,3,5,20,100; excursion maxima were enumerated for B=1..8.
  This avoids relying on the earlier generic illustrative cycle's zero-cost
  shortcuts when checking a decreasing subclass. No stochastic or minimax
  conclusion is inferred from these deterministic checks.
- Integer consumption is essential to the displayed cardinality bound; for
  positive real consumption one must state a positive minimum cycle cost and
  derive a corresponding finite bound. Do not infer this bound for arbitrary
  random consumption.
- Rewards in the proof are r_theta(S_t,A_t), the expected reward function. A
  version with realized rewards assumes realized reward in [0,1] a.s.; with
  bounded tau, conditional expectation justifies the same expected excursion.
- A canonical bias on D_B aligns the span diagnostic with a legitimate AROE
  solution for a restricted lifted MDP. Applying Chae's learning theorem still
  needs its feature/norm, reward, known-span and safe-action adaptation checks.
  This does not prove a subclass minimax lower bound.
- Primary literature confirms the no-zero-consumption-cycle definition:
  https://link.springer.com/chapter/10.1007/978-3-030-53291-8_22
  Generic bias-span RL reference:
  https://proceedings.mlr.press/v258/chae25a.html
  General bias nonuniqueness reference supplied by the user:
  https://arxiv.org/abs/2209.15141

## Boundaries and Non-Claims

This proves the four foundational claims, not a B-free full-span theorem,
not a monotonicity theorem across capacities, and not minimax statistical
freedom/cost of the counter. S-minus-1 is well-posedness, not the paper novelty.
Neither B-independent launch time to every augmented state nor O(1) lifted
diameter/mixing has been assumed. No algorithms or UAV experiments ran.

## Open Risks

1. S0 must bound positive canonical excursions without importing an O(1)
   augmented hitting-time/diameter assumption in disguise.
2. Removing unreachable states does not remove rare-but-reachable state effects;
   S1 must check whether their bias affects the minimax learning problem.
3. Sup over a fixed Theta can still use B,T-dependent hardest parameters from
   that same class; a claimed counter-specific lower bound needs an explicit
   controlled comparison and information calculation.
4. Known support and viable q_B must hold throughout the fixed class and B range.
5. No independent proof audit or formal Lean proof has been completed.

# Derivation Package: S0 Counter-vs-Rare-Transition Kill Test

## Target

Prove or refute the proposed chain: largest reachable resource representative
-> base-simple safe access path -> positive access probability -> nonpositive
centered charger cycle -> capacity-free canonical span.

## Status

**COHERENT AS STATED with the explicit base/recovery assumptions below.** The
handwritten proof closes all four steps. It excludes unbounded capacity span
under uniform base support probabilities and recovery. It does NOT establish
a minimax equality or a new learning theorem. Exact-rational checks are sanity
checks, not an independent or machine proof. AI-assisted derivation.

## Invariant Object

The unique normalized canonical bias on D_B=K_B^reach, with h(q_B)=0, for a
capacity lift of one fixed B-independent parameter class.

## Assumptions

Retain all S-minus-1 assumptions: finite base S,A, known common exact support,
nonnegative integer battery-independent costs, fixed reward function class in
[0,1], single full reload q, decreasing support, viable q_B, safely reachable
domain closed under all safe successors, scalar optimal gain/AROE existence.
Rewards in stopped sums mean r_t=r_theta(S_t,A_t); a realized-reward version
requires reward samples in [0,1] a.s. and their conditional means r_theta.

Additionally assume safe recovery to q_B: for every B,theta,x in D_B there
EXISTS a safe policy with E_x tau_q<=H_R, where H_R is independent of B,theta.
This is not an assumption that all policies return in O(1) time.

For the class-uniform corollary assume

\[
 p_{\min}(\theta)=\min_{s,a,s'\in E(s,a)}P_\theta(s'|s,a),
 \qquad \inf_{\theta\in\Theta}p_{\min}(\theta)\ge p_0>0.
\]

For a single fixed theta, finite exact support already gives p_min(theta)>0;
the pointwise theorem does not require this additional uniform lower bound.
No augmented diameter, span, mixing time, or access probability to arbitrary
resource levels is assumed O(1).

## Notation

For s represented in D_B and s!=q define

\[
 b_{\max}(s)=\max\{b:(s,b)\in D_B\},\qquad x_s=(s,b_{\max}(s)).
\]

Write rho=rho_B,theta, h=h_B,theta, M=max_D_B h>=0. If M>0 choose a
nonreload representative x_star maximizing h. Let L be the length of its
base-simple supported safe path from q_B, and alpha the product of the true
successor probabilities along that path. Then 1<=L<=S-1 and alpha>0.

## Derivation Strategy

Explicit upward-safe coupling -> delete cycles in a maximal-budget history ->
attempt the resulting path once -> recover on first deviation, exploit canonical
excursion on success -> apply the stopped identity to the whole charger cycle.

## Derivation Map

1. More resource preserves existence of safe continuations and safety of each
   lifted action copied from a lower virtual budget.
2. A repeated base state after the last charger visit consumes positively;
   deleting that segment would reach strictly higher battery, a contradiction.
3. The finite path actions remain globally safe, including off-path successors.
4. Recovery bounds losses on deviation branches; canonical greedy excursion
   gives exactly h(x_star) on the success branch.
5. A complete charger cycle has nonpositive expected centered reward by AROE.
6. Combine the upper positive bound with min h>=-rho H_R.

## Main Derivation

### Lemma S0a: upward-safe continuation (separate from reachability)

For fixed B, if (s,b) belongs to K_B, s!=q and b' in [b,B], then (s,b') belongs
to K_B. Moreover any action safe at (s,b) is safe at (s,b').

Proof. Simulate an infinite safe lower-budget policy from the higher budget
using the virtual budget controller from S-minus-1. Before the next reload,
the real-minus-virtual budget is b'-b>=0. Costs and base transitions do not
depend on battery. Actions remain affordable on every supported trajectory;
after reload both budgets equal B. This is an infinite safe continuation from
(s,b'), so it belongs to K_B. For a safe action at (s,b), every nonreload
supported successor at b-c is viable, and upward closure makes the successor
at b'-c viable; reload successors are the same q_B. Thus the action is safe
at the higher state as well. QED.

Upward viability does NOT imply upward reachability. To prove the latter for
a particular copied path, construct a safe supported history from q_B first.

### Lemma S0b: maximal-resource representatives have base-simple access

Take a finite supported safe history reaching x_s. Keep only its suffix from
the LAST visit to q_B; it has no intermediate q. Suppose that suffix repeats
some nonreload base state. The subwalk between two equal base states contains
a cycle. Decreasing and nonnegative costs imply its total cost kappa>0.

Delete that subwalk. At the splice, the base state is unchanged, and the battery
is higher by kappa. Copy each remaining original suffix action/successor using
Lemma S0a. Every copied action is safe, not merely safe on the chosen successor;
every chosen successor has positive base probability. The shorter history is
therefore a supported SAFE history from q_B. Its final battery is
b_max(s)+kappa>b_max(s). It is still <=B since it is B minus a nonnegative
cost sum. This contradicts maximal reachability.

Thus the suffix contains no repeated base states. It is a q-to-s base-simple
safe path of length L_s<=S-1. QED.

Within-capacity monotonicity now implies that a positive global maximum M is
attained at some such x_s. If M=0, the negative recovery bound alone suffices.

### Lemma S0c: any integrable complete charger cycle is centered-nonpositive

Apply the earlier stopped Bellman identity starting at q_B, but stop at
tau_q^+, not tau_q=0. S-minus-1 bounds tau_q^+ for every safe policy.
Both initial and terminal bias equal zero, so

\[
 E_{q_B}^\pi\sum_{t<\tau_q^+}(r_t-\rho)
 =-E_{q_B}^\pi\sum_{t<\tau_q^+}d_h(X_t,A_t)\le0.
\]

This establishes cycle optimality from AROE; no iid-renewal theorem or exchange
of an infinite-horizon gain limit with a hitting time is needed.

### Theorem S0: positive bias and full span are capacity-free

If M>0, take Lemma S0b's path to x_star, of length L. Attempt its prescribed
actions until the first successor deviation or until successful completion.
These actions are safe at every expected on-path state. On a first deviation,
the successor remains in D_B by safe successor closure. If it is q_B the cycle
ends; otherwise switch to an H_R recovery policy selected for that successor.
On successful completion switch to a canonical Bellman-greedy return policy
from x_star; its expected centered excursion is exactly M.

This defines a safe nonanticipating policy from q_B until its first positive
return. On-path vertices other than the start avoid q. The success event A has
probability alpha=product of the path transition probabilities. Off-path
recoveries obey the expectation bound even after conditioning on the observed
deviation, since an appropriate recovery policy is selected at that state.

Let sigma be the length of the attempted prefix, stopped on deviation or
success. Then sigma<=L. Its centered sum is bounded below by -rho sigma,
because r_t>=0. Conditional on failure, additional recovery centered reward is
at least -rho H_R in expectation; on success the expected additional centered
reward is M. Thus

\[
\begin{aligned}
0&\ge E\sum_{t<\tau_q^+}(r_t-\rho)\\
 &\ge -\rho E\sigma+\alpha M-(1-\alpha)\rho H_R\\
 &\ge -\rho L+\alpha M-(1-\alpha)\rho H_R.
\end{aligned}
\]

Consequently

\[
 \boxed{M\le\rho\frac{L+(1-\alpha)H_R}{\alpha}.}
\]

Together with min h>=-rho H_R this gives

\[
 \boxed{\operatorname{sp}_{D_B}(h)
 \le\rho\frac{L+H_R}{\alpha}
 \le\frac{S-1+H_R}{p_{\min}(\theta)^{S-1}}.}
\]

The last step uses alpha>=p_min(theta)^L>=p_min(theta)^(S-1) and rho<=1.
If M=0, span<=rho H_R also satisfies the final bound; if S=1, the normalized
reachable domain contains only q_B and span is exactly zero. QED.

The probability lower bound is DERIVED from base regularity and maximal-budget
path simplification, not assumed for arbitrary augmented target states.

### Corollary S0: uniform regular bases exclude capacity span growth

Under the fixed-class assumptions p_min>=p_0 and uniform H_R,

\[
 \boxed{\sup_{B\ge B_0,\theta\in\Theta}
 \operatorname{sp}_{D_{B,\theta}}(h^\star_{B,\theta})
 \le C_0:=\frac{S-1+H_R}{p_0^{S-1}}<\infty.}
\]

Hence an Omega(B) canonical-span counterexample satisfying ALL these conditions
is impossible. For each fixed theta with uniform recovery, the bound is also
uniform in B without a class-wide p_0; it may diverge as theta varies.

## Exact Example: pointwise B-free is not class-uniform B-free

Use three fixed base states q,w,z. From q the sole action has cost 1, reward 0,
and moves to w with probability p in (0,1), or to z with probability 1-p.
At w, work costs 1, rewards 1, and returns to w; return costs 1, rewards 0,
and moves to q. At z, the sole action costs 1, rewards 0, and moves to q.
The fixed parameter class is Theta=(0,1); all costs, rewards and supports are
unchanged across B. Take B>=2.

The reachable domain is q_B, w_1,...,w_(B-1), and z_(B-1). All base cycles
consume positively. Immediate return from w or z gives H_R=1. Define

\[
 k=B-2,\quad \rho=\frac{p k}{2+p k},\quad
 h(q_B)=0,\quad h(z_{B-1})=-\rho,\quad
 h(w_b)=(b-1)(1-\rho)-\rho.
\]

Substitution into every safe AROE equation verifies this gain/bias. Work is
greedy whenever b>=2; return is required at b=1. Thus

\[
 \operatorname{sp}(h)=\frac{2(B-2)}{2+p(B-2)}\le\frac2p.
\]

The maximal-bias access path q->w has L=1, alpha=p, and M=rho(2/p-1).
The theorem's path-specific positive bound is an equality in this example:
M=rho[1+(1-p)H_R]/p. It demonstrates why failure-branch recovery cannot be
discarded, and why base rarity is a real constant in the bound.

For every fixed p, span remains bounded as B->infinity. But for the SAME fixed
class Theta=(0,1), sup_p sp(h)=B-2 (a supremum, not an attained maximum).
Choosing p=1/B gives sp(h)=2B(B-2)/(3B-2)=Theta(B). This is legitimate
fixed-class minimax parameter selection, yet the growing span accompanies a
vanishing base transition probability. It is NOT a counter-specific regret
lower bound. A uniform p_0>0 removes this example's linear capacity span.

## Remarks and Interpretation

- Exact-arithmetic sanity audit: 438 Python Fraction checks passed for the
  three-state example (p=1/2,1/4,1/10; B=2,3,5,10,100, plus p=1/B cases).
  Checks cover every reachable-state AROE, the cycle gain, positive-bias bound
  equality, and span formula. This is not verification of the general proof.
- This completes S0 for the explicitly regular subclass, not for all base
  classes. It is not a necessary-and-sufficient regularity characterization.
- The path and greedy return used in the proof can depend on the true model.
  They are analytical witness policies, not a claimed implementable learning
  algorithm that knows p or h. Safety masks still depend only on known support.
- The factor p_0^(-(S-1)) can be exponentially poor in S. A B-free result is
  not a near-optimal dependence on base reachability/recovery complexity.
- The main mechanism is more-resource feasibility plus bounded-length access
  to bias-maximizing representatives. This does not imply O(1) diameter to
  EVERY reachable resource state; low battery levels may require long paths.
- Combining C_0 with the shared-parameter lift and Chae et al. conditionally
  gives a leading B-free regret UPPER bound. Its norm/reward assumptions,
  known span input H>=2 C_0, and safe-action proof adaptation still need checking
  before claiming a full learning theorem. A matching minimax lower bound is
  not obtained merely from this span bound.
- Primary sources checked for background only, not as sources of this proof:
  https://link.springer.com/chapter/10.1007/978-3-030-53291-8_22
  https://proceedings.mlr.press/v258/chae25a.html

## Boundaries and Non-Claims

S0's regular-class capacity-growing canonical-span hypothesis is refuted by the
upper bound. This does not prove exactly equal minimax regret at different B,
nor zero multiplicative constants, nor a matching statistical/computational
separation. General mean-payoff NP-hardness remains an unclosed side reduction.
No new algorithm, UAV run, training, CONFIRM access or formal Gate analysis ran.
No independent proof audit or Lean probability verification is claimed.

## Open Risks

1. Need independent audit of cycle deletion with stochastic off-path successors.
2. S1 must distinguish a B-free achievable regret upper bound from a matching
   class-specific minimax rate; degenerate reward classes may have zero regret.
3. Existing property-targeted hitting/span bounds may make this structural lemma
   a direct corollary; no theorem-level novelty claim has been established.
4. Bound blowup as p_0 shrinks is not shown minimax-optimal by the example.
5. Uniform support and recovery may be strong application assumptions; do not
   transfer the result to UAV option/POMDP dynamics without a separate model.
