# Recurrence-Tail Derivation Package

## Target

Construct a mathematically honest theory line for a recurrence-tail history
abstraction.  The immediate target is not a finished theorem.  It is to fix the
objects needed to answer the following question:

> When may two partially observed histories be merged without changing the
> recurrence class of the controlled return process, and when does the proposed
> tail discrepancy reduce to ordinary one-dimensional Wasserstein distance?

The intended result type is a sequence of definitions, exact identities, local
propositions, counterexamples, and proof obligations.  No novelty is claimed;
only the entries explicitly listed in the formal-verification ledger are
claimed to have passed machine checking.

## Status

**COHERENT AFTER REFRAMING / PARTIALLY MACHINE VERIFIED**

The earlier absolute policy-uniform uniform-integrability condition is removed
from the abstraction definition.  It is replaced by a pairwise discrepancy
between histories that the abstraction proposes to merge.  Positive-throughput
claims are postponed until the cycle process and its ergodic assumptions are
fixed.

The deterministic tail-sequence core has now been checked in Lean 4.35.0-rc1
with mathlib.  This does not yet verify the probability tail-sum identity, the
defective generating-function identity, the controlled-policy quantifiers, or
any Markov recurrence theorem.

## Invariant Object

The invariant object is the **policy-conditioned return-time law from a legal
history**, including the possibility of non-return.  Its survival sequence is
the primary representation:

\[
S_h^\pi(t)
:=
\Pr_h^\pi(\tau_{\mathcal R}^{+}>t),
\qquad t\in\mathbb N_0.
\]

The abstraction question compares these survival sequences only for histories
that are merged by the same recursively updateable representation.

## Assumptions

### A. Controlled latent process

At option or recharge decision epoch \(k\), let:

- \(X_k\) be the latent state;
- \(A_k\in\mathcal A\) be the selected option;
- \(T_k\in\mathbb N\) be its holding time;
- \(Y_k\) be its bounded task yield;
- \(O_{k+1}\) be the next observation.

A standard latent controlled Markov-renewal model factorizes as

\[
K(dx',dt,dy\mid x,a)
\]

and

\[
Q(do\mid x',t,y,a).
\]

The legal history \(H_k\) is used to form a belief or predictive state.  The
primitive kernel is not declared to depend on the full history.  If future work
requires a kernel \(K(\cdot\mid h,a)\), the model must instead be called a
history-dependent controlled semi-Markov process.

### B. Return time

Let \(\mathcal R\) denote the recharge decision set and define

\[
\tau_{\mathcal R}^{+}
:=
\inf\{t\ge 1:H_t\in\mathcal R\}
\in\mathbb N\cup\{\infty\}.
\]

The convention is \(\gamma^\infty=0\) for \(0<\gamma<1\).

### C. Representation and policy class

The representation is recursively updateable:

\[
Z_k=\Phi(H_k),
\qquad
Z_{k+1}=U(Z_k,A_k,O_{k+1},T_k,Y_k).
\]

The policy class \(\Pi_{\rm adm}\) must be fixed independently of the pair of
histories being compared.  It may be restricted by a separately defined safety
or option-admissibility contract.  Defining the policy class through the desired
recurrence conclusion would be circular.

### D. Measurability and conditional laws

All conditional probabilities below are assumed to exist as regular
conditional probabilities.  Any final theorem must state the measurable spaces
and conditions that guarantee this.

## Notation

- \(h,h'\): legal histories at a decision epoch.
- \(\Phi(h)\): recursively maintainable abstract state.
- \(\pi\in\Pi_{\rm adm}\): an admissible history or abstract policy.
- \(\tau\): shorthand for \(\tau_{\mathcal R}^{+}\).
- \(S_h^\pi(t)=\Pr_h^\pi(\tau>t)\): return-time survival sequence.
- \(p_h^\pi=\Pr_h^\pi(\tau<\infty)\): eventual-return probability.
- \(G_h^\pi(\gamma)=\mathbb E_h^\pi[\gamma^\tau
  \mathbf 1\{\tau<\infty\}]\): defective return-time probability generating
  function.
- \(D_{\rm rec}^\pi(h,h')\): pairwise recurrence-tail discrepancy.
- \(D_{\rm rec}^\Phi\): worst pairwise discrepancy over histories merged by
  \(\Phi\) and policies in \(\Pi_{\rm adm}\).

## Derivation Strategy

1. Start from the defective return-time law rather than from average reward.
2. Express eventual return, mean return time, and the generating function using
   the same survival sequence.
3. Define a pairwise history-merging discrepancy on that sequence.
4. determine exactly what finiteness or smallness of the discrepancy implies.
5. Compare the discrepancy with one-dimensional \(W_1\), explicitly separating
   proper finite-moment laws from defective or infinite-moment laws.
6. Isolate what remains representation-specific: policy quantification,
   recursive updateability, strict compression, and finite-data certification.

## Derivation Map

1. Tail-sum identities determine return probability and expected return time.
2. The proposed \(\ell_1\) tail discrepancy bounds differences of finite mean
   return times and prevents a positive/null recurrence mismatch.
3. Summation by parts connects the same discrepancy to the behavior of
   \(G(\gamma)\) as \(\gamma\uparrow1\).
4. On \(\mathcal P_1(\mathbb R_+)\), the discrepancy is ordinary \(W_1\).
5. Outside \(\mathcal P_1\), the standard finite-valued \(W_1\) formulation is
   unavailable or must be replaced by an explicitly extended object.
6. These steps alone are classical.  A publishable result would still need a
   nontrivial recursive representation theorem, a strict-compression result, or
   a finite-sample policy-uniform certification result.

## Main Derivation

### Step 1: Tail identities

For \(\tau\in\mathbb N\cup\{\infty\}\), define

\[
S(t)=\Pr(\tau>t).
\]

**Identity 1 (eventual return).**

\[
\lim_{t\to\infty}S(t)=\Pr(\tau=\infty)=1-p.
\]

**Identity 2 (tail sum).**  In the extended nonnegative reals,

\[
\mathbb E[\tau]
=
\sum_{t=0}^{\infty}S(t).
\]

Consequently, conditional on \(p=1\):

\[
\sum_{t=0}^{\infty}S(t)<\infty
\quad\Longleftrightarrow\quad
\mathbb E[\tau]<\infty.
\]

These are exact identities, not representation-learning results.

### Step 2: Pairwise tail discrepancy

For a fixed policy \(\pi\), define

\[
D_{\rm rec}^{\pi}(h,h')
:=
\sum_{t=0}^{\infty}
\left|
S_h^\pi(t)-S_{h'}^\pi(t)
\right|
\in[0,\infty].
\]

For an abstraction \(\Phi\), define

\[
D_{\rm rec}^{\Phi}
:=
\sup_{\substack{\Phi(h)=\Phi(h')\\
                  \pi\in\Pi_{\rm adm}}}
D_{\rm rec}^{\pi}(h,h').
\]

This is pairwise: it measures the cost of a proposed merge.  It does not assume
that every policy in \(\Pi_{\rm adm}\) is positive recurrent.

### Step 3: Recurrence-class preservation

**Candidate Proposition 1.**  If

\[
D_{\rm rec}^{\pi}(h,h')<\infty,
\]

then:

1. \(p_h^\pi=p_{h'}^\pi\);
2. if this common probability is one, then
   \(\mathbb E_h^\pi[\tau]<\infty\) if and only if
   \(\mathbb E_{h'}^\pi[\tau]<\infty\).

**Verification status.**  The deterministic sequence statements underlying
both items have been machine checked.  Absolute summability of \(S-T\) implies
that \(S\) is summable if and only if \(T\) is summable.  If \(S\to a\) and
\(T\to b\), it also implies \(a=b\).  Applying these lemmas to survival
functions still requires the standard probability facts
\(S(t)\to\Pr(\tau=\infty)\) and
\(\mathbb E\tau=\sum_tS(t)\), which have not yet been formalized here.

Thus, finiteness is intended to preserve the three operational classes:

\[
\begin{array}{ll}
\text{transient/non-returning:} & p_h^\pi<1,\\
\text{null recurrent:} & p_h^\pi=1,\ \mathbb E_h^\pi[\tau]=\infty,\\
\text{positive recurrent:} & p_h^\pi=1,\ \mathbb E_h^\pi[\tau]<\infty.
\end{array}
\]

This is an operational return-set classification.  It must not be silently
identified with irreducible-chain state recurrence without the additional
Markov and irreducibility assumptions.

### Step 4: Finite-mean perturbation bound

If both expectations are finite, then the tail-sum identity gives the candidate
bound

\[
\left|
\mathbb E_h^\pi[\tau]
-
\mathbb E_{h'}^\pi[\tau]
\right|
\le
D_{\rm rec}^{\pi}(h,h').
\]

The corresponding deterministic tail-sum inequality has been machine checked:
for summable real sequences \(S,T\),

\[
\left|\sum_t S(t)-\sum_t T(t)\right|
\le \sum_t|S(t)-T(t)|.
\]

Its interpretation as an expectation bound still depends on the unformalized
probability tail-sum identity.

If both expectations are infinite, the expression on the left is not defined
as a real-number difference.  Therefore \(D_{\rm rec}\le\epsilon\) does not by
itself yield a mean-return perturbation statement for null-recurrent pairs.

### Step 5: Generating-function boundary

Define

\[
G_h^\pi(\gamma)
=
\mathbb E_h^\pi
[\gamma^\tau\mathbf1\{\tau<\infty\}],
\qquad 0<\gamma<1.
\]

**Candidate Identity 3 (summation by parts).**

\[
1-G_h^\pi(\gamma)
=
(1-\gamma)
\sum_{t=0}^{\infty}
\gamma^tS_h^\pi(t).
\]

For two histories, this would imply

\[
\left|
G_h^\pi(\gamma)-G_{h'}^\pi(\gamma)
\right|
\le
(1-\gamma)D_{\rm rec}^{\pi}(h,h').
\]

Hence

\[
D_{\rm rec}^{\pi}(h,h')\le\epsilon
\quad\Longrightarrow\quad
\limsup_{\gamma\uparrow1}
\frac{|G_h^\pi(\gamma)-G_{h'}^\pi(\gamma)|}{1-\gamma}
\le\epsilon.
\]

This is an \(O(1-\gamma)\) statement, not an \(o(1-\gamma)\) statement.
The little-o condition would force equality of finite mean return times when the
boundary derivatives exist and is therefore too strong for approximate
recurrence preservation.

**Non-equivalence warning.**  The reverse implication is not established and
is not expected without extra assumptions.  The generating-function difference
sums signed tail differences before taking an absolute value, whereas
\(D_{\rm rec}\) sums their absolute values.  Cancellation may therefore make
the boundary ratio small while \(D_{\rm rec}\) is large or infinite.

There is already a finite-support counterexample.  Let \(\tau_A=2\) almost
surely and let \(\tau_B\in\{1,3\}\) with probability \(1/2\) each.  Both means
equal two, while

\[
D_{\rm rec}(A,B)=\tfrac12+\tfrac12=1.
\]

Their generating functions satisfy

\[
G_A(\gamma)-G_B(\gamma)
=
\gamma^2-\tfrac12\gamma-\tfrac12\gamma^3
=
-\tfrac12\gamma(1-\gamma)^2.
\]

Therefore

\[
\frac{|G_A(\gamma)-G_B(\gamma)|}{1-\gamma}
=\tfrac12\gamma(1-\gamma)\to0,
\]

despite nonzero tail discrepancy.  The two nonzero discrepancy contributions
and the polynomial factorization have been machine checked; the complete
probability-law encoding has not.  This proves mathematically that even a little-o
boundary difference does not recover equality of return-time laws or zero
tail discrepancy; in the finite-mean case it detects equality of first moments
only.

### Step 6: Wasserstein reduction

Let \(\mu_h^\pi\) and \(\mu_{h'}^\pi\) be proper probability laws on
\(\mathbb N\), both with finite first moments.  The one-dimensional transport
identity suggests

\[
W_1(\mu_h^\pi,\mu_{h'}^\pi)
=
\sum_{t=0}^{\infty}
\left|
S_h^\pi(t)-S_{h'}^\pi(t)
\right|
=
D_{\rm rec}^{\pi}(h,h').
\]

Accordingly, on the proper finite-first-moment domain
\(\mathcal P_1(\mathbb R_+)\), the proposed discrepancy reduces exactly to a
standard \(W_1\) distance.  No novelty may be claimed for this reduction.

If \(\Pr(\tau=\infty)>0\) or \(\mathbb E[\tau]=\infty\), the law is not an
ordinary element of \(\mathcal P_1(\mathbb R_+)\).  In that regime,
\(D_{\rm rec}\) should be described as an **extended tail discrepancy** unless
a precise extended optimal-transport space and cost are supplied.  Calling it
standard \(W_1\) without this qualification would be incorrect.

### Step 7: Throughput is a separate layer

Suppose, only for an initial simplified theorem, that cycles
\((Y_k,T_k)\) are i.i.d., \(0\le Y_k\le Y_{\max}\), and

\[
0<\underline m
\le
\mathbb E[T_k]
<\infty.
\]

Under the hypotheses of the renewal-reward theorem,

\[
\rho
=
\lim_{t\to\infty}\frac{N(t)}{t}
=
\frac{\mathbb E[Y_k]}{\mathbb E[T_k]}
\qquad a.s.
\]

For two finite-mean cycle laws, ordinary ratio perturbation gives a candidate
bound of the form

\[
|\rho_h^\pi-\rho_{h'}^\pi|
\le
\frac{|\mathbb E_h^\pi[Y]-\mathbb E_{h'}^\pi[Y]|}{\underline m}
+
\frac{Y_{\max}}{\underline m^2}
D_{\rm rec}^{\pi}(h,h').
\]

This bound is not a main contribution: under the stated assumptions it is a
ratio perturbation consequence.  For non-i.i.d. controlled cycles, an a.s.
throughput claim requires a separately stated Markov-renewal, stationary
ergodic, or martingale law of large numbers.

## Remarks and Interpretation

- The old absolute modulus
  \(\sup_{h,\pi}\mathbb E[(\tau-H)_+]\) is a global regularity assumption.  It
  is not itself a criterion for deciding whether a particular pair of histories
  may be merged.
- Pairwise \(D_{\rm rec}\) correctly targets merge error, but on proper
  finite-mean laws it is exactly one-dimensional \(W_1\).  The metric alone is
  therefore not a Spotlight-level contribution.
- The potentially nonclassical work begins only after adding all of:
  policy quantification, recursive history state, partial observability, strict
  compression, and finite-data certification.
- A fixed-policy result is insufficient for control.  A supremum over all
  imaginable policies may be vacuous.  The admissible policy class must be
  independently motivated and neither too narrow nor circularly stable.

## Boundaries and Non-Claims

- The deterministic sequence core of Candidate Proposition 1 has been checked
  in Lean; its probability and controlled-process lifting has not.
- Candidate Identity 3 has not yet been formalized or checked in Lean.
- This document does not claim that \(D_{\rm rec}\) is novel.
- This document does not claim an equivalence between pairwise tail discrepancy,
  uniform integrability, and the generating-function boundary condition.
- This document does not claim that finite data can certify an infinite tail
  without structural assumptions.
- This document does not claim a strict compression relative to causal states,
  probabilistic bisimulation, weak lumpability, or option/SMDP abstractions.
- This document does not treat the current UAV branch guard as an infinite-time
  non-return event.
- This document does not transfer the i.i.d. renewal-reward formula to the
  repository environment, where latent world properties need not reset after
  recharge.

## Formal Verification Ledger

The project is at `theory/lean`, pinned to Lean 4.35.0-rc1 and mathlib
v4.35.0-rc1.  The checked source is
`theory/lean/RecurrenceTail/TailSequence.lean`.

Verified with no `sorry` or `admit`:

1. absolute summable tail difference preserves summability in both directions;
2. absolute summable tail difference plus existence of both limits forces equal
   limits;
3. the difference of two finite tail sums is bounded by the absolute tail
   discrepancy;
4. geometric weighting for \(0\le\gamma\le1\) does not amplify that discrepancy;
5. multiplication by \(1-\gamma\) gives the corresponding linear boundary-rate
   bound;
6. the two nonzero tail-discrepancy terms and generating-function
   factorization used by the finite-support cancellation counterexample.

Not yet verified:

1. the measure-theoretic tail-sum and defective generating-function identities;
2. laws with mass at \(\tau=\infty\) represented directly in `ENNReal`;
3. the exact one-dimensional \(W_1\) identity and its domain restrictions;
4. policy-uniform lifting, recursive realizability, strict compression, or
   finite-sample certification;
5. any renewal-reward or controlled Markov recurrence conclusion.

## Open Risks

1. **Wasserstein reduction:** all finite-mean quantitative conclusions may be
   textbook \(W_1\) consequences.
2. **Weak lumpability:** existing denumerable-chain aggregation theory may
   already provide the relevant recurrence preservation result.
3. **Policy quantification:** a useful policy-uniform condition may force full
   controlled-kernel preservation or become impossible to certify.
4. **Recursive realizability:** equality or bounded discrepancy of return-time
   laws need not admit a finite-dimensional recursive state update.
5. **Strict compression:** obvious nuisance-variable examples do not establish
   meaningful compression.
6. **Finite-sample tails:** without parametric tails, known drift structure,
   censoring assumptions, or lower/upper hazard envelopes, no finite rollout
   can certify unseen heavy-tail behavior.
7. **Controlled throughput:** history-dependent policies and persistent latent
   regimes invalidate an i.i.d. cycle proof unless a valid ergodic structure is
   established.

## Proof and Falsification Queue

The former queue is superseded by the lumpability kill gate in
`docs/LUMPABILITY_COLLAPSE_KILL_GATE_20260916.md`.

That gate established the following dichotomy:

1. policy-uniform return-law preservation plus pathwise recursive updating can
   be strictly weaker than controlled lumpability, but need not define an
   autonomous abstract control process;
2. requiring a representative-independent controlled semi-Markov kernel makes
   the condition equivalent to strong controlled lumpability of the
   history-state process;
3. the available strictness witness is property-specific/nuisance compression,
   while passage-time-preserving semi-Markov aggregation already exists in the
   prior literature.

Therefore the current exact recurrence-tail abstraction is **not** promoted to
a central theorem, and further Lean probability lifting is paused.  Any future
mathematical work requires a new, separately approved target: either a
finite-sample certification theorem under an explicit tail/drift model class,
or a property-targeted abstraction theorem demonstrably outside existing
passage-time aggregation and trace-equivalence results.
