# Lumpability Collapse Kill Gate

**Date:** 2026-09-16  
**Scope:** one theorem-level falsification round only; no learning algorithm,
UAV implementation, or new Lean development.  
**Status:** **the broad collapse conjecture is false, but the proposed exact
abstraction story does not survive the novelty gate.**

## 1. Target

We test the following research question.

> Can a recursively updateable history representation preserve, uniformly over
> an independently specified admissible policy class, the return-time
> information needed for recurrence and throughput while being strictly weaker
> than controlled semi-Markov lumpability?

Two distinct meanings of "representation" must be separated:

1. **pathwise recursive encoding:** the next code can be computed from the
   current code, action, and next observation;
2. **autonomous control state:** the conditional law of the next code, holding
   time, and yield is determined by the current code and action, independently
   of the history represented by that code.

The first does not imply the second.  This distinction determines the result of
the kill test.

## 2. Model and notation

Let \(\mathcal H\) be the histories at controlled renewal decision epochs.  For
\(h\in\mathcal H\), action \(a\), next observation \(O^+\), holding time
\(T\), and cycle yield \(Y\), let the environment induce

\[
  P(dh^+,dT,dY,dO^+\mid h,a).
\]

Let \(\mathcal R\) be a designated renewal/return set and

\[
  \tau_{\mathcal R}^+
  :=\inf\{t>0:X_t\in\mathcal R\}.
\]

A history abstraction is a measurable map
\(\Phi:\mathcal H\to\mathcal Z\).

### 2.1 Pathwise recursive updateability

\(\Phi\) is pathwise recursively updateable if a measurable \(U\) exists such
that, on every realizable transition,

\[
  \Phi(h^+)=U\bigl(\Phi(h),a,O^+,T,Y\bigr).
  \tag{R}
\]

Condition (R) is computational.  It does **not** say that histories with the same
code induce the same distribution over the next code.

### 2.2 Policy-uniform return-property equivalence

For an admissible policy class \(\Pi_{\rm adm}\), define

\[
 h\equiv_{\rm ret}h'
 \quad\Longleftrightarrow\quad
 \mathcal L_h^\pi(\tau_{\mathcal R}^+,Y_{\rm cyc})
 =
 \mathcal L_{h'}^\pi(\tau_{\mathcal R}^+,Y_{\rm cyc})
 \quad\forall\pi\in\Pi_{\rm adm}.
 \tag{P}
\]

The yield can be omitted when only recurrence is studied.  Equality in (P)
implies zero return-tail discrepancy for every policy, hence preservation of
every *single-cycle* functional determined by this joint law whenever it is
defined.  A long-run throughput conclusion additionally requires that the
equivalence be maintained after renewal, or another valid regenerative/ergodic
argument; it does not follow from a single-cycle marginal alone.

### 2.3 Autonomous abstract-kernel sufficiency

The representation is an autonomous semi-Markov control state if a kernel
\(K_{\mathcal Z}\) exists such that

\[
 \mathcal L\!\left(
   \Phi(H^+),T,Y\mid H=h,A=a
 \right)
 =K_{\mathcal Z}(\cdot\mid \Phi(h),a)
 \tag{K}
\]

for every history-action pair.  Equivalently, whenever
\(\Phi(h)=\Phi(h')\),

\[
 \mathcal L\!\left(\Phi(H^+),T,Y\mid h,a\right)
 =
 \mathcal L\!\left(\Phi(H^+),T,Y\mid h',a\right)
 \quad\forall a.
 \tag{L}
\]

Condition (L) is precisely strong controlled lumpability of the history-state
semi-Markov process with respect to the partition induced by \(\Phi\), after
including holding time and yield in the transition label.

## 3. Proposition 1: the unrestricted collapse conjecture is false

### Construction

Consider a finite controlled Markov process with states

\[
  \{s,s',u,v,r\}, \qquad \mathcal R=\{r\},
\]

actions \(\{a,b\}\), unit holding times, and transitions

\[
\begin{array}{c|cc}
  &a&b\\ \hline
s &u&v\\
s'&v&u
\end{array},
\qquad
u\to r,
\qquad
v\to r,
\tag{C1}
\]

where the transitions from \(u\) and \(v\) do not depend on the action.  Assign
the same cycle yield on all four two-step paths, and let \(r\) start every next
cycle through the same fixed transition (for example \(r\to s\)).

Define \(\Phi(s)=\Phi(s')=z_0\), while \(u\), \(v\), and \(r\) retain distinct
codes.  The next observation identifies \(u\) or \(v\), so (R) holds.

### Claim

For every history-dependent policy \(\pi\),

\[
 \mathcal L_s^\pi(\tau_r^+,Y_{\rm cyc})
 =
 \mathcal L_{s'}^\pi(\tau_r^+,Y_{\rm cyc})
 =\delta_{(2,y)},
 \tag{C2}
\]

but \(\Phi\) is not controlled-lumpable.

### Verification

Equation (C2) is independent of the selected action: every path reaches \(r\)
in exactly two steps with the same yield.  However, under action \(a\),

\[
 P(\Phi(H^+)=\Phi(u)\mid s,a)=1,
 \qquad
 P(\Phi(H^+)=\Phi(u)\mid s',a)=0.
\]

Thus (L) fails, with one-step total-variation distance equal to one, even though
the complete return-time law has discrepancy zero.

### Consequence

\[
 \boxed{
   \text{policy-uniform return-law preservation}
   \not\Rightarrow
   \text{controlled lumpability}
 }
\]

when "recursive representation" means only (R).  The same example refutes any
generic approximate implication: its return-law error is \(0\), while its
one-step controlled-kernel error is maximal.

## 4. Proposition 2: autonomous control-state sufficiency collapses

### Statement

For the history-state semi-Markov process, the following are equivalent at the
kernel-identification level.  We assume a finite/countable quotient, or more
generally a measurable quotient admitting the regular conditional kernels and
measurable representative/section needed below.

1. A representative-independent abstract controlled semi-Markov kernel
   \(K_{\mathcal Z}\) satisfying (K) exists.
2. For every \(h,h'\) with \(\Phi(h)=\Phi(h')\), condition (L) holds for every
   action.
3. The partition induced by \(\Phi\) is strongly controlled-lumpable for the
   labelled kernel containing next abstract state, holding time, and yield.

### Proof

**1 implies 2.** Both conditional laws equal
\(K_{\mathcal Z}(\cdot\mid\Phi(h),a)\).

**2 implies 1.** For \(z\in\mathcal Z\), select any representative
\(h_z\in\Phi^{-1}(z)\) and define

\[
 K_{\mathcal Z}(\cdot\mid z,a)
 :=
 \mathcal L\!\left(\Phi(H^+),T,Y\mid h_z,a\right).
\]

Condition (L) makes this definition independent of the representative.  The
standing measurability assumption makes it a valid kernel; without that
assumption, the setwise argument alone does not establish measurability on an
arbitrary quotient space.

**2 iff 3.** This is the defining equal-block-kernel condition for strong
controlled lumpability, with \((T,Y)\) included in the transition mark.

### Consequence

Pathwise updateability does not give a control state.  If we add the condition
needed to run a Markov/semi-Markov controller solely on \(z\), the proposed
exact representation condition becomes controlled lumpability rather than a
strictly weaker alternative.

## 5. The structural dichotomy

The kill test yields a sharp two-branch result.

| Representation contract | Strictly weaker than lumpability? | Deployable as an autonomous Markov/semi-Markov control state? | Novelty status |
|---|---:|---:|---|
| Return/property trace equivalence + pathwise update (P)+(R) | Yes | Not in general | Classical property/trace aggregation territory |
| Representative-independent abstract kernel (K) | No | Yes | Controlled lumpability by construction |

The five-state example lives in the first row.  It is mathematically valid but
does not produce the desired representation theorem: the abstract transition
law from \(z_0\) depends on whether its hidden representative is \(s\) or
\(s'\).

## 6. Prior-work kill matrix

This matrix reports only what could be established from primary abstracts or
accessible primary manuscripts.  A blank or "not established" entry is not a
claim that the feature is absent.

| Object | Serfozo (1971) | Sumita--Rieders (1988) | Bradley (2002) | Guenther et al. (2011) | Current proposed object |
|---|---|---|---|---|---|
| Base process | semi-Markov | semi-Markov | semi-Markov | semi-Markov | controlled partially observed renewal/semi-Markov |
| Main abstraction target | function remains semi-Markov, independent of initial distribution | necessary/sufficient lumpability conditions via first-exit/first-passage transforms | state aggregation preserving passage-time distributions between selected states | aggregation for efficient passage-time computation | policy-relevant return/cycle law |
| First-passage information | not established from abstract | central | central | central | central |
| Control/policy quantification | not established | not established | not established from author abstract | no control formulation used in inspected manuscript | explicit goal |
| Partial observation/history | not established | not established | not established | not established | explicit goal |
| Recursive learned representation | not established | not established | state aggregation algorithm, not learned history encoder | aggregation algorithm | desired, not achieved |
| Approximate tail metric | not established | transform-domain exact characterization | exact passage-time preservation in abstract | exact/approximate computational aggregation context | \(D_{\rm rec}=W_1\) on proper finite-mean laws |
| Finite-sample certification | not established | not established | not established | not established | open |
| Strict compression | aggregation is the subject | aggregation is the subject | explicitly constructs smaller SMPs | aggregation is explicit | only a weak property-specific witness obtained |

### Prior-work interpretation

The 1988 abstract states that a necessary and sufficient semi-Markov
lumpability condition is reinterpreted through first-exit times and that a new
necessary and sufficient condition is obtained through first-passage times in
the Laplace-transform domain.  This is close enough that no first-passage-based
exact lumpability theorem can be claimed novel without full theorem-by-theorem
comparison.

More damagingly, Bradley (2002) explicitly proposes a state-based equivalence
that preserves passage-time distributions between selected states, and the
later aggregation literature describes construction of smaller semi-Markov
models preserving passage-time quantities.  Therefore, the mere existence of
a property-specific passage-time quotient strictly coarser than full model
equivalence is prior art, not a new representation principle.

The general semantic separation is also mature: trace/property equivalence can
be coarser than probabilistic bisimulation because it preserves selected
observable behaviours rather than the branching kernel.  The counterexample in
Section 3 instantiates that known distinction in a return-time setting.

## 7. Why the strict-compression witness is insufficient

The construction merges states with different primitive futures while
preserving the target return law.  It therefore proves logical strictness, but
not a publishable compression result.

1. The differing intermediate branch is irrelevant to return time and cycle
   yield; relative to the claimed property it is nuisance information.
2. The quotient is not an autonomous controlled process unless \(u\) and
   \(v\) are merged as well, or the hidden representative is retained.
3. Passage-time-preserving state aggregation already exists in the semi-Markov
   literature.

Thus the requested non-nuisance strict compression theorem has **not** been
obtained.

## 8. Finite-sample certification

No positive finite-sample result follows from this round.  Without an explicit
tail class—such as a known geometric envelope, a parametric survival family,
a verified drift/minorization condition, or bounded support—finite trajectories
cannot exclude an unseen late-return component.  Consequently they cannot
certify equality of infinite return laws or recurrence class uniformly over
policies.

A future learning theorem would need to fix the structural class first and
state exactly which policy coverage, censoring, and tail-envelope constants are
known.  That is a different research direction; it is not smuggled into the
current exact-abstraction claim.

## 9. Answers to the two kill-gate questions

### Q1. Is policy-uniform recursive return-tail preservation strictly weaker
than semi-Markov/controlled lumpability?

**Qualified yes.** It is strictly weaker if "recursive" means only pathwise
updateability and the preserved object is a selected return/cycle trace law.
Proposition 1 is an exact counterexample.

**No for the deployable Markov-state interpretation.** If the representation
must possess a representative-independent controlled semi-Markov kernel,
Proposition 2 shows that the requirement is controlled lumpability itself.

### Q2. Does the strictness yield nontrivial strict compression or finite-sample
certification?

**No in this round.** The obtained compression is property-specific/nuisance
compression and lies in established passage-time aggregation territory.  No
finite-sample certificate is possible without additional tail structure, and
no new certificate was derived.

## 10. Decision

\[
\boxed{
\begin{array}{l}
\text{Kill the current exact recurrence-tail abstraction theorem as the}\
\text{central Spotlight novelty claim.}
\end{array}}
\]

This decision does not kill the persistent-autonomy problem.  It kills the
specific hope that "policy-uniform recursive return-tail preservation" alone
both escapes lumpability and supplies a deployable lower-dimensional control
state.

The only plausible continuation is a separately justified pivot to one of:

1. finite-data certification under an explicit, scientifically defensible tail
   or drift model class; or
2. a broader property-targeted predictive-abstraction theory with a result not
   reducible to existing trace equivalence, passage-time aggregation, or
   lumpability.

Neither pivot is approved or developed here.

## 11. Evidence boundary and sources

- Sumita, U. and Rieders, M. (1988), *First passage times and lumpability of
  semi-Markov processes*, Journal of Applied Probability 25(4), 675--687,
  DOI: 10.2307/3214288.  Only the publisher abstract and metadata were
  available in this pass:
  <https://www.cambridge.org/core/journals/journal-of-applied-probability/article/abs/first-passage-times-and-lumpability-of-semimarkov-processes/F5601F9E5DF2EC1C3259DDE88907E665>.
- Serfozo, R. F. (1971), *Functions of Semi-Markov Processes*, SIAM Journal on
  Applied Mathematics 20(3), DOI: 10.1137/0120055:
  <https://epubs.siam.org/doi/10.1137/0120055>.
- Bradley, J. T. (2002), *A Passage-Time Preserving Equivalence for Semi-Markov
  Processes*, author publication record and abstract:
  <https://www.doc.ic.ac.uk/~jb/reports/abstract.html>.
- Guenther, M. C., Bradley, J. T., and Knottenbelt, W. J. (2011),
  *Passage-time Computation and Aggregation Strategies for Large Semi-Markov
  Processes*, accessible author manuscript:
  <https://eprints.maths.manchester.ac.uk/1550/3/smp-passage-time-aggregation.pdf>.
- Bian, G. and Abate, A. (2017), *On the Relationship between Bisimulation and
  Trace Equivalence in an Approximate Probabilistic Context*:
  <https://arxiv.org/abs/1701.04547>.

No claim about the exact hypotheses or quantifier structure of the inaccessible
1988 full theorems is made beyond its publisher abstract.  The present verdict
does not depend on pretending otherwise: Bradley's explicit passage-time-
preserving equivalence plus Proposition 2's kernel-sufficiency collapse already
eliminate the proposed exact abstraction as a defensible central novelty.
