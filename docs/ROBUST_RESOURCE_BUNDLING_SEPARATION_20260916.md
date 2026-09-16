# Derivation Package: Robust finite-budget resource-bundling separation

## Target

Answer whether the existing strict finite-budget certificate survives on an
open parameter region, without new horizons, sampled runs, or learner changes.
Preserve equality of execution comparators and of the maximal instance-dependent
asymptotic coefficient ACROSS capacities, at each parameter point.

## Status

**COHERENT AFTER REFRAMING / EXTRA ASSUMPTION.** A handwritten robustness theorem
and explicit four-dimensional open box are established below, conditional on
the existing exact anchor certificate. This is not an independent proof audit.
The box is open in a specified structural family, NOT in the unrestricted space
of all independent hypothesis reward means. Continuity is standard mathematics;
robustness removes numerical fine-tuning, not the structured-experiment novelty risk.

## Invariant Object

Fix the physical graph, debit-before-arrival reload semantics, path durations,
feedback protocol, and initial charger. Fix the three hypothesis LABELS. Only
the reward parameters below vary for this theoretical generalization. For each
fixed parameter point, capacity alone changes the known safe experiment set.
This does not amend the sealed empirical fixture or authorize new experiments.

\[
z=(r,u,v,w),\qquad z_0=(1/20,1/20,3/10,19/20),
\]
\[
\Theta_z=\{\theta_q=(u,r),\ \theta_A=(v,r),\ \theta_B=(v,w)\}.
\]

The common dock reward is Bernoulli(r). A/B inspections have the corresponding
hypothesis means. Rewards are independent conditional on the chosen primitive
row, with at most one Bernoulli reward per primitive step. All other rewards
are zero. Inspection observations are also their realized rewards.

\[
\mathfrak R_{B,z}(T)=\inf_a\max_{i\in\{q,A,B\}}
\left[T\rho_i(z)-E_{i,z}^a\sum_{t=0}^{T-1}Y_t\right].
\]

The policy class includes all safe primitive-feedback adaptive policies, with
arbitrary legal terminal states, exactly as in the anchor lower certificate.
For an upper bound we can restrict a witness to charger-terminal committed
paths. Infima are taken separately at each horizon. Parameter z is public;
the true hypothesis label is unknown. Each z has its own fixed hypothesis class,
which is identical at capacities 3 and 4.

## Assumptions

1. Path durations: dock 1, empty 2, A/B 3, AB/BA 4.
2. Capacity 3 forbids a joint measurement trip; capacity 4/on permits it.
   Turning bundling off leaves exactly the old paths and physical feedback.
3. The B inspection mean under q and A is structurally equal to the common dock
   mean r. This sharing identity is essential for the exact coefficient equality.
4. Strict execution inequalities:
   \(0<r,u,v,w<1,\ u<3r,\ v>3r,\ w>3v\).
5. The anchor exact certificates in
   `artifacts/bundling_finite_budget_theory_20260916/strict_benefit_certificate.json`
   are valid. They were checked by separate primitive and route Bellman code,
   but have not undergone an external mathematical audit.

## Notation

kl denotes natural-log Bernoulli KL. A policy is a complete mapping from legal
physical histories and internal randomization to actions; freezing a mapping
does not require that it be Bayes optimal at a perturbed z. Let
\(L_T=T(T+3)/2\). Let static-terminal risk mean the minimax risk of policies whose
inspection feedback does not affect actions and whose trajectories end at q.
This restricted static comparison is not an arbitrary-terminal comparison.

## Derivation Strategy and Map

Fixed safe graph -> uniform coupling bound for every policy -> Lipschitz
minimax risk -> strict anchor margin survives. Separately, derive execution
ordering and coefficient identities from structural sharing, and verify their
strict inequalities throughout the box. Do NOT infer equality of coefficients
from continuity: an exact equality needs a structural reason.

## Main Derivation

### Proposition 1: finite-horizon certificate stability

Consider two parameter points z,z' with the same legal policy spaces and
\(\|z-z'\|_\infty\le\epsilon\). Couple every policy's internal randomness.
Until the first reward/observation mismatch, histories and actions agree.
Using a common uniform draw, the conditional mismatch probability at any
primitive step is at most epsilon. Hence at step t the expected absolute
reward difference is at most \((t+1)\epsilon\), including both a current
mismatch and divergence caused by previous mismatches. Therefore, uniformly
over every history-dependent randomized policy and hypothesis label,

\[
\left|E_{i,z}^a\sum_{t<T}Y_t-E_{i,z'}^a\sum_{t<T}Y_t\right|
\le {T(T+1)\over2}\epsilon.
\]

Below the gains are r,v/3,w/3, so their change is at most epsilon. Thus

\[
|R_{i,z}^a(T)-R_{i,z'}^a(T)|\le L_T\epsilon.
\]

Taking max over labels and inf over the SAME space of policy mappings gives

\[
|\mathfrak R_{B,z}(T)-\mathfrak R_{B,z'}(T)|\le L_T\epsilon. \tag{1}
\]

Knowledge of different public parameter values does not invalidate this step:
at each fixed z the learner induces one mapping in that common policy space.
The same bound holds for the restricted static-terminal policy class. No
optimal-policy uniqueness, continuity of policy selection, or posterior
continuity is required.

More generally, if a low-capacity lower certificate is l and a fixed safe
high-capacity witness has worst risk at most U at an anchor, then

\[
\mathfrak R_{3,z}(T)-\mathfrak R_{4,z}(T)
\ge l-U-2L_T\epsilon. \tag{2}
\]

This is a sufficient condition based on a positive certificate margin; it is
not a necessary-and-sufficient condition for beneficial experiment geometry.

### Proposition 2: exact execution equality and asymptotic identities

Under the strict inequalities, the unique optimal committed path TYPES at
both capacities are dock, A, B respectively, with

\[
\rho_q=r,\qquad\rho_A=v/3,\qquad\rho_B=w/3.
\]

For q, joint-path reward rate is (u+r)/4<r; for A it is (v+r)/4<v/3;
for B it is (v+w)/4<w/3. The solo and empty/dock comparisons also follow
from the stated inequalities. At capacity 4 the second measurement is the
only possible continuation besides return after the first inspection; its
reward law is independent of that first observation. When the model is known,
feedback-dependent path mixing cannot improve expected reward per completed
cycle beyond the maximum path rate. At T=12 or 24, repetition of each optimal
path fits the horizon exactly, and every charger-terminal policy has expected
reward at most T times its gain. Thus the known-model charger-terminal optimum
is exactly \(T\rho_i(z)\) at BOTH capacities and ALL labels.

The confusing-alternative allocation coefficients are

\[
C_{q,3}(z)=C_{q,4}(z)={3r-u\over\operatorname{kl}(u,v)}, \tag{3}
\]
\[
C_{A,3}(z)={v-r\over\operatorname{kl}(r,w)},\qquad
C_{A,4}(z)={v/3-r\over\operatorname{kl}(r,w)},\qquad C_{B,3}=C_{B,4}=0. \tag{4}
\]

Reason for (3): distinguishing q from A requires A observations; the cost of
one such observation is 3r-u for BOTH A-only and AB. This constraint also
suffices for the q-versus-B alternative, which has the same A mean v and
additional B KL. Extra B observations cannot relax q-versus-A discrimination.
For A, its optimal path supplies A observations free of leading regret; only
B distinguishes the confusing B alternative. One B observation costs v-r on
B-only, versus v/3-r on AB. At B, optimal B feedback distinguishes both other
labels for free. These are the cost-weighted classical allocation coefficients,
not a new allocation theorem.

If \(C_q>C_{A,3}\), (3)-(4) yield the EXACT cross-capacity equality

\[
\max_i C_{i,3}(z)=\max_i C_{i,4}(z)=C_q(z). \tag{5}
\]

This is the maximum of fixed-instance asymptotic coefficients. It is NOT an
exchange-of-limit-and-supremum theorem about finite-time minimax asymptotics.
The common value may vary with z; equality is across capacities at each z.

### Theorem: explicit open-region strict separation

Let

\[
\mathcal U=\{z:\|z-z_0\|_\infty<10^{-4}\}.
\]

This is a nonempty open box in four independent structural coordinates,
with positive four-dimensional Lebesgue measure. All execution inequalities
hold throughout it: their anchor margins .10,.15,.05 lose at most .0004.
All reward means remain strictly interior.

The standard bounds \(\operatorname{kl}(a,b)\le(a-b)^2/[b(1-b)]\)
(KL <= chi-square) and \(\operatorname{kl}(a,b)\ge2(a-b)^2\) (Pinsker)
give throughout the box

\[
C_q\ge{(.1-.0004)(.2999)(.7001)\over(.25+.0002)^2}>.33,
\]
\[
C_{A,3}\le{.25+.0002\over2(.90-.0002)^2}<.16.
\]

Thus (5) holds everywhere in U, without numerical KL evaluation.
At the anchor T=12, the exact lower/upper certificates give

\[
l={2173\over8000},\quad
U={8604621\over40000000},\quad l-U=.056509475.
\]

Since \(2L_{12}=180\), (2) proves for EVERY z in U

\[
\boxed{\mathfrak R_{3,z}(12)-\mathfrak R_{4,z}(12)>.038509475.} \tag{6}
\]

The anchor high policy mixture can be frozen without reoptimizing a single
posterior, allocation, or prior at the new z. It remains physically safe and
charger-terminal and supplies the upper side of (6).

Bundling off leaves identical experiments, feedback and comparator at both
capacities, so for every z in U and EVERY finite T,

\[
\mathfrak R_{4,\mathrm{off},z}(T)
=\mathfrak R_{3,\mathrm{off},z}(T)
=\mathfrak R_{3,\mathrm{on},z}(T). \tag{7}
\]

Finally, the anchor static-terminal lower value 32/115 and the same adaptive
witness U imply throughout U

\[
\mathfrak R^{\mathrm{static,terminal}}_{4,z}(12)
-\max_i R_{i,z}^{\mathrm{witness}}(12)
>{32\over115}-.215115525-.018>.045145344.
\tag{8}
\]

Thus a feedback-free charger-terminal bundled schedule does not explain the
witness benefit, even after perturbation. This does not assert that within-trip
adaptation is necessary: the witness adapts BETWEEN completed trips.

Corollary using ONLY previously certified horizons: on the smaller open box
\(\|z-z_0\|_\infty<10^{-5}\), (2) and the anchor gaps >.05 imply
\(\mathfrak R_{3,z}(T)-\mathfrak R_{4,z}(T)>.04352\) for every integer
T=12,...,24, since \(2L_T\le648\). No additional horizon was enumerated.

## Remarks and Interpretation

The sufficient condition is a strict certificate margin exceeding a uniform
feedback-law perturbation penalty, together with structural comparator and
coefficient identities. It produces an open family, not thirteen isolated
numeric points. Its conservative radius is a proved radius, not a tuned or
maximal robustness radius.

The result establishes robust resource-enabled FINITE-budget separation despite
equal maximal INSTANCE-dependent logarithmic coefficients. It does not establish
that resources change a universal learning principle outside structured experiments.

## Boundaries and Nonclaims

- Sharing B-baseline with dock is a genuine restriction. If these means vary
  independently, q's AB cost minus its A-only cost is r minus the B-baseline.
  Exact equality (3) is then generally lost. Do not call U open in that larger
  unrestricted model space. Strict risk separation itself survives small changes,
  but the full package of exact equalities does not follow by continuity alone.
- No assertion that the canonical bias span remains exactly equal across
  capacities throughout U; that additional anchor property was not generalized.
- No theorem at T=4096, no positive empirical validation, and no repair or rerun
  of the frozen learner. Its negative calibration remains part of the record.
- No efficient general minimax solver or necessity characterization.
- Conditional handwritten proof plus rational arithmetic verification, NOT Lean
  verification or independent researcher review.

## Open Risks

The openness objection is resolved within the explicitly defined structural
family. Generality across different route graphs, operational budgets, and
novelty relative to cost-weighted structured experiments remain separate questions.
See `RESOURCE_BUNDLING_THEOREM_PRIOR_ART_AUDIT_20260916.md` for the bounded
source-grounded novelty audit; it does not certify absence of prior art.

## Verification record

- Existing receipt recomputation: all 13 rational anchor certificates passed;
  no floating optimization in that verification command.
- `scripts/verify_bundling_robust_separation.py`: exact rational box margins,
  coefficient inequality bounds and risk-gap constants passed. This checks
  algebraic constants, not the analytic coupling proof itself.
- `pytest tests/test_bundling_robust_separation.py
  tests/test_bundling_finite_budget_theory.py -q`: 9 passed.
- Frozen calibration core SHA256 remains
  `6590733c3e1df21f97f653fa5c0d17eeba90342a063db63287f57dde55646d2d`;
  runner remains
  `974aafc8c3aeec9f611b2cc7b2ba7a07e197f732fb89f3151fbf0ef24dfdce03`.
- Mathematical outcome enumeration only; zero sampled trajectories.
