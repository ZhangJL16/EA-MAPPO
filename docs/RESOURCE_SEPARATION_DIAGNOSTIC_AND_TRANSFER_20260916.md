# A reusable diagnostic for asymptotically neutral finite-budget resource gains

## Draft: diagnostic proposition (manuscript insert)

Consider two nested finite controlled-experiment systems with the same finite hypothesis class, initial condition, hypothesis-wise execution comparator, and exactly embedded old actions, durations, and feedback laws. Randomization and perfect recall are allowed. Fix an integer budget \(T\). Write \(r_j(a)\in\mathbb R^{|\Theta|}\) for a safe policy's expected regret vector and

\[
v_j=\min_a\max_\theta r_j(a)_\theta,\qquad
b_j(\pi)=\min_a\pi\cdot r_j(a),\qquad
\mathcal L_1=\arg\max_{\pi\in\Delta(\Theta)}b_1(\pi).
\]

For the asymptotic statement, assume the confusing-alternative allocation programs have unchanged alternative sets, nonnegative experiment costs, and finite attained primal and dual optima. Their interpretation as attainable logarithmic regret coefficients requires the existing attainability assumptions; it is not automatic for every controlled experiment.

**Proposition (two-certificate diagnostic).** The enlargement preserves \(C_1^{\max}\) and strictly lowers \(v_1\) if and only if the following two certificates exist:

1. **Surviving asymptotic price.** For some old maximizing hypothesis \(\theta_*\), an optimal old allocation dual \(y_*\ge0\) satisfies every new experiment constraint

   \[
   \sum_{\theta'\in\operatorname{Alt}_{\theta_*}}
   y_{*,\theta'}\operatorname{KL}(Q_{\theta_*e}\Vert Q_{\theta'e})
   \le c_{\theta_*e}.
   \]

2. **Finite decision coverage.** There is a finite library \(\mathcal W\) of safe new-system policies such that

   \[
   \forall\pi\in\mathcal L_1,\quad
   \min_{a\in\mathcal W}\pi\cdot r_2(a)<v_1.
   \]

The library may include old policies. Its policies need not be selected independently for each true hypothesis: a policy is a complete unknown-label feedback tree, and the condition only uses prior-weighted values of such implementable trees.

**Proof.** Nesting makes each instance allocation value nonincreasing. A surviving optimal dual gives equality at an old maximizing hypothesis, hence equality of the maximum. Conversely, equality of the maxima forces some new maximizing hypothesis to have the old maximal value; a new optimal dual is then also old-optimal and supplies the required certificate.

Finite minimax duality gives \(v_j=\max_\pi b_j(\pi)\), and nesting gives \(b_2\le b_1\). Let \(g=b_1-b_2\ge0\). Then

\[
v_1-v_2=\min_\pi\{v_1-b_1(\pi)+g(\pi)\}.
\]

Both summands are continuous and nonnegative on the compact prior simplex. Their sum has a positive minimum exactly when \(g>0\) on every old least-favorable prior. Certificate 2 ensures this condition because \(b_2\le\min_{a\in\mathcal W}\pi\cdot r_2(a)<v_1=b_1(\pi)\) there. Necessity follows by taking the finite set of new deterministic policy trees as the library. Finiteness follows from finite histories and feedback alphabets; randomized trees convexify their risk vectors. This proves both directions. \(\square\)

**A quantitative certificate that does not require identifying least-favorable priors.** Choose any public old prior \(\pi_0\), certify its old Bayes regret \(l=b_1(\pi_0)\), and evaluate one implementable new randomized policy \(a_+\). If

\[
U=\max_\theta r_2(a_+)_\theta<l,
\]

then \(v_1-v_2\ge l-U>0\). The same policy covers every old least-favorable prior because \(\pi\cdot r_2(a_+)\le U<l\le v_1\). A lower-certificate prior need not itself be least-favorable. This is the computable sufficient form used by the resource theorem.

### Scope of the diagnostic

The proposition repackages established finite minimax and allocation duality into two separate comparison checks. It is not a new foundational theorem, a graph-local condition, or an efficient general solver. Positive continuation value at one prior is insufficient. Cheaper acquisition and additive KL are also insufficient without finite decision coverage. A new allocation row may be tight: neutrality does not require inactivity.

The current example supplies the two certificates independently. The \(q\)-versus-\(A\) price survives the bundled route because its centered cost is \(3r-u\), the same as the single \(A\) route, and its \(B\) observation contributes zero KL against that alternative. At the anchor, the public old prior gives \(l=2173/8000\), whereas the frozen three-tree randomized new policy gives \(U=8604621/40000000\). Thus \(l-U=2260379/40000000\). The existing perturbation bound gives the open-region margin \(>0.038509475\). The predicate is useful because it separates two witnesses that need not concern the same prior or hypothesis, not because duality itself is new.

Sources: [existing exact bottleneck derivation](RESOURCE_ENABLED_FINITE_BUDGET_PRINCIPLE_20260916.md), [strict anchor](STRICT_FINITE_BUDGET_CAPACITY_BENEFIT_20260916.md), [open region](ROBUST_RESOURCE_BUNDLING_SEPARATION_20260916.md). Analytical intellectual debt includes [controlled sensing](https://arxiv.org/abs/1205.0858) and [structured allocation / OSSB](https://arxiv.org/abs/1711.00400).

## Draft: secondary realization outside navigation

### Automated assay batches with a replaceable consumable cartridge

Consider an idealized automated screening bench. The learner chooses which assay types to run while the material regime is unknown. Each assay produces an independent Bernoulli yield indicator, observed as its reward. The public hypotheses are the same \(\Theta_z=\{(u,r),(v,r),(v,w)\}\). This is a mathematical model of experimental operation, not a claim about a measured laboratory or a clinical diagnosis system.

A known cartridge capacity limits safe batch length. Preparing a batch consumes one time/resource unit, each assay consumes one, and a mandatory cleanup/reset consumes one. Cleanup is debited before a fresh cartridge is installed. A prepared batch can run each assay type at most once, and must retain enough capacity for cleanup. At the reset station, a standalone reference operation consumes one unit, produces a Bernoulli reward of mean \(r\), and restores the cartridge after that unit is debited. Empty preparation-and-cleanup is allowed.

At capacity three, single-assay batches fit but a two-assay batch does not. At capacity four, both assay types can share preparation and cleanup. With the batch-sharing permission off, cleanup is mandatory immediately after the first assay regardless of capacity. The reward laws and all old operations remain unchanged.

| Inspection realization | Screening-bench realization | Preserved object |
| --- | --- | --- |
| Charger / completed return | Reset station / completed cleanup | Initial and regenerative state |
| Docking operation | Standalone reference operation | One time/resource unit; Bernoulli(r) reward |
| Preparation/undocking | Batch preparation | One zero-reward time/resource unit; no premature reload |
| A or B inspection | Assay A or B | Corresponding public Bernoulli law and feedback |
| Return | Cleanup/reset | One zero-reward debit, then full replenishment |
| Battery capacity | Replaceable cartridge capacity | Known safe-feasibility counter |
| Joint safe sortie | Shared-preparation assay batch | Duration four, two observations |

**Transfer corollary.** For every \(z\) in the existing open structural neighborhood, both realizations have identical hypothesis-wise risk vectors under corresponding policies, at every finite budget and each capacity/permission condition. Consequently the known-model gains, the qualified charger/reset-terminal execution equalities, the maximal fixed-instance coefficients, and the certified finite-budget risk separation transfer exactly.

**Proof.** Map a primitive history by replacing each action and physical state with its corresponding operation and apparatus state in the table, while retaining the counter, reward observations, time index, and randomization. Preparation, assay, and cleanup have the same legal transitions, time/resource debits, and reload order. Therefore the map is a bijection on supported safe histories, including partially completed terminal batches. Copy a policy's conditional action probabilities through this bijection. Induction on primitive steps gives the same history probabilities under each hypothesis; the reverse map gives the converse policy correspondence. Rewards and comparators agree, so corresponding expected regret vectors agree. Taking the same minimax optimization gives equal risks. Complete regenerative experiment durations and feedback laws also agree, giving identical allocation coefficients. \(\square\)

This answers the navigation-specific objection: the physical implementation can be batch preparation/cleanup rather than travel/recharge. It does **not** add an independent statistical problem class, experimental dataset, or new proof of generality. Calling this a second empirical validation would be incorrect. Its role is exact non-navigation realizability; the open-region theorem supplies robustness, and the diagnostic supplies the reusable comparison test.

## Scientific interpretation for the paper

What changes is the set of feasible joint-feedback and continuation protocols. Two individually queryable parameters can remain individually queryable at both resource levels, while shared acquisition changes which feedback combinations can inform the next allocation of calendar budget. This is a missing comparison axis, not a claim that feedback geometry alone determines risk. Reward laws, comparators, durations, and hypothesis class remain necessary inputs.

The claim that survives is therefore: **under resource-realizable experiment nesting, two specific execution and asymptotic summaries can fail to rank optimal finite-budget learning risk; the failure is certified by a surviving asymptotic price together with finite decision coverage.** The resource result supplies an open family realizing both certificates. It remains within controlled experiments and does not assert first discovery of finite/asymptotic discrepancies or adaptive sensing.

## Author notes

Writing axes: manuscript / research / theorem-method insert + title / Chinese-to-English / generic ICML. User-supplied central claim and boundaries were retained; the sections were not rebuilt around a new theory claim.

Paragraph jobs: diagnostic assumptions and definition; price certificate; finite coverage certificate; proof; practical lower/upper test; boundaries; secondary operational model; history map; transfer proof; interpretation.

Terminology follows the main paper: maximal fixed-instance coefficient (not asymptotic minimax constant); finite-budget minimax risk; safe primitive policy; charger/reset-terminal execution value; between-excursion adaptive continuation.

Claim–evidence map: diagnostic supported by the proof and existing duality derivation; quantitative separation supported by existing certificates and the new implementation-diverse receipt; cross-domain transfer supported by the history bijection, not experimental data. Independent external proof review and global priority remain unclaimed.

Structure choice: comparison criterion is explanatory, not a new headline theorem; non-navigation realization is a transfer appendix, not a second empirical domain; the original separation stays Theorem 1. Targeted corrections should name a predicate or mapping rather than regenerate the whole paper.
