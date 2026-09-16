# Positioning within controlled sensing: asymptotic-neutral finite-budget separation

Two-page-equivalent positioning draft. Scope: the existing theorems only; no new
experiment, solver, parameter family, or priority claim. The accompanying
[evidence audit](CONTROLLED_SENSING_POSITIONING_EVIDENCE_20260916.md) records the
comparison boundaries.

## Related-work positioning

**Controlled experiments are the parent formulation, not an excluded competitor.**
Our finite-hypothesis resource system is representable as a controlled experiment:
the physical state and known counter determine legal actions, feedback updates
the posterior, and calendar time determines the remaining decision budget.
Committed safe excursions additionally form structured experiments with joint
feedback. This representation preserves the resource coupling when durations and
feasibility are retained. We therefore do not claim an irreducible formulation,
new minimax duality, or that physical trajectories place the problem outside
controlled sensing. The contribution is a constrained separation result within
this established framework.

**Adaptivity and costly sensing already have substantial theory.** Nitinawarat,
Atia, and Veeravalli characterize fixed-sample controlled-testing error exponents
and exhibit a multihypothesis advantage of causal over open-loop sensing.
Naghshvar and Javidi analyze sequentiality and adaptivity through expected sample
cost plus a terminal error penalty. Nitinawarat and Veeravalli further treat
controlled Markov observations and nonuniform sensing costs. These works rule out
claims that adaptive information use or control-dependent acquisition cost is
new here. Their named results concern testing performance, rather than our joint
comparison of nested resource-realizable systems with preserved execution value,
preserved maximal fixed-instance regret coefficient, and strictly different
finite-budget minimax regret. [Controlled sensing](https://arxiv.org/pdf/1205.0858),
[adaptivity gains](https://arxiv.org/pdf/1211.2291),
[Markov sensing with costs](https://arxiv.org/pdf/1310.1844).

**Finite sensing/exploitation value is also established.** Ferrari, Zhao, and
Scaglione maximize finite-horizon utility while choosing when and where to sense;
their stopping thresholds depend on remaining time and unresolved resources.
Thus posterior-dependent continuation value is not a new learning principle,
and it would be incorrect to characterize controlled sensing as inference-only.
Our continuation-value criterion is an explanatory certificate built from
standard Bellman/minimax tools. Its role is to identify which finite-budget
least-favorable priors must benefit, independently of which asymptotic allocation
dual remains feasible. [Finite-horizon utility sensing](https://arxiv.org/pdf/1705.05960).

**Structured bandits already provide the allocation machinery.** Combes,
Magureanu, and Proutiere derive confusing-alternative allocation lower bounds and
give OSSB attainability guarantees under explicit regularity assumptions.
Joint path feedback and shared measurements are compatible with structured
experiments; resource-induced acquisition geometry is not, by itself, evidence
of a new allocation theory. Our path cost
\(c_{\theta p}=\rho_\theta\ell_p-E_\theta Y_p\) accounts for calendar-time
opportunity cost. Unequal durations require faithful accounting, but this does
not defeat the structured-experiment reduction or create novelty merely by
reweighting its LP. [OSSB](https://proceedings.neurips.cc/paper/2017/file/e19347e1c3ca0c0b97de5fb3b690855a-Paper.pdf).

**Asymptotic summaries can already miss finite-budget difficulty.** Wu, György,
and Szepesvári explicitly distinguish finite-time side-observation difficulty from
its asymptotic logarithmic bound, including generalized full-information cases
with zero logarithmic coefficient but potentially large transient regret. Their
Theorem 1 also converts a worst-case performance envelope into an
instance-specific finite-time information lower bound. Adusumilli characterizes
Bayes risk and minimax policies under local diffusion asymptotics, emphasizing
that rate-optimality need not rank policies. Consequently, neither the broad
finite-versus-asymptotic distinction nor least-favorable-prior analysis should be
claimed as first discovered here. [Side observations](https://proceedings.neurips.cc/paper/2015/file/8e82ab7243b7c66d768f1b8ce1c967eb-Paper.pdf),
[bandit decision risk](https://doi.org/10.3982/ECTA21075).

**The narrower separation is the manuscript's defensible center.** For a fixed
public hypothesis class, resource augmentation embeds all old experiments
without changing their feedback. Our existing construction simultaneously
preserves execution gains, preserves charger-terminal known-model execution
value at the certified endpoint budgets, and preserves the maximum of attainable
fixed-instance logarithmic regret coefficients. Nevertheless, it strictly lowers
finite-budget minimax regret. The result survives an open neighborhood inside
the stated shared-reward parameter family. A capacity-only, bundling-disabled
control has identical risk. Standard tools can express and certify this result;
the named neighboring theorems do not supply this conjunction simply by invoking
their conclusions. This is a concrete robust separation, not a universal claim
that adding informative actions is beneficial or that resource variables are
statistically costly.

Two qualifications prevent overstatement. Equality concerns
\(\max_\theta C_{B,\theta}\), not every hypothesis's coefficient and not an
unproved interchange of minimax and asymptotic limits. The improved hypothesis
need not be the one setting that maximum. Moreover, the T=4096 frozen
horizon-free learner did not demonstrate the predicted learning advantage;
the finite-budget certificate concerns an optimal risk and a different,
budget-specific adaptive witness. The ML implication is therefore evaluative:
an unchanged worst fixed-instance information price is insufficient to establish
finite-budget equivalence of physically feasible learning protocols. Empirical
benefit for a deployed learner is a separate claim, not supplied by this theorem.

## Theorem comparison

The last column concerns the identified result, not an assertion that the whole
literature lacks a related example.

| Primary work / inspected result | Established object and conclusion | What does not follow by direct invocation |
| --- | --- | --- |
| Nitinawarat–Atia–Veeravalli; Theorems 1–2, Section V | Open-loop testing exponent; causal bounds and an adaptivity example | The execution-preserving resource separation in cumulative regret |
| Naghshvar–Javidi; Theorems 1–3 | Asymptotically tight costs for testing-policy classes | Equality of maximal fixed-instance regret prices under nested resource protocols |
| Nitinawarat–Veeravalli; Theorems 4.2, 5.1 | Testing risk constraints; asymptotically optimal nonuniform-cost Markov sensing | Our finite-calendar-budget minimax regret gap |
| Ferrari–Zhao–Scaglione; Lemma 1, Theorem 1 | Convex belief values and time-dependent stopping thresholds | The simultaneous neutral-allocation/strict-minimax comparison |
| Combes–Magureanu–Proutiere; Theorems 1–2 | Structured KL allocation and OSSB asymptotic attainability | Strict finite-budget risk separation from equality of the maximal coefficient |
| Wu–György–Szepesvári; Section 3, Theorem 1 | Finite-time envelope lower bound; distinction from logarithmic allocation | The resource realization and preserved execution constraints of our certificate |
| Adusumilli; Section 2.4, Theorem 2 (2025 preprint) | Least-favorable-prior analysis; discrete Bayes-risk convergence to a diffusion PDE | Our fixed-hypothesis, exact short-budget resource separation |

## Contribution sentence

We establish a robust, resource-realizable separation in which nested safe
experiment spaces have equal execution gains and equal maximal fixed-instance
asymptotic regret coefficients, yet strictly different finite-budget minimax
regret; standard allocation duality and Bayes continuation values explain the two
distinct bottlenecks.
