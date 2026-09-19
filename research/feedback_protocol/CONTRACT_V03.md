# FPL v0.3 — scientific DEV comparison

This extends v0.2, not the frozen original resource-separation experiment.
Finite known hypotheses and independent Bernoulli channels remain the scope.
Protocols are committed before departure, end at reset, and release feedback
together. Evaluator-only utility does not become a policy observation. Regret
uses matching reset-terminal known-model finite-budget utility, not `T * rho`.
No neural learner or generic likelihood adapter is introduced.

## Scientific identity

`RegistryV2` records distribution/template IDs, conservative name-invariant clone
hash, parameter/structure seeds, split, OOD axis and root group. Distribution
and template may cross train/dev/test; structural clones may not. Capacity,
horizon and reward reskins stay grouped. The legacy V1 registry/config remains
a historical engineering fixture.

The scientific plan freezes 44 public configurations. IID train/dev/test share
the same directed random-graph distribution. Structural exclusion conditions
sampling: **same-generator, clone-disjoint IID**, not unconditional independent
graph draws. The WL hash may conservatively merge nonisomorphic structures.
Generation consults no oracle, policy, trajectory or risk. Rejected structure
seeds are recorded. OOD axes explicitly identify size, topology, duration/energy
ratio and hypothesis count. OOD-cost also uses independent structures, so it
is not a matched causal intervention on cost alone.

Only 12 DEV configurations from four root groups are evaluated: two IID roots,
one dense-scale root and one sparse-scale root. Scale diagnostics are labeled
separately, not described as IID test. Public test/OOD manifest generation is
not final-test evaluation; the evaluator never loads their problem files.

## Certified tiny minimax: completeness and limits

Finite deterministic contingent policy trees are enumerated up to explicit caps.
Trees with identical hypothesis-wise expected utility vectors can be merged
because each subtree begins at reset. Mixtures represent randomized policies;
only released public feedback branches the tree.

Primal and dual rational LP bases are searched **independently**. On a primal
support of size `s`, normalization and `s` independent active hypothesis risk
inequalities determine a vertex in `s` weights plus value. A basic solution can
have support at most the number of hypotheses. Enumerating supports and active
sets includes degenerate bases. The dual argument is symmetric. The best
feasible primal/dual witnesses are retained separately; they may have unequal
supports and be discovered in different iterations. Exact feasibility and
equal rational objectives alone authorize the returned certificate. Sixty
seeded random matrices are cross-checked against SciPy LP in tests; runtime has
no SciPy dependency.

This gives uncapped finite-matrix basis coverage, not polynomial complexity or
guaranteed completion under caps. Policy products, unique vectors, LP candidates,
model calls and time can hit limits. `PlanningLimit` / `UnresolvedCertificate`
means unresolved, never nonexistence, zero gap or an approximate optimum.
`CertifiedMinimaxSmall` is a descriptive alias; `ExactMinimax` remains compatible.

## Shared computation budget

One `SearchBudget` covers an entire online `select`, including every recursive
candidate, outcome and continuation call. It does not reset inside recursion.

- Expansions count eligible-source edge attempts and explicit open-loop DP
  transitions. Tree combinations and LP candidates are additionally reported.
- Model charges count hypothesis-operation utility evaluations, likelihood
  factors and explicit mean/variance row evaluations.
- Wall-time checks are cooperative, not OS-preemptive; one Python operation may
  overshoot. Work counters are implementation units, not FLOPs.

Exact references have separate offline budgets and saved construction costs.
Their online table/tree lookup latency is NOT end-to-end planning cost and is
excluded from the online Pareto set. Common environment simulation/posterior
update time is outside search timing. No machine-independent efficiency claim
follows from one timing run.

On exhaustion, approximate methods return a completed legal incumbent, a
prechecked reset self-loop or stop. Pruning and budget-limit reasons are logged.
Candidate heuristics have no approximation guarantee. ChannelCover retains
bounded exhaustive enumeration; Beam, posterior sampling and OneStepVOI do not.
OneStepVOI uses one posterior update followed by a nonadaptive knapsack tail
over bounded candidates; it is not globally exact VOI or multi-step adaptive
planning. Beam can lose useful candidates during heuristic pruning.

## CRN and estimation

Evaluator-only hash-separated pseudorandom streams are keyed by `(root_group,
theta, noise_seed, channel, query_index)`. Paired methods/capacities/budgets share
the same draw for each channel's kth query, independent of query order. Marginal
Bernoulli sampling has ordinary finite-precision PRNG accuracy. Policies receive
neither truth nor future stream keys.

The sampling unit is the root graph. Capacity/horizon conditions are repeated
comparisons; policy/noise repeats are nested Monte Carlo samples, not extra
graphs. This pilot uses two fixed policy seeds and four noise seeds, no trained
model seeds. Per-hypothesis risk subtracts mean utility from the known-model
oracle. Bayes risk weights these means by the public prior. Worst-hypothesis
risk is the **maximum of estimated means**, never the maximum sample shortfall.
The maximum retains finite-replicate selection bias. Two policy seeds can
poorly represent a minimax mixture: exact certified risks and its Monte Carlo
execution estimates must be reported separately.

Noise-block MC standard errors average policy seeds first and are conditional
on that fixed seed set and instance. They are not generalization intervals.
Root summaries average repeated conditions before cross-root interpretation.
Reference means may cover fewer solved conditions: compare only explicit matched
subsets, not pooled reference averages against all conditions. This pilot has
no significance tests, power claims, population superiority or automatic B2
promotion. Empirical Pareto status is noisy and timing dependent.

## Recovery and integrity

Sources/configuration/dataset identity are sealed before episodes. Public
reference checkpoints include content hashes and offline costs, even when
unresolved. Each complete episode is flushed/fsynced. An interrupted episode is
replayed from its seeds; it is not counted as complete. Resume refuses changed
seals, duplicates, inconsistent episode IDs and incomplete JSONL tails. One
writer holds a lock. Analysis requires the full scheduled axis product and
retains unresolved cells. Old CONFIRM and final-test evaluation are not options.
