# FPL v0.2: finite Bernoulli task families and bounded planning

This version extends the first-batch contract. The public family is still finite
and its feedback laws are independent Bernoulli draws conditional on the label.
No continuous or generic discrete likelihood adapter is claimed.

## Observation and task utility

`Operation.channels` defines the observation model through the public hypothesis
matrix. `Operation.utility` independently specifies a `UtilitySpec`:

- `None`: backward-compatible sum of observed bits.
- `observation_weights`: weighted task utility from those same draws; an empty
  tuple makes observation utility zero.
- `by_hypothesis`: deterministic task payoff for each public hypothesis.

Explicit utilities require `objective=cumulative_task_utility`. These task payoffs
are evaluator scores, unavailable as additional online feedback. Only declared
observation channels enter the posterior. If an application reveals a task payoff
that is informative about the hidden model, it needs an explicit likelihood
adapter; this implementation must not be used while ignoring that information.
Negative task utilities are allowed; stopping is always worth zero at reset.
Utility does not alter time, resource, return or measurement limits.

## Generation and split enforcement

`generator.generate` supports complete, directed-chain, star and directed-ring
measurement topologies, arbitrary positive channel counts, at least two hypotheses,
public random means, independent or paired shared parameters, separate time/energy,
capacity and deployment budget. Probes have zero immediate utility; task actions
at reset produce model-dependent task utility without observations. This is one
synthetic model family with four graph templates, not four application domains.

Generated variants of a topology template share a family identifier across sizes,
priors/reward reskins, capacities, horizons and sharing choices. `Registry` rejects
cross-split reuse of a family, and also rejects equal name-invariant incidence-graph
fingerprints under different families. Fingerprints ignore numeric model, cost,
capacity and budget fields. Color-refinement collisions can conservatively group
non-isomorphic graphs; this never licenses splitting an isomorphic clone.

All generated files must be registered before publication of a dataset. The family
plan produces 20 unfiltered engineering instances, not a frozen scientific split.
The model loader itself does not discover other dataset registries on disk; global
enforcement is with respect to the supplied common registry. External producers
must preserve template provenance. No mechanism-only oracle-equality rejection
sampler or formal benchmark distribution is implemented yet.

## Reference and baseline roles

ExactBayes and known-model finite-budget DP now optimize the declared utility.
`ExactMinimax.solve` enumerates deterministic contingent policy trees, retaining a
representative of each distinct hypothesis-wise expected-utility vector. It solves
the resulting regret game by rational vertex search and returns both a randomized
policy-tree mixture and a least-favorable prior. Acceptance requires exact primal
and dual inequalities against all enumerated vectors and equal objective values.
Every public hypothesis is considered, including zero-prior hypotheses. Explicit
tree-product, vector-count, LP-candidate and time limits abort with `PlanningLimit`.
This solver is for tiny games; it is not a scalable minimax algorithm, and it has
not re-certified the old T=12 separation theorem.

BeamBayes generates candidates by bounded prefix beam search without calling
`enumerate_protocols`. It uses public posterior, remaining budget, resource and
measurement feasibility; only reset-returned candidates are executable. Prefixes
are heuristically ranked by expected task utility plus model variance per unit
time. Finite-depth posterior lookahead ranks complete candidates; a feasible
open-loop repeated-route tail values the unused budget at the search leaf. This
tail is a heuristic baseline, not an error bound. Search pruning/exhaustion is
reported. Wall-time checks are cooperative, so a bounded inner search may overrun
the nominal deadline; no real-time deadline guarantee is made.

PosteriorSampling draws a hypothesis from public belief using its own policy RNG,
then chooses a high expected-utility/time candidate from bounded prefix generation.
It is independent of the old allocation wrapper and of Bayes recursion. On tasks
where exploitation provides no feedback it may under-explore; no consistency or
minimax guarantee is asserted. ChannelCover remains the transparent simple baseline.

## Profiling, output and resumability

`fpl.profile` measures exhaustive catalogue count/time/Python-allocation peak,
Bayes states/likelihood branches/time/peak, and beam search statistics. A capped
enumeration has an unresolved total count, never a reported exact partial count.
Peak memory is tracemalloc Python allocation, not process RSS or GPU memory.
Timing includes tracing overhead and is single-machine, single-run descriptive.

Each completed cell is flushed and fsynced. Explicit `--resume` validates the
config/source seal and skips completed cells; a partial JSON line fails visibly.
A cell interrupted before writing is recomputed only on explicit resume. This is
profiling resumability, not a resumable multi-instance scientific learning runner.
Offline training/teacher data, posterior inference and online planning budgets
are not conflated. No full neural training or external-task adapter is supplied.
