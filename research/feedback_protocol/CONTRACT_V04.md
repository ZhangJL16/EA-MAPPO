# FPL v0.4: fixed-policy expectations and the final pre-B2 study

Scope remains finite hypotheses, Bernoulli feedback, known deterministic
resource updates, committed reset-terminal batches and evaluator-only task
utility. This is not a generic POMDP solver or a learned FPL method. Prior v0.3
evidence remains immutable in its archive; its source snapshot is included there.

## Exact fixed-policy evaluation

`ExactPolicyEvaluator` integrates environmental feedback, not a maximum over
protocols. It obtains the selected protocol from a **complete copied policy
state**, adds its exact hypothesis-wise expected utility and recursively weights
continuations by each hypothesis's feedback likelihood.

There is deliberately no `(belief,H)`-only memoization: channel coverage depends
on history, seeded stochastic planners on their owned RNG, and episode-limited
planners on remaining work. Every branch copies the state *after* selection.
Cached Bayes/minimax executors retain their table/tree cursor. Initial priors
must have full support for the reported all-hypothesis vector.

`exact_conditional_policy` means an exact expectation over environment feedback
**conditional on the supplied initial policy state/seed**. It does not integrate
all algorithm random seeds. Future learned policies must be deterministic in
evaluation mode or own all copied randomness; global RNG, external services,
time-dependent choices and hidden mutable shared state violate this contract.

Node/feedback caps or watchdog expiry return unresolved with no partial value
labeled exact. Only then can the study use its separately frozen CRN MC fallback;
MC can never satisfy the exact-gap GO criterion. Fixed-policy evaluation can
also be exponentially costly; it is not universally easier than optimization.

## Deterministic work, episode limits and watchdogs

`PlanningWorkBudget` charges expansions and model calls. Its wall watchdog raises
`WatchdogExpired`, **not** `PlanningLimit`; planners cannot catch it as normal
budget exhaustion and silently choose a different action. A watchdog invalidates
the evaluation. Latency is recorded separately. This gives reproducible action
semantics conditional on code, work limits, seeds and ordinary numerical runtime;
not a claim of bit-identical floating arithmetic on arbitrary hardware.

Existing selectors accept an optional injected account. Without it their legacy
default behavior remains unchanged. WorkPolicy supplies a new selection account
capped by both the selection limits and, in episode mode, the remaining global
episode balance. All simulated recursion shares that account. Spending is
deducted from the episode balance before feedback branching; the balance is
copied independently for each realized continuation.

Episode ceilings are four times the corresponding selection allowance in this
study. A protocol that triggers more decisions does not obtain unlimited new
planning work. Medium-tier selection-only controls are explicitly separate from
episode-limited runs. Exact evaluation reports expected total work and largest
branch total, not only a root-selection counter.

After search exhaustion, a score-blind, prevalidated reset self-loop can serve
as a common fixed autopilot without fresh model queries. It can be suboptimal;
gaps characterize these implemented budgeted planners, not a lower bound on all
classical policies. Structural legality checks/common public posterior updates
are distinct from instrumented planner work. Charges are not FLOPs.

## Classical baseline

BeliefMCTS uses root-sampled public hypotheses, UCT selection, progressive
widening and random legal prefix proposals. It never enumerates a complete
protocol catalogue. Whole-batch outcomes are sampled and released before a
future belief-tree action; no mid-protocol feedback is used.

Rollout action selection uses public posterior expected utility, never the
simulated hidden hypothesis. The simulated hypothesis only supplies likelihoods
and task outcomes. The return guard requires a feasible direct reset edge at
each prefix; that covers the generated families but restricts arbitrary graphs.
This is a custom classical baseline, not official POMCP code, a reproduction of
its published performance, or evidence that all tuned MCTS variants fail.

## Difficulty DEV-V2 and root interpretation

Two seeds fixed before execution generate hierarchical diagnostic models:
coarse sensors identify a subtree; specialized sensors distinguish within it;
outside that subtree a specialized sensor is a fair coin. Larger capacity can
admit joint measurement routes. Operation durations independently expose
preparation, probe, transit, cleanup and task costs; energy is separate.

Each root has six configurations: base, weaker information, costly acquisition,
longer horizon, redundant channels and deeper hierarchy. These are repeated
configurations, not twelve independent samples. No instance was filtered by
oracle or planner performance. Both base roots and both redundant configurations
have distinct conservative structural fingerprints.

**Dimension-change caveat:** the generator's sequential RNG also changes priors
and some sparse edges when redundancy/depth changes. Those are prespecified
joint configuration shifts, not isolated causal effects of channel count.
Quality/cost/horizon changes keep the corresponding root model/graph fixed.
The GO quality witnesses compare algorithms on identical configurations;
the compute-growth witness uses the same-model horizon shift. No causal claim
about redundancy alone is supported. No post-outcome generator repair was made.

Legacy DEV is a separate deterministic-work policy audit, not fresh-root evidence
and not a reconstruction of every wall-clock-dependent old decision. It retains
the old capacities/horizons and public problem hashes; no new noise seeds are
used when exact integration succeeds.

## Final operational decision

The frozen JSON declares the complete numerical rule before any DEV-V2 results:

- On the same condition at both roots, all SMALL Beam/VOI/MCTS policies must have
  exact Bayes gaps at least `0.05 H`.
- A LARGE policy must be within `0.02 H` of solved Bayes and use at least four
  times its corresponding SMALL policy's realized expected model-call work.
- Reference work must grow by at least fourfold on a scale axis at both roots.
- Otherwise: HOLD if scale reference remains unresolved; STOP otherwise.

These are **engineering decision thresholds**, not a venue criterion, universal
hardness definition or proof that neural approximation is necessary. GO only
authorizes the research rationale for a later amortized-planning study. This
batch neither trains nor establishes superiority of a learned model.

## Public benchmark versus future confirmation

The already published v1 test/OOD seeds are public fixtures, not unopened blind
CONFIRM. A new high-entropy seed-material hash is committed without generating
instances. Its preimage is stored outside the repo with mode 0600, never printed
or included in evidence. Custody is local/user-controlled, **not independent
external blinding**. The final method and full protocol still need to be frozen
before the custodian releases that material; a seed hash alone is not a frozen
confirmation experiment. Old CONFIRM remains untouched.
