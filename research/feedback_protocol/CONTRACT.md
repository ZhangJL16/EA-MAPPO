# FPL v0.1 first-batch contract

Historical v0.1 contract. For the current utility, generation, splitting and
planning extensions see [CONTRACT_V02.md](CONTRACT_V02.md). Legacy fixture behavior
and reset-terminal semantics below remain the regression reference.

Scope: configurable finite public Bernoulli families on known deterministic operation graphs. This is the A0/A1 and comparison skeleton authorized on 2026-09-19, not completed delivery B/C.

## Decision and information contract

Every decision occurs at the unique reset node with full capacity. A policy receives an immutable `PublicProblem` and `PlannerState` (public posterior, remaining calendar budget, resource, released history). It commits a sequence ending at its first return to reset. No policy callback occurs during execution. Observations are independent conditional on the hypothesis, generated at operation completion and released together on return. Two operations sharing a channel share its Bernoulli parameter, not its random draw. All observed bits also contribute one unit of reward; other task utilities require a new versioned adapter.

`PrivateTruth` (label and RNG seed) belongs to the environment/evaluator. The runner does not pass the environment, evaluator oracle, truth or future randomness to policies. Python interface isolation is not a sandbox against malicious code. Bayes targets depend on public belief only. The reference includes zero-prior hypotheses only as public models; realized data impossible under the chosen prior raises an error rather than silently resetting belief.

## Resource, time and terminal contract

Operation duration is a positive integer; energy is an independently specified nonnegative integer. Debit before arrival-based reload. Return/cleanup takes the configured positive time; there is no free implicit reset edge. A prepared node is distinct from reset even if physically colocated. Protocols obey maximum measurements and per-channel count limits. Positive duration makes the finite-budget catalogue finite even with zero-energy cycles. Enumeration has an explicit node limit.

New primary terminal rule is `reset_by_budget`: finish at reset at or before T. Stopping early earns zero for the unused time. An unfinished sortie is not legal. The current synchronous smoke runner executes complete protocols; it has no mid-protocol checkpoint/resume claim. A future resumable runner must store pending operations, remaining operation time, generated-but-unreleased feedback, private RNG and public history atomically.

## Evaluation contract

Primary target is `V_known(theta,T,reset_by_budget) - E_theta[sum reward]` with the same committed protocol class and terminal rule. `known_model_value` is evaluator-only. `ExactBayes` maximizes public-prior expected cumulative reward and is not a minimax solver. The CLI's single-trajectory `realized_shortfall` is a noisy sample, can be negative and is not reported as expected regret or scientific evidence. Old `T*rho` certificates remain separate. On the anchor at T=12, the test verifies known values 3/5, 6/5, 19/5 for both capacities.

No minimax teacher, scientific sweep, neural learner, OOD generator or external-task adapter is implemented in v0.1. Exact planning uses rational arithmetic with state, wall-time, protocol-node and joint-outcome limits. A limit exception means unresolved, never an optimal or approximate answer. Python recursion/memory limits also abort; no hard memory cap or certified approximate solver is claimed.

## Fixture mapping and comparisons

The config contains dock, empty, A, B, AB and BA. The original calibration collapses BA into AB as an equivalent committed route type; the new graph keeps both orders. Set capacity=3/4 and max_measurements=1/2 for the four cells. No old result is reproduced by changing its protocol class: regression compares only the old committed, reset-terminal reference at T=6. The old arbitrary-terminal primitive lower certificate is outside this new control contract.

ChannelCover covers informative channels once, maximizing uncovered count per time, then posterior reward per time. It is intentionally a simple independent heuristic, not an implementation of OSSB or cost-allocation attainability. ExactBayes is the independent small-instance finite-budget planning reference. Both enumerate feasible protocols and therefore make no large-scale computation claim.

## Data and compute boundaries

Only `debug-only` inputs with budget <=12 are accepted by the CLI. No existing DEV/CONFIRM registry is imported. Same topology and parameter-family variants must share `family_id` and the same eventual split. The loader stores these identifiers but does not yet implement a dataset registry or enforce global split uniqueness.

The debug record seals all new package Python sources, pyproject, input-config bytes and effective-problem hash; it logs commitment, pre/post resource, operation times, release time, public state, decision time and solver state/branch counts. No dependencies beyond Python 3.11+ stdlib are imported. JSON probabilities should be supplied as rational strings for exact reproducibility. No installation into the old environment is needed.

Initial profiling, when separately scheduled, should vary channels, hypotheses, graph structure and horizon one dimension at a time, recording catalogue count, state/likelihood calls, latency and peak memory. Teacher, offline training and online inference cost must remain separate. No runtime-derived training budget has been estimated by this engineering smoke.
