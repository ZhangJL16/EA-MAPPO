# T5.1e — exact Bellman-residual decomposition

Only the 1,171 census states and original 32 variants / eight pilot structures
are in scope. No new roots, changed rewards, V2.1/V3 or learned methods.
Small/large budgets, width 8, Beam depth 2 and episode multiplier 4 reproduce
the fourth-batch policy settings, not a newly tuned weak baseline. This task
does **not** revisit fourth-batch DEV roots; it cannot causally attribute their
specific gaps using a different teacher family.

## Two distinct comparisons

1. `state_residuals.jsonl`: fresh planner at each census state, full per-select
   allowance and a fresh episode pool. Compute exact `V*-Q*(selected action)`
   from saved complete Q catalogues. Small-minus-large is a paired, unweighted
   *decision* comparison on the same public state. It is not a deployment gain.
2. `occupancy_decomposition.json`: roll each frozen policy from the original
   prior, branching its full owned state, including episode work and history.
   A belief/H match is NOT enough to merge policy contexts. Save path/context,
   pre/post work, action, exact occupancy mass, residual, and weighted residual.
   Sum by sufficient state only after evaluation, keeping context rows visible.

An actual policy trajectory may leave the measurement-only depth-two census
through task actions or longer histories. On-demand **evaluation-only** exact
V/Q labels on these same public problems are necessary for complete telescoping.
They are logged separately, never admitted to a training corpus. Original
per-label caps apply; failure remains unresolved. No synthetic root is generated.

For each successful episode, require rational equality:

`sum_context occupancy * (V* - Q*(chosen)) = V*(root) - V_policy(root)`.

STOP is an action with Q=0 and absorbs the remaining horizon; its residual is
V*, not zero. At H=0 the terminal value is zero. Chosen-route Q includes optimal
future continuation, whereas V_policy uses the actual preserved-budget policy.
Additionally cross-check V_policy against the unchanged ExactPolicyEvaluator's
hypothesis-wise utilities mixed by the public prior, with separate work reporting.

## Attribution

Keep prior/nonprior, measurement-required/optional/nonmeasuring-optimal,
optimal protocol-type set (ties preserved), chosen type, remaining H, capacity,
family, structural group and fold. Report work-limit/autopilot flags as possible
implementation bottlenecks, not evidence of deep information reasoning.
Small/large occupancy distributions differ: subtract their total gaps (or signed
category contributions) rather than interpreting uniform state-residual changes
as the episode improvement.

No numerical acceptance threshold is invented. Report exact counts, all gaps,
category shares and structural coverage, then recommend one route with its
scope/limitations. Sparse required sensing alone neither proves nor disproves
the need for amortization. Zero on a subset is not zero across the task family.
All timings are one-machine diagnostics, not expected latency estimates.

Implementation is a new script outside frozen fpl/fpl_v2. Per-variant source/
plan/archive seals, atomic receipts and no interrupted automatic retry preserve
the old data and permit unchanged resume. Run focused tests and a first-variant
identity check, then complete the explicitly requested bounded analysis.
