# T5.1d — bounded adaptive-state reachability census

Scope is exactly the 32 already completed variants from the eight V2 pilot
groups in the pinned `teacher_v2_sizing_evidence.zip`. No new roots, public
problem changes, folds, objectives, V3 design, neural training or held-out runs.
The prior V2 runtime, mask, design and raw evidence remain unchanged.

## Prospective census definition

Depth 0 is the root prior. Depth 1 and 2 enumerate every legal complete
reset-to-first-reset **measuring-only batch**, followed by every positive
probability feedback. Joint-measurement routes are included, not just singleton
sensors. A measuring-only batch has nonempty observations and identically zero
task utility on every primitive operation. A legacy observation-as-reward
operation is not zero utility. No pure task/idle/empty-return batch is inserted.
Repeated sensing across batches is allowed; within-batch measurement limits
and debit-before-reload remain the original contract.

Deduplicate `(posterior, remaining horizon, reset resource)` within each
variant. Expand each reachable **(depth, sufficient state)** once. This retains
all available next decisions without incorrectly allowing a depth-2-only
arrival to expand again. Persist all incoming edges and depth memberships,
and one legal positive-probability witness history per state. These are
reachable-state counts, not sampling frequencies or policy probabilities.

All discovered states are labeled with the same exact Bayes objective and
original V2 per-state limits (3000 states, 10000 protocol nodes, 32 outcomes,
1e6 expansions, 3e6 model calls, 5-second watchdog). Existing-label intersections
are independently re-solved and checked for equal V/optimal sets; no partial
targets or retry on unresolved. Closure has separate prospective caps in
`configs/teacher_v2_census_v1.json` (100000 states, 1e6 edges, 30-second watchdog).
A capped closure is **incomplete**, not evidence of adaptive-state absence.

Prelabel closure checkpoints, source/plan/archive seal, single-writer lock and
per-variant completion receipts support unchanged resume. An interrupted
variant refuses automatic retry. Run startup check then finish the user's
explicitly requested bounded census; do not promote to another data run.

## Verification and interpretation

A separate normalized-belief BFS reconstructs all depth memberships and edges
without using the discovery traversal. It shares the model/likelihood runtime:
this is an implementation cross-check, not external independent verification.
Full records undergo public-history, candidate-completeness and rational Q/V
arithmetic audit. Report exact/unresolved, reachable non-prior states, sensing
required/optimal, and adaptive counts by family, fold, group and depth. Compare
state IDs against the old collector; preserve all new/missed adaptive witnesses.
Depth-specific counts can overlap and must not be summed as unique states.

Distinguish three outcomes without inventing a performance threshold:

- Many newly discovered adaptive states across multiple structures: evidence
  for a collection gap; do not redesign family on the old Case C alone.
- Continued scarcity even after complete depth-2 closure: evidence for a task
  economics/coverage problem **in this envelope**, not proof of absence under
  arbitrary histories or across all possible V2 roots.
- Incomplete closure, unresolved labels, or mixed/concentrated findings:
  report the uncertainty; do not force a binary impossibility claim.

The user's symmetric one-specialist inequality is useful motivation, not an
exact description of every V2 posterior. Outsider hypotheses, asymmetric priors,
joint routes and further continuation affect actual Bayes decisions. Exact
labels, not this simplified formula, determine census categories. No V3 or
collector redesign is executed in this task.
