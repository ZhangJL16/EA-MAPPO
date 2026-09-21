# Real-State Counterfactual Branching Atlas v1

User-authorized 2026-09-21. This is a privileged diagnostic in the existing
PersistentUAVThroughput-v1, not a new environment or proposed deployable method.
No training, reward change, new task distribution, margin adjustment, map change,
or modification to the frozen navigator/energy/charging/queue/timeout semantics.
The previous theory project is not used to select states or interpret outcomes.

## Frozen roots

Read and verify all archive members from B5's 270 validation runs and B4's
270 threshold=.75 validation runs. Only high-level `decision` events with queue
size >=2 and original remaining time >=1308.6 seconds are eligible. Decision
index is the zero-based ordinal among decision events, not the event-array index;
both indices are stored for exact historical reconstruction.

Sort SHA256 of compact JSON `[policy,regime,seed,decision_index]` in each of
54 source/regime cells, take the first four. Policy strings are `B4_0.75` and
`B5`; numeric entries are integers. Short cells retain all eligible roots without
replacement or supplementation. No failure, safety, energy or outcome filter.
Initial decision states are eligible. Roots shared physically across sources
are retained with their source labels; they are not independent replications.

The manifest records exact source archives, complete member verification, root
selection hashes, source revision, runtime/model/native hashes, dependencies,
frozen calibration and thread settings before physical branching.

## Exact reconstruction and pairing

Recreate each original episode with its archived seed and unchanged complete
workload. Replay the archived B4 or B5 actions, checking every event and decision
observation exactly. B5 passive diagnostic event fields are reproduced; B4 events
are not retroactively enriched. A mismatch is an engineering error, never unsafe.

Clone the full simulator and Python/NumPy/Torch RNG state. Preserve physics hidden
state, collision recovery history, clock, battery, queue, task IDs, arrival times,
stream cursor and future stream. Oracle clones never advance the actual parent or
its RNG. Stream hashes are verified at reconstruction and continuation completion.
There is one physical reset for historical replay, zero resets inside branches.

## Experiment A

Enumerate every queued task, in task-ID order, and direct return. Task branches
execute the actual task followed immediately by the actual return, stopping at
charger arrival before charging. If already docked, return is a zero-duration,
zero-energy success; no illegal full-station recharge is executed.

Safe_1 is true only when the task and return both succeed. Preserve task/return
energy and time, completion status, post-task battery/position, failures and the
unchanged strict B5 predicted feasibility. Failed leg costs are partial, not
complete mission labels. Branches reaching original T unfinished are unknown,
not unsafe. No extension of original T or navigator option-step limits.

## Experiment B and the explicit window clarification

The user explicitly selected **[t_root,t_root+1308.6s]**. The first selected task
counts toward both time and completed tasks. All actions from a root share the
same absolute window and future arrival stream. The root filter guarantees the
window does not exceed original T (floating grid tolerance 1e-7).

For every root with at least two known Safe_1 tasks, run each safe first task from
a fresh identical clone. Verify its completed first-task trajectory against A
except at a reporting cutoff that truncates a physical policy step. After that
use only the diagnostic Oracle-Safe SJF controller:

- Branch every current task through immediate return using original T, not the
  shorter reporting deadline. Among true-safe tasks choose the shortest **frozen
  estimator task time**, breaking ties by task ID. This avoids an estimator
  safety test while fixing one transparent ranking rule.
- With no safe task, take direct return only if it succeeds and recharge is legal.
- With an empty queue away from station, test and take direct return if safe.
- With an empty queue at station, wait using the original dock-idle semantics.
- If no tested safe legal action exists, end the diagnostic as
  `no_safe_continuation`. Never add illegal wait/rejection, force a known-unsafe
  task, or relabel the halt as physical death.

The fixed window is an observation cutoff: a temporary copy of the unchanged
config bounds the final flight/stationary step, then the original config is
restored. The complete stream is neither shortened nor redrawn. Nested oracle
queries always use the original physical horizon. Physical time spent computing
counterfactual queries is offline computation, not simulated operating time.

## Metrics, local comparisons and missingness

Primary: N(W), depletion/navigation failure and survival F(W), task+return safety,
time alive/failure time. Also record forced returns and their reasons, actual
mode times, overflow and explanatory battery/position/queue fields.

For physical failure within W, N(W) remains the attained task count and F(W)=0.
For a full surviving window F(W)=1. If the diagnostic stops with no safe action,
N(W) and F(W) are **unknown**, with N_observed and halt time preserved. No zero
imputation. Report unresolved roots separately; compute max-min and heuristic gaps
only when all known-safe first-action continuations at a root have resolved W.
Unknown A branches are retained and their possible effect on safe counts disclosed.

Local first-action comparisons use actual A task time (Safe-SJF), actual A task
energy (Safe-Energy), and actual post-task battery minus actual return energy
(Safe-Reserve). Ties use task ID. Report frozen-predicted Safe-SJF separately to
connect to B5, without substituting it for the requested actual-time comparison.
The offline best first action maximizes N(W), with task-ID ties; it is not an
optimal future scheduler. All futures use the same diagnostic controller.

Report 0/1/>=2 safe-task roots, including unknown categories, by battery 2/4/6,
source and their cross-tabulation. Distinguish predicted false-safe decisions
from ranking among true-safe tasks. Reversals compare actual task time, energy,
and post-task battery against future N(W); use strict inequalities with 1e-7
numerical tolerance and report comparable pair and root denominators. These
are descriptive within-root comparisons, not a proof that battery causally
explains or fails to explain all variation.

## Execution and retention

One added production script: `scripts/real_state_branching_atlas.py`.
Focused engineering tests exercise hash selection, legality, cutoff accounting,
paired streams, nested clone isolation/resume, and unknown-result handling.
One bounded real smoke covers both source replay formats and midflight recovery.
Production uses 16 single-thread workers, physical-step/time checkpoints, and
SIGTERM/SIGINT resume. Check the first checkpoints, then hand the continuing run
back. A separate supervisor only checks completion/errors and performs collect
when every assigned root is complete; it cannot tune or remove jobs.

Outputs under `evidence/real_state_branching_atlas_v1/`: manifest.json, roots.json,
verified inputs, per-worker resumable snapshots and raw branch traces, then
one_step_branches.jsonl, continuation_branches.jsonl, summary.json and integrity.json.
No summary is produced from a partial root set. Slow runs retain all roots.

Interpretation follows the user's frozen five cases: rare multi-safe states favor
safety/timely-return diagnosis; small gaps weaken a scheduling story; frequent
large gaps missed by local scalars motivate further mechanism analysis. The atlas
does not automatically authorize new methods, training or a new theoretical toy.
