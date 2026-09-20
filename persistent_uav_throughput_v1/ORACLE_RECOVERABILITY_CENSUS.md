# Oracle Recoverability Census

The review of a7f298f authorizes a bounded, privileged diagnostic, not a deployable
policy, OracleSafe-SJF sweep, MPC, training, or 98% kill test.

## Fixed selection before branch outcomes

Use the already published B5 and stranding evidence, with all archive hashes checked.

| Group | Historical predecessor | States | Branches |
|---|---|---:|---:|
| A | Full-station predicted-infeasible fallback task decision | 67 | 182 |
| B | Empty-queue decision immediately after last task completion: 40 waiting-margin losses + 8 waiting depletions | 48 | 48 |
| C | Accepted-task decision: 37 task-error margin flips + 53 task depletions | 90 | 331 |
| Total | Distinct recorded decisions | 205 | 561 |

For A/C, branch all queued tasks, each followed by immediate return. For B, branch
immediate return. Also test legal direct return in C: no successful task candidate
does not exclude safe return/recharge. Do not invent a recharge action at a full
station in A. No branch selection or early stopping depends on success outcomes.

## State reconstruction and cloning

Historical storage retained final/recent checkpoints, not all predecessor states.
Reconstruct each selected run from its original regime, frozen map, seed and task
stream. Replay archived actions, verify that unchanged B5 would choose them, and
compare every emitted event and decision observation exactly after JSON type
normalization. Verify the complete prefix and target observation before branching.
A mismatch is an engineering error and stops the affected worker; it is never a
scientific unsafe label or silently omitted case.

Save the reconstructed full environment (including hidden physical/controller/RNG
state and original task stream) plus Python/NumPy/Torch RNG state. Each branch
starts from an independent serialized clone of that root. Position/battery-only
reconstruction is prohibited. Preserve immutable root snapshots for later audit.
The replay is diagnostic reconstruction, not another B5 evaluation or a new policy.

## Operational predicate and limits

`safe=true` means that the chosen task completed and the charger was reached, or
that direct return reached the charger, without energy depletion or navigation
timeout under the unchanged frozen navigator. Stop at charger arrival, before
charging or executing another task. If a task ends at the charger, no illegal
zero-time recharge is inserted. Collision recovery remains nonterminal and its
unified count is recorded; this predicate is not collision-free certification.

The original absolute horizon is retained. An unfinished leg at the horizon is
`safe=null` / `horizon_censored`, not a failure or success. Depletion and navigation
timeout are distinct. Failed-leg energy is censored; do not report it as completed
mission cost. The physical option limit remains 4000 steps per leg, with an explicit
branch watchdog at 8000 policy steps for task+return or 4000 for direct return.
Replay is bounded by the original horizon divided by the physical timestep.

This measures **bounded continuation success under the frozen controller at
selected failure-predecessor states**. It is not the maximal controlled viability
kernel, an all-policy impossibility proof, or an estimate of the 5% episode risk
budget. One frozen-controller timeout does not prove energy infeasibility. Even
no successful tested task+return branch does not exclude multi-task routes or
another low-level controller; conclusions must stay within this diagnostic scope.

For B, compare immediate-return outcome at the verified pre-idle state with the
already observed later failure. This is a paired diagnostic of the idle choice,
not a measured throughput result for a modified empty-queue policy.

## Execution

`scripts/oracle_recoverability_census.py` provides `prepare`, `smoke`, `launch`,
`worker`, `check`, and manual `collect`. One immutable plan fixes inputs, code,
runtime hashes, all states and branches before launch. 16 CPU workers receive
disjoint round-robin state lists. Original runtime sources and provenance stay
unchanged; the diagnostic is a separate script.

Workers checkpoint every 100 replay/branch policy steps or 60 seconds, and on
SIGTERM/SIGINT after the current step. Resume verifies the plan/implementation/
source hashes and exclusive worker lock. The last valid checkpoint is retained
on an error. Branch roots and completed traces are saved independently.

Only necessary tests, one tiny real nonzero-time replay/clone/resume smoke and
bounded first-checkpoint checks precede handoff. Startup health may be in replay:
`historical_prefix_exact` does not claim that every branch root is already reached.
No background collector or automatic subsequent policy is launched.
