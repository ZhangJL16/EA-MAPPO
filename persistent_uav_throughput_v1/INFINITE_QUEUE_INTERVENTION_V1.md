# Infinite-queue intervention v1 — frozen causal replay scope

User authorizes only this intervention on the 41 previously selected matched-divergent pairs. Original benchmark and completed Atlas remain immutable. No training, new seeds, expanded roots or alternative downstream policies.

## Intervention

Restore each exact archived root snapshot, including navigator/physics, task stream/cursor and RNG. From root onward set only queue_capacity to max(5, number of tasks in the complete frozen stream). This finite bound is >= all tasks that can ever occupy the queue, even in nested original-horizon oracle queries, and is therefore equivalent to no overflow on this bounded workload. Extend queue-occupancy accounting bins with zeros; preserve prior bin values, queue, current task, clock and all counters. Never resurrect tasks rejected before root.

Keep original root first actions, first-task physical trajectory, SAC, map, battery/energy, recharge, original horizon, absolute arrival times/locations, root-origin W=1308.6, and the unchanged Atlas Oracle-Safe SJF implementation. All nested queries inherit the intervened queue capacity. Full continuation first-task completion must match its archived A completion exactly. Queue size can exceed five; all candidates must still be enumerated. No cap on oracle candidates and no fallback heuristic for runtime savings.

41 pairs map to 70 unique (root, first task) continuations on 29 roots. Compute each unique continuation once, retain every pair mapping, and compare to its frozen original K=5 N(W). Selection is explicitly conditioned on prior divergence, so this is not a population-average intervention effect.

## Frozen comparison

For every pair retain signed delta_old=N_i-N_j, delta_new, absolute gaps and change abs(delta_new)-abs(delta_old). Categories: disappeared (new zero), smaller/equal/larger absolute gap, sign reversed. Report all pairs, pair- and root-balanced means, and source/battery strata. Report new outcomes, physical failure, no-safe diagnostic halt and unknowns. Compute paired changes only when both new branches are resolved, comparing against the same subset's original gap; no-safe-continuation stays unknown, not zero or death. Never delete slow/unfavorable configurations.

All actual future arrivals in completed intervention windows must be accepted, with zero new overflow; across paired complete branches the full arrival IDs/times must match the archived exogenous stream. Report if the intervention creates unknown continuations. No numerical significance claim or IID test; pairs share branches, roots, runs and seeds.

Interpret as total effect of removing the future finite-buffer constraint from these historical states. Larger queues also change candidate sets and service decisions; gap reduction is not a uniquely identified mediation fraction. Unchanged mean gap cannot rule out heterogeneous admission effects. No next intervention is auto-authorized.

## Execution and integrity

Freeze plan, pair/job manifests, implementation/dependency hashes and source revision before full launch. Use existing snapshots without replaying prefixes under altered queues. Focused engineering tests cover capacity bound/accounting and clone isolation; one real archived root first-task control comparison plus nested checkpoint/resume smoke; then 16 single-thread workers with resumable full-state snapshots and one startup health check. Separate monitor only waits for completion/errors and collects after every worker completes. No intermediate outcome-based tuning. Stop at startup handoff; do not continuously monitor scientific results during the turn.
