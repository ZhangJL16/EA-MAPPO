# Mechanism Intervention Sweep v1 — frozen full campaign

Authorized after 22bb326. Preserve the running 70-job infinite-queue v1 byte-for-byte. Freeze this complete design before reading its scientific results; wait only for its verified completion, import all 70 jobs, then run every remaining job without outcome-driven condition selection. No training or new roots/seeds/maps. Full optional K dose panel and broader structural panel are included.

## Populations and deduplication

Primary: all 77 same-arrival + joint-matched pairs in retained-counterpart v1: 41 prior divergent and 36 tied, 50 roots, 130 unique first actions. Keep all ties and failures. Broad panel: original 128 fully-resolved multi-safe Atlas roots, all 420 first actions (not the 17 original unresolved roots). C1 and C4 cover the broad panel, with the primary population a subset. Erasure is pair-specific: 154 distinct pair-side jobs, not shared even when first actions coincide.

C0 is the archived original, never rerun for science. C1–C7 primary = 910; D1/D2 = 260; erasure = 154; K8/K12 = 260; broad C1/C4 add 580 after overlap, total 2164 condition jobs. Import 70 exact C1 jobs; 2094 new jobs. All populations, baseline hashes and mappings are frozen in the manifest. This is retrospective mechanism sampling, not natural occupancy sampling or prospective discovery validation.

## Conditions

- C1 Infinite Queue: exact existing definition, root onward queue_capacity=max(5,len(full frozen task stream)); extend only accounting bins; no resurrection of pre-root drops. Existing 70 reused only after current batch integrity passes.
- C2 Instant Charge: retain physical return and finite energy. On arriving at charger, next charging transition fills to B with zero elapsed time and records charge_complete. No instantaneous transport. Docked partial battery also fills on legal charging. Return oracle stops at charger arrival as before.
- C3 No Future Arrivals: preserve root queue/history/cursor, remove only unarrived suffix from active task stream; active hash differs and original hash is retained. No future arrivals in main or nested query. Config's positive arrival-rate metadata is not used to generate replacements.
- C4 No Energy: user-confirmed task-only oracle success (task completed, no navigation failure), no return test, recharge or return action. Empty queue waits at present location. If no task-only safe continuation, unknown diagnostic halt. Set root battery/capacity to 1e9, preserve physical consumption telemetry; assert energy never approaches half this sentinel so inadequate bound is an engineering error, not a no-energy depletion result. Frozen SAC observation has no energy input; no navigator retraining. Navigation timeout remains.
- C5 Post-first Forced Regeneration: once after first task, perform real return then ordinary full charge, all time charged to root-origin W; then original Oracle-Safe SJF. Arrivals/overflow continue during return/charge. If already fully docked, no illegal zero-time recharge. This aligns station/full-charge state only after unequal travel/charge times, not histories or absolute phases.
- C6 Infinite Queue + Instant Charge: combine C1/C2 definitions.
- C7 No Arrivals + No Energy: combine C3/C4. Preserve finite waiting queue definition (no new arrivals); navigation timeout remains.
- E Counterpart Erasure: each pair-side removes only the retained counterpart immediately after first completion and endpoint arrivals, before downstream action. No reward/completion credit or elapsed time. Record erasure explicitly and adjust task-conservation accounting with a separate erased count, never fabricate completion. Preserve positions, resources, times, history and other tasks.
- D1/D2: original environment and task+return oracle-safe menu, choose minimum measured task energy / maximum measured return reserve respectively, task ID ties. Same empty/no-safe rules as Atlas. Original SJF uses frozen predicted task time.
- K8/K12: only future queue cap and occupancy accounting dimension change to 8 or 12.

All conditions preserve root first action and original absolute W=1308.6s. Nested task/return feasibility uses original T, not W, except removal of return in C4/C7. Never limit candidates due to queue growth. No main/nested RNG reset, task redraw, or physical reset. Root snapshots restored under the original environment before branch-local edits.

## Metrics and collection

At W=436.2,872.4,1308.6 seconds, score prefixes of the saved longest-window traces, not independently rerun shorter-policy episodes. Record signed/absolute pair gaps, mean pair throughput, disappeared (old nonzero,new zero), smaller/equal/larger absolute gap, reversed (strict opposite nonzero signs), induced (old zero,new nonzero). Compare each condition against C0 at the same W. Unknown is never imputed; a halted branch may still have valid earlier prefixes. Real failures are absorbing with known task count, failure status remains explicit.

Process outputs: flight/charge/wait/failed time, recharge requests, forced returns, overflow, accepted arrivals, queue occupancy integral and completed task IDs. Also retain physical energy consumed from the unchanged physics ledger and erasure events. For archival imports lacking dedicated integral fields, derive from actual event queues. No-arrival runs report available-task count / task exhaustion to distinguish mechanical finite-workload saturation from a uniquely identified feedback mechanism.

Report primary 77 pairs (prior divergent/tied separately) for all conditions; broad C1/C4 all 551 root-internal pairs and root max-minus-min gaps. Pair and root-weighted summaries, source/battery strata, same-resolved-subset C0 baselines. Raw response fingerprints for every pair/condition/window, no outcome-tuned clustering or weights. C6 interaction can be described by the common-complete quartet of C0/C1/C2/C6 signed and absolute gaps; this is not automatically unique mediation.

## Frozen interpretation boundaries

Gap disappearance is an exact within-pair observation, not a causal attribution percentage. Large/small aggregate effects will be reported continuously rather than inventing a post-hoc cutoff. Queue/charge/energy interventions change reachable states and candidate choices as well as the named edge. C3/C7 can collapse gaps merely because both finish the finite root task set. C5 does not align absolute time/remaining queue. Energy-free gap persistence establishes existence under that modified system, not that energy never matters. Erasure reduces workload; controller robustness only concerns these three fixed policies. Window-dependent counts do not identify required planning horizon.

The panel can support or weaken mechanism candidates; no single collapse proves necessity, sufficiency or generality. No IID p-values or significance claims; shared branches/roots/seeds and retrospective selection are retained. Do not call observed sign reversals proof of stochastic amplification. No next experiment auto-authorized.

## Execution

Separate intervention subclass/module: never edit original persistent_uav runtime or the running infinite-queue script. Freeze source SHA, script/engine/protocol hashes and every job before launch. Focused engineering tests for each intervention, physical smoke for energy-free first-task path and checkpoint/resume, and parity with C1 before enabling 16 single-thread workers. Preserve the current 16-worker allocation until it finishes; a deferred supervisor imports only after its integrity.json is complete, then launches this sweep, verifies first nonzero checkpoints, and collects/analyzes only after all jobs finish. Queue order determined by job IDs, never outcomes. Full-state parent/nested checkpoints; crashes retain resume and error records. All conditions remain required even if C1 looks uninteresting. No required completion time is promised.
