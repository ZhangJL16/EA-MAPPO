# P2-A2 startup evidence (not completed results)

See [protocol and handoff](../../../../docs/REGENERATIVE_CONTROL_P2A2_20260919.md).
`contract.json` pins the actual runtime source/checkpoint and >=.98 kill threshold.
`root.json.gz` contains the canonical-H paid setup and forced first-task branches,
plus the original nominated-state descriptor. `smoke_node.json.gz` is one actual
hashed full branch receipt, not a synthetic solver test. `startup_health.json`
records the health observation13 saved nodes/frontier12 and the PID at handoff.
It is deliberately not refreshed by monitoring the experiment to completion.
The runtime retains full pickle snapshots and atomic node receipts for resume.

Seven focused tests passed. No completed ratio or scientific kill conclusion is
in this bundle. `launch_failed_interpreter.json` preserves the initial import-only
launcher failure; `launch.json` records the corrected virtualenv process.
No new training, random census, environment edits, state binning or RNG lookahead.
