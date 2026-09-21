# Mechanism Intervention Sweep v1

Frozen scope: 77 pairs / 130 primary branches; C1/C4 extend to 128 roots / 420 branches. C1–C7, D1/D2, pair-specific counterpart erasure and K8/K12 are all included. Deduplicated total: **2164 jobs**, including **70 prior C1 imports** and **2094 new runs**. No scientific results are available in this record.

C4/C7 use task-only oracle safety, no return/recharge, and wait at the current position when the queue is empty. No safe task is diagnostic unknown. Untested return measurements are null, never zero-cost successful returns.

The existing 16 IQ workers continue unchanged. The deferred supervisor waits for their complete integrity record and process exit, verifies/imports all70 raw results, then starts16 single-thread sweep workers. It checks first resumable checkpoints and only collects after all2094 new jobs complete. No outcomes influence condition selection. Worker errors preserve checkpoints and block complete-only collection.

Protocol: ../../MECHANISM_INTERVENTION_SWEEP_V1.md. Source/protocol/data hashes and source commit are in manifest.json. Populations are in bases.json, pairs.json and jobs.json. No training, new roots/seeds/maps or automatic follow-on experiments.

Engineering validation:21 focused tests (including reused Atlas checks), one historical task-only first-task/nested-resume smoke,420 archived full-window count and mode-time accounting checks. Raw physical smoke snapshot is local at /mnt/workspace/zjl-exp/mechanism-sweep-engineering-v1; source-stage hashes are retained there. Final smoke proof records any subsequent telemetry-only edits.

Run entry: `scripts/mechanism_intervention_sweep.py`. Commands: prepare, smoke, defer, worker, check, collect. A stopped worker resumes with `worker --worker <0..15> --resume`; never restart an existing shard without --resume. All commands verify the frozen contract. Active status lives in supervisor_status.json, worker_*/status.json and startup_health.json when the new workers have started. Completion requires integrity.json, not a startup record.

Long traces are gzip JSON in worker directories; checkpoints stay local. Collection produces all three windows (436.2,872.4,1308.6s), pair responses, broad-panel responses, root spreads, process metrics and complete integrity hashes. Unknown outcomes are never imputed. Finite-workload exhaustion is reported separately; intervention gap collapse does not alone establish a unique causal mechanism or required planning horizon.
