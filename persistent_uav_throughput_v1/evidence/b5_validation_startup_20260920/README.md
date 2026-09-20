# B5 validation startup handoff — 2026-09-20

Started 2026-09-20T20:15:16 Asia/Shanghai. Exactly 270 predeclared B5 reserve-SJF jobs
(27 frozen regimes × 10 validation seeds), partitioned across 16 single-thread CPU
workers. Workers 00–10 own 20 jobs each; workers 11–15 own 10 each. No NPU/GPU use.
Runtime source commit: e218c60. All 16 bounded first-checkpoint checks passed;
checksums/deserialization, manifest and provenance, public prediction telemetry,
finite observations, process identity and actual thread counts verified.

41 focused/regression tests passed. One tiny frozen-navigation telemetry smoke
passed, including midflight recovery equivalence. Frozen estimator, scheduler,
physics, model, dependencies, calibration and regimes are unchanged. Added source
is passive diagnostics; evaluator calls it only for reserve_sjf. Compatibility
checks continue to reject old B4 source provenance. No calibration rerun or margin
tuning occurred. Existing empty-queue SOC threshold is 0.25; strict task+return
reserve uses zero added margin. Full-station infeasible fallback is retained.

Production output: `/mnt/workspace/zjl-exp/repo/persistent_uav_throughput_v1/artifacts/b5_validation_arm64_20260920`.

`launch.json` records worker PIDs/commands. Each worker owns its manifest,
status/resume pointer, atomic checkpoints and run artifacts. Each B5 decision
records predictions for all currently visible candidates; failures distinguish
service, return and waiting/charging. Completed run artifacts add leg outcomes
and separately labeled final-run outcome flags. See ../../B5_VALIDATION.md.

The evidence contains immutable startup manifests and checks, not a copy of the
ongoing changing production snapshots. The initial health report is not a live
status page. `smoke.py` and `record_compatibility.py` are copies of host audit
scripts from `migration_20260920/b5_validation`; their relative path assumptions
refer to that original location.

To stop safely, verify the live command against launch.json and send SIGTERM to
that worker PID; the runner checkpoints after the current policy step. Resume
only stopped/incomplete workers with the corresponding resume_commands.txt line.
Never launch a second writer into an active worker directory.

Manual collection after all workers finish:

```bash
/mnt/workspace/zjl-exp/.venv/bin/python /mnt/workspace/zjl-exp/repo/persistent_uav_throughput_v1/scripts/b5_validation.py collect --output /mnt/workspace/zjl-exp/repo/persistent_uav_throughput_v1/artifacts/b5_validation_arm64_20260920
```

Collection verifies all 270 jobs, snapshots, per-run artifacts and telemetry, and
reports completed tasks plus depletion/navigation/recharge failure rates for all
27 regimes. Empirical eligibility means 0/10 observed depletions, not a safety
certificate or proof that the underlying regime is feasible.

Handed back running after startup health. No completion monitoring, automatic
collection, B0–B3 sweep, evaluation, training, MPC/Oracle or 98% kill test. Next
interpretation requires feasibility/estimation/planning decomposition.
