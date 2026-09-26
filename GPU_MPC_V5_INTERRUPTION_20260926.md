# 16-worker GPU-MPC interruption audit

Audited 2026-09-26 around 13:30 CST. The v5 orchestrator and all `gpu2d.evaluate` processes were absent, although `artifacts/gpu_mpc_parallel_v5_16workers_20260926/status.json` still said `running` (last written 13:26:20 CST). This status file is stale after an abrupt WSL restart; it is not evidence that the campaign is still running.

Evidence:

- The campaign has 37/48 completed job summaries and 11 partial jobs. Every partial job has a readable `checkpoint.pkl`. All 48 job event logs parse and agree with their per-job `status.json` decision counts. All 11 partial checkpoints unpickle and agree with the saved event counts.
- No current job `error.json` or `orchestrator_error.json` exists. The one `error.v1.json` found by a broad text search belongs to the preserved, previously repaired `hold_steps` failure; it is not a new v5 error.
- WSL boot history and the preceding journal show that the Linux instance restarted around 13:26–13:28 CST. Windows itself had not rebooted (`LastBootUpTime` 2026-09-21 23:10:12).
- The previous Linux kernel journal has no OOM-kill, killed-process, or memory-cgroup exhaustion entry near the interruption. Windows Resource-Exhaustion Detector/Resolver had no events in the queried 13:00–13:30 interval. A normal Python `MemoryError` or CUDA out-of-memory traceback is also absent.
- At the 16-worker startup check, about 7.7 GiB Linux `MemAvailable` remained, GPU memory usage was about 2.9 GiB of 8 GiB, and swap usage was zero. This was **not** a continuous memory trace; peak memory immediately before the restart is unknown. The WSL configuration limits its VM to 24 GB RAM and provides 8 GB swap.
- The previous WSL journal also contains repeated connectivity and `dxg` GPU-interface errors earlier in the run. They do not establish the restart cause.

Conclusion: a standard, logged Linux RAM or CUDA OOM is **not supported** by the available evidence. An external WSL shutdown/restart occurred; its initiating cause remains unknown. Late Windows/WSL resource pressure cannot be excluded without a contemporaneous host memory trace or a specific Windows/WSL termination event. Correlation with 16 workers alone does not prove a memory crash.

No experiment was restarted during this audit. Resuming should preserve the 37 completed results and 11 valid partial checkpoints, the frozen job list, and the original deadline. A lower concurrency level would require a new runner/output manifest rather than silently changing the v5 `workers=16` provenance.
