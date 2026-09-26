# 8-worker GPU-MPC resume after WSL restart

At the user's request on 2026-09-26, `gpu_parallel_campaign_v6.py` resumed from the interrupted v5 directory in a separate output directory: `artifacts/gpu_mpc_parallel_v6_8workers_20260926/`. The v5 source artifacts were preserved. The new manifest fixes `workers=8`, records the unchanged algorithm source hashes and all 48 jobs, and inherits the original campaign deadline of 2026-09-26 20:58:14 CST.

The migration copied 48 job directories: 37 completed and 11 valid partial checkpoints. Startup health check found 8 active jobs, 41/48 completed, 11 checkpoints written after the v6 manifest, no new job/orchestrator errors, about 14 GiB Linux `MemAvailable`, and 0 swap used. This is an initial health check, not the final result. The run remains resumable and was not monitored to completion.

The preceding 16-worker interruption analysis is in `GPU_MPC_V5_INTERRUPTION_20260926.md`; it did not establish an out-of-memory failure.
