# 16-worker GPU-MPC v5 switch and startup

On 2026-09-26, the user clarified that “16GB” meant **16 parallel workers**. The 8-worker v4 process group was stopped after retaining 6 complete jobs and 8 partial checkpoints. The original v4 directory and `user_switch_to_16workers.json` remain intact.

`gpu_parallel_campaign_v5.py` copied 14 existing job directories into `artifacts/gpu_mpc_parallel_v5_16workers_20260926/`. All 14 copied event logs remained exact prefixes after resume. The per-job algorithm, physical environment, exact safety certificate, job list, map IDs, and calibration are unchanged. The v5 runner records 16 workers in a new manifest and inherits the v4 start and deadline, so the original eight-hour wall limit still ends at **2026-09-26 20:58:14 CST**.

The earlier 16-worker short benchmark peaked at 12,950 MiB summed unique worker RAM, left at least 8,346 MiB system `MemAvailable`, used at most 3,279 MiB GPU memory, and used no swap. The v5 runner requires at least 22 GiB physical RAM and 18 GiB available RAM before a 16-worker start; it does not count swap as capacity.

Startup health check: 16 jobs active; 7/48 complete; 23 job checkpoints present (including newly created checkpoints); no job or orchestrator error; approximately 7.7 GiB `MemAvailable`, 2,883 MiB GPU memory used, and 0 swap used. This is a startup check, not a completed experiment or an estimate of final performance. The campaign remains resumable and is handed back without monitoring to completion.
