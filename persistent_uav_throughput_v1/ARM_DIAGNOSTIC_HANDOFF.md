# ARM diagnostic handoff after v1.4 wait-grid repair

Local B4 validation stopped at 382/1080, at regime 9 / threshold 0.5 / seed 920260922.
All 382 completed rows and the last valid checkpoint are preserved in evidence/v1_4.
These are historical partial local results, not inputs to the new ARM result aggregate.
No local production run was resumed by this repair.

## Cause and repair

At t=2618.1999999998893, the next arrival was 2764.9. The subtraction produced
146.70000000011078 seconds. The old ceiling-based membership test incorrectly
compared it with 146.75, although its nearest-grid error is only about 1.1e-10 seconds.
Stationary intervals now use nearest-grid membership with the existing 1e-7 second
tolerance, then snap to that grid interval. Nonfinite, negative and truly off-grid
values are rejected. Flight clock/physics, task stream generation and frozen regimes
are unchanged. Explicit charge and away-hover energy use the accepted grid duration.
Dock idle remains zero consumption without automatic charging.

On exceptions, evaluation now writes status=error while retaining the last valid
checkpoint pointer and checkpoint progress. It does not snapshot partially mutated
failed state. The original stale status/error files are preserved in original_failure;
the old local runtime status has been corrected to error.

## Evidence

- evidence/v1_4/tests.txt: 34 tests PASS.
- evidence/v1_4/wait_grid_replay.json: identical states/events/summaries until step 51;
  old method raises, new method reaches t=2764.9, receives task and returns to IDLE.
- evidence/v1_4/continuing_smoke/: real frozen SAC, 3 services, 1 recharge, dock idle;
  full outcome after midflight resume equal, physical resets=1.
- evidence/v1_4/local_partial_results.json: all 382 historical local summaries.
- evidence/v1_4/original_failure/: original manifest/error/status/checkpoint and reproduction.
- evidence/v1_4/calibration_compatibility.json: local source-change audit only.
  The prior v1.3 compatibility record remains unchanged.

## Instructions for the ARM Codex

1. Synchronize this repair while preserving your migration_20260920 files, uv project
   and local ARM binary. Do not blindly discard local changes or overwrite ARM .so
   with a checked-in x86_64 binary.
2. Extend your existing ARM migration compatibility record to the repaired source.
   Preserve historical manifests. The local v1.4 compatibility file embeds local
   paths/builds and is NOT the ARM compatibility file. Verify changes, model/SAC hash,
   frozen-regime hash, and record the ARM binary hash; do not disable provenance.
3. Run the updated focused tests and one continuing smoke on ARM. Native solver must
   still load. The full archived checkpoint replay is local x86 evidence; cross-platform
   bitwise equivalence is not required or claimed. Its numerical regression is in tests.
4. Start a NEW B4 validation output directory for all 27 regimes, the same four thresholds
   and ten validation seeds: 1080 runs. No local-382 import, no cross-platform checkpoint
   resume and no duplicate rows. Reuse existing completed calibration and frozen regimes.
5. Check one resumable startup checkpoint, then hand back the running process. Do not
   monitor completion, automatically advance to evaluation, or launch additional runs.

No neural training, MPC/Oracle, new calibration, regime/seed selection, or final 98% kill test.
The current entry point is sequential; this repair does not add parallel workers.
Any future sharding must keep nonoverlapping job IDs and audited aggregation.
Use the existing /mnt/workspace/persistent-uav/.venv interpreter and
PERSISTENT_UAV_LEGACY_ROOT pointing to the cloned repository.
User-facing terminal commands must each fit on ONE line.
