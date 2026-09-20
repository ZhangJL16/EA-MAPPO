# ARM B4 validation — 2026-09-20

Completed at 19:13:14 Asia/Shanghai. All 16 workers completed 1080 unique jobs: 27 regimes × 4 thresholds × 10 validation seeds. No missing or duplicate jobs; no original-server x86 results imported. Runtime source revision: `5bf3025`.

## Results

| Threshold | Runs | Energy depletion | Depletion rate | Mean completed tasks |
|---|---:|---:|---:|---:|
| 0.10 | 270 | 270 | 100.00% | 4.29 |
| 0.25 | 270 | 260 | 96.30% | 6.66 |
| 0.50 | 270 | 192 | 71.11% | 15.44 |
| 0.75 | 270 | 139 | 51.48% | 20.56 |

All 27 regimes have null selected thresholds: none of the four candidates meets the predeclared empirical depletion-rate budget of 5%. Each candidate has 10 validation seeds per regime, so empirical eligibility requires zero observed depletions. This is not a statistical 5% safety certificate. Pooled values above are descriptive across regimes, not a replacement for per-regime selection.

These are B4 validation results, not a B0–B5 comparison or a final viability/kill test. Estimator-error auditing and estimation/planning decomposition remain outstanding. Evaluation, neural training, MPC and Oracle were not launched.

## Evidence

- `results.json`: all 1080 run summaries, ordered by the predeclared job list.
- `summary.json`: 108 regime/threshold cells, including uncertainty and failure rates.
- `thresholds.json`: all 27 selection outcomes and calibration hash.
- `integrity.json`: collection checks.
- `validation_raw.tar.gz`: original worker manifests, results/events, workload streams, logs, final checkpoints, resume pointers, launch and sharding records.
- `migration_and_repair.tar.gz`: migration compatibility, smoke/test records, orchestration scripts, startup health and directory-repair records. Repeated pre-repair output backups and compiled binaries are retained locally, not duplicated in this archive.
- `archive_inventory.json`: SHA256 and size of every archived file.
- `SHA256SUMS`: hashes of all other files in this evidence directory.

## Directory migration incident

Renaming the workspace during execution left active processes writing absolute paths under the old directory. Seven workers failed during final threshold output because the calibration file was absent at that path. Outputs were backed up and merged, original canonical paths restored, and those seven workers resumed from validated checkpoints. All subsequently completed. No experimental source, seeds, parameters or frozen data were changed.

The canonical repository remains `/mnt/workspace/persistent-uav/repo`; `/mnt/workspace/zjl-exp/repo` links to it. The old environment and migration paths link to their counterparts in `zjl-exp`. Historical manifests intentionally retain absolute paths. Archived scripts are host-specific audit evidence; relocation requires an explicit compatibility audit.

The frozen navigation model is referenced by existing hash/inventory; this result upload does not duplicate its 26 MB runtime asset or replace the tracked native library with a host-specific ARM build.
