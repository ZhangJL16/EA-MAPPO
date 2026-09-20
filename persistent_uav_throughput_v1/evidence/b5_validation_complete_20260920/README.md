# Complete B5 reserve-SJF validation — 2026-09-20

Completed 2026-09-20T20:59:15 Asia/Shanghai. All 16 workers finished 270 unique jobs: 27 frozen regimes × 10 original validation seeds. No missing or duplicate jobs. Runtime source commit: `e218c60`; startup evidence commit: `99ad1d7`.

## Results

| Metric | Result |
|---|---:|
| Mean completed tasks | 14.61 |
| Energy depletion | 227/270 (84.07%) |
| Navigation failure | 42/270 (15.56%) |
| Recharge failure | 166/270 (61.48%) |

Recharge failures overlap other failure categories and must not be added to them. All 27 regimes fail the predeclared empirical 5% depletion budget: none has 0/10 observed depletions. This is an empirical diagnostic, not a safety certificate or a proof of regime infeasibility.

| Normalized battery capacity | Runs | Mean completed | Depletion | Navigation failure | Recharge failure |
|---|---:|---:|---:|---:|---:|
| 2 | 90 | 4.70 | 90/90 | 0/90 | 80/90 |
| 4 | 90 | 15.81 | 76/90 | 14/90 | 51/90 |
| 6 | 90 | 23.32 | 61/90 | 28/90 | 35/90 |

## Initial descriptive failure audit

227 energy depletions split into 166 return legs, 53 task legs and 8 waiting intervals. Of the 166 failed returns, 144 already had nonpositive predicted direct-return reserve at the return decision; 22 had positive predicted reserve. All 53 failed task legs were selected as estimated task+return feasible, not full-station infeasible fallbacks. All 42 navigation failures occurred on task legs.

These are links between logged predictions and observed outcomes, not an oracle-feasibility audit. They do not establish whether the initial problem is feasible, isolate all estimation versus planning effects, or support an RL-necessity claim. An executed return starts at the actual post-service position, whereas the earlier task+return predictor uses the task endpoint; these quantities must not be treated as identical counterfactual labels. Next work remains the separately authorized feasibility/estimation/planning decomposition.

## Unchanged scope

Frozen estimator and calibration, all 27 regimes, seeds, physics, navigator and parameters were retained. Strict reserve inequality has zero added margin; the existing full-station minimum-estimated-total fallback and empty-queue SOC threshold of 0.25 were retained. No B0–B3 sweep, evaluation, neural training, MPC/Oracle or final 98% kill test was started.

## Evidence

- `results.json`, `summary.json`, `eligibility.json`, `integrity.json`: complete ordered results, per-regime outcomes and collection checks.
- `initial_failure_audit.json`: descriptive prediction/failure counts above.
- `validation_raw.tar.gz`: every worker manifest, per-run events and decision diagnostics, workload streams, final results/checkpoints, resume pointers, logs and launch/health/plan records.
- `archive_inventory.json`: exact member sizes and SHA256 hashes.
- `SHA256SUMS`: all other published evidence files, including the archive.
- `workspace_consolidation/`: path relocation audit and calibration compatibility record; no credentials or tool-session data are included.
- [Startup evidence](../b5_validation_startup_20260920/README.md): 41 tests, real-navigation smoke and all 16 first-checkpoint checks.

The workspace is now entirely `/mnt/workspace/zjl-exp`, with a real `repo` directory. The old `/mnt/workspace/persistent-uav` directory is gone. Historical B4 archives remain immutable and retain their original path records. The current production outputs and final B5 checkpoints are preserved locally. The frozen model and host-built ARM native binary are referenced through hashes and existing inventories rather than duplicated in this upload.
