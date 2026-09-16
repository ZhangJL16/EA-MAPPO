# Bundling calibration startup audit

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-09-16
- Verification Status: UNVERIFIED scientifically; startup integrity checks passed
- Version: bundling-calibration-startup-v1
- Authority: user-approved prepare + startup only

## Commands and execution

Working directory: `/home/zjl/mappo`.

```bash
.venv/bin/python scripts/run_bundling_calibration.py prepare --output artifacts/bundling_calibration_startup_v1_20260916
.venv/bin/python scripts/run_bundling_calibration.py startup --output artifacts/bundling_calibration_startup_v1_20260916
```

Both commands exited 0. Prepare sealed the gain, alternative and coefficient
audits with source hashes before scientific sampling. Startup reported about
0.145 seconds of internal grid execution and stopped normally.

## Integrity findings

- One fixed paired seed; four cells times five methods: 20/20 durable checkpoints.
- Every trajectory has exactly 128 primitive steps and time denominator 128.
- Read-only replay of action/state fields matched pose, debit-before-reload,
  protocol restrictions and primitive clock in all 2,560 logged steps.
- Source and prediction seals revalidated after startup.
- RNG checkpoint structures accepted by Python's `setstate`; full continuation
  identity was already covered by the focused test, not rerun on scientific data.
- Some checkpoints stop inside a sortie; pending action offset and feedback are
  retained. They are not truncated/replaced by complete-cycle outcomes.
- Artifact directory contains 23 files, 831,132 bytes at audit time.
- No runner lock remains; no background continuation was launched.
- No legacy DEV/CONFIRM read is in this isolated runner's execution path.
  Runner receipt records CONFIRM access 0 and routing-library worlds 0.

Only integrity fields were inspected in this audit. No scientific coefficient
ordering, pseudo-regret contrast, task-outcome comparison, confidence interval
or mechanism Gate was calculated or reported. The startup is NOT a completed
4096-step calibration or evidence that its predicted ordering occurs.

## Artifacts and next boundary

- `artifacts/bundling_calibration_startup_v1_20260916/manifest.json`
- `artifacts/bundling_calibration_startup_v1_20260916/sealed_predictions.json`
- `artifacts/bundling_calibration_startup_v1_20260916/health_128.json`
- Twenty cell/method checkpoints in the same directory.

Execution is stopped and resumable. The `resume` command to time 4096, scientific
analysis, routing descriptors and seed sweeps were NOT executed or automatically
authorized. Prior experiment holds and frozen semantics remain unchanged.
