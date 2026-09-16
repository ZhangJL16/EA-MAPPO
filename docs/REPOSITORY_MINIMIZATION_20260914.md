# Repository minimization record — 2026-09-14

## Retained scope

- Active formal lineage: `artifacts/paired_advantage_identification_20260914_v3`.
- Runtime model files: repaired energy head checkpoint/features, its repair
  record, and the frozen navigation `model.zip`.
- Minimal contract-only provenance needed by the active validator: CMI v2/v3/v4,
  PAI v1/v2, the v3 checkpoint head, and the selected acceleration decision.
- Current PAI protocol, contract template, locked collision protocol, collector,
  validator, analyzer, model helpers, external-anchor helper, and generic
  confirmation helpers.
- All environment and method implementation packages: `envs`, `experiments`,
  `agent`, `common`, `network`, `policy`, `calibration`, and `cert_runtime`.
- Python environment and package caches are untouched: `.venv`, `.uvcache`,
  `.uvtmp`, requirements, and `uv.toml`.

## Removed scope

- Historical/failed/paused/smoke experiment outputs outside the active runtime
  and validator closure.
- Raw outputs from invalid PAI v1/v2, retaining only contract/invalid markers.
- Historical research reports, venue reviews, paper drafts, old literature
  folders, plots/logs/caches, and obsolete experiment orchestration scripts.
- The GitHub Actions workflow, so a future push does not launch the old
  `review_bundle` test job.
- `review_bundle` reports, experiments, scripts, tests, caches, and its old
  physical source tree.

## review_bundle migration

`envs/UAVEnergyDeliverySAC.py` and the active PAI runtime transitively import
`review_bundle.envs` and `review_bundle.safety`. Their byte-identical code was
moved to `runtime_support/review_bundle` without editing any environment or
method import. A compatibility symlink named `review_bundle` preserves the
existing module path for active and future spawned children. A recursive diff,
direct import check, formal contract validation, and service health check all
passed after migration.

## Safety and recovery

This cleanup does not alter Git remotes, branches, `.git`, the active systemd
service, or any environment/method implementation file. Deleted historical
artifacts are not recoverable from the working tree; tracked files remain
recoverable from Git history. The active formal lineage and runtime checkpoints
remain resumable.

## Result

The workspace decreased from approximately 204 GiB to 7.4 GiB, of which the
untouched `.venv` accounts for about 6.8 GiB. `artifacts/` decreased from about
197 GiB to 331 MiB. Scripts decreased from 243 files to 20, and 12 focused tests
passed after cleanup. No GitHub workflow was executed.
