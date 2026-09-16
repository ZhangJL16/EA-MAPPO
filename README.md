# Paired-Advantage Identification

This repository is currently scoped to one research question: whether the
paired C/R advantage has stable decision heterogeneity (Gate H), and only then
whether full legal history adds predictive value beyond the latest legal frame
(Gate I).

## Completed experiment

The active formal lineage is
`artifacts/paired_advantage_identification_20260914_v3`. It uses 96 fresh DEV
worlds, 64 paired-CRN resamples per action and anchor, four world processes, and
eight branch workers per world. The initial service stopped after a child Python
SIGSEGV at 84/96; the unchanged resume completed all 96 worlds.

The complete raw audit passed. Gate H passed, establishing stable paired C/R
advantage heterogeneity on DEV. Gate I failed: the tested full legal history did
not improve out-of-world prediction or selected utility over latest/constant
baselines. The frozen stopping rule keeps CONFIRM unopened and METHOD-TRAIN
unauthorized. See `docs/PAIRED_ADVANTAGE_DEV_RESULT_20260915.md`.

Protocol and machine contract:

- `docs/PAIRED_ADVANTAGE_IDENTIFICATION_PROTOCOL_20260914.md`
- `docs/PAIRED_ADVANTAGE_IDENTIFICATION_CONTRACT_TEMPLATE.json`
- `docs/LOCKED_COLLISION_RECOVERY_PROTOCOL.md`

## Current research direction

Larger history-model training is stopped. An existing-DEV-only factor audit
finds that C/R sign changes are dominated by controller-conditioned task
reachability/stuckness rather than energy exhaustion or contact. Compact legal
progress/stall history is suggestive but not yet proven decision-sufficient;
simple privileged straight-path geometry adds no held-out value. See
`docs/LATENT_FACTOR_OBSERVABILITY_DEV_RESULT_20260915.md`. No new training or
CONFIRM access has been started.

The user subsequently authorized the independent PSPS-v1 prospective test.
Its 20-D compact legal progress/stall state, Gate O then Gate D rules, 96 fresh
DEV worlds and 128 unopened CONFIRM worlds are externally frozen under
`artifacts/psps_v1_20260915`. The formal resumable DEV service passed its first
checkpoint health check and is running with automatic analysis, confirmation,
and training disabled. See `docs/PSPS_V1_PROSPECTIVE_PROTOCOL_20260915.md` and
`artifacts/psps_v1_20260915/startup_health_psps_v1.md`.

Startup audit:
`artifacts/paired_advantage_identification_20260914_v3/startup_health_pai_v3.md`.

## Repository layout

- `scripts/`: only the active collector/validator/analyzer/freezer closure and
  generic confirmation helpers.
- `tests/`: focused current runtime/model/coordinator tests.
- `artifacts/`: active v3 evidence, minimal provenance contracts, and the exact
  frozen model files required at runtime.
- `envs/`, `experiments/`, `agent/`, `common/`, `network/`, `policy/`,
  `calibration/`, `cert_runtime/`: environment and method implementations,
  intentionally untouched by repository cleanup.
- `runtime_support/review_bundle`: byte-identical runtime safety/environment
  support moved out of the old review bundle. The root `review_bundle` symlink
  preserves unchanged imports for the active frozen run.

## Health commands

```bash
.venv/bin/python scripts/validate_paired_advantage_contract.py \
  artifacts/paired_advantage_identification_20260914_v3/contract.json \
  --receipt artifacts/paired_advantage_identification_20260914_v3/freeze_receipt.json
```

Do not run the DEV analyzer until collection reaches 96/96 with a complete raw
audit. Repository minimization details are in
`docs/REPOSITORY_MINIMIZATION_20260914.md`.
