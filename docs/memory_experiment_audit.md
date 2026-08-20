# Memory Experiment Audit Report

**Date:** 2026-08-20  
**Auditor:** fresh same-family reviewer via GPT-5.5 fallback, provisional  
**Final artifact:** `artifacts/memory_closed_loop_controlled_final_audited_20260820/`

## Final Verdict: WARN

`FATAL = 0`

`CRITICAL = 0`

`MAJOR = 0`

`MINOR = 0`

The warning is a scope/claim ceiling, not an unresolved artifact mismatch.

## Initial audit and remediation

The first audit failed with two critical issues:

1. closed-loop development radii reused the held-out test predictions that
   also supported reported estimation metrics;
2. the artifact omitted hashes for the calibration/test NPZs and several
   imported runtime source modules.

Both are closed in the final artifact:

- radii use `validation.npz` and `validation_predictions.npz` only;
- `heldout_test_used_for_radii = false`;
- validation is honestly identified as reused for checkpoint selection, so the
  maximum-residual box has no conformal or coverage claim;
- calibration/test data and predictions, estimator checkpoints, SAC
  checkpoint, summary, and 13 executed source modules are SHA-256 bound;
- the estimator training-source hash matches the loaded checkpoint artifact;
- all current executed-source hashes match the final artifact manifest.

The final reviewer found one minor stale latency value (`23.4894 ms` versus
`23.7764 ms`), which was corrected and rechecked. The reviewer then reported
zero unresolved findings at every severity.

## Integrity checks

| Check | Status | Evidence |
|---|---|---|
| Ground-truth provenance | PASS | Simulator latent velocity/acceleration and trajectories, never model-generated labels. |
| Score normalization | PASS | Raw MAE/RMSE and physical widths; no self-normalized score. |
| Result existence/numeric match | PASS | Required artifacts exist; ledger numbers match after correction. |
| Held-out test leakage into closed loop | PASS | Test does not configure radii; development validation does. |
| Source/input provenance | PASS | 13 source hashes and 15 input hashes recorded and verified. |
| Invalid-artifact exclusion | PASS | Three invalid final-like runs carry `INVALID.json` and are excluded. |
| Evaluation type | SIMULATION_ONLY | 5k synthetic trajectories, 12 controlled scenarios, 60,152 transitions. |
| Claim scope | WARN | No real-UAV, repeated-time, general robustness, or learned deterministic-safety claim is allowed. |

## Claim ceiling

The evidence supports only hash-bound controlled-simulation descriptions and
conditional ideal-arithmetic propositions. It does not support deterministic
safety for learned/Kalman/IMM boxes, a coverage guarantee for development
envelopes, equal-certification intervention/energy superiority, or physical
UAV safety.
