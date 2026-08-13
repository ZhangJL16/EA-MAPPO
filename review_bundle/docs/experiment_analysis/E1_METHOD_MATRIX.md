# E1 Method Matrix

Date: 2026-08-13

## Status Definitions

- **IMPLEMENTED:** executable estimator code exists.
- **RAN:** estimator fitting/evaluation reached execution in an artifact.
- **COMPLETED:** result and model checkpoint were written.
- **VALID_FOR_COMPARISON:** completed under an auditable common protocol.

## Matrix

| ID | Method | Implemented | Ran | Completed | Valid for comparison | Notes |
|---|---|---|---|---|---|---|
| B0 | Distance × empirical energy/m | YES | NO | NO | NO | Formula exists, but cost/m is computed only after all collection sorties. |
| B1 | Monte-Carlo return regression | YES | NO | NO | NO | Supervised model exists; no model checkpoint or result exists. |
| B2 | Scalar TD energy critic, gamma=1 | YES | NO | NO | NO | SSP target code exists; fitting was not reached. |
| B3 | Distributional/quantile TD | YES | NO | NO | NO | Quantiles 0.50/0.90/0.95/0.99 exist; fitting was not reached. |
| B4 | Distributional TD + split conformal | YES | NO | NO | NO | Only the 0.95 quantile is configured for calibration; calibration was not reached. |

No energy method can be ranked. The 43 seed-0 trajectories are raw collection only and were never split into a completed train/calibration/test artifact.

## Switching Matrix

| Rule | Implemented | Ran | Valid result |
|---|---|---|---|
| Fixed SOC threshold | NO | NO | NO |
| Distance threshold | NO | NO | NO |
| Learned energy threshold | NO | NO | NO |
| Calibrated learned threshold | NO | NO | NO |
| One-way TASK→CHARGER commitment telemetry | NO | NO | NO |

The current collector always commits to the charger after exactly one task action. That hard-coded collection behavior is not a switching policy or switching experiment.
