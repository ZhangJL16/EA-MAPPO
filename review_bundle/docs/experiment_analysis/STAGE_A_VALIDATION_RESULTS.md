# Stage A 4x4 Energy Estimation Validation

Date: 2026-08-14

Follow-up: the same-data TD repair removes the B2/B3 scale collapse but does not beat B0 and does not repair B3 upper-tail coverage. See `STAGE_A_TD_REPAIR_RESULTS.md`. The numbers below remain the immutable pre-repair validation result.

## Decision

`VALIDATION_RUN_COMPLETED = TRUE`

`FORMAL_STAGE_A_RUN_COMPLETED = FALSE`

`LEARNED_MODEL_ADDS_VALUE = FALSE`

`READY_FOR_16X16 = FALSE`

The 150-sortie validation establishes that the formal collection, split, training, held-out evaluation, and visualization protocol is executable. It does not establish that a learned Energy Critic adds value over the Euclidean-distance baseline. No formal multi-seed run and no 16x16 experiment should start from this result.

## Provenance

Artifact:

`artifacts/energy_transfer/stage_a_validation_seed0_20260814_v2/`

Command:

```bash
PYTHONPATH=. /home/zjl/mappo/.venv/bin/python scripts/run_stage_a_energy_bootstrap.py \
  --seed 0 \
  --checkpoint artifacts/phase1_sb3_sac_1m_gpu/seed0/checkpoint_step_1000000.zip \
  --output-dir artifacts/energy_transfer/stage_a_validation_seed0_20260814_v2 \
  --sorties 150 \
  --task-prefix-min-steps 0 \
  --task-prefix-max-steps 160 \
  --max-return-steps 800 \
  --operational-energy-capacity 5.0 \
  --training-updates 800 \
  --device cpu \
  --run-kind validation
```

The artifact records Git HEAD `48e0ae634eaab29c26f6489402d441c450880165` because the Stage A implementation was still uncommitted when the run started. The recorded code hash is `4d94e4d0e5ae631cfc382497250a24c0b480989e8386324a9db749eb270357ca`; recomputation after commit `a60e99b6dc1b8c336d1a312272f4fc4dcce0c9e2` matches exactly. The frozen SAC checkpoint SHA-256 is `56f50be556888a2b567ea76b3c9726319187802e1327a78db48082de195c6fce`.

The validation uses one top-level experiment seed and 150 deterministic, distinct sortie seeds (`20000` through `20149`). This is adequate to check the protocol but not to establish multi-seed reproducibility.

## Collection and Split

| Item | Value |
|---|---:|
| Attempted sorties | 150 |
| Completed charger returns | 150 |
| Censored returns | 0 |
| Completion rate | 100% |
| Valid return-to-go labels | 5,712 transitions |
| Censored transitions carrying labels | 0 |
| Prefix range | 0-160 steps |
| Unique prefix lengths | 150 |

The split unit is the complete sortie, with fixed split seed `8003`:

| Split | Sorties | Completed | Censored | Supervision transitions |
|---|---:|---:|---:|---:|
| Train | 90 | 90 | 0 | 3,412 |
| Calibration | 30 | 30 | 0 | 1,170 |
| Test | 30 | 30 | 0 | 1,130 |

All 150 sortie IDs are unique and assigned exactly once. Transition leakage across splits is impossible by construction. Exact sortie IDs and seeds are preserved in `split.json`.

## Held-Out Test Metrics

All point metrics below use the same 1,130 transitions from 30 held-out sorties. Relative error is mean absolute relative error.

| Method | MAE | RMSE | Mean signed error | Underestimate rate | Mean underestimate | Worst underestimate | Relative error |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 distance x train Wh/m | **0.05509** | **0.07397** | +0.02702 | **28.67%** | **0.04896** | **0.25613** | **33.17%** |
| B1 Monte-Carlo regression | 0.06140 | 0.07933 | -0.01309 | 51.95% | 0.07170 | 0.33843 | 44.04% |
| B2 scalar TD, gamma=1 | 0.12098 | 0.16864 | +0.07615 | 30.71% | 0.07299 | 0.34420 | 72.16% |
| B3 q0.50 prediction | 4.92040 | 4.93723 | +4.92040 | 0.00% | 0.00000 | 0.00000 | 4,158.75% |

`B0` is the best method on MAE, RMSE, relative error, underestimation rate, mean underestimation magnitude, and worst underestimation. `B1` is the best learned method, but it is 11.45% worse than B0 in transition-level MAE.

Paired held-out sortie comparison against B0:

| Learned method | Mean sortie-MAE delta | Bootstrap 95% CI | Sorties better than B0 |
|---|---:|---:|---:|
| B1 | +0.00652 | [-0.00365, +0.01582] | 33.3% |
| B2 | +0.07048 | [+0.04578, +0.09600] | 13.3% |
| B3 q0.50 | +4.85838 | [+4.79430, +4.92396] | 0.0% |

The B1 confidence interval includes zero and its mean difference favors B0. B2 and B3 are decisively worse on these held-out sorties.

## B3 Quantile Audit

| Quantile | Empirical coverage | Nominal under-coverage | Upper-tail MAE |
|---|---:|---:|---:|
| q0.50 | 100% | 0 | 5.11349 |
| q0.90 | 100% | 0 | 5.05551 |
| q0.95 | 100% | 0 | 5.10117 |
| q0.99 | 100% | 0 | 5.07259 |

No quantile crossing occurs. Nevertheless, 100% coverage at every level is not successful calibration: all quantiles are grossly over-conservative. B3 q0.50 has held-out MAE `4.92040`, while actual return energies are below one energy unit in this setting. On 30 held-out terminal samples, B3 q0.50 MAE is `4.31169`, compared with `0.03873`, `0.06025`, and `0.07056` for B0, B1, and B2. This is semantic collapse or a terminal-anchor failure, not a calibrated safety bound.

All three learned objectives have finite, decreasing optimization losses. Finite loss therefore does not establish valid return-energy semantics.

## Distance-Only Analysis

Observed correlations on held-out transitions:

- actual return energy versus Euclidean charger distance: `r = 0.92902`;
- actual return energy versus realized remaining path length: `r = 0.93997`.

The simple distance signal explains most variation in this open 4x4 environment. B0 signed residual correlations are:

| Variable | Pearson correlation with B0 signed error |
|---|---:|
| Current velocity magnitude | +0.41098 |
| Velocity direction relative to charger | +0.28043 |
| Action magnitude | +0.14904 |
| Absolute altitude difference | -0.14590 |
| Task prefix length | +0.08526 |

The residual structure shows that energy is not literally a function of Euclidean distance alone. In particular, B0 mildly underpredicts the lowest-speed bins and overpredicts the highest-speed bins. However, none of B1-B3 converts that additional state information into lower held-out error. The existence of residual structure is therefore not evidence that the current learned critic is useful.

## Coverage Audit

`DATA_COVERAGE_SUFFICIENT = TRUE` under the preregistered validation checks:

- return-start XY occupancy: 48 of 64 cells (`75%`) on an 8x8 grid;
- charger-distance range: `0.42617` to `4.33213`, span `3.90596`;
- prefix span: `160` steps;
- return-start speed mean/std/p90: `0.23413 / 0.10982 / 0.38176` m/s;
- observed position range: `[0.377, 0.223, 0.197]` to `[3.636, 3.772, 1.759]`.

Coverage is broad enough for this validation. The negative learned-model result is not explained by the original 10-sortie, 0-12-step smoke limitation.

## Scientific Answer

The current 4x4 frozen-SAC trajectories are sufficient to fit a strong empirical distance baseline, but this validation does **not** show a meaningful, generalizing Energy Critic with extra value beyond that baseline.

- B0 is not globally underestimating: its mean signed error is positive and its underestimation rate is 28.67%. It still has a nontrivial worst underestimation of `0.25613`.
- B1 mildly underestimates on average and underestimates 51.95% of test transitions; it is less attractive for safety than B0.
- B2 has finite training loss but poor held-out and terminal prediction semantics.
- B3 is unusably over-conservative and its nominal coverage is vacuous.

Thus `LEARNED_MODEL_ADDS_VALUE = FALSE`, not `unclear`.

## Stage A Gate

Passed:

1. frozen SAC charger-return completion rate;
2. broad validation data coverage;
3. whole-sortie split with no leakage;
4. finite optimization losses;
5. complete result, raw-data, split, config, model, and figure artifacts.

Failed:

1. B2 held-out and terminal semantic stability;
2. B3 semantic stability and meaningful quantile calibration;
3. stable learned-model improvement over B0;
4. formal multi-seed Stage A evidence.

Therefore `READY_FOR_16X16 = FALSE`.

## Next Minimum Step

Do not launch the formal run yet. First diagnose the existing B2/B3 Bellman-target and terminal-anchor behavior without adding a new network or changing the frozen SAC. The next run should remain a bounded 4x4 validation rerun using the same data/split protocol after that repair. A formal scale should be chosen only if B2/B3 predictions become semantically valid and at least one learned method demonstrates stable held-out value over B0. No formal scale is approved from the current validation.

## Artifact Inventory

The validation directory contains:

- `config.json`, `results.json`, `split.json`, `raw_sorties.jsonl`;
- B1, B2, and B3 model checkpoints and SHA-256 hashes;
- `COLLECTION_COMPLETED.json` and `COMPLETED.json`;
- all six required analysis figures.

The artifact is a validation result, not a paper result and not formal multi-seed evidence.

## Verification

Current verification results:

- `PYTHONPATH=. /home/zjl/mappo/.venv/bin/python -m pytest -q tests/new_route`: `34 passed`;
- Python compilation for the Stage A runner and analysis modules: passed;
- artifact file inventory and raw JSONL SHA-256: passed;
- recorded/current Stage A code-hash equality: passed;
- B1/B2 scalar and B3 quantile checkpoint reload: passed;
- train/calibration/test sortie-disjointness: passed;
- raw completed-transition label count: exactly `5,712`.

The test suite emits only dependency deprecation warnings from Matplotlib/PyParsing; no test fails.

## Files Changed for Stage A

Core implementation and analysis:

- `experiments/energy_transfer/stage_a_bootstrap.py`;
- `experiments/energy_transfer/analysis.py`;
- `experiments/energy_transfer/control.py`;
- `scripts/run_stage_a_energy_bootstrap.py`;
- `envs/navigation/operational_energy.py`;
- `envs/navigation/environment.py`;
- `safety/energy/inference.py`;
- `safety/switching/energy.py`.

Focused validation tests:

- `tests/new_route/test_energy_bootstrap_stage_a.py`;
- `tests/new_route/test_stage_a_analysis.py`;
- `tests/new_route/test_energy_inference.py`;
- `tests/new_route/test_operational_energy.py`;
- related navigation and switching regression tests under `tests/new_route/`.

Documentation:

- `docs/experiment_analysis/ENERGY_ROUTE_REFACTOR_AUDIT.md`;
- `docs/experiment_analysis/STAGE_A_IMPLEMENTATION_STATUS.md`;
- this validation report;
- `README.md`.
