# Stage A B2/B3 TD Semantic Repair

Date: 2026-08-14

## Decision

`B2_VALUE_SEMANTICS_REPAIRED = TRUE`

`B3_VALUE_SCALE_SEMANTICS_REPAIRED = TRUE`

`B3_UPPER_QUANTILE_COVERAGE_REPAIRED = FALSE`

`LEARNED_MODEL_ADDS_VALUE = FALSE`

`READY_FOR_16X16 = FALSE`

The repair removed the pathological B2/B3 value scale and terminal errors without collecting new data, increasing the 800-update budget, changing the frozen SAC, or adding a model. B0 remains best. B3's median is now meaningful, but q0.90/q0.95/q0.99 remain severely under-covered and are not safety bounds.

## Immutable Data Reuse

Source:

`artifacts/energy_transfer/stage_a_validation_seed0_20260814_v2/`

Valid repair artifact:

`artifacts/energy_transfer/stage_a_td_repair_seed0_20260814_v2/`

Command:

```bash
PYTHONPATH=. /home/zjl/mappo/.venv/bin/python scripts/retrain_stage_a_energy_critics.py \
  --source-dir artifacts/energy_transfer/stage_a_validation_seed0_20260814_v2 \
  --output-dir artifacts/energy_transfer/stage_a_td_repair_seed0_20260814_v2 \
  --seed 0 \
  --total-updates 800 \
  --selected-mc-pretrain-updates 600 \
  --device cpu
```

The raw data SHA-256 remains `b7d54f020483c4400f5b45bf3ff46ae7752119304c26456b3a2ad12dfb0a82b8`. The copied split SHA-256 remains `001f97c67e70d186f3fb9294c8b0df85ae683c9ef85e0d5d566ae2688da301ad`. The retrainer records `data_recollected=false` and never loads or modifies the SAC checkpoint.

The original 150 sorties and split are unchanged:

| Split | Sorties | Transitions |
|---|---:|---:|
| Train | 90 | 3,412 |
| Calibration | 30 | 1,170 |
| Test | 30 | 1,130 |

An earlier repair artifact without the `_v2` suffix is marked `INVALID_FOR_COMPARISON` because its ablation variants used different initialization seeds. The valid `_v2` artifact uses seed 0 for every variant.

## Bellman and Terminal Audit

The implemented chain is:

```text
Q(s_t, a_t)
target_t = c_t + 1{not charger_hit_t} Q_target(s_{t+1}, a_{t+1})
```

- Terminal target is exactly the measured final-step cost.
- No post-terminal value is bootstrapped.
- Non-terminal `a_{t+1}` is the action actually executed by the same frozen charger-goal SAC trajectory.
- Training and inference both consume `(state, action)`; no `Q(s,a)`/post-action `V(s')` mixture exists.
- `charger_hit` is the actual charger-goal radius event and can occur only on the final recorded transition.
- The environment emits an observation for a newly sampled goal after a hit, but the terminal mask makes that next observation/action irrelevant to the target.

Thus the original failure was not a terminal-mask, label, next-action, or Q/V-definition bug.

## Root Causes

### B2 Scalar TD

The train set contains 90 terminal and 3,322 non-terminal transitions: only `2.6377%` are terminal anchors. The default positive softplus output starts near `0.69`, much larger than the final-step cost near `0.012`. Full-batch gamma=1 bootstrapping with hard target copies propagates this positive scale through many non-terminal targets while terminal correction is weak.

B2 does not exhibit the same runaway upward drift as B3: in the controlled original loop its train MC MAE decreases from `0.449` to `0.146`. It instead exhibits slow anchor propagation and horizon-dependent function-approximation bias. A ten-step deterministic trajectory is recoverable exactly by the scalar target, confirming that the Bellman formula is sound.

### B3 Quantile TD

B3 has two additional failures:

1. cumulative softplus initialization starts the four output means near `(0.710, 1.458, 2.128, 2.826)`, despite returns below one;
2. equal pairwise weighting treats nonuniform target atoms `(0.50, 0.90, 0.95, 0.99)` as four equal-probability samples, over-representing the upper tail.

Under the controlled original objective, train MC median MAE moves in the wrong direction:

| Update | Train MC median MAE | Mean q0.50 |
|---:|---:|---:|
| 0 | 0.449 | 0.710 |
| 200 | 3.538 | 3.800 |
| 400 | 3.880 | 4.142 |
| 600 | 4.159 | 4.422 |
| 800 | 4.410 | 4.672 |

The Bellman loss stays finite near `0.015` while MC error rises. This is direct evidence of B3 gamma=1 TD semantic drift.

## Minimal Repair

The selected protocol keeps the same critic networks and total 800 optimizer steps:

1. normalize costs/returns by the train-split q0.95 return (`0.56865`) and invert the scale at inference;
2. initialize normalized scalar/q0.50 output at `0.05` and quantile increments at `0.005`;
3. use 600 MC return-to-go pretraining steps;
4. synchronize the target network exactly at the MC-to-TD boundary;
5. use 200 gamma=1 TD fine-tuning steps;
6. weight horizons 1/2/3 by 20/10/5, normalized to unit mean;
7. use Polyak `0.01` during TD fine-tuning;
8. weight nonuniform target quantile atoms by probability intervals `(0.700, 0.225, 0.045, 0.030)`.

No test or calibration data determines normalization or training weights.

## Deterministic Semantic Tests

The 1-step, 2-step, and 10-step tests use known cost `0.1` per step and a smaller 400-step unit-test budget split into 300 MC plus 100 TD steps.

| Length | B2 MAE | B2 max error | B3 median MAE | B3 max error | Ordered |
|---:|---:|---:|---:|---:|---|
| 1 | 0.00000000 | 0.00000000 | 0.001450 | 0.001450 | yes |
| 2 | 0.00000007 | 0.00000013 | 0.000953 | 0.000953 | yes |
| 10 | 0.00000026 | 0.00000083 | 0.001581 | 0.002724 | yes |

These tests prove implementation consistency on deterministic finite trajectories; they do not prove deep-TD convergence generally.

## B2 Before and After

| Metric | Original | Repaired |
|---|---:|---:|
| Held-out MAE | 0.12098 | **0.06365** |
| Held-out RMSE | 0.16864 | **0.08678** |
| Terminal MAE | 0.07056 | **0.01725** |
| Mean signed error | +0.07615 | -0.03408 |
| Underestimation rate | 30.71% | 70.44% |
| Worst underestimation | 0.34420 | 0.37835 |

The repaired model has correct scale and much lower point error, but its safety error shifts toward underprediction. It is not a calibrated upper bound.

B2 horizon MAE:

| Horizon | Original | Repaired |
|---|---:|---:|
| 1 | 0.07056 | **0.01725** |
| 2 | 0.06233 | **0.00939** |
| 3 | 0.05497 | **0.01123** |
| 4-5 | 0.04890 | **0.02071** |
| 6-10 | 0.04752 | **0.03898** |
| 11-20 | 0.09949 | **0.04744** |
| 21-40 | 0.16935 | **0.08397** |
| >40 | 0.17985 | **0.13810** |

There is no longer an uncontrolled positive value-scale drift. Error still grows with horizon and the repaired B2 underestimates long returns. This is a remaining generalization limitation, not a claim of complete TD success.

## B3 Before and After

| Metric | Original | Repaired |
|---|---:|---:|
| Median held-out MAE | 4.92040 | **0.07131** |
| Median held-out RMSE | 4.93723 | **0.09894** |
| Terminal median MAE | 4.31169 | **0.02809** |
| Mean signed error | +4.92040 | -0.01069 |
| Quantile crossings | 0 | 0 |

B3 horizon median MAE:

| Horizon | Original | Repaired |
|---|---:|---:|
| 1 | 4.31169 | **0.02809** |
| 2 | 4.33956 | **0.01936** |
| 3 | 4.36783 | **0.01388** |
| 4-5 | 4.43807 | **0.01943** |
| 6-10 | 4.56746 | **0.03752** |
| 11-20 | 5.03004 | **0.04897** |
| 21-40 | 5.13713 | **0.09366** |
| >40 | 4.97200 | **0.17760** |

Coverage:

| Quantile | Original | Repaired | Nominal |
|---|---:|---:|---:|
| q0.50 | 100.00% | **44.69%** | 50% |
| q0.90 | 100.00% | 52.12% | 90% |
| q0.95 | 100.00% | 55.40% | 95% |
| q0.99 | 100.00% | 59.91% | 99% |

The quantiles remain ordered and increase with level, and q0.50 is now on the correct scale with reasonable median coverage. The upper quantiles are severely under-covered. Therefore B3 value-scale semantics pass, but distributional safety semantics and calibration do not.

## Ablation Findings

B2 held-out MAE / terminal MAE:

| Variant | Overall | Terminal |
|---|---:|---:|
| Controlled original | 0.13683 | 0.08320 |
| MC pretraining only | 0.07084 | 0.07533 |
| Terminal weighting only | 0.12189 | 0.01781 |
| Normalization only | 0.09865 | 0.06472 |
| Polyak only | 0.34002 | 0.15022 |
| Combined, hard-20 | 0.07360 | **0.01389** |
| Combined, Polyak-0.01 | **0.06365** | 0.01725 |

Terminal weighting fixes the local anchor but not the full value function. MC pretraining fixes global scale but not terminal accuracy. Polyak alone preserves the wrong initial target and is harmful. The coupled repair is necessary.

B3 median held-out MAE / terminal MAE:

| Variant | Overall | Terminal |
|---|---:|---:|
| Controlled original | 4.38141 | 3.70561 |
| Low initialization only | 0.81220 | 0.54261 |
| Atom weighting only | 0.79898 | 0.28501 |
| MC pretraining only | 0.35156 | 0.37335 |
| Terminal weighting only | 4.12091 | 0.13288 |
| Combined, hard-20 | 0.08405 | 0.03344 |
| Combined, Polyak-0.01 | **0.07131** | **0.02809** |

No single mechanism repairs B3. This supports a coupled root cause rather than post-hoc update inflation.

## Comparison with B0/B1

| Method | Held-out MAE | RMSE |
|---|---:|---:|
| B0 distance | **0.05509** | **0.07397** |
| B1 MC | 0.06140 | 0.07933 |
| B2 repaired | 0.06365 | 0.08678 |
| B3 repaired median | 0.07131 | 0.09894 |

B2's paired sortie MAE delta versus B0 is `+0.00905`, bootstrap 95% CI `[-0.00318, +0.02082]`. B3's delta is `+0.01585`, CI `[+0.00326, +0.02998]`. Neither learned TD method establishes improvement over B0.

B0, B1, B2, and B3 all have their largest errors at horizon `>40` (`0.12387`, `0.12247`, `0.13810`, and `0.17760`). The environment is strongly distance-dominated, but the learned critics also retain long-horizon underprediction and B3 upper-tail failure. B0's lead cannot be attributed solely to an unresolved catastrophic TD bug after this repair.

## Answers to the Required Questions

1. **B2 root cause:** sparse terminal anchoring plus high positive output initialization, unnormalized gamma=1 targets, and target-network propagation; not a target-mask or Q/V bug.
2. **B3 root cause:** the B2 mechanisms plus cumulative softplus scale bias and equal-mass treatment of nonuniform quantile atoms.
3. **Terminal-anchor dilution:** yes, terminal fraction is `2.6377%`; terminal weighting alone proves it affects terminal MAE but is not the whole failure.
4. **Gamma=1 bootstrap drift:** definite runaway upward drift for B3; B2 has slow anchor propagation/horizon bias but not the same runaway trace.
5. **Softplus bias:** yes; initial q0.50/q0.90/q0.95/q0.99 means are about `0.710/1.458/2.128/2.826`.
6. **Synthetic tests:** 1-step, 2-step, and 10-step B2/B3 tests pass.
7. **B2 before/after:** MAE `0.12098 -> 0.06365`; terminal `0.07056 -> 0.01725`; all horizon bins improve, with residual `>40` MAE `0.13810`.
8. **B3 before/after:** median MAE `4.92040 -> 0.07131`; terminal `4.31169 -> 0.02809`; coverage changes from all 100% to `44.69/52.12/55.40/59.91%`.
9. **Best method:** B0 remains best.
10. **Why B0 remains best:** both distance dominance and remaining learned-model long-horizon/tail limitations; not catastrophic value-scale failure anymore.
11. **Learned value:** `FALSE`.
12. **16x16:** still `FALSE` because B0 is better, B3 upper-tail coverage is invalid, and no formal multi-seed Stage A evidence exists.
13. **Files changed:** listed below.
14. **Tests:** `38 passed`; only dependency deprecation warnings were emitted.
15. **Artifact:** `artifacts/energy_transfer/stage_a_td_repair_seed0_20260814_v2/`.

## Files Changed

- `safety/energy/critics.py`;
- `safety/energy/td.py`;
- `safety/energy/inference.py`;
- `safety/energy/__init__.py`;
- `experiments/energy_transfer/analysis.py`;
- `experiments/energy_transfer/retraining.py`;
- `scripts/retrain_stage_a_energy_critics.py`;
- `tests/new_route/test_energy_td.py`;
- `tests/new_route/test_energy_inference.py`;
- `tests/new_route/test_energy_bootstrap_stage_a.py`;
- `tests/new_route/test_stage_a_analysis.py`;
- Stage A experiment-analysis documentation.

## Research Interpretation

The original B2/B3 failures were real implementation/optimization failures and are now isolated. Fixing them does not make TD the best estimator. In this deterministic open 4x4 task, Euclidean distance remains a stronger and simpler predictor. B3 additionally lacks usable upper-tail coverage. No conformal calibration, formal run, autonomous switching experiment, SAC retraining, collision expansion, or 16x16 run was performed.

## Verification

- `PYTHONPATH=. /home/zjl/mappo/.venv/bin/python -m pytest -q tests/new_route`: `38 passed`;
- Stage A repair modules compile successfully;
- source raw JSONL and split hashes match the immutable validation artifact;
- copied split is byte-identical and sortie-disjoint;
- repaired scalar and quantile checkpoints reload with inverse normalization intact;
- quantile ordering is preserved after reload;
- frozen SAC code/checkpoint, Collision Module, and 16x16 experiment paths are untouched;
- `READY_FOR_16X16` remains false.
