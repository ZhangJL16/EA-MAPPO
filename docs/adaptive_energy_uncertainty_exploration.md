# Adaptive Energy Uncertainty Exploration

## Status

This study keeps the frozen 500k SAC navigation policy and the existing MC
Energy-to-Go point estimator. It does **not** restore TD, retrain SAC, alter the
battery capacity/reserve, or launch a formal 500k Phase 2 run.

The completed exploration artifact is:

```text
artifacts/uav_energy_adaptive_uncertainty_20260817_214303/
```

The development comparison used the already exposed v2 diagnostic set. Method
selection was then frozen before collecting a fresh test v3 consisting of 2,000
goal trajectories and 1,000 mission trajectories. The fresh test is now consumed;
any post-hoc change to the method or nominal risk level requires a new final test.

## Executive Decision

```text
ROOT CAUSE OF LONG-DISTANCE UNDER-COVERAGE:
near-zero mean error but distance- and boundary-dependent upper-tail variance;
the 7D state omits absolute/boundary context that changes future navigation.

HETEROSCEDASTIC_TAIL_RISK_CONFIRMED = TRUE
EMPIRICAL_7D_INFORMATION_LOSS = TRUE
STRUCTURAL_7D_MARKOV_SUFFICIENCY = FALSE

CURRENT POINT ESTIMATOR = KEEP

BEST GOAL-LEVEL ADAPTIVE MODEL = HETEROSCEDASTIC LAPLACE + TRAJECTORY CONFORMAL
GOAL SUBGROUP SAFETY GATE AT NOMINAL 95% = NOT PASSED

BEST MISSION METHOD = POINT MISSION MEAN + HETEROSCEDASTIC LAPLACE RESIDUAL SCALE
                      + MISSION-TRAJECTORY CONFORMAL

MC-SUPERVISED DIRECT CQR = NOT RECOMMENDED IN ITS CURRENT FORM

FORMAL 500K PHASE2 = NOT LAUNCHED
```

The operational switching test uses `return-now` and direct mission bounds. On
fresh v3, heteroscedastic Laplace achieved at least 96.1% empirical coverage in
every CHARGER distance group and 96.0% worst-bucket mission coverage. Its poor
TASK-only subgroup coverage must still be reported, but the direct mission head—not
the TASK component bound—is used for the mission stopping decision.

## Data and Protocol

| Role | Data | Use |
|---|---:|---|
| Point model training | 4,000 trajectories | Existing MC point model only |
| Point validation | 500 trajectories | Existing model selection |
| Original calibration | 500 trajectories | Calibration development |
| Focused v2 calibration | 2,000 trajectories | Tail/group calibration development |
| Exposed v2 goal diagnostic | 2,000 trajectories | Candidate selection only |
| Mission development | 1,000 missions | 600/200/200 train/validation/calibration split |
| Exposed v2 mission diagnostic | 1,000 missions | Candidate selection only |
| Fresh goal test v3 | 2,000 trajectories | One-shot final evaluation |
| Fresh mission test v3 | 1,000 missions | One-shot final evaluation |

The point model and SAC hashes are recorded in the run `config.json`. All
calibration scores use one maximum-underestimation score per trajectory or mission,
not individual states as independent conformal samples.

## Residual Root Cause

Define the state residual as

\[
r_t = E_{\mathrm{true},t} - \widehat E_t,
\]

and the trajectory score as

\[
R_i = \max_{t\in i} r_t.
\]

Fresh v3 TASK residuals show that long-distance failure is primarily a tail-scale
problem rather than a mean-bias problem:

| TASK distance | Mean residual | Std residual | P95 | P99 | State MAE |
|---|---:|---:|---:|---:|---:|
| 100–500 m | 0.0164 | 0.2117 | 0.2841 | 0.7081 | 0.1269 |
| 500–1500 m | -0.0078 | 0.1533 | 0.1915 | 0.4017 | 0.0944 |
| 1500–2500 m | 0.0081 | 0.1380 | 0.1688 | 0.5177 | 0.0812 |
| 2500–4000 m | -0.0039 | 0.2252 | 0.2595 | 0.8489 | 0.1228 |
| >4000 m | -0.0000 | 0.2719 | 0.2811 | 0.8474 | 0.1349 |

Compared with the two middle-distance buckets, the two long-distance buckets have:

```text
long/mid residual-std ratio = 1.706
long/mid P99 residual ratio = 1.845
max absolute long-distance mean residual < 0.004
```

Therefore:

```text
HETEROSCEDASTIC_TAIL_RISK_CONFIRMED = TRUE
MEAN_BIAS_IS_PRIMARY_CAUSE = FALSE
```

The v2 trajectory diagnostics identify a concrete mechanism beyond distance alone:

| TASK distance | Boundary-contact trajectories | Mean minimum boundary distance | Mean max underestimation |
|---|---:|---:|---:|
| 100–500 m | 0.0% | 87.6 m | 0.279 |
| 500–1500 m | 4.5% | 68.9 m | 0.287 |
| 1500–2500 m | 14.3% | 51.7 m | 0.288 |
| 2500–4000 m | 47.4% | 26.0 m | 0.441 |
| >4000 m | 83.7% | 6.7 m | 0.449 |

Long routes are much more likely to encounter the boundary projection. The
projection itself is not charged as propulsion acceleration, but it changes velocity
and forces subsequent re-acceleration and path extension. That future event is not
represented in the current 7D energy state.

## 7D State Aliasing

The current energy input is velocity (3), goal direction (3), and normalized
distance (1). In a 100,000-state nearest-neighbor audit on fresh v3:

```text
closest-1% 7D feature-distance threshold = 0.00205
median true-return difference             = 0.0195
P95 true-return difference                = 0.0828
maximum true-return difference            = 0.3642
fresh point MAE                           = 0.0901
```

The closest-pair return difference is not merely numerical noise: its P95 is about
the point model MAE, and its maximum is about four times the MAE. Within these close
pairs, differences in remaining realized path length have Spearman correlation
0.918 with return differences. Absolute `x/y/z` and distance-to-boundary have weaker
but nonzero close-pair rank correlations (approximately 0.27–0.42).

There is also a structural, not merely empirical, sufficiency failure: the environment
transition applies boundary projection using absolute position, while absolute
position is absent from the 7D input. Two states can therefore share the same 7D
representation and action but have different next velocities and future costs.

Candidate context classification:

| Context | Decision | Evidence |
|---|---|---|
| Predicted remaining safe-path length | **Useful** | Strongest nearest-neighbor explanation; directly captures detour |
| Distance to boundary / anticipated boundary intervention | **Useful** | Structural transition dependence and long-distance contact rates |
| Compact safe-trajectory detour/turning context | **Useful** | Available before execution and aligned with future obstacle extension |
| Goal type / operational segment | **Useful with caution** | Strong empirical TASK/CHARGER tail difference; may encode data-distribution rather than physics |
| Absolute position | **Useful but undesirable raw input** | Repairs boundary Markov state, but harms translation invariance; prefer derived boundary context |
| Predicted acceleration/turning intensity | **Weakly useful** | Moderate trajectory correlation; unavailable without a rollout/planner |
| Path ratio | **Weakly useful** | Trajectory max-underestimation Spearman about 0.40, but it is future information unless predicted |
| Vertical displacement | **Redundant** | Already reconstructible from goal direction and distance |
| Initial velocity and goal direction | **Redundant** | Already present in 7D |
| Raw LiDAR in this obstacle-free study | **Not justified now** | No obstacles; would add complexity without current evidence |

## Adaptive Scale Diagnostics

The learned Laplace scale is input-dependent and correlates with absolute error on
fresh v3:

```text
Spearman(scale, |residual|)                    = 0.565
Pearson(scale, |residual|)                     = 0.557
Spearman(scale, positive underestimation)      = 0.158
Spearman(max trajectory scale, max residual)   = 0.268
scale P05 / P50 / P95                          = 0.0229 / 0.0573 / 0.2079
```

Thus the scale head learns general difficulty, but it is only weakly aligned with
the one-sided trajectory maximum that drives safety. This explains why it reduces
average width while failing to repair the worst TASK subgroup.

## Fresh Test v3: 95% Method Matrix

### Goal trajectories

| Method | Point MAE | Trajectory coverage | TASK | CHARGER | Worst goal×distance | Mean width | P95 width | Conservatism |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Point + global conformal | 0.0901 | 95.75% | 91.68% | 98.54% | 84.42% | 0.751 | 0.751 | 0.748 |
| Point + group conformal | 0.0901 | **97.15%** | **95.32%** | 98.54% | **92.16%** | 0.839 | 0.934 | 0.836 |
| Point + Gaussian scale + conformal | 0.0901 | 94.80% | 91.03% | 96.92% | 83.77% | 0.485 | 0.786 | 0.482 |
| Point + Laplace scale + conformal | 0.0901 | 95.10% | 92.07% | 97.08% | 85.71% | **0.478** | 0.777 | **0.475** |
| MC quantiles + one-sided CQR | 0.1517 | 95.00% | 91.42% | 97.73% | 78.57% | 0.710 | 1.196 | 0.800 |

The Laplace method reduces mean conservatism by 36.5% relative to global conformal,
while retaining marginal 95% coverage. It does **not** satisfy empirical subgroup
coverage. Point+group remains the strongest goal-level subgroup baseline but is the
most conservative at nominal 95%.

### Missions

| Method | Point MAE | Mission coverage | Worst distance group | Mean width | P95 width | Conservatism | Unnecessary return proxy | Estimated accepted states |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Point + global conformal | 0.2339 | 95.70% | 93.50% | **1.036** | 1.036 | **1.104** | **0.359%** | 80.04% |
| Point + distance-group conformal | 0.2339 | **96.90%** | 95.50% | 1.171 | 1.313 | 1.239 | 0.403% | 80.00% |
| Point + Laplace mission scale + conformal | 0.2339 | 96.70% | **96.00%** | 1.067 | 1.333 | 1.135 | 0.376% | 80.02% |
| Direct mission quantiles + CQR | 0.6048 | 95.80% | 91.00% | 3.024 | 4.209 | 2.712 | 0.891% | 79.61% |

Mission heteroscedastic calibration is the best balance: versus mission group
conformal it improves worst-group coverage by 0.5 points while reducing mean width
8.9%, conservatism 8.4%, and the false-return proxy. Global conformal is slightly
tighter on average but misses the 95% empirical worst-group target.

The proxy is intentionally low because battery capacity (378.7 units) is much larger
than the 1–3 unit differences among bounds; it is a diagnostic, not a claim about
formal task throughput.

## Safety–Efficiency Trade-off

### Goal: heteroscedastic Laplace

| Nominal target | Coverage | Worst subgroup | Mean width |
|---:|---:|---:|---:|
| 90% | 90.50% | 76.62% | 0.382 |
| 95% | 95.10% | 85.71% | 0.478 |
| 97.5% | 96.90% | 91.56% | 0.573 |
| 99% | 98.70% | 96.75% | 0.792 |

### Mission: heteroscedastic Laplace

| Nominal target | Coverage | Worst subgroup | Mean width | Unnecessary return proxy | Accepted states |
|---:|---:|---:|---:|---:|---:|
| 90% | 91.90% | 91.00% | 0.786 | 0.284% | 80.10% |
| 95% | 96.70% | 96.00% | 1.067 | 0.376% | 80.02% |
| 97.5% | 98.10% | 97.00% | 1.224 | 0.427% | 79.98% |
| 99% | 98.50% | 97.50% | 1.398 | 0.479% | 79.94% |

The nominal risk level is the user-facing parameter. No manually enlarged fixed
margin was introduced.

## Why Direct CQR Did Not Win

The candidate used true MC returns and no bootstrapping, so this is not the old TD
failure. Nevertheless:

```text
goal raw quantile crossing rate    = 76.7%
mission raw quantile crossing rate = 85.5%
```

Post-hoc sorting preserves ordering at inference but does not repair poor quantile
identification. Goal Q50 MAE increased from 0.090 to 0.152. Direct mission Q50 MAE
was 0.605 and its 95% CQR mean width was 3.024. The mission model had only 600
training missions, making seven independent quantile heads data-inefficient. CQR is
therefore rejected for the next formal run, not rejected as a general method.

## Mission Error Dependence

Task and return-after-task point residuals were nearly independent on the 1,000
mission diagnostic trajectories:

```text
overall Pearson  = -0.0009
overall Spearman =  0.0057
joint upper-decile frequency = 1.0% (independence reference 1.0%)
```

No bucket showed meaningful positive dependence; the largest joint upper-decile
frequency was 2.0% for >4000 m. Therefore the previous mission undercoverage is not
explained by strong positive component-error correlation. It is better explained by
trajectory-wise maximum exposure, subgroup heterogeneity, and limited mission-level
calibration.

The recommended mission design is consequently simple: retain the accurate sum of
the two point predictions and learn/calibrate one mission-level residual scale. The
direct 14D mission quantile model is not justified by current evidence.

## Oracle Forensics

For the top 50 goal underestimations, deterministic model-based rollout reproduced
actual initial energy with mean difference below `2e-7` units. Of those cases, 36/50
were TASK trajectories and 15/50 were TASK×2500–4000 m; 22/50 contacted the
boundary. The point model underestimated their initial energy by 0.729 units on
average.

For the top 50 mission underestimations, the point model was low by 1.049 units on
average, while oracle-minus-actual mission energy was only 0.0087 ± 0.040 units.
This confirms that the forensic cases are not simulator stochasticity or an MC-label
error; they are approximation/context errors in the learned point model.

## Short Phase 2 Comparison

One fixed-seed, 20,000-transition run compared the previous group-conformal
estimator with the selected heteroscedastic goal+mission estimator:

| Metric | Group conformal | Adaptive heteroscedastic |
|---|---:|---:|
| Tasks completed | 34 | 34 |
| Tasks / 1,000 transitions | 1.7 | 1.7 |
| Completed recharge cycles | 2 | 2 |
| Successful returns | 2/2 | 2/2 |
| Energy exhaustion | 0 | 0 |
| Mean arrival energy fraction | 11.37% | 11.37% |

The trajectories were identical in this short stream because neither bound changed
the first two switching decisions. Therefore:

```text
SHORT_RUN_SAFETY_REGRESSION = NOT OBSERVED
SHORT_RUN_DELIVERY_EFFICIENCY_GAIN = NOT ESTABLISHED
```

This one-seed short run is a state-machine/behavioral check, not a paper result.

## Conformal Guarantee Boundary

Standard split conformal gives finite-sample **marginal** coverage under exchangeable
calibration/test units. Here the unit is a complete trajectory or mission score.
Dependent states within a trajectory are not treated as independent calibration
examples.

The following claims are not made:

* overall 95% implies every goal/distance subgroup has 95% coverage;
* empirical subgroup coverage is a distribution-free conditional guarantee;
* coverage persists under policy, obstacle, wind, payload, or battery shift;
* fresh-v3 tuning remains a valid final test after inspecting these results.

Group/Mondrian calibration can provide category-conditional validity when categories
are fixed in advance and exchangeability holds within category, but it does not imply
arbitrary pointwise conditional coverage. Exact distribution-free conditional
coverage is generally impossible without assumptions or vacuous intervals.

## Primary Literature Map

| Primary work | Direct lesson for this study |
|---|---|
| [Nix & Weigend, Estimating the Mean and Variance, 1994](https://doi.org/10.1109/ICNN.1994.374138) | Input-dependent scale via likelihood is a classical heteroscedastic baseline. |
| [Kendall & Gal, Aleatoric and Epistemic Uncertainty, NeurIPS 2017](https://proceedings.neurips.cc/paper_files/paper/2017/hash/2650d6089a6d640c5e85b2b88265dc2b-Abstract.html) | Learned aleatoric scale does not by itself capture epistemic/context omission. |
| [Lakshminarayanan et al., Deep Ensembles, NeurIPS 2017](https://proceedings.neurips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html) | Ensembles are a future epistemic baseline, but were outside this study's minimal architecture scope. |
| [Koenker & Bassett, Regression Quantiles, 1978](https://people.eecs.berkeley.edu/~jordan/sail/readings/koenker-bassett.pdf) | Pinball loss directly estimates conditional quantiles. |
| [Narayan et al., Regularization Strategies for Quantile Regression, 2021](https://arxiv.org/abs/2102.05135) | Non-crossing and quantile regularization matter; current raw crossing rates are unacceptable. |
| [Lei et al., Distribution-Free Predictive Inference for Regression, 2018](https://arxiv.org/abs/1604.04173) | Split conformal gives marginal coverage and discusses locally varying bands. |
| [Romano et al., Conformalized Quantile Regression, NeurIPS 2019](https://proceedings.neurips.cc/paper_files/paper/2019/hash/5103c3584b063c431bd1268e9b5e76fb-Abstract.html) | CQR combines adaptive quantiles with finite-sample marginal calibration. |
| [Papadopoulos et al., Normalized Nonconformity Measures, 2008](https://actapress.com/Abstract.aspx?paperId=32280) | Residual normalization can tighten intervals when difficulty is predictable. |
| [Barber et al., Limits of Distribution-Free Conditional Predictive Inference, 2021](https://arxiv.org/abs/1903.04684) | Arbitrary exact conditional coverage cannot be claimed distribution-free. |
| [Tibshirani et al., Conformal Prediction Under Covariate Shift, NeurIPS 2019](https://proceedings.neurips.cc/paper_files/paper/2019/hash/8fb21ee7a2207526da55a679f0332de2-Abstract.html) | Weighted conformal needs a covariate-shift model/likelihood ratio; ordinary calibration is not shift-robust. |
| [Guan, Localized Conformal Prediction, 2021](https://arxiv.org/abs/2106.08460) | Localization targets local efficiency but preserves only stated marginal/local guarantees. |
| [Izbicki et al., Flexible Conditional Predictive Bands, AISTATS 2020](https://proceedings.mlr.press/v108/izbicki20a.html) | Conditional-density methods can improve local coverage under additional consistency conditions. |
| [Boström et al., Mondrian Conformal Predictive Distributions, 2021](https://proceedings.mlr.press/v152/bostrom21a.html) | Predefined categories can improve group-specific tightness and calibration. |
| [Gibbs & Candès, Adaptive Conformal Inference Under Distribution Shift, 2021](https://arxiv.org/abs/2106.00170) | Online coverage-frequency adaptation addresses temporal shift, not static conditional coverage. |
| [Gibbs & Candès, Online Prediction with Arbitrary Distribution Shifts, JMLR 2024](https://www.jmlr.org/papers/v25/22-1218.html) | Online step-size adaptation is relevant after the system begins persistent deployment. |
| [Bates et al., Distribution-Free Risk-Controlling Prediction Sets, 2021](https://arxiv.org/abs/2101.02703) | Calibrate a declared operational loss/risk rather than treating interval width as the only objective. |
| [Stolaroff et al., Energy Use of Delivery Drones, Nature Communications 2018](https://www.nature.com/articles/s41467-017-02411-5) | Measured speed/acceleration affect flight energy and delivery efficiency. |
| [Abeywickrama et al., Comprehensive UAV Energy Model, IEEE Access 2018](https://opus.lib.uts.edu.au/handle/10453/131279) | Empirical flight regimes and activities materially change energy consumption. |
| [Liu et al., Multi-Rotor Power Consumption Model, ICUAS 2017](https://its.berkeley.edu/node/6301) | Physics-based propulsion models provide interpretable trajectory context. |
| [Gao et al., UAV Energy Model Validation and Generalization, 2021](https://arxiv.org/abs/2005.01305) | Speed, direction, acceleration, and curved flight require richer context than distance alone. |
| [Bauersfeld & Scaramuzza, Range and Endurance Estimates, RA-L 2022](https://arxiv.org/abs/2109.04741) | Battery, motor, aerodynamic, and speed context matter under real multicopter deployment. |
| [Góra et al., ML Energy Consumption Model for UAV, 2022](https://www.mdpi.com/1996-1073/15/18/6810) | Payload, altitude, orientation, velocity, and weather motivate later domain-shift tests. |

The literature supports adaptive widths, but it does not imply that a generic scale
head will repair omitted-state aliasing. That distinction matches the experiment:
Laplace scale predicts generic error magnitude, while boundary-dependent one-sided
trajectory tails remain undercovered.

## Future Obstacle and CBF Extension

With obstacles, the 7D estimator cannot be Markov sufficient because energy depends
on future detours and safety interventions absent from the input. Raw LiDAR-to-energy
is possible but entangles perception, policy, and energy prediction and is difficult
to calibrate across sensor changes.

The preferred future architecture is:

```text
nominal goal-conditioned SAC
        ↓
planner / CBF / collision safety layer
        ↓
compact predicted safe-trajectory context
        ↓
Energy-to-Go point + residual uncertainty
```

The compact context should initially contain:

```text
predicted safe-path length
expected detour ratio
vertical travel
turning / acceleration intensity
minimum predicted boundary/obstacle clearance
expected CBF intervention count or duration
```

Only if this compact context fails should a learned LiDAR embedding be added. This is
the proposed Safe-Trajectory-Conditioned Energy-to-Go direction; it is a future design,
not a result established by the current obstacle-free experiment.

## Recommendation for the Next Formal Run

Do not launch the 500k Phase 2 yet. The evidence supports the following estimator
configuration for a new pre-registered validation/final-test cycle:

```text
Point estimator:
  keep existing MC Energy-to-Go model

Immediate return-now bound:
  heteroscedastic Laplace scale + trajectory-level conformal
  requested marginal risk level = 95%

Mission continuation bound:
  point task + point return-after-task
  + mission heteroscedastic Laplace residual scale
  + mission-trajectory conformal
  requested marginal risk level = 95%

Fallback audit baseline:
  goal/distance group conformal

Reserve:
  unchanged 10%

Battery:
  unchanged calibrated capacity

Policy:
  frozen existing SAC
```

Before formal 500k, generate a new untouched test v4 because final v3 has informed
this recommendation. Pre-register operational gates on CHARGER groups and mission
distance groups, rather than requiring TASK-component coverage that is not directly
used by the direct mission stopping rule. Also retain TASK-component results as an
explicit model-quality diagnostic.

## Required Questions Answered

1. **Why does long distance under-cover?** Longer routes enter boundary-affected
   regimes far more often, creating unobserved path extension/re-acceleration and a
   heavier positive residual tail.
2. **Bias or heteroscedastic tail?** Tail. Long-distance means remain approximately
   zero while standard deviation and P99 increase by about 1.7–1.85×.
3. **Does 7D omit information?** Yes empirically and structurally. Absolute/boundary
   context changes transitions but is absent from 7D.
4. **Is fixed conformal overconservative?** Globally yes: Laplace adaptive width is
   36.5% smaller, but fixed group margins still provide better TASK subgroup coverage.
5. **Which method is best?** Goal: group conformal for subgroup robustness, Laplace
   adaptive for tight marginal/CHARGER bounds. Mission: Laplace residual scale plus
   conformal. Current CQR is inferior.
6. **Best safety-efficiency trade-off?** Mission Laplace adaptive. Goal Laplace is
   efficient but not subgroup-safe enough for a generic all-goal claim.
7. **How to handle mission uncertainty?** Calibrate a direct mission residual scale on
   the sum of two accurate point estimates; component residual correlation is negligible.
8. **What fails with obstacles?** The 7D input cannot see detours or CBF interventions,
   so energy-to-go becomes more severely aliased.
9. **What context should be added?** Predicted safe-path length, detour, vertical travel,
   turning intensity, clearance, and intervention context—not raw LiDAR first.
10. **What should the next formal estimator be?** Frozen MC point model plus
    heteroscedastic Laplace conformal bounds for return-now and direct mission, after
    pre-registration and a new untouched v4 test.

## Evidence Files

```text
fresh_test_v3/method_matrix.csv
fresh_test_v3/goal_results.json
fresh_test_v3/mission_results.json
posthoc_analysis/fresh_mission_tradeoff_matrix.csv
posthoc_analysis/diagnostic_decisions.json
posthoc_analysis/fresh_test_v3_state_residual_groups.csv
posthoc_analysis/fresh_test_v3_scale_diagnostics.json
posthoc_analysis/fresh_test_v3_aliasing_summary.json
posthoc_analysis/fresh_test_v3_aliasing_context.csv
diagnostics/oracle_top50_goal_forensics.json
diagnostics/oracle_top50_mission_forensics.json
short_phase2/summary.json
```

