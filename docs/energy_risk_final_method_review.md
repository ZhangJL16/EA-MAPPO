# Energy Risk Final-Method Review

## Scope And Decision

This review fixes the statistical interpretation used for Energy Risk v5. It
does not treat a raw observed subgroup percentage as a finite-sample guarantee,
and it does not claim distribution-free conditional coverage at every state.

The selected target is:

1. one complete Goal or Mission trajectory is one exchangeable calibration
   unit;
2. the nominal target is 95%;
3. physically predefined finite groups receive separate split-conformal
   calibration distributions (Mondrian calibration);
4. fresh-test subgroup results are reported with covered count, sample count,
   exact and Wilson intervals;
5. a familywise one-sided undercoverage stress test is reported across the
   preregistered primary groups;
6. arbitrary post-hoc slices remain empirical diagnostics only.

This is **predefined group-conditional marginal coverage**, conditional on the
fixed group label and the stated within-group exchangeability assumption. It is
not arbitrary conditional coverage at `X=x`.

## Three Coverage Concepts

### Marginal Coverage

For an exchangeable calibration trajectory and fresh trajectory,

`P(Y <= U(X)) >= 1-alpha`.

The probability averages over the full deployment distribution. Standard
split conformal supports this finite-sample statement when model fitting is
independent of calibration and the fresh unit is exchangeable with calibration
units.

### Predefined Group-Conditional Coverage

For a fixed finite partition chosen before calibration/test inspection,

`P(Y <= U_g(X) | G=g) >= 1-alpha`.

Direct Mondrian calibration supports this statement separately in each group
when the calibration and fresh units within that group are exchangeable and
the finite-sample conformal rank exists. Pooling a parent group as a fallback
changes the guarantee to the parent group; it does not preserve the missing
child-group guarantee.

### Arbitrary Conditional Coverage

`P(Y <= U(X) | X=x) >= 1-alpha` for every `x` is substantially stronger.
Without structural/distributional assumptions, nontrivial finite-sample,
distribution-free exact conditional coverage is impossible. Local, multivalid,
and feature-adaptive methods offer useful relaxations or assumption-dependent
results, not an unrestricted replacement for this impossibility.

## Statistical Audit Of The Old Raw Gate

The v4 Goal failure `TASK x 2500-4000 m` was `215/231 = 93.0736%`.

- exact two-sided 95% interval: `[88.9955%, 95.9894%]`;
- Wilson 95% interval: approximately `[89.05%, 95.69%]`;
- plug-in standard error: `1.6706` percentage points.

The v4 Mission failure `>4000 m` was `375/400 = 93.75%`.

- exact two-sided 95% interval: `[90.9118%, 95.9149%]`;
- plug-in standard error: `1.2103` percentage points.

Both exact intervals contain 95%. This does not prove valid 95% coverage, but
it shows that the observed deviations alone do not establish systematic
undercoverage.

If true coverage equals exactly 95%, the probability that the raw empirical
fraction is at least 95% is:

| subgroup n | `P(empirical >= .95 | p=.95)` | p needed for 90% pass | p needed for 95% pass |
| ---: | ---: | ---: | ---: |
| 100 | 61.60% | 96.818% | 97.355% |
| 200 | 58.31% | 96.463% | 96.885% |
| 250 | 51.75% | 96.518% | 96.897% |
| 500 | 55.29% | 96.036% | 96.331% |
| 1,000 | 53.75% | 95.774% | 95.995% |

At `n=231`, the pass probability is about `51.22%`. Therefore raw empirical
coverage `>=95%` is not a well-designed sole pass/fail gate for a method whose
true coverage is exactly 95%: it rejects such a method roughly half the time.

## Candidate Gates

### Gate A: Raw Empirical Coverage At Least 95%

- Strength: transparent and strict in the observed sample.
- Defect: unstable at finite sample size and has about 50% power to pass a true
  95% method for common subgroup sizes.
- Decision: report as a descriptive metric, not the sole scientific gate.

### Gate B: Finite-Sample Conformal Construction Plus Empirical Stress Tests

- Strength: ties the claimed guarantee to the construction and its explicit
  exchangeability/partition assumptions; preserves nominal 95% without tuning
  the test-set margin.
- Defect: guarantee is over calibration-and-test draws within predefined
  groups, not a statement that every realized test subgroup percentage must be
  at least 95%.
- Decision: primary paper gate.

### Gate C: Empirical Target Plus Confidence/Hypothesis Test

- Strength: quantifies fresh-test uncertainty and can detect material
  undercoverage.
- Defect: failure to reject is not proof of validity; multiple groups require
  multiplicity correction and power disclosure.
- Decision: use as the stress-test component of Gate B. Use one-sided exact
  binomial tests of `H0: p>=.95` versus `H1: p<.95`, with Holm familywise
  correction over preregistered primary groups. Also report two-sided exact and
  Wilson intervals.

## Primary And Diagnostic Groups

### Goal Primary Groups

Canonical Goal type is fixed before v5:

- `TASK`;
- `CHARGER`, combining direct charger trajectories and
  `TASK_ENDPOINT_TO_CHARGER` trajectories because both deploy the same frozen
  charger-goal policy and estimate the same physical Goal object.

Distance groups are fixed in metres:

- `100-500`;
- `500-1500`;
- `1500-2500`;
- `2500-4000`;
- `>4000` where geometrically feasible.

The formal Goal partition is canonical `goal_type x distance`. Each supported
interaction group receives its own calibration score distribution. Overall,
goal-type-only, and distance-only numbers are reported aggregations.

### Mission Primary Groups

- initial task distance `100-500`;
- `500-1500`;
- `1500-2500`;
- `2500-4000`;
- `>4000`.

Mission is calibrated directly on complete `state -> TASK -> CHARGER` mission
scores rather than by adding two separately calibrated Goal bounds.

### Diagnostic Groups

The following are stress tests, not formal conditional guarantees:

- speed/acceleration bins;
- path-ratio bins;
- boundary proximity/contact bins;
- vertical displacement bins;
- short-rollout difficulty deciles;
- any post-v5 slice not listed in preregistration.

## Primary-Paper Matrix

| Paper | Method | Guarantee type | Conditioning | Sample assumptions | Strength | Limitation | Relevance |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [Vovk 2012, Conditional Validity of Inductive Conformal Predictors](https://proceedings.mlr.press/v25/vovk12.html) | Taxonomy and modifications of inductive conformal validity | Finite-sample conformal validity; explores conditional variants | Taxonomy includes label/group/object notions | Exchangeability appropriate to each construction | Establishes that “conditional” has several non-equivalent meanings | Does not make unrestricted object-conditional validity free | Supports precise naming of our group guarantee |
| [Lei and Wasserman 2014, Distribution-Free Prediction Bands](https://doi.org/10.1111/rssb.12021) | Distribution-free regression bands with local adaptation | Marginal finite-sample; asymptotic conditional properties under regularity | Local feature neighborhoods | i.i.d.; smoothness for conditional efficiency/results | Foundational local adaptivity | Conditional claims require assumptions and enough local data | Warns against claiming exact arbitrary conditional coverage |
| [Lei et al. 2018, Distribution-Free Predictive Inference for Regression](https://doi.org/10.1080/01621459.2017.1307116) | Split/full conformal regression analysis | Distribution-free marginal coverage | None exact; efficiency can adapt | i.i.d./exchangeability | Clarifies finite-sample rank and efficiency | Marginal validity can hide subgroup failures | Basis for trajectory split conformal |
| [Barber et al. 2021, Limits of Distribution-Free Conditional Predictive Inference](https://arxiv.org/abs/1903.04684) | Impossibility and approximate conditional relaxations | Proves exact distribution-free conditional coverage is generally impossible nontrivially | Rich covariate subsets / pointwise | Distribution-free target exposes impossibility | Defines defensible middle ground | Relaxations trade width or restrict groups | Rules out “all subgroups at 95%” as a valid demand |
| [Romano et al. 2019, Conformalized Quantile Regression](https://proceedings.neurips.cc/paper_files/paper/2019/hash/5103c3584b063c431bd1268e9b5e76fb-Abstract.html) | Quantile regression plus conformal correction | Finite-sample marginal coverage | Input-adaptive width, not exact `X=x` coverage | Exchangeable calibration/test; fitted quantiles independent of calibration | Efficient under heteroscedasticity | Bad quantile estimation can leave local undercoverage before correction | Motivates adaptive positive-residual/scale models |
| [Tibshirani et al. 2019, Conformal Prediction Under Covariate Shift](https://proceedings.neurips.cc/paper_files/paper/2019/hash/8fb21ee7a2207526da55a679f0332de2-Abstract.html) | Likelihood-ratio weighted conformal | Distribution-free under weighted exchangeability/covariate shift | Target covariate distribution | Known or accurately estimated density ratio; invariant conditional label law | Handles a specific deployment shift | Not valid for arbitrary policy/conditional shift | Future CBF policy shift may need reweighting or new calibration |
| [Guan 2023, Localized Conformal Prediction](https://academic.oup.com/biomet/article/110/1/33/6647831) | Test-local weighted score distribution | Assumption-free finite-sample marginal; local guarantees under added assumptions | Local around test feature | i.i.d.; assumptions for local coverage | Continuous difficulty adaptivity | Naive local quantiles can under-cover; implementation and tuning are heavier | Candidate only if fixed groups remain too coarse |
| [Hore and Barber 2025, Conformal Prediction With Local Weights](https://arxiv.org/abs/2310.07850) | Randomized local weighting | Marginal and local-weight guarantees defined by method | Random local neighborhoods | Exchangeability plus method randomization | Sharper formal local interpretation | Newer and more complex than fixed Mondrian groups | Not preferred unless width improves materially |
| [Gibbs and Candès 2021, Adaptive Conformal Inference Under Distribution Shift](https://proceedings.neurips.cc/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html) | Online update of miscoverage level | Long-run average coverage under shifts | Time/adaptive sequence | Sequential feedback assumptions | Responds to nonstationarity | Not a static fresh-test subgroup guarantee | Future policy/battery drift extension, not current v5 core |
| [Gibbs and Candès 2024, Online Prediction With Arbitrary Distribution Shifts](https://arxiv.org/abs/2208.08401) | Online conformal control under broad shifts | Long-run calibration/control | Sequential time | Online feedback and boundedness-type conditions | Handles changing distributions | Long-run result differs from one-shot trajectory coverage | Relevant after online UAV adaptation begins |
| [Angelopoulos et al. 2024, Conformal Risk Control](https://proceedings.iclr.cc/paper_files/paper/2024/file/f3549ef9b5ff520a7e41ff3cc306ab2b-Paper-Conference.pdf) | Calibrates a nested decision family for monotone bounded loss | Expected risk control, tight up to finite-sample factor | None unless loss/groups encode it | Exchangeable random loss functions; monotonicity | Controls more general losses than coverage | Expected risk is not per-group trajectory coverage | Useful for future direct exhaustion/unnecessary-return risk control |
| [Jung et al. 2022, Batch Multivalid Conformal Prediction](https://arxiv.org/abs/2209.15145) | Multivalid calibration over many groups | Approximate simultaneous group calibration | User-specified/intersecting groups | Calibration sample and learnability/complexity conditions | Covers richer group families | Approximate, data-intensive, more complex | Overkill for a small physical partition unless diagnostics demand it |
| [Bairaktari et al. 2025, Kandinsky Conformal Prediction](https://arxiv.org/abs/2502.17264) | Feature- and label-dependent group coverage via learned/randomized construction | Finite-sample coverage over a specified group class | Beyond simple class/covariate partitions | Exchangeability and method-specific group representation | Rich group expressiveness | Very recent; complexity and relevance exceed current fixed partition | Review candidate, not default final method |
| [Blot et al. 2024, Automatically Adaptive Conformal Risk Control](https://arxiv.org/abs/2406.17819) | Data-adaptive risk-control parameters | Risk-control guarantee for adaptive families under stated setup | Adaptive model/output difficulty | Exchangeability and nested/monotone risk family conditions | Reduces manual scale design | Does not directly solve trajectory group coverage | Candidate when optimizing risk/width jointly, not needed first |
| [Bellotti 2021, Approximation to Object Conditional Validity](https://proceedings.mlr.press/v152/bellotti21a.html) | Learns approximations to object-conditional validity | Approximate/model-dependent conditional validity | Individual object | Correctness of fitted conditional error model | Makes local failure visible | Loses assumption-free exact conditional guarantee | Frames our adaptive risk network as empirical, not certifying by itself |
| [Barber et al. 2023, Conformal Prediction Beyond Exchangeability](https://arxiv.org/abs/2202.13415) | Weighted quantiles and randomization for drift/nonsymmetry | Robust coverage bounds; recovers standard guarantee under exchangeability | Recency/weights | Quantified departures and weights | Explicitly treats nonexchangeability | Bounds degrade with shift; does not erase policy drift | Future obstacle/CBF data require a new shift audit |
| [Lindemann et al. 2023, Safe Planning in Dynamic Environments Using Conformal Prediction](https://arxiv.org/abs/2210.10254) | Trajectory prediction regions inside MPC | Probabilistic planning statement from trajectory-level conformal regions | Entire predicted environment trajectory | Offline trajectory exchangeability and planner assumptions | Demonstrates trajectory calibration in autonomy | Different target: obstacle trajectories, not energy-to-go | Supports complete-trajectory Goal score as safety unit |
| [Dixit et al. 2023, Adaptive Conformal Prediction for Motion Planning](https://proceedings.mlr.press/v211/dixit23a/dixit23a.pdf) | Online multistep uncertainty sets used by MPC | Long-run average probabilistic coverage | Time-varying multistep predictions | Delayed online observations and adaptive conformal conditions | Connects online conformal and drone planning | Average temporal coverage differs from fresh independent Goal trajectories | Future online recalibration reference |
| [Sun et al. 2023, PlanCP](https://papers.nips.cc/paper_files/paper/2023/hash/fe318a2b6c699808019a456b706cd845-Abstract-Conference.html) | Conformal uncertainty for planning with learned dynamics | Marginal prediction/planning uncertainty under calibration assumptions | Rollout/planning trajectory | Calibration trajectories compatible with deployment | Shows conformalized model-based planning | Full rollout cost and model mismatch remain | Motivates short rollout as deployable context, full rollout as oracle |
| [Stolaroff et al. 2018, Energy Use of Drones for Commercial Package Delivery](https://www.nature.com/articles/s41467-017-02411-5) | First-principles/life-cycle delivery energy analysis | Physical model/sensitivity, not predictive coverage | Vehicle, payload, route | Specified vehicle/aerodynamic scenarios | Establishes physical route/payload dependence | Not an uncertainty-calibration method | Supports physical grouping and future payload/wind shifts |
| [Rodrigues et al. 2022, Drone Flight Data Reveal Energy Savings](https://pmc.ncbi.nlm.nih.gov/articles/PMC9403403/) | Empirical flight-regime energy model from 188 flights | Held-out predictive accuracy, not conformal safety | Flight regime, payload, speed, altitude | Train/test split by flight; measured telemetry | Demonstrates path/time/regime effects and flight-level splitting | Specific platform/range; no upper-tail guarantee | Supports trajectory split and decision-time motion context |
| [Muli et al. 2022, Comparative Study on Energy Consumption Models for Drones](https://arxiv.org/abs/2206.01609) | Compares analytical and empirical drone energy models | Empirical model-comparison evidence | Dynamics/environment variables | Dataset/model-specific | Shows energy models vary materially by assumptions | No deployment coverage guarantee | Reinforces keeping realized telemetry separate from risk calibration |
| [Li et al. 2024, Rotary-Wing UAV Energy Model](https://doi.org/10.1002/rob.22359) | Data/physics-informed rotary-wing energy modeling | Predictive validation under experiment conditions | Flight state and vehicle dynamics | Platform-specific measured data | Richer physical effects than distance-only models | Transfer requires new calibration | Future real-UAV feature roadmap, not a current conformal theorem |

## Method Selection Logic

### Goal

The frozen 7D point model remains the center unless contextual point retraining
materially improves held-out MAE and P95/P99 underestimation. Difficulty is
handled by a separate deployable risk model. Candidate ordering is:

1. direct positive residual prediction;
2. future-suffix trajectory maximum residual prediction;
3. hybrid state and suffix risk;
4. compact decision-time position/boundary context;
5. short frozen-SAC rollout context only if it improves the coverage-width
   frontier enough to justify latency.

Each candidate receives independent direct Mondrian interaction-group
calibration. The learned risk network is an efficiency mechanism; the formal
finite-sample statement comes from the held-out calibration rank under the
stated exchangeability assumptions.

### Mission

The frozen heteroscedastic Laplace model remains the base because it is already
tighter than summing Goal upper bounds. The first correction is direct Mission
distance-group Mondrian conformal calibration. A new Mission architecture is
not justified unless this simple correction fails development diagnostics.

### Future Obstacles And CBF

The recommended interface is:

`motion state + goal geometry + compact safe-trajectory context`.

Raw LiDAR should first enter the collision filter. The executed CBF-modified
trajectory then changes the Energy data distribution, so new MC trajectories
and fresh Energy calibration are required. Current obstacle-free v5 coverage
does not transfer automatically.

## Claim Boundaries

- The finite-sample statement is conditional on the fixed group and within-group
  exchangeability of complete trajectories.
- A learned scale or residual network alone is not a coverage guarantee.
- Fresh-test confidence intervals quantify evaluation uncertainty; failure to
  reject undercoverage is not proof that every future subgroup has 95% coverage.
- No result supports arbitrary state-conditional coverage.
- No result transfers unconditionally to obstacles, CBF intervention, wind,
  payload shift, battery aging, or navigation-policy drift.
