# Phase1B Energy TD Root-Cause Report

## 1. Executive conclusion

**一句话结论：当前 TD 的上行发散不是 SAC 漂移、terminal mask、边界碰撞、`gamma=1`、单一学习率或单一 target-network 参数造成的，而是一个缺乏标准 QR-DQN 投影保证的“四个稀疏且高度不均匀 quantile landmark 的递归 Bellman 投影”在长时域神经函数逼近中产生系统性正偏，并由共享 cumulative-softplus base、稀薄 terminal anchor 和一步 bootstrap 共同放大。**

该结论定位到一个 **CONFIRMED ROOT-CAUSE FAMILY**，但不能声称已经从数学上唯一分解出其中每个耦合项的独立贡献。证据权重如下：

| 根因候选 | 主观证据权重 | 判定 |
|---|---:|---|
| 四个非均匀 tail landmarks 的自定义 distributional Bellman projection 与递归 bootstrap 的交互 | 0.60 | **CONFIRMED ROOT-CAUSE FAMILY** |
| cumulative-softplus 共享 base 将所有 quantile 梯度直接耦合 | 0.18 | **STRONGLY SUPPORTED AMPLIFIER** |
| 约 500-step horizon、约 0.18% terminal anchors 与一步传播 | 0.15 | **STRONGLY SUPPORTED AMPLIFIER** |
| 正式 runner 的整段插入顺序、在线 replay turnover 与固定数据诊断的差异 | 0.05 | **POSSIBLE SEVERITY AMPLIFIER** |
| 未发现的数值或表示问题 | 0.02 | **POSSIBLE BUT LOW SUPPORT** |

关键反事实证据是：相同数据、输入、网络尺度、`gamma=1`、target network 和优化器下，scalar TD 在 100k updates 的 held-out MAE 为 **0.5495**，20-step scalar TD 为 **0.2297**，MC scalar 为 **0.2329**；只有当前四 landmark quantile TD 上升到 **20.5981**。标准 32 个均匀 quantile midpoint 的 TD 为 **0.5301**，说明“distributional TD 一概不稳定”也不成立。

因此当前 `GoalConditionedQuantileTDEnergyEstimator` 不应进入 Phase2。下一次正式验证应比较：

1. state-only supervised MC scalar value；
2. 20-step scalar TD；
3. known-model frozen-policy rollout oracle；
4. 仅在存在真实随机上下文后，再评估 supervised distribution/ensemble/calibrated residual。

本报告没有启动 Phase2 或任何新的 500k 正式实验。

## 2. Current implementation audit

### 2.1 SAC freeze audit

Phase1B 使用的 frozen SAC checkpoint 为：

```text
artifacts/uav_energy_delivery_v3_formal_20260816_004619/
  phase1_navigation/checkpoint_transition_500000.zip
SHA256=df9a4354420a1a043b60d704292c6fe06ddd2127e937d3d9c72278b287c1ff0a
```

代码证据：

- `scripts/train_uav_energy_delivery_sac.py:692-709` 对所有 policy 参数执行 `requires_grad_(False)` 并设置 eval mode；
- `scripts/train_uav_energy_delivery_sac.py:712-787` 从 500k checkpoint 重载并验证 7D observation、3D action 和冻结状态；
- `scripts/train_uav_energy_delivery_sac.py:1176-1246` 的 Phase1B loop 只调用 deterministic `policy.predict`，没有 SAC `learn` 或 policy optimizer；
- `phase1_td/phase_boundary.json` 记录 `sac_frozen=true`、source transition 500000、source checkpoint hash；
- 诊断数据采集前后 parameter SHA256 均为 `fa047331...f00`，全部 `requires_grad=false`、training mode false、optimizer calls 0。

**结论：动作网络在 Phase1B 后期学坏已 RULED OUT。** checkpoint 文件 hash 与参数 hash 使用不同序列化口径，因此数值不同并不矛盾。

### 2.2 End-to-end TD data flow

| Stage | Shape | Unit / range | Semantics |
|---|---|---|---|
| Physics velocity | `(3,)` | m/s | propulsion speed saturation 后的实际速度 |
| Physics acceleration | `(3,)` | m/s² | `(v_after_propulsion-v_before)/physics_dt`，不含 boundary projection impulse |
| Substep duration | scalar | 0.05 s | 每个实际执行 physics substep |
| Substep energy | scalar | synthetic energy units | `TelemetryCostModel.realized_cost` |
| Policy transition energy | scalar | synthetic energy units | 实际 1–4 个 substep energy 之和；诊断 mean 0.042218、p99 0.062550、max 0.075077 |
| Energy state | `(7,)` | dimensionless | normalized velocity 3 + goal direction 3 + linear `distance/d_max` 1 |
| SAC action | `(3,)` | `[-1,1]` | frozen deterministic SAC action |
| Critic feature | `(10,)` or `[B,10]` | dimensionless | state 7 + action 3 |
| Online quantiles | `[B,4]` | cumulative synthetic energy | Q50/Q90/Q95/Q99 |
| Next quantiles | `[B,4]` | cumulative synthetic energy | detached target network output |
| Bellman target | `[B,4]` | cumulative synthetic energy | `cost + (1-terminal)*next_quantiles` |
| Pairwise residual | `[B,4,4]` | synthetic energy | each predicted quantile against each target atom |

The trajectory flow is:

```text
physical substeps
  -> realized transition energy
  -> pending old-goal transition
  -> successful goal segment completion
  -> EnergyTDTransition replay insertion
  -> uniform minibatch
  -> online [B,4]
  -> detached target [B,4]
  -> gamma=1 SSP target
  -> weighted pairwise quantile-Huber loss
  -> Adam online update
  -> Polyak target update, tau=0.01
```

Relevant code is `envs/UAVEnergyDeliverySAC.py:737-850`, `envs/UAVEnergyDeliverySAC.py:1263-1294`, and `envs/UAVEnergyDeliverySAC.py:203-238`.

There is no return-energy target normalization. Inputs are normalized but Bellman values remain in raw synthetic units. This is not by itself a bug: scalar TD and MC regression remain stable at the same scale.

### 2.3 Terminal and detach semantics

`quantile_ssp_target` in `review_bundle/safety/energy/td.py:22-35` computes exactly:

```text
target = cost + (~goal_reached) * next_quantiles
```

`envs/UAVEnergyDeliverySAC.py:821-850` keeps the old goal when constructing the terminal transition, sets `goal_reached=True`, and sets the next action to zero. `envs/UAVEnergyDeliverySAC.py:223-225` evaluates the target under `torch.no_grad()`.

The final formal replay deque was not serialized, so byte-identical replay inspection is impossible. Instead, 100 independently reconstructed terminal samples were collected with the exact frozen checkpoint, environment, energy model, and final TD checkpoint. Every sample had:

```text
bootstrap_mask = 0
computed_target_quantiles = [realized_last_cost] * 4
target_equals_realized_cost = true
```

Example: realized terminal cost `0.0456185`, next target Q values approximately `1.51e8, 6.27e8, 6.33e8, 6.40e8`, but all four computed targets remained `0.0456185`.

**Conclusion: erroneous post-terminal bootstrap and missing detach are RULED OUT.**

### 2.4 Current distributional operator is not standard QR-DQN

The production critic uses:

```text
levels = (0.50, 0.90, 0.95, 0.99)
target atom masses = (0.70, 0.225, 0.045, 0.03)
```

The masses are midpoint/Voronoi intervals from `quantile_atom_weights` in `review_bundle/safety/energy/td.py:78-92`. This is a coarse quadrature heuristic over four reported landmarks. Standard QR-DQN instead uses equal-probability atoms at uniformly spaced quantile midpoints. The standard contraction/projection results therefore do not establish the current recursive operator.

The monotone critic in `review_bundle/safety/energy/critics.py:39-76` defines:

```text
Q50 = softplus(base)
Q90 = Q50 + softplus(delta1)
Q95 = Q90 + softplus(delta2)
Q99 = Q95 + softplus(delta3)
```

Every upper-quantile loss therefore sends gradient directly through the common base. In the 50k diagnostic using learning rate `3e-5`, gradient norms were base `0.1614`, increment head `0.01594`, backbone `0.1487`; quantile means were about `28.61, 28.72, 28.77, 28.89`. This is not the production learning-rate trace, but it isolates the same structural behavior: most upward motion was the shared base, not widening uncertainty. The production-parameter 100k checkpoint likewise had means `36.302, 36.539, 36.625, 36.847`, with only `0.239, 0.0848, 0.221` mean increments.

## 3. Existing artifact forensic analysis

### 3.1 Formal failure curve

The formal held-out set contains 500 fixed tasks at every evaluation. Its curve is:

| TD collection transitions | MAE | RMSE | Bias | Q50 coverage | Q95 coverage |
|---:|---:|---:|---:|---:|---:|
| 50k | 14.150 | 14.939 | +13.279 | 0.9238 | 0.9273 |
| 100k | 56.501 | 57.858 | +56.501 | 1.0000 | 1.0000 |
| 150k | 831.780 | 891.209 | +831.780 | 1.0000 | 1.0000 |
| 200k | 6,777.689 | 7,348.630 | +6,777.689 | 1.0000 | 1.0000 |
| 250k | 55,704.206 | 59,977.800 | +55,704.206 | 1.0000 | 1.0000 |
| 300k | 462,840.820 | 499,132.000 | +462,840.820 | 0.9998 | 1.0000 |
| 350k | 3,207,727.389 | 3,408,780.000 | +3,207,727.389 | 1.0000 | 1.0000 |
| 400k | 12,722,080.104 | 13,264,200.000 | +12,722,080.104 | 1.0000 | 1.0000 |
| 450k | 35,264,289.394 | 36,294,700.000 | +35,264,289.394 | 1.0000 | 1.0000 |
| 500k | 81,549,785.814 | 83,325,977.963 | +81,549,785.814 | 1.0000 | 1.0000 |

At 500k, true mean task energy was only **23.770**. Thus 100% coverage is catastrophic overprediction, not calibration.

### 3.2 True return and terminal-anchor scale

The fixed diagnostic corpus uses the same frozen SAC, dynamics, TelemetryCostModel, seeds, state/action contract, and task stratification for every method:

- train: 200 trajectories, 110,821 transitions;
- held-out: 500 trajectories, 281,514 transitions;
- held-out state-level MC return median 13.4265, p90 35.7306, p95 41.4880, p99 47.4710, max 57.8728;
- step energy mean 0.042218, p99 0.062550, max 0.075077.

The formal artifact reports 968 successful segments in 500,000 transitions, implying an aggregate terminal rate near **0.1936%**, one terminal per 516.5 transitions, and an iid batch-128 zero-terminal probability near **78.0%**. The independently reconstructed dataset gives exact values:

| Statistic | Value |
|---|---:|
| Terminal transitions | 200 / 110,821 |
| Terminal fraction | 0.18047% |
| Mean segment length | 554.105 |
| Median segment length | 501.5 |
| p90 / p95 / max length | 1058.1 / 1118.6 / 1269 |
| P(batch has zero terminal), batch 128 | 79.357% |
| Mean terminal samples per batch | 0.2310 |

Terminal anchors are therefore sparse. However, they are an amplifier rather than a sufficient root cause: scalar TD learns correctly with the same replay, while forcing 10% terminal samples makes current quantile TD worse.

### 3.3 Earliest value-scale drift

The formal run stored held-out metrics every 50k but only the final TD model, so it cannot retrospectively provide intermediate online/target tensors. The controlled fixed-dataset current-operator trace uses the same frozen policy, task protocol, dynamics, state/action contract, and energy model to localize the onset:

| Updates | Held-out MAE | Bias | Predicted median | Bellman target mean | Parameter L2 | Gradient L2 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 15.338 | -15.294 | 1.000 | 1.190 | 10.341 | 0.000 |
| 10k | 8.559 | -7.822 | 7.713 | 8.471 | 17.947 | 0.058 |
| 20k | 4.650 | -1.881 | 13.385 | 14.442 | 21.415 | 0.096 |
| 25k | 4.247 | +0.290 | 15.537 | 16.620 | 22.792 | 0.147 |
| 50k | 8.311 | +8.227 | 24.407 | 24.577 | 26.783 | 0.605 |
| 75k | 14.419 | +14.126 | 31.037 | 30.508 | 30.137 | 0.241 |
| 100k | 20.598 | +20.008 | 38.256 | 36.355 | 33.944 | 0.944 |

At 50k, online-target primary absolute gap was only **0.0656**; at 100k it was **0.0907**. The target network is not a stationary correct anchor lagging behind a runaway online network. Online and target move upward together, and the Bellman target itself inherits the wrong scale.

### 3.4 Horizon and terminal proximity

Direct evaluation of the saved 100k checkpoints on the same 281,514 held-out transitions gives:

| Horizon | Current Quantile TD MAE | Scalar TD MAE | MC Scalar MAE |
|---|---:|---:|---:|
| terminal / 1 | 3.5113 | 0.2350 | 0.1958 |
| 2–5 | 4.0302 | 0.1761 | 0.1207 |
| 6–20 | 7.2264 | 0.1742 | 0.0951 |
| 21–100 | 10.8195 | 0.2263 | 0.1331 |
| 101–500 | 18.5128 | 0.2731 | 0.1813 |
| >500 | 29.6147 | 1.1564 | 0.3697 |

Current quantile error grows monotonically with bootstrap horizon. Its terminal prediction is also wrong despite the correct terminal target because shared function approximation does not interpolate all sparse anchors exactly.

### 3.5 Boundary is not causal

Both clean and boundary-contact formal trajectories diverge. At the final online trajectory aggregate, clean MAE was about **9.44 million** and boundary-contact MAE about **6.78 million**. The held-out formal 500k evaluation reached **81.55 million** overall. Boundary contact changes the sampled distribution but cannot explain the shared upward failure.

## 4. Literature findings

The evidence matrix in `docs/energy_estimation_literature_matrix.csv` contains **34 primary papers** across physics-based propulsion, data-driven power, mission energy, uncertainty, return/charging autonomy, and RL value estimation.

Main findings:

1. UAV mission-energy work predominantly models instantaneous power and integrates it along a planned, measured, or simulated trajectory. It does not recursively bootstrap four sparse tail-return landmarks.
2. Real-flight energy depends on payload, wind, attitude, vehicle mass, vertical motion, acceleration, and battery context. `TelemetryCostModel` is useful as a synthetic generator, not as a physically calibrated UAV model.
3. The closest uncertainty precedent, Choudhry et al., predicts power, performs Monte Carlo forward simulation, and computes CVaR over mission energy. It separates trajectory uncertainty from point power prediction.
4. Bellemare et al. make the projection part of the distributional algorithm. QR-DQN uses equal-mass atoms at uniformly spaced quantile midpoints; IQN samples continuous quantile fractions. Current `(0.50,0.90,0.95,0.99)` landmarks with custom masses are not covered by those standard results.
5. Target networks improve timescale separation only under assumptions; they do not make arbitrary nonlinear neural TD stable. Here behavior and target policies are the same frozen deterministic SAC, so a generic “off-policy deadly triad” explanation is too broad.
6. Multi-step TD, fitted evaluation, and direct return regression are all literature-supported alternatives, but uncertainty coverage still requires explicit held-out assumptions and calibration.

The full paper-by-paper inputs, targets, uncertainty methods, rollout/MC/bootstrapping flags, datasets, metrics, limitations, and primary links are in the literature matrix and `docs/energy_estimation_literature_review.md`.

## 5. Hypothesis table

| Hypothesis | Evidence for | Evidence against | Diagnostic experiment | Result | Status |
|---|---|---|---|---|---|
| SAC changed during Phase1B | Would make target policy nonstationary | Hash unchanged; frozen params; no optimizer calls | checkpoint/code/hash audit | No change | **RULED OUT** |
| Terminal transitions bootstrap after goal | Could make SSP cyclic | 100/100 terminal targets equal final cost despite huge next Q | real-policy terminal audit | mask exactly zero | **RULED OUT** |
| Missing target detach | Could optimize through target | `torch.no_grad()` around target model | code and gradient-path audit | detached | **RULED OUT** |
| Boundary collision causes large costs | Formal boundary trajectories exist | Clean trajectories also diverge; step costs bounded | clean vs boundary split | clean also catastrophic | **RULED OUT** |
| `gamma=1` alone is unstable | Long undiscounted horizons amplify errors | scalar TD with same gamma is accurate | scalar TD | MAE 0.5495 | **RULED OUT AS SOLE CAUSE** |
| Learning rate too high | Lower LR can delay large updates | 1e-4, 3e-5, 1e-5 still drift upward by 50k | LR sweep | MAE 10.55–17.33 | **RULED OUT AS SOLE CAUSE** |
| Polyak `tau=0.01` too fast | Target follows online closely | slower/hard targets suppress propagation by leaving values severely low | tau/hard sweep | tau .001 MAE 12.72, hard10k 15.00 | **AMPLIFIER, NOT ROOT FIX** |
| Terminal anchors too rare | 79.36% batches contain none | scalar TD survives; terminal10 worsens | terminal-balanced replay | MAE 29.03 | **AMPLIFIER, NOT SUFFICIENT** |
| State representation is irreducibly aliased | Absolute position omitted | closest 1% cross-trajectory 10D pairs differ in return by mean 0.0231; MC fits well | NN conditional analysis + MC | no strong ambiguity | **RULED OUT FOR CURRENT OPEN WORLD** |
| Action conditioning is essential | Future filtered action may matter | frozen deterministic policy fixes action | state-only MC | MAE 0.2246 vs action MC 0.2329 | **RULED OUT FOR CURRENT POLICY** |
| Monotone cumulative head alone causes failure | All upper losses flow through base | free independent four-head TD also drifts | free-four quantile TD | MAE 13.93 | **STRONG AMPLIFIER, NOT SOLE CAUSE** |
| Four sparse extreme landmarks and recursive projection cause positive drift | current/free-four drift; uniform32 stable; MC quantile stable | simple 10-step SSP does not explode | projection, bootstrap, synthetic SSP | strongest separation | **CONFIRMED ROOT-CAUSE FAMILY** |
| Equal weighting would fix target masses | Simpler standard-looking loss | equal weight on nonuniform extreme landmarks is not QR-DQN | unweighted-four | MAE 17.9 million at 50k | **RULED OUT; MUCH WORSE** |
| Fixed known-model rollout is inaccurate | Numerical reset mismatch possible | same policy/dynamics/cost model should reproduce trajectory | 100-state rollout | MAE 3.20e-7 | **RULED OUT** |

## 6. Controlled experiments

### 6.1 Primary fixed-data comparison, 100k updates

| Method | Bootstrap | Distributional representation | Held-out MAE | Bias | Predicted median |
|---|---|---|---:|---:|---:|
| Current Quantile TD | 1-step | four nonuniform landmarks + custom masses + monotone head | 20.5981 | +20.0085 | 38.2560 |
| Scalar TD | 1-step | scalar | 0.5495 | -0.3345 | 13.3587 |
| MC scalar | none | scalar | 0.2329 | -0.0443 | 13.4102 |
| MC quantile | none | same monotone four outputs | 0.2229 median | -0.0269 | 13.4324 |
| 20-step scalar TD | 20-step | scalar | 0.2297 | -0.0075 | 13.4390 |
| State-only MC scalar | none | scalar V(s,g) | 0.2246 | -0.0150 | 13.4242 |
| Uniform 32-quantile TD | 1-step | standard midpoint grid | 0.5301 median | -0.2104 | 13.3908 |
| Free four-landmark TD | 1-step | independent four outputs, same sparse levels/masses | 13.9287 | +13.9286 | 26.0448 |

These are single-seed diagnostics, not paper-ready method rankings. Their role is causal isolation.

### 6.2 Projection isolation

- Same extreme `(0.50,0.90,0.95,0.99)` atoms with equal weights exploded by 50k: MAE **17,892,717.8**, prediction median **18,909,052**, gradient norm **49,438**.
- Four uniform midpoint levels `(0.125,0.375,0.625,0.875)` with equal weights remained finite at 50k: selected central-head MAE **2.5109** and prediction median **13.1735**. This is a stability diagnostic, not an exact Q50 estimator because the selected level is 0.375.
- 32 uniform midpoint quantiles achieved 100k median MAE **0.5301**.

The custom midpoint masses mitigate the obviously wrong equal-weight extreme-atom distribution, but they do not supply the dense, equal-mass projection used by standard QR-DQN.

### 6.3 Target update and learning-rate sweeps

At 50k:

| Variant | MAE | Bias | Predicted median | Interpretation |
|---|---:|---:|---:|---|
| tau 0.005 | 5.4889 | -1.7197 | 14.5593 | slower propagation; still horizon bias |
| tau 0.001 | 12.7226 | -12.4862 | 3.7229 | fails to propagate value scale |
| tau 0.0001 | 15.0349 | -14.9816 | 1.3036 | essentially frozen low value |
| hard 500 | 10.2590 | -9.6111 | 6.6261 | under-propagation |
| hard 2,000 | 13.8115 | -13.6927 | 2.5255 | under-propagation |
| hard 10,000 | 15.0035 | -14.9442 | 1.3394 | under-propagation |
| LR 1e-4 | 10.5461 | +9.9431 | 25.0959 | positive drift remains |
| LR 3e-5 | 14.0652 | +12.3170 | 29.5207 | positive drift remains |
| LR 1e-5 | 17.3251 | +16.0408 | 32.8089 | positive drift remains |

Changing timescale or step size trades upward propagation against long-horizon underestimation; it does not repair the operator.

### 6.4 Synthetic SSPs

| SSP | True scale | Scalar TD MAE | Current Quantile TD MAE | MC scalar MAE |
|---|---|---:|---:|---:|
| 10 steps × cost 1 | start 10 | 0.00546 | 0.03127 | 0.00788 |
| 500 steps × cost 0.05 | start 25 | 0.12839 | 0.86907 | 0.00703 |
| 100-step random positive cost | finite known RTG | 0.07243 | 0.27778 | 0.07271 |

The current operator is not a trivial always-diverging code path. It exhibits growing positive bias as horizon increases, while the severe UAV explosion requires nonlinear shared approximation, broad state coverage, recursive replay, and long horizons together.

### 6.5 Model-based rollout

On 100 held-out states, cloning the current physical state, rolling the exact frozen policy and known dynamics to the goal, and integrating `TelemetryCostModel` produced:

| Metric | Value |
|---|---:|
| MAE | 3.204e-7 |
| RMSE | 4.775e-7 |
| Bias | -7.220e-8 |
| Mean rollout length | 367.78 policy steps |
| Mean inference time | 0.5257 s |
| p95 inference time | 1.2423 s |
| Mean time / simulated step | 1.429 ms |

It is an accuracy oracle in the deterministic known simulator, but is too slow for high-frequency evaluation of many candidate actions without batching, caching, reduced models, or lower switching frequency.

## 7. Recommended estimator architecture

### Level A — Instantaneous energy model

Keep `TelemetryCostModel` as the current simulator's synthetic realized-cost generator. Do not label its units Wh or J and do not claim real-UAV validity. For hardware transfer, replace or augment it with a physics/data hybrid conditioned on payload, wind, attitude, vehicle mass, vertical flight, battery voltage/SOC, and vehicle-specific calibration.

### Level B — Energy-to-go estimator

Recommended next formal comparison:

```text
Primary stable learned baseline:
    state-only supervised MC scalar V_E^pi(s,g)

Online adaptation candidate:
    20-step scalar TD V_E^pi(s,g)

Accuracy oracle / deployment baseline:
    frozen-policy dynamics rollout + realized power integration
```

Why state-only: under the frozen deterministic policy, `a=pi(s,g)` is nearly determined by state and goal. Removing action reduced held-out MAE from 0.2329 to 0.2246 in the diagnostic run. When a future CBF changes executed action, the estimator must instead condition on the filtered-policy context or learn under executed trajectories.

Why MC first: complete successful trajectories already exist; labels are exact realized return-to-go samples; no recursively moving target exists. Why retain 20-step TD: it achieved comparable accuracy and offers a path to incremental updates before full trajectories finish.

The next formal experiment should use at least three seeds, the same 500 held-out tasks, equal source transitions, and report wall-clock/inference cost. A single diagnostic seed is insufficient to declare MC or n-step TD the final paper method.

### Level C — Conservative risk estimate

Do not use raw current Q95 as a bound. In the current deterministic simulator, repeated state-conditioned return distributions are nearly degenerate, so four tail outputs have weak scientific meaning.

Preferred sequence:

1. establish a stable point estimator;
2. introduce declared stochastic/context variables such as wind, payload, battery aging, and executed CBF corrections;
3. estimate uncertainty using model ensemble/bootstrap or Monte Carlo rollout;
4. calibrate a one-sided residual bound on a strictly held-out calibration set;
5. report empirical coverage, underestimation, width, and distribution-shift failures.

For a trajectory-forward model, CVaR or an upper quantile of Monte Carlo mission energy is more interpretable than recursively bootstrapping four sparse neural return landmarks.

### Hybrid candidate

The strongest longer-term architecture is:

```text
physics/data-driven instantaneous power model
    -> short/full rollout under executed navigation+safety policy
    -> learned scalar residual or value correction
    -> held-out uncertainty calibration
    -> mission switching
```

It is more explainable and physically calibratable than pure TD, and can include CBF detours by rolling out the executed filtered policy. It still requires experiments; this report does not claim it is already superior.

## 8. Why alternatives were rejected

- **Keep current Quantile TD and tune it:** rejected. LR, tau, hard targets, and terminal oversampling do not repair the operator.
- **Set gamma below one:** rejected for the SSP target because it changes the quantity from actual cumulative energy to discounted energy.
- **Equal-weight the existing four tail atoms:** rejected; it caused the fastest explosion.
- **Use terminal-balanced replay as the primary fix:** rejected; 10% terminal sampling worsened MAE to 29.03.
- **Add absolute position immediately:** rejected for the current open world. Closest 1% cross-trajectory 10D neighbors had mean return difference 0.0231 despite mean absolute-position separation 1069 m; MC regression already fits accurately.
- **Keep action conditioning by default:** rejected for the current frozen deterministic policy because state-only is slightly better and simpler. Reconsider after action filtering.
- **Use model rollout alone everywhere:** not rejected as a baseline, but currently 0.526 s mean per query is operationally expensive. It should be optimized or evaluated at a slower decision cadence.
- **Call 100% Q95 coverage safe:** rejected. The bound is eight orders of magnitude too large and causes zero-throughput behavior.
- **Declare MC universally superior:** rejected. MC is stable here because successful full trajectories are available; online, censored, or changing-policy settings may favor multi-step or model-based methods.

## 9. Implications for Phase2 switching

Phase2 must remain blocked for the current checkpoint.

The recorded full-capacity TD-managed scene used calibrated capacity **378.7263** and reserve **37.8726**, but the initial predicted mission Q95 was **354,401,816**. The resulting margin was **-354,401,475**. The agent committed immediately, completed **0 tasks**, and produced **21 zero-task battery cycles**, including **20 one-step zero-task cycles** after first recharge. This is a self-confirmed charger loop, not conservative success.

The separate 5%-battery trace correctly reached the environment's `energy_exhausted` terminal, showing that the state-machine failure path works. It does not validate the estimator.

Required gate before Phase2:

1. stable held-out point error over the full training schedule;
2. no value-scale alarm relative to empirical MC p99;
3. declared one-sided calibration on unseen trajectories;
4. switching smoke with nonzero task throughput, successful return, recharge, and no zero-task charger loop;
5. frozen estimator checkpoint and exact provenance.

## 10. Implications for future obstacle+CBF system

Obstacle-free state-only value works because the frozen policy and relative goal geometry almost determine the trajectory. After adding obstacles and a CBF:

```text
nominal SAC action
  -> local perception / CBF
  -> executed filtered action
  -> detour and altered energy
```

The energy estimator must be trained or rolled out under the **executed** policy, not nominal SAC alone. Candidate context includes local obstacle representation, filtered action, wind, payload, battery condition, and safety intervention history where it affects future trajectory.

Recommended progression:

1. validate scalar MC / 20-step TD / rollout in the current open world;
2. freeze the selected estimator protocol, not just one checkpoint;
3. introduce obstacle layouts with the same frozen SAC and explicit CBF execution;
4. recollect executed-policy energy trajectories;
5. test whether rollout-model mismatch or state aliasing emerges;
6. only then add residual learning and calibrated uncertainty.

This preserves causal attribution: navigation, instantaneous power, cumulative energy prediction, uncertainty, and switching remain separately testable layers.

## Evidence artifacts

- Formal run: `artifacts/uav_energy_delivery_v3_energy_formal_20260816_175614/`
- Diagnostic corpus/audit: `artifacts/energy_td_diagnostics_20260817_v1/dataset_audit.json`
- Terminal audit: `artifacts/energy_td_diagnostics_20260817_v1/terminal_target_audit.json`
- Fixed-data matrices: `artifacts/energy_td_diagnostics_20260817_v1/matrix_*`
- Saved-checkpoint horizon audit: `artifacts/energy_td_diagnostics_20260817_v1/checkpoint_grouped_metrics.json`
- Synthetic SSPs: `artifacts/energy_td_diagnostics_20260817_v1/synthetic_ssp/synthetic_results.json`
- State ambiguity: `artifacts/energy_td_diagnostics_20260817_v1/state_ambiguity.json`
- Model rollout: `artifacts/energy_td_diagnostics_20260817_v1/model_based_rollout_100/model_based_rollout.json`
- Phase2 failure GIF summary: `artifacts/uav_energy_delivery_v3_energy_formal_20260816_175614/energy_managed_scene_full_current_td_20260817/summary.json`
- Literature review: `docs/energy_estimation_literature_review.md`
- Literature matrix: `docs/energy_estimation_literature_matrix.csv`
