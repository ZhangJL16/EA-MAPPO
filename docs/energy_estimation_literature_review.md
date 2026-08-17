# Energy Estimation Literature Review

## Scope and source policy

This review supports the Phase1B Energy-TD failure diagnosis. It covers 34 primary papers: 18 UAV propulsion, data-driven power, mission-energy, uncertainty, and charging papers, plus 16 TD, fitted evaluation, and distributional-RL papers. Sources are official DOI pages, IEEE/AAAI/NeurIPS/PMLR proceedings, or author arXiv records. Reviews were used only to expand candidate chains and are not evidence rows. MDPI sources were excluded under the project source policy.

The reusable, field-complete evidence table is `docs/energy_estimation_literature_matrix.csv`.

## 1. Physics-based propulsion and trajectory integration

The recurring architecture in the UAV literature is not a learned Bellman return critic. It is:

```text
instantaneous propulsion/power model
        + planned or simulated trajectory
        -> numerical integration
        -> mission energy
```

[Zeng et al.](https://doi.org/10.1109/TWC.2019.2902559) derive the canonical rotary-wing speed-power model. [Dai et al.](https://doi.org/10.1109/LWC.2022.3195787) extend propulsion modeling to acceleration and direction changes; their [wind-aware extension](https://doi.org/10.1109/TWC.2023.3292290) explicitly conditions trajectory energy on stochastic 3-D wind. [Gong et al.](https://doi.org/10.1109/TAES.2023.3288846) separately model forward, ascent, and descent power and validate on a DJI M210. These papers support retaining an explicit instantaneous-energy layer and integrating along the actual policy trajectory.

For the present simulator, `TelemetryCostModel` is adequate as a *synthetic* instantaneous cost generator because it uses realized velocity, realized propulsion acceleration, and actual substep duration. It is not physically calibrated evidence for a real multirotor: it omits rotor aerodynamics, mass/payload, attitude, vertical-flight asymmetry, battery voltage/SOC effects, and wind.

## 2. Data-driven instantaneous power models

[Gao et al.](https://doi.org/10.23919/JCC.2021.07.020) compare analytical fitting and a model-free DNN on about 12,000 real power-speed points and extend to general 2-D motion. [She et al.](https://doi.org/10.23919/ACC45564.2020.9147622) learn instantaneous UAV power without a full dynamics model. [Muli et al.](https://doi.org/10.1007/978-3-031-20936-9_16) compare five physical models with an LSTM on delivery-drone data and report that the LSTM fits their dataset better.

This cluster suggests a realistic migration path: calibrate or learn the *power model or its residual*, then roll the frozen navigation/safety policy forward. It does not support recursively bootstrapping four tail return landmarks from their own predictions.

## 3. Direct mission-energy prediction

[Prasetia et al.](https://doi.org/10.1109/ACCESS.2019.2903644) segment mission primitives and use Elastic Net regression, reporting 98.773% mean mission-energy accuracy on two held-out surveillance-like patterns. [Dorling et al.](https://doi.org/10.1109/TSMC.2016.2582745) experimentally model payload- and battery-mass effects and integrate segment costs inside delivery routing. [Stolaroff et al.](https://doi.org/10.1038/s41467-017-02411-5) and [Rodrigues et al.](https://doi.org/10.1016/j.patter.2022.100569) provide real-flight evidence that payload, speed, wind, and operating context alter trip energy. Coverage-planning work likewise computes path energy by integrating a model along the path ([Di Franco and Buttazzo](https://doi.org/10.1109/ICARSC.2015.17); [Cabreira et al.](https://doi.org/10.1109/LRA.2018.2854967)).

The literature therefore offers two non-TD estimators that are directly relevant here:

1. supervised return-to-go regression from complete frozen-policy trajectories;
2. policy rollout with instantaneous-energy integration.

## 4. Uncertainty and conservative mission energy

The closest high-quality precedent is [Choudhry et al.](https://doi.org/10.1109/ICRA48506.2021.9561658): a TCN predicts power from time-varying and contextual inputs, Monte Carlo forward simulation produces a flight-energy distribution, and CVaR summarizes tail risk. They report a 29% power-prediction improvement over an analytical comparator on their real-flight data.

This is importantly different from the current code. Choudhry et al. sample uncertainty in a trajectory-forward model. The current implementation treats four learned values at `(0.50, 0.90, 0.95, 0.99)` as both quantile outputs and a sparse approximation to the next-state return distribution. Its custom target masses are not calibrated epistemic uncertainty and are not a standard QR-DQN projection.

## 5. Return-to-home and charging autonomy

[Alyassi et al.](https://doi.org/10.1109/TASE.2022.3175565) combine a learned energy-expenditure model with route/recharge optimization. [Cai et al.](https://arxiv.org/abs/2310.07729) use TSP guidance and MCTS for a mobile charging vehicle. [AutoCharge](https://doi.org/10.1109/ICRA48891.2023.10161503) demonstrates repeated physical docking over a 10-hour mission. [Ahmadi-Javid and Meskar](https://arxiv.org/abs/2211.00842) show that routing can accept a general supplied energy function.

These works separate three layers that the current design should also separate:

- instantaneous energy modeling;
- trajectory/mission energy prediction;
- charger switching and mission planning.

None is evidence that a raw neural Q95 is a safety bound.

## 6. What standard distributional RL actually specifies

[Bellemare et al.](https://proceedings.mlr.press/v70/bellemare17a.html) make the projected distributional Bellman operator part of the algorithm. [Dabney et al. QR-DQN](https://ojs.aaai.org/index.php/AAAI/article/view/11791) use **fixed, equal probabilities** on adjustable atom locations corresponding to uniformly spaced quantile midpoints. [IQN](https://proceedings.mlr.press/v80/dabney18a.html) samples a continuous quantile fraction and learns the full quantile function. [Rowland et al.](https://proceedings.mlr.press/v84/rowland18a.html) further show that convergence claims depend on the exact projection and metric.

The current four-point implementation is therefore a custom approximation, not standard QR-DQN:

```text
reported levels:          0.50, 0.90, 0.95, 0.99
custom target masses:     0.70, 0.225, 0.045, 0.03
monotone outputs:         softplus base + cumulative softplus increments
```

The masses are midpoint/Voronoi quadrature weights over the listed landmarks. They are a defensible numerical heuristic for integrating a coarse quantile function, but the QR-DQN contraction/projection result does not establish this four-landmark recursive operator. The controlled experiments show that equal weighting these same extreme landmarks is even less stable, while a standard uniform quantile grid remains stable.

## 7. TD stability and target networks

Classical TD convergence results are conditional. [Tsitsiklis and Van Roy](https://doi.org/10.1109/9.580874) analyze linear function approximation; [GTD](https://doi.org/10.1145/1553374.1553501) constructs convergent linear stochastic-gradient methods. Modern target-network analyses also require explicit conditions: [Fellows et al.](https://arxiv.org/abs/2302.12537), [Chen et al.](https://doi.org/10.1137/22M1499261), and [Che et al.](https://proceedings.mlr.press/v235/che24a.html). [Manek and Kolter](https://proceedings.neurips.cc/paper_files/paper/2022/hash/e78457d4a04b8565f1fe5077df13cddb-Abstract-Conference.html) show that generic regularization does not solve TD divergence.

The present data are generated by the same frozen deterministic policy used for next actions, so labeling the failure simply as classic off-policy deadly-triad divergence would be inaccurate. Function approximation and bootstrapping remain, but behavior/target-policy mismatch is negligible. Target-network ablations in this repository confirm the literature-level caution: slowing or hard-freezing the target can stop upward propagation only by leaving long-horizon values severely underestimated.

## 8. Multi-step and fitted alternatives

[Retrace](https://papers.nips.cc/paper_files/paper/2016/hash/c3992e9a68c5ae12bd18488bc579b30d-Abstract.html) and [true-online TD(lambda)](https://proceedings.mlr.press/v32/seijen14.html) provide principled ways to trade bootstrap depth against sampled returns. Fitted-evaluation work ([Le et al.](https://proceedings.mlr.press/v97/le19a.html); [Hao et al.](https://proceedings.mlr.press/v139/hao21b.html); [Zhang et al.](https://proceedings.mlr.press/v162/zhang22al.html)) separates point estimation from statistical inference and makes coverage assumptions explicit.

This supports the repository's diagnostic result: 20-step scalar TD is stable and accurate, but it should be compared against direct supervised return regression and model rollout rather than declared the final method from one seed.

## 9. Literature-grounded design conclusion

For the current deterministic simulator and frozen policy, the strongest next design is:

```text
TelemetryCostModel (synthetic instantaneous cost)
        -> frozen-policy model rollout or supervised MC/n-step scalar V(s,g)
        -> held-out residual/ensemble or conformal calibration
        -> mission-energy composition and switching
```

For a real UAV, replace or augment `TelemetryCostModel` with a physics/data hybrid conditioned on payload, wind, attitude, battery state, and vehicle mass. For future obstacle+CBF operation, roll out or learn under the *executed filtered policy*, not the nominal SAC action alone. Distributional learning becomes scientifically meaningful only after introducing real stochastic/contextual variation and evaluating held-out tail coverage; four raw bootstrapped quantiles are not a conservative guarantee.
