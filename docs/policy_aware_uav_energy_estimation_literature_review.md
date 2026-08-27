# UAV Energy Estimation and Policy-Aware Energy Learning: Literature Review

## 1. Scope and evidence policy

- Search date: 2026-08-26.
- Research question: how prior work estimates UAV power, trajectory energy, mission energy, or energy risk; and which work can support a policy-aware Energy Critic when navigation and safety policies change.
- Source policy: primary publisher/proceedings pages or author arXiv versions. Surveys were used only to expand search terms, not as evidence for method claims.
- This document complements the broader 34-paper review in `docs/energy_estimation_literature_review.md`. It is narrower: it organizes the papers around the current JSEB design and the proposed two-layer policy-aware learning loop.
- Terminology boundary: battery state-of-charge/state-of-health estimation is not the same problem as predicting propulsion energy-to-go. Battery papers are therefore not treated as direct Energy Critic precedents.

## 2. Main finding

Prior UAV work predominantly follows one of four pipelines:

```text
A. analytical/empirical instantaneous power model
   + known or planned trajectory
   -> numerical integration
   -> mission energy

B. telemetry features
   -> supervised instantaneous-power model
   -> trajectory integration

C. complete mission descriptors or motion primitives
   -> supervised direct mission-energy regression

D. uncertain trajectory/context model
   -> Monte Carlo energy distribution
   -> tail-risk statistic such as CVaR
```

Directly learning an undiscounted, action-conditioned UAV energy-to-go value by TD is not the dominant UAV approach. Policy-conditioned value functions, policy fingerprints, and two-timescale actor--critic updates exist in general RL, but the reviewed primary work does not directly solve the full combination:

```text
changing goal-conditioned UAV policy
+ HOCBF-executed rather than nominal actions
+ obstacle-induced braking and detours
+ policy-aware energy-to-go distribution
+ energy-aware policy adaptation and charger commitment
```

That absence does not by itself establish novelty. Policy conditioning and two-timescale learning are established mechanisms; a paper must show a specific, reproducible failure of policy-agnostic energy estimation and a measurable benefit from the proposed coupling.

## 3. Physics-based and empirical propulsion models

| Paper | Estimated object | Inputs and method | Output/use | Relation to current work |
|---|---|---|---|---|
| Zeng, Xu, and Zhang, **Energy Minimization for Wireless Communication With Rotary-Wing UAV**, IEEE TWC 2019 ([DOI](https://doi.org/10.1109/TWC.2019.2902559)) | Instantaneous rotary-wing propulsion power | Closed-form blade-profile, induced, and parasite power terms as functions of flight speed and vehicle parameters | Integrates the power model while jointly optimizing trajectory, communication allocation, and mission time | Canonical analytical baseline. It predicts energy along an optimized trajectory; it is not a learned return critic and is not policy-aware. |
| Abeywickrama et al., **Empirical Power Consumption Model for UAVs**, VTC-Fall 2018 ([DOI](https://doi.org/10.1109/VTCFall.2018.8690666)) | Instantaneous/segment propulsion power | Empirical flight measurements and fitted relationships between operating state and power | Provides an energy model suitable for mission planning | Supports an empirical power-model baseline. It does not learn long-horizon energy through Bellman bootstrapping. |
| Stolaroff et al., **Energy Use and Life Cycle Greenhouse Gas Emissions of Drones for Commercial Package Delivery**, Nature Communications 2018 ([article](https://www.nature.com/articles/s41467-017-02411-5)) | Delivery-trip energy and range | Physical thrust/drag model calibrated with flight campaigns; explicitly varies payload, speed, battery mass, and wind | Computes loaded outbound plus unloaded return energy and applies a reserve factor | Shows that distance alone is insufficient when payload, wind, and flight regime change. It remains trajectory/segment integration rather than policy-conditioned value learning. |
| Dai et al., **Energy-Efficient UAV Communications: A Generalized Propulsion Energy Consumption Model**, IEEE WCL 2022 ([DOI](https://doi.org/10.1109/LWC.2022.3195787)) | Generalized rotary-wing propulsion energy | Adds thrust-to-weight effects caused by velocity, acceleration, and direction changes | Used inside trajectory and scheduling optimization | Directly supports including realized acceleration and direction changes in energy accounting. |
| Dai, Duo, and Yuan, **Energy-Efficient UAV Communications in the Presence of Wind: 3D Modeling and Trajectory Design**, IEEE TWC 2023 ([DOI](https://doi.org/10.1109/TWC.2023.3292290)) | Wind-aware 3-D propulsion energy | Generalized propulsion model with stochastic wind and 3-D force analysis | Joint trajectory/communication design under wind | Important robustness comparator: a learned model should be tested under wind shift rather than only homogeneous open-space dynamics. |
| Gong et al., **Modeling Power Consumptions for Multirotor UAVs**, IEEE TAES 2023 ([DOI](https://doi.org/10.1109/TAES.2023.3288846)) | Forward-flight, vertical-ascent, vertical-descent, and generic 3-D power | Theoretical derivation using multirotor/single-rotor relationships; validated using a DJI M210 | Predicts power across 3-D flight regimes and studies payload/rotor effects | Strong physical comparator for the current synthetic `TelemetryCostModel`; highlights missing vertical asymmetry, payload, and calibrated vehicle physics. |

### Implication

The strongest physics baseline is not Euclidean distance alone. It is:

```text
realized state/action trajectory
-> calibrated instantaneous power model
-> integrate power over actual transition duration
```

For JSEB, the trajectory must be the HOCBF-executed trajectory. Integrating power along nominal SAC actions would omit intervention, braking, detour, and boundary effects.

## 4. Data-driven instantaneous-power estimation

| Paper | Estimated object | Inputs and method | Output/use | Relation to current work |
|---|---|---|---|---|
| She, Lin, and Lang, **A Data-Driven Power Consumption Model for Electric UAVs**, ACC 2020 ([IEEE](https://ieeexplore.ieee.org/document/9147622)) | Instantaneous electrical power | Data-driven regression from UAV operating signals, avoiding a complete first-principles propulsion derivation | Power estimate for energy-aware control/planning | Closest category to replacing `TelemetryCostModel` with a learned physical-power layer; still not a long-horizon Energy Critic. |
| Gao et al., **Energy Model for UAV Communications: Experimental Validation and Model Generalization**, China Communications 2021 ([DOI](https://doi.org/10.23919/JCC.2021.07.020), [arXiv](https://arxiv.org/abs/2005.01305)) | Instantaneous power in straight and general 2-D flight | About 12,000 real power-speed samples; compares analytical curve fitting with a model-free DNN; extends with velocity, direction, and acceleration-dependent modeling | Validates/generalizes propulsion-power prediction | Demonstrates that compact DNN power fitting can match analytical curves on measured regimes. It does not establish long-horizon tail coverage or policy-transfer validity. |
| Choudhry et al., **CVaR-based Flight Energy Risk Assessment for Multirotor UAVs using a Deep Energy Model**, ICRA 2021 ([arXiv](https://arxiv.org/abs/2105.15189), [author PDF](https://www.ri.cmu.edu/app/uploads/2022/05/ICRA2021_choudhry.pdf)) | Instantaneous power and induced flight-energy distribution | A TCN consumes time-varying airspeed/body-velocity/vertical-speed/attitude history; static context is fused separately. Monte Carlo forward simulations propagate dynamics/environment uncertainty | Total flight-energy distribution and CVaR risk before flight | Closest high-quality uncertainty precedent. It predicts energy on proposed trajectories and propagates uncertainty by simulation; it does not train four recursive return quantiles as safety bounds. |
| Rodrigues et al., **In-flight Positional and Energy Use Data Set of a DJI Matrice 100 Quadcopter for Small Package Delivery**, Scientific Data 2021 ([article](https://www.nature.com/articles/s41597-021-00930-x)) | Dataset rather than estimator | 209 flights, approximately 10 h 45 min and 65 km; includes GPS/IMU, voltage/current, wind, commanded speed, payload, and altitude variation | Enables supervised power/energy model development and held-out path tests | Provides a model for what real validation must measure. The current simulator has synthetic energy units and cannot support real-UAV physical claims. |

### Implication

These papers learn or calibrate **power first**. Mission energy is then obtained by integration. This should remain a compulsory baseline against any TD Energy Critic:

```text
learned/calibrated P(x_t, u_t, context_t)
-> rollout current executed policy
-> sum P_t * delta_t
```

## 5. Direct mission-energy regression and recharge planning

| Paper | Estimated object | Inputs and method | Output/use | Relation to current work |
|---|---|---|---|---|
| Prasetia et al., **Mission-Based Energy Consumption Prediction of Multirotor UAV**, IEEE Access 2019 ([DOI](https://doi.org/10.1109/ACCESS.2019.2903644)) | Complete mission energy | Flight logs are segmented into movement types, including horizontal acceleration/deceleration; Elastic Net regression maps mission primitives to energy | Direct energy prediction for unseen surveillance-like mission patterns | Strong non-TD baseline when complete trajectories exist. It avoids recursive bootstrap drift but depends on a mission representation and supervised coverage. |
| Dorling et al., **Vehicle Routing Problems for Drone Delivery**, IEEE TSMC:S 2017 ([DOI](https://doi.org/10.1109/TSMC.2016.2582745)) | Route-leg energy inside delivery routing | Experimentally motivated payload/battery-mass-dependent energy model embedded in MIP and simulated annealing vehicle-routing formulations | Energy-feasible delivery tours with vehicle reuse/recharge | Shows the established separation between leg-energy modeling and high-level routing. It does not learn low-level collision-aware actions. |
| Alyassi et al., **Autonomous Recharging and Flight Mission Planning for Battery-Operated Autonomous Drones**, IEEE TASE 2022 ([DOI](https://doi.org/10.1109/TASE.2022.3175565), [arXiv](https://arxiv.org/abs/1703.10049)) | Route/leg energy and recharge-feasible mission plan | Machine-learning energy model includes real-world and meteorological factors; its predictions feed a multi-criteria asymmetric TSP with recharge decisions and online replanning | Time-efficient energy-feasible tour and charging schedule | Closest precedent for “learned energy estimate -> recharge decision.” The policy/trajectory planner consumes the model, but the energy model is not explicitly conditioned on a changing HOCBF-filtered policy. |
| Di Franco and Buttazzo, **Energy-Aware Coverage Path Planning of UAVs**, ICARSC 2015 ([DOI](https://doi.org/10.1109/ICARSC.2015.17)) | Coverage-path energy | Computes path energy from flight phases and path geometry | Selects coverage paths respecting battery limits | Planning baseline showing that path structure, not only endpoint distance, determines energy. |
| Cabreira et al., **Energy-Aware Spiral Coverage Path Planning for UAV Photogrammetric Applications**, IEEE RA-L 2018 ([DOI](https://doi.org/10.1109/LRA.2018.2854967)) | Coverage mission energy | Uses an energy-aware path construction rather than a learned value function | Energy-efficient feasible coverage trajectories | Relevant as a trajectory-structure baseline, not as an online estimator. |

### Implication

Supervised Monte Carlo return-to-go regression is not a weak baseline. Under a fixed deployed policy with successful complete trajectories, it directly learns labels

\[
E_t^{\mathrm{MC}}=\sum_{k=t}^{T_g-1} c_k
\]

without moving Bellman targets. Any scalar or distributional TD method must justify its extra complexity through online adaptation, censored/incomplete trajectories, data efficiency, or better transfer.

## 6. Distributional return learning: useful machinery, not UAV-specific novelty

| Paper | Core method | Relevance and limitation |
|---|---|---|
| Bellemare, Dabney, and Munos, **A Distributional Perspective on Reinforcement Learning**, ICML 2017 ([PMLR](https://proceedings.mlr.press/v70/bellemare17a.html)) | Learns a return distribution using a specified projected distributional Bellman operator | Establishes that the operator/projection is part of the algorithm. It does not validate arbitrary sparse quantile landmarks or provide calibrated safety bounds. |
| Dabney et al., **Distributional Reinforcement Learning with Quantile Regression**, AAAI 2018 ([paper](https://ojs.aaai.org/index.php/AAAI/article/view/11791)) | QR-DQN represents uniformly weighted quantile atoms and uses quantile regression | Appropriate standard baseline for quantile TD. Four hand-selected levels with custom masses are not automatically QR-DQN. |
| Dabney et al., **Implicit Quantile Networks for Distributional Reinforcement Learning**, ICML 2018 ([PMLR](https://proceedings.mlr.press/v80/dabney18a.html)) | Samples quantile fractions and learns the quantile function continuously | Offers richer tail modeling, but adds architecture and does not solve policy drift or calibration by itself. |

### Implication

A neural `Q95` is a learned conditional quantile, not a guaranteed upper bound. Safety language requires held-out coverage, underestimation rate, shift tests, and a stated calibration assumption.

## 7. Policy-aware value/model learning and two-timescale adaptation

These are not primarily UAV-energy papers. They are the closest methodological precedents for the proposed policy-aware Energy Critic.

| Paper | What is conditioned on | How it works | Relevance to JSEB |
|---|---|---|---|
| Harb et al., **Policy Evaluation Networks**, 2020 ([arXiv](https://arxiv.org/abs/2002.11833)) | A learned policy fingerprint | Learns to evaluate many policies and uses a differentiable compact representation of policy behavior | Direct precedent for policy fingerprints. Adding a fingerprint to an Energy Critic is therefore not independently novel. |
| Faccio, Kirsch, and Schmidhuber, **Parameter-Based Value Functions**, ICLR 2021 ([OpenReview](https://openreview.net/forum?id=tV6oBfuyLTQ), [arXiv](https://arxiv.org/abs/2006.09226)) | Policy parameters | A single state/action value function takes policy parameters as additional input and is trained using Monte Carlo or TD | Direct theorem-level precedent for policy-conditioned values. Feeding raw SAC parameters is likely impractical at current scale, but the semantics are established. |
| Faccio et al., **General Policy Evaluation and Improvement by Learning to Identify Few But Crucial States**, 2022 ([arXiv](https://arxiv.org/abs/2207.01566)) | Actions produced by a policy on learned probing states | Combines policy embeddings and parameter-based evaluation to represent policies through a small set of behaviorally informative probes | Suggests a feasible fingerprint: SAC actions on fixed probe observations rather than millions of raw weights. |
| Chen et al., **Policy-conditioned Environment Models are More Generalizable**, ICML 2024 ([PMLR](https://proceedings.mlr.press/v235/chen24g.html)) | Policy representation | Learns a meta-dynamics model that adapts predictions to the evaluation policy and reduces value gaps under behavior-policy shift | Strong precedent for the claim that the data-collection policy changes model validity. The paper concerns dynamics models, not UAV energy or HOCBF execution, but creates a serious novelty overlap. |
| Zhang et al., **Provably Convergent Two-Timescale Off-Policy Actor-Critic with Function Approximation**, ICML 2020 ([PMLR](https://proceedings.mlr.press/v119/zhang20s.html)) | Separate critic/actor stochastic-approximation timescales | Critics update on the faster timescale and the target policy changes more slowly; convergence is shown under the paper's linear-critic assumptions | Supports the update ordering “critic tracks current policy, actor moves slowly.” It does not prove convergence for the present deep quantile critic, HOCBF projection, or gamma-one SSP. |

## 8. Energy constraints and safety layers that are not energy estimators

| Paper | What it does | Why it is not an Energy Critic precedent |
|---|---|---|
| Notomista, Ruf, and Egerstedt, **Persistification of Robotic Tasks Using Control Barrier Functions**, IEEE RA-L 2018 ([DOI](https://doi.org/10.1109/LRA.2018.2789848)) | Encodes battery sufficiency and charging behavior as CBF constraints | Uses a specified energy/task model to construct a safety condition; it does not learn policy-conditioned return energy from trajectories. |
| Fouad, Varadharajan, and Beltrame, **Energy Sufficiency in Unknown Environments via Control Barrier Functions**, 2023 ([arXiv](https://arxiv.org/abs/2306.15115)) | Adds an energy-sufficiency CBF layer over navigation/planning | Addresses energy-safe control under assumptions about energy/distance information; not an estimator comparison or policy-aware TD method. |

These works make a generic “energy gate plus safety filter” contribution weak. The research contribution must concern the learned object, policy shift, safety-conditioned trajectory distribution, uncertainty, or experimentally demonstrated coupling failure.

## 9. Closest-work matrix for the current research question

| Candidate method | Instantaneous energy learned? | Long-horizon energy learned? | Policy-aware? | Safety-filter-aware? | Uncertainty/tail? | Charger decision? |
|---|---:|---:|---:|---:|---:|---:|
| Zeng/Gong/Dai analytical power integration | No | Integrated on supplied trajectory | No | Only if supplied trajectory contains it | No | No |
| Gao/She learned power + integration | Yes | Integrated on supplied trajectory | Indirectly through supplied trajectory | Indirectly | Usually point prediction | No |
| Prasetia mission regression | Indirect | Yes, supervised complete mission | Fixed data-generating policy only | Only if represented in training logs | No | No |
| Choudhry TCN + Monte Carlo + CVaR | Yes | Yes, simulated trajectory distribution | Uses proposed trajectory, not a learned policy fingerprint | Can include obstacle/wind effects through simulation | Yes | Risk assessment, not online commitment |
| Alyassi learned energy + ATSP recharge | Yes/route level | Yes, route-level | Planner consumes predictions | Not HOCBF-executed-policy conditioned | Limited | Yes |
| Generic distributional TD | No UAV power model required | Yes | Normally one target policy at a time | Only through data and state | Return quantiles | Not inherently |
| PBVF/PEN/PCM | Generic return/model | Yes/generic | Yes | Not specifically | Not inherently | No |
| Proposed policy-aware JSEB loop | `TelemetryCostModel` initially supplies synthetic realized cost | Goal-conditioned return energy | Must tag/condition on current policy | Must use HOCBF-executed actions and intervention context | Quantiles plus held-out calibration | High-level one-way commitment |

## 10. What prior work suggests implementing

### 10.1 Mandatory estimator baselines

1. **Distance times empirical energy per metre.** Tests whether the environment is distance dominated.
2. **Analytical/synthetic power integration.** Roll out the current executed policy and integrate `TelemetryCostModel` over actual substep duration.
3. **Supervised Monte Carlo return regression.** Uses complete goal trajectories and exact backward return-to-go labels.
4. **Scalar n-step TD.** Tests whether incremental bootstrapping adds value without relying on unstable sparse tail recursion.
5. **Standard quantile TD.** Use a standard quantile grid before claiming a custom four-landmark model is necessary.
6. **Monte Carlo/ensemble uncertainty propagation.** Closest to Choudhry et al.; separates power-model uncertainty from return-distribution approximation.

### 10.2 Policy-shift baselines

1. **Frozen-policy critic:** current control experiment.
2. **Naive mixed replay:** old and current policies mixed without policy identity; expected to expose stale semantics.
3. **Current-policy window:** only recent trajectories from policy version `k`.
4. **Version-tagged replay:** train separate/current heads or stratify by policy version without a learned fingerprint.
5. **Policy-fingerprint critic:** condition on actions produced at fixed probe states.
6. **Alternating adaptation:** update Energy Critic to a held-out gate, freeze it for a small actor block, then recollect and refit.

The policy fingerprint should be introduced only if current-policy windowing and alternating updates are insufficient. Otherwise it adds a known mechanism without demonstrated necessity.

### 10.3 Safety-coupling baselines

1. Nominal SAC trajectory energy.
2. HOCBF-executed trajectory energy.
3. HOCBF-executed energy with intervention context.
4. Standard minimum-deviation HOCBF action.
5. Energy-aware choice within the HOCBF-feasible action set.
6. Actor trained with shield consistency, evaluated by reduction in intervention rather than reward alone.

Collision constraints must remain feasibility constraints. Energy must not cancel collision risk through a weighted scalar reward.

## 11. Research gap and claim boundary

### Supported problem statement

The most defensible research question is:

> How should a goal-conditioned UAV energy-to-go estimator and navigation policy be updated when the executed trajectory is jointly induced by a changing SAC policy and a hard HOCBF safety projection, so that energy predictions remain valid while the learned actor progressively internalizes safe and energy-efficient actions?

### What would constitute evidence

The project needs to demonstrate all of the following:

1. changing the SAC/HOCBF-executed policy measurably degrades a frozen Energy Critic;
2. degradation is not explained only by ordinary state-distribution shift or insufficient data;
3. current-policy refitting or alternating updates repair the error;
4. a policy fingerprint improves over simpler policy-version/window baselines if it is retained;
5. the adapted actor reduces energy or intervention without reducing task completion or increasing collision/return failure;
6. held-out energy underestimation and quantile coverage remain acceptable after each policy update.

### Non-claims

- Policy-conditioned value functions are not new.
- Two-timescale actor--critic is not new.
- Learned UAV power models are not new.
- Learned energy plus recharge planning is not new.
- HOCBF plus an energy objective is not automatically a new safety theorem.
- A neural quantile is not a calibrated safety guarantee.

## 12. Recommended immediate decision

The current frozen-policy JSEB run should remain the control group. The next method should be the minimum-complexity alternating baseline:

```text
freeze policy pi_k
-> collect complete HOCBF-executed trajectories
-> fit/evaluate Energy Critic for pi_k
-> freeze critic snapshot
-> make a small KL-limited actor update
-> collect fresh trajectories
-> measure critic deterioration
-> refit or roll back
```

Only after this baseline is reproducibly better than the old continuously changing-policy Phase2B should the project add a policy fingerprint. This ordering follows the literature and prevents a known architectural component from being mistaken for the scientific contribution.
