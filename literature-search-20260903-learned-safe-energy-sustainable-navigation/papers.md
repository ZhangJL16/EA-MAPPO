# Literature Search: Learned Safe and Energy-Sustainable Navigation

Date: 2026-09-03  
Search purpose: determine whether a neural policy that jointly learns collision
avoidance, task motion, finite-battery return, and repeated recharging has a
defensible theory gap.  
Target venue/family: ICLR/ICML/NeurIPS plus L4DC, CAV, robotics, stochastic
control, and operations research.  
Source-quality policy: primary proceedings, publisher pages, or author arXiv
records; MDPI and untraceable aggregators excluded.

## Summary

- Closest-work clusters: minimum-cost reach-avoid RL; probability/quantile
  constrained RL; reachability safety; consumption MDPs with reload states;
  energy-sufficiency filters; task-and-charge planning.
- Opportunity map: the broad claim is **covered central claim**.  A fixed-budget
  joint finite-battery query is also equivalent to reach-avoid on an augmented
  state.  The remaining **theory/analysis gap** is a finite-sample certificate
  for quantitative repeated recharge under continuous unknown dynamics and
  function approximation.
- Strongest baselines: RC-PPO, RAPCPO, QCPO, SDAC, RESPO/RCRL, Sauté RL,
  Back-to-Base, and consumption-MDP synthesis.
- Novelty risks: RC-PPO already handles deterministic minimum-cost reach-avoid;
  RAPCPO handles stochastic probabilistic reach-avoid plus expected cost;
  consumption MDPs already formalize batteries, reload states, and repeated
  almost-sure behavior.
- Recommended next action: do not claim novelty for battery augmentation,
  return switching, safe energy minimization, or the fixed-budget CDF identity.
  Retain the extended-real distribution as an implementation object, then seek
  a nontrivial finite-sample Bellman-error-to-lifecycle-risk result and test the
  raw policy across repeated recharge cycles.

## Paper Table

Scores are 1--5 and assess each source, not the viability of our idea.

| # | Title | Year | Venue/source | Link | Type | Insight | Completeness | Numeric evidence | Overall | Notes |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| 1 | Stochastic Minimum-Cost Reach-Avoid Reinforcement Learning | 2026 | ICML 2026 / arXiv record | [paper](https://arxiv.org/abs/2605.11975) | theory/proof + method | 5 | 4 | 4 | Risk | RAPCPO enforces probabilistic reach-avoid in stochastic dynamics while minimizing expected cost. It is the nearest current result and invalidates a generic stochastic-safe-energy claim. |
| 2 | Solving Minimum-Cost Reach Avoid using Reinforcement Learning | 2024 | NeurIPS | [paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/3750e99b522bd36a099d2e8b9f0550c7-Abstract-Conference.html) | theory/proof + method | 5 | 5 | 4 | Risk | RC-PPO uses augmented dynamics and HJ reachability to minimize cumulative cost under reach-avoid constraints; theory assumes deterministic dynamics. |
| 3 | Qualitative Controller Synthesis for Consumption Markov Decision Processes | 2020 | CAV | [paper](https://link.springer.com/chapter/10.1007/978-3-030-53291-8_22) | theory/proof | 5 | 5 | 3 | Risk | Already models finite capacity, non-negative consumption, atomic reload states, and almost-sure repeated Büchi objectives in finite known MDPs. |
| 4 | Quantile Constrained Reinforcement Learning | 2022 | NeurIPS | [paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/2a07348a6a7b2c208ab5cb1ee0e78ab5-Abstract-Conference.html) | theory/proof + method | 5 | 4 | 4 | A | QCPO constrains a quantile of cumulative cost, exactly targeting outage probability rather than expected cost. |
| 5 | Trust Region-Based Safe Distributional Reinforcement Learning for Multiple Constraints | 2023 | NeurIPS | [paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/3f20f2b0315c72201e23512fdbd1ee91-Abstract-Conference.html) | pure method | 4 | 4 | 5 | A | SDAC treats collision, energy, and balance as multiple risk-averse constraints; it does not make charging reachability one joint random object. |
| 6 | Back to Base: Towards Hands-Off Learning via Safe Resets with Reach-Avoid Safety Filters | 2025 | L4DC | [paper](https://proceedings.mlr.press/v283/begzadic25a.html) | pure method | 4 | 4 | 3 | Risk | A learned reach-avoid value provides an external filter that avoids unsafe states and returns to a target/charging station by a deadline. |
| 7 | Energy Sufficiency in Unknown Environments via Control Barrier Functions | 2023 | arXiv / robotics-control preprint | [paper](https://arxiv.org/abs/2306.15115) | pure method | 4 | 4 | 4 | A | Provides a CBF layer over a path planner with energy-sufficiency guarantees and real-robot evidence; relies on an external planner/reference path. |
| 8 | Sauté RL: Almost Surely Safe Reinforcement Learning Using State Augmentation | 2022 | ICML | [paper](https://proceedings.mlr.press/v162/sootla22a.html) | theory/proof + method | 4 | 4 | 4 | A | Establishes safety-budget state augmentation and a Bellman equation; battery-in-state alone is therefore not novel. |
| 9 | Reachability Constrained Reinforcement Learning | 2022 | ICML | [paper](https://proceedings.mlr.press/v162/yu22d.html) | theory/proof + method | 5 | 4 | 4 | A | Learns the largest feasible set with a reachability safety value rather than imposing discounted expected-cost safety. |
| 10 | Iterative Reachability Estimation for Safe Reinforcement Learning | 2023 | NeurIPS | [paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/dca63f2650fe9e88956c1b68440b8ee9-Abstract-Conference.html) | theory/proof + method | 5 | 4 | 4 | A | RESPO handles stochastic feasible/infeasible regions and persistent state-wise safety, but not a finite-battery recharge distribution. |
| 11 | Reinforcement Learning with Almost Sure Constraints | 2022 | L4DC | [paper](https://proceedings.mlr.press/v168/castellano22a.html) | theory/proof | 5 | 4 | 3 | A | Derives the minimal safe budget as the smallest fixed point of a Bellman-like operator and shows why stationary policies may be insufficient. |
| 12 | Density Constrained Reinforcement Learning | 2021 | ICML | [paper](https://proceedings.mlr.press/v139/qin21a.html) | theory/proof + method | 4 | 4 | 4 | B | Constrains state occupancy density; includes an EV example with remaining energy and charging stations. It is not sample-path energy sufficiency. |
| 13 | Recovery RL: Safe Reinforcement Learning With Learned Recovery Zones | 2021 | IEEE RA-L | [paper](https://ieeexplore.ieee.org/document/9392290/) | pure method | 4 | 4 | 4 | A | Learns distinct task and recovery policies from constraint data; useful baseline for learned switching but lacks a recharge-tail certificate. |
| 14 | Simultaneous Task and Energy Planning Using Deep Reinforcement Learning | 2022 | Information Sciences | [paper](https://www.sciencedirect.com/science/article/pii/S0020025522005977) | pure method | 3 | 4 | 4 | B | End-to-end neural combinatorial planning covers repeated stationary/mobile/solar charging; it is high-level routing rather than continuous collision-safe control. |
| 15 | Dual Formulation for Chance Constrained Stochastic Shortest Path with Application to Autonomous Vehicle Behavior Planning | 2023 | arXiv | [paper](https://arxiv.org/abs/2302.13115) | theory/proof + method | 4 | 4 | 4 | A | Exact finite-state chance-constrained SSP and resource augmentation show that joint path-risk constraints are not new outside deep RL. |

## Clusters

### 1. Minimum-cost reach-avoid

- Representative papers: RC-PPO and RAPCPO.
- What this cluster already solves: reaching a goal, avoiding unsafe states,
  and minimizing accumulated cost; RAPCPO covers stochastic dynamics and a
  prescribed reach-avoid probability.
- Remaining gap: their primary cost objective is expectation/minimization.  The
  finite battery event `energy used before safe charger arrival <= remaining
  charge` is a joint tail event, not just a low expected cost.
- Rescue route: learn the distribution of a stopped resource variable with a
  failure atom, and optimize task return subject to its calibrated CDF.

### 2. Distributional and budgeted safety

- Representative papers: QCRL, SDAC, Sauté RL, and almost-sure constraints.
- What this cluster already solves: budget state augmentation, cumulative-cost
  quantiles, multiple risk-averse constraints, and minimal safe budgets.
- Remaining gap: these works do not appear, from the audited statements, to
  identify collision/deadline failure with the mass at infinity of a
  resource-to-*recharge* first-passage law.
- Rescue route: use their estimators/optimization machinery, but claim only the
  new stochastic object and the resulting certificate if a full-text theorem
  audit confirms separation.

### 3. Reachability and recovery

- Representative papers: RCRL, RESPO, Recovery RL, and Back-to-Base.
- What this cluster already solves: learned feasible sets, recovery regions,
  task/recovery decomposition, and deadline-aware return filters.
- Remaining gap: a learned feasibility value usually records reach/avoid, not
  the whole energy-needed distribution under finite charge.
- Rescue route: treat collision and missed deadline as failure mass in the same
  resource law; let a neural gate learn task/return behavior rather than use a
  fixed SOC threshold.

### 4. Replenishable-resource formal methods

- Representative papers: consumption MDP synthesis and chance-constrained SSP.
- What this cluster already solves: battery capacity, reload states, resource
  safety, repeated temporal objectives, and finite-state stochastic planning.
- Remaining gap: scalable, model-free continuous-control learning from LiDAR
  with statistical calibration and raw-policy evaluation.
- Rescue route: borrow the correct regenerative semantics and theorem targets;
  do not claim the underlying battery/reload model as new.

### 5. Robotics energy layers and high-level scheduling

- Representative papers: energy-sufficiency CBF and STEP.
- What this cluster already solves: planner-side energy guarantees and learned
  task/charging schedules.
- Remaining gap: these either add a filter to a planner or plan discrete tours;
  they do not establish that an end-to-end continuous navigation policy has
  learned a calibrated energy-feasible return set.

## Opportunity Map

| Cluster | Status | Open gap | Possible direction | Evidence needed | Risk |
| --- | --- | --- | --- | --- | --- |
| Minimum-cost reach-avoid | covered central claim | Tail of the energy-to-safe-recharge law | Extended-real resource-to-recharge distribution | Full theorem comparison with RC-PPO/RAPCPO | Very high |
| Distributional safe RL | crowded but open | Joint first-passage failure/energy law | CDF feasibility critic with lower confidence calibration | Uniform or finite-sample calibration evidence | High |
| Recovery and reachability | crowded but open | Battery-dependent learned task/return policy | Budget-conditioned mixture/gating policy | Raw-policy gains over Recovery RL/Back-to-Base | High |
| Consumption MDPs | covered central claim in finite known models | Continuous unknown dynamics and function approximation | Model-free neural approximation with explicit non-asymptotic error | Continuous benchmark plus calibration theorem | High |
| Energy CBF/planning | deployment/system gap | Policy-internal learning without permanent planner dependence | CBF only for safe data collection; evaluate raw policy separately | Raw vs filtered deployment study | Medium |
| Lifecycle sustainability | theory/analysis gap | Quantitative continuous-control guarantee under approximation error | Finite-sample Bellman-subsolution bound plus repeated-cycle composition | Exact abstractions, shift-aware calibration, and long multi-cycle tests | High |

## Benchmark And Dataset Candidates

| Name | Link | Task | Metrics | Baselines | Fit | Risks |
| --- | --- | --- | --- | --- | --- | --- |
| Current R3 continuous LiDAR environment | local project | Task navigation plus return-to-charge | task success, collision, boundary contact, energy outage, charger arrival, path ratio | R3 SAC, fixed SOC/distance/oracle switching | Highest implementation fit | Custom benchmark needs exact protocol publication |
| RC-PPO MuJoCo suite | [project](https://oswinso.xyz/rcppo/) | Minimum-cost reach-avoid | reach rate and cumulative cost | RC-PPO, RESPO, CPPO, PPO, SAC | Strong theorem baseline | Deterministic dynamics; not rechargeable |
| Safety-Gym/MuJoCo suites used by SDAC/RESPO | respective paper pages above | Multi-constraint continuous control | reward and constraint violations | SDAC, RESPO, CPO family | Tests generic safe RL | Weak charger/lifecycle semantics |
| Consumption-MDP scenarios | [CAV paper](https://link.springer.com/chapter/10.1007/978-3-030-53291-8_22) | Finite-state resource/reload synthesis | qualitative safety/Büchi feasibility and runtime | exact synthesis | Semantic oracle for small abstractions | State/action discretization mismatch |

## Citation And Positioning Cautions

- Claims that need direct citation: battery augmentation, quantile/outage
  equivalence, reachability feasible sets, minimum-cost reach-avoid, reload-state
  MDPs, and energy-sufficiency CBF guarantees.
- Papers that most weaken novelty: RAPCPO, RC-PPO, consumption MDP synthesis,
  QCRL, Back-to-Base.
- Papers that mainly supply machinery: SDAC, Sauté RL, RESPO, RCRL, Recovery RL.
- “The policy is safe” is invalid if only the filtered execution is tested.
  Report raw-policy and filtered-policy results separately.
- A fixed per-cycle failure probability does not imply infinite-horizon
  sustainability; the lifecycle claim needs zero risk or a summable risk
  schedule.
