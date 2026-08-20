# UAV Safety + Energy Joint Action Filter: Literature Review

## Scope and Source Policy

- Search date: 2026-08-19.
- Purpose: closest-work and novelty attack for a frozen-SAC deployment safety layer for a second-order 3D UAV.
- Sources: primary papers from PMLR, NeurIPS proceedings, IEEE/DOI pages, Robotics Proceedings, and arXiv author versions when no open proceedings page was available.
- Excluded from evidence: surveys as primary novelty evidence, secondary blog summaries, and venue-excluded sources.
- Screened primary papers retained below: **49**.
- Papers from 2020–2026: **41**.

## Search Queries

1. high-order / exponential control barrier function high relative degree;
2. sampled-data, zero-order-hold, input-delay, and robust CBF;
3. CBF safety filter, safe action projection, shielding, and safe RL;
4. multi-constraint, composite, nonsmooth, and log-sum-exp CBF;
5. energy sufficiency, battery persistification, charging, and energy-aware CBF;
6. UAV / quadrotor collision avoidance with bounded acceleration;
7. CPO, FOCOPS, CVPO, PID-Lagrangian, RCRL, Sauté RL, and action-intervention baselines;
8. LiDAR / vision / point-cloud perception uncertainty for CBFs.

## Primary Paper Inventory

### A. CBF, HOCBF, and multi-constraint foundations

1. Ames et al., **Control Barrier Function Based Quadratic Programs for Safety Critical Systems**, ACC 2014. Establishes CBF-QP minimal intervention. [DOI](https://doi.org/10.1109/ACC.2014.6856521)
2. Ames et al., **Control Barrier Function Based Quadratic Programs with Application to Adaptive Cruise Control**, IEEE TAC 2017. Formal zeroing-CBF/QP foundation with actuator-aware examples. [DOI](https://doi.org/10.1109/TAC.2016.2638961)
3. Nguyen and Sreenath, **Exponential Control Barrier Functions for Enforcing High Relative-Degree Safety-Critical Constraints**, ACC 2016. Relative-degree extension preceding general HOCBF. [DOI](https://doi.org/10.1109/ACC.2016.7524935)
4. Xiao and Belta, **Control Barrier Functions for Systems with High Relative Degree**, IEEE TAC 2022. General HOCBF recursion and feasibility analysis. [DOI](https://doi.org/10.1109/TAC.2021.3105491)
5. Glotfelter, Cortés, and Egerstedt, **Nonsmooth Barrier Functions with Applications to Multi-Robot Systems**, IEEE L-CSS 2017. Nonsmooth composition for multiple safety constraints. [DOI](https://doi.org/10.1109/LCSYS.2017.2710943)
6. Wang, Ames, and Egerstedt, **Safety Barrier Certificates for Collisions-Free Multirobot Systems**, IEEE TRO 2017. Pairwise barrier certificates and real-time quadratic programs. [DOI](https://doi.org/10.1109/TRO.2017.2659727)
7. Wang et al., **Multi-Constraint Safe Reinforcement Learning via Closed-form Solution for Log-Sum-Exp Approximation of Control Barrier Functions**, L4DC 2025. Direct overlap with aggregate CBF plus closed-form action correction. [PMLR](https://proceedings.mlr.press/v283/wang25c.html)
8. Manda, Chen, and Fazlyab, **Learning Performance-oriented Control Barrier Functions Under Complex Safety Constraints and Limited Actuation**, CoRL 2024/Proceedings 2025. Directly attacks performance and limited-actuation conservatism. [PMLR](https://proceedings.mlr.press/v270/manda25a.html)
9. Sun et al., **Learn With Imagination: Safe Set Guided State-wise Constrained Policy Optimization**, L4DC 2025. Safe-set-guided RL with state-wise safety. [PMLR](https://proceedings.mlr.press/v283/sun25a.html)
10. Tayal et al., **A Physics-Informed Machine Learning Framework for Safe and Optimal Control of Autonomous Systems**, ICML 2025. Joint performance and certificate learning using physics-informed structure. [PMLR](https://proceedings.mlr.press/v267/tayal25a.html)

### B. Sampled-data, delays, robustness, and layered models

11. Singletary, Chen, and Ames, **Control Barrier Functions for Sampled-Data Systems with Input Delays**, CDC 2020. Robust invariance under ZOH, state uncertainty, and input delays. [arXiv](https://arxiv.org/abs/2005.06418)
12. Bahati, Ong, and Ames, **Control Barrier Functions in Sampled-Data Systems**, IEEE L-CSS 2022. Formal sampled-data CBF conditions. [DOI](https://doi.org/10.1109/LCSYS.2021.3076127)
13. Schilliger et al., **Safety of Sampled-Data Systems with Control Barrier Functions via Approximate Discrete Time Models**, CDC 2022. Uses approximate discrete dynamics and quantified model error. [DOI](https://doi.org/10.1109/CDC51059.2022.9993226)
14. Oruganti et al., **Robust Control Barrier Functions for Sampled-Data Systems**, IEEE L-CSS 2024. Robust sampled-data margins rather than naïve continuous-time reuse. [DOI](https://doi.org/10.1109/LCSYS.2023.3346311)
15. Tan et al., **Zero-Order Control Barrier Functions for Sampled-Data Systems with State and Input Dependent Safety Constraints**, ACC 2025. Difference-based barrier conditions without repeated differentiation. [arXiv](https://arxiv.org/abs/2411.17079)
16. Compton, Cohen, and Ames, **Learning for Layered Safety-Critical Control with Predictive Control Barrier Functions**, L4DC 2025. Learned predictive robustness for reduced/full-model mismatch. [PMLR](https://proceedings.mlr.press/v283/compton25a.html)
17. Cheng, Zhao, and Hovakimyan, **Safe and Efficient Reinforcement Learning using Disturbance-Observer-Based Control Barrier Functions**, L4DC 2023. Robust CBF filter using pointwise disturbance estimation. [PMLR](https://proceedings.mlr.press/v211/cheng23a.html)
18. Brunke, Zhou, and Schoellig, **Barrier Bayesian Linear Regression: Online Learning of Control Barrier Conditions for Safety-Critical Control of Uncertain Systems**, L4DC 2022. Learns uncertainty in barrier conditions online. [PMLR](https://proceedings.mlr.press/v168/brunke22a.html)

### C. Perception-aware and learned safety filters

19. Taylor et al., **Learning for Safety-Critical Control with Control Barrier Functions**, L4DC 2020. Learns model error where it affects CBF safety. [PMLR](https://proceedings.mlr.press/v120/taylor20a.html)
20. Cheng et al., **Reinforcement Learning for Safety-Critical Control under Model Uncertainty, using Control Lyapunov Functions and Control Barrier Functions**, RSS 2020. RL plus CLF/CBF under uncertainty. [RSS](https://doi.org/10.15607/RSS.2020.XVI.088)
21. Emam et al., **Model-based Reinforcement Learning with Provable Safety Guarantees via Control Barrier Functions**, ICRA 2021. Model learning and CBF safety filtering. [DOI](https://doi.org/10.1109/ICRA48506.2021.9561253)
22. Robey et al., **Learning Control Barrier Functions from Expert Demonstrations**, CDC 2020. Learns certificates from expert safe/unsafe information. [arXiv](https://arxiv.org/abs/2004.03315)
23. Dawson et al., **Safe Nonlinear Control Using Robust Neural Lyapunov-Barrier Functions**, CoRL 2021/Proceedings 2022. Robust neural certificates with explicit model-error treatment. [PMLR](https://proceedings.mlr.press/v164/dawson22a.html)
24. Liu et al., **Safe Control using Vision-based Control Barrier Function (V-CBF)**, ICRA 2023. High-dimensional visual input to learned CBF. [DOI](https://doi.org/10.1109/ICRA48891.2023.10160805)
25. Dai et al., **Point Cloud-Based Control Barrier Function Regression for Safe and Efficient Vision-Based Control**, ICRA 2024. Directly relevant to point-cloud/LiDAR safety interfaces. [DOI](https://doi.org/10.1109/ICRA57147.2024.10610647)
26. Capone et al., **Learning Safe Control via On-the-Fly Bandit Exploration**, ICML 2025. Safe online learning with robust CBFs and information acquisition. [PMLR](https://proceedings.mlr.press/v267/capone25a.html)
27. Yun and Azizan, **ATOM-CBF: Adaptive Safe Perception-Based Control under Out-of-Distribution Measurements**, L4DC 2026. Adaptive perception-error margins for LiDAR/RGB under OOD shift. [PMLR](https://proceedings.mlr.press/v331/yun26a.html)
28. Li et al., **System-level safety guard: Safe tracking control through uncertain neural network dynamics models**, L4DC 2024. Layered tracking safety under learned model uncertainty. [PMLR](https://proceedings.mlr.press/v242/li24a.html)

### D. Action projection, shielding, and CMDP safe RL

29. Achiam et al., **Constrained Policy Optimization**, ICML 2017. Expected discounted cost constraint with near-constraint-satisfaction policy updates. [PMLR](https://proceedings.mlr.press/v70/achiam17a.html)
30. Zhang, Vuong, and Ross, **First Order Constrained Optimization in Policy Space**, NeurIPS 2020. First-order policy-space constrained optimization. [NeurIPS](https://proceedings.neurips.cc/paper/2020/hash/af5d5ef24881f3c3049a7b9bfe74d58b-Abstract.html)
31. Stooke, Achiam, and Abbeel, **Responsive Safety in Reinforcement Learning by PID Lagrangian Methods**, ICML 2020. Damped dual updates for expected-cost constraints. [PMLR](https://proceedings.mlr.press/v119/stooke20a.html)
32. Liu et al., **Constrained Variational Policy Optimization for Safe Reinforcement Learning**, ICML 2022. Variational constrained policy optimization. [PMLR](https://proceedings.mlr.press/v162/liu22b.html)
33. Yu et al., **Reachability Constrained Reinforcement Learning**, ICML 2022. State-wise feasible-set safety via learned reachability value. [PMLR](https://proceedings.mlr.press/v162/yu22d.html)
34. Xu, Liang, and Lan, **CRPO: A New Approach for Safe Reinforcement Learning with Convergence Guarantee**, ICML 2021. Reward/cost objective switching without explicit dual variables. [PMLR](https://proceedings.mlr.press/v139/xu21a.html)
35. Chow et al., **Safe Policy Learning for Continuous Control**, CoRL 2020/Proceedings 2021. Lyapunov-induced action or policy projection. [PMLR](https://proceedings.mlr.press/v155/chow21a.html)
36. Wagener, Boots, and Cheng, **Safe Reinforcement Learning Using Advantage-Based Intervention**, ICML 2021. Learned intervention policy for chance-constrained safety. [PMLR](https://proceedings.mlr.press/v139/wagener21a.html)
37. Sootla et al., **Sauté RL: Almost Surely Safe Reinforcement Learning Using State Augmentation**, ICML 2022. State-augmented budget semantics. [PMLR](https://proceedings.mlr.press/v162/sootla22a.html)
38. Thananjeyan et al., **Recovery RL: Safe Reinforcement Learning with Learned Recovery Zones**, IEEE RAL 2021. Recovery policy/intervention architecture. [arXiv](https://arxiv.org/abs/2010.15920)
39. Pfrommer et al., **Safe Reinforcement Learning with Chance-constrained Model Predictive Control**, L4DC 2022. MPC safety guide plus policy gradient. [PMLR](https://proceedings.mlr.press/v168/pfrommer22a.html)
40. Wachi and Sui, **Safe Reinforcement Learning in Constrained Markov Decision Processes**, ICML 2020. CMDP safe exploration under structured assumptions. [PMLR](https://proceedings.mlr.press/v119/wachi20a.html)
41. Dawood et al., **Constraint-Aware Reinforcement Learning via Adaptive Action Scaling**, L4DC 2026. Lightweight action scaling for constrained off-policy RL. [PMLR](https://proceedings.mlr.press/v331/dawood26a.html)
42. Suttle et al., **Sampling-based Safe Reinforcement Learning for Nonlinear Dynamical Systems**, AISTATS 2024. Hard-constraint sampling approach designed to retain convergence properties that projection layers can disrupt. [PMLR](https://proceedings.mlr.press/v238/suttle24a.html)

### E. Energy sufficiency, battery persistence, and long-duration autonomy

43. Notomista, Ruf, and Egerstedt, **Persistification of Robotic Tasks Using Control Barrier Functions**, IEEE RAL 2018. Directly encodes battery sufficiency as a forward-invariant set while minimizing nominal-control deviation. [DOI](https://doi.org/10.1109/LRA.2018.2789848)
44. Notomista and Egerstedt, **Persistification of Robotic Tasks**, IEEE TCST 2021. General persistent-task optimization under battery constraints. [DOI](https://doi.org/10.1109/TCST.2020.2978913)
45. Fouad, Varadharajan, and Beltrame, **Energy Sufficiency in Unknown Environments via Control Barrier Functions**, 2023. Generic energy-sufficiency CBF layer over a planner. [arXiv](https://arxiv.org/abs/2306.15115)
46. Dan et al., **Persistent Object Search and Surveillance Control With Safety Certificates for Drone Networks Based on Control Barrier Functions**, Frontiers in Robotics and AI 2021. Combines collision, charging, and persistent-task CBF specifications for drones. [primary article](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2021.740460/full)
47. Michel, Saveriano, and Lee, **A Novel Safety-Aware Energy Tank Formulation Based on Control Barrier Functions**, IEEE RAL 2024. CBF/HOCBF constraints for robot energy and power, but in passivity/energy-tank semantics. [author PDF](https://iris.unitn.it/retrieve/65851528-21fe-49c5-8577-bb150f776aed/A_Novel_Safety-Aware_Energy_Tank_Formulation_Based_on_Control_Barrier_Functions.pdf)
48. Cai et al., **Energy-Aware, Collision-Free Information Gathering for Heterogeneous Robot Teams**, IEEE TRO 2023. Energy-aware planning with exponential CBF collision constraints. [author PDF](https://existentialrobotics.org/ref/Cai_NonmonotoneInfoGathering_TRO23.pdf)
49. Notomista et al., **Robot Ecology: Constraint-Based Control Design for Long Duration Autonomy**, Annual Reviews in Control 2018. Long-duration autonomy and battery-aware constraint-based control. [DOI](https://doi.org/10.1016/j.arcontrol.2018.09.006)

## Closest-Work Findings

### Finding 1: standard HOCBF-QP is established

The second-order spherical obstacle derivation and minimum-deviation QP are standard consequences of exponential/HOCBF theory. They are the correct baseline, not a novelty claim.

### Finding 2: aggregate closed-form CBF is directly covered

Wang et al. 2025 directly addresses multi-constraint CBF aggregation with log-sum-exp and a closed-form correction for safe RL. A generic “aggregate CBF for 1024 LiDAR constraints” claim is therefore high-overlap. A publishable gap would require a distinct theorem or empirically necessary structure, such as sampled-data 3D perception error with explicit computational guarantees.

### Finding 3: sampled-data correction is mandatory and populated

Singletary et al., Bahati et al., Schilliger et al., Oruganti et al., and Tan et al. show that continuous-time constraints cannot simply be checked at 20 Hz. Any margin must be derived from a specific sampled-data theorem or bounded residual evolution.

### Finding 4: energy viability CBF is directly covered

Notomista et al. 2018 and Fouad et al. 2023 already use CBFs to ensure enough energy remains for charging/mission continuation. Dan et al. 2021 combines collision and energy-persistence certificates for drones. Consequently, Candidate C (`battery - reserve - return_energy`) is **not novel in generic form**.

### Finding 5: energy-aware projection is useful engineering, not a supported new method

The exact combination of a frozen 3D UAV SAC, hard second-order collision HOCBF, synthetic propulsion quadratic cost, learned goal-conditioned energy-to-go gradient, and dense LiDAR compute constraints was not found as a single identical package. That absence is not sufficient novelty evidence. Existing work separately covers CBF projection, energy-aware planning, persistent battery CBFs, point-cloud CBFs, and aggregate CBF computation.

The completed held-out ablation further narrows the claim. A normalized instantaneous propulsion term reduced trajectory energy by 3.56% relative to sampled-data HOCBF at matched zero observed collisions, but the learned energy-to-go gradient did not establish an additional energy reduction: its fixed-scale 95% bootstrap interval crossed zero, while the unit-direction version increased mean energy. The surviving contribution is therefore an implementation and measurement result about an established sampled-data HOCBF variant, not a new generic safety formula.

## Recommended Baselines

1. **CPO**: canonical trust-region CMDP baseline.
2. **FOCOPS**: first-order expected-cost constrained policy baseline.
3. **PID-Lagrangian** or **CVPO**: practical stable constrained-learning baseline.
4. **Standard HOCBF-QP**: primary pointwise-safety baseline.
5. **Log-sum-exp aggregate CBF**: compute-efficiency baseline.
6. **One-step reachable supporting-halfspace projection**: weaker but cheap action-projection baseline.

CMDP baselines must receive the same compact obstacle information and collision-cost definition. Expected CMDP cost satisfaction must not be described as pointwise forward invariance.

## Literature-Grounded Novelty Status

**NOVELTY STATUS: NOT SUPPORTED.**

- Standard collision HOCBF-QP: existing method.
- Multi-obstacle log-sum-exp aggregation: directly covered.
- Generic battery-returnability CBF: directly covered.
- Instantaneous acceleration-energy objective: mathematically valid and empirically useful in the controlled prototype, but an obvious convex secondary cost rather than a supported conceptual contribution.
- Energy-to-go gradient inside a hard 3D sampled-data HOCBF layer: local gradients passed finite-difference checks, but controlled ablations did not establish incremental trajectory-energy benefit. It is rejected as the primary mechanism.
