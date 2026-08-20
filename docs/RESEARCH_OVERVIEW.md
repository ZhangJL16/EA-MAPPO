# Current Research Overview

Last updated: 2026-08-20

## Purpose

This document is the entry point for the research currently preserved in this
repository. It separates implemented mechanisms, controlled empirical evidence,
conditional mathematical results, rejected novelty claims, and future work.
Negative results are retained rather than rewritten as successful theory.

Raw training artifacts are intentionally excluded from Git when they are large.
The documents below record their paths, hashes, protocols, and numerical
summaries so that claims can be traced back to the corresponding local artifact.

## Executive Status

| Research line | Strongest supported result | Unsupported or rejected claim | Current disposition |
|---|---|---|---|
| Navigation and energy-risk estimation | Frozen goal-conditioned navigation plus Monte Carlo energy-to-go and predefined-group conformal calibration form a usable obstacle-free component | Arbitrary state-conditional coverage, shift-free guarantees, and physical-UAV safety | **KEEP AS COMPONENT** |
| Sampled-data collision and energy action filter | Obstacle-wise sampled-data HOCBF with exact geometry, bounded-input feasibility diagnostics, and telemetry-aware secondary optimization is a rigorous systems baseline | Generic theoretical novelty of HOCBF projection, energy viability, or energy-aware filtering | **ENGINEERING / SYSTEMS RESULT** |
| History-conditioned safe trajectory sampling | Exact clipped-physics rollout, inter-sample verification, coarse safe rejection, and lexicographic safety-progress-energy selection work as a test platform | Novelty of history-guided safe MPPI, terminal backup, adaptive tubes, or learned proposals | **ENGINEERING ONLY; THEORY REJECTED** |
| Generic recurrent safety memory | Observer/tube/HOCBF interfaces and counterexample suites clarify what memory may and may not certify | Recurrent memory itself creates a certificate or a new safe-set theorem | **REJECT AS PAPER CORE** |
| Control-grounded structured memory | Trusted historical constraints can contract a certified set; real Route/Energy labels were collected and tested on 100,122 controlled transitions | FOGM-specific contraction, Route/Energy control benefit, and theorem-level separation from set membership or active-set selection | **FINAL REJECTION; ENGINEERING ONLY** |

## Research Line 1: Energy Estimation and Risk

### Question

Can an obstacle-free persistent UAV estimate goal/return energy conservatively
enough to support battery-aware task-to-charger decisions without placing SOC or
charger identity in the navigation policy?

### What was implemented and tested

- A frozen goal-conditioned SAC navigation policy and physical telemetry-energy
  integration.
- Distance, Monte Carlo, scalar TD, and quantile/distributional energy baselines.
- Trajectory-disjoint training, calibration, and evaluation protocols.
- Group-aware conformal upper bounds for Goal and Mission energy.
- Battery-cycle and Phase-2 switching audits, including correction of a prior
  artificial battery-reset metric interpretation.

### Current result

The final v5 energy-risk construction passed its declared fresh-test marginal and
predefined finite-group stress gates. The supported statement remains conditional
on the declared exchangeability and group definitions. It is not arbitrary
conditional validity. The paired Phase-2 study also found that the fixed reserve
dominated switching: historical switches were 100% reserve-attributed and the
reserve was about 31 times the uncertainty margin. Narrower uncertainty therefore
did not establish a throughput gain under the fixed reserve.

### Authoritative records

- [`energy_risk_root_cause_and_solution_v5.md`](energy_risk_root_cause_and_solution_v5.md)
- [`energy_risk_final_method_review.md`](energy_risk_final_method_review.md)
- [`formal_mc_run_20260817_metric_correction.md`](formal_mc_run_20260817_metric_correction.md)
- [`energy_estimation_literature_review.md`](energy_estimation_literature_review.md)
- [`energy_estimation_literature_matrix.csv`](energy_estimation_literature_matrix.csv)

## Research Line 2: Sampled-Data UAV Safety and Energy Filtering

### Question

Can a frozen 7D navigation SAC be wrapped by a separate LiDAR-driven safety
filter that enforces short-timescale collision constraints while energy remains a
parallel, non-compensating consideration?

### What was implemented and tested

- Sphere, cylinder, box, and raw-LiDAR geometry models.
- Relative-degree-two HOCBF constraints for 3D double-integrator dynamics.
- Sample-and-hold residual strengthening, stale sensing, delay, dropout, and
  actuator/velocity feasibility checks.
- Closed-form single-halfspace projection, polyhedral QP projection, top-k and
  aggregate alternatives.
- Exact ZOH and constant-jerk inter-sample clearance checks.
- Telemetry-aware energy objective, energy-to-go gradient diagnostics, and
  explicit fallback/infeasibility accounting.

### Controlled result

The obstacle-wise sampled-data method was retained because aggregate and one-step
alternatives failed safety in hard geometries. The instantaneous telemetry-energy
secondary objective is a defensible engineering choice. The energy-to-go gradient
did not establish incremental trajectory-energy value, and a separate energy
viability constraint was rejected as overlapping with existing viability/CBF
work. Thirty-four unsafe fallback steps in the selected controlled run remain an
explicit systems limitation rather than being hidden as certified behavior.

### Authoritative records

- [`uav_safety_energy_experiment_report.md`](uav_safety_energy_experiment_report.md)
- [`uav_safety_energy_theory_derivation.md`](uav_safety_energy_theory_derivation.md)
- [`uav_safety_energy_completion_audit.md`](uav_safety_energy_completion_audit.md)
- [`uav_safety_energy_novelty_audit.md`](uav_safety_energy_novelty_audit.md)
- [`uav_safety_energy_literature_review.md`](uav_safety_energy_literature_review.md)
- [`uav_safety_energy_literature_matrix.csv`](uav_safety_energy_literature_matrix.csv)

## Research Line 3: History-Conditioned Safe Trajectories

### Question

Does temporal obstacle history support a new physics-guided trajectory proposal
and certification method beyond safe MPPI, reachability-guided sampling, terminal
safe MPC, and learned trajectory generators?

### What survived

- Exact clipped UAV physics and realized-acceleration rollout.
- Continuous inter-sample sphere and boundary certification.
- Safe upper-clearance rejection and lexicographic safety-progress-energy
  candidate selection.
- A modular history buffer, physical motion features, proposal families, and
  counterexample suite useful for systems benchmarking.

### What failed

- The proposed adaptive history tube did not obtain an independently valid
  true-state-containment theorem.
- Longer fixed history was not uniformly better under abrupt changes.
- Random safe sampling matched learned-proposal recall in the bounded diagnostic,
  so there was no measured bottleneck that justified training a proposal model.
- Closest work already covers safe MPPI, reachability guards, terminal-safe
  sampling, learned trajectory proposals, and GPU sampling MPC.

### Authoritative records

- [`history_safe_trajectory_theory.md`](history_safe_trajectory_theory.md)
- [`history_safe_trajectory_experiments.md`](history_safe_trajectory_experiments.md)
- [`history_safe_trajectory_counterexamples.md`](history_safe_trajectory_counterexamples.md)
- [`history_safe_trajectory_literature_review.md`](history_safe_trajectory_literature_review.md)
- [`history_safe_trajectory_completion_audit.md`](history_safe_trajectory_completion_audit.md)
- [`final_theory_target_mode_decision.md`](final_theory_target_mode_decision.md)
- [`final_theory_prior_art_attack.md`](final_theory_prior_art_attack.md)

## Research Line 4: Recurrent Safety Memory

### Question

Can a recurrent observer or structured memory turn history into a smaller
uncertainty object that yields a genuinely new certified-control consequence?

### Result

The observer, uncertainty tube, and robust-HOCBF interfaces are useful research
infrastructure. The proof audits found that empirical residual boxes and neural
predictions are not certificates. Valid safety statements require independently
trusted containment premises. The strongest safe-set and intervention
monotonicity statements are standard robust-control consequences and do not make
recurrence proof-relevant.

### Authoritative records

- [`memory_observer_theory.md`](memory_observer_theory.md)
- [`memory_uncertainty_tube_derivation.md`](memory_uncertainty_tube_derivation.md)
- [`memory_hocbf_theory.md`](memory_hocbf_theory.md)
- [`memory_proof_audit.md`](memory_proof_audit.md)
- [`memory_closed_loop_experiments.md`](memory_closed_loop_experiments.md)
- [`memory_result_to_claim.md`](memory_result_to_claim.md)
- [`memory_theory_novelty_attack.md`](memory_theory_novelty_attack.md)
- [`ccfa-review-reports/memory_route_round1/REVIEW_REPORT.md`](../ccfa-review-reports/memory_route_round1/REVIEW_REPORT.md)

## Research Line 5: Control-Grounded Structured Memory

### Final experiment

The final route explicitly tested both previously missing bridges:

1. trusted historical measurements to certified set contraction;
2. real executed Route/Energy labels to downstream prediction and control value.

The accepted dataset contains 100,122 transitions from 180 trajectories, split
by complete trajectory into 108 train, 36 validation, and 36 test trajectories.
The accepted slice contains zero collision, infeasible, and unsafe-fallback
steps. Three learned-model seeds were used.

### Certified contraction result

All evaluated methods had 100% true-state containment on 500 held-out snapshots.
All-history set membership had geometric volume ratio 0.000101 and mean width
56.344 relative to the current-only set. Generic GRU, FOGM, and oracle selection
were numerically identical at volume ratio 0.016264 and width 100.179. FOGM was
only about 1.15 ms faster than all-history while producing a much looser set.

### Route and energy result

- Route accuracy: current safety state 0.8059, generic GRU 0.7563, FOGM 0.6721.
- Safe Energy-to-Go MAE: distance/velocity baseline 0.26743, generic GRU 0.50444,
  FOGM 0.68027.
- FOGM was worse than generic GRU on Route and Energy-to-Go in all three seeds.
- FOGM improved 40-step safety-energy overhead prediction in all three seeds,
  but this auxiliary gain did not improve path, intervention, or total energy.
- In matched closed loop, explicit side hysteresis had fewer interventions and
  switches and lower energy than FOGM; no route memory had the best path ratio.

### Final decision

The learned memory proposes constraints; independently trusted set-membership
premises and the verifier create the certificate. No theorem-level difference
from set-membership estimation, constraint pruning, learned active-set selection,
or proposal-verifier architectures survived hostile review. The final novelty
score was 11/30 and the Area Chair score was 4/10 Reject. The long experiment was
therefore not launched.

### Authoritative records

- [`grounded_memory_real_diagnostic.md`](grounded_memory_real_diagnostic.md)
- [`grounded_memory_verified_contraction.md`](grounded_memory_verified_contraction.md)
- [`grounded_memory_proof_audit.md`](grounded_memory_proof_audit.md)
- [`grounded_memory_final_rejection.md`](grounded_memory_final_rejection.md)
- [`grounded_memory_literature_review.md`](grounded_memory_literature_review.md)
- [`grounded_memory_literature_matrix.csv`](grounded_memory_literature_matrix.csv)
- [`ccfa-review-reports/grounded_memory_final/REVIEW.md`](../ccfa-review-reports/grounded_memory_final/REVIEW.md)

## Literature Corpus

The complete paper-level records are preserved in the source matrices and
reviews rather than duplicated here. Counts are not additive because several
papers appear in more than one route.

| Corpus | Primary-paper coverage | Main topics |
|---|---:|---|
| [`energy_estimation_literature_matrix.csv`](energy_estimation_literature_matrix.csv) | 34 papers | UAV propulsion and telemetry energy, mission energy, return/charging autonomy, TD and distributional RL, calibration |
| [`uav_safety_energy_literature_matrix.csv`](uav_safety_energy_literature_matrix.csv) | 49 papers | CBF/HOCBF, sampled-data safety, perception uncertainty, shielding, CMDP, persistent energy safety |
| [`history_safe_trajectory_literature_review.md`](history_safe_trajectory_literature_review.md) | 70 papers in its combined audit | Safe MPPI, reachability, terminal safety, trajectory generation, world models, GPU sampling |
| [`grounded_memory_literature_matrix.csv`](grounded_memory_literature_matrix.csv) | 76 papers | Grounded communication, modular/object memory, observers, belief safety, set membership, active constraints, proposal-verifier systems |

### Representative closest papers actually used in decisions

- Zeng, Xu, and Zhang, [Energy Minimization for Wireless Communication With Rotary-Wing UAV](https://doi.org/10.1109/TWC.2019.2902559).
- Choudhry et al., [CVaR-based Flight Energy Risk Assessment for Multirotor UAVs using a Deep Energy Model](https://doi.org/10.1109/ICRA48506.2021.9561658).
- Bellemare, Dabney, and Munos, [A Distributional Perspective on Reinforcement Learning](https://proceedings.mlr.press/v70/bellemare17a.html).
- Dabney et al., [Distributional Reinforcement Learning with Quantile Regression](https://ojs.aaai.org/index.php/AAAI/article/view/11791).
- Lei et al., [Distribution-Free Predictive Inference for Regression](https://doi.org/10.1080/01621459.2017.1307116).
- Gibbs and Candes, [Conformal Inference for Online Prediction with Arbitrary Distribution Shifts](https://arxiv.org/abs/2208.08401).
- Ames et al., [Control Barrier Functions: Theory and Applications](https://arxiv.org/abs/1903.11199).
- Xiao and Belta, [High-Order Control Barrier Functions](https://arxiv.org/abs/1903.04706).
- Dean et al., [Guaranteeing Safety of Learned Perception Modules via Measurement-Robust Control Barrier Functions](https://proceedings.mlr.press/v155/dean21a.html).
- Williams, Aldrich, and Theodorou, [Model Predictive Path Integral Control](https://doi.org/10.2514/1.G001921).
- Gandhi et al., [Safe Importance Sampling in Model Predictive Path Integral Control](https://arxiv.org/abs/2303.03441).
- Yin et al., [Safe Beyond the Horizon](https://www.roboticsproceedings.org/rss21/p112.html).
- Chi et al., [Diffusion Policy](https://doi.org/10.15607/RSS.2023.XIX.026).
- Hafner et al., [Learning Latent Dynamics for Planning from Pixels](https://proceedings.mlr.press/v97/hafner19a.html).
- Li et al., [Learning the Uncertainty Sets of Linear Control Systems via Set Membership](https://proceedings.mlr.press/v235/li24ci.html).
- Tang et al., [Uncertainty Quantification of Set-Membership Estimation in Control and Perception](https://proceedings.mlr.press/v242/tang24a.html).
- Misra et al., [Learning for Constrained Optimization: Identifying Optimal Active Constraint Sets](https://arxiv.org/abs/1802.09639).
- Jackson et al., [Certified Control: An Architecture for Verifiable Safety of Autonomous Vehicles](https://arxiv.org/abs/2104.06178).
- Didier, Wabersich, and Zeilinger, [Adaptive Model Predictive Safety Certification for Learning-based Control](https://arxiv.org/abs/2109.13033).
- Dixit et al., [Adaptive Conformal Prediction for Motion Planning among Dynamic Agents](https://proceedings.mlr.press/v211/dixit23a.html).

## Current Claim Boundary

The repository supports:

- synthetic-software and controlled-simulation evidence;
- exact geometric and optimization statements under explicit models;
- conditional set containment when sensor, motion, and association bounds are
  independently valid;
- finite-sample conformal statements only under their declared calibration and
  exchangeability assumptions;
- negative conclusions when learned mechanisms fail simpler matched baselines.

It does not support:

- real-UAV physical safety;
- arbitrary-environment reachability or recursive feasibility;
- arbitrary conditional conformal coverage or distribution-shift-free coverage;
- deep neural TD global convergence;
- certification created by a neural memory/proposal module;
- a general theoretical novelty claim for the rejected memory or trajectory
  architectures.

## Recommended Next Direction

Stop generic memory/theory rebranding. Preserve the verified set-membership,
sampled-data HOCBF, explicit route hysteresis, telemetry-energy model, and
calibrated Monte Carlo estimator as fixed components. The next paper should be a
robotics/autonomous-systems study driven by a reproducible physical failure mode,
with matched simple baselines and explicit safety, efficiency, latency, and
calibration evidence.
