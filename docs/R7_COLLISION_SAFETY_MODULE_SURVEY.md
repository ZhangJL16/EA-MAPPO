# R7 Collision-Safety Module Survey and Selection

Date: 2026-09-01  
Scope: collision safety for learned UAV navigation; collision safety is reused
infrastructure, not the paper's claimed core contribution.

## Executive decision

Use a two-layer design:

1. **CPPO-PID** remains the policy-optimization layer. It constrains an
   episodic scalar correction cost and reduces the tendency of the neural
   policy to rely on the shield.
2. **Robust sampled-data HOCBF-QP** is the always-on execution layer during
   both training and deployment. It minimally projects the PPO action onto the
   pointwise admissible action set at every 0.05 s physics step.

This is the best fit for the current project. The UAV is already modeled as a
double integrator, the policy directly emits three acceleration components,
actuator bounds are known, and the environment already contains a tested
HOCBF-QP implementation. A predictive safety filter or Hamilton--Jacobi shield
would add a second planning/learning problem; a learned recovery critic would
provide weaker hard-safety evidence; CMDP optimization alone cannot establish
zero collision at every step.

The recommendation is deliberately not presented as a universally best safe-RL
algorithm. It is the strongest mature option under this project's model,
runtime, and evidence constraints.

## The key distinction: expected cost is not hard collision safety

A CMDP optimizes

\[
\max_\theta J_R(\pi_\theta)
\quad\text{s.t.}\quad
J_C(\pi_\theta)=
\mathbb E_{\pi_\theta}\!\left[\sum_{t=0}^{T-1}\gamma_C^t c_t\right]
\le d.
\]

CPO, FOCOPS, CUP, PPO-Lagrangian, and CPPO-PID differ in how they update the
policy or multiplier, but this constraint is distributional/episodic. A policy
can meet it while still producing a rare collision. This makes these methods
appropriate for learning a low-intervention policy, but insufficient by
themselves for the project's empirical requirement of zero obstacle-collision
steps.

A safety filter instead enforces a statewise admissibility condition on each
executed action. The 2024 safety-filter review unifies CBF, predictive, and
reachability filters under this runtime-assurance role. The SafePO/OmniSafe
benchmarks support CPPO-PID as a stable first-order CMDP baseline, but do not
turn it into a categorical collision guarantee.

## Families screened

| Family | Strength | Limitation for this project | Decision |
|---|---|---|---|
| CPPO-PID / PPO-Lagrangian / CPO / FOCOPS / CUP | Directly controls expected episode cost; standard safe-RL baselines | No per-step zero-collision guarantee | Keep CPPO-PID as the learning layer |
| CBF/HOCBF action projection | Fast, minimally invasive, pointwise constraints, natural for acceleration control | Guarantee is conditional on model, sensing, feasibility, and digital implementation | Selected execution layer |
| Predictive safety filter / MPC shield | Multi-step recoverability and recursive feasibility | Requires a terminal invariant set, horizon solver, and substantially more CPU | Reserve for a later real-system extension |
| Hamilton--Jacobi reachability / ISAACS | Strong robust reachability semantics | Six-dimensional UAV state plus local obstacle representation makes exact HJ expensive; learned approximations become another research project | Not selected now |
| Recovery RL / learned safety critic | Can operate from high-dimensional observations and unknown dynamics | Learned risk prediction is approximate and needs an additional dataset, critic, threshold, and recovery policy | Not selected for hard collision safety |
| Differentiable CBF or state-dependent safe-action sampling | Better coupling between policy learning and hard action constraints | Changes the policy distribution/optimization substantially and adds implementation risk | Useful ablation, not the baseline |
| DOB-CBF / measurement-robust CBF | Handles bounded dynamics or perception error | Needs calibrated disturbance/measurement bounds not yet available in the nominal simulator | Required before a real-world safety claim |

## Selected mathematical module

The local translational model is

\[
\dot p=v,\qquad \dot v=u,\qquad u\in\mathcal U,
\]

where \(\mathcal U\) contains the horizontal and vertical acceleration limits.
For a LiDAR surface point \(q_i\) and inflated clearance \(d_i\), use

\[
h_i(p)=\lVert p-q_i\rVert_2^2-d_i^2.
\]

Because collision clearance has relative degree two with respect to
acceleration, define

\[
\psi_{i,1}=\dot h_i+k_1h_i.
\]

The continuous-time exponential HOCBF inequality is linear in \(u\):

\[
2(p-q_i)^\top u \ge
-\left[
2\lVert v\rVert_2^2
+2(k_1+k_2)(p-q_i)^\top v
+k_1k_2h_i
\right].
\]

The controller is digital and holds an action between samples. Therefore use a
sampled-data tightening \(\Delta_i(\delta)>0\), derived from the hold time and
bounded relative acceleration:

\[
2(p-q_i)^\top u \ge b_i(p,v)+\Delta_i(\delta).
\]

At every physics step, solve

\[
u_{\mathrm{safe}}
=\arg\min_{u\in\mathcal U}
\frac12\lVert u-u_{\mathrm{nom}}\rVert_2^2
\quad\text{s.t.}\quad
A_{\mathrm{HOCBF}}(p,v)u\ge b_{\mathrm{HOCBF}}(p,v,\delta).
\]

All currently perceived constraints should remain in the QP. If constraint
selection later becomes necessary, the project's controlled ranking study
supports nominal HOCBF slack or time-to-collision over distance-only selection.

## How PPO must interact with the filter

For an on-policy method, the correct training tuple is

\[
(s_t, u_t^{\mathrm{nom}}, r(s_t,u_t^{\mathrm{safe}}), s_{t+1}),
\quad
s_{t+1}\sim P(\cdot\mid s_t,u_t^{\mathrm{safe}}).
\]

The **nominal** stochastic PPO action must be stored for the probability ratio;
the **safe** action must drive the simulator. Replacing the stored action by the
projected action makes the on-policy likelihood ratio incorrect. The current R7
rollout follows the nominal-action convention.

The 2024 quadrotor study found that filtering during training and penalizing
the squared correction worked better than penalizing only violation events.
Accordingly, replace the current binary intervention cost by

\[
c_t^{\mathrm{corr}}
=\lVert u_t^{\mathrm{nom}}-u_t^{\mathrm{safe}}\rVert_2^2.
\]

CPPO-PID constrains the episode sum of this correction cost. Actual collision
and boundary contact remain fail-closed diagnostics and formal evaluation
gates, not ordinary events that the optimizer is allowed to trade for reward.
A collision surcharge may remain for fault visibility, but it cannot substitute
for the zero-collision gate.

## Project-specific evidence

The controlled local artifact
`artifacts/uav_safety_filter_1000_controlled_20260819_022302/summary.json`
compared 1,000 method--scenario rollouts. It is prototype evidence rather than
the current R7 formal protocol, but it is highly relevant to module selection:

| Method | Success | Collision-rollout rate | Collision-step rate | Mean path ratio | Mean filter time |
|---|---:|---:|---:|---:|---:|
| Ordinary second-order HOCBF | 1.000 | 0.184 | 0.0558 | 1.0337 | 1.137 ms |
| Robust sampled-data HOCBF | 1.000 | 0.000 | 0.0000 | 1.0388 | 1.005 ms |
| Sampled-data energy-gradient HOCBF | 1.000 | 0.000 | 0.0000 | 1.0386 | 1.670 ms |

The plain sampled-data HOCBF is preferred: it achieved the same zero-collision
result and similar path ratio without coupling collision infrastructure to the
paper's energy model. The evidence also identified a concrete defect in the
pre-survey R7 configuration: `hocbf_sampled_data_robust` was hard-coded to
`False`; the post-survey trainer now defaults it to `True`.

## Required implementation changes before the next formal run

1. Expose and enable `hocbf_sampled_data_robust=True` in R7 training and
   deterministic evaluation.
2. Use squared normalized action correction as the CPPO-PID scalar cost; log
   collision, boundary contact, QP infeasibility, emergency braking, and minimum
   barrier margins separately.
3. Retain the nominal PPO action/latent variable in the rollout buffer and use
   only the projected action for state transition and physical reward.
4. Keep the filter enabled during both training and deployment. Do not train
   shielded and evaluate unshielded.
5. Require resets to satisfy obstacle clearance and initial HOCBF feasibility.
6. Keep all LiDAR constraints for the present 1 ms-scale solver. Do not use
   distance-only top-K pruning.
7. Treat world boundaries as analytic half-space barriers in addition to LiDAR
   obstacle points; this is a physical safety specification, not a navigation
   rule.
8. Inflate point-cloud clearance by body radius, perception/discretization
   uncertainty, and the sampled-data margin. A real-world claim additionally
   requires calibrated measurement and disturbance bounds.
9. Abort promotion if any deterministic evaluation has collision, unsafe
   fallback, non-finite QP output, or missed hard real-time deadline.

## Evidence ladder

The module should be promoted in four bounded stages:

1. **Unit invariants:** exact HOCBF inequality, actuator bounds, inter-sample
   margin monotonicity, and nominal=executed when already safe.
2. **Adversarial safety stress:** high closing speed, narrow passage, two-sided
   squeeze, boundary approach, maximum acceleration, and sensor-range entry.
3. **Short policy gate:** real goal arrivals, zero deterministic collision,
   zero unsafe fallback, and decreasing squared correction cost.
4. **Formal navigation gate:** the immutable 500-task protocol: overall success
   at least 0.98, every distance bucket at least 0.95, mean path ratio at most
   1.10, boundary-contact step rate below 0.01, and zero obstacle-collision
   steps. Report filter intervention and infeasibility in addition to the gate.

The formal zero-collision result is an empirical result over the declared task
population. A mathematical forward-invariance statement must remain conditional
on safe initialization, perception coverage/error bounds, model/disturbance
bounds, QP feasibility, and meeting the sample/compute deadline.

## Primary sources and maintained references

- Hsu, Hu & Fisac, *The Safety Filter: A Unified View of Safety-Critical
  Control in Autonomous Systems*, Annual Review of Control, Robotics, and
  Autonomous Systems, 2024. <https://doi.org/10.1146/annurev-control-071723-102940>
- Xiao & Belta, *High-Order Control Barrier Functions*, IEEE TAC, 2022.
  <https://doi.org/10.1109/TAC.2021.3105491>
- Taylor et al., *Control Barrier Functions in Sampled-Data Systems*, IEEE
  Control Systems Letters. <https://arxiv.org/abs/2103.03677>
- Cheng et al., *End-to-End Safe Reinforcement Learning through Barrier
  Functions*, AAAI, 2019. <https://doi.org/10.1609/aaai.v33i01.33013387>
- Cheng, Zhao & Hovakimyan, *Safe and Efficient Reinforcement Learning Using
  Disturbance-Observer-Based Control Barrier Functions*, L4DC, 2023.
  <https://proceedings.mlr.press/v211/cheng23a.html>
- Dean et al., *Guaranteeing Safety of Learned Perception Modules via
  Measurement-Robust Control Barrier Functions*, CoRL, 2020.
  <https://proceedings.mlr.press/v155/dean21a.html>
- Wabersich & Zeilinger, *A Predictive Safety Filter for Learning-Based Control
  of Constrained Nonlinear Dynamical Systems*, Automatica, 2021.
  <https://doi.org/10.1016/j.automatica.2021.109597>
- Pizarro Bejarano, Brunke & Schoellig, *Safety Filtering While Training*, 2024.
  <https://arxiv.org/abs/2410.11671>
- Suttle et al., *Sampling-based Safe Reinforcement Learning for Nonlinear
  Dynamical Systems*, AISTATS, 2024.
  <https://proceedings.mlr.press/v238/suttle24a.html>
- Achiam et al., *Constrained Policy Optimization*, ICML, 2017.
  <https://proceedings.mlr.press/v70/achiam17a.html>
- Stooke, Achiam & Abbeel, *Responsive Safety in Reinforcement Learning by PID
  Lagrangian Methods*, ICML, 2020.
  <https://proceedings.mlr.press/v119/stooke20a.html>
- Yang et al., *Constrained Update Projection Approach to Safe Policy
  Optimization*, NeurIPS, 2022.
  <https://proceedings.neurips.cc/paper_files/paper/2022/hash/3ba7560b4c3e66d760fbdd472cf4a5a9-Abstract-Conference.html>
- Ji et al., *Safety-Gymnasium*, NeurIPS Datasets and Benchmarks, 2023.
  <https://papers.neurips.cc/paper_files/paper/2023/hash/3c557a3d6a48cc99444f85e924c66753-Abstract-Datasets_and_Benchmarks.html>
- Ji et al., *OmniSafe: An Infrastructure for Accelerating Safe Reinforcement
  Learning Research*, JMLR, 2024. <https://jmlr.org/papers/v25/23-0681.html>
- Thananjeyan et al., *Recovery RL: Safe Reinforcement Learning with Learned
  Recovery Zones*, IEEE RA-L, 2021. <https://arxiv.org/abs/2010.15920>
- Hsu, Nguyen & Fisac, *ISAACS: Iterative Soft Adversarial Actor-Critic for
  Safety*, L4DC, 2023. <https://proceedings.mlr.press/v211/hsu23a.html>
