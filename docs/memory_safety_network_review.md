# Memory Safety Network Review

## Review question

Can a recurrent memory model turn UAV/LiDAR history into a physically meaningful obstacle-motion estimate and a valid uncertainty set that a physics-based safety filter can use?

## Current evidence in this repository

- The matched 5,000-trajectory benchmark uses 3,000/1,000/1,000 disjoint
  train/validation/test trajectories. The ego-compensated L16 MLP is the best
  point estimator (`joint MAE = 0.5303`), ahead of LSTM (`0.5895`), RNN
  (`0.5964`), GRU (`0.5982`), Physics-GRU (`0.6683`), IMM (`0.7772`), and the
  contractive physical-memory model (`0.8362`).
- At the 1 s horizon and nominal q95 calibration, ego L16 has coverage `0.961`
  with mean full-width L2 `8.3024`; GRU has `0.949/9.3619`, Physics-GRU
  `0.960/9.9987`, contractive memory `0.960/10.2082`, and IMM
  `0.967/9.6142`. Recurrent models therefore do not improve the
  coverage-width frontier over the simple ego fixed window.
- Sudden-direction adaptation within the observed 0.3 s window is `0.7297`
  for ego L16, `0.4595` for GRU, `0.3423` for Physics-GRU, `0.1892` for
  contractive memory, and `0.0270` for IMM. Crossing-like turns remain poor for
  every family. A recurrent hidden state did not solve abrupt adaptation.
- The deterministic 200k counterexample slice preserves analytic containment
  under declared bounds and fails it immediately under residual-bound
  violations. Reset containment is `1.0` only conditional on the independent
  base-set assumption; unconditional reset containment is `0.07777`.
- The unresolved object is therefore not prediction MAE. It is a
  feedback-valid uncertainty set under partial observation, dropout, motion
  change, and association uncertainty.

## Primary-source closest-work clusters

| Cluster | Primary source | What it already covers | Consequence for this project |
|---|---|---|---|
| Contractive recurrent models | Revay and Manchester, *Contracting Implicit Recurrent Neural Networks*, L4DC 2020, https://proceedings.mlr.press/v120/revay20a.html | Convexly parameterized contracting recurrent models and hidden-state stability | Hidden-state contraction is prior art and cannot be the paper claim. |
| Learned robust observers | Miao and Gatsis, *Learning Robust State Observers using Neural ODEs*, L4DC 2023, https://proceedings.mlr.press/v211/miao23a.html | Learned Luenberger/KKL observers and convergence-robustness analysis | A learned physics observer is not new by itself. |
| Kalman/RNN hybrid | Revach et al., *KalmanNet*, 2021, https://arxiv.org/abs/2107.10043 | Recurrent learning embedded in a model-based Kalman filtering flow | “Physics + GRU” is directly exposed to obvious-combination criticism. |
| Recurrent covariance propagation | Mortada et al., *Recursive KalmanNet*, 2025, https://arxiv.org/abs/2506.11639 | Recurrent state estimation with recursive covariance and uncertainty consistency | Learned nominal plus recursive uncertainty is not categorically new. |
| Measurement-robust CBF | Dean et al., *Guaranteeing Safety of Learned Perception Modules via Measurement-Robust CBFs*, CoRL 2020, https://proceedings.mlr.press/v155/dean21a.html | Perception-error bounds tightened into CBF conditions | Perception error to safe-control tightening is established. |
| Observer-based CBF/HOCBF | Wang and Xu, *Observer-based Control Barrier Functions for Safety Critical Systems*, ACC 2022, https://arxiv.org/abs/2110.00923 | Error-quantified observers combined with relative-degree-one and higher-order CBFs | Observer plus HOCBF is established. |
| Belief-space CBF | Vahs et al., *Belief Control Barrier Functions for Risk-aware Control*, 2023, https://arxiv.org/abs/2309.06499 | Mean/covariance belief dynamics directly used for risk-aware barriers | Covariance-aware safety is established. |
| Neural reachable-tube verification | Lin and Bansal, *Verification of neural reachable tubes via scenario optimization and conformal prediction*, L4DC 2024, https://proceedings.mlr.press/v242/lin24a.html | Probabilistic verification of learned reachable tubes | Conformalizing a recurrent tube is not enough for novelty. |
| State-dependent perception bounds | Waite et al., *State-Dependent Conformal Perception Bounds for Neuro-Symbolic Verification*, 2025, https://proceedings.mlr.press/v288/waite25a.html | State-dependent conformal perception errors in formal verification | Confidence-conditioned tube calibration is prior art in broad form. |
| Adaptive perception CBF | Yun and Azizan, *ATOM-CBF*, L4DC 2026, https://proceedings.mlr.press/v331/yun26a.html | Online adaptive epistemic error margins in CBF filters | Online uncertainty inflation under sensing shift is already active prior art. |
| Multi-object belief CBF | Han et al., *Risk-Aware Belief Control Barrier Functions over Random Finite Sets*, 2026, https://arxiv.org/abs/2607.15016 | Multi-object state uncertainty, particle beliefs, continuous and discrete safety updates | Tracking/association uncertainty cannot be ignored or claimed as unexplored. |
| Observer-controller CBF synthesis | Agrawal and Panagou, *Safe and Robust Observer-Controller Synthesis using Control Barrier Functions*, 2022, https://arxiv.org/abs/2211.14364 | Bounded-error/ISS observers coupled to estimate-feedback CBF-QPs, including a quadrotor example | The end-to-end observer-error-to-CBF-QP chain is directly covered; UAV application does not create a new theorem. |
| Guaranteed obstacle-set flow + CBF | Matias and Silvestre, *Safe Navigation under Uncertain Obstacle Dynamics using Control Barrier Functions and Constrained Convex Generators*, 2026, https://arxiv.org/abs/2601.07715 | Finite-horizon guaranteed obstacle estimation, sampled-interval set flow, CBF conversion and QP control for uncertain linear obstacle dynamics, including second-order strict-feedback agents | This is the closest theorem-level overlap and directly covers the proposed set-observer-to-sampled-safety architecture at a more general set representation. |
| Robust sampled-data high-order CBF | Oruganti, Naghizadeh, and Ahmed, *Robust Control Barrier Functions for Sampled-Data Systems*, 2023, https://arxiv.org/abs/2309.08050 | Bounded disturbance/measurement error, piecewise-constant control, sampled-data state evolution bounds, and relative-degree-one/higher safety constraints | The jerk-aware hold margin is a specialization, not a distinct sampled-data robust-HOCBF object. |
| Interval sampled-data CBF | Zhang, Walters, and Xu, *Control Barrier Function Meets Interval Analysis*, 2021, https://arxiv.org/abs/2110.00915 | Interval reachable overapproximation and sampled-data CBF conditions with measurement/actuation uncertainty, including higher relative degree | Interval uncertainty plus sampled-data CBF is established prior art. |

## Exact analytic object retained for falsification

The strongest auditable object is an **explicit kinematic interval observer with
an optional recurrent nominal jerk and an analytic orthotope error recursion**:

1. the explicit state is obstacle position, velocity, acceleration, interval radii, innovation score, missed-frame count, and reset status;
2. the recurrent network predicts only nominal jerk;
3. a componentwise interval recursion, not the network output, carries the certificate;
4. dropout expands the interval analytically;
5. inconsistent innovations or association ambiguity reset to an independently declared base set;
6. the orthotope support function enters a directional robust HOCBF;
7. uncertainty-set inclusion implies safe-action-set and QP-objective monotonicity.

## Current novelty assessment

The broad method is heavily covered by the sources above. The only precise
difference is the use of componentwise constant-jerk intervals and their exact
directional support inside a sampled-data second-order CBF. That is an
implementation specialization, not a new theoretical object. The closest 2026
CCG paper makes the overlap stronger: it already couples guaranteed
finite-horizon obstacle-set flow to sampled-data CBF-QP navigation. The recurrent
families neither beat the fixed-window estimator nor independently shrink the
deterministic residual bound, so the memory-to-smaller-certified-set premise is
not established. Hostile review scores novelty **9/30, below the 22/30 gate**.
Experimental success cannot repair this overlap.

## Non-claims

- Hidden-state contraction does not imply physical estimation-error contraction.
- A low validation MAE does not validate an uncertainty tube.
- A reset is sound only if its base set contains the true motion state.
- LiDAR range flow identifies radial motion, not arbitrary tangential velocity.
- Association ambiguity requires set union/inflation or fallback.
- A continuously enforced HOCBF theorem does not automatically prove a 20 Hz sampled implementation safe.
