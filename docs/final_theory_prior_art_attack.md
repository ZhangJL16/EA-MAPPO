# Final Prior-Art Attack: History-Adaptive Tubes, Search Reduction, and Recoverability

## Search Purpose

This search attempts to disprove novelty of the strongest remaining chain:

```text
causal history
-> set-membership uncertainty contraction
-> adaptive trajectory tube
-> smaller safe-search burden
-> exact trajectory verification
-> terminal recoverability
-> finite-horizon energy selection
```

Only primary paper pages, official proceedings, DOI pages, and author/arXiv
manuscripts are used. The search targets formula and guarantee structure, not
method names.

## Primary Closest Work

| Work | Primary mechanism and guarantee | Collision with current candidate | Residual difference |
|---|---|---|---|
| Didier, Wabersich, and Zeilinger, [Adaptive Model Predictive Safety Certification for Learning-based Control](https://arxiv.org/abs/2109.13033), 2021 | set-membership parameter refinement, robust tubes, terminal safe set, recursive safety, monotonically expanding feasible/safe sets | **Fatal** to “history contracts uncertainty and non-decreasingly improves safety” | our uncertainty concerns obstacle kinematics rather than plant parameters |
| Boggio, Novara, and Taragna, [Set Membership Based Nonlinear Model Predictive Control](https://doi.org/10.1016/j.ejcon.2023.100857), European Journal of Control 2023 | data-derived bounds on an NMPC law reduce search dimension and search-domain volume; the corresponding arXiv entry was withdrawn, so the published DOI is the authoritative source | strong overlap with “set membership reduces the online trajectory search region” | our search variables are UAV action primitives and safety uses an independent verifier; the current candidate never proved containment of its optimum in the reduced region |
| Gao et al., [Learning-based Rigid Tube Model Predictive Control](https://proceedings.mlr.press/v242/gao24a.html), 2024 | online learned disturbance sets update rigid tubes; statistical gaps support probabilistic recursive-feasibility analysis | **Fatal** to generic learned/adaptive tube novelty | our tube is deterministic under explicit jerk bounds |
| Aboudonia and Lygeros, [Adaptive Learning-based MPC for Uncertain Interconnected Systems](https://arxiv.org/abs/2404.16514), 2024 | set-membership uncertainty learning adapts tube, tightened constraints, controller, and terminal ingredients | **Fatal** to the complete adaptation chain | different uncertain object and distributed-system context |
| Dey and Bhasin, [Adaptive Tube MPC: Beyond a Common Quadratically Stabilizing Feedback Gain](https://arxiv.org/abs/2603.15912), 2026 | online parameter-set contraction updates tube cross-sections and terminal set with recursive feasibility and robust constraints | **Fatal** to “shrinking history set yields shrinking tubes and less conservative control” | their setting is uncertain LTI plant dynamics |
| Dixit et al., [Adaptive Conformal Prediction for Motion Planning among Dynamic Agents](https://proceedings.mlr.press/v211/dixit23a.html), 2023 | delayed online observations create adaptive multi-step uncertainty sets used by MPC around dynamic agents | **Fatal** to adaptive history-dependent trajectory uncertainty as a broad concept | probabilistic rather than deterministic bounded-error semantics |
| Zhou, Lindemann, and Sesia, [Conformalized Adaptive Forecasting of Heterogeneous Trajectories](https://proceedings.mlr.press/v235/zhou24l.html), 2024 | simultaneous pathwise bands for heterogeneous trajectories | **Fatal** to a generic trajectory-level calibrated band claim | our proposed tube is analytic and bounded-jerk |
| Contreras, Shorinwa, and Schwager, [SODA-MPC](https://proceedings.mlr.press/v283/contreras25a.html), 2025 | prediction confidence monitor switches from learned MPC to reachability fallback under OOD | **Fatal** to “confidence-gated fast mode plus conservative safe mode” | their detector is ensemble/conformal and their fallback is reachability-based |
| Chee et al., [Uncertainty Quantification and Robustification of Model-Based Controllers Using Conformal Prediction](https://proceedings.mlr.press/v242/chee24a.html), 2024 | modular online model-error bounds robustify model-based controllers with constraint guarantees | strong overlap with predictor-independent robustification | our deterministic physical error set avoids distributional coverage claims |
| Jaulin et al. / recent comparative study, [Set-membership techniques for state estimation, fault detection and collision avoidance](https://doi.org/10.1016/j.ejcon.2024.101161), 2025 | bounded-noise state sets and their collision-avoidance use | **Fatal** to set-membership collision estimation as a new object | our third-order exact LP projection is UAV-specific |
| Sakcak et al., [Sampling-based optimal kinodynamic planning with motion primitives](https://arxiv.org/abs/1809.02399), 2018 | motion-primitive gridding, probabilistic completeness/asymptotic optimality, and resolution-cost bounds | **Fatal** to generic primitive-cover and cost-suboptimality novelty | our receding-horizon cycle and telemetry energy differ |
| Safe Beyond the Horizon, [RSS 2025](https://www.roboticsproceedings.org/rss21/p112.html) | sampling MPC with terminal safety beyond the finite horizon | **Fatal** to terminal-safe sampling novelty | different terminal certificate construction |
| SC-MPPI, [Robust Sampling Based Model Predictive Control](https://roboticsproceedings.org/rss14/p42.html) and related barrier-state sampling work | robust sampling MPC and safety-aware rollout | fatal to generic safe-sampling architecture | our exact quartic verifier and energy telemetry are implementation differences |
| Adaptive MHE for time-varying parameters, [arXiv:2404.09566](https://arxiv.org/abs/2404.09566), 2024 | moving-window state/parameter estimation with adaptive regularization under changing observability | overlap with adaptive history length and changing regimes | no direct trajectory-safety verifier |
| Bhatnagar et al., [Improved Online Conformal Prediction via Strongly Adaptive Online Learning](https://proceedings.mlr.press/v202/bhatnagar23a.html), 2023 | interval-wise adaptation under arbitrary distribution change | overlap with fast uncertainty adaptation after regime changes | statistical coverage, not deterministic physical containment |

## Formula-Level Attack

### 1. History-set contraction

Our inclusion

\[
\mathcal X_t^{(L+1)}\subseteq\mathcal X_t^{(L)}
\]

is the defining behavior of a feasible set formed by intersecting additional
bounded-noise constraints. Adaptive MPSC and adaptive tube MPC use the same
uncertainty-set contraction to expand feasible/safe sets or reduce tightening.
The obstacle-state application does not change the theorem.

### 2. Contracted set to smaller tube

Propagating an updated set through bounded dynamics is standard tube control
and set-membership filtering. The exact term

\[
e_p+\tau e_v+\tfrac12\tau^2e_a+\tfrac16\tau^3\bar j
\]

is useful and physically interpretable, but it is an integration identity for
third-order kinematics.

### 3. Confidence-triggered robust mode

SODA-MPC already instantiates the same logical architecture: use a less
conservative predictive mode when confidence is accepted and switch to a
reachability fallback when it is not. Our deterministic residual test changes
the monitor, not the hybrid-safety principle.

### 4. Uncertainty contraction to search reduction

SM-NMPC explicitly uses set-membership bounds to reduce the dimension and
volume of the online NMPC search domain. Therefore the proposed connection is
not new even before adding safe sampling. For random candidate availability,

\[
P(\text{one safe sample})=1-(1-p_{safe})^N
\]

is elementary. It does not prove global coverage in a narrow corridor.

### 5. Primitive cover to energy suboptimality

Motion-primitive planning already studies resolution/computation tradeoffs,
resolution completeness, and resolution-cost bounds. The proposed
\(L_J\epsilon\) energy gap is the standard Lipschitz cover argument and is
weaker than established kinodynamic optimality results.

### 6. Terminal recoverability

Shifted candidate plus invariant terminal backup is standard robust MPC and is
already coupled with adaptive set membership in adaptive MPSC. No new terminal
set or backup policy has been derived for the multi-obstacle UAV.

## No-Free-Lunch Boundary

The one defensible negative statement under the declared third-order model is
that a smooth history cannot reduce deterministic uncertainty from an
unannounced admissible future jerk. Two systems can have identical histories
and opposite future jerk, forcing a radius of at least
\(\bar j\tau^3/6\). The analogous \(\bar a\tau^2/2\) statement requires a
separate second-order model that permits direct acceleration switching. The
jerk result is useful for rejecting unsafe confidence claims, but it is an
elementary indistinguishability argument rather than a new framework.

## Closest-Work Verdict by Candidate

| Candidate | Strongest closest-work attack | Survives? |
|---|---|---:|
| deterministic history-set contraction | adaptive MPSC / adaptive tube MPC | NO |
| adaptive history-conditioned tube | set-membership filtering + adaptive tube MPC | NO |
| confidence-triggered robust fallback | SODA-MPC / reachability fallback | NO |
| history reduces trajectory-search volume | SM-NMPC | NO |
| deterministic primitive cover | kinodynamic motion-primitive completeness | NO |
| cover-to-energy suboptimality | resolution-optimal planning and Lipschitz covers | NO |
| uncertainty-adaptive terminal recoverability | adaptive MPSC terminal-set theorem | NO |
| history-only future lower bound | standard worst-case indistinguishability | valid, insufficient as paper core |

## Opportunity Remaining After Attack

No positive theorem-level difference survives. A new contribution would require
at least one object not supplied by the cited families, such as a computable
multi-obstacle invariant/recoverable set with a genuinely new structure or a
strictly stronger finite-time sample-complexity theorem under realistic
nonconvex safety geometry. Neither object has been constructed here.

## Decision

**THEORY NOVELTY ATTACK: FAILED.** Fixed-model set membership should be retained
as a conditional estimation utility, and the counterexample should be retained
as a guard against unjustified history confidence. The proposed adaptive
feasibility gate is not a certificate and should not be presented as paper-core
theory.
