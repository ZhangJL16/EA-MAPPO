# Decision-relevant evidence cards

## 1. Constrained Policy Optimization

- Source: Achiam et al., ICML 2017, [PMLR](https://proceedings.mlr.press/v70/achiam17a.html).
- Supported result: trust-region policy updates obtain a bound on expected
  constraint change and empirically approach per-update constraint satisfaction.
- Limitation: on-policy second-order optimization and an expected discounted
  cost do not give state-wise or pathwise collision invariance.
- Transfer decision: not suitable as the first R3 repair because it replaces
  SAC and does not target the observed Bellman-semantic error.
- Confidence: direct.

## 2. PID Lagrangian safe RL

- Source: Stooke et al., ICML 2020, [PMLR](https://proceedings.mlr.press/v119/stooke20a.html).
- Supported result: integral-only dual control can oscillate and overshoot;
  proportional and derivative feedback improve constraint responsiveness.
- Limitation: still needs cost-value estimation and optimizes an average CMDP
  budget. Threshold and PID tuning remain.
- Transfer decision: credible SAC baseline, but it adds a cost critic and solves
  a broader problem than binary collision avoidance.
- Confidence: direct.

## 3. Early-Terminated MDP

- Source: Sun et al., [arXiv:2107.04200](https://arxiv.org/abs/2107.04200).
- Supported result: for the paper's finite deterministic CMDP construction, an
  absorbing unsafe state with a sufficiently low termination reward can share
  the constrained optimum; early termination can discard invalid rollout tails.
- Limitation: hard termination creates limited state visitation and a sparse
  recovery signal; the equivalence assumptions do not prove deep continuous
  control convergence.
- Transfer decision: use first-contact absorbing termination as a simple
  control, not the selected method.
- Confidence: direct.

## 4. Constraints as Terminations (CaT)

- Source: Chane-Sane et al., IROS 2024,
  [DOI](https://doi.org/10.1109/IROS58592.2024.10802334),
  [author manuscript](https://arxiv.org/abs/2403.18765).
- Supported result: converts continuous constraint violation into a stochastic
  loss of future reward, producing dense feedback with no additional critic or
  policy. The paper distinguishes learning termination from environment reset.
- Limitation: its chance-constraint motivation is not a hard-safety theorem;
  reward lower-boundedness and violation scaling matter.
- Transfer decision: strongest mechanism fit, provided the UAV constraint is a
  single physically meaningful sampled-data collision violation rather than a
  list of hand-written style rules.
- Confidence: direct for the primitive; transfer is inferred.

## 5. SoloParkour

- Source: Chane-Sane et al., CoRL 2024,
  [PMLR](https://proceedings.mlr.press/v270/chane-sane25a.html).
- Supported result: implements CaT in off-policy DDPG-family learning and
  recomputes termination probabilities from violations stored in replay.
- Limitation: the full visual algorithm also uses privileged replay and a REDQ
  ensemble; those components are task-specific and unnecessary for R3.
- Transfer decision: evidence that CaT is not intrinsically PPO-only. Transfer
  only the replay-compatible termination primitive, not the architecture.
- Confidence: direct.

## 6. Recovery RL

- Source: Thananjeyan et al., RA-L 2021,
  [author manuscript](https://arxiv.org/abs/2010.15920).
- Supported result: a safety critic estimates future violation probability and
  a distinct recovery policy takes over in risky states.
- Limitation: two policies plus a safety critic; empirical rather than an
  unconditional certificate. It requires a useful recovery behavior.
- Transfer decision: reject. The project's measured R3 return-now policy is
  safe on only 275/310 anchors, so it cannot serve as the assumed reliable
  backup and the architecture violates the minimality requirement.
- Confidence: direct plus local evidence.

## 7. Advantage-Based Intervention (SAILR)

- Source: Wagener et al., ICML 2021,
  [PMLR](https://proceedings.mlr.press/v139/wagener21a.html).
- Supported result: intervenes according to a cost advantage relative to a
  backup/intervention policy while retaining an off-the-shelf RL learner.
- Limitation: needs a valid intervention rule and learned cost action-value; it
  adds another decision layer over R3's HOCBF.
- Transfer decision: mathematically relevant but redundant for the first patch.
- Confidence: direct.

## 8. Reach-Avoid RL

- Source: Hsu et al., RSS 2021,
  [proceedings](https://roboticsproceedings.org/rss17/p077.html).
- Supported result: derives a discounted reach-avoid recursion and uses the
  learned value as an untrusted oracle inside supervision.
- Limitation: introduces a reach-avoid value and supervisor; approximation alone
  is not a certificate.
- Transfer decision: best later framework for joint goal/collision/return
  theory, but unnecessarily large for fixing R3's immediate collision objective.
- Confidence: direct.

## 9. Reachability-Constrained RL

- Source: Yu et al., ICML 2022,
  [PMLR](https://proceedings.mlr.press/v162/yu22d.html).
- Supported result: replaces expected cumulative cost by a persistent feasible
  set defined through worst-over-time reachability values.
- Limitation: requires a learned reachability object and additional constrained
  optimization; deep approximation is not deterministic invariance.
- Transfer decision: correct semantics for a future state-wise method, but not
  the smallest causal repair.
- Confidence: direct.

## 10. Lyapunov safe RL and safe action projection

- Sources: Chow et al., NeurIPS 2018,
  [proceedings](https://papers.neurips.cc/paper_files/paper/2018/hash/4fe5149039b52765bde64beb9f674940-Abstract.html),
  and Chow et al., CoRL 2020,
  [PMLR](https://proceedings.mlr.press/v155/chow21a.html).
- Supported result: a safe baseline can induce local Lyapunov inequalities and
  feasible policy/action updates; continuous-control implementations can project
  actions.
- Limitation: theory relies on a feasible baseline/Lyapunov object; action
  projection is already R3's execution mechanism and can hide unsafe nominal
  behavior.
- Transfer decision: reject for the first patch.
- Confidence: direct.

## 11. Sampling-based safe RL

- Source: Suttle et al., AISTATS 2024,
  [PMLR](https://proceedings.mlr.press/v238/suttle24a.html).
- Supported result: safe actions can be generated/sampled under nonlinear-system
  assumptions, including a quadrotor obstacle-avoidance study.
- Limitation: changes the action-generation interface and adds sampling cost;
  its assumptions must be re-established for the LiDAR/HOCBF implementation.
- Transfer decision: useful non-projection baseline later, not a minimal R3
  fine-tune.
- Confidence: direct.

## 12. HOCBF and sampled-data CBF

- Sources: Xiao and Belta, IEEE TAC 2022,
  [DOI](https://doi.org/10.1109/TAC.2021.3105491), and Bahati et al., IEEE L-CSS
  2022, [DOI](https://doi.org/10.1109/LCSYS.2021.3076127).
- Supported result: higher-relative-degree constraints can be converted into
  affine action inequalities; sampled implementation requires a discrete-time
  condition or intersample margin rather than a continuous-time check alone.
- Limitation: safety is conditional on correct dynamics, sensing/bounds, safe
  initialization, and feasible online constraints.
- Transfer decision: use R3's already implemented sampled-data robust HOCBF
  feasible set as the single source of collision semantics. Do not learn another
  barrier network.
- Confidence: direct for theory; local implementation equivalence needs tests.
