# Literature Review: History-Conditioned Physics-Guided Safe Trajectories

## Search purpose and source policy

The search asks whether a history-conditioned, physics-guided, coarse-to-fine trajectory sampler with CBF/reachability verification and energy ranking contains a theorem-level contribution beyond safe MPPI, sampling MPC with safety filters, terminal-safe MPC, and learned trajectory proposal systems.

Only primary paper pages, official proceedings, DOI records, or arXiv manuscript pages are included. Search terms covered safe MPPI, barrier-guided importance sampling, reachability-guided sampling, sampled-data trajectory certificates, terminal backup sets, candidate-set coverage, diffusion trajectory planning, history-conditioned motion prediction, and physics-guided MPC.

## Count

- Existing verified primary matrix: **49 papers** in [uav_safety_energy_literature_matrix.csv](uav_safety_energy_literature_matrix.csv).
- Added trajectory/history closest works below: **21 papers**.
- **Total primary papers screened: 70.**
- Papers published in 2020 or later: **59/70 (84.3%)**.

The existing matrix contributes:

- IDs 1–10: foundational, high-relative-degree, and performance-oriented CBF methods;
- IDs 11–16: sampled-data and predictive barrier methods;
- IDs 17–28: learned, perception-based, and uncertainty-aware barrier methods;
- IDs 29–42: constrained RL, intervention, recovery, reachability, and sampling-safe RL;
- IDs 43–49: persistent robotics and energy-aware safety.

## Added primary papers

| ID | Paper | Year / venue | Primary source | Main structure | Overlap with candidate |
|---:|---|---|---|---|---|
| 50 | Williams, Aldrich, Theodorou, “Model Predictive Path Integral Control: From Theory to Parallel Computation” | 2017, JGCD | [DOI](https://doi.org/10.2514/1.G001921) | Parallel path-integral trajectory sampling | Base massive-sampling mechanism |
| 51 | Gandhi et al., “Robust Model Predictive Path Integral Control: Analysis and Performance Guarantees” | 2021, RA-L | [DOI](https://doi.org/10.1109/LRA.2021.3057563) | Robust MPPI and performance analysis | Robust sampling baseline |
| 52 | Gandhi, Almubarak, Theodorou, “Safe Importance Sampling in Model Predictive Path Integral Control” | 2023, arXiv | [arXiv](https://arxiv.org/abs/2303.03441) | Barrier-state control embedded in importance sampling | Fatal overlap with safety-guided proposal |
| 53 | Rabiee, Hoagg, “Guaranteed-Safe MPPI Through Composite Control Barrier Functions for Efficient Sampling in Multi-Constrained Robotic Systems” | 2024, arXiv | [arXiv](https://arxiv.org/abs/2410.02154) | Composite CBF closed-form policy makes sampled trajectories safe | Fatal overlap with MPPI plus CBF safety |
| 54 | Borquez et al., “DualGuard MPPI: Safe and Performant Optimal Control by Combining Sampling-Based MPC and Hamilton-Jacobi Reachability” | 2025, arXiv | [arXiv](https://arxiv.org/abs/2502.01924) | Reachability controller guards MPPI samples | Fatal overlap with reachability-verified sampling |
| 55 | Yin et al., “Safe Beyond the Horizon: Efficient Sampling-based MPC with Neural Control Barrier Functions” | 2025, RSS | [RSS](https://www.roboticsproceedings.org/rss21/p112.html) | Terminal neural CBF and sampling MPC | Fatal overlap with terminal-safe sampling |
| 56 | Parwana et al., “BR-MPPI: Barrier Rate Guided MPPI for Enforcing Multiple Inequality Constraints with Learned Signed Distance Field” | 2025, arXiv | [arXiv](https://arxiv.org/abs/2506.07325) | Barrier-rate-guided sampling with learned geometry | Strong overlap with CBF-gradient proposal |
| 57 | Yin, Zhang, Tsiotras, “Risk-Aware Model Predictive Path Integral Control Using Conditional Value-at-Risk” | 2023, ICRA | [DOI](https://doi.org/10.1109/ICRA48891.2023.10161100) | CVaR trajectory-risk sampling | Risk-adjusted MPPI baseline |
| 58 | Lambert et al., “Stein Variational Model Predictive Control” | 2021, CoRL | [PMLR](https://proceedings.mlr.press/v155/lambert21a.html) | Multimodal particle-based MPC | Diverse proposal baseline |
| 59 | Bhardwaj et al., “STORM: An Integrated Framework for Fast Joint-Space Model-Predictive Control for Reactive Manipulation” | 2022, CoRL | [PMLR](https://proceedings.mlr.press/v164/bhardwaj22a.html) | GPU sampling MPC for real-time control | Compute and latency baseline |
| 60 | Holmes et al., “Reachable Sets for Safe, Real-Time Manipulator Trajectory Design” | 2020, RSS | [RSS](https://www.roboticsproceedings.org/rss16/p100.html) | Reachable-set trajectory library with online safety checking | Strong overlap with certified trajectory library |
| 61 | Michaux et al., “Reachability-based Trajectory Design with Neural Implicit Safety Constraints” | 2023, RSS | [RSS](https://roboticsproceedings.org/rss19/p062.html) | Reachability plus neural implicit obstacle constraints | Strong overlap with learned proposal and exact verification |
| 62 | Begzadic et al., “Back to Base: Towards Hands-Off Learning via Safe Resets with Reach-Avoid Safety Filters” | 2025, L4DC | [PMLR](https://proceedings.mlr.press/v283/begzadic25a.html) | Reach-avoid backup policy and safe resets | Terminal return/backup overlap |
| 63 | Janner et al., “Planning with Diffusion for Flexible Behavior Synthesis” | 2022, ICML | [PMLR](https://proceedings.mlr.press/v162/janner22a.html) | Generative trajectory planning | Learned proposal baseline; no certification |
| 64 | Chi et al., “Diffusion Policy: Visuomotor Policy Learning via Action Diffusion” | 2023, RSS | [DOI](https://doi.org/10.15607/RSS.2023.XIX.026) | History-conditioned action-sequence diffusion | Learned history proposal overlap |
| 65 | Salzmann et al., “Trajectron++: Dynamically-Feasible Trajectory Forecasting with Heterogeneous Data” | 2020, ECCV | [DOI](https://doi.org/10.1007/978-3-030-58523-5_40) | Dynamics-aware multimodal trajectory prediction | History/motion prediction only, not control safety |
| 66 | Yuan et al., “AgentFormer: Agent-Aware Transformers for Socio-Temporal Multi-Agent Forecasting” | 2021, ICCV | [DOI](https://doi.org/10.1109/ICCV48922.2021.00967) | Temporal multi-agent trajectory modeling | Dynamic-obstacle proposal input |
| 67 | Shi et al., “Motion Transformer with Global Intention Localization and Local Movement Refinement” | 2022, NeurIPS | [DOI](https://doi.org/10.52202/068431-0473) | Coarse-to-fine multimodal motion prediction | Coarse-to-fine history proposal analogy |
| 68 | Hafner et al., “Learning Latent Dynamics for Planning from Pixels” | 2019, ICML | [PMLR](https://proceedings.mlr.press/v97/hafner19a.html) | Latent world model and online planning | Learned pseudo-rollout comparator; no exact certificate |
| 69 | Hafner et al., “Dream to Control: Learning Behaviors by Latent Imagination” | 2020, ICLR | [OpenReview](https://openreview.net/forum?id=S1lOTC4tDS) | History-conditioned latent imagination | Proposal acceleration comparator |
| 70 | Chua et al., “Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models” | 2018, NeurIPS | [NeurIPS](https://proceedings.neurips.cc/paper/2018/hash/3de568f8597b94bda53149c7d7f5958c-Abstract.html) | Ensemble dynamics plus CEM MPC | Uncertainty-aware proposal and CEM baseline |

## Closest-work clusters

### 1. Safe MPPI and safety-guided sampling

SC-MPPI, GS-MPPI, DualGuard MPPI, BR-MPPI, CVaR-MPPI, and Safe Beyond the Horizon already cover the broad template “sample trajectories, bias or guard samples with safety structure, then execute a performant safe control.” This cluster makes the generic framework non-novel.

The present implementation differs operationally in three ways:

1. exact clipped UAV physics and telemetry-energy rollout;
2. a separate continuous-time quartic verifier after proposal;
3. explicit safety → progress → energy lexicographic selection.

These are useful system choices, not yet a theorem-level distinction.

### 2. Reachability and terminal safety

ARMTD, neural implicit reachability trajectory design, DualGuard, Back to Base, predictive CBFs, and standard terminal-set MPC cover finite trajectory certification and terminal recoverability. The usual shifted-sequence-plus-backup recursive-feasibility proof is prior art.

### 3. History-conditioned and learned trajectory proposals

Diffusion Policy, Diffuser, Trajectron++, AgentFormer, Motion Transformer, PlaNet, Dreamer, and PETS establish that temporal context, learned dynamics, multimodal proposals, and coarse-to-fine prediction are mature ideas. None makes a proposed UAV action safe without an independent verifier, but history-conditioned proposal generation itself is not novel.

### 4. GPU sampling and multimodal optimization

MPPI, Robust MPPI, Stein MPC, and STORM provide strong baselines for GPU-batched candidate generation and multimodal search. A claim of fast massive sampling requires direct P95/P99 evidence, not architecture language.

### 5. Sampled-data and exact certification

Existing matrix IDs 11–16 cover sampled-data CBF theory; IDs 1–10 cover CBF/HOCBF synthesis; IDs 17–28 cover learning/perception uncertainty. The quartic static-sphere interval minimum is mathematically exact and useful, but it is a specialized verifier rather than a broad new safety principle.

## Closest-prior attack

| Candidate claim | Closest prior | Result |
|---|---|---|
| Sampling around SAC plus CBF verification is new | SC-MPPI, GS-MPPI, BR-MPPI | **REJECT** |
| Reachability-guarded trajectory sampling is new | DualGuard, ARMTD, neural implicit reachability | **REJECT** |
| Terminal backup makes sampling recursively feasible | terminal-set MPC, backup CBF, Safe Beyond Horizon | **REJECT** |
| History-conditioned trajectory proposals are new | Diffusion Policy, AgentFormer, MTR, world models | **REJECT** |
| Learning only proposes while physics/certificate decide | safe sampling and shielding literature | **NOT UNIQUE**, but scientifically sound |
| Certified upper-clearance rejection is a new theorem | triangle inequality with bounded approximation error | **CORRECT BUT ELEMENTARY** |
| Candidate-set epsilon cover gives energy suboptimality | standard Lipschitz covering argument | **CORRECT BUT STANDARD** |
| Full UAV system couples exact safety, progress, and telemetry energy | no single paper found with the identical stack | **POSSIBLE ENGINEERING INTEGRATION**, not theoretical novelty |

## Opportunity map

The only potentially defensible new object would be a **causal, history-adaptive trajectory error tube** that is:

1. tighter than a global worst-case disturbance tube;
2. computed within the 50 ms cycle;
3. valid under the closed-loop selection distribution;
4. used to certify rejection or acceptance without depending on a neural safety decision.

No such object is established here. Empirical residuals or conformal calibration alone do not deliver a deterministic feedback-valid tube.

## Literature verdict

**Novelty status: ENGINEERING CONTRIBUTION ONLY at the current stage.**

The architecture is justified as a rigorous experimental platform and may improve safe recall, freeze, energy, or latency. Generic “history + massive sampling + CBF/reachability verifier” is heavily covered. A paper-level theory claim requires a genuinely new history-adaptive error set or a nonstandard safe-pruning/coverage result with assumptions that are both verifiable and non-vacuous.
