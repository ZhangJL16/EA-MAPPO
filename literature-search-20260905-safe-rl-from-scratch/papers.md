# Safe / constrained navigation: failure-driven literature review

Date: 2026-09-05. Decision: implement a **from-scratch FOCOPS baseline next**, not another reach-avoid SAC fine-tune. R3 remains a reference, not a required initialization. No new training was started during this review.

## Scope and evidence depth

Screened 17 candidates; retained the 14 entries below, including one benchmark paper. This is a targeted search, not an exhaustive systematic review. Venue/title verification uses primary proceedings. Deep reading focused on FOCOPS (paper and author algorithm/collector), SafeMPO (paper algorithm and theoretical conditions), and C-TRPO (theory/practical distinction and implementation). Other entries have shallower evidence as marked. No results from another environment establish efficacy in this UAV world.

Scores are provisional **project-selection judgments**, not paper quality rankings: I = relevant mathematical insight, C = implementation completeness verified in this review, N = numerical evidence relevance to this project; 1–5, higher is stronger. N does NOT mean independently reproduced results; no source here directly tests this project's simulator. Abstract-only records have deliberately limited scores/confidence.

| ID / paper | Verified venue | Mechanism and fit | Read depth / I,C,N | Decision |
|---|---|---|---|---|
| P01 [Constrained Policy Optimization](https://proceedings.mlr.press/v70/achiam17a.html) | ICML 2017 | Reward and cost surrogates with trust-region constrained update. Foundational alternative to manually choosing a fixed collision penalty. | Proceedings; 5,2,2 | Theory/reference comparator; second-order machinery is not the smallest first implementation. |
| P02 [First Order Constrained Optimization in Policy Space](https://proceedings.neurips.cc/paper/2020/hash/af5d5ef24881f3c3049a7b9bfe74d58b-Abstract.html) | NeurIPS 2020 | On-policy reward/cost advantages; nonparametric constrained improvement, then parametric projection. | Paper practical algorithm + author source + OmniSafe source; 5,5,3 | **First implementation candidate**. Empirical candidate, not proven best for UAV. |
| P03 [Responsive Safety in Reinforcement Learning by PID Lagrangian Methods](https://proceedings.mlr.press/v119/stooke20a.html) | ICML 2020 | P/D terms address oscillatory multiplier feedback, not observability or a misdefined cost. | Proceedings + existing local adaptation; 4,4,2 | Already represented by R7; do not relabel the old correction-cost run as a contact-probability baseline. |
| P04 [CRPO: A New Approach for Safe Reinforcement Learning with Convergence Guarantee](https://proceedings.mlr.press/v139/xu21a.html) | ICML 2021 | Alternate reward improvement and violated-constraint reduction; avoids a dual variable. | Proceedings; 4,2,2 | Minimal alternative if multiplier adaptation is the isolated failure. Its natural-policy-gradient theory is not a deep PPO guarantee. |
| P05 [Constrained Update Projection Approach to Safe Policy Optimization](https://proceedings.neurips.cc/paper_files/paper/2022/hash/3ba7560b4c3e66d760fbdd472cf4a5a9-Abstract-Conference.html) | NeurIPS 2022 | GAE-based reward update followed by constraint projection. | Proceedings + repository README; 4,3,2 | Coherent alternative, not an extra module to stack on FOCOPS. |
| P06 [Constrained Variational Policy Optimization for Safe Reinforcement Learning](https://proceedings.mlr.press/v162/liu22b.html) | ICML 2022 | Constrained variational E-step and neural M-step. | Proceedings; 4,2,2 | Off-policy alternative; requires a faithful optimizer/critic implementation. |
| P07 [Sauté RL: Almost Surely Safe Reinforcement Learning Using State Augmentation](https://proceedings.mlr.press/v162/sootla22a.html) | ICML 2022 | Remaining safety budget becomes state. | Proceedings; 4,2,2 | Especially relevant later to energy resources. Not automatically suitable for learning sparse first-contact avoidance with zero budget. |
| P08 [Safety-Gymnasium: A Unified Safe Reinforcement Learning Benchmark](https://proceedings.neurips.cc/paper_files/paper/2023/hash/3c557a3d6a48cc99444f85e924c66753-Abstract-Datasets_and_Benchmarks.html) | NeurIPS 2023 Datasets and Benchmarks | Benchmark / reference algorithms for assessing cost-return tradeoffs. | Proceedings + public documentation; 4,4,3 | Contract and reporting reference, not evidence of safety in this simulator. |
| P09 [Off-Policy Primal-Dual Safe Reinforcement Learning](https://proceedings.iclr.cc/paper_files/paper/2024/hash/2f8b56543953d60f262fb2c4b85c50b3-Abstract-Conference.html) | ICLR 2024 | CAL addresses cost underestimation through uncertainty-aware conservative optimization and local policy convexification. | Proceedings + author README/configuration; 5,4,3 | Explains a relevant off-policy failure mechanism; not proof of our specific causal failure. Implementation is more than adding a multiplier to SAC. |
| P10 [Feasibility Consistent Representation Learning for Safe Reinforcement Learning](https://proceedings.mlr.press/v235/cen24b.html) | ICML 2024 | Feasibility representation learning addresses weak/sparse constraint supervision. | Proceedings; 4,2,2 | Defer until evidence implicates representation rather than objective/update semantics. |
| P11 [Safety-Prioritizing Curricula for Constrained Reinforcement Learning](https://proceedings.iclr.cc/paper_files/paper/2025/hash/120ed726cf129dbeb8375b6f8a0686f8-Abstract-Conference.html) | ICLR 2025 | Curriculum adapted to constrained learning. | Proceedings; 4,2,2 | Only introduce if the matched unconstrained learner demonstrates an exploration bottleneck. |
| P12 [Embedding Safety into RL: A New Take on Trust Region Methods](https://proceedings.mlr.press/v267/milosevic25a.html) | ICML 2025 | C-TRPO modifies policy-space geometry using safety barriers. | Paper theory/practical sections + author source; 5,4,3 | Strong comparator. Exact safe-interior result and approximate implementation/recovery branch must be distinguished. |
| P13 [ActSafe: Active Exploration with Safety Constraints for Reinforcement Learning](https://iclr.cc/virtual/2025/poster/29177) | ICLR 2025 | Model-based safe exploration. | Official conference/project overview; 3,2,2 | Defer: introduces learned dynamics/model uncertainty; not the minimal way to isolate current failure. |
| P14 [SafeMPO: Constrained Reinforcement Learning with Probabilistic Incremental Improvement](https://proceedings.iclr.cc/paper_files/paper/2026/hash/1fa0c4e5a7e189729230d018b229abc7-Abstract-Conference.html) | ICLR 2026 | Incremental movement toward feasibility, then neural policy cloning; reward and cost Retrace critics. | Paper algorithm and selected theory/proofs; 5,3,3 | Strong next-generation candidate, especially unsafe initialization. Author implementation not located in this search; do not fabricate one. |

## Actual implementation lessons

### FOCOPS: selected reference

Author implementation: [repository](https://github.com/ymzhang01/focops), [training algorithm](https://raw.githubusercontent.com/ymzhang01/focops/main/focops_main.py), [collector](https://raw.githubusercontent.com/ymzhang01/focops/main/data_generator.py). Both source files were read, not executed or installed.

It uses a Gaussian actor, reward state value and cost state value. The practical loss includes new-to-old KL and an importance-ratio weighted reward-minus-cost advantage, with a per-state KL mask and a batch KL stopping criterion. Its dual variable is updated from estimated episode cost. This is not a PPO clip objective renamed “FOCOPS”.

The original defaults (including gamma_c=0.99, discounted cost returns, independently normalized reward/cost advantages, old Gym terminal handling and a finite complete-episode cost history) must not silently become our physical event-probability contract. In particular, a batch without completed episodes cannot produce a valid completed-episode mean. Fast completed failures can also bias such a mean when long episodes are censored.

[OmniSafe's implementation](https://raw.githubusercontent.com/PKU-Alignment/omnisafe/main/omnisafe/algorithms/on_policy/first_order/focops.py) was cross-checked: its combined-advantage scaling and KL stopping details are not identical to the author version. Pin one reference version when implementing; document intentional adaptations instead of mixing variants invisibly. Do not copy serial Python value evaluation: batch policy and both value predictions on CUDA.

### CAL: off-policy safety is not “free sample reuse”

[Author repository](https://github.com/ZifanWu/CAL) documents cost-value ensembles, conservatism parameters and local convexification, including an eight-member cost ensemble configuration for MuJoCo. These are purposeful algorithmic components, not evidence that any extra network is gratuitous. We defer CAL to keep the first diagnosis narrow, not because it is invalid.

### C-TRPO: the initialization caveat matters

[Author implementation](https://github.com/milosen/ctrpo) uses conjugate gradients, Hessian-vector products, line search and a recovery branch outside the intended feasible region. The paper's ideal divergence is defined in the safe occupancy interior; an approximate implementation with noisy cost estimates is a different object. It would be incorrect to say the code cannot start unsafe, or that its recovery branch inherits unconditional training safety.

### SafeMPO: promising, but reproduce the whole method

Read [paper Algorithm 1 and theoretical conditions](https://proceedings.iclr.cc/paper_files/paper/2026/file/1fa0c4e5a7e189729230d018b229abc7-Paper-Conference.pdf). It maintains reward and cost critics using Retrace, solves a nonparametric constrained improvement problem, then performs a KL-controlled cloning step. Incremental surrogate feasibility does not certify a noisy neural critic or every UAV trajectory. A stripped-down one-step-TD SAC variant should not be called SafeMPO.

## Selection logic

The useful change is **separate utility and physical safety semantics, then control policy updates**. FOCOPS offers the smallest verified first-order implementation among the deeply inspected candidates, avoiding the previous actor's direct exploitation of replay-trained action values. On-policy state-value estimation still has approximation error; this is a reason to test it, not a guarantee it will succeed.

First compare from-scratch PPO and FOCOPS under the same simulator/task/reward/action/episode contract. R3 is a frozen historical reference. Do not add curriculum, recurrent memory, safety shield, feasibility encoder, energy critic and multiple uncertainty models simultaneously.
