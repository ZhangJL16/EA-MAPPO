# Controlled-sensing positioning: completed evidence and claim audit

Date: 2026-09-16. Writing axes: manuscript / research / related-work / zh-to-en /
generic (ICML). User supplied the argument and boundaries; no additional drafting
confirmation was needed. Nature-style publication claims were not imposed.

## Decision

**Representation attack: accepted.** The current class is a finite controlled
experiment/belief-state decision problem. A committed excursion is a structured
experiment. No mathematical irreducibility claim survives.

**Tool novelty: rejected.** Minimax duality, posterior continuation values,
confusing-alternative KL allocation, and shared feedback have established
precedents. Neither physical cost nor a resource-weighted LP alone is a new
learning principle.

**Separation positioning: retained, narrowly.** The cited named results do not
directly establish the existing robust conjunction: resource-realizable nesting,
unchanged execution comparators at the qualified budgets, unchanged maximal
fixed-instance coefficient, and strictly improved finite-budget minimax regret.
The manuscript can defend this explicit separation, not universal priority over
controlled sensing. Expressibility in an older framework is not the same as the
older theorem already providing the separation certificate. Conversely, merely
adding extra qualifications does not guarantee scientific significance.

This closes the requested positioning task. It does not convert a bounded
literature audit into a proof of worldwide absence or a venue acceptance forecast.

## Primary-source locators

Publisher/preprint versions were deduplicated, not counted as independent works.
Locators below refer to the linked PDF versions; theorem numbering may differ
from final typesetting.

1. [Controlled Sensing for Multihypothesis Testing](https://arxiv.org/pdf/1205.0858):
   Theorems 1–2, Sections III and V; fixed-sample error-exponent analysis and a
   causal/open-loop comparison. The resource model is representable in the broader
   paradigm; the testing theorem's objective is not our regret objective.
2. [Sequentiality and Adaptivity Gains in Active Hypothesis Testing](https://arxiv.org/pdf/1211.2291):
   equation (1), Theorems 1–3, Sections II–IV. Objective: expected sample count
   plus terminal error penalty. These are strong precedents for adaptivity, not
   evidence that sensing controls were previously static.
3. [Controlled Sensing for Multihypothesis Testing with Controlled Markovian
   Observations and Non-Uniform Control Cost](https://arxiv.org/pdf/1310.1844):
   Section 5, Theorem 5.1; Section 4.2, Theorem 4.2. Inspecting the latter prevents
   falsely describing this work as having no nonasymptotic safety/risk guarantee.
   Its testing risk is distinct from cumulative control regret.
4. [Utility Maximizing Sequential Sensing Over a Finite Horizon](https://arxiv.org/pdf/1705.05960):
   Section II, equations (19)–(21), Lemma 1 and Theorem 1. Sensing and stopping are
   coupled; finite-horizon exploitation utility is explicitly modeled. This is
   the direct counterweight to the inference-only strawman.
5. [Minimal Exploration in Structured Stochastic Bandits](https://proceedings.neurips.cc/paper/2017/file/e19347e1c3ca0c0b97de5fb3b690855a-Paper.pdf):
   Section 3 (including combinatorial feedback), Theorems 1–2, Section 6.
   Theorem 2 requires its stated regularity and gives a parameter-dependent
   multiplicative factor approaching one as algorithm parameters vanish. Do not
   summarize it as an unconditional exact finite-horizon optimizer.
6. [Online Learning with Gaussian Payoffs and Side Observations](https://proceedings.neurips.cc/paper/2015/file/8e82ab7243b7c66d768f1b8ce1c967eb-Paper.pdf):
   Section 2's generalized full-information discussion; equation (1), Section
   3.1 Theorem 1 and Remark 2. This is also a precedent for the performance-envelope
   information lower bound used in our finite-time development, not only for
   observation coupling.
7. [Risk and Optimal Policies in Bandit Experiments](https://doi.org/10.3982/ECTA21075):
   Econometrica 93(3), 1003–1029 (2025), publisher metadata verified. The
   [May 2025 preprint](https://arxiv.org/pdf/2112.06363) supplies Section 2.4,
   equation (3.1), and Theorem 2: fixed-n Gaussian Bayes risk converges to the
   diffusion PDE. Local shrinking-gap asymptotics are not fixed-hypothesis
   logarithmic asymptotics; both are relevant, but cannot be equated.

## Search receipt and limits

Academic MCP tools were unavailable. The provided academic-search preflight
reported PubMed, CrossRef REST and arXiv API reachable (3/3). Evidence was obtained
from primary arXiv PDFs, official NeurIPS proceedings, and publisher metadata.
Search results, ResearchGate, and secondary summaries were discovery aids only,
not theorem evidence. Searches combined controlled sensing with finite/asymptotic
cost, structured bandits with side observations, and resource/minimax/bundling;
reference-following additionally surfaced the active-testing and 2025
decision-risk works above.

The broader 2022 *Active Sampling for the Quickest Detection of Markov Networks*
paper was discovered, but full-text fetching timed out. It is not credited as an
inspected theorem or used to support an absence claim. This is a focused
positioning audit, not a systematic review of all controlled sensing.

## Claim–evidence map

| Manuscript claim | Evidence / disposition |
| --- | --- |
| Exact controlled-experiment representation | Accepted reduction; latest abstraction's invariant object |
| Allocation and decision bottlenecks differ in this class | Existing abstraction plus strict and robust certificates |
| Robust strict resource-enabled separation | Existing open structural box and T=12 certificate; no new proof asserted |
| Execution value unchanged | Gains across capacities; charger-terminal finite-horizon equality at T=12,24 only |
| Asymptotic price unchanged | Maximum over fixed hypotheses; individual A coefficient changes |
| Resource-aware deployed learner benefits | Unsupported by T=4096 calibration; exclude from claims |
| New foundational controlled-sensing principle | Reject; the cited papers establish the ingredients and broader distinction |
| Direct corollary of one inspected neighboring theorem | Not established: those theorem conclusions do not supply the full certificate |

## Terminology and argument structure

Use **maximal fixed-instance asymptotic regret coefficient**, not “asymptotic
minimax constant”; **finite-budget minimax regret**, not generic inference error;
**Bayes continuation value**, not a novel posterior principle; and
**resource-realizable experiment enlargement**, not a new bandit formulation.

Paragraph order: accept the reduction; credit adaptivity/cost; credit finite
utility; credit allocation; acknowledge finite/asymptotic precedents; state the
specific separation and empirical limits. This order confronts the strongest
attack before explaining the surviving contribution.

Chinese positioning: 不是“我们不能被 controlled sensing 覆盖”，而是“在 controlled
experiments 这一已有母框架内，我们证明了具有资源实现和执行等价约束的稳健
finite-budget separation”。不能把最大固定实例系数相等写成“所有实例信息价格相等”，
也不能把精确有限预算 witness 写成 T=4096 冻结 learner 的经验成功。

## Scope integrity

Only positioning documents and task/notes entries changed. No theorem source,
sealed prediction, learner, calibration outcome, simulator, seed or runtime was
changed. Independent proof validation is a separate status and is not claimed by
this literature/writing deliverable.
