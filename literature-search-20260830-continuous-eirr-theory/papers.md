# Literature Search: Continuous EIRR, Representation Quotients, and First-Passage Risk

Date: 2026-08-30  
Mode: standard  
Search purpose: determine whether a continuous/neural executed-interface EIRR
theorem can be more than a routine composition of known kernel regression,
off-policy evaluation, and resolvent perturbation results.  
Target venue family: ICLR/ICML/NeurIPS oral-level theory, with mathematical
foundations from COLT, Annals of Statistics, SIAM, Mathematics of Operations
Research, and Stochastic Processes and their Applications.  
Source-quality policy: primary proceedings, archival journals, and author/arXiv
copies only; policy-excluded sources were not used.

## Summary

- A direct continuous-interface theorem of the form “estimate a conditional mean
  embedding, insert its error into a resolvent perturbation bound” is already a
  composition of established results. It is not a safe novelty claim.
- Function-class-restricted coverage, not raw state/action density, is already
  known to characterize sharp off-policy evaluation difficulty. State aggregation
  induced by the function class can also be harder than the original MDP.
- Kernel conditional mean embedding has rigorous existence, misspecified optimal
  rates, matching lower bounds, and RL confidence-set applications. Merely choosing
  an RKHS or neural tangent kernel is therefore not the contribution.
- Continuous bisimulation already supplies value-preserving pseudometrics and
  approximation guarantees. Any proposed interface quotient must distinguish
  itself by the multiplicative first-passage object and the irreversible boundary,
  not by the word “quotient.”
- The 2025 refresh strengthens this collision boundary: KROPE proves stability
  and Bellman completeness for bisimulation-based offline value representations,
  while recent ICML/AAAI/NeurIPS work covers risk reductions, transient
  ERM/EVaR dynamic programming, and total-reward risk-sensitive Q-learning.
- The viable theorem target is a two-layer object: a **risk-observable interface
  quotient** characterizes which distinctions affect the killed Feynman--Kac
  grid, and a separate stopped-margin functional characterizes which propagated
  errors alter the one-way return decision. The target rate should depend on
  quotient complexity under risk occupation rather than ambient interface
  density, then reach the boundary functional through the positive resolvent.
- No inspected source states that full chain. This is an inferred opportunity,
  not established novelty.

## Paper Table

| # | Title | Year | Venue/source | Type | Insight | Completeness | Numeric evidence | Overall | Relevance |
| ---: | --- | ---: | --- | --- | ---: | ---: | ---: | --- | --- |
| 1 | [Minimax-Optimal Off-Policy Evaluation with Linear Function Approximation](https://proceedings.mlr.press/v119/duan20b.html) | 2020 | ICML | theory/proof | 5 | 5 | 4 | Risk | Conditional-mean-operator estimation is minimax optimal; error is governed by function-class restricted chi-square divergence. This is the closest statistical template after the Doob reduction. |
| 2 | [Offline Reinforcement Learning: Role of State Aggregation and Trajectory Data](https://proceedings.mlr.press/v247/jia24a.html) | 2024 | COLT | theory/proof | 5 | 5 | 1 | Risk | Shows that function-class/data-induced aggregated concentrability, not original-MDP concentrability, governs OPE; trajectory data need not repair the obstruction. |
| 3 | [Offline Reinforcement Learning: Fundamental Barriers for Value Function Approximation](https://proceedings.mlr.press/v178/foster22a.html) | 2022 | COLT | theory/proof | 5 | 5 | 1 | Risk | Coverage plus value realizability can still be insufficient; warns that an encoder trained by supervised prediction alone cannot certify sequential transfer. |
| 4 | [Optimally Tackling Covariate Shift in RKHS-Based Nonparametric Regression](https://doi.org/10.1214/23-AOS2268) | 2023 | Annals of Statistics | theory/proof | 5 | 5 | 2 | Risk | Gives minimax KRR rates under bounded or finite-second-moment likelihood ratios; truncation is already a standard response to unbounded shift. |
| 5 | [Optimal Rates for Regularized Conditional Mean Embedding Learning](https://proceedings.neurips.cc/paper_files/paper/2022/hash/1c71cd4032da425409d8ada8727bad42-Abstract.html) | 2022 | NeurIPS | theory/proof | 5 | 5 | 2 | Risk | Establishes adaptive misspecified CME rates and a matching lower bound, including infinite-dimensional output RKHSs. |
| 6 | [A Rigorous Theory of Conditional Mean Embeddings](https://doi.org/10.1137/19M1305069) | 2020 | SIAM Journal on Mathematics of Data Science | theory/proof | 5 | 5 | 1 | A | Clarifies when operator CME formulas are actually valid; prevents silently assuming an inverse covariance operator exists pointwise. |
| 7 | [Value Function Approximations via Kernel Embeddings for No-Regret Reinforcement Learning](https://proceedings.mlr.press/v189/chowdhury23a.html) | 2023 | ACML | theory/method | 4 | 4 | 3 | A | Builds transition CME confidence sets and obtains an effective-dimension regret bound in continuous/large spaces. |
| 8 | [Bisimulation Metrics for Continuous Markov Decision Processes](https://doi.org/10.1137/10080484X) | 2011 | SIAM Journal on Computing | theory/proof | 5 | 5 | 2 | Risk | Gives continuous-state behavioral pseudometrics, sampling approximation, and value continuity; a generic representation quotient is covered prior art. |
| 9 | [Linear Variance Bounds for Particle Approximations of Time-Homogeneous Feynman--Kac Formulae](https://arxiv.org/abs/1108.3988) | 2012 | Stochastic Processes and their Applications | theory/proof | 5 | 5 | 3 | A | Multiplicative drift plus regularity can replace naive exponential-in-time Monte Carlo variance by a linear-in-time particle bound. |
| 10 | [Risk-Averse Control of Undiscounted Transient Markov Models](https://doi.org/10.1137/13093902X) | 2014 | SIAM Journal on Control and Optimization | theory/proof | 5 | 5 | 2 | A | Risk transience, undiscounted dynamic programming, stochastic shortest path, and optimal stopping are established mathematical foundations. |
| 11 | [Offline Reinforcement Learning with Realizability and Single-Policy Concentrability](https://proceedings.mlr.press/v178/zhan22a.html) | 2022 | COLT | theory/method | 5 | 5 | 3 | Risk | A primal--dual density-ratio method already achieves polynomial sample complexity under weak single-policy coverage and realizability assumptions. |
| 12 | [Pessimism Meets Risk: Risk-Sensitive Offline Reinforcement Learning](https://proceedings.mlr.press/v235/zhang24aq.html) | 2024 | ICML | theory/method | 5 | 5 | 4 | Risk | Entropic offline RL, exponential Bellman errors, coverage, pessimism, variance refinement, and tight finite-sample analysis are already combined for linear finite-horizon MDPs. |
| 13 | [Stable Offline Value Function Learning with Bisimulation-based Representations](https://proceedings.mlr.press/v267/pavse25a.html) | 2025 | ICML | theory/method | 5 | 5 | 4 | Risk | KROPE proves spectral stability of LSPE representations and a Bellman-completeness route; generic stable quotient representations are now an explicit collision. |
| 14 | [A Reductions Approach to Risk-Sensitive Reinforcement Learning with Optimized Certainty Equivalents](https://proceedings.mlr.press/v267/wang25bl.html) | 2025 | ICML | theory/method | 5 | 5 | 3 | Risk | Reduces static OCE objectives, including entropic risk, to standard RL oracles in augmented MDPs and covers rich observations. |
| 15 | [Online Learning in Risk Sensitive constrained MDP](https://proceedings.mlr.press/v267/ghosh25c.html) | 2025 | ICML | theory/method | 4 | 5 | 2 | A | Gives entropic-risk constraint regret and violation bounds through a continuous-budget augmentation. |
| 16 | [Risk-averse Total-reward MDPs with ERM and EVaR](https://ojs.aaai.org/index.php/AAAI/article/view/34275) | 2025 | AAAI | theory/proof | 5 | 5 | 2 | Risk | Establishes stationary-policy and dynamic-programming foundations for ERM/EVaR under transient total reward. |
| 17 | [Risk-Averse Total-Reward Reinforcement Learning](https://proceedings.neurips.cc/paper_files/paper/2025/hash/8606dcf42b74b9f29649408b7fd0b163-Abstract-Conference.html) | 2025 | NeurIPS | theory/method | 4 | 5 | 3 | Risk | Supplies model-free Q-learning with convergence and performance guarantees for total-reward ERM/EVaR. |

Scores assess source quality and proximity, not venue acceptance probability.
For pure theory papers, the numeric-evidence score records the amount of explicit
simulation/finite-example evidence and is not a judgment on proof quality.

## Closest-Work Clusters

### 1. Function-class-restricted statistical difficulty

- Representative work: Duan--Jia--Wang; Jia--Rakhlin--Sekhari--Wei;
  Foster--Krishnamurthy--Simchi-Levi--Xu; Zhan et al.
- Already covered: restricted chi-square/concentrability, function-class-induced
  aggregation, realizability/coverage tradeoffs, and offline lower bounds.
- Remaining gap: the target occupation in this project is non-normalized,
  exponentially tilted, killed at charger hitting, composed through a queryable
  safety execution interface, and finally localized at an irreversible decision
  boundary.
- Differentiation condition: prove that this quotient-plus-boundary composition
  gives a distinct and sharper complexity than applying standard OPE to the full
  known Doob transform. If it does not, EIRR must be positioned as a
  specialization.

### 2. Continuous conditional-transition learning

- Representative work: Li et al.; Klebanov--Schuster--Sullivan;
  Chowdhury--Oliveira; Ma--Pathak--Wainwright.
- Already covered: valid CME definitions, optimal misspecified rates, confidence
  sets, effective dimension, likelihood-ratio truncation, and minimax covariate-
  shift KRR.
- Remaining gap: target-specific prediction of the killed exponential continuation
  family under a learned representation and dependent first-passage trajectories.
- Rescue route: estimate only the risk-observable conditional features required by
  the Feynman--Kac/return-boundary function class. Do not estimate a characteristic
  embedding of the full transition distribution unless the theorem requires it.

### 3. Behavioral quotients and representation sufficiency

- Representative work: Ferns--Panangaden--Precup; Jia et al.
- Already covered: continuous bisimulation pseudometrics, value continuity, and
  function-class-induced aggregation.
- Remaining gap: a pseudometric jointly indexed by risk level, charger killing,
  executed action, and stopped return margin, with a matching estimation lower
  bound.
- Rescue route: define the quotient by equality of the conditional multiplicative
  continuation operators needed by the decision certificate, not by Euclidean
  similarity, policy ID, or one-step mean prediction.

### 4. Exponential first-passage stability

- Representative work: Whiteley--Kantas--Jasra; Cavus--Ruszczynski;
  Zhang et al.
- Already covered: risk transience, Feynman--Kac stability tools, entropic Bellman
  analysis, and risk-sensitive pessimism.
- Remaining gap: source-to-target learning under an executed policy--filter
  composition and a one-way charger commitment.
- Rescue route: combine a multiplicative drift function with a localized
  risk--boundary representation error. A global horizon factor alone is neither
  new nor empirically aligned with the R3 residual-tail result.

### 5. 2025 representation and total-risk collision refresh

- Representative work: Pavse et al.; Wang et al.; Ghosh--Moharrami; Su et al.
- Already covered: stable bisimulation-style OPE features, Bellman completeness,
  rich-observation risk reductions, transient ERM/EVaR dynamic programming, and
  risk-sensitive total-reward Q-learning.
- Remaining gap: none of these sources couples intrinsic quotient estimation
  error simultaneously to a collapsing Perron mode and an irreversible stopped
  margin.
- Differentiation condition: the project must prove and test the joint
  (d_Q,mathfrak g,Delta,kappa) phase. A generic risk-aware kernel,
  representation-stability result, or transient exponential Bellman algorithm
  is already occupied.

## Opportunity Map

| Cluster | Status | Open gap | Possible direction | Evidence needed | Risk |
| --- | --- | --- | --- | --- | --- |
| Generic continuous CME EIRR | covered central claim | Only project-specific composition remains | Use as baseline/lemma | Direct theorem reduction to CME/OPE | Very high |
| Raw interface density ratio | covered central claim | Ambient overlap may be overly pessimistic | Replace with restricted quotient coefficient | Matching example where raw ratio diverges but quotient coefficient is finite | High |
| Risk-observable interface quotient | theory/analysis gap | No located theorem joins multiplicative first passage, executed interface, and boundary invariance | Prove exact/approximate quotient sufficiency and necessity | Upper bound, packing lower bound, reduction audit | Medium-high |
| Risk--boundary localized representation | theory/analysis gap | Standard bisimulation controls global value, not stopped one-way decisions | Weight quotient distortion by risk occupation and oracle-shadow boundary mass | First-disagreement/Pareto theorem with matching construction | Medium-high |
| Learned neural encoder | crowded but open | Architecture claims are not theorems | Derive loss from quotient distortion; instantiate with kernels or spectral-normalized nets | Generalization bound plus failure when representation collapses a risk-distinguishable pair | High |
| Particle/Feynman--Kac estimator | covered central claim | Composition-shift interface not addressed | Baseline or variance-control module | Fair comparison with direct MC/CME/DICE | Medium |

## Frozen Theorem Target for the Next Derivation Stage

Let (q=(x,u)) denote an executed interface, let (Lambda) be a finite risk
grid, and let (mathcal F) be a continuation class containing the candidate
Feynman--Kac values. Define

\[
\mathcal K_\lambda f(q)
=\mathbb E\!\left[e^{\lambda C}
\{\mathbf 1_{X'\in G}+\mathbf 1_{X'\notin G}f(X')\}\mid q\right]
\]

and the risk-observable pseudometric

\[
d_{\Lambda,\mathcal F}(q,q')
=\sup_{\lambda\in\Lambda,\,f\in\mathcal F_1}
\left|\mathcal K_\lambda f(q)-\mathcal K_\lambda f(q')\right|.
\]

The next theorem should prove all four parts, or be rejected as insufficient:

1. **Exact quotient sufficiency.** If an encoder (h) separates equivalence
   classes of (d_{\Lambda,\mathcal F}=0) and (mathcal F) is multiplicatively
   Bellman closed, the killed log-MGF grid and the resulting exact-risk
   ReturnManager decision factor through (h(q)).
2. **Approximate decision stability.** If within-code distortion is at most
   (arepsilon), derive an explicit resolvent/log-MGF error, then a one-sided
   first-disagreement and stranding--throughput rectangle under the stopped margin
   law. The constant must expose risk transience rather than hide an arbitrary
   horizon.
3. **Statistical rate on the quotient.** Give an estimator whose rate depends on
   a risk-restricted quotient coefficient/effective dimension under source data,
   not the minimum density of every raw continuous interface.
4. **Matching necessity.** Construct a packing of risk-distinguishable quotient
   classes showing that any encoder or estimator with smaller effective quotient
   capacity incurs either log-MGF error or irreversible decision error.

This target is intentionally stronger than “an RKHS estimator converges.” It gives
future network/loss design a theorem-backed rule: preserve conditional exponential
continuation features that are reachable under risk occupation and consequential
at the return boundary; invariance to all other interface variation is permitted.

## Stop Conditions

- Stop the quotient headline if it is algebraically identical to Duan et al.'s
  restricted chi-square theorem after the known Doob transform.
- Stop any claim that importance weighting is always required: under well-specified
  bounded shift, unweighted KRR can already be minimax optimal.
- Stop any claim that a characteristic kernel or full transition model is
  necessary; the intended object is task/risk/decision sufficient, not universally
  distribution identifying.
- Stop the oral theory route if no matching necessity result or strict structural
  advantage over primitive plug-in/DICE/CME can be proved.
- Keep the finite-interface Theorems 13--15 as rigorous foundations even if this
  stronger target fails.

## Benchmark and Method Implications

No new benchmark is authorized by this search. When the formal Gate is reopened,
the closest theoretical baselines are primitive model plug-in, fitted conditional
mean/CME, Doob-transformed restricted-OPE/DICE, direct exponential TD, and the
existing risk-sensitive pessimistic method. The empirical (2\pi\times2\Pi)
design must report both raw interface coverage and the learned quotient/risk-
restricted coefficient; otherwise the structural theorem cannot be tested.

## Citation and Positioning Cautions

- Cite Duan et al. for conditional-mean-operator OPE and restricted chi-square.
- Cite Jia et al. and Foster et al. before claiming a representation/coverage
  theorem.
- Cite Li et al. and Klebanov et al. before claiming CME validity or rate novelty.
- Cite Ferns et al. before using “bisimulation,” “behavioral metric,” or quotient
  representation language.
- Cite Whiteley et al. and Cavus--Ruszczynski for exponential drift/risk transience.
- Treat the four-part risk-observable quotient plus stopped-boundary theorem as a
  candidate until a full proof and a second closest-theorem audit are complete.
