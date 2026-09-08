# Literature Search: Stopped-Lp Quotient Learning

Date: 2026-08-30  
Mode: standard  
Search purpose: audit whether cross-fitted stopped-occupation regression and its
margin transfer can support a new return-to-charge theorem beyond existing
classification, policy-learning, covariate-shift, OPE, and representation work.  
Target venue family: ICLR/ICML/NeurIPS, with theory anchors from COLT and the
Annals of Statistics.  
Source-quality policy: official proceedings, archival pages, and author/arXiv
records only; policy-excluded and secondary sources were not used.

## Summary

- The generic conversion from an \(L_p\) score error and a margin condition to
  decision/regret powers is occupied. Audibert--Tsybakov establish the plug-in
  classification foundation, and Kim--Zubizarreta state the corresponding
  finite-\(L_q\) policy-learning comparison inequality.
- Cross-fitting and second-order nuisance effects are occupied by orthogonal
  statistical learning. Fast finite-library selection under nuisance-dependent
  squared losses is occupied by causal Q-aggregation.
- Target-law nonparametric regression under covariate shift already has sharp
  minimax characterizations. Function-class-restricted OPE and KROPE already
  cover structure-dependent coverage and representation stability.
- Therefore neither “boundary-weighted loss,” “cross-fit the weights,” nor
  “learn a lower-dimensional representation” is an Oral-level claim alone.
- A defensible project theorem must couple a certified risk-observable executed-
  interface chart to killed exponential first-passage pseudo-outcomes, stopped
  target occupation, transience amplification, and the irreversible
  stranding--throughput boundary. Even that composition remains conditional
  until a learned chart has a rate unavailable to matched transformed baselines.

## Paper Table

| # | Title | Year | Venue/source | Type | Insight | Completeness | Numeric evidence | Overall | Relevance |
| ---: | --- | ---: | --- | --- | ---: | ---: | ---: | --- | --- |
| 1 | [Fast learning rates for plug-in classifiers](https://arxiv.org/abs/0708.2321) | 2007 | Annals of Statistics / author record | theory/proof | 5 | 5 | 2 | Risk | Classical margin-based plug-in rates and matching minimax lower bounds; generic margin localization is occupied. |
| 2 | [Toward Fair and Robust Policy Learning](https://proceedings.mlr.press/v202/kim23ab.html) | 2023 | ICML | theory/method | 4 | 4 | 4 | Risk | Its comparison lemma converts finite \(L_q\) contrast error into policy regret with the same margin powers used by the stopped-decision theorem. |
| 3 | [A New Similarity Measure for Covariate Shift with Applications to Nonparametric Regression](https://proceedings.mlr.press/v162/pathak22a.html) | 2022 | ICML | theory/proof | 5 | 5 | 3 | Risk | Gives sharp target-risk rates through local source/target ball-mass ratios; a generic stopped-target regression rate is not new. |
| 4 | [Optimally Tackling Covariate Shift in RKHS-Based Nonparametric Regression](https://doi.org/10.1214/23-AOS2268) | 2023 | Annals of Statistics | theory/proof | 5 | 5 | 2 | Risk | Establishes minimax KRR rates under bounded and finite-second-moment likelihood ratios. |
| 5 | [Statistical Learning with a Nuisance Component](https://proceedings.mlr.press/v99/foster19c.html) | 2019 | COLT | theory/proof | 5 | 5 | 2 | Risk | Sample splitting plus Neyman orthogonality already yields second-order nuisance effects and oracle rates for nonparametric target classes. |
| 6 | [Causal Q-Aggregation for CATE Model Selection](https://proceedings.mlr.press/v238/lan24a.html) | 2024 | AISTATS | theory/method | 5 | 5 | 4 | Risk | Fast \(\log M/n\) model-selection aggregation with nuisance error is occupied; ordinary ERM may only give a square-root selection rate. |
| 7 | [Minimax-Optimal Off-Policy Evaluation with Linear Function Approximation](https://proceedings.mlr.press/v119/duan20b.html) | 2020 | ICML | theory/proof | 5 | 5 | 4 | Risk | Conditional-mean operator estimation and function-class-restricted chi-square coverage are nearly minimax optimal after an ordinary OPE reduction. |
| 8 | [Stable Offline Value Function Learning with Bisimulation-based Representations](https://proceedings.mlr.press/v267/pavse25a.html) | 2025 | ICML | theory/method | 5 | 5 | 4 | Risk | KROPE directly covers representation stability and Bellman-completeness structure for offline value learning. |
| 9 | [Why Should I Trust You, Bellman? The Bellman Error is a Poor Replacement for Value Error](https://proceedings.mlr.press/v162/fujimoto22a.html) | 2022 | ICML | theory/method | 5 | 5 | 5 | A | Shows that small Bellman error need not imply small value error; validates direct stopped-score validation rather than a Bellman-loss proxy. |
| 10 | [Orthogonal Random Forest for Causal Inference](https://proceedings.mlr.press/v97/oprescu19a.html) | 2019 | ICML | theory/method | 5 | 5 | 4 | A | Combines orthogonal conditional moments with adaptive local regression; generic “orthogonal local learner” language is occupied. |
| 11 | [Pessimism Meets Risk: Risk-Sensitive Offline Reinforcement Learning](https://proceedings.mlr.press/v235/zhang24aq.html) | 2024 | ICML | theory/method | 5 | 5 | 4 | Risk | Entropic Bellman errors, coverage, pessimism, and finite-sample risk-sensitive offline learning are already combined in finite-horizon linear MDPs. |

Scores measure source quality and proximity, not acceptance probability. Numeric
evidence scores for theory papers record finite-example/empirical coverage, not
proof quality.

## Closest-Work Clusters

### Margin and policy comparison

- Already covered: plug-in sign error and regret conversion from \(L_q\) score
  error under a low-noise/margin law.
- Remaining gap: the score here is a two-branch killed exponential
  first-passage requirement observed at an irreversible stopping law.
- Positioning: Theorem 31 is a specialized handoff lemma, not a headline.

### Cross-fitted nuisance learning and model selection

- Already covered: sample-split orthogonal learning and fast Q-aggregation over
  a finite candidate library.
- Remaining gap: nuisance blocks are risk occupation, executed composition,
  killed continuation, and a learned quotient chart.
- Positioning: use these results as primitives; do not rename them EIRR theory.

### Covariate shift and restricted OPE

- Already covered: local-mass target regression, likelihood-ratio complexity,
  conditional-mean OPE, and restricted chi-square limits.
- Remaining gap: the target law is the oracle-shadow stopping occupation and the
  response is transience-amplified log-risk rather than ordinary reward.
- Positioning: any quotient rate must be compared with Doob-transformed
  restricted OPE under matched information.

### Representation stability

- Already covered: bisimulation/KROPE stability and Bellman-completeness routes.
- Remaining gap: a chart certificate that preserves the finite exponential-risk
  witness grid and improves stopped target risk, not merely global value error.
- Positioning: a lower intrinsic dimension is an assumption until the chart
  learner itself has a certified rate.

## Opportunity Map

| Cluster | Status | Open gap | Possible direction | Evidence needed | Risk |
| --- | --- | --- | --- | --- | --- |
| Generic \(L_p\)-margin transfer | covered central claim | Only first-passage specialization | Keep as lemma | Cite closest comparison theorem | Very high |
| Weighted stopped regression | crowded but open | Exact stopped law and killed-risk pseudooutcome | Prove a conditional oracle bound | Cross-fit/no-leakage theorem and target-risk validation | High |
| Finite encoder selection | covered central claim | Project-specific candidate construction | Use Q-aggregation baseline | Fast oracle penalty plus nuisance terms | High |
| Risk-observable learned chart | theory/analysis gap | No certified chart-learning rate located for this stopped exponential object | Learn witness-preserving chart on an independent fold | Distortion rate, local mass, matching lower bound | Medium-high |
| Strict advantage over transformed OPE | theory/analysis gap | Correct baselines may exploit the same quotient | Prove an information or adaptation separation | Matched-information lower bound | Very high |

## Citation and Positioning Cautions

- Cite Audibert--Tsybakov and Kim--Zubizarreta for margin comparison powers.
- Cite Pathak--Ma--Wainwright before claiming a local-mass covariate-shift rate.
- Cite Foster--Syrgkanis and Lan--Syrgkanis for cross-fitting and aggregation.
- Cite Duan--Wang and KROPE before claiming structure-dependent OPE or
  representation advantages.
- A rate comparison against an ambient local smoother is not a universal lower
  bound against an adaptive Doob/DICE/KROPE estimator.

