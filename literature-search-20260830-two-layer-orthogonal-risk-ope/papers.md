# Literature Search: Two-Layer Orthogonal Risk OPE

Date: 2026-08-30  
Search purpose: closest-theorem and novelty audit for Theorem 17 and the next
energy-risk theorem  
Target venue/family: ICLR oral-level theory, with statistics/control foundations  
Source-quality policy: applied; primary proceedings/journal pages prioritized

## Summary

- Closest-work clusters: semiparametric DRL; marginalized density-ratio OPE;
  off-policy risk/CDF estimation; structural and off-environment transfer;
  exponential-risk learning.
- Strongest collision: Huang et al. already derive a doubly robust estimator for
  the return CDF in MDPs, efficiency under correct models, and matching-order
  minimax risk-estimation bounds. Kallus--Uehara already supply the efficient
  influence-function route and fourth-root product-rate logic for MDP OPE.
- Theorem 17 status: useful algebraic estimator-design lemma, not standalone
  novelty.
- Remaining candidate: the canonical gradient and information bound for a
  **randomly stopped killed Feynman--Kac functional under a known target execution
  composition and source primitive sampling**, followed by a quotient-coverage
  phase transition and an irreversible-boundary minimax consequence.

## Paper Table

| # | Title | Year | Venue/source | Link | Type | Insight | Completeness | Numeric evidence | Overall | Notes |
| --- | --- | ---: | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| 1 | Double Reinforcement Learning for Efficient Off-Policy Evaluation in Markov Decision Processes | 2020 | JMLR | [paper](https://www.jmlr.org/papers/v21/19-827.html) | theory/proof | 5 | 5 | 4 | Risk | Efficient influence functions, cross-fitting, fourth-root nuisance products, and double robustness directly collide with generic claims around Theorem 17. |
| 2 | Doubly Robust Off-policy Value Evaluation for Reinforcement Learning | 2016 | ICML/PMLR | [paper](https://proceedings.mlr.press/v48/jiang16.html) | pure method | 5 | 4 | 4 | A | Foundational sequential DR estimator and hardness analysis; stochastic-shortest-path experiments do not by themselves establish random-horizon Feynman--Kac theory. |
| 3 | Off-Policy Risk Assessment for Markov Decision Processes | 2022 | AISTATS/PMLR | [paper](https://proceedings.mlr.press/v151/huang22b.html) | theory/proof | 5 | 5 | 4 | Risk | Doubly robust return-CDF estimation, efficiency under correct specification, and matching-order minimax risk bounds are the closest risk-functional collision. |
| 4 | Minimax Weight and Q-Function Learning for Off-Policy Evaluation | 2020 | ICML/PMLR | [paper](https://proceedings.mlr.press/v119/uehara20a.html) | theory/proof | 5 | 5 | 4 | Risk | Marginalized weights and Q functions can be combined doubly robustly; discriminator-class dependence is directly relevant to a quotient witness class. |
| 5 | DualDICE: Behavior-Agnostic Estimation of Discounted Stationary Distribution Corrections | 2019 | NeurIPS | [paper](https://papers.nips.cc/paper_files/paper/2019/hash/cf9a242b70f45317ffd281241fa66502-Abstract.html) | pure method | 5 | 4 | 4 | A | Establishes behavior-agnostic stationary density-ratio estimation; a required baseline after the known Doob/killed-chain reduction. |
| 6 | CoinDICE: Off-Policy Confidence Interval Estimation | 2020 | NeurIPS | [paper](https://papers.nips.cc/paper_files/paper/2020/hash/6aaba9a124857622930ca4e50f5afed2-Abstract.html) | method + benchmark | 4 | 5 | 4 | A | Gives asymptotic and finite-sample OPE confidence intervals from generalized estimating equations; challenges any generic EIRR-certificate novelty. |
| 7 | Minimax-Optimal Off-Policy Evaluation with Linear Function Approximation | 2020 | ICML/PMLR | [paper](https://proceedings.mlr.press/v119/duan20b.html) | theory/proof | 5 | 5 | 3 | Risk | CME/FQI equivalence, restricted-chi-square instance dependence, and near-matching minimax bounds directly delimit quotient-rate novelty. |
| 8 | Semiparametrically Efficient Off-Policy Evaluation in Linear Markov Decision Processes | 2023 | ICML/PMLR | [paper](https://proceedings.mlr.press/v202/xie23d.html) | theory/proof | 5 | 5 | 3 | Risk | Canonical efficiency bound, efficient estimator, asymptotic normality, and confidence intervals in structured linear MDPs raise the bar for a new efficiency theorem. |
| 9 | Marginalized Importance Sampling for Off-Environment Policy Evaluation | 2023 | CoRL/PMLR | [paper](https://proceedings.mlr.press/v229/katdare23a.html) | method + benchmark | 4 | 4 | 4 | Risk | Uses a simulator intermediate and two separately learned density-ratio factors under environment shift; close to the proposed execution/primitive factorization. |
| 10 | Understanding the Curse of Horizon in Off-Policy Evaluation via Conditional Importance Sampling | 2020 | ICML/PMLR | [paper](https://proceedings.mlr.press/v119/liu20a.html) | theory/proof | 5 | 4 | 3 | A | Explains when Markov/marginal conditioning removes or retains horizon dependence; needed for any random-hitting-time rate claim. |
| 11 | Risk-Sensitive Reinforcement Learning: Near-Optimal Risk-Sample Tradeoff in Regret | 2020 | NeurIPS | [paper](https://papers.nips.cc/paper_files/paper/2020/hash/fdc42b6b0ee16a2f866281508ef56730-Abstract.html) | theory/proof | 5 | 5 | 4 | Risk | Proves exponential dependence on risk parameter and horizon is unavoidable in episodic exponential-utility RL; supports a genuine risk-severity lower factor but not OPE or stopping boundaries. |
| 12 | Doubly Robust Distributionally Robust Off-Policy Evaluation and Learning | 2022 | ICML/PMLR | [paper](https://proceedings.mlr.press/v162/kallus22a.html) | theory/proof | 5 | 5 | 4 | A | Extends double robustness to environment distribution shift and worst-case objectives; generic distribution-shift robustness is already occupied. |
| 13 | Selective Machine Learning of Doubly Robust Functionals | 2024 | Biometrika | [paper](https://academic.oup.com/biomet/article/111/2/517/7282948) | theory/proof | 4 | 5 | 3 | A | Places nuisance-product cancellation in a broad semiparametric class; confirms that product-bias algebra and cross-fitting are statistical foundations, not RL-specific novelty. |

## Clusters

### Cluster 1: Semiparametric DRL and efficient influence functions

- Representative papers: Kallus--Uehara; Jiang--Li; Uehara--Huang--Jiang;
  Xie--Yang--Zhang; Bonvini et al.
- Already solved: double robustness, nuisance-product remainder, fourth-root
  sufficient rates, canonical efficiency analysis, and structured linear-MDP
  inference for standard policy value.
- Remaining gap: derive the tangent space and canonical gradient for the stopped
  multiplicative charger-hitting functional under the project's two-layer data
  law, rather than merely writing another unbiased score.
- Rescue route: prove that random killing, known composition, and shared primitive
  produce a different information geometry or a smaller efficient variance than
  generic trajectory/CDF OPE.

### Cluster 2: Density ratios, confidence intervals, and horizon

- Representative papers: DualDICE, CoinDICE, MWL/MQL, conditional IS.
- Already solved: behavior-agnostic marginalized ratios, estimating-equation
  confidence intervals, and several horizon-variance characterizations.
- Remaining gap: non-normalized exponential risk occupation before a random
  charger-hitting time, especially near the spectral/transience boundary.
- Rescue route: establish an information-bound phase transition controlled by a
  risk-resolvent/quotient coefficient, then show it is necessary for irreversible
  return decisions.

### Cluster 3: Risk functionals and exponential utility

- Representative papers: Huang et al.; Fei et al.; Kallus et al.
- Already solved: doubly robust CDF/risk estimation for episodic MDPs, minimax
  risk-functional lower bounds, and unavoidable exponential risk--horizon cost in
  online learning.
- Remaining gap: random-horizon charger hitting plus composition transfer through
  a shared executed-action primitive. This is narrower than generic risk OPE.
- Rescue route: compare directly against OPRA's DR-CDF estimator. Win only if the
  Feynman--Kac sufficient statistic gives a provably smaller rate/variance or if
  the stopped decision admits a sharper task-specific lower bound.

### Cluster 4: Structural and off-environment transfer

- Representative papers: Duan et al.; Katdare et al.
- Already solved: restricted-function-class divergence and two-factor
  off-environment ratio decompositions.
- Remaining gap: target execution is queryable while primitive physics is shared,
  and only risk-observable quotient directions matter.
- Rescue route: prove a strict raw-support-versus-quotient-support separation in
  the full stopped nonlinear model, not only the fixed linear Gaussian slice.

## Opportunity Map

| Cluster | Status | Open gap | Possible direction | Evidence needed | Risk |
| --- | --- | --- | --- | --- | --- |
| Generic two-layer DR score | covered central claim | Project notation and random killing only | Keep as lemma/baseline | Exact reduction to existing EIF/DRL | Very high |
| DR estimation of risk functionals | covered central claim | Fixed/episodic CDF versus random killed MGF | Stopped Feynman--Kac EIF | Tangent-space derivation and OPRA comparison | Very high |
| Risk occupation near transience boundary | theory/analysis gap | Variance/identification as spectral radius approaches one | Efficiency phase transition | Upper/lower information bound | Medium |
| Queryable execution plus shared primitive | crowded but open | Known composition may remove one nuisance layer | Rao--Blackwellized canonical gradient | Efficiency comparison theorem | Medium |
| Quotient support for stopped risk | theory/analysis gap | Existing restricted OPE is mostly additive/discounted | Nonlinear quotient necessity/sufficiency | General-state upper and packing lower bound | Medium-high |
| Irreversible return boundary | theory/analysis gap | OPE error rarely ends in a stopped action-loss lower bound | Boundary-weighted local asymptotic minimax theorem | Margin law, Le Cam/LAN reduction | Medium |

## Citation And Positioning Cautions

- Never describe Theorem 17's double robustness or product-rate cancellation as
  new; cite DRL/MWL-MQL and general doubly robust functionals.
- Never claim first doubly robust risk OPE in MDPs; Huang et al. 2022 has that
  claim and matching-order lower bounds.
- Never claim the quotient coefficient alone as new; Duan et al. already connect
  restricted chi-square divergence, CME/FQI, upper bounds, and minimax limits.
- The paper can still claim a new theorem only if random killed first passage,
  queryable execution composition, and irreversible stopping alter the canonical
  gradient, efficiency boundary, or minimax decision consequence.

