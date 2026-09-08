# Literature Map: Executed-Interface Risk Resolvents

Date: 2026-08-30  
Mode: standard search  
Question: Which mathematical results can support a theorem-level treatment of
return-to-charge energy under policy--safety composition shift, and which parts
are already prior art?

## Decision summary

The literature rules out three weak novelty claims: exponential utility or an
exponential Bellman equation is not new; EVaR for transient total reward is not
new; and Itakura--Saito regression for exponential value learning is not new.
The viable gap is the interaction of four objects that existing papers usually
treat separately:

1. first-passage total resource in a stochastic shortest-path process;
2. an executed-action interface induced by a policy and a safety filter;
3. risk-tilted, rather than ordinary, occupancy under composition shift; and
4. an irreversible stopping boundary whose error is locally concentrated.

The proposed theory should therefore be positioned as a transfer,
identifiability, and decision-stability theory for a killed Feynman--Kac
operator. It must not be sold as a new risk measure.

## Final screened set

| Work | Venue / year | Role | Key reusable result | Relevance / rigor / proximity | Route |
|---|---|---|---|---|---|
| Bertsekas & Tsitsiklis, *An Analysis of Stochastic Shortest Path Problems* | Mathematics of Operations Research, 1991 | theory/proof | Proper-policy SSP assumptions, value equations, and convergence boundaries for undiscounted total cost | 5 / 5 / 2 | A |
| Van Dijk & Puterman, *Perturbation Theory for Markov Reward Processes* | Advances in Applied Probability, 1988 | theory/proof | Perturbation bounds including first-passage total reward | 5 / 4 / 3 | A |
| Piunovskiy & Zhang, *Continuity of Occupation Measures for Absorbing MDPs* | Applied Mathematics & Optimization, 2024 | theory/proof | Uniform absorption as the condition behind occupation-measure continuity | 4 / 5 / 2 | A |
| Ruszczynski & Shapiro, *Conditional Risk Mappings* | Mathematics of Operations Research, 2006 | theory/proof | Dynamic conditional risk construction and time consistency | 5 / 5 / 2 | B |
| Iyengar, *Robust Dynamic Programming* | Mathematics of Operations Research, 2005 | theory/proof | Rectangular uncertainty and robust Bellman recursion | 5 / 5 / 3 | B |
| Baeuerle & Jaskiewicz, *MDPs with Risk-Sensitive Criteria: An Overview* | Mathematical Methods of Operations Research, 2024 | synthesis/theory | Existing entropic, OCE, CVaR, and infinite-horizon risk-sensitive theory | 5 / 4 / 4 | Risk |
| Murthy, Moharrami & Srikant, *Modified Policy Iteration for Exponential Cost Risk Sensitive MDPs* | L4DC, 2023 | theory/algorithm | Multiplicative Bellman equation and convergence of risk-sensitive MPI | 4 / 5 / 4 | Risk |
| Su, Petrik & Grand-Clement, *Risk-Averse Total-Reward MDPs with ERM and EVaR* | AAAI, 2025 | theory/algorithm | ERM/EVaR in transient total-reward MDPs | 5 / 4 / 5 | Risk |
| Udovichenko et al., *Risk-Averse RL with Itakura--Saito Loss* | arXiv, 2025 | theory/algorithm | Stable positive-value learning loss for exponential Bellman targets | 4 / 4 / 5 | Risk |
| Rowland et al., *An Analysis of Quantile Temporal-Difference Learning* | JMLR, 2024 | theory/proof | QTD operators need not be contractions and may have multiple fixed points | 5 / 5 / 4 | Risk |
| Khan et al., *Off-Policy Evaluation Beyond Overlap* | ICML, 2024 | theory/method | Identification can sometimes survive support failure only with extra structure | 4 / 5 / 4 | C |
| Chen & Jiang, *Offline RL under Value and Density-Ratio Realizability* | UAI, 2022 | theory/method | Density-ratio realizability, coverage, and gap structure | 4 / 5 / 4 | C |
| Barber et al., *Conformal Prediction Beyond Exchangeability* | Annals of Statistics, 2023 | theory/method | Coverage degradation under non-exchangeability and weighted constructions | 4 / 5 / 3 | C |
| Tibshirani et al., *Conformal Prediction under Covariate Shift* | NeurIPS, 2019 | theory/method | Weighted conformal coverage under a specified density ratio | 4 / 5 / 3 | C |
| Audibert & Tsybakov, *Fast Learning Rates for Plug-in Classifiers* | Annals of Statistics, 2007 | theory/proof | Margin/noise conditions convert estimation error into fast decision rates | 5 / 5 / 3 | D |
| Belomestny, *Pricing Bermudan Options by Nonparametric Regression* | Finance and Stochastics, 2011 | theory/proof | Uniform continuation-value error plus a boundary/margin condition yields fast stopping-value rates | 4 / 5 / 3 | D |
| Belomestny, *On the Rates of Convergence of Simulation-Based Optimization Algorithms for Optimal Stopping Problems* | Annals of Applied Probability, 2011 | theory/proof | Nonasymptotic and optimal convergence rates for learned stopping policies | 4 / 5 / 3 | D |
| Dai et al., *CoinDICE: Off-Policy Confidence Interval Estimation* | NeurIPS, 2020 | theory/method | Behavior-agnostic density-ratio OPE with asymptotic and finite-sample value confidence intervals | 5 / 5 / 5 | Risk |
| Essakine & Vernade, *Tight Sample Complexity Bounds for Entropic Best Policy Identification* | COLT, 2026 | theory/algorithm | Matching exponential-risk sample complexity using sharp exponential-utility concentration and a stopping rule | 5 / 5 / 4 | Risk |
| Maurer & Pontil, *Empirical Bernstein Bounds and Sample-Variance Penalization* | COLT, 2009 | theory/proof | Data-dependent variance-sensitive confidence radii with bounded observations | 5 / 5 / 3 | Risk |
| Chen et al., *Policy-Conditioned Environment Models* | ICML, 2024 | method/benchmark | Policy conditioning for environment-model transfer | 4 / 4 / 5 | Risk |
| Farahmand et al., *Value-Aware Loss Function for Model-Based Reinforcement Learning* | NeurIPS, 2018 | theory/method | One-step likelihood is not aligned with downstream decision value | 4 / 4 / 4 | Risk |
| Choudhry et al., *Risk-Aware Energy-Efficient Motion Planning for UAVs* | ICRA, 2021 | application/method | CVaR-style UAV energy risk is already an established application direction | 3 / 4 / 4 | E |
| Fouad et al., *Energy Sufficiency Control Barrier Functions* | arXiv, 2023 | theory/control | Energy sufficiency can be embedded in a barrier-function constraint | 3 / 3 / 4 | E |
| Kontoyiannis & Meyn, *Spectral Theory and Limit Theorems for Geometrically Ergodic Markov Processes* | Annals of Applied Probability, 2003 | theory/proof | Multiplicative Poisson equations, spectral theory, and exponential additive functionals under geometric ergodicity | 5 / 5 / 4 | Risk |
| Borkar, *A Sensitivity Formula for Risk-Sensitive Cost and the Actor--Critic Algorithm* | Systems & Control Letters, 2001 | theory/proof | Risk-sensitive sensitivity formula and actor--critic construction for finite Markov chains | 5 / 4 / 4 | Risk |
| Anantharam & Borkar, *A Variational Formula for Risk-Sensitive Reward* | SIAM Journal on Control and Optimization, 2017 | theory/proof | Donsker--Varadhan/Perron--Frobenius occupation formulation for risk-sensitive reward | 5 / 5 / 4 | Risk |
| Chatterjee & Diaconis, *The Sample Size Required in Importance Sampling* | Annals of Applied Probability, 2018 | theory/proof | Approximately \(\exp(D(\nu\|\mu))\) samples are necessary and sufficient for importance sampling | 5 / 5 / 5 | Risk |
| Fei et al., *Risk-Sensitive Reinforcement Learning: Near-Optimal Risk--Sample Tradeoff in Regret* | NeurIPS, 2020 | theory/algorithm | Exponential dependence on risk sensitivity and horizon is unavoidable | 5 / 5 / 5 | Risk |
| Fei et al., *Exponential Bellman Equation and Improved Regret Bounds for Risk-Sensitive RL* | NeurIPS, 2021 | theory/algorithm | Exponential Bellman transformation and near-matching risk-sensitive regret analysis | 5 / 5 / 5 | Risk |
| Karmakar & Bhatnagar, *On Tight Bounds for Function Approximation Error in Risk-Sensitive RL* | Systems & Control Letters, 2021 | theory/proof | Perron--Frobenius perturbation bounds for exponential-value approximation | 4 / 4 / 5 | Risk |
| Duan, Jia & Wang, *Minimax-Optimal Off-Policy Evaluation with Linear Function Approximation* | ICML, 2020 | theory/method | Restricted \(\chi^2\) occupancy mismatch controls upper and lower OPE limits | 5 / 5 / 4 | C |
| Nachum et al., *DualDICE* | NeurIPS, 2019 | theory/method | Adjoint Bellman moment and saddle estimation of stationary density ratios | 5 / 5 / 5 | Risk |
| Zhang et al., *Pessimism Meets Risk: Risk-Sensitive Offline RL* | ICML, 2024 | theory/algorithm | Offline entropic-risk exponential Bellman residuals, coverage conditions, and finite-sample pessimism | 5 / 5 / 5 | Risk |
| Granados & Pacheco, *Risk-Sensitive Exponential Actor Critic* | AAAI, 2026 | theory/algorithm | Exponential-twisted state occupancy in on/off-policy risk-sensitive gradients and stabilized exponential critics | 4 / 4 / 5 | Risk |

Scores are on a 1--5 scale: relevance to the target problem, mathematical rigor,
and proximity/novelty risk. `Risk` means a claim that must be explicitly avoided;
`A`--`E` denote the opportunity clusters below.

## Opportunity clusters

### A. Absorbing chains, SSPs, and resolvents

Known: properness or uniform absorption is needed for stable infinite-horizon
first-passage values. The ordinary resolvent

\[
N=(I-Q)^{-1}=\sum_{t\ge 0}Q^t
\]

turns local expected-cost error into an occupancy-weighted global error.

Gap: the corresponding exact perturbation identity for the *risk-tilted killed
operator* can expose which executed interfaces dominate tail energy, rather than
mean energy.

### B. Risk-sensitive control and Feynman--Kac operators

Known: exponential utility, multiplicative Bellman recursions, ERM/EVaR, robust
duality, and risk-sensitive policy iteration are established.

Gap: these works do not by themselves give a model-transfer theorem under an
unseen policy--safety-filter composition, nor an interface-specific impossibility
theorem tied to the target first-passage tail.

### C. Coverage, density ratios, and calibration

Known: ordinary occupancy ratios correct policy distribution shift; lack of
overlap can sometimes be overcome only by additional bridge, realizability, or
structural assumptions. Weighted conformal likewise needs a valid shift model.

Gap: ordinary occupancy can look benign while exponential path weights make the
risk-relevant occupancy singular or extremely concentrated. This motivates a
new risk-tilted executed-interface concentrability coefficient.

### D. Optimal stopping and boundary margins

Known: plug-in decisions can converge faster than global regression error when
little mass lies near the decision boundary. Optimal-stopping errors likewise
depend on continuation-boundary regularity.

Gap: connect an upper return-energy requirement learned through a killed
Feynman--Kac model to a one-way `CONTINUE/RETURN` stopping rule and derive both
disagreement and regret rates along sequential state occupancy.

### E. UAV energy safety

Known: CVaR planning and energy-sufficiency barrier functions already exist.

Gap: neither replaces a theorem about tail Resource-to-Go transfer through an
executed safety-filter interface. They are application baselines, not the core
mathematical novelty.

## Candidate contribution package

Working name: **Executed-Interface Risk Resolvent (EIRR)**.

1. Executed-interface quotient theorem: for a sufficient Markov state, a policy
   and safety filter affect return-energy law only through their induced
   executed Feynman--Kac operator.
2. Risk-resolvent perturbation identity: exact decomposition of target log-MGF
   error into risk-tilted occupancy times local operator error.
3. Risk-tilted overlap theorem and two-model impossibility result: characterize
   when source-only learning can and cannot identify target tail energy.
4. Certified chance-bound corollary: a high-confidence upper bound on learned
   log-MGF yields a return-energy chance guarantee through Chernoff/EVaR.
5. Irreversible boundary theorem: local estimator error plus a sequential margin
   condition yields disagreement and excess-resource/stranding regret rates.
6. Exact task--return composition identity: avoid invalid addition of marginal
   quantiles without an independence assumption.

## Oral-level bar

The package is only plausibly oral-level if all of the following survive proof
and counterexample work:

- at least one theorem is strictly sharper than an ordinary occupancy or
  total-variation bound because it uses the risk-tilted resolvent;
- the no-overlap theorem separates ordinary support from tail-relevant support;
- the result handles policy and safety-filter composition without treating their
  IDs as magic side information;
- the stopping theorem changes an algorithmic design choice, loss weight, or
  data-collection rule;
- experiments include a case where mean prediction is accurate but tail-risk
  coverage or return timing fails, and EIRR predicts that failure;
- the theory generalizes beyond one UAV simulator and one filter family.

Without those items this is a sound workshop/theory appendix direction, not yet
an oral-level contribution.

The theorem-level follow-up audit in `closest-theorem-audit.md` further narrows
this judgment: none of the nine finite-state EIRR statements is individually a
safe novelty claim. The only plausible contribution is a new joint result for
unlabelled source-to-target transfer through an executed safety interface,
killed first-passage resource, and irreversible return decisions.

## Primary links

- [Stochastic shortest path analysis](https://pubsonline.informs.org/doi/pdf/10.1287/moor.16.3.580)
- [First-passage reward perturbation](https://www.cambridge.org/core/journals/advances-in-applied-probability/article/abs/perturbation-theory-for-markov-reward-processes-with-applications-to-queueing-systems/418639DD06032EF07077F5C35ECA1116)
- [Absorbing-MDP occupation continuity](https://link.springer.com/article/10.1007/s00245-024-10124-7)
- [Conditional risk mappings](https://pubsonline.informs.org/doi/10.1287/moor.1060.0204)
- [Robust dynamic programming](https://pubsonline.informs.org/doi/abs/10.1287/moor.1040.0129)
- [Risk-sensitive MDP overview](https://link.springer.com/article/10.1007/s00186-024-00857-0)
- [Exponential-cost modified policy iteration](https://proceedings.mlr.press/v211/murthy23a.html)
- [Risk-averse total-reward MDPs](https://ojs.aaai.org/index.php/AAAI/article/view/34275)
- [Itakura--Saito loss for risk-averse RL](https://arxiv.org/abs/2505.16925)
- [Quantile TD analysis](https://jmlr.org/papers/v25/23-0154.html)
- [OPE beyond overlap](https://proceedings.mlr.press/v235/khan24b.html)
- [Offline RL and density-ratio realizability](https://proceedings.mlr.press/v180/chen22g.html)
- [Conformal prediction beyond exchangeability](https://projecteuclid.org/journals/annals-of-statistics/volume-51/issue-2/Conformal-prediction-beyond-exchangeability/10.1214/23-AOS2276.pdf)
- [Conformal prediction under covariate shift](https://arxiv.org/abs/1904.06019)
- [Plug-in classifier margin rates](https://arxiv.org/abs/0708.2321)
- [Optimal stopping boundary analysis](https://arxiv.org/abs/0909.3570)
- [Policy-conditioned environment models](https://proceedings.mlr.press/v235/chen24g.html)
- [Value-aware model learning](https://papers.neurips.cc/paper_files/paper/2018/hash/7a2347d96752880e3d58d72e9813cc14-Abstract.html)
- [Risk-aware UAV energy planning](https://arxiv.org/abs/2105.15189)
- [Energy-sufficiency barrier functions](https://arxiv.org/abs/2306.15115)
- [Multiplicative spectral theory](https://arxiv.org/abs/math/0209200)
- [Risk-sensitive sensitivity formula](https://www.sciencedirect.com/science/article/pii/S0167691101001529)
- [Risk-sensitive variational formula](https://arxiv.org/abs/1501.00676)
- [Importance-sampling sample size](https://projecteuclid.org/journals/annals-of-applied-probability/volume-28/issue-2/The-sample-size-required-in-importance-sampling/10.1214/17-AAP1326.pdf)
- [Near-optimal risk--sample tradeoff](https://proceedings.neurips.cc/paper/2020/hash/fdc42b6b0ee16a2f866281508ef56730-Abstract.html)
- [Exponential Bellman equation](https://papers.nips.cc/paper/2021/hash/ab6439fa2daf0246f92eea433bca5ac4-Abstract.html)
- [Risk-sensitive function-approximation error](https://www.sciencedirect.com/science/article/pii/S0167691121000293)
- [Minimax-optimal OPE](https://proceedings.mlr.press/v119/duan20b.html)
- [DualDICE](https://proceedings.neurips.cc/paper_files/paper/2019/hash/cf9a242b70f45317ffd281241fa66502-Abstract.html)
- [Risk-sensitive offline RL](https://proceedings.mlr.press/v235/zhang24aq.html)
- [Risk-sensitive exponential actor--critic](https://ojs.aaai.org/index.php/AAAI/article/view/39280)
