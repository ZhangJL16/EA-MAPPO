# Literature Search: Critical Perron Information for Stopped EIRR

Date: 2026-08-30  
Mode: standard closest-theorem and novelty audit  
Target: finite-state near-critical stopped exponential energy risk under exact
interface quotienting  
Source policy: primary journal/proceedings pages preferred; MDPI excluded

## Executive Summary

- The component mathematics is mature. Absorbing-chain inference predates modern
  RL; nearly transient Markov chains have classical spectral condition-number
  theory; killed-process/QSD work studies principal eigenfunctions and their
  perturbations; Feynman--Kac work gives variance and joint time--particle limit
  theory; SSP and risk-sensitive RL provide separate learning lower bounds.
- Therefore neither a Perron resolvent expansion nor a generic claim that
  near-critical systems are statistically hard is safe novelty.
- First-order right/left eigenvector perturbation through eigenprojectors,
  group inverses, and reduced resolvents is classical for nonnormal matrices.
  Thus \(\|(M-\rho I)^\#\|\) in Theorem 23 is a conditioning coordinate, not a
  new perturbation theorem.
- Online allocation with unknown stratum variances already has oracle-regret
  upper and lower theory, and empirical-variance-sensitive concentration is
  classical. Theorem 23 may claim only the coupled near-critical phase, not
  generic pilot allocation or variance estimation.
- The closest statistical collision is Blanchet--Glynn--Zheng: QSD estimation,
  a multidimensional CLT, slow convergence caused by an eigenvalue condition,
  and an averaging remedy. It does **not**, from its stated scope, give the
  conditional-primitive canonical gradient for a stopped exponential target.
- The closest joint-limit collision is Bérard--Del Moral--Doucet: a lognormal CLT
  when horizon and particle count grow jointly, including particle absorption.
  This is why pointwise LAN cannot be silently promoted to a
  \(\Delta_n\)-triangular-array theorem.
- Searcher inference, not a novelty verdict: the screened sources do not state
  the exact joint object of (i) source quotient coverage, (ii) conditional noise
  projected on the critical right Perron mode, (iii) semiparametric efficiency
  for the killed hitting MGF, and (iv) an irreversible stopped-decision margin.

## Ranked Paper Table

| # | Title | Year | Venue/source | Link | Type | Insight | Completeness | Numeric evidence | Overall | Closest collision |
| --- | --- | ---: | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| 1 | Analysis of a Stochastic Approximation Algorithm for Computing Quasi-stationary Distributions | 2016 | Advances in Applied Probability | [paper](https://doi.org/10.1017/apr.2016.28) | theory/proof | 5 | 5 | 3 | Risk | Principal left-eigenvector estimation, multidimensional CLT, eigenvalue-driven slow convergence, and averaging remedy. |
| 2 | A Lognormal Central Limit Theorem for Particle Approximations of Normalizing Constants | 2014 | Electronic Journal of Probability | [paper](https://doi.org/10.1214/EJP.v19-3428) | theory/proof | 5 | 5 | 2 | Risk | Joint time--particle limit with explicit particle-absorption specialization; directly warns against unproved joint asymptotics. |
| 3 | Linear Variance Bounds for Particle Approximations of Time-Homogeneous Feynman--Kac Formulae | 2012 | Stochastic Processes and their Applications | [paper](https://doi.org/10.1016/j.spa.2012.02.002) | theory/proof | 5 | 5 | 2 | Risk | Spectral/multiplicative stability and nonasymptotic variance for nonnegative kernels, including risk-sensitive control. |
| 4 | Sensitivity of the Stationary Distribution of a Markov Chain | 1994 | SIAM J. Matrix Analysis and Applications | [paper](https://doi.org/10.1137/S0895479892228900) | theory/proof | 5 | 5 | 2 | Risk | Eigenvalue-based upper and lower conditioning bounds make generic spectral sensitivity classical. |
| 5 | Uniform Stability of Markov Chains | 1994 | SIAM J. Matrix Analysis and Applications | [paper](https://doi.org/10.1137/S0895479892237562) | theory/proof | 5 | 5 | 2 | Risk | Tight perturbation bounds with an explicit nearly-transient-chain application. |
| 6 | Perturbation Theory for Killed Markov Processes and Quasi-stationary Distributions | 2025 | Advances in Applied Probability | [paper](https://doi.org/10.1017/apr.2025.3) | theory/proof | 5 | 5 | 1 | Risk | Killed-generator eigenfunction and QSD perturbation bounds; rules out generic killed-spectrum perturbation as novelty. |
| 7 | Some Problems of Statistical Inference in Absorbing Markov Chains | 1965 | Biometrika | [paper](https://doi.org/10.1093/biomet/52.1-2.127) | theory/proof | 4 | 5 | 2 | A | Establishes a long statistical-inference lineage for absorbing chains with independent replicates. |
| 8 | A Sharp First Order Analysis of Feynman--Kac Particle Models, Part I | 2018 | Stochastic Processes and their Applications | [paper](https://doi.org/10.1016/j.spa.2017.04.007) | theory/proof | 4 | 5 | 2 | A | Sharp nonasymptotic first-order bias and propagation-of-chaos expansions. |
| 9 | Theoretical Properties of Quasi-stationary Monte Carlo Methods | 2019 | Annals of Applied Probability | [paper](https://doi.org/10.1214/18-AAP1422) | theory/proof | 4 | 5 | 2 | A | QSD existence and convergence rates for killed diffusions; supplies the continuous-state foundation. |
| 10 | Reaching Goals is Hard: Settling the Sample Complexity of the Stochastic Shortest Path | 2023 | ALT/PMLR | [paper](https://proceedings.mlr.press/v201/chen23a.html) | theory/proof | 5 | 5 | 3 | Risk | Matching SSP lower/upper bounds depend on expected cost and minimum cost; generic goal-reaching hardness is occupied. |
| 11 | Sample Complexity Bounds for Stochastic Shortest Path with a Generative Model | 2021 | ALT/PMLR | [paper](https://proceedings.mlr.press/v132/tarbouriech21a.html) | theory/proof | 4 | 5 | 3 | A | PAC learning for possibly improper policies and unbounded random horizons. |
| 12 | Fast Learning Rates for Plug-in Classifiers | 2007 | Annals of Statistics | [paper](https://doi.org/10.1214/009053606000001217) | theory/proof | 5 | 5 | 2 | Risk | Margin assumptions, plug-in excess-decision rates, and minimax lower bounds make a generic margin conversion classical. |
| 13 | Markov Decision Processes with Risk-sensitive Criteria: An Overview | 2024 | Mathematical Methods of Operations Research | [paper](https://doi.org/10.1007/s00186-024-00857-0) | survey/theory | 4 | 5 | 1 | A | Maps entropic/optimized-certainty-equivalent dynamic programming and existence theory. |
| 14 | Risk-Sensitive Reinforcement Learning: Near-Optimal Risk-Sample Tradeoff in Regret | 2020 | NeurIPS | [paper](https://papers.nips.cc/paper_files/paper/2020/hash/fdc42b6b0ee16a2f866281508ef56730-Abstract.html) | theory/proof | 4 | 5 | 3 | A | Exponential risk--horizon dependence and lower bounds are already known in finite-horizon online RL. |
| 15 | Adaptive Optimal Allocation in Stratified Sampling Methods | 2010 | Methodology and Computing in Applied Probability | [paper](https://doi.org/10.1007/s11009-008-9108-0) | theory/algorithm | 5 | 5 | 3 | Risk | Adaptive proportions converge to variance-optimal allocation and the estimator attains the minimal asymptotic variance. |
| 16 | Optimal Stratification of Survey Experiments | 2023 | arXiv working paper | [paper](https://arxiv.org/abs/2111.08157) | theory/proof | 4 | 5 | 3 | Risk | Heterogeneous-cost fixed-budget design and pilot-estimated oracle efficiency make generic cost-aware plug-in allocation occupied. |
| 17 | Trajectory Stratification of Stochastic Dynamics | 2018 | SIAM Review | [paper](https://doi.org/10.1137/16M1104329) | theory/framework | 4 | 5 | 3 | A | General trajectory-fragment stratification for rare events and first-hitting functionals limits broad replay-stratification claims. |
| 18 | First-Order Perturbation Theory for Eigenvalues and Eigenvectors | 2020 | SIAM Review | [paper](https://doi.org/10.1137/19M124784X) | theory/proof | 5 | 5 | 2 | Risk | General nonnormal simple-eigenvalue/eigenvector derivatives via eigenprojectors; reduced-resolvent conditioning is established machinery. |
| 19 | Derivatives and Perturbations of Eigenvectors | 1988 | SIAM Journal on Numerical Analysis | [paper](https://doi.org/10.1137/0725041) | theory/proof | 5 | 5 | 1 | Risk | Differentiates normalized simple eigenvectors, analyzes sensitivity, and applies the formulas to perturbed finite Markov chains. |
| 20 | Adaptive Strategy for Stratified Monte Carlo Sampling | 2015 | Journal of Machine Learning Research | [paper](https://www.jmlr.org/papers/v16/carpentier15a.html) | theory/algorithm | 5 | 5 | 3 | Risk | UCB sampling learns unknown stratum standard deviations and is compared with the variance-aware oracle, with distribution-dependent and distribution-free regret bounds. |
| 21 | Minimax Number of Strata for Online Stratified Sampling: The Case of Noisy Samples | 2014 | Theoretical Computer Science | [paper](https://doi.org/10.1016/j.tcs.2014.09.027) | theory/proof | 5 | 5 | 3 | Risk | Matching-order adaptive-allocation pseudo-regret upper/lower bounds make generic unknown-variance allocation hardness occupied. |
| 22 | Empirical Bernstein Bounds and Sample Variance Penalization | 2009 | COLT | [paper](https://www.cs.mcgill.ca/~colt2009/papers/012.pdf) | theory/proof | 4 | 5 | 2 | Risk | Variance-sensitive data-dependent confidence bounds make bounded sample-variance concentration a proof tool, not a contribution. |
| 23 | Optimal Sample Allocation to Strata Using Convex Programming | 1970 | JRSS Series C | [paper](https://doi.org/10.2307/2346332) | theory/method | 5 | 5 | 2 | Risk | Convex-program allocation for several survey characteristics makes generic shared multi-estimand stratification classical. |
| 24 | An Optimal Multivariate Stratified Sampling Design Using Dynamic Programming | 2003 | Australian & New Zealand Journal of Statistics | [paper](https://doi.org/10.1111/1467-842X.00264) | theory/algorithm | 5 | 5 | 3 | Risk | Minimizes a weighted sum of several sampling variances under a shared integer allocation. |
| 25 | Optimal Off-Policy Evaluation from Multiple Logging Policies | 2021 | ICML/PMLR | [paper](https://proceedings.mlr.press/v139/kallus21a.html) | theory/proof | 5 | 5 | 3 | Risk | Gives the efficiency bound and an efficient OPE estimator under fixed stratified samples from multiple logging policies. |
| 26 | Data-Efficient Policy Evaluation Through Behavior Policy Search | 2017 | ICML/PMLR | [paper](https://proceedings.mlr.press/v70/hanna17a.html) | theory/algorithm | 4 | 5 | 3 | Risk | Directly searches behavior policies to reduce OPE estimator variance, occupying broad active-data-collection language. |

Scores are 1--5 relevance/completeness indicators for this audit, not paper-
quality judgments. `Risk` means a direct novelty collision; `A` means a mandatory
foundation or baseline.

## Collision Clusters

### 1. Spectral conditioning and killed-process perturbation

Meyer; Ipsen--Meyer; Rudolf--Wang.

- Already established: eigenvalue-controlled conditioning, nearly transient
  instability, and killed eigenfunction/QSD perturbation.
- Unsafe claim: “the Perron gap causes sensitivity.”
- Remaining specialization: the **conditional statistical noise projection** in
  the efficient influence function, including an exact degenerate phase.

### 2. Feynman--Kac and QSD statistical limits

Whiteley--Kantas--Jasra; Bérard--Del Moral--Doucet;
Chan--Del Moral--Jasra; Blanchet--Glynn--Zheng.

- Already established: particle variance control, first-order expansions,
  principal-eigenvector CLTs, and some joint asymptotics.
- Unsafe claim: “Feynman--Kac estimation becomes noisy near criticality.”
- Remaining specialization: an observed-data efficiency bound for iid source
  interface primitives, rather than a particle-algorithm variance theorem.

### 3. Goal-reaching and exponential-risk learning

Chen et al.; Tarbouriech et al.; Fei et al.; Bäuerle--Jaśkiewicz.

- Already established: SSP hardness, improper-policy complications, and
  exponential risk--horizon cost.
- Unsafe claim: “random hitting times or exponential utility make RL hard.”
- Remaining specialization: the source-design/quotient/Perron coefficient for
  the charger-hitting functional and its relation to a stopped commitment rule.

### 4. Boundary decisions

Audibert--Tsybakov.

- Already established: margin assumptions convert regression estimation into
  fast plug-in decision rates with minimax lower bounds.
- Unsafe claim: a generic margin-to-decision conversion.
- Remaining specialization: a nonregular stopped decision whose estimation
  scale itself changes phase with quotient coverage and the killed spectrum.

### 5. Adaptive and trajectory stratification

Étoré--Jourdain; Cytrynbaum; Dinner et al.

- Already established: adaptive variance-optimal stratum proportions,
  heterogeneous-cost designs, pilot consistency, and trajectory stratification
  for rare-event/hitting quantities.
- Unsafe claim: optimal or adaptive allocation based on estimated within-stratum
  variance.
- Remaining specialization: prove that the killed-resolvent singularity cancels
  from normalized critical-mode priorities, yielding allocation consistency even
  when the pilot lies below the \(m\Delta^2\) risk-value scale.

### 6. Nonnormal eigenvector perturbation and moment estimation

Greenbaum--Li--Overton; Meyer--Stewart; Maurer--Pontil.

- Already established: first-order left/right eigenvector derivatives for a
  general nonnormal matrix, group-inverse/eigenprojector sensitivity formulas,
  and empirical-variance-sensitive concentration for bounded observations.
- Unsafe claim: “we introduce effective Perron separation” or “we derive a new
  variance-aware pilot estimator.”
- Remaining specialization: the *joint* phase coordinate in Theorem 23,
  combining reduced-resolvent conditioning, directional covariance anisotropy,
  quotient bias, projected-noise moments, and the critical-proxy remainder.

### 7. Shared multi-query design and OPE data collection

Huddleston--Claypool--Hocking; Khan--Khan--Ahsan; Kallus--Saito--Uehara;
Hanna et al.

- Already established: convex-program compromise allocations for multiple
  characteristics, weighted sums of sampling variances, efficient OPE under
  multiple logging strata, and behavior-policy search for OPE variance.
- Unsafe claim: “we introduce a shared replay distribution for many targets” or
  “we optimize data collection for OPE.”
- Remaining specialization: an allocation objective induced by the *stopped
  margin law*, with Perron-mode sensitivities and a critical cancellation/failure
  theorem. Its KKT rule can be useful algorithmically but is not standalone
  novelty without the coupled phase or a matching adaptive/lower result.

## Opportunity Map

| Candidate claim | Collision verdict | What would be needed to survive |
| --- | --- | --- |
| Perron resolvent blows up as \(1/\Delta\) | occupied | Treat only as lemma. |
| Raw/log EIRR variance has critical exponents | composition-level candidate | Prove exact observed-data efficiency constants and both noise phases; do not call exponent alone novel. |
| Stopped margin inherits a normal error floor | occupied pointwise | Add and prove uniform triangular-array LAN or explicitly stay pointwise. |
| Optimal replay/source sampling follows critical eigenmodes | generic oracle and adaptive Neyman allocation are occupied | Only the critical-scale cancellation and its precise failure modes can remain a candidate. |
| Learned quotient removes raw-action positivity | high-risk candidate | Prove identifiability and finite-sample recovery without outcome-selected pooling. |
| Collapsing effective Perron separation causes an allocation lower bound | perturbation tool occupied; exact allocation slice is composition-level candidate | Keep the local Le Cam construction and avoid claiming group-inverse sensitivity itself as new. |
| Vanishing projected noise has one universal sample exponent | false without a distribution class | Retain standardized fourth moment/range, or state a narrower parametric family. |
| One replay allocation minimizes the full stopped-boundary radius | generic multi-estimand design occupied | Derive the margin-powered Perron objective, critical cancellation, and learned adaptive guarantee; do not sell convex allocation itself. |
| Pareto improvement from the theorem | unproved | Requires navigation recovery and the predeclared stranding--throughput Gate. |

## Audit Verdict

Theorem 20 is useful and mathematically correct after separating pointwise from
uniform triangular-array LAN. It is **not yet an oral-level standalone theorem**.
The safest contribution claim is a conditional composition: exact quotient-
aware semiparametric information, Perron-mode noise phase split, and a stopped
decision consequence under additional uniform LAN. To raise it to oral level,
the algorithm-generating route must go beyond adaptive Neyman allocation. The
surviving candidate is a **critical-scale separation theorem** showing that
normalized design learning avoids the \(\Delta^{-2}\) risk-value cost while
stating exactly when secondary eigengap collapse, projected-noise degeneracy, or
learned quotient error destroys that separation. A plain extra neural loss is
not enough.

Theorem 23 now supplies a conditioned sufficient phase and a locally matching
effective-separation/anisotropy lower slice. The audit classifies its individual
ingredients as occupied. Its surviving candidate is the coupled phase

\[
\frac{(1+\chi)(m^{-1/2}+\tau_M)}{\mathfrak g}
+\sqrt{\frac{\varkappa}{m}}+\frac{b^2}{m}
+\tau_\sigma+\frac{\Delta}{\sigma_*^2}\to0,
\]

together with a stopped-decision consequence. This is a searcher inference,
not an exhaustive novelty or acceptance verdict.

## Benchmark Search

Not applicable as the primary deliverable is a mathematical closest-theorem
audit. The SSP and risk-OPE papers above determine later baselines; they do not
replace the original project's predeclared navigation Gate.
