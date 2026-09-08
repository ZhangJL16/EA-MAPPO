# Closest-Theorem Audit for EIRR

Date: 2026-08-30  
Mode: standard theorem-level novelty audit  
Evidence boundary: public paper pages and inspectable primary PDFs; no private
project prose was used as a search query.

## Verdict

None of the current finite-state EIRR statements is individually a defensible
novelty claim. The mathematical foundations are strong but mostly classical or
direct specializations. A contribution remains plausible only as a joint theorem
for **unlabelled source-to-target transfer of killed first-passage resource
through a queryable policy--safety execution interface, with an irreversible
stopping consequence**.

## Statement-by-statement audit

| Project statement | Closest established result | Coverage judgment | Safe paper role | Required differentiation |
| --- | --- | --- | --- | --- |
| Executed-interface path-law identifiability | Ionescu--Tulcea path construction; standard controlled-kernel factorization | Covered foundation | Definition/lemma | Prove a nontrivial quotient or minimal sufficient interface under composition shift |
| Non-identification outside source support | Classical two-point/Le Cam reasoning; OPE beyond overlap requires extra bridge structure | Covered foundation | Impossibility lemma | Make the target set risk-relevant and executed-interface-specific; compare against bridge assumptions |
| Finite-horizon Wasserstein transport | Simulation/perturbation lemmas under coupling and continuation regularity | Covered foundation | Supporting proposition | Derive a sharper risk-resolvent or boundary-local quantity than global transport |
| Proper-SSP truncation | SSP and absorbing-chain tail bounds | Covered foundation | Technical lemma | None; cite and use |
| Deterministic return-threshold stability | Plug-in classifier and optimal-stopping margin literature | Elementary/currently weak | Supporting lemma | Prove sequential first-disagreement and excess mission-loss rates under stopped occupancy |
| Killed Feynman--Kac resolvent | Multiplicative Poisson/Feynman--Kac spectral theory | Classical | Mathematical setup | General-state exponential drift or a new transfer consequence, not the inverse formula |
| Exact risk-occupation residual identity | Exponential Bellman error decompositions in risk-sensitive offline RL; twisted occupancies in risk-sensitive policy gradients; adjoint Bellman ratios in DICE | Strong collision | Specialization/bridge lemma | Show why killed first-passage source-to-target execution composition requires a different measure or yields a sharper certificate |
| Learned-model resolvent perturbation | Markov reward perturbation; Perron--Frobenius function-approximation error bounds | Covered algebra | Supporting lemma | Establish minimax sharpness under the proposed interface class |
| Zero-support minimax lower bound | Standard two-point identification lower bound | Covered technique | Explicit counterexample | Couple it to queryable components and show which additional interventions restore identification |
| Perfect ordinary support but large risk ratio | Exponential twisting and importance-weight degeneracy | Known phenomenon, useful construction | Diagnostic proposition | Quantify a separation from every ordinary concentrability coefficient under matched horizon |
| Common-support exponential trajectory lower bound | Chatterjee--Diaconis exponential importance-sampling law; Fei et al. exponential risk--sample lower bound | Central claim covered in broader forms | Corollary/example only | A sharper first-passage/interface minimax rate with upper bound, or remove novelty emphasis |
| Simultaneous-grid chance certificate | Chernoff/EVaR plus simultaneous confidence | Classical | Corollary | Construct valid learned upper log-MGF confidence under target composition shift |
| Cross-fitted clipped residual certificate | Clipped importance weighting plus Bernstein concentration | Standard | Finite-sample implementation lemma | Prove nuisance rates for the non-normalized killed risk ratio and dependent trajectories |
| Irreversible first-disagreement/Pareto stability | Sequential coupling, plug-in classification margins, and optimal-stopping boundary analysis | Standard technique with project-specific semantics | Decision bridge theorem | Compose a non-vacuous source-to-target risk-estimation rate with the stopped boundary law and formal Pareto statistics |
| Simultaneous EIRR certificate to conservative headroom | CoinDICE-style off-policy confidence intervals; risk-sensitive offline pessimism; Belomestny stopping-margin rates | Individual ingredients strongly covered; exact joint killed-interface composition not located | End-to-end bridge theorem, not standalone novelty | Use the finite primitive plug-in rate from Theorem 14 or derive a strictly stronger learned-Doob/continuous-interface rate; do not leave \(a_n\) unexplained |
| Finite-interface Bernstein--resolvent transfer and coverage--risk lower bound | Empirical Bernstein concentration; Markov-kernel perturbation; Le Cam/coupling lower bounds | Techniques covered; exact \(e^{\lambda L}/\mu\) executed-interface specialization not located | Finite-tabular attainability/limitation theorem | Extend beyond enumerable interfaces or show a strict advantage over primitive plug-in; do not sell concentration or two-point technique as new |
| Coverage--risk lower bound for irreversible Pareto decisions | Le Cam testing reductions; plug-in classification and optimal-stopping margin lower bounds | Technique covered; project-specific risk/coverage/Pareto embedding | Consequence theorem | Treat as alignment of the hard family with the mission criterion, not a standalone novelty claim |

## Highest-risk primary sources

| Source | Type | Insight | Completeness | Numeric evidence | Overall | Why it matters |
| --- | --- | ---: | ---: | ---: | --- | --- |
| Zhang et al., ICML 2024 | theory/algorithm | 5 | 5 | 4 | Risk | Already combines offline data, entropic risk, exponential Bellman error, coverage, pessimism, and finite-sample guarantees |
| Dai et al., NeurIPS 2020 (CoinDICE) | theory/method | 5 | 5 | 5 | Risk | Already gives behavior-agnostic density-ratio off-policy confidence intervals with finite-sample validity |
| Essakine & Vernade, COLT 2026 | theory/algorithm | 5 | 5 | 4 | Risk | Closes entropic best-policy sample-complexity bounds using sharp concentration and an explicit stopping rule |
| Belomestny, Finance and Stochastics / AAP 2011 | theory/proof | 4 | 5 | 3 | Risk | Already converts continuation-estimation accuracy and boundary regularity into optimal-stopping performance rates |
| Granados & Pacheco, AAAI 2026 | theory/algorithm | 4 | 4 | 4 | Risk | Explicit exponential-twisted state occupancy and off-policy risk-sensitive gradients |
| Fei et al., NeurIPS 2020 | theory/algorithm | 5 | 5 | 4 | Risk | Establishes unavoidable exponential risk--horizon dependence |
| Fei et al., NeurIPS 2021 | theory/algorithm | 5 | 5 | 4 | Risk | Uses exponential Bellman equations for sharp learning analysis |
| Chatterjee & Diaconis, AAP 2018 | theory/proof | 5 | 5 | N/A | Risk | Gives a broader information-theoretic exponential sample-size law for importance sampling |
| DualDICE, NeurIPS 2019 | theory/method | 5 | 5 | 4 | Risk | Adjoint Bellman density-ratio moment and finite-data saddle estimator are already established |
| Duan et al., ICML 2020 | theory/method | 5 | 5 | 4 | A | Restricted occupancy mismatch sharply controls OPE upper/lower bounds |
| Kontoyiannis & Meyn, AAP 2003 | theory/proof | 5 | 5 | N/A | A | Supplies the multiplicative spectral and Poisson-equation foundation |
| Anantharam & Borkar, SICON 2017 | theory/proof | 5 | 5 | N/A | A | Connects risk-sensitive reward to occupation and Donsker--Varadhan variational structure |
| Karmakar & Bhatnagar, SCL 2021 | theory/proof | 4 | 4 | N/A | Risk | Direct risk-sensitive function-approximation perturbation bounds |

Scores assess source quality and proximity, not acceptance probability.

## Opportunity map

| Cluster | Status | Remaining gap | Viable route | Evidence needed |
| --- | --- | --- | --- | --- |
| Exponential Bellman / entropic risk | covered central claim | None at the level of the risk measure or recursion | Use as foundation only | Correct citations and baseline implementation |
| Risk-twisted occupancy | covered central claim | Source-to-target killed execution-interface ratio may still differ operationally | Prove an exact reduction/equivalence to DICE or a strict separation | Theorem comparing risk resolvent with discounted Doob-transformed occupancy |
| Offline finite-sample risk learning | crowded but open | Existing theory is mostly fixed-horizon linear MDP, not proper first passage with a safety execution kernel | Proper-SSP transfer theorem with nuisance and trajectory dependence | Matching upper/lower rates under declared model class |
| Exponential sample complexity | covered central claim | First-passage/interface-specific sharp constants are not enough alone | Treat current result as a limitation, not headline novelty | Lower bound paired with an attainable estimator |
| Irreversible return decision | theory/analysis gap | No located result joins learned killed risk transfer to one-way charger commitment and mission Pareto loss | Sequential boundary-margin theorem under common stopped law | First-disagreement coupling, overshoot, calibration failure, and mission-loss bound |
| Safety-filter composition | mechanism gap | Existing risk-sensitive RL normally acts directly on plant actions | Executed-action quotient plus identifiable primitive shared across queryable policy/filter pairs | Two genuinely different safety kernels and held-out composition evidence |

## Concrete salvage route

1. Prove that every finite risk resolvent with \(\rho(M_\lambda)<1\) is a
   diagonally scaled discounted occupancy resolvent of a Doob-transformed killed
   chain. This determines exactly which parts reduce to DualDICE.
2. Define the residual object left after this reduction: target Doob dynamics are
   not directly sampled, while the policy and safety kernels are queryable and
   only the executed primitive is shared.
3. Theorem 14 now proves a finite-interface primitive plug-in upper bound and a
   matching \(e^{\lambda L}/\mu\) hard family. This closes finite-tabular
   attainability without claiming a new concentration inequality.
4. Theorem 13 now composes a simultaneous EIRR radius with positive MGF interval
   propagation, a one-sided stopped boundary law, and the stranding--throughput
   rectangle. The remaining obligation is no longer the algebraic composition:
   it is to decide whether a learned killed-Doob or continuous-interface method
   offers a theorem beyond the tabular plug-in baseline, and to provide formal
   cycle-level evidence.

## Stop conditions

- Stop claiming a new risk measure, exponential Bellman equation, twisted
  occupancy, DICE-style density-ratio objective, generic exponential sample
  lower bound, Chernoff certificate, or clipped Bernstein bound.
- Stop the oral theory headline if the Doob-transformed problem reduces fully to
  existing DICE plus a standard plug-in threshold without a new transfer or
  sequential-decision bound.
- Retain the work as a rigorous diagnostic/negative-result paper only if the
  risk-resolvent coefficient reliably predicts failures missed by ordinary
  occupancy after horizon matching.
