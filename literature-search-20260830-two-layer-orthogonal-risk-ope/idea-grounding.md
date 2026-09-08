# Idea-Grounding Packet

## Scope And Evidence Boundary

- Topic: source-to-target estimation of stopped exponential charger-hitting risk
  under queryable execution composition and shared primitive dynamics.
- Search date: 2026-08-30.
- Source-supported facts: generic DRL, risk-CDF DR, marginalized ratios,
  restricted-chi-square rates, and exponential risk--horizon lower costs exist.
- Searcher inference: the promising unsolved joint object is a stopped
  Feynman--Kac canonical-gradient/efficiency theorem linked to an irreversible
  decision boundary.
- Unknown: exhaustive coverage of stochastic-control and semiparametric
  statistics literature under non-RL terminology.

## Evidence Cards

| Source | Supported observation | Reported limitation / scope | Mechanism primitive | Protocol anchor | Transfer condition | Confidence |
| --- | --- | --- | --- | --- | --- | --- |
| Kallus--Uehara 2020 | DRL is efficient with cross-fitted Q and marginalized ratios at fourth-root rates | Standard MDP policy value | Efficient influence function | DRL baseline | MDP memorylessness | direct |
| Huang et al. 2022 | DR return-CDF estimator and matching-order minimax risk bounds | Episodic return-distribution formulation | Recursive CDF augmentation | OPRA-DR | Trajectory overlap/model | direct |
| Uehara et al. 2020 | Weight and Q-function learning combine doubly robustly | Discounted infinite horizon | Minimax moment equations | MWL/MQL | Function-class realizability | direct |
| Duan et al. 2020 | OPE rate can depend on restricted chi-square over a function class | Linear approximation/additive value | CME/FQI equivalence | FQE/primitive plug-in | Restricted coverage | direct |
| Xie et al. 2023 | Structured linear MDP OPE has a canonical efficiency bound and efficient estimator | Known linear feature map | Semiparametric projection | Efficient linear OPE | Linear MDP | direct |
| Katdare et al. 2023 | Off-environment MIS can factor a density ratio through a simulator intermediate | Simulator plus real offline data | Two-step ratio product | Off-environment MIS | Simulator bridge | direct |
| Fei et al. 2020 | Exponential risk and horizon incur unavoidable statistical cost | Online episodic regret | Exponential utility lower bound | RSVI/RSQ | Finite horizon | direct |
| Biometrika 2024 | Product-bias cancellation and cross-fitting belong to a broad DR-functional theory | Generic iid semiparametric model | Influence function | DML/TMLE-style checks | Pathwise differentiability | direct |

## Cross-Source Relations

| Source pair / cluster | Relation | Open gap or conflict | Why it matters | Evidence needed next |
| --- | --- | --- | --- | --- |
| DRL + OPRA | supports | Efficient mean OPE and DR risk-CDF OPE both exist | Generic orthogonal risk estimation is covered | Show stopped-MGF EIF differs or improves |
| Duan + risk quotient | conflicts-with | Restricted quotient rates already characterize OPE limits | Quotient alone is not new | Nonlinear killed/stopped separation theorem |
| Katdare + two-layer EIRR | conflicts-with | Two-factor transfer ratios already appear | “Two layers” is insufficient novelty | Exact observed-data-model efficiency gain |
| Fei + EIRR lower bound | supports | Exponential risk severity has necessary cost | Validates but does not originate exponential factor | Random-hitting-time matching lower bound |
| CoinDICE + EIRR certificate | conflicts-with | Generic OPE confidence intervals already exist | Certificate novelty needs task-specific propagation | Non-vacuous width and boundary theorem |

## Idea Constraints

- Already covered central claims: generic double robustness; fourth-root nuisance
  products; marginalized density ratios; generic DR risk OPE; restricted
  function-class coverage.
- Transferable mechanism primitives: canonical gradients, cross-fitting,
  marginalized ratios, restricted divergence, Feynman--Kac resolvents, Le Cam
  local alternatives.
- Stale or overcrowded routes: “risk-weighted Bellman loss,” “two density ratios,”
  “CME plus resolvent,” and “DR estimator for energy risk” without a new bound.
- Minimum viable research questions:
  1. What is the canonical gradient of
     \(\nu(I-M_{\lambda,P})^{-1}r_{\lambda,P}\) when \(L\) is known and only the
     source primitive law varies?
  2. Does queryable \(L\) Rao--Blackwellize away one nuisance and strictly lower
     the efficiency bound versus trajectory-level OPRA?
  3. Is finiteness of a risk-observable quotient leverage coefficient necessary
     and sufficient for regular estimation near the transience boundary?
  4. Can the information bound be converted into a sharp local minimax bound for
     premature-return/stranding loss under a stopped margin law?
