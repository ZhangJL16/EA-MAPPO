# Idea-Grounding Packet: Risk--Boundary Observable Interface Quotient

## Scope and evidence boundary

- Topic: a theorem-driven representation/loss principle for continuous executed
  interfaces in killed first-passage Energy-to-Go estimation.
- Search date: 2026-08-30.
- Source-supported facts: CME rates, restricted OPE difficulty, aggregated
  concentrability, continuous bisimulation, covariate-shift KRR, risk transience,
  and fixed-horizon entropic offline RL all have strong prior art.
- Searcher inference: their joint localization to a charger-hitting exponential
  value and irreversible return boundary remains a possible theorem gap.
- Unknown: whether the complete four-part theorem survives reduction to existing
  restricted OPE after the known Doob transform.

## Evidence cards

| Source | Supported observation | Mechanism primitive | Protocol anchor | Transfer condition | Confidence |
| --- | --- | --- | --- | --- | --- |
| Duan et al. 2020 | OPE error is sharply controlled by restricted chi-square over the function class | conditional mean operator | fitted evaluation and confidence bound | closure plus episode data | direct |
| Jia et al. 2024 | Function-class/data-induced aggregation determines OPE difficulty | aggregated transition quotient | trajectory/admissible data lower bounds | value realizability setting | direct |
| Foster et al. 2022 | Coverage and supervised realizability alone do not ensure offline learnability | over-coverage barrier | offline lower bound | stronger representation or coverage needed | direct |
| Li et al. 2022 | Regularized CME has adaptive misspecified rates and a matching lower bound | vector-valued interpolation/CME | regularized operator regression | eigenvalue/source assumptions | direct |
| Ma et al. 2023 | KRR is minimax under bounded shift; truncated weighting handles finite second moment | target/source likelihood ratio | KRR under covariate shift | shared conditional law | direct |
| Ferns et al. 2011 | Continuous bisimulation metrics control value differences | behavioral pseudometric | state aggregation | discounted risk-neutral value | direct |
| Whiteley et al. 2012 | Multiplicative drift can stabilize Feynman--Kac particle variance | drift/minorization | particle approximation | nonnegative kernel regularity | direct |
| Cavus and Ruszczynski 2014 | Risk transience supports undiscounted DP and stopping | multikernel/risk transience | transient MDP | risk-transience assumptions | direct |

## Cross-source relations

| Sources | Relation | Open gap or conflict | Why it matters | Evidence needed next |
| --- | --- | --- | --- | --- |
| Duan 2020 + project Doob theorem | possible reduction | EIRR may be standard restricted OPE on a transformed killed chain | Could eliminate continuous-theory novelty | Write exact reduction with all normalizations |
| Jia 2024 + Ferns 2011 | complementary | statistical quotient versus behavioral metric | Suggests a representation theorem but also creates prior-art risk | Define the minimal multiplicative continuation quotient |
| Li 2022 + Ma 2023 | complementary | operator regression under misspecification and target shift | Supplies a baseline upper rate | Derive target-risk-weighted operator norm and lower rate |
| Whiteley 2012 + Cavus 2014 | complementary | variance stability versus risk transience | Replaces arbitrary finite horizon with structural conditions | Prove weighted resolvent stability for the charger-hitting kernel |
| Zhang 2024 + Duan 2020 | overlap threat | risk-sensitive pessimism and OPE already cover much of learning theory | A new loss alone is insufficient | Show first-passage/interface/boundary distinction or stop |
| Pavse 2025 + Ferns 2011 | direct collision | KROPE joins bisimulation-style representations to stable offline value iteration and Bellman completeness | Generic stable representation is no longer a viable headline | Prove a joint stopped-risk phase not implied by expected-return stability |
| Su 2025 + Wang 2025 | direct collision | Transient ERM/EVaR dynamic programming and augmented-MDP risk reductions are covered | Transience or risk augmentation alone cannot support novelty | Isolate intrinsic quotient estimation, Perron conditioning, and irreversible margin jointly |

## Idea constraints

- Already covered central claims: exponential Bellman learning, generic CME,
  generic bisimulation, stable bisimulation-based OPE representations, generic
  density-ratio OPE, KRR under covariate shift, transient total-reward ERM/EVaR,
  and rich-observation risk reduction.
- Transferable primitives: restricted chi-square, aggregated concentrability,
  behavioral pseudometric, vector-valued regression, multiplicative drift,
  first-disagreement margin law.
- Stale route: add policy/filter ID, intervention norm, or a generic uncertainty
  head as structured neural inputs without a sufficiency/necessity theorem.
- Minimum viable research question: can the risk-observable quotient be strictly
  smaller than the raw executed interface while preserving the exact killed
  log-MGF grid, and can its propagated error reach a separate stopped-boundary
  functional with statistically matching sufficient and necessary rates?

## Theorem-to-design rule

If the theorem succeeds, a representation (h_\theta(x,u)) should be trained to
preserve the conditional features

\[
\mathbb E[e^{\lambda C}
(\mathbf 1_G+\mathbf 1_{G^c}f(X'))\mid x,u]
\]

for (\lambda\in\Lambda) and learned continuation witnesses (f), with errors
weighted by cross-fitted risk occupation and stopped boundary mass. Invariance is
allowed only between interfaces that are indistinguishable to this witness class.
This is a theorem target, not yet an authorized architecture or result claim.
