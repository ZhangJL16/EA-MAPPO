# Search Notes

## Purpose and mode

Standard theorem-level opportunity search. The search tests whether continuous or
neural EIRR has a mathematical contribution beyond known conditional mean
embedding, covariate-shift regression, off-policy evaluation, state aggregation,
bisimulation, and Feynman--Kac stability.

## Safe public queries used

- conditional mean embedding operator concentration Markov decision process RKHS
- Feynman--Kac operator RKHS approximation concentration continuous state
- offline RL function-class coverage concentrability minimax theorem
- risk-sensitive stochastic shortest path exponential Lyapunov drift
- risk-sensitive bisimulation and state aggregation
- pushforward/aggregated concentrability representation offline RL
- covariate-shift KRR minimax target effective dimension
- multiplicative drift particle Feynman--Kac variance

No private draft sentence, unpublished result, local path, or user identity was
sent in a query.

## Sources checked

- PMLR pages/PDFs for ICML, COLT, and ACML papers
- Official NeurIPS proceedings
- SIAM journal pages and open author copies
- Annals of Statistics DOI/author copy
- arXiv only where it was the accessible author version of an archival result
- 2025 PMLR pages for KROPE, OCE reductions, and entropic constrained learning
- official AAAI and NeurIPS 2025 pages for transient total-reward ERM/EVaR

## Exclusions and screening

- Policy-excluded venues and domains were omitted.
- ResearchGate, Wikipedia, commercial summaries, and search snippets were not
  used as evidence for final claims.
- Application-only energy papers were excluded because this search concerns the
  theorem collision, not a new benchmark survey.
- Generic neural-operator and neural-tangent-kernel papers were screened out:
  without a risk--boundary theorem they do not change the project boundary.

## Source-supported observations

1. Conditional-mean-operator fitted evaluation and restricted chi-square minimax
   analysis already exist for OPE.
2. Function-class-induced aggregation can have exponentially worse coverage than
   the raw MDP even with trajectory data.
3. CME existence, misspecified upper rates, and matching lower rates are covered.
4. RKHS covariate-shift regression already has minimax bounded/second-moment ratio
   results, including truncation.
5. Continuous bisimulation already justifies value-preserving state aggregation.
6. Multiplicative drift and risk transience are established tools for noncompact
   Feynman--Kac/undiscounted settings.
7. KROPE now directly covers spectral stability and a Bellman-completeness route
   for bisimulation-based offline value representations.
8. Recent risk-sensitive reductions and total-reward ERM/EVaR results make
   generic augmented-state, transience, or exponential Bellman claims occupied.

## Inferences, not source claims

- A theorem composing a risk-observable executed-interface quotient with a
  separate irreversible stopped-boundary functional was not located.
- No inspected 2025 source states the joint intrinsic-quotient
  dimension--Perron-separation--risk-transience--stopped-margin phase.
- The proposed four-part theorem may be new only as a composition of killed
  first passage, queryable safety execution, quotient complexity, and irreversible
  boundary consequences.
- A matching packing lower bound is probably necessary for oral-level theory;
  this is a project judgment, not a statement from one source.

## Unknowns

- Whether a differently named multiplicative/risk-sensitive bisimulation theorem
  already proves the exact pseudometric target.
- Whether Duan et al.'s restricted-OPE theorem transfers verbatim to the
  non-normalized killed Doob chain without additional stability assumptions.
- Whether dependent first-passage trajectory sampling preserves a quotient rate
  without regeneration or mixing assumptions.
- Whether a neural encoder can be analyzed beyond an RKHS/finite linear class in
  a way that is both non-vacuous and empirically estimable.

## Handoff packet

- Next derivation owner: formula derivation.
- Frozen object: the four-part risk-observable quotient plus stopped-boundary
  theorem in `papers.md`, reframed algebraically in
  `docs/RISK_BOUNDARY_QUOTIENT_DERIVATION.md`.
- Mandatory collision baselines: Duan 2020, Jia 2024, Ferns 2011, Li 2022, Ma
  2023, Zhang 2024.
- Stop rule: if the theorem reduces exactly to standard restricted OPE on the
  known Doob transform, record a negative equivalence theorem and remove the oral
  novelty claim.
