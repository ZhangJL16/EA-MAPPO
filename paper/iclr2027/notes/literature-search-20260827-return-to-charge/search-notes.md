# Search Notes

## Safe Public Queries Used

- policy-conditioned environment model ICML 2024
- executed-action and compositional world-model generalization
- constrained and safe distributional actor-critic
- UAV energy prediction CVaR and return-to-base
- predictive safety network for resource-constrained multi-agent systems
- density-constrained RL electric-vehicle charging and persistent-UAV recharge rendezvous
- model-based offline RL uncertainty (MOPO, MOReL, COMBO)
- selective prediction, deep ensembles, conformal regression under shift
- ICLR 2025/2026 world-model uncertainty and composition

## Sources Checked

PMLR, NeurIPS proceedings, OpenReview, IEEE metadata/DOI, DBLP, and author project pages for discovery followed by primary-page verification.

## Exclusions and Unknowns

- MDPI sources were excluded by policy.
- ICLR 2026 search returned workshop papers, withdrawn submissions, or papers not close enough to support a manuscript claim; none enters the main bibliography solely to satisfy recency.
- Generic return-to-base heuristics were screened but no verified close paper covered the full executed-interface-to-return-boundary chain.
- The 2026-08-28 cynical-review pass found three previously missing primary records: Guo & B{\"u}rger (PMLR 100), Qin et al. (PMLR 139), and Mathew et al. (T-RO 31(1)); all were added because they materially tighten the nearest-work comparison.
- Altman (1999) is a canonical CMDP book but was left out of the current BibTeX pending publisher-record verification; CPO provides the needed main-text anchor.

## Handoff Notes

- For writing: position PCM as the closest risk and avoid “first policy-conditioned model” language.
- For experiments: executed-WM + Deep Ensemble and direct CMDP switching are mandatory strong baselines.
- For review: attack any claim that pair novelty alone causes error, any unconditional conformal guarantee, and any q95/q99 physical-tail wording.
- Treat omission or weak engagement with Predictive Safety Network as a novelty-positioning failure.
