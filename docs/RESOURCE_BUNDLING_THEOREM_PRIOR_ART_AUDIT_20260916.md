# Theorem-level prior-art and generalization audit

Date: 2026-09-16. Scope: the full separation property bundle, not the keyword
combination 'resource bandit'. This is a bounded primary-source audit, NOT an
exhaustive systematic review or a certification of priority.

## Result

Generalization passes in an explicitly parameterized structural family: see
`ROBUST_RESOURCE_BUNDLING_SEPARATION_20260916.md`. It supplies a positive-volume
four-parameter region with equal execution comparators, equal maximal fixed-instance
asymptotic coefficients across capacities, and a strict finite-budget risk gap.
It does not establish openness in an unrestricted independent reward matrix.

Novelty is NOT cleared. The ingredients are classical, and the current route
class remains exactly a finite controlled statistical experiment with known
durations. Physical provenance does not prevent that reduction. The defensible
candidate contribution is the simultaneous resource-realizable separation of
execution value, maximal asymptotic allocation coefficient, and finite-budget
optimal risk, robust in a structural family. Do not claim that cost-weighted
experiments, adaptive sensing, or stability of finite games are new.

## Workflow and search limitations

Used nature-academic-search multi-source-search workflow. No academic MCP tools
were mounted. Provided preflight script passed PubMed, Crossref and arXiv endpoint
checks. Provided OpenAlex fallback query `finite horizon side observations`
returned irrelevant astronomy results, which were excluded rather than used as
evidence. Subsequent discovery used web search; evidence below uses official
proceedings, arXiv and author-hosted full text only. Publisher/preprint duplicates
were collapsed by title and authors. Search-result crawl dates are not publication
dates. No citation-count or 'absence from Scholar' claim is made.

Search clusters included: finite horizon/asymptotic/controlled sensing/minimax;
same asymptotic constant/finite/bandit; measurement bundling/observations/regret;
side observations/finite-time lower bounds; resource/controlled sensing; Blackwell
strict finite horizon; structured bandit allocation. Broad searches were noisy.
This audit did not perform a complete forward/backward citation crawl, independent
reviewer consultation, or theorem-by-theorem inspection of all controlled sensing
literature. Confidence in absence of the entire property combination is therefore
limited. A source not found in this search is not evidence it does not exist.

## Theorem matrix: what prior work already establishes

| Primary source / evidence level | Established neighboring object/result | Implication for our claim | Difference still requiring defense |
| --- | --- | --- | --- |
| [Blackwell, Comparison of Experiments (1951)](https://digicoll.lib.berkeley.edu/nanna/record/112749/files/math_s2_article-08.pdf?registerDownload=1&version=1&withMetadata=0&withWatermark=0); original proceedings identified | Comparison of experiment informativeness through attainable risks | More usable information can improve a decision; weak dominance is not a discovery | A known physical feasibility lift with unchanged reward execution and asymptotic allocation coefficient is not supplied merely by this comparison |
| [Nitinawarat, Atia, Veeravalli, Controlled Sensing for Multihypothesis Testing (2013; preprint 2012)](https://arxiv.org/abs/1205.0858), [author full text](https://vvv.ece.illinois.edu/papers/journal/niti-atia-veer-tac-2013.pdf); abstract and fixed-sample/sequential formulation | Three-hypothesis example where causal controls strictly outperform open-loop controls; asymptotically optimal sequential tests | Adaptivity advantage, including a three-hypothesis witness, is DIRECT prior art | Testing error/exponent objective rather than cumulative calendar regret; this result does not by itself establish nested capacity, unchanged execution and equal maximal log-regret coefficient |
| [Nitinawarat, Veeravalli, Controlled Sensing ... Controlled Markovian Observations and Non-Uniform Control Cost](https://arxiv.org/abs/1310.1844), 2013 submission/2014 revision; primary abstract/formulation | Controlled observation memory, arbitrary accumulated control costs, asymptotically optimal tests and non-asymptotic risk constraints | 'Information acquisition is controlled and costs physical time' is NOT a new learning principle | No equivalence should be asserted or ruled out without comparing the actual cumulative-regret objective and recharge constraints; this is a major reduction threat |
| [Wu, Gyorgy, Szepesvari, Online Learning with Gaussian Payoffs and Side Observations (NeurIPS 2015)](https://proceedings.neurips.cc/paper/2015/file/8e82ab7243b7c66d768f1b8ce1c967eb-Paper.pdf); Sections 1-3 inspected | Finite-time information-constrained regret lower bounds, side observation geometry and different finite/asymptotic regimes | 'Asymptotic information coefficients do not determine finite-horizon difficulty' is already a neighboring insight | Gaussian unit-round observation model, not the current known-duration safe trajectory implementation; NOT enough to certify new statistical machinery here |
| [Combes, Magureanu, Proutiere, Minimal Exploration in Structured Stochastic Bandits (NeurIPS 2017)](https://proceedings.neurips.cc/paper/2017/file/e19347e1c3ca0c0b97de5fb3b690855a-Paper.pdf); semi-bandit example and lower-bound sections | Structured GL allocation and matching OSSB; product vector feedback includes combinatorial semi-bandits | Bundled observations and a resource-filtered catalogue do not evade structured experiment allocation | Unequal calendar durations need correct accounting, but that alone does not prove irreducibility or novelty |
| [Degenne, Garcelon, Perchet, Bandits with Side Observations: Bounded vs. Logarithmic Regret (UAI 2018)](https://www.auai.org/uai2018/proceedings/papers/182.pdf); finite-regime discussion inspected | Extra free observations change attainable regret; finite-regime transition and matching bounds | Finite relevance of additional information is not new | Their side observations are exogenous/free, and can change logarithmic versus bounded asymptotics; our equal maximal coefficient and physical bundling package differs |
| [Huang, Cohen, Zhao, Active Anomaly Detection in Heterogeneous Processes](https://arxiv.org/abs/1704.00766); accepted-paper abstract, not complete theorem audit | Adaptive subset probing; same asymptotic optimality with different finite performance | Equal asymptotic performance plus finite improvement is not novel in isolation | Algorithm comparison for sequential detection, not a strict comparison of optimal finite-budget risks under nested resource experiment sets |
| [Ferrari, Zhao, Scaglione, Utility Maximizing Sequential Sensing Over a Finite Horizon (2017)](https://arxiv.org/abs/1705.05960); primary abstract/formulation | Finite-horizon sensing/exploitation POMDP maximizing reward minus penalties and sensing costs | Finite-budget economics of acquiring knowledge and exploiting it is DIRECT adjacent work, not an empty theoretical space | Their formulation alone does not certify the simultaneous cross-capacity execution/coefficient identities of our separation; detailed comparison is still needed |

The matrix separates verified neighboring statements from unverified absence
claims. It does NOT claim these papers cannot encode our instance through an
appropriate generalized experiment formulation.

Priority for the remaining close-paper audit: Ferrari et al. for finite-budget
sensing/exploitation cost; Wu et al. for finite-time information lower bounds;
Nitinawarat et al. for adaptive versus open-loop controlled sensing. These are
more relevant threats than searching additional battery keywords.

## Exact reduction challenge

For a committed path p, define its duration ell_p, vector observation law
Q_i,p and expected reward R_i(p). The learner chooses a finite known-duration
structured experiment with unknown label i; full route feedback is its sample.
At finite calendar budget, the state is remaining time and previous experiment
feedback, plus physical position if within-trip choices are allowed. Thus the
current finite game CAN be represented without loss as a controlled experiment
or belief-state decision problem. We must not answer the equivalence challenge
'No, because battery': battery defines which experiments can be realized but
does not erase this mathematical representation.

The previous general allocation theorem and committed-path attainability are
cost-weighted structured-allocation results. They are support for this separation,
not evidence of a new universal exploration algorithm. The finite risk proof
uses standard Bayes/minimax duality and exact Bellman calculations. The new
robustness extension uses standard uniform coupling continuity. None of those
tools is itself a claimed innovation.

## What the new generalization does and does not resolve

Resolved: the strict difference is not restricted to three finely tuned numeric
points. A single frozen adaptive high-capacity witness works for every parameter
in a nonempty four-dimensional box. No new policy optimization is needed there.
Exact equality of maximal coefficients has a visible structural cause: under q,
the B-baseline equals the dock mean, so bundling provides an A measurement at
the SAME centered cost as A-only. Strict inequality makes q remain the worst
asymptotic coefficient over that box.

Not resolved: whether this is a broadly useful structural criterion, whether
the full property package appeared in an existing controlled-experiment example,
and whether the phenomenon is visible in an operational embodied budget. Openness
is weaker than a graph-level characterization. A tiny conservative box cannot
be used as evidence that bundling generally improves learning. Exact same
coefficients are not robust to unconstrained perturbation of every independent
hypothesis mean. The old T=4096 negative calibration remains negative.

## Recommended precise claim

We establish an open-family, physically resource-realizable separation:
unchanged optimal execution values and unchanged maximal instance-dependent
asymptotic regret coefficients do not imply unchanged finite-budget minimax
learning risk. A safe shared-measurement trip and feedback-dependent scheduling
can strictly improve the latter. The proof does not require within-trip adaptivity.

Do not claim: first adaptive sensing result; new Blackwell ordering; new GL
principle; full Consumption-MDP minimax characterization; 'asymptotic minimax
equivalence' without specifying max fixed-instance coefficients; verified
real-world robotics relevance; or novelty established by not finding a title.

## Audit disposition

**Generalization gate: PASS within the explicit structural family.**
**Prior-art gate: NOT CLEARED; neighboring theory strongly overlaps.**
**Independent proof gate: OPEN.**

No sampled experiments, seed changes, horizon extension, learner modifications,
CONFIRM access, or benchmark/library launches occurred in this audit.
