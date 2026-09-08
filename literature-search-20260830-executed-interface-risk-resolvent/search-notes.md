# Search Notes

## Scope

The search prioritised operations research, probability, statistics, stochastic
control, and applied mathematics. ML venues were retained only where they define
a close novelty boundary or provide a directly reusable estimator.

Topics queried and screened:

- stochastic shortest path, first-passage total reward, absorbing-chain
  resolvents, and perturbation theory;
- Feynman--Kac semigroups, exponential utility, multiplicative Bellman equations,
  entropic risk, EVaR, and risk-sensitive control;
- occupation-measure continuity, density-ratio estimation, lack of overlap, and
  partial identification;
- optimal stopping, plug-in decision margins, and continuation-boundary error;
- conformal calibration under covariate or non-exchangeable shift;
- distributional/QTD failure modes and value-aware model learning;
- UAV energy risk and energy-sufficiency barrier functions.

## Inclusion policy

Included papers had to supply at least one of: a theorem needed in the derivation,
a counterexample or assumption boundary, a direct novelty collision, or an
application baseline. Peer-reviewed journal and proceedings versions were
preferred. Recent arXiv work was retained only where no archival version was
located or where it is a direct current novelty risk.

## Exclusion policy

Excluded:

- generic energy-aware RL papers that add an energy reward or state feature but
  do not analyze first-passage resource;
- papers using CVaR only as an application objective without transferable theory;
- generic world-model or ensemble-uncertainty work without a theorem for the
  target shift;
- duplicate preprint and proceedings versions;
- papers whose only relevance was a keyword match to Feynman--Kac;
- non-primary summaries when an original paper was available.

## Novelty collisions found

1. Exponential utility and multiplicative Bellman equations are classical and
   heavily developed.
2. ERM and EVaR for transient total-reward MDPs were explicitly studied by Su,
   Petrik, and Grand-Clement (2025).
3. Itakura--Saito loss for exponential-value learning was explicitly proposed by
   Udovichenko et al. (2025).
4. Policy-conditioned environment models already address policy identity as a
   conditioning variable; a new result must instead exploit the executed-action
   quotient or prove when identity is insufficient.
5. Weighted conformal prediction does not justify arbitrary composition-shift
   coverage without a valid weight or exchangeability condition.

## Search saturation judgment

The foundation clusters are saturated enough to fix the direction: additional
broad keyword search is unlikely to change the conclusion that the risk measure
itself is prior art. The unresolved and highest-value search is narrower:

- a paper giving the exact risk-tilted *source-to-target* resolvent perturbation
  identity for first-passage total cost;
- a paper defining a density ratio for the non-normalized Feynman--Kac occupation
  measure in offline RL;
- a sequential plug-in stopping theorem with a risk-sensitive learned model and
  an executed safety-filter interface.

Until those three gaps are checked by theorem-level comparison, EIRR is a strong
candidate direction rather than a certified novel theorem.

## Targeted near-neighbour check on 2026-08-30

Exact-phrase and concept searches combining `Feynman--Kac`, `density ratio`,
`occupancy ratio`, `risk-sensitive`, `off-policy evaluation`, and
`first-passage` did not surface an RL paper with the complete proposed object.
The closest returned clusters were ordinary discounted occupancy-ratio
evaluation, mean-variance off-policy learning, Feynman--Kac density ratios for
diffusion/SMC sampling, and risk-sensitive control without source-to-target
executed-interface identification. This is evidence of a plausible gap, not a
proof of novelty; citation-chain and theorem-body inspection remains required.

## Theorem-level follow-up on 2026-08-30

Public queries added combinations of `multiplicative Poisson equation`,
`risk-sensitive sensitivity formula`, `exponential Bellman residual`,
`risk-sensitive offline RL`, `twisted occupancy`, `importance-sampling sample
size`, `restricted chi-square OPE`, and `DICE adjoint Bellman`.

This follow-up found four material novelty collisions:

1. Zhang et al. (ICML 2024) already develop finite-sample offline entropic-risk
   learning from exponential Bellman evaluation errors under explicit data
   coverage conditions.
2. Granados and Pacheco (AAAI 2026) explicitly derive risk-sensitive gradients
   under exponential-twisted state occupancies, including off-policy variants.
3. Fei et al. (NeurIPS 2020, 2021) already prove unavoidable exponential
   risk--horizon dependence and algorithms based on the exponential Bellman
   equation.
4. Chatterjee and Diaconis (Annals of Applied Probability 2018) prove a much
   broader exponential sample-size law for importance sampling, while DualDICE
   and related work already derive adjoint Bellman density-ratio objectives.

Accordingly, the finite-state resolvent, risk occupation, common-support lower
bound, Chernoff certificate, and clipped Bernstein certificate must be cited as
known foundations or specializations. The remaining search question is not
whether those pieces exist, but whether their *joint source-to-target killed
first-passage execution-interface stopping theorem* has appeared.

MDPI results were excluded from screening and final evidence. Search snippets,
secondary paper-note sites, and ResearchGate mirrors were used only for
discovery when a primary proceedings, publisher, DOI, or arXiv page was also
available.

## Theorem-13 confidence-to-stopping follow-up on 2026-08-30

Public queries combined `off-policy confidence interval`, `density ratio`,
`entropic risk`, `pessimism`, `optimal stopping`, `plug-in`, `margin condition`,
and `stopping-rule sample complexity`. Primary proceedings, publisher, and arXiv
records exposed three additional collision boundaries:

1. CoinDICE already gives behavior-agnostic density-ratio off-policy value
   confidence intervals with finite-sample validity. A generic "DICE plus a
   confidence interval" claim is therefore unavailable.
2. Belomestny's 2011 optimal-stopping analyses already turn continuation-value
   estimation and boundary regularity into nonasymptotic stopping-performance
   rates. The margin exponent in Theorems 12--13 is a foundation, not novelty.
3. Essakine and Vernade (COLT 2026) derive matching entropic best-policy sample
   complexity using sharper exponential-utility concentration and a stopping
   rule. A generic "entropic confidence plus stopping" headline is unsafe.
4. Maurer and Pontil (COLT 2009) supply the empirical Bernstein foundation used
   to make the finite-interface primitive-kernel radius data dependent. The
   variance-adaptive concentration step is prior art, not a contribution.

No inspected source joined all of: source data collected under another executed
policy--safety composition, a killed first-passage Feynman--Kac occupation ratio,
unknown Doob-transform estimation, a one-way return commitment, and mission
stranding--throughput headroom. This absence is a bounded search finding, not a
proof of novelty. Theorem 13 must be positioned as a bridge; oral-level theory
still depends on deriving a new source-to-target nuisance rate and matching lower
bound.
