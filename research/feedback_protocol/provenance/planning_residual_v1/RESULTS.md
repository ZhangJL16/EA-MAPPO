# T5.1e — where does planning compute create value?

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-09-19
- Verification Status: 128 exact Bellman identities and independent policy integrations checked
- Version Label: planning_residual_v1

## Decision: Route C — stop the current learned-planner main line

Do not launch full AFPP, a learned residual selector, V2.1 expansion or V3. Retain the theory and planning/evaluation framework. This is a research decision on the current evidence, **not** a theorem that amortization can never help larger or different tasks. The earlier two-root DEV GO remains a historical observation, not a guarantee that a new learned method is needed.

Large OneStepVOI is Bayes-optimal on all 32 variants and has zero fresh-decision residual on all 1,171 census states. Small VOI has only 9 nonzero fresh-state residuals, but episode-budget exhaustion changes its later actions. Its deployment gap is dominated by fixed score-blind fallback/execution decisions, not rare continued-sensing decisions.

## Exact deployment gaps

| Planner | Tier | Positive-gap variants | Mean gap | Positive-gap structures | Mean model calls / episode |
|---|---|---:|---:|---:|---:|
| beam_bayes | large | 1/32 | 0.006250 | 1/8 | 63928.00 |
| beam_bayes | small | 28/32 | 1.238922 | 7/8 | 10111.50 |
| one_step_voi | large | 0/32 | 0.000000 | 0/8 | 9480.87 |
| one_step_voi | small | 20/32 | 0.754581 | 7/8 | 5087.74 |

Means equally weight the 32 paired variants for description; there are only eight structural groups. These are Bayes utility gaps `V* - V_policy`, not minimax-regret estimates.

## Where the Small VOI loss occurs

| Exclusive categorization (each dimension separately) | Share of exact pooled gap |
|---|---:|
| autopilot: autopilot | 79.7229% |
| autopilot: normal | 20.2771% |
| decision_class: measurement_required | 17.4411% |
| decision_class: nonmeasuring_optimal | 82.5589% |
| limit: limit_hit | 100.0000% |
| prior: nonprior | 37.8640% |
| prior: prior | 62.1360% |

Different dimensions overlap: their percentages must not be added. All positive Small Beam residuals likewise coincide with a work-limit hit; 70.82% of its pooled gap is at nonmeasuring-optimal states. All selected actions carrying positive occupancy-weighted residual are task actions. A work-limit flag is an observed association, not proof that merely raising a cap is the uniquely optimal repair.

**Adaptive-subset contribution is zero for all four policies.** None of the actual feedback trees visits a non-prior measurement-required state, even though six such census states exist. Their existence and rarity are not the source of this study's deployment quality gap. See adaptive_contribution.json.

## Fresh census decisions versus deployment contexts

| Planner | Tier | Nonzero residuals / 1,171 | Structural groups |
|---|---|---:|---:|
| beam_bayes | large | 0 | 0 |
| beam_bayes | small | 203 | 7 |
| one_step_voi | large | 0 | 0 |
| one_step_voi | small | 9 | 4 |

Fresh Small VOI errors are 8 prior-belief states and 1 non-prior state, all measurement-required and work-limit-associated. This does not describe its episode losses: cumulative work can be exhausted on later otherwise simple task decisions. Large Beam has no census fresh-state errors yet has a 1/5 episode gap on one variant, again from limited-work nonmeasuring decisions. A belief/H-only occupancy analysis would miss this distinction.

## Files and verification

- `state_residuals.jsonl`: all 4,684 fresh census-state × planner rows, exact V/action/Q/residual, ties, categories and work.
- `occupancy_decomposition.json`: 128 complete policy trees; full-context occupancy and state aggregates, exact weighted residuals, EPE cross-check.
- `small_vs_large.json`: 2,342 paired fresh-state differences plus 64 paired episode gaps. Uniform decision differences are not occupancy gains.
- `summary.json`: exclusive-category shares, work, positive-state counts and complete concentration curves; no artificial acceptance cutoff.
- `supplemental_labels.json`: 452 evaluation-only labels for actually visited states outside the census; all exact, same problems/objective/caps.

128/128 identities `sum d*delta = V*(root)-V_policy(root)` hold as rational equalities and agree with the unchanged hypothesis-wise ExactPolicyEvaluator. Whole-policy state is deep-copied at each feedback branch; budgets are not refreshed on branches. STOP Q=0 is handled explicitly. State aggregates retain all context IDs because the same belief/H can have different remaining planning work.

## Scope and interpretation

This analysis does not rerun DEV 2201/2202 or explain their specific fourth-batch gap. It diagnoses the current teacher pilot. Large VOI is already exact here at 9,480.87 mean model calls per episode, versus Small VOI 5,087.74. Measured mean summed planning latency was about 16.6 ms versus 9.0 ms (one local CPU run, not a population or hardware-independent claim). This does not establish a compelling learned-planner advantage to pursue on this family.

State-set hardness is not the same as occupancy-weighted performance. Sparse fresh-state errors do not automatically justify a learned selector when most deployment loss is fallback behavior and a tested ordinary planner already attains the optimum.

63 tests passed. No new roots, raw-label retries, objective changes, neural training, V2.1/V3, formal DEV/test/OOD or CONFIRM access. Original frozen code and evidence remain unchanged.
