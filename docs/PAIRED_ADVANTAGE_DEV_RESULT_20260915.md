# Paired-advantage DEV v3 result

Date: 2026-09-15. The 96-world raw collection and complete audit passed before
the Gate analysis. The result follows the preregistered order: Gate H first,
then Gate I only because H passed.

## Gate H: passed

The fresh paired-CRN DEV evidence supports stable, decision-relevant C/R
advantage heterogeneity for this frozen environment and policy:

- Held-out selector gain over the cross-fitted best constant action: 0.15959.
- World-bootstrap 95% interval: [0.07650, 0.25668].
- Noise-corrected advantage variance: 0.66922, with 95% interval
  [0.35246, 1.00137].
- Both decisions were represented in both replicate-half swaps. R was selected
  at 9.90%--10.42% of anchors across 14--15 worlds; C was selected at
  89.58%--90.10% across 91 worlds.
- Raw/Monte-Carlo validity and all four Gate-H checks passed.

This establishes heterogeneity on DEV; it does not by itself establish that the
tested legal history can predict which action is better.

## Gate I: failed

Full legal history did not add out-of-world predictive or selected-utility value
beyond the latest legal frame or constant baselines:

- Full-history MSE: 1.31760; latest-frame MSE: 0.80292; constant MSE: 0.68407.
- Full-minus-latest MSE improvement: -0.51467, simultaneous 95% interval
  [-1.63343, 0.60408].
- Full-minus-constant MSE improvement: -0.63352, interval
  [-1.76333, 0.49629].
- Full-minus-latest selected utility: 0.00594, interval
  [-0.07369, 0.08557].
- Full-minus-best-constant selected utility: -0.03507, interval
  [-0.11219, 0.04204].
- LINEAR_RIDGE and RFF_RIDGE both had negative full-minus-latest MSE
  improvements (approximately -0.5147 and -0.5146).
- Full/latest actions disagreed on 10.94% of anchors, so failure is not explained
  by identical decisions; the disagreements simply did not improve held-out
  utility.

Only finite/support and action-disagreement checks passed. The simultaneous
interval, both relative-MSE, constant-utility and family-direction checks failed.

## Frozen decision

Gate H passed and Gate I failed. Therefore CONFIRM must remain unopened and no
information-utilization method or METHOD-TRAIN experiment is authorized by this
lineage. The result is a useful negative finding: action-value heterogeneity is
real in the paired experiment, but the tested full legal history and fixed model
families do not identify it better than the latest/constant baselines.

## Analysis-execution repair disclosure

The externally anchored PAI analyzer imported a predecessor loader with a
hard-coded predecessor contract ID. Before Gate execution, a local hash-linked
repair contract bound the parent contract, raw results, audit, original analyzer
and all transitive analysis sources. The repair validated the PAI-v3 ID, adapted
only that ID on an ephemeral copy for the predecessor loader, and then executed
the byte-identical frozen Gate logic. No estimand, threshold, feature, fold,
bootstrap, model or raw record changed. The repair contract was not separately
submitted to Rekor; this limitation must accompany any formal use of the DEV
result.
