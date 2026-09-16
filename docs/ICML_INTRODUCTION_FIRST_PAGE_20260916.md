# Introduction — first-page draft

## Draft

When a learning agent must physically execute an experiment, increasing its
resources can change how it learns even if it cannot improve what an informed
controller achieves. Consider an inspection agent that leaves a charging station,
takes noisy measurements, and returns before choosing its next route. Additional
capacity may permit two measurements within one safe excursion. Its value need
not be a better final inspection policy: it may instead change which feedback
can inform the allocation of the remaining deployment budget. This motivates a
question distinct from resource-feasible planning: **can resource augmentation
strictly improve finite-budget learning while leaving both execution performance
and the worst fixed-instance asymptotic information price unchanged?**

Execution value and asymptotic regret characterize different aspects of this
question. The first measures performance with the model already known. The
second prices the exploration needed to distinguish decision-relevant
alternatives in the long run. Structured-bandit theory expresses that price as a
KL-constrained allocation problem and provides asymptotic attainability under
regularity assumptions. Neither summary, however, specifies the sequence of
feedback-dependent decisions available before a finite budget expires.
Side-observation theory already demonstrates that asymptotic logarithmic bounds
can obscure substantial finite-time difficulty. Our question is therefore not
whether finite and asymptotic criteria differ in general, but whether a
physically realizable resource enlargement can separate them while execution is
held fixed. [Structured allocation](https://proceedings.neurips.cc/paper/2017/file/e19347e1c3ca0c0b97de5fb3b690855a-Paper.pdf),
[finite-time side observations](https://proceedings.neurips.cc/paper/2015/file/8e82ab7243b7c66d768f1b8ce1c967eb-Paper.pdf).

Controlled sensing is the appropriate parent framework for this investigation.
It already accounts for action-dependent information, adaptive experiment
selection, and finite-horizon sensing/exploitation utility. We do not introduce
these ingredients or claim that resource constraints make our system
irreducible to controlled experiments. Instead, we compare nested safe experiment
spaces over the same public hypothesis class, preserving all old feedback laws
and durations. The cited testing and utility results provide essential tools;
their conclusions do not by themselves establish our simultaneous
execution-preserving, asymptotic-neutral resource separation.
[Controlled sensing](https://arxiv.org/pdf/1205.0858),
[finite-horizon sensing utility](https://arxiv.org/pdf/1705.05960).

We establish such a separation in a regenerative resource-constrained system.
Increasing capacity enables a shared-measurement trajectory without changing
any hypothesis's optimal average execution gain. The maximum of attainable
fixed-instance logarithmic regret coefficients also remains unchanged, yet the
optimal finite-budget minimax regret strictly decreases. Exact certificates and
an open structural parameter neighborhood establish that this is not a sampled
learning-curve fluctuation or a single tuned parameter point. Known-model
charger-terminal execution value is identical at the certified endpoint budgets;
disabling measurement bundling makes the capacity levels statistically
equivalent. Equality of the maximal coefficient does not imply equality of every
instance's coefficient, and we make no interchange-of-limits claim about
asymptotic minimax regret.

The interpretation concerns how information is organized around future budget
commitments, not merely how much information is collected. Separate measurements
require separate travel-and-return cycles; bundling makes their joint feedback
available within one completed cycle for the next route decision. Our improving
witness still commits to finishing each excursion, but adapts subsequent choices
to observed feedback. A feedback-free charger-terminal control does not account
for its performance. The asymptotic bottleneck nevertheless survives because an
old maximizing hypothesis's optimal KL price remains feasible for the new
experiment. At finite budgets, the relevant bottleneck is instead the value of
feedback-dependent continuation at least-favorable priors. These are two distinct
objects, not two names for sample efficiency.

The resulting lesson for learning-system evaluation is specific: equal informed
execution performance and equal maximal fixed-instance information prices are
insufficient to certify finite-budget equivalence of resource configurations.
Feasible feedback-and-continuation schedules must also be examined. This is an
optimal-risk separation, not a guarantee that an arbitrary learner exploits it.
Our frozen horizon-free calibration did not exhibit the advantage at its tested
budget; we retain it as a diagnostic rather than empirical confirmation. The
paper's central contribution is the robust separation and its resource
realization, not a new sensing formulation or a claim of deployed algorithmic
superiority.

## Author notes — claim control, not manuscript prose

### One-sentence argument

Within controlled experiments, the existing resource-realizable construction
shows that execution value and maximal fixed-instance information price can both
remain unchanged while finite-budget optimal learning risk strictly improves;
the interpretation concerns feasible feedback-dependent budget allocation.

### Section outline

- Paragraph 1: concrete learning decision and why execution-only comparisons miss it.
- Paragraph 2: the two performance summaries and the specific unresolved comparison.
- Paragraph 3: acknowledge the parent framework and restrict the novelty claim.
- Paragraph 4: state the proven separation and its strongest controls.
- Paragraph 5: explain the feedback/commitment interpretation without inventing timing effects.
- Paragraph 6: give the evaluation takeaway and retain the negative calibration.

### Terminology ledger

| Canonical term | Meaning / prohibited substitution |
| --- | --- |
| Execution value | Known-model control performance; finite-horizon equality has terminal/budget qualifications |
| Maximal fixed-instance asymptotic regret coefficient | Maximum of the attained instance coefficients; not an asymptotic minimax constant |
| Finite-budget minimax regret | Worst-hypothesis risk minimized over one unknown-model learner |
| Feedback-dependent continuation | Subsequent decisions use observed feedback; not necessarily within-excursion adaptivity |
| Budget commitment | Interpretive description of choosing a physical route; no new formal complexity quantity |

### Claim–evidence map

| Claim | Existing evidence | Status |
| --- | --- | --- |
| Strict finite-budget improvement | `STRICT_FINITE_BUDGET_CAPACITY_BENEFIT_20260916.md`: exact T=12 lower/upper gap; certified integers 12–24 | Supported by existing certificates |
| Robustness | `ROBUST_RESOURCE_BUNDLING_SEPARATION_20260916.md`: four-dimensional structural box, radius 1e-4, T=12 gap > .038509475 | Supported; not an unrestricted reward matrix |
| Execution control | Same gains at all hypotheses; charger-terminal known-model equality at T=12,24 | Supported with qualifications |
| Neutral maximal information price | Surviving maximizing-hypothesis dual; individual A coefficient decreases | Supported; equality is at maximum level only |
| Adaptivity is relevant to the witness | High committed policy mixture adapts between excursions; feedback-free charger-terminal value 32/115 exceeds its upper bound | Supported for the specified control class |
| Timing of information is the sole causal mechanism | No experiment holds cost and feedback laws fixed while varying only release timing | Not established; not claimed |
| Empirical learner advantage | Frozen T=4096 calibration has high/on pseudo-regret 4.30 versus low/on 4.10 | Unsupported; negative result retained |

### Two substantive corrections to the proposed mechanism wording

1. “Feedback before committing” applies to **future** route choices. The high
   witness does not revoke its current route commitment, obtain a new early-return
   capability, or demonstrate that its first measurement arrives earlier.
   The defensible interpretation is shared acquisition followed by adaptive
   continuation, not timing-only causality or a universally smaller commitment.
2. “Asymptotic dual price inactive” is too strong and is false if interpreted as
   a slack new LP constraint. At the anchor, the new AB row is **tight** under
   the surviving q dual. Neutrality means that an old optimal dual remains
   feasible, not that the new experiment has zero information or a strictly
   inactive constraint. Likewise, positive Bellman backup at one prior is not
   sufficient: the existing criterion concerns every old least-favorable prior.

### Why this structure

- The opening names an actual budget decision rather than claiming embodied AI
  universally assumes free information.
- Prior theory is credited before the separation, preventing a manufactured gap.
- The interpretation stays downstream of the certificates; it does not promise a
  new commitment-gap theorem, an efficient solver, or successful deployment.

Assumptions or missing inputs: none needed to complete this first-page writing
task. Existing proof-audit and global-priority limitations are not removed by
prose. No methods, experiments, numerical certificates or frozen sources changed.

中文核心表达：**资源可能不改变“已知模型时能做到什么”，也不改变最坏固定实例的
渐近信息价格，却改变“有限预算内，哪些反馈可以共同用于下一次行动承诺”。**
这是当前证据支持的 scientific interpretation；不是“提前获得反馈”或“消除承诺”的新定理。

To redirect: identify a paragraph or claim; revise it locally without changing the
existing theorem or calibration record.
