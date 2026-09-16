# Theorem section outline — one-page core

Writing scope: manuscript / research / theorem-section architecture (method
fragment used only for assumptions and boundary) / zh-to-en / generic ICML.
This reorganizes existing results; it adds no theorem, proof or experiment.

## 3. Resource-realizable separation of learning criteria

**Reader takeaway.** Two familiar summaries—known-model execution performance
and maximal fixed-instance asymptotic regret coefficient—can agree across resource
configurations while their optimal finite-budget learning risks strictly differ.
The missing comparison is the feasible feedback-dependent decision protocol.

### 3.1 Comparison objects and resource-specific assumptions

Fix a public parameter point \(z=(r,u,v,w)\), the hypothesis labels
\(\Theta_z=\{(u,r),(v,r),(v,w)\}\), physical graph, reward laws, initial charger
and feedback protocol. The true label is unknown. A known replenishing counter
of capacity B determines legal trajectories. Capacity 4 permits a joint
measurement excursion; capacity 3 does not. Old experiments embed with identical
durations and feedback. Use debit-before-arrival reload semantics and at most
one Bernoulli measurement per primitive step.

Define \(v_{B,z}(T)=\inf_a\max_\theta[T\rho_\theta-E_\theta^a\sum_{t<T}Y_t]\)
over safe primitive-feedback policies, and \(C_{B,z}^{\max}=\max_\theta C_{B,z}(\theta)\)
over the attainable fixed-instance logarithmic allocation coefficients. Define
known-model finite execution separately, with charger-terminal policies.
The shared identity “B mean under q and A = dock mean r” is explicit, not a
generic resource assumption. Openness below is within this structural family.

### 3.2 Main Theorem — robust asymptotic-neutral finite-budget separation

For every \(z\) in the nonempty open box
\(\|z-(.05,.05,.30,.95)\|_\infty<10^{-4}\), the capacity-3 and capacity-4 systems have:

\[
\rho_{3,\theta}(z)=\rho_{4,\theta}(z),\qquad
C_{3,z}^{\max}=C_{4,z}^{\max},\qquad
v_{3,z}(12)-v_{4,z}(12)>.038509475.
\]

At T=12 and 24, every hypothesis's known-model charger-terminal optimum is
\(T\rho_\theta(z)\) at both capacities. At the anchor, strict finite-budget
gap exceeds 1/20 for every integer T=12,…,24. These are distinct budget
qualifications, not one unrestricted execution-equality assertion.

**Role:** the conjunction is the main scientific result. Do not advertise full
asymptotic information-geometry equality: the A coefficient decreases, while q
continues to set the same maximum. No asymptotic minimax limit is asserted.
Proof pointers: existing strict-certificate and robust-separation documents.

### 3.3 Explanation — two bottleneck tests, not two new foundational theorems

Present the existing abstraction as a proposition. For nested experiment systems,
write old Bayes regret \(b_1(\pi)\), new Bayes reward improvement
\(g_T(\pi)=V_2(T,\pi)-V_1(T,\pi)\), and old least-favorable priors \(\mathcal L_1\).
Finite strictness holds exactly when \(g_T(\pi)>0\) for every
\(\pi\in\mathcal L_1\). Asymptotic maximum neutrality holds when an old
maximizing hypothesis has an optimal allocation dual feasible for all new rows.
These are standard minimax and LP consequences applied to different objects.

In the resource realization, q's dual survives the joint excursion, while an
adaptive high-capacity witness certifies strict finite improvement. The witness
commits to each completed sortie and adapts later route choices. Do not infer
strictness from additive KL, one positive prior backup, or earlier feedback.

### 3.4 Realization and controls

Keep these as corollaries/controls, not invented main theorems: bundling-off gives
exact cross-capacity risk equality; the high witness terminates at the charger
while the low lower bound allows arbitrary legal endpoints; the anchor T=12
feedback-free charger-terminal value is 32/115, above the adaptive witness upper.
Together they isolate resource-enabled shared acquisition and adaptive
continuation, without proving timing-only causality.

**Positioning and implication.** This is representable within controlled sensing
and structured experiments. The contribution is the explicit robust separation
under resource-realizable nesting and qualified execution invariance, not a new
formulation or allocation tool. The comparison audit identifies why named prior
theorem conclusions do not directly supply this certificate. For ML evaluation,
agreement of the two summaries cannot substitute for comparing finite-budget
feedback protocols. The negative T=4096 frozen-learner calibration remains a
diagnostic; it is not this theorem's empirical confirmation.

---

## Author-only audit and evidence pointers

### Four reviewer questions — direct answers

| Question | Answer |
| --- | --- |
| What is the main theorem? | Section 3.2: robust strict finite-budget separation with equal gains and equal maximal fixed-instance coefficient |
| Why not simply an existing controlled-sensing theorem? | The framework and tools are existing; the contribution is the separately proved resource-realizable conjunction, not inability to represent it |
| Which assumptions are resource-specific? | Known counter, reload/debit semantics, safe excursion feasibility, and exact old-path embedding; shared reward identity is an additional class restriction |
| Which conclusion is the separation? | Equal execution summaries and equal maximum asymptotic coefficient coexist with a strict finite-budget optimal-risk gap |

### Terminology ledger

| Adopt | Do not substitute |
| --- | --- |
| Maximal fixed-instance asymptotic regret coefficient | Full asymptotic geometry; asymptotic minimax constant |
| Nested resource-realizable experiment enlargement | Statistically indistinguishable systems |
| Feedback-dependent continuation | Earlier feedback; removal of current commitment |
| Robust structural family | Arbitrary reward matrix or general Consumption MDP |

### Claim–evidence map

- Main separation, exact anchor and terminal controls:
  [strict result](STRICT_FINITE_BUDGET_CAPACITY_BENEFIT_20260916.md).
- Open-box separation and structural sharing:
  [robust result](ROBUST_RESOURCE_BUNDLING_SEPARATION_20260916.md).
- Least-favorable-prior and surviving-dual criteria:
  [existing abstraction](RESOURCE_ENABLED_FINITE_BUDGET_PRINCIPLE_20260916.md).
- Framework overlap and named theorem comparison:
  [positioning](CONTROLLED_SENSING_REDUCTION_POSITIONING_20260916.md).
- Scientific interpretation and negative calibration boundary:
  [introduction](ICML_INTRODUCTION_FIRST_PAGE_20260916.md).

All five are existing evidence, not results newly established by this outline.
Independent proof validation and worldwide priority are not certified here.

### Why this structure

Lead with the phenomenon; then explain the two tests; then expose the physical
realization and controls. Standard machinery serves the separation instead of
being presented as the contribution. No additional theorem is needed to finish
this architecture task. Figure production is outside this one-page-outline task.

中文 punchline：**两个学习系统即使在执行能力和最大固定实例渐近信息价格这两个摘要上
完全一致，其最优有限预算学习风险仍可严格不同。资源扩张改变的是可实现反馈协议；
它不必改变上述两个摘要。** 这里不能将“摘要一致”加强成“完整信息几何一致”。

To redirect: specify a subsection or claim for a local revision, without changing
the underlying certificates or diagnostic record.
