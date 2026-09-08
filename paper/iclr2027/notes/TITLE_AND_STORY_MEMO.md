# Title and Story Memo

**Status:** Working narrative for approval; empirical claims remain pending.  
**Target:** ICLR 2027, oral-candidate evidence standard.  
**Evidence boundary:** The current simulator supports deterministic point Resource-to-Go and epistemic prediction reliability. HOCBF remains the collision-safety layer. The learned estimator is not a hard safety certificate, and SIRP remains a gated candidate rather than an assumed final method.

## A. Candidate Titles

### Problem-first

| # | Candidate title | Clarity | Memorability | ML generality | Overclaim risk | Reviewer expectation |
|---:|---|---|---|---|---|---|
| 1 | **When Should an Agent Return? Reliable Resource-to-Go under Closed-Loop Composition Shift** | High: names the decision and the prediction problem | High: the opening question is easy to retain | High: applies beyond UAVs and batteries | Low--medium: “Reliable” states the target, while evidence must remain bounded in the abstract | A decision-centered ML paper connecting prediction, shift, and autonomous control |
| 2 | **When to Stop Working and Recharge: Resource-to-Go Reliability for Autonomous Agents** | Very high for the application question | High, but “working” is slightly informal | Medium--high: “recharge” narrows the resource type | Low | A practical autonomy paper; the compositional-shift mechanism is not visible in the title |
| 3 | **Return Before It Is Too Late: Reliable Resource-to-Go for Persistent Autonomy** | High, though less technically specific | Very high | High | Medium: can sound safety-promotional unless carefully bounded | A broad persistent-autonomy paper with strong mission-level evidence |

### ML-mechanism-first

| # | Candidate title | Clarity | Memorability | ML generality | Overclaim risk | Reviewer expectation |
|---:|---|---|---|---|---|---|
| 4 | **Executed-Interface Identifiability for Resource-to-Go under Policy--Safety Composition Shift** | High for theory-aware readers | Medium | High | Medium: foregrounding identifiability invites a theorem-first paper and complete proof audit | Formal identification results plus experiments that isolate support from pair novelty |
| 5 | **Resource-to-Go under Compositional Closed-Loop Shift: Interface Support, Error Accumulation, and Return Decisions** | High but dense | Medium | Very high | Low | A mechanism paper with a complete theory-to-experiment chain |
| 6 | **Predicting Resource-to-Go through the Executed Interface** | High and concise | High | High | Low | A prediction-method paper; shift, reliability, and the stopping decision are under-specified |

### Balanced

| # | Candidate title | Clarity | Memorability | ML generality | Overclaim risk | Reviewer expectation |
|---:|---|---|---|---|---|---|
| 7 | **Reliable Return-to-Replenishment under Policy--Safety Composition Shift** | High after “replenishment” is defined | Medium | Very high | Medium: “reliable return” can be misread as a safety guarantee | A general resource-aware autonomy framework validated across domains |
| 8 | **From Executed-Interface Extrapolation to Return Decisions: Reliable Resource-to-Go for Autonomous Agents** | High, with the causal chain visible | Medium--high | High | Low--medium | A diagnostic-to-decision paper with matched-shift experiments and mission outcomes |

### Recommendation

**Recommended working title: _When Should an Agent Return? Reliable Resource-to-Go under Closed-Loop Composition Shift_.**

This title wins on the first-page anchor: it starts from the irreversible decision that gives the work practical and scientific meaning, then names the ML object and the source of difficulty. It is broader than UAV energy management without hiding the actual mechanism. Candidate 5 is the strongest fallback if the completed evidence becomes primarily a theory-and-benchmark contribution rather than a return-decision contribution.

## B. One-Sentence Falsifiable Thesis

> **Under a shared executed-action primitive and queryable policy and safety operator, an unseen pair's charger Resource-to-Go remains identifiable when the primitive is identified over its target executed-interface occupancy; empirically, we hypothesize that error is governed by interface extrapolation and hitting horizon and becomes mission-critical mainly when underestimation crosses the return boundary.**

The empirical part of this thesis is falsified if matched-horizon interface-identifiable unseen pairs exhibit a systematic pair-novelty penalty, if interface extrapolation and horizon do not predict Resource-to-Go failure, or if boundary-local error does not explain stranding and premature-return behavior better than global prediction error. The identification clause holds only under its stated structural assumptions; the remaining clauses are empirical hypotheses and must not be written as findings before their registered experiments pass.

## C. Abstract Candidates

### C.1 Task-first

Resource-limited autonomous agents must decide when to stop productive work and commit to replenishment: returning too early wastes usable capacity, while returning too late causes stranding. Fixed state-of-charge and distance rules ignore the trajectory the system will actually execute, which is jointly determined by a learned policy, a safety operator, the dynamics, and the resource model. We study reliable charger Resource-to-Go prediction when policy--operator compositions change. Our central hypothesis is that an unseen component pair is not inherently out of distribution; reliability instead depends on whether its executed state--action interface is identifiable and on how local errors accumulate over the remaining hitting horizon. We formalize paired identification and non-identification results, derive an assumption-explicit long-horizon error decomposition, and connect prediction error to irreversible decisions through a return-boundary stability result. The resulting framework combines an executed-interface estimator with a common one-way ReturnManager and introduces additional reliability machinery only if a strong executed-world-model ensemble proves insufficient. **[RESULT PENDING: ORACLE HEADROOM] [RESULT PENDING: PAIR VS INTERFACE] [RESULT PENDING: FINAL RETURN PARETO]**. This formulation separates collision safety, retained by the hard safety operator, from the learned question of when Resource-to-Go predictions are trustworthy enough to support return decisions.

### C.2 Theory-first

An unseen policy--safety-operator pair need not define an unidentified closed loop. When the policy and operator are queryable, the executed-action primitive is shared, and that primitive is identified on the target executed interface, the charger hitting-resource law is identifiable without pair-level trajectory overlap; outside that interface support, observationally equivalent primitives can imply different Resource-to-Go. Building on this boundary, we decompose finite-horizon prediction error into occupancy-weighted local interface error under explicit coupling and continuation regularity, add a proper-shortest-path truncation term, and show that bounded requirement error can change a threshold return decision only near its return boundary. These results motivate a practical question: when should a resource-limited agent stop its current task and irreversibly return to replenish? We instantiate the framework with executed-interface Resource-to-Go prediction, epistemic reliability, and a common ReturnManager, while retaining a hard safety operator for collision avoidance. The empirical program tests whether the theoretical distinction survives controlled closed-loop shifts and improves mission utility. **[RESULT PENDING: ORACLE HEADROOM] [RESULT PENDING: PAIR VS INTERFACE] [RESULT PENDING: FINAL RETURN PARETO]**. The present deterministic simulator supports point prediction and epistemic reliability, not claims about physical aleatoric tails or learned safety certification.

### C.3 Balanced

When should a resource-limited agent stop working and commit to replenishment? Returning too early sacrifices throughput, but returning too late risks stranding, and neither state of charge nor geometric distance captures the future trajectory produced by a learned policy composed with a hard safety operator. We study charger Resource-to-Go under this compositional closed-loop shift. The key distinction is between pair novelty and executed-interface extrapolation: an unseen policy--operator pair can remain identifiable when its target executed interface is covered, whereas behavior outside that support is not identifiable without additional structure. We pair this identification boundary with an assumption-explicit account of long-horizon error accumulation and a decision-stability result that confines decision changes from bounded prediction error to a band around the return boundary. Our framework uses executed-interface Resource-to-Go estimates in an irreversible ReturnManager, adds epistemic margins or abstention only when justified, and leaves collision safety to the hard operator. **[RESULT PENDING: ORACLE HEADROOM] [RESULT PENDING: PAIR VS INTERFACE] [RESULT PENDING: FINAL RETURN PARETO]**. The planned evaluation tests whether this view predicts failure across controlled policy--operator compositions and improves the stranding--throughput frontier beyond simple heuristics, direct switching, world-model ensembles, and separately trained constrained-control methods.

### Recommendation

**Recommended abstract: C.3 (balanced).** It preserves the memorable stopping problem, exposes the non-obvious interface insight before naming machinery, and makes the theoretical and empirical obligations legible without presenting pending evidence as fact. C.1 is the strongest alternative if the final paper becomes more robotics-facing; C.2 should be used only if the proof audit and interface-support experiments become the dominant evidence.

### Abstract Reviewer Test

| Test | C.1 task-first | C.2 theory-first | C.3 balanced |
|---|---|---|---|
| What is the problem? | One-way return-to-replenishment timing | Identifiability of Resource-to-Go and its use in return timing | One-way return timing under closed-loop shift |
| Why does it matter? | Early return loses throughput; late return strands the agent | Identification determines whether unseen compositions can support decisions | Early/late errors create a mission trade-off |
| What is insufficient? | SOC and distance omit the executed closed loop | Pair-level overlap and unsupported extrapolation arguments | SOC/distance and pair-OOD labels miss the executed interface |
| What is the core insight? | Interface support and horizon, not pair novelty alone | Identifiability is an executed-interface property | Pair novelty differs from interface extrapolation; decision relevance is boundary-local |
| What is the approach? | Interface estimator, theory, common ReturnManager, gated reliability | Paired theory results plus estimator and ReturnManager | Executed-interface estimation, bounded theory, and selective reliability when justified |
| Which results are pending? | Oracle headroom, pair-vs-interface, final Pareto | Oracle headroom, pair-vs-interface, final Pareto | Oracle headroom, pair-vs-interface, final Pareto |

## D. Six-Paragraph Introduction Logic

1. **Decision problem.** A persistent agent must choose the last safe moment to stop productive work and irreversibly return for replenishment, because early commitment reduces throughput while late commitment can strand the agent.
2. **Why simple observables are insufficient.** State of charge and distance omit the executed chain from policy output through safety filtering, dynamics, and realized resource consumption, so equal batteries and equal distances can imply different charger Resource-to-Go.
3. **Existing progress and remaining gap.** Energy prediction, constrained RL, policy-conditioned and compositional models, uncertainty methods, and safety filters each address part of this chain, but none alone establishes when a long-horizon executed-interface prediction remains trustworthy enough for an irreversible return decision under policy--operator recomposition.
4. **Core scientific insight.** Pair novelty is not the same as executed-interface extrapolation, and global Resource-to-Go error is not the same as mission-critical error because only sufficiently large underestimation near the return boundary can flip the stopping decision.
5. **Approach and theory.** We model charger Resource-to-Go through the executed interface, state paired identification and non-identification conditions, trace local model error and hitting-time tails over the horizon, and feed only justified point estimates and epistemic margins or abstentions into a one-way ReturnManager while HOCBF retains collision-safety authority.
6. **Contributions, evidence, and boundary.** We contribute a decision-centered formulation, an interface-support theory package, and a preregistered evidence ladder from Oracle headroom to matched compositional shift and mission Pareto evaluation, while all empirical conclusions, cross-domain generality, independent proof acceptance, and any need for SIRP remain explicitly pending.

## Approval Boundary

Approval of this memo freezes only the working title and top-level narrative. It does not promote any pending experiment, provisional proof, reliability mechanism, or generality claim to a completed contribution.
