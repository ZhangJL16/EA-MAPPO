# ICLR Energy-Safety Story Blueprint

Date: 2026-09-01

Purpose: turn the current pre-results manuscript into a focused ICLR study in
which collision safety is frozen infrastructure and energy safety is the central
scientific contribution.

## Executive Decision

The research direction is viable, but the current evidence package does not meet
submission standard. The strongest route is:

> Freeze a sampled-data robust collision-safety layer, then test whether
> executed-interface Resource-to-Go gives reliable, decision-relevant energy
> estimates and improves the stranding--throughput frontier under composition
> shift.

Do **not** make a new collision-avoidance method part of the ICLR contribution.
The present simulator has known dynamics, bounded acceleration, static obstacles,
and fixed sample-and-hold execution; these conditions match sampled-data robust
HOCBF more directly than learned or predictive alternatives. Collision filtering
is scientifically important because it changes the executed trajectory and
energy law, but its role is experimental infrastructure and a source of
composition shift.

## Normalized Idea Card

| Field | Decision |
| --- | --- |
| Task | Decide when an agent must irreversibly stop useful work and return to charge under uncertain remaining resource. |
| Gap | Common SOC, distance, nominal-model, and direct-switching rules do not explicitly estimate the stopped resource requirement induced by the executed controller/filter interface. |
| Root challenge | The collision filter changes actions, occupancy, hitting time, and energy; the error matters most near a one-way decision boundary. |
| Core insight | Treat the frozen collision stack as a queryable executed interface and learn the first-passage Resource-to-Go of the resulting closed loop, with calibrated abstention near the viability boundary. |
| Main method | Executed-interface Resource-to-Go estimator plus reliability-aware ReturnManager. |
| Primary comparison | Same estimator and information contract at the nominal interface. |
| Decision baselines | SOC threshold, geometric/distance threshold, direct switcher, and one matched high-level constrained-control comparator. |
| Theoretical role | Identify when nominal observations are insufficient; bound executed-interface error under explicit occupancy/coupling conditions; connect calibrated error to boundary stability. |
| Primary endpoint | Stranding probability versus useful throughput, using paired frozen tasks/seeds and uncertainty intervals. |
| Kill test | If an Oracle with the same decision timing and information contract has no headroom over strong baselines, stop the learned-manager claim. |
| Non-contribution | HOCBF/QP projection, sampled-data compensation, ensemble uncertainty, and threshold switching by themselves. |

## Collision-Layer Decision

### Recommended stack

Use a **sampled-data robust HOCBF-QP** with:

- declared double-integrator/acceleration model and sample-and-hold interval;
- hard input bounds inside the feasibility problem;
- an inter-sample residual or reachable-set margin;
- explicit measurement/model-error bounds where they are actually supported;
- a safe initial-set contract;
- a solver-failure and infeasibility backup whose safety is itself checked;
- exact swept/ZOH clearance, not endpoint distance alone;
- logging of intervention, infeasibility, violation, fallback, and collision events.

This choice is grounded in established HOCBF work
([Xiao and Belta](https://doi.org/10.1109/TAC.2021.3105491)), sampled-data CBF
conditions ([Breeden et al.](https://doi.org/10.1109/LCSYS.2021.3076127)), and
robust sampled-data extensions
([Oruganti et al.](https://arxiv.org/abs/2309.08050)). It is a “best fit under the
declared contract,” not a universal state-of-the-art claim.

### Decision matrix

| Candidate | Fit to current scene | Evidence/cost | Decision |
| --- | --- | --- | --- |
| Existing ordinary HOCBF | Poor: ignores the key sampled-execution mismatch in its default R3 form | R3 records 2,242 collision steps; cannot support hard-safety language | Reject as the frozen layer |
| Sampled-data robust HOCBF | High: matches known model, bounded input, fixed timestep, and existing code path | Controlled prototype is encouraging, but formal integration/fallback audit remains | **Adopt and validate** |
| Composite CBF, Harms et al. ICRA 2025 | High when dense raw LiDAR scalability is central | Strong recent quadrotor evidence, but faithful reimplementation is a separate workstream | Cite as strongest recent comparator; implement only if scalability is required |
| Predictive safety filter | Medium | Adds online planning, model, terminal-set, and compute obligations | Related work or optional expensive baseline |
| Learned/neural/black-box filter | Low under known dynamics/map | Adds training and verification burden, blurs energy novelty | Exclude from core paper |
| Energy-gradient collision filter | Technically plausible | Couples collision and energy mechanisms, obscuring causal attribution | Exclude from core mechanism |

The local `aggregate_hocbf` prototype must not be labeled Composite CBF: it is not
a faithful implementation of Harms et al., and its existing diagnostic performance
is poor. A recent predictive sampled-data quadrotor method may be mentioned as a
preprint signal rather than peer-reviewed evidence
([Gao et al.](https://arxiv.org/abs/2510.05456)).

## Frozen Collision Contract

Before any energy result is admissible, record and hash-bind:

1. filter implementation and configuration;
2. nominal policy checkpoint and observation/action preprocessing;
3. sample time, integration scheme, input limits, obstacle inflation, and agent
   radius;
4. perception availability, range, noise, latency, and dropout assumptions;
5. allowed initial-state set and recovery behavior;
6. QP solver tolerances, infeasibility definition, and backup action;
7. collision and minimum-clearance computation over the complete swept segment;
8. evaluation task set, seeds, intervention rate, runtime, infeasibility,
   fallback, violation, and collision counts.

“Hard safety” is allowed only under the stated assumptions and only if the proof
and execution contract cover the backup/fallback path. Zero observed collisions
is an empirical result, not a certificate. If perception dropout is out of scope,
say so explicitly; if it is in scope, a safe backup is required.

## Why the Energy Evidence Must Restart After the Freeze

The safety layer is upstream of every energy estimand:

```text
nominal action
      ↓
collision filter ──→ executed action ──→ trajectory / hitting time ──→ energy
      │                                                        ↓
      └──────── changes intervention and occupancy ──→ Resource-to-Go law
```

Replacing ordinary HOCBF with sampled-data robust HOCBF changes the executed
action distribution, path length, task time, and return feasibility. Therefore
the existing battery calibration can remain historical evidence, but it cannot
authorize Oracle, learned-estimator, or Pareto claims for the new stack. The
correct order is collision freeze, then calibration, then Oracle headroom, then
learning.

## Single Scientific Story

### Problem statement

In resource-limited autonomy, the agent must decide whether to continue useful
work or begin an irreversible return. A collision filter may keep geometric
constraints under a declared contract while changing the policy's commands. The
resource decision is therefore made under a closed-loop composition shift: the
nominal policy does not determine the actual energy-to-go distribution.

### Hypothesis

At the same observation and decision-information budget, an executed-interface
Resource-to-Go estimator will be better calibrated near the return boundary than
a nominal-interface counterpart, and that difference will reduce stranding at
matched useful throughput. If prediction improves without decision improvement,
the method claim fails.

### Contributions to target

1. A precise formulation of irreversible return as a stopped first-passage
   resource problem under a fixed action-transforming safety interface.
2. A conditional identifiability/error analysis showing which executed-interface
   quantities nominal observations do not recover and when estimation error
   affects boundary decisions.
3. A reliability-aware executed-interface estimator/ReturnManager evaluated
   against matched nominal, direct, heuristic, and constrained-control baselines.
4. A paired evaluation showing the full stranding--throughput frontier under
   declared composition shifts, with collision safety held fixed rather than
   credited to the energy method.

These are target contributions, not current claims. Each becomes claimable only
after its evidence Gate passes.

## Minimal Experiment Program

### Gate C0 — Collision infrastructure

- Compare no filter, ordinary HOCBF, and sampled-data robust HOCBF under the same
  frozen nominal actions/tasks.
- Primary checks: exact swept collisions, minimum clearance, QP infeasibility,
  unsafe fallback, violation events, runtime, and intervention rate.
- Declare sensing conditions separately: exact/bounded-noise main contract and
  dropout/combined stress as out-of-contract diagnosis unless a verified backup
  is added.
- Promotion rule: no hard-safety wording unless proof assumptions and all code
  paths align. Otherwise use “collision-safety filter under X assumptions.”

### Gate C1 — Energy recalibration

- Re-estimate capacity/endurance under the frozen collision stack.
- Require true-depletion endpoints, zero unreported censoring, disjoint calibration
  and evaluation seeds, and the predeclared tolerance.

### Gate C2 — Oracle Decision Headroom

- Use the exact future executed rollouts only as an Oracle diagnostic.
- Compare against SOC and geometric thresholds at matched throughput or stranding.
- Run on unseen paired tasks/seeds.
- Kill rule: if the Oracle cannot materially improve the frontier, stop; do not
  train an estimator to recover nonexistent decision headroom.

### Gate C3 — Estimator mechanism

Use one main estimator family and a factorial that answers one question:

| Interface | Supervision | Purpose |
| --- | --- | --- |
| Nominal | direct/Monte Carlo | strong matched counterfactual |
| Executed | direct/Monte Carlo | isolates interface effect |
| Executed | TD or compact model-based variant | tests whether temporal structure adds value |

Do not promote every candidate. Deep ensembles are a reliability device, not a
novelty claim. SIRP remains gated and should disappear if it does not produce a
clear, predeclared advantage.

Report overall and boundary-stratified MAE, calibration/coverage, false-safe rate,
and decision regret. The boundary stratum must be declared before seeing the final
test results.

### Gate C4 — Decision relevance

Required baselines under the same observation/history/compute contract:

- SOC threshold;
- distance/geometric threshold;
- nominal-interface Resource-to-Go;
- executed-interface Resource-to-Go;
- direct continue/return classifier or switcher;
- one high-level constrained-control comparator if it can be matched fairly.

Primary plot: paired stranding probability versus useful throughput, with
confidence regions. Secondary metrics: completed tasks, returns, wasted reserve,
decision regret, time-to-return, collisions, filter intervention, solver/fallback
events, and compute latency.

### Gate C5 — Reliability and abstention

- Compare point estimate, uncertainty margin, and abstention/early-return rule.
- Show coverage and calibration globally and near the boundary.
- Separate epistemic shift from sensing failures. An ensemble cannot certify
  obstacle dropout or unmodeled dynamics.

### Gate C6 — Bounded transfer

Choose one affordable shift that exercises the stated mechanism, for example:

- obstacle-density/layout shift that changes filter interventions; or
- nominal-policy shift with the same collision layer; or
- battery/drag perturbation within declared calibration bounds.

A second simulator is helpful but not mandatory if the claim is explicitly
bounded and the mechanism test is strong. Do not promise broad robotics
generality from one deterministic environment.

## Claim Ladder

### Allowed now

- The manuscript formulates an executed-interface Resource-to-Go hypothesis.
- The repository has a governed calibration/Oracle Gate process.
- The selected R3 platform does not pass a deployment-quality collision-safety
  gate.
- Controlled diagnostics motivate sampled-data robustness but do not certify it.

### Allowed after C0--C2

- The chosen collision layer met the declared empirical contract on the test set.
- The recalibrated system has measurable Oracle decision headroom, if observed.

### Allowed after C3--C5

- Executed-interface estimation improves specified calibration/error metrics, if
  statistically supported.
- Reliability-aware switching improves the paired stranding--throughput frontier,
  if supported.

### Never without extra proof/evidence

- unconditional collision safety;
- universal state of the art;
- generalization to arbitrary robots, maps, sensing failures, or dynamics;
- certification from zero observed collisions;
- novelty for standard HOCBF, QP projection, ensembles, or threshold switching.

## Scope Cuts

For the main paper, cut or demote anything that does not identify the energy
mechanism:

- no new collision-network architecture;
- no energy-gradient term inside the collision barrier;
- no faithful Composite CBF reimplementation unless dense LiDAR scale is a
  required experimental axis;
- no large menu of direct, TD, predictive, planning, model-based, SIRP, CMDP, and
  second-domain variants in the main claim;
- no replication claim until one primary causal mechanism and frontier are
  complete.

The minimum viable ICLR paper is one executed-interface method, one matched
nominal counterfactual, one direct switcher, two simple heuristics, and a
predeclared Oracle kill test.

## Story Structure for the Revision

1. **Introduction:** irreversible resource decision; collision layer changes
   execution; energy reliability is the unsolved question.
2. **Problem:** fixed collision contract, nominal/executed interfaces, stopped
   Resource-to-Go, stranding/throughput endpoint.
3. **Analysis:** non-identifiability and conditional error/boundary stability;
   state assumptions next to each claim.
4. **Method:** one executed-interface estimator and reliability-aware manager.
5. **Experimental contract:** first document collision freeze and recalibration,
   then paired energy evaluation.
6. **Results:** mechanism first, decision frontier second, shift/limitations last.
7. **Limitations:** sensing/model/fallback conditions, no unconditional collision
   certificate, single-system scope, cost of Oracle supervision.

The collision filter should occupy a short, precise infrastructure subsection.
Its benchmark belongs in the appendix unless collision timing robustness changes
the paper's main causal conclusion.

## Kill Criteria and Fallback Publication Route

| Failure | Interpretation | Required response |
| --- | --- | --- |
| C0 cannot eliminate unsafe fallback/collision under the declared contract | Infrastructure is not mature enough | Stop energy claims or narrow the operating contract honestly |
| Recalibration fails | Resource estimand is unstable | Repair the physical/measurement model before Oracle work |
| Oracle has no frontier headroom | No useful decision signal under this setup | Stop learned ReturnManager; publish a diagnostic/negative result only if sufficiently general |
| Executed estimator does not beat matched nominal estimator | Interface hypothesis unsupported | Do not claim executed-interface advantage |
| Prediction improves but decision frontier does not | Surrogate is not decision-relevant | Reframe as estimation analysis, not safety/control improvement |
| Gain exists only under one handpicked seed/threshold | Selection artifact | Reject promotion; expand paired inference or stop |

Fallback: if Oracle headroom fails but the collision-induced occupancy shift and
non-identifiability result are robust, a narrower theory/benchmark paper may be
possible. It must be presented as an analysis of when return prediction is or is
not learnable, not as a successful learned safety manager.

## Submission-Critical Schedule

ICLR 2027 lists the abstract deadline as 2026-09-18 and the paper deadline as
2026-09-25 AOE
([Author Guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines)). With the
current date of 2026-09-01, the complete original experiment matrix is not a
credible minimum plan. The only defensible schedule is:

1. freeze and audit C0 immediately;
2. recalibrate C1;
3. execute the Oracle C2 kill test;
4. continue only one minimal estimator/decision route if C2 passes;
5. reserve final time for paired inference, disclosure, anonymization, proof, and
   build audit.

If C2 has not passed with enough time for C3--C5, do not submit the current paper
as a completed empirical contribution. Preserving claim integrity is more
valuable than filling result cells with underpowered evidence.

## Compliance Repair

The final AI-use statement must describe actual assistance, including all
applicable work on theory/claims, proofs, hypotheses, methodology, experiment
design, implementation, interpretation, code/artifacts, literature, and writing.
ICLR's policy makes authors responsible for all content and requires disclosure
of multiple scientific uses, not merely generated prose
([AI Policy for Authors](https://iclr.cc/Conferences/2027/AIPolicyForAuthors)).
An author rewrite does not retroactively narrow the development history.

## Bottom Line

The paper's best idea is not “a safer collision filter.” It is that resource
safety must be learned and evaluated at the executed closed-loop interface, and
that better prediction matters only if it changes an irreversible decision
frontier. Freeze a mature sampled-data robust HOCBF as infrastructure, restart
energy calibration after that freeze, make Oracle headroom the kill test, and let
paired stranding--throughput evidence decide whether the ICLR claim survives.
