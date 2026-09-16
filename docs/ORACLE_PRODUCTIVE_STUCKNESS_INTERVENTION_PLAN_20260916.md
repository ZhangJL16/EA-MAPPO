# Oracle Productive-Stuckness Intervention — Decisive Experiment Plan

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-09-16
- Verification Status: REPOSITORY-AUDITED, NOT YET EXECUTED
- Version Label: oracle_stuckness_plan_v1
- Prior lineages: PAI-v3 and PSPS-v1 are evidence/design inputs only
- Forbidden reuse: unopened PAI/PSPS CONFIRM worlds and the paused PSPS raw run

## Experiment Overview

- **Title:** Oracle Productive-Stuckness Intervention
- **Objective:** Determine whether perfect privileged knowledge that the current
  task will fail to complete within a fixed future horizon causes a meaningful
  improvement in long-run completed tasks per simulated hour.
- **Primary hypothesis:** A privileged intervention that immediately abandons
  doomed task attempts will increase tasks/hour by reducing long unproductive
  task-mode dwell, without deriving its gain from fewer contacts or fewer energy
  failures.
- **Decisive alternative:** If the oracle has no meaningful effect, productive
  stuckness is not a causal throughput bottleneck. If a simple timeout or legal
  no-progress rule captures essentially all oracle gain, the phenomenon is real
  but too trivial to support the proposed scientific story.
- **Type:** paired controlled simulation; no training.

## Setup

### Frozen Scope

This experiment is the only authorized empirical gate in this lineage. It does
not train a predictor, actor, critic, history encoder, or representation. It
does not access any existing CONFIRM split. It does not alter collision physics,
the frozen SAC navigator, LiDAR inputs, battery capacity, recharge mechanics, or
the keyed task generator.

The following existing repository components are reused unchanged:

- `experiments/directional_navigation/threshold_stress.py` for unlimited
  task/episode clocks and repeated service;
- `experiments/directional_navigation/battery_sortie.py` for return commitment;
- `experiments/directional_navigation/standard_baselines.py` for exact simulator
  snapshot/restore;
- `scripts/run_dvoi_h_collection.py::FrozenNavigator` for the frozen
  deterministic low-level SAC policy;
- the user-locked nonterminal collision recovery semantics.

The paused PSPS-v1 run remains untouched. Its current status is 27 committed
worlds, 35 accessed worlds, and `PAUSED_RESUMABLE`; it is not an input dataset
for this experiment.

## Research Question and Estimand

For physical world \(w\) and high-level intervention policy \(m\), let

\[
  Y_{w,m}
  :=\frac{3600N_{w,m}(T)}{T}
\]

be completed tasks per simulated hour over a fixed horizon \(T\). The primary
causal estimand within the simulator is the paired world-average intervention
effect

\[
  \Delta_{\rm oracle}
  :=\mathbb E_w[Y_{w,\rm oracle}-Y_{w,\rm base}].
\]

The scientific question is not whether future information predicts stuckness;
it is whether acting on perfect future information changes long-run throughput.

## Plant and Evaluation Horizon

- Fixed navigation model: existing frozen SAC checkpoint.
- Obstacles: 48, matching the high-stuckness diagnostic setting.
- Initial SOC: 1.0.
- Base return policy: commit return when SOC is at or below 0.40.
- Per-task and whole-episode artificial guards: disabled inside the continuing
  evaluation plant.
- Evaluation duration: \(T=7200\) simulated seconds per method/world.
- Recharge: instantaneous service at the charger using the existing plant;
  physical travel to the charger remains part of elapsed time.
- Collision behavior: contact repairs position, zeros velocity, and continues;
  one unified contact count at most per policy step.

The two-hour horizon is fixed before outcome access. Runs terminating through
energy exhaustion remain in the denominator and are not dropped.

## Fresh Worlds and Pairing

- Development-only gate: 24 fresh physical worlds.
- One excluded smoke world, disjoint from the 24 formal worlds.
- Every world is evaluated under every registered method.
- World generation and method order are frozen before formal access.
- The same keyed physical world and task schedule seed are used across methods.
- No existing PAI, PSPS, CMI, smoke, DEV, or CONFIRM identity may overlap.
- Whole-world paired bootstrap is the inferential unit; policy steps are never
  treated as independent replicates.

This is a phenomenon gate, not a final generalization claim. No CONFIRM split is
allocated until this gate survives.

## Registered Methods

### M0 — Base SOC40

Continue toward the active task while SOC \(>0.40\); commit return at SOC
\(\le0.40\). This is the reference policy.

### M1 — Fixed-timeout Return

Use M0, but commit return when the current task clock reaches 4000 policy steps,
whichever trigger occurs first.

### M2 — Legal no-progress Return

Use M0, but maintain a legal rolling window of 256 policy steps. Commit return
when all conditions hold:

1. task mode throughout the window;
2. no task completion in the window;
3. net Euclidean task-distance progress is non-positive;
4. SOC remains above 0.40, so the event is a proactive progress intervention.

The window equals the already investigated 64 samples at stride four. The
zero-progress threshold is fixed, untuned, and uses only online task-relative
position information already present in the legal observation.

### M3 — Legal no-progress Replan

Use the same trigger as M2, but abort the current task and request the next keyed
task without returning or recharging. The physical position and remaining
energy are preserved; velocity is set to zero exactly as in the existing
continuous-task rollover. This method tests whether any gain is merely a
trivial task-reset effect.

M3 is an experiment-only high-level action and must be reported separately from
the original Continue/Return action set.

### M4 — Privileged completion oracle Return

At the beginning of every new task, provided SOC \(>0.40\):

1. snapshot the complete simulator and RNG state;
2. force Continue under the frozen navigator for at most
   \(K=4000\) policy steps, with return commitment disabled in the probe;
3. label the task `doomed` if it does not complete before \(K\), energy
   exhaustion, or another physical terminal;
4. restore the exact pre-probe snapshot;
5. if `doomed`, immediately commit Return; otherwise execute M0.

Probe steps do not advance the evaluation clock, consume energy, alter task/RNG
state, or enter any metric. The oracle label and probe trajectory are privileged
diagnostics only.

The oracle acts at task start, not after observing a 12,000-step branch guard.
Consequently, an oracle benefit cannot be mechanically attributed to that guard.

## Why This Is an Intervention Rather Than Offline Relabelling

Each method is executed forward after its own decision. Oracle trajectories can
change return times, recharge events, later starting states, and future task
encounters. Tasks/hour is measured on the resulting run. No method's outcome is
assembled by selecting branches from another method's logged trajectory.

Snapshot lookahead supplies privileged knowledge, but the effect is the outcome
difference caused by applying the associated action in the simulator.

## Inputs

| Input | Path | Description |
|---|---|---|
| Frozen navigator | `artifacts/new_navigation_energy_global_scale_repair_20260908_v1/repair.json` | Hash-bound SAC and observation feature contract |
| Locked plant | `envs/UAVEnergyDeliverySAC.py` | Physics, task stream, energy, service, and collision behavior |
| Continuing wrapper | `experiments/directional_navigation/threshold_stress.py` | Unlimited clocks and repeated recharge accounting |
| Prior diagnostic | `docs/LATENT_FACTOR_OBSERVABILITY_DEV_RESULT_20260915.md` | Motivation only; no outcome reuse |

## Expected Outputs

| Output | Planned path | Format | Success criterion |
|---|---|---|---|
| Frozen contract | `artifacts/oracle_stuckness_intervention_20260916/contract.json` | JSON | Hash-valid; all worlds and methods assigned pre-access |
| Raw runs | `artifacts/oracle_stuckness_intervention_20260916/raw/` | JSON/NPZ | Complete paired 24-world method grid |
| Analysis | `artifacts/oracle_stuckness_intervention_20260916/analysis.json` | JSON | Whole-world paired estimates and simultaneous intervals |
| Result report | `docs/ORACLE_PRODUCTIVE_STUCKNESS_RESULT_20260916.md` | Markdown | Explicit pass/kill decision and mechanism accounting |

These paths are planned, not yet created except for this plan.

## Outcomes

### Primary

1. tasks per simulated hour;
2. paired oracle-minus-base difference;
3. paired oracle-minus-best-simple-baseline difference, where the simple
   baseline is selected only from M1--M3 by outer leave-one-world-out selection.

The best simple baseline is not chosen on the same test world.

### Mechanism and reviewer-attack outcomes

- total task-mode dwell seconds with no task completion;
- number and duration of task attempts exceeding 4000 policy steps;
- completed tasks, recharge count, and zero-task returns;
- energy exhaustion rate and time alive;
- unified collision count;
- return-travel time;
- oracle-doomed task count and oracle intervention count;
- timeout/no-progress/replan intervention counts;
- fraction of oracle gain captured by the best simple baseline.

The mechanism report must separately show whether throughput changes came from
less unproductive dwell, less return travel, different energy failure, or
different contact behavior.

## Analysis Plan

- Unit: physical world.
- Point estimates: mean and median paired differences across worlds.
- Uncertainty: 20,000-draw whole-world paired bootstrap with one simultaneous
  max-t family covering the two primary contrasts.
- Descriptive robustness: paired sign count and world-level scatter.
- No policy-step p-values and no seed-level pseudo-replication.
- No threshold or window tuning after formal access.

### Gate O — Oracle causal headroom

Pass only if all conditions hold:

1. simultaneous 95% lower bound for oracle minus base tasks/hour is strictly
   positive;
2. mean relative gain over base is at least 10%;
3. at least 60% of worlds have a positive paired oracle effect;
4. oracle intervention occurs in at least five worlds.

Failure closes the productive-stuckness causal story.

### Gate M — Claimed mechanism

Conditional on Gate O, pass only if:

1. oracle reduces mean unproductive task-mode dwell by at least 20%;
2. energy-exhaustion incidence does not decrease enough to account for more than
   half of the added completed tasks under an explicit failure-time accounting;
3. the result is not driven by excluding failed runs or changing collision
   semantics.

Failure means the throughput effect exists but the proposed stuckness mechanism
is unsupported.

### Gate N — Nontriviality against simple intervention

Conditional on O and M, the strong scientific story passes only if:

1. the simultaneous 95% lower bound for oracle minus cross-fitted best-simple
   tasks/hour is strictly positive; and
2. the best simple baseline captures less than 80% of the oracle-minus-base
   mean gain.

If O/M pass but N fails, productive stuckness is real and actionable, but the
current phenomenon is adequately handled by a simple timeout/progress/replan
rule and does not justify a representation-learning story.

## Kill Decisions

| Outcome | Decision |
|---|---|
| Gate O fails | Kill stuckness as the main causal throughput bottleneck |
| O passes, M fails | Kill the claimed mechanism; inspect alternative cause only descriptively |
| O/M pass, N fails | Retain engineering finding; kill Spotlight representation story |
| O/M/N pass | Phenomenon survives; only then design the next observability/control study |

No result in this experiment authorizes neural-method training.

## Leakage and Integrity Checks

1. Oracle probe outcome must never enter M0--M3.
2. Probe trajectories must leave post-restore simulator bytes and RNG states
   identical to the pre-probe snapshot.
3. Probe computation time is wall-clock overhead, not simulated mission time.
4. Formal world identities and method assignment must be frozen before access.
5. Existing CONFIRM identities are permanently denied.
6. The 12,000-step branch guard is absent from the evaluation plant.
7. All methods use the same frozen low-level navigator and locked collision
   mechanism.
8. Every incomplete/failed run remains in the throughput denominator.

## Focused Tests Required Before Launch

1. snapshot--probe--restore produces byte-identical state/RNG continuation;
2. oracle label agrees with a direct 4000-step continuation on a tiny fixture;
3. probe steps do not change simulated time, energy, task count, or logs;
4. M1/M2/M3 trigger exactly at their frozen conditions;
5. M3 preserves position/energy and zeros velocity while advancing one keyed
   task;
6. collision accounting remains one unified count per policy step;
7. tasks/hour denominator remains exactly 7200 seconds for every method;
8. identity registry rejects overlap with every retained experimental lineage.

## Monitoring Configuration

- Formal execution: resumable, one world-method record at a time.
- Tiny smoke: one excluded world, 120 simulated seconds, all five methods.
- Startup check: stop after the first complete formal physical-world method
  grid and verify pairing, hashes, resource use, and zero CONFIRM access.
- Hard timeout for smoke: 30 minutes wall time.
- Formal run: no automatic hard timeout beyond the service limit recorded in
  the launch contract.
- Automatic analysis, confirmation, and training: disabled.
- After startup health: hand back to the user; do not monitor to completion.

## Planned Entry Command

No executable command is registered yet. Under the experiment-agent safety
contract, implementation and execution require a subsequent explicit run
authorization after this plan is accepted. The future command must name the
frozen contract, receipt, output directory, device, and startup stop condition;
it may not reuse the paused PSPS service or artifact root.

## Current Feasibility Verdict

**FEASIBLE WITH A NEW ISOLATED RUNNER.** Existing environment code already
provides repeated recharge, fixed-time throughput, unlimited task clocks,
frozen navigation, and exact snapshot/restore. The only new execution logic is
the five registered high-level interventions, privileged probe isolation, fresh
identity freeze/validation, and paired analysis.

The decisive unresolved issue is empirical, not architectural: whether M4 has
meaningful throughput headroom after M1--M3 are included.
