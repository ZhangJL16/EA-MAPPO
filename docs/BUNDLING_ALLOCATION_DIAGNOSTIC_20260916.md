# Frozen learner allocation diagnostic at T=4096

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate / existing-transcript diagnostic
- Origin Date: 2026-09-16
- Verification Status: ANALYZED; implementation/trace reconstruction, not proof certification
- Scope: existing four resource-path trajectories only; zero new observations

## Determination

**The LP is connected correctly. The finite-budget deviation is caused by
all-path coverage and certificate-first scheduling, not unavailable AB or an
incorrect allocation solution.** No frozen learner/environment/manifest was
changed and no rollout, seed or horizon was added.

This corrects a possible overinterpretation of the calibration: the implemented
procedure is the documented asymptotically attaining learner, not a resource-blind
naive learner silently substituted for it. Finite-T cost is not guaranteed to
equal C log T. The calibration does not establish asymptotic attainability failure
or a systematic gap for standard algorithms.

## Evidence and reconstruction

`scripts/diagnose_bundling_allocation.py` imports the frozen public learner,
feeds ONLY the recorded past feedback, and reconstructs every recorded decision.
It never calls a plant rollout or draws RNG rewards. Chosen paths, reason tags,
completed/exploration counts, log likelihood, totals and pending-path offsets
match the stored checkpoints. Source/prediction hashes remain unchanged.

Command:

```bash
.venv/bin/python scripts/diagnose_bundling_allocation.py --source artifacts/bundling_calibration_startup_v1_20260916
```

Output: `allocation_diagnostic_4096/diagnostic.json` and provenance receipt under
the existing run. Every decision has its time, MLE, confidence set, Q counters,
forced threshold and LP quota targets recorded. The saved source checkpoint
hashes bind this diagnostic to the unmodified calibration.

## Theoretical allocation versus observed allocation

Use completed NONOPTIMAL path counts divided by log T. Optimal A execution has
O(T) plays and is not a variable in this LP; including it in a vector-distance
calculation would create a meaningless divergence. The unfinished path remains
a separate bounded contribution to P, not a completed information experiment.

| Cell | Path | lambda-star | Complete N | N/log(4096) |
|---|---|---:|---:|---:|
| low/on | dock | 0 | 8 | .961797 |
| low/on | empty | 0 | 8 | .961797 |
| low/on | B | .377359 | 8 | .961797 |
| high/on | dock | 0 | 8 | .961797 |
| high/on | empty | 0 | 8 | .961797 |
| high/on | B | 0 | 8 | .961797 |
| high/on | AB | .377359 | 8 | .961797 |

The nonoptimal-coordinate L1 distance is **2.508031** low/on versus **3.469828**
high/on. Off cells equal low/on. These are diagnostics of this realized finite
transcript, not confidence intervals or bounds on expected allocation error.

Cost is the primary invariant: complete-path cost/log T minus C is **.386559**
split versus **.510120** high/on. Counting all coordinates equally is less
scientifically useful than showing that the B-only coordinate has target zero
but remains costly at high capacity. No claim about arbitrary LP optimizer
uniqueness or vector convergence follows from this calibration table.

## Four component checks

### Allocation solver — correctly uses calendar cost

Against theta_B, B-only and AB supply the SAME B-channel KL; the A channel adds
zero information against that alternative. Costs are .25 versus .05. High/on
LP solution is therefore B=0, AB=1/kl(.05,.95)=.377359, with C=.01886796.
Low/on solution is B=.377359, C=.09433980. The in-memory model-indexed solutions
match sealed predictions. Objective is c, not c/length, for resource-path learning.

### Feasibility — AB not masked out

AB is present in the high/on catalogue and actually executes eight complete
times. It is absent in the other three cells. Primitive trajectory integrity
was already checked; no safety or motion change is needed to access this path.

### Forced schedule — covers path types, not just informative channels

The theoretical procedure initializes EVERY path and uses min_p Q_p < sqrt(j),
j=sum Q, before any LP allocation branch. It covers dock, empty and B-only even
when the truth-specific LP gives them zero allocation.

At T=4096 both cells have j=57 and sqrt(j)=7.549834. Counters:

- low/on Q=(dock8, empty8, A33, B8).
- high/on Q=(dock8, empty8, A25, B8, AB8).

High/on has seven forced B-only sorties and seven forced AB sorties, in addition
to initialization. Three forced B-only decisions occur even after the MLE is
the true hypothesis. Five correct-MLE forced decisions occur while AB quota is
deficient, so forced sampling takes priority over the economically preferred
informative experiment. There are **zero correct-MLE AB information-quota plays**
in this transcript. Low/on has one correct-MLE B information-quota play.

This is intentional general-proof coverage overhead in the documented learner,
not an accidentally missing geometry-aware LP implementation. At this budget
it is material and produces the failed ordering.

### Confidence/certificate — quotas are sufficient, not mandatory

Certificate is checked BEFORE forced sampling and LP quotas. Once all plausible
hypotheses agree on a path, that path is executed and Q is unchanged, but all
observations continue updating likelihood.

At the endpoint f(T)=13.941756 and eta_57=.601965. Buffered true-model LP target
is approximately 8.428 plays of B (split) or AB (high/on). Q is8, below that
target. Nonetheless recorded likelihood supports a certificate:

- split observed log-LR theta_A versus theta_B: 23.555512.
- high/on observed log-LR theta_A versus theta_B: 41.222146.

Both exceed the confidence threshold. The last actual decisions at t=4095
split and t=4094 high/on have the singleton plausible set {theta_A} and choose
A as certified. This does NOT violate the documented policy: filling LP quotas
is a sufficient route to certification, not a constraint that overrides an
already valid likelihood certificate. The first certificates occur at t=147
split and t=155 high/on; no wrong-certificate allegation follows from the
finite ordering.

## What is and is not resolved

Resolved: the actual finite allocation is far from the LP target; costly B-only
plays are forced rather than correctly selected by the high/on information LP.
Frozen code faithfully follows the documented scheduling priorities on all
recorded decisions. The broad asymptotic procedure is not economical at the
approved calibration budget.

Unresolved: independent validity of the attainability proof, system-wide behavior
across seeds/instances, and whether a revised learner provides robust finite-time
benefit. No conclusion that "100 seeds would all fail" or "a longer horizon
cannot help" is supported by this audit; neither action was undertaken.

## Narrow next design direction — proposal, not implemented

A targeted candidate is **informative-channel coverage instead of exhaustive
path-type coverage**. At this fixed public hypothesis class, the necessary
unknown channels are A and B. Dock has a known identical law, empty supplies no
feedback, and high/on AB covers both channels. Thus a coverage basis can exclude
information-redundant paths without consulting private truth.

This does NOT mean simply deleting forced sampling or choosing AB with oracle
true means. It requires public-class coverage of decision-relevant alternatives,
control of wrong-MLE selections, shared primitive likelihood accounting and
reassessment of the proof's minimum-row-sample and exploration-count arguments.
It is a proposal for a revised procedure, not a proved finite-time advantage or
claimed novel algorithm. A cost-aware structured reference must receive the same
coverage improvement; do not manufacture an algorithmic separation.

Any revised procedure must be a separately named post-calibration exploratory
lineage, prospectively specified before its own outcomes. It cannot overwrite
v1, inherit v1's pre-outcome authority, or be presented as v1 confirming the
original prediction. Current instruction authorizes this diagnostic only: no
revised learner was implemented or run.

## Statistical caution

One paired seed; no p-values/CI or asymptotic estimation. The eleven methodological
checks remain as in the result report: no pooling to hide direction, no ecological
generalization, fixed sample selection, no collider adjustment, no base-rate
metric, no regression-to-mean selection, all complete/partial paths retained,
no favorable-method substitution, post-outcome diagnostic explicitly labeled,
no deployment causal generalization and no reverse-causality claim. Broad
interpretation remains CAUTION. Diagnostic execution is not an experiment retry.
