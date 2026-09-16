# Calibration-only implementation handoff

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: implementation preparation
- Origin Date: 2026-09-16
- Verification Status: UNVERIFIED — focused tests passed; no scientific run
- Version: bundling-calibration-startup-v1
- Authority: one calibration fixture only; no routing library or seed sweep

## Implemented boundary

The isolated runner uses the proposal's equilateral route geometry, fixed
three-hypothesis class, four capacity/protocol cells and five methods. It imports
no legacy plant, navigator, policy or experiment artifacts. All mission steps
advance time, pose and energy together; reload follows a legal nonnegative debit.
AB/BA have identical channel counts and lengths; a fixed AB representative is
used for learning, while the primitive-state audit includes both possible orders.

The question is: **Does resource capacity alter learning difficulty when optimal
execution capability is held fixed?** The high/off versus low/off contrast is a
capacity-only negative control within the SAME bundling-off protocol and feasible
catalogue. The on/off intervention changes an explicitly known operational action
restriction, not geometry or reward laws.

## Pre-outcome audits

`prepare` creates the gain/optimal-path, confusing-alternative and independent
reachable-state AROE audits before any scientific reward draws. It then seals
their prediction bytes with source hashes in a manifest. Resume refuses changed
sources or predictions. Primary predictions remain:

| Cell | Gain | Span | B-query calendar cost | C |
|---|---:|---:|---:|---:|
| low/on | .1 | .2 | .25 | .09433979774864129 |
| high/on | .1 | .2 | .05 via AB | .01886795954972826 |
| low/off | .1 | .2 | .25 | .09433979774864129 |
| high/off | .1 | .2 | .25 | .09433979774864129 |

The learner receives public hypotheses, public routes and past observations only.
It receives no private truth, prediction file, evaluator summary or future draws.
This is capability/API isolation for these reviewed implementations, NOT an OS
sandbox against hostile code running as the same operating-system user.

The strongest cost-aware reference has an independent scalar decision scan but
deliberately implements the SAME finite-hypothesis theoretical scheduling rule.
It is not a stock OSSB package or a claimed algorithmic improvement. Exact ties
with resource-path learning are expected. Cost-blind calibration can also tie.
The oracle allocation schedule is privileged and deliberately explores; it is
not an optimal execution policy or a universal finite-time upper bound.

## Focused verification

Command executed:

```bash
cd /home/zjl/mappo
.venv/bin/python -m pytest tests/test_bundling_calibration.py -q
```

Result: **9 passed**. Coverage: predictions/optimal decisions/alternatives;
all four cells' route clock, pose, debit/reload and masks; reference equivalence;
learner interface leak denial; prediction/source sealing; incomplete-sortie
resume identity including policy and RNG; capacity-only observable continuation.
Short test trajectories are excluded integrity fixtures, not scientific samples.

## Proposed exact startup command — awaiting confirmation

```bash
cd /home/zjl/mappo
.venv/bin/python scripts/run_bundling_calibration.py prepare --output artifacts/bundling_calibration_startup_v1_20260916
.venv/bin/python scripts/run_bundling_calibration.py startup --output artifacts/bundling_calibration_startup_v1_20260916
```

The startup command uses ONE fixed paired seed (20260916), all four cells and
five methods, stops at primitive time 128 per trajectory and writes resumable
checkpoints plus an integrity-only health receipt. It does not print scientific
outcomes, perform formal inference or continue automatically. Source hashes
and prediction bytes are committed before scientific samples are generated.

An explicitly requested `resume` stops at 4096, the proposal's FIRST calibration
horizon; that follow-on command is not authorized by startup. There is no CLI
for descriptors, seeds, truth changes, thresholds or a routing library. Nested
checkpoints belong to the same trajectory, not independent replicates. Exact
continuation is tested; no automatic crash retry is performed.

No scientific calibration or prediction-ranking result is available. One seed
cannot establish expected regret coefficient convergence, confidence intervals
or general embodied-learning relevance. Finite ordering discrepancies must not
be described as refuting the asymptotic theorem or repaired by outcome-driven
changes to hypotheses, horizons, confidence schedules or routes.
