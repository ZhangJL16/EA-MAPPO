# JSEB Navigation Gate Repair Protocol

## Activation Rule

This protocol is dormant until the independent 500-task evaluation of
`checkpoint_transition_500000.zip` produces a complete formal artifact.  It is
activated only if that artifact fails at least one preregistered navigation
predicate.  No implementation or result from downstream battery calibration,
Oracle return management, Energy TD, or Stage C may be used to reinterpret a
navigation failure.

The repair must preserve the research problem:

- 4000 x 4000 x 400 m world;
- 24 static obstacles with 50--120 m radii;
- 128 x 8 LiDAR with range and hit-valid channels;
- the same 20/5 m/s velocity and 5/3 m/s2 acceleration limits;
- HOCBF as the final hard collision authority;
- one 3D continuous SAC action;
- exactly 500,000 training environment transitions;
- the same success, path, boundary, and collision Gate thresholds.

It is forbidden to repair the result by lowering Gate thresholds, reducing
obstacle size/count, increasing LiDAR range, weakening episode censoring,
counting HOCBF intervention as actor success, or continuing the current failed
checkpoint beyond the fixed comparison budget.

The protocol was activated by the completed 500-task evaluation at
`artifacts/jseb500k_navigation_gate_parallel_20260826_211448`.  The frozen
500k checkpoint achieved 0.84 overall success, 0.82 minimum bucket success,
and 1.63125 mean path ratio.  It passed the safety predicates with zero
obstacle-collision steps and a boundary-contact step rate of `3.3311e-06`, but
failed every navigation/readiness predicate.  Battery calibration therefore
stopped with `STOPPED_NAVIGATION_NOT_READY`; no Oracle, TD, or return-decision
result was produced from this checkpoint.

## Evidence-Motivated Hypotheses

### H1: Flat LiDAR representation is sample-inefficient

The current policy receives a 2055-dimensional flat vector:

```text
goal/velocity features: 7
128 x 8 normalized LiDAR ranges: 1024
128 x 8 LiDAR hit-valid mask: 1024
```

SB3's default MLP treats adjacent and circularly neighboring rays as unrelated
coordinates.  The current 500k diagnostic ended at 87.2% cumulative training
success and mean path ratio 1.522 despite nearly one gradient update per
transition.  A structure-aware ray encoder is therefore the first hypothesis;
it is not yet a demonstrated root cause.

### H2: Local Jacobian supervision loses coverage after policy drift

At 496k, projection geometry was valid for 91.4% of collected transitions, but
the mean valid fraction after applying the 0.35 actor-to-anchor trust region was
only 11.1%.  The bridge loss ran on almost every optimizer step, yet most replay
samples did not provide valid local Jacobian supervision to the current actor.

After the protocol was drafted, a logging-only 2k smoke directly measured the
same mechanism in a sampled batch: `0.9922` pre-trust validity fell to `0.03125`
post-trust validity, with action-delta p95 `1.4393` and sample-age p95 `1487.12`.
This smoke establishes instrumentation and the existence of coverage collapse;
it does not establish that recency sampling improves navigation.

The local trust region must not simply be widened: the linearized projection is
not justified arbitrarily far from its anchor.  The hypothesis is instead that
recency-aware sampling can increase local coverage without violating the same
trust-region semantics.

### H3: The current failure is not an update-ratio bug

The 500k checkpoint records 494,992 actor-gradient steps and the training curve
records a 0.9899 update/transition ratio at 496k.  This rules out the previously
identified `gradient_steps=1` vector-environment bug as the primary explanation
for this run.

### H4: The original bridge compared different SAC action samples

Recency-only smokes exposed a second, more fundamental issue.  The projection
Jacobian was collected at the stochastic SAC rollout action, while the bridge
loss evaluated the current deterministic actor mean.  Even after restricting
sample-age p95 to 58.6 transitions, actor-to-anchor action-delta p50 remained
0.98 and post-trust validity remained 1.67%.  Therefore replay age alone cannot
repair the local linearization semantics.

R4 uses common-random-number coupling.  At collection time it infers the base
Gaussian noise that generated the executed stochastic SAC action.  During the
bridge update, the current actor is evaluated with that same base noise before
the unchanged 0.35 trust test is applied.  This compares the same policy-noise
quantile across actor versions; it does not widen the local Jacobian region.  A
2k mechanics smoke with a 512-transition recency window raised mean post-trust
validity from about 1.7% to 95.5%, with action-delta p95 0.175.  These are
implementation diagnostics, not navigation-performance evidence.

## Minimal Controlled Comparison

Only one factor changes at a time.

| ID | LiDAR encoder | Jacobian bridge | Bridge replay policy | Scientific question |
| --- | --- | --- | --- | --- |
| R0 | flat default MLP | current | uniform 50k replay | archived current reference |
| R1 | structured 2D ray encoder | off | none | does representation alone fix navigation? |
| R2 | structured 2D ray encoder | current | uniform 50k replay | does the current Jacobian add value after representation is fixed? |
| R3 | structured 2D ray encoder | current | recency-controlled, same 0.35 trust region | does valid local bridge coverage add value? |
| R4 | structured 2D ray encoder | noise-coupled current bridge | recency-controlled, same 0.35 trust region | does matched stochastic-action supervision add value? |

R0 is the archived run and is not extended.  R1--R4 use the same seed/task
sampling protocol, replay capacity for SAC, optimizer budget, rewards, physics,
obstacles, HOCBF, and evaluation tasks.  R1 disables only the auxiliary bridge;
HOCBF still executes as the hard final safety layer in every method.

R3 is retained as the recency-only control even though its mechanics smoke did
not restore coverage.  R4 changes only action-sample correspondence relative to
R3.  Thus R3 versus R4 isolates common-random-noise coupling rather than
confounding it with the structured encoder or replay window.

The first pilot may use seed 0 to kill implausible variants.  Any method that is
promoted as a research result must subsequently use at least three preregistered
seeds and report uncertainty over seeds.  A single seed cannot establish a
Jacobian advantage.

## Structured LiDAR Encoder Contract

The environment observation contract remains a flat `(2055,)` vector so old
artifacts remain loadable.  A custom SB3 feature extractor slices it into:

- a 7D goal/velocity branch;
- a two-channel `(range, valid)` LiDAR tensor of shape `(2, 8, 128)`.

The LiDAR branch must preserve horizontal circular adjacency.  A minimal
implementation uses circular padding on the 128-ray azimuth dimension,
ordinary bounded padding on the eight elevation rows, small 2D convolutions,
and adaptive pooling.  The 7D branch uses a small MLP; the branch embeddings are
concatenated before the actor/critic heads.

The parameter count and downstream actor/critic widths must be recorded.  R1,
R2, R3, and R4 use the identical extractor and comparable optimization settings, so
the Jacobian comparison is not confounded by model capacity.

Required extractor tests:

1. output shape and finite gradients;
2. exact observation slicing for 7 + 1024 + 1024;
3. azimuth wrap-around equivariance for a circularly shifted ray pattern before
   the goal branch is fused;
4. invalid rays cannot be confused with zero-distance occupied rays;
5. checkpoint save/load preserves deterministic actions;
6. old flat-policy checkpoints still load through their original policy class.

## Recency-Controlled Bridge Contract

R3 may change only auxiliary bridge sampling.  It must retain:

- the same projection Jacobian definition;
- the same 0.35 local trust-region acceptance test;
- the same shield-loss formula and weight;
- the same SAC replay and SAC update budget.

The sampler must expose sample age and report, per logging interval:

- pre-trust valid-Jacobian fraction;
- post-trust valid fraction;
- actor-to-anchor action delta percentiles;
- sampled transition-age percentiles;
- nonzero shield-loss fraction;
- HOCBF intervention and emergency-brake rates.

A recency choice is acceptable only if it increases post-trust valid coverage
without increasing held-out collision steps or degrading navigation.  It must
not select samples using future episode success labels.

For R4, replay additionally stores the inferred three-dimensional base Gaussian
noise.  With unchanged actor parameters, the coupled action must reconstruct
the collected normalized SAC action.  Tests cover reconstruction, missing-noise
rejection, finite stored noise, and checkpoint round trips.  The coupled action
remains differentiable through the current actor mean and standard deviation.

## Evaluation and Gates

Mechanics are first checked with a short smoke that is explicitly non-scientific.
Scientific comparison then uses exactly 500,000 training transitions and
phase-end-only deterministic evaluation.  Evaluation interactions are excluded
from every training budget.

Every candidate is judged by the unchanged navigation Gate:

1. overall success rate at least 0.98;
2. each distance-bucket success rate at least 0.95;
3. mean path ratio at most 1.10;
4. boundary-contact step rate below 0.01;
5. zero obstacle-collision steps.

Secondary diagnostics include task steps, HOCBF intervention/emergency rates,
nominal-safe action rate, and valid bridge coverage.  They cannot compensate
for a failed primary Gate.

The current selection task set may adjudicate repair candidates because it is
already labeled as model selection.  It must not become the final paper test
set.  A separately seeded, immutable final test set is generated only after the
architecture and hyperparameters are frozen.

## Decision Rules

- **R1 passes, R2/R3 do not improve:** retain the structured navigation policy
  and drop Jacobian navigation claims.
- **R2 improves over R1 across seeds:** the Jacobian bridge has evidence of
  value beyond representation.
- **R3 improves R2 while preserving safety:** replay-policy drift was a material
  limitation of local bridge supervision.
- **R3 remains invalid but R4 restores coverage/performance:** stochastic action
  mismatch, rather than replay age alone, was the material bridge defect.
- **No candidate passes:** stop downstream return-to-charge experiments and
  revisit the navigation formulation; do not spend Energy-TD compute on an
  unready policy.
- **A candidate passes:** freeze the selected 500k checkpoint and resume the
  fail-closed calibration -> Oracle -> TD chain.  Navigation remains frozen in
  all decision comparisons.

## Claim Boundary

Passing this repair establishes only navigation readiness in the current static
obstacle simulator.  It does not establish dynamic-obstacle safety, stochastic
Energy-to-Go calibration, Oracle return headroom, an ICLR-level compositional
claim, or superiority over executed-action world-model baselines.  Those remain
separate downstream Gates in the completion ledger.
