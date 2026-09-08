# Fresh HOCBF correction-supervised SAC

2026-09-08. User initially requested code only, then explicitly authorized
launch: “请继续实验”. The initial launch had no automatic evaluation. The later
explicit request for post-training500-task testing is implemented by a separate
successor, without modifying this live trainer. See
`docs/HOCBF_FIXED500_AND_ENERGY_HANDOFF_20260908.md`. No energy run is launched yet.

## Question and minimal change

Can an already competent navigation SAC internalize sensor-based HOCBF action
corrections while retaining arrival efficiency? This is an adaptation experiment,
not a new energy-sustainability result and not a novel projection theorem.

Start from `recovery_sac_ppo_scratch_20260906_v1/sac/checkpoint_000524288/model.zip`:
same2056 observations, all1024 ranges and1024 flags, ordered structured LiDAR,
2081 extracted features, original SAC actor/twin Q networks. Warm start copies
all policy/critic/target parameters and entropy coefficient bit-for-bit. Fresh
optimizers and empty replay prevent mixing old unshielded transitions into the
new shielded transition law. Adaptation steps are counted separately from524288.

The user-locked collision recovery/reward/count protocol is inherited unchanged.
HOCBF operates at physics substeps, but contact still repairs position, zeros all
velocity components at the repair instant, and continues the same task. All
statistics use one unified contact count per policy step. The dormant existing
safety-intervention reward coefficient is explicitly set to0: this experiment
must not silently add a new reward penalty when HOCBF becomes active.

## Actor loss and action semantics

Keep ordinary SAC reward/entropy/nominal-action critic and Bellman equations.
For a uniformly sampled subbatch of its replay observations, reuse the actor's
CURRENT reparameterized sampled actions from that update. Map to physical
acceleration with the SAME horizontal disk clipping and component scales as the
environment, query HOCBF anew, and add

    L_actor = L_SAC + lambda * mean_i [m_i / 2 * ||u_i - stopgrad(u_i*)||^2].

This supervises sampled actions, not a deterministic mean against an unrelated
historical random-action label. Targets never use future rollout information or
the mean of subsequent physical-substep actions. No additional critic, encoder,
prediction model, cached Jacobian, local trust-region mask, or trajectory
simulation is introduced. Gradient passes through the actuator map to the SAC
mean AND scale parameters. The physical-unit squared loss is not numerically
comparable to historical normalized-action JSEB loss/weight. The actuator map
itself can have zero derivatives in saturated directions, so a nonzero physical
correction does not guarantee a nonzero parameter gradient in every sample.

For a nonempty closed convex set and its exact Euclidean projection, half squared
distance has gradient `u - P_C(u)`. This standard fact motivates the loss. It is
not a proof for learned policy safety, partial-sensor obstacle coverage, network
optimization, fallback actions, or the entire multi-substep closed loop.

## Sensor and teacher validity

Teacher reconstructs relative points from the existing observation's ranges,
hit flags, and declared ray directions; reconstructs velocity from its known
normalization. Translation invariance removes the need for absolute position.
No obstacle centers/radii/full map enter the actor or teacher. Actor input is
unchanged. Native execution uses float64 sensor packets; teacher reconstructs
from float32 observations. These are approximately, not bitwise, equal near
numerical thresholds; an anchor-filter test checks a valid fixture within2e-4
physical acceleration units. All ray constraints remain enabled, not top-k.

Accept only feasible, converged solves with finite actuator-feasible targets,
verified inequality residuals, and nonnegative initial h/psi1. Fallback/emergency
region/unconverged/invalid constraints are excluded and counted by rejection
reason. Invalid finite placeholders contribute zero loss/gradient; average over
ALL queried samples, so declining valid coverage cannot amplify survivors.
Offline teacher timing is recorded but is NOT a numerical validity predicate:
wall-clock jitter must not change labels or break exact numerical continuation.

Known limitation observed in focused checks: a wall5m away with velocity2m/s
toward it and nominal acceleration4m/s^2 toward it makes the inherited native
solver report `stagnated_infeasible_or_ill_conditioned`; the teacher rejects it.
Other queries may be feasible but not converged and are also excluded. This
work does NOT repair that existing solver behavior or claim all unsafe actions
receive supervision. Formal logs must quantify nonzero supervision/rejections.
Existing ordinary HOCBF parameters, margin1m, and sampled-data-robust=False
remain explicit; continuous-time theory is not a full sensor/sampled guarantee.

## Prepared adaptation settings

- One correction-supervised run, initial seed0;131072 ADDITIONAL transitions.
-8 environments,4000-step horizon,24 static obstacles, CUDA SAC.
- Original SAC lr3e-4, gamma.99, batch256, buffer200000, tau.005,
  train_freq1, gradient_steps=-1, target entropy-3; new-data warmup5000.
- Correction weight.1,8 queried examples every8 optimizer updates. These are
  explicit starting settings, not selected by preliminary performance gates.
- Approximately126072 extra actor/Q updates and126072 teacher queries at the
  complete planned budget; no hundreds-of-QPs-per-every-gradient minibatch.
- Checkpoints after4096 and8192 transitions, then every32768 and at requested
  block-boundary pause/end. First learned checkpoint8192, not4096 warmup.
- No wall-clock kill limit; explicit `--resume --additional-steps N` can extend
  a completed adaptation while retaining full state. No automatic promotion.

Runner defaults to preparation only. Launch ONLY after explicit approval:

```bash
.venv/bin/python scripts/run_hocbf_correction_sac.py \
  --output-dir artifacts/hocbf_correction_sac_20260908_v1
```

The authorized launch adds `--start`; resuming adds `--start --resume`.
`PAUSE` in the run root or SIGTERM/SIGINT requests pause at a complete block.
If paused using the marker, remove only that validated marker before resume.
Source/config/model hashes lock the run. The subclass loader restores actor,
critics, target, all optimizers, entropy, correction schedule/counters, nominal
replay, worker states/cursors, pending observation, and Python/NumPy/Torch/CUDA RNG.
Hard crashes may lose work after the last checkpoint, not already committed work.

## Evidence and future comparison

This initial run alone does not isolate supervision from HOCBF/warm adaptation.
The controlled comparator is identical adaptation with `--correction-weight 0`
in a separate output directory. It is supported but not automatically launched.
After training, both raw and filtered inference must be evaluated on matched
tasks. Reduced intervention alone is not success: check raw zero-contact arrival,
arrival, timeout, path ratio, energy and unified contacts. A stationary policy
that reduces interventions while increasing timeouts fails the intended claim.
Retain HOCBF at runtime unless filter-free safety has separate evidence.

Related full-method references already inspected:
- https://arxiv.org/html/2509.12833 (action aliasing, §§6–8; direct distance loss
  is local guidance, not guaranteed long-horizon safety/optimality).
- https://arxiv.org/html/2110.05415 (§III-C; differentiable RCBF/SAC is a different,
  more intrusive integration requiring consistent executed-action Q semantics).

Historical reusable evidence: `JacobianBridgeSAC` and earlyR3 already used a
shield-consistency loss; R3 average valid coverage after trust filtering24.61%
vs86.44% before. This new version avoids that cached-Jacobian approximation,
but fresh QP validity can still limit supervision. No historical source or
artifact was overwritten.

## Verification and launch handoff

14 focused tests passed in12.29s (new tests plus locked collision regressions).
Compile and diff whitespace checks passed. Verified bit-identical lambda0/native
SAC updates, complete real-source weight identity and empty new replay/optimizers,
nonzero correction gradients, exact CPU checkpoint continuation (except measured
wall time), no-launch preparation, and one16-transition subprocess smoke.
Tests are numerical/execution checks, not performance-selection experiments.
Formal adaptation launched detached, Python PID9399, output
`artifacts/hocbf_correction_sac_20260908_v1`. Startup health:

- checkpoint8192 adaptation transitions committed; source524288 unchanged;
- actor/critic3192 updates each, finite losses/entropy/weights;
-399 auxiliary updates,395 nonzero,3192 queries,3176 valid,1496 corrected;
-16 solver-invalid targets rejected; no emergency/constraint-invalid labels;
- checkpoint model/replay/state sizes and runtime source/config hashes verified;
- latest startup block110.97s, collection9.88s, total teacher time2.94s;
- live PID and no ERROR. Stop monitoring after this check; await user.

These numbers prove the intended execution/learning path is live, NOT improved
collision-free navigation. Fixed-task results are not yet available.
