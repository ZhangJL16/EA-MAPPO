# R6 SAC–PPO Navigation Failure Analysis

Date: 2026-09-01  
Scope: read-only diagnosis of the frozen R3 SAC evidence and the completed
262,144-transition R6 PPO training trace. The still-running R6 phase-end
evaluation is not used to infer training success and was not interrupted.

## Executive conclusion

R6 did not fail because PPO lacks a basic clipping or GAE term, nor because it
received only 262k rather than 500k transitions. Its implementation is
numerically finite and the nominal-action likelihood is internally consistent
with a deterministic safety projection viewed as part of the environment.

The learned policy nevertheless converged to the wrong operating point: on the
same 50 initial evaluation states, its deterministic acceleration norm is only
`0.115`, versus `1.422` for R3 SAC at 250k transitions. R6 points approximately
toward the goal (`94%` positive goal-action dot product) but acts roughly twelve
times too weakly, while its stochastic action standard deviation remains
`0.337`, about three times its deterministic mean norm. Training therefore
contains large, mostly zero-mean exploratory accelerations, whereas deterministic
evaluation exposes a timid policy that cannot complete the task.

That symptom is explained by four mutually reinforcing design defects:

1. SAC performed about **85.5 times as many 256-sample optimizer updates** as PPO
   at the aligned 256k-transition point, and KL early stopping removed another
   28% of the already small PPO schedule.
2. PPO combines actor, three value heads, the shared encoder, and distillation in
   one loss, then clips their joint gradient to `0.5`. The observed pre-clip norm
   rose to `15–24`. R3 separates actor and critic feature extractors and permits
   actor gradients up to `10`.
3. The intervention constraint is set to the mathematically degenerate budget
   `d_I=0`. Because intervention cost is nonnegative, its dual multiplier can
   only increase and can never recover; it reached `0.42784`. Low acceleration is
   the easiest way to satisfy this pressure before navigation has been learned.
4. PPO distils the current policy mean toward the safety-filter output generated
   from a *different stochastic sampled action*. Early symmetric exploration
   therefore supplies a near-zero conditional-average teacher and reinforces the
   timid mean. It is not a same-sample shield derivative or a valid projection of
   the current mean.

The structured spatial encoder, automatic entropy tuning, and long-horizon
off-policy Bellman propagation in SAC are additional material advantages. A
simple increase to 500k transitions would preserve the same local optimum and
is not justified.

## 1. Aligned evidence

| Environment transitions | R3 SAC completed/success/failed | R6 PPO completed/success/failed | SAC optimizer updates | PPO minibatch updates |
|---:|---:|---:|---:|---:|
| 32,000 / 32,768 | 23 / 23 / 0 | 8 / 0 / 8 | 26,992 | 473 |
| 64,000 / 65,536 | 44 / 43 / 1 | 16 / 0 / 16 | 58,992 | 941 |
| 128,000 / 131,072 | 94 / 89 / 5 | 32 / 0 / 32 | 122,992 | 1,777 |
| 256,000 / 262,144 | 269 / 263 / 6 | 64 / 0 / 64 | 250,992 | 2,936 |

All 64 completed PPO episodes terminated at the 4,000-step limit. SAC had
already produced 23 successful episodes by 32k transitions, so the missing
237,856 transitions between the R6 development budget and the 500k R3 budget
cannot explain the onset failure.

At 262,144 transitions R6 had 128 rollout updates. A full schedule would have
used `128 × 4 × 8 = 4096` minibatches, because each rollout contains 32 recurrent
sequences and each minibatch contains four sequences (256 time samples). In fact,
66 of 128 rollout updates hit KL early stopping, leaving only 2,936 minibatches:

\[
N_{\mathrm{present},\mathrm{PPO}}=2936\times256=751,616.
\]

At approximately 256k transitions SAC had 250,992 replay updates with batch size
256:

\[
N_{\mathrm{present},\mathrm{SAC}}=250,992\times256=64,253,952,
\qquad
\frac{N_{\mathrm{SAC}}}{N_{\mathrm{PPO}}}=85.49.
\]

This comparison does not imply that stale on-policy data should be replayed 85
times in PPO. It shows that the current PPO system has radically less critic and
actor fitting capacity per expensive physical transition and needs an
on-policy-compatible remedy rather than more simulator time alone.

## 2. What is not missing or broken

The audit found no evidence for the following candidate explanations:

- **Action transform:** the Beta variable on `[0,1]` is mapped to `[-1,1]`, and
  the `-log(2)` Jacobian term is included once per action dimension.
- **PPO ratio and clipping:** old nominal-action log probabilities, likelihood
  ratios, and the clipped surrogate are present.
- **GAE indexing:** terminal masks, final bootstrap values, and backward GAE
  recursion are aligned.
- **Recurrent reset:** stored pre-observation hidden states and episode-start
  resets are consistent with the rollout loop; truncated BPTT is 64 steps.
- **LiDAR direction order:** the R6 direction table agrees with the simulator to
  numerical tolerance (`max absolute error < 3e-8`).
- **Numerical stability:** losses remained finite; there were no NaNs, action
  bound violations, or CUDA failures.
- **R3 bridge as the sole explanation:** the R1 SAC variant, which disables the
  Jacobian shield auxiliary, also had 21/21 successes at 32k. SAC can learn basic
  navigation without the R3 bridge term.
- **GPU use:** the PPO actor and optimizer were on CUDA. Simulation speed affects
  wall time, not the observed zero-success learning trajectory.

Nominal versus executed action is subtle but not by itself an unbiasedness bug.
If the safety map is deterministic,

\[
u_t\sim\pi_\theta(\cdot\mid s_t),\quad
\tilde u_t=S(s_t,u_t),\quad
s_{t+1}\sim P(\cdot\mid s_t,\tilde u_t),
\]

then the composition defines a valid transition kernel over the nominal action.
PPO may use `log π(u_t|s_t)`. The practical problem is that a many-to-one safety
map makes many nominal actions observationally equivalent, flattening the return
surface and increasing policy-gradient variance. The current auxiliary loss
does not correctly undo that aliasing.

## 3. Ranked causal diagnosis

### 3.1 Critical: policy learning is suppressed by optimizer exposure and joint clipping

R6 uses one shared encoder and one optimizer for the policy and all three value
functions. Its typical objective is numerically dominated by value fitting:
`0.5 × value_loss` is commonly around `2`, while mean policy loss is near zero
and `0.2 × distillation_loss` is around `0.03`. Loss magnitude alone does not
prove gradient dominance, but the smoke test already showed value residuals
monopolising the encoder, and the development run's pre-clip norm grew from
`0.83` to `15.34` (maximum rollout-average `23.96`). Global clipping to `0.5`
therefore scales every policy update by roughly `0.02–0.06` late in training.

R3 uses separate actor and critic extractors (`share_features_extractor=False`),
separate optimizers, 495k actor updates over 500k transitions, and an actor clip
of `10` only for the auxiliary bridge path. PPO is missing gradient isolation,
value-scale control (for example PopArt/return normalization and value clipping),
and sufficient non-stale optimization—not merely another training epoch flag.

### 3.2 Critical: zero intervention budget creates an irreversible dual

R6 updates

\[
\lambda_{k+1}=\Pi_{[0,20]}\left[\lambda_k+
0.05\left(\bar c_{I,k}-d_I\right)\right],\qquad d_I=0,
\]

with `c_I = ||u_exec-u_nom|| ≥ 0`. Consequently
`λ_{k+1} ≥ λ_k` for every rollout. Even if the policy later becomes perfectly
intervention-free, the multiplier only stops increasing; it never decreases.
This is not an adaptive constraint for a learning phase. It permanently records
all early exploration mistakes.

The final multiplier was `0.42784`, while R3's successful training policy still
used nonzero intervention. Thus R6 was simultaneously asked to discover
navigation and immediately meet a stricter-than-baseline zero-use shield target.
It learned the feasible but useless local response: small goal-aligned actions.

The missing mechanism is a feasible positive budget in consistent units, a
navigation-first warm-up/ramp, and either a recoverable dual or an explicit slack
penalty. Collision and intervention constraints should not be activated as if a
random initial policy were already deployment-ready.

### 3.3 Critical symptom: stochastic training and deterministic deployment diverge

On the fixed 50-state audit:

| Policy | Deterministic action norm | Mean stochastic action std | Positive goal dot product |
|---|---:|---:|---:|
| R6 PPO, 32,768 | 0.038 | 0.471 | 92% |
| R6 PPO, 262,144 | 0.115 | 0.337 | 94% |
| R3 SAC, 250,000 | 1.422 | not required for conclusion | 100% |
| R3 SAC, 500,000 | 1.484 | not required for conclusion | 96% |

PPO did learn coarse direction, but its mean action remains tiny and its sampled
action noise remains much larger. Independent acceleration noise tends to cancel
over time, while the safety layer removes unsafe components. This produces local
progress rewards without coherent traversal—the exact pattern in the logs:
mean reward and progress improve, yet success remains zero.

PPO's fixed entropy coefficient is `0.002`. R3 SAC uses automatic entropy
tuning; its learned coefficient is about `0.0478` at 250k and `0.0426` at 500k.
The raw coefficients are not numerically interchangeable across Beta and
tanh-Gaussian policies, but R6 lacks the adaptive feedback mechanism. Its entropy
fell from `1.81` to `1.03` over the first/last ten updates before any successful
episode existed. The repair should control exploration relative to mean action
and task progress, not blindly increase or decrease a fixed entropy bonus.

### 3.4 High: the safety distillation target is sample-mismatched

For an intervention transition R6 minimises

\[
\|\mu_\theta(s_t)-S(s_t,u_t^{\mathrm{old}})\|^2,
\qquad u_t^{\mathrm{old}}\sim\pi_{\theta_{\mathrm{old}}}.
\]

The current mean `μθ(s)` did not generate the teacher target. Under broad,
approximately symmetric exploration, the regression optimum approaches the
conditional average of heterogeneous projected samples, which can be close to
zero even when each individual safe manoeuvre has high magnitude. The loss fell
from about `0.257` to `0.168`, but that is compatible with mean collapse rather
than learning the correct side of an obstacle.

A valid replacement must couple the same policy noise/sample through the shield,
differentiate a local projection model at that sample, or train a separate
executed-action policy with a correctly defined likelihood/interface objective.

### 3.5 High: the encoder discarded the strongest successful spatial prior

R3 reshapes LiDAR into `[2,8,128]`, applies circular-azimuth and bounded-elevation
convolutions, and retains both mean and max pooled spatial features. Its actor and
critics receive separate copies of this extractor. R6 encodes all 1,024 rays and
then uses one goal-conditioned query to pool them into a single vector before the
GRU.

Direction tokens make R6 mathematically capable of representing geometry, but a
single early pooling bottleneck must learn adjacency, free-space width, and
multi-modal left/right alternatives from reward alone. The successful SAC result
does not isolate algorithm from encoder: R1–R4 all use the structured convolution.
Therefore “SAC beats PPO” is not yet a controlled causal claim. R6 is missing a
local/circular spatial inductive bias at least as strong as the R3 encoder.

### 3.6 Medium: PPO's effective credit horizon is too short for first success

R6 uses `γ=0.99`, `λ=0.95`, so a reward `k` steps away enters GAE with weight

\[
(\gamma\lambda)^k=0.9405^k.
\]

The weights are `0.0197` at 64 steps, `3.89e-4` at 128, and `1.51e-7` at 256.
The 100-point completion reward is therefore effectively invisible to states
more than a few dozen decisions away, while successful SAC trajectories average
roughly 800–1,100 steps. Dense progress reward gives a local direction signal,
which explains the positive goal alignment, but it does not distinguish a
globally coherent detour from an initially promising dead end.

SAC repeatedly bootstraps a replayed Q function and can propagate local value
changes across many updates. PPO needs a higher horizon, better value scaling,
and possibly a curriculum that produces early complete trajectories. Increasing
`λ` alone is unsafe until critic scaling and gradient isolation are fixed.

## 4. What PPO is actually missing

Before another formal run, the PPO branch needs all of the following as one
coherent training contract:

1. **Separate actor and critic encoders/optimizers**, per-component gradient
   diagnostics, value normalization or PopArt, and actor-only clipping.
2. **A feasible constraint schedule:** navigation warm-up, positive intervention
   budget/slack, correctly scaled cost objective, and a dual that can decrease.
3. **A same-sample shield learning objective**, rather than regression of the
   current mean onto projected actions from unrelated exploration samples.
4. **A structured circular LiDAR encoder** or local sparse attention that
   preserves neighbourhood and alternative-corridor information before pooling.
5. **Adaptive exploration control** and a training/deployment consistency gate
   based on deterministic mean magnitude versus stochastic standard deviation.
6. **Long-horizon credit support:** tuned `γ/λ`, longer return context, and
   critic stabilisation; optionally a short-distance curriculum without any
   planner- or heuristic-action input.
7. **An on-policy-compatible update budget.** More epochs are allowed only while
   aggregate KL and clip fraction remain controlled; critic replay or an
   auxiliary value phase can increase data reuse without pretending old samples
   are fresh PPO policy data.

This remains end-to-end learned navigation. None of these additions requires A*,
waypoints, a PD controller, or a hand-coded deadlock action.

## 5. Minimal falsification plan before a new long experiment

Do not immediately rerun 500k. Use staged kill gates:

1. **One-rollout gradient audit (no claim):** log separate actor, reward-value,
   cost-value, entropy, and distillation gradient norms/cosines before clipping.
   Reject if actor gradients are attenuated by more than 10× by other heads.
2. **32k mechanics test:** structured encoder, actor/critic separation, value
   normalisation, and no intervention dual during warm-up. The benchmark is R3's
   onset evidence: at least one deterministic-policy success must occur by 32k.
   Otherwise stop; more budget is not justified.
3. **64k constraint test:** activate the feasible intervention schedule and
   same-sample shield objective. Require increasing deterministic mean action,
   nonzero success, and decreasing intervention without entropy collapse.
4. **Only after those gates:** run matched 500k SAC/PPO budgets on multiple seeds
   and the immutable held-out task set. Report success, path ratio, collision,
   intervention, deterministic/stochastic action gap, and wall-clock separately.

The current evidence supports repairing PPO. It does not support claiming that
PPO is intrinsically incapable of this navigation problem.
