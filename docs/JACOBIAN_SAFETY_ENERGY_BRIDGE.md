# Jacobian Safety-Energy Bridge for Static-Obstacle UAV Delivery

## Target

Design, implement, and evaluate a static-obstacle UAV delivery method in which
the local differential geometry of the deployed HOCBF projection serves two
roles without replacing the hard safety filter:

1. during obstacle-aware navigation training, it teaches the SAC actor to emit
   actions closer to fixed points of the HOCBF projection;
2. during energy-aware fine tuning, it maps gradients of realized
   executed-action energy back to the nominal policy-action coordinates.

The intended empirical claim is:

> Safety-projection geometry learned during obstacle-aware navigation provides
> a reusable bridge to obstacle-conditioned energy learning, and the projection
> Jacobian maps executed-action energy gradients back to policy action space
> while a hard HOCBF shield remains in the loop.

## Status

`COHERENT AFTER REFRAMING / EXTRA ASSUMPTION`

The fixed-active-set QP derivative is coherent. The original diagonal action
coordinate formula is valid only where the environment's normalized-to-physical
action map is locally linear. The deployed environment additionally applies a
horizontal radial saturation. JSEB therefore uses the full coordinate-chain
Jacobian and marks radial or active-set switching boundaries invalid rather than
silently applying a constant diagonal transform.

## Invariant Object

The invariant object is the **actually executed physical acceleration**

\[
u_{\mathrm{exec}} = P_s(u_{\mathrm{nom}}),
\]

because both collision constraints and telemetry energy are functions of the
executed closed-loop motion. Neural predictions, Jacobian descriptors, and
SafetyBridge latents are auxiliary representations of this executed-policy
geometry; none is a safety certificate.

## Existing Contract Audit

The current authoritative implementation has the following contract.

- SAC emits a normalized action \(a\in[-1,1]^3\).
- `UAVEnergyDeliverySACEnv._normalized_action_to_acceleration` converts it to
  physical acceleration. The vertical map is linear, while the horizontal map
  radially clips actions whose XY norm exceeds one.
- `UAVSafetyActionFilter.filter` constructs HOCBF and physical actuator rows in
  physical acceleration coordinates and solves a convex projection QP.
- `UAVEnergyDeliverySACEnv._acceleration_to_normalized_action` maps the projected
  physical acceleration back to the SAC action coordinates.
- The HOCBF is evaluated at every 0.05 s physics substep. A 0.2 s policy
  transition therefore has up to four local projection geometries.
- Telemetry energy uses realized propulsion acceleration after speed saturation
  but before boundary/collision projection.

The formal static-obstacle protocol remains:

| Quantity | Value |
|---|---:|
| map | 4000 x 4000 x 400 m |
| static cylinders | 24 |
| obstacle radius | 50--120 m |
| LiDAR | 128 x 8 = 1024 rays |
| LiDAR range | 100 m |
| policy interval | 0.2 s |
| physics/HOCBF interval | 0.05 s |
| velocity limits | [20, 20, 5] m/s |
| acceleration limits | [5, 5, 3] m/s^2 |

## Assumptions

### Projection assumptions

1. The local projection problem is a strictly convex QP with positive-definite
   Hessian \(H\).
2. At a valid anchor, the QP is feasible and converged and no fallback or
   emergency brake replaces its solution.
3. The active set is locally unchanged, its effective rows have stable SVD
   rank, active multipliers are separated from zero, and inactive slacks are
   separated from zero.
4. The local action-coordinate maps are differentiable at the anchor.
5. The Jacobian is used only inside a declared normalized-action trust region.

### Learning assumptions

1. Phase-1 safety labels come from the deployed HOCBF operating on the same
   executed trajectory that generated the replay sample.
2. Monte-Carlo energy labels are suffix sums of realized TelemetryCostModel
   energy from complete successful goal trajectories.
3. Action-conditioned energy training uses executed actions, never nominal
   actions relabeled as executed actions.
4. Conformal train, calibration, and final-test trajectories are disjoint.
5. Final coverage claims are made only after Phase 2C recollection under the
   final frozen policy.

### Static-world boundary

Obstacles are static cylinders. This design contains no obstacle-velocity
learning and supports no dynamic-obstacle claim.

## Notation

- \(o\): SAC observation, including velocity, relative active-goal features,
  1024 LiDAR ranges, and 1024 validity indicators.
- \(a\): nominal normalized SAC action.
- \(u\): physical acceleration.
- \(f(a)\): normalized-to-physical action map.
- \(g(u)\): physical-to-normalized action map.
- \(P_s\): physical HOCBF QP projection at state/sensing context \(s\).
- \(a_{\mathrm{exec}}=g(P_s(f(a)))\): executed normalized action.
- \(A u\ge b\): complete local QP constraints.
- \(A_{\mathcal A}\): effective active rows with positive KKT multipliers.
- \(H\): QP Hessian.
- \(J_{\mathrm{phys}}\): derivative of physical projection output with respect
  to its physical QP center.
- \(J_{\mathrm{norm}}\): derivative from nominal normalized SAC action to
  executed normalized action.
- \(z_J\): compact local projection-geometry descriptor.
- \(E_{\mathrm{go}}(t)\): suffix realized energy to a successfully reached goal.
- \(B_J(t)\): suffix diagnostic safety burden; not physical energy.

## Derivation Strategy

1. derive the fixed-active-set QP projection and its local Jacobian;
2. compose that derivative with the real normalized/physical action maps;
3. use an affine local projection model whose derivative is exactly the stored
   normalized projection Jacobian;
4. derive the Phase-1 shield-consistency gradient;
5. train energy predictors only on realized executed trajectories;
6. pass action-conditioned energy gradients through the same local projection
   model during Phase 2B;
7. recollect and recalibrate under the final frozen policy in Phase 2C.

## Derivation Map

1. QP KKT stationarity and active equalities imply the physical projection
   derivative under a fixed active set.
2. SVD pseudo-inversion handles redundant active rows without changing the
   projected tangent subspace.
3. The normalized Jacobian is a chain-rule object. The diagonal formula is a
   special case of locally linear action maps.
4. A stop-gradient affine offset makes the local surrogate match the anchor
   exactly while preserving the desired derivative.
5. Shield consistency removes actor motion in locally blocked directions.
6. The same affine surrogate maps an executed-action MC critic gradient to
   nominal-action coordinates.
7. Conformal validity belongs to the final-policy recollection distribution,
   not to the Phase-2A policy after Phase-2B changes it.

## Main Derivation

### 1. Fixed-active-set projection identity

At a fixed state and sensor packet, consider

\[
\min_u \frac12(u-u_0)^\top H(u-u_0)
\quad\text{s.t.}\quad Au\ge b,
\]

with \(H\succ0\). For a locally fixed effective active set \(\mathcal A\), the
active inequalities are equalities. The KKT conditions are

\[
H(u-u_0)-A_{\mathcal A}^\top\lambda=0,
\qquad
A_{\mathcal A}u=b_{\mathcal A}.
\]

Substitution gives the exact local affine solution

\[
u = u_0 + H^{-1}A_{\mathcal A}^\top
\left(A_{\mathcal A}H^{-1}A_{\mathcal A}^\top\right)^\dagger
(b_{\mathcal A}-A_{\mathcal A}u_0).
\]

Differentiating with fixed \(A_{\mathcal A}\), \(b_{\mathcal A}\), and \(H\)
gives

\[
J_{\mathrm{phys}}
=
I-H^{-1}A_{\mathcal A}^\top
\left(A_{\mathcal A}H^{-1}A_{\mathcal A}^\top\right)^\dagger
A_{\mathcal A}.
\]

For the current minimum-intervention HOCBF, \(H=I\), hence

\[
J_{\mathrm{phys}}
=I-A_{\mathcal A}^\top
(A_{\mathcal A}A_{\mathcal A}^\top)^\dagger A_{\mathcal A}.
\]

This is an exact derivative only inside a fixed active-set region. It is not a
derivative theorem at active-set switching boundaries.

### 2. Barrier-only and total geometry

Let \(A_{\mathcal A}^{B}\) contain active HOCBF rows and
\(A_{\mathcal A}^{P}\) active physical actuator rows. JSEB records

\[
J_{B}=J(A_{\mathcal A}^{B}),
\qquad
J_{\mathrm{{total}}}=J([A_{\mathcal A}^{B};A_{\mathcal A}^{P}]).
\]

The actor losses use the total geometry because the executed action obeys both
barrier and physical constraints. The barrier-only Jacobian is a diagnostic that
separates safety restrictions from actuator-bound restrictions.

### 3. Stable SVD pseudo-inverse

For

\[
M=A_{\mathcal A}H^{-1}A_{\mathcal A}^\top=U\Sigma V^\top,
\]

use

\[
M^\dagger=V\Sigma^\dagger U^\top,
\qquad
\Sigma^\dagger_{ii}=
\begin{cases}
1/\Sigma_{ii},&\Sigma_{ii}>\tau_{\mathrm{svd}},\\
0,&\text{otherwise}.
\end{cases}
\]

The tolerance is scale-aware and recorded. Redundant active rows therefore do
not spuriously remove additional action dimensions.

### 4. Normalized-action coordinate chain

The deployed normalized projection is

\[
\Pi_s(a)=g(P_s(f(a))).
\]

At a differentiable anchor,

\[
J_{\mathrm{norm}}
=J_g(u_{\mathrm{exec}})J_{\mathrm{phys}}J_f(a).
\]

In the unsaturated linear region,

\[
f(a)=Da,
\quad
g(u)=D^{-1}u,
\quad
D=\operatorname{diag}(a_{xy}^{\max},a_{xy}^{\max},a_z^{\max}),
\]

so the formula reduces to

\[
J_{\mathrm{norm}}=D^{-1}J_{\mathrm{phys}}D.
\]

For horizontal radial saturation, \(a_{xy}\mapsto a_{xy}/\|a_{xy}\|\), the
local derivative away from \(\|a_{xy}\|=1\) is

\[
\frac{1}{r}\left(I-\frac{aa^\top}{r^2}\right),\qquad r=\|a_{xy}\|>1.
\]

At the radial switching surface and normalized box boundaries the geometry is
marked unstable and excluded from Jacobian losses.

### 5. Local geometry diagnostics

For a valid normalized total Jacobian \(J\), define

\[
\mathrm{authority}=\frac{\operatorname{tr}(J)}{3},
\qquad
\mathrm{blocked\_fraction}=1-\mathrm{authority}.
\]

Also record

\[
\mathrm{normal\_action\_fraction}
=\frac{\|(I-J)a_{\mathrm{nom}}\|_2}
{\|a_{\mathrm{nom}}\|_2+\epsilon}.
\]

These are local diagnostics, not safety certificates or global degrees of
freedom. Numerical rank comes from the singular values of \(J\).

### 6. Anchor-exact local projection surrogate

For replay anchor
\((o,a_0,a_{\mathrm{exec},0},J_0)\), define

\[
\widehat a_{\mathrm{exec}}(a)
=\operatorname{sg}(a_{\mathrm{exec},0}-J_0a_0)+J_0a,
\]

where `sg` is stop-gradient. This surrogate satisfies

\[
\widehat a_{\mathrm{exec}}(a_0)=a_{\mathrm{exec},0},
\qquad
\frac{\partial\widehat a_{\mathrm{exec}}}{\partial a}=J_0.
\]

It is used only when the current actor action remains inside
\(\|a-a_0\|_2\le\delta_J\).

### 7. Phase-1 shield consistency

Let \(a_\theta(o)\) be the current actor action. Define

\[
L_{\mathrm{shield}}
=\mathbb E\left[
m_J\left\|a_\theta(o)-
\widehat a_{\mathrm{exec}}(a_\theta(o))\right\|_2^2
\right],
\]

where \(m_J\) masks invalid, unstable, fallback, emergency, and out-of-trust-
region samples. The Phase-1 actor objective is

\[
L_{\mathrm{actor}}^{(1)}
=L_{\mathrm{SAC}}+\lambda_{\mathrm{shield}}L_{\mathrm{shield}}.
\]

If no constraint is active, \(J=I\) and the anchor offset is zero, so this loss
vanishes. When directions are blocked, the loss teaches the actor to approach a
local fixed point of the shield rather than relying on repeated intervention.

### 8. Safety-burden labels

For every complete successful goal trajectory, realized telemetry energy is

\[
E_{\mathrm{go}}(t)=\sum_{k=t}^{T-1}e_k.
\]

Define the diagnostic burden

\[
B_J(t)=\sum_{k=t}^{T-1}
\left[c_1\|a_{\mathrm{exec},k}-a_{\mathrm{nom},k}\|_2
+c_2(1-\operatorname{tr}(J_k)/3)
+c_3\mathbf1\{\mathrm{emergency}_k\}\right].
\]

\(B_J\) is an auxiliary supervision target. It is not energy and is never
added to TelemetryCostModel labels.

### 9. Compact safety context

The predefined raw descriptor is

\[
z_J=[\mathrm{authority},\sigma_1,\sigma_2,\sigma_3,
\mathrm{rank}/3,n_B/K,\|\Delta a\|/2,
\mathbf1_{\mathrm{nominal\ safe}},
\widetilde s_{\mathrm{nom}},\widetilde s_{\mathrm{exec}},
\mathbf1_{\mathrm{emergency}}].
\]

Slack normalization constants and \(K=16\) are selected on pilot/training data
only and frozen before formal evaluation. A small optional SafetyBridge encoder
may predict current \(z_J\) and suffix \(B_J\). It is retained only if the E2
ablation improves held-out energy prediction over raw \(z_J\).

### 10. Energy point models

All point models target complete-trajectory MC energy-to-go.

- E0: existing compact motion/goal/boundary context;
- E1: E0 plus current raw \(z_J\);
- E2: E0 plus a learned SafetyBridge latent.

Raw 1024-ray LiDAR is not inserted into the energy model by default. Selection
uses held-out validation trajectories; final metrics use disjoint test
trajectories.

### 11. Action-conditioned executed-energy critic

Train

\[
Q_E(x_E,z_J,a_{\mathrm{exec}})\approx E_{\mathrm{go}}
\]

using supervised MC labels and executed normalized actions. It is not the sole
source of the conformal switching bound. Its role is a differentiable local
energy objective.

### 12. Phase-2 Jacobian energy bridge

With the anchor-exact safe surrogate,

\[
L_{\mathrm{energy}}
=\mathbb E\left[
m_J Q_E(x_E,z_J,
\widehat a_{\mathrm{exec}}(a_\theta(o)))
\right].
\]

Automatic differentiation gives

\[
\frac{\partial L_{\mathrm{energy}}}{\partial a_\theta}
=J_0^\top
\frac{\partial Q_E}{\partial a_{\mathrm{exec}}},
\]

which is the desired bridge from executed-action energy geometry to the actor's
nominal action space. The critic parameters are frozen during actor-gradient
steps; only the actor receives this gradient.

The Phase-2B actor objective is

\[
L_{\mathrm{actor}}^{(2)}=
L_{\mathrm{SAC-task}}
+\lambda_{\mathrm{shield}}L_{\mathrm{shield}}
+\lambda_{\mathrm{energy}}(t)
\frac{L_{\mathrm{energy}}}{C_E},
\]

where \(C_E\) is a training-only energy scale and
\(\lambda_{\mathrm{energy}}(t)\) uses a frozen warmup/ramp schedule. Actor
gradient clipping is applied after combining objectives.

## Phase Algorithms

### Phase 1: 500,000 transitions

- static obstacles, LiDAR, and HOCBF remain enabled;
- SAC task/navigation reward contains no energy penalty;
- every substep records projection geometry;
- the policy-transition anchor is the first-substep geometry because it matches
  the replay state and nominal actor action;
- later substep geometries remain trajectory diagnostics;
- `JacobianSAC` adds the masked shield-consistency objective;
- complete trajectories are saved to the Safety Bridge dataset with MC energy
  and burden labels, but Phase-1 policy optimization never uses energy loss.

### Phase 2A: 100,000 transitions

- freeze the Phase-1 policy;
- keep static obstacles, LiDAR, and HOCBF enabled;
- recollect complete executed-policy trajectories;
- fit E0/E1/E2 point models and the action-conditioned MC energy critic;
- choose the compact energy context using validation trajectories only.

### Phase 2B: 300,000 transitions

- resume actor/task critic training;
- retain HOCBF as final action authority;
- retain shield consistency;
- enable the normalized, trust-region-masked Jacobian energy loss after warmup;
- keep the action-conditioned energy critic frozen during actor updates;
- collect policy-shift diagnostics, but make no final conformal claim from the
  Phase-2A calibration distribution.

### Phase 2C: 100,000 transitions

- freeze the final policy;
- collect fresh complete trajectories under the final executed policy;
- create trajectory-disjoint train/calibration/test IDs;
- refit selected energy point/risk models;
- perform predefined Goal-type x distance Mondrian calibration;
- evaluate final coverage and run persistent-delivery switching evaluation as
  separately counted evaluation transitions.

The exact training budget is

\[
500{,}000+100{,}000+300{,}000+100{,}000=1{,}000{,}000.
\]

Evaluation, GIF, calibration-test, and persistent-delivery evaluation
transitions are recorded separately and never included in this sum.

## Ablations

| ID | Navigation bridge | Energy context | Jacobian energy gradient |
|---|---|---|---|
| A Current baseline | no | existing compact baseline | no |
| B J-Safety | shield consistency | existing compact baseline | no |
| C J-Context | shield consistency | raw/selected Jacobian context | no |
| D Full JSEB | shield consistency | selected Jacobian context | yes |

Only short pilots are run for all four. A formal 1M run uses parameters frozen
after the pilot and does not redefine the protocol in response to formal test
results.

## Metrics

### Safety

- obstacle and boundary collisions;
- nominal-safe action rate;
- HOCBF intervention rate and mean/P90 magnitude;
- emergency/fallback rate;
- active-set switching/invalid-Jacobian frequency.

### Navigation

- overall and distance-bucket success;
- path ratio and steps per task.

### Energy

- realized energy per task, meter, and simulation minute;
- E0/E1/E2 MAE/RMSE and group metrics;
- conformal width and coverage;
- action-conditioned critic gradient magnitude and finite-difference checks.

### Persistent delivery

- tasks per 1000 transitions, battery cycle, and simulation hour;
- charger return and recharge success;
- exhaustion, arrival SOC, and unnecessary-return proxy.

### Jacobian bridge

- authority and rank distributions;
- intervention/energy as functions of authority;
- future energy residual versus \(B_J\);
- masked/unmasked sample counts and trust-region rejection rate.

## Statistical Protocol

- The radius2x 500-task evaluation is the fixed paired Phase-1 baseline.
- Pilot model/loss selection uses pilot validation tasks only.
- Formal navigation uses the existing fixed 500-task set and gate:
  overall success at least 98%, every distance bucket at least 95%, mean path
  ratio at most 1.10, and zero obstacle collision.
- Energy split units are complete trajectories, never transitions.
- Goal type and distance bucket are predefined conformal groups.
- Jacobian groups are diagnostic unless predefined before final collection and
  sufficiently populated.
- Pilot results are never reported as formal results.

## Failure Modes and Masks

The Jacobian is `valid=False` and excluded from actor auxiliary losses when any
of the following occurs:

- QP infeasible or not converged;
- fallback or emergency braking;
- nonfinite rows, multipliers, or Jacobian;
- active multiplier near zero or inactive slack near zero;
- unstable numerical rank;
- normalized/physical radial or box switching boundary;
- current actor action outside the frozen trust region;
- the first-substep anchor does not align with the replay state/action.

Invalid geometry is logged, not replaced by a fabricated zero Jacobian.

## Gates

### Unit/deterministic gate

- identity, one-row, two-row, redundant-row, coordinate-transform, finite-
  difference, and active-switch tests pass;
- J-disabled environment behavior is regression-identical;
- old `UAVEnergyDelivery.py` remains unchanged.

### Pilot gate

- no NaN/Inf;
- correct exact transition accounting;
- nonzero finite shield and energy gradients;
- invalid/fallback samples are masked;
- no material collision-safety regression;
- acceptable runtime overhead;
- complete artifacts and logs.

### Formal downstream gate

The 500k Phase-1 navigation gate remains authoritative. Failure stops Phase2;
it is not hidden by zero collision alone.

## Completed 50k Pilot and Frozen Formal Parameters

The completed diagnostic suite is stored at
`artifacts/jseb_pilot_suite_20260824_180719`. Each of A/B/C/D used exactly
25k/5k/15k/5k training environment transitions; evaluation transitions were
excluded. All numerical gates passed and every fixed navigation evaluation had
zero obstacle-collision steps. These are pilot diagnostics, not paper results.

The main observations were:

- A: success 0.60, path ratio 1.158, intervention rate 0.282, nominal-safe rate
  0.727.
- B/C Phase-1 J-Safety: success 0.50, path ratio 2.291, intervention rate
  0.223, nominal-safe rate 0.929, valid-J rate 0.967. Thus J-Safety reduced
  intervention and increased nominal-safe actions, but the small pilot also
  showed worse success/path efficiency and a higher emergency rate. H1 is not
  established.
- C final goal-energy diagnostics: E0/E1/E2 MAE = 6.648/6.207/6.743. Raw
  Jacobian context E1 gave a small pilot improvement; the learned E2 latent did
  not. No representation claim is made.
- D exercised 13,000 nonzero energy-bridge updates with action-conditioned
  critic gradient norm 0.228 and finite losses. Its E0/E1/E2 MAE =
  8.509/7.875/8.406, worse than C, so the pilot provides no evidence that the
  actor energy bridge improves final energy prediction or persistent delivery.
- Some 5k pilot collection stages did not contain enough complete disjoint
  TASK-to-CHARGER missions. Those variants explicitly omit mission conformal
  and persistent-switching claims; real goal-segment MC labels are used only
  for numerical/gradient diagnostics. Formal runs retain the hard complete-
  mission requirement.

Because the 10-task pilot is too small for reliable performance tuning, the
formal parameters are frozen from the numerically stable configuration rather
than optimized against noisy success numbers:

- `lambda_shield = 0.1`;
- `lambda_energy = 0.02`;
- normalized-action trust radius `delta = 0.35`;
- energy warmup/ramp = 25k/75k Phase-2B transitions;
- bridge replay capacity 50k, learning start 10k, batch size 256;
- supervised energy fitting: 30 epochs, hidden width 128, learning rate 3e-4;
- conformal target coverage 0.95.

The formal 500-task Phase-1 gate remains decisive. In particular, the pilot's
mixed B/C/D navigation behavior is a material risk: if the 500k checkpoint
does not satisfy the existing navigation gate, the runner stops before energy
training rather than presenting a zero-collision engineering result as an AI
learning success.

## Boundaries and Non-Claims

- HOCBF remains the final hard safety layer; Jacobian learning does not certify
  the actor.
- No differentiability is claimed at active-set or action-map switching points.
- No arbitrary-environment recursive feasibility is claimed.
- No dynamic-obstacle result is claimed.
- No arbitrary conditional or shift-free conformal coverage is claimed.
- No real-UAV physical safety is claimed.
- Phase-2A calibration is not transferred to the changed Phase-2B policy.
- Energy improvement is not claimed unless matched ablations establish it.

## Open Risks

1. The first-substep local Jacobian may explain only part of a four-substep
   policy transition; later-substep summaries must be audited.
2. The current cyclic projection solver's multiplier estimates require explicit
   export and KKT residual validation before use as active-set evidence.
3. A high-dimensional SAC observation with a separate compact safety replay may
   make auxiliary batches distributionally different from SAC replay batches.
4. Shield consistency can reduce intervention while harming exploration or
   path efficiency; pilot lambda selection must use all navigation metrics.
5. The action-conditioned energy critic can exploit spurious action-state
   correlations. Held-out complete trajectories and gradient finite differences
   are mandatory.
6. Phase-2B policy shift can make Phase-2A energy gradients unreliable far from
   anchors; trust-region masking and Phase-2C recollection are mandatory.
