# Persistent Goal Stream with Certified Recoverability Backup

The independent 2x-energy recovery-teacher protocol is documented in
`docs/CERTIFIED_RECOVERY_TEACHER_2X.md`. It reuses the frozen foundation here and
only revalidates energy-related numerical bounds with the simulator's exact
`flight_energy_multiplier`; it does not alter the theory or expose certificate
quantities to the clean SB3 actor observation.

## Status and scope

The main persistent method has one trainable policy: `PersistentGeneratorSAC`. It emits only a
three-dimensional continuous latent. `EnergyManagementSAC` and its categorical SMDP remain
ablation-only compatibility code and are never instantiated by `make_persistent_uav_env`.
Persistent certificate validation, acceptance rollouts, training, and evaluation were deliberately
not run in this implementation round. All bounds and charging values remain synthetic.

## Authority and task semantics

```text
Environment             assigns the next certified goal
PersistentGeneratorSAC  controls task flight and voluntary station approach
Certificate runtime     restricts every accepted action to A_rec
Frozen kappa            acts only after backup takeover
Charger support         constrains the complete Generator to remain docked while departure is closed
Charger hold            exceptional fallback if constrained support cannot be certified
```

The environment samples goals from the finite `CertifiedGoalNetwork`; the charging station is not a
normal goal. Reaching a goal assigns the next goal without terminating the episode. A voluntary
station visit preserves the task ID and pending goal. Leaving the charger resumes that same goal.

The normal action is

\[
u_t\sim\pi_\theta(\cdot\mid o_t),\quad \eta_t=\tanh u_t,\quad
a_t=c(z_t)+G(z_t)\eta_t,
\]

where `u_t` has dimension three. The persistent default center is `safety_neutral`; it does not
encode the task-goal direction or station-return decision. The complete Generator must remain
full-rank and must pass the recoverability verifier.

## Recoverability certificates

The exact joint first-passage value and executable upper certificate are distinct; the canonical definitions and proof are in `docs/theory/PAPER_THEOREM.md`. Define recoverability using lower battery energy:

\[
\mathcal R=\{z:\text{a valid certified kappa chain exists and }
e^--E^\kappa(z)-e_G-m_e\ge 0\}.
\]

Membership means kappa is available as a certified backup; it does not mean kappa currently has
control. Runtime certifies complete action supports. The recoverability predecessor collection is

\[
\mathfrak A_{\rm rec}(z)=\{S\subseteq\mathcal A_{\rm act}(z):
\widehat{\operatorname{Post}}_{\Xi_S}(z,S)\subseteq\mathcal R\}.
\]

`Post_hat` is the uncertainty-aware interval/zonotope successor envelope; true `Post` is the physical relation. The verifier jointly checks
actuator limits, velocity bounds, swept FREE geometry, tracking/dynamics bounds, and the successor
energy inequality

\[
e^+_{\rm lower}\ge E^\kappa(z^+)_{\rm upper}+e_G+m_e.
\]

The state-level Generator is accepted only after direct complete-set verification of recoverability and the full actuator, swept-tube, collision, velocity, energy-prefix, and version schema:

\[
C_{\rm run}(z)=c(z)+G(z)[-1,1]^3\in\mathfrak A_{\rm safe}(z)
\subseteq\mathfrak A_{\rm rec}(z).
\]

No sampled action, center, or finite rollout substitutes for this inclusion check.

## T_REC1 and T_REC2

**T_REC1 (one-step recoverability preservation).** If `z_t` belongs to `R` and runtime publishes an
action from a directly verified complete support `C_run(z_t) in A_safe_set(z_t)`, every state represented by the certified
successor envelope belongs to `R`.

**T_REC2 (recursive recoverability).** If `z_0` belongs to `R` and every learned action is published
from a newly verified `C_run`, induction on T_REC1 gives `z_t in R` for every normal-policy step.
This preserves the existence of a certified recovery option; it does not claim that the learned
policy itself returns to the station.

At the configured interior switching margin, `NO_GENERATOR_SET`, invalid evidence/version,
or a task-worker failure covered by the independent staged publisher, authority switches to kappa. Publisher/bus/watchdog failure itself is outside the positive theorem unless an independent hardware fail-safe proves publication. The existing
strict corridor descent and E3 energy recursion then provide the conditional finite-time return
result. If kappa's own certificate is invalid, execution fails closed. A valid kappa preview is not
sufficient: if the final publication-time recheck loses coverage after an RL, kappa, or nominal
preview, the published bookkeeping command is labelled `uncertified_emergency_brake`, authority is
`FAIL_CLOSED`, the episode terminates, no recovery child is committed, and the Bellman continuation
is zero. Mid-cycle certificate-version mutation is handled by the same rule. A preview already
classified `FAIL_CLOSED` uses a dedicated path that does not re-evaluate recovery and therefore
cannot resurrect kappa or advance a proof child.

## Voluntary charging and departure

Voluntary station approach is inferred from continuous behavior, not a discrete policy output. If
a covered normal flight ends in both `R` and the charging-admissible set, a fresh
successor-bound station-hold certificate permits the zero-duration
`RUN-ARRIVE` mode update and the visit is logged as voluntary. The flight
interval receives no charging gain. Charging begins only on a later transition
whose source is `CHARGING_RL`; remaining inside the set then executes the
certified physical hold and applies synthetic charging

\[
e_{t+1}=\min(e_{\max},e_t-d_E(z_t,h_t,w_t)+r_c\Delta t).
\]

The fixture uses capacity `30.0`, rate `2.0` units/s, and `dt=0.2` s (`0.4` units per cycle).
These are not calibrated physical values.

While the departure gate is closed, the certificate path constructs
`C_charge(z) in A_charge_set(z)`, whose direct complete-support certificate checks actuator bounds, the swept tube, the nonnegative energy prefix, current versions, and that every successor remains in the charging
set, including the lower-energy condition before charge is added. The normal accepted policy action is therefore also the physically executed action; unsafe
departure directions are absent from its support rather than post-hoc replaced. When the departure
energy and manifest checks pass, the ordinary directly verified `C_run in A_safe_set` support is restored. A state-dependent certified hold remains only an explicit certificate/numerical fallback and is physically executed and recorded. The pending goal is unchanged,
and future charging never reduces the energy required to reach the station from a flight state.

Execution sources are mode and rank restricted. A certified `kappa` step
requires a positive selected recovery level and a non-charging source. At level
zero, normal authority may execute `RUN-N` or `RUN-ARRIVE`, but cannot fabricate
a recovery step. In `CHARGING_RL`, a closed departure gate admits only verified
constrained charging or hold, and an open gate admits only verified departure;
missing support fails closed without recovery-child commitment.

After every covered κ interval, runtime reads back the exact child named by the
published recovery certificate. A positive-rank child remains active in
`BACKUP_RECOVERY` even when its realized state geometrically overlaps the
charging terminal. Only the exact committed level-zero child, together with a
fresh station-hold check, performs the zero-duration `KAPPA-ARRIVE` update. The
active child identifier, rank, and hash are part of the certificate snapshot;
they cannot be cleared or replaced between readback and mode commit. Like
`RUN-ARRIVE`, this update adds no same-step charging gain.

## Manifest and policy-authority gates

The aggregate manifest stores one `SharedBoundVersions` object for dynamics, tracking, energy,
terminal, recoverable-set/action rules, and the runtime configuration. Those assumptions must match
across every edge. Geometry, corridor, mission-manifest, and kappa identities are edge-local; each is
hash-bound by an `EdgeDependencyBinding`, and every binding hash enters the aggregate manifest.
Different edge-local IDs are therefore expected, while a changed shared bound or a tampered edge
dependency remains a hard `VERSION_MISMATCH`.

`PERSISTENT_SAFETY_GATE` applies typed prerequisites. `TASK_EDGE` and `DEPARTURE_EDGE` require their
recovery chain plus verified task/departure successor support. `RECOVERY_EDGE` requires only the
complete kappa chain, strict descent, geometry, actuator/velocity bounds, E3, terminal linkage, and
hashes; it does not require a full-rank Generator. A recoverable state with `NO_GENERATOR_SET` is a
valid kappa-backup state, not automatically a safety-certificate failure.

`POLICY_AUTHORITY_GATE` checks only states where normal RL authority is represented: task,
departure, and constrained charging support. Task states still require both goal- and
station-directed residual authority. Recovery-only cells are audited separately for kappa validity.
`POLICY_AUTHORITY_COVERAGE` reports how many eligible RL roots have a verified full-rank Generator;
coverage is a learnability/performance metric, not a safety theorem. Sigma, condition, and volume
aggregates exclude kappa-only cells. These remain synthetic software gates, not physical calibration.

The latest corrected synthetic validation passes both gates for `persistent_open` and
`persistent_energy_tight` with 1353/1353 RL-authority roots and all 36984 kappa-only cells valid in
each scenario. `persistent_obstacle` fails the safety and policy gates: 906 `recover_C_S`, 591
`task_C_B`, 1017 `task_C_D`, and 809 `task_D_C` cells have complete swept-geometry containment
failures (`minimum_geometry_slack=-1.0` in the first witnesses). Their hashes, E3 residuals,
velocity bounds, and strict descent links remain valid. This is a real synthetic certificate
infeasibility, not a version bug, typed-gate bug, or `NO_GENERATOR_SET` constructor failure.

## Replay and optimization

Replay records `u`, `eta`, `c`, `G`, candidate, executed and measured actions, backup state/reason,
energy margin, charging/departure events, pending task, manifest, and bound versions. Critics train
on `executed_action`. Only accepted Generator transitions use the affine-tanh continuous density;
backup atoms do not. `c` and `G` remain detached during actor updates.

`PersistentExecutionAuthority` is the single immutable classifier used to serialize runtime
authority into replay. Its outcomes are `RL_GENERATOR`, `KAPPA_BACKUP`, `CHARGER_CONSTRAINED`, and
`FAIL_CLOSED`. `PersistentGeneratorSAC` selects its next-state Bellman branch from that recorded
authority, not merely from mathematical Generator existence or a pre-publication preview. A
mandatory next-step backup uses
`kappa(z_next)` without Generator entropy; a closed charger uses the certified `C_charge` Generator
when available, otherwise an explicitly recorded atomic hold; fail-closed next states do not
bootstrap. This is runtime/training semantic closure, not a new SAC convergence claim.

## Manual commands

The following formal commands were created or updated but were not run in this round:

```bash
.venv/bin/python scripts/validate_persistent_certificate.py
.venv/bin/python scripts/run_persistent_env_acceptance.py --scenario persistent_open --probe all --strict
.venv/bin/python scripts/train_persistent_generator_sac.py --scenario persistent_open --legacy-fixed-graph --steps 50000
.venv/bin/python scripts/evaluate_persistent_generator_sac.py --scenario persistent_open --legacy-fixed-graph --checkpoint <path>
.venv/bin/python scripts/run_persistent_single_policy_baselines.py --scenario persistent_open --legacy-fixed-graph
```

The old `train_energy_management_sac.py` and related scripts are explicitly hierarchical ablations,
not the main method. Any future outputs remain synthetic empirical evidence and cannot establish
real-flight safety, calibrated physical bounds, or hard WCET.

## Task-independent random-goal main problem

The main path separates physical/certificate state `x` from externally assigned task goal `g`.
The actor is goal-conditioned, `pi_theta(a | x, g)`, but the certified support is not:

```text
A_rec_set(x) = {S subset A_act(x) : Post_hat_Xi_S(x,S) subset R}
A_safe_set(x) = {S in A_rec_set(x) : the complete swept-tube,
                 collision, velocity, energy-prefix, and version predicates hold}
C_run(x) = c(x) + G(x)[-1,1]^3 belongs to A_safe_set(x)
```

`CertifiedRecoverabilityAtlas` covers a certified subset of the free workspace with recovery
cells. Each cell binds geometry, dynamics, tracking, energy, terminal, and frozen-kappa proof
dependencies. Atlas construction consumes no current goal, goal seed, task edge, task waypoint,
route index, or reward. `RandomPersistentTaskWrapper` samples reproducible starts and continuous
horizontal goals only from certified atlas interiors. Reaching a goal samples the next goal
without resetting the plant; charging and backup preserve the same pending goal.

The fixed `CertifiedGoalNetwork` and `TASK_EDGE` fixtures remain legacy regressions and ablations.
They are not prerequisites for normal authority in the random-goal main method.

## T_RAND contracts

**T_RAND1 (random certified initialization).** If the initial distribution has support inside the
certified atlas, the existing T_REC initialization premise holds.

**T_RAND2 (goal-independent recursive recoverability).** For any admissible goal sequence and any
goal-conditioned learned policy, T_REC2 remains valid when every normal action is published from
the task-independent directly verified `C_run(x) in A_safe_set(x)`. This guarantees recoverability, not sampled-goal
completion.

**T_RAND3 (goal-independent support).** At identical physical/certificate state and versions,
changing only `g` leaves `E^kappa`, recoverable membership, kappa proof, `c`, `G`, action bounds,
and atlas identity unchanged. Actor output may change.

## Recovery versus RL-authority viability

`R` contains every state with a certified finite kappa return and sufficient recovery energy.
An ideal energy-augmented `R_RL` would be the task-independent atlas fixed point of recoverable
cells that also have full-rank Generator support with a complete successor in `R_RL` or
`G_charge`. Therefore `R_RL subset R`; cells in `R` but outside `R_RL` remain legitimate
kappa-only recovery cells. The current implementation computes only a topology/cell-ID candidate
kernel and does not establish equality with this state-level fixed point.

The canonical safety theorem requires `C_run in A_safe_set` and hence return to `R`. The optional
stronger normal-authority theorem would additionally require direct complete-support continuation into `R_RL union G_charge`, where `A_cont`
preserves `R_RL` or enters the certified charging set. The safety-neutral center may apply
atlas-state feedback needed for invariant support, but it receives no goal, task route, waypoint,
or reward. Goal changes must leave `R`, `E^kappa`, `c`, `G`, and certificate identity unchanged;
an implemented stronger capability would also bind `R_RL` and its continuation target.

The charging terminal has a formal level-zero recovery certificate.  It binds terminal geometry,
dynamics, tracking, energy, terminal and kappa versions, and the atlas core hash; its recovery
energy upper bound is exactly zero. It additionally binds a local terminal hold controller whose complete
successor envelope must stay in the charging set; zero-step recovery therefore does not imply a
zero acceleration command in the presence of residual velocity. This prevents completed recovery
from being reinterpreted as a missing nonterminal successor. A closed departure gate uses charger-constrained support or certified hold;
an open departure in the canonical theorem is accepted only when its complete successor returns to
`R`. Returning to `R_RL` is an optional stronger gate that the current candidate kernel does not
certify at state level.

Generator-SAC diagnostics decompose normalized and physical log density. The physical density keeps
the affine determinant term exactly. The primary default uses the normalized-coordinate alpha
residual `log pi_eta`; actor and Bellman terms still use `log pi_a`, so the executed policy density
and certificate semantics are unchanged. This is equivalent to a state-dependent physical entropy
target shifted by `log|det G|` and is invariant to action-unit or uniform affine-support scaling.
A constant physical-coordinate target remains an explicit ablation rather than the primary default.

The persistent reward uses `backup_intervention_cost` as an event cost only when authority first
transfers into `KAPPA_BACKUP`. Kappa continuation does not repeat that intervention charge. Recovery
steps still pay elapsed-time and energy costs, and charging steps retain their dwell cost. Telemetry
therefore distinguishes `backup_recovery_count`, `kappa_backup_steps`, and
`backup_intervention_reward_events`.

Q-value ranking alone is not treated as evidence that the actor receives an actionable gradient.
`actor_gradient_diagnostics.py` audits the local critic action derivative, affine-tanh Jacobian
transmission, separate Q/entropy parameter gradients, interpolation landscapes, goal-conditioned
actor and critic Jacobians, and critic action-input conditioning. Its frozen-critic Q-only update
uses only recorded certified RL states and critic gradients; the oracle is evaluation-only and never
enters replay, loss targets, or rewards.

The counterfactual goal critic audit additionally freezes the complete physical/certificate state
and changes only `goal_delta` and `distance_to_goal`. It verifies that `c`, `G`, recoverable-set
membership, RL-authority membership, `E^kappa`, continuation support, and certificate hashes remain
unchanged. Goal-dependent critic values are reported separately from goal-dependent action gradients
and searched critic-preferred actions; the former does not imply the latter.

The Bellman/replay audit next asks whether the one-step training target itself contains a usable
goal-by-action interaction and whether rollout replay identifies it locally. It constructs
counterfactual non-completion matrices `R(g,a)`, the physical-density Bellman target `Y(g,a)`, and
the learned `Q(g,a)` over the same certified action set. Additive explained variance is reported
alongside finite-set preferred-action sensitivity; interaction variance alone is not treated as
task-control competence when the preferred action remains goal-invariant.

Replay neighborhoods are goal-independent and use recovery cell, physical state, energy margin,
and mode. The audit reports goal angular spread, action covariance, normalized support coverage,
and effective rank of `g tensor-product eta`. Hypothetical counterfactual augmentation recomputes
goal observations, rewards, and targets for the same physical transition only when the relabeled
goal is not completed. It is diagnostic only: no relabeled item is inserted into replay and no
actor or critic is updated.

## Task-neutral support expressiveness

Random-persistent normal support no longer follows the directed velocity/action of the offline
coverage trace. Coverage positions seed zero-velocity safety cells; each cell retains its own
independently certified kappa return chain. The normal center is a local position/velocity
stabilizer, while multiple goal-independent viable successor cells are considered and Generator
scales are enlarged only through the complete verifier. Task goals affect the policy latent, not
`c`, `G`, recovery evidence, or successor eligibility.

The best-in-Generator oracle is a diagnostic, not a controller in the main method and not a safety
proof. It establishes whether a goal-aware selector can make progress inside the certified support
before additional SAC training is justified.

## Crossed causal optimization audit

`audit_crossed_horizon_goal_coverage.py` evaluates horizons 1, 3, 5, and 10 under observed-goal and
strict non-completion counterfactual coverage. Every comparison reuses the same physical branch,
candidate actions, disturbance convention, reward coefficients, certificates, `c`, and `G`.
Observed replay has one goal per physical transition, so within-state goal-preference sensitivity is
not identifiable there; this absence is reported rather than filled with invented transitions.

The bootstrap is reported as reward return, target Q, normalized-policy entropy, and
`log|det G|` support-volume contributions. Physical, no-entropy, and normalized-entropy targets are
diagnostic views of frozen data. A disposable QNetwork fit uses fixed targets and an order-preserving
per-state-goal action-advantage transform to test representation capacity. It never changes actor,
critics, target critics, replay, reward, or safety support.

## Task completion and early goal exposure

Task completion is evaluated only from `TASK_RL` against the pre-transition pending goal. The
goal-radius set is closed, including the exact numerical boundary. Distances observed while
`KAPPA_BACKUP` or `CHARGING_RL` remain telemetry only and cannot complete the pending task.

The initial 2k physical-temperature runs completed no tasks and therefore exposed each network to
one continuous random goal. They must be labeled `SINGLE_GOAL_TRAINING_DISTRIBUTION`; their
counterfactual goal sensitivity does not characterize learning under a true multi-goal training
distribution. The training entry point now supports explicit episodic multi-goal exposure through
`--goal-exposure-reset-steps`. This is a collector-only reset: it preserves the same SAC agent and
replay, stores the real final physical successor, and prevents Bellman bootstrap across the reset
boundary. Persistent evaluation semantics remain unchanged and never perform periodic exposure
resets.
