> **LEGACY / SUPERSEDED:** Historical research provenance only. This file is not an active method, theorem, implementation, test, or README authority. The active route is in docs/new_theory/.

# Open Theory Issues

The pre-repair theorem failed a fresh blind audit with one FATAL and eight CRITICAL findings. The canonical statement was rewritten, and a fresh normalized same-family review accepted the repair with zero open issues. Internal mathematical closure is therefore conditional/provisional; physical and deployment premises remain unverified in every case.

## Closed internal proof re-review gate

- `OTI-P01`: **Closed conditionally.** Compatible initial hybrid domains and the exhaustive `RUN/KAPPA/CHARGE/HOLD/DEPART` relation passed the blind counterexample audit.
- `OTI-P02`: **Closed conditionally.** Selected-child coverage, terminal seeding, lower-energy algebra, within-step tubes, and the finite-arrival coverage horizon passed the rank/energy audit.
- `OTI-P03`: **Closed conditionally.** `Verify_t` is a primitive finite schema and pathwise `tau_cov(omega)` is only the first actually uncovered complete transition, not an assumed observable stopping time for latent failure.

## Optional stronger theory, out of scope of the canonical theorem

- `OTI-S01`: Implement the energy-augmented state-level `R_RL` greatest fixed point with complete full-rank supports and an explicit charging seed. Current code exposes only a candidate graph kernel.
- `OTI-S02`: Prove perpetual full-interval certificate availability or stochastic availability if a claim beyond `tau_cov(omega)` is desired.
- `OTI-S03`: Prove task completion/liveness if a claim beyond same-task identity resumption is desired.

## Empirical and implementation follow-ups

- `OTI-E01`: Generator residual benefit is not isolated from center, support, fallback, observations, or data distribution.
- `OTI-E02`: The goal-independent feature whitelist, separate mean/scale fields, dedicated proposal RNG, training-behavior ranker, authority bypass, NaN fallback, zonotope mapping, and certificate import boundary are implemented and unit-tested. Offline labels, calibration checkpoints, training-script shadow wiring, and causal benefit evidence remain open. Deployed K-sample ranking is excluded because it would change the affine-tanh policy density.
- `OTI-E03`: **Closed for the three synthetic random-persistent 2x fixtures.** `tests/test_two_x_energy_domination.py` and the current action-rule-v2 artifact `artifacts/theory/two_x_energy_domination_action_rule_v2.json` compare every cell's stored one-step upper against the exact separable plant maximum on its full velocity and tracking-expanded action boxes, then check the cumulative recursion. Two independent current runs are byte-identical; the earlier artifact is preserved but has pre-v2 atlas hashes and is not current-snapshot evidence. This remains synthetic numerical evidence, not aircraft calibration.
- `OTI-E04`: Teacher replay prefill and actor imitation require a 2x2 factorial.
- `OTI-E05`: **Closed at the software-contract level.** `tests/test_multigoal_exposure_protocol.py` now traces real RL Generator, kappa backup, physically executed charger hold, unavailable-hold fail-closed, and direct fail-closed cycles through transition construction, replay validation/sample, and `PersistentGeneratorSAC.update`; task/goal/energy metadata are complete on every exercised branch. Physical publication remains an external premise.
- `OTI-E06`: Regenerate the RL ablation artifacts after removing the invalid train/eval modulo join.
- `OTI-E07`: **Closed for the exercised random-open runtime trace.** The real kappa round-trip test now checks that the next selected cell is exactly the current certificate's committed child and has strictly smaller level before replay/update. Broader scenario/property coverage remains desirable software evidence, not a new theorem premise.
- `OTI-E08`: The default temperature update now uses the support-normalized `log pi_eta` residual while actor/Bellman terms retain the physical density; constant physical-coordinate targeting is an ablation. Observation-level Markov sufficiency remains unproved because the critic does not receive the full certificate/proof identity, so no neural-observation contraction claim is retained.
- `OTI-E09`: Generator transitions now reject nonfinite/misshaped actions and singular current/next `G`; the proposal ranker consumes each candidate's nominal successor features when scoring recovery energy. These close software-contract findings, not learned-benefit evidence.
- `OTI-E10`: Successful-only teacher replay is explicitly outcome-conditioned. In stochastic settings it is not claimed to be an unbiased Bellman sample without deterministic-transition or outcome-independent-selection assumptions.
- `OTI-E11`: **Closed at the software-contract level.** Valid KAPPA, RL, and nominal previews followed by failed publication-time recovery checks now publish `uncertified_emergency_brake`, record `FAIL_CLOSED`, terminate, exclude κ metrics/child commitment, and yield zero Bellman bootstrap. A preview already classified `FAIL_CLOSED` uses a dedicated path that cannot re-evaluate and resurrect κ. All branches pass the 440-test current-snapshot suite.
- `OTI-E12`: **Closed for the synthetic plant contract.** `RecoverabilityActionCertificate` now hash-binds explicit complete current swept-hull and within-step energy-prefix predicates; endpoint-safe/interior-gap and endpoint-energy-safe/prefix-unsafe counterexamples are regressions. Physical tube/envelope calibration remains external.
- `OTI-E13`: **Closed at the executable source prerequisite.** Departure compares the certificate-state lower battery `e^-` with route requirement plus margin rather than nominal plant energy. This scalar check is explicitly not the final handoff gate.
- `OTI-E14`: **Closed at the software-contract level.** A certificate-version mutation during task candidate construction no longer rejects the task and then publishes a stale κ label. The watchdog emits an uncovered emergency source, runtime clears the stale recovery hash, persistent authority becomes `FAIL_CLOSED`, and no proof child is committed. Targeted and 440-test full-suite regressions pass.
- `OTI-E15`: **Closed at the software-contract level.** Acceptance traces now distinguish a covered certified κ fallback from an uncovered emergency command. Concrete root causes remain in `fallback_reason` instead of being erased by the authority category; the complete failure matrix and full suite pass.
- `OTI-E16`: **Closed by targeted software regressions; complete-suite rerun pending.** The final departure gate now requires the prepared complete support to name a candidate normal-authority successor and retain a strict robust successor switching margin. Runtime commits and reads back that exact normal successor before changing to `TASK_RL`; terminal geometric overlap cannot silently replace it with a rank-zero recovery node. This is not closure of optional `OTI-S01`.

## External evidence limitations

- Aircraft sensor/dynamics/tracking/energy/terminal calibration, including nearest-return sensing soundness.
- RTOS WCET, atomic actuator-bus publication, action readback, and independent hardware fail-safe behavior.
- HIL and real-flight validation.
- Dynamic obstacles and global unknown-environment coverage.

## Novelty blocker

- `OTI-N01`: Task persistification with battery-state forward invariance, nominal-input modification, and charging is established by Notomista and Egerstedt; limited-duration safety/value learning with task abandonment and charging is established by Ohnishi et al. The remaining robust certificate/execution-authority/task-identity composition lacks a separation theorem showing it cannot be reduced to augmented-state CBF/LDCBF machinery. Until such a separation or decisive matched evidence exists, novelty criterion K remains partial.

## Repairs confirmed by the internal blind review

- Current-step swept-tube containment added to the action predicate.
- Negative plant-energy successors retained by the outer envelope.
- Executable reserve fixed constant; general state-dependent formula corrected.
- Exact covered-path joint `J_cov^kappa` separated from upper certificate `E^kappa`; post-coverage physical cost is explicitly unclaimed.
- Recoverability uses lower battery energy.
- Selected proof-certificate rank replaces minimum membership rank.
- Terminal certificate validates finite zero recovery energy.
- Certified charger hold is physically executed and recorded; charging-set verification includes energy.
- Hybrid charging and departure predicates are explicit in the canonical theorem.
- Publisher/bus failure removed from positive kappa-takeover claims.
- Invariance restricted to existing transitions before pathwise `tau_cov(omega)`; no latent calibration failure is assumed to be an observable controller stopping time.
- Pending task identity is a separate transition invariant.
- Replay epoch composes runtime certificate epoch and mission manifest.
- Primary method fixed to Persistent Generator-SAC; SB3 studies cannot lend it evidence.
