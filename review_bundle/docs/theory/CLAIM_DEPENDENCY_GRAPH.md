> **LEGACY / SUPERSEDED:** Historical research provenance only. This file is not an active method, theorem, implementation, test, or README authority. The active route is in docs/new_theory/.

# Claim Dependency Graph

```text
primitive Verify_t schema + calibrated dynamics / tracking / sensing / energy / timing envelopes
  -> sound Post_hat, Tube_hat, EnergyTube_hat on complete covered intervals
  -> frozen-kappa selected-child progress, tube, and one-step cost certificates
       -> versioned outward-rounded upper recursion E^kappa
       -> exact covered-path first-passage J_cov^kappa
       -> domination lemma J_cov^kappa <= E^kappa
  -> recoverable set R
  -> complete-set predecessor collection A_rec_set
  -> A_safe_set and directly verified full support C_run
  -> exhaustive execution authority / mode / task transition relation
       -> RL_GENERATOR: continuous affine-tanh branch
       -> KAPPA_BACKUP: deterministic atom
       -> CHARGER_CONSTRAINED: verified C_charge or certified hold
       -> FAIL_CLOSED: no bootstrap continuation
       -> RUN-ARRIVE: successor-bound zero-duration NORMAL-to-CHARGING update, no same-step charge
       -> KAPPA-N/KAPPA-ARRIVE: exact committed child readback, positive child stays BACKUP, exact level-zero child enters CHARGING
  -> recursive recoverability and within-step collision/energy feasibility
  -> finite kappa recovery
  -> charging invariance
  -> certified departure into R
  -> same pending-task resume over backup/charge/departure
```

The optional stronger branch is separate from the canonical theorem:

```text
recoverable energy-augmented cells + complete positive-volume supports + G_charge seed
  -> finite-atlas monotone viability operator Phi
  -> ideal greatest fixed point R_RL subset R
  -> A_cont
  -> persistent positive-volume normal-authority capability
```

The learned-field branch is parallel:

```text
collision labels/operator -> B_theta + uncertainty
recovery-energy labels/SSP operator -> E_psi + uncertainty
  -> proposal ranking / sampling efficiency only
  -> final continuous verifier
  -> C_run
```

No edge from a learned prediction directly reaches strict certification.

## Cycle audit

The intended order is acyclic because `kappa` progress and energy certificates are built before task-action support. The strict normal-authority predecessor branch is bounded by `R`; the lifecycle theorem continues through recovery, charging, certified departure, and same-task resumption, but it does not depend on the optional `R_RL` branch. The current topology-only graph kernel must not be substituted for the energy-augmented state-level fixed point.

## Implementation breaks in the graph

1. Hold execution and replay epoch identity are repaired at the software level.
2. The `candidate graph kernel -> R_RL` edge is invalid: energy, complete support, and terminal seeding are missing.
3. The `safety lifecycle -> same task resume` edge requires the explicit task transition relation; runtime authority alone does not imply it.
4. `CmdCovered(t)` is established before command publication for the whole physical interval; `HybridCovered(t)` additionally binds postinterval RUN classifiers and κ-child readback before hybrid-state publication. Treating a latent lost physical premise as an observable stopping time would be an invalid edge.
5. `KAPPA_BACKUP` additionally requires a positive selected recovery rank and a non-charging source. Rank-zero normal refusal or any charging-source certificate failure ends the positive prefix fail-closed.
6. Recovery proof validity compares construction and live dependency fingerprints, versions, and validity time; snapshot equality alone cannot detect a dependency replacement that predates all snapshots.
7. A geometric terminal overlap cannot choose the κ child or clear an active positive-rank commitment; child identity/rank/hash and the mode commit are snapshot-bound.
