> **LEGACY / SUPERSEDED:** Historical research provenance only. This file is not an active method, theorem, implementation, test, or README authority. The active route is in docs/new_theory/.

# Canonical Paper Theorem

Status: **REPAIRED; CURRENT-HASH BLIND RE-AUDIT PENDING**. A prior same-family review accepted the theorem conditionally, but the dependency ledger and runtime-alignment record changed after that audit. Software tests never discharge the physical soundness premises.

This is the single canonical theorem source. `DERIVATION_PACKAGE.md` is historical background and is not a proof dependency.

## 1. Typed objects and primitive certificate schemas

Let `x in X_cert` contain the physical uncertainty state, lower battery energy `e^-(x)`, evidence versions, timing metadata, and the identity of any selected proof node. Let `q in Q` be the pending task identity, fix `Next:Q -> Q`, and let the covered hybrid mode be `m in {NORMAL, BACKUP, CHARGING}`. `DEPART` is a guarded transition label, not a mode. `FAIL_CLOSED` is outside the positive theorem.

For a physical action set `A`, `Post(x,A)` and `Tube(x,A)` are the true robust endpoint and complete within-step swept-tube correspondences. `EnergyTube(x,A)` is the true lower-energy trajectory over the whole command interval. Their certificate objects are `Post_hat`, `Tube_hat`, and `EnergyTube_hat`.

`Verify_t(A,Xi)` is a primitive, deterministic pre-publication check of a certificate record `Xi` bound to action set `A`. It checks the declared action set, calibrated descriptor, hashes and versions, validity interval, actuator bounds, set inclusions, and publisher/readback binding. The Level-0 software model has one non-reentrant controller: `step` and `reset` calls are serial, the snapshot reader is pure, and every in-scope asynchronous certificate-state writer must acquire the shared transaction lock. Covered software publication uses that lock for compare-and-publish. Concurrent/reentrant environment calls, unregistered writers, hardware/RTOS writers, and actuator-bus atomicity are outside Level 0 and require deployment evidence. `Verify_t` does not prove its own physical calibration. Fix an admissible model/calibration package and certificate epoch. The external physical premises, universally quantified over every admissible disturbance and outcome in the declared domain, are:

The live check compares both version identifiers and the bound dependency
fingerprints used to construct the proof object.  Timestamp/validity metadata is
part of the publication snapshot used for freshness, although ordinary elapsed
time is excluded from the replay-group epoch.  Equality of human-readable
version strings alone is not a dependency check.

```text
Verify_t(A,Xi) and Xi valid on the complete interval [t,t+1]
  => Post(x_t,A)       subset Post_hat_Xi(x_t,A),
     Tube(x_t,A)       subset Tube_hat_Xi(x_t,A),
     EnergyTube(x_t,A) bounded below by EnergyTube_hat_Xi(x_t,A),
     the realized command belongs to A;
FREE_Xi subset true obstacle-free space;
for every verified recovery node nu and realized kappa outcome,
     d_t <= d_bar_nu and e^-(x_{t+1}) >= e^-(x_t)-d_bar_nu.
```

Thus “verified” below always means a finite schema of independently checkable predicates; it is not shorthand for the theorem conclusion.

For a maximal physical execution `omega` with command-transition indices
`t<T(omega)`, distinguish two stages. `CmdCovered_omega(t)` means that the
physical-command certificate is successfully checked before command
publication, is valid for the entire interval `[t,t+1]`, and atomic
publication/readback binds the realized command to its verified set.
`HybridCovered_omega(t)` means `CmdCovered_omega(t)` and, for any branch whose
returned hybrid successor requires a realized-successor classifier or committed
child readback, that second certificate is checked after the physical interval
but before publication of `(x_{t+1},m_{t+1},q_{t+1})`, under one atomic
snapshot-check/mode-commit transaction.  For a `RUN`, a fresh `NORMAL` or
`ARRIVE` result satisfies the second stage; `INVALID` does not.  For `KAPPA`,
the second stage binds the realized state and active proof identity to the
hash-declared lower-rank child and permits `CHARGING` only for the exact
level-zero child.  Branches with no additional successor classifier satisfy
the second stage trivially.  Define

```text
Covered_omega(t) = HybridCovered_omega(t).
```

Thus an `INVALID` arrival can end hybrid coverage at index `t` while
`CmdCovered_omega(t)` still proves safety of the already executed physical
`RUN` interval. Define the pathwise proof horizon

```text
tau_cov(omega) = inf ({t<T(omega) : not Covered_omega(t)} union {T(omega)}),
```

where `T(omega)` and the infimum may be `+infinity`. This is a pathwise discrete complete-step boundary, not an asserted controller stopping time for a latent calibration failure. The physical execution need not have a covered tail; the covered relation below is required exactly at existing indices `t<tau_cov(omega)`. An uncovered emergency-brake or publisher failure is not relabelled as certified `kappa`.

## 2. Selected frozen-recovery certificate

Within one certificate epoch, `kappa` is frozen and its selected recovery proof is a finite directed acyclic graph with rank map `r:V -> N_0`. A proof node `nu` contains a region `K_nu`, finite nonnegative one-step consumption upper `d_bar_nu in R_{>=0}`, finite nonnegative stored energy upper `E_nu in R_{>=0}`, and hash-bound child nodes `Child(nu)`. Every realized consumption is a finite scalar `d_t in R_{>=0}`. The graph certificate covers every node and edge through the endpoint of each covered transition; the selected child proof is installed at that endpoint with its identity and validity intact. Outward rounding obeys `round_up(z)>=z`. The graph obeys:

```text
r(nu)=0  <=> Child(nu)=empty, K_nu subset G_ch, and E_nu=0;
r(nu)>0  => Child(nu) is nonempty and, for every robust successor x+ under kappa,
             the certificate selects a child sigma_nu(x+) in Child(nu),
             x+ in K_{sigma_nu(x+)},
             r(sigma_nu(x+)) < r(nu),
             E_nu >= round_up(d_bar_nu + max_{mu in Child(nu)} E_mu).
```

Every nonterminal node certificate additionally checks the `kappa` actuator set and the explicit predicates

```text
Tube_hat_nu(x,kappa(x)) subset FREE_Xi_nu,
EnergyTube_hat_nu(x,kappa(x)) >= 0 over the complete interval,
Post_hat_nu(x,kappa(x)) subset union_{mu in Child(nu)} K_mu,
```

with the displayed selector assigning every successor to a child. The external physical premise, not the Boolean checks alone, supplies

```text
realized one-step consumption d_t <= d_bar_nu,
e^-(x_{t+1}) >= e^-(x_t) - d_bar_nu.
```

For a robust `kappa` execution whose recovery transitions remain covered until first entry into `G_ch`, define the exact covered first-passage energy

```text
J_cov^kappa(x) = sup_covered_path sum_{t < tau_G} d_t.
```

The supremum ranges only over robust paths for which every recovery transition through `tau_G` is covered. Finite selected-rank descent and the outward-rounded recursion imply `J_cov^kappa(x) <= E_nu`. No statement is made about the true physical first-passage cost after coverage is lost, and no bound of the form `E_nu <= r(nu)c_max` is asserted.

Fix a constant executable reserve `m_e >= 0` and terminal reserve `e_G >= 0`. The recoverable set is

```text
R = {x : x lies in a current selected node nu,
         its recovery and geometry certificates are valid,
         e^-(x) >= E_nu + e_G + m_e}.
```

Rank zero is charge-terminal and is included in `R` only when its energy inequality also holds.

## 3. Normal, charging, and departure certificates

Because runtime certification is performed on a complete support, define set-valued predecessor certificates rather than infer them from unverified singleton objects:

```text
A_rec_set(x) = {S subset A_act(x) : Post_hat_Xi_S(x,S) subset R},

A_safe_set(x) = {S in A_rec_set(x) :
                 Tube_hat_Xi_S(x,S) subset FREE_Xi_S,
                 Post_hat_Xi_S(x,S) satisfies successor collision and velocity bounds,
                 EnergyTube_hat_Xi_S(x,S) >= 0 over the complete interval,
                 all versions and the complete-set certificate are current}.
```

The tube condition covers the complete present swept motion. The energy condition covers the entire interval, not merely its endpoint. A learned Generator command is executable only after its complete support is verified directly:

```text
C_run(x)=c(x)+G(x)[-1,1]^3 belongs to A_safe_set(x).
```

The charge-admissible set `G_ch` includes position, velocity, lower energy, free-space evidence, charge-contact admissibility, and current versions. An action support `S` has a complete charging certificate record `Xi_ch,S` at `x in G_ch` exactly when all of the following are checked:

```text
S subset A_act(x),
Tube_hat_Xi_ch,S(x,S) subset FREE_Xi_ch,S,
EnergyTube_hat_Xi_ch,S(x,S) >= 0 over the whole interval,
Post_hat_Xi_ch,S(x,S) subset G_ch.
```

`Post_hat_Xi_ch,S` contains the joint kinematic disturbance, flight consumption, and charging gain

```text
e+ = min(e_max, e - d_E(x,a,w) + r(x,w) Delta t).
```

Both `C_charge(x)` and a certified plant hold `H(x)` must satisfy this complete schema. Endpoint charging gain cannot compensate for an uncertified negative-energy prefix.

Let `E_dep^req(x)` be the versioned scalar departure-energy requirement and define

```text
D_e(x) = e^-(x) - E_dep^req(x) - e_G - m_e.
```

`D_e(x)>=0` is only the source battery prerequisite.  Let `h_N` be the
hash-bound normal-successor proof node selected by the prepared support, let
`K_{h_N} subset R` be its certified state cell, and let `m_sw` be the strict
normal-to-recovery switching margin.  The final departure handoff gate is open
only when `D_e(x)>=0` and a complete support satisfies

```text
C_dep(x) belongs to A_safe_set(x),
h_N belongs to the implemented candidate normal-authority kernel,
Post_hat_Xi_dep(x,C_dep(x)) subset K_{h_N} subset R,
inf_{x+ in Post_hat_Xi_dep} [e^-(x+) - E_{h_N}] > m_sw.
```

Before the returned hybrid state is published, the realized successor is read
back against the same `h_N`, dependency snapshot, and strict energy margin in
the certificate transaction.  Only then is the mode changed to `NORMAL`.  If
the scalar prerequisite is true but this final handoff gate is false, the gate
is closed for the hybrid relation and only verified `CHARGE/HOLD` is covered.
The scalar gate is necessary but never substitutes for complete predecessor,
normal-successor identity, or readback.

## 4. Covered hybrid transition relation

For each `q`, define the compatible hybrid domain

```text
I_N(q) = {(x,NORMAL,q)   : x in R},
I_B(q) = {(x,BACKUP,q)   : x in R with selected rank at least 1},
I_C(q) = {(x,CHARGING,q) : x in G_ch},
I(q)   = I_N(q) union I_B(q) union I_C(q).
```

After a covered `RUN` physical interval and before the returned hybrid state is
published, let the successor classifier take values in
`{NORMAL, ARRIVE, INVALID}`.  A non-charge-admissible successor is `NORMAL`.
For a charge-admissible candidate, a snapshot-bound check returns `ARRIVE` only
if it proves `x+ in G_ch`, verifies the state-dependent station hold, and the
live dependency snapshot is unchanged; a fresh negative check is `NORMAL`, and
dependency drift or check failure is `INVALID`.  Write `Arrive_t(x+)` for the
`ARRIVE` case.  Its associated mode update has zero physical duration and zero
energy gain.  Since `RUN` already proves `x+ in R`, an `Arrive_t` successor lies
in `R intersection G_ch`.  `INVALID` is outside the covered relation: runtime
terminates without entering `CHARGING`, while the already executed physical
`RUN` is not retroactively relabelled as an emergency command.

The covered relation is exhaustive:

1. `RUN-N`: from `I_N(q)`, execute a member of verified `C_run`. Let `q'=q` without task completion and `q'=Next(q)` with an explicit completion event. If the successor classifier is `NORMAL`, the successor mode remains `NORMAL` and the transition enters `I_N(q')`.
2. `RUN-ARRIVE`: execute the same covered `RUN` physical interval and the same task update rule, but require `Arrive_t(x+)`. The atomic zero-duration successor-mode update is `NORMAL -> CHARGING`, yielding `I_C(q')`. No charging gain is credited on the `RUN` interval; charging begins only on a later `CHARGE` or `HOLD` transition whose source is `I_C`.
3. `KAPPA`: from a positive-rank state in `I_N(q)` after a normal-authority refusal, or from a positive-rank state in `I_B(q)`, execute the selected certified `kappa` command. A positive-rank child enters `I_B(q)`; a rank-zero child enters `I_C(q)`. The selected child identity, not minimum overlap rank, is committed for the next step. A rank-zero `I_N(q)` state has no fictitious recovery action: it may execute `RUN-N` or `RUN-ARRIVE`, or the covered execution prefix ends. `KAPPA` is never a covered source branch from `I_C`.
4. `CHARGE` or `HOLD`: from `I_C(q)` with the departure gate closed, execute verified `C_charge` or `H` and remain in `I_C(q)`.
5. `DEPART`: from `I_C(q)` with the final handoff gate open, execute verified `C_dep`, commit its selected `h_N`, and read back the realized successor against `K_{h_N}`, current dependencies, and the strict normal margin. The atomic successor mode update is then `NORMAL`, yielding `I_N(q)`. A source state satisfying only the scalar battery prerequisite remains on the closed-gate `CHARGE/HOLD` branch.

No covered transition except a completed `RUN-N` or `RUN-ARRIVE` changes `q`. In particular, normal noncompletion, `NORMAL -> BACKUP`, every `KAPPA` arrival, charging, hold, and departure preserve the same pending task identity.

## 5. Theorem — covered regenerative recoverability and same-task resumption

Fix any admissible model/calibration package and certificate epoch satisfying Sections 1–3. For every `q_0 in Q`, every compatible initial state `(x_0,m_0,q_0) in I(q_0)`, every maximal physical execution `omega`, and every admissible disturbance path, assume the command branch at each existing index `t<tau_cov(omega)` is exactly one of the covered relations in Section 4. Then, for every `t<T(omega)` with `t<tau_cov(omega)`:

1. The entire realized swept motion is collision-free and the lower battery trajectory is nonnegative over `[t,t+1]`.
2. The successor belongs to `I(q_{t+1})`; hence the union of task-indexed compatible domains is invariant over covered transitions.
3. `q_{t+1}=q_t` except on an explicit completed `RUN-N` or `RUN-ARRIVE` transition, where `q_{t+1}=Next(q_t)`.
4. Every `RUN-N`, `RUN-ARRIVE`, or `DEPART` physical successor is in `R`; a `DEPART` successor additionally belongs to its selected `K_{h_N}` and retains the configured next-cycle normal margin. Every `RUN-ARRIVE`, closed-gate `CHARGE`, or `HOLD` successor is in `G_ch`. A `RUN-ARRIVE` interval receives no charging gain.
5. If `KAPPA` is entered with selected rank `r_0` and coverage persists through the recovery segment, `G_ch` is reached in at most `r_0` such transitions and its robust covered energy is at most `E_nu`. If coverage ends earlier, only strict rank decrease and safety on the covered prefix are claimed.
6. Any covered backup–arrival–charging–departure segment resumes `NORMAL` with exactly the task identity that was pending when backup began.

The theorem does not guarantee task completion, perpetual proof refresh,
availability of positive-volume normal support beyond the committed immediate
handoff, or safety of any uncovered `FAIL_CLOSED`, publisher, hardware, or
calibration failure.  The implemented candidate kernel named above is not the
optional state-level `R_RL` fixed point of Section 8.

## 6. Proof

Fix a covered transition and assume its source is in the compatible domain.

For either `RUN` subcase, direct complete-set verification gives `C_run in A_safe_set`. Sound outer containment, `FREE_Xi_run subset true obstacle-free space`, and the complete-set tube and energy conjuncts prove within-step collision and energy feasibility. The complete-set predecessor conjunct gives `x+ in R`, and the explicit task rule selects `q'` as `q` or `Next(q)` only on completion. For `RUN-N`, the mode remains `NORMAL`, so the successor is in `I_N(q')`. For `RUN-ARRIVE`, the successor-bound `Arrive_t` certificate additionally gives `x+ in G_ch`; its zero-duration, zero-energy mode update therefore yields `I_C(q')` without retroactively adding charge to the flight interval.

For `KAPPA` at node `nu`, its complete certificate gives the same within-step safety. Every realized successor is assigned to a committed child `mu` with smaller rank. Moreover,

```text
e^-(x+) >= e^-(x)-d_bar_nu
          >= E_nu+e_G+m_e-d_bar_nu
          >= E_mu+e_G+m_e.
```

Thus a positive-rank child remains in `R` and `BACKUP`; a rank-zero child lies in `G_ch` and enters `CHARGING`. Strict descent in `N_0` proves arrival after at most the initial rank, provided all recovery transitions through arrival are covered. Induction over the same recursion proves `J_cov^kappa <= E_nu` on that covered path class.

For `CHARGE` and `HOLD`, the complete hybrid tube, true-FREE premise,
energy-prefix, and successor predicates give within-step collision/energy
feasibility and `x+ in G_ch`; the transition rule preserves both `CHARGING`
and `q`. For `DEPART`, `C_dep in A_safe_set` gives within-step feasibility,
complete-set containment gives `x+ in K_{h_N} subset R`, and the lower-envelope
inequality gives the strict next-cycle normal margin.  Snapshot-bound realized
successor readback preserves that same node identity despite geometric overlap;
only then does the atomic mode rule give `NORMAL` while preserving `q`.

These five branch labels, with the two mutually exclusive `RUN` subcases, exhaust the covered relation. Induction on transition index proves items 1–4. Applying the rank induction to each maximal recovery segment proves item 5. Applying the task-update rule across backup, `KAPPA` arrival, charge, and departure proves item 6. All conclusions stop before the first existing transition with `not Covered_omega(t)`. QED.

## 7. Why ordinary safe-set augmentation is insufficient

A collision-safe invariant set augmented with battery does not imply the theorem. A task loop can consume energy forever without a proper route to a charger; a charger can be absorbing without a certified departure; endpoint energy can hide intra-step depletion; and safety authority alone does not preserve task identity. Proper selected-rank recovery, complete hybrid tube certificates, departure predecessor inclusion, and an independent task transition rule are all necessary.

## 8. Optional stronger capability, excluded

An energy-augmented greatest fixed point `R_RL` could certify persistent positive-volume normal authority. Current code computes only a topology/cell-ID candidate kernel and therefore does not establish state-level `R_RL`, perpetual normal authority, or lifecycle liveness.
