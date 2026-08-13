> **LEGACY / SUPERSEDED:** Historical research provenance only. This file is not an active method, theorem, implementation, test, or README authority. The active route is in docs/new_theory/.

# EA-MAPPO Theory Master

Status: **CANONICAL NARROW THEOREM UNDER FINAL ARRIVE/AUTHORITY RE-AUDIT — FULL ICLR SUBMISSION NOT READY**  
Date: 2026-08-11  
Scope: `review_bundle/`

This is the living theory workspace. `PAPER_THEOREM.md` is the single canonical theorem source and supersedes conflicting historical statements in `DERIVATION_PACKAGE.md`.

## 1. Primary method decision

The only primary SAC-family method is **Persistent Generator-SAC with independently verified runtime authority**. Standard SB3 SAC is a learnability/baseline implementation, and recovery-guided SB3 SAC is a supporting training-intervention protocol. Neither supplies evidence for the learned Generator residual.

This closes method identity, not the empirical gap: current artifacts support the certified runtime and task-oriented center on synthetic fixtures, but do not show a material learned-residual advantage. The paper must say so until a matched non-saturated Generator ablation is run.

## 2. Frozen project constraints

1. One learned continuous SAC-family task policy remains the task controller.
2. Strict physical safety is independent of learned-field prediction accuracy.
3. The frozen recovery controller `kappa` is the independently certified last-resort controller.
4. No learned discrete CHARGE, RETURN, or GO_HOME head is introduced.
5. Task goals may condition the actor but may not alter the strict certificate unless a later theorem explicitly changes this premise.
6. Synthetic proof-object tests do not imply real-flight calibration or hard real-time validity.

## 3. Canonical objects and optional extension

Let `x` denote the physical/certificate state, `g` the externally assigned goal, and `o_task = Omega(x,g)` the actor observation. The learned policy is `pi_theta(a | x,g)`. The frozen recovery controller is `kappa(x)`. The true correspondence is `Post(x,a)` and its verified outer approximation is `Post_hat(x,a)`. Endpoint membership does not imply safety of the current motion segment, so `A_safe` also requires complete current-step swept-tube containment.

The exact robust first-passage transit-energy object used by the canonical theorem is restricted to recovery paths whose command transitions remain covered through arrival:

```text
J_cov^kappa(x) = 0,                                         x in G_charge,
J_cov^kappa(x) = sup over covered joint one-step outcomes
                 [c_real(x,kappa(x),w) + J_cov^kappa(x')],  otherwise.
```

The versioned executable certificate uses the conservative selected-node recursion `E_nu >= round_up(d_bar_nu + max_{mu in Child(nu)} E_mu)` with finite nonnegative quantities. It is an upper certificate for `J_cov^kappa`, not an identity for the exact joint value and not a claim about behavior after coverage loss. Terminal reserve `e_G` is added outside it. Existence and finiteness require a nonnegative finite certified descent rank; SAC discounting is irrelevant.

Recovery rank is the level of the runtime-selected, hash-bound proof certificate. It is not `min{i:x in K_i}` because atlas cells overlap; every linked successor certificate must carry a strictly smaller level.

The recoverable set is

```text
R = {x : a current kappa proof chain exists,
         collision/geometry proof objects are valid,
         e_lower(x) >= E_bar^kappa(x) + e_G + m_e}.
```

Here `m_e` is a constant executable reserve. A state-dependent reserve requires
its successor increase to appear inside the recovery recursion and is outside
the canonical theorem.

Runtime acts on complete supports, so the canonical predecessor object is

```text
A_rec_set(x) = {C subset A_act(x) : Post_hat_C(x,C) subset R}.
```

`A_rec_set` is only a recoverability predecessor. Direct complete-support actuator, current swept-tube, successor collision/velocity, energy-prefix, and version predicates form `A_safe_set`; runtime verifies membership in that collection without a singleton-inheritance argument.

For the optional stronger normal-authority capability, take finite recoverable energy-augmented candidate cells `V_G`, verified positive-volume action sets `U_i`, and the terminal seed `G_charge`, and define

```text
Phi(S) = {i in V_G : exists nonempty verified U_i such that
                     Post_hat(K_i,U_i) subset union_{j in S} K_j union G_charge}.
```

The ideal normal-authority set is the greatest fixed point

```text
R_RL = union_{i in nu Phi} K_i,
nu Phi = intersection_{n >= 0} Phi^n(V_G).
```

For a finite atlas, descending iteration terminates in at most `|V_G|` strict removals. The current `_build_rl_authority_domain` is only a topology/cell-ID graph 1-core: it omits energy-augmented membership, full action-set verification, and an explicit terminal seed. It is therefore a candidate kernel, not an implementation of this state-level `Phi`.

Define

```text
A_cont(x) = {a : Post_hat(x,a) subset R_RL union G_charge}.
C_run(x) belongs to A_safe_set(x) and satisfies the optional A_cont full-support predicate.
```

In this optional stronger construction, while departure is closed,

```text
C_charge(x) is a directly verified complete support with
             Post_hat_charge(x,C_charge(x)) subset G_charge.
```

These optional definitions are mathematically conditional and are not implemented by the current topology-only kernel. The canonical theorem below requires departure into a named certified cell `K_h subset R`, not `R_RL`. The executable handoff additionally requires the complete successor lower-energy margin to exceed the normal-to-recovery switching margin and reads the realized successor back against the same `h` before changing mode. This establishes the immediate next normal-authority state without claiming the state-level fixed point.

## 4. Provisional guarantee hierarchy

The organizing lifecycle invariant is indexed by the unchanged pending task `q`:

```text
I_q = (R x {NORMAL,BACKUP}) union (G_charge x {CHARGING}).
```

`q` changes only on an explicit normal-mode task-completion transition. Backup, charging, and departure preserve it. `FAIL_CLOSED` is outside the positive guarantee. The unimplemented stronger `R_RL` is not a premise of the canonical safety theorem.

A covered normal flight has two exhaustive successor-mode subcases. `RUN-N`
keeps `NORMAL`. `RUN-ARRIVE` first completes the same verified physical `RUN`,
then uses a live successor-bound station certificate to perform a zero-duration
`NORMAL -> CHARGING` update. Its successor lies in `R intersection G_charge`,
and the flight interval receives no charging gain. Charging-source states never
fall back to `kappa`; an open gate admits only certified `DEPART`, and every
`kappa` source requires positive selected rank. A κ successor is classified by
its hash-declared committed child, never by minimum geometric overlap: a
positive-rank child remains `BACKUP`, while only the exact level-zero child plus
a fresh terminal/hold check enters `CHARGING`.

The charging gate is two-stage. The scalar lower-battery predicate is only a
prerequisite. The final `DEPART` handoff opens after the prepared full support
names a candidate normal-authority successor, proves complete successor
containment and a strict next-cycle normal energy margin, and the realized
successor is transactionally read back against that identity. If the scalar
predicate opens earlier, the hybrid gate remains closed and a certified
`CHARGE/HOLD` continues; SAC does not receive authority yet.

Coverage has two stages. `CmdCovered_t` binds the command certificate before
the physical interval. `HybridCovered_t` additionally binds any realized
RUN-arrival classifier or κ-child readback before publishing the returned
hybrid state. An invalid second stage ends hybrid coverage at `t`, even though
command coverage still proves the already executed physical interval.

Execution authority is determined by the command and complete certificate valid at atomic compare-and-publication, not by an earlier preview. Level-0 assumes a single non-reentrant controller; in-scope asynchronous certificate writers, hold execution, and hybrid mode commits share that transaction. Reentrant `step/reset`, unregistered writers, and hardware/bus atomicity remain external. In particular,

```text
actual source = task or kappa and covered_at_publication = true
    => record the matching RL_GENERATOR or KAPPA_BACKUP authority;
any preview and covered_at_publication = false
    => publish an explicitly uncertified emergency-brake command for bookkeeping,
       record FAIL_CLOSED, terminate, commit no recovery child, and bootstrap zero;
preview = FAIL_CLOSED
    => use the dedicated fail-closed path, never re-evaluate/commit a newly valid kappa.
```

The second branch is not a safety claim for the emergency brake. It prevents an uncovered command from inheriting the certified `kappa` label.

- Level 0: software semantic correctness.
- Level 1: synthetic certificate validity under declared simulator contracts.
- Level 2: conditional robust mathematical safety if every physical envelope and proof object is independently valid.
- Level 3: calibrated hardware certificate.
- Level 4: real-flight empirical performance.

Current repository evidence reaches Levels 0–1 for selected fixtures. Prior normalized and current-hash reviews accepted an earlier narrow theorem as a Level-2 conditional specification, but the explicit `RUN-ARRIVE` case and source/rank authority repairs changed its hashes; that verdict is stale until a fresh audit accepts the new relation. Any Level-2 judgment remains same-family/provisional. Levels 3–4 are not established.

## 5. Current strict-safety claim boundary

The intended claim is recursive recoverability and collision/energy preservation from a certified initial state under valid proof objects, verified execution support, timely authority switching, and a certified frozen recovery suffix. It is not a claim of arbitrary goal completion, learned-field correctness, SAC convergence, global unknown-environment safety, calibrated aircraft safety, or hard-WCET execution.

## 6. Excluded stronger claims and submission gates

1. A proposal-only dual-field core is implemented and unit-tested, but it has no trained/calibrated checkpoint or benefit evaluation; dual-field benefit remains a hypothesis.
2. Current atlas pruning is a candidate graph kernel, not the optional state-level `R_RL` fixed point; `R_RL` is excluded from the canonical safety theorem. The departure handoff commits one candidate-kernel successor only as a software identity/readback contract.
3. Current artifacts do not establish learned Generator residual benefit, autonomous policy charging, or teacher benefit.
4. The 2× synthetic descriptors are hash-bound and now pass a deterministic full-cell-domain numerical domination audit; this does not discharge aircraft energy calibration.
5. Full ICLR readiness requires a matched non-saturated Generator study and regenerated non-mixed-provenance ablation tables.
6. The broad persistent-task/energy/charging lifecycle is not novel: Persistification and limited-duration CBF work already establish closely overlapping augmented-state safety and charger-return ideas. Only the narrow robust-certificate/publication-authority/task-identity composition remains a candidate, and it currently lacks a separation result or decisive matched evidence.

The canonical theorem now makes selected-child recursion, current swept tubes, within-step energy, complete charge inclusion, `RUN-ARRIVE`, source/rank-restricted authority, publication-time dependency/time binding, and complete-step coverage explicit. Targeted regressions cover preview/recheck loss, stale pre-step versions and fingerprints, mid-cycle expiry, rank-zero/charging κ rejection, actual publication replay provenance, successor-observation alignment, charger hold, thin-gap swept motion, unsafe energy prefixes, and the lower-energy departure gate. The earlier 440-test result predates these repairs and is not current evidence; a fresh complete suite and duplicate 2× audit are pending. The last accepted 2× analytic artifact hash remains historical evidence only until that rerun; neither tests nor review establish physical calibration.

## 7. Non-claims

No theorem is accepted merely because a unit test passes, a finite rollout has no collision, a neural residual is small, or a synthetic manifest is internally consistent. The primary-source closest-work search invalidates broad lifecycle novelty; the surviving narrow contribution remains provisional rather than being promoted by omission of prior art.
