> **LEGACY / SUPERSEDED:** Historical research provenance only. This file is not an active method, theorem, implementation, test, or README authority. The active route is in docs/new_theory/.

# Canonical Regenerative-Recoverability Proof Skeleton

Audit target: `PAPER_THEOREM.md`  
Prior Round-5 review result: `PASS`, same-family and provisional. A Round-6 current-hash blind audit also returned `PASS` with two cosmetic provenance/dependency-wording findings; final hash closure follows those edits.  
Scope: the narrow complete-step covered conditional theorem, not `R_RL`, task completion, learned-field accuracy, or hardware calibration.

## 1. Typed symbol table

| Symbol | Type and domain | Meaning / fixed dependence |
| --- | --- | --- |
| `x` | element of certificate state space `X_cert` | Physical state plus uncertainty, evidence versions, proof-object identity, and timing state |
| `q` | element of pending-task identity set `Q` | Exogenous task identifier; not an input to strict certificate construction |
| `m` | `{NORMAL, BACKUP, CHARGING}` | Covered hybrid mode; `RUN-ARRIVE` and `DEPART` include zero-duration successor-mode updates, while `FAIL_CLOSED` exits the positive theorem |
| `a` | element of physical action space `A subset R^3` | Published and realized covered action |
| `Post(x,A_0)` | subset of `X_cert` | True robust one-step successor correspondence for all `a in A_0` and admissible disturbances |
| `Post_hat(x,A_0)` | subset of `X_cert` | Verified outer successor with `Post subset Post_hat` |
| `Tube_hat(x,A_0)` | set of physical swept points | Complete current-step tube for every `a in A_0` |
| `EnergyTube_hat(x,A_0)` | interval-valued trajectory bound | Certified lower battery energy throughout the complete command interval |
| `Verify_t(A,Xi)` | finite Boolean schema | Pre-command check of certificate record `Xi` bound to action set `A`; includes hash/version/validity/inclusion/compare-and-publication binding and is not a self-proving soundness predicate |
| `CmdCovered_t` / `HybridCovered_t` | pathwise Booleans | The first proves the physical interval from a pre-command certificate; the second additionally requires any realized-successor classifier/committed-child check before the returned hybrid state is published |
| `kappa` | fixed map `X_cert -> A` within one certificate epoch | Independently certified recovery controller |
| `r:V->N_0` | nonnegative integer rank map | Level of the runtime-selected hash-bound recovery node, not minimum atlas membership rank |
| `J_cov^kappa(x)` | `[0,+infinity]` | Exact joint robust undiscounted first-passage energy over recovery paths covered through arrival |
| `d_bar_nu` | finite scalar in `[0,+infinity)` | Verified one-step recovery-consumption upper at selected node `nu` |
| `E_nu` | finite scalar in `[0,+infinity)` | Stored outward-rounded recovery-energy supersolution at selected node `nu` |
| `e^-(x)` | energy units | Certified lower battery energy |
| `e_G,m_e` | nonnegative energy units | Terminal energy and constant executable reserve |
| `R` | subset of `X_cert` | States with valid recovery/geometry proof and enough lower energy for the selected `E_nu` plus reserve |
| `A_rec_set(x)` | collection of action sets | Complete supports whose directly verified full-set successor lies in `R` |
| `A_safe_set(x)` | collection of action sets | Full-support actuator, tube, collision, velocity, energy-prefix, version, and predecessor certificates |
| `C_run(x)` | zonotope subset of `A` | Complete learned continuous support verified inside `A_safe` |
| `G_ch` | subset of `X_cert` | Charge-admissible position/velocity/energy/evidence set |
| `Post_hat_charge` | subset of `X_cert` | Sound hybrid flight-cost-plus-charge outer successor |
| `C_charge,H` | subsets of `A` | Verified closed-gate charging support or certified hold |
| `C_dep` | subset of `A` | Departure support satisfying scalar reserve and complete predecessor checks |
| `ArrivalClass_t(x+)` | `{NORMAL,ARRIVE,INVALID}` | Successor-snapshot-bound classifier; `ARRIVE` proves `G_ch` and valid hold, `INVALID` stops the covered relation, and no case causes same-step charge |
| `tau_cov(omega)` | pathwise complete-step proof horizon | First actually uncovered existing transition, or execution length if none; not a controller stopping time claim |

## 2. Dependency DAG

```text
D0 typed state/action/task/mode spaces and primitive Verify_t schema
  -> A1 sound Post_hat, Tube_hat, and EnergyTube_hat on complete covered intervals
  -> A2 frozen selected kappa certificate with exhaustive committed-child linkage and strict descent
       -> A3 one-step recovery cost upper and finite-DAG supersolution E^kappa
            -> L1 J_cov^kappa <= E^kappa and finite covered kappa hitting time
            -> D1 recoverable set R
                 -> D2 A_rec_set and A_safe_set
                      -> L2 verified normal/departure successor remains in R
  -> A4 complete charge certificate: actuator + collision tube + energy prefix + Post_hat_charge into G_ch
       -> L3 verified charging successor remains in G_ch
  -> A5 exhaustive RUN-N/RUN-ARRIVE/KAPPA/CHARGE/HOLD/DEPART task/mode transition rule
       -> L4 q is preserved through backup/charging/departure
L1 + L2 + L3 + L4
  -> T1 complete-step covered invariant family I(q) and regenerative recoverability theorem
```

`R_RL`, learned fields, SAC optimization, and empirical policy performance are not ancestors of `T1`. No edge points from `C_run`, a learned predictor, or `R` back into construction of the frozen recovery proof.

## 3. Canonical quantified statements

### QS-1: sound outer sets

For every existing `t < tau_cov(omega)`, every compatible source state `x_t`, and every action set `A_t` whose certificate was checked before publication for all of `[t,t+1]`,

```text
Post(x_t,A_t) subset Post_hat(x_t,A_t),
realized_current_tube(x_t,A_t) subset Tube_hat(x_t,A_t).
lower_energy_path(x_t,A_t) >= EnergyTube_hat(x_t,A_t) >= 0.
Tube_hat_Xi(x_t,A_t) subset FREE_Xi subset true obstacle-free space.
```

This is an assumption requiring calibration; it is not inferred from unit tests.

### QS-2: recovery supersolution

For each selected node, `r(nu) in N_0`; rank zero is exactly a childless zero-energy subset of `G_ch`. For each positive-rank node, the child set is nonempty, every robust successor is mapped to a committed hash-bound child `mu` with `r(mu)<r(nu)`, one-step realized energy is at most `d_bar_nu`, and

```text
E_nu >= round_up(d_bar_nu + max_{mu in Child(nu)} E_mu),
r(nu)=0 <=> Child(nu)=empty, E_nu=0, and K_nu subset G_ch.
```

The exact joint cost on paths covered through arrival satisfies `J_cov^kappa(x) <= E_nu`; no post-coverage physical cost claim is made.

### QS-3: normal predecessor

For every `x in R` and every directly verified complete support `C_run in A_safe_set(x)`,

```text
Post(x,C_run) subset Post_hat_Crun(x,C_run) subset R,
Tube(x,C_run) subset Tube_hat_Crun(x,C_run) subset FREE_C.
```

### QS-4: charging successor

For every `x in G_ch` under a closed departure gate and every covered `a in C_charge(x)` or certified hold `a in H(x)`,

```text
Post_charge(x,a) subset Post_hat_charge(x,a) subset G_ch.
Tube_charge(x,a) subset Tube_hat_charge(x,a) subset FREE_C.
lower_energy_path_charge(x,a) >= EnergyTube_hat_charge(x,a) >= 0.
```

### QS-5: complete-step covered lifecycle

For `Next:Q->Q`, every `q_0`, compatible initial `(x_0,m_0,q_0) in I(q_0)`, maximal physical execution `omega`, and admissible disturbance path obeying the hybrid-covered relation at existing indices before `tau_cov(omega)`, each such transition enters `I(q_{t+1})`. A `RUN-N` successor remains normal; a successor-certified `RUN-ARRIVE` enters charging without same-step charge. An `INVALID` successor ends hybrid coverage at that index although `CmdCovered_t` still proves the physical RUN interval. Task identity is unchanged except on an explicit completed `RUN-N` or `RUN-ARRIVE`, which uses `Next(q_t)`.

## 4. Assumption discharge ledger

| ID | Assumption | Used by | Current discharge |
| --- | --- | --- | --- |
| PA-1 | `Post_hat`, `Tube_hat`, and `EnergyTube_hat` are sound for complete action sets on the entire covered interval | MC-1, MC-2, MC-6, MC-8 | Declared physical premise; synthetic construction/tests only at guarantee Levels 0-1 |
| PA-2 | FREE/current-tube, actuator, velocity, version predicates are sound | MC-2, MC-6 | Declared physical premise; current D10 includes swept tube |
| PA-3 | Selected recovery certificate is current and hash-bound to live versions, dependency fingerprints, validity time, and a positive rank before any `KAPPA`; every robust successor is assigned a committed lower-rank child; rank zero is a zero-energy subset of `G_ch` | MC-3, MC-4, MC-7 | Canonical physical premise; runtime compares construction/live bindings and publication timestamp, while committed-child/lower-rank traces are exercised by regressions |
| PA-4 | One-step recovery energy upper dominates realized cost and lower-energy propagation | MC-4, MC-5 | Symbolically required; the three synthetic 2x atlases pass the full-domain deterministic audit, while physical calibration remains external |
| PA-5 | Reserve is constant or explicitly propagated | MC-5 | Canonical theorem selects constant executable reserve; general state-dependent reserve theorem is not claimed |
| PA-6 | Complete charge/hold schema covers actuator, collision tube, nonnegative energy prefix, flight cost plus charge gain, and successor inclusion in `G_ch` | MC-8 | Definition/premise; physical calibration remains external; software executes physical hold then applies charge |
| PA-7 | Departure complete support lies in `R` and scalar reserve gate holds | MC-9 | Runtime verifier/gate and tests provide software evidence; physical outer-set soundness is PA-1 |
| PA-8 | Mode/task update changes `q` only on explicit `RUN` completion; `RUN-ARRIVE` requires a fresh `ARRIVE` classifier result, is zero-duration, and receives no same-step charge; `INVALID` stops coverage | MC-10 to MC-13 | Manager/wrapper tests cover voluntary arrival, arrival-time drift, backup, charging, and departure; theorem treats the task/mode rule as independent |
| PA-9 | In the Level-0 single-controller/non-reentrant model, command coverage is established before publication for the whole physical interval; hybrid coverage additionally binds any postinterval classifier/child before hybrid-state publication. Snapshot reads are pure and asynchronous in-scope writers share the transaction; reentrant calls and hardware atomicity remain external | all dynamic claims | Explicit two-stage `tau_cov` boundary; runtime regressions cover dependency replacement, expiry, arrival drift, and active-child drift |

## 5. Micro-claim inventory

| ID | Context | Goal | Rule / required side conditions | Current status |
| --- | --- | --- | --- | --- |
| MC-1 | `C in A_rec_set(x)` and PA-1 | `Post(x,C) subset R` | Direct full-set verification plus subset transitivity | Discharged conditionally |
| MC-2 | `C in A_safe_set(x)` and PA-1/PA-2 | Every realized member has collision/actuator/velocity/energy-prefix safety | Complete-set predicates and outer containment | Discharged conditionally |
| MC-3 | Selected node `nu` with `r(nu)>0` and PA-3 | Every linked successor has selected child `mu` with `r(mu)<r(nu)` | Certificate link invariant, not atlas-overlap minimum | Assumed physically and exercised in runtime regression |
| MC-4 | MC-3 and supersolution QS-2 | `J_cov^kappa(x) <= E_nu` | Finite induction on `N_0` rank for paths covered through arrival | Discharged conditionally |
| MC-5 | `e^-(x)>=E_nu+e_G+m_e`, PA-4/PA-5 | Successor lower energy covers `E_mu+e_G+m_e` | Three-line lower-energy inequality using `d_bar_nu` | Discharged conditionally; blind check passed |
| MC-6 | MC-2/MC-3 | Each backup step is collision safe and energy feasible throughout the interval | Chain certificate includes current tube, energy prefix, and linked successor predicates | Discharged conditionally; blind check passed |
| MC-7 | Strict integer level descent | Kappa hits level zero in at most initial level | Well-founded induction on nonnegative integers | Discharged conditionally |
| MC-8 | `x in G_ch`, PA-6, verified hold/support | Charging interval is safe and successor lies in `G_ch` | Complete actuator/tube/energy-prefix/hybrid-successor schema | Discharged conditionally; blind check passed |
| MC-9 | Departure gate, PA-1/PA-7 | Departure successor lies in `R` | Complete predecessor inclusion; scalar energy alone is insufficient | Discharged conditionally |
| MC-10 | Normal accepted action or normal refusal | `RUN-N` enters normal; verified `RUN-ARRIVE` enters charging with no same-step charge; positive-rank-source `KAPPA` follows its exact committed child; `q` changes only on completed `RUN` | Exhaustive authority/task transition rule plus two-stage successor classifier/readback | Discharged conditionally; fresh blind recheck required |
| MC-11 | Covered κ with declared child `mu` | Positive `r(mu)` remains backup even under terminal overlap; exact `r(mu)=0` plus fresh hold enters charging; `q` unchanged | Snapshot-bound child id/rank/hash, realized membership, and atomic mode commit | Repaired by exact executable counterexamples; fresh blind recheck required |
| MC-12 | Closed charging | Mode remains charging; `q` unchanged | Charge-set inclusion plus task rule | Discharged conditionally |
| MC-13 | Certified departure | Mode becomes normal; `q` unchanged | Departure reset and task rule | Discharged conditionally |
| MC-14 | `FAIL_CLOSED` or invalid premise | No positive-theorem continuation is asserted | Stopping-scope rule | Discharged as a non-claim, not a safety action proof |
| MC-15 | MC-1 to MC-14 | `I(q)` family invariant for existing transitions before `tau_cov(omega)` | Exhaustive case induction over `RUN-N/RUN-ARRIVE/KAPPA/CHARGE/HOLD/DEPART` | Repaired; fresh blind recheck required |

## 6. Counterexample suite

| Candidate | Failure exposed | Canonical disposition |
| --- | --- | --- |
| Endpoint step from `-1` to `1` through obstacle at `0` | Endpoint successor safety does not imply current-step safety | `A_current_tube` required |
| True energy `0.25-1=-0.75` but envelope clipped at zero | Outer successor unsound | Negative successor retained |
| State-dependent reserve rises by one at successor | Old energy induction fails | Canonical theorem uses constant executable reserve |
| Zero-cost nonterminal self-loop | Bellman equality finite but no arrival | Non-hitting paths have infinite `J`; strict finite selected-level descent required |
| Paired outcomes `(10,G)` and `(0,h)` with `E(h)=100` | Exact joint value differs from separated `d_bar_nu+sup E` | `J_cov^kappa` and stored upper `E_nu` separated |
| Overlapping cells with minimum membership rank | Nominal certificate level need not decrease minimum rank | Runtime-selected certificate level is the induction measure |
| Full battery but departure support exits `R` | Scalar energy gate does not certify departure | Complete `Post_hat(C_dep) subset R` required |
| Position/velocity hold safe but energy falls below terminal minimum | Geometric station hold is not charge-set invariant | `Post_hat_charge` includes lower energy |
| Scheduler changes task at station | Safety authority does not imply same-task resume | Explicit `q+=q` rule required |
| Arbitrary supersolution `E_1=2`, `d_bar=1`, terminal `E_0=0` | Supersolution does not imply `E_i<=i c_max` | Upper-rank-times-cost claim removed |
| Charger endpoint `e+=1` after path dips below zero and later charges | Endpoint charge inclusion does not prove prefix feasibility | Complete `EnergyTube_hat_charge>=0` required |
| Certificate expires halfway through a published command | State-index expiry does not protect the interval | Full-interval coverage required before publication |
| Initial `x in R\\G_ch` with mode `CHARGING` | Unqualified product invariant admits incompatible states | Use mode-compatible domains `I_N`, `I_B`, `I_C` |
| Accepted normal flight ends inside certified `G_ch` but runtime returns `CHARGING` | Old `RUN` case required every successor mode to remain normal | Add successor-bound `RUN-ARRIVE`; prove `x+ in R intersection G_ch`, zero-duration mode update, and no same-step charge |
| Positive-rank κ child also overlaps the charging terminal | Minimum-overlap/terminal shortcut erases the selected child and breaks rank induction | Bind active child id/rank/hash; preserve positive child and allow KAPPA-ARRIVE only at the exact declared level-zero child |
| Postinterval arrival/child check succeeds, then dependency or active identity changes before mode publication | Separate check and mode write do not establish a covered hybrid successor | Use one transaction and define hybrid coverage separately from physical command coverage |
| Open-gate charging state lacks a departure Generator | Generic fallback produced an unmodelled `I_C -> KAPPA` charging step | Charging sources admit only closed-gate `CHARGE/HOLD` or open-gate `DEPART`; otherwise the positive prefix ends fail-closed |
| Rank-zero normal state refuses the Generator | Generic fallback fabricated a terminal `kappa` action | Require positive selected rank for every `KAPPA` source; rank zero has `RUN` or stopped-prefix semantics only |

## 7. Internal proof-gate disposition

The Round-5 normalized blind review found zero FATAL, CRITICAL, MAJOR, or MINOR issues and completed the counterexample pass on its recorded hashes. The Round-6 current-hash audit found zero FATAL, CRITICAL, or MAJOR issues and two cosmetic documentation issues, which are repaired here before final hash closure. The mathematical gate passes conditionally on PA-1 through PA-9; cross-family semantic review, physical calibration, complete-interval publication evidence, and hardware qualification remain outside this same-family provisional judgment.
