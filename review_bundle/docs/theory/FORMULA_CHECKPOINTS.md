> **LEGACY / SUPERSEDED:** Historical research provenance only. This file is not an active method, theorem, implementation, test, or README authority. The active route is in docs/new_theory/.

# Formula-Derivation Checkpoints

## C1 — Recovery energy and recoverability

- Identity: exact joint robust first-passage `J_cov^kappa` uses a supremum over paired `(cost,successor)` outcomes on recovery paths covered through arrival.
- Conservative proposition: for every positive-rank selected node `nu`, finite nonnegative `d_bar_nu` and `E_nu` satisfying `E_nu >= round_up(d_bar_nu + max_{mu in Child(nu)} E_mu)` upper-bound `J_cov^kappa` under nonnegative finite descent and complete recovery coverage; no post-coverage physical bound is claimed.
- Certificate predicate: `e^-(x) >= E_nu + e_G + m_e` for the current selected node `nu`.
- Refuted shortcut: SAC discount contraction cannot prove finite recovery or finite undiscounted energy.
- Complete one-step action verification now exposes two separate predicates. For the synthetic straight-segment plant contract, the componentwise hull of the uncertain initial-position box and robust endpoint box contains every swept segment. After physical-footprint/geometry inflation, the complete hull—not merely its endpoints—must lie in the certified FREE union. Since every calibrated flight-energy term is nonnegative, the step-prefix lower battery bound is `min(e^-_t,e^-_{t+1})=e^-_{t+1}`; it is stored and checked separately even though the successor reserve predicate is stronger on valid fixtures.

## C2 — Affine-tanh Generator-SAC

For invertible `G(x)` and `u ~ N(mu_theta,sigma_theta)`,

```text
a = c(x) + G(x)tanh(u),
log pi_A(a|x,g) = log N(u;mu,sigma)
                  - sum_i log(1-tanh(u_i)^2)
                  - log|det G(x)|.
```

The map is a diffeomorphism from `R^3` to the zonotope interior; the boundary has zero probability. Actor and Bellman terms therefore use `log pi_A`. The primary default tunes temperature with `log pi_eta`, which is equivalent at each state to a physical-coordinate target shifted by `log|det G(x)|` because `log pi_A + (H_target + log|det G|) = log pi_eta + H_target`. This makes the alpha residual invariant to action-unit or uniform support scaling while retaining the physical density in the soft value. The alternative constant physical-coordinate target remains available only as an explicit ablation. No convergence theorem is claimed for changing certificates.

The authority law is a mixed measure:

```text
Pi = beta(x) Pi_cont + (1-beta(x)) delta_kappa.
```

It has no single Lebesgue density. The deterministic `kappa`/hold atom receives no continuous differential-entropy term. `FAIL_CLOSED` has no bootstrap continuation. A frozen bounded-reward soft Bellman operator on the fully augmented Markov certificate state is a `gamma` contraction for `gamma<1`; the current neural observation is not proved sufficient for that state. The implemented result is therefore branch-target semantic closure, not an observation-level contraction or learning-convergence theorem.

A Gymnasium time-limit truncation is a collector boundary, not the `FAIL_CLOSED` authority state. When `bootstrap_on_truncation=True`, the transition must therefore retain the real post-step certificate epoch, authority, and corresponding continuous `c,G` or atomic authority action before the environment resets. Only a true terminal transition has no successor authority context. The formal training entry point now inherits the normalized-coordinate temperature default from `GeneratorSACConfig`; physical-coordinate temperature remains an explicit ablation rather than an accidental CLI default.

## C3 — Hybrid charging and departure

For a physically executed certified hold `h(x)`, the charging successor is

```text
(p+,v+) = f_kin(x,h(x),w),
e+ = min(e_max, e - d_E(x,h(x),w) + r(x,w) Delta t).
```

An outer `Post_hat_charge` must include tracking, geometry/evidence updates, cost upper, and charge lower/upper. Station invariance requires its complete image to lie in `G_charge`; `r Delta t >= 0` alone is insufficient. A lower-energy condition is

```text
e^- - d_bar_hold + r_lower Delta t >= e_G + m_e.
```

Departure uses a scalar prerequisite and a distinct final authority-handoff gate:

```text
e^- >= E_dep_bar + e_G + m_dep,
h_N(C_dep) in V_N,
Post_hat(x,C_dep(x)) subset K_{h_N} subset R,
inf_{x+ in Post_hat} [e^-(x+) - E_kappa(h_N)] > m_sw.
```

The realized successor and current dependency snapshot are read back against
the same `h_N` before `CHARGING -> NORMAL` is committed. If the scalar
inequality is true but the remaining conjuncts are false, the final departure
gate is still closed and only a verified hold/charge transition is covered.

Here `V_N` is the implemented finite candidate normal-authority kernel.
Targeting the state-level energy-augmented `R_RL` instead is the optional
stronger condition for persistent positive-volume normal authority and is not
claimed by the current implementation.

The energy inequality is necessary, not sufficient. Same-task resume follows a separate `q+=q` mode-transition rule.

## C4 — Dual proposal fields

`B_theta(x,a)` is local/action-conditioned with a collision/unknown boundary and lower-margin calibration. `E_psi(x)` is global under frozen `kappa`, has a charging first-passage boundary, and needs upper calibration. Proposition H0 gives an explicit three-proposal ordering that no scalar bottleneck with strictly monotone readouts can preserve for both tasks. Proposition H1 gives `|Ehat_nu-E*_nu| <= epsilon_0+r(nu)epsilon` for a finite selected DAG with uniform rectangular-recursion residual. These results justify distinct outputs/calibration budgets, not necessarily separate encoders or networks; the performance advantage remains an empirical hypothesis.

## C5 — Recovery teacher and off-policy SAC

Every teacher tuple produced by the learner environment records a real transition instance only if it contains the clean observation, executed action, reward, successor, and termination semantics. Privileged atlas data may choose behavior or labels but may not leak into the learner observation or alter the transition kernel. The present buffer retains only complete successful cycles, so it is explicitly **outcome-conditioned initialization data**. In a stochastic MDP, future-success filtering can change the conditional successor law given `(s,a)` and must not be described as an unbiased Bellman sample unless transitions are deterministic or the selection is proved outcome-independent. Replay prefill changes the finite sample distribution; actor imitation changes initialization. Neither implies asymptotic improvement or charging competence. The prefill-only, warm-start-only, both, and neither factorial is required to identify the mechanism.

## C6 — Publication-time authority and fail-closed race

Let `D_t` be the preview authority decision, `S_t^pub` the actual command source, and `C_t^pub` the final certificate-coverage bit at atomic publication. Recorded authority and proof-state commitment are functions of the actual published command, not `D_t` alone:

```text
D_t = KAPPA_BACKUP and C_t^pub = 1  => publish kappa, record KAPPA_BACKUP;
D_t = KAPPA_BACKUP and C_t^pub = 0  => publish emergency brake for bookkeeping,
                                       record FAIL_CLOSED, terminate, bootstrap 0.
D_t = RL_GENERATOR and C_t^pub = 0   => the same FAIL_CLOSED branch;
D_t = FAIL_CLOSED                    => dedicated emergency path, no recovery re-evaluation;
commit_child                         <=> C_t^pub = 1 and
                                         S_t^pub in {task,kappa}.
```

The uncovered lines are not safety theorems for the emergency brake. They prevent an uncovered action from being relabelled as certified `kappa`, prevent a stale proof-child update, and prevent a fail-closed preview from being “rescued” by a later certificate. A version change after the branch's publication snapshot is selected forces `C_t^pub=0`; rejecting only the task bundle is insufficient if the fallback certificate is stale. The final freshness observation must rebuild the full certificate state, because a bound `prepared_state.snapshot` callback sees referenced geometry mutations but can miss copied calibration or kappa versions. Publisher stage conflict is itself uncovered: a pre-staged emergency or mismatched epoch/action cannot be promoted to certified `kappa` before freshness is checked. Forced KAPPA/RL/nominal/hold/stage-conflict races plus parameterized bound-version mutations verify authority mapping, κ-metric exclusion, noncommitment, pre-motion hold rejection, and the zero-bootstrap Bellman branch.

Snapshot equality is necessary but not sufficient when a dependency changes
before every snapshot.  Therefore the selected recovery cell must also satisfy

```text
cell_versions = construction_versions = live_versions,
cell_fingerprints = construction_fingerprints = live_fingerprints,
t_pub <= expiry(cell).
```

Validity time is included in publication-snapshot equality, but not in the SAC
replay-group epoch, so ordinary clock advance does not clear replay.

## C7 — Exhaustive hybrid source and arrival relation

Let `r_t` be the selected recovery rank and `m_t` the source mode.  Certified
`kappa` authority is permitted only when

```text
m_t in {NORMAL, BACKUP} and r_t > 0.
```

For `m_t=CHARGING`, a closed departure gate admits only a verified
`CHARGE/HOLD`; an open gate admits only verified `DEPART`.  Missing support ends
the covered prefix fail-closed rather than selecting κ.  A covered normal
action has two successor-mode subcases:

```text
RUN-N:      x+ in R, ArrivalClass_t(x+)=NORMAL,  m+ = NORMAL;
RUN-ARRIVE: x+ in R intersection G_charge,
            ArrivalClass_t(x+)=ARRIVE,            m+ = CHARGING;
INVALID:    dependency drift/check failure,       covered prefix ends.
```

`RUN-ARRIVE` is the composition of the already covered physical RUN interval
and a zero-duration mode update.  It adds no same-step charging energy and uses
the ordinary RUN task-completion rule.  This partitions the executable hybrid
branches without inventing a second physical action.

For a covered κ command at node `nu`, the postinterval transaction must read
back the exact hash-declared committed child `mu=sigma_nu(x+)`:

```text
r(mu)>0: m+=BACKUP and active_child+=mu, even if x+ also overlaps G_charge;
r(mu)=0: verify the terminal/hold certificate under the same snapshot,
         then atomically set m+=CHARGING with no same-step charging gain;
otherwise: HybridCovered_t=0.
```

Thus `CmdCovered_t` proves the physical command interval, while
`HybridCovered_t` additionally proves the realized-successor classifier or
committed-child/mode update before the returned hybrid state is published.
