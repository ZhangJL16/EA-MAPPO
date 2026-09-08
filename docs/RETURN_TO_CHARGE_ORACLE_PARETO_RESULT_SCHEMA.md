# Return-to-Charge Oracle/Pareto Evidence Schema

## Scope and current verdict

This contract specifies the raw evidence needed to evaluate Oracle Decision
Headroom and the stranding--throughput Pareto frontier. It does not contain an
empirical result. R3 has completed and passed its disjoint-seed 100-run endurance
validation, so the Oracle Headroom Gate is authorized. The first optimized formal
run nevertheless stopped before producing a valid result because eleven of twelve
diagnostic Oracle shadows missed the declared 4000-step completion horizon. Under
the v7 contract this is represented as finite-deadline infeasibility, not as an
exception and not as evidence of infinite-horizon unreachability. Version 7
also separates the task--then--return stopping boundary from the direct-return
certificate after observed hybrid viability inversions invalidated the v6
branch maximum. The formal Oracle-headroom verdict remains `NOT_EVALUATED`.

The machine contract lives in
`experiments/energy_mc/oracle_pareto_records.py`. Its empty template is valid only
with `formal_eligible=false` and `claim_status=NOT_EVALUATED`.

## Statistical and pairing units

- Oracle-headroom uncertainty unit: one independently seeded battery cycle.
- Decision-mechanism unit: one pre-action decision event nested within a cycle.
- Dependence handling: decision events from the same cycle are not counted as
  independent cycles.
- Pairing: every compared method and Oracle-shadow path receives the same
  `paired_schedule_id`, task sequence, obstacle realization, initial state,
  policy, safety filter, and decision-check schedule. Task sampling is keyed by
  `(evaluation_seed, battery_cycle, task_index)`; merely resetting separate
  environments with the same RNG seed is insufficient because different return
  times otherwise change sampling positions and RNG consumption.
- Information rule: a method feature must be available at `information_time`,
  before its commit/continue action. Realized return Energy, terminal duration,
  final path length, and future intervention counts are labels or audit fields,
  never deployment features.

## Top-level bundle

| Field | Required content |
| --- | --- |
| `schema_version` | `return-to-charge-oracle-pareto-v7` |
| `evidence_mode` | `EXPLORATORY` or `FORMAL` |
| `claim_status` | Initially `NOT_EVALUATED`; promoted only by a passing Gate |
| `formal_eligible` | Boolean, false unless all upstream prerequisites pass |
| `provenance` | Code/config hashes, explicitly authorized navigation completion wrapper, loaded checkpoint SHA, chained calibration/validation hashes, reconstructed navigation environment kwargs, declared Stage-B runtime overrides, exact command |
| `protocol` | Methods, parameter grid, seeds, pairing schedule, common stranding ceiling, minimum cycles, throughput threshold |
| `gates` | Navigation, probability semantics, battery calibration, battery validation, Oracle headroom status/artifact |
| `oracle_shadow_coupling_audit` | Pre-first-disagreement equality audit against the independent same-schedule/same-reserve Oracle run |
| `simultaneous_stranding_audit` | Complete candidate family, Bonferroni allocation, and one-sided exact upper bound for every point |
| `stranding_certification_power_audit` | Design rate, target power, planned cycles, certifiable failure count, achieved power, and minimum powered sample size |
| `paired_throughput_interval` | Full-family directional paired max-t critical values and four pointwise rate-bound arrays, eligible subsets, selected points, and conservative gain lower/upper bounds |
| `decision_events` | Paired method/Oracle decisions at each pre-action check |
| `cycles` | Independent cycle outcomes and paired Oracle counterfactual |
| `method_summaries` | Cycle-level aggregate estimates and uncertainty |
| `theory_diagnostics` | Chart, witness, overlap, stopped-error, margin, and transience diagnostics |
| `pareto_frontier` | Nondominated points computed only from eligible method summaries |

## Per-decision event contract

Each event records identifiers; `information_time`; remaining/reserve Energy;
whether the Oracle shadow is active; exact Oracle requirements for return-now
and task-then-return while it is active; the corresponding estimated
requirements; the task--then--return effective stopping requirement; Oracle and method commit actions;
commitment state before/after; Oracle and method score margins; the first-
disagreement marker and direction; decision-check interval; threshold overshoot;
observed interval requirement drift; and feature provenance.

For goal (g), operational horizon (H), hitting time (T_g), and nonnegative
step cost (c_t), the deadline-completion resource is the extended-real object

\[
E_H(x,g)=
\begin{cases}
\sum_{t<T_g}c_t, & T_g\le H,\\
+\infty, & T_g>H.
\end{cases}
\]

Every exact component is therefore stored as a tagged pair: the finite numeric
payload plus `*_deadline_feasible`. When the tag is false, the pair denotes
(+\infty); the numeric payload is only the observed truncated-prefix Energy and
must not be interpreted as completed-goal Resource-to-Go. The exact effective
commitment requirement is

\[
R^\star_{t,H}=R^\star_{t,H,\mathrm{task\ then\ return}}.
\]

The direct-return branch separately labels a commitment as certified or
emergency. It is not an additional stopping boundary because goal switching
and the task-service reset make the two macro-action feasible sets non-nested.

With available Energy \(B_t\) and reserve \(b\), the Oracle score margin is

\[
M^\star_{t,H}=B_t-b-R^\star_{t,H}.
\]

Both exact and estimated records declare
`*_effective_requirement_semantics=task_then_return_stopping_boundary`. The JSON representation uses
`exact_effective_requirement_is_infinite=true` and a null numeric effective
requirement for (+\infty); it uses false plus a finite number for a finite
requirement. `None` is reserved for an Oracle shadow that was not evaluated and
is never interchangeable with either finite or infinite.

The first time the method and Oracle commit indicators differ is the stopped
first-disagreement event. For non-Oracle methods, exact shadow evaluation may
stop after recording that event; later method-path records carry
`oracle_shadow_active=false`, null exact requirements, and a null infinite tag,
and are outside the coupling estimand. The independent Oracle run
continues to the cycle outcome. Post-disagreement records must not inflate the
independent sample count.

## Per-cycle contract

Each cycle records method/parameter/seed/pair IDs; terminal reason; charger
return, stranding, and Energy-exhaustion indicators; tasks; simulated time;
tasks/hour; unused Energy at the charger; first-disagreement step/direction; and
the paired Oracle return, stranding, and task outcome.

Required aggregate columns are:

| Method | Parameter | Independent cycles | Stranding count | Stranding rate | Wilson 95% upper (descriptive) | Familywise exact upper (Gate) | Tasks/hour | Tasks/cycle | Mean unused charger Energy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Oracle | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| SOC | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Distance | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Learned methods | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

No `TBD` cell is evidence.

## Theory-diagnostic block

For each independently certified chart and risk level, retain:

- chart ID, partition/code map, structure/calibration/evaluation folds;
- bounded witness dictionary, cell counts, empirical radius and upper radius;
- pushed source and target masses \(p_j,q_j\), and overlap
  \(C_h=\sum_j q_j^2/p_j\);
- risk tilt, clipping, effective sample size, and fresh unique trajectory/cycle
  count;
- stopped \(L_p\) score error, margin exponent/radius, first-disagreement rate,
  and boundary-weighted loss;
- transience or resolvent gap when it is defined by the fitted mathematical
  object.

These diagnostics may explain a Pareto result but cannot replace cycle-level
stranding and throughput evidence.

## Gate logic

1. Fixed-platform identity, probability semantics, battery calibration, and
   battery validation must each have the contract-appropriate passing status and
   a named artifact before `formal_eligible=true`. For the R3 conditional route,
   historical navigation performance is descriptive while checkpoint/config
   identity, policy freezing, complete failure taxonomy, and endurance remain
   fail-closed.
2. Oracle Headroom uses a power-audited independent-cycle count; the default
   12-point grid preregisters 320 cycles per point at a 1% design rate. Each of
   the two required safe families has at least 95% certification power, giving
   a dependence-robust joint lower bound above 90%. Eligibility uses simultaneous one-sided
   Clopper--Pearson upper bounds with Bonferroni familywise confidence 0.95,
   not pointwise Wilson bounds. The exact power audit must have
   `status=PASS_DESIGN_POWER`; zero-event evaluability alone is insufficient.
   For throughput, every paired bootstrap replicate resamples cycle IDs jointly
   and constructs directional studentized max-t rate bands over the full
   candidate family. Selected safe Oracle lower rates and heuristic upper rates
   define the conservative gain lower bound; Oracle upper rates and heuristic
   lower rates define the gain upper bound. PASS requires the point estimate and
   lower bound above 5%; `FAIL_INSUFFICIENT_ORACLE_HEADROOM` requires the upper
   bound below 5%; all threshold crossings are
   `INCONCLUSIVE_ORACLE_HEADROOM`. Defaults are 10,000 replicates and seed
   20260830; a selected-max percentile interval is diagnostic only.
3. `ORACLE_HEADROOM_GATE_PASS` additionally requires the Oracle headroom record
   itself to pass.
4. `PARETO_EVIDENCE_READY` requires Oracle PASS, nonempty eligible frontier
   points, and the paired decision/cycle records.
5. Missing records, a nonpositive heuristic lower-rate denominator, emergency
   guards, future-derived deployment features, or a failed prerequisite leave
   the claim at `NOT_EVALUATED`, `INCONCLUSIVE_ORACLE_HEADROOM`, or a named
   failure; they are not silently imputed.
6. For every method/schedule/reserve group, the independent Oracle run must
   contain the same pre-action events through the first disagreement. Position,
   velocity, task goal, remaining Energy, tagged extended-real exact effective
   requirement, both deadline-feasibility tags, and the Oracle commit indicator
   must agree within the declared numerical tolerance.
   Post-disagreement trajectories are not compared. Any mismatch invalidates
   formal eligibility even if aggregate headroom statistics would otherwise
   pass.

The Stage-B smoke for this contract generated 369 decision events across five
cycles, recorded three first disagreements, and joined every cycle to its
same-schedule/same-reserve Oracle result. The bundle remained
`SMOKE_ONLY_NOT_FORMAL_EVIDENCE` with `formal_eligible=false`; these counts test
the executable critical path and are not paper results.

The coupling smoke compared 214 decision events across three method/schedule
groups through three stopped first disagreements. Maximum position and exact-
requirement errors were both zero. This validates the audit implementation only;
the bundle remained `formal_eligible=false`.
