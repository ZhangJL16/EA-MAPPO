# Experimental Falsification of UAV Theory Candidates

## Scope and gate

These experiments test theorem-relevant failure modes. They are not a formal
500k training result. The frozen navigation SAC and energy estimator were not
updated. The protocol stopped after the mandatory randomized-state and 1000+
adversarial-trajectory gates because the proposed two-stage method failed both
navigation and real-time criteria; proceeding to 10k/50k comparisons would not
repair the theorem.

## Provenance

- Frozen SAC:
  `artifacts/uav_energy_delivery_v3_formal_20260816_004619/phase1_navigation/checkpoint_transition_500000.zip`
- Symbolic and randomized validation:
  `artifacts/uav_theory_candidate_validation_20260819/report_100k_all_final.json`
- Adversarial trajectory artifact:
  `artifacts/uav_theory_adversarial_1002_20260819/`
- Physical hard-state search:
  `artifacts/uav_hard_feasibility_set_20260819_v3/`
- Scenario seed: `20260819`
- Scenario count: 334
- Method trajectories: 1002
- Families: sparse, medium, dense, narrow, long, vertical
- Device: CPU with 8 independent workers
- Formal 500k: not started

## Symbolic checks

SymPy verified:

1. the second-order sphere HOCBF is affine in acceleration;
2. the ZOH sphere barrier has degree four and its derivative degree three;
3. the exact derivative identity;
4. the one-step smooth-value quadratic has Hessian
   `L_E (T^2+T^4/4) I`, hence is positive semidefinite for `L_E>=0`.

## Randomized validation

| Check | Cases | Result |
| --- | ---: | --- |
| Cylindrical support function | 100,000 | max absolute error `1.42e-14`; all constructed maximizers inside input set |
| Exact ZOH sphere minimum | 100,000 | exact minimum never exceeded random sampled minimum; max interior stationarity residual `3.41e-13` |
| Endpoint-safe / inter-sample-unsafe search | 100,000 | 74 counterexamples found |
| Joint primal-dual margin | 100,000 | 0 primal certificate failures, 0 dual failures, 0 sign mismatches; max gap `9.97e-8` |
| Smooth quadratic value bound | 100,000 | 0 violations for the exactly smooth unit-quadratic test |

Joint-margin timing over 100,000 random cases:

- mean: `2.994 ms`;
- P95: `7.081 ms`;
- P99: `32.456 ms`;
- maximum: `393.650 ms`;
- 50 ms deadline misses: `19/100000`;
- validation-only global certificate fallbacks: `3/100000`.

The typical timing is compatible with 20 Hz, but the tail explicitly prevents a
hard real-time guarantee. The KKT dual-recovery LP is necessary because the
direct nonsmooth dual SLSQP occasionally stops at a nearby nonoptimal point; a
rare global simplex search was used only to validate the mathematical identity,
not as a deployable 20 Hz mechanism.

## Physical `HARD_FEASIBILITY_SET`

A separate 100,000-state search used the actual 4 km by 4 km by 400 m UAV
limits (`20 m/s` horizontal, `5 m/s` vertical, `5 m/s^2` horizontal, and
`3 m/s^2` vertical) and the repository's sphere HOCBF rows.  Seven targeted
families covered high closing speed, low clearance, moving multi-obstacle
conflicts, narrow passages, horizontal and vertical velocity saturation, and
sample delays from `0.05` to `0.20 s`.  Every sampled position was geometrically
outside its obstacle primitives.

| Hard-state result | Count |
| --- | ---: |
| Continuous HOCBF action set infeasible | `53,621 / 100,000` |
| Sample-strengthened action set infeasible | `58,059 / 100,000` |
| Continuous feasible but sampled infeasible | `4,438 / 100,000` |
| `h>=0` and `psi1>=0`, yet sampled action set infeasible | `10,225 / 100,000` |
| Hardest states retained with full physical provenance | `1,000` |

The retained set has sampled margins from `-2570.96` to `-1489.19
m^2/s^2`.  It therefore reproduces the *failure type* behind the historical
fallbacks on intentionally same-or-harder physical states.  Exact replay of the
original 34 Candidate-A fallback states is impossible because that older
artifact saved aggregate counts but not per-step position, velocity, and
obstacle rows; this provenance limitation is recorded in the hard-set summary
rather than silently approximated.

The exact method audit reports `0/1000` certified actions for standard HOCBF,
Candidate A, and the lexicographic candidate on the retained set.  This is not
three separate solver failures: all three methods share the same sampled hard
feasible set, and an energy objective cannot create an action when its exact
joint margin is negative.  The artifact deliberately does not invent rollout
collision or task-success labels for these static states.

On this adversarial distribution, computing both continuous and sampled
certificates had mean/P95/P99 latency `27.266/119.294/311.720 ms`, maximum
`2873.109 ms`, and `12,518` state-level 50 ms misses.  There were 175
validation-only global certificate recoveries.  These timings are not directly
comparable to the preceding generic one-certificate benchmark, but they show
that hard-state certificate recovery is emphatically not worst-case 20 Hz.

## Explicit counterexamples

1. **Pointwise margin is not recursive.** `rho=0.1` at the current state but
   becomes `-0.1` after an allowed action.
2. **Feasibility barrier is circular.** The original row `u>=0.75` and added
   feasibility row `u<=0.25` conflict.
3. **Endpoint safety misses collision.** Both endpoint barriers equal `3.75`,
   while the exact interior minimum is `-0.25` at `0.25 s`.
4. **Pointwise action energy is not mission energy.** Lower action energy at
   each of two steps still gives total `1.7`, above the one-step total `1.6`.
5. **ReLU has no global gradient-Lipschitz constant.** Across a kink, the
   required constant grows from `5` to `5000` as epsilon shrinks from `1e-1`
   to `1e-4`.

## Adversarial trajectory comparison

All three methods used identical 334 scenarios.

| Metric | Standard sampled-data HOCBF | Candidate A weighted energy, 0.1 | Two-stage lexicographic candidate |
| --- | ---: | ---: | ---: |
| Success | `333/334` (`99.70%`) | `333/334` (`99.70%`) | `69/334` (`20.66%`) |
| Collision steps | 0 | 0 | 0 |
| HOCBF residual violations | 194 | 89 | 11 |
| Infeasible steps | 305 | 119 | 15 |
| Uncertified fallbacks | 305 | 119 | 15 |
| Mean realized energy | 4.8466 | 4.6659 | 15.9561 |
| Mean path ratio | 1.0360 | 1.0383 | 3.4560 |
| Mean filter latency | 1.087 ms | 0.935 ms | 8.194 ms |
| Worst-rollout P95 | 34.890 ms | 20.573 ms | 89.104 ms |
| Worst-rollout P99 | 56.459 ms | 53.781 ms | 145.238 ms |
| Deadline misses | 180 | 70 | 16,476 |

The lexicographic candidate reduces observed infeasible steps relative to both
baselines, but it does not eliminate them and creates a much larger practical
failure: 265 of 334 tasks time out. Its mean energy is 235.7% above standard
HOCBF and 249.9% above Candidate A on the paired ratio statistic. It beats
standard HOCBF energy in only 14/334 scenarios and Candidate A in only 11/334.

The zero-collision count for the lexicographic method is not evidence of a good
safety controller because most trajectories fail to reach the goal and the
method frequently behaves like a conservative detour/freeze policy.

## Relation to the previously observed 34 fallbacks

The earlier fixed held-out 125-scenario experiment reported:

- standard sampled-data HOCBF: 73 infeasible steps;
- Candidate A at weight 0.1: 34 infeasible/fallback steps;
- 125/125 success and zero observed collision for Candidate A.

The new 334-scenario set is larger and intentionally harder, so its raw counts
must not be compared as if exposure were equal. Within the new matched set,
Candidate A has 119 fallbacks and the two-stage candidate has 15. The required
theory target was zero uncertified fallbacks; it was not met.

## Energy guarantee separation

- **Instantaneous:** the second-stage optimizer has pointwise quadratic-energy
  dominance over actions satisfying exactly the same constraints.
- **One step:** no deployed certified learned-value upper bound exists because
  the required verified smoothness constant is absent.
- **Trajectory:** empirical energy is substantially worse for the candidate;
  no total-energy theorem exists.

## Sampled-data guarantee separation

The exact quartic routine certifies a given constant-acceleration action for a
static sphere over one hold. The rollout collision metric also checks swept
segments rather than policy-step endpoints only. However, the online filter
does not enforce the nonconvex exact quartic certificate as its action set; it
uses an existing sampled-data HOCBF strengthening. Therefore no new exact
sampled-data synthesis theorem is claimed.

## Decision

The proposed theory candidate fails the computable-algorithm and experimental
falsification gates:

1. recursive feasibility remains unproved and empirically nonzero;
2. 10,225 hard states are already in the standard HOCBF domain but have an
   empty sampled-data action set;
3. navigation performance collapses;
4. energy is much worse, not better;
5. P95/P99 violate the 20 Hz requirement.

`FORMAL_500K = NOT_STARTED`.
