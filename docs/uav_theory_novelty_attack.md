# Independent Novelty Attack: UAV Feasibility and Energy Filter

## Verdict first

**The current theory candidate should be rejected as a theory contribution.**

- `CLAIM_LEVEL = ENGINEERING_CONTRIBUTION_ONLY`
- Current conference-theory readiness: **low**
- Development potential as a robotics systems/measurement study: **medium**
- Literature status: searched through 2026-08-19 using primary proceedings,
  arXiv records, and full equations where available
- Confidence: **high**

The rejection is not based on missing polish. The proposed recursive-feasibility
theorem is false without an invariant/backup domain, the valid feasibility and
energy statements are established constructions, and the implemented two-stage
filter fails navigation and 20 Hz falsification tests.

The additional physical hard-state search strengthens the negative conclusion:
the sampled action set is empty in 58,059/100,000 targeted states, including
10,225 states satisfying `h>=0` and `psi1>=0`. This establishes the importance
of the failure mode, but a diagnostic distribution is not a recursively
feasible controller and does not rescue novelty.

## Normalized idea

- **Problem:** sampled-data multi-obstacle HOCBF constraints can conflict with
  bounded UAV acceleration, creating uncertified fallback steps.
- **Proposed insight:** monitor a max-min control-authority margin and preserve
  sufficient authority while minimizing physical energy inside the safe set.
- **Mechanism:** pointwise margin plus a two-stage progress/energy optimizer.
- **Expected claim:** recursive feasibility, sampled-data collision safety, and
  energy optimality at 20 Hz.
- **Observed limitation:** the margin is only pointwise; the candidate does not
  construct the invariant domain needed for recursion.

## Closest prior art

| Closest work | What it already does | Fatal overlap or remaining delta |
| --- | --- | --- |
| [Xiao, Belta, Cassandras: feasibility CBF](https://arxiv.org/abs/2011.08248) | Adds a compatibility-preserving feasibility constraint to CBF/CLF optimization | Directly covers the feasibility-CBF strategy. Our generic `rho` barrier is weaker because compatibility of the added row is not proved. |
| [Breeden and Panagou: multiple CBFs under input constraints](https://arxiv.org/abs/2210.01354) | Constructs viability domains where simultaneous CBF rows remain feasible | Covers the missing invariant-domain ingredient. UAV notation and a cylindrical input set do not create a new theorem. |
| [Tan and Dimarogonas: compatibility checking](https://arxiv.org/abs/2209.02284) | Verifies/falsifies compatibility of multiple input-constrained CBFs | The max-min margin is another pointwise compatibility diagnostic, not recursive feasibility. |
| [Fikri: robust safety filtering with bounded input](https://arxiv.org/abs/2608.10872) | Uses exact scalar feasible-input interval and interval width as a feasibility margin | Strong equation-level overlap with the proposed authority margin. Multivariable support-function duality is standard. |
| [Alan and De Schutter: uniform feasible backup CBF](https://arxiv.org/abs/2511.13499) | Gives a priori feasibility conditions for smoothed backup safe sets | Covers the natural rescue route; no new UAV backup invariant set was derived here. |
| [Nakano et al.: recursive-feasible CBF reference governor](https://arxiv.org/abs/2604.04001) | Guarantees recursive feasibility by design using a dynamic safety margin | Shows that a margin becomes meaningful only with a separate invariant reference mechanism. |
| [Liu, Xiao, Belta: Sampling-Aware CBF](https://arxiv.org/abs/2511.11897) | Gives continuous-time ZOH safety using sampling-aware bounds for high-relative-degree constraints | Covers generic sampled-data safety. Our quartic minimum is an exact static-sphere verifier for a selected action, not a new synthesis theorem. |
| [Tan et al.: ZOCBF](https://arxiv.org/abs/2411.17079) | Enforces sampled-data safety through consecutive-sample barrier conditions | Further reduces novelty of a generic sampled-data claim. |
| [Manda et al.: performance-oriented CBF under limited actuation](https://proceedings.mlr.press/v270/manda25a.html) | Optimizes performance and invariant-set size under complex constraints and actuation limits | A secondary physical energy objective is not by itself a new safety theory. |

## Strongest novelty-collapse argument

The candidate is a composition of three known facts:

1. `max_u min_i slack_i` is a pointwise compatibility margin with a standard
   support-function/minimax dual;
2. recursive feasibility requires a controlled-invariant, backup, terminal, or
   governor set, all established routes;
3. minimizing a convex energy cost over a fixed hard-safe set is pointwise
   energy-optimal by definition.

The attempted bridge between items 1 and 2 is false, and item 3 does not produce
a trajectory-energy theorem. Therefore the combination has no surviving
theorem-level delta.

## Independent reviewer panel

### Field expert

- Score tendency: `2/5`
- Positive signal: the 34 original fallback steps expose a real robotics safety
  failure mode.
- Rejection-grade concern: established CBF compatibility and viability work
  already names and addresses that failure mode.
- Score-change condition: derive a non-equivalent controlled-invariant domain
  with strictly weaker assumptions or lower complexity than closest work.
- Confidence: `5/5`.

### Method/theory expert

- Score tendency: `1/5`
- Positive signal: the pointwise primal-dual theorem is correct and numerically
  validated.
- Rejection-grade concern: `rho>=0` is not recursive, and a barrier on `rho` can
  make a previously feasible action set empty. This invalidates the core claim.
- Score-change condition: provide a noncircular admissible control witness for
  every state in a declared set and prove its discrete-time invariance.
- Confidence: `5/5`.

### Experiment expert

- Score tendency: `3/5`
- Positive signal: 100,000-state checks and 1002 matched trajectories directly
  falsify the theorem candidate rather than hiding failure.
- Rejection-grade concern: the proposed method succeeds on only `69/334`
  trajectories, uses 235.7% more energy than standard HOCBF, and has
  worst-rollout P99 latency `145.24 ms`.
- Score-change condition: zero uncertified fallbacks, near-baseline navigation,
  and P99 below 50 ms on the same adversarial states.
- Confidence: `5/5`.

### AC / venue expert

- Score tendency: `2/5`
- Positive signal: the negative result can support a careful robotics systems
  paper about feasibility tails and sampled-data deployment.
- Rejection-grade concern: a main-track theory submission would offer known
  lemmas plus an invalid core theorem and a failed algorithm.
- Score-change condition: one valid non-equivalent core theorem with a practical
  algorithm that resolves the measured fallback failure.
- Confidence: `5/5`.

### Skeptical prior-art expert

- Score tendency: `1/5`
- Strongest objection: “This is support mismatch among CBF input halfspaces,
  followed by standard secondary convex optimization; recursive feasibility is
  delegated to the backup/viability literature.”
- Fatal risk: formula-level overlap remains even if the method is renamed.
- Score-change condition: none within the current `rho + lexicographic energy`
  formulation; the mathematical mechanism must change.
- Confidence: `5/5`.

## Scorecard

| Dimension | Weight | Score | Confidence | Deduction / evidence | Repair condition |
| --- | ---: | ---: | ---: | --- | --- |
| Problem importance | 12 | 4 | 5 | Uncertified fallback is important and observed | Retain the failure-driven problem |
| Novelty | 14 | 1 | 5 | Feasibility CBF, compatibility, viability, backup, and sampled-data safety are directly populated | New non-equivalent invariant mechanism |
| Conceptual innovation | 12 | 2 | 5 | Margin plus secondary cost is a standard composition | Mechanism that changes future feasibility, not just diagnoses it |
| Method soundness | 14 | 2 | 5 | Core recursive implication is false; algorithm still has 15 fallbacks | Complete noncircular proof and zero fallback in theorem domain |
| Elegance | 8 | 2 | 5 | Two-stage progress maximization creates pathological behavior | A single well-motivated invariant construction |
| Feasibility under resources | 8 | 2 | 5 | P99 145 ms and severe task timeouts | P99 below 50 ms and baseline-like completion |
| Experimental convincibility | 10 | 4 | 5 | Strong falsification evidence, but no positive theorem result | Positive matched validation of a new mechanism |
| Venue fit | 8 | 2 | 4 | Better as robotics systems evidence than theory | Reframe venue and claims or obtain a real theorem |
| Timeliness | 6 | 4 | 5 | 2025--2026 literature is active | No change needed |
| Risk-adjusted acceptance | 8 | 1 | 5 | Novelty collapse plus failed algorithm | Requires changing the central mechanism |

Weighted final score: `2.34/5` (`4.68/10`). This is not an acceptance
probability.

## Novelty attack result

`NOVELTY_SUPPORTED = FALSE`.

The strongest defensible artifact is not a theory method. It is a negative,
measurement-backed result:

> Pointwise CBF action compatibility and pointwise physical energy optimality do
> not close recursive feasibility or mission-energy guarantees; a naive
> lexicographic realization can reduce infeasibility counts while catastrophically
> degrading navigation and real-time performance.

This may be useful in a robotics/autonomous-systems paper if framed as a
deployment study with established safety filters. It is not a new general
control theorem.
