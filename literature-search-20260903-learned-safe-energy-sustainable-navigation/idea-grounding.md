# Idea-Grounding Packet

## Scope And Evidence Boundary

- Topic / seed: a neural policy that learns safe, finite-battery,
  return-to-recharge navigation with repeated task cycles.
- Search date: 2026-09-03.
- Source-supported facts: minimum-cost reach-avoid, stochastic probabilistic
  reach-avoid plus expected cost, quantile cost constraints, budget
  augmentation, learned recovery, and reload-state MDPs all already exist.
- Searcher inference: the joint first-passage resource distribution is the
  correct implementation object, but a fixed-budget slice is equivalent to
  augmented-state reach-avoid.  The defensible residual gap must add
  finite-sample approximation control and repeated-recharge lifecycle semantics
  in continuous unknown dynamics.
- Unknowns: exact theorem-level uniqueness and whether it merits an oral-level
  paper after empirical testing.

## Evidence Cards

| Source | Supported observation | Reported limitation | Mechanism primitive | Protocol anchor | Transfer condition | Confidence |
| --- | --- | --- | --- | --- | --- | --- |
| RC-PPO | Direct minimum-cost reach-avoid can be learned through augmented dynamics | Theory depends on deterministic dynamics | cost-bound-conditioned reachability value | six MuJoCo reach-avoid tasks | Reimplement or adapt to current dynamics | direct |
| RAPCPO | Probabilistic reach-avoid and expected cost can be combined in stochastic RL | Practical normalized estimator is a non-certified surrogate and does not enforce feasibility every update | reach-avoid probability certificate | MuJoCo stochastic tasks | Compare joint tail calibration and repeated recharge | direct |
| Consumption MDP | Battery, atomic recharge, and almost-sure repeated objectives admit exact synthesis in finite models | Finite known MDP; qualitative objectives | minimal resource/counter strategy | finite-state scenarios | Use small abstractions as semantic oracle | direct |
| QCRL | Cumulative-cost quantiles characterize outage constraints | Not charger-first-passage specific | quantile policy gradient and distributional critic | constrained RL tasks | Replace generic cost with joint stopped resource | direct |
| SDAC | Multiple risk-averse constraints including collision and energy are trainable | Separate constraints do not express one mission event | distributional constraints and gradient integration | multi-constraint robotics | Strong algorithmic baseline | direct |
| Back-to-Base | A learned reach-avoid value can filter a controller back to a target by a deadline | External filter; cartpole demonstration | reach-avoid safety filter | modified SAC reset task | Compare filter dependence and raw policy | direct |
| Energy CBF | Energy sufficiency can be guaranteed over a path planner | Requires planner/reference path | energy barrier function | simulator and real robots | Use only as safety infrastructure/baseline | direct |
| Recovery RL | Task and recovery policies can be learned separately | No recharge-tail certificate | learned recovery zone and policy | sim plus physical robot | Baseline for neural switching | direct |
| Regret-Free LTL RL | Unknown finite MDPs can learn infinite-horizon temporal specifications with finite-time bounds | Finite state/action setting | reach-avoid reduction for LTL/Büchi | finite unknown MDP | Continuous function approximation remains open | direct |

## Cross-Source Relations

| Source pair / cluster | Relation | Open gap or conflict | Why it matters | Evidence needed next |
| --- | --- | --- | --- | --- |
| RC-PPO / RAPCPO | RAPCPO extends the deterministic problem to stochastic probability constraints | Neither stated abstract targets battery-tail feasibility across recharge cycles | Generic safe energy claim is occupied; only tail/lifecycle framing remains | Equation-level comparison |
| QCRL / consumption MDP | Quantile risk and reload semantics are complementary | No verified neural continuous-control synthesis joins them | Candidate mathematical seam | Small exact-MDP validation plus continuous RL |
| RCRL/RESPO / SDAC | Reachable feasible sets and risk distributions address different failure modes | Separate critics may not yield joint event control | Motivates one joint stopped random variable | Counterexample to marginal constraints |
| Back-to-Base / Recovery RL | Both separate nominal performance from recovery | Dependence on filters or separate policies may hide raw-policy weakness | User explicitly wants learned navigation | Raw and filtered evaluation |

## Idea Constraints

- Already covered central claims: battery in state; fixed return threshold;
  minimum expected energy reach-avoid; probabilistic reach-avoid plus expected
  cost; external energy/reach-avoid filter; finite reload-state synthesis.
- Transferable mechanism primitives: distributional critic, calibrated lower
  CDF, reachability certificate, budget-conditioned actor, learned latent mode,
  safe exploration filter, and regenerative risk accounting.
- Protocols suitable for direct comparison: current R3 frozen baseline and task
  seeds; RC-PPO reach/cost metrics; QCPO outage constraint; raw/filter split.
- Stale or overcrowded routes: reward shaping, adding SOC as one feature, adding
  a third ordinary critic, or claiming CBF itself as the contribution.
- Minimum viable research question: can a model-free continuous policy obtain a
  finite-sample lower bound on repeated-recharge satisfaction probability from
  an approximate distributional Bellman subsolution under deployment shift,
  while the raw actor retains task reward and learns the battery/reach/avoid
  boundary?
