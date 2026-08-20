# Control-Grounded Structured Memory Architecture

## Objective

The architecture answers three separate questions:

1. What must be retained across time?
2. Which module is allowed to consume each retained quantity?
3. Which retained quantities may alter a certified collision constraint?

It is not an end-to-end recurrent policy and does not replace the frozen navigation policy or the hard safety filter.

## Typed memory state

At control step `t`, define

\[
M_t=(M_t^{\mathrm{ego}},M_t^{\mathrm{obj}},M_t^{\mathrm{safe}},
M_t^{\mathrm{route}},M_t^{\mathrm{energy}}),
\qquad
M_t=(M_t^{\mathrm{cert}},M_t^{\mathrm{learn}}).
\]

### Ego memory

Certified/explicit fields:

- current velocity and actuator limits;
- realized acceleration;
- nominal SAC action and executed action;
- current sample/hold duration.

Learned/statistical fields:

- short histories of intervention, braking, vertical maneuver, progress, and realized propulsion energy.

### Object memory

One slot per active track:

\[
M_{t,i}^{\mathrm{obj}}=(\hat p_{t,i},\hat v_{t,i},\hat a_{t,i},
r^p_{t,i},r^v_{t,i},r^a_{t,i},\mathrm{age}_{t,i},\mathrm{miss}_{t,i},
\ell_{t,i}).
\]

The centers, track metadata, and radii belong to the certified channel only when their declared measurement, association, process, and dropout premises hold. The maneuver latent `ell` is always learned and uncertified.

### Safety memory

- certified obstacle intervals and track provenance;
- minimum clearance/TTC summaries;
- HOCBF values/slack and control-authority margin;
- previous certified action and feasibility mode.

Safety memory never accepts a learned reduction of `r^p,r^v,r^a`.

### Route memory

- previous avoidance side and safe corridor;
- recent safe trajectory/proposal;
- CLF progress, detour, turn direction, and freeze history.

Route memory may alter the nominal proposal but not the hard feasible set.

### Energy memory

- realized power and acceleration energy;
- intervention and detour energy;
- safe-path length context and recent energy-to-go residual.

Energy memory may rank feasible actions/trajectories. It cannot relax collision constraints.

## Update semantics

The update is typed:

\[
M_{t+1}^k=F_k(M_t^{\mathrm{pa}(k)},o_{t+1},u_t^{\mathrm{nom}},u_t^{\mathrm{exec}}),
\]

with fixed parent sets

\[
\begin{aligned}
\mathrm{pa}(\mathrm{obj})&=\{\mathrm{perception},\mathrm{ego}\},\\
\mathrm{pa}(\mathrm{safe})&=\{\mathrm{ego},\mathrm{obj}_{\mathrm{cert}}\},\\
\mathrm{pa}(\mathrm{route})&=\{\mathrm{ego},\mathrm{obj}_{\mathrm{learn}},\mathrm{safe\ summary}\},\\
\mathrm{pa}(\mathrm{energy})&=\{\mathrm{ego},\mathrm{route},\mathrm{safe\ summary}\}.
\end{aligned}
\]

The executed action and realized transition feed back into all relevant memories. Track identity changes reset the corresponding object slot and invalidate its certificate until a new trusted base interval is supplied.

## Routing contract

| Producer | Safety constraint | Nominal route | Energy ranking |
|---|---:|---:|---:|
| Ego explicit | yes | yes | yes |
| Object certified | yes | optional summary | optional summary |
| Object learned latent | **no** | yes | yes |
| Safety learned probe | **no** | yes | yes |
| Route memory | **no** | yes | yes |
| Energy memory | **no** | optional feasible-action ranking | yes |

The hard route is

\[
u_t^{\mathrm{nom}}=\pi(x_t,M_t^{\mathrm{learn}}),\qquad
u_t^{\mathrm{exec}}=\mathcal F(u_t^{\mathrm{nom}};M_t^{\mathrm{cert}}).
\]

Learned memory may propose, rank, or tighten. It may not reduce a certified uncertainty radius or remove a certified obstacle constraint.

## Grounding objectives

Grounding heads are module-specific:

- motion: relative position/velocity/acceleration and short-horizon physical motion;
- safety: `h`, `psi1`, TTC, robust slack, control-authority margin, future clearance, and feasible-action existence;
- intervention: executed-minus-nominal action and future intervention frequency;
- route: progress, CLF decrease, avoidance side, detour, and path length;
- energy: realized safe-rollout energy and safety-induced overhead.

The exploratory training protocol does not rely on one unnormalized weighted sum. Each loss is normalized by a training-split scale and alternated in fixed batches. The main candidate is selected on a preregistered validation vector, not by retuning scalar weights on test performance.

## Architecture-family disposition

| Family | Disposition | Reason |
|---|---|---|
| A single GRU | baseline | no typed semantics |
| B object GRU slots | baseline | established object-memory design |
| C physical plus residual GRU | baseline | established hybrid observer pattern |
| D RIM | baseline | established sparse modular recurrence |
| E slots plus module memories | keep for architecture study | semantically aligned but not theoretically new |
| F structured SSM | comparator | efficient temporal model, no routing guarantee |
| G learned sparse graph | reject for main candidate | extra optimization and verification burden |
| H fixed physical graph | keep | interpretable and testable |
| I dual certified/learned channels | mandatory | required for safety claim boundaries |
| J grounded modular candidate | provisional | enters short diagnostic only |

## Provisional candidate

The provisional candidate is **Fixed-Routing Object-Centric Grounded Memory (FOGM)**:

- explicit ego memory;
- one shared recurrent update per object slot;
- certified interval fields stored separately from learned maneuver latents;
- fixed Safety/Route/Energy routing;
- module-specific grounding heads;
- hard safety filter built exclusively from certified fields;
- learned messages fall back to zero without disabling the certified baseline.

FOGM is an algorithmic architecture hypothesis. Its individual components are existing; novelty depends on a nontrivial theorem-level connection or a robust closed-loop effect that cannot be explained by a generic observer or GRU.
