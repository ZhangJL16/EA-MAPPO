# Unified Belief-Conditioned UAV Agent: Theory Exploration

## Target

Determine whether one learning agent can use richer onboard information to:

1. navigate toward the current task point;
2. avoid static and dynamic obstacles;
3. estimate task and charger energy-to-go;
4. decide whether to continue the task or commit to the charger.

The intended contribution is not "more observation dimensions." The candidate
research question is whether a shared history-dependent belief representation is
decision-sufficient for these coupled predictions and decisions under partial
observability.

## Status

`COHERENT AFTER REFRAMING / EXTRA ASSUMPTION`

The engineering objective is feasible in simulation. A generic safety or novelty
claim is not supported merely by training one larger SAC policy. Dynamic-obstacle
state, energy conditions, and the consequences of rejected actions are only
partially observed; explicit assumptions or memory are necessary.

## Current-System Contrast

The current repository separates:

- a 7D goal-relative SAC that only decides how to fly;
- an independently trained energy estimator and TASK-to-CHARGER stopping rule;
- a separately evaluated sampled-data collision filter.

The proposed direction gives a single agent a common temporal representation,
but should retain semantically distinct output heads and constraints.

## Invariant Object

The top-level object is a resource-constrained partially observable control
problem over one battery cycle, not an unconstrained scalar reward.

Let the latent physical state be

\[
x_t=(p_t,v_t,e_t,g_t,g_c,\mathcal O_t,w_t),
\]

where \(p_t,v_t\) are UAV position and velocity, \(e_t\) is remaining energy,
\(g_t\) is the task point, \(g_c\) is the charger, \(\mathcal O_t\) is the
obstacle state, and \(w_t\) contains latent disturbances such as wind or payload.

The deployment observation is

\[
o_t=h(x_t)+\nu_t,
\]

and a history encoder constructs

\[
b_t=f_\psi(o_{0:t},a_{0:t-1}).
\]

The actor never receives simulator-only obstacle ground truth or future energy.
Those quantities may be used by an asymmetric critic during training, but must
not leak into the deployment actor.

## Proposed Deployment Information

The minimum richer observation should contain physically available quantities:

- normalized velocity and optionally realized acceleration;
- relative task direction and full-range task distance;
- relative charger direction and full-range charger distance;
- remaining-energy fraction and battery-health context;
- timestamped local range observations with validity masks;
- a short history or tracked relative obstacle motion;
- previous action and realized transition duration.

A single LiDAR frame is insufficient for dynamic-obstacle velocity and does not
represent an occluded obstacle. Absolute obstacle maps or future trajectories
must not be provided at deployment unless the sensing system genuinely supplies
them.

## Structured Single-Agent Decision

The agent has one shared encoder but four semantically distinct outputs:

\[
\begin{aligned}
a_t &\sim \pi_\theta(\cdot\mid b_t,g_t^{\mathrm{active}}),\\
Z_C &= F_\phi(b_t,a_t),\\
Z_E &= G_\omega(b_t,a_t,g),\\
d_t &\sim \mu_\eta(\cdot\mid b_t,Z_C,Z_E,e_t),
\end{aligned}
\]

where \(a_t\) is continuous flight control, \(Z_C\) is a short-horizon
collision-outcome distribution, \(Z_E\) is a goal-conditioned cumulative-energy
distribution, and \(d_t\in\{\mathrm{CONTINUE\_TASK},\mathrm{COMMIT\_CHARGER}\}\).

This is one agent because the encoder and training objective are shared. It is
not one undifferentiated output: collision, energy, and task utility cannot be
allowed to cancel each other through arbitrary reward weights.

## Objective and Constraints

The task objective can be written as

\[
\max_\pi \; \mathbb E_\pi\left[
N_{\mathrm{task}}-\lambda_T T-\lambda_U\sum_t\|a_t\|^2
\right]
\]

subject to separate risk statements

\[
\Pr_\pi(\text{collision in sortie})\le \Delta_C,
\qquad
\Pr_\pi(\text{energy exhaustion before charger})\le \alpha_E.
\]

If \(U_C(b_t,a_t)\) is a calibrated upper collision-risk estimate and
\(U_E(b_t,a_t,g_c)\) is a calibrated upper return-energy estimate, the deployed
continue decision should require both

\[
U_C(b_t,a_t)\le\delta_{C,t}
\]

and

\[
e_t^-\ge U_E(b_t,a_t,g_c)+m.
\]

These inequalities are decision semantics, not automatic neural-network
guarantees. Their validity is conditional on calibration, support, sensing, and
shift assumptions.

## One-Way Commitment

To avoid oscillation, retain the existing irreversible stopping semantics:

\[
d_t=\mathrm{COMMIT\_CHARGER}\Longrightarrow
d_{t+1}=\mathrm{COMMIT\_CHARGER}
\]

until charger arrival. The agent may learn the stopping time, but not whether the
commitment can be reversed. This gives the deterministic property that there is
at most one TASK-to-CHARGER transition per battery cycle.

## Reactive-Policy Impossibility Boundary

### Proposition

Suppose two latent environments induce the same current observation \(o_t\) but
have different hidden obstacle states. If action \(a\) is collision-free in the
first environment and colliding in the second, no memoryless policy
\(\pi(a\mid o_t)\) can select the correct action with probability one in both.

### Reason

The policy receives the same input distribution in both environments and must
therefore produce the same action distribution. If the sets of correct actions
are disjoint, at least one environment receives a nonzero probability of an
incorrect action.

### Consequence

Adding more current-frame channels only helps if they separate the latent cases.
Dynamic obstacles with occlusion or dropout require history, a belief model, an
independently conservative uncertainty set, or an explicit contingency action.
This is a standard POMDP observability boundary and is not itself a novelty
claim.

## Learnability Versus Safety

### What can be learned

With sufficient coverage and realizable function classes, simulation can train:

- goal-directed continuous control;
- collision probability or time-to-collision prediction;
- goal-conditioned energy return distributions;
- a task-versus-charger commitment policy;
- a recurrent latent state for obstacle motion and energy disturbances.

### What does not follow

- Universal approximation does not imply finite-data identification.
- Low training collision rate does not imply deployment safety.
- A scalar reward can trade collision against task reward and energy.
- A recurrent state is not a certified belief set.
- Simulator ground truth used by the actor invalidates a sensor-only deployment
  claim.
- Expected CMDP constraints do not imply zero per-trajectory violations.

## Prior-Art Attack

The components are individually mature:

- UVFAs and hindsight replay cover goal-conditioned value generalization.
- CMDPs and CPO cover reward optimization under expected cumulative constraints.
- Distributional RL covers return-distribution learning.
- SafeDreamer combines a learned world model with safe RL objectives.
- Safe exploration and recovery/shielding methods separate exploration from
  unsafe execution.
- UAV energy-risk work already learns trajectory energy and evaluates tail risk.
- UAV obstacle-avoidance policies already use LiDAR, depth, or local maps.

Therefore, "LiDAR + SOC + task point + charger point into SAC" is an obvious
combination and should not be the paper claim.

## Candidate Research Contribution

The strongest honest candidate is a robotics/autonomous-systems question:

> When dynamic obstacles and energy disturbances are only partially observable,
> does a shared recurrent belief improve the consistency between local avoidance,
> mission-energy prediction, and charger commitment compared with separately
> trained modules or a monolithic scalar-reward policy?

The key measurable phenomenon is cross-head consistency under intervention:

- Does an avoidance detour update return-energy predictions quickly enough?
- Does the commitment decision account for the energy cost of collision
  avoidance rather than Euclidean distance alone?
- Does shared history improve both obstacle-motion prediction and energy
  residual prediction under wind/payload shifts?
- Does joint training reduce task throughput or destabilize an already competent
  navigation skill?

This is currently a systems/empirical candidate, not a new general RL theorem.

## Minimum Method Blueprint

1. Preserve one shared recurrent encoder over deployable observations.
2. Use a low-level continuous control head conditioned on the active goal.
3. Use separate collision-risk and goal-conditioned energy-distribution heads.
4. Use a slow commitment head with an irreversible charger latch.
5. Train with an asymmetric critic that may access simulator state, while the
   deployment actor cannot.
6. Keep collision and energy as separate constraints during action/commitment
   selection.
7. Compare against the current modular system and a monolithic scalar-reward SAC.

## Minimum Decisive Experiment

Use matched task streams, obstacle trajectories, wind, payload, battery, and
evaluation seeds for:

1. current modular navigation + energy + collision filter;
2. monolithic feed-forward SAC with richer observation;
3. recurrent unified agent with scalar reward;
4. recurrent unified multi-head agent with separate constraints;
5. asymmetric-critic version of method 4.

Report:

- completed tasks per simulated hour and per battery cycle;
- task and charger arrival rates;
- collision and near-collision rates;
- energy exhaustion and autonomous-return success;
- energy MAE, tail underestimation, and calibration by detour length;
- commitment timing and remaining SOC at charger;
- performance under obstacle dropout, motion change, wind, payload, and sensor
  delay;
- intervention consistency: prediction change after avoidance detours;
- compute latency and catastrophic failure counts.

The first gate is not whether the unified network gets higher reward. It must
beat the modular and monolithic baselines on at least one coupled phenomenon,
such as energy prediction after avoidance detours, without worsening collision
or exhaustion.

## Boundaries and Non-Claims

- No claim of real-UAV safety without physical calibration and sensing tests.
- No claim that richer information monotonically improves learning.
- No claim that one network is preferable to modular control before matched
  ablations.
- No guarantee under arbitrary obstacle occlusion or distribution shift.
- No deep-RL global convergence theorem.
- No novelty claim for combining collision, energy, and task observations.

## Decision

`TECHNICALLY_FEASIBLE = TRUE`

`END_TO_END_SCALAR_REWARD_RECOMMENDED = FALSE`

`SHARED_BELIEF_MULTI_HEAD_AGENT_WORTH_PILOTING = TRUE`

`GENERAL_RL_THEORY_NOVELTY_ESTABLISHED = FALSE`

`BEST_CURRENT_POSITIONING = ROBOTICS_OR_AUTONOMOUS_SYSTEMS_EMPIRICAL_STUDY`
