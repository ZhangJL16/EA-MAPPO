# Measurement Bundling: Minimal Embodied Validation Plan

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-09-16
- Verification Status: UNVERIFIED — design and oracle calculations, no learning run
- Version Label: bundling_validation_proposal_v1
- Authority: proposal only; not an approved run, sealed preregistration or command

## Experiment Overview

**Question.** Can additional capacity lower the statistical opportunity cost of
learning the SAME optimal behavior by allowing measurements to share a safe route?

**Target claim.** Capacity can improve learning without improving optimal execution.
The mechanism is measurement bundling, not information being costly in general,
new unknown parameters, or a new deep-RL algorithm.

**Scope.** One idealized 2D inspection robot with known fixed motion routes,
integer energy expenditure, a single instantaneous reload dock and stationary
hidden inspection rewards. No UAV, SAC/PPO, LLM, unknown safety or realistic
charging physics. Do not label graph reskins as cross-domain replication.

The existing lower and finite-hypothesis upper are the theory inputs. Do not
use the old continuous-class C3=1.816 / C4=1.326 as predictions for this new
finite-hypothesis learner: changing Theta changes the confusing alternatives.

## Setup and Execution Boundary

- Working directory: /home/zjl/mappo.
- Implementation: not created or changed in this planning round.
- Entry command: NONE APPROVED; specify after the implementation scope is approved.
- Expected implementation: isolated CPU/headless route plant, learner interfaces,
  prediction calculator, leakage validator and paired analyzer; no learned navigator.
- Original collision semantics, navigator, plant and hash-pinned experiments stay
  untouched. No existing DEV/CONFIRM data enters this benchmark.
- No experiment has been launched. Runtime, dependencies and wall-clock budget
  remain to be established during an explicitly approved implementation/startup.

## Physical Route Model

Position must actually evolve during the mission, not be an animation added to
an unrelated bandit transcript. A known fixed-speed segment advances position,
mission time and battery together. Inspection is emitted only at its designated
site arrival. Transit and undocking rewards are zero. Fixed route descriptors
provide the complete safe action catalogue. Inspection outcomes, not motion,
are stochastic. Audit route time and energy against executed plant steps.

For the exact calibration, use an equilateral layout: dock q=(0,0), site A=(1,0),
site B=(1/2,sqrt(3)/2); every inter-site travel segment takes one time/energy unit.
Undocking takes one additional unit. Sensing at the arrival instant has no
additional duration in this ideal model. Each site can be inspected at most once
per sortie under a known batch protocol; repeated sorties produce fresh IID draws.
At the dock, a known local inspection/service action takes one unit and yields
Bernoulli reward with known mean r=0.05. Debit before transition; on dock arrival
reset to capacity B, exactly as in the theory.

Thus A-only and B-only take 3 units; AB and BA take 4. A depart/return empty
sortie is allowed with zero reward and positive cost. The physical state includes
position, undocking/inspection-mask flags and battery, so the last inspection
position is never silently identified with a different pose. Extra position
states are acceptable; there is no requirement to retain literally five states.

This is an idealized embodied-route model, not evidence about actuator noise,
perception, charging duration, or real robot deployment.

## Exact Finite-Hypothesis Calibration

Freeze the SAME unknown-mean class in every cell:

    Theta = {theta_A=(.30,.05), theta_B=(.30,.95), theta_q=(.05,.05)}.

All measurement rows of A share its mean; likewise B. Known dock draws have
the same .05 law under every hypothesis. The primary true hypothesis is theta_A;
truth is an evaluator-only index, never a learner input. Each learner receives
the entire fixed Theta, the public routes, safety mask and observed rewards.
Use fixed truth-independent likelihood tie breaking. The learner specification
must also apply unchanged to theta_B and theta_q, not hard-code the primary truth.

### Oracle execution check before sampling

| Hypothesis | Dock rate | A rate | B rate | AB rate when allowed | Unique optimum |
|---|---:|---:|---:|---:|---|
| theta_A | .05 | .10 | .016667 | .0875 | A-only |
| theta_B | .05 | .10 | .316667 | .3125 | B-only |
| theta_q | .05 | .016667 | .016667 | .025 | dock |

Every hypothesis has the SAME optimum and gain with or without AB. At primary
truth gain=.10 and normalized reachable canonical bias span=.20 at both
capacities. The largest positive bias is .10, and the negative minimum is -.10;
additional battery changes some intermediate values, not their span. Both unknown
means are already accessible with capacity 3. Return takes at most one unit from
the nontransit calibration decision states.

This improves causal interpretation over the previous continuous example:
the set of different-optimum alternatives does not change either. Capacity is
not allowed to secretly improve execution in a different possible hypothesis.

### Oracle information allocation — existing theorem instantiated, not a new theorem

At theta_A the only costly confusing alternative is theta_B: A observations
are identical and the optimum changes to B. theta_q differs on the optimal
A path and can be eliminated with zero-cost optimal execution.
Let a=kl(.05,.95)=2.649995081249796. Each B measurement supplies a KL.

    c(A)=0;
    c(B)=3*.10-.05=.25;
    c(AB)=4*.10-.30-.05=.05.

The information constraint is a*x_B>=1. Hence

    C_split = .25/a = .09433979774864129;
    C_bundle = .05/a = .01886795954972826.

An optimal A traversal can be folded into a B-querying sortie. That A measurement
adds no information against theta_B; its productive reward compensates part of
the time cost. This is opportunity-cost bundling, not duplication of information.
The 80% reduction is an ASYMPTOTIC COEFFICIENT prediction, not a forecast of an
80% finite-budget task-count improvement. Oracle scalar calculations were checked
read-only in this planning round; no scientific simulation outcome was inspected.

## Factorial Mechanism Intervention

Use a 2x2 paired design, the same route plant, rewards, Theta and seed streams:

| Cell | Capacity | Protocol allows two-site sortie? | Feasible informative paths | Predicted C at theta_A |
|---|---:|---|---|---:|
| low / on | 3 | yes, subject to energy safety | A,B | .0943398 |
| high / on | 4 | yes, subject to energy safety | A,B,AB,BA | .0188680 |
| low / off | 3 | no | A,B | .0943398 |
| high / off | 4 | no | A,B | .0943398 |

The bundling-off protocol requires docking/reloading after one inspection.
It is a KNOWN operational action restriction, not a change to geometry, energy
cost, inspection laws or charging timing. Report it as such. At low capacity
the on/off feasible catalogues coincide; with paired reward streams an identical
learner should give identical observable continuations. At high capacity the
extra battery remains available, but cannot create a joint measurement sortie.

Only the high/on cell should have the smaller leading coefficient. This estimates
the total effect of permitting bundled safe trajectories, not an intervention
that independently fixes the information allocation while changing its cost.

## Beyond Calibration: Prospective Routing Instance Library

Proposal: 12 distinct eligible routing descriptors, each with 20 paired reward
seeds. These numbers are a computational design recommendation, not a power
calculation or an approved sample size. Final instance/seeding choices require
author adjudication before outcomes.

Generate integer-length routes on fixed 2D waypoint graphs, with two inspection
channels sharing the same Theta above. Geometry, route lengths, energy costs and
known dock mean are fixed PER INSTANCE before capacity cells are built. For
example, choose the dock mean as .15/ell_A, so it is half the primary A rate;
this known constant never changes with capacity. Require:

- all singleton measurements safely accessible at both capacities;
- a high capacity supporting a joint route absent at the low capacity;
- unique optima under every hypothesis and an unchanged optimal path type/gain
  under every capacity/bundling cell;
- known fixed movement, positive-consumption cycles and proper dock returns;
- distinct acquisition catalogue signatures, not merely rotated drawings.

Choose the first eligible distinct descriptors in a fixed generator order.
Do NOT rank/select by learner outcomes or require predicted improvements to
exceed a chosen magnitude. Retain eligible zero-benefit cases if the new joint
route is acquisition-dominated: these are additional theoretical null predictions.
Freeze the selection rule, generator, descriptors, private truth assignment,
all C values and predicted allocations BEFORE the first learning outcome.

If 12 eligible descriptors cannot be obtained, report that design limitation;
do not quietly change the admissibility criteria or claim 12 independent worlds
from coordinate reskins. This remains one-domain routing validation, not general
embodied-AI coverage.

## Methods and Fairness

| Method | Information available | Role |
|---|---|---|
| Oracle allocation schedule | true theta | privileged coefficient/scheduling reference |
| Resource-path learner | Theta, routes, observed rewards | existing finite-class attainable learner |
| Cost-aware structured baseline | SAME Theta, routes, rewards, durations | strongest adjacent paradigm |
| Cost-blind structured learner | same public inputs, rate-gap allocation objective | time-accounting ablation |
| Independent duration-aware UCB | same feasible paths and primitive rewards | sharing/structure ablation, secondary |

The oracle ALLOCATION is not an optimal known-model execution policy: it
deliberately performs the prescribed informative paths although truth is known.
Do not call it a universal finite-time upper bound. Separately plot the known-model
EXECUTION oracle, which repeatedly executes p_theta and has zero complete-cycle
pseudo-regret. No significance contest against the allocation oracle is required.

The cost-aware baseline must genuinely use shared feedback and calendar cost.
Specify and audit its reference procedure before implementation; do not claim
an unmodified stock OSSB theorem supports a modified implementation. An independent
code path can check the same weighted experiment allocation. Equal leading
coefficients with our learner are expected, not an experimental failure.

The cost-blind allocation replaces c(p) by c(p)/ell_p but keeps the information
constraints. Precompute its selected allocation and ACTUAL calendar cost too.
It may coincide with the optimum: in the exact calibration both objectives can
prefer AB. Report such coincidences as controls; do not manufacture a mandatory
baseline defeat or represent an allocation tie as evidence of failure.

Independent UCB must select reward RATE, not raw total reward; otherwise it
optimizes a different objective. Treat its lack of sharing as an explicit ablation,
not evidence that all existing bandit methods misunderstand physical resources.

## Inputs, Isolation and Leakage

Proposed future inputs (not created yet): sealed route/seed manifest, public
hypothesis file, private evaluator truth file and precomputed prediction registry.
The learner cannot read true means, truth-index fields, predictions C(theta),
oracle allocations keyed by truth, future draws or evaluation summaries. It CAN
compute model-indexed LPs for ALL public hypotheses, as required by its theorem.

Use independent A/B/dock draw streams, addressed by query number, and reuse them
across methods/cells within a paired seed. This is legal CRN pairing, not future
outcome access. Keep algorithm RNG separate. Shared parameters remain pooled in
the likelihood. No old experiment data, pretrained policy or external model.

## Expected Outputs

All paths below are anticipated artifact types, NOT existing output claims:

- provenance/identity manifest and prediction registry;
- primitive time, pose, battery, action and reward trace;
- completed excursion counts, shared observation counts, realized task counts;
- expected-reward pseudo-regret curve and realized-reward curve;
- prediction-versus-observation plots and paired capacity/bundling contrasts;
- failure/leakage/runtime receipt, with no silent omission of incomplete runs.

## Analysis Plan

Proposal horizons: T=4096,16384,65536 primitive time units, nested checkpoints
of ONE continuing trajectory per cell/method/seed. Do not treat checkpoints as
independent samples or change the final horizon after inspecting curves.

Primary estimator is evaluator-computed pseudo-regret

    P(T)=sum_{t<T}[rho_theta^* - r_theta(i_t)].

Its expectation equals the defined expected regret by conditional expectation.
It uses PRIVATE true means for evaluation only. It is not realized task loss.
Realized task counts and regret remain secondary: random rewards on the optimal
route alone create order sqrt(T) noise, which can swamp a log(T) learning signal.
Do not secretly replace their empirical variance by the smaller pseudo-regret
variance or claim that a coefficient reduction is the same as tasks/hour gain.

For each route descriptor report P(T)/log T beside the SEALED C prediction,
allocation counts, P(T)-C log T, and the privileged allocation schedule reference.
No free post-outcome rescaling, intercept-adjustment to C, or fitted coefficient
presented as an a priori prediction.

Primary mechanism contrast is

    I(T) = [P_low,on(T)-P_high,on(T)]
           -[P_low,off(T)-P_high,off(T)].

Its predicted leading coefficient is the corresponding sealed difference of
four C values. In calibration it is .07547183819891303. Retain all eligible
null predictions and report residuals; do not analyze only improving worlds.
Use whole-route-descriptor paired summaries; reward seeds repeat stochastic
realizations inside descriptors. Jointly resample trajectories/checkpoints when
reporting paired uncertainty. Per-descriptor C varies and must be compared to
its own prediction, not a pooled universal constant.

### Finite-budget interpretation and decision rules

- Invalid experiment: clock/energy mismatch, changed execution oracle, leakage,
  unmatched catalogues in designated identical cells, or unapproved exclusions.
  Stop interpretation; fix integrity only under a documented amendment.
- Supported practical mechanism: in the declared budget range, capacity benefits
  concentrate in bundling-enabled cells and are attributable to bundled information
  acquisition, with prediction residuals and uncertainty transparently reported.
- No practical support: valid cost-aware learning shows no reproducible benefit
  in the declared range. Report limited finite-budget relevance; do not extend
  horizons or alter worlds to obtain the desired result.
- A finite-horizon discrepancy does NOT refute an asymptotic theorem. Forced
  sampling has o(log T) but potentially material finite-budget cost, and constants
  need not be small. Do not promise an EARLY slope equal to C or set an arbitrary
  closeness threshold without a justified finite-time analysis/power design.

Numeric scientific pass/fail thresholds and multiplicity handling are not sealed
by this proposal. They need author adjudication before formal outcomes; there
is no automatic Spotlight promotion from a positive confidence interval.

## Monitoring and Next Authorized Boundary

No monitoring is active. If implementation is approved separately, permit only
focused integrity tests, one excluded small smoke and one startup-health check,
then hand back. Use a resumable path and a user-approved timeout; no automatic
promotion to the formal routing library or retry after failures. Existing holds
on remaining oracle-stuckness DEV worlds and CONFIRM remain unchanged.

## Claim Boundaries

This plan validates a physical-route opportunity-cost mechanism predicted before
learning, not a new formulation solely because a robot has coordinates. It does
not certify generic structured bandits are unable to express the model, and does
not promise algorithm novelty. Equal performance of the strongest cost-aware
baseline is consistent with the mechanism. Real charging/navigation systems and
broader AI relevance remain outside the approved theoretical assumptions.
