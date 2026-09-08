# New-navigation return-energy study, 2026-09-08

## Decision and evidence audit

Continue research on the completed correction-SAC131072 adaptation, initialized
from the524288-step raw SAC. Freeze its FINAL checkpoint and HOCBF execution.
This is a new energy-data experiment, not another navigation-training variant.

Completed fixed500 results (one training seed; descriptive audit only):

| Interface | Arrivals | Zero-contact arrivals | Timeouts | Unified contacts |
|---|---:|---:|---:|---:|
| New network, raw |484/500|450/500|16|40072|
| Same network + HOCBF |492/500|492/500|8|1|

Task identity/prefix and unified safe-arrival semantics were checked against the
immutable five-distance-bin task file. HOCBF changes42 tasks from unsafe/nonarrival
to zero-contact arrival and loses none of the450 raw safe arrivals. Its one
contact occurs in a timeout, not among the492 arrivals. HOCBF intervenes on
74875/326837 policy steps (22.91%). Raw timeout trajectories account for36008
of40072 contacts: median0 contacts does not eliminate a severe tail.

No significance or training-seed-generalization claim; no matched lambda0
adaptation exists. These results do not isolate the correction loss's causal
effect. For483 tasks where both interfaces arrive, paired HOCBF-minus-raw mean
energy is−0.4914 in the simulator's energy units. Do not interpret the larger
all-task mean difference as a pure flight-efficiency gain: timeout exposure
differs. The data supports using HOCBF as the energy collection interface, not
a proof of zero collision, reliable continuation decisions, or sustainability.

## What500k means and what starts now

Working interpretation announced to the user:500000 ADDITIONAL actual policy
steps across workers, for energy data under frozen navigation. These are NOT
500000 navigation optimizer updates. The previous runner used a similar
frozen-navigation collection budget, but its old labels/encoder are not reused.

The launched stage collects complete return trajectories and supervised
cost-to-go/outcome labels. Energy-head fitting and closed-loop mission decisions
are subsequent work, not falsely described as already running. Collection stops
after500000 total transitions; no automatic model fitting or longer experiment.

## Minimal collection changes

- Load `hocbf_correction_sac_20260908_v1/sac/checkpoint_000131072/model.zip`;
  deterministic SAC inference, eval mode, all parameters requires_grad=False.
  Check policy tensor hashes before/after; record navigation_updates=0.
- Eight parallel CPU environments with batched CUDA inference, current native
  HOCBF/physics and all1024 LiDAR rays/hit flags. No new future-rollout estimator,
  no new safety layer, and no optimization performed inside the simulator.
- Fresh scene seeds1040000001 onward, interleaved by worker. Never train on the
  fixed500 navigation test or on old-policy energy labels.
- Target is the ACTUAL known charging station, not an arbitrary waypoint relabeled
  as a charger. Sample legal starting positions within the existing sampling
  bounds and beyond minimum goal distance. Initial velocity is independent
  uniform([-8,-8,-2],[8,8,2])m/s, within existing actuator speed limits. This is
  declared moving-start return coverage, not task-policy occupancy sampling.
- The simulator generates obstacles with the existing start/charger exclusions.
  No obstacle layouts/centers/radii/privileged clearances enter observations or
  saved predictor inputs. The2056-D navigation observation's relative goal now
  points at the known charger. Sensor rays and flags are unchanged.
- No added disturbance law in this first port. Across fresh scenes, the partial
  observation does not reveal the full future path. This dataset is NOT repeated
  physical-noise evidence at identical full states. Preserve the earlier
  stochastic-interface study as a separate experiment, not silently merged data.
- Keep the existing4000-policy-step finite return horizon and its remaining-time
  observation. This is a declared return rollout limit, not a newly added mission
  time deadline. Conditional predictions must carry this same remaining horizon.
- Collect virtual energy without battery exhaustion terminating the rollout;
  this avoids observing only cheap returns. It measures the current simulator's
  battery-independent flight cost, not voltage-dependent real-battery behavior.
  Later compare measured available battery with this learned resource law.

## Collision and target semantics

Contact does NOT terminate or reset a trajectory. Retain repair, zero velocity
at repair,−1.2/−0.42 reward, and one unified contact bit per policy step.

For a completed trajectory of lengthT, save every realized step energy e_t and
unified contact bit c_t. For each retained pre-action observation o_t, compute

    E_t = sum_(k=t)^(T-1) e_k
    S_t = 1{charger reached by the existing horizon} × product_(k=t)^(T-1)(1-c_k)
    Y_t = E_t if S_t=1; +infinity otherwise.

Past contacts beforet do not make the remaining return impossible. A contact
on step t DOES invalidate the zero-future-contact event from its pre-action
observation, even if the same step arrives. Finite energy consumed on a failed
rollout is saved for diagnosis, never passed off as finite energy-to-safe-return.

Save predictor observations every16 steps to avoid redundant disk copies;
the ACTOR still consumes the full2056 observation EVERY step, and every energy
and contact step is retained. This is dataset subsampling, not sensor reduction
or physical-step skipping. A full trajectory can supply correct suffix labels
without counterfactual simulation at every state.

## Mathematical consequence for later learning

For fixed executed interface I, let A_t indicate charger arrival during step t,
and let F_h(b,x)=P(Y<=b | x,I,h). With nonnegative e_t, nonterminal contact recovery
in the physical environment, and a zero-contact event in the VALUE target,

    F_h(b,x) = E[1{e_t<=b}(1-c_t)
                    (A_t + (1-A_t)F_(h-1)(b-e_t,X_(t+1))) | x,I].

The factor(1-c_t) kills only the event being predicted; it MUST NOT terminate
the physical episode. At a non-arrived horizon boundary F_0=0. The equation is
stated for sufficient statex. A memoryless LiDAR observation is not guaranteed
Markov, so naively treating this equation as an exact observation-only Bellman
equation can introduce approximation bias. Monte Carlo suffix labels avoid
requiring an observation-only bootstrap assumption for the initial comparison.

The identity F_h(b,x)=p_h(x)F_finite(b|S=1,x,h) implies that no battery budget
can meet target probability1-delta if p_h(x)<1-delta. In particular, a high
conditional finite-energy quantile alone does not enforce safe return. This
is the reason to distinguish failure probability from expensive successful
return, not a claim that the elementary factorization is a novel theorem.

## Planned comparison on this common new-policy dataset

1. Conditional finite mean-energy MC regression, with clearly defined handling
   of failure outside that regression; not arbitrary finite failure caps.
2. Conditional finite quantile regression (pinball loss), an uncertainty-aware
   baseline using the same realizations and features. Do NOT call this Quantile-TD.
3. Factorized safe-return probability plus conditional finite-energy CDF,
   evaluated against budget events including failures.

Reuse the frozen new navigation's structured LiDAR encoder where useful; do
not add another navigation network or modify its weights. A later Quantile-TD
arm requires the correct transition/survival target and an explicit treatment
of partial observability and failure mass. Copying the old success-only TD loss
would not be the same estimand. Fixed battery thresholds belong in the subsequent
closed-loop mission comparison, not this return-prediction-only dataset.

Before fitting, inspect label support and domains as POST-experiment analysis,
not a prelaunch gate. If there is little failure support, report it; do not invent
negative labels to favor the factorized model. Evaluate dangerous false-safe
decisions, budget Brier/calibration, and retained coverage, not only energy MAE.
The full mission experiment still needs task-to-return transitions, battery
budgets and utility outcomes; success on this stage cannot stand in for it.

## Split, censoring and recovery

Scene seeds hash deterministically into train/calibration/test proportions
70/20/10. All states from the same seeded world remain together. This is a
scene-held-out development comparison, not multiple independent policy seeds.

At the500000-step cutoff, unfinished trajectories are RIGHT CENSORED. Preserve
their full worker and partial data state. Do not label them as timeout, safe,
zero-cost or finite-cost failure. The dataset index accounts exactly for
completed_steps+censored_steps=500000. Fitting must read only the COMMITTED index,
never glob all episode files (a hard crash may leave uncommitted replayable rows).
Excluding censored tails may induce length selection; report their count and
energy. Any later tail completion must count its extra physical steps explicitly.

Checkpoints every8192 transitions, on pause, and at budget end include workers,
environment RNG/cursors, observations and partial trajectories. Episode files
are atomic and keyed by scene. A hard-crash replay verifies prior row identity
rather than silently overwriting it. No replay buffer or navigator optimizer is
needed; the actor is frozen and deterministic.

```bash
.venv/bin/python scripts/collect_new_navigation_return_energy.py \
  --source artifacts/hocbf_correction_sac_20260908_v1 \
  --output-dir artifacts/new_navigation_return_energy_500k_20260908_v1
```

Resume with the same command plus `--resume`. A root PAUSE marker or SIGTERM
requests full-state pause after the current vector step. Only remove a validated
PAUSE marker when authorized. No hard wall-clock kill or automatic promotion.

## Necessary checks only

Four focused tests cover suffix-contact labels/censor rejection; charger-target
observation and exact worker restoration; idempotent episode commit; and ONE
tiny actual checkpoint runner8 steps followed by continuation to16. These are
execution tests, not performance gates. Then check the first8192-step committed
formal checkpoint and hand off without monitoring to completion.

Startup handoff: detached PID96663; artifact
`artifacts/new_navigation_return_energy_500k_20260908_v1`. First8192 checkpoint
verified:16 complete episodes,397 retained observations,1959 in-progress
transitions,8 full worker states,79840691-byte checkpoint. Accounting and finite
data checked, source hashes unchanged, process alive, no ERROR. Recorded
navigation_updates=0. First checkpoint collection elapsed6.97s is an early
throughput observation, NOT a full-run runtime guarantee. Stop monitoring.

The first launcher resolved the venv interpreter symlink to system Python and
failed importing numpy before producing any run output. Relaunch uses the
absolute venv path without following the symlink. No experiment data was lost;
the external console log retains the failed import for provenance.
